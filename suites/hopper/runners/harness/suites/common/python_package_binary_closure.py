#!/usr/bin/env python3
"""Build and verify an exact kernel index for installed Python distributions.

The installed distribution's ``RECORD`` file is the authority for ownership.
This deliberately does not walk arbitrary site-packages directories: only
regular, non-symlink CUDA binaries owned by a distribution can enter the
index.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Iterable


INDEX_SCHEMA = "hopper-python-package-binary-index-v1"
CUDA_BINARY_SUFFIXES = (".cubin", ".fatbin", ".so")


def canonical_distribution_name(value: str) -> str:
    """Return the normalized package name used for exact allow-list matching."""
    normalized = re.sub(r"[-_.]+", "-", str(value).strip()).lower()
    if not normalized:
        raise RuntimeError("Python distribution name is empty")
    return normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_cuda_binary(relative: str) -> bool:
    name = Path(relative).name
    return name.endswith(CUDA_BINARY_SUFFIXES) or ".so." in name


def _assert_no_symlink(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"Python distribution binary closure contains symlink: {cursor}")


def _decode_record_sha(value: str) -> str:
    algorithm, separator, encoded = value.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        raise RuntimeError(f"Python distribution RECORD needs sha256, got {value!r}")
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except ValueError as exc:
        raise RuntimeError(f"invalid Python distribution RECORD digest {value!r}") from exc
    if len(raw) != 32:
        raise RuntimeError(f"invalid Python distribution RECORD SHA-256 {value!r}")
    return raw.hex()


def _metadata(record: Path) -> tuple[str, str]:
    metadata = record.parent / "METADATA"
    if not metadata.is_file() or metadata.is_symlink():
        raise RuntimeError(f"distribution metadata is missing or a symlink: {metadata}")
    parsed = Parser().parsestr(metadata.read_text(encoding="utf-8"))
    name = str(parsed.get("Name") or "").strip()
    version = str(parsed.get("Version") or "").strip()
    if not name or not version:
        raise RuntimeError(f"distribution metadata lacks Name/Version: {metadata}")
    return name, version


def _collect_one_distribution(
    *, record: Path, site_root: Path, build_root: Path
) -> dict | None:
    if record.is_symlink() or not record.is_file():
        raise RuntimeError(f"distribution RECORD is not a regular file: {record}")
    _assert_no_symlink(record, site_root)
    name, version = _metadata(record)
    binaries: list[dict] = []
    with record.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        if len(row) != 3 or not _is_cuda_binary(row[0]):
            continue
        relative = row[0]
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(
                f"distribution {name} RECORD binary path escapes site root: {relative}"
            )
        path = site_root / candidate
        _assert_no_symlink(path, site_root)
        try:
            path.resolve(strict=True).relative_to(site_root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"distribution {name} RECORD binary path escapes or is missing: {relative}"
            ) from exc
        if not path.is_file():
            raise RuntimeError(f"distribution binary is not a regular file: {path}")
        try:
            expected_size = int(row[2])
        except ValueError as exc:
            raise RuntimeError(
                f"distribution {name} RECORD binary lacks an exact size: {relative}"
            ) from exc
        if expected_size <= 0 or path.stat().st_size != expected_size:
            raise RuntimeError(f"distribution {name} binary size mismatch: {relative}")
        binaries.append(
            {
                "path": relative,
                "size": expected_size,
                "record_sha256": _decode_record_sha(row[1]),
            }
        )
    if not binaries:
        return None
    return {
        "name": name,
        "version": version,
        "root_kind": "frozen_build_root",
        "site_root": os.path.relpath(site_root, build_root),
        "record": os.path.relpath(record, build_root),
        "record_sha256": sha256(record),
        "binaries": binaries,
    }


def collect_distribution_binaries(
    build_root: Path, distribution_names: Iterable[str] | None = None
) -> list[dict]:
    """Collect exact CUDA binaries owned by installed wheel distributions."""
    build_root = build_root.resolve(strict=True)
    selected = (
        {canonical_distribution_name(name) for name in distribution_names}
        if distribution_names is not None
        else None
    )
    if selected is not None and not selected:
        raise RuntimeError("Python distribution binary closure allow-list is empty")
    distributions: list[dict] = []
    found: set[str] = set()
    for record in sorted(build_root.rglob("*.dist-info/RECORD")):
        site_root = record.parent.parent
        name, _ = _metadata(record)
        canonical_name = canonical_distribution_name(name)
        if selected is not None and canonical_name not in selected:
            continue
        row = _collect_one_distribution(
            record=record, site_root=site_root, build_root=build_root,
        )
        if row is not None:
            distributions.append(row)
            found.add(canonical_name)
    if selected is not None and found != selected:
        missing = ", ".join(sorted(selected - found))
        raise RuntimeError(
            "selected Python distributions lack owned CUDA binaries under "
            f"{build_root}: {missing}"
        )
    if not distributions:
        raise RuntimeError(f"no Python distribution-owned CUDA binaries under {build_root}")
    return distributions


def _contains_exact_symbol(payload: bytes, kernel: bytes) -> bool:
    start = 0
    while True:
        position = payload.find(kernel, start)
        if position < 0:
            return False
        before = payload[position - 1] if position else 0
        end = position + len(kernel)
        after = payload[end] if end < len(payload) else 0
        # CUDA ELF string tables delimit mangled symbols with NUL.  Permit
        # other non-identifier delimiters for fatbins while rejecting a name
        # that is merely a prefix of another symbol.
        if not (chr(before).isalnum() or before == 95) and not (
            chr(after).isalnum() or after == 95
        ):
            return True
        start = position + 1


def build_binary_index(
    *,
    build_root: Path,
    kernel_names: Iterable[str],
    output: Path,
    symbol_verifier: Callable[[Path, str], bool],
    workers: int = 8,
    distribution_names: Iterable[str] | None = None,
) -> dict:
    """Hash a package closure once and map every requested kernel exactly."""
    candidate_index = classify_binary_candidates(
        build_root=build_root,
        kernel_names=kernel_names,
        symbol_verifier=symbol_verifier,
        workers=workers,
        distribution_names=distribution_names,
    )
    build_root = build_root.resolve(strict=True)
    kernels = sorted(candidate_index["kernels"])
    classification = candidate_index["kernels"]
    compact_distributions = candidate_index["distributions"]
    resolved = {
        kernel: row["matches"][0]
        for kernel, row in classification.items()
        if row["exact_match_count"] == 1
    }
    classification_payload = {
        "schema_version": "hopper-python-package-binary-classification-v1",
        "kernel_count": len(kernels),
        "zero_match_count": sum(
            row["exact_match_count"] == 0 for row in classification.values()
        ),
        "unique_match_count": sum(
            row["exact_match_count"] == 1 for row in classification.values()
        ),
        "multiple_match_count": sum(
            row["exact_match_count"] > 1 for row in classification.values()
        ),
        "kernels": classification,
    }
    classification_payload["content_sha256"] = _canonical_digest(
        classification_payload
    )
    classification_path = output.with_name(
        output.stem + ".classification" + output.suffix
    )
    classification_path.parent.mkdir(parents=True, exist_ok=True)
    classification_temporary = classification_path.with_name(
        classification_path.name + f".tmp.{os.getpid()}"
    )
    classification_temporary.write_text(
        json.dumps(classification_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(classification_temporary, classification_path)
    invalid = [
        (kernel, row["exact_match_count"])
        for kernel, row in classification.items()
        if row["exact_match_count"] != 1
    ]
    if invalid:
        kernel, count = invalid[0]
        raise RuntimeError(
            f"kernel {kernel!r} has {count} exact Python distribution binary matches; "
            f"classification has {len(invalid)} invalid kernels"
        )

    payload = {
        "schema_version": INDEX_SCHEMA,
        "source_build_root": str(build_root),
        "kernel_count": len(kernels),
        "distribution_count": len(compact_distributions),
        "distributions": compact_distributions,
        "kernels": resolved,
    }
    payload["content_sha256"] = _canonical_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return payload


def classify_binary_candidates(
    *,
    build_root: Path,
    kernel_names: Iterable[str],
    symbol_verifier: Callable[[Path, str], bool],
    workers: int = 8,
    distribution_names: Iterable[str] | None = None,
) -> dict:
    """Return all exact package candidates without requiring uniqueness.

    This is used by case-level binders that have stronger runtime module
    identity than a symbol alone.  It performs the same RECORD, SHA, size,
    symlink, and CUDA symbol checks as :func:`build_binary_index`.
    """
    build_root = build_root.resolve(strict=True)
    kernels = sorted(set(str(name) for name in kernel_names))
    if not kernels or any(not name for name in kernels):
        raise RuntimeError("Python package binary index needs nonempty kernel names")
    distributions = collect_distribution_binaries(
        build_root, distribution_names=distribution_names
    )
    patterns = [(name, name.encode("utf-8")) for name in kernels]

    work: list[tuple[dict, dict, Path]] = []
    for distribution in distributions:
        site_root = (
            build_root / distribution["site_root"]
        )
        for binary in distribution["binaries"]:
            work.append((distribution, binary, site_root / binary["path"]))

    def inspect(item: tuple[dict, dict, Path]) -> tuple[dict, list[str]]:
        distribution, binary, path = item
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != binary["record_sha256"] or len(payload) != binary["size"]:
            raise RuntimeError(
                f"installed distribution binary differs from RECORD: {path}"
            )
        matches = [name for name, pattern in patterns if _contains_exact_symbol(payload, pattern)]
        identity = {
            "distribution": distribution["name"],
            "version": distribution["version"],
            "root_kind": distribution["root_kind"],
            "site_root": distribution["site_root"],
            "path": binary["path"],
            "size": binary["size"],
            "sha256": actual,
        }
        return identity, matches

    kernel_candidates = {name: [] for name in kernels}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        for identity, matches in executor.map(inspect, work):
            for kernel in matches:
                kernel_candidates[kernel].append(identity)

    classification: dict[str, dict] = {}
    for kernel in kernels:
        verified = []
        for identity in kernel_candidates[kernel]:
            path = build_root / identity["site_root"] / identity["path"]
            verification = symbol_verifier(path, kernel)
            if verification:
                match = dict(identity)
                if isinstance(verification, dict):
                    match["sass_verification"] = verification
                verified.append(match)
        classification[kernel] = {
            "string_candidate_count": len(kernel_candidates[kernel]),
            "exact_match_count": len(verified),
            "matches": verified,
        }
    compact_distributions = [
        {
            "name": row["name"],
            "version": row["version"],
            "root_kind": row["root_kind"],
            "site_root": row["site_root"],
            "record": row["record"],
            "record_sha256": row["record_sha256"],
            "binary_count": len(row["binaries"]),
            "binary_bytes": sum(int(item["size"]) for item in row["binaries"]),
            "binary_inventory_sha256": _canonical_digest(row["binaries"]),
        }
        for row in distributions
    ]
    payload = {
        "schema_version": "hopper-python-package-binary-candidates-v1",
        "source_build_root": str(build_root),
        "kernel_count": len(kernels),
        "distribution_count": len(compact_distributions),
        "distributions": compact_distributions,
        "kernels": classification,
    }
    payload["content_sha256"] = _canonical_digest(payload)
    return payload


def load_and_bind_index(
    index_path: Path,
    build_root: Path,
    distribution_names: Iterable[str] | None = None,
) -> dict:
    """Bind an index to another exact install without rescanning all cubins."""
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != INDEX_SCHEMA:
        raise RuntimeError("unsupported Python package binary index schema")
    expected = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if expected != _canonical_digest(unsigned):
        raise RuntimeError("Python package binary index content digest mismatch")
    build_root = build_root.resolve(strict=True)
    current = collect_distribution_binaries(
        build_root, distribution_names=distribution_names
    )
    current_compact = [
        {
            "name": row["name"],
            "version": row["version"],
            "root_kind": row["root_kind"],
            "site_root": row["site_root"],
            "record": row["record"],
            "record_sha256": row["record_sha256"],
            "binary_count": len(row["binaries"]),
            "binary_bytes": sum(int(item["size"]) for item in row["binaries"]),
            "binary_inventory_sha256": _canonical_digest(row["binaries"]),
        }
        for row in current
    ]
    if current_compact != payload.get("distributions"):
        raise RuntimeError("Python distribution binary closure differs from indexed install")
    return payload


def indexed_binary(index: dict, build_root: Path, kernel_name: str) -> tuple[Path, dict]:
    identity = (index.get("kernels") or {}).get(kernel_name)
    if not isinstance(identity, dict):
        raise RuntimeError(f"kernel {kernel_name!r} is absent from Python package binary index")
    build_root = build_root.resolve(strict=True)
    site_root = (
        build_root / str(identity.get("site_root") or "")
    )
    relative = Path(str(identity.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("indexed Python package binary path escapes site root")
    path = site_root / relative
    try:
        path.resolve(strict=True).relative_to(site_root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError("indexed Python package binary escapes or is missing") from exc
    _assert_no_symlink(path, site_root)
    if not path.is_file() or path.stat().st_size != int(identity.get("size") or 0):
        raise RuntimeError("indexed Python package binary size/type mismatch")
    if sha256(path) != identity.get("sha256"):
        raise RuntimeError("indexed Python package binary SHA-256 mismatch")
    return path, dict(identity)
