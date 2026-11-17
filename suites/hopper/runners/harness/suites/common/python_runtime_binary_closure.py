#!/usr/bin/env python3
"""Bind captured Python kernels to exact task or distribution binaries.

Runtime-generated extensions are case-local and are never discovered by
walking a shared cache.  Distribution candidates are restricted by RECORD;
when a symbol is linked into multiple FlashInfer modules, the public operation
and its recorded arguments must identify one exact module URI.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Iterable

import python_package_binary_closure as package_closure


CASE_INDEX_SCHEMA = "hopper-python-runtime-case-binary-index-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"runtime binary closure contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"runtime binary closure contains non-regular file: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def _case_module_path(case: dict) -> tuple[str, str] | None:
    """Return an exact FlashInfer package module path from public API inputs."""
    operation = str(case.get("operation") or "")
    if operation == "merge_states":
        return (
            "flashinfer_jit_cache/jit_cache/cascade/cascade.so",
            "flashinfer_merge_states_public_api_cascade_module",
        )
    if operation != "single_decode" or bool(case.get("use_tensor_cores")):
        return None
    dtype = {"fp16": "f16", "bf16": "bf16"}.get(str(case.get("dtype") or ""))
    head_dim = int(case.get("head_dim") or 0)
    posenc = {"NONE": 0, "ROPE_LLAMA": 1, "ALIBI": 2}.get(
        str(case.get("pos_encoding_mode") or "NONE")
    )
    if not dtype or head_dim <= 0 or posenc is None:
        raise RuntimeError("single_decode module identity lacks exact dtype/head_dim/posenc")
    if int(case.get("window_left", -1)) != -1:
        use_swa = True
    else:
        use_swa = False
    logits_cap = float(case.get("logits_soft_cap") or 0.0) > 0.0
    uri = (
        f"single_decode_with_kv_cache_dtype_q_{dtype}_dtype_kv_{dtype}_"
        f"dtype_o_{dtype}_head_dim_qk_{head_dim}_head_dim_vo_{head_dim}_"
        f"posenc_{posenc}_use_swa_{use_swa}_use_logits_cap_{logits_cap}"
    )
    return (
        f"flashinfer_jit_cache/jit_cache/{uri}/{uri}.so",
        "flashinfer_single_decode_public_api_module_uri",
    )


def validate_runtime_closure(runtime_root: Path) -> tuple[Path, dict]:
    runtime_root = runtime_root.resolve(strict=True)
    closure_root = runtime_root.parent
    marker = closure_root / "ACCEPTED"
    manifest_path = closure_root / "runtime_binary_closure_manifest.json"
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError("runtime binary closure lacks ACCEPTED marker")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("runtime binary closure lacks manifest")
    accepted = json.loads(marker.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        accepted.get("status") != "accepted"
        or manifest.get("status") != "accepted"
        or manifest.get("runtime_root") != runtime_root.name
        or accepted.get("manifest_sha256") != sha256(manifest_path)
    ):
        raise RuntimeError("runtime binary closure identity is inconsistent")
    for path in [runtime_root, *runtime_root.rglob("*")]:
        if path.is_symlink():
            raise RuntimeError(f"runtime binary closure contains symlink: {path}")
        if path.stat().st_mode & 0o222:
            raise RuntimeError(f"runtime binary closure is writable: {path}")
    if _canonical_digest(_file_inventory(runtime_root)) != manifest.get(
        "file_inventory_sha256"
    ):
        raise RuntimeError("runtime binary closure file inventory identity differs")
    return runtime_root, manifest


def _runtime_candidates(
    *, case_id: str, runtime_root: Path, runtime_manifest: dict
) -> list[dict]:
    row = (runtime_manifest.get("cases") or {}).get(case_id)
    if not isinstance(row, dict):
        raise RuntimeError(f"runtime binary closure has no case {case_id}")
    candidates = []
    for identity in row.get("binaries") or []:
        relative = Path(str(identity.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"{case_id}: runtime binary path escapes closure")
        path = runtime_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{case_id}: runtime binary is missing or a symlink")
        if path.stat().st_size != int(identity.get("size") or -1):
            raise RuntimeError(f"{case_id}: runtime binary size differs from closure")
        if sha256(path) != str(identity.get("sha256") or ""):
            raise RuntimeError(f"{case_id}: runtime binary SHA differs from closure")
        candidates.append({**identity, "_absolute_path": path})
    return candidates


def _resolve_case(
    *,
    case_id: str,
    kernel: str,
    case: dict,
    runtime_root: Path,
    runtime_manifest: dict,
    package_candidates: list[dict],
    package_root: Path,
    symbol_verifier: Callable[[Path, str], bool],
) -> dict:
    runtime_matches = []
    for identity in _runtime_candidates(
        case_id=case_id, runtime_root=runtime_root, runtime_manifest=runtime_manifest
    ):
        verification = symbol_verifier(identity["_absolute_path"], kernel)
        if verification:
            match = {key: value for key, value in identity.items() if key != "_absolute_path"}
            match.update(
                {
                    "root_kind": "frozen_case_runtime",
                    "resolution": "exact_case_runtime_symbol",
                }
            )
            if isinstance(verification, dict):
                match["sass_verification"] = verification
            runtime_matches.append(match)
    runtime_contents = {row["sha256"] for row in runtime_matches}
    if len(runtime_contents) > 1:
        raise RuntimeError(f"{case_id}: primary kernel occurs in multiple case runtime binaries")
    if runtime_matches:
        selected = sorted(runtime_matches, key=lambda row: row["path"])[0]
        selected["equivalent_paths"] = sorted(row["path"] for row in runtime_matches)
        return selected

    matches = list(package_candidates)
    route = _case_module_path(case)
    resolution = "unique_distribution_symbol"
    if len(matches) != 1 and route is not None:
        expected, resolution = route
        matches = [row for row in matches if str(row.get("path") or "") == expected]
    if len(matches) != 1:
        raise RuntimeError(
            f"{case_id}: primary kernel has {len(matches)} exact binaries after runtime "
            "module binding"
        )
    selected = dict(matches[0])
    selected["root_kind"] = "frozen_python_distribution"
    selected["resolution"] = resolution
    if route is not None:
        selected["expected_module_path"] = route[0]
    path = package_root / selected["site_root"] / selected["path"]
    if path.is_symlink() or not path.is_file() or sha256(path) != selected["sha256"]:
        raise RuntimeError(f"{case_id}: selected distribution binary identity changed")
    return selected


def _validate_package_runtime_identity(
    *,
    identity_path: Path,
    package_root: Path,
    distribution_names: Iterable[str],
    package_distributions: list[dict],
    cases: dict[str, dict],
) -> dict:
    """Bind selected package code to the runtime identity captured on H100.

    The identity inventories every file in the immutable image distribution.
    Finalization only re-hashes CUDA candidates through RECORD validation, then
    proves that every selected binary has the exact size and hash recorded by
    the H100 runtime identity.  This avoids re-hashing unrelated package data.
    """
    identity_path = identity_path.resolve(strict=True)
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "hopper.flashinfer-runtime-identity.v1"
        or payload.get("status") != "pass"
    ):
        raise RuntimeError("unsupported or unsuccessful FlashInfer runtime identity")
    selected = {
        package_closure.canonical_distribution_name(name)
        for name in distribution_names
    }
    if not selected:
        raise RuntimeError("package runtime identity distribution allow-list is empty")
    distribution_rows: dict[str, dict] = {}
    for row in package_distributions:
        canonical = package_closure.canonical_distribution_name(row.get("name") or "")
        if canonical in distribution_rows:
            raise RuntimeError(f"duplicate selected Python distribution {canonical}")
        distribution_rows[canonical] = row
    if set(distribution_rows) != selected:
        raise RuntimeError("package candidate distributions differ from runtime allow-list")

    runtime_packages = payload.get("packages") or {}
    summaries = []
    inventories: dict[str, dict[str, dict]] = {}
    for canonical in sorted(selected):
        matching = [
            row
            for name, row in runtime_packages.items()
            if package_closure.canonical_distribution_name(name) == canonical
        ]
        if len(matching) != 1 or not isinstance(matching[0], dict):
            raise RuntimeError(
                f"FlashInfer runtime identity lacks one exact {canonical} distribution"
            )
        runtime_row = matching[0]
        package_row = distribution_rows[canonical]
        if runtime_row.get("version") != package_row.get("version"):
            raise RuntimeError(
                f"FlashInfer runtime identity version differs for {canonical}"
            )
        location_text = str(runtime_row.get("location") or "")
        location = Path(location_text)
        if not location.is_absolute():
            raise RuntimeError(f"FlashInfer runtime identity location is not absolute: {canonical}")
        files = runtime_row.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(
                f"FlashInfer runtime identity has no file inventory for {canonical}"
            )
        inventory: dict[str, dict] = {}
        normalized_rows = []
        for item in files:
            if not isinstance(item, dict):
                raise RuntimeError(f"invalid runtime identity file row for {canonical}")
            relative_text = str(item.get("relative") or "")
            relative = Path(relative_text)
            absolute = Path(str(item.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise RuntimeError(f"runtime identity file escapes distribution: {relative_text}")
            try:
                rebound = absolute.relative_to(location)
            except ValueError as exc:
                raise RuntimeError(
                    f"runtime identity file is outside distribution: {absolute}"
                ) from exc
            if rebound.as_posix() != relative.as_posix():
                raise RuntimeError(
                    f"runtime identity relative path differs for {relative_text}"
                )
            try:
                size = int(item["bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid runtime identity file size: {relative_text}"
                ) from exc
            digest = str(item.get("sha256") or "")
            if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RuntimeError(f"invalid runtime identity file identity: {relative_text}")
            key = relative.as_posix()
            if key in inventory:
                raise RuntimeError(f"duplicate runtime identity file: {relative_text}")
            normalized = {"path": key, "size": size, "sha256": digest}
            inventory[key] = normalized
            normalized_rows.append(normalized)
        inventories[canonical] = inventory
        summaries.append(
            {
                "name": package_row["name"],
                "version": package_row["version"],
                "captured_location": location_text,
                "rebound_package_root": str(package_root),
                "file_count": len(normalized_rows),
                "file_inventory_sha256": _canonical_digest(
                    sorted(normalized_rows, key=lambda row: row["path"])
                ),
            }
        )

    for case_id, row in cases.items():
        binary = row.get("binary") or {}
        if binary.get("root_kind") != "frozen_python_distribution":
            continue
        canonical = package_closure.canonical_distribution_name(
            binary.get("distribution") or ""
        )
        if canonical not in selected:
            raise RuntimeError(f"{case_id}: selected package binary is outside allow-list")
        relative = Path(str(binary.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"{case_id}: selected package binary path escapes")
        expected = inventories[canonical].get(relative.as_posix())
        if expected is None or (
            expected["size"] != int(binary.get("size") or -1)
            or expected["sha256"] != str(binary.get("sha256") or "")
        ):
            raise RuntimeError(
                f"{case_id}: selected package binary differs from H100 runtime identity"
            )
    return {
        "schema_version": payload["schema_version"],
        "source_sha256": sha256(identity_path),
        "expected_version": payload.get("expected_version"),
        "distributions": summaries,
    }


def build_case_binary_index(
    *,
    package_root: Path,
    runtime_root: Path,
    case_requests: dict[str, dict],
    output: Path,
    symbol_verifier: Callable[[Path, str], bool],
    workers: int = 8,
    package_distribution_names: Iterable[str] | None = None,
    package_runtime_identity: Path | None = None,
) -> dict:
    package_root = package_root.resolve(strict=True)
    runtime_root, runtime_manifest = validate_runtime_closure(runtime_root)
    kernels = sorted({str(row["kernel"]) for row in case_requests.values()})
    package = package_closure.classify_binary_candidates(
        build_root=package_root,
        kernel_names=kernels,
        symbol_verifier=symbol_verifier,
        workers=workers,
        distribution_names=package_distribution_names,
    )
    cases = {}
    for case_id, request in sorted(case_requests.items()):
        kernel = str(request.get("kernel") or "")
        case = request.get("case")
        if not kernel or not isinstance(case, dict):
            raise RuntimeError(f"{case_id}: binary request is incomplete")
        candidates = (package["kernels"].get(kernel) or {}).get("matches") or []
        cases[case_id] = {
            "kernel": kernel,
            "binary": _resolve_case(
                case_id=case_id,
                kernel=kernel,
                case=case,
                runtime_root=runtime_root,
                runtime_manifest=runtime_manifest,
                package_candidates=candidates,
                package_root=package_root,
                symbol_verifier=symbol_verifier,
            ),
        }
    identity = None
    if package_runtime_identity is not None:
        if package_distribution_names is None:
            raise RuntimeError(
                "package runtime identity requires an exact distribution allow-list"
            )
        identity = _validate_package_runtime_identity(
            identity_path=package_runtime_identity,
            package_root=package_root,
            distribution_names=package_distribution_names,
            package_distributions=package["distributions"],
            cases=cases,
        )
    payload = {
        "schema_version": CASE_INDEX_SCHEMA,
        "case_count": len(cases),
        "kernel_count": len(kernels),
        "package_distributions": package["distributions"],
        "package_candidates_content_sha256": package["content_sha256"],
        "runtime_closure_content_sha256": runtime_manifest.get("content_sha256"),
        "cases": cases,
    }
    if identity is not None:
        payload["package_runtime_identity"] = identity
    payload["content_sha256"] = _canonical_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return payload


def bind_case_index_runtime_identity(index: dict, identity_path: Path) -> None:
    """Require a supplied runtime identity to be the one bound into an index."""
    retained = index.get("package_runtime_identity") or {}
    if retained.get("source_sha256") != sha256(identity_path.resolve(strict=True)):
        raise RuntimeError("runtime case binary index package identity differs")


def indexed_case_binary(
    index: dict,
    *,
    case_id: str,
    package_root: Path,
    runtime_root: Path,
) -> tuple[Path, dict]:
    if index.get("schema_version") != CASE_INDEX_SCHEMA:
        raise RuntimeError("unsupported Python runtime case binary index schema")
    unsigned = dict(index)
    expected = unsigned.pop("content_sha256", None)
    if expected != _canonical_digest(unsigned):
        raise RuntimeError("Python runtime case binary index content digest mismatch")
    row = (index.get("cases") or {}).get(case_id)
    if not isinstance(row, dict):
        raise RuntimeError(f"case {case_id!r} is absent from runtime binary index")
    identity = dict(row.get("binary") or {})
    if row.get("kernel") != identity.get("kernel", row.get("kernel")):
        raise RuntimeError(f"{case_id}: runtime binary kernel identity mismatch")
    root_kind = identity.get("root_kind")
    if root_kind == "frozen_case_runtime":
        root = runtime_root.resolve(strict=True)
        relative = Path(str(identity.get("path") or ""))
    elif root_kind == "frozen_python_distribution":
        root = (package_root / str(identity.get("site_root") or "")).resolve(strict=True)
        relative = Path(str(identity.get("path") or ""))
    else:
        raise RuntimeError(f"{case_id}: unsupported runtime binary root kind")
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{case_id}: indexed runtime binary path escapes root")
    path = root / relative
    try:
        path.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"{case_id}: indexed runtime binary is missing or escapes") from exc
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{case_id}: indexed runtime binary is not a regular file")
    if path.stat().st_size != int(identity.get("size") or -1) or sha256(path) != str(
        identity.get("sha256") or ""
    ):
        raise RuntimeError(f"{case_id}: indexed runtime binary identity changed")
    return path, identity
