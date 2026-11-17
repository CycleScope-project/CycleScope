#!/usr/bin/env python3
"""Generic capture-case adapter: one real invocation -> hoppersim capture.

``marker_selected_kernel_occurrence`` selects the declared one-based repeated
kernel occurrence from the actual capture inventory; absent occurrences fail
qualification instead of silently selecting a warmup or the largest launch.

Stage-6 cutover contract:

- The only real capture entry is ``hoppersim capture`` (original launch,
  actually loaded code, topology, dynamic facts).  This adapter resolves one
  case out of a suite manifest into a PURE invocation description and drives
  that entry; it never re-implements injection or the post-capture chain.
- No suite/library-name semantics live here.  A case without an explicit
  ``argv`` fails closed; there is no library-name fallback resolver, and the
  primary kernel is selected downstream from this launch's own inventory
  (``launch_facts_raw.csv`` / ``kernelslist``) by explicit kernel_file or
  launch ordinal, never by name tokens.
- Independent marker measurement (qualification timing) stays outside this
  path: it may read these captures, but never reshapes them.

Invocation schema (``hopper.capture-case-invocation.v1``), taken from the
case row's ``invocation`` object or from ``--invocation-json``::

    {
      "argv": ["/abs/prog", "--flag", "value"],   # required, exact program argv
      "program_file": "/abs/prog",                # archived identity anchor
                                                 # (default: argv[0] when it
                                                 #  is a file)
      "cwd": "/abs/dir",                          # optional working directory
      "env": {"KEY": "VALUE"},                    # optional str->str delta
      "source_file": "/abs/path",                 # optional program source
      "prep": [{"argv": [...], "cwd": "...", "env": {...}}],  # optional
                                                 # workload-only preparation
                                                 # commands (JIT warm/build);
                                                 # run untraced before capture
      "kernel_range": "",                         # NVBit DYNAMIC_KERNEL_RANGE;
                                                 # empty records every launch
      "tma_recorder": "/abs/libtma_tensor_map_recorder.so",
      "tracer_root": "/abs/tracer_nvbit",
      "carveout": "auto" | "required" | "off",
      "allow_missing_lineinfo": false,
      "skip_enrich": false,                      # defer enrichment to the CPU
                                                 # wrapper: hoppersim capture still
                                                 # completes the raw capture
                                                 # (manifest/tracegs/sidecars as
                                                 # traced, pc evidence) and
                                                 # records enrichment status
                                                 # skipped; never faked
      "marker_contract": {"clock_marker_policy": "outermost_pair"},
      "marker_selected_kernel": "<resolved kernel symbol>",
      "marker_selected_kernel_occurrence": 4,    # 1-based; three warmups then formal call
      "unit_selection": {"type": "cta" | "cluster" | "launch",
                         "kernel_file": "kernel-1-....trace",
                         "ctas": [[x, y, z]]}
                         # type=launch: complete ordinary bounded launch
                         # unit; ctas enumerates the original launch's whole
                         # grid (exactly 1 or 2 CTAs; cluster launches use
                         # type=cluster instead)
    }

Outputs (inside ``--output``, which must not exist yet): the standard
hoppersim capture directory plus ``selected_runtime.sass`` (cuobjdump control
word dump of the actually dumped live cubins), ``capture_metadata.json``
(wrapper + ``manifest_case`` for the conversion group staging),
``qualification.json`` (capture qualification facts; timing lives elsewhere)
and ``invocation.json`` (the resolved invocation, verbatim).

``comparison_contract.identity`` shape follows the traced launch facts and
the invocation's explicit selector.  A single-launch capture records the
per-launch list.  When the invocation carries ``marker_selected_kernel``
(the untraced measurement's resolved symbol, threaded through gen
--marker-variant), the identity records
``{identity_kind=repeated_launch_protocol_run, selected_kernel_symbol,
launch_count, grid, block, launches[ordinal/file/name/seq],
selected_launch}`` -- every physical launch keeps its own trace file
(tracer_tool.cu names kernel-%d-ctx per launch) and ordinal; nothing is
folded into one file and file ids are never treated as code identity.  A
multi-kernel capture without a selector keeps the per-launch list, which
refuses at conversion because a unit cannot span kernels.  The denominator
marker interval is measured by the same program run bare (its own 27->9
protocol, reduced to the unit's slots); the capture only carries the real
executed CS2R instructions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
from hoppersim.kernel_symbols import kernel_symbol_contains  # noqa: E402


INVOCATION_SCHEMA = "hopper.capture-case-invocation.v1"
QUALIFICATION_SCHEMA = "hopper.capture-case-qualification.v1"
CAPTURE_METADATA_SCHEMA = "hopper-python-single-sm-capture-v1"
MARKER_POLICIES = (
    "auto",
    "whole_kernel_pair",
    "outermost_pair",
    "internal_measurement_reads",
)
UNIT_TYPES = ("cta", "cluster", "launch")
CARVEOUT_MODES = ("auto", "required", "off")


class AdapterError(RuntimeError):
    """Fail-closed adapter contract violation."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str, *, anchor: Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve()


def _require_file(value: object, *, anchor: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{label} must be a non-empty path string")
    path = _resolve_path(value, anchor=anchor, label=label)
    if not path.is_file():
        raise AdapterError(f"{label} does not name a file: {path}")
    return path


def load_invocation(
    suite_manifest: Path, case_id: str, invocation_json: Path | None
) -> dict[str, Any]:
    """Resolve one case's invocation; argv is mandatory, never inferred."""
    if invocation_json is not None:
        try:
            payload = json.loads(invocation_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterError(f"cannot read --invocation-json: {exc}")
        if not isinstance(payload, dict):
            raise AdapterError("--invocation-json must be a JSON object")
        # The standalone document IS the invocation (optionally carrying a
        # case_id); a wrapper {"invocation": {...}} is also accepted so the
        # same file shape works standalone and merged into manifests.
        invocation = (
            payload.get("invocation")
            if isinstance(payload.get("invocation"), dict)
            else payload
        )
        return _validate_invocation(invocation, case_id=case_id)
    else:
        try:
            manifest = json.loads(suite_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterError(f"cannot read suite manifest: {exc}")
        rows = [
            row
            for row in (manifest.get("cases") or [])
            if isinstance(row, dict) and str(row.get("case_id") or "") == case_id
        ]
        if len(rows) != 1:
            raise AdapterError(
                f"case {case_id!r} matches {len(rows)} rows in {suite_manifest}; "
                "exactly one is required"
            )
        case = rows[0]
    invocation = case.get("invocation")
    if not isinstance(invocation, dict):
        raise AdapterError(
            f"case {case_id} has no invocation object (argv is required); "
            "library-name resolution is not a capture input"
        )
    return _validate_invocation(invocation, case_id=case_id)


def _validate_invocation(invocation: Mapping[str, Any], *, case_id: str) -> dict[str, Any]:
    argv = invocation.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise AdapterError(
            f"{case_id}: invocation.argv must be a non-empty list of strings; "
            "library-name resolution is not a capture input"
        )
    anchor = PROJECT_ROOT
    program_file = invocation.get("program_file")
    if program_file is not None:
        program = _require_file(program_file, anchor=anchor, label="program_file")
    else:
        candidate = _resolve_path(argv[0], anchor=anchor, label="argv[0]")
        if not candidate.is_file():
            raise AdapterError(
                f"{case_id}: argv[0] is not a file and no program_file is "
                f"declared: {argv[0]}"
            )
        program = candidate
    env = invocation.get("env") or {}
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise AdapterError(f"{case_id}: invocation.env must be str -> str")
    cwd: Path | None = None
    if invocation.get("cwd") is not None:
        if not isinstance(invocation["cwd"], str):
            raise AdapterError(f"{case_id}: invocation.cwd must be a path string")
        cwd = _resolve_path(invocation["cwd"], anchor=anchor, label="cwd")
        if not cwd.is_dir():
            raise AdapterError(f"{case_id}: invocation.cwd is not a directory: {cwd}")
    source_file: Path | None = None
    if invocation.get("source_file") is not None:
        source_file = _require_file(
            invocation["source_file"], anchor=anchor, label="source_file"
        )
    kernel_range = invocation.get("kernel_range", "")
    if not isinstance(kernel_range, str):
        raise AdapterError(f"{case_id}: kernel_range must be a string")
    carveout = str(invocation.get("carveout") or "auto")
    if carveout not in CARVEOUT_MODES:
        raise AdapterError(
            f"{case_id}: carveout must be one of {CARVEOUT_MODES}, got {carveout}"
        )
    tma_recorder: Path | None = None
    if invocation.get("tma_recorder") is not None:
        tma_recorder = _require_file(
            invocation["tma_recorder"], anchor=anchor, label="tma_recorder"
        )
    tracer_root: Path | None = None
    if invocation.get("tracer_root") is not None:
        # Boring contract: tracer_root IS the CLI root directory that
        # hoppersim capture's --tracer-root expects -- never the tracer .so
        # file.  The tool layout under it is validated here so a wrong
        # shape fails closed at invocation resolution, not mid-capture.
        if not isinstance(invocation["tracer_root"], str):
            raise AdapterError(f"{case_id}: tracer_root must be a path string")
        root = _resolve_path(
            invocation["tracer_root"], anchor=anchor, label="tracer_root"
        )
        if not root.is_dir():
            raise AdapterError(
                f"{case_id}: tracer_root is not a directory (the CLI root "
                f"is expected, not the tracer .so): {root}"
            )
        for tool_rel in (
            "tracer_tool/tracer_tool.so",
            "tracer_tool/traces-processing/post-traces-processing",
        ):
            if not (root / tool_rel).exists():
                raise AdapterError(
                    f"{case_id}: tracer_root is missing {tool_rel}: {root}"
                )
        tracer_root = root
    marker = invocation.get("marker_contract") or {}
    if not isinstance(marker, dict):
        raise AdapterError(f"{case_id}: marker_contract must be an object")
    policy = str(marker.get("clock_marker_policy") or "auto")
    if policy not in MARKER_POLICIES:
        raise AdapterError(
            f"{case_id}: clock_marker_policy must be one of {MARKER_POLICIES}, "
            f"got {policy}"
        )
    unit = invocation.get("unit_selection")
    if unit is not None:
        if not isinstance(unit, dict) or unit.get("type") not in UNIT_TYPES:
            raise AdapterError(
                f"{case_id}: unit_selection.type must be one of {UNIT_TYPES}"
            )
        # kernel_file is the CONVERTER-side binding (which trace file the
        # selected launch instance lives in) and is optional at capture
        # time: the capture-first flow declares the unit (type + ctas)
        # before the trace files exist, and the same-unit facts bind the
        # selected physical launch file afterwards (gen
        # --marker-unit-from-capture).  When present it must be non-empty.
        if "kernel_file" in unit and (
                not isinstance(unit["kernel_file"], str)
                or not unit["kernel_file"]):
            raise AdapterError(
                f"{case_id}: unit_selection.kernel_file, when present, must "
                "be a non-empty string"
            )
        ctas = unit.get("ctas")
        if not isinstance(ctas, list) or not ctas or not all(
            isinstance(item, list) and len(item) == 3
            and all(isinstance(axis, int) for axis in item)
            for item in ctas
        ):
            raise AdapterError(
                f"{case_id}: unit_selection.ctas must be [x, y, z] integer lists"
            )
        if unit["type"] == "launch" and len(ctas) not in (1, 2):
            raise AdapterError(
                f"{case_id}: unit_selection.type=launch is the complete "
                "ordinary bounded-launch unit and must enumerate exactly the "
                "1 or 2 CTAs of the original launch's whole grid"
            )
    prep: list[dict[str, Any]] = []
    for item in invocation.get("prep") or []:
        if not isinstance(item, dict) or not isinstance(item.get("argv"), list):
            raise AdapterError(f"{case_id}: each prep entry needs an argv list")
        prep.append(item)
    marker_selected_kernel = invocation.get("marker_selected_kernel")
    if marker_selected_kernel is not None and (
            not isinstance(marker_selected_kernel, str)
            or not marker_selected_kernel.strip()):
        raise AdapterError(
            f"{case_id}: marker_selected_kernel must be a non-empty kernel "
            "symbol string"
        )
    if (isinstance(unit, dict) and unit.get("type") in UNIT_TYPES
            and not unit.get("kernel_file")
            and not (marker_selected_kernel or "").strip()):
        # Fail at invocation resolution, not after the GPU run: a
        # capture-time unit without kernel_file is bound through the
        # explicit selector's selected launch, and without that bare-run
        # kernel fact there is nothing to bind -- a launch/file count is
        # never a selector guess.
        raise AdapterError(
            f"{case_id}: unit_selection without kernel_file needs "
            "marker_selected_kernel (the bare measurement's resolved "
            "symbol); refusing before capture"
        )
    occurrence = invocation.get("marker_selected_kernel_occurrence", 1)
    if (isinstance(occurrence, bool) or not isinstance(occurrence, int)
            or occurrence < 1):
        raise AdapterError(
            f"{case_id}: marker_selected_kernel_occurrence must be a positive integer"
        )
    if "marker_selected_kernel_occurrence" in invocation and not marker_selected_kernel:
        raise AdapterError(
            f"{case_id}: marker_selected_kernel_occurrence needs marker_selected_kernel"
        )
    allow_missing_lineinfo = bool(invocation.get("allow_missing_lineinfo"))
    skip_enrich = invocation.get("skip_enrich", False)
    if not isinstance(skip_enrich, bool):
        raise AdapterError(f"{case_id}: skip_enrich must be a boolean")
    return {
        "argv": list(argv),
        "program_file": program,
        "cwd": cwd,
        "env": dict(env),
        "source_file": source_file,
        "kernel_range": kernel_range,
        "carveout": carveout,
        "tma_recorder": tma_recorder,
        "tracer_root": tracer_root,
        "marker_selected_kernel": (
            str(marker_selected_kernel) if marker_selected_kernel else None),
        "marker_selected_kernel_occurrence": occurrence,
        "clock_marker_policy": policy,
        "unit_selection": dict(unit) if isinstance(unit, dict) else None,
        "prep": prep,
        "allow_missing_lineinfo": allow_missing_lineinfo,
        "skip_enrich": skip_enrich,
    }


def _cuobjdump() -> str:
    tool = os.environ.get("CUOBJDUMP") or shutil.which("cuobjdump")
    if not tool:
        raise AdapterError(
            "cuobjdump not found (set CUOBJDUMP); the static control-word dump "
            "is a capture identity requirement, not an optional extra"
        )
    return tool


def build_selected_sass(capture_dir: Path) -> tuple[Path, dict[str, str]]:
    """cuobjdump -sass over the actually dumped live cubins (dedup by sha).

    The dump is the conversion-side static control-word source; it is built
    only from the live-module dumps this capture archived, never from a
    rebuilt binary or a name match.
    """
    manifest_path = capture_dir / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cubins: dict[str, str] = {}  # sha -> rel path
    for launch in manifest.get("launches") or []:
        code = launch.get("code") or {}
        entry = code.get("cubin")
        if isinstance(entry, dict) and entry.get("path"):
            rel = str(entry["path"])
            sha = str(entry.get("sha256") or _sha256(capture_dir / rel))
            cubins.setdefault(sha, rel)
    if not cubins:
        raise AdapterError(
            f"{capture_dir}: no live dumped cubin in capture_manifest.json; "
            "static SASS identity cannot be built (nvbit_dump_cubin produced "
            "nothing for this run)"
        )
    tool = _cuobjdump()
    sections: list[str] = []
    for sha, rel in sorted(cubins.items()):
        completed = subprocess.run(
            [tool, "-sass", str(capture_dir / rel)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            raise AdapterError(
                f"cuobjdump -sass failed for {rel}: {completed.stderr[-400:]}"
            )
        sections.append(completed.stdout)
        (capture_dir / f"code/{Path(rel).name}.sha256").write_text(
            f"{sha}  {Path(rel).name}\n", encoding="utf-8"
        )
    target = capture_dir / "selected_runtime.sass"
    target.write_text("\n".join(sections), encoding="utf-8")
    return target, dict(cubins)


def _launch_inventory(capture_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        (capture_dir / "capture_manifest.json").read_text(encoding="utf-8")
    )
    topology: dict[str, dict[str, Any]] = {}
    cta_map = capture_dir / "cta_sm_map.csv"
    if cta_map.is_file():
        with cta_map.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item = topology.setdefault(
                    str(row.get("kernel_file") or ""),
                    {"ctas": set(), "sm_ids": set()},
                )
                key = (
                    row.get("cta_x"),
                    row.get("cta_y"),
                    row.get("cta_z"),
                )
                item["ctas"].add(key)
                item["sm_ids"].add(row.get("sm_id") or "")
    inventory = []
    for launch in manifest.get("launches") or []:
        kernel_file = str(launch.get("kernel_file") or "")
        entry = {
            "kernel_file": kernel_file,
            "kernel_name": launch.get("kernel_name_mangled"),
            "launch_seq": launch.get("launch_seq"),
            "context_id": launch.get("context_id"),
            "grid": launch.get("grid"),
            "block": launch.get("block"),
        }
        topo = topology.get(kernel_file)
        if topo is not None:
            entry["cta_count"] = len(topo["ctas"])
            entry["sm_ids"] = sorted(value for value in topo["sm_ids"] if value)
        inventory.append(entry)
    if not inventory:
        raise AdapterError(
            f"{capture_dir}: capture_manifest.json records no launch; refusing "
            "to qualify an empty capture"
        )
    return inventory


def write_case_documents(
    *,
    output: Path,
    case_id: str,
    invocation: Mapping[str, Any],
    capture_returncode: int,
) -> dict[str, Any]:
    sass_path, cubins = build_selected_sass(output)
    inventory = _launch_inventory(output)
    manifest = json.loads(
        (output / "capture_manifest.json").read_text(encoding="utf-8")
    )
    run = manifest.get("run") or {}
    # Identity shape follows the traced launch facts.  The tracer writes
    # EACH traced launch its own trace file (tracer_tool.cu: kernel-%d-ctx
    # with ctx_kernelid++ per launch) -- physical file ids are NOT code
    # identity and launch counts never guess the producer kernel.  The
    # producer kernel is the EXPLICIT selector the invocation carries from
    # the untraced measurement's resolved symbol
    # (invocation["marker_selected_kernel"]); every launch keeps its own
    # file/ordinal and the selector names the representative instance.
    # Without a selector (single-launch run-once flows) the identity stays
    # the per-launch list; a multi-launch capture without a selector keeps
    # the list, which refuses downstream (a unit cannot span kernels).
    selector = str(invocation.get("marker_selected_kernel") or "")

    selected_launches = [
        (index, row) for index, row in enumerate(inventory)
        if selector and kernel_symbol_contains(str(row.get("kernel_name") or ""), selector)
    ]
    selected_identities = {
        (row["kernel_name"], tuple(row.get("grid") or []), tuple(row.get("block") or []))
        for _, row in selected_launches
    }
    if selector and len(selected_identities) != 1:
        raise AdapterError(
            f"{case_id}: producer selector {selector!r} matches "
            f"{len(selected_identities)} distinct kernel identities; "
            "an absent or ambiguous producer cannot select a physical launch"
        )
    if selector:
        occurrence = invocation["marker_selected_kernel_occurrence"]
        if occurrence > len(selected_launches):
            raise AdapterError(
                f"{case_id}: marker_selected_kernel_occurrence {occurrence} exceeds "
                f"the {len(selected_launches)} captured launches of {selector!r}"
            )
        selected_ordinal, selected_row = selected_launches[occurrence - 1]
        identity: Any = {
            "identity_kind": "repeated_launch_protocol_run",
            "selected_kernel_symbol": selector,
            "launch_count": len(selected_launches),
            "grid": list(selected_row.get("grid") or []),
            "block": list(selected_row.get("block") or []),
            "launches": [
                {
                    "ordinal": index,
                    "kernel_file": row["kernel_file"],
                    "kernel_name": row["kernel_name"],
                    "launch_seq": row["launch_seq"],
                }
                for index, row in selected_launches
            ],
            "selected_launch": {
                "ordinal": selected_ordinal,
                "kernel_file": selected_row["kernel_file"],
            },
        }
    elif len(inventory) == 1:
        identity = [
            {
                "kernel_file": inventory[0]["kernel_file"],
                "kernel_name": inventory[0]["kernel_name"],
                "launch_seq": inventory[0]["launch_seq"],
            }
        ]
    else:
        identity = [
            {
                "kernel_file": row["kernel_file"],
                "kernel_name": row["kernel_name"],
                "launch_seq": row["launch_seq"],
            }
            for row in inventory
        ]
    manifest_case: dict[str, Any] = {
        "name": case_id,
        "case_id": case_id,
        "status": "captured",
        "selected_sass": str(output / "selected_runtime.sass"),
        "selected_binary_origin": "nvbit_dump_cubin_live_module",
        "clock_marker_policy": invocation["clock_marker_policy"],
        "comparison_contract": {
            "identity": identity,
            "code_identity": {
                "selected_sass_sha256": _sha256(sass_path),
                "archived_binary_sha256": (run.get("binary") or {}).get("sha256"),
                "cubin_sha256": cubins,
            },
            "contract_note": (
                "capture-side code identity only; the timing comparison "
                "contract is owned by the independent marker measurement"
            ),
        },
        "traced_kernels": inventory,
        "primary_kernel_selection": None,
        "primary_kernel_selection_note": (
            "explicit selection at conversion time by kernel_file/launch_seq "
            "from traced_kernels; no name-token classification"
        ),
        "traced_kernel_range": invocation["kernel_range"],
    }
    if len(cubins) == 1:
        manifest_case["selected_cubin"] = str(
            output / next(iter(cubins.values()))
        )
    else:
        manifest_case["selected_cubins"] = [
            str(output / rel) for rel in sorted(cubins.values())
        ]
    if invocation.get("unit_selection"):
        unit = dict(invocation["unit_selection"])
        if not unit.get("kernel_file"):
            if not isinstance(identity, dict) or not identity.get("selected_launch"):
                raise AdapterError(
                    f"{case_id}: capture-time unit needs an explicit kernel "
                    "selector identifying its physical launch"
                )
            unit["kernel_file"] = identity["selected_launch"]["kernel_file"]
        manifest_case["unit_selection"] = unit
    capture_metadata = {
        "schema_version": CAPTURE_METADATA_SCHEMA,
        "case_id": case_id,
        "gpu_name": (run.get("gpu") or {}).get("name"),
        "trace_enrichment_version": None if invocation.get("skip_enrich") else 4,
        "trace_enrichment_status": (
            "skipped" if invocation.get("skip_enrich") else "ran"
        ),
        "capture_returncode": capture_returncode,
        "manifest_case": manifest_case,
    }
    (output / "capture_metadata.json").write_text(
        json.dumps(capture_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    qualification = {
        "schema_version": QUALIFICATION_SCHEMA,
        "case_id": case_id,
        "status": "qualified",
        "qualified_at": _utcnow(),
        "enrichment_status": (
            "skipped" if invocation.get("skip_enrich") else "ran"
        ),
        "capture": {
            "directory": str(output),
            "manifest_sha256": _sha256(output / "capture_manifest.json"),
            "run_id": run.get("run_id"),
            "command": run.get("command"),
            "cwd": run.get("cwd"),
        },
        "launches": inventory,
        "clock_marker_policy": invocation["clock_marker_policy"],
        "carveout": (manifest.get("carveout_pass") or {}).get("status"),
        "tma_recorder": bool(invocation.get("tma_recorder")),
        "code": {
            "selected_sass_sha256": _sha256(sass_path),
            "cubin_sha256": cubins,
        },
        "note": (
            "capture qualification facts only; hardware marker measurement is "
            "an independent program and is not part of this document"
        ),
    }
    (output / "qualification.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "invocation.json").write_text(
        json.dumps(
            {
                "schema_version": INVOCATION_SCHEMA,
                "case_id": case_id,
                "argv": invocation["argv"],
                "program_file": str(invocation["program_file"]),
                "cwd": str(invocation["cwd"]) if invocation.get("cwd") else None,
                "env": invocation["env"],
                "kernel_range": invocation["kernel_range"],
                "carveout": invocation["carveout"],
                "marker_selected_kernel": invocation.get("marker_selected_kernel"),
                **({"marker_selected_kernel_occurrence":
                    invocation["marker_selected_kernel_occurrence"]}
                   if invocation.get("marker_selected_kernel") else {}),
                "clock_marker_policy": invocation["clock_marker_policy"],
                "unit_selection": invocation.get("unit_selection"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return qualification


def run_prep(invocation: Mapping[str, Any], scratch: Path) -> list[dict[str, Any]]:
    """Workload-only preparation commands; untraced, recorded verbatim."""
    records = []
    for index, item in enumerate(invocation.get("prep") or []):
        argv = [str(part) for part in item.get("argv") or []]
        if not argv:
            raise AdapterError(f"prep[{index}] has an empty argv")
        cwd = (
            _resolve_path(str(item["cwd"]), anchor=PROJECT_ROOT, label="prep cwd")
            if item.get("cwd")
            else None
        )
        env = os.environ.copy()
        for key, value in (invocation.get("env") or {}).items():
            env[str(key)] = str(value)
        for key, value in (item.get("env") or {}).items():
            env[str(key)] = str(value)
        log = scratch / f"prep{index}.log"
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write_text(completed.stdout, encoding="utf-8")
        records.append(
            {
                "argv": argv,
                "cwd": str(cwd) if cwd else None,
                "returncode": completed.returncode,
                "log": f"prep/prep{index}.log",
            }
        )
        if completed.returncode != 0:
            raise AdapterError(
                f"prep[{index}] failed with rc={completed.returncode}: {argv}; log={log}"
            )
    return records


def compose_capture_command(
    invocation: Mapping[str, Any],
    *,
    case_id: str,
    output: Path,
    argv_json: Path,
    env_json: Path,
    metadata_out: Path,
    passthrough: Mapping[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "hoppersim",
        "capture",
        "--capture-gate",
        "open",
        "--binary",
        str(invocation["program_file"]),
        "--case-id",
        case_id,
        "--out",
        str(output),
        "--metadata-out",
        str(metadata_out),
        "--argv-json",
        str(argv_json),
        "--env-json",
        str(env_json),
        "--kernel-range",
        str(invocation["kernel_range"]),
        "--carveout",
        str(invocation["carveout"]),
    ]
    if invocation.get("source_file") is not None:
        command += ["--source", str(invocation["source_file"])]
    if invocation.get("tma_recorder") is not None:
        command += ["--tma-recorder", str(invocation["tma_recorder"])]
    if invocation.get("tracer_root") is not None:
        command += ["--tracer-root", str(invocation["tracer_root"])]
    if invocation.get("cwd") is not None:
        command += ["--cwd", str(invocation["cwd"])]
    if invocation.get("allow_missing_lineinfo"):
        command += ["--allow-missing-lineinfo"]
    if invocation.get("skip_enrich"):
        command += ["--skip-enrich"]
    for name in (
        "idle_retries",
        "idle_retry_seconds",
        "inactivity_timeout",
        "absolute_timeout",
    ):
        value = passthrough.get(name)
        if value is not None:
            command += [f"--{name.replace('_', '-')}", str(value)]
    if passthrough.get("gpu_lock_file") is not None:
        command += ["--gpu-lock-file", str(passthrough["gpu_lock_file"])]
    if passthrough.get("run_id"):
        command += ["--run-id", str(passthrough["run_id"])]
    if passthrough.get("resume_closed_raw"):
        command += ["--resume-closed-raw"]
    return command


def _run(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(
        description="Generic capture-case adapter (pure invocation -> hoppersim capture)"
    )
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invocation-json", type=Path, default=None)
    parser.add_argument("--idle-retries", type=int, default=None)
    parser.add_argument("--idle-retry-seconds", type=float, default=None)
    parser.add_argument("--inactivity-timeout", type=float, default=None)
    parser.add_argument("--absolute-timeout", type=float, default=None)
    parser.add_argument("--gpu-lock-file", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--resume-closed-raw",
        action="store_true",
        help="reuse a closed raw directory; do not replace it and do not retrace",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the invocation and print the composed capture command "
        "without running anything",
    )
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    if output.exists() and not args.resume_closed_raw:
        raise AdapterError(
            f"refusing to replace existing capture directory: {output}"
        )
    invocation = load_invocation(
        args.suite_manifest.expanduser().resolve(),
        args.case_id,
        args.invocation_json.expanduser().resolve()
        if args.invocation_json is not None
        else None,
    )
    with tempfile.TemporaryDirectory(prefix="capture_case_") as tmp_name:
        tmp = Path(tmp_name)
        argv_json = tmp / "argv.json"
        argv_json.write_text(
            json.dumps(list(invocation["argv"])), encoding="utf-8"
        )
        env_json = tmp / "env.json"
        env_json.write_text(
            json.dumps(dict(invocation["env"])), encoding="utf-8"
        )
        metadata_out = tmp / "capture_metadata_raw.json"
        command = compose_capture_command(
            invocation,
            case_id=args.case_id,
            output=output,
            argv_json=argv_json,
            env_json=env_json,
            metadata_out=metadata_out,
            passthrough={
                "idle_retries": args.idle_retries,
                "idle_retry_seconds": args.idle_retry_seconds,
                "inactivity_timeout": args.inactivity_timeout,
                "absolute_timeout": args.absolute_timeout,
                "gpu_lock_file": args.gpu_lock_file,
                "run_id": args.run_id,
                "resume_closed_raw": args.resume_closed_raw,
            },
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "case_id": args.case_id,
                        "invocation_valid": True,
                        "capture_command": command,
                        "output": str(output),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        prep_scratch = tmp
        prep_records = []
        if args.resume_closed_raw:
            qualification_path = output / "qualification.json"
            if qualification_path.is_file():
                try:
                    previous = json.loads(qualification_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise AdapterError(f"cannot read resumed prep evidence: {exc}") from exc
                if not isinstance(previous, dict):
                    raise AdapterError("resumed qualification must be a JSON object")
                prep_records = previous.get("prep", [])
            if not isinstance(prep_records, list):
                raise AdapterError("resumed prep evidence must be a list")
            if invocation.get("prep") and not prep_records:
                raise AdapterError("resumed raw is missing original prep evidence")
            for record in prep_records:
                if not isinstance(record, dict) or record.get("returncode") != 0:
                    raise AdapterError("resumed raw lacks successful original prep evidence")
                log = record.get("log")
                if not isinstance(log, str) or Path(log).parts[:1] != ("prep",) or len(Path(log).parts) != 2:
                    raise AdapterError("resumed prep log must refer to the original prep directory")
                _require_file(log, anchor=output, label="resumed prep log")
        else:
            if invocation.get("prep"):
                prep_scratch = Path(tempfile.mkdtemp(
                    prefix=f".{output.name}.prep-", dir=output.parent
                ))
            prep_records = run_prep(invocation, prep_scratch)
        capture_env = os.environ.copy()
        inherited_pythonpath = capture_env.get("PYTHONPATH", "")
        capture_env["PYTHONPATH"] = str(PYTHON_ROOT) + (
            os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
        )
        capture_log = output.parent / f".{output.name}.capture.log"
        log_handle = (
            tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8",
                prefix=f".{output.name}.resume-", suffix=".capture.log",
                dir=output.parent, delete=False,
            ) if args.resume_closed_raw else capture_log.open("w", encoding="utf-8")
        )
        with log_handle as handle:
            capture_log = Path(handle.name)
            completed = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=capture_env,
            )
            returncode = completed.wait()
        captured_text = capture_log.read_text(encoding="utf-8", errors="replace")
        if returncode != 0:
            sys.stderr.write(captured_text[-4000:])
            return returncode
        if prep_records and not args.resume_closed_raw:
            prep_dir = output / "prep"
            prep_dir.mkdir()
            for record in prep_records:
                source = prep_scratch / Path(record["log"]).name
                shutil.copy2(source, prep_dir / source.name)
        qualification = write_case_documents(
            output=output,
            case_id=args.case_id,
            invocation=invocation,
            capture_returncode=completed.returncode,
        )
        if prep_records:
            qualification["prep"] = prep_records
            (output / "qualification.json").write_text(
                json.dumps(qualification, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "status": "captured",
                    "case_id": args.case_id,
                    "output": str(output),
                    "launches": len(qualification["launches"]),
                    "capture_metadata": str(output / "capture_metadata.json"),
                    "qualification": str(output / "qualification.json"),
                },
                sort_keys=True,
            )
        )
        if prep_records and not args.resume_closed_raw:
            shutil.rmtree(prep_scratch)
        return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except AdapterError as exc:
        print(f"capture_case: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as exc:
        print(f"capture_case: {exc}", file=sys.stderr)
        raise SystemExit(3)
