"""Record the driver-selected launch context of the traced kernel with ncu.

The NVBit tracer records instructions, not the shared-memory carveout the
driver chose for the launch.  The simulator's measured L1 capacity policy needs
``launch__shared_mem_config_size`` for exactly the launch that was traced, so
the capture re-runs the same harness command (same warmup sequence, same
``cudaProfilerStart/Stop`` window) under ``ncu --profile-from-start off`` and
keeps the kernel at the traced launch ordinal inside that window.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Mapping, Sequence

from progress_timeout import run_with_progress

SCHEMA = "hopper.launch-context.v1"
FILE_NAME = "launch_context.json"
# Deferred stamping: capture hosts without ncu write a self-contained pending
# file instead; a later ncu-capable image replays it via
# scripts/stamp_launch_context.py.
PENDING_SCHEMA = "hopper.launch-context-pending.v1"
PENDING_FILE_NAME = "launch_context_pending.json"
PENDING_STAMPED_SUFFIX = ".stamped"
DEFER_ENV_VAR = "HOPPER_DEFER_NCU"
METRICS = (
    "launch__shared_mem_config_size",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
)
# Environment that only the NVBit / TMA-recorder pass needs.
_TRACER_ONLY = ("CUDA_INJECTION64_PATH", "LD_PRELOAD", "HOPPER_TENSOR_MAP_SIDECAR")
# JIT dump directories are indexed by finalize; the ncu rerun must not add files there.
_REDIRECTED_DUMPS = ("CUTE_DSL_DUMP_DIR",)


def find_ncu() -> Path:
    candidate = os.environ.get("HOPPER_NCU") or shutil.which("ncu")
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError("ncu not found: set HOPPER_NCU or put ncu on PATH")
    return Path(candidate)


def defer_requested() -> bool:
    """True when capture may defer a missing ncu to a pending file."""
    return os.environ.get(DEFER_ENV_VAR, "") == "1"


def missing_ncu_error() -> RuntimeError | None:
    """Why ncu is unavailable, or None when it is present.

    With HOPPER_DEFER_NCU=1 a missing ncu is reported as the
    returned error instead of raised, so the caller may write a pending file
    and finish the capture; without the flag this keeps the historical
    fail-closed behaviour.
    """

    try:
        find_ncu()
    except RuntimeError as exc:
        if not defer_requested():
            raise
        return exc
    return None


def write_launch_context_pending(
    *,
    case_dir: Path,
    suite: str,
    case_id: str,
    command: Sequence[str],
    env_delta: Mapping[str, str],
    cwd: Path | str | None,
    primary: Mapping[str, object],
    ordinal: int,
    kernel_count: int,
    reason: str,
) -> Path:
    """Write the self-contained replay record for a deferred launch-context pass.

    ``command``/``env_delta``/``cwd``/``primary``/``ordinal``/``kernel_count``
    mirror the exact ``profile_launch_context`` arguments the capture would
    have made: ``env_delta`` holds only the keys layered over the ambient
    environment, and the stamp driver replays ``os.environ | env_delta``.
    """

    payload = {
        "schema_version": PENDING_SCHEMA,
        "suite": suite,
        "case_id": case_id,
        "command": [str(part) for part in command],
        "env_delta": {str(key): str(value) for key, value in env_delta.items()},
        "cwd": None if cwd is None else str(cwd),
        "primary": dict(primary),
        "ordinal": int(ordinal),
        "kernel_count": int(kernel_count),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": str(reason),
    }
    path = case_dir / PENDING_FILE_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_launch_context_pending(path: Path) -> dict:
    """Read and validate a pending file written by ``write_launch_context_pending``."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path}: unreadable launch-context pending: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != PENDING_SCHEMA:
        raise RuntimeError(f"{path}: not a {PENDING_SCHEMA} pending file")
    command = payload.get("command")
    env_delta = payload.get("env_delta")
    primary = payload.get("primary")
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise RuntimeError(f"{path}: pending 'command' must be a non-empty argv list")
    if env_delta is not None and not isinstance(env_delta, dict):
        raise RuntimeError(f"{path}: pending 'env_delta' must be a mapping")
    if not isinstance(primary, dict) or not primary.get("kernel_name") or not primary.get("kernel_file"):
        raise RuntimeError(
            f"{path}: pending 'primary' must carry kernel_name and kernel_file"
        )
    for key in ("suite", "case_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise RuntimeError(f"{path}: pending {key!r} must be a non-empty string")
    if not isinstance(payload.get("ordinal"), int) or payload["ordinal"] < 0:
        raise RuntimeError(f"{path}: pending 'ordinal' must be a non-negative int")
    if not isinstance(payload.get("kernel_count"), int) or payload["kernel_count"] < 1:
        raise RuntimeError(f"{path}: pending 'kernel_count' must be a positive int")
    return payload


def launch_context_present(case_dir: Path) -> bool:
    """True when ``case_dir`` already carries a valid stamped launch context."""

    path = Path(case_dir) / FILE_NAME
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") == SCHEMA


def _parse_rows(csv_text: str) -> list[dict[str, str]]:
    start = csv_text.find('"ID"')
    if start < 0:
        raise RuntimeError("ncu produced no CSV table")
    rows = list(csv.reader(csv_text[start:].splitlines()))
    header = rows[0]
    data = [dict(zip(header, row)) for row in rows[1:] if len(row) == len(header) and row[0].isdigit()]
    return sorted(data, key=lambda row: int(row["ID"]))


def profile_launch_context(
    command: Sequence[str],
    *,
    case_dir: Path,
    env: Mapping[str, str],
    cwd: Path | None,
    primary: Mapping[str, object],
    ordinal: int,
    kernel_count: int,
) -> dict:
    """Run ``command`` under ncu and write ``case_dir/launch_context.json``.

    ``primary`` is the traced kernel the case simulates, ``ordinal`` its launch
    order inside the profiler window and ``kernel_count`` the number of kernels
    the tracer recorded in that window.
    """
    scratch = case_dir / "launch_context"
    scratch.mkdir(exist_ok=True)
    home = scratch / "home"
    home.mkdir(exist_ok=True)
    environment = {key: value for key, value in env.items() if key not in _TRACER_ONLY}
    for key in _REDIRECTED_DUMPS:
        if key in environment:
            environment[key] = str(scratch / "dump")
    environment["HOME"] = str(home)
    environment["XDG_CONFIG_HOME"] = str(home)
    environment["XDG_CACHE_HOME"] = str(home)
    ncu_csv = case_dir / "launch_context.ncu.csv"
    ncu_command = [
        str(find_ncu()),
        "--csv", "--page", "raw", "--print-units", "base", "--print-kernel-base", "mangled",
        "--profile-from-start", "off", "--target-processes", "all", "--clock-control", "none",
        "--metrics", ",".join(METRICS),
        "--log-file", str(ncu_csv),
        "--", *command,
    ]
    completed = run_with_progress(
        ncu_command,
        cwd=cwd,
        env=environment,
        watched_paths=[scratch, ncu_csv.parent],
        log_path=case_dir / "launch_context.run.log",
        inactivity_timeout=1200,
        absolute_timeout=21600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{case_dir.name}: ncu launch-context pass exited {completed.returncode}")
    rows = _parse_rows(ncu_csv.read_text(encoding="utf-8", errors="replace"))
    if len(rows) != kernel_count:
        raise RuntimeError(
            f"{case_dir.name}: ncu profiled {len(rows)} kernels in the profiler window, "
            f"tracer recorded {kernel_count}: {[row.get('Kernel Name') for row in rows]}"
        )
    row = rows[ordinal]
    missing = [name for name in METRICS if name not in row]
    if missing:
        raise RuntimeError(f"{case_dir.name}: ncu output lacks {missing}")
    ncu_name = row.get("Kernel Name")
    traced_name = primary["kernel_name"]
    if ncu_name != traced_name:
        raise RuntimeError(
            f"{case_dir.name}: ncu kernel {ncu_name!r} at ordinal {ordinal} "
            f"does not match traced {traced_name!r}; window={[r.get('Kernel Name') for r in rows]}"
        )
    payload = {
        "schema_version": SCHEMA,
        "kernel_name": ncu_name,
        "traced_kernel_name": traced_name,
        "traced_kernel_file": primary["kernel_file"],
        "kernel_identity_match": True,
        "operation_kernel_ordinal": ordinal,
        "profiler_window_kernel_count": len(rows),
        "profiler_window_kernel_names": [item.get("Kernel Name") for item in rows],
        "gpu_name": row.get("device__attribute_display_name") or row.get("Device"),
        "ncu": str(ncu_command[0]),
        "ncu_home": str(home),
        **{name: int(float(row[name].replace(",", ""))) for name in METRICS},
    }
    payload.update(_case_identity_evidence(case_dir))
    (case_dir / FILE_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _case_identity_evidence(case_dir: Path) -> dict:
    """Attach same-binary / same-shape evidence already on disk; never copy metrics."""
    evidence: dict = {}
    trace_result_path = case_dir / "trace_result.json"
    if trace_result_path.is_file():
        try:
            trace_result = json.loads(trace_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            trace_result = None
        case = (trace_result or {}).get("case") if isinstance(trace_result, dict) else None
        shape = (case or {}).get("shape") if isinstance(case, dict) else None
        if isinstance(shape, dict) and {"m", "n", "k"} <= set(shape):
            evidence["problem_shape"] = {
                "case_id": case.get("case_id"),
                "m": shape.get("m"),
                "n": shape.get("n"),
                "k": shape.get("k"),
            }
    metadata_path = case_dir / "capture_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict):
            binary = {
                key: metadata[key]
                for key in ("binary", "binary_sha256", "selected_sass_sha256")
                if metadata.get(key)
            }
            if binary:
                evidence["binary_identity"] = binary
    return evidence
