"""FA3 main-kernel-only timing: matched CUDA activity start/end for one kernel.

Public-operation semantics stay unchanged. This module adds a second, parallel
timing channel that measures exactly one operation kernel (the case's primary
mechanism kernel) with 9 recorded samples / 36 raw launches, using
suites/common/matched_kernel_timing.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


COMMON_ROOT = Path(__file__).resolve().parents[3] / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
from matched_kernel_timing import (  # noqa: E402
    CONTRACT_SCHEMA,
    cuda_activity_rows,
    profile_matched_kernel,
    summarize_matched_samples,
)


def build_marker_expected(
    case: dict,
    main_kernel_activity: dict | None = None,
    device_topo: dict | None = None,
    *,
    raw_launches: int = 27,
) -> dict:
    """Expected main-kernel marker signature for marker_timing.select_main_kernel_launches.

    ``main_kernel_activity`` is the primary row of one captured operation
    (trace_main_kernel_activities); ``device_topo`` is read_device_delivery_topo's
    output when this build recorded in-kernel smid/cluster stores.
    """
    activity = main_kernel_activity or {}
    grids = activity.get("grids") or []
    grid = grids[0] if grids and isinstance(grids[0], (list, tuple)) else grids
    grid = [int(value) for value in grid] if grid else None
    cluster_ncta = None
    if isinstance(device_topo, dict):
        value = device_topo.get("cluster_nctaid")
        if isinstance(value, list):
            cluster_ncta = int(value[0]) if len(value) == 1 else None
        elif value is not None:
            cluster_ncta = int(value)
    operation = str(case.get("operation") or "")
    # No fallback from the grid product: two ordinary (non-cluster) CTAs share
    # the same grid product but record cluster_ncta == 1 in the marker rows.
    cluster_ncta = 1 if operation in ("fixed_bwd", "varlen_bwd") else cluster_ncta
    # Backward never launches as a cluster (flash_bwd_launch_template.h hardcodes
    # ClusterShape 1x1x1), while a cluster-eligible fwd sibling of the same op is
    # a persistent launch whose device_topo cluster echo (grid={num_sm}) must not
    # be attributed to the bwd main kernel. Pinning 1 lets the cluster_ncta
    # filter drop the fwd's rows and select exactly the bwd launches.
    durations = []
    device_time = activity.get("duration_ns", activity.get("device_time_us"))
    if device_time is not None:
        value = float(device_time)
        durations.append(value * 1000.0 if value < 1e4 else value)
    return {
        "operation": str(case.get("operation") or ""),
        "kernel_name": str(activity.get("name") or "") or None,
        "grid": grid,
        "cluster_ncta": cluster_ncta,
        "activity_durations_ns": durations,
        "expected_launches": int(raw_launches),
    }



# operation -> main kernel matcher over a Kineto kernel name
MAIN_KERNEL_KIND = {
    "fixed_fwd": "fa3_fwd",
    "varlen_fwd": "fa3_fwd",
    "kvcache": "fa3_fwd",
    "fixed_bwd": "fa3_bwd",
    "varlen_bwd": "fa3_bwd",
    "combine": "fa3_combine",
}


def classify_kernel(name: str) -> str:
    if "FlashAttnFwdSm90" in name:
        return "fa3_fwd"
    if "FlashAttnBwdSm90" in name:
        return "fa3_bwd"
    if "FlashAttnFwdCombine" in name:
        return "fa3_combine"
    if "BwdPreprocess" in name:
        return "fa3_bwd_preprocess"
    if "BwdPostprocess" in name:
        return "fa3_bwd_postprocess"
    if "prepare_varlen" in name:
        return "fa3_prepare_varlen"
    if name.startswith("at::native") or name.startswith("void at::") or "FillFunctor" in name:
        return "pytorch_helper"
    if "globaltimer" in name.lower() or "delivery_clock" in name.lower():
        return "pytorch_helper"
    return "other"


def main_kernel_kind(operation: str) -> str:
    try:
        return MAIN_KERNEL_KIND[str(operation)]
    except KeyError:
        raise ValueError(f"unsupported operation for main-kernel timing: {operation}") from None


def select_main_kernel(activities: list[dict], operation: str) -> dict:
    """Pick the single main-kernel activity row of one captured operation."""
    kind = main_kernel_kind(operation)
    rows = [row for row in activities or [] if classify_kernel(str(row.get("name") or "")) == kind]
    if not rows:
        raise RuntimeError(f"no {kind} kernel captured for operation={operation}")
    if len(rows) > 1:
        # same symbol launched more than once inside one public op is not a
        # single-launch main kernel; refuse instead of guessing an ordinal.
        raise RuntimeError(
            f"operation={operation} captured {len(rows)} {kind} launches; "
            "main-kernel timing needs exactly one"
        )
    return rows[0]


def _main_kernel_symbol(row: dict, activities: list[dict]) -> str:
    """Full Kineto symbol of the main kernel.

    matched_kernel_timing._names_match's stem fallback matches every CUTLASS
    kernel (all share the `cutlass::device_kernel` stem), so a shortened symbol
    cannot be unique. The full symbol is exact for the main kernel, and the
    sibling rows that also match through the stem fallback are disambiguated by
    same_name_ordinal, which is computed over the rows the FA3 classifier
    assigns to the main-kernel class.
    """
    return str(row.get("name") or "")


def build_main_kernel_contract(primary: dict, activities: list[dict]) -> dict:
    """Identity contract for one complete main-kernel launch of one public op.

    same_name_ordinal / same_name_launch_count count launches inside a single
    profiler window of one public operation call, not the whole raw campaign.
    """
    name = _main_kernel_symbol(primary, activities)
    if not name:
        raise RuntimeError("main kernel has no symbol")
    main_rows = [
        row for row in activities
        if classify_kernel(str(row.get("name") or ""))
        == classify_kernel(str(primary.get("name") or ""))
    ]
    ordinal = next(
        index for index, row in enumerate(main_rows)
        if row is primary
    )
    # matched_kernel_timing also accepts sibling CUTLASS kernels through the
    # shared qualified stem, so the expected candidate count is the number of
    # rows that can match this symbol, not just the main-kernel class rows.
    from matched_kernel_timing import _names_match

    matching = [
        row for row in activities
        if _names_match(name, str(row.get("name") or ""))[0]
    ]
    same_name_ordinal = next(
        index for index, row in enumerate(matching)
        if row is primary
    )
    grids = primary.get("grids") or []
    grid = grids[0] if grids and isinstance(grids[0], (list, tuple)) else grids
    cta_count = None
    if grid:
        product = 1
        for value in grid:
            product *= int(value)
        cta_count = product
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "comparison_scope": "one_complete_cuda_kernel_launch",
        "hardware_metric": "cuda_activity_kernel_start_to_end",
        "simulation_metric": "full_kernel_cycles",
        "identity": {
            "kernel_name": name,
            "full_kernel_symbol": str(primary.get("name") or ""),
            "trace_kernel_file": None,
            "trace_launch_id": None,
            "operation_kernel_ordinal": ordinal,
            "same_name_ordinal": same_name_ordinal,
            "same_name_launch_count": len(matching),
            "cta_count": int(cta_count or primary.get("cta_count") or 0),
            "active_sm_count": 0,
            "sm_ids": [],
        },
        "ambiguity_policy": (
            "reject_unless_symbol_and_same_name_ordinal_select_one_launch; "
            "same_name_ordinal indexes the rows matched_kernel_timing accepts for "
            "this symbol (FA3 main kernel plus the sibling CUTLASS kernels that "
            "share the cutlass::device_kernel qualified stem)"
        ),
    }
    encoded = json.dumps(
        {key: value for key, value in payload.items() if key != "contract_digest"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    import hashlib

    payload["contract_digest"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _grid_rows(torch: Any, run: Callable[[], object]) -> list[dict]:
    """Grid/block dimensions of one operation's kernels (Kineto chrome trace)."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run()
        torch.cuda.synchronize()
    import tempfile

    with tempfile.TemporaryDirectory(prefix="fa3-main-kernel-grid-") as folder:
        path = Path(folder) / "trace.json"
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    rows: list[dict] = []
    for event in events:
        if event.get("cat") != "kernel" or float(event.get("dur", 0)) <= 0:
            continue
        args = event.get("args") or {}
        rows.append({
            "name": str(event.get("name") or ""),
            "device_time_us": float(event.get("dur", 0)),
            "grids": [args["grid"]] if args.get("grid") is not None else [],
            "blocks": [args["block"]] if args.get("block") is not None else [],
        })
    return rows


def trace_main_kernel_activities(torch, run: Callable[[], object]) -> list[dict]:
    """One operation's kernels: identity rows plus grid rows where available."""
    identity = cuda_activity_rows(
        torch,
        _profile(torch, [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], run),
    )
    grids = _grid_rows(torch, run)
    by_name: dict[str, dict] = {}
    for row in grids:
        by_name.setdefault(row["name"], row)
    merged = []
    for row in identity:
        extra = by_name.get(row["name"]) or {}
        merged.append({
            "name": row["name"],
            "duration_ns": row["duration_ns"],
            "device_time_us": extra.get("device_time_us", row["duration_ns"] / 1000.0),
            "grids": extra.get("grids") or [],
            "blocks": extra.get("blocks") or [],
        })
    return merged


def _profile(torch: Any, activities, run: Callable[[], object]):
    with torch.profiler.profile(activities=activities, record_shapes=False) as profiler:
        run()
        torch.cuda.synchronize()
    return profiler


def measure_main_kernel(
    torch,
    run: Callable[[], object],
    operation: str,
    *,
    repeats: int = 9,
) -> dict:
    """Recorded samples (4 raw launches each) of the main kernel only."""
    if repeats < 1:
        raise ValueError("main-kernel timing requires at least one recorded sample")
    activities = trace_main_kernel_activities(torch, run)
    primary = select_main_kernel(activities, operation)
    contract = build_main_kernel_contract(primary, activities)

    raw: list[dict] = []
    discarded: list[dict] = []
    for index in range(repeats * 4 + 3):
        _, activity = profile_matched_kernel(torch, run, contract)
        if index < 3:
            discarded.append(activity)
        else:
            raw.append(activity)
    return {
        "contract": contract,
        "primary": primary,
        "raw_activities": raw,
        "discarded_activities": discarded,
    }


def measure_main_kernel_with_ratio(
    torch,
    run: Callable[[], object],
    operation: str,
    *,
    fixed_clock_per_globaltimer: float,
    repeats: int = 9,
) -> dict:
    """Same as measure_main_kernel, converted to GPU clock cycles."""
    measured = measure_main_kernel(torch, run, operation, repeats=repeats)
    summary = summarize_matched_samples(
        measured["contract"],
        measured["raw_activities"],
        fixed_clock_per_nanosecond=float(fixed_clock_per_globaltimer),
        discarded_activities=measured["discarded_activities"],
    )
    return {
        "main_kernel": {
            "kind": main_kernel_kind(operation),
            "name": measured["primary"]["name"],
            "grid": measured["primary"].get("grids") or None,
            "blocks": measured["primary"].get("blocks") or None,
            "device_time_us": measured["primary"].get("device_time_us"),
            "same_name_ordinal": (measured["contract"]["identity"] or {}).get("same_name_ordinal"),
            "same_name_launch_count": (measured["contract"]["identity"] or {}).get(
                "same_name_launch_count"
            ),
        },
        "contract": measured["contract"],
        "timing": summary,
    }
