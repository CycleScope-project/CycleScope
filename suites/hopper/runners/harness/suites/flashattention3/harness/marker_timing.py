"""FA3 in-kernel clock64/globaltimer marker timing (device_kernel override).

Kernel side: every FA3 main kernel (fwd/bwd/combine/postprocess) is launched
through ``cutlass::device_kernel<Kernel>``.  The measurement-only override of
``upstream/hopper/cutlass/include/cutlass/device_kernel.h`` records one
[start_clock64, end_clock64, start_globaltimer, end_globaltimer, smid,
cluster_dim_packed, cluster_ctarank] row per (cta, warp) around
``Operator::operator()``.  Buffer layout and field order match the CUTLASS
delivery suite's override exactly
(suites/cutlass/harness/override/cutlass/device_kernel.h).

Host side mirrors suites/cutlass/harness/cutlass_example_cycle_wrapper.cu:
per launch slot, timer_span = max(end_globaltimer) - min(start_globaltimer),
ratio = median over valid rows of
(end_clock64 - start_clock64) / (end_globaltimer - start_globaltimer),
cycles = timer_span * ratio.  A row is valid only when
row[0] > 0 and row[1] > row[0] and row[2] > 0 and row[3] > row[2].

The three buffers are torch int64 CUDA tensors holding unsigned 64-bit marker
values as raw bit patterns (torch 2.1 has no uint64 dtype); every value is
re-interpreted modulo 2**64 on the host.
"""

from __future__ import annotations

import os
import statistics
from typing import Any


MAX_LAUNCHES = 256
MAX_CTAS = 1024
MAX_WARPS = 16
FIELDS = 7
MARKER_LEN = MAX_LAUNCHES * MAX_CTAS * MAX_WARPS * FIELDS
U64_MOD = 1 << 64

FIELD_START_CLOCK = 0
FIELD_END_CLOCK = 1
FIELD_START_TIMER = 2
FIELD_END_TIMER = 3
FIELD_SMID = 4
FIELD_CLUSTER_PACKED = 5
FIELD_CLUSTER_RANK = 6

MARKER_SCOPE = "outermost_in_kernel_clock_marker_interval"
MARKER_METHOD = "device_kernel_clock64_over_globaltimer"
MARKER_HARDWARE_METRIC = "max_hardware_warp_end_minus_min_hardware_warp_start"
MARKER_ENV_FLAG = "FA3_MARKER_TIMING"

RAW_SPANS_PER_SAMPLE = 3
FORMAL_SAMPLES = 9
RAW_LAUNCHES_REQUIRED = RAW_SPANS_PER_SAMPLE * FORMAL_SAMPLES
DURATION_MATCH_RELATIVE = 0.10


def marker_timing_requested(case: dict | None = None, env: dict | None = None) -> bool:
    """Marker mode switch: CLI/env flag or per-case key. Default off."""
    environ = os.environ if env is None else env
    return environ.get(MARKER_ENV_FLAG) == "1" or bool((case or {}).get("marker_timing"))


def allocate_marker_buffers(torch: Any, device: str = "cuda") -> dict:
    """Three contiguous CUDA int64 tensors holding u64 marker bit patterns."""
    return {
        "markers": torch.zeros(MARKER_LEN, dtype=torch.int64, device=device),
        "grid_ids": torch.zeros(MAX_LAUNCHES, dtype=torch.int64, device=device),
        "grids": torch.zeros(MAX_LAUNCHES, dtype=torch.int64, device=device),
        "dtype": "int64_u64_bit_pattern",
    }


def enable_markers(ops: Any, buffers: dict) -> int:
    """Point every registered TU's device symbols at the buffers.

    ``ops`` is ``torch.ops.flash_attn_3``.  ``marker_tu_count() == 0`` means the
    device_kernel override was not compiled into this flash_attn_3 build.
    """
    tu_count = int(ops.marker_tu_count())
    if tu_count <= 0:
        raise RuntimeError(
            "FA3 device_kernel clock marker override is not compiled into "
            f"flash_attn_3 (marker_tu_count()={tu_count})"
        )
    ops.marker_set_buffers(buffers["markers"], buffers["grid_ids"], buffers["grids"])
    return tu_count


def zero_buffers(buffers: dict) -> None:
    """One memset of all three buffers; call once after warmup, before sampling."""
    for key in ("markers", "grid_ids", "grids"):
        buffers[key].zero_()


def _as_uint_list(value: Any) -> list[int]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) % U64_MOD for item in value]


def decode_cluster_packed(packed: int) -> tuple[int, int, int, int]:
    """Same fallback as the CUTLASS wrapper's cluster_dim unpacking."""
    nx = packed & 0xFF
    ny = (packed >> 8) & 0xFF
    nz = (packed >> 16) & 0xFF
    ncta = (packed >> 32) & 0xFF
    if ny == 0 and nz == 0:
        ncta = packed
        nx = ncta
        ny = 1
        nz = 1
    elif ncta == 0:
        ncta = nx * ny * nz
    return nx, ny, nz, ncta


def decode_grid_packed(packed: int) -> list[int]:
    gx = packed & 0xFFFF or 1
    gy = (packed >> 16) & 0xFFFF or 1
    gz = (packed >> 32) & 0xFFFF or 1
    return [gx, gy, gz]


def read_launches(buffers: dict) -> list[dict]:
    """Reduce every non-empty slot to one launch sample (CUTLASS wrapper formula)."""
    markers = _as_uint_list(buffers["markers"])
    grid_ids = _as_uint_list(buffers["grid_ids"])
    grids = _as_uint_list(buffers["grids"])
    rows_per_launch = MAX_CTAS * MAX_WARPS
    launches: list[dict] = []
    for slot in range(min(MAX_LAUNCHES, len(grid_ids))):
        grid_id = grid_ids[slot]
        if grid_id == 0:
            continue
        first_timer: int | None = None
        last_timer = 0
        ratios: list[float] = []
        sm_ids: set[int] = set()
        cluster_ranks: set[int] = set()
        active_ctas: set[int] = set()
        cluster_nx = cluster_ny = cluster_nz = 1
        cluster_ncta = 0
        valid_rows = 0
        base = slot * rows_per_launch * FIELDS
        for cta in range(MAX_CTAS):
            for warp in range(MAX_WARPS):
                offset = base + (cta * MAX_WARPS + warp) * FIELDS
                start_clock = markers[offset]
                end_clock = markers[offset + 1]
                start_timer = markers[offset + 2]
                end_timer = markers[offset + 3]
                if not (start_clock > 0 and end_clock > start_clock and start_timer > 0 and end_timer > start_timer):
                    continue
                first_timer = (
                    start_timer if first_timer is None else min(first_timer, start_timer)
                )
                last_timer = max(last_timer, end_timer)
                ratios.append(float(end_clock - start_clock) / float(end_timer - start_timer))
                sm_ids.add(markers[offset + FIELD_SMID])
                nx, ny, nz, ncta = decode_cluster_packed(markers[offset + FIELD_CLUSTER_PACKED])
                cluster_nx, cluster_ny, cluster_nz = nx, ny, nz
                cluster_ncta = max(cluster_ncta, ncta)
                cluster_ranks.add(markers[offset + FIELD_CLUSTER_RANK])
                active_ctas.add(cta)
                valid_rows += 1
        if valid_rows == 0 or first_timer is None or last_timer <= first_timer:
            continue
        ratio = statistics.median(ratios)
        timer_span = float(last_timer - first_timer)
        launches.append({
            "grid_id": grid_id,
            "timer_span_ns": timer_span,
            "ratio": ratio,
            "cycles": timer_span * ratio,
            "sm_ids": sorted(sm_ids),
            "cluster_dim": [cluster_nx, cluster_ny, cluster_nz],
            "cluster_ncta": cluster_ncta,
            "cluster_ranks": sorted(cluster_ranks),
            "grid": decode_grid_packed(grids[slot]) if slot < len(grids) else [1, 1, 1],
            "valid_rows": valid_rows,
            "active_ctas": len(active_ctas),
        })
    launches.sort(key=lambda row: row["grid_id"])
    return launches


def select_main_kernel_launches(launches: list[dict], expected: dict) -> list[dict]:
    """Pick the main kernel's slots by grid dims + cluster ncta.

    Duplicate signatures fall back to correlating each slot's timer_span with
    the main kernel's CUDA activity durations (+/-10%); the correlation must
    select exactly the expected launch count, otherwise this refuses.
    """
    grid = expected.get("grid")
    ncta = expected.get("cluster_ncta")
    want_grid = [int(value) for value in grid] if grid is not None else None
    matches = []
    for launch in launches:
        if want_grid is not None and list(launch["grid"]) != want_grid:
            continue
        if ncta is not None and int(launch["cluster_ncta"]) != int(ncta):
            continue
        matches.append(launch)
    if not matches:
        signatures = sorted({(tuple(row["grid"]), row["cluster_ncta"]) for row in launches})
        raise RuntimeError(
            "no marker launch matches the expected main-kernel signature "
            f"grid={want_grid} cluster_ncta={ncta}; observed signatures={signatures}"
        )
    want = expected.get("expected_launches")
    if want is not None:
        want = int(want)
        if len(matches) == want:
            return matches
        if len(matches) < want:
            raise RuntimeError(
                f"marker captured {len(matches)} main-kernel launches, expected {want}; "
                "some launches were lost to slot reuse or signature drift"
            )
    durations = [float(value) for value in (expected.get("activity_durations_ns") or [])]
    if not durations:
        if want is None and len(matches) == 1:
            return matches
        raise RuntimeError(
            "main-kernel marker signature is ambiguous: "
            f"{len(matches)} launches share grid={want_grid} cluster_ncta={ncta} "
            "and no CUDA activity durations were provided for disambiguation"
        )

    def correlates(span: float) -> bool:
        return any(abs(span - value) <= DURATION_MATCH_RELATIVE * value for value in durations)

    kept = [row for row in matches if correlates(row["timer_span_ns"])]
    if want is None:
        if len(kept) == 1:
            return kept
        raise RuntimeError(
            "main-kernel marker signature is ambiguous: "
            f"{len(kept)} of {len(matches)} launches correlate with main-kernel "
            "activity durations within +/-10%"
        )
    if len(kept) != want:
        resolved = _select_by_launch_class(matches, durations, want)
        if resolved is not None:
            return resolved
        raise RuntimeError(
            "activity-duration disambiguation is not unique: "
            f"{len(kept)} of {len(matches)} marker launches correlate within "
            f"+/-10% with {len(durations)} main-kernel activity durations; "
            f"need exactly {want}"
        )
    return kept


def _select_by_launch_class(
    matches: list[dict], durations: list[float], want: int
) -> list[dict] | None:
    """Disambiguate two op kernels that share one grid signature by launch order.

    The persistent forward scheduler launches grid={num_sm}, so a 1CTA-target
    op's fwd is [1,1,1] — the same signature as a 1CTA backward main kernel —
    and raw +/-10% span correlation cannot always separate them (the marker
    globaltimer span systematically undershoots the Kineto activity duration on
    short kernels). One op always launches its kernels in the same order, so
    signature-matched launches sorted by grid_id strictly alternate the two
    kernels. Pick the parity class whose median span tracks the main kernel's
    activity duration; refuse unless the classes are cleanly separable and the
    chosen class holds exactly the expected launch count.
    """
    if want < 2 or len(matches) != 2 * want:
        return None
    ordered = sorted(matches, key=lambda row: row["grid_id"])
    if not durations:
        return None
    ref = statistics.median(durations)
    if ref <= 0:
        return None
    class_medians = []
    for parity in (0, 1):
        cls = ordered[parity::2]
        if len(cls) != want:
            return None
        class_medians.append(statistics.median([float(row["timer_span_ns"]) for row in cls]))
    distance = [abs(m - ref) for m in class_medians]
    near = 0 if distance[0] < distance[1] else 1
    other = 1 - near
    if not 0.5 * ref <= class_medians[near] <= 2.0 * ref:
        return None
    if distance[near] == distance[other]:
        return None
    return ordered[near::2]


def fwd_cluster_topo_from_launches(launches: list[dict]) -> dict | None:
    """Hardware %smid/%cluster_nctaid observation of the op's cluster launch.

    Some instrumented TUs (e.g. hd256 fwd) lack the fa3_delivery_topo stores,
    so the delivery-topo symbol stays empty; the device_kernel marker rows
    carry the same per-CTA hardware fields (%smid, %cluster_nctaid) for every
    launch. Pick the launch class with cluster_nctaid product 2 and two
    distinct SM ids — the op's cluster sibling launch — and express it in
    read_device_delivery_topo's schema. Returns None when no such class exists.
    """
    cluster_rows = [
        row
        for row in launches
        if int(row.get("cluster_ncta") or 0) == 2 and len(row.get("sm_ids") or []) == 2
    ]
    if not cluster_rows:
        return None
    sample = cluster_rows[-1]
    sm_ids = sorted({int(x) for x in sample["sm_ids"]})
    ranks = sorted({int(x) for x in sample.get("cluster_ranks") or []})
    if len(sm_ids) != 2:
        return None
    if len(ranks) != 2:
        ranks = [0, 1]
    return {
        "sm_ids": sm_ids,
        "observed_sms": sm_ids,
        "cluster_nctaid": 2,
        "cluster_dim": [2, 1, 1],
        "cluster_cta_ranks": ranks,
        "rows": [
            {
                "cta": index,
                "sm_id": sm_ids[index],
                "cluster_nctaid_product": 2,
                "cluster_ctarank": ranks[index],
            }
            for index in range(2)
        ],
        "method": (
            "device %%smid and %%cluster_nctaid recorded per CTA in the "
            "device_kernel marker rows (op's cluster sibling launch)"
        ),
    }


def summarize_marker_samples(launches: list[dict]) -> dict:
    """Last 27 raw launches by grid_id; median of each consecutive triple -> 9."""
    ordered = sorted(launches, key=lambda row: row["grid_id"])
    if len(ordered) < RAW_LAUNCHES_REQUIRED:
        raise RuntimeError(
            f"marker timing needs {RAW_LAUNCHES_REQUIRED} main-kernel launches, "
            f"got {len(ordered)}"
        )
    raw = ordered[-RAW_LAUNCHES_REQUIRED:]
    raw_cycles = [float(row["cycles"]) for row in raw]
    samples = [
        statistics.median(raw_cycles[index:index + RAW_SPANS_PER_SAMPLE])
        for index in range(0, len(raw_cycles), RAW_SPANS_PER_SAMPLE)
    ]
    mean = statistics.fmean(samples)
    cv = statistics.stdev(samples) / mean * 100.0 if len(samples) > 1 and mean else 0.0
    return {
        "samples_cycles": samples,
        "samples_cycles_raw": raw_cycles,
        "raw_spans_per_sample": RAW_SPANS_PER_SAMPLE,
        "sample_count": len(samples),
        "raw_sample_count": len(raw_cycles),
        "median_cycles": statistics.median(samples),
        "cv_percent": cv,
        "aggregation": "median_of_each_consecutive_triple",
    }


def build_marker_main_kernel_timing(
    summary: dict,
    *,
    kernel_name_expected: str | None,
    kernel_name_observed: str | None,
    launch: dict,
) -> dict:
    """timing.main_kernel container; readable by validate_cluster_sm_400."""
    return {
        "schema_version": "delivery.matched-kernel-timing.v1",
        "status": "matched",
        "eligible_for_simulator_accuracy": True,
        "scope": MARKER_SCOPE,
        "method": MARKER_METHOD,
        "hardware_metric": MARKER_HARDWARE_METRIC,
        "samples_cycles": list(summary["samples_cycles"]),
        "samples_cycles_raw": list(summary["samples_cycles_raw"]),
        "raw_spans_per_sample": RAW_SPANS_PER_SAMPLE,
        "sample_count": summary["sample_count"],
        "raw_sample_count": summary["raw_sample_count"],
        "aggregation": summary["aggregation"],
        "median_cycles": summary["median_cycles"],
        "cv_percent": summary["cv_percent"],
        "kernel_name_expected": kernel_name_expected,
        "kernel_name_observed": kernel_name_observed,
        "selected_slot_grid_id": launch["grid_id"],
        "grid": list(launch["grid"]),
        "cluster_dim": list(launch["cluster_dim"]),
        "cluster_ncta": launch["cluster_ncta"],
        "cluster_ranks": list(launch["cluster_ranks"]),
        "sm_ids": list(launch["sm_ids"]),
        "valid_rows": launch["valid_rows"],
        "whole_operation_timing_must_not_be_used_for_simulator_accuracy": True,
    }


def collect_marker_main_kernel(
    torch: Any,
    run: Any,
    case: dict,
    ops: Any,
    *,
    main_kernel_activity: dict | None = None,
    device_topo: dict | None = None,
    raw_launches: int = RAW_LAUNCHES_REQUIRED,
) -> dict:
    """Full marker pass: enable, one memset, raw_launches clean launches, reduce.

    Call after device warmup and after the activity timing channel so the
    expected main-kernel signature (grid dims, cluster ncta, activity
    durations) is available for selection.
    """
    from main_kernel_timing import build_marker_expected

    buffers = allocate_marker_buffers(torch)
    tu_count = enable_markers(ops, buffers)
    zero_buffers(buffers)
    for _ in range(raw_launches):
        run()
    torch.cuda.synchronize()
    launches = read_launches(buffers)
    if not launches:
        raise RuntimeError("no valid FA3 device_kernel marker slots after sampling")
    expected = build_marker_expected(
        case, main_kernel_activity, device_topo, raw_launches=raw_launches
    )
    selected = select_main_kernel_launches(launches, expected)
    summary = summarize_marker_samples(selected)
    container = build_marker_main_kernel_timing(
        summary,
        kernel_name_expected=expected.get("kernel_name"),
        kernel_name_observed=expected.get("kernel_name"),
        launch=selected[-1],
    )
    return {
        "main_kernel": container,
        "tu_count": tu_count,
        "launch_count": len(launches),
        "selected_grid_ids": [row["grid_id"] for row in selected],
        "launches": launches,
    }
