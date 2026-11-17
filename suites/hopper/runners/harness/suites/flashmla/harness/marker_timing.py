"""FlashMLA in-kernel clock64/globaltimer marker timing (sm90 dense decode).

Kernel side: the delivery build (FLASHMLA_DELIVERY_MARKER_TIMING, see
harness/override/flashmla_marker.h) instruments the three kernels launched by one
`flash_mla_with_kvcache` operation — flash_fwd_splitkv_mla_kernel (main),
flash_fwd_mla_combine_kernel (combine) and get_mla_metadata_kernel (sched meta) — with one
[start_clock64, end_clock64, start_globaltimer, end_globaltimer, smid, cluster_dim_packed,
cluster_ctarank] row per (cta, warp) around each kernel's outermost body.  Field order and
validity rule match the FA3/CUTLASS device_kernel override exactly
(suites/cutlass/harness/override/cutlass/device_kernel.h).

Host side mirrors suites/flashattention3/harness/marker_timing.py: per launch,
timer_span = max(end_globaltimer) - min(start_globaltimer) over valid rows,
ratio = median over valid rows of
(end_clock64 - start_clock64) / (end_globaltimer - start_globaltimer),
cycles = timer_span * ratio.  A row is valid only when
row[0] > 0 and row[1] > row[0] and row[2] > 0 and row[3] > row[2].
The formal protocol records 27 raw main-kernel launches (RAW_SPANS_PER_SAMPLE=3 x
FORMAL_SAMPLES=9) and reduces them to 9 samples by the median of each consecutive triple.
The formal denominator is the MAIN region only; the combine and sched-meta regions are
recorded as secondary evidence.

Unlike FA3's slotted device_kernel buffer, each raw launch is read back individually
(zero buffers -> one operation -> synchronize -> read), so no launch-slot machinery is
needed: every read belongs to exactly one launch.  The three region buffers are torch
int64 CUDA tensors holding unsigned 64-bit marker values as raw bit patterns (torch has no
uint64 dtype); every value is re-interpreted modulo 2**64 on the host.

The whole-operation globaltimer span must never be used as the marker denominator
(docs/simulator.md §2.5; scripts/install/build_cluster_sm_296_hardware_rows.py):
scope stays `outermost_in_kernel_clock_marker_interval`,
method stays `device_kernel_clock64_over_globaltimer`.
"""

from __future__ import annotations

import ctypes
import os
import statistics
from pathlib import Path
from typing import Any, Callable


MAX_ROWS = 1 << 18
FIELDS = 7
MARKER_LEN = MAX_ROWS * FIELDS
U64_MOD = 1 << 64

FIELD_START_CLOCK = 0
FIELD_END_CLOCK = 1
FIELD_START_TIMER = 2
FIELD_END_TIMER = 3
FIELD_SMID = 4
FIELD_CLUSTER_PACKED = 5
FIELD_CLUSTER_RANK = 6

MAIN_REGION = "main"
COMBINE_REGION = "combine"
SCHED_REGION = "sched_meta"
REGIONS = (MAIN_REGION, COMBINE_REGION, SCHED_REGION)

MAIN_KERNEL_NAME = "flash_fwd_splitkv_mla_kernel"
COMBINE_KERNEL_NAME = "flash_fwd_mla_combine_kernel"
SCHED_KERNEL_NAME = "get_mla_metadata_kernel"

MARKER_SCOPE = "outermost_in_kernel_clock_marker_interval"
MARKER_METHOD = "device_kernel_clock64_over_globaltimer"
MARKER_HARDWARE_METRIC = "max_hardware_warp_end_minus_min_hardware_warp_start"
MARKER_ENV_FLAG = "FLASHMLA_MARKER_TIMING"

RAW_SPANS_PER_SAMPLE = 3
FORMAL_SAMPLES = 9
RAW_LAUNCHES_REQUIRED = RAW_SPANS_PER_SAMPLE * FORMAL_SAMPLES
DURATION_MATCH_RELATIVE = 0.10


def marker_timing_requested(case: dict | None = None, env: dict | None = None) -> bool:
    """Marker mode switch: env flag or per-case key. Default off."""
    environ = os.environ if env is None else env
    return environ.get(MARKER_ENV_FLAG) == "1" or bool((case or {}).get("marker_timing"))


def locate_extension(flash_mla_cuda: Any) -> Path:
    """Path of the built flash_mla/cuda.so (the setter is compiled into it)."""
    return Path(flash_mla_cuda.__file__).resolve()


class MarkerChannel:
    """Buffers + ctypes setter bridge to the instrumented flash_mla/cuda.so."""

    def __init__(self, torch: Any, extension: Path):
        if not extension.is_file():
            raise RuntimeError(f"flash_mla extension not found: {extension}")
        self.torch = torch
        self.lib = ctypes.CDLL(str(extension))
        try:
            setter = self.lib.flash_mla_delivery_marker_set
            tu_count_fn = self.lib.flash_mla_delivery_marker_tu_count
        except AttributeError as error:
            raise RuntimeError(
                f"{extension} does not export flash_mla_delivery_marker_set: it was built "
                "without FLASHMLA_DELIVERY_MARKER_TIMING; rebuild via "
                "scripts/build_cluster_sm.sh (delivery builds export the flag)"
            ) from error
        setter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        setter.restype = None
        tu_count_fn.argtypes = []
        tu_count_fn.restype = ctypes.c_int
        self.tu_count = int(tu_count_fn())
        if self.tu_count <= 0:
            raise RuntimeError(
                "flash_mla/cuda.so carries no marker TU registrations "
                f"(flash_mla_delivery_marker_tu_count()={self.tu_count}); expected 4 nvcc TUs"
            )
        self.buffers = {
            region: torch.zeros(MARKER_LEN, dtype=torch.int64, device="cuda")
            for region in REGIONS
        }
        setter(
            ctypes.c_void_p(self.buffers[MAIN_REGION].data_ptr()),
            ctypes.c_void_p(self.buffers[COMBINE_REGION].data_ptr()),
            ctypes.c_void_p(self.buffers[SCHED_REGION].data_ptr()),
        )

    def zero_buffers(self) -> None:
        """One memset of all three region buffers; call before each raw launch."""
        for region in REGIONS:
            self.buffers[region].zero_()

    def read_region_rows(self, region: str) -> list[list[int]]:
        """Valid 7-field rows of one region (host rule, ported from FA3)."""
        table = self.buffers[region].view(MAX_ROWS, FIELDS)
        start_clock = table[:, FIELD_START_CLOCK]
        end_clock = table[:, FIELD_END_CLOCK]
        start_timer = table[:, FIELD_START_TIMER]
        end_timer = table[:, FIELD_END_TIMER]
        valid = (
            (start_clock > 0) & (end_clock > start_clock)
            & (start_timer > 0) & (end_timer > start_timer)
        )
        rows = table[valid].cpu().tolist()
        return [[int(value) % U64_MOD for value in row] for row in rows]


def decode_cluster_packed(packed: int) -> tuple[int, int, int, int]:
    """Same field packing as the CUTLASS/FA3 device_kernel marker rows."""
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


def reduce_region_rows(rows: list[list[int]]) -> dict | None:
    """One launch sample for one region (CUTLASS wrapper formula; FA3 read_launches).

    Applies the FA3 row validity rule itself: a row counts only when
    row[0] > 0 and row[1] > row[0] and row[2] > 0 and row[3] > row[2]."""
    rows = [
        row for row in rows
        if row[FIELD_START_CLOCK] > 0
        and row[FIELD_END_CLOCK] > row[FIELD_START_CLOCK]
        and row[FIELD_START_TIMER] > 0
        and row[FIELD_END_TIMER] > row[FIELD_START_TIMER]
    ]
    if not rows:
        return None
    first_timer = min(row[FIELD_START_TIMER] for row in rows)
    last_timer = max(row[FIELD_END_TIMER] for row in rows)
    if last_timer <= first_timer:
        return None
    ratios = [
        float(row[FIELD_END_CLOCK] - row[FIELD_START_CLOCK])
        / float(row[FIELD_END_TIMER] - row[FIELD_START_TIMER])
        for row in rows
    ]
    sm_ids = sorted({row[FIELD_SMID] for row in rows})
    cluster_ranks = sorted({row[FIELD_CLUSTER_RANK] for row in rows})
    cluster_nx = cluster_ny = cluster_nz = 1
    cluster_ncta = 0
    for row in rows:
        nx, ny, nz, ncta = decode_cluster_packed(row[FIELD_CLUSTER_PACKED])
        cluster_nx, cluster_ny, cluster_nz = nx, ny, nz
        cluster_ncta = max(cluster_ncta, ncta)
    ratio = statistics.median(ratios)
    timer_span = float(last_timer - first_timer)
    return {
        "timer_span_ns": timer_span,
        "ratio": ratio,
        "cycles": timer_span * ratio,
        "sm_ids": sm_ids,
        "cluster_dim": [cluster_nx, cluster_ny, cluster_nz],
        "cluster_ncta": cluster_ncta,
        "cluster_ranks": cluster_ranks,
        "valid_rows": len(rows),
    }


def measure_main_kernel_activity_ns(torch: Any, run: Callable[[], object]) -> dict:
    """CUDA activity durations of one operation call (kineto), for the FA3-style
    duration-match guard: the marker globaltimer span must track the main kernel's
    activity duration within +/-10% (DURATION_MATCH_RELATIVE)."""
    from torch.profiler import ProfilerActivity, profile

    run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    durations: dict[str, list[float]] = {
        MAIN_KERNEL_NAME: [],
        COMBINE_KERNEL_NAME: [],
        SCHED_KERNEL_NAME: [],
    }
    for event in prof.events():
        for name in durations:
            if name in str(getattr(event, "name", "")):
                device_time_us = float(getattr(event, "device_time", 0.0) or 0.0)
                if device_time_us > 0:
                    durations[name].append(device_time_us * 1000.0)
    if not durations[MAIN_KERNEL_NAME]:
        raise RuntimeError(
            f"no CUDA activity for {MAIN_KERNEL_NAME} in the profiler window; cannot apply "
            "the marker duration-match guard"
        )
    return {name: values for name, values in durations.items() if values}


def summarize_marker_samples(launches: list[dict]) -> dict:
    """Last 27 raw launches by launch index; median of each consecutive triple -> 9."""
    ordered = list(launches)
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


def build_marker_main_kernel_timing(summary: dict, launch: dict) -> dict:
    """timing.main_kernel container; same schema as the FA3/CUTLASS suites and readable by
    the cluster-SM hardware-rows builder (scope/method are the 296 marker contract)."""
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
        "kernel_name_expected": MAIN_KERNEL_NAME,
        "kernel_name_observed": MAIN_KERNEL_NAME,
        "grid_id": launch["launch_index"],
        "cluster_dim": list(launch["cluster_dim"]),
        "cluster_ncta": launch["cluster_ncta"],
        "cluster_ranks": list(launch["cluster_ranks"]),
        "sm_ids": list(launch["sm_ids"]),
        "valid_rows": launch["valid_rows"],
        "whole_operation_timing_must_not_be_used_for_simulator_accuracy": True,
    }


def collect_marker_main_kernel(torch: Any, run: Callable[[], object], flash_mla_cuda: Any) -> dict:
    """Full marker pass: enable the setter, then per raw launch zero -> run -> sync ->
    read all three regions; reduce 27 raw main-region samples to 9 formal samples and
    apply the FA3-style duration-match guard against the main kernel's CUDA activity.

    Call after device warmup (the op's first call runs get_mla_metadata_kernel only once,
    outside the recorded launches)."""
    channel = MarkerChannel(torch, locate_extension(flash_mla_cuda))
    activity_ns = measure_main_kernel_activity_ns(torch, run)
    activity_median_ns = statistics.median(activity_ns[MAIN_KERNEL_NAME])
    raw_launches: list[dict] = []
    secondary: dict[str, list[dict | None]] = {region: [] for region in (COMBINE_REGION, SCHED_REGION)}
    for index in range(RAW_LAUNCHES_REQUIRED):
        channel.zero_buffers()
        run()
        torch.cuda.synchronize()
        main_rows = channel.read_region_rows(MAIN_REGION)
        launch = reduce_region_rows(main_rows)
        if launch is None:
            raise RuntimeError(
                f"marker launch {index}: no valid {MAIN_KERNEL_NAME} marker rows; the "
                "delivery binary is not instrumented or the pointer was not set"
            )
        launch["launch_index"] = index
        raw_launches.append(launch)
        for region in (COMBINE_REGION, SCHED_REGION):
            secondary[region].append(reduce_region_rows(channel.read_region_rows(region)))

    median_span_ns = statistics.median(row["timer_span_ns"] for row in raw_launches)
    duration_match = abs(median_span_ns - activity_median_ns) <= (
        DURATION_MATCH_RELATIVE * activity_median_ns
    )
    if not duration_match:
        raise RuntimeError(
            "marker duration-match guard failed: median marker globaltimer span "
            f"{median_span_ns:.1f} ns vs {MAIN_KERNEL_NAME} CUDA activity "
            f"{activity_median_ns:.1f} ns exceeds +/-{int(DURATION_MATCH_RELATIVE * 100)}%"
        )

    summary = summarize_marker_samples(raw_launches)
    container = build_marker_main_kernel_timing(summary, raw_launches[-1])
    secondary_reduced = {
        region: [sample for sample in samples if sample is not None]
        for region, samples in secondary.items()
    }
    secondary_summary = {
        region: {
            "kernel_name": (
                COMBINE_KERNEL_NAME if region == COMBINE_REGION else SCHED_KERNEL_NAME
            ),
            "role": "secondary_evidence_not_in_formal_denominator",
            "launch_count": len(samples),
            "median_cycles": (
                statistics.median([sample["cycles"] for sample in samples])
                if samples else None
            ),
            "median_timer_span_ns": (
                statistics.median([sample["timer_span_ns"] for sample in samples])
                if samples else None
            ),
        }
        for region, samples in secondary_reduced.items()
    }
    return {
        "main_kernel": container,
        "tu_count": channel.tu_count,
        "raw_launch_count": len(raw_launches),
        "duration_match": {
            "marker_median_span_ns": median_span_ns,
            "main_kernel_activity_median_ns": activity_median_ns,
            "relative_difference": abs(median_span_ns - activity_median_ns) / activity_median_ns,
            "allowed_relative": DURATION_MATCH_RELATIVE,
            "status": "pass",
        },
        "main_kernel_activity_ns": activity_ns,
        "regions_secondary": secondary_summary,
        "raw_launches": raw_launches,
    }
