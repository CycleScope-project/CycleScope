"""Device-globaltimer timing for a possibly multi-kernel CUDA operation.

Shared measurement timer (stage-6 consolidation): the former per-suite
copies under triton/flashmla/flashinfer harness directories were
byte-identical except for the cluster-sm mode env-var name; this copy
generalizes that check. It is measurement-only and never part of the
capture path.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import statistics
import sys
from typing import Callable


COMMON_ROOT = Path(__file__).resolve().parent
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
from matched_kernel_timing import (  # noqa: E402
    profile_matched_kernel,
    summarize_matched_samples,
)


MARKER_COUNT = 11
CALIBRATION_WAIT_CYCLES = 200_000


def coefficient_of_variation(samples: list[float]) -> float:
    mean = statistics.fmean(samples)
    return statistics.stdev(samples) / mean * 100.0 if len(samples) > 1 and mean else 0.0


def consecutive_phase_balanced_medians(samples: list[float]) -> list[float]:
    if len(samples) % 4:
        raise ValueError("raw timing sample count must be divisible by four")
    return [statistics.median(samples[index:index + 4]) for index in range(0, len(samples), 4)]


class GlobaltimerOperationTimer:
    def __init__(self, torch, library: Path):
        if not library.is_file():
            raise RuntimeError(f"missing globaltimer library: {library}")
        self.torch = torch
        self.library = library
        self.lib = ctypes.CDLL(str(library))
        for name in (
            "delivery_clock64_calibrate",
            "delivery_clock64_block",
            "delivery_clock64_start",
            "delivery_clock64_end",
            "delivery_globaltimer_allocate_flag",
            "delivery_globaltimer_free_flag",
        ):
            function = getattr(self.lib, name)
            function.restype = ctypes.c_int
        self.lib.delivery_clock64_calibrate.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p
        ]
        self.lib.delivery_clock64_block.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.delivery_clock64_start.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.delivery_clock64_end.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.delivery_globaltimer_allocate_flag.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.delivery_globaltimer_free_flag.argtypes = [ctypes.c_void_p]
        self.lib.delivery_globaltimer_reset_flag.argtypes = [ctypes.c_void_p]
        self.lib.delivery_globaltimer_reset_flag.restype = None
        self.lib.delivery_globaltimer_release_flag.argtypes = [ctypes.c_void_p]
        self.lib.delivery_globaltimer_release_flag.restype = None
        self.markers = torch.zeros(MARKER_COUNT, dtype=torch.int64, device="cuda")
        self.timed_stream = torch.cuda.Stream()
        self.host_flag = ctypes.c_void_p()
        self.device_flag = ctypes.c_void_p()
        self._check(
            self.lib.delivery_globaltimer_allocate_flag(
                ctypes.byref(self.host_flag), ctypes.byref(self.device_flag)
            ),
            "mapped release-flag allocation",
        )

    @staticmethod
    def _pointer(value: int) -> ctypes.c_void_p:
        return ctypes.c_void_p(value)

    def _check(self, status: int, operation: str) -> None:
        if status:
            raise RuntimeError(f"{operation} failed with CUDA error {status}")

    def sample(self, run: Callable[[], object]) -> tuple[object, int, dict]:
        torch = self.torch
        stream = self.timed_stream
        with torch.cuda.stream(stream):
            self.markers.zero_()
        self.lib.delivery_globaltimer_reset_flag(self.host_flag)
        stream.synchronize()

        marker_pointer = self._pointer(self.markers.data_ptr())
        flag_pointer = self.device_flag
        stream_pointer = self._pointer(stream.cuda_stream)

        self._check(
            self.lib.delivery_clock64_calibrate(
                marker_pointer, CALIBRATION_WAIT_CYCLES, stream_pointer
            ),
            "clock64 calibration launch",
        )
        stream.synchronize()
        self._check(
            self.lib.delivery_clock64_block(flag_pointer, marker_pointer, stream_pointer),
            "timed-stream blocker launch",
        )
        self._check(
            self.lib.delivery_clock64_start(marker_pointer, stream_pointer),
            "operation start-marker launch",
        )
        with torch.cuda.stream(stream):
            value = run()
        self._check(
            self.lib.delivery_clock64_end(marker_pointer, stream_pointer),
            "operation end-marker launch",
        )
        self.lib.delivery_globaltimer_release_flag(self.host_flag)
        stream.synchronize()

        fields = [int(value) for value in self.markers.cpu().tolist()]
        if fields[10]:
            raise RuntimeError("timed stream was not released before the five-second timeout")
        clock_delta = fields[2] - fields[0]
        calibration_timer_delta = fields[3] - fields[1]
        operation_timer_delta = fields[8] - fields[5]
        if clock_delta <= 0 or calibration_timer_delta <= 0 or operation_timer_delta <= 0:
            raise RuntimeError(f"invalid clock64/globaltimer markers: {fields}")
        clock_per_globaltimer = clock_delta / calibration_timer_delta
        return value, operation_timer_delta, {
            "calibration_start_clock64": fields[0],
            "calibration_start_globaltimer": fields[1],
            "calibration_end_clock64": fields[2],
            "calibration_end_globaltimer": fields[3],
            "operation_start_clock64": fields[4],
            "operation_start_globaltimer": fields[5],
            "operation_start_smid": fields[6],
            "operation_end_clock64": fields[7],
            "operation_end_globaltimer": fields[8],
            "operation_end_smid": fields[9],
            "clock_per_globaltimer": clock_per_globaltimer,
            "operation_globaltimer_span": operation_timer_delta,
            "direct_clock64_delta_same_sm": (
                fields[7] - fields[4] if fields[6] == fields[9] and fields[7] > fields[4] else None
            ),
        }


def measure_operation(
    torch,
    run: Callable[[], object],
    repeats: int,
    library: Path,
    comparison_contract: dict | None = None,
):
    if repeats != 9 and not any(
        os.environ.get(name) == "1"
        for name in (
            "FLASHMLA_CLUSTER_SM_MODE",
            "FLASHINFER_CLUSTER_SM_MODE",
            "TRITON_CLUSTER_SM_MODE",
            "CLUSTER_SM_TIMING_MODE",
        )
    ):
        raise ValueError("formal delivery timing requires exactly 9 recorded samples")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    timer = GlobaltimerOperationTimer(torch, library)
    value = run()
    torch.cuda.synchronize()
    cache_state = {
        # Canonical count: every execution of the measured public operation
        # that precedes the first recorded sample.  The JIT/first run, the
        # explicit warmup loop and the timer-settling samples all launch the
        # same target operation, so the first recorded sample follows seven
        # prior launches, not three.
        "policy": "warm_after_seven_operation_launches",
        "warmup_launches": 7,
        "staged_launches": {
            "jit_single_run": 1,
            "warmup_loop": 3,
            "timer_settling": 3,
        },
        "warmup_launches_source": "executed_pre_recorded_sequence",
        # Warmup and settling launches produce no per-launch SM records; the
        # marker SM IDs below belong to the measured samples only.
        "warmup_sm_ids_observed": None,
        "sm_residency_evidence": "not_observed",
    }
    for _ in range(3):
        value = run()
        torch.cuda.synchronize()
    discarded_globaltimer_samples = []
    discarded_kernel_activities = []
    for _ in range(3):
        if comparison_contract is None:
            value, globaltimer_ticks, _ = timer.sample(run)
        else:
            sample, activity = profile_matched_kernel(
                torch, lambda: timer.sample(run), comparison_contract
            )
            value, globaltimer_ticks, _ = sample
            discarded_kernel_activities.append(activity)
        discarded_globaltimer_samples.append(globaltimer_ticks)
    raw_globaltimer_samples = []
    marker_samples = []
    kernel_activities = []
    for _ in range(repeats * 4):
        if comparison_contract is None:
            value, globaltimer_ticks, markers = timer.sample(run)
        else:
            sample, activity = profile_matched_kernel(
                torch, lambda: timer.sample(run), comparison_contract
            )
            value, globaltimer_ticks, markers = sample
            kernel_activities.append(activity)
        raw_globaltimer_samples.append(globaltimer_ticks)
        marker_samples.append(markers)
    globaltimer_samples = consecutive_phase_balanced_medians(raw_globaltimer_samples)
    fixed_ratio = statistics.median(
        marker["clock_per_globaltimer"] for marker in marker_samples
    )
    raw_cycle_samples = [value * fixed_ratio for value in raw_globaltimer_samples]
    cycle_samples = consecutive_phase_balanced_medians(raw_cycle_samples)
    result = {
        "method": "queued device globaltimer whole-operation span",
        "definition": "median of each consecutive phase-balanced group of four raw globaltimer spans between device start/end markers",
        "host_dispatch_excluded": True,
        "warmup_launches": 7,
        "launches_before_each_raw_sample": [
            7 + index for index in range(repeats * 4)
        ],
        "discarded_timer_settling_samples": discarded_globaltimer_samples,
        "sample_count": repeats,
        "raw_sample_count": repeats * 4,
        "raw_launches_per_sample": 4,
        "samples_globaltimer": globaltimer_samples,
        "samples_globaltimer_raw": raw_globaltimer_samples,
        "median_globaltimer": statistics.median(globaltimer_samples),
        "globaltimer_unit": "native ticks (one nanosecond on the tested Hopper GPUs)",
        "cache_state": dict(cache_state),
        "fixed_clock_per_globaltimer_for_compatibility": fixed_ratio,
        "samples_cycles_compatibility": cycle_samples,
        "samples_cycles_compatibility_raw": raw_cycle_samples,
        "marker_samples": marker_samples,
        "median_cycles": statistics.median(cycle_samples),
        "cv_percent": coefficient_of_variation(globaltimer_samples),
    }
    if comparison_contract is not None:
        result["simulator_comparison"] = summarize_matched_samples(
            comparison_contract,
            kernel_activities,
            fixed_clock_per_nanosecond=fixed_ratio,
            discarded_activities=discarded_kernel_activities,
        )
    else:
        result["simulator_comparison"] = {
            "status": "unavailable_without_captured_kernel_contract",
            "eligible_for_simulator_accuracy": False,
            "whole_operation_timing_must_not_be_used_for_simulator_accuracy": True,
        }
    return value, result


def measure_two_phase_operation(
    torch,
    public_run: Callable[[], object],
    phase_one: Callable[[], object],
    phase_two: Callable[[], object],
    repeats: int,
    library: Path,
    comparison_contract: dict | None = None,
):
    """Measure two GPU phases separated by an unavoidable host synchronization.

    Each phase is independently bracketed on the device.  The primary sample is
    the sum of the two raw globaltimer spans, so the data-dependent host read and
    host allocation between the phases are deliberately excluded.
    """
    if repeats != 9 and not any(
        os.environ.get(name) == "1"
        for name in (
            "FLASHMLA_CLUSTER_SM_MODE",
            "FLASHINFER_CLUSTER_SM_MODE",
            "TRITON_CLUSTER_SM_MODE",
            "CLUSTER_SM_TIMING_MODE",
        )
    ):
        raise ValueError("formal delivery timing requires exactly 9 recorded samples")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    timer = GlobaltimerOperationTimer(torch, library)
    value = public_run()
    torch.cuda.synchronize()
    cache_state = {
        # Canonical count: executions of the identical public operation
        # (each launch = one full phase_one + phase_two pair) that precede
        # the first recorded sample: JIT/first run plus the explicit warmup
        # loop plus the timer-settling samples.
        "policy": "warm_after_seven_operation_launches",
        "warmup_launches": 7,
        "launch_unit": "one_complete_public_operation_each_executing_both_target_kernels_once",
        "staged_launches": {
            "jit_single_run": 1,
            "warmup_loop": 3,
            "timer_settling": 3,
        },
        "warmup_launches_source": "executed_pre_recorded_sequence",
        # Warmup and settling launches produce no per-launch SM records; the
        # marker SM IDs below belong to the measured samples only.
        "warmup_sm_ids_observed": None,
        "sm_residency_evidence": "not_observed",
    }
    for _ in range(3):
        value = public_run()
        torch.cuda.synchronize()

    def direct_sample():
        _, phase_one_ticks, phase_one_markers = timer.sample(phase_one)
        value, phase_two_ticks, phase_two_markers = timer.sample(phase_two)
        return value, phase_one_ticks + phase_two_ticks, {
            "phases": [phase_one_markers, phase_two_markers],
            "phase_globaltimer_spans": [phase_one_ticks, phase_two_ticks],
            "operation_globaltimer_span": phase_one_ticks + phase_two_ticks,
        }

    discarded_globaltimer_samples = []
    discarded_kernel_activities = []
    for _ in range(3):
        if comparison_contract is None:
            value, globaltimer_ticks, _ = direct_sample()
        else:
            sample, activity = profile_matched_kernel(
                torch, direct_sample, comparison_contract
            )
            value, globaltimer_ticks, _ = sample
            discarded_kernel_activities.append(activity)
        discarded_globaltimer_samples.append(globaltimer_ticks)
    raw_globaltimer_samples = []
    marker_samples = []
    kernel_activities = []
    for _ in range(repeats * 4):
        if comparison_contract is None:
            value, globaltimer_ticks, markers = direct_sample()
        else:
            sample, activity = profile_matched_kernel(
                torch, direct_sample, comparison_contract
            )
            value, globaltimer_ticks, markers = sample
            kernel_activities.append(activity)
        raw_globaltimer_samples.append(globaltimer_ticks)
        marker_samples.append(markers)
    globaltimer_samples = consecutive_phase_balanced_medians(raw_globaltimer_samples)
    ratios = [
        phase["clock_per_globaltimer"]
        for sample in marker_samples
        for phase in sample["phases"]
    ]
    fixed_ratio = statistics.median(ratios)
    raw_cycle_samples = [sample * fixed_ratio for sample in raw_globaltimer_samples]
    cycle_samples = consecutive_phase_balanced_medians(raw_cycle_samples)
    result = {
        "method": "two-phase queued device globaltimer GPU-work sum",
        "definition": "median of each consecutive phase-balanced group of four sums of raw device globaltimer spans for both GPU phases",
        "host_dispatch_excluded": True,
        "inter_phase_host_synchronization_excluded": True,
        "warmup_launches": 7,
        "launches_before_each_raw_sample": [
            7 + index for index in range(repeats * 4)
        ],
        "cache_state": dict(cache_state),
        "discarded_timer_settling_samples": discarded_globaltimer_samples,
        "sample_count": repeats,
        "raw_sample_count": repeats * 4,
        "raw_launches_per_sample": 4,
        "samples_globaltimer": globaltimer_samples,
        "samples_globaltimer_raw": raw_globaltimer_samples,
        "median_globaltimer": statistics.median(globaltimer_samples),
        "globaltimer_unit": "native ticks (one nanosecond on the tested Hopper GPUs)",
        "fixed_clock_per_globaltimer_for_compatibility": fixed_ratio,
        "samples_cycles_compatibility": cycle_samples,
        "samples_cycles_compatibility_raw": raw_cycle_samples,
        "marker_samples": marker_samples,
        "median_cycles": statistics.median(cycle_samples),
        "cv_percent": coefficient_of_variation(globaltimer_samples),
    }
    if comparison_contract is not None:
        result["simulator_comparison"] = summarize_matched_samples(
            comparison_contract,
            kernel_activities,
            fixed_clock_per_nanosecond=fixed_ratio,
            discarded_activities=discarded_kernel_activities,
        )
    else:
        result["simulator_comparison"] = {
            "status": "unavailable_without_captured_kernel_contract",
            "eligible_for_simulator_accuracy": False,
            "whole_operation_timing_must_not_be_used_for_simulator_accuracy": True,
        }
    return value, result
