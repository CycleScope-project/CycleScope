"""Match one captured CUDA kernel to its hardware activity duration.

The delivery suites retain whole-public-operation timing for product-level
latency.  Simulator accuracy is narrower: one prepared simulator bundle
contains one selected kernel launch, so its hardware reference must be the
same launch and no surrounding helper kernels.

PyTorch's profiler uses CUDA activity records for device start/end times.  The
helpers below intentionally fail on an ambiguous symbol match; silently
choosing a similarly named launch would make an accuracy number meaningless.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
from typing import Any, Callable, Mapping


CONTRACT_SCHEMA = "delivery.kernel-comparison.v1"
TIMING_SCHEMA = "delivery.matched-kernel-timing.v1"
_KERNEL_FILE = re.compile(r"^kernel-(\d+)-")


class KernelTimingError(RuntimeError):
    """A hardware activity cannot be matched uniquely to the captured kernel."""


def _contract_digest(payload: Mapping[str, Any]) -> str:
    value = {key: item for key, item in payload.items() if key != "contract_digest"}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _launch_id(kernel: Mapping[str, Any]) -> int | None:
    match = _KERNEL_FILE.match(str(kernel.get("kernel_file") or ""))
    return int(match.group(1)) if match is not None else None


def build_comparison_contract(
    primary: Mapping[str, Any],
    traced_kernels: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe the exact trace launch that hardware timing must select."""

    name = str(primary.get("kernel_name") or "").strip()
    if not name:
        raise KernelTimingError("selected trace kernel has no symbol")
    ordered = sorted(
        traced_kernels,
        key=lambda row: (
            _launch_id(row) is None,
            _launch_id(row) if _launch_id(row) is not None else 0,
            str(row.get("kernel_file") or ""),
        ),
    )
    primary_key = (
        str(primary.get("kernel_file") or ""),
        name,
    )
    operation_ordinal = next(
        (
            index
            for index, row in enumerate(ordered)
            if (str(row.get("kernel_file") or ""), str(row.get("kernel_name") or ""))
            == primary_key
        ),
        None,
    )
    if operation_ordinal is None:
        raise KernelTimingError("selected trace kernel is absent from traced kernel list")
    same_name_rows = [row for row in ordered if str(row.get("kernel_name") or "") == name]
    same_name_ordinal = next(
        index
        for index, row in enumerate(same_name_rows)
        if str(row.get("kernel_file") or "") == primary_key[0]
    )
    payload: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "comparison_scope": "one_complete_cuda_kernel_launch",
        "hardware_metric": "cuda_activity_kernel_start_to_end",
        "simulation_metric": "full_kernel_cycles",
        "identity": {
            "kernel_name": name,
            "trace_kernel_file": primary_key[0],
            "trace_launch_id": _launch_id(primary),
            "operation_kernel_ordinal": operation_ordinal,
            "same_name_ordinal": same_name_ordinal,
            "same_name_launch_count": len(same_name_rows),
            "cta_count": int(primary.get("cta_count") or 0),
            "active_sm_count": int(primary.get("active_sm_count") or 0),
            "sm_ids": [int(value) for value in primary.get("sm_ids") or []],
        },
        "ambiguity_policy": "reject_unless_symbol_and_same_name_ordinal_select_one_launch",
    }
    payload["contract_digest"] = _contract_digest(payload)
    return payload


def build_marker_comparison_contract(
    primary: Mapping[str, Any],
    traced_kernels: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the identity plus the shared outermost-marker timing interval."""

    payload = build_comparison_contract(primary, traced_kernels)
    payload.update(
        comparison_scope="outermost_in_kernel_clock_marker_interval",
        hardware_metric="max_hardware_warp_end_minus_min_hardware_warp_start",
        simulation_metric="model.marker_timing.marker_interval_cycles",
        clock_marker_policy="outermost_pair",
    )
    payload["contract_digest"] = _contract_digest(payload)
    return payload


def attach_code_identity(
    contract: Mapping[str, Any],
    *,
    binary_sha256: str,
    sass_sha256: str,
) -> dict[str, Any]:
    payload = dict(contract)
    payload["code_identity"] = {
        "binary_sha256": str(binary_sha256),
        "selected_sass_sha256": str(sass_sha256),
    }
    payload["contract_digest"] = _contract_digest(payload)
    return payload


def load_comparison_contract(path: Path, case_id: str) -> dict[str, Any]:
    """Load one contract from a direct file, capture metadata, or corpus manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    contract: Any = None
    if payload.get("schema_version") == CONTRACT_SCHEMA:
        contract = payload
    elif isinstance(payload.get("comparison_contract"), Mapping):
        contract = payload["comparison_contract"]
    elif isinstance(payload.get("cases"), list):
        matches = [
            row
            for row in payload["cases"]
            if str(row.get("name") or row.get("case_id") or "") == case_id
        ]
        if len(matches) != 1:
            raise KernelTimingError(
                f"expected one comparison contract for {case_id}, found {len(matches)}"
            )
        contract = matches[0].get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise KernelTimingError(f"no comparison contract for {case_id} in {path}")
    result = dict(contract)
    if result.get("schema_version") != CONTRACT_SCHEMA:
        raise KernelTimingError("unsupported comparison contract schema")
    if result.get("contract_digest") != _contract_digest(result):
        raise KernelTimingError("comparison contract digest mismatch")
    return result


def validate_marker_code_identity(
    contract: Mapping[str, Any],
    kernel_entries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require the timed marker kernel to have the captured SASS exactly.

    JIT cubins can differ in non-instruction bytes between otherwise identical
    builds.  The selected function's complete cuobjdump SASS is therefore the
    required identity; the cubin digest is retained as an additional fact.
    """

    identity = contract.get("code_identity")
    if not isinstance(identity, Mapping):
        raise KernelTimingError("comparison contract has no captured code identity")
    expected_sass = str(identity.get("selected_sass_sha256") or "")
    if not expected_sass:
        raise KernelTimingError("comparison contract has no selected SASS digest")
    matches = [
        dict(row)
        for row in kernel_entries
        if str(row.get("sass_sha256") or "") == expected_sass
    ]
    if len(matches) != 1:
        raise KernelTimingError(
            "captured SASS must match exactly one timed kernel; "
            f"expected={expected_sass}, matches={len(matches)}"
        )
    selected = matches[0]
    expected_binary = str(identity.get("binary_sha256") or "")
    actual_binary = str(selected.get("cubin_sha256") or "")
    return {
        "status": "matched",
        "identity_basis": "exact_selected_cuobjdump_sass_sha256",
        "selected_sass_sha256": expected_sass,
        "captured_binary_sha256": expected_binary or None,
        "timed_binary_sha256": actual_binary or None,
        "binary_sha256_exact_match": bool(
            expected_binary and actual_binary and expected_binary == actual_binary
        ),
    }


def _demangle(name: str) -> str:
    if not name.startswith("_Z") or shutil.which("c++filt") is None:
        return name
    completed = subprocess.run(
        ["c++filt", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=5,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else name


def _canonical_name(name: str) -> str:
    value = " ".join(str(name).strip().split())
    for prefix in ("void ", "__global__ void "):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def _qualified_stem(name: str) -> str:
    value = _canonical_name(name)
    before_arguments = value.split("(", 1)[0]
    before_templates = before_arguments.split("<", 1)[0]
    return before_templates.rsplit(" ", 1)[-1]


def _names_match(target: str, observed: str) -> tuple[bool, str]:
    target_forms = {_canonical_name(target), _canonical_name(_demangle(target))}
    observed_forms = {_canonical_name(observed), _canonical_name(_demangle(observed))}
    if target_forms & observed_forms:
        return True, "exact_symbol_or_demangled_symbol"
    target_stems = {_qualified_stem(value) for value in target_forms}
    observed_stems = {_qualified_stem(value) for value in observed_forms}
    shared = {value for value in target_stems & observed_stems if len(value) >= 16}
    if shared:
        return True, "unique_qualified_function_stem"
    # Kineto/CUPTI may truncate very long generated names.  Accept a long
    # common prefix only when the caller later proves that it selects one row.
    for left in target_forms:
        for right in observed_forms:
            shorter = min(len(left), len(right))
            if shorter >= 96 and (left.startswith(right) or right.startswith(left)):
                return True, "unique_long_symbol_prefix"
    return False, "no_symbol_match"


def select_activity(
    events: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    identity = contract.get("identity")
    if not isinstance(identity, Mapping):
        raise KernelTimingError("comparison contract has no identity")
    target = str(identity.get("kernel_name") or "")
    candidates: list[dict[str, Any]] = []
    for event in events:
        matched, method = _names_match(target, str(event.get("name") or ""))
        if matched:
            row = dict(event)
            row["match_method"] = method
            candidates.append(row)
    if not candidates:
        observed = [str(row.get("name") or "") for row in events[:16]]
        raise KernelTimingError(
            f"captured kernel symbol was not observed; target={target!r}, observed={observed!r}"
        )
    ordinal = int(identity.get("same_name_ordinal") or 0)
    expected_count = int(identity.get("same_name_launch_count") or 1)
    if len(candidates) != expected_count:
        raise KernelTimingError(
            "matched-kernel launch count changed: "
            f"expected {expected_count}, observed {len(candidates)} for {target!r}"
        )
    if ordinal < 0 or ordinal >= len(candidates):
        raise KernelTimingError(
            f"matched-kernel ordinal {ordinal} is outside {len(candidates)} launches"
        )
    selected = candidates[ordinal]
    duration = float(selected.get("duration_ns") or 0.0)
    if duration <= 0:
        raise KernelTimingError("matched CUDA activity has a non-positive duration")
    selected["matched_candidate_count"] = len(candidates)
    selected["selected_same_name_ordinal"] = ordinal
    return selected


def _event_duration_ns(event: Any) -> float:
    time_range = getattr(event, "time_range", None)
    elapsed = getattr(time_range, "elapsed_us", None)
    if callable(elapsed):
        value = float(elapsed()) * 1000.0
        if value > 0:
            return value
    for attribute in (
        "self_device_time_total",
        "device_time_total",
        "self_cuda_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, attribute, None)
        if value is not None and float(value) > 0:
            return float(value) * 1000.0
    return 0.0


def _event_start_ns(event: Any, fallback: int) -> float:
    time_range = getattr(event, "time_range", None)
    value = getattr(time_range, "start", None)
    return float(value) * 1000.0 if value is not None else float(fallback)


def cuda_activity_rows(torch: Any, profiler: Any) -> list[dict[str, Any]]:
    cuda_type = getattr(getattr(torch, "autograd", None), "DeviceType", None)
    cuda_value = getattr(cuda_type, "CUDA", None)
    rows = []
    for index, event in enumerate(profiler.events()):
        device_type = getattr(event, "device_type", None)
        is_cuda = device_type == cuda_value or str(device_type).upper().endswith("CUDA")
        if not is_cuda:
            continue
        duration = _event_duration_ns(event)
        if duration <= 0:
            continue
        rows.append(
            {
                "name": str(getattr(event, "name", "")),
                "duration_ns": duration,
                "start_ns": _event_start_ns(event, index),
                "device_index": getattr(event, "device_index", None),
                "device_resource_id": getattr(event, "device_resource_id", None),
            }
        )
    rows.sort(key=lambda row: (float(row["start_ns"]), str(row["name"])))
    return rows


def profile_matched_kernel(
    torch: Any,
    invoke: Callable[[], Any],
    contract: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Run one operation and return one exact CUDA kernel activity."""

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as profiler:
        result = invoke()
        torch.cuda.synchronize()
    return result, select_activity(cuda_activity_rows(torch, profiler), contract)


def coefficient_of_variation(samples: list[float]) -> float:
    mean = statistics.fmean(samples)
    return statistics.stdev(samples) / mean * 100.0 if len(samples) > 1 and mean else 0.0


def consecutive_phase_balanced_medians(samples: list[float]) -> list[float]:
    """Reduce four launches to one sample without aliasing period-two modes."""

    if len(samples) % 4:
        raise KernelTimingError("raw matched-kernel sample count must be divisible by four")
    return [
        statistics.median(samples[index:index + 4])
        for index in range(0, len(samples), 4)
    ]


def summarize_matched_samples(
    contract: Mapping[str, Any],
    raw_activities: list[Mapping[str, Any]],
    *,
    fixed_clock_per_nanosecond: float,
    discarded_activities: list[Mapping[str, Any]],
) -> dict[str, Any]:
    simulation_metric = str(contract.get("simulation_metric") or "")
    if simulation_metric != "full_kernel_cycles":
        raise KernelTimingError(
            "CUDA activity timing requires a full-kernel comparison contract"
        )
    comparison_scope = str(contract.get("comparison_scope") or "")
    if comparison_scope != "one_complete_cuda_kernel_launch":
        raise KernelTimingError(
            "CUDA activity timing requires a one-complete-kernel comparison scope"
        )
    raw_ns = [float(row["duration_ns"]) for row in raw_activities]
    samples_ns = consecutive_phase_balanced_medians(raw_ns)
    raw_cycles = [value * fixed_clock_per_nanosecond for value in raw_ns]
    samples_cycles = consecutive_phase_balanced_medians(raw_cycles)
    first = dict(raw_activities[0])
    return {
        "schema_version": TIMING_SCHEMA,
        "status": "matched",
        "eligible_for_simulator_accuracy": True,
        "scope": comparison_scope,
        "method": "CUDA activity kernel start/end timestamps",
        "hardware_metric": contract.get("hardware_metric"),
        "contract_digest": contract.get("contract_digest"),
        "kernel_name_expected": (contract.get("identity") or {}).get("kernel_name"),
        "kernel_name_observed": first.get("name"),
        "match_method": first.get("match_method"),
        "same_name_ordinal": first.get("selected_same_name_ordinal"),
        "matched_candidate_count_per_sample": [
            int(row.get("matched_candidate_count") or 0) for row in raw_activities
        ],
        "sample_count": len(samples_ns),
        "raw_sample_count": len(raw_ns),
        "raw_launches_per_sample": 4,
        "aggregation": "median_of_each_consecutive_phase_balanced_quartet",
        "discarded_timer_settling_samples_ns": [
            float(row["duration_ns"]) for row in discarded_activities
        ],
        "samples_duration_ns": samples_ns,
        "samples_duration_ns_raw": raw_ns,
        "median_duration_ns": statistics.median(samples_ns),
        "fixed_clock_per_nanosecond": fixed_clock_per_nanosecond,
        "samples_cycles": samples_cycles,
        "samples_cycles_raw": raw_cycles,
        "median_cycles": statistics.median(samples_cycles),
        "cv_percent": coefficient_of_variation(samples_ns),
        "simulation_metric": simulation_metric,
        "whole_operation_timing_must_not_be_used_for_simulator_accuracy": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_static_code_identity(
    contract: Mapping[str, Any],
    *,
    binary: Path,
    sass: Path,
) -> dict[str, Any]:
    """Require a prebuilt timed binary and its SASS to equal the capture."""

    identity = contract.get("code_identity")
    if not isinstance(identity, Mapping):
        raise KernelTimingError("comparison contract has no captured code identity")
    if not binary.is_file() or not sass.is_file():
        raise KernelTimingError(f"timed binary or SASS is missing: {binary}, {sass}")
    expected_binary = str(identity.get("binary_sha256") or "")
    expected_sass = str(identity.get("selected_sass_sha256") or "")
    actual_binary = sha256_file(binary)
    actual_sass = sha256_file(sass)
    if actual_binary != expected_binary or actual_sass != expected_sass:
        raise KernelTimingError(
            "timed CUTLASS code differs from the captured simulator code"
        )
    return {
        "status": "matched",
        "identity_basis": "exact_binary_and_cuobjdump_sass_sha256",
        "binary_sha256": actual_binary,
        "selected_sass_sha256": actual_sass,
    }


def validate_runtime_binary_identity(
    contract: Mapping[str, Any],
    roots: list[Path],
) -> dict[str, Any]:
    """Find the exact captured JIT binary used by an attention run."""

    identity = contract.get("code_identity")
    if not isinstance(identity, Mapping):
        raise KernelTimingError("comparison contract has no captured code identity")
    expected = str(identity.get("binary_sha256") or "")
    if not expected:
        raise KernelTimingError("comparison contract has no binary digest")
    candidates = sorted({
        path.resolve()
        for root in roots
        if root.exists()
        for pattern in ("*.cubin", "*.so")
        for path in root.rglob(pattern)
        if path.is_file()
    })
    matches = [path for path in candidates if sha256_file(path) == expected]
    if not matches:
        raise KernelTimingError(
            "no runtime binary matches the simulator capture; "
            f"searched {len(candidates)} binaries"
        )
    return {
        "status": "matched",
        "identity_basis": "exact_runtime_binary_sha256",
        "binary_sha256": expected,
        "matching_paths": [str(path) for path in matches],
        "selected_sass_sha256": identity.get("selected_sass_sha256"),
    }
