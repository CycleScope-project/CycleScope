#!/usr/bin/env python3
"""Run one DeepGEMM SM90 attention case on the full GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

COMMON_ROOT = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
from matched_kernel_timing import (  # noqa: E402
    load_comparison_contract,
    validate_marker_code_identity,
)

from tools.microbench.deepgemm_sm90_runner import (
    _consecutive_medians,
    _delivery_raw_per_recorded_sample,
    _dump_sass,
    _gpu_profile_matches,
    _minimum_full_gpu_sms,
    _normal_span_cycles,
    _nvidia_snapshot,
    _selected_gpu_profile,
)
from tools.microbench.deepgemm_sm90_suite import (
    CLUSTER_SM_MAX_CYCLES,
    CLUSTER_SM_MIN_CYCLES,
    CLUSTER_SM_MODE,
    coefficient_of_variation,
    summarize_marker_rows,
    validate_cluster_sm_topology,
    SuiteManifestError,
)


MAX_CTAS = 16_384
MAX_WARPS = 16
FIELDS = 7
BANKS = 2
MARKER_BYTES = MAX_CTAS * MAX_WARPS * FIELDS * 8


def import_deepgemm(package: Path, cache_root: Path):
    os.environ["DG_JIT_CACHE_DIR"] = str(cache_root.resolve())
    sys.path.insert(0, str(package.resolve()))
    import deep_gemm  # type: ignore
    from tools.microbench.actual_binding import preserve_device_library_root
    preserve_device_library_root(deep_gemm, package)
    return deep_gemm


def marker_tensor(output: torch.Tensor) -> torch.Tensor:
    storage = output.untyped_storage()
    storage_bytes = int(storage.nbytes())
    total_marker_bytes = BANKS * MARKER_BYTES
    if storage_bytes < total_marker_bytes:
        raise RuntimeError("attention output does not contain the two marker banks")
    byte_view = torch.empty(0, dtype=torch.uint8, device=output.device)
    byte_view = byte_view.set_(
        storage,
        storage_bytes - total_marker_bytes,
        (total_marker_bytes,),
        (1,),
    )
    return byte_view.view(torch.int64).view(BANKS, MAX_CTAS, MAX_WARPS, FIELDS)


def read_marker_rows(marker: torch.Tensor) -> list[dict[str, int]]:
    values = marker.detach().cpu()
    coordinates = (values != 0).any(dim=-1).nonzero(as_tuple=False)
    rows: list[dict[str, int]] = []
    for coordinate in coordinates:
        bank, cta_id, warp = (int(v) for v in coordinate.tolist())
        fields = values[bank, cta_id, warp].tolist()
        rows.append({
            "bank": bank,
            "cta_id": cta_id,
            "warp": warp,
            "start_clock64": int(fields[0]),
            "end_clock64": int(fields[1]),
            "start_globaltimer": int(fields[2]),
            "end_globaltimer": int(fields[3]),
            "sm_id": int(fields[4]),
            "cluster_cta_rank": int(fields[5]),
            "cluster_cta_count": int(fields[6]),
        })
    if not rows:
        raise RuntimeError("instrumented attention launch produced no valid marker rows")
    if any(row["cta_id"] >= MAX_CTAS - 1 for row in rows):
        raise RuntimeError("attention marker CTA capacity was exhausted")
    return rows


def globaltimer_span(rows: list[dict[str, int]]) -> float:
    if not rows:
        raise RuntimeError("invalid globaltimer marker rows")
    for row in rows:
        if int(row["end_globaltimer"]) <= int(row["start_globaltimer"]):
            raise RuntimeError("marker row lacks positive globaltimer")
    start = min(row["start_globaltimer"] for row in rows)
    end = max(row["end_globaltimer"] for row in rows)
    if end <= start:
        raise RuntimeError("invalid globaltimer marker rows")
    return float(end - start)


def traced_launch(launch: Any) -> torch.Tensor:
    """Place exactly one public DeepGEMM operation inside the profiler range."""

    cudart = torch.cuda.cudart()
    torch.cuda.synchronize()
    started = False
    stopped = False
    try:
        start_status = int(cudart.cudaProfilerStart())
        if start_status != 0:
            raise RuntimeError(f"cudaProfilerStart failed: {start_status}")
        started = True
        output = launch()
        torch.cuda.synchronize()
        stop_status = int(cudart.cudaProfilerStop())
        stopped = True
        if stop_status != 0:
            raise RuntimeError(f"cudaProfilerStop failed: {stop_status}")
    finally:
        if started and not stopped:
            cudart.cudaProfilerStop()
    torch.cuda.synchronize()
    return output


def logits_dtype(tag: str) -> torch.dtype:
    return torch.bfloat16 if tag == "bf16" else torch.float32


def expected_value(num_heads: int, head_dim: int) -> float:
    return float(num_heads * head_dim) / 256.0


def check_prefill(output: torch.Tensor, case: dict[str, Any]) -> dict[str, Any]:
    seq_len_kv = int(case["seq_len_kv"])
    expected = expected_value(int(case["num_heads"]), int(case["head_dim"]))
    window_start = seq_len_kv // 4
    window_end = seq_len_kv * 3 // 4
    finite_values: list[float] = []
    invalid_values: list[float] = []
    for row in (0, output.shape[0] - 1):
        if case["mode"] == "prefill_compressed":
            columns = (0, (window_end - window_start) // 2, window_end - window_start - 1)
        else:
            columns = (window_start, (window_start + window_end) // 2, window_end - 1)
            invalid_values.extend(float(output[row, col].item()) for col in (0, window_start - 1, window_end, seq_len_kv - 1))
        finite_values.extend(float(output[row, col].item()) for col in columns)
    max_error = max(abs(value - expected) for value in finite_values)
    invalid_ok = all(math.isinf(value) and value < 0 for value in invalid_values)
    tolerance = 0.25 if case["logits_dtype"] == "bf16" else 0.05
    return {
        "status": "pass" if max_error <= tolerance and invalid_ok else "fail",
        "expected": expected,
        "max_abs_error": max_error,
        "tolerance": tolerance,
        "invalid_region_negative_infinity": invalid_ok,
    }


def prepare_prefill(deep_gemm: Any, case: dict[str, Any]):
    seq_len = int(case["seq_len"])
    seq_len_kv = int(case["seq_len_kv"])
    num_heads = int(case["num_heads"])
    head_dim = int(case["head_dim"])
    q = torch.full((seq_len, num_heads, head_dim), 0.0625, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    kv = torch.full((seq_len_kv, head_dim), 0.0625, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    kv_scale = torch.ones((seq_len_kv,), dtype=torch.float32, device="cuda")
    weights = torch.ones((seq_len, num_heads), dtype=torch.float32, device="cuda")
    window_start = seq_len_kv // 4
    window_end = seq_len_kv * 3 // 4
    ks = torch.full((seq_len,), window_start, dtype=torch.int32, device="cuda")
    ke = torch.full((seq_len,), window_end, dtype=torch.int32, device="cuda")
    clean = case["mode"] == "prefill_clean"
    max_seqlen_k = 0 if clean else window_end - window_start
    dtype = logits_dtype(str(case["logits_dtype"]))

    def launch() -> torch.Tensor:
        return deep_gemm.fp8_fp4_mqa_logits(
            (q, None), (kv, kv_scale), weights, ks, ke,
            clean_logits=clean, max_seqlen_k=max_seqlen_k, logits_dtype=dtype,
        )

    return launch, check_prefill, ["sm90_fp8_mqa_logits", *( ["smxx_clean_logits"] if clean else [])]


def check_paged(output: torch.Tensor, case: dict[str, Any]) -> dict[str, Any]:
    next_n = int(case["next_n"])
    avg_kv = int(case["avg_kv"])
    expected = expected_value(int(case["num_heads"]), int(case["head_dim"]))
    values: list[float] = []
    for query_index in range(next_n):
        row = query_index
        context = avg_kv - next_n + query_index + 1
        for column in (0, context // 2, context - 1):
            values.append(float(output[row, column].item()))
    max_error = max(abs(value - expected) for value in values)
    tolerance = 0.25 if case["logits_dtype"] == "bf16" else 0.05
    return {
        "status": "pass" if max_error <= tolerance else "fail",
        "expected": expected,
        "max_abs_error": max_error,
        "tolerance": tolerance,
    }


def prepare_paged(deep_gemm: Any, case: dict[str, Any], num_sms: int):
    batch_size = int(case["batch_size"])
    next_n = int(case["next_n"])
    avg_kv = int(case["avg_kv"])
    num_heads = int(case["num_heads"])
    head_dim = int(case["head_dim"])
    block_kv = 64
    blocks_per_sequence = (avg_kv + block_kv - 1) // block_kv
    num_blocks = batch_size * blocks_per_sequence
    q = torch.full((batch_size, next_n, num_heads, head_dim), 0.0625, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.ones((batch_size * next_n, num_heads), dtype=torch.float32, device="cuda")
    context_row = torch.arange(avg_kv - next_n + 1, avg_kv + 1, dtype=torch.int32, device="cuda")
    context_lens = context_row.unsqueeze(0).repeat(batch_size, 1)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(batch_size, blocks_per_sequence)

    fused_flat = torch.empty((num_blocks, block_kv * (head_dim + 4)), dtype=torch.uint8, device="cuda")
    values = torch.full((num_blocks, block_kv * head_dim), 0.0625, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    fused_flat[:, : block_kv * head_dim] = values.view(torch.uint8)
    scales = torch.ones((num_blocks, block_kv), dtype=torch.float32, device="cuda")
    fused_flat[:, block_kv * head_dim :] = scales.view(torch.uint8).view(num_blocks, block_kv * 4)
    fused_cache = fused_flat.view(num_blocks, block_kv, 1, head_dim + 4)
    # Untimed input for the logits kernel. Independent metadata coverage is
    # the separately timed core case `fp8_paged_mqa_metadata`.
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, None)
    dtype = logits_dtype(str(case["logits_dtype"]))

    def launch() -> torch.Tensor:
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, None), fused_cache, weights, context_lens, block_table, schedule_meta,
            max_context_len=avg_kv, clean_logits=False, logits_dtype=dtype, indices=None,
        )

    return launch, check_paged, ["sm90_fp8_paged_mqa_logits"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    matches = [dict(case) for case in manifest["cases"] if case["case_id"] == args.case_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown case: {args.case_id}")
    case = matches[0]
    comparison_contract = (
        load_comparison_contract(args.comparison_contract, args.case_id)
        if args.comparison_contract is not None
        else None
    )
    if comparison_contract is not None and (
        comparison_contract.get("comparison_scope")
        != "outermost_in_kernel_clock_marker_interval"
        or comparison_contract.get("clock_marker_policy") != "outermost_pair"
    ):
        raise RuntimeError(
            "DeepGEMM attention comparison requires the outermost marker interval"
        )
    snapshot = _nvidia_snapshot()
    gpu_profile = _selected_gpu_profile()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else ""
    if (
        torch.cuda.device_count() != 1
        or not _gpu_profile_matches(str(snapshot.get("name", "")), gpu_profile)
        or not _gpu_profile_matches(device_name, gpu_profile)
    ):
        raise RuntimeError(f"attention suite requires one {gpu_profile} GPU")
    device_sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    minimum_active_sms = _minimum_full_gpu_sms(device_sm_count)
    snapshot.update({
        "gpu_profile": gpu_profile,
        "device_sm_count": device_sm_count,
        "simulation_scope": "cluster_sm" if args.mode == "cluster_sm" else "full_gpu",
        "full_device_minimum_active_sms": minimum_active_sms,
        "minimum_required_active_sms": minimum_active_sms,
    })
    args.cache_root.mkdir(parents=True, exist_ok=True)
    deep_gemm = import_deepgemm(args.package, args.cache_root)
    if getattr(deep_gemm, "__version__", None) != "2.6.1":
        raise RuntimeError("unexpected DeepGEMM version")
    cluster_sm = args.mode == CLUSTER_SM_MODE
    if cluster_sm:
        if int(args.num_sms) not in (1, 2):
            raise RuntimeError("attention cluster_sm requires --num-sms 1 or 2")
        requested_num_sms = int(args.num_sms)
    else:
        if int(getattr(args, "num_sms", 0) or 0):
            raise RuntimeError("full_gpu attention cannot override num_sms")
        requested_num_sms = device_sm_count
    deep_gemm.set_num_sms(requested_num_sms)
    deep_gemm.set_pdl(False)
    marker_metadata_path = args.package / "deep_gemm" / "sm90_suite_marker.json"
    marker_metadata = json.loads(marker_metadata_path.read_text(encoding="utf-8"))
    from build_deepgemm_full7_package import is_full7_package
    if not is_full7_package(args.package):
        raise RuntimeError(
            "attention run refuses a marker package that is not full7 "
            "native-heuristic+layout_transpose_fp32+paged_mqa_logits_metadata"
        )
    if case["mode"] == "paged":
        launch, checker, modules = prepare_paged(deep_gemm, case, requested_num_sms)
    else:
        launch, checker, modules = prepare_prefill(deep_gemm, case)

    if getattr(args, "pilot", False):
        warmups = 1
        trials = 1
        raw_per_recorded_sample = 1
        raw_trials = 1
    else:
        warmups = int(manifest["selection"]["warmups"])
        trials = int(manifest["selection"]["trials"])
        raw_per_recorded_sample = 1 if args.trace else _delivery_raw_per_recorded_sample()
        raw_trials = 1 if args.trace else trials * raw_per_recorded_sample
    cache_state = {
        # Canonical count: executions of the target attention operation that
        # precede the first recorded sample (the explicit warmup loop only;
        # the attention runner has no extra JIT or correctness launch).
        "policy": "warm_after_three_launches",
        "warmup_launches": warmups,
        "staged_launches": {
            "warmup_loop": warmups,
        },
        "warmup_launches_source": "executed_warmup_loop",
        # The attention preheats run without marker rows, so no per-warmup SM
        # identity is observed; never claim measured residency here.
        "warmup_sm_ids_observed": None,
        "sm_residency_evidence": "not_observed",
    }
    last_output: torch.Tensor | None = None
    for _ in range(warmups):
        last_output = launch()
        torch.cuda.synchronize()

    samples: list[float] = []
    globaltimer_samples: list[float] = []
    active_sms: list[int] = []
    cta_counts: list[int] = []
    bank_ctas: list[dict[str, int]] = []
    cluster_sizes: list[list[int]] = []
    correctness_rows: list[dict[str, Any]] = []
    from tools.microbench.actual_binding import (
        cursor,
        export_evidence,
        publish_simulator_accuracy,
        record_trial,
    )
    marker_trials = []
    timing_rejection = None
    for trial_index in range(raw_trials):
        lo = cursor() if not args.trace else None
        output = traced_launch(launch) if args.trace else launch()
        if not args.trace:
            torch.cuda.synchronize()
        hi = cursor() if not args.trace else None
        rows = read_marker_rows(marker_tensor(output))
        trial = {"raw_trial_index": trial_index, "example_rows": rows,
                 "actual_launch_window": {"lo_exclusive": lo, "hi_inclusive": hi},
                 "marker_slot_count": BANKS * MAX_CTAS * MAX_WARPS,
                 "touched_slot_count": len(rows),
                 "untouched_slot_count": BANKS * MAX_CTAS * MAX_WARPS - len(rows)}
        record_trial(trial)
        marker_trials.append(trial)
        try:
            summary = summarize_marker_rows(rows)
            if timing_rejection is None:
                samples.append(_normal_span_cycles(rows))
                globaltimer_samples.append(globaltimer_span(rows))
        except (RuntimeError, SuiteManifestError) as exc:
            summary = {"cta_count": 0, "sm_ids": [], "cluster_cta_counts": [], "cluster_cta_ranks": []}
            timing_rejection = str(exc)
            samples.clear()
            globaltimer_samples.clear()
        active_sms.append(len(summary["sm_ids"]))
        cta_counts.append(int(summary["cta_count"]))
        bank_ctas.append({
            "mqa": len({row["cta_id"] for row in rows if row["bank"] == 0}),
            "clean_logits": len({row["cta_id"] for row in rows if row["bank"] == 1}),
        })
        cluster_sizes.append(sorted({int(row["cluster_cta_count"]) for row in rows}))
        correctness_rows.append(checker(output, case))
        last_output = output

    if timing_rejection is not None:
        raw_cycle_samples = []
        raw_globaltimer_samples = []
        samples = []
        globaltimer_samples = []
        median_cycles = None
        cv_percent = None
        raw_cv_percent = None
    else:
        raw_cycle_samples = list(samples)
        raw_globaltimer_samples = list(globaltimer_samples)
        samples = _consecutive_medians(raw_cycle_samples, raw_per_recorded_sample)
        globaltimer_samples = _consecutive_medians(
            raw_globaltimer_samples, raw_per_recorded_sample
        )
        median_cycles = statistics.median(samples)
        cv_percent = coefficient_of_variation(globaltimer_samples)
        raw_cv_percent = coefficient_of_variation(raw_globaltimer_samples)
    maximum_cv_percent = float(manifest["selection"]["maximum_cv_percent"])
    if gpu_profile == "h200":
        maximum_cv_percent = max(maximum_cv_percent, 3.0)
    required_banks = {0, 1} if case["mode"] == "prefill_clean" else {0}
    actual_banks = {
        bank for bank, name in ((0, "mqa"), (1, "clean_logits"))
        if any(row[name] > 0 for row in bank_ctas)
    }
    topology_minimum_active_sms = 1
    snapshot["minimum_required_active_sms"] = 1
    topology_pass = (
        min(active_sms) >= 1
        and min(cta_counts) >= 1
        and actual_banks == required_banks
        and all(row["mqa"] > 0 for row in bank_ctas)
        and all(
            row["clean_logits"] > 0 if case["mode"] == "prefill_clean" else row["clean_logits"] == 0
            for row in bank_ctas
        )
    )
    cluster_error = None
    last_summary = {"cta_count": 0, "sm_ids": [], "cluster_cta_counts": [], "cluster_cta_ranks": []}
    if cluster_sm:
        last_rows = read_marker_rows(marker_tensor(last_output)) if last_output is not None else []
        last_summary = summarize_marker_rows(last_rows) if last_rows else {"cta_count": 0, "sm_ids": [], "cluster_cta_counts": [], "cluster_cta_ranks": []}
        try:
            # Multi-kernel attention: each required bank must itself be 1 CTA / 1 SM
            # or one 2CTA cluster. Do not sum CTAs across MQA + clean_logits.
            for bank in sorted(required_banks):
                bank_rows = [row for row in last_rows if int(row["bank"]) == int(bank)]
                if not bank_rows:
                    raise SuiteManifestError(f"attention bank {bank} produced no markers")
                validate_cluster_sm_topology(requested_num_sms, summarize_marker_rows(bank_rows))
        except SuiteManifestError as exc:
            topology_pass = False
            cluster_error = str(exc)
    correctness_pass = all(row["status"] == "pass" for row in correctness_rows)
    kernel = _dump_sass(args.cache_root)
    marked_entries = [
        entry for entry in kernel["entries"]
        if int(entry.get("clock_marker_static_count") or 0) >= 2
        and int(entry.get("globaltimer_marker_static_count") or 0) >= 2
    ]
    sass_pass = bool(marked_entries) and any(int(entry.get("wgmma_static_count") or 0) > 0 for entry in marked_entries)
    if case["mode"] == "prefill_clean":
        sass_pass = sass_pass and len(marked_entries) >= 2
    contract_code_identity = (
        validate_marker_code_identity(comparison_contract, marked_entries)
        if comparison_contract is not None
        else None
    )
    status = "pass"
    if timing_rejection is not None:
        status = "unmeasured"
    elif not correctness_pass:
        status = "correctness_failed"
    elif not topology_pass:
        status = "topology_failed"
    elif not sass_pass:
        status = "marker_sass_failed"
    elif not cluster_sm and not args.trace and cv_percent > maximum_cv_percent:
        status = "unstable"
    if args.trace and status == "pass":
        status = "trace_pass"
    marker_capacity = {"max_warps_per_cta": int(marker_metadata["max_warps_per_cta"])}
    actual_evidence = export_evidence(marker_trials, raw_per_recorded_sample, marker_capacity=marker_capacity) if not args.trace else None
    unit_denominator = actual_evidence["unit_denominator"] if actual_evidence else None
    accuracy = publish_simulator_accuracy(unit_denominator, comparison_contract)
    return {
        "schema_version": "deepgemm-sm90-full-gpu-attention-result-v1",
        "case_id": case["case_id"],
        "source_case_id": case.get("source_case_id", case["case_id"]),
        "execution_mode": args.mode,
        "requested_num_sms": requested_num_sms,
        "round": args.round,
        "configuration": case,
        "implementation_modules": modules,
        "hardware": snapshot,
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "trials": correctness_rows,
        },
        "timing": {
            "method": "in-kernel globaltimer span",
            "scope": "whole_operation_globaltimer_across_marked_kernels",
            "kernels": list(modules),
            "within_requested_range": (
                False if median_cycles is None else (
                    CLUSTER_SM_MIN_CYCLES <= median_cycles <= CLUSTER_SM_MAX_CYCLES
                    if cluster_sm else True
                )
            ),
            "cycle_window": (
                [CLUSTER_SM_MIN_CYCLES, CLUSTER_SM_MAX_CYCLES] if cluster_sm else None
            ),
            "definition": (
                "median of each consecutive phase-balanced group of four raw "
                "globaltimer spans across all marked warps and CTAs"
                if raw_per_recorded_sample == 4
                else (
                    "median of each consecutive group of three raw globaltimer "
                    "spans across all marked warps and CTAs"
                    if raw_per_recorded_sample == 3
                    else "raw globaltimer span across all marked warps and CTAs"
                )
            ),
            "warmup_launches": warmups,
            "launches_before_each_raw_sample": [
                warmups + index for index in range(len(raw_globaltimer_samples))
            ],
            "cache_state": dict(cache_state),
            "marker_trials": marker_trials,
            "actual_launch_evidence": actual_evidence,
            "unit_denominator": unit_denominator,
            "timing_rejection": timing_rejection,
            "samples_globaltimer": globaltimer_samples,
            "samples_globaltimer_raw": raw_globaltimer_samples,
            "median_globaltimer": (
                None if not globaltimer_samples else statistics.median(globaltimer_samples)
            ),
            "cycle_compatibility_definition": "max(all_warp_end_clock64)-min(all_warp_start_clock64)",
            "samples_cycles": samples,
            "samples_cycles_raw": raw_cycle_samples,
            "median_cycles": median_cycles,
            "cv_percent": cv_percent,
            "raw_cv_percent": raw_cv_percent,
            "maximum_cv_percent": maximum_cv_percent,
            "reference_cycle_range_enforced": gpu_profile == "h100",
            "sample_aggregation": {
                "recorded_sample_count": len(globaltimer_samples),
                "raw_span_count": len(raw_globaltimer_samples),
                "raw_spans_per_recorded_sample": raw_per_recorded_sample,
                "operator": "median",
                "grouping": "consecutive_non_overlapping",
                "discarded_raw_spans": 0,
            },
            "simulator_comparison": {
                "schema_version": "delivery.marker-kernel-timing.v1",
                "status": accuracy["status"],
                "eligible_for_simulator_accuracy": accuracy["eligible_for_simulator_accuracy"],
                "scope": "outermost_in_kernel_clock_marker_interval",
                "hardware_metric": accuracy["hardware_metric"],
                "hardware_value_field": accuracy["hardware_value_field"],
                "simulation_metric": "model.marker_timing.marker_interval_cycles",
                "clock_marker_policy": "outermost_pair",
                "marker_interval_definition": "max(all_warp_end)-min(all_warp_start)",
                "selected_operation_kernel": accuracy["selected_operation_kernel"],
                "selected_unit_bank": accuracy["selected_unit_bank"],
                "samples_cycles": accuracy["samples_cycles"],
                "samples_cycles_raw": accuracy["samples_cycles_raw"],
                "median_cycles": accuracy["median_cycles"],
                "reason": accuracy["reason"],
                "contract_digest": accuracy["contract_digest"],
                "code_identity": contract_code_identity,
                "whole_operation_cycles_must_not_be_used_for_simulator_accuracy": True,
            },
        },
        "topology": {
            "status": "pass" if topology_pass else "fail",
            "error": cluster_error,
            "requested_num_sms": requested_num_sms,
            "active_sm_count_min": min(active_sms),
            "full_device_minimum_active_sms": minimum_active_sms,
            "minimum_required_active_sms": 1 if cluster_sm else 1,
            "observed_sm_count_min": min(active_sms),
            "observed_cta_count_min": min(cta_counts),
            "observed_cta_count": min(cta_counts),
            "observed_sm_count": min(active_sms),
            "cta_count": min(cta_counts),
            "sm_ids": last_summary.get("sm_ids") if cluster_sm else None,
            "cluster_cta_ranks": last_summary.get("cluster_cta_ranks") if cluster_sm else None,
            "coverage_scope": (
                "cluster_sm" if cluster_sm else (
                "full_device"
                if min(active_sms) >= minimum_active_sms
                else "launch_limited"
                )
            ),
            "cta_count_min": min(cta_counts),
            "active_sm_count_trials": active_sms,
            "marker_ctas_per_bank": bank_ctas,
            "cluster_cta_counts": cluster_sizes,
            "grid": {"x": min(cta_counts), "y": 1, "z": 1},
            "cluster_dim": {
                "x": max((max(row) if row else 1) for row in cluster_sizes) if cluster_sizes else 1,
                "y": 1,
                "z": 1,
            },
            "kernels": list(modules),
        },
        "kernel": kernel,
        "marked_sass_entry_count": len(marked_entries),
        "sass_validation": "pass" if sass_pass else "fail",
        "status": status,
        "trace_launch": bool(args.trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--comparison-contract",
        type=Path,
        help="captured contract or corpus manifest for strict simulator comparison",
    )
    parser.add_argument("--mode", choices=("full_gpu", CLUSTER_SM_MODE), default="full_gpu")
    parser.add_argument("--num-sms", type=int, default=0, help="cluster_sm: 1 or 2; full_gpu must stay 0")
    parser.add_argument("--pilot", action="store_true", help="one warmup and one recorded sample")
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    parser.add_argument("--trace", action="store_true", help="profile exactly one public attention operation")
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "case_id": result["case_id"], "status": result["status"],
        "median_cycles": result["timing"]["median_cycles"],
        "cv_percent": result["timing"]["cv_percent"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
