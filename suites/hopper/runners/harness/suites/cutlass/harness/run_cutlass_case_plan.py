#!/usr/bin/env python3
"""Run one shard of a CUTLASS case plan and preserve raw and parsed output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

COMMON_ROOT = Path(__file__).resolve().parents[2] / "common"
HARNESS_ROOT = Path(__file__).resolve().parent
for _path in (str(COMMON_ROOT), str(HARNESS_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from matched_kernel_timing import (  # noqa: E402
    load_comparison_contract,
    validate_static_code_identity,
)
from group_inputs import resolve_case_args  # noqa: E402


def parse_json_lines(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "schema_version" in value:
            rows.append(value)
    return rows


def extract_cycle_samples(row: dict) -> list[float]:
    values = row.get("whole_gpu_cycles")
    if isinstance(values, list):
        return [float(value) for value in values]
    if isinstance(values, (int, float)):
        return [float(values)]
    raise ValueError("measurement JSON has no cycle result")


def extract_globaltimer_samples(row: dict) -> list[float]:
    values = row.get("globaltimer_ns")
    if isinstance(values, list):
        return [float(value) for value in values]
    if isinstance(values, (int, float)):
        return [float(values)]
    raise ValueError("measurement JSON has no globaltimer result")


def command_for_sample_count(binary: str, arguments: list[str], count: int) -> list[str]:
    """Ask a dense multi-trial runner for ``count`` raw samples.

    ``--trials`` is that runner's sample count. Example ``--iterations`` is the
    captured workload and is left unchanged; the example wrapper reads
    ``CUTLASS_DELIVERY_SAMPLE_COUNT`` instead.
    """
    command = [binary]
    replaced = False
    for argument in arguments:
        if argument.startswith("--trials="):
            command.append(f"--trials={count}")
            replaced = True
        else:
            command.append(argument)
    if not replaced:
        return [binary, *arguments]
    return command



def _cli_warmup_argument(args: list[str] | None) -> int | None:
    for argument in args or []:
        text = str(argument)
        for name in ("--warmup=", "--warmups="):
            if text.startswith(name):
                return int(text[len(name):])
    return None


def _cli_iterations_argument(args: list[str] | None) -> int | None:
    for argument in args or []:
        text = str(argument)
        if text.startswith("--iterations="):
            return int(text[len("--iterations="):])
    return None


def recorded_cache_state(row: dict, kind: str) -> dict:
    """Return the warmup context actually executed for this case."""
    args = [str(value) for value in row.get("args") or []]
    warmup_cli = _cli_warmup_argument(args)
    if kind == "dense_harness":
        warmup_launches = 3 if warmup_cli is None else warmup_cli
        return {
            "policy": f"warm_after_{warmup_launches}_launches",
            "warmup_launches": warmup_launches,
            "warmup_launches_source": "executed_cli_argument",
            "staged_launches": {"warmup_loop": warmup_launches},
            "non_target_setup_launches": 0,
            "warmup_sm_ids_observed": None,
            "sm_residency_evidence": "not_observed",
        }
    if kind != "upstream_example":
        raise RuntimeError(f"unsupported CUTLASS case kind: {kind!r}")
    binary_name = str(row.get("binary_name") or "")
    post_mode = row.get("post_mode")
    staged: dict | None = None
    if binary_name == "example56_ptr_array":
        if post_mode not in (None, "pingpong", "cooperative"):
            raise RuntimeError(f"unsupported example 56 post mode: {post_mode!r}")
        warmup_launches = 4
        warmup_loop = 3
        source = "wrapper_post_warmup_loop_plus_upstream_correctness"
    elif binary_name.startswith("example55_"):
        if warmup_cli is None:
            raise RuntimeError("example 55 case lacks a parsable --warmup= value")
        warmup_launches = 1 + warmup_cli
        warmup_loop = warmup_cli
        source = "upstream_correctness_plus_cli_warmup_loop"
    elif binary_name in ("example61_topk_softmax", "example63_weight_prefetch"):
        warmup_launches = 1
        warmup_loop = 0
        source = "upstream_correctness_launch"
    elif binary_name == "example52_gather_scatter":
        iterations = _cli_iterations_argument(args)
        benchmark_launches = iterations if iterations is not None else 20
        warmup_launches = 1 + benchmark_launches + 3
        warmup_loop = 3
        staged = {
            "upstream_verify": 1,
            "upstream_benchmark_fused": benchmark_launches,
            "warmup_loop": warmup_loop,
        }
        source = "upstream_verify_plus_benchmark_plus_wrapper_post_warmup_loop"
    elif binary_name == "example62_sparse":
        warmup_launches = 3
        warmup_loop = 3
        source = "wrapper_post_warmup_loop"
    elif binary_name in (
        "example49_collective_builder",
        "example50_epilogue_swizzle",
        "example54_fp8_fused",
        "example57_grouped_fastacc",
        "example67_blockwise",
        "example67_groupwise_streamk",
        "example68_grouped_blockwise_coop",
        "example68_grouped_blockwise_pingpong",
        "example69_grouped_mixed",
        "example69_grouped_int4_bf16",
        "example69_grouped_int4_fp8",
        "delivery53_permute_full_gpu",
    ):
        warmup_launches = 0 if warmup_cli is None else warmup_cli
        warmup_loop = warmup_launches
        source = "cli_warmup_or_none"
    else:
        raise RuntimeError(f"unsupported CUTLASS upstream binary: {binary_name!r}")
    return {
        "policy": f"warm_after_{warmup_launches}_launches",
        "warmup_launches": warmup_launches,
        "warmup_launches_source": source,
        "staged_launches": staged if staged is not None else {"warmup_loop": warmup_loop},
        "non_target_setup_launches": 0,
        "non_target_setup_note": (
            "upstream verify() reference kernels are different kernel "
            "instantiations and are excluded from the target count"
        ),
        "warmup_sm_ids_observed": None,
        "sm_residency_evidence": "not_observed",
    }


def apply_delivery_env(case, env):
    mode = case.get("delivery_mode") or "full_gpu"
    env["CUTLASS_DELIVERY_MODE"] = str(mode)
    if case.get("post_mode"):
        env["CUTLASS_DELIVERY_POST_MODE"] = case["post_mode"]
    if case.get("cluster_sm_count") is not None:
        env["CUTLASS_CLUSTER_SM_COUNT"] = str(int(case["cluster_sm_count"]))
    if case.get("compiled_cluster"):
        env["CUTLASS_CLUSTER_SHAPE"] = str(case["compiled_cluster"])
    return env

def consecutive_triplet_medians(samples: list[float]) -> list[float]:
    if len(samples) % 3:
        raise ValueError("raw timing sample count must be divisible by three")
    return [statistics.median(samples[index:index + 3]) for index in range(0, len(samples), 3)]


_DISK_STOP_BYTES = 80 * 1024 ** 3


def _shared_data_free(path: Path) -> int:
    """Free bytes of a path on the artifact-root volume.

    Paths outside that volume (fixture tmp dirs) are not the GPU write guard.
    """
    resolved = path.resolve()
    root = Path(os.environ.get("CYCLESCOPE_ARTIFACT_ROOT", "/")).resolve()
    if resolved != root and root not in resolved.parents:
        return _DISK_STOP_BYTES
    return shutil.disk_usage(resolved).free


def _source_case_id(case: dict) -> str:
    source = str(case.get("source_case_id") or case["case_id"])
    suffix = "_cluster_sm"
    if source.endswith(suffix):
        return source[: -len(suffix)]
    return source


def _live_unit_publication(
    *,
    case_id: str,
    documents: list[dict],
    timing_argv: list[str],
    binary: Path,
    capture_dir: Path,
    bundle_dir: Path,
    log_mtime_ns: int,
) -> dict:
    experiment = Path(__file__).resolve().parents[3] / "scripts" / "experiments" / "T0016"
    if str(experiment) not in sys.path:
        sys.path.insert(0, str(experiment))
    from cutlass_marker_unit import bind_live_measurement

    try:
        return bind_live_measurement(
            case_id=case_id,
            documents=documents,
            timing_argv=timing_argv,
            binary=binary,
            capture_dir=capture_dir,
            bundle_dir=bundle_dir,
            log_mtime_ns=log_mtime_ns,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "publication_status": "refused",
            "median_cycles": None,
            "unit_denominator": None,
            "refusal": {"fact": "capture_binding", "detail": f"{type(exc).__name__}: {exc}"},
            "checks": {},
        }


def _attach_live_unit(result: dict, published: dict) -> bool:
    """Write the unit denominator onto the summary row. Returns whether it matched."""
    matched = (
        published.get("publication_status") == "same_unit_matched"
        and isinstance(published.get("unit_denominator"), dict)
    )
    result["marker_unit_publication"] = {
        "publication_status": published.get("publication_status"),
        "refusal": published.get("refusal"),
        "checks": published.get("checks"),
    }
    if matched:
        result["median_cycles"] = published["median_cycles"]
        result["unit_denominator"] = published["unit_denominator"]
        if result["unit_denominator"].get("median_cycles") != result["median_cycles"]:
            raise RuntimeError(f"{result.get('case_id')} parent median drifted from the unit denominator")
        result["simulator_comparison"]["status"] = "matched"
        result["simulator_comparison"]["eligible_for_simulator_accuracy"] = True
        return True
    result["unit_denominator"] = {
        "status": "refused",
        "eligible_for_simulator_accuracy": False,
        "median_cycles": None,
        "refusal": published.get("refusal"),
    }
    result["simulator_comparison"]["status"] = "marker_unit_unproven"
    result["simulator_comparison"]["eligible_for_simulator_accuracy"] = False
    result["simulator_comparison"]["refusal"] = published.get("refusal")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--delivery-group", type=int)
    parser.add_argument("--only-case")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        help="capture corpus whose marker interval and code identity match this run",
    )
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--bundle-map", type=Path)
    args = parser.parse_args()
    if (args.capture_root is None) != (args.bundle_map is None):
        raise RuntimeError("--capture-root and --bundle-map are both required to publish a unit denominator")
    bundle_map = (
        json.loads(args.bundle_map.read_text(encoding="utf-8"))
        if args.bundle_map is not None else {}
    )
    cluster_sm_plan = os.environ.get("CUTLASS_DELIVERY_MODE") == "cluster_sm"
    if cluster_sm_plan:
        if args.repeats < 1 or args.repeats > 9:
            raise ValueError("cluster_sm timing requires 1..9 recorded GPU-cycle samples")
    elif args.repeats != 9:
        raise ValueError("formal delivery timing requires exactly 9 recorded triplet medians")
    timeout_seconds = int(os.environ.get("CUTLASS_DELIVERY_TIMEOUT_SECONDS", "600"))
    artifact_root_text = os.environ.get("CUTLASS_DELIVERY_ARTIFACTS_ROOT", "").strip()
    if not artifact_root_text:
        raise RuntimeError("CUTLASS_DELIVERY_ARTIFACTS_ROOT is required")
    artifact_root = Path(artifact_root_text)

    plan = json.loads(args.plan.read_text())
    if args.delivery_group is None:
        cases = [
            case for index, case in enumerate(plan["cases"])
            if index % args.shard_count == args.shard_id
        ]
    else:
        cases = [
            case for case in plan["cases"]
            if int(case.get("delivery_group", -1)) == args.delivery_group
        ]
    if args.only_case is not None:
        cases = [case for case in cases if str(case["case_id"]) == args.only_case]
        if len(cases) != 1:
            raise RuntimeError(
                f"case is not in the selected CUTLASS group: {args.only_case}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"summary_shard{args.shard_id}.jsonl"
    failures = 0

    with summary_path.open("w") as summary_file:
        for position, case in enumerate(cases, 1):
            if args.capture_root is not None and _shared_data_free(args.output_dir) < _DISK_STOP_BYTES:
                stopped = {
                    **case,
                    "passed": False,
                    "correct": False,
                    "return_codes": [],
                    "logs": [],
                    "median_cycles": None,
                    "simulator_comparison": {
                        "schema_version": "delivery.marker-kernel-timing.v1",
                        "status": "infra_disk_stop",
                        "eligible_for_simulator_accuracy": False,
                        "refusal": {
                            "fact": "infra_disk_stop",
                            "detail": "/data free dropped below 80GiB; stopped before this case",
                        },
                    },
                    "unit_denominator": {
                        "status": "refused",
                        "eligible_for_simulator_accuracy": False,
                        "median_cycles": None,
                        "refusal": {"fact": "infra_disk_stop", "detail": "/data free below 80GiB"},
                    },
                }
                summary_file.write(json.dumps(stopped, sort_keys=True) + "\n")
                summary_file.flush()
                failures += 1
                print(f"[{position}/{len(cases)}] {case['case_id']} infra_disk_stop", flush=True)
                break
            binary_path = artifact_root / "build" / case["binary_name"]
            binary = str(binary_path)
            cycle_values = []
            globaltimer_values = []
            measurements = []
            case_logs = []
            return_codes = []
            start = time.time()
            invocation = 0
            cluster_sm = str(case.get("delivery_mode") or "") == "cluster_sm"
            raw_sample_count = args.repeats if cluster_sm else args.repeats * 3
            workload_args = resolve_case_args([str(arg) for arg in case["args"]])
            while len(cycle_values) < raw_sample_count:
                invocation_sample_count = min(
                    9, raw_sample_count - len(cycle_values)
                )
                env = os.environ.copy()
                env["CUTLASS_DELIVERY_CASE_ID"] = case["case_id"]
                env["CUTLASS_DELIVERY_ALGORITHM"] = case["algorithm"]
                env["CUTLASS_DELIVERY_SAMPLE_COUNT"] = str(
                    invocation_sample_count
                )
                apply_delivery_env(case, env)
                command = command_for_sample_count(
                    binary,
                    workload_args,
                    invocation_sample_count,
                )
                try:
                    completed = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                        timeout=timeout_seconds,
                    )
                    command_output = completed.stdout
                    return_code = completed.returncode
                except subprocess.TimeoutExpired as exc:
                    command_output = exc.stdout or ""
                    if isinstance(command_output, bytes):
                        command_output = command_output.decode(errors="replace")
                    command_output += (
                        f"\nDELIVERY_ERROR: timed out after {timeout_seconds} seconds\n"
                    )
                    return_code = 124
                log_path = args.output_dir / f"{case['case_id']}.invocation{invocation}.log"
                log_path.write_text(command_output)
                case_logs.append(str(log_path))
                return_codes.append(return_code)
                rows = parse_json_lines(command_output)
                if rows:
                    row = rows[-1]
                    values = extract_cycle_samples(row)
                    timer_values = extract_globaltimer_samples(row)
                    if len(values) != len(timer_values):
                        raise ValueError(
                            f"{case['case_id']} returned different cycle/globaltimer sample counts"
                        )
                    remaining = raw_sample_count - len(cycle_values)
                    if len(values) > remaining:
                        raise ValueError(
                            f"{case['case_id']} returned {len(values)} samples with only {remaining} requested"
                        )
                    cycle_values.extend(values)
                    globaltimer_values.extend(timer_values)
                    measurements.append(row)
                invocation += 1
                if not rows or return_code != 0:
                    break

            text_outputs = "\n".join(Path(path).read_text() for path in case_logs)
            process_ok = bool(return_codes) and all(code == 0 for code in return_codes)
            correct_text = (
                process_ok
                and "Failed" not in text_outputs
                and "DELIVERY_ERROR" not in text_outputs
                and "ERROR:" not in text_outputs
            )
            correct_json = bool(measurements) and all(
                bool(measurement.get("correct", True))
                and int(measurement.get("upstream_return_code", 0)) == 0
                and int(measurement.get("post_return_code", 0)) == 0
                for measurement in measurements
            )
            active_sm_counts = [
                int(measurement.get(
                    "active_sm_count",
                    measurement.get("active_sm_count_min", 0),
                ))
                for measurement in measurements
            ]
            device_sm_counts = [
                int(measurement.get("device_sm_count", 132))
                for measurement in measurements
            ]
            minimum_required_active_sms = 1
            active_sm_count_ok = min(active_sm_counts, default=0) >= 1
            passed = (
                len(cycle_values) == raw_sample_count
                and len(globaltimer_values) == raw_sample_count
                and process_ok
                and correct_text
                and correct_json
            )
            if cluster_sm:
                recorded_cycle_values = list(cycle_values)
                recorded_globaltimer_values = list(globaltimer_values)
            else:
                recorded_cycle_values = consecutive_triplet_medians(cycle_values) if cycle_values else []
                recorded_globaltimer_values = consecutive_triplet_medians(globaltimer_values) if globaltimer_values else []
            comparison_contract = (
                load_comparison_contract(args.comparison_manifest, str(case["case_id"]))
                if args.comparison_manifest is not None
                else None
            )
            contract_code_identity = (
                validate_static_code_identity(
                    comparison_contract,
                    binary=binary_path,
                    sass=binary_path.with_suffix(".sass"),
                )
                if comparison_contract is not None
                else None
            )
            comparison_ready = bool(recorded_cycle_values) and comparison_contract is not None
            last_measurement = measurements[-1] if measurements else {}
            def unpack_cluster_ncta(raw):
                if raw is None:
                    return None, None
                value = int(raw)
                if value > 255:
                    nx = value & 0xFF
                    ny = (value >> 8) & 0xFF
                    nz = (value >> 16) & 0xFF
                    ncta = (value >> 32) & 0xFF
                    if ncta == 0:
                        ncta = max(1, nx * max(ny, 1) * max(nz, 1))
                    if ny == 0:
                        ny = 1
                    if nz == 0:
                        nz = 1
                    return ncta, [nx, ny, nz]
                return value, None
            raw_ncta = last_measurement.get("observed_cluster_ncta")
            unpacked_ncta, unpacked_dim = unpack_cluster_ncta(raw_ncta)
            observed_dim = last_measurement.get("cluster_dim") or unpacked_dim
            topology = {
                "grid": last_measurement.get("grid"),
                "cta_count": last_measurement.get("cta_count", last_measurement.get("active_cta_count")),
                "requested_sm_count": last_measurement.get("requested_sm_count"),
                "device_sm_count": last_measurement.get("device_sm_count"),
                "compiled_cluster": last_measurement.get("cluster") or last_measurement.get("compiled_cluster") or case.get("compiled_cluster"),
                "cluster_dim": observed_dim,
                "observed_cluster_ncta": unpacked_ncta,
                "cluster_ranks": last_measurement.get("cluster_ranks"),
                "sm_ids": last_measurement.get("sm_ids"),
                "active_sm_count": last_measurement.get("active_sm_count", last_measurement.get("active_sm_count_min")),
                "gpu_name": last_measurement.get("gpu_name"),
                "source_log": case_logs[-1] if case_logs else None,
            }
            result = {
                **case,
                "binary": binary,
                "repeats": args.repeats,
                "sample_count": len(recorded_cycle_values),
                "raw_sample_count": len(cycle_values),
                "process_invocations": invocation,
                "return_codes": return_codes,
                "logs": case_logs,
                "cycle_samples": recorded_cycle_values,
                "cycle_samples_raw": cycle_values,
                "globaltimer_samples": recorded_globaltimer_values,
                "globaltimer_samples_raw": globaltimer_values,
                "median_globaltimer": (
                    statistics.median(recorded_globaltimer_values) if recorded_globaltimer_values else None
                ),
                "median_cycles": statistics.median(recorded_cycle_values) if recorded_cycle_values else None,
                "timing": {
                    "cache_state": recorded_cache_state(case, case.get("kind")),
                    "scope": "outermost_in_kernel_clock_marker_interval",
                    "method": "device_kernel_clock64_over_globaltimer",
                    "cluster_sm_raw_samples": cluster_sm,
                },
                "observed_topology": topology,
                "cv_percent": (
                    statistics.stdev(recorded_globaltimer_values)
                    / statistics.fmean(recorded_globaltimer_values) * 100.0
                    if len(recorded_globaltimer_values) > 1 else 0.0
                ),
                "simulator_comparison": {
                    "schema_version": "delivery.marker-kernel-timing.v1",
                    "status": "matched" if comparison_ready else "missing_capture_contract",
                    "eligible_for_simulator_accuracy": comparison_ready,
                    "scope": "outermost_in_kernel_clock_marker_interval",
                    "hardware_metric": (
                        comparison_contract.get("hardware_metric")
                        if comparison_contract is not None
                        else "median_cycles"
                    ),
                    "hardware_value_field": "median_cycles",
                    "simulation_metric": "model.marker_timing.marker_interval_cycles",
                    "clock_marker_policy": "outermost_pair",
                    "marker_interval_definition": "max(all_warp_end)-min(all_warp_start)",
                    "whole_kernel_cycles_must_not_be_mixed_with_marker_interval": True,
                    "contract_digest": (
                        comparison_contract.get("contract_digest")
                        if comparison_contract is not None else None
                    ),
                    "code_identity": contract_code_identity,
                },
                "active_sm_count_min": min(active_sm_counts) if active_sm_counts else 0,
                "device_sm_count": min(device_sm_counts) if device_sm_counts else 0,
                "minimum_required_active_sms": minimum_required_active_sms,
                "expected_active_sm_count": None,
                "correct": bool(correct_text and correct_json),
                "passed": passed,
                "elapsed_seconds": time.time() - start,
            }
            if args.capture_root is not None:
                source_id = _source_case_id(case)
                bundle_text = bundle_map.get(source_id)
                if not measurements or bundle_text is None:
                    published = {
                        "publication_status": "refused",
                        "median_cycles": None,
                        "unit_denominator": None,
                        "refusal": {
                            "fact": "stdout" if bundle_text is not None else "capture_binding",
                            "detail": (
                                "measurement produced no JSON document"
                                if bundle_text is not None
                                else f"bundle map has no {source_id}"
                            ),
                        },
                        "checks": {},
                    }
                else:
                    published = _live_unit_publication(
                        case_id=source_id,
                        documents=measurements,
                        timing_argv=workload_args,
                        binary=binary_path,
                        capture_dir=args.capture_root / source_id,
                        bundle_dir=Path(str(bundle_text)),
                        log_mtime_ns=max(Path(path).stat().st_mtime_ns for path in case_logs),
                    )
                if passed and not _attach_live_unit(result, published):
                    passed = False
                    result["passed"] = False
            summary_file.write(json.dumps(result, sort_keys=True) + "\n")
            summary_file.flush()
            if not passed:
                failures += 1
            print(
                f"[{position}/{len(cases)}] {case['case_id']} "
                f"globaltimer={result['median_globaltimer']} cycles={result['median_cycles']} "
                f"active_sms={result['active_sm_count_min']} "
                f"passed={passed}",
                flush=True,
            )

    print(f"shard {args.shard_id} complete: {len(cases) - failures}/{len(cases)} passed")
    if not cases:
        raise RuntimeError("CUTLASS case plan selected no cases")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
