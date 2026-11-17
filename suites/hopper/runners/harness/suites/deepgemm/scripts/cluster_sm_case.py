#!/usr/bin/env python3
"""Run one DeepGEMM cluster_sm case: topology first, then K/seq tune, then 9 samples."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent.parent
# Scaler still aims at [100k, 500k] when topology can grow. Qualification
# accepts DG medians in (0, 500k] after the 2026-09-21 user exception.
MIN_CYCLES = 100_000
MAX_CYCLES = 500_000
TARGET_CYCLES = 250_000
# SM90ArchSpec::smem_capacity in upstream/csrc/jit_kernels/heuristics/sm90.hpp
SM90_SMEM_CAPACITY = 232448
TRANSPOSE_FP32_BLOCK_MN = 64


def _transpose_fp32_smem_bytes(sf_k: int) -> int:
    padded = int(sf_k) + (1 - (int(sf_k) % 2))
    return TRANSPOSE_FP32_BLOCK_MN * padded * 4


def _cap_transpose_fp32_sf_k(sf_k: int) -> int:
    value = max(32, int(sf_k))
    while value > 32 and _transpose_fp32_smem_bytes(value) > SM90_SMEM_CAPACITY:
        value -= 1
    return value


def _align(value: int, multiple: int = 128, minimum: int = 128) -> int:
    aligned = max(minimum, (int(value) // multiple) * multiple)
    return aligned if aligned >= minimum else minimum


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scale_core(shape: dict[str, Any], factor: float, class_id: str, *, target_ctas: int) -> dict[str, Any]:
    out = dict(shape)
    factor = min(4.0, max(0.25, float(factor)))
    one = int(target_ctas) == 1

    def scale_key(key: str, multiple: int = 128, minimum: int = 128) -> None:
        if key in out:
            out[key] = _align(max(minimum, int(round(int(out[key]) * factor))), multiple, minimum)

    if "ks" in out:
        small_k = max(int(v) for v in out["ks"]) <= 384 and len(out["ks"]) >= 8
        if small_k:
            scale_key("m")
            scale_key("n")
        else:
            out["ks"] = [_align(max(128, int(round(int(v) * factor))), 128, 128) for v in out["ks"]]
            out["k_per_group"] = max(out["ks"])
            scale_key("m", minimum=128)
            scale_key("n", minimum=256 if "cluster" in class_id else 128)
        return out
    if "tokens_per_group" in out:
        scale_key("k")
        scale_key("n", minimum=256)
        scale_key("tokens_per_group", minimum=128)
        return out
    if "expected_m" in out:
        scale_key("k")
        scale_key("n", minimum=256)
        scale_key("expected_m", minimum=128)
        # max_m must track expected_m monotonically; the vendor API derives the
        # masked layout from it, so never let it grow back after a shrink.
        out["max_m"] = min(
            int(out.get("max_m") or 0) or int(out["expected_m"]) * 2,
            int(out["expected_m"]) * 2,
        )
        out["max_m"] = _align(out["max_m"], 128, int(out["expected_m"]))
        # masked m-grouped work scales with num_groups; shrink it when the
        # workload is above the ceiling so the shape actually contracts.
        if "num_groups" in out:
            out["num_groups"] = max(1, int(round(int(out["num_groups"]) * factor)))
        return out
    if "sf_k" in out and "mn" in out:
        if one:
            # 1CTA grid is (ceil_div(mn, 64), num_sf_batches); mn>64 adds CTAs.
            out["mn"] = min(TRANSPOSE_FP32_BLOCK_MN, max(1, int(out.get("mn") or TRANSPOSE_FP32_BLOCK_MN)))
            out["sf_k"] = _cap_transpose_fp32_sf_k(max(32, int(round(int(out["sf_k"]) * factor))))
        else:
            scale_key("mn", multiple=4, minimum=256)
        return out
    if "avg_kv" in out and "batch_size" in out:
        scale_key("avg_kv", multiple=64, minimum=64)
        scale_key("batch_size", multiple=1, minimum=8)
        return out
    if {"b", "h", "d", "r"} <= set(out):
        scale_key("r", minimum=128)
        scale_key("b", multiple=1, minimum=16)
        return out
    if {"s", "m", "n", "k"} <= set(out) and "token_count" not in out:
        scale_key("k")
        if one:
            out["m"] = 64
            out["n"] = 64
            out["s"] = max(1, int(out.get("s") or 1))
        else:
            scale_key("s", multiple=1, minimum=1)
        return out
    if "token_count" in out:
        scale_key("k")
        if one:
            out["m"] = 64
            out["token_count"] = 64
        else:
            scale_key("m", minimum=128)
            out["token_count"] = int(out["m"])
        return out
    if "n" in out and "skip_head" in class_id:
        scale_key("k")
        # head_splits 128+128=256
        scale_key("n", multiple=256, minimum=256)
        scale_key("m", minimum=128)
        return out
    if {"m", "n", "k"} <= set(out):
        scale_key("k")
        scale_key("n", minimum=256)
        scale_key("m", minimum=128)
        return out
    return out


def _scale_attention(case: dict[str, Any], factor: float) -> dict[str, Any]:
    out = dict(case)
    factor = min(4.0, max(0.25, float(factor)))
    if out.get("mode") == "paged":
        out["avg_kv"] = _align(max(64, int(round(int(out["avg_kv"]) * factor))), 64, 64)
        out["batch_size"] = max(8, int(round(int(out["batch_size"]) * min(factor, 2.0))))
    else:
        out["seq_len"] = _align(max(64, int(round(int(out["seq_len"]) * factor))), 64, 64)
        out["seq_len_kv"] = _align(max(128, int(round(int(out["seq_len_kv"]) * factor))), 64, 128)
        if int(out["seq_len_kv"]) < int(out["seq_len"]):
            out["seq_len_kv"] = int(out["seq_len"])
    return out


def _grow_tiles(shape: dict[str, Any]) -> dict[str, Any]:
    out = dict(shape)
    if "n" in out:
        out["n"] = _align(int(out["n"]) * 2, 128, 256)
    elif "r" in out:
        out["r"] = _align(int(out["r"]) * 2, 128, 256)
    elif "mn" in out:
        out["mn"] = _align(int(out["mn"]) * 2, 4, 512)
    return out


def _noncluster_ordinary_2cta(case: dict[str, Any]) -> bool:
    """Non-cluster GemmDesc case that must run as two ordinary CTAs at num_sms=2.

    Vendor config.hpp:47 requires even num_sms for GemmDesc kernels, so num_sms=1
    is illegal here. num_sms=2 is legal; the launch must stay cluster_dim=1.
    """
    if case.get("uses_cluster"):
        return False
    return int(case.get("target_ctas") or 0) == 2 and bool(case.get("plain_2cta_fallback"))


def _command(
    *,
    case: dict[str, Any],
    package: Path,
    cache: Path,
    output: Path,
    attention_manifest: Path,
    runtime_manifest: Path,
    pilot: bool,
) -> list[str]:
    env_python = [sys.executable]
    if case["suite_part"] == "attention":
        selected = {
            "schema_version": "deepgemm-attention-cluster-sm-v1",
            "selection": {
                "warmups": 3,
                "trials": 9,
                "maximum_cv_percent": 100.0,
            },
            "cases": [case],
        }
        selected_path = output.with_suffix(".attention.manifest.json")
        _write_json(selected_path, selected)
        cmd = env_python + [
            str(ROOT / "harness/deepgemm_attention_runner.py"),
            "--manifest", str(selected_path),
            "--package", str(package),
            "--cache-root", str(cache),
            "--output", str(output),
            "--case-id", case["case_id"],
            "--mode", "cluster_sm",
            "--num-sms", str(int(case["target_ctas"])),
            "--round", "1",
        ]
        if pilot:
            cmd.append("--pilot")
        return cmd
    cmd = env_python + [
        "-m", "tools.microbench.deepgemm_sm90_runner",
        "--manifest", str(runtime_manifest),
        "--package", str(package),
        "--cache-root", str(cache),
        "--output", str(output),
        "--case-id", case["case_id"],
        "--class-id", str(case["class_id"]),
        "--shape-json", json.dumps(case["shape"], separators=(",", ":")),
        "--mode", "cluster_sm",
        "--num-sms", str(int(case["target_ctas"])),
        "--uses-cluster", "true" if case.get("uses_cluster") else "false",
        "--split", "collection",
    ]
    if pilot:
        cmd.append("--pilot")
    return cmd


@contextlib.contextmanager
def _exclusive_gpu(work: Path):
    path = Path(os.environ.get("CLUSTER_SM_GPU_LOCK") or work / "tmp" / "gpu.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("CMD " + " ".join(cmd) + "\n")
        handle.flush()
        completed = subprocess.run(
            cmd, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
        handle.write(f"\nEXIT {completed.returncode}\n")
    return int(completed.returncode)


def _result_ok(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def run_one(case: dict[str, Any], *, work: Path, package: Path, runtime_manifest: Path) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    current = dict(case)
    env = dict(os.environ)
    env["GEMM_DELIVERY_GPU"] = env.get("GEMM_DELIVERY_GPU", "h100")
    env["DEEPGEMM_DELIVERY_RAW_PER_SAMPLE"] = env.get("DEEPGEMM_DELIVERY_RAW_PER_SAMPLE", "4")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    harness = ROOT / "harness"

    def execute(tag: str, pilot: bool) -> dict[str, Any]:
        attempt_dir = work / "attempts" / case["case_id"] / tag
        output = attempt_dir / "result.json"
        cache = attempt_dir / "cache"
        log = attempt_dir / "run.log"
        cmd = _command(
            case=current,
            package=package,
            cache=cache,
            output=output,
            attention_manifest=ROOT / "manifests/deepgemm_attention_14.json",
            runtime_manifest=runtime_manifest,
            pilot=pilot,
        )
        cwd = harness if current["suite_part"] == "core" else harness
        with _exclusive_gpu(work):
            code = _run(cmd, cwd=cwd, env=env, log=log)
        payload = _result_ok(output)
        row = {
            "tag": tag,
            "exit_code": code,
            "log": str(log),
            "result": str(output) if output.is_file() else None,
            "parameters": {
                "shape": current.get("shape"),
                "seq_len": current.get("seq_len"),
                "seq_len_kv": current.get("seq_len_kv"),
                "batch_size": current.get("batch_size"),
                "avg_kv": current.get("avg_kv"),
                "target_ctas": current.get("target_ctas"),
            },
            "status": None if payload is None else payload.get("status"),
            "median_cycles": None if payload is None else (payload.get("timing") or {}).get("median_cycles"),
            "topology_status": None if payload is None else (payload.get("topology") or {}).get("status"),
            "correctness_status": None if payload is None else (payload.get("correctness") or {}).get("status"),
        }
        attempts.append(row)
        _write_json(attempt_dir / "attempt.json", row)
        return {"row": row, "payload": payload}

    if current.get("vendor_legal_num_sms") is False and not current.get("plain_2cta_fallback"):
        probe = execute("illegal_num_sms_probe", True)
        log_text = ""
        log_path = Path(probe["row"]["log"])
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        asserted = "num_sms % 2" in log_text or "num_sms % 2 == 0" in log_text
        reason = str(current.get("skip_reason") or (
            "single-CTA illegal: vendor GemmDesc requires even num_sms"
        ))
        reason += (
            f"; probe_exit={probe['row']['exit_code']}; "
            f"assert_seen={asserted}; log={probe['row']['log']}"
        )
        summary = {
            "case_id": case["case_id"],
            "source_case_id": case["source_case_id"],
            "status": "skipped",
            "reason": reason,
            "attempts": attempts,
            "final_parameters": probe["row"]["parameters"],
            "mechanisms_preserved": False,
            "mechanism_evidence": (
                "did not relax config.hpp GemmDesc even-SM assert; "
                f"modules={case.get('implementation_modules')}; {reason}"
            ),
            "formal": {
                "result": probe["row"]["result"],
                "log": probe["row"]["log"],
                "exit_code": probe["row"]["exit_code"],
                "status": "skipped",
                "payload": probe["payload"] or {},
            },
        }
        _write_json(work / "summaries" / f"{case['case_id']}.json", summary)
        return summary

    if current.get("uses_cluster") and current.get("heuristic_cluster_at_2") is False:
        probe = execute("cluster_mechanism_probe", True)
        payload = probe["payload"] or {}
        topo = payload.get("topology") or {}
        counts = topo.get("cluster_cta_counts") or []
        flat = []
        for row in counts:
            if isinstance(row, list):
                flat.extend(row)
            else:
                flat.append(row)
        observed = max((int(v) for v in flat), default=0)
        if observed != 2:
            reason = str(current.get("skip_reason") or (
                "vendor get_best_config at num_sms=2 did not select a 2CTA cluster"
            ))
            reason += (
                f"; observed_cluster_cta_counts={counts}; cta_count={topo.get('cta_count')}; "
                f"sm_ids={topo.get('sm_ids')}; jit={topo.get('jit_cluster')}; "
                f"log={probe['row']['log']}"
            )
            summary = {
                "case_id": case["case_id"],
                "source_case_id": case["source_case_id"],
                "status": "skipped",
                "reason": reason,
                "attempts": attempts,
                "final_parameters": probe["row"]["parameters"],
                "mechanisms_preserved": False,
                "mechanism_evidence": reason,
                "formal": {
                    "result": probe["row"]["result"],
                    "log": probe["row"]["log"],
                    "exit_code": probe["row"]["exit_code"],
                    "status": "skipped",
                    "payload": payload,
                },
            }
            _write_json(work / "summaries" / f"{case['case_id']}.json", summary)
            return summary

    last = execute("pilot0", True)
    for index in range(1, 7):
        payload = last["payload"]
        if payload is None:
            break
        topology = (payload.get("topology") or {}).get("status")
        correctness = (payload.get("correctness") or {}).get("status")
        cycles = (payload.get("timing") or {}).get("median_cycles")
        if topology != "pass":
            if current.get("uses_cluster") and current.get("heuristic_cluster_at_2") and current["suite_part"] == "core" and index <= 2:
                current["shape"] = _grow_tiles(dict(current.get("shape") or {}))
                last = execute(f"topo{index}", True)
                continue
            break
        if correctness != "pass":
            break
        if not isinstance(cycles, (int, float)):
            break
        if MIN_CYCLES <= float(cycles) <= MAX_CYCLES:
            break
        if _noncluster_ordinary_2cta(current):
            # Non-cluster GemmDesc cases may fall below 100k: user-authorized
            # DG window is (0, 500000]. Shrink only while above the ceiling;
            # never grow a shape that already fits, or the scaler oscillates
            # between two legal shapes and the formal lands on the big one.
            if float(cycles) <= MAX_CYCLES:
                break
            # Only ever shrink: a factor > 1 would grow the shape back.
            factor = min(1.0, TARGET_CYCLES / max(float(cycles), 1.0))
            before = dict(current.get("shape") or {})
            current["shape"] = _scale_core(
                before, factor, str(current["class_id"]), target_ctas=int(current["target_ctas"])
            )
            if dict(current.get("shape") or {}) == before:
                break
            last = execute(f"tune{index}", True)
            continue
        factor = TARGET_CYCLES / max(float(cycles), 1.0)
        if current["suite_part"] == "attention":
            before = {k: current.get(k) for k in ("seq_len", "seq_len_kv", "batch_size", "avg_kv")}
            current = _scale_attention(current, factor)
            after = {k: current.get(k) for k in ("seq_len", "seq_len_kv", "batch_size", "avg_kv")}
            if after == before:
                break
        else:
            before = dict(current.get("shape") or {})
            current["shape"] = _scale_core(before, factor, str(current["class_id"]), target_ctas=int(current["target_ctas"]))
            if dict(current.get("shape") or {}) == before:
                break
        last = execute(f"tune{index}", True)

    formal = execute("formal", False)
    formal_payload = formal["payload"] or {}
    summary = {
        "case_id": case["case_id"],
        "source_case_id": case["source_case_id"],
        "attempts": attempts,
        "final_parameters": formal["row"]["parameters"],
        "formal": {
            "result": formal["row"]["result"],
            "log": formal["row"]["log"],
            "exit_code": formal["row"]["exit_code"],
            "status": formal["row"]["status"],
            "payload": formal_payload,
        },
    }
    _write_json(work / "summaries" / f"{case['case_id']}.json", summary)
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=ROOT / "manifests/deepgemm_core_runtime_manifest.json",
    )
    args = parser.parse_args()
    manifest = _load_json(args.manifest)
    matches = [row for row in manifest["cases"] if row["case_id"] == args.case_id]
    if len(matches) != 1:
        raise SystemExit(f"unknown case {args.case_id}")
    summary = run_one(
        matches[0], work=args.work, package=args.package, runtime_manifest=args.runtime_manifest
    )
    print(json.dumps({
        "case_id": summary["case_id"],
        "status": summary.get("status") or summary["formal"]["status"],
        "exit_code": summary["formal"]["exit_code"],
        "median_cycles": (summary["formal"]["payload"].get("timing") or {}).get("median_cycles")
        if summary["formal"]["payload"] else None,
        "attempts": len(summary["attempts"]),
    }, sort_keys=True))
    if summary.get("status") == "skipped":
        return 0
    return 0 if summary["formal"]["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
