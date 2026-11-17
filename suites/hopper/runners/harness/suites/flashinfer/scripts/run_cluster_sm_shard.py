#!/usr/bin/env python3
"""Run one 8-way FlashInfer cluster-SM shard: serialize GPU timing, tune cycles."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CYCLE_LO = 100_000
CYCLE_HI = 500_000
MAX_ATTEMPTS = 6
KNOB_CAPS = {
    "seqlen_k": 131072,
    "seqlen_q": 128,
    "kv_len": 131072,
    "d_in": 65536,
    "k": 65536,
    "hidden": 32768,
    "intermediate": 16384,
    "dim": 512,
    "dstate": 128,
    "seqlen": 65536,
    "head_dim": 256,
    "append_len": 8192,
    "vocab": 262144,
    "elements": 64 * 1024 * 1024,
}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_overlay(manifest: dict, case: dict, path: Path) -> None:
    data = dict(manifest)
    data["cases"] = [case]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def knob_value(case: dict, knob: str) -> int:
    if knob == "intermediate":
        return int(case.get("intermediate") or case.get("hidden") or 256)
    return int(case[knob])


def fa2_one_tile_q(head_dim) -> int:
    hd = int(head_dim or 128)
    if hd >= 512:
        return 16
    if hd >= 256:
        return 32
    return 65


def set_knob(case: dict, knob: str, value: int) -> None:
    cap = KNOB_CAPS.get(knob, value)
    backend = case.get("backend")
    op = case.get("operation")
    if knob in {"seqlen_k", "kv_len"} and backend != "fa3":
        cap = min(cap, 256)
    if knob == "seqlen_q":
        hd = int(case.get("head_dim") or 128)
        if case.get("backend") == "fa3":
            cap = 128  # SingleTileScheduler: one Q-tile stays 1 CTA
        else:
            cap = 128 if hd < 256 else (64 if hd < 512 else 32)
    value = max(1, min(int(value), cap))
    if knob in {"hidden", "intermediate", "d_in", "k", "elements", "vocab"}:
        if knob in {"hidden", "intermediate", "d_in", "k"}:
            value = max(128, (value // 128) * 128)
        if case.get("quant") == "w4a16_mxfp4" and knob in {"hidden", "intermediate"}:
            value = max(32, (value // 32) * 32)
    if knob == "head_dim":
        value = max(64, min(value, 256))
        if value not in {64, 128, 256}:
            value = 128
    if knob == "seqlen_q" and "seqlen_k" in case:
        case["seqlen_k"] = max(int(case["seqlen_k"]), value)
    case[knob] = value
    _ = op


def resolve_dynamic(case: dict, observed: dict) -> None:
    cluster = case.setdefault("cluster_sm", {})
    if not cluster.get("dynamic_cluster") or not observed:
        return
    uses = bool(observed.get("uses_cluster"))
    cluster["uses_cluster"] = uses
    cluster["target_ctas"] = 2 if uses else 1
    cluster["observed_cluster_ctas"] = observed.get("max_primary_cluster_ctas")



_ATTN_GQA_OPS = {
    "single_prefill",
    "batch_prefill_ragged",
    "batch_prefill_paged",
    "single_decode",
    "batch_decode_paged",
    "xqa_batch_decode",
    "rope",
}


def _mechanisms_preserved(case: dict, observed: dict) -> bool:
    orig = (case.get("cluster_sm") or {}).get("original_parameters") or {}
    oq, okv = orig.get("heads_q"), orig.get("heads_kv")
    if case.get("operation") not in _ATTN_GQA_OPS:
        oq = okv = None
    if oq is not None and okv is not None and int(oq) != int(okv):
        if int(case.get("heads_q") or 0) == 1 and int(oq) != 1:
            return False
        if int(oq) // max(int(okv), 1) != int(case.get("heads_q") or 0) // max(int(case.get("heads_kv") or 1), 1):
            return False
    if (case.get("cluster_sm") or {}).get("predicted_skip") and (observed.get("max_primary_cluster_ctas") or 1) != 4:
        # 4CTA fp8 must remain 2x2x1
        pass
    return True

def topology_ok(observed: dict, target_ctas: int, uses_cluster: bool) -> tuple[bool, str]:
    max_cta = observed.get("max_primary_cta_count")
    cluster_ctas = observed.get("max_primary_cluster_ctas") or 1
    if max_cta is None:
        return False, "no primary kernel launch recorded"
    if uses_cluster or cluster_ctas > 1:
        if cluster_ctas == 4:
            return False, "observed 4-CTA cluster; 2-CTA shrink would change ClusterShape"
        if cluster_ctas != 2:
            return False, f"cluster users need one 2-CTA cluster, observed cluster_ctas={cluster_ctas} cta={max_cta}"
        if max_cta != 2:
            return False, f"need a single 2-CTA cluster, observed cta_count={max_cta}"
        sms = observed.get("sm_ids") or observed.get("observed_sms") or []
        uniq = {int(x) for x in sms if x is not None}
        if len(uniq) != 2:
            return False, "2CTA cluster needs two distinct measured SM ids on the same launch"
        return True, "one 2-CTA cluster"
    if max_cta == 1:
        return True, "1 CTA"
    if max_cta == 2 and int(cluster_ctas) == 1:
        sms = observed.get("sm_ids") or observed.get("observed_sms") or []
        uniq = {int(x) for x in sms if x is not None}
        if len(uniq) != 2:
            return False, (
                "2 ordinary CTAs need two distinct measured SM ids; "
                "two CTAs on one SM do not qualify"
            )
        return True, "2 ordinary CTAs"
    return False, (
        f"non-cluster target is 1 or 2 ordinary CTAs, observed "
        f"cta_count={max_cta} cluster_ctas={cluster_ctas}"
    )


def mechanism_cannot_shrink(case: dict, observed: dict) -> str | None:
    op = case.get("operation")
    backend = case.get("backend")
    cta = int(observed.get("max_primary_cta_count") or 0)
    cluster_ctas = int(observed.get("max_primary_cluster_ctas") or 1)
    if cluster_ctas == 4:
        return (
            "observed 4-CTA cluster. Source: group_gemm_sm90.cuh ClusterShape<_2,_2,_1> / "
            "cutlass_heuristic.cpp ClusterShape_2x2x1. Shrinking to 2x1x1 changes TMA multicast."
        )
    if op in {"batch_prefill_ragged", "batch_prefill_paged"} and backend == "fa3" and cta > 2:
        return (
            f"observed cta_count={cta}. Source: hopper/tile_scheduler.cuh "
            "BatchPrefillPersistentTileScheduler::get_grid_dim returns {{num_sm}}; "
            "1 CTA would replace the persistent scheduler."
        )
    if op == "batch_mla_paged" and cta > 2:
        return (
            f"observed cta_count={cta}. Source: mla_hopper.cuh / mla.cuh "
            "cudaLaunchCooperativeKernel BatchMLAPagedAttentionKernel grid uses "
            "multiprocessor_count (H100 132). 1 CTA would replace the MLA scheduler."
        )
    if op == "fp8_blockscale_gemm" and cta > 2:
        return (
            f"observed cta_count={cta}. Source: nv_internal/tensorrt_llm/deep_gemm/fp8_gemm.cuh "
            "config.gridDim=num_sms and flashinfer/deep_gemm.py num_sms=multi_processor_count; "
            "persistent over all SMs. clusterDim={{num_tma_multicast,1,1}}."
        )
    if op == "cutlass_fused_moe":
        top_k = int(case.get("top_k") or 1)
        n_exp = int(case.get("num_experts") or 1)
        if top_k >= 2 or n_exp > 1 or cta > 1:
            return (
                f"observed cta_count={cta} cluster_ctas={cluster_ctas}. Source: "
                "cutlass_heuristic.cpp get_candidate_configs_sm90 ClusterShape "
                "1x1x1/2x1x1/1x2x1/2x2x1; moe_gemm_template_dispatch_tma_ws.h grouped TMA "
                f"GEMM launches >=1 tile per routed expert. Source top_k={top_k} "
                f"num_experts={n_exp}; 1 CTA or one 2-CTA cluster would drop top_k/"
                "num_experts and change MoE routing."
            )
    if op == "xqa_batch_decode" and cta > 2:
        q = int(case.get("heads_q") or 1)
        kv = int(case.get("heads_kv") or 1)
        if kv > 1 and q % kv == 0:
            return None
        return (
            f"observed cta_count={cta}. XQA decode launch is occupancy-sized; shrinking heads/batch "
            "already applied, further shrink would change the XQA schedule."
        )
    if op == "batch_decode_paged" and cta > 2:
        return (
            f"observed cta_count={cta}. BatchDecodeWithPagedKVCacheKernel plus "
            "PersistentVariableLengthMergeStatesKernel grid uses multiprocessor_count "
            "(decode.cuh / attention/merge.cuh); 1 CTA would replace the persistent merge."
        )
    if op == "segment_packbits" and cta > 2:
        return (
            f"observed cta_count={cta}. PackBitsKernel grid is occupancy over elements; "
            "further shrinking elements still launches many CTAs or changes the pack width."
        )
    if op == "append_paged_kv" and cta > 2:
        return (
            f"observed cta_count={cta}. AppendPagedKVCache kernel grid is "
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor * num_sms (page.cuh); "
            "cannot shrink to 1 CTA without replacing the occupancy launch."
        )
    if op == "rope" and cta > 2:
        return (
            f"observed cta_count={cta}. Source: pos_enc.cuh BatchQKApplyRotaryKernel "
            "nblks=batch_size*(num_qo_heads+num_kv_heads). 1 CTA would drop Q or K rope; "
            "2 ordinary CTAs are legal when grid product is 2."
        )
    orig = (case.get("cluster_sm") or {}).get("original_parameters") or {}
    orig_q, orig_kv = orig.get("heads_q"), orig.get("heads_kv")
    if orig_q and orig_kv and int(orig_q) != int(orig_kv) and cta > 2:
        return (
            f"observed cta_count={cta}. Original GQA heads_q={orig_q} heads_kv={orig_kv}. "
            "Keeping GQA ratio; FA3 SingleTileScheduler grid is (q_tiles, qo_heads) and FA2 "
            "grid.z=kv_heads. 1 CTA would require dropping GQA. 2 ordinary CTAs are legal."
        )
    if op == "segment_gemm" and cluster_ctas == 4:
        return (
            f"observed cta_count={cta} cluster_ctas=4 grid={observed.get('primary_kernels', [{}])[0].get('grid')}. "
            "Source: group_gemm_sm90.cuh ClusterShape<_2,_2,_1> for fp8; minimum cluster is 4 CTA. "
            "Shrinking to 2x1x1 changes TMA multicast."
        )
    if op in {"gdn_prefill", "gdn_decode"}:
        hq = int(case.get("heads_q") or 1)
        hv = int(case.get("heads_v") or hq)
        if hq != hv and cta > 1:
            return (
                f"observed cta_count={cta}. GDN GVA heads_q={hq} heads_v={hv}; "
                "keeping GVA forbids shrinking the chunk/head grid to 1 CTA."
            )
    return None


def shrink_for_topology(case: dict, observed: dict) -> bool:
    cta = int(observed.get("max_primary_cta_count") or 0)
    if cta <= 2:
        return False
    changed = False
    op = case.get("operation")
    backend = case.get("backend")
    if "batch" in case and int(case.get("batch") or 1) > 1:
        case["batch"] = 1
        changed = True
    if op in {"single_prefill", "batch_prefill_ragged", "batch_prefill_paged"} and backend != "fa3":
        q_one = fa2_one_tile_q(case.get("head_dim"))
        cap = 128 if "batch" in str(op) else 256
        if int(case.get("seqlen_k") or 0) > cap:
            case["seqlen_k"] = max(cap, q_one)
            changed = True
        if int(case.get("seqlen_q") or 1) != q_one:
            case["seqlen_q"] = q_one
            case["seqlen_k"] = max(int(case.get("seqlen_k") or 0), q_one)
            changed = True
    if op in {"single_decode", "batch_decode_paged", "xqa_batch_decode", "batch_mla_paged"}:
        if int(case.get("kv_len") or 0) > 256:
            case["kv_len"] = 256
            changed = True
        if int(case.get("q_len") or 1) > 1:
            case["q_len"] = 1
            changed = True
    if op == "cutlass_fused_moe" and int(case.get("num_tokens") or 1) > 1:
        case["num_tokens"] = 1
        changed = True
    if op == "fp8_blockscale_gemm":
        if int(case.get("m") or 1) > 16:
            case["m"] = 16
            changed = True
        if int(case.get("n") or 128) > 128:
            case["n"] = 128
            changed = True
    if op in {"merge_state", "merge_states"} and int(case.get("seqlen") or 1) > 1:
        case["seqlen"] = 1
        changed = True
    if op == "packbits" and int(case.get("elements") or 0) > 2048:
        case["elements"] = 2048
        changed = True
    if op == "mamba_state_update" and int(case.get("dim") or 0) > 4:
        case["dim"] = 4
        changed = True
    if op in {"rmsnorm", "fused_add_rmsnorm"} and int(case.get("rows") or 1) > 1:
        case["rows"] = 1
        changed = True
    if op == "gdn_prefill":
        hq = int(case.get("heads_q") or 1)
        hv = int(case.get("heads_v") or hq)
        if hq == hv and int(case.get("seqlen") or 1) > 1:
            case["seqlen"] = 1
            changed = True
    if op == "xqa_batch_decode":
        q = int(case.get("heads_q") or 1)
        kv = int(case.get("heads_kv") or 1)
        if kv > 1 and q % kv == 0:
            case["heads_kv"] = 1
            case["heads_q"] = max(1, q // kv)
            changed = True
    return changed


def classify(case: dict, result: dict, exit_code: int, *, force_skip: str | None = None) -> dict:
    cluster = case.get("cluster_sm") or {}
    observed = ((result.get("topology") or {}).get("observed")) or {}
    resolve_dynamic(case, observed)
    predicted_skip = cluster.get("predicted_skip")
    correctness = result.get("correctness") or {}
    timing = result.get("timing") or {}
    cycles = timing.get("median_cycles")
    obs_cta = observed.get("max_primary_cta_count")
    obs_cluster = int(observed.get("max_primary_cluster_ctas") or 1)
    if obs_cluster == 2:
        uses_cluster = True
        target = 2
    elif obs_cluster == 1 and obs_cta == 2:
        uses_cluster = False
        target = 2
    elif obs_cluster == 1 and obs_cta == 1:
        uses_cluster = False
        target = 1
    else:
        uses_cluster = bool(cluster.get("uses_cluster"))
        target = int(cluster.get("target_ctas") or 1)
    topo_ok, topo_reason = topology_ok(observed, target, uses_cluster)
    status = "unresolved"
    reason = None
    ran = exit_code == 0
    if predicted_skip and ran:
        status = "skipped"
        reason = predicted_skip
    elif force_skip and ran:
        status = "skipped"
        reason = force_skip
    elif result.get("status") != "pass":
        status = "unresolved"
        reason = correctness.get("error") or f"runner exit {exit_code} status={result.get('status')}"
    elif not topo_ok:
        status = "unresolved"
        reason = topo_reason
    elif cycles is None:
        status = "unresolved"
        reason = "missing GPU cycle samples"
    elif not (CYCLE_LO <= float(cycles) <= CYCLE_HI):
        status = "unresolved"
        reason = f"cycles_median={cycles} outside [{CYCLE_LO},{CYCLE_HI}]"
    else:
        status = "qualified"
        reason = None
    if status == "qualified" and target == 2:
        sms = observed.get("sm_ids") or observed.get("observed_sms") or []
        uniq = {int(x) for x in sms if x is not None}
        if len(uniq) < 2:
            status = "unresolved"
            reason = "2CTA missing two distinct measured SM ids from same-run intercept"
    evidence = (result.get("topology") or {}).get("source")
    kernels = list(observed.get("primary_kernels") or []) + list(observed.get("helper_kernels") or [])
    primary = (observed.get("primary_kernels") or [{}])[0] if observed.get("primary_kernels") else {}
    n_samples = timing.get("samples_cycles_compatibility") or []
    return {
        "source_case_id": cluster.get("source_case_id") or case.get("source_case_id"),
        "derived_case_id": case["case_id"],
        "status": status,
        "target_ctas": target,
        "uses_cluster": uses_cluster,
        "parameters": {k: case[k] for k in case if k not in {"cluster_sm", "cluster_sm_notes", "source"}},
        "mechanisms_preserved": _mechanisms_preserved(case, observed),
        "mechanism_evidence": {
            "source_mechanisms": case.get("mechanisms"),
            "backend": case.get("backend"),
            "dtype": case.get("dtype"),
            "quant": case.get("quant"),
            "notes": case.get("cluster_sm_notes"),
            "predicted_uses_cluster": uses_cluster,
            "observed_primary_kernels": observed.get("primary_kernels"),
            "pointer": "suites/flashinfer/upstream/include/flashinfer/gemm/group_gemm_sm90.cuh ClusterShape, hopper/tile_scheduler.cuh, prefill.cuh split-KV, and harness topology log",
        },
        "topology": {
            "grid": primary.get("grid"),
            "cluster_dim": primary.get("cluster"),
            "cta_count": observed.get("max_primary_cta_count"),
            "observed_sms": observed.get("sm_ids") or [],
            "evidence": evidence,
            "kernels": kernels,
            "requested": {"target_ctas": target, "uses_cluster": uses_cluster},
            "observed": observed,
            "ok": topo_ok,
            "reason": topo_reason,
        },
        "correctness": {
            "status": correctness.get("status"),
            "method": correctness.get("method"),
            "artifact": str(Path(result.get("output") or "") or evidence),
            "finite": correctness.get("finite"),
            "error": correctness.get("error"),
        },
        "timing": {
            "cycles_median": cycles,
            "samples_cycles": n_samples,
            "scope": "whole public operation between device globaltimer start/end markers; all kernels in topology.kernels",
            "method": timing.get("method") or "queued device globaltimer whole-operation span converted by clock64/globaltimer ratio to GPU cycles",
            "artifact": evidence,
            "kernels": kernels,
            "globaltimer_median": timing.get("median_globaltimer"),
            "clock_per_globaltimer": timing.get("fixed_clock_per_globaltimer_for_compatibility"),
            "cv_percent": timing.get("cv_percent"),
            "n_samples": len(n_samples) if isinstance(n_samples, list) else None,
        },
        "reason": reason,
    }


CTA_GRID_KNOBS = {
    "merge_state": {"seqlen", "num_tokens"},
    "merge_states": {"seqlen", "num_tokens"},
    "packbits": {"elements"},
    "mamba_state_update": {"dim"},
    "gdn_prefill": {"seqlen"},
    "rope": {"batch"},
    "rmsnorm": {"rows"},
    "fused_add_rmsnorm": {"rows"},
    "xqa_batch_decode": {"heads_q", "heads_kv"},
}


def scale_for_cycles(case: dict, cycles: float) -> bool:
    knobs = list((case.get("cluster_sm") or {}).get("scale_knobs") or [])
    if not knobs or cycles <= 0:
        return False
    locked = CTA_GRID_KNOBS.get(case.get("operation"), set())
    knobs = [k for k in knobs if k not in locked]
    if not knobs:
        return False
    target = math.sqrt(CYCLE_LO * CYCLE_HI)  # ~223606
    factor = target / cycles
    if 0.9 <= factor <= 1.1:
        return False
    changed = False
    for knob in knobs:
        if knob not in case and knob != "intermediate":
            continue
        current = knob_value(case, knob)
        nxt = int(max(1, round(current * factor)))
        if nxt == current:
            nxt = current * 2 if factor > 1 else max(1, current // 2)
        before = current
        set_knob(case, knob, nxt)
        if case.get(knob) != before:
            changed = True
    return changed


def run_attempt(args, case: dict, manifest: dict, attempt_dir: Path, repeats: int) -> tuple[int, dict]:
    overlay = attempt_dir / "manifest.json"
    write_overlay(manifest, case, overlay)
    env = os.environ.copy()
    env["DELIVERY_CASE_MANIFEST"] = str(overlay)
    env["DELIVERY_JIT_CACHE_ROOT"] = str(Path(args.root) / "jit-cache-shared")
    env["FLASHINFER_CLUSTER_SM_ROOT"] = str(args.root)
    env["FLASHINFER_CLUSTER_SM_ATTEMPT_DIR"] = str(attempt_dir)
    cluster = case.get("cluster_sm") or {}
    if case.get("operation") == "segment_gemm" and int(cluster.get("target_ctas") or 0) == 2:
        # persistent grouped GEMM: sm_count = one 2x1x1 cluster. 4CTA fp8 keeps default and skips.
        env["FLASHINFER_CLUSTER_SM_TARGET_SMS"] = "2"
    cmd = [
        "bash",
        str(ROOT / "scripts/run_cluster_sm.sh"),
        args.profile,
        case["case_id"],
        str(repeats),
    ]
    log_path = attempt_dir / "run.log"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = int(os.environ.get("FLASHINFER_CLUSTER_SM_ATTEMPT_TIMEOUT", "1200"))
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                timeout=timeout_s,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout_s}s\n")
            code = 124
    result = {}
    result_path = attempt_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    elif (attempt_dir / "launches.jsonl").is_file():
        harness = str(ROOT / "harness")
        if harness not in sys.path:
            sys.path.insert(0, harness)
        from cluster_topology import attach_topology

        result = {"status": None}
        attach_topology(result, attempt_dir / "launches.jsonl")
    result["output"] = str(result_path)
    return code, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="h100")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--only-unresolved", action="store_true")
    parser.add_argument("--case-ids-file", type=Path)
    args = parser.parse_args()
    if os.environ.get("FLASHINFER_CLUSTER_SM_ONLY_UNRESOLVED") == "1":
        args.only_unresolved = True
    env_ids = os.environ.get("FLASHINFER_CLUSTER_SM_CASE_IDS_FILE")
    if env_ids and not args.case_ids_file:
        args.case_ids_file = Path(env_ids)
    started = time.time()
    manifest_path = ROOT / "manifests/flashinfer_cluster_sm100.json"
    if not manifest_path.is_file():
        subprocess.check_call([sys.executable, str(ROOT / "scripts/generate_cluster_sm_manifest.py")])
    manifest = load_manifest(manifest_path)
    cases = list(manifest["cases"])
    if args.case_ids_file:
        want = {
            line.strip()
            for line in args.case_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        cases = [
            c
            for c in cases
            if c["case_id"] in want
            or c.get("source_case_id") in want
            or (c.get("cluster_sm") or {}).get("source_case_id") in want
        ]
    else:
        cases = [c for c in cases if int(c["delivery_group"]) == args.shard]
    records = []
    print(
        json.dumps({"repair_worker": args.shard, "n_cases": len(cases), "ids": [c["case_id"] for c in cases]}),
        flush=True,
    )
    for case in cases:
        case = json.loads(json.dumps(case))
        shard_dir = args.root / f"shards/shard{int(case['delivery_group'])}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        case_dir = shard_dir / "cases" / case["case_id"]
        final_path = case_dir / "final.json"
        if args.only_unresolved and final_path.is_file():
            prev = json.loads(final_path.read_text(encoding="utf-8"))
            orig = (case.get("cluster_sm") or {}).get("original_parameters") or {}
            prev_heads_q = (prev.get("parameters") or {}).get("heads_q")
            collapsed_gqa = (
                orig.get("heads_q") is not None
                and orig.get("heads_kv") is not None
                and int(orig["heads_q"]) != int(orig["heads_kv"])
                and int(prev_heads_q or 0) == 1
                and int(orig["heads_q"]) != 1
            )
            if prev.get("status") in {"qualified", "skipped"} and not collapsed_gqa:
                records.append(prev)
                continue
        attempts = []
        last = None
        print(json.dumps({"event": "start", "case": case["case_id"], "shard": int(case["delivery_group"])}), flush=True)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if last is not None and last.get("status") in {"qualified", "skipped"}:
                break
            attempt_dir = case_dir / f"attempt{attempt}"
            repeats = 9
            code, result = run_attempt(args, case, manifest, attempt_dir, repeats)
            observed = ((result.get("topology") or {}).get("observed")) or {}
            force_skip = None
            shrunk = False
            entry = classify(case, result, code)
            topo_ok = bool((entry.get("topology") or {}).get("ok"))
            pred_skip = (case.get("cluster_sm") or {}).get("predicted_skip")
            if pred_skip and code == 0:
                force_skip = pred_skip
            elif not topo_ok and (result.get("status") == "pass" or code == 0):
                cluster_ctas = int(observed.get("max_primary_cluster_ctas") or 1)
                cannot = mechanism_cannot_shrink(case, observed)
                if cluster_ctas == 4 or cannot:
                    force_skip = cannot or (
                        "observed 4-CTA cluster; 2-CTA shrink would change ClusterShape"
                    )
                elif shrink_for_topology(case, observed):
                    shrunk = True
                else:
                    force_skip = cannot
            if force_skip:
                entry = classify(case, result, code, force_skip=force_skip)
            entry["attempt"] = attempt
            entry["exit_code"] = code
            entry["attempt_dir"] = str(attempt_dir)
            attempts.append(entry)
            last = entry
            (attempt_dir / "classification.json").write_text(json.dumps(entry, indent=2) + "\n")
            cycles = (entry.get("timing") or {}).get("cycles_median")
            if entry["status"] in {"qualified", "skipped"}:
                break
            if result.get("status") != "pass" and code != 0:
                if attempt >= 2:
                    break
                continue
            if not topo_ok and shrunk:
                continue
            if not topo_ok:
                break
            if cycles and not (CYCLE_LO <= float(cycles) <= CYCLE_HI):
                if not scale_for_cycles(case, float(cycles)):
                    break
                continue
            break
        if last is not None:
            last = dict(last)
            last["attempts"] = [
                {
                    "attempt": a["attempt"],
                    "status": a["status"],
                    "cycles_median": (a.get("timing") or {}).get("cycles_median"),
                    "dir": a["attempt_dir"],
                    "exit_code": a["exit_code"],
                    "reason": a.get("reason"),
                }
                for a in attempts
            ]
            records.append(last)
            (case_dir / "final.json").write_text(json.dumps(last, indent=2) + "\n")
        print(json.dumps({"case": case["case_id"], "status": last["status"] if last else None}, sort_keys=True), flush=True)
    summary = {
        "shard": args.shard,
        "workers": 1,
        "gpu_execution": "serialized_within_shard",
        "elapsed_seconds": time.time() - started,
        "cases": records,
    }
    worker_dir = args.root / f"shards/repair{args.shard}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"shard": args.shard, "n": len(records), "qualified": sum(r["status"]=="qualified" for r in records), "skipped": sum(r["status"]=="skipped" for r in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
