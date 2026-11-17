#!/usr/bin/env python3
"""Derive flashinfer_cluster_sm100.json from the registered 100-case full-GPU set."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/flashinfer_100.json"
OUTPUT = ROOT / "manifests/flashinfer_cluster_sm100.json"

CLUSTER_2CTA = {
    "flashinfer_segment_gemm_01",
    "flashinfer_segment_gemm_02",
    "flashinfer_segment_gemm_04",
    "flashinfer_segment_gemm_05",
}
CLUSTER_4CTA = {
    "flashinfer_segment_gemm_03",
    "flashinfer_segment_gemm_06",
}
DYNAMIC = {
    "flashinfer_cutlass_fused_moe_01",
    "flashinfer_cutlass_fused_moe_02",
    "flashinfer_cutlass_fused_moe_03",
    "flashinfer_cutlass_fused_moe_04",
    "flashinfer_cutlass_fused_moe_05",
    "flashinfer_cutlass_fused_moe_06",
    "flashinfer_cutlass_fused_moe_07",
    "flashinfer_cutlass_fused_moe_08",
    "flashinfer_fp8_blockscale_gemm_01",
    "flashinfer_fp8_blockscale_gemm_02",
    "flashinfer_fp8_blockscale_gemm_03",
    "flashinfer_fp8_blockscale_gemm_04",
}

# Native sequential knobs used to land GPU cycles in [1e5, 5e5] without padding loops.
ATTN_K = 16384
FA2_KV = 256  # prefill.cuh: chunk_size >= 256; kv_len<=256 => num_chunks=1
DECODE_K = 256  # decode.cuh: seq_len<=256 launches grid (1, num_kv_heads)
GEMM_K = 8192
FP8_K = 16384
HIDDEN = 8192


def fa2_one_tile_q(head_dim) -> int:
    hd = int(head_dim or 128)
    if hd >= 512:
        return 16
    if hd >= 256:
        return 32
    return 65  # FA2DetermineCtaTileQ: packed_qo>64 and hd<256 => CTA_TILE_Q=128, still 1 tile


def gqa_note(orig_q, orig_kv, scaled_q, scaled_kv, *, fa3: bool) -> dict:
    if fa3:
        reason = (
            "SingleTileScheduler grid is (q_tiles, qo_heads); keeping GQA ratio forbids 1 CTA"
            if orig_q != orig_kv
            else "SingleTileScheduler grid is (q_tiles, qo_heads); 1 CTA requires one Q head"
        )
    else:
        reason = (
            "FA2 grid is (q_tiles, kv_chunks, kv_heads); keeping GQA ratio forbids 1 CTA"
            if orig_q != orig_kv
            else (
                "FA2 SinglePrefillWithKVCacheKernel grid is (q_tiles, kv_chunks, kv_heads); "
                "1 CTA requires one Q tile, one KV head, and kv_len<=256 so split-KV is off"
            )
        )
    return {
        "original_heads_q": orig_q,
        "original_heads_kv": orig_kv,
        "scaled_heads_q": scaled_q,
        "scaled_heads_kv": scaled_kv,
        "reason": reason,
    }


def scale_case(src: dict) -> dict:
    case = deepcopy(src)
    source_id = src["case_id"]
    op = src["operation"]
    backend = src.get("backend")
    case["source_case_id"] = source_id
    case["case_id"] = f"{source_id}_cluster_sm"
    predicted_skip = None
    target_ctas = 1
    uses_cluster = False
    knobs: list[str] = []

    if source_id in CLUSTER_2CTA:
        target_ctas = 2
        uses_cluster = True
        case["groups"] = 2 if src.get("unequal_segments") else 1
        case["tokens_per_group"] = 128
        case["d_out"] = 128
        case["d_in"] = GEMM_K
        knobs = ["d_in"]
    elif source_id in CLUSTER_4CTA:
        target_ctas = 4
        uses_cluster = True
        case["groups"] = 2 if src.get("unequal_segments") else 1
        case["tokens_per_group"] = 256
        case["d_out"] = 128
        case["d_in"] = GEMM_K
        knobs = ["d_in"]
        predicted_skip = (
            "compile-time ClusterShape<_2,_2,_1> in group_gemm_sm90.cuh; "
            "minimum launch is one 4-CTA cluster. Shrinking to 2x1x1 changes TMA multicast."
        )
    elif op in {"single_prefill", "batch_prefill_ragged", "batch_prefill_paged"}:
        fa3 = backend == "fa3"
        orig_q, orig_kv = int(src.get("heads_q") or 1), int(src.get("heads_kv") or 1)
        gqa = orig_q != orig_kv
        if not gqa:
            case["heads_q"] = 1
            case["heads_kv"] = 1
        # GQA: keep original heads (ratio). 1 CTA is then scheduler-impossible.
        note = gqa_note(
            orig_q, orig_kv, int(case["heads_q"]), int(case["heads_kv"]), fa3=fa3
        )
        if "batch" in case:
            case["batch"] = 1
        if fa3 and op == "single_prefill":
            # SingleTileScheduler grid is (q_tiles, qo_heads). CTA_Q is 128 or 192
            # (prefill_sm90.cuh). One tile + one head is still 1 CTA; seqlen_q=1
            # leaves cycles ~30k even at seqlen_k=131072.
            case["seqlen_q"] = 128
            case["seqlen_k"] = ATTN_K
            knobs = ["seqlen_k", "seqlen_q"]
        elif fa3:
            case["seqlen_q"] = 1
            case["seqlen_k"] = ATTN_K
            knobs = ["seqlen_k"]
            case.setdefault("cluster_sm_notes", {})["scheduler"] = (
                "FA3 BatchPrefillPersistentTileScheduler::get_grid_dim returns {num_sm} "
                "(tile_scheduler.cuh); 1 CTA would replace the persistent scheduler."
            )
        else:
            q_len = fa2_one_tile_q(src.get("head_dim"))
            case["seqlen_q"] = q_len
            case["seqlen_k"] = max(128 if "batch" in op else FA2_KV, q_len)
            knobs = ["seqlen_q"]
            case.setdefault("cluster_sm_notes", {})["split_kv"] = (
                "prefill.cuh: chunk_size=max(ceil_div(kv_len, max_num_kv_chunks), 256); "
                "num_chunks<=1 (or tmp==nullptr) launches grid (q_tiles, 1, kv_heads)"
            )
        case.setdefault("cluster_sm_notes", {})["heads"] = note
        if gqa:
            predicted_skip = (
                f"GQA heads {orig_q}:{orig_kv}; "
                "FA3 SingleTileScheduler grid is (q_tiles, qo_heads) and FA2 grid is "
                "(q_tiles, kv_chunks, kv_heads). Keeping GQA ratio forbids 1 CTA."
            )
    elif op in {"single_decode", "batch_decode_paged", "xqa_batch_decode"}:
        orig_q, orig_kv = int(src.get("heads_q") or 1), int(src.get("heads_kv") or 1)
        if orig_q == orig_kv:
            case["heads_q"] = 1
            case["heads_kv"] = 1
        case["kv_len"] = DECODE_K
        if "batch" in case:
            case["batch"] = 1
        knobs = ["kv_len"]
        case.setdefault("cluster_sm_notes", {})["heads"] = gqa_note(
            orig_q,
            orig_kv,
            int(case["heads_q"]),
            int(case["heads_kv"]),
            fa3=backend == "fa3",
        )
        if orig_q != orig_kv and orig_kv > 1:
            predicted_skip = (
                f"GQA heads {orig_q}:{orig_kv}; decode grid is dim3(1, num_kv_heads) "
                "(decode.cuh). Keeping GQA ratio forbids 1 CTA."
            )
        case["cluster_sm_notes"]["split_kv"] = (
            "decode.cuh SingleDecodeWithKVCacheDispatched: seq_len<=256 launches dim3(1, num_kv_heads)"
        )
    elif op == "batch_mla_paged":
        case["batch"] = 1
        case["q_len"] = 1
        case["kv_len"] = FA2_KV if backend != "fa3" else ATTN_K
        knobs = ["kv_len"]
        if backend == "fa3":
            case.setdefault("cluster_sm_notes", {})["scheduler"] = (
                "MLA FA3 launch uses multiprocessor_count in prefill_sm90.cuh; observe launched grid"
            )
    elif op == "cutlass_fused_moe":
        case["num_tokens"] = 1
        knobs = ["hidden", "intermediate"]
        top_k = int(src.get("top_k") or 1)
        n_exp = int(src.get("num_experts") or 1)
        case.setdefault("cluster_sm_notes", {})["dynamic_cluster"] = (
            "SM90 tactic chooses ClusterShape 1x1x1/2x1x1/1x2x1/2x2x1 at runtime "
            "(cutlass_heuristic.cpp get_candidate_configs_sm90)"
        )
        case.setdefault("cluster_sm_notes", {})["moe_routing"] = {
            "original_top_k": top_k,
            "original_num_experts": n_exp,
            "scaled_num_tokens": 1,
            "reason": (
                "grouped TMA MoE GEMM launches at least one tile per routed expert; "
                "keeping top_k/num_experts forbids 1 CTA and a single 2-CTA cluster"
            ),
        }
        if top_k >= 2 or n_exp > 1:
            predicted_skip = (
                f"grouped TMA MoE GEMM launches >=1 tile per routed expert "
                f"(moe_gemm_template_dispatch_tma_ws.h) with runtime ClusterShape "
                f"1x1x1/2x1x1/1x2x1/2x2x1 (cutlass_heuristic.cpp get_candidate_configs_sm90). "
                f"Source top_k={top_k} num_experts={n_exp}; 1 CTA or one 2-CTA cluster would "
                f"drop top_k/num_experts and change MoE routing."
            )
    elif op == "fp8_blockscale_gemm":
        if source_id.endswith("_04"):
            case["m"] = 16
        else:
            case["m"] = 16
        case["n"] = 128
        case["k"] = FP8_K
        knobs = ["k"]
        case.setdefault("cluster_sm_notes", {})["dynamic_cluster"] = (
            "DeepGEMM clusterDim={num_tma_multicast,1,1} and gridDim=num_sms "
            "(fp8_gemm.cuh; deep_gemm.py multi_processor_count)"
        )
    elif op == "mamba_state_update":
        case["batch"] = 1
        case["heads"] = 1
        case["dim"] = max(int(src.get("dim") or 64), 256)
        knobs = ["dim", "dstate"]
    elif op in {"gdn_prefill", "gdn_decode"}:
        case["batch"] = 1
        case["heads_q"] = 1
        orig_q, orig_v = src.get("heads_q") or 1, src.get("heads_v") or 1
        ratio = max(1, orig_v // orig_q) if orig_q else 1
        case["heads_v"] = ratio
        if op == "gdn_prefill":
            case["seqlen"] = 2048
            knobs = ["seqlen"]
        else:
            case["seqlen"] = 1
            knobs = ["head_dim"]
    elif op in {"rmsnorm", "fused_add_rmsnorm"}:
        case["rows"] = 1
        case["hidden"] = HIDDEN
        knobs = ["hidden"]
    elif op == "rope":
        case["batch"] = 1
        orig_q, orig_kv = int(src.get("heads_q") or 1), int(src.get("heads_kv") or 1)
        if orig_q == orig_kv:
            case["heads_q"] = 1
            case["heads_kv"] = 1
        else:
            predicted_skip = (
                f"Original GQA heads_q={orig_q} heads_kv={orig_kv}. Keeping GQA ratio; "
                "pos_enc.cuh BatchQKApplyRotaryKernel "
                "nblks=batch_size*(num_qo_heads+num_kv_heads). "
                "1 CTA would require dropping GQA."
            )
        case["seqlen"] = 4096
        knobs = ["seqlen"]
    elif op == "append_paged_kv":
        case["batch"] = 1
        case["heads_kv"] = 1
        case["append_len"] = 1024
        knobs = ["append_len"]
    elif op in {"sample", "top_k_sample", "top_p_sample"}:
        case["batch"] = 1
        case["vocab"] = max(int(src.get("vocab") or 32000), 128256)
        knobs = ["vocab"]
    elif op == "packbits":
        case["elements"] = 8 * 1024 * 1024
        knobs = ["elements"]
    elif op == "segment_packbits":
        case["elements"] = 8 * 1024 * 1024
        case["segments"] = 1
        knobs = ["elements"]
    elif op in {"merge_state", "merge_states"}:
        case["heads"] = 1
        case["seqlen"] = 16384
        knobs = ["seqlen"]
    else:
        knobs = []

    case["cluster_sm"] = {
        "source_case_id": source_id,
        "target_ctas": target_ctas,
        "uses_cluster": uses_cluster,
        "predicted_skip": predicted_skip,
        "scale_knobs": knobs,
        "original_parameters": {
            key: src[key]
            for key in src
            if key
            not in {
                "case_id",
                "source",
                "category",
                "qualification_status",
                "mechanisms",
                "delivery_group",
                "delivery_index",
                "operation",
            }
        },
        "dynamic_cluster": source_id in DYNAMIC,
    }
    case["qualification_status"] = "cluster_sm_derived_unmeasured"
    return case


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    derived = deepcopy(data)
    derived["name"] = "FlashInfer Hopper 1CTA/2CTA cluster-SM derived set"
    derived["selection"] = {
        "mode": "cluster_sm_scale_down",
        "source_manifest": "manifests/flashinfer_100.json",
        "sm_limit": "1 CTA, or one 2-CTA cluster when the algorithm uses CGA",
        "cycle_filter": "target_gpu_cycles_100000_500000",
    }
    derived["delivery"] = {
        "groups": 8,
        "group_assignment": "delivery_index modulo 8",
        "runner": "scripts/run_cluster_sm.sh",
    }
    derived["cases"] = [scale_case(case) for case in data["cases"]]
    for case in derived["cases"]:
        case["delivery_group"] = int(case["delivery_index"]) % 8
    OUTPUT.write_text(json.dumps(derived, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} cases={len(derived['cases'])}")


if __name__ == "__main__":
    main()
