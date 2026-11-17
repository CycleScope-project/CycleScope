"""FA3 1CTA/2CTA cluster-SM derivation: keep mechanism, scale work, inject sm_margin."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

from full_gpu_cases import (
    EVIDENCE,
    PRIMARY_OK,
    ceil_div,
    combine_block_m,
    derive_traits,
    tile_size_fwd_sm90,
)

SUITE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = SUITE_ROOT / "manifests" / "fa3_100.json"
DERIVED_MANIFEST = SUITE_ROOT / "manifests" / "fa3_cluster_sm100.json"
H100_SMS = 132
CYCLES_MIN = 100_000
CYCLES_MAX = 500_000

# Observed ClusterShape=2x1x1 on any timed kernel of the full-GPU 100 (includes bwd→fwd).
OBSERVED_CLUSTER_SOURCE_IDS = frozenset({
    "fa3_bwd_dqacc_atomic_d256_02",
    "fa3_bwd_dqacc_bulk_04",
    "fa3_cluster_even_02",
    "fa3_cluster_even_03",
    "fa3_cluster_even_04",
    "fa3_fp8_hd192_cluster_02",
    "fa3_fp8_hd192_cluster_03",
    "fa3_fp8_hd192_cluster_04",
    "fa3_hd192_dv128_01",
    "fa3_hd192_dv128_02",
    "fa3_pack_gqa_04",
})

STRIP_KEYS = (
    "kernel_activities",
    "h100_timing",
    "gpu",
    "cycles_hint_band",
    "measured_source_revision",
    "build_root",
    "qualification_status",
    "tuning",
    "delivery_group",
    "delivery_index",
)

FA3_KERNEL_RE = re.compile(
    r"FlashAttn(?:FwdSm90|BwdSm90|BwdPreprocess|BwdPostprocessConvertdQ|FwdCombine)|"
    r"prepare_varlen_num_blocks"
)
CLUSTER_TUPLE_RE = re.compile(
    r"CollectiveMainloop(?:Fwd|Bwd)Sm90<.*?cute::tuple<cute::C<(\d+)>, cute::C<(\d+)>, cute::C<(\d+)>"
)


def sm_margin_for(target_ctas: int, sm_count: int = H100_SMS) -> int:
    if target_ctas not in (1, 2):
        raise ValueError(f"target_ctas must be 1 or 2, got {target_ctas}")
    if sm_count < target_ctas:
        raise ValueError(f"sm_count={sm_count} < target_ctas={target_ctas}")
    return int(sm_count) - int(target_ctas)


def _seqlen_for_nm(kbm: int, qh_per: int, n_m: int) -> int:
    qh_per = max(int(qh_per), 1)
    return max(1, int(n_m) * int(kbm) // qh_per)


def _align_page(seqlen_k: int, page_size: int | None) -> int:
    if not page_size:
        return max(1, int(seqlen_k))
    page = int(page_size)
    return max(page, (int(seqlen_k) // page) * page)


def _gqa_heads(src: dict) -> tuple[int, int]:
    hq = int(src["heads_q"])
    hkv = int(src.get("heads_kv") or hq)
    if hq != hkv:
        return hq // hkv, 1
    return 1, 1


def _fwd_kbm(case: dict) -> tuple[int, int]:
    traits = derive_traits(case)
    return int(traits["kBlockM"]), int(traits["kBlockN"])


def derive_one(src: dict) -> dict[str, Any]:
    # c450 rows set exact_lengths: their attended length is the work
    # dimension and is used as given (bounded by the reference S matrix);
    # every other row keeps the cluster_sm clamps.
    exact = bool(src.get("exact_lengths"))

    def cap(value: int, limit: int) -> int:
        return value if exact else min(value, limit)

    primary = str(src["primary_mechanism"])
    source_id = str(src["case_id"])
    uses_cluster = source_id in OBSERVED_CLUSTER_SOURCE_IDS
    # Non-cluster may use 2 ordinary CTAs (prefer 1). split_kv_fixed
    # SingleTileScheduler grid Y includes num_splits, so preserving split
    # is a 2-CTA non-cluster launch (cluster_dim stays 1x1x1).
    target_ctas = 2 if uses_cluster or primary == "split_kv_fixed" else 1

    case = {k: copy.deepcopy(v) for k, v in src.items() if k not in STRIP_KEYS}
    case["source_case_id"] = source_id
    case["case_id"] = f"{source_id}_cluster_sm"
    case["suite"] = "flashattention3"
    case["cluster_sm"] = True
    case["target_ctas"] = target_ctas
    case["uses_cluster"] = uses_cluster

    hq, hkv = _gqa_heads(src)
    if primary == "pack_gqa":
        hq, hkv = max(2, int(src["heads_q"]) // int(src["heads_kv"])), 1
        case["pack_gqa"] = True
    case["heads_q"] = hq
    case["heads_kv"] = hkv
    case["batch"] = 1

    page_size = src.get("page_size")
    kbm_probe = dict(case)
    kbm_probe["seqlen_q"] = int(src.get("seqlen_q") or 128)
    kbm, kbn = _fwd_kbm(kbm_probe)
    pack = bool(case.get("pack_gqa"))
    qh_per = (hq // hkv) if pack else 1

    seqlen_q = int(src.get("seqlen_q") or 1)
    seqlen_k = int(src.get("seqlen_k") or seqlen_q)
    chunk = int(src.get("attention_chunk") or 0)

    if uses_cluster:
        seqlen_q = _seqlen_for_nm(kbm, qh_per, 2)
        if primary == "pack_gqa":
            seqlen_q = max(seqlen_q, 32)
        seqlen_k = max(seqlen_k, 2048)
    elif primary == "cluster_odd":
        seqlen_q = _seqlen_for_nm(kbm, qh_per, 1)
        seqlen_k = max(1024, min(seqlen_k, 4096))
    elif primary == "paged_tma":
        seqlen_q = max(kbm + 1, kbm * 2)
        seqlen_k = _align_page(cap(max(seqlen_k, 2048), 8192), page_size)
    elif primary == "kvcache_paged_decode":
        seqlen_q = min(int(src["seqlen_q"]), 8)
        seqlen_k = _align_page(cap(max(int(src["seqlen_k"]), 2048), 8192), page_size)
    elif primary == "paged_nontma":
        seqlen_q = min(max(int(src["seqlen_q"]), 64), kbm)
        seqlen_k = _align_page(cap(max(seqlen_k, 2048), 8192), page_size)
    elif primary == "attention_chunk":
        seqlen_q = max(chunk, kbm)
        seqlen_k = max(seqlen_q, 1024)
    elif primary == "combine_standalone":
        cbm = combine_block_m(int(case.get("head_dim_v") or case["head_dim"]))
        seqlen_q = cbm
        seqlen_k = seqlen_q
        case["num_splits"] = max(8, int(src.get("num_splits") or 8))
        case["batch"] = 1
        hq, hkv = 1, 1
        case["heads_q"] = hq
        case["heads_kv"] = hkv
    elif primary == "split_kv_fixed":
        # Combine helper grid X = ceil(seqlen_q*h/combine_block_m). Keep that
        # launch at 1 CTA (prefer); fwd is 2 ordinary CTAs via num_splits=2.
        cbm = combine_block_m(int(case.get("head_dim_v") or case["head_dim"]))
        seqlen_q = cbm
        seqlen_k = max(2048, cap(int(src["seqlen_k"]), 8192))
        case["num_splits"] = 2
        case["heads_q"] = 1
        case["heads_kv"] = 1
        case["target_ctas"] = 2
        case["uses_cluster"] = False
        target_ctas = 2
        uses_cluster = False
    elif primary.startswith("split_varlen_sparse"):
        case["batch"] = 17
        long_q = max(kbm, 128)
        case["seqlens_q"] = [16] * 16 + [long_q]
        case["seqlens_k"] = [64] * 16 + [max(512, cap(int(src["seqlen_k"]), 2048))]
        case["seqlen_q"] = long_q
        case["seqlen_k"] = max(case["seqlens_k"])
        case["num_splits"] = max(2, int(src.get("num_splits") or 4))
        seqlen_q = long_q
        seqlen_k = case["seqlen_k"]
    elif primary.startswith("split_varlen_dense"):
        q = max(kbm, 128)
        k = max(256, cap(int(src["seqlen_k"]), 1024))
        case["batch"] = 2
        case["seqlens_q"] = [q, q]
        case["seqlens_k"] = [k, k]
        case["seqlen_q"] = q
        case["seqlen_k"] = k
        case["num_splits"] = max(2, int(src.get("num_splits") or 4))
        seqlen_q, seqlen_k = q, k
    elif src["operation"] in ("varlen_fwd", "varlen_bwd"):
        q = max(kbm, min(int(src["seqlen_q"]), 256))
        k = max(q, cap(int(src["seqlen_k"]), 1024))
        case["batch"] = 2
        case["seqlens_q"] = [q, max(kbm // 2, q // 2)]
        case["seqlens_k"] = [k, max(kbm, k // 2)]
        case["seqlen_q"] = max(case["seqlens_q"])
        case["seqlen_k"] = max(case["seqlens_k"])
        seqlen_q, seqlen_k = case["seqlen_q"], case["seqlen_k"]
    elif src["operation"] == "kvcache":
        seqlen_q = min(max(int(src["seqlen_q"]), 1), 32)
        if src.get("seqlen_new"):
            case["seqlen_new"] = min(int(src["seqlen_new"]), seqlen_q)
        seqlen_k = _align_page(cap(max(int(src["seqlen_k"]), 1024), 8192), page_size)
        if src.get("seqlens_k"):
            used = min(int(src["seqlens_k"][0]), seqlen_k)
            case["seqlens_k"] = [max(seqlen_q, used)]
    else:
        seqlen_q = min(max(seqlen_q, kbm), kbm * (2 if uses_cluster else 1))
        seqlen_k = max(seqlen_k, seqlen_q, 1024)
        seqlen_k = cap(seqlen_k, 8192)

    if chunk:
        seqlen_q = max(seqlen_q, chunk)
    case["seqlen_q"] = int(seqlen_q)
    case["seqlen_k"] = int(_align_page(seqlen_k, page_size))
    if exact:
        # reference attention materializes S = seqlen_q x attended length
        limit = int(8 * 2**30 // (4 * max(1, int(case.get("heads_q") or 1)) * case["seqlen_q"]))
        if case["seqlen_k"] > limit or any(int(x) > limit for x in (case.get("seqlens_k") or [])):
            raise ValueError(f"attended length exceeds the reference S-matrix limit {limit}: {case.get('case_id')}")

    if case["operation"] in ("varlen_fwd", "varlen_bwd") and not case.get("seqlens_q"):
        case["seqlens_q"] = [case["seqlen_q"]]
        case["seqlens_k"] = [case["seqlen_k"]]
        case["batch"] = 1

    if case.get("cache_leftpad") and isinstance(case.get("seqlens_k"), list) and case["seqlens_k"]:
        case["seqlens_k"] = [max(int(x), int(case["seqlen_q"]) + 8) for x in case["seqlens_k"]]
        case["seqlen_k"] = max(int(x) for x in case["seqlens_k"])

    case = clip_seq_lists(case)

    traits = derive_traits(case)
    checker = PRIMARY_OK[primary]
    if not checker(case, traits):
        case = _repair_primary(src, case, primary)
        traits = derive_traits(case)
        if not checker(case, traits):
            case["derive_warning"] = (
                f"PRIMARY_OK failed after repair for {primary}: traits={ {k: traits[k] for k in ('cluster_m','enable_cluster','n_m_tiles','pagedkv_tma','pagedkv_nontma','pack_gqa','combine_sparse','combine_dense','is_split','is_local')} }"
            )

    case["trigger"] = traits
    case["predicted_cluster_m"] = int(traits["cluster_m"])
    # CTA count = SM count. 2 ordinary CTAs still need 2 distinct SMs.
    case["sm_margin"] = sm_margin_for(target_ctas)
    evidence = EVIDENCE[primary]
    case["mechanism_evidence"] = {
        "primary": primary,
        "files": list(evidence.get("files") or []),
        "symbols": list(evidence.get("symbols") or []),
        "condition": evidence.get("condition"),
        "source_launch": "upstream/hopper/flash_fwd_launch_template.h Enable_cluster/CLUSTER_SWITCH/UsePersistentScheduler",
        "sm_margin": "official flash_attn_* sm_margin → params.num_sm = multiProcessorCount - sm_margin",
        "observed_cluster_source": uses_cluster,
    }
    case["source_revision"] = src.get("source_revision")
    case["implementation"] = "FA3"
    return case


def _repair_primary(src: dict, case: dict, primary: str) -> dict:
    """One-shot shape nudges that keep dtype/layout/backend/mask/quant."""
    kbm, _kbn = _fwd_kbm(case)
    if primary == "cluster_even" or primary == "fp8_hd192_cluster":
        pack = bool(case.get("pack_gqa"))
        qh_per = (int(case["heads_q"]) // int(case["heads_kv"])) if pack else 1
        case["seqlen_q"] = _seqlen_for_nm(kbm, qh_per, 2)
        case["causal"] = False
        case["window_size"] = [-1, -1]
        case["attention_chunk"] = 0
        case["num_splits"] = 1
    elif primary == "cluster_odd":
        case["seqlen_q"] = _seqlen_for_nm(kbm, 1, 1)
        case["heads_q"] = case["heads_kv"] = 1
        case["causal"] = False
        case["num_splits"] = 1
        case["window_size"] = [-1, -1]
        case["attention_chunk"] = 0
    elif primary == "paged_tma":
        case["seqlen_q"] = kbm * 2
        case["append_kv"] = False
        case["cache_leftpad"] = False
    elif primary == "paged_nontma":
        case["page_size"] = int(src["page_size"])
        case["cache_leftpad"] = False
        case["append_kv"] = False
        if int(case["page_size"]) % _fwd_kbm(case)[1] == 0:
            case["seqlen_q"] = min(case["seqlen_q"], kbm)
    elif primary == "pack_gqa":
        case["pack_gqa"] = True
        case["heads_q"] = max(2, int(src["heads_q"]) // int(src["heads_kv"]))
        case["heads_kv"] = 1
        if case.get("uses_cluster"):
            case["causal"] = False
            case["seqlen_q"] = _seqlen_for_nm(kbm, case["heads_q"], 2)
    elif primary == "split_varlen_sparse_combine":
        long_q = max(256, kbm)
        case["batch"] = 33
        case["seqlens_q"] = [16] * 32 + [long_q]
        case["seqlens_k"] = [64] * 32 + [1024]
        case["seqlen_q"] = long_q
        case["seqlen_k"] = 1024
        case["num_splits"] = max(2, int(case.get("num_splits") or 4))
    elif primary == "split_varlen_dense_combine":
        q = max(kbm, 128)
        case["batch"] = 2
        case["seqlens_q"] = [q, q]
        case["seqlens_k"] = [q, q]
        case["seqlen_q"] = q
        case["seqlen_k"] = q
        case["num_splits"] = max(2, int(case.get("num_splits") or 4))
    elif primary == "local_window":
        win = src.get("window_size") or [256, 256]
        case["window_size"] = list(win)
        case["causal"] = False
        case["attention_chunk"] = 0
        case["seqlen_q"] = max(int(case["seqlen_q"]), max(int(win[0]), int(win[1]), kbm))
        case["seqlen_k"] = max(int(case["seqlen_k"]), case["seqlen_q"])
    return case


def derive_all(source_cases: list[dict]) -> list[dict]:
    if len(source_cases) != 100:
        raise ValueError(f"expected 100 source cases, got {len(source_cases)}")
    derived = [derive_one(src) for src in source_cases]
    ids = [c["source_case_id"] for c in derived]
    if len(set(ids)) != 100:
        raise ValueError("source_case_id not unique after derive")
    return derived


def parse_cluster_shape(kernel_name: str) -> list[int] | None:
    """Compile-time ClusterShape echo in the kernel type. Not launch clusterDim."""
    match = CLUSTER_TUPLE_RE.search(kernel_name)
    if not match:
        return None
    return [int(match.group(1)), int(match.group(2)), int(match.group(3))]


def launch_grid(grids) -> list[int] | None:
    if not grids:
        return None
    grid = grids[0] if isinstance(grids[0], (list, tuple)) else grids
    try:
        return [int(x) for x in grid]
    except (TypeError, ValueError):
        return None


def launch_ctas(grids) -> int | None:
    grid = launch_grid(grids)
    if not grid:
        return None
    return int(math.prod(grid))


def grid_ctas(grid) -> int:
    value = launch_ctas(grid)
    return int(value or 0)


def classify_kernel(name: str) -> str:
    if "FlashAttnFwdSm90" in name:
        return "fa3_fwd"
    if "FlashAttnBwdSm90" in name:
        return "fa3_bwd"
    if "FlashAttnBwdPreprocess" in name:
        return "fa3_bwd_preprocess"
    if "FlashAttnBwdPostprocess" in name:
        return "fa3_bwd_postprocess"
    if "FlashAttnFwdCombine" in name:
        return "fa3_combine"
    if "prepare_varlen" in name:
        return "fa3_prepare_varlen"
    if name.startswith("at::native") or name.startswith("void at::") or "FillFunctor" in name:
        return "pytorch_helper"
    if "globaltimer" in name.lower() or "delivery_clock" in name.lower():
        return "pytorch_helper"
    return "other"


def is_op_kernel(kind: str) -> bool:
    return kind not in ("pytorch_helper",)


def extra_cta_op_kernels(kernels: list[dict]) -> list[dict]:
    extra = []
    for kernel in kernels:
        if not is_op_kernel(kernel.get("kind") or ""):
            continue
        ctas = kernel.get("cta_count")
        if ctas not in (1, 2):
            extra.append({
                "kind": kernel.get("kind"),
                "cta_count": ctas,
                "grid": launch_grid(kernel.get("grids") or []),
                "name": (kernel.get("name") or "")[:160],
            })
    return extra


def topology_from_activities(
    activities: list[dict],
    *,
    target_ctas: int,
    uses_cluster: bool,
    evidence: str | None = None,
    ptx: dict | None = None,
) -> dict:
    kernels = []
    for row in activities or []:
        name = str(row.get("name") or "")
        grids = list(row.get("grids") or [])
        blocks = list(row.get("blocks") or [])
        kind = classify_kernel(name)
        ctas = launch_ctas(grids)
        kernels.append({
            "name": name,
            "kind": kind,
            "count": int(row.get("count") or 0),
            "grids": grids,
            "grid": launch_grid(grids),
            "blocks": blocks,
            "cta_count": ctas,
            "compile_time_cluster_shape_echo": parse_cluster_shape(name),
            "device_time_us": row.get("device_time_us"),
        })
    op_kernels = [k for k in kernels if is_op_kernel(k["kind"])]
    fa3 = [k for k in kernels if str(k["kind"]).startswith("fa3_")]
    fwd = [k for k in fa3 if k["kind"] == "fa3_fwd"]
    primary = fwd[0] if fwd else (fa3[0] if fa3 else (op_kernels[0] if op_kernels else None))
    grid = primary.get("grid") if primary else None
    cta = primary.get("cta_count") if primary else None
    extra = extra_cta_op_kernels(kernels)

    cluster_dim = None
    observed_sms = None
    sm_method = "none"
    ptx = ptx or {}
    sm_ids = [int(x) for x in (ptx.get("sm_ids") or ptx.get("observed_sms") or []) if x is not None]
    ncta = ptx.get("cluster_nctaid") or ptx.get("cluster_dim")
    ncta_prod = None
    if isinstance(ncta, (list, tuple)) and ncta:
        ncta_prod = int(math.prod(int(x) for x in ncta))
        cluster_from_ptx = [int(x) for x in ncta]
    elif ncta is not None:
        ncta_prod = int(ncta)
        cluster_from_ptx = [ncta_prod, 1, 1]
    else:
        cluster_from_ptx = None
    if uses_cluster and sm_ids and ncta_prod == 2 and len(set(sm_ids)) == 2:
        cluster_dim = cluster_from_ptx
        observed_sms = sm_ids[:2]
        sm_method = "device %%smid and %%cluster_nctaid stores in FA3 kernel bodies"
    elif not uses_cluster:
        cluster_dim = [1, 1, 1]
        if cta == 1:
            observed_sms = 1
            sm_method = "1CTA launched grid product 1 infers one SM"
        elif cta == 2:
            uniq = []
            for sid in sm_ids:
                if sid not in uniq:
                    uniq.append(sid)
            rows = ptx.get("rows") if isinstance(ptx, dict) else None
            occupied = len(rows) if isinstance(rows, list) else None
            # Two smids must come from the same target-kernel launch (exactly
            # two occupied device slots). Do not take uniq[:2] from a larger
            # bag that mixed helper 1CTA stores.
            same_launch = occupied in (None, 2)
            if len(uniq) == 2 and same_launch:
                observed_sms = uniq
                sm_method = (
                    "non-cluster 2 ordinary CTAs; device %%smid stores from the "
                    "same target-kernel launch; two distinct SMs; "
                    "%%cluster_nctaid product 1 (not ClusterShape)"
                )
            else:
                observed_sms = uniq if uniq else None
                sm_method = (
                    "non-cluster 2 ordinary CTAs require two distinct measured smids "
                    "from the same target-kernel launch (CTA count=SM count); "
                    f"got sm_ids={observed_sms} occupied_cta_rows={occupied}. "
                    "two CTAs sharing one SM cannot qualify; helper smids are not unioned"
                )
        else:
            sm_method = "non-cluster launch cluster_dim 1x1x1; grid product is not 1"

    return {
        "method": (
            "Kineto chrome-trace launched grid (gridX*gridY*gridZ of one launch). "
            "Non-cluster cluster_dim is 1x1x1. "
            + sm_method
        ),
        "grid": grid,
        "cluster_dim": cluster_dim,
        "cta_count": cta,
        "observed_sms": observed_sms,
        "evidence": evidence,
        "target_ctas": target_ctas,
        "uses_cluster_requested": uses_cluster,
        "observed_uses_cluster": bool(uses_cluster and cluster_dim is not None and int(math.prod(cluster_dim)) == 2),
        "observed_fwd_cta_count": (fwd[0]["cta_count"] if fwd else cta),
        "kernels": kernels,
        "fa3_kernel_count": len(fa3),
        "extra_cta_op_kernels": extra,
        "ptx": ptx or None,
        "note": (
            "Every FA3 fwd/bwd/preprocess/postprocess/combine/varlen kernel is listed. "
            "PyTorch fill/elementwise helpers are listed but not topology. "
            "compile_time_cluster_shape_echo is the kernel type, not launch clusterDim. "
            "sm_margin is a requested persistent occupancy cap, not observed_sms."
        ),
    }


def mechanism_still_ok(case: dict) -> tuple[bool, dict]:
    traits = derive_traits(case)
    primary = case["primary_mechanism"]
    ok = bool(PRIMARY_OK[primary](case, traits))
    return ok, traits


def clip_seq_lists(case: dict) -> dict:
    """Keep seqlens_* length equal to batch after 1CTA/2CTA batch collapse."""
    out = copy.deepcopy(case)
    batch = max(1, int(out.get("batch") or 1))
    for key in ("seqlens_k", "seqlens_q"):
        vals = out.get(key)
        if not isinstance(vals, list) or not vals:
            continue
        if len(vals) > batch:
            out[key] = [int(x) for x in vals[:batch]]
        elif len(vals) < batch:
            out[key] = [int(x) for x in vals] + [int(vals[-1])] * (batch - len(vals))
        else:
            out[key] = [int(x) for x in vals]
    if isinstance(out.get("seqlens_q"), list) and out["seqlens_q"]:
        out["seqlen_q"] = max(int(x) for x in out["seqlens_q"])
        out["batch"] = len(out["seqlens_q"])
    if isinstance(out.get("seqlens_k"), list) and out["seqlens_k"]:
        used = max(int(x) for x in out["seqlens_k"])
        extra = int(out.get("seqlen_new") or 0) if out.get("append_kv") else 0
        page = out.get("page_size")
        cap = used + extra + (8 if out.get("append_kv") else 0)
        out["seqlen_k"] = _align_page(max(int(out.get("seqlen_k") or 0), cap), page)
    return out


def _ensure_append_capacity(out: dict) -> dict:
    if not out.get("append_kv"):
        return out
    used = max(out.get("seqlens_k") or [0]) if out.get("seqlens_k") else 0
    extra = int(out.get("seqlen_new") or out.get("seqlen_q") or 0)
    page = out.get("page_size")
    out["seqlen_k"] = _align_page(max(int(out.get("seqlen_k") or 0), used + extra + 8), page)
    return out


def scale_seqlen_k(case: dict, seqlen_k: int) -> dict:
    out = clip_seq_lists(case)
    page = out.get("page_size")
    seqlen_k = _align_page(max(1, int(seqlen_k)), page)
    out["seqlen_k"] = seqlen_k
    if out.get("seqlens_k"):
        src = list(out["seqlens_k"])
        max_old = max(src) or 1
        target_used = seqlen_k
        if out.get("append_kv"):
            extra = int(out.get("seqlen_new") or out.get("seqlen_q") or 0)
            target_used = max(1, seqlen_k - extra - 8)
        scaled = [max(1, int(round(v * target_used / max_old))) for v in src]
        if page:
            scaled = [_align_page(v, page) for v in scaled]
        out["seqlens_k"] = scaled
        out["seqlen_k"] = max(scaled)
    if out.get("cache_leftpad") and out.get("seqlens_k"):
        out["seqlens_k"] = [max(int(v), int(out["seqlen_q"]) + 8) for v in out["seqlens_k"]]
        out["seqlen_k"] = max(out["seqlens_k"])
    return _ensure_append_capacity(out)


def scale_seqlen_q(case: dict, seqlen_q: int) -> dict:
    out = clip_seq_lists(case)
    seqlen_q = max(1, int(seqlen_q))
    chunk = int(out.get("attention_chunk") or 0)
    if chunk:
        seqlen_q = max(seqlen_q, chunk)
    out["seqlen_q"] = seqlen_q
    if out.get("seqlens_q"):
        src = list(out["seqlens_q"])
        old = max(src) or 1
        out["seqlens_q"] = [max(1, int(round(v * seqlen_q / old))) for v in src]
        out["seqlen_q"] = max(out["seqlens_q"])
        out["batch"] = len(out["seqlens_q"])
    if out.get("append_kv"):
        out["seqlen_new"] = int(out["seqlen_q"])
    elif out.get("seqlen_new"):
        out["seqlen_new"] = min(int(out["seqlen_new"]), out["seqlen_q"])
    return _ensure_append_capacity(out)


def scale_num_splits(case: dict, num_splits: int) -> dict:
    out = clip_seq_lists(case)
    out["num_splits"] = max(2, int(num_splits))
    return out
