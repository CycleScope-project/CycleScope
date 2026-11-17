#!/usr/bin/env python3
"""FA3 full-GPU candidate inventory: mechanism triggers, seeds, and 100-case selector.

Shapes are real-token workloads. Tuning changes seq/batch/heads, never repeats the
kernel to inflate cycles. Primary names are dispatch paths, not dtype×shape grids.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

SOURCE_REVISION = "98eb7998a0eba4047c7a30375522569d4b8efb20"
SOURCE_URL = "https://github.com/Dao-AILab/flash-attention/tree/main/hopper"
INTERFACE = "upstream/hopper/flash_attn_interface.py"
CYCLES_MIN = 100_000
CYCLES_MAX = 1_500_000
CV_LIMIT = 2.5
DRIFT_LIMIT = 2.5
TARGET_SELECTED = 100
RECORDED_SAMPLES = 9
RAW_SAMPLES = 36
GROUP_FLOOR = 3
GROUP_CEILING = 4
FORMAL_INPUT_SEED = 0
OPERATIONS = ("fixed_fwd", "varlen_fwd", "fixed_bwd", "varlen_bwd", "kvcache", "combine")
DTYPES = ("bf16", "fp16", "fp8_e4m3")
OUTER_SHARDS = 4
_COMMON = Path(__file__).resolve().parents[2] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
from delivery_schema import cases_header  # noqa: E402
TIMING_AGGREGATION = {
    "median_cycles": "median_of_round_medians",
    "cv_percent": "max_of_round_cv_percent",
    "drift_percent": "relative_drift_of_round_medians",
}
SKIP_ROUND_FILES = frozenset({
    "contract.json",
    "summary.json",
    "probe.json",
    "manifest.json",
    "tuning_report.json",
    "inventory_report.json",
})
SHAPE_KEYS = (
    "operation",
    "dtype",
    "batch",
    "seqlen_q",
    "seqlen_k",
    "heads_q",
    "heads_kv",
    "head_dim",
    "head_dim_v",
    "num_splits",
    "causal",
    "window_size",
    "attention_chunk",
    "softcap",
    "pack_gqa",
    "page_size",
    "append_kv",
    "seqlen_new",
    "seqlens_q",
    "seqlens_k",
    "rotary_dim",
    "qv",
    "deterministic",
    "rotary_interleaved",
    "cache_leftpad",
    "cache_batch_idx",
    "reuse_scheduler_metadata",
)
TUNING_AXES = ("batch", "seqlen_k", "num_splits", "repeat_seqlens_batch")
TUNING_RULE = (
    "Pick one value from tuning.values and apply via apply_tuning(). "
    "That changes real tokens/batch/splits. Never loop the same launch to inflate cycles. "
    "Every listed value must keep primary_mechanism triggers."
)

# Copied from upstream/hopper/tile_size.h tile_size_fwd_sm90.
def tile_size_fwd_sm90(
    headdim: int,
    headdim_v: int,
    is_causal: bool,
    is_local: bool,
    element_size: int = 2,
    v_colmajor: bool = False,
    paged_kv_non_tma: bool = False,
    softcap: bool = False,
) -> tuple[int, int, bool, bool]:
    if element_size == 2:
        if headdim <= 64:
            if headdim_v == 512:
                return 64, 64, False, False
            if headdim_v == 256:
                return 128, 96, True, False
            use_blockn_128 = is_causal or is_local or paged_kv_non_tma
            return 192, 128 if use_blockn_128 else 192, use_blockn_128, True
        if headdim <= 96:
            return 192, 128 if (is_local or paged_kv_non_tma) else 144, False, True
        if headdim <= 128:
            use_blockn_128 = is_causal or is_local or paged_kv_non_tma
            return 128, 128 if use_blockn_128 else 176, True, True
        if headdim <= 192:
            return 128, 96 if (paged_kv_non_tma or is_local) else (128 if headdim_v <= 128 else 112), True, True
        return 128, 64 if is_local else 80, True, True
    if headdim <= 64:
        return 192, 160, True, True
    if headdim <= 96:
        return 192, 128, True, True
    if headdim <= 128:
        return 128, 160 if paged_kv_non_tma else (192 if (v_colmajor or (softcap and is_local)) else 224), True, True
    if headdim <= 192:
        return 128, 128 if ((paged_kv_non_tma or softcap) and is_local) else 160, True, True
    return 128, 64 if is_local else 128, True, not paged_kv_non_tma


def ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def combine_block_m(head_dim_v: int) -> int:
    k_block_k = 64 if int(head_dim_v) <= 64 else 128
    if k_block_k % 128 == 0:
        return 8
    if k_block_k % 64 == 0:
        return 16
    return 32


def is_local_mask(case: dict) -> bool:
    window = case.get("window_size") or [-1, -1]
    chunk = int(case.get("attention_chunk") or 0)
    causal = bool(case.get("causal"))
    return (int(window[0]) >= 0 or int(window[1]) >= 0 or chunk >= 1) and not causal


def derive_traits(case: dict) -> dict[str, Any]:
    dtype = case["dtype"]
    elem = 1 if dtype == "fp8_e4m3" else 2
    hd = int(case["head_dim"])
    hdv = int(case.get("head_dim_v") or hd)
    causal = bool(case.get("causal"))
    local = is_local_mask(case)
    split = int(case.get("num_splits") or 1) > 1
    operation = case["operation"]
    varlen = operation in ("varlen_fwd", "varlen_bwd") or (
        bool(case.get("seqlens_q") or case.get("seqlens_k")) and operation != "kvcache"
    )
    page_size = case.get("page_size")
    paged = page_size is not None and int(page_size) > 0
    append = bool(case.get("append_kv"))
    leftpad = bool(case.get("cache_leftpad"))
    softcap = float(case.get("softcap") or 0.0) > 0.0
    v_colmajor = bool(case.get("v_colmajor"))
    hq = int(case["heads_q"])
    hkv = int(case.get("heads_kv") or hq)
    seqlen_q = int(case.get("seqlen_q") or max(case.get("seqlens_q") or [0]))
    pack = case.get("pack_gqa")
    paged_tma = False
    paged_nontma = False
    kbm, kbn, rs, overlap = tile_size_fwd_sm90(hd, hdv, causal, local, elem, v_colmajor, False, softcap)
    if paged:
        if leftpad or append:
            paged_nontma = True
            paged_tma = False
        else:
            paged_tma = (int(page_size) % kbn == 0) and (seqlen_q * (hq // hkv) > kbm)
            paged_nontma = not paged_tma
        if paged_nontma:
            kbm, kbn, rs, overlap = tile_size_fwd_sm90(hd, hdv, causal, local, elem, v_colmajor, True, softcap)
    enable_cluster = ((elem == 2 and hd >= 128) or (elem == 1 and hd == 192))
    enable_cluster = enable_cluster and (not causal) and (not local) and (not split) and (not paged_nontma) and (not varlen) and operation in ("fixed_fwd",)
    if pack is None:
        pack = bool(varlen and hq != hkv) or paged_nontma or split
        if not pack and hq != hkv and not paged_tma:
            # heuristics.h should_pack_gqa for non-varlen
            round_up = lambda a, b: (a + b - 1) // b * b
            nopack = seqlen_q / float(round_up(seqlen_q, kbm))
            packed = (seqlen_q * (hq // hkv)) / float(round_up(seqlen_q * (hq // hkv), kbm))
            pack = nopack < 0.9 * packed
        if hq == hkv:
            pack = False
        if paged_nontma or split:
            pack = True
    qh_per = 1 if not pack else hq // hkv
    n_m = ceil_div(seqlen_q * qh_per, kbm)
    use_cluster = bool(enable_cluster and n_m % 2 == 0)
    seqlens_q = [int(x) for x in (case.get("seqlens_q") or [])]
    combine_sparse = False
    if split and seqlens_q:
        total_q = sum(seqlens_q)
        max_q = max(seqlens_q)
        cbm = combine_block_m(hdv)
        lower = ceil_div(total_q, cbm)
        grid = len(seqlens_q) * ceil_div(max_q, cbm)
        combine_sparse = (2 * lower) < grid
    return {
        "kBlockM": kbm,
        "kBlockN": kbn,
        "mma_pv_is_rs": rs,
        "intra_wg_overlap": overlap,
        "enable_cluster": enable_cluster,
        "n_m_tiles": n_m,
        "cluster_m": 2 if use_cluster else 1,
        "pagedkv_tma": paged_tma,
        "pagedkv_nontma": paged_nontma,
        "pack_gqa": bool(pack),
        "fp8_transpose_v": dtype == "fp8_e4m3" and not v_colmajor,
        "has_qv": bool(case.get("qv")) and hd == 64 and hdv >= 256 and dtype != "fp8_e4m3",
        "dqacc_bulk_tma": operation in ("fixed_bwd", "varlen_bwd") and hd < 256,
        "dqacc_atomic_d256": operation in ("fixed_bwd", "varlen_bwd") and hd == 256,
        "combine_sparse": combine_sparse,
        "combine_dense": bool(split and seqlens_q and not combine_sparse),
        "element_size": elem,
        "is_local": local,
        "is_varlen": varlen,
        "is_split": split,
    }


EVIDENCE: dict[str, dict[str, Any]] = {
    "cluster_even": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h"],
        "symbols": ["Enable_cluster", "CLUSTER_SWITCH"],
        "condition": "SM90 && (16bit hd>=128 || fp8 hd==192) && !causal && !local && !split && !PagedKVNonTMA && !varlen && ceil(seqlen_q*gqa/kBlockM)%2==0 → ClusterM=2",
    },
    "cluster_odd": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h"],
        "symbols": ["Enable_cluster", "CLUSTER_SWITCH"],
        "condition": "cluster-eligible but m-tiles odd → ClusterM=1",
    },
    "mma_pv_ss_hd64": {
        "files": ["upstream/hopper/tile_size.h", "upstream/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp"],
        "symbols": ["tile_size_fwd_sm90", "MmaPV_is_RS"],
        "condition": "hd<=64 && dv not 256/512 && !causal && !local → BlockN=192, MmaPV_is_RS=false (SS)",
    },
    "mma_pv_rs_hd128": {
        "files": ["upstream/hopper/tile_size.h"],
        "symbols": ["tile_size_fwd_sm90", "MmaPV_is_RS"],
        "condition": "hd<=128 causal → BlockN=128, MmaPV_is_RS=true (RS); causal disables cluster",
    },
    "mma_pv_ss_hd96": {
        "files": ["upstream/hopper/tile_size.h"],
        "symbols": ["tile_size_fwd_sm90"],
        "condition": "hd<=96 16bit → MmaPV_is_RS=false, BlockM=192",
    },
    "fp8_transpose_hd96": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h", "upstream/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp"],
        "symbols": ["FP8_TransposeV", "Transpose_V"],
        "condition": "Is_FP8 && !V_colmajor; hd96 uses BlockN=128, LDSM.T/STSM transpose V",
    },
    "fp8_hd192_cluster": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h"],
        "symbols": ["Enable_cluster"],
        "condition": "fp8 Enable_cluster only when kHeadDim==192",
    },
    "paged_tma": {
        "files": ["upstream/hopper/flash_api.cpp"],
        "symbols": ["get_pagedkv_tma"],
        "condition": "page_table && !leftpad && !knew && page_size%kBlockN==0 && seqlen_q*(h/h_k)>kBlockM",
    },
    "paged_nontma": {
        "files": ["upstream/hopper/flash_api.cpp", "upstream/hopper/paged_kv.h"],
        "symbols": ["get_pagedkv_tma", "PagedKVManager"],
        "condition": "paged but page_size%kBlockN!=0 or seqlen_q too small → CpAsync PagedKVNonTMA",
    },
    "dv512_mla": {
        "files": ["upstream/hopper/tile_size.h", "upstream/hopper/generate_kernels.py"],
        "symbols": ["headdim_v == 512", "MmaPV_use_RS_WG1"],
        "condition": "hd=64 dv=512 → {64,64,SS}; SM90 bf16/fp16 only",
    },
    "dv256_qv": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h", "upstream/hopper/tile_size.h"],
        "symbols": ["HasQv", "headdim_v == 256"],
        "condition": "qv_ptr && hd==64 && dv>=256 && !FP8",
    },
    "hd192_dv128": {
        "files": ["upstream/hopper/flash_api.cpp", "upstream/hopper/generate_kernels.py"],
        "symbols": ["run_mha_fwd_<*>192, 128"],
        "condition": "QK (128,192] and V (96,128]",
    },
    "pack_gqa": {
        "files": ["upstream/hopper/heuristics.h", "upstream/hopper/flash_api.cpp"],
        "symbols": ["should_pack_gqa", "get_pack_gqa"],
        "condition": "explicit pack_gqa=true with HQ!=HKV",
    },
    "causal_lpt": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h"],
        "symbols": ["LPT", "DynamicPersistentTileScheduler"],
        "condition": "Is_causal → LPT scheduler; hd64 causal uses RS BlockN=128",
    },
    "local_window": {
        "files": ["upstream/hopper/flash_api.cpp", "upstream/hopper/static_switch.h"],
        "symbols": ["is_local", "CAUSAL_LOCAL_SWITCH"],
        "condition": "window_size_left/right >=0 && !causal",
    },
    "attention_chunk": {
        "files": ["upstream/hopper/flash_api.cpp"],
        "symbols": ["attention_chunk", "is_local"],
        "condition": "attention_chunk>=1 forces local window clamp",
    },
    "softcap": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h", "upstream/hopper/static_switch.h"],
        "symbols": ["Has_softcap", "SOFTCAP_SWITCH"],
        "condition": "softcap>0 templates Has_softcap",
    },
    "varlen_fwd": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h"],
        "symbols": ["VarlenDynamicPersistentTileScheduler", "VARLEN_SWITCH"],
        "condition": "cu_seqlens/seqused/leftpad → Varlen persistent scheduler",
    },
    "split_kv_fixed": {
        "files": ["upstream/hopper/flash_fwd_launch_template.h", "upstream/hopper/heuristics.h"],
        "symbols": ["UsePersistentScheduler", "SingleTileScheduler", "num_splits"],
        "condition": "Split && !Varlen → SingleTileScheduler then fwd_combine",
    },
    "split_varlen_dense_combine": {
        "files": ["upstream/hopper/flash_fwd_combine_kernel.h", "upstream/hopper/flash_api.cpp"],
        "symbols": ["choose_scheduling_algo", "STANDARD", "run_mha_fwd_combine"],
        "condition": "varlen+split; 2*ceil(total_q/kBlockM) >= B*ceil(max_q/kBlockM) → STANDARD",
    },
    "split_varlen_sparse_combine": {
        "files": ["upstream/hopper/flash_fwd_combine_kernel.h"],
        "symbols": ["LINEARIZE_M_AND_BATCH", "choose_scheduling_algo"],
        "condition": "varlen+split; grid >50% empty → linearize M and batch",
    },
    "kvcache_append_rotary": {
        "files": ["upstream/hopper/flash_api.cpp", "upstream/hopper/rotary.h"],
        "symbols": ["APPENDKV_SWITCH", "knew_ptr", "rotary_cos_ptr"],
        "condition": "k_new/v_new requires seqused_k; rotary at cache_seqlens",
    },
    "kvcache_paged_decode": {
        "files": ["upstream/hopper/flash_attn_interface.py", "upstream/hopper/flash_api.cpp"],
        "symbols": ["flash_attn_with_kvcache", "page_table"],
        "condition": "small seqlen_q paged decode; seqlen_q*(h/h_k)<=kBlockM disables TMA",
    },
    "kvcache_leftpad": {
        "files": ["upstream/hopper/flash_api.cpp"],
        "symbols": ["leftpad_k", "get_pagedkv_tma"],
        "condition": "leftpad_k forces pagedkv_tma=false",
    },
    "bwd_deterministic": {
        "files": ["upstream/hopper/flash_bwd_launch_template.h", "upstream/hopper/flash_api.cpp"],
        "symbols": ["Deterministic", "dk_semaphore"],
        "condition": "deterministic && hd<256; GQA allocates dk/dv semaphores",
    },
    "bwd_dqacc_bulk": {
        "files": ["upstream/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp"],
        "symbols": ["dQacc_use_TMA", "SM90_BULK_REDUCE_ADD"],
        "condition": "dQacc_use_TMA = kHeadDim<256",
    },
    "bwd_dqacc_atomic_d256": {
        "files": ["upstream/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp"],
        "symbols": ["dQacc_use_TMA", "Slice_dQKV_Mma", "atomicAdd"],
        "condition": "kHeadDim==256 → atomicAdd dQaccum; deterministic unsupported",
    },
    "bwd_gqa_varlen": {
        "files": ["upstream/hopper/flash_bwd_launch_template.h", "upstream/hopper/epilogue_bwd.hpp"],
        "symbols": ["GQA", "Varlen", "CollectiveEpilogueBwdGQA"],
        "condition": "varlen bwd with HQ!=HKV",
    },
    "bwd_causal_softcap": {
        "files": ["upstream/hopper/flash_bwd_launch_template.h"],
        "symbols": ["run_mha_bwd_hdim64", "run_mha_bwd_hdim128", "run_mha_bwd_hdim192", "Has_softcap"],
        "condition": "causal+softcap bwd. hd64 uses 96x128 (not 128x128); hd128 uses run_mha_bwd_hdim128; hd192 uses 64x96. Secondary: bwd_softcap_hdim64_96x128 / bwd_hdim128 / bwd_hdim192",
    },
    "combine_standalone": {
        "files": ["upstream/hopper/flash_attn_interface.py", "upstream/hopper/flash_api.cpp"],
        "symbols": ["flash_attn_combine", "mha_combine"],
        "condition": "public dense fwd_combine; no cu_seqlens",
    },
    "hd256_fwd": {
        "files": ["upstream/hopper/tile_size.h", "upstream/hopper/generate_kernels.py"],
        "symbols": ["headdim <= 256", "MmaPV_is_RS"],
        "condition": "hd256 fwd BlockN=80 RS; causal so cluster stays off",
    },
    "reuse_scheduler_metadata": {
        "files": ["upstream/hopper/flash_attn_interface.py", "upstream/hopper/flash_api.cpp"],
        "symbols": ["scheduler_metadata", "skip_scheduler_metadata_computation"],
        "condition": "kvcache reuse_scheduler_metadata reuses prepared varlen/split metadata",
    },
}



def doubling(start: int, cap: int, count: int = 4) -> list[int]:
    values: list[int] = []
    value = max(1, int(start))
    for _ in range(count):
        if value > cap:
            break
        values.append(value)
        value *= 2
    if len(values) < 2:
        values.append(int(start) * 2)
    return values


def choose_tuning(case: dict[str, Any], primary: str) -> tuple[str, list[int]]:
    if case.get("seqlens_q") or case.get("seqlens_k"):
        return "repeat_seqlens_batch", [1, 2, 4, 8]
    if primary in ("split_kv_fixed", "combine_standalone") or int(case.get("num_splits") or 1) > 1:
        return "num_splits", doubling(max(2, int(case["num_splits"])), 128)
    page_size = case.get("page_size")
    if page_size:
        step = int(page_size)
        start = int(case["seqlen_k"])
        if start % step != 0:
            raise ValueError(f"{case.get('case_id', case.get('primary_mechanism'))} seqlen_k={start} not multiple of page_size={step}")
        values = []
        value = start
        for _ in range(4):
            values.append(value)
            value *= 2
            if value > 16384:
                break
        return "seqlen_k", values
    return "batch", doubling(int(case["batch"]), 64)


def apply_tuning(case: dict[str, Any], value: int) -> dict[str, Any]:
    """Return a copy with one legal tuning value applied. Trigger must stay valid."""
    tuning = case["tuning"]
    axis = tuning["axis"]
    allowed = [int(v) for v in tuning["values"]]
    value = int(value)
    if value not in allowed:
        raise ValueError(f"{case.get('case_id')} tuning value {value} not in {allowed}")
    out = dict(case)
    for key in ("seqlens_q", "seqlens_k", "window_size", "mechanisms"):
        if isinstance(out.get(key), list):
            out[key] = list(out[key])
    if axis == "batch":
        out["batch"] = value
    elif axis == "seqlen_k":
        out["seqlen_k"] = value
    elif axis == "num_splits":
        if value < 2:
            raise ValueError("num_splits tuning must stay >= 2")
        out["num_splits"] = value
    elif axis == "repeat_seqlens_batch":
        if value < 1:
            raise ValueError("repeat factor must be >= 1")
        if case.get("seqlens_q"):
            out["seqlens_q"] = list(case["seqlens_q"]) * value
            out["batch"] = len(out["seqlens_q"])
            out["seqlen_q"] = max(out["seqlens_q"])
        if case.get("seqlens_k"):
            used_max = max(int(x) for x in case["seqlens_k"])
            spare = max(0, int(case["seqlen_k"]) - used_max)
            out["seqlens_k"] = list(case["seqlens_k"]) * value
            if not case.get("seqlens_q"):
                out["batch"] = len(out["seqlens_k"])
            # Keep cache capacity spare (append_kv room). Do not collapse to max used.
            out["seqlen_k"] = max(out["seqlens_k"]) + spare
        if not case.get("seqlens_q") and not case.get("seqlens_k"):
            raise ValueError(f"{case.get('case_id')} repeat_seqlens_batch requires seqlens_q or seqlens_k")
    else:
        raise ValueError(f"unsupported tuning axis {axis}")
    traits = derive_traits(out)
    if not PRIMARY_OK[out["primary_mechanism"]](out, traits):
        raise RuntimeError(f"{out.get('case_id')} {axis}={value} broke trigger {traits}")
    out["trigger"] = traits
    return out


def _api(operation: str) -> str:
    return {
        "fixed_fwd": "flash_attn_func",
        "fixed_bwd": "flash_attn_func",
        "varlen_fwd": "flash_attn_varlen_func",
        "varlen_bwd": "flash_attn_varlen_func",
        "kvcache": "flash_attn_with_kvcache",
        "combine": "flash_attn_combine",
    }[operation]


def mk(
    primary: str,
    operation: str,
    dtype: str,
    *,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    heads_q: int,
    heads_kv: int | None = None,
    head_dim: int,
    head_dim_v: int | None = None,
    causal: bool = False,
    window_size: list[int] | None = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    pack_gqa: bool | None = None,
    num_splits: int = 1,
    deterministic: bool = False,
    qv: bool = False,
    page_size: int | None = None,
    append_kv: bool = False,
    seqlen_new: int | None = None,
    rotary_dim: int = 0,
    rotary_interleaved: bool = True,
    cache_leftpad: bool = False,
    cache_batch_idx: bool = False,
    reuse_scheduler_metadata: bool = False,
    seqlens_q: list[int] | None = None,
    seqlens_k: list[int] | None = None,
    extra_mechanisms: list[str] | None = None,
    tuning_axis: str | None = None,
    tuning_values: list[int] | None = None,
) -> dict[str, Any]:
    heads_kv = int(heads_kv if heads_kv is not None else heads_q)
    head_dim_v = int(head_dim_v if head_dim_v is not None else head_dim)
    used_k = max(int(x) for x in seqlens_k) if seqlens_k else 0
    if seqlens_q:
        batch = len(seqlens_q)
        seqlen_q = max(seqlens_q)
    if seqlens_k and not seqlens_q:
        batch = len(seqlens_k)
    seqlen_k = max(int(seqlen_k), used_k)
    if append_kv:
        seqlen_k = max(seqlen_k, used_k + int(seqlen_new or 0))
    case: dict[str, Any] = {
        "operation": operation,
        "dtype": dtype,
        "primary_mechanism": primary,
        "delivery_group": 0,
        "head_dim": int(head_dim),
        "head_dim_v": head_dim_v,
        "heads_q": int(heads_q),
        "heads_kv": heads_kv,
        "batch": int(batch),
        "seqlen_q": int(seqlen_q),
        "seqlen_k": int(seqlen_k),
        "causal": bool(causal),
        "window_size": list(window_size or [-1, -1]),
        "attention_chunk": int(attention_chunk),
        "softcap": float(softcap),
        "pack_gqa": pack_gqa,
        "num_splits": int(num_splits),
        "deterministic": bool(deterministic),
        "qv": bool(qv),
        "page_size": page_size,
        "append_kv": bool(append_kv),
        "seqlen_new": seqlen_new,
        "rotary_dim": int(rotary_dim),
        "rotary_interleaved": bool(rotary_interleaved),
        "cache_leftpad": bool(cache_leftpad),
        "cache_batch_idx": bool(cache_batch_idx),
        "reuse_scheduler_metadata": bool(reuse_scheduler_metadata),
        "source": INTERFACE,
        "api": _api(operation),
        "suite": "flashattention3",
        "implementation": "FA3",
        "qualification_status": "candidate",
        "source_revision": SOURCE_REVISION,
        "cycles_hint_band": [CYCLES_MIN, CYCLES_MAX],
        "source_evidence": dict(EVIDENCE[primary]),
    }
    if seqlens_q is not None:
        case["seqlens_q"] = [int(x) for x in seqlens_q]
    if seqlens_k is not None:
        case["seqlens_k"] = [int(x) for x in seqlens_k]
    axis, values = choose_tuning(case, primary)
    if tuning_axis is not None:
        axis = tuning_axis
    if tuning_values is not None:
        values = [int(v) for v in tuning_values]
    if axis not in TUNING_AXES:
        raise ValueError(f"{primary} bad tuning axis {axis}")
    if values != sorted(values) or len(values) != len(set(values)) or len(values) < 2:
        raise ValueError(f"{primary} tuning.values must be unique ordered, got {values}")
    case["tuning"] = {
        "axis": axis,
        "values": values,
        "rule": TUNING_RULE,
        "forbidden": "do not repeat the same launch / loop the kernel to inflate cycles",
    }
    traits = derive_traits(case)
    case["trigger"] = traits
    case["mechanisms"] = collect_mechanisms(case, traits, extra_mechanisms)
    return case


def collect_mechanisms(
    case: dict[str, Any],
    traits: dict[str, Any] | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    traits = traits if traits is not None else derive_traits(case)
    mechanisms = [case["primary_mechanism"]]
    if traits["cluster_m"] == 2:
        mechanisms.append("cluster_m2")
    if traits["mma_pv_is_rs"]:
        mechanisms.append("mma_pv_rs")
    else:
        mechanisms.append("mma_pv_ss")
    if traits["pagedkv_tma"]:
        mechanisms.append("paged_tma")
    if traits["pagedkv_nontma"]:
        mechanisms.append("paged_nontma")
    if traits["fp8_transpose_v"]:
        mechanisms.append("fp8_transpose_v")
    if traits["is_split"]:
        mechanisms.append("split_kv")
    if traits["combine_sparse"]:
        mechanisms.append("combine_sparse")
    if traits["combine_dense"]:
        mechanisms.append("combine_dense")
    if bool(case.get("pack_gqa")):
        mechanisms.append("pack_gqa")
    if int(case["heads_q"]) != int(case["heads_kv"]):
        mechanisms.append("gqa")
    if case["operation"] in ("fixed_bwd", "varlen_bwd"):
        hd = int(case["head_dim"])
        if hd == 64 and bool(case.get("causal")) and float(case.get("softcap") or 0) > 0:
            mechanisms.append("bwd_softcap_hdim64_96x128")
        if hd == 128:
            mechanisms.append("bwd_hdim128")
        if hd == 192:
            mechanisms.append("bwd_hdim192")
    if extra:
        mechanisms.extend(extra)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in mechanisms:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


PRIMARY_OK: dict[str, Callable[[dict, dict], bool]] = {
    "cluster_even": lambda c, t: t["cluster_m"] == 2 and c["operation"] == "fixed_fwd",
    "cluster_odd": lambda c, t: t["enable_cluster"] and t["cluster_m"] == 1,
    "mma_pv_ss_hd64": lambda c, t: c["head_dim"] == 64 and t["mma_pv_is_rs"] is False and not c["causal"],
    "mma_pv_rs_hd128": lambda c, t: c["head_dim"] == 128 and t["mma_pv_is_rs"] is True and c["causal"] and t["cluster_m"] == 1,
    "mma_pv_ss_hd96": lambda c, t: c["head_dim"] == 96 and t["mma_pv_is_rs"] is False and c["dtype"] != "fp8_e4m3",
    "fp8_transpose_hd96": lambda c, t: c["dtype"] == "fp8_e4m3" and c["head_dim"] == 96 and t["fp8_transpose_v"],
    "fp8_hd192_cluster": lambda c, t: c["dtype"] == "fp8_e4m3" and c["head_dim"] == 192 and t["cluster_m"] == 2,
    "paged_tma": lambda c, t: t["pagedkv_tma"] is True,
    "paged_nontma": lambda c, t: t["pagedkv_nontma"] is True and not bool(c.get("cache_leftpad")),
    "dv512_mla": lambda c, t: c["head_dim"] == 64 and int(c["head_dim_v"]) == 512,
    "dv256_qv": lambda c, t: t["has_qv"] and int(c["head_dim_v"]) == 256,
    "hd192_dv128": lambda c, t: c["head_dim"] == 192 and int(c["head_dim_v"]) == 128,
    "pack_gqa": lambda c, t: bool(c.get("pack_gqa")) and int(c["heads_q"]) != int(c["heads_kv"]),
    "causal_lpt": lambda c, t: bool(c["causal"]) and c["head_dim"] == 64 and c["operation"] == "fixed_fwd",
    "local_window": lambda c, t: t["is_local"] and int(c.get("attention_chunk") or 0) == 0,
    "attention_chunk": lambda c, t: int(c.get("attention_chunk") or 0) >= 1 and t["is_local"],
    "softcap": lambda c, t: float(c.get("softcap") or 0) > 0 and c["operation"] == "fixed_fwd",
    "varlen_fwd": lambda c, t: c["operation"] == "varlen_fwd" and int(c.get("num_splits") or 1) == 1,
    "split_kv_fixed": lambda c, t: t["is_split"] and c["operation"] == "fixed_fwd",
    "split_varlen_dense_combine": lambda c, t: t["is_split"] and t["combine_dense"],
    "split_varlen_sparse_combine": lambda c, t: t["is_split"] and t["combine_sparse"],
    "kvcache_append_rotary": lambda c, t: c["operation"] == "kvcache" and bool(c.get("append_kv")) and int(c.get("rotary_dim") or 0) > 0,
    "kvcache_paged_decode": lambda c, t: c["operation"] == "kvcache" and int(c.get("page_size") or 0) > 0 and int(c["seqlen_q"]) <= 8,
    "kvcache_leftpad": lambda c, t: c["operation"] == "kvcache" and bool(c.get("cache_leftpad")),
    "bwd_deterministic": lambda c, t: c["operation"] == "fixed_bwd" and bool(c.get("deterministic")) and c["head_dim"] < 256,
    "bwd_dqacc_bulk": lambda c, t: t["dqacc_bulk_tma"] and c["operation"] == "fixed_bwd" and not c.get("deterministic"),
    "bwd_dqacc_atomic_d256": lambda c, t: t["dqacc_atomic_d256"] and c["operation"] == "fixed_bwd",
    "bwd_gqa_varlen": lambda c, t: c["operation"] == "varlen_bwd" and int(c["heads_q"]) != int(c["heads_kv"]),
    "bwd_causal_softcap": lambda c, t: c["operation"] == "fixed_bwd" and c["causal"] and float(c.get("softcap") or 0) > 0,
    "combine_standalone": lambda c, t: c["operation"] == "combine",
    "hd256_fwd": lambda c, t: c["operation"] == "fixed_fwd" and c["head_dim"] == 256 and c["causal"],
    "reuse_scheduler_metadata": lambda c, t: c["operation"] == "kvcache" and bool(c.get("reuse_scheduler_metadata")),
}


def _seeds() -> dict[str, list[dict[str, Any]]]:
    s: dict[str, list[dict[str, Any]]] = {}
    s["cluster_even"] = [
        mk("cluster_even", "fixed_fwd", "bf16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=128),
        mk("cluster_even", "fixed_fwd", "fp16", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=16, head_dim=128),
        mk("cluster_even", "fixed_fwd", "bf16", batch=4, seqlen_q=2048, seqlen_k=1024, heads_q=8, head_dim=192),
        mk("cluster_even", "fixed_fwd", "fp16", batch=8, seqlen_q=768, seqlen_k=1024, heads_q=16, head_dim=128),
    ]
    s["cluster_odd"] = [
        mk("cluster_odd", "fixed_fwd", "bf16", batch=4, seqlen_q=1152, seqlen_k=1024, heads_q=16, head_dim=128),
        mk("cluster_odd", "fixed_fwd", "fp16", batch=2, seqlen_q=896, seqlen_k=1536, heads_q=16, head_dim=128),
        mk("cluster_odd", "fixed_fwd", "bf16", batch=4, seqlen_q=1408, seqlen_k=1024, heads_q=8, head_dim=192),
        mk("cluster_odd", "fixed_fwd", "fp16", batch=2, seqlen_q=640, seqlen_k=2048, heads_q=16, head_dim=128),
    ]
    s["mma_pv_ss_hd64"] = [
        mk("mma_pv_ss_hd64", "fixed_fwd", "bf16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=64),
        mk("mma_pv_ss_hd64", "fixed_fwd", "fp16", batch=8, seqlen_q=768, seqlen_k=768, heads_q=16, head_dim=64),
        mk("mma_pv_ss_hd64", "fixed_fwd", "bf16", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=32, head_dim=64),
        mk("mma_pv_ss_hd64", "fixed_fwd", "fp16", batch=4, seqlen_q=1280, seqlen_k=1280, heads_q=8, head_dim=64),
    ]
    s["mma_pv_rs_hd128"] = [
        mk("mma_pv_rs_hd128", "fixed_fwd", "bf16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=128, causal=True),
        mk("mma_pv_rs_hd128", "fixed_fwd", "fp16", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=16, head_dim=128, causal=True),
        mk("mma_pv_rs_hd128", "fixed_fwd", "bf16", batch=4, seqlen_q=2048, seqlen_k=2048, heads_q=8, head_dim=128, causal=True),
        mk("mma_pv_rs_hd128", "fixed_fwd", "fp16", batch=8, seqlen_q=768, seqlen_k=768, heads_q=16, head_dim=128, causal=True),
    ]
    s["mma_pv_ss_hd96"] = [
        mk("mma_pv_ss_hd96", "fixed_fwd", "bf16", batch=4, seqlen_q=960, seqlen_k=960, heads_q=16, head_dim=96),
        mk("mma_pv_ss_hd96", "fixed_fwd", "fp16", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=16, head_dim=96),
        mk("mma_pv_ss_hd96", "fixed_fwd", "bf16", batch=4, seqlen_q=1152, seqlen_k=1152, heads_q=8, head_dim=96),
        mk("mma_pv_ss_hd96", "fixed_fwd", "fp16", batch=8, seqlen_q=768, seqlen_k=768, heads_q=16, head_dim=96),
    ]
    s["fp8_transpose_hd96"] = [
        mk("fp8_transpose_hd96", "fixed_fwd", "fp8_e4m3", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=96),
        mk("fp8_transpose_hd96", "fixed_fwd", "fp8_e4m3", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=16, head_dim=96),
        mk("fp8_transpose_hd96", "fixed_fwd", "fp8_e4m3", batch=4, seqlen_q=768, seqlen_k=1536, heads_q=32, head_dim=96),
        mk("fp8_transpose_hd96", "fixed_fwd", "fp8_e4m3", batch=8, seqlen_q=576, seqlen_k=1152, heads_q=16, head_dim=96),
    ]
    s["fp8_hd192_cluster"] = [
        mk("fp8_hd192_cluster", "fixed_fwd", "fp8_e4m3", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=192),
        mk("fp8_hd192_cluster", "fixed_fwd", "fp8_e4m3", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=8, head_dim=192),
        mk("fp8_hd192_cluster", "fixed_fwd", "fp8_e4m3", batch=4, seqlen_q=768, seqlen_k=1536, heads_q=16, head_dim=192),
        mk("fp8_hd192_cluster", "fixed_fwd", "fp8_e4m3", batch=2, seqlen_q=2048, seqlen_k=1024, heads_q=8, head_dim=192),
    ]
    s["paged_tma"] = [
        mk("paged_tma", "kvcache", "bf16", batch=4, seqlen_q=256, seqlen_k=2048, heads_q=16, head_dim=128, causal=True, page_size=128, seqlens_k=[2048, 2048, 1536, 1024]),
        mk("paged_tma", "kvcache", "fp16", batch=2, seqlen_q=384, seqlen_k=4096, heads_q=16, head_dim=128, causal=True, page_size=128, seqlens_k=[4096, 3072]),
        mk("paged_tma", "kvcache", "bf16", batch=8, seqlen_q=256, seqlen_k=1536, heads_q=8, head_dim=128, causal=True, page_size=128),
        mk("paged_tma", "kvcache", "fp16", batch=4, seqlen_q=256, seqlen_k=2048, heads_q=16, head_dim=64, causal=True, page_size=128),
    ]
    s["paged_nontma"] = [
        mk("paged_nontma", "kvcache", "bf16", batch=4, seqlen_q=256, seqlen_k=1920, heads_q=16, head_dim=128, causal=True, page_size=96),
        mk("paged_nontma", "kvcache", "fp16", batch=2, seqlen_q=128, seqlen_k=2880, heads_q=16, head_dim=128, causal=True, page_size=96),
        mk("paged_nontma", "kvcache", "bf16", batch=8, seqlen_q=64, seqlen_k=1920, heads_q=8, head_dim=64, page_size=96),
        mk("paged_nontma", "kvcache", "fp16", batch=4, seqlen_q=192, seqlen_k=3840, heads_q=16, head_dim=96, causal=True, page_size=96),
    ]
    s["dv512_mla"] = [
        mk("dv512_mla", "fixed_fwd", "bf16", batch=2, seqlen_q=512, seqlen_k=1024, heads_q=8, heads_kv=1, head_dim=64, head_dim_v=512, extra_mechanisms=["gqa"]),
        mk("dv512_mla", "fixed_fwd", "fp16", batch=4, seqlen_q=384, seqlen_k=768, heads_q=8, heads_kv=1, head_dim=64, head_dim_v=512),
        mk("dv512_mla", "fixed_fwd", "bf16", batch=2, seqlen_q=768, seqlen_k=1536, heads_q=4, heads_kv=1, head_dim=64, head_dim_v=512, causal=True),
        mk("dv512_mla", "fixed_fwd", "fp16", batch=1, seqlen_q=1024, seqlen_k=1024, heads_q=8, heads_kv=2, head_dim=64, head_dim_v=512),
    ]
    s["dv256_qv"] = [
        mk("dv256_qv", "fixed_fwd", "bf16", batch=2, seqlen_q=512, seqlen_k=1024, heads_q=8, head_dim=64, head_dim_v=256, qv=True),
        mk("dv256_qv", "fixed_fwd", "fp16", batch=4, seqlen_q=384, seqlen_k=768, heads_q=8, head_dim=64, head_dim_v=256, qv=True),
        mk("dv256_qv", "fixed_fwd", "bf16", batch=2, seqlen_q=768, seqlen_k=768, heads_q=16, head_dim=64, head_dim_v=256, qv=True, causal=True),
        mk("dv256_qv", "fixed_fwd", "fp16", batch=4, seqlen_q=256, seqlen_k=1024, heads_q=8, heads_kv=2, head_dim=64, head_dim_v=256, qv=True),
    ]
    s["hd192_dv128"] = [
        mk("hd192_dv128", "fixed_fwd", "bf16", batch=2, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=192, head_dim_v=128),
        mk("hd192_dv128", "fixed_fwd", "fp16", batch=4, seqlen_q=768, seqlen_k=768, heads_q=8, head_dim=192, head_dim_v=128),
        mk("hd192_dv128", "fixed_fwd", "bf16", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=8, head_dim=192, head_dim_v=128, causal=True),
        mk("hd192_dv128", "fixed_fwd", "fp16", batch=4, seqlen_q=1024, seqlen_k=2048, heads_q=8, head_dim=192, head_dim_v=128),
    ]
    s["pack_gqa"] = [
        mk("pack_gqa", "fixed_fwd", "bf16", batch=4, seqlen_q=256, seqlen_k=2048, heads_q=32, heads_kv=4, head_dim=128, pack_gqa=True, causal=True),
        mk("pack_gqa", "fixed_fwd", "fp16", batch=2, seqlen_q=128, seqlen_k=4096, heads_q=32, heads_kv=8, head_dim=128, pack_gqa=True),
        mk("pack_gqa", "fixed_fwd", "bf16", batch=4, seqlen_q=192, seqlen_k=1536, heads_q=16, heads_kv=2, head_dim=64, pack_gqa=True, causal=True),
        mk("pack_gqa", "fixed_fwd", "fp16", batch=8, seqlen_q=64, seqlen_k=2048, heads_q=32, heads_kv=4, head_dim=128, pack_gqa=True),
    ]
    s["causal_lpt"] = [
        mk("causal_lpt", "fixed_fwd", "bf16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=64, causal=True),
        mk("causal_lpt", "fixed_fwd", "fp16", batch=2, seqlen_q=2048, seqlen_k=2048, heads_q=16, head_dim=64, causal=True),
        mk("causal_lpt", "fixed_fwd", "bf16", batch=4, seqlen_q=1536, seqlen_k=1536, heads_q=8, head_dim=64, causal=True),
        mk("causal_lpt", "fixed_fwd", "fp16", batch=8, seqlen_q=768, seqlen_k=768, heads_q=16, head_dim=64, causal=True),
    ]
    s["local_window"] = [
        mk("local_window", "fixed_fwd", "bf16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=64, window_size=[256, 256]),
        mk("local_window", "fixed_fwd", "fp16", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=16, head_dim=128, window_size=[128, 128]),
        mk("local_window", "fixed_fwd", "bf16", batch=4, seqlen_q=2048, seqlen_k=2048, heads_q=8, head_dim=64, window_size=[512, 0]),
        mk("local_window", "fixed_fwd", "fp16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=96, window_size=[192, 192]),
    ]
    s["attention_chunk"] = [
        mk("attention_chunk", "fixed_fwd", "bf16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=64, attention_chunk=256),
        mk("attention_chunk", "fixed_fwd", "fp16", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=16, head_dim=128, attention_chunk=512),
        mk("attention_chunk", "fixed_fwd", "bf16", batch=4, seqlen_q=2048, seqlen_k=2048, heads_q=8, head_dim=64, attention_chunk=128),
        mk("attention_chunk", "fixed_fwd", "fp16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=96, attention_chunk=192),
    ]
    s["softcap"] = [
        mk("softcap", "fixed_fwd", "bf16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=64, softcap=50.0),
        mk("softcap", "fixed_fwd", "fp16", batch=2, seqlen_q=1536, seqlen_k=1536, heads_q=16, head_dim=128, softcap=30.0, causal=True),
        mk("softcap", "fixed_fwd", "bf16", batch=4, seqlen_q=768, seqlen_k=1536, heads_q=16, head_dim=96, softcap=50.0),
        mk("softcap", "fixed_fwd", "fp16", batch=8, seqlen_q=512, seqlen_k=1024, heads_q=16, head_dim=64, softcap=20.0),
    ]
    s["varlen_fwd"] = [
        mk("varlen_fwd", "varlen_fwd", "bf16", batch=8, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=128, seqlens_q=[1024, 768, 1024, 512, 896, 640, 1024, 768], seqlens_k=[1024, 1024, 768, 1024, 896, 1024, 640, 768]),
        mk("varlen_fwd", "varlen_fwd", "fp16", batch=4, seqlen_q=1536, seqlen_k=1536, heads_q=16, head_dim=64, causal=True, seqlens_q=[1536, 1024, 1280, 768], seqlens_k=[1536, 1536, 1024, 1280]),
        mk("varlen_fwd", "varlen_fwd", "bf16", batch=8, seqlen_q=768, seqlen_k=2048, heads_q=32, heads_kv=8, head_dim=128, pack_gqa=True, seqlens_q=[768, 512, 768, 256, 640, 768, 384, 512], seqlens_k=[2048, 1536, 2048, 1024, 1792, 2048, 1280, 1536]),
        mk("varlen_fwd", "varlen_fwd", "fp16", batch=6, seqlen_q=1152, seqlen_k=1152, heads_q=8, head_dim=96, seqlens_q=[1152, 768, 960, 512, 1152, 640], seqlens_k=[1152, 1152, 768, 1024, 896, 1152]),
    ]
    s["split_kv_fixed"] = [
        mk("split_kv_fixed", "fixed_fwd", "bf16", batch=1, seqlen_q=256, seqlen_k=4096, heads_q=8, head_dim=128, num_splits=4),
        mk("split_kv_fixed", "fixed_fwd", "fp16", batch=1, seqlen_q=128, seqlen_k=8192, heads_q=16, head_dim=64, num_splits=8),
        mk("split_kv_fixed", "fixed_fwd", "bf16", batch=2, seqlen_q=192, seqlen_k=4096, heads_q=8, head_dim=128, num_splits=4),
        mk("split_kv_fixed", "fixed_fwd", "fp16", batch=1, seqlen_q=256, seqlen_k=6144, heads_q=8, head_dim=96, num_splits=6),
    ]
    s["split_varlen_dense_combine"] = [
        mk("split_varlen_dense_combine", "varlen_fwd", "bf16", batch=8, seqlen_q=1024, seqlen_k=2048, heads_q=16, head_dim=128, num_splits=4, seqlens_q=[1024] * 8, seqlens_k=[2048] * 8),
        mk("split_varlen_dense_combine", "varlen_fwd", "fp16", batch=4, seqlen_q=768, seqlen_k=3072, heads_q=8, head_dim=64, num_splits=8, seqlens_q=[768, 768, 640, 768], seqlens_k=[3072, 3072, 2560, 3072]),
        mk("split_varlen_dense_combine", "varlen_fwd", "bf16", batch=6, seqlen_q=512, seqlen_k=2048, heads_q=16, head_dim=128, num_splits=4, seqlens_q=[512] * 6, seqlens_k=[2048] * 6, causal=True),
        mk("split_varlen_dense_combine", "varlen_fwd", "fp16", batch=8, seqlen_q=640, seqlen_k=2560, heads_q=8, head_dim=96, num_splits=6, seqlens_q=[640] * 8, seqlens_k=[2560] * 8),
    ]
    s["split_varlen_sparse_combine"] = [
        mk("split_varlen_sparse_combine", "varlen_fwd", "bf16", batch=32, seqlen_q=1024, seqlen_k=2048, heads_q=8, head_dim=128, num_splits=4, seqlens_q=[16] * 31 + [1024], seqlens_k=[64] * 31 + [2048]),
        mk("split_varlen_sparse_combine", "varlen_fwd", "fp16", batch=24, seqlen_q=768, seqlen_k=1536, heads_q=8, head_dim=64, num_splits=8, seqlens_q=[8] * 23 + [768], seqlens_k=[32] * 23 + [1536]),
        mk("split_varlen_sparse_combine", "varlen_fwd", "bf16", batch=16, seqlen_q=1024, seqlen_k=4096, heads_q=16, heads_kv=4, head_dim=128, num_splits=4, pack_gqa=True, seqlens_q=[32] * 15 + [1024], seqlens_k=[128] * 15 + [4096]),
        mk("split_varlen_sparse_combine", "varlen_fwd", "fp16", batch=32, seqlen_q=512, seqlen_k=2048, heads_q=8, head_dim=96, num_splits=6, seqlens_q=[16] * 31 + [512], seqlens_k=[64] * 31 + [2048]),
    ]
    s["kvcache_append_rotary"] = [
        mk("kvcache_append_rotary", "kvcache", "bf16", batch=4, seqlen_q=16, seqlen_k=2560, heads_q=16, heads_kv=4, head_dim=128, causal=True, append_kv=True, seqlen_new=16, rotary_dim=128, seqlens_k=[1024, 1536, 2048, 768]),
        mk("kvcache_append_rotary", "kvcache", "fp16", batch=8, seqlen_q=8, seqlen_k=4096, heads_q=32, heads_kv=8, head_dim=64, causal=True, append_kv=True, seqlen_new=8, rotary_dim=64),
        mk("kvcache_append_rotary", "kvcache", "bf16", batch=4, seqlen_q=32, seqlen_k=2048, heads_q=16, heads_kv=4, head_dim=128, causal=True, append_kv=True, seqlen_new=32, rotary_dim=128, rotary_interleaved=False),
        mk("kvcache_append_rotary", "kvcache", "fp16", batch=2, seqlen_q=64, seqlen_k=3072, heads_q=16, heads_kv=4, head_dim=128, causal=True, append_kv=True, seqlen_new=64, rotary_dim=128),
    ]
    s["kvcache_paged_decode"] = [
        mk("kvcache_paged_decode", "kvcache", "bf16", batch=16, seqlen_q=1, seqlen_k=4096, heads_q=32, heads_kv=8, head_dim=128, causal=True, page_size=128),
        mk("kvcache_paged_decode", "kvcache", "fp16", batch=32, seqlen_q=1, seqlen_k=2048, heads_q=16, heads_kv=2, head_dim=64, causal=True, page_size=64),
        mk("kvcache_paged_decode", "kvcache", "bf16", batch=8, seqlen_q=4, seqlen_k=8192, heads_q=32, heads_kv=8, head_dim=128, causal=True, page_size=256),
        mk("kvcache_paged_decode", "kvcache", "fp16", batch=16, seqlen_q=8, seqlen_k=4032, heads_q=16, heads_kv=4, head_dim=96, causal=True, page_size=96),
    ]
    s["kvcache_leftpad"] = [
        mk("kvcache_leftpad", "kvcache", "bf16", batch=8, seqlen_q=16, seqlen_k=2048, heads_q=16, head_dim=128, causal=True, cache_leftpad=True, seqlens_k=[1024, 2048, 1536, 768, 2048, 1280, 896, 1792]),
        mk("kvcache_leftpad", "kvcache", "fp16", batch=4, seqlen_q=32, seqlen_k=4096, heads_q=8, head_dim=64, causal=True, cache_leftpad=True),
        mk("kvcache_leftpad", "kvcache", "bf16", batch=8, seqlen_q=8, seqlen_k=3072, heads_q=16, head_dim=128, causal=True, page_size=128, cache_leftpad=True),
        mk("kvcache_leftpad", "kvcache", "fp16", batch=4, seqlen_q=16, seqlen_k=2048, heads_q=16, head_dim=96, causal=True, cache_leftpad=True),
    ]
    s["bwd_deterministic"] = [
        mk("bwd_deterministic", "fixed_bwd", "bf16", batch=2, seqlen_q=768, seqlen_k=768, heads_q=8, head_dim=64, causal=True, deterministic=True),
        mk("bwd_deterministic", "fixed_bwd", "fp16", batch=2, seqlen_q=512, seqlen_k=512, heads_q=8, head_dim=128, causal=True, deterministic=True),
        mk("bwd_deterministic", "fixed_bwd", "bf16", batch=1, seqlen_q=1024, seqlen_k=1024, heads_q=8, heads_kv=2, head_dim=64, deterministic=True),
        mk("bwd_deterministic", "fixed_bwd", "fp16", batch=2, seqlen_q=640, seqlen_k=640, heads_q=8, head_dim=96, causal=True, deterministic=True),
    ]
    s["bwd_dqacc_bulk"] = [
        mk("bwd_dqacc_bulk", "fixed_bwd", "bf16", batch=2, seqlen_q=768, seqlen_k=768, heads_q=8, head_dim=128, causal=True),
        mk("bwd_dqacc_bulk", "fixed_bwd", "fp16", batch=2, seqlen_q=1024, seqlen_k=1024, heads_q=8, head_dim=64),
        mk("bwd_dqacc_bulk", "fixed_bwd", "bf16", batch=1, seqlen_q=1536, seqlen_k=1536, heads_q=8, head_dim=96, causal=True),
        mk("bwd_dqacc_bulk", "fixed_bwd", "fp16", batch=2, seqlen_q=512, seqlen_k=512, heads_q=16, head_dim=128),
    ]
    s["bwd_dqacc_atomic_d256"] = [
        mk("bwd_dqacc_atomic_d256", "fixed_bwd", "bf16", batch=1, seqlen_q=512, seqlen_k=512, heads_q=8, head_dim=256, causal=True),
        mk("bwd_dqacc_atomic_d256", "fixed_bwd", "fp16", batch=1, seqlen_q=768, seqlen_k=768, heads_q=4, head_dim=256),
        mk("bwd_dqacc_atomic_d256", "fixed_bwd", "bf16", batch=2, seqlen_q=384, seqlen_k=384, heads_q=8, head_dim=256, causal=True),
        mk("bwd_dqacc_atomic_d256", "fixed_bwd", "fp16", batch=1, seqlen_q=640, seqlen_k=640, heads_q=8, head_dim=256),
    ]
    s["bwd_gqa_varlen"] = [
        mk("bwd_gqa_varlen", "varlen_bwd", "bf16", batch=4, seqlen_q=768, seqlen_k=768, heads_q=16, heads_kv=4, head_dim=64, causal=True, seqlens_q=[768, 512, 640, 768], seqlens_k=[768, 768, 512, 640]),
        mk("bwd_gqa_varlen", "varlen_bwd", "fp16", batch=4, seqlen_q=512, seqlen_k=1024, heads_q=16, heads_kv=2, head_dim=128, seqlens_q=[512, 384, 512, 256], seqlens_k=[1024, 768, 1024, 512]),
        mk("bwd_gqa_varlen", "varlen_bwd", "bf16", batch=3, seqlen_q=1024, seqlen_k=1024, heads_q=8, heads_kv=1, head_dim=64, causal=True, seqlens_q=[1024, 768, 512], seqlens_k=[1024, 1024, 768]),
        mk("bwd_gqa_varlen", "varlen_bwd", "fp16", batch=4, seqlen_q=640, seqlen_k=640, heads_q=16, heads_kv=4, head_dim=96, seqlens_q=[640, 512, 640, 384], seqlens_k=[640, 640, 512, 640]),
    ]
    s["bwd_causal_softcap"] = [
        mk("bwd_causal_softcap", "fixed_bwd", "bf16", batch=2, seqlen_q=768, seqlen_k=768, heads_q=8, head_dim=64, causal=True, softcap=50.0),
        mk("bwd_causal_softcap", "fixed_bwd", "fp16", batch=1, seqlen_q=1024, seqlen_k=1024, heads_q=8, head_dim=64, causal=True, softcap=30.0),
        mk("bwd_causal_softcap", "fixed_bwd", "bf16", batch=2, seqlen_q=512, seqlen_k=512, heads_q=8, head_dim=128, causal=True, softcap=50.0),
        mk("bwd_causal_softcap", "fixed_bwd", "fp16", batch=2, seqlen_q=640, seqlen_k=640, heads_q=8, head_dim=192, causal=True, softcap=20.0),
    ]
    s["combine_standalone"] = [
        mk("combine_standalone", "combine", "bf16", batch=4, seqlen_q=1024, seqlen_k=1024, heads_q=16, head_dim=128, num_splits=16),
        mk("combine_standalone", "combine", "fp16", batch=8, seqlen_q=512, seqlen_k=512, heads_q=16, head_dim=64, num_splits=32),
        mk("combine_standalone", "combine", "bf16", batch=4, seqlen_q=2048, seqlen_k=2048, heads_q=8, head_dim=128, num_splits=8),
        mk("combine_standalone", "combine", "fp16", batch=2, seqlen_q=1024, seqlen_k=1024, heads_q=32, head_dim=128, num_splits=64),
    ]
    s["hd256_fwd"] = [
        mk("hd256_fwd", "fixed_fwd", "bf16", batch=2, seqlen_q=768, seqlen_k=768, heads_q=8, head_dim=256, causal=True),
        mk("hd256_fwd", "fixed_fwd", "fp16", batch=2, seqlen_q=512, seqlen_k=512, heads_q=8, head_dim=256, causal=True),
        mk("hd256_fwd", "fixed_fwd", "bf16", batch=1, seqlen_q=1024, seqlen_k=1024, heads_q=8, head_dim=256, causal=True),
        mk("hd256_fwd", "fixed_fwd", "fp16", batch=2, seqlen_q=640, seqlen_k=640, heads_q=8, head_dim=256, causal=True),
    ]
    s["reuse_scheduler_metadata"] = [
        mk("reuse_scheduler_metadata", "kvcache", "bf16", batch=8, seqlen_q=16, seqlen_k=2048, heads_q=16, heads_kv=4, head_dim=128, causal=True, reuse_scheduler_metadata=True),
        mk("reuse_scheduler_metadata", "kvcache", "fp16", batch=16, seqlen_q=8, seqlen_k=4096, heads_q=32, heads_kv=8, head_dim=64, causal=True, reuse_scheduler_metadata=True),
        mk("reuse_scheduler_metadata", "kvcache", "bf16", batch=4, seqlen_q=32, seqlen_k=2048, heads_q=16, head_dim=128, causal=True, reuse_scheduler_metadata=True),
        mk("reuse_scheduler_metadata", "kvcache", "fp16", batch=8, seqlen_q=16, seqlen_k=3072, heads_q=16, head_dim=96, causal=True, reuse_scheduler_metadata=True),
    ]
    return s


PRIMARY_NAMES: tuple[str, ...] = tuple(EVIDENCE.keys())


def build_candidates() -> list[dict[str, Any]]:
    seeds = _seeds()
    missing = [name for name in PRIMARY_NAMES if name not in seeds]
    extra = [name for name in seeds if name not in PRIMARY_NAMES]
    if missing or extra:
        raise RuntimeError(f"mechanism table mismatch missing={missing} extra={extra}")
    cases: list[dict[str, Any]] = []
    index = 0
    for primary in PRIMARY_NAMES:
        group = seeds[primary]
        if not 3 <= len(group) <= 5:
            raise RuntimeError(f"{primary} has {len(group)} seeds, want 3-5")
        for seed_i, case in enumerate(group, start=1):
            case = dict(case)
            case["case_id"] = f"fa3_{primary}_{seed_i:02d}"
            case["delivery_index"] = index
            case["delivery_group"] = index % OUTER_SHARDS
            case["seed_index"] = seed_i
            traits = derive_traits(case)
            case["trigger"] = traits
            check = PRIMARY_OK[primary]
            if not check(case, traits):
                raise RuntimeError(f"{case['case_id']} failed trigger for {primary}: {traits}")
            cases.append(case)
            index += 1
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case_id")
    return cases


def candidate_manifest(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cases = cases if cases is not None else build_candidates()
    counts = Counter(c["primary_mechanism"] for c in cases)
    return {
        "schema_version": "flashattention3-full-gpu-candidates-v1",
        "name": "FlashAttention-3 Hopper full-GPU mechanism candidate pool",
        "implementation": "FA3",
        "architecture": "sm_90a",
        "source_revision": SOURCE_REVISION,
        "source_url": SOURCE_URL,
        "candidate_count": len(cases),
        "primary_mechanism_count": len(counts),
        "seeds_per_primary": dict(counts),
        "h100_cycle_filter": {
            "minimum_cycles": CYCLES_MIN,
            "maximum_cycles": CYCLES_MAX,
            "status": "unmeasured_candidates",
            "measurement": "GPU kernel cycles collected during qualification; not a simulator parameter",
        },
        "tuning_policy": "axis in {batch,seqlen_k,num_splits,repeat_seqlens_batch}; values is an ordered legal list. Default is batch doubling. Varlen/sparse uses repeat_seqlens_batch to keep long/short ratio. Apply with apply_tuning(); never loop the same launch.",
        "qualification_status": "candidate",
        "cases": cases,
    }


def assert_candidate_invariants(cases: list[dict[str, Any]]) -> None:
    if len(cases) < 100:
        raise AssertionError(f"need >=100 candidates, got {len(cases)}")
    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise AssertionError("case_id not unique")
    counts = Counter(c["primary_mechanism"] for c in cases)
    if set(counts) != set(PRIMARY_NAMES) or len(counts) != len(PRIMARY_NAMES):
        raise AssertionError(
            f"need all {len(PRIMARY_NAMES)} primaries, got {len(counts)} "
            f"missing={sorted(set(PRIMARY_NAMES) - set(counts))}"
        )
    for name, n in counts.items():
        if not GROUP_FLOOR <= n <= 5:
            raise AssertionError(f"{name} has {n} seeds")
    required = ("case_id", "operation", "dtype", "primary_mechanism", "mechanisms", "delivery_group", "head_dim", "heads_q")
    for case in cases:
        for key in required:
            if key not in case:
                raise AssertionError(f"{case.get('case_id')} missing {key}")
        if case["operation"] not in OPERATIONS:
            raise AssertionError(f"bad operation {case['operation']}")
        if case["dtype"] not in DTYPES:
            raise AssertionError(f"bad dtype {case['dtype']}")
        if case.get("v_colmajor"):
            raise AssertionError("v_colmajor is not a supported case flag")
        if case.get("dropout") or case.get("learnable_sink"):
            raise AssertionError("dropout/sink not allowed")
        if case["operation"] in ("fixed_bwd", "varlen_bwd"):
            if case["dtype"] == "fp8_e4m3":
                raise AssertionError("fp8 backward not supported")
            if case.get("qv") or int(case.get("attention_chunk") or 0) >= 1:
                raise AssertionError("bwd forbids qv/chunk")
            if int(case.get("head_dim_v") or case["head_dim"]) > 256:
                raise AssertionError("bwd forbids dv>256")
            if case.get("deterministic") and int(case["head_dim"]) >= 256:
                raise AssertionError("deterministic+hd>=256")
        if case["operation"] == "combine" and case.get("seqlens_q"):
            raise AssertionError("standalone combine is dense only")
        traits = derive_traits(case)
        if not PRIMARY_OK[case["primary_mechanism"]](case, traits):
            raise AssertionError(f"{case['case_id']} trigger mismatch")
        tuning = case.get("tuning") or {}
        axis = tuning.get("axis")
        values = [int(v) for v in (tuning.get("values") or [])]
        if axis not in TUNING_AXES:
            raise AssertionError(f"{case['case_id']} bad tuning axis {axis}")
        if "bounds" in tuning or "step" in tuning:
            raise AssertionError(f"{case['case_id']} use tuning.values, not bounds/step")
        if values != sorted(values) or len(set(values)) != len(values) or len(values) < 2:
            raise AssertionError(f"{case['case_id']} tuning.values must be unique ordered: {values}")
        if axis == "batch" and values[0] != int(case["batch"]):
            raise AssertionError(f"{case['case_id']} tuning.values[0] must be current batch")
        if axis == "seqlen_k" and values[0] != int(case["seqlen_k"]):
            raise AssertionError(f"{case['case_id']} tuning.values[0] must be current seqlen_k")
        if axis == "num_splits" and values[0] != int(case["num_splits"]):
            raise AssertionError(f"{case['case_id']} tuning.values[0] must be current num_splits")
        if axis == "repeat_seqlens_batch" and values[0] != 1:
            raise AssertionError(f"{case['case_id']} repeat_seqlens_batch values[0] must be 1")
        if axis == "seqlen_k" and case.get("page_size") and int(case["seqlen_k"]) % int(case["page_size"]) != 0:
            raise AssertionError(f"{case['case_id']} seqlen_k not page-aligned")
        if bool(case.get("pack_gqa")) and "pack_gqa" not in case["mechanisms"]:
            raise AssertionError(f"{case['case_id']} pack_gqa missing from mechanisms")
        if int(case["heads_q"]) != int(case["heads_kv"]) and "gqa" not in case["mechanisms"]:
            raise AssertionError(f"{case['case_id']} gqa missing from mechanisms")
        if case.get("append_kv"):
            used = max(int(x) for x in case["seqlens_k"]) if case.get("seqlens_k") else 0
            if int(case["seqlen_k"]) < used + int(case.get("seqlen_new") or 0):
                raise AssertionError(f"{case['case_id']} append overflow capacity={case['seqlen_k']} used={used} new={case.get('seqlen_new')}")
        if case["primary_mechanism"] == "bwd_causal_softcap" and int(case["head_dim"]) == 64:
            if "bwd_softcap_hdim64_96x128" not in case["mechanisms"]:
                raise AssertionError(f"{case['case_id']} missing hd64 96x128 secondary")
        if case["primary_mechanism"] == "bwd_causal_softcap" and int(case["head_dim"]) == 128:
            if "bwd_softcap_hdim64_96x128" in case["mechanisms"]:
                raise AssertionError(f"{case['case_id']} D128 must not claim D64 96x128 branch")
            if "bwd_hdim128" not in case["mechanisms"]:
                raise AssertionError(f"{case['case_id']} missing bwd_hdim128 secondary")
        if case["operation"] in ("fixed_bwd", "varlen_bwd") and int(case["head_dim"]) == 192:
            if "bwd_hdim192" not in case["mechanisms"]:
                raise AssertionError(f"{case['case_id']} missing bwd_hdim192 secondary")
        for value in values:
            tuned = apply_tuning(case, value)
            if tuned["case_id"] != case["case_id"] or tuned["primary_mechanism"] != case["primary_mechanism"]:
                raise AssertionError(f"{case['case_id']} tuning mutated identity")
            if case.get("append_kv"):
                used_t = max(int(x) for x in tuned["seqlens_k"]) if tuned.get("seqlens_k") else 0
                if int(tuned["seqlen_k"]) < used_t + int(tuned.get("seqlen_new") or 0):
                    raise AssertionError(f"{case['case_id']} tuning collapsed cache capacity")


def _attempt_rank(path: Path) -> tuple[int, str]:
    rank = -1
    for part in path.parts:
        if part.startswith("attempt"):
            suffix = part[len("attempt") :]
            try:
                rank = int(suffix)
            except ValueError:
                pass
    return rank, str(path)


def _round_filenames(round_index: int) -> set[str]:
    return {f"round{round_index}.json", f"round_{round_index}.json"}


def _parse_result_items(data: Any) -> list[dict]:
    if isinstance(data, dict) and "case_id" in data:
        return [data]
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [x for x in data["cases"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _ingest_result_file(store: dict[str, tuple[tuple[int, str], dict]], file: Path) -> None:
    if file.name in SKIP_ROUND_FILES:
        return
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    rank = _attempt_rank(file)
    for item in _parse_result_items(data):
        timing = item.get("timing") if isinstance(item.get("timing"), dict) else {}
        if timing.get("probe") is True or item.get("probe") is True:
            continue
        case_id = item.get("case_id")
        if not case_id:
            continue
        key = str(case_id)
        prev = store.get(key)
        if prev is None or rank >= prev[0]:
            store[key] = (rank, item)


def _finalize_store(store: dict[str, tuple[tuple[int, str], dict]]) -> dict[str, dict]:
    return {key: value[1] for key, value in store.items()}


def _named_round_presence(path: Path) -> tuple[bool, bool]:
    names = {path.name} if path.is_file() else {p.name for p in path.rglob("*.json") if p.is_file()}
    return bool(names & _round_filenames(1)), bool(names & _round_filenames(2))


def _is_mixed_campaign_tree(path: Path) -> bool:
    has1, has2 = _named_round_presence(path)
    return has1 and has2


def _load_round_dir(path: Path) -> dict[str, dict]:
    """Load one measured round directory. Probe/summary/manifest/tuning_report are ignored.

    A campaign tree that contains both round1.json and round2.json is not a
    single-round directory. Pair those with load_paired_rounds / --result-root.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if _is_mixed_campaign_tree(path):
        raise FileNotFoundError(
            f"{path} contains both round1.json and round2.json; "
            "pair by filename with load_paired_rounds(root=...) / --result-root"
        )
    store: dict[str, tuple[tuple[int, str], dict]] = {}
    files = [path] if path.is_file() else [p for p in path.rglob("*.json") if p.is_file()]
    for file in files:
        _ingest_result_file(store, file)
    return _finalize_store(store)


def _load_named_round_files(root: Path, round_index: int) -> dict[str, dict]:
    allowed = _round_filenames(round_index)
    store: dict[str, tuple[tuple[int, str], dict]] = {}
    for file in root.rglob("*.json"):
        if file.name in allowed:
            _ingest_result_file(store, file)
    return _finalize_store(store)


def _pair_named_or_subdirs(root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    named1 = _load_named_round_files(root, 1)
    named2 = _load_named_round_files(root, 2)
    if named1 and named2:
        return named1, named2
    sub1, sub2, sub0 = root / "round1", root / "round2", root / "round0"
    if sub1.is_dir() and sub2.is_dir():
        return _load_round_dir(sub1), _load_round_dir(sub2)
    if sub0.is_dir() and sub1.is_dir():
        return _load_round_dir(sub0), _load_round_dir(sub1)
    raise FileNotFoundError(
        f"no paired rounds under {root}: need round1.json+round2.json or round0/+round1/ dirs"
    )


def load_paired_rounds(
    root: Path | None = None,
    dirs: list[Path] | None = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Pair formal round1/round2. Never mix probe/summary into either map.

    --result-root (filename pairing) is the campaign contract. A mixed tree
    passed via --result-dirs is also paired by filename, never last-write-wins.
    """
    if root is not None:
        return _pair_named_or_subdirs(Path(root))
    dirs = [Path(p) for p in (dirs or [])]
    if len(dirs) == 1:
        return _pair_named_or_subdirs(dirs[0])
    if len(dirs) >= 2:
        d0, d1 = dirs[0], dirs[1]
        same = d0.resolve() == d1.resolve()
        mixed0 = _is_mixed_campaign_tree(d0)
        mixed1 = _is_mixed_campaign_tree(d1)
        if mixed0 or mixed1:
            if mixed0 and mixed1 and same:
                return _pair_named_or_subdirs(d0)
            raise FileNotFoundError(
                f"--result-dirs mixed campaign trees must be the same path; "
                f"got {d0} and {d1}. Use --result-root for one campaign tree."
            )
        return _load_round_dir(d0), _load_round_dir(d1)
    raise FileNotFoundError("select needs --result-root or two --result-dirs")


def _correct(row: dict) -> bool:
    status = str(row.get("status") or "").lower()
    if status not in {"pass", "ok", "qualified", "success"}:
        return False
    correctness = row.get("correctness")
    if correctness is True:
        return True
    if isinstance(correctness, dict):
        inner = str(correctness.get("status") or "").lower()
        return inner in {"pass", "ok", "qualified", "success"}
    return False


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nested_case(row: dict) -> dict[str, Any] | None:
    case = row.get("case") or row.get("final_case") or row.get("measured_case")
    return case if isinstance(case, dict) else None


def _shape_sig(case: dict[str, Any]) -> tuple:
    out: list[tuple[str, Any]] = []
    for key in SHAPE_KEYS:
        value = case.get(key)
        if isinstance(value, list):
            out.append((key, tuple(value)))
        else:
            out.append((key, value))
    return tuple(out)


def _run_identity(row: dict) -> tuple[str, str, str] | None:
    gpu = row.get("gpu")
    uuid = gpu.get("uuid") if isinstance(gpu, dict) else None
    build = row.get("build_root")
    rev = row.get("source_revision")
    if not uuid or not build or not rev:
        return None
    return str(uuid), str(build), str(rev)


def _gpu_name(row: dict) -> str | None:
    gpu = row.get("gpu")
    if isinstance(gpu, dict):
        name = gpu.get("name")
        return str(name) if name else None
    if isinstance(gpu, str) and gpu:
        return gpu
    return None


def _is_h100(row: dict) -> bool:
    name = _gpu_name(row)
    return isinstance(name, str) and "NVIDIA H100" in name


def _explicit_seed(row: dict, case: dict[str, Any] | None) -> int | None:
    if "input_seed" not in row or not isinstance(case, dict) or "input_seed" not in case:
        return None
    try:
        top = int(row["input_seed"])
        inner = int(case["input_seed"])
    except (TypeError, ValueError):
        return None
    if top != inner:
        return None
    return top


def _int_list(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    try:
        return tuple(int(x) for x in value)
    except (TypeError, ValueError):
        return None


def _realized_leftpad(row: dict, case: dict[str, Any] | None) -> tuple[int, ...] | None:
    if isinstance(row.get("realized_leftpad"), list):
        return _int_list(row["realized_leftpad"])
    if isinstance(case, dict) and isinstance(case.get("realized_leftpad"), list):
        return _int_list(case["realized_leftpad"])
    correctness = row.get("correctness")
    if isinstance(correctness, dict) and isinstance(correctness.get("realized_leftpad"), list):
        return _int_list(correctness["realized_leftpad"])
    return None


def _round_timing_view(timing: dict[str, Any]) -> dict[str, Any]:
    return {
        "median_cycles": timing["median_cycles"],
        "cv_percent": timing["cv_percent"],
        "sample_count": timing["sample_count"],
        "raw_sample_count": timing["raw_sample_count"],
        "samples_cycles": list(timing["samples_cycles"]),
    }


def _timing(row: dict) -> dict[str, Any] | None:
    timing = row.get("timing")
    if not isinstance(timing, dict):
        return None
    if timing.get("probe") is True or row.get("probe") is True:
        return None
    try:
        cycles = float(timing["median_cycles"])
        cv = float(timing["cv_percent"])
        recorded = int(timing["sample_count"])
        raw = int(timing["raw_sample_count"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(cycles) or not math.isfinite(cv):
        return None
    if recorded != RECORDED_SAMPLES or raw != RAW_SAMPLES:
        return None
    samples = timing.get("samples_cycles")
    if not isinstance(samples, list) or len(samples) != RECORDED_SAMPLES:
        return None
    if not all(_finite_number(x) for x in samples):
        return None
    return {
        "median_cycles": cycles,
        "cv_percent": cv,
        "sample_count": recorded,
        "raw_sample_count": raw,
        "samples_cycles": [float(x) for x in samples],
        "raw": timing,
    }


def relative_drift(a: float, b: float) -> float:
    denom = (abs(a) + abs(b)) / 2.0
    if denom == 0:
        return 0.0
    return abs(a - b) / denom * 100.0


def _activities(row: dict) -> list:
    act = row.get("kernel_activities")
    if isinstance(act, list):
        return list(act)
    if isinstance(act, dict):
        return list(act.keys())
    if act:
        return [act]
    return []


def qualified_from_rounds(round_a: dict[str, dict], round_b: dict[str, dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for case_id in set(round_a) & set(round_b):
        a, b = round_a[case_id], round_b[case_id]
        if not (_correct(a) and _correct(b)):
            continue
        if not (_is_h100(a) and _is_h100(b)):
            continue
        ta, tb = _timing(a), _timing(b)
        if ta is None or tb is None:
            continue
        if not (CYCLES_MIN <= ta["median_cycles"] <= CYCLES_MAX):
            continue
        if not (CYCLES_MIN <= tb["median_cycles"] <= CYCLES_MAX):
            continue
        if ta["cv_percent"] >= CV_LIMIT or tb["cv_percent"] >= CV_LIMIT:
            continue
        ident_a, ident_b = _run_identity(a), _run_identity(b)
        if ident_a is None or ident_a != ident_b:
            continue
        measured_a, measured_b = _nested_case(a), _nested_case(b)
        if measured_a is None or measured_b is None:
            continue
        if _shape_sig(measured_a) != _shape_sig(measured_b):
            continue
        seed_a = _explicit_seed(a, measured_a)
        seed_b = _explicit_seed(b, measured_b)
        if seed_a != FORMAL_INPUT_SEED or seed_b != FORMAL_INPUT_SEED:
            continue
        leftpad = bool(measured_a.get("cache_leftpad")) or bool(measured_b.get("cache_leftpad"))
        if leftpad:
            pad_a = _realized_leftpad(a, measured_a)
            pad_b = _realized_leftpad(b, measured_b)
            if pad_a is None or pad_a != pad_b:
                continue
        drift = relative_drift(ta["median_cycles"], tb["median_cycles"])
        if drift >= DRIFT_LIMIT:
            continue
        median = statistics.median([ta["median_cycles"], tb["median_cycles"]])
        activities = _activities(a) + _activities(b)
        view_a = _round_timing_view(ta)
        view_b = _round_timing_view(tb)
        out[case_id] = {
            "median_cycles": median,
            "cv_percent": max(ta["cv_percent"], tb["cv_percent"]),
            "drift_percent": drift,
            "aggregation": dict(TIMING_AGGREGATION),
            "round_timings": {"round1": view_a, "round2": view_b},
            "kernel_activities": activities,
            "measured_case": measured_a,
            "round0": a,
            "round1": b,
            "gpu": a.get("gpu") or b.get("gpu"),
            "build_root": ident_a[1],
            "source_revision": ident_a[2],
            "input_seed": FORMAL_INPUT_SEED,
        }
    return out


def _rec_measured(rec: dict[str, Any]) -> dict[str, Any] | None:
    measured = rec.get("measured_case")
    if isinstance(measured, dict):
        return measured
    a = _nested_case(rec.get("round0") or {})
    b = _nested_case(rec.get("round1") or {})
    if isinstance(a, dict) and isinstance(b, dict) and _shape_sig(a) == _shape_sig(b):
        return a
    return None


def overlay_measured_case(base: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    measured = _rec_measured(rec)
    if not isinstance(measured, dict):
        raise RuntimeError(f"{base['case_id']} missing equal nested measured cases from both rounds")
    row = dict(base)
    skip = {
        "case_id",
        "primary_mechanism",
        "source_evidence",
        "qualification_status",
        "delivery_index",
        "delivery_group",
        "seed_index",
        "suite",
        "implementation",
        "source",
        "api",
        "source_revision",
        "mechanisms",
    }
    for key, value in measured.items():
        if key in skip:
            continue
        if isinstance(value, list):
            row[key] = list(value)
        elif isinstance(value, dict) and key != "tuning":
            row[key] = dict(value)
        else:
            row[key] = value
    row["case_id"] = base["case_id"]
    row["primary_mechanism"] = base["primary_mechanism"]
    row["source_evidence"] = dict(base["source_evidence"])
    traits = derive_traits(row)
    if not PRIMARY_OK[row["primary_mechanism"]](row, traits):
        raise RuntimeError(f"{row['case_id']} measured shape broke trigger {traits}")
    row["trigger"] = traits
    row["mechanisms"] = collect_mechanisms(row, traits)
    return row


def _group_unique_secondary_ids(group: list[dict[str, Any]]) -> set[str]:
    owners: dict[str, set[str]] = defaultdict(set)
    for case in group:
        primary = case["primary_mechanism"]
        for tag in case.get("mechanisms") or []:
            if tag != primary:
                owners[str(tag)].add(case["case_id"])
    return {next(iter(ids)) for ids in owners.values() if len(ids) == 1}


def _round_views(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    stored = rec.get("round_timings")
    if isinstance(stored, dict) and "round1" in stored and "round2" in stored:
        r1, r2 = stored["round1"], stored["round2"]
        if isinstance(r1, dict) and isinstance(r2, dict):
            return dict(r1), dict(r2)
    t0 = _timing(rec.get("round0") or {})
    t1 = _timing(rec.get("round1") or {})
    if t0 is None or t1 is None:
        return None
    return _round_timing_view(t0), _round_timing_view(t1)


def select_official(
    candidates: list[dict[str, Any]],
    qualified: dict[str, dict[str, Any]],
    target: int = TARGET_SELECTED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_groups = len(PRIMARY_NAMES)
    extra_slots = target - GROUP_FLOOR * n_groups
    if extra_slots < 0:
        raise RuntimeError(f"target {target} < {GROUP_FLOOR}×{n_groups} floor")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    filled_recs: dict[str, dict[str, Any]] = {}
    for case in candidates:
        rec = qualified.get(case["case_id"])
        if rec is None:
            continue
        measured = _rec_measured(rec)
        if measured is None:
            continue
        rec = dict(rec)
        rec["measured_case"] = measured
        views = _round_views(rec)
        if views is None:
            continue
        r1, r2 = views
        rec["round_timings"] = {"round1": r1, "round2": r2}
        rec.setdefault("aggregation", dict(TIMING_AGGREGATION))
        rec.setdefault("cv_percent", max(float(r1["cv_percent"]), float(r2["cv_percent"])))
        rec.setdefault("median_cycles", statistics.median([float(r1["median_cycles"]), float(r2["median_cycles"])]))
        rec.setdefault("drift_percent", relative_drift(float(r1["median_cycles"]), float(r2["median_cycles"])))
        rec.setdefault("kernel_activities", [])
        rec.setdefault("gpu", {"name": "NVIDIA H100"})
        rec.setdefault("source_revision", SOURCE_REVISION)
        rec.setdefault("input_seed", FORMAL_INPUT_SEED)
        row = overlay_measured_case(case, rec)
        filled_recs[case["case_id"]] = rec
        by_group[case["primary_mechanism"]].append(row)
    present = {name: by_group.get(name, []) for name in PRIMARY_NAMES}
    missing = [name for name, group in present.items() if not group]
    under = [name for name, group in present.items() if 0 < len(group) < GROUP_FLOOR]
    total_q = sum(len(v) for v in present.values())
    if missing or under:
        raise RuntimeError(
            f"refuse official {target}: missing_groups={missing} under_floor={under} "
            f"qualified={total_q}. Cannot fill {target} from {n_groups - len(missing)} groups."
        )
    if total_q < target:
        raise RuntimeError(
            f"qualified {total_q} < {target}; groups={ {n: len(present[n]) for n in PRIMARY_NAMES} }"
        )
    quota: dict[str, int] = {name: GROUP_FLOOR for name in PRIMARY_NAMES}
    cap = {name: min(len(present[name]), GROUP_CEILING) for name in PRIMARY_NAMES}
    extras = extra_slots
    must = [
        name
        for name in PRIMARY_NAMES
        if len(_group_unique_secondary_ids(present[name])) > quota[name] and cap[name] > quota[name]
    ]
    if len(must) > extras:
        raise RuntimeError(
            f"unique-within-primary secondaries need extra on {must}, only {extras} extra slots"
        )
    for name in must:
        quota[name] += 1
        extras -= 1
    spare = sorted((name for name in PRIMARY_NAMES if quota[name] < cap[name]), reverse=True)
    for name in spare:
        if extras <= 0:
            break
        quota[name] += 1
        extras -= 1
    assigned = sum(quota.values())
    if assigned != target or extras != 0:
        raise RuntimeError(f"allocation produced {assigned} leftover_extras={extras}, want {target}")
    if any(quota[n] < GROUP_FLOOR or quota[n] > GROUP_CEILING for n in PRIMARY_NAMES):
        raise RuntimeError(f"quota not {GROUP_FLOOR} or {GROUP_CEILING}: {quota}")
    mid = math.sqrt(CYCLES_MIN * CYCLES_MAX)
    locked_all: set[str] = set()

    def score(case: dict[str, Any]) -> tuple[float, float, str]:
        rec = filled_recs[case["case_id"]]
        return (
            abs(math.log(rec["median_cycles"]) - math.log(mid)),
            rec["cv_percent"],
            case["case_id"],
        )

    selected: list[dict[str, Any]] = []
    for name in PRIMARY_NAMES:
        cases = present[name]
        take = quota[name]
        locked_ids = _group_unique_secondary_ids(cases)
        if len(locked_ids) > take:
            raise RuntimeError(
                f"{name} unique-within-primary secondaries {len(locked_ids)} exceed quota {take}"
            )
        locked_all.update(locked_ids)
        locked_cases = [c for c in cases if c["case_id"] in locked_ids]
        rest = sorted((c for c in cases if c["case_id"] not in locked_ids), key=score)
        chosen = locked_cases + rest[: take - len(locked_cases)]
        if len(chosen) != take:
            raise RuntimeError(f"{name} selected {len(chosen)} != quota {take}")
        for case in chosen:
            rec = filled_recs[case["case_id"]]
            views = rec["round_timings"]
            row = dict(case)
            row["qualification_status"] = "qualified_h100_two_rounds"
            row["input_seed"] = FORMAL_INPUT_SEED
            row["h100_timing"] = {
                "median_cycles": rec["median_cycles"],
                "cv_percent": rec["cv_percent"],
                "drift_percent": rec["drift_percent"],
                "aggregation": dict(rec.get("aggregation") or TIMING_AGGREGATION),
                "observation_scope": "whole_operation",
                "rounds": {
                    "round1": dict(views["round1"]),
                    "round2": dict(views["round2"]),
                },
            }
            row["kernel_activities"] = rec["kernel_activities"]
            row["gpu"] = rec["gpu"]
            row["build_root"] = rec.get("build_root")
            row["measured_source_revision"] = rec["source_revision"]
            selected.append(row)
    selected.sort(key=lambda c: c["case_id"])
    for i, case in enumerate(selected):
        case["delivery_index"] = i
        case["delivery_group"] = i % OUTER_SHARDS
    if len(selected) != target:
        raise RuntimeError(f"selected {len(selected)} != {target}")
    counts = Counter(c["primary_mechanism"] for c in selected)
    if set(counts) != set(PRIMARY_NAMES) or any(
        n not in (GROUP_FLOOR, GROUP_CEILING) for n in counts.values()
    ):
        raise RuntimeError(f"selected groups {dict(counts)} not {n_groups}×{{3,4}}")
    meta = {
        "selected_count": len(selected),
        "qualified_available": total_q,
        "per_group_quota": quota,
        "per_group_selected": dict(counts),
        "short_groups": [],
        "groups_without_any_qualified": missing,
        "max_min": max(counts.values()) - min(counts.values()),
        "unique_secondary_locked": sorted(locked_all),
    }
    return selected, meta

def official_manifest(selected: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    header = cases_header(
        "flashattention3",
        name="FlashAttention-3 Hopper full-GPU 100-case qualified delivery set",
        selected_count=len(selected),
    )
    return {
        **header,
        "implementation": "FA3",
        "source_revision": SOURCE_REVISION,
        "source_url": SOURCE_URL,
        "h100_cycle_filter": {
            "minimum_cycles": CYCLES_MIN,
            "maximum_cycles": CYCLES_MAX,
            "status": "qualified_two_rounds",
            "maximum_cv_percent_exclusive": CV_LIMIT,
            "maximum_round_drift_percent_exclusive": DRIFT_LIMIT,
            "required_recorded_samples": RECORDED_SAMPLES,
            "required_raw_samples": RAW_SAMPLES,
            "measurement": "Queued device globaltimer whole-operation span converted by observed clock64/globaltimer ratio; not a simulator parameter",
        },
        "selection": meta,
        "cases": selected,
    }


def _formal_row(case: dict[str, Any], cycles: float, **extra: Any) -> dict[str, Any]:
    n = int(extra.get("n", RECORDED_SAMPLES))
    raw_n = int(extra.get("raw_n", RAW_SAMPLES))
    samples = extra.get("samples")
    if samples is None:
        samples = [cycles] * n
    timing = {
        "median_cycles": cycles,
        "cv_percent": extra.get("cv", 0.4),
        "samples_cycles": samples,
        "observation_scope": "whole_operation",
        "probe": extra.get("probe", False),
        "sample_count": n,
        "raw_sample_count": raw_n,
    }
    measured = extra.get("measured", case)
    if isinstance(measured, dict):
        measured = dict(measured)
        for key in ("seqlens_q", "seqlens_k", "window_size", "mechanisms"):
            if isinstance(measured.get(key), list):
                measured[key] = list(measured[key])
    row = {
        "case_id": case["case_id"],
        "status": extra.get("status", "pass"),
        "correctness": {"status": extra.get("correct_status", "pass")},
        "timing": timing,
        "kernel_activities": extra.get("kernel_activities", [{"name": case["api"]}]),
        "gpu": extra.get("gpu", {"name": "NVIDIA H100", "uuid": "GPU-A"}),
        "source_revision": extra.get("rev", SOURCE_REVISION),
        "build_root": extra.get("build", "/build-A"),
        "round": extra.get("round", 1),
        "case": measured,
    }
    seed = extra["input_seed"] if "input_seed" in extra else FORMAL_INPUT_SEED
    if seed is not None:
        row["input_seed"] = int(seed)
        if isinstance(measured, dict):
            measured["input_seed"] = int(seed)
    if "realized_leftpad" in extra:
        pads = extra["realized_leftpad"]
        if pads is not None:
            row["realized_leftpad"] = list(pads)
    elif isinstance(measured, dict) and bool(measured.get("cache_leftpad")):
        row["realized_leftpad"] = [0] * int(measured.get("batch") or 1)
    return row


def _review_regression() -> dict[str, Any]:
    """CPU checks that previously failed the coverage review. Must pass."""
    import tempfile

    cases = build_candidates()
    assert_candidate_invariants(cases)
    base = next(c for c in cases if c["case_id"] == "fa3_varlen_fwd_01")
    tuned = apply_tuning(base, 4)

    qa: dict[str, dict] = {}
    qb: dict[str, dict] = {}
    for i, case in enumerate(cases):
        measured = apply_tuning(case, case["tuning"]["values"][-1]) if case["tuning"]["axis"] == "repeat_seqlens_batch" else case
        qa[case["case_id"]] = _formal_row(case, 200000 + i, measured=measured, round=1)
        qb[case["case_id"]] = _formal_row(case, 200200 + i, measured=measured, round=2)
    selected, meta = select_official(cases, qualified_from_rounds(qa, qb))
    got = next(c for c in selected if c["case_id"] == base["case_id"])
    if got["batch"] == base["batch"] and got.get("seqlens_q") == base.get("seqlens_q"):
        raise AssertionError("official selected pasted base shape instead of nested measured case")
    if got["batch"] != tuned["batch"]:
        # values[-1] is 8; apply_tuning(base,4) is an in-band check that overlay happened
        got_last = apply_tuning(base, base["tuning"]["values"][-1])
        if got["batch"] != got_last["batch"]:
            raise AssertionError(f"selected batch {got['batch']} != measured {got_last['batch']}")

    with tempfile.TemporaryDirectory() as tmp:
        attempt = Path(tmp) / "fa3_varlen_fwd_01" / "attempt0"
        attempt.mkdir(parents=True)
        (attempt / "manifest.json").write_text(json.dumps({"cases": [tuned]}), encoding="utf-8")
        (attempt / "probe.json").write_text(
            json.dumps(_formal_row(base, 400000, probe=True, n=1, raw_n=1, cv=0.0, samples=[400000], measured=tuned)),
            encoding="utf-8",
        )
        (attempt / "round1.json").write_text(
            json.dumps(_formal_row(base, 410000, n=9, raw_n=36, cv=0.5, measured=tuned, round=1)),
            encoding="utf-8",
        )
        (attempt / "round2.json").write_text(
            json.dumps(_formal_row(base, 412000, n=9, raw_n=36, cv=0.6, measured=tuned, round=2)),
            encoding="utf-8",
        )
        (Path(tmp) / "fa3_varlen_fwd_01" / "tuning_report.json").write_text(
            json.dumps({"case_id": base["case_id"], "status": "pass", "case": tuned}),
            encoding="utf-8",
        )
        try:
            mixed = _load_round_dir(Path(tmp)).get(base["case_id"], {})
        except FileNotFoundError:
            mixed = {}
        if mixed.get("timing", {}).get("probe") is True:
            raise AssertionError("probe leaked into round loader")
        if mixed.get("timing", {}).get("sample_count") == 1:
            raise AssertionError("probe 1-sample won last-write")
        paired = load_paired_rounds(root=Path(tmp))
        if base["case_id"] not in paired[0] or base["case_id"] not in paired[1]:
            raise AssertionError("round1/round2 pairing failed")
        if paired[0][base["case_id"]]["timing"]["sample_count"] != 9:
            raise AssertionError("paired round1 is not the formal 9-sample file")
        collapsed = load_paired_rounds(dirs=[Path(tmp), Path(tmp)])
        c0 = collapsed[0][base["case_id"]]["timing"]["median_cycles"]
        c1 = collapsed[1][base["case_id"]]["timing"]["median_cycles"]
        if c0 == c1:
            raise AssertionError(f"--result-dirs same campaign tree collapsed both rounds to {c0}")
        if (c0, c1) != (410000, 412000):
            raise AssertionError(f"same-tree dirs pairing got {c0}/{c1}, want 410000/412000")
        other = Path(tmp) / "other_tree" / "fa3_varlen_fwd_01" / "attempt0"
        other.mkdir(parents=True)
        (other / "round1.json").write_text((attempt / "round1.json").read_text(encoding="utf-8"), encoding="utf-8")
        (other / "round2.json").write_text((attempt / "round2.json").read_text(encoding="utf-8"), encoding="utf-8")
        try:
            load_paired_rounds(dirs=[Path(tmp), other.parent.parent])
            raise AssertionError("two different mixed trees must not silently pair one")
        except FileNotFoundError:
            pass
        try:
            _load_round_dir(Path(tmp))
            raise AssertionError("mixed campaign tree must not last-write-wins via _load_round_dir")
        except FileNotFoundError:
            pass

    probe_a = _formal_row(cases[0], 500000, probe=True, n=1, raw_n=1, cv=0.0, samples=[500000], build="/A", gpu={"uuid": "GPU-A"})
    probe_b = _formal_row(
        cases[0],
        501000,
        probe=True,
        n=1,
        raw_n=1,
        cv=float("nan"),
        samples=[float("nan")],
        build="/B",
        gpu={"uuid": "GPU-B"},
        rev="deadbeef",
    )
    if cases[0]["case_id"] in qualified_from_rounds({cases[0]["case_id"]: probe_a}, {cases[0]["case_id"]: probe_b}):
        raise AssertionError("NaN/probe/mismatched uuid-build was qualified")

    gqa = next(c for c in cases if "gqa" in c["mechanisms"] and c["primary_mechanism"] != "gqa")
    fq: dict[str, dict[str, Any]] = {}
    for case in cases:
        cycles = float(CYCLES_MAX) if case["case_id"] == gqa["case_id"] else 400000.0
        measured = case
        row = _formal_row(case, cycles, measured=measured)
        fq[case["case_id"]] = {
            "median_cycles": cycles,
            "cv_percent": 0.4,
            "drift_percent": 0.1,
            "sample_count": RECORDED_SAMPLES,
            "raw_sample_count": RAW_SAMPLES,
            "samples_cycles": [cycles] * RECORDED_SAMPLES,
            "kernel_activities": ["k"],
            "measured_case": measured,
            "round0": row,
            "round1": dict(row),
            "gpu": {"name": "NVIDIA H100", "uuid": "GPU-A"},
            "build_root": "/build-A",
            "source_revision": SOURCE_REVISION,
        }
    selected_gqa, _ = select_official(cases, fq)
    if not any("gqa" in c["mechanisms"] for c in selected_gqa):
        raise AssertionError(f"unique/secondary gqa dropped ({gqa['case_id']})")

    missing_pack = [
        c["case_id"]
        for c in cases
        if c.get("pack_gqa") and c["primary_mechanism"] != "pack_gqa" and "pack_gqa" not in c["mechanisms"]
    ]
    if missing_pack:
        raise AssertionError(f"pack_gqa secondary missing: {missing_pack}")

    app = next(c for c in cases if c["case_id"] == "fa3_kvcache_append_rotary_01")
    if max(app["seqlens_k"]) + int(app["seqlen_new"]) > int(app["seqlen_k"]):
        raise AssertionError(f"append overflow {app['seqlen_k']} {max(app['seqlens_k'])} {app['seqlen_new']}")
    repeated = apply_tuning(app, 4)
    if max(repeated["seqlens_k"]) + int(repeated["seqlen_new"]) > int(repeated["seqlen_k"]):
        raise AssertionError("repeat_seqlens_batch collapsed append capacity")

    soft = [(c["case_id"], c["head_dim"]) for c in cases if c["primary_mechanism"] == "bwd_causal_softcap"]
    if any(cid.endswith("_03") and hd == 128 and "bwd_softcap_hdim64_96x128" in next(c for c in cases if c["case_id"] == cid)["mechanisms"] for cid, hd in soft):
        raise AssertionError("D128 seed still tagged as D64 96x128 branch")
    d192 = [c["case_id"] for c in cases if c["operation"] in ("fixed_bwd", "varlen_bwd") and c["head_dim"] == 192]
    if not d192:
        raise AssertionError("D192 bwd still missing")
    pack_base = next(c for c in cases if c.get("pack_gqa") and c["primary_mechanism"] != "pack_gqa")
    pack_measured = dict(pack_base)
    pack_measured["pack_gqa"] = False
    over = overlay_measured_case(pack_base, {"measured_case": pack_measured})
    if over.get("pack_gqa") is not False:
        raise AssertionError("overlay did not take measured pack_gqa=False")
    if "pack_gqa" in over["mechanisms"]:
        raise AssertionError("overlay merged stale base pack_gqa tag")

    h20 = _formal_row(cases[0], 400000, gpu={"name": "NVIDIA H20", "uuid": "GPU-A"})
    h20b = _formal_row(cases[0], 400200, gpu={"name": "NVIDIA H20", "uuid": "GPU-A"}, round=2)
    if cases[0]["case_id"] in qualified_from_rounds({cases[0]["case_id"]: h20}, {cases[0]["case_id"]: h20b}):
        raise AssertionError("non-H100 gpu was qualified")

    unseeded = _formal_row(cases[0], 400000, input_seed=None)
    unseeded_b = _formal_row(cases[0], 400200, input_seed=None, round=2)
    if cases[0]["case_id"] in qualified_from_rounds(
        {cases[0]["case_id"]: unseeded}, {cases[0]["case_id"]: unseeded_b}
    ):
        raise AssertionError("unseeded record was qualified")

    left = next(c for c in cases if c["primary_mechanism"] == "kvcache_leftpad")
    pad_a = _formal_row(left, 400000, realized_leftpad=[1, 2, 3, 4])
    pad_b = _formal_row(left, 400200, realized_leftpad=[8, 7, 6, 5], round=2)
    if left["case_id"] in qualified_from_rounds({left["case_id"]: pad_a}, {left["case_id"]: pad_b}):
        raise AssertionError("mismatched realized_leftpad was qualified")
    flag_off = dict(left)
    flag_off["cache_leftpad"] = False
    if _shape_sig(left) == _shape_sig(flag_off):
        raise AssertionError("SHAPE_KEYS omitted cache_leftpad")

    qa_full: dict[str, dict] = {}
    qb_full: dict[str, dict] = {}
    for i, case in enumerate(cases):
        qa_full[case["case_id"]] = _formal_row(case, 400000, measured=case, round=1)
        qb_full[case["case_id"]] = _formal_row(case, 400200, measured=case, round=2)
    full_q = qualified_from_rounds(qa_full, qb_full)
    no_odd = {cid: rec for cid, rec in full_q.items() if not cid.startswith("fa3_cluster_odd_")}
    try:
        select_official(cases, no_odd)
        raise AssertionError("select filled 100 from 31 groups")
    except RuntimeError as exc:
        if "missing_groups" not in str(exc) and "cluster_odd" not in str(exc):
            raise AssertionError(f"missing-group error unclear: {exc}") from exc
    under_odd = dict(full_q)
    odd_ids = [c["case_id"] for c in cases if c["primary_mechanism"] == "cluster_odd"]
    for cid in odd_ids[2:]:
        under_odd.pop(cid, None)
    try:
        select_official(cases, under_odd)
        raise AssertionError("select allowed a group with fewer than 3")
    except RuntimeError as exc:
        if "under_floor" not in str(exc) and "cluster_odd" not in str(exc):
            raise AssertionError(f"under-floor error unclear: {exc}") from exc

    costly = dict(full_q)
    for case in cases:
        if case["primary_mechanism"] != "bwd_causal_softcap":
            continue
        cycles = 1_499_000.0 if int(case["head_dim"]) == 128 else 400000.0
        costly[case["case_id"]] = {
            **full_q[case["case_id"]],
            "median_cycles": cycles,
            "round0": _formal_row(case, cycles, measured=case, round=1),
            "round1": _formal_row(case, cycles + 200, measured=case, round=2),
        }
    kept_soft, _ = select_official(cases, costly)
    soft_sel = [c for c in kept_soft if c["primary_mechanism"] == "bwd_causal_softcap"]
    hds = {int(c["head_dim"]) for c in soft_sel}
    if not {64, 128, 192} <= hds:
        raise AssertionError(f"causal+softcap lost a real hd branch: {sorted(hds)}")
    if not any("bwd_hdim128" in c["mechanisms"] for c in soft_sel):
        raise AssertionError("causal+softcap hd128 secondary dropped")
    if not any("bwd_hdim192" in c["mechanisms"] for c in soft_sel):
        raise AssertionError("causal+softcap hd192 secondary dropped")

    timing = next(c["h100_timing"] for c in selected)
    if "rounds" not in timing or "aggregation" not in timing:
        raise AssertionError("h100_timing missing rounds/aggregation")
    if "samples_cycles" in timing:
        raise AssertionError("aggregated h100_timing still carries one-round samples")
    r1 = timing["rounds"]["round1"]
    r2 = timing["rounds"]["round2"]
    if len(r1["samples_cycles"]) != RECORDED_SAMPLES or len(r2["samples_cycles"]) != RECORDED_SAMPLES:
        raise AssertionError("round samples not 9")
    if set(meta["per_group_selected"]) != set(PRIMARY_NAMES):
        raise AssertionError("selected primary set is not 32")
    if any(n not in (GROUP_FLOOR, GROUP_CEILING) for n in meta["per_group_selected"].values()):
        raise AssertionError(f"group sizes not 3 or 4: {meta['per_group_selected']}")

    return {
        "status": "pass",
        "selected_nested": True,
        "probe_excluded": True,
        "nan_rejected": True,
        "gqa_kept": True,
        "append_capacity": True,
        "d192_bwd": d192,
        "softcap_hd": soft,
        "h100_gated": True,
        "input_seed_gated": True,
        "leftpad_gated": True,
        "all_primaries": True,
    }


def _demo() -> None:
    cases = build_candidates()
    assert_candidate_invariants(cases)
    fake_rows_a: dict[str, dict] = {}
    fake_rows_b: dict[str, dict] = {}
    for i, case in enumerate(cases):
        cycles = 120000 + (i * 7000) % 800000
        fake_rows_a[case["case_id"]] = _formal_row(case, cycles, measured=case, round=1)
        fake_rows_b[case["case_id"]] = _formal_row(case, cycles + 200, measured=case, round=2)
    selected, meta = select_official(cases, qualified_from_rounds(fake_rows_a, fake_rows_b))
    assert len(selected) == 100
    assert set(meta["per_group_selected"]) == set(PRIMARY_NAMES)
    assert all(n in (GROUP_FLOOR, GROUP_CEILING) for n in meta["per_group_selected"].values())
    assert meta["max_min"] <= 1
    assert "rounds" in selected[0]["h100_timing"]
    assert "samples_cycles" not in selected[0]["h100_timing"]
    review = _review_regression()
    axes = Counter(c["tuning"]["axis"] for c in cases)
    print(
        json.dumps(
            {
                "candidates": len(cases),
                "primaries": len(PRIMARY_NAMES),
                "selected": 100,
                "max_min": meta["max_min"],
                "tuning_axes": dict(axes),
                "review": review["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _demo()
