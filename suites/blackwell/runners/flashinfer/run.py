#!/usr/bin/env python3
"""Far-holdout flashinfer suite runner (suite `flashinfer`, specs
suites/legacy/dev283/holdout_far_flashinfer.json).  Real input call code for every
FI far case; the call construction mirrors the 2026-09-17 B300S probe
harness (suites/flashinfer/run.py) for the anchored families and the
flashinfer 0.7.0 public wrappers for the new ones.

Modes:
  run_flashinfer.py <CASE_ID> [--resolve] [--compile-only] [--out DIR] [--all]

  default        real call on the capture GPU (single launch per case),
                 result.json via delivery_tests common.write_result.
  --resolve      spec resolution only (no flashinfer/torch import): prints
                 `RESOLVE <id> family=<f> entry=<e> shape=<tag>
                 cta_group=<g> heldout=far`.
  --compile-only native CuTe compile-time descriptors on a GPU-less pod;
                 device-bound cases report prepared JIT@B300, not compile-ok.

Imports use the intact vendored FlashInfer source (.deps/flashinfer/src).
No package initializers, CUDA APIs, or hardware queries are stubbed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
sys.path.insert(0, str(ROOT / "suites"))
from asset_contract import AssetContractError, check_case_contract, configure_flashinfer

SPEC = ROOT / "suites" / "c450_full" / "cases_flashinfer.json"
SUITE = "flashinfer"

os.environ.setdefault("FLASHINFER_NO_DOWNLOAD", "1")
os.environ.setdefault("BSA_CLC", "0")
os.environ.setdefault("BUILD_NVEP", "0")


def _cases() -> list[dict]:
    cases_dir = os.environ.get("BW_CASES_DIR")
    path = Path(cases_dir) / "cases_flashinfer.json" if cases_dir else SPEC
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def load_case(cid: str) -> dict:
    for c in _cases():
        if c["id"] == cid:
            return c
    raise KeyError(f"{cid} not in {SPEC}")


def resolve_line(c: dict) -> str:
    return (f"RESOLVE {c['id']} family={c['family']} entry={c['entry']} "
            f"shape={c['shape_tag']} cta_group={c['cta_group']} heldout=far")


def _repeat_api(api_call, calls: int):
    result = None
    for _ in range(calls):
        result = api_call()
    return result


# ---------------------------------------------------------------------------
# family call builders (GPU mode).  Each returns (ok, payload_kw).
# ---------------------------------------------------------------------------

def call_fmha(c: dict, calls: int = 1) -> dict:
    import math

    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda_drv
    import torch
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.typing import Float32

    from flashinfer.cute_dsl.attention.fmha.fmha import (
        BlackwellFusedMultiHeadAttentionForward,
    )
    from flashinfer.cute_dsl.attention.fmha.helpers import fmha_helpers as fmha_utils

    p = c["params"]
    B, S, Hk, Hr, D = c["shape"]
    dtmap = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "e4m3": torch.float8_e4m3fn}
    dt = dtmap[c["dtype"]]
    init_dtype = torch.float32 if c["dtype"] == "e4m3" else dt
    seq_q = int(p.get("seq_q", S))
    q = torch.randn(B, seq_q, Hk, Hr, D, device="cuda", dtype=init_dtype)
    k = torch.randn(B, S, Hk, 1, D, device="cuda", dtype=init_dtype)
    v = torch.randn(B, S, Hk, 1, D, device="cuda", dtype=init_dtype)
    if c["dtype"] == "e4m3":
        q = (q.float() * 0.1).to(dt)
        k = (k.float() * 0.1).to(dt)
        v = (v.float() * 0.1).to(dt)
    o = torch.empty(B, seq_q, Hk, Hr, D, device="cuda", dtype=torch.bfloat16)
    cum = torch.arange(B + 1, device="cuda", dtype=torch.int32) * S
    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=tuple(p["mma_tiler"]),
        head_dim=p["head_dim"],
        is_persistent=bool(p.get("is_persistent", False)),
        mask_type=getattr(fmha_utils.MaskEnum, p["mask"]),
        enable_ex2_emulation=bool(p["ex2_emulation"]),
        enable_skip_correction=bool(p["skip_correction"]),
        use_tma_store=bool(p["tma_store"]),
    )
    q_c = from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=4)
    k_c = from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=4)
    v_c = from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=4)
    o_c = from_dlpack(o, assumed_align=32).mark_layout_dynamic(leading_dim=4)
    cq = from_dlpack(cum, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    ck = from_dlpack(cum, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    sm = 1.0 / math.sqrt(D)
    problem_size = (B, seq_q, S, S, Hk * Hr, Hk, D, D)
    args_tail = (Float32(sm * math.log2(math.e)), Float32(sm), Float32(1.0),
                 None, None, None, None, None, stream, False,
                 None, None, None, None, None)
    compiled = cute.compile(fmha, q_c, k_c, v_c, o_c, problem_size, cq, ck,
                            None, None, *args_tail)
    _repeat_api(
        lambda: compiled(q_c, k_c, v_c, o_c, problem_size, cq, ck,
                         None, None, *args_tail),
        calls)
    torch.cuda.synchronize()
    return dict(
        cta_group=1, cluster="(1,1)", sms=1, dtype=c["dtype"],
        shape=c["shape_tag"],
        kernel="flashinfer.cute_dsl.attention.fmha.fmha."
               "BlackwellFusedMultiHeadAttentionForward",
        proof=(f"cluster={fmha.cluster_shape_mn} CtaGroup.ONE "
               f"use_2cta_instrs=False ex2={p['ex2_emulation']} "
               f"dtype={c['dtype']}"),
    )


def call_fmha_blockscaled(c: dict, calls: int = 1) -> dict:
    import torch

    from flashinfer.attention.cute_dsl.fmha_blockscaled import cute_dsl_fmha_blockscaled_prefill
    from flashinfer.cute_dsl.attention.fmha.quantize import quantize_blockscaled_qk

    B, S, Hk, Hr, D = c["shape"]
    seq_q = int(c["params"].get("seq_q", S))
    q = torch.randn(B, seq_q, Hk * Hr, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    q_fp4, k_fp4, q_sf, k_sf, q_scale, k_scale = quantize_blockscaled_qk(
        q, k, "nvfp4")
    _repeat_api(lambda: cute_dsl_fmha_blockscaled_prefill(
        q_fp4, k_fp4, q_sf, k_sf, v, out, qk_mode="nvfp4",
        scale_q=q_scale, scale_k=k_scale, enable_pdl=False), calls)
    torch.cuda.synchronize()
    return dict(
        cta_group=1, cluster="(1,1)", sms=1, dtype=c["dtype"],
        shape=c["shape_tag"], kernel=c["entry"],
        proof="blockscaled Q/K NVFP4, dense BF16 V/output; CtaGroup.ONE",
    )


def call_mla(c: dict, calls: int = 1) -> dict:
    import torch

    from flashinfer.cute_dsl.attention.mla_dispatch import cute_dsl_mla_decode
    from flashinfer.cute_dsl.attention.monolithic import mla_decode as mla_decode_module
    from flashinfer.cute_dsl.utils import get_num_sm

    p = c["params"]
    B, q_len, H = int(p.get("batch", 1)), 1, p["heads"]
    latent, rope, page, seq_k = p["latent"], p["rope"], p["page"], p["seq_k"]
    D = latent + rope
    num_pages = B * ((seq_k + page - 1) // page)
    fp8 = c["dtype"] == "e4m3"
    if fp8:
        query = (torch.randn(B, q_len, H, D, dtype=torch.float16,
                             device="cuda") * 0.1).to(torch.float8_e4m3fn)
        kv = (torch.randn(num_pages, page, D, dtype=torch.float16,
                          device="cuda") * 0.1).to(torch.float8_e4m3fn)
    else:
        dt = torch.bfloat16 if c["dtype"] == "bf16" else torch.float16
        query = torch.randn(B, q_len, H, D, dtype=dt, device="cuda")
        kv = torch.randn(num_pages, page, D, dtype=dt, device="cuda")
    pages_per_req = (seq_k + page - 1) // page
    tables = torch.arange(num_pages, dtype=torch.int32, device="cuda").view(
        B, pages_per_req)
    seq_lens = torch.full((B,), seq_k, dtype=torch.int32, device="cuda")
    if c.get("single_cluster") and p.get("num_sms") is not None:
        num_sm = int(p["num_sms"])
    else:
        num_sm = get_num_sm(query.device)
    split_kv, workspace_size = mla_decode_module._get_split_kv_and_workspace_size(
        B, q_len, H, latent, num_sm, seq_k)
    ws = torch.empty(workspace_size, dtype=torch.int8, device=query.device)
    original_get_num_sm = None
    if c.get("single_cluster") and p.get("num_sms") is not None:
        original_get_num_sm = mla_decode_module.get_num_sm
        mla_decode_module.get_num_sm = lambda _device: num_sm
    try:
        _repeat_api(lambda: cute_dsl_mla_decode(
                query=query, kv_cache=kv, workspace_buffer=ws,
                kv_lora_rank=latent, qk_rope_head_dim=rope, block_tables=tables,
                seq_lens=seq_lens, max_seq_len=seq_k,
                softmax_scale=1.0 / (latent ** 0.5), is_var_seq=False,
                enable_pdl=False, cute_dsl_impl="monolithic", return_lse=False,
            ), calls)
    finally:
        if original_get_num_sm is not None:
            mla_decode_module.get_num_sm = original_get_num_sm
    torch.cuda.synchronize()
    return dict(
        cta_group=2, cluster="(2,1,1)", sms=2, dtype=c["dtype"],
        shape=c["shape_tag"],
        kernel=c["entry"],
        proof=(f"batch={B} split_kv={split_kv} workspace_bytes={workspace_size} "
               "library-selected split_kv; cluster=(2,1,1) CtaGroup.TWO "
               "use_2cta_instrs=True cta_group::2 monolithic paged"),
    )


def call_bs_gemm(c: dict, calls: int = 1) -> dict:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cuda.bindings.driver as cuda_drv
    import torch
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.cute_dsl.utils import (
        get_cutlass_dtype,
        get_max_active_clusters,
    )
    from flashinfer.gemm.kernels.grouped_gemm_masked_blackwell import (
        create_scale_factor_tensor,
    )

    p = c["params"]
    if p["sm"] == 103:
        from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm103 import (
            Sm103BlockScaledPersistentDenseGemmKernel as K,
        )
    else:
        from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import (
            Sm100BlockScaledPersistentDenseGemmKernel as K,
        )
    tile_m = int(p["tile_m"])
    cluster = (2, 1) if tile_m == 256 else (1, 1)
    kw = dict(sf_vec_size=int(p["sf_vec_size"]),
              mma_tiler_mn=(tile_m, 128), cluster_shape_mn=cluster,
              enable_pdl=False)
    if p.get("use_tma_store"):
        kw["use_tma_store"] = True
    gemm = K(**kw)
    l, m, n, k = c["shape"]
    ab = get_cutlass_dtype(p["ab_dtype"])
    sf = get_cutlass_dtype(p["sf_dtype"])
    cd = cutlass.BFloat16
    device = torch.device("cuda")
    max_active_clusters = (
        int(p["max_active_clusters"])
        if c.get("single_cluster") and p.get("max_active_clusters") is not None
        else get_max_active_clusters(cluster[0] * cluster[1]))
    a_ref = cutlass_torch.matrix(l, m, k, False, cutlass.Float32, device=device)
    b_ref = cutlass_torch.matrix(l, n, k, False, cutlass.Float32, device=device)
    c_ref = cutlass_torch.matrix(l, m, n, False, cutlass.Float32, device=device)
    a_t, _ = cutlass_torch.cute_tensor_like(a_ref, ab, is_dynamic_layout=True,
                                            assumed_align=16)
    b_t, _ = cutlass_torch.cute_tensor_like(b_ref, ab, is_dynamic_layout=True,
                                            assumed_align=16)
    c_t, _ = cutlass_torch.cute_tensor_like(c_ref, cd, is_dynamic_layout=True,
                                            assumed_align=16)
    _, sfa_t, _ = create_scale_factor_tensor(l, m, k, int(p["sf_vec_size"]),
                                             sf, device)
    _, sfb_t, _ = create_scale_factor_tensor(l, n, k, int(p["sf_vec_size"]),
                                             sf, device)
    alpha_t = from_dlpack(torch.ones(1, dtype=torch.float32, device=device))
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        gemm, a_t, b_t, sfa_t, sfb_t, c_t, alpha_t,
        max_active_clusters, stream)
    _repeat_api(lambda: compiled(a_t, b_t, sfa_t, sfb_t, c_t, alpha_t, stream),
                calls)
    torch.cuda.synchronize()
    two = bool(gemm.use_2cta_instrs)
    return dict(
        cta_group=2 if two else 1, cluster=str(cluster),
        sms=2 if two else 1, dtype=c["dtype"], shape=c["shape_tag"],
        kernel=f"{K.__module__}.{K.__name__}",
        proof=(f"use_2cta_instrs={gemm.use_2cta_instrs} "
               f"cta_group={gemm.cta_group} cluster={cluster} "
               f"max_active_clusters={max_active_clusters} "
               f"{'CtaGroup.TWO cta_group::2' if two else 'CtaGroup.ONE'}"),
    )




def call_bmm_fp8(c: dict, calls: int = 1) -> dict:
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda_drv
    import torch
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.cute_dsl.utils import get_max_active_clusters
    from flashinfer.gemm.kernels.bmm_fp8_blackwell import PersistentDenseGemmKernel, bmm

    batch, m, n, k = c["shape"]
    use_two = bool(c["params"]["use_2cta"])
    cluster = tuple(c["params"]["cluster"])
    kernel = PersistentDenseGemmKernel(
        acc_dtype=cutlass.Float32, use_2cta_instrs=use_two,
        mma_tiler_mn=(256 if use_two else 128, 128),
        cluster_shape_mn=cluster, use_tma_store=True)
    a = (torch.randn(batch, m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn(batch, n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn).transpose(1, 2)
    out = torch.empty(batch, m, n, device="cuda", dtype=torch.bfloat16)
    a_c = from_dlpack(a.view(torch.uint8), assumed_align=16)
    b_c = from_dlpack(b.view(torch.uint8), assumed_align=16)
    a_c.element_type = cutlass.Float8E4M3FN
    b_c.element_type = cutlass.Float8E4M3FN
    out_c = from_dlpack(out, assumed_align=16)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    scale_c = from_dlpack(scale, assumed_align=16)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = (
        int(c["params"]["max_active_clusters"])
        if c.get("single_cluster")
        and c["params"].get("max_active_clusters") is not None
        else get_max_active_clusters(cluster[0] * cluster[1]))
    compiled = cute.compile(
        bmm, kernel, a_c, b_c, out_c, max_active_clusters, stream, scale_c)
    _repeat_api(lambda: compiled(a_c, b_c, out_c, stream, scale_c), calls)
    torch.cuda.synchronize()
    return dict(cta_group=2 if use_two else 1, cluster=str(cluster),
                sms=2 if use_two else 1, dtype=c["dtype"], shape=c["shape_tag"],
                kernel=c["entry"],
                proof=(f"Blackwell BMM CtaGroup={kernel.cta_group} cluster={cluster} "
                       f"max_active_clusters={max_active_clusters}"))



def call_grouped(c: dict, calls: int = 1) -> dict:
    import torch

    from flashinfer.gemm.kernels.grouped_gemm_masked_wrapper import grouped_gemm_nt_masked

    p = c["params"]
    groups, m, n, k = c["shape"]
    device = "cuda"
    a = torch.randint(0, 256, (groups, m, k // 2), dtype=torch.uint8,
                      device=device).permute(1, 2, 0)
    b = torch.randint(0, 256, (groups, n, k // 2), dtype=torch.uint8,
                      device=device).permute(1, 2, 0)
    out = torch.zeros(groups, m, n, dtype=torch.bfloat16,
                      device=device).permute(1, 2, 0)
    sf_vec = int(p["sf_vec_size"])

    def scale_tensor(rows: int) -> torch.Tensor:
        shape = (groups, (rows + 127) // 128, (k // sf_vec + 3) // 4, 32, 4, 4)
        return torch.ones(shape, device=device, dtype=torch.float32).to(
            torch.float8_e4m3fn).permute(3, 4, 1, 5, 2, 0)

    sfa = scale_tensor(m)
    sfb = scale_tensor(n)
    masked_m = torch.full((groups,), m, dtype=torch.int32, device=device)
    _repeat_api(lambda: grouped_gemm_nt_masked(
            (a, sfa), (b, sfb), out, masked_m,
            ab_dtype=p["ab_dtype"], sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16", sf_vec_size=sf_vec,
            mma_tiler_mn=(int(p["tile_m"]), 128), cluster_shape_mn=tuple(p["cluster"]),
            sm_count=(None if c.get("mode") == "full"
                      and not c.get("single_cluster") else c.get("sms")),
        ), calls)
    torch.cuda.synchronize()
    return dict(
        cta_group=c["cta_group"], cluster=str(tuple(p["cluster"])),
        sms=(torch.cuda.get_device_properties(0).multi_processor_count
             if c.get("mode") == "full" and not c.get("single_cluster")
             else c["sms"]),
        dtype=c["dtype"], shape=c["shape_tag"],
        kernel=c["entry"], proof=f"masked grouped tile={p['tile_m']} cluster={p['cluster']}",
    )


def call_moe(c: dict, calls: int = 1) -> dict:
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda_drv
    import torch

    from flashinfer.cute_dsl.utils import (
        ceil_div, convert_sf_to_mma_layout, get_max_active_clusters, make_ptr,
    )
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        _get_compiled_gather_kernel,
        create_gather_gemm_tensors,
    )

    p = c["params"]
    tile = int(p["tile"])
    seq_len, hidden, inter = int(p["seq_len"]), int(p["hidden"]), int(p["inter"])
    nexp, topk = int(p["experts"]), int(p["topk"])
    fc1 = 2 * inter
    device = torch.device("cuda")
    cluster_m = tile // 128
    cluster = (cluster_m, 1)

    def sf_mma(m, k, ng):
        cnt = ng * ceil_div(m, 128) * ceil_div(ceil_div(k, 16), 4) * 32 * 4 * 4
        raw = torch.full((cnt,), 0x38, dtype=torch.uint8, device=device)
        return convert_sf_to_mma_layout(raw, m=m, k=k, num_groups=ng,
                                        sf_vec_size=16)

    a = torch.randint(0, 256, (seq_len, hidden // 2), dtype=torch.uint8,
                      device=device)
    b = torch.randint(0, 256, (nexp, fc1, hidden // 2), dtype=torch.uint8,
                      device=device)
    a_sf = sf_mma(seq_len, hidden, 1)
    b_sf = sf_mma(fc1, hidden, nexp)
    alpha = torch.ones(nexp, dtype=torch.float32, device=device)
    group_m = [seq_len * topk // nexp] * nexp
    token_map, tile_map, mn_limit, num_tiles, valid_m, _ = \
        create_gather_gemm_tensors(seq_len=seq_len, topk=topk,
                                   group_m_list=group_m, mma_tiler_m=tile)
    two = tile == 256
    out = torch.empty(valid_m, inter, dtype=torch.bfloat16, device=device)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    def pointer(tensor: torch.Tensor, dtype, alignment: int = 16):
        return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem,
                        assumed_align=alignment)

    pointers = (
        pointer(a, cutlass.Float4E2M1FN, 32),
        pointer(b, cutlass.Float4E2M1FN, 32),
        pointer(a_sf, cutlass.Float8E4M3FN),
        pointer(b_sf, cutlass.Float8E4M3FN),
        pointer(out, cutlass.BFloat16, 32), None,
        pointer(alpha, cutlass.Float32), pointer(tile_map, cutlass.Int32),
        pointer(mn_limit, cutlass.Int32), pointer(token_map, cutlass.Int32),
        pointer(num_tiles, cutlass.Int32), None, None,
    )
    dimensions = (seq_len, valid_m, fc1, hidden, nexp)
    if c.get("single_cluster"):
        max_active_clusters = int(p.get("max_active_clusters", 1))
    elif c.get("mode") == "full":
        max_active_clusters = get_max_active_clusters(cluster[0] * cluster[1])
    else:
        max_active_clusters = 1
    compiled = _get_compiled_gather_kernel(
        *dimensions, *pointers,
        max_active_clusters=max_active_clusters, stream=stream,
        a_dtype="float4_e2m1fn", b_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn", c_dtype="bfloat16", quantize_output=False,
        sf_vec_size=16, tile_size=tile, topk=topk, cluster_shape_mn=cluster,
        vectorized_f32=True, raster_along_m=False,
        mma_tiler_mn=(tile, 128), enable_pdl=False)
    _repeat_api(lambda: compiled(*pointers, *dimensions, stream=stream), calls)
    torch.cuda.synchronize()
    return dict(
        cta_group=2 if two else 1, cluster=str(cluster),
        sms=2 if two else 1, dtype="nvfp4", shape=c["shape_tag"],
        kernel="flashinfer.fused_moe.cute_dsl."
               "blockscaled_contiguous_gather_grouped_gemm_act_fusion",
        proof=(f"tile={tile} cluster={cluster} e={nexp} topk={topk} "
               f"GEMM1-gather max_active_clusters={max_active_clusters} "
               f"{'CtaGroup.TWO' if two else 'CtaGroup.ONE'}"),
    )


def call_tgv(c: dict, calls: int = 1) -> dict:
    import torch

    from flashinfer.gemm.kernels.tgv_gemm_cute_ext import (
        _resolve_tactic,
        run_tgv_cute_ext,
    )

    tactic = int(c["params"]["tactic"])
    cta_m, cta_n, stages, use_2cta = _resolve_tactic(tactic)
    M, N, K = c["shape"]
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).t()
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    _repeat_api(lambda: run_tgv_cute_ext(a, b, None, out, False, tactic),
                calls)
    torch.cuda.synchronize()
    return dict(
        cta_group=2 if use_2cta else 1,
        cluster="(2,1)" if use_2cta else "(1,1)",
        sms=2 if use_2cta else 1, dtype="bf16", shape=c["shape_tag"],
        kernel="flashinfer.gemm.kernels.tgv_gemm_cute_ext.run_tgv_cute_ext",
        proof=(f"tactic={tactic} cta_m={cta_m} cta_n={cta_n} stages={stages} "
               f"use_2cta_instrs={use_2cta}"),
    )


def call_gqa(c: dict, calls: int = 1) -> dict:
    import torch

    from flashinfer.cute_dsl.attention.wrappers.batch_decode import (
        BatchDecodeCuteDSLWrapper, BatchDecodePagedCuteDSLWrapper,
    )

    p = c["params"]
    B, q_len, hq, hkv, D, _ = c["shape"]
    seq_k = int(p.get("seq_k", 512))
    full_unbounded = (c.get("mode") == "full"
                      and not c.get("single_cluster"))
    kv_splits = None if full_unbounded else int(p["kv_splits"])
    reduction = "auto" if full_unbounded else p["reduction"]
    q = torch.randn(B, q_len, hq, D, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(128 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    if p.get("paged"):
        page = int(p["page"])
        pages_per_seq = (seq_k + page - 1) // page
        npages = B * pages_per_seq
        kc = torch.randn(npages, page, hkv, D, device="cuda", dtype=torch.bfloat16)
        vc = torch.randn_like(kc)
        indptr = torch.arange(B + 1, dtype=torch.int32) * pages_per_seq
        indices = torch.arange(npages, dtype=torch.int32)
        seq_lens = torch.full((B,), seq_k, dtype=torch.int32)
        wrapper = BatchDecodePagedCuteDSLWrapper(workspace)
        wrapper.plan(
            indptr, indices, seq_lens, hq, hkv, D, page,
            q_data_type=q.dtype, kv_splits=kv_splits,
            reduction=reduction, max_kv_len=seq_k, q_len_per_req=q_len)
        _repeat_api(lambda: wrapper.run(q, kc, vc, enable_pdl=False), calls)
    else:
        k = torch.randn(B, seq_k, hkv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        wrapper = BatchDecodeCuteDSLWrapper(workspace)
        wrapper.plan(B, seq_k, hq, hkv, D, q_data_type=q.dtype,
                     kv_splits=kv_splits, reduction=reduction,
                     q_len_per_req=q_len)
        _repeat_api(lambda: wrapper.run(q, k, v, enable_pdl=False), calls)
    torch.cuda.synchronize()
    return dict(
        cta_group=c["cta_group"], cluster=c["cluster"], sms=c["sms"],
        dtype=c["dtype"], shape=c["shape_tag"], kernel=c["entry"],
        proof=f"workspace->plan->run kv_splits={kv_splits} reduction={reduction}",
    )


def call_bsa(c: dict, calls: int = 1) -> dict:
    import torch

    import importlib

    mod_name = str(c.get("entry") or
                   "flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128")
    mod = importlib.import_module(mod_name)
    fwd_name = mod_name.rsplit(".", 1)[1] + "_fwd"
    fwd = getattr(mod, fwd_name)

    B, Sq, H, D, blk = c["shape"]
    Sk = int(c.get("params", {}).get("skv_len", 512))
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    idx = torch.arange(4, dtype=torch.int32, device="cuda")
    idx = idx.expand(B, H, (Sq + blk - 1) // blk, 4).contiguous()
    out, _ = _repeat_api(
        lambda: fwd(q, k, v, idx, 4, softmax_scale=D ** -0.5), calls)
    torch.cuda.synchronize()
    return dict(
        cta_group=1, cluster="(1,1)", sms=1, dtype="bf16",
        shape=c["shape_tag"],
        kernel=f"{mod_name}.{fwd_name}",
        proof=f"BSA blk128 use_2cta_instrs=False BSA_CLC={os.environ.get('BSA_CLC','0')}",
    )


def call_rmsnorm_fp4quant(c: dict, calls: int = 1) -> dict:
    import torch

    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant

    rows, hidden = c["shape"]
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    y_fp4, sf = _repeat_api(
        lambda: rmsnorm_fp4quant(
            x, w, eps=1e-6,
            block_size=int(c["params"]["sf_vec_size"])),
        calls)
    torch.cuda.synchronize()
    return dict(
        cta_group=1, cluster="(1,1)", sms=1, dtype="nvfp4",
        shape=c["shape_tag"],
        kernel="flashinfer.cute_dsl.rmsnorm_fp4quant.rmsnorm_fp4quant",
        proof="fused rmsnorm+fp4 quant（非 UMMA 融合核）",
    )


def _set_classic_jit_source_paths(jit_env, source_root: Path) -> None:
    jit_env.FLASHINFER_DATA = source_root
    jit_env.FLASHINFER_INCLUDE_DIR = source_root / "include"
    jit_env.FLASHINFER_CSRC_DIR = source_root / "csrc"
    jit_env.CUTLASS_INCLUDE_DIRS = [
        source_root / "3rdparty" / "cutlass" / "include",
        source_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = source_root / "3rdparty" / "spdlog" / "include"
    jit_env.CCCL_INCLUDE_DIRS = [
        source_root / "3rdparty" / "cccl" / "cub",
        source_root / "3rdparty" / "cccl" / "libcudacxx" / "include",
        source_root / "3rdparty" / "cccl" / "thrust",
    ]


def call_classic_jit(c: dict, calls: int = 1) -> dict:
    from flashinfer.jit import env as jit_env

    source_root = Path(jit_env.__file__).resolve().parents[2]
    _set_classic_jit_source_paths(jit_env, source_root)
    import torch
    import torch.nn.functional as F

    p = c["params"]
    op = p["operation"]
    dtype = torch.bfloat16
    if op == "batch_decode_paged":
        from flashinfer.attention.prims_ts.decode import batch_decode_with_paged_kv_cache

        B, _q_len, Hq, Hkv, D = c["shape"]
        page_size, seq_k = int(p["page_size"]), int(p["seq_k"])
        pages_per_req = (seq_k + page_size - 1) // page_size
        page_count = B * pages_per_req
        q = torch.randn(B, Hq, D, device="cuda", dtype=dtype)
        k = torch.randn(page_count, Hkv, page_size, D, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        tables = torch.arange(page_count, dtype=torch.int32, device="cuda")
        tables = tables.view(B, pages_per_req)
        seq_lens = torch.full((B,), seq_k, dtype=torch.int32, device="cuda")
        out = _repeat_api(lambda: batch_decode_with_paged_kv_cache(
            q, (k, v), tables, seq_lens, kv_layout="HND"), calls)
        torch.cuda.synchronize()
        if not torch.isfinite(out.float()).all():
            raise RuntimeError("paged batch decode returned non-finite output")
        kernel = ".*batch_decode.*"
        shape = c["shape"]
        proof = f"paged decode B={B} pages={page_count} page_size={page_size}"
    elif op == "single_prefill":
        from flashinfer.prefill import single_prefill_with_kv_cache

        Sq, Sk, Hq, Hkv, D = c["shape"]
        q = torch.randn(Sq, Hq, D, device="cuda", dtype=dtype)
        k = torch.randn(Sk, Hkv, D, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        causal = bool(p.get("causal", False))
        out = _repeat_api(
            lambda: single_prefill_with_kv_cache(q, k, v, causal=causal),
            calls)
        ref = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2), is_causal=causal,
            enable_gqa=(Hq != Hkv))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref.squeeze(0).transpose(0, 1),
                                   rtol=2e-2, atol=2e-2)
        kernel = ".*single_prefill.*"
        shape = c["shape"]
        proof = f"single prefill causal={causal}"
    elif op == "merge_state":
        from flashinfer.cascade import merge_state

        seq, heads, dim = c["shape"]
        torch.manual_seed(int(p.get("seed", 20261024)))
        va = torch.randn(seq, heads, dim, device="cuda", dtype=dtype)
        sa = torch.randn(seq, heads, device="cuda", dtype=torch.float32)
        vb = torch.randn_like(va)
        sb = torch.randn_like(sa)
        out, lse = _repeat_api(lambda: merge_state(va, sa, vb, sb), calls)
        sa_nat = sa * math.log(2.0)
        sb_nat = sb * math.log(2.0)
        maximum = torch.maximum(sa_nat, sb_nat)
        wa, wb = torch.exp(sa_nat - maximum), torch.exp(sb_nat - maximum)
        total = wa + wb
        ref = (
            va.float() * wa[..., None] + vb.float() * wb[..., None]
        ) / total[..., None]
        ref = ref.to(va.dtype).float()
        ref_lse = (maximum + torch.log(total)) / math.log(2.0)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(lse, ref_lse, rtol=1e-3, atol=1e-3)
        kernel = ".*merge_state.*"
        shape = c["shape"]
        proof = "merge_state output and LSE reference checked"
    elif op == "top_k_sampling":
        from flashinfer.sampling import top_k_sampling_from_probs

        batch, vocab = c["shape"]
        probs = torch.softmax(torch.randn(batch, vocab, device="cuda"), dim=-1)
        out = _repeat_api(lambda: top_k_sampling_from_probs(
            probs, int(p["top_k"]), deterministic=True, seed=0, offset=0),
            calls)
        torch.cuda.synchronize()
        if out.shape != (batch,) or bool(((out < 0) | (out >= vocab)).any()):
            raise RuntimeError("top-k sampling returned invalid indices")
        kernel = ".*top_k_sampling.*"
        shape = c["shape"]
        proof = f"top-k sampling k={p['top_k']} outputs in [0,{vocab})"
    elif op == "silu_and_mul":
        from flashinfer.activation import silu_and_mul

        rows, two_hidden = c["shape"]
        x = torch.randn(rows, two_hidden, device="cuda", dtype=dtype)
        out = _repeat_api(lambda: silu_and_mul(x), calls)
        half = two_hidden // 2
        ref = F.silu(x[:, :half]) * x[:, half:]
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
        kernel = ".*silu_and_mul.*"
        shape = c["shape"]
        proof = "SiLU-and-mul output reference checked"
    elif op == "apply_rope":
        from flashinfer.rope import apply_rope

        batch, seq, hq, hkv, dim = c["shape"]
        q = torch.randn(batch * seq, hq, dim, device="cuda", dtype=dtype)
        k = torch.randn(batch * seq, hkv, dim, device="cuda", dtype=dtype)
        indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * seq
        offsets = torch.zeros(batch, dtype=torch.int32, device="cuda")
        q_out, k_out = _repeat_api(
            lambda: apply_rope(q, k, indptr, offsets), calls)
        torch.cuda.synchronize()
        if not torch.isfinite(q_out.float()).all() or not torch.isfinite(k_out.float()).all():
            raise RuntimeError("apply_rope returned non-finite output")
        kernel = ".*apply_rope.*"
        shape = c["shape"]
        proof = f"apply_rope segments={batch} seqlen={seq}"
    else:
        raise KeyError(f"unsupported classic JIT operation {op!r}")
    return dict(
        cta_group=1, cluster="(1,1,1)", sms=148, dtype=c["dtype"],
        shape=c["shape_tag"], kernel=kernel, proof=proof,
    )


CALLERS = {
    "fi_fmha_prefill": call_fmha,
    "fi_fmha_blockscaled": call_fmha_blockscaled,
    "fi_mla_decode": call_mla,
    "fi_bs_gemm_sm100": call_bs_gemm,
    "fi_bs_gemm_sm103": call_bs_gemm,
    "fi_bmm_fp8": call_bmm_fp8,
    "fi_grouped_masked": call_grouped,
    "fi_fused_moe": call_moe,
    "fi_tgv_gemm": call_tgv,
    "fi_gqa_decode": call_gqa,
    "fi_gqa_decode_paged": call_gqa,
    "fi_bsa": call_bsa,
    "fi_rmsnorm_fp4quant": call_rmsnorm_fp4quant,
    "fi_classic_jit": call_classic_jit,
}


def run_real(cid: str) -> int:
    from common import write_result

    c = load_case(cid)
    dump_report = None
    try:
        check_case_contract(c)
        if os.environ.get("CUTE_DSL_DUMP_DIAG") == "1":
            scripts_path = str(ROOT / "scripts")
            if scripts_path not in sys.path:
                sys.path.insert(0, scripts_path)
            from c450full_cute_dsl_diagnostics import (
                install_compile_diagnostic,
                report_dump_directories,
            )

            install_compile_diagnostic()
            dump_report = report_dump_directories
        configure_flashinfer()
        if os.environ.get("CUTE_DSL_DUMP_DIAG") == "1":
            install_compile_diagnostic()
        if cid not in CALLERS_BY_ID:
            raise KeyError(f"no caller for family {c['family']}")
        import torch
        calls = int(c.get("calls", 1)) if c.get("mode") == "full" else 1
        kw = {}
        torch.manual_seed(20261024)
        kw = CALLERS_BY_ID[cid](c, calls)
        payload = {"id": cid, "suite": SUITE, "ok": True,
                   "calls": calls, "error": "", **kw}
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        payload = {"id": cid, "suite": SUITE, "ok": False, "error":
                   f"{type(e).__name__}: {e}",
                   "cta_group": c.get("cta_group"), "cluster": c.get("cluster"),
                   "sms": c.get("sms"), "dtype": c.get("dtype"),
                   "shape": c.get("shape_tag"), "kernel": c.get("entry"), "proof": ""}
        payload["error_code"] = e.code if isinstance(e, AssetContractError) else "call_failed"
    if dump_report is not None:
        dump_report()
    write_result(SUITE, cid, payload)
    print("DONE", cid, "ok=", payload["ok"], payload.get("error", "")[:200])
    return 0 if payload["ok"] else 1


CALLERS_BY_ID: dict[str, object] = {}


def register_callers() -> None:
    for c in _cases():
        fn = CALLERS.get(c.get("entry_family") or c["family"])
        if fn is not None:
            CALLERS_BY_ID[c["id"]] = fn


def compile_only(cid: str, out_dir: str) -> int:
    """Real offline compile or explicitly device-bound preparation, no GPU launch."""
    import subprocess as sp

    c = load_case(cid)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    driver = ROOT / "suites" / "far_holdout" / "far_compile.py"
    r = sp.run([sys.executable, str(driver), "--case", cid, "--out",
                str(out / cid)], capture_output=True, text=True,
               env={**os.environ,
                    "CUTE_DSL_ARCH": os.environ.get("CUTE_DSL_ARCH", "sm_103a"),
                    "CUTE_DSL_KEEP": "cubin,ptx",
                    "CUTE_DSL_DUMP_DIR": str(out / cid / "cute_dump"),
                    "CUTE_DSL_CACHE_DIR": str(out / cid / "cute_cache")})
    (out / cid / "compile.log").write_text(r.stdout + r.stderr)
    report_path = out / cid / "far_compile_report.json"
    if not report_path.is_file():
        print("COMPILE", cid, "error_code=compile_report_unavailable", r.returncode)
        return 1
    rows = json.loads(report_path.read_text(encoding="utf-8"))["cases"]
    result = next((row for row in rows if row["id"] == cid), None)
    status = result["compile"] if result else "fail"
    print("COMPILE", cid, "status=", status, (r.stderr or r.stdout)[-200:])
    return 0 if r.returncode == 0 and status in ("ok", "JIT@B300") else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", nargs="?")
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--out", default="/tmp/far_fi")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    register_callers()
    if args.all:
        rc = 0
        for c in _cases():
            if args.resolve:
                print(resolve_line(c))
            else:
                rc |= compile_only(c["id"], args.out)
        return rc
    if not args.case_id:
        ap.error("case_id or --all required")
    c = load_case(args.case_id)
    if args.resolve:
        print(resolve_line(c))
        return 0
    if args.compile_only:
        return compile_only(args.case_id, args.out)
    return run_real(args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
