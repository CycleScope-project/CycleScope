#!/usr/bin/env python3
"""Run one pinned FlashInfer Hopper delivery case through its public API."""

from __future__ import annotations

import os
import sys

_delivery_site = os.environ.get("FLASHINFER_DELIVERY_SITE")
if _delivery_site:
    sys.path.insert(0, _delivery_site)

import argparse
from itertools import accumulate
import json
import math
from pathlib import Path
import time
from typing import Callable


# The shared delivery timer lives in suites/common (stage-6
# consolidation; the former suite-local harness copy is gone), so common
# must be importable before the timer import below.
COMMON_ROOT = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
from globaltimer_timer import (  # noqa: E402
    measure_operation,
    measure_two_phase_operation,
)
from matched_kernel_timing import (  # noqa: E402
    load_comparison_contract,
    validate_runtime_binary_identity,
)


# Run-once capture mode: public-API build plus this many warm runs before
# the single captured call (same warmup count the pre-stage-6 capture window
# used; keeps JIT/module materialization outside the captured launch).
CAPTURE_WARMUP_RUNS = 3

def load_case(path: Path, case_id: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    found = [case for case in data["cases"] if case["case_id"] == case_id]
    if len(found) != 1:
        raise ValueError(f"expected one case named {case_id}, found {len(found)}")
    return found[0]


def dtype(torch, name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3": torch.float8_e4m3fn,
    }[name]


def random_tensor(torch, shape, value_dtype, scale=0.5):
    return (torch.randn(*shape, dtype=torch.float32, device="cuda") * scale).to(value_dtype)


def lengths(batch: int, base: int, unequal: bool) -> list[int]:
    if not unequal:
        return [max(1, base)] * batch
    step = max(1, base // 8)
    return [max(1, base + (index % 5 - 2) * step) for index in range(batch)]


def cumsum_indptr(torch, values: list[int]):
    data = [0]
    for value in values:
        data.append(data[-1] + value)
    return torch.tensor(data, dtype=torch.int32, device="cuda")


def page_plan(torch, kv_lens: list[int], page_size: int, scattered: bool):
    pages = [(length + page_size - 1) // page_size for length in kv_lens]
    used = sum(pages)
    spare = used if scattered else 0
    pool = used + spare
    if scattered:
        order = torch.randperm(pool, device="cuda", dtype=torch.int32)[:used]
    else:
        order = torch.arange(used, dtype=torch.int32, device="cuda")
    indptr = cumsum_indptr(torch, pages)
    last = torch.tensor(
        [length - (count - 1) * page_size for length, count in zip(kv_lens, pages)],
        dtype=torch.int32,
        device="cuda",
    )
    return pool, order.contiguous(), indptr, last


def paged_kv_shape(layout: str, pages: int, page_size: int, heads: int, dim: int, stacked: bool):
    if layout == "HND":
        core = (pages, heads, page_size, dim)
    else:
        core = (pages, page_size, heads, dim)
    return ((pages, 2, heads, page_size, dim) if layout == "HND" else (pages, 2, page_size, heads, dim)) if stacked else core


def sdpa(torch, q, k, v, *, causal, window_left, mask, alibi, soft_cap, sm_scale):
    q32, k32, v32 = q.float(), k.float(), v.float()
    heads_q, heads_kv = q32.shape[1], k32.shape[1]
    if heads_q != heads_kv:
        repeat = heads_q // heads_kv
        k32 = k32.repeat_interleave(repeat, dim=1)
        v32 = v32.repeat_interleave(repeat, dim=1)
    qh = q32.permute(1, 0, 2)
    kh = k32.permute(1, 0, 2)
    vh = v32.permute(1, 0, 2)
    scale = (1.0 / math.sqrt(q32.shape[-1])) if sm_scale is None else sm_scale
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    seq_q, seq_k = q32.shape[0], k32.shape[0]
    if alibi:
        slopes = torch.tensor(
            [2 ** (-(head + 1) * 8 / heads_q) for head in range(heads_q)],
            device=q32.device,
            dtype=torch.float32,
        )
        q_pos = torch.arange(seq_q, device=q32.device)
        k_pos = torch.arange(seq_k, device=q32.device)
        scores = scores + slopes[:, None, None] * (k_pos[None, None, :] - q_pos[None, :, None]).float()
    if soft_cap:
        scores = soft_cap * torch.tanh(scores / soft_cap)
    bias = torch.zeros(seq_q, seq_k, device=q32.device, dtype=torch.float32)
    if mask is not None:
        bias = bias.masked_fill(~mask, float("-inf"))
    else:
        if causal:
            q_idx = torch.arange(seq_q, device=q32.device)[:, None]
            k_idx = torch.arange(seq_k, device=q32.device)[None, :]
            bias = bias.masked_fill(k_idx > (k32.shape[0] - seq_q) + q_idx, float("-inf"))
        if window_left is not None and window_left >= 0:
            q_idx = torch.arange(seq_q, device=q32.device)[:, None]
            k_idx = torch.arange(seq_k, device=q32.device)[None, :]
            start = (seq_k - seq_q) + q_idx - window_left
            bias = bias.masked_fill(k_idx < start, float("-inf"))
    scores = scores + bias
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vh).permute(1, 0, 2)
    return out


def apply_llama_rope(torch, q, k, theta=10000.0):
    def rotate(x):
        seq, _, dim = x.shape
        half = dim // 2
        pos = torch.arange(seq, device=x.device, dtype=torch.float32)
        freq = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
        ang = pos[:, None] * freq[None, :]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        x1, x2 = x[..., :half].float(), x[..., half:].float()
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)
    return rotate(q), rotate(k)


def close(torch, actual, expected, *, rtol, atol, name):
    actual_f, expected_f = actual.float(), expected.float()
    if actual_f.numel() == 0:
        raise AssertionError(f"{name}: empty output")
    if not torch.isfinite(actual_f).all():
        raise AssertionError(f"{name}: non-finite output")
    if float(actual_f.abs().max()) == 0.0 and float(expected_f.abs().max()) > 1e-4:
        raise AssertionError(f"{name}: output is all zeros")
    if not torch.allclose(actual_f, expected_f, rtol=rtol, atol=atol):
        delta = (actual_f - expected_f).abs().max().item()
        raise AssertionError(f"{name}: max abs error {delta} exceeds rtol={rtol} atol={atol}")


def fp8_block_quant(torch, weight, block=128):
    experts, rows, cols = weight.shape
    if rows % block or cols % block:
        raise ValueError("fp8 blockscale weights need dims divisible by 128")
    view = weight.float().reshape(experts, rows // block, block, cols // block, block)
    amax = view.abs().amax(dim=(2, 4)).clamp(min=1e-4)
    scale = amax / 448.0
    quant = (view / scale[:, :, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(experts, rows, cols)
    return quant, scale.contiguous()


def dequant_fp8_block(quant, scale, block=128):
    return quant.float() * scale.repeat_interleave(block, -2).repeat_interleave(block, -1)


def merge_pair_log2(torch, va, sa, vb, sb):
    # cascade.cuh MergeStateKernel: LSE is logsumexp with base 2.
    s_max = torch.maximum(sa, sb)
    a = torch.exp2(sa - s_max)
    b = torch.exp2(sb - s_max)
    denom = a + b
    s_ref = torch.log2(denom) + s_max
    v_ref = (a / denom)[..., None] * va.float() + (b / denom)[..., None] * vb.float()
    return v_ref, s_ref


def merge_many_log2(torch, values, lse):
    s_max = lse.max(dim=1, keepdim=True).values
    weights = torch.exp2(lse - s_max)
    denom = weights.sum(dim=1)
    s_ref = torch.log2(denom) + s_max.squeeze(1)
    v_ref = (weights.unsqueeze(-1) * values.float()).sum(dim=1) / denom.unsqueeze(-1)
    return v_ref, s_ref


MXFP4_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def dequant_mxfp4(torch, packed, scale_e8m0):
    lut = torch.tensor(MXFP4_LUT, dtype=torch.float32, device=packed.device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = lut[torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1).long()]
    scale = torch.exp2(scale_e8m0.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)
    return values * scale


def dequant_int4(torch, packed, scale, group_size=128):
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1).float()
    return unpacked * scale.float().repeat_interleave(group_size, dim=-1)


def interleave_sm90_scales(torch, flashinfer, scales, group_size=32, k_dim=None):
    # 0.6.17 public helper accepts uint8 E8M0 only (MXFP4 / core.py:800-849).
    # INT4 bf16 scales use TRT-LLM SM90 mixed-GEMM K-alignment: factor 4/2/1
    # when K is a multiple of 512/256, not 128/group_size (that is MXFP4-only).
    if scales.dtype == torch.uint8:
        return flashinfer.fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(
            scales, group_size=group_size
        )
    experts, rows, groups = scales.shape
    if k_dim is None:
        k_dim = groups * group_size
    factor = 4 if k_dim % 512 == 0 else (2 if k_dim % 256 == 0 else 1)
    folded = (
        scales.reshape(experts, rows, groups // factor, factor)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return folded.reshape(experts, groups // factor, rows * factor)


def per_token_group_dequant(torch, x, group=128):
    # DeepSeek FP8 blockscale MoE quantizes BF16 activations per 128-col group.
    tokens, hidden = x.shape
    view = x.float().reshape(tokens, hidden // group, group)
    amax = view.abs().amax(dim=-1).clamp(min=1e-4)
    scale = amax / 448.0
    quant = (view / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (quant.float() * scale.unsqueeze(-1)).reshape(tokens, hidden)


def custom_mask_matrix(torch, seq_q: int, seq_k: int):
    mask = torch.zeros(seq_q, seq_k, dtype=torch.bool, device="cuda")
    mask[:, : max(1, seq_k // 2)] = True
    diag = min(seq_q, seq_k)
    mask[torch.arange(diag, device="cuda"), torch.arange(seq_k - diag, seq_k, device="cuda")] = True
    return mask


class Prepared:
    def __init__(self, run: Callable[[], object], verify: Callable[[object], dict]):
        self.run = run
        self.verify = verify


def build_attention(torch, flashinfer, case: dict) -> Prepared:
    op = case["operation"]
    value_dtype = dtype(torch, case["dtype"])
    layout = case.get("kv_layout", "NHD")
    sm_scale = None

    if op == "single_prefill":
        q = random_tensor(torch, (case["seqlen_q"], case["heads_q"], case["head_dim"]), value_dtype)
        k_nhd = random_tensor(torch, (case["seqlen_k"], case["heads_kv"], case["head_dim"]), value_dtype)
        v_nhd = random_tensor(torch, k_nhd.shape, value_dtype)
        # FA3 SM90 host (single_prefill_sm90.cu) always reads kv_len=k.size(0)
        # and num_kv_heads=k.size(1). HND tensors make group_size=qo_heads/seq=0
        # and then SIGFPE in the kernel. Paged FA3 HND is a different path.
        run_layout = "NHD" if (layout == "HND" and case.get("backend") == "fa3") else layout
        k = k_nhd if run_layout == "NHD" else k_nhd.transpose(0, 1).contiguous()
        v = v_nhd if run_layout == "NHD" else v_nhd.transpose(0, 1).contiguous()
        mask = custom_mask_matrix(torch, case["seqlen_q"], case["seqlen_k"]) if case.get("custom_mask") else None
        kwargs = {
            "causal": case["causal"] and mask is None,
            "kv_layout": run_layout,
            "pos_encoding_mode": case["pos_encoding_mode"],
            "window_left": case["window_left"],
            "logits_soft_cap": case["logits_soft_cap"],
            "backend": case["backend"],
        }
        if mask is not None:
            kwargs["custom_mask"] = mask
        if case["dtype"].startswith("fp8"):
            kwargs["o_dtype"] = torch.bfloat16
            kwargs["scale_q"] = torch.ones(case["heads_q"], device="cuda", dtype=torch.float32)
            kwargs["scale_k"] = torch.ones(case["heads_kv"], device="cuda", dtype=torch.float32)
            kwargs["scale_v"] = torch.ones(case["heads_kv"], device="cuda", dtype=torch.float32)
        q_ref, k_ref = q.float(), k_nhd.float()
        if case["pos_encoding_mode"] == "ROPE_LLAMA":
            q_ref, k_ref = apply_llama_rope(torch, q_ref, k_ref)

        def run():
            return flashinfer.single_prefill_with_kv_cache(q, k, v, **kwargs)

        def verify(value):
            out = value[0] if isinstance(value, tuple) else value
            expected = sdpa(
                torch, q_ref, k_ref, v_nhd.float(),
                causal=case["causal"] and mask is None,
                window_left=case["window_left"],
                mask=mask,
                alibi=case["pos_encoding_mode"] == "ALIBI",
                soft_cap=case["logits_soft_cap"],
                sm_scale=sm_scale,
            )
            rtol = 0.2 if case["dtype"].startswith("fp8") else 0.08
            close(torch, out, expected, rtol=rtol, atol=0.15, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": "torch_sdpa_reference"}

        return Prepared(run, verify)

    if op == "batch_prefill_ragged":
        q_lens = lengths(case["batch"], case["seqlen_q"], case.get("unequal_seqlens", False))
        k_lens = lengths(case["batch"], case["seqlen_k"], case.get("unequal_seqlens", False))
        q = random_tensor(torch, (sum(q_lens), case["heads_q"], case["head_dim"]), value_dtype)
        k_nhd = random_tensor(torch, (sum(k_lens), case["heads_kv"], case["head_dim"]), value_dtype)
        v_nhd = random_tensor(torch, k_nhd.shape, value_dtype)
        # Same FA3 SM90 ragged host bug as single_prefill: k.size(1) is heads only in NHD.
        run_layout = "NHD" if (layout == "HND" and case.get("backend") == "fa3") else layout
        k = k_nhd if run_layout == "NHD" else k_nhd.transpose(0, 1).contiguous()
        v = v_nhd if run_layout == "NHD" else v_nhd.transpose(0, 1).contiguous()
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, kv_layout=run_layout, backend=case["backend"]
        )
        qo_indptr = cumsum_indptr(torch, q_lens)
        kv_indptr = cumsum_indptr(torch, k_lens)
        mask = None
        if case.get("custom_mask"):
            parts = [custom_mask_matrix(torch, q_len, k_len).reshape(-1) for q_len, k_len in zip(q_lens, k_lens)]
            mask = torch.cat(parts)
        wrapper.plan(
            qo_indptr, kv_indptr, case["heads_q"], case["heads_kv"], case["head_dim"],
            causal=case["causal"] and mask is None,
            pos_encoding_mode=case.get("pos_encoding_mode", "NONE"),
            q_data_type=value_dtype, kv_data_type=value_dtype,
            custom_mask=mask,
            disable_split_kv=case.get("disable_split_kv", False),
        )

        def run():
            return wrapper.run(q, k, v)

        def verify(value):
            out = value[0] if isinstance(value, tuple) else value
            pieces = []
            for index, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
                qs, qe = int(qo_indptr[index]), int(qo_indptr[index + 1])
                ks, ke = int(kv_indptr[index]), int(kv_indptr[index + 1])
                q_ref, k_ref = q[qs:qe].float(), k_nhd[ks:ke].float()
                if case.get("pos_encoding_mode") == "ROPE_LLAMA":
                    q_ref, k_ref = apply_llama_rope(torch, q_ref, k_ref)
                local_mask = None
                if mask is not None:
                    local_mask = custom_mask_matrix(torch, q_len, k_len)
                pieces.append(sdpa(
                    torch, q_ref, k_ref, v_nhd[ks:ke].float(),
                    causal=case["causal"] and mask is None,
                    window_left=-1, mask=local_mask,
                    alibi=case.get("pos_encoding_mode") == "ALIBI",
                    soft_cap=None, sm_scale=None,
                ))
            expected = torch.cat(pieces, dim=0)
            close(torch, out, expected, rtol=0.08, atol=0.15, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": "torch_sdpa_ragged_reference"}

        return Prepared(run, verify)

    if op == "batch_prefill_paged":
        q_lens = lengths(case["batch"], case["seqlen_q"], case.get("unequal_seqlens", False))
        k_lens = lengths(case["batch"], case["seqlen_k"], case.get("unequal_seqlens", True) if case.get("unequal_seqlens") else False)
        if not case.get("unequal_seqlens", False):
            k_lens = [case["seqlen_k"]] * case["batch"]
        pool, indices, kv_indptr, last = page_plan(
            torch, k_lens, case["page_size"], case.get("scattered_pages", False)
        )
        q = random_tensor(torch, (sum(q_lens), case["heads_q"], case["head_dim"]), value_dtype)
        k = random_tensor(torch, paged_kv_shape(layout, pool, case["page_size"], case["heads_kv"], case["head_dim"], False), value_dtype)
        v = random_tensor(torch, k.shape, value_dtype)
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout=layout, backend=case["backend"]
        )
        qo_indptr = cumsum_indptr(torch, q_lens)
        mask = None
        if case.get("custom_mask"):
            parts = [custom_mask_matrix(torch, q_len, k_len).reshape(-1) for q_len, k_len in zip(q_lens, k_lens)]
            mask = torch.cat(parts)
        wrapper.plan(
            qo_indptr, kv_indptr, indices, last, case["heads_q"], case["heads_kv"],
            case["head_dim"], case["page_size"], causal=case["causal"] and mask is None,
            pos_encoding_mode=case.get("pos_encoding_mode", "NONE"),
            q_data_type=value_dtype, kv_data_type=value_dtype, custom_mask=mask,
            disable_split_kv=case.get("disable_split_kv", False),
        )

        def run():
            return wrapper.run(q, (k, v))

        def verify(value):
            out = value[0] if isinstance(value, tuple) else value
            k_parts, v_parts = [], []
            batch = case["batch"]
            for request in range(batch):
                start, end = int(kv_indptr[request]), int(kv_indptr[request + 1])
                page_ids = indices[start:end].tolist()
                k_rows, v_rows = [], []
                for offset, page in enumerate(page_ids):
                    valid = case["page_size"] if offset + 1 < len(page_ids) else int(last[request].item())
                    if layout == "HND":
                        k_rows.append(k[page, :, :valid, :].transpose(0, 1))
                        v_rows.append(v[page, :, :valid, :].transpose(0, 1))
                    else:
                        k_rows.append(k[page, :valid])
                        v_rows.append(v[page, :valid])
                k_parts.append(torch.cat(k_rows, dim=0))
                v_parts.append(torch.cat(v_rows, dim=0))
            pieces = []
            for index, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
                qs, qe = int(qo_indptr[index]), int(qo_indptr[index + 1])
                q_ref, k_ref = q[qs:qe].float(), k_parts[index].float()
                if case.get("pos_encoding_mode") == "ROPE_LLAMA":
                    q_ref, k_ref = apply_llama_rope(torch, q_ref, k_ref)
                local_mask = custom_mask_matrix(torch, q_len, k_len) if mask is not None else None
                pieces.append(sdpa(
                    torch, q_ref, k_ref, v_parts[index].float(),
                    causal=case["causal"] and mask is None,
                    window_left=-1, mask=local_mask,
                    alibi=case.get("pos_encoding_mode") == "ALIBI",
                    soft_cap=None, sm_scale=None,
                ))
            close(torch, out, torch.cat(pieces, dim=0), rtol=0.08, atol=0.2, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": "torch_sdpa_paged_reference"}

        return Prepared(run, verify)

    if op == "single_decode":
        q = random_tensor(torch, (case["heads_q"], case["head_dim"]), value_dtype)
        k_nhd = random_tensor(torch, (case["kv_len"], case["heads_kv"], case["head_dim"]), value_dtype)
        v_nhd = random_tensor(torch, k_nhd.shape, value_dtype)
        k = k_nhd.transpose(0, 1).contiguous() if layout == "HND" else k_nhd
        v = v_nhd.transpose(0, 1).contiguous() if layout == "HND" else v_nhd

        def run():
            return flashinfer.single_decode_with_kv_cache(
                q, k, v, kv_layout=layout,
                use_tensor_cores=case["use_tensor_cores"],
                pos_encoding_mode=case["pos_encoding_mode"],
            )

        def verify(value):
            out = value[0] if isinstance(value, tuple) else value
            q_ref = q.float().unsqueeze(0)
            k_ref = k_nhd.float()
            if case["pos_encoding_mode"] == "ROPE_LLAMA":
                q_ref, k_ref = apply_llama_rope(torch, q_ref, k_ref)
            expected = sdpa(
                torch, q_ref, k_ref, v_nhd.float(),
                causal=False, window_left=-1, mask=None, alibi=False, soft_cap=None, sm_scale=None,
            ).squeeze(0)
            close(torch, out, expected, rtol=0.08, atol=0.15, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": "torch_sdpa_decode_reference"}

        return Prepared(run, verify)

    if op == "batch_decode_paged":
        k_lens = lengths(case["batch"], case["kv_len"], case.get("unequal_seqlens", False))
        pool, indices, indptr, last = page_plan(
            torch, k_lens, case["page_size"], case.get("scattered_pages", False)
        )
        kv_is_fp8 = case["dtype"].startswith("fp8")
        backend = case.get("backend")
        # FA3 SM90 wgmma has no mixed bf16-Q / e4m3-KV. Public
        # is_fa3_backend_supported() requires FP8 query when KV is FP8.
        # FA2 tensor-core decode keeps bf16 Q.
        if kv_is_fp8 and backend == "fa3":
            q_dtype = value_dtype
            o_dtype = torch.bfloat16
        elif kv_is_fp8:
            q_dtype = torch.bfloat16
            o_dtype = torch.bfloat16
        else:
            q_dtype = value_dtype
            o_dtype = q_dtype
        q = random_tensor(torch, (case["batch"], case["heads_q"], case["head_dim"]), q_dtype)
        kv = random_tensor(
            torch,
            paged_kv_shape(layout, pool, case["page_size"], case["heads_kv"], case["head_dim"], True),
            value_dtype,
        )
        workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        # SIMT decode.cuh DISPATCH_GQA_GROUP_SIZE instantiates 1,2,3,4,8 only.
        group = case["heads_q"] // case["heads_kv"]
        use_tc = bool(case["use_tensor_cores"]) or group not in {1, 2, 3, 4, 8}
        wrapper_kwargs = {
            "kv_layout": layout,
            "use_tensor_cores": use_tc,
        }
        if backend and backend != "auto":
            wrapper_kwargs["backend"] = backend
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, **wrapper_kwargs)
        wrapper.plan(
            indptr, indices, last, case["heads_q"], case["heads_kv"],
            case["head_dim"], case["page_size"],
            pos_encoding_mode=case.get("pos_encoding_mode", "NONE"),
            kv_data_type=value_dtype, q_data_type=q_dtype, o_data_type=o_dtype,
            disable_split_kv=case.get("disable_split_kv", False),
        )

        def run():
            return wrapper.run(q, kv)

        def verify(value):
            out = value[0] if isinstance(value, tuple) else value
            pieces = []
            for request in range(case["batch"]):
                start, end = int(indptr[request]), int(indptr[request + 1])
                page_ids = indices[start:end].tolist()
                k_rows, v_rows = [], []
                for offset, page in enumerate(page_ids):
                    valid = case["page_size"] if offset + 1 < len(page_ids) else int(last[request].item())
                    if layout == "HND":
                        k_rows.append(kv[page, 0, :, :valid, :].transpose(0, 1))
                        v_rows.append(kv[page, 1, :, :valid, :].transpose(0, 1))
                    else:
                        k_rows.append(kv[page, 0, :valid])
                        v_rows.append(kv[page, 1, :valid])
                k_ref = torch.cat(k_rows, dim=0).float()
                v_ref = torch.cat(v_rows, dim=0).float()
                q_ref = q[request].float().unsqueeze(0)
                if case.get("pos_encoding_mode") == "ROPE_LLAMA":
                    q_ref, k_ref = apply_llama_rope(torch, q_ref, k_ref)
                pieces.append(sdpa(
                    torch, q_ref, k_ref, v_ref,
                    causal=False, window_left=-1, mask=None, alibi=False, soft_cap=None, sm_scale=None,
                ).squeeze(0))
            rtol = 0.25 if case["dtype"].startswith("fp8") else 0.08
            close(torch, out, torch.stack(pieces, dim=0), rtol=rtol, atol=0.2, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": "torch_sdpa_batch_decode_reference"}

        return Prepared(run, verify)

    raise ValueError(op)


def build_mla(torch, flashinfer, case: dict) -> Prepared:
    kv_dtype = dtype(torch, case["dtype"])
    q_dtype = torch.bfloat16 if case["dtype"].startswith("fp8") else kv_dtype
    kv_lens = lengths(case["batch"], case["kv_len"], case.get("unequal_seqlens", False))
    q_lens = [case["q_len"]] * case["batch"]
    pool, indices, kv_indptr, last = page_plan(
        torch, kv_lens, case["page_size"], case.get("scattered_pages", False)
    )
    # last is used only to size pages; MLA wrapper takes kv_len_arr directly.
    del last
    q_nope = random_tensor(torch, (sum(q_lens), case["num_heads"], case["head_dim_ckv"]), q_dtype, 0.2)
    q_pe = random_tensor(torch, (sum(q_lens), case["num_heads"], case["head_dim_kpe"]), q_dtype, 0.2)
    ckv = random_tensor(torch, (pool, case["page_size"], case["head_dim_ckv"]), kv_dtype, 0.2)
    kpe = random_tensor(torch, (pool, case["page_size"], case["head_dim_kpe"]), kv_dtype, 0.2)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.BatchMLAPagedAttentionWrapper(workspace, backend=case["backend"])
    qo_indptr = cumsum_indptr(torch, q_lens)
    kv_len_arr = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / math.sqrt(128 + 64)
    wrapper.plan(
        qo_indptr, kv_indptr, indices, kv_len_arr,
        case["num_heads"], case["head_dim_ckv"], case["head_dim_kpe"],
        case["page_size"], case["causal"], sm_scale, q_dtype, kv_dtype,
    )
    run_kwargs = {}
    if case["dtype"].startswith("fp8"):
        run_kwargs["ckv_scale"] = 1.0
        run_kwargs["kpe_scale"] = 1.0

    def run():
        return wrapper.run(q_nope, q_pe, ckv, kpe, **run_kwargs)

    def verify(value):
        out = value[0] if isinstance(value, tuple) else value
        pieces = []
        for request in range(case["batch"]):
            qs, qe = int(qo_indptr[request]), int(qo_indptr[request + 1])
            start, end = int(kv_indptr[request]), int(kv_indptr[request + 1])
            page_ids = indices[start:end].tolist()
            remaining = kv_lens[request]
            ckv_rows, kpe_rows = [], []
            for page in page_ids:
                take = min(case["page_size"], remaining)
                ckv_rows.append(ckv[page, :take].float())
                kpe_rows.append(kpe[page, :take].float())
                remaining -= take
            ckv_tok = torch.cat(ckv_rows, dim=0)
            kpe_tok = torch.cat(kpe_rows, dim=0)
            qn = q_nope[qs:qe].float()
            qp = q_pe[qs:qe].float()
            logits = torch.einsum("qhd,kd->qhk", qn, ckv_tok) + torch.einsum("qhe,ke->qhk", qp, kpe_tok)
            logits = logits * sm_scale
            if case["causal"] and qn.shape[0] > 1:
                q_idx = torch.arange(qn.shape[0], device=qn.device)[:, None]
                k_idx = torch.arange(ckv_tok.shape[0], device=qn.device)[None, :]
                causal_mask = k_idx > (ckv_tok.shape[0] - qn.shape[0]) + q_idx
                logits = logits.masked_fill(causal_mask[:, None, :], float("-inf"))
            attn = torch.softmax(logits, dim=-1)
            pieces.append(torch.einsum("qhk,kd->qhd", attn, ckv_tok))
        expected = torch.cat(pieces, dim=0)
        rtol = 0.25 if case["dtype"].startswith("fp8") else 0.12
        close(torch, out, expected, rtol=rtol, atol=0.25, name=case["case_id"])
        return {"status": "pass", "finite": True, "method": "torch_mla_absorbed_reference"}

    return Prepared(run, verify)


def moe_torch(torch, x, w1, w2, expert_ids, expert_scales, activation: str):
    tokens = x.shape[0]
    hidden = x.shape[1]
    out = torch.zeros(tokens, hidden, device=x.device, dtype=torch.float32)
    gated = activation == "swiglu"
    intermediate = w1.shape[1] // (2 if gated else 1)
    x_f, w1_f, w2_f = x.float(), w1.float(), w2.float()
    for token in range(tokens):
        acc = torch.zeros(hidden, device=x.device, dtype=torch.float32)
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route].item())
            if gated:
                up = x_f[token] @ w1_f[expert, :intermediate].T
                gate = torch.nn.functional.silu(x_f[token] @ w1_f[expert, intermediate:].T)
                hidden_act = gate * up
            else:
                hidden_act = torch.relu(x_f[token] @ w1_f[expert].T) ** 2
            acc = acc + expert_scales[token, route].float() * (hidden_act @ w2_f[expert].T)
        out[token] = acc
    return out


def moe_torch_w4a8(torch, x, w1, w2, expert_ids, expert_scales, activation, input_scale):
    tokens, hidden = x.shape
    out = torch.zeros(tokens, hidden, device=x.device, dtype=torch.float32)
    gated = activation == "swiglu"
    intermediate = w1.shape[1] // (2 if gated else 1)
    x_f, w1_f, w2_f = x.float(), w1.float(), w2.float()
    scale = float(input_scale)
    fp8_max = 448.0
    for token in range(tokens):
        acc = torch.zeros(hidden, device=x.device, dtype=torch.float32)
        xq = (x_f[token] / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).float() * scale
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route].item())
            if gated:
                up = xq @ w1_f[expert, :intermediate].T
                gate = torch.nn.functional.silu(xq @ w1_f[expert, intermediate:].T)
                hidden_act = gate * up
            else:
                hidden_act = torch.relu(xq @ w1_f[expert].T) ** 2
            hq = (hidden_act / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).float() * scale
            acc = acc + expert_scales[token, route].float() * (hq @ w2_f[expert].T)
        out[token] = acc
    return out


def build_xqa(torch, flashinfer, case):
    layout = case["kv_layout"]
    page_size = case["page_size"]
    head_dim = case["head_dim"]
    batch = case["batch"]
    q_dtype = dtype(torch, case["dtype"])
    kv_dtype_name = case.get("kv_dtype", case["dtype"])
    kv_is_fp8 = kv_dtype_name.startswith("fp8")
    kv_lens = lengths(batch, case["kv_len"], case.get("unequal_seqlens", False))
    pool, indices, indptr, last = page_plan(
        torch, kv_lens, page_size, case.get("scattered_pages", False)
    )
    q = random_tensor(torch, (batch, case["heads_q"], head_dim), q_dtype)
    kv_shape = paged_kv_shape(layout, pool, page_size, case["heads_kv"], head_dim, True)
    kv_hi = random_tensor(torch, kv_shape, torch.bfloat16)
    if kv_is_fp8:
        amax = kv_hi.float().abs().max().clamp(min=1e-4)
        kv_scale = float(amax / 448.0)
        kv = (kv_hi.float() / kv_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    else:
        kv_scale = 1.0
        kv = kv_hi.to(dtype(torch, kv_dtype_name))
    pages_per = [(length + page_size - 1) // page_size for length in kv_lens]
    max_pages = max(pages_per)
    block_tables = torch.zeros(batch, max_pages, dtype=torch.int32, device="cuda")
    for request, n_pages in enumerate(pages_per):
        start = int(indptr[request])
        block_tables[request, :n_pages] = indices[start : start + n_pages]
    seq_lens = torch.tensor(kv_lens, dtype=torch.uint32, device="cuda")
    # decode.py:3673 splits this buffer: first 8MB semaphore, remainder scratch.
    # Same 128MB recommendation as BatchDecodeWithPagedKVCacheWrapper; must be zero
    # on first use.
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    bmm1_scale = head_dim ** -0.5
    decode = flashinfer.decode.xqa_batch_decode_with_kv_cache

    def run():
        return decode(
            query=q,
            kv_cache=kv,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max(kv_lens),
            bmm1_scale=bmm1_scale * kv_scale,
            bmm2_scale=kv_scale,
            kv_layout=layout,
        )

    def verify(value):
        out = value[0] if isinstance(value, tuple) else value
        pieces = []
        for request in range(batch):
            start, end = int(indptr[request]), int(indptr[request + 1])
            page_ids = indices[start:end].tolist()
            k_rows, v_rows = [], []
            for offset, page in enumerate(page_ids):
                valid = page_size if offset + 1 < len(page_ids) else int(last[request].item())
                if layout == "HND":
                    k_part = kv[page, 0, :, :valid, :].transpose(0, 1)
                    v_part = kv[page, 1, :, :valid, :].transpose(0, 1)
                else:
                    k_part = kv[page, 0, :valid]
                    v_part = kv[page, 1, :valid]
                k_rows.append(k_part.float() * kv_scale)
                v_rows.append(v_part.float() * kv_scale)
            # SM90 F16QToF8Converter::store casts Q directly to the FP8 cache type.
            q_ref = q[request].to(torch.float8_e4m3fn) if kv_is_fp8 else q[request]
            pieces.append(
                sdpa(
                    torch,
                    q_ref.float().unsqueeze(0),
                    torch.cat(k_rows, dim=0),
                    torch.cat(v_rows, dim=0),
                    causal=False,
                    window_left=-1,
                    mask=None,
                    alibi=False,
                    soft_cap=None,
                    sm_scale=bmm1_scale,
                ).squeeze(0)
            )
        expected = torch.stack(pieces, dim=0).to(out.dtype)
        rtol = 0.25 if kv_is_fp8 else 0.08
        atol = 0.002
        close(
            torch,
            out,
            expected,
            rtol=rtol,
            atol=atol,
            name=case["case_id"],
        )
        return {
            "status": "pass",
            "finite": True,
            "method": "torch_sdpa_xqa_dequant_reference",
        }

    return Prepared(run, verify)


def build_operation(torch, flashinfer, case: dict) -> Prepared:
    op = case["operation"]
    if op in {
        "single_prefill", "batch_prefill_ragged", "batch_prefill_paged",
        "single_decode", "batch_decode_paged",
    }:
        return build_attention(torch, flashinfer, case)
    if op == "batch_mla_paged":
        return build_mla(torch, flashinfer, case)
    if op == "xqa_batch_decode":
        return build_xqa(torch, flashinfer, case)

    if op == "segment_gemm":
        value_dtype = dtype(torch, case["dtype"])
        groups = case["groups"]
        base = case["tokens_per_group"]
        seg_lens = lengths(groups, base, case.get("unequal_segments", False))
        total = sum(seg_lens)
        x = random_tensor(torch, (total, case["d_in"]), value_dtype)
        weights = random_tensor(torch, (groups, case["d_out"], case["d_in"]), value_dtype)
        if case.get("weight_indices"):
            weight_indices = torch.arange(groups, dtype=torch.int64, device="cuda")
            weight_indices = (weight_indices * 5 + 3) % groups
        else:
            weight_indices = None
        out = torch.empty(total, case["d_out"], dtype=torch.bfloat16 if case["dtype"].startswith("fp8") else value_dtype, device="cuda")
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        wrapper = flashinfer.SegmentGEMMWrapper(workspace, backend=case["backend"])
        indptr = torch.tensor([0, *accumulate(seg_lens)], dtype=torch.int64, device="cuda")

        def run():
            return wrapper.run(
                x, weights, groups, case["weight_column_major"], out=out,
                seg_indptr=indptr, weight_indices=weight_indices,
            )

        def verify(value):
            pieces = []
            xf, wf = x.float(), weights.float()
            for group in range(groups):
                sl = slice(int(indptr[group]), int(indptr[group + 1]))
                expert = int(weight_indices[group].item()) if weight_indices is not None else group
                pieces.append(xf[sl] @ wf[expert].T)
            close(torch, value, torch.cat(pieces, dim=0), rtol=0.15 if case["dtype"].startswith("fp8") else 0.03, atol=0.1, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": "torch_matmul_segment_reference"}

        return Prepared(run, verify)

    if op == "cutlass_fused_moe":
        from flashinfer.fused_moe import ActivationType, cutlass_fused_moe_workspace_size
        tokens, hidden, intermediate = case["num_tokens"], case["hidden"], case["intermediate"]
        experts, top_k = case["num_experts"], case["top_k"]
        gated = case["activation"] == "swiglu"
        act = ActivationType.Swiglu if gated else ActivationType.Relu2
        w1_rows = (2 if gated else 1) * intermediate
        x_bf16 = random_tensor(torch, (tokens, hidden), torch.bfloat16, 0.1)
        w1 = random_tensor(torch, (experts, w1_rows, hidden), torch.bfloat16, 0.1)
        w2 = random_tensor(torch, (experts, hidden, intermediate), torch.bfloat16, 0.1)
        logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
        scales, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
        scales = scales / scales.sum(dim=-1, keepdim=True)
        ids = ids.to(torch.int32)
        quant = case["quant"]
        w4 = quant in {"w4a16_mxfp4", "w4a8_int4"}
        packed_w4 = quant == "w4a8_int4"
        w1_ref, w2_ref = w1, w2
        x_ref = x_bf16
        w4a8_s_in = None
        if quant == "none":
            x_in, w1_in, w2_in = x_bf16, w1, w2
            quant_scales = None
            x_dtype = torch.bfloat16
            w_dtype = torch.bfloat16
        elif quant == "fp8_per_tensor":
            def quant_tensor(matrix):
                amax = matrix.float().abs().max().clamp(min=1e-4)
                scale = amax / 448.0
                return (matrix.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale.to(torch.float32)
            x_in, x_scale = quant_tensor(x_bf16)
            w1_in = torch.empty_like(w1, dtype=torch.float8_e4m3fn)
            w2_in = torch.empty_like(w2, dtype=torch.float8_e4m3fn)
            w1_scales = torch.empty(experts, device="cuda", dtype=torch.float32)
            w2_scales = torch.empty(experts, device="cuda", dtype=torch.float32)
            for expert in range(experts):
                w1_in[expert], w1_scales[expert] = quant_tensor(w1[expert])
                w2_in[expert], w2_scales[expert] = quant_tensor(w2[expert])
            one = torch.ones((), device="cuda", dtype=torch.float32)
            quant_scales = [
                (w1_scales * x_scale).float(),
                one,
                w2_scales.float(),
                x_scale.float(),
            ]
            x_dtype = torch.float8_e4m3fn
            w_dtype = torch.float8_e4m3fn
            x_ref = x_in.float() * x_scale
            w1_ref = w1_in.float() * w1_scales[:, None, None]
            w2_ref = w2_in.float() * w2_scales[:, None, None]
        elif quant == "fp8_blockscale":
            w1_in, w1_scale = fp8_block_quant(torch, w1)
            w2_in, w2_scale = fp8_block_quant(torch, w2)
            x_in = x_bf16
            quant_scales = [w1_scale.float().contiguous(), w2_scale.float().contiguous()]
            x_dtype = torch.bfloat16
            w_dtype = torch.float8_e4m3fn
            x_ref = per_token_group_dequant(torch, x_bf16)
            w1_ref = dequant_fp8_block(w1_in, w1_scale)
            w2_ref = dequant_fp8_block(w2_in, w2_scale)
        elif quant == "w4a16_mxfp4":
            # tests/moe/test_trtllm_cutlass_fused_moe.py::test_moe_bf16_mxfp4
            # core.py:800-907 SM90 mixed-input MXFP4 W4A16.
            w1_pack = torch.randint(0, 256, (experts, w1_rows, hidden // 2), device="cuda", dtype=torch.uint8)
            w2_pack = torch.randint(0, 256, (experts, hidden, intermediate // 2), device="cuda", dtype=torch.uint8)
            w1_sf = torch.randint(118, 123, (experts, w1_rows, hidden // 32), device="cuda", dtype=torch.uint8)
            w2_sf = torch.randint(118, 123, (experts, hidden, intermediate // 32), device="cuda", dtype=torch.uint8)
            w1_in = flashinfer.fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w1_pack, "fp4")
            w2_in = flashinfer.fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w2_pack, "fp4")
            w1_sf_il = interleave_sm90_scales(torch, flashinfer, w1_sf, 32)
            w2_sf_il = interleave_sm90_scales(torch, flashinfer, w2_sf, 32)
            quant_scales = [w1_sf_il.view(torch.int32), w2_sf_il.view(torch.int32)]
            x_in = x_bf16
            x_dtype = torch.bfloat16
            w_dtype = torch.uint8
            w1_ref = dequant_mxfp4(torch, w1_pack, w1_sf).to(torch.bfloat16)
            w2_ref = dequant_mxfp4(torch, w2_pack, w2_sf).to(torch.bfloat16)
        elif quant == "w4a8_int4":
            # tests/moe/test_trtllm_cutlass_fused_moe.py::test_moe_w4a8
            # uint8 packed INT4 + use_packed_weights + use_w4_group_scaling.
            group = 128
            w1_pack = torch.randint(0, 256, (experts, w1_rows, hidden // 2), device="cuda", dtype=torch.uint8)
            w2_pack = torch.randint(0, 256, (experts, hidden, intermediate // 2), device="cuda", dtype=torch.uint8)
            w1_gs = torch.randn(experts, w1_rows, hidden // group, device="cuda", dtype=torch.bfloat16) * 0.005
            w2_gs = torch.randn(experts, hidden, intermediate // group, device="cuda", dtype=torch.bfloat16) * 0.005
            w1_in = flashinfer.fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w1_pack, "int4")
            w2_in = flashinfer.fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w2_pack, "int4")
            w1_gs_il = interleave_sm90_scales(torch, flashinfer, w1_gs, group, k_dim=hidden)
            w2_gs_il = interleave_sm90_scales(torch, flashinfer, w2_gs, group, k_dim=intermediate)
            empty = torch.empty(0, device="cuda", dtype=torch.bfloat16)
            w4a8_s_in = (x_bf16.float().abs().max().clamp(min=1e-4) / 448.0).to(torch.float32)
            inv = float(1.0 / w4a8_s_in)
            quant_scales = [
                w1_gs_il,
                w2_gs_il,
                torch.full((hidden,), inv, device="cuda", dtype=torch.bfloat16),
                torch.full((intermediate,), inv, device="cuda", dtype=torch.bfloat16),
                empty,
                empty,
                torch.full((experts,), float(w4a8_s_in), device="cuda", dtype=torch.float32),
                torch.full((experts,), float(w4a8_s_in), device="cuda", dtype=torch.float32),
            ]
            x_in = x_bf16
            x_dtype = torch.bfloat16
            w_dtype = torch.uint8
            w1_ref = dequant_int4(torch, w1_pack, w1_gs, group).to(torch.bfloat16)
            w2_ref = dequant_int4(torch, w2_pack, w2_gs, group).to(torch.bfloat16)
        else:
            raise ValueError(f"unsupported fused MoE quant {quant}")
        workspace_bytes = cutlass_fused_moe_workspace_size(
            tokens, hidden, intermediate, experts, top_k,
            x_dtype=x_dtype, weight_dtype=w_dtype, output_dtype=torch.bfloat16,
            activation_type=act, use_deepseek_fp8_block_scale=(quant == "fp8_blockscale"),
            use_w4_group_scaling=w4, use_packed_weights=packed_w4,
            use_fused_finalize=False, device=x_in.device,
        )
        workspace = torch.empty(int(workspace_bytes), dtype=torch.uint8, device="cuda")
        output = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda")

        def run():
            return flashinfer.cutlass_fused_moe(
                x_in, ids, scales, w1_in, w2_in, torch.bfloat16,
                quant_scales=quant_scales, output=output,
                tp_size=1, tp_rank=0, ep_size=1, ep_rank=0,
                use_deepseek_fp8_block_scale=(quant == "fp8_blockscale"),
                use_w4_group_scaling=w4, use_packed_weights=packed_w4,
                activation_type=act, use_fused_finalize=False, enable_pdl=False,
                workspace_buffer=workspace,
            )

        def verify(value):
            result = value[0] if isinstance(value, (tuple, list)) else value
            if quant == "w4a8_int4":
                expected = moe_torch_w4a8(
                    torch, x_ref, w1_ref, w2_ref, ids, scales, case["activation"], w4a8_s_in
                )
                method = "torch_w4a8_fp8act_dequant_moe_reference"
            else:
                expected = moe_torch(torch, x_ref, w1_ref, w2_ref, ids, scales, case["activation"])
                method = {
                    "w4a16_mxfp4": "torch_mxfp4_dequant_moe_reference",
                    "fp8_per_tensor": "torch_fp8_dequant_moe_reference",
                    "fp8_blockscale": "torch_fp8_blockscale_dequant_moe_reference",
                }.get(quant, "torch_per_expert_swiglu_reference")
            rtol = 0.08 if quant == "none" else 0.25
            atol = (0.01 if quant == "none" else 0.03) * expected.float().abs().max().item()
            close(torch, result, expected, rtol=rtol, atol=atol, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": method}

        return Prepared(run, verify)

    if op == "fp8_blockscale_gemm":
        from flashinfer.gemm import fp8_blockscale_gemm_sm90
        from flashinfer.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
        m, n, k = case["m"], case["n"], case["k"]
        a_bf16 = random_tensor(torch, (m, k), torch.bfloat16, 0.2)
        b_bf16 = random_tensor(torch, (n, k), torch.bfloat16, 0.2)
        if case["scale_mode"] == "internal":
            a_in, b_in, a_scale, b_scale = a_bf16, b_bf16, None, None
        elif case["scale_mode"] == "block_128":
            a_in, a_scale = a_bf16, None
            b_in, b_scale = per_block_cast_to_fp8(b_bf16)
        else:
            a_in, a_scale = per_token_cast_to_fp8(a_bf16)
            b_in, b_scale = per_block_cast_to_fp8(b_bf16)
            # Native SM90 TMA loads activation scales in column-major order.
            a_scale = a_scale.T.contiguous().T
        a_ref = a_bf16.float() if a_scale is None else a_in.float() * a_scale.repeat_interleave(128, 1)
        b_ref = b_bf16.float() if b_scale is None else b_in.float() * b_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)

        def run():
            return fp8_blockscale_gemm_sm90(a_in, b_in, a_scale, b_scale)

        def verify(value):
            expected = a_ref @ b_ref.T
            close(torch, value, expected, rtol=0.2, atol=0.3, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": "torch_matmul_fp8_blockscale_reference"}

        return Prepared(run, verify)

    if op == "mamba_state_update":
        b, h, d, ds = case["batch"], case["heads"], case["dim"], case["dstate"]
        state0 = torch.zeros(b, h, d, ds, dtype=torch.float32, device="cuda")
        state = state0.clone()
        x = torch.full((b, h, d), 0.01, dtype=torch.bfloat16, device="cuda")
        dt_base = torch.full((b, h), 0.01, dtype=torch.float32, device="cuda")
        dt = dt_base.as_strided((b, h, d), (h, 1, 0))
        a_base = torch.full((h,), -1.0, dtype=torch.float32, device="cuda")
        a = a_base.as_strided((h, d, ds), (1, 0, 0))
        weights_b = torch.full((b, h, ds), 0.01, dtype=torch.bfloat16, device="cuda")
        weights_c = torch.full((b, h, ds), 0.01, dtype=torch.bfloat16, device="cuda")
        d_base = torch.full((h,), 0.1, dtype=torch.float32, device="cuda")
        skip = d_base.as_strided((h, d), (1, 0))

        def run():
            state.copy_(state0)
            return flashinfer.mamba.selective_state_update(
                state, x, dt, a, weights_b, weights_c, skip,
                dt_softplus=case["dt_softplus"], algorithm=case["algorithm"],
            )

        def verify(value):
            out = value[0] if isinstance(value, tuple) else value
            if not torch.isfinite(out.float()).all():
                raise AssertionError("mamba output non-finite")
            if float(state.float().abs().max()) == 0.0:
                raise AssertionError("mamba state was not updated")
            if float(out.float().abs().max()) == 0.0:
                raise AssertionError("mamba output is all zeros")
            return {"status": "pass", "finite": True, "method": "mamba_state_mutation_invariant"}

        return Prepared(run, verify)

    if op == "gdn_prefill":
        b, s, hq, hv, d = case["batch"], case["seqlen"], case["heads_q"], case["heads_v"], case["head_dim"]
        total = b * s
        q = random_tensor(torch, (total, hq, d), torch.bfloat16, 0.2)
        k = torch.nn.functional.normalize(
            torch.randn(total, hq, d, dtype=torch.float32, device="cuda"), dim=-1
        ).to(torch.bfloat16)
        v = random_tensor(torch, (total, hv, d), torch.bfloat16, 0.2)
        heads = max(hq, hv)
        g = torch.rand(total, heads, dtype=torch.float32, device="cuda")
        beta = torch.rand_like(g).sigmoid()
        cu = torch.arange(0, total + 1, s, dtype=torch.int64, device="cuda")

        def run():
            return flashinfer.chunk_gated_delta_rule(
                q, k, v, g, beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=True
            )

        def verify(value):
            out = value[0] if isinstance(value, tuple) else value
            if not torch.isfinite(out.float()).all() or float(out.float().abs().max()) == 0.0:
                raise AssertionError("gdn_prefill output empty or non-finite")
            flipped = flashinfer.chunk_gated_delta_rule(
                q, k, -v, g, beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=True
            )
            flipped_out = flipped[0] if isinstance(flipped, tuple) else flipped
            if torch.allclose(out.float(), flipped_out.float(), atol=1e-3, rtol=1e-3):
                raise AssertionError("gdn_prefill ignored value tensor")
            return {"status": "pass", "finite": True, "method": "gdn_prefill_value_sensitivity"}

        return Prepared(run, verify)

    if op == "gdn_decode":
        from flashinfer.gdn_decode import gated_delta_rule_decode
        b, hq, hv, d = case["batch"], case["heads_q"], case["heads_v"], case["head_dim"]
        q = random_tensor(torch, (b, 1, hq, d), torch.bfloat16)
        k = random_tensor(torch, (b, 1, hq, d), torch.bfloat16)
        v = random_tensor(torch, (b, 1, hv, d), torch.bfloat16)
        state0 = torch.zeros(b, hv, d, d, dtype=torch.float32, device="cuda")
        state = state0.clone()
        a_log = -torch.rand(hv, dtype=torch.float32, device="cuda")
        a = random_tensor(torch, (b, 1, hv), torch.bfloat16)
        dt_bias = torch.zeros(hv, dtype=torch.float32, device="cuda")
        beta = random_tensor(torch, (b, 1, hv), torch.bfloat16)

        def run():
            state.copy_(state0)
            return gated_delta_rule_decode(q, k, v, state, a_log, a, dt_bias, beta)

        def verify(value):
            out = value[0] if isinstance(value, tuple) else value
            if not torch.isfinite(out.float()).all() or float(out.float().abs().max()) == 0.0:
                raise AssertionError("gdn_decode output empty or non-finite")
            if float(state.float().abs().max()) == 0.0:
                raise AssertionError("gdn_decode state was not updated")
            return {"status": "pass", "finite": True, "method": "gdn_decode_state_mutation_invariant"}

        return Prepared(run, verify)

    if op in {"rmsnorm", "fused_add_rmsnorm"}:
        value_dtype = dtype(torch, case["dtype"])
        x0 = random_tensor(torch, (case["rows"], case["hidden"]), value_dtype)
        x = x0.clone()
        weight = torch.ones(case["hidden"], dtype=value_dtype, device="cuda")
        residual0 = random_tensor(torch, x.shape, value_dtype) if op == "fused_add_rmsnorm" else None
        residual = None if residual0 is None else residual0.clone()

        def rmsnorm_ref(tensor):
            tensor_f = tensor.float()
            return (tensor_f * torch.rsqrt(tensor_f.pow(2).mean(-1, keepdim=True) + 1e-6) * weight.float())

        def run():
            if op == "rmsnorm":
                return flashinfer.norm.rmsnorm(x0, weight, enable_pdl=case["enable_pdl"])
            x.copy_(x0)
            residual.copy_(residual0)
            flashinfer.norm.fused_add_rmsnorm(x, residual, weight, enable_pdl=case["enable_pdl"])
            return x

        def verify(value):
            if op == "rmsnorm":
                close(torch, value, rmsnorm_ref(x0), rtol=0.03, atol=0.02, name=case["case_id"])
            else:
                close(torch, value, rmsnorm_ref(x0.float() + residual0.float()), rtol=0.03, atol=0.02, name=case["case_id"])
                close(torch, residual, x0.float() + residual0.float(), rtol=0.03, atol=0.02, name=case["case_id"] + ".residual")
            return {"status": "pass", "finite": True, "method": "torch_rmsnorm_reference"}

        return Prepared(run, verify)

    if op == "rope":
        value_dtype = dtype(torch, case["dtype"])
        total = case["batch"] * case["seqlen"]
        q0 = random_tensor(torch, (total, case["heads_q"], case["head_dim"]), value_dtype)
        k0 = random_tensor(torch, (total, case["heads_kv"], case["head_dim"]), value_dtype)
        q, k = q0.clone(), k0.clone()
        indptr = torch.arange(0, total + 1, case["seqlen"], dtype=torch.int32, device="cuda")
        offsets = torch.arange(case["batch"], dtype=torch.int32, device="cuda")

        def run():
            q.copy_(q0)
            k.copy_(k0)
            return flashinfer.apply_rope(
                q, k, indptr, offsets, rotary_dim=case["rotary_dim"], interleave=case["interleave"]
            )

        def verify(value):
            q_out, k_out = value
            if torch.allclose(q_out.float(), q0.float(), atol=1e-4) and case["rotary_dim"] > 0:
                raise AssertionError("rope left query unchanged")
            if torch.allclose(k_out.float(), k0.float(), atol=1e-4) and case["rotary_dim"] > 0:
                raise AssertionError("rope left key unchanged")
            # Rotating twice with the same positions must not match a zeroed rotary slice.
            if float((q_out.float() - q0.float())[..., : case["rotary_dim"]].abs().max()) == 0.0:
                raise AssertionError("rope rotary dims were not written")
            return {"status": "pass", "finite": True, "method": "rope_mutation_invariant"}

        return Prepared(run, verify)

    if op == "append_paged_kv":
        value_dtype = dtype(torch, case["dtype"])
        b, length, page = case["batch"], case["append_len"], case["page_size"]
        kv_lens = [length] * b
        pool, indices, indptr, last = page_plan(torch, kv_lens, page, case.get("scattered_pages", False))
        key = random_tensor(torch, (b * length, case["heads_kv"], case["head_dim"]), value_dtype)
        value = random_tensor(torch, key.shape, value_dtype)
        batch_indices = torch.arange(b, dtype=torch.int32, device="cuda").repeat_interleave(length)
        positions = torch.arange(length, dtype=torch.int32, device="cuda").repeat(b)
        cache0 = torch.zeros(
            pool, 2, page, case["heads_kv"], case["head_dim"], dtype=value_dtype, device="cuda",
        )
        cache = cache0.clone()

        def run():
            cache.copy_(cache0)
            flashinfer.append_paged_kv_cache(
                key, value, batch_indices, positions, cache, indices, indptr, last
            )
            return cache

        def verify(value):
            written = value[indices[0], 0, : min(page, length)].float()
            source = key[: min(page, length)].float()
            if float(value.float().abs().max()) == 0.0:
                raise AssertionError("append_paged_kv left cache at zero")
            close(torch, written[: source.shape[0]], source[: written.shape[0]], rtol=0.01, atol=0.01, name=case["case_id"])
            return {"status": "pass", "finite": True, "method": "append_paged_kv_slot_reference"}

        return Prepared(run, verify)

    if op in {"sample", "top_k_sample", "top_p_sample"}:
        probs0 = torch.rand(case["batch"], case["vocab"], dtype=torch.float32, device="cuda")
        probs0 = probs0 / probs0.sum(dim=-1, keepdim=True)
        probs = probs0.clone()

        def sample_once():
            if op == "sample":
                return flashinfer.sampling.sampling_from_probs(probs, seed=17, offset=0)
            if op == "top_k_sample":
                return flashinfer.sampling.top_k_sampling_from_probs(
                    probs, case["top_k"], seed=17, offset=0
                )
            return flashinfer.sampling.top_p_sampling_from_probs(
                probs, case["top_p"], seed=17, offset=0
            )

        def run():
            probs.copy_(probs0)
            return sample_once()

        def verify(value):
            samples = value[0] if isinstance(value, tuple) else value
            if samples.dtype.is_floating_point:
                raise AssertionError("sampling returned float ids")
            if int(samples.min()) < 0 or int(samples.max()) >= case["vocab"]:
                raise AssertionError("sampled ids outside vocab")
            probs.copy_(probs0)
            again = sample_once()
            again_ids = again[0] if isinstance(again, tuple) else again
            if not torch.equal(samples, again_ids):
                raise AssertionError("sampling is not deterministic after probability reset")
            if op == "top_k_sample":
                top = torch.topk(probs0, case["top_k"], dim=-1).indices
                for row in range(samples.shape[0]):
                    if int(samples[row]) not in set(top[row].tolist()):
                        raise AssertionError("top_k sample left the top-k set")
            if op == "top_p_sample":
                sorted_probs, sorted_idx = torch.sort(probs0, dim=-1, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                keep = cdf <= case["top_p"]
                keep[:, 0] = True
                for row in range(samples.shape[0]):
                    allowed = set(sorted_idx[row][keep[row]].tolist())
                    # include the boundary token that first exceeds p
                    if not keep[row].all():
                        first = int((~keep[row]).nonzero()[0])
                        allowed.add(int(sorted_idx[row, first]))
                    if int(samples[row]) not in allowed:
                        raise AssertionError("top_p sample left the nucleus")
            return {"status": "pass", "finite": True, "method": "sampling_range_nucleus_and_reset"}

        return Prepared(run, verify)

    if op in {"packbits", "segment_packbits"}:
        bits = torch.randint(0, 2, (case["elements"],), dtype=torch.bool, device="cuda")
        if op == "packbits":
            def run():
                return flashinfer.packbits(bits, bitorder=case["bitorder"])

            def verify(value):
                packed = value
                width = 8
                n = bits.numel()
                expected = torch.zeros((n + 7) // 8, dtype=torch.uint8, device="cuda")
                for bit in range(width):
                    take = bits[bit::width][: expected.numel()]
                    if case["bitorder"] == "little":
                        expected[: take.numel()] |= take.to(torch.uint8) << bit
                    else:
                        expected[: take.numel()] |= take.to(torch.uint8) << (7 - bit)
                # Vectorised reconstruction: compare unpacked bits rather than reimplement packing loops.
                unpacked = torch.zeros_like(bits)
                packed_u = packed.to(torch.int32)
                for bit in range(8):
                    shift = bit if case["bitorder"] == "little" else 7 - bit
                    unpacked[bit::8] = ((packed_u >> shift) & 1).bool()[: unpacked[bit::8].numel()]
                if not torch.equal(unpacked[:n], bits):
                    raise AssertionError("packbits round-trip mismatch")
                return {"status": "pass", "finite": True, "method": "packbits_roundtrip"}

            return Prepared(run, verify)
        segment_size = case["elements"] // case["segments"]
        indptr = torch.arange(0, case["elements"] + 1, segment_size, dtype=torch.int32, device="cuda")

        def run():
            return flashinfer.segment_packbits(bits, indptr, bitorder=case["bitorder"])

        def verify(value):
            packed, new_indptr = value if isinstance(value, tuple) else (value, None)
            if packed.numel() == 0:
                raise AssertionError("segment_packbits returned empty")
            if not torch.isfinite(packed.float() if packed.is_floating_point() else packed.to(torch.float32)).all() and packed.is_floating_point():
                raise AssertionError("segment_packbits non-finite")
            # Semantic: packed byte count matches ceil(segment/8) * segments.
            expected_bytes = case["segments"] * ((segment_size + 7) // 8)
            if packed.numel() != expected_bytes:
                raise AssertionError(f"segment_packbits size {packed.numel()} != {expected_bytes}")
            return {"status": "pass", "finite": True, "method": "segment_packbits_size_invariant"}

        return Prepared(run, verify)

    if op in {"merge_state", "merge_states"}:
        value_dtype = dtype(torch, case["dtype"])
        shape = (case["seqlen"], case["heads"], case["head_dim"])
        if op == "merge_state":
            va, vb = random_tensor(torch, shape, value_dtype), random_tensor(torch, shape, value_dtype)
            sa = torch.randn(shape[:-1], dtype=torch.float32, device="cuda")
            sb = torch.randn_like(sa)

            def run():
                return flashinfer.merge_state(va, sa, vb, sb)

            def verify(value):
                v, s = value
                v_ref, s_ref = merge_pair_log2(torch, va, sa, vb, sb)
                close(torch, v, v_ref, rtol=0.03, atol=0.02, name=case["case_id"])
                close(torch, s, s_ref, rtol=0.03, atol=0.02, name=case["case_id"] + ".lse")
                return {"status": "pass", "finite": True, "method": "torch_log2_merge_state"}

            return Prepared(run, verify)
        values = random_tensor(
            torch, (case["seqlen"], case["states"], case["heads"], case["head_dim"]), value_dtype
        )
        lse = torch.randn(case["seqlen"], case["states"], case["heads"], dtype=torch.float32, device="cuda")

        def run():
            return flashinfer.merge_states(values, lse)

        def verify(value):
            v, s = value
            v_ref, s_ref = merge_many_log2(torch, values, lse)
            close(torch, v, v_ref, rtol=0.03, atol=0.02, name=case["case_id"])
            close(torch, s, s_ref, rtol=0.03, atol=0.02, name=case["case_id"] + ".lse")
            return {"status": "pass", "finite": True, "method": "torch_log2_merge_states"}

        return Prepared(run, verify)
    raise ValueError(f"unsupported operation: {op}")


def build_segment_packbits_phases(torch, flashinfer, case: dict):
    """Reproduce the two GPU phases around segment_packbits' required .item()."""
    from flashinfer.quantization.packbits import get_quantization_module

    bits = torch.randint(0, 2, (case["elements"],), dtype=torch.bool, device="cuda")
    segment_size = case["elements"] // case["segments"]
    indptr = torch.arange(
        0, case["elements"] + 1, segment_size, dtype=torch.int32, device="cuda"
    )
    seglen = torch.empty(case["segments"], dtype=torch.int32, device="cuda")
    packed_len = torch.empty_like(seglen)
    indptr_new = torch.empty(case["segments"] + 1, dtype=torch.int32, device="cuda")
    output_nnzs = case["segments"] * ((segment_size + 7) // 8)
    output = torch.empty(output_nnzs, dtype=torch.uint8, device="cuda")
    module = get_quantization_module()

    def phase_one():
        torch.sub(indptr[1:], indptr[:-1], out=seglen)
        torch.add(seglen, 7, out=packed_len)
        torch.div(packed_len, 8, rounding_mode="floor", out=packed_len)
        indptr_new.zero_()
        torch.cumsum(packed_len, 0, out=indptr_new[1:])
        return indptr_new

    def phase_two():
        module.segment_packbits(
            bits, indptr, indptr_new, case["bitorder"], output
        )
        return output, indptr_new

    public_run = lambda: flashinfer.segment_packbits(
        bits, indptr, bitorder=case["bitorder"]
    )
    return public_run, phase_one, phase_two


def tensors(value):
    if hasattr(value, "is_cuda"):
        return [value]
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(tensors(item))
        return result
    return []


def check_output(torch, value, verify: Callable[[object], dict] | None = None) -> dict:
    outputs = tensors(value)
    shapes = [list(tensor.shape) for tensor in outputs]
    finite = bool(outputs) and all(
        tensor.dtype == torch.bool or not tensor.is_floating_point() or torch.isfinite(tensor).all().item()
        for tensor in outputs
    )
    try:
        if not finite:
            raise AssertionError("non-finite or empty output")
        if verify is not None:
            result = verify(value)
            result.setdefault("output_shapes", shapes)
            result.setdefault("finite", True)
            return result
        return {
            "status": "pass",
            "finite": True,
            "output_shapes": shapes,
            "method": "public API completion and full-output finite check",
        }
    except Exception as exc:
        return {
            "status": "fail",
            "finite": finite,
            "output_shapes": shapes,
            "method": "independent_reference_or_invariant",
            "error": str(exc),
        }


def measure(
    torch,
    run: Callable[[], object],
    repeats: int,
    comparison_contract: dict | None,
) -> tuple[object, dict]:
    artifact_root = Path(os.environ["FLASHINFER_DELIVERY_ARTIFACTS_ROOT"])
    return measure_operation(
        torch,
        run,
        repeats,
        artifact_root / "build/libdelivery_globaltimer_timer.so",
        comparison_contract,
    )


def measure_segment_packbits(
    torch,
    flashinfer,
    case: dict,
    repeats: int,
    comparison_contract: dict | None,
):
    artifact_root = Path(os.environ["FLASHINFER_DELIVERY_ARTIFACTS_ROOT"])
    public_run, phase_one, phase_two = build_segment_packbits_phases(
        torch, flashinfer, case
    )
    return measure_two_phase_operation(
        torch,
        public_run,
        phase_one,
        phase_two,
        repeats,
        artifact_root / "build/libdelivery_globaltimer_timer.so",
        comparison_contract,
    )


def unit_covers_complete_launch(unit_spec: dict | None) -> bool:
    """True exactly when the unit's unique CTAs are the FULL original grid.

    Mechanical coverage fact (native_capture/campaign semantics), never the
    unit TYPE: a legal 1-CTA launch selected as type=cta and a complete
    2-CTA cluster covering the whole grid both cover, and the campaign's
    cross-check (covers_complete_launch == (unit ctas == grid coordinates))
    must agree with what the hardware row records.
    """
    if not isinstance(unit_spec, dict):
        raise ValueError(
            "unit facts are unusable for the coverage fact: no unit document"
        )
    try:
        ctas = {tuple(int(v) for v in cta) for cta in (unit_spec.get("ctas") or [])}
        grid = [int(v) for v in (unit_spec.get("grid") or [])]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unit facts are unusable for the coverage fact: {exc}"
        ) from exc
    if not ctas or len(grid) != 3 or any(axis < 1 for axis in grid):
        raise ValueError(
            f"unit facts are unusable for the coverage fact: ctas="
            f"{unit_spec.get('ctas')!r} grid={unit_spec.get('grid')!r}"
        )
    # Coverage WITHOUT enumerating the original grid (it can be far larger
    # than the <=2-CTA research unit): the unique in-bounds CTA coordinates
    # are at most grid-product many, so unique count == product AND every
    # coordinate in bounds proves the unit is exactly the full grid set.
    gx, gy, gz = grid
    product = gx * gy * gz
    return (
        len(ctas) == product
        and all(0 <= x < gx and 0 <= y < gy and 0 <= z < gz for x, y, z in ctas)
    )


def unit_marker_participant_counts(ctas: list, clock_marker_warps: list) -> dict:
    """Denominator counts from the shared observed participant list.

    ``required_rows_per_launch`` and ``slot_count`` are CTA count times the
    participating warp count. The list is the flat set already verified
    identical on every selected CTA.
    """
    if not isinstance(clock_marker_warps, list) or not clock_marker_warps:
        raise ValueError("clock_marker_warps is missing")
    if clock_marker_warps != sorted(set(clock_marker_warps)):
        raise ValueError("clock_marker_warps must be a sorted unique warp id list")
    if any(not isinstance(warp, int) or isinstance(warp, bool) or warp < 0 for warp in clock_marker_warps):
        raise ValueError("clock_marker_warps contains a non-warp id")
    if not ctas:
        raise ValueError("unit ctas are empty")
    rows = len(ctas) * len(clock_marker_warps)
    return {
        "clock_marker_warps": list(clock_marker_warps),
        "slot_count": rows,
        "required_rows_per_launch": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--marker-timing",
        action="store_true",
        dest="marker_timing",
        help="in-kernel clock-marker denominator pass (T0005 rev5): patches the wheel tree, "
             "builds JIT modules with -DFI_DELIVERY_MARKER_TIMING and records "
             "timing.main_kernel in the 296 outermost_in_kernel_clock_marker_interval scope; "
             "the whole-operation globaltimer stays as secondary evidence only",
    )
    parser.add_argument(
        "--marker-unit-json",
        type=Path,
        default=None,
        help="strict same-unit marker denominator (stage-6): JSON {ctas, grid, block} "
             "selecting the research unit's (cta, warp) slots; the marker pass reduces "
             "only those rows and the result records timing.marker_unit.  Omitted = "
             "whole-launch reduction (legacy behaviour)",
    )
    parser.add_argument(
        "--marker-pass-only",
        action="store_true",
        dest="marker_pass_only",
        help="capture-entry producer run: build the op and run only the marker pass "
             "(no whole-operation globaltimer protocol, no output-equality check -- "
             "the untraced measurement run of the same binary owns both); used by the "
             "unified capture entry so the traced launches still execute the real "
             "CS2R pairs",
    )
    parser.add_argument(
        "--capture-window",
        action="store_true",
        dest="capture_window",
        help="run-once capture mode (stage-6 unified entry): build the op through the "
             "public API, warm it like the timing path, then exactly one real op call; "
             "no marker patching, no sm_cap clamp, no profiler bracket — the tracer env "
             "and all trace post-processing belong to common/capture_case.py "
             "-> capture_binary.py",
    )
    parser.add_argument(
        "--comparison-contract",
        type=Path,
        help="capture metadata or corpus manifest selecting the one simulated kernel",
    )
    parser.add_argument(
        "--case-overrides",
        type=Path,
        help="JSON {case_id: entry} applied after manifest load (T0011.d far45 disposition "
             "sidecar).  Each entry is either the override map itself or carries it under "
             "--override-field (default param_overrides; use gqa_probe for the rev1-② probe "
             "runs).  Applied values are recorded in the result payload",
    )
    parser.add_argument(
        "--override-field",
        default="param_overrides",
        help="entry key holding the override map when the entry is a container (default: "
             "param_overrides)",
    )
    parser.add_argument(
        "--comparison-code-root",
        type=Path,
        action="append",
        default=[],
        help="directory containing the JIT binary used by this timed run",
    )
    args = parser.parse_args()
    # Loud refusal for marker-pass argument combinations that would
    # otherwise be silent no-ops (2026-10-10 review): both flags live inside
    # the --marker-timing path, --capture-window returns before it, and the
    # pass-only producer run reduces no timing at all.
    if args.marker_pass_only and not args.marker_timing:
        parser.error(
            "--marker-pass-only is the capture-entry producer mode of the "
            "marker pass and requires --marker-timing"
        )
    if args.marker_unit_json is not None and not args.marker_timing:
        parser.error(
            "--marker-unit-json scopes the marker pass reduction and "
            "requires --marker-timing"
        )
    if args.capture_window and (
            args.marker_pass_only or args.marker_unit_json is not None):
        parser.error(
            "--capture-window (run-once workload mode) is mutually exclusive "
            "with --marker-pass-only/--marker-unit-json"
        )
    if args.marker_pass_only and args.marker_unit_json is not None:
        parser.error(
            "--marker-pass-only captures the producer run and reduces no "
            "timing; --marker-unit-json has no effect here"
        )
    case = load_case(args.manifest, args.case_id)
    case_overrides: dict = {}
    if args.case_overrides is not None:
        all_overrides = json.loads(args.case_overrides.read_text(encoding="utf-8"))
        entry = all_overrides.get(case["case_id"]) or {}
        if isinstance(entry, dict) and "param_overrides" in entry:
            # Container entry (far45 disposition sidecar): pick the requested field; a
            # missing/None probe field for this case means "no overrides for this run".
            field = entry.get(args.override_field)
            if field is None:
                field = {}
            case_overrides = dict(field)
        else:
            case_overrides = dict(entry)
        for override_key, override_value in case_overrides.items():
            case[override_key] = override_value
    if args.dry_run:
        print(json.dumps({"status": "pass", "case": case, "case_overrides": case_overrides},
                         indent=2))
        return

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("FlashInfer Hopper suite requires a compute capability 9.x CUDA GPU")
    import flashinfer
    if (os.environ.get("FLASHINFER_EFFECTIVE_SM_COUNT")
            and not getattr(flashinfer.utils,
                            "HOPPER_EFFECTIVE_SM_COUNT_PATCH", False)):
        raise SystemExit(
            "FLASHINFER_EFFECTIVE_SM_COUNT set but flashinfer at "
            f"{flashinfer.__file__} lacks the patch")

    diagnostic_timing = os.environ.get("FLASHINFER_DIAGNOSTIC_TIMING") == "1"
    if diagnostic_timing:
        print(f"FLASHINFER_DIAG_PHASE=imports_done case={args.case_id}",
              flush=True)
    torch.manual_seed(20260818 + int(case["delivery_index"]))
    started = time.time()

    if args.capture_window:
        # Run-once capture mode (stage-6 unified capture entry): this process
        # is a plain workload for common/capture_case.py -> capture_binary.py.
        # Build the op through the public API (the first run materializes the
        # JIT modules the op needs), warm it like the timing path, then make
        # exactly one real op call.  No marker wheel patching, no marker
        # channel, no sm_cap clamp, no profiler bracket and no post-capture
        # chain here: the unified entry injects the tracer env and owns all
        # trace post-processing; the binary captured is whatever the pinned
        # runtime actually loads.
        phase_started = time.monotonic()
        if diagnostic_timing:
            print(f"FLASHINFER_DIAG_PHASE=build_start case={args.case_id}",
                  flush=True)
        prepared = build_operation(torch, flashinfer, case)
        build_seconds = time.monotonic() - phase_started
        if diagnostic_timing:
            print(
                f"FLASHINFER_DIAG_PHASE=build_done case={args.case_id} "
                f"seconds={build_seconds:.3f}",
                flush=True,
            )
        warmup_started = time.monotonic()
        for _ in range(CAPTURE_WARMUP_RUNS):
            prepared.run()
        torch.cuda.synchronize()
        warmup_seconds = time.monotonic() - warmup_started
        if diagnostic_timing:
            print(
                f"FLASHINFER_DIAG_PHASE=warmup_done case={args.case_id} "
                f"seconds={warmup_seconds:.3f}",
                flush=True,
            )
        started = time.time()
        capture_started = time.monotonic()
        value = prepared.run()
        torch.cuda.synchronize()
        capture_seconds = time.monotonic() - capture_started
        if diagnostic_timing:
            print(
                f"FLASHINFER_DIAG_PHASE=capture_done case={args.case_id} "
                f"seconds={capture_seconds:.3f}",
                flush=True,
            )
        payload = {
            "schema_version": "flashinfer-hopper-run-once-v1",
            "status": "captured",
            "case_id": case["case_id"],
            "case": case,
            "mode": "run_once",
            "warmup_launches": CAPTURE_WARMUP_RUNS,
            "captured_launches": 1,
            "output_tensors": [list(t.shape) for t in tensors(value)],
            "elapsed_seconds": time.time() - started,
        }
        if diagnostic_timing:
            payload["diagnostic_timing"] = {
                "build_seconds": build_seconds,
                "warmup_seconds": warmup_seconds,
                "capture_seconds": capture_seconds,
            }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(
            {"case_id": case["case_id"], "status": "captured", "mode": "run_once"},
            sort_keys=True,
        ))
        return

    marker_mode = args.marker_timing
    if marker_mode:
        import marker_mode as mm
        import marker_timing as mt
        artifact_root = Path(os.environ["FLASHINFER_DELIVERY_ARTIFACTS_ROOT"])
        mm.apply_marker_mode(flashinfer, artifact_root)
        # T0011.d (moe/mamba ninja-failure mitigation): cap JIT ninja parallelism unless the
        # wave driver pins it.  cutlass_fused_moe's marker-state JIT died with
        # "ninja: build stopped" at default parallelism (T0005 rounds).
        os.environ.setdefault("MAX_JOBS", "8")
    # T0011.d rev1 item 2: host-side SM cap — python plan side reads the same clamped SM
    # count the wheel-patched C++ plan/launch sites use (single-case process; the shim stays
    # active for the process lifetime and is a no-op unless $FI_DELIVERY_SM_CAP is set).
    # Measurement paths only: the run-once capture path above never clamps.
    import sm_cap
    sm_cap.sm_cap_properties(torch).__enter__()

    comparison_contract = (
        load_comparison_contract(args.comparison_contract, args.case_id)
        if args.comparison_contract is not None
        else None
    )


    prepared = None
    value = None
    if args.marker_pass_only:
        # Stage-6 capture-entry mode: build the op and run ONLY the marker
        # pass -- no whole-operation globaltimer protocol (its five-second
        # stream-release watch is calibrated for untraced launches and the
        # traced run is the producer program, not a timing sample).  The
        # captured launches still execute the real CS2R pairs; output
        # equality stays with the untraced measurement run.
        prepared = build_operation(torch, flashinfer, case)
        # One real launch loads the instrumented JIT module into the
        # registry -- install_channel needs a non-empty registry.  Its rows
        # (if any) are dropped by the marker pass's own zero_all.
        prepared.run()
        torch.cuda.synchronize()
        timing = {
            "marker_pass_only": True,
            "median_globaltimer": None,
            "median_cycles": None,
            "cv_percent": None,
            "simulator_comparison": {},
            "whole_operation_timing_must_not_be_used_for_simulator_accuracy": True,
        }
    elif case["operation"] == "segment_packbits":
        value, timing = measure_segment_packbits(
            torch, flashinfer, case, args.repeats, comparison_contract
        )
        prepared = build_operation(torch, flashinfer, case)
    else:
        prepared = build_operation(torch, flashinfer, case)
        value, timing = measure(torch, prepared.run, args.repeats, comparison_contract)
    if args.marker_timing:
        # T0011.g freshness fix: force-build rebuild callable (see capture path).
        def _rebuild_force_build():
            rebuilt = build_operation(torch, flashinfer, case)
            rebuilt.run()
            return rebuilt

        freshness_gate = mm.ensure_fresh_instrumentation(
            case["operation"], case, _rebuild_force_build,
            flashinfer_module=flashinfer,
        )
        if freshness_gate.get("status") == "rebuilt":
            prepared = build_operation(torch, flashinfer, case)
        channel, per_key = mm.install_channel(torch, case["operation"], case)
        if args.marker_pass_only:
            # Capture one physical operation with the real marker pointers
            # installed.  NVBit ACTIVE_FROM_START=0 excludes setup launches;
            # this is not a timing sample or the bare 27-to-9 protocol.
            channel.ensure_entries()
            channel.zero_all()
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            try:
                prepared.run()
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
            payload = {
                "case_id": case["case_id"], "status": "captured",
                "mode": "marker_pass_only", "captured_operations": 1,
            }
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            print(json.dumps(payload, sort_keys=True))
            return
        unit_slot_filter = None
        unit_warps_per_cta = None
        if args.marker_unit_json is not None:
            unit_spec = json.loads(args.marker_unit_json.read_text(encoding="utf-8"))
            unit_slot_filter = mt.unit_slot_set(
                unit_spec["ctas"], unit_spec["grid"], unit_spec["block"])
            unit_warps_per_cta = mt.warps_per_cta(unit_spec["block"])
        marker = mm.run_marker_pass(
            torch, prepared, case["operation"], case, channel, per_key,
            unit_slot_filter=unit_slot_filter, unit_warps_per_cta=unit_warps_per_cta)
        timing["main_kernel"] = marker["main_kernel"]
        timing["marker_unit"] = (
            {
                "source": str(args.marker_unit_json),
                "ctas": unit_spec["ctas"],
                "grid": unit_spec["grid"],
                "block": unit_spec["block"],
                "slot_count": len(unit_slot_filter),
            }
            if args.marker_unit_json is not None
            else None
        )
        timing["marker_channel"] = {
            "schema": "delivery.marker-channel.v1",
            "scope": marker["main_kernel"]["scope"],
            "method": marker["main_kernel"]["method"],
            "registry_entries": marker["registry_entries"],
            "channel": per_key,
            "raw_launch_count": marker["raw_launch_count"],
            "duration_match": marker["duration_match"],
            "freshness_gate": freshness_gate,
            "case_overrides": case_overrides,
        }
        if unit_slot_filter is not None:
            # Strict same-unit denominator contract
            # (delivery.unit-marker-denominator.v1): the median/samples are
            # reduced ONLY over the unit's own (cta, warp) slots of the same
            # instrumented binary; the whole-kernel CUDA activity rides as a
            # diagnostic (unit mode is the only mode here: --marker-unit-json
            # implies the marker pass).  covers_complete_launch is the
            # MECHANICAL coverage fact (unique unit CTAs == the full original
            # grid; a legal 1-CTA grid under type=cta and a complete 2-CTA
            # cluster both cover), matching native_capture and the campaign
            # cross-check -- never the unit type.
            unit_type = unit_spec.get("type") if isinstance(unit_spec, dict) else "launch"
            covers = unit_covers_complete_launch(unit_spec)
            dm = marker["duration_match"]
            ctas = unit_spec.get("ctas") if isinstance(unit_spec, dict) else None
            participants = unit_marker_participant_counts(ctas, marker.get("clock_marker_warps"))
            timing["unit_denominator"] = {
                "schema": "delivery.unit-marker-denominator.v1",
                "eligible_for_simulator_accuracy": (
                    marker["main_kernel"].get("status") == "matched"
                    and marker["main_kernel"].get("sample_count") == 9
                    and dm.get("status") == "diagnostic_only"
                ),
                "status": "same_unit_matched",
                "scope": marker["main_kernel"]["scope"],
                "unit_scope": {
                    "type": unit_type,
                    "ctas": ctas,
                    "grid": unit_spec.get("grid") if isinstance(unit_spec, dict) else None,
                    "block": unit_spec.get("block") if isinstance(unit_spec, dict) else None,
                    "block_warps": unit_warps_per_cta,
                    "slot_count": participants["slot_count"],
                    "clock_marker_warps": participants["clock_marker_warps"],
                    "covers_complete_launch": covers,
                },
                "member_warp_binding": {
                    "unit_ctas": ctas,
                    "required_rows_per_launch": participants["required_rows_per_launch"],
                    "clock_marker_warps": participants["clock_marker_warps"],
                    "launches_reduced": marker["raw_launch_count"],
                },
                "code_identity": {
                    "resolved_kernel": dm.get("resolved_kernel_name"),
                    "activity_kernel_name": dm.get("activity_kernel_name"),
                    "marker_registry_entries": marker["registry_entries"],
                },
                "median_cycles": marker["main_kernel"].get("median_cycles"),
                "sample_count": marker["main_kernel"].get("sample_count"),
                "samples_cycles": marker["main_kernel"].get("samples_cycles"),
                "activity_diagnostic": {
                    "status": dm.get("status"),
                    "median_ns": dm.get("primary_kernel_activity_median_ns"),
                    "matched_name": dm.get("activity_kernel_name"),
                },
                "provenance": {
                    "unit_json": str(args.marker_unit_json),
                    # far45 disposition sidecar arrives as --case-overrides
                    # (argparse dest case_overrides); there is no --disposition.
                    "disposition": str(args.case_overrides) if args.case_overrides else None,
                },
            }
        else:
            timing["unit_denominator"] = None
        timing["whole_operation_timing_must_not_be_used_for_simulator_accuracy"] = True
    if comparison_contract is not None:
        timing["simulator_comparison"]["code_identity"] = (
            validate_runtime_binary_identity(
                comparison_contract, args.comparison_code_root
            )
        )
    if args.marker_pass_only:
        correctness = {
            "status": "marker_pass_only",
            "reason": (
                "capture-entry producer run: output equality is checked by "
                "the untraced measurement run of the same binary"
            ),
        }
    else:
        correctness = check_output(torch, value, prepared.verify if prepared is not None else None)
    result = {
        "schema_version": "flashinfer-hopper-result-v1",
        "status": correctness["status"],
        "case_id": case["case_id"],
        "case": case,
        "flashinfer_version": flashinfer.__version__,
        "hardware": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "correctness": correctness,
        "timing": timing,
        "elapsed_seconds": time.time() - started,
    }
    trace_log = os.environ.get("FLASHINFER_CLUSTER_LAUNCH_LOG")
    if trace_log:
        try:
            from cluster_topology import attach_topology
            attach_topology(result, Path(trace_log))
        except Exception as exc:
            result.setdefault("topology", {})["error"] = f"{type(exc).__name__}: {exc}"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"case_id": result["case_id"], "status": result["status"]}
    summary.update(
        median_globaltimer=timing["median_globaltimer"],
        median_cycles_compatibility=timing["median_cycles"],
        cv_percent=timing["cv_percent"],
        simulator_comparison_cycles=(timing.get("simulator_comparison") or {}).get(
            "median_cycles"
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
