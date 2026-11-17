"""FA3 full-GPU public operations and independent numerical checks.

build_operation(torch, interface, case) -> (run, state)
validate_outputs(torch, case, state) -> dict

Uses the official FA3 Python API only (flash_attn_func / flash_attn_varlen_func /
flash_attn_with_kvcache / flash_attn_combine / get_scheduler_metadata). Input
construction, frozen input snapshots, one untimed warmup (custom_op / autograd
first-dispatch compile), and host/device sync stay in build_operation. Timed
callables have none of that. The runner times then profiles the same callable;
validate_outputs compares the latest live out/grads/cache/LSE against those frozen
snapshots and never copies snapshots back onto live tensors. Official softmax.h
writes LSE=-INFINITY when a row softmax sum is 0 (causal Q>K fully masked rows);
LSE is not required to be globally finite. Closures still return out; autograd
is still taken w.r.t. out. LSE is requested via return_attn_probs (fixed/varlen)
or return_softmax_lse (kvcache) and is produced by the kernel itself.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

# Hopper FA3 public API cannot reach these; claiming them is a delivery bug.
_UNREACHABLE = {
    "dropout",
    "fp8vcol",
    "fp8_vcol",
    "fp8_v_colmajor",
    "v_colmajor",
    "vcol",
    "qkrs",
    "qk_rs",
    "rs-ss",
    "rs_ss",
    "sink",
    "learnable_sink",
    "attention_sink",
}


def dtype_from_name(torch, name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp32": torch.float32,
    }[name]


def _round_up_headdim(head_size: int) -> int:
    if head_size <= 64:
        return 64
    if head_size <= 96:
        return 96
    if head_size <= 128:
        return 128
    if head_size <= 192:
        return 192
    return 256


def _as_int(value, default=0) -> int:
    if value is None:
        return int(default)
    return int(value)


def _as_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _window(case: dict) -> tuple[int, int]:
    window = case.get("window_size")
    if window is None:
        return (-1, -1)
    left, right = window
    return (int(left), int(right))


def _group(case: dict) -> int:
    heads_q = int(case["heads_q"])
    heads_kv = int(case.get("heads_kv", heads_q))
    if heads_q % heads_kv != 0:
        raise ValueError(f"heads_q={heads_q} is not divisible by heads_kv={heads_kv}")
    return heads_q // heads_kv


def _head_dim_v(case: dict) -> int:
    return int(case.get("head_dim_v", case["head_dim"]))


def _mech_names(case: dict) -> list[str]:
    names = []
    primary = case.get("primary_mechanism")
    if primary:
        names.append(str(primary))
    for item in case.get("mechanisms") or []:
        names.append(str(item))
    return names


def _norm_mech(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def reject_unreachable(case: dict) -> None:
    """Fail closed on mechanisms FA3 cannot actually dispatch."""
    for name in _mech_names(case):
        key = _norm_mech(name)
        if key in _UNREACHABLE or any(token in _UNREACHABLE for token in key.split("+")):
            raise ValueError(f"FA3 cannot reach mechanism {name!r}; refusing to fake it")
    for field in ("dropout", "dropout_p", "learnable_sink", "sink", "v_colmajor", "qkrs"):
        value = case.get(field)
        if value not in (None, False, 0, 0.0):
            raise ValueError(f"FA3 cannot reach {field}={value!r}")
    operation = case["operation"]
    dtype = case["dtype"]
    head_dim = int(case["head_dim"])
    head_dim_v = _head_dim_v(case)
    qv = _as_bool(case.get("qv"))
    chunk = _as_int(case.get("attention_chunk"))
    deterministic = _as_bool(case.get("deterministic"))
    if operation == "combine" and (
        _as_bool(case.get("varlen")) or case.get("seqlens_q") is not None
    ):
        raise ValueError("public flash_attn_combine is fixed-shape only; varlen combine is covered by varlen split")
    if qv and dtype == "fp8_e4m3":
        raise ValueError("FA3 qv is not supported for fp8_e4m3")
    if qv and head_dim != 64:
        raise ValueError("FA3 qv requires head_dim=64")
    if operation in {"fixed_bwd", "varlen_bwd"}:
        if dtype == "fp8_e4m3":
            raise ValueError("FA3 backward is not supported for fp8")
        if qv:
            raise ValueError("FA3 backward is not supported with qv")
        if chunk:
            raise ValueError("FA3 backward is not supported with attention_chunk")
        if head_dim_v > 256:
            raise ValueError("FA3 backward is not supported for head_dim_v>256")
    if deterministic:
        rounded = _round_up_headdim(max(head_dim, head_dim_v))
        if rounded >= 256:
            raise ValueError("FA3 deterministic backward is not supported for hdim 256")
        if operation not in {"fixed_bwd", "varlen_bwd"}:
            raise ValueError("FA3 deterministic flag is a backward-only trait")
    if _as_bool(case.get("reuse_scheduler_metadata")) and operation != "kvcache":
        raise ValueError("reuse_scheduler_metadata is only valid for kvcache")
    if _as_int(case.get("rotary_dim")) and not _as_bool(case.get("append_kv")):
        raise ValueError("FA3 rotary is only applied when appending KV")


def make_tensor(torch, shape, dtype, requires_grad=False, scale=0.5):
    base = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale
    tensor = base.to(dtype)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def cu_seqlens(torch, lengths: list[int]):
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _softmax_scale(case: dict) -> float:
    head_dim = int(case["head_dim"])
    extra = _head_dim_v(case) if _as_bool(case.get("qv")) else 0
    return 1.0 / math.sqrt(head_dim + extra)


def _descale(torch, case: dict, batch: int):
    if case["dtype"] != "fp8_e4m3":
        return None, None, None
    heads_kv = int(case.get("heads_kv", case["heads_q"]))
    q_descale = torch.rand(batch, heads_kv, device="cuda", dtype=torch.float32) * 2
    k_descale = torch.rand(batch, heads_kv, device="cuda", dtype=torch.float32) * 2
    v_descale = torch.rand(batch, heads_kv, device="cuda", dtype=torch.float32) * 2
    return q_descale, k_descale, v_descale


def _common_fwd_kwargs(case: dict, *, qv=None, descales=None) -> dict:
    kwargs = {"causal": _as_bool(case.get("causal"))}
    window = _window(case)
    if window != (-1, -1):
        kwargs["window_size"] = window
    chunk = _as_int(case.get("attention_chunk"))
    if chunk:
        kwargs["attention_chunk"] = chunk
    softcap = float(case.get("softcap") or 0.0)
    if softcap:
        kwargs["softcap"] = softcap
    pack_gqa = case.get("pack_gqa")
    if pack_gqa is not None:
        kwargs["pack_gqa"] = bool(pack_gqa)
    if case.get("num_splits") is not None:
        kwargs["num_splits"] = int(case["num_splits"])
    if case.get("sm_margin") is not None:
        kwargs["sm_margin"] = int(case["sm_margin"])
    if _as_bool(case.get("deterministic")):
        kwargs["deterministic"] = True
    if qv is not None:
        kwargs["qv"] = qv
        kwargs["softmax_scale"] = _softmax_scale(case)
    if descales is not None and descales[0] is not None:
        kwargs["q_descale"], kwargs["k_descale"], kwargs["v_descale"] = descales
    return kwargs


def apply_rotary(x, cos, sin, offsets, interleaved: bool, same_pos: bool):
    """RoPE on x[..., :rotary_dim]. offsets: (batch,) start positions."""
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    seqlen = x.shape[1]
    if same_pos:
        positions = offsets.to(dtype=torch.long)[:, None].expand(-1, seqlen)
    else:
        positions = offsets.to(dtype=torch.long)[:, None] + torch.arange(
            seqlen, device=x.device, dtype=torch.long
        )
    positions = positions.clamp(0, cos.shape[0] - 1)
    cos_g = cos[positions][:, :, None, :]
    sin_g = sin[positions][:, :, None, :]
    if interleaved:
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        out_rot = torch.stack((x1 * cos_g - x2 * sin_g, x1 * sin_g + x2 * cos_g), dim=-1).flatten(-2)
    else:
        half = rotary_dim // 2
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        out_rot = torch.cat((x1 * cos_g - x2 * sin_g, x1 * sin_g + x2 * cos_g), dim=-1)
    return torch.cat((out_rot.to(dtype=x.dtype), x_pass), dim=-1)


def gather_paged(paged, page_table, seqlen: int):
    blocks = paged[page_table.long()]
    batch, nblocks, page = blocks.shape[0], blocks.shape[1], blocks.shape[2]
    return blocks.reshape(batch, nblocks * page, *paged.shape[2:])[:, :seqlen]


def _row_mask(torch, query_index, seqlen_q: int, seqlen_k: int, case: dict, *, used_k=None, leftpad=0, device=None):
    """Original K columns. Chunk/local match test_util: shift col by leftpad, align with n_valid=used_k-leftpad."""
    device = torch.device("cuda") if device is None else device
    columns = torch.arange(seqlen_k, device=device)
    allowed = torch.ones(seqlen_k, dtype=torch.bool, device=device)
    pad = int(leftpad or 0)
    sk_raw = int(used_k) if used_k is not None else seqlen_k
    n_valid = sk_raw - pad
    aligned = query_index + n_valid - seqlen_q
    left, right = _window(case)
    if _as_bool(case.get("causal")):
        right = 0
    if left >= 0:
        allowed = allowed & (columns >= (aligned - left + pad))
    if right >= 0:
        allowed = allowed & (columns <= (aligned + right + pad))
    chunk = _as_int(case.get("attention_chunk"))
    if chunk > 0:
        base = aligned - (aligned % chunk)
        allowed = allowed & (columns >= base + pad) & (columns < base + chunk + pad)
    if used_k is not None:
        allowed = allowed & (columns < sk_raw)
    if pad:
        allowed = allowed & (columns >= pad)
    return allowed


def _sample_points(length: int) -> list[int]:
    if length <= 0:
        return []
    if length == 1:
        return [0]
    return sorted({0, length // 2, length - 1})


def _selected_batches(batch: int) -> list[int]:
    return sorted({0, batch - 1}) if batch else []


def _selected_heads(heads: int) -> list[int]:
    return sorted({0, heads - 1}) if heads else []


def sampled_attention_error(torch, case: dict, state: dict) -> tuple[float, int, float]:
    """Compare selected rows on the real full shape against an explicit FP32 reference."""
    q = state["q_ref"]
    k = state["k_ref"]
    v = state["v_ref"]
    qv = state.get("qv_ref")
    out = state["out"]
    varlen = case["operation"].startswith("varlen")
    if varlen:
        q_lengths = [int(x) for x in case["seqlens_q"]]
        k_lengths = [int(x) for x in case["seqlens_k"]]
        q_offsets = [0]
        k_offsets = [0]
        for length in q_lengths:
            q_offsets.append(q_offsets[-1] + length)
        for length in k_lengths:
            k_offsets.append(k_offsets[-1] + length)
        batches = range(len(q_lengths))
    else:
        q_lengths = [int(case["seqlen_q"])] * int(case["batch"])
        k_lengths = [int(state.get("ref_seqlen_k", case["seqlen_k"]))] * int(case["batch"])
        q_offsets = k_offsets = None
        batches = range(int(case["batch"]))

    scale = _softmax_scale(case)
    group = _group(case)
    heads_q = int(case["heads_q"])
    heads_kv = int(case.get("heads_kv", heads_q))
    maximum = 0.0
    relative_maximum = 0.0
    samples = 0
    q_descale, k_descale, v_descale = state.get("descales", (None, None, None))
    used_ks = state.get("used_k")
    leftpads = state.get("leftpad")
    for batch in batches:
        sq, sk = q_lengths[batch], k_lengths[batch]
        used_k = int(used_ks[batch]) if used_ks is not None else sk
        leftpad = int(leftpads[batch]) if leftpads is not None else 0
        for query_index in _sample_points(sq):
            for head in _selected_heads(heads_q):
                kv_head = head // group
                if q_offsets is None:
                    query = q[batch, query_index, head]
                    keys = k[batch, :sk, kv_head]
                    values = v[batch, :sk, kv_head]
                    actual = out[batch, query_index, head].float()
                    qv_row = qv[batch, query_index, head] if qv is not None else None
                    if q_descale is not None:
                        query = query.float() * q_descale[batch, kv_head]
                        keys = keys.float() * k_descale[batch, kv_head]
                        values = values.float() * v_descale[batch, kv_head]
                        if qv_row is not None:
                            qv_row = qv_row.float() * q_descale[batch, kv_head]
                    else:
                        query, keys, values = query.float(), keys.float(), values.float()
                        qv_row = qv_row.float() if qv_row is not None else None
                else:
                    query = q[q_offsets[batch] + query_index, head]
                    keys = k[k_offsets[batch]:k_offsets[batch + 1], kv_head]
                    values = v[k_offsets[batch]:k_offsets[batch + 1], kv_head]
                    actual = out[q_offsets[batch] + query_index, head].float()
                    qv_row = qv[q_offsets[batch] + query_index, head] if qv is not None else None
                    if q_descale is not None:
                        query = query.float() * q_descale[batch, kv_head]
                        keys = keys.float() * k_descale[batch, kv_head]
                        values = values.float() * v_descale[batch, kv_head]
                        if qv_row is not None:
                            qv_row = qv_row.float() * q_descale[batch, kv_head]
                    else:
                        query, keys, values = query.float(), keys.float(), values.float()
                        qv_row = qv_row.float() if qv_row is not None else None
                scores = torch.mv(keys, query) * scale
                if qv_row is not None:
                    scores = scores + torch.mv(values, qv_row) * scale
                cap = float(case.get("softcap") or 0.0)
                if cap:
                    scores = torch.tanh(scores / cap) * cap
                allowed = _row_mask(
                    torch, query_index, sq, keys.shape[0], case, used_k=used_k, leftpad=leftpad, device=keys.device
                )
                scores = scores.masked_fill(~allowed, float("-inf"))
                if not bool(allowed.any().item()):
                    reference = torch.zeros_like(actual)
                else:
                    weights = torch.softmax(scores, dim=0)
                    reference = torch.mv(values.transpose(0, 1), weights)
                error = float((actual - reference).abs().max().item())
                reference_scale = max(float(reference.abs().max().item()), 1e-3)
                maximum = max(maximum, error)
                relative_maximum = max(relative_maximum, error / reference_scale)
                samples += 1
    return maximum, samples, relative_maximum


def _fwd_mask_matrix(torch, seqlen_q: int, seqlen_k: int, case: dict, *, used_k=None, leftpad=0, device=None):
    queries = torch.arange(seqlen_q, device=device)[:, None]
    return _row_mask(
        torch, queries, seqlen_q, seqlen_k, case,
        used_k=used_k, leftpad=leftpad, device=device,
    ).expand(seqlen_q, seqlen_k)


def _head_autograd(torch, q, k, v, grad_out, case: dict, *, used_k=None, leftpad=0):
    """FP32 autograd for one (seqlen_q, D) head on the real sequence length."""
    q = q.float().detach().clone().requires_grad_(True)
    k = k.float().detach().clone().requires_grad_(True)
    v = v.float().detach().clone().requires_grad_(True)
    scale = _softmax_scale(case)
    scores = q @ k.transpose(0, 1) * scale
    cap = float(case.get("softcap") or 0.0)
    if cap:
        scores = torch.tanh(scores / cap) * cap
    mask = _fwd_mask_matrix(
        torch, q.shape[0], k.shape[0], case, used_k=used_k, leftpad=leftpad, device=q.device
    )
    scores = scores.masked_fill(~mask, float("-inf"))
    all_masked = ~mask.any(dim=-1)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(all_masked[:, None], torch.zeros_like(weights), weights)
    out = weights @ v
    out.backward(grad_out.float())
    return q.grad, k.grad, v.grad


def _clone_detached(tensor):
    if tensor is None:
        return None
    return tensor.detach().clone()


def _unpack_fwd(result):
    """Official APIs return `out` or `(out, softmax_lse, ...)`."""
    if isinstance(result, (tuple, list)):
        out = result[0]
        lse = result[1] if len(result) > 1 else None
        return out, lse
    return result, None


def _official_out_shape(case: dict) -> tuple:
    dv = _head_dim_v(case)
    hq = int(case["heads_q"])
    op = case["operation"]
    if op in {"fixed_fwd", "fixed_bwd", "kvcache", "combine"}:
        return (int(case["batch"]), int(case["seqlen_q"]), hq, dv)
    if op in {"varlen_fwd", "varlen_bwd"}:
        return (sum(int(x) for x in case["seqlens_q"]), hq, dv)
    raise ValueError(f"unsupported operation: {op}")


def _official_lse_shape(case: dict) -> tuple:
    """Kernel fwd LSE is (B,H,S) / varlen (H,Tq). combine public LSE is (B,S,H)."""
    hq = int(case["heads_q"])
    op = case["operation"]
    if op in {"fixed_fwd", "fixed_bwd", "kvcache"}:
        return (int(case["batch"]), hq, int(case["seqlen_q"]))
    if op in {"varlen_fwd", "varlen_bwd"}:
        return (hq, sum(int(x) for x in case["seqlens_q"]))
    if op == "combine":
        return (int(case["batch"]), int(case["seqlen_q"]), hq)
    raise ValueError(f"unsupported operation: {op}")


def _lse_tolerance(case: dict) -> float:
    if case["operation"] == "combine":
        return 1e-5
    # LSE is FP32, not the quantized attention output.
    return 2e-3


def compare_lse(torch, actual, reference, tolerance: float) -> dict:
    """Inf signs must match. Finite rows compared numerically. Global LSE finite is not required.

    Official softmax.h finalize: sum==0 or NaN sum writes LSE=-INFINITY (causal Q>K fully masked rows).
    """
    if actual is None or reference is None:
        return {
            "lse_pass": False,
            "lse_global_finite_required": False,
            "lse_missing": True,
            "lse_inf_sign_ok": False,
            "lse_finite_max_absolute_error": None,
            "lse_finite_rows": 0,
            "lse_neg_inf_rows": 0,
            "lse_pos_inf_rows": 0,
            "lse_nan_rows": 0,
            "lse_tolerance": tolerance,
        }
    if tuple(actual.shape) != tuple(reference.shape):
        return {
            "lse_pass": False,
            "lse_global_finite_required": False,
            "lse_missing": False,
            "lse_shape_mismatch": True,
            "lse_inf_sign_ok": False,
            "lse_finite_max_absolute_error": None,
            "lse_finite_rows": 0,
            "lse_neg_inf_rows": 0,
            "lse_pos_inf_rows": 0,
            "lse_nan_rows": 0,
            "lse_tolerance": tolerance,
        }
    act = actual.float()
    ref = reference.float()
    nan_rows = int(torch.isnan(act).sum().item())
    act_neg = torch.isneginf(act)
    act_pos = torch.isposinf(act)
    ref_neg = torch.isneginf(ref)
    ref_pos = torch.isposinf(ref)
    inf_sign_ok = bool((act_neg == ref_neg).all().item() and (act_pos == ref_pos).all().item())
    finite = torch.isfinite(act) & torch.isfinite(ref)
    finite_rows = int(finite.sum().item())
    if finite_rows:
        finite_err = float((act[finite] - ref[finite]).abs().max().item())
    else:
        finite_err = 0.0
    finite_ok = bool(finite_err <= tolerance)
    return {
        "lse_pass": bool(nan_rows == 0 and inf_sign_ok and finite_ok),
        "lse_global_finite_required": False,
        "lse_missing": False,
        "lse_inf_sign_ok": inf_sign_ok,
        "lse_finite_max_absolute_error": finite_err,
        "lse_finite_rows": finite_rows,
        "lse_neg_inf_rows": int(act_neg.sum().item()),
        "lse_pos_inf_rows": int(act_pos.sum().item()),
        "lse_nan_rows": nan_rows,
        "lse_tolerance": tolerance,
        "lse_ref_neg_inf_rows": int(ref_neg.sum().item()),
    }


def attention_lse_error(torch, case: dict, state: dict) -> dict:
    """Full-row inf-sign vs mask; sampled finite rows vs FP32 logsumexp. LSE need not be globally finite."""
    lse = state.get("lse")
    tolerance = _lse_tolerance(case)
    if lse is None:
        return compare_lse(torch, None, None, tolerance)
    q = state["q_ref"]
    k = state["k_ref"]
    v = state["v_ref"]
    qv = state.get("qv_ref")
    varlen = case["operation"].startswith("varlen")
    if varlen:
        q_lengths = [int(x) for x in case["seqlens_q"]]
        k_lengths = [int(x) for x in case["seqlens_k"]]
        q_offsets = [0]
        k_offsets = [0]
        for length in q_lengths:
            q_offsets.append(q_offsets[-1] + length)
        for length in k_lengths:
            k_offsets.append(k_offsets[-1] + length)
        batches = list(range(len(q_lengths)))
    else:
        q_lengths = [int(case["seqlen_q"])] * int(case["batch"])
        k_lengths = [int(state.get("ref_seqlen_k", case["seqlen_k"]))] * int(case["batch"])
        q_offsets = k_offsets = None
        batches = list(range(int(case["batch"])))
    scale = _softmax_scale(case)
    group = _group(case)
    heads_q = int(case["heads_q"])
    q_descale, k_descale, v_descale = state.get("descales", (None, None, None))
    used_ks = state.get("used_k")
    leftpads = state.get("leftpad")
    ref = torch.full(_official_lse_shape(case), float("nan"), dtype=torch.float32, device=lse.device)
    for batch in batches:
        sq, sk = q_lengths[batch], k_lengths[batch]
        used_k = int(used_ks[batch]) if used_ks is not None else sk
        leftpad = int(leftpads[batch]) if leftpads is not None else 0
        mask = _fwd_mask_matrix(torch, sq, sk, case, used_k=used_k, leftpad=leftpad, device=lse.device)
        fully = ~mask.any(dim=-1)
        for head in range(heads_q):
            kv_head = head // group
            if q_offsets is None:
                dest = ref[batch, head]
            else:
                dest = ref[head, q_offsets[batch]:q_offsets[batch] + sq]
            dest.copy_(torch.where(fully, torch.full_like(dest, float("-inf")), dest))
            # Numerical only on sampled query rows that are not fully masked; inf-sign covers every row.
            for query_index in _sample_points(sq):
                if bool(fully[query_index].item()):
                    continue
                if q_offsets is None:
                    query = q[batch, query_index, head]
                    keys = k[batch, :sk, kv_head]
                    values = v[batch, :sk, kv_head]
                    qv_row = qv[batch, query_index, head] if qv is not None else None
                    if q_descale is not None:
                        query = query.float() * q_descale[batch, kv_head]
                        keys = keys.float() * k_descale[batch, kv_head]
                        values = values.float() * v_descale[batch, kv_head]
                        if qv_row is not None:
                            qv_row = qv_row.float() * q_descale[batch, kv_head]
                    else:
                        query, keys, values = query.float(), keys.float(), values.float()
                        qv_row = qv_row.float() if qv_row is not None else None
                else:
                    query = q[q_offsets[batch] + query_index, head]
                    keys = k[k_offsets[batch]:k_offsets[batch + 1], kv_head]
                    values = v[k_offsets[batch]:k_offsets[batch + 1], kv_head]
                    qv_row = qv[q_offsets[batch] + query_index, head] if qv is not None else None
                    if q_descale is not None:
                        query = query.float() * q_descale[batch, kv_head]
                        keys = keys.float() * k_descale[batch, kv_head]
                        values = values.float() * v_descale[batch, kv_head]
                        if qv_row is not None:
                            qv_row = qv_row.float() * q_descale[batch, kv_head]
                    else:
                        query, keys, values = query.float(), keys.float(), values.float()
                        qv_row = qv_row.float() if qv_row is not None else None
                scores = torch.mv(keys, query) * scale
                if qv_row is not None:
                    scores = scores + torch.mv(values, qv_row) * scale
                cap = float(case.get("softcap") or 0.0)
                if cap:
                    scores = torch.tanh(scores / cap) * cap
                allowed = mask[query_index]
                scores = scores.masked_fill(~allowed, float("-inf"))
                dest[query_index] = torch.logsumexp(scores, dim=0)
    # Rows that stay NaN in ref were unsampled finite rows: copy actual so they don't fail numerical,
    # but inf-sign vs fully-masked -inf is already written for every row.
    both_finite_unsampled = torch.isnan(ref) & torch.isfinite(lse.float())
    ref = torch.where(both_finite_unsampled, lse.float(), ref)
    still_nan = torch.isnan(ref)
    if bool(still_nan.any().item()):
        # Unsampled finite ref slots whose actual is also non-finite: keep NaN -> compare_lse nan_rows fail
        # unless actual is -inf and we expected finite; write +0 so inf-sign fails instead of NaN skip.
        ref = torch.where(still_nan, torch.zeros_like(ref), ref)
    report = compare_lse(torch, lse, ref, tolerance)
    report["lse_layout"] = {True: "HT", False: "BHS"}[varlen]
    return report


def _max_abs_diff(a, b) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return float((a.float() - b.float()).abs().max().item())


def _make_leftpad(torch, cache_seqlens, seqlen_q=1):
    """randint(0, cache_seqlens[i]) per batch; at least one pad >= min(limit-1, max(Sq, 64))."""
    device = cache_seqlens.device
    batch = int(cache_seqlens.shape[0])
    values = []
    longest_i, longest = 0, 0
    for i in range(batch):
        limit = int(cache_seqlens[i].item())
        if limit > longest:
            longest_i, longest = i, limit
        if limit <= 1:
            values.append(0)
        else:
            values.append(int(torch.randint(0, limit, (1,), device=device).item()))
    if longest > 1:
        need = min(longest - 1, max(int(seqlen_q), 64, longest // 4))
        if values[longest_i] < need:
            values[longest_i] = need
    return torch.tensor(values, dtype=torch.int32, device=device)


def _bwd_rtol(case: dict) -> float:
    if case["dtype"].startswith("fp8"):
        return 0.30
    if case["dtype"] == "bf16":
        return 0.25
    if case["dtype"] == "fp16":
        return 0.15
    return 1e-4


def _bwd_atol(case: dict) -> float:
    if case["dtype"].startswith("fp8"):
        return 1e-2
    if case["dtype"] == "bf16":
        return 1e-3
    if case["dtype"] == "fp16":
        return 5e-4
    return 1e-6


def _unselected_grad_abs(torch, case: dict, state: dict) -> float:
    dq, dk, dv = state["grads"]
    varlen = case["operation"] == "varlen_bwd"
    heads_q = int(case["heads_q"])
    group = _group(case)
    selected_heads = set(_selected_heads(heads_q))
    selected_kv = {head // group for head in selected_heads}
    dq_hold = dq.detach().float().abs().clone()
    dk_hold = dk.detach().float().abs().clone()
    dv_hold = dv.detach().float().abs().clone()
    if varlen:
        q_lengths = [int(x) for x in case["seqlens_q"]]
        k_lengths = [int(x) for x in case["seqlens_k"]]
        batches = set(_selected_batches(len(q_lengths)))
        q0 = 0
        k0 = 0
        for batch, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
            if batch in batches:
                for head in selected_heads:
                    for query_index in _sample_points(sq):
                        dq_hold[q0 + query_index, head] = 0
                for kv_head in selected_kv:
                    dk_hold[k0:k0 + sk, kv_head] = 0
                    dv_hold[k0:k0 + sk, kv_head] = 0
            q0 += sq
            k0 += sk
    else:
        batch_n, sq = int(case["batch"]), int(case["seqlen_q"])
        sk = int(case["seqlen_k"])
        batches = set(_selected_batches(batch_n))
        for batch in batches:
            for head in selected_heads:
                for query_index in _sample_points(sq):
                    dq_hold[batch, query_index, head] = 0
            for kv_head in selected_kv:
                dk_hold[batch, :sk, kv_head] = 0
                dv_hold[batch, :sk, kv_head] = 0
    return max(
        float(dq_hold.max().item()) if dq_hold.numel() else 0.0,
        float(dk_hold.max().item()) if dk_hold.numel() else 0.0,
        float(dv_hold.max().item()) if dv_hold.numel() else 0.0,
    )


def _grad_rel(abs_err: float, ref_scale: float, floor: float) -> float:
    return abs_err / max(ref_scale, floor)


def sampled_backward_error(torch, case: dict, state: dict) -> dict:
    """Full dQ/dK/dV on selected batch/head of the real shape, sparse query-row grad_out.

    Unselected batch/head/row gradients must be ~0. dQ, dK, dV each have their own
    rel = abs / max(|ref|, 1e-3); a shared scale must not hide a near-zero tensor.
    """
    q = state["q_ref"] if "q_ref" in state else state["q"]
    k = state["k_ref"] if "k_ref" in state else state["k"]
    v = state["v_ref"] if "v_ref" in state else state["v"]
    dq, dk, dv = state["grads"]
    grad_out = state["grad_out"]
    varlen = case["operation"] == "varlen_bwd"
    group = _group(case)
    heads_q = int(case["heads_q"])
    dq_abs = dk_abs = dv_abs = 0.0
    dq_scale = dk_scale = dv_scale = 0.0
    samples = 0
    if varlen:
        q_lengths = [int(x) for x in case["seqlens_q"]]
        k_lengths = [int(x) for x in case["seqlens_k"]]
        q_offsets = [0]
        k_offsets = [0]
        for length in q_lengths:
            q_offsets.append(q_offsets[-1] + length)
        for length in k_lengths:
            k_offsets.append(k_offsets[-1] + length)
        batches = _selected_batches(len(q_lengths))
    else:
        q_lengths = [int(case["seqlen_q"])] * int(case["batch"])
        k_lengths = [int(case["seqlen_k"])] * int(case["batch"])
        q_offsets = k_offsets = None
        batches = _selected_batches(int(case["batch"]))
    heads = _selected_heads(heads_q)
    for batch in batches:
        sq, sk = q_lengths[batch], k_lengths[batch]
        kv_acc = {}
        for head in heads:
            kv_head = head // group
            if q_offsets is None:
                q_h = q[batch, :sq, head]
                k_h = k[batch, :sk, kv_head]
                v_h = v[batch, :sk, kv_head]
                go = grad_out[batch, :sq, head]
                dq_act = dq[batch, :sq, head].float()
                dk_act = dk[batch, :sk, kv_head].float()
                dv_act = dv[batch, :sk, kv_head].float()
            else:
                q0, k0 = q_offsets[batch], k_offsets[batch]
                q_h = q[q0:q0 + sq, head]
                k_h = k[k0:k0 + sk, kv_head]
                v_h = v[k0:k0 + sk, kv_head]
                go = grad_out[q0:q0 + sq, head]
                dq_act = dq[q0:q0 + sq, head].float()
                dk_act = dk[k0:k0 + sk, kv_head].float()
                dv_act = dv[k0:k0 + sk, kv_head].float()
            dq_ref, dk_ref, dv_ref = _head_autograd(torch, q_h, k_h, v_h, go, case)
            dq_abs = max(dq_abs, float((dq_act - dq_ref).abs().max().item()))
            dq_scale = max(dq_scale, float(dq_ref.abs().max().item()))
            samples += 1
            acc = kv_acc.setdefault(kv_head, [None, None, dk_act, dv_act])
            acc[0] = dk_ref if acc[0] is None else acc[0] + dk_ref
            acc[1] = dv_ref if acc[1] is None else acc[1] + dv_ref
        for kv_head, (dk_ref, dv_ref, dk_act, dv_act) in kv_acc.items():
            dk_abs = max(dk_abs, float((dk_act - dk_ref).abs().max().item()))
            dv_abs = max(dv_abs, float((dv_act - dv_ref).abs().max().item()))
            dk_scale = max(dk_scale, float(dk_ref.abs().max().item()))
            dv_scale = max(dv_scale, float(dv_ref.abs().max().item()))
            samples += 2
    rtol, atol = _bwd_rtol(case), _bwd_atol(case)
    floor = 1e-3
    dq_rel = _grad_rel(dq_abs, dq_scale, floor)
    dk_rel = _grad_rel(dk_abs, dk_scale, floor)
    dv_rel = _grad_rel(dv_abs, dv_scale, floor)
    relative = max(dq_rel, dk_rel, dv_rel)
    maximum = max(dq_abs, dk_abs, dv_abs)
    ref_scale = max(dq_scale, dk_scale, dv_scale)
    selected_ok = bool(
        samples > 0
        and dq_rel <= rtol
        and dk_rel <= rtol
        and dv_rel <= rtol
        and dq_scale > floor
        and dk_scale > floor
        and dv_scale > floor
    )
    unselected = _unselected_grad_abs(torch, case, state)
    unselected_ok = bool(unselected <= atol)
    return {
        "selected_abs_error": maximum,
        "selected_relative_error": relative,
        "selected_ref_scale": ref_scale,
        "dq_abs_error": dq_abs,
        "dk_abs_error": dk_abs,
        "dv_abs_error": dv_abs,
        "dq_relative_error": dq_rel,
        "dk_relative_error": dk_rel,
        "dv_relative_error": dv_rel,
        "dq_ref_scale": dq_scale,
        "dk_ref_scale": dk_scale,
        "dv_ref_scale": dv_scale,
        "unselected_abs_error": unselected,
        "unselected_ok": unselected_ok,
        "samples": samples,
        "rtol": rtol,
        "atol": atol,
        "pass": bool(selected_ok and unselected_ok),
    }


def _fwd_tolerance(case: dict) -> float:
    """Per-row relative error, normalized by max(reference magnitude, 1e-3)."""
    if case["dtype"].startswith("fp8"):
        return 0.15
    return 0.04 if case["dtype"] == "bf16" else 0.02



def _combine_tolerance(case: dict) -> float:
    return 0.04 if case["dtype"] == "bf16" else 0.02


def _describe_grad_pattern(case: dict) -> dict:
    """Sparse dO values only. Q/K/V/dO stay the real full case shapes for the kernel."""
    varlen = case["operation"] == "varlen_bwd"
    heads = _selected_heads(int(case["heads_q"]))
    if varlen:
        lengths = [int(x) for x in case["seqlens_q"]]
        batches = _selected_batches(len(lengths))
        query_rows = {str(b): _sample_points(lengths[b]) for b in batches}
        dO_layout = "varlen (total_q, heads_q, head_dim_v)"
    else:
        batches = _selected_batches(int(case["batch"]))
        sq = int(case["seqlen_q"])
        query_rows = {str(b): _sample_points(sq) for b in batches}
        dO_layout = "fixed (batch, seqlen_q, heads_q, head_dim_v)"
    return {
        "kind": "sparse_query_rows_on_full_shape_dO",
        "dO_layout": dO_layout,
        "full_shape_qkv_retained": True,
        "full_shape_kernel_work_retained": True,
        "nonzero_batches": list(batches),
        "nonzero_heads": list(heads),
        "nonzero_query_rows_by_batch": query_rows,
        "nonzero_row_fill": "linspace(-0.5, 0.5, head_dim_v)",
        "timing_uses_identical_dO": True,
        "finite_does_not_replace_numerical": True,
    }


def _sparse_grad_out(torch, case: dict, shape, dtype):
    grad = torch.zeros(*shape, device="cuda", dtype=dtype)
    varlen = case["operation"] == "varlen_bwd"
    heads_q = int(case["heads_q"])
    dv = shape[-1]
    pattern = torch.linspace(-0.5, 0.5, dv, device="cuda", dtype=torch.float32).to(dtype)
    if varlen:
        lengths = [int(x) for x in case["seqlens_q"]]
        offset = 0
        for batch, sq in enumerate(lengths):
            take = batch in set(_selected_batches(len(lengths)))
            for query_index in _sample_points(sq):
                if take:
                    for head in _selected_heads(heads_q):
                        grad[offset + query_index, head].copy_(pattern)
            offset += sq
    else:
        batch, sq = int(case["batch"]), int(case["seqlen_q"])
        for b in _selected_batches(batch):
            for query_index in _sample_points(sq):
                for head in _selected_heads(heads_q):
                    grad[b, query_index, head].copy_(pattern)
    return grad


def _warmup_dispatch(torch, run, state: dict) -> None:
    """Absorb FA3 custom_op / autograd first-dispatch compile and any first-launch sync."""
    run()
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()
    state["warmup_completed"] = True
    state["warmup_note"] = (
        "one untimed public launch in build_operation; timed callable has no host/device sync"
    )


def _fixed_qkv(torch, case: dict, requires_grad: bool):
    dtype = dtype_from_name(torch, case["dtype"])
    dv = _head_dim_v(case)
    scale = float(case.get("softcap") or 4.0) / 4.0
    q = make_tensor(
        torch,
        (case["batch"], case["seqlen_q"], case["heads_q"], case["head_dim"]),
        dtype,
        requires_grad,
        scale=scale,
    )
    k = make_tensor(
        torch,
        (case["batch"], case["seqlen_k"], case.get("heads_kv", case["heads_q"]), case["head_dim"]),
        dtype,
        requires_grad,
    )
    v = make_tensor(
        torch,
        (case["batch"], case["seqlen_k"], case.get("heads_kv", case["heads_q"]), dv),
        dtype,
        requires_grad,
    )
    qv = None
    if _as_bool(case.get("qv")):
        qv = make_tensor(
            torch,
            (case["batch"], case["seqlen_q"], case["heads_q"], dv),
            dtype,
            False,
        )
    return q, k, v, qv


def _varlen_qkv(torch, case: dict, requires_grad: bool):
    dtype = dtype_from_name(torch, case["dtype"])
    qlens = [int(x) for x in case["seqlens_q"]]
    klens = [int(x) for x in case["seqlens_k"]]
    dv = _head_dim_v(case)
    scale = float(case.get("softcap") or 4.0) / 4.0
    q = make_tensor(torch, (sum(qlens), case["heads_q"], case["head_dim"]), dtype, requires_grad, scale=scale)
    k = make_tensor(
        torch,
        (sum(klens), case.get("heads_kv", case["heads_q"]), case["head_dim"]),
        dtype,
        requires_grad,
    )
    v = make_tensor(
        torch,
        (sum(klens), case.get("heads_kv", case["heads_q"]), dv),
        dtype,
        requires_grad,
    )
    qv = None
    if _as_bool(case.get("qv")):
        qv = make_tensor(torch, (sum(qlens), case["heads_q"], dv), dtype, False)
    return q, k, v, qv, cu_seqlens(torch, qlens), cu_seqlens(torch, klens)


def _build_paged_cache(torch, case: dict, dtype, cache_batch: int, seqlen_k: int, heads_kv: int, head_dim: int, dv: int):
    page_size = int(case["page_size"])
    nblocks = math.ceil(seqlen_k / page_size)
    num_blocks = max(nblocks * cache_batch * 2, nblocks * cache_batch + 1)
    k_paged = make_tensor(torch, (num_blocks, page_size, heads_kv, head_dim), dtype)
    v_paged = make_tensor(torch, (num_blocks, page_size, heads_kv, dv), dtype)
    perm = torch.randperm(num_blocks, device="cuda", dtype=torch.int32)
    page_table = perm[: cache_batch * nblocks].view(cache_batch, nblocks)
    return k_paged, v_paged, page_table


def _kvcache_inputs(torch, interface, case: dict):
    dtype = dtype_from_name(torch, case["dtype"])
    batch = int(case["batch"])
    seqlen_q = int(case["seqlen_q"])
    seqlen_k = int(case["seqlen_k"])
    heads_q = int(case["heads_q"])
    heads_kv = int(case.get("heads_kv", heads_q))
    head_dim = int(case["head_dim"])
    dv = _head_dim_v(case)
    append = _as_bool(case.get("append_kv"))
    seqlen_new = _as_int(case.get("seqlen_new"), seqlen_q if append else 0)
    use_idx = _as_bool(case.get("cache_batch_idx"))
    cache_batch = batch * 2 if use_idx else batch
    q = make_tensor(torch, (batch, seqlen_q, heads_q, head_dim), dtype)
    qv = None
    if _as_bool(case.get("qv")):
        qv = make_tensor(torch, (batch, seqlen_q, heads_q, dv), dtype)
    page_size = case.get("page_size")
    if page_size:
        k_cache, v_cache, page_table = _build_paged_cache(
            torch, case, dtype, cache_batch, seqlen_k, heads_kv, head_dim, dv
        )
        logical_k = gather_paged(k_cache, page_table, seqlen_k)
        logical_v = gather_paged(v_cache, page_table, seqlen_k)
    else:
        k_cache = make_tensor(torch, (cache_batch, seqlen_k, heads_kv, head_dim), dtype)
        v_cache = make_tensor(torch, (cache_batch, seqlen_k, heads_kv, dv), dtype)
        page_table = None
        logical_k, logical_v = k_cache, v_cache
    cache_batch_idx = None
    if use_idx:
        cache_batch_idx = torch.randperm(cache_batch, device="cuda", dtype=torch.int32)[:batch]
        logical_k = logical_k[cache_batch_idx.long()]
        logical_v = logical_v[cache_batch_idx.long()]
        if page_table is not None:
            page_table_view = page_table[cache_batch_idx.long()]
        else:
            page_table_view = None
    else:
        page_table_view = page_table
    if append:
        room = seqlen_k - seqlen_new
        if room < 0:
            raise ValueError("seqlen_k must be >= seqlen_new for append_kv")
        cache_seqlens = torch.full((batch,), room, dtype=torch.int32, device="cuda")
        k_new = make_tensor(torch, (batch, seqlen_new, heads_kv, head_dim), dtype)
        v_new = make_tensor(torch, (batch, seqlen_new, heads_kv, dv), dtype)
    else:
        cache_seqlens = torch.full((batch,), seqlen_k, dtype=torch.int32, device="cuda")
        k_new = v_new = None
        seqlen_new = 0
    raw_seqlens = case.get("cache_seqlens")
    if raw_seqlens is None and isinstance(case.get("seqlens_k"), (list, tuple)):
        raw_seqlens = case["seqlens_k"]
    if raw_seqlens is not None:
        if isinstance(raw_seqlens, int):
            cache_seqlens = torch.full((batch,), int(raw_seqlens), dtype=torch.int32, device="cuda")
        else:
            cache_seqlens = torch.tensor(list(raw_seqlens), dtype=torch.int32, device="cuda")
        if append:
            overflow = int(cache_seqlens.max().item()) + seqlen_new
            if overflow > seqlen_k:
                raise ValueError("cache_seqlens + seqlen_new exceeds seqlen_k capacity")
    cache_leftpad = None
    if _as_bool(case.get("cache_leftpad")):
        cache_leftpad = _make_leftpad(torch, cache_seqlens, seqlen_q)
        # Invalid prefix is masked; plant outliers so skipping leftpad cannot hide under atol.
        if page_table is not None:
            ps = int(page_size)
            page_table_host = page_table.cpu()
            pg_idx, off_idx = [], []
        for i in range(batch):
            pad = int(cache_leftpad[i].item())
            if pad <= 0:
                continue
            cache_i = int(cache_batch_idx[i].item()) if cache_batch_idx is not None else i
            if page_table is None:
                k_cache[cache_i, :pad].fill_(8)
                v_cache[cache_i, :pad].fill_(8)
            else:
                for col in range(pad):
                    pg_idx.append(
                        int(page_table_host[cache_i, col // ps].item())
                    )
                    off_idx.append(col % ps)
        if page_table is not None and pg_idx:
            pg_idx = torch.tensor(pg_idx, dtype=torch.long, device="cuda")
            off_idx = torch.tensor(off_idx, dtype=torch.long, device="cuda")
            k_cache[pg_idx, off_idx] = 8
            v_cache[pg_idx, off_idx] = 8
        if page_table is not None:
            logical_k = gather_paged(k_cache, page_table, seqlen_k)
            logical_v = gather_paged(v_cache, page_table, seqlen_k)
        else:
            logical_k, logical_v = k_cache, v_cache
        if cache_batch_idx is not None:
            logical_k = logical_k[cache_batch_idx.long()]
            logical_v = logical_v[cache_batch_idx.long()]
    rotary_dim = _as_int(case.get("rotary_dim"))
    if rotary_dim and not append:
        raise ValueError("FA3 rotary is only applied when appending KV")
    rotary_interleaved = _as_bool(case.get("rotary_interleaved"), True)
    cos = sin = None
    rotary_offsets = cache_seqlens
    local = _window(case) != (-1, -1)
    causal = _as_bool(case.get("causal"))
    input_snapshots = {
        "q": _clone_detached(q),
        "k": _clone_detached(k_cache),
        "v": _clone_detached(v_cache),
        "qv": _clone_detached(qv),
        "k_new": _clone_detached(k_new),
        "v_new": _clone_detached(v_new),
    }
    if rotary_dim:
        if rotary_dim % 16 != 0 or rotary_dim > head_dim:
            raise ValueError("rotary_dim must be a multiple of 16 and <= head_dim")
        seqlen_ro = max(seqlen_k, int(cache_seqlens.max().item()) + max(seqlen_q, seqlen_new) + 8)
        angle = torch.rand(seqlen_ro, rotary_dim // 2, device="cuda", dtype=torch.float32) * (2 * math.pi)
        cos = torch.cos(angle).to(dtype=torch.float32).to(dtype)
        sin = torch.sin(angle).to(dtype=torch.float32).to(dtype)
        q_ref = apply_rotary(
            input_snapshots["q"].float(),
            cos.float(),
            sin.float(),
            rotary_offsets,
            rotary_interleaved,
            same_pos=not (causal or local),
        ).to(q.dtype)
        if input_snapshots["k_new"] is not None:
            k_new_rot = apply_rotary(
                input_snapshots["k_new"].float(),
                cos.float(),
                sin.float(),
                rotary_offsets,
                rotary_interleaved,
                same_pos=False,
            ).to(k_new.dtype)
        else:
            k_new_rot = None
    else:
        q_ref = input_snapshots["q"]
        k_new_rot = k_new
    k_expected = logical_k.clone()
    v_expected = logical_v.clone()
    used_k = []
    leftpad = []
    for i in range(batch):
        start = int(cache_seqlens[i].item())
        used = start + seqlen_new
        used_k.append(used)
        pad = int(cache_leftpad[i].item()) if cache_leftpad is not None else 0
        leftpad.append(pad)
        if k_new_rot is not None:
            k_expected[i, start:start + seqlen_new].copy_(k_new_rot[i])
            v_expected[i, start:start + seqlen_new].copy_(v_new[i])
    descales = _descale(torch, case, batch)
    kwargs = _common_fwd_kwargs(case, qv=qv, descales=descales)
    kwargs.pop("deterministic", None)
    kwargs.update(
        cache_seqlens=cache_seqlens,
        rotary_interleaved=rotary_interleaved,
    )
    if cache_batch_idx is not None:
        kwargs["cache_batch_idx"] = cache_batch_idx
    if cache_leftpad is not None:
        kwargs["cache_leftpad"] = cache_leftpad
    if page_table is not None:
        kwargs["page_table"] = page_table
    if cos is not None:
        kwargs["rotary_cos"] = cos
        kwargs["rotary_sin"] = sin
    if _as_bool(case.get("reuse_scheduler_metadata")):
        kwargs["scheduler_metadata"] = interface.get_scheduler_metadata(
            batch,
            seqlen_q,
            page_table.shape[1] * int(case["page_size"]) if page_table is not None else seqlen_k,
            heads_q,
            heads_kv,
            head_dim,
            cache_seqlens,
            qkv_dtype=dtype,
            headdim_v=dv,
            cache_leftpad=cache_leftpad,
            page_size=int(page_size) if page_size else None,
            max_seqlen_k_new=seqlen_new,
            causal=causal,
            window_size=_window(case),
            attention_chunk=_as_int(case.get("attention_chunk")),
            has_softcap=float(case.get("softcap") or 0.0) > 0,
            num_splits=int(kwargs.get("num_splits", 0)),
            pack_gqa=kwargs.get("pack_gqa"),
            sm_margin=int(kwargs.get("sm_margin", 0)),
        )
    # k_before/v_before alias the F2 snapshot; do not clone cache a second time.
    state = {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "qv": qv,
        "q_ref": q_ref if q_ref is not None else input_snapshots["q"],
        "k_ref": k_expected,
        "v_ref": v_expected,
        "qv_ref": input_snapshots["qv"],
        "k_new": k_new,
        "v_new": v_new,
        "input_snapshots": input_snapshots,
        "k_new_rot": k_new_rot,
        "k_before": input_snapshots["k"],
        "v_before": input_snapshots["v"],
        "page_table": page_table,
        "page_table_view": page_table_view,
        "cache_batch_idx": cache_batch_idx,
        "cache_seqlens": cache_seqlens,
        "seqlen_new": seqlen_new,
        "used_k": used_k,
        "leftpad": leftpad,
        "descales": descales,
        "kwargs": kwargs,
        "ref_seqlen_k": seqlen_k,
        "paged": bool(page_size),
        "append": append,
    }
    return state


def _logical_after(state: dict, seqlen_k: int):
    k_cache, v_cache = state["k"], state["v"]
    idx = state.get("cache_batch_idx")
    if state["paged"]:
        table = state["page_table"]
        if idx is not None:
            table = table[idx.long()]
        k_log = gather_paged(k_cache, table, seqlen_k)
        v_log = gather_paged(v_cache, table, seqlen_k)
    else:
        k_log = k_cache if idx is None else k_cache[idx.long()]
        v_log = v_cache if idx is None else v_cache[idx.long()]
    return k_log, v_log


def build_operation(torch, interface, case: dict) -> tuple[Callable[[], object], dict]:
    reject_unreachable(case)
    operation = case["operation"]
    state: dict = {}

    if operation in {"fixed_fwd", "fixed_bwd"}:
        needs_grad = operation == "fixed_bwd"
        q, k, v, qv = _fixed_qkv(torch, case, needs_grad)
        descales = _descale(torch, case, int(case["batch"]))
        kwargs = _common_fwd_kwargs(case, qv=qv, descales=descales)
        dv = _head_dim_v(case)
        out_dtype = torch.bfloat16 if case["dtype"].startswith("fp8") else dtype_from_name(torch, case["dtype"])
        grad_out = None
        if needs_grad:
            grad_out = _sparse_grad_out(
                torch, case, (case["batch"], case["seqlen_q"], case["heads_q"], dv), out_dtype
            )

        kwargs["return_attn_probs"] = True

        def run_fixed():
            out, lse = _unpack_fwd(interface.flash_attn_func(q, k, v, **kwargs))
            if needs_grad:
                grads = torch.autograd.grad(out, (q, k, v), grad_out, retain_graph=False)
                state["grads"] = grads
            state["out"] = out
            state["lse"] = lse
            return out

        snaps = {
            "q": _clone_detached(q), "k": _clone_detached(k), "v": _clone_detached(v),
            "qv": _clone_detached(qv),
        }
        state.update(
            q=q, k=k, v=v, qv=qv,
            q_ref=snaps["q"], k_ref=snaps["k"], v_ref=snaps["v"], qv_ref=snaps["qv"],
            kwargs=kwargs, descales=descales, grad_out=grad_out,
            input_snapshots=snaps, expect_lse=True,
            kernel_work_shapes={
                "q": list(q.shape), "k": list(k.shape), "v": list(v.shape),
                "dO": None if grad_out is None else list(grad_out.shape),
            },
        )
        if needs_grad:
            state["grad_pattern"] = _describe_grad_pattern(case)
        state["snapshot_phase"] = "build_pre_warmup"
        state["snapshots_frozen"] = True
        _warmup_dispatch(torch, run_fixed, state)
        return run_fixed, state

    if operation in {"varlen_fwd", "varlen_bwd"}:
        needs_grad = operation == "varlen_bwd"
        q, k, v, qv, cu_q, cu_k = _varlen_qkv(torch, case, needs_grad)
        descales = _descale(torch, case, len(case["seqlens_q"]))
        kwargs = _common_fwd_kwargs(case, qv=qv, descales=descales)
        dv = _head_dim_v(case)
        out_dtype = torch.bfloat16 if case["dtype"].startswith("fp8") else dtype_from_name(torch, case["dtype"])
        grad_out = None
        if needs_grad:
            grad_out = _sparse_grad_out(
                torch, case, (sum(int(x) for x in case["seqlens_q"]), case["heads_q"], dv), out_dtype
            )

        kwargs["return_attn_probs"] = True

        def run_varlen():
            out, lse = _unpack_fwd(
                interface.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_q,
                    cu_k,
                    max(case["seqlens_q"]),
                    max(case["seqlens_k"]),
                    **kwargs,
                )
            )
            if needs_grad:
                grads = torch.autograd.grad(out, (q, k, v), grad_out, retain_graph=False)
                state["grads"] = grads
            state["out"] = out
            state["lse"] = lse
            return out

        snaps = {
            "q": _clone_detached(q), "k": _clone_detached(k), "v": _clone_detached(v),
            "qv": _clone_detached(qv),
        }
        state.update(
            q=q, k=k, v=v, qv=qv,
            q_ref=snaps["q"], k_ref=snaps["k"], v_ref=snaps["v"], qv_ref=snaps["qv"],
            cu_q=cu_q, cu_k=cu_k, kwargs=kwargs, descales=descales, grad_out=grad_out,
            input_snapshots=snaps, expect_lse=True,
            kernel_work_shapes={
                "q": list(q.shape), "k": list(k.shape), "v": list(v.shape),
                "dO": None if grad_out is None else list(grad_out.shape),
            },
        )
        if needs_grad:
            state["grad_pattern"] = _describe_grad_pattern(case)
        state["snapshot_phase"] = "build_pre_warmup"
        state["snapshots_frozen"] = True
        _warmup_dispatch(torch, run_varlen, state)
        return run_varlen, state

    if operation == "kvcache":
        state = _kvcache_inputs(torch, interface, case)
        kwargs = state["kwargs"]
        q, k_cache, v_cache = state["q"], state["k"], state["v"]
        k_new, v_new = state["k_new"], state["v_new"]

        kwargs["return_softmax_lse"] = True
        state["expect_lse"] = True

        def run_kvcache():
            out, lse = _unpack_fwd(interface.flash_attn_with_kvcache(q, k_cache, v_cache, k_new, v_new, **kwargs))
            state["out"] = out
            state["lse"] = lse
            return out

        state["kernel_work_shapes"] = {
            "q": list(q.shape), "k": list(k_cache.shape), "v": list(v_cache.shape),
        }
        state["snapshot_phase"] = "build_pre_warmup"
        state["snapshots_frozen"] = True
        _warmup_dispatch(torch, run_kvcache, state)
        if state.get("append"):
            state["k_after_warmup"] = k_cache.detach().clone()
            state["v_after_warmup"] = v_cache.detach().clone()
        return run_kvcache, state

    if operation == "combine":
        dtype = dtype_from_name(torch, case["dtype"])
        splits = int(case["num_splits"])
        batch = int(case["batch"])
        seqlen = int(case["seqlen_q"])
        heads = int(case["heads_q"])
        dim = _head_dim_v(case)
        out_partial = make_tensor(torch, (splits, batch, seqlen, heads, dim), torch.float32)
        lse_partial = make_tensor(
            torch, (splits, batch, heads, seqlen), torch.float32
        ).transpose(-1, -2)

        combine = getattr(interface, "flash_attn_combine", None)
        if combine is None:
            combine = torch.ops.flash_attn_3.fwd_combine

        def run_combine():
            out, lse = combine(out_partial, lse_partial, out_dtype=dtype)
            state["out"] = out
            state["lse"] = lse
            return out

        snaps = {
            "out_partial": _clone_detached(out_partial),
            "lse_partial": _clone_detached(lse_partial),
        }
        state.update(
            out_partial=out_partial,
            lse_partial=lse_partial,
            kwargs={},
            input_snapshots=snaps,
            expect_lse=True,
            kernel_work_shapes={
                "out_partial": list(out_partial.shape),
                "lse_partial": list(lse_partial.shape),
            },
        )
        state["snapshot_phase"] = "build_pre_warmup"
        state["snapshots_frozen"] = True
        _warmup_dispatch(torch, run_combine, state)
        return run_combine, state

    raise ValueError(f"unsupported operation: {operation}")


def _logical_from(state: dict, k_src, v_src, seqlen_k: int):
    idx = state.get("cache_batch_idx")
    if state["paged"]:
        table = state["page_table"]
        if idx is not None:
            table = table[idx.long()]
        return gather_paged(k_src, table, seqlen_k), gather_paged(v_src, table, seqlen_k)
    k_log = k_src if idx is None else k_src[idx.long()]
    v_log = v_src if idx is None else v_src[idx.long()]
    return k_log, v_log


def _cache_tol(case: dict, rotary: bool) -> tuple[float, float]:
    if case["dtype"].startswith("fp8"):
        return 0.10, 0.10
    if rotary:
        return 1e-3, 0.0
    return 0.0, 0.0


def _cache_snapshot_kv(state: dict):
    """Prefer the F2 input snapshot; tests may pass k_before as an alias."""
    snaps = state.get("input_snapshots") or {}
    k_src = snaps.get("k", state.get("k_before"))
    v_src = snaps.get("v", state.get("v_before"))
    return k_src, v_src


def _cache_write_error(torch, case: dict, state: dict) -> dict:
    seqlen_k = int(case["seqlen_k"])
    k_now, v_now = _logical_after(state, seqlen_k)
    k_before, v_before = _cache_snapshot_kv(state)
    if k_before is None or v_before is None:
        return {
            "append_checked": bool(state.get("append")),
            "cache_write_ok": False,
            "readonly_cache_checked": False,
            "snapshot_missing": True,
        }
    k_old, v_old = _logical_from(state, k_before, v_before, seqlen_k)
    leftpads = [int(x) for x in (state.get("leftpad") or [])]
    result = {
        "cache_leftpad": leftpads,
        "cache_leftpad_max": max(leftpads) if leftpads else 0,
        "cache_leftpad_span_ratio": (
            max((p / max(int(state["cache_seqlens"][i].item()), 1) for i, p in enumerate(leftpads)), default=0.0)
            if leftpads else 0.0
        ),
        "cache_snapshot_reused": bool(state.get("input_snapshots") and state.get("input_snapshots").get("k") is k_before),
    }
    if not state.get("append"):
        # Physical cache vs F2 snapshot (same tensors as inputs_unmodified k/v).
        readonly_k = _max_abs_diff(state.get("k"), k_before)
        readonly_v = _max_abs_diff(state.get("v"), v_before)
        logical_k = float((k_now.float() - k_old.float()).abs().max().item())
        logical_v = float((v_now.float() - v_old.float()).abs().max().item())
        tol = 1e-3 if case["dtype"].startswith("fp8") else 0.0
        ok = bool(readonly_k <= tol and readonly_v <= tol and logical_k <= tol and logical_v <= tol)
        result.update(
            append_checked=False,
            append_idempotent=None,
            append_idempotent_measured=False,
            readonly_cache_checked=True,
            readonly_cache_ok=ok,
            readonly_cache_k_max_absolute_error=max(readonly_k, logical_k),
            readonly_cache_v_max_absolute_error=max(readonly_v, logical_v),
            cache_write_ok=ok,
        )
        return result
    idemp_k = _max_abs_diff(state.get("k"), state.get("k_after_warmup"))
    idemp_v = _max_abs_diff(state.get("v"), state.get("v_after_warmup"))
    measured = state.get("k_after_warmup") is not None and state.get("v_after_warmup") is not None
    idemp_tol = 1e-3 if case["dtype"].startswith("fp8") else 0.0
    append_idempotent = bool(measured and idemp_k <= idemp_tol and idemp_v <= idemp_tol)
    result.update(
        append_idempotent=append_idempotent,
        append_idempotent_measured=bool(measured),
        append_idempotent_k_max_absolute_error=idemp_k,
        append_idempotent_v_max_absolute_error=idemp_v,
    )
    k_exp, v_exp = state["k_ref"], state["v_ref"]
    seqlens = state["cache_seqlens"]
    seqlen_new = int(state["seqlen_new"])
    k_new_max = 0.0
    v_new_max = 0.0
    old_k_max = 0.0
    old_v_max = 0.0
    for batch in range(k_now.shape[0]):
        start = int(seqlens[batch].item())
        stop = start + seqlen_new
        k_new_max = max(k_new_max, float((k_now[batch, start:stop].float() - k_exp[batch, start:stop].float()).abs().max().item()))
        v_new_max = max(v_new_max, float((v_now[batch, start:stop].float() - v_exp[batch, start:stop].float()).abs().max().item()))
        if start > 0:
            old_k_max = max(old_k_max, float((k_now[batch, :start].float() - k_old[batch, :start].float()).abs().max().item()))
            old_v_max = max(old_v_max, float((v_now[batch, :start].float() - v_old[batch, :start].float()).abs().max().item()))
        if stop < seqlen_k:
            old_k_max = max(old_k_max, float((k_now[batch, stop:].float() - k_old[batch, stop:].float()).abs().max().item()))
            old_v_max = max(old_v_max, float((v_now[batch, stop:].float() - v_old[batch, stop:].float()).abs().max().item()))
    rotary = _as_int(case.get("rotary_dim")) > 0
    k_tol, v_tol = _cache_tol(case, rotary)
    old_ok = old_k_max <= (1e-3 if case["dtype"].startswith("fp8") else 0.0) and old_v_max <= (
        1e-3 if case["dtype"].startswith("fp8") else 0.0
    )
    new_ok = k_new_max <= k_tol and v_new_max <= v_tol
    result.update(
        append_checked=True,
        new_k_max_absolute_error=k_new_max,
        new_v_max_absolute_error=v_new_max,
        untouched_k_max_absolute_error=old_k_max,
        untouched_v_max_absolute_error=old_v_max,
        rotary_checked=rotary,
        cache_write_ok=bool(new_ok and old_ok and append_idempotent),
    )
    return result


def _readonly_input_error(case: dict, state: dict) -> dict:
    snaps = state.get("input_snapshots")
    if not snaps:
        return {"inputs_unmodified_checked": False, "inputs_unmodified": None, "inputs_max_absolute_error": None}
    names = ["q", "qv", "k_new", "v_new", "out_partial", "lse_partial"]
    # Non-append kvcache is read-only: check k/v against the same F2 snapshot.
    if case["operation"] != "kvcache" or not state.get("append"):
        names.extend(["k", "v"])
    maximum = 0.0
    compared = 0
    compared_names = []
    for name in names:
        snap = snaps.get(name)
        live = state.get(name)
        if snap is None:
            continue
        maximum = max(maximum, _max_abs_diff(live, snap))
        compared += 1
        compared_names.append(name)
    ok = bool(compared > 0 and maximum == 0.0)
    return {
        "inputs_unmodified_checked": True,
        "inputs_unmodified": ok,
        "inputs_max_absolute_error": maximum,
        "inputs_compared": compared,
        "inputs_compared_names": compared_names,
    }


def _layout_ok(case: dict, state: dict) -> dict:
    out = state.get("out")
    info = {
        "out_shape": None if out is None else list(out.shape),
        "out_shape_ok": False,
        "lse_shape": None,
        "lse_shape_ok": False,
        "lse_checked": False,
        "lse_layout": None,
    }
    try:
        expected_out = _official_out_shape(case)
        expected_lse = _official_lse_shape(case)
    except (KeyError, TypeError, ValueError) as exc:
        info["layout_error"] = str(exc)
        return info
    info["out_shape_expected"] = list(expected_out)
    info["lse_shape_expected"] = list(expected_lse)
    info["out_shape_ok"] = out is not None and tuple(out.shape) == expected_out
    lse = state.get("lse")
    expect_lse = bool(state.get("expect_lse")) or case["operation"] == "combine" or lse is not None
    if lse is None:
        info["lse_checked"] = bool(expect_lse)
        info["lse_shape_ok"] = False if expect_lse else None
        info["lse_layout"] = None
    else:
        info["lse_checked"] = True
        info["lse_shape"] = list(lse.shape)
        info["lse_shape_ok"] = tuple(lse.shape) == expected_lse
        if case["operation"] == "combine":
            info["lse_layout"] = "BSH"
            partial = state.get("lse_partial")
            if partial is not None:
                info["lse_partial_shape"] = list(partial.shape)
                info["lse_partial_stride_seqlen"] = int(partial.stride(-2)) if partial.ndim >= 2 else None
                splits = int(case.get("num_splits") or partial.shape[0])
                want = (splits, int(case["batch"]), int(case["seqlen_q"]), int(case["heads_q"]))
                info["lse_partial_shape_ok"] = tuple(partial.shape) == want
                info["lse_partial_seqlen_contiguous"] = info["lse_partial_stride_seqlen"] == 1
        elif case["operation"].startswith("varlen"):
            info["lse_layout"] = "HT"
        else:
            info["lse_layout"] = "BHS"
    return info


def validate_outputs(torch, case: dict, state: dict) -> dict:
    out = state["out"]
    finite = bool(torch.isfinite(out.float()).all().item())
    nonzero = bool(out.float().abs().max().item() > 0)
    numerical_ok = False
    result = {
        "status": "fail",
        "finite": finite,
        "nonzero": nonzero,
        "maximum_absolute_output": float(out.float().abs().max().item()),
        "method": "independent numerical reference; finite is required but never sufficient",
        "warmup_completed": bool(state.get("warmup_completed")),
        "kernel_work_shapes": state.get("kernel_work_shapes"),
        "unsupported": {
            "fp8_vcol": False,
            "qkrs": False,
            "sink": False,
            "dropout": False,
            "deterministic_hdim256": False,
            "public_varlen_combine": False,
        },
    }
    result.update(_readonly_input_error(case, state))
    result.update(_layout_ok(case, state))
    layout_ready = result.get("out_shape_ok") is True
    if layout_ready and case["operation"] in {"fixed_fwd", "fixed_bwd", "varlen_fwd", "varlen_bwd", "kvcache"}:
        error, samples, relative_error = sampled_attention_error(torch, case, state)
        tolerance = _fwd_tolerance(case)
        result.update(
            sampled_reference_rows=samples,
            sampled_reference_max_absolute_error=error,
            sampled_reference_relative_error=relative_error,
            sampled_reference_relative_tolerance=tolerance,
            sampled_reference_scale_floor=1e-3,
        )
        result["method"] = (
            "full-output finite/nonzero plus sampled full-shape FP32 attention reference; "
            "finite does not replace the numerical check"
        )
        numerical_ok = bool(finite and nonzero and samples > 0 and relative_error <= tolerance)
        lse = state.get("lse")
        if result.get("lse_shape_ok") is True and lse is not None:
            lse_rep = attention_lse_error(torch, case, state)
            result.update(lse_rep)
            result["method"] = (
                "full-output finite/nonzero plus sampled full-shape FP32 attention and LSE; "
                "LSE -inf on fully masked rows is official (softmax.h sum==0); "
                "compare inf signs plus finite-row values; LSE is not required to be globally finite"
            )
            numerical_ok = bool(numerical_ok and lse_rep["lse_pass"])
        elif state.get("expect_lse"):
            result.update(
                lse_missing=True,
                lse_global_finite_required=False,
                pass_lse=False,
            )
            numerical_ok = False
    if case["operation"] in {"fixed_bwd", "varlen_bwd"}:
        grads = state.get("grads", ())
        grad_finite = len(grads) == 3 and all(
            torch.isfinite(grad.float()).all().item() and grad.float().abs().max().item() > 0 for grad in grads
        )
        result["gradient_finite_nonzero"] = bool(grad_finite)
        report = sampled_backward_error(torch, case, state)
        pattern = state.get("grad_pattern") or _describe_grad_pattern(case)
        result.update(
            selected_fullshape_grad_samples=report["samples"],
            selected_fullshape_grad_max_absolute_error=report["selected_abs_error"],
            selected_fullshape_grad_relative_error=report["selected_relative_error"],
            selected_fullshape_grad_ref_scale=report["selected_ref_scale"],
            selected_fullshape_grad_rtol=report["rtol"],
            selected_fullshape_grad_atol=report["atol"],
            dq_relative_error=report["dq_relative_error"],
            dk_relative_error=report["dk_relative_error"],
            dv_relative_error=report["dv_relative_error"],
            dq_ref_scale=report["dq_ref_scale"],
            dk_ref_scale=report["dk_ref_scale"],
            dv_ref_scale=report["dv_ref_scale"],
            unselected_grad_max_absolute_error=report["unselected_abs_error"],
            unselected_grad_ok=report["unselected_ok"],
            grad_pattern=pattern,
        )
        result["method"] = (
            "full-shape FA3 fwd+bwd kernel; sparse dO values on selected query rows; "
            "FP32 autograd dQ/dK/dV on those batch/head slices of the real full shape; "
            "unselected batch/head/row grads must be ~0; relative rtol*scale+atol; "
            "finite does not replace the numerical check"
        )
        numerical_ok = bool(numerical_ok and grad_finite and report["pass"])
    if case["operation"] == "kvcache":
        cache = _cache_write_error(torch, case, state)
        result.update(cache)
        result["method"] = (
            "full-output finite plus sampled full-shape FP32 attention reference"
            + ("; append cache slots and RoPE checked independently" if state.get("append") else "; readonly cache bit-equal")
            + "; finite does not replace the numerical check"
        )
        if not cache.get("cache_write_ok", False):
            numerical_ok = False
    if case["operation"] == "combine":
        snaps = state.get("input_snapshots") or {}
        partial = (snaps.get("out_partial") if snaps.get("out_partial") is not None else state["out_partial"]).float()
        lse_partial = (snaps.get("lse_partial") if snaps.get("lse_partial") is not None else state["lse_partial"]).float()
        ref_lse = torch.logsumexp(lse_partial, dim=0)
        scale = torch.exp(lse_partial - ref_lse)
        scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
        ref = (scale.unsqueeze(-1) * partial).sum(dim=0)
        error = float((out.float() - ref).abs().max().item())
        tolerance = _combine_tolerance(case)
        lse = state.get("lse")
        lse_tol = _lse_tolerance(case)
        lse_rep = compare_lse(torch, lse, ref_lse, lse_tol)
        result.update(lse_rep)
        result.update(
            max_absolute_error=error,
            tolerance=tolerance,
            lse_max_absolute_error=lse_rep["lse_finite_max_absolute_error"],
        )
        result["method"] = (
            "full-output finite plus independent split-combine FP32 reference including LSE; "
            "inf signs must match; finite rows compared; LSE is not required to be globally finite"
        )
        numerical_ok = bool(
            finite and nonzero and error <= tolerance and lse_rep["lse_pass"] and result.get("lse_shape_ok", False)
        )
    if result.get("inputs_unmodified_checked") and result.get("inputs_unmodified") is False:
        numerical_ok = False
    if result.get("out_shape_ok") is not True:
        numerical_ok = False
    if result.get("lse_checked") and result.get("lse_shape_ok") is not True:
        numerical_ok = False
    result["snapshots_frozen"] = bool(state.get("snapshots_frozen"))
    result["snapshot_phase"] = state.get("snapshot_phase")
    result["validate_resets_live_inputs"] = False
    result["numerical_checked"] = bool(numerical_ok)
    result["status"] = "pass" if numerical_ok else "fail"
    return result


def _cpu_attn_out(torch, q, k, v, case, *, used_k=None, leftpad=0):
    group = _group(case)
    scale = _softmax_scale(case)
    batch, sq, hq, _ = q.shape
    sk = k.shape[1]
    out = torch.zeros(batch, sq, hq, v.shape[-1])
    for b in range(batch):
        uk = sk if used_k is None else int(used_k[b] if not isinstance(used_k, int) else used_k)
        pad = 0 if leftpad is None else int(leftpad[b] if not isinstance(leftpad, int) else leftpad)
        for h in range(hq):
            kv = h // group
            keys = k[b, :sk, kv].float()
            values = v[b, :sk, kv].float()
            for qi in range(sq):
                scores = torch.mv(keys, q[b, qi, h].float()) * scale
                allowed = _row_mask(torch, qi, sq, sk, case, used_k=uk, leftpad=pad, device=q.device)
                scores = scores.masked_fill(~allowed, float("-inf"))
                if bool(allowed.any().item()):
                    out[b, qi, h] = torch.mv(values.transpose(0, 1), torch.softmax(scores, dim=0))
    return out


def _failure_injections(T) -> dict:
    """CPU proofs that the F2-F9 bugs now fail closed. No GPU."""
    proofs = {}

    # F2: live Q mutated vs pre-launch snapshot must fail both alias-ref and unmodified check.
    T.manual_seed(1)
    case_f2 = {
        "operation": "fixed_fwd", "dtype": "fp32", "head_dim": 8, "head_dim_v": 8,
        "heads_q": 1, "heads_kv": 1, "batch": 1, "seqlen_q": 4, "seqlen_k": 4,
    }
    q = T.randn(1, 4, 1, 8)
    k = T.randn(1, 4, 1, 8)
    v = T.randn(1, 4, 1, 8)
    snap_q, snap_k, snap_v = q.clone(), k.clone(), v.clone()
    out_orig = _cpu_attn_out(T, snap_q, snap_k, snap_v, case_f2)
    q_mut = q * 2
    out_mut = _cpu_attn_out(T, q_mut, k, v, case_f2)
    alias_state = {
        "q": q_mut, "k": k, "v": v, "q_ref": q_mut, "k_ref": k, "v_ref": v,
        "out": out_mut, "descales": (None, None, None),
        "input_snapshots": {"q": snap_q, "k": snap_k, "v": snap_v},
    }
    snap_state = {
        "q": q_mut, "k": k, "v": v, "q_ref": snap_q, "k_ref": snap_k, "v_ref": snap_v,
        "out": out_mut, "descales": (None, None, None),
        "input_snapshots": {"q": snap_q, "k": snap_k, "v": snap_v},
    }
    alias_err, _, _ = sampled_attention_error(T, case_f2, alias_state)
    snap_err, _, snap_relative = sampled_attention_error(T, case_f2, snap_state)
    alias_val = validate_outputs(T, case_f2, alias_state)
    snap_val = validate_outputs(T, case_f2, snap_state)
    if snap_val["status"] == "pass" or snap_val.get("inputs_unmodified") is not False:
        raise AssertionError(("F2 snapshot must fail", snap_val, snap_err, alias_err))
    if alias_err > 1e-5:
        raise AssertionError("F2 control: alias ref should self-pass numerically")
    if snap_relative <= _fwd_tolerance(case_f2):
        raise AssertionError(("F2 snapshot numerical must exceed tol", snap_err))
    proofs["F2"] = {
        "alias_ref_error": alias_err,
        "snapshot_ref_error": snap_err,
        "validate_status": snap_val["status"],
        "inputs_unmodified": snap_val["inputs_unmodified"],
    }

    # F3: substantial leftpad; ignoring it must exceed bf16 fwd tol.
    T.manual_seed(2)
    case_f3 = {
        "operation": "kvcache", "dtype": "bf16", "head_dim": 8, "head_dim_v": 8,
        "heads_q": 1, "heads_kv": 1, "batch": 1, "seqlen_q": 16, "seqlen_k": 256, "causal": True,
    }
    q = T.randn(1, 16, 1, 8)
    k = T.randn(1, 256, 1, 8)
    v = T.randn(1, 256, 1, 8)
    leftpad = 64
    used = 256
    k[:, :leftpad].fill_(8)
    v[:, :leftpad].fill_(8)
    out_pad = _cpu_attn_out(T, q, k, v, case_f3, used_k=used, leftpad=leftpad)
    out_ignore = _cpu_attn_out(T, q, k, v, case_f3, used_k=used, leftpad=0)
    ignore_err, n_f3, ignore_relative = sampled_attention_error(
        T, case_f3,
        {"q_ref": q, "k_ref": k, "v_ref": v, "out": out_ignore, "descales": (None, None, None),
         "used_k": [used], "leftpad": [leftpad], "ref_seqlen_k": 256},
    )
    pad_err, _, _ = sampled_attention_error(
        T, case_f3,
        {"q_ref": q, "k_ref": k, "v_ref": v, "out": out_pad, "descales": (None, None, None),
         "used_k": [used], "leftpad": [leftpad], "ref_seqlen_k": 256},
    )
    fake_seqlens = T.tensor([used], dtype=T.int32)
    made = _make_leftpad(T, fake_seqlens, seqlen_q=16)
    if int(made.max().item()) < min(used - 1, max(16, 64)):
        raise AssertionError(("F3 leftpad span", made, used))
    ignore_val = validate_outputs(
        T, case_f3,
        {"q_ref": q, "k_ref": k, "v_ref": v, "out": out_ignore, "descales": (None, None, None),
         "used_k": [used], "leftpad": [leftpad], "ref_seqlen_k": 256, "k": k, "v": v,
         "input_snapshots": {"k": k.clone(), "v": v.clone(), "q": q.clone()},
         "cache_seqlens": fake_seqlens, "append": False, "paged": False, "cache_batch_idx": None},
    )
    if pad_err > 1e-5:
        raise AssertionError(("F3 correct leftpad ref", pad_err))
    if ignore_relative <= _fwd_tolerance(case_f3) or ignore_val["status"] == "pass":
        raise AssertionError(("F3 ignore-leftpad must fail", ignore_err, ignore_val["status"], n_f3))
    tiny_case = dict(case_f3)
    out_pad1 = _cpu_attn_out(T, q, k, v, tiny_case, used_k=used, leftpad=1)
    tiny_err = float((out_pad1 - out_ignore).abs().max().item())
    proofs["F3"] = {
        "leftpad": leftpad,
        "span_ratio": leftpad / used,
        "ignore_error": ignore_err,
        "correct_error": pad_err,
        "leftpad1_error": tiny_err,
        "fwd_tol": _fwd_tolerance(case_f3),
        "validate_status": ignore_val["status"],
        "generated_leftpad": int(made.max().item()),
    }

    # F4: all-zero grads and unselected-head garbage must fail relative/zero checks.
    T.manual_seed(3)
    case_f4 = {
        "operation": "fixed_bwd", "dtype": "bf16", "head_dim": 8, "head_dim_v": 8,
        "heads_q": 4, "heads_kv": 4, "batch": 3, "seqlen_q": 8, "seqlen_k": 9, "causal": False,
    }
    q = T.randn(3, 8, 4, 8)
    k = T.randn(3, 9, 4, 8)
    v = T.randn(3, 9, 4, 8)
    go = _sparse_grad_out.__wrapped__ if False else None
    go = T.zeros(3, 8, 4, 8)
    for b in _selected_batches(3):
        for qi in _sample_points(8):
            for h in _selected_heads(4):
                go[b, qi, h] = T.linspace(-0.5, 0.5, 8)
    # true grads via per-head autograd assembled
    dq = T.zeros_like(q)
    dk = T.zeros_like(k)
    dv = T.zeros_like(v)
    for b in _selected_batches(3):
        for h in _selected_heads(4):
            dqr, dkr, dvr = _head_autograd(T, q[b, :, h], k[b, :, h], v[b, :, h], go[b, :, h], case_f4)
            dq[b, :, h] = dqr
            dk[b, :, h] = dk[b, :, h] + dkr
            dv[b, :, h] = dv[b, :, h] + dvr
    good = sampled_backward_error(
        T, case_f4,
        {"q_ref": q, "k_ref": k, "v_ref": v, "grads": (dq, dk, dv), "grad_out": go},
    )
    if not good["pass"]:
        raise AssertionError(("F4 good grads must pass", good))
    zeros = sampled_backward_error(
        T, case_f4,
        {"q_ref": q, "k_ref": k, "v_ref": v, "grads": (T.zeros_like(dq), T.zeros_like(dk), T.zeros_like(dv)), "grad_out": go},
    )
    if zeros["pass"] or zeros["dq_relative_error"] < 0.9 or zeros["dk_relative_error"] < 0.9 or zeros["dv_relative_error"] < 0.9:
        raise AssertionError(("F4 all-zero must fail per-tensor rel", zeros))
    garbage = (dq.clone(), dk.clone(), dv.clone())
    garbage[0][:, :, 1].fill_(50.0)  # unselected head 1
    bad_unsel = sampled_backward_error(
        T, case_f4,
        {"q_ref": q, "k_ref": k, "v_ref": v, "grads": garbage, "grad_out": go},
    )
    if bad_unsel["unselected_ok"] or bad_unsel["pass"]:
        raise AssertionError(("F4 unselected garbage must fail", bad_unsel))
    dv_only = sampled_backward_error(
        T, case_f4,
        {"q_ref": q, "k_ref": k, "v_ref": v, "grads": (T.zeros_like(dq), T.zeros_like(dk), dv.clone()), "grad_out": go},
    )
    shared_would_pass = bool(
        dv_only["dq_abs_error"] <= dv_only["rtol"] * dv_only["dv_ref_scale"] + dv_only["atol"]
    )
    if dv_only["pass"] or dv_only["dq_relative_error"] < 0.9 or dv_only["dk_relative_error"] < 0.9:
        raise AssertionError(("F4 dV-only must fail per-tensor rel", dv_only, shared_would_pass))
    zero_val = validate_outputs(
        T, case_f4,
        {"q_ref": q, "k_ref": k, "v_ref": v, "out": _cpu_attn_out(T, q, k, v, case_f4),
         "grads": (T.zeros_like(dq), T.zeros_like(dk), T.zeros_like(dv)), "grad_out": go,
         "descales": (None, None, None)},
    )
    if zero_val["status"] == "pass":
        raise AssertionError(("F4 validate all-zero", zero_val))
    proofs["F4"] = {
        "good_pass": good["pass"],
        "allzero_abs": zeros["selected_abs_error"],
        "allzero_relative": zeros["selected_relative_error"],
        "allzero_dq_rel": zeros["dq_relative_error"],
        "allzero_dk_rel": zeros["dk_relative_error"],
        "allzero_dv_rel": zeros["dv_relative_error"],
        "allzero_pass": zeros["pass"],
        "dv_only_pass": dv_only["pass"],
        "dv_only_dq_rel": dv_only["dq_relative_error"],
        "dv_only_shared_scale_would_pass": shared_would_pass,
        "unselected_garbage_abs": bad_unsel["unselected_abs_error"],
        "unselected_ok": bad_unsel["unselected_ok"],
        "validate_allzero_status": zero_val["status"],
        "old_abs_tol_would_pass": bool(zeros["selected_abs_error"] <= 0.20),
        "shared_scale_hides_dq": bool(0.022 <= 0.25 * 0.507 + 0.001 and _grad_rel(0.022, 0.022, 1e-3) > 0.25),
    }

    # F5: correct out + zero LSE must fail; wrong LSE shape must fail.
    splits, b, s, h, d = 3, 2, 5, 4, 8
    partial = T.randn(splits, b, s, h, d)
    lse_p = T.randn(splits, b, s, h)
    ref_lse = T.logsumexp(lse_p, dim=0)
    scale = T.exp(lse_p - ref_lse)
    scale = T.where(T.isfinite(scale), scale, T.zeros_like(scale))
    ref = (scale.unsqueeze(-1) * partial).sum(0)
    case_c = {"operation": "combine", "dtype": "fp32", "head_dim": d, "head_dim_v": d, "heads_q": h, "batch": b, "seqlen_q": s}
    zero_lse = validate_outputs(T, case_c, {"out": ref.clone(), "lse": T.zeros_like(ref_lse), "out_partial": partial, "lse_partial": lse_p})
    if zero_lse["status"] == "pass":
        raise AssertionError(("F5 zero LSE must fail", zero_lse))
    bad_shape = validate_outputs(
        T, case_c,
        {"out": ref.clone(), "lse": ref_lse.transpose(1, 2).contiguous(), "out_partial": partial, "lse_partial": lse_p},
    )
    if bad_shape["status"] == "pass" or bad_shape.get("lse_shape_ok") is not False:
        raise AssertionError(("F5 LSE shape must fail", bad_shape))
    proofs["F5"] = {
        "zero_lse_status": zero_lse["status"],
        "zero_lse_error": zero_lse["lse_max_absolute_error"],
        "bad_shape_status": bad_shape["status"],
        "bad_shape_ok": bad_shape.get("lse_shape_ok"),
    }

    # F5 attention LSE + causal Q>K fully masked rows (softmax.h:151 LSE=-inf).
    T.manual_seed(5)
    case_qk = {
        "operation": "fixed_fwd", "dtype": "fp32", "head_dim": 8, "head_dim_v": 8,
        "heads_q": 1, "heads_kv": 1, "batch": 1, "seqlen_q": 5, "seqlen_k": 2, "causal": True,
    }
    q = T.randn(1, 5, 1, 8)
    k = T.randn(1, 2, 1, 8)
    v = T.randn(1, 2, 1, 8)
    out_qk = _cpu_attn_out(T, q, k, v, case_qk)
    lse_qk = T.zeros(1, 1, 5)
    masked_rows = 0
    for qi in range(5):
        allowed = _row_mask(T, qi, 5, 2, case_qk, device=q.device)
        if not bool(allowed.any().item()):
            lse_qk[0, 0, qi] = float("-inf")
            masked_rows += 1
        else:
            scores = T.mv(k[0, :, 0].float(), q[0, qi, 0].float()) * (8 ** -0.5)
            scores = scores.masked_fill(~allowed, float("-inf"))
            lse_qk[0, 0, qi] = T.logsumexp(scores, dim=0)
    if masked_rows < 1:
        raise AssertionError("causal Q>K must produce fully masked rows")
    if bool(T.isfinite(lse_qk).all().item()):
        raise AssertionError("official LSE is not globally finite on fully masked rows")
    state_qk = {
        "q_ref": q, "k_ref": k, "v_ref": v, "out": out_qk, "lse": lse_qk,
        "descales": (None, None, None), "expect_lse": True,
    }
    good_qk = validate_outputs(T, case_qk, state_qk)
    if good_qk["status"] != "pass" or good_qk.get("lse_inf_sign_ok") is not True:
        raise AssertionError(("causal Q>K official -inf LSE must pass", good_qk))
    if good_qk.get("lse_global_finite_required") is not False:
        raise AssertionError("must not globally require LSE finite")
    wrong_sign = lse_qk.clone()
    wrong_sign[T.isneginf(wrong_sign)] = 0
    bad_qk = validate_outputs(T, case_qk, {**state_qk, "lse": wrong_sign})
    if bad_qk["status"] == "pass" or bad_qk.get("lse_inf_sign_ok") is not False:
        raise AssertionError(("zero instead of -inf must fail inf-sign", bad_qk))
    wrong_finite = lse_qk.clone()
    finite_idx = T.isfinite(wrong_finite)
    if bool(finite_idx.any().item()):
        wrong_finite[finite_idx] = wrong_finite[finite_idx] + 3
    bad_fin = validate_outputs(T, case_qk, {**state_qk, "lse": wrong_finite})
    if bad_fin["status"] == "pass":
        raise AssertionError(("wrong finite LSE must fail", bad_fin))
    # varlen causal Q>K (fa3_varlen_fwd_02 / fa3_bwd_gqa_varlen_01 shape class)
    case_vl = {
        "operation": "varlen_fwd", "dtype": "fp32", "head_dim": 8, "head_dim_v": 8,
        "heads_q": 2, "heads_kv": 1, "seqlens_q": [5], "seqlens_k": [2], "causal": True,
    }
    qv = T.randn(5, 2, 8)
    kv = T.randn(2, 1, 8)
    vv = T.randn(2, 1, 8)
    out_vl = T.zeros(5, 2, 8)
    lse_vl = T.zeros(2, 5)
    scale = 8 ** -0.5
    for qi in range(5):
        allowed = _row_mask(T, qi, 5, 2, case_vl, device=qv.device)
        for h in range(2):
            if not bool(allowed.any().item()):
                lse_vl[h, qi] = float("-inf")
                continue
            scores = T.mv(kv[:, 0].float(), qv[qi, h].float()) * scale
            scores = scores.masked_fill(~allowed, float("-inf"))
            w = T.softmax(scores, dim=0)
            out_vl[qi, h] = (vv[:, 0].float() * w[:, None]).sum(0)
            lse_vl[h, qi] = T.logsumexp(scores, dim=0)
    good_vl = validate_outputs(
        T, case_vl,
        {"q_ref": qv, "k_ref": kv, "v_ref": vv, "out": out_vl, "lse": lse_vl,
         "descales": (None, None, None), "expect_lse": True},
    )
    if good_vl["status"] != "pass" or good_vl.get("lse_neg_inf_rows", 0) < 1:
        raise AssertionError(("varlen causal Q>K LSE", good_vl, lse_vl))
    proofs["F5"]["attention_lse_checked"] = True
    proofs["F5"]["causal_qk_status"] = good_qk["status"]
    proofs["F5"]["causal_qk_neg_inf_rows"] = good_qk.get("lse_neg_inf_rows")
    proofs["F5"]["causal_qk_global_finite"] = bool(T.isfinite(lse_qk).all().item())
    proofs["F5"]["wrong_inf_sign_status"] = bad_qk["status"]
    proofs["F5"]["wrong_finite_lse_status"] = bad_fin["status"]
    proofs["F5"]["varlen_causal_qk_status"] = good_vl["status"]
    proofs["F5"]["varlen_neg_inf_rows"] = good_vl.get("lse_neg_inf_rows")

    # F6: append_idempotent is measured, not constant True.
    T.manual_seed(4)
    k_cache = T.randn(1, 8, 1, 4)
    v_cache = T.randn(1, 8, 1, 4)
    k_before = k_cache.clone()
    v_before = v_cache.clone()
    k_exp = k_cache.clone()
    v_exp = v_cache.clone()
    k_exp[0, 4:6].fill_(3)
    v_exp[0, 4:6].fill_(3)
    k_now = k_exp.clone()
    v_now = v_exp.clone()
    k_now[0, 5].fill_(9)  # second write diverged
    case_f6 = {"operation": "kvcache", "dtype": "fp32", "seqlen_k": 8, "rotary_dim": 0, "head_dim": 4, "head_dim_v": 4, "heads_q": 1, "batch": 1, "seqlen_q": 2}
    snaps_f6 = {"k": k_before, "v": v_before}
    state_f6 = {
        "append": True, "paged": False, "cache_batch_idx": None,
        "k": k_now, "v": v_now, "k_ref": k_exp, "v_ref": v_exp,
        "input_snapshots": snaps_f6,
        "k_before": snaps_f6["k"], "v_before": snaps_f6["v"],
        "k_after_warmup": k_exp.clone(), "v_after_warmup": v_exp.clone(),
        "cache_seqlens": T.tensor([4], dtype=T.int32), "seqlen_new": 2, "leftpad": [0],
    }
    cache_f6 = _cache_write_error(T, case_f6, state_f6)
    if cache_f6["append_idempotent"] is True or cache_f6["cache_write_ok"]:
        raise AssertionError(("F6 non-idempotent must fail", cache_f6))
    state_ok = dict(state_f6)
    state_ok["k"] = k_exp.clone()
    state_ok["v"] = v_exp.clone()
    cache_ok = _cache_write_error(T, case_f6, state_ok)
    if cache_ok["append_idempotent"] is not True:
        raise AssertionError(("F6 matching second write", cache_ok))
    proofs["F6"] = {
        "diverged_idempotent": cache_f6["append_idempotent"],
        "diverged_k_err": cache_f6["append_idempotent_k_max_absolute_error"],
        "matched_idempotent": cache_ok["append_idempotent"],
        "measured": cache_f6["append_idempotent_measured"],
    }

    # F7: chunk+leftpad matches official test_util construct_chunk_mask; old used_k align differs.
    sq, sk, chunk, used, pad = 16, 64, 8, 64, 5
    case_f7 = {"operation": "kvcache", "dtype": "fp32", "head_dim": 8, "causal": False, "attention_chunk": chunk, "window_size": [-1, -1]}
    differ_old = 0
    differ_new = 0
    for qi in range(sq):
        got = _row_mask(T, qi, sq, sk, case_f7, used_k=used, leftpad=pad, device=T.device("cpu"))
        n_valid = used - pad
        aligned = qi + n_valid - sq
        base = aligned - (aligned % chunk)
        official = T.zeros(sk, dtype=T.bool)
        for c in range(sk):
            if c < pad or c >= used:
                official[c] = False
                continue
            shifted = c - pad
            official[c] = (base <= shifted) and (shifted < base + chunk)
        old_aligned = qi + used - sq
        old_base = old_aligned - (old_aligned % chunk)
        old = (T.arange(sk) >= old_base) & (T.arange(sk) < old_base + chunk) & (T.arange(sk) < used) & (T.arange(sk) >= pad)
        if not bool((got == official).all().item()):
            differ_new += 1
        if not bool((old == official).all().item()):
            differ_old += 1
    if differ_new != 0:
        raise AssertionError(("F7 new mask must match official", differ_new))
    if differ_old == 0:
        raise AssertionError("F7 control: old mask should differ from official")
    proofs["F7"] = {"old_mismatch_rows": differ_old, "new_mismatch_rows": differ_new, "rows": sq}

    # F8: official out/LSE shapes; non-append cache mutation vs F2 snapshot must fail.
    case_fixed = {
        "operation": "fixed_fwd", "dtype": "fp32", "head_dim": 8, "head_dim_v": 8,
        "heads_q": 2, "heads_kv": 2, "batch": 3, "seqlen_q": 5, "seqlen_k": 7,
    }
    good_out = T.randn(3, 5, 2, 8)
    good_lse = T.randn(3, 2, 5)  # (B,H,S)
    q = T.randn(3, 5, 2, 8)
    k = T.randn(3, 7, 2, 8)
    v = T.randn(3, 7, 2, 8)
    good_attn = _cpu_attn_out(T, q, k, v, case_fixed)
    layout_ok = _layout_ok(case_fixed, {"out": good_attn, "lse": good_lse, "expect_lse": True})
    if not layout_ok["out_shape_ok"] or not layout_ok["lse_shape_ok"] or layout_ok["lse_layout"] != "BHS":
        raise AssertionError(("F8 good fixed layout", layout_ok))
    bad_out = validate_outputs(
        T, case_fixed,
        {"q_ref": q, "k_ref": k, "v_ref": v, "out": good_attn.transpose(1, 2).contiguous(),
         "lse": good_lse, "descales": (None, None, None), "expect_lse": True},
    )
    if bad_out["status"] == "pass" or bad_out.get("out_shape_ok") is not False:
        raise AssertionError(("F8 transposed out must fail", bad_out))
    # Official combine LSE is (B,S,H); (B,H,S) must fail.
    bad_lse = validate_outputs(
        T, case_fixed,
        {"q_ref": q, "k_ref": k, "v_ref": v, "out": good_attn, "lse": T.randn(3, 5, 2),
         "descales": (None, None, None), "expect_lse": True},
    )
    if bad_lse["status"] == "pass" or bad_lse.get("lse_shape_ok") is not False:
        raise AssertionError(("F8 BSH lse on fixed must fail", bad_lse))
    missing_lse = validate_outputs(
        T, case_fixed,
        {"q_ref": q, "k_ref": k, "v_ref": v, "out": good_attn,
         "descales": (None, None, None), "expect_lse": True},
    )
    if missing_lse["status"] == "pass" or missing_lse.get("lse_shape_ok") is not False:
        raise AssertionError(("F8 missing lse must fail when expected", missing_lse))
    case_varlen = {
        "operation": "varlen_fwd", "dtype": "fp32", "head_dim": 8, "head_dim_v": 8,
        "heads_q": 2, "heads_kv": 2, "seqlens_q": [3, 5], "seqlens_k": [4, 6],
    }
    varlen_out = T.randn(8, 2, 8)
    varlen_lse = T.randn(2, 8)  # (H,Tq)
    varlen_layout = _layout_ok(case_varlen, {"out": varlen_out, "lse": varlen_lse, "expect_lse": True})
    if not varlen_layout["out_shape_ok"] or not varlen_layout["lse_shape_ok"] or varlen_layout["lse_layout"] != "HT":
        raise AssertionError(("F8 varlen layout", varlen_layout))
    case_cache = {
        "operation": "kvcache", "dtype": "fp32", "head_dim": 4, "head_dim_v": 4,
        "heads_q": 1, "heads_kv": 1, "batch": 1, "seqlen_q": 2, "seqlen_k": 8,
    }
    cache_k = T.randn(1, 8, 1, 4)
    cache_v = T.randn(1, 8, 1, 4)
    snap_k, snap_v = cache_k.clone(), cache_v.clone()
    cache_k[0, 0].fill_(7)
    cache_out = T.randn(1, 2, 1, 4)
    cache_lse = T.randn(1, 1, 2)
    mutated = validate_outputs(
        T, case_cache,
        {"q_ref": T.randn(1, 2, 1, 4), "k_ref": snap_k, "v_ref": snap_v, "out": cache_out,
         "lse": cache_lse, "descales": (None, None, None), "expect_lse": True,
         "k": cache_k, "v": cache_v, "input_snapshots": {"k": snap_k, "v": snap_v},
         "cache_seqlens": T.tensor([8], dtype=T.int32), "append": False, "paged": False,
         "cache_batch_idx": None, "leftpad": [0]},
    )
    if mutated["status"] == "pass" or mutated.get("readonly_cache_ok") is not False or mutated.get("inputs_unmodified") is not False:
        raise AssertionError(("F8 mutated non-append cache must fail", mutated))
    const_pass = mutated.get("append_idempotent")
    if const_pass is True:
        raise AssertionError("F8 non-append must not constant-pass append_idempotent")
    combine_case = {"operation": "combine", "dtype": "fp32", "head_dim": 8, "head_dim_v": 8, "heads_q": 4, "batch": 2, "seqlen_q": 5, "num_splits": 3}
    combine_layout = _layout_ok(
        combine_case,
        {"out": T.randn(2, 5, 4, 8), "lse": T.randn(2, 5, 4), "expect_lse": True,
         "lse_partial": T.randn(3, 2, 4, 5).transpose(-1, -2)},
    )
    if not combine_layout["out_shape_ok"] or not combine_layout["lse_shape_ok"] or combine_layout["lse_layout"] != "BSH":
        raise AssertionError(("F8 combine layout", combine_layout))
    if combine_layout.get("lse_partial_seqlen_contiguous") is not True:
        raise AssertionError(("F8 combine lse_partial stride", combine_layout))
    proofs["F8"] = {
        "fixed_out_shape_ok": layout_ok["out_shape_ok"],
        "fixed_lse_layout": layout_ok["lse_layout"],
        "transposed_out_status": bad_out["status"],
        "bsh_lse_on_fixed_status": bad_lse["status"],
        "missing_lse_status": missing_lse["status"],
        "varlen_lse_layout": varlen_layout["lse_layout"],
        "mutated_cache_status": mutated["status"],
        "mutated_readonly_ok": mutated.get("readonly_cache_ok"),
        "nonappend_idempotent_field": const_pass,
        "combine_lse_layout": combine_layout["lse_layout"],
        "combine_partial_stride": combine_layout.get("lse_partial_stride_seqlen"),
    }

    # validate compares only: mutated live Q must stay mutated (no snapshot copy-back).
    case_nr = {
        "operation": "fixed_fwd", "dtype": "fp32", "head_dim": 8, "head_dim_v": 8,
        "heads_q": 1, "heads_kv": 1, "batch": 1, "seqlen_q": 4, "seqlen_k": 4,
    }
    live_q = T.randn(1, 4, 1, 8)
    live_k = T.randn(1, 4, 1, 8)
    live_v = T.randn(1, 4, 1, 8)
    snap_q, snap_k, snap_v = live_q.clone(), live_k.clone(), live_v.clone()
    live_q.mul_(3)
    before_q = live_q.clone()
    nr_out = _cpu_attn_out(T, snap_q, snap_k, snap_v, case_nr)
    nr_val = validate_outputs(
        T, case_nr,
        {"q": live_q, "k": live_k, "v": live_v, "q_ref": snap_q, "k_ref": snap_k, "v_ref": snap_v,
         "out": nr_out, "descales": (None, None, None),
         "input_snapshots": {"q": snap_q, "k": snap_k, "v": snap_v},
         "snapshots_frozen": True, "snapshot_phase": "build_pre_warmup"},
    )
    if float((live_q - before_q).abs().max().item()) != 0.0:
        raise AssertionError("validate must not reset live inputs")
    if nr_val.get("validate_resets_live_inputs") is not False:
        raise AssertionError(nr_val)
    if nr_val["status"] == "pass" or nr_val.get("inputs_unmodified") is not False:
        raise AssertionError(("mutated live Q must fail unmodified", nr_val))
    proofs["validate_no_reset"] = {
        "live_q_unchanged": True,
        "status": nr_val["status"],
        "inputs_unmodified": nr_val.get("inputs_unmodified"),
    }

    # F9: a tiny nonzero output must not pass when the real attention is small.
    T.manual_seed(7)
    case_f9 = {"operation": "fixed_fwd", "dtype": "bf16", "batch": 1,
               "seqlen_q": 2, "seqlen_k": 2048, "heads_q": 2, "heads_kv": 2,
               "head_dim": 64, "head_dim_v": 64}
    q = T.randn(1, 2, 2, 64) * 0.5
    k = T.randn(1, 2048, 2, 64) * 0.5
    v = T.randn_like(k) * 0.5
    scores = T.einsum("bqhd,bkhd->bhqk", q, k) / 8
    state = {"q": q, "k": k, "v": v,
             "q_ref": q.clone(), "k_ref": k.clone(), "v_ref": v.clone(),
             "out": T.full_like(q, 1e-6), "lse": T.logsumexp(scores, -1),
             "input_snapshots": {"q": q.clone(), "k": k.clone(), "v": v.clone()}}
    nearzero = validate_outputs(T, case_f9, state)
    if nearzero["status"] != "fail" or not nearzero["lse_pass"] or not nearzero["inputs_unmodified"]:
        raise AssertionError(("F9 nearzero must fail numerically", nearzero))
    state["out"] = T.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v)
    correct = validate_outputs(T, case_f9, state)
    if correct["status"] != "pass":
        raise AssertionError(("F9 correct output must pass", correct))
    proofs["F9"] = {"nearzero_status": nearzero["status"],
                    "relative_error": nearzero["sampled_reference_relative_error"],
                    "relative_tolerance": nearzero["sampled_reference_relative_tolerance"],
                    "correct_output_status": correct["status"]}

    return proofs


def _self_check() -> dict:
    """CPU checks of reference math and reject gates. No FA3 kernel."""
    import torch as T

    # unreachable mechanisms
    base = {
        "operation": "fixed_fwd",
        "dtype": "bf16",
        "head_dim": 64,
        "heads_q": 4,
        "heads_kv": 2,
        "batch": 1,
        "seqlen_q": 8,
        "seqlen_k": 8,
    }
    for bad in (
        {**base, "mechanisms": ["dropout"]},
        {**base, "mechanisms": ["fp8vcol"]},
        {**base, "primary_mechanism": "qkrs"},
        {**base, "learnable_sink": True},
        {**base, "operation": "combine", "varlen": True, "num_splits": 2, "mechanisms": []},
        {**base, "operation": "fixed_bwd", "qv": True},
        {**base, "operation": "fixed_bwd", "attention_chunk": 64},
        {**base, "operation": "fixed_bwd", "head_dim": 256, "deterministic": True},
        {**base, "operation": "fixed_bwd", "dtype": "fp8_e4m3"},
        {**base, "qv": True, "dtype": "fp8_e4m3"},
        {**base, "qv": True, "head_dim": 128},
        {**base, "operation": "kvcache", "reuse_scheduler_metadata": True} if False else {**base, "reuse_scheduler_metadata": True},
    ):
        try:
            reject_unreachable(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject {bad}")

    # sampled attention matches batched einsum
    T.manual_seed(0)
    case = {
        "operation": "fixed_fwd",
        "dtype": "fp32",
        "head_dim": 8,
        "head_dim_v": 8,
        "heads_q": 4,
        "heads_kv": 2,
        "batch": 2,
        "seqlen_q": 6,
        "seqlen_k": 7,
        "causal": True,
        "window_size": [3, 0],
        "attention_chunk": 4,
        "softcap": 15.0,
        "qv": True,
    }
    q = T.randn(2, 6, 4, 8)
    k = T.randn(2, 7, 2, 8)
    v = T.randn(2, 7, 2, 8)
    qv = T.randn(2, 6, 4, 8)
    k_rep = k.repeat_interleave(2, dim=2)
    v_rep = v.repeat_interleave(2, dim=2)
    scale = (8 + 8) ** -0.5
    scores = T.einsum("bthd,bshd->bhts", q, k_rep) * scale
    scores = scores + T.einsum("bthd,bshd->bhts", qv, v_rep) * scale
    scores = T.tanh(scores / 15.0) * 15.0
    mask = T.stack(
        [
            T.stack(
                [
                    _row_mask(T, qi, 6, 7, case, device=q.device)
                    for qi in range(6)
                ],
                dim=0,
            )
            for _ in range(2)
        ],
        dim=0,
    )
    scores = scores.masked_fill(~mask[:, None], float("-inf"))
    weights = T.softmax(scores, dim=-1)
    weights = T.where(~mask[:, None].any(dim=-1, keepdim=True), T.zeros_like(weights), weights)
    out = T.einsum("bhts,bshd->bthd", weights, v_rep)
    err, n, _ = sampled_attention_error(
        T,
        case,
        {"q_ref": q, "k_ref": k, "v_ref": v, "qv_ref": qv, "out": out, "descales": (None, None, None)},
    )
    if n < 1 or err > 1e-5:
        raise AssertionError(f"sampled fwd mismatch {err} samples={n}")

    # combine reference
    splits, b, s, h, d = 3, 2, 5, 4, 8
    partial = T.randn(splits, b, s, h, d)
    lse = T.randn(splits, b, s, h)
    ref_lse = T.logsumexp(lse, dim=0)
    scale = T.exp(lse - ref_lse)
    scale = T.where(T.isfinite(scale), scale, T.zeros_like(scale))
    ref = (scale.unsqueeze(-1) * partial).sum(0)
    fake_out = ref.clone()
    result = validate_outputs(
        T,
        {"operation": "combine", "dtype": "fp32", "head_dim": d, "head_dim_v": d, "heads_q": h, "batch": b, "seqlen_q": s},
        {"out": fake_out, "lse": ref_lse, "out_partial": partial, "lse_partial": lse},
    )
    if result["status"] != "pass":
        raise AssertionError(result)

    # rotary interleaved roundtrip-ish: unit vectors rotate
    x = T.zeros(1, 2, 1, 4)
    x[0, 0, 0, 0] = 1
    x[0, 1, 0, 0] = 1
    ang = T.tensor([[0.0, 0.0], [T.pi / 2, 0.0]])  # (Sro=2, rd/2=2) wait rotary_dim=4 -> rd/2=2
    # positions 0 and 1: angle 0 and pi/2 on first pair
    ang = T.zeros(4, 2)
    ang[1, 0] = T.pi / 2
    cos, sin = T.cos(ang), T.sin(ang)
    off = T.tensor([0])
    y = apply_rotary(x, cos, sin, off, True, False)
    # token 1 pair0 should be (0, 1) from (1, 0) rotated by 90deg
    if float((y[0, 1, 0, :2] - T.tensor([0.0, 1.0])).abs().max()) > 1e-5:
        raise AssertionError(y[0, 1, 0, :2])

    # sparse bwd vs autograd on one head
    case_b = {
        "operation": "fixed_bwd",
        "dtype": "fp32",
        "head_dim": 8,
        "head_dim_v": 8,
        "heads_q": 2,
        "heads_kv": 2,
        "batch": 1,
        "seqlen_q": 5,
        "seqlen_k": 6,
        "causal": True,
    }
    q = T.randn(1, 5, 2, 8, requires_grad=True)
    k = T.randn(1, 6, 2, 8, requires_grad=True)
    v = T.randn(1, 6, 2, 8, requires_grad=True)
    go = T.zeros(1, 5, 2, 8)
    go[0, 0, 0] = T.linspace(-0.5, 0.5, 8)
    go[0, 2, 0] = T.linspace(-0.5, 0.5, 8)
    go[0, 4, 0] = T.linspace(-0.5, 0.5, 8)
    scale = 8 ** -0.5
    scores = T.einsum("bthd,bshd->bhts", q, k) * scale
    mask = T.stack([_row_mask(T, qi, 5, 6, case_b, device=q.device) for qi in range(5)])
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    w = T.softmax(scores, dim=-1)
    out = T.einsum("bhts,bshd->bthd", w, v)
    dq, dk, dv = T.autograd.grad(out, (q, k, v), go)
    bwd_rep = sampled_backward_error(
        T,
        case_b,
        {"q": q.detach(), "k": k.detach(), "v": v.detach(), "q_ref": q.detach(), "k_ref": k.detach(), "v_ref": v.detach(),
         "grads": (dq, dk, dv), "grad_out": go},
    )
    berr, bn = bwd_rep["selected_abs_error"], bwd_rep["samples"]
    if bn < 1 or berr > 1e-4 or not bwd_rep["pass"]:
        raise AssertionError(f"bwd mismatch {bwd_rep}")

    pattern = _describe_grad_pattern(case_b)
    if not pattern["full_shape_kernel_work_retained"] or pattern["kind"] != "sparse_query_rows_on_full_shape_dO":
        raise AssertionError(pattern)
    finite_only = validate_outputs(
        T,
        {"operation": "fixed_fwd", "dtype": "fp32", "head_dim": 8, "heads_q": 1, "heads_kv": 1, "batch": 1, "seqlen_q": 2, "seqlen_k": 2},
        {"q_ref": T.ones(1, 2, 1, 8), "k_ref": T.ones(1, 2, 1, 8), "v_ref": T.ones(1, 2, 1, 8),
         "out": T.ones(1, 2, 1, 8) * 1e3, "descales": (None, None, None)},
    )
    if finite_only["status"] != "fail":
        raise AssertionError(finite_only)

    injections = _failure_injections(T)

    return {
        "status": "pass",
        "sampled_fwd_error": err,
        "sampled_bwd_error": berr,
        "samples_fwd": n,
        "samples_bwd": bn,
        "grad_pattern_kind": pattern["kind"],
        "failure_injections": injections,
    }


if __name__ == "__main__":
    print(_self_check())
