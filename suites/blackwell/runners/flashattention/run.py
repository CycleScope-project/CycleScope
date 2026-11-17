#!/usr/bin/env python3
"""FA4 1SM/2SM cases on B300S. One file. No hopper-prototype capture."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
PREFIX = ROOT / ".deps" / "fa4"
os.environ.pop("FA_CLC", None)

# SC delivery writes to its own corpus root; set BEFORE common.py
# reads BW_CORPUS_ROOT at import time. Explicit env wins.
if "--delivery300sc" in sys.argv:
    os.environ.setdefault("BW_CORPUS_ROOT", str(ROOT / "artifacts" / "corpus_sc"))

if str(ROOT / "suites") not in sys.path:
    sys.path.insert(0, str(ROOT / "suites"))

from common import ART, write_result  # noqa: E402
INSTALL_LOG = ART / "flashattention" / "_install.txt"

SUITE = "flashattention"
CAPTURE: dict = {}


def site_packages() -> list[str]:
    pats = (
        str(PREFIX / "lib" / "python*" / "site-packages"),
        str(PREFIX / "lib" / "python*" / "dist-packages"),
        str(PREFIX / "local" / "lib" / "python*" / "site-packages"),
        str(PREFIX / "local" / "lib" / "python*" / "dist-packages"),
    )
    out: list[str] = []
    for pat in pats:
        out.extend(glob.glob(pat))
    return sorted(set(out))


def _use_prefix() -> None:
    extras: list[str] = []
    for p in site_packages():
        extras.append(p)
        extras.extend(glob.glob(os.path.join(p, "nvidia_cutlass_dsl", "dsl_packages")))
        extras.extend(glob.glob(os.path.join(p, "nvidia_cutlass_dsl", "python_packages")))
    for p in reversed(extras):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    sys.path[:] = [
        p for p in sys.path
        if "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl" not in p
    ]
    for name in list(sys.modules):
        if name == "cutlass" or name.startswith(("cutlass.", "nvidia_cutlass_dsl", "quack", "flash_attn")):
            del sys.modules[name]
    _log_install("sys.path[0:6]=" + repr(sys.path[:6]))


def _log_install(text: str) -> None:
    INSTALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with INSTALL_LOG.open("a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _pip(args: list[str]) -> None:
    cmd = [
        sys.executable, "-m", "pip", "install", "--pre",
        "--prefix", str(PREFIX),
        "--ignore-installed",
        *args,
    ]
    _log_install("$ " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _log_install(p.stdout or "")
    if p.returncode != 0:
        raise RuntimeError(f"pip failed rc={p.returncode}\n{(p.stdout or '')[-3000:]}")


def ensure_fa4() -> None:
    PREFIX.mkdir(parents=True, exist_ok=True)
    _use_prefix()
    if os.environ.get("CUTE_DSL_DUMP_DIAG") == "1":
        from c450full_cute_dsl_diagnostics import install_compile_diagnostic

        install_compile_diagnostic()
    try:
        import flash_attn.cute  # noqa: F401
        _log_install("flash_attn.cute already importable")
        return
    except Exception as e:
        _log_install(f"import miss: {type(e).__name__}: {e}")
    try:
        _pip(["--no-deps", "flash-attn-4[cu13]"])
    except Exception as e:
        _log_install(f"wheel[cu13] failed: {e}")
        try:
            _pip(["--no-deps", "flash-attn-4"])
        except Exception as e2:
            _log_install(f"wheel failed: {e2}")
            src = PREFIX / "src"
            if not (src / "flash_attn" / "cute").exists():
                _log_install("git clone Dao-AILab/flash-attention")
                subprocess.check_call(
                    ["git", "clone", "--depth", "1",
                     "https://github.com/Dao-AILab/flash-attention", str(src)],
                )
            _pip(["--no-deps", str(src / "flash_attn" / "cute")])
    try:
        _pip(["--no-deps", "quack-kernels>=0.5.3"])
    except Exception as e:
        _log_install(f"quack optional: {e}")
    try:
        _pip(["--no-deps", "nvidia-cutlass-dsl[cu13]>=4.6.2"])
    except Exception as e:
        _log_install(f"cutlass-dsl optional: {e}")
    _use_prefix()
    import flash_attn.cute  # noqa: F401


def _configure_case_environment(case: dict) -> None:
    if case.get("mode") == "full" and case.get("clc"):
        os.environ["FA_CLC"] = "1"
    else:
        os.environ.pop("FA_CLC", None)


def set_disable_2cta(on: bool) -> None:
    os.environ["FA_DISABLE_2CTA"] = "1" if on else "0"
    try:
        import flash_attn.cute.utils as u

        u._fa_disable_2cta_enabled = on
    except Exception:
        pass


def hook_cls(cls) -> None:
    if getattr(cls, "_fa4_hooked", False):
        return
    orig = cls.__init__

    def wrapped(self, *a, **kw):
        orig(self, *a, **kw)
        rec = {
            "cls": cls.__name__,
            "use_2cta_instrs": getattr(self, "use_2cta_instrs", kw.get("use_2cta_instrs")),
            "cluster_shape_mn": getattr(self, "cluster_shape_mn", None),
            "cta_group_size": getattr(self, "cta_group_size", None),
            "cluster_size": getattr(self, "cluster_size", kw.get("cluster_size")),
        }
        CAPTURE.setdefault("ctors", []).append(rec)
        CAPTURE["last"] = rec

    cls.__init__ = wrapped
    cls._fa4_hooked = True


def install_hooks() -> None:
    import flash_attn.cute.interface as iface

    for name in (
        "FlashAttentionForwardSm100",
        "FlashAttentionBackwardSm100",
        "BlackwellFusedMultiHeadAttentionForward",
        "BlackwellFusedMultiHeadAttentionBackward",
        "FlashAttentionMLAForwardSm100",
    ):
        cls = getattr(iface, name, None)
        if cls is not None:
            hook_cls(cls)


def dump_cache_proof() -> str:
    parts = []
    try:
        from flash_attn.cute import interface as iface

        for fn_name in ("_flash_attn_fwd", "_flash_attn_bwd"):
            fn = getattr(iface, fn_name, None)
            cache = getattr(fn, "compile_cache", None) if fn is not None else None
            if cache is None:
                continue
            items = getattr(cache, "items", None)
            if not callable(items):
                parts.append(f"{fn_name}.cache_type={type(cache).__name__}")
                continue
            for k, v in list(items())[-4:]:
                parts.append(f"{fn_name}.key={k!r}")
                parts.append(f"{fn_name}.mod={type(v).__name__}:{v!r}"[:800])
    except Exception as e:
        parts.append(f"cache_err={e}")
    return "\n".join(parts)


def proof_text(want_2c: bool) -> tuple[str, bool]:
    recs = CAPTURE.get("ctors") or []
    last = CAPTURE.get("last") or {}
    bits = [f"ctors={recs!r}", f"last={last!r}", dump_cache_proof()]
    if CAPTURE.get("kernels"):
        bits.append("kernels=" + ",".join(CAPTURE["kernels"]))
    bits.extend(str(n) for n in CAPTURE.get("notes", []))
    text = "\n".join(bits)
    use = last.get("use_2cta_instrs")
    cta = last.get("cta_group_size")
    cluster = last.get("cluster_shape_mn")
    if use is True:
        text += "\nuse_2cta_instrs=True"
    elif use is False:
        text += "\nuse_2cta_instrs=False"
    if cluster in ((2, 1), (2, 1, 1)):
        text += "\ncluster_shape_mn=(2,1)"
    if cta == 2:
        text += "\ncta_group_size=2"
    has = (
        "use_2cta_instrs=True" in text
        or "cta_group::2" in text
        or "kNumMulticast=2" in text
        or "cta_group_size=2" in text
        or "cluster_shape_mn=(2,1)" in text
    )
    return text, (has if want_2c else True)


def qkv(B, Sq, Sk, Hq, Hk, hd, hdv, dtype, requires_grad=False):
    import torch

    gen = torch.bfloat16 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else dtype
    q = torch.randn(B, Sq, Hq, hd, device="cuda", dtype=gen)
    k = torch.randn(B, Sk, Hk, hd, device="cuda", dtype=gen)
    v = torch.randn(B, Sk, Hk, hdv, device="cuda", dtype=gen)
    if dtype != gen:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if requires_grad:
        q, k, v = q.requires_grad_(), k.requires_grad_(), v.requires_grad_()
    return q, k, v


def as_tensor(x):
    return x[0] if isinstance(x, (tuple, list)) else x




def _dtype_map() -> dict:
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "e4m3": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2,
    }


def _note(text: str) -> None:
    CAPTURE.setdefault("notes", []).append(text)


def _cu_seqlens(nseq: int, Sq: int, Sk: int):
    import torch

    lq = [Sq // nseq] * (nseq - 1) + [Sq - (Sq // nseq) * (nseq - 1)]
    lk = [Sk // nseq] * (nseq - 1) + [Sk - (Sk // nseq) * (nseq - 1)]
    cq, ck = [0], [0]
    for a, b in zip(lq, lk):
        cq.append(cq[-1] + a)
        ck.append(ck[-1] + b)
    return (
        torch.tensor(cq, device="cuda", dtype=torch.int32),
        torch.tensor(ck, device="cuda", dtype=torch.int32),
        max(lq),
        max(lk),
    )


def _gather_kv_indices(B: int, Sq: int, Sk: int, topk: int):
    import torch

    # Deterministic per-row distinct in-bounds indices; length % 128 == 0 is
    # required by the DSA/topk path (interface.py:995-1000).
    step = max(Sk // int(topk), 1)
    base = torch.arange(int(topk), device="cuda", dtype=torch.int32)
    offs = (step * torch.arange(Sq, device="cuda", dtype=torch.int32)) % Sk
    rows = (base[None, :] + offs[:, None]) % Sk
    return rows.unsqueeze(0).expand(B, Sq, int(topk)).contiguous()


def _block_sparse_tensors(B: int, H: int, Sq: int, Sk: int):
    """Block-causal sparsity metadata: each (m,n) block pair appears exactly once,
    as mask (j <= i) or full (j > i), so softmax denominators stay complete.
    block_size[0] must be a multiple of q_stage*tile_m=256 (interface.py:345-347,
    block_sparsity.py:520-536)."""
    import torch
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    tile_m = tile_n = 128
    q_stage = 2 if Sq > tile_m else 1
    base_m = q_stage * tile_m
    mb = -(-Sq // base_m)
    nb = -(-Sk // tile_n)
    i = torch.arange(mb, device="cuda", dtype=torch.int32)
    j = torch.arange(nb, device="cuda", dtype=torch.int32)
    mask_cnt = torch.minimum(i + 1, j.new_tensor(nb))
    mask_idx = j[None, :].expand(mb, nb)
    full_cnt = torch.clamp(nb - 1 - i, min=0)
    full_idx = (i + 1)[:, None] + j[None, :]
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt.expand(B, H, mb).contiguous(),
        mask_block_idx=mask_idx.expand(B, H, mb, nb).contiguous(),
        full_block_cnt=full_cnt.expand(B, H, mb).contiguous(),
        full_block_idx=full_idx.expand(B, H, mb, nb).contiguous(),
        block_size=(base_m, tile_n),
    )


def _fwd_option_kwargs(case: dict, B: int, Sq: int, Sk: int, Hq: int) -> dict:
    import torch

    kw: dict = {"causal": bool(case.get("causal", False))}
    if case.get("window_size") is not None:
        kw["window_size"] = tuple(int(x) for x in case["window_size"])
    if case.get("softcap"):
        kw["softcap"] = float(case["softcap"])
    if case.get("num_splits") is not None:
        kw["num_splits"] = int(case["num_splits"])
    if case.get("pack_gqa") is not None:
        kw["pack_gqa"] = bool(case["pack_gqa"])
    if case.get("deterministic"):
        kw["deterministic"] = True
    if case.get("sink"):
        # learnable_sink: shape (num_head,), fp16/bf16/fp32 — interface.py:671-674
        kw["learnable_sink"] = torch.zeros(Hq, dtype=torch.float32, device="cuda")
    if case.get("score_mod"):
        from flash_attn.cute import utils

        # tanh closure equivalent to softcap (utils.py:166-174); softcap kwarg stays unset
        kw["score_mod"] = utils.create_softcap_scoremod(float(case.get("softcap") or 30.0))
    if case.get("block_sparse"):
        kw["block_sparse_tensors"] = _block_sparse_tensors(B, Hq, Sq, Sk)
    return kw


def _effective_shape(case: dict) -> tuple:
    shape = [int(x) for x in case["shape"]]
    if case["kind"] == "mla_bwd" or (
            case.get("topk") and case["kind"] == "mla"):
        # Top-k/DSA uses MQA-128: flash_fwd_mla_sm100.py:77-80 and interface.py:2813.
        if shape[3:5] != [128, 1]:
            _note(
                f"topk MQA-128 override: spec Hq/Hk={shape[3]}/{shape[4]}"
                " -> 128/1, pack_gqa=True"
            )
        shape[3], shape[4] = 128, 1
    return tuple(shape)


def _bottom_right_causal_mask(seq_q: int, seq_k: int, device):
    import torch

    q_pos = torch.arange(seq_q, device=device) + (seq_k - seq_q)
    k_pos = torch.arange(seq_k, device=device)
    return k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)


def _backward_profiled(out, dout) -> list[str]:
    import torch

    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            out.backward(dout)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.name]
        CAPTURE["kernels"] = names
        return names
    except Exception:
        out.backward(dout)
        torch.cuda.synchronize()
        return []


def run_one(case: dict):
    import torch
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    if not case.get("_retain_capture"):
        CAPTURE.clear()
    set_disable_2cta(bool(case.get("disable_2cta")))
    install_hooks()

    dtype = _dtype_map()[case["dtype"]]
    B, Sq, Sk, Hq, Hk, hd, hdv = _effective_shape(case)
    causal = bool(case.get("causal", False))
    want_2c = case["cta_group"] == 2
    kind = case["kind"]
    kernel = ""
    extra_kernels: list[str] = []
    reference_k = None
    reference_v = None

    try:
        if kind == "fwd":
            if case.get("paged"):
                page_size = int(case.get("page_size", 128))
                if Sk % page_size:
                    raise ValueError(
                        f"paged seqlen_k {Sk} is not page aligned to {page_size}")
                num_pages = Sk // page_size
                q = torch.randn(B, Sq, Hq, hd, device="cuda", dtype=dtype)
                k = torch.randn(
                    num_pages, page_size, Hk, hd, device="cuda", dtype=dtype)
                v = torch.randn(
                    num_pages, page_size, Hk, hdv, device="cuda", dtype=dtype)
                page_table = torch.arange(
                    num_pages, device="cuda", dtype=torch.int32
                ).unsqueeze(0).expand(B, num_pages).contiguous()
                seqused_k = torch.full(
                    (B,), Sk, device="cuda", dtype=torch.int32)
                out = as_tensor(flash_attn_varlen_func(
                    q, k, v, page_table=page_table, seqused_k=seqused_k,
                    max_seqlen_q=Sq, max_seqlen_k=Sk,
                    **_fwd_option_kwargs(case, B, Sq, Sk, Hq),
                ))
                reference_k = k.reshape(1, Sk, Hk, hd).expand(B, -1, -1, -1)
                reference_v = v.reshape(1, Sk, Hk, hdv).expand(B, -1, -1, -1)
                _note(
                    f"paged KV: page_table=(B,{num_pages}) int32, "
                    f"page_size={page_size}, seqused_k=Sk")
            else:
                q, k, v = qkv(B, Sq, Sk, Hq, Hk, hd, hdv, dtype)
                reference_k, reference_v = k, v
                if case.get("descale"):
                    from flash_attn.cute.interface import _flash_attn_fwd

                    # q/k/v_descale: (batch, num_head_kv) float32, FP8 only —
                    # interface.py:571-573, 757-768
                    dsc = torch.ones(B, Hk, dtype=torch.float32, device="cuda")
                    out = _flash_attn_fwd(
                        q, k, v, causal=causal,
                        q_descale=dsc, k_descale=dsc, v_descale=dsc
                    )[0]
                    _note(
                        "descale via _flash_attn_fwd("
                        "q/k/v_descale=(B,Hkv) fp32 ones)")
                else:
                    out = as_tensor(flash_attn_func(
                        q, k, v, **_fwd_option_kwargs(case, B, Sq, Sk, Hq)))
            if case.get("clc"):
                _note("FA_CLC=1 public CLC scheduler switch")
            torch.cuda.synchronize()
            kernel = (CAPTURE.get("last") or {}).get("cls") or "flash_attn_func"
        elif kind == "bwd":
            q, k, v = qkv(B, Sq, Sk, Hq, Hk, hd, hdv, torch.bfloat16, requires_grad=True)
            kw: dict = {"causal": causal}
            if case.get("pack_gqa") is not None:
                kw["pack_gqa"] = bool(case["pack_gqa"])
            if case.get("deterministic"):
                kw["deterministic"] = True
            if case.get("window_size") is not None:
                kw["window_size"] = tuple(int(x) for x in case["window_size"])
            out = as_tensor(flash_attn_func(q, k, v, **kw))
            dout = torch.randn_like(out)
            try:
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
                    out.backward(dout)
                    torch.cuda.synchronize()
                extra_kernels = [e.name for e in prof.events() if e.name]
                CAPTURE["kernels"] = extra_kernels
            except Exception:
                out.backward(dout)
                torch.cuda.synchronize()
            recs = CAPTURE.get("ctors") or []
            main = [r for r in recs if "Backward" in r.get("cls", "") and "Pre" not in r.get("cls", "") and "Post" not in r.get("cls", "")]
            kernel = (main[-1] if main else CAPTURE.get("last") or {}).get("cls") or "flash_attn_bwd"
            if extra_kernels:
                kernel = kernel + "|kernels=" + ",".join(dict.fromkeys(extra_kernels))[:400]
        elif kind == "mla":
            q = torch.randn(B, Sq, Hq, hd, device="cuda", dtype=torch.bfloat16)
            qv = torch.randn(B, Sq, Hq, hdv, device="cuda", dtype=torch.bfloat16)
            if case.get("paged"):
                # paged KV: k/v (num_pages, page_size, Hkv, d); page_table
                # (B, max_num_pages) int32 — interface.py:630-641, 798-804.
                page_size = 128  # == MLA tile_n keeps the TMA KV path
                num_pages = Sk // page_size
                k = torch.randn(num_pages, page_size, Hk, hd, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(num_pages, page_size, Hk, hdv, device="cuda", dtype=torch.bfloat16)
                page_table = (
                    torch.arange(num_pages, device="cuda", dtype=torch.int32)
                    .unsqueeze(0)
                    .expand(B, num_pages)
                    .contiguous()
                )
                seqused_k = torch.full((B,), Sk, device="cuda", dtype=torch.int32)
                out = as_tensor(flash_attn_varlen_func(
                    q, k, v, qv=qv, page_table=page_table, seqused_k=seqused_k,
                    max_seqlen_q=Sq, max_seqlen_k=Sk, causal=causal,
                ))
                _note("paged KV: page_table=(B,32) int32 arange, page_size=128, seqused_k=Sk")
            elif case.get("pure_qv"):
                # k is v (same object) + hdimv=512 -> FlashAttnFunc.forward
                # (interface.py:3216-3221) sets q=k=None, qv=q.
                v = torch.randn(B, Sk, Hk, hdv, device="cuda", dtype=torch.bfloat16)
                out = as_tensor(flash_attn_func(qv, v, v, causal=causal))
                _note("pure_qv: flash_attn_func(qv, v, v) k-is-v hdimv=512 -> q=k=None path")
            elif case.get("topk"):
                k = torch.randn(B, Sk, Hk, hd, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(B, Sk, Hk, hdv, device="cuda", dtype=torch.bfloat16)
                g = _gather_kv_indices(B, Sq, Sk, case["topk"])
                out = as_tensor(flash_attn_func(
                    q, k, v, qv=qv, gather_kv_indices=g, pack_gqa=True, causal=causal,
                ))
                _note(f"topk gather: gather_kv_indices (B,Sq,{case['topk']}) int32, len%128==0")
            elif case.get("varlen"):
                q2 = torch.randn(Sq, Hq, hd, device="cuda", dtype=torch.bfloat16)
                qv2 = torch.randn(Sq, Hq, hdv, device="cuda", dtype=torch.bfloat16)
                k2 = torch.randn(Sk, Hk, hd, device="cuda", dtype=torch.bfloat16)
                v2 = torch.randn(Sk, Hk, hdv, device="cuda", dtype=torch.bfloat16)
                cu_q, cu_k, mq, mk = _cu_seqlens(int(case.get("nseq") or 2), Sq, Sk)
                out = as_tensor(flash_attn_varlen_func(
                    q2, k2, v2, qv=qv2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=mq, max_seqlen_k=mk, causal=causal,
                ))
                _note(f"mla varlen: nseq={int(case.get('nseq') or 2)} cu_seqlens segments")
            else:
                k = torch.randn(B, Sk, Hk, hd, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(B, Sk, Hk, hdv, device="cuda", dtype=torch.bfloat16)
                out = as_tensor(flash_attn_func(q, k, v, qv=qv, causal=causal))
            torch.cuda.synchronize()
            kernel = (CAPTURE.get("last") or {}).get("cls") or "FlashAttentionMLAForwardSm100"
        elif kind == "mla_bwd":
            topk = int(case.get("topk") or 512)
            q = torch.randn(B, Sq, Hq, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            qv = torch.randn(B, Sq, Hq, hdv, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, Sk, Hk, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(B, Sk, Hk, hdv, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            g = _gather_kv_indices(B, Sq, Sk, topk)
            out = as_tensor(flash_attn_func(
                q, k, v, qv=qv, gather_kv_indices=g, pack_gqa=True, causal=causal,
            ))
            torch.cuda.synchronize()
            fwd_cls = (CAPTURE.get("last") or {}).get("cls") or "FlashAttentionMLAForwardSm100"
            dout = torch.randn_like(out)
            extra_kernels = _backward_profiled(out, dout)
            kernel = fwd_cls + "|mla_sparse_bwd"
            if extra_kernels:
                kernel = kernel + "|kernels=" + ",".join(dict.fromkeys(extra_kernels))[:400]
        elif kind == "varlen":
            nseq = int(case.get("nseq") or 1)
            q = torch.randn(Sq, Hq, hd, device="cuda", dtype=dtype)
            k = torch.randn(Sk, Hk, hd, device="cuda", dtype=dtype)
            v = torch.randn(Sk, Hk, hdv, device="cuda", dtype=dtype)
            cu_q, cu_k, mq, mk = _cu_seqlens(nseq, Sq, Sk)
            out = as_tensor(flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=mq, max_seqlen_k=mk, causal=causal,
            ))
            if nseq > 1:
                _note(f"varlen nseq={nseq}: Sq/Sk split into {nseq} equal segments, totals preserved")
            torch.cuda.synchronize()
            kernel = (CAPTURE.get("last") or {}).get("cls") or "flash_attn_varlen_func"
        else:
            raise ValueError(kind)

        proof, has = proof_text(want_2c)
        if (case.get("mode") == "full"
                and case.get("_verify_full", True)):
            out_check = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(out_check.float()).all():
                raise RuntimeError("full-mode output contains non-finite values")
            if (kind == "fwd" and not case.get("descale")
                    and not case.get("sink")
                    and not case.get("window_size")
                    and not case.get("softcap")
                    and not case.get("score_mod")
                    and not case.get("mask_mod")
                    and not case.get("block_sparse")
                    and dtype in (torch.bfloat16, torch.float16)):
                import torch.nn.functional as F
                causal_mask = (_bottom_right_causal_mask(Sq, Sk, q.device)
                               if causal else None)
                ref = F.scaled_dot_product_attention(
                    q.transpose(1, 2), reference_k.transpose(1, 2),
                    reference_v.transpose(1, 2),
                    attn_mask=causal_mask, is_causal=False,
                    enable_gqa=(Hq != Hk))
                torch.testing.assert_close(
                    out_check, ref.transpose(1, 2), rtol=2e-2, atol=2e-2)
                proof += "\nfull-mode reference=scaled_dot_product_attention"
                if causal:
                    proof += " with bottom-right causal mask"
        last = CAPTURE.get("last") or {}
        observed_2c = last.get("use_2cta_instrs") is True or last.get("cta_group_size") == 2
        ok = True
        err = None
        if want_2c and not has:
            ok = False
            err = "2CTA requested but no use_2cta_instrs/cta_group evidence"
        if want_2c and has and last and last.get("use_2cta_instrs") is False and last.get("cta_group_size") not in (2, None):
            ok = False
            err = "compile said 1CTA"
        if want_2c and last.get("use_2cta_instrs") is False and last.get("cta_group_size") != 2:
            ok = False
            err = "use_2cta_instrs=False"
            has = False
        if not want_2c and observed_2c:
            # 1C case accidentally 2C — still record, not the requested specialization
            ok = False
            err = "wanted 1CTA but observed 2CTA"
        return {
            "ok": ok,
            "kernel": kernel,
            "proof": proof,
            "error": err,
            "observed_2c": observed_2c,
        }
    except Exception as e:
        proof, _ = proof_text(want_2c)
        recs = CAPTURE.get("ctors") or []
        kernel = (CAPTURE.get("last") or {}).get("cls") or ""
        if extra_kernels:
            proof += "\nkernels=" + ",".join(dict.fromkeys(extra_kernels))[:800]
        proof += "\n" + traceback.format_exc()[-2000:]
        return {
            "ok": False,
            "kernel": kernel or type(e).__name__,
            "proof": proof,
            "error": f"{type(e).__name__}: {e}",
            "observed_2c": False,
        }



_SPEC_SC_PATH = (
    Path(os.environ["BW_CASES_DIR"]) / "cases_flashattention.json"
    if os.environ.get("BW_CASES_DIR") else
    ROOT / "suites" / "c450_full" / "cases_flashattention.json"
)


def _load_spec_cases(spec_path: Path) -> list[dict]:
    """delivery300sc spec -> case dicts; stdlib-only so `--list` never imports torch."""
    cases: list[dict] = []
    if not spec_path.exists():
        return cases
    raw = json.loads(spec_path.read_text())
    for c in raw.get("cases", []):
        d = dict(c)
        d["shape"] = tuple(int(x) for x in d["shape"])
        if d.get("window_size") is not None:
            d["window_size"] = tuple(int(x) for x in d["window_size"])
        cases.append(d)
    return cases


SPEC_SC_CASES = _load_spec_cases(_SPEC_SC_PATH)


def main() -> None:
    argv = sys.argv[1:]
    if "--list" in argv:
        for c in SPEC_SC_CASES:
            print(c["id"])
        return
    ids = [a for a in argv if not a.startswith("--")]
    cases = SPEC_SC_CASES
    if ids:
        want = set(ids)
        cases = [c for c in cases if c["id"] in want]

    if "--resolve" in argv:
        # T0046 harness-stage dry-run (additive): print resolved entries and
        # exit before ensure_fa4()/torch import (no GPU, no FA prefix needed).
        for c in cases:
            print("RESOLVE " + json.dumps(c, sort_keys=True, default=list))
        return

    if len(cases) > 1:
        # Batch mode: one fresh subprocess per case. The in-process kernel
        # cache keyed by compile key (shape-independent) skips the kernel
        # ctor on later rungs, so the ctor hooks record no 2CTA evidence
        # even though the launched SASS is unchanged. A fresh process
        # re-runs the ctor and the evidence chain for every case.
        import subprocess
        child_fail = []
        # Forward the spec-selection flag so the child resolves the same SC
        # spec.
        flags = [f for f in ("--delivery300sc",) if f in argv]
        for c in cases:
            print("RUN", c["id"], flush=True)
            env = {k: v for k, v in os.environ.items() if k != "FA_DISABLE_2CTA"}
            p = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *flags, c["id"]], env=env)
            if p.returncode != 0:
                child_fail.append((c["id"], p.returncode))
                print("CHILD_FAIL", c["id"], p.returncode, flush=True)
        print("BATCH_DONE", len(cases), "child_fail", child_fail, flush=True)
        return

    if cases:
        _configure_case_environment(cases[0])
    try:
        ensure_fa4()
        import torch
        import flash_attn.cute  # noqa: F401
    except Exception as e:
        _log_install(f"IMPORT_FAIL {type(e).__name__}: {e}\n{traceback.format_exc()}")
        print("IMPORT_FAIL", e)
        raise
    print("cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
    print("site", site_packages())
    for c in cases:
        print("RUN", c["id"], flush=True)
        calls = int(c.get("calls", 1)) if c.get("mode") == "full" else 1
        r = {"ok": True, "error": None, "proof": "", "kernel": "",
             "observed_2c": False}
        for call_index in range(calls):
            torch.manual_seed(20261016)
            call_case = dict(
                c, _retain_capture=call_index > 0,
                _verify_full=call_index + 1 == calls,
            )
            r = run_one(call_case)
            if not r["ok"]:
                break
        shape = c.get("shape") or []
        shape_tag = (f"B{shape[0]}_Sq{shape[1]}_Sk{shape[2]}_Hq{shape[3]}"
                     f"_Hkv{shape[4]}_d{shape[5]}_dv{shape[6]}"
                     if len(shape) == 7 else str(shape))
        res = {
            "id": c["id"], "suite": SUITE, "ok": r["ok"],
            "calls": calls,
            "cta_group": c.get("cta_group"), "cluster": 1, "sms": c.get("cta_group"),
            "dtype": c.get("dtype"),
            "shape": shape_tag,
            "kernel": r["kernel"],
            "proof": r["proof"],
            "error": r["error"],
        }
        for key in ("family", "est_cycles", "est_basis", "heldout"):
            if key in c:
                res[key] = c[key]
        write_result(SUITE, c["id"], res)
        print("DONE", c["id"], "ok", r["ok"], "err", r["error"], flush=True)
        torch.cuda.empty_cache()
    if os.environ.get("CUTE_DSL_DUMP_DIAG") == "1":
        from c450full_cute_dsl_diagnostics import report_dump_directories

        report_dump_directories()

if __name__ == "__main__":
    main()
