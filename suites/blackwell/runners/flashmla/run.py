#!/usr/bin/env python3
"""FlashMLA runner: one real-call entry per case (suites/legacy/dev283/cases_flashmla.json).

Kinds:
  sparse_decode   flash_mla_with_kvcache on an FP8 KV cache (MQA, d_qk=512+64,
                  indices-driven sparse decode; ±split via num_sm_parts none)
  sparse_prefill  flash_mla_sparse_fwd (topk indices, d_qk=512)
  dense_fwd       flash_attn_varlen_func forward (CUTLASS FMHA sm100)
  norm_rope       fused_norm_rope_attn_rope_cast (norm + rope fusion)

Each case carries `family`, `main_kernel` (regex full-matched on the mangled
kernel name in launches.jsonl/NCU), `shape`, `params`.  The runner builds
inputs, runs once, checks vs a torch reference where a cheap one exists, and
writes result.json via suites/common.py write_result.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
sys.path.insert(0, str(ROOT / "suites"))
SPEC = ROOT / "suites" / "c450_full" / "cases_flashmla.json"
SUITE = "flashmla"
PREFIX = os.environ.get("BW_FLASHMLA_PREFIX",
                        str(Path(os.environ.get("BW_DEPS_DIR", "")) / "flashmla"))
for p in (PREFIX, str(Path(PREFIX) / "lib"), str(ROOT.parents[0] / "tools" / "src" / "flashmla")):
    if p and p not in sys.path:
        sys.path.insert(0, p)
if PREFIX not in sys.path:
    sys.path.insert(0, PREFIX)
# the inplace build puts cuda.abi3.so inside flash_mla/ under the source prefix
for p in (PREFIX, str(Path(PREFIX).parent / "tools" / "src" / "flashmla")):
    if p not in sys.path:
        sys.path.insert(0, p)
# fall back to the inplace-built source checkout only when the prefix does
# not already carry a flash_mla package (c450v3: BW_FLASHMLA_PREFIX points at
# the FLASHMLA_EFFECTIVE_SM_COUNT-patched build under deps/flashmla-c450v2)
if not (Path(PREFIX) / "flash_mla" / "__init__.py").is_file():
    sys.path.insert(0, os.environ.get(
        "BW_FLASHMLA_SRC",
        str(Path(os.environ.get("CYCLESCOPE_ARTIFACT_ROOT", ""))
            / "tools" / "blackwell-flashmla-source")))


def _spec_path():
    cases_dir = os.environ.get("BW_CASES_DIR")
    return Path(cases_dir) / "cases_flashmla.json" if cases_dir else SPEC


def _load_cases():
    return json.loads(_spec_path().read_text())["cases"]


def load_case(cid):
    for c in _load_cases():
        if c["id"] == cid:
            return c
    raise KeyError(cid)


def _call_many(case, fn):
    result = None
    for _ in range(int(case.get("calls", 1)) if case.get("mode") == "full" else 1):
        result = fn()
    return result


def _build_cos_sin_cache(max_token_position, device, rope_dim=64):
    import torch

    base = 100000
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32,
                              device=device) / rope_dim))
    freqs = torch.outer(
        torch.arange(max_token_position, dtype=torch.float32, device=device),
        inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.cat((freqs_cis.real, freqs_cis.imag), dim=-1)


def _configure_sm_env(c: dict) -> None:
    if c.get("effective_sms") is not None and (
            c.get("mode") != "full" or c.get("single_cluster")):
        os.environ["FLASHMLA_EFFECTIVE_SM_COUNT"] = str(c["effective_sms"])
    elif c.get("mode") == "full":
        os.environ.pop("FLASHMLA_EFFECTIVE_SM_COUNT", None)


def run(cid):
    import torch
    from common import write_result
    c = load_case(cid)
    _configure_sm_env(c)
    dev = torch.device("cuda")
    torch.manual_seed(20261016)
    ok, err, proof = True, "", ""
    try:
        kind = c["kind"]
        # aliases: head-dim/scheduler variants share the same host API; the
        # library selects the kernel template internally from the shapes.
        kind = {"sparse_decode_h128": "sparse_decode",
                "sparse_prefill_h64": "sparse_prefill",
                "sparse_prefill_st": "sparse_prefill"}.get(kind, kind)
        p = c.get("params", {})
        s = c["shape"]
        import flash_mla
        if kind == "sparse_decode":
            B, Hq, _dqk, topk, kv_len = s
            q = torch.randn(B, 1, Hq, 512, device=dev, dtype=torch.bfloat16)
            nb = (kv_len + 63) // 64
            raw = torch.randn(nb * 64, 1, 512, device=dev,
                              dtype=torch.bfloat16)
            data = raw.to(torch.float8_e4m3fn).view(torch.uint8)
            scales = torch.ones(nb * 64, 1, 16, device=dev,
                                dtype=torch.float32)
            kvc = torch.cat(
                [data.view(nb * 64, 1, 512),
                 scales.to(torch.float8_e8m0fnu).view(torch.uint8)],
                dim=-1).view(nb, 64, 1, 528)
            idx = torch.randint(0, nb * 64, (B, 1, topk),
                                device=dev, dtype=torch.int32)
            meta, _ = flash_mla.get_mla_metadata()
            out, lse = _call_many(
                c, lambda: flash_mla.flash_mla_with_kvcache(
                    q, kvc, None, None, 512, meta,
                    softmax_scale=512 ** -0.55, causal=False,
                    is_fp8_kvcache=True, indices=idx,
                    enable_batch_invariant=bool(c.get("no_split", False))))
            proof = f"sparse_decode out={tuple(out.shape)}"
        elif kind == "sparse_prefill":
            S_q, Hq, _dqk, topk, kv_len = s
            q = torch.randn(S_q, Hq, 512, device=dev, dtype=torch.bfloat16)
            kv = torch.randn(kv_len, 1, 512, device=dev,
                             dtype=torch.bfloat16)
            idx = torch.randint(0, kv_len, (S_q, 1, topk),
                                device=dev, dtype=torch.int32)
            out, maxl, lse = _call_many(
                c, lambda: flash_mla.flash_mla_sparse_fwd(
                    q, kv, idx, sm_scale=512 ** -0.5, d_v=512))
            proof = f"sparse_prefill out={tuple(out.shape)}"
        elif kind == "dense_fwd":
            from flash_mla import flash_attn_varlen_func
            B, S, Hq, Hkv, D = s
            Sk = int(c.get("seqlen_k") or S)
            q = torch.randn(B * S, Hq, D, device=dev, dtype=torch.bfloat16)
            k = torch.randn(B * Sk, Hkv, D, device=dev, dtype=torch.bfloat16)
            v = torch.randn(B * Sk, Hkv, D, device=dev, dtype=torch.bfloat16)
            cu_q = torch.arange(0, (B + 1) * S, S, device=dev, dtype=torch.int32)
            cu_k = torch.arange(0, (B + 1) * Sk, Sk, device=dev, dtype=torch.int32)
            out = _call_many(
                c, lambda: flash_attn_varlen_func(
                    q, k, v, cu_q, cu_k, S, Sk,
                    softmax_scale=D ** -0.5, is_varlen=True,
                    causal=bool(p.get("causal", True))))
            proof = f"dense_fwd out={tuple(out[0].shape) if isinstance(out,tuple) else tuple(out.shape)}"
        elif kind == "dense_bwd":
            from flash_mla import flash_attn_varlen_func
            B, S, Hq, Hkv, D = s
            Sk = int(c.get("seqlen_k") or S)
            q_seed = torch.randn(B * S, Hq, D, device=dev, dtype=torch.bfloat16)
            k_seed = torch.randn(B * Sk, Hkv, D, device=dev, dtype=torch.bfloat16)
            v_seed = torch.randn(B * Sk, Hkv, D, device=dev, dtype=torch.bfloat16)
            dout = torch.randn(B * S, Hq, D, device=dev, dtype=torch.bfloat16)
            cu_q = torch.arange(0, (B + 1) * S, S, device=dev, dtype=torch.int32)
            cu_k = torch.arange(0, (B + 1) * Sk, Sk, device=dev, dtype=torch.int32)
            calls = int(c.get("calls", 1)) if c.get("mode") == "full" else 1
            for _ in range(calls):
                q = q_seed.detach().requires_grad_()
                k = k_seed.detach().requires_grad_()
                v = v_seed.detach().requires_grad_()
                out = flash_attn_varlen_func(
                    q, k, v, cu_q, cu_k, S, Sk,
                    softmax_scale=D ** -0.5, is_varlen=True,
                    causal=bool(p.get("causal", True)))
                o = out[0] if isinstance(out, tuple) else out
                o.backward(dout)
            proof = f"dense_bwd out={tuple(o.shape)}"
        elif kind == "norm_rope":
            from flash_mla.fused_norm_rope_attn_rope_cast import prefill
            s_q, h_q, d_qk, s_kv, topk = s
            q = torch.randn(s_q, h_q, d_qk, device=dev, dtype=torch.bfloat16)
            kv = torch.randn(s_kv, 1, d_qk, device=dev, dtype=torch.bfloat16)
            indices = torch.randint(
                s_kv, (s_q, 1, topk), device=dev, dtype=torch.int32)
            token_positions = torch.arange(s_q, device=dev, dtype=torch.int32)
            cos_sin_cache = _build_cos_sin_cache(s_q, dev)
            out = _call_many(c, lambda: prefill(
                enable_q_norm=True,
                rms_norm_eps=1e-4,
                token_positions=token_positions,
                is_rope_neox_style=False,
                rope_dim=64,
                cos_sin_cache=cos_sin_cache,
                n_wv_group=h_q // 8,
                num_per_channels=32,
                use_tma_aligned_col_major_sf=True,
                round_sf=True,
                use_packed_ue8m0=True,
                q=q,
                kv=kv,
                indices=indices,
                sm_scale=512 ** -0.5,
                d_v=512))
            proof = f"norm_rope out={tuple(out[0].shape)}"
        else:
            raise KeyError(kind)
        torch.cuda.synchronize()
        if c.get("mode") == "full":
            target = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(target.float()).all():
                raise RuntimeError("full-mode output contains non-finite values")
            if kind == "dense_fwd":
                import torch.nn.functional as F
                B, S, Hq, Hkv, D = s
                Sk = int(c.get("seqlen_k") or S)
                q_ref = q.view(B, S, Hq, D).transpose(1, 2)
                k_ref = k.view(B, Sk, Hkv, D).transpose(1, 2)
                v_ref = v.view(B, Sk, Hkv, D).transpose(1, 2)
                causal = bool(p.get("causal", True))
                if causal:
                    q_positions = torch.arange(S, device=dev) + (Sk - S)
                    kv_positions = torch.arange(Sk, device=dev)
                    attn_mask = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
                    ref = F.scaled_dot_product_attention(
                        q_ref, k_ref, v_ref, attn_mask=attn_mask,
                        is_causal=False, enable_gqa=(Hq != Hkv))
                else:
                    ref = F.scaled_dot_product_attention(
                        q_ref, k_ref, v_ref, is_causal=False,
                        enable_gqa=(Hq != Hkv))
                torch.testing.assert_close(
                    target.view(B, S, Hq, D), ref.transpose(1, 2),
                    rtol=2e-2, atol=2e-2)
                proof += "; full-mode SDPA reference checked after final call"
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        ok, err = False, f"{type(e).__name__}: {e}"
    write_result(SUITE, cid, {"id": cid, "suite": SUITE, "ok": ok,
                              "calls": int(c.get("calls", 1)),
                              "kind": c.get("kind"), "shape": c.get("shape_tag"),
                              "kernel": c.get("main_kernel"), "proof": proof,
                              "error": err})
    print("DONE", cid, "ok=", ok, err[:200])
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", nargs="?")
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.resolve:
        for c in _load_cases():
            if args.case_id is None or c["id"] == args.case_id:
                print("RESOLVE", json.dumps(c, sort_keys=True))
        return 0
    if args.all:
        rc = 0
        for c in _load_cases():
            rc |= run(c["id"])
        return rc
    return run(args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
