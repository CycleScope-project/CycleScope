#!/usr/bin/env python3
"""Far-holdout triton suite runner (suite `triton`, specs
suites/legacy/dev283/holdout_far_triton.json).

Modes:
  run_triton.py <CASE_ID> [--resolve] [--compile-only] [--out DIR]

  default      real call on the capture GPU: allocate inputs with the spec
               seed-derived values, launch the kernel once (grid=(1,) or one
               2-CTA cluster via num_ctas), write result.json via
               suites/common.py write_result (suite `triton`).
  --resolve    stop after spec resolution (no triton/torch import); prints
               `RESOLVE <id> family=<f> kernel=<k> shape=<s> num_ctas=<n>
               heldout=far` — the routing/dry-run evidence line.
  --compile-only  offline compile (GPU-less pod): triton.compile with an
               explicit GPUTarget (arch from TRITON_ARCH, default 103),
               dump cubin+ptx under --out, run the cuobjdump static gate via
               experiments/holdout_far/probe_sass.py semantics inline
               (identity fingerprint + family counts; sealed cases keep
               identity fingerprints only — enforced by far_compile.py, this
               path just reports compile ok/fail).

Single-cluster discipline: every kernel is single-tile; grid=(1,) for 1C and
one 2-CTA cluster for num_ctas=2 — the grid never depends on the shape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
sys.path.insert(0, str(ROOT / "suites"))

SPEC = ROOT / "suites" / "c450_full" / "cases_triton.json"
SUITE = "triton"


def load_case(cid: str) -> dict:
    data = json.loads(SPEC.read_text(encoding="utf-8"))
    for c in data["cases"]:
        if c["id"] == cid:
            return c
    raise KeyError(f"{cid} not in {SPEC}")


def resolve_line(c: dict) -> str:
    return (f"RESOLVE {c['id']} family={c['family']} kernel={c['entry']} "
            f"shape={c['shape_tag']} num_ctas={c['params']['num_ctas']} "
            f"cta_group={c['cta_group']} heldout=far")


def kernel_name(c: dict) -> str:
    return c["entry"].split(":", 1)[1]


def _dt(c):
    import torch
    return {"bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn,
            "fp16": torch.float16, "tf32": torch.float32,
            "fp32": torch.float32}[c["dtype"]]


def _check(name, c, fn, consts, num_ctas, dev):
    import torch
    dt = _dt(c)
    shp = c["shape"]
    if name in ("matmul_fp16", "matmul_bf16", "matmul_fp8", "matmul_tf32"):
        m, n, k = shp
        a = torch.randn(m, k, device=dev).to(dt)
        b = torch.randn(k, n, device=dev).to(dt)
        odt = dt if c["dtype"] in ("bf16", "fp16") else torch.float32
        out = torch.empty(m, n, device=dev, dtype=odt)
        fn[(1,)](a, b, out, **consts, num_ctas=num_ctas)
        ref = a.float() @ b.float()
    elif name == "matmul_bf16_gelu":
        m, n, k = shp
        a = torch.randn(m, k, device=dev).to(dt)
        b = torch.randn(k, n, device=dev).to(dt)
        bias = torch.randn(n, device=dev, dtype=dt)
        out = torch.empty(m, n, device=dev, dtype=torch.float32)
        fn[(1,)](a, b, bias, out, **consts, num_ctas=num_ctas)
        ref = torch.nn.functional.gelu(a.float() @ b.float() + bias.float())
    elif name == "matmul_tma":
        from triton.tools.tensor_descriptor import TensorDescriptor
        m, n, k = shp
        a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, n, device=dev, dtype=torch.float32)
        fn[(1,)](TensorDescriptor.from_tensor(a, (m, k)),
                 TensorDescriptor.from_tensor(b, (k, n)), out,
                 **consts, num_ctas=num_ctas)
        ref = a.float() @ b.float()
    elif name == "matmul_ws":
        m, n, k = shp
        a = torch.randn(m, k, device=dev).to(dt)
        b = torch.randn(k, n, device=dev).to(dt)
        out = torch.empty(m, n, device=dev, dtype=torch.float32)
        fn[(1,)](a, b, out, **consts, num_ctas=num_ctas)
        ref = a.float() @ b.float()
    elif name == "matmul_persist":
        m, n, k = shp
        tiles = int(consts["TILES"])
        a = torch.randn(tiles, m, k, device=dev).to(dt)
        b = torch.randn(tiles, k, n, device=dev).to(dt)
        out = torch.empty(tiles, m, n, device=dev, dtype=torch.float32)
        fn[(1,)](a, b, out, **consts, num_ctas=num_ctas)
        ref = torch.bmm(a.float(), b.float())
    elif name == "matmul_grouped":
        m, n, k = shp
        g = int(consts["G"])
        a = torch.randn(g, m, k, device=dev).to(dt)
        b = torch.randn(g, k, n, device=dev).to(dt)
        out = torch.empty(g, m, n, device=dev, dtype=torch.float32)
        fn[(1,)](a, b, out, **consts, num_ctas=num_ctas)
        ref = torch.bmm(a.float(), b.float())
    elif name == "matmul_scaled":
        m, n, k = shp
        a = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(k, n, device=dev).to(torch.float8_e4m3fn)
        # tl.dot_scaled takes the e8m0 scale as raw uint8 bytes
        # (triton's dtype map lacks torch.float8_e8m0fnu on this build).
        sa = torch.full((m, max(1, k // 32)), 0x7F, device=dev,
                        dtype=torch.uint8)
        sb = torch.full((k, max(1, n // 128)), 0x7F, device=dev,
                        dtype=torch.uint8)
        out = torch.empty(m, n, device=dev, dtype=torch.float32)
        fn[(1,)](a, sa, b, sb, out, **consts, num_ctas=num_ctas)
        ref = a.float() @ b.float()
    elif name in ("attn_fwd_bf16", "attn_fwd_causal"):
        causal = name == "attn_fwd_causal"
        _b, _h, sq, d = shp
        q = torch.randn(128, d, device=dev, dtype=torch.bfloat16)
        kk = torch.randn(sq, d, device=dev, dtype=torch.bfloat16)
        vv = torch.randn(sq, d, device=dev, dtype=torch.bfloat16)
        o = torch.empty(128, d, device=dev, dtype=torch.bfloat16)
        fn[(1,)](q, kk, vv, o, **consts, num_ctas=num_ctas)
        p = (q.float() @ kk.float().T) * 0.08838834764831845
        if causal:
            mask = torch.ones(128, sq, device=dev, dtype=torch.bool).tril()
            p = p.masked_fill(~mask, float("-inf"))
        ref = torch.softmax(p, dim=-1) @ vv.float()
        out = o
    elif name == "rmsnorm_fp32":
        rows, ncols = shp
        x = torch.randn(rows, ncols, device=dev, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        fn[(1,)](x, out, **consts, num_ctas=num_ctas)
        ref = x.float() * torch.rsqrt(
            (x.float() * x.float()).mean(-1, keepdim=True) + 1e-6)
    elif name == "softmax_rows":
        rows, ncols = int(consts["ROWS"]), int(consts["N"])
        x = torch.randn(rows, ncols, device=dev, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        fn[(1,)](x, out, **consts, num_ctas=num_ctas)
        ref = torch.softmax(x.float(), dim=-1)
    elif name == "layernorm_rows":
        rows, ncols = int(consts["ROWS"]), int(consts["N"])
        x = torch.randn(rows, ncols, device=dev, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        fn[(1,)](x, out, **consts, num_ctas=num_ctas)
        ref = torch.nn.functional.layer_norm(x.float(), (ncols,), eps=1e-6)
    elif name == "layernorm_rows_bwd":
        rows, ncols = int(consts["ROWS"]), int(consts["N"])
        x = torch.randn(rows, ncols, device=dev,
                        dtype=torch.bfloat16).requires_grad_()
        dy = torch.randn(rows, ncols, device=dev, dtype=torch.bfloat16)
        torch.nn.functional.layer_norm(x, (ncols,), eps=1e-6).backward(dy)
        out = torch.empty_like(x)
        fn[(1,)](dy, x.detach(), out, **consts, num_ctas=num_ctas)
        ref = x.grad.float()
    else:
        raise KeyError(name)
    torch.cuda.synchronize()
    denom = float(ref.float().abs().max()) or 1.0
    err = float((out.float() - ref.float()).abs().max()) / denom
    return err <= 0.05, f"rel_err={err:.4g}"


def run_real(cid: str) -> int:
    import torch

    from common import write_result

    c = load_case(cid)
    name = kernel_name(c)
    mods = __import__("triton_kernels", globals(), locals(),
                      fromlist=["define_kernels"])
    fn, _ = mods.define_kernels()[name]
    consts = mods.constexpr_for(name, c["shape"])
    num_ctas = int(c["params"]["num_ctas"])
    dev = torch.device("cuda")
    torch.manual_seed(20261011)
    ok = True
    err = ""
    proof_bits = [f"triton num_ctas={num_ctas}", f"constexpr={consts}"]
    try:
        passed, err = _check(name, c, fn, consts, num_ctas, dev)
        ok = passed
        if not ok:
            err = "selfcheck " + err
    except Exception as e:  # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    write_result(SUITE, cid, {
        "id": cid, "suite": SUITE, "ok": ok,
        "cta_group": c.get("cta_group"), "cluster": c.get("cluster"),
        "sms": c.get("sms"), "dtype": c.get("dtype"), "shape": c.get("shape_tag"),
        "kernel": c.get("entry"), "proof": " ".join(proof_bits), "error": err,
    })
    print("DONE", cid, "ok=", ok, err[:200])
    return 0 if ok else 1


def compile_only(cid: str, out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = load_case(cid)
    name = kernel_name(c)
    mods = __import__("triton_kernels", globals(), locals(),
                      fromlist=["define_kernels"])
    fn, cnames = mods.define_kernels()[name]
    consts = dict(zip(cnames, [int(x) for x in
                               _shape_to_consts(name, c["shape"])]))
    num_ctas = int(c["params"]["num_ctas"])
    arch = int(os.environ.get("TRITON_ARCH", "103"))
    entry: dict = {"id": cid, "compile": "fail", "arch": arch,
                   "num_ctas": num_ctas}
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as tc

        sig_spec = mods.KERNELS[name]
        kinds = sig_spec["sig"].split(",")
        names = fn.arg_names
        sigd = {n: ("constexpr" if n in consts else kinds[i])
                for i, n in enumerate(names)}
        try:
            src = ASTSource(fn, signature=sigd, constexprs=consts)
        except TypeError:
            src = ASTSource(fn, signature=sigd)
        k = tc(src, target=GPUTarget("cuda", arch, 32),
               options={"num_ctas": num_ctas})
        cubin = out / f"{cid}.cubin"
        cubin.write_bytes(k.asm["cubin"])
        (out / f"{cid}.ptx").write_text(k.asm.get("ptx", ""))
        entry.update({"compile": "ok",
                      "cubin_sha256": hashlib.sha256(
                          k.asm["cubin"]).hexdigest()})
    except Exception as e:  # noqa: BLE001
        entry.update({"compile": f"fail: {type(e).__name__}: {e}"[:400]})
    (out / f"{cid}.json").write_text(json.dumps(entry, indent=1))
    print("COMPILE", json.dumps(entry)[:300])
    return 0 if entry["compile"] == "ok" else 1


def _shape_to_consts(name: str, shape: list[int]) -> list[int]:
    if name.startswith("matmul"):
        return list(shape)
    if name == "attn_fwd_bf16":
        return [shape[2], shape[3]]
    return [shape[1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", nargs="?")
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--out", default="/tmp/far_triton")
    ap.add_argument("--all", action="store_true",
                    help="iterate every spec case (compile-only/dry-run)")
    args = ap.parse_args()
    if args.all:
        rc = 0
        for c in json.loads(SPEC.read_text(encoding="utf-8"))["cases"]:
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
