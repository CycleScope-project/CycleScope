"""Canonical Triton kernel sources for the far-holdout triton suite
(holdout_far_triton.json).

Deliberately tutorial-plain, no CUTLASS anywhere: these kernels ARE the
far-holdout units (non-CUTLASS codegen).  Every kernel is single-tile
(M/N/... constexpr, grid=(1,) at capture, or one 2-CTA cluster with
num_ctas=2) per the single-cluster discipline; shapes come from the spec
params, never edited per case.

The same module serves three consumers:
  * run_triton.py (capture-time real call, GPU)
  * run_triton.py --compile-only (offline compile on a GPU-less pod:
    triton.compile with an explicit GPUTarget — the far-holdout offline
    compile path, mirrors the probe evidence)
  * tests (import-level checks only; no torch/triton import at module scope)
"""
from __future__ import annotations

# signature strings (triton ASTSource style) per kernel, and the constexpr
# params consumed from the spec shape.
KERNELS = {
    "matmul_bf16": {
        "sig": "*bf16,*bf16,*bf16",
        "dtypes": ("bf16", "bf16", "bf16"),
        "loop": "none",
    },
    "matmul_fp8": {
        "sig": "*fp8e4nv,*fp8e4nv,*fp32",
        "dtypes": ("fp8e4nv", "fp8e4nv", "fp32"),
        "loop": "none",
    },
    "matmul_tf32": {
        # tf32 is a DOT input-precision mode, not a storage dtype: fp32
        # storage + tl.dot(..., input_precision="tf32")
        "sig": "*fp32,*fp32,*fp32",
        "dtypes": ("fp32", "fp32", "fp32"),
        "loop": "none",
    },
    "attn_fwd_bf16": {
        "sig": "*bf16,*bf16,*bf16,*bf16",
        "dtypes": ("bf16", "bf16", "bf16", "bf16"),
        "loop": "kv",
    },
    "rmsnorm_fp32": {
        "sig": "*bf16,*bf16",
        "dtypes": ("bf16", "bf16"),
        "loop": "none",
    },
}


def define_kernels():  # pragma: no cover - needs triton at call time only
    """Build and return {name: (jit_fn, constexpr_names)} lazily."""
    import triton
    import triton.language as tl

    @triton.jit
    def matmul_bf16(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
            offs_k = k0 + tl.arange(0, TILE_K)
            a = tl.load(a_ptr + offs_m[:, None] * KK + offs_k[None, :])
            b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))
    @triton.jit
    def matmul_fp8(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
            offs_k = k0 + tl.arange(0, TILE_K)
            a = tl.load(a_ptr + offs_m[:, None] * KK + offs_k[None, :])
            b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
    @triton.jit
    def matmul_tf32(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
            offs_k = k0 + tl.arange(0, TILE_K)
            a = tl.load(a_ptr + offs_m[:, None] * KK + offs_k[None, :])
            b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b, input_precision="tf32")
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
    @triton.jit
    def attn_fwd_bf16(q_ptr, k_ptr, v_ptr, o_ptr, S: tl.constexpr,
                      D: tl.constexpr):
        offs_m = tl.arange(0, 128)
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :])
        acc = tl.zeros((128, D), dtype=tl.float32)
        m_i = tl.full((128,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((128,), dtype=tl.float32)
        for start_n in range(0, S, 128):
            offs_n = start_n + tl.arange(0, 128)
            kk = tl.load(k_ptr + offs_n[:, None] * D + offs_d[None, :])
            qk = tl.dot(q, tl.trans(kk))
            qk = qk * 0.08838834764831845
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            vv = tl.load(v_ptr + offs_n[:, None] * D + offs_d[None, :])
            acc = tl.dot(p.to(tl.bfloat16), vv, acc)
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :],
                 acc.to(tl.bfloat16))

    @triton.jit
    def rmsnorm_fp32(x_ptr, y_ptr, ROWS: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, N)
        for row in range(0, ROWS):
            x = tl.load(x_ptr + row * N + offs).to(tl.float32)
            var = tl.sum(x * x, 0) / N
            y = x * tl.rsqrt(var + 1e-6)
            tl.store(y_ptr + row * N + offs, y.to(tl.bfloat16))


    @triton.jit
    def matmul_fp16(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
            offs_k = k0 + tl.arange(0, TILE_K)
            a = tl.load(a_ptr + offs_m[:, None] * KK + offs_k[None, :])
            b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
    @triton.jit
    def matmul_bf16_gelu(a_ptr, b_ptr, bias_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
            offs_k = k0 + tl.arange(0, TILE_K)
            acc += tl.dot(tl.load(a_ptr + offs_m[:, None] * KK + offs_k[None, :]),
                          tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :]))
        acc = acc + tl.load(bias_ptr + offs_n)[None, :]
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc * 0.5 * (1 + tl.math.erf(acc * 0.70710678)))
    @triton.jit
    def matmul_tma(desc_a, desc_b, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
            a = desc_a.load([0, k0])
            b = desc_b.load([k0, 0])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
    @triton.jit
    def matmul_ws(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
            offs_k = k0 + tl.arange(0, TILE_K)
            acc += tl.dot(tl.load(a_ptr + offs_m[:, None] * KK + offs_k[None, :]),
                          tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :]))
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    @triton.jit
    def matmul_persist(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILES: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        for t in range(0, TILES):
            acc = tl.zeros((M, N), dtype=tl.float32)
            for k0 in range(0, KK, TILE_K):
                offs_k = k0 + tl.arange(0, TILE_K)
                acc += tl.dot(tl.load(a_ptr + t*M*KK + offs_m[:, None]*KK + offs_k[None, :]),
                              tl.load(b_ptr + t*KK*N + offs_k[:, None]*N + offs_n[None, :]))
            tl.store(c_ptr + t*M*N + offs_m[:, None]*N + offs_n[None, :], acc)
    @triton.jit
    def matmul_scaled(a_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
            offs_k = k0 + tl.arange(0, TILE_K)
            a = tl.load(a_ptr + offs_m[:, None] * KK + offs_k[None, :])
            sa = tl.load(sa_ptr + offs_m[:, None] * (KK // 32) + k0 // 32 + tl.arange(0, TILE_K // 32)[None, :])
            b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
            acc += tl.dot_scaled(a, sa, "e4m3", b, None, "e4m3")
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
    @triton.jit
    def matmul_grouped(a_ptr, b_ptr, c_ptr, G: tl.constexpr, M: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TILE_K: tl.constexpr):
        offs_m = tl.arange(0, M); offs_n = tl.arange(0, N)
        for g in range(0, G):
            acc = tl.zeros((M, N), dtype=tl.float32)
            for k0 in tl.range(0, KK, TILE_K, warp_specialize=True):
                offs_k = k0 + tl.arange(0, TILE_K)
                acc += tl.dot(tl.load(a_ptr + g*M*KK + offs_m[:, None]*KK + offs_k[None, :]),
                              tl.load(b_ptr + g*KK*N + offs_k[:, None]*N + offs_n[None, :]))
            tl.store(c_ptr + g*M*N + offs_m[:, None]*N + offs_n[None, :], acc)
    @triton.jit
    def layernorm_rows(x_ptr, y_ptr, ROWS: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, N)
        for r in range(0, ROWS):
            x = tl.load(x_ptr + r * N + offs).to(tl.float32)
            mu = tl.sum(x, 0) / N
            xc = x - mu
            var = tl.sum(xc * xc, 0) / N
            tl.store(y_ptr + r * N + offs, (xc * tl.rsqrt(var + 1e-6)).to(tl.bfloat16))

    @triton.jit
    def layernorm_rows_bwd(dy_ptr, x_ptr, dx_ptr, ROWS: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, N)
        for r in range(0, ROWS):
            x = tl.load(x_ptr + r * N + offs).to(tl.float32)
            dy = tl.load(dy_ptr + r * N + offs).to(tl.float32)
            mu = tl.sum(x, 0) / N
            xc = x - mu
            var = tl.sum(xc * xc, 0) / N
            rstd = tl.rsqrt(var + 1e-6)
            dxc = dy * rstd
            dx = dxc - tl.sum(dxc, 0) / N - xc * tl.sum(dxc * xc, 0) / (N * var + 1e-6 * N)
            tl.store(dx_ptr + r * N + offs, dx.to(tl.bfloat16))


    @triton.jit
    def softmax_rows(x_ptr, y_ptr, ROWS: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, N)
        for r in range(0, ROWS):
            x = tl.load(x_ptr + r * N + offs).to(tl.float32)
            x = x - tl.max(x, 0)
            e = tl.exp(x)
            tl.store(y_ptr + r * N + offs, (e / tl.sum(e, 0)).to(tl.bfloat16))

    @triton.jit
    def attn_fwd_causal(q_ptr, k_ptr, v_ptr, o_ptr, S: tl.constexpr, D: tl.constexpr):
        offs_m = tl.arange(0, 128)
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :])
        acc = tl.zeros((128, D), dtype=tl.float32)
        m_i = tl.full((128,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((128,), dtype=tl.float32)
        for start_n in range(0, S, 128):
            offs_n = start_n + tl.arange(0, 128)
            kk = tl.load(k_ptr + offs_n[:, None] * D + offs_d[None, :])
            qk = tl.dot(q, tl.trans(kk))
            qk = qk * 0.08838834764831845
            qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            vv = tl.load(v_ptr + offs_n[:, None] * D + offs_d[None, :])
            acc = tl.dot(p.to(tl.bfloat16), vv, acc)
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :], acc.to(tl.bfloat16))

    return {
        "matmul_fp16": (matmul_fp16, ("M", "N", "KK", "TILE_K")),
        "matmul_bf16_gelu": (matmul_bf16_gelu, ("M", "N", "KK", "TILE_K")),
        "matmul_tma": (matmul_tma, ("M", "N", "KK", "TILE_K")),
        "matmul_ws": (matmul_ws, ("M", "N", "KK", "TILE_K")),
        "matmul_persist": (matmul_persist, ("M", "N", "KK", "TILES", "TILE_K")),
        "matmul_scaled": (matmul_scaled, ("M", "N", "KK", "TILE_K")),
        "matmul_grouped": (matmul_grouped, ("G", "M", "N", "KK", "TILE_K")),
        "layernorm_rows": (layernorm_rows, ("ROWS", "N")),
        "layernorm_rows_bwd": (layernorm_rows_bwd, ("ROWS", "N")),
        "softmax_rows": (softmax_rows, ("ROWS", "N")),
        "attn_fwd_causal": (attn_fwd_causal, ("S", "D")),
        "matmul_bf16": (matmul_bf16, ("M", "N", "KK", "TILE_K")),
        "matmul_fp8": (matmul_fp8, ("M", "N", "KK", "TILE_K")),
        "matmul_tf32": (matmul_tf32, ("M", "N", "KK", "TILE_K")),
        "attn_fwd_bf16": (attn_fwd_bf16, ("S", "D")),
        "rmsnorm_fp32": (rmsnorm_fp32, ("ROWS", "N")),
    }


def constexpr_for(name: str, shape: list[int]) -> dict:
    """Spec shape -> constexpr dict for the kernel (fixed mapping)."""
    if name == "matmul_persist":
        m, n, k = shape
        return {"M": m, "N": n, "KK": k, "TILES": 4, "TILE_K": min(64, k)}
    if name == "matmul_grouped":
        m, n, k = shape
        return {"G": 4, "M": m, "N": n, "KK": k, "TILE_K": min(64, k)}
    if name == "matmul_ws":
        m, n, k = shape
        return {"M": m, "N": n, "KK": k, "TILE_K": min(64, k)}
    if name.startswith("matmul"):
        m, n, k = shape
        return {"M": m, "N": n, "KK": k, "TILE_K": min(64, k)}
    if name.startswith("attn_fwd"):
        _b, _h, s, d = shape
        return {"S": s, "D": d}
    if name == "softmax_rows":
        return {"ROWS": 32, "N": shape[1]}
    if name == "rmsnorm_fp32":
        return {"ROWS": 32, "N": shape[1]}
    if name.startswith("layernorm_rows"):
        return {"ROWS": 32, "N": shape[1]}
    raise KeyError(name)
