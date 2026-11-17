#!/usr/bin/env python3
"""Emit wgmma_impl.cuh: per-cell asm op + kernel + dispatch table.

Cells (protocol.json cells.h100):
  HGMMA f16 (acc F32 and F16), bf16, tf32   n = 8..256 step 8
  QGMMA e4m3.e4m3 / e4m3.e5m2 / e5m2.e5m2   n = 8..256 step 8
  IGMMA s8.s8                              n = ISA list (see NS_I8)
  every cell x {ss, rs}; m = 64.

Modes (probe.h100): lat U in {0,1}; cad U in {16,32,64,128} straight-line
in one commit group; wg2 = two warpgroups run the cad window (bf16 ss,
n in {64,128,256}).  Same window shape for both: t_s = clock64;
wgmma.fence; U x wgmma.mma_async (same D, scale-d = 1);
wgmma.commit_group; wgmma.wait_group 0; t_w = clock64.

Operand counts (PTX ISA "wgmma.mma_async" register fragments):
  D: n/2 regs for .f32 / .s32 accumulators; n/4 .b32 for .f16 acc.
  A (rs): 4 x .b32 per thread for every kind — the 64 x k A fragment
  is divided across the 128 threads of the warpgroup
  (64*k*elem_bytes/128 = 16 B for k=16 f16/bf16, k=8 tf32, k=32 fp8/i8).
"""

from __future__ import annotations

import sys

NS_FULL = list(range(8, 257, 8))
NS_I8 = [8, 16, 24, 32, 48, 64, 80, 96, 112, 128,
         144, 160, 176, 192, 208, 224, 240, 256]

# kind -> (k, acc, ptx dtypes, dreg_div, dest type)
KINDS = {
    "f16":   (16, "f32", "f16.f16",   2, "float"),
    "f16a":  (16, "f16", "f16.f16",   4, "std::uint32_t"),
    "bf16":  (16, "f32", "bf16.bf16", 2, "float"),
    "tf32":  (8,  "f32", "tf32.tf32", 2, "float"),
    "e4m3e4m3": (32, "f32", "e4m3.e4m3", 2, "float"),
    "e4m3e5m2": (32, "f32", "e4m3.e5m2", 2, "float"),
    "e5m2e5m2": (32, "f32", "e5m2.e5m2", 2, "float"),
    "s8s8":  (32, "s32", "s8.s8",     2, "int"),
    # protocol-v2 sparse addendum: .sp forms, metadata reg + sel 0
    "sp_f16":  (32, "f32", "f16.f16",   2, "float"),
    "sp_bf16": (32, "f32", "bf16.bf16", 2, "float"),
    "sp_e4m3e4m3": (64, "f32", "e4m3.e4m3", 2, "float"),
}
CLS = {"f16": "HGMMA", "f16a": "HGMMA", "bf16": "HGMMA", "tf32": "HGMMA",
       "e4m3e4m3": "QGMMA", "e4m3e5m2": "QGMMA", "e5m2e5m2": "QGMMA",
       "s8s8": "IGMMA",
       "sp_f16": "HGMMA", "sp_bf16": "HGMMA", "sp_e4m3e4m3": "QGMMA"}
SP_KINDS = {"sp_f16", "sp_bf16", "sp_e4m3e4m3"}
AREGS = 4  # .b32 per thread, all kinds (PTX ISA wgmma register fragment)


def cell_list() -> list[tuple]:
    cells = []
    for kind, (k, acc, dt, _dd, _t) in KINDS.items():
        ns = NS_I8 if kind == "s8s8" else NS_FULL
        for n in ns:
            for src in ("ss", "rs"):
                cells.append((kind, CLS[kind], k, acc, dt, n, src))
    return cells


def is_sp(kind: str) -> bool:
    return kind in SP_KINDS


# trailing immediates after the B operand (PTX ISA wgmma.mma_async):
#   16-bit kinds: ss scale-d,scale-a,scale-b,tnsp-a,tnsp-b = "1,1,1,0,0"
#                 rs drops tnsp-a                          = "1,1,1,0"
#   fp8 (e4m3/e5m2): scale-d,scale-a,scale-b (no tnsp)     = "1,1,1"
#   tf32: scale-d,scale-a,scale-b (no tnsp, ss and rs)     = "1,1,1"
#   s8: scale-d only (no scales, no tnsp)                  = "1"
TAILS = {
    ("f16", "ss"): "1, 1, 1, 0, 0", ("f16", "rs"): "1, 1, 1, 0",
    ("f16a", "ss"): "1, 1, 1, 0, 0", ("f16a", "rs"): "1, 1, 1, 0",
    ("bf16", "ss"): "1, 1, 1, 0, 0", ("bf16", "rs"): "1, 1, 1, 0",
    ("tf32", "ss"): "1, 1, 1", ("tf32", "rs"): "1, 1, 1",
    ("e4m3e4m3", "ss"): "1, 1, 1", ("e4m3e4m3", "rs"): "1, 1, 1",
    ("e4m3e5m2", "ss"): "1, 1, 1", ("e4m3e5m2", "rs"): "1, 1, 1",
    ("e5m2e5m2", "ss"): "1, 1, 1", ("e5m2e5m2", "rs"): "1, 1, 1",
    ("s8s8", "ss"): "1", ("s8s8", "rs"): "1",
}


def asm_for(kind: str, n: int, src: str, idx: int) -> str:
    k, acc, dt, ddiv, dt_t = KINDS[kind]
    nd = n // ddiv
    d_list = ", ".join(f"%{i}" for i in range(nd))
    outs = ", ".join(f'"+{"r" if acc in ("f16", "s32") else "f"}"(d[{i}])'
                     for i in range(nd))
    sp = is_sp(kind)
    tail = TAILS.get((kind, src)) or TAILS[(kind[3:], src)]
    if sp:
        # wgmma.mma_async.sp: d, a, b-desc, sp-meta(.b32 reg),
        # sparsity-selector imm, then the dense trailing immediates
        base = f'"wgmma.mma_async.sp.sync.aligned.m64n{n}k{k}.{acc}.{dt} '
        if src == "ss":
            ins = (base + f'{{{d_list}}}, %{nd}, %{nd + 1}, '
                   f'%{nd + 2}, 0, {tail};"')
            ins_in = '"l"(da), "l"(db), "r"(meta)'
        else:
            a_list = ", ".join(f"%{nd + i}" for i in range(AREGS))
            ins = (base + f'{{{d_list}}}, {{{a_list}}}, '
                   f'%{nd + AREGS}, %{nd + AREGS + 1}, 0, {tail};"')
            ins_in = (", ".join(f'"r"(a[{i}])' for i in range(AREGS))
                      + ', "l"(db), "r"(meta)')
    elif src == "ss":
        ins = (f'"wgmma.mma_async.sync.aligned.m64n{n}k{k}.{acc}.{dt} '
               f'{{{d_list}}}, %{nd}, %{nd + 1}, {tail};"')
        ins_in = '"l"(da), "l"(db)'
    else:
        a_list = ", ".join(f"%{nd + i}" for i in range(AREGS))
        ins = (f'"wgmma.mma_async.sync.aligned.m64n{n}k{k}.{acc}.{dt} '
               f'{{{d_list}}}, {{{a_list}}}, %{nd + AREGS}, {tail};"')
        ins_in = (", ".join(f'"r"(a[{i}])' for i in range(AREGS))
                  + ', "l"(db)')
    return f'asm volatile({ins} : {outs} : {ins_in});'


ONLY_SRC = None


def cells_now():
    # returns (global_cid, cell_tuple) pairs; filter preserves global ids
    cl = list(enumerate(cell_list()))
    if ONLY_SRC:
        cl = [(i, c) for i, c in cl if c[6] == ONLY_SRC]
    return cl


def emit_cells() -> str:
    out = []
    for i, (kind, _cls, _k, _acc, dt, n, src) in cells_now():
        _k, _acc, _dt, ddiv, dt_t = KINDS[kind]
        nd = n // ddiv
        ab = (f"std::uint32_t a[{AREGS}] = {{1u,1u,1u,1u}};\n      "
              if src == "rs" else "")
        if is_sp(kind):
            ab += "std::uint32_t meta = 0u;\n      "
        out.append(
            f"template <> struct WOp<{i}> {{\n"
            f"  static __device__ __forceinline__ void op({dt_t}* d,\n"
            f"      std::uint64_t da, std::uint64_t db) {{\n"
            f"      {ab}{asm_for(kind, n, src, i)}\n  }}\n}};\n")
    return "\n".join(out)


def emit_kernel() -> str:
    return r"""
template <int CID, int U, int WG>
__global__ void __launch_bounds__(WG * 128, 1) k_wgmma(
    std::uint64_t* out, std::uint32_t* dval, int reps,
    int wgroups, std::uint32_t seed) {
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  const int warpgroup = threadIdx.x >> 7;
  if (warpgroup >= wgroups) return;
  using T = typename WD<CID>::dty;
  T dd[WD<CID>::ND];
  for (int i = 0; i < WD<CID>::ND; ++i) dd[i] = T{};
  const int lane = threadIdx.x & 31;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const std::uint64_t t_s = mb_clock();
    // cad U>16: runtime loop of a straight 16-op body (r3 addendum:
    // keeps the unrolled window at 16 regs' worth of live wgmma state)
    if constexpr (U > 16) {
#pragma unroll 1
      for (int g = 0; g < U / 16; ++g) {
#pragma unroll
        for (int i = 0; i < 16; ++i) WOp<CID>::op(dd, da, db);
      }
    } else {
#pragma unroll
      for (int i = 0; i < U; ++i) WOp<CID>::op(dd, da, db);
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    const std::uint64_t t_w = mb_clock();
    out[2 * rep] = t_s;
    out[2 * rep + 1] = t_w;
  }
  if (threadIdx.x == 0) {
    std::uint32_t v;
    std::memcpy(&v, &dd[0], sizeof(v) < sizeof(T) ? sizeof(v) : sizeof(T));
    dval[0] = v;
  }
}

// serialized-latency cells (protocol-v2 addendum): one warpgroup, the
// timed window holds U complete iterations of
//   fence; G x mma_async (same D); commit_group; wait_group 0
// MODE 0 -> G = 1 (ser), MODE 1 -> G = 4 (grp).  The kernel never runs
// with two warpgroups (wg2 is meaningless here).
template <int CID, int MODE, int U>
__global__ void __launch_bounds__(128, 1) k_sergrp(
    std::uint64_t* out, std::uint32_t* dval, int reps,
    int wgroups, std::uint32_t seed) {
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  const int warpgroup = threadIdx.x >> 7;
  if (warpgroup >= wgroups) return;
  using T = typename WD<CID>::dty;
  T dd[WD<CID>::ND];
  for (int i = 0; i < WD<CID>::ND; ++i) dd[i] = T{};
  const int lane = threadIdx.x & 31;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t_s = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int j = 0; j < (MODE ? 4 : 1); ++j) WOp<CID>::op(dd, da, db);
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    const std::uint64_t t_w = mb_clock();
    out[2 * rep] = t_s;
    out[2 * rep + 1] = t_w;
  }
  if (threadIdx.x == 0) {
    std::uint32_t v;
    std::memcpy(&v, &dd[0], sizeof(v) < sizeof(T) ? sizeof(v) : sizeof(T));
    dval[0] = v;
  }
}
"""


LAT_US = (0, 1)
CAD_US = (16, 32, 64, 128)
SER_US = (1, 2, 4, 8)
GRP_US = (1, 2, 4)
GRP_N = (64, 128, 256)
GRP_KINDS = ("bf16", "e4m3e4m3")


def emit_table() -> str:
    rows = []
    seen = set()
    for i, (kind, cls, k, acc, _dt, n, src) in cells_now():
        if kind not in seen:
            if seen:
                rows.append("#endif")
            seen.add(kind)
            rows.append(f"#if WGMMA_KIND_{kind.upper()}")
        wg2 = (kind == "bf16" and src == "ss"
               and n in (64, 128, 256))
        for u in LAT_US:
            rows.append(
                f'  {{"{cls}","{kind}","{acc}",{k},{n},"{src}",'
                f'"lat",{u},&k_wgmma<{i},{u},1>,'
                + (f'&k_wgmma<{i},{u},2>' if wg2 else 'nullptr')
                + '},')
        for u in CAD_US:
            rows.append(
                f'  {{"{cls}","{kind}","{acc}",{k},{n},"{src}",'
                f'"cad",{u},&k_wgmma<{i},{u},1>,'
                + (f'&k_wgmma<{i},{u},2>' if wg2 else 'nullptr')
                + '},')
        # ser/grp (h100-r4 addendum): serialized single/group latency
        for u in SER_US:
            rows.append(
                f'  {{"{cls}","{kind}","{acc}",{k},{n},"{src}",'
                f'"ser",{u},&k_sergrp<{i},0,{u}>,nullptr}},')
        if src == "ss" and kind in GRP_KINDS and n in GRP_N:
            for u in GRP_US:
                rows.append(
                    f'  {{"{cls}","{kind}","{acc}",{k},{n},"{src}",'
                    f'"grp",{u},&k_sergrp<{i},1,{u}>,nullptr}},')
    rows.append("#endif  // WGMMA_KIND_*")
    return ("""
struct WCell {
  const char* cls; const char* kind; const char* acc;
  int k; int n; const char* src; const char* mode; int u;
  WKern kern; WKern kern2;
};
static const WCell W_CELLS[] = {
""" + "\n".join(rows) + """
};
static const int W_CELLS_N = sizeof(W_CELLS) / sizeof(WCell);
""")


def main() -> None:
    global ONLY_SRC
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--only-src="):
            ONLY_SRC = a.split("=", 1)[1]
    out = args[0]
    jout = args[1] if len(args) > 1 else None
    if jout:
        import json
        cells = [{"cid": i, "class": cls, "kind": kind, "k": k,
                  "acc": acc, "n": n, "src": src, "sp": is_sp(kind)}
                 for i, (kind, cls, k, acc, _d, n, src)
                 in cells_now()]
        with open(jout, "w") as jf:
            json.dump(cells, jf)
    fwd = (
        "// generated by gen_wgmma.py -- do not edit\n"
        "template <int CID> struct WOp;\n"
        "template <int CID> struct WD;\n"
        "using WKern = void (*)(std::uint64_t*, std::uint32_t*, int,\n"
        "                       int, std::uint32_t);\n\n")
    # dest-size traits per cell
    nd_rows = []
    for i, (kind, _c, _k, _a, _d, n, _s) in cells_now():
        nd = n // KINDS[kind][3]
        dt_t = KINDS[kind][4]
        nd_rows.append(
            f"template <> struct WD<{i}> {{ "
            f"static constexpr int ND = {nd}; "
            f"using dty = {dt_t}; }};")
    with open(out, "w") as f:
        f.write(fwd + "\n".join(nd_rows) + "\n\n"
                + emit_cells() + emit_kernel() + emit_table())


if __name__ == "__main__":
    main()
