// tools/microbench/wgmma_overlap/probe/overlap_v4.cu — H100 (sm_90a).
//
// protocol-v6.json: overlap_v3.cu plus W = 3 and two softmax-like
// window kinds (smx: I % 5 -> ex2 / fma / fma / max.NaN / add; smxc:
// the ex2 unit replaced by an fma on a8/b8).  U = 140, G = 16.
// Run 2 (protocol-v6 addendum 1): kinds mufu8 / mufu_rsq8 / shfl8 at
// U = 64 plus smx; forms bf16 n64/n128/n256 ss and n256 rs; G = 32 for
// n64 / n128 and 16 for n256; self.w1-w2 cells for mufu8 and shfl8 on
// n128 ss / n256 rs; a smx g8 other.w2 tail cell; and a record-only
// mufu_ts kernel (32 x {ex2; clock64} per window warp, offsets to an
// extra buffer).
// Run 3 (protocol-v6 addendum 2): mufu2 / mufu4 units; self.w1 G
// sweep; MODE 4 "both" (each warpgroup issues and waits, no window);
// per-unit-clock cells mufu_ts / shfl_ts / ffma_ts / lds_ts on n128
// ss G = 24 other.w2 + ctrl.w1; revisit n256 ss g8 other.w2.
// protocol-v4.json: kernel operand forms of the issuing warpgroup's
// streams (uniform / constant-bank operand, .reuse, F2FP, MUFU) while
// its wgmma ops are in flight.  Derived from overlap_v2.cu (v2/v3 data):
// streams (per op shape N, independent of read count / G); C2 =
// dependent-issue latency L of co-resident warps (park control
// separates wgmma cause from parked-warp cause).
// One CTA of 4*W warps (W in {1,2}), role-split rep loops as v1
// addendum 2 (issue / window / wait in one straight-line region;
// named barrier 1 with the full CTA count).  Per rep (issuer role):
//   bar1; t_i0; wgmma.fence; G x (op; commit_group); t_i1; bar1;
//   [self: t_s; window; t_w]; wgmma.wait_group 0; t_d; record.
// Window-only role: bar1; bar1; t_s; window; t_w; record.
// park: wg0 issues no wgmma; after its two bar1 it parks at
// bar.sync 2,256, which wg1 joins after its window (C2 control).
// Lane 0 of every warp records (t_i0, t_i1, t_s, t_w, t_d); inactive
// fields are literal 0.  One jsonl row per (variant, launch): samples
// = active warps x reps (rep>=1) t_w - t_s; "tails" = wg0 t_d - max
// active t_w; "spans" = t_d - min active t_s; "issue" = wg0 t_i1-t_i0.
// Descriptor/op helpers come from the generated wgmma_impl.cuh
// (tensor_pipe/probe/gen_wgmma.py); rs cells supply their own A
// register fragment inside WOp<CID>::op.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../tensor_pipe/mb.cuh"

// no-swizzle K-major descriptor (same as wgmma_handoff/probe/handoff.cu).
__device__ __forceinline__ std::uint64_t w_desc(const void* smem_ptr) {
  const std::uint32_t addr =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem_ptr));
  std::uint64_t desc = 0;
  desc |= static_cast<std::uint64_t>((addr >> 4) & 0x3fff);
  desc |= static_cast<std::uint64_t>((128 >> 4) & 0x3fff) << 16;
  desc |= static_cast<std::uint64_t>((256 >> 4) & 0x3fff) << 32;
  return desc;
}

// all kinds on so the CID enumeration matches gen_wgmma.py cell_list()
#include "wgmma_impl.cuh"

static constexpr int kSmemBytes = 48 * 1024;

// v2 window kinds (numbering follows contention_v3 / overlap.cu):
//   0 dep3r  2 dep2  9 fmnmx  8 imad   (dependent chains, stall 4)
//   4 ilp8   5 ilp8n  15 fadd1  16 fmul2  (streams)
struct CellArgs {
  float a, b;          // reusable pair (dep3r/dep2/ilp8)
  float c[4];          // fmnmx max.NaN ring
  float a8[8], b8[8];  // ilp8n / fmul2: distinct, never reusable
  unsigned ia, ib;     // imad
  float p;             // warp-uniform multiplier (kernel parameter, no +z):
                       // ptxas may place it in a UR or as a c[0x0][..] operand
};

#define g_sink (*sinkp)
template <int V, int I>
__device__ __forceinline__ void unit(float* d, const CellArgs& o,
                                     unsigned* sinkp) {
  if constexpr (V == 0) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[0]) : "f"(o.a), "f"(o.b));
  } else if constexpr (V == 2) {
    asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                 : "+f"(d[0]) : "f"(o.a));
  } else if constexpr (V == 4) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[I & 7]) : "f"(o.a), "f"(o.b));
  } else if constexpr (V == 5) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[I & 7]) : "f"(o.a8[I & 7]), "f"(o.b8[I & 7]));
  } else if constexpr (V == 8) {
    unsigned* r = reinterpret_cast<unsigned*>(d);
    asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                 : "+r"(r[0]) : "r"(o.ia), "r"(o.ib));
  } else if constexpr (V == 9) {
    // max.NaN defeats ptxas's FMNMX3 pairing (contention v1)
    asm volatile("max.NaN.f32 %0, %0, %1;"
                 : "+f"(d[0]) : "f"(o.c[I & 3]));
  } else if constexpr (V == 15) {
    // fadd1: 1 GPR read + immediate, no reuse
    asm volatile("add.f32 %0, %0, 0f3F800000;" : "+f"(d[I & 7]));
  } else if constexpr (V == 16) {
    // fmul2: 2 distinct GPR reads, no reuse
    asm volatile("mul.f32 %0, %0, %1;"
                 : "+f"(d[I & 7]) : "f"(o.a8[I & 7]));
  } else if constexpr (V == 17) {
    // fmul_r: 2 GPR reads, the shared multiplier a is reuse-served
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(d[I & 7]) : "f"(o.a));
  } else if constexpr (V == 18) {
    // fadd2n: 2 distinct GPR reads, no reuse (FADD pipe)
    asm volatile("add.f32 %0, %0, %1;"
                 : "+f"(d[I & 7]) : "f"(o.a8[I & 7]));
  } else if constexpr (V == 19) {
    // ffma_ur: plain C so ptxas may keep the uniform p out of the GPRs
    // (UR or c[] operand); 2 GPR reads, no reuse
    d[I & 7] = fmaf(d[I & 7], o.p, o.b8[I & 7]);
  } else if constexpr (V == 20) {
    // ffma_ur_r: uniform p + shared GPR a (reuse-served)
    d[I & 7] = fmaf(d[I & 7], o.p, o.a);
  } else if constexpr (V == 21) {
    // fmul_ur: 1 GPR read + uniform
    d[I & 7] = d[I & 7] * o.p;
  } else if constexpr (V == 22) {
    // f2fp: F2FP.F16.F32.PACK_AB of two GPRs; the result goes to a
    // per-unit sink register (asm volatile keeps every unit)
    unsigned h;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;"
                 : "=r"(h) : "f"(d[(2 * I) & 7]), "f"(d[(2 * I + 1) & 7]));
    g_sink ^= h;
  } else if constexpr (V == 23) {
    // mufu8: 8 independent MUFU.EX2 chains (1 GPR read, XU pipe)
    asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(d[I & 7]));
  } else if constexpr (V == 28) {
    // mufu2: 2 independent MUFU.EX2 chains
    asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(d[I & 1]));
  } else if constexpr (V == 29) {
    // mufu4: 4 independent MUFU.EX2 chains
    asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(d[I & 3]));
  } else if constexpr (V == 26) {
    // mufu_rsq8: 8 independent MUFU.RSQ chains (1 GPR read, XU pipe)
    asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(d[I & 7]));
  } else if constexpr (V == 27) {
    // shfl8: butterfly shuffle by 1 of the bit pattern of d[I & 7]
    unsigned* u = reinterpret_cast<unsigned*>(d);
    asm volatile(
        "shfl.sync.bfly.b32 %0, %0, 1, 31, 0xffffffff;"
        : "+r"(u[I & 7]));
  } else if constexpr (V == 24) {
    // smx: softmax-like mix (I % 5): ex2 / fma / fma / max.NaN / add
    if constexpr (I % 5 == 0) {
      asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(d[(I / 5) & 3]));
    } else if constexpr (I % 5 == 1) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                   : "+f"(d[4]) : "f"(o.a), "f"(o.b));
    } else if constexpr (I % 5 == 2) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                   : "+f"(d[5]) : "f"(o.a), "f"(o.b));
    } else if constexpr (I % 5 == 3) {
      asm volatile("max.NaN.f32 %0, %0, %1;"
                   : "+f"(d[6]) : "f"(o.c[I & 3]));
    } else {
      asm volatile("add.f32 %0, %0, %1;"
                   : "+f"(d[7]) : "f"(o.c[I & 3]));
    }
  } else if constexpr (V == 25) {
    // smxc: smx with the ex2 unit replaced by an fma on a8/b8
    if constexpr (I % 5 == 0) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                   : "+f"(d[(I / 5) & 3])
                   : "f"(o.a8[(I / 5) & 3]), "f"(o.b8[(I / 5) & 3]));
    } else if constexpr (I % 5 == 1) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                   : "+f"(d[4]) : "f"(o.a), "f"(o.b));
    } else if constexpr (I % 5 == 2) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                   : "+f"(d[5]) : "f"(o.a), "f"(o.b));
    } else if constexpr (I % 5 == 3) {
      asm volatile("max.NaN.f32 %0, %0, %1;"
                   : "+f"(d[6]) : "f"(o.c[I & 3]));
    } else {
      asm volatile("add.f32 %0, %0, %1;"
                   : "+f"(d[7]) : "f"(o.c[I & 3]));
    }
  }
}

template <int V, int U, int I, int N>
struct EmitBlock {
  static __device__ __forceinline__ void go(float* d, const CellArgs& o,
                                            unsigned* sinkp) {
    if constexpr (I < U && N > 0) {
      unit<V, I>(d, o, sinkp);
      EmitBlock<V, U, I + 1, N - 1>::go(d, o, sinkp);
    }
  }
};

template <int V, int U, int I = 0>
struct EmitUnits {
  // two-level unroll: ptxas caps template recursion depth (~200);
  // 64-unit inner blocks keep depth at U/64 + 64 (as contention_v3)
  static __device__ __forceinline__ void go(float* d, const CellArgs& o,
                                            unsigned* sinkp) {
    EmitBlock<V, U, I, 64>::go(d, o, sinkp);
    if constexpr (I + 64 < U)
      EmitUnits<V, U, I + 64>::go(d, o, sinkp);
  }
};

// MODE: 0 = self, 1 = other, 2 = ctrl, 3 = park (CID is a dummy for
// ctrl / park).  WB = warps-per-SMSP launch bound (1..3).
__device__ __forceinline__ void bar_sync1(int nthreads) {
  asm volatile("bar.sync 1, %0;" ::"r"(nthreads) : "memory");
}
__device__ __forceinline__ void bar_park() {
  asm volatile("bar.sync 2, 256;" ::: "memory");
}

// C = ops per commit group (run 4): commit_group after every C-th
// op and after the last; C = 1 is the original one-op-per-group.
template <int CID, int G, int V, int U, int MODE, int WB, int C = 1>
__global__ void __launch_bounds__(128 * WB, 1) k_ov3(
    std::uint64_t* out, std::uint32_t* dval, int reps, int wgroups,
    CellArgs args, const float* zeros, const unsigned* uzeros) {
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wg = warp >> 2;
  using T = typename WD<CID>::dty;
  T dd[WD<CID>::ND];
  for (int i = 0; i < WD<CID>::ND; ++i) dd[i] = T{};
  // every float/int constant goes through a per-lane add of a loaded
  // zero so ptxas must keep it in a GPR -- no uniform registers, no LDC
  // rematerialisation inside the timed window (contention_v3)
  const float z = zeros[threadIdx.x];
  const unsigned uz = uzeros[threadIdx.x];
  CellArgs o = args;
  o.a += z;
  o.b += z;
  for (int i = 0; i < 4; ++i) o.c[i] += z;
  for (int i = 0; i < 8; ++i) {
    o.a8[i] += z;
    o.b8[i] += z;
  }
  o.ia += uz;
  o.ib += uz;
  float d[8];
  for (int i = 0; i < 8; ++i) d[i] = 0.25f + warp + i;
  unsigned sink = 0;  // f2fp results
  const int nth = 128 * wgroups;
  const int slot = (warp * reps) * 5;
  if constexpr (MODE == 2) {
    // ctrl: one loop for all warps, two barriers then the window
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      bar_sync1(nth);
      const std::uint64_t t_s = mb_clock();
      EmitUnits<V, U>::go(d, o, &sink);
      const std::uint64_t t_w = mb_clock();
      if (lane == 0) {
        out[slot + rep * 5 + 0] = 0;
        out[slot + rep * 5 + 1] = 0;
        out[slot + rep * 5 + 2] = t_s;
        out[slot + rep * 5 + 3] = t_w;
        out[slot + rep * 5 + 4] = 0;
      }
    }
  } else if (MODE == 4 || wg == 0) {
    if constexpr (MODE == 3) {
      // park issuer role: no wgmma; park at named barrier 2, which the
      // window role joins only after its window
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        bar_sync1(nth);
        bar_sync1(nth);
        bar_park();
        if (lane == 0) {
          out[slot + rep * 5 + 0] = 0;
          out[slot + rep * 5 + 1] = 0;
          out[slot + rep * 5 + 2] = 0;
          out[slot + rep * 5 + 3] = 0;
          out[slot + rep * 5 + 4] = 0;
        }
      }
    } else {
      // issuer role: issue, window, wait in one straight-line region
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        bar_sync1(nth);
        const std::uint64_t t_i0 = mb_clock();
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        // C ops per commit group: G ops in ceil(G / C) groups stay
        // in flight (C = 1: one group per op)
#pragma unroll
        for (int g = 0; g < G; ++g) {
          WOp<CID>::op(dd, da, db);
          if constexpr (C == 1) {
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
          } else {
            if ((g + 1) % C == 0 || g + 1 == G)
              asm volatile("wgmma.commit_group.sync.aligned;"
                           ::: "memory");
          }
        }
        const std::uint64_t t_i1 = mb_clock();
        bar_sync1(nth);
        std::uint64_t t_s = 0, t_w = 0;
        if constexpr (MODE == 0) {  // self: wg0 runs the window too
          t_s = mb_clock();
          EmitUnits<V, U>::go(d, o, &sink);
          t_w = mb_clock();
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        const std::uint64_t t_d = mb_clock();
        if (lane == 0) {
          out[slot + rep * 5 + 0] = t_i0;
          out[slot + rep * 5 + 1] = t_i1;
          out[slot + rep * 5 + 2] = t_s;
          out[slot + rep * 5 + 3] = t_w;
          out[slot + rep * 5 + 4] = t_d;
        }
      }
    }
  } else {
    // window-only role (warpgroups >= 1): same two barriers per rep;
    // park joins barrier 2 after the window to release the parked role
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      bar_sync1(nth);
      const std::uint64_t t_s = mb_clock();
      EmitUnits<V, U>::go(d, o, &sink);
      const std::uint64_t t_w = mb_clock();
      if constexpr (MODE == 3) bar_park();
      if (lane == 0) {
        out[slot + rep * 5 + 0] = 0;
        out[slot + rep * 5 + 1] = 0;
        out[slot + rep * 5 + 2] = t_s;
        out[slot + rep * 5 + 3] = t_w;
        out[slot + rep * 5 + 4] = 0;
      }
    }
  }
  // fold every live register into a sink (keep accumulators + window
  // registers live across the rep loop)
  if (threadIdx.x == 0) {
    std::uint32_t s = 0;
    for (int i = 0; i < 8; ++i) s += __float_as_uint(d[i]);
    s += sink;
    for (int i = 0; i < WD<CID>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &dd[i], sizeof(w0) < sizeof(T) ? sizeof(w0) : sizeof(T));
      s += w0;
    }
    dval[0] = s;
  }
}

// record-only mufu_ts kernel (protocol-v6 addendum 1): window = 32 x
// {ex2.approx.ftz on d[i & 7]; ts[i] = clock}.  MODE 1 = other (wg0
// issues G ops and waits; warps of wg >= 1 run the window); MODE 2 =
// ctrl (no wgmma).  Lane 0 of every window warp writes the 32 offsets
// ts[i] - t_s to ts_out[(warp * reps + rep) * 32 + i].
template <int K>
__device__ __forceinline__ void ts_window(float* d, const CellArgs& o,
                                          float* sp, std::uint64_t* ts) {
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    if constexpr (K == 0) {
      asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(d[i & 7]));
    } else if constexpr (K == 1) {
      unsigned* u = reinterpret_cast<unsigned*>(d);
      asm volatile(
          "shfl.sync.bfly.b32 %0, %0, 1, 31, 0xffffffff;"
          : "+r"(u[i & 7]));
    } else if constexpr (K == 2) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                   : "+f"(d[i & 7])
                   : "f"(o.a8[i & 7]), "f"(o.b8[i & 7]));
    } else {
      d[i & 7] += sp[i & 7];  // lds_ts: LDS + FADD per unit
    }
    ts[i] = mb_clock();
  }
}

template <int CID, int G, int MODE, int WB, int K, int C = 1>
__global__ void __launch_bounds__(128 * WB, 1) k_ts(
    std::uint64_t* out, std::uint64_t* ts_out, std::uint32_t* dval,
    int reps, int wgroups, CellArgs args, const float* zeros,
    const unsigned* uzeros) {
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wg = warp >> 2;
  using T = typename WD<CID>::dty;
  T dd[WD<CID>::ND];
  for (int i = 0; i < WD<CID>::ND; ++i) dd[i] = T{};
  const float z = zeros[threadIdx.x];
  const unsigned uz = uzeros[threadIdx.x];
  CellArgs o = args;
  o.a += z;
  o.b += z;
  for (int i = 0; i < 4; ++i) o.c[i] += z;
  for (int i = 0; i < 8; ++i) {
    o.a8[i] += z;
    o.b8[i] += z;
  }
  o.ia += uz;
  o.ib += uz;
  float d[8];
  for (int i = 0; i < 8; ++i) d[i] = 0.25f + warp + i;
  // lds_ts scratch: 8 floats above the two 16 KiB wgmma tiles
  float* sp = reinterpret_cast<float*>(smem + 32768);
  if (threadIdx.x < 8) sp[threadIdx.x] = 0.0f;
  const int nth = 128 * wgroups;
  const int slot = (warp * reps) * 5;
  if constexpr (MODE == 2) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      bar_sync1(nth);
      const std::uint64_t t_s = mb_clock();
      std::uint64_t ts[32];
      ts_window<K>(d, o, sp, ts);
      const std::uint64_t t_w = mb_clock();
      if (lane == 0) {
        out[slot + rep * 5 + 0] = 0;
        out[slot + rep * 5 + 1] = 0;
        out[slot + rep * 5 + 2] = t_s;
        out[slot + rep * 5 + 3] = t_w;
        out[slot + rep * 5 + 4] = 0;
        for (int i = 0; i < 32; ++i)
          ts_out[(warp * reps + rep) * 32 + i] = ts[i] - t_s;
      }
    }
  } else if (wg == 0) {
    // other issuer role: issue, wait, no window (as k_ov3 MODE 1)
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      const std::uint64_t t_i0 = mb_clock();
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int g = 0; g < G; ++g) {
        WOp<CID>::op(dd, da, db);
        if constexpr (C == 1) {
          asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        } else {
          if ((g + 1) % C == 0 || g + 1 == G)
            asm volatile("wgmma.commit_group.sync.aligned;"
                         ::: "memory");
        }
      }
      const std::uint64_t t_i1 = mb_clock();
      bar_sync1(nth);
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      const std::uint64_t t_d = mb_clock();
      if (lane == 0) {
        out[slot + rep * 5 + 0] = t_i0;
        out[slot + rep * 5 + 1] = t_i1;
        out[slot + rep * 5 + 2] = 0;
        out[slot + rep * 5 + 3] = 0;
        out[slot + rep * 5 + 4] = t_d;
      }
    }
  } else {
    // window-only role with the timestamping window
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      bar_sync1(nth);
      const std::uint64_t t_s = mb_clock();
      std::uint64_t ts[32];
      ts_window<K>(d, o, sp, ts);
      const std::uint64_t t_w = mb_clock();
      if (lane == 0) {
        out[slot + rep * 5 + 0] = 0;
        out[slot + rep * 5 + 1] = 0;
        out[slot + rep * 5 + 2] = t_s;
        out[slot + rep * 5 + 3] = t_w;
        out[slot + rep * 5 + 4] = 0;
        for (int i = 0; i < 32; ++i)
          ts_out[(warp * reps + rep) * 32 + i] = ts[i] - t_s;
      }
    }
  }
  if (threadIdx.x == 0) {
    std::uint32_t s = 0;
    for (int i = 0; i < 8; ++i) s += __float_as_uint(d[i]);
    for (int i = 0; i < WD<CID>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &dd[i], sizeof(w0) < sizeof(T) ? sizeof(w0) : sizeof(T));
      s += w0;
    }
    dval[0] = s;
  }
}

// run 6 (protocol-v6 addendum 7): three warpgroups in a named-barrier
// ring, flash-attention-shaped sections.  Per rep: __syncthreads,
// trep, warpgroup 2 arrives once on barrier id 1 (primes warpgroup
// 0's first sync).  Per iteration it of IT = 6: barrier sync on id
// 1 + wg with count 256 (own 128 threads + 128 arrivals of the
// previous warpgroup); t0; NQ = 6 ops of form A + commit_group,
// NP = 9 ops of form B + commit_group; t1; barrier arrive on id
// 1 + (wg + 1) % 3 count 256 (skipped by warpgroup 2's last
// iteration); wgmma.wait_group 1; t2; window of U = 390 smx units
// (kind 24); t3.  After the loop wgmma.wait_group 0.
// MODE: 0 = seq3 (barriers + ops), 1 = seq3.nomma (barriers, no
// ops / no wait_group), 2 = seq1 (no barriers, with ops; launched
// with 128 threads so only warpgroup 0 exists).  Lane 0 of every
// warp writes (t0, t1, t2, t3) - trep at
// out[((warp * reps + rep) * IT + it) * 4 + j].
template <int CIDA, int CIDB, int MODE>
__global__ void __launch_bounds__(384, 1) k_seq3(
    std::uint64_t* out, std::uint32_t* dval, int reps, CellArgs args,
    const float* zeros, const unsigned* uzeros) {
  constexpr int IT = 6, NQ = 6, NP = 9, U = 390;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wg = warp >> 2;
  using TA = typename WD<CIDA>::dty;
  using TB = typename WD<CIDB>::dty;
  TA dda[WD<CIDA>::ND];
  TB ddb[WD<CIDB>::ND];
  for (int i = 0; i < WD<CIDA>::ND; ++i) dda[i] = TA{};
  for (int i = 0; i < WD<CIDB>::ND; ++i) ddb[i] = TB{};
  const float z = zeros[threadIdx.x & 255];
  const unsigned uz = uzeros[threadIdx.x & 255];
  CellArgs o = args;
  o.a += z;
  o.b += z;
  for (int i = 0; i < 4; ++i) o.c[i] += z;
  for (int i = 0; i < 8; ++i) {
    o.a8[i] += z;
    o.b8[i] += z;
  }
  o.ia += uz;
  o.ib += uz;
  float d[8];
  for (int i = 0; i < 8; ++i) d[i] = 0.25f + warp + i;
  unsigned sink = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t trep = mb_clock();
    if constexpr (MODE != 2) {
      if (wg == 2)
        asm volatile("bar.arrive 1, 256;" ::: "memory");
    }
#pragma unroll 1
    for (int it = 0; it < IT; ++it) {
      if constexpr (MODE != 2) {
        switch (wg) {
          case 0:
            asm volatile("bar.sync 1, 256;" ::: "memory");
            break;
          case 1:
            asm volatile("bar.sync 2, 256;" ::: "memory");
            break;
          default:
            asm volatile("bar.sync 3, 256;" ::: "memory");
            break;
        }
      }
      const std::uint64_t t0 = mb_clock();
      if constexpr (MODE != 1) {
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
        for (int q = 0; q < NQ; ++q) WOp<CIDA>::op(dda, da, db);
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#pragma unroll
        for (int p = 0; p < NP; ++p) WOp<CIDB>::op(ddb, da, db);
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      }
      const std::uint64_t t1 = mb_clock();
      if constexpr (MODE != 2) {
        if (!(wg == 2 && it == IT - 1)) {
          switch (1 + (wg + 1) % 3) {
            case 1:
              asm volatile("bar.arrive 1, 256;" ::: "memory");
              break;
            case 2:
              asm volatile("bar.arrive 2, 256;" ::: "memory");
              break;
            default:
              asm volatile("bar.arrive 3, 256;" ::: "memory");
              break;
          }
        }
      }
      if constexpr (MODE != 1)
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      const std::uint64_t t2 = mb_clock();
      EmitUnits<24, U>::go(d, o, &sink);
      const std::uint64_t t3 = mb_clock();
      if (lane == 0) {
        std::uint64_t* s = &out[((warp * reps + rep) * IT + it) * 4];
        s[0] = t0 - trep;
        s[1] = t1 - trep;
        s[2] = t2 - trep;
        s[3] = t3 - trep;
      }
    }
    if constexpr (MODE != 1)
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  if (threadIdx.x == 0) {
    std::uint32_t s = sink;
    for (int i = 0; i < 8; ++i) s += __float_as_uint(d[i]);
    for (int i = 0; i < WD<CIDA>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &dda[i],
                  sizeof(w0) < sizeof(TA) ? sizeof(w0) : sizeof(TA));
      s += w0;
    }
    for (int i = 0; i < WD<CIDB>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &ddb[i],
                  sizeof(w0) < sizeof(TB) ? sizeof(w0) : sizeof(TB));
      s += w0;
    }
    dval[0] = s;
  }
}

// run 7 (addendum 8): two warpgroups, each issues NG = 4 commit
// groups of NC = 3 ops into one accumulator set, then
// wgmma.wait_group 3 / 2 / 1 / 0 with a clock after each (completion
// of its 1st .. 4th group).  MODE 0 = together (both start at the rep
// barrier), 1 = staggered (warpgroup 1 waits on bar.sync 1,256 which
// warpgroup 0 arrives on right after its last issue), 2 = solo
// (warpgroup 1 idles to the rep barrier).  256 threads in all modes.
// Lane 0 of every warp writes (ti0, ti1, tc0, tc1, tc2, tc3) - trep
// at out[(warp * reps + rep) * 6 + j]; inactive warps write zeros.
template <int CID, int MODE>
__global__ void __launch_bounds__(256, 1) k_grp2(
    std::uint64_t* out, std::uint32_t* dval, int reps, CellArgs args,
    const float* zeros, const unsigned* uzeros) {
  constexpr int NG = 4, NC = 3;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wg = warp >> 2;
  using T = typename WD<CID>::dty;
  T dd[WD<CID>::ND];
  for (int i = 0; i < WD<CID>::ND; ++i) dd[i] = T{};
  const float z = zeros[threadIdx.x & 255];
  const unsigned uz = uzeros[threadIdx.x & 255];
  CellArgs o = args;
  o.a += z;
  o.b += z;
  for (int i = 0; i < 4; ++i) o.c[i] += z;
  for (int i = 0; i < 8; ++i) {
    o.a8[i] += z;
    o.b8[i] += z;
  }
  o.ia += uz;
  o.ib += uz;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t trep = mb_clock();
    const bool active = MODE != 2 || wg == 0;
    if constexpr (MODE == 1) {
      if (wg == 1)
        asm volatile("bar.sync 1, 256;" ::: "memory");
    }
    std::uint64_t t[6] = {};
    if (active) {
      t[0] = mb_clock();  // ti0
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int g = 0; g < NG; ++g) {
#pragma unroll
        for (int q = 0; q < NC; ++q) WOp<CID>::op(dd, da, db);
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      }
      t[1] = mb_clock();  // ti1
      if constexpr (MODE == 1) {
        if (wg == 0)
          asm volatile("bar.arrive 1, 256;" ::: "memory");
      }
      asm volatile("wgmma.wait_group.sync.aligned 3;" ::: "memory");
      t[2] = mb_clock();  // tc0
      asm volatile("wgmma.wait_group.sync.aligned 2;" ::: "memory");
      t[3] = mb_clock();  // tc1
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      t[4] = mb_clock();  // tc2
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      t[5] = mb_clock();  // tc3
    }
    if (lane == 0) {
      std::uint64_t* s = &out[(warp * reps + rep) * 6];
      for (int j = 0; j < 6; ++j)
        s[j] = t[j] ? t[j] - trep : 0;
    }
  }
  if (threadIdx.x == 0) {
    std::uint32_t s = __float_as_uint(o.a) + uz;
    for (int i = 0; i < WD<CID>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &dd[i],
                  sizeof(w0) < sizeof(T) ? sizeof(w0) : sizeof(T));
      s += w0;
    }
    dval[0] = s;
  }
}

template <int N, int K>
__device__ __forceinline__ void grp_wait_one(std::uint64_t* t) {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N - K)
               : "memory");
  t[2 + K] = mb_clock();
}

template <int NG, std::size_t... Ks>
__device__ __forceinline__ void grp_wait_all(std::uint64_t* t,
                                             std::index_sequence<Ks...>) {
  (grp_wait_one<NG - 1, static_cast<int>(Ks)>(t), ...);
}

// run 8 (addendum 9): k_grp2's together cell with the cell chosen at
// run time so every cell executes the same SASS.  `mask` bit g =
// warpgroup g issues; an inactive warpgroup skips to the end of the
// rep and records zeros.  `delay`: warpgroup 1 executes `delay`
// dependent adds before its ti0 clock; each step adds a word read
// from the kernel-argument pointer uzeros so the loop cannot be
// folded (values are unknown at compile time); x is stored to
// dval[1 + warp] at kernel end so the chain is live in every warp.
// No barrier besides the rep's __syncthreads.  Lane 0 of every warp
// writes (ti0, ti1, tc[0..NG-1]) - trep at
// out[(warp * reps + rep) * (2 + NG) + j]; NG <= 12.
template <int CID, int NG, int NC>
__global__ void __launch_bounds__(256, 1) k_grp2r(
    std::uint64_t* out, std::uint32_t* dval, int reps, CellArgs args,
    const float* zeros, const unsigned* uzeros, unsigned mask,
    unsigned delay) {
  constexpr int NT = 2 + NG;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wg = warp >> 2;
  using T = typename WD<CID>::dty;
  T dd[WD<CID>::ND];
  for (int i = 0; i < WD<CID>::ND; ++i) dd[i] = T{};
  const float z = zeros[threadIdx.x & 255];
  const unsigned uz = uzeros[threadIdx.x & 255];
  CellArgs o = args;
  o.a += z;
  o.b += z;
  for (int i = 0; i < 4; ++i) o.c[i] += z;
  for (int i = 0; i < 8; ++i) {
    o.a8[i] += z;
    o.b8[i] += z;
  }
  o.ia += uz;
  o.ib += uz;
  unsigned x = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t trep = mb_clock();
    const bool active = (mask >> wg) & 1u;
    std::uint64_t t[NT] = {};
    if (active) {
      if (wg == 1) {
#pragma unroll 1
        for (unsigned i = 0; i < delay; ++i) x += uzeros[i & 255];
      }
      t[0] = mb_clock();  // ti0
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int g = 0; g < NG; ++g) {
#pragma unroll
        for (int q = 0; q < NC; ++q) WOp<CID>::op(dd, da, db);
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      }
      t[1] = mb_clock();  // ti1
      grp_wait_all<NG>(t, std::make_index_sequence<NG>{});
    }
    if (lane == 0) {
      std::uint64_t* s = &out[(warp * reps + rep) * (NT + 1)];
      for (int j = 0; j < NT; ++j)
        s[j] = t[j] ? t[j] - trep : 0;
      s[NT] = trep;
    }
  }
  dval[1 + warp] = x;
  if (threadIdx.x == 0) {
    std::uint32_t s = __float_as_uint(o.a) + uz;
    for (int i = 0; i < WD<CID>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &dd[i],
                  sizeof(w0) < sizeof(T) ? sizeof(w0) : sizeof(T));
      s += w0;
    }
    dval[0] = s;
  }
}

// run 9 (addendum 13): k_grpq<CID, SHAPE>.  No `active` predicate --
// every thread executes the ops unconditionally; the number of issuing
// warpgroups is the launch's blockDim (128 / 256 / 384).  After the
// rep's __syncthreads and trep every warp spins `wg * delay` iterations
// of a dependent integer chain (x = x * 1664525 + uz, uz from the
// kernel-argument pointer; x stored at kernel end), then ti0,
// wgmma.fence, the ops with their commit_group markers, ti1, and a
// literal wgmma.wait_group sequence with a clock after each.
// Shapes: 0 = q6 (6 x [1 op + commit]; wait 5..0), 1 = g4c3
// (4 x [3 ops + commit]; wait 3..0), 2 = fa (6 ops + commit, 9 ops +
// commit; wait 1..0).  Lane 0 of every warp writes (ti0, ti1, tc[..])
// - trep at out[(warp * reps + rep) * NT + j], NT = 2 + nwait.
// out[] is sized for 12 warps x reps x 8.
#define GRPQ_BODY(DD, DA, DB, OPS, WAITS)                                \
  do {                                                                   \
    t[0] = mb_clock();                                                   \
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");              \
    OPS                                                                  \
    t[1] = mb_clock();                                                   \
    WAITS                                                                \
  } while (0)
#define GRPQ_OP(DD, DA, DB)                                              \
  do {                                                                   \
    WOp<CID>::op(DD, DA, DB);                                            \
  } while (0)
#define GRPQ_COMMIT()                                                    \
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define GRPQ_WAIT(T, K, N)                                               \
  do {                                                                   \
    asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N)            \
                 : "memory");                                            \
    T[K] = mb_clock();                                                   \
  } while (0)

template <int CID, int SHAPE>
__global__ void __launch_bounds__(384, 1) k_grpq(
    std::uint64_t* out, std::uint32_t* dval, int reps, CellArgs args,
    const float* zeros, const unsigned* uzeros, unsigned delay) {
  constexpr int NT = SHAPE == 0 ? 8 : (SHAPE == 1 ? 6 : 4);
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wg = warp >> 2;
  using T = typename WD<CID>::dty;
  T dd[WD<CID>::ND];
  for (int i = 0; i < WD<CID>::ND; ++i) dd[i] = T{};
  const float z = zeros[threadIdx.x & 255];
  const unsigned uz = uzeros[threadIdx.x & 255];
  CellArgs o = args;
  o.a += z;
  o.b += z;
  for (int i = 0; i < 4; ++i) o.c[i] += z;
  for (int i = 0; i < 8; ++i) {
    o.a8[i] += z;
    o.b8[i] += z;
  }
  o.ia += uz;
  o.ib += uz;
  unsigned x = uz;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t trep = mb_clock();
#pragma unroll 1
    for (unsigned i = 0, n = (unsigned)wg * delay; i < n; ++i)
      x = x * 1664525u + uz;
    std::uint64_t t[NT] = {};
    if constexpr (SHAPE == 0) {
      GRPQ_BODY(dd, da, db,
                { GRPQ_OP(dd, da, db); GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_COMMIT(); },
                { GRPQ_WAIT(t, 2, 5); GRPQ_WAIT(t, 3, 4);
                   GRPQ_WAIT(t, 4, 3); GRPQ_WAIT(t, 5, 2);
                   GRPQ_WAIT(t, 6, 1); GRPQ_WAIT(t, 7, 0); });
    } else if constexpr (SHAPE == 1) {
      GRPQ_BODY(dd, da, db,
                { GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_COMMIT(); },
                { GRPQ_WAIT(t, 2, 3); GRPQ_WAIT(t, 3, 2);
                   GRPQ_WAIT(t, 4, 1); GRPQ_WAIT(t, 5, 0); });
    } else {
      GRPQ_BODY(dd, da, db,
                { GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_COMMIT(); }
                 { GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_OP(dd, da, db);
                   GRPQ_OP(dd, da, db); GRPQ_COMMIT(); },
                { GRPQ_WAIT(t, 2, 1); GRPQ_WAIT(t, 3, 0); });
    }
    if (lane == 0) {
      std::uint64_t* s = &out[(warp * reps + rep) * (NT + 1)];
      for (int j = 0; j < NT; ++j)
        s[j] = t[j] ? t[j] - trep : 0;
      s[NT] = trep;
    }
  }
  dval[1 + warp] = x;
  if (threadIdx.x == 0) {
    std::uint32_t s = __float_as_uint(o.a) + uz;
    for (int i = 0; i < WD<CID>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &dd[i],
                  sizeof(w0) < sizeof(T) ? sizeof(w0) : sizeof(T));
      s += w0;
    }
    dval[0] = s;
  }
}

using O3Kern = void (*)(std::uint64_t*, std::uint32_t*, int, int, CellArgs,
                        const float*, const unsigned*);
using TsKern = void (*)(std::uint64_t*, std::uint64_t*, std::uint32_t*,
                        int, int, CellArgs, const float*, const unsigned*);
using SeqKern = void (*)(std::uint64_t*, std::uint32_t*, int, CellArgs,
                         const float*, const unsigned*);

static const struct { int v; const char* name; } KINDS[] = {
    {23, "mufu8"}, {28, "mufu2"}, {29, "mufu4"},
};

static const struct { int cid; const char* name; } FORMS[] = {
    {142, "hgmma_bf16_n64_ss"},
    {158, "hgmma_bf16_n128_ss"},
    {190, "hgmma_bf16_n256_ss"},
    {191, "hgmma_bf16_n256_rs"},
};

struct O3Cell {
  int fi; int g; int u; int ki; int mode; int wmax; int c;
  O3Kern kern[3];  // kern[W - 1]
};

#define KM(CID, G, V, U, M)                                          \
  {&k_ov3<CID, G, V, U, M, 1>, &k_ov3<CID, G, V, U, M, 2>,           \
   &k_ov3<CID, G, V, U, M, 3>}
#define KM_C(CID, G, V, U, M, C)                                     \
  {&k_ov3<CID, G, V, U, M, 1, C>, &k_ov3<CID, G, V, U, M, 2, C>,     \
   &k_ov3<CID, G, V, U, M, 3, C>}
// other cells run W = 2..3 (wg_lo = 2); self cells W = 1..2
#define O3(FI, CID, G, U, KI, V)                                     \
  {FI, G, U, KI, 1, 3, 0, KM(CID, G, V, U, 1)},
#define S2(FI, CID, G, U, KI, V)                                     \
  {FI, G, U, KI, 0, 2, 0, KM(CID, G, V, U, 0)},

// self w1 issue sweep (a): mufu8 U64; revisit other.w2 (d): n256 ss
// G = 8 U64 mufu2 / mufu4 / mufu8
#define S1(FI, CID, G)                                               \
  {FI, G, 64, 0, 0, 1, 0, KM(CID, G, 23, 64, 0)},
static const O3Cell O3_CELLS[] = {
    S1(1, 158, 4) S1(1, 158, 6) S1(1, 158, 8)
    S1(1, 158, 10) S1(1, 158, 12) S1(1, 158, 24)
    S1(0, 142, 12) S1(0, 142, 24)
    S1(2, 190, 12) S1(2, 190, 24)
    {2, 8, 64, 1, 1, 2, 0, KM(190, 8, 28, 64, 1)},
    {2, 8, 64, 2, 1, 2, 0, KM(190, 8, 29, 64, 1)},
    {2, 8, 64, 0, 1, 2, 0, KM(190, 8, 23, 64, 1)},
    // run 5 (addendum 5): n128 ss G = 24 mufu8 U64, other.w2 and
    // self w1/w2, C = 1 controls and C = 6; c = 0 keeps the legacy
    // name (no .cN segment)
    {1, 24, 64, 0, 1, 2, 1, KM_C(158, 24, 23, 64, 1, 1)},
    {1, 24, 64, 0, 1, 2, 6, KM_C(158, 24, 23, 64, 1, 6)},
    {1, 24, 64, 0, 0, 2, 1, KM_C(158, 24, 23, 64, 0, 1)},
    {1, 24, 64, 0, 0, 2, 6, KM_C(158, 24, 23, 64, 0, 6)},
};
static const int O3_CELLS_N = sizeof(O3_CELLS) / sizeof(O3Cell);


// ctrl cells: one kernel per (kind, U, W), CID / G are dummies
struct CCell { int ki; int u; int wmax; O3Kern kern[3]; };
#define CR(KI, V, U, WM)                                            \
  {KI, U, WM, {&k_ov3<190, 0, V, U, 2, 1>, &k_ov3<190, 0, V, U, 2, 2>, \
           &k_ov3<190, 0, V, U, 2, 3>}},
static const CCell C_CELLS[] = {
    CR(0, 23, 64, 2) CR(1, 28, 64, 1) CR(2, 29, 64, 1)
};
static const int C_CELLS_N = sizeof(C_CELLS) / sizeof(CCell);

static void emit_row(MbWriter& w, const std::string& variant, int u,
                     const std::string& params, const std::uint64_t* h,
                     int reps, int wgs, int launch,
                     const std::string& extra2 = "", int owner_wp = 0) {
  std::vector<double> samples, tails, spans, issue, i0_rel, i1_rel;
  for (int r = 1; r < reps; ++r) {
    std::uint64_t tw_max = 0, ts_min = ~0ull;
    std::uint64_t i0_min = ~0ull, i1_max = 0;
    for (int wp = 0; wp < 4 * wgs; ++wp) {
      const std::uint64_t ts = h[(wp * reps + r) * 5 + 2];
      const std::uint64_t tw = h[(wp * reps + r) * 5 + 3];
      if (ts != 0 && tw != 0) {
        samples.push_back(static_cast<double>(tw - ts));
        tw_max = std::max(tw_max, tw);
        ts_min = std::min(ts_min, ts);
      }
      const std::uint64_t i0 = h[(wp * reps + r) * 5 + 0];
      const std::uint64_t i1 = h[(wp * reps + r) * 5 + 1];
      if (i0 != 0 && i1 != 0) {
        issue.push_back(static_cast<double>(i1 - i0));
        if (wp < owner_wp) {
          i0_min = std::min(i0_min, i0);
          i1_max = std::max(i1_max, i1);
        }
      }
    }
    for (int wp = 0; wp < 4 && wp < 4 * wgs; ++wp) {
      const std::uint64_t td = h[(wp * reps + r) * 5 + 4];
      if (td != 0 && tw_max != 0) {
        tails.push_back(static_cast<double>(
            static_cast<std::int64_t>(td - tw_max)));
        spans.push_back(static_cast<double>(td - ts_min));
      }
    }
    if (owner_wp > 0 && i0_min != ~0ull && ts_min != ~0ull) {
      i0_rel.push_back(static_cast<double>(
          static_cast<std::int64_t>(i0_min - ts_min)));
      i1_rel.push_back(static_cast<double>(
          static_cast<std::int64_t>(i1_max - ts_min)));
    }
  }
  auto jvec = [](const std::vector<double>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
      char b[32];
      std::snprintf(b, sizeof(b), i ? ",%.1f" : "%.1f", v[i]);
      s += b;
    }
    return s + "]";
  };
  const std::string extra = "\"tails\":" + jvec(tails) +
                          ",\"spans\":" + jvec(spans) +
                          ",\"issue\":" + jvec(issue) +
                          (owner_wp > 0
                               ? ",\"i0_rel\":" + jvec(i0_rel) +
                                     ",\"i1_rel\":" + jvec(i1_rel)
                               : "") +
                          extra2;
  w.row(variant, u, params, samples, extra);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 12 * args.reps * 5));
  MB_CHECK(cudaMalloc(&d_val, sizeof(std::uint32_t) * 16));
  std::uint64_t* d_ts = nullptr;
  MB_CHECK(cudaMalloc(&d_ts, sizeof(std::uint64_t) * 12 * args.reps * 32));
  std::uint64_t* d_seq = nullptr;
  MB_CHECK(cudaMalloc(&d_seq,
                     sizeof(std::uint64_t) * 12 * args.reps * 6 * 4));
  float* d_zeros = nullptr;
  unsigned* d_uzeros = nullptr;
  MB_CHECK(cudaMalloc(&d_zeros, sizeof(float) * 256));
  MB_CHECK(cudaMalloc(&d_uzeros, sizeof(unsigned) * 256));
  MB_CHECK(cudaMemset(d_zeros, 0, sizeof(float) * 256));
  MB_CHECK(cudaMemset(d_uzeros, 0, sizeof(unsigned) * 256));
  CellArgs a{};
  a.a = 1.0000001f;
  a.b = 0.5f;
  a.c[0] = 0.75f; a.c[1] = 1.5f; a.c[2] = 2.25f; a.c[3] = 3.0f;
  for (int i = 0; i < 8; ++i) {
    a.a8[i] = 1.0000001f + 0.0078125f * i;
    a.b8[i] = 0.5f + 0.03125f * i;
  }
  a.ia = 3u;
  a.ib = 5u;
  a.p = 1.0000002f;  // uniform multiplier (never mixed with z)
  std::vector<std::uint64_t> h;
  auto params = [](const std::string& form, int cid, const char* kind,
                   int u, const char* mode, int wg, int g, int l) {
    return "\"form\":\"" + form + "\",\"cid\":" + std::to_string(cid) +
           ",\"kind\":\"" + kind + "\",\"u\":" + std::to_string(u) +
           ",\"mode\":\"" + mode + "\",\"w\":" + std::to_string(wg) +
           ",\"g\":" + std::to_string(g) + ",\"launch\":" +
           std::to_string(l);
  };
  auto emit = [&](const std::string& variant, int u, const char* kind,
                  O3Kern kern, const std::string& form, int cid,
                  const char* mode, int wg, int g) {
    if (!mb_selected(args, variant)) return;
    MB_CHECK(cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    for (int l = 0; l < args.launches; ++l) {
      kern<<<1, 128 * wg, kSmemBytes>>>(d_out, d_val, args.reps, wg, a,
                                        d_zeros, d_uzeros);
      const cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      MB_CHECK(cudaDeviceSynchronize());
      h.assign(4 * wg * args.reps * 5, 0);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      emit_row(w, variant, u,
               params(form, cid, kind, u, mode, wg, g, l), h.data(),
               args.reps, wg, l, "", std::strcmp(mode, "other") == 0 ? 4 : 0);
    }
  };
  for (int ci = 0; ci < O3_CELLS_N; ++ci) {
    const O3Cell& c = O3_CELLS[ci];
    const int wg_lo = c.mode == 1 ? 2 : 1;
    for (int wg = wg_lo; wg <= c.wmax; ++wg) {
      const std::string variant =
          std::string(FORMS[c.fi].name) + ".g" + std::to_string(c.g) +
          (c.c ? ".c" + std::to_string(c.c) : "") +
          "." + KINDS[c.ki].name + ".u" + std::to_string(c.u) + "." +
          (c.mode == 0 ? "self" : "other") + ".w" + std::to_string(wg);
      emit(variant, c.u, KINDS[c.ki].name, c.kern[wg - 1],
           FORMS[c.fi].name, FORMS[c.fi].cid,
           c.mode == 0 ? "self" : "other", wg, c.g);
    }
  }
  for (int ci = 0; ci < C_CELLS_N; ++ci) {
    const CCell& c = C_CELLS[ci];
    for (int wg = 1; wg <= c.wmax; ++wg) {
      const std::string variant = "ctrl." + std::string(KINDS[c.ki].name) +
                                  ".u" + std::to_string(c.u) + ".w" +
                                  std::to_string(wg);
      emit(variant, c.u, KINDS[c.ki].name, c.kern[wg - 1], "ctrl", -1,
           "ctrl", wg, 0);
    }
  }
  // per-unit clock cells (c): other.w2 n128 ss G = 24 + ctrl.w1
  const struct {
    const char* variant; const char* kind; int fi; int g;
    const char* mode; int wg; TsKern kern;
  } TS_CELLS[] = {
    {"hgmma_bf16_n128_ss.g24.mufu_ts.u32.other.w2", "mufu_ts", 1, 24,
     "other", 2, &k_ts<158, 24, 1, 2, 0>},
    {"hgmma_bf16_n128_ss.g24.shfl_ts.u32.other.w2", "shfl_ts", 1, 24,
     "other", 2, &k_ts<158, 24, 1, 2, 1>},
    {"hgmma_bf16_n128_ss.g24.ffma_ts.u32.other.w2", "ffma_ts", 1, 24,
     "other", 2, &k_ts<158, 24, 1, 2, 2>},
    {"hgmma_bf16_n128_ss.g24.lds_ts.u32.other.w2", "lds_ts", 1, 24,
     "other", 2, &k_ts<158, 24, 1, 2, 3>},
    // run 5 (addendum 5): same ts cells with C = 6 ops per group
    {"hgmma_bf16_n128_ss.g24.c6.mufu_ts.u32.other.w2", "mufu_ts", 1, 24,
     "other", 2, &k_ts<158, 24, 1, 2, 0, 6>},
    {"hgmma_bf16_n128_ss.g24.c6.shfl_ts.u32.other.w2", "shfl_ts", 1, 24,
     "other", 2, &k_ts<158, 24, 1, 2, 1, 6>},
    {"hgmma_bf16_n128_ss.g24.c6.ffma_ts.u32.other.w2", "ffma_ts", 1, 24,
     "other", 2, &k_ts<158, 24, 1, 2, 2, 6>},
    {"ctrl.mufu_ts.u32.w1", "mufu_ts", -1, 0, "ctrl", 1,
     &k_ts<190, 0, 2, 1, 0>},
    {"ctrl.shfl_ts.u32.w1", "shfl_ts", -1, 0, "ctrl", 1,
     &k_ts<190, 0, 2, 1, 1>},
    {"ctrl.ffma_ts.u32.w1", "ffma_ts", -1, 0, "ctrl", 1,
     &k_ts<190, 0, 2, 1, 2>},
    {"ctrl.lds_ts.u32.w1", "lds_ts", -1, 0, "ctrl", 1,
     &k_ts<190, 0, 2, 1, 3>},
  };
  std::vector<std::uint64_t> ts_h;
  for (const auto& c : TS_CELLS) {
    const std::string variant = c.variant;
    if (!mb_selected(args, variant)) continue;
    MB_CHECK(cudaFuncSetAttribute(
        c.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    const bool is_ctrl = c.fi < 0;
    for (int l = 0; l < args.launches; ++l) {
      c.kern<<<1, 128 * c.wg, kSmemBytes>>>(d_out, d_ts, d_val, args.reps,
                                            c.wg, a, d_zeros, d_uzeros);
      const cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      MB_CHECK(cudaDeviceSynchronize());
      h.assign(4 * c.wg * args.reps * 5, 0);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      ts_h.assign(4 * c.wg * args.reps * 32, 0);
      MB_CHECK(cudaMemcpy(ts_h.data(), d_ts, ts_h.size() * 8,
                          cudaMemcpyDeviceToHost));
      std::string tsv = ",\"ts\":[";
      const int rmax = std::min(args.reps - 1, 8);
      const int wp0 = is_ctrl ? 0 : 4;
      for (int r = 1; r <= rmax; ++r) {
        for (int wp = wp0; wp < 4 * c.wg; ++wp) {
          for (int i = 0; i < 32; ++i) {
            char b[24];
            std::snprintf(b, sizeof(b), "%s%llu",
                          (r == 1 && wp == wp0 && i == 0) ? "" : ",",
                          static_cast<unsigned long long>(
                              ts_h[(wp * args.reps + r) * 32 + i]));
            tsv += b;
          }
        }
      }
      tsv += "]";
      emit_row(w, variant, 32,
               params(is_ctrl ? "ctrl" : FORMS[c.fi].name,
                      is_ctrl ? -1 : FORMS[c.fi].cid, c.kind, 32,
                      c.mode, c.wg, c.g, l),
               h.data(), args.reps, c.wg, l, tsv, is_ctrl ? 0 : 4);
    }
  }
  // both cells (b): MODE 4, both warpgroups issue and wait, no window
  const struct { const char* variant; int fi; int g; O3Kern kern; }
  BOTH_CELLS[] = {
    {"hgmma_bf16_n128_ss.g16.both.w2", 1, 16,
     &k_ov3<158, 16, 23, 64, 4, 2>},
    {"hgmma_bf16_n256_ss.g16.both.w2", 2, 16,
     &k_ov3<190, 16, 23, 64, 4, 2>},
  };
  auto jarr = [](const std::vector<double>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
      char b[32];
      std::snprintf(b, sizeof(b), i ? ",%.1f" : "%.1f", v[i]);
      s += b;
    }
    return s + "]";
  };
  for (const auto& c : BOTH_CELLS) {
    const std::string variant = c.variant;
    if (!mb_selected(args, variant)) continue;
    MB_CHECK(cudaFuncSetAttribute(
        c.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    for (int l = 0; l < args.launches; ++l) {
      c.kern<<<1, 256, kSmemBytes>>>(d_out, d_val, args.reps, 2, a,
                                     d_zeros, d_uzeros);
      const cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      MB_CHECK(cudaDeviceSynchronize());
      h.assign(8 * args.reps * 5, 0);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      std::vector<double> issue0, issue1, done0, done1, start1;
      for (int r = 1; r < args.reps; ++r) {
        std::uint64_t t0a = ~0ull, t0b = ~0ull, t0all = ~0ull;
        std::vector<std::uint64_t> i0, i1, td;
        for (int wp = 0; wp < 8; ++wp) {
          const std::uint64_t* sl = &h[(wp * args.reps + r) * 5];
          i0.push_back(sl[0]); i1.push_back(sl[1]); td.push_back(sl[4]);
          t0all = std::min(t0all, sl[0]);
          if (wp < 4) t0a = std::min(t0a, sl[0]);
          else t0b = std::min(t0b, sl[0]);
        }
        auto med = [](std::vector<double> v) {
          std::sort(v.begin(), v.end());
          return v[v.size() / 2];
        };
        std::vector<double> d0, d1, f0, f1;
        for (int wp = 0; wp < 8; ++wp) {
          (wp < 4 ? d0 : d1).push_back(
              static_cast<double>(i1[wp] - i0[wp]));
          (wp < 4 ? f0 : f1).push_back(
              static_cast<double>(td[wp] - t0all));
        }
        issue0.push_back(med(d0));
        issue1.push_back(med(d1));
        done0.push_back(med(f0));
        done1.push_back(med(f1));
        start1.push_back(static_cast<double>(
            static_cast<std::int64_t>(t0b - t0a)));
      }
      const std::string extra =
          "\"issue0\":" + jarr(issue0) + ",\"issue1\":" + jarr(issue1) +
          ",\"done0\":" + jarr(done0) + ",\"done1\":" + jarr(done1) +
          ",\"start1\":" + jarr(start1);
      w.row(variant, 0,
            params(FORMS[c.fi].name, FORMS[c.fi].cid, "both", 0, "both",
                   2, c.g, l),
            {}, extra);
    }
  }
  // seq3 cells (run 6, addendum 7): three-warpgroup ring, two forms
  const struct {
    const char* variant; const char* mode; int nthreads; SeqKern kern;
  } SEQ_CELLS[] = {
    {"seq3.u390", "seq3", 384, &k_seq3<162, 150, 0>},
    {"seq3.nomma.u390", "seq3.nomma", 384, &k_seq3<162, 150, 1>},
    {"seq1.u390", "seq1", 128, &k_seq3<162, 150, 2>},
  };
  for (const auto& c : SEQ_CELLS) {
    const std::string variant = c.variant;
    if (!mb_selected(args, variant)) continue;
    MB_CHECK(cudaFuncSetAttribute(
        c.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    const int nwarps = c.nthreads / 32;
    for (int l = 0; l < args.launches; ++l) {
      c.kern<<<1, c.nthreads, kSmemBytes>>>(d_seq, d_val, args.reps, a,
                                          d_zeros, d_uzeros);
      const cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      MB_CHECK(cudaDeviceSynchronize());
      h.assign(nwarps * args.reps * 6 * 4, 0);
      MB_CHECK(cudaMemcpy(h.data(), d_seq, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      // t[rep >= 1][warp][it][4]
      std::string tv = "\"t\":[";
      for (int r = 1; r < args.reps; ++r) {
        tv += "[";
        for (int wp = 0; wp < nwarps; ++wp) {
          tv += "[";
          for (int it = 0; it < 6; ++it) {
            char b[96];
            const std::uint64_t* s =
                &h[((wp * args.reps + r) * 6 + it) * 4];
            std::snprintf(b, sizeof(b), "%s[%llu,%llu,%llu,%llu]",
                          it ? "," : "",
                          static_cast<unsigned long long>(s[0]),
                          static_cast<unsigned long long>(s[1]),
                          static_cast<unsigned long long>(s[2]),
                          static_cast<unsigned long long>(s[3]));
            tv += b;
          }
          tv += "]";
          if (wp + 1 < nwarps) tv += ",";
        }
        tv += "]";
        if (r + 1 < args.reps) tv += ",";
      }
      tv += "]";
      const std::string p =
          "\"form\":\"hgmma_bf16_n144_ss+hgmma_bf16_n96_ss\""
          ",\"cid_a\":162,\"cid_b\":150,\"kind\":\"smx\",\"u\":390"
          ",\"mode\":\"" + std::string(c.mode) +
          "\",\"w\":" + std::to_string(nwarps / 4) +
          ",\"it\":6,\"nq\":6,\"np\":9,\"launch\":" +
          std::to_string(l);
      w.row(variant, 390, p, {}, tv);
    }
  }
  // grp2 cells (run 7): two warpgroups, 4 commit groups of 3 ops;
  // t[rep >= 1][warp][6] = (ti0, ti1, tc0, tc1, tc2, tc3) - trep
  const struct {
    const char* variant; const char* mode; SeqKern kern;
  } GRP2_CELLS[] = {
    {"grp2.together", "together", &k_grp2<158, 0>},
    {"grp2.staggered", "staggered", &k_grp2<158, 1>},
    {"grp2.solo", "solo", &k_grp2<158, 2>},
  };
  for (const auto& c : GRP2_CELLS) {
    const std::string variant = c.variant;
    if (!mb_selected(args, variant)) continue;
    MB_CHECK(cudaFuncSetAttribute(
        c.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    for (int l = 0; l < args.launches; ++l) {
      c.kern<<<1, 256, kSmemBytes>>>(d_seq, d_val, args.reps, a, d_zeros,
                                     d_uzeros);
      const cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      MB_CHECK(cudaDeviceSynchronize());
      h.assign(8 * args.reps * 6, 0);
      MB_CHECK(cudaMemcpy(h.data(), d_seq, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      std::string tv = "\"t\":[";
      for (int r = 1; r < args.reps; ++r) {
        tv += "[";
        for (int wp = 0; wp < 8; ++wp) {
          char b[160];
          const std::uint64_t* s = &h[(wp * args.reps + r) * 6];
          std::snprintf(
              b, sizeof(b), "%s[%llu,%llu,%llu,%llu,%llu,%llu]",
              wp ? "," : "",
              static_cast<unsigned long long>(s[0]),
              static_cast<unsigned long long>(s[1]),
              static_cast<unsigned long long>(s[2]),
              static_cast<unsigned long long>(s[3]),
              static_cast<unsigned long long>(s[4]),
              static_cast<unsigned long long>(s[5]));
          tv += b;
        }
        tv += "]";
        if (r + 1 < args.reps) tv += ",";
      }
      tv += "]";
      const std::string p =
          "\"form\":\"hgmma_bf16_n128_ss\",\"cid\":158"
          ",\"mode\":\"" + std::string(c.mode) +
          "\",\"w\":2,\"ng\":4,\"nc\":3,\"launch\":" +
          std::to_string(l);
      w.row(variant, 0, p, {}, tv);
    }
  }
  // grp2r cells (run 8): same compiled kernel for all cells of one
  // (NG, NC); mask / delay are kernel arguments.
  //   t[rep >= 1][warp][2 + NG] = (ti0, ti1, tc[0..NG-1]) - trep
  using GrpRKern = void (*)(std::uint64_t*, std::uint32_t*, int, CellArgs,
                            const float*, const unsigned*, unsigned,
                            unsigned);
  const struct {
    const char* shape; int ng; int nc; GrpRKern kern;
  } GRP2R_SHAPES[] = {
    {"g4c3", 4, 3, &k_grp2r<158, 4, 3>},
    {"g1c12", 1, 12, &k_grp2r<158, 1, 12>},
    {"g12c1", 12, 1, &k_grp2r<158, 12, 1>},
  };
  const struct { const char* cell; unsigned mask; unsigned delay; }
  GRP2R_MODES[] = {
    {"solo", 1, 0},
    {"together", 3, 0},
    {"d150", 3, 150},
    {"d600", 3, 600},
  };
  for (const auto& s : GRP2R_SHAPES) {
    for (const auto& m : GRP2R_MODES) {
      const std::string variant =
          "grp2r." + std::string(s.shape) + "." + m.cell;
      if (!mb_selected(args, variant)) continue;
      MB_CHECK(cudaFuncSetAttribute(
          s.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      const int nt = 2 + s.ng;
      for (int l = 0; l < args.launches; ++l) {
        s.kern<<<1, 256, kSmemBytes>>>(d_seq, d_val, args.reps, a,
                                       d_zeros, d_uzeros, m.mask,
                                       m.delay);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        h.assign(8 * args.reps * nt, 0);
        MB_CHECK(cudaMemcpy(h.data(), d_seq, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        std::string tv = "\"t\":[";
        for (int r = 1; r < args.reps; ++r) {
          tv += "[";
          for (int wp = 0; wp < 8; ++wp) {
            tv += "[";
            for (int j = 0; j < nt; ++j) {
              char b[24];
              std::snprintf(b, sizeof(b), "%s%llu", j ? "," : "",
                            static_cast<unsigned long long>(
                                h[(wp * args.reps + r) * nt + j]));
              tv += b;
            }
            tv += "]";
            if (wp + 1 < 8) tv += ",";
          }
          tv += "]";
          if (r + 1 < args.reps) tv += ",";
        }
        tv += "]";
        const std::string p =
            "\"form\":\"hgmma_bf16_n128_ss\",\"cid\":158"
            ",\"cell\":\"" + std::string(m.cell) +
            "\",\"mask\":" + std::to_string(m.mask) +
            ",\"delay\":" + std::to_string(m.delay) +
            ",\"w\":2,\"ng\":" + std::to_string(s.ng) +
            ",\"nc\":" + std::to_string(s.nc) +
            ",\"launch\":" + std::to_string(l);
        w.row(variant, 0, p, {}, tv);
      }
    }
  }
  // grpq cells (run 9, addendum 13): no predicate; the block size is the
  // number of issuing warpgroups; delay = integer-chain iterations per
  // warpgroup index.  t[rep >= 1][warp][nt] = (ti0, ti1, tc[..]) - trep
  using GrpQKern = void (*)(std::uint64_t*, std::uint32_t*, int, CellArgs,
                            const float*, const unsigned*, unsigned);
  const struct {
    const char* shape; int nt; int ng; int nc; GrpQKern kern;
  } GRPQ_SHAPES[] = {
    {"q6", 8, 6, 1, &k_grpq<158, 0>},
    {"g4c3", 6, 4, 3, &k_grpq<158, 1>},
    {"fa", 4, 2, 0, &k_grpq<158, 2>},
  };
  const struct {
    const char* cell; int wgs; unsigned delay;
  } GRPQ_MODES[] = {
    {"w1", 1, 0},
    {"w2.d0", 2, 0},
    {"w2.dA", 2, 7},
    {"w2.dB", 2, 17},
    {"w3.d0", 3, 0},
    {"w3.dA", 3, 7},
  };
  for (const auto& s : GRPQ_SHAPES) {
    for (const auto& m : GRPQ_MODES) {
      const std::string variant =
          "grpq." + std::string(s.shape) + "." + m.cell;
      if (!mb_selected(args, variant)) continue;
      MB_CHECK(cudaFuncSetAttribute(
          s.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      for (int l = 0; l < args.launches; ++l) {
        s.kern<<<1, 128 * m.wgs, kSmemBytes>>>(d_seq, d_val, args.reps, a,
                                             d_zeros, d_uzeros, m.delay);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        h.assign(4 * m.wgs * args.reps * (s.nt + 1), 0);
        MB_CHECK(cudaMemcpy(h.data(), d_seq, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        std::string tv = "\"t\":[";
        std::string tr = ",\"trep\":[";
        for (int r = 1; r < args.reps; ++r) {
          tv += "[";
          tr += "[";
          for (int wp = 0; wp < 4 * m.wgs; ++wp) {
            tv += "[";
            char tb[24];
            std::snprintf(tb, sizeof(tb), "%s%llu", wp ? "," : "",
                          static_cast<unsigned long long>(
                              h[(wp * args.reps + r) * (s.nt + 1) + s.nt]));
            tr += tb;
            for (int j = 0; j < s.nt; ++j) {
              char b[24];
              std::snprintf(b, sizeof(b), "%s%llu", j ? "," : "",
                            static_cast<unsigned long long>(
                                h[(wp * args.reps + r) * (s.nt + 1) + j]));
              tv += b;
            }
            tv += "]";
            if (wp + 1 < 4 * m.wgs) tv += ",";
          }
          tv += "]";
          tr += "]";
          if (r + 1 < args.reps) { tv += ","; tr += ","; }
        }
        tv += "]";
        tr += "]";
        tv += tr;
        const std::string p =
            "\"form\":\"hgmma_bf16_n128_ss\",\"cid\":158"
            ",\"cell\":\"" + std::string(m.cell) +
            "\",\"delay\":" + std::to_string(m.delay) +
            ",\"w\":" + std::to_string(m.wgs) +
            ",\"ng\":" + std::to_string(s.ng) +
            ",\"nt\":" + std::to_string(s.nt) +
            ",\"launch\":" + std::to_string(l);
        w.row(variant, 0, p, {}, tv);
      }
    }
  }
  return 0;
}
