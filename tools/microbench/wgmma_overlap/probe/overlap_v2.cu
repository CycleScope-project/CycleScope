// tools/microbench/wgmma_overlap/probe/overlap_v2.cu — H100 (sm_90a).
//
// protocol-v2.json: C1 = issue floor F of the issuing warpgroup's
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
};

template <int V, int I>
__device__ __forceinline__ void unit(float* d, const CellArgs& o) {
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
  }
}

template <int V, int U, int I, int N>
struct EmitBlock {
  static __device__ __forceinline__ void go(float* d, const CellArgs& o) {
    if constexpr (I < U && N > 0) {
      unit<V, I>(d, o);
      EmitBlock<V, U, I + 1, N - 1>::go(d, o);
    }
  }
};

template <int V, int U, int I = 0>
struct EmitUnits {
  // two-level unroll: ptxas caps template recursion depth (~200);
  // 64-unit inner blocks keep depth at U/64 + 64 (as contention_v3)
  static __device__ __forceinline__ void go(float* d, const CellArgs& o) {
    EmitBlock<V, U, I, 64>::go(d, o);
    if constexpr (I + 64 < U)
      EmitUnits<V, U, I + 64>::go(d, o);
  }
};

// MODE: 0 = self, 1 = other, 2 = ctrl, 3 = park (CID is a dummy for
// ctrl / park).  WB = warps-per-SMSP launch bound (1..2).
__device__ __forceinline__ void bar_sync1(int nthreads) {
  asm volatile("bar.sync 1, %0;" ::"r"(nthreads) : "memory");
}
__device__ __forceinline__ void bar_park() {
  asm volatile("bar.sync 2, 256;" ::: "memory");
}

template <int CID, int G, int V, int U, int MODE, int WB>
__global__ void __launch_bounds__(128 * WB, 1) k_ov2(
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
  const int nth = 128 * wgroups;
  const int slot = (warp * reps) * 5;
  if constexpr (MODE == 2) {
    // ctrl: one loop for all warps, two barriers then the window
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      bar_sync1(nth);
      const std::uint64_t t_s = mb_clock();
      EmitUnits<V, U>::go(d, o);
      const std::uint64_t t_w = mb_clock();
      if (lane == 0) {
        out[slot + rep * 5 + 0] = 0;
        out[slot + rep * 5 + 1] = 0;
        out[slot + rep * 5 + 2] = t_s;
        out[slot + rep * 5 + 3] = t_w;
        out[slot + rep * 5 + 4] = 0;
      }
    }
  } else if (wg == 0) {
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
        // commit_group per op: G groups of one op stay in flight
#pragma unroll
        for (int g = 0; g < G; ++g) {
          WOp<CID>::op(dd, da, db);
          asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        }
        const std::uint64_t t_i1 = mb_clock();
        bar_sync1(nth);
        std::uint64_t t_s = 0, t_w = 0;
        if constexpr (MODE == 0) {  // self: wg0 runs the window too
          t_s = mb_clock();
          EmitUnits<V, U>::go(d, o);
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
      EmitUnits<V, U>::go(d, o);
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
    for (int i = 0; i < WD<CID>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &dd[i], sizeof(w0) < sizeof(T) ? sizeof(w0) : sizeof(T));
      s += w0;
    }
    dval[0] = s;
  }
}

using O2Kern = void (*)(std::uint64_t*, std::uint32_t*, int, int, CellArgs,
                        const float*, const unsigned*);

static const struct { int v; const char* name; } KINDS[] = {
    {4, "ilp8"},  {5, "ilp8n"}, {15, "fadd1"}, {16, "fmul2"},
    {0, "dep3r"}, {2, "dep2"},  {9, "fmnmx"},  {8, "imad"},
};

static const struct { int cid; const char* name; } FORMS[] = {
    {190, "hgmma_bf16_n256_ss"},
    {158, "hgmma_bf16_n128_ss"},
    {142, "hgmma_bf16_n64_ss"},
    {191, "hgmma_bf16_n256_rs"},
    {318, "qgmma_e4m3_n256_ss"},
};

struct O2Cell {
  int fi; int g; int u; int ki; int mode; O2Kern kern[2];  // kern[W - 1]
};

#define KM(CID, G, V, U, M)                                          \
  {&k_ov2<CID, G, V, U, M, 1>, &k_ov2<CID, G, V, U, M, 2>}
#define R2(FI, CID, G, U, KI, V)                                     \
  {FI, G, U, KI, 0, KM(CID, G, V, U, 0)},                            \
      {FI, G, U, KI, 1, KM(CID, G, V, U, 1)},

#define FOR_K(FI, CID, G, U)                                         \
  R2(FI, CID, G, U, 0, 4)                                            \
  R2(FI, CID, G, U, 1, 5)                                            \
  R2(FI, CID, G, U, 2, 15)                                           \
  R2(FI, CID, G, U, 3, 16)                                           \
  R2(FI, CID, G, U, 4, 0)                                            \
  R2(FI, CID, G, U, 5, 2)                                            \
  R2(FI, CID, G, U, 6, 9)                                            \
  R2(FI, CID, G, U, 7, 8)

static const O2Cell O2_CELLS[] = {
    FOR_K(0, 190, 2, 32)
    FOR_K(0, 190, 4, 64)
    FOR_K(0, 190, 8, 128)
    FOR_K(1, 158, 4, 32)
    FOR_K(1, 158, 8, 64)
    FOR_K(2, 142, 8, 32)
    FOR_K(3, 191, 8, 128)
    FOR_K(4, 318, 8, 128)
};
static const int O2_CELLS_N = sizeof(O2_CELLS) / sizeof(O2Cell);

// park cells: form-independent, one kernel per (kind, U), W = 2
struct PCell { int ki; int u; O2Kern kern; };
#define PR(KI, V, U) {KI, U, &k_ov2<190, 0, V, U, 3, 2>},
static const PCell P_CELLS[] = {
    PR(0, 4, 32) PR(0, 4, 64) PR(0, 4, 128)
    PR(1, 5, 32) PR(1, 5, 64) PR(1, 5, 128)
    PR(2, 15, 32) PR(2, 15, 64) PR(2, 15, 128)
    PR(3, 16, 32) PR(3, 16, 64) PR(3, 16, 128)
    PR(4, 0, 32) PR(4, 0, 64) PR(4, 0, 128)
    PR(5, 2, 32) PR(5, 2, 64) PR(5, 2, 128)
    PR(6, 9, 32) PR(6, 9, 64) PR(6, 9, 128)
    PR(7, 8, 32) PR(7, 8, 64) PR(7, 8, 128)
};
static const int P_CELLS_N = sizeof(P_CELLS) / sizeof(PCell);

// ctrl cells: one kernel per (kind, U, W), CID / G are dummies
struct CCell { int ki; int u; O2Kern kern[2]; };
#define CR(KI, V, U)                                                \
  {KI, U, {&k_ov2<190, 0, V, U, 2, 1>, &k_ov2<190, 0, V, U, 2, 2>}},
static const CCell C_CELLS[] = {
    CR(0, 4, 32) CR(0, 4, 64) CR(0, 4, 128)
    CR(1, 5, 32) CR(1, 5, 64) CR(1, 5, 128)
    CR(2, 15, 32) CR(2, 15, 64) CR(2, 15, 128)
    CR(3, 16, 32) CR(3, 16, 64) CR(3, 16, 128)
    CR(4, 0, 32) CR(4, 0, 64) CR(4, 0, 128)
    CR(5, 2, 32) CR(5, 2, 64) CR(5, 2, 128)
    CR(6, 9, 32) CR(6, 9, 64) CR(6, 9, 128)
    CR(7, 8, 32) CR(7, 8, 64) CR(7, 8, 128)
};
static const int C_CELLS_N = sizeof(C_CELLS) / sizeof(CCell);

static void emit_row(MbWriter& w, const std::string& variant, int u,
                     const std::string& params, const std::uint64_t* h,
                     int reps, int wgs, int launch) {
  std::vector<double> samples, tails, spans, issue;
  for (int r = 1; r < reps; ++r) {
    std::uint64_t tw_max = 0, ts_min = ~0ull;
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
      if (i0 != 0 && i1 != 0)
        issue.push_back(static_cast<double>(i1 - i0));
    }
    for (int wp = 0; wp < 4 && wp < 4 * wgs; ++wp) {
      const std::uint64_t td = h[(wp * reps + r) * 5 + 4];
      if (td != 0 && tw_max != 0) {
        tails.push_back(static_cast<double>(
            static_cast<std::int64_t>(td - tw_max)));
        spans.push_back(static_cast<double>(td - ts_min));
      }
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
                          ",\"issue\":" + jvec(issue);
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
  MB_CHECK(cudaMalloc(&d_val, sizeof(std::uint32_t) * 8));
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
                  O2Kern kern, const std::string& form, int cid,
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
               args.reps, wg, l);
    }
  };
  for (int ci = 0; ci < O2_CELLS_N; ++ci) {
    const O2Cell& c = O2_CELLS[ci];
    const int wg_lo = c.mode == 1 ? 2 : 1;
    for (int wg = wg_lo; wg <= 2; ++wg) {
      const std::string variant =
          std::string(FORMS[c.fi].name) + ".g" + std::to_string(c.g) +
          "." + KINDS[c.ki].name + ".u" + std::to_string(c.u) + "." +
          (c.mode == 0 ? "self" : "other") + ".w" + std::to_string(wg);
      emit(variant, c.u, KINDS[c.ki].name, c.kern[wg - 1],
           FORMS[c.fi].name, FORMS[c.fi].cid,
           c.mode == 0 ? "self" : "other", wg, c.g);
    }
  }
  for (int ci = 0; ci < P_CELLS_N; ++ci) {
    const PCell& c = P_CELLS[ci];
    const std::string variant = "park." + std::string(KINDS[c.ki].name) +
                                ".u" + std::to_string(c.u) + ".w2";
    emit(variant, c.u, KINDS[c.ki].name, c.kern, "park", -1, "park", 2, 0);
  }
  for (int ci = 0; ci < C_CELLS_N; ++ci) {
    const CCell& c = C_CELLS[ci];
    for (int wg = 1; wg <= 2; ++wg) {
      const std::string variant = "ctrl." + std::string(KINDS[c.ki].name) +
                                  ".u" + std::to_string(c.u) + ".w" +
                                  std::to_string(wg);
      emit(variant, c.u, KINDS[c.ki].name, c.kern[wg - 1], "ctrl", -1,
           "ctrl", wg, 0);
    }
  }
  return 0;
}
