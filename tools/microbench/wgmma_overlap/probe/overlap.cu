// tools/microbench/wgmma_overlap/probe/overlap.cu — H100 (sm_90a).
//
// protocol.json: while the G chained wgmma ops of warpgroup 0 occupy the
// tensor pipe, a fully unrolled window of U units of kind V runs on the
// active warps; the per-warp excess over the ctrl cell attributes the
// issue slowdown to a candidate unit (R-F / R-P / R-S).
// One CTA of 4*W warps (W in {1,2,3}: warp w sits on SMSP w % 4, warpgroup
// g = warps 4g..4g+3, so every SMSP hosts W warps).  48 KiB dynamic smem
// holds the wgmma A / B tiles (no-swizzle K-major descriptors as in
// wgmma_handoff/probe/handoff.cu).  Per rep:
//   bar.sync; wg0 (MODE != ctrl): wgmma.fence; G x WOp<CID>::op(dd, da, db)
//   chained into one accumulator set; wgmma.commit_group; bar.sync;
//   active warps (self: all; other: wg >= 1; ctrl: all):
//     t_s = clock64; U x unit<V>; t_w = clock64;
//   wg0 (MODE != ctrl): wgmma.wait_group 0; t_d = clock64.
// Lane 0 of every warp records (t_s, t_w, t_d); inactive warps write 0 and
// t_d is 0 outside warpgroup 0.  One jsonl row per (variant, launch):
// samples = per-rep (rep >= 1) per-active-warp t_w - t_s; extra_json
// "tails"/"spans" = warpgroup 0's t_d - t_w / t_d - t_s of that launch.
// Window units copied from dispatch_contention/probe/contention_v3.cu
// (dep3r V0, dep2 V2, ilp8 V4, ilp8n V5, imad V8, fmnmx V9 max.NaN,
// mufu.ex2 V10) plus uadd V14: a warp-uniform 64-bit add chain on
// kernel-parameter values (no asm, no +z) so ptxas may choose the
// uniform datapath; the addend rotates over up8[8] so the U adds
// cannot fold into one multiply.
// Descriptor/op helpers come from the generated wgmma_impl.cuh
// (tensor_pipe/probe/gen_wgmma.py); for rs cells WOp<CID>::op takes the
// same (d, da, db) signature and supplies its own A register fragment.
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

// kinds: 0 dep3r  2 dep2  9 fmnmx  5 ilp8n  4 ilp8  8 imad
//        10 mufu.ex2  14 uadd (protocol window_kinds; numbering follows
//        contention_v3's variant table where they overlap)
struct CellArgs {
  float a, b;          // reusable pair (dep3r/dep2/ilp8)
  float c[4];          // fmnmx max.NaN ring
  float a8[8], b8[8];  // ilp8n: 16 distinct, never reusable
  unsigned ia, ib;     // imad
  unsigned long long uu;      // uadd: chain init (kernel-parameter uniform)
  unsigned long long up8[8];  // uadd: per-step addends, distinct so the
                              //       U adds cannot fold into one multiply
};

template <int V, int I>
__device__ __forceinline__ void unit(float* d, const CellArgs& o,
                                     unsigned long long& u) {
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
  } else if constexpr (V == 10) {
    asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(d[0]));
  } else if constexpr (V == 14) {
    // warp-uniform plain-C chain (protocol addendum 1): u and up8 come
    // straight from the kernel parameter block (no asm, no per-lane +z),
    // so ptxas may keep both on the uniform datapath (UIADD3); the gate
    // requires U* opcodes, else unrealised:not_uniform.
    u += o.up8[I & 7];
  }
}

template <int V, int U, int I, int N>
struct EmitBlock {
  static __device__ __forceinline__ void go(float* d, const CellArgs& o,
                                            unsigned long long& u) {
    if constexpr (I < U && N > 0) {
      unit<V, I>(d, o, u);
      EmitBlock<V, U, I + 1, N - 1>::go(d, o, u);
    }
  }
};

template <int V, int U, int I = 0>
struct EmitUnits {
  // two-level unroll: ptxas caps template recursion depth (~200);
  // 64-unit inner blocks keep depth at U/64 + 64 (as contention_v3)
  static __device__ __forceinline__ void go(float* d, const CellArgs& o,
                                            unsigned long long& u) {
    EmitBlock<V, U, I, 64>::go(d, o, u);
    if constexpr (I + 64 < U)
      EmitUnits<V, U, I + 64>::go(d, o, u);
  }
};

// MODE: 0 = self (wg0 issues the ops, all warps run the window)
//       1 = other (wg0 issues + waits, warpgroups 1..W-1 run the window)
//       2 = ctrl (no wgmma at all; CID is a dummy)
// WB = warps-per-SMSP bound of the instance (1..3): the launch bound
// 128 * WB gives the W = 1 / 2 kernels the full register budget (the
// n256 accumulator set alone is 128 registers); only W = 3 is capped.
//
// Role-split rep loops (protocol addendum 2): ptxas C7518 serialises
// wgmma.mma_async whose issue / window / wait live in different branch
// regions, so each role runs its own rep loop and issue, window and
// wait share one straight-line region.  __syncthreads() is replaced by
// named barrier 1 with the full CTA count so both roles meet at the
// same two barriers per rep.
__device__ __forceinline__ void bar_sync1(int nthreads) {
  asm volatile("bar.sync 1, %0;" ::"r"(nthreads) : "memory");
}

template <int CID, int G, int V, int U, int MODE, int WB>
__global__ void __launch_bounds__(128 * WB, 1) k_overlap(
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
  // rematerialisation inside the timed window (contention_v3).  The
  // uadd operands are exempt on purpose: they must stay warp-uniform.
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
  // mufu chains start at 0.5; fma/fmnmx chains as contention_v3
  for (int i = 0; i < 8; ++i)
    d[i] = (V == 10) ? 0.5f + 0.0625f * i : 0.25f + warp + i;
  unsigned long long u = o.uu;
  const int nth = 128 * wgroups;
  if constexpr (MODE == 2) {
    // ctrl: one loop for all warps, two barriers then the window
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      bar_sync1(nth);
      const std::uint64_t t_s = mb_clock();
      EmitUnits<V, U>::go(d, o, u);
      const std::uint64_t t_w = mb_clock();
      if (lane == 0) {
        out[(warp * reps + rep) * 3 + 0] = t_s;
        out[(warp * reps + rep) * 3 + 1] = t_w;
        out[(warp * reps + rep) * 3 + 2] = 0;
      }
    }
  } else if (wg == 0) {
    // issuer role: issue, window, wait in one straight-line region
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      // commit_group per op (protocol addendum 1): G groups of one op
      // issue back-to-back and stay in flight
#pragma unroll
      for (int g = 0; g < G; ++g) {
        WOp<CID>::op(dd, da, db);
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      }
      bar_sync1(nth);
      std::uint64_t t_s = 0, t_w = 0;
      if constexpr (MODE == 0) {  // self: wg0 runs the window too
        t_s = mb_clock();
        EmitUnits<V, U>::go(d, o, u);
        t_w = mb_clock();
      }
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      const std::uint64_t t_d = mb_clock();
      if (lane == 0) {
        out[(warp * reps + rep) * 3 + 0] = t_s;
        out[(warp * reps + rep) * 3 + 1] = t_w;
        out[(warp * reps + rep) * 3 + 2] = t_d;
      }
    }
  } else {
    // window-only role (warpgroups >= 1): same two barriers per rep
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      bar_sync1(nth);
      bar_sync1(nth);
      const std::uint64_t t_s = mb_clock();
      EmitUnits<V, U>::go(d, o, u);
      const std::uint64_t t_w = mb_clock();
      if (lane == 0) {
        out[(warp * reps + rep) * 3 + 0] = t_s;
        out[(warp * reps + rep) * 3 + 1] = t_w;
        out[(warp * reps + rep) * 3 + 2] = 0;
      }
    }
  }
  // fold every live register into a sink (keep accumulators + window
  // registers live across the rep loop)
  if (threadIdx.x == 0) {
    std::uint32_t s = 0;
    for (int i = 0; i < 8; ++i) s += __float_as_uint(d[i]);
    s += static_cast<std::uint32_t>(u) +
         static_cast<std::uint32_t>(u >> 32);
    for (int i = 0; i < WD<CID>::ND; ++i) {
      std::uint32_t w0;
      std::memcpy(&w0, &dd[i], sizeof(w0) < sizeof(T) ? sizeof(w0) : sizeof(T));
      s += w0;
    }
    dval[0] = s;
  }
}

using OKern = void (*)(std::uint64_t*, std::uint32_t*, int, int, CellArgs,
                       const float*, const unsigned*);

// kind table: (v, name, u list) — u index ui into KINDS[ki].us
static const struct {
  int v; const char* name; int us[2]; int nu;
} KINDS[] = {
    {0, "dep3r",    {128, 0},   1},
    {2, "dep2",     {128, 0},   1},
    {9, "fmnmx",    {128, 0},   1},
    {5, "ilp8n",    {128, 256}, 2},
    {4, "ilp8",     {128, 256}, 2},
    {8, "imad",     {128, 0},   1},
    {10, "mufu.ex2", {128, 0},  1},
    {14, "uadd",    {128, 0},   1},
};
static const int KINDS_N = sizeof(KINDS) / sizeof(KINDS[0]);

static const struct { int cid; const char* name; int g; } FORMS[] = {
    {190, "hgmma_bf16_n256_ss", 16},
    {191, "hgmma_bf16_n256_rs", 16},
    {142, "hgmma_bf16_n64_ss", 64},
    {318, "qgmma_e4m3_n256_ss", 16},
};

struct OCell {
  int fi; int ki; int ui; int mode; OKern kern[3];  // kern[W - 1]
};

#define KW(CID, G, V, U, MODE)                                        \
  {&k_overlap<CID, G, V, U, MODE, 1>, &k_overlap<CID, G, V, U, MODE, 2>, \
   &k_overlap<CID, G, V, U, MODE, 3>}
#define OMODES(FI, CID, G, KI, V, UI, U)                              \
  {FI, KI, UI, 0, KW(CID, G, V, U, 0)},                              \
      {FI, KI, UI, 1, KW(CID, G, V, U, 1)},

#define FOR_KINDS(FI, CID, G)                                         \
  OMODES(FI, CID, G, 0, 0, 0, 128)                                    \
  OMODES(FI, CID, G, 1, 2, 0, 128)                                    \
  OMODES(FI, CID, G, 2, 9, 0, 128)                                    \
  OMODES(FI, CID, G, 3, 5, 0, 128)                                    \
  OMODES(FI, CID, G, 3, 5, 1, 256)                                    \
  OMODES(FI, CID, G, 4, 4, 0, 128)                                    \
  OMODES(FI, CID, G, 4, 4, 1, 256)                                    \
  OMODES(FI, CID, G, 5, 8, 0, 128)                                    \
  OMODES(FI, CID, G, 6, 10, 0, 128)                                   \
  OMODES(FI, CID, G, 7, 14, 0, 128)

static const OCell O_CELLS[] = {
    FOR_KINDS(0, 190, 16)
    FOR_KINDS(1, 191, 16)
    FOR_KINDS(2, 142, 64)
    FOR_KINDS(3, 318, 16)
};
static const int O_CELLS_N = sizeof(O_CELLS) / sizeof(OCell);

// ctrl cells: form-independent (CID is a dummy), one kernel per (kind, u)
#define CROW(KI, V, UI, U) {0, KI, UI, 2, KW(190, 16, V, U, 2)},
static const OCell C_CELLS[] = {
    CROW(0, 0, 0, 128)
    CROW(1, 2, 0, 128)
    CROW(2, 9, 0, 128)
    CROW(3, 5, 0, 128)
    CROW(3, 5, 1, 256)
    CROW(4, 4, 0, 128)
    CROW(4, 4, 1, 256)
    CROW(5, 8, 0, 128)
    CROW(6, 10, 0, 128)
    CROW(7, 14, 0, 128)
};
static const int C_CELLS_N = sizeof(C_CELLS) / sizeof(OCell);

static void run_cell(const MbArgs& args, MbWriter& w, const OCell& c,
                     int nwarps_w, const CellArgs& a, std::uint64_t* d_out,
                     std::uint32_t* d_val, const float* d_zeros,
                     const unsigned* d_uzeros) {
  const int u = KINDS[c.ki].us[c.ui];
  const char* mode_name = c.mode == 0 ? "self" : c.mode == 1 ? "other" : "ctrl";
  const int wg_lo = c.mode == 1 ? 2 : 1;
  for (int wg = wg_lo; wg <= nwarps_w; ++wg) {
    const std::string variant =
        (c.mode == 2 ? std::string("ctrl") : std::string(FORMS[c.fi].name)) +
        "." + KINDS[c.ki].name + ".u" + std::to_string(u) +
        (c.mode == 2 ? "" : std::string(".") + mode_name) + ".w" +
        std::to_string(wg);
    if (!mb_selected(args, variant)) continue;
    OKern kern = c.kern[wg - 1];
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
      std::vector<std::uint64_t> h(4 * wg * args.reps * 3);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      std::vector<double> samples, tails, spans;
      for (int r = 1; r < args.reps; ++r) {
        // overlap check (protocol G_rule): the ops must outlive the
        // window of every active warp -> tail = t_d - max active t_w
        // (in `other` mode warpgroup 0 has no window of its own)
        std::uint64_t tw_max = 0, ts_min = ~0ull;
        for (int wp = 0; wp < 4 * wg; ++wp) {
          const std::uint64_t ts = h[(wp * args.reps + r) * 3 + 0];
          const std::uint64_t tw = h[(wp * args.reps + r) * 3 + 1];
          if (ts != 0 && tw != 0) {
            samples.push_back(static_cast<double>(tw - ts));
            tw_max = std::max(tw_max, tw);
            ts_min = std::min(ts_min, ts);
          }
        }
        for (int wp = 0; wp < 4 && wp < 4 * wg; ++wp) {
          const std::uint64_t td = h[(wp * args.reps + r) * 3 + 2];
          if (td != 0 && tw_max != 0) {
            tails.push_back(static_cast<double>(
                static_cast<std::int64_t>(td - tw_max)));
            spans.push_back(static_cast<double>(td - ts_min));
          }
        }
      }
      std::string params =
          "\"form\":\"" +
          (c.mode == 2 ? std::string("ctrl")
                       : std::string(FORMS[c.fi].name)) +
          "\",\"cid\":" +
          std::to_string(c.mode == 2 ? -1 : FORMS[c.fi].cid) +
          ",\"kind\":\"" + KINDS[c.ki].name + "\",\"u\":" +
          std::to_string(u) + ",\"mode\":\"" + mode_name + "\",\"w\":" +
          std::to_string(wg) + ",\"g\":" +
          std::to_string(c.mode == 2 ? 0 : FORMS[c.fi].g) +
          ",\"launch\":" + std::to_string(l);
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
                                ",\"spans\":" + jvec(spans);
      w.row(variant, u, params, samples, extra);
    }
  }
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 12 * args.reps * 3));
  MB_CHECK(cudaMalloc(&d_val, sizeof(std::uint32_t) * 8));
  float* d_zeros = nullptr;
  unsigned* d_uzeros = nullptr;
  MB_CHECK(cudaMalloc(&d_zeros, sizeof(float) * 384));
  MB_CHECK(cudaMalloc(&d_uzeros, sizeof(unsigned) * 384));
  MB_CHECK(cudaMemset(d_zeros, 0, sizeof(float) * 384));
  MB_CHECK(cudaMemset(d_uzeros, 0, sizeof(unsigned) * 384));
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
  a.uu = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < 8; ++i) a.up8[i] = 0x100000001b3ULL + 0x9e3779b9ULL * i;
  for (int ci = 0; ci < O_CELLS_N; ++ci)
    run_cell(args, w, O_CELLS[ci], 3, a, d_out, d_val, d_zeros, d_uzeros);
  for (int ci = 0; ci < C_CELLS_N; ++ci)
    run_cell(args, w, C_CELLS[ci], 3, a, d_out, d_val, d_zeros, d_uzeros);
  return 0;
}
