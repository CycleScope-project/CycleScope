// tools/microbench/dispatch_contention/probe/contention_v3.cu -- protocol v3.
// One CTA of 4*W warps (W in {1,2,3,4}: warp w sits on SMSP w % 4 so
// every SMSP hosts W warps), window = U units fully unrolled, U in
// {64,128,256}, bar.sync before every rep.  Lane 0 of every active warp
// records BOTH clocks (t_s, t_w); the host derives the per-warp window
// T_w = t_w - t_s and the CTA window max_w t_w - min_w t_s per rep and
// emits one jsonl row per window kind (variant ".w<W>" / ".cta.w<W>").
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "tensor_pipe/mb.cuh"

// variants: 0 dep3r  1 dep3n  2 dep2  3 dep1  4 ilp8  5 ilp8n  6 ilp8n2
//           7 ilp2  8 imad  9 mixed  10..13 mufu.{ex2,rsqrt,lg2,tanh}
// (v2's mufu.rcp folded to nothing and mufu.sin expanded; both dropped)
struct CellArgs {
  float a, b;          // reusable pair (dep3r/dep2/ilp8/ilp2/mixed)
  float c[4];          // mixed FMNMX ring
  float ca[4], cb[4];  // dep3n ring: never reusable
  float a8[8], b8[8];  // ilp8n / ilp8n2: 16 distinct, never reusable
  unsigned ia, ib;     // imad
};

template <int V, int I>
__device__ __forceinline__ void unit(float *d, const CellArgs &o) {
  if constexpr (V == 0 || (V == 9 && (I & 1) == 0)) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[0]) : "f"(o.a), "f"(o.b));
  } else if constexpr (V == 1) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[0]) : "f"(o.ca[I & 3]), "f"(o.cb[I & 3]));
  } else if constexpr (V == 2) {
    asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                 : "+f"(d[0]) : "f"(o.a));
  } else if constexpr (V == 3) {
    asm volatile("fma.rn.f32 %0, %0, 0f3F800001, %0;"
                 : "+f"(d[0]));
  } else if constexpr (V == 4) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[I & 7]) : "f"(o.a), "f"(o.b));
  } else if constexpr (V == 5) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[I & 7]) : "f"(o.a8[I & 7]), "f"(o.b8[I & 7]));
  } else if constexpr (V == 6) {
    asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                 : "+f"(d[I & 7]) : "f"(o.a8[I & 7]));
  } else if constexpr (V == 7) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[I & 1]) : "f"(o.a), "f"(o.b));
  } else if constexpr (V == 8) {
    unsigned *u = reinterpret_cast<unsigned *>(d);
    asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                 : "+r"(u[0]) : "r"(o.ia), "r"(o.ib));
  } else if constexpr (V == 9) {
    // max.NaN defeats ptxas's FMNMX3 pairing (v1)
    asm volatile("max.NaN.f32 %0, %0, %1;"
                 : "+f"(d[0]) : "f"(o.c[I & 3]));
  } else if constexpr (V == 10) {
    asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(d[0]));
  } else if constexpr (V == 11) {
    asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(d[0]));
  } else if constexpr (V == 12) {
    asm volatile("lg2.approx.ftz.f32 %0, %0;" : "+f"(d[0]));
  } else if constexpr (V == 13) {
    asm volatile("tanh.approx.f32 %0, %0;" : "+f"(d[0]));
  }
}

template <int V, int U, int I, int N>
struct EmitBlock {
  static __device__ __forceinline__ void go(float *d, const CellArgs &o) {
    if constexpr (I < U && N > 0) {
      unit<V, I>(d, o);
      EmitBlock<V, U, I + 1, N - 1>::go(d, o);
    }
  }
};

template <int V, int U, int I = 0>
struct EmitUnits {
  // two-level unroll: ptxas caps template recursion depth (~200);
  // 64-unit inner blocks keep depth at U/64 + 64
  static __device__ __forceinline__ void go(float *d, const CellArgs &o) {
    EmitBlock<V, U, I, 64>::go(d, o);
    if constexpr (I + 64 < U)
      EmitUnits<V, U, I + 64>::go(d, o);
  }
};

template <int V, int U>
__global__ void k_cell(std::uint64_t *out, float *dval, int reps,
                       int wgroups, CellArgs args, const float *zeros,
                       const unsigned *uzeros) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool active = warp < 4 * wgroups;
  // every constant goes through a per-lane add of a loaded zero before
  // the rep loop: the value is non-uniform and depends on a load, so
  // ptxas must keep it in a GPR -- no uniform registers, no LDC
  // rematerialisation inside the timed window (v3)
  const float z = zeros[threadIdx.x];
  const unsigned uz = uzeros[threadIdx.x];
  CellArgs o;
  o.a = args.a + z; o.b = args.b + z;
  for (int i = 0; i < 4; ++i) {
    o.c[i] = args.c[i] + z;
    o.ca[i] = args.ca[i] + z;
    o.cb[i] = args.cb[i] + z;
  }
  for (int i = 0; i < 8; ++i) {
    o.a8[i] = args.a8[i] + z;
    o.b8[i] = args.b8[i] + z;
  }
  o.ia = args.ia + uz; o.ib = args.ib + uz;
  float d[8];
  // mufu chains start at 0.5 (ex2/lg2 chains may run to inf/nan -- fine,
  // MUFU latency is value-independent); fma chains as v1
  for (int i = 0; i < 8; ++i)
    d[i] = (V >= 10) ? 0.5f + 0.0625f * i : 0.25f + warp + i;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (!active) continue;
    const std::uint64_t t_s = mb_clock();
    EmitUnits<V, U>::go(d, o);
    const std::uint64_t t_w = mb_clock();
    if (lane == 0) {
      out[(warp * reps + rep) * 2 + 0] = t_s;
      out[(warp * reps + rep) * 2 + 1] = t_w;
    }
  }
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int i = 0; i < 8; ++i) s += d[i];
    dval[0] = s;
  }
}

using CKern = void (*)(std::uint64_t *, float *, int, int, CellArgs,
                       const float *, const unsigned *);

#define ROWS(V, NAME)                                                  \
  {NAME, 64, &k_cell<V, 64>}, {NAME, 128, &k_cell<V, 128>},            \
  {NAME, 256, &k_cell<V, 256>},
static const struct { const char *variant; int u; CKern kern; } C_CELLS[] = {
  ROWS(0, "dep3r") ROWS(1, "dep3n") ROWS(2, "dep2") ROWS(3, "dep1")
  ROWS(4, "ilp8") ROWS(5, "ilp8n") ROWS(6, "ilp8n2") ROWS(7, "ilp2")
  ROWS(8, "imad") ROWS(9, "mixed")
  ROWS(10, "mufu.ex2") ROWS(11, "mufu.rsqrt") ROWS(12, "mufu.lg2")
  ROWS(13, "mufu.tanh")
};

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t *d_out = nullptr;
  float *d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 16 * args.reps * 2));
  MB_CHECK(cudaMalloc(&d_val, sizeof(float) * 8));
  float *d_zeros = nullptr;
  unsigned *d_uzeros = nullptr;
  MB_CHECK(cudaMalloc(&d_zeros, sizeof(float) * 1024));
  MB_CHECK(cudaMalloc(&d_uzeros, sizeof(unsigned) * 1024));
  MB_CHECK(cudaMemset(d_zeros, 0, sizeof(float) * 1024));
  MB_CHECK(cudaMemset(d_uzeros, 0, sizeof(unsigned) * 1024));
  for (const auto &c : C_CELLS) {
    const bool mufu = std::strncmp(c.variant, "mufu.", 5) == 0;
    for (int wg = 1; wg <= 4; ++wg) {
      if (mufu && wg > 1) continue;  // mufu cells run at W = 1 only
      const std::string variant = std::string(c.variant) + ".w" +
                                  std::to_string(wg);
      if (!mb_selected(args, variant) &&
          !mb_selected(args, std::string(c.variant) + ".cta.w" +
                            std::to_string(wg)))
        continue;
      CellArgs a{};
      a.a = 1.0000001f;
      a.b = 0.5f;
      a.c[0] = 0.75f; a.c[1] = 1.5f; a.c[2] = 2.25f; a.c[3] = 3.0f;
      for (int i = 0; i < 4; ++i) {
        a.ca[i] = 1.0000001f + 0.03125f * i;
        a.cb[i] = 0.5f + 0.0625f * i;
      }
      for (int i = 0; i < 8; ++i) {
        a.a8[i] = 1.0000001f + 0.0078125f * i;
        a.b8[i] = 0.5f + 0.03125f * i;
      }
      a.ia = 3u + wg;
      a.ib = 5u + wg;
      std::vector<double> warp_samples, cta_samples;
      for (int l = 0; l < args.launches; ++l) {
        c.kern<<<1, 128 * wg>>>(d_out, d_val, args.reps, wg, a, d_zeros,
                                d_uzeros);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(4 * wg * args.reps * 2);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r) {
          std::uint64_t ts_min = ~0ull, tw_max = 0;
          for (int wp = 0; wp < 4 * wg; ++wp) {
            const std::uint64_t ts = h[(wp * args.reps + r) * 2 + 0];
            const std::uint64_t tw = h[(wp * args.reps + r) * 2 + 1];
            warp_samples.push_back(static_cast<double>(tw - ts));
            ts_min = std::min(ts_min, ts);
            tw_max = std::max(tw_max, tw);
          }
          cta_samples.push_back(static_cast<double>(tw_max - ts_min));
        }
      }
      const std::string base = "\"variant\":\"" +
          std::string(c.variant) + "\",\"w\":" + std::to_string(wg);
      w.row(variant, c.u, base + ",\"window\":\"warp\"", warp_samples);
      w.row(std::string(c.variant) + ".cta.w" + std::to_string(wg),
            c.u, base + ",\"window\":\"cta\"", cta_samples);
    }
  }
  return 0;
}
