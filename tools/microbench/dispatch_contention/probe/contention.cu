// tools/microbench/dispatch_contention/probe/contention.cu -- protocol v1.
// One CTA of 4*W warps (W in {1,2,3,4}: warp w sits on SMSP w % 4 so
// every SMSP hosts W warps), window = U dependent-chain units fully
// unrolled, U in {64,128,256}.  Every warp's lane 0 times its own
// window with clock64; all lanes execute the chain.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "tensor_pipe/mb.cuh"

// variants: 0 ffma  1 fmnmx  2 mufu  3 imad  4 mixed  5 ilp8
template <int V, int I>
__device__ __forceinline__ void unit(float *d, float a, float b,
                                     const float *c, unsigned ia,
                                     unsigned ib) {
  if constexpr (V == 0 || (V == 4 && (I & 1) == 0)) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[0]) : "f"(a), "f"(b));
  } else if constexpr (V == 1 || (V == 4 && (I & 1) == 1)) {
    // max.NaN defeats ptxas's FMNMX3 pairing (two max.f32 on the ring
    // would fuse into one trinary op on Blackwell)
    asm volatile("max.NaN.f32 %0, %0, %1;"
                 : "+f"(d[0]) : "f"(c[I & 3]));
  } else if constexpr (V == 2) {
    asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(d[0]));
  } else if constexpr (V == 3) {
    unsigned *u = reinterpret_cast<unsigned *>(d);
    asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                 : "+r"(u[0]) : "r"(ia), "r"(ib));
  } else if constexpr (V == 5) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(d[I & 7]) : "f"(a), "f"(b));
  }
}

template <int V, int U, int I, int N>
struct EmitBlock {
  static __device__ __forceinline__ void go(float *d, float a, float b,
                                            const float *c, unsigned ia,
                                            unsigned ib) {
    if constexpr (I < U && N > 0) {
      unit<V, I>(d, a, b, c, ia, ib);
      EmitBlock<V, U, I + 1, N - 1>::go(d, a, b, c, ia, ib);
    }
  }
};

template <int V, int U, int I = 0>
struct EmitUnits {
  // two-level unroll: ptxas caps template recursion depth (~200);
  // 64-unit inner blocks keep depth at U/64 + 64
  static __device__ __forceinline__ void go(float *d, float a, float b,
                                            const float *c, unsigned ia,
                                            unsigned ib) {
    EmitBlock<V, U, I, 64>::go(d, a, b, c, ia, ib);
    if constexpr (I + 64 < U)
      EmitUnits<V, U, I + 64>::go(d, a, b, c, ia, ib);
  }
};

template <int V, int U>
__global__ void k_cell(std::uint64_t *out, float *dval, int reps,
                       int wgroups, float a, float b, float c0, float c1,
                       float c2, float c3, unsigned ia, unsigned ib) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool active = warp < 4 * wgroups;
  float d[8];
  const float c[4] = {c0, c1, c2, c3};
  for (int i = 0; i < 8; ++i) d[i] = 0.25f + warp + i;
  for (int rep = 0; rep < reps; ++rep) {
    // addendum 2026-11-06: bar.sync before every window so the W warps
    // per SMSP overlap instead of drifting apart between reps
    __syncthreads();
    if (!active) continue;
    const std::uint64_t t_s = mb_clock();
    EmitUnits<V, U>::go(d, a, b, c, ia, ib);
    const std::uint64_t t_w = mb_clock();
    if (lane == 0)
      out[warp * reps + rep] = t_w - t_s;
  }
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int i = 0; i < 8; ++i) s += d[i];
    dval[0] = s;
  }
}

using CKern = void (*)(std::uint64_t *, float *, int, int, float, float,
                       float, float, float, float, unsigned, unsigned);
struct CCell { const char *variant; CKern kern; };

#define ROWS(V, NAME)                                              \
  {NAME ".u64", &k_cell<V, 64>}, {NAME ".u128", &k_cell<V, 128>},  \
  {NAME ".u256", &k_cell<V, 256>},
static const struct { const char *variant; int u; CKern kern; } C_CELLS[] = {
  {"ffma", 64, &k_cell<0, 64>},   {"ffma", 128, &k_cell<0, 128>},
  {"ffma", 256, &k_cell<0, 256>}, {"fmnmx", 64, &k_cell<1, 64>},
  {"fmnmx", 128, &k_cell<1, 128>},{"fmnmx", 256, &k_cell<1, 256>},
  {"mufu", 64, &k_cell<2, 64>},   {"mufu", 128, &k_cell<2, 128>},
  {"mufu", 256, &k_cell<2, 256>}, {"imad", 64, &k_cell<3, 64>},
  {"imad", 128, &k_cell<3, 128>}, {"imad", 256, &k_cell<3, 256>},
  {"mixed", 64, &k_cell<4, 64>},  {"mixed", 128, &k_cell<4, 128>},
  {"mixed", 256, &k_cell<4, 256>},{"ilp8", 64, &k_cell<5, 64>},
  {"ilp8", 128, &k_cell<5, 128>}, {"ilp8", 256, &k_cell<5, 256>},
};

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t *d_out = nullptr;
  float *d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 16 * args.reps));
  MB_CHECK(cudaMalloc(&d_val, sizeof(float) * 8));
  for (const auto &c : C_CELLS) {
    for (int wg = 1; wg <= 4; ++wg) {
      std::string variant = std::string(c.variant) + ".w" +
                            std::to_string(wg);
      if (!mb_selected(args, variant)) continue;
      std::vector<double> samples;
      for (int l = 0; l < args.launches; ++l) {
        c.kern<<<1, 128 * wg>>>(d_out, d_val, args.reps, wg, 1.0000001f,
                                0.5f, 0.75f, 1.5f, 2.25f, 3.0f,
                                3u + wg, 5u + wg);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(4 * wg * args.reps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int wp = 0; wp < 4 * wg; ++wp)
          for (int r = 1; r < args.reps; ++r)
            samples.push_back(static_cast<double>(h[wp * args.reps + r]));
      }
      const std::string params = "\"variant\":\"" +
          std::string(c.variant) + "\",\"w\":" + std::to_string(wg);
      w.row(variant, c.u, params, samples);
    }
  }
  return 0;
}
