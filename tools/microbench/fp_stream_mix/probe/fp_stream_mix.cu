// tools/microbench/fp_stream_mix/probe/fp_stream_mix.cu -- protocol.json of
// the same directory.  Harness of dispatch_contention/probe/contention_v3.cu
// (one CTA of 4*W warps, W in {1,2,3,4}: warp w sits on SMSP w % 4, window =
// U instructions fully unrolled, bar.sync before every rep, lane 0 of every
// warp records t_s / t_w; rows per warp window and per CTA window).
// Variants (instruction I of the window):
//   0 mufu8      ex2 on d[I & 7]                       (independent MUFU.EX2)
//   1 ffma8      fma on d[I & 7], distinct operands    (control, no MUFU)
//   2 mix5       I % 5 == 0: ex2 on d[(I / 5) & 7]; 1..3: fma on e[I & 3];
//                4: max.NaN chain on m
//   3 mix5c      mix5 with the ex2 replaced by an fma on d[(I / 5) & 7]
//   4 run3       I % 7 in 0..2: ex2 on d[I & 7]; 3: max.NaN chain on m;
//                4..6: fma on e[I & 3]
//   5 mufumax    even I: ex2 on d[(I / 2) & 7]; odd I: max.NaN chain on m
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "tensor_pipe/mb.cuh"

struct CellArgs { float a8[8], b8[8], c[4]; };
struct Regs { float d[8], e[4], m; };

template <int V, int I>
__device__ __forceinline__ void unit(Regs &r, const CellArgs &o) {
  auto ex2 = [&](int k) {
    asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(r.d[k]));
  };
  auto fmad = [&](int k) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(r.d[k]) : "f"(o.a8[k]), "f"(o.b8[k]));
  };
  auto fmae = [&](int k) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(r.e[k]) : "f"(o.a8[k]), "f"(o.b8[k + 4]));
  };
  auto mx = [&]() {
    asm volatile("max.NaN.f32 %0, %0, %1;" : "+f"(r.m) : "f"(o.c[I & 3]));
  };
  if constexpr (V == 0) {
    ex2(I & 7);
  } else if constexpr (V == 1) {
    fmad(I & 7);
  } else if constexpr (V == 2 || V == 3) {
    if constexpr (I % 5 == 0) {
      if constexpr (V == 2) ex2((I / 5) & 7); else fmad((I / 5) & 7);
    } else if constexpr (I % 5 == 4) {
      mx();
    } else {
      fmae(I & 3);
    }
  } else if constexpr (V == 4) {
    if constexpr (I % 7 < 3) ex2(I & 7);
    else if constexpr (I % 7 == 3) mx();
    else fmae(I & 3);
  } else if constexpr (V == 5) {
    if constexpr ((I & 1) == 0) ex2((I / 2) & 7); else mx();
  }
}

template <int V, int U, int I, int N>
struct EmitBlock {
  static __device__ __forceinline__ void go(Regs &r, const CellArgs &o) {
    if constexpr (I < U && N > 0) {
      unit<V, I>(r, o);
      EmitBlock<V, U, I + 1, N - 1>::go(r, o);
    }
  }
};
template <int V, int U, int I = 0>
struct EmitUnits {
  static __device__ __forceinline__ void go(Regs &r, const CellArgs &o) {
    EmitBlock<V, U, I, 64>::go(r, o);
    if constexpr (I + 64 < U) EmitUnits<V, U, I + 64>::go(r, o);
  }
};

template <int V, int U>
__global__ void k_cell(std::uint64_t *out, float *dval, int reps,
                       int wgroups, CellArgs args, const float *zeros) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool active = warp < 4 * wgroups;
  // constants through a per-lane add of a loaded zero: non-uniform, kept in
  // GPRs, no LDC inside the window (as contention_v3)
  const float z = zeros[threadIdx.x];
  CellArgs o;
  for (int i = 0; i < 8; ++i) { o.a8[i] = args.a8[i] + z; o.b8[i] = args.b8[i] + z; }
  for (int i = 0; i < 4; ++i) o.c[i] = args.c[i] + z;
  Regs r;
  for (int i = 0; i < 8; ++i) r.d[i] = 0.5f + 0.0625f * i + z;
  for (int i = 0; i < 4; ++i) r.e[i] = 0.25f + warp + i + z;
  r.m = 0.125f + z;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (!active) continue;
    const std::uint64_t t_s = mb_clock();
    EmitUnits<V, U>::go(r, o);
    const std::uint64_t t_w = mb_clock();
    if (lane == 0) {
      out[(warp * reps + rep) * 2 + 0] = t_s;
      out[(warp * reps + rep) * 2 + 1] = t_w;
    }
  }
  if (threadIdx.x == 0) {
    float s = r.m;
    for (int i = 0; i < 8; ++i) s += r.d[i];
    for (int i = 0; i < 4; ++i) s += r.e[i];
    dval[0] = s;
  }
}

using CKern = void (*)(std::uint64_t *, float *, int, int, CellArgs, const float *);
#define ROWS(V, NAME) \
  {NAME, 70, &k_cell<V, 70>}, {NAME, 140, &k_cell<V, 140>}, {NAME, 280, &k_cell<V, 280>},
static const struct { const char *variant; int u; CKern kern; } C_CELLS[] = {
  ROWS(0, "mufu8") ROWS(1, "ffma8") ROWS(2, "mix5") ROWS(3, "mix5c")
  ROWS(4, "run3") ROWS(5, "mufumax")
};

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t *d_out = nullptr;
  float *d_val = nullptr, *d_zeros = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 16 * args.reps * 2));
  MB_CHECK(cudaMalloc(&d_val, sizeof(float) * 8));
  MB_CHECK(cudaMalloc(&d_zeros, sizeof(float) * 1024));
  MB_CHECK(cudaMemset(d_zeros, 0, sizeof(float) * 1024));
  for (const auto &c : C_CELLS) {
    for (int wg = 1; wg <= 4; ++wg) {
      const std::string variant = std::string(c.variant) + ".w" + std::to_string(wg);
      if (!mb_selected(args, variant) &&
          !mb_selected(args, std::string(c.variant) + ".cta.w" + std::to_string(wg)))
        continue;
      CellArgs a{};
      for (int i = 0; i < 8; ++i) { a.a8[i] = 1.0000001f + 0.0078125f * i; a.b8[i] = 0.5f + 0.03125f * i; }
      a.c[0] = 0.75f; a.c[1] = 1.5f; a.c[2] = 2.25f; a.c[3] = 3.0f;
      std::vector<double> warp_samples, cta_samples;
      for (int l = 0; l < args.launches; ++l) {
        c.kern<<<1, 128 * wg>>>(d_out, d_val, args.reps, wg, a, d_zeros);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(), cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(4 * wg * args.reps * 2);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
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
      const std::string base = "\"variant\":\"" + std::string(c.variant) + "\",\"w\":" + std::to_string(wg);
      w.row(variant, c.u, base + ",\"window\":\"warp\"", warp_samples);
      w.row(std::string(c.variant) + ".cta.w" + std::to_string(wg), c.u, base + ",\"window\":\"cta\"", cta_samples);
    }
  }
  return 0;
}
