// tools/microbench/shfl_scope/probe/shfl_scope.cu -- protocol v1.
// One CTA of 32*n threads (n warps, warp w lands on SMSP w); per rep a
// bar.sync then a per-warp clock64 window of U = 128 fully unrolled
// shfl.sync.{bfly,idx}.b32 on rotating destinations d[i & 7] with
// lane-varying data (mask 0xffffffff, asm volatile so ptxas cannot
// merge or reorder).  The chain cells (n = 1 only) are dependent
// d = shfl(d) and reproduce the installed SHFL result latency.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "tensor_pipe/mb.cuh"

// V: 0 bfly stream  1 idx stream  2 bfly chain  3 idx chain.
// Streams read+write their destination (eight parallel chains): a
// pure-write SHFL is dead for all but the last write to each d[i]
// (ptxas folds the window to 8 ops / 1 op for a warp-uniform idx) --
// the "+r" form keeps every op live.  idx passes the lane operand as a
// register so the op cannot be proven idempotent.
template <int V, int I>
__device__ __forceinline__ void unit(unsigned *d, unsigned lane) {
  if constexpr (V == 0) {
    asm volatile("shfl.sync.bfly.b32 %0, %0, 0x1, 0x1f, 0xffffffff;"
                 : "+r"(d[I & 7]));
  } else if constexpr (V == 1) {
    asm volatile("shfl.sync.idx.b32 %0, %0, %1, 0x1f, 0xffffffff;"
                 : "+r"(d[I & 7]) : "r"(lane));
  } else if constexpr (V == 2) {
    asm volatile("shfl.sync.bfly.b32 %0, %0, 0x1, 0x1f, 0xffffffff;"
                 : "+r"(d[0]));
  } else {
    asm volatile("shfl.sync.idx.b32 %0, %0, %1, 0x1f, 0xffffffff;"
                 : "+r"(d[0]) : "r"(lane));
  }
}

template <int V, int U, int I, int N>
struct EmitBlock {
  static __device__ __forceinline__ void go(unsigned *d,
                                            unsigned lane) {
    if constexpr (I < U && N > 0) {
      unit<V, I>(d, lane);
      EmitBlock<V, U, I + 1, N - 1>::go(d, lane);
    }
  }
};

template <int V, int U, int I = 0>
struct EmitUnits {
  static __device__ __forceinline__ void go(unsigned *d,
                                            unsigned lane) {
    EmitBlock<V, U, I, 64>::go(d, lane);
    if constexpr (I + 64 < U) EmitUnits<V, U, I + 64>::go(d, lane);
  }
};

template <int V, int N>
__global__ void k_cell(std::uint64_t *out, unsigned *dval, int reps,
                       unsigned seed) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned d[8];
  for (int i = 0; i < 8; ++i)
    d[i] = seed + warp * 37u + lane * 1664525u + (unsigned)i;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t_s = mb_clock();
    EmitUnits<V, 128>::go(d, lane);
    const std::uint64_t t_w = mb_clock();
    if (lane == 0) {
      out[(warp * reps + rep) * 2 + 0] = t_s;
      out[(warp * reps + rep) * 2 + 1] = t_w;
    }
  }
  if (threadIdx.x == 0) {
    unsigned s = 0;
    for (int i = 0; i < 8; ++i) s += d[i];
    dval[0] = s;
  }
}

using CKern = void (*)(std::uint64_t *, unsigned *, int, unsigned);

static const struct { const char *variant; const char *kind;
                      int n; CKern kern; } C_CELLS[] = {
  {"bfly.n1", "bfly", 1, &k_cell<0, 1>},
  {"bfly.n2", "bfly", 2, &k_cell<0, 2>},
  {"bfly.n4", "bfly", 4, &k_cell<0, 4>},
  {"idx.n1", "idx", 1, &k_cell<1, 1>},
  {"idx.n2", "idx", 2, &k_cell<1, 2>},
  {"idx.n4", "idx", 4, &k_cell<1, 4>},
  {"chain.bfly.n1", "bfly", 1, &k_cell<2, 1>},
  {"chain.idx.n1", "idx", 1, &k_cell<3, 1>},
};

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t *d_out = nullptr;
  unsigned *d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 4 * args.reps * 2));
  MB_CHECK(cudaMalloc(&d_val, sizeof(unsigned) * 8));
  for (const auto &c : C_CELLS) {
    const std::string variant = c.variant;
    if (!mb_selected(args, variant)) continue;
    for (int l = 0; l < args.launches; ++l) {
      c.kern<<<1, 32 * c.n>>>(d_out, d_val, args.reps,
                              0x9e3779b9u + 131u * l);
      const cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(c.n * args.reps * 2);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      std::vector<double> warp_samples;
      for (int r = 1; r < args.reps; ++r)
        for (int wp = 0; wp < c.n; ++wp)
          warp_samples.push_back(static_cast<double>(
              h[(wp * args.reps + r) * 2 + 1] -
              h[(wp * args.reps + r) * 2 + 0]));
      const std::string base = "\"kind\":\"" + std::string(c.kind) +
          "\",\"n\":" + std::to_string(c.n) +
          ",\"launch\":" + std::to_string(l + 1);
      w.row(variant, 128, base, warp_samples);
    }
  }
  return 0;
}
