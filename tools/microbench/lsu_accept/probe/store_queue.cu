// tools/microbench/lsu_accept/probe/store_queue.cu -- protocol v4 (frozen
// tools/microbench/lsu_accept/protocol-v4.json): the LSU global-store
// queue.  One CTA of 32*W threads (W in {1,2,4,8}; warp w sits on SMSP
// w % 4), bar.sync before every repetition.  Window = D = 24 back-to-back
// global stores on 24 distinct 64-bit pointer registers (per-lane
// coalesced, each request its own 128 B-aligned region, distinct per
// warp and per k), with a %clock64 read after EVERY store; lane 0 of
// each warp records the 25 stamps (t_0 before the first store, t_k after
// store k).  The per-store issue gap t_k - t_{k-1} is the SMSP's store
// issue interval while the store queue has room and becomes the queue's
// acceptance / drain interval once the k-th store has to wait for room
// (lg_throttle).  Variants: stg16 (st.v4.b32, 512 B per request), stg8
// (st.v2.b32, 256 B), stg4 (st.b32, 128 B), none (24 clock reads with
// no store between them: the clock read's own cost).  Protocol v5
// (protocol-v5.json) added stg8 and none to the v4 probe.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "tensor_pipe/mb.cuh"

enum : int { V_STG16, V_STG4, V_STG8, V_NONE };
constexpr int kD = 24;

template <int V>
__device__ __forceinline__ void store_one(std::uint64_t p, unsigned s0,
                                          unsigned s1, unsigned s2,
                                          unsigned s3) {
  if constexpr (V == V_STG16)
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "r"(s0), "r"(s1), "r"(s2), "r"(s3) : "memory");
  else if constexpr (V == V_STG8)
    asm volatile("st.global.v2.b32 [%0], {%1, %2};"
                 :: "l"(p), "r"(s0), "r"(s1) : "memory");
  else if constexpr (V == V_STG4)
    asm volatile("st.global.b32 [%0], %1;" :: "l"(p), "r"(s0) : "memory");
  else
    asm volatile("" :: "l"(p), "r"(s0) : "memory");  // none: clock only
}

template <int V, int I>
struct Chain {
  static __device__ __forceinline__ void go(const std::uint64_t *p,
                                            std::uint64_t *t,
                                            const unsigned *src) {
    if constexpr (I < kD) {
      store_one<V>(p[I], src[0] + I, src[1], src[2], src[3]);
      t[I + 1] = mb_clock();
      Chain<V, I + 1>::go(p, t, src);
    }
  }
};

template <int V>
__global__ void k_cell(std::uint64_t *out, unsigned *sink, char *base,
                       const unsigned *uzeros, const std::uint64_t *offs,
                       int reps, int bytes_per_lane) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned uz = uzeros[threadIdx.x];
  std::uint64_t p[kD];
  std::uint64_t t[kD + 1];
  const unsigned src[4] = {0x11223344u + uz, 0x55667788u + uz,
                           0x99aabbccu + uz, 0xddeeff00u + uz};
  for (int k = 0; k < kD; ++k)
    // per-(k, warp, lane) offsets from a table: opaque to ptxas, each
    // request its own 512 B region (lane * bytes within it)
    p[k] = reinterpret_cast<std::uint64_t>(base) +
           offs[(k * 8 + warp) * 32 + lane] * bytes_per_lane / 16 + uz;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    // drain: every store of the previous repetition acknowledged before
    // the next window (a fence waits for the stores to be performed)
    asm volatile("fence.sc.gpu;" ::: "memory");
    __syncthreads();
    t[0] = mb_clock();
    Chain<V, 0>::go(p, t, src);
    asm volatile("" ::: "memory");
    if (lane == 0)
      for (int k = 0; k <= kD; ++k)
        out[(warp * reps + rep) * (kD + 1) + k] = t[k];
  }
  if (threadIdx.x == 0) {
    unsigned s = 0;
    for (int k = 0; k < kD; ++k) s += static_cast<unsigned>(p[k]);
    sink[0] = s;
  }
}

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t *d_out = nullptr, *d_offs = nullptr;
  unsigned *d_sink = nullptr, *d_uzeros = nullptr;
  char *d_base = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 8 * args.reps * (kD + 1)));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(unsigned) * 8));
  MB_CHECK(cudaMalloc(&d_uzeros, sizeof(unsigned) * 1024));
  MB_CHECK(cudaMemset(d_uzeros, 0, sizeof(unsigned) * 1024));
  MB_CHECK(cudaMalloc(&d_base, 1u << 20));  // 24 req * 8 w * 512 B < 1 MB
  MB_CHECK(cudaMalloc(&d_offs, sizeof(std::uint64_t) * kD * 8 * 32));
  {
    // offsets in units of a 16 B lane slot: request (k, warp) owns the
    // 512 B region (k * 8 + warp); stg4 scales them down by 4 in-kernel
    std::vector<std::uint64_t> ho(kD * 8 * 32);
    for (int i = 0; i < kD * 8 * 32; ++i)
      ho[i] = static_cast<std::uint64_t>(i) * 16;
    MB_CHECK(cudaMemcpy(d_offs, ho.data(), ho.size() * 8,
                        cudaMemcpyHostToDevice));
  }
  struct Cell { const char *variant; int v; int bytes; };
  const Cell cells[] = {{"stg16", V_STG16, 16}, {"stg8", V_STG8, 8},
                        {"stg4", V_STG4, 4}, {"none", V_NONE, 16}};
  for (const auto &c : cells) {
    for (int wg : {1, 2, 4, 8}) {
      const std::string variant = std::string(c.variant) + ".w" +
                                  std::to_string(wg);
      if (!mb_selected(args, variant)) continue;
      // samples per k: gap t_k - t_{k-1}; one row per k (u = k)
      std::vector<std::vector<double>> gaps(kD + 1);
      for (int l = 0; l < args.launches; ++l) {
        switch (c.v) {
          case V_STG16:
            k_cell<V_STG16><<<1, 32 * wg>>>(d_out, d_sink, d_base, d_uzeros,
                                            d_offs, args.reps, c.bytes);
            break;
          case V_STG8:
            k_cell<V_STG8><<<1, 32 * wg>>>(d_out, d_sink, d_base, d_uzeros,
                                           d_offs, args.reps, c.bytes);
            break;
          case V_STG4:
            k_cell<V_STG4><<<1, 32 * wg>>>(d_out, d_sink, d_base, d_uzeros,
                                           d_offs, args.reps, c.bytes);
            break;
          default:
            k_cell<V_NONE><<<1, 32 * wg>>>(d_out, d_sink, d_base, d_uzeros,
                                           d_offs, args.reps, c.bytes);
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(wg * args.reps * (kD + 1));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r)
          for (int wp = 0; wp < wg; ++wp) {
            const std::uint64_t *row = &h[(wp * args.reps + r) * (kD + 1)];
            for (int k = 1; k <= kD; ++k)
              gaps[k].push_back(static_cast<double>(row[k] - row[k - 1]));
            gaps[0].push_back(static_cast<double>(row[kD] - row[0]));
          }
      }
      for (int k = 0; k <= kD; ++k) {
        const std::string params =
            "\"variant\":\"" + std::string(c.variant) + "\",\"w\":" +
            std::to_string(wg) + ",\"k\":" + std::to_string(k) +
            ",\"bytes_per_lane\":" + std::to_string(c.bytes) +
            ",\"window\":\"" + (k ? "gap" : "total") + "\"";
        w.row(variant, k, params, gaps[k]);
      }
    }
  }
  return 0;
}
