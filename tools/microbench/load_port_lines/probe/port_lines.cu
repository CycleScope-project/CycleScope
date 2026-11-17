// tools/microbench/load_port_lines/probe/port_lines.cu -- protocol.json of this directory.
// One CTA of W warps on one SM.  Per rep every warp issues U independent scalar global
// loads (lane l of load i reads 4 B at warp_base + pos(i) * STEP + l * S, STEP = max(128,
// 32 * S) bytes so that different loads never share a line), sums them, and passes a
// data-dependent predicated store before the closing clock read, so the window ends
// when the last datum is there.  Lane spread S: 4 (32 lanes in one line), 32 (8 lines,
// one sector per lane), 128 and 512 (32 lines).
// Levels: l1   pos(i) = i % 8 with the first rep dropped (lines stay in the L1);
//         l2   every rep reads new positions of a region a warm-up launch has just read;
//         dram every rep reads new positions after an evicting launch (512 MiB read).
//   port_lines --level l1|l2|dram --warps W --spread S --launches L
// stdout: META line, then for U in 16, 32, 64: "CELL <level> w<W> s<S> u<U> n <samples>
// warp_p50 <..> cta_p50 <..>" (per-warp window, and max end - min start of the CTA).
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)
static const char* arg(int c, char** v, const char* n, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!std::strcmp(v[i], n)) return v[i + 1]; return d; }
__device__ __forceinline__ unsigned long long clk() { unsigned long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
__global__ void k_port(const unsigned* buf, size_t seg_words, size_t step_words, unsigned spread_words, int U, int ring,
                       int reps, unsigned mask, unsigned long long* rec, unsigned* sink) {
  const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  const unsigned* wb = buf + (size_t)warp * seg_words + lane * spread_words;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const unsigned* rb = ring ? wb : wb + (size_t)rep * U * step_words;
    unsigned acc = 0;
    const unsigned long long t0 = clk();
    for (int i = 0; i < U; ++i) acc += rb[(size_t)(ring ? (i & (ring - 1)) : i) * step_words];
    if (acc & mask) sink[threadIdx.x] = acc;          // mask = 0: never taken, but waits for acc
    const unsigned long long t1 = clk();
    if (lane == 0) { rec[((size_t)warp * reps + rep) * 2] = t0; rec[((size_t)warp * reps + rep) * 2 + 1] = t1; }
  }
}
__global__ void k_touch(const unsigned* buf, size_t words, unsigned* out) {
  unsigned s = 0;
  for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 8; i < words; i += (size_t)gridDim.x * blockDim.x * 8) s += buf[i];
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}
int main(int argc, char** argv) {
  const std::string level = arg(argc, argv, "--level", "l2");
  const int W = atoi(arg(argc, argv, "--warps", "4")), S = atoi(arg(argc, argv, "--spread", "128")), L = atoi(arg(argc, argv, "--launches", "6"));
  const bool l1 = level == "l1", dram = level == "dram";
  const int ring = l1 ? 8 : 0, reps = l1 ? 9 : 3, UMAX = 64;
  const size_t step = std::max(128, 32 * S), seg = l1 ? (size_t)8 * step : (size_t)reps * UMAX * step;   // bytes per warp
  const size_t bytes = (size_t)W * seg + 65536, evict = (size_t)512 << 20;
  unsigned *buf, *ebuf, *sink, *tout; unsigned long long* rec;
  CK(cudaMalloc(&buf, bytes)); CK(cudaMemset(buf, 1, bytes)); CK(cudaMalloc(&sink, 4096 * 4)); CK(cudaMalloc(&tout, 4096 * 4));
  CK(cudaMalloc(&rec, (size_t)W * reps * 16)); CK(cudaMalloc(&ebuf, evict)); CK(cudaMemset(ebuf, 2, evict));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  std::printf("META device=\"%s\" level=%s warps=%d spread=%d step=%zu seg_bytes=%zu reps=%d ring=%d launches=%d\n", pr.name, level.c_str(), W, S, step, seg, reps, ring, L);
  std::vector<unsigned long long> h((size_t)W * reps * 2);
  for (int U : {16, 32, 64}) {
    std::vector<unsigned long long> wv, cv;
    for (int l = 0; l < L; ++l) {
      if (dram) { k_touch<<<2048, 256>>>(ebuf, evict / 4, tout); CK(cudaDeviceSynchronize()); }
      else if (!l1) { k_touch<<<2048, 256>>>(buf, bytes / 4, tout); CK(cudaDeviceSynchronize()); }
      k_port<<<1, 32 * W>>>(buf, seg / 4, step / 4, S / 4, U, ring, reps, 0u, rec, sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), rec, h.size() * 8, cudaMemcpyDeviceToHost));
      if (l == 0) continue;
      for (int r = l1 ? 1 : 0; r < reps; ++r) {
        unsigned long long lo = ~0ull, hi = 0;
        for (int w = 0; w < W; ++w) { const unsigned long long a = h[((size_t)w * reps + r) * 2], b = h[((size_t)w * reps + r) * 2 + 1]; wv.push_back(b - a); lo = std::min(lo, a); hi = std::max(hi, b); }
        cv.push_back(hi - lo);
      }
    }
    std::sort(wv.begin(), wv.end()); std::sort(cv.begin(), cv.end());
    std::printf("CELL %s w%d s%d u%d n %zu warp_p50 %llu warp_p10 %llu warp_p90 %llu cta_p50 %llu\n", level.c_str(), W, S, U, wv.size(), wv[wv.size() / 2], wv[wv.size() / 10], wv[wv.size() * 9 / 10], cv[cv.size() / 2]);
  }
  return 0;
}
