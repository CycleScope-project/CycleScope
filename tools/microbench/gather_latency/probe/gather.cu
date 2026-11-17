// tools/microbench/gather_latency/probe/gather.cu -- protocol.json of this directory.
// Every warp, per iteration: K 128-bit global loads per lane from one row of a table
// shared by all CTAs (row = hash(cta, warp, iteration) % rows, 16 KiB per row; lane l,
// load j reads 16 B at row + l * 512 + j * 64), then uses the data.  Lane 0 records the
// clock before the first load and after the use.  One CTA per SM (grid = SM count) or
// --grid 1.  Before every launch a separate buffer of --evict-mib is read to push the
// table out of the L2.
//   gather --warps W --k K --rows R --iters I --grid G --launches L --evict-mib E
// stdout: META line; per launch "LAUNCH l span <cycles of CTA 0>" and
// "IT l <iteration> n <samples> p10 p50 p90" over all CTAs / warps.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)
static const char* arg(int c, char** v, const char* n, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!std::strcmp(v[i], n)) return v[i + 1]; return d; }
__device__ __forceinline__ unsigned long long clk() { unsigned long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned h32(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
__global__ void k_gather(const uint4* table, unsigned rows, int k, int iters, unsigned long long* rec, float* sink) {
  const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  float acc = 0.f;
  const unsigned long long tb = clk();
  for (int it = 0; it < iters; ++it) {
    const unsigned row = h32(blockIdx.x * 7919u + warp * 104729u + (unsigned)it * 15485863u) % rows;
    const uint4* p = table + (size_t)row * 1024 + lane * 32;   // 16 KiB rows, 512 B per lane
    __syncwarp();
    const unsigned long long t0 = clk();
    uint4 v[8];
    for (int j = 0; j < 8; ++j) if (j < k) v[j] = p[j * 4];     // 64 B apart
    unsigned s = 0;
    for (int j = 0; j < 8; ++j) if (j < k) s += v[j].x ^ v[j].w;
    acc += (float)(s & 1023u);
    const unsigned long long t1 = clk();
    if (lane == 0) { const size_t o = (((size_t)blockIdx.x * nw + warp) * iters + it) * 2; rec[o] = t0 - tb; rec[o + 1] = t1 - t0; }
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = acc + (float)(clk() - tb);
}
__global__ void k_evict(const uint4* buf, size_t n, unsigned* out) {
  unsigned s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) s += buf[i].x;
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}
int main(int argc, char** argv) {
  const int W = atoi(arg(argc, argv, "--warps", "16")), K = atoi(arg(argc, argv, "--k", "7")), I = atoi(arg(argc, argv, "--iters", "16"));
  const unsigned rows = (unsigned)atoi(arg(argc, argv, "--rows", "64"));
  const int L = atoi(arg(argc, argv, "--launches", "5")); const size_t evict = (size_t)atoi(arg(argc, argv, "--evict-mib", "512")) << 20;
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  const int G = atoi(arg(argc, argv, "--grid", "0")) ? atoi(arg(argc, argv, "--grid", "0")) : pr.multiProcessorCount;
  uint4 *table, *ebuf; unsigned long long* rec; float* sink; unsigned* eout;
  const size_t tb = (size_t)rows * 16384;
  CK(cudaMalloc(&table, tb)); CK(cudaMemset(table, 1, tb)); CK(cudaMalloc(&ebuf, evict)); CK(cudaMemset(ebuf, 2, evict));
  const size_t nrec = (size_t)G * W * I * 2;
  CK(cudaMalloc(&rec, nrec * 8)); CK(cudaMalloc(&sink, G * 4)); CK(cudaMalloc(&eout, 4096 * 4));
  int khz = 0; CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  std::printf("META device=\"%s\" sms=%d grid=%d warps=%d k=%d rows=%u iters=%d launches=%d evict_mib=%zu table_mib=%zu clock_khz=%d\n",
              pr.name, pr.multiProcessorCount, G, W, K, rows, I, L, evict >> 20, tb >> 20, khz);
  std::vector<unsigned long long> h(nrec);
  for (int l = 0; l < L; ++l) {
    k_evict<<<4096, 256>>>(ebuf, evict / 16, eout); CK(cudaDeviceSynchronize());
    k_gather<<<G, 32 * W>>>(table, rows, K, I, rec, sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), rec, nrec * 8, cudaMemcpyDeviceToHost));
    unsigned long long span = 0;
    for (int w = 0; w < W; ++w) span = std::max(span, h[(((size_t)w) * I + I - 1) * 2] + h[(((size_t)w) * I + I - 1) * 2 + 1]);
    std::printf("LAUNCH %d span %llu\n", l, span);
    for (int it = 0; it < I; ++it) {
      std::vector<unsigned long long> v;
      for (size_t c = 0; c < (size_t)G * W; ++c) v.push_back(h[(c * I + it) * 2 + 1]);
      std::sort(v.begin(), v.end());
      std::printf("IT %d %d n %zu %llu %llu %llu\n", l, it, v.size(), v[v.size() / 10], v[v.size() / 2], v[v.size() * 9 / 10]);
    }
  }
  return 0;
}
