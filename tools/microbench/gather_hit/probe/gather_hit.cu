// tools/microbench/gather_hit/probe/gather_hit.cu -- protocol.json.
// One CTA per participating SM, W warps (warp w on SMSP w % 4); all CTAs
// meet on a global arrival counter.  Per round: bar.sync; clock t0;
// every warp issues K ld.global.v4.u32 back to back (lane l of load k
// reads 16 B at row r(warp, k, l mod L, round); lanes l and l + L share
// a row and read different 16 B offsets when L = 16); xor every loaded
// word into a register; clock t1 (lane 0 records t1 - t0); bar.sync.
// Row cursors are per (cta, warp): row = ((gw * ROUNDS + round) * K +
// k) * L + l % L, taken mod the table -- unique inside a launch while
// the table budget allows (hit mode 24 MiB / cold mode 1 GiB; the
// largest cells exceed the table and wrap, evenly).
//   gather_hit --mode hit|cold --ctas N --warps W --loads K --lines L
//              --rounds R --launches L --out file.jsonl
// Modes: hit = 24 MiB table, a k_touch kernel passes over the whole
//   table before every timed launch; cold = 1 GiB table, L2 evicted by
//   a 512 MiB cudaMemset before every timed launch.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); \
  std::exit(1); } } while (0)
__device__ __forceinline__ unsigned long long clk() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint4 ld128(const char* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

constexpr int kRounds = 64;

template <int K>
__global__ void __launch_bounds__(128, 1) k_gather_hit(
    const char* table, unsigned nrows, int L, int rounds,
    unsigned* arrived, unsigned nparticipants,
    unsigned long long* rec, unsigned* sink) {
  const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31,
               nw = blockDim.x >> 5;
  const unsigned gw = blockIdx.x * nw + warp;
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(arrived, 1u);
    while (*((volatile unsigned*)arrived) < nparticipants) { }
  }
  __syncthreads();
  const unsigned lane_row = lane % L, lane_off = lane / L;
  const unsigned long long base =
      ((unsigned long long)gw * kRounds) * K * L;
  unsigned acc = 0;
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    const char* p[K > 0 ? K : 1];
    for (int k = 0; k < K; ++k) {
      const unsigned long long row =
          (base + ((unsigned long long)r * K + k) * L + lane_row) %
          nrows;
      p[k] = table + row * 128 + lane_off * 16;
    }
    __syncthreads();
    const unsigned long long t0 = clk();
    uint4 v[K > 0 ? K : 1];
    for (int k = 0; k < K; ++k) v[k] = ld128(p[k]);
    unsigned s = 0;
    for (int k = 0; k < K; ++k)
      s ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    const unsigned long long t1 = clk();
    acc += s;
    if (lane == 0) rec[(size_t)gw * rounds + r] = t1 - t0;
    __syncthreads();
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = acc;
}

// hit-mode warmup: pass over the whole table once so it is L2-resident.
__global__ void k_touch(const uint4* table, size_t n, unsigned* sink) {
  unsigned s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) s += table[i].x;
  if (threadIdx.x == 0) sink[blockIdx.x] = s;
}

int main(int argc, char** argv) {
  std::string out = "gather_hit.jsonl", mode = "hit";
  int N = 1, W = 4, K = 8, L = 32, launches = 8, rounds = kRounds;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto gi = [&](const char* k, int* d) {
      if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; }
      return false; };
    if (a == "--out" && i + 1 < argc) { out = argv[++i]; continue; }
    if (a == "--mode" && i + 1 < argc) { mode = argv[++i]; continue; }
    if (gi("--ctas", &N) || gi("--warps", &W) || gi("--loads", &K) ||
        gi("--lines", &L) || gi("--launches", &launches) ||
        gi("--rounds", &rounds)) continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  if (mode != "hit" && mode != "cold") {
    std::fprintf(stderr, "--mode hit|cold\n"); return 2; }
  if (K != 0 && K != 1 && K != 4 && K != 8) {
    std::fprintf(stderr, "--loads 0|1|4|8\n"); return 2; }
  if (L != 16 && L != 32) { std::fprintf(stderr, "--lines 16|32\n");
    return 2; }
  if (W < 1 || W > 4) { std::fprintf(stderr, "--warps 1..4\n"); return 2; }
  if (rounds > kRounds) { std::fprintf(stderr, "--rounds <= %d\n",
                                       kRounds); return 2; }

  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  if (N > pr.multiProcessorCount) {
    std::fprintf(stderr, "ctas > SMs\n"); return 2; }
  const size_t table_bytes = mode == "hit" ? (size_t)24 << 20
                                           : (size_t)1 << 30;
  const unsigned nrows = (unsigned)(table_bytes / 128);
  char* table; char* evict = nullptr;
  CK(cudaMalloc(&table, table_bytes));
  CK(cudaMemset(table, 1, table_bytes));
  if (mode == "cold") {
    CK(cudaMalloc(&evict, (size_t)512 << 20));
    CK(cudaMemset(evict, 2, (size_t)512 << 20));
  }
  unsigned* arrived; unsigned long long* rec; unsigned* sink;
  const size_t nrec = (size_t)N * W * rounds;
  CK(cudaMalloc(&arrived, 4));
  CK(cudaMalloc(&rec, nrec * 8));
  CK(cudaMalloc(&sink, std::max(N, 4096) * 4));
  CK(cudaMemset(arrived, 0, 4));
  std::vector<unsigned long long> h(nrec);
  std::vector<unsigned long long> samples;
  auto run_one = [&](int kk, FILE* f) {
    for (int l = 0; l < launches; ++l) {
      if (mode == "hit") {
        k_touch<<<1024, 256>>>((const uint4*)table,
                               table_bytes / 16, sink);
      } else {
        CK(cudaMemset(evict, l, (size_t)512 << 20));
      }
      CK(cudaMemset(arrived, 0, 4));
      CK(cudaDeviceSynchronize());
      switch (kk) {
        case 0: k_gather_hit<0><<<N, 32 * W>>>(table, nrows, L, rounds,
              arrived, N, rec, sink); break;
        case 1: k_gather_hit<1><<<N, 32 * W>>>(table, nrows, L, rounds,
              arrived, N, rec, sink); break;
        case 4: k_gather_hit<4><<<N, 32 * W>>>(table, nrows, L, rounds,
              arrived, N, rec, sink); break;
        case 8: k_gather_hit<8><<<N, 32 * W>>>(table, nrows, L, rounds,
              arrived, N, rec, sink); break;
      }
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), rec, nrec * 8, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < nrec; ++i)
        if (i % rounds != 0) samples.push_back(h[i]);   // round 0 dropped
    }
    std::sort(samples.begin(), samples.end());
    auto q = [&](double p) {
      return samples.empty() ? 0ull
          : samples[(size_t)(p * (samples.size() - 1))]; };
    std::fprintf(f,
        "{\"schema\":\"cyclescope.microbench_sample.v1\","
        "\"family\":\"gather_hit.%s.N%d.W%d.K%d.L%d\",\"mode\":\"%s\","
        "\"ctas\":%d,\"warps\":%d,\"loads\":%d,\"lines\":%d,"
        "\"p10\":%llu,\"p50\":%llu,\"p90\":%llu,\"n\":%zu,"
        "\"gpu_name\":\"%s\",\"sm_count\":%d}\n",
        mode.c_str(), N, W, kk, L, mode.c_str(), N, W, kk, L,
        q(0.10), q(0.50), q(0.90), samples.size(), pr.name,
        pr.multiProcessorCount);
    samples.clear();
  };
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str());
    return 1; }
  run_one(K, f);
  std::fclose(f);
  return 0;
}
