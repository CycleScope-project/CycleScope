// memlat_loaded probe: load-to-use latency of one dependent chase while the
// rest of the GPU streams reads through the same shared level.
//
// One kernel, grid = 1 + N CTAs, one CTA per SM (dynamic smem forces it):
//   CTA 0        chaser: warp 0 lane 0 walks a random page chain
//                (level dram: 2 GiB cold cycle, level l2: 2 MiB warmed cycle),
//                clock64 bracket over `hops` dependent loads after `warm_hops`
//                untimed hops that let the loaders reach steady state.
//   CTA 1..N     loaders: W warps each issue ld.global.cg.v4 (16 B / thread,
//                4 independent loads per iteration) over their own slice of
//                the loader buffer (level dram: 2 GiB total > L2; level l2:
//                16 MiB total, warmed before the start barrier) until the
//                chaser raises the stop flag.  Each CTA records bytes read and
//                its globaltimer window -> achieved read bandwidth.
// All CTAs arrive on a global counter before any timed work so the loaders
// are co-resident with the chaser (grid <= SM count, 1 CTA / SM).
//
// Output: one jsonl row per cell (blackwell.mechanism_curve_point.v1):
// samples = chaser cycles / hop per repetition; extra: loader bandwidth
// (B/ns, per repetition) and per-repetition raw numbers.
#include "common.cuh"
#include "curves_io.cuh"

#include <algorithm>
#include <numeric>
#include <random>

static std::vector<std::uint32_t> make_random_cycle(std::size_t n, std::mt19937_64& rng) {
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<std::uint32_t> chain(n);
  for (std::size_t k = 0; k < n; ++k) chain[order[k]] = order[(k + 1) % n];
  return chain;
}

__global__ void init_chain_kernel(std::uint32_t* buf, const std::uint32_t* perm,
                                  std::uint64_t npages) {
  const std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
  if (i < npages) buf[i * 1024] = perm[i];
}

__device__ __forceinline__ std::uint32_t ld_cg_u32(const std::uint32_t* p) {
  std::uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_cg_v4(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

struct Sync {
  unsigned arrived;   // CTAs that reached the start barrier
  unsigned stop;      // chaser sets 1 after its timed window
};

struct ChaserOut {
  long long cycles;        // clock64 over `hops`
  long long g0, g1;        // globaltimer ns at bracket start / end
  long long idx;           // keep the chain alive
};

struct LoaderOut {
  unsigned long long bytes;
  long long g0, g1;        // globaltimer ns window of the streaming loop
  unsigned sink;
};

// 1 CTA / SM: ask for more dynamic smem than half the SM (both devices have
// 228 KB per SM); the kernel never touches it.
constexpr int kSmemBytes = 120 * 1024;

__global__ void memlat_loaded_kernel(const std::uint32_t* chain, std::uint32_t start,
                                     int warm_hops, int hops,
                                     const uint4* lbuf, std::uint64_t lbuf_vec4_per_cta,
                                     int warm_loaders, Sync* sync, ChaserOut* cout,
                                     LoaderOut* lout) {
  extern __shared__ unsigned char smem_unused[];
  const bool chaser = blockIdx.x == 0;
  const int warps = blockDim.x / 32;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  // loader slice: [cta-1] * per_cta vec4s; warp w reads vec4 index
  // base + it*(warps*32*4) + k*(warps*32) + w*32 + lane  (k = 0..3)
  const uint4* my = lbuf + (chaser ? 0 : (blockIdx.x - 1) * lbuf_vec4_per_cta);
  const std::uint64_t step = (std::uint64_t)warps * 32u * 4u;
  const std::uint64_t iters = lbuf_vec4_per_cta / step;
  unsigned acc = 0;
  if (!chaser && warm_loaders) {
    // warm pass over the slice (level l2: make the loader footprint resident)
    for (std::uint64_t it = 0; it < iters; ++it) {
      const uint4* p = my + it * step + warp * 32 + lane;
      uint4 a = ld_cg_v4(p), b = ld_cg_v4(p + warps * 32),
            c = ld_cg_v4(p + 2 * warps * 32), d = ld_cg_v4(p + 3 * warps * 32);
      acc ^= a.x ^ b.y ^ c.z ^ d.w;
    }
  }
  if (chaser && warm_loaders) {
    // level l2: full walk of the chain so every page is resident
    if (threadIdx.x == 0) {
      std::uint32_t idx = start;
      for (int i = 0; i < warm_hops; ++i) idx = ld_cg_u32(chain + (std::uint64_t)idx * 1024);
      cout->idx = idx;
    }
  }
  __syncthreads();
  // start barrier: every CTA of the grid is resident before any timed work
  if (threadIdx.x == 0) {
    atomicAdd(&sync->arrived, 1u);
    while (*((volatile unsigned*)&sync->arrived) < gridDim.x) { }
  }
  __syncthreads();
  if (chaser) {
    if (threadIdx.x == 0) {
      std::uint32_t idx = start;
      // untimed hops: loaders ramp up (dram level: these are the first touches)
      for (int i = 0; i < warm_hops; ++i) idx = ld_cg_u32(chain + (std::uint64_t)idx * 1024);
      const long long g0 = (long long)globaltimer();
      const std::uint64_t t0 = rdclk64();
#pragma unroll 1
      for (int i = 0; i < hops; ++i) idx = ld_cg_u32(chain + (std::uint64_t)idx * 1024);
      const std::uint64_t t1 = rdclk64();
      const long long g1 = (long long)globaltimer();
      cout->cycles = (long long)(t1 - t0);
      cout->g0 = g0; cout->g1 = g1; cout->idx = idx;
      __threadfence();
      atomicExch(&sync->stop, 1u);
    }
    return;
  }
  // loader: stream the slice until stop
  const long long g0 = (long long)globaltimer();
  unsigned long long bytes = 0;
  std::uint64_t it = 0;
  while (true) {
    if ((it & 15u) == 0 && *((volatile unsigned*)&sync->stop)) break;
    const uint4* p = my + (it % iters) * step + warp * 32 + lane;
    uint4 a = ld_cg_v4(p), b = ld_cg_v4(p + warps * 32),
          c = ld_cg_v4(p + 2 * warps * 32), d = ld_cg_v4(p + 3 * warps * 32);
    acc ^= a.x ^ b.y ^ c.z ^ d.w;
    bytes += 64;  // 4 x 16 B per thread
    ++it;
  }
  const long long g1 = (long long)globaltimer();
  // one record per CTA: thread 0 sums the warps' bytes through smem
  __shared__ unsigned long long cta_bytes;
  if (threadIdx.x == 0) cta_bytes = 0;
  __syncthreads();
  if (lane == 0) atomicAdd(&cta_bytes, bytes * 32ull);
  __syncthreads();
  if (threadIdx.x == 0) {
    LoaderOut& o = lout[blockIdx.x - 1];
    o.bytes = cta_bytes; o.g0 = g0; o.g1 = g1; o.sink = acc;
  }
  (void)smem_unused;
}

__global__ void memset_kernel(std::uint32_t* p, std::uint64_t n) {
  const std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] = (std::uint32_t)i;
}

static std::string arr_ll(const std::vector<long long>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    s += std::to_string(v[i]);
    if (i + 1 < v.size()) s += ",";
  }
  return s + "]";
}
static std::string arr_d(const std::vector<double>& v) {
  std::string s = "[";
  char b[64];
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::snprintf(b, sizeof(b), "%.3f", v[i]);
    s += b;
    if (i + 1 < v.size()) s += ",";
  }
  return s + "]";
}

int main(int argc, char** argv) {
  std::string level = "dram", out = "memlat_loaded.jsonl";
  int loaders = 0, warps = 16, samples = 8, hops = -1, warm_hops = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto s = [&](const char* k, std::string* d) { if (a == k && i + 1 < argc) { *d = argv[++i]; return true; } return false; };
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    if (s("--level", &level) || s("--out", &out) || n("--loaders", &loaders) ||
        n("--warps", &warps) || n("--samples", &samples) || n("--hops", &hops) ||
        n("--warm-hops", &warm_hops)) continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  const bool l2 = level == "l2";
  if (!l2 && level != "dram") { std::fprintf(stderr, "level dram|l2\n"); return 2; }
  const DeviceInfo dev = query_device();
  if (loaders + 1 > dev.sm_count) { std::fprintf(stderr, "loaders+1 > SMs\n"); return 2; }
  if (hops < 0) hops = l2 ? 512 : 256;
  if (warm_hops < 0) warm_hops = l2 ? 0 : 64;  // l2: warm walk is the full chain
  std::mt19937_64 rng(20261109);

  // chase chain: one u32 per 4 KiB page
  const std::size_t chain_bytes = l2 ? (2ull << 20) : (2ull << 30);
  const std::uint64_t pages = chain_bytes / 4096;
  auto chain = make_random_cycle(pages, rng);
  std::uint32_t *chaind = nullptr, *permd = nullptr;
  CUDA_CHECK(cudaMalloc(&chaind, chain_bytes));
  CUDA_CHECK(cudaMalloc(&permd, pages * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMemcpy(permd, chain.data(), pages * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
  init_chain_kernel<<<(unsigned)((pages + 255) / 256), 256>>>(chaind, permd, pages);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const int l2_warm_hops = (int)pages;  // full permutation walk for level l2

  // loader buffer: dram 2 GiB (> L2 on both devices), l2 16 MiB (< L2 on both)
  const std::size_t lbuf_bytes = l2 ? (16ull << 20) : (2ull << 30);
  uint4* lbufd = nullptr;
  CUDA_CHECK(cudaMalloc(&lbufd, lbuf_bytes));
  memset_kernel<<<(unsigned)((lbuf_bytes / 4 + 255) / 256), 256>>>((std::uint32_t*)lbufd, lbuf_bytes / 4);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const std::uint64_t step = (std::uint64_t)warps * 32u * 4u;
  std::uint64_t per_cta = loaders ? (lbuf_bytes / 16) / loaders : (lbuf_bytes / 16);
  per_cta -= per_cta % step;

  // evict scratch for the dram level (same as memlat: 512 MiB memset)
  std::uint32_t* scratch = nullptr;
  const std::size_t scratch_bytes = 512ull << 20;
  CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));

  Sync* syncd; ChaserOut* coutd; LoaderOut* loutd;
  CUDA_CHECK(cudaMalloc(&syncd, sizeof(Sync)));
  CUDA_CHECK(cudaMalloc(&coutd, sizeof(ChaserOut)));
  CUDA_CHECK(cudaMalloc(&loutd, sizeof(LoaderOut) * std::max(1, loaders)));
  CUDA_CHECK(cudaFuncSetAttribute(memlat_loaded_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));

  std::vector<double> per_hop, bw_bns, win_ns;
  std::vector<long long> raw;
  std::string err;
  for (int s = 0; s < samples + 1; ++s) {  // first launch = warm-up, discarded
    if (!l2) {
      memset_kernel<<<(unsigned)((scratch_bytes / 4 + 255) / 256), 256>>>(scratch, scratch_bytes / 4);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemset(syncd, 0, sizeof(Sync)));
    CUDA_CHECK(cudaMemset(loutd, 0, sizeof(LoaderOut) * std::max(1, loaders)));
    const std::uint32_t start = (std::uint32_t)(rng() % pages);
    memlat_loaded_kernel<<<1 + loaders, warps * 32, kSmemBytes>>>(
        chaind, start, l2 ? l2_warm_hops : warm_hops, hops, lbufd, per_cta,
        l2 ? 1 : 0, syncd, coutd, loutd);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    if (s == 0) continue;
    ChaserOut co; CUDA_CHECK(cudaMemcpy(&co, coutd, sizeof(co), cudaMemcpyDeviceToHost));
    std::vector<LoaderOut> lo(std::max(1, loaders));
    CUDA_CHECK(cudaMemcpy(lo.data(), loutd, sizeof(LoaderOut) * lo.size(), cudaMemcpyDeviceToHost));
    raw.push_back(co.cycles);
    per_hop.push_back((double)co.cycles / hops);
    win_ns.push_back((double)(co.g1 - co.g0));
    // achieved loader read bandwidth: sum over CTAs of bytes / own window
    double bw = 0;
    for (int i = 0; i < loaders; ++i)
      if (lo[i].g1 > lo[i].g0) bw += (double)lo[i].bytes / (double)(lo[i].g1 - lo[i].g0);
    bw_bns.push_back(bw);
  }
  CurvePoint p;
  char fam[128];
  std::snprintf(fam, sizeof(fam), "memlat_loaded.%s.N%d.W%d", level.c_str(), loaders, warps);
  p.family = fam;
  p.quantity = "latency";
  p.x_name = "loader_ctas";
  p.x = loaders;
  p.y_name = "cycles_per_hop";
  p.samples = per_hop;
  char b[4096];
  std::snprintf(b, sizeof(b),
                "\"level\":\"%s\",\"loaders\":%d,\"warps_per_loader\":%d,\"hops\":%d,"
                "\"warm_hops\":%d,\"raw_elapsed_cycles\":%s,\"chaser_window_ns\":%s,"
                "\"loader_bytes_per_ns\":%s,\"loader_bytes_per_ns_median\":%.3f,"
                "\"chain_bytes\":%zu,\"loader_bytes\":%zu,\"policy\":\"cg\","
                "\"source\":\"memlat_loaded.measured\",\"gpu_name\":\"%s\","
                "\"sm_clock_mhz\":%d,\"sm_count\":%d,\"clock_locked\":false",
                level.c_str(), loaders, warps, hops, l2 ? l2_warm_hops : warm_hops,
                arr_ll(raw).c_str(), arr_d(win_ns).c_str(), arr_d(bw_bns).c_str(),
                bw_bns.empty() ? 0.0 : median_of(bw_bns), chain_bytes, lbuf_bytes,
                dev.name.c_str(), dev.sm_clock_mhz, dev.sm_count);
  p.extra_json = b;
  if (!err.empty()) p.error = err;
  append_jsonl(out.c_str(), p);
  cudaFree(chaind); cudaFree(permd); cudaFree(lbufd); cudaFree(scratch);
  cudaFree(syncd); cudaFree(coutd); cudaFree(loutd);
  return err.empty() ? 0 : 1;
}
