/*
 * dispatch.cu -- B300 (sm_103a) CTA dispatch facts for the full-GPU
 * simulator (M4; rebuild/full-gpu-plan.md S0.11 / S1.11 / S2.6):
 *
 *   waves <launches> <ctas_per_sm> <waves> <hold_us>
 *       Launch a grid of sm_count * ctas_per_sm * waves CTAs whose dynamic
 *       smem request lets exactly <ctas_per_sm> CTAs share one SM.  Every CTA
 *       records %smid, %warpid of its first warp, entry/exit %globaltimer and
 *       clock64, then holds the SM for <hold_us> microseconds before exiting
 *       (so the next wave can only start when a slot is released).  Each
 *       thread block has 128 threads (4 warps).
 *       Host output: one CTA line per CTA with all timestamps.  The analysis
 *       (fullgpu_dispatch_analyze.py) derives per SM: first-wave entry order
 *       and spacing, exit(wave k) -> entry(wave k+1) gap on the same SM
 *       (slot release + dispatch + launch), whether later-wave CTAs take CTA
 *       ids in order and whether they reuse the freed warp slots.
 *   tail <launches> <hold_us>
 *       Grid of sm_count + 1 CTAs (one CTA per SM): the extra CTA can only
 *       enter when the first CTA exits -> the single slot-free-to-entry gap.
 *
 * Output lines: META ..., CTA launch=.. cta=.. smid=.. warpid=.. gt_in=..
 * clk_in=.. gt_out=.. clk_out=.., LAUNCH launch=.. host_ns=.. (host wall
 * time of the launch call to the end of the kernel, for sanity), DONE.
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(err_),      \
              __FILE__, __LINE__);                                         \
      exit(2);                                                             \
    }                                                                      \
  } while (0)

struct Rec {
  unsigned smid, warpid;
  unsigned long long gt_in, clk_in, gt_out, clk_out;
};

__device__ __forceinline__ unsigned smid_reg() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
__device__ __forceinline__ unsigned warpid_reg() {
  unsigned s;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(s));
  return s;
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

extern __shared__ unsigned char dyn_smem[];

__global__ void __launch_bounds__(128) hold_kernel(Rec* rec, unsigned long long hold_ns) {
  const unsigned long long t0 = gtimer();
  const unsigned long long c0 = clock64();
  if (threadIdx.x == 0) {
    rec[blockIdx.x].smid = smid_reg();
    rec[blockIdx.x].warpid = warpid_reg();
    rec[blockIdx.x].gt_in = t0;
    rec[blockIdx.x].clk_in = c0;
    dyn_smem[0] = (unsigned char)t0;
  }
  while (gtimer() - t0 < hold_ns) { }
  __syncthreads();
  if (threadIdx.x == 0) {
    rec[blockIdx.x].gt_out = gtimer();
    rec[blockIdx.x].clk_out = clock64();
  }
}

__global__ void __launch_bounds__(128) refill_kernel(Rec* rec, unsigned long long hold_ns,
                                                   unsigned long long* first, int first_wave) {
  const unsigned long long t0 = gtimer();
  const unsigned long long c0 = clock64();
  if (threadIdx.x == 0) {
    rec[blockIdx.x].smid = smid_reg();
    rec[blockIdx.x].warpid = warpid_reg();
    rec[blockIdx.x].gt_in = t0;
    rec[blockIdx.x].clk_in = c0;
    dyn_smem[0] = (unsigned char)t0;
    if ((int)blockIdx.x < first_wave) {
      atomicMin(first, t0);
    }
  }
  __syncthreads();
  if ((int)blockIdx.x < first_wave) {
    // Spin until (global earliest first-wave entry) + hold_ns so all
    // first-wave slots free within one tick.
    unsigned long long rel;
    do { rel = *first; } while (rel == ~0ull);
    rel += hold_ns;
    while (gtimer() < rel) { }
  } else {
    while (gtimer() - t0 < hold_ns) { }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    rec[blockIdx.x].gt_out = gtimer();
    rec[blockIdx.x].clk_out = clock64();
  }
}

__global__ void __launch_bounds__(128) blocker_kernel(const unsigned* mask, Rec* rec) {
  if (threadIdx.x == 0) {
    rec[blockIdx.x].smid = smid_reg();
    rec[blockIdx.x].warpid = warpid_reg();
    rec[blockIdx.x].gt_in = gtimer();
    rec[blockIdx.x].clk_in = clock64();
  }
  __syncthreads();
  const unsigned s = rec[blockIdx.x].smid;
  if (mask[s >> 5] & (1u << (s & 31))) {
    const unsigned long long t0 = gtimer();
    while (gtimer() - t0 < 2000ULL * 1000ULL) { }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    rec[blockIdx.x].gt_out = gtimer();
    rec[blockIdx.x].clk_out = clock64();
  }
}

// Minimal scan of configs/hardware/b300_sm_topology.json for the blocker
// masks: gpcs[].sms arrays and tpc_pairs[][2].
static std::set<unsigned> topo_mask(const std::string& text, const std::string& mask) {
  std::set<unsigned> out;
  if (mask == "gpc0123") {
    size_t pos = 0;
    while ((pos = text.find("\"gpc\"", pos)) != std::string::npos) {
      const char* colon = text.c_str() + pos + 5;   // '"gpc":' -> colon at pos+5
      if (*colon != ':') { ++pos; continue; }
      const int g = atoi(colon + 1);
      const size_t sms = text.find("\"sms\"", pos);
      const size_t end = text.find(']', sms);
      if (g < 4 && sms != std::string::npos && end != std::string::npos) {
        const char* p = std::strstr(text.c_str() + sms, "[");
        while (p && p < text.c_str() + end) {
          ++p;
          while (p < text.c_str() + end && !isdigit((unsigned char)*p)) ++p;
          if (p >= text.c_str() + end) break;
          out.insert((unsigned)strtoul(p, (char**)&p, 10));
        }
      }
      pos = end == std::string::npos ? std::string::npos : end;
    }
  } else if (mask == "half_tpc") {
    const size_t tp = text.find("\"tpc_pairs\"");
    const size_t end = text.find("\"tpc_pairs_complete\"", tp);
    const char* p = text.c_str() + tp;
    const char* lim = text.c_str() + (end == std::string::npos ? text.size() : end);
    while (p && p < lim) {
      while (p < lim && !isdigit((unsigned char)*p)) ++p;
      if (p >= lim) break;
      const unsigned a = strtoul(p, (char**)&p, 10);
      while (p < lim && !isdigit((unsigned char)*p)) ++p;
      const unsigned b = strtoul(p, (char**)&p, 10);
      out.insert(std::min(a, b));
    }
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s waves <launches> <ctas_per_sm> <waves> <hold_us> | tail <launches> <hold_us> | tput <launches> <ctas_per_sm> <waves> | refill <launches> <ctas_per_sm> <hold_us> | blocked <launches> <ctas_per_sm> <waves> <none|gpc0123|half_tpc> <topology.json>\n", argv[0]);
    return 1;
  }
  const std::string mode = argv[1];
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int smem_optin = 0, smem_per_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  const int threads = 128;
  const int n_sm = prop.multiProcessorCount;

  int launches = 0, ctas_per_sm = 1, waves = 1;
  unsigned long long hold_us = 0;
  if (mode == "waves" && argc >= 6) {
    launches = atoi(argv[2]); ctas_per_sm = atoi(argv[3]); waves = atoi(argv[4]); hold_us = strtoull(argv[5], nullptr, 10);
  } else if (mode == "tput" && argc >= 5) {
    // waves mode with hold 0: CTA body = timestamps only; the dispatcher is
    // continuously backlogged once the first wave is resident.
    launches = atoi(argv[2]); ctas_per_sm = atoi(argv[3]); waves = atoi(argv[4]); hold_us = 0;
  } else if (mode == "refill" && argc >= 5) {
    launches = atoi(argv[2]); ctas_per_sm = atoi(argv[3]); waves = 2; hold_us = strtoull(argv[4], nullptr, 10);
  } else if (mode == "blocked" && argc >= 7) {
    launches = atoi(argv[2]); ctas_per_sm = atoi(argv[3]); waves = atoi(argv[4]); hold_us = 0;
  } else if (mode == "tail" && argc >= 4) {
    launches = atoi(argv[2]); hold_us = strtoull(argv[3], nullptr, 10);
  } else {
    fprintf(stderr, "bad arguments\n");
    return 1;
  }
  if (launches < 1 || ctas_per_sm < 1 || ctas_per_sm > 32 || waves < 1) { fprintf(stderr, "bad arguments\n"); return 1; }
  // smem request so that exactly ctas_per_sm CTAs fit on one SM: the per-SM
  // pool minus the 1 KiB/CTA reservation, divided by ctas_per_sm, rounded
  // down to 1 KiB; must not exceed the opt-in maximum.
  size_t smem = (size_t)((smem_per_sm - 1024 * ctas_per_sm) / ctas_per_sm) / 1024 * 1024;
  if (smem > (size_t)smem_optin) smem = (size_t)smem_optin;
  CUDA_CHECK(cudaFuncSetAttribute(hold_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CUDA_CHECK(cudaFuncSetAttribute(refill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  std::set<unsigned> blocked_set;
  if (mode == "blocked") {
    const std::string mask = argv[5];
    if (mask != "none" && mask != "gpc0123" && mask != "half_tpc") {
      fprintf(stderr, "bad mask\n"); return 1; }
    if (mask != "none") {
      std::ifstream f(argv[6]);
      if (!f) { fprintf(stderr, "cannot open topology %s\n", argv[6]); return 1; }
      std::stringstream ss; ss << f.rdbuf();
      blocked_set = topo_mask(ss.str(), mask);
      if (blocked_set.empty()) { fprintf(stderr, "empty mask set\n"); return 1; }
    }
    CUDA_CHECK(cudaFuncSetAttribute(blocker_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_optin));
  }
  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, hold_kernel, threads, smem));
  const int grid = mode == "tail" ? n_sm + 1 : n_sm * ctas_per_sm * waves;
  printf("META mode=%s gpu=%s sms=%d cc=%d.%d smem_per_sm=%d smem_optin=%d smem=%zu threads=%d "
         "ctas_per_sm=%d occupancy_api=%d waves=%d hold_us=%llu grid=%d launches=%d%s%s\n",
         mode.c_str(), prop.name, n_sm, prop.major, prop.minor, smem_per_sm, smem_optin, smem,
         threads, ctas_per_sm, occ, waves, hold_us, grid, launches,
         mode == "blocked" ? " mask=" : "",
         mode == "blocked" ? argv[5] : "");
  if (mode == "blocked")
    printf("META blocked_sms=%zu\n", blocked_set.size());
  if (occ != ctas_per_sm && mode != "tail") {
    printf("OCCMISMATCH requested=%d api=%d\n", ctas_per_sm, occ);
  }
  Rec* rec = nullptr;
  unsigned long long* first = nullptr;
  CUDA_CHECK(cudaMalloc(&rec, (size_t)grid * sizeof(Rec)));
  CUDA_CHECK(cudaMalloc(&first, sizeof(unsigned long long)));
  unsigned* dmask = nullptr;
  Rec* brec = nullptr;
  cudaStream_t sa = nullptr, sb = nullptr;
  std::vector<Rec> bhost;
  if (mode == "blocked") {
    unsigned hmask[5] = {0, 0, 0, 0, 0};
    for (unsigned s : blocked_set) hmask[s >> 5] |= 1u << (s & 31);
    CUDA_CHECK(cudaMalloc(&dmask, sizeof(hmask)));
    CUDA_CHECK(cudaMemcpy(dmask, hmask, sizeof(hmask), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&brec, (size_t)n_sm * sizeof(Rec)));
    bhost.resize(n_sm);
    CUDA_CHECK(cudaStreamCreate(&sa)); CUDA_CHECK(cudaStreamCreate(&sb));
  }
  std::vector<Rec> host(grid);
  for (int l = 0; l < launches; ++l) {
    CUDA_CHECK(cudaMemset(rec, 0xff, (size_t)grid * sizeof(Rec)));
    CUDA_CHECK(cudaMemset(first, 0xff, sizeof(unsigned long long)));
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto h0 = std::chrono::steady_clock::now();
    if (mode == "blocked") {
      CUDA_CHECK(cudaMemset(brec, 0xff, (size_t)n_sm * sizeof(Rec)));
      blocker_kernel<<<n_sm, threads, smem_optin, sa>>>(dmask, brec);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      hold_kernel<<<grid, threads, smem, sb>>>(rec, 0);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(sa)); CUDA_CHECK(cudaStreamSynchronize(sb));
    } else if (mode == "refill")
      refill_kernel<<<grid, threads, smem>>>(rec, hold_us * 1000ULL, first,
                                           n_sm * ctas_per_sm);
    else
      hold_kernel<<<grid, threads, smem>>>(rec, hold_us * 1000ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto h1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(host.data(), rec, (size_t)grid * sizeof(Rec), cudaMemcpyDeviceToHost));
    unsigned long long gmin = ~0ULL;
    for (auto& r : host) gmin = std::min(gmin, r.gt_in);
    for (int c = 0; c < grid; ++c) {
      const Rec& r = host[c];
      printf("CTA launch=%d cta=%d smid=%u warpid=%u gt_in=%llu clk_in=%llu gt_out=%llu clk_out=%llu\n",
             l, c, r.smid, r.warpid, r.gt_in - gmin, r.clk_in, r.gt_out - gmin, r.clk_out);
    }
    printf("LAUNCH launch=%d host_ns=%lld\n", l,
           (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(h1 - h0).count());
    if (mode == "blocked") {
      CUDA_CHECK(cudaMemcpy(bhost.data(), brec, (size_t)n_sm * sizeof(Rec),
                            cudaMemcpyDeviceToHost));
      for (int c = 0; c < n_sm; ++c) {
        const unsigned s = bhost[c].smid;
        if (s != 0xffffffffu && blocked_set.count(s))
          printf("BLOCK launch=%d cta=%d smid=%u\n", l, c, s);
      }
    }
  }
  CUDA_CHECK(cudaFree(rec));
  printf("DONE\n");
  return 0;
}
