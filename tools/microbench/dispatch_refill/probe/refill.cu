// tools/microbench/dispatch_refill/probe/refill.cu -- protocol.json of the
// same directory.
//
// One multi-wave launch per cell: grid = sm_count * occ * waves CTAs of
// <threads> threads.  Every CTA records %smid, %warpid of thread 0's warp and
// clock64 / %globaltimer at its first and last instruction, and stays resident
// for <hold> SM cycles measured from its own first clock read:
//   spin     every thread loops on clock64 until the hold has elapsed
//   sleep    every thread loops on nanosleep.u32 64 until the hold has elapsed
//   ldg      every thread first loads 16 B of its own from a buffer the host
//            evicted from L2 (CTA b reads bytes [b*threads*16, +threads*16)),
//            the hold length depends on the loaded value (zero), then spins
//   cpasync  as ldg, the 16 B go global -> shared with cp.async.cg and are
//            read back from shared memory after cp.async.wait_group 0
//   rms      (addendum 1) the shape of the FIF-RMSNORM kernel, 128 threads:
//            cp.async of 16 B per thread from the evicted buffer (2 KiB per
//            CTA), wait, read back, spin hold / 2, __syncthreads, the lanes
//            with (tid & 15) < 8 load 2 x 16 B of the CTA's own 2 KiB again
//            and 2 x 16 B of a 512 B buffer shared by all CTAs, spin hold / 2,
//            the same lanes store 1 B and 8 B to two output buffers
//            (576 B per CTA).  The cp.async is the L1-caching form
//            (cp.async.ca, LDGSTS without BYPASS) as in that kernel.
//            --grid N sets the grid directly; --rms-parts bit 0 = the second
//            loads, bit 1 = the stores (default 3).
// --pdl N (addendum 2, every mode): bit 0 = every thread executes
// griddepcontrol.wait right after the first clock read, bit 1 = every thread
// executes griddepcontrol.launch_dependents as its last instruction before
// exit (after the last clock read).  The launch itself carries no
// programmatic-dependent-launch attribute.
// Addendum 4: --pdl-cta P (permille, default 1000) only the CTAs whose index
// hashes below P execute launch_dependents; --pdl-warps W (default 0 = all)
// only the threads of the first W warps of such a CTA execute it; --pdl-early 1
// they execute it right after the first clock read instead of before exit;
// --pdl-attr 1 launches with cudaLaunchAttributeProgrammaticStreamSerialization.
// --mem-permille P (default 1000; modes ldg / cpasync) only the CTAs whose
// index hashes below P do the cold read, the others only hold.
// --jitter J (permille): the hold of CTA b is hold * (1 +- J/1000), uniform by
// a hash of b.
// Occupancy: with --occ 0 no extra shared memory is requested (the device's
// own limit applies and is reported); with --occ N the dynamic shared memory
// request is sized so that exactly N CTAs fit one SM.
//
// Output: META ..., CTA launch=.. cta=.. smid=.. warpid=.. gt_in=.. clk_in=..
// gt_out=.. clk_out=.., LAUNCH launch=.. host_ns=.., DONE.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

enum Mode : unsigned { kSpin = 0, kSleep = 1, kLdg = 2, kCpAsync = 3, kRms = 4 };

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
__device__ __forceinline__ void cp_async16(std::uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async16_ca(std::uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// bytes [0, 32): thread 0's marker; bytes [32, 32 + threads * 16): cp.async
// destination
extern __shared__ __align__(16) unsigned char dyn_smem[];

__device__ __forceinline__ unsigned hash32(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}

__global__ void cell_kernel(Rec* rec, const uint4* src, unsigned long long hold,
                            unsigned jitter_permille, unsigned mode, const uint4* wgt,
                            unsigned char* out8, unsigned long long* out64,
                            unsigned rms_parts, unsigned pdl, unsigned pdl_cta_permille,
                            unsigned pdl_warps, unsigned pdl_early, unsigned mem_permille) {
  const unsigned long long t0 = gtimer();
  const unsigned long long c0 = clock64();
  if (pdl & 1) asm volatile("griddepcontrol.wait;" ::: "memory");
  const bool pdl_thread = (pdl & 2) && (hash32(blockIdx.x ^ 0x9e3779b9u) % 1000u) < pdl_cta_permille &&
                          (pdl_warps == 0 || (threadIdx.x >> 5) < pdl_warps);
  if (pdl_thread && pdl_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const bool do_mem = (hash32(blockIdx.x + 0x51ed270bu) % 1000u) < mem_permille;
  const unsigned b = blockIdx.x;
  if (threadIdx.x == 0) {
    rec[b].smid = smid_reg();
    rec[b].warpid = warpid_reg();
    rec[b].gt_in = t0;
    rec[b].clk_in = c0;
    dyn_smem[0] = (unsigned char)t0;
  }
  unsigned long long h = hold;
  if (jitter_permille) {
    // u in [0, 2048): h = hold * (1000 - J + 2 J u / 2048) / 1000
    const unsigned long long u = hash32(b) & 2047u;
    h = hold * (1000ull - jitter_permille + (2ull * jitter_permille * u) / 2048ull) / 1000ull;
  }
  if (mode == kLdg && do_mem) {
    const uint4 v = src[(size_t)b * blockDim.x + threadIdx.x];
    h += (v.x ^ v.y ^ v.z ^ v.w);
  } else if (mode == kCpAsync && do_mem) {
    const std::uint32_t dst =
        (std::uint32_t)__cvta_generic_to_shared(dyn_smem + 32 + threadIdx.x * 16);
    cp_async16(dst, src + ((size_t)b * blockDim.x + threadIdx.x));
    cp_commit();
    cp_wait_all();
    const uint4 v = *reinterpret_cast<const uint4*>(dyn_smem + 32 + threadIdx.x * 16);
    h += (v.x ^ v.y ^ v.z ^ v.w);
  } else if (mode == kRms) {
    const unsigned tid = threadIdx.x;
    const std::uint32_t dst =
        (std::uint32_t)__cvta_generic_to_shared(dyn_smem + 32 + tid * 16);
    cp_async16_ca(dst, src + ((size_t)b * 128 + tid));
    cp_commit();
    cp_wait_all();
    const uint4 v = *reinterpret_cast<const uint4*>(dyn_smem + 32 + tid * 16);
    unsigned acc = v.x ^ v.y ^ v.z ^ v.w;
    const unsigned long long c1 = clock64();
    while (clock64() - c1 < (hold >> 1) + acc) { }
    __syncthreads();
    const bool lane = (tid & 15) < 8;
    if (lane && (rms_parts & 1)) {
      const uint4* p = src + ((size_t)b * 128 + (tid >> 4) * 16 + (tid & 15) * 2);
      const uint4 a0 = p[0], a1 = p[1];
      const uint4 w0 = wgt[(tid & 15) * 2], w1 = wgt[(tid & 15) * 2 + 1];
      acc ^= a0.x ^ a1.y ^ w0.z ^ w1.w;
    }
    const unsigned long long c2 = clock64();
    while (clock64() - c2 < (hold >> 1) + acc) { }
    if (lane && (rms_parts & 2)) {
      const size_t o = (size_t)b * 64 + (tid >> 4) * 8 + (tid & 15);
      out8[o] = (unsigned char)acc;
      out64[o] = acc;
    }
    h = 0;
  }
  if (mode == kSleep) {
    while (clock64() - c0 < h) { asm volatile("nanosleep.u32 64;"); }
  } else {
    while (clock64() - c0 < h) { }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    rec[b].gt_out = gtimer();
    rec[b].clk_out = clock64();
  }
  if (pdl_thread && !pdl_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

static const char* arg(int argc, char** argv, const char* name, const char* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], name)) return argv[i + 1];
  return dflt;
}

int main(int argc, char** argv) {
  const std::string mode_s = arg(argc, argv, "--mode", "spin");
  const int threads = atoi(arg(argc, argv, "--threads", "128"));
  const int occ_req = atoi(arg(argc, argv, "--occ", "0"));
  const int waves = atoi(arg(argc, argv, "--waves", "4"));
  const unsigned long long hold = strtoull(arg(argc, argv, "--hold", "0"), nullptr, 10);
  const unsigned jitter = (unsigned)atoi(arg(argc, argv, "--jitter", "0"));
  const int launches = atoi(arg(argc, argv, "--launches", "5"));
  const size_t evict_mib = strtoull(arg(argc, argv, "--evict-mib", "512"), nullptr, 10);
  const int grid_req = atoi(arg(argc, argv, "--grid", "0"));
  const unsigned rms_parts = (unsigned)atoi(arg(argc, argv, "--rms-parts", "3"));
  const unsigned pdl = (unsigned)atoi(arg(argc, argv, "--pdl", "0"));
  const unsigned pdl_cta = (unsigned)atoi(arg(argc, argv, "--pdl-cta", "1000"));
  const unsigned pdl_warps = (unsigned)atoi(arg(argc, argv, "--pdl-warps", "0"));
  const unsigned pdl_early = (unsigned)atoi(arg(argc, argv, "--pdl-early", "0"));
  const unsigned pdl_attr = (unsigned)atoi(arg(argc, argv, "--pdl-attr", "0"));
  const unsigned mem_permille = (unsigned)atoi(arg(argc, argv, "--mem-permille", "1000"));
  unsigned mode;
  if (mode_s == "spin") mode = kSpin;
  else if (mode_s == "sleep") mode = kSleep;
  else if (mode_s == "ldg") mode = kLdg;
  else if (mode_s == "cpasync") mode = kCpAsync;
  else if (mode_s == "rms") mode = kRms;
  else { fprintf(stderr, "bad --mode\n"); return 1; }
  if (threads < 32 || threads > 1024 || threads % 32 || occ_req < 0 || occ_req > 32 ||
      waves < 1 || launches < 1 || jitter >= 1000 || grid_req < 0 ||
      (mode == kRms && (threads != 128 || occ_req != 0))) {
    fprintf(stderr, "bad arguments\n");
    return 1;
  }
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int smem_optin = 0, smem_per_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  const int n_sm = prop.multiProcessorCount;
  // the cp.async destination needs 32 + threads * 16 bytes in every mode (one
  // binary, one shared layout); --occ N raises the request to the size that
  // lets exactly N CTAs share an SM
  size_t smem = 32 + (size_t)threads * 16;
  if (occ_req) {
    const size_t lim = (size_t)((smem_per_sm - 1024 * occ_req) / occ_req) / 1024 * 1024;
    if (lim < smem) { fprintf(stderr, "occ %d leaves less than %zu B\n", occ_req, smem); return 1; }
    smem = std::min(lim, (size_t)smem_optin);
  }
  CUDA_CHECK(cudaFuncSetAttribute(cell_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, cell_kernel, threads, smem));
  if (occ_req && occ != occ_req) printf("OCCMISMATCH requested=%d api=%d\n", occ_req, occ);
  const int grid = grid_req ? grid_req : n_sm * occ * waves;
  printf("META probe=dispatch_refill mode=%s gpu=%s sms=%d cc=%d.%d smem_per_sm=%d smem=%zu threads=%d "
         "occ_request=%d occupancy_api=%d waves=%d hold=%llu jitter_permille=%u grid=%d launches=%d evict_mib=%zu rms_parts=%u pdl=%u pdl_cta=%u pdl_warps=%u pdl_early=%u pdl_attr=%u mem_permille=%u\n",
         mode_s.c_str(), prop.name, n_sm, prop.major, prop.minor, smem_per_sm, smem, threads,
         occ_req, occ, waves, hold, jitter, grid, launches, evict_mib, rms_parts, pdl, pdl_cta, pdl_warps, pdl_early, pdl_attr, mem_permille);
  Rec* rec = nullptr;
  CUDA_CHECK(cudaMalloc(&rec, (size_t)grid * sizeof(Rec)));
  uint4* src = nullptr;
  unsigned char* evict = nullptr;
  const bool mem = mode == kLdg || mode == kCpAsync || mode == kRms;
  const size_t src_bytes = (size_t)grid * threads * sizeof(uint4);
  if (mem) {
    CUDA_CHECK(cudaMalloc(&src, src_bytes));
    CUDA_CHECK(cudaMemset(src, 0, src_bytes));
    CUDA_CHECK(cudaMalloc(&evict, evict_mib << 20));
  }
  uint4* wgt = nullptr;
  unsigned char* out8 = nullptr;
  unsigned long long* out64 = nullptr;
  if (mode == kRms) {
    CUDA_CHECK(cudaMalloc(&wgt, 32 * sizeof(uint4)));
    CUDA_CHECK(cudaMemset(wgt, 0, 32 * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&out8, (size_t)grid * 64));
    CUDA_CHECK(cudaMalloc(&out64, (size_t)grid * 64 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(out8, 0, (size_t)grid * 64));
    CUDA_CHECK(cudaMemset(out64, 0, (size_t)grid * 64 * sizeof(unsigned long long)));
  }
  std::vector<Rec> host(grid);
  for (int l = 0; l < launches; ++l) {
    CUDA_CHECK(cudaMemset(rec, 0xff, (size_t)grid * sizeof(Rec)));
    // push the source (and rms output / weight) buffers out of L2; the rec
    // buffer was written just before and is pushed out as well
    if (mem) CUDA_CHECK(cudaMemset(evict, l & 0xff, evict_mib << 20));
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto h0 = std::chrono::steady_clock::now();
    if (pdl_attr) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(grid);
      cfg.blockDim = dim3(threads);
      cfg.dynamicSmemBytes = smem;
      cfg.stream = 0;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      CUDA_CHECK(cudaLaunchKernelEx(&cfg, cell_kernel, rec, (const uint4*)src, hold, jitter, mode,
                                    (const uint4*)wgt, out8, out64, rms_parts, pdl, pdl_cta, pdl_warps,
                                    pdl_early, mem_permille));
    } else {
      cell_kernel<<<grid, threads, smem>>>(rec, src, hold, jitter, mode, wgt, out8, out64, rms_parts, pdl,
                                           pdl_cta, pdl_warps, pdl_early, mem_permille);
    }
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
  }
  CUDA_CHECK(cudaFree(rec));
  if (src) CUDA_CHECK(cudaFree(src));
  if (evict) CUDA_CHECK(cudaFree(evict));
  if (wgt) CUDA_CHECK(cudaFree(wgt));
  if (out8) CUDA_CHECK(cudaFree(out8));
  if (out64) CUDA_CHECK(cudaFree(out64));
  printf("DONE\n");
  return 0;
}
