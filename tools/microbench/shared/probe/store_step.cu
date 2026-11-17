// shared protocol-v3 probe: the shared-store step of an epilogue round.
//
// One CTA of 128 threads (4 warps) on one SM.  Every warp runs `rounds`
// rounds of
//     [K x st.shared by the warps selected by `mask`]  TAIL
// with no clock read inside the round; each warp reads clock64 once before
// and once after its loop, the host reports the per-warp round period.
//
//   WIDE 1: st.shared.v4.b32 (STS.128, 16 B per lane)   WIDE 0: st.shared.b32
//   store k of a thread goes to
//       base + warp * 8192 + warp * wshift + lane * stride + k * kstep
//   stride 64, kstep 16, wshift 0 is the GEMM epilogue pattern of
//   tma_store_loaded/probe/store_round.cu (16 lanes per bank group = 16
//   wavefronts per STS.128, the K stores of a thread on disjoint banks, all
//   warps on the same banks).
//   TAIL 0: nothing
//        1: bar.sync 1, 128
//        2: bar.sync ; bar.sync
//        3: FILL
//        4: FILL ; bar.sync ; bar.sync
//        5: bar.sync ; bar.sync ; FILL
//        6: fence.proxy.async.shared::cta ; bar.sync ; bar.sync
//   FILL = NFILL dependent mad.lo.u32 on one register that is carried from
//   round to round (a serial chain: it cannot overlap itself, it can overlap
//   anything that does not hold the warp).
// Warps outside `mask` run the same code with the stores predicated off (or
// branched over; the SASS gate records which) and take part in every barrier.
#include "../../tma_cold/mb.cuh"

#include <string>
#include <vector>

constexpr int kWarpRegion = 8192;
constexpr int kSmemBytes = 4 * kWarpRegion + 2048;

__device__ __forceinline__ void st_v4(std::uint32_t a, std::uint32_t v) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_v2(std::uint32_t a, std::uint32_t v) {
  asm volatile("st.shared.v2.b32 [%0], {%1, %1};" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_b32(std::uint32_t a, std::uint32_t v) {
  asm volatile("st.shared.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void bar_sync_round() { asm volatile("bar.sync 1, 128;" ::: "memory"); }
__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ std::uint32_t fill_step(std::uint32_t x, std::uint32_t c) {
  std::uint32_t y;
  asm volatile("mad.lo.u32 %0, %1, %2, %2;" : "=r"(y) : "r"(x), "r"(c));
  return y;
}

struct StepOut {
  long long c0[4], c1[4];
  unsigned x[4];
  unsigned smid, pad;
};

template <int WIDE, int K, int TAIL, int NFILL>
__global__ void __launch_bounds__(128, 1) k_step(int mask, int stride, int kstep, int wshift, int rounds,
                                                 std::uint32_t fillc, StepOut* out) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const std::uint32_t base = (mb_smem_u32(smem) + 1023u) & ~1023u;
  const std::uint32_t a0 = base + warp * kWarpRegion + warp * wshift + lane * stride;
  std::uint32_t a[K];
#pragma unroll
  for (int k = 0; k < K; ++k) a[k] = a0 + k * kstep;
  // bits 0-3: writer warps; bits 8+: number of leading lanes that store (0 = all 32; protocol-v5 addendum 'width')
  const int store_lanes = (mask >> 8) ? (mask >> 8) : 32;
  const bool writer = ((mask >> warp) & 1) != 0 && lane < store_lanes;
  const std::uint32_t v = threadIdx.x;
  std::uint32_t x = threadIdx.x + 1u;
  __syncthreads();
  const long long c0 = static_cast<long long>(mb_clock());
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    if (writer) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        if (WIDE == 1) st_v4(a[k], v);
        else if (WIDE == 2) st_v2(a[k], v);  // 64-bit store (protocol-v5 addendum)
        else st_b32(a[k], v);
      }
    }
    if (TAIL == 3 || TAIL == 4) {
#pragma unroll
      for (int i = 0; i < NFILL; ++i) x = fill_step(x, fillc);
    }
    if (TAIL == 6) fence_async_shared();
    if (TAIL == 1 || TAIL == 2 || TAIL == 4 || TAIL == 5 || TAIL == 6) bar_sync_round();
    if (TAIL == 2 || TAIL == 4 || TAIL == 5 || TAIL == 6) bar_sync_round();
    if (TAIL == 5) {
#pragma unroll
      for (int i = 0; i < NFILL; ++i) x = fill_step(x, fillc);
    }
  }
  const long long c1 = static_cast<long long>(mb_clock());
  if (lane == 0) {
    out->c0[warp] = c0;
    out->c1[warp] = c1;
    out->x[warp] = x;
  }
  if (threadIdx.x == 0) out->smid = mb_smid();
}

// (WIDE, K, TAIL, NFILL) instantiations of protocol-v3
#define STEP_KERNELS(X)                                                                       \
  X(1, 1, 0, 0) X(1, 2, 0, 0) X(1, 4, 0, 0) X(1, 8, 0, 0)                                      \
  X(1, 1, 1, 0) X(1, 2, 1, 0) X(1, 4, 1, 0) X(1, 8, 1, 0)                                      \
  X(1, 1, 2, 0) X(1, 2, 2, 0) X(1, 4, 2, 0) X(1, 8, 2, 0)                                      \
  X(1, 1, 6, 0) X(1, 2, 6, 0) X(1, 4, 6, 0) X(1, 8, 6, 0)                                      \
  X(1, 1, 3, 8) X(1, 2, 3, 8) X(1, 4, 3, 8) X(1, 8, 3, 8)                                      \
  X(1, 1, 3, 24) X(1, 2, 3, 24) X(1, 4, 3, 24) X(1, 8, 3, 24)                                  \
  X(1, 1, 3, 64) X(1, 2, 3, 64) X(1, 4, 3, 64) X(1, 8, 3, 64)                                  \
  X(1, 4, 4, 24) X(1, 4, 4, 64) X(1, 4, 5, 24) X(1, 4, 5, 64)                                  \
  X(0, 4, 0, 0) X(0, 8, 0, 0) X(0, 4, 2, 0) X(0, 8, 2, 0)                                      \
  X(2, 4, 0, 0) X(2, 8, 0, 0)

static bool launch(int wide, int k, int tail, int nfill, int mask, int stride, int kstep, int wshift, int rounds,
                   std::uint32_t fillc, StepOut* out) {
#define X(W, KK, T, N)                                                                                     \
  if (wide == W && k == KK && tail == T && nfill == N) {                                                   \
    MB_CHECK(cudaFuncSetAttribute(k_step<W, KK, T, N>, cudaFuncAttributeMaxDynamicSharedMemorySize,        \
                                  kSmemBytes));                                                            \
    k_step<W, KK, T, N><<<1, 128, kSmemBytes>>>(mask, stride, kstep, wshift, rounds, fillc, out);          \
    return true;                                                                                           \
  }
  STEP_KERNELS(X)
#undef X
  return false;
}

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

int main(int argc, char** argv) {
  std::string out = "store_step.jsonl";
  int wide = 1, k = 4, tail = 2, nfill = 0, mask = 15, stride = 64, kstep = 16, wshift = 0, rounds = 2048,
      samples = 8, lanes = 32;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* key, int* d) {
      if (a == key && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; }
      return false;
    };
    if (a == "--out" && i + 1 < argc) { out = argv[++i]; continue; }
    if (n("--wide", &wide) || n("--k", &k) || n("--tail", &tail) || n("--nfill", &nfill) || n("--mask", &mask) ||
        n("--stride", &stride) || n("--kstep", &kstep) || n("--wshift", &wshift) || n("--rounds", &rounds) ||
        n("--samples", &samples) || n("--lanes", &lanes))
      continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  const int span = 31 * stride + (k - 1) * kstep + 3 * wshift + 16;
  if (mask < 0 || mask > 15 || stride < 0 || kstep < 0 || wshift < 0 || stride % 4 || kstep % 4 || wshift % 4 ||
      span > kWarpRegion || rounds < 16 || samples < 1 || lanes < 1 || lanes > 32) {
    std::fprintf(stderr, "bad cell\n");
    return 2;
  }
  const MbDevice dev = mb_query_device();
  StepOut* dout = nullptr;
  MB_CHECK(cudaMalloc(&dout, sizeof(StepOut)));
  std::vector<double> period[4];
  std::string err;
  unsigned smid = 0;
  for (int rep = 0; rep < samples + 1 && err.empty(); ++rep) {  // rep 0 discarded
    MB_CHECK(cudaMemset(dout, 0, sizeof(StepOut)));
    if (!launch(wide, k, tail, nfill, lanes == 32 ? mask : (mask | (lanes << 8)), stride, kstep, wshift, rounds,
                0x9E3779B1u, dout)) {
      std::fprintf(stderr, "no kernel for wide=%d k=%d tail=%d nfill=%d\n", wide, k, tail, nfill);
      return 2;
    }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    if (rep == 0) continue;
    StepOut h{};
    MB_CHECK(cudaMemcpy(&h, dout, sizeof(StepOut), cudaMemcpyDeviceToHost));
    smid = h.smid;
    for (int w = 0; w < 4; ++w)
      period[w].push_back(static_cast<double>(h.c1[w] - h.c0[w]) / static_cast<double>(rounds));
  }
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str()); return 1; }
  char fam[200];
  std::snprintf(fam, sizeof(fam), "store_step.w%d.k%d.t%d.n%d.m%d.s%d.ks%d.ws%d", wide, k, tail, nfill, mask, stride,
                kstep, wshift);
  std::fprintf(f,
               "{\"schema\":\"cyclescope.microbench.store_step.v1\",\"family\":\"%s\",\"wide\":%d,\"k\":%d,\"tail\":%d,"
               "\"nfill\":%d,\"mask\":%d,\"stride\":%d,\"kstep\":%d,\"wshift\":%d,\"rounds\":%d,\"samples\":%d,"
               "\"period_median\":[%.3f,%.3f,%.3f,%.3f],\"period_p10\":[%.3f,%.3f,%.3f,%.3f],"
               "\"period_p90\":[%.3f,%.3f,%.3f,%.3f],\"smid\":%u,\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,"
               "\"sm_count\":%d",
               fam, wide, k, tail, nfill, mask, stride, kstep, wshift, rounds, samples, pct(period[0], 0.5),
               pct(period[1], 0.5), pct(period[2], 0.5), pct(period[3], 0.5), pct(period[0], 0.1), pct(period[1], 0.1),
               pct(period[2], 0.1), pct(period[3], 0.1), pct(period[0], 0.9), pct(period[1], 0.9), pct(period[2], 0.9),
               pct(period[3], 0.9), smid, dev.name.c_str(), dev.sm_clock_khz, dev.sm_count);
  std::fprintf(f, ",\"lanes\":%d", lanes);
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("%s period=%.1f/%.1f/%.1f/%.1f%s\n", fam, pct(period[0], 0.5), pct(period[1], 0.5), pct(period[2], 0.5),
              pct(period[3], 0.5), err.empty() ? "" : " ERR");
  cudaFree(dout);
  return err.empty() ? 0 : 1;
}
