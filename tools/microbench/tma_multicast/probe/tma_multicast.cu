// tma_multicast probe (protocol.json v1): is a multicast TMA box charged
// per destination CTA on the delivery / shared read path.  Grid = N CTAs
// (N even), one per SM (135 KB dynamic smem), launched with
// cluster dimension (2,1,1); 256 threads.  Slot s of a cluster is issued
// by rank s % 2 (mc_both) or always by rank 0 (mc_one); the multicast
// load (ctaMask 0b11) drops the same smem offset's box + mbarrier in both
// CTAs.  Per round the receiver arms its own mbarrier (expect_tx) then
// signals the issuer through a shared::cluster mbarrier arrive; the
// issuer waits that handshake, arms its own mbarrier, issues, and spins
// on test_wait.  plain does the identical handshake but a non-multicast
// load into the issuer's own CTA; plain_nohs skips the handshake (=
// tma_loaded, every CTA issues all its slots).
#include "../../tma_cold/mb.cuh"

#include <cuda.h>
#include <dlfcn.h>

#include <algorithm>
#include <string>
#include <vector>

constexpr int kInner = 64;
constexpr int kBoxRows = 128;
constexpr int kBoxBytes = kInner * 2 * kBoxRows;
constexpr int kMaxSlots = 8;
constexpr int kSmemBytes = kMaxSlots * kBoxBytes + 4096;
constexpr int kLatSamples = 512;

__device__ __forceinline__ void tma_load(std::uint32_t dst, const CUtensorMap* map, int x, int y, std::uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(mbar)
      : "memory");
}
__device__ __forceinline__ void tma_load_mc(std::uint32_t dst, const CUtensorMap* map, int x, int y,
                                            std::uint32_t mbar, std::uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(mbar), "h"(mask)
      : "memory");
}
__device__ __forceinline__ void expect_tx(std::uint32_t mbar, std::uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}
// One arrive on `peer`'s copy of the smem word at local address `a`.
__device__ __forceinline__ void hs_arrive(std::uint32_t a, std::uint32_t peer) {
  std::uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(peer));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(r) : "memory");
}

struct CtaOut {
  unsigned long long bytes_issued;
  unsigned long long bytes_recv;
  long long c0, c1, g0, g1;
  unsigned smid;
  unsigned check_fail;
  unsigned lat[kLatSamples];
};

// MODE: 1 plain (handshake, own-CTA load), 2 mc_both, 3 mc_one.
// MODE 0 plain_nohs shares this kernel only for its no-handshake path.
template <int MODE>
__global__ void __launch_bounds__(256, 1) k_mcast(const __grid_constant__ CUtensorMap map, int rows,
                                                  int slots, int rounds, int start_stride,
                                                  unsigned* arrived, CtaOut* out,
                                                  unsigned nparticipants) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) std::uint64_t mbar[kMaxSlots];
  __shared__ __align__(8) std::uint64_t hs[kMaxSlots];
  __shared__ unsigned long long cta_issued, cta_recv;
  __shared__ int cta_lat_n;
  const std::uint32_t smem_base = mb_smem_u32(smem);
  const std::uint32_t box_base = (smem_base + 1023u) & ~1023u;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const std::uint32_t rank = MODE == 0 ? 0 : mb_ctarank();
  const int idx = blockIdx.x;
  if (threadIdx.x == 0) {
    for (int s = 0; s < slots; ++s) {
      mb_mbar_init(mb_smem_u32(&mbar[s]), 1);
      mb_mbar_init(mb_smem_u32(&hs[s]), 1);
    }
    mb_fence_mbar_init();
    cta_issued = 0; cta_recv = 0; cta_lat_n = 0;
  }
  __syncthreads();
  if (MODE != 0) mb_cluster_sync();   // mbarrier init visible cluster-wide
  if (threadIdx.x == 0) {
    atomicAdd(arrived, 1u);
    while (*((volatile unsigned*)arrived) < nparticipants) { }
  }
  __syncthreads();
  CtaOut& o = out[idx];
  const long long g0 = (long long)mb_gtimer();
  const long long c0 = (long long)mb_clock();
  const int region_rows = start_stride;
  const int row0 = (idx * region_rows) % (rows - region_rows + 1);
  const int boxes = region_rows / kBoxRows;
  const int my_rounds = rounds / slots;
  unsigned lat[kLatSamples / kMaxSlots];
  int nlat = 0;
  unsigned long long issued = 0, recv = 0;

  // issuer of slot s: plain_nohs -> every CTA; mc_one -> rank 0;
  // plain / mc_both -> rank s % 2.
  const bool issuer = (MODE == 0) ? true
                      : (MODE == 3) ? (rank == 0) : (int)(warp % 2) == (int)rank;
  const bool receiver = (MODE == 0) ? false
                      : (MODE == 3) ? (rank == 1) : !issuer;
  const std::uint32_t peer = rank ^ 1u;   // cluster of 2

  if (warp < slots && lane == 0) {
    int b = warp % boxes;
    const std::uint32_t mb = mb_smem_u32(&mbar[warp]);
    const std::uint32_t h = mb_smem_u32(&hs[warp]);
    const std::uint32_t dst = box_base + warp * kBoxBytes;
    unsigned phase = 0;
    if (receiver) {
      for (int r = 0; r < my_rounds; ++r) {
        if (MODE == 1) {
          // handshake only; nothing is delivered, so the issuer acks on
          // the receiver's own hs[w] to keep the two in lock-step (a free
          // running receiver would advance the phase counter past the
          // issuer's parity wait and deadlock it).
          hs_arrive(h, peer);
          mb_mbar_spin_to(h, phase);
          phase ^= 1u;
        } else {
          expect_tx(mb, kBoxBytes);        // arm before the issuer may send
          hs_arrive(h, peer);              // tell the issuer the slot is armed
          mb_mbar_spin_to(mb, phase);
          phase ^= 1u;
          recv += kBoxBytes;
        }
      }
    } else {
      for (int r = 0; r < my_rounds; ++r) {
        if (MODE != 0) {
          mb_mbar_spin_to(h, phase);       // peer armed -> may issue
        }
        expect_tx(mb, kBoxBytes);
        const long long t0 = (long long)mb_clock();
        if (MODE == 1)
          tma_load(dst, &map, 0, row0 + b * kBoxRows, mb);
        else if (MODE != 0)
          tma_load_mc(dst, &map, 0, row0 + b * kBoxRows, mb, 3u);
        else
          tma_load(dst, &map, 0, row0 + b * kBoxRows, mb);
        mb_mbar_spin_to(mb, phase);
        const long long t1 = (long long)mb_clock();
        if (nlat < kLatSamples / kMaxSlots) lat[nlat++] = (unsigned)(t1 - t0);
        phase ^= 1u;
        if (MODE == 1) hs_arrive(h, peer);  // ack the receiver's handshake
        issued += kBoxBytes;
        recv += kBoxBytes;                 // issuer always lands its own box
        b += slots; if (b >= boxes) b = warp % boxes;
      }
    }
    atomicAdd(&cta_issued, issued);
    atomicAdd(&cta_recv, recv);
    if (nlat) {
      const int base = atomicAdd(&cta_lat_n, nlat);
      for (int i = 0; i < nlat && base + i < kLatSamples; ++i) o.lat[base + i] = lat[i];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const long long c1 = (long long)mb_clock();
    const long long g1 = (long long)mb_gtimer();
    o.bytes_issued = cta_issued; o.bytes_recv = cta_recv;
    o.c0 = c0; o.c1 = c1; o.g0 = g0; o.g1 = g1; o.smid = mb_smid();
    o.check_fail = 0;
    // controls.data: a receiving CTA checks the first 64 B of a slot it
    // did not issue (host pattern byte 0x01 -> sum 64).  plain receivers
    // get no box; only the multicast modes have receiving slots.
    if (MODE == 2 || MODE == 3) {
      int s = -1;
      for (int t = 0; t < slots; ++t) {
        const bool t_issued = (MODE == 3) ? (rank == 0) : (t % 2) == (int)rank;
        if (!t_issued) { s = t; break; }
      }
      if (s >= 0) {
        const unsigned char* box = smem + ((box_base - smem_base) + s * kBoxBytes);
        unsigned sum = 0;
        for (int i = 0; i < 64; ++i) sum += box[i];
        if (sum != 64u) o.check_fail = 1;
      }
    }
    for (int i = cta_lat_n; i < kLatSamples; ++i) o.lat[i] = 0;
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encoder() {
  void* sym = dlsym(RTLD_DEFAULT, "cuTensorMapEncodeTiled");
  if (!sym) sym = dlsym(dlopen("libcuda.so.1", RTLD_NOW), "cuTensorMapEncodeTiled");
  return reinterpret_cast<EncodeTiledFn>(sym);
}
static CUtensorMap make_map(void* g, int rows) {
  CUtensorMap m{};
  const cuuint64_t dim[2] = {static_cast<cuuint64_t>(kInner), static_cast<cuuint64_t>(rows)};
  const cuuint64_t stride[1] = {static_cast<cuuint64_t>(kInner * 2)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(kInner), static_cast<cuuint32_t>(kBoxRows)};
  const cuuint32_t es[2] = {1u, 1u};
  auto fn = encoder();
  if (!fn || fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, dim, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
    std::exit(2);
  }
  return m;
}

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}
static std::string arr(const std::vector<double>& v, const char* fmt = "%.3f") {
  std::string s = "[";
  char b[64];
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::snprintf(b, sizeof(b), fmt, v[i]);
    s += b;
    if (i + 1 < v.size()) s += ",";
  }
  return s + "]";
}

int main(int argc, char** argv) {
  std::string out = "tma_multicast.jsonl", mode = "plain";
  int N = 2, S = 4, rounds = 2048, samples = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    if (a == "--out" && i + 1 < argc) { out = argv[++i]; continue; }
    if (a == "--mode" && i + 1 < argc) { mode = argv[++i]; continue; }
    if (n("--ctas", &N) || n("--slots", &S) || n("--rounds", &rounds) || n("--samples", &samples)) continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  if (S < 1 || S > kMaxSlots || S % 2) { std::fprintf(stderr, "slots 2..%d even\n", kMaxSlots); return 2; }
  int m = -1;
  if (mode == "plain_nohs") m = 0; else if (mode == "plain") m = 1;
  else if (mode == "mc_both") m = 2; else if (mode == "mc_one") m = 3;
  else { std::fprintf(stderr, "mode plain|plain_nohs|mc_both|mc_one\n"); return 2; }
  const MbDevice dev = mb_query_device();
  if (N <= 0) N = dev.sm_count & ~1;          // all SMs, even count
  if (N > dev.sm_count || N % 2) { std::fprintf(stderr, "ctas even, <= %d\n", dev.sm_count & ~1); return 2; }
  const int rows = static_cast<int>((24ull << 20) / (kInner * 2));
  void* g = nullptr;
  MB_CHECK(cudaMalloc(&g, static_cast<std::size_t>(rows) * kInner * 2));
  MB_CHECK(cudaMemset(g, 1, static_cast<std::size_t>(rows) * kInner * 2));
  const CUtensorMap map = make_map(g, rows);
  unsigned* arrived = nullptr;
  CtaOut* d_out = nullptr;
  MB_CHECK(cudaMalloc(&arrived, sizeof(unsigned)));
  MB_CHECK(cudaMalloc(&d_out, sizeof(CtaOut) * N));
  for (auto k : {k_mcast<0>, k_mcast<1>, k_mcast<2>, k_mcast<3>})
    MB_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  int start_stride = (rows / N) / kBoxRows * kBoxRows;
  if (start_stride < S * kBoxRows) start_stride = S * kBoxRows;
  std::vector<double> iss_bpc, rcv_bpc, lat_all, agg_bns;
  std::string err;
  for (int rep = 0; rep < samples + 1; ++rep) {
    MB_CHECK(cudaMemset(arrived, 0, sizeof(unsigned)));
    MB_CHECK(cudaMemset(d_out, 0, sizeof(CtaOut) * N));
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(N, 1, 1); cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = kSmemBytes;
    cfg.attrs = at; cfg.numAttrs = (m == 0) ? 0 : 1;
    cudaError_t e;
    switch (m) {
      case 0: e = cudaLaunchKernelEx(&cfg, k_mcast<0>, map, rows, S, rounds, start_stride, arrived, d_out, (unsigned)N); break;
      case 1: e = cudaLaunchKernelEx(&cfg, k_mcast<1>, map, rows, S, rounds, start_stride, arrived, d_out, (unsigned)N); break;
      case 2: e = cudaLaunchKernelEx(&cfg, k_mcast<2>, map, rows, S, rounds, start_stride, arrived, d_out, (unsigned)N); break;
      default: e = cudaLaunchKernelEx(&cfg, k_mcast<3>, map, rows, S, rounds, start_stride, arrived, d_out, (unsigned)N); break;
    }
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    if (mb_take_err()) { err = "mbarrier wait timeout"; break; }
    if (rep == 0) continue;
    std::vector<CtaOut> h(N);
    MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(CtaOut) * N, cudaMemcpyDeviceToHost));
    double total_bytes = 0, gmin = 1e300, gmax = 0;
    for (int i = 0; i < N; ++i) {
      const double cyc = static_cast<double>(h[i].c1 - h[i].c0);
      if (h[i].bytes_issued) iss_bpc.push_back(h[i].bytes_issued / cyc);
      if (h[i].bytes_recv) rcv_bpc.push_back(h[i].bytes_recv / cyc);
      total_bytes += static_cast<double>(h[i].bytes_recv);
      gmin = std::min(gmin, static_cast<double>(h[i].g0));
      gmax = std::max(gmax, static_cast<double>(h[i].g1));
      for (int k = 0; k < kLatSamples; ++k) if (h[i].lat[k]) lat_all.push_back(h[i].lat[k]);
      if (h[i].check_fail && err.empty()) err = "data check failed";
    }
    agg_bns.push_back(total_bytes / (gmax - gmin));
  }
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str()); return 1; }
  const double med_i = pct(iss_bpc, 0.5), med_r = pct(rcv_bpc, 0.5);
  std::fprintf(f,
               "{\"schema\":\"blackwell.mechanism_curve_point.v1\",\"family\":\"tma_multicast.%s.N%d.S%d\","
               "\"quantity\":\"throughput\",\"cta_group\":1,\"x_name\":\"ctas\",\"x\":%d,"
               "\"y_name\":\"bytes_per_cycle_per_cta\",\"y_unit\":\"bytes/cycle\","
               "\"median\":%.4f,\"p10\":%.4f,\"p90\":%.4f,\"n\":%zu,\"holdout\":false,"
               "\"ctas\":%d,\"slots\":%d,\"rounds\":%d,\"box_bytes\":%d,"
               "\"issued_bpc_median\":%.4f,\"received_bpc_median\":%.4f,"
               "\"issued_boxes_per_1k\":%.3f,\"received_boxes_per_1k\":%.3f,"
               "\"aggregate_bytes_per_ns_median\":%.3f,"
               "\"latency_p10\":%.1f,\"latency_p50\":%.1f,\"latency_p90\":%.1f,\"latency_n\":%zu,"
               "\"source\":\"tma_multicast.measured\",\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,\"sm_count\":%d",
               mode.c_str(), N, S, N, med_i, pct(iss_bpc, 0.10), pct(iss_bpc, 0.90), iss_bpc.size(),
               N, S, rounds, kBoxBytes, med_i, med_r,
               med_i * 1000.0 / kBoxBytes, med_r * 1000.0 / kBoxBytes,
               pct(agg_bns, 0.5),
               pct(lat_all, 0.10), pct(lat_all, 0.50), pct(lat_all, 0.90), lat_all.size(),
               dev.name.c_str(), dev.sm_clock_khz, dev.sm_count);
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("tma_multicast %s N=%d S=%d issued/1k=%.3f recv/1k=%.3f lat p50=%.0f%s\n",
              mode.c_str(), N, S, med_i * 1000.0 / kBoxBytes, med_r * 1000.0 / kBoxBytes,
              pct(lat_all, 0.5), err.empty() ? "" : " ERR");
  cudaFree(g); cudaFree(arrived); cudaFree(d_out);
  return err.empty() ? 0 : 1;
}
