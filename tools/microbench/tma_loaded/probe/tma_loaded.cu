// tma_loaded probe: TMA tile-load delivery (bytes / cycle per SM and
// aggregate) and completion latency when N SMs run TMA pipelines at once.
//
// Grid = N CTAs, one per SM (dynamic smem 135 KB), 256 threads each.
// Warp s (s < S) lane 0 owns slot s: a 16 KB SWIZZLE_128B smem box and one
// mbarrier; it issues one UTMALDG (cp.async.bulk.tensor.2d, bf16 64 x 128
// box), spins on mbarrier.test_wait until complete_tx lands, records
// issue -> complete clock64, re-issues the next box; R / S rounds per warp
// so S boxes are in flight per CTA at all times.  Boxes walk a hot tensor shared by all CTAs (rows
// = 16 MiB / 128 B, warmed by a first launch so every box is an L2 hit; the
// each CTA cycles inside its own region of rows / N rows, >= S boxes).
// All CTAs meet on a global arrival counter before timing.  Per CTA: bytes,
// clock64 window, globaltimer window, first 512 latencies.
//
// jsonl row per cell: family tma_loaded.N<N>.S<S>, samples = per-CTA
// bytes/cycle, extra: aggregate B/ns, latency p50/p90 (all CTAs pooled),
// per-CTA elapsed.
#include "../../tma_cold/mb.cuh"

#include <cuda.h>
#include <dlfcn.h>

#include <algorithm>
#include <string>
#include <vector>

constexpr int kInner = 64;       // bf16 per box row = 128 B
constexpr int kBoxRows = 128;    // 16 KB box
constexpr int kBoxBytes = kInner * 2 * kBoxRows;
constexpr int kMaxSlots = 8;
constexpr int kSmemBytes = kMaxSlots * kBoxBytes + 4096;  // 135168 -> 1 CTA / SM
constexpr int kLatSamples = 512;

__device__ __forceinline__ void tma_load(std::uint32_t dst, const CUtensorMap* map, int x, int y, std::uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(mbar)
      : "memory");
}
__device__ __forceinline__ void expect_tx(std::uint32_t mbar, std::uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}

struct CtaOut {
  unsigned long long bytes;
  long long c0, c1, g0, g1;
  unsigned smid;
  unsigned lat[kLatSamples];
};

__global__ void __launch_bounds__(256, 1) k_tma_loaded(const __grid_constant__ CUtensorMap map, int rows,
                                                       int slots, int rounds, int start_stride,
                                                       unsigned* arrived, CtaOut* out,
                                                       int sel0, int sel1, unsigned nparticipants) {
  // v2: one warp per slot (lane 0 issues, waits with test_wait spin, re-issues);
  // the v1 single-thread handler serialised completions on the try_wait
  // suspend grid (~400 cycles per box) and measured that, not the TMA path.
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) std::uint64_t mbar[kMaxSlots];
  __shared__ unsigned long long cta_bytes;
  __shared__ int cta_lat_n;
  const std::uint32_t smem_base = mb_smem_u32(smem);
  const std::uint32_t box_base = (smem_base + 1023u) & ~1023u;  // SWIZZLE_128B: 1024 B aligned
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  // pair mode (sel0 >= 0): only the CTAs that landed on SM sel0 / sel1 take
  // part (TPC sharing question); every other CTA records nothing and exits
  __shared__ int my_slot_idx;
  if (sel0 >= 0) {
    if (threadIdx.x == 0) {
      const int sm = (int)mb_smid();
      my_slot_idx = (sm == sel0) ? 0 : (sm == sel1) ? 1 : -1;
    }
    __syncthreads();
    if (my_slot_idx < 0) return;
  } else if (threadIdx.x == 0) {
    my_slot_idx = (int)blockIdx.x;
  }
  __syncthreads();
  const int idx = my_slot_idx;
  if (threadIdx.x == 0) {
    for (int s = 0; s < slots; ++s) mb_mbar_init(mb_smem_u32(&mbar[s]), 1);
    mb_fence_mbar_init();
    cta_bytes = 0; cta_lat_n = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {  // all CTAs resident before any timed TMA
    atomicAdd(arrived, 1u);
    while (*((volatile unsigned*)arrived) < nparticipants) { }
  }
  __syncthreads();
  CtaOut& o = out[idx];
  const long long g0 = (long long)mb_gtimer();
  const long long c0 = (long long)mb_clock();
  if (warp < slots && lane == 0) {
    // this CTA's region (>= slots boxes); warp s walks boxes s, s+slots, ...
    const int region_rows = start_stride;
    // shared mode (start_stride < 0): every CTA walks the same boxes from row 0
    const int row0 = region_rows < 0 ? 0
        : (idx * region_rows) % (rows - region_rows + 1);
    const int boxes = region_rows < 0 ? rows / kBoxRows : region_rows / kBoxRows;
    int b = warp % boxes;
    const std::uint32_t mb = mb_smem_u32(&mbar[warp]);
    const std::uint32_t dst = box_base + warp * kBoxBytes;
    unsigned phase = 0;
    unsigned long long bytes = 0;
    unsigned lat[kLatSamples / kMaxSlots];
    int nlat = 0;
    const int my_rounds = rounds / slots;
    for (int r = 0; r < my_rounds; ++r) {
      expect_tx(mb, kBoxBytes);
      const long long t0 = (long long)mb_clock();
      tma_load(dst, &map, 0, row0 + b * kBoxRows, mb);
      mb_mbar_spin_to(mb, phase);
      const long long t1 = (long long)mb_clock();
      if (nlat < kLatSamples / kMaxSlots) lat[nlat++] = (unsigned)(t1 - t0);
      phase ^= 1u;
      bytes += kBoxBytes;
      b += slots; if (b >= boxes) b = warp % boxes;
    }
    atomicAdd(&cta_bytes, bytes);
    const int base = atomicAdd(&cta_lat_n, nlat);
    for (int i = 0; i < nlat && base + i < kLatSamples; ++i) o.lat[base + i] = lat[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const long long c1 = (long long)mb_clock();
    const long long g1 = (long long)mb_gtimer();
    o.bytes = cta_bytes; o.c0 = c0; o.c1 = c1; o.g0 = g0; o.g1 = g1; o.smid = mb_smid();
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
  std::string out = "tma_loaded.jsonl", mode = "hot", pair = "";
  int N = 1, S = 4, rounds = 2048, samples = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    if (a == "--out" && i + 1 < argc) { out = argv[++i]; continue; }
    if (a == "--mode" && i + 1 < argc) { mode = argv[++i]; continue; }
    if (a == "--pair" && i + 1 < argc) { pair = argv[++i]; continue; }
    if (n("--ctas", &N) || n("--slots", &S) || n("--rounds", &rounds) || n("--samples", &samples)) continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  if (S < 1 || S > kMaxSlots) { std::fprintf(stderr, "slots 1..%d\n", kMaxSlots); return 2; }
  const MbDevice dev = mb_query_device();
  int sel0 = -1, sel1 = -1;
  unsigned nparticipants = 0;
  if (!pair.empty()) {  // pair mode: grid = all SMs, 2 participants on chosen SM ids
    N = dev.sm_count; sel0 = 0; sel1 = (pair == "same") ? 1 : (pair == "diff") ? 2 : -1;
    if (sel1 < 0) { std::fprintf(stderr, "pair same|diff\n"); return 2; }
    nparticipants = 2;
  }
  if (N > dev.sm_count) { std::fprintf(stderr, "ctas > SMs\n"); return 2; }
  // hot: 24 MiB of 128 B rows (L2 resident on both devices);
  // shared_cold: 2 GiB, every CTA walks the same boxes, L2 evicted before each launch
  const bool shared_cold = mode == "shared_cold";
  const bool shared_hot = mode == "shared_hot";  // every CTA walks the same hot boxes (L2 resident)
  const int rows = static_cast<int>((shared_cold ? (2048ull << 20) : (24ull << 20)) / (kInner * 2));
  std::uint32_t* scratch = nullptr;
  const std::size_t scratch_bytes = 512ull << 20;
  if (shared_cold) MB_CHECK(cudaMalloc(&scratch, scratch_bytes));
  void* g = nullptr;
  MB_CHECK(cudaMalloc(&g, static_cast<std::size_t>(rows) * kInner * 2));
  MB_CHECK(cudaMemset(g, 1, static_cast<std::size_t>(rows) * kInner * 2));
  const CUtensorMap map = make_map(g, rows);
  unsigned* arrived = nullptr;
  CtaOut* d_out = nullptr;
  MB_CHECK(cudaMalloc(&arrived, sizeof(unsigned)));
  const int nout = nparticipants ? 2 : N;
  MB_CHECK(cudaMalloc(&d_out, sizeof(CtaOut) * nout));
  MB_CHECK(cudaFuncSetAttribute(k_tma_loaded, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  // per-CTA region: rows / N rounded down to boxes, at least S boxes
  if (!nparticipants) nparticipants = N;
  int start_stride = (rows / std::max(nout, 1)) / kBoxRows * kBoxRows;
  if (start_stride < S * kBoxRows) start_stride = S * kBoxRows;
  if (shared_cold || shared_hot) start_stride = -1;
  std::vector<double> bpc, elapsed, lat_all, agg_bns;
  std::string err;
  for (int rep = 0; rep < samples + 1; ++rep) {  // rep 0 = warm (L2 fill), discarded
    if (shared_cold) MB_CHECK(cudaMemset(scratch, rep, scratch_bytes));  // evict L2
    MB_CHECK(cudaMemset(arrived, 0, sizeof(unsigned)));
    MB_CHECK(cudaMemset(d_out, 0, sizeof(CtaOut) * nout));
    k_tma_loaded<<<N, 256, kSmemBytes>>>(map, rows, S, rounds, start_stride, arrived, d_out, sel0, sel1, nparticipants);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    if (mb_take_err()) { err = "mbarrier wait timeout"; break; }
    if (rep == 0) continue;
    std::vector<CtaOut> h(nout);
    MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(CtaOut) * nout, cudaMemcpyDeviceToHost));
    double total_bytes = 0, gmin = 1e300, gmax = 0;
    for (int i = 0; i < nout; ++i) {
      const double cyc = static_cast<double>(h[i].c1 - h[i].c0);
      bpc.push_back(static_cast<double>(h[i].bytes) / cyc);
      elapsed.push_back(cyc);
      total_bytes += static_cast<double>(h[i].bytes);
      gmin = std::min(gmin, static_cast<double>(h[i].g0));
      gmax = std::max(gmax, static_cast<double>(h[i].g1));
      for (int k = 0; k < kLatSamples; ++k) if (h[i].lat[k]) lat_all.push_back(h[i].lat[k]);
    }
    agg_bns.push_back(total_bytes / (gmax - gmin));
  }
  // jsonl
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str()); return 1; }
  const double med_bpc = pct(bpc, 0.5), p10 = pct(bpc, 0.10), p90 = pct(bpc, 0.90);
  std::fprintf(f,
               "{\"schema\":\"blackwell.mechanism_curve_point.v1\",\"family\":\"tma_loaded%s%s.N%d.S%d\","
               "\"quantity\":\"throughput\",\"cta_group\":1,\"x_name\":\"ctas\",\"x\":%d,"
               "\"y_name\":\"bytes_per_cycle_per_cta\",\"y_unit\":\"bytes/cycle\","
               "\"median\":%.4f,\"p10\":%.4f,\"p90\":%.4f,\"n\":%zu,\"holdout\":false,"
               "\"ctas\":%d,\"slots\":%d,\"rounds\":%d,\"box_bytes\":%d,"
               "\"aggregate_bytes_per_ns\":%s,\"aggregate_bytes_per_ns_median\":%.3f,"
               "\"latency_p10\":%.1f,\"latency_p50\":%.1f,\"latency_p90\":%.1f,\"latency_n\":%zu,"
               "\"elapsed_cycles_median\":%.0f,\"samples\":%s,"
               "\"source\":\"tma_loaded.measured\",\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,\"sm_count\":%d",
               shared_cold ? ".shared_cold" : shared_hot ? ".shared_hot" : "", pair.empty() ? "" : (pair == "same" ? ".pair_same" : ".pair_diff"), nout, S, nout, med_bpc, p10, p90, bpc.size(), N, S, rounds, kBoxBytes,
               arr(agg_bns).c_str(), pct(agg_bns, 0.5),
               pct(lat_all, 0.10), pct(lat_all, 0.50), pct(lat_all, 0.90), lat_all.size(),
               pct(elapsed, 0.5), N <= 16 ? arr(bpc).c_str() : "[]",
               dev.name.c_str(), dev.sm_clock_khz, dev.sm_count);
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("tma_loaded N=%d S=%d bpc/cta med=%.2f agg=%.1f B/ns lat p50=%.0f p90=%.0f%s\n", N, S, med_bpc,
              pct(agg_bns, 0.5), pct(lat_all, 0.5), pct(lat_all, 0.9), err.empty() ? "" : " ERR");
  cudaFree(g); cudaFree(arrived); cudaFree(d_out); if (scratch) cudaFree(scratch);
  return err.empty() ? 0 : 1;
}
