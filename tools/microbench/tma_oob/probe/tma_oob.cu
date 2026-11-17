// tma_oob probe: same as tma_loaded (TMA tile-load delivery and completion
// latency) plus an x-coordinate axis: the 64 x 128 bf16 SWIZZLE_128B box is
// loaded at x = 0 (in bounds), x = 32 (half the row out of bounds) or x = 64
// (fully out of bounds, fill only). Hot mode only; after the last timed
// round slot 0 lane 0 sums the first 64 B of its smem box into check.
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
// jsonl row per cell: family tma_oob.x<X>.N<N>.S<S>, samples = per-CTA
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
__device__ __forceinline__ void tma_load3(std::uint32_t dst, const CUtensorMap* map, int x, int y, int z, std::uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(mbar)
      : "memory");
}
__device__ __forceinline__ void expect_tx(std::uint32_t mbar, std::uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}

struct CtaOut {
  unsigned long long bytes;
  long long c0, c1, g0, g1;
  unsigned smid;
  unsigned check;
  unsigned lat[kLatSamples];
};

__global__ void __launch_bounds__(256, 1) k_tma_oob(const __grid_constant__ CUtensorMap map, int rows,
                                                       int slots, int rounds, int start_stride, int xcoord, int box_rows,
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
    const int boxes = region_rows < 0 ? rows / box_rows : region_rows / box_rows;
    int b = warp % boxes;
    const std::uint32_t mb = mb_smem_u32(&mbar[warp]);
    const std::uint32_t dst = box_base + warp * kBoxBytes;
    unsigned phase = 0;
    unsigned long long bytes = 0;
    unsigned lat[kLatSamples / kMaxSlots];
    int nlat = 0;
    const int my_rounds = rounds / slots;
    for (int r = 0; r < my_rounds; ++r) {
      expect_tx(mb, kInner * 2 * box_rows);
      const long long t0 = (long long)mb_clock();
      tma_load(dst, &map, xcoord, row0 + b * box_rows, mb);
      mb_mbar_spin_to(mb, phase);
      const long long t1 = (long long)mb_clock();
      if (nlat < kLatSamples / kMaxSlots) lat[nlat++] = (unsigned)(t1 - t0);
      phase ^= 1u;
      bytes += kInner * 2 * box_rows;
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
    // data check (outside timing): slot 0 box first 64 B
    unsigned sum = 0;
    const unsigned char* box = smem + (box_base - smem_base);
    for (int i = 0; i < 64; ++i) sum += box[i];
    o.check = sum;
    o.bytes = cta_bytes; o.c0 = c0; o.c1 = c1; o.g0 = g0; o.g1 = g1; o.smid = mb_smid();
    for (int i = cta_lat_n; i < kLatSamples; ++i) o.lat[i] = 0;
  }
}

// rank-3 variant (rows-in mode): 'rows' is the plane count P, region
// entries are planes; each box reads (x, 0, plane).
__global__ void __launch_bounds__(256, 1) k_tma_oob3(const __grid_constant__ CUtensorMap map, int rows,
                                                       int slots, int rounds, int start_stride, int xcoord, int box_rows,
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
    const int boxes = region_rows < 0 ? rows : region_rows;
    int b = warp % boxes;
    const std::uint32_t mb = mb_smem_u32(&mbar[warp]);
    const std::uint32_t dst = box_base + warp * kBoxBytes;
    unsigned phase = 0;
    unsigned long long bytes = 0;
    unsigned lat[kLatSamples / kMaxSlots];
    int nlat = 0;
    const int my_rounds = rounds / slots;
    for (int r = 0; r < my_rounds; ++r) {
      expect_tx(mb, kInner * 2 * box_rows);
      const long long t0 = (long long)mb_clock();
      tma_load3(dst, &map, xcoord, 0, row0 + b, mb);
      mb_mbar_spin_to(mb, phase);
      const long long t1 = (long long)mb_clock();
      if (nlat < kLatSamples / kMaxSlots) lat[nlat++] = (unsigned)(t1 - t0);
      phase ^= 1u;
      bytes += kInner * 2 * box_rows;
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
    // data check (outside timing): slot 0 box first 64 B
    unsigned sum = 0;
    const unsigned char* box = smem + (box_base - smem_base);
    for (int i = 0; i < 64; ++i) sum += box[i];
    o.check = sum;
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
static CUtensorMap make_map(void* g, int rows, int box_rows) {
  CUtensorMap m{};
  const cuuint64_t dim[2] = {static_cast<cuuint64_t>(kInner), static_cast<cuuint64_t>(rows)};
  const cuuint64_t stride[1] = {static_cast<cuuint64_t>(kInner * 2)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(kInner), static_cast<cuuint32_t>(box_rows)};
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
// rank-3 rows-in map: dims {64, R rows, P planes}, box {64, box_rows, 1};
// a box at (x, 0, plane) sees rows R..box_rows-1 out of bounds.
static CUtensorMap make_map3(void* g, int rows_in, int planes, int box_rows) {
  CUtensorMap m{};
  const cuuint64_t dim[3] = {static_cast<cuuint64_t>(kInner), static_cast<cuuint64_t>(rows_in),
                             static_cast<cuuint64_t>(planes)};
  const cuuint64_t stride[2] = {static_cast<cuuint64_t>(kInner * 2),
                                static_cast<cuuint64_t>(rows_in) * kInner * 2};
  const cuuint32_t box[3] = {static_cast<cuuint32_t>(kInner), static_cast<cuuint32_t>(box_rows), 1u};
  const cuuint32_t es[3] = {1u, 1u, 1u};
  auto fn = encoder();
  if (!fn || fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, g, dim, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
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
  std::string out = "tma_oob.jsonl";
  int N = 1, S = 4, rounds = 2048, samples = 8, xcoord = 0, box_rows = kBoxRows, rows_in = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    if (a == "--out" && i + 1 < argc) { out = argv[++i]; continue; }
    if (n("--ctas", &N) || n("--slots", &S) || n("--rounds", &rounds) || n("--samples", &samples) || n("--x", &xcoord) || n("--box-rows", &box_rows) || n("--rows-in", &rows_in)) continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  if (S < 1 || S > kMaxSlots) { std::fprintf(stderr, "slots 1..%d\n", kMaxSlots); return 2; }
  if (xcoord < 0 || xcoord > 64 || xcoord % 8 != 0) { std::fprintf(stderr, "--x multiple of 8 in [0,64]\n"); return 2; }
  if (box_rows != 128 && box_rows != 32) { std::fprintf(stderr, "--box-rows 32|128\n"); return 2; }
  if (rows_in == 0 || rows_in < -1) { std::fprintf(stderr, "--rows-in R>0\n"); return 2; }
  const MbDevice dev = mb_query_device();
  const int sel0 = -1, sel1 = -1;
  unsigned nparticipants = 0;
  if (N > dev.sm_count) { std::fprintf(stderr, "ctas > SMs\n"); return 2; }
  // hot: ~24 MiB (L2 resident on both devices); rows-in mode is rank 3
  // {64, R, P} with P = 24 MiB / (128 * R) planes.
  const bool rank3 = rows_in > 0;
  const int rows = rank3 ? static_cast<int>((24ull << 20) / ((std::uint64_t)kInner * 2 * rows_in))
                         : static_cast<int>((24ull << 20) / (kInner * 2));
  void* g = nullptr;
  MB_CHECK(cudaMalloc(&g, static_cast<std::size_t>(rows) * kInner * 2 * (rank3 ? rows_in : 1)));
  MB_CHECK(cudaMemset(g, 1, static_cast<std::size_t>(rows) * kInner * 2 * (rank3 ? rows_in : 1)));
  const CUtensorMap map = rank3 ? make_map3(g, rows_in, rows, box_rows) : make_map(g, rows, box_rows);
  unsigned* arrived = nullptr;
  CtaOut* d_out = nullptr;
  MB_CHECK(cudaMalloc(&arrived, sizeof(unsigned)));
  const int nout = nparticipants ? 2 : N;
  MB_CHECK(cudaMalloc(&d_out, sizeof(CtaOut) * nout));
  MB_CHECK(cudaFuncSetAttribute(rank3 ? k_tma_oob3 : k_tma_oob, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  // per-CTA region: rows / N rounded down to boxes, at least S boxes
  nparticipants = N;
  // 2D: start_stride in rows (multiple of box_rows); 3D: in planes (>= slots)
  int start_stride = rank3 ? std::max(rows / std::max(nout, 1), S)
                           : (rows / std::max(nout, 1)) / box_rows * box_rows;
  if (!rank3 && start_stride < S * box_rows) start_stride = S * box_rows;
  std::vector<double> bpc, elapsed, lat_all, agg_bns;
  std::string err;
  for (int rep = 0; rep < samples + 1; ++rep) {  // rep 0 = warm (L2 fill), discarded
    MB_CHECK(cudaMemset(arrived, 0, sizeof(unsigned)));
    MB_CHECK(cudaMemset(d_out, 0, sizeof(CtaOut) * nout));
    (rank3 ? k_tma_oob3 : k_tma_oob)<<<N, 256, kSmemBytes>>>(map, rows, S, rounds, start_stride, xcoord, box_rows, arrived, d_out, sel0, sel1, nparticipants);
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
    const unsigned chk = h[0].check;  // CTA 0 slot-0 box first 64 B
    const bool ok = (xcoord == 0 && (!rank3 || rows_in == box_rows)) ? (chk == 64) : (xcoord == 64) ? (chk == 0) : (chk <= 64);
    if (!ok) { char eb[64]; std::snprintf(eb, sizeof(eb), "data check %u", chk); err = eb; break; }
  }
  // jsonl
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str()); return 1; }
  const double med_bpc = pct(bpc, 0.5), p10 = pct(bpc, 0.10), p90 = pct(bpc, 0.90);
  char rbuf[16];
  if (rank3) std::snprintf(rbuf, sizeof(rbuf), "%d", rows_in); else std::snprintf(rbuf, sizeof(rbuf), "full");
  std::fprintf(f,
               "{\"schema\":\"blackwell.mechanism_curve_point.v1\",\"family\":\"tma_oob.b%d.x%d.r%s.N%d.S%d\","
               "\"quantity\":\"throughput\",\"cta_group\":1,\"x_name\":\"ctas\",\"x\":%d,"
               "\"y_name\":\"bytes_per_cycle_per_cta\",\"y_unit\":\"bytes/cycle\","
               "\"median\":%.4f,\"p10\":%.4f,\"p90\":%.4f,\"n\":%zu,\"holdout\":false,"
               "\"ctas\":%d,\"slots\":%d,\"rounds\":%d,\"box_bytes\":%d,"
               "\"x_coord\":%d,\"in_bounds_bytes\":%d,\"box_rows\":%d,\"rows_in\":%d,\"rank\":%d,"
               "\"aggregate_bytes_per_ns\":%s,\"aggregate_bytes_per_ns_median\":%.3f,"
               "\"latency_p10\":%.1f,\"latency_p50\":%.1f,\"latency_p90\":%.1f,\"latency_n\":%zu,"
               "\"elapsed_cycles_median\":%.0f,\"samples\":%s,"
               "\"source\":\"tma_oob.measured\",\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,\"sm_count\":%d",
               box_rows, xcoord, rbuf, nout, S, nout, med_bpc, p10, p90, bpc.size(), N, S, rounds, kInner * 2 * box_rows, xcoord, kInner * 2 * (rank3 ? std::min(rows_in, box_rows) : box_rows) * (64 - xcoord) / 64, box_rows, rows_in, rank3 ? 3 : 2,
               arr(agg_bns).c_str(), pct(agg_bns, 0.5),
               pct(lat_all, 0.10), pct(lat_all, 0.50), pct(lat_all, 0.90), lat_all.size(),
               pct(elapsed, 0.5), N <= 16 ? arr(bpc).c_str() : "[]",
               dev.name.c_str(), dev.sm_clock_khz, dev.sm_count);
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("tma_oob x=%d N=%d S=%d bpc/cta med=%.2f agg=%.1f B/ns lat p50=%.0f p90=%.0f%s\n", xcoord, N, S, med_bpc,
              pct(agg_bns, 0.5), pct(lat_all, 0.5), pct(lat_all, 0.9), err.empty() ? "" : " ERR");
  cudaFree(g); cudaFree(arrived); cudaFree(d_out);
  return err.empty() ? 0 : 1;
}
