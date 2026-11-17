// tma_store_loaded probe: period of a GEMM-style store epilogue (smem
// write -> async-proxy fence -> named barrier -> TMA store -> commit ->
// wait_group<stages-1> -> named barrier) when N SMs run it at once, with
// and without TMA tile loads in flight on the same SM.
//
// Grid = N CTAs, one per SM (135 KB dynamic smem), 256 threads each.
//   warps 0-3 (128 threads, named barrier 1): the epilogue ring.  Per
//     round every thread writes 64 B (4 x st.shared.v4) into stage buffer
//     r % stages (128 threads x 64 B = one 8 KiB box), fence.proxy.async,
//     bar.sync 1; thread 0 issues one UTMASTG (2d tile, 128 x 16-bit
//     columns x 32 rows = 8 KiB), cp.async.bulk.commit_group,
//     cp.async.bulk.wait_group.read <stages-1>; bar.sync 1.
//     KIND = stg replaces the whole round by 4 x st.global.v4 per thread
//     (same 8 KiB per round, written straight to the destination rows).
//   warps 4-7 lane 0 (loaders > 0): one 16 KiB UTMALDG slot each over a
//     hot tensor, mbarrier.test_wait spin, re-issue (the tma_loaded v2
//     loader), until the epilogue ring finishes (or load_rounds when the
//     epilogue is off).
// Destination: every CTA owns region_boxes consecutive boxes of one
// tensor; round r stores box r % region_boxes (region_boxes = stages:
// "ring", rewritten rows; region_boxes = rounds: "fresh", every row
// written once).  All CTAs meet on a global arrival counter first.  The
// first kWarmRounds rounds are not counted.
//
// STAMPS = 0: no clock read inside the round (period = window / rounds).
// STAMPS = 1: clock64 after each step; per-step sums of thread 0 (elected
// warp) and thread 32 (a non-elected warp), first kSamples per-round
// wait_group / round times.
//
// jsonl row per cell: family
// tma_store_loaded.<kind>.N<N>.L<L>.D<D>.<dst>.<state>.s<stamps>[.epi0]
#include "../../tma_cold/mb.cuh"

#include <cuda.h>
#include <dlfcn.h>

#include <algorithm>
#include <string>
#include <vector>

constexpr int kStoreInner = 128;  // 16-bit elements per row = 256 B
constexpr int kStoreRows = 32;    // 8 KiB box
constexpr int kStoreRowBytes = kStoreInner * 2;
constexpr int kStoreBytes = kStoreRowBytes * kStoreRows;
constexpr int kMaxStages = 4;
constexpr int kLoadInner = 64;    // bf16 per row = 128 B
constexpr int kLoadRows = 128;    // 16 KiB box
constexpr int kLoadBytes = kLoadInner * 2 * kLoadRows;
constexpr int kLoadSlots = 4;
constexpr int kSmemBytes = 135168;  // -> 1 CTA / SM
constexpr int kWarmRounds = 8;
constexpr int kSamples = 96;

__device__ __forceinline__ void sts128(std::uint32_t a, std::uint32_t v) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void stg128(const unsigned char* p, std::uint32_t v) {
  asm volatile("st.global.v4.b32 [%0], {%1, %1, %1, %1};" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void bar_sync_epilogue() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}
__device__ __forceinline__ void tma_store(const CUtensorMap* map, int x, int y, std::uint32_t src) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
          reinterpret_cast<std::uint64_t>(map)),
      "r"(x), "r"(y), "r"(src)
      : "memory");
}
__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
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
  long long c0, c1, g0, g1;          // counted epilogue window (clock64, globaltimer)
  unsigned long long rounds;         // counted rounds
  unsigned long long seg[5];         // thread 0 sums: fence, bar1, issue, wait, bar2
  unsigned long long other[3];       // thread 32 sums: fence, bar1, rest (bar1 exit -> bar2 exit)
  unsigned long long load_bytes;
  long long lc0, lc1;                // loader window (slot 0)
  unsigned smid, pad;
  unsigned wait[kSamples];           // per-round wait_group time (STAMPS)
  unsigned round[kSamples];          // per-round start-to-start time (STAMPS)
};

template <int STAMPS, int KIND, int STAGES>
__global__ void __launch_bounds__(256, 1) k_epilogue(const __grid_constant__ CUtensorMap smap,
                                                     const __grid_constant__ CUtensorMap lmap,
                                                     unsigned char* gdst, int region_boxes, int load_rows,
                                                     int load_region_rows, int rounds, int loaders, int epi,
                                                     int load_rounds, unsigned* arrived, CtaOut* out,
                                                     unsigned nparticipants) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) std::uint64_t mbar[kLoadSlots];
  __shared__ unsigned long long cta_load_bytes;
  __shared__ int stop;
  __shared__ unsigned s_wait[kSamples];
  __shared__ unsigned s_round[kSamples];
  const std::uint32_t base = (mb_smem_u32(smem) + 1023u) & ~1023u;
  const std::uint32_t store_base = base;
  const std::uint32_t load_base = base + kMaxStages * kStoreBytes;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int idx = static_cast<int>(blockIdx.x);
  if (threadIdx.x == 0) {
    for (int s = 0; s < kLoadSlots; ++s) mb_mbar_init(mb_smem_u32(&mbar[s]), 1);
    mb_fence_mbar_init();
    cta_load_bytes = 0;
    stop = 0;
    for (int i = 0; i < kSamples; ++i) { s_wait[i] = 0; s_round[i] = 0; }
  }
  __syncthreads();
  if (threadIdx.x == 0) {  // all CTAs resident before any timed work
    atomicAdd(arrived, 1u);
    while (*((volatile unsigned*)arrived) < nparticipants) { }
  }
  __syncthreads();
  CtaOut& o = out[idx];
  if (warp < 4) {
    if (epi) {
      const int row0 = idx * region_boxes * kStoreRows;
      unsigned long long s_fence = 0, s_bar1 = 0, s_issue = 0, s_waitsum = 0, s_bar2 = 0, s_rest = 0;
      long long c0 = 0, g0 = 0, tprev = 0;
      const std::uint32_t v = threadIdx.x;
      for (int r = 0; r < rounds; ++r) {
        if (r == kWarmRounds) {
          g0 = static_cast<long long>(mb_gtimer());
          c0 = static_cast<long long>(mb_clock());
        }
        const int b = r % STAGES;
        const int box = r % region_boxes;
        long long tA = 0, tB = 0, tC = 0, tD = 0, tE = 0, tF = 0;
        if (STAMPS) tA = static_cast<long long>(mb_clock());
        if (KIND == 0) {
          const std::uint32_t a = store_base + b * kStoreBytes + threadIdx.x * 64;
          sts128(a, v);
          sts128(a + 16, v);
          sts128(a + 32, v);
          sts128(a + 48, v);
          fence_async_shared();
          if (STAMPS) tB = static_cast<long long>(mb_clock());
          bar_sync_epilogue();
          if (STAMPS) tC = static_cast<long long>(mb_clock());
          tD = tC;
          tE = tC;
          if (threadIdx.x == 0) {
            tma_store(&smap, 0, row0 + box * kStoreRows, store_base + b * kStoreBytes);
            bulk_commit();
            if (STAMPS) tD = static_cast<long long>(mb_clock());
            bulk_wait_read<STAGES - 1>();
            if (STAMPS) tE = static_cast<long long>(mb_clock());
          }
          bar_sync_epilogue();
          if (STAMPS) tF = static_cast<long long>(mb_clock());
        } else {
          const unsigned char* p = gdst +
              static_cast<std::size_t>(row0 + box * kStoreRows) * kStoreRowBytes + threadIdx.x * 64;
          stg128(p, v);
          stg128(p + 16, v);
          stg128(p + 32, v);
          stg128(p + 48, v);
          if (STAMPS) tB = static_cast<long long>(mb_clock());
          tC = tB; tD = tB; tE = tB; tF = tB;
        }
        if (STAMPS && r >= kWarmRounds) {
          s_fence += static_cast<unsigned long long>(tB - tA);
          s_bar1 += static_cast<unsigned long long>(tC - tB);
          s_issue += static_cast<unsigned long long>(tD - tC);
          s_waitsum += static_cast<unsigned long long>(tE - tD);
          s_bar2 += static_cast<unsigned long long>(tF - tE);
          s_rest += static_cast<unsigned long long>(tF - tC);
          const int i = r - kWarmRounds;
          if (threadIdx.x == 0 && i < kSamples) {
            s_wait[i] = static_cast<unsigned>(tE - tD);
            s_round[i] = i ? static_cast<unsigned>(tA - tprev) : 0u;
          }
          tprev = tA;
        }
      }
      const long long c1 = static_cast<long long>(mb_clock());
      const long long g1 = static_cast<long long>(mb_gtimer());
      if (threadIdx.x == 0) {
        *((volatile int*)&stop) = 1;
        o.c0 = c0; o.c1 = c1; o.g0 = g0; o.g1 = g1;
        o.rounds = static_cast<unsigned long long>(rounds - kWarmRounds);
        o.seg[0] = s_fence; o.seg[1] = s_bar1; o.seg[2] = s_issue; o.seg[3] = s_waitsum; o.seg[4] = s_bar2;
        for (int i = 0; i < kSamples; ++i) { o.wait[i] = s_wait[i]; o.round[i] = s_round[i]; }
      }
      if (threadIdx.x == 32) {
        o.other[0] = s_fence; o.other[1] = s_bar1; o.other[2] = s_rest;
      }
    }
  } else if (warp - 4 < loaders && lane == 0) {
    const int s = warp - 4;
    const int boxes = load_region_rows / kLoadRows;
    const int lrow0 = (idx * load_region_rows) % (load_rows - load_region_rows + 1);
    int b = s % boxes;
    const std::uint32_t mb = mb_smem_u32(&mbar[s]);
    const std::uint32_t dst = load_base + s * kLoadBytes;
    unsigned phase = 0;
    unsigned long long bytes = 0;
    const long long l0 = static_cast<long long>(mb_clock());
    for (int r = 0; epi ? (*((volatile int*)&stop) == 0) : (r < load_rounds); ++r) {
      expect_tx(mb, kLoadBytes);
      tma_load(dst, &lmap, 0, lrow0 + b * kLoadRows, mb);
      mb_mbar_spin_to(mb, phase);
      phase ^= 1u;
      bytes += kLoadBytes;
      b += loaders;
      if (b >= boxes) b = s % boxes;
    }
    const long long l1 = static_cast<long long>(mb_clock());
    atomicAdd(&cta_load_bytes, bytes);
    if (s == 0) { o.lc0 = l0; o.lc1 = l1; }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    o.load_bytes = cta_load_bytes;
    o.smid = mb_smid();
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
static CUtensorMap make_map(void* g, std::uint64_t rows, int inner, int box_rows, CUtensorMapSwizzle swizzle,
                            CUtensorMapL2promotion promo) {
  CUtensorMap m{};
  const cuuint64_t dim[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(rows)};
  const cuuint64_t stride[1] = {static_cast<cuuint64_t>(inner * 2)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(inner), static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t es[2] = {1u, 1u};
  auto fn = encoder();
  if (!fn || fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, dim, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle, promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
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

struct Launch {
  CUtensorMap smap, lmap;
  unsigned char* gdst;
  int region_boxes, load_rows, load_region_rows, rounds, loaders, epi, load_rounds;
  unsigned* arrived;
  CtaOut* out;
  unsigned n;
};
template <int STAMPS, int KIND, int STAGES>
static void launch_one(const Launch& l) {
  MB_CHECK(cudaFuncSetAttribute(k_epilogue<STAMPS, KIND, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kSmemBytes));
  k_epilogue<STAMPS, KIND, STAGES><<<l.n, 256, kSmemBytes>>>(l.smap, l.lmap, l.gdst, l.region_boxes, l.load_rows,
                                                            l.load_region_rows, l.rounds, l.loaders, l.epi,
                                                            l.load_rounds, l.arrived, l.out, l.n);
}
static void launch(int stamps, int kind, int stages, const Launch& l) {
  const int key = stamps * 4 + kind * 2 + (stages == 4 ? 1 : 0);
  switch (key) {
    case 0: launch_one<0, 0, 1>(l); break;
    case 1: launch_one<0, 0, 4>(l); break;
    case 2: launch_one<0, 1, 1>(l); break;
    case 3: launch_one<0, 1, 4>(l); break;
    case 4: launch_one<1, 0, 1>(l); break;
    case 5: launch_one<1, 0, 4>(l); break;
    case 6: launch_one<1, 1, 1>(l); break;
    default: launch_one<1, 1, 4>(l); break;
  }
}

int main(int argc, char** argv) {
  std::string out = "tma_store_loaded.jsonl", kind = "tma", dst = "ring", state = "warm";
  int N = 1, L = 0, D = 4, rounds = 264, samples = 8, stamps = 0, epi = 1, load_rounds = 256;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    auto s = [&](const char* k, std::string* d) { if (a == k && i + 1 < argc) { *d = argv[++i]; return true; } return false; };
    if (s("--out", &out) || s("--kind", &kind) || s("--dst", &dst) || s("--state", &state)) continue;
    if (n("--ctas", &N) || n("--loaders", &L) || n("--stages", &D) || n("--rounds", &rounds) ||
        n("--samples", &samples) || n("--stamps", &stamps) || n("--epi", &epi) || n("--load-rounds", &load_rounds))
      continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  if ((kind != "tma" && kind != "stg") || (dst != "ring" && dst != "fresh") || (state != "warm" && state != "cold") ||
      (D != 1 && D != 4) || (L != 0 && L != kLoadSlots) || (stamps != 0 && stamps != 1) || (epi != 0 && epi != 1) ||
      rounds <= kWarmRounds) {
    std::fprintf(stderr, "bad cell\n");
    return 2;
  }
  const MbDevice dev = mb_query_device();
  if (N < 1 || N > dev.sm_count) { std::fprintf(stderr, "ctas 1..SMs\n"); return 2; }
  const int region_boxes = dst == "ring" ? kMaxStages : rounds;
  const std::uint64_t store_rows = static_cast<std::uint64_t>(N) * region_boxes * kStoreRows;
  const std::size_t store_bytes = static_cast<std::size_t>(store_rows) * kStoreRowBytes;
  const int load_rows = static_cast<int>((24ull << 20) / (kLoadInner * 2));
  const std::size_t load_bytes = static_cast<std::size_t>(load_rows) * kLoadInner * 2;
  int load_region_rows = (load_rows / N) / kLoadRows * kLoadRows;
  if (load_region_rows < kLoadSlots * kLoadRows) load_region_rows = kLoadSlots * kLoadRows;
  unsigned char* gdst = nullptr;
  void* gsrc = nullptr;
  std::uint32_t* scratch = nullptr;
  const std::size_t scratch_bytes = 512ull << 20;
  MB_CHECK(cudaMalloc(&gdst, store_bytes));
  MB_CHECK(cudaMalloc(&gsrc, load_bytes));
  if (state == "cold") MB_CHECK(cudaMalloc(&scratch, scratch_bytes));
  Launch l{};
  l.smap = make_map(gdst, store_rows, kStoreInner, kStoreRows, CU_TENSOR_MAP_SWIZZLE_NONE,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE);
  l.lmap = make_map(gsrc, static_cast<std::uint64_t>(load_rows), kLoadInner, kLoadRows, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  l.gdst = gdst; l.region_boxes = region_boxes; l.load_rows = load_rows; l.load_region_rows = load_region_rows;
  l.rounds = rounds; l.loaders = L; l.epi = epi; l.load_rounds = load_rounds; l.n = static_cast<unsigned>(N);
  MB_CHECK(cudaMalloc(&l.arrived, sizeof(unsigned)));
  MB_CHECK(cudaMalloc(&l.out, sizeof(CtaOut) * N));
  const int kind_i = kind == "stg" ? 1 : 0;
  std::vector<double> period, agg_bns, cpn, load_bpc, wait_all, round_all;
  std::vector<double> seg[5], other[3];
  std::string err;
  for (int rep = 0; rep < samples + 1; ++rep) {  // rep 0 discarded
    if (scratch) MB_CHECK(cudaMemset(scratch, rep, scratch_bytes));  // evict L2
    MB_CHECK(cudaMemset(gsrc, 1, load_bytes));                       // loads stay L2 hits
    if (state == "warm") MB_CHECK(cudaMemset(gdst, 0, store_bytes));
    MB_CHECK(cudaMemset(l.arrived, 0, sizeof(unsigned)));
    MB_CHECK(cudaMemset(l.out, 0, sizeof(CtaOut) * N));
    launch(stamps, kind_i, D, l);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    if (mb_take_err()) { err = "mbarrier wait timeout"; break; }
    if (rep == 0) continue;
    std::vector<CtaOut> h(N);
    MB_CHECK(cudaMemcpy(h.data(), l.out, sizeof(CtaOut) * N, cudaMemcpyDeviceToHost));
    double total = 0, gmin = 1e300, gmax = 0;
    for (int i = 0; i < N; ++i) {
      if (epi) {
        const double cyc = static_cast<double>(h[i].c1 - h[i].c0);
        const double ns = static_cast<double>(h[i].g1 - h[i].g0);
        const double nr = static_cast<double>(h[i].rounds);
        period.push_back(cyc / nr);
        if (ns > 0) cpn.push_back(cyc / ns);
        total += nr * kStoreBytes;
        gmin = std::min(gmin, static_cast<double>(h[i].g0));
        gmax = std::max(gmax, static_cast<double>(h[i].g1));
        if (stamps) {
          for (int k = 0; k < 5; ++k) seg[k].push_back(static_cast<double>(h[i].seg[k]) / nr);
          for (int k = 0; k < 3; ++k) other[k].push_back(static_cast<double>(h[i].other[k]) / nr);
          for (int k = 0; k < kSamples; ++k) {
            if (k < static_cast<int>(h[i].rounds)) wait_all.push_back(h[i].wait[k]);
            if (k && k < static_cast<int>(h[i].rounds)) round_all.push_back(h[i].round[k]);
          }
        }
      }
      if (L && h[i].lc1 > h[i].lc0)
        load_bpc.push_back(static_cast<double>(h[i].load_bytes) / static_cast<double>(h[i].lc1 - h[i].lc0));
    }
    if (epi && gmax > gmin) agg_bns.push_back(total / (gmax - gmin));
  }
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str()); return 1; }
  char fam[160];
  std::snprintf(fam, sizeof(fam), "tma_store_loaded.%s.N%d.L%d.D%d.%s.%s.s%d%s", kind.c_str(), N, L, D, dst.c_str(),
                state.c_str(), stamps, epi ? "" : ".epi0");
  std::fprintf(f,
               "{\"schema\":\"cyclescope.microbench.tma_store_loaded.v1\",\"family\":\"%s\",\"kind\":\"%s\","
               "\"ctas\":%d,\"loaders\":%d,\"stages\":%d,\"dst\":\"%s\",\"state\":\"%s\",\"stamps\":%d,\"epi\":%d,"
               "\"rounds\":%d,\"warm_rounds\":%d,\"region_boxes\":%d,\"store_box_bytes\":%d,\"load_box_bytes\":%d,"
               "\"period_median\":%.3f,\"period_p10\":%.3f,\"period_p90\":%.3f,\"period_n\":%zu,"
               "\"aggregate_store_bytes_per_ns\":%s,\"aggregate_store_bytes_per_ns_median\":%.3f,"
               "\"cycles_per_ns_median\":%.5f,"
               "\"load_bytes_per_cycle_median\":%.3f,\"load_bytes_per_cycle_p10\":%.3f,\"load_bytes_per_cycle_p90\":%.3f,"
               "\"seg_fence\":%.2f,\"seg_bar1\":%.2f,\"seg_issue\":%.2f,\"seg_wait\":%.2f,\"seg_bar2\":%.2f,"
               "\"other_fence\":%.2f,\"other_bar1\":%.2f,\"other_rest\":%.2f,"
               "\"wait_p10\":%.1f,\"wait_p50\":%.1f,\"wait_p90\":%.1f,\"round_p10\":%.1f,\"round_p50\":%.1f,\"round_p90\":%.1f,"
               "\"periods\":%s,\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,\"sm_count\":%d",
               fam, kind.c_str(), N, L, D, dst.c_str(), state.c_str(), stamps, epi, rounds, kWarmRounds, region_boxes,
               kStoreBytes, kLoadBytes, pct(period, 0.5), pct(period, 0.10), pct(period, 0.90), period.size(),
               arr(agg_bns).c_str(), pct(agg_bns, 0.5), pct(cpn, 0.5), pct(load_bpc, 0.5), pct(load_bpc, 0.10),
               pct(load_bpc, 0.90), pct(seg[0], 0.5), pct(seg[1], 0.5), pct(seg[2], 0.5), pct(seg[3], 0.5),
               pct(seg[4], 0.5), pct(other[0], 0.5), pct(other[1], 0.5), pct(other[2], 0.5), pct(wait_all, 0.10),
               pct(wait_all, 0.5), pct(wait_all, 0.90), pct(round_all, 0.10), pct(round_all, 0.5),
               pct(round_all, 0.90), N <= 16 ? arr(period).c_str() : "[]", dev.name.c_str(), dev.sm_clock_khz,
               dev.sm_count);
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("%s period=%.1f agg=%.1f B/ns load=%.1f B/cyc wait_p50=%.0f%s\n", fam, pct(period, 0.5),
              pct(agg_bns, 0.5), pct(load_bpc, 0.5), pct(wait_all, 0.5), err.empty() ? "" : " ERR");
  cudaFree(gdst); cudaFree(gsrc); cudaFree(l.arrived); cudaFree(l.out);
  if (scratch) cudaFree(scratch);
  return err.empty() ? 0 : 1;
}
