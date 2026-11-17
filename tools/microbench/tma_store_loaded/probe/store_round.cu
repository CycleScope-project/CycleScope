// tma_store_loaded v2 probe: per-warp timeline of a GEMM-style store
// epilogue round, by round shape (MODE), store tensor-map geometry and
// TMA loads in flight on the same SM (protocol-v2.json).
//
// Grid = N CTAs, one per SM (135 KB dynamic smem), 256 threads.
//   warps 0-3 (128 threads, named barrier 1) run `rounds` rounds:
//     MODE 0 full : [warps < writers: 4 x st.shared.v4 (64 B / thread)]
//                   fence.proxy.async ; bar.sync 1 ; thread 0: UTMASTG +
//                   commit_group + wait_group.read <STAGES-1> ; bar.sync 1
//     MODE 1 fence: the same without the thread-0 TMA block
//     MODE 2 store: bar.sync 1 ; thread 0 TMA block ; bar.sync 1 (source
//                   boxes written and fenced once before the rounds)
//     MODE 3 sts  : the smem writes ; bar.sync 1 ; bar.sync 1
//   warps 4-7 lane 0 (loaders = 4): one 16 KiB UTMALDG slot each over a
//     read-warmed tensor region of 64 boxes per CTA, test_wait spin.
// Store destination: every CTA owns 4 boxes of one tensor (ring).  The
// store map geometry (row bytes, swizzle, L2 promotion, global row
// stride) is a cell parameter.
// STAMPS 1: every warp of 0-3 accumulates the absolute clock64 of the six
// round stamps (A round start, B after the fence step, C after bar 1,
// D after UTMASTG + commit, E after wait_group, F after bar 2; D = E = C
// for threads other than thread 0) over the counted rounds; the host
// reports per-warp mean offsets from warp 0's A.  STAMPS 0: no clock
// read inside the round.
#include "../../tma_cold/mb.cuh"

#include <cuda.h>
#include <dlfcn.h>

#include <algorithm>
#include <string>
#include <vector>

constexpr int kMaxStages = 4;
constexpr int kMaxBoxBytes = 16384;
constexpr int kLoadInner = 64;
constexpr int kLoadRows = 128;
constexpr int kLoadBytes = kLoadInner * 2 * kLoadRows;
constexpr int kLoadSlots = 4;
constexpr int kLoadRegionBoxes = 64;
constexpr int kSmemBytes = 135168;
constexpr int kWarmRounds = 8;

__device__ __forceinline__ void sts128(std::uint32_t a, std::uint32_t v) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};" ::"r"(a), "r"(v) : "memory");
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
  long long c0, c1, g0, g1;      // counted window (thread 0)
  unsigned long long rounds;
  unsigned long long t[4][6];    // per epilogue warp: sums of absolute stamps A..F
  unsigned long long load_bytes;
  long long lc0, lc1;
  unsigned smid, pad;
};

template <int STAMPS, int MODE, int STAGES>
__global__ void __launch_bounds__(256, 1) k_round(const __grid_constant__ CUtensorMap smap,
                                                  const __grid_constant__ CUtensorMap lmap, int box_bytes,
                                                  int box_rows, int writers, int rounds, int loaders, int epi,
                                                  int load_rounds, unsigned* arrived, CtaOut* out,
                                                  unsigned nparticipants) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) std::uint64_t mbar[kLoadSlots];
  __shared__ unsigned long long cta_load_bytes;
  __shared__ int stop;
  const std::uint32_t base = (mb_smem_u32(smem) + 1023u) & ~1023u;
  const std::uint32_t store_base = base;
  const std::uint32_t load_base = base + kMaxStages * kMaxBoxBytes;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int idx = static_cast<int>(blockIdx.x);
  if (threadIdx.x == 0) {
    for (int s = 0; s < kLoadSlots; ++s) mb_mbar_init(mb_smem_u32(&mbar[s]), 1);
    mb_fence_mbar_init();
    cta_load_bytes = 0;
    stop = 0;
  }
  __syncthreads();
  if (MODE == 2 && warp < 4) {  // source boxes for the store-only rounds
    for (int i = threadIdx.x * 16; i < kMaxStages * box_bytes; i += 128 * 16) sts128(store_base + i, threadIdx.x);
    fence_async_shared();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(arrived, 1u);
    while (*((volatile unsigned*)arrived) < nparticipants) { }
  }
  __syncthreads();
  CtaOut& o = out[idx];
  if (warp < 4) {
    if (epi) {
      const int row0 = idx * kMaxStages * box_rows;
      unsigned long long sA = 0, sB = 0, sC = 0, sD = 0, sE = 0, sF = 0;
      long long c0 = 0, g0 = 0;
      const std::uint32_t v = threadIdx.x;
      const bool writer = warp < writers;
      for (int r = 0; r < rounds; ++r) {
        if (r == kWarmRounds) {
          g0 = static_cast<long long>(mb_gtimer());
          c0 = static_cast<long long>(mb_clock());
        }
        const int b = r % STAGES;
        unsigned long long tA = 0, tB = 0, tC = 0, tD = 0, tE = 0, tF = 0;
        if (STAMPS) tA = mb_clock();
        if (MODE != 2) {
          if (writer) {
            const std::uint32_t a = store_base + b * box_bytes + threadIdx.x * 64;
            sts128(a, v);
            sts128(a + 16, v);
            sts128(a + 32, v);
            sts128(a + 48, v);
          }
          if (MODE != 3) fence_async_shared();
        }
        if (STAMPS) tB = mb_clock();
        bar_sync_epilogue();
        if (STAMPS) tC = mb_clock();
        tD = tC;
        tE = tC;
        if (MODE == 0 || MODE == 2) {
          if (threadIdx.x == 0) {
            tma_store(&smap, 0, row0 + b * box_rows, store_base + b * box_bytes);
            bulk_commit();
            if (STAMPS) tD = mb_clock();
            bulk_wait_read<STAGES - 1>();
            if (STAMPS) tE = mb_clock();
          }
        }
        bar_sync_epilogue();
        if (STAMPS) tF = mb_clock();
        if (STAMPS && r >= kWarmRounds) {
          sA += tA; sB += tB; sC += tC; sD += tD; sE += tE; sF += tF;
        }
      }
      const long long c1 = static_cast<long long>(mb_clock());
      const long long g1 = static_cast<long long>(mb_gtimer());
      if (threadIdx.x == 0) {
        *((volatile int*)&stop) = 1;
        o.c0 = c0; o.c1 = c1; o.g0 = g0; o.g1 = g1;
        o.rounds = static_cast<unsigned long long>(rounds - kWarmRounds);
      }
      if (lane == 0) {
        o.t[warp][0] = sA; o.t[warp][1] = sB; o.t[warp][2] = sC;
        o.t[warp][3] = sD; o.t[warp][4] = sE; o.t[warp][5] = sF;
      }
    }
  } else if (warp - 4 < loaders && lane == 0) {
    const int s = warp - 4;
    const int lrow0 = idx * kLoadRegionBoxes * kLoadRows;
    int b = s;
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
      b += kLoadSlots;
      if (b >= kLoadRegionBoxes) b = s;
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
// 16-bit elements: `inner` elements per box row, `width` elements per global row (stride = 2 x width bytes)
static CUtensorMap make_map(void* g, std::uint64_t rows, int inner, std::uint64_t width, int box_rows,
                            CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promo) {
  CUtensorMap m{};
  const cuuint64_t dim[2] = {static_cast<cuuint64_t>(width), static_cast<cuuint64_t>(rows)};
  const cuuint64_t stride[1] = {static_cast<cuuint64_t>(width * 2)};
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

struct Launch {
  CUtensorMap smap, lmap;
  int box_bytes, box_rows, writers, rounds, loaders, epi, load_rounds;
  unsigned* arrived;
  CtaOut* out;
  unsigned n;
};
template <int STAMPS, int MODE, int STAGES>
static void launch_one(const Launch& l) {
  MB_CHECK(cudaFuncSetAttribute(k_round<STAMPS, MODE, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kSmemBytes));
  k_round<STAMPS, MODE, STAGES><<<l.n, 256, kSmemBytes>>>(l.smap, l.lmap, l.box_bytes, l.box_rows, l.writers,
                                                         l.rounds, l.loaders, l.epi, l.load_rounds, l.arrived,
                                                         l.out, l.n);
}
static void launch(int stamps, int mode, int stages, const Launch& l) {
  const int key = stamps * 8 + mode * 2 + (stages == 4 ? 1 : 0);
  switch (key) {
    case 0: launch_one<0, 0, 1>(l); break;
    case 1: launch_one<0, 0, 4>(l); break;
    case 2: launch_one<0, 1, 1>(l); break;
    case 3: launch_one<0, 1, 4>(l); break;
    case 4: launch_one<0, 2, 1>(l); break;
    case 5: launch_one<0, 2, 4>(l); break;
    case 6: launch_one<0, 3, 1>(l); break;
    case 7: launch_one<0, 3, 4>(l); break;
    case 8: launch_one<1, 0, 1>(l); break;
    case 9: launch_one<1, 0, 4>(l); break;
    case 10: launch_one<1, 1, 1>(l); break;
    case 11: launch_one<1, 1, 4>(l); break;
    case 12: launch_one<1, 2, 1>(l); break;
    case 13: launch_one<1, 2, 4>(l); break;
    case 14: launch_one<1, 3, 1>(l); break;
    default: launch_one<1, 3, 4>(l); break;
  }
}

int main(int argc, char** argv) {
  std::string out = "store_round.jsonl", mode = "full", swizzle = "64", promo = "128";
  int N = 1, L = 0, D = 4, rounds = 264, samples = 8, stamps = 0, epi = 1, load_rounds = 64;
  int row_bytes = 64, stride_bytes = 0, box_bytes = 8192, writers = 4;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    auto s = [&](const char* k, std::string* d) { if (a == k && i + 1 < argc) { *d = argv[++i]; return true; } return false; };
    if (s("--out", &out) || s("--mode", &mode) || s("--swizzle", &swizzle) || s("--promo", &promo)) continue;
    if (n("--ctas", &N) || n("--loaders", &L) || n("--stages", &D) || n("--rounds", &rounds) ||
        n("--samples", &samples) || n("--stamps", &stamps) || n("--epi", &epi) || n("--load-rounds", &load_rounds) ||
        n("--row-bytes", &row_bytes) || n("--stride-bytes", &stride_bytes) || n("--box-bytes", &box_bytes) ||
        n("--writers", &writers))
      continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  const int mode_i = mode == "full" ? 0 : mode == "fence" ? 1 : mode == "store" ? 2 : mode == "sts" ? 3 : -1;
  const CUtensorMapSwizzle sw = swizzle == "none" ? CU_TENSOR_MAP_SWIZZLE_NONE
                                : swizzle == "64" ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
  const CUtensorMapL2promotion pr = promo == "none" ? CU_TENSOR_MAP_L2_PROMOTION_NONE
                                    : promo == "64" ? CU_TENSOR_MAP_L2_PROMOTION_L2_64B
                                    : promo == "128" ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                                                     : CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  if (!stride_bytes) stride_bytes = row_bytes;
  if (mode_i < 0 || (D != 1 && D != 4) || (L != 0 && L != kLoadSlots) || (stamps != 0 && stamps != 1) ||
      (epi != 0 && epi != 1) || rounds <= kWarmRounds || (row_bytes != 64 && row_bytes != 128 && row_bytes != 256) ||
      box_bytes % row_bytes || box_bytes > kMaxBoxBytes || (mode_i != 2 && box_bytes != 8192) || writers < 0 ||
      writers > 4 || stride_bytes < row_bytes || stride_bytes % 16) {
    std::fprintf(stderr, "bad cell\n");
    return 2;
  }
  const MbDevice dev = mb_query_device();
  if (N < 1 || N > dev.sm_count) { std::fprintf(stderr, "ctas 1..SMs\n"); return 2; }
  const int box_rows = box_bytes / row_bytes;
  const std::uint64_t store_rows = static_cast<std::uint64_t>(N) * kMaxStages * box_rows;
  const std::size_t store_bytes = static_cast<std::size_t>(store_rows) * stride_bytes;
  const std::uint64_t load_rows = static_cast<std::uint64_t>(N) * kLoadRegionBoxes * kLoadRows;
  const std::size_t load_bytes = static_cast<std::size_t>(load_rows) * kLoadInner * 2;
  unsigned char* gdst = nullptr;
  void* gsrc = nullptr;
  MB_CHECK(cudaMalloc(&gdst, store_bytes));
  MB_CHECK(cudaMalloc(&gsrc, load_bytes));
  MB_CHECK(cudaMemset(gdst, 0, store_bytes));
  MB_CHECK(cudaMemset(gsrc, 1, load_bytes));
  Launch l{};
  l.smap = make_map(gdst, store_rows, row_bytes / 2, static_cast<std::uint64_t>(stride_bytes / 2), box_rows, sw, pr);
  l.lmap = make_map(gsrc, load_rows, kLoadInner, kLoadInner, kLoadRows, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  l.box_bytes = box_bytes; l.box_rows = box_rows; l.writers = writers; l.rounds = rounds; l.loaders = L;
  l.epi = epi; l.load_rounds = load_rounds; l.n = static_cast<unsigned>(N);
  MB_CHECK(cudaMalloc(&l.arrived, sizeof(unsigned)));
  MB_CHECK(cudaMalloc(&l.out, sizeof(CtaOut) * N));
  std::string err;
  auto run = [&](const Launch& x) {
    MB_CHECK(cudaMemset(x.arrived, 0, sizeof(unsigned)));
    MB_CHECK(cudaMemset(x.out, 0, sizeof(CtaOut) * N));
    launch(stamps, mode_i, D, x);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); return false; }
    if (mb_take_err()) { err = "mbarrier wait timeout"; return false; }
    return true;
  };
  if (L) {  // read-warm every CTA's load region (two passes), untimed
    Launch w = l;
    w.epi = 0; w.loaders = kLoadSlots; w.load_rounds = kLoadRegionBoxes / kLoadSlots;
    for (int k = 0; k < 2 && err.empty(); ++k) run(w);
  }
  std::vector<double> period, agg_bns, cpn, load_bpc;
  std::vector<double> off[4][6];
  for (int rep = 0; rep < samples + 1 && err.empty(); ++rep) {  // rep 0 discarded
    if (!run(l)) break;
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
        if (mode_i == 0 || mode_i == 2) total += nr * box_bytes;
        gmin = std::min(gmin, static_cast<double>(h[i].g0));
        gmax = std::max(gmax, static_cast<double>(h[i].g1));
        if (stamps) {
          const double a0 = static_cast<double>(h[i].t[0][0]);
          for (int w = 0; w < 4; ++w)
            for (int k = 0; k < 6; ++k) off[w][k].push_back((static_cast<double>(h[i].t[w][k]) - a0) / nr);
        }
      }
      if (L && h[i].lc1 > h[i].lc0)
        load_bpc.push_back(static_cast<double>(h[i].load_bytes) / static_cast<double>(h[i].lc1 - h[i].lc0));
    }
    if (epi && gmax > gmin) agg_bns.push_back(total / (gmax - gmin));
  }
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str()); return 1; }
  char fam[200];
  std::snprintf(fam, sizeof(fam), "store_round.%s.r%ds%sp%sw%d.b%d.W%d.N%d.L%d.D%d.s%d%s", mode.c_str(), row_bytes,
                swizzle.c_str(), promo.c_str(), stride_bytes, box_bytes, writers, N, L, D, stamps, epi ? "" : ".epi0");
  std::string tl = "[";
  for (int w = 0; w < 4; ++w) {
    tl += "[";
    for (int k = 0; k < 6; ++k) {
      char b[32];
      std::snprintf(b, sizeof(b), "%.1f%s", pct(off[w][k], 0.5), k < 5 ? "," : "");
      tl += b;
    }
    tl += w < 3 ? "]," : "]";
  }
  tl += "]";
  std::fprintf(f,
               "{\"schema\":\"cyclescope.microbench.store_round.v2\",\"family\":\"%s\",\"mode\":\"%s\",\"row_bytes\":%d,"
               "\"swizzle\":\"%s\",\"promo\":\"%s\",\"stride_bytes\":%d,\"box_bytes\":%d,\"writers\":%d,\"ctas\":%d,"
               "\"loaders\":%d,\"stages\":%d,\"stamps\":%d,\"epi\":%d,\"rounds\":%d,\"warm_rounds\":%d,"
               "\"period_median\":%.3f,\"period_p10\":%.3f,\"period_p90\":%.3f,\"period_n\":%zu,"
               "\"aggregate_store_bytes_per_ns_median\":%.3f,\"cycles_per_ns_median\":%.5f,"
               "\"load_bytes_per_cycle_median\":%.3f,\"timeline\":%s,"
               "\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,\"sm_count\":%d",
               fam, mode.c_str(), row_bytes, swizzle.c_str(), promo.c_str(), stride_bytes, box_bytes, writers, N, L, D,
               stamps, epi, rounds, kWarmRounds, pct(period, 0.5), pct(period, 0.10), pct(period, 0.90), period.size(),
               pct(agg_bns, 0.5), pct(cpn, 0.5), pct(load_bpc, 0.5), tl.c_str(), dev.name.c_str(), dev.sm_clock_khz,
               dev.sm_count);
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("%s period=%.1f load=%.1f B/cyc%s\n", fam, pct(period, 0.5), pct(load_bpc, 0.5),
              err.empty() ? "" : " ERR");
  cudaFree(gdst); cudaFree(gsrc); cudaFree(l.arrived); cudaFree(l.out);
  return err.empty() ? 0 : 1;
}
