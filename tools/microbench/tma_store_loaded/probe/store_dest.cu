// tma_store_loaded v5 probe: does the destination of a TMA store decide when the store completes?
// Grid = N CTAs, one per SM, 128 threads.  Round = bar.sync ; thread 0: UTMASTG or UTMAREDG.add of one
// box + commit_group + wait_group.read <STAGES-1> ; bar.sync.  The source boxes are written and fenced
// once.  Destination: a ring of 4 boxes per CTA (always the same L2 lines) or a fresh box every round;
// the host clears the destination and then writes an eviction buffer before every launch.
// Helpers and the tensor-map builder are the ones of store_round.cu (copied verbatim).
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


__device__ __forceinline__ void tma_reduce_add(const CUtensorMap* map, int x, int y, std::uint32_t src) {
  asm volatile(
      "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
          reinterpret_cast<std::uint64_t>(map)),
      "r"(x), "r"(y), "r"(src)
      : "memory");
}

struct DestOut {
  long long c0, c1;
  unsigned smid, pad;
};

// RED 0: UTMASTG, 1: UTMAREDG .add.  STAGES: wait_group.read <STAGES-1> after every commit.
// fresh 0: every CTA cycles over its own 4 destination boxes (the ring of store_round);
// fresh 1: round r of CTA idx writes box idx * rounds + r, never written before in this launch.
template <int RED, int STAGES>
__global__ void __launch_bounds__(128, 1) k_dest(const __grid_constant__ CUtensorMap smap, int box_bytes, int box_rows,
                                                 int rounds, int fresh, unsigned* arrived, DestOut* out,
                                                 unsigned nparticipants, const unsigned char* active) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const std::uint32_t base = (mb_smem_u32(smem) + 1023u) & ~1023u;
  const int idx = static_cast<int>(blockIdx.x);
  for (int i = threadIdx.x * 16; i < kMaxStages * box_bytes; i += 128 * 16) sts128(base + i, 0u);
  fence_async_shared();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(arrived, 1u);
    while (*((volatile unsigned*)arrived) < nparticipants) { }
  }
  __syncthreads();
  // placement cells: a CTA whose SM is not in the active set takes part in the start handshake only
  if (active && !active[mb_smid() & 255u]) {
    if (threadIdx.x == 0) { out[idx].c0 = 0; out[idx].c1 = 0; out[idx].smid = mb_smid(); out[idx].pad = 1; }
    return;
  }
  long long c0 = 0;
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    if (r == kWarmRounds) c0 = static_cast<long long>(mb_clock());
    const int b = r % kMaxStages;
    const int box = fresh ? idx * rounds + r : idx * kMaxStages + b;
    bar_sync_epilogue();
    if (threadIdx.x == 0) {
      if (RED) tma_reduce_add(&smap, 0, box * box_rows, base + b * box_bytes);
      else tma_store(&smap, 0, box * box_rows, base + b * box_bytes);
      bulk_commit();
      bulk_wait_read<STAGES - 1>();
    }
    bar_sync_epilogue();
  }
  const long long c1 = static_cast<long long>(mb_clock());
  if (threadIdx.x == 0) {
    bulk_wait_read<0>();
    out[idx].c0 = c0; out[idx].c1 = c1; out[idx].smid = mb_smid();
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


#define DEST_KERNELS(X) X(0, 2) X(0, 4) X(1, 2) X(1, 4)
struct DL { CUtensorMap smap; int box_bytes, box_rows, rounds, fresh; unsigned* arrived; DestOut* out; unsigned n;
            const unsigned char* active; };
static bool launch(int red, int stages, const DL& l) {
#define X(R, S)                                                                                              \
  if (red == R && stages == S) {                                                                             \
    MB_CHECK(cudaFuncSetAttribute(k_dest<R, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));   \
    k_dest<R, S><<<l.n, 128, kSmemBytes>>>(l.smap, l.box_bytes, l.box_rows, l.rounds, l.fresh, l.arrived,    \
                                           l.out, l.n, l.active);                                                      \
    return true;                                                                                             \
  }
  DEST_KERNELS(X)
#undef X
  return false;
}

int main(int argc, char** argv) {
  std::string out = "store_dest.jsonl", active_file, tag;
  int N = 1, red = 0, stages = 2, fresh = 0, rounds = 264, samples = 8, box_bytes = 16384, evict_mib = 2048;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    if (a == "--out" && i + 1 < argc) { out = argv[++i]; continue; }
    if (a == "--active-file" && i + 1 < argc) { active_file = argv[++i]; continue; }
    if (a == "--tag" && i + 1 < argc) { tag = argv[++i]; continue; }
    if (n("--ctas", &N) || n("--red", &red) || n("--stages", &stages) || n("--fresh", &fresh) || n("--rounds", &rounds) ||
        n("--samples", &samples) || n("--box-bytes", &box_bytes) || n("--evict-mib", &evict_mib))
      continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    return 2;
  }
  const int row_bytes = 128;
  const MbDevice dev = mb_query_device();
  if (N < 1 || N > dev.sm_count || rounds <= kWarmRounds || box_bytes % row_bytes || box_bytes > kMaxBoxBytes) {
    std::fprintf(stderr, "bad cell\n");
    return 2;
  }
  const int box_rows = box_bytes / row_bytes;
  const std::uint64_t boxes = static_cast<std::uint64_t>(N) * (fresh ? rounds : kMaxStages);
  const std::uint64_t rows = boxes * box_rows;
  const std::size_t bytes = static_cast<std::size_t>(rows) * row_bytes;
  unsigned char* gdst = nullptr;
  unsigned char* evict = nullptr;
  MB_CHECK(cudaMalloc(&gdst, bytes));
  if (evict_mib > 0) MB_CHECK(cudaMalloc(&evict, static_cast<std::size_t>(evict_mib) << 20));
  DL l{};
  l.smap = make_map(gdst, rows, row_bytes / 2, static_cast<std::uint64_t>(row_bytes / 2), box_rows,
                    CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  l.box_bytes = box_bytes; l.box_rows = box_rows; l.rounds = rounds; l.fresh = fresh; l.n = static_cast<unsigned>(N);
  MB_CHECK(cudaMalloc(&l.arrived, sizeof(unsigned)));
  MB_CHECK(cudaMalloc(&l.out, sizeof(DestOut) * N));
  int n_active = N;
  if (!active_file.empty()) {  // whitespace-separated SM ids; the grid must cover every SM
    std::vector<unsigned char> act(256, 0);
    FILE* af = std::fopen(active_file.c_str(), "r");
    if (!af || N != dev.sm_count) { std::fprintf(stderr, "active file needs --ctas <all SMs>\n"); return 2; }
    n_active = 0;
    for (int v; std::fscanf(af, "%d", &v) == 1;) if (v >= 0 && v < 256 && !act[v]) { act[v] = 1; ++n_active; }
    std::fclose(af);
    unsigned char* dact = nullptr;
    MB_CHECK(cudaMalloc(&dact, 256));
    MB_CHECK(cudaMemcpy(dact, act.data(), 256, cudaMemcpyHostToDevice));
    l.active = dact;
  }
  std::vector<double> period;
  std::vector<double> per_sm[256];
  std::string err;
  for (int rep = 0; rep < samples + 1 && err.empty(); ++rep) {  // rep 0 discarded
    MB_CHECK(cudaMemset(gdst, 0, bytes));
    // push the destination out of L2 before every launch: write a buffer much larger than L2
    if (evict) MB_CHECK(cudaMemset(evict, rep, static_cast<std::size_t>(evict_mib) << 20));
    MB_CHECK(cudaMemset(l.arrived, 0, sizeof(unsigned)));
    MB_CHECK(cudaMemset(l.out, 0, sizeof(DestOut) * N));
    MB_CHECK(cudaDeviceSynchronize());
    if (!launch(red, stages, l)) { std::fprintf(stderr, "no kernel\n"); return 2; }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    if (rep == 0) continue;
    std::vector<DestOut> h(N);
    MB_CHECK(cudaMemcpy(h.data(), l.out, sizeof(DestOut) * N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
      if (h[i].pad) continue;  // inactive SM
      const double pr = static_cast<double>(h[i].c1 - h[i].c0) / static_cast<double>(rounds - kWarmRounds);
      period.push_back(pr);
      per_sm[h[i].smid & 255u].push_back(pr);
    }
  }
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str()); return 1; }
  std::fprintf(f,
               "{\"schema\":\"cyclescope.microbench.store_dest.v1\",\"red\":%d,\"stages\":%d,\"fresh\":%d,\"ctas\":%d,"
               "\"box_bytes\":%d,\"rounds\":%d,\"evict_mib\":%d,\"period_median\":%.3f,\"period_p10\":%.3f,"
               "\"period_p90\":%.3f,\"period_n\":%zu,\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,\"sm_count\":%d",
               red, stages, fresh, N, box_bytes, rounds, evict_mib, pct(period, 0.5), pct(period, 0.10),
               pct(period, 0.90), period.size(), dev.name.c_str(), dev.sm_clock_khz, dev.sm_count);
  std::fprintf(f, ",\"tag\":\"%s\",\"active\":%d,\"per_sm\":[", tag.c_str(), n_active);
  {
    bool first = true;
    for (int sm = 0; sm < 256; ++sm) {
      if (per_sm[sm].empty()) continue;
      std::fprintf(f, "%s[%d,%.1f]", first ? "" : ",", sm, pct(per_sm[sm], 0.5));
      first = false;
    }
  }
  std::fprintf(f, "]");
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("store_dest red=%d stages=%d fresh=%d N=%d box=%d period=%.1f%s\n", red, stages, fresh, N, box_bytes,
              pct(period, 0.5), err.empty() ? "" : " ERR");
  cudaFree(gdst); if (evict) cudaFree(evict); cudaFree(l.arrived); cudaFree(l.out);
  return err.empty() ? 0 : 1;
}
