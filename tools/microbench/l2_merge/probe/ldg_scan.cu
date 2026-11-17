// Ported verbatim from hopper-prototype tools/microbench/l2_dram_scan_microbench.cu at
// hopper-prototype 16c47f7426adbff493f81b156a178beec006949d (the revision that produced the
// M1 r4 scan behind the installed h100 delivered_bytes_per_cycle; sha256 736937ed...)
// except: (1) the nsm upper bound 132 is the device's multiProcessorCount queried at
// runtime; (2) clock_khz / mem_khz come from cudaDeviceGetAttribute (cudaDeviceProp
// .clockRate / .memoryClockRate are gone in CUDA 13). Protocol-v2 addendum 2026-11-05.
/*
 * l2_dram_scan_microbench.cu -- shared memory-system scan for H100 (sm_90a):
 * L2 / DRAM read and write throughput, and loaded / unloaded latency, as a
 * function of the number of active SMs (M1/M2 of full-gpu-plan-v2 §4.4).
 *
 * One CTA per SM: `nsm` CTAs are launched at once (<= 132, first wave, so
 * the hardware places them on distinct SMs; the per-CTA %smid is recorded
 * and checked on the host).  Every CTA records clock64 and %globaltimer at
 * the start and end of its own work, so each CTA yields bytes/cycle at its
 * own SM clock and the whole launch yields bytes/ns over the union window;
 * the ratio of the two is the effective SM clock (MHz) during the run.
 *
 * modes
 *   ld    : every CTA streams 16-byte loads (ld.global.cg.v4) over its region
 *   st    : every CTA streams 16-byte stores (st.global.cg.v4)
 *   chase : CTA `probe` runs a dependent pointer chase (one lane, 128 B
 *           random cyclic permutation over chase_mb) while the other CTAs
 *           stream (ld or st, chosen by `streamer`) until the probe sets a
 *           stop flag -> latency under fabric load.  nsm=1 gives unloaded.
 *   hist  : like chase with nsm CTAs launched but only `probe` works; the
 *           probe records every step's latency (clock64 deltas) -> per-step
 *           distribution (near/far partition probe).  HISTCHECK compares the
 *           sum of per-step deltas with the total chase time; if the ratio is
 *           not ~1 the per-step clock reads were not ordered after the data
 *           return and the histogram must be discarded.
 *
 * regions (stream modes)
 *   shared : all CTAs stream the same `footprint_mb` region, each starting
 *            at a rotated offset (L2-resident test when footprint << L2)
 *   slice  : the footprint is split into nsm/group disjoint slices; the
 *            `group` CTAs of a slice share it (group=1: one slice per CTA,
 *            streamed once -> DRAM test when footprint >> L2)
 * Each CTA touches `per_cta_mb` bytes (shared) or its whole slice (slice).
 * Between trials of a `slice` run the L2 is flushed by reading 256 MiB.
 * slice layout knobs (S3 round 2, M1 follow-up on the layout dependence):
 *   group=G         G consecutive CTAs read the same slice (GEMM-like shared
 *                   operand); group_rot=1 rotates each member's start by
 *                   slice/G so members are not on the same line at once
 *   slice_stride_mb=N  per-slice base stride in MB (default: slice size),
 *                   e.g. 2 to put every slice on a 2 MB boundary
 *   perm=1          random permutation of slice index over CTA groups
 *
 * usage: l2_dram_scan <mode> <nsm> <threads> <footprint_mb> <per_cta_mb>
 *          <trials> [region=shared|slice] [probe=N] [steps=N]
 *          [chase_mb=N] [streamer=ld|st] [unroll=8] [group=G]
 *          [group_rot=0|1] [slice_stride_mb=N] [perm=0|1]
 * output: META ..., PLACEMENT ..., per trial CTA lines, TRIAL summaries,
 *         RESULT medians over trials (trial 0 is warm-up, not reported).
 */
#include <algorithm>
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

#ifndef UNROLL
#define UNROLL 8
#endif
constexpr int kMaxThreads = 1024;
constexpr unsigned kLineBytes = 128;
constexpr unsigned kLineWords = kLineBytes / 4;

struct Rec {
  unsigned smid;
  unsigned flags;
  unsigned long long clk0, clk1, gt0, gt1;
  unsigned long long bytes;
  unsigned long long steps;
  unsigned long long chase_cycles;
};

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned smid_reg() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
__device__ __forceinline__ uint4 ld_cg(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_cg(uint4* p, uint4 v) {
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

struct StreamArgs {
  uint4* buf;
  unsigned long long region_len;     // elements (uint4) per region
  unsigned long long region_stride;  // 0 = shared region, else per-slice base
  unsigned long long rot;            // start rotation per CTA within its group
  unsigned group;                    // CTAs per slice (shared: nsm)
  const unsigned* slice_map;         // optional CTA -> slice index
  unsigned long long iters;          // iterations of UNROLL*T elements; 0 = until stop
  unsigned long long max_iters;      // cap for stop-flag mode
  volatile unsigned* stop;
  unsigned* sink;
  Rec* rec;
  unsigned seed;
  int probe;                         // CTA index that runs the chase (-1 none)
  const unsigned* chase;             // 128B-line permutation (index per line)
  unsigned steps;
  unsigned* hist;                    // per-step latencies (hist mode) or null
  unsigned long long settle_ns;      // chase: wait before probing
  unsigned long long smset[3];       // bit s set => SM s streams; all zero = every CTA
};
__device__ __forceinline__ bool sm_active(const StreamArgs& a, unsigned smid) {
  if ((a.smset[0] | a.smset[1] | a.smset[2]) == 0ULL) return true;
  return smid < 192 && ((a.smset[smid >> 6] >> (smid & 63)) & 1ULL);
}

template <bool kStore>
__device__ __forceinline__ void stream_body(const StreamArgs& a, Rec& r) {
  const unsigned T = blockDim.x;
  const unsigned group = a.group ? a.group : 1u;
  const unsigned slice =
      a.slice_map ? a.slice_map[blockIdx.x] : blockIdx.x / group;
  const unsigned long long base =
      a.region_stride ? a.region_stride * slice : 0ULL;
  const unsigned long long start =
      (a.rot * (blockIdx.x % group)) % a.region_len;
  uint4* region = a.buf + base;
  unsigned long long pos = start + threadIdx.x;
  const unsigned long long step = (unsigned long long)UNROLL * T;
  uint4 acc = make_uint4(0, 0, 0, 0);
  const uint4 val = make_uint4(a.seed ^ threadIdx.x, blockIdx.x, 0x9e3779b9u, 1u);
  unsigned long long done = 0;
  __syncthreads();
  unsigned long long c0 = clock64(), g0 = gtimer();
  const unsigned long long limit = a.iters ? a.iters : a.max_iters;
#pragma unroll 1
  for (unsigned long long it = 0; it < limit; ++it) {
    if (a.iters == 0 && (it & 7ULL) == 0) {
      if (*a.stop) break;
      if (gtimer() - g0 > 4000000000ULL) { r.flags = 2; break; }  // 4 s safety cap
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      uint4* p = region + pos + (unsigned long long)u * T;
      if (kStore) {
        st_cg(p, val);
      } else {
        const uint4 v = ld_cg(p);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
      }
    }
    pos += step;
    if (pos >= a.region_len) pos -= a.region_len;
    ++done;
  }
  if (kStore) __threadfence();  // this thread's stores performed at L2
  const unsigned live = kStore ? 0u : (acc.x ^ acc.y ^ acc.z ^ acc.w);
  const int any = __syncthreads_or(live == 0x5bd1e995u);
  if (threadIdx.x == 0) {
    r.clk1 = clock64();
    r.gt1 = gtimer();
    r.clk0 = c0;
    r.gt0 = g0;
    r.bytes = done * step * sizeof(uint4);
    r.steps = done;
    if (any) a.sink[blockIdx.x] = live;
  }
}

// Dependent chase over 128 B lines; returns the index reached (keeps the
// chain alive).  Per-step latencies go to `hist` when non-null.
__device__ __forceinline__ unsigned chase_body(const unsigned* chase,
                                               unsigned steps, unsigned* hist,
                                               unsigned long long& cycles) {
  unsigned idx = 0;
  unsigned long long t0 = clock64();
  unsigned long long tp = t0;
#pragma unroll 1
  for (unsigned s = 0; s < steps; ++s) {
    const unsigned* p = chase + (unsigned long long)idx * kLineWords;
    unsigned v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    unsigned dep;
    asm volatile("add.u32 %0, %1, 0;" : "=r"(dep) : "r"(v) : "memory");
    if (hist) {
      unsigned long long t1;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
      hist[s] = (unsigned)(t1 - tp);
      tp = t1;
    }
    idx = dep;
  }
  cycles = clock64() - t0;
  return idx;
}

template <bool kStreamerStore>
__global__ void __launch_bounds__(kMaxThreads, 1) scan_kernel(StreamArgs a) {
  Rec& r = a.rec[blockIdx.x];
  if (threadIdx.x == 0) {
    r.smid = smid_reg();
    r.flags = 0;
    r.bytes = 0; r.steps = 0; r.chase_cycles = 0;
    r.clk0 = clock64(); r.gt0 = gtimer(); r.clk1 = r.clk0; r.gt1 = r.gt0;
  }
  if ((int)blockIdx.x == a.probe) {
    if (threadIdx.x == 0) {
      const unsigned long long g_begin = gtimer();
      while (gtimer() - g_begin < a.settle_ns) { }
      r.flags = 1;
      unsigned long long cyc = 0;
      r.clk0 = clock64(); r.gt0 = gtimer();
      const unsigned reached = chase_body(a.chase, a.steps, a.hist, cyc);
      r.clk1 = clock64(); r.gt1 = gtimer();
      r.chase_cycles = cyc;
      r.steps = a.steps;
      __threadfence();
      *a.stop = 1u;
      if (reached == 0xffffffffu) a.sink[blockIdx.x] = reached;
    }
    return;
  }
  if (a.probe >= 0 && a.hist != nullptr) return;  // hist mode: others idle
  if (!sm_active(a, smid_reg())) return;           // smset: this SM stays idle
  stream_body<kStreamerStore>(a, r);
}

__global__ void flush_kernel(const uint4* scratch, unsigned long long elems,
                             unsigned* sink) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (; i < elems; i += stride) {
    const uint4 v = ld_cg(scratch + i);
    acc ^= v.x ^ v.w;
  }
  if (acc == 0x5bd1e995u) sink[0] = acc;
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static unsigned long long arg_kv(int argc, char** argv, const char* key,
                                 unsigned long long def) {
  const size_t n = strlen(key);
  for (int i = 7; i < argc; ++i)
    if (!strncmp(argv[i], key, n) && argv[i][n] == '=') return strtoull(argv[i] + n + 1, nullptr, 10);
  return def;
}
static std::string arg_kvs(int argc, char** argv, const char* key, const char* def) {
  const size_t n = strlen(key);
  for (int i = 7; i < argc; ++i)
    if (!strncmp(argv[i], key, n) && argv[i][n] == '=') return std::string(argv[i] + n + 1);
  return def;
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s <ld|st|chase|hist> <nsm> <threads> <footprint_mb> "
            "<per_cta_mb> <trials> [region=shared|slice] [probe=N] [steps=N] "
            "[chase_mb=N] [streamer=ld|st] [settle_us=N]\n", argv[0]);
    return 1;
  }
  const std::string mode = argv[1];
  const int nsm = atoi(argv[2]), threads = atoi(argv[3]);
  const unsigned long long footprint_mb = strtoull(argv[4], nullptr, 10);
  const unsigned long long per_cta_mb = strtoull(argv[5], nullptr, 10);
  const int trials = atoi(argv[6]);
  const std::string region = arg_kvs(argc, argv, "region", "shared");
  const int probe = (mode == "chase" || mode == "hist") ? (int)arg_kv(argc, argv, "probe", 0) : -1;
  const unsigned steps = (unsigned)arg_kv(argc, argv, "steps", 4096);
  const unsigned long long chase_mb = arg_kv(argc, argv, "chase_mb", 4);
  const std::string streamer = arg_kvs(argc, argv, "streamer", "ld");
  const unsigned long long settle_us = arg_kv(argc, argv, "settle_us", 50);
  const bool chase_warm = arg_kv(argc, argv, "warm", 1) != 0;  // chase lines L2-resident?
  const unsigned group = (unsigned)arg_kv(argc, argv, "group", 1);
  const bool group_rot = arg_kv(argc, argv, "group_rot", 0) != 0;
  const unsigned long long slice_stride_mb = arg_kv(argc, argv, "slice_stride_mb", 0);
  const bool perm_slices = arg_kv(argc, argv, "perm", 0) != 0;
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int clock_khz = 0, mem_khz = 0;  // cudaDeviceProp.clockRate is gone in CUDA 13
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  if (nsm < 1 || nsm > prop.multiProcessorCount || threads < 32 || threads > kMaxThreads || threads % 32 ||
      trials < 1 || footprint_mb == 0 || (region != "shared" && region != "slice") ||
      (mode != "ld" && mode != "st" && mode != "chase" && mode != "hist") ||
      probe >= nsm || group < 1 || (region == "slice" && nsm % group != 0)) {
    fprintf(stderr, "bad arguments\n");
    return 1;
  }
  const bool store_stream = (mode == "st") || (mode == "chase" && streamer == "st");
  const unsigned long long T = threads;
  const unsigned long long step = (unsigned long long)UNROLL * T;  // elements per iteration
  const unsigned long long footprint_elems = footprint_mb << 20 >> 4;
  unsigned long long region_len, region_stride, rot, iters;
  unsigned cta_group = group, n_slices = 1;
  if (region == "shared") {
    region_len = footprint_elems / step * step;
    region_stride = 0;
    cta_group = (unsigned)nsm;
    rot = (region_len / (unsigned long long)nsm) / step * step;
    iters = (per_cta_mb << 20 >> 4) / step;
  } else {
    n_slices = (unsigned)nsm / group;
    region_len = (footprint_elems / (unsigned long long)n_slices) / step * step;
    region_stride = std::max(region_len, (slice_stride_mb << 20 >> 4));
    rot = group_rot ? (region_len / (unsigned long long)group) / step * step : 0;
    iters = per_cta_mb ? (per_cta_mb << 20 >> 4) / step
                       : region_len / step;  // 0 = one pass over the slice
  }
  if (region_len == 0 || (iters == 0 && mode != "hist")) { fprintf(stderr, "footprint too small for threads*unroll\n"); return 1; }
  // smset=a-b,c,d-e : only CTAs whose %smid is in the set stream (others idle)
  unsigned long long smset[3] = {0, 0, 0};
  int smset_count = 0;
  {
    const std::string spec = arg_kvs(argc, argv, "smset", "");
    size_t pos = 0;
    while (pos < spec.size()) {
      size_t comma = spec.find(',', pos);
      if (comma == std::string::npos) comma = spec.size();
      const std::string item = spec.substr(pos, comma - pos);
      const size_t dash = item.find('-');
      const int lo = atoi(item.c_str());
      const int hi = dash == std::string::npos ? lo : atoi(item.c_str() + dash + 1);
      for (int sm = lo; sm <= hi && sm < 192; ++sm) {
        if (!((smset[sm >> 6] >> (sm & 63)) & 1ULL)) ++smset_count;
        smset[sm >> 6] |= 1ULL << (sm & 63);
      }
      pos = comma + 1;
    }
  }
  const bool stop_mode = (mode == "chase");
  const unsigned long long max_iters = stop_mode ? (1ULL << 40) : 0;

  printf("META mode=%s nsm=%d threads=%d footprint_mb=%llu per_cta_mb=%llu trials=%d "
         "region=%s probe=%d steps=%u chase_mb=%llu warm=%d streamer=%s unroll=%d "
         "region_len_elems=%llu iters=%llu smset_count=%d group=%u group_rot=%d "
         "slice_stride_elems=%llu n_slices=%u perm=%d "
         "gpu=%s sms=%d l2_bytes=%d clock_khz=%d mem_khz=%d\n",
         mode.c_str(), nsm, threads, footprint_mb, per_cta_mb, trials, region.c_str(),
         probe, steps, chase_mb, (int)chase_warm, streamer.c_str(), UNROLL, region_len, iters, smset_count,
         cta_group, (int)group_rot, region_stride, n_slices, (int)perm_slices,
         prop.name, prop.multiProcessorCount, prop.l2CacheSize, clock_khz,
         mem_khz);

  uint4* buf = nullptr;
  const unsigned long long laid_out_elems =
      region_stride ? region_stride * (n_slices - 1) + region_len : footprint_elems;
  const size_t buf_bytes = (size_t)std::max<unsigned long long>(
      std::max(footprint_elems, laid_out_elems), step) * sizeof(uint4);
  CUDA_CHECK(cudaMalloc(&buf, buf_bytes));
  CUDA_CHECK(cudaMemset(buf, 0x11, buf_bytes));
  // optional CTA -> slice permutation (deterministic xorshift, printed)
  unsigned* slice_map = nullptr;
  if (perm_slices && region == "slice") {
    std::vector<unsigned> slice_perm(n_slices);
    for (unsigned i = 0; i < n_slices; ++i) slice_perm[i] = i;
    unsigned long long prng = 0x2545f4914f6cdd1dULL;
    for (unsigned i = n_slices - 1; i > 0; --i) {
      prng ^= prng << 13; prng ^= prng >> 7; prng ^= prng << 17;
      std::swap(slice_perm[i], slice_perm[prng % (i + 1)]);
    }
    std::vector<unsigned> map_host(nsm);
    for (int b = 0; b < nsm; ++b) map_host[b] = slice_perm[b / group];
    printf("SLICEMAP");
    for (int b = 0; b < nsm; ++b) printf(" %u", map_host[b]);
    printf("\n");
    CUDA_CHECK(cudaMalloc(&slice_map, (size_t)nsm * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(slice_map, map_host.data(), (size_t)nsm * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
  }
  // chase permutation over 128 B lines (Sattolo: one cycle)
  const unsigned long long lines = (chase_mb << 20) / kLineBytes;
  std::vector<unsigned> perm(lines);
  for (unsigned long long i = 0; i < lines; ++i) perm[i] = (unsigned)i;
  unsigned long long rng = 0x9e3779b97f4a7c15ULL;
  for (unsigned long long i = lines - 1; i > 0; --i) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    const unsigned long long j = rng % i;
    std::swap(perm[i], perm[j]);
  }
  std::vector<unsigned> chase_host(lines * kLineWords, 0u);
  for (unsigned long long i = 0; i < lines; ++i) chase_host[i * kLineWords] = perm[i];
  unsigned* chase = nullptr;
  CUDA_CHECK(cudaMalloc(&chase, chase_host.size() * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(chase, chase_host.data(), chase_host.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
  unsigned* hist = nullptr;
  if (mode == "hist") CUDA_CHECK(cudaMalloc(&hist, (size_t)steps * sizeof(unsigned)));
  unsigned* stop = nullptr;
  unsigned* sink = nullptr;
  Rec* rec = nullptr;
  CUDA_CHECK(cudaMalloc(&stop, sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&sink, (size_t)nsm * sizeof(unsigned) + 4096));
  CUDA_CHECK(cudaMalloc(&rec, (size_t)nsm * sizeof(Rec)));
  uint4* scratch = nullptr;
  const unsigned long long scratch_elems = (256ULL << 20) >> 4;
  CUDA_CHECK(cudaMalloc(&scratch, scratch_elems * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(scratch, 0, scratch_elems * sizeof(uint4)));

  StreamArgs a{};
  a.buf = buf; a.region_len = region_len; a.region_stride = region_stride; a.rot = rot;
  a.iters = stop_mode ? 0 : iters; a.max_iters = max_iters; a.stop = stop; a.sink = sink;
  a.rec = rec; a.probe = probe; a.chase = chase; a.steps = steps; a.hist = hist;
  a.settle_ns = settle_us * 1000ULL;
  a.smset[0] = smset[0]; a.smset[1] = smset[1]; a.smset[2] = smset[2];
  a.group = cta_group; a.slice_map = slice_map;

  std::vector<Rec> recs(nsm);
  std::vector<double> agg_bpns, agg_window_ns, med_bpc, med_mhz, chase_cpstep, chase_ns_step;
  for (int t = 0; t <= trials; ++t) {
    if (region == "slice" || mode == "chase" || mode == "hist") {
      flush_kernel<<<1056, 256>>>(scratch, scratch_elems, sink);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      if ((mode == "chase" || mode == "hist") && chase_warm) {
        // warm=1: bring the chase lines back into L2 after the flush (L2-hit
        // chase); warm=0 leaves them cold (DRAM chase, every step a miss as
        // long as steps < lines)
        flush_kernel<<<1056, 256>>>(reinterpret_cast<const uint4*>(chase),
                                   chase_host.size() * sizeof(unsigned) / 16, sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }
    CUDA_CHECK(cudaMemset(stop, 0, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(rec, 0, (size_t)nsm * sizeof(Rec)));
    a.seed = 0x1234u + t;
    if (store_stream) scan_kernel<true><<<nsm, threads>>>(a);
    else scan_kernel<false><<<nsm, threads>>>(a);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(recs.data(), rec, (size_t)nsm * sizeof(Rec), cudaMemcpyDeviceToHost));
    if (t == 0) {
      std::vector<unsigned> sm;
      for (auto& r : recs) sm.push_back(r.smid);
      std::sort(sm.begin(), sm.end());
      const size_t distinct = std::unique(sm.begin(), sm.end()) - sm.begin();
      printf("PLACEMENT distinct_sm=%zu nsm=%d\n", distinct, nsm);
      continue;
    }
    unsigned long long total = 0, gmin = ~0ULL, gmax = 0;
    std::vector<double> bpc, mhz;
    for (int c = 0; c < nsm; ++c) {
      const Rec& r = recs[c];
      const double dc = (double)(r.clk1 - r.clk0), dn = (double)(r.gt1 - r.gt0);
      printf("CTA trial=%d cta=%d smid=%u flags=%u cycles=%llu ns=%llu bytes=%llu iters=%llu "
             "chase_cycles=%llu mhz=%.1f\n", t, c, r.smid, r.flags, (unsigned long long)dc,
             (unsigned long long)dn, r.bytes, r.steps, r.chase_cycles, dn > 0 ? dc / dn * 1e3 : 0.0);
      if (c == probe) {
        if (r.chase_cycles) {
          chase_cpstep.push_back((double)r.chase_cycles / steps);
          chase_ns_step.push_back(dn / steps);
        }
        continue;
      }
      if (r.bytes == 0) continue;  // idle (smset) or non-streaming CTA
      total += r.bytes;
      gmin = std::min(gmin, r.gt0); gmax = std::max(gmax, r.gt1);
      if (dc > 0) bpc.push_back(r.bytes / dc);
      if (dn > 0) mhz.push_back(dc / dn * 1e3);
    }
    const double window = gmax > gmin ? (double)(gmax - gmin) : 0.0;
    const double agg = window > 0 ? total / window : 0.0;
    agg_bpns.push_back(agg); agg_window_ns.push_back(window);
    med_bpc.push_back(median(bpc)); med_mhz.push_back(median(mhz));
    printf("TRIAL %d total_bytes=%llu window_ns=%.0f agg_B_per_ns=%.3f per_sm_B_per_cycle_med=%.3f "
           "mhz_med=%.1f active_sms=%zu chase_cyc_per_step=%.2f chase_ns_per_step=%.2f\n", t, total, window, agg,
           median(bpc), median(mhz), bpc.size(), chase_cpstep.empty() ? 0.0 : chase_cpstep.back(),
           chase_ns_step.empty() ? 0.0 : chase_ns_step.back());
    if (mode == "hist" && t == trials) {
      std::vector<unsigned> lat(steps);
      CUDA_CHECK(cudaMemcpy(lat.data(), hist, (size_t)steps * sizeof(unsigned), cudaMemcpyDeviceToHost));
      unsigned long long sum = 0;
      std::vector<unsigned long long> buckets(512, 0);
      for (unsigned s = 1; s < steps; ++s) {  // step 0 includes the first-use effects
        sum += lat[s];
        buckets[std::min<unsigned>(lat[s] / 8, 511)]++;
      }
      for (size_t b = 0; b < buckets.size(); ++b)
        if (buckets[b]) printf("HIST cycles_ge=%zu count=%llu\n", b * 8, buckets[b]);
      printf("HISTCHECK sum=%llu total=%llu ratio=%.4f\n", sum, recs[probe].chase_cycles,
             recs[probe].chase_cycles ? (double)sum / recs[probe].chase_cycles : 0.0);
    }
  }
  printf("RESULT agg_B_per_ns=%.3f per_sm_B_per_cycle=%.3f mhz=%.1f window_ns=%.0f "
         "chase_cyc_per_step=%.2f chase_ns_per_step=%.2f\n", median(agg_bpns), median(med_bpc),
         median(med_mhz), median(agg_window_ns), median(chase_cpstep), median(chase_ns_step));
  CUDA_CHECK(cudaFree(scratch)); CUDA_CHECK(cudaFree(rec)); CUDA_CHECK(cudaFree(sink));
  CUDA_CHECK(cudaFree(stop)); if (hist) CUDA_CHECK(cudaFree(hist));
  CUDA_CHECK(cudaFree(chase)); CUDA_CHECK(cudaFree(buf));
  if (slice_map) CUDA_CHECK(cudaFree(slice_map));
  return 0;
}
