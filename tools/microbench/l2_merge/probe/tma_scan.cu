/*
 * tma_scan_microbench.cu -- M6: multi-SM TMA tensor-load (UTMALDG.2D)
 * aggregate throughput from L2 / DRAM (H100, sm_90a), the TMA-path twin of
 * l2_dram_scan_microbench.cu (same output lines, same runner parsing).
 *
 * One CTA per SM (nsm CTAs <= 132, first wave), one issuing thread per CTA,
 * a ring of `stages` shared-memory boxes (64 bf16 columns = 128 B rows x
 * box_rows rows) each with an mbarrier; tile i goes to stage i % stages after
 * that stage's previous load completed.  Every CTA records clock64 and
 * %globaltimer around its stream -> bytes/cycle per SM, bytes/ns aggregate
 * over the union window (RESULT line, like l2_dram_scan).
 *
 * regions
 *   slice   : footprint split into nsm/group disjoint slices (rows); the
 *             `group` consecutive CTAs of a slice read the same tiles
 *             (group=1: one slice per CTA); group_rot=1 rotates each
 *             member's start tile by slice/group (members on different
 *             tiles), group_rot=0 lockstep (same tile at the same time)
 *   multicast=k : cluster of k CTAs; rank 0 issues .multicast::cluster
 *             loads of the cluster's slice into all k CTAs' shared memory
 *             (one L2 read delivered to k SMs); a cluster barrier every
 *             `stages` tiles keeps the ring phases aligned.  Bytes are
 *             counted per receiving CTA (SM-side delivered bytes).
 *   sync_start=1 : every CTA waits for one globaltimer target published by
 *             CTA 0 before streaming (aligned start within the 32 ns timer
 *             resolution); shift_cycles=N additionally delays group member m
 *             by m * N SM cycles (controlled inter-request gap between the
 *             members' reads of the same tile; implies sync_start=1).  The
 *             CTA lines report start_ns = this CTA's start - earliest start.
 *             nosync=1 applies shift_cycles from each CTA's own dispatch
 *             instead (no globaltimer barrier; closed-loop form).
 *   far=1   : the members of a group are the CTAs nsm/group apart in block
 *             index (slice = block index mod nsm/group) instead of `group`
 *             consecutive ones.
 *   table=<file> : the slice is chosen by SM id.  The file has one header
 *             line `slices <n>` and one line `<sm> <slice> <member>` per SM
 *             id; every SM must run exactly one CTA (nsm = SM count,
 *             group = 1).  SMs with the same slice read the same tiles;
 *             member orders them for shift_cycles; member 65535 = the SM's
 *             CTA stays idle (no loads, bytes 0).
 *   log=<file> : write the output lines to the file instead of stdout.
 * Each CTA streams per_cta_mb bytes cyclically over its slice (0 = one pass).
 * Between trials of a DRAM-sized footprint the L2 is flushed (256 MiB read).
 *
 * usage: tma_scan <nsm> <footprint_mb> <per_cta_mb> <trials> [box_rows=128]
 *          [stages=4] [group=G] [group_rot=0|1] [multicast=k]
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(err_),      \
              __FILE__, __LINE__);                                         \
      exit(2);                                                             \
    }                                                                      \
  } while (0)

constexpr int kCols = 64;          // bf16 columns -> 128 B rows
constexpr int kMaxStages = 8;
constexpr unsigned long long kSyncMarginNs = 100000;  // 100 us: covers the first-wave dispatch spread

struct Rec {
  unsigned smid, flags;
  unsigned long long clk0, clk1, gt0, gt1, bytes, steps, chase_cycles;
};

struct Args {
  unsigned long long slice_rows;   // rows per slice
  unsigned long long tiles;        // tiles per CTA to stream
  unsigned long long slice_tiles;  // tiles in a slice (cyclic)
  unsigned group;                  // CTAs per slice
  unsigned rot_tiles;              // start rotation per member (tiles)
  unsigned multicast;              // cluster size for multicast (0 = off)
  unsigned long long shift_cycles; // member m starts m * shift_cycles after the aligned start
  unsigned long long* start_ns;    // sync_start: globaltimer value every CTA waits for (0 = unset)
  unsigned far_slices;             // far=1: number of slices (0 = consecutive grouping)
  const unsigned short* sm_slice;  // table: slice by SM id (nullptr = by block index)
  const unsigned short* sm_member; // table: member by SM id
  int box_rows, stages;
  Rec* rec;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
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
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
}

__global__ void __launch_bounds__(32, 1)
tma_scan_kernel(const __grid_constant__ CUtensorMap tmap, Args a) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[kMaxStages];
  Rec& r = a.rec[blockIdx.x];
  if (threadIdx.x != 0) return;
  unsigned rank = 0;
  if (a.multicast > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const uint32_t box_bytes = static_cast<uint32_t>(a.box_rows) * kCols * 2u;
  for (int s = 0; s < a.stages; ++s)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&bars[s])));
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if (a.multicast > 1) cluster_sync();
  unsigned slice = a.multicast > 1 ? blockIdx.x / a.multicast : blockIdx.x / a.group;
  unsigned member = a.multicast > 1 ? 0u : blockIdx.x % a.group;
  if (a.far_slices != 0) { slice = blockIdx.x % a.far_slices; member = blockIdx.x / a.far_slices; }
  if (a.sm_slice != nullptr) { const unsigned sm = smid_reg(); slice = a.sm_slice[sm]; member = a.sm_member[sm]; }
  const unsigned long long row_base = slice * a.slice_rows;
  unsigned long long tile = (static_cast<unsigned long long>(a.rot_tiles) * member) % a.slice_tiles;
  const uint64_t tmap_addr = reinterpret_cast<uint64_t>(&tmap);
  const uint16_t mask = static_cast<uint16_t>((1u << a.multicast) - 1u);
  r.smid = smid_reg(); r.flags = 0;
  if (a.start_ns != nullptr && blockIdx.x == 0) {
    __threadfence();
    atomicExch(reinterpret_cast<unsigned long long*>(a.start_ns), gtimer() + kSyncMarginNs);
  }
  if (member == 0xFFFFu) {  // idle SM (table): publishes the start target if it is CTA 0, loads nothing
    r.clk0 = r.clk1 = r.gt0 = r.gt1 = 0; r.bytes = 0; r.steps = 0; r.chase_cycles = 0;
    return;
  }
  if (a.start_ns != nullptr) {
    // Aligned start: CTA 0 publishes a globaltimer target (kSyncMarginNs
    // ahead), everyone spins on %globaltimer until it is reached (register
    // read: alignment within the 32 ns timer resolution), then member m of
    // a group waits m * shift_cycles on its own clock64.
    // volatile re-load: nvcc dropped the back-edge of an asm-volatile ld loop
    // here (the SASS had a single LDG, so non-zero CTAs never waited).
    volatile unsigned long long* flag = a.start_ns;
    unsigned long long target = 0;
    while ((target = *flag) == 0ULL) { __nanosleep(200); }
    if (gtimer() >= target) r.flags |= 4u;  // late: target already passed
    while (gtimer() < target) { }
  }
  if (const unsigned long long delay = a.shift_cycles * member; delay != 0ULL) {
    const unsigned long long d0 = clock64();
    while (clock64() - d0 < delay) { }
  }
  const unsigned long long c0 = clock64(), g0 = gtimer();
  for (unsigned long long i = 0; i < a.tiles; ++i) {
    const int s = static_cast<int>(i % a.stages);
    const uint32_t bar = smem_u32(&bars[s]);
    if (i >= static_cast<unsigned long long>(a.stages))
      mbar_wait(bar, static_cast<uint32_t>((i / a.stages - 1) & 1));
    if (a.multicast > 1 && s == 0 && i != 0) cluster_sync();  // ring phases aligned
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(bar), "r"(box_bytes) : "memory");
    const uint32_t dst = smem_u32(smem + static_cast<size_t>(s) * box_bytes);
    const int cx = 0, cy = static_cast<int>(row_base + tile * a.box_rows);
    if (a.multicast > 1) {
      if (rank == 0)
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
            "bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n"
            ::"r"(dst), "l"(tmap_addr), "r"(cx), "r"(cy), "r"(bar), "h"(mask) : "memory");
    } else {
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n"
          ::"r"(dst), "l"(tmap_addr), "r"(cx), "r"(cy), "r"(bar) : "memory");
    }
    if (++tile == a.slice_tiles) tile = 0;
  }
  const unsigned long long first = a.tiles > static_cast<unsigned long long>(a.stages)
                                       ? a.tiles - a.stages : 0;
  for (unsigned long long i = first; i < a.tiles; ++i)
    mbar_wait(smem_u32(&bars[i % a.stages]), static_cast<uint32_t>((i / a.stages) & 1));
  r.clk1 = clock64(); r.gt1 = gtimer(); r.clk0 = c0; r.gt0 = g0;
  r.bytes = a.tiles * box_bytes; r.steps = a.tiles; r.chase_cycles = 0;
  if (a.multicast > 1) cluster_sync();  // no CTA exits while multicasts target it
}

__global__ void flush_kernel(const uint4* scratch, unsigned long long elems, unsigned* sink) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (; i < elems; i += stride) acc ^= scratch[i].x ^ scratch[i].w;
  if (acc == 0x5bd1e995u) sink[0] = acc;
}

static unsigned long long arg_kv(int argc, char** argv, const char* key, unsigned long long def) {
  const size_t n = strlen(key);
  for (int i = 5; i < argc; ++i)
    if (!strncmp(argv[i], key, n) && argv[i][n] == '=') return strtoull(argv[i] + n + 1, nullptr, 10);
  return def;
}
static const char* arg_str(int argc, char** argv, const char* key) {
  const size_t n = strlen(key);
  for (int i = 5; i < argc; ++i)
    if (!strncmp(argv[i], key, n) && argv[i][n] == '=') return argv[i] + n + 1;
  return nullptr;
}
static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <nsm> <footprint_mb> <per_cta_mb> <trials> [box_rows=128] "
            "[stages=4] [group=G] [group_rot=0|1] [multicast=k] [sync_start=0|1] "
            "[shift_cycles=N] [nosync=1] [far=0|1] [table=FILE] [log=FILE]\n", argv[0]);
    return 1;
  }
  const int nsm = atoi(argv[1]);
  const unsigned long long footprint_mb = strtoull(argv[2], nullptr, 10);
  const unsigned long long per_cta_mb = strtoull(argv[3], nullptr, 10);
  const int trials = atoi(argv[4]);
  const int box_rows = (int)arg_kv(argc, argv, "box_rows", 128);
  const int stages = (int)arg_kv(argc, argv, "stages", 4);
  const unsigned group = (unsigned)arg_kv(argc, argv, "group", 1);
  const bool group_rot = arg_kv(argc, argv, "group_rot", 0) != 0;
  const unsigned multicast = (unsigned)arg_kv(argc, argv, "multicast", 0);
  const unsigned long long shift_cycles = arg_kv(argc, argv, "shift_cycles", 0);
  // nosync=1: shift without the globaltimer barrier (member m spins m *
  // shift_cycles from its own dispatch; adjacent CTAs are dispatched ~3
  // cycles apart) -- the closed-loop form, which keeps the spin out of the
  // captured instruction stream.
  const bool nosync = arg_kv(argc, argv, "nosync", 0) != 0;
  const bool sync_start = !nosync && (shift_cycles != 0 || arg_kv(argc, argv, "sync_start", 0) != 0);
  const bool far = arg_kv(argc, argv, "far", 0) != 0;
  const char* table_path = arg_str(argc, argv, "table");
  if (const char* log_path = arg_str(argc, argv, "log"); log_path != nullptr && freopen(log_path, "w", stdout) == nullptr) {
    fprintf(stderr, "cannot open log\n"); return 1;
  }
  int dev = 0; cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int clock_khz = 0;  // cudaDeviceProp.clockRate is gone in CUDA 13
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
  if (nsm < 1 || nsm > prop.multiProcessorCount || trials < 1 || footprint_mb == 0 || box_rows < 8 ||
      box_rows > 256 || stages < 1 || stages > kMaxStages || group < 1 ||
      nsm % group != 0 || (multicast > 1 && (nsm % multicast != 0 || multicast > 16)) ||
      (multicast > 1 && group != 1) || (far && (group < 2 || multicast > 1)) ||
      (table_path != nullptr && (group != 1 || multicast > 1 || far || group_rot ||
                                 nsm != prop.multiProcessorCount || nsm > 65535))) {
    fprintf(stderr, "bad arguments\n");
    return 1;
  }
  const unsigned per_slice_ctas = multicast > 1 ? multicast : group;
  unsigned n_slices = nsm / per_slice_ctas;
  // table: slice / member of every SM id
  std::vector<unsigned short> h_slice, h_member;
  if (table_path != nullptr) {
    FILE* tf = fopen(table_path, "r");
    unsigned ts = 0;
    if (tf == nullptr || fscanf(tf, "slices %u", &ts) != 1 || ts < 1 || ts > (unsigned)nsm) {
      fprintf(stderr, "bad table header\n"); return 1;
    }
    h_slice.assign(nsm, 0xFFFF); h_member.assign(nsm, 0);
    unsigned sm, sl, mb; int rows = 0;
    while (fscanf(tf, "%u %u %u", &sm, &sl, &mb) == 3) {
      if (sm >= (unsigned)nsm || sl >= ts || mb > 0xFFFF || h_slice[sm] != 0xFFFF) { fprintf(stderr, "bad table row\n"); return 1; }
      h_slice[sm] = (unsigned short)sl; h_member[sm] = (unsigned short)mb; ++rows;
    }
    fclose(tf);
    if (rows != nsm) { fprintf(stderr, "table does not cover every SM id\n"); return 1; }
    n_slices = ts;
  }
  const unsigned long long box_bytes = (unsigned long long)box_rows * kCols * 2;
  const unsigned long long total_rows = footprint_mb << 20 >> 7;
  const unsigned long long slice_rows = total_rows / n_slices / box_rows * box_rows;
  const unsigned long long slice_tiles = slice_rows / box_rows;
  const unsigned long long tiles = per_cta_mb ? (per_cta_mb << 20) / box_bytes : slice_tiles;
  if (slice_tiles == 0 || tiles == 0) { fprintf(stderr, "footprint too small\n"); return 1; }
  const unsigned long long rows = slice_rows * n_slices;
  printf("META mode=tma nsm=%d threads=32 footprint_mb=%llu per_cta_mb=%llu trials=%d "
         "region=slice box_rows=%d stages=%d group=%u group_rot=%d multicast=%u n_slices=%u "
         "slice_tiles=%llu tiles=%llu box_bytes=%llu sync_start=%d shift_cycles=%llu far=%d table=%s gpu=%s "
         "sms=%d l2_bytes=%d clock_khz=%d\n",
         nsm, footprint_mb, per_cta_mb, trials, box_rows, stages, per_slice_ctas, (int)group_rot,
         multicast, n_slices, slice_tiles, tiles, box_bytes, (int)sync_start, shift_cycles, (int)far,
         table_path != nullptr ? table_path : "none",
         prop.name, prop.multiProcessorCount, prop.l2CacheSize, clock_khz);
  void* buf = nullptr;
  CUDA_CHECK(cudaMalloc(&buf, rows * kCols * 2));
  CUDA_CHECK(cudaMemset(buf, 0x11, rows * kCols * 2));
  CUtensorMap tmap;
  const cuuint64_t dims[2] = {kCols, rows};
  const cuuint64_t strides[1] = {kCols * 2};
  const cuuint32_t box[2] = {kCols, (cuuint32_t)box_rows};
  const cuuint32_t estr[2] = {1, 1};
  if (cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, buf, dims, strides, box,
                             estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    fprintf(stderr, "cuTensorMapEncodeTiled failed\n"); return 2;
  }
  Rec* rec = nullptr; unsigned* sink = nullptr; uint4* scratch = nullptr;
  unsigned long long* start_ns = nullptr;
  const unsigned long long scratch_elems = (256ULL << 20) >> 4;
  CUDA_CHECK(cudaMalloc(&rec, (size_t)nsm * sizeof(Rec)));
  if (sync_start) CUDA_CHECK(cudaMalloc(&start_ns, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&sink, 4096));
  CUDA_CHECK(cudaMalloc(&scratch, scratch_elems * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(scratch, 0, scratch_elems * sizeof(uint4)));
  const size_t smem = box_bytes * stages;
  CUDA_CHECK(cudaFuncSetAttribute(tma_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)smem));
  if (multicast > 1)
    CUDA_CHECK(cudaFuncSetAttribute(tma_scan_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  Args a{};
  a.slice_rows = slice_rows; a.tiles = tiles; a.slice_tiles = slice_tiles;
  a.group = group; a.rot_tiles = group_rot ? (unsigned)(slice_tiles / per_slice_ctas) : 0u;
  a.multicast = multicast; a.box_rows = box_rows; a.stages = stages; a.rec = rec;
  a.shift_cycles = shift_cycles; a.start_ns = start_ns;
  a.far_slices = far ? n_slices : 0u;
  unsigned short* d_slice = nullptr; unsigned short* d_member = nullptr;
  if (table_path != nullptr) {
    CUDA_CHECK(cudaMalloc(&d_slice, (size_t)nsm * sizeof(unsigned short)));
    CUDA_CHECK(cudaMalloc(&d_member, (size_t)nsm * sizeof(unsigned short)));
    CUDA_CHECK(cudaMemcpy(d_slice, h_slice.data(), (size_t)nsm * sizeof(unsigned short), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_member, h_member.data(), (size_t)nsm * sizeof(unsigned short), cudaMemcpyHostToDevice));
  }
  a.sm_slice = d_slice; a.sm_member = d_member;
  // member of CTA c (as the kernel computes it), for the alignment check
  auto member_of = [&](int c, unsigned smid) -> unsigned {
    if (table_path != nullptr) return smid < (unsigned)nsm ? h_member[smid] : 0u;
    if (far) return (unsigned)c / n_slices;
    return multicast > 1 ? 0u : (unsigned)c % group;
  };
  const bool flush = footprint_mb > 48;
  std::vector<Rec> recs(nsm);
  std::vector<double> agg_bpns, agg_window, med_bpc, med_mhz;
  for (int t = 0; t <= trials; ++t) {
    if (flush) {
      flush_kernel<<<1056, 256>>>(scratch, scratch_elems, sink);
      CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemset(rec, 0, (size_t)nsm * sizeof(Rec)));
    if (start_ns != nullptr) CUDA_CHECK(cudaMemset(start_ns, 0, sizeof(unsigned long long)));
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute attr{};
    if (multicast > 1) {
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = multicast; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
      cfg.attrs = &attr; cfg.numAttrs = 1;
    }
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, tma_scan_kernel, tmap, a));
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(recs.data(), rec, (size_t)nsm * sizeof(Rec), cudaMemcpyDeviceToHost));
    if (t == 0) {
      std::vector<unsigned> sm; for (auto& x : recs) sm.push_back(x.smid);
      std::sort(sm.begin(), sm.end());
      printf("PLACEMENT distinct_sm=%zu nsm=%d\n",
             (size_t)(std::unique(sm.begin(), sm.end()) - sm.begin()), nsm);
      continue;
    }
    unsigned long long total = 0, gmin = ~0ULL, gmax = 0, g0min = ~0ULL;
    std::vector<double> bpc, mhz;
    for (int c = 0; c < nsm; ++c) if (recs[c].bytes != 0) g0min = std::min(g0min, recs[c].gt0);
    for (int c = 0; c < nsm; ++c) {
      const Rec& x = recs[c];
      const double dc = (double)(x.clk1 - x.clk0), dn = (double)(x.gt1 - x.gt0);
      // start_ns: this CTA's stream start relative to the earliest CTA (the
      // measured alignment: ~0 for lockstep, member * shift_cycles / f for shift)
      printf("CTA trial=%d cta=%d smid=%u flags=%u cycles=%llu ns=%llu bytes=%llu iters=%llu "
             "chase_cycles=0 mhz=%.1f start_ns=%llu\n", t, c, x.smid, x.flags,
             (unsigned long long)dc, (unsigned long long)dn, x.bytes, x.steps,
             dn > 0 ? dc / dn * 1e3 : 0.0, (unsigned long long)(x.bytes != 0 ? x.gt0 - g0min : 0));
      if (x.bytes == 0) continue;  // idle SM
      total += x.bytes; gmin = std::min(gmin, x.gt0); gmax = std::max(gmax, x.gt1);
      if (dc > 0) bpc.push_back(x.bytes / dc);
      if (dn > 0) mhz.push_back(dc / dn * 1e3);
    }
    if (sync_start) {
      // Alignment check: members without shift should start within the
      // globaltimer resolution; a CTA that found the target already passed
      // (flags & 4) means the margin was too small.
      unsigned long long max_m0 = 0; int late = 0;
      for (int c = 0; c < nsm; ++c) {
        if ((recs[c].flags & 4u) != 0) ++late;
        if (recs[c].bytes != 0 && member_of(c, recs[c].smid) == 0) max_m0 = std::max(max_m0, recs[c].gt0 - g0min);
      }
      printf("SYNC trial=%d m0_max_start_ns=%llu late_ctas=%d\n", t, max_m0, late);
    }
    const double window = gmax > gmin ? (double)(gmax - gmin) : 0.0;
    const double agg = window > 0 ? total / window : 0.0;
    agg_bpns.push_back(agg); agg_window.push_back(window);
    med_bpc.push_back(median(bpc)); med_mhz.push_back(median(mhz));
    printf("TRIAL %d total_bytes=%llu window_ns=%.0f agg_B_per_ns=%.3f per_sm_B_per_cycle_med=%.3f "
           "mhz_med=%.1f active_sms=%zu chase_cyc_per_step=0 chase_ns_per_step=0\n",
           t, total, window, agg, median(bpc), median(mhz), bpc.size());
  }
  printf("RESULT agg_B_per_ns=%.3f per_sm_B_per_cycle=%.3f mhz=%.1f window_ns=%.0f "
         "chase_cyc_per_step=0 chase_ns_per_step=0\n", median(agg_bpns), median(med_bpc),
         median(med_mhz), median(agg_window));
  CUDA_CHECK(cudaFree(scratch)); CUDA_CHECK(cudaFree(sink)); CUDA_CHECK(cudaFree(rec));
  if (start_ns != nullptr) CUDA_CHECK(cudaFree(start_ns));
  if (d_slice != nullptr) { CUDA_CHECK(cudaFree(d_slice)); CUDA_CHECK(cudaFree(d_member)); }
  CUDA_CHECK(cudaFree(buf));
  return 0;
}
