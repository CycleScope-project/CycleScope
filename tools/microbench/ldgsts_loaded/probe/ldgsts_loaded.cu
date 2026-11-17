// ldgsts_loaded probe: cp.async (LDGSTS) fill delivery per SM and aggregate,
// and group completion latency, when N SMs stream at once.
//
// Grid N CTAs (one per SM, 135 KB dynamic smem), 32*W threads. Each warp w
// owns a ring of 4 x 2 KB smem slots: per round every lane issues 4 x
// cp.async.cg 16 B (global -> shared, L1 bypass = LDGSTS.E.BYPASS.128) =
// 2 KB per warp, commit_group, then wait_group 3 so 4 groups stay in flight
// per warp (v2; v1 waited each group and was latency-bound); group latency
// = clock at retirement - its issue clock; R rounds; bytes / cycles per CTA; all
// CTAs meet on an arrival counter first. Hot: each CTA cycles inside its own
// region of a 24 MiB buffer (L2 resident after the discarded warm launch).
#include "../../tma_cold/mb.cuh"

#include <algorithm>
#include <string>
#include <vector>

constexpr int kMaxWarps = 8;
constexpr int kSlotBytes = 2048;  // 32 lanes x 4 x 16 B
constexpr int kGroups = 4;        // v2: groups in flight per warp (ring of kGroups slots)
constexpr int kSmemBytes = 135 * 1024;  // 1 CTA / SM
constexpr int kLatSamples = 512;
constexpr int kStamps = 64;       // globaltimer of thread 0 at kStamps evenly spaced rounds

__device__ __forceinline__ void cp_async16(std::uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_g3() { asm volatile("cp.async.wait_group 3;" ::: "memory"); }

struct CtaOut {
  unsigned long long bytes;
  long long c0, c1, g0, g1;
  unsigned smid;
  unsigned late;   // paced rounds of warp 0 that started more than 32 cycles after their due time
  unsigned lat[kLatSamples];
  long long stamp[kStamps];
};

// kPaced: the paced loop (one group in flight, --pace) with globaltimer stamps; the unpaced
// instantiation keeps the loop of the original probe (no pace branch, no stamps)
template <bool kPaced>
__global__ void __launch_bounds__(256, 1) k_ldgsts_loaded(const uint4* buf, std::uint64_t region_vec4,
                                                          int rounds, unsigned* arrived, CtaOut* out,
                                                          int group, int rot, int shift, int far, int smid_mask,
                                                          int slice_count, const unsigned short* sm_slice,
                                                          const unsigned short* sm_member, int pace) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ unsigned long long cta_bytes;
  __shared__ int cta_lat_n;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  if (threadIdx.x == 0) { cta_bytes = 0; cta_lat_n = 0; }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(arrived, 1u);
    while (*((volatile unsigned*)arrived) < gridDim.x) { }
  }
  __syncthreads();
  // shift: member m of a group starts m x shift SM cycles after the barrier
  // far: the members of a group are the CTAs gridDim.x / group apart (not
  // consecutive ones); slice = region index, member = position in the group
  // smid_mask >= 0: the group is chosen by SM id instead -- SMs that differ
  // only in the mask bits read the same region (slice = smid & ~mask)
  const unsigned slices = gridDim.x / group;
  unsigned slice = far ? blockIdx.x % slices : blockIdx.x / group;
  unsigned member = far ? blockIdx.x / slices : blockIdx.x % group;
  if (smid_mask >= 0) {
    const unsigned sm = mb_smid();
    slice = (sm & ~(unsigned)smid_mask) % (unsigned)slice_count;
    member = sm & (unsigned)smid_mask;
  }
  // table: slice / member listed per SM id (file `slices <n>` + `<sm> <slice> <member>` rows)
  if (sm_slice != nullptr) {
    const unsigned sm = mb_smid();
    slice = sm_slice[sm];
    member = sm_member[sm];
  }
  CtaOut& o = out[blockIdx.x];
  if (member == 0xFFFFu) {  // idle SM (table): no loads, bytes 0
    if (threadIdx.x == 0) { o.bytes = 0; o.c0 = o.c1 = o.g0 = o.g1 = 0; o.smid = mb_smid(); }
    return;
  }
  if (shift > 0 && member) {
    const long long until = (long long)mb_clock() + (long long)member * shift;
    while ((long long)mb_clock() < until) { }
  }
  const long long g0 = (long long)mb_gtimer();
  const long long c0 = (long long)mb_clock();
  // region: this CTA's slice; warp w walks 2 KB chunks w, w+warps, ...
  // group > 1: the `group` consecutive CTAs of a slice read the same region
  // from the same start (lockstep); rot: member m starts chunks / group later
  const uint4* my = buf + slice * region_vec4;
  const std::uint64_t chunks = region_vec4 / 128;  // 128 vec4 = 2 KB
  const std::uint64_t first =
      (warp + (rot ? member * (chunks / group) : 0)) % chunks;
  std::uint64_t chunk = first;
  const std::uint32_t slot0 = mb_smem_u32(smem) + warp * kSlotBytes * kGroups;
  unsigned long long bytes = 0;
  unsigned lat[kLatSamples / kMaxWarps];
  int nlat = 0;
  long long issue_clk[kGroups];
  unsigned late = 0;
  // v2: keep kGroups groups in flight per warp; wait_group kGroups-1 retires
  // the oldest, whose latency is clock(now) - its issue clock
  // stamps: thread 0 records the globaltimer at round stamp_next (no per-round division)
  const int stamp_every = rounds / kStamps > 0 ? rounds / kStamps : 1;
  int stamp_next = 0, stamp_n = 0;
  for (int r = 0; r < rounds; ++r) {
    const int g = r % kGroups;
    const std::uint32_t slot = slot0 + g * kSlotBytes;
    const uint4* src = my + chunk * 128 + lane;
    // pace > 0: round r is issued at c0 + r x pace and waited before the next
    // one (one group in flight, start offsets between CTAs stay fixed)
    if (kPaced) {
      const long long due = c0 + (long long)r * pace;
      if ((long long)mb_clock() > due + 32) ++late;
      while ((long long)mb_clock() < due) { }
    } else if (r >= kGroups) {
      cp_wait_g3();  // group r - kGroups is complete
      const long long t1 = (long long)mb_clock();
      if (lane == 0 && nlat < kLatSamples / kMaxWarps) lat[nlat++] = (unsigned)(t1 - issue_clk[g]);
      bytes += 64;
    }
    issue_clk[g] = (long long)mb_clock();
    if (kPaced && r == stamp_next) {
      if (threadIdx.x == 0 && stamp_n < kStamps) o.stamp[stamp_n] = (long long)mb_gtimer();
      ++stamp_n; stamp_next += stamp_every;
    }
    cp_async16(slot + lane * 16, src);
    cp_async16(slot + 512 + lane * 16, src + 32);
    cp_async16(slot + 1024 + lane * 16, src + 64);
    cp_async16(slot + 1536 + lane * 16, src + 96);
    cp_commit();
    if (kPaced) {
      cp_wait_all();
      const long long t1 = (long long)mb_clock();
      if (lane == 0 && nlat < kLatSamples / kMaxWarps) lat[nlat++] = (unsigned)(t1 - issue_clk[g]);
      bytes += 64;
    }
    chunk += warps; if (chunk >= chunks) chunk = first;
  }
  cp_wait_all();
  if (!kPaced) bytes += 64ull * kGroups;
  __syncwarp();
  if (lane == 0) {
    atomicAdd(&cta_bytes, bytes * 32ull);
    const int base = atomicAdd(&cta_lat_n, nlat);
    for (int i = 0; i < nlat && base + i < kLatSamples; ++i) o.lat[base + i] = lat[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    o.bytes = cta_bytes; o.c0 = c0; o.c1 = (long long)mb_clock(); o.g0 = g0; o.g1 = (long long)mb_gtimer();
    o.smid = mb_smid(); o.late = late;
    for (int i = cta_lat_n; i < kLatSamples; ++i) o.lat[i] = 0;
  }
}

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}
static std::string arr(const std::vector<double>& v) {
  std::string s = "["; char b[64];
  for (std::size_t i = 0; i < v.size(); ++i) { std::snprintf(b, sizeof(b), "%.3f", v[i]); s += b; if (i + 1 < v.size()) s += ","; }
  return s + "]";
}

int main(int argc, char** argv) {
  std::string out = "ldgsts_loaded.jsonl", table;
  int N = 1, W = 4, rounds = 4096, samples = 8, G = 1, rot = 0, shift = 0, far = 0, smid_mask = -1, pace = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    if (a == "--out" && i + 1 < argc) { out = argv[++i]; continue; }
    if (a == "--table" && i + 1 < argc) { table = argv[++i]; continue; }
    if (n("--ctas", &N) || n("--warps", &W) || n("--rounds", &rounds) || n("--samples", &samples) ||
        n("--group", &G) || n("--rot", &rot) || n("--shift", &shift) || n("--far", &far) || n("--smid-mask", &smid_mask) || n("--pace", &pace))
      continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2;
  }
  if (W < 1 || W > kMaxWarps || G < 1 || N % G || (rot != 0 && rot != 1) || shift < 0 ||
      (far != 0 && far != 1))
    return 2;
  const MbDevice dev = mb_query_device();
  if (N > dev.sm_count) { std::fprintf(stderr, "ctas > SMs\n"); return 2; }
  const std::size_t bytes = 24ull << 20;
  uint4* buf = nullptr;
  MB_CHECK(cudaMalloc(&buf, bytes));
  MB_CHECK(cudaMemset(buf, 1, bytes));
  int slice_count = smid_mask >= 0 ? dev.sm_count : std::max(N / G, 1);
  std::vector<unsigned short> h_slice, h_member;
  unsigned short* d_slice = nullptr; unsigned short* d_member = nullptr;
  if (!table.empty()) {
    if (N != dev.sm_count || G != 1 || rot || far || smid_mask >= 0) { std::fprintf(stderr, "--table needs every SM and no other grouping\n"); return 2; }
    FILE* tf = std::fopen(table.c_str(), "r");
    unsigned ts = 0;
    if (!tf || std::fscanf(tf, "slices %u", &ts) != 1 || ts < 1 || ts > (unsigned)N) { std::fprintf(stderr, "bad table header\n"); return 2; }
    h_slice.assign(N, 0xFFFF); h_member.assign(N, 0);
    unsigned sm, sl, mb; int rows = 0;
    while (std::fscanf(tf, "%u %u %u", &sm, &sl, &mb) == 3) {
      if (sm >= (unsigned)N || sl >= ts || mb > 0xFFFF || h_slice[sm] != 0xFFFF) { std::fprintf(stderr, "bad table row\n"); return 2; }
      h_slice[sm] = (unsigned short)sl; h_member[sm] = (unsigned short)mb; ++rows;
    }
    std::fclose(tf);
    if (rows != N) { std::fprintf(stderr, "table does not cover every SM id\n"); return 2; }
    slice_count = (int)ts;
    MB_CHECK(cudaMalloc(&d_slice, sizeof(unsigned short) * N));
    MB_CHECK(cudaMalloc(&d_member, sizeof(unsigned short) * N));
    MB_CHECK(cudaMemcpy(d_slice, h_slice.data(), sizeof(unsigned short) * N, cudaMemcpyHostToDevice));
    MB_CHECK(cudaMemcpy(d_member, h_member.data(), sizeof(unsigned short) * N, cudaMemcpyHostToDevice));
  }
  std::uint64_t region = (bytes / 16) / slice_count;
  region -= region % 128;
  if (region < (std::uint64_t)W * 128) region = (std::uint64_t)W * 128;
  unsigned* arrived = nullptr; CtaOut* d_out = nullptr;
  MB_CHECK(cudaMalloc(&arrived, sizeof(unsigned)));
  MB_CHECK(cudaMalloc(&d_out, sizeof(CtaOut) * N));
  MB_CHECK(cudaFuncSetAttribute(k_ldgsts_loaded<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  MB_CHECK(cudaFuncSetAttribute(k_ldgsts_loaded<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  std::vector<double> bpc, lat_all, agg, smids, cyc_by_cta, start_by_cta;
  double late_rounds = 0;
  std::string stamps;  // last repetition: per CTA the globaltimer stamps, ns after the earliest start
  std::string err;
  for (int rep = 0; rep < samples + 1; ++rep) {
    MB_CHECK(cudaMemset(arrived, 0, sizeof(unsigned)));
    MB_CHECK(cudaMemset(d_out, 0, sizeof(CtaOut) * N));
    if (pace > 0)
      k_ldgsts_loaded<true><<<N, 32 * W, kSmemBytes>>>(buf, region, rounds, arrived, d_out, G, rot, shift, far, smid_mask,
                                                       slice_count, d_slice, d_member, pace);
    else
      k_ldgsts_loaded<false><<<N, 32 * W, kSmemBytes>>>(buf, region, rounds, arrived, d_out, G, rot, shift, far, smid_mask,
                                                        slice_count, d_slice, d_member, pace);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    if (rep == 0) continue;
    std::vector<CtaOut> h(N);
    MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(CtaOut) * N, cudaMemcpyDeviceToHost));
    double tb = 0, gmin = 1e300, gmax = 0;
    smids.clear(); cyc_by_cta.clear(); start_by_cta.clear();
    double g0min = 1e300;
    for (int i = 0; i < N; ++i) if (h[i].bytes) g0min = std::min(g0min, (double)h[i].g0);
    stamps = "[";
    for (int i = 0; i < N; ++i) {
      smids.push_back((double)h[i].smid);
      cyc_by_cta.push_back((double)(h[i].c1 - h[i].c0));     // stream cycles of the last repetition
      start_by_cta.push_back(h[i].bytes ? (double)h[i].g0 - g0min : -1.0);  // start, ns after the earliest CTA
      stamps += "[";
      for (int k = 0; k < kStamps; ++k) {
        char b[32];
        std::snprintf(b, sizeof(b), "%.0f", h[i].bytes && h[i].stamp[k] ? (double)h[i].stamp[k] - g0min : -1.0);
        stamps += b; if (k + 1 < kStamps) stamps += ",";
      }
      stamps += i + 1 < N ? "]," : "]";
    }
    stamps += "]";
    for (int i = 0; i < N; ++i) {
      if (h[i].bytes == 0) continue;  // idle SM
      late_rounds += h[i].late;
      bpc.push_back((double)h[i].bytes / (double)(h[i].c1 - h[i].c0));
      tb += (double)h[i].bytes; gmin = std::min(gmin, (double)h[i].g0); gmax = std::max(gmax, (double)h[i].g1);
      for (int k = 0; k < kLatSamples; ++k) if (h[i].lat[k]) lat_all.push_back(h[i].lat[k]);
    }
    agg.push_back(tb / (gmax - gmin));
  }
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) return 1;
  std::fprintf(f,
    "{\"schema\":\"blackwell.mechanism_curve_point.v1\",\"family\":\"ldgsts_loaded.N%d.W%d\",\"quantity\":\"throughput\","
    "\"cta_group\":1,\"x_name\":\"ctas\",\"x\":%d,\"y_name\":\"bytes_per_cycle_per_cta\",\"y_unit\":\"bytes/cycle\","
    "\"median\":%.4f,\"p10\":%.4f,\"p90\":%.4f,\"n\":%zu,\"holdout\":false,\"ctas\":%d,\"warps\":%d,\"rounds\":%d,"
    "\"group_bytes\":%d,\"share_group\":%d,\"rot\":%d,\"shift_cycles\":%d,\"far\":%d,\"smid_mask\":%d,\"table\":\"%s\",\"smid_by_cta\":%s,\"cycles_by_cta\":%s,\"start_ns_by_cta\":%s,\"stamps_ns_by_cta\":%s,\"pace_cycles\":%d,\"late_rounds\":%.0f,\"aggregate_bytes_per_ns\":%s,\"aggregate_bytes_per_ns_median\":%.3f,"
    "\"latency_p10\":%.1f,\"latency_p50\":%.1f,\"latency_p90\":%.1f,\"latency_n\":%zu,\"samples\":%s,"
    "\"source\":\"ldgsts_loaded.measured\",\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,\"sm_count\":%d",
    N, W, N, pct(bpc, .5), pct(bpc, .1), pct(bpc, .9), bpc.size(), N, W, rounds, kSlotBytes, G, rot, shift, far, smid_mask, table.c_str(), arr(smids).c_str(),
    arr(cyc_by_cta).c_str(), arr(start_by_cta).c_str(), stamps.empty() ? "[]" : stamps.c_str(), pace, late_rounds,
    arr(agg).c_str(), pct(agg, .5), pct(lat_all, .1), pct(lat_all, .5), pct(lat_all, .9), lat_all.size(),
    N <= 16 ? arr(bpc).c_str() : "[]", dev.name.c_str(), dev.sm_clock_khz, dev.sm_count);
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n"); std::fclose(f);
  std::printf("ldgsts_loaded N=%d W=%d bpc/cta=%.2f agg=%.1f B/ns lat p50=%.0f p90=%.0f%s\n", N, W, pct(bpc, .5),
              pct(agg, .5), pct(lat_all, .5), pct(lat_all, .9), err.empty() ? "" : " ERR");
  cudaFree(buf); cudaFree(arrived); cudaFree(d_out);
  return err.empty() ? 0 : 1;
}
