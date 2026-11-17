// ldgsts_issue probe: what does issuing a cp.async copy (LDGSTS) cost the warp, as a function of how many copies
// the SM has in flight?  Grid N CTAs (one per SM), 32 x W threads.  Every warp of an active SM loops: read clock;
// issue C copies back to back (each a warp-wide LDGSTS.E.BYPASS.128 of 512 B from the warp's own part of the CTA's
// region, so no read is shared); read clock; commit_group; wait_group K (K groups stay in flight); read clock.
// Per CTA: summed issue cycles (first to second clock) and wait cycles (second to third) over all warps and rounds.
// SMs whose table member is 65535 take part in the start handshake only.  Destinations: a ring of 16 x 512 B per warp.
#include "../../tma_cold/mb.cuh"

#include <algorithm>
#include <string>
#include <vector>

constexpr int kMaxWarps = 8;
constexpr int kRingSlots = 16;
constexpr int kSmemBytes = kMaxWarps * kRingSlots * 512 + 2048;

__device__ __forceinline__ void cp_async16(std::uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int K>
__device__ __forceinline__ void cp_wait() {
  if constexpr (K == 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
  else if constexpr (K == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 3;" ::: "memory");
}

struct IssueOut {
  unsigned long long issue, wait, total, rounds;
  unsigned smid, pad;
};

// M 0: three clock reads per round (issue / wait split); M 1: no clock inside the loop (total only).
// interleave 0: every warp walks its own contiguous part of the region; 1: warp w takes the 2 KB chunks w, w + warps,
// ... of the whole region (the order of ldgsts_loaded.cu; C must be 4).
template <int C, int K, int M = 0>
__global__ void __launch_bounds__(256, 1) k_issue(const uint4* buf, std::uint64_t region_vec4, int rounds,
                                                  unsigned* arrived, IssueOut* out, const unsigned short* sm_slice,
                                                  const unsigned short* sm_member, int interleave) {
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ unsigned long long s_issue, s_wait;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  if (threadIdx.x == 0) { s_issue = 0; s_wait = 0; }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(arrived, 1u);
    while (*((volatile unsigned*)arrived) < gridDim.x) { }
  }
  __syncthreads();
  const unsigned sm = mb_smid();
  const unsigned slice = sm_slice[sm], member = sm_member[sm];
  IssueOut& o = out[blockIdx.x];
  if (member == 0xFFFFu) {
    if (threadIdx.x == 0) { o.issue = 0; o.wait = 0; o.total = 0; o.rounds = 0; o.smid = sm; o.pad = 1; }
    return;
  }
  const uint4* my = buf + slice * region_vec4;
  const std::uint64_t per_warp = (region_vec4 / 32) / warps;  // 512 B chunks of this warp's part
  const uint4* mine = interleave ? my + (std::uint64_t)warp * 128 : my + (std::uint64_t)warp * per_warp * 32;
  const std::uint64_t stride = interleave ? (std::uint64_t)warps * 4 : (std::uint64_t)C;        // 512 B chunks per round
  const std::uint64_t limit = interleave ? region_vec4 / 32 - (std::uint64_t)warps * 4 : per_warp; // chunks
  // destinations start on a 1024 B boundary of shared memory (two static words precede the dynamic array; with the
  // ring at offset 16 every LDGSTS was delivered as 32 sectors instead of 16)
  const std::uint32_t ring = ((mb_smem_u32(smem) + 1023u) & ~1023u) + warp * kRingSlots * 512;
  std::uint64_t chunk = 0;
  unsigned slot = 0;
  long long issue = 0, wait = 0;
  const long long c0 = static_cast<long long>(mb_clock());
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    const long long t0 = M == 0 ? static_cast<long long>(mb_clock()) : 0;
#pragma unroll
    for (int c = 0; c < C; ++c)
      cp_async16(ring + ((slot + c) % kRingSlots) * 512 + lane * 16, mine + (chunk + c) * 32 + lane);
    const long long t1 = M == 0 ? static_cast<long long>(mb_clock()) : 0;
    cp_commit();
    cp_wait<K>();
    const long long t2 = M == 0 ? static_cast<long long>(mb_clock()) : 0;
    issue += t1 - t0;
    wait += t2 - t1;
    slot = (slot + C) % kRingSlots;
    chunk += stride;
    if (chunk + stride > limit) chunk = 0;
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  const long long c1 = static_cast<long long>(mb_clock());
  if (lane == 0) {
    atomicAdd(&s_issue, static_cast<unsigned long long>(issue));
    atomicAdd(&s_wait, static_cast<unsigned long long>(wait));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    o.issue = s_issue; o.wait = s_wait; o.total = static_cast<unsigned long long>(c1 - c0);
    o.rounds = static_cast<unsigned long long>(rounds); o.smid = sm; o.pad = 0;
  }
}

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

#define ISSUE_KERNELS(X) \
  X(1, 0) X(2, 0) X(4, 0) X(8, 0) X(16, 0) X(1, 1) X(2, 1) X(4, 1) X(8, 1) X(16, 1) X(1, 3) X(2, 3) X(4, 3) X(8, 3) X(16, 3)

int main(int argc, char** argv) {
  std::string out = "ldgsts_issue.jsonl", table, tag;
  int N = 1, W = 1, rounds = 2048, samples = 4, C = 4, K = 0, noclock = 0, interleave = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto n = [&](const char* k, int* d) { if (a == k && i + 1 < argc) { *d = std::atoi(argv[++i]); return true; } return false; };
    if (a == "--out" && i + 1 < argc) { out = argv[++i]; continue; }
    if (a == "--table" && i + 1 < argc) { table = argv[++i]; continue; }
    if (a == "--tag" && i + 1 < argc) { tag = argv[++i]; continue; }
    if (n("--ctas", &N) || n("--warps", &W) || n("--rounds", &rounds) || n("--samples", &samples) || n("--copies", &C) ||
        n("--keep", &K) || n("--noclock", &noclock) || n("--interleave", &interleave))
      continue;
    std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2;
  }
  if (W < 1 || W > kMaxWarps || table.empty() || (interleave && C != 4) || (noclock && !(K == 3 && (C == 4 || C == 16)))) return 2;
  const MbDevice dev = mb_query_device();
  if (N != dev.sm_count) { std::fprintf(stderr, "--ctas must be the SM count\n"); return 2; }
  std::vector<unsigned short> h_slice(N, 0xFFFF), h_member(N, 0);
  {
    FILE* tf = std::fopen(table.c_str(), "r");
    unsigned ts = 0, sm, sl, mb; int rows = 0;
    if (!tf || std::fscanf(tf, "slices %u", &ts) != 1 || ts < 1 || ts > (unsigned)N) { std::fprintf(stderr, "bad table header\n"); return 2; }
    while (std::fscanf(tf, "%u %u %u", &sm, &sl, &mb) == 3) {
      if (sm >= (unsigned)N || sl >= ts || mb > 0xFFFF || h_slice[sm] != 0xFFFF) { std::fprintf(stderr, "bad table row\n"); return 2; }
      h_slice[sm] = (unsigned short)sl; h_member[sm] = (unsigned short)mb; ++rows;
    }
    std::fclose(tf);
    if (rows != N) { std::fprintf(stderr, "table does not cover every SM id\n"); return 2; }
  }
  const std::size_t bytes = 24ull << 20;
  uint4* buf = nullptr;
  MB_CHECK(cudaMalloc(&buf, bytes));
  MB_CHECK(cudaMemset(buf, 1, bytes));
  std::uint64_t region = (bytes / 16) / N;
  region -= region % (32 * kMaxWarps);
  unsigned short* d_slice = nullptr; unsigned short* d_member = nullptr;
  MB_CHECK(cudaMalloc(&d_slice, sizeof(unsigned short) * N));
  MB_CHECK(cudaMalloc(&d_member, sizeof(unsigned short) * N));
  MB_CHECK(cudaMemcpy(d_slice, h_slice.data(), sizeof(unsigned short) * N, cudaMemcpyHostToDevice));
  MB_CHECK(cudaMemcpy(d_member, h_member.data(), sizeof(unsigned short) * N, cudaMemcpyHostToDevice));
  unsigned* arrived = nullptr; IssueOut* d_out = nullptr;
  MB_CHECK(cudaMalloc(&arrived, sizeof(unsigned)));
  MB_CHECK(cudaMalloc(&d_out, sizeof(IssueOut) * N));
  std::vector<double> issue_per_copy, wait_per_group, bpc;
  std::string per_sm, err;
  int active = 0;
  for (int rep = 0; rep < samples + 1; ++rep) {
    MB_CHECK(cudaMemset(arrived, 0, sizeof(unsigned)));
    MB_CHECK(cudaMemset(d_out, 0, sizeof(IssueOut) * N));
    bool launched = false;
#define X(C_, K_)                                                                                                    \
    if (!noclock && C == C_ && K == K_) {                                                                            \
      MB_CHECK(cudaFuncSetAttribute(k_issue<C_, K_>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));       \
      k_issue<C_, K_><<<N, 32 * W, kSmemBytes>>>(buf, region, rounds, arrived, d_out, d_slice, d_member, interleave); \
      launched = true;                                                                                               \
    }
    ISSUE_KERNELS(X)
#undef X
#define Y(C_)                                                                                                        \
    if (noclock && C == C_ && K == 3) {                                                                              \
      MB_CHECK(cudaFuncSetAttribute(k_issue<C_, 3, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));     \
      k_issue<C_, 3, 1><<<N, 32 * W, kSmemBytes>>>(buf, region, rounds, arrived, d_out, d_slice, d_member, interleave); \
      launched = true;                                                                                               \
    }
    Y(4) Y(16)
#undef Y
    if (!launched) { std::fprintf(stderr, "no kernel for copies %d keep %d\n", C, K); return 2; }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    if (rep == 0) continue;
    std::vector<IssueOut> h(N);
    MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(IssueOut) * N, cudaMemcpyDeviceToHost));
    per_sm = "["; active = 0;
    for (int i = 0; i < N; ++i) {
      if (h[i].rounds == 0) continue;
      const double groups = (double)W * (double)h[i].rounds;
      const double ic = (double)h[i].issue / (groups * C), wg = (double)h[i].wait / groups;
      const double b = groups * C * 512.0 / (double)h[i].total;
      issue_per_copy.push_back(ic); wait_per_group.push_back(wg); bpc.push_back(b);
      char t[96];
      std::snprintf(t, sizeof(t), "%s[%u,%.2f,%.1f,%.2f]", active ? "," : "", h[i].smid, ic, wg, b);
      per_sm += t; ++active;
    }
    per_sm += "]";
  }
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) return 1;
  std::fprintf(f,
    "{\"schema\":\"cyclescope.microbench.ldgsts_issue.v1\",\"tag\":\"%s\",\"copies\":%d,\"keep\":%d,\"noclock\":%d,\"interleave\":%d,\"warps\":%d,\"rounds\":%d,"
    "\"ctas\":%d,\"active\":%d,\"issue_per_copy_median\":%.3f,\"issue_per_copy_p10\":%.3f,\"issue_per_copy_p90\":%.3f,"
    "\"wait_per_group_median\":%.2f,\"wait_per_group_p10\":%.2f,\"wait_per_group_p90\":%.2f,"
    "\"bytes_per_cycle_per_cta_median\":%.3f,\"n\":%zu,\"table\":\"%s\",\"per_sm\":%s,\"gpu_name\":\"%s\","
    "\"sm_clock_khz\":%d,\"sm_count\":%d",
    tag.c_str(), C, K, noclock, interleave, W, rounds, N, active, pct(issue_per_copy, .5), pct(issue_per_copy, .1), pct(issue_per_copy, .9),
    pct(wait_per_group, .5), pct(wait_per_group, .1), pct(wait_per_group, .9), pct(bpc, .5), issue_per_copy.size(),
    table.c_str(), per_sm.empty() ? "[]" : per_sm.c_str(), dev.name.c_str(), dev.sm_clock_khz, dev.sm_count);
  if (!err.empty()) std::fprintf(f, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(f, "}\n"); std::fclose(f);
  std::printf("ldgsts_issue %s C=%d K=%d W=%d active=%d issue/copy=%.2f wait/group=%.1f bpc=%.2f%s\n", tag.c_str(), C, K, W,
              active, pct(issue_per_copy, .5), pct(wait_per_group, .5), pct(bpc, .5), err.empty() ? "" : " ERROR");
  return err.empty() ? 0 : 1;
}
