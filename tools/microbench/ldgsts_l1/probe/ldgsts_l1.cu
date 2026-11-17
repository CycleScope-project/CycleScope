// ldgsts_l1 probe (protocol.json): does an async copy (cp.async) allocate
// in the SM's L1 and evict lines that plain loads left there?
// One CTA, one warp.  Per repetition:
//   1. victims: K load instructions, all lanes of one instruction on the
//      same word (v2: one line per instruction, K lines 4 KiB apart), loaded
//      twice -> L1 resident;
//   2. polluter over a separate L2-resident region, `bytes` per warp:
//      MODE 0 none, 1 cp.async.ca .. L2::128B (16 B per lane), 2 cp.async.cg
//      (16 B per lane), 3 plain ld.global.v4 (16 B per lane); async forms
//      are committed and waited (wait_group 0) every 8 copies;
//   3. timed: the K victim loads as one dependent chain (each address is
//      offset by the previous value, all zero) between two clock64 reads.
// Output: cycles per victim load (lane 0), repetition 0 dropped.
#include "../../tma_cold/mb.cuh"
#include <algorithm>
#include <string>
#include <vector>
constexpr int kVictimLoads = 8;
constexpr int kSmem = 16384;
__device__ __forceinline__ std::uint32_t ldg(const unsigned char* p) {
  std::uint32_t v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ std::uint32_t ldg_v4(const unsigned char* p) {
  std::uint32_t a, b, c, d;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  return a ^ b ^ c ^ d;
}
template <int MODE>
__global__ void __launch_bounds__(32, 1) k_l1(const unsigned char* victims, const unsigned char* region, int bytes,
                                              int reps, std::uint64_t* out) {
  extern __shared__ __align__(128) unsigned char smem[];
  const std::uint32_t sbase = mb_smem_u32(smem);
  const int lane = threadIdx.x;
  std::uint32_t sink = 0;
  for (int rep = 0; rep < reps; ++rep) {
    for (int pass = 0; pass < 2; ++pass)
      for (int k = 0; k < kVictimLoads; ++k) sink ^= ldg(victims + k * 4096);
    const int copies = bytes / 512;
    for (int c = 0; c < copies; ++c) {
      const unsigned char* src = region + static_cast<std::size_t>(c) * 512 + lane * 16;
      const std::uint32_t dst = sbase + (c & 15) * 512 + lane * 16;
      if (MODE == 1) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
      if (MODE == 2) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
      if (MODE == 3) sink ^= ldg_v4(src);
      if ((MODE == 1 || MODE == 2) && (c & 7) == 7) {
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
      }
    }
    if (MODE == 1 || MODE == 2) {
      asm volatile("cp.async.commit_group;" ::: "memory");
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    std::uint32_t off = sink & 0u;
    const std::uint64_t t0 = mb_clock();
    for (int k = 0; k < kVictimLoads; ++k) off += ldg(victims + k * 4096 + off);
    const std::uint64_t t1 = mb_clock();
    if (lane == 0) { out[2 * rep] = t1 - t0; out[2 * rep + 1] = off + sink; }
  }
}
template <int MODE>
static void launch(const unsigned char* v, const unsigned char* r, int bytes, int reps, std::uint64_t* out) {
  MB_CHECK(cudaFuncSetAttribute(k_l1<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  k_l1<MODE><<<1, 32, kSmem>>>(v, r, bytes, reps, out);
}
int main(int argc, char** argv) {
  std::string outp = "ldgsts_l1.jsonl";
  int reps = 17, launches = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "--out" && i + 1 < argc) outp = argv[++i];
  }
  const MbDevice dev = mb_query_device();
  const std::size_t region_bytes = 4u << 20, victim_bytes = kVictimLoads * 32 * 128;
  unsigned char *victims = nullptr, *region = nullptr;
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&victims, victim_bytes));
  MB_CHECK(cudaMalloc(&region, region_bytes));
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * reps));
  MB_CHECK(cudaMemset(victims, 0, victim_bytes));
  MB_CHECK(cudaMemset(region, 0, region_bytes));
  FILE* f = std::fopen(outp.c_str(), "a");
  if (!f) return 1;
  const char* names[4] = {"none", "async_ca", "async_cg", "ldg"};
  for (int mode = 0; mode < 4; ++mode) {
    for (int bytes : {0, 4096, 16384, 65536, 262144, 1048576}) {
      if (mode == 0 && bytes) continue;
      if (mode != 0 && !bytes) continue;
      std::vector<double> s;
      for (int l = 0; l < launches; ++l) {
        if (mode == 0) launch<0>(victims, region, bytes, reps, d_out);
        if (mode == 1) launch<1>(victims, region, bytes, reps, d_out);
        if (mode == 2) launch<2>(victims, region, bytes, reps, d_out);
        if (mode == 3) launch<3>(victims, region, bytes, reps, d_out);
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(2 * reps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(std::uint64_t) * 2 * reps, cudaMemcpyDeviceToHost));
        if (l == 0) continue;
        for (int r = 1; r < reps; ++r) s.push_back(static_cast<double>(h[2 * r]) / kVictimLoads);
      }
      std::sort(s.begin(), s.end());
      std::fprintf(f, "{\"schema\":\"cyclescope.microbench.ldgsts_l1.v1\",\"mode\":\"%s\",\"bytes\":%d,\"victim_lines\":%d,"
                      "\"cycles_per_load_p10\":%.1f,\"cycles_per_load_median\":%.1f,\"cycles_per_load_p90\":%.1f,\"n\":%zu,"
                      "\"gpu_name\":\"%s\",\"sm_clock_khz\":%d}\n",
                   names[mode], bytes, kVictimLoads * 32, s[s.size() / 10], s[s.size() / 2], s[s.size() * 9 / 10], s.size(),
                   dev.name.c_str(), dev.sm_clock_khz);
      std::printf("%s bytes=%d cycles/load median=%.1f\n", names[mode], bytes, s[s.size() / 2]);
    }
  }
  std::fclose(f);
  return 0;
}
