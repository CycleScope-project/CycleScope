// load_after_stores probe (protocol.json): does an L1-hit load wait behind
// global stores the SM has issued but not yet drained?
// One CTA, 2 warps.  Per repetition: 8 victim lines loaded twice (L1
// resident, all lanes of a load on one word); then K store instructions
// (st.global.u16, every lane its own sector, the scattered per-element
// store of a GEMM epilogue) issued by warp 0 (MODE 0) or by warp 1 (MODE 1,
// warp 0 waits on a named barrier until warp 1 has issued them); then warp 0
// times 8 dependent victim loads between two clock64 reads.
#include "../../tma_cold/mb.cuh"
#include <algorithm>
#include <string>
#include <vector>
__device__ __forceinline__ std::uint32_t ldg(const unsigned char* p) {
  std::uint32_t v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void stg16(unsigned char* p, std::uint32_t v) {
  asm volatile("st.global.u16 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
template <int MODE>
__global__ void __launch_bounds__(64, 1) k_las(const unsigned char* victims, unsigned char* dst, int K, int reps,
                                               std::uint64_t* out) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  std::uint32_t sink = 0;
  for (int rep = 0; rep < reps; ++rep) {
    if (warp == 0)
      for (int pass = 0; pass < 2; ++pass)
        for (int k = 0; k < 8; ++k) sink ^= ldg(victims + k * 4096);
    __syncthreads();
    if (warp == MODE)
      for (int k = 0; k < K; ++k) stg16(dst + (static_cast<std::size_t>(rep & 7) * 64 + k) * 2048 + lane * 64, lane);
    if (MODE == 1) __syncthreads();
    if (warp == 0) {
      std::uint32_t off = sink & 0u;
      const std::uint64_t t0 = mb_clock();
      for (int k = 0; k < 8; ++k) off += ldg(victims + k * 4096 + off);
      const std::uint64_t t1 = mb_clock();
      if (lane == 0) { out[2 * rep] = t1 - t0; out[2 * rep + 1] = off + sink; }
    }
    __syncthreads();
  }
}
int main(int argc, char** argv) {
  std::string outp = "load_after_stores.jsonl";
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--out" && i + 1 < argc) outp = argv[++i];
  const int reps = 17, launches = 8;
  const MbDevice dev = mb_query_device();
  unsigned char *v = nullptr, *d = nullptr; std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&v, 1 << 16)); MB_CHECK(cudaMemset(v, 0, 1 << 16));
  MB_CHECK(cudaMalloc(&d, 2 << 20)); MB_CHECK(cudaMemset(d, 0, 2 << 20));
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * reps));
  FILE* f = std::fopen(outp.c_str(), "a");
  if (!f) return 1;
  for (int mode = 0; mode < 2; ++mode)
    for (int K : {0, 1, 2, 4, 8, 16, 32, 64}) {
      std::vector<double> s;
      for (int l = 0; l < launches; ++l) {
        if (mode == 0) k_las<0><<<1, 64>>>(v, d, K, reps, d_out); else k_las<1><<<1, 64>>>(v, d, K, reps, d_out);
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(2 * reps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(std::uint64_t) * 2 * reps, cudaMemcpyDeviceToHost));
        if (l == 0) continue;
        for (int r = 1; r < reps; ++r) s.push_back(static_cast<double>(h[2 * r]));
      }
      std::sort(s.begin(), s.end());
      std::fprintf(f, "{\"schema\":\"cyclescope.microbench.load_after_stores.v1\",\"mode\":\"%s\",\"stores\":%d,"
                      "\"chain_cycles_p10\":%.1f,\"chain_cycles_median\":%.1f,\"chain_cycles_p90\":%.1f,\"n\":%zu,"
                      "\"gpu_name\":\"%s\"}\n",
                   mode ? "other_warp" : "same_warp", K, s[s.size() / 10], s[s.size() / 2], s[s.size() * 9 / 10],
                   s.size(), dev.name.c_str());
    }
  std::fclose(f);
  return 0;
}
