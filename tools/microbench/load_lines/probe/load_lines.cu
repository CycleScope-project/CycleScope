// load_lines probe (protocol.json): cost of one load instruction as a
// function of how many distinct lines its lanes touch, for global loads
// (addresses chosen) and for local (stack) loads (each thread's own frame).
// One CTA, one warp; lanes >= `lanes` idle.  Data are loaded twice first
// (L1 resident).  Timed: 8 loads as one dependent chain between two
// clock64 reads; cycles per load of lane 0, repetition 0 dropped.
//   FAM 0 global: lane l reads g + k * 4096 + l * spread (spread 0: one
//         word for all lanes; 32: one sector per lane; 128: one line per lane)
//   FAM 1 local : lane l reads its own buf[(k * 32 + off) & 255]
//         (unsigned buf[256] on the thread's stack, written first)
#include "../../tma_cold/mb.cuh"
#include <algorithm>
#include <string>
#include <vector>
__device__ __forceinline__ std::uint32_t ldg(const unsigned char* p) {
  std::uint32_t v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
template <int FAM>
__global__ void __launch_bounds__(32, 1) k_lines(const unsigned char* g, int lanes, int spread, int reps,
                                                 std::uint64_t* out) {
  const int lane = threadIdx.x;
  volatile unsigned buf[256];
  if (FAM == 1)
    for (int i = 0; i < 256; ++i) buf[i] = 0u;
  if (lane >= lanes) return;
  std::uint32_t sink = 0;
  for (int rep = 0; rep < reps; ++rep) {
    for (int pass = 0; pass < 2; ++pass)
      for (int k = 0; k < 8; ++k)
        sink ^= FAM == 0 ? ldg(g + k * 4096 + lane * spread) : buf[(k * 32 + (sink & 0u)) & 255];
    std::uint32_t off = sink & 0u;
    const std::uint64_t t0 = mb_clock();
    for (int k = 0; k < 8; ++k)
      off += FAM == 0 ? ldg(g + k * 4096 + lane * spread + off) : buf[(k * 32 + off) & 255];
    const std::uint64_t t1 = mb_clock();
    if (lane == 0) { out[2 * rep] = t1 - t0; out[2 * rep + 1] = off + sink; }
  }
}
int main(int argc, char** argv) {
  std::string outp = "load_lines.jsonl";
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--out" && i + 1 < argc) outp = argv[++i];
  const int reps = 17, launches = 8;
  const MbDevice dev = mb_query_device();
  unsigned char* g = nullptr; std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&g, 1 << 20)); MB_CHECK(cudaMemset(g, 0, 1 << 20));
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * reps));
  FILE* f = std::fopen(outp.c_str(), "a");
  if (!f) return 1;
  for (int fam = 0; fam < 2; ++fam)
    for (int spread : {0, 32, 128}) {
      if (fam == 1 && spread) continue;
      for (int lanes : {1, 2, 4, 8, 16, 32}) {
        std::vector<double> s;
        for (int l = 0; l < launches; ++l) {
          if (fam == 0) k_lines<0><<<1, 32>>>(g, lanes, spread, reps, d_out);
          else k_lines<1><<<1, 32>>>(g, lanes, spread, reps, d_out);
          MB_CHECK(cudaDeviceSynchronize());
          std::vector<std::uint64_t> h(2 * reps);
          MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(std::uint64_t) * 2 * reps, cudaMemcpyDeviceToHost));
          if (l == 0) continue;
          for (int r = 1; r < reps; ++r) s.push_back(static_cast<double>(h[2 * r]) / 8.0);
        }
        std::sort(s.begin(), s.end());
        std::fprintf(f, "{\"schema\":\"cyclescope.microbench.load_lines.v1\",\"family\":\"%s\",\"spread\":%d,\"lanes\":%d,"
                        "\"cycles_per_load_p10\":%.1f,\"cycles_per_load_median\":%.1f,\"cycles_per_load_p90\":%.1f,"
                        "\"n\":%zu,\"gpu_name\":\"%s\"}\n",
                     fam ? "local" : "global", spread, lanes, s[s.size() / 10], s[s.size() / 2], s[s.size() * 9 / 10],
                     s.size(), dev.name.c_str());
      }
    }
  std::fclose(f);
  return 0;
}
