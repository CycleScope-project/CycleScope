// instruction.fetch hierarchy probe: straight-line bodies of N IADD3
// (k_line<N>, footprint 16*N bytes) and M-block taken-branch chains of one
// 128 B line per block (k_jump<M,PERM>).  K=4 passes timed by per-warp
// clock64 pairs inside a counted loop; lane 0 of each warp stores the pass
// span.  Counter-only probe (--reps 1), one LAUNCH line per launch+pass.
#include "../mb.cuh"
#include <ifetch_bodies.hpp>  // generated into $OUT/gen by build.sh
#include <ifetch_jump.hpp>

#include <string>
#include <vector>

template <int N>
__global__ void __launch_bounds__(512, 1) k_line(std::uint64_t* out, int K) {
  std::uint32_t r[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = threadIdx.x + 17 * i;
  const int nwarp = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
#pragma unroll 1
  for (int k = 0; k < K; ++k) {
    const std::uint64_t t0 = mb_clock();
    IfetchLine<N>::run(r, K);
    const std::uint64_t t1 = mb_clock();
    if ((threadIdx.x & 31) == 0) out[1 + k * nwarp + warp] = t1 - t0;
  }
  std::uint32_t x = r[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) x ^= r[i];
  if (threadIdx.x == 0) {
    out[0] = mb_smid();
    out[1 + K * nwarp] = x;
  }
}

// k_mix<N, W>: W warps each run their own straight-line body at a
// distinct code address.  Amendment[4]: L0 privacy and L1->L0
// fill-bandwidth parallelism need per-warp distinct code; amendment[5]
// adds W=8/16 to test whether the cadence is per-stream (per warp
// stream) or per-SMSP.
template <int N, int W>
__global__ void __launch_bounds__(32 * W, 1) k_mix(std::uint64_t* out, int K) {
  std::uint32_t r[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = threadIdx.x + 17 * i;
  const int warp = threadIdx.x >> 5;
#pragma unroll 1
  for (int k = 0; k < K; ++k) {
    const std::uint64_t t0 = mb_clock();
    // if constexpr guards keep W=4 instantiations from requiring bodies
    // for warp ids they never execute.
    if (warp == 0) IfetchMix<N, 0>::run(r, K);
    else if (warp == 1) IfetchMix<N, 1>::run(r, K);
    else if (warp == 2) IfetchMix<N, 2>::run(r, K);
    else if (warp == 3) IfetchMix<N, 3>::run(r, K);
    else if constexpr (W > 4) {
      if (warp == 4) IfetchMix<N, 4>::run(r, K);
      else if (warp == 5) IfetchMix<N, 5>::run(r, K);
      else if (warp == 6) IfetchMix<N, 6>::run(r, K);
      else if (warp == 7) IfetchMix<N, 7>::run(r, K);
      else if constexpr (W > 8) {
        if (warp == 8) IfetchMix<N, 8>::run(r, K);
        else if (warp == 9) IfetchMix<N, 9>::run(r, K);
        else if (warp == 10) IfetchMix<N, 10>::run(r, K);
        else if (warp == 11) IfetchMix<N, 11>::run(r, K);
        else if (warp == 12) IfetchMix<N, 12>::run(r, K);
        else if (warp == 13) IfetchMix<N, 13>::run(r, K);
        else if (warp == 14) IfetchMix<N, 14>::run(r, K);
        else IfetchMix<N, 15>::run(r, K);
      }
    }
    const std::uint64_t t1 = mb_clock();
    if ((threadIdx.x & 31) == 0) out[1 + k * W + warp] = t1 - t0;
  }
  std::uint32_t x = r[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) x ^= r[i];
  if (threadIdx.x == 0) {
    out[0] = mb_smid();
    out[1 + K * W] = x;
  }
}

template <int M, int PERM>
__global__ void __launch_bounds__(512, 1) k_jump(std::uint64_t* out, int K) {
  std::uint32_t r[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = threadIdx.x + 17 * i;
  const int nwarp = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
#pragma unroll 1
  for (int k = 0; k < K; ++k) {
    const std::uint64_t t0 = mb_clock();
    IfetchJump<M, PERM>::run(r, K);
    const std::uint64_t t1 = mb_clock();
    if ((threadIdx.x & 31) == 0) out[1 + k * nwarp + warp] = t1 - t0;
  }
  std::uint32_t x = r[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) x ^= r[i];
  if (threadIdx.x == 0) {
    out[0] = mb_smid();
    out[1 + K * nwarp] = x;
  }
}

namespace {
constexpr int K_PASSES = 4;

void report(const std::uint64_t* h, const char* name, int launch, int w) {
  const std::uint64_t smid = h[0];
  for (int k = 0; k < K_PASSES; ++k) {
    std::string line;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "LAUNCH ifetch variant=%s launch=%d smid=%llu pass=%d cycles=",
                  name, launch, static_cast<unsigned long long>(smid), k);
    line += buf;
    for (int i = 0; i < w; ++i) {
      std::snprintf(buf, sizeof(buf), "%s%llu", i ? "," : "",
                    static_cast<unsigned long long>(h[1 + k * w + i]));
      line += buf;
    }
    std::printf("%s\n", line.c_str());
  }
}
}  // namespace

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  if (args.reps != 1) {
    std::fprintf(stderr, "counter-only probe requires --reps 1\n");
    return 2;
  }
  using K = void (*)(std::uint64_t*, int);
  const struct {
    int n;
    K f;
  } lines[] = {
#ifdef IFETCH_N
      // per-footprint build (build.sh IFETCH_N=<N>): one k_line, its k_mix
      // widths, no jump kernels; the per-N runs are concatenated by run_ncu.sh
      {IFETCH_N, &k_line<IFETCH_N>}
#else
      {256, &k_line<256>},       {512, &k_line<512>},
      {1024, &k_line<1024>},     {1280, &k_line<1280>},
      {1536, &k_line<1536>},     {1792, &k_line<1792>},
      {1856, &k_line<1856>},     {1920, &k_line<1920>},
      {1984, &k_line<1984>},     {2048, &k_line<2048>},
      {2112, &k_line<2112>},     {2176, &k_line<2176>},
      {2240, &k_line<2240>},     {2304, &k_line<2304>},
      {2432, &k_line<2432>},     {2560, &k_line<2560>},
      {2816, &k_line<2816>},     {3072, &k_line<3072>},
      {4096, &k_line<4096>},     {5120, &k_line<5120>},
      {6144, &k_line<6144>},     {7168, &k_line<7168>},
      {8192, &k_line<8192>},     {10240, &k_line<10240>},
      {12288, &k_line<12288>},   {16384, &k_line<16384>},
      {32768, &k_line<32768>}
#endif
  };
  const struct {
    int m;
    int perm;
    K f;
  } jumps[] = {
#ifndef IFETCH_N
      {16, 0, &k_jump<16, 0>},     {16, 1, &k_jump<16, 1>},
      {64, 0, &k_jump<64, 0>},     {64, 1, &k_jump<64, 1>},
      {256, 0, &k_jump<256, 0>},   {256, 1, &k_jump<256, 1>},
      {1024, 0, &k_jump<1024, 0>}, {1024, 1, &k_jump<1024, 1>},
      {4096, 0, &k_jump<4096, 0>}, {4096, 1, &k_jump<4096, 1>}
#else
      {0, 0, nullptr}
#endif
  };
  const int ws[] = {1, 4};
  const struct {
    int n;
    int w;
    K f;
  } mixes[] = {
#ifdef IFETCH_N
#define IFETCH_MIX(n, w) {n, w, &k_mix<n, w>},
#if IFETCH_N == 512 || IFETCH_N == 1536 || IFETCH_N == 1792 || IFETCH_N == 3072
      IFETCH_MIX(IFETCH_N, 4)
#elif IFETCH_N == 1024 || IFETCH_N == 2048 || IFETCH_N == 4096
      IFETCH_MIX(IFETCH_N, 4) IFETCH_MIX(IFETCH_N, 8) IFETCH_MIX(IFETCH_N, 16)
#elif IFETCH_N == 8192
      IFETCH_MIX(IFETCH_N, 4) IFETCH_MIX(IFETCH_N, 8)
#endif
#undef IFETCH_MIX
      {0, 0, nullptr}
#else
      {512, 4, &k_mix<512, 4>},     {1024, 4, &k_mix<1024, 4>},
      {1536, 4, &k_mix<1536, 4>},   {1792, 4, &k_mix<1792, 4>},
      {2048, 4, &k_mix<2048, 4>},   {3072, 4, &k_mix<3072, 4>},
      {4096, 4, &k_mix<4096, 4>},   {8192, 4, &k_mix<8192, 4>},
      {1024, 8, &k_mix<1024, 8>},   {2048, 8, &k_mix<2048, 8>},
      {4096, 8, &k_mix<4096, 8>},   {8192, 8, &k_mix<8192, 8>},
      {1024, 16, &k_mix<1024, 16>}, {2048, 16, &k_mix<2048, 16>},
      {4096, 16, &k_mix<4096, 16>}
#endif
  };
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, (2 + K_PASSES * 64) * sizeof(std::uint64_t)));
  char name[64];
  for (const auto& kv : lines) {
    for (int w : ws) {
      std::snprintf(name, sizeof(name), "ifetch.line.n%d.w%d", kv.n, w);
      if (!mb_selected(args, name)) continue;
      const int nwords = 2 + K_PASSES * w;
      for (int l = 0; l < args.launches; ++l) {
        kv.f<<<1, 32 * w>>>(d_out, K_PASSES);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        MB_CHECK(cudaGetLastError());
        std::vector<std::uint64_t> h(static_cast<std::size_t>(nwords));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToHost));
        report(h.data(), name, l, w);
      }
    }
  }
  for (const auto& kv : mixes) {
    if (!kv.f) continue;
    std::snprintf(name, sizeof(name), "ifetch.mix.n%d.w%d", kv.n, kv.w);
    if (!mb_selected(args, name)) continue;
    const int nwords = 2 + K_PASSES * kv.w;
    for (int l = 0; l < args.launches; ++l) {
      kv.f<<<1, 32 * kv.w>>>(d_out, K_PASSES);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      MB_CHECK(cudaGetLastError());
      std::vector<std::uint64_t> h(static_cast<std::size_t>(nwords));
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
      report(h.data(), name, l, kv.w);
    }
  }
  for (const auto& kv : jumps) {
    if (!kv.f) continue;
    std::snprintf(name, sizeof(name), "ifetch.jump.m%d.p%d.w1", kv.m, kv.perm);
    if (!mb_selected(args, name)) continue;
    const int nwords = 2 + K_PASSES;
    for (int l = 0; l < args.launches; ++l) {
      kv.f<<<1, 32>>>(d_out, K_PASSES);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      MB_CHECK(cudaGetLastError());
      std::vector<std::uint64_t> h(static_cast<std::size_t>(nwords));
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
      report(h.data(), name, l, 1);
    }
  }
  return 0;
}
