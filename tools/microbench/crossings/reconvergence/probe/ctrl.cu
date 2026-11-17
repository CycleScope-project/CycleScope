// Control-flow cost: taken branches, loop back-edges, divergence and
// reconvergence (BRA / BSSY / BSYNC / WARPSYNC), one warp, between two clock
// reads.  Each kernel runs N iterations of a body with B independent FFMA
// chains (one FFMA per chain per iteration):
//
//   loop.bB.uU     a counted loop, unrolled U times (U = 1, 2, 4, 64):
//                  per-iteration time - B FFMA issue = the back-edge cost
//                  (counter + compare + taken BRA) / U
//   div.bB         per iteration: if (lane & 1) chain A += ... else chain B
//                  += ... on per-lane data -> BSSY / BRA / BSYNC around two
//                  serialized paths
//   uni.bB         per iteration: if (k & 1) (k warp-uniform, from a kernel
//                  argument, flipped each iteration) -> a uniform branch,
//                  no divergence, taken every other iteration
// The slope over N (N = 64, 128, 256) is the per-iteration cost; the SASS
// gate records the loop body (instructions and their stalls) so the cost
// beyond the SASS stall counts is visible.
#include "../mb.cuh"

#include <string>
#include <utility>
#include <vector>

template <int B, int U, int N>
__global__ void __launch_bounds__(32, 1) k_loop(std::uint64_t* out, std::uint32_t* sink, int reps, int,
                                                std::uint32_t s) {
  float x[B];
#pragma unroll
  for (int j = 0; j < B; ++j) x[j] = 1.0f + 0.001f * static_cast<float>((threadIdx.x + j + s) & 7);
  const float m = 1.0f + 1e-7f * static_cast<float>(s & 3);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll U
    for (int i = 0; i < N; ++i) {
#pragma unroll
      for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(x[j]) : "f"(m));
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < B; ++j) acc += x[j];
  sink[threadIdx.x] = __float_as_uint(acc);
}

template <int B, int N>
__global__ void __launch_bounds__(32, 1) k_div(std::uint64_t* out, std::uint32_t* sink, int reps, int,
                                               std::uint32_t s) {
  float x[B], y[B];
#pragma unroll
  for (int j = 0; j < B; ++j) {
    x[j] = 1.0f + 0.001f * static_cast<float>((threadIdx.x + j + s) & 7);
    y[j] = 2.0f - 0.001f * static_cast<float>((threadIdx.x + j) & 7);
  }
  const float m = 1.0f + 1e-7f * static_cast<float>(s & 3);
  const bool odd = (threadIdx.x ^ s) & 1u;  // per-lane: the warp diverges every iteration
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll 1
    for (int i = 0; i < N; ++i) {
      if (odd) {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(x[j]) : "f"(m));
      } else {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(y[j]) : "f"(m));
      }
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < B; ++j) acc += x[j] + y[j];
  sink[threadIdx.x] = __float_as_uint(acc);
}

template <int B, int N>
__global__ void __launch_bounds__(32, 1) k_uni(std::uint64_t* out, std::uint32_t* sink, int reps, int,
                                               std::uint32_t s) {
  float x[B], y[B];
#pragma unroll
  for (int j = 0; j < B; ++j) {
    x[j] = 1.0f + 0.001f * static_cast<float>((threadIdx.x + j + s) & 7);
    y[j] = 2.0f - 0.001f * static_cast<float>((threadIdx.x + j) & 7);
  }
  const float m = 1.0f + 1e-7f * static_cast<float>(s & 3);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint32_t k = s;  // warp-uniform
    const std::uint64_t t0 = mb_clock();
#pragma unroll 1
    for (int i = 0; i < N; ++i) {
      if (k & 1u) {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(x[j]) : "f"(m));
      } else {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(y[j]) : "f"(m));
      }
      k ^= 1u;
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < B; ++j) acc += x[j] + y[j];
  sink[threadIdx.x] = __float_as_uint(acc);
}

using Kern = void (*)(std::uint64_t*, std::uint32_t*, int, int, std::uint32_t);

#define MB_LOOP(B, U)                                                                        \
  {"loop.b" #B ".u" #U, 64, &k_loop<B, U, 64>}, {"loop.b" #B ".u" #U, 128, &k_loop<B, U, 128>}, \
      {"loop.b" #B ".u" #U, 256, &k_loop<B, U, 256>}
#define MB_BR(KIND, KER, B) \
  {KIND ".b" #B, 64, &KER<B, 64>}, {KIND ".b" #B, 128, &KER<B, 128>}, {KIND ".b" #B, 256, &KER<B, 256>}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "ctrl");
  struct C {
    std::string variant;
    int n;
    Kern k;
  };
  // ptxas if-converts short if/else bodies into predicated FFMAs; 16 FFMA
  // per path keep real branches (BSSY/BSYNC for the divergent form)
  const std::vector<C> cases = {
      MB_LOOP(1, 1), MB_LOOP(1, 2), MB_LOOP(1, 4), MB_LOOP(1, 64), MB_LOOP(4, 1), MB_LOOP(4, 2),
      MB_LOOP(4, 4), MB_LOOP(4, 64), MB_BR("div", k_div, 16), MB_BR("uni", k_uni, 16),
      MB_BR("div", k_div, 8), MB_BR("div", k_div, 32),
      MB_BR("uni", k_uni, 8), MB_BR("uni", k_uni, 32),
  };
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * args.reps));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * 32));
  for (const auto& c : cases) {
    if (!mb_selected(args, c.variant)) continue;
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      c.k<<<1, 32>>>(d_out, d_sink, args.reps, c.n, static_cast<std::uint32_t>(l));
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      mb_collect(d_out, args.reps, 1, 0, s);
    }
    w.row(c.variant, c.n, "\"n\":" + std::to_string(c.n), s);
  }
  return w.failures ? 1 : 0;
}
