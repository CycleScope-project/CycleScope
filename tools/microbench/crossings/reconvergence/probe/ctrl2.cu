// Control-flow cost, version 2 (crossings/reconvergence protocol-v2):
// divergence and reconvergence (BSSY / BRA / BSYNC) one warp, between two
// clock reads.  Each kernel runs N iterations of a body with B independent
// FFMA chains per path (one FFMA per chain per iteration).
//
//   div.bB   per iteration: if (odd) chain A += ... else chain B += ...,
//            odd a pinned per-lane predicate -> BSSY / BRA / BSYNC around
//            two serialized paths every iteration
//   uni.bB   the same if/else on a pinned warp-uniform predicate toggled
//            each iteration -> a uniform branch, no divergence
//   div2.bB  two sequential independent divergent if/else regions per
//            iteration on different per-lane predicates (lane & 1, lane & 2)
//   div3.bB  three-way divergence on lane % 3 (if / else if / else, three
//            serialized paths)
//
// Every value derived from a kernel argument (the per-lane predicate, the
// uniform seed, the FFMA multiplier) is pinned in a register before the
// timed loop with mb_pin (volatile shared slot round trip) so ptxas cannot
// rematerialise it from constant memory inside the window (v1 failed the
// SASS gate on sm_90a: LDC c[0x0][0x228] in the window).
// The slope over N (N = 64, 128, 256) is the per-iteration cost.
#include "../mb.cuh"

#include <string>
#include <vector>

// Per-lane pin slot (every lane pins its own word; a shared slot indexed by
// threadIdx.x).
#define MB_PINS __shared__ std::uint32_t mb_pins[32]

template <int B, int N>
__global__ void __launch_bounds__(32, 1) k_div(std::uint64_t* out, std::uint32_t* sink, int reps, int,
                                               std::uint32_t s) {
  MB_PINS;
  float x[B], y[B];
#pragma unroll
  for (int j = 0; j < B; ++j) {
    x[j] = 1.0f + 0.001f * static_cast<float>((threadIdx.x + j + s) & 7);
    y[j] = 2.0f - 0.001f * static_cast<float>((threadIdx.x + j) & 7);
  }
  const float m0 = 1.0f + 1e-7f * static_cast<float>(s & 3);
  const float m = __uint_as_float(mb_pin(__float_as_uint(m0), &mb_pins[threadIdx.x]));
  const std::uint32_t odd = mb_pin((threadIdx.x ^ s) & 1u, &mb_pins[threadIdx.x]);  // per-lane
  const std::uint32_t n = mb_pin(N, &mb_pins[threadIdx.x]);
  const std::uint32_t i0 = mb_pin(0, &mb_pins[threadIdx.x]);  // pinned start: no zero-init in the window  // opaque bound: no UDP loop counter
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll 1
    for (std::uint32_t i = i0; i < n; ++i) {
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
  MB_PINS;
  float x[B], y[B];
#pragma unroll
  for (int j = 0; j < B; ++j) {
    x[j] = 1.0f + 0.001f * static_cast<float>((threadIdx.x + j + s) & 7);
    y[j] = 2.0f - 0.001f * static_cast<float>((threadIdx.x + j) & 7);
  }
  const float m0 = 1.0f + 1e-7f * static_cast<float>(s & 3);
  const float m = __uint_as_float(mb_pin(__float_as_uint(m0), &mb_pins[threadIdx.x]));
  const std::uint32_t k0 = mb_pin(s, &mb_pins[threadIdx.x]);  // warp-uniform seed
  const std::uint32_t n = mb_pin(N, &mb_pins[threadIdx.x]);
  const std::uint32_t i0 = mb_pin(0, &mb_pins[threadIdx.x]);  // pinned start: no zero-init in the window
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint32_t k = k0;
    const std::uint64_t t0 = mb_clock();
#pragma unroll 1
    for (std::uint32_t i = i0; i < n; ++i) {
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

template <int B, int N>
__global__ void __launch_bounds__(32, 1) k_div2(std::uint64_t* out, std::uint32_t* sink, int reps, int,
                                                std::uint32_t s) {
  MB_PINS;
  float x[B], y[B];
#pragma unroll
  for (int j = 0; j < B; ++j) {
    x[j] = 1.0f + 0.001f * static_cast<float>((threadIdx.x + j + s) & 7);
    y[j] = 2.0f - 0.001f * static_cast<float>((threadIdx.x + j) & 7);
  }
  const float m0 = 1.0f + 1e-7f * static_cast<float>(s & 3);
  const float m = __uint_as_float(mb_pin(__float_as_uint(m0), &mb_pins[threadIdx.x]));
  const std::uint32_t p1 = mb_pin(threadIdx.x & 1u, &mb_pins[threadIdx.x]);
  const std::uint32_t p2 = mb_pin(threadIdx.x & 2u, &mb_pins[threadIdx.x]);
  const std::uint32_t n = mb_pin(N, &mb_pins[threadIdx.x]);
  const std::uint32_t i0 = mb_pin(0, &mb_pins[threadIdx.x]);  // pinned start: no zero-init in the window
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll 1
    for (std::uint32_t i = i0; i < n; ++i) {
      if (p1) {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(x[j]) : "f"(m));
      } else {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(y[j]) : "f"(m));
      }
      if (p2) {
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
__global__ void __launch_bounds__(32, 1) k_div3(std::uint64_t* out, std::uint32_t* sink, int reps, int,
                                                std::uint32_t s) {
  MB_PINS;
  float x[B], y[B], z[B];
#pragma unroll
  for (int j = 0; j < B; ++j) {
    x[j] = 1.0f + 0.001f * static_cast<float>((threadIdx.x + j + s) & 7);
    y[j] = 2.0f - 0.001f * static_cast<float>((threadIdx.x + j) & 7);
    z[j] = 3.0f - 0.002f * static_cast<float>((threadIdx.x + j) & 7);
  }
  const float m0 = 1.0f + 1e-7f * static_cast<float>(s & 3);
  const float m = __uint_as_float(mb_pin(__float_as_uint(m0), &mb_pins[threadIdx.x]));
  const std::uint32_t r = threadIdx.x % 3u;
  // pinned opaque predicates: ptxas cannot tell they come from one r, so the
  // compound selector stays two binary BSSY/BSYNC regions instead of a
  // constant-bank jump table (BRX + LDC, h100-r1 gate failure)
  const std::uint32_t p0 = mb_pin(static_cast<std::uint32_t>(r == 0), &mb_pins[threadIdx.x]);
  const std::uint32_t p1 = mb_pin(static_cast<std::uint32_t>(r == 1), &mb_pins[threadIdx.x]);
  const std::uint32_t n = mb_pin(N, &mb_pins[threadIdx.x]);
  const std::uint32_t i0 = mb_pin(0, &mb_pins[threadIdx.x]);  // pinned start: no zero-init in the window
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll 1
    for (std::uint32_t i = i0; i < n; ++i) {
      if (p0) {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(x[j]) : "f"(m));
      } else if (p1) {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(y[j]) : "f"(m));
      } else {
#pragma unroll
        for (int j = 0; j < B; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(z[j]) : "f"(m));
      }
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < B; ++j) acc += x[j] + y[j] + z[j];
  sink[threadIdx.x] = __float_as_uint(acc);
}

using Kern = void (*)(std::uint64_t*, std::uint32_t*, int, int, std::uint32_t);

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
  const std::vector<C> cases = {
      MB_BR("div", k_div, 8),   MB_BR("div", k_div, 16),   MB_BR("div", k_div, 32),
      MB_BR("uni", k_uni, 8),   MB_BR("uni", k_uni, 16),   MB_BR("uni", k_uni, 32),
      MB_BR("div2", k_div2, 8), MB_BR("div2", k_div2, 16), MB_BR("div2", k_div2, 32),
      MB_BR("div3", k_div3, 8), MB_BR("div3", k_div3, 16), MB_BR("div3", k_div3, 32),
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
