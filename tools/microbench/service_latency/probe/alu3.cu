// alu3: classes not covered by alu.cu/alu2.cu/misc.cu/svc.cu —
// plain register ops (dependent-chain lat + thr), warp-level mma.sync
// (HMMA/IMMA/BMMA + sparse forms), and ERRBAR fence cells.
// Scheme identical to alu.cu: lat OP U = one warp, U dependent
// instances between two clock reads; thr OP K = 16 warps x 8 chains
// x K steps between barriers.  Variant names: lat.<OP> / thr.<OP>.
#include "../mb.cuh"
#include <string>
#include <vector>

#define OP2(NAME, ROOT, ASM)                                              \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    static constexpr const char* root = ROOT;                             \
    __device__ __forceinline__ static void op(                            \
        std::uint32_t& d, std::uint32_t a, std::uint32_t b,               \
        std::uint32_t k) {                                                \
      asm volatile(ASM : "=r"(d) : "r"(a), "r"(b), "r"(k));               \
    }                                                                     \
  };
#define OPF(NAME, ROOT, ASM)                                              \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    static constexpr const char* root = ROOT;                             \
    __device__ __forceinline__ static void op(                            \
        std::uint32_t& d, std::uint32_t a, std::uint32_t b,               \
        std::uint32_t k) {                                                \
      float r;                                                            \
      asm volatile(ASM : "=f"(r)                                          \
                   : "f"(__uint_as_float(a)),                             \
                     "f"(__uint_as_float(b)),                             \
                     "f"(__uint_as_float(k)));                            \
      d = __float_as_uint(r);                                             \
    }                                                                     \
  };

// Cells with a single-instruction SASS realisation only (protocol-v2
// addendum, alu3 revision after the r3 gate): the predicate / fused /
// moved forms (DSETP, FSETP, PLOP3, LEA, MOV, UFLO, VIADD, VIADDMNMX,
// VHMNMX, HADD2.F32) have no PTX that compiles to one SASS instruction
// and stay unrealised.
OP2(IMADHI, "IMAD.HI", "mul.hi.u32 %0, %1, %2;")
// F2F chain: f32 -> f64 -> f32, two F2F per step (both F2F root); the
// derivation halves the slope
OP2(F2F_, "F2F", "{\n\t.reg .f32 a;\n\t.reg .f64 d;\n\tmov.b32 a, %1;\n\tcvt.f64.f32 d, a;\n\tcvt.rn.f32.f64 a, d;\n\tmov.b32 %0, a;\n\t}")

// ---------- lat: one warp, U dependent ops ----------
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_lat(std::uint64_t* out, const std::uint32_t* seeds, std::uint32_t* sink,
      int reps) {
  std::uint32_t x = seeds[threadIdx.x], y = seeds[32 + threadIdx.x],
                k = seeds[64 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if (i & 1) O::op(y, x, y, k); else O::op(x, y, x, k);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = x ^ y;
}

// ---------- thr: 16 warps x 8 independent chains x K steps ----------
template <class O, int K>
__global__ void __launch_bounds__(512, 1)
k_thr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  __shared__ __align__(16) std::uint32_t pinb[512 * 16];
  const std::uint32_t base =
      mb_smem_u32(pinb) + threadIdx.x * 4u;
  constexpr std::uint32_t kStride = 512u * 4u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + j * kStride), "r"(s + threadIdx.x + j));
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + (8 + j) * kStride), "r"(s * 3u + j));
  }
  std::uint32_t x[8], y[8];
  const std::uint32_t k = s ^ 0x3f800000u;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(x[j]) : "r"(base + j * kStride));
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(y[j]) : "r"(base + (8 + j) * kStride));
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (i & 1) O::op(y[j], x[j], y[j], k);
        else O::op(x[j], y[j], x[j], k);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + j * kStride), "r"(x[j]));
      asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + (8 + j) * kStride), "r"(y[j]));
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = x[0] ^ y[0];
}

// ---------- warp-level mma.sync (HMMA/IMMA/BMMA) ----------
// Dependent chain: accumulate D = A*B + D, same A/B fragments every
// step -> the accumulator is the dependency.  lat measures the
// chain slope; thr uses 4 independent accumulators per warp.
template <int U>
__global__ void __launch_bounds__(32, 1)
k_mma_f16(std::uint64_t* out, const std::uint32_t* seeds,
          std::uint32_t* sink, int reps) {
  float d[4] = {0, 0, 0, 0};
  const std::uint32_t a0 = seeds[threadIdx.x],
                      a1 = seeds[32 + threadIdx.x],
                      a2 = seeds[64 + threadIdx.x] ^ 0x5555u,
                      a3 = seeds[96 + threadIdx.x] ^ 0x3333u;
  const std::uint32_t b0 = seeds[64 + threadIdx.x],
                      b1 = seeds[96 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i)
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = __float_as_uint(d[0] + d[1] + d[2] + d[3]);
}

template <int K>
__global__ void __launch_bounds__(512, 1)
k_mma_f16_thr(std::uint64_t* out, const std::uint32_t* seeds,
              std::uint32_t* sink, int reps) {
  float d[4][4];
#pragma unroll
  for (int c = 0; c < 4; ++c)
    for (int i = 0; i < 4; ++i) d[c][i] = 0.f;
  const std::uint32_t a0 = seeds[threadIdx.x],
                      a1 = seeds[32 + threadIdx.x],
                      a2 = seeds[64 + threadIdx.x] ^ 0x5555u,
                      a3 = seeds[96 + threadIdx.x] ^ 0x3333u,
                      b0 = seeds[64 + threadIdx.x],
                      b1 = seeds[96 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i)
#pragma unroll
      for (int c = 0; c < 4; ++c)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  float acc = 0;
#pragma unroll
  for (int c = 0; c < 4; ++c) acc += d[c][0];
  sink[threadIdx.x] = __float_as_uint(acc);
}

// s8 IMMA m16n8k32
template <int U>
__global__ void __launch_bounds__(32, 1)
k_mma_s8(std::uint64_t* out, const std::uint32_t* seeds,
         std::uint32_t* sink, int reps) {
  std::uint32_t d[4] = {0, 0, 0, 0};
  const std::uint32_t a0 = seeds[threadIdx.x],
                      a1 = seeds[32 + threadIdx.x],
                      a2 = seeds[64 + threadIdx.x] ^ 0x5555u,
                      a3 = seeds[96 + threadIdx.x] ^ 0x3333u,
                      b0 = seeds[64 + threadIdx.x],
                      b1 = seeds[96 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i)
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = d[0] ^ d[1] ^ d[2] ^ d[3];
}


using Kern = void (*)(std::uint64_t*, std::uint32_t*, int, std::uint32_t);
struct Case { std::string name, mode; int u; Kern k; };

template <class O, int... Us>
void add_lat(std::vector<Case>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({"lat." + std::string(O::name), "lat", Us,
                (Kern)&k_lat<O, Us>}), ...);
}
template <class O, int... Ks>
void add_thr(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({"thr." + std::string(O::name), "thr", Ks,
                (Kern)&k_thr<O, Ks>}), ...);
}
#define MB_ADD(O)                                            \
  add_lat<O>(cases, std::integer_sequence<int,8,16,32,64>{}); \
  add_thr<O>(cases, std::integer_sequence<int,16,32,64>{})

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "alu3");
  std::vector<Case> cases;
  MB_ADD(IMADHI); MB_ADD(F2F_);
#undef MB_ADD
  // MMA cells: lat/thr for f16 m16n8k16 and s8 m16n8k32
  cases.push_back({"lat.mma.HMMA", "lat", 16, (Kern)&k_mma_f16<16>});
  cases.push_back({"lat.mma.HMMA", "lat", 32, (Kern)&k_mma_f16<32>});
  cases.push_back({"lat.mma.HMMA", "lat", 64, (Kern)&k_mma_f16<64>});
  cases.push_back({"thr.mma.HMMA", "mma_thr", 16, (Kern)&k_mma_f16_thr<16>});
  cases.push_back({"thr.mma.HMMA", "mma_thr", 32, (Kern)&k_mma_f16_thr<32>});
  cases.push_back({"thr.mma.HMMA", "mma_thr", 64, (Kern)&k_mma_f16_thr<64>});
  cases.push_back({"lat.mma.IMMA", "lat", 16, (Kern)&k_mma_s8<16>});
  cases.push_back({"lat.mma.IMMA", "lat", 32, (Kern)&k_mma_s8<32>});
  cases.push_back({"lat.mma.IMMA", "lat", 64, (Kern)&k_mma_s8<64>});
  // ERRBAR / CGAERRBAR: no PTX form; derived per protocol-v2 addendum
  // (barrier_release v2 release-minus-relaxed difference minus
  // MEMBAR.ALL.GPU), no cell here.
  // seeds: deterministic operands for every lane (lat kernels read
  // 96 words, mma kernels 128); sink: one word per thread of the
  // largest launch
  std::vector<std::uint32_t> h_seeds(128);
  for (std::size_t i = 0; i < h_seeds.size(); ++i)
    h_seeds[i] = 0x9e3779b9u * static_cast<std::uint32_t>(i + 1) + 0x3f800000u;
  std::uint64_t* d_out; std::uint32_t *d_sink, *d_seeds;
  MB_CHECK(cudaMalloc(&d_out, args.reps * 8));
  MB_CHECK(cudaMalloc(&d_sink, 512 * 4));
  MB_CHECK(cudaMalloc(&d_seeds, h_seeds.size() * 4));
  MB_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), h_seeds.size() * 4,
                      cudaMemcpyHostToDevice));
  for (const auto& c : cases) {
    if (args.only.size() && c.name.find(args.only) == std::string::npos) continue;
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      if (c.mode == "lat" || c.mode == "mma_thr") {
        auto fn = reinterpret_cast<void (*)(std::uint64_t*,
              const std::uint32_t*, std::uint32_t*, int)>(c.k);
        if (c.mode == "lat") fn<<<1, 32>>>(d_out, d_seeds, d_sink, args.reps);
        else fn<<<1, 512>>>(d_out, d_seeds, d_sink, args.reps);
      } else {
        auto fn = reinterpret_cast<void (*)(std::uint64_t*,
              std::uint32_t*, int, std::uint32_t)>(c.k);
        fn<<<1, 512>>>(d_out, d_sink, args.reps, 0x9e3779b9u + l);
      }
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(args.reps);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      for (int r = 1; r < args.reps; ++r) s.push_back(h[r]);
    }
    w.row(c.name, c.u, "\"op\":\"" + c.name + "\"", s);
  }
  cudaFree(d_out); cudaFree(d_sink); cudaFree(d_seeds);
  return 0;
}
