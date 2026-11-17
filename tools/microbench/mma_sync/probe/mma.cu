// mma_sync protocol v1 (tools/microbench/mma_sync/protocol.json):
// warp-level mma.sync / mma.sp.sync result latency and per-SMSP issue
// interval for DMMA / HMMA / IMMA / sparse classes, on the
// service_latency alu.cu harness (same mb.cuh clock window, same k_lat /
// k_thr shape, same jsonl output).
//
//   lat CELL U   one warp, U dependent mma instances between two clock
//                reads; each instance's D accumulator registers are the
//                next instance's C, so the chain is a true dependence
//                (A / B fragments are loop-invariant seed registers).
//   thr CELL K   4 warps (wps = 1, one per SMSP), each runs 8 independent
//                accumulator chains x K steps between bar.sync's.
// The SASS gate requires exactly U (8K) cell-shaped instructions in the
// window and nothing else but pin traffic / loop control.
#include "../../service_latency/mb.cuh"

#include <cuda_fp16.h>

#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------- cells
// Each cell: register-fragment types for A (packed input), B, and the
// D/C accumulator; run() executes one mma whose C operand is D itself.
struct DMMA {
  static constexpr const char* name = "DMMA";
  static constexpr const char* root = "DMMA";
  static constexpr const char* shape = "8x8x4";
  static constexpr bool sparse = false;
  using A = double; using B = double; using C = double;
  static constexpr int NC = 2;
  __device__ __forceinline__ static void run(double d[NC],
                                             const A a[1], const B b[1]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+d"(d[0]), "+d"(d[1])
        : "d"(a[0]), "d"(b[0]));
#endif
  }
};

// 16x8x16 f16/bf16 -> f32 accumulator (HMMA.16816*).
struct HMMA_F16 {
  static constexpr const char* name = "HMMA_F16";
  static constexpr const char* root = "HMMA";
  static constexpr const char* shape = "16816";
  static constexpr bool sparse = false;
  using A = std::uint32_t; using B = std::uint32_t; using C = float;
  static constexpr int NC = 4;
  __device__ __forceinline__ static void run(float d[NC],
                                             const A a[4], const B b[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
  }
};

struct HMMA_BF16 {
  static constexpr const char* name = "HMMA_BF16";
  static constexpr const char* root = "HMMA";
  static constexpr const char* shape = "16816";
  static constexpr bool sparse = false;
  using A = std::uint32_t; using B = std::uint32_t; using C = float;
  static constexpr int NC = 4;
  __device__ __forceinline__ static void run(float d[NC],
                                             const A a[4], const B b[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
  }
};

// 16x8x32 s8 -> s32 accumulator (IMMA.16832.S8).
struct IMMA_S8 {
  static constexpr const char* name = "IMMA_S8";
  static constexpr const char* root = "IMMA";
  static constexpr const char* shape = "16832";
  static constexpr bool sparse = false;
  using A = std::uint32_t; using B = std::uint32_t; using C = std::uint32_t;
  static constexpr int NC = 4;
  __device__ __forceinline__ static void run(std::uint32_t d[NC],
                                             const A a[4], const B b[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
  }
};

// 16x8x32 f16 sparse (sparse HMMA; metadata register + selector 0).
struct HMMA_SP {
  static constexpr const char* name = "HMMA_SP";
  static constexpr const char* root = "HMMA";
  static constexpr const char* shape = "16832";
  static constexpr bool sparse = true;
  using A = std::uint32_t; using B = std::uint32_t; using C = float;
  static constexpr int NC = 4;
  __device__ __forceinline__ static void run(float d[NC],
                                             const A a[4], const B b[4]) {
#if __CUDA_ARCH__ >= 800
    const std::uint32_t meta = 0x44444444u;
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
        "{%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(meta));
#endif
  }
};

// 16x8x64 s8 sparse (sparse IMMA; metadata register + selector 0).
struct IMMA_SP {
  static constexpr const char* name = "IMMA_SP";
  static constexpr const char* root = "IMMA";
  static constexpr const char* shape = "16864";
  static constexpr bool sparse = true;
  using A = std::uint32_t; using B = std::uint32_t; using C = std::uint32_t;
  static constexpr int NC = 4;
  __device__ __forceinline__ static void run(std::uint32_t d[NC],
                                             const A a[4], const B b[4]) {
#if __CUDA_ARCH__ >= 800
    const std::uint32_t meta = 0x44444444u;
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
        "{%0,%1,%2,%3}, %12, 0x0;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(meta));
#endif
  }
};

// 16x8x256 b1 and.popc -> s32 accumulator (BMMA; protocol v2).
struct BMMA {
  static constexpr const char* name = "BMMA";
  static constexpr const char* root = "BMMA";
  static constexpr const char* shape = "";
  static constexpr bool sparse = false;
  using A = std::uint32_t; using B = std::uint32_t; using C = std::uint32_t;
  static constexpr int NC = 4;
  __device__ __forceinline__ static void run(std::uint32_t d[NC],
                                             const A a[4], const B b[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
  }
};

// ---------------------------------------------------------------- seeds
template <class T>
__device__ __forceinline__ T mma_seed(std::uint32_t s) {
  if constexpr (std::is_same_v<T, float>)
    return 0.0009765625f + static_cast<float>(s & 15) * 0.001f;  // ~1/1024
  else if constexpr (std::is_same_v<T, double>)
    return 0.0009765625 + static_cast<double>(s & 7) * 0.001;   // ~0.001
  else
    return 0x3c023c02u + (s & 7);            // small packed halves
}

template <class C, int N>
__device__ __forceinline__ void mma_seed_frag(C f[N], std::uint32_t s) {
#pragma unroll
  for (int i = 0; i < N; ++i) f[i] = mma_seed<C>(s + 3 * i);
}

template <class C>
__device__ __forceinline__ std::uint32_t mma_fold(C v) {
  if constexpr (std::is_same_v<C, float>) return __float_as_uint(v);
  else if constexpr (std::is_same_v<C, double>)
    return static_cast<std::uint32_t>(__double_as_longlong(v));
  else return static_cast<std::uint32_t>(v);
}

// fragment element counts per cell: A regs, B regs
template <class O> struct frag_n;
template <> struct frag_n<DMMA>     { static constexpr int A = 1, B = 1; };
template <> struct frag_n<HMMA_F16> { static constexpr int A = 4, B = 2; };
template <> struct frag_n<HMMA_BF16>{ static constexpr int A = 4, B = 2; };
template <> struct frag_n<IMMA_S8>  { static constexpr int A = 4, B = 2; };
template <> struct frag_n<HMMA_SP>  { static constexpr int A = 4, B = 4; };
template <> struct frag_n<IMMA_SP>  { static constexpr int A = 4, B = 4; };
template <> struct frag_n<BMMA>     { static constexpr int A = 4, B = 2; };

// ---------------------------------------------------------------- kernels
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_lat(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  constexpr int NA = frag_n<O>::A, NB = frag_n<O>::B, NC = O::NC;
  typename O::C d[NC];
  typename O::A a[NA];
  typename O::B b[NB];
  // A / B loop-invariant from the seed; C seeded per lane
  mma_seed_frag<typename O::A, NA>(a, s + threadIdx.x);
  mma_seed_frag<typename O::B, NB>(b, s * 3u + 1u + 5u * threadIdx.x);
  mma_seed_frag<typename O::C, NC>(d, s * 7u + 3u + threadIdx.x);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) O::run(d, a, b);
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t fold = 0;
#pragma unroll
  for (int i = 0; i < NC; ++i) fold ^= mma_fold(d[i]);
  sink[threadIdx.x] = fold ^ mma_fold(a[0]) ^ mma_fold(b[0]);
}

// 8 independent accumulator chains per warp; seeds pinned through
// volatile shared like alu.cu so ptxas cannot hoist the chains out of
// the clock window (the loads/stores are inside it by construction).
// WPS warps per SMSP: launch 128 * WPS threads (4 x WPS warps), all
// warps run their chains.  pins per warp = 8 x NC words x 32 lanes
// (NC KiB); dynamic shared so WPS = 4 x NC = 4 (64 KiB) fits past the
// 48 KiB static limit.  Thread 0's window is recorded as today.
template <class O, int K, int WPS>
__global__ void __launch_bounds__(512, 1)
k_thr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  constexpr int NA = frag_n<O>::A, NB = frag_n<O>::B, NC = O::NC;
  extern __shared__ __align__(16) std::uint32_t pinb[];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5;
  const bool active = true;  // every warp runs (WPS per SMSP)
  constexpr std::uint32_t kPerWarp = 8u * NC * 32u;  // words per warp
  const std::uint32_t base =
      mb_smem_u32(pinb + warp * kPerWarp) + lane * 4u;
  constexpr std::uint32_t kStride = 32u * 4u;   // 32 lanes, one word each
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < NC; ++i)
      asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(
                       base + static_cast<std::uint32_t>((j * NC + i)) * kStride),
                   "r"(mma_seed<std::uint32_t>(s + lane + j + i)) : "memory");
  }
  typename O::A a[NA];
  typename O::B b[NB];
  mma_seed_frag<typename O::A, NA>(a, s * 11u + lane);
  mma_seed_frag<typename O::B, NB>(b, s * 13u + lane);
  // pin the seeds in registers: ptxas may recompute the mask arithmetic
  // (LOP3 & 7) inside the rep window on sm_90a (r4 stray)
#pragma unroll
  for (int i = 0; i < NA; ++i) asm volatile("" ::"r"(mma_fold(a[i])) : "memory");
#pragma unroll
  for (int i = 0; i < NB; ++i) asm volatile("" ::"r"(mma_fold(b[i])) : "memory");
  typename O::C d[8][NC];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    // pin reload + conversion must stay OUTSIDE the clock window: it is
    // seed traffic, not chain work (r3 gate: I2FP / I2F / LOP3 strays)
    if (active) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < NC; ++i) {
          std::uint32_t v;
          asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) :
                       "r"(base + static_cast<std::uint32_t>(j * NC + i) * kStride) : "memory");
          d[j][i] = static_cast<typename O::C>(v);
        }
    }
    const std::uint64_t t0 = mb_clock();
    if (active) {
#pragma unroll
      for (int i = 0; i < K; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) O::run(d[j], a, b);
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < NC; ++i)
          asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(
                           base + static_cast<std::uint32_t>(j * NC + i) * kStride),
                       "r"(mma_fold(d[j][i])) : "memory");
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t v;
  asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(base)
               : "memory");
  sink[threadIdx.x] = v ^ mma_fold(d[0][0]);
}

// ---------------------------------------------------------------- cases
struct Case {
  std::string op, mode;
  int u, chains, wps, smem;
  void (*k)(std::uint64_t*, std::uint32_t*, int, std::uint32_t);
};

template <class O, int... Us>
void add_lat(std::vector<Case>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({O::name, "lat", Us, 1, 0, 0, &k_lat<O, Us>}), ...);
}
template <class O, int K, int W>
void add_thr1(std::vector<Case>& v) {
  // 4 * W warps x 8 chains x NC words x 32 lanes x 4 B
  constexpr int smem = 4 * W * 8 * O::NC * 32 * 4;
  v.push_back({O::name, "thr", K, 8, W, smem, &k_thr<O, K, W>});
}
template <class O, int... Ks>
void add_thr(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {
  (add_thr1<O, Ks, 1>(v), ...);
  (add_thr1<O, Ks, 2>(v), ...);
  (add_thr1<O, Ks, 4>(v), ...);
}

#define MB_CELL(O)                                               \
  add_lat<O>(cases, std::integer_sequence<int, 4, 8, 16, 32>{}); \
  add_thr<O>(cases, std::integer_sequence<int, 4, 8, 16>{});

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "mma");
  std::vector<Case> cases;
  MB_CELL(DMMA) MB_CELL(HMMA_F16) MB_CELL(HMMA_BF16) MB_CELL(IMMA_S8)
      MB_CELL(HMMA_SP) MB_CELL(IMMA_SP) MB_CELL(BMMA)
#undef MB_CELL
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * args.reps));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * 512));
  for (const auto& c : cases) {
    const std::string variant =
        c.mode + "." + c.op +
        (c.wps ? ".wps" + std::to_string(c.wps) : "");
    if (!mb_selected(args, variant)) continue;
    if (c.smem > 48 * 1024)
      MB_CHECK(cudaFuncSetAttribute(
          c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, c.smem));
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      c.k<<<1, c.wps ? 128 * c.wps : 32, c.smem>>>(
          d_out, d_sink, args.reps, static_cast<std::uint32_t>(l));
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      mb_collect(d_out, args.reps, 1, 0, s);
    }
    w.row(variant, c.u,
          "\"op\":\"" + c.op + "\",\"mode\":\"" + c.mode +
              "\",\"wps\":" + std::to_string(c.wps) +
              ",\"chains_per_thread\":" + std::to_string(c.chains),
          s);
  }
  return w.failures ? 1 : 0;
}
