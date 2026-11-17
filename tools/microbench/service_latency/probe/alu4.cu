// alu4: service_latency protocol v3 cells — the placeholder classes
// left after the 2026-11-05 cleanups.  Harness identical to alu3.cu:
// lat OP U = one warp, U dependent instances between two clock reads;
// thr OP K = 16 warps x 8 chains x K steps between barriers.
// The gate (sass_check_alu4.py) credits the class it finds, not the
// intended one; cells that provoke a different root are unrealised
// for the intended class.  Cells with no plausible PTX idiom are
// listed in cells-v3.json as unrealised:no_form.
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

// -- SINGLE cells (composition {T:1}) ----------------------------------
OP2(VIADDMNMX, "VIADDMNMX", "{\n\t.reg .s32 t;\n\tadd.s32 t, %1, %3;\n\tmin.s32 %0, t, %2;\n\t}")
OP2(VIMNMX3, "VIMNMX3", "{\n\t.reg .s32 t;\n\tmin.s32 t, %1, %2;\n\tmin.s32 %0, t, %3;\n\t}")
OP2(SHFL_BFLY, "SHFL.BFLY", "shfl.sync.bfly.b32 %0, %1, 5, 0x1f, 0xffffffff;")
OP2(SHFL_UP, "SHFL.UP", "shfl.sync.up.b32 %0, %1, 1, 0x0, 0xffffffff;")
OP2(SHFL_DOWN, "SHFL.DOWN", "shfl.sync.down.b32 %0, %1, 2, 0x1f, 0xffffffff;")
OP2(SHFL_IDX, "SHFL.IDX", "shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;")
OP2(VOTE, "VOTE", "{\n\t.reg .pred p, r;\n\tsetp.ne.u32 p, %1, 0;\n\t"
    "vote.sync.any.pred r, p, 0xffffffff;\n\tselp.u32 %0, 1, 0, r;\n\t}")
OP2(SEL, "SEL", "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t"
    "selp.b32 %0, %2, %1, p;\n\t}")
OP2(HFMA2, "HFMA2", "fma.rn.f16x2 %0, %1, %2, %3;")
// -- PAIR cells --------------------------------------------------------
OP2(DSETP, "DSETP", "{\n\t.reg .pred p;\n\t.reg .f64 fa, fb;\n\t.reg .f32 o;\n\t"
    "cvt.rn.f64.u32 fa, %1;\n\tcvt.rn.f64.u32 fb, %2;\n\t"
    "setp.gt.f64 p, fa, fb;\n\tselp.f32 o, 1.0, 0.0, p;\n\t"
    "cvt.rzi.u32.f32 %0, o;\n\t}")
OP2(CREDUX, "CREDUX", "redux.sync.min.u32 %0, %1, 0xffffffff;")
OP2(REDUX, "REDUX", "redux.sync.add.u32 %0, %1, 0xffffffff;")
OP2(IABS, "IABS", "{\n\t.reg .b32 t;\n\tabs.s32 t, %1;\n\t"
    "xor.b32 %0, t, %3;\n\t}")
OP2(ISETP, "ISETP", "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, %1, %3;\n\t"
    "selp.b32 %0, %2, %1, p;\n\t}")
OP2(FSETP, "FSETP", "{\n\t.reg .pred p;\n\t.reg .f32 fa, fb;\n\t"
    "cvt.rn.f32.u32 fa, %1;\n\tcvt.rn.f32.u32 fb, %2;\n\t"
    "setp.lt.f32 p, fa, fb;\n\tselp.b32 %0, %1, %2, p;\n\t}")
// -- f64 thr cells (8 distinctly seeded chains, no lat cell) ------------
struct DADD {
  static constexpr const char* name = "DADD";
  static constexpr const char* root = "DADD";
  __device__ __forceinline__ static void dop(
      double& d, double a, double b) {
    asm volatile("add.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};
struct DFMA {
  static constexpr const char* name = "DFMA";
  static constexpr const char* root = "DFMA";
  __device__ __forceinline__ static void dop(
      double& d, double a, double b) {
    asm volatile("fma.rn.f64 %0, %1, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};
struct DMUL {
  static constexpr const char* name = "DMUL";
  static constexpr const char* root = "DMUL";
  __device__ __forceinline__ static void dop(
      double& d, double a, double b) {
    asm volatile("mul.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};

// ---------- kernels (alu3 shape) ----------
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

template <class O, int K>
__global__ void __launch_bounds__(512, 1)
k_thr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  __shared__ __align__(16) std::uint32_t pinb[512 * 16];
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 4u;
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

template <class O, int K>
__global__ void __launch_bounds__(512, 1)
k_dthr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  double x[8], y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {           // distinct seeds per chain
    x[j] = 1.0001 * (s + j * 7919) + j;
    y[j] = 0.9999 * (s + j * 104729) - j;
  }
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        O::dop(x[j], x[j], y[j]);
#pragma unroll
    for (int j = 0; j < 8; ++j)   // keep all 8 chains live through t1
      asm volatile("" ::"d"(x[j]), "d"(y[j]) : "memory");
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t f = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    f ^= (std::uint32_t)__double_as_longlong(x[j] + y[j]);
  sink[threadIdx.x] = f;
}

using Kern = void (*)(std::uint64_t*, const std::uint32_t*,
                      std::uint32_t*, int);
struct Case {
  std::string name, mode;
  int u;
  Kern k;
};

#define MB_ALU4_OPS(X)                                                     \
  X(VIADDMNMX) X(VIMNMX3) X(SHFL_BFLY) X(SHFL_UP) X(SHFL_DOWN)            \
  X(SHFL_IDX) X(VOTE) X(SEL) X(HFMA2) X(DSETP) X(CREDUX) X(REDUX)        \
  X(IABS) X(ISETP) X(FSETP)

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "alu4");
  std::vector<Case> cases;
#define MB_ADD(O)                                                          \
  cases.push_back({"lat." #O, "lat", 16, (Kern)&k_lat<O, 16>});           \
  cases.push_back({"lat." #O, "lat", 32, (Kern)&k_lat<O, 32>});           \
  cases.push_back({"lat." #O, "lat", 64, (Kern)&k_lat<O, 64>});           \
  cases.push_back({"thr." #O, "thr", 16, (Kern)&k_thr<O, 16>});           \
  cases.push_back({"thr." #O, "thr", 32, (Kern)&k_thr<O, 32>});           \
  cases.push_back({"thr." #O, "thr", 64, (Kern)&k_thr<O, 64>});
  MB_ALU4_OPS(MB_ADD)
#undef MB_ADD
  for (int kk : {16, 32, 64}) {
    if (kk == 16) {
      cases.push_back({"dthr.DFMA", "dthr", 16, (Kern)&k_dthr<DFMA, 16>});
      cases.push_back({"dthr.DADD", "dthr", 16, (Kern)&k_dthr<DADD, 16>});
      cases.push_back({"dthr.DMUL", "dthr", 16, (Kern)&k_dthr<DMUL, 16>});
    } else if (kk == 32) {
      cases.push_back({"dthr.DFMA", "dthr", 32, (Kern)&k_dthr<DFMA, 32>});
      cases.push_back({"dthr.DADD", "dthr", 32, (Kern)&k_dthr<DADD, 32>});
      cases.push_back({"dthr.DMUL", "dthr", 32, (Kern)&k_dthr<DMUL, 32>});
    } else {
      cases.push_back({"dthr.DFMA", "dthr", 64, (Kern)&k_dthr<DFMA, 64>});
      cases.push_back({"dthr.DADD", "dthr", 64, (Kern)&k_dthr<DADD, 64>});
      cases.push_back({"dthr.DMUL", "dthr", 64, (Kern)&k_dthr<DMUL, 64>});
    }
  }
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
    if (args.only.size() && c.name.find(args.only) == std::string::npos)
      continue;
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      if (c.mode == "lat") {
        auto fn = reinterpret_cast<void (*)(std::uint64_t*,
              const std::uint32_t*, std::uint32_t*, int)>(c.k);
        fn<<<1, 32>>>(d_out, d_seeds, d_sink, args.reps);
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
