// alu5: service_latency protocol v4 cells — MATCH, DSETP/FSETP pair
// (pair partners live in f64/f32 registers), DADD_RZ / DMUL_RZ (non-
// default rounding so the chain is not contracted into DFMA), HFMA2
// repeat (h100), CS2R_CLOCK / CS2R_GT issue cells (window between the
// first and the last read holds exactly K reads).
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

OP2(MATCH, "MATCH", "match.any.sync.b32 %0, %1, 0xffffffff;")
OP2(HFMA2, "HFMA2", "fma.rn.f16x2 %0, %1, %2, %3;")

// f64 pair: one dependent f64 chain; step = setp.gt.f64 p, x, y ;
// selp.f64 x, a, b, p  -> {DSETP:1, FSEL:2} (the two FSEL of the
// 64-bit select are parallel: one FSEL on the critical path).
struct DSETP_P {
  static constexpr const char* name = "DSETP_P";
  static constexpr const char* root = "DSETP";
  __device__ __forceinline__ static void dop(
      double& x, double y, double a, double b) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.gt.f64 p, %0, %1;\n\t"
        "selp.f64 %0, %2, %3, p;\n\t}"
        : "+d"(x) : "d"(y), "d"(a), "d"(b));
  }
};
// f32 pair: {FSETP:1, FSEL:1}, both on the critical path.
struct FSETP_P {
  static constexpr const char* name = "FSETP_P";
  static constexpr const char* root = "FSETP";
  __device__ __forceinline__ static void fop(
      float& x, float y, float a, float b) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.gt.f32 p, %0, %1;\n\t"
        "selp.f32 %0, %2, %3, p;\n\t}"
        : "+f"(x) : "f"(y), "f"(a), "f"(b));
  }
};
struct DADD_RZ {
  static constexpr const char* name = "DADD_RZ";
  static constexpr const char* root = "DADD";
  __device__ __forceinline__ static void dop(
      double& d, double a, double b) {
    asm volatile("add.rz.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};
struct DMUL_RZ {
  static constexpr const char* name = "DMUL_RZ";
  static constexpr const char* root = "DMUL";
  __device__ __forceinline__ static void dop(
      double& d, double a, double b) {
    asm volatile("mul.rz.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};

// u32 lat/thr harness (alu3 shape) for the OP2 cells
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

// f64 pair lat: x chain dependent, y/a/b refreshed per step from the
// pinned seeds (f64 regs kept across the kernel).
template <int U>
__global__ void __launch_bounds__(32, 1)
k_lat_d64(std::uint64_t* out, const double* seeds, std::uint32_t* sink,
          int reps) {
  double x = seeds[threadIdx.x], y = seeds[32 + threadIdx.x],
         a = seeds[64 + threadIdx.x], b = seeds[96 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      DSETP_P::dop(x, y, a, b);
      DSETP_P::dop(y, x, b, a);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] =
      (std::uint32_t)(__double_as_longlong(x + y + a + b));
}
template <int U>
__global__ void __launch_bounds__(32, 1)
k_lat_f32(std::uint64_t* out, const std::uint32_t* seeds,
          std::uint32_t* sink, int reps) {
  float x = __uint_as_float(seeds[threadIdx.x]),
        y = __uint_as_float(seeds[32 + threadIdx.x]),
        a = __uint_as_float(seeds[64 + threadIdx.x]),
        b = __uint_as_float(seeds[96 + threadIdx.x]);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      FSETP_P::fop(x, y, a, b);
      FSETP_P::fop(y, x, b, a);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = __float_as_uint(x + y + a + b);
}

// f64 thr: 8 distinctly seeded chains, all pinned (alu3 shape).
template <class O, int K>
__global__ void __launch_bounds__(512, 1)
k_dthr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  double x[8], y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
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
    for (int j = 0; j < 8; ++j)
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

// f64 lat (result) for DADD_RZ / DMUL_RZ: one dependent chain.
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_dlat(std::uint64_t* out, const double* seeds, std::uint32_t* sink,
       int reps) {
  double x = seeds[threadIdx.x], y = seeds[32 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      O::dop(x, x, y);
      O::dop(y, y, x);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = (std::uint32_t)__double_as_longlong(x + y);
}

// CS2R issue cells: K back-to-back reads between first and last read;
// window = t_last - t_first.  Values summed after the window.
template <int K, int WHICH>
__global__ void __launch_bounds__(32, 1)
k_cs2r(std::uint64_t* out, std::uint64_t* sink, int reps) {
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint64_t acc = 0, t_first = 0, t_last = 0, v = 0;
#pragma unroll
    for (int i = 0; i < K; ++i) {
      if (WHICH == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(v));
      else
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
      if (i == 0) t_first = v;
      t_last = v;
      acc += v & 0xffff;
    }
    if (threadIdx.x == 0) out[rep] = t_last - t_first;
    sink[threadIdx.x] = acc;
  }
}

using Kern = void (*)(std::uint64_t*, const std::uint32_t*,
                      std::uint32_t*, int);
struct Case {
  std::string name, mode;
  int u;
  Kern k;
};

#define MB_ALU5_OP2(X)  X(MATCH) X(HFMA2)

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "alu5");
  std::vector<Case> cases;
#define MB_ADD2(O)                                                          \
  cases.push_back({"lat." #O, "lat", 16, (Kern)&k_lat<O, 16>});           \
  cases.push_back({"lat." #O, "lat", 32, (Kern)&k_lat<O, 32>});           \
  cases.push_back({"lat." #O, "lat", 64, (Kern)&k_lat<O, 64>});           \
  cases.push_back({"thr." #O, "thr", 16, (Kern)&k_thr<O, 16>});           \
  cases.push_back({"thr." #O, "thr", 32, (Kern)&k_thr<O, 32>});           \
  cases.push_back({"thr." #O, "thr", 64, (Kern)&k_thr<O, 64>});
  MB_ALU5_OP2(MB_ADD2)
#undef MB_ADD2
#define MB_ADDU(u)                                                          \
  cases.push_back({"lat.DSETP_P", "dlat", u, (Kern)&k_lat_d64<u>});        \
  cases.push_back({"lat.FSETP_P", "dlat", u, (Kern)&k_lat_f32<u>});        \
  cases.push_back({"lat.DADD_RZ", "dlat", u, (Kern)&k_dlat<DADD_RZ, u>});  \
  cases.push_back({"lat.DMUL_RZ", "dlat", u, (Kern)&k_dlat<DMUL_RZ, u>});  \
  cases.push_back({"dthr.DADD_RZ", "dthr", u, (Kern)&k_dthr<DADD_RZ, u>}); \
  cases.push_back({"dthr.DMUL_RZ", "dthr", u, (Kern)&k_dthr<DMUL_RZ, u>}); \
  cases.push_back({"issue.CS2R_CLOCK", "cs2r", u, (Kern)&k_cs2r<u, 0>});   \
  cases.push_back({"issue.CS2R_GT", "cs2r", u, (Kern)&k_cs2r<u, 1>});
  MB_ADDU(16) MB_ADDU(32) MB_ADDU(64)
#undef MB_ADDU
  std::vector<std::uint32_t> h_seeds(128);
  for (std::size_t i = 0; i < h_seeds.size(); ++i)
    h_seeds[i] = 0x9e3779b9u * static_cast<std::uint32_t>(i + 1) + 0x3f800000u;
  std::vector<double> h_dseeds(128);
  for (std::size_t i = 0; i < h_dseeds.size(); ++i)
    h_dseeds[i] = 0.5 + 0.25 * (i % 7) + 1.0e-3 * i;
  std::uint64_t* d_out; std::uint32_t *d_sink, *d_seeds;
  double* d_dseeds;
  std::uint64_t* d_sink64;
  MB_CHECK(cudaMalloc(&d_out, args.reps * 8));
  MB_CHECK(cudaMalloc(&d_sink, 512 * 4));
  MB_CHECK(cudaMalloc(&d_sink64, 512 * 8));
  MB_CHECK(cudaMalloc(&d_seeds, h_seeds.size() * 4));
  MB_CHECK(cudaMalloc(&d_dseeds, h_dseeds.size() * 8));
  MB_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), h_seeds.size() * 4,
                      cudaMemcpyHostToDevice));
  MB_CHECK(cudaMemcpy(d_dseeds, h_dseeds.data(), h_dseeds.size() * 8,
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
      } else if (c.mode == "dlat") {
        auto fn = reinterpret_cast<void (*)(std::uint64_t*,
              const double*, std::uint32_t*, int)>(c.k);
        fn<<<1, 32>>>(d_out, d_dseeds, d_sink, args.reps);
      } else if (c.mode == "dthr" || c.mode == "thr") {
        auto fn = reinterpret_cast<void (*)(std::uint64_t*,
              std::uint32_t*, int, std::uint32_t)>(c.k);
        fn<<<1, 512>>>(d_out, d_sink, args.reps, 0x9e3779b9u + l);
      } else {  // cs2r
        auto fn = reinterpret_cast<void (*)(std::uint64_t*,
              std::uint64_t*, int)>(c.k);
        fn<<<1, 32>>>(d_out, d_sink64, args.reps);
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
  cudaFree(d_dseeds); cudaFree(d_sink64);
  return 0;
}
