// alu6: service_latency protocol v5 cells — CS2R_CLOCK / CS2R_GT issue
// windows, HFMA2 extended-K, DADD/DMUL .RN (survey-chosen idioms), the
// b300 v4 FP64 rerun cells (.RZ + DSETP_P, bodies identical to alu5),
// and MATCH distinct-value sweep d in {1,2,4,8,16,32}.
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

OP2(HFMA2, "HFMA2", "fma.rn.f16x2 %0, %1, %2, %3;")

// f64 cells, alu5 bodies unchanged (RZ = v4 rerun; RN = survey choice).
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
struct DADD_RN {
  static constexpr const char* name = "DADD_RN";
  static constexpr const char* root = "DADD";
  __device__ __forceinline__ static void dop(
      double& d, double a, double b) {
    asm volatile("add.rn.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};
struct DMUL_RN {
  static constexpr const char* name = "DMUL_RN";
  static constexpr const char* root = "DMUL";
  __device__ __forceinline__ static void dop(
      double& d, double a, double b) {
    asm volatile("mul.rn.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};
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

// ---- harness (alu5 shapes verbatim) ------------------------------------
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

// DSETP_P f64 lat (alu5 k_lat_d64 body).
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

// ---- v5 cells ----------------------------------------------------------
// issue.CS2R_CLOCK: K reads; first = t_first, last = t_last, the K - 2
// middle reads write one scratch register that is never read.
template <int K>
__global__ void __launch_bounds__(32, 1)
k_clock(std::uint64_t* out, std::uint64_t* sink, int reps) {
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint64_t scratch = 0, t_first, t_last;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_first));
#pragma unroll
    for (int i = 0; i < K - 2; ++i)
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(scratch));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_last));
    if (threadIdx.x == 0) out[rep] = t_last - t_first;
    sink[threadIdx.x] = scratch;
  }
}

// issue.CS2R_GT: t0 / t1 are clock64; K globaltimer reads between them;
// all but the last write one scratch register, the last is folded into
// the sink after t1.
template <int K>
__global__ void __launch_bounds__(32, 1)
k_gt(std::uint64_t* out, std::uint64_t* sink, int reps) {
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint64_t scratch = 0, last = 0;
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K - 1; ++i)
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(scratch));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(last));
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    sink[threadIdx.x] = scratch + last;
  }
}

// lat.MATCH.d<d>: one dependent chain — the result mask feeds the next
// MATCH (lanes of one group share one mask so the distinct count stays
// d): v_{i+1} = m_i + lane % d.
template <int D, int U>
__global__ void __launch_bounds__(32, 1)
k_mlat(std::uint64_t* out, std::uint32_t* sink, int reps,
       std::uint32_t seed) {
  const std::uint32_t lane = threadIdx.x & 31;
  std::uint32_t v = seed + (lane % D);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      std::uint32_t m;
      asm volatile("match.any.sync.b32 %0, %1, 0xffffffff;"
                   : "=r"(m) : "r"(v));
      v = m;  // lanes of one group share one mask -> d stays
    }
    const std::uint64_t t1 = mb_clock();
    if (lane == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = v;
}

// thr.MATCH.d<d>: 8 independent chains, K steps each (k_dthr shape).
template <int D, int K>
__global__ void __launch_bounds__(32, 1)
k_mthr(std::uint64_t* out, std::uint32_t* sink, int reps,
       std::uint32_t seed) {
  const std::uint32_t lane = threadIdx.x & 31;
  std::uint32_t v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    v[j] = seed + j * 104729 + (lane % D);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        std::uint32_t m;
        asm volatile("match.any.sync.b32 %0, %1, 0xffffffff;"
                     : "=r"(m) : "r"(v[j]));
        v[j] = m;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      asm volatile("" ::"r"(v[j]) : "memory");
    const std::uint64_t t1 = mb_clock();
    if (lane == 0) out[rep] = t1 - t0;
  }
  std::uint32_t f = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) f ^= v[j];
  sink[threadIdx.x] = f;
}

using Kern = void (*)(std::uint64_t*, const std::uint32_t*,
                      std::uint32_t*, int);

struct Case {
  std::string name, mode;
  int u;
  int which;  // launch shape
};

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "alu6");
  std::vector<Case> cases;
  // HFMA2: lat U {16,32,64}, thr K extended to {16,24,32,48,64,96,128}
  for (int u : {16, 32, 64})
    cases.push_back({"lat.HFMA2", "lat", u, 0});
  for (int k : {16, 24, 32, 48, 64, 96, 128})
    cases.push_back({"thr.HFMA2", "thr", k, 0});
  // f64 cells: lat + dthr, U/K {16,32,64}
#define ADD64(O)                                                          \
  for (int u : {16, 32, 64}) {                                            \
    cases.push_back({"lat." #O, "dlat", u, 0});                           \
    cases.push_back({"dthr." #O, "dthr", u, 0});                          \
  }
  ADD64(DADD_RN) ADD64(DMUL_RN) ADD64(DADD_RZ) ADD64(DMUL_RZ)
#undef ADD64
  for (int u : {16, 32, 64})
    cases.push_back({"lat.DSETP_P", "dlat64", u, 0});
  // issue cells
  for (int k : {16, 32, 64}) {
    cases.push_back({"issue.CS2R_CLOCK", "clock", k, 0});
    cases.push_back({"issue.CS2R_GT", "gt", k, 0});
  }
  // MATCH distinct-value sweep
  for (int d : {1, 2, 4, 8, 16, 32}) {
    for (int u : {16, 32, 64}) {
      cases.push_back({"lat.MATCH_d" + std::to_string(d), "mlat", u, d});
      cases.push_back({"thr.MATCH_d" + std::to_string(d), "mthr", u, d});
    }
  }
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
      const std::uint32_t seed = 0x9e3779b9u + l * 97;
      if (c.mode == "lat") {
        int u = c.u;
        void (*fn)(std::uint64_t*, const std::uint32_t*,
                   std::uint32_t*, int) =
            u == 16 ? &k_lat<HFMA2, 16>
                    : u == 32 ? &k_lat<HFMA2, 32> : &k_lat<HFMA2, 64>;
        fn<<<1, 32>>>(d_out, d_seeds, d_sink, args.reps);
      } else if (c.mode == "thr") {
        int k = c.u;
        void (*fn)(std::uint64_t*, std::uint32_t*, int,
                   std::uint32_t) = nullptr;
        switch (k) {
          case 16: fn = &k_thr<HFMA2, 16>; break;
          case 24: fn = &k_thr<HFMA2, 24>; break;
          case 32: fn = &k_thr<HFMA2, 32>; break;
          case 48: fn = &k_thr<HFMA2, 48>; break;
          case 64: fn = &k_thr<HFMA2, 64>; break;
          case 96: fn = &k_thr<HFMA2, 96>; break;
          default: fn = &k_thr<HFMA2, 128>; break;
        }
        fn<<<1, 512>>>(d_out, d_sink, args.reps, seed);
      } else if (c.mode == "dlat" || c.mode == "dthr") {
        void (*fl)(std::uint64_t*, const double*, std::uint32_t*,
                   int) = nullptr;
        void (*ft)(std::uint64_t*, std::uint32_t*, int,
                   std::uint32_t) = nullptr;
        int u = c.u;
        const std::string& n = c.name;
        if (c.mode == "dlat") {
          if (n == "lat.DADD_RN")
            fl = u == 16 ? &k_dlat<DADD_RN, 16>
                 : u == 32 ? &k_dlat<DADD_RN, 32> : &k_dlat<DADD_RN, 64>;
          else if (n == "lat.DMUL_RN")
            fl = u == 16 ? &k_dlat<DMUL_RN, 16>
                 : u == 32 ? &k_dlat<DMUL_RN, 32> : &k_dlat<DMUL_RN, 64>;
          else if (n == "lat.DADD_RZ")
            fl = u == 16 ? &k_dlat<DADD_RZ, 16>
                 : u == 32 ? &k_dlat<DADD_RZ, 32> : &k_dlat<DADD_RZ, 64>;
          else
            fl = u == 16 ? &k_dlat<DMUL_RZ, 16>
                 : u == 32 ? &k_dlat<DMUL_RZ, 32> : &k_dlat<DMUL_RZ, 64>;
          fl<<<1, 32>>>(d_out, d_dseeds, d_sink, args.reps);
        } else {
          if (n == "dthr.DADD_RN")
            ft = u == 16 ? &k_dthr<DADD_RN, 16>
                 : u == 32 ? &k_dthr<DADD_RN, 32> : &k_dthr<DADD_RN, 64>;
          else if (n == "dthr.DMUL_RN")
            ft = u == 16 ? &k_dthr<DMUL_RN, 16>
                 : u == 32 ? &k_dthr<DMUL_RN, 32> : &k_dthr<DMUL_RN, 64>;
          else if (n == "dthr.DADD_RZ")
            ft = u == 16 ? &k_dthr<DADD_RZ, 16>
                 : u == 32 ? &k_dthr<DADD_RZ, 32> : &k_dthr<DADD_RZ, 64>;
          else
            ft = u == 16 ? &k_dthr<DMUL_RZ, 16>
                 : u == 32 ? &k_dthr<DMUL_RZ, 32> : &k_dthr<DMUL_RZ, 64>;
          ft<<<1, 512>>>(d_out, d_sink, args.reps, seed);
        }
      } else if (c.mode == "dlat64") {
        void (*fn)(std::uint64_t*, const double*, std::uint32_t*,
                   int) = c.u == 16 ? &k_lat_d64<16>
                            : c.u == 32 ? &k_lat_d64<32>
                                        : &k_lat_d64<64>;
        fn<<<1, 32>>>(d_out, d_dseeds, d_sink, args.reps);
      } else if (c.mode == "clock") {
        void (*fn)(std::uint64_t*, std::uint64_t*, int) =
            c.u == 16 ? &k_clock<16>
                      : c.u == 32 ? &k_clock<32> : &k_clock<64>;
        fn<<<1, 32>>>(d_out, d_sink64, args.reps);
      } else if (c.mode == "gt") {
        void (*fn)(std::uint64_t*, std::uint64_t*, int) =
            c.u == 16 ? &k_gt<16> : c.u == 32 ? &k_gt<32> : &k_gt<64>;
        fn<<<1, 32>>>(d_out, d_sink64, args.reps);
      } else {  // mlat / mthr
        int d = c.which, u = c.u;
        if (c.mode == "mlat") {
          void (*fn)(std::uint64_t*, std::uint32_t*, int,
                     std::uint32_t) = nullptr;
          switch ((d << 8) | u) {
#define ML(D, U) case (D << 8) | U: fn = &k_mlat<D, U>; break;
            ML(1,16) ML(1,32) ML(1,64) ML(2,16) ML(2,32) ML(2,64)
            ML(4,16) ML(4,32) ML(4,64) ML(8,16) ML(8,32) ML(8,64)
            ML(16,16) ML(16,32) ML(16,64) ML(32,16) ML(32,32) ML(32,64)
#undef ML
          }
          fn<<<1, 32>>>(d_out, d_sink, args.reps, seed);
        } else {
          void (*fn)(std::uint64_t*, std::uint32_t*, int,
                     std::uint32_t) = nullptr;
          switch ((d << 8) | u) {
#define MT(D, U) case (D << 8) | U: fn = &k_mthr<D, U>; break;
            MT(1,16) MT(1,32) MT(1,64) MT(2,16) MT(2,32) MT(2,64)
            MT(4,16) MT(4,32) MT(4,64) MT(8,16) MT(8,32) MT(8,64)
            MT(16,16) MT(16,32) MT(16,64) MT(32,16) MT(32,32) MT(32,64)
#undef MT
          }
          fn<<<1, 32>>>(d_out, d_sink, args.reps, seed);
        }
      }
      const cudaError_t e1 = cudaGetLastError();
      const cudaError_t e2 = cudaDeviceSynchronize();
      if (e1 != cudaSuccess || e2 != cudaSuccess) {
        cudaError_t e = e1 != cudaSuccess ? e1 : e2;
        std::fprintf(stdout, "FAIL %s launch %d: %s\n",
                     c.name.c_str(), l, cudaGetErrorName(e));
        w.row(c.name, c.u, "\"op\":\"" + c.name +
              "\",\"status\":\"unrealised:" + cudaGetErrorName(e) +
              "\"", {});
        break;
      }
      std::vector<std::uint64_t> h(args.reps);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      if (l == 0) continue;
      for (int r = 1; r < args.reps; ++r)
        s.push_back(static_cast<double>(h[r]));
    }
    if (s.size())
      w.row(c.name, c.u, "\"op\":\"" + c.name + "\"", s);
  }
  cudaFree(d_out); cudaFree(d_sink); cudaFree(d_seeds);
  cudaFree(d_dseeds); cudaFree(d_sink64);
  return 0;
}
