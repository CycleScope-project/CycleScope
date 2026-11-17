// alu7: service_latency protocol v6 cells — b300 reproduction of the
// high-volume blackwell-prototype classes (repro_b300), fp64 dthr v5 rerun
// (fp64_b300_v6), and MATCH d32 / d16 thr rerun (MATCH_d32).
#include "../mb.cuh"
#include <string>
#include <vector>

constexpr int kTab = 1024;
__constant__ std::uint32_t ctab[kTab];

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

// f32x2 ops on 64-bit register pairs
#define OPX2(NAME, ROOT, ASM)                                             \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    static constexpr const char* root = ROOT;                             \
    __device__ __forceinline__ static void op(                            \
        unsigned long long& d, unsigned long long a,                    \
        unsigned long long b, unsigned long long k) {                     \
      asm volatile(ASM : "=l"(d) : "l"(a), "l"(b), "l"(k));               \
    }                                                                     \
  };

#ifdef SVC_SM100
OPX2(FFMA2, "FFMA2", "fma.rn.f32x2 %0, %1, %2, %3;")
OPX2(FADD2, "FADD2", "add.rn.f32x2 %0, %1, %2;")
OPX2(FMUL2, "FMUL2", "mul.rn.f32x2 %0, %1, %2;")
#else
#define OPX2_STUB(NAME, ROOT)                                             \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    static constexpr const char* root = ROOT;                             \
    __device__ __forceinline__ static void op(                            \
        unsigned long long& d, unsigned long long a,                      \
        unsigned long long b, unsigned long long k) {                     \
      d = a;                                                              \
    }                                                                     \
  };
OPX2_STUB(FFMA2, "FFMA2")
OPX2_STUB(FADD2, "FADD2")
OPX2_STUB(FMUL2, "FMUL2")
#define OP2_STUB(NAME, ROOT)                                              \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    static constexpr const char* root = ROOT;                             \
    __device__ __forceinline__ static void op(std::uint32_t& d,             \
                                            std::uint32_t a,              \
                                            std::uint32_t b,              \
                                            std::uint32_t k) {            \
      d = a;                                                              \
    }                                                                     \
  };
OP2_STUB(FMNMX3, "FMNMX3")
#endif

struct MUFU_EX2 {   // unary f32
  static constexpr const char* name = "MUFU_EX2";
  static constexpr const char* root = "MUFU";
  __device__ __forceinline__ static void op(std::uint32_t& d,
                                          std::uint32_t a,
                                          std::uint32_t b,
                                          std::uint32_t k) {
    asm volatile("{\n\t.reg .f32 t;\n\tmov.b32 t, %1;\n\t"
                 "ex2.approx.ftz.f32 t, t;\n\tmov.b32 %0, t;\n\t}"
                 : "=r"(d) : "r"(a));
  }
};
#ifndef SVC_SM100_SKIP_FMNMX3
#ifdef SVC_SM100
struct FMNMX3 {     // three-input max.f32
  static constexpr const char* name = "FMNMX3";
  static constexpr const char* root = "FMNMX3";
  __device__ __forceinline__ static void op(std::uint32_t& d,
                                          std::uint32_t a,
                                          std::uint32_t b,
                                          std::uint32_t k) {
    asm volatile("{\n\t.reg .f32 t1, t2, t3;\n\t"
                 "mov.b32 t1, %1;\n\tmov.b32 t2, %2;\n\tmov.b32 t3, %3;\n\t"
                 "max.f32 t1, t1, t2, t3;\n\tmov.b32 %0, t1;\n\t}"
                 : "=r"(d) : "r"(a), "r"(b), "r"(k));
  }
};
#endif
#endif
struct HADD2_F32 {  // add.rn.f32.f16 mixed precision: ptxas has
                          // no single-instruction lowering (alu3 note);
                          // compiled as a stub -> unrealised:no_ptx
  static constexpr const char* name = "HADD2_F32";
  static constexpr const char* root = "HADD2";
  __device__ __forceinline__ static void op(std::uint32_t& d,
                                          std::uint32_t a,
                                          std::uint32_t b,
                                          std::uint32_t k) {
    asm volatile("add.rn.f32 %0, %1, %1;" : "=r"(d) : "r"(a));
  }
};
struct LEA_SHLADD { // v3 survey LEA_shladd: shl.b32 + add.s32
  static constexpr const char* name = "LEA_SHLADD";
  static constexpr const char* root = "LEA";
  __device__ __forceinline__ static void op(std::uint32_t& d,
                                          std::uint32_t a,
                                          std::uint32_t b,
                                          std::uint32_t k) {
    asm volatile("{\n\t.reg .u32 t;\n\tshl.b32 t, %1, 3;\n\t"
                 "add.s32 %0, t, %2;\n\t}"
                 : "=r"(d) : "r"(a), "r"(b));
  }
};

// ---- fp64 dthr rerun (v5 bodies) ----------------------------------------
struct DADD_RN {
  static constexpr const char* name = "DADD_RN";
  static constexpr const char* root = "DADD";
  __device__ __forceinline__ static void dop(double& d, double a,
                                           double b) {
    asm volatile("add.rn.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};
struct DMUL_RN {
  static constexpr const char* name = "DMUL_RN";
  static constexpr const char* root = "DMUL";
  __device__ __forceinline__ static void dop(double& d, double a,
                                           double b) {
    asm volatile("mul.rn.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};
struct DADD_RZ {
  static constexpr const char* name = "DADD_RZ";
  static constexpr const char* root = "DADD";
  __device__ __forceinline__ static void dop(double& d, double a,
                                           double b) {
    asm volatile("add.rz.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};
struct DMUL_RZ {
  static constexpr const char* name = "DMUL_RZ";
  static constexpr const char* root = "DMUL";
  __device__ __forceinline__ static void dop(double& d, double a,
                                           double b) {
    asm volatile("mul.rz.f64 %0, %1, %2;" : "=d"(d) : "d"(a), "d"(b));
  }
};

// ---- harnesses (alu5 / alu2 shapes) -------------------------------------
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
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    __syncthreads();
  }
  std::uint32_t f = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) f ^= x[j] ^ y[j];
  sink[threadIdx.x] = f;
}

#ifdef SVC_SM100
// f32x2 pair harness: one dependent f32x2 op per U / 8 chains per K
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_latx2(std::uint64_t* out, const std::uint32_t* seeds,
        std::uint32_t* sink, int reps) {
  unsigned long long x = seeds[threadIdx.x], y = seeds[32 + threadIdx.x],
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
  sink[threadIdx.x] = (std::uint32_t)(x ^ y);
}

template <class O, int K>
__global__ void __launch_bounds__(512, 1)
k_thrx2(std::uint64_t* out, std::uint32_t* sink, int reps,
        std::uint32_t s) {
  __shared__ __align__(16) std::uint32_t pinb[512 * 16];
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 4u;
  constexpr std::uint32_t kStride = 512u * 4u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + j * kStride), "r"(s + threadIdx.x + j));
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + (8 + j) * kStride), "r"(s * 3u + j));
  }
  unsigned long long x[8], y[8];
  const unsigned long long k = s ^ 0x3f800000u;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(*reinterpret_cast<std::uint32_t*>(&x[j])) : "r"(base + j * kStride));
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(*reinterpret_cast<std::uint32_t*>(&y[j])) : "r"(base + (8 + j) * kStride));
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (i & 1) O::op(y[j], x[j], y[j], k);
        else O::op(x[j], y[j], x[j], k);
      }
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    __syncthreads();
  }
  std::uint32_t f = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) f ^= (std::uint32_t)(x[j] ^ y[j]);
  sink[threadIdx.x] = f;
}

#endif
// fp64 8-chain thr (v5 k_dthr verbatim)
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

// uniform harness (alu2 k_ulat verbatim) — ULEA
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_ulat(std::uint64_t* out, std::uint32_t sx, std::uint32_t sy,
       std::uint32_t sk, int reps) {
  std::uint32_t x = sx, y = sy;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if (i & 1) O::op(y, x, y, sk);
      else O::op(x, y, x, sk);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) {
      out[rep] = t1 - t0;
      out[reps + 1 + rep] = ctab[(x ^ y) & (kTab - 1)];
    }
  }
}

// MATCH thr (v5 k_mthr verbatim)
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

__global__ void k_ctab_base(std::uint32_t* o) {
  std::uint32_t b;
  asm volatile("mov.u32 %0, ctab;" : "=r"(b));
  *o = b;
}

struct Case { std::string name, mode; int u, which; };

template <class O>
static void add_lat_thr(std::vector<Case>& cs, const char* n,
                        int x2, int uniform) {
  for (int u : {16, 32, 64}) {
    std::string m = x2 ? "latx2" : uniform ? "ulat" : "lat";
    cs.push_back({std::string("lat.") + n, m, u, 0});
    m = x2 ? "thrx2" : "thr";
    cs.push_back({std::string("thr.") + n, m, u, 0});
  }
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "alu7");
  std::vector<Case> cases;
  // repro_b300 (b300 only; run.sh selects by device)
#ifdef SVC_SM100
  add_lat_thr<FFMA2>(cases, "FFMA2", 1, 0);
  add_lat_thr<FADD2>(cases, "FADD2", 1, 0);
  add_lat_thr<FMUL2>(cases, "FMUL2", 1, 0);
  add_lat_thr<MUFU_EX2>(cases, "MUFU_EX2", 0, 0);
  add_lat_thr<FMNMX3>(cases, "FMNMX3", 0, 0);
  add_lat_thr<HADD2_F32>(cases, "HADD2_F32", 0, 0);
#endif
  add_lat_thr<LEA_SHLADD>(cases, "LEA_SHLADD", 0, 0);
  for (int u : {16, 32, 64})
    cases.push_back({"lat.ULEA_SHLADD", "ulat", u, 0});
  // fp64_b300_v6: dthr only
  for (int u : {16, 32, 64}) {
    cases.push_back({"dthr.DADD_RN", "dthr", u, 0});
    cases.push_back({"dthr.DMUL_RN", "dthr", u, 0});
    cases.push_back({"dthr.DADD_RZ", "dthr", u, 0});
    cases.push_back({"dthr.DMUL_RZ", "dthr", u, 0});
  }
  // MATCH_d32 (both devices) + d16 (b300): thr over full K set
  for (int d : {16, 32})
    for (int k : {16, 24, 32, 48, 64, 96, 128})
      cases.push_back({"thr.MATCH_d" + std::to_string(d), "mthr",
                       k, d});

  std::vector<std::uint32_t> h_seeds(128);
  for (std::size_t i = 0; i < h_seeds.size(); ++i)
    h_seeds[i] = 0x9e3779b9u * (std::uint32_t)(i + 1) + 0x3f800000u;
  std::uint64_t *d_out, *d_sink64;
  std::uint32_t *d_sink, *d_seeds, *d_b;
  MB_CHECK(cudaMalloc(&d_out, 2 * args.reps * 8));
  MB_CHECK(cudaMalloc(&d_sink, 512 * 4));
  MB_CHECK(cudaMalloc(&d_sink64, 512 * 8));
  MB_CHECK(cudaMalloc(&d_seeds, h_seeds.size() * 4));
  MB_CHECK(cudaMalloc(&d_b, 4));
  MB_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), h_seeds.size() * 4,
                      cudaMemcpyHostToDevice));
  k_ctab_base<<<1, 1>>>(d_b);
  std::uint32_t hb;
  cudaMemcpy(&hb, d_b, 4, cudaMemcpyDeviceToHost);
  std::vector<std::uint32_t> tab(kTab);
  for (int i = 0; i < kTab; ++i) tab[i] = 0x5bd1e995u * i;
  cudaMemcpyToSymbol(ctab, tab.data(), sizeof(tab));

  for (const auto& c : cases) {
    if (args.only.size() && c.name.find(args.only) == std::string::npos)
      continue;
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      const std::uint32_t seed = 0x9e3779b9u + l * 97;
      const int k = c.u;
      if (c.mode == "lat") {
        void (*fn)(std::uint64_t*, const std::uint32_t*,
                   std::uint32_t*, int) = nullptr;
        auto pick = [&](auto tag) {
          return k == 16 ? &k_lat<decltype(tag), 16>
               : k == 32 ? &k_lat<decltype(tag), 32>
                         : &k_lat<decltype(tag), 64>; };
        if (c.name.find("MUFU_EX2") != std::string::npos)
          fn = pick(MUFU_EX2{});
        else if (c.name.find("FMNMX3") != std::string::npos)
          fn = pick(FMNMX3{});
        else if (c.name.find("HADD2_F32") != std::string::npos)
          fn = pick(HADD2_F32{});
        else
          fn = pick(LEA_SHLADD{});
        fn<<<1, 32>>>(d_out, d_seeds, d_sink, args.reps);
      } else if (c.mode == "thr") {
        void (*fn)(std::uint64_t*, std::uint32_t*, int,
                   std::uint32_t) = nullptr;
        auto pick = [&](auto tag) {
          return k == 16 ? &k_thr<decltype(tag), 16>
               : k == 32 ? &k_thr<decltype(tag), 32>
                         : &k_thr<decltype(tag), 64>; };
        if (c.name.find("MUFU_EX2") != std::string::npos)
          fn = pick(MUFU_EX2{});
        else if (c.name.find("FMNMX3") != std::string::npos)
          fn = pick(FMNMX3{});
        else if (c.name.find("HADD2_F32") != std::string::npos)
          fn = pick(HADD2_F32{});
        else
          fn = pick(LEA_SHLADD{});
        fn<<<1, 512>>>(d_out, d_sink, args.reps, seed);
      } else if (c.mode == "latx2" || c.mode == "thrx2") {
#ifdef SVC_SM100
        auto pick = [&](auto tag) {
          if (c.mode == "latx2")
            return (void*)(k == 16 ? (void*)&k_latx2<decltype(tag), 16>
                          : k == 32 ? (void*)&k_latx2<decltype(tag), 32>
                                    : (void*)&k_latx2<decltype(tag), 64>);
          return (void*)(k == 16 ? (void*)&k_thrx2<decltype(tag), 16>
                        : k == 32 ? (void*)&k_thrx2<decltype(tag), 32>
                                  : (void*)&k_thrx2<decltype(tag), 64>);
        };
        void* fn;
        if (c.name.find("FFMA2") != std::string::npos) fn = pick(FFMA2{});
        else if (c.name.find("FADD2") != std::string::npos) fn = pick(FADD2{});
        else fn = pick(FMUL2{});
        if (c.mode == "latx2")
          ((void(*)(std::uint64_t*, const std::uint32_t*,
                    std::uint32_t*, int))fn)<<<1, 32>>>(d_out, d_seeds,
                                                        d_sink, args.reps);
        else
          ((void(*)(std::uint64_t*, std::uint32_t*, int,
                    std::uint32_t))fn)<<<1, 512>>>(d_out, d_sink,
                                                  args.reps, seed);
#endif
      } else if (c.mode == "ulat") {
        void (*fn)(std::uint64_t*, std::uint32_t, std::uint32_t,
                   std::uint32_t, int) =
            k == 16 ? &k_ulat<LEA_SHLADD, 16>
            : k == 32 ? &k_ulat<LEA_SHLADD, 32>
                      : &k_ulat<LEA_SHLADD, 64>;
        fn<<<1, 32>>>(d_out, seed, seed * 3, seed * 7, args.reps);
      } else if (c.mode == "dthr") {
        void (*fn)(std::uint64_t*, std::uint32_t*, int,
                   std::uint32_t) = nullptr;
        auto pick = [&](auto tag) {
          return k == 16 ? &k_dthr<decltype(tag), 16>
               : k == 32 ? &k_dthr<decltype(tag), 32>
                         : &k_dthr<decltype(tag), 64>; };
        if (c.name.find("DADD_RN") != std::string::npos) fn = pick(DADD_RN{});
        else if (c.name.find("DMUL_RN") != std::string::npos) fn = pick(DMUL_RN{});
        else if (c.name.find("DADD_RZ") != std::string::npos) fn = pick(DADD_RZ{});
        else fn = pick(DMUL_RZ{});
        fn<<<1, 512>>>(d_out, d_sink, args.reps, seed);
      } else {  // mthr
        void (*fn)(std::uint64_t*, std::uint32_t*, int,
                   std::uint32_t) = nullptr;
        int d = c.which;
        switch ((d << 8) | k) {
#define MT(D, K) case (D << 8) | K: fn = &k_mthr<D, K>; break;
          MT(16,16) MT(16,24) MT(16,32) MT(16,48) MT(16,64) MT(16,96)
          MT(16,128) MT(32,16) MT(32,24) MT(32,32) MT(32,48) MT(32,64)
          MT(32,96) MT(32,128)
#undef MT
        }
        fn<<<1, 32>>>(d_out, d_sink, args.reps, seed);
      }
      const cudaError_t e1 = cudaGetLastError();
      const cudaError_t e2 = cudaDeviceSynchronize();
      if (e1 != cudaSuccess || e2 != cudaSuccess) {
        cudaError_t e = e1 != cudaSuccess ? e1 : e2;
        std::fprintf(stdout, "FAIL %s launch %d: %s\n",
                     c.name.c_str(), l, cudaGetErrorName(e));
        w.row(c.name, c.u, "\"status\":\"unrealised:" +
              std::string(cudaGetErrorName(e)) + "\"", {});
        cudaGetLastError();
        break;
      }
      std::vector<std::uint64_t> h(2 * args.reps);
      cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost);
      if (l == 0) continue;
      for (int r = 1; r < args.reps; ++r) s.push_back((double)h[r]);
    }
    if (s.size()) w.row(c.name, c.u, "", s);
  }
  return 0;
}
