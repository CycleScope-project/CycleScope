// instruction.service result latency, batch 2: select / compare / 3-input
// min-max, constant-bank loads, and uniform-datapath forms.
//
//   vlat OP U  one warp, U dependent instances (alternating x/y form as in
//              alu.cu) on per-lane values -> vector datapath
//   ulat OP U  same chain on kernel-argument values whose only consumer is a
//              uniform constant-bank address -> ptxas keeps the chain on the
//              uniform datapath (UIADD3, UIMAD, ULOP3, USHF, UPRMT, USEL,
//              UISETP, UPOPC, UFLO, LDCU)
//   uthr OP K  uniform-datapath throughput: every warp runs 8 independent
//              uniform chains (kernel-argument seeds, carried across reps,
//              uniform constant-bank consumer after the loop), K steps per
//              window; 1 or 4 warps per SMSP; slope over K / 8 = cycles per
//              uniform instruction per warp (x wps = per SMSP if shared)
// Pair forms (ISETP+SEL, FSETP+FSEL, UISETP+USEL) are timed as pairs; the
// select alone is timed too, so the compare = pair - select.
// LDC / LDCU: pointer chase through a __constant__ table of constant-bank
// addresses (constant-cache hits after the first lap).
#include "../mb.cuh"

#include <string>
#include <utility>
#include <vector>

constexpr int kTab = 1024;
__constant__ std::uint32_t ctab[kTab];

#define OP2(NAME, ROOT, ASM)                                                                   \
  struct NAME {                                                                               \
    static constexpr const char* name = #NAME;                                                \
    static constexpr const char* root = ROOT;                                                 \
    __device__ __forceinline__ static void op(std::uint32_t& d, std::uint32_t a, std::uint32_t b, \
                                              std::uint32_t k) {                              \
      asm volatile(ASM : "=r"(d) : "r"(a), "r"(b), "r"(k));                                  \
    }                                                                                         \
  };
#define OPF(NAME, ROOT, ASM)                                                                   \
  struct NAME {                                                                               \
    static constexpr const char* name = #NAME;                                                \
    static constexpr const char* root = ROOT;                                                 \
    __device__ __forceinline__ static void op(std::uint32_t& d, std::uint32_t a, std::uint32_t b, \
                                              std::uint32_t k) {                              \
      float r;                                                                                \
      asm volatile(ASM : "=f"(r) : "f"(__uint_as_float(a)), "f"(__uint_as_float(b)), "f"(__uint_as_float(k))); \
      d = __float_as_uint(r);                                                                 \
    }                                                                                         \
  };

// k: loop-invariant third operand (a predicate source for the selects)
OP2(SEL, "SEL", "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\tselp.b32 %0, %1, %2, p;\n\t}")
OP2(ISETP_SEL, "ISETP", "{\n\t.reg .pred p;\n\tsetp.lt.u32 p, %1, %3;\n\tselp.b32 %0, %1, %2, p;\n\t}")
OPF(FSEL, "FSEL", "{\n\t.reg .pred p;\n\tsetp.ne.f32 p, %3, 0f00000000;\n\tselp.f32 %0, %1, %2, p;\n\t}")
OPF(FSETP_FSEL, "FSETP", "{\n\t.reg .pred p;\n\tsetp.lt.f32 p, %1, %3;\n\tselp.f32 %0, %1, %2, p;\n\t}")
OPF(FMNMX3, "FMNMX3", "max.f32 %0, %1, %2, %3;")
OP2(VIMNMX3, "VIMNMX3", "{\n\t.reg .s32 t;\n\tmin.s32 t, %1, %2;\n\tmin.s32 %0, t, %3;\n\t}")
OP2(IADD3, "IADD3", "{\n\t.reg .u32 t;\n\tadd.u32 t, %1, %2;\n\tadd.u32 %0, t, %1;\n\t}")
OP2(IMAD, "IMAD", "mad.lo.u32 %0, %1, %2, %1;")
OP2(LOP3, "LOP3", "lop3.b32 %0, %1, %2, %3, 0x96;")
OP2(SHF, "SHF", "shf.l.wrap.b32 %0, %1, %2, 7;")
OP2(PRMT, "PRMT", "prmt.b32 %0, %1, %2, 0x6152;")
OP2(POPC, "POPC", "{\n\t.reg .u32 t;\n\tpopc.b32 t, %1;\n\tmov.u32 %0, t;\n\t}")
OP2(FLO, "FLO", "bfind.u32 %0, %1;")
OP2(LDC, "LDC", "ld.const.u32 %0, [%1];")

#define VEC_OPS(X) X(SEL) X(ISETP_SEL) X(FSEL) X(FSETP_FSEL) X(FMNMX3) X(VIMNMX3) X(LDC)
#define UNI_OPS(X) X(SEL) X(ISETP_SEL) X(IADD3) X(IMAD) X(LOP3) X(SHF) X(PRMT) X(POPC) X(FLO) X(LDC)
#define UTHR_OPS(X) X(SEL) X(ISETP_SEL) X(IADD3) X(IMAD) X(LOP3) X(SHF) X(PRMT) X(POPC) X(FLO)


// vector: per-lane seeds read from global memory (not provably uniform)
template <class O, int U>
__global__ void __launch_bounds__(32, 1) k_vlat(std::uint64_t* out, const std::uint32_t* seeds, std::uint32_t* sink,
                                                int reps) {
  std::uint32_t x = seeds[threadIdx.x], y = seeds[32 + threadIdx.x], k = seeds[64 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if (i & 1) O::op(y, x, y, k);
      else O::op(x, y, x, k);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = x ^ y;
}

// uniform: kernel-argument seeds; the only consumer is a uniform ld.const
template <class O, int U>
__global__ void __launch_bounds__(32, 1) k_ulat(std::uint64_t* out, std::uint32_t sx, std::uint32_t sy, std::uint32_t sk,
                                                int reps) {
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
      out[reps + 1 + rep] = ctab[(x ^ y) & (kTab - 1)];  // uniform constant-bank consumer
    }
  }
}

// uniform throughput: 8 chains per warp, all uniform (kernel arguments)
template <class O, int K>
__global__ void __launch_bounds__(512, 1) k_uthr(std::uint64_t* out, std::uint32_t sx, std::uint32_t sy, std::uint32_t sk,
                                                 int reps) {
  std::uint32_t x[8], y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = sx + 0x01010101u * j;
    y[j] = sy ^ (0x10203040u * j);
  }
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (i & 1) O::op(y[j], x[j], y[j], sk);
        else O::op(x[j], y[j], x[j], sk);
      }
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t acc = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) acc ^= x[j] ^ y[j];
  if (threadIdx.x == 0) out[reps + 1] = ctab[acc & (kTab - 1)];  // uniform constant-bank consumer
}

__global__ void k_ctab_base(std::uint32_t* o) {
  std::uint32_t b;
  asm volatile("mov.u32 %0, ctab;" : "=r"(b));
  *o = b;
}

using VKern = void (*)(std::uint64_t*, const std::uint32_t*, std::uint32_t*, int);
using UKern = void (*)(std::uint64_t*, std::uint32_t, std::uint32_t, std::uint32_t, int);
struct VCase {
  std::string op;
  int u;
  VKern k;
};
struct UCase {
  std::string op;
  int u;
  UKern k;
};

template <class O, int... Us>
void addv(std::vector<VCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({O::name, Us, &k_vlat<O, Us>}), ...);
}
template <class O, int... Us>
void addu(std::vector<UCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({O::name, Us, &k_ulat<O, Us>}), ...);
}
template <class O, int... Ks>
void addut(std::vector<UCase>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({O::name, Ks, &k_uthr<O, Ks>}), ...);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "alu2");
  std::uint32_t* d_b = nullptr;
  MB_CHECK(cudaMalloc(&d_b, 4));
  k_ctab_base<<<1, 1>>>(d_b);
  MB_CHECK(cudaDeviceSynchronize());
  std::uint32_t base = 0;
  MB_CHECK(cudaMemcpy(&base, d_b, 4, cudaMemcpyDeviceToHost));
  std::vector<std::uint32_t> tab(kTab);
  for (int i = 0; i < kTab; ++i) tab[i] = base + 4u * static_cast<std::uint32_t>((i * 37 + 1) % kTab);
  MB_CHECK(cudaMemcpyToSymbol(ctab, tab.data(), sizeof(std::uint32_t) * kTab));
  // seeds: x, y (for LDC: constant-bank addresses), k (select predicate / third operand)
  std::vector<std::uint32_t> seeds(96);
  // every lane gets the same constant-bank address (a warp-uniform value in a
  // vector register, as in LDC parameter loads), so LDC is not serialised over
  // 32 distinct addresses; the ALU ops see per-lane-distinct x/y via k
  for (int l = 0; l < 32; ++l) {
    seeds[l] = base;
    seeds[32 + l] = base + 4u * 64u;
    seeds[64 + l] = 1u + static_cast<std::uint32_t>(l);
  }
  std::uint32_t* d_seeds = nullptr;
  std::uint32_t* d_sink = nullptr;
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_seeds, 4 * 96));
  MB_CHECK(cudaMalloc(&d_sink, 4 * 32));
  MB_CHECK(cudaMalloc(&d_out, 8 * (2 * args.reps + 2)));
  MB_CHECK(cudaMemcpy(d_seeds, seeds.data(), 4 * 96, cudaMemcpyHostToDevice));
  std::vector<VCase> vc;
  std::vector<UCase> uc;
#define AV(O) addv<O>(vc, std::integer_sequence<int, 8, 16, 32, 64>{});
#define AU(O) addu<O>(uc, std::integer_sequence<int, 8, 16, 32, 64>{});
  VEC_OPS(AV)
  UNI_OPS(AU)
  std::vector<UCase> ut;
#define AUT(O) addut<O>(ut, std::integer_sequence<int, 16, 32, 64>{});
  UTHR_OPS(AUT)
  for (const auto& c : vc) {
    const std::string variant = "vlat." + c.op;
    if (!mb_selected(args, variant)) continue;
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      c.k<<<1, 32>>>(d_out, d_seeds, d_sink, args.reps);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      mb_collect(d_out, args.reps, 1, 0, s);
    }
    w.row(variant, c.u, "\"op\":\"" + c.op + "\",\"mode\":\"vlat\"", s);
  }
  for (const auto& c : uc) {
    const std::string variant = "ulat." + c.op;
    if (!mb_selected(args, variant)) continue;
    const bool ldc = c.op == "LDC";
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      c.k<<<1, 32>>>(d_out, ldc ? base + 4u * static_cast<std::uint32_t>(l) : 3u + l,
                     ldc ? base + 4u * static_cast<std::uint32_t>(512 + l) : 0x9e3779b9u, 1u + l, args.reps);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      mb_collect(d_out, args.reps, 1, 0, s);
    }
    w.row(variant, c.u, "\"op\":\"" + c.op + "\",\"mode\":\"ulat\"", s);
  }
  for (const auto& c : ut) {
    for (int wps : {1, 4}) {
      const std::string variant = "uthr." + c.op + ".wps" + std::to_string(wps);
      if (!mb_selected(args, variant)) continue;
      std::vector<double> s;
      for (int l = 0; l < args.launches; ++l) {
        c.k<<<1, 128 * wps>>>(d_out, 3u + l, 0x9e3779b9u, 1u + l, args.reps);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        mb_collect(d_out, args.reps, 1, 0, s);
      }
      w.row(variant, c.u, "\"op\":\"" + c.op + "\",\"mode\":\"uthr\",\"wps\":" + std::to_string(wps), s);
    }
  }
  return w.failures ? 1 : 0;
}
