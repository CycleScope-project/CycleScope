// idioms.cu — compile-only SASS survey for service_latency v3 cell
// candidates.  No timing semantics: every kernel just holds a chain of
// the candidate idiom between two clock64 reads; survey_idioms.py
// decomposes the window per step.  Kernel names encode the harness and
// cell as _Z5k_<harness>I<idlen><id>Li8E...
#include "../mb.cuh"
#include <string>
#include <vector>

#define IDIOM(NAME, ASM)                                                  \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    __device__ __forceinline__ static void op(                            \
        std::uint32_t& d, std::uint32_t a, std::uint32_t b,               \
        std::uint32_t k) {                                                \
      asm volatile(ASM : "=r"(d) : "r"(a), "r"(b), "r"(k));               \
    }                                                                     \
  };
#define IDIOM_K(NAME, ASM)  /* op reads only a, k */                      \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    __device__ __forceinline__ static void op(                            \
        std::uint32_t& d, std::uint32_t a, std::uint32_t,                 \
        std::uint32_t k) {                                                \
      asm volatile(ASM : "=r"(d) : "r"(a), "r"(k));                       \
    }                                                                     \
  };
#define IDIOM_D(NAME, ASM)  /* f64 chain */                               \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    __device__ __forceinline__ static void op(                            \
        double& d, double a, double b, std::uint32_t) {                   \
      asm volatile(ASM : "=d"(d) : "d"(a), "d"(b));                       \
    }                                                                     \
  };
// predicate chain: setp inits + U pred ops + selp out inside ONE asm
// block (nvcc has no external "p" operand constraint)
#define X8(s) s s s s s s s s
#define IDIOM_P(NAME, SETP, OPASM)                                        \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    __device__ __forceinline__ static void chain(                         \
        std::uint32_t& d, std::uint32_t a, std::uint32_t b) {             \
      asm volatile("{\n\t.reg .pred p,q;\n\t" SETP OPASM              \
                   "selp.u32 %0, 1, 0, p;\n\t}"                         \
                   : "=r"(d) : "r"(a), "r"(b));                           \
    }                                                                     \
  };
#define IDIOM_I(NAME, ASM)  /* issue-only */                              \
  struct NAME {                                                           \
    static constexpr const char* name = #NAME;                            \
    __device__ __forceinline__ static std::uint32_t iop(std::uint32_t s) {\
      std::uint32_t d;                                                    \
      asm volatile(ASM : "=r"(d) : "r"(s));                               \
      return d;                                                           \
    }                                                                     \
  };

// ---- integer / bit cells ----
IDIOM(IABS_abs, "abs.s32 %0, %1;")
IDIOM(IADD_add64, "{\n\t.reg .b64 t, kk;\n\tcvt.u64.u32 t, %1;\n\tcvt.u64.u32 kk, %3;\n\tadd.s64 t, t, kk;\n\tcvt.u32.u64 %0, t;\n\t}")
IDIOM(IADD_addc, "{\n\t.reg .u32 lo;\n\tadd.cc.s32 lo, %1, %2;\n\taddc.s32 %0, lo, 0;\n\t}")
IDIOM(IADD_sub, "sub.s32 %0, %1, %2;")
IDIOM(VIADD_bigimm, "add.s32 %0, %1, 0x12345;")
IDIOM(VIADD_smallimm, "add.s32 %0, %1, 7;")
IDIOM(VIADDMNMX_minadd, "{\n\t.reg .s32 t;\n\tadd.s32 t, %1, %3;\n\tmin.s32 %0, t, %2;\n\t}")
IDIOM(VIADDMNMX_maxadd, "{\n\t.reg .s32 t;\n\tadd.s32 t, %1, %3;\n\tmax.s32 %0, t, %2;\n\t}")
IDIOM(VIMNMX3_min2_s32, "{\n\t.reg .s32 t;\n\tmin.s32 t, %1, %2;\n\tmin.s32 %0, t, %3;\n\t}")
IDIOM(VIMNMX3_min2_u32, "{\n\t.reg .u32 t;\n\tmin.u32 t, %1, %2;\n\tmin.u32 %0, t, %3;\n\t}")
IDIOM(LEA_shladd, "{\n\t.reg .u32 t;\n\tshl.b32 t, %1, 3;\n\tadd.s32 %0, t, %2;\n\t}")
IDIOM(LEA_madwide, "{\n\t.reg .b64 t, c;\n\tcvt.u64.u32 c, %2;\n\tmad.wide.u32 t, %1, 8, c;\n\tcvt.u32.u64 %0, t;\n\t}")
// ---- fp16 / f64 ----
IDIOM(HFMA2_fma, "fma.rn.f16x2 %0, %1, %2, %3;")
IDIOM_D(DADD_add, "add.f64 %0, %1, %2;")
IDIOM_D(DMUL_mul, "mul.f64 %0, %1, %2;")
// protocol-v5 DADD_DMUL_RN survey, in the protocol's order
IDIOM_D(RN1_addrn, "add.rn.f64 %0, %1, %2;")
IDIOM_D(RN2_subrn, "sub.rn.f64 %0, %1, %2;")
IDIOM_D(RN3_mulrn, "mul.rn.f64 %0, %1, %2;")
IDIOM_D(RN4_mul, "mul.f64 %0, %1, %2;")
IDIOM_D(RN5_addrn_neg, "{\n\t.reg .f64 n;\n\tneg.f64 n, %2;\n\tadd.rn.f64 %0, %1, n;\n\t}")
// ---- select / compare ----
IDIOM(SEL_selp, "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\tselp.b32 %0, %2, %1, p;\n\t}")
IDIOM(FSEL_selp, "{\n\t.reg .pred p;\n\t.reg .f32 fa, fb, fo;\n\tsetp.ne.u32 p, %3, 0;\n\tcvt.rn.f32.u32 fa, %1;\n\tcvt.rn.f32.u32 fb, %2;\n\tselp.f32 fo, fb, fa, p;\n\tcvt.rzi.u32.f32 %0, fo;\n\t}")
IDIOM(ISETP_pair, "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, %1, %3;\n\tselp.b32 %0, %2, %1, p;\n\t}")
IDIOM(FSETP_pair, "{\n\t.reg .pred p;\n\t.reg .f32 fa, fb;\n\tcvt.rn.f32.u32 fa, %1;\n\tcvt.rn.f32.u32 fb, %2;\n\tsetp.lt.f32 p, fa, fb;\n\tselp.b32 %0, %1, %2, p;\n\t}")
IDIOM_P(PLOP3_xor,
        "setp.ne.u32 p, %1, 0;\n\tsetp.ne.u32 q, %2, 0;\n\t",
        X8("xor.pred p, p, q;\n\t"))
IDIOM_P(PLOP3_and,
        "setp.ne.u32 p, %1, 0;\n\tsetp.ne.u32 q, %2, 0;\n\t",
        X8("and.pred p, p, q;\n\t"))
IDIOM_P(ISETP_pred,
        "setp.ne.u32 p, %1, 0;\n\tsetp.ne.u32 q, %2, 0;\n\t",
        X8("{\n\t.reg .b32 v;\n\tselp.u32 v, %1, %2, p;\n\t"
           "setp.ne.and.b32 p, v, 0, q;\n\t}\n\t"))
IDIOM(DSETP_pair, "{\n\t.reg .pred p;\n\t.reg .f64 fa, fb;\n\t.reg .f32 o;\n\tcvt.rn.f64.u32 fa, %1;\n\tcvt.rn.f64.u32 fb, %2;\n\tsetp.gt.f64 p, fa, fb;\n\tselp.f32 o, 1.0, 0.0, p;\n\tcvt.rzi.u32.f32 %0, o;\n\t}")
// ---- warp collectives ----
IDIOM_K(VOTE_any, "{\n\t.reg .pred p, r;\n\tsetp.ne.u32 p, %1, 0;\n\tvote.sync.any.pred r, p, 0xffffffff;\n\tselp.u32 %0, 1, 0, r;\n\t}")
IDIOM_K(VOTE_all, "{\n\t.reg .pred p, r;\n\tsetp.ne.u32 p, %1, 0;\n\tvote.sync.all.pred r, p, 0xffffffff;\n\tselp.u32 %0, 1, 0, r;\n\t}")
IDIOM_K(VOTE_ballot, "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\tvote.sync.ballot.b32 %0, p, 0xffffffff;\n\t}")
IDIOM(ELECT_mask, "{\n\t.reg .pred p;\n\telect.sync _|p, 0xffffffff;\n\tselp.u32 %0, 1, 0, p;\n\t}")
IDIOM(SHFL_imm_idx, "shfl.sync.idx.b32 %0, %1, 3, 0x1f, 0xffffffff;")
IDIOM(SHFL_imm_bfly, "shfl.sync.bfly.b32 %0, %1, 5, 0x1f, 0xffffffff;")
IDIOM(SHFL_imm_up, "shfl.sync.up.b32 %0, %1, 1, 0x0, 0xffffffff;")
IDIOM(SHFL_imm_down, "shfl.sync.down.b32 %0, %1, 2, 0x1f, 0xffffffff;")
IDIOM(SHFL_reg_idx, "shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;")
IDIOM(REDUX_add, "redux.sync.add.u32 %0, %1, 0xffffffff;")
IDIOM(REDUX_min, "redux.sync.min.u32 %0, %1, 0xffffffff;")
IDIOM(REDUX_max, "redux.sync.max.u32 %0, %1, 0xffffffff;")
IDIOM(REDUX_and, "redux.sync.and.b32 %0, %1, 0xffffffff;")
// CREDUX f32 cluster redux: ptxas rejects redux.f32 on sm_103 — no form.

// ---- uniform candidates (driven by kernel params in k_ulat) ----
IDIOM_K(USGXT_bfe, "bfe.u32 %0, %1, 8, 16;")
IDIOM_K(USGXT_cvt, "{\n\t.reg .b32 t;\n\tcvt.s32.s8 t, %1;\n\tmov.b32 %0, t;\n\t}")
IDIOM_K(UFLO_bfind, "bfind.u32 %0, %1;")
IDIOM_K(UPOPC_popc, "popc.b32 %0, %1;")
IDIOM_K(UPRMT_prmt, "prmt.b32 %0, %1, %1, 0x6152;")
IDIOM_K(USHF_shfl, "shf.l.wrap.b32 %0, %1, %1, 7;")
IDIOM_K(USEL_slct, "slct.u32.s32 %0, %1, %2, %1;")
IDIOM_K(UISETP_sel, "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\tselp.u32 %0, %2, %1, p;\n\t}")
IDIOM(UMOV_mov, "mov.b32 %0, %1;")
IDIOM(R2UR_bound, "{\n\t.reg .pred p;\n\t.reg .u32 t;\n\tand.b32 t, %1, 7;\n\tsetp.ne.u32 p, t, 0;\n\tselp.u32 %0, %1, t, p;\n\t}")
// predicate-heavy: 8 live preds folded
IDIOM(P2R_pack8, "{\n\t.reg .pred p0,p1;\n\t.reg .u32 r;\n\tsetp.ne.u32 p0, %1, 0;\n\tsetp.ne.u32 p1, %2, 0;\n\tselp.u32 r, 1, 0, p0;\n\tselp.u32 %0, r, 2, p1;\n\t}")
IDIOM(R2P_unpack, "{\n\t.reg .pred p;\n\t.reg .u32 t;\n\tsetp.ne.u32 p, %1, 0;\n\tselp.u32 t, %1, %2, p;\n\tselp.u32 %0, t, %2, p;\n\t}")
IDIOM(UP2UR_predparam, "{\n\t.reg .pred p;\n\t.reg .u32 t;\n\tsetp.ne.u32 p, %3, 0;\n\tselp.u32 t, %1, %2, p;\n\tselp.u32 %0, t, %1, p;\n\t}")
// ---- issue-only ----
IDIOM_I(S2R_warpid, "mov.u32 %0, %%warpid;")
IDIOM_I(S2R_smid, "mov.u32 %0, %%smid;")
IDIOM_I(S2R_laneid, "mov.u32 %0, %%laneid;")
IDIOM_I(S2R_tid, "mov.u32 %0, %%tid.x;")
IDIOM_I(S2UR_ctaid, "mov.u32 %0, %%ctaid.x;")
IDIOM_I(S2UR_nctaid, "mov.u32 %0, %%nctaid.x;")
IDIOM_I(S2UR_smid, "mov.u32 %0, %%smid;")
IDIOM_I(CS2R_clock, "{\n\t.reg .b64 t;\n\tmov.u64 t, %%clock64;\n\tcvt.u32.u64 %0, t;\n\t}")
IDIOM_I(CS2R_gtimer, "{\n\t.reg .b64 t;\n\tmov.u64 t, %%globaltimer;\n\tcvt.u32.u64 %0, t;\n\t}")
IDIOM_I(CS2R_zero, "{\n\t.reg .b64 t;\n\tmov.b64 t, 0;\n\tcvt.u32.u64 %0, t;\n\t}")
IDIOM(FCHK_div, "{\n\t.reg .f32 a, b;\n\tcvt.rn.f32.u32 a, %1;\n\tcvt.rn.f32.u32 b, %1;\n\tdiv.rn.f32 b, a, b;\n\tcvt.rzi.u32.f32 %0, b;\n\t}")

// ================= harnesses =================
// vector lat: per-lane seeds
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_lat(std::uint64_t* out, const std::uint32_t* seeds, std::uint32_t* sink, int reps) {
  std::uint32_t x = seeds[threadIdx.x], y = seeds[32 + threadIdx.x],
                k = seeds[64 + threadIdx.x];
  const std::uint64_t t0 = mb_clock();
#pragma unroll
  for (int i = 0; i < U; ++i) {
    if (i & 1) O::op(y, x, y, k); else O::op(x, y, x, k);
  }
  const std::uint64_t t1 = mb_clock();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = x ^ y;
}

// uniform lat: u/v from kernel params (warp-uniform, provable)
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_ulat(std::uint64_t* out, std::uint32_t* sink, std::uint32_t p0,
       std::uint32_t p1, std::uint32_t p2) {
  std::uint32_t u = p0, v = p1;   // kernel params: warp-uniform
  const std::uint64_t t0 = mb_clock();
#pragma unroll
  for (int i = 0; i < U; ++i) O::op(u, v, u, p2);
  const std::uint64_t t1 = mb_clock();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = u;
}

// predicate lat: p,q predicate regs; q derived from a kernel param BEFORE
// the window; a .pred result folded to a register AFTER the window
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_plat(std::uint64_t* out, std::uint32_t* sink, std::uint32_t p0) {
  std::uint32_t d = 0;
  const std::uint64_t t0 = mb_clock();
  O::chain(d, p0, p0 ^ 1u);   // U pred ops inside one asm block
  const std::uint64_t t1 = mb_clock();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = d;
}

// vector thr: 16 warps x 8 chains (alu3 shape)
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
  __syncthreads();
  const std::uint64_t t1 = mb_clock();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = x[0] ^ y[0];
}

// f64 thr: 8 chains of .f64
template <class O, int K>
__global__ void __launch_bounds__(512, 1)
k_dthr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  double x[8], y[8];
  const double a = 1.0001 + (s & 7) * 0.01, b = 0.9999 - (s & 3) * 0.001;
#pragma unroll
  for (int j = 0; j < 8; ++j) { x[j] = a + j; y[j] = b - j; }
  const std::uint64_t t0 = mb_clock();
#pragma unroll
  for (int i = 0; i < K; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (i & 1) O::op(y[j], x[j], y[j], s);
      else O::op(x[j], y[j], x[j], s);
    }
  const std::uint64_t t1 = mb_clock();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = (std::uint32_t)__double_as_longlong(x[0] + y[0]);
}

// f64 lat: one dependent chain, two ops per step (alu5 k_dlat shape)
template <class O, int U>
__global__ void __launch_bounds__(32, 1)
k_dlat(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  double x = 1.0001 + (s & 7) * 0.01, y = 0.9999 - (s & 3) * 0.001;
  const std::uint64_t t0 = mb_clock();
#pragma unroll
  for (int i = 0; i < U; ++i) {
    O::op(x, x, y, s);
    O::op(y, y, x, s);
  }
  const std::uint64_t t1 = mb_clock();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = (std::uint32_t)__double_as_longlong(x + y);
}

// issue-only: K independent instances, results folded AFTER the window
template <class O, int K>
__global__ void __launch_bounds__(32, 1)
k_issue(std::uint64_t* out, const std::uint32_t* seeds, std::uint32_t* sink, int reps) {
  std::uint32_t r = seeds[threadIdx.x];
  std::uint32_t acc[K];
  const std::uint64_t t0 = mb_clock();
#pragma unroll
  for (int i = 0; i < K; ++i) acc[i] = O::iop(r + i);
  const std::uint64_t t1 = mb_clock();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  std::uint32_t f = 0;
#pragma unroll
  for (int i = 0; i < K; ++i) f ^= acc[i];
  sink[threadIdx.x] = f;
}

// ---------------------------------------------------------------------
using Kern = void (*)(std::uint64_t*, const std::uint32_t*, std::uint32_t*, int);
struct Case { std::string name; void (*k)(void); };

// register one case per harness instantiation (all compile-time U/K)
#define C_LAT(O)  {#O, (void (*)())&k_lat<O, 8>}
#define C_ULAT(O) {#O, (void (*)())&k_ulat<O, 8>}
#define C_PLAT(O) {#O, (void (*)())&k_plat<O, 8>}
#define C_THR(O)  {#O, (void (*)())&k_thr<O, 2>}
#define C_DTHR(O) {#O, (void (*)())&k_dthr<O, 2>}
#define C_DLAT(O) {#O, (void (*)())&k_dlat<O, 8>}
#define C_ISSUE(O) {#O, (void (*)())&k_issue<O, 8>}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  (void)args;
  std::vector<Case> cases = {
    // vector lat
    C_LAT(IABS_abs), C_LAT(IADD_add64), C_LAT(IADD_addc), C_LAT(IADD_sub),
    C_LAT(VIADD_bigimm), C_LAT(VIADD_smallimm), C_LAT(VIADDMNMX_minadd),
    C_LAT(VIADDMNMX_maxadd), C_LAT(VIMNMX3_min2_s32), C_LAT(VIMNMX3_min2_u32),
    C_LAT(LEA_shladd), C_LAT(LEA_madwide), C_LAT(HFMA2_fma),
    C_LAT(SEL_selp), C_LAT(FSEL_selp), C_LAT(ISETP_pair), C_LAT(FSETP_pair),
    C_LAT(DSETP_pair), C_LAT(VOTE_any), C_LAT(VOTE_all), C_LAT(VOTE_ballot),
    C_LAT(ELECT_mask), C_LAT(SHFL_imm_idx), C_LAT(SHFL_imm_bfly),
    C_LAT(SHFL_imm_up), C_LAT(SHFL_imm_down), C_LAT(SHFL_reg_idx),
    C_LAT(REDUX_add), C_LAT(REDUX_min), C_LAT(REDUX_max), C_LAT(REDUX_and),
    C_LAT(UMOV_mov), C_LAT(R2UR_bound), C_LAT(P2R_pack8), C_LAT(R2P_unpack),
    C_LAT(UP2UR_predparam), C_LAT(FCHK_div),
    // uniform lat (kernel params)
    C_ULAT(USGXT_bfe), C_ULAT(USGXT_cvt), C_ULAT(UFLO_bfind),
    C_ULAT(UPOPC_popc), C_ULAT(UPRMT_prmt), C_ULAT(USHF_shfl),
    C_ULAT(USEL_slct), C_ULAT(UISETP_sel), C_ULAT(UMOV_mov),
    // predicate lat
    C_PLAT(PLOP3_xor), C_PLAT(PLOP3_and), C_PLAT(ISETP_pred),
    // vector / f64 thr
    C_THR(IADD_sub), C_THR(VIADD_smallimm), C_THR(SEL_selp),
    C_DTHR(DADD_add), C_DTHR(DMUL_mul),
    C_DTHR(RN1_addrn), C_DTHR(RN2_subrn), C_DTHR(RN3_mulrn), C_DTHR(RN4_mul),
    C_DTHR(RN5_addrn_neg),
    C_DLAT(RN1_addrn), C_DLAT(RN2_subrn), C_DLAT(RN3_mulrn), C_DLAT(RN4_mul),
    C_DLAT(RN5_addrn_neg),
    // issue-only
    C_ISSUE(S2R_warpid), C_ISSUE(S2R_smid), C_ISSUE(S2R_laneid),
    C_ISSUE(S2R_tid), C_ISSUE(S2UR_ctaid), C_ISSUE(S2UR_nctaid),
    C_ISSUE(S2UR_smid), C_ISSUE(CS2R_clock), C_ISSUE(CS2R_gtimer),
    C_ISSUE(CS2R_zero),
  };
  // launch each once so the binary is exercised end-to-end on device
  std::uint64_t* d_out; std::uint32_t *d_sink, *d_seeds;
  MB_CHECK(cudaMalloc(&d_out, 8));
  MB_CHECK(cudaMalloc(&d_sink, 512 * 4));
  MB_CHECK(cudaMalloc(&d_seeds, 128 * 4));
  for (const auto& c : cases) {
    if (args.only.size() && c.name.find(args.only) == std::string::npos)
      continue;
    // dispatch by harness: all kernel pointers share the codegen path;
    // launch signature differs, so call through the known members.
  }
  // explicit launches (types differ per harness)
  std::uint32_t p0 = 0x12345678u + (args.launches & 7), p1 = 0x76543210u, p2 = 0x0badf00du;
#define L_LAT(O)  k_lat<O, 8><<<1, 32>>>(d_out, d_seeds, d_sink, 1)
#define L_ULAT(O) k_ulat<O, 8><<<1, 32>>>(d_out, d_sink, p0, p1, p2)
#define L_PLAT(O) k_plat<O, 8><<<1, 32>>>(d_out, d_sink, p0)
#define L_THR(O)  k_thr<O, 2><<<1, 512>>>(d_out, d_sink, 1, 0x9e37u)
#define L_DTHR(O) k_dthr<O, 2><<<1, 512>>>(d_out, d_sink, 1, 0x9e37u)
#define L_DLAT(O) k_dlat<O, 8><<<1, 32>>>(d_out, d_sink, 1, 0x9e37u)
#define L_ISSUE(O) k_issue<O, 8><<<1, 32>>>(d_out, d_seeds, d_sink, 1)
  L_LAT(IABS_abs); L_LAT(IADD_add64); L_LAT(IADD_addc); L_LAT(IADD_sub);
  L_LAT(VIADD_bigimm); L_LAT(VIADD_smallimm); L_LAT(VIADDMNMX_minadd);
  L_LAT(VIADDMNMX_maxadd); L_LAT(VIMNMX3_min2_s32); L_LAT(VIMNMX3_min2_u32);
  L_LAT(LEA_shladd); L_LAT(LEA_madwide); L_LAT(HFMA2_fma);
  L_LAT(SEL_selp); L_LAT(FSEL_selp); L_LAT(ISETP_pair); L_LAT(FSETP_pair);
  L_LAT(DSETP_pair); L_LAT(VOTE_any); L_LAT(VOTE_all); L_LAT(VOTE_ballot);
  L_LAT(ELECT_mask); L_LAT(SHFL_imm_idx); L_LAT(SHFL_imm_bfly);
  L_LAT(SHFL_imm_up); L_LAT(SHFL_imm_down); L_LAT(SHFL_reg_idx);
  L_LAT(REDUX_add); L_LAT(REDUX_min); L_LAT(REDUX_max); L_LAT(REDUX_and);
  L_LAT(UMOV_mov); L_LAT(R2UR_bound); L_LAT(P2R_pack8); L_LAT(R2P_unpack);
  L_LAT(UP2UR_predparam); L_LAT(FCHK_div);
  L_ULAT(USGXT_bfe); L_ULAT(USGXT_cvt); L_ULAT(UFLO_bfind);
  L_ULAT(UPOPC_popc); L_ULAT(UPRMT_prmt); L_ULAT(USHF_shfl);
  L_ULAT(USEL_slct); L_ULAT(UISETP_sel); L_ULAT(UMOV_mov);
  L_PLAT(PLOP3_xor); L_PLAT(PLOP3_and); L_PLAT(ISETP_pred);
  L_THR(IADD_sub); L_THR(VIADD_smallimm); L_THR(SEL_selp);
  L_DTHR(DADD_add); L_DTHR(DMUL_mul);
  L_DTHR(RN1_addrn); L_DTHR(RN2_subrn); L_DTHR(RN3_mulrn); L_DTHR(RN4_mul);
  L_DTHR(RN5_addrn_neg);
  L_DLAT(RN1_addrn); L_DLAT(RN2_subrn); L_DLAT(RN3_mulrn); L_DLAT(RN4_mul);
  L_DLAT(RN5_addrn_neg);
  L_ISSUE(S2R_warpid); L_ISSUE(S2R_smid); L_ISSUE(S2R_laneid);
  L_ISSUE(S2R_tid); L_ISSUE(S2UR_ctaid); L_ISSUE(S2UR_nctaid);
  L_ISSUE(S2UR_smid); L_ISSUE(CS2R_clock); L_ISSUE(CS2R_gtimer);
  L_ISSUE(CS2R_zero);
  MB_CHECK(cudaGetLastError());
  MB_CHECK(cudaDeviceSynchronize());
  std::fprintf(stderr, "idioms: %zu kernels launched\n", cases.size());
  return 0;
}
