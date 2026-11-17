// instruction.service result latency (plan field 1) and per-SMSP issue
// interval (field 7) for single-instruction ALU / conversion / SFU / shuffle
// classes.
//
//   lat OP U   one warp, U dependent instances of OP between two clock reads;
//              each instance reads the previous instance's destination
//              (alternating x/y "Fibonacci" form where the op has two inputs,
//              so ptxas cannot fold the chain).  slope over U = issue ->
//              dependent issue = engine result_latency (register ready =
//              issue + field 1).
//   thr OP K   4 * wps warps (wps per SMSP), each runs 8 independent chains,
//              K steps each, between bar.sync's; slope over K / (8 * wps) =
//              cycles per instance per SMSP (field 7 semantics when the
//              instruction admission is the limit).
// The SASS gate (sass_check.py) requires exactly U (8K) target instructions
// in the window and nothing else but NOPs.
#include "../mb.cuh"

#include <cuda_fp16.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// ---------------------------------------------------------------- op table
// T: register type; op(d, a, b): d = f(a, b); the chain alternates
// x = f(y, x), y = f(x, y), so every instance reads the previous one's result
#define FIB_U32(NAME, ROOT, ASM)                                                       \
  struct NAME {                                                                       \
    using T = std::uint32_t;                                                          \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) { asm volatile(ASM : "=r"(d) : "r"(a), "r"(b)); } \
  };
#define FIB_U32_EC(NAME, ROOT, ASM)                                                    \
  struct NAME {                                                                       \
    using T = std::uint32_t;                                                          \
    static constexpr bool single_chain = true;                                        \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) {         \
      asm volatile(ASM : "=&r"(d) : "r"(a), "r"(b));                                \
    }                                                                                 \
  };
#define UNARY_U32(NAME, ROOT, ASM)                                                     \
  struct NAME {                                                                       \
    using T = std::uint32_t;                                                          \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T, std::uint32_t) { asm volatile(ASM : "=r"(d) : "r"(a)); } \
  };
#define FIB_F32(NAME, ROOT, ASM)                                                       \
  struct NAME {                                                                       \
    using T = float;                                                                  \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) { asm volatile(ASM : "=f"(d) : "f"(a), "f"(b)); } \
  };
#define UNARY_F32(NAME, ROOT, ASM)                                                     \
  struct NAME {                                                                       \
    using T = float;                                                                  \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T, std::uint32_t) { asm volatile(ASM : "=f"(d) : "f"(a)); } \
  };
#define FIB_F64(NAME, ROOT, ASM)                                                       \
  struct NAME {                                                                       \
    using T = double;                                                                 \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) { asm volatile(ASM : "=d"(d) : "d"(a), "d"(b)); } \
  };
#define FIB_B64(NAME, ROOT, ASM)                                                       \
  struct NAME {                                                                       \
    using T = std::uint64_t;                                                          \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) { asm volatile(ASM : "=l"(d) : "l"(a), "l"(b)); } \
  };

#define FIB_H16(NAME, ROOT, ASM)                                                       \
  struct NAME {                                                                        \
    using T = std::uint16_t;                                                           \
    static constexpr bool single_chain = true;                                         \
    static constexpr const char* name = #NAME;                                         \
    static constexpr const char* root = ROOT;                                          \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) {         \
      asm volatile(ASM : "=&h"(d) : "h"(a), "h"(b));                                \
    }                                                                                  \
  };

// integer
// three register inputs: ptxas keeps IADD3 (a two-input add may become IMAD.IADD)
struct IADD3 {
  using T = std::uint32_t;
  static constexpr const char* name = "IADD3";
  static constexpr const char* root = "IADD3";
  __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) {
    asm volatile("{\n\t.reg .u32 t;\n\tadd.u32 t, %1, %2;\n\tadd.u32 %0, t, %3;\n\t}" : "=r"(d) : "r"(a), "r"(b), "r"(a ^ 0u));
  }
};
FIB_U32(IMAD, "IMAD", "mad.lo.u32 %0, %1, %2, %1;")
FIB_U32(LOP3, "LOP3", "lop3.b32 %0, %1, %2, 0x5a5a5a5a, 0x96;")
FIB_U32(SHF, "SHF", "shf.l.wrap.b32 %0, %1, %2, 7;")
FIB_U32(PRMT, "PRMT", "prmt.b32 %0, %1, %2, 0x6152;")
FIB_U32(VIMNMX, "VIMNMX", "min.u32 %0, %1, %2;")
FIB_U32(IMADHI, "IMAD", "mul.hi.u32 %0, %1, %2;")
UNARY_U32(POPC, "POPC", "popc.b32 %0, %1;")
UNARY_U32(FLO, "FLO", "bfind.u32 %0, %1;")
UNARY_U32(BREV, "BREV", "brev.b32 %0, %1;")
UNARY_U32(I2F, "I2F", "cvt.rn.f32.s16 %0, %1;")
UNARY_U32(I2FP, "I2FP", "cvt.rn.f32.u32 %0, %1;")
UNARY_U32(F2I, "F2I", "cvt.rzi.s32.f32 %0, %1;")
UNARY_U32(F2FP, "F2FP", "cvt.rn.bf16x2.f32 %0, %1, %1;")
// source lane = lane ^ 1 (register): a constant source lane makes the value
// uniform after one step and ptxas drops the rest of the chain
struct SHFL_IDX {
  using T = std::uint32_t;
  static constexpr const char* name = "SHFL_IDX";
  static constexpr const char* root = "SHFL";
  __device__ __forceinline__ static void op(T& d, T a, T, std::uint32_t) {
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(d) : "r"(a), "r"((threadIdx.x & 31u) ^ 1u));
  }
};
UNARY_U32(SHFL_BFLY, "SHFL", "shfl.sync.bfly.b32 %0, %1, 1, 0x1f, 0xffffffff;")
UNARY_U32(SHFL_UP, "SHFL", "shfl.sync.up.b32 %0, %1, 1, 0, 0xffffffff;")
UNARY_U32(SHFL_DOWN, "SHFL", "shfl.sync.down.b32 %0, %1, 1, 0x1f, 0xffffffff;")
UNARY_U32(REDUX, "REDUX", "redux.sync.add.u32 %0, %1, 0xffffffff;")
// fp32
FIB_F32(FFMA, "FFMA", "fma.rn.f32 %0, %1, %2, %1;")
FIB_F32(FADD, "FADD", "add.rn.f32 %0, %1, %2;")
FIB_F32(FMUL, "FMUL", "mul.rn.f32 %0, %1, %2;")
FIB_F32(FMNMX, "FMNMX", "max.f32 %0, %1, %2;")
// rcp(rcp(x)) folds: MUFU.RCP + FADD pairs (the FADD latency is subtracted)
FIB_F32(MUFU_RCP, "MUFU", "{\n\t.reg .f32 t;\n\trcp.approx.ftz.f32 t, %1;\n\tadd.rn.f32 %0, t, %2;\n\t}")
UNARY_F32(MUFU_RSQ, "MUFU", "rsqrt.approx.ftz.f32 %0, %1;")
UNARY_F32(MUFU_LG2, "MUFU", "lg2.approx.ftz.f32 %0, %1;")
UNARY_F32(MUFU_SIN, "MUFU", "sin.approx.ftz.f32 %0, %1;")
UNARY_F32(MUFU_TANH, "MUFU", "tanh.approx.f32 %0, %1;")
UNARY_F32(MUFU_SQRT, "MUFU", "sqrt.approx.ftz.f32 %0, %1;")
UNARY_F32(MUFU_EX2, "MUFU", "ex2.approx.ftz.f32 %0, %1;")
// packed
FIB_U32_EC(HFMA2, "HFMA2", "fma.rn.f16x2 %0, %1, %2, %1;")
FIB_H16(HADD2, "HADD2", "add.rn.f16 %0, %1, %2;")
FIB_H16(HMUL2, "HMUL2", "mul.rn.f16 %0, %1, %2;")
FIB_U32_EC(F16X2_ADD, "HFMA2", "add.rn.f16x2 %0, %1, %2;")
FIB_U32_EC(F16X2_SUB, "HFMA2", "sub.rn.f16x2 %0, %1, %2;")
FIB_U32_EC(F16X2_MUL, "HFMA2", "mul.rn.f16x2 %0, %1, %2;")
FIB_U32_EC(F16X2_MAX, "HMNMX2", "max.f16x2 %0, %1, %2;")
FIB_H16(F16_SCALAR_SUB, "HADD2", "sub.rn.f16 %0, %1, %2;")
FIB_H16(F16_SCALAR_FMA, "HFMA2", "fma.rn.f16 %0, %1, %2, %1;")
FIB_H16(F16_SCALAR_MIN, "HMNMX2", "min.f16 %0, %1, %2;")
FIB_H16(F16_SCALAR_MAX, "HMNMX2", "max.f16 %0, %1, %2;")
FIB_U32_EC(BF16X2_ADD, "HFMA2", "add.rn.bf16x2 %0, %1, %2;")
FIB_U32_EC(BF16X2_SUB, "HFMA2", "sub.rn.bf16x2 %0, %1, %2;")
FIB_U32_EC(BF16X2_MUL, "HFMA2", "mul.rn.bf16x2 %0, %1, %2;")
FIB_U32_EC(BF16X2_FMA, "HFMA2", "fma.rn.bf16x2 %0, %1, %2, %1;")
FIB_U32_EC(BF16X2_MIN, "HMNMX2", "min.bf16x2 %0, %1, %2;")
FIB_U32_EC(BF16X2_MAX, "HMNMX2", "max.bf16x2 %0, %1, %2;")
FIB_H16(BF16_SCALAR_ADD, "HADD2", "add.rn.bf16 %0, %1, %2;")
FIB_H16(BF16_SCALAR_SUB, "HADD2", "sub.rn.bf16 %0, %1, %2;")
FIB_H16(BF16_SCALAR_MUL, "HMUL2", "mul.rn.bf16 %0, %1, %2;")
FIB_H16(BF16_SCALAR_FMA, "HFMA2", "fma.rn.bf16 %0, %1, %2, %1;")
FIB_H16(BF16_SCALAR_MIN, "HMNMX2", "min.bf16 %0, %1, %2;")
FIB_H16(BF16_SCALAR_MAX, "HMNMX2", "max.bf16 %0, %1, %2;")
FIB_U32_EC(HSETP2, "HSETP2",
        "{\n\t.reg .pred p, q;\n\tsetp.neu.f16x2 p|q, %1, %2;\n\tselp.b32 %0, %1, %2, p;\n\t}")
FIB_U32_EC(HSETP2_BF16, "HSETP2",
        "{\n\t.reg .pred p, q;\n\tsetp.neu.bf16x2 p|q, %1, %2;\n\tselp.b32 %0, %1, %2, p;\n\t}")

struct F16_CVT_F32 {
  using T = std::uint32_t;
  static constexpr bool single_chain = true;
  static constexpr const char* name = "F16_CVT_F32";
  static constexpr const char* root = "HADD2";
  __device__ __forceinline__ static void op(T& d, T a, T, std::uint32_t) {
    float result;
    asm volatile("cvt.f32.f16 %0, %1;" : "=&f"(result)
                 : "h"(static_cast<std::uint16_t>(a)));
    d = __float_as_uint(result);
  }
};

struct FHADD {
  using T = std::uint32_t;
  static constexpr bool single_chain = true;
  static constexpr const char* name = "FHADD";
  static constexpr const char* root = "FHADD";
  __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) {
    float result;
    asm volatile("add.rn.f32.f16 %0, %1, %2;" : "=&f"(result)
                 : "h"(static_cast<std::uint16_t>(a)),
                   "f"(__uint_as_float(b)));
    d = __float_as_uint(result);
  }
};

struct FHFMA {
  using T = std::uint32_t;
  static constexpr bool single_chain = true;
  static constexpr const char* name = "FHFMA";
  static constexpr const char* root = "FHFMA";
  __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t c) {
    float result;
    asm volatile("fma.rn.f32.f16 %0, %1, %2, %3;" : "=&f"(result)
                 : "h"(static_cast<std::uint16_t>(a)),
                   "h"(static_cast<std::uint16_t>(b)),
                   "f"(__uint_as_float(c)));
    d = __float_as_uint(result);
  }
};

struct FHADD_BF16 {
  using T = std::uint32_t;
  static constexpr bool single_chain = true;
  static constexpr const char* name = "FHADD_BF16";
  static constexpr const char* root = "FHADD";
  __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) {
    float result;
    asm volatile("add.rn.f32.bf16 %0, %1, %2;" : "=&f"(result)
                 : "h"(static_cast<std::uint16_t>(a)),
                   "f"(__uint_as_float(b)));
    d = __float_as_uint(result);
  }
};

struct FHFMA_BF16 {
  using T = std::uint32_t;
  static constexpr bool single_chain = true;
  static constexpr const char* name = "FHFMA_BF16";
  static constexpr const char* root = "FHFMA";
  __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t c) {
    float result;
    asm volatile("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=&f"(result)
                 : "h"(static_cast<std::uint16_t>(a)),
                   "h"(static_cast<std::uint16_t>(b)),
                   "f"(__uint_as_float(c)));
    d = __float_as_uint(result);
  }
};
FIB_B64(FFMA2, "FFMA2", "fma.rn.f32x2 %0, %1, %2, %1;")
FIB_B64(FADD2, "FADD2", "add.rn.f32x2 %0, %1, %2;")
FIB_B64(FMUL2, "FMUL2", "mul.rn.f32x2 %0, %1, %2;")
// fp64
FIB_F64(DFMA, "DFMA", "fma.rn.f64 %0, %1, %2, %1;")
FIB_F64(DADD, "DADD", "add.rn.f64 %0, %1, %2;")
FIB_F64(DMUL, "DMUL", "mul.rn.f64 %0, %1, %2;")

// batch 4 (pipe membership / issue interval of the select, compare, 3-input
// and misc integer classes).  k: a loop-invariant kernel-argument operand;
// the selects' predicate comes from k, so ptxas hoists that compare and the
// window holds the select alone; the compare+select pairs are timed as pairs
#define FIBK_U32(NAME, ROOT, ASM)                                                      \
  struct NAME {                                                                       \
    using T = std::uint32_t;                                                          \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t k) {      \
      asm volatile(ASM : "=r"(d) : "r"(a), "r"(b), "r"(k));                           \
    }                                                                                 \
  };
#define FIBK_F32(NAME, ROOT, ASM)                                                      \
  struct NAME {                                                                       \
    using T = float;                                                                  \
    static constexpr const char* name = #NAME;                                        \
    static constexpr const char* root = ROOT;                                         \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t k) {      \
      asm volatile(ASM : "=f"(d) : "f"(a), "f"(b), "f"(__uint_as_float(k)));          \
    }                                                                                 \
  };
FIBK_U32(SEL, "SEL", "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\tselp.b32 %0, %1, %2, p;\n\t}")
FIBK_U32(ISETP_SEL, "ISETP", "{\n\t.reg .pred p;\n\tsetp.lt.u32 p, %1, %3;\n\tselp.b32 %0, %1, %2, p;\n\t}")
FIBK_F32(FSEL, "FSEL", "{\n\t.reg .pred p;\n\tsetp.ne.f32 p, %3, 0f00000000;\n\tselp.f32 %0, %1, %2, p;\n\t}")
FIBK_F32(FSETP_FSEL, "FSETP", "{\n\t.reg .pred p;\n\tsetp.lt.f32 p, %1, %3;\n\tselp.f32 %0, %1, %2, p;\n\t}")
FIBK_F32(FMNMX3, "FMNMX3", "max.f32 %0, %1, %2, %3;")
FIBK_U32(VIMNMX3, "VIMNMX3", "{\n\t.reg .s32 t;\n\tmin.s32 t, %1, %2;\n\tmin.s32 %0, t, %3;\n\t}")
UNARY_F32(FRND, "FRND", "cvt.rni.f32.f32 %0, %1;")
// (no LEA / IABS: ptxas splits (a << k) + b between LEA and IMAD to balance
// pipes, and folds abs(abs(x)))
FIB_U32(IDP4A, "IDP", "dp4a.u32.u32 %0, %1, %2, %1;")
FIB_U32(BMSK, "BMSK", "bmsk.clamp.b32 %0, %1, %2;")
FIB_U32(SGXT, "SGXT", "szext.clamp.s32 %0, %1, %2;")
// warp-collective: MATCH (lanes with equal values), VOTE.ballot of a compare
UNARY_U32(MATCH, "MATCH", "match.any.sync.b32 %0, %1, 0xffffffff;")
FIBK_U32(ISETP_VOTE, "VOTE", "{\n\t.reg .pred p;\n\tsetp.lt.u32 p, %1, %3;\n\tvote.sync.ballot.b32 %0, p, 0xffffffff;\n\t}")

// batch 6: mixed f32/b16 arithmetic and packed BF16 forms. The PTX unpack
// exposes both halves of the 32-bit chain operand; ptxas must select the
// declared low/high half directly for the measured instruction.
#define FIB_MIXED_ADD(NAME, ROOT, TYPE, ROUND, SAT, HALF)                       \
  struct NAME {                                                                \
    using T = float;                                                           \
    static constexpr const char* name = #NAME;                                 \
    static constexpr const char* root = ROOT;                                  \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) { \
      asm volatile("{\n\t.reg .b16 lo, hi;\n\t"                             \
                   "mov.b32 {lo, hi}, %1;\n\t"                                \
                   "add." ROUND SAT ".f32." TYPE " %0, " HALF ", %2;\n\t}" \
                   : "=f"(d)                                                  \
                   : "r"(__float_as_uint(a)), "f"(b));                        \
    }                                                                          \
  };
#define FIB_MIXED_FMA(NAME, ROOT, TYPE, ROUND, SAT, HALF_A, HALF_B)            \
  struct NAME {                                                                \
    using T = float;                                                           \
    static constexpr const char* name = #NAME;                                 \
    static constexpr const char* root = ROOT;                                  \
    __device__ __forceinline__ static void op(T& d, T a, T b, std::uint32_t) { \
      asm volatile("{\n\t.reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"             \
                   "mov.b32 {a_lo, a_hi}, %1;\n\t"                            \
                   "mov.b32 {b_lo, b_hi}, %2;\n\t"                            \
                   "fma." ROUND SAT ".f32." TYPE " %0, " HALF_A ", "       \
                   HALF_B ", %3;\n\t}"                                        \
                   : "=f"(d)                                                  \
                   : "r"(__float_as_uint(a)), "r"(__float_as_uint(b)),       \
                     "f"(a));                                                 \
    }                                                                          \
  };
#define FIB_MIXED_TYPE(TAG, TYPE)                                               \
  FIB_MIXED_ADD(FHADD_##TAG##_RN, "FHADD", TYPE, "rn", "", "lo")             \
  FIB_MIXED_ADD(FHADD_##TAG##_RN_SAT, "FHADD", TYPE, "rn", ".sat", "lo")     \
  FIB_MIXED_ADD(FHADD_##TAG##_RZ, "FHADD", TYPE, "rz", "", "lo")             \
  FIB_MIXED_ADD(FHADD_##TAG##_RZ_SAT, "FHADD", TYPE, "rz", ".sat", "lo")     \
  FIB_MIXED_ADD(FHADD_##TAG##_RM, "FHADD", TYPE, "rm", "", "lo")             \
  FIB_MIXED_ADD(FHADD_##TAG##_RM_SAT, "FHADD", TYPE, "rm", ".sat", "lo")     \
  FIB_MIXED_ADD(FHADD_##TAG##_RP, "FHADD", TYPE, "rp", "", "lo")             \
  FIB_MIXED_ADD(FHADD_##TAG##_RP_SAT, "FHADD", TYPE, "rp", ".sat", "lo")     \
  FIB_MIXED_FMA(FHFMA_##TAG##_RN, "FHFMA", TYPE, "rn", "", "a_lo", "b_lo") \
  FIB_MIXED_FMA(FHFMA_##TAG##_RN_SAT, "FHFMA", TYPE, "rn", ".sat",          \
                "a_lo", "b_lo")                                               \
  FIB_MIXED_FMA(FHFMA_##TAG##_RZ, "FHFMA", TYPE, "rz", "", "a_lo", "b_lo")  \
  FIB_MIXED_FMA(FHFMA_##TAG##_RZ_SAT, "FHFMA", TYPE, "rz", ".sat",          \
                "a_lo", "b_lo")                                               \
  FIB_MIXED_FMA(FHFMA_##TAG##_RM, "FHFMA", TYPE, "rm", "", "a_lo", "b_lo")  \
  FIB_MIXED_FMA(FHFMA_##TAG##_RM_SAT, "FHFMA", TYPE, "rm", ".sat",          \
                "a_lo", "b_lo")                                               \
  FIB_MIXED_FMA(FHFMA_##TAG##_RP, "FHFMA", TYPE, "rp", "", "a_lo", "b_lo")  \
  FIB_MIXED_FMA(FHFMA_##TAG##_RP_SAT, "FHFMA", TYPE, "rp", ".sat",          \
                "a_lo", "b_lo")                                               \
  FIB_MIXED_ADD(FHSUB_##TAG##_RN, "FHADD", TYPE, "rn", "", "lo")             \
  FIB_MIXED_ADD(FHADD_##TAG##_RN_H1, "FHADD", TYPE, "rn", "", "hi")         \
  FIB_MIXED_FMA(FHFMA_##TAG##_RN_H1, "FHFMA", TYPE, "rn", "", "a_hi",       \
                "b_hi")
FIB_MIXED_TYPE(BF16, "bf16")
FIB_MIXED_TYPE(F16, "f16")
#undef FIB_MIXED_TYPE
#undef FIB_MIXED_FMA
#undef FIB_MIXED_ADD

FIB_U32(HFMA2_BF16, "HFMA2", "fma.rn.bf16x2 %0, %1, %2, %1;")
FIB_U32(HMNMX2_BF16, "HMNMX2", "min.bf16x2 %0, %1, %2;")
FIBK_U32(HSETP2_SEL, "HSETP2",
         "{\n\t.reg .pred p, q;\n\tsetp.ne.f16x2 p|q, %1, %3;\n\t"
         "selp.b32 %0, %1, %2, p;\n\t}")
FIBK_U32(HSETP2_BF16_SEL, "HSETP2",
         "{\n\t.reg .pred p, q;\n\tsetp.ne.bf16x2 p|q, %1, %3;\n\t"
         "selp.b32 %0, %1, %2, p;\n\t}")
struct HADD2_F32 {
  using T = float;
  static constexpr const char* name = "HADD2_F32";
  static constexpr const char* root = "HADD2";
  __device__ __forceinline__ static void op(T& d, T a, T, std::uint32_t) {
    asm volatile("{\n\t.reg .f16 lo, hi;\n\t"
                 "mov.b32 {lo, hi}, %1;\n\tcvt.f32.f16 %0, lo;\n\t}"
                 : "=f"(d)
                 : "r"(__float_as_uint(a)));
  }
};

#define MB_FP16_ALU_OPS(X)                                                       \
  X(FHADD_BF16_RN) X(FHADD_BF16_RN_SAT) X(FHADD_BF16_RZ)                      \
  X(FHADD_BF16_RZ_SAT) X(FHADD_BF16_RM) X(FHADD_BF16_RM_SAT)                  \
  X(FHADD_BF16_RP) X(FHADD_BF16_RP_SAT) X(FHFMA_BF16_RN)                     \
  X(FHFMA_BF16_RN_SAT) X(FHFMA_BF16_RZ) X(FHFMA_BF16_RZ_SAT)                 \
  X(FHFMA_BF16_RM) X(FHFMA_BF16_RM_SAT) X(FHFMA_BF16_RP)                     \
  X(FHFMA_BF16_RP_SAT) X(FHSUB_BF16_RN) X(FHADD_BF16_RN_H1)                  \
  X(FHFMA_BF16_RN_H1) X(FHADD_F16_RN) X(FHADD_F16_RN_SAT)                    \
  X(FHADD_F16_RZ) X(FHADD_F16_RZ_SAT) X(FHADD_F16_RM)                        \
  X(FHADD_F16_RM_SAT) X(FHADD_F16_RP) X(FHADD_F16_RP_SAT)                    \
  X(FHFMA_F16_RN) X(FHFMA_F16_RN_SAT) X(FHFMA_F16_RZ)                        \
  X(FHFMA_F16_RZ_SAT) X(FHFMA_F16_RM) X(FHFMA_F16_RM_SAT)                    \
  X(FHFMA_F16_RP) X(FHFMA_F16_RP_SAT) X(FHSUB_F16_RN)                       \
  X(FHADD_F16_RN_H1) X(FHFMA_F16_RN_H1) X(HFMA2_BF16) X(HMNMX2_BF16)         \
  X(HSETP2_SEL) X(HSETP2_BF16_SEL) X(HADD2_F32)

#define MB_ALU_OPS(X)                                                                                          \
  X(IADD3) X(IMAD) X(LOP3) X(SHF) X(PRMT) X(VIMNMX) X(IMADHI) X(POPC) X(FLO) X(BREV) X(I2F)                     \
  X(I2FP) X(F2I) X(F2FP) X(SHFL_IDX) X(SHFL_BFLY) X(SHFL_UP) X(SHFL_DOWN) X(FFMA) X(FADD)              \
  X(FMUL) X(FMNMX) X(MUFU_RCP) X(MUFU_RSQ) X(MUFU_LG2) X(MUFU_SIN) X(MUFU_TANH) X(MUFU_SQRT) X(MUFU_EX2)     \
  X(HFMA2) X(HADD2) X(HMUL2) X(FFMA2) X(FADD2) X(FMUL2) X(DFMA) X(DADD) X(DMUL)                            \
  X(F16X2_ADD) X(F16X2_SUB) X(F16X2_MUL) X(F16X2_MAX)                                                       \
  X(F16_SCALAR_SUB) X(F16_SCALAR_FMA) X(F16_SCALAR_MIN) X(F16_SCALAR_MAX)                                  \
  X(BF16X2_ADD) X(BF16X2_SUB) X(BF16X2_MUL) X(BF16X2_FMA) X(BF16X2_MIN) X(BF16X2_MAX)                      \
  X(BF16_SCALAR_ADD) X(BF16_SCALAR_SUB) X(BF16_SCALAR_MUL) X(BF16_SCALAR_FMA) X(BF16_SCALAR_MIN)           \
  X(BF16_SCALAR_MAX) X(FHADD) X(FHFMA) X(FHADD_BF16) X(FHFMA_BF16) X(F16_CVT_F32)                          \
  X(HSETP2) X(HSETP2_BF16)                                                                                 \
  X(SEL) X(ISETP_SEL) X(FSEL) X(FSETP_FSEL) X(FMNMX3) X(VIMNMX3) X(FRND) X(IDP4A) \
  X(BMSK) X(SGXT) X(MATCH) X(ISETP_VOTE)

template <class T>
__device__ __forceinline__ T seed_val(std::uint32_t s) {
  if constexpr (std::is_same_v<T, float>) return 1.0f + static_cast<float>(s & 7) * 0.001f;
  else if constexpr (std::is_same_v<T, double>) return 1.0 + static_cast<double>(s & 7) * 0.001;
  else if constexpr (std::is_same_v<T, std::uint64_t>) return 0x3f8000003f800000ull + (s & 7);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return static_cast<T>(0x3c00u + (s & 7));
  else return 0x3c003c00u + (s & 7);
}

template <class T>
__device__ __forceinline__ std::uint32_t fold(T v) {
  if constexpr (std::is_same_v<T, float>) return __float_as_uint(v);
  else if constexpr (std::is_same_v<T, double>) return static_cast<std::uint32_t>(__double_as_longlong(v));
  else return static_cast<std::uint32_t>(v);
}

template <class O, class = void>
struct single_chain_op : std::false_type {};
template <class O>
struct single_chain_op<O, std::void_t<decltype(O::single_chain)>>
    : std::bool_constant<O::single_chain> {};
template <class O>
__device__ __forceinline__ void single_chain_step(
    typename O::T& x, typename O::T& z, std::uint32_t k) {
  O::op(x, x, z, k);
}

// UNI = 0: per-lane seeds (vector datapath); UNI = 1: warp-uniform seeds, so
// ptxas runs integer chains on the uniform datapath (UIADD3, ULOP3, ...).
template <class O, int U, int UNI>
__global__ void __launch_bounds__(32, 1) k_lat(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  using T = typename O::T;
  const std::uint32_t lane = UNI ? 0u : threadIdx.x;
  T x = seed_val<T>(s + lane), y = seed_val<T>(s * 3u + 1u + 5u * lane);
  T z = seed_val<T>(s * 11u + 9u + lane);
  const std::uint32_t k = s ^ 0x3f800000u;  // loop-invariant, not provably zero / nonzero
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if constexpr (single_chain_op<O>::value) {
        single_chain_step<O>(x, z, k);
      } else if (i & 1) O::op(y, x, y, k);
      else O::op(x, y, x, k);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = fold(x) ^ fold(y) ^ fold(z);
}

template <class T>
__device__ __forceinline__ T vload(std::uint32_t a) {
  T v;
  if constexpr (std::is_same_v<T, float>) asm volatile("ld.volatile.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  else if constexpr (std::is_same_v<T, double>) asm volatile("ld.volatile.shared.f64 %0, [%1];" : "=d"(v) : "r"(a) : "memory");
  else if constexpr (std::is_same_v<T, std::uint64_t>) asm volatile("ld.volatile.shared.u64 %0, [%1];" : "=l"(v) : "r"(a) : "memory");
  else if constexpr (std::is_same_v<T, std::uint16_t>) asm volatile("ld.volatile.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory");
  else asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}

template <class T>
__device__ __forceinline__ void vstore(std::uint32_t a, T v) {
  if constexpr (std::is_same_v<T, float>) asm volatile("st.volatile.shared.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
  else if constexpr (std::is_same_v<T, double>) asm volatile("st.volatile.shared.f64 [%0], %1;" ::"r"(a), "d"(v) : "memory");
  else if constexpr (std::is_same_v<T, std::uint64_t>) asm volatile("st.volatile.shared.u64 [%0], %1;" ::"r"(a), "l"(v) : "memory");
  else if constexpr (std::is_same_v<T, std::uint16_t>) asm volatile("st.volatile.shared.u16 [%0], %1;" ::"r"(a), "h"(v) : "memory");
  else asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

// The chains start from volatile shared loads after the first bar.sync and
// end in volatile shared stores before the second: ptxas moves register
// arithmetic across bar.sync and clock reads, the memory operations it
// cannot, so every warp's instances sit between the two barriers.
template <class O, int K, int UNI>
__global__ void __launch_bounds__(512, 1) k_thr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  using T = typename O::T;
  __shared__ __align__(16) T pinb[512 * 16];
  // UNI = 2: the same 16 warps, but only the four on SMSP 0 (warp % 4 == 0)
  // run the chains: per-SMSP pipe -> same per-op interval as wps1 x 4
  // warps; a pipe shared by the SM's four SMSPs -> 4x faster than wps4
  const std::uint32_t lane = UNI == 1 ? (threadIdx.x >> 5) : threadIdx.x;
  const bool active = UNI != 2 || ((threadIdx.x >> 5) & 3u) == 0u;
  // slot j of thread t at (j * 512 + t): consecutive lanes hit consecutive
  // words, so the pin loads/stores are bank-conflict free
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(T));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(T));
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    vstore<T>(base + j * kStride, seed_val<T>(s + lane + j));
    vstore<T>(base + (8 + j) * kStride, seed_val<T>(s * 3u + j + 5u * lane));
  }
  T x[8], y[8];
  const std::uint32_t k = s ^ 0x3f800000u;
  T chain_z = seed_val<T>(s * 7u + 3u + threadIdx.x);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
    if (active) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x[j] = vload<T>(base + j * kStride);
        y[j] = vload<T>(base + (8 + j) * kStride);
      }
#pragma unroll
      for (int i = 0; i < K; ++i) {
        if constexpr (single_chain_op<O>::value) {
          single_chain_step<O>(x[0], chain_z, k);
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            if (i & 1) O::op(y[j], x[j], y[j], k);
            else O::op(x[j], y[j], x[j], k);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        vstore<T>(base + j * kStride, x[j]);
        vstore<T>(base + (8 + j) * kStride, y[j]);
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = fold(vload<T>(base));
}


// Mixed throughput: chains 0..3 run op A, chains 4..7 run op B (same pin
// scheme).  Per-SMSP cycles per (A + B) pair against the A-only and B-only
// rates tell whether A and B share an issue resource (sum) or overlap (max).
template <class A, class B, int K>
__global__ void __launch_bounds__(512, 1) k_mix(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  using TA = typename A::T;
  using TB = typename B::T;
  __shared__ __align__(16) std::uint64_t pinb[512 * 16];
  constexpr std::uint32_t kStride = 512u * 8u;
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 8u;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    vstore<TA>(base + j * kStride, seed_val<TA>(s + threadIdx.x + j));
    vstore<TA>(base + (8 + j) * kStride, seed_val<TA>(s * 3u + j + 5u * threadIdx.x));
    vstore<TB>(base + (4 + j) * kStride, seed_val<TB>(s + threadIdx.x + j));
    vstore<TB>(base + (12 + j) * kStride, seed_val<TB>(s * 3u + j + 5u * threadIdx.x));
  }
  TA xa[4], ya[4];
  TB xb[4], yb[4];
  const std::uint32_t k = s ^ 0x3f800000u;
  TA chain_z_a[4];
  TB chain_z_b[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    chain_z_a[j] = seed_val<TA>(s + threadIdx.x + j + 7u);
    chain_z_b[j] = seed_val<TB>(s * 3u + threadIdx.x + j + 11u);
  }
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
    if constexpr (single_chain_op<A>::value || single_chain_op<B>::value) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        xa[j] = vload<TA>(base + j * kStride);
        ya[j] = vload<TA>(base + (8 + j) * kStride);
        xb[j] = vload<TB>(base + (4 + j) * kStride);
        yb[j] = vload<TB>(base + (12 + j) * kStride);
      }
#pragma unroll 1
      for (int i = 0; i < K; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          single_chain_step<A>(xa[j], chain_z_a[j], k);
          single_chain_step<B>(xb[j], chain_z_b[j], k);
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        vstore<TA>(base + j * kStride, xa[j]);
        vstore<TA>(base + (8 + j) * kStride, ya[j]);
        vstore<TB>(base + (4 + j) * kStride, xb[j]);
        vstore<TB>(base + (12 + j) * kStride, yb[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        xa[j] = vload<TA>(base + j * kStride);
        ya[j] = vload<TA>(base + (8 + j) * kStride);
        xb[j] = vload<TB>(base + (4 + j) * kStride);
        yb[j] = vload<TB>(base + (12 + j) * kStride);
      }
#pragma unroll
      for (int i = 0; i < K; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (i & 1) {
            A::op(ya[j], xa[j], ya[j], k);
            B::op(yb[j], xb[j], yb[j], k);
          } else {
            A::op(xa[j], ya[j], xa[j], k);
            B::op(xb[j], yb[j], xb[j], k);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        vstore<TA>(base + j * kStride, xa[j]);
        vstore<TA>(base + (8 + j) * kStride, ya[j]);
        vstore<TB>(base + (4 + j) * kStride, xb[j]);
        vstore<TB>(base + (12 + j) * kStride, yb[j]);
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = fold(vload<TA>(base)) ^ fold(vload<TB>(base + 4 * kStride));
}


// Issue-slot test (scheduler.policy issue_slots, sched_rules.issue_slots_ok):
// NA of the 8 chains run op A, the other 8 - NA op B.  With A = FFMA (1 cycle
// per warp instruction per SMSP) and B a 2-cycle pipe at NA = 6, the pipes
// allow 6 + 2 instructions per 6 cycles per warp-step group, so a single
// issue slot (1 warp instruction / cycle / SMSP) is the binding limit.
template <class A, class B, int NA, int K>
__global__ void __launch_bounds__(512, 1) k_mixr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  using TA = typename A::T;
  using TB = typename B::T;
  constexpr int NB = 8 - NA;
  __shared__ __align__(16) std::uint64_t pinb[512 * 16];
  constexpr std::uint32_t kStride = 512u * 8u;
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 8u;
#pragma unroll
  for (int j = 0; j < NA; ++j) {
    vstore<TA>(base + j * kStride, seed_val<TA>(s + threadIdx.x + j));
    vstore<TA>(base + (8 + j) * kStride, seed_val<TA>(s * 3u + j + 5u * threadIdx.x));
  }
#pragma unroll
  for (int j = 0; j < NB; ++j) {
    vstore<TB>(base + (NA + j) * kStride, seed_val<TB>(s + threadIdx.x + j));
    vstore<TB>(base + (8 + NA + j) * kStride, seed_val<TB>(s * 3u + j + 5u * threadIdx.x));
  }
  TA xa[NA], ya[NA];
  TB xb[NB], yb[NB];
  const std::uint32_t k = s ^ 0x3f800000u;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < NA; ++j) {
      xa[j] = vload<TA>(base + j * kStride);
      ya[j] = vload<TA>(base + (8 + j) * kStride);
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      xb[j] = vload<TB>(base + (NA + j) * kStride);
      yb[j] = vload<TB>(base + (8 + NA + j) * kStride);
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j < NA) {
          if (i & 1) A::op(ya[j], xa[j], ya[j], k);
          else A::op(xa[j], ya[j], xa[j], k);
        } else {
          if (i & 1) B::op(yb[j - NA], xb[j - NA], yb[j - NA], k);
          else B::op(xb[j - NA], yb[j - NA], xb[j - NA], k);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < NA; ++j) {
      vstore<TA>(base + j * kStride, xa[j]);
      vstore<TA>(base + (8 + j) * kStride, ya[j]);
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      vstore<TB>(base + (NA + j) * kStride, xb[j]);
      vstore<TB>(base + (8 + NA + j) * kStride, yb[j]);
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = fold(vload<TA>(base)) ^ fold(vload<TB>(base + NA * kStride));
}

using Kern = void (*)(std::uint64_t*, std::uint32_t*, int, std::uint32_t);
struct Case {
  std::string op, mode;
  int u, chains_per_thread;
  Kern k;
};

template <class O, int UNI, int... Us>
void add_lat(std::vector<Case>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({std::string(UNI ? "U." : "") + O::name, "lat", Us, 1,
                &k_lat<O, Us, UNI>}), ...);
}
template <class A, class B, int... Ks>
void add_mix(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({std::string(A::name) + "+" + B::name, "mix", Ks, 4,
                &k_mix<A, B, Ks>}), ...);
}
template <class A, class B, int NA, int... Ks>
void add_mixr(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({std::string(A::name) + "+" + B::name + ".r" + std::to_string(NA),
                "slots", Ks, 8, &k_mixr<A, B, NA, Ks>}), ...);
}
template <class O, int UNI, int... Ks>
void add_thr(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {
  constexpr int chains = single_chain_op<O>::value ? 1 : 8;
  (v.push_back({std::string(UNI == 1 ? "U." : UNI == 2 ? "S0." : "") +
                O::name, "thr", Ks, chains, &k_thr<O, Ks, UNI>}), ...);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "alu");
  std::vector<Case> cases;
#define MB_ADD(O)                                                        \
  add_lat<O, 0>(cases, std::integer_sequence<int, 8, 16, 32, 64>{});    \
  add_thr<O, 0>(cases, std::integer_sequence<int, 16, 32, 64>{});
  MB_ALU_OPS(MB_ADD)
  MB_FP16_ALU_OPS(MB_ADD)
#undef MB_ADD
  // The mixed-precision FP16/BF16 issue windows need a longer K to pass the
  // frozen 1% linearity gate; retain the existing 16/32/64 points as well.
#define MB_LONG_THR(OP) \
  add_thr<OP, 0>(cases, std::integer_sequence<int, 128, 256>{});
  MB_LONG_THR(FHADD) MB_LONG_THR(FHFMA)
  MB_LONG_THR(FHADD_BF16) MB_LONG_THR(FHFMA_BF16)
#undef MB_LONG_THR
  // shared-resource tests (see k_mix)
  add_mix<IMAD, FFMA>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<FFMA2, FFMA>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<MUFU_RSQ, FFMA>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<MUFU_RSQ, IMAD>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<MUFU_EX2, MUFU_RSQ>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<F2I, MUFU_RSQ>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<POPC, MUFU_RSQ>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<SHFL_BFLY, FFMA>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<SHFL_BFLY, MUFU_RSQ>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<IMAD, FFMA2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  // pipe membership matrix: every candidate X against FFMA (fma pipe), IMAD
  // (the half-rate pipe IMAD shares with FFMA2) and IADD3 (integer ALU)
#define MB_MIX3(X)                                                             \
  add_mix<X, FFMA>(cases, std::integer_sequence<int, 16, 32, 64>{});          \
  add_mix<X, IMAD>(cases, std::integer_sequence<int, 16, 32, 64>{});          \
  add_mix<X, IADD3>(cases, std::integer_sequence<int, 16, 32, 64>{});
  MB_MIX3(LOP3) MB_MIX3(SHF) MB_MIX3(PRMT) MB_MIX3(VIMNMX) MB_MIX3(FMNMX) MB_MIX3(F2FP) MB_MIX3(I2FP)
  MB_MIX3(HFMA2) MB_MIX3(FADD) MB_MIX3(FMUL) MB_MIX3(IMADHI) MB_MIX3(DFMA)
  MB_MIX3(FADD2) MB_MIX3(FMUL2) MB_MIX3(HADD2) MB_MIX3(HMUL2) MB_MIX3(SHFL_IDX) MB_MIX3(SHFL_UP)
  MB_MIX3(SHFL_DOWN) MB_MIX3(DADD) MB_MIX3(DMUL) MB_MIX3(MUFU_EX2) MB_MIX3(F2I) MB_MIX3(POPC)
  MB_MIX3(SEL) MB_MIX3(ISETP_SEL) MB_MIX3(FSEL) MB_MIX3(FSETP_FSEL) MB_MIX3(FMNMX3) MB_MIX3(VIMNMX3)
  MB_MIX3(FRND) MB_MIX3(IDP4A) MB_MIX3(BMSK) MB_MIX3(SGXT)
  MB_MIX3(F16X2_ADD) MB_MIX3(F16X2_SUB) MB_MIX3(F16X2_MUL) MB_MIX3(F16X2_MAX)
  MB_MIX3(F16_SCALAR_SUB) MB_MIX3(F16_SCALAR_FMA) MB_MIX3(F16_SCALAR_MIN) MB_MIX3(F16_SCALAR_MAX)
  MB_MIX3(BF16X2_ADD) MB_MIX3(BF16X2_SUB) MB_MIX3(BF16X2_MUL) MB_MIX3(BF16X2_FMA)
  MB_MIX3(BF16X2_MIN) MB_MIX3(BF16X2_MAX) MB_MIX3(BF16_SCALAR_ADD) MB_MIX3(BF16_SCALAR_SUB)
  MB_MIX3(BF16_SCALAR_MUL) MB_MIX3(BF16_SCALAR_FMA) MB_MIX3(BF16_SCALAR_MIN) MB_MIX3(BF16_SCALAR_MAX)
  MB_MIX3(FHADD) MB_MIX3(FHFMA) MB_MIX3(FHADD_BF16) MB_MIX3(FHFMA_BF16) MB_MIX3(F16_CVT_F32)
  MB_MIX3(HSETP2) MB_MIX3(HSETP2_BF16)
  MB_MIX3(MATCH) MB_MIX3(ISETP_VOTE)
  MB_MIX3(FHADD_BF16_RN) MB_MIX3(FHADD_F16_RN) MB_MIX3(FHFMA_BF16_RN)
  MB_MIX3(FHFMA_F16_RN) MB_MIX3(HSETP2_SEL) MB_MIX3(HADD2_F32)
#undef MB_MIX3
  add_mix<IADD3, FFMA>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<IADD3, IMAD>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<FFMA2, IADD3>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<MUFU_RSQ, IADD3>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<SHFL_BFLY, IADD3>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mix<SHFL_BFLY, IMAD>(cases, std::integer_sequence<int, 16, 32, 64>{});

  // issue slots: 6 FFMA + 2 {LOP3 | IMAD} chains per warp step
  add_mixr<FFMA, LOP3, 6>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mixr<FFMA, IMAD, 6>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_mixr<FFMA, LOP3, 4>(cases, std::integer_sequence<int, 16, 32, 64>{});

  // pipe scope: four warps on SMSP 0 only (variant thr.S0.<op>.wps4)
  add_thr<FFMA, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<IMAD, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<IADD3, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<FFMA2, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<MUFU_RSQ, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<MUFU_EX2, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<SHFL_BFLY, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<DFMA, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<DADD, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<F2I, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<HFMA2, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<HADD2, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  add_thr<HMUL2, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
#define MB_S0(OP) add_thr<OP, 2>(cases, std::integer_sequence<int, 16, 32, 64>{});
  MB_S0(F16X2_ADD) MB_S0(F16X2_SUB) MB_S0(F16X2_MUL) MB_S0(F16X2_MAX)
  MB_S0(F16_SCALAR_SUB) MB_S0(F16_SCALAR_FMA) MB_S0(F16_SCALAR_MIN) MB_S0(F16_SCALAR_MAX)
  MB_S0(BF16X2_ADD) MB_S0(BF16X2_SUB) MB_S0(BF16X2_MUL) MB_S0(BF16X2_FMA)
  MB_S0(BF16X2_MIN) MB_S0(BF16X2_MAX) MB_S0(BF16_SCALAR_ADD) MB_S0(BF16_SCALAR_SUB)
  MB_S0(BF16_SCALAR_MUL) MB_S0(BF16_SCALAR_FMA) MB_S0(BF16_SCALAR_MIN) MB_S0(BF16_SCALAR_MAX)
  MB_S0(FHADD) MB_S0(FHFMA) MB_S0(FHADD_BF16) MB_S0(FHFMA_BF16) MB_S0(F16_CVT_F32)
  MB_S0(HSETP2) MB_S0(HSETP2_BF16)
  MB_S0(SEL)
#undef MB_S0
#define MB_LONG_S0(OP) \
  add_thr<OP, 2>(cases, std::integer_sequence<int, 128, 256>{});
  MB_LONG_S0(FHADD) MB_LONG_S0(FHFMA)
  MB_LONG_S0(FHADD_BF16) MB_LONG_S0(FHFMA_BF16)
#undef MB_LONG_S0

  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * args.reps));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * 512));
  for (const auto& c : cases) {
    const bool s0 = c.op.rfind("S0.", 0) == 0;
    for (int wps : (c.mode == "lat" ? std::vector<int>{0} : (c.mode == "mix" || c.mode == "slots" || s0 ? std::vector<int>{4} : std::vector<int>{1, 4}))) {
      const std::string variant = c.mode + "." + c.op + (wps ? ".wps" + std::to_string(wps) : "");
      if (!mb_selected(args, variant)) continue;
      std::vector<double> s;
      for (int l = 0; l < args.launches; ++l) {
        c.k<<<1, wps ? 128 * wps : 32>>>(d_out, d_sink, args.reps, static_cast<std::uint32_t>(l));
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        mb_collect(d_out, args.reps, 1, 0, s);
      }
      w.row(variant, c.u, "\"op\":\"" + c.op + "\",\"mode\":\"" +
            c.mode + "\",\"wps\":" + std::to_string(wps) +
            ",\"chains_per_thread\":" +
            std::to_string(c.chains_per_thread), s);
    }
  }
  return w.failures ? 1 : 0;
}
