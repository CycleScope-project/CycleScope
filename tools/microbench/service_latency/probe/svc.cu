// instruction.service, wave 7: predicated-off service facts and active
// pipe/fence facts for the remaining placeholder groups.
//
//   poff.<R>   512 threads (16 warps, 4/SMSP): K steps x 8 chains of
//              predicated-off instances of representative R; the predicate
//              comes from kz[] (all zeros) at run time, so every instance is
//              off.  slope over K / (8*4) = cycles per predicated-off warp
//              instruction per SMSP.
//   poff0.<R>  same window, only warps with warpid%4==0 (four on SMSP0) run
//              the body: per-SMSP vs SM-shared discrimination.
//   act.<P>    active chains of P (same shape); act0.<P> = SMSP0-only.
//   mix.<R>+<P> per chain step one predicated-off R and one active P on
//              independent chains; interval is per pair.
//   amix.<A>+<B> two active streams (EX2 pipe-sharing positive controls).
//   fence.<F>.w<W> K fences per participating warp between bar.syncs;
//              W: 1=one warp, 4x1=4 warps one/SMSP, 4s0=4 warps on SMSP0,
//              16=4/SMSP, grid=one 4-warp CTA per SM (host reports max).
//   upoff/uact/umix: uniform-datapath forms (UIADD3/ULOP3/USHF/UIMAD, S2UR,
//              LDCU) built on kernel-argument chains, same shapes.
//   tpoff (tmem_ld/tmem_st) and umma (UTCHMMA, elected single thread) live
//              in their own kernels so a gate failure there does not hide
//              the vector results.
#include "../mb.cuh"
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include "tmem_ops.cuh"
#include "umma_impl.cuh"

#include <functional>
#include <string>
#include <vector>

__device__ std::uint32_t svc_gbuf[1024];
constexpr int kTab = 1024;
__constant__ std::uint32_t svc_ctab[kTab];

// ------------------------------------------------------------ op structs
// off(d, s, k): one predicated-off instance: the instruction writes d under
// @p (p = k != 0, k = 0 at run time).  The destination must be a live chain
// register: with a dead temp ptxas if-converts the whole block into a
// branch, which is not a predicated-off issue.  on(d, a, b): active step.

#define DEFOP(SN, TY, CSTR, LB, RT, OFFA, ONA)                                 \
  struct SN {                                                                 \
    using T = TY;                                                             \
    static constexpr const char* name = LB;                                   \
    static constexpr const char* root = RT;                                   \
    static constexpr bool has_on = true;                                      \
    __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {    \
      asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"     \
                   "@p " OFFA ";\n\t}" : "+" CSTR(d) : CSTR(s), "r"(k));      \
    }                                                                         \
    __device__ __forceinline__ static void on(T& d, T a, T b) {                 \
      asm volatile(ONA ";" : "=" CSTR(d) : CSTR(a), CSTR(b));                  \
    }                                                                         \
    __device__ __forceinline__ static void pat(T& d, T a, T) {                  \
      asm volatile(ONA ";" : "=" CSTR(d) : CSTR(a), CSTR(a));                  \
    }                                                                         \
    __device__ __forceinline__ static void offS(T& d, std::uint32_t k) {       \
      asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t"     \
                   "@p " ONA ";\n\t}"                                       \
                   : "+" CSTR(d) : CSTR(d), CSTR(d), "r"(k));                  \
    }                                                                         \
  };

#define DF32(SN, LB, RT, OFFA, ONA) DEFOP(SN, float, "f", LB, RT, OFFA, ONA)
#define DU32(SN, LB, RT, OFFA, ONA) DEFOP(SN, std::uint32_t, "r", LB, RT, OFFA, ONA)
#define DU64(SN, LB, RT, OFFA, ONA) DEFOP(SN, std::uint64_t, "l", LB, RT, OFFA, ONA)
#define DF64(SN, LB, RT, OFFA, ONA) DEFOP(SN, double, "d", LB, RT, OFFA, ONA)

DF32(O_FFMA, "FFMA", "FFMA", "fma.rn.f32 %0, %0, %1, %0", "fma.rn.f32 %0, %1, %2, %1")
DF32(O_FADD, "FADD", "FADD", "add.rn.f32 %0, %0, %1", "add.rn.f32 %0, %1, %2")
DF32(O_FMUL, "FMUL", "FMUL", "mul.rn.f32 %0, %0, %1", "mul.rn.f32 %0, %1, %2")
DF32(O_FMNMX, "FMNMX", "FMNMX", "max.f32 %0, %0, %1", "max.f32 %0, %1, %2")
DF32(O_MUFURSQ, "MUFU.RSQ", "MUFU", "rsqrt.approx.ftz.f32 %0, %0", "rsqrt.approx.ftz.f32 %0, %1")
DF32(O_MUFUEX2, "MUFU.EX2", "MUFU", "ex2.approx.ftz.f32 %0, %0", "ex2.approx.ftz.f32 %0, %1")

DU32(O_HFMA2, "HFMA2", "HFMA2", "fma.rn.f16x2 %0, %0, %1, %0", "fma.rn.f16x2 %0, %1, %2, %1")
DU32(O_HADD2, "HADD2", "HADD2", "add.rn.f16x2 %0, %0, %1", "add.rn.f16x2 %0, %1, %2")
DU32(O_HMUL2, "HMUL2", "HMUL2", "mul.rn.f16x2 %0, %0, %1", "mul.rn.f16x2 %0, %1, %2")
#if SVC_SM100
DU64(O_FFMA2, "FFMA2", "FFMA2", "fma.rn.f32x2 %0, %0, %1, %0", "fma.rn.f32x2 %0, %1, %2, %1")
DU64(O_FADD2, "FADD2", "FADD2", "add.rn.f32x2 %0, %0, %1", "add.rn.f32x2 %0, %1, %2")
DU64(O_FMUL2, "FMUL2", "FMUL2", "mul.rn.f32x2 %0, %0, %1", "mul.rn.f32x2 %0, %1, %2")
#endif

DU32(O_IMAD, "IMAD", "IMAD", "mad.lo.u32 %0, %0, %1, %0", "mad.lo.u32 %0, %1, %2, %1")
DU32(O_IMADHI, "IMAD.HI", "IMAD", "mul.hi.u32 %0, %0, %1", "mul.hi.u32 %0, %1, %2")
DU32(O_IDP, "IDP", "IDP", "dp4a.u32.u32 %0, %0, %1, %0", "dp4a.u32.u32 %0, %1, %2, %1")

DU32(O_IADD3, "IADD3", "IADD3", "add.u32 %0, %0, %1", "add.u32 %0, %1, %2")
DU32(O_LOP3, "LOP3", "LOP3", "lop3.b32 %0, %0, %1, 0x5a5a5a5a, 0x96", "lop3.b32 %0, %1, %2, 0x5a5a5a5a, 0x96")
DU32(O_SHF, "SHF", "SHF", "shf.l.wrap.b32 %0, %0, %1, 7", "shf.l.wrap.b32 %0, %1, %2, 7")
DU32(O_PRMT, "PRMT", "PRMT", "prmt.b32 %0, %0, %1, 0x6152", "prmt.b32 %0, %1, %2, 0x6152")
DU32(O_POPC, "POPC", "POPC", "popc.b32 %0, %0", "popc.b32 %0, %1")
DU32(O_FLO, "FLO", "FLO", "bfind.u32 %0, %0", "bfind.u32 %0, %1")
DU32(O_BREV, "BREV", "BREV", "brev.b32 %0, %0", "brev.b32 %0, %1")

DF64(O_DFMA, "DFMA", "DFMA", "fma.rn.f64 %0, %0, %1, %0", "fma.rn.f64 %0, %1, %2, %1")
DF64(O_DADD, "DADD", "DADD", "add.rn.f64 %0, %0, %1", "add.rn.f64 %0, %1, %2")
DF64(O_DMUL, "DMUL", "DMUL", "mul.rn.f64 %0, %0, %1", "mul.rn.f64 %0, %1, %2")

// --- special forms (hand-written: mixed register types / extra lanes)

// cvt.rn.bf16x2.f32: two f32 in, u32 out; the predicated cvt writes a u32
// temp that an unconditional mov.b32 renames into the f32 chain register.
struct O_F2FP {
  using T = float;
  static constexpr const char* name = "F2FP";
  static constexpr const char* root = "F2FP";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\t.reg .b32 t;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p cvt.rn.bf16x2.f32 t, %0, %1;\n\tmov.b32 %0, t;\n\t}" : "+f"(d) : "f"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("{\n\t.reg .b32 t;\n\t.reg .f32 lo;\n\tcvt.rn.bf16x2.f32 t, %1, %2;\n\t"
                 "mov.b32 lo, t;\n\tadd.rn.f32 %0, lo, %1;\n\t}" : "=f"(d) : "f"(a), "f"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_I2FP {  // cvt.rn.f32.u32: u32 in, f32 out
  using T = std::uint32_t;
  static constexpr const char* name = "I2FP";
  static constexpr const char* root = "I2FP";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\t.reg .f32 t;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p cvt.rn.f32.u32 t, %1;\n\tcvt.rzi.u32.f32 %0, t;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("{\n\t.reg .f32 t;\n\tcvt.rn.f32.u32 t, %1;\n\t"
                 "cvt.rzi.u32.f32 %0, t;\n\t}" : "=r"(d) : "r"(a), "r"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_F2I {  // cvt.rzi.s32.f32
  using T = float;
  static constexpr const char* name = "F2I";
  static constexpr const char* root = "F2I";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\t.reg .s32 t;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p cvt.rzi.s32.f32 t, %1;\n\tcvt.rn.f32.s32 %0, t;\n\t}" : "+f"(d) : "f"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("{\n\t.reg .s32 t;\n\tcvt.rzi.s32.f32 t, %1;\n\t"
                 "cvt.rn.f32.s32 %0, t;\n\t}" : "=f"(d) : "f"(a), "f"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_I2F {  // cvt.rn.f32.s32
  using T = std::uint32_t;
  static constexpr const char* name = "I2F";
  static constexpr const char* root = "I2F";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\t.reg .f32 t;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p cvt.rn.f32.s32 t, %1;\n\tcvt.rzi.u32.f32 %0, t;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("{\n\t.reg .f32 t;\n\tcvt.rn.f32.s32 t, %1;\n\t"
                 "cvt.rzi.u32.f32 %0, t;\n\t}" : "=r"(d) : "r"(a), "r"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_ISETP {  // the compare itself, predicated; result kept via selp
  using T = std::uint32_t;
  static constexpr const char* name = "ISETP";
  static constexpr const char* root = "ISETP";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p setp.lt.u32 q, %0, %1;\n\tselp.b32 %0, %0, %1, q;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("{\n\t.reg .pred q;\n\tsetp.lt.u32 q, %1, %2;\n\t"
                 "selp.b32 %0, %1, %2, q;\n\t}" : "=r"(d) : "r"(a), "r"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_SEL {  // predicated select; select predicate q also from k
  using T = std::uint32_t;
  static constexpr const char* name = "SEL";
  static constexpr const char* root = "SEL";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "setp.eq.u32 q, %2, 0;\n\t@p selp.b32 %0, %0, %1, q;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("{\n\t.reg .pred q;\n\tsetp.lt.u32 q, %1, %2;\n\t"
                 "selp.b32 %0, %1, %2, q;\n\t}" : "=r"(d) : "r"(a), "r"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_SHFLBFLY {
  using T = std::uint32_t;
  static constexpr const char* name = "SHFL.BFLY";
  static constexpr const char* root = "SHFL";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 0x1f, 0xffffffff;" : "=r"(d) : "r"(a), "r"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_SHFLIDX {
  using T = std::uint32_t;
  static constexpr const char* name = "SHFL.IDX";
  static constexpr const char* root = "SHFL";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p shfl.sync.idx.b32 %0, %0, %1, 0x1f, 0xffffffff;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(d) : "r"(a), "r"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_S2R {
  using T = std::uint32_t;
  static constexpr const char* name = "S2R";
  static constexpr const char* root = "S2R";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p mov.u32 %0, %%laneid;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(d));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
struct O_MATCH {
  using T = std::uint32_t;
  static constexpr const char* name = "MATCH";
  static constexpr const char* root = "MATCH";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p match.any.sync.b32 %0, %1, 0xffffffff;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    asm volatile("match.any.sync.b32 %0, %1, 0xffffffff;" : "=r"(d) : "r"(a), "r"(b));
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};
// LEA: (a<<3)+b as two predicated ops; ptxas fuses them into one @P LEA
// when it can, which the gate verifies.
struct O_LEA {
  using T = std::uint32_t;
  static constexpr const char* name = "LEA";
  static constexpr const char* root = "LEA";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(T& d, T s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\t.reg .u32 t;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p shl.b32 t, %0, 3;\n\t@p add.u32 %0, t, %1;\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) {
    d = (a << 3) + b;
  }
  __device__ __forceinline__ static void pat(T& d, T a, T) {
    on(d, a, a);
  }

};

// predicated-off-only ops (no pipelined chain form): fences, barriers, the
// cctl invalidation inside the gpu/sys fence expansion.
#define DEF_OFFONLY(SN, LB, RT, BODY)                                          \
  struct SN {                                                                  \
    using T = std::uint32_t;                                                   \
    static constexpr const char* name = LB;                                    \
    static constexpr const char* root = RT;                                    \
    static constexpr bool has_on = false;                                      \
    __device__ __forceinline__ static void off(T, T, std::uint32_t k) {          \
      asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %0, 0;\n\t"      \
                   "@p " BODY ";\n\t}" ::"r"(k) : "memory");                  \
    }                                                                          \
    __device__ __forceinline__ static void on(T& d, T a, T b) { d = a ^ b; }     \
    __device__ __forceinline__ static void pat(T, T, std::uint32_t) {            \
      asm volatile(BODY ";" ::: "memory");                                     \
    }                                                                          \
  };

DEF_OFFONLY(O_MEMBARALLCTA, "MEMBAR.ALL.CTA", "MEMBAR", "fence.acq_rel.cta")
DEF_OFFONLY(O_MEMBARSCCTA, "MEMBAR.SC.CTA", "MEMBAR", "fence.sc.cta")
DEF_OFFONLY(O_MEMBARALLGPU, "MEMBAR.ALL.GPU", "MEMBAR", "fence.acq_rel.gpu")
DEF_OFFONLY(O_MEMBARSCGPU, "MEMBAR.SC.GPU", "MEMBAR", "fence.sc.gpu")
DEF_OFFONLY(O_MEMBARALLSYS, "MEMBAR.ALL.SYS", "MEMBAR", "fence.acq_rel.sys")
struct O_CCTLIVALL {
  using T = std::uint32_t;
  static constexpr const char* name = "CCTL.IVALL";
  static constexpr const char* root = "CCTL";
  static constexpr bool has_on = false;
  __device__ __forceinline__ static void off(T, T, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\t"
                 "@p discard.global.L2 [%0], 128;\n\t}" ::"l"(&svc_gbuf[0]), "r"(k) : "memory");
  }
  __device__ __forceinline__ static void on(T& d, T a, T b) { d = a ^ b; }
  __device__ __forceinline__ static void pat(T, T, std::uint32_t) {
    asm volatile("discard.global.L2 [%0], 128;" ::"l"(&svc_gbuf[0]) : "memory");
  }
};
DEF_OFFONLY(O_FENCEASYNCS, "FENCE.VIEW.ASYNC.S", "FENCE", "fence.proxy.async.shared::cta")
DEF_OFFONLY(O_FENCEASYNCG, "FENCE.VIEW.ASYNC.G", "FENCE", "fence.proxy.async.global")

extern __shared__ __align__(16) char mb_dyn_smem[];

// kernels that pin through mb_dyn_smem need the 128 KiB dynamic opt-in
// before every launch (default cap 48 KiB -> cudaErrorInvalidValue)
template <class... P, class... A>
static void mb_dyn_launch(void (*k)(P...), A... a) {
  constexpr std::size_t kDyn = 128u * 1024u;
  cudaFuncSetAttribute(reinterpret_cast<const void*>(k),
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(kDyn));
  k<<<1, 512, kDyn>>>(a...);
}

// SYNCS.ARRIVE: predicated mbarrier.arrive on a per-warp smem mbarrier.
// k_syncs<K, OFF> has a dedicated kernel because the chain is barrier state.
// ------------------------------------------------- uniform-datapath ops
// Kernel-argument chains keep ptxas on the uniform datapath; the predicate
// comes from a kernel argument (k=0 at run time), which ptxas may turn into
// a UP-predicated uniform instruction.  on() is the active chain step.
#define DEF_UOP(SN, LB, RT, BODY)                                              \
  struct SN {                                                                  \
    using T = std::uint32_t;                                                   \
    static constexpr const char* name = LB;                                    \
    static constexpr const char* root = RT;                                    \
    static constexpr bool has_on = true;                                       \
    __device__ __forceinline__ static void off(std::uint32_t& d, std::uint32_t s, std::uint32_t k) { \
      asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"             \
                   "@p " BODY ";\n\t}" : "+r"(d) : "r"(s), "r"(k));             \
    }                                                                          \
    __device__ __forceinline__ static void on(std::uint32_t& d, std::uint32_t a, std::uint32_t b) { \
      asm volatile(BODY ";" : "+r"(d) : "r"(a), "r"(b));                         \
    }                                                                          \
    __device__ __forceinline__ static void pat(std::uint32_t& d, std::uint32_t a, std::uint32_t) { \
      asm volatile(BODY ";" : "+r"(d) : "r"(a), "r"(a));                         \
    }                                                                          \
  };

DEF_UOP(U_ULEA, "ULEA", "ULEA", "shl.b32 %0, %1, 3; add.u32 %0, %0, %2")
DEF_UOP(U_UPOPC, "UPOPC", "UPOPC", "popc.b32 %0, %1")
DEF_UOP(U_UIADD3, "UIADD3", "UIADD3", "add.u32 %0, %1, %2")
DEF_UOP(U_ULOP3, "ULOP3", "ULOP3", "lop3.b32 %0, %1, %2, 0x5a5a5a5a, 0x96")
DEF_UOP(U_USHF, "USHF", "USHF", "shf.l.wrap.b32 %0, %1, %2, 7")
DEF_UOP(U_UIMAD, "UIMAD", "UIMAD", "mad.lo.u32 %0, %1, %2, %1")
struct U_S2UR {
  using T = std::uint32_t;
  static constexpr const char* name = "S2UR";
  static constexpr const char* root = "S2UR";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(std::uint32_t& d, std::uint32_t s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\t"
                 "@p mov.u32 %0, %%smid;\n\t}" : "+r"(d) : "r"(k));
  }
  __device__ __forceinline__ static void on(std::uint32_t& d, std::uint32_t a, std::uint32_t b) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(d));
    d += a & 0;
  }
  __device__ __forceinline__ static void pat(std::uint32_t& d, std::uint32_t, std::uint32_t) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(d));
  }
};
struct U_LDCU {
  using T = std::uint32_t;
  static constexpr const char* name = "LDCU";
  static constexpr const char* root = "LDCU";
  static constexpr bool has_on = true;
  __device__ __forceinline__ static void off(std::uint32_t& d, std::uint32_t s, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                 "@p ld.const.u32 %0, [%1];\n\t}" : "+r"(d) : "r"(s), "r"(k));
  }
  __device__ __forceinline__ static void on(std::uint32_t& d, std::uint32_t a, std::uint32_t) {
    asm volatile("{\n\t.reg .u64 t;\n\tmov.u64 t, svc_ctab;\n\tadd.u64 t, t, %1;\n\t"
                 "ld.const.u32 %0, [t];\n\t}" : "=r"(d) : "l"((unsigned long long)(a & (4 * kTab - 4))));
  }
  __device__ __forceinline__ static void pat(std::uint32_t& d, std::uint32_t a, std::uint32_t) {
    on(d, a, a);
  }
};

// ------------------------------------------------------------ pin helpers
template <class T>
__device__ __forceinline__ T vload(std::uint32_t a) {
  T v;
  if constexpr (std::is_same_v<T, float>) asm volatile("ld.volatile.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  else if constexpr (std::is_same_v<T, double>) asm volatile("ld.volatile.shared.f64 %0, [%1];" : "=d"(v) : "r"(a) : "memory");
  else if constexpr (std::is_same_v<T, std::uint64_t>) asm volatile("ld.volatile.shared.u64 %0, [%1];" : "=l"(v) : "r"(a) : "memory");
  else asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
template <class T>
__device__ __forceinline__ void vstore(std::uint32_t a, T v) {
  if constexpr (std::is_same_v<T, float>) asm volatile("st.volatile.shared.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
  else if constexpr (std::is_same_v<T, double>) asm volatile("st.volatile.shared.f64 [%0], %1;" ::"r"(a), "d"(v) : "memory");
  else if constexpr (std::is_same_v<T, std::uint64_t>) asm volatile("st.volatile.shared.u64 [%0], %1;" ::"r"(a), "l"(v) : "memory");
  else asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
template <class T>
__device__ __forceinline__ T seed_t(std::uint32_t s) {
  if constexpr (std::is_same_v<T, float>) return 1.0f + static_cast<float>(s & 7) * 0.001f;
  else if constexpr (std::is_same_v<T, double>) return 1.0 + static_cast<double>(s & 7) * 0.001;
  else if constexpr (std::is_same_v<T, std::uint64_t>) return 0x3f8000003f800000ull + (s & 7);
  else return 0x3c003c00u + (s & 7);
}
template <class T>
__device__ __forceinline__ std::uint32_t fold_t(T v) {
  if constexpr (std::is_same_v<T, float>) return __float_as_uint(v);
  else if constexpr (std::is_same_v<T, double>) return static_cast<std::uint32_t>(__double_as_longlong(v));
  else return static_cast<std::uint32_t>(v);
}

// ------------------------------------------------------------- kernels
// S0 = true: only warps with warpid%4==0 (all on SMSP0) run the body.
// SELF = true: operand-form experiment — predicated self chain (a=op(a,a,..))
// instead of the ping-pong off() chain.
template <class O, int K, bool S0, bool SELF = false>
__global__ void __launch_bounds__(512, 1) k_poff(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  static_assert(!SELF || O::has_on);
  using T = typename O::T;
  T* const pinb = reinterpret_cast<T*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(T));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(T));
  const std::uint32_t k = kz[threadIdx.x & 31];  // 0 at run time: all instances off
  const bool active = !S0 || ((threadIdx.x >> 5) & 3u) == 0u;
#pragma unroll
  for (int j = 0; j < 16; ++j) vstore<T>(base + j * kStride, seed_t<T>(threadIdx.x + 37 * j));
  T x[8], y[8];
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
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if constexpr (SELF) O::offS(x[j], k);
          else if (i & 1) O::off(y[j], x[j], k);
          else O::off(x[j], y[j], k);
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
  out[reps] = fold_t(vload<T>(base));
}

// active streams
template <class O, int K, bool S0>
__global__ void __launch_bounds__(512, 1) k_act(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  using T = typename O::T;
  T* const pinb = reinterpret_cast<T*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(T));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(T));
  const bool active = !S0 || ((threadIdx.x >> 5) & 3u) == 0u;
#pragma unroll
  for (int j = 0; j < 16; ++j) vstore<T>(base + j * kStride, seed_t<T>(threadIdx.x + 37 * j));
  T x[8], y[8];
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
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (i & 1) O::on(y[j], x[j], y[j]);
          else O::on(x[j], y[j], x[j]);
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
  out[reps] = fold_t(vload<T>(base));
}

// one predicated-off R + one active P per chain step, independent chains
template <class R, class P, int K>
__global__ void __launch_bounds__(512, 1) k_mix(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  using TR = typename R::T;
  using TP = typename P::T;
  std::uint64_t* const pinb = reinterpret_cast<std::uint64_t*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * 8u;
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 8u;
  const std::uint32_t k = kz[threadIdx.x & 31];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    vstore<TR>(base + j * kStride, seed_t<TR>(threadIdx.x + 37 * j));
    vstore<TR>(base + (8 + j) * kStride, seed_t<TR>(threadIdx.x + 11 * j));
    vstore<TP>(base + (16 + j) * kStride, seed_t<TP>(threadIdx.x + 37 * j));
    vstore<TP>(base + (24 + j) * kStride, seed_t<TP>(threadIdx.x + 11 * j));
  }
  TR rx[8], ry[8];
  TP px[8], py[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      rx[j] = vload<TR>(base + j * kStride);
      ry[j] = vload<TR>(base + (8 + j) * kStride);
      px[j] = vload<TP>(base + (16 + j) * kStride);
      py[j] = vload<TP>(base + (24 + j) * kStride);
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (i & 1) R::off(ry[j], rx[j], k);
        else R::off(rx[j], ry[j], k);
        P::on(px[j], py[j], px[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      vstore<TR>(base + j * kStride, rx[j]);
      vstore<TR>(base + (8 + j) * kStride, ry[j]);
      vstore<TP>(base + (16 + j) * kStride, px[j]);
      vstore<TP>(base + (24 + j) * kStride, py[j]);
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  out[reps] = fold_t(vload<TR>(base)) ^ fold_t(vload<TP>(base + 16 * kStride));
}

// two active streams (A and B share or overlap)
template <class A, class B, int K>
__global__ void __launch_bounds__(512, 1) k_amix(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  using TA = typename A::T;
  using TB = typename B::T;
  std::uint64_t* const pinb = reinterpret_cast<std::uint64_t*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * 8u;
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 8u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    vstore<TA>(base + j * kStride, seed_t<TA>(threadIdx.x + 37 * j));
    vstore<TA>(base + (8 + j) * kStride, seed_t<TA>(threadIdx.x + 11 * j));
    vstore<TB>(base + (16 + j) * kStride, seed_t<TB>(threadIdx.x + 37 * j));
    vstore<TB>(base + (24 + j) * kStride, seed_t<TB>(threadIdx.x + 11 * j));
  }
  TA ax[8], ay[8];
  TB bx[8], by[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ax[j] = vload<TA>(base + j * kStride);
      ay[j] = vload<TA>(base + (8 + j) * kStride);
      bx[j] = vload<TB>(base + (16 + j) * kStride);
      by[j] = vload<TB>(base + (24 + j) * kStride);
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (i & 1) A::on(ay[j], ax[j], ay[j]);
        else A::on(ax[j], ay[j], ax[j]);
        B::on(bx[j], by[j], bx[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      vstore<TA>(base + j * kStride, ax[j]);
      vstore<TA>(base + (8 + j) * kStride, ay[j]);
      vstore<TB>(base + (16 + j) * kStride, bx[j]);
      vstore<TB>(base + (24 + j) * kStride, by[j]);
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  out[reps] = fold_t(vload<TA>(base)) ^ fold_t(vload<TB>(base + 16 * kStride));
}

// ------------------------------------------------- uniform-datapath kernels
// 8 uniform chains carried across reps; the only consumer is a uniform
// constant-bank address.
template <class O, int K, bool S0>
__global__ void __launch_bounds__(512, 1) k_upoff(std::uint64_t* out, const std::uint32_t* kz,
                                                std::uint32_t sx, std::uint32_t sy, std::uint32_t sk,
                                                int reps) {
  const bool active = !S0 || ((threadIdx.x >> 5) & 3u) == 0u;
  std::uint32_t x[8], y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = sx + 0x01010101u * j;
    y[j] = sy ^ (0x10203040u * j);
  }
  const std::uint32_t k = sk;  // 0 at run time: every instance is off
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
    if (active) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (i & 1) O::off(y[j], x[j], k);
          else O::off(x[j], y[j], k);
        }
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t acc = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) acc ^= x[j] ^ y[j];
  if (threadIdx.x == 0) out[reps + 1] = svc_ctab[acc & (kTab - 1)];
}

template <class O, int K, bool S0>
__global__ void __launch_bounds__(512, 1) k_uact(std::uint64_t* out, const std::uint32_t* kz,
                                               std::uint32_t sx, std::uint32_t sy, std::uint32_t sk,
                                               int reps) {
  const bool active = !S0 || ((threadIdx.x >> 5) & 3u) == 0u;
  std::uint32_t x[8], y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = sx + 0x01010101u * j;
    y[j] = sy ^ (0x10203040u * j);
  }
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
    if (active) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (i & 1) O::on(y[j], x[j], y[j]);
          else O::on(x[j], y[j], x[j]);
        }
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t acc = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) acc ^= x[j] ^ y[j];
  if (threadIdx.x == 0) out[reps + 1] = svc_ctab[acc & (kTab - 1)];
}

// uniform predicated-off R + vector active P
template <class R, class P, int K>
__global__ void __launch_bounds__(512, 1) k_umx(std::uint64_t* out, const std::uint32_t* kz,
                                               std::uint32_t sx, std::uint32_t sy, std::uint32_t sk,
                                               int reps) {
  using TP = typename P::T;
  TP* const pinb = reinterpret_cast<TP*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(TP));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(TP));
  std::uint32_t x[8], y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = sx + 0x01010101u * j;
    y[j] = sy ^ (0x10203040u * j);
    vstore<TP>(base + j * kStride, seed_t<TP>(threadIdx.x + 37 * j));
  }
  const std::uint32_t k = sk;
  TP px[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) px[j] = vload<TP>(base + j * kStride);
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (i & 1) R::off(y[j], x[j], k);
        else R::off(x[j], y[j], k);
        P::on(px[j], px[j], px[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) vstore<TP>(base + j * kStride, px[j]);
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t acc = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) acc ^= x[j] ^ y[j] ^ fold_t(vload<TP>(base + j * kStride));
  if (threadIdx.x == 0) out[reps + 1] = svc_ctab[acc & (kTab - 1)];
}

// ------------------------------------------------- dedicated kernels
// SYNCS: predicated (OFF=1) or active mbarrier.arrive on a per-warp smem
// mbarrier; 8 "chains" = the same mbarrier touched 8x per step.
template <int K, int OFF>
__global__ void __launch_bounds__(512, 1) k_syncs(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  __shared__ __align__(8) std::uint64_t mbar[16];
  const std::uint32_t a = mb_smem_u32(&mbar[threadIdx.x >> 5]);
  const std::uint32_t k = kz[threadIdx.x & 31];
  if (threadIdx.x % 32 == 0) mb_mbar_init(a, 32);
  __syncthreads();
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (OFF) {
          asm volatile("{\n\t.reg .pred p;\n\t.reg .b64 st;\n\tsetp.ne.u32 p, %1, 0;\n\t"
                       "@p mbarrier.arrive.shared::cta.b64 st, [%0];\n\t}" ::"r"(a), "r"(k) : "memory");
        } else {
          asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.shared::cta.b64 st, [%0];\n\t}" ::"r"(a) : "memory");
        }
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
}

#if SVC_SM100
// tmem_ld / tmem_st: predicated-off tcgen05.ld / tcgen05.st.  The TMEM
// allocation happens once up front; tcgen05 ops may reject @p, in which
// case this kernel fails the gate and the group is reported unmeasured.
template <int K, int ST>
__global__ void __launch_bounds__(512, 1) k_tpoff(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  __shared__ std::uint32_t alloc_slot[4];
  const std::uint32_t k = kz[threadIdx.x & 31];
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) mb_tmem_alloc<1>(mb_smem_u32(alloc_slot), 64);
  __syncthreads();
  const std::uint32_t taddr = alloc_slot[0];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (ST) {
          const std::uint32_t v = k + j;
          asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                       "@p tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n\t}"
                       ::"r"(taddr + j), "r"(v), "r"(k) : "memory");
        } else {
          std::uint32_t r0;
          asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"
                       "@p tcgen05.ld.sync.aligned.32x32b.x1.b32 {%1}, [%0];\n\t}"
                       : "=r"(r0) : "r"(taddr + j), "r"(k) : "memory");
        }
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) {
    mb_tmem_dealloc<1>(taddr, 64);
    mb_tmem_relinquish<1>();
  }
}

// UTCHMMA predicated-off via the instruction's own enable-input predicate:
// one elected thread issues K disabled MMAs (en = kz[0] = 0).
template <int K>
__global__ void __launch_bounds__(32, 1) k_umma(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  __shared__ std::uint32_t alloc_slot[4];
  __shared__ __align__(16) std::uint32_t sab[64 * 16];
  const std::uint32_t en = kz[0];
  if (threadIdx.x < 32) mb_tmem_alloc<1>(mb_smem_u32(alloc_slot), 64);
  __syncwarp();
  const std::uint32_t taddr = alloc_slot[0];
  const std::uint32_t sbase = mb_smem_u32(sab);
  const std::uint64_t adesc = make_sdesc(sbase, 16, 0);
  const std::uint64_t bdesc = make_sdesc(sbase + 8 * 16, 8, 0);
  const std::uint32_t idesc = make_idesc<kF16>(128, 64);
  std::uint32_t lane;
  asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xffffffff;\n\tselp.b32 %0, 1, 0, p;\n\t}" : "=r"(lane));
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncwarp();
    const std::uint64_t t0 = mb_clock();
    if (lane) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        umma<kF16, 1, 0>(taddr, adesc, 0, bdesc, idesc, 0, 0, en);
      }
    }
    __syncwarp();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  if (threadIdx.x < 32) {
    mb_tmem_dealloc<1>(taddr, 64);
    mb_tmem_relinquish<1>();
  }
}

// active LDTM/STTM issue-bound streams: 8 x tcgen05.ld/st (32x32b.x1) per
// step on this warp's own TMEM columns; the completion waits live outside
// the timed chain (one wait::ld / wait::st per rep, after the window).
template <int K, int ST>
__global__ void __launch_bounds__(512, 1) k_tact(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  __shared__ std::uint32_t alloc_slot[4];
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) mb_tmem_alloc<1>(mb_smem_u32(alloc_slot), 64);
  __syncthreads();
  const std::uint32_t taddr = alloc_slot[0] + ((threadIdx.x >> 5) & 3) * 8;  // per-SMSP columns
  std::uint32_t acc = threadIdx.x;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (ST) {
          const std::uint32_t v = acc + j;
          asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(taddr + j), "r"(v) : "memory");
        } else {
          std::uint32_t r0;
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r0) : "r"(taddr + j) : "memory");
          acc += r0;
        }
      }
    }
    // completion waits stay outside the timed window
    if constexpr (ST) asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    else asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  out[reps] = acc;
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) {
    mb_tmem_dealloc<1>(alloc_slot[0], 64);
    mb_tmem_relinquish<1>();
  }
}

// TMEM op (active, issue stream) mixed with an active P chain stream.
template <int ST, class P, int K>
__global__ void __launch_bounds__(512, 1) k_tamix(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  using TP = typename P::T;
  __shared__ std::uint32_t alloc_slot[4];
  TP* const pinb = reinterpret_cast<TP*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(TP));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(TP));
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) mb_tmem_alloc<1>(mb_smem_u32(alloc_slot), 64);
  __syncthreads();
  const std::uint32_t taddr = alloc_slot[0] + ((threadIdx.x >> 5) & 3) * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) vstore<TP>(base + j * kStride, seed_t<TP>(threadIdx.x + 37 * j));
  TP px[8];
  std::uint32_t acc = threadIdx.x;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) px[j] = vload<TP>(base + j * kStride);
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (ST) {
          const std::uint32_t v = acc + j;
          asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(taddr + j), "r"(v) : "memory");
        } else {
          std::uint32_t r0;
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r0) : "r"(taddr + j) : "memory");
          acc += r0;
        }
        P::on(px[j], px[j], px[j]);
      }
    }
    if constexpr (ST) asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    else asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int j = 0; j < 8; ++j) vstore<TP>(base + j * kStride, px[j]);
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  out[reps] = acc ^ fold_t(vload<TP>(base));
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) {
    mb_tmem_dealloc<1>(alloc_slot[0], 64);
    mb_tmem_relinquish<1>();
  }
}

// UMMA issue stream vs a co-running P stream on warps 1..15: warp 0's
// elected lane issues K UTCHMMAs (en=1) while warps 1..15 run 8 P chains.
// UMMA=0: warp 0 idles -> solo-P reference on the same warp set.
template <class P, int K, int UMMA>
__global__ void __launch_bounds__(512, 1) k_umix(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  using TP = typename P::T;
  __shared__ std::uint32_t alloc_slot[4];
  __shared__ __align__(16) std::uint32_t sab[64 * 16];
  TP* const pinb = reinterpret_cast<TP*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(TP));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(TP));
  const int warp = threadIdx.x >> 5;
  __syncthreads();
  if (warp == 0) mb_tmem_alloc<1>(mb_smem_u32(alloc_slot), 64);
  __syncthreads();
  const std::uint32_t taddr = alloc_slot[0];
  const std::uint64_t adesc = make_sdesc(mb_smem_u32(sab), 16, 0);
  const std::uint64_t bdesc = make_sdesc(mb_smem_u32(sab) + 8 * 16, 8, 0);
  const std::uint32_t idesc = make_idesc<kF16>(128, 64);
#pragma unroll
  for (int j = 0; j < 8; ++j) vstore<TP>(base + j * kStride, seed_t<TP>(threadIdx.x + 37 * j));
  std::uint32_t lane = 0;
  if (warp == 0)
    asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xffffffff;\n\tselp.b32 %0, 1, 0, p;\n\t}" : "=r"(lane));
  TP px[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
    if (UMMA && warp == 0 && lane) {
#pragma unroll
      for (int i = 0; i < K; ++i) umma<kF16, 1, 0>(taddr, adesc, 0, bdesc, idesc, 0, 0, 1u);
    } else if (warp != 0) {
#pragma unroll
      for (int j = 0; j < 8; ++j) px[j] = vload<TP>(base + j * kStride);
#pragma unroll
      for (int i = 0; i < K; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) P::on(px[j], px[j], px[j]);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) vstore<TP>(base + j * kStride, px[j]);
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    // warp 1 records the co-run/solo P interval
    if (threadIdx.x == 32) out[rep] = t1 - t0;
  }
  __syncthreads();
  if (warp == 0) {
    mb_tmem_dealloc<1>(taddr, 64);
    mb_tmem_relinquish<1>();
  }
}

#endif
// fences: W = 1 (one warp), 2 (4 warps, one per SMSP), 3 (4 warps on SMSP0),
// 4 (16 warps), 5 (one 4-warp CTA on every SM; out per CTA)
#define DEF_FENCE(FN, LB, ASM)                                                 \
  struct FN {                                                                  \
    static constexpr const char* name = LB;                                    \
    __device__ __forceinline__ static void run() {                             \
      asm volatile(ASM ";" ::: "memory");                                      \
    }                                                                          \
  };
DEF_FENCE(F_CTA_ACQREL, "cta.acq_rel", "fence.acq_rel.cta")
DEF_FENCE(F_CTA_SC, "cta.sc", "fence.sc.cta")
DEF_FENCE(F_GPU_ACQREL, "gpu.acq_rel", "fence.acq_rel.gpu")
DEF_FENCE(F_GPU_SC, "gpu.sc", "fence.sc.gpu")
DEF_FENCE(F_SYS_ACQREL, "sys.acq_rel", "fence.acq_rel.sys")
DEF_FENCE(F_ASYNC_S, "async.shared", "fence.proxy.async.shared::cta")
DEF_FENCE(F_ASYNC_G, "async.global", "fence.proxy.async.global")

template <class F, int K, int W>
__global__ void __launch_bounds__(512, 1) k_fence(std::uint64_t* out, int reps) {
  const int warp = threadIdx.x >> 5;
  bool part = true;
  if constexpr (W == 3) part = (warp & 3) == 0;
  __shared__ std::uint32_t sbuf[512];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    if constexpr (W == 1) __syncwarp();
    else __syncthreads();
    const std::uint64_t t0 = mb_clock();
    if (part) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        F::run();
      }
    }
    if constexpr (W == 1) __syncwarp();
    else __syncthreads();
    const std::uint64_t t1 = mb_clock();
    const int b = W == 5 ? blockIdx.x : 0;
    if (threadIdx.x == 0) out[rep * (W == 5 ? gridDim.x : 1) + b] = t1 - t0;
  }
  sbuf[threadIdx.x] = static_cast<std::uint32_t>(out[0]);
}

__global__ void k_cbase(std::uint32_t* o) {
  std::uint32_t b;
  asm volatile("mov.u32 %0, svc_ctab;" : "=r"(b));
  *o = b;
}


// --------------------------------------------------- guard-patched kernels
// pk_* kernels emit ACTIVE streams plus ONE natively-predicated-off seed op
// before each window, so ptxas materializes the runtime-false kz predicate
// into a real Px register.  sass_patch.py rewrites the guard nibble (bits
// 12-15 of the first qword) of every target instruction in the window to
// that Px; the patched cubin is loaded via cuModuleLoadData at run time.
// The unpatched copies are never launched.

__device__ __forceinline__ void pk_seed(std::uint32_t& keep, std::uint32_t k) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\t"
               "@p popc.b32 %0, %0;\n\t}" : "+r"(keep) : "r"(k));
}

// PP = true: operand-form experiment — unpredicated ping-pong on() chains
// (a=op(b,a,..); b=op(a,b,..)) that the patcher arms; false = self pat().
template <class O, int K, bool S0, bool PP = false>
__device__ __forceinline__ void pk_poff_body(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  using T = typename O::T;
  T* const pinb = reinterpret_cast<T*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(T));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(T));
  const std::uint32_t k = kz[threadIdx.x & 31];
  const bool active = !S0 || ((threadIdx.x >> 5) & 3u) == 0u;
  std::uint32_t keep = threadIdx.x;
#pragma unroll
  for (int j = 0; j < 16; ++j) vstore<T>(base + j * kStride, seed_t<T>(threadIdx.x + 37 * j));
  T x[8], y[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (active) pk_seed(keep, k);
    const std::uint64_t t0 = mb_clock();
    if (active) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x[j] = vload<T>(base + j * kStride);
        y[j] = vload<T>(base + (8 + j) * kStride);
      }
#pragma unroll
      for (int i = 0; i < K; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if constexpr (PP) {
            if (i & 1) O::on(y[j], x[j], x[j]);
            else O::on(x[j], y[j], y[j]);
          } else {
            if (i & 1) O::pat(y[j], x[j], k);
            else O::pat(x[j], y[j], k);
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
  out[reps] = fold_t(vload<T>(base)) ^ keep;
}

// patched-off stream for ops with no register-chain form (fences, membar,
// cctl): the window is K*8 bare instructions, patched to @Px.
template <class O, int K, bool S0>
__device__ __forceinline__ void pk_poffx_body(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  const std::uint32_t k = kz[threadIdx.x & 31];
  const bool active = !S0 || ((threadIdx.x >> 5) & 3u) == 0u;
  std::uint32_t keep = threadIdx.x;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (active) pk_seed(keep, k);
    const std::uint64_t t0 = mb_clock();
    if (active) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) O::pat(keep, keep, k);
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  out[reps] = keep;
}

// patched-off R + active P (register chains), per chain step
template <class R, class P, int K>
__device__ __forceinline__ void pk_pmix_body(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  using TP = typename P::T;
  TP* const pinb = reinterpret_cast<TP*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(TP));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(TP));
  const std::uint32_t k = kz[threadIdx.x & 31];
  std::uint32_t keep = threadIdx.x;
  std::uint32_t rx[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    rx[j] = threadIdx.x * 2654435761u + 37 * j;
    vstore<TP>(base + j * kStride, seed_t<TP>(threadIdx.x + 37 * j));
  }
  TP px[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    pk_seed(keep, k);
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) px[j] = vload<TP>(base + j * kStride);
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        R::pat(rx[j], rx[j], k);
        P::on(px[j], px[j], px[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) vstore<TP>(base + j * kStride, px[j]);
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  out[reps] = keep ^ rx[0] ^ fold_t(vload<TP>(base));
}

// uniform-datapath patched stream: UPx seed on a uniform value
__device__ __forceinline__ void pk_useed(std::uint32_t& keep, std::uint32_t k) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\t"
               "@p add.u32 %0, %0, 1;\n\t}" : "+r"(keep) : "r"(k));
}

template <class O, int K, bool S0>
__device__ __forceinline__ void pk_upoff_body(std::uint64_t* out, const std::uint32_t* kz,
                                            std::uint32_t sx, std::uint32_t sy, std::uint32_t sk,
                                            int reps) {
  const bool active = !S0 || ((threadIdx.x >> 5) & 3u) == 0u;
  std::uint32_t x[8], y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = sx + 0x01010101u * j;
    y[j] = sy ^ (0x10203040u * j);
  }
  std::uint32_t keep = sx;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (active) pk_useed(keep, sk);
    const std::uint64_t t0 = mb_clock();
    if (active) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (i & 1) O::pat(y[j], x[j], sk);
          else O::pat(x[j], y[j], sk);
        }
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t acc = keep;
#pragma unroll
  for (int j = 0; j < 8; ++j) acc ^= x[j] ^ y[j];
  if (threadIdx.x == 0) out[reps + 1] = svc_ctab[acc & (kTab - 1)];
}

// uniform patched-off R + vector active P
template <class R, class P, int K>
__device__ __forceinline__ void pk_umix_body(std::uint64_t* out, const std::uint32_t* kz,
                                             std::uint32_t sx, std::uint32_t sy, std::uint32_t sk,
                                             int reps) {
  using TP = typename P::T;
  TP* const pinb = reinterpret_cast<TP*>(mb_dyn_smem);
  constexpr std::uint32_t kStride = 512u * static_cast<std::uint32_t>(sizeof(TP));
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * static_cast<std::uint32_t>(sizeof(TP));
  std::uint32_t x[8], y[8];
  std::uint32_t keep = sx;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = sx + 0x01010101u * j;
    y[j] = sy ^ (0x10203040u * j);
    vstore<TP>(base + j * kStride, seed_t<TP>(threadIdx.x + 37 * j));
  }
  TP px[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    pk_useed(keep, sk);
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) px[j] = vload<TP>(base + j * kStride);
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        R::pat(x[j], y[j], sk);
        P::on(px[j], px[j], px[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) vstore<TP>(base + j * kStride, px[j]);
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  std::uint32_t acc = keep;
#pragma unroll
  for (int j = 0; j < 8; ++j) acc ^= x[j] ^ y[j] ^ fold_t(vload<TP>(base + j * kStride));
  if (threadIdx.x == 0) out[reps + 1] = svc_ctab[acc & (kTab - 1)];
}

// SYNCS: patched-off mbarrier.arrive stream (active arrive asm, guard
// patched off; the mbarrier is initialized but never arrives).
template <int K>
__device__ void pk_syncs_body(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  __shared__ __align__(8) std::uint64_t mbar[16];
  const std::uint32_t a = mb_smem_u32(&mbar[threadIdx.x >> 5]);
  const std::uint32_t k = kz[threadIdx.x & 31];
  std::uint32_t keep = threadIdx.x;
  if (threadIdx.x % 32 == 0) mb_mbar_init(a, 32);
  __syncthreads();
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    pk_seed(keep, k);
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.shared::cta.b64 st, [%0];\n\t}"
                     ::"r"(a) : "memory");
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  out[reps] = keep;
}

#if SVC_SM100
// TMEM: patched-off tcgen05.ld / tcgen05.st stream
template <int K, int ST>
__device__ void pk_tpoff_body(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  __shared__ std::uint32_t alloc_slot[4];
  const std::uint32_t k = kz[threadIdx.x & 31];
  std::uint32_t keep = threadIdx.x;
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) mb_tmem_alloc<1>(mb_smem_u32(alloc_slot), 64);
  __syncthreads();
  const std::uint32_t taddr = alloc_slot[0];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    pk_seed(keep, k);
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (ST) {
          const std::uint32_t v = k + j;
          asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                       ::"r"(taddr + j), "r"(v) : "memory");
        } else {
          std::uint32_t r0;
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%1}, [%0];"
                       : "=r"(r0) : "r"(taddr + j) : "memory");
          keep ^= r0;
        }
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) {
    mb_tmem_dealloc<1>(taddr, 64);
    mb_tmem_relinquish<1>();
  }
  out[reps] = keep;
}

// UTCHMMA: one elected thread issues K MMAs; patched to @Px off.
template <int K>
__device__ void pk_umma_body(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  __shared__ std::uint32_t alloc_slot[4];
  __shared__ __align__(16) std::uint32_t sab[64 * 16];
  const std::uint32_t k = kz[0];
  std::uint32_t keep = threadIdx.x;
  if (threadIdx.x < 32) mb_tmem_alloc<1>(mb_smem_u32(alloc_slot), 64);
  __syncwarp();
  const std::uint32_t taddr = alloc_slot[0];
  const std::uint32_t sbase = mb_smem_u32(sab);
  const std::uint64_t adesc = make_sdesc(sbase, 16, 0);
  const std::uint64_t bdesc = make_sdesc(sbase + 8 * 16, 8, 0);
  const std::uint32_t idesc = make_idesc<kF16>(128, 64);
  const std::uint32_t lane = threadIdx.x == 0 ? 1 : 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncwarp();
    pk_seed(keep, k);
    const std::uint64_t t0 = mb_clock();
    if (lane) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        umma<kF16, 1, 0>(taddr, adesc, 0, bdesc, idesc, 0, 0, 1);
      }
    }
    __syncwarp();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  if (threadIdx.x < 32) {
    mb_tmem_dealloc<1>(taddr, 64);
    mb_tmem_relinquish<1>();
  }
  out[reps] = keep;
}


#endif
// extern "C" entry points into the patched cubin; names are fixed so the
// host can resolve them with cuModuleGetFunction after sass_patch.py
// rewrites the guard nibbles.
#define PKP3(O)                                                            \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkp_##O##_16_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 16, false>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkp_##O##_32_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 32, false>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkp_##O##_64_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 64, false>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkp_##O##_16_1(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 16, true>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkp_##O##_32_1(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 32, true>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkp_##O##_64_1(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 64, true>(o, k, r); }

#define PKQ3(O)                                                            \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkq_##O##_16_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 16, false, true>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkq_##O##_32_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 32, false, true>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkq_##O##_64_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poff_body<O, 64, false, true>(o, k, r); }

#define PKX3(O)                                                            \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkx_##O##_16_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poffx_body<O, 16, false>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkx_##O##_32_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poffx_body<O, 32, false>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkx_##O##_64_0(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poffx_body<O, 64, false>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkx_##O##_16_1(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poffx_body<O, 16, true>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkx_##O##_32_1(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poffx_body<O, 32, true>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkx_##O##_64_1(std::uint64_t* o, const std::uint32_t* k, int r) { pk_poffx_body<O, 64, true>(o, k, r); }

#define PKM3(R, P, TAG)                                                         \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkm_##TAG##_16(std::uint64_t* o, const std::uint32_t* k, int r) { pk_pmix_body<R, P, 16>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkm_##TAG##_32(std::uint64_t* o, const std::uint32_t* k, int r) { pk_pmix_body<R, P, 32>(o, k, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkm_##TAG##_64(std::uint64_t* o, const std::uint32_t* k, int r) { pk_pmix_body<R, P, 64>(o, k, r); }

#define PKU3(O)                                                            \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pku_##O##_16_0(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_upoff_body<O, 16, false>(o, k, a, b, c, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pku_##O##_32_0(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_upoff_body<O, 32, false>(o, k, a, b, c, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pku_##O##_64_0(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_upoff_body<O, 64, false>(o, k, a, b, c, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pku_##O##_16_1(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_upoff_body<O, 16, true>(o, k, a, b, c, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pku_##O##_32_1(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_upoff_body<O, 32, true>(o, k, a, b, c, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pku_##O##_64_1(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_upoff_body<O, 64, true>(o, k, a, b, c, r); }

#define PKUM3(R, P, TAG)                                                        \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkmu_##TAG##_16(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_umix_body<R, P, 16>(o, k, a, b, c, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkmu_##TAG##_32(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_umix_body<R, P, 32>(o, k, a, b, c, r); } \
  extern "C" __global__ void __launch_bounds__(512, 1)                          \
      pkmu_##TAG##_64(std::uint64_t* o, const std::uint32_t* k, std::uint32_t a, std::uint32_t b, std::uint32_t c, int r) { pk_umix_body<R, P, 64>(o, k, a, b, c, r); }

// patched stream targets
// chain-form ops (validation classes + unobtainable classes)
#define PCHAIN(X) \
  X(O_FFMA) X(O_FADD) X(O_FMUL) X(O_IMAD) X(O_IMADHI) X(O_SHF) X(O_PRMT) \
  X(O_FMNMX) X(O_SEL) X(O_POPC) X(O_MUFURSQ) X(O_MUFUEX2) \
  X(O_IADD3) X(O_LOP3) X(O_ISETP) X(O_F2FP) X(O_I2FP) \
  X(O_HFMA2) X(O_HADD2) X(O_HMUL2) \
  X(O_F2I) X(O_I2F) X(O_FLO) X(O_BREV) \
  X(O_SHFLBFLY) X(O_SHFLIDX) X(O_DFMA) X(O_DADD) X(O_DMUL) \
  X(O_LEA) X(O_S2R) X(O_MATCH)
#if SVC_SM100
#define PCHAIN_X2(X) X(O_FFMA2) X(O_FADD2) X(O_FMUL2)
#else
#define PCHAIN_X2(X)
#endif
// bare-instruction targets (fences, membar, cctl)
#define PBARE(X) \
  X(O_MEMBARALLCTA) X(O_MEMBARSCCTA) X(O_MEMBARALLGPU) X(O_MEMBARSCGPU) \
  X(O_MEMBARALLSYS) X(O_CCTLIVALL) X(O_FENCEASYNCS) X(O_FENCEASYNCG)
// uniform-datapath targets
#define PUNI(X) \
  X(U_UIADD3) X(U_ULOP3) X(U_USHF) X(U_UIMAD) X(U_S2UR) X(U_LDCU) \
  X(U_ULEA) X(U_UPOPC)
// mix R targets (patched-off side); P fixed below
#define PMXR(X) \
  X(O_LEA) X(O_S2R) X(O_MATCH) X(O_MEMBARALLCTA) X(O_MEMBARSCCTA) \
  X(O_MEMBARALLGPU) X(O_MEMBARSCGPU) X(O_MEMBARALLSYS) X(O_CCTLIVALL) \
  X(O_FENCEASYNCS) X(O_FENCEASYNCG)
// uniform R in mixes
#define PMXRU(X) \
  X(U_UIADD3) X(U_ULOP3) X(U_USHF) X(U_UIMAD) X(U_S2UR) X(U_LDCU) \
  X(U_ULEA) X(U_UPOPC)

PCHAIN(PKP3)
PCHAIN_X2(PKP3)
PCHAIN(PKQ3)
PCHAIN_X2(PKQ3)
PBARE(PKX3)
PUNI(PKU3)
#if SVC_PART == 3
#define _PKM_ALLP(R) PKM3(R, O_FFMA, R##_##O_FFMA) PKM3(R, O_IMAD, R##_##O_IMAD) PKM3(R, O_SHF, R##_##O_SHF) PKM3(R, O_MUFURSQ, R##_##O_MUFURSQ)
PMXR(_PKM_ALLP)
#endif
#if SVC_PART == 2
#define _PKMU_ALLP(R) PKUM3(R, O_FFMA, R##_##O_FFMA) PKUM3(R, O_IMAD, R##_##O_IMAD) PKUM3(R, O_SHF, R##_##O_SHF) PKUM3(R, O_MUFURSQ, R##_##O_MUFURSQ)
PMXRU(_PKMU_ALLP)
#endif

extern "C" __global__ void __launch_bounds__(512, 1) pk_syncs_16(std::uint64_t* o, const std::uint32_t* k, int r) { pk_syncs_body<16>(o, k, r); }
extern "C" __global__ void __launch_bounds__(512, 1) pk_syncs_32(std::uint64_t* o, const std::uint32_t* k, int r) { pk_syncs_body<32>(o, k, r); }
extern "C" __global__ void __launch_bounds__(512, 1) pk_syncs_64(std::uint64_t* o, const std::uint32_t* k, int r) { pk_syncs_body<64>(o, k, r); }
#if SVC_SM100
extern "C" __global__ void __launch_bounds__(512, 1) pk_tpoff_ldtm_16(std::uint64_t* o, const std::uint32_t* k, int r) { pk_tpoff_body<16, 0>(o, k, r); }
extern "C" __global__ void __launch_bounds__(512, 1) pk_tpoff_ldtm_32(std::uint64_t* o, const std::uint32_t* k, int r) { pk_tpoff_body<32, 0>(o, k, r); }
extern "C" __global__ void __launch_bounds__(512, 1) pk_tpoff_ldtm_64(std::uint64_t* o, const std::uint32_t* k, int r) { pk_tpoff_body<64, 0>(o, k, r); }
extern "C" __global__ void __launch_bounds__(512, 1) pk_tpoff_sttm_16(std::uint64_t* o, const std::uint32_t* k, int r) { pk_tpoff_body<16, 1>(o, k, r); }
extern "C" __global__ void __launch_bounds__(512, 1) pk_tpoff_sttm_32(std::uint64_t* o, const std::uint32_t* k, int r) { pk_tpoff_body<32, 1>(o, k, r); }
extern "C" __global__ void __launch_bounds__(512, 1) pk_tpoff_sttm_64(std::uint64_t* o, const std::uint32_t* k, int r) { pk_tpoff_body<64, 1>(o, k, r); }
extern "C" __global__ void __launch_bounds__(32, 1) pk_umma_16(std::uint64_t* o, const std::uint32_t* k, int r) { pk_umma_body<16>(o, k, r); }
extern "C" __global__ void __launch_bounds__(32, 1) pk_umma_32(std::uint64_t* o, const std::uint32_t* k, int r) { pk_umma_body<32>(o, k, r); }
extern "C" __global__ void __launch_bounds__(32, 1) pk_umma_64(std::uint64_t* o, const std::uint32_t* k, int r) { pk_umma_body<64>(o, k, r); }
#endif

// ----------------------------------------------------------------- host
using PKern = void (*)(std::uint64_t*, const std::uint32_t*, int);
using UKernS = void (*)(std::uint64_t*, const std::uint32_t*, std::uint32_t, std::uint32_t, std::uint32_t, int);
struct PCase { std::string variant; int k; PKern fn; };
struct UCase2 { std::string variant; int k; UKernS fn; };

template <class O, int... Ks>
void add_poff(std::vector<PCase>& v, const char* tag, bool s0, std::integer_sequence<int, Ks...>) {
  std::string t = tag;
  if (s0) (v.push_back({"poff0." + t, Ks, &k_poff<O, Ks, true>}), ...);
  else (v.push_back({"poff." + t, Ks, &k_poff<O, Ks, false>}), ...);
}

// native predicated-off in the SELF operand form (a = op(a, a, ..)):
// ptxas emits a natively-predicated single-register chain
template <class O, int... Ks>
void add_poff_self(std::vector<PCase>& v, const char* tag, std::integer_sequence<int, Ks...>) {
  const std::string t = tag;
  (v.push_back({"poffself." + t, Ks, &k_poff<O, Ks, false, true>}), ...);
}
template <class O, int... Ks>
void add_act(std::vector<PCase>& v, const char* tag, bool s0, std::integer_sequence<int, Ks...>) {
  std::string t = tag;
  if (s0) (v.push_back({"act0." + t, Ks, &k_act<O, Ks, true>}), ...);
  else (v.push_back({"act." + t, Ks, &k_act<O, Ks, false>}), ...);
}
template <class R, class P, int... Ks>
void add_mix(std::vector<PCase>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({std::string("mix.") + R::name + "+" + P::name, Ks, &k_mix<R, P, Ks>}), ...);
}
template <class A, class B, int... Ks>
void add_amix(std::vector<PCase>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({std::string("amix.") + A::name + "+" + B::name, Ks, &k_amix<A, B, Ks>}), ...);
}
template <class O, int... Ks>
void add_upoff(std::vector<UCase2>& v, const char* tag, bool s0, int act, std::integer_sequence<int, Ks...>) {
  std::string t = tag;
  if (act == 0 && !s0) (v.push_back({"upoff." + t, Ks, &k_upoff<O, Ks, false>}), ...);
  if (act == 0 && s0) (v.push_back({"upoff0." + t, Ks, &k_upoff<O, Ks, true>}), ...);
  if (act == 1 && !s0) (v.push_back({"uact." + t, Ks, &k_uact<O, Ks, false>}), ...);
  if (act == 1 && s0) (v.push_back({"uact0." + t, Ks, &k_uact<O, Ks, true>}), ...);
}
template <class R, class P, int... Ks>
void add_umix(std::vector<UCase2>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({std::string("umix.") + R::name + "+" + P::name, Ks, &k_umx<R, P, Ks>}), ...);
}

using K3 = std::integer_sequence<int, 16, 32, 64>;
using K3F = std::integer_sequence<int, 8, 16, 32>;

#define VOPS(X) \
  X(O_FFMA) X(O_FADD) X(O_FMUL) \
  X(O_HFMA2) X(O_HADD2) X(O_HMUL2) \
  X(O_IMAD) X(O_IMADHI) X(O_IDP) \
  X(O_IADD3) X(O_LOP3) X(O_SHF) X(O_PRMT) X(O_FMNMX) X(O_F2FP) X(O_I2FP) X(O_ISETP) X(O_SEL) \
  X(O_MUFURSQ) X(O_MUFUEX2) X(O_F2I) X(O_I2F) X(O_POPC) X(O_FLO) X(O_BREV) \
  X(O_SHFLBFLY) X(O_SHFLIDX) \
  X(O_DFMA) X(O_DADD) X(O_DMUL) \
  X(O_LEA) X(O_S2R) X(O_MATCH)
#define OFFONLY(X) \
  X(O_MEMBARALLCTA) X(O_MEMBARSCCTA) X(O_MEMBARALLGPU) X(O_MEMBARSCGPU) X(O_MEMBARALLSYS) \
  X(O_CCTLIVALL) X(O_FENCEASYNCS) X(O_FENCEASYNCG)
#if SVC_SM100
#define VOPS_X2(X) X(O_FFMA2) X(O_FADD2) X(O_FMUL2)
#else
#define VOPS_X2(X)
#endif
#define UOPS(X) X(U_UIADD3) X(U_ULOP3) X(U_USHF) X(U_UIMAD) X(U_S2UR) X(U_LDCU)


// ------------------------- patched-cubin launch path ----------------------
static CUmodule g_pmod = (CUmodule)0;
static bool g_pmod_tried = false;

static CUfunction pk_get(const std::string& sym) {
  if (!g_pmod_tried) {
    g_pmod_tried = true;
    const char* path = getenv("SVC_PATCHED_CUBIN");
    std::string p = path && *path ? path : "";
    if (p.empty()) {
      const char* out = getenv("OUT");
      p = (out ? std::string(out) : ".") + "/bin/svc.patched.cubin";
    }
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "[svc] no patched cubin at %s\n", p.c_str()); return nullptr; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(static_cast<std::size_t>(n));
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) { fclose(f); return nullptr; }
    fclose(f);
    if (cuModuleLoadData(&g_pmod, buf.data()) != CUDA_SUCCESS) {
      fprintf(stderr, "[svc] cuModuleLoadData failed for %s\n", p.c_str());
      g_pmod = nullptr;
      return nullptr;
    }
    fprintf(stderr, "[svc] patched cubin: %s\n", p.c_str());
  }
  CUfunction f = nullptr;
  if (g_pmod && cuModuleGetFunction(&f, g_pmod, sym.c_str()) != CUDA_SUCCESS) {
    fprintf(stderr, "[svc] missing kernel %s in patched cubin\n", sym.c_str());
    return nullptr;
  }
  return f;
}

struct PKCase { std::string variant; int k; std::string sym; int nargs; int threads; };

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "svc");
  std::uint32_t* d_kz = nullptr;
  std::uint64_t* d_out = nullptr;
  const int smsp = mb_query_device().sm_count;
  MB_CHECK(cudaMalloc(&d_kz, 4 * 512));
  MB_CHECK(cudaMalloc(&d_out, 8 * (args.reps + 2) * (smsp + 1)));
  MB_CHECK(cudaMemset(d_kz, 0, 4 * 512));
  std::uint32_t ctab_base = 0;
  {
    std::uint32_t* d_b = nullptr;
    MB_CHECK(cudaMalloc(&d_b, 4));
    k_cbase<<<1, 1>>>(d_b);
    MB_CHECK(cudaDeviceSynchronize());
    MB_CHECK(cudaMemcpy(&ctab_base, d_b, 4, cudaMemcpyDeviceToHost));
    cudaFree(d_b);
  }
  std::vector<std::uint32_t> tab(kTab);
  for (int i = 0; i < kTab; ++i) tab[i] = ctab_base + 4u * static_cast<std::uint32_t>((i * 37 + 1) % kTab);
  MB_CHECK(cudaMemcpyToSymbol(svc_ctab, tab.data(), sizeof(std::uint32_t) * kTab));

  std::vector<PCase> pc;
  std::vector<PKCase> pk;
  std::vector<UCase2> uc;
  {
#define AP(O)  add_poff<O>(pc, O::name, false, K3{}); add_poff<O>(pc, O::name, true, K3{});
  VOPS(AP) VOPS_X2(AP) OFFONLY(AP)
#undef AP
#define AA(O)  add_act<O>(pc, O::name, false, K3{}); add_act<O>(pc, O::name, true, K3{});
  VOPS(AA) VOPS_X2(AA)
#undef AA
  // mix: every R against FFMA/IMAD/IADD3/MUFU.RSQ, plus its own active form
#define MX(O)                                                              \
  add_mix<O, O_FFMA>(pc, K3{}); add_mix<O, O_IMAD>(pc, K3{});              \
  add_mix<O, O_IADD3>(pc, K3{}); add_mix<O, O_MUFURSQ>(pc, K3{});          \
  if constexpr (O::has_on) add_mix<O, O>(pc, K3{});
  VOPS(MX) VOPS_X2(MX) OFFONLY(MX)
#undef MX
  add_amix<O_MUFUEX2, O_MUFURSQ>(pc, K3{});
  add_amix<O_MUFUEX2, O_FFMA>(pc, K3{});
  add_amix<O_MUFUEX2, O_IMAD>(pc, K3{});
  // dedicated: SYNCS, tmem, umma
  for (int kk : {16, 32, 64}) {
    pc.push_back({"poff.SYNCS", kk, nullptr});
    pc.push_back({"act.SYNCS", kk, nullptr});
    pc.push_back({"poff.LDTM", kk, nullptr});
    pc.push_back({"poff.STTM", kk, nullptr});
    pc.push_back({"umma.UTCHMMA", kk, nullptr});
    pc.push_back({"act.LDTM", kk, nullptr});
    pc.push_back({"act.STTM", kk, nullptr});
    for (const char* pn : {"FFMA", "IMAD", "IADD3", "MUFU.RSQ"}) {
      pc.push_back({std::string("amix.LDTM+") + pn, kk, nullptr});
      pc.push_back({std::string("amix.STTM+") + pn, kk, nullptr});
    }
    pc.push_back({"umix.UMMA+MUFU", kk, nullptr});
    pc.push_back({"umix.UMMA+IMAD", kk, nullptr});
    pc.push_back({"usolo.MUFU.RSQ", kk, nullptr});
    pc.push_back({"usolo.IMAD", kk, nullptr});
  }
#define PKREG(O)                                                          \
  for (int kk : {16, 32, 64}) {                                                \
    pk.push_back({std::string("ppoff.") + O::name, kk, "pkp_" #O "_" + std::to_string(kk) + "_0", 3, 512}); \
    pk.push_back({std::string("ppoff0.") + O::name, kk, "pkp_" #O "_" + std::to_string(kk) + "_1", 3, 512}); \
  }
  PCHAIN(PKREG)
#undef PKREG
#define PKREGX(O)                                                         \
  for (int kk : {16, 32, 64}) {                                                \
    pk.push_back({std::string("ppoff.") + O::name, kk, "pkx_" #O "_" + std::to_string(kk) + "_0", 3, 512}); \
    pk.push_back({std::string("ppoff0.") + O::name, kk, "pkx_" #O "_" + std::to_string(kk) + "_1", 3, 512}); \
  }
  PBARE(PKREGX)
#undef PKREGX
#define PKREGU(O)                                                         \
  for (int kk : {16, 32, 64}) {                                                \
    pk.push_back({std::string("pupoff.") + O::name, kk, "pku_" #O "_" + std::to_string(kk) + "_0", 6, 512}); \
    pk.push_back({std::string("pupoff0.") + O::name, kk, "pku_" #O "_" + std::to_string(kk) + "_1", 6, 512}); \
  }
  PUNI(PKREGU)
#undef PKREGU
#define PKREGM(R, P)                                                           \
  for (int kk : {16, 32, 64})                                                  \
    pk.push_back({std::string("pmix.") + R::name + "+" + P::name, kk, "pkm_" #R "_" #P "_" + std::to_string(kk), 3, 512});
#define _PKREG_ALLP(R) PKREGM(R, O_FFMA) PKREGM(R, O_IMAD) PKREGM(R, O_SHF) PKREGM(R, O_MUFURSQ)
  PMXR(_PKREG_ALLP)
#undef _PKREG_ALLP
#undef PKREGM
#define PKREGMU(R, P)                                                          \
  for (int kk : {16, 32, 64})                                                  \
    pk.push_back({std::string("pumix.") + R::name + "+" + P::name, kk, "pkmu_" #R "_" #P "_" + std::to_string(kk), 6, 512});
#define _PKREGU_ALLP(R) PKREGMU(R, O_FFMA) PKREGMU(R, O_IMAD) PKREGMU(R, O_SHF) PKREGMU(R, O_MUFURSQ)
  PMXRU(_PKREGU_ALLP)
#undef _PKREGU_ALLP
#undef PKREGMU
  for (int kk : {16, 32, 64}) {
    pk.push_back({"ppoff.SYNCS", kk, "pk_syncs_" + std::to_string(kk), 3, 512});
    pk.push_back({"ppoff.LDTM", kk, "pk_tpoff_ldtm_" + std::to_string(kk), 3, 512});
    pk.push_back({"ppoff.STTM", kk, "pk_tpoff_sttm_" + std::to_string(kk), 3, 512});
    pk.push_back({"ppoff.UTCHMMA", kk, "pk_umma_" + std::to_string(kk), 3, 32});
  }

#define AUP(O) add_upoff<O>(uc, O::name, false, 0, K3{}); add_upoff<O>(uc, O::name, true, 0, K3{});
  UOPS(AUP)
#undef AUP
#define AUA(O) add_upoff<O>(uc, O::name, false, 1, K3{}); add_upoff<O>(uc, O::name, true, 1, K3{});
  UOPS(AUA)
#undef AUA
  // operand-form experiment: native self-form poff for the 8 validation
  // classes; patched ping-pong (pkq) for every chain target.  Run via
  // --forms (family 4) or --only filters.
#define FOP(O)  add_poff_self<O>(pc, O::name, K3{});
  FOP(O_FFMA) FOP(O_FADD) FOP(O_FMUL) FOP(O_IMAD) FOP(O_SHF) FOP(O_PRMT)
  FOP(O_POPC) FOP(O_MUFURSQ)
#undef FOP
#define FPQ(O)                                                            \
    for (int kk : {16, 32, 64})                                                \
      pk.push_back({std::string("ppoff2.") + O::name, kk, "pkq_" #O "_" + std::to_string(kk) + "_0", 3, 512});
  PCHAIN(FPQ)
#undef FPQ
#define UMX(O)                                                              \
  add_umix<O, O_FFMA>(uc, K3{}); add_umix<O, O_IMAD>(uc, K3{});             \
  add_umix<O, O_IADD3>(uc, K3{}); add_umix<O, O_MUFURSQ>(uc, K3{});         \
  if constexpr (!std::is_same_v<O, U_LDCU>) add_umix<O, O>(uc, K3{});
  UOPS(UMX)
#undef UMX
  }

  static const char* kForm8[] = {"FFMA", "FADD", "FMUL", "IMAD", "SHF", "PRMT", "POPC", "MUFU.RSQ"};
  auto form_ok = [&](const std::string& v) {
    if (!args.forms) return true;
    if (v.rfind("ppoff2.", 0) == 0) return true;   // patched pp, all chain targets
    for (const char* c : kForm8) {
      const std::string C = c;
      if (v == "poff." + C || v == "poffself." + C || v == "ppoff." + C) return true;
    }
    return false;
  };
  if (args.family == 3) {
    for (const auto& c : pk) if (form_ok(c.variant)) puts(c.variant.c_str());
    return 0;
  }
  const int reps = args.reps;
  auto run_rows = [&](const std::string& variant, int u, const char* ipk, auto&& launch, int blocks) {
    if (!mb_selected(args, variant)) return;
    fprintf(stderr, "[svc] %s u=%d\n", variant.c_str(), u);
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      launch();
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      if (blocks == 1) mb_collect(d_out, reps, 1, 0, s);
      else {
        // grid variant: max over CTAs per rep
        std::vector<std::uint64_t> h(static_cast<std::size_t>(reps) * blocks);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < reps; ++r) {
          std::uint64_t m = 0;
          for (int b = 0; b < blocks; ++b) m = std::max(m, h[static_cast<std::size_t>(r) * blocks + b]);
          s.push_back(static_cast<double>(m));
        }
      }
    }
    w.row(variant, u, std::string("\"ipk\":") + ipk, s);
  };

  for (const auto& c : pc) {
    const std::string& v = c.variant;
    if (args.family == 2) continue;
    if (!form_ok(v)) continue;
    if (v.rfind("poff.SYNCS", 0) == 0) {
      int kk = c.k;
      if (kk == 16) run_rows(v, kk, "32", [&] { k_syncs<16, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      else if (kk == 32) run_rows(v, kk, "32", [&] { k_syncs<32, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      else run_rows(v, kk, "32", [&] { k_syncs<64, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      continue;
    }
    if (v.rfind("act.SYNCS", 0) == 0) {
      int kk = c.k;
      if (kk == 16) run_rows(v, kk, "32", [&] { k_syncs<16, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      else if (kk == 32) run_rows(v, kk, "32", [&] { k_syncs<32, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      else run_rows(v, kk, "32", [&] { k_syncs<64, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      continue;
    }
#if SVC_SM100
    if (v.rfind("poff.LDTM", 0) == 0) {
      int kk = c.k;
      if (kk == 16) run_rows(v, kk, "32", [&] { k_tpoff<16, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      else if (kk == 32) run_rows(v, kk, "32", [&] { k_tpoff<32, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      else run_rows(v, kk, "32", [&] { k_tpoff<64, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      continue;
    }
    if (v.rfind("poff.STTM", 0) == 0) {
      int kk = c.k;
      if (kk == 16) run_rows(v, kk, "32", [&] { k_tpoff<16, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      else if (kk == 32) run_rows(v, kk, "32", [&] { k_tpoff<32, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      else run_rows(v, kk, "32", [&] { k_tpoff<64, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      continue;
    }
    if (v.rfind("act.LDTM", 0) == 0 || v.rfind("act.STTM", 0) == 0) {
      const int st = v[4] == 'S';
      int kk = c.k;
      if (st) {
        if (kk == 16) run_rows(v, kk, "32", [&] { k_tact<16, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
        else if (kk == 32) run_rows(v, kk, "32", [&] { k_tact<32, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
        else run_rows(v, kk, "32", [&] { k_tact<64, 1><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      } else {
        if (kk == 16) run_rows(v, kk, "32", [&] { k_tact<16, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
        else if (kk == 32) run_rows(v, kk, "32", [&] { k_tact<32, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
        else run_rows(v, kk, "32", [&] { k_tact<64, 0><<<1, 512>>>(d_out, d_kz, reps); }, 1);
      }
      continue;
    }
    if (v.rfind("amix.LDTM+", 0) == 0 || v.rfind("amix.STTM+", 0) == 0) {
      const int st = v[5] == 'S';
      const std::string pn = v.substr(10);
      int kk = c.k;
#define TAMIX(P)                                                              \
      if (st) { if (kk == 16) run_rows(v, kk, "64", [&] { mb_dyn_launch(&k_tamix<1, P, 16>, d_out, d_kz, reps); }, 1);          \
                else if (kk == 32) run_rows(v, kk, "64", [&] { mb_dyn_launch(&k_tamix<1, P, 32>, d_out, d_kz, reps); }, 1);   \
                else run_rows(v, kk, "64", [&] { mb_dyn_launch(&k_tamix<1, P, 64>, d_out, d_kz, reps); }, 1); }               \
      else { if (kk == 16) run_rows(v, kk, "64", [&] { mb_dyn_launch(&k_tamix<0, P, 16>, d_out, d_kz, reps); }, 1);           \
             else if (kk == 32) run_rows(v, kk, "64", [&] { mb_dyn_launch(&k_tamix<0, P, 32>, d_out, d_kz, reps); }, 1);      \
             else run_rows(v, kk, "64", [&] { mb_dyn_launch(&k_tamix<0, P, 64>, d_out, d_kz, reps); }, 1); }
      if (pn == "FFMA") TAMIX(O_FFMA)
      else if (pn == "IMAD") TAMIX(O_IMAD)
      else if (pn == "IADD3") TAMIX(O_IADD3)
      else TAMIX(O_MUFURSQ)
#undef TAMIX
      continue;
    }
    if (v.rfind("umix.UMMA+", 0) == 0 || v.rfind("usolo.", 0) == 0) {
      const int with_umma = v.rfind("umix.", 0) == 0;
      const bool mufu = v.find("MUFU") != std::string::npos;
      int kk = c.k;
#define UMIX(P, ON)                                                           \
      if (kk == 16) run_rows(v, kk, "32", [&] { mb_dyn_launch(&k_umix<P, 16, ON>, d_out, d_kz, reps); }, 1);   \
      else if (kk == 32) run_rows(v, kk, "32", [&] { mb_dyn_launch(&k_umix<P, 32, ON>, d_out, d_kz, reps); }, 1); \
      else run_rows(v, kk, "32", [&] { mb_dyn_launch(&k_umix<P, 64, ON>, d_out, d_kz, reps); }, 1);
      if (mufu) { if (with_umma) UMIX(O_MUFURSQ, 1) else UMIX(O_MUFURSQ, 0) }
      else { if (with_umma) UMIX(O_IMAD, 1) else UMIX(O_IMAD, 0) }
#undef UMIX
      continue;
    }
    if (v.rfind("umma.", 0) == 0) {
      int kk = c.k;
      if (kk == 16) run_rows(v, kk, "1", [&] { k_umma<16><<<1, 32>>>(d_out, d_kz, reps); }, 1);
      else if (kk == 32) run_rows(v, kk, "1", [&] { k_umma<32><<<1, 32>>>(d_out, d_kz, reps); }, 1);
      else run_rows(v, kk, "1", [&] { k_umma<64><<<1, 32>>>(d_out, d_kz, reps); }, 1);
      continue;
    }
#endif
    // dedicated variants whose launch path is compiled out on this
    // target (tcgen05 / UMMA cells on sm_90a) have no kernel: skip
    if (!c.fn) continue;
    const size_t smem = 128u * 1024u;
    cudaFuncSetAttribute((const void*)c.fn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    run_rows(v, c.k, "32", [&] { c.fn<<<1, 512, smem>>>(d_out, d_kz, reps); }, 1);
  }
  for (const auto& c : uc) {
    if (args.family == 2) continue;
    const bool ldcu = c.variant.find("LDCU") != std::string::npos;
    const std::uint32_t sx = ldcu ? ctab_base : 0x9e3779b9u;
    const std::uint32_t sy = ldcu ? ctab_base + 4u * 512u : 0x10203040u;
    const size_t smem = 128u * 1024u;
    cudaFuncSetAttribute((const void*)c.fn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    run_rows(c.variant, c.k, "32", [&] {
      c.fn<<<1, 512, smem>>>(d_out, d_kz, sx, sy, 0u, reps);
    }, 1);
  }
  // fences
  if (args.family != 2)
  {
    const struct { const char* name; } wl[5] = {{"1"}, {"4x1"}, {"4s0"}, {"16"}, {"grid"}};
#define FENCE_ONE(F, KK, WW, LANES, GRID)                                        \
    run_rows(std::string("fence.") + F::name + ".w" + wl[WW - 1].name, KK, "1", \
             [&] { k_fence<F, KK, WW><<<GRID, LANES>>>(d_out, reps); }, GRID)
#define FENCE_ALL(F)                                                            \
    FENCE_ONE(F, 8, 1, 32, 1);   FENCE_ONE(F, 16, 1, 32, 1);  FENCE_ONE(F, 32, 1, 32, 1); \
    FENCE_ONE(F, 8, 2, 128, 1);  FENCE_ONE(F, 16, 2, 128, 1); FENCE_ONE(F, 32, 2, 128, 1); \
    FENCE_ONE(F, 8, 3, 512, 1);  FENCE_ONE(F, 16, 3, 512, 1); FENCE_ONE(F, 32, 3, 512, 1); \
    FENCE_ONE(F, 8, 4, 512, 1);  FENCE_ONE(F, 16, 4, 512, 1); FENCE_ONE(F, 32, 4, 512, 1); \
    FENCE_ONE(F, 8, 5, 128, smsp); FENCE_ONE(F, 16, 5, 128, smsp); FENCE_ONE(F, 32, 5, 128, smsp);
    FENCE_ALL(F_CTA_ACQREL) FENCE_ALL(F_CTA_SC) FENCE_ALL(F_GPU_ACQREL) FENCE_ALL(F_GPU_SC)
    FENCE_ALL(F_SYS_ACQREL) FENCE_ALL(F_ASYNC_S) FENCE_ALL(F_ASYNC_G)
#undef FENCE_ALL
#undef FENCE_ONE
  }
  // guard-patched variants: launched from svc.patched.cubin via driver API
  if (args.family != 1)
  for (const auto& c : pk) {
    if (!mb_selected(args, c.variant) || !form_ok(c.variant)) continue;
    CUfunction f = pk_get(c.sym);
    if (!f) { w.row(c.variant, c.k, "\"ipk\":0", {}); continue; }
    run_rows(c.variant, c.k, "32", [&] {
      if (c.nargs == 3) {
        void* pargs[] = {&d_out, &d_kz, (void*)&reps};
        cuLaunchKernel(f, 1, 1, 1, c.threads, 1, 1, 0, 0, pargs, nullptr);
      } else {
        std::uint32_t sx = 0x9e3779b9u, sy = 0x10203040u, sk = 0;
        void* pargs[] = {&d_out, &d_kz, &sx, &sy, &sk, (void*)&reps};
        cuLaunchKernel(f, 1, 1, 1, c.threads, 1, 1, 0, 0, pargs, nullptr);
      }
    }, 1);
  }
  return w.failures ? 1 : 0;
}
