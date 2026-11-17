// tensor.umma.issue (timing kind 11) and the UMMA issue cadence
// (instruction.service UTCHMMA/UTCQMMA/UTCOMMA plan field 7), every
// umma_kind x cta_group x a_src cell, N in {64,128,256} (M=128 per CTA).
//
// A blocked waiter observes an mbarrier phase only at its try_wait poll
// grid (sync_xw: uniform ~55..225 cycles after the arrive), so a same-thread
// mma->commit->try_wait chain quantizes the MMA latency to that grid.  The
// latency is therefore measured cross-warp with a randomized phase:
//   xw  (rank 0) warp 0 lane 0: jittered delay ; t_s ; U x mma(accumulate) ;
//       commit [A] ;  warp 1 lane 0: test_wait spin on [A] ; t_w.
//       U = 0 is the commit-only baseline.
//   cad (rank 0) thread 0 chain { U x mma ; commit ; try_wait }  (cadence only)
// Engine (engine.cpp TensorOp::UmmaIssue/UmmaCommit): op complete = issue +
// kind 11; commit arrive = max(complete of committed ops) + kind 12; next MMA
// issue >= previous + plan field 7.  With E[.] the mean over >=1000 jittered
// samples and o_s(U) the stall cycles from t_s to the first MMA (U>0) or to
// the UTCBAR (U=0):
//   kind11 = (E[t_w - t_s | U=1] - E[... | U=0]) - (o_s(1) - o_s(0))
//   field7 = slope over U>=1 of E[t_w - t_s]   (front-end bound if it equals
//            the per-MMA issue stall cycles)
// D is read back after every launch and compared with K * #MMAs.
#pragma once

#include "mb.cuh"

#include <utility>

enum UKind { kF16 = 0, kTF32 = 1, kF8F6F4 = 2, kMXF8F6F4 = 3, kMXF4 = 4, kMXF4NVF4 = 5 };
enum UMode { kXw = 0, kCad = 2 };

template <int KIND>
struct KindInfo;
template <> struct KindInfo<kF16> { static constexpr int K = 16, FMT = 0, BS = 0, SCALE = 0; static constexpr std::uint32_t FILL = 0x3C003C00u, SF = 0; static constexpr const char* NAME = "f16"; };
template <> struct KindInfo<kTF32> { static constexpr int K = 8, FMT = 2, BS = 0, SCALE = 0; static constexpr std::uint32_t FILL = 0x3F800000u, SF = 0; static constexpr const char* NAME = "tf32"; };
template <> struct KindInfo<kF8F6F4> { static constexpr int K = 32, FMT = 0, BS = 0, SCALE = 0; static constexpr std::uint32_t FILL = 0x38383838u, SF = 0; static constexpr const char* NAME = "f8f6f4"; };
template <> struct KindInfo<kMXF8F6F4> { static constexpr int K = 32, FMT = 0, BS = 1, SCALE = 1; static constexpr std::uint32_t FILL = 0x38383838u, SF = 0x7F7F7F7Fu; static constexpr const char* NAME = "mxf8f6f4"; };
template <> struct KindInfo<kMXF4> { static constexpr int K = 64, FMT = 1, BS = 1, SCALE = 1; static constexpr std::uint32_t FILL = 0x22222222u, SF = 0x7F7F7F7Fu; static constexpr const char* NAME = "mxf4"; };
template <> struct KindInfo<kMXF4NVF4> { static constexpr int K = 64, FMT = 1, BS = 1, SCALE = 0; static constexpr std::uint32_t FILL = 0x22222222u, SF = 0x38383838u; static constexpr const char* NAME = "mxf4nvf4"; };

// Instruction descriptor (cute/arch/mma_sm100_desc.hpp InstrDescriptor /
// InstrDescriptorBlockScaled bit layout), K-major A and B, F32 accumulator.
template <int KIND>
__device__ __forceinline__ std::uint32_t make_idesc(int m, int n) {
  using I = KindInfo<KIND>;
  std::uint32_t d = 0;
  if constexpr (!I::BS) d |= 1u << 4;  // c_format F32
  d |= static_cast<std::uint32_t>(I::FMT) << 7;
  d |= static_cast<std::uint32_t>(I::FMT) << 10;
  d |= (static_cast<std::uint32_t>(n) >> 3) << 17;
  if constexpr (I::BS) d |= static_cast<std::uint32_t>(I::SCALE) << 23;
  d |= (static_cast<std::uint32_t>(m) >> 4) << 24;
  return d;
}

// Shared-memory matrix descriptor (SmemDescriptor, version 1).
// lay 0: SWIZZLE_NONE K-major, 8x16B core matrices, LBO = rows*16 (K step),
//        SBO = 128 (next 8-row group).  lay 1: SWIZZLE_128B K-major,
//        LBO = 16 (unused), SBO = 1024 (8 rows x 128 B).
__device__ __forceinline__ std::uint64_t make_sdesc(std::uint32_t saddr, int rows, int lay) {
  const std::uint64_t lbo = lay == 0 ? static_cast<std::uint64_t>(rows) * 16u : 16u;
  const std::uint64_t sbo = lay == 0 ? 128u : 1024u;
  const std::uint64_t layout = lay == 0 ? 0u : 2u;
  return ((static_cast<std::uint64_t>(saddr) & 0x3FFFFu) >> 4) | (((lbo >> 4) & 0x3FFFu) << 16) |
         (((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46) | (layout << 61);
}

#define UMMA_ZERO4 "{%5, %6, %7, %8}"
#define UMMA_ZERO8 "{%5, %6, %7, %8, %9, %10, %11, %12}"

template <int KIND, int CG, int ASRC, bool WS = false>
__device__ __forceinline__ void umma(std::uint32_t d, std::uint64_t a, std::uint32_t a_tmem, std::uint64_t b,
                                     std::uint32_t idesc, std::uint32_t sfa, std::uint32_t sfb, std::uint32_t en) {
  const std::uint32_t z = 0;
#define UMMA_PLAIN(CGS, KS, AOP, ZL, ...)                                                                   \
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"                                            \
               "tcgen05.mma.cta_group::" CGS ".kind::" KS " [%0], " AOP ", %2, %3, " ZL ", p;\n\t}\n" ::"r"(d), \
               __VA_ARGS__, "l"(b), "r"(idesc), "r"(en), "r"(z), "r"(z), "r"(z), "r"(z)                    \
                   CG2_ZEROS                                                                                \
               : "memory")
#define UMMA_BS(CGS, KS, AOP, ...)                                                                          \
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"                                            \
               "tcgen05.mma.cta_group::" CGS ".kind::" KS " [%0], " AOP ", %2, %3, [%5], [%6], p;\n\t}\n" ::"r"(d), \
               __VA_ARGS__, "l"(b), "r"(idesc), "r"(en), "r"(sfa), "r"(sfb)                                 \
               : "memory")
#define UMMA_WS(KS, AOP, ...)                                                                              \
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"                                            \
               "tcgen05.mma.ws.cta_group::1.kind::" KS " [%0], " AOP ", %2, %3, p;\n\t}\n"                \
               ::"r"(d), __VA_ARGS__, "l"(b), "r"(idesc), "r"(en)                                        \
               : "memory")
  if constexpr (CG == 1) {
#define CG2_ZEROS
    if constexpr (KIND == kF16) {
      if constexpr (WS) {
        if constexpr (ASRC == 0) UMMA_WS("f16", "%1", "l"(a));
        else UMMA_WS("f16", "[%1]", "r"(a_tmem));
      } else {
        if constexpr (ASRC == 0) UMMA_PLAIN("1", "f16", "%1", UMMA_ZERO4, "l"(a));
        else UMMA_PLAIN("1", "f16", "[%1]", UMMA_ZERO4, "r"(a_tmem));
      }
    }
    if constexpr (KIND == kTF32) {
      if constexpr (WS) {
        if constexpr (ASRC == 0) UMMA_WS("tf32", "%1", "l"(a));
        else UMMA_WS("tf32", "[%1]", "r"(a_tmem));
      } else if constexpr (ASRC == 0) {
        UMMA_PLAIN("1", "tf32", "%1", UMMA_ZERO4, "l"(a));
      } else {
        UMMA_PLAIN("1", "tf32", "[%1]", UMMA_ZERO4, "r"(a_tmem));
      }
    }
    if constexpr (KIND == kF8F6F4) {
      if constexpr (WS) {
        if constexpr (ASRC == 0) UMMA_WS("f8f6f4", "%1", "l"(a));
        else UMMA_WS("f8f6f4", "[%1]", "r"(a_tmem));
      } else if constexpr (ASRC == 0) {
        UMMA_PLAIN("1", "f8f6f4", "%1", UMMA_ZERO4, "l"(a));
      } else {
        UMMA_PLAIN("1", "f8f6f4", "[%1]", UMMA_ZERO4, "r"(a_tmem));
      }
    }
#undef CG2_ZEROS
    if constexpr (KIND == kMXF8F6F4) { if constexpr (ASRC == 0) UMMA_BS("1", "mxf8f6f4.block_scale", "%1", "l"(a)); else UMMA_BS("1", "mxf8f6f4.block_scale", "[%1]", "r"(a_tmem)); }
    if constexpr (KIND == kMXF4) { if constexpr (ASRC == 0) UMMA_BS("1", "mxf4.block_scale.block32", "%1", "l"(a)); else UMMA_BS("1", "mxf4.block_scale.block32", "[%1]", "r"(a_tmem)); }
    if constexpr (KIND == kMXF4NVF4) { if constexpr (ASRC == 0) UMMA_BS("1", "mxf4nvf4.block_scale.block16", "%1", "l"(a)); else UMMA_BS("1", "mxf4nvf4.block_scale.block16", "[%1]", "r"(a_tmem)); }
  } else {
#define CG2_ZEROS , "r"(z), "r"(z), "r"(z), "r"(z)
    if constexpr (KIND == kF16) { if constexpr (ASRC == 0) UMMA_PLAIN("2", "f16", "%1", UMMA_ZERO8, "l"(a)); else UMMA_PLAIN("2", "f16", "[%1]", UMMA_ZERO8, "r"(a_tmem)); }
    if constexpr (KIND == kTF32) { if constexpr (ASRC == 0) UMMA_PLAIN("2", "tf32", "%1", UMMA_ZERO8, "l"(a)); else UMMA_PLAIN("2", "tf32", "[%1]", UMMA_ZERO8, "r"(a_tmem)); }
    if constexpr (KIND == kF8F6F4) { if constexpr (ASRC == 0) UMMA_PLAIN("2", "f8f6f4", "%1", UMMA_ZERO8, "l"(a)); else UMMA_PLAIN("2", "f8f6f4", "[%1]", UMMA_ZERO8, "r"(a_tmem)); }
#undef CG2_ZEROS
    if constexpr (KIND == kMXF8F6F4) { if constexpr (ASRC == 0) UMMA_BS("2", "mxf8f6f4.block_scale", "%1", "l"(a)); else UMMA_BS("2", "mxf8f6f4.block_scale", "[%1]", "r"(a_tmem)); }
    if constexpr (KIND == kMXF4) { if constexpr (ASRC == 0) UMMA_BS("2", "mxf4.block_scale.block32", "%1", "l"(a)); else UMMA_BS("2", "mxf4.block_scale.block32", "[%1]", "r"(a_tmem)); }
    if constexpr (KIND == kMXF4NVF4) { if constexpr (ASRC == 0) UMMA_BS("2", "mxf4nvf4.block_scale.block16", "%1", "l"(a)); else UMMA_BS("2", "mxf4nvf4.block_scale.block16", "[%1]", "r"(a_tmem)); }
  }
#undef UMMA_PLAIN
#undef UMMA_BS
#undef UMMA_WS
}

template <int KIND, int CG, int ASRC, bool WS = false>
__device__ __forceinline__ void umma_sparse(
    std::uint32_t d, std::uint64_t a, std::uint32_t a_tmem,
    std::uint64_t b, std::uint32_t spmeta, std::uint32_t idesc,
    std::uint32_t sfa, std::uint32_t sfb, std::uint32_t en) {
  const std::uint32_t z = 0;
#define UMMA_PLAIN_SP1(KS, AOP, ...)                                                                \
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %5, 0;\n\t"                                  \
               "tcgen05.mma.sp.cta_group::1.kind::" KS " [%0], " AOP                              \
               ", %2, [%3], %4, {%6, %7, %8, %9}, p;\n\t}\n"                                     \
               ::"r"(d), __VA_ARGS__, "l"(b), "r"(spmeta), "r"(idesc), "r"(en),                    \
                 "r"(z), "r"(z), "r"(z), "r"(z) : "memory")
#define UMMA_PLAIN_SP2(KS, AOP, ...)                                                                \
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %5, 0;\n\t"                                  \
               "tcgen05.mma.sp.cta_group::2.kind::" KS " [%0], " AOP                              \
               ", %2, [%3], %4, {%6, %7, %8, %9, %10, %11, %12, %13}, p;\n\t}\n"                   \
               ::"r"(d), __VA_ARGS__, "l"(b), "r"(spmeta), "r"(idesc), "r"(en),                    \
                 "r"(z), "r"(z), "r"(z), "r"(z), "r"(z), "r"(z), "r"(z), "r"(z) : "memory")
#define UMMA_BS_SP(CGS, KS, AOP, ...)                                                               \
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %7, 0;\n\t"                                  \
               "tcgen05.mma.sp.cta_group::" CGS ".kind::" KS                                        \
               " [%0], " AOP ", %2, [%3], %4, [%5], [%6], p;\n\t}\n"                              \
               ::"r"(d), __VA_ARGS__, "l"(b), "r"(spmeta), "r"(idesc), "r"(sfa), "r"(sfb),       \
                 "r"(en) : "memory")
#define UMMA_WS_SP(KS, AOP, ...)                                                                    \
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %5, 0;\n\t"                                  \
               "tcgen05.mma.ws.sp.cta_group::1.kind::" KS " [%0], " AOP                           \
               ", %2, [%3], %4, p;\n\t}\n"                                                        \
               ::"r"(d), __VA_ARGS__, "l"(b), "r"(spmeta), "r"(idesc), "r"(en) : "memory")
  if constexpr (KIND == kF16) {
    if constexpr (WS) {
      if constexpr (ASRC == 0) UMMA_WS_SP("f16", "%1", "l"(a));
      else UMMA_WS_SP("f16", "[%1]", "r"(a_tmem));
    } else if constexpr (CG == 1) {
      if constexpr (ASRC == 0) UMMA_PLAIN_SP1("f16", "%1", "l"(a));
      else UMMA_PLAIN_SP1("f16", "[%1]", "r"(a_tmem));
    } else {
      if constexpr (ASRC == 0) UMMA_PLAIN_SP2("f16", "%1", "l"(a));
      else UMMA_PLAIN_SP2("f16", "[%1]", "r"(a_tmem));
    }
  } else if constexpr (KIND == kTF32) {
    if constexpr (WS) {
      if constexpr (ASRC == 0) UMMA_WS_SP("tf32", "%1", "l"(a));
      else UMMA_WS_SP("tf32", "[%1]", "r"(a_tmem));
    } else if constexpr (CG == 1) {
      if constexpr (ASRC == 0) UMMA_PLAIN_SP1("tf32", "%1", "l"(a));
      else UMMA_PLAIN_SP1("tf32", "[%1]", "r"(a_tmem));
    } else {
      if constexpr (ASRC == 0) UMMA_PLAIN_SP2("tf32", "%1", "l"(a));
      else UMMA_PLAIN_SP2("tf32", "[%1]", "r"(a_tmem));
    }
  } else if constexpr (KIND == kF8F6F4) {
    if constexpr (WS) {
      if constexpr (ASRC == 0) UMMA_WS_SP("f8f6f4", "%1", "l"(a));
      else UMMA_WS_SP("f8f6f4", "[%1]", "r"(a_tmem));
    } else if constexpr (CG == 1) {
      if constexpr (ASRC == 0) UMMA_PLAIN_SP1("f8f6f4", "%1", "l"(a));
      else UMMA_PLAIN_SP1("f8f6f4", "[%1]", "r"(a_tmem));
    } else {
      if constexpr (ASRC == 0) UMMA_PLAIN_SP2("f8f6f4", "%1", "l"(a));
      else UMMA_PLAIN_SP2("f8f6f4", "[%1]", "r"(a_tmem));
    }
  } else if constexpr (KIND == kMXF8F6F4) {
    if constexpr (CG == 1) {
      if constexpr (ASRC == 0) UMMA_BS_SP("1", "mxf8f6f4.block_scale", "%1", "l"(a));
      else UMMA_BS_SP("1", "mxf8f6f4.block_scale", "[%1]", "r"(a_tmem));
    } else {
      if constexpr (ASRC == 0) UMMA_BS_SP("2", "mxf8f6f4.block_scale", "%1", "l"(a));
      else UMMA_BS_SP("2", "mxf8f6f4.block_scale", "[%1]", "r"(a_tmem));
    }
  } else if constexpr (KIND == kMXF4) {
    if constexpr (CG == 1) {
      if constexpr (ASRC == 0) UMMA_BS_SP("1", "mxf4.block_scale.block32", "%1", "l"(a));
      else UMMA_BS_SP("1", "mxf4.block_scale.block32", "[%1]", "r"(a_tmem));
    } else {
      if constexpr (ASRC == 0) UMMA_BS_SP("2", "mxf4.block_scale.block32", "%1", "l"(a));
      else UMMA_BS_SP("2", "mxf4.block_scale.block32", "[%1]", "r"(a_tmem));
    }
  } else if constexpr (KIND == kMXF4NVF4) {
    if constexpr (CG == 1) {
      if constexpr (ASRC == 0) UMMA_BS_SP("1", "mxf4nvf4.block_scale.block16", "%1", "l"(a));
      else UMMA_BS_SP("1", "mxf4nvf4.block_scale.block16", "[%1]", "r"(a_tmem));
    } else {
      if constexpr (ASRC == 0) UMMA_BS_SP("2", "mxf4nvf4.block_scale.block16", "%1", "l"(a));
      else UMMA_BS_SP("2", "mxf4nvf4.block_scale.block16", "[%1]", "r"(a_tmem));
    }
  }
#undef UMMA_PLAIN_SP
#undef UMMA_PLAIN_SP1
#undef UMMA_PLAIN_SP2
#undef UMMA_BS_SP
#undef UMMA_WS_SP
}

template <int CG>
__device__ __forceinline__ void umma_commit(std::uint32_t mbar) {
  if constexpr (CG == 1)
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory");
  else
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(mbar),
        "h"(static_cast<std::uint16_t>(0x3))
        : "memory");
}

// A/B regions are 64 KiB each so runtime smem-control forms with large
// MN-major LBO (up to 16 KiB) stay inside the allocation; barrier after B.
constexpr int kSmemA = 0, kSmemB = 65536, kSmemBar = 131072, kSmemBytes = 131072 + 64 + 1024;
constexpr std::uint32_t kColA = 256, kColSfa = 320, kColSfb = 352;
constexpr std::uint32_t kXwDelay = 3000;

#ifndef MECH_UMMA_ROWS_PER_CTA
#define MECH_UMMA_ROWS_PER_CTA 128
#endif

// Signaler delay before t_s: base + a per-(launch, rep) hash in [0, 2048),
// i.e. many detector poll periods, so the completion phase relative to the
// waiter's poll grid is uniformly randomized and averages out of E[.].
__device__ __forceinline__ void xw_delay(int rep, std::uint32_t base, std::uint32_t seed) {
  const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
  const std::uint32_t d = base + ((h ^ (h >> 15)) & 2047u);
  const std::uint64_t t = mb_clock();
  while (mb_clock() - t < d) {
  }
}

// FORM (umma_forms probe; 0 = the measured baseline form, unchanged code):
// 1 BF16 a/b format, 2 B MN-major, 3 A MN-major, 4 BF16 + A/B MN-major,
// 5 SW128 descriptor LBO = 0, 6/7/8 descriptor start +32/+64/+96 B inside the atom.
template <int FORM>
__device__ __forceinline__ std::uint32_t form_idesc(std::uint32_t d) {
  if constexpr (FORM == 1) return d | (1u << 7) | (1u << 10);
  if constexpr (FORM == 2) return d | (1u << 16);
  if constexpr (FORM == 3) return d | (1u << 15);
  if constexpr (FORM == 4) return d | (1u << 7) | (1u << 10) | (1u << 15) | (1u << 16);
  return d;
}
template <int FORM>
__device__ __forceinline__ std::uint64_t form_sdesc(std::uint64_t d) {
  if constexpr (FORM == 5) return d & ~(0x3FFFull << 16);
  if constexpr (FORM >= 6 && FORM <= 8) return d + static_cast<std::uint64_t>((FORM - 5) * 2);  // +32 B per step (16 B units)
  return d;
}

// Runtime form overrides (umma_forms manifest tuples): the captured
// (idesc OR bits, smem A/B control word, start-offset class) replaces the
// probe baseline.  Zero fields = baseline form.
struct Fp {
  std::uint32_t idesc_or;      // OR'd into make_idesc (captured desc ^ base)
  std::uint64_t sdesc_a_ctrl;  // nonzero: replaces control bits (>=14) of sdesc A
  std::uint64_t sdesc_b_ctrl;
  std::uint32_t sdesc_a_start; // nonzero: start %1024 in bytes replaces start field
  std::uint32_t sdesc_b_start;
  std::uint32_t idesc_clear;   // bits cleared before idesc_or is applied
};

// only the low-14-bit start field may change the addressed range; reject
// out-of-atom starts at generation time (>=1024 never emitted by gen script).
__device__ __forceinline__ std::uint64_t apply_sdesc(std::uint64_t d,
                                                     std::uint64_t ctrl,
                                                     std::uint32_t start) {
  if (ctrl) d = (d & 0x3FFFull) | ctrl;
  if (start)
    d = (d & ~0x3Full) | (static_cast<std::uint64_t>(start / 16) & 0x3Full);
  return d;
}

template <int KIND, int CG, int ASRC, int MODE, int U, int FORM = 0, bool WS = false>
__global__ void __launch_bounds__(128, 1) k_umma(std::uint64_t* out, std::uint32_t* dval, int reps, int n, int lay,
                                                  std::uint32_t en_run, std::uint32_t seed, Fp fp = {}) {
  using I = KindInfo<KIND>;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t rank = CG == 2 ? mb_ctarank() : 0u;
  __shared__ std::uint32_t slot;
  // operands: every word = 1.0 in the kind's element format
  auto* words = reinterpret_cast<std::uint32_t*>(smem);
  for (int i = threadIdx.x; i < kSmemBar / 4; i += blockDim.x) words[i] = I::FILL;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const std::uint32_t mbar = mb_opaque(base + kSmemBar), ack = mb_opaque(base + kSmemBar + 8);
  if (threadIdx.x == 0) {
    mb_mbar_init(mbar, 1);
    mb_mbar_init(ack, 1);
    mb_fence_mbar_init();
  }
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  if (warp == 0) {
    mb_tmem_alloc<CG>(mb_smem_u32(&slot), 512);
    mb_tmem_relinquish<CG>();
  }
  mb_tc_fence_before();
  __syncthreads();
  mb_tc_fence_after();
  const std::uint32_t tbase = slot;
  // seed this warp's lane quarter: A (TS) and scale-factor columns
  {
    const std::uint32_t lane_base = tbase + (static_cast<std::uint32_t>(warp * 32) << 16);
    mb_tmem_fill(lane_base + kColA, 32, I::FILL);
    mb_tmem_fill(lane_base + kColSfa, 512 - kColSfa, I::BS ? I::SF : 0u);
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();

  const int rows_a = MECH_UMMA_ROWS_PER_CTA, rows_b = CG == 2 ? n / 2 : n;
  const std::uint64_t ad = apply_sdesc(form_sdesc<FORM>(make_sdesc(base + kSmemA, rows_a, lay)),
                                       fp.sdesc_a_ctrl, fp.sdesc_a_start);
  const std::uint64_t bd = apply_sdesc(form_sdesc<FORM>(make_sdesc(base + kSmemB, rows_b, lay)),
                                       fp.sdesc_b_ctrl, fp.sdesc_b_start);
  const std::uint32_t base_idesc = form_idesc<FORM>(
      make_idesc<KIND>(CG * MECH_UMMA_ROWS_PER_CTA, n));
  const std::uint32_t idesc = (base_idesc & ~fp.idesc_clear) | fp.idesc_or;
  const std::uint32_t d = tbase, atm = tbase + kColA, sfa = tbase + kColSfa, sfb = tbase + kColSfb;
  const std::uint32_t en_o = mb_opaque(en_run);
  std::uint32_t ph = 0;
  if (rank == 0 && lane == 0) {
    if constexpr (MODE == kCad) {
      if (warp == 0) {
#pragma unroll 1
        for (int rep = 0; rep < reps; ++rep) {
          // enable_input_d resolved before the clock read (no LDC/SEL in window)
          const std::uint32_t en_first = mb_opaque(rep == 0 ? 0u : en_o);
          const std::uint64_t t0 = mb_clock();
#pragma unroll
          for (int i = 0; i < U; ++i)
            umma<KIND, CG, ASRC, WS>(d, ad, atm, bd, idesc, sfa, sfb, i == 0 ? en_first : en_o);
          umma_commit<CG>(mbar);
          mb_mbar_wait_to(mbar, ph);
          ph ^= 1u;
          const std::uint64_t t1 = mb_clock();
          out[2 * rep] = 0;
          out[2 * rep + 1] = t1 - t0;
        }
      }
    } else {
      if (warp == 0) {
#pragma unroll 1
        for (int rep = 0; rep < reps; ++rep) {
          const std::uint32_t en_first = mb_opaque(rep == 0 ? 0u : en_o);
          xw_delay(rep, kXwDelay, seed);
          const std::uint64_t t_s = mb_clock();
#pragma unroll
          for (int i = 0; i < U; ++i)
            umma<KIND, CG, ASRC, WS>(d, ad, atm, bd, idesc, sfa, sfb, i == 0 ? en_first : en_o);
          umma_commit<CG>(mbar);
          out[2 * rep] = t_s;
          mb_mbar_wait_to(ack, ph);
          ph ^= 1u;
        }
      } else if (warp == 1) {
#pragma unroll 1
        for (int rep = 0; rep < reps; ++rep) {
          mb_mbar_spin_to(mbar, ph);
          const std::uint64_t t_w = mb_clock();
          out[2 * rep + 1] = t_w;
          ph ^= 1u;
          mb_mbar_arrive(ack);
        }
      }
    }
  }
  __syncwarp();
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();
  {
    std::uint32_t v;
    const std::uint32_t lane_base = tbase + (static_cast<std::uint32_t>(warp * 32) << 16);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(v) : "r"(lane_base) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (threadIdx.x == 0) dval[rank] = v;
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  if (warp == 0) mb_tmem_dealloc<CG>(tbase, 512);
}

using UKern = void (*)(std::uint64_t*, std::uint32_t*, int, int, int, std::uint32_t, std::uint32_t, Fp);
struct UCase {
  std::string kind, mode;
  int cg, asrc, u, kk;
  UKern k;
  int form = 0;  // umma_forms variant (k_umma FORM or forms_cases index); 0 = baseline
  Fp fp{};
  int n_filter = 0;
  bool ws = false;
};

template <int KIND, int CG, int ASRC, int MODE, bool WS = false, int... Us>
void uadd_rt(std::vector<UCase>& v, int form, const Fp& fp,
             std::integer_sequence<int, Us...>, int n_filter = 0) {
  static const char* modes[] = {"xw", "-", "cad"};
  (v.push_back({KindInfo<KIND>::NAME, modes[MODE], CG, ASRC, Us, KindInfo<KIND>::K,
                &k_umma<KIND, CG, ASRC, MODE, Us, 0, WS>, form, fp,
                n_filter, WS}), ...);
}

template <int KIND, int CG, int ASRC, int MODE, int... Us>
void uadd(std::vector<UCase>& v, std::integer_sequence<int, Us...>) {
  static const char* modes[] = {"xw", "-", "cad"};
  (v.push_back({KindInfo<KIND>::NAME, modes[MODE], CG, ASRC, Us, KindInfo<KIND>::K, &k_umma<KIND, CG, ASRC, MODE, Us>}), ...);
}

template <int KIND, int CG, int ASRC, int MODE, int FORM, int... Us>
void uadd_form(std::vector<UCase>& v, std::integer_sequence<int, Us...>) {
  static const char* modes[] = {"xw", "-", "cad"};
  (v.push_back({KindInfo<KIND>::NAME, modes[MODE], CG, ASRC, Us, KindInfo<KIND>::K,
                &k_umma<KIND, CG, ASRC, MODE, Us, FORM>, FORM}), ...);
}

template <int KIND, int CG, int ASRC>
void uadd_cell(std::vector<UCase>& v) {
  uadd<KIND, CG, ASRC, kXw>(v, std::integer_sequence<int, 1, 2, 4, 8, 16>{});
  uadd<KIND, CG, ASRC, kCad>(v, std::integer_sequence<int, 1, 2, 4, 8, 16>{});
}

inline int umma_main(int argc, char** argv, const char* probe, const std::vector<UCase>& cases) {
  std::uint32_t seed_base = 0;
  int forwarded = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--seed-base") != 0) {
      argv[forwarded++] = argv[i];
      continue;
    }
    if (++i == argc) {
      std::fprintf(stderr, "missing --seed-base value\n");
      return 2;
    }
    char* end = nullptr;
    const unsigned long value = std::strtoul(argv[i], &end, 10);
    if (argv[i][0] < '0' || argv[i][0] > '9' || *end || value > 0xfffffffful) {
      std::fprintf(stderr, "invalid --seed-base value\n");
      return 2;
    }
    seed_base = static_cast<std::uint32_t>(value);
  }
  argc = forwarded;
  argv[argc] = nullptr;
  const MbArgs args = mb_parse(argc, argv);
  if (seed_base > 0xffffffffu - static_cast<std::uint32_t>(args.launches - 1)) {
    std::fprintf(stderr, "--seed-base range overflows\n");
    return 2;
  }
  MbWriter w(args, probe);
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * args.reps));
  MB_CHECK(cudaMalloc(&d_val, sizeof(std::uint32_t) * 2));
  for (const auto& c : cases) {
    MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    for (int n : {64, 128, 256}) {
      if (c.n_filter && c.n_filter != n) continue;
      for (int lay : {1, 0}) {
        if (lay == 0 && !(c.kind == "f16" && c.asrc == 0)) continue;  // swizzle sensitivity: f16 SS only
        if (c.form != 0 && lay != 1) continue;  // forms are SW128 variants only
        const std::string variant = c.mode + "." + c.kind + ".cg" + std::to_string(c.cg) + "." +
                                    (c.asrc ? "tmem" : "smem") + ".n" + std::to_string(n) + (lay ? ".sw128" : ".none") +
                                    (c.ws ? ".ws" : "") +
                                    (c.form ? ".form" + std::to_string(c.form) : "");
        if (!mb_selected(args, variant)) continue;
        std::vector<double> samples;
        std::vector<float> dv;
        bool timed_out = false;
        for (int l = 0; l < args.launches; ++l) {
          cudaLaunchConfig_t cfg{};
          cfg.gridDim = dim3(c.cg, 1, 1);
          cfg.blockDim = dim3(128, 1, 1);
          cfg.dynamicSmemBytes = kSmemBytes;
          cudaLaunchAttribute attr[1];
          attr[0].id = cudaLaunchAttributeClusterDimension;
          attr[0].val.clusterDim.x = c.cg;
          attr[0].val.clusterDim.y = 1;
          attr[0].val.clusterDim.z = 1;
          cfg.attrs = attr;
          cfg.numAttrs = 1;
          MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, d_out, d_val, args.reps, n, lay, 1u, seed_base + static_cast<std::uint32_t>(l), c.fp));
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          if (mb_take_err()) {
            timed_out = true;
            break;
          }
          {
            std::vector<std::uint64_t> h(2 * static_cast<std::size_t>(args.reps));
            MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
            for (int r = 1; r < args.reps; ++r)
              samples.push_back(static_cast<double>(static_cast<std::int64_t>(h[2 * r + 1] - h[2 * r])));
          }
          std::uint32_t hv[2] = {0, 0};
          MB_CHECK(cudaMemcpy(hv, d_val, sizeof(hv), cudaMemcpyDeviceToHost));
          for (int r = 0; r < c.cg; ++r) {
            float f;
            std::memcpy(&f, &hv[r], 4);
            dv.push_back(f);
          }
        }
        if (timed_out) {
          w.fail(variant, c.u, "\"kind\":\"" + c.kind + "\"", "mbarrier wait timed out");
          continue;
        }
        const int mmas = args.reps * c.u;
        const double expect = static_cast<double>(c.kk) * mmas;
        bool ok = true;
        if (c.u > 0)
          for (float f : dv) ok = ok && f == static_cast<float>(expect);
        char extra[256];
        std::snprintf(extra, sizeof(extra), "\"d_expect\":%.1f,\"d_first\":%.1f,\"d_ok\":%s", expect,
                      dv.empty() ? -1.0 : static_cast<double>(dv[0]), ok ? "true" : "false");
        const std::string params = "\"kind\":\"" + c.kind + "\",\"mode\":\"" + c.mode + "\",\"cta_group\":" +
                                   std::to_string(c.cg) + ",\"a_src\":\"" + (c.asrc ? "tmem" : "smem") +
                                   "\",\"m\":" + std::to_string(c.cg * MECH_UMMA_ROWS_PER_CTA) + ",\"n\":" + std::to_string(n) +
                                   ",\"k\":" + std::to_string(c.kk) + ",\"layout\":\"" + (lay ? "sw128" : "none") +
                                   "\",\"form\":" + std::to_string(c.form) + ",\"operation\":\"" +
                                   (c.ws ? "WS" : "default") + "\",\"jitter_seed_base\":" + std::to_string(seed_base);
        w.row(variant, c.u, params, samples, extra);
      }
    }
  }
  return w.failures ? 1 : 0;
}
