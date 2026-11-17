#pragma once

#include "umma_impl.cuh"

#include <cuda_fp16.h>

__device__ __forceinline__ std::uint64_t mb_opaque64(std::uint64_t v) {
  asm("" : "+l"(v) :: "memory");
  return v;
}
constexpr int kFactorStages = 2;
constexpr int kFactorABytes = 52 * 1024;
constexpr int kFactorBBytes = 52 * 1024;
constexpr int kFactorStageBytes = kFactorABytes + kFactorBBytes;
constexpr int kFactorDataBytes = kFactorStages * kFactorStageBytes;
constexpr int kFactorBZeroBytes = 16 * 1024;
constexpr int kFactorBZeroOffset = kFactorDataBytes;
constexpr int kFactorMbar = kFactorDataBytes + kFactorBZeroBytes;
constexpr int kFactorSmemBytes =
    kFactorDataBytes + kFactorBZeroBytes + 64 + 1024;
constexpr std::uint32_t kFactorSpmeta = 384;

struct FactorForm {
  const char* form_id;
  const char* cell_id;
  const char* factor;
  const char* level;
  const char* kind;
  const char* a_type;
  const char* b_type;
  const char* d_type;
  const char* scale_type;
  const char* operation;
  const char* swizzle;
  const char* major_a;
  const char* major_b;
  int kind_id;
  int cta_group;
  int a_src;
  int m;
  int n;
  int k;
  std::uint32_t idesc;
  std::uint64_t smem_a_control;
  std::uint64_t smem_b_control;
  std::uint32_t start_a;
  std::uint32_t start_b;
  std::uint8_t a_bits;
  std::uint8_t b_bits;
  std::uint32_t a_one;
  std::uint32_t b_one;
  std::uint8_t scale_bits;
  std::uint32_t scale_one;
  std::uint8_t sf_id_a;
  std::uint8_t sf_id_b;
  std::uint8_t d_f16;
  std::int8_t sign;
  std::uint8_t reuse;
  std::uint8_t ws;
  std::uint8_t max_shift;
  std::uint8_t lbo_abs_a;
  std::uint8_t lbo_abs_b;
  std::uint8_t b_zero;
  std::uint8_t sparse;
  std::uint8_t sparsity_version;
  std::uint8_t overlap;
  std::uint32_t overlap_bytes;
  int acc;
  int arot;
  int brot;
  int grp;
  int issuers;
  int kalu;
  int elected;
};

struct FactorParams {
  std::uint32_t idesc;
  std::uint64_t smem_a_control;
  std::uint64_t smem_b_control;
  std::uint32_t start_a;
  std::uint32_t start_b;
  std::uint32_t a_one;
  std::uint32_t b_one;
  std::uint32_t scale_one;
  std::uint8_t a_bits;
  std::uint8_t b_bits;
  std::uint8_t scale_bits;
  std::uint8_t sf_id_a;
  std::uint8_t sf_id_b;
  std::uint8_t d_f16;
  std::int8_t sign;
  std::uint8_t reuse;
  std::uint8_t lbo_abs_a;
  std::uint8_t lbo_abs_b;
  std::uint8_t b_zero;
  std::uint8_t sparse;
  std::uint8_t sparsity_version;
  std::uint8_t overlap;
  std::uint32_t overlap_bytes;
};

constexpr int kFactorTelemetryWords = 6;
using FactorKernel = void (*)(std::uint64_t*, std::uint32_t*, int, int,
                              std::uint32_t, FactorParams);

struct FactorCase {
  const FactorForm* form;
  const char* mode;
  int u;
  FactorKernel kernel;
  FactorKernel check_kernel;
  int k;
};

__device__ __forceinline__ std::uint32_t factor_pattern_word(
    std::uint32_t one, std::uint32_t bits, int word_in_128b) {
  std::uint32_t result = 0;
  const int base_bit = word_in_128b * 32;
  for (int bit = 0; bit < 32; ++bit) {
    if ((one >> ((base_bit + bit) % bits)) & 1u) {
      result |= 1u << bit;
    }
  }
  return result;
}

__device__ __forceinline__ std::uint64_t factor_sdesc(
    std::uint32_t addr, int rows, std::uint64_t control,
    std::uint32_t start) {
  const std::uint64_t d = (make_sdesc(addr, rows, 1) & 0x3FFFull) | control;
  return (d & ~0x3FFFull) |
         ((static_cast<std::uint64_t>(
               ((d & 0x3FFFull) + (start >> 4)) & 0x3FFFull)));
}

template <int KIND, int CG, int ASRC, int MODE, int U, int M_ROWS,
          bool WS = false, bool REUSE = false, bool SPARSE = false,
          int ACC = 1, int AROT = 0, int BROT = 0, int GRP = 0,
          int ISSUERS = 1, int KALU = 0, int ELECTED = 0>
__global__ void __launch_bounds__(128, 1) k_factor(
    std::uint64_t* out, std::uint32_t* dval, int reps, int n,
    std::uint32_t seed, FactorParams f) {
  using I = KindInfo<KIND>;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t aligned = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (aligned - raw);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t rank = CG == 2 ? mb_ctarank() : 0u;
  __shared__ std::uint32_t slot;
  __shared__ std::uint32_t idesc_shared;
  if (threadIdx.x == 0) idesc_shared = f.idesc;
  __syncthreads();
  const std::uint32_t idesc = mb_opaque(idesc_shared);
  __syncthreads();
  auto* words = reinterpret_cast<std::uint32_t*>(smem);
  constexpr int words_per_stage = kFactorStageBytes / 4;
  for (int i = threadIdx.x;
       i < kFactorDataBytes / 4; i += blockDim.x) {
    const int stage = i / words_per_stage;
    const int in_stage = i % words_per_stage;
    const bool is_a = in_stage < kFactorABytes / 4;
    const int word_in_region = is_a ? in_stage : in_stage - kFactorABytes / 4;
    const std::uint32_t one = is_a ? f.a_one : f.b_one;
    const std::uint32_t bits = is_a ? f.a_bits : f.b_bits;
    (void)stage;
    words[i] = factor_pattern_word(one, bits, word_in_region & 3);
  }
  for (int i = threadIdx.x; i < kFactorBZeroBytes / 4; i += blockDim.x)
    words[kFactorBZeroOffset / 4 + i] = 0;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const std::uint32_t mbar = mb_opaque(aligned + kFactorMbar);
  const std::uint32_t ack = mb_opaque(aligned + kFactorMbar + 8);
  if (threadIdx.x == 0) {
    mb_mbar_init(mbar, 1);
    mb_mbar_init(ack, 1);
    if (GRP) mb_mbar_init(mbar + 16, 1);
    if (ISSUERS > 1)
      for (int w = 1; w < ISSUERS; ++w)
        mb_mbar_init(mbar + 32 * w, 1);
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
  constexpr std::uint32_t sfa_col = kColSfa;
  constexpr std::uint32_t sfb_col = kColSfb;
  if constexpr (KIND == kMXF8F6F4 || KIND == kMXF4 || KIND == kMXF4NVF4) {
    const std::uint32_t lane_base =
        tbase + (static_cast<std::uint32_t>(warp * 32) << 16);
    const std::uint32_t sf = factor_pattern_word(
        f.scale_one, f.scale_bits, 0);
    for (std::uint32_t sf_id = 0; sf_id < 4; ++sf_id) {
      const std::uint32_t sf_base =
          ((lane_base + kColSfa) & 0x3fffffffu) | (sf_id << 30);
      mb_tmem_fill(
          sf_base, (SPARSE ? kFactorSpmeta : 512) - kColSfa, sf);
    }
  }
  if constexpr (ASRC == 1) {
    const std::uint32_t lane_base =
        tbase + (static_cast<std::uint32_t>(warp * 32) << 16);
    const std::uint32_t ta = factor_pattern_word(
        f.a_one, f.a_bits, 0);
    constexpr int kACols = (I::K * (KIND == kTF32 ? 4 : KIND == kF16 ? 2 : 1)) / 4;
    mb_tmem_fill(lane_base + kColA, AROT ? ACC * kACols : 32, ta);
  }
  if constexpr (SPARSE) {
    const std::uint32_t lane_base =
        tbase + (static_cast<std::uint32_t>(warp * 32) << 16);
    mb_tmem_fill(lane_base + kFactorSpmeta, 128, 0);
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();

  const int rows_a = M_ROWS;
  const int rows_b = CG == 2 ? n / 2 : n;
  std::uint64_t ad[kFactorStages]{};
  std::uint64_t bd[kFactorStages]{};
#pragma unroll
  for (int stage = 0; stage < kFactorStages; ++stage) {
    const int descriptor_stage = REUSE ? 0 : stage;
    const std::uint32_t base =
        aligned + descriptor_stage * kFactorStageBytes;
    ad[stage] = factor_sdesc(
        base, rows_a, f.smem_a_control, f.start_a);
    const std::uint32_t bbase = f.b_zero
        ? aligned + kFactorBZeroOffset
        : f.overlap == 2 ? base
        : f.overlap == 1 ? base + f.overlap_bytes
        : base + kFactorABytes;
    bd[stage] = factor_sdesc(
        bbase, rows_b, f.smem_b_control, f.start_b);
    if (f.lbo_abs_a) {
      const std::uint32_t next =
          aligned + ((descriptor_stage + 1) % kFactorStages) * kFactorStageBytes;
      const std::uint32_t next_start =
          ((next + f.start_a) >> 4) & 0x3fffu;
      ad[stage] = (ad[stage] & ~(0x3fffull << 16)) |
                  (static_cast<std::uint64_t>(next_start) << 16);
    }
    if (f.lbo_abs_b) {
      const std::uint32_t next = f.b_zero
          ? aligned + kFactorBZeroOffset
          : aligned + ((descriptor_stage + 1) % kFactorStages) *
                kFactorStageBytes + kFactorABytes;
      const std::uint32_t next_start =
          ((next + f.start_b) >> 4) & 0x3fffu;
      bd[stage] = (bd[stage] & ~(0x3fffull << 16)) |
                  (static_cast<std::uint64_t>(next_start) << 16);
    }
  }
  // Operand rotation: AROT/BROT give each of the ACC operand slots its own
  // sub-tile inside the two stage regions (4 per stage, 8 KB apart).
  std::uint64_t ad_rot[ACC]{};
  std::uint64_t bd_rot[ACC]{};
  if constexpr (MODE == kCad && (AROT || BROT)) {
    constexpr int kRotSlots = ACC < 1 ? 1 : ACC;
    const std::uint64_t sa_ctl = mb_opaque64(f.smem_a_control);
    const std::uint64_t sb_ctl = mb_opaque64(f.smem_b_control);
    const std::uint32_t sa0 = mb_opaque(f.start_a);
    const std::uint32_t sb0 = mb_opaque(f.start_b);
#pragma unroll
    for (int slot = 0; slot < kRotSlots; ++slot) {
      const std::uint32_t base =
          aligned + (REUSE ? 0 : (slot & (kFactorStages - 1))) * kFactorStageBytes;
      if constexpr (AROT) {
        if (ASRC == 0)
          ad_rot[slot] = mb_opaque64(factor_sdesc(
              base + (slot / kFactorStages) * 8192,
              rows_a, sa_ctl, sa0));
      }
      if constexpr (BROT) {
        const std::uint32_t bbase =
            base + kFactorABytes + (slot / kFactorStages) * 8192;
        bd_rot[slot] = mb_opaque64(factor_sdesc(
            bbase, rows_b, sb_ctl, sb0));
      }
    }
  }
  constexpr int kACols = (I::K * (KIND == kTF32 ? 4 : KIND == kF16 ? 2 : 1)) / 4;
  const std::uint32_t sfa =
      ((tbase + sfa_col) & 0x3fffffffu) |
      (static_cast<std::uint32_t>(f.sf_id_a) << 30);
  const std::uint32_t sfb =
      ((tbase + sfb_col) & 0x3fffffffu) |
      (static_cast<std::uint32_t>(f.sf_id_b) << 30);

  // ACC > 1: the i-th UMMA of the chain writes accumulator
  // tbase + (i % ACC) * n fp32 columns (TMEM allocation is 512 cols, so
  // ACC * N must stay <= 512 — enforced by the form generator).
  const std::uint32_t atm = tbase + kColA;
  const std::uint32_t spmeta = tbase + kFactorSpmeta;
  std::uint32_t ph = 0;
  {
    if constexpr (MODE == kCad) {
      __shared__ std::uint64_t sh_t0[4], sh_t1[4];
      // ELECTED: whole issuer warps enter the outer (warp-uniform) branch
      // and elect.sync picks the issuing lane inside the converged region;
      // folding `warp < ISSUERS` into the single-lane predicate makes ptxas
      // wrap every UMMA in a wave-serialisation loop.
      const bool issuer_warp = rank == 0 && warp < ISSUERS;
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        __syncthreads();
        if (issuer_warp) {
          if (ELECTED ? mb_elect_one() != 0 : lane == 0) {
          const std::uint32_t en_first = mb_opaque(rep == 0 ? 0u : 1u);
          const std::uint64_t t0 = mb_clock();
#pragma unroll
          for (int i = 0; i < U; ++i) {
            const int stage = REUSE ? 0 : (i & (kFactorStages - 1));
            const std::uint32_t d = tbase +
                (static_cast<std::uint32_t>(warp * ACC + i % ACC) *
                 static_cast<std::uint32_t>(n));
            const std::uint32_t atm_i = (AROT && ASRC == 1)
                ? atm + (static_cast<std::uint32_t>(warp * ACC + i % ACC) * kACols)
                : atm;
            const std::uint64_t ad_i = (AROT && ASRC == 0) ? ad_rot[i % ACC] : ad[stage];
            const std::uint64_t bd_i = BROT ? bd_rot[i % ACC] : bd[stage];
            if constexpr (SPARSE)
              umma_sparse<KIND, CG, ASRC, WS>(
                  d, ad_i, atm_i, bd_i, spmeta, idesc, sfa, sfb,
                  i == 0 ? en_first : 1u);
            else
              umma<KIND, CG, ASRC, WS>(
                  d, ad_i, atm_i, bd_i, idesc, sfa, sfb,
                  i == 0 ? en_first : 1u);
            if constexpr (GRP)
              if ((i & 7) == 7)
                umma_commit<CG>(mbar + 16);
            if constexpr (KALU) {
              if (i + 1 < U) {
                std::uint32_t kscratch;
#pragma unroll
                for (int k = 0; k < KALU; ++k)
                  asm volatile("add.s32 %0, %1, %2;"
                               : "=r"(kscratch) : "r"(i), "r"(k));
              }
            }
          }
          umma_commit<CG>(mbar + (ISSUERS > 1 ? 32 * warp : 0));
          mb_mbar_wait_to(mbar + (ISSUERS > 1 ? 32 * warp : 0), ph);
          ph ^= 1u;
          const std::uint64_t t1 = mb_clock();
          if constexpr (ISSUERS > 1) {
            sh_t0[warp] = t0;
            sh_t1[warp] = t1;
          } else {
            out[2 * rep] = 0;
            out[2 * rep + 1] = t1 - t0;
          }
          }
        }
        __syncthreads();
        if (ISSUERS > 1 && warp == 0 && lane == 0) {
          std::uint64_t tmin = sh_t0[0], tmax = sh_t1[0];
#pragma unroll
          for (int w = 1; w < ISSUERS; ++w) {
            tmin = tmin < sh_t0[w] ? tmin : sh_t0[w];
            tmax = tmax > sh_t1[w] ? tmax : sh_t1[w];
          }
          out[2 * rep] = 0;
          out[2 * rep + 1] = tmax - tmin;
        }
      }
    } else if (rank == 0 && warp == 0 &&
               (ELECTED ? mb_elect_one() != 0 : lane == 0)) {
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        const std::uint32_t en_first = mb_opaque(rep == 0 ? 0u : 1u);
        xw_delay(rep, kXwDelay, seed);
        const std::uint64_t t_s = mb_clock();
#pragma unroll
        for (int i = 0; i < U; ++i) {
          const int stage = REUSE ? 0 : (i & (kFactorStages - 1));
          const std::uint32_t d = tbase + (static_cast<std::uint32_t>(i % ACC) * static_cast<std::uint32_t>(n));
          const std::uint32_t atm_i = (AROT && ASRC == 1 && MODE == kCad)
              ? atm + (static_cast<std::uint32_t>(i % ACC) * kACols)
              : atm;
          const std::uint64_t ad_i = (AROT && ASRC == 0 && MODE == kCad) ? ad_rot[i % ACC] : ad[stage];
          const std::uint64_t bd_i = (BROT && MODE == kCad) ? bd_rot[i % ACC] : bd[stage];
          if constexpr (SPARSE)
            umma_sparse<KIND, CG, ASRC, WS>(
                d, ad_i, atm_i, bd_i, spmeta, idesc, sfa, sfb,
                i == 0 ? en_first : 1u);
          else
            umma<KIND, CG, ASRC, WS>(
                d, ad_i, atm_i, bd_i, idesc, sfa, sfb,
                i == 0 ? en_first : 1u);
        }
        umma_commit<CG>(mbar);
        out[2 * rep] = t_s;
        mb_mbar_wait_to(ack, ph);
        ph ^= 1u;
      }
    } else if (rank == 0 && lane == 0 && warp == 1) {
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        mb_mbar_spin_to(mbar, ph);
        out[2 * rep + 1] = mb_clock();
        ph ^= 1u;
        mb_mbar_arrive(ack);
      }
    }
  }
  __syncwarp();
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();
  {
    std::uint32_t value;
    const std::uint32_t lane_base =
        tbase + (static_cast<std::uint32_t>(warp * 32) << 16);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=r"(value) : "r"(lane_base) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (threadIdx.x == 0) {
      const int base = static_cast<int>(rank) * kFactorTelemetryWords;
      dval[base] = value;
      dval[base + 1] = idesc;
      const std::uint64_t ad0 = ASRC == 0 ? ad[0] : 0;
      dval[base + 2] = static_cast<std::uint32_t>(ad0);
      dval[base + 3] = static_cast<std::uint32_t>(ad0 >> 32);
      dval[base + 4] = static_cast<std::uint32_t>(bd[0]);
      dval[base + 5] = static_cast<std::uint32_t>(bd[0] >> 32);
    }
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  if (warp == 0) mb_tmem_dealloc<CG>(tbase, 512);
}
