// instruction.service, batch 3.
//
//   lat OP U    dependent chains (alu.cu form): 64-bit shift-add (LEA +
//               LEA.HI.X per step; the LEA -> LEA dependency is the critical
//               path), uniform shift-add (ULEA).  (VIADD chains of immediates
//               fold, and VIADD+LOP3 pairs cross pipes at 5 cycles per hop,
//               which does not split; not timed.)
//   shot OP     one instance between two clock reads, followed by the
//               consumer the hardware must wait for:
//                 membar_cta  fence.acq_rel.cta      MEMBAR.ALL.CTA
//                 membar_gpu  fence.sc.gpu           MEMBAR.SC.GPU (+ERRBAR, CGAERRBAR, CCTL.IVALL)
//                 s2r         %warpid -> S2R SR_VIRTID + consumers
//                 s2ur        %smid   -> S2UR + UIADD3
//                 fence_acqrel_gpu / fence_sc_cta / fence_acqrel_sys, and
//                 stg_fence_*: one st.global pending at the fence (store
//                 visibility wait)
//               extra = E[t1 - t0] - (sum of SASS stall counts in the
//               window): the cycles the hardware adds beyond the control
//               words (a blocking fence, a scoreboard wait).
//   thr OP K    predicated-off throughput (FFMA / IMAD / MUFU.RSQ with a
//               runtime-false predicate), alu.cu k_thr form with pinned
//               chains: cycles per predicated-off warp instruction per SMSP.
#include "../mb.cuh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

constexpr int kTab = 1024;
__constant__ std::uint32_t ctab[kTab];

struct VIADD_LOP3 {
  static constexpr const char* name = "VIADD_LOP3";
  __device__ __forceinline__ static void op(std::uint32_t& d, std::uint32_t a, std::uint32_t) {
    asm volatile("{\n\t.reg .u32 t;\n\tadd.u32 t, %1, 0x12345;\n\txor.b32 %0, t, 0x5a5a5a5a;\n\t}" : "=r"(d) : "r"(a));
  }
};
struct LOP3_IMM {
  static constexpr const char* name = "LOP3_IMM";
  __device__ __forceinline__ static void op(std::uint32_t& d, std::uint32_t a, std::uint32_t) {
    asm volatile("xor.b32 %0, %1, 0x5a5a5a5a;" : "=r"(d) : "r"(a));
  }
};
struct ULEA {
  static constexpr const char* name = "ULEA";
  __device__ __forceinline__ static void op(std::uint32_t& d, std::uint32_t a, std::uint32_t b) {
    asm volatile("{\n\t.reg .u32 t;\n\tshl.b32 t, %1, 5;\n\tadd.u32 %0, t, %2;\n\t}" : "=r"(d) : "r"(a), "r"(b));
  }
};

template <class O, int U>
__global__ void __launch_bounds__(32, 1) k_vlat(std::uint64_t* out, const std::uint32_t* seeds, int reps) {
  std::uint32_t x = seeds[threadIdx.x], y = seeds[32 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if (i & 1) O::op(y, x, y);
      else O::op(x, y, x);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) {
      out[rep] = t1 - t0;
      out[reps + 1 + rep] = x ^ y;
    }
  }
}

template <class O, int U>
__global__ void __launch_bounds__(32, 1) k_ulat(std::uint64_t* out, std::uint32_t sx, std::uint32_t sy, int reps) {
  std::uint32_t x = sx, y = sy;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if (i & 1) O::op(y, x, y);
      else O::op(x, y, x);
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) {
      out[rep] = t1 - t0;
      out[reps + 1 + rep] = ctab[(x ^ y) & (kTab - 1)];
    }
  }
}

// 64-bit x = (x << 4) + y: LEA (low) + LEA.HI.X (high) per step
template <int U>
__global__ void __launch_bounds__(32, 1) k_lea64(std::uint64_t* out, const std::uint64_t* seeds, int reps) {
  std::uint64_t x = seeds[threadIdx.x], y = seeds[32 + threadIdx.x];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if (i & 1) asm volatile("{\n\t.reg .u64 t;\n\tshl.b64 t, %1, 4;\n\tadd.u64 %0, t, %2;\n\t}" : "=l"(y) : "l"(x), "l"(y));
      else asm volatile("{\n\t.reg .u64 t;\n\tshl.b64 t, %1, 4;\n\tadd.u64 %0, t, %2;\n\t}" : "=l"(x) : "l"(y), "l"(x));
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) {
      out[rep] = t1 - t0;
      out[reps + 1 + rep] = x ^ y;
    }
  }
}

// single shots
template <int OP>
__global__ void __launch_bounds__(32, 1) k_shot(std::uint64_t* out, int reps, std::uint32_t* gbuf) {
  __shared__ std::uint32_t sbuf[64];
  __shared__ std::uint64_t sres[64];
  std::uint32_t acc = threadIdx.x;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncwarp();
    const std::uint64_t t0 = mb_clock();
    if constexpr (OP == 0) {
      asm volatile("fence.acq_rel.cta;" ::: "memory");
    } else if constexpr (OP == 1) {
      asm volatile("fence.sc.gpu;" ::: "memory");
    } else if constexpr (OP == 2) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(mb_smem_u32(&sbuf[threadIdx.x])), "r"(acc) : "memory");
    } else if constexpr (OP == 3) {
      std::uint32_t v;
      asm volatile("mov.u32 %0, %%warpid;" : "=r"(v));
      acc += v;
    } else if constexpr (OP == 5) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    } else if constexpr (OP == 6) {
      asm volatile("fence.sc.cta;" ::: "memory");
    } else if constexpr (OP == 7) {
      asm volatile("fence.acq_rel.sys;" ::: "memory");
    } else if constexpr (OP == 8) {
      // one global store pending at the fence
      asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf + threadIdx.x), "r"(acc) : "memory");
      asm volatile("fence.acq_rel.cta;" ::: "memory");
    } else if constexpr (OP == 9) {
      asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf + threadIdx.x), "r"(acc) : "memory");
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    } else if constexpr (OP == 10) {
      asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf + threadIdx.x), "r"(acc) : "memory");
      asm volatile("fence.sc.gpu;" ::: "memory");
    } else if constexpr (OP == 11) {
      // CTA barrier alone (one-warp CTA) and with a pending global store:
      // does bar.sync wait for the store's L2 acknowledgment?
      asm volatile("bar.sync 0;" ::: "memory");
    } else if constexpr (OP == 12) {
      asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf + threadIdx.x), "r"(acc) : "memory");
      asm volatile("bar.sync 0;" ::: "memory");
    } else if constexpr (OP == 13) {
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(mb_smem_u32(&sbuf[threadIdx.x])), "r"(acc) : "memory");
      asm volatile("bar.sync 0;" ::: "memory");
    } else if constexpr (OP == 19) {
      // async-proxy fences (FENCE.VIEW.ASYNC.*): alone and after a pending
      // st.shared / st.global (generic-proxy writes made visible to the
      // async proxy, as before a TMA store or UMMA reading smem)
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    } else if constexpr (OP == 20) {
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(mb_smem_u32(&sbuf[threadIdx.x])), "r"(acc) : "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    } else if constexpr (OP == 21) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
    } else if constexpr (OP == 22) {
      asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf + threadIdx.x), "r"(acc) : "memory");
      asm volatile("fence.proxy.async.global;" ::: "memory");
    } else if constexpr (OP == 23) {
      asm volatile("fence.proxy.async;" ::: "memory");
    } else if constexpr (OP >= 14 && OP <= 18) {
      // store acknowledgment vs store shape, seen through fence.sc.gpu:
      // 14 one lane 4 B, 18 one lane 8 B, 15 32 lanes x 16 B (one 512 B
      // span, 4 lines), 16 32 lanes x 4 B strided 128 B (32 lines), 17 two
      // 128 B warp stores to two lines
      if constexpr (OP == 14) {
        if (threadIdx.x == 0) asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf), "r"(acc) : "memory");
      } else if constexpr (OP == 18) {
        if (threadIdx.x == 0)
          asm volatile("st.global.u64 [%0], %1;" ::"l"(gbuf), "l"(static_cast<std::uint64_t>(acc)) : "memory");
      } else if constexpr (OP == 15) {
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};" ::"l"(gbuf + 4 * threadIdx.x), "r"(acc) : "memory");
      } else if constexpr (OP == 16) {
        asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf + 32 * threadIdx.x), "r"(acc) : "memory");
      } else {
        asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf + threadIdx.x), "r"(acc) : "memory");
        asm volatile("st.global.u32 [%0], %1;" ::"l"(gbuf + 512 + threadIdx.x), "r"(acc) : "memory");
      }
      asm volatile("fence.sc.gpu;" ::: "memory");
    } else {
      std::uint32_t v;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
      acc += v;
    }
    const std::uint64_t t1 = mb_clock();
    // results go to shared memory inside the loop: a global store here
    // would still be pending (unacknowledged) at the next rep's fence
    if (threadIdx.x == 0) sres[rep & 63] = t1 - t0;
  }
  __syncwarp();
  if (threadIdx.x == 0) {
    for (int rep = 0; rep < reps && rep < 64; ++rep) out[rep] = sres[rep];
    out[reps + 1] = acc;
  }
}

// predicated-off throughput (runtime-false predicate from memory)
template <class T>
__device__ __forceinline__ T vload(std::uint32_t a) {
  T v;
  if constexpr (std::is_same_v<T, float>) asm volatile("ld.volatile.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  else asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
template <class T>
__device__ __forceinline__ void vstore(std::uint32_t a, T v) {
  if constexpr (std::is_same_v<T, float>) asm volatile("st.volatile.shared.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
  else asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

struct FFMA_OFF {
  using T = float;
  static constexpr const char* name = "FFMA_OFF";
  __device__ __forceinline__ static void op(T& x, T y, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t@p fma.rn.f32 %0, %0, %1, %0;\n\t}" : "+f"(x) : "f"(y), "r"(k));
  }
};
struct IMAD_OFF {
  using T = std::uint32_t;
  static constexpr const char* name = "IMAD_OFF";
  __device__ __forceinline__ static void op(T& x, T y, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t@p mad.lo.u32 %0, %0, %1, %0;\n\t}" : "+r"(x) : "r"(y), "r"(k));
  }
};
struct MUFU_OFF {
  using T = float;
  static constexpr const char* name = "MUFU_OFF";
  __device__ __forceinline__ static void op(T& x, T, std::uint32_t k) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\t@p rsqrt.approx.ftz.f32 %0, %0;\n\t}" : "+f"(x) : "r"(k));
  }
};

template <class O, int K>
__global__ void __launch_bounds__(512, 1) k_thr(std::uint64_t* out, const std::uint32_t* kz, int reps) {
  using T = typename O::T;
  __shared__ __align__(16) T pinb[512 * 16];
  constexpr std::uint32_t kStride = 512u * 4u;
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 4u;
  const std::uint32_t k = kz[threadIdx.x];  // 0 at run time: every instance is predicated off
#pragma unroll
  for (int j = 0; j < 16; ++j) vstore<T>(base + j * kStride, static_cast<T>(1 + j));
  T x[8], y[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = vload<T>(base + j * kStride);
      y[j] = vload<T>(base + (8 + j) * kStride);
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (i & 1) O::op(y[j], x[j], k);
        else O::op(x[j], y[j], k);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      vstore<T>(base + j * kStride, x[j]);
      vstore<T>(base + (8 + j) * kStride, y[j]);
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
}

__global__ void k_ctab_base(std::uint32_t* o) {
  std::uint32_t b;
  asm volatile("mov.u32 %0, ctab;" : "=r"(b));
  *o = b;
}

struct Case {
  std::string variant;
  int u;
  std::function<void(int)> launch;
};

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "misc");
  std::uint32_t *d_b = nullptr, *d_seeds = nullptr, *d_kz = nullptr;
  std::uint64_t *d_out = nullptr, *d_seeds64 = nullptr;
  MB_CHECK(cudaMalloc(&d_b, 4));
  k_ctab_base<<<1, 1>>>(d_b);
  MB_CHECK(cudaDeviceSynchronize());
  std::uint32_t base = 0;
  MB_CHECK(cudaMemcpy(&base, d_b, 4, cudaMemcpyDeviceToHost));
  std::vector<std::uint32_t> tab(kTab);
  for (int i = 0; i < kTab; ++i) tab[i] = static_cast<std::uint32_t>(i * 2654435761u);
  MB_CHECK(cudaMemcpyToSymbol(ctab, tab.data(), sizeof(std::uint32_t) * kTab));
  std::vector<std::uint32_t> seeds(64), kz(512, 0u);
  std::vector<std::uint64_t> seeds64(64);
  for (int l = 0; l < 64; ++l) {
    seeds[l] = 0x9e3779b9u * (l + 1);
    seeds64[l] = 0x9e3779b97f4a7c15ull * (l + 1);
  }
  MB_CHECK(cudaMalloc(&d_seeds, 4 * 64));
  MB_CHECK(cudaMalloc(&d_seeds64, 8 * 64));
  MB_CHECK(cudaMalloc(&d_kz, 4 * 512));
  MB_CHECK(cudaMalloc(&d_out, 8 * (2 * args.reps + 2)));
  MB_CHECK(cudaMemcpy(d_seeds, seeds.data(), 4 * 64, cudaMemcpyHostToDevice));
  MB_CHECK(cudaMemcpy(d_seeds64, seeds64.data(), 8 * 64, cudaMemcpyHostToDevice));
  MB_CHECK(cudaMemcpy(d_kz, kz.data(), 4 * 512, cudaMemcpyHostToDevice));
  const int reps = args.reps;
  std::vector<Case> cases;
#define LATV(O, U) cases.push_back({std::string("lat.") + O::name, U, [&](int) { k_vlat<O, U><<<1, 32>>>(d_out, d_seeds, reps); }});
#define LATU(O, U) cases.push_back({std::string("ulat.") + O::name, U, [&](int l) { k_ulat<O, U><<<1, 32>>>(d_out, 3u + l, 0x9e3779b9u, reps); }});
#define LAT64(U) cases.push_back({"lat.LEA64", U, [&](int) { k_lea64<U><<<1, 32>>>(d_out, d_seeds64, reps); }});
  LATU(ULEA, 8) LATU(ULEA, 16) LATU(ULEA, 32) LATU(ULEA, 64)
  LAT64(8) LAT64(16) LAT64(32) LAT64(64)
  std::uint32_t* d_g = nullptr;
  MB_CHECK(cudaMalloc(&d_g, 4 * 1024));
#define SHOT(N, NAME) cases.push_back({std::string("shot.") + NAME, N, [&](int) { k_shot<N><<<1, 32>>>(d_out, reps, d_g); }});
  SHOT(0, "membar_cta") SHOT(1, "membar_gpu") SHOT(3, "s2r") SHOT(4, "s2ur")
  SHOT(5, "fence_acqrel_gpu") SHOT(6, "fence_sc_cta") SHOT(7, "fence_acqrel_sys")
  SHOT(8, "stg_fence_acqrel_cta") SHOT(9, "stg_fence_acqrel_gpu") SHOT(10, "stg_fence_sc_gpu")
  SHOT(11, "bar") SHOT(12, "stg_bar") SHOT(13, "sts_bar")
  SHOT(14, "stg1x4_fence_gpu") SHOT(15, "stg32x16_fence_gpu") SHOT(16, "stg32x4s128_fence_gpu")
  SHOT(17, "stg2x128_fence_gpu") SHOT(18, "stg1x8_fence_gpu")
  SHOT(19, "fence_async_shared") SHOT(20, "sts_fence_async_shared") SHOT(21, "fence_async_global")
  SHOT(22, "stg_fence_async_global") SHOT(23, "fence_async")
#undef SHOT
  // store acknowledgment per address: the stg_fence_sc_gpu shot (32 lanes x 4 B,
  // one 128 B line) on 64 lines spread over a 256 MiB buffer (pseudo-random
  // 4 KiB-aligned offsets), one row per line: does the ack depend on which
  // L2 slice / die the line lives in?
  std::uint32_t* d_lines = nullptr;
  constexpr std::size_t kLineSpan = std::size_t(256) << 20;
  MB_CHECK(cudaMalloc(&d_lines, kLineSpan));
  MB_CHECK(cudaMemset(d_lines, 0, kLineSpan));
  for (int i = 0; i < 64; ++i) {
    const std::size_t off = ((static_cast<std::size_t>(i) * 2654435761u) % (kLineSpan / 4096)) * 4096 / 4;
    cases.push_back({"shot.stg_line." + std::to_string(i), 10,
                     [&, off](int) { k_shot<10><<<1, 32>>>(d_out, reps, d_lines + off); }});
  }
#define THR(O, K) cases.push_back({std::string("thr.") + O::name, K, [&](int) { k_thr<O, K><<<1, 512>>>(d_out, d_kz, reps); }});
  THR(FFMA_OFF, 16) THR(FFMA_OFF, 32) THR(FFMA_OFF, 64)
  THR(IMAD_OFF, 16) THR(IMAD_OFF, 32) THR(IMAD_OFF, 64)
  THR(MUFU_OFF, 16) THR(MUFU_OFF, 32) THR(MUFU_OFF, 64)
  for (const auto& c : cases) {
    if (!mb_selected(args, c.variant)) continue;
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      c.launch(l);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      mb_collect(d_out, args.reps, 1, 0, s);
    }
    w.row(c.variant, c.u, "\"mode\":\"" + c.variant.substr(0, c.variant.find('.')) + "\"", s);
  }
  return w.failures ? 1 : 0;
}
