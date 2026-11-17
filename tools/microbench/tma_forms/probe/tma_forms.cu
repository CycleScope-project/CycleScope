// tma_forms probe (tools/microbench/tma_forms/protocol.json):
//   LDGSTS  cp.async.cg.shared.global 16 B x 32 lanes, C copies/group,
//           commit_group + wait_group 0 in the issuing warp
//   STAS    st.async.shared::cluster.mbarrier::complete_tx 4/8/16 B to
//           the remote CTA of a 2-CTA cluster; remote warp test_wait
//   GATHER4 cp.async.bulk.tensor.2d ... tile::gather4 (sm_103a only)
//   SCATTER4 cp.async.bulk.tensor.2d ... tile::scatter4 (sm_103a only)
//   IM2COL  cp.async.bulk.tensor.3d ... im2col (both devices)
// measurement scheme as tma_cold/probe/tma.cu: rank0 warp0 lane0 (or
// remote warp for STAS) timestamps around U ops inside one group;
// U=0 is the empty-group control; one probing cluster per launch.
#include "../mb.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

__device__ __forceinline__ void jitter(std::uint32_t rep,
                                       std::uint32_t seed) {
  std::uint32_t h = (rep * 2654435761u) ^ (seed * 2246822519u);
  std::uint32_t d = 96u + ((h ^ (h >> 15)) & 1023u);
#pragma unroll 1
  for (std::uint32_t i = 0; i < d; ++i) asm volatile("");
}
__device__ __forceinline__ void commit_cp() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void ldsts16(std::uint32_t dst,
                                        const void* src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src)
      : "memory");
}
__device__ __forceinline__ void mbar_init(std::uint32_t mbar,
                                        std::uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar),
               "r"(count));
}
__device__ __forceinline__ void expect_tx(std::uint32_t mbar,
                                        std::uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
          mbar), "r"(bytes)
      : "memory");
}
__device__ __forceinline__ void mbar_wait(std::uint32_t mbar,
                                        std::uint32_t parity) {
  asm volatile(
      "{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,"
      " [%0], %1; @!p bra W;}" ::"r"(mbar), "r"(parity)
      : "memory");
}
__device__ __forceinline__ std::uint32_t mb_mapa(std::uint32_t a,
                                                std::uint32_t rank) {
  std::uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void csync() {
  asm volatile("barrier.cluster.arrive;\n"
               "barrier.cluster.wait;" ::: "memory");
}

// ---- LDGSTS ---------------------------------------------------------
// latency cell: U groups (U>0) or empty-group control (U=0) of C copies
// per group; cadence cell: U in {1,2,4,8} groups of C=8.
template <int C, int U>
__global__ void __launch_bounds__(32, 1) k_ldgsts(
    const unsigned char* src, std::uint64_t* out, int reps,
    std::uint32_t seed) {
  extern __shared__ unsigned char smem[];
  const std::uint32_t sa =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const std::uint32_t lane = threadIdx.x & 31;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const unsigned char* base =
        src + (((rep * 131u + seed * 977u) & 4095u) << 4);
    jitter(rep, seed);
    const std::uint64_t t_s = mb_clock();
#pragma unroll
    for (int g = 0; g < U; ++g) {
#pragma unroll
      for (int c = 0; c < C; ++c)
        ldsts16(sa + ((lane * C + c) << 4 & 0xFFF0u),
                base + (((lane * C + c) & 16383u) << 4));
      commit_cp();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    const std::uint64_t t_w = mb_clock();
    out[2 * rep] = t_s;
    out[2 * rep + 1] = t_w;
  }
}

// ---- STAS ------------------------------------------------------------
// 2-CTA cluster.  rank 0 warp0 lane0 issues U stores of B bytes into
// the remote CTA's smem, each with complete_tx on the remote mbarrier;
// the remote warp test_waits and writes t_w via global out[] slot.
// out: [0]=t_s (rank0 lane0), [1]=t_w (rank1 lane0).  U=0 control:
// remote warp spins on a bare arrive.
template <int B, int U>
__global__ void __launch_bounds__(64, 1) k_stas(
    std::uint64_t* out, int reps, std::uint32_t seed) {
  extern __shared__ unsigned char smem[];
  std::uint32_t mbar =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  std::uint32_t data = mbar + 64;
  const std::uint32_t rank = mb_ctarank();
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    mbar_init(mbar, 1);
    asm volatile("fence.cluster;" ::: "memory");
  }
  csync();
  // remote-window addresses once
  const std::uint32_t rmbar = mb_mapa(mbar, rank ^ 1u);
  const std::uint32_t rdata = mb_mapa(data, rank ^ 1u);
  for (int rep = 0; rep < reps; ++rep) {
    if (rank == 0 && lane == 0 && warp == 0) {
      if (rep == 0) {
        jitter(0, seed);
        const std::uint64_t t_s = mb_clock();
        if constexpr (U == 0) {
          // control: a bare remote arrive
          asm volatile(
              "mbarrier.arrive.release.cluster.shared::cluster.b64 _,"
              " [%0];" ::"r"(rmbar)
              : "memory");
        }
#pragma unroll
        for (int i = 0; i < U; ++i) {
          if constexpr (B == 4)
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes"
                ".u32 [%0], %1, [%2];" ::"r"(rdata + i * 4u),
                "r"(seed + i), "r"(rmbar)
                : "memory");
          else if constexpr (B == 8)
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes"
                ".u64 [%0], %1, [%2];" ::"r"(rdata + i * 8u),
                "l"((std::uint64_t)seed + i), "r"(rmbar)
                : "memory");
          else
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes"
                ".v4.u32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
                    rdata + i * 16u),
                "r"(seed), "r"(i + 1u), "r"(2u), "r"(3u), "r"(rmbar)
                : "memory");
        }
        out[0] = t_s;
      }
    } else if (rank == 1 && lane == 0 && warp == 0) {
      if (rep == 0) {
        // own arrive + expect the U*B tx bytes (0 for the control:
        // parity flips on the bare remote arrive).
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                "r"(mbar), "r"((std::uint32_t)(U * B))
            : "memory");
        mbar_wait(mbar, 0);
        out[1] = mb_clock();
      }
    }
    csync();
  }
}

// ---- GATHER4 (sm_103a) ----------------------------------------------
// 4 rows x box_b bytes each, coords {col0..col3} per op.
template <int U>
__global__ void __launch_bounds__(32, 1) k_gather4(
    const __grid_constant__ CUtensorMap map, std::uint64_t* out,
    int reps, int box_b, std::uint32_t mbar_bytes,
    std::uint32_t seed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  extern __shared__ unsigned char smem[];
  const std::uint32_t mbar =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const std::uint32_t dst = mbar + 64;
  const std::uint32_t lane = threadIdx.x & 31;
  if (threadIdx.x == 0) mbar_init(mbar, 1);
  __syncthreads();
  for (int rep = 0; rep < reps; ++rep) {
    if (lane == 0) {
      expect_tx(mbar, mbar_bytes);
      jitter(rep, seed);
      const std::uint64_t t_s = mb_clock();
#pragma unroll
      for (int i = 0; i < U; ++i) {
        int x = (rep * 4096 + i * 512 + seed * 128) & ~255;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".tile::gather4.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(dst),
            "l"(reinterpret_cast<std::uint64_t>(&map)), "r"(0),
            "r"(x), "r"(x + 4), "r"(x + 8), "r"(x + 12),
            "r"(mbar)
            : "memory");
      }
      const std::uint64_t t_e = mb_clock();
      mbar_wait(mbar, rep & 1);
      const std::uint64_t t_w = mb_clock();
      out[3 * rep] = t_s;
      out[3 * rep + 1] = t_e;
      out[3 * rep + 2] = t_w;
    }
  }
#endif
}

// ---- UBLKCP global->shared plain bulk copy ------------------------
// cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes;
// one CTA, one waiter warp, L2-hot source (same gsrc window reused).
template <int U>
__global__ void __launch_bounds__(32, 1) k_ublkcp(
    const unsigned char* gsrc, std::uint64_t* out, int reps, int bytes,
    std::uint32_t seed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ unsigned char smem[];
  const std::uint32_t mbar =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const std::uint32_t dst = mbar + 64;
  const std::uint32_t lane = threadIdx.x & 31;
  if (threadIdx.x == 0) mbar_init(mbar, 1);
  __syncthreads();
  for (int rep = 0; rep < reps; ++rep) {
    if (lane == 0) {
      jitter(rep, seed);
      const std::uint64_t t_s = mb_clock();
      if constexpr (U == 0) {
        // U=0 control: a plain arrive satisfies the barrier (init
        // count 1); the waiter still spins so the control window
        // matches the U>0 shape (like the tile cells).
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::
                     "r"(mbar) : "memory");
        const std::uint64_t t_e0 = mb_clock();
        mbar_wait(mbar, rep & 1);
        const std::uint64_t t_w0 = mb_clock();
        out[3 * rep] = t_s;
        out[3 * rep + 1] = t_e0;
        out[3 * rep + 2] = t_w0;
        continue;
      }
      expect_tx(mbar, (std::uint32_t)bytes * U);
#pragma unroll
      for (int i = 0; i < U; ++i) {
        const unsigned char* src =
            gsrc + (((rep * U + i) * (std::uint32_t)bytes + seed * 128) &
                    ~255u);
        asm volatile(
            "cp.async.bulk.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
            "r"(dst + (std::uint32_t)i * bytes),
            "l"(reinterpret_cast<std::uint64_t>(src)),
            "r"(bytes), "r"(mbar)
            : "memory");
      }
      const std::uint64_t t_e = mb_clock();
      mbar_wait(mbar, rep & 1);
      const std::uint64_t t_w = mb_clock();
      out[3 * rep] = t_s;
      out[3 * rep + 1] = t_e;
      out[3 * rep + 2] = t_w;
    }
  }
#endif
}

template <int DC, int U>
__global__ void __launch_bounds__(32, 1) k_ublkcp_mc(
    const unsigned char* gsrc, std::uint64_t* out, std::uint64_t* ack,
    int reps, int bytes, std::uint32_t seed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ unsigned char smem[];
  const std::uint32_t mbar =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const std::uint32_t dst = mbar + 64;
  const std::uint32_t rank = mb_ctarank();
  // every CTA owns a barrier (count 1); made visible cluster-wide
  if (threadIdx.x == 0) {
    mbar_init(mbar, 1);
    mb_fence_mbar_init();
  }
  csync();
  const std::uint16_t mask = (1u << DC) - 1u;
  for (int rep = 0; rep < reps; ++rep) {
    if (threadIdx.x == 0) {
      if constexpr (U > 0) {
        expect_tx(mbar, (std::uint32_t)bytes * U);
      } else {
        // U=0 control: a plain arrive satisfies the barrier
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::
                     "r"(mbar) : "memory");
      }
    }
    csync();  // expect_tx / arrive ordered before the issue
    if (threadIdx.x == 0) {
      std::uint64_t t_s = 0, t_e = 0;
      if (rank == 0) {
        jitter(rep, seed);
        t_s = mb_clock();
#pragma unroll
        for (int i = 0; i < U; ++i) {
          const unsigned char* src =
              gsrc + (((rep * U + i) * (std::uint32_t)bytes +
                       seed * 128) & ~255u);
          asm volatile(
              "cp.async.bulk.shared::cluster.global"
              ".mbarrier::complete_tx::bytes.multicast::cluster"
              " [%0], [%1], %2, [%3], %4;" ::
              "r"(dst + (std::uint32_t)i * bytes),
              "l"(reinterpret_cast<std::uint64_t>(src)),
              "r"(bytes), "r"(mbar), "h"(mask)
              : "memory");
        }
        t_e = mb_clock();
      }
      mbar_wait(mbar, rep & 1);
      const std::uint64_t t_w = mb_clock();
      if (rank == 0) {
        out[3 * rep] = t_s;        // completion_own (v2 rule, dc > 1)
        out[3 * rep + 1] = t_e;
        out[3 * rep + 2] = t_w;
      }
      ack[rep * DC + rank] = t_w;  // slowest peer -> t_ack (reported)
    }
    csync();  // rep boundary: no copy still in flight into next rep
  }
#endif
}

static void fail_cell(MbWriter& w, const char* variant, int u,
                      const std::string& params, int launch,
                      cudaError_t e) {
  std::fprintf(stdout, "FAIL %s launch %d: %s\n", variant, launch,
               cudaGetErrorName(e));
  w.row(variant, u,
        params + ",\"status\":\"unrealised:" + cudaGetErrorName(e) + "\"",
        {});
}

static void run_ub_mc(const MbArgs& a, MbWriter& w, void* g,
                      std::uint64_t* dout, std::uint64_t* dack,
                      std::uint64_t* h, std::uint64_t* hack, int bytes,
                      int dc) {
  for (int u : {0, 1, 2, 4}) {
    std::vector<double> samples, ackd;
    char v[80];
    std::snprintf(v, sizeof v, "lat.ublkcpmc.dc%d.b%d.u%d", dc, bytes, u);
    const std::string params =
        "\"class\":\"UBLKCP\",\"bytes\":" + std::to_string(bytes) +
        ",\"destination_count\":" + std::to_string(dc);
    cudaError_t bad = cudaSuccess;
    for (int l = 0; l < a.launches && bad == cudaSuccess; ++l) {
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(dc, 1, 1);
      cfg.blockDim = dim3(32, 1, 1);
      cfg.dynamicSmemBytes = 200704;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = dc;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      void (*kfn)(const unsigned char*, std::uint64_t*, std::uint64_t*,
                  int, int, std::uint32_t) = nullptr;
      if (dc == 2) {
        switch (u) {
          case 0: kfn = k_ublkcp_mc<2, 0>; break;
          case 1: kfn = k_ublkcp_mc<2, 1>; break;
          case 2: kfn = k_ublkcp_mc<2, 2>; break;
          default: kfn = k_ublkcp_mc<2, 4>; break;
        }
      } else {
        switch (u) {
          case 0: kfn = k_ublkcp_mc<4, 0>; break;
          case 1: kfn = k_ublkcp_mc<4, 1>; break;
          case 2: kfn = k_ublkcp_mc<4, 2>; break;
          default: kfn = k_ublkcp_mc<4, 4>; break;
        }
      }
      cudaFuncSetAttribute(
          kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 200704);
      cudaLaunchKernelEx(&cfg, kfn,
          reinterpret_cast<const unsigned char*>(g), dout, dack, a.reps,
          bytes, (std::uint32_t)l);
      bad = cudaDeviceSynchronize();
      if (bad != cudaSuccess) {
        cudaGetLastError();
        fail_cell(w, v, u, params, l, bad);
        break;
      }
      cudaMemcpy(h, dout, 3 * a.reps * 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(hack, dack, dc * a.reps * 8, cudaMemcpyDeviceToHost);
      if (l == 0) continue;
      for (int r = 1; r < a.reps; ++r) {
        samples.push_back(double(h[3 * r + 2] - h[3 * r]));
        std::uint64_t mx = 0;
        for (int p = 0; p < dc; ++p)
          if (hack[r * dc + p] > mx) mx = hack[r * dc + p];
        ackd.push_back(double(mx - h[3 * r + 2]));  // t_ack - t_w(rank0)
      }
    }
    if (bad != cudaSuccess) continue;
    std::sort(ackd.begin(), ackd.end());
    const double ack_med = ackd.empty() ? 0.0 : ackd[ackd.size() / 2];
    w.row(v, u, params, samples,
          "\"t_ack_minus_own_median\":" + std::to_string(ack_med));
  }
}

template <int U>
static void run_ub(const MbArgs& a, MbWriter& w, void* g,
                   std::uint64_t* dout, std::uint64_t* h, int bytes) {
  std::vector<double> samples;
  for (int l = 0; l < a.launches; ++l) {
    cudaFuncSetAttribute(
      k_ublkcp<U>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200704);
    char v[80];
    std::snprintf(v, sizeof v, "lat.ublkcp.b%d.u%d", bytes, U);
    k_ublkcp<U><<<1, 32, 200704>>>(
        reinterpret_cast<const unsigned char*>(g), dout, a.reps, bytes,
        l);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      cudaGetLastError();
      fail_cell(w, v, U,
                "\"class\":\"UBLKCP\",\"bytes\":" +
                    std::to_string(bytes), l, e);
      return;
    }
    cudaMemcpy(h, dout, 3 * a.reps * 8, cudaMemcpyDeviceToHost);
    if (l == 0) continue;
    for (int r = 1; r < a.reps; ++r)
      samples.push_back(double(h[3 * r + 2] - h[3 * r]));
  }
  char v[80];
  std::snprintf(v, sizeof v, "lat.ublkcp.b%d.u%d", bytes, U);
  w.row(v, U, "\"class\":\"UBLKCP\",\"bytes\":" + std::to_string(bytes),
        samples);
}

// ---- SCATTER4 (sm_103a) ----------------------------------------------
// smem -> global tile::scatter4, bulk-group completion.
template <int U>
__global__ void __launch_bounds__(32, 1) k_scatter4(
    const __grid_constant__ CUtensorMap map, std::uint64_t* out,
    int reps, int box_b, std::uint32_t seed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  extern __shared__ unsigned char smem[];
  const std::uint32_t src =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const std::uint32_t lane = threadIdx.x & 31;
  for (int rep = 0; rep < reps; ++rep) {
    if (lane == 0) {
      jitter(rep, seed);
      const std::uint64_t t_s = mb_clock();
#pragma unroll
      for (int i = 0; i < U; ++i) {
        int x = (rep * 4096 + i * 512 + seed * 128) & ~255;
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4"
            ".bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];" ::"l"(
                reinterpret_cast<std::uint64_t>(&map)),
            "r"(0), "r"(x), "r"(x + 4), "r"(x + 8), "r"(x + 12),
            "r"(src)
            : "memory");
      }
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      const std::uint64_t t_e = mb_clock();
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      const std::uint64_t t_w = mb_clock();
      out[3 * rep] = t_s;
      out[3 * rep + 1] = t_e;
      out[3 * rep + 2] = t_w;
    }
  }
#endif
}

// ---- IM2COL ----------------------------------------------------------
// 4d im2col load: pixels_per_column P columns x 128 B channels.
template <int U>
__global__ void __launch_bounds__(32, 1) k_im2col(
    const __grid_constant__ CUtensorMap map, std::uint64_t* out,
    int reps, int ppc, std::uint32_t mbar_bytes, std::uint32_t seed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ unsigned char smem[];
  const std::uint32_t mbar =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const std::uint32_t dst = mbar + 64;
  const std::uint32_t lane = threadIdx.x & 31;
  if (threadIdx.x == 0) mbar_init(mbar, 1);
  __syncthreads();
  for (int rep = 0; rep < reps; ++rep) {
    if (lane == 0) {
      expect_tx(mbar, mbar_bytes);
      jitter(rep, seed);
      const std::uint64_t t_s = mb_clock();
#pragma unroll
      for (int i = 0; i < U; ++i) {
        int x = (rep * 64 + i * 8) & ~7, y = seed & 7;
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cluster.global.im2col"
            ".mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4, %5}], [%6], {%7, %8};" ::"r"(dst),
            "l"(reinterpret_cast<std::uint64_t>(&map)), "r"(x), "r"(y),
            "r"(0), "r"(0), "r"(mbar), "h"((std::uint16_t)0),
            "h"((std::uint16_t)0)
            : "memory");
      }
      const std::uint64_t t_e = mb_clock();
      mbar_wait(mbar, rep & 1);
      const std::uint64_t t_w = mb_clock();
      out[3 * rep] = t_s;
      out[3 * rep + 1] = t_e;
      out[3 * rep + 2] = t_w;
    }
  }
#endif
}

// ---- host -------------------------------------------------------------
#include <dlfcn.h>
typedef CUresult (*EncTiled)(CUtensorMap*, CUtensorMapDataType,
    cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
    const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
typedef CUresult (*EncIm2col)(CUtensorMap*, CUtensorMapDataType,
    cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
    const int*, const int*, cuuint32_t, cuuint32_t, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);
static void* sym_(const char* n) {
  void* f = dlsym(RTLD_DEFAULT, n);
  if (!f) f = dlsym(dlopen("libcuda.so.1", RTLD_NOW), n);
  return f;
}
static void* gm(std::size_t n) {
  void* p = nullptr;
  if (cudaMalloc(&p, n) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    std::exit(2);
  }
  return p;
}

// 2d tiled map over the 512 MiB buffer for gather4/scatter4 cells:
// inner dim box_b bytes as bf16 elems (box_b/2), outer rows.
static CUresult map2d(void* g, std::uint64_t rows, std::uint32_t box_b,
                      CUtensorMap* out) {
  auto fn = reinterpret_cast<EncTiled>(sym_("cuTensorMapEncodeTiled"));
  if (!fn) return CUDA_ERROR_NOT_FOUND;
  CUtensorMap m{};
  cuuint64_t dim[2] = {(std::uint64_t)(box_b / 2), rows};
  cuuint64_t str[1] = {(std::uint64_t)box_b};
  for (cuuint32_t bx : {(std::uint32_t)(box_b / 2), 256u, 128u}) {
    cuuint32_t box[2] = {bx, 1}, es[2] = {1, 1};
    CUresult r = fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, dim, str,
                    box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_NONE,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r == CUDA_SUCCESS) { *out = m; return r; }
  }
  return CUDA_ERROR_INVALID_VALUE;
}

// 4d im2col map: pixels {32,64,128} per column x 64 bf16 channels
static CUresult map_im2col(void* g, std::uint32_t ppc,
                           CUtensorMap* out) {
  auto fn = reinterpret_cast<EncIm2col>(sym_("cuTensorMapEncodeIm2col"));
  CUtensorMap m{};
  cuuint64_t dim[4] = {64, 512, 512, 512};           // C,W,H,N
  cuuint64_t str[3] = {512 * 128, 512 * 512 * 128,
                       512ull * 512 * 512 * 128};
  int lo[2] = {0, 0}, up[2] = {15, 15};
  cuuint32_t es[4] = {1, 1, 1, 1};
  if (!fn) return CUDA_ERROR_NOT_FOUND;
  return fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, g, dim, str,
            lo, up, 64, ppc, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
      ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

template <int C, int U>
static void run_ld(const MbArgs& a, MbWriter& w, unsigned char* src,
                   std::uint64_t* dout, std::uint64_t* h, const char* var,
                   int cname) {
  cudaFuncSetAttribute(
      k_ldgsts<C, U>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      65536);
  std::vector<double> samples;
  for (int l = 0; l < a.launches; ++l) {
    k_ldgsts<C, U><<<1, 32, 65536>>>(src, dout, a.reps, l);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      cudaGetLastError();
      char vv[80]; std::snprintf(vv, sizeof vv, "%s.c%d.u%d", var, cname, U);
      fail_cell(w, vv, U,
                "\"class\":\"LDGSTS\",\"copies\":" +
                    std::to_string(cname), l, e);
      return;
    }
    cudaMemcpy(h, dout, 2 * a.reps * 8, cudaMemcpyDeviceToHost);
    if (l == 0) continue;      // discard warmup launch
    for (int r = 1; r < a.reps; ++r)
      samples.push_back(double(h[2 * r + 1] - h[2 * r]));
  }
  char v[80];
  std::snprintf(v, sizeof v, "%s.c%d.u%d", var, cname, U);
  w.row(v, U, "\"class\":\"LDGSTS\",\"copies\":" + std::to_string(cname),
        samples);
}

template <int B, int U>
static void run_stas(const MbArgs& a, MbWriter& w, std::uint64_t* dout,
                     const char* var) {
  std::vector<double> samples;
  for (int l = 0; l < a.launches; ++l) {
    std::uint64_t h[2] = {0, 0};
    cudaMemset(dout, 0, 16);
    cudaFuncSetAttribute(
      k_stas<B, U>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      65664);
  cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(2); cfg.blockDim = dim3(64);
    cfg.dynamicSmemBytes = 65664;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, k_stas<B, U>, dout, a.reps,
                       (std::uint32_t)l);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      cudaGetLastError();
      char vv[80]; std::snprintf(vv, sizeof vv, "%s.b%d.u%d", var, B, U);
      fail_cell(w, vv, U,
                "\"class\":\"STAS\",\"bytes\":" + std::to_string(B),
                l, e);
      return;
    }
    cudaMemcpy(h, dout, 16, cudaMemcpyDeviceToHost);
    if (l > 0 && h[0] && h[1]) samples.push_back(double(h[1] - h[0]));
  }
  char v[80];
  std::snprintf(v, sizeof v, "%s.b%d.u%d", var, B, U);
  w.row(v, U, "\"class\":\"STAS\",\"bytes\":" + std::to_string(B),
        samples);
}

template <int U>
static void run_g4(const MbArgs& a, MbWriter& w, void* g,
                   std::uint64_t* dout, std::uint64_t* h, int box_b) {
  CUtensorMap m{};
  if (map2d(g, (512ull << 20) / box_b, box_b, &m) != CUDA_SUCCESS) {
    char vv[80]; std::snprintf(vv, sizeof vv, "lat.gather4.b%d.u%d", box_b, U);
    w.row(vv, U, "\"class\":\"GATHER4\",\"status\":\"unrealised:encode\"", {});
    return;
  }
  std::vector<double> samples;
  for (int l = 0; l < a.launches; ++l) {
    cudaFuncSetAttribute(
      k_gather4<U>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    k_gather4<U><<<1, 32, 65536>>>(m, dout, a.reps, box_b,
                                  4u * box_b, l);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      cudaGetLastError();
      char vv[80]; std::snprintf(vv, sizeof vv, "lat.gather4.b%d.u%d", box_b, U);
      fail_cell(w, vv, U,
                "\"class\":\"GATHER4\",\"bytes\":" +
                    std::to_string(4 * box_b), l, e);
      return;
    }
    cudaMemcpy(h, dout, 3 * a.reps * 8, cudaMemcpyDeviceToHost);
    if (l == 0) continue;
    for (int r = 1; r < a.reps; ++r)
      samples.push_back(double(h[3 * r + 2] - h[3 * r]));
  }
  char v[80];
  std::snprintf(v, sizeof v, "lat.gather4.b%d.u%d", box_b, U);
  w.row(v, U, "\"class\":\"GATHER4\",\"bytes\":" +
                  std::to_string(4 * box_b), samples);
}

template <int U>
static void run_s4(const MbArgs& a, MbWriter& w, void* g,
                   std::uint64_t* dout, std::uint64_t* h, int box_b) {
  CUtensorMap m{};
  if (map2d(g, (512ull << 20) / box_b, box_b, &m) != CUDA_SUCCESS) {
    char vv[80]; std::snprintf(vv, sizeof vv, "lat.scatter4.b%d.u%d", box_b, U);
    w.row(vv, U, "\"class\":\"SCATTER4\",\"status\":\"unrealised:encode\"", {});
    return;
  }
  std::vector<double> samples;
  for (int l = 0; l < a.launches; ++l) {
    cudaFuncSetAttribute(
      k_scatter4<U>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    k_scatter4<U><<<1, 32, 65536>>>(m, dout, a.reps, box_b, l);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      cudaGetLastError();
      char vv[80]; std::snprintf(vv, sizeof vv, "lat.scatter4.b%d.u%d", box_b, U);
      fail_cell(w, vv, U,
                "\"class\":\"SCATTER4\",\"bytes\":" +
                    std::to_string(4 * box_b), l, e);
      return;
    }
    cudaMemcpy(h, dout, 3 * a.reps * 8, cudaMemcpyDeviceToHost);
    if (l == 0) continue;
    for (int r = 1; r < a.reps; ++r)
      samples.push_back(double(h[3 * r + 2] - h[3 * r]));
  }
  char v[80];
  std::snprintf(v, sizeof v, "lat.scatter4.b%d.u%d", box_b, U);
  w.row(v, U, "\"class\":\"SCATTER4\",\"bytes\":" +
                  std::to_string(4 * box_b), samples);
}

template <int U>
static void run_i2(const MbArgs& a, MbWriter& w, void* g,
                   std::uint64_t* dout, std::uint64_t* h, int ppc) {
  CUtensorMap m{};
  if (map_im2col(g, ppc, &m) != CUDA_SUCCESS) {
    char vv[80]; std::snprintf(vv, sizeof vv, "lat.im2col.p%d.u%d", ppc, U);
    w.row(vv, U, "\"class\":\"IM2COL\",\"status\":\"unrealised:encode\"", {});
    return;
  }
  std::vector<double> samples;
  for (int l = 0; l < a.launches; ++l) {
    cudaFuncSetAttribute(
      k_im2col<U>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    k_im2col<U><<<1, 32, 65536>>>(m, dout, a.reps, ppc,
                                 64u * ppc * 128u, l);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      cudaGetLastError();
      char vv[80]; std::snprintf(vv, sizeof vv, "lat.im2col.p%d.u%d", ppc, U);
      fail_cell(w, vv, U,
                "\"class\":\"IM2COL\",\"pixels\":" +
                    std::to_string(ppc), l, e);
      return;
    }
    cudaMemcpy(h, dout, 3 * a.reps * 8, cudaMemcpyDeviceToHost);
    if (l == 0) continue;
    for (int r = 1; r < a.reps; ++r)
      samples.push_back(double(h[3 * r + 2] - h[3 * r]));
  }
  char v[80];
  std::snprintf(v, sizeof v, "lat.im2col.p%d.u%d", ppc, U);
  w.row(v, U, "\"class\":\"IM2COL\",\"pixels\":" + std::to_string(ppc),
        samples);
}

int main(int argc, char** argv) {
  MbArgs args = mb_parse(argc, argv);
  const char* probe = "tma_forms";
  MbWriter w(args, probe);
  unsigned char* gsrc = static_cast<unsigned char*>(gm(512ull << 20));
  std::uint64_t* dout =
      static_cast<std::uint64_t*>(gm(sizeof(std::uint64_t) * 3 * args.reps));
  std::uint64_t* hout = new std::uint64_t[3 * args.reps];
  std::uint64_t* dack =
      static_cast<std::uint64_t*>(gm(sizeof(std::uint64_t) * 8 * args.reps));
  std::uint64_t* hack = new std::uint64_t[8 * args.reps];
  // --cells <family>: one process per family (protocol addendum (a));
  // empty = all families
  const std::string cells = args.cells;
  auto fam = [&](const char* f) {
    return cells.empty() || cells == f;
  };
  const bool F_ld = fam("ldgsts"), F_st = fam("stas"),
             F_g4 = fam("gather4"), F_s4 = fam("scatter4"),
             F_ub = fam("ublkcp"), F_mc = fam("ublkcp_mc"),
             F_i2 = fam("im2col");
  if (F_ld) {
  run_ld<1, 1>(args, w, gsrc, dout, hout, "lat.ldgsts", 1);
  run_ld<2, 1>(args, w, gsrc, dout, hout, "lat.ldgsts", 2);
  run_ld<4, 1>(args, w, gsrc, dout, hout, "lat.ldgsts", 4);
  run_ld<8, 1>(args, w, gsrc, dout, hout, "lat.ldgsts", 8);
  run_ld<16, 1>(args, w, gsrc, dout, hout, "lat.ldgsts", 16);
  run_ld<32, 1>(args, w, gsrc, dout, hout, "lat.ldgsts", 32);
  run_ld<1, 0>(args, w, gsrc, dout, hout, "ctl.ldgsts", 0);
  run_ld<8, 2>(args, w, gsrc, dout, hout, "cad.ldgsts", 8);
  run_ld<8, 4>(args, w, gsrc, dout, hout, "cad.ldgsts", 8);
  run_ld<8, 8>(args, w, gsrc, dout, hout, "cad.ldgsts", 8);
  }
  if (F_st) {
  run_stas<4, 0>(args, w, dout, "ctl.stas");
  run_stas<4, 1>(args, w, dout, "lat.stas");
  run_stas<8, 1>(args, w, dout, "lat.stas");
  run_stas<16, 1>(args, w, dout, "lat.stas");
  run_stas<4, 8>(args, w, dout, "cad.stas");
  run_stas<8, 8>(args, w, dout, "cad.stas");
  run_stas<16, 8>(args, w, dout, "cad.stas");
  }

  int cc = 0;
  cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, 0);
  if (cc >= 10 && F_g4) {              // gather4: sm_103a only
    run_g4<1>(args, w, gsrc, dout, hout, 128);
    run_g4<1>(args, w, gsrc, dout, hout, 256);
    run_g4<1>(args, w, gsrc, dout, hout, 512);
    run_g4<0>(args, w, gsrc, dout, hout, 256);
  }
  if (cc >= 10 && F_s4) {              // scatter4: sm_103a only
    run_s4<1>(args, w, gsrc, dout, hout, 128);
    run_s4<1>(args, w, gsrc, dout, hout, 256);
    run_s4<1>(args, w, gsrc, dout, hout, 512);
    run_s4<0>(args, w, gsrc, dout, hout, 256);
  }
  if (F_ub) {                          // UBLKCP g->s: 1/4/16/32 KiB
    const int szs[] = {1024, 4096, 16384, 32768};
    for (int b : szs) {
      run_ub<0>(args, w, gsrc, dout, hout, b);
      run_ub<1>(args, w, gsrc, dout, hout, b);
      run_ub<2>(args, w, gsrc, dout, hout, b);
      run_ub<4>(args, w, gsrc, dout, hout, b);
      // addendum (b): no b32768.u8 (8 x 32 KiB > the smem buffer)
      if (b != 32768) run_ub<8>(args, w, gsrc, dout, hout, b);
    }
  }
  if (F_mc) {                          // UBLKCP multicast: dc {2,4} x
    for (int b : {4096, 16384})        //   {4,16} KiB x U {0,1,2,4}
      for (int dc : {2, 4})
        run_ub_mc(args, w, gsrc, dout, dack, hout, hack, b, dc);
  }
  if (cc >= 9 && F_i2) {               // im2col: both devices
    run_i2<1>(args, w, gsrc, dout, hout, 32);
    run_i2<1>(args, w, gsrc, dout, hout, 64);
    run_i2<1>(args, w, gsrc, dout, hout, 128);
    run_i2<0>(args, w, gsrc, dout, hout, 64);
  }

  delete[] hout;
  delete[] hack;
  cudaFree(dack);
  cudaFree(dout);
  cudaFree(gsrc);
  return 0;
}
