// TMA tensor load: storage.tma.completion (timing kind 38) and
// storage.async.service_start (kind 21), destination_count 1/2/4/8.
//
// Engine (engine.cpp AsynchronousTransfer): an accepted UTMALDG starts
// service at max(accept, previous service start + kind 21, capacity) and
// posts complete_tx to every destination mbarrier at service start +
// kind 38.  Cross-warp, jittered phase, test_wait spin waiter (see
// umma_impl.cuh for why):
//   rank 0 warp 0 lane 0: delay ; t_s ; arrive.expect_tx [A], U*B ;
//       U x cp.async.bulk.tensor.2d (.multicast::cluster mask 2^DC-1 if DC>1)
//       into U distinct SW128 boxes ; t_e      (U = 0: plain arrive [A])
//   rank 0 warp 1 lane 0: test_wait spin [A] ; t_w
//   ranks > 0: arrive.expect_tx own [A], U*B ; spin own [A] ; arrive leader ACK
// bf16 tensor, inner box 64 elements (128 B), box rows 32/64/128/256
// (B = 4/8/16/32 KiB).  hot: the same U boxes every rep (L2 resident);
// cold: every rep reads new rows of a 512 MiB tensor (DRAM).  Launch l
// probes from cluster (41 l) mod #clusters of a one-wave grid (the rest exit
// at once), so the samples cover many SMs (L2/GPC distance); %smid recorded.
//   kind38(dc) = (E[t_w-t_s | U=1] - o_s(t_s->UTMALDG)) - (E[.. | U=0] - o_s(t_s->ARRIVE))
//   kind21(dc) = slope over U>=1 of E[t_w-t_s]   (16 KiB hot = the canonical cell)
//   issue cadence = slope over U of t_e - t_s
#include "../mb.cuh"

#include <cuda.h>
#include <dlfcn.h>

#include <string>
#include <utility>
#include <vector>

constexpr std::uint32_t kXwDelay = 3000;
constexpr int kInner = 64;  // bf16 elements per box row (128 B, SWIZZLE_128B)
constexpr int kInnerNone = 256;  // bf16 elements per box row (512 B, SWIZZLE_NONE)

__device__ __forceinline__ void tma_load(std::uint32_t dst, const CUtensorMap* map, int x, int y, std::uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void tma_load_mc(std::uint32_t dst, const CUtensorMap* map, int x, int y, std::uint32_t mbar,
                                            std::uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, "
      "%3}], [%4], %5;" ::"r"(dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(mbar), "h"(mask)
      : "memory");
}

__device__ __forceinline__ void tma_load3(std::uint32_t dst, const CUtensorMap* map, int x, int y, int z,
                                          std::uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(
          dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void tma_load4(std::uint32_t dst, const CUtensorMap* map, int x, int y,
                                          std::uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(
          dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(0), "r"(0), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void tma_load5(std::uint32_t dst, const CUtensorMap* map, int x, int y,
                                          std::uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(
          dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(0), "r"(0), "r"(0), "r"(mbar)
      : "memory");
}

template <int R>
__device__ __forceinline__ void tma_load_rk(std::uint32_t dst, const CUtensorMap* map, int x, int y,
                                            std::uint32_t mbar) {
  if constexpr (R == 2)
    tma_load(dst, map, x, y, mbar);
  else if constexpr (R == 3)
    tma_load3(dst, map, x, y, 0, mbar);
  else if constexpr (R == 4)
    tma_load4(dst, map, x, y, mbar);
  else
    tma_load5(dst, map, x, y, mbar);
}

__device__ __forceinline__ void expect_tx(std::uint32_t mbar, std::uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void arrive_cluster(std::uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");
}

static std::uint64_t* g_out2 = nullptr;

__device__ __forceinline__ std::uint32_t mapa0(std::uint32_t a) {
  std::uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(a));
  return r;
}

template <int DC, int U>
__global__ void __launch_bounds__(64, 1) k_tma(const __grid_constant__ CUtensorMap map, std::uint64_t* out, int reps,
                                               int box_rows, int rows, int cold, std::uint32_t seed, int probe_cluster,
                                               std::uint32_t* smid_out,
                                               std::uint64_t* out2) {
  // one probing cluster per launch; the others exit at once, so the probe
  // lands on a different SM (L2 / GPC distance) for different probe_cluster
  if (static_cast<int>(blockIdx.x) / DC != probe_cluster) return;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t buf = (raw + 1023u) & ~1023u;
  __shared__ __align__(8) std::uint64_t mbar_s[2];
  const std::uint32_t rank = DC > 1 ? mb_ctarank() : 0u;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t a = mb_opaque(mb_smem_u32(&mbar_s[0])), ack = mb_opaque(mb_smem_u32(&mbar_s[1]));
  const std::uint32_t ack_leader = mb_opaque(DC > 1 ? mapa0(ack) : ack);
  // kernel parameters pinned into registers before any timed window
  __shared__ std::uint32_t pin[64][3];
  const std::uint32_t box_bytes = mb_pin(static_cast<std::uint32_t>(box_rows) * kInner * 2u, &pin[threadIdx.x][0]);
  const int ystep = static_cast<int>(mb_pin(static_cast<std::uint32_t>(box_rows), &pin[threadIdx.x][1]));
  const std::uint32_t tx = mb_pin(box_bytes * U, &pin[threadIdx.x][2]);
  if (threadIdx.x == 0 && rank == 0) *smid_out = mb_smid();
  if (threadIdx.x == 0) {
    mb_mbar_init(a, 1);
    mb_mbar_init(ack, U > 0 ? DC : 1);
    mb_fence_mbar_init();
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<std::uint64_t>(&map)) : "memory");
  }
  __syncthreads();
  if constexpr (DC > 1) mb_cluster_sync();
  std::uint32_t ph = 0;
  if (lane == 0 && rank == 0 && warp == 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = kXwDelay + ((h ^ (h >> 15)) & 2047u);
      // first row of this rep's U boxes (resolved before t_s)
      const int y0 = mb_opaque(cold ? static_cast<int>(((static_cast<std::uint64_t>(seed) * reps + rep) * U *
                                                         static_cast<std::uint64_t>(box_rows)) %
                                                        static_cast<std::uint64_t>(rows))
                                    : 0);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      const std::uint64_t t_s = mb_clock();
      if constexpr (U == 0) {
        mb_mbar_arrive(a);
      } else {
        expect_tx(a, tx);
#pragma unroll
        for (int i = 0; i < U; ++i) {
          if constexpr (DC > 1)
            tma_load_mc(buf + static_cast<std::uint32_t>(i) * box_bytes, &map, 0, y0 + i * ystep, a,
                        static_cast<std::uint16_t>((1u << DC) - 1u));
          else
            tma_load(buf + static_cast<std::uint32_t>(i) * box_bytes, &map, 0, y0 + i * ystep, a);
        }
      }
      const std::uint64_t t_e = mb_clock();
      out[3 * rep + 0] = t_s;
      out[3 * rep + 2] = t_e;
      mb_mbar_wait_to(ack, ph);
      out2[rep] = mb_clock();          // t_ack: all destinations' ACK
      ph ^= 1u;
    }
  } else if (lane == 0 && rank == 0 && warp == 1) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      mb_mbar_spin_to(a, ph);
      out[3 * rep + 1] = mb_clock();
      ph ^= 1u;
      arrive_cluster(ack_leader);
    }
  } else if (lane == 0 && rank > 0 && warp == 0 && U > 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      expect_tx(a, tx);
      mb_mbar_spin_to(a, ph);
      ph ^= 1u;
      arrive_cluster(ack_leader);
    }
  }
  __syncwarp();
  __syncthreads();
  if constexpr (DC > 1) mb_cluster_sync();
}


template <int U>
__global__ void __launch_bounds__(64, 1) k_tma3(const __grid_constant__ CUtensorMap map, std::uint64_t* out, int reps,
                                               int box_rows, int rows, int cold, std::uint32_t seed, int probe_cluster,
                                               std::uint32_t* smid_out,
                                               std::uint64_t* out2) {
  // one probing cluster per launch; the others exit at once, so the probe
  // lands on a different SM (L2 / GPC distance) for different probe_cluster
  if (static_cast<int>(blockIdx.x) != probe_cluster) return;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t buf = (raw + 1023u) & ~1023u;
  __shared__ __align__(8) std::uint64_t mbar_s[2];
  const std::uint32_t rank = 0u;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t a = mb_opaque(mb_smem_u32(&mbar_s[0])), ack = mb_opaque(mb_smem_u32(&mbar_s[1]));
  const std::uint32_t ack_leader = mb_opaque(ack);
  // kernel parameters pinned into registers before any timed window
  __shared__ std::uint32_t pin[64][3];
  // rank-3 box {64 elems, box_rows, 2 slices}: bytes = rows*128*2
  const std::uint32_t box_bytes = mb_pin(static_cast<std::uint32_t>(box_rows) * kInner * 4u, &pin[threadIdx.x][0]);
  const int ystep = static_cast<int>(mb_pin(static_cast<std::uint32_t>(box_rows), &pin[threadIdx.x][1]));
  const std::uint32_t tx = mb_pin(box_bytes * U, &pin[threadIdx.x][2]);
  if (threadIdx.x == 0 && rank == 0) *smid_out = mb_smid();
  if (threadIdx.x == 0) {
    mb_mbar_init(a, 1);
    mb_mbar_init(ack, 1);
    mb_fence_mbar_init();
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<std::uint64_t>(&map)) : "memory");
  }
  __syncthreads();
  if constexpr (false) mb_cluster_sync();
  std::uint32_t ph = 0;
  if (lane == 0 && rank == 0 && warp == 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = kXwDelay + ((h ^ (h >> 15)) & 2047u);
      // first row of this rep's U boxes (resolved before t_s)
      const int y0 = mb_opaque(cold ? static_cast<int>(((static_cast<std::uint64_t>(seed) * reps + rep) * U *
                                                         static_cast<std::uint64_t>(box_rows)) %
                                                        static_cast<std::uint64_t>(rows))
                                    : 0);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      const std::uint64_t t_s = mb_clock();
      if constexpr (U == 0) {
        mb_mbar_arrive(a);
      } else {
        expect_tx(a, tx);
#pragma unroll
        for (int i = 0; i < U; ++i) {
          tma_load3(buf + static_cast<std::uint32_t>(i) * box_bytes, &map, 0, y0 + i * ystep, 0, a);
        }
      }
      const std::uint64_t t_e = mb_clock();
      out[3 * rep + 0] = t_s;
      out[3 * rep + 2] = t_e;
      mb_mbar_wait_to(ack, ph);
      out2[rep] = mb_clock();          // t_ack: all destinations' ACK
      ph ^= 1u;
    }
  } else if (lane == 0 && rank == 0 && warp == 1) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      mb_mbar_spin_to(a, ph);
      out[3 * rep + 1] = mb_clock();
      ph ^= 1u;
      arrive_cluster(ack_leader);
    }
  } else if (false && lane == 0 && U > 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      expect_tx(a, tx);
      mb_mbar_spin_to(a, ph);
      ph ^= 1u;
      arrive_cluster(ack_leader);
    }
  }
  __syncwarp();
  __syncthreads();
  if constexpr (false) mb_cluster_sync();
}


// cta_group::2 forms (the ones the captured 2CTA kernels execute:
// UTMALDG.nD.2CTA and UTMALDG.nD.MULTICAST.2CTA).  Cluster of 2; the leader
// issues and its barrier receives every complete_tx (CUTLASS 2SM loads mask
// the barrier address to the even CTA):
//   MC2 = 0: .2CTA unicast into the leader's smem, expect_tx U*B
//   MC2 = 1: .2CTA multicast mask 0b11 into both CTAs, leader expect_tx 2*U*B
__device__ __forceinline__ void tma_load_2cta(std::uint32_t dst, const CUtensorMap* map, int x, int y,
                                              std::uint32_t mbar) {
#if __CUDA_ARCH__ >= 1000
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
      "[%4];" ::"r"(dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(mbar)
      : "memory");
#endif
}

__device__ __forceinline__ void tma_load_2cta_mc(std::uint32_t dst, const CUtensorMap* map, int x, int y,
                                                 std::uint32_t mbar, std::uint16_t mask) {
#if __CUDA_ARCH__ >= 1000
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
      "[%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(mbar), "h"(mask)
      : "memory");
#endif
}

template <int MC2, int U>
__global__ void __launch_bounds__(64, 1) k_tma2(const __grid_constant__ CUtensorMap map, std::uint64_t* out, int reps,
                                                int box_rows, int rows, int cold, std::uint32_t seed, int probe_cluster,
                                                std::uint32_t* smid_out,
                                               std::uint64_t* out2) {
  if (static_cast<int>(blockIdx.x) / 2 != probe_cluster) return;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t buf = (raw + 1023u) & ~1023u;
  __shared__ __align__(8) std::uint64_t mbar_s[2];
  __shared__ std::uint32_t pin[64][3];
  const std::uint32_t rank = mb_ctarank();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t a = mb_pin(mb_smem_u32(&mbar_s[0]), &pin[threadIdx.x][0]);
  const std::uint32_t ack = mb_smem_u32(&mbar_s[1]);
  const std::uint32_t box_bytes = mb_pin(static_cast<std::uint32_t>(box_rows) * kInner * 2u, &pin[threadIdx.x][1]);
  const int ystep = static_cast<int>(mb_pin(static_cast<std::uint32_t>(box_rows), &pin[threadIdx.x][2]));
  const std::uint32_t tx = box_bytes * U * (MC2 ? 2u : 1u);
  if (threadIdx.x == 0 && rank == 0) *smid_out = mb_smid();
  if (threadIdx.x == 0) {
    mb_mbar_init(a, 1);
    mb_mbar_init(ack, 1);
    mb_fence_mbar_init();
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<std::uint64_t>(&map)) : "memory");
  }
  __syncthreads();
  mb_cluster_sync();
  std::uint32_t ph = 0;
  if (lane == 0 && rank == 0 && warp == 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = kXwDelay + ((h ^ (h >> 15)) & 2047u);
      const int y0 = mb_opaque(cold ? static_cast<int>(((static_cast<std::uint64_t>(seed) * reps + rep) * U *
                                                         static_cast<std::uint64_t>(box_rows)) %
                                                        static_cast<std::uint64_t>(rows))
                                    : 0);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      const std::uint64_t t_s = mb_clock();
      expect_tx(a, tx);
#pragma unroll
      for (int i = 0; i < U; ++i) {
        if constexpr (MC2)
          tma_load_2cta_mc(buf + static_cast<std::uint32_t>(i) * box_bytes, &map, 0, y0 + i * ystep, a, 0x3);
        else
          tma_load_2cta(buf + static_cast<std::uint32_t>(i) * box_bytes, &map, 0, y0 + i * ystep, a);
      }
      const std::uint64_t t_e = mb_clock();
      out[3 * rep + 0] = t_s;
      out[3 * rep + 2] = t_e;
      mb_mbar_wait_to(ack, ph);
      out2[rep] = mb_clock();          // t_ack: all destinations' ACK
      ph ^= 1u;
    }
  } else if (lane == 0 && rank == 0 && warp == 1) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      mb_mbar_spin_to(a, ph);
      out[3 * rep + 1] = mb_clock();
      ph ^= 1u;
      mb_mbar_arrive(ack);
    }
  }
  __syncwarp();
  __syncthreads();
  mb_cluster_sync();
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeTiledFn encoder() {
  void* sym = dlsym(RTLD_DEFAULT, "cuTensorMapEncodeTiled");
  if (!sym) sym = dlsym(dlopen("libcuda.so.1", RTLD_NOW), "cuTensorMapEncodeTiled");
  return reinterpret_cast<EncodeTiledFn>(sym);
}

static CUtensorMap make_map3(void* g, int rows, int box_rows) {
  // rank-3 SWIZZLE_128B box {64 elems, box_rows, 2 slices}
  CUtensorMap m{};
  const cuuint64_t dim[3] = {static_cast<cuuint64_t>(kInner), static_cast<cuuint64_t>(rows), 2};
  const cuuint64_t stride[2] = {static_cast<cuuint64_t>(kInner * 2),
                              static_cast<cuuint64_t>(rows) * kInner * 2};
  const cuuint32_t box[3] = {static_cast<cuuint32_t>(kInner), static_cast<cuuint32_t>(box_rows), 2};
  const cuuint32_t es[3] = {1u, 1u, 1u};
  auto fn = encoder();
  if (!fn || fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, g, dim, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
    std::exit(2);
  }
  return m;
}

static CUtensorMap make_map(void* g, int rows, int box_rows, int swizzle_none) {
  CUtensorMap m{};
  const std::uint32_t inner = swizzle_none ? kInnerNone : kInner;
  const cuuint64_t dim[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(rows)};
  const cuuint64_t stride[1] = {static_cast<cuuint64_t>(inner * 2)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(inner), static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t es[2] = {1u, 1u};
  auto fn = encoder();
  if (!fn || fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, dim, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle_none ? CU_TENSOR_MAP_SWIZZLE_NONE : CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
    std::exit(2);
  }
  return m;
}

static CUtensorMap make_map_rk(void* g, int rank, int rows, int box_rows) {
  // rank-N SWIZZLE_128B box {64 elems, box_rows, 1, ...}: single 2D tile
  CUtensorMap m{};
  cuuint64_t dim[5] = {}, stride[4] = {};
  cuuint32_t box[5] = {}, es[5] = {};
  dim[0] = kInner;
  dim[1] = rows;
  for (int i = 2; i < rank; ++i) dim[i] = 1;
  stride[0] = kInner * 2;
  for (int i = 1; i + 1 < rank; ++i) stride[i] = static_cast<cuuint64_t>(rows) * kInner * 2;
  box[0] = kInner;
  box[1] = box_rows;
  for (int i = 2; i < rank; ++i) box[i] = 1;
  for (int i = 0; i < rank; ++i) es[i] = 1;
  auto fn = encoder();
  if (!fn || fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, g, dim, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
    std::exit(2);
  }
  return m;
}

// rank-protocol load cell (tma_rank_protocol.json): single-plane rank-N
// tile, dc1, hot; same timed window as the rank-2/3 probes.
template <int R, int U>
__global__ void __launch_bounds__(64, 1) k_tmar(const __grid_constant__ CUtensorMap map, std::uint64_t* out,
                                               int reps, int box_rows, int rows, int cold, std::uint32_t seed,
                                               int probe_cluster, std::uint32_t* smid_out,
                                               std::uint64_t* out2) {
  if (static_cast<int>(blockIdx.x) != probe_cluster) return;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t buf = (raw + 1023u) & ~1023u;
  __shared__ __align__(8) std::uint64_t mbar_s[2];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t a = mb_opaque(mb_smem_u32(&mbar_s[0])), ack = mb_opaque(mb_smem_u32(&mbar_s[1]));
  const std::uint32_t ack_leader = mb_opaque(ack);
  __shared__ std::uint32_t pin[64][3];
  const std::uint32_t box_bytes = mb_pin(static_cast<std::uint32_t>(box_rows) * kInner * 2u, &pin[threadIdx.x][0]);
  const int ystep = static_cast<int>(mb_pin(static_cast<std::uint32_t>(box_rows), &pin[threadIdx.x][1]));
  const std::uint32_t tx = mb_pin(box_bytes * U, &pin[threadIdx.x][2]);
  if (threadIdx.x == 0) *smid_out = mb_smid();
  if (threadIdx.x == 0) {
    mb_mbar_init(a, 1);
    mb_mbar_init(ack, 1);
    mb_fence_mbar_init();
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<std::uint64_t>(&map)) : "memory");
  }
  __syncthreads();
  std::uint32_t ph = 0;
  if (lane == 0 && warp == 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = kXwDelay + ((h ^ (h >> 15)) & 2047u);
      const int y0 = mb_opaque(cold ? static_cast<int>(((static_cast<std::uint64_t>(seed) * reps + rep) * U *
                                                         static_cast<std::uint64_t>(box_rows)) %
                                                        static_cast<std::uint64_t>(rows))
                                    : 0);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      const std::uint64_t t_s = mb_clock();
      if constexpr (U == 0) {
        mb_mbar_arrive(a);
      } else {
        expect_tx(a, tx);
#pragma unroll
        for (int i = 0; i < U; ++i) {
          tma_load_rk<R>(buf + static_cast<std::uint32_t>(i) * box_bytes, &map, 0, y0 + i * ystep, a);
        }
      }
      const std::uint64_t t_e = mb_clock();
      out[3 * rep + 0] = t_s;
      out[3 * rep + 2] = t_e;
      mb_mbar_wait_to(ack, ph);
      ph ^= 1u;
    }
  } else if (lane == 0 && warp == 1) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      mb_mbar_spin_to(a, ph);
      out[3 * rep + 1] = mb_clock();
      ph ^= 1u;
      arrive_cluster(ack_leader);
    }
  }
  __syncwarp();
  __syncthreads();
}

using TKern = void (*)(CUtensorMap, std::uint64_t*, int, int, int, int, std::uint32_t, int, std::uint32_t*, std::uint64_t*);
struct TCase {
  int dc, u;
  TKern k;
  int mode;  // 0 plain (.multicast::cluster when dc > 1), 1 .2CTA, 2 .2CTA multicast 0b11
  int swizzle_none;  // 1 = SWIZZLE_NONE map with a 512 B inner box row
  int u6set = 0;     // 1 = tma_32k_protocol cells (U {0,1,2,4,6}, hot only)
  int u3set = 0;     // 1 = tma_64k_protocol cells (U {0,1,2,3}, hot only)
  int rank = 2;      // 3 = rank-3 box {64, box_rows, 2}
};

template <int DC, int... Us>
void tadd(std::vector<TCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({DC, Us, &k_tma<DC, Us>, 0, 0}), ...);
}

template <int MC2, int... Us>
void tadd2(std::vector<TCase>& v, std::integer_sequence<int, Us...>) {
#ifdef TMA_2CTA
  (v.push_back({2, Us, &k_tma2<MC2, Us>, 1 + MC2, 0}), ...);
#endif
}

// SWIZZLE_NONE dc1 cells reuse the k_tma<1,U> kernels; only the tensor
// map differs (512 B inner box row, unswizzled, instead of 128 B).
template <int... Us>
void tadd_swn(std::vector<TCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({1, Us, &k_tma<1, Us>, 0, 1}), ...);
}

// tma_32k_protocol cells: U {0,1,2,4,6} at 16/32 KiB per dc; U=6 x
// 32 KiB + 1 KiB = 193 KiB fits the 200 KiB dynamic smem cap that U=8
// x 32 KiB (257 KiB) exceeded in tma-bytes-formal1.
template <int DC, int... Us>
void tadd_u6(std::vector<TCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({DC, Us, &k_tma<DC, Us>, 0, 0, 1}), ...);
}

// tma_64k_protocol cells: U {0,1,2,3} dc1 hot.  rank3 = k_tma3 with the
// {64, box_rows, 2} box (32768/65536 B); rank2 = the 32 KiB control
// re-measured with the U3 set.
template <int... Us>
void tadd_u3r3(std::vector<TCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({1, Us, &k_tma3<Us>, 0, 0, 0, 1, 3}), ...);
}
template <int... Us>
void tadd_u3r2(std::vector<TCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({1, Us, &k_tma<1, Us>, 0, 0, 0, 1, 2}), ...);
}

static int run_store_cells(const MbArgs& args);
static int run_rank_cells(const MbArgs& args);
static int run_cold3_cells(const MbArgs& args);
static int run_blkseq_cells(const MbArgs& args);

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  if (args.cells == "store") return run_store_cells(args);
  if (args.cells == "rank") return run_rank_cells(args);
  if (args.cells == "cold3") return run_cold3_cells(args);
  if (args.cells == "blkseq") return run_blkseq_cells(args);
  MbWriter w(args, "tma");
  std::vector<TCase> cases;
  tadd<1>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8>{});
  tadd<2>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8>{});
  tadd<4>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8>{});
  tadd<8>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8>{});
  tadd<16>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8>{});
  tadd2<0>(cases, std::integer_sequence<int, 1, 2, 4, 8>{});
  tadd2<1>(cases, std::integer_sequence<int, 1, 2, 4>{});
  // SWIZZLE_NONE 512 B-inner cells removed from the protocol (f7c853a3):
  // a 512 B inner box row exceeds the unswizzled 256 B TMA limit (the
  // .2d load faults) and 64/128 KiB boxes cannot run U >= 2 within the
  // shared-memory cap.  tadd_swn stays compiled for future use.
  // tadd_swn(cases, std::integer_sequence<int, 0, 1, 2, 4, 8>{});
  tadd_u6<1>(cases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tadd_u6<2>(cases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tadd_u6<4>(cases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tadd_u6<8>(cases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tadd_u3r3(cases, std::integer_sequence<int, 0, 1, 2, 3>{});
  tadd_u3r2(cases, std::integer_sequence<int, 0, 1, 2, 3>{});
  const int hot_rows = 4096, cold_rows = 4 * 1024 * 1024;  // 512 KiB / 512 MiB
  void *g_hot = nullptr, *g_cold = nullptr;
  MB_CHECK(cudaMalloc(&g_hot, static_cast<std::size_t>(hot_rows) * kInner * 2));
  MB_CHECK(cudaMalloc(&g_cold, static_cast<std::size_t>(cold_rows) * kInner * 2));
  MB_CHECK(cudaMemset(g_hot, 0x3c, static_cast<std::size_t>(hot_rows) * kInner * 2));
  MB_CHECK(cudaMemset(g_cold, 0x3c, static_cast<std::size_t>(cold_rows) * kInner * 2));
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 3 * args.reps));
  std::uint32_t* d_smid = nullptr;
  MB_CHECK(cudaMalloc(&d_smid, sizeof(std::uint32_t)));
  int nsm = 0;
  MB_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const bool want_32k = args.cells == "32k";
  const bool want_64k = args.cells == "64k";
  for (const auto& c : cases) {
    // exactly one cell set per run: --cells 32k/64k runs only that set,
    // the default runs only the tma_bytes protocol set
    if (!!c.u6set != want_32k) continue;
    if (!!c.u3set != want_64k) continue;
    // protocol cells: dc1 SWIZZLE_128B inner 128 B, box_rows 1..256
    // (128 B..32 KiB); dc1 SWIZZLE_NONE inner 512 B, box_rows 64/128/256
    // (32/64/128 KiB); multicast dc 2/4/8 hot at 512 B/2/8/32 KiB; the
    // 2CTA forms keep the canonical 16 KiB box only.  u6set: hot only,
    // box_rows {128, 256} at every dc.
    std::vector<int> rows_set;
    if (c.u3set && c.rank == 3)
      rows_set = {128, 256};
    else if (c.u3set)
      rows_set = {256};
    else if (c.u6set)
      rows_set = {128, 256};
    else if (c.mode == 0 && c.dc == 1 && !c.swizzle_none)
      rows_set = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    else if (c.mode == 0 && c.dc == 1 && c.swizzle_none)
      rows_set = {64, 128, 256};
    else if (c.mode == 0 && c.dc > 1)
      rows_set = {4, 16, 64, 128, 256};
    else
      rows_set = {128};
    for (int box_rows : rows_set) {
      for (int cold : {0, 1}) {
        if ((c.u6set || c.u3set) && cold) continue;
        // 2CTA forms: the canonical 16 KiB box only
        if (c.mode && box_rows != 128) continue;
        // multicast: hot cells only per the frozen protocol (the old
        // dc>1 b16k cold cell stays runnable)
        if (c.mode == 0 && c.dc > 1 && cold && box_rows != 128) continue;
        // U = 0 is box-independent: run it once per (dc, hot)
        const int u0_row =
            (c.u3set && rows_set[0] != 128) ? rows_set[0] : 128;
        if (c.u == 0 && (box_rows != u0_row || cold)) continue;
        const int inner = c.swizzle_none ? kInnerNone : kInner;
        const int box_bytes = box_rows * inner * 2 * (c.rank == 3 ? 2 : 1);
        // the buffers are sized in kInner rows; the swn map sees them as
        // half as many 512 B rows
        const int rows = (cold ? cold_rows : hot_rows) * kInner / inner;
        const int dyn = c.u * box_bytes + 1024;
        if (dyn > 200 * 1024) continue;
        const char* form = c.mode == 1 ? "2cta." : (c.mode == 2 ? "2ctamc." : "");
        const std::string size = (box_bytes % 1024)
            ? "b" + std::to_string(box_bytes)
            : "b" + std::to_string(box_bytes / 1024) + "k";
        const std::string variant = std::string("tma.") + (c.rank == 3 ? "r3." : "") +
                                    (c.u6set ? "u6set." : "") +
                                    (c.swizzle_none ? "swn." : "") + form +
                                    "dc" + std::to_string(c.dc) + "." + size + "." +
                                    (cold ? "cold" : "hot") + (c.u3set ? ".u3set" : "");
        if (!mb_selected(args, variant)) continue;
        const std::string params = "\"dc\":" + std::to_string(c.dc) + ",\"form\":" + std::to_string(c.mode) +
                                   ",\"box_bytes\":" + std::to_string(box_bytes) +
                                   ",\"cold\":" + std::to_string(cold) +
                                   (c.swizzle_none ? ",\"swizzle\":\"none\"" : ",\"swizzle\":\"128B\"") +
                                   (c.u6set ? ",\"cellset\":\"32k\"" : "") +
                                   (c.u3set ? ",\"cellset\":\"64k\",\"rank\":" + std::to_string(c.rank) : "");
        MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
        if (c.dc > 1) MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        const CUtensorMap map = c.rank == 3
            ? make_map3(cold ? g_cold : g_hot, rows, box_rows)
            : make_map(cold ? g_cold : g_hot, rows, box_rows, c.swizzle_none);
        std::vector<double> k, issue, acks;
        std::string smids;
        bool timed_out = false;
        for (int l = 0; l < args.launches && !timed_out; ++l) {
          // launch l probes from cluster (l * 41) mod (#clusters in one wave)
          const int nclusters = nsm / c.dc;
          const int pcl = (l * 41) % nclusters;
          cudaLaunchConfig_t cfg{};
          cfg.gridDim = dim3(nclusters * c.dc, 1, 1);
          cfg.blockDim = dim3(64, 1, 1);
          cfg.dynamicSmemBytes = dyn;
          cudaLaunchAttribute attr[1];
          attr[0].id = cudaLaunchAttributeClusterDimension;
          attr[0].val.clusterDim.x = c.dc;
          attr[0].val.clusterDim.y = 1;
          attr[0].val.clusterDim.z = 1;
          cfg.attrs = attr;
          cfg.numAttrs = 1;
          if (!g_out2) {
          MB_CHECK(cudaMalloc(&g_out2,
                              sizeof(std::uint64_t) * args.reps));
          MB_CHECK(cudaMemset(g_out2, 0,
                              sizeof(std::uint64_t) * args.reps));
        }
        MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, map, d_out, args.reps, box_rows, rows, cold,
                                      static_cast<std::uint32_t>(l), pcl, d_smid,
                                      g_out2));
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          timed_out = mb_take_err();
          std::uint32_t sm = 0;
          MB_CHECK(cudaMemcpy(&sm, d_smid, 4, cudaMemcpyDeviceToHost));
          smids += (smids.empty() ? "" : ",") + std::to_string(sm);
          std::vector<std::uint64_t> h(3 * static_cast<std::size_t>(args.reps));
          MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
          std::vector<std::uint64_t> h2(static_cast<std::size_t>(args.reps));
          MB_CHECK(cudaMemcpy(h2.data(), g_out2, h2.size() * 8,
                              cudaMemcpyDeviceToHost));
          for (int r = 1; r < args.reps; ++r) {
            k.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 1] - h[3 * r])));
            issue.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 2] - h[3 * r])));
            acks.push_back(static_cast<double>(
                static_cast<std::int64_t>(h2[r] - h[3 * r])));
          }
        }
        if (timed_out) {
          w.fail(variant, c.u, params, "mbarrier wait timed out");
          continue;
        }
        const std::string p2 = params + ",\"smid_per_launch\":[" + smids + "]";
        w.row(variant, c.u, p2, k);
        w.row(variant + ".issue", c.u, p2, issue);
        if (c.dc > 1)
          w.row(variant + ".ack", c.u, p2, acks);
      }
    }
  }
  return w.failures ? 1 : 0;
}

// ---------------------------------------------------------------- cold3
// tma_cold_protocol amendments[0] (130bddcc): rank-3 cold tiles and
// deeper multicast bytes.  (a) dc1 single-plane rank-3 tile
// {64 bf16, rows, 1} via make_map_rk + k_tmar<3,U>; (b) dc2 multicast
// rank-2 via k_tma<2,U>; (c) dc4 multicast via k_tma<4,U>.  U {0,1,2,4};
// U=0 runs once per (dc, cold-state) at the first rows value.
struct Cold3Case {
  int dc, u;
  TKern k;
  int rank;     // 3 = k_tmar<3,U> single-plane; 2 = k_tma<DC,U> multicast
};

template <int... Us>
void c3add_rk3(std::vector<Cold3Case>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({1, Us, &k_tmar<3, Us>, 3}), ...);
}
template <int DC, int... Us>
void c3add_dc(std::vector<Cold3Case>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({DC, Us, &k_tma<DC, Us>, 2}), ...);
}

static int run_cold3_cells(const MbArgs& args) {
  std::vector<Cold3Case> cases;
  c3add_rk3(cases, std::integer_sequence<int, 0, 1, 2, 4>{});
  c3add_dc<2>(cases, std::integer_sequence<int, 0, 1, 2, 4>{});
  c3add_dc<4>(cases, std::integer_sequence<int, 0, 1, 2, 4>{});
  const int hot_rows = 4096, cold_rows = 4 * 1024 * 1024;
  void *g_hot = nullptr, *g_cold = nullptr;
  MB_CHECK(cudaMalloc(&g_hot, static_cast<std::size_t>(hot_rows) * kInner * 2));
  MB_CHECK(cudaMalloc(&g_cold, static_cast<std::size_t>(cold_rows) * kInner * 2));
  MB_CHECK(cudaMemset(g_hot, 0x3c, static_cast<std::size_t>(hot_rows) * kInner * 2));
  MB_CHECK(cudaMemset(g_cold, 0x3c, static_cast<std::size_t>(cold_rows) * kInner * 2));
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 3 * args.reps));
  std::uint32_t* d_smid = nullptr;
  MB_CHECK(cudaMalloc(&d_smid, sizeof(std::uint32_t)));
  int nsm = 0;
  MB_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  MbWriter w(args, "tma");
  for (const auto& c : cases) {
    std::vector<int> rows_set;
    if (c.rank == 3)
      rows_set = {4, 12, 16, 32, 64, 128, 256};
    else if (c.dc == 2)
      rows_set = {3, 6, 16, 64};
    else
      rows_set = {3, 6, 32};
    for (int box_rows : rows_set) {
      for (int cold : {0, 1}) {
        // U = 0 is box-independent: once per (dc, cold-state)
        if (c.u == 0 && box_rows != rows_set[0]) continue;
        const int box_bytes = box_rows * kInner * 2;
        const int rows = cold ? cold_rows : hot_rows;
        const int dyn = c.u * box_bytes + 1024;
        if (dyn > 200 * 1024) continue;
        const std::string size = (box_bytes % 1024)
            ? "b" + std::to_string(box_bytes)
            : "b" + std::to_string(box_bytes / 1024) + "k";
        const std::string variant =
            std::string("tma.") + (c.rank == 3 ? "rk3." : "") +
            "dc" + std::to_string(c.dc) + "." + size + "." +
            (cold ? "cold" : "hot");
        if (!mb_selected(args, variant)) continue;
        const std::string params =
            "\"dc\":" + std::to_string(c.dc) +
            ",\"form\":0,\"box_bytes\":" +
            std::to_string(box_bytes) +
            ",\"cold\":" + std::to_string(cold) +
            ",\"swizzle\":\"128B\",\"cellset\":\"cold3\"" +
            (c.rank == 3 ? ",\"rank\":3" : "");
        MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
        if (c.dc > 1)
          MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        const CUtensorMap map = c.rank == 3
            ? make_map_rk(cold ? g_cold : g_hot, 3, rows, box_rows)
            : make_map(cold ? g_cold : g_hot, rows, box_rows, 0);
        std::vector<double> k, issue, acks;
        std::string smids;
        bool timed_out = false;
        for (int l = 0; l < args.launches && !timed_out; ++l) {
          const int nclusters = nsm / c.dc;
          const int pcl = (l * 41) % nclusters;
          cudaLaunchConfig_t cfg{};
          cfg.gridDim = dim3(nclusters * c.dc, 1, 1);
          cfg.blockDim = dim3(64, 1, 1);
          cfg.dynamicSmemBytes = dyn;
          cudaLaunchAttribute attr[1];
          attr[0].id = cudaLaunchAttributeClusterDimension;
          attr[0].val.clusterDim.x = c.dc;
          attr[0].val.clusterDim.y = 1;
          attr[0].val.clusterDim.z = 1;
          cfg.attrs = attr;
          cfg.numAttrs = 1;
          if (!g_out2) {
          MB_CHECK(cudaMalloc(&g_out2,
                              sizeof(std::uint64_t) * args.reps));
          MB_CHECK(cudaMemset(g_out2, 0,
                              sizeof(std::uint64_t) * args.reps));
        }
        MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, map, d_out, args.reps, box_rows, rows, cold,
                                      static_cast<std::uint32_t>(l), pcl, d_smid,
                                      g_out2));
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          timed_out = mb_take_err();
          std::uint32_t sm = 0;
          MB_CHECK(cudaMemcpy(&sm, d_smid, 4, cudaMemcpyDeviceToHost));
          smids += (smids.empty() ? "" : ",") + std::to_string(sm);
          std::vector<std::uint64_t> h(3 * static_cast<std::size_t>(args.reps));
          MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
          std::vector<std::uint64_t> h2(static_cast<std::size_t>(args.reps));
          MB_CHECK(cudaMemcpy(h2.data(), g_out2, h2.size() * 8,
                              cudaMemcpyDeviceToHost));
          for (int r = 1; r < args.reps; ++r) {
            k.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 1] - h[3 * r])));
            issue.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 2] - h[3 * r])));
            acks.push_back(static_cast<double>(
                static_cast<std::int64_t>(h2[r] - h[3 * r])));
          }
        }
        if (timed_out) {
          w.fail(variant, c.u, params, "mbarrier wait timed out");
          continue;
        }
        const std::string p2 = params + ",\"smid_per_launch\":[" + smids + "]";
        w.row(variant, c.u, p2, k);
        w.row(variant + ".issue", c.u, p2, issue);
        if (c.dc > 1)
          w.row(variant + ".ack", c.u, p2, acks);
      }
    }
  }
  return w.failures ? 1 : 0;
}

// ---------------------------------------------------------------- tmast
// Store-completion probe (tma_store_protocol.json): one warp,
// smem source pre-filled and fenced; t_s; U back-to-back stores of B
// bytes each to distinct global boxes; commit_group; wait_group 0;
// t_w.  U = 0: t_s; commit; wait; t_w.
__device__ __forceinline__ void store_tensor_2d(std::uint32_t src, const CUtensorMap* map, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
          reinterpret_cast<std::uint64_t>(map)),
      "r"(x), "r"(y), "r"(src)
      : "memory");
}

__device__ __forceinline__ void store_tensor_2d_red(std::uint32_t src, const CUtensorMap* map, int x, int y) {
  asm volatile(
      "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
          reinterpret_cast<std::uint64_t>(map)),
      "r"(x), "r"(y), "r"(src)
      : "memory");
}

__device__ __forceinline__ void store_bulk(std::uint64_t dst, std::uint32_t src, std::uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void commit_group() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_group_0() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// OP: 0 UTMASTG, 1 UTMAREDG (f32), 2 UBLKCP.G.S.  Global side is a flat
// L2-resident buffer; box i lands at dst + i * box_bytes.
template <int OP, int U>
__global__ void __launch_bounds__(64, 1) k_tmast(const __grid_constant__ CUtensorMap map, unsigned char* dst,
                                                 std::uint64_t* out, int reps, int box_bytes, std::uint32_t seed,
                                                 int probe_cluster, std::uint32_t* smid_out,
                                               std::uint64_t* out2) {
  if (static_cast<int>(blockIdx.x) != probe_cluster) return;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t buf = (raw + 1023u) & ~1023u;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __shared__ std::uint32_t pin[64][2];
  const std::uint32_t bb = mb_pin(static_cast<std::uint32_t>(box_bytes), &pin[threadIdx.x][0]);
  const int ystep = OP == 2 ? 0 : static_cast<int>(mb_pin(bb / 128u, &pin[threadIdx.x][1]));
  if (threadIdx.x == 0 && blockIdx.x == 0) *smid_out = mb_smid();
  // pre-fill the smem source and make it visible to the async proxy.
  // buf is a shared-space address (asm operand), not a generic pointer:
  // write through smem_raw at the same byte offset.
  for (int i = threadIdx.x; i < U * static_cast<int>(bb) / 4; i += 64)
    reinterpret_cast<std::uint32_t*>(smem_raw)[(buf - raw) / 4 + i] = seed + i;
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (lane == 0 && warp == 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = kXwDelay + ((h ^ (h >> 15)) & 2047u);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      const std::uint64_t t_s = mb_clock();
#pragma unroll
      for (int i = 0; i < U; ++i) {
        if constexpr (OP == 0)
          store_tensor_2d(buf + static_cast<std::uint32_t>(i) * bb, &map, 0, i * ystep);
        else if constexpr (OP == 1)
          store_tensor_2d_red(buf + static_cast<std::uint32_t>(i) * bb, &map, 0, i * ystep);
        else
          store_bulk(reinterpret_cast<std::uint64_t>(dst) + static_cast<std::uint64_t>(i) * bb,
                     buf + static_cast<std::uint32_t>(i) * bb, bb);
      }
      commit_group();
      wait_group_0();
      out[2 * rep] = t_s;
      out[2 * rep + 1] = mb_clock();
    }
  }
  __syncwarp();
  __syncthreads();
}

static CUtensorMap make_store_map(void* g, int rows, int box_rows, int f32) {
  CUtensorMap m{};
  const std::uint32_t inner = f32 ? 32u : kInner;  // f32 elems per 128 B row
  const cuuint64_t dim[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(rows)};
  const cuuint64_t stride[1] = {static_cast<cuuint64_t>(inner * (f32 ? 4 : 2))};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(inner), static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t es[2] = {1u, 1u};
  auto fn = encoder();
  if (!fn || fn(&m, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, dim, stride,
                box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
    std::exit(2);
  }
  return m;
}

using TStKern = void (*)(CUtensorMap, unsigned char*, std::uint64_t*, int, int, std::uint32_t, int, std::uint32_t*, std::uint64_t*);
struct TStCase {
  int op, u;
  TStKern k;
};

template <int OP, int... Us>
void tstadd(std::vector<TStCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({OP, Us, &k_tmast<OP, Us>}), ...);
}

static int run_store_cells(const MbArgs& args) {
  std::vector<TStCase> cases;
  tstadd<0>(cases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tstadd<1>(cases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tstadd<2>(cases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  const int dst_rows = 4 * 1024 * 1024;  // 512 MiB flat dst
  void* g_dst = nullptr;
  MB_CHECK(cudaMalloc(&g_dst, static_cast<std::size_t>(dst_rows) * kInner * 2));
  MB_CHECK(cudaMemset(g_dst, 0x3c, static_cast<std::size_t>(dst_rows) * kInner * 2));
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * args.reps));
  std::uint32_t* d_smid = nullptr;
  MB_CHECK(cudaMalloc(&d_smid, sizeof(std::uint32_t)));
  int nsm = 0;
  MB_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  MbWriter w(args, "tma");
  const char* opname[3] = {"utmastg", "utma_redg", "ublkcpgs"};
  for (const auto& c : cases) {
    for (int box_bytes : {512, 2048, 8192, 16384, 32768}) {
      const int box_rows = box_bytes / 128;
      const int dyn = c.u * box_bytes + 1024;
      if (dyn > 200 * 1024) continue;  // smem cap -> cell missing
      const std::string variant = std::string("tmast.") + opname[c.op] + ".b" +
                                  std::to_string(box_bytes / 1024) + "k.hot";
      if (!mb_selected(args, variant)) continue;
      const std::string params = "\"op\":\"" + std::string(opname[c.op]) + "\",\"box_bytes\":" +
                                 std::to_string(box_bytes) + ",\"cold\":0,\"swizzle\":\"128B\",\"cellset\":\"store\"";
      MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
      const CUtensorMap map = make_store_map(g_dst, c.u * box_rows + 64, box_rows, c.op == 1);
      std::vector<double> k;
      std::string smids;
      bool timed_out = false;
      for (int l = 0; l < args.launches && !timed_out; ++l) {
        const int pcl = (l * 41) % nsm;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(nsm, 1, 1);
        cfg.blockDim = dim3(64, 1, 1);
        cfg.dynamicSmemBytes = dyn;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        if (!g_out2) {
          MB_CHECK(cudaMalloc(&g_out2,
                              sizeof(std::uint64_t) * args.reps));
          MB_CHECK(cudaMemset(g_out2, 0,
                              sizeof(std::uint64_t) * args.reps));
        }
        MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, map, reinterpret_cast<unsigned char*>(g_dst), d_out, args.reps,
                                    box_bytes, static_cast<std::uint32_t>(l), pcl, d_smid,
                                      g_out2));
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        timed_out = mb_take_err();
        std::uint32_t sm = 0;
        MB_CHECK(cudaMemcpy(&sm, d_smid, 4, cudaMemcpyDeviceToHost));
        smids += (smids.empty() ? "" : ",") + std::to_string(sm);
        std::vector<std::uint64_t> h(2 * static_cast<std::size_t>(args.reps));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r)
          k.push_back(static_cast<double>(static_cast<std::int64_t>(h[2 * r + 1] - h[2 * r])));
      }
      if (timed_out) {
        w.fail(variant, c.u, params, "store wait timed out");
        continue;
      }
      const std::string p2 = params + ",\"smid_per_launch\":[" + smids + "]";
      w.row(variant, c.u, p2, k);
    }
  }
  return 0;
}

// ===== rank protocol (tma_rank_protocol.json) =====
// rank-3/4/5 single-plane tile store + the load cells that use the
// k_tmar kernel above.  Stores: same timed window as k_tmast but the
// opcode is cp.async.bulk.tensor.<N>d for the cell's rank.
__device__ __forceinline__ void store_tensor_3d(std::uint32_t src, const CUtensorMap* map, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];" ::"l"(
          reinterpret_cast<std::uint64_t>(map)),
      "r"(x), "r"(y), "r"(0), "r"(src)
      : "memory");
}
__device__ __forceinline__ void store_tensor_4d(std::uint32_t src, const CUtensorMap* map, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4}], [%5];" ::"l"(
          reinterpret_cast<std::uint64_t>(map)),
      "r"(x), "r"(y), "r"(0), "r"(0), "r"(src)
      : "memory");
}
__device__ __forceinline__ void store_tensor_5d(std::uint32_t src, const CUtensorMap* map, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];" ::"l"(
          reinterpret_cast<std::uint64_t>(map)),
      "r"(x), "r"(y), "r"(0), "r"(0), "r"(0), "r"(src)
      : "memory");
}
template <int R>
__device__ __forceinline__ void store_tensor_rk(std::uint32_t src, const CUtensorMap* map, int x, int y) {
  if constexpr (R == 2)
    store_tensor_2d(src, map, x, y);
  else if constexpr (R == 3)
    store_tensor_3d(src, map, x, y);
  else if constexpr (R == 4)
    store_tensor_4d(src, map, x, y);
  else
    store_tensor_5d(src, map, x, y);
}

template <int R, int U>
__global__ void __launch_bounds__(64, 1) k_tmast_r(const __grid_constant__ CUtensorMap map, unsigned char* dst,
                                                 std::uint64_t* out, int reps, int box_bytes, std::uint32_t seed,
                                                 int probe_cluster, std::uint32_t* smid_out,
                                               std::uint64_t* out2) {
  if (static_cast<int>(blockIdx.x) != probe_cluster) return;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t buf = (raw + 1023u) & ~1023u;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __shared__ std::uint32_t pin[64][2];
  const std::uint32_t bb = mb_pin(static_cast<std::uint32_t>(box_bytes), &pin[threadIdx.x][0]);
  const int ystep = static_cast<int>(mb_pin(bb / 128u, &pin[threadIdx.x][1]));
  if (threadIdx.x == 0 && blockIdx.x == 0) *smid_out = mb_smid();
  for (int i = threadIdx.x; i < U * static_cast<int>(bb) / 4; i += 64)
    reinterpret_cast<std::uint32_t*>(smem_raw)[(buf - raw) / 4 + i] = seed + i;
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (lane == 0 && warp == 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = kXwDelay + ((h ^ (h >> 15)) & 2047u);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      const std::uint64_t t_s = mb_clock();
#pragma unroll
      for (int i = 0; i < U; ++i) {
        store_tensor_rk<R>(buf + static_cast<std::uint32_t>(i) * bb, &map, 0, i * ystep);
      }
      commit_group();
      wait_group_0();
      out[2 * rep] = t_s;
      out[2 * rep + 1] = mb_clock();
    }
  }
  __syncwarp();
  __syncthreads();
}

using TStRKern = void (*)(CUtensorMap, unsigned char*, std::uint64_t*, int, int, std::uint32_t, int, std::uint32_t*, std::uint64_t*);
struct TStRCase {
  int rank, u;
  TStRKern k;
};
using TLdKern = void (*)(CUtensorMap, std::uint64_t*, int, int, int, int, std::uint32_t, int, std::uint32_t*, std::uint64_t*);
struct TLdRCase {
  int rank, u;
  TLdKern k;
};

template <int R, int... Us>
void tmradd_ld(std::vector<TLdRCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({R, Us, &k_tmar<R, Us>}), ...);
}
template <int R, int... Us>
void tmradd_st(std::vector<TStRCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({R, Us, &k_tmast_r<R, Us>}), ...);
}

static int run_rank_cells(const MbArgs& args) {
  std::vector<TLdRCase> ldcases;
  tmradd_ld<2>(ldcases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tmradd_ld<3>(ldcases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tmradd_ld<4>(ldcases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tmradd_ld<5>(ldcases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  std::vector<TStRCase> stcases;
  tmradd_st<2>(stcases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tmradd_st<3>(stcases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tmradd_st<4>(stcases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  tmradd_st<5>(stcases, std::integer_sequence<int, 0, 1, 2, 4, 6>{});
  const int hot_rows = 4096;                        // 512 KiB, L2-resident
  const int dst_rows = 4 * 1024 * 1024;             // 512 MiB flat dst
  void *g_hot = nullptr, *g_dst = nullptr;
  MB_CHECK(cudaMalloc(&g_hot, static_cast<std::size_t>(hot_rows) * kInner * 2));
  MB_CHECK(cudaMalloc(&g_dst, static_cast<std::size_t>(dst_rows) * kInner * 2));
  MB_CHECK(cudaMemset(g_hot, 0x3c, static_cast<std::size_t>(hot_rows) * kInner * 2));
  MB_CHECK(cudaMemset(g_dst, 0x3c, static_cast<std::size_t>(dst_rows) * kInner * 2));
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 3 * args.reps));
  std::uint32_t* d_smid = nullptr;
  MB_CHECK(cudaMalloc(&d_smid, sizeof(std::uint32_t)));
  int nsm = 0;
  MB_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  MbWriter w(args, "tma");
  // loads: box_rows {32,128,256} -> 4/16/32 KiB; U0 once per rank at 128
  for (const auto& c : ldcases) {
    for (int box_rows : {32, 128, 256}) {
      if (c.u == 0 && box_rows != 128) continue;
      const int box_bytes = box_rows * kInner * 2;
      const int dyn = c.u * box_bytes + 1024;
      if (dyn > 200 * 1024) continue;
      const std::string variant = "tmar.ld.r" + std::to_string(c.rank) + ".b" +
                                  std::to_string(box_bytes / 1024) + "k.hot";
      if (!mb_selected(args, variant)) continue;
      const std::string params = "\"op\":\"utmaldg\",\"rank\":" + std::to_string(c.rank) +
                                 ",\"box_bytes\":" + std::to_string(box_bytes) +
                                 ",\"cold\":0,\"swizzle\":\"128B\",\"cellset\":\"rank\"";
      MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
      const CUtensorMap map = make_map_rk(g_hot, c.rank, hot_rows, box_rows);
      std::vector<double> k, issue;
      std::string smids;
      bool timed_out = false;
      for (int l = 0; l < args.launches && !timed_out; ++l) {
        const int pcl = (l * 41) % nsm;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(nsm, 1, 1);
        cfg.blockDim = dim3(64, 1, 1);
        cfg.dynamicSmemBytes = dyn;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        if (!g_out2) {
          MB_CHECK(cudaMalloc(&g_out2,
                              sizeof(std::uint64_t) * args.reps));
          MB_CHECK(cudaMemset(g_out2, 0,
                              sizeof(std::uint64_t) * args.reps));
        }
        MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, map, d_out, args.reps, box_rows, hot_rows, 0,
                                    static_cast<std::uint32_t>(l), pcl, d_smid,
                                      g_out2));
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        timed_out = mb_take_err();
        std::uint32_t sm = 0;
        MB_CHECK(cudaMemcpy(&sm, d_smid, 4, cudaMemcpyDeviceToHost));
        smids += (smids.empty() ? "" : ",") + std::to_string(sm);
        std::vector<std::uint64_t> h(3 * static_cast<std::size_t>(args.reps));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r) {
          k.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 1] - h[3 * r])));
          issue.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 2] - h[3 * r])));
        }
      }
      if (timed_out) {
        w.fail(variant, c.u, params, "load wait timed out");
        continue;
      }
      const std::string p2 = params + ",\"smid_per_launch\":[" + smids + "]";
      w.row(variant, c.u, p2, k);
      w.row(variant + ".issue", c.u, p2, issue);
    }
  }
  // stores: box_rows {16,64,256} -> 2/8/32 KiB
  for (const auto& c : stcases) {
    for (int box_rows : {16, 64, 256}) {
      if (c.u == 0 && box_rows != 64) continue;
      const int box_bytes = box_rows * kInner * 2;
      const int dyn = c.u * box_bytes + 1024;
      if (dyn > 200 * 1024) continue;
      const std::string variant = "tmar.st.r" + std::to_string(c.rank) + ".b" +
                                  std::to_string(box_bytes / 1024) + "k.hot";
      if (!mb_selected(args, variant)) continue;
      const std::string params = "\"op\":\"utmastg\",\"rank\":" + std::to_string(c.rank) +
                                 ",\"box_bytes\":" + std::to_string(box_bytes) +
                                 ",\"cold\":0,\"swizzle\":\"128B\",\"cellset\":\"rank\"";
      MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
      const CUtensorMap map = make_map_rk(g_dst, c.rank, c.u * box_rows + 64, box_rows);
      std::vector<double> k;
      std::string smids;
      bool timed_out = false;
      for (int l = 0; l < args.launches && !timed_out; ++l) {
        const int pcl = (l * 41) % nsm;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(nsm, 1, 1);
        cfg.blockDim = dim3(64, 1, 1);
        cfg.dynamicSmemBytes = dyn;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        if (!g_out2) {
          MB_CHECK(cudaMalloc(&g_out2,
                              sizeof(std::uint64_t) * args.reps));
          MB_CHECK(cudaMemset(g_out2, 0,
                              sizeof(std::uint64_t) * args.reps));
        }
        MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, map, reinterpret_cast<unsigned char*>(g_dst), d_out, args.reps,
                                    box_bytes, static_cast<std::uint32_t>(l), pcl, d_smid,
                                      g_out2));
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        timed_out = mb_take_err();
        std::uint32_t sm = 0;
        MB_CHECK(cudaMemcpy(&sm, d_smid, 4, cudaMemcpyDeviceToHost));
        smids += (smids.empty() ? "" : ",") + std::to_string(sm);
        std::vector<std::uint64_t> h(2 * static_cast<std::size_t>(args.reps));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r)
          k.push_back(static_cast<double>(static_cast<std::int64_t>(h[2 * r + 1] - h[2 * r])));
      }
      if (timed_out) {
        w.fail(variant, c.u, params, "store wait timed out");
        continue;
      }
      const std::string p2 = params + ",\"smid_per_launch\":[" + smids + "]";
      w.row(variant, c.u, p2, k);
    }
  }
  return 0;
}

// ------------------------------------------------------------- blkseq
// bulk_store_frontend_protocol.json (round 9): does the issuing warp
// wait for a bulk store's completion at commit_group?  Per element i:
// [STS=2: 2x STS.128 into the element's smem chunk] ->
// fence.proxy.async.shared::cta -> OP(B bytes -> distinct global chunk)
// -> cp.async.bulk.commit_group; U elements back to back.  Three clocks:
// t_s before the first element, t_i after the last commit, t_w after
// cp.async.bulk.wait_group 0.  U = 0 control: window = fence + commit.
// OP issuer: elect.sync inside the converged issuing warp (k_factor
// lesson: the warp predicate stays an outer branch, never folded into
// the elect predicate).
__device__ __forceinline__ void blkred_f32(std::uint64_t dst, std::uint32_t src, std::uint32_t bytes) {
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" ::"l"(dst),
      "r"(src), "r"(bytes)
      : "memory");
}

// one STS.128 warp store (16 B per lane) into the element's smem chunk.
__device__ __forceinline__ void sts128(std::uint32_t dst, std::uint32_t v) {
  asm volatile("st.shared::cta.v4.b32 [%0], {%1, %1, %1, %1};" ::"r"(dst), "r"(v)
               : "memory");
}

__device__ __forceinline__ void fence_async_cta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// OP: 0 UBLKCP, 1 UBLKRED(f32).  bytes is a kernel argument: the SASS
// window does not depend on it (the PTX operand is a register).
template <int OP, int U, int STS>
__global__ void __launch_bounds__(32, 1) k_blkseq(unsigned char* dst, std::uint64_t* out, int reps,
                                                  int bytes, std::uint32_t seed, int probe_cluster,
                                                  std::uint32_t* smid_out,
                                               std::uint64_t* out2) {
  if (static_cast<int>(blockIdx.x) != probe_cluster) return;
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t buf = (raw + 1023u) & ~1023u;
  const int warp = threadIdx.x >> 5;
  __shared__ std::uint32_t pin[32][1];
  // 64 KiB smem source window; element i reuses chunk (i mod chunks).
  const std::uint32_t chunks = static_cast<std::uint32_t>(65536) / static_cast<std::uint32_t>(bytes);
  const std::uint32_t bb = mb_pin(static_cast<std::uint32_t>(bytes), &pin[threadIdx.x][0]);
  if (threadIdx.x == 0 && blockIdx.x == 0) *smid_out = mb_smid();
  for (int i = threadIdx.x; i < 65536 / 4; i += 32)
    reinterpret_cast<std::uint32_t*>(smem_raw)[(buf - raw) / 4 + i] = seed + i;
  __syncthreads();
  if (warp == 0) {
    const std::uint32_t el = mb_elect_one();
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = kXwDelay + ((h ^ (h >> 15)) & 2047u);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      std::uint64_t t_s = 0;
      if (el) t_s = mb_clock();
#pragma unroll
      for (int i = 0; i < U; ++i) {
        const std::uint32_t s = buf + (static_cast<std::uint32_t>(i) % chunks) * bb;
        if constexpr (STS == 2) {
          sts128(s, seed + static_cast<std::uint32_t>(i));
          sts128(s + 512u, seed + static_cast<std::uint32_t>(i));
        }
        if (el) {
          fence_async_cta();
          if constexpr (OP == 1)
            blkred_f32(reinterpret_cast<std::uint64_t>(dst) + static_cast<std::uint64_t>(i) * bb, s, bb);
          else
            store_bulk(reinterpret_cast<std::uint64_t>(dst) + static_cast<std::uint64_t>(i) * bb, s, bb);
          commit_group();
        }
      }
      if constexpr (U == 0) {
        if (el) {
          fence_async_cta();
          commit_group();
        }
      }
      if (el) {
        const std::uint64_t t_i = mb_clock();
        wait_group_0();
        const std::uint64_t t_w = mb_clock();
        out[3 * rep + 0] = t_s;
        out[3 * rep + 1] = t_i;
        out[3 * rep + 2] = t_w;
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

using TBkKern = void (*)(unsigned char*, std::uint64_t*, int, int, std::uint32_t, int, std::uint32_t*, std::uint64_t*);
struct TBkCase {
  int op, u, sts;
  TBkKern k;
};

template <int OP, int STS, int... Us>
void tbkadd(std::vector<TBkCase>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({OP, Us, STS, &k_blkseq<OP, Us, STS>}), ...);
}

static int run_blkseq_cells(const MbArgs& args) {
  std::vector<TBkCase> cases;
  tbkadd<0, 0>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8, 16, 32>{});
  tbkadd<0, 2>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8, 16, 32>{});
  tbkadd<1, 0>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8, 16, 32>{});
  tbkadd<1, 2>(cases, std::integer_sequence<int, 0, 1, 2, 4, 8, 16, 32>{});
  // flat L2-resident destination: max U*B = 32 x 16 KiB = 512 KiB
  void* g_dst = nullptr;
  MB_CHECK(cudaMalloc(&g_dst, 512 * 1024));
  MB_CHECK(cudaMemset(g_dst, 0x3c, 512 * 1024));
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 3 * args.reps));
  std::uint32_t* d_smid = nullptr;
  MB_CHECK(cudaMalloc(&d_smid, sizeof(std::uint32_t)));
  int nsm = 0;
  MB_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  MbWriter w(args, "tma");
  const char* opname[2] = {"ublkcp", "ublkred"};
  for (const auto& c : cases) {
    for (int bytes : {1024, 4096, 16384}) {
      const std::string variant = std::string("k_blkseq.") + opname[c.op] + ".b" +
                                  std::to_string(bytes / 1024) + "k.sts" + std::to_string(c.sts);
      if (!mb_selected(args, variant)) continue;
      const std::string params = "\"op\":\"" + std::string(opname[c.op]) + "\",\"bytes\":" +
                                 std::to_string(bytes) + ",\"sts\":" + std::to_string(c.sts) +
                                 ",\"cellset\":\"blkseq\"";
      MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, 66560));
      std::vector<double> issue, total;
      std::string smids;
      bool timed_out = false;
      for (int l = 0; l < args.launches && !timed_out; ++l) {
        const int pcl = (l * 41) % nsm;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(nsm, 1, 1);
        cfg.blockDim = dim3(32, 1, 1);
        cfg.dynamicSmemBytes = 66560;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        if (!g_out2) {
          MB_CHECK(cudaMalloc(&g_out2,
                              sizeof(std::uint64_t) * args.reps));
          MB_CHECK(cudaMemset(g_out2, 0,
                              sizeof(std::uint64_t) * args.reps));
        }
        MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, reinterpret_cast<unsigned char*>(g_dst), d_out, args.reps,
                                    bytes, static_cast<std::uint32_t>(l), pcl, d_smid,
                                      g_out2));
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        timed_out = mb_take_err();
        std::uint32_t sm = 0;
        MB_CHECK(cudaMemcpy(&sm, d_smid, 4, cudaMemcpyDeviceToHost));
        smids += (smids.empty() ? "" : ",") + std::to_string(sm);
        std::vector<std::uint64_t> h(3 * static_cast<std::size_t>(args.reps));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r) {
          issue.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 1] - h[3 * r])));
          total.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 2] - h[3 * r])));
        }
      }
      if (timed_out) {
        w.fail(variant, c.u, params, "bulk store wait timed out");
        continue;
      }
      const std::string p2 = params + ",\"smid_per_launch\":[" + smids + "]";
      w.row(variant + ".issue", c.u, p2, issue);
      w.row(variant + ".total", c.u, p2, total);
    }
  }
  return w.failures ? 1 : 0;
}
