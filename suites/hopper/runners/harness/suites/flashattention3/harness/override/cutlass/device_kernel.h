/***************************************************************************************************
 * Local measurement-only override of CUTLASS device_kernel.h for the FA3 delivery harness.
 *
 * The public CUTLASS entry points (Type2Type / GetUnderlyingKernel / Kernel / Kernel2 /
 * device_kernel, CUTLASS_GRID_CONSTANT macro logic) are preserved 1:1 from upstream.  The 3.x
 * entry point additionally records per-CTA/per-warp clock64/globaltimer markers around
 * Operator::operator(), matching the CUTLASS delivery suite override contract:
 * layout MAX_LAUNCHES=256 / MAX_CTAS=1024 / MAX_WARPS=16 / FIELDS=7,
 * field order [start_clock64, end_clock64, start_globaltimer, end_globaltimer, sm_id,
 * cluster_dim_packed, cluster_cta_rank], slot = grid_id % 256.  256 launch slots hold
 * multi-kernel formal campaigns (bwd: ~4 device_kernel launches x 27 iterations).
 *
 * Multi-TU support: FA3 links many nvcc TUs into one .so.  Each TU that includes this header
 * gets its own `static __device__` marker symbols plus a same-TU host setter, and a static
 * registrar object enrolls the setter into the cross-TU registry (fa3_marker_registry.h).
 * With FA3_DELIVERY_MARKER_OVERRIDE=1 the build places this directory ahead of CUTLASS on the
 * include path, so every TU sees this header instead of the upstream one.  When the marker
 * pointer is nullptr the kernel body is identical to upstream (marker stores are skipped).
 **************************************************************************************************/
#pragma once

#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/arch/synclog.hpp>
#include <cutlass/platform/platform.h>
#include "fa3_marker_registry.h"

#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7))) && !CUTLASS_CLANG_CUDA
#  define CUTLASS_GRID_CONSTANT_SUPPORTED
#endif
#if defined(CUTLASS_GRID_CONSTANT_SUPPORTED) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#  define CUTLASS_GRID_CONSTANT_ENABLED
#endif
#if !defined(CUTLASS_GRID_CONSTANT)
#  if defined(CUTLASS_GRID_CONSTANT_ENABLED)
#    define CUTLASS_GRID_CONSTANT __grid_constant__
#  else
#    define CUTLASS_GRID_CONSTANT
#  endif
#endif

namespace cutlass {

template <typename T> struct Type2Type { using type = T; };
template <typename T> struct GetUnderlyingKernel : public Type2Type<T> {};
template <uint64_t shader_guid, unsigned index, template <uint64_t, unsigned> class Wrapper>
struct GetUnderlyingKernel<Wrapper<shader_guid, index>> : public Wrapper<shader_guid, index> {};
template <typename T> using GetUnderlyingKernel_t = typename GetUnderlyingKernel<T>::type;

#if defined(__CUDACC__)
static __device__ unsigned long long* cutlass_single_cta_marker_ptr = nullptr;
static __device__ unsigned long long* cutlass_single_cta_marker_grid_ids = nullptr;
static __device__ unsigned long long* cutlass_single_cta_marker_grids = nullptr;
static constexpr unsigned long long cutlass_marker_max_ctas = 1024;
static constexpr unsigned long long cutlass_marker_max_warps = 16;
static constexpr unsigned long long cutlass_marker_fields = 7;
static constexpr unsigned long long cutlass_marker_max_launches = 256;
#endif  // __CUDACC__

#if defined(__CUDACC__)
template <typename Operator>
CUTLASS_GLOBAL
void Kernel(typename Operator::Params params) {
  extern __shared__ int SharedStorageBase[];
  typename Operator::SharedStorage* shared_storage =
      reinterpret_cast<typename Operator::SharedStorage*>(SharedStorageBase);
  Operator op;
  op(params, *shared_storage);
  cutlass::arch::synclog_print();
}

template <typename Operator>
CUTLASS_GLOBAL
void Kernel2(typename Operator::Params params) {
  extern __shared__ int SharedStorageBase[];
  typename Operator::SharedStorage* shared_storage =
      reinterpret_cast<typename Operator::SharedStorage*>(SharedStorageBase);
  Operator::invoke(params, *shared_storage);
  cutlass::arch::synclog_print();
}
#endif  // __CUDACC__

#if defined(__CUDACC__)
template <typename Operator>
CUTLASS_GLOBAL
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params) {
  extern __shared__ char smem[];
  int const lane = int(threadIdx.x) & 31;
  int const warp = int(threadIdx.x) >> 5;
  unsigned long long const cta = static_cast<unsigned long long>(blockIdx.x) +
      static_cast<unsigned long long>(gridDim.x) *
      (static_cast<unsigned long long>(blockIdx.y) +
       static_cast<unsigned long long>(gridDim.y) * static_cast<unsigned long long>(blockIdx.z));
  unsigned long long start_clock = 0;
  unsigned long long start_timer = 0;
  unsigned long long grid_id = 0;
  unsigned long long launch_slot = 0;
  unsigned int sm_id = 0;
  unsigned int cluster_ncta = 1;
  unsigned int cluster_rank = 0;
  unsigned int cluster_nx = 1;
  unsigned int cluster_ny = 1;
  unsigned int cluster_nz = 1;
  if (cutlass_single_cta_marker_grid_ids != nullptr) {
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_id));
    launch_slot = grid_id % cutlass_marker_max_launches;
  }
  if (lane == 0 && cutlass_single_cta_marker_ptr != nullptr &&
      cta < cutlass_marker_max_ctas && warp < cutlass_marker_max_warps) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_timer));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_nx));
    asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(cluster_ny));
    asm volatile("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(cluster_nz));
    cluster_ncta = cluster_nx * cluster_ny * cluster_nz;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cluster_rank));
#endif
    unsigned long long cluster_dim_packed =
        (static_cast<unsigned long long>(cluster_nx) & 0xffull) |
        ((static_cast<unsigned long long>(cluster_ny) & 0xffull) << 8) |
        ((static_cast<unsigned long long>(cluster_nz) & 0xffull) << 16) |
        ((static_cast<unsigned long long>(cluster_ncta) & 0xffull) << 32);
    unsigned long long* row = cutlass_single_cta_marker_ptr +
        ((launch_slot * cutlass_marker_max_ctas + cta) * cutlass_marker_max_warps +
         static_cast<unsigned long long>(warp)) * cutlass_marker_fields;
    if (cta == 0 && warp == 0 && cutlass_single_cta_marker_grid_ids != nullptr) {
      cutlass_single_cta_marker_grid_ids[launch_slot] = grid_id;
    }
    if (cta == 0 && warp == 0 && cutlass_single_cta_marker_grids != nullptr) {
      cutlass_single_cta_marker_grids[launch_slot] =
          (static_cast<unsigned long long>(gridDim.x) & 0xffffull) |
          ((static_cast<unsigned long long>(gridDim.y) & 0xffffull) << 16) |
          ((static_cast<unsigned long long>(gridDim.z) & 0xffffull) << 32);
    }
    row[0] = start_clock;
    row[2] = start_timer;
    row[4] = static_cast<unsigned long long>(sm_id);
    row[5] = cluster_dim_packed;
    row[6] = static_cast<unsigned long long>(cluster_rank);
  }

  Operator op;
  op(params, smem);

  if (lane == 0 && cutlass_single_cta_marker_ptr != nullptr &&
      cta < cutlass_marker_max_ctas && warp < cutlass_marker_max_warps) {
    unsigned long long end_clock = 0;
    unsigned long long end_timer = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_timer));
    unsigned long long* row = cutlass_single_cta_marker_ptr +
        ((launch_slot * cutlass_marker_max_ctas + cta) * cutlass_marker_max_warps +
         static_cast<unsigned long long>(warp)) * cutlass_marker_fields;
    row[1] = end_clock;
    row[3] = end_timer;
  }
  cutlass::arch::synclog_print();
}
#endif  // __CUDACC__

#if defined(__CUDACC__)
namespace {
// Per-TU host setter: cudaMemcpyToSymbol into this TU's own static __device__ symbols.
// CUDA is not initialized / buffer is nullptr at this point is harmless: the device code
// treats a nullptr marker pointer as "no markers" (identical to upstream behavior).
void cutlass_fa3_marker_set_this_tu_buffers(unsigned long long* markers,
                                            unsigned long long* grid_ids,
                                            unsigned long long* grids) {
  cudaMemcpyToSymbol(cutlass_single_cta_marker_ptr, &markers, sizeof(markers));
  cudaMemcpyToSymbol(cutlass_single_cta_marker_grid_ids, &grid_ids, sizeof(grid_ids));
  cudaMemcpyToSymbol(cutlass_single_cta_marker_grids, &grids, sizeof(grids));
}
// Static registration object: enrolls this TU's setter at .so load time.
const fa3_marker::RegistrarToken cutlass_fa3_marker_registrar_instance{
    &cutlass_fa3_marker_set_this_tu_buffers};
}  // namespace
#endif  // __CUDACC__

}  // namespace cutlass
