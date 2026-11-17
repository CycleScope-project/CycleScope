/***************************************************************************************************
 * FlashMLA delivery measurement-only in-kernel clock marker (T0005 修订 3-② / 修订 5).
 *
 * Instruments the three SM90 dense-decode kernels launched by one `flash_mla_with_kvcache`
 * operation — flash_fwd_splitkv_mla_kernel (main), flash_fwd_mla_combine_kernel (combine) and
 * get_mla_metadata_kernel (sched meta) — with one
 *
 *     [start_clock64, end_clock64, start_globaltimer, end_globaltimer, smid,
 *      cluster_dim_packed, cluster_ctarank]
 *
 * row per (cta, warp) around the outermost body of each kernel.  Field layout, row validity
 * rule and the host-side reduction mirror the FA3/CUTLASS device_kernel override exactly
 * (suites/cutlass/harness/override/cutlass/device_kernel.h and
 * suites/flashattention3/harness/marker_timing.py; see
 * $HOPPER_ARTIFACT_ROOT/tasks/T0005/marker_timing_plan_flashmla.md).
 *
 * Multi-TU support (same mechanism as fa3_marker_registry.h): the extension links four nvcc
 * TUs (fp16.cu, bf16.cu, combine.cu, get_decoding_sched_meta.cu).  Each TU that includes this
 * header owns private `static __device__` marker symbols plus a same-TU host setter, and a
 * static registrar object enrolls the setter into the cross-TU registry below.  api.cpp
 * defines flash_mla_delivery_marker_set / flash_mla_delivery_marker_tu_count, which the
 * harness calls through ctypes on the built flash_mla/cuda.so ("the runner sets the device
 * pointer before launching the kernel"; no kernel signature or ABI changes).
 *
 * Without FLASHMLA_DELIVERY_MARKER_TIMING this header expands to nothing: the kernel sources
 * keep their `#ifdef`-guarded includes and instrumentation out of the pinned upstream tree's
 * compiled semantics; the no-marker build's SASS is verified against the pristine upstream
 * (formal-denominator kernel byte-identical; combine/sched-meta differ only in
 * FLASH_DEVICE_ASSERT __LINE__/__FILE__ immediates — see the T0005 marker check record).
 **************************************************************************************************/
#pragma once

#if defined(FLASHMLA_DELIVERY_MARKER_TIMING)

#include <cuda_runtime.h>

#include <vector>

namespace flashmla_marker {

// ---------------------------------------------------------------------------------------------
// Cross-TU host-side registry (pure C++; header-only; one shared instance per .so via an
// inline function's ODR-static).  Host-only TUs such as api.cpp include this part too.
// ---------------------------------------------------------------------------------------------

// Each TU's host setter runs cudaMemcpyToSymbol on its own three static __device__ symbols.
using MarkerSetterFn = void (*)(const void* main_ptr, const void* combine_ptr,
                                const void* sched_ptr);

inline std::vector<MarkerSetterFn>& registry() {
  static std::vector<MarkerSetterFn> instance;
  return instance;
}

// Registers one setter per TU; de-duplicates by function pointer.  Returns the registry size.
inline int register_marker_setter(MarkerSetterFn fn) {
  auto& setters = registry();
  for (MarkerSetterFn existing : setters) {
    if (existing == fn) {
      return static_cast<int>(setters.size());
    }
  }
  setters.push_back(fn);
  return static_cast<int>(setters.size());
}

// Static registrar token: construct one per TU (internal linkage) to enroll the TU's setter
// at static-init time (i.e. when Python imports the extension .so).
struct RegistrarToken {
  explicit RegistrarToken(MarkerSetterFn fn) { register_marker_setter(fn); }
};

inline int registry_size() { return static_cast<int>(registry().size()); }

// Walks the registry and points every TU's device symbols at the shared buffers.
inline void set_all_tu_buffers(const void* main_ptr, const void* combine_ptr,
                               const void* sched_ptr) {
  for (MarkerSetterFn fn : registry()) {
    fn(main_ptr, combine_ptr, sched_ptr);
  }
}

}  // namespace flashmla_marker

// Defined once in csrc/api/api.cpp; called by the harness via ctypes.
extern "C" void flash_mla_delivery_marker_set(const void* main_ptr, const void* combine_ptr,
                                              const void* sched_ptr);
extern "C" int flash_mla_delivery_marker_tu_count();

#if defined(__CUDACC__)

// Per-TU device symbols: one buffer pointer per instrumented kernel region.
static __device__ unsigned long long* flashmla_marker_main_ptr = nullptr;
static __device__ unsigned long long* flashmla_marker_combine_ptr = nullptr;
static __device__ unsigned long long* flashmla_marker_sched_ptr = nullptr;
// 262144 rows per region: the dense-decode grid (num_m_blocks * h_k * num_sm_parts CTAs, 8
// warps each, num_sm_parts ~ 132) peaks around 1e3 rows for the delivery shapes; the guard
// below keeps out-of-range rows from writing instead of relying on the bound.
static constexpr unsigned long long flashmla_marker_max_rows = 1ull << 18;
static constexpr unsigned long long flashmla_marker_fields = 7;

// Entry half of the outermost marker pair: clock64 + globaltimer + SM/cluster identity,
// matching fields [0], [2], [4], [5], [6] of the FA3/CUTLASS row layout.
__device__ __forceinline__ void flashmla_marker_entry(unsigned long long* region,
                                                      unsigned long long row) {
  if (region == nullptr || row >= flashmla_marker_max_rows) {
    return;
  }
  unsigned long long* r = region + row * flashmla_marker_fields;
  unsigned long long clock_value = 0;
  unsigned long long timer_value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock_value) :: "memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer_value) :: "memory");
  unsigned int sm_id = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
  unsigned long long cluster_packed = 0;
  unsigned int cluster_rank = 0;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
  unsigned int cluster_nx = 1, cluster_ny = 1, cluster_nz = 1;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_nx));
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(cluster_ny));
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(cluster_nz));
  unsigned int cluster_ncta = cluster_nx * cluster_ny * cluster_nz;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cluster_rank));
  cluster_packed = (static_cast<unsigned long long>(cluster_nx) & 0xffull) |
                   ((static_cast<unsigned long long>(cluster_ny) & 0xffull) << 8) |
                   ((static_cast<unsigned long long>(cluster_nz) & 0xffull) << 16) |
                   ((static_cast<unsigned long long>(cluster_ncta) & 0xffull) << 32);
#endif
  r[0] = clock_value;
  r[2] = timer_value;
  r[4] = static_cast<unsigned long long>(sm_id);
  r[5] = cluster_packed;
  r[6] = static_cast<unsigned long long>(cluster_rank);
}

// Exit half of the outermost marker pair: fields [1] and [3].
__device__ __forceinline__ void flashmla_marker_exit(unsigned long long* region,
                                                     unsigned long long row) {
  if (region == nullptr || row >= flashmla_marker_max_rows) {
    return;
  }
  unsigned long long* r = region + row * flashmla_marker_fields;
  unsigned long long clock_value = 0;
  unsigned long long timer_value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock_value) :: "memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer_value) :: "memory");
  r[1] = clock_value;
  r[3] = timer_value;
}

// Linear (cta, warp) row index inside one kernel region; 8 warps per CTA on every
// instrumented kernel (NUM_THREADS 256 / 256 / 32).
__device__ __forceinline__ unsigned long long flashmla_marker_row_index() {
  unsigned long long cta =
      static_cast<unsigned long long>(blockIdx.x) +
      static_cast<unsigned long long>(gridDim.x) * static_cast<unsigned long long>(blockIdx.y) +
      static_cast<unsigned long long>(gridDim.x) * gridDim.y *
          static_cast<unsigned long long>(blockIdx.z);
  return cta * (static_cast<unsigned long long>(blockDim.x) / 32) +
         static_cast<unsigned long long>(threadIdx.x / 32);
}

// Per-TU host setter: point this TU's three static __device__ symbols at the shared buffers.
static void flashmla_marker_set_tu_buffers(const void* main_ptr, const void* combine_ptr,
                                           const void* sched_ptr) {
  cudaMemcpyToSymbol(flashmla_marker_main_ptr, &main_ptr, sizeof(main_ptr));
  cudaMemcpyToSymbol(flashmla_marker_combine_ptr, &combine_ptr, sizeof(combine_ptr));
  cudaMemcpyToSymbol(flashmla_marker_sched_ptr, &sched_ptr, sizeof(sched_ptr));
}

// Enroll this TU's setter into the cross-TU registry at static-init time.
static const flashmla_marker::RegistrarToken flashmla_marker_tu_registrar{
    flashmla_marker_set_tu_buffers};

#endif  // __CUDACC__

#endif  // FLASHMLA_DELIVERY_MARKER_TIMING
