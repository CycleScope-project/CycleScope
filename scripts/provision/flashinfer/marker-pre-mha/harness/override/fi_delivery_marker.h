/***************************************************************************************************
 * FlashInfer delivery measurement-only in-kernel clock marker (T0005 修订 3-② / 修订 5).
 *
 * Instruments the JIT-compiled FlashInfer kernels with one
 *
 *     [start_clock64, end_clock64, start_globaltimer, end_globaltimer, smid,
 *      cluster_dim_packed, cluster_ctarank]
 *
 * row per (cta, warp) around the outermost body of each primary kernel.  Field layout, row
 * validity rule and the host-side reduction mirror the FA3/CUTLASS device_kernel override and the
 * FlashMLA marker patch exactly (delivery_tests/cutlass/harness/override/cutlass/device_kernel.h,
 * delivery_tests/flashattention3/harness/marker_timing.py,
 * delivery_tests/flashmla/harness/override/flashmla_marker.h; see
 * $HOPPER_ARTIFACT_ROOT/tasks/T0005/marker_readiness.md).
 *
 * Cross-TU wiring = the FlashMLA multi-TU registry, adapted to FlashInfer's one-.so-per-module
 * JIT layout.  Device symbols are `static __device__` (whole-program compilation rejects
 * external-linkage inline device variables), so every TU that includes this header owns private
 * symbols plus a private static host setter, and the patcher appends a
 * FI_DELIVERY_MARKER_REGISTER_MODULE("<module-key>") registrar at the end of each patched file.
 * The registrar enrolls the TU setter, at .so load time, into the shared registry inside
 * build/libdelivery_fi_marker.so — every JIT module links that library via the upstream
 * FLASHINFER_EXTRA_LDFLAGS hook (harness/marker_timing.py::enable_jit_marker_mode).  The
 * harness drives the registry through ctypes: flashinfer_delivery_marker_register /
 * _set_buffers / _module_count / _module_names ("the runner sets the device pointer before
 * launching the kernel"; no kernel signature, ABI or linkage change).
 *
 * Region numbering inside a TU: 0 = primary (formal denominator), 1..3 = secondary evidence
 * (harness/marker_channel_config.py per op class; multi-kernel ops record the primary kernel
 * only in timing.main_kernel, like the FlashMLA combine/sched regions).
 *
 * Without a patch from scripts/patch_wheel_marker.py this header is never included: JIT sources
 * keep their `#ifdef`-guarded instrumentation out of the compiled semantics and a no-marker JIT
 * build reproduces the upstream kernels bit-for-bit (same guarantee as the FlashMLA lane).  With
 * the patch, markers are compiled in but stay inactive until the buffers are set: the null-
 * pointer guard makes every warp skip the CS2R SR_CLOCKLO/SR_GLOBALTIMERLO reads (FA3 behaviour),
 * so an un-set capture degrades loudly in the dynamic audit, never silently.
 **************************************************************************************************/
#pragma once

#if defined(FI_DELIVERY_MARKER_TIMING)

// ---------------------------------------------------------------------------------------------
// Registry entry points (defined in build/libdelivery_fi_marker.so, linked into every JIT
// module; declared here so the registrar can call them from any TU).
// ----------------------------------------------------------------------------------------------
extern "C" {
void flashinfer_delivery_marker_register(void (*setter_fn)(const void* const*, int),
                                         const char* module_name);
void flashinfer_delivery_marker_set_buffers(const void* const* region_ptrs, int num_regions);
int flashinfer_delivery_marker_module_count();
int flashinfer_delivery_marker_module_names(char* storage, int slot_stride, int max_slots);
}

#if defined(__CUDACC__)

#include <cuda_runtime.h>

namespace fi_delivery_marker {

// Per-TU device symbols: one buffer pointer per instrumented region (262144 rows each; the
// far-set shapes peak around 1e3-1e4 (cta, warp) rows; the row guard keeps out-of-range rows
// from writing instead of relying on the bound).
static __device__ unsigned long long* fi_marker_primary_ptr = nullptr;
static __device__ unsigned long long* fi_marker_secondary_ptr = nullptr;
static __device__ unsigned long long* fi_marker_tertiary_ptr = nullptr;
static __device__ unsigned long long* fi_marker_quaternary_ptr = nullptr;
inline constexpr unsigned long long kMaxRows = 1ull << 18;
inline constexpr unsigned long long kFields = 7;

// Entry half of the outermost marker pair: clock64 + globaltimer + SM/cluster identity,
// fields [0], [2], [4], [5], [6] of the FA3/CUTLASS/FlashMLA row layout.
__device__ __forceinline__ void fi_marker_entry(unsigned long long* region,
                                                unsigned long long row) {
  if (region == nullptr || row >= kMaxRows) {
    return;
  }
  unsigned long long* r = region + row * kFields;
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
__device__ __forceinline__ void fi_marker_exit(unsigned long long* region,
                                               unsigned long long row) {
  if (region == nullptr || row >= kMaxRows) {
    return;
  }
  unsigned long long* r = region + row * kFields;
  unsigned long long clock_value = 0;
  unsigned long long timer_value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock_value) :: "memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer_value) :: "memory");
  r[1] = clock_value;
  r[3] = timer_value;
}

// Linear (cta, warp) row index inside one kernel region; covers 1d/2d/3d blocks.
__device__ __forceinline__ unsigned long long fi_marker_row_index() {
  unsigned long long cta =
      static_cast<unsigned long long>(blockIdx.x) +
      static_cast<unsigned long long>(gridDim.x) * static_cast<unsigned long long>(blockIdx.y) +
      static_cast<unsigned long long>(gridDim.x) * gridDim.y *
          static_cast<unsigned long long>(blockIdx.z);
  unsigned int warps_per_cta = (blockDim.x + 31) / 32;
  unsigned int warp = threadIdx.x / 32;
  if (blockDim.y > 1) {
    warp += static_cast<unsigned int>(threadIdx.y) * ((blockDim.x + 31) / 32);
    warps_per_cta *= static_cast<unsigned int>(blockDim.y);
  }
  if (blockDim.z > 1) {
    warp += static_cast<unsigned int>(threadIdx.z) * ((blockDim.x + 31) / 32) *
            static_cast<unsigned int>(blockDim.y);
    warps_per_cta *= static_cast<unsigned int>(blockDim.z);
  }
  return cta * static_cast<unsigned long long>(warps_per_cta) +
         static_cast<unsigned long long>(warp);
}

// One warp lane per warp executes the marker (single writer); lane 0 by convention.
__device__ __forceinline__ bool fi_marker_warp_leader() { return (threadIdx.x & 31) == 0; }

// Per-TU host setter: point this TU's region symbols at the shared buffers.
static void fi_marker_set_tu_buffers(const void* const* region_ptrs, int num_regions) {
  // cudaMemcpyToSymbol resolves compiler symbol stubs, not runtime device addresses — one
  // explicit call per region slot (same shape as the FlashMLA per-TU setter).
  if (num_regions > 0) {
    cudaMemcpyToSymbol(fi_marker_primary_ptr, &region_ptrs[0], sizeof(region_ptrs[0]));
  }
  if (num_regions > 1) {
    cudaMemcpyToSymbol(fi_marker_secondary_ptr, &region_ptrs[1], sizeof(region_ptrs[1]));
  }
  if (num_regions > 2) {
    cudaMemcpyToSymbol(fi_marker_tertiary_ptr, &region_ptrs[2], sizeof(region_ptrs[2]));
  }
  if (num_regions > 3) {
    cudaMemcpyToSymbol(fi_marker_quaternary_ptr, &region_ptrs[3], sizeof(region_ptrs[3]));
  }
}
}  // namespace fi_delivery_marker

using fi_delivery_marker::fi_marker_entry;
using fi_delivery_marker::fi_marker_exit;
using fi_delivery_marker::fi_marker_row_index;
using fi_delivery_marker::fi_marker_warp_leader;
using fi_delivery_marker::fi_marker_primary_ptr;
using fi_delivery_marker::fi_marker_secondary_ptr;
using fi_delivery_marker::fi_marker_tertiary_ptr;
using fi_delivery_marker::fi_marker_quaternary_ptr;

// Enroll this TU's setter into the delivery registry at static-init time (i.e. when the harness
// imports the module .so).  module_key identifies the patch file in registry metadata
// (harness/marker_channel_config.py module keys); several TUs of one module may share a key,
// each keeping its own entry.  Usable at any namespace scope: it re-opens
// namespace fi_delivery_marker.  NOTE: no // comments inside the macro body — a line
// comment would swallow the backslash continuations and with them the closing braces.
#define FI_DELIVERY_MARKER_CONCAT_(a, b) a##b
#define FI_DELIVERY_MARKER_CONCAT(a, b) FI_DELIVERY_MARKER_CONCAT_(a, b)
#define FI_DELIVERY_MARKER_REGISTER_MODULE(module_key_literal)                                \
  namespace fi_delivery_marker {                                                              \
  namespace {                                                                                 \
  const auto FI_DELIVERY_MARKER_CONCAT(fi_marker_registrar_, __COUNTER__) = []() {           \
    flashinfer_delivery_marker_register(&fi_marker_set_tu_buffers, module_key_literal);       \
    return 0;                                                                                 \
  }();                                                                                        \
  }                                                                                            \
  }

#else  // host-only TU: no device symbols, registrar unavailable (kernels live in device TUs).

#define FI_DELIVERY_MARKER_REGISTER_MODULE(module_key_literal)

#endif  // __CUDACC__

#endif  // FI_DELIVERY_MARKER_TIMING
