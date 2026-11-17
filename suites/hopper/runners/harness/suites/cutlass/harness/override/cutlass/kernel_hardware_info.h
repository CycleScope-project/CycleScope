/***************************************************************************************************
 * Measurement-only override of CUTLASS kernel_hardware_info.h.
 *
 * Default full-GPU behavior is unchanged. When CUTLASS_DELIVERY_MODE=cluster_sm
 * and CUTLASS_CLUSTER_SM_COUNT is a positive integer, persistent kernels that
 * query SM count through this helper see 1 or 2 SMs. This file is ahead of
 * upstream on the include path and is used only by the delivery binaries.
 **************************************************************************************************/
#pragma once

#include "cutlass/device_kernel.h"
#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#include <cstdlib>
#include <cstring>
#endif
#include <cute/int_tuple.hpp>

namespace cutlass {

struct KernelHardwareInfo {
  int device_id = 0;
  int sm_count  = 0;
  int max_active_clusters = 0;
  dim3 cluster_shape = {0,0,0};
  dim3 cluster_shape_fallback = {0,0,0};

#if !defined(__CUDACC_RTC__)
  static inline int
  query_device_multiprocessor_count(int device_id = 0) {
    char const* mode = std::getenv("CUTLASS_DELIVERY_MODE");
    if (mode != nullptr && std::strcmp(mode, "cluster_sm") == 0) {
      char const* text = std::getenv("CUTLASS_CLUSTER_SM_COUNT");
      if (text != nullptr && text[0] != '\0') {
        int value = std::atoi(text);
        if (value > 0) {
          return value;
        }
      }
    }
    cudaError_t result = cudaGetDevice(&device_id);
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST(
        "  cudaGetDevice() returned error "
        << cudaGetErrorString(result));
      return 0;
    }
    int multiprocessor_count;
    result = cudaDeviceGetAttribute(&multiprocessor_count,
      cudaDevAttrMultiProcessorCount, device_id);
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST(
        "  cudaDeviceGetAttribute() returned error "
        << cudaGetErrorString(result));
      return 0;
    }
    return multiprocessor_count;
  }

  static inline int
  query_device_max_active_clusters(
      dim3 cluster_dims,
      uint32_t threads_per_block,
      void const* kernel_ptr) {
    char const* mode = std::getenv("CUTLASS_DELIVERY_MODE");
    if (mode != nullptr && std::strcmp(mode, "cluster_sm") == 0) {
      return 1;
    }
    int max_active_clusters = 0;
#if defined(CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED)
    ClusterLauncher::LaunchConfig cluster_launch_config = ClusterLauncher::make_cluster_launch_config(
                                                            cluster_dims /* minimum grid dim */, cluster_dims, {threads_per_block, 1, 1});
    cudaError_t result = cudaOccupancyMaxActiveClusters(&max_active_clusters, kernel_ptr, &cluster_launch_config.launch_config);
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST(
        "  cudaGetDevice() returned error "
        << cudaGetErrorString(result));
      return 0;
    }
    CUTLASS_TRACE_HOST("cudaOccupancyMaxActiveClusters: maximum number of clusters that could co-exist on the target device = "
        << max_active_clusters << "\n");
    return max_active_clusters;
#else
    CUTLASS_TRACE_HOST("ClusterLauncher: CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED not defined! Aborting cluster occupancy query.");
    return max_active_clusters;
#endif
  }

  template <typename Kernel>
  static inline int
  query_device_max_active_clusters() {
    dim3 cluster_dims(cute::size<0>(typename Kernel::ClusterShape{}),
                      cute::size<1>(typename Kernel::ClusterShape{}),
                      cute::size<2>(typename Kernel::ClusterShape{}));
    uint32_t threads_per_block = Kernel::MaxThreadsPerBlock;
    void const* kernel_ptr = (void*)(device_kernel<Kernel>);
    return query_device_max_active_clusters(cluster_dims, threads_per_block, kernel_ptr);
  }

  template <typename Kernel>
  static inline KernelHardwareInfo
  make_kernel_hardware_info(int const device_id = 0, int sm_count = 0, int max_active_clusters = 0) {
    if (sm_count == 0) {
      sm_count = query_device_multiprocessor_count(device_id);
    }
    if (max_active_clusters == 0) {
      max_active_clusters = query_device_max_active_clusters<Kernel>();
    }
    return {device_id, sm_count, max_active_clusters};
  }
#endif
};

} // namespace cutlass
