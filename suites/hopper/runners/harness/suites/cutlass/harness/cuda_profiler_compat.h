#pragma once

#include <cuda_runtime_api.h>

#if defined(__has_include)
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#define HOPPER_HAVE_CUDA_PROFILER_API_HEADER 1
#endif
#endif

#ifndef HOPPER_HAVE_CUDA_PROFILER_API_HEADER
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif
