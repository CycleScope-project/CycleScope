// Device-globaltimer whole-operation timer for the Hopper delivery suites.
// Keep this file byte-for-byte equivalent to the FA4 delivery timer source.

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ std::uint64_t read_clock64() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
}

__device__ __forceinline__ std::uint64_t read_globaltimer() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

__device__ __forceinline__ std::uint32_t read_smid() {
  std::uint32_t value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value) :: "memory");
  return value;
}

__device__ __forceinline__ unsigned int read_release_flag(
    unsigned int const* release_flag) {
  unsigned int value;
  asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(value) : "l"(release_flag) : "memory");
  return value;
}

__global__ void calibrate_kernel(std::uint64_t* markers,
                                 std::uint64_t wait_cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  std::uint64_t start_clock = read_clock64();
  std::uint64_t start_timer = read_globaltimer();
  while (read_clock64() - start_clock < wait_cycles) {
    __nanosleep(64);
  }
  markers[0] = start_clock;
  markers[1] = start_timer;
  markers[2] = read_clock64();
  markers[3] = read_globaltimer();
}

__global__ void blocker_kernel(unsigned int* release_flag,
                               std::uint64_t* markers) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  std::uint64_t deadline = read_globaltimer() + 5000000000ull;
  while (read_release_flag(release_flag) == 0u) {
    if (read_globaltimer() >= deadline) {
      markers[10] = 1;
      return;
    }
    __nanosleep(256);
  }
}

__global__ void start_kernel(std::uint64_t* markers) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  markers[4] = read_clock64();
  markers[5] = read_globaltimer();
  markers[6] = read_smid();
}

__global__ void end_kernel(std::uint64_t* markers) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  markers[7] = read_clock64();
  markers[8] = read_globaltimer();
  markers[9] = read_smid();
}

}  // namespace

extern "C" cudaError_t delivery_globaltimer_allocate_flag(
    unsigned int** host_flag, unsigned int** device_flag) {
  cudaError_t status = cudaHostAlloc(
      reinterpret_cast<void**>(host_flag), sizeof(unsigned int), cudaHostAllocMapped);
  if (status != cudaSuccess) return status;
  status = cudaHostGetDevicePointer(
      reinterpret_cast<void**>(device_flag), *host_flag, 0);
  if (status != cudaSuccess) cudaFreeHost(*host_flag);
  return status;
}

extern "C" cudaError_t delivery_globaltimer_free_flag(unsigned int* host_flag) {
  return cudaFreeHost(host_flag);
}

extern "C" void delivery_globaltimer_reset_flag(unsigned int* host_flag) {
  __atomic_store_n(host_flag, 0u, __ATOMIC_RELEASE);
}

extern "C" void delivery_globaltimer_release_flag(unsigned int* host_flag) {
  __atomic_store_n(host_flag, 1u, __ATOMIC_RELEASE);
}

extern "C" cudaError_t delivery_clock64_calibrate(
    std::uint64_t* markers, std::uint64_t wait_cycles, cudaStream_t stream) {
  calibrate_kernel<<<1, 1, 0, stream>>>(markers, wait_cycles);
  return cudaGetLastError();
}

extern "C" cudaError_t delivery_clock64_block(
    unsigned int* release_flag, std::uint64_t* markers, cudaStream_t stream) {
  blocker_kernel<<<1, 1, 0, stream>>>(release_flag, markers);
  return cudaGetLastError();
}

extern "C" cudaError_t delivery_clock64_start(
    std::uint64_t* markers, cudaStream_t stream) {
  start_kernel<<<1, 1, 0, stream>>>(markers);
  return cudaGetLastError();
}

extern "C" cudaError_t delivery_clock64_end(
    std::uint64_t* markers, cudaStream_t stream) {
  end_kernel<<<1, 1, 0, stream>>>(markers);
  return cudaGetLastError();
}
