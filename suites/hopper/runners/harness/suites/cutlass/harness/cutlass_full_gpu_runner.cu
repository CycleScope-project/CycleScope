#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include "cuda_profiler_compat.h"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

namespace cutlass {
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Int<1>, int64_t> packed_stride(
    cute::Stride<IntT, cute::Int<1>, int64_t> stride,
    cute::Shape<int, int, int> shape) {
  cute::get<0>(stride) = static_cast<IntT>(cute::get<1>(shape));
  cute::get<2>(stride) = cute::get<2>(shape) > 1
      ? static_cast<int64_t>(cute::get<0>(shape) * cute::get<1>(shape)) : 0;
  return stride;
}

template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, IntT, int64_t> packed_stride(
    cute::Stride<cute::Int<1>, IntT, int64_t> stride,
    cute::Shape<int, int, int> shape) {
  cute::get<1>(stride) = static_cast<IntT>(cute::get<0>(shape));
  cute::get<2>(stride) = cute::get<2>(shape) > 1
      ? static_cast<int64_t>(cute::get<0>(shape) * cute::get<1>(shape)) : 0;
  return stride;
}
}

namespace cutlass_full_gpu {

constexpr int kMarkerFields = 7;
constexpr int kMaxWarps = 16;
constexpr int kMaxCtas = 16384;

std::string env_or(char const* name, char const* fallback) {
  char const* value = std::getenv(name);
  return value == nullptr ? std::string(fallback) : std::string(value);
}

bool matches_gpu_profile(std::string const& name, std::string const& profile) {
  if (profile == "h100") {
    return name.find("H100") != std::string::npos;
  }
  if (profile == "h200") return name.find("H200") != std::string::npos;
  if (profile == "h20") {
    return name.find("H20") != std::string::npos && name.find("H200") == std::string::npos;
  }
  throw std::runtime_error("unsupported GEMM_DELIVERY_GPU profile: " + profile);
}

int minimum_full_gpu_sms(int sm_count) {
  (void)sm_count;
  return 1;
}

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::ostringstream stream_;                                                \
      stream_ << #expr << " failed: " << cudaGetErrorString(status_);            \
      throw std::runtime_error(stream_.str());                                   \
    }                                                                            \
  } while (0)

struct Options {
  std::string variant = "f16_nt_coop_128x128";
  std::string case_id = "pilot";
  int m = 128;
  int n = 128;
  int k = 32768;
  float alpha = 1.0f;
  float beta = 0.0f;
  int warmups = 3;
  int trials = 9;
};

std::string argument_value(int argc, char const** argv, std::string const& name,
                           std::string const& fallback) {
  std::string prefix = "--" + name + "=";
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind(prefix, 0) == 0) {
      return arg.substr(prefix.size());
    }
  }
  return fallback;
}

Options parse_options(int argc, char const** argv) {
  Options options;
  options.variant = argument_value(argc, argv, "variant", options.variant);
  options.case_id = argument_value(argc, argv, "case", options.case_id);
  options.m = std::stoi(argument_value(argc, argv, "m", std::to_string(options.m)));
  options.n = std::stoi(argument_value(argc, argv, "n", std::to_string(options.n)));
  options.k = std::stoi(argument_value(argc, argv, "k", std::to_string(options.k)));
  options.alpha = std::stof(argument_value(argc, argv, "alpha", std::to_string(options.alpha)));
  options.beta = std::stof(argument_value(argc, argv, "beta", std::to_string(options.beta)));
  options.warmups = std::stoi(argument_value(argc, argv, "warmups", std::to_string(options.warmups)));
  options.trials = std::stoi(argument_value(argc, argv, "trials", std::to_string(options.trials)));
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.warmups < 0 || options.trials <= 0) {
    throw std::runtime_error("invalid positive dimension/trial argument");
  }
  return options;
}

template <class T>
__global__ void fill_kernel(T* output, std::size_t count, float value) {
  std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = T(value);
  }
}

template <class T>
__global__ void fill_alternating_kernel(T* output, std::size_t count, float value) {
  std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = T((index & 1) ? -value : value);
  }
}

template <class T>
void fill_device(T* output, std::size_t count, float value) {
  int threads = 256;
  int blocks = int((count + threads - 1) / threads);
  fill_kernel<<<blocks, threads>>>(output, count, value);
  CUDA_CHECK(cudaGetLastError());
}

template <class T>
void fill_alternating_device(T* output, std::size_t count, float value) {
  int threads = 256;
  int blocks = int((count + threads - 1) / threads);
  fill_alternating_kernel<<<blocks, threads>>>(output, count, value);
  CUDA_CHECK(cudaGetLastError());
}

template <class T>
struct DeviceBuffer {
  T* pointer = nullptr;
  std::size_t count = 0;

  explicit DeviceBuffer(std::size_t size = 0) : count(size) {
    if (count) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
    }
  }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  ~DeviceBuffer() {
    if (pointer) {
      cudaFree(pointer);
    }
  }
};

double mean(std::vector<double> const& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / double(values.size());
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  std::size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

double cv(std::vector<double> const& values) {
  double average = mean(values);
  if (values.size() < 2 || average == 0.0) {
    return 0.0;
  }
  double sum = 0.0;
  for (double value : values) {
    double delta = value - average;
    sum += delta * delta;
  }
  return std::sqrt(sum / double(values.size() - 1)) / average;
}

std::string json_array(std::vector<double> const& values) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) stream << ",";
    stream << std::fixed << std::setprecision(3) << values[i];
  }
  stream << "]";
  return stream.str();
}

template <class ElementAB, class LayoutA, class LayoutB, class ElementCD,
          class LayoutCD, class TileShape, class ClusterShape,
          class MainloopSchedule, class EpilogueSchedule,
          class TileScheduler>
int run_gemm(Options const& options, char const* dtype, char const* layout,
             char const* tile, char const* schedule, char const* cluster,
             char const* tile_scheduler) {
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementCD>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementCD>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementCD, LayoutCD, AlignmentC,
      ElementCD, LayoutCD, AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementAB, LayoutA, AlignmentA,
      ElementAB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  auto problem = typename GemmKernel::ProblemShape{options.m, options.n, options.k, 1};
  StrideA stride_a = cutlass::packed_stride(StrideA{}, make_shape(options.m, options.k, 1));
  StrideB stride_b = cutlass::packed_stride(StrideB{}, make_shape(options.n, options.k, 1));
  StrideC stride_c = cutlass::packed_stride(StrideC{}, make_shape(options.m, options.n, 1));
  StrideD stride_d = cutlass::packed_stride(StrideD{}, make_shape(options.m, options.n, 1));

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::size_t count_a = std::size_t(options.m) * options.k;
  std::size_t count_b = std::size_t(options.n) * options.k;
  std::size_t count_cd = std::size_t(options.m) * options.n;
  DeviceBuffer<ElementAB> a(count_a);
  DeviceBuffer<ElementAB> b(count_b);
  DeviceBuffer<ElementCD> c(count_cd);
  DeviceBuffer<ElementCD> d(count_cd);
  DeviceBuffer<unsigned long long> markers(std::size_t(kMaxCtas) * kMaxWarps * kMarkerFields);
  if constexpr (std::is_same<ElementAB, cutlass::float_e4m3_t>::value) {
    if (options.k % 2 != 0) throw std::runtime_error("FP8 alternating input requires even K");
    fill_alternating_device(a.pointer, count_a, 0.0625f);
  } else {
    fill_device(a.pointer, count_a, 0.0625f);
  }
  fill_device(b.pointer, count_b, 0.0625f);
  fill_device(c.pointer, count_cd, 0.5f);
  CUDA_CHECK(cudaMemset(d.pointer, 0, count_cd * sizeof(ElementCD)));
  CUDA_CHECK(cudaMemset(markers.pointer, 0, markers.count * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemcpyToSymbol(cutlass::cutlass_single_cta_marker_ptr,
                                &markers.pointer, sizeof(markers.pointer)));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {a.pointer, stride_a, b.pointer, stride_b},
      {{options.alpha, options.beta}, c.pointer, stride_c, d.pointer, stride_d},
      hw_info};

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS can_implement rejected the case");
  }
  std::size_t workspace_bytes = Gemm::get_workspace_size(arguments);
  DeviceBuffer<unsigned char> workspace(workspace_bytes);
  status = gemm.initialize(arguments, workspace.pointer);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS initialize failed");
  }
  dim3 grid = Gemm::get_grid_shape(gemm.params());
  dim3 block = GemmKernel::get_block_shape();
  std::uint64_t cta_count = std::uint64_t(grid.x) * grid.y * grid.z;
  std::string delivery_mode = env_or("CUTLASS_DELIVERY_MODE", "full_gpu");
  bool trace_mode = env_or("CUTLASS_DELIVERY_TRACE", "0") == "1";
  if (cta_count == 0 || cta_count > kMaxCtas) {
    std::ostringstream stream;
    stream << "expected 1.." << kMaxCtas
           << " CTAs, got grid=("
           << grid.x << "," << grid.y << "," << grid.z << ")";
    throw std::runtime_error(stream.str());
  }
  if (trace_mode && options.trials != 1) {
    throw std::runtime_error("CUTLASS_DELIVERY_TRACE requires exactly one measured trial");
  }
  int warp_count = int(block.x * block.y * block.z) / 32;
  if (warp_count <= 0 || warp_count > kMaxWarps) {
    throw std::runtime_error("unexpected CUTLASS warp count");
  }

  for (int i = 0; i < options.warmups; ++i) {
    status = gemm.initialize(arguments, workspace.pointer);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS warmup workspace initialization failed");
    }
    status = gemm.run();
    if (status != cutlass::Status::kSuccess) throw std::runtime_error("CUTLASS warmup launch failed");
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<double> cycle_samples;
  std::vector<double> timer_ns_samples;
  std::vector<int> active_sm_counts;
  std::vector<unsigned int> last_active_sm_ids;
  unsigned int observed_cluster_ncta = 0;
  unsigned int observed_cluster_nx = 1;
  unsigned int observed_cluster_ny = 1;
  unsigned int observed_cluster_nz = 1;
  std::vector<unsigned int> last_cluster_ranks;
  std::vector<unsigned long long> host_markers(markers.count);
  for (int trial = 0; trial < options.trials; ++trial) {
    CUDA_CHECK(cudaMemset(markers.pointer, 0, markers.count * sizeof(unsigned long long)));
    status = gemm.initialize(arguments, workspace.pointer);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS measured workspace initialization failed");
    }
    if (trace_mode) CUDA_CHECK(cudaProfilerStart());
    status = gemm.run();
    if (status != cutlass::Status::kSuccess) throw std::runtime_error("CUTLASS measured launch failed");
    CUDA_CHECK(cudaDeviceSynchronize());
    if (trace_mode) CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaMemcpy(host_markers.data(), markers.pointer,
                          markers.count * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    unsigned long long start_timer = ~0ull;
    unsigned long long end_timer = 0;
    std::vector<double> clock_per_timer_ratios;
    std::vector<unsigned int> active_sm_ids;
    std::vector<unsigned int> cluster_ranks;
    unsigned int cluster_ncta = 0;
    int valid = 0;
    for (std::uint64_t cta = 0; cta < cta_count; ++cta) {
      for (int warp = 0; warp < warp_count; ++warp) {
        auto const* row = host_markers.data() +
            (std::size_t(cta) * kMaxWarps + warp) * kMarkerFields;
        if (row[0] > 0 && row[1] > row[0] && row[2] > 0 && row[3] > row[2]) {
          start_timer = std::min(start_timer, row[2]);
          end_timer = std::max(end_timer, row[3]);
          clock_per_timer_ratios.push_back(
              double(row[1] - row[0]) / double(row[3] - row[2]));
          active_sm_ids.push_back(static_cast<unsigned int>(row[4]));
          {
            unsigned long long packed = row[5];
            unsigned int nx = static_cast<unsigned int>(packed & 0xffull);
            unsigned int ny = static_cast<unsigned int>((packed >> 8) & 0xffull);
            unsigned int nz = static_cast<unsigned int>((packed >> 16) & 0xffull);
            unsigned int ncta = static_cast<unsigned int>((packed >> 32) & 0xffull);
            if (ny == 0 && nz == 0) {
              ncta = static_cast<unsigned int>(packed);
              nx = ncta;
              ny = 1;
              nz = 1;
            } else if (ncta == 0) {
              ncta = nx * ny * nz;
            }
            cluster_ncta = std::max(cluster_ncta, ncta);
            observed_cluster_nx = nx;
            observed_cluster_ny = ny;
            observed_cluster_nz = nz;
          }
          cluster_ranks.push_back(static_cast<unsigned int>(row[6]));
          ++valid;
        }
      }
    }
    if (valid != int(cta_count) * warp_count) {
      std::ostringstream stream;
      stream << "only " << valid << " of " << cta_count * warp_count
             << " CTA/warp rows produced valid markers";
      throw std::runtime_error(stream.str());
    }
    std::sort(active_sm_ids.begin(), active_sm_ids.end());
    active_sm_ids.erase(std::unique(active_sm_ids.begin(), active_sm_ids.end()), active_sm_ids.end());
    std::sort(cluster_ranks.begin(), cluster_ranks.end());
    cluster_ranks.erase(std::unique(cluster_ranks.begin(), cluster_ranks.end()), cluster_ranks.end());
    if (active_sm_ids.empty()) {
      throw std::runtime_error("launch produced no physical SM ids");
    }
    double ratio = median(clock_per_timer_ratios);
    cycle_samples.push_back(double(end_timer - start_timer) * ratio);
    timer_ns_samples.push_back(double(end_timer - start_timer));
    active_sm_counts.push_back(int(active_sm_ids.size()));
    last_active_sm_ids = active_sm_ids;
    observed_cluster_ncta = cluster_ncta;
    last_cluster_ranks = cluster_ranks;
  }

  // The NVBit tracer can terminate immediately after the selected kernel
  // when it observes a later launch.  This is enabled only for trace capture;
  // formal timing and correctness runs leave the environment variable unset.
  if (char const* value = std::getenv("HOPPER_TRACE_SENTINEL");
      value != nullptr && std::strcmp(value, "1") == 0) {
    fill_kernel<<<1, 1>>>(d.pointer, std::size_t(0), 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<ElementCD> host_d(count_cd);
  CUDA_CHECK(cudaMemcpy(host_d.data(), d.pointer, count_cd * sizeof(ElementCD), cudaMemcpyDeviceToHost));
  ElementAB host_a{};
  ElementAB host_b{};
  ElementCD host_c{};
  CUDA_CHECK(cudaMemcpy(&host_a, a.pointer, sizeof(ElementAB), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&host_b, b.pointer, sizeof(ElementAB), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&host_c, c.pointer, sizeof(ElementCD), cudaMemcpyDeviceToHost));
  double input_a = static_cast<double>(static_cast<float>(host_a));
  double input_b = static_cast<double>(static_cast<float>(host_b));
  double input_c = static_cast<double>(static_cast<float>(host_c));
  double dot_product = std::is_same<ElementAB, cutlass::float_e4m3_t>::value
      ? 0.0 : double(options.k) * input_a * input_b;
  double expected = double(options.alpha) * dot_product + double(options.beta) * input_c;
  double max_abs_error = 0.0;
  int nonfinite = 0;
  for (ElementCD value : host_d) {
    double observed = static_cast<double>(static_cast<float>(value));
    if (!std::isfinite(observed)) ++nonfinite;
    max_abs_error = std::max(max_abs_error, std::abs(observed - expected));
  }
  double tolerance = std::is_same<ElementCD, float>::value ? std::max(1e-3, std::abs(expected) * 2e-5)
                                                           : std::max(0.5, std::abs(expected) * 0.01);
  bool correct = nonfinite == 0 && max_abs_error <= tolerance;

  std::ostringstream sm_stream;
  sm_stream << "[";
  for (std::size_t i = 0; i < last_active_sm_ids.size(); ++i) {
    if (i) sm_stream << ",";
    sm_stream << last_active_sm_ids[i];
  }
  sm_stream << "]";

  std::cout << "{";
  std::cout << "\"schema_version\":\"cutlass-full-gpu-sm90-v2\",";
  std::cout << "\"case_id\":\"" << options.case_id << "\",";
  std::cout << "\"delivery_mode\":\"" << delivery_mode << "\",";
  std::cout << "\"gpu_profile\":\"" << env_or("GEMM_DELIVERY_GPU", "h100") << "\",";
  std::cout << "\"gpu_name\":\"" << props.name << "\",";
  std::cout << "\"device_sm_count\":" << props.multiProcessorCount << ",";
  std::cout << "\"variant\":\"" << options.variant << "\",";
  std::cout << "\"dtype\":\"" << dtype << "\",";
  std::cout << "\"layout\":\"" << layout << "\",";
  std::cout << "\"tile\":\"" << tile << "\",";
  std::cout << "\"schedule\":\"" << schedule << "\",";
  std::cout << "\"cluster\":\"" << cluster << "\",";
  std::cout << "\"tile_scheduler\":\"" << tile_scheduler << "\",";
  std::cout << "\"m\":" << options.m << ",\"n\":" << options.n << ",\"k\":" << options.k << ",";
  std::cout << "\"alpha\":" << options.alpha << ",\"beta\":" << options.beta << ",";
  std::cout << "\"grid\":[" << grid.x << "," << grid.y << "," << grid.z << "],";
  std::cout << "\"block\":[" << block.x << "," << block.y << "," << block.z << "],";
  std::cout << "\"warp_count\":" << warp_count << ",";
  std::cout << "\"cta_count\":" << cta_count << ",";
  std::cout << "\"requested_sm_count\":" << hw_info.sm_count << ",";
  std::cout << "\"observed_cluster_ncta\":" << observed_cluster_ncta << ",";
  std::cout << "\"cluster_dim\":[" << observed_cluster_nx << "," << observed_cluster_ny << "," << observed_cluster_nz << "],";
  std::cout << "\"cluster_ranks\":[";
  for (std::size_t i = 0; i < last_cluster_ranks.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << last_cluster_ranks[i];
  }
  std::cout << "],";
  std::cout << "\"active_sm_count_min\":"
            << *std::min_element(active_sm_counts.begin(), active_sm_counts.end()) << ",";
  std::cout << "\"shared_storage_bytes\":" << GemmKernel::SharedStorageSize << ",";
  std::cout << "\"workspace_bytes\":" << workspace_bytes << ",";
  std::cout << "\"whole_gpu_cycles\":" << json_array(cycle_samples) << ",";
  std::cout << "\"whole_gpu_median_cycles\":" << std::fixed << std::setprecision(3) << median(cycle_samples) << ",";
  std::cout << "\"whole_gpu_mean_cycles\":" << mean(cycle_samples) << ",";
  std::cout << "\"whole_gpu_cv\":" << cv(cycle_samples) << ",";
  std::cout << "\"globaltimer_ns\":" << json_array(timer_ns_samples) << ",";
  std::cout << "\"sm_ids\":" << sm_stream.str() << ",";
  std::cout << "\"input_a\":" << input_a << ",\"input_b\":" << input_b
            << ",\"input_c\":" << input_c << ",";
  std::cout << "\"input_pattern\":\""
            << (std::is_same<ElementAB, cutlass::float_e4m3_t>::value ? "a_alternating_sign" : "constant")
            << "\",";
  std::cout << "\"expected\":" << expected << ",";
  std::cout << "\"max_abs_error\":" << max_abs_error << ",";
  std::cout << "\"tolerance\":" << tolerance << ",";
  std::cout << "\"correct\":" << (correct ? "true" : "false");
  std::cout << "}" << std::endl;
  return correct ? 0 : 2;
}

template <class Element, class LayoutA, class LayoutB, class ElementCD,
          class Tile, class Cluster, class Mainloop, class Epilogue,
          class TileScheduler = cutlass::gemm::PersistentScheduler>
int dispatch(Options const& options, char const* dtype, char const* layout,
             char const* tile, char const* schedule, char const* cluster = "1x1x1",
             char const* tile_scheduler = "persistent") {
  return run_gemm<Element, LayoutA, LayoutB, ElementCD, cutlass::layout::ColumnMajor,
                  Tile, Cluster, Mainloop, Epilogue, TileScheduler>(
                      options, dtype, layout, tile, schedule, cluster, tile_scheduler);
}

int run_variant(Options const& options) {
  using RM = cutlass::layout::RowMajor;
  using CM = cutlass::layout::ColumnMajor;
  using C1 = Shape<_1, _1, _1>;
  using C2M = Shape<_2, _1, _1>;
  using C2N = Shape<_1, _2, _1>;
  using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using CoopEpi = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using PingpongEpi = cutlass::epilogue::TmaWarpSpecialized;
  using WS = cutlass::gemm::KernelTmaWarpSpecialized;
  using Tma = cutlass::gemm::KernelTma;
  using NoSmemEpi = cutlass::epilogue::NoSmemWarpSpecialized;
  using Cp = cutlass::gemm::KernelCpAsyncWarpSpecialized;
  using CpCoop = cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative;
  using CpPingpong = cutlass::gemm::KernelCpAsyncWarpSpecializedPingpong;
  using FP8 = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using FP8Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using FP8Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using StreamK = cutlass::gemm::StreamKScheduler;
  using T128 = Shape<_128, _128, _64>;
  using T64 = Shape<_64, _128, _64>;
  using T256 = Shape<_128, _256, _64>;
  using T256M = Shape<_256, _128, _64>;
  using TTF32 = Shape<_128, _64, _32>;
  using TFP8 = Shape<_128, _128, _128>;
  using TFP8_256M = Shape<_256, _128, _128>;

  if (options.variant == "f16_nt_coop_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C1, Coop, CoopEpi>(options, "fp16", "NT", "128x128x64", "tma_cooperative");
  if (options.variant == "f16_nn_coop_128x256")
    return dispatch<cutlass::half_t, RM, RM, cutlass::half_t, T256, C1, Coop, CoopEpi>(options, "fp16", "NN", "128x256x64", "tma_cooperative");
  if (options.variant == "f16_tn_ws_128x128")
    return dispatch<cutlass::half_t, CM, CM, cutlass::half_t, T128, C1, WS, NoSmemEpi>(options, "fp16", "TN", "128x128x64", "tma_warp_specialized_no_smem_epilogue");
  if (options.variant == "f16_nt_tma_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C1, Tma, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "tma_non_warp_specialized");
  if (options.variant == "f16_tt_pingpong_64x128")
    return dispatch<cutlass::half_t, CM, RM, cutlass::half_t, T64, C1, Pingpong, PingpongEpi>(options, "fp16", "TT", "64x128x64", "tma_pingpong");
  if (options.variant == "f16_nt_cpasync_coop_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C1, CpCoop, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_cooperative");
  if (options.variant == "f16_nt_cpasync_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C1, Cp, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_warp_specialized");
  if (options.variant == "f16_nt_cpasync_pingpong_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C1, CpPingpong, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_pingpong");
  if (options.variant == "f16_nt_streamk_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C1, Coop, CoopEpi, StreamK>(options, "fp16", "NT", "128x128x64", "tma_cooperative", "1x1x1", "stream_k");
  if (options.variant == "f16_nt_cluster2m_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2M, Coop, CoopEpi>(options, "fp16", "NT", "128x128x64", "tma_cooperative", "2x1x1");
  if (options.variant == "f16_nt_cluster2n_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2N, Coop, CoopEpi>(options, "fp16", "NT", "128x128x64", "tma_cooperative", "1x2x1");
  if (options.variant == "bf16_nt_coop_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C1, Coop, CoopEpi>(options, "bf16", "NT", "128x128x64", "tma_cooperative");
  if (options.variant == "bf16_nn_coop_128x256")
    return dispatch<cutlass::bfloat16_t, RM, RM, cutlass::bfloat16_t, T256, C1, Coop, CoopEpi>(options, "bf16", "NN", "128x256x64", "tma_cooperative");
  if (options.variant == "bf16_tn_ws_128x128")
    return dispatch<cutlass::bfloat16_t, CM, CM, cutlass::bfloat16_t, T128, C1, WS, NoSmemEpi>(options, "bf16", "TN", "128x128x64", "tma_warp_specialized_no_smem_epilogue");
  if (options.variant == "bf16_tt_pingpong_64x128")
    return dispatch<cutlass::bfloat16_t, CM, RM, cutlass::bfloat16_t, T64, C1, Pingpong, PingpongEpi>(options, "bf16", "TT", "64x128x64", "tma_pingpong");
  if (options.variant == "bf16_nt_cpasync_coop_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C1, CpCoop, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "cpasync_cooperative");
  if (options.variant == "bf16_nt_cpasync_pingpong_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C1, CpPingpong, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "cpasync_pingpong");
  if (options.variant == "bf16_nt_streamk_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C1, Coop, CoopEpi, StreamK>(options, "bf16", "NT", "128x128x64", "tma_cooperative", "1x1x1", "stream_k");
  if (options.variant == "bf16_nt_cluster2m_256x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T256M, C2M, Coop, CoopEpi>(options, "bf16", "NT", "256x128x64", "tma_cooperative", "2x1x1");
  if (options.variant == "bf16_nt_cluster2n_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2N, Coop, CoopEpi>(options, "bf16", "NT", "128x128x64", "tma_cooperative", "1x2x1");
  if (options.variant == "tf32_nt_coop_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C1, Coop, CoopEpi>(options, "tf32", "NT", "128x128x64", "tma_cooperative");
  if (options.variant == "tf32_nt_cpasync_pingpong_128x64")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, TTF32, C1, CpPingpong, NoSmemEpi>(options, "tf32", "NT", "128x64x32", "cpasync_pingpong");
  if (options.variant == "tf32_nt_cpasync_coop_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C1, CpCoop, NoSmemEpi>(options, "tf32", "NT", "128x128x64", "cpasync_cooperative");
  if (options.variant == "tf32_nt_streamk_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C1, Coop, CoopEpi, StreamK>(options, "tf32", "NT", "128x128x64", "tma_cooperative", "1x1x1", "stream_k");
  if (options.variant == "fp8_nt_fastacc_coop_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C1, FP8Coop, CoopEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_cooperative_fast_accum");
  if (options.variant == "fp8_nt_fastacc_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C1, FP8, NoSmemEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_warp_specialized_fast_accum");
  if (options.variant == "fp8_nt_fastacc_pingpong_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C1, FP8Pingpong, PingpongEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_pingpong_fast_accum");
  if (options.variant == "fp8_nt_standard_coop_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C1, Coop, CoopEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_cooperative_standard_accum");
  if (options.variant == "fp8_nt_fastacc_cluster2m_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C2M, FP8Coop, CoopEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_cooperative_fast_accum", "2x1x1");
  if (options.variant == "f16_nt_tma_cluster2m_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2M, Tma, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "tma_non_warp_specialized", "2x1x1");
  if (options.variant == "f16_nt_tma_cluster2n_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2N, Tma, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "tma_non_warp_specialized", "1x2x1");
  if (options.variant == "f16_tn_ws_cluster2m_128x128")
    return dispatch<cutlass::half_t, CM, CM, cutlass::half_t, T128, C2M, WS, NoSmemEpi>(options, "fp16", "TN", "128x128x64", "tma_warp_specialized_no_smem_epilogue", "2x1x1");
  if (options.variant == "f16_tn_ws_cluster2n_128x128")
    return dispatch<cutlass::half_t, CM, CM, cutlass::half_t, T128, C2N, WS, NoSmemEpi>(options, "fp16", "TN", "128x128x64", "tma_warp_specialized_no_smem_epilogue", "1x2x1");
  if (options.variant == "f16_tt_pingpong_cluster2m_64x128")
    return dispatch<cutlass::half_t, CM, RM, cutlass::half_t, T64, C2M, Pingpong, PingpongEpi>(options, "fp16", "TT", "64x128x64", "tma_pingpong", "2x1x1");
  if (options.variant == "f16_tt_pingpong_cluster2n_64x128")
    return dispatch<cutlass::half_t, CM, RM, cutlass::half_t, T64, C2N, Pingpong, PingpongEpi>(options, "fp16", "TT", "64x128x64", "tma_pingpong", "1x2x1");
  if (options.variant == "f16_nt_cpasync_cluster2m_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2M, Cp, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_warp_specialized", "2x1x1");
  if (options.variant == "f16_nt_cpasync_cluster2n_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2N, Cp, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_warp_specialized", "1x2x1");
  if (options.variant == "f16_nt_cpasync_pingpong_cluster2m_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2M, CpPingpong, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_pingpong", "2x1x1");
  if (options.variant == "f16_nt_cpasync_coop_cluster2m_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2M, CpCoop, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_cooperative", "2x1x1");
  if (options.variant == "bf16_tn_ws_cluster2m_128x128")
    return dispatch<cutlass::bfloat16_t, CM, CM, cutlass::bfloat16_t, T128, C2M, WS, NoSmemEpi>(options, "bf16", "TN", "128x128x64", "tma_warp_specialized_no_smem_epilogue", "2x1x1");
  if (options.variant == "bf16_tt_pingpong_cluster2n_64x128")
    return dispatch<cutlass::bfloat16_t, CM, RM, cutlass::bfloat16_t, T64, C2N, Pingpong, PingpongEpi>(options, "bf16", "TT", "64x128x64", "tma_pingpong", "1x2x1");
  if (options.variant == "bf16_nt_cpasync_coop_cluster2m_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2M, CpCoop, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "cpasync_cooperative", "2x1x1");
  if (options.variant == "bf16_nt_cpasync_pingpong_cluster2m_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2M, CpPingpong, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "cpasync_pingpong", "2x1x1");
  if (options.variant == "tf32_nt_coop_cluster2m_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C2M, Coop, CoopEpi>(options, "tf32", "NT", "128x128x64", "tma_cooperative", "2x1x1");
  if (options.variant == "tf32_nt_coop_cluster2n_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C2N, Coop, CoopEpi>(options, "tf32", "NT", "128x128x64", "tma_cooperative", "1x2x1");
  if (options.variant == "fp8_nt_fastacc_cluster2n_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C2N, FP8Coop, CoopEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_cooperative_fast_accum", "1x2x1");
  if (options.variant == "fp8_nt_fastacc_pingpong_cluster2m_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C2M, FP8Pingpong, PingpongEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_pingpong_fast_accum", "2x1x1");
  if (options.variant == "fp8_nt_standard_coop_cluster2m_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C2M, Coop, CoopEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_cooperative_standard_accum", "2x1x1");
  if (options.variant == "f16_nt_cpasync_pingpong_cluster2n_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2N, CpPingpong, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_pingpong", "1x2x1");
  if (options.variant == "f16_nt_cpasync_coop_cluster2n_128x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T128, C2N, CpCoop, NoSmemEpi>(options, "fp16", "NT", "128x128x64", "cpasync_cooperative", "1x2x1");
  if (options.variant == "bf16_tn_ws_cluster2n_128x128")
    return dispatch<cutlass::bfloat16_t, CM, CM, cutlass::bfloat16_t, T128, C2N, WS, NoSmemEpi>(options, "bf16", "TN", "128x128x64", "tma_warp_specialized_no_smem_epilogue", "1x2x1");
  if (options.variant == "bf16_tt_pingpong_cluster2m_64x128")
    return dispatch<cutlass::bfloat16_t, CM, RM, cutlass::bfloat16_t, T64, C2M, Pingpong, PingpongEpi>(options, "bf16", "TT", "64x128x64", "tma_pingpong", "2x1x1");
  if (options.variant == "bf16_nt_cpasync_coop_cluster2n_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2N, CpCoop, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "cpasync_cooperative", "1x2x1");
  if (options.variant == "bf16_nt_cpasync_pingpong_cluster2n_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2N, CpPingpong, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "cpasync_pingpong", "1x2x1");
  if (options.variant == "fp8_nt_fastacc_pingpong_cluster2n_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C2N, FP8Pingpong, PingpongEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_pingpong_fast_accum", "1x2x1");
  if (options.variant == "fp8_nt_standard_coop_cluster2n_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C2N, Coop, CoopEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_cooperative_standard_accum", "1x2x1");
  if (options.variant == "f16_nn_coop_cluster2m_128x256")
    return dispatch<cutlass::half_t, RM, RM, cutlass::half_t, T256, C2M, Coop, CoopEpi>(options, "fp16", "NN", "128x256x64", "tma_cooperative", "2x1x1");
  if (options.variant == "f16_nn_coop_cluster2n_128x256")
    return dispatch<cutlass::half_t, RM, RM, cutlass::half_t, T256, C2N, Coop, CoopEpi>(options, "fp16", "NN", "128x256x64", "tma_cooperative", "1x2x1");
  if (options.variant == "bf16_nn_coop_cluster2m_128x256")
    return dispatch<cutlass::bfloat16_t, RM, RM, cutlass::bfloat16_t, T256, C2M, Coop, CoopEpi>(options, "bf16", "NN", "128x256x64", "tma_cooperative", "2x1x1");
  if (options.variant == "bf16_nn_coop_cluster2n_128x256")
    return dispatch<cutlass::bfloat16_t, RM, RM, cutlass::bfloat16_t, T256, C2N, Coop, CoopEpi>(options, "bf16", "NN", "128x256x64", "tma_cooperative", "1x2x1");
  if (options.variant == "bf16_nt_cluster2m_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2M, Coop, CoopEpi>(options, "bf16", "NT", "128x128x64", "tma_cooperative", "2x1x1");
  if (options.variant == "fp8_nt_fastacc_ws_cluster2m_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C2M, FP8, NoSmemEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_warp_specialized_fast_accum", "2x1x1");
  if (options.variant == "fp8_nt_fastacc_ws_cluster2n_128x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8, C2N, FP8, NoSmemEpi>(options, "fp8_e4m3", "NT", "128x128x128", "tma_warp_specialized_fast_accum", "1x2x1");
  if (options.variant == "tf32_nt_cpasync_coop_cluster2m_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C2M, CpCoop, NoSmemEpi>(options, "tf32", "NT", "128x128x64", "cpasync_cooperative", "2x1x1");
  if (options.variant == "tf32_nt_cpasync_coop_cluster2n_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C2N, CpCoop, NoSmemEpi>(options, "tf32", "NT", "128x128x64", "cpasync_cooperative", "1x2x1");
  if (options.variant == "tf32_nt_cpasync_pingpong_cluster2m_128x64")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, TTF32, C2M, CpPingpong, NoSmemEpi>(options, "tf32", "NT", "128x64x32", "cpasync_pingpong", "2x1x1");
  if (options.variant == "tf32_nt_cpasync_pingpong_cluster2n_128x64")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, TTF32, C2N, CpPingpong, NoSmemEpi>(options, "tf32", "NT", "128x64x32", "cpasync_pingpong", "1x2x1");
  if (options.variant == "bf16_nt_tma_cluster2m_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2M, Tma, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "tma_non_warp_specialized", "2x1x1");
  if (options.variant == "bf16_nt_tma_cluster2n_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2N, Tma, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "tma_non_warp_specialized", "1x2x1");
  if (options.variant == "f16_tn_coop_cluster2m_128x128")
    return dispatch<cutlass::half_t, CM, CM, cutlass::half_t, T128, C2M, Coop, CoopEpi>(options, "fp16", "TN", "128x128x64", "tma_cooperative", "2x1x1");
  if (options.variant == "f16_tn_coop_cluster2n_128x128")
    return dispatch<cutlass::half_t, CM, CM, cutlass::half_t, T128, C2N, Coop, CoopEpi>(options, "fp16", "TN", "128x128x64", "tma_cooperative", "1x2x1");
  if (options.variant == "f16_tt_coop_cluster2m_128x128")
    return dispatch<cutlass::half_t, CM, RM, cutlass::half_t, T128, C2M, Coop, CoopEpi>(options, "fp16", "TT", "128x128x64", "tma_cooperative", "2x1x1");
  if (options.variant == "f16_tt_coop_cluster2n_128x128")
    return dispatch<cutlass::half_t, CM, RM, cutlass::half_t, T128, C2N, Coop, CoopEpi>(options, "fp16", "TT", "128x128x64", "tma_cooperative", "1x2x1");
  if (options.variant == "bf16_nt_cpasync_cluster2m_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2M, Cp, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "cpasync_warp_specialized", "2x1x1");
  if (options.variant == "bf16_nt_cpasync_cluster2n_128x128")
    return dispatch<cutlass::bfloat16_t, RM, CM, cutlass::bfloat16_t, T128, C2N, Cp, NoSmemEpi>(options, "bf16", "NT", "128x128x64", "cpasync_warp_specialized", "1x2x1");
  if (options.variant == "tf32_nt_ws_cluster2m_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C2M, WS, NoSmemEpi>(options, "tf32", "NT", "128x128x64", "tma_warp_specialized_no_smem_epilogue", "2x1x1");
  if (options.variant == "tf32_nt_ws_cluster2n_128x128")
    return dispatch<cutlass::tfloat32_t, RM, CM, float, T128, C2N, WS, NoSmemEpi>(options, "tf32", "NT", "128x128x64", "tma_warp_specialized_no_smem_epilogue", "1x2x1");
  if (options.variant == "f16_nt_coop_cluster2m_256x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T256M, C2M, Coop, CoopEpi>(options, "fp16", "NT", "256x128x64", "tma_cooperative", "2x1x1");
  if (options.variant == "f16_nt_coop_cluster2n_256x128")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T256M, C2N, Coop, CoopEpi>(options, "fp16", "NT", "256x128x64", "tma_cooperative", "1x2x1");
  if (options.variant == "f16_nt_coop_cluster2m_128x256")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T256, C2M, Coop, CoopEpi>(options, "fp16", "NT", "128x256x64", "tma_cooperative", "2x1x1");
  if (options.variant == "f16_nt_coop_cluster2n_128x256")
    return dispatch<cutlass::half_t, RM, CM, cutlass::half_t, T256, C2N, Coop, CoopEpi>(options, "fp16", "NT", "128x256x64", "tma_cooperative", "1x2x1");
  if (options.variant == "fp8_nt_fastacc_coop_cluster2m_256x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8_256M, C2M, FP8Coop, CoopEpi>(options, "fp8_e4m3", "NT", "256x128x128", "tma_cooperative_fast_accum", "2x1x1");
  if (options.variant == "fp8_nt_fastacc_coop_cluster2n_256x128")
    return dispatch<cutlass::float_e4m3_t, RM, CM, cutlass::half_t, TFP8_256M, C2N, FP8Coop, CoopEpi>(options, "fp8_e4m3", "NT", "256x128x128", "tma_cooperative_fast_accum", "1x2x1");
  if (options.variant == "f16_tt_pingpong_cluster2m_128x128")
    return dispatch<cutlass::half_t, CM, RM, cutlass::half_t, T128, C2M, Pingpong, PingpongEpi>(options, "fp16", "TT", "128x128x64", "tma_pingpong", "2x1x1");
  if (options.variant == "f16_tt_pingpong_cluster2n_128x128")
    return dispatch<cutlass::half_t, CM, RM, cutlass::half_t, T128, C2N, Pingpong, PingpongEpi>(options, "fp16", "TT", "128x128x64", "tma_pingpong", "1x2x1");
  throw std::runtime_error("unknown variant: " + options.variant);
}

}  // namespace cutlass_full_gpu

int main(int argc, char const** argv) {
  try {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::string profile = cutlass_full_gpu::env_or("GEMM_DELIVERY_GPU", "h100");
    if (!cutlass_full_gpu::matches_gpu_profile(props.name, profile) || props.major != 9 || props.minor != 0) {
      throw std::runtime_error(
          std::string("requires selected NVIDIA ") + profile + " sm_90 profile; got " + props.name);
    }
    return cutlass_full_gpu::run_variant(cutlass_full_gpu::parse_options(argc, argv));
  } catch (std::exception const& error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 1;
  }
}
