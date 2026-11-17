// Copyright (c) 2026 The CycleScope authors. SPDX-License-Identifier: MIT
//
// Compile with:
//   nvcc ... -DCUTLASS_DELIVERY_EXAMPLE='"examples/.../example.cu"' \
//     cutlass_example_cycle_wrapper.cu
//
// The selected upstream example remains the source of the CUTLASS algorithm,
// argument validation, data initialization, and correctness check.  This file
// only enforces the selected SM90 product and reports the in-kernel full-GPU time span
// recorded by harness/override/cutlass/device_kernel.h.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include "cuda_profiler_compat.h"

#ifndef CUTLASS_DELIVERY_EXAMPLE
#error "CUTLASS_DELIVERY_EXAMPLE must name one upstream CUTLASS example source"
#endif

#define main cutlass_delivery_upstream_main
#include CUTLASS_DELIVERY_EXAMPLE
#undef main

namespace cutlass_delivery {

constexpr int kMarkerFields = 7;
constexpr int kMaxWarps = 16;
constexpr int kMaxCtas = 16384;
constexpr int kMaxLaunches = 16;

void cuda_check(cudaError_t status, char const* expression) {
  if (status != cudaSuccess) {
    std::ostringstream stream;
    stream << expression << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(stream.str());
  }
}

#define DELIVERY_CUDA_CHECK(expr) ::cutlass_delivery::cuda_check((expr), #expr)

class ScopedCaptureMarkers {
 public:
  ScopedCaptureMarkers() {
    std::size_t marker_rows_per_launch = std::size_t(kMaxCtas) * kMaxWarps;
    std::size_t marker_count =
        std::size_t(kMaxLaunches) * marker_rows_per_launch * kMarkerFields;
    try {
      DELIVERY_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&markers_),
                                     marker_count * sizeof(unsigned long long)));
      DELIVERY_CUDA_CHECK(cudaMemset(markers_, 0,
                                     marker_count * sizeof(unsigned long long)));
      DELIVERY_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&grid_ids_),
                                     kMaxLaunches * sizeof(unsigned long long)));
      DELIVERY_CUDA_CHECK(cudaMemset(grid_ids_, 0,
                                     kMaxLaunches * sizeof(unsigned long long)));
      DELIVERY_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&grids_),
                                     kMaxLaunches * sizeof(unsigned long long)));
      DELIVERY_CUDA_CHECK(cudaMemset(grids_, 0,
                                     kMaxLaunches * sizeof(unsigned long long)));
      DELIVERY_CUDA_CHECK(cudaMemcpyToSymbol(cutlass::cutlass_single_cta_marker_ptr,
                                             &markers_, sizeof(markers_)));
      DELIVERY_CUDA_CHECK(cudaMemcpyToSymbol(
          cutlass::cutlass_single_cta_marker_grid_ids,
          &grid_ids_, sizeof(grid_ids_)));
      DELIVERY_CUDA_CHECK(cudaMemcpyToSymbol(
          cutlass::cutlass_single_cta_marker_grids,
          &grids_, sizeof(grids_)));
    } catch (...) {
      reset();
      throw;
    }
  }

  ScopedCaptureMarkers(ScopedCaptureMarkers const&) = delete;
  ScopedCaptureMarkers& operator=(ScopedCaptureMarkers const&) = delete;

  ~ScopedCaptureMarkers() { reset(); }

 private:
  void reset() noexcept {
    unsigned long long* empty = nullptr;
    // A trace-only process normally exits immediately after this launch, but
    // clear the device symbols before freeing their backing storage so later
    // launches can never observe a dangling measurement pointer.
    cudaMemcpyToSymbol(cutlass::cutlass_single_cta_marker_ptr,
                       &empty, sizeof(empty));
    cudaMemcpyToSymbol(cutlass::cutlass_single_cta_marker_grid_ids,
                       &empty, sizeof(empty));
    cudaMemcpyToSymbol(cutlass::cutlass_single_cta_marker_grids,
                       &empty, sizeof(empty));
    if (grid_ids_ != nullptr) cudaFree(grid_ids_);
    if (grids_ != nullptr) cudaFree(grids_);
    if (markers_ != nullptr) cudaFree(markers_);
    grid_ids_ = nullptr;
    grids_ = nullptr;
    markers_ = nullptr;
  }

  unsigned long long* markers_ = nullptr;
  unsigned long long* grid_ids_ = nullptr;
  unsigned long long* grids_ = nullptr;
};

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

int requested_sample_count();

int requested_warmup_argument(int argc, char const** argv) {
  for (int i = 0; i < argc; ++i) {
    std::string argument = argv[i] ? argv[i] : "";
    for (char const* name : {"--warmup=", "--warmups="}) {
      if (argument.rfind(name, 0) == 0) {
        return std::max(0, std::stoi(argument.substr(std::strlen(name))));
      }
    }
  }
  return 0;
}

// Trace warmups must reproduce each family's timed protocol exactly, counted
// as the total number of same-target-kernel launches that execute before the
// first recorded timing sample.  The CLI-driven sample families (55/61/63)
// keep their upstream benchmark, so upstream_main first issues one
// correctness launch, then the profiling loop (its warmup iterations for the
// families with a ``--warmup`` option, then measured iterations).  The
// post-example family 56 clears the markers after upstream_main and re-issues
// one correctness launch context plus three wrapper warmups.  Dense cases do
// not go through this wrapper path; they use the runner's own warmup loop.
int requested_trace_warmups(int argc, char const** argv, int warmup_argument) {
  (void)argc;
  (void)argv;
#if defined(CUTLASS_DELIVERY_EXAMPLE55_F16_FP8) || \
    defined(CUTLASS_DELIVERY_EXAMPLE55_INT4_BF16) || \
    defined(CUTLASS_DELIVERY_EXAMPLE55_INT4_FP8)
  // upstream_main: 1 correctness launch + warmup_argument loop iterations.
  return 1 + std::max(0, warmup_argument);
#elif defined(CUTLASS_DELIVERY_EXAMPLE61) || defined(CUTLASS_DELIVERY_EXAMPLE63)
  // upstream_main: 1 correctness launch, then --iterations measured launches.
  return 1;
#elif defined(CUTLASS_DELIVERY_EXAMPLE56)
  // Timing path: upstream_main correctness launch + three post-example
  // wrapper warmups before the first recorded sample.
  return 4;
#elif defined(CUTLASS_DELIVERY_EXAMPLE52)
  // Timing path: the post example issues three wrapper warmups of the fused
  // kernel before the recorded samples (under cluster_sm upstream_main is
  // skipped both in timing and in the trace entry), so the trace entry
  // covers exactly those three post warmups.
  return 3;
#elif defined(CUTLASS_DELIVERY_EXAMPLE53) || defined(CUTLASS_DELIVERY_EXAMPLE54)
  // Timing path: 1 correctness launch, then --iterations measured launches.
  return 1;
#elif defined(CUTLASS_DELIVERY_EXAMPLE57) || \
      defined(CUTLASS_DELIVERY_EXAMPLE68) || \
      defined(CUTLASS_DELIVERY_EXAMPLE68_SPARSE_GROUPS) || \
      defined(CUTLASS_DELIVERY_EXAMPLE69_MIXED) || \
      defined(CUTLASS_DELIVERY_EXAMPLE69_INT4_BF16) || \
      defined(CUTLASS_DELIVERY_EXAMPLE69_INT4_FP8)
  // Timing path: 1 correctness launch, then the --iterations measured loop
  // of the same Gemm instantiation.
  return 1;
#elif defined(CUTLASS_DELIVERY_EXAMPLE67) || defined(CUTLASS_DELIVERY_EXAMPLE67_GROUPWISE)
  // Timing path: 1 correctness launch, then a warmup+measured loop that
  // starts its timer at iter==warmup.
  return 1 + std::max(0, warmup_argument);
#elif defined(CUTLASS_DELIVERY_EXAMPLE62)
  // Timing path: the post example's three sparse-runner warmup run()s
  // precede the recorded samples; the trace entry covers those three.
  return 3;
#elif defined(CUTLASS_DELIVERY_EXAMPLE49) || defined(CUTLASS_DELIVERY_EXAMPLE50)
  // Timing path: the example/post runner issues exactly one target launch
  // per invocation (these Options have no warmup knob); the trace entry
  // reproduces that single launch with no extra pre-sample launches.
  return 0;
#else
  return 0;
#endif
}

#define CUTLASS_DELIVERY_CHECK(expr)                                     \
  do {                                                                  \
    cutlass::Status status_ = (expr);                                   \
    if (status_ != cutlass::Status::kSuccess) {                         \
      throw std::runtime_error("cutlass call failed at " #expr);        \
    }                                                                   \
  } while (0)

template <typename Gemm, typename Arguments, typename Launcher>
int capture_one_initialized_gemm(Arguments arguments, Launcher launcher,
                                 int warmups) {
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_DELIVERY_CHECK(gemm.can_implement(arguments));
  CUTLASS_DELIVERY_CHECK(gemm.initialize(arguments, workspace.get()));
  // Trace and timing must take the same in-kernel marker branch.  The former
  // used to leave these symbols null for upstream examples, which preserved
  // SASS identity but changed dynamic control flow and made marker timing
  // incomparable to the captured simulator interval.
  for (int warmup = 0; warmup < warmups; ++warmup) {
    CUTLASS_DELIVERY_CHECK(launcher(gemm));
    DELIVERY_CUDA_CHECK(cudaDeviceSynchronize());
  }
  ScopedCaptureMarkers markers;
  DELIVERY_CUDA_CHECK(cudaProfilerStart());
  CUTLASS_DELIVERY_CHECK(launcher(gemm));
  DELIVERY_CUDA_CHECK(cudaProfilerStop());
  DELIVERY_CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

template <typename Gemm, typename Arguments>
int capture_one_default_gemm(Arguments arguments, int warmups) {
  return capture_one_initialized_gemm<Gemm>(
      arguments, [](Gemm& gemm) { return gemm.run(); }, warmups);
}

// Some delivery runners keep an already can_implement'd and initialized
// fused GEMM as a member (derived 53) or wrap one in their own runner type
// (example 62) whose run() re-initializes from the captured arguments.
// Bracket exactly one launch of that live instance; warmups replay the
// timed protocol's pre-sample launches of the same kernel.  run() may
// return cutlass::Status or void; check it only when it is a status.
template <typename Gemm>
inline void run_and_check(Gemm& gemm) {
  if constexpr (std::is_void_v<decltype(gemm.run())>) {
    gemm.run();
  } else {
    CUTLASS_DELIVERY_CHECK(gemm.run());
  }
}

template <typename Gemm>
int capture_one_live_gemm(Gemm& gemm, int warmups) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    run_and_check(gemm);
    DELIVERY_CUDA_CHECK(cudaDeviceSynchronize());
  }
  ScopedCaptureMarkers markers;
  DELIVERY_CUDA_CHECK(cudaProfilerStart());
  run_and_check(gemm);
  DELIVERY_CUDA_CHECK(cudaProfilerStop());
  DELIVERY_CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

#if defined(CUTLASS_DELIVERY_EXAMPLE49)
// Reproduce ExampleRunner::run's argument construction exactly — same
// operand blocks/strides, same epilogue thread wiring including the custom
// EVT nesting for the custom_evt mode — then bracket one initialized launch
// in the profiler window.  UseCustomEVT must match the runner's own template
// argument; the class guarantees EVT only on TmaWarpSpecialized epilogues.
template <bool UseCustomEVT, class Runner>
int capture_example49_runner(Runner& runner, Options const& options,
                             cutlass::KernelHardwareInfo const& hw_info,
                             int warmups) {
  using Gemm = typename Runner::Gemm;
  using ProblemShapeType = typename Runner::ProblemShapeType;
  ProblemShapeType problem_size{options.m, options.n, options.k, options.l};
  runner.initialize(problem_size);
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    {runner.block_A.get(), runner.stride_A, runner.block_B.get(), runner.stride_B},
    {{}, // epilogue.thread
     runner.block_C.get(), runner.stride_C, runner.block_D.get(), runner.stride_D},
    hw_info
  };
  if constexpr (UseCustomEVT) {
    arguments.epilogue.thread =
      {                      // ternary op : beta * C + (alpha * acc)
        {{options.beta}},    // leaf op+args : beta
        {},                  // leaf op+args : C
        {                    // binary op : alpha * acc
          {{options.alpha}}, // leaf op+args : alpha
          {},                // leaf op+args : acc
          {}                 // binary args : multiplies
        },                   // end binary op
        {}                   // ternary args : multiply_add
      };                     // end ternary op
  } else {
    arguments.epilogue.thread.alpha = options.alpha;
    arguments.epilogue.thread.beta = options.beta;
  }
  return capture_one_initialized_gemm<Gemm>(
      arguments, [](Gemm& gemm) { return gemm.run(); }, warmups);
}
#endif

#if defined(CUTLASS_DELIVERY_EXAMPLE50)
// Example 50 assembles its Gemm type inside main().  Rebuild the identical
// type at namespace scope so the isolated trace entry can name it; every
// ingredient below is a compile-time constant of the included source, so
// the instantiation — and therefore the kernel symbol — matches the timed
// launch exactly (int8 TN, 128x64x128 tile, 1x2x1 cluster, 8 stages).
using Element50A = int8_t;
using Element50B = int8_t;
using Element50Acc = int32_t;
using Element50Output = int8_t;
using Layout50A = cutlass::layout::RowMajor;
using Layout50B = cutlass::layout::ColumnMajor;
using Layout50C = cutlass::layout::ColumnMajor;
using Layout50D = cutlass::layout::ColumnMajor;
using TileShape50 = cute::Shape<cute::_128, cute::_64, cute::_128>;
using ClusterShape50 = cute::Shape<cute::_1, cute::_2, cute::_1>;
constexpr int PipelineStages50 = 8;
using DispatchPolicy50 = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<
    PipelineStages50, ClusterShape50, cutlass::gemm::KernelTmaWarpSpecialized>;
static constexpr cute::GMMA::Major GmmaMajor50A = cute::GMMA::Major::K;
static constexpr cute::GMMA::Major GmmaMajor50B = cute::GMMA::Major::K;
using TiledMma50 = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
    Element50A, Element50B, Element50Acc, TileShape50,
    GmmaMajor50A, GmmaMajor50B>()));
using GmemTiledCopy50A =
    std::conditional<cute::size(cute::shape<1>(ClusterShape50{})) == 1,
                     cute::SM90_TMA_LOAD, cute::SM90_TMA_LOAD_MULTICAST>::type;
using GmemTiledCopy50B =
    std::conditional<cute::size(cute::shape<0>(ClusterShape50{})) == 1,
                     cute::SM90_TMA_LOAD, cute::SM90_TMA_LOAD_MULTICAST>::type;
using SmemLayoutAtom50A =
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GmmaMajor50A, Element50A,
        decltype(cute::get<0>(TileShape50{})),
        decltype(cute::get<2>(TileShape50{}))>());
using SmemLayoutAtom50B =
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GmmaMajor50B, Element50B,
        decltype(cute::get<1>(TileShape50{})),
        decltype(cute::get<2>(TileShape50{}))>());
using CollectiveMainloop50 = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy50,
    TileShape50,
    Element50A,
    cutlass::gemm::TagToStrideA_t<Layout50A>,
    Element50B,
    cutlass::gemm::TagToStrideB_t<Layout50B>,
    TiledMma50,
    GmemTiledCopy50A,
    SmemLayoutAtom50A,
    void, // Does not need a SmemCopyAtom, since A is read directly from SMEM
    cute::identity,
    GmemTiledCopy50B,
    SmemLayoutAtom50B,
    void, // Does not need a SmemCopyAtom, since B is read directly from SMEM
    cute::identity
  >;
// Break the 128 along TILE_M into chunks of 32, to get a 128B leading dimension
using PreSwizzleLayout50 = cute::Layout<
    cute::Shape<cute::Shape<cute::_32, cute::_4>, cute::_64>,
    cute::Stride<cute::Stride<cute::_1, cute::_2048>, cute::_32>>;
// 128 threads loading 16 elements each (to get vectorized global stores)
using TileShapeS2R50 = cute::Shape<cute::_128, cute::_16>;
// Layout to ensure bank-conflict free loads & stores
using SmemLayout50 = cute::ComposedLayout<
    cute::Swizzle<3, 4, 3>,
    smem_ptr_flag_bits<sizeof_bits<Element50Acc>::value>,
    PreSwizzleLayout50>;
// Tiled copy from Smem to Registers
using TiledCopyS2R50 = cute::TiledCopy<
    cute::Copy_Atom<cute::DefaultCopy, Element50Acc>,
    cute::Layout<cute::Shape<cute::_128, cute::_16>,
                 cute::Stride<cute::_16, cute::_1>>,
    TileShapeS2R50>;
using Epilogue50 = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
  cutlass::epilogue::collective::Epilogue<
    cutlass::gemm::TagToStrideC_t<Layout50C>,
    cutlass::gemm::TagToStrideC_t<Layout50D>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    SmemLayout50,
    cute::Copy_Atom<cute::DefaultCopy, Element50Acc>,
    TiledCopyS2R50,
    cute::Copy_Atom<cute::DefaultCopy, Element50Output>>>;
using GemmKernel50 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop50,
    Epilogue50
  >;
using Example50Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel50>;
#endif

int run_selected_trace_example(int argc, char const** argv) {
  int const trace_warmups = requested_trace_warmups(
      argc, argv, requested_warmup_argument(argc, argv));
#if defined(CUTLASS_DELIVERY_EXAMPLE52)
  // Match timing: finish upstream verification/benchmarks, then construct
  // a fresh runner for the three warmups and the profiler-scoped launch.
  // Under cluster_sm the timed path skips upstream_main entirely (skip_upstream
  // in run()), so the trace entry skips it too; replaying it would only add
  // non-target launches and trips the unfused-optimized benchmark's
  // "Error Internal" on some cpasync variants.
  if (env_or("CUTLASS_DELIVERY_MODE", "full_gpu") != "cluster_sm") {
    int const upstream_return_code = cutlass_delivery_upstream_main(argc, argv);
    DELIVERY_CUDA_CHECK(cudaDeviceSynchronize());
    if (upstream_return_code != 0) return upstream_return_code;
  }
  example::Options options;
  options.parse(argc, argv);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto capture_mode = [&](auto& runner) {
    using Gemm = typename std::decay_t<decltype(runner)>::Gemm;
    return capture_one_initialized_gemm<Gemm>(
        runner.arguments,
        [](Gemm& gemm) { return gemm.run(); },
        trace_warmups);
  };
  switch (options.mode) {
    case 0: {
      using Runner = example::ExampleRunner<
          cutlass::half_t, cutlass::layout::RowMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::ColumnMajor, example::NoGather,
          cutlass::half_t, cutlass::layout::RowMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::RowMajor, example::IndexedGather<int>,
          float, float>;
      Runner runner(options, hw_info);
      return capture_mode(runner);
    }
    case 1: {
      using Runner = example::ExampleRunner<
          cutlass::half_t, cutlass::layout::RowMajor, example::NoGather,
          cutlass::half_t, cutlass::layout::ColumnMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::ColumnMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::ColumnMajor, example::IndexedGather<int>,
          float, float>;
      Runner runner(options, hw_info);
      return capture_mode(runner);
    }
    case 2: {
      using Runner = example::ExampleRunner<
          cutlass::half_t, cutlass::layout::ColumnMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::RowMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::RowMajor, example::NoGather,
          cutlass::half_t, cutlass::layout::RowMajor, example::NoGather,
          float, float>;
      Runner runner(options, hw_info);
      return capture_mode(runner);
    }
    default:
      throw std::runtime_error("unsupported example 52 gather mode");
  }
#elif defined(CUTLASS_DELIVERY_EXAMPLE55_F16_FP8)
  MixedDtypeOptions options;
  options.parse(argc, argv);
  ::initialize(options);
  if (options.mode == MixedDtypeGemmMode::ConvertOnly) {
    return capture_one_default_gemm<GemmConvertOnly>(
        ::args_from_options<typename GemmConvertOnly::Arguments>(options), trace_warmups);
  }
  if (options.mode == MixedDtypeGemmMode::ScaleOnly) {
    return capture_one_default_gemm<GemmScaleOnly>(
        ::args_from_options<typename GemmScaleOnly::Arguments>(options), trace_warmups);
  }
  if (options.mode == MixedDtypeGemmMode::ScaleWithZeroPoint) {
    return capture_one_default_gemm<GemmScaleWithZeroPoint>(
        ::args_from_options<typename GemmScaleWithZeroPoint::Arguments>(options), trace_warmups);
  }
  throw std::runtime_error("unsupported example 55 f16/fp8 mode for trace capture");
#elif defined(CUTLASS_DELIVERY_EXAMPLE55_INT4_BF16)
  Options options;
  options.parse(argc, argv);
  ::initialize(options);
  if (options.mode == MixedDtypeGemmMode::ConvertOnly) {
    if (options.shuffle) {
      return capture_one_default_gemm<GemmConvertOnlyShuffled>(
          ::args_from_options<GemmConvertOnlyShuffled>(options), trace_warmups);
    }
    return capture_one_default_gemm<GemmConvertOnly>(
        ::args_from_options<GemmConvertOnly>(options), trace_warmups);
  }
  if (options.mode == MixedDtypeGemmMode::ScaleOnly) {
    if (options.shuffle) {
      return capture_one_default_gemm<GemmScaleOnlyShuffled>(
          ::args_from_options<GemmScaleOnlyShuffled>(options), trace_warmups);
    }
    return capture_one_default_gemm<GemmScaleOnly>(
        ::args_from_options<GemmScaleOnly>(options), trace_warmups);
  }
  if (options.mode == MixedDtypeGemmMode::ScaleWithZeroPoint) {
    if (options.shuffle) {
      return capture_one_default_gemm<GemmScaleWithZeroPointShuffled>(
          ::args_from_options<GemmScaleWithZeroPointShuffled>(options), trace_warmups);
    }
    return capture_one_default_gemm<GemmScaleWithZeroPoint>(
        ::args_from_options<GemmScaleWithZeroPoint>(options), trace_warmups);
  }
  throw std::runtime_error("unsupported example 55 int4/bf16 mode for trace capture");
#elif defined(CUTLASS_DELIVERY_EXAMPLE55_INT4_FP8)
  Options options;
  options.parse(argc, argv);
  ::initialize(options);
  if (options.shuffle) {
    return capture_one_default_gemm<GemmShuffled>(
        ::args_from_options<GemmShuffled>(options), trace_warmups);
  }
  return capture_one_default_gemm<GemmScaleOnly>(
      ::args_from_options<GemmScaleOnly>(options), trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE56)
  Options options;
  options.parse(argc, argv);
  // Example 56 keeps its pointer arrays and backing allocations in global
  // objects.  Match the upstream run<GemmT>() setup order exactly: allocate
  // those arrays before initialize() fills and copies their pointers.
  ::allocate(options);
  ::initialize(options);
  std::string mode = env_or("CUTLASS_DELIVERY_POST_MODE", "pingpong");
  if (mode == "cooperative") {
    return capture_one_default_gemm<Gemm>(::args_from_options<Gemm>(options), trace_warmups);
  }
  if (mode == "pingpong") {
    return capture_one_default_gemm<GemmPingpong>(
        ::args_from_options<GemmPingpong>(options), trace_warmups);
  }
  throw std::runtime_error("unsupported example 56 trace mode: " + mode);
#elif defined(CUTLASS_DELIVERY_EXAMPLE61)
  Options options;
  options.parse(argc, argv);
  ::initialize(options);
  return capture_one_default_gemm<Gemm>(::args_from_options(options), trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE63)
  Options options;
  options.parse(argc, argv);
  ::initialize(options);
  auto arguments = ::args_from_options(options);
  bool launch_with_pdl = options.overlap_ratio >= 0;
  return capture_one_initialized_gemm<Gemm>(
      arguments,
      [launch_with_pdl](Gemm& gemm) {
        return gemm.run(nullptr, nullptr, launch_with_pdl);
      },
      trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE49)
  // The qualification's recorded sample is the post-example runner's single
  // device_kernel launch; select the identical runner instantiation by
  // CUTLASS_DELIVERY_POST_MODE and bracket one launch of it.
  Options options;
  options.parse(argc, argv);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  std::string mode = env_or("CUTLASS_DELIVERY_POST_MODE", "custom_evt");
  if (mode == "tma") {
    ExampleRunner<cutlass::gemm::KernelTma,
                  cutlass::epilogue::NoSmemWarpSpecialized> runner;
    return capture_example49_runner<false>(
        runner, options, hw_info, trace_warmups);
  } else if (mode == "tma_ws") {
    ExampleRunner<cutlass::gemm::KernelTmaWarpSpecialized,
                  cutlass::epilogue::NoSmemWarpSpecialized> runner;
    return capture_example49_runner<false>(
        runner, options, hw_info, trace_warmups);
  } else if (mode == "tma_pingpong") {
    ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                  cutlass::epilogue::TmaWarpSpecialized> runner;
    return capture_example49_runner<false>(
        runner, options, hw_info, trace_warmups);
  } else if (mode == "stream_k") {
    ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                  cutlass::epilogue::TmaWarpSpecializedCooperative,
                  cutlass::gemm::collective::StageCountAuto,
                  cutlass::gemm::StreamKScheduler> runner;
    return capture_example49_runner<false>(
        runner, options, hw_info, trace_warmups);
  } else if (mode == "custom_evt") {
    ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                  cutlass::epilogue::TmaWarpSpecializedCooperative,
                  cutlass::gemm::collective::StageCountAuto,
                  cutlass::gemm::PersistentScheduler,
                  true> runner;
    return capture_example49_runner<true>(
        runner, options, hw_info, trace_warmups);
  }
  throw std::runtime_error("unsupported example 49 trace mode: " + mode);
#elif defined(CUTLASS_DELIVERY_EXAMPLE50)
  // Same single-launch protocol as the example's main(): one runner, one
  // launch, with the alpha/beta epilogue thread built inline.
  Options options;
  options.parse(argc, argv);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  ExampleRunner<Example50Gemm> runner;
  using Gemm = Example50Gemm;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  ProblemShapeType problem_size{options.m, options.n, options.k, options.l};
  runner.initialize(problem_size);
  typename Gemm::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    {runner.block_A.get(), runner.stride_A, runner.block_B.get(), runner.stride_B},
    {{options.alpha, options.beta}, runner.block_C.get(), runner.stride_C, runner.block_D.get(), runner.stride_D},
    hw_info
  };
  return capture_one_initialized_gemm<Gemm>(
      arguments, [](Gemm& gemm) { return gemm.run(); }, trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE53)
  // The qualification's recorded sample is the process's last device_kernel
  // launch: the final accepted runner's fused GEMM+permute kernel.  Mirror
  // derived main()'s dispatch tail exactly to select that runner, then
  // bracket one launch of its already-initialized fused GEMM.
  example::Options options;
  options.parse(argc, argv);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  int const m = options.problem_size.m();
  int const n = options.problem_size.n();
  int const k = options.problem_size.k();
  int const batch = options.problem_size.batch();
  // 4D/5D permute D-factors with tile 128 require M/N tile alignment. Tail
  // shapes still exercise fused permute through BMM with D dividing batch.
  bool const regular_permute_ok =
      (m % 128 == 0) && (n % 128 == 0) &&
      (m % 16 == 0) && (n % 32 == 0) && (k % 32 == 0);
  int bmm_d = 0;
  for (int d : {8, 4, 2}) {
    if (batch % d == 0) {
      bmm_d = d;
      break;
    }
  }
  using Element53A = cutlass::half_t;
  using Element53B = cutlass::half_t;
  using Element53C = cutlass::half_t;
  using Element53D = cutlass::half_t;
  using Element53Accumulator = float;
  using Element53Epilogue = float;
  // M=64 for TMA epilogue; 1x1x1 cluster per the derived full-H100 shape.
  using TileShape53 = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using ClusterShape53 = cute::Shape<cute::_1, cute::_1, cute::_1>;
  if (bmm_d == 8 || bmm_d == 4 || bmm_d == 2) {
    // Last accepted runner: "Tensor D: RowMajor, Tensor4DPermuteBMM0213<D>".
    auto capture_bmm = [&](auto d_constant) -> int {
      using Runner = example::ExampleRunner<
          Element53A, cutlass::layout::RowMajor, cutlass::layout::NoPermute,
          Element53B, cutlass::layout::RowMajor, cutlass::layout::NoPermute,
          Element53C, cutlass::layout::RowMajor, cutlass::layout::NoPermute,
          Element53D, cutlass::layout::RowMajor,
          cutlass::layout::Tensor4DPermuteBMM0213RowMajor<
              decltype(d_constant)::value>,
          Element53Accumulator, Element53Epilogue,
          TileShape53, ClusterShape53>;
      Runner runner(options, hw_info);
      return capture_one_live_gemm(runner.fused_gemm(), trace_warmups);
    };
    if (bmm_d == 8) return capture_bmm(cute::Int<8>{});
    if (bmm_d == 4) return capture_bmm(cute::Int<4>{});
    return capture_bmm(cute::Int<2>{});
  }
  if (regular_permute_ok) {
    // Last accepted runner: "Tensor D: RowMajor, Tensor5DPermute20314<16,8,4>".
    using Runner = example::ExampleRunner<
        Element53A, cutlass::layout::RowMajor,
        cutlass::layout::Tensor5DPermute20314RowMajor<16, 8, 4>,
        Element53B, cutlass::layout::RowMajor, cutlass::layout::NoPermute,
        Element53C, cutlass::layout::RowMajor, cutlass::layout::NoPermute,
        Element53D, cutlass::layout::RowMajor, cutlass::layout::NoPermute,
        Element53Accumulator, Element53Epilogue,
        TileShape53, ClusterShape53>;
    Runner runner(options, hw_info);
    return capture_one_live_gemm(runner.fused_gemm(), trace_warmups);
  }
  throw std::runtime_error(
      "derived 53: no fused permute variant accepts this problem shape");
#elif defined(CUTLASS_DELIVERY_EXAMPLE54)
  // Timing path: run<Gemm> issues one correctness launch, then --iterations
  // measured launches; the trace entry replays the correctness-equivalent
  // warmup and brackets one measured launch.
  Options<RasterOrderOptions> options;
  options.parse(argc, argv);
  ::initialize(options);
  return capture_one_default_gemm<Gemm>(::args_from_options(options), trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE57)
  // Timed path: the post example's ::run<GemmT>(options) — one correctness
  // launch then the measured loop — is what the markers recorded.  Reproduce
  // its setup and bracket one measured launch of the same schedule.
  Options options;
  options.parse(argc, argv);
  std::string mode = env_or("CUTLASS_DELIVERY_POST_MODE", "pingpong");
  if (mode == "cooperative") {
    ::allocate(options);
    ::initialize(options);
    return capture_one_initialized_gemm<Gemm>(
        ::args_from_options<Gemm>(options),
        [](Gemm& gemm) { return gemm.run(); }, trace_warmups);
  }
  if (mode == "pingpong") {
    ::allocate(options);
    ::initialize(options);
    return capture_one_initialized_gemm<GemmPingpong>(
        ::args_from_options<GemmPingpong>(options),
        [](GemmPingpong& gemm) { return gemm.run(); }, trace_warmups);
  }
  throw std::runtime_error("unsupported example 57 trace mode: " + mode);
#elif defined(CUTLASS_DELIVERY_EXAMPLE62)
  // Timed path: the post example's sparse runner (three warmup run()s, then
  // samples).  initialize(options) fills the global problem shapes and
  // compressed operands that make_args reads — upstream run() always calls
  // it before constructing the runner; skipping it leaves zero shapes and
  // trips the make_tma_copy_desc gmem_prob_shape assertion.
  Options options;
  options.parse(argc, argv);
  if (!::initialize(options)) {
    throw std::runtime_error("example 62: sparse problem initialization failed");
  }
  Runner<Gemm> sparse_gemm(make_args(options));
  return capture_one_live_gemm(sparse_gemm, trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE67)
  // Same run() protocol as example 54: one correctness launch then the
  // measured loop over --iterations.
  Options<RasterOrderOptions> options;
  options.parse(argc, argv);
  ::initialize(options);
  return capture_one_default_gemm<Gemm>(::args_from_options(options), trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE67_GROUPWISE)
  // Stream-K groupwise variant of example 67; identical protocol.
  Options<RasterOrderOptions> options;
  options.parse(argc, argv);
  ::initialize(options);
  return capture_one_default_gemm<Gemm>(
      ::args_from_options<typename Gemm::Arguments>(options), trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE68)
  // Timed path: the post example's ::run(options, true) — correctness launch
  // then the measured loop with host problem shapes available.
  Options<ProblemShape> options;
  options.parse(argc, argv);
  ::allocate(options);
  ::initialize(options);
  return capture_one_initialized_gemm<Gemm>(
      ::args_from_options<typename Gemm::Arguments>(options, true),
      [](Gemm& gemm) { return gemm.run(); }, trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE68_SPARSE_GROUPS)
  // No post example for this binary: the qualification recorded upstream
  // main's last launch, which is the second ::run(options, false) measured
  // loop (host problem shapes unavailable).  Replicate that setup.
  Options<ProblemShape> options;
  options.parse(argc, argv);
  ::allocate(options);
  ::initialize(options);
  return capture_one_initialized_gemm<Gemm>(
      ::args_from_options<typename Gemm::Arguments>(options, false),
      [](Gemm& gemm) { return gemm.run(); }, trace_warmups);
#elif defined(CUTLASS_DELIVERY_EXAMPLE69_MIXED) || \
      defined(CUTLASS_DELIVERY_EXAMPLE69_INT4_BF16) || \
      defined(CUTLASS_DELIVERY_EXAMPLE69_INT4_FP8)
  // Timed path: main() runs exactly one schedule selected by --mode, with
  // host problem shapes unavailable, one correctness launch then the
  // measured loop.  Mirror the mode->Gemm mapping of each source's main().
  Options options;
  options.parse(argc, argv);
  ::allocate(options);
  ::initialize(options);
#if defined(CUTLASS_DELIVERY_EXAMPLE69_MIXED)
  if (options.mode == MixedDtypeGemmMode::ConvertOnly) {
    return capture_one_initialized_gemm<GemmConvertOnly>(
        ::args_from_options<GemmConvertOnly>(options, false),
        [](GemmConvertOnly& gemm) { return gemm.run(); }, trace_warmups);
  }
  if (options.mode == MixedDtypeGemmMode::ScaleOnly) {
    return capture_one_initialized_gemm<GemmScaleOnly>(
        ::args_from_options<GemmScaleOnly>(options, false),
        [](GemmScaleOnly& gemm) { return gemm.run(); }, trace_warmups);
  }
  throw std::runtime_error("unsupported example 69 mixed mode for trace capture");
#elif defined(CUTLASS_DELIVERY_EXAMPLE69_INT4_BF16)
  if (options.mode == MixedDtypeGemmMode::ConvertOnly) {
    if (options.shuffle) {
      return capture_one_initialized_gemm<GemmConvertOnlyShuffled>(
          ::args_from_options<GemmConvertOnlyShuffled>(options, false),
          [](GemmConvertOnlyShuffled& gemm) { return gemm.run(); }, trace_warmups);
    }
    return capture_one_initialized_gemm<GemmConvertOnly>(
        ::args_from_options<GemmConvertOnly>(options, false),
        [](GemmConvertOnly& gemm) { return gemm.run(); }, trace_warmups);
  }
  if (options.mode == MixedDtypeGemmMode::ScaleOnly) {
    if (options.shuffle) {
      return capture_one_initialized_gemm<GemmScaleOnlyShuffled>(
          ::args_from_options<GemmScaleOnlyShuffled>(options, false),
          [](GemmScaleOnlyShuffled& gemm) { return gemm.run(); }, trace_warmups);
    }
    return capture_one_initialized_gemm<GemmScaleOnly>(
        ::args_from_options<GemmScaleOnly>(options, false),
        [](GemmScaleOnly& gemm) { return gemm.run(); }, trace_warmups);
  }
  throw std::runtime_error("unsupported example 69 int4/bf16 mode for trace capture");
#else
  if (options.shuffle) {
    return capture_one_initialized_gemm<GemmShuffled>(
        ::args_from_options<GemmShuffled>(options, false),
        [](GemmShuffled& gemm) { return gemm.run(); }, trace_warmups);
  }
  return capture_one_initialized_gemm<GemmScaleOnly>(
      ::args_from_options<GemmScaleOnly>(options, false),
      [](GemmScaleOnly& gemm) { return gemm.run(); }, trace_warmups);
#endif
#else
  (void)argc;
  (void)argv;
  (void)trace_warmups;
  throw std::runtime_error("official example does not provide an isolated trace entry");
#endif
}

int run_selected_post_example(int argc, char const** argv) {
#if defined(CUTLASS_DELIVERY_EXAMPLE49)
  Options options;
  options.parse(argc, argv);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  std::string mode = env_or("CUTLASS_DELIVERY_POST_MODE", "custom_evt");
  bool passed = false;
  if (mode == "tma") {
    ExampleRunner<cutlass::gemm::KernelTma,
                  cutlass::epilogue::NoSmemWarpSpecialized> runner;
    passed = runner.run(options, hw_info);
  } else if (mode == "tma_ws") {
    ExampleRunner<cutlass::gemm::KernelTmaWarpSpecialized,
                  cutlass::epilogue::NoSmemWarpSpecialized> runner;
    passed = runner.run(options, hw_info);
  } else if (mode == "tma_pingpong") {
    ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                  cutlass::epilogue::TmaWarpSpecialized> runner;
    passed = runner.run(options, hw_info);
  } else if (mode == "stream_k") {
    ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                  cutlass::epilogue::TmaWarpSpecializedCooperative,
                  cutlass::gemm::collective::StageCountAuto,
                  cutlass::gemm::StreamKScheduler> runner;
    passed = runner.run(options, hw_info);
  } else if (mode == "custom_evt") {
    ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                  cutlass::epilogue::TmaWarpSpecializedCooperative,
                  cutlass::gemm::collective::StageCountAuto,
                  cutlass::gemm::PersistentScheduler,
                  true> runner;
    passed = runner.run(options, hw_info);
  } else {
    throw std::runtime_error("unsupported example 49 post mode: " + mode);
  }
  return passed ? 0 : 3;
#elif defined(CUTLASS_DELIVERY_EXAMPLE52)
  // The upstream gather/scatter example benchmarks three different paths in
  // succession: fused, unfused-default, and unfused-optimized.  Correctness
  // has already been checked by upstream_main.  Recreate the selected runner
  // and issue only its fused CUTLASS kernel so every returned marker is one
  // repetition of the same operation.
  example::Options options;
  options.parse(argc, argv);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto issue_samples = [](auto& runner) {
    // Match the other formal runners: warm the exact allocation and fused
    // kernel three times, then let the wrapper retain only the final nine
    // launch IDs as recorded samples.
    for (int warmup = 0; warmup < 3; ++warmup) {
      runner.run_gemm(runner.gemm);
    }
    for (int iteration = 0; iteration < requested_sample_count(); ++iteration) {
      runner.run_gemm(runner.gemm);
    }
  };
  switch (options.mode) {
    case 0: {
      using Runner = example::ExampleRunner<
          cutlass::half_t, cutlass::layout::RowMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::ColumnMajor, example::NoGather,
          cutlass::half_t, cutlass::layout::RowMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::RowMajor, example::IndexedGather<int>,
          float, float>;
      Runner runner(options, hw_info);
      issue_samples(runner);
      break;
    }
    case 1: {
      using Runner = example::ExampleRunner<
          cutlass::half_t, cutlass::layout::RowMajor, example::NoGather,
          cutlass::half_t, cutlass::layout::ColumnMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::ColumnMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::ColumnMajor, example::IndexedGather<int>,
          float, float>;
      Runner runner(options, hw_info);
      issue_samples(runner);
      break;
    }
    case 2: {
      using Runner = example::ExampleRunner<
          cutlass::half_t, cutlass::layout::ColumnMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::RowMajor, example::IndexedGather<int>,
          cutlass::half_t, cutlass::layout::RowMajor, example::NoGather,
          cutlass::half_t, cutlass::layout::RowMajor, example::NoGather,
          float, float>;
      Runner runner(options, hw_info);
      issue_samples(runner);
      break;
    }
    default:
      throw std::runtime_error("unsupported example 52 gather mode");
  }
  return 0;
#elif defined(CUTLASS_DELIVERY_EXAMPLE56)
  Options options;
  options.parse(argc, argv);
  std::string mode = env_or("CUTLASS_DELIVERY_POST_MODE", "pingpong");
  if (mode == "cooperative") {
    return ::run<Gemm>(options);
  }
  if (mode == "pingpong") {
    return ::run<GemmPingpong>(options);
  }
  throw std::runtime_error("unsupported example 56 post mode: " + mode);
#elif defined(CUTLASS_DELIVERY_EXAMPLE57)
  Options options;
  options.parse(argc, argv);
  std::string mode = env_or("CUTLASS_DELIVERY_POST_MODE", "pingpong");
  if (mode == "cooperative") {
    return ::run<Gemm>(options);
  }
  if (mode == "pingpong") {
    return ::run<GemmPingpong>(options);
  }
  throw std::runtime_error("unsupported example 57 post mode: " + mode);
#elif defined(CUTLASS_DELIVERY_EXAMPLE68)
  // Upstream main launches twice (with/without host problem shapes).  Isolate
  // one path after clearing markers so every recorded sample is one cluster.
  Options<ProblemShape> options;
  options.parse(argc, argv);
  return ::run(options, true);
#elif defined(CUTLASS_DELIVERY_EXAMPLE62)
  // The upstream sparse example benchmarks the dense reference last.  Its
  // correctness check has already passed.  Warm and sample the exact sparse
  // runner in this process after clearing markers; the wrapper retains only
  // the final requested launch IDs, so none of the warmups are reported.
  Options options;
  options.parse(argc, argv);
  Runner<Gemm> sparse_gemm(make_args(options));
  for (int warmup = 0; warmup < 3; ++warmup) {
    sparse_gemm.run();
  }
  for (int iteration = 0; iteration < requested_sample_count(); ++iteration) {
    sparse_gemm.run();
  }
  return 0;
#else
  (void)argc;
  (void)argv;
  return 0;
#endif
}

double median(std::vector<double> values) {
  if (values.empty()) {
    throw std::runtime_error("no values for median");
  }
  std::sort(values.begin(), values.end());
  std::size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle]
                           : 0.5 * (values[middle - 1] + values[middle]);
}

std::string json_array(std::vector<double> const& values) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) stream << ",";
    stream << std::fixed << std::setprecision(3) << values[index];
  }
  stream << "]";
  return stream.str();
}

int requested_sample_count() {
  std::string text = env_or("CUTLASS_DELIVERY_SAMPLE_COUNT", "1");
  int count = std::stoi(text);
  if (count < 1 || count > 9) {
    throw std::runtime_error("CUTLASS_DELIVERY_SAMPLE_COUNT must be 1..9");
  }
  return count;
}

int run(int argc, char const** argv) {
  cudaDeviceProp props{};
  DELIVERY_CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::string gpu_profile = env_or("GEMM_DELIVERY_GPU", "h100");
  if (!matches_gpu_profile(props.name, gpu_profile) || props.major != 9 || props.minor != 0) {
    throw std::runtime_error(
        std::string("requires selected NVIDIA ") + gpu_profile + " sm_90 profile; got " + props.name);
  }

  if (env_or("CUTLASS_DELIVERY_TRACE_ONLY", "0") == "1") {
    return run_selected_trace_example(argc, argv);
  }

  std::size_t marker_rows_per_launch = std::size_t(kMaxCtas) * kMaxWarps;
  std::size_t marker_count = std::size_t(kMaxLaunches) * marker_rows_per_launch * kMarkerFields;
  unsigned long long* device_markers = nullptr;
  unsigned long long* device_grid_ids = nullptr;
  unsigned long long* device_grids = nullptr;
  DELIVERY_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_markers),
                                 marker_count * sizeof(unsigned long long)));
  DELIVERY_CUDA_CHECK(cudaMemset(device_markers, 0,
                                 marker_count * sizeof(unsigned long long)));
  DELIVERY_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_grid_ids),
                                 kMaxLaunches * sizeof(unsigned long long)));
  DELIVERY_CUDA_CHECK(cudaMemset(device_grid_ids, 0,
                                 kMaxLaunches * sizeof(unsigned long long)));
  DELIVERY_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_grids),
                                 kMaxLaunches * sizeof(unsigned long long)));
  DELIVERY_CUDA_CHECK(cudaMemset(device_grids, 0,
                                 kMaxLaunches * sizeof(unsigned long long)));
  DELIVERY_CUDA_CHECK(cudaMemcpyToSymbol(cutlass::cutlass_single_cta_marker_ptr,
                                         &device_markers, sizeof(device_markers)));
  DELIVERY_CUDA_CHECK(cudaMemcpyToSymbol(cutlass::cutlass_single_cta_marker_grid_ids,
                                         &device_grid_ids, sizeof(device_grid_ids)));
  DELIVERY_CUDA_CHECK(cudaMemcpyToSymbol(cutlass::cutlass_single_cta_marker_grids,
                                         &device_grids, sizeof(device_grids)));

  int upstream_return_code = 0;
  bool skip_upstream = false;
  char const* delivery_mode = std::getenv("CUTLASS_DELIVERY_MODE");
#if defined(CUTLASS_DELIVERY_EXAMPLE52) || defined(CUTLASS_DELIVERY_EXAMPLE56)
  skip_upstream = delivery_mode != nullptr && std::strcmp(delivery_mode, "cluster_sm") == 0;
#endif
  if (!skip_upstream) {
    upstream_return_code = cutlass_delivery_upstream_main(argc, argv);
    DELIVERY_CUDA_CHECK(cudaDeviceSynchronize());
  }

#if defined(CUTLASS_DELIVERY_EXAMPLE49) || defined(CUTLASS_DELIVERY_EXAMPLE52) || \
    defined(CUTLASS_DELIVERY_EXAMPLE56) || \
    defined(CUTLASS_DELIVERY_EXAMPLE57) || defined(CUTLASS_DELIVERY_EXAMPLE62) || \
    defined(CUTLASS_DELIVERY_EXAMPLE68)
  DELIVERY_CUDA_CHECK(cudaMemset(device_markers, 0,
                                 marker_count * sizeof(unsigned long long)));
  DELIVERY_CUDA_CHECK(cudaMemset(device_grid_ids, 0,
                                 kMaxLaunches * sizeof(unsigned long long)));
  DELIVERY_CUDA_CHECK(cudaMemset(device_grids, 0,
                                 kMaxLaunches * sizeof(unsigned long long)));
  int post_return_code = run_selected_post_example(argc, argv);
  DELIVERY_CUDA_CHECK(cudaDeviceSynchronize());
#else
  int post_return_code = 0;
#endif

  std::vector<unsigned long long> markers(marker_count);
  std::vector<unsigned long long> grid_ids(kMaxLaunches);
  std::vector<unsigned long long> packed_grids(kMaxLaunches);
  DELIVERY_CUDA_CHECK(cudaMemcpy(markers.data(), device_markers,
                                 marker_count * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost));
  DELIVERY_CUDA_CHECK(cudaMemcpy(grid_ids.data(), device_grid_ids,
                                 kMaxLaunches * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost));
  DELIVERY_CUDA_CHECK(cudaMemcpy(packed_grids.data(), device_grids,
                                 kMaxLaunches * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost));

  struct LaunchSample {
    unsigned long long grid_id;
    double timer;
    double cycles;
    double ratio;
    std::size_t active_sms;
    std::size_t active_ctas;
    std::size_t valid_rows;
    unsigned int cluster_ncta;
    unsigned int cluster_nx;
    unsigned int cluster_ny;
    unsigned int cluster_nz;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
    std::vector<unsigned int> sm_ids;
    std::vector<unsigned int> cluster_ranks;
  };
  std::vector<LaunchSample> launch_samples;
  for (int slot = 0; slot < kMaxLaunches; ++slot) {
    if (grid_ids[slot] == 0) continue;
    unsigned long long first_timer = ~0ull;
    unsigned long long last_timer = 0;
    std::vector<double> ratios;
    std::set<unsigned int> sm_ids;
    std::set<int> cta_ids;
    std::set<unsigned int> cluster_ranks;
    unsigned int cluster_ncta = 0;
    unsigned int sample_nx = 1;
    unsigned int sample_ny = 1;
    unsigned int sample_nz = 1;
    std::size_t valid_rows = 0;
    for (int cta = 0; cta < kMaxCtas; ++cta) {
      for (int warp = 0; warp < kMaxWarps; ++warp) {
        unsigned long long const* row = markers.data() +
            ((std::size_t(slot) * marker_rows_per_launch +
              std::size_t(cta) * kMaxWarps + warp) * kMarkerFields);
        if (row[0] > 0 && row[1] > row[0] && row[2] > 0 && row[3] > row[2]) {
          first_timer = std::min(first_timer, row[2]);
          last_timer = std::max(last_timer, row[3]);
          ratios.push_back(double(row[1] - row[0]) / double(row[3] - row[2]));
          sm_ids.insert(static_cast<unsigned int>(row[4]));
          {
            unsigned long long packed = row[5];
            unsigned int nx = static_cast<unsigned int>(packed & 0xffull);
            unsigned int ny = static_cast<unsigned int>((packed >> 8) & 0xffull);
            unsigned int nz = static_cast<unsigned int>((packed >> 16) & 0xffull);
            unsigned int ncta = static_cast<unsigned int>((packed >> 32) & 0xffull);
            if (ny == 0 && nz == 0) {
              ncta = static_cast<unsigned int>(packed);
              nx = ncta; ny = 1; nz = 1;
            } else if (ncta == 0) {
              ncta = nx * ny * nz;
            }
            cluster_ncta = std::max(cluster_ncta, ncta);
            sample_nx = nx; sample_ny = ny; sample_nz = nz;
          }
          cluster_ranks.insert(static_cast<unsigned int>(row[6]));
          cta_ids.insert(cta);
          ++valid_rows;
        }
      }
    }
    if (valid_rows > 0 && last_timer > first_timer) {
      double ratio = median(ratios);
      double timer = double(last_timer - first_timer);
      unsigned int gx = static_cast<unsigned int>(packed_grids[slot] & 0xffffull);
      unsigned int gy = static_cast<unsigned int>((packed_grids[slot] >> 16) & 0xffffull);
      unsigned int gz = static_cast<unsigned int>((packed_grids[slot] >> 32) & 0xffffull);
      if (gx == 0) gx = 1;
      if (gy == 0) gy = 1;
      if (gz == 0) gz = 1;
      launch_samples.push_back({
          grid_ids[slot], timer, timer * ratio, ratio,
          sm_ids.size(), cta_ids.size(), valid_rows, cluster_ncta,
          sample_nx, sample_ny, sample_nz,
          gx, gy, gz,
          std::vector<unsigned int>(sm_ids.begin(), sm_ids.end()),
          std::vector<unsigned int>(cluster_ranks.begin(), cluster_ranks.end())});
    }
  }
  if (launch_samples.empty()) {
    throw std::runtime_error("upstream example produced no valid CUTLASS device-kernel markers");
  }
  std::sort(launch_samples.begin(), launch_samples.end(),
            [](LaunchSample const& left, LaunchSample const& right) {
              return left.grid_id < right.grid_id;
            });
  int sample_count = std::min<int>(requested_sample_count(), launch_samples.size());
  launch_samples.erase(launch_samples.begin(), launch_samples.end() - sample_count);
  std::vector<double> timer_samples;
  std::vector<double> cycle_samples;
  std::vector<double> ratio_samples;
  std::size_t active_sm_count = ~std::size_t(0);
  std::size_t active_cta_count = ~std::size_t(0);
  std::size_t valid_marker_rows = ~std::size_t(0);
  unsigned int observed_cluster_ncta = 0;
  unsigned int observed_cluster_nx = 1;
  unsigned int observed_cluster_ny = 1;
  unsigned int observed_cluster_nz = 1;
  unsigned int observed_grid_x = 0;
  unsigned int observed_grid_y = 0;
  unsigned int observed_grid_z = 0;
  std::vector<unsigned int> last_sm_ids;
  std::vector<unsigned int> last_cluster_ranks;
  for (auto const& sample : launch_samples) {
    timer_samples.push_back(sample.timer);
    cycle_samples.push_back(sample.cycles);
    ratio_samples.push_back(sample.ratio);
    active_sm_count = std::min(active_sm_count, sample.active_sms);
    active_cta_count = std::min(active_cta_count, sample.active_ctas);
    valid_marker_rows = std::min(valid_marker_rows, sample.valid_rows);
    observed_cluster_ncta = sample.cluster_ncta;
    observed_cluster_nx = sample.cluster_nx;
    observed_cluster_ny = sample.cluster_ny;
    observed_cluster_nz = sample.cluster_nz;
    observed_grid_x = sample.grid_x;
    observed_grid_y = sample.grid_y;
    observed_grid_z = sample.grid_z;
    last_sm_ids = sample.sm_ids;
    last_cluster_ranks = sample.cluster_ranks;
  }

  auto json_u32 = [](std::vector<unsigned int> const& values) {
    std::ostringstream stream;
    stream << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index) stream << ",";
      stream << values[index];
    }
    stream << "]";
    return stream.str();
  };
  std::string requested_sm = env_or("CUTLASS_CLUSTER_SM_COUNT", "0");

  std::cout << "{"
            << "\"schema_version\":\"cutlass-example-sm90-v2\","
            << "\"case_id\":\"" << env_or("CUTLASS_DELIVERY_CASE_ID", "pilot") << "\","
            << "\"algorithm\":\"" << env_or("CUTLASS_DELIVERY_ALGORITHM", "unknown") << "\","
            << "\"upstream_source\":\"" << CUTLASS_DELIVERY_EXAMPLE << "\","
            << "\"delivery_mode\":\"" << env_or("CUTLASS_DELIVERY_MODE", "full_gpu") << "\","
            << "\"gpu_profile\":\"" << gpu_profile << "\","
            << "\"gpu_name\":\"" << props.name << "\","
            << "\"device_sm_count\":" << props.multiProcessorCount << ","
            << "\"requested_sm_count\":" << std::stoi(requested_sm.empty() ? "0" : requested_sm) << ","
            << "\"compiled_cluster\":\"" << env_or("CUTLASS_CLUSTER_SHAPE", "") << "\","
            << "\"grid\":[" << observed_grid_x << "," << observed_grid_y << "," << observed_grid_z << "],"
            << "\"observed_cluster_ncta\":" << observed_cluster_ncta << ","
            << "\"cluster_dim\":[" << observed_cluster_nx << "," << observed_cluster_ny << "," << observed_cluster_nz << "],"
            << "\"cluster_ranks\":" << json_u32(last_cluster_ranks) << ","
            << "\"sm_ids\":" << json_u32(last_sm_ids) << ","
            << "\"upstream_return_code\":" << upstream_return_code << ","
            << "\"post_return_code\":" << post_return_code << ","
            << "\"valid_marker_rows\":" << valid_marker_rows << ","
            << "\"active_sm_count\":" << active_sm_count << ","
            << "\"active_cta_count\":" << active_cta_count << ","
            << "\"globaltimer_ns\":" << json_array(timer_samples) << ","
            << "\"clock_per_globaltimer\":" << std::fixed << std::setprecision(6)
            << median(ratio_samples) << ","
            << "\"whole_gpu_cycles\":" << json_array(cycle_samples)
            << "}" << std::endl;

  DELIVERY_CUDA_CHECK(cudaFree(device_markers));
  DELIVERY_CUDA_CHECK(cudaFree(device_grid_ids));
  DELIVERY_CUDA_CHECK(cudaFree(device_grids));
  return upstream_return_code != 0 ? upstream_return_code : post_return_code;
}

}  // namespace cutlass_delivery

int main(int argc, char const** argv) {
  try {
    return cutlass_delivery::run(argc, argv);
  } catch (std::exception const& error) {
    std::cerr << "DELIVERY_ERROR: " << error.what() << std::endl;
    return 2;
  }
}
