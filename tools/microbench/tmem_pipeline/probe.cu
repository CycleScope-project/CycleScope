#include "../tensor_pipe_forms/umma_impl.cuh"

#include <cmath>
#include <string>
#include <vector>

template <int CG, int SH>
__device__ __forceinline__ void copy_scale(std::uint32_t t, std::uint64_t desc) {
#define COPY(CGS, FORM) asm volatile("tcgen05.cp.cta_group::" CGS "." FORM " [%0], %1;" :: "r"(t), "l"(desc) : "memory")
  if constexpr (CG == 1) {
    if constexpr (SH == 0) COPY("1", "128x256b");
    if constexpr (SH == 1) COPY("1", "4x256b");
    if constexpr (SH == 2) COPY("1", "128x128b");
    if constexpr (SH == 3) COPY("1", "64x128b.warpx2::02_13");
    if constexpr (SH == 4) COPY("1", "32x128b.warpx4");
  } else {
    if constexpr (SH == 0) COPY("2", "128x256b");
    if constexpr (SH == 1) COPY("2", "4x256b");
    if constexpr (SH == 2) COPY("2", "128x128b");
    if constexpr (SH == 3) COPY("2", "64x128b.warpx2::02_13");
    if constexpr (SH == 4) COPY("2", "32x128b.warpx4");
  }
#undef COPY
}

template <int U, int M, int ROT, int PRODUCERS, int SPLIT = 0, int CG = 1, int SH = 4>
__global__ void __launch_bounds__(128, 1) k_pipeline(
    std::uint64_t* out, std::uint32_t* values, int reps, std::uint32_t seed) {
  __shared__ __align__(1024) unsigned char src[24576];
  __shared__ __align__(8) std::uint64_t bars[4];
  __shared__ std::uint32_t slot;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const std::uint32_t rank = CG == 2 ? mb_ctarank() : 0;
  auto* words = reinterpret_cast<std::uint32_t*>(src);
  for (int i = threadIdx.x; i < 6144; i += blockDim.x)
    words[i] = i >= 4096 ? 0x38383838u : 0x22222222u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; ++i) mb_mbar_init(mb_smem_u32(&bars[i]), 1);
    mb_fence_mbar_init();
  }
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  if (warp == 0) {
    mb_tmem_alloc<CG>(mb_smem_u32(&slot), 512);
    mb_tmem_relinquish<CG>();
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();
  const std::uint32_t tbase = slot;
  const std::uint32_t row_base = tbase + (static_cast<std::uint32_t>(warp * 32) << 16);
  mb_tmem_fill(row_base + 256, 256, 0x38383838u);
  for (int p = 0; p < PRODUCERS; ++p) {
    if (SPLIT ? p == 0 || M == 0 : U > 0 || M == 0) {
      const std::uint32_t t = row_base + 256 + p * 128;
      const std::uint32_t z = 0;
      asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%1,%1,%1};"
                   :: "r"(t), "r"(z) : "memory");
    }
  }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();
  const int producer = warp / 2;
  const std::uint32_t d = tbase + producer * 128;
  const std::uint32_t sf = tbase + 256 + producer * 128;
  const std::uint32_t mbar = mb_opaque(mb_smem_u32(&bars[producer * 2]));
  const std::uint32_t ack = mb_opaque(mb_smem_u32(&bars[producer * 2 + 1]));
  const std::uint64_t ad = make_sdesc(mb_smem_u32(src), 128, 0);
  const std::uint64_t bd = make_sdesc(mb_smem_u32(src + 8192), 128, 0);
  constexpr int copy_rows[] = {128, 4, 128, 64, 32};
  const std::uint64_t cd = make_sdesc(mb_smem_u32(src + 16384), copy_rows[SH], 0);
  const std::uint32_t idesc = make_idesc<kMXF4NVF4>(128 * CG, 128);
  const bool elected = mb_elect_one() != 0;
  std::uint32_t phase = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (rank == 0 && warp < 2 * PRODUCERS && elected) {
      if (warp % 2 == 0) {
        xw_delay(rep, kXwDelay, seed);
        const std::uint64_t start = mb_clock();
        if (!SPLIT || producer == 0) {
#pragma unroll
          for (int i = 0; i < U; ++i)
            copy_scale<CG, SH>(sf + (ROT ? i * (SH == 4 ? 4 : 8) : 0), cd);
        }
        if (!SPLIT || producer == 1) {
#pragma unroll
          for (int i = 0; i < M; ++i)
            umma<kMXF4NVF4, CG, 0>(d, ad, 0, bd, idesc, sf, sf + 4, i != 0);
        }
        umma_commit<CG>(mbar);
        mb_mbar_wait_to(ack, phase);
        asm volatile("membar.cta;" ::: "memory");
        out[2 * (producer * reps + rep)] = start;
        phase ^= 1;
      } else {
        mb_mbar_spin_to(mbar, phase);
        out[2 * (producer * reps + rep) + 1] = mb_clock();
        phase ^= 1;
        mb_mbar_arrive(ack);
      }
    }
  }
  __syncwarp();
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();
  if (warp % 2 == 0 && warp < 2 * PRODUCERS) {
    std::uint32_t value;
    const std::uint32_t address = (M && (!SPLIT || producer == 1) ? d : sf) +
        (static_cast<std::uint32_t>(warp * 32) << 16);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=r"(value) : "r"(address) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (lane == 0 && rank == 0) values[producer] = value;
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  if (warp == 0) mb_tmem_dealloc<CG>(tbase, 512);
}

using Kernel = void (*)(std::uint64_t*, std::uint32_t*, int, std::uint32_t);
struct Cell { int copies, mma, rotate, producers, split, cta_group, shape; Kernel kernel; };

template <int CG, int M, int ROT, int P, int... Us>
void add(std::vector<Cell>& cells, std::integer_sequence<int, Us...>) {
  (cells.push_back({Us, M, ROT, P, 0, CG, 4, &k_pipeline<Us, M, ROT, P, 0, CG>}), ...);
}

template <int CG, int M, int... Us>
void add_split(std::vector<Cell>& cells, std::integer_sequence<int, Us...>) {
  (cells.push_back({Us, M, 1, 2, 1, CG, 4, &k_pipeline<Us, M, 1, 2, 1, CG>}), ...);
}

template <int CG, int SH, int... Us>
void add_shape(std::vector<Cell>& cells, std::integer_sequence<int, Us...>) {
  (cells.push_back({Us, 0, 1, 1, 0, CG, SH, &k_pipeline<Us, 0, 1, 1, 0, CG, SH>}), ...);
}

template <int CG, int P>
void add_producers(std::vector<Cell>& cells) {
  using Counts = std::integer_sequence<int, 0, 1, 2, 4, 8, 16, 32>;
  add<CG, 0, 1, P>(cells, Counts{});
  add<CG, 1, 1, P>(cells, Counts{});
  add<CG, 4, 1, P>(cells, Counts{});
  add<CG, 0, 0, P>(cells, std::integer_sequence<int, 1, 2, 4, 8, 16, 32>{});
}

template <int CG>
void add_group(std::vector<Cell>& cells) {
  add_producers<CG, 1>(cells);
  add_producers<CG, 2>(cells);
  using SplitCounts = std::integer_sequence<int, 0, 8, 16, 32>;
  add_split<CG, 0>(cells, SplitCounts{});
  add_split<CG, 1>(cells, SplitCounts{});
  add_split<CG, 4>(cells, SplitCounts{});
  using ShapeCounts = std::integer_sequence<int, 0, 1, 2, 4, 8, 16>;
  add_shape<CG, 0>(cells, ShapeCounts{});
  add_shape<CG, 1>(cells, ShapeCounts{});
  add_shape<CG, 2>(cells, ShapeCounts{});
  add_shape<CG, 3>(cells, ShapeCounts{});
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter writer(args, "tmem_pipeline");
  std::uint64_t* output = nullptr;
  std::uint32_t* values = nullptr;
  MB_CHECK(cudaMalloc(&output, sizeof(std::uint64_t) * 4 * args.reps));
  MB_CHECK(cudaMalloc(&values, sizeof(std::uint32_t) * 2));
  std::vector<Cell> cells;
  add_group<1>(cells);
  add_group<2>(cells);
  for (const auto& cell : cells) {
    const std::string name = "cp" + std::to_string(cell.copies) +
        ".mma" + std::to_string(cell.mma) + ".rot" + std::to_string(cell.rotate) +
        ".p" + std::to_string(cell.producers) + (cell.split ? ".split" : "") +
        (cell.cta_group == 2 ? ".cg2" : "") +
        (cell.shape == 4 ? "" : ".shape" + std::to_string(cell.shape));
    if (!mb_selected(args, name)) continue;
    std::vector<std::vector<double>> samples(cell.producers + 1);
    std::vector<std::uint64_t> start_skews;
    bool bad = false;
    for (int launch = -1; launch < args.launches; ++launch) {
      int reps = launch == -1 ? 1 : args.reps;
      std::uint32_t seed = static_cast<std::uint32_t>(launch + 1);
      void* arguments[] = {&output, &values, &reps, &seed};
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(cell.cta_group);
      config.blockDim = dim3(128);
      cudaLaunchAttribute attribute{};
      attribute.id = cudaLaunchAttributeClusterDimension;
      attribute.val.clusterDim.x = cell.cta_group;
      attribute.val.clusterDim.y = attribute.val.clusterDim.z = 1;
      config.attrs = &attribute;
      config.numAttrs = 1;
      MB_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(cell.kernel), arguments));
      MB_CHECK(cudaDeviceSynchronize());
      std::uint32_t result[2] = {};
      MB_CHECK(cudaMemcpy(result, values, sizeof(result), cudaMemcpyDeviceToHost));
      bad |= mb_take_err();
      for (int p = 0; p < cell.producers; ++p) {
        float actual = 0;
        std::memcpy(&actual, &result[p], sizeof(actual));
        const int mma = cell.split && p == 0 ? 0 : cell.mma;
        const int copies = cell.split && p == 1 ? 0 : cell.copies;
        bad |= mma ? actual != 64.0f * mma :
            result[p] != (copies ? 0x38383838u : 0u);
      }
      if (bad) break;
      if (launch == -1) continue;
      std::vector<std::uint64_t> host(2 * cell.producers * reps);
      MB_CHECK(cudaMemcpy(host.data(), output, host.size() * 8, cudaMemcpyDeviceToHost));
      for (int rep = 1; rep < reps; ++rep) {
        std::uint64_t first = UINT64_MAX, last = 0, latest_start = 0;
        for (int p = 0; p < cell.producers; ++p) {
          const auto begin = host[2 * (p * reps + rep)];
          const auto end = host[2 * (p * reps + rep) + 1];
          if (end < begin) { bad = true; break; }
          samples[p].push_back(static_cast<double>(end - begin));
          first = std::min(first, begin);
          latest_start = std::max(latest_start, begin);
          last = std::max(last, end);
        }
        if (bad) break;
        samples.back().push_back(static_cast<double>(last - first));
        start_skews.push_back(latest_start - first);
      }
      if (bad) break;
    }
    const std::string params = "\"copies\":" + std::to_string(cell.copies) +
        ",\"mma\":" + std::to_string(cell.mma) + ",\"rotate\":" +
        std::to_string(cell.rotate) + ",\"producers\":" + std::to_string(cell.producers) +
        ",\"split\":" + std::to_string(cell.split);
    const std::string all_params = params + ",\"cta_group\":" +
        std::to_string(cell.cta_group) + ",\"shape\":" + std::to_string(cell.shape);
    if (bad) {
      writer.fail(name, cell.copies, all_params, "bad value, timestamp or timeout");
      continue;
    }
    std::string extra = "\"start_skew_samples\":[";
    for (std::size_t i = 0; i < start_skews.size(); ++i)
      extra += (i ? "," : "") + std::to_string(start_skews[i]);
    extra += "]";
    for (int p = 0; p <= cell.producers; ++p)
      writer.row(name + (p == cell.producers ? ".joint" : ".w" + std::to_string(p)),
                 cell.copies, all_params, samples[p], p == cell.producers ? extra : "");
  }
  MB_CHECK(cudaFree(values));
  MB_CHECK(cudaFree(output));
  return writer.failures ? 1 : 0;
}
