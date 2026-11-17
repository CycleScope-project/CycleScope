#include "../tensor_pipe_forms/umma_impl.cuh"

#include <string>
#include <vector>

template <int CG>
__device__ __forceinline__ void admission_copy(std::uint32_t t, std::uint64_t desc) {
  if constexpr (CG == 1)
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(t), "l"(desc) : "memory");
  else
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                 :: "r"(t), "l"(desc) : "memory");
}

template <int U, int M, int CG, int AS = 0>
__global__ void __launch_bounds__(128, 1) k_admission(
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
  if constexpr (AS == 1) mb_tmem_fill(row_base, 32, 0x22222222u);
  mb_tmem_fill(row_base + 256, 128, 0);
  mb_tmem_fill(row_base + 384, 128, M ? 0x38383838u : 0u);
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();
  const int producer = warp / 2;
  const std::uint32_t d = tbase + 128;
  const std::uint32_t sf = tbase + 256 + producer * 128;
  const std::uint32_t mbar = mb_opaque(mb_smem_u32(&bars[producer * 2]));
  const std::uint32_t ack = mb_opaque(mb_smem_u32(&bars[producer * 2 + 1]));
  const std::uint64_t ad = make_sdesc(mb_smem_u32(src), 128, 0);
  const std::uint64_t bd = make_sdesc(mb_smem_u32(src + 8192), 128, 0);
  const std::uint64_t cd = make_sdesc(mb_smem_u32(src + 16384), 32, 0);
  const std::uint32_t idesc = make_idesc<kMXF4NVF4>(128 * CG, 128);
  const bool elected = mb_elect_one() != 0;
  std::uint32_t phase = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (rank == 0 && elected) {
      if (warp % 2 == 0) {
        xw_delay(rep, kXwDelay, seed);
        const std::uint64_t start = mb_clock();
        if (producer == 0) {
#pragma unroll
          for (int i = 0; i < U; ++i) admission_copy<CG>(sf + i * 4, cd);
        } else {
#pragma unroll
          for (int i = 0; i < M; ++i)
            umma<kMXF4NVF4, CG, AS>(d, ad, tbase, bd, idesc, sf, sf + 4, i != 0);
        }
        const std::uint64_t admitted = mb_clock();
        umma_commit<CG>(mbar);
        mb_mbar_wait_to(ack, phase);
        asm volatile("membar.cta;" ::: "memory");
        out[3 * (producer * reps + rep)] = start;
        out[3 * (producer * reps + rep) + 1] = admitted;
        phase ^= 1;
      } else {
        mb_mbar_spin_to(mbar, phase);
        out[3 * (producer * reps + rep) + 2] = mb_clock();
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
  if (warp % 2 == 0) {
    const std::uint32_t address = (producer == 1 && M ? d : sf) +
        (static_cast<std::uint32_t>(warp * 32) << 16);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      std::uint32_t value;
      const std::uint32_t last_column = producer == 0 && U ? U * 4 - 1 : 127;
      const std::uint32_t at = address + i * last_column;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                   : "=r"(value) : "r"(at) : "memory");
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      if (lane == 0) values[rank * 4 + producer * 2 + i] = value;
    }
    if (rank == 0 && lane == 0) {
      std::uint32_t warp_slot;
      asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_slot));
      values[8 + producer] = warp_slot;
    }
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  if (warp == 0) mb_tmem_dealloc<CG>(tbase, 512);
}

using AdmissionKernel = void (*)(std::uint64_t*, std::uint32_t*, int, std::uint32_t);
struct AdmissionCell { int copies, mma, cta_group, a_source; AdmissionKernel kernel; };

template <int CG>
void admission_cells(std::vector<AdmissionCell>& cells) {
  cells.push_back({0, 0, CG, 0, &k_admission<0, 0, CG>});
  cells.push_back({8, 0, CG, 0, &k_admission<8, 0, CG>});
  cells.push_back({32, 0, CG, 0, &k_admission<32, 0, CG>});
  cells.push_back({0, 4, CG, 0, &k_admission<0, 4, CG>});
  cells.push_back({32, 4, CG, 0, &k_admission<32, 4, CG>});
  cells.push_back({0, 4, CG, 1, &k_admission<0, 4, CG, 1>});
  cells.push_back({32, 0, CG, 1, &k_admission<32, 0, CG, 1>});
  cells.push_back({32, 4, CG, 1, &k_admission<32, 4, CG, 1>});
}

template <class T>
std::string admission_array(const std::vector<T>& values) {
  std::string s = "[";
  for (std::size_t i = 0; i < values.size(); ++i)
    s += (i ? "," : "") + std::to_string(values[i]);
  return s + "]";
}

int main(int argc, char** argv) {
  std::uint32_t seed_base = 0;
  int forwarded = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) != "--seed-base") {
      argv[forwarded++] = argv[i];
      continue;
    }
    if (++i == argc) {
      std::fprintf(stderr, "missing --seed-base value\n");
      return 2;
    }
    char* end = nullptr;
    const auto value = std::strtoul(argv[i], &end, 10);
    if (end == argv[i] || *end || argv[i][0] == '-' || value > UINT32_MAX) {
      std::fprintf(stderr, "invalid --seed-base value\n");
      return 2;
    }
    seed_base = static_cast<std::uint32_t>(value);
  }
  argc = forwarded;
  argv[argc] = nullptr;
  const MbArgs args = mb_parse(argc, argv);
  if (seed_base > UINT32_MAX - static_cast<std::uint32_t>(args.launches)) {
    std::fprintf(stderr, "--seed-base range overflows\n");
    return 2;
  }
  MbWriter writer(args, "tmem_admission");
  std::uint64_t* output = nullptr;
  std::uint32_t* values = nullptr;
  MB_CHECK(cudaMalloc(&output, sizeof(std::uint64_t) * 6 * args.reps));
  MB_CHECK(cudaMalloc(&values, sizeof(std::uint32_t) * 10));
  std::vector<AdmissionCell> cells;
  admission_cells<1>(cells);
  admission_cells<2>(cells);
  for (const auto& cell : cells) {
    const std::string name = "cp" + std::to_string(cell.copies) +
        ".mma" + std::to_string(cell.mma) + ".cg" + std::to_string(cell.cta_group) +
        (cell.a_source ? ".tmem_a" : "");
    if (!mb_selected(args, name)) continue;
    std::vector<double> complete[3], admission[2];
    std::vector<std::uint64_t> start_skews;
    std::vector<std::uint32_t> warp_slots[2];
    bool bad = false;
    for (int launch = -1; launch < args.launches; ++launch) {
      int reps = launch == -1 ? 1 : args.reps;
      std::uint32_t seed = seed_base + static_cast<std::uint32_t>(launch + 1);
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
      std::uint32_t result[10] = {};
      MB_CHECK(cudaMemcpy(result, values, sizeof(result), cudaMemcpyDeviceToHost));
      bad |= mb_take_err();
      for (int rank = 0; rank < cell.cta_group; ++rank) {
        for (int p = 0; p < 2; ++p) {
          for (int i = 0; i < 2; ++i) {
            const std::uint32_t value = result[rank * 4 + p * 2 + i];
            float actual = 0;
            std::memcpy(&actual, &value, sizeof(actual));
            bad |= p == 1 && cell.mma ? actual != 64.0f * cell.mma :
                value != (p == 0 && cell.copies ? 0x38383838u : 0u);
          }
        }
      }
      if (bad) break;
      if (launch == -1) continue;
      for (int p = 0; p < 2; ++p) warp_slots[p].push_back(result[8 + p]);
      std::vector<std::uint64_t> host(6 * reps);
      MB_CHECK(cudaMemcpy(host.data(), output, host.size() * 8, cudaMemcpyDeviceToHost));
      for (int rep = 1; rep < reps; ++rep) {
        std::uint64_t first = UINT64_MAX, last = 0, latest_start = 0;
        for (int p = 0; p < 2; ++p) {
          const auto begin = host[3 * (p * reps + rep)];
          const auto issued = host[3 * (p * reps + rep) + 1];
          const auto end = host[3 * (p * reps + rep) + 2];
          if (issued < begin || end < issued) { bad = true; break; }
          admission[p].push_back(static_cast<double>(issued - begin));
          complete[p].push_back(static_cast<double>(end - begin));
          first = std::min(first, begin);
          latest_start = std::max(latest_start, begin);
          last = std::max(last, end);
        }
        if (bad) break;
        complete[2].push_back(static_cast<double>(last - first));
        start_skews.push_back(latest_start - first);
      }
      if (bad) break;
    }
    const std::string params = "\"copies\":" + std::to_string(cell.copies) +
        ",\"mma\":" + std::to_string(cell.mma) + ",\"cta_group\":" +
        std::to_string(cell.cta_group) + ",\"producers\":2,\"split\":1,\"shape\":4,\"seed_base\":" +
        std::to_string(seed_base) + ",\"a_source\":" + std::to_string(cell.a_source);
    if (bad) {
      writer.fail(name, cell.copies, params, "bad value, timestamp or timeout");
      continue;
    }
    for (int p = 0; p < 3; ++p) {
      const std::string extra = p < 2 ?
          "\"admission_samples\":" + admission_array(admission[p]) +
          ",\"warp_slot_samples\":" + admission_array(warp_slots[p]) :
          "\"start_skew_samples\":" + admission_array(start_skews);
      writer.row(name + (p == 2 ? ".joint" : ".w" + std::to_string(p)),
                 cell.copies, params, complete[p], extra);
    }
  }
  MB_CHECK(cudaFree(values));
  MB_CHECK(cudaFree(output));
  return writer.failures ? 1 : 0;
}
