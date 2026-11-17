#include "../tensor_pipe_forms/umma_impl.cuh"

#include <string>
#include <vector>

template <int CG>
__device__ __forceinline__ void interleave_copy(std::uint32_t t, std::uint64_t desc) {
  if constexpr (CG == 1)
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(t), "l"(desc) : "memory");
  else
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                 :: "r"(t), "l"(desc) : "memory");
}

constexpr std::uint32_t kIlColInd = 400;

template <int C, int R, int CG, int AS, int N, int IND>
__global__ void __launch_bounds__(128, 1) k_interleave(
    std::uint64_t* out, std::uint32_t* values, int reps, std::uint32_t seed) {
  __shared__ __align__(1024) unsigned char src[24576];
  __shared__ __align__(8) std::uint64_t bars[2];
  __shared__ std::uint32_t slot;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const std::uint32_t rank = CG == 2 ? mb_ctarank() : 0;
  auto* words = reinterpret_cast<std::uint32_t*>(src);
  for (int i = threadIdx.x; i < 6144; i += blockDim.x)
    words[i] = i >= 4096 ? 0x38383838u : 0x22222222u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; ++i) mb_mbar_init(mb_smem_u32(&bars[i]), 1);
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
  if constexpr (AS == 1) mb_tmem_fill(row_base + kColA, 32, 0x22222222u);
  mb_tmem_fill(row_base + kColSfa, 64, (C == 0 || IND) ? 0x38383838u : 0u);
  mb_tmem_fill(row_base + 384, 64, 0u);
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  mb_tc_fence_after();
  const std::uint32_t d = tbase;
  const std::uint32_t sfa = tbase + kColSfa, sfb = tbase + kColSfb;
  const std::uint32_t dst0 = IND ? tbase + kIlColInd : sfa;
  const std::uint32_t dst1 = IND ? tbase + kIlColInd + 4 : sfb;
  const std::uint32_t dst2 = dst1 + 4;
  const std::uint32_t mbar = mb_opaque(mb_smem_u32(&bars[0]));
  const std::uint32_t ack = mb_opaque(mb_smem_u32(&bars[1]));
  const std::uint64_t ad = make_sdesc(mb_smem_u32(src), 128, 0);
  const std::uint64_t bd = make_sdesc(mb_smem_u32(src + 8192), CG == 2 ? N / 2 : N, 0);
  const std::uint64_t cd = make_sdesc(mb_smem_u32(src + 16384), 32, 0);
  const std::uint32_t idesc = make_idesc<kMXF4NVF4>(128 * CG, N);
  const bool elected = mb_elect_one() != 0;
  constexpr int kNeed = N == 256 ? 3 : 2;
  std::uint32_t phase = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (rank == 0 && elected) {
      if (warp == 0) {
        xw_delay(rep, kXwDelay, seed);
        const std::uint64_t start = mb_clock();
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
          for (int j = 0; j < C; ++j) {
            const int which = j % kNeed;
            interleave_copy<CG>(which == 0 ? dst0 : which == 1 ? dst1 : dst2, cd);
          }
          umma<kMXF4NVF4, CG, AS>(d, ad, tbase + kColA, bd, idesc, sfa, sfb, i != 0);
        }
        const std::uint64_t admitted = mb_clock();
        umma_commit<CG>(mbar);
        mb_mbar_wait_to(ack, phase);
        asm volatile("membar.cta;" ::: "memory");
        out[3 * rep] = start;
        out[3 * rep + 1] = admitted;
        phase ^= 1;
      } else if (warp == 1) {
        mb_mbar_spin_to(mbar, phase);
        out[3 * rep + 2] = mb_clock();
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
  {
    const std::uint32_t address = d + (static_cast<std::uint32_t>(warp * 32) << 16);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      std::uint32_t value;
      const std::uint32_t at = address + (i == 0 ? 0u : i == 1 ? static_cast<std::uint32_t>(N - 1) :
                                          (IND ? kIlColInd : kColSfa));
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                   : "=r"(value) : "r"(at) : "memory");
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      if (lane == 0) values[rank * 12 + warp * 3 + i] = value;
    }
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (CG == 2) mb_cluster_sync();
  if (warp == 0) mb_tmem_dealloc<CG>(tbase, 512);
}

using InterleaveKernel = void (*)(std::uint64_t*, std::uint32_t*, int, std::uint32_t);
struct InterleaveCell { int copies, mma, cta_group, a_source, n, independent; InterleaveKernel kernel; };

template <int CG, int AS, int N>
void interleave_cells(std::vector<InterleaveCell>& cells) {
  constexpr int kNeed = N == 256 ? 3 : 2;
  cells.push_back({0, 8, CG, AS, N, 0, &k_interleave<0, 8, CG, AS, N, 0>});
  cells.push_back({kNeed, 8, CG, AS, N, 0, &k_interleave<kNeed, 8, CG, AS, N, 0>});
  cells.push_back({8, 8, CG, AS, N, 0, &k_interleave<8, 8, CG, AS, N, 0>});
}

template <int CG, int AS, int N>
void interleave_independent_cells(std::vector<InterleaveCell>& cells) {
  constexpr int kNeed = N == 256 ? 3 : 2;
  cells.push_back({kNeed, 8, CG, AS, N, 1, &k_interleave<kNeed, 8, CG, AS, N, 1>});
  cells.push_back({8, 8, CG, AS, N, 1, &k_interleave<8, 8, CG, AS, N, 1>});
}

std::string interleave_array(const std::vector<double>& values) {
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
  MbWriter writer(args, "tmem_interleave");
  std::uint64_t* output = nullptr;
  std::uint32_t* values = nullptr;
  MB_CHECK(cudaMalloc(&output, sizeof(std::uint64_t) * 3 * args.reps));
  MB_CHECK(cudaMalloc(&values, sizeof(std::uint32_t) * 24));
  std::vector<InterleaveCell> cells;
  interleave_cells<1, 0, 128>(cells);
  interleave_cells<1, 1, 128>(cells);
  interleave_cells<1, 0, 256>(cells);
  interleave_cells<2, 0, 256>(cells);
  interleave_cells<2, 0, 128>(cells);
  interleave_independent_cells<1, 1, 128>(cells);
  interleave_independent_cells<2, 0, 256>(cells);
  for (const auto& cell : cells) {
    const std::string name = "c" + std::to_string(cell.copies) + ".r" + std::to_string(cell.mma) +
        ".cg" + std::to_string(cell.cta_group) + ".as" + std::to_string(cell.a_source) +
        ".n" + std::to_string(cell.n) + (cell.independent ? ".ind" : "");
    if (!mb_selected(args, name)) continue;
    std::vector<double> complete, admission;
    bool bad = false;
    const std::uint32_t marker = cell.copies || !cell.independent ? 0x38383838u : 0u;
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
      std::uint32_t result[24] = {};
      MB_CHECK(cudaMemcpy(result, values, sizeof(result), cudaMemcpyDeviceToHost));
      bad |= mb_take_err();
      for (int rank = 0; rank < cell.cta_group; ++rank) {
        for (int w = 0; w < 4; ++w) {
          for (int i = 0; i < 2; ++i) {
            const std::uint32_t value = result[rank * 12 + w * 3 + i];
            float actual = 0;
            std::memcpy(&actual, &value, sizeof(actual));
            bad |= actual != 64.0f * cell.mma;
          }
          bad |= result[rank * 12 + w * 3 + 2] != marker;
        }
      }
      if (bad) break;
      if (launch == -1) continue;
      std::vector<std::uint64_t> host(3 * reps);
      MB_CHECK(cudaMemcpy(host.data(), output, host.size() * 8, cudaMemcpyDeviceToHost));
      for (int rep = 1; rep < reps; ++rep) {
        const auto begin = host[3 * rep], issued = host[3 * rep + 1], end = host[3 * rep + 2];
        if (issued < begin || end < issued) { bad = true; break; }
        admission.push_back(static_cast<double>(issued - begin));
        complete.push_back(static_cast<double>(end - begin));
      }
      if (bad) break;
    }
    const std::string params = "\"copies\":" + std::to_string(cell.copies) +
        ",\"mma\":" + std::to_string(cell.mma) + ",\"cta_group\":" +
        std::to_string(cell.cta_group) + ",\"a_source\":" + std::to_string(cell.a_source) +
        ",\"n\":" + std::to_string(cell.n) + ",\"independent\":" + std::to_string(cell.independent) +
        ",\"shape\":4,\"seed_base\":" + std::to_string(seed_base);
    if (bad) {
      writer.fail(name, cell.copies, params, "bad value, timestamp or timeout");
      continue;
    }
    writer.row(name + ".w0", cell.copies, params, complete,
               "\"admission_samples\":" + interleave_array(admission));
  }
  MB_CHECK(cudaFree(values));
  MB_CHECK(cudaFree(output));
  return writer.failures ? 1 : 0;
}
