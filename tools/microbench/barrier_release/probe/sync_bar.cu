// CTA and cluster barrier release: synchronization.cta.release (timing kind
// 17) and synchronization.cluster.release (kind 42).
//
// Engine: a barrier releases every participant at last arrival + kind
// 17/42 (engine.cpp cta/cluster barrier states).  The last arriver is made
// explicit: it waits a jittered delay, reads t_s and arrives; everyone else
// is already blocked.  All clocks compared are on one SM.
//   bar.cta.wN    CTA of N warps (N = 2..32); warp 0 arrives last at
//                 bar.sync 0; t_w read by warp 0 (the arriver), warp 1 and
//                 warp N-1 right after the barrier
//   bar.cta.wN.all.wK  same, every warp K records its wake (K = 0..N-1)
//   bar.cta.stg.wN     as bar.cta.wN, the arriver has one st.global pending:
//                      does a CTA barrier wait for the store's L2 ack?
//   bar.cluster.cS cluster of S CTAs (1 warp each); rank 0 arrives last:
//                 barrier.cluster.arrive{.release|.relaxed}.aligned ;
//                 barrier.cluster.wait.aligned ; t_w on rank 0
//   kind = E[t_w - t_s] - o_s(t_s -> arrive) - o_w(wait -> t_w)  (sass_check)
#include "../mb.cuh"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

constexpr std::uint32_t kDelay = 3000;

__device__ __forceinline__ void jitter_delay(int rep, std::uint32_t seed) {
  const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
  const std::uint32_t d = kDelay + ((h ^ (h >> 15)) & 2047u);
  const std::uint64_t t = mb_clock();
  while (mb_clock() - t < d) {
  }
}

template <int NW, int STG = 0>
__global__ void __launch_bounds__(NW * 32, 1) k_bar(std::uint64_t* out, int reps, std::uint32_t seed) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint64_t t_s = 0;
    if (warp == 0) {
      jitter_delay(rep, seed);
      t_s = mb_clock();
      // STG = 1: the last arriver has one st.global pending at the barrier
      if constexpr (STG) asm volatile("st.global.u32 [%0], %1;" ::"l"(out + 4 * reps + 2 + lane), "r"(lane) : "memory");
    }
    __syncthreads();
    const std::uint64_t t_w = mb_clock();
    if (lane == 0) {
      if (warp == 0) {
        out[4 * rep + 0] = t_s;
        out[4 * rep + 1] = t_w;
      }
      if (warp == 1) out[4 * rep + 2] = t_w;
      if (warp == NW - 1) out[4 * rep + 3] = t_w;
    }
  }
}

// Every warp's wake: same CTA barrier, each warp's t_w kept in shared
// memory (bar.cta.wN.all.wK rows): the release order / stagger across warps.
constexpr int kAllReps = 64;

template <int NW>
__global__ void __launch_bounds__(NW * 32, 1) k_barall(std::uint64_t* out, int reps, std::uint32_t seed) {
  __shared__ std::uint64_t ts[kAllReps][NW + 1];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint64_t t_s = 0;
    if (warp == 0) {
      jitter_delay(rep, seed);
      t_s = mb_clock();
    }
    __syncthreads();
    const std::uint64_t t_w = mb_clock();
    if (lane == 0) {
      ts[rep][1 + warp] = t_w;
      if (warp == 0) ts[rep][0] = t_s;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < reps * (NW + 1); i += blockDim.x) out[i] = ts[i / (NW + 1)][i % (NW + 1)];
}

// Timestamps go to shared memory inside the loop and to global memory after
// it: a pending global store would make the .release arrive wait for it.
constexpr int kMaxReps = 256;

template <int CS, int RELAXED>
__global__ void __launch_bounds__(32, 1) k_cbar(std::uint64_t* out, int reps, std::uint32_t seed) {
  __shared__ std::uint64_t ts[kMaxReps][2];
  const std::uint32_t rank = mb_ctarank();
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint64_t t_s = 0;
    if (rank == 0) {
      jitter_delay(rep, seed);
      t_s = mb_clock();
    }
    if constexpr (RELAXED) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    else asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    const std::uint64_t t_w = mb_clock();
    if (rank == 0 && threadIdx.x == 0) {
      ts[rep][0] = t_s;
      ts[rep][1] = t_w;
    }
  }
  if (rank == 0 && threadIdx.x == 0) {
    for (int rep = 0; rep < reps; ++rep) {
      out[4 * rep + 0] = ts[rep][0];
      out[4 * rep + 1] = ts[rep][1];
    }
  }
}

using BKern = void (*)(std::uint64_t*, int, std::uint32_t);
struct BCase {
  std::string variant;
  int threads, cluster;
  BKern k;
};

template <int... Ns>
void add_cta(std::vector<BCase>& v, std::integer_sequence<int, Ns...>) {
  (v.push_back({"bar.cta.w" + std::to_string(Ns), Ns * 32, 1, &k_bar<Ns>}), ...);
  (v.push_back({"bar.cta.stg.w" + std::to_string(Ns), Ns * 32, 1, &k_bar<Ns, 1>}), ...);
}

template <int... Ss>
void add_cluster(std::vector<BCase>& v, std::integer_sequence<int, Ss...>) {
  (v.push_back({"bar.cluster.c" + std::to_string(Ss) + ".release", 32, Ss, &k_cbar<Ss, 0>}), ...);
  (v.push_back({"bar.cluster.c" + std::to_string(Ss) + ".relaxed", 32, Ss, &k_cbar<Ss, 1>}), ...);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "sync_bar");
  std::vector<BCase> cases;
  add_cta(cases, std::integer_sequence<int, 2, 4, 8, 12, 16, 32>{});
  add_cluster(cases, std::integer_sequence<int, 2, 4, 8>{});
  if (args.reps > kMaxReps) {
    std::fprintf(stderr, "reps > %d\n", kMaxReps);
    return 2;
  }
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * (33 * kMaxReps + 64)));
  // all-warp wake rows
  using AK = void (*)(std::uint64_t*, int, std::uint32_t);
  const std::vector<std::pair<int, AK>> all = {{2, &k_barall<2>},   {4, &k_barall<4>},   {8, &k_barall<8>},
                                               {12, &k_barall<12>}, {16, &k_barall<16>}, {32, &k_barall<32>}};
  for (const auto& [nw, k] : all) {
    const std::string variant = "bar.cta.w" + std::to_string(nw) + ".all";
    if (!mb_selected(args, variant)) continue;
    const int reps = std::min(args.reps, kAllReps);
    std::vector<std::vector<double>> per(nw);
    for (int l = 0; l < args.launches; ++l) {
      k<<<1, nw * 32>>>(d_out, reps, static_cast<std::uint32_t>(l));
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(static_cast<std::size_t>(reps) * (nw + 1));
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
      for (int r = 1; r < reps; ++r)
        for (int wi = 0; wi < nw; ++wi)
          per[wi].push_back(static_cast<double>(static_cast<std::int64_t>(h[r * (nw + 1) + 1 + wi] - h[r * (nw + 1)])));
    }
    for (int wi = 0; wi < nw; ++wi)
      w.row(variant + ".w" + std::to_string(wi), wi, "\"threads\":" + std::to_string(nw * 32) + ",\"warp\":" + std::to_string(wi),
            per[wi]);
  }
  for (const auto& c : cases) {
    if (!mb_selected(args, c.variant)) continue;
    std::vector<double> k0, k1, klast;
    for (int l = 0; l < args.launches; ++l) {
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(c.cluster, 1, 1);
      cfg.blockDim = dim3(c.threads, 1, 1);
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = c.cluster;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, d_out, args.reps, static_cast<std::uint32_t>(l)));
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(4 * static_cast<std::size_t>(args.reps));
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
      for (int r = 1; r < args.reps; ++r) {
        k0.push_back(static_cast<double>(static_cast<std::int64_t>(h[4 * r + 1] - h[4 * r])));
        if (c.cluster == 1) {
          k1.push_back(static_cast<double>(static_cast<std::int64_t>(h[4 * r + 2] - h[4 * r])));
          klast.push_back(static_cast<double>(static_cast<std::int64_t>(h[4 * r + 3] - h[4 * r])));
        }
      }
    }
    const std::string params = "\"threads\":" + std::to_string(c.threads) + ",\"cluster\":" + std::to_string(c.cluster);
    w.row(c.variant + ".arriver", 1, params, k0);
    if (c.cluster == 1) {
      w.row(c.variant + ".warp1", 1, params, k1);
      w.row(c.variant + ".warplast", 1, params, klast);
    }
  }
  return w.failures ? 1 : 0;
}
