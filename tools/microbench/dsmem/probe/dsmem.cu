// dsmem probe (protocol.json): SASS form of peer DSMEM access +
// report-only peer load latency / store visibility.  Cluster of 2 CTAs.
#include "../mb.cuh"
#include <cuda.h>
#include <string>
#include <vector>

// mapa.shared::cluster.u32: shared address of `addr` as seen by rank
// `rank` of this cluster.
__device__ __forceinline__ std::uint32_t mapa_u32(std::uint32_t addr,
                                                std::uint32_t rank) {
  std::uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ std::uint32_t ld_shared_cluster(
    std::uint32_t addr) {
  std::uint32_t v;
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(addr)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_shared_cluster(std::uint32_t addr,
                                                  std::uint32_t v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(addr), "r"(v)
               : "memory");
}
__device__ __forceinline__ void cluster_arrive_wait() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// sass_form cell: one ld + one st to the peer rank and one each to the
// own rank through the cluster window (compile-only gate target).
__global__ void __launch_bounds__(32, 1)
k_sass_form(std::uint32_t* sink) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  __shared__ std::uint32_t buf[64];
  const std::uint32_t rank = mb_ctarank();
  const std::uint32_t base =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(buf));
  const std::uint32_t peer = mapa_u32(base, rank ^ 1);
  const std::uint32_t own = mapa_u32(base, rank);
  if (threadIdx.x == 0) {
    st_shared_cluster(peer, 0xdead0000u + rank);
    st_shared_cluster(own, 0xbeef0000u + rank);
    const std::uint32_t v0 = ld_shared_cluster(peer);
    const std::uint32_t v1 = ld_shared_cluster(own);
    sink[rank] = v0 ^ v1;
  }
  cluster_arrive_wait();
#endif
}

// peer_load_latency: rank0 lane0 pointer chase, U dependent loads.
// TARGET: 0 = peer rank, 1 = own rank through the cluster window,
//         2 = plain ld.shared (control).
template <int TARGET, int U>
__global__ void __launch_bounds__(32, 1)
k_peer_load(std::uint64_t* out, std::uint32_t* sink, int reps,
            std::uint32_t seed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  __shared__ std::uint32_t tab[1024];
  const std::uint32_t rank = mb_ctarank();
  const std::uint32_t lane = threadIdx.x & 31;
  // every CTA fills its own 4 KiB chase table (v -> next index)
  for (int i = lane; i < 1024; i += 32)
    tab[i] = (i * 2654435761u + seed) & 1023u;
  cluster_arrive_wait();
  if (rank == 0 && lane == 0) {
    const std::uint32_t base =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(tab));
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      std::uint32_t v = seed + rep;
      const std::uint64_t t0 = mb_clock();
#pragma unroll
      for (int i = 0; i < U; ++i) {
        if constexpr (TARGET == 0) {
          const std::uint32_t a =
              mapa_u32(base + (v & 1023u) * 4u, 1);
          v = ld_shared_cluster(a);
        } else if constexpr (TARGET == 1) {
          const std::uint32_t a =
              mapa_u32(base + (v & 1023u) * 4u, 0);
          v = ld_shared_cluster(a);
        } else {
          asm volatile("ld.shared.u32 %0, [%1];"
                       : "=r"(v) : "r"(base + (v & 1023u) * 4u)
                       : "memory");
        }
      }
      const std::uint64_t t1 = mb_clock();
      out[2 * rep] = t1 - t0;
      out[2 * rep + 1] = v;
    }
  }
  cluster_arrive_wait();
  if (threadIdx.x == 0) sink[rank * 16 + lane] = tab[lane];
#endif
}

// peer_store_visibility: rank0 lane0 does S peer stores, then
// barrier.cluster arrive.release/wait; rank1 reads the stored words
// after its wait.  out on rank0: t(before stores) -> t(after wait).
template <int S>
__global__ void __launch_bounds__(32, 1)
k_peer_store(std::uint64_t* out, std::uint32_t* sink, int reps,
             std::uint32_t seed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  __shared__ std::uint32_t dst[64];
  const std::uint32_t rank = mb_ctarank();
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t base =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(dst));
  const std::uint32_t peer = mapa_u32(base, rank ^ 1);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    if (rank == 0 && lane == 0) {
      const std::uint64_t t0 = mb_clock();
#pragma unroll
      for (int i = 0; i < S; ++i)
        st_shared_cluster(peer + i * 4, seed + rep * 16 + i);
      out[2 * rep] = t0;
    }
    cluster_arrive_wait();   // one call site, all threads
    if (rank == 0 && lane == 0)
      out[2 * rep] = mb_clock() - out[2 * rep];
    if (rank == 1 && lane == 0) {
      std::uint32_t acc = 0;
#pragma unroll
      for (int i = 0; i < S; ++i)
        acc += dst[i];
      out[2 * rep + 1] = acc;
    }
    cluster_arrive_wait();
  }
  if (threadIdx.x == 0) sink[rank * 16 + lane] = dst[lane];
#endif
}

static int launch2(void (*k)(std::uint64_t*, std::uint32_t*, int,
                            std::uint32_t),
                   std::uint64_t* out, std::uint32_t* sink, int reps,
                   std::uint32_t seed) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(2, 1, 1);
  cfg.blockDim = dim3(32, 1, 1);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, k, out, sink, reps, seed);
}

int main(int argc, char** argv) {
  const MbArgs a = mb_parse(argc, argv);
  MbWriter w(a, "dsmem");
  std::uint64_t* d_out;
  std::uint32_t* d_sink;
  cudaMalloc(&d_out, 2 * a.reps * 8);
  cudaMalloc(&d_sink, 64 * 4);
  // sass_form is compile-only on the device side but launch it once so
  // the run has a marker row
  {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, k_sass_form, d_sink);
    if (cudaDeviceSynchronize() != cudaSuccess)
      std::fprintf(stdout, "FAIL sass_form: %s\n",
                   cudaGetErrorName(cudaGetLastError()));
  }

  struct Cell { const char* var; int which; int u; };
  const Cell cells[] = {
      {"lat.peer", 0, 16}, {"lat.peer", 0, 32}, {"lat.peer", 0, 64},
      {"lat.own_window", 1, 16}, {"lat.own_window", 1, 32},
      {"lat.own_window", 1, 64},
      {"lat.lds", 2, 16}, {"lat.lds", 2, 32}, {"lat.lds", 2, 64},
      {"vis.store", 3, 0}, {"vis.store", 3, 1}, {"vis.store", 3, 4},
  };
  for (const auto& c : cells) {
    if (!mb_selected(a, c.var)) continue;
    std::vector<double> s;
    for (int l = 0; l < a.launches; ++l) {
      int rc = 0;
      if (c.which == 0) {
        void (*k)(std::uint64_t*, std::uint32_t*, int, std::uint32_t) =
            c.u == 16 ? &k_peer_load<0, 16>
                      : c.u == 32 ? &k_peer_load<0, 32>
                                  : &k_peer_load<0, 64>;
        rc = launch2(k, d_out, d_sink, a.reps, 0x9e37u + l);
      } else if (c.which == 1) {
        void (*k)(std::uint64_t*, std::uint32_t*, int, std::uint32_t) =
            c.u == 16 ? &k_peer_load<1, 16>
                      : c.u == 32 ? &k_peer_load<1, 32>
                                  : &k_peer_load<1, 64>;
        rc = launch2(k, d_out, d_sink, a.reps, 0x9e37u + l);
      } else if (c.which == 2) {
        void (*k)(std::uint64_t*, std::uint32_t*, int, std::uint32_t) =
            c.u == 16 ? &k_peer_load<2, 16>
                      : c.u == 32 ? &k_peer_load<2, 32>
                                  : &k_peer_load<2, 64>;
        rc = launch2(k, d_out, d_sink, a.reps, 0x9e37u + l);
      } else {
        void (*k)(std::uint64_t*, std::uint32_t*, int, std::uint32_t) =
            c.u == 0 ? &k_peer_store<0>
                     : c.u == 1 ? &k_peer_store<1> : &k_peer_store<4>;
        rc = launch2(k, d_out, d_sink, a.reps, 0x9e37u + l);
      }
      const cudaError_t e =
          (rc ? cudaError_t(rc) : cudaSuccess) != cudaSuccess
              ? cudaError_t(rc) : cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stdout, "FAIL %s launch %d: %s\n", c.var, l,
                     cudaGetErrorName(e));
        w.row(c.var, c.u, "\"status\":\"unrealised:" +
                          std::string(cudaGetErrorName(e)) + "\"", {});
        break;
      }
      std::vector<std::uint64_t> h(2 * a.reps);
      cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost);
      if (l == 0) continue;
      for (int r = 1; r < a.reps; ++r) s.push_back(double(h[2 * r]));
    }
    if (s.size())
      w.row(c.var, c.u, "\"class\":\"DSMEM\"", s);
  }
  return 0;
}
