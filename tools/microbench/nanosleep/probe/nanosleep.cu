// tools/microbench/nanosleep/probe/nanosleep.cu -- protocol v1
// (frozen tools/microbench/nanosleep/protocol.json).
// syncs(H,D): warp0 sleeps in try_wait(H ns hint) on an mbarrier;
// warp1 arrives at t_0+D cycles; delta = t_x(consumer wake) - t_a.
// plain(H): clock64 around nanosleep.u32 H.  bar.sync between reps.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "tensor_pipe/mb.cuh"


__device__ __forceinline__ unsigned sm_addr(const void* p) {
  unsigned u;
  asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; "
               "cvt.u32.u64 %0, t; }" : "=r"(u) : "l"(p));
  return u;
}

template <int H>
__global__ void k_syncs(std::uint64_t* out, const std::uint64_t* dsel,
                        int reps, unsigned phase) {
  __shared__ __align__(8) std::uint64_t bar_obj;
  __shared__ std::uint64_t t0;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                 :: "r"(sm_addr(&bar_obj)));
  }
  __syncthreads();
  const unsigned bar = sm_addr(&bar_obj);
  const std::uint64_t D = *dsel;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (warp == 0 && lane == 0) {
      unsigned p = 0;
      // the arrive completes phase (rep & 1): parity of the phase
      // that THIS rep's arrive completes (phases alternate 0,1,...)
      const unsigned parity = rep & 1u;
      unsigned spin_n = 0;
      do {
        asm volatile(
            "{ .reg .pred P1;\n"
            "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2, %3;\n"
            "selp.b32 %0, 1, 0, P1; }"
            : "=r"(p) : "r"(bar), "r"(parity), "n"(H) : "memory");
        if (++spin_n > 4194304u) {
          g_mb_err = 1;
          break;
        }
      } while (!p);
      const std::uint64_t t_x = mb_clock();
      out[rep * 4 + 0] = t_x;
    } else if (warp == 1 && lane == 0) {
      const std::uint64_t t_0 = mb_clock();
      while (mb_clock() < t_0 + D) {
      }
      const std::uint64_t t_a = mb_clock();
      asm volatile("mbarrier.arrive.shared.b64 _, [%0];"
                   :: "r"(bar) : "memory");
      out[rep * 4 + 1] = t_a;
      out[rep * 4 + 2] = t_0;
    }
    __syncthreads();
  }
}

template <int H>
__global__ void k_plain(std::uint64_t* out, int reps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp != 0 || lane != 0) return;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t_s = mb_clock();
    asm volatile("nanosleep.u32 %0;" :: "n"(H));
    const std::uint64_t t_w = mb_clock();
    out[rep] = t_w - t_s;
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t* d_out = nullptr;
  std::uint64_t* d_d = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 65536));
  MB_CHECK(cudaMalloc(&d_d, sizeof(std::uint64_t)));
  static const int HS[] = {0, 200, 1000, 10000, 100000, 1000000};
  static const int DS[] = {2000, 20000, 200000};
  using KSyncs = void (*)(std::uint64_t*, const std::uint64_t*, int,
                          unsigned);
  using KPlain = void (*)(std::uint64_t*, int);
  static const KSyncs ks[6] = {k_syncs<0>, k_syncs<200>, k_syncs<1000>,
                               k_syncs<10000>, k_syncs<100000>,
                               k_syncs<1000000>};
  static const KPlain kp[6] = {k_plain<0>, k_plain<200>, k_plain<1000>,
                               k_plain<10000>, k_plain<100000>,
                               k_plain<1000000>};
  for (int hi = 0; hi < 6; ++hi) {
    for (int di = 0; di < 3; ++di) {
      char vname[64];
      std::snprintf(vname, sizeof vname, "syncs_h%d_d%d", HS[hi],
                    DS[di]);
      if (!mb_selected(args, vname)) continue;
      const std::uint64_t d64 = DS[di];
      MB_CHECK(cudaMemcpy(d_d, &d64, 8, cudaMemcpyHostToDevice));
      std::vector<double> samples;
      for (int l = 0; l < args.launches; ++l) {
        ks[hi]<<<1, 64>>>(d_out, d_d, args.reps, 0);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        int err = 0;
        MB_CHECK(cudaMemcpyFromSymbol(&err, g_mb_err, sizeof err));
        if (err) { w.failures++; break; }
        std::vector<std::uint64_t> h(args.reps * 4);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int rep = 0; rep < args.reps; ++rep) {
          const long long tx = (long long)h[rep * 4];
          const long long ta = (long long)h[rep * 4 + 1];
          if (tx && ta) samples.push_back((double)(tx - ta));
        }
      }
      char params[128];
      std::snprintf(params, sizeof params,
                    "\"variant\":\"syncs\",\"h\":%d,\"d\":%d", HS[hi],
                    DS[di]);
      w.row(vname, HS[hi], params, samples);
    }
    char pname[64];
    std::snprintf(pname, sizeof pname, "plain_h%d", HS[hi]);
    if (mb_selected(args, pname)) {
      std::vector<double> samples;
      for (int l = 0; l < args.launches; ++l) {
        kp[hi]<<<1, 64>>>(d_out, args.reps);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(args.reps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int rep = 0; rep < args.reps; ++rep)
          samples.push_back((double)h[rep]);
      }
      char params[64];
      std::snprintf(params, sizeof params,
                    "\"variant\":\"plain\",\"h\":%d", HS[hi]);
      w.row(pname, HS[hi], params, samples);
    }
  }
  cudaFree(d_out); cudaFree(d_d);
  return w.failures ? 1 : 0;
}
