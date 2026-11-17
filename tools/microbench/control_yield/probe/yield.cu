// tools/microbench/control_yield/probe/yield.cu -- protocol v1
// (frozen tools/microbench/control_yield/protocol.json).  One CTA of
// 4*W warps (W in {1,2,4}; warp w on SMSP w % 4), one shared mbarrier
// completed before the window so every try_wait succeeds at once;
// bar.sync before every rep; lane 0 records t_s / t_w around a
// `.pragma nounroll` loop of K iterations (K in {64,128,256}).
// spin:        try_wait parity + @!p bra (YIELD at loop head) + FFMA
// nospin:      ISETP counter + bra (no YIELD) + same FFMA
// spin_nowait: test_wait + same structure (YIELD presence recorded)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "tensor_pipe/mb.cuh"

enum : int { V_SPIN, V_NOSPIN, V_SPIN_NOWAIT };

__device__ __forceinline__ unsigned sm_addr(const void* p) {
  unsigned u;
  asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; "
               "cvt.u32.u64 %0, t; }" : "=r"(u) : "l"(p));
  return u;
}

template <int V, int K>
__global__ void k_cell(std::uint64_t* out, unsigned* sink, int reps,
                       int warps) {
  __shared__ __align__(8) std::uint64_t bar_obj;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                 :: "r"(sm_addr(&bar_obj)));
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];"
                 :: "r"(sm_addr(&bar_obj)));
  }
  __syncthreads();
  const unsigned bar = sm_addr(&bar_obj);
  unsigned phase = 0, acc = lane + warp;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (lane == 0) {
      const std::uint64_t t_s = mb_clock();
#pragma nounroll
      for (int k = 0; k < K; ++k) {
        if constexpr (V == V_SPIN) {
          unsigned p;
          asm volatile(
              "{ .reg .pred P1;\n"
              "LAB_WAIT:\n"
              "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2;\n"
              "@P1 bra DONE_%=;\n"
              "bra LAB_WAIT;\n"
              "DONE_%=: }\n"
              : "=r"(p) : "r"(bar), "r"(phase) : "memory");
          acc += p;
        } else if constexpr (V == V_NOSPIN) {
          unsigned p = k < K ? 1u : 0u;
          acc += p;
        } else {
          unsigned p;
          asm volatile(
              "{ .reg .pred P1;\n"
              "mbarrier.test_wait.parity.shared.b64 P1, [%1], %2;\n"
              "selp.b32 %0, 1, 0, P1; }"
              : "=r"(p) : "r"(bar), "r"(phase) : "memory");
          acc += p;
        }
        acc = acc * 3 + 1;
        float fa = (float)acc;
        asm volatile("fma.rn.f32 %0, %0, 0.5, %0;" : "+f"(fa));
        acc += (unsigned)__float_as_uint(fa) & 1;
      }
      const std::uint64_t t_w = mb_clock();
      out[rep * 32 * 8 + warp] = t_w - t_s;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) sink[0] = acc;
}

using CKern = void (*)(std::uint64_t*, unsigned*, int, int);

int main(int argc, char** argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t* d_out = nullptr; unsigned* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 65536));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(unsigned) * 64));
  MB_CHECK(cudaMemset(d_out, 0, sizeof(std::uint64_t) * 65536));
  const char* names[] = {"spin", "nospin", "spin_nowait"};
  static const CKern kern[3][3] = {
    {k_cell<V_SPIN, 64>, k_cell<V_SPIN, 128>, k_cell<V_SPIN, 256>},
    {k_cell<V_NOSPIN, 64>, k_cell<V_NOSPIN, 128>, k_cell<V_NOSPIN, 256>},
    {k_cell<V_SPIN_NOWAIT, 64>, k_cell<V_SPIN_NOWAIT, 128>,
     k_cell<V_SPIN_NOWAIT, 256>},
  };
  static const int KS[] = {64, 128, 256};
  for (int vi = 0; vi < 3; ++vi)
    for (int wg = 1; wg <= 4; wg *= 2)
      for (int ki = 0; ki < 3; ++ki) {
        char vname[64];
        std::snprintf(vname, sizeof vname, "%s.w%d", names[vi], wg);
        if (!mb_selected(args, vname)) continue;
        std::vector<double> samples;
        for (int l = 0; l < args.launches; ++l) {
          kern[vi][ki]<<<1, 32 * 4 * wg>>>(d_out, d_sink,
                                         args.reps, 4 * wg);
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          std::vector<std::uint64_t> h(args.reps * 32 * 8);
          MB_CHECK(cudaMemcpy(h.data(), d_out,
                              h.size() * 8, cudaMemcpyDeviceToHost));
          for (int rep = 0; rep < args.reps; ++rep)
            for (int ww = 0; ww < 4 * wg; ++ww)
              samples.push_back((double)h[rep * 32 * 8 + ww]);
        }
        char params[128];
        std::snprintf(params, sizeof params,
                      "\"variant\":\"%s\",\"w\":%d,\"k\":%d,"
                      "\"window\":\"warp\"", names[vi], wg, KS[ki]);
        w.row(vname, KS[ki], params, samples);
      }
  cudaFree(d_out); cudaFree(d_sink);
  return w.failures ? 1 : 0;
}
