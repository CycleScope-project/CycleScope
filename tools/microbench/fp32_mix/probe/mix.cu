// fp32_mix: softmax-quartet dispatch study (protocol.json).
// One CTA of 4*W warps; warp w -> SMSP w%4; per-warp window = U
// quartets of interleaved 4-chain ops; variants quartet/trio/mufu/fmnmx.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../tensor_pipe/mb.cuh"

#define MNMX "min.f32 %0, %0, %3;"
#define FFMA "fma.rn.f32 %0, %0, %2, %1;"
#define MUFU "ex2.approx.ftz.f32 %0, %0;"
#define FADD "add.f32 %0, %0, %4;"

template <int VARIANT>
__device__ __forceinline__ void four(float (&x)[4], const float* c) {
  if constexpr (VARIANT == 0) {  // quartet: FMNMX FFMA MUFU FADD
    asm volatile(MNMX : "+f"(x[0]) : "f"(x[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]));
    asm volatile(FFMA : "+f"(x[1]) : "f"(x[2]), "f"(c[0]), "f"(c[3]), "f"(c[4]));
    asm volatile(MUFU : "+f"(x[2]) : "f"(x[3]), "f"(c[0]), "f"(c[5]), "f"(c[6]));
    asm volatile(FADD : "+f"(x[3]) : "f"(x[0]), "f"(c[0]), "f"(c[7]), "f"(c[8]));
  } else if constexpr (VARIANT == 1) {  // trio: FMNMX FFMA FADD
    asm volatile(MNMX : "+f"(x[0]) : "f"(x[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]));
    asm volatile(FFMA : "+f"(x[1]) : "f"(x[2]), "f"(c[0]), "f"(c[3]), "f"(c[4]));
    asm volatile(FADD : "+f"(x[3]) : "f"(x[0]), "f"(c[0]), "f"(c[7]), "f"(c[8]));
  } else if constexpr (VARIANT == 2) {  // mufu only
    asm volatile(MUFU : "+f"(x[0]) : "f"(x[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]));
    asm volatile(MUFU : "+f"(x[1]) : "f"(x[2]), "f"(c[0]), "f"(c[3]), "f"(c[4]));
    asm volatile(MUFU : "+f"(x[2]) : "f"(x[3]), "f"(c[0]), "f"(c[5]), "f"(c[6]));
    asm volatile(MUFU : "+f"(x[3]) : "f"(x[0]), "f"(c[0]), "f"(c[7]), "f"(c[8]));
  } else {  // fmnmx only
    asm volatile(MNMX : "+f"(x[0]) : "f"(x[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]));
    asm volatile(MNMX : "+f"(x[1]) : "f"(x[2]), "f"(c[0]), "f"(c[3]), "f"(c[4]));
    asm volatile(MNMX : "+f"(x[2]) : "f"(x[3]), "f"(c[0]), "f"(c[5]), "f"(c[6]));
    asm volatile(MNMX : "+f"(x[3]) : "f"(x[0]), "f"(c[0]), "f"(c[7]), "f"(c[8]));
  }
}

template <int VARIANT, int U>
__global__ void __launch_bounds__(512, 1) k_cell(
    std::uint64_t* out, const float* opq, int reps, int warps) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= warps) return;
  float x[4];
  const float* c = opq + (lane & 15) * 16;  // per-lane zero-table
  for (int i = 0; i < 4; ++i)
    x[i] = c[i] + (float)(warp * 1e-4f) + (float)(lane * 3e-5f);
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t_s = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) four<VARIANT>(x, c);
    const std::uint64_t t_w = mb_clock();
    x[0] += (float)(t_w & 3) * 1e-5f;
    out[(rep * 4 + warp) * 2] = t_s;
    out[(rep * 4 + warp) * 2 + 1] = t_w;
  }
  out[(reps * 4 + warp) * 2] =
      (std::uint64_t)__float_as_uint(x[0] + x[1] + x[2] + x[3]);
}

static const char* NAMES[4] = {"quartet", "trio", "mufu", "fmnmx"};

template <int V>
void run_variant(MbWriter& w, const MbArgs& a, float* d_opq,
                 std::uint64_t* d_out, int vi) {
  using K = void (*)(std::uint64_t*, const float*, int, int);
  static const K ks[3] = {k_cell<V, 32>, k_cell<V, 64>, k_cell<V, 128>};
  static const int US[3] = {32, 64, 128};
  for (int wi = 1; wi <= 4; ++wi) {
    for (int ui = 0; ui < 3; ++ui) {
      char vname[40];
      std::snprintf(vname, sizeof vname, "%s.u%d.w%d", NAMES[vi],
                    US[ui], wi);
      if (!mb_selected(a, vname)) continue;
      std::vector<double> samples[16];
      for (int l = 0; l < a.launches; ++l) {
        ks[ui]<<<1, wi * 128>>>(d_out, d_opq, a.reps, wi * 4);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(a.reps * 8);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int rep = 0; rep < a.reps; ++rep)
          for (int wp = 0; wp < wi * 4; ++wp)
            samples[wp].push_back(
                (double)(h[(rep * 4 + wp) * 2 + 1] - h[(rep * 4 + wp) * 2]));
      }
      for (int wp = 0; wp < wi * 4; ++wp) {
        char params[128];
        std::snprintf(params, sizeof params,
                      "\"variant\":\"%s\",\"u\":%d,\"w\":%d,\"warp\":%d",
                      NAMES[vi], US[ui], wi, wp);
        char vn[48];
        std::snprintf(vn, sizeof vn, "%s.warp%d", vname, wp);
        w.row(vn, US[ui], params, samples[wp]);
      }
    }
  }
}

int main(int argc, char** argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t* d_out = nullptr;
  float* d_opq = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 8192));
  MB_CHECK(cudaMalloc(&d_opq, sizeof(float) * 256));
  float opq[256];
  for (int i = 0; i < 256; ++i)
    opq[i] = 1.001f + (float)(i % 16) * 1e-3f;
  MB_CHECK(cudaMemcpy(d_opq, opq, sizeof opq, cudaMemcpyHostToDevice));
  run_variant<0>(w, args, d_opq, d_out, 0);
  run_variant<1>(w, args, d_opq, d_out, 1);
  run_variant<2>(w, args, d_opq, d_out, 2);
  run_variant<3>(w, args, d_opq, d_out, 3);
  cudaFree(d_out);
  cudaFree(d_opq);
  return w.failures ? 1 : 0;
}
