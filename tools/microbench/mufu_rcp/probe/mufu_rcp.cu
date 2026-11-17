// mufu_rcp: calibrated MUFU op + FMUL dependent chains (protocol-v6
// addendum).  One warp, W=1; U in {64,128,256}; cell = <op>_mul:
// alternating MUFU.<OP> + mul.f32 on one register.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../tensor_pipe/mb.cuh"

#define OP_RCP "rcp.approx.ftz.f32 %0, %0;"
#define OP_EX2 "ex2.approx.ftz.f32 %0, %0;"
#define OP_RSQ "rsqrt.approx.ftz.f32 %0, %0;"
#define OP_SQT "sqrt.approx.ftz.f32 %0, %0;"
#define OP_SIN "sin.approx.ftz.f32 %0, %0;"

template <int OP, int U>
__global__ void __launch_bounds__(32, 1) k_cell(std::uint64_t* out,
                                                const float* opq,
                                                int reps) {
  const int lane = threadIdx.x & 31;
  float x = opq[lane & 15] + (float)(lane * 1e-4f);
  for (int rep = 0; rep < reps; ++rep) {
    __syncwarp();
    const std::uint64_t t_s = mb_clock();
    float acc = x;
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if constexpr (OP == 0)
        asm volatile(OP_RCP : "+f"(acc));
      else if constexpr (OP == 1)
        asm volatile(OP_EX2 : "+f"(acc));
      else if constexpr (OP == 2)
        asm volatile(OP_RSQ : "+f"(acc));
      else if constexpr (OP == 3)
        asm volatile(OP_SQT : "+f"(acc));
      else
        asm volatile(OP_SIN : "+f"(acc));
      asm volatile("mul.f32 %0, %0, %1;" : "+f"(acc) : "f"(opq[0]));
    }
    const std::uint64_t t_w = mb_clock();
    x = acc + (float)(t_w & 7) * 1e-5f;
    out[rep * 2] = t_s;
    out[rep * 2 + 1] = t_w;
  }
  out[reps * 2] = (std::uint64_t)__float_as_uint(x);
}

static const char* NAMES[5] = {"rcp_mul", "ex2_mul", "rsq_mul",
                               "sqrt_mul", "sin_mul"};

template <int OP>
void run_cell(MbWriter& w, const MbArgs& a, float* d_opq,
              std::uint64_t* d_out, int op) {
  using K = void (*)(std::uint64_t*, const float*, int);
  static const K ks[3] = {k_cell<OP, 64>, k_cell<OP, 128>,
                          k_cell<OP, 256>};
  static const int US[3] = {64, 128, 256};
  for (int ui = 0; ui < 3; ++ui) {
    char vname[32];
    std::snprintf(vname, sizeof vname, "%s.u%d", NAMES[op], US[ui]);
    if (!mb_selected(a, vname)) continue;
    std::vector<double> samples;
    for (int l = 0; l < a.launches; ++l) {
      ks[ui]<<<1, 32>>>(d_out, d_opq, a.reps);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(a.reps * 2);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      for (int rep = 0; rep < a.reps; ++rep)
        samples.push_back((double)(h[rep * 2 + 1] - h[rep * 2]));
    }
    char params[96];
    std::snprintf(params, sizeof params,
                  "\"variant\":\"%s\",\"u\":%d,\"w\":1", NAMES[op],
                  US[ui]);
    w.row(vname, US[ui], params, samples);
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
  MB_CHECK(cudaMalloc(&d_opq, sizeof(float) * 16));
  float opq[16];
  for (int i = 0; i < 16; ++i) opq[i] = 1.0009765625f + i * 1e-3f;
  MB_CHECK(cudaMemcpy(d_opq, opq, sizeof opq, cudaMemcpyHostToDevice));
  run_cell<0>(w, args, d_opq, d_out, 0);
  run_cell<1>(w, args, d_opq, d_out, 1);
  run_cell<2>(w, args, d_opq, d_out, 2);
  run_cell<3>(w, args, d_opq, d_out, 3);
  run_cell<4>(w, args, d_opq, d_out, 4);
  cudaFree(d_out);
  cudaFree(d_opq);
  return w.failures ? 1 : 0;
}
