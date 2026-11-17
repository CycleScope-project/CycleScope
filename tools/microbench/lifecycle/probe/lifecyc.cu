// kernel.lifecycle envelope (sched_rules.envelope): the hardware cost of a
// launch outside the instruction timeline.  Every thread reads the clock as
// the kernel's first instruction (t0), runs a dependent FFMA chain of L
// instances, reads the clock again (t1); lane 0 of every warp stores t0/t1
// and thread 0 of every CTA stores %smid, so the host can reconstruct the
// per-SM marker span.  Axes C (cluster size), W (warps), S (dynamic smem),
// L (body length); counter-only probe (--reps 1), one LAUNCH line per launch.
#include "../mb.cuh"

#include <string>
#include <vector>

extern __shared__ std::uint32_t k_life_dsm[];

template <int L>
__global__ void __launch_bounds__(1024, 1) k_life(std::uint64_t* out, float s) {
  // t0 must be the kernel's first instruction: read the clock before any
  // threadIdx/parameter use so ptxas has no operand to schedule ahead of it.
  const std::uint64_t t0 = mb_clock();
  float x = s + threadIdx.x;
#pragma unroll
  for (int i = 0; i < L; ++i) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(x) : "f"(s));
  const std::uint64_t t1 = mb_clock();
  const int nwarp = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int words = gridDim.x * nwarp;
  if ((threadIdx.x & 31) == 0) {
    out[2 * (blockIdx.x * nwarp + warp)] = t0;
    out[2 * (blockIdx.x * nwarp + warp) + 1] = t1;
  }
  if (threadIdx.x == 0) {
    out[2 * words + blockIdx.x] = mb_smid();
    std::uint32_t mix = __float_as_uint(x);
    // A real dynamic-smem allocation must be touched, but the access has to
    // stay outside the timed window (no memory instructions between the CS2R
    // reads), so it happens after t1 and feeds back into the output word.
    std::uint32_t dsz = 0;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz >= 4) {
      const std::uint32_t a = mb_smem_u32(k_life_dsm);
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(static_cast<std::uint32_t>(t1)) : "memory");
      std::uint32_t v;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
      mix ^= v;
    }
    out[2 * words + gridDim.x] = mix;
  }
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  if (args.reps != 1) {
    std::fprintf(stderr, "counter-only probe requires --reps 1\n");
    return 2;
  }
  using K = void (*)(std::uint64_t*, float);
  const struct {
    int l;
    K f;
  } kernels[] = {{0, &k_life<0>}, {1024, &k_life<1024>}, {4096, &k_life<4096>}};
  const int cs[] = {1, 2};
  const int ss[] = {0, 204800};
  const int ws[] = {1, 4, 8, 16, 32};
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, (2 * 2 * 32 + 2 + 1) * sizeof(std::uint64_t)));
  bool smem_attr_set = false;
  for (int c : cs) {
    for (int s : ss) {
      for (int w : ws) {
        for (const auto& kk : kernels) {
          char name[64];
          std::snprintf(name, sizeof(name), "life.c%d.w%d.s%d.l%d", c, w, s, kk.l);
          if (!mb_selected(args, name)) continue;
          const int nwords = 2 * c * w + c + 1;
          if (s > 0 && !smem_attr_set) {
            for (const auto& k2 : kernels) {
              MB_CHECK(cudaFuncSetAttribute(
                  reinterpret_cast<const void*>(k2.f),
                  cudaFuncAttributeMaxDynamicSharedMemorySize, 232448));
            }
            smem_attr_set = true;
          }
          for (int l = 0; l < args.launches; ++l) {
            if (c == 2) {
              cudaLaunchConfig_t cfg = {};
              cfg.gridDim = dim3(2, 1, 1);
              cfg.blockDim = dim3(32 * w, 1, 1);
              cfg.dynamicSmemBytes = static_cast<std::size_t>(s);
              cudaLaunchAttribute attr[1];
              attr[0].id = cudaLaunchAttributeClusterDimension;
              attr[0].val.clusterDim.x = 2;
              attr[0].val.clusterDim.y = 1;
              attr[0].val.clusterDim.z = 1;
              cfg.attrs = attr;
              cfg.numAttrs = 1;
              MB_CHECK(cudaLaunchKernelEx(&cfg, kk.f, d_out, 1.0f));
            } else {
              kk.f<<<1, 32 * w, s>>>(d_out, 1.0f);
              MB_CHECK(cudaGetLastError());
            }
            MB_CHECK(cudaDeviceSynchronize());
            MB_CHECK(cudaGetLastError());
            std::vector<std::uint64_t> h(static_cast<std::size_t>(nwords));
            MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(std::uint64_t),
                                cudaMemcpyDeviceToHost));
            std::string line;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "LAUNCH k_life variant=%s launch=%d smid=", name, l);
            line += buf;
            for (int b = 0; b < c; ++b) {
              std::snprintf(buf, sizeof(buf), "%s%llu", b ? "," : "",
                            static_cast<unsigned long long>(h[2 * c * w + b]));
              line += buf;
            }
            for (const char* tag : {" t0=", " t1="}) {
              line += tag;
              for (int i = 0; i < c * w; ++i) {
                std::snprintf(buf, sizeof(buf), "%s%llu", i ? "," : "",
                              static_cast<unsigned long long>(h[2 * i + (tag[2] == '1')]));
                line += buf;
              }
            }
            std::printf("%s\n", line.c_str());
          }
        }
      }
    }
  }
  return 0;
}
