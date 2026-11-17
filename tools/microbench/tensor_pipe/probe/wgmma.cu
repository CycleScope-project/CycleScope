// tools/microbench/tensor_pipe/probe/wgmma.cu — H100 (sm_90a) wgmma probe.
//
// protocol-v2 probe.h100: one CTA of one warpgroup (128 threads), operands
// in shared memory with valid no-swizzle K-major descriptors (B always
// SMEM; A from SMEM = ss, A from registers = rs), accumulator F32 (S32 for
// IGMMA; F16 for the f16-acc cells).  Timing thread = warp 0 lane 0.
//   lat: t_s clock64; wgmma.fence; U x wgmma.mma_async (same D, scale-d=1);
//        wgmma.commit_group; wgmma.wait_group 0; t_w clock64; U in {0,1}.
//   cad: same window, U in {16,32,64,128} straight-line, one commit group.
//   wg2: two warpgroups of one CTA run the cad window concurrently
//        (bf16 ss, n in {64,128,256}).
// Output: mechtree.mb.v1 jsonl rows (variant, u, params, samples).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../mb.cuh"

// no-swizzle K-major shared-memory descriptor (same layout the hopper-prototype
// hgmma probes use): start_addr>>4, leading-byte-offset>>4, stride>>4.
__device__ __forceinline__ std::uint64_t w_desc(const void* smem_ptr) {
  const std::uint32_t addr =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem_ptr));
  std::uint64_t desc = 0;
  desc |= static_cast<std::uint64_t>((addr >> 4) & 0x3fff);
  desc |= static_cast<std::uint64_t>((128 >> 4) & 0x3fff) << 16;
  desc |= static_cast<std::uint64_t>((256 >> 4) & 0x3fff) << 32;
  return desc;
}

// kind gates: all on unless the build defines a subset (WGMMA_KIND_*)
#if !defined(WGMMA_KIND_F16) && !defined(WGMMA_KIND_F16A) && \
    !defined(WGMMA_KIND_BF16) && !defined(WGMMA_KIND_TF32) && \
    !defined(WGMMA_KIND_E4M3E4M3) && !defined(WGMMA_KIND_E4M3E5M2) && \
    !defined(WGMMA_KIND_E5M2E5M2) && !defined(WGMMA_KIND_S8S8)
#define WGMMA_KIND_F16 1
#define WGMMA_KIND_F16A 1
#define WGMMA_KIND_BF16 1
#define WGMMA_KIND_TF32 1
#define WGMMA_KIND_E4M3E4M3 1
#define WGMMA_KIND_E4M3E5M2 1
#define WGMMA_KIND_E5M2E5M2 1
#define WGMMA_KIND_S8S8 1
#endif

#include "wgmma_impl.cuh"

static constexpr int kSmemBytes = 48 * 1024;

int main(int argc, char** argv) {
  MbArgs args = mb_parse(argc, argv);
  // one jsonl per per-kind binary (basename, no extension)
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * args.reps));
  MB_CHECK(cudaMalloc(&d_val, sizeof(std::uint32_t) * 2));
  for (int ci = 0; ci < W_CELLS_N; ++ci) {
    const WCell& c = W_CELLS[ci];
    const bool wg2_ok = std::strcmp(c.kind, "bf16") == 0 &&
                        std::strcmp(c.src, "ss") == 0 &&
                        (c.n == 64 || c.n == 128 || c.n == 256) &&
                        std::strcmp(c.mode, "cad") == 0;
    for (int wgroups = 1; wgroups <= (wg2_ok ? 2 : 1); ++wgroups) {
      std::string variant = std::string(c.mode) + "." + c.kind + "." +
                            c.acc + "." + c.src + ".n" +
                            std::to_string(c.n) + ".u" +
                            std::to_string(c.u) +
                            (wgroups == 2 ? ".wg2" : "");
      if (!mb_selected(args, variant)) continue;
      std::vector<double> samples;
      for (int l = 0; l < args.launches; ++l) {
        WKern kern = wgroups == 2 ? c.kern2 : c.kern;
        if (!kern) continue;
        MB_CHECK(cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSmemBytes));
        kern<<<1, 128 * wgroups, kSmemBytes>>>(
            d_out, d_val, args.reps, wgroups,
            static_cast<std::uint32_t>(l));
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(2 * static_cast<std::size_t>(args.reps));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r)
          samples.push_back(static_cast<double>(
              static_cast<std::int64_t>(h[2 * r + 1] - h[2 * r])));
      }
      char extra[64];
      std::snprintf(extra, sizeof(extra), "\"wg2\":%s",
                    wgroups == 2 ? "true" : "false");
      const std::string params =
          "\"class\":\"" + std::string(c.cls) + "\",\"dtype\":\"" +
          c.kind + "\",\"acc\":\"" + c.acc + "\",\"src\":\"" + c.src +
          "\",\"m\":64,\"n\":" + std::to_string(c.n) +
          ",\"k\":" + std::to_string(c.k) + ",\"mode\":\"" + c.mode + "\"";
      w.row(variant, c.u, params, samples, extra);
    }
  }
  return 0;
}
