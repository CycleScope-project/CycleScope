// descvis: publish -> acquire visibility delay (protocol-v1).
#include "../mb.cuh"
#include <cuda.h>
#include <string>
#include <vector>

__global__ void __launch_bounds__(64, 1) k_descvis(
    unsigned char* map, std::uint64_t* out, int reps,
    std::uint64_t d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ unsigned char smem[];
  const std::uint32_t mbar =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  // (b) SWIZZLE_128B needs a 1024 B aligned shared destination
  const std::uint32_t sdst = (mbar + 64 + 1023u) & ~1023u;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) mb_mbar_init(mbar, 1);
  mb_fence_mbar_init();
  const std::uint64_t gmap = (std::uint64_t)map;
  // (a) the 128 map bytes live in registers; the publish stores the same
  // bytes back, so the map stays valid
  std::uint32_t mv[8][4];
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      asm volatile(
          "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(mv[i][0]), "=r"(mv[i][1]), "=r"(mv[i][2]), "=r"(mv[i][3])
          : "l"(map + i * 16) : "memory");
    }
  }
  for (int rep = 0; rep < reps; ++rep) {
    if (warp == 0 && (threadIdx.x & 31) == 0) {
      const std::uint64_t t_s = mb_clock();
      if (d != ~0ull) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          unsigned char* p = map + i * 16;
          asm volatile(
              "st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
              "l"(p), "r"(mv[i][0]), "r"(mv[i][1]),
              "r"(mv[i][2]), "r"(mv[i][3])
              : "memory");
        }
        asm volatile("fence.proxy.tensormap::generic.release.gpu;"
                     ::: "memory");
      }
      if (d != ~0ull && d) {
        const std::uint64_t t0 = mb_clock();
        while (mb_clock() - t0 < d) {
        }
      }
      const std::uint64_t t_a = mb_clock();
      asm volatile(
          "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::
          "l"(gmap) : "memory");
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
          "r"(mbar), "r"(8192) : "memory");
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
          ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
          "r"(sdst), "l"(gmap), "r"(0), "r"(0), "r"(mbar)
          : "memory");
      const std::uint64_t t_i = mb_clock();
      if (threadIdx.x == 0) {
        out[4 * rep] = t_s; out[4 * rep + 1] = t_a;
        out[4 * rep + 2] = t_i;
      }
    }
    __syncthreads();
    if (warp == 1 && (threadIdx.x & 31) == 0) {
      mb_mbar_wait(mbar, rep & 1);
      const std::uint64_t t_w = mb_clock();
      out[4 * rep + 3] = t_w;
    }
    __syncthreads();
  }
#endif
}

int main(int argc, char** argv) {
  const MbArgs a = mb_parse(argc, argv);
  MbWriter w(a, "descvis");
  const int H = 128, W = 128;
  std::vector<unsigned short> hbuf(H * W);
  for (size_t i = 0; i < hbuf.size(); ++i) hbuf[i] = i & 0x3ff;
  void* d_buf; void* d_map; std::uint64_t* d_out;
  cudaMalloc(&d_buf, H * W * 2);
  cudaMemcpy(d_buf, hbuf.data(), H * W * 2, cudaMemcpyHostToDevice);
  cudaMalloc(&d_map, 128);
  CUtensorMap tmap;
  cuuint64_t dims[2] = {(cuuint64_t)W, (cuuint64_t)H};
  cuuint64_t strides[1] = {(cuuint64_t)W * 2};
  cuuint32_t box[2] = {64, 64};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d_buf, dims, strides,
      box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "encode fail %d\n", r); return 1; }
  cudaMemcpy(d_map, &tmap, 128, cudaMemcpyHostToDevice);
  cudaMalloc(&d_out, 4 * a.reps * 8);
  cudaFuncSetAttribute(k_descvis,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 8192 + 64 + 8192);
  for (std::uint64_t d : {0ull, 128ull, 256ull, 512ull, 1024ull,
                          2048ull, 4096ull, 8192ull, ~0ull}) {
    const std::string cname =
        (d == ~0ull) ? "ctl" : "d" + std::to_string((unsigned long long)d);
    if (!a.cell.empty() && a.cell != cname) continue;
    std::vector<double> s;
    bool bad = false;
    for (int l = 0; l < a.launches && !bad; ++l) {
      k_descvis<<<1, 64, 8192 + 64 + 8192>>>(
          (unsigned char*)d_map, d_out, a.reps, d);
      cudaError_t e1 = cudaGetLastError();
      cudaError_t e2 = cudaDeviceSynchronize();
      if (e1 != cudaSuccess || e2 != cudaSuccess) {
        cudaError_t e = (e1 != cudaSuccess) ? e1 : e2;
        fprintf(stdout, "FAIL cell %s launch %d: %s\n", cname.c_str(), l,
                cudaGetErrorName(e));
        char v[64];
        if (d == ~0ull) std::snprintf(v, sizeof v, "ctl.descvis");
        else std::snprintf(v, sizeof v, "lat.descvis.d%llu",
                           (unsigned long long)d);
        w.row(v, (int)(d == ~0ull ? 0 : d),
              "\"class\":\"DESCVIS\",\"delay\":" +
                  std::to_string((long long)(d == ~0ull ? -1 : d)) +
                  ",\"status\":\"unrealised:" + cudaGetErrorName(e) +
                  "\"", {});
        bad = true;
        break;
      }
      std::vector<std::uint64_t> h(4 * a.reps);
      cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost);
      if (l == 0) {
        fprintf(stderr, "rep1 t_s=%llu t_a=%llu t_i=%llu t_w=%llu\n",
                (unsigned long long)h[4], (unsigned long long)h[5],
                (unsigned long long)h[6], (unsigned long long)h[7]);
        continue;
      }
      for (int rr = 1; rr < a.reps; ++rr)
        s.push_back(double(h[4 * rr + 3] - h[4 * rr + 1]));
    }
    char v[64];
    if (d == ~0ull) std::snprintf(v, sizeof v, "ctl.descvis");
    else std::snprintf(v, sizeof v, "lat.descvis.d%llu",
                       (unsigned long long)d);
    w.row(v, (int)(d == ~0ull ? 0 : d),
          "\"class\":\"DESCVIS\",\"delay\":" +
              std::to_string((long long)(d == ~0ull ? -1 : d)), s);
  }
  return 0;
}
