#define main store_queue_original_main
#include "../lsu_accept/probe/store_queue.cu"
#undef main

template <>
__device__ __forceinline__ void store_one<4>(std::uint64_t p, unsigned s0,
                                            unsigned s1, unsigned s2,
                                            unsigned s3) {
  asm volatile("st.global.u16 [%0], %1;" :: "l"(p), "r"(s0) : "memory");
}

int main(int argc, char** argv) {
  MbArgs args = mb_parse(argc, argv);
  MbWriter writer(args, "store_scatter");
  std::uint64_t *out = nullptr, *offsets = nullptr;
  unsigned *sink = nullptr, *zeros = nullptr;
  char* base = nullptr;
  MB_CHECK(cudaMalloc(&out, sizeof(std::uint64_t) * 8 * args.reps * (kD + 1)));
  MB_CHECK(cudaMalloc(&offsets, sizeof(std::uint64_t) * kD * 8 * 32));
  MB_CHECK(cudaMalloc(&sink, sizeof(unsigned) * 8));
  MB_CHECK(cudaMalloc(&zeros, sizeof(unsigned) * 1024));
  MB_CHECK(cudaMemset(zeros, 0, sizeof(unsigned) * 1024));
  MB_CHECK(cudaMalloc(&base, 2u << 20));
  MB_CHECK(cudaMemset(base, 0, 2u << 20));
  struct Cell { const char* name; int variant; int width; };
  const Cell cells[] = {{"stg2", 4, 2}, {"stg4", V_STG4, 4},
                        {"stg8", V_STG8, 8}, {"stg16", V_STG16, 16},
                        {"none", V_NONE, 16}};
  for (const auto& c : cells) {
    for (int stride : {c.width, 32, 128}) {
      if (c.variant == V_NONE && stride != c.width) continue;
      std::vector<std::uint64_t> host_offsets(kD * 8 * 32);
      for (int i = 0; i < kD * 8 * 32; ++i)
        host_offsets[i] = (std::uint64_t(i / 32) * 8192 +
                           std::uint64_t(i % 32) * stride) * 16 / c.width;
      MB_CHECK(cudaMemcpy(offsets, host_offsets.data(), host_offsets.size() * 8,
                          cudaMemcpyHostToDevice));
      for (int warps : {1, 2}) {
        const std::string name = std::string(c.name) + ".s" +
            std::to_string(stride) + ".w" + std::to_string(warps);
        if (!mb_selected(args, name)) continue;
        std::vector<std::vector<double>> gaps(kD + 1);
        for (int launch = 0; launch < args.launches; ++launch) {
          switch (c.variant) {
            case 4:
              k_cell<4><<<1, 32 * warps>>>(out, sink, base, zeros, offsets, args.reps, c.width);
              break;
            case V_STG4:
              k_cell<V_STG4><<<1, 32 * warps>>>(out, sink, base, zeros, offsets, args.reps, c.width);
              break;
            case V_STG8:
              k_cell<V_STG8><<<1, 32 * warps>>>(out, sink, base, zeros, offsets, args.reps, c.width);
              break;
            case V_STG16:
              k_cell<V_STG16><<<1, 32 * warps>>>(out, sink, base, zeros, offsets, args.reps, c.width);
              break;
            default:
              k_cell<V_NONE><<<1, 32 * warps>>>(out, sink, base, zeros, offsets, args.reps, c.width);
          }
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          std::vector<std::uint64_t> host(warps * args.reps * (kD + 1));
          MB_CHECK(cudaMemcpy(host.data(), out, host.size() * 8, cudaMemcpyDeviceToHost));
          for (int rep = 1; rep < args.reps; ++rep)
            for (int warp = 0; warp < warps; ++warp) {
              const auto* row = host.data() + (warp * args.reps + rep) * (kD + 1);
              for (int k = 1; k <= kD; ++k)
                gaps[k].push_back(static_cast<double>(row[k] - row[k - 1]));
              gaps[0].push_back(static_cast<double>(row[kD] - row[0]));
            }
        }
        for (int k = 0; k <= kD; ++k) {
          const std::string params = "\"width_bytes\":" + std::to_string(c.width) +
              ",\"stride_bytes\":" + std::to_string(stride) +
              ",\"warps\":" + std::to_string(warps) +
              ",\"k\":" + std::to_string(k);
          writer.row(name, k, params, gaps[k]);
        }
      }
    }
  }
  MB_CHECK(cudaFree(base));
  MB_CHECK(cudaFree(zeros));
  MB_CHECK(cudaFree(sink));
  MB_CHECK(cudaFree(offsets));
  MB_CHECK(cudaFree(out));
  return 0;
}
