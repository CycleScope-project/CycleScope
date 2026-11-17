// tensor_pipe protocol v2 (tools/microbench/tensor_pipe/protocol-v2.json):
// the umma.cu cells (6 umma kinds x cta_group {1,2} x A from SMEM / TMEM) with
// xw U in {0, 1} (single-op latency) and cad U in {16, 32, 64, 128} (steady-
// state cadence), over every N the instruction descriptor allows at M = 128
// per CTA (16..256 step 16), SWIZZLE_128B only.  Kernels, descriptors and the
// measurement windows are umma_impl.cuh (verbatim blackwell-prototype); main() is
// umma_impl.cuh umma_main() with only the N loop and the layout loop changed.
#include "../umma_impl.cuh"

static int cadence_main(int argc, char** argv, const char* probe, const std::vector<UCase>& cases) {
  std::uint32_t seed_base = 0;
  int forwarded = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--seed-base") != 0) {
      argv[forwarded++] = argv[i];
      continue;
    }
    if (++i == argc) {
      std::fprintf(stderr, "missing --seed-base value\n");
      return 2;
    }
    char* end = nullptr;
    const unsigned long value = std::strtoul(argv[i], &end, 10);
    if (argv[i][0] < '0' || argv[i][0] > '9' || *end || value > 0xfffffffful) {
      std::fprintf(stderr, "invalid --seed-base value\n");
      return 2;
    }
    seed_base = static_cast<std::uint32_t>(value);
  }
  argc = forwarded;
  argv[argc] = nullptr;
  const MbArgs args = mb_parse(argc, argv);
  if (seed_base > 0xffffffffu - static_cast<std::uint32_t>(args.launches - 1)) {
    std::fprintf(stderr, "--seed-base range overflows\n");
    return 2;
  }
  MbWriter w(args, probe);
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * args.reps));
  MB_CHECK(cudaMalloc(&d_val, sizeof(std::uint32_t) * 2));
  for (const auto& c : cases) {
    MB_CHECK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    for (int n = 16; n <= 256; n += 16) {
      if (c.n_filter && c.n_filter != n) continue;
      for (int lay : {1}) {
        if (lay == 0 && !(c.kind == "f16" && c.asrc == 0)) continue;  // swizzle sensitivity: f16 SS only
        if (c.form != 0 && lay != 1) continue;  // forms are SW128 variants only
        const std::string variant = c.mode + "." + c.kind + ".cg" + std::to_string(c.cg) + "." +
                                    (c.asrc ? "tmem" : "smem") + ".n" + std::to_string(n) + (lay ? ".sw128" : ".none") +
                                    (c.ws ? ".ws" : "") +
                                    (c.form ? ".form" + std::to_string(c.form) : "");
        if (!mb_selected(args, variant)) continue;
        std::vector<double> samples;
        std::vector<float> dv;
        bool timed_out = false;
        for (int l = 0; l < args.launches; ++l) {
          cudaLaunchConfig_t cfg{};
          cfg.gridDim = dim3(c.cg, 1, 1);
          cfg.blockDim = dim3(128, 1, 1);
          cfg.dynamicSmemBytes = kSmemBytes;
          cudaLaunchAttribute attr[1];
          attr[0].id = cudaLaunchAttributeClusterDimension;
          attr[0].val.clusterDim.x = c.cg;
          attr[0].val.clusterDim.y = 1;
          attr[0].val.clusterDim.z = 1;
          cfg.attrs = attr;
          cfg.numAttrs = 1;
          MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, d_out, d_val, args.reps, n, lay, 1u, seed_base + static_cast<std::uint32_t>(l), c.fp));
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          if (mb_take_err()) {
            timed_out = true;
            break;
          }
          {
            std::vector<std::uint64_t> h(2 * static_cast<std::size_t>(args.reps));
            MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
            for (int r = 1; r < args.reps; ++r)
              samples.push_back(static_cast<double>(static_cast<std::int64_t>(h[2 * r + 1] - h[2 * r])));
          }
          std::uint32_t hv[2] = {0, 0};
          MB_CHECK(cudaMemcpy(hv, d_val, sizeof(hv), cudaMemcpyDeviceToHost));
          for (int r = 0; r < c.cg; ++r) {
            float f;
            std::memcpy(&f, &hv[r], 4);
            dv.push_back(f);
          }
        }
        if (timed_out) {
          w.fail(variant, c.u, "\"kind\":\"" + c.kind + "\"", "mbarrier wait timed out");
          continue;
        }
        const int mmas = args.reps * c.u;
        const double expect = static_cast<double>(c.kk) * mmas;
        bool ok = true;
        if (c.u > 0)
          for (float f : dv) ok = ok && f == static_cast<float>(expect);
        char extra[256];
        std::snprintf(extra, sizeof(extra), "\"d_expect\":%.1f,\"d_first\":%.1f,\"d_ok\":%s", expect,
                      dv.empty() ? -1.0 : static_cast<double>(dv[0]), ok ? "true" : "false");
        const std::string params = "\"kind\":\"" + c.kind + "\",\"mode\":\"" + c.mode + "\",\"cta_group\":" +
                                   std::to_string(c.cg) + ",\"a_src\":\"" + (c.asrc ? "tmem" : "smem") +
                                   "\",\"m\":" + std::to_string(c.cg * MECH_UMMA_ROWS_PER_CTA) + ",\"n\":" + std::to_string(n) +
                                   ",\"k\":" + std::to_string(c.kk) + ",\"layout\":\"" + (lay ? "sw128" : "none") +
                                   "\",\"form\":" + std::to_string(c.form) + ",\"operation\":\"" +
                                   (c.ws ? "WS" : "default") + "\",\"jitter_seed_base\":" + std::to_string(seed_base);
        w.row(variant, c.u, params, samples, extra);
      }
    }
  }
  return w.failures ? 1 : 0;
}

template <int KIND, int CG, int ASRC>
void cadence_cell(std::vector<UCase>& v) {
  uadd<KIND, CG, ASRC, kXw>(v, std::integer_sequence<int, 1>{});
  uadd<KIND, CG, ASRC, kCad>(v, std::integer_sequence<int, 16, 32, 64, 128>{});
}

int main(int argc, char** argv) {
  std::vector<UCase> v;
  uadd<kF16, 1, 0, kXw>(v, std::integer_sequence<int, 0>{});
  uadd<kF16, 2, 0, kXw>(v, std::integer_sequence<int, 0>{});
#define CELL(KIND)                 \
  cadence_cell<KIND, 1, 0>(v);     \
  cadence_cell<KIND, 2, 0>(v);     \
  cadence_cell<KIND, 1, 1>(v);     \
  cadence_cell<KIND, 2, 1>(v);
  CELL(kF16) CELL(kTF32) CELL(kF8F6F4) CELL(kMXF8F6F4) CELL(kMXF4) CELL(kMXF4NVF4)
#undef CELL
  return cadence_main(argc, argv, "umma_cadence", v);
}
