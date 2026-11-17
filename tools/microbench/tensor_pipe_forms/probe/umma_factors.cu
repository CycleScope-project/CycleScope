#include "../umma_factor_impl.cuh"
#include "umma_factor_forms.hpp"

#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef UMMA_FACTOR_KIND_FILTER
#define UMMA_FACTOR_KIND_FILTER -1
#endif
#ifndef UMMA_FACTOR_CG_FILTER
#define UMMA_FACTOR_CG_FILTER 0
#endif
#ifndef UMMA_FACTOR_INCLUDE_WS
#define UMMA_FACTOR_INCLUDE_WS 1
#endif
#ifndef UMMA_FACTOR_INCLUDE_SPARSE
#define UMMA_FACTOR_INCLUDE_SPARSE 1
#endif

template <int KIND, int CG, int ASRC, int MODE, int U, int M_ROWS,
          bool WS, bool REUSE, bool SPARSE, int ACC,
          int AROT = 0, int BROT = 0, int GRP = 0,
          int ISSUERS = 1, int KALU = 0, int ELECTED = 0>
void append_case_t(std::vector<FactorCase>& out, const FactorForm& form) {
  FactorParams params{
      form.idesc, form.smem_a_control, form.smem_b_control,
      form.start_a, form.start_b, form.a_one, form.b_one, form.scale_one,
      form.a_bits, form.b_bits, form.scale_bits, form.sf_id_a, form.sf_id_b,
      form.d_f16, form.sign, form.reuse, form.lbo_abs_a, form.lbo_abs_b,
      form.b_zero, form.sparse, form.sparsity_version, form.overlap,
      form.overlap_bytes,
  };
  out.push_back({
      &form, MODE == kXw ? "xw" : "cad", U,
      &k_factor<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, ACC, AROT, BROT, GRP, ISSUERS, KALU, ELECTED>,
      &k_factor<KIND, CG, ASRC, kXw, 1, M_ROWS, WS, REUSE, SPARSE, ACC, AROT, BROT, GRP, ISSUERS, KALU, ELECTED>,
      form.k,
  });
}

template <int KIND, int CG, int ASRC, int MODE, int U, int M_ROWS,
          bool WS, bool REUSE, bool SPARSE, int ELECTED>
void append_case_d(std::vector<FactorCase>& out, const FactorForm& form) {
  if (form.grp == 8 && form.acc == 8 && form.arot && form.brot) {
    append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 1, 8, 1, 0, ELECTED>(out, form);
    return;
  }
  if (form.arot && form.brot && form.kalu) {
    switch (form.kalu) {
      case 2: append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 1, 0, 1, 2, ELECTED>(out, form); break;
      case 4: append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 1, 0, 1, 4, ELECTED>(out, form); break;
      case 8: append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 1, 0, 1, 8, ELECTED>(out, form); break;
      default: append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 1, 0, 1, 16, ELECTED>(out, form); break;
    }
    return;
  }
  if (form.arot && form.brot && form.issuers > 1) {
    if (form.issuers == 4)
      append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 1, 0, 4, 0, ELECTED>(out, form);
    else
      append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 1, 0, 2, 0, ELECTED>(out, form);
    return;
  }
  if (form.arot && form.brot) {
    append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 1, 0, 1, 0, ELECTED>(out, form);
    return;
  }
  if (form.arot) {
    append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 1, 0, 0, 1, 0, ELECTED>(out, form);
    return;
  }
  if (form.brot) {
    append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 0, 1, 0, 1, 0, ELECTED>(out, form);
    return;
  }
  switch (form.acc) {
    case 2: append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 2, 0, 0, 0, 1, 0, ELECTED>(out, form); break;
    case 4: append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 4, 0, 0, 0, 1, 0, ELECTED>(out, form); break;
    case 8: append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 8, 0, 0, 0, 1, 0, ELECTED>(out, form); break;
    default: append_case_t<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 1, 0, 0, 0, 1, 0, ELECTED>(out, form); break;
  }
}

template <int KIND, int CG, int ASRC, int MODE, int U, int M_ROWS,
          bool WS, bool REUSE, bool SPARSE>
void append_case(std::vector<FactorCase>& out, const FactorForm& form) {
  if (form.elected)
    append_case_d<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 1>(out, form);
  else
    append_case_d<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, 0>(out, form);
}

template <int KIND, int CG, int ASRC, int M_ROWS, bool WS, bool REUSE,
          bool SPARSE>
void append_form(std::vector<FactorCase>& out, const FactorForm& form) {
  append_case<KIND, CG, ASRC, kXw, 0, M_ROWS, WS, REUSE, SPARSE>(out, form);
  append_case<KIND, CG, ASRC, kXw, 1, M_ROWS, WS, REUSE, SPARSE>(out, form);
  append_case<KIND, CG, ASRC, kCad, 32, M_ROWS, WS, REUSE, SPARSE>(out, form);
  append_case<KIND, CG, ASRC, kCad, 64, M_ROWS, WS, REUSE, SPARSE>(out, form);
  append_case<KIND, CG, ASRC, kCad, 128, M_ROWS, WS, REUSE, SPARSE>(out, form);
}

template <int KIND, int CG, int ASRC, bool WS, bool REUSE, bool SPARSE>
void append_factor_rows(std::vector<FactorCase>& out, const FactorForm& f,
                        int rows) {
#if UMMA_FACTOR_INCLUDE_WS
  if constexpr (WS) {
    if (rows == 32)
      append_form<KIND, CG, ASRC, 32, WS, REUSE, SPARSE>(out, f);
  }
#endif
  if (rows == 64)
    append_form<KIND, CG, ASRC, 64, WS, REUSE, SPARSE>(out, f);
  else if (rows == 128)
    append_form<KIND, CG, ASRC, 128, WS, REUSE, SPARSE>(out, f);
}

template <int KIND, bool WS, bool REUSE, bool SPARSE>
void append_factor_cell(std::vector<FactorCase>& out, const FactorForm& f) {
  const int rows = f.m / f.cta_group;
  if constexpr (WS) {
#if UMMA_FACTOR_CG_FILTER != 2
    if (f.cta_group == 1) {
      if (f.a_src == 0)
        append_factor_rows<KIND, 1, 0, WS, REUSE, SPARSE>(out, f, rows);
      else
        append_factor_rows<KIND, 1, 1, WS, REUSE, SPARSE>(out, f, rows);
    }
#endif
  } else {
#if UMMA_FACTOR_CG_FILTER == 1
  if (f.cta_group != 1) return;
  if (f.a_src == 0)
    append_factor_rows<KIND, 1, 0, WS, REUSE, SPARSE>(out, f, rows);
  else
    append_factor_rows<KIND, 1, 1, WS, REUSE, SPARSE>(out, f, rows);
#elif UMMA_FACTOR_CG_FILTER == 2
  if (f.cta_group != 2) return;
  if (f.a_src == 0)
    append_factor_rows<KIND, 2, 0, WS, REUSE, SPARSE>(out, f, rows);
  else
    append_factor_rows<KIND, 2, 1, WS, REUSE, SPARSE>(out, f, rows);
#else
  if (f.cta_group == 1 && f.a_src == 0)
    append_factor_rows<KIND, 1, 0, WS, REUSE, SPARSE>(out, f, rows);
  else if (f.cta_group == 1 && f.a_src == 1)
    append_factor_rows<KIND, 1, 1, WS, REUSE, SPARSE>(out, f, rows);
  else if (f.cta_group == 2 && f.a_src == 0)
    append_factor_rows<KIND, 2, 0, WS, REUSE, SPARSE>(out, f, rows);
  else if (f.cta_group == 2 && f.a_src == 1)
    append_factor_rows<KIND, 2, 1, WS, REUSE, SPARSE>(out, f, rows);
#endif
  }
}

template <int KIND>
void append_factor_kind(std::vector<FactorCase>& out, const FactorForm& f) {
  if (f.ws && KIND != kF16 && KIND != kTF32 && KIND != kF8F6F4) return;
  if (f.ws && f.cta_group != 1) return;
  if (f.reuse) return;
  if (f.sparse) {
#if UMMA_FACTOR_INCLUDE_SPARSE
    if (f.ws) {
#if UMMA_FACTOR_INCLUDE_WS
      append_factor_cell<KIND, true, false, true>(out, f);
#endif
    } else {
      append_factor_cell<KIND, false, false, true>(out, f);
    }
#endif
  } else if (f.ws) {
#if UMMA_FACTOR_INCLUDE_WS
    append_factor_cell<KIND, true, false, false>(out, f);
#endif
  } else {
    append_factor_cell<KIND, false, false, false>(out, f);
  }
}

void append_factor_form(std::vector<FactorCase>& out, const FactorForm& f) {
#if UMMA_FACTOR_KIND_FILTER >= 0
  if (f.kind_id != UMMA_FACTOR_KIND_FILTER ||
      f.cta_group != UMMA_FACTOR_CG_FILTER) {
#if UMMA_FACTOR_KIND_FILTER != 0 || UMMA_FACTOR_CG_FILTER != 1
    if (f.kind_id == 0 && f.cta_group == 1 &&
        (std::strcmp(f.cell_id, "C2") == 0 ||
         std::strcmp(f.cell_id, "C3") == 0)) {
      append_form<kF16, 1, 0, 128, false, false, false>(out, f);
    }
#endif
    return;
  }
#endif
  switch (f.kind_id) {
#if UMMA_FACTOR_KIND_FILTER < 0 || UMMA_FACTOR_KIND_FILTER == 0
    case 0: append_factor_kind<kF16>(out, f); break;
#endif
#if UMMA_FACTOR_KIND_FILTER < 0 || UMMA_FACTOR_KIND_FILTER == 1
    case 1: append_factor_kind<kTF32>(out, f); break;
#endif
#if UMMA_FACTOR_KIND_FILTER < 0 || UMMA_FACTOR_KIND_FILTER == 2
    case 2: append_factor_kind<kF8F6F4>(out, f); break;
#endif
#if UMMA_FACTOR_KIND_FILTER < 0 || UMMA_FACTOR_KIND_FILTER == 3
    case 3: append_factor_kind<kMXF8F6F4>(out, f); break;
#endif
#if UMMA_FACTOR_KIND_FILTER < 0 || UMMA_FACTOR_KIND_FILTER == 4
    case 4: append_factor_kind<kMXF4>(out, f); break;
#endif
#if UMMA_FACTOR_KIND_FILTER < 0 || UMMA_FACTOR_KIND_FILTER == 5
    case 5: append_factor_kind<kMXF4NVF4>(out, f); break;
#endif
    default: break;
  }
}

std::string variant_name(const FactorCase& c) {
  const auto& f = *c.form;
  std::string name = std::string(c.mode) + "." + f.kind + ".cg" +
      std::to_string(f.cta_group) + "." + (f.a_src ? "tmem" : "smem") +
      ".m" + std::to_string(f.m) + ".n" + std::to_string(f.n) +
      "." + f.operation + ".factor." + f.form_id + ".u" +
      std::to_string(c.u) + (f.acc != 1 ? ".acc" + std::to_string(f.acc) : "") +
      (f.grp == 8 ? ".grp8" : f.brot ? ".arotbrot" : f.arot ? ".arot" : "") +
      (f.issuers > 1 ? ".w" + std::to_string(f.issuers) : "") +
      (f.kalu ? ".k" + std::to_string(f.kalu) : "") +
      (f.elected ? ".elect" : "");
  return name;
}

bool launch_factor(FactorKernel kernel, int cg, int n, int reps,
                   std::uint32_t seed, FactorParams params,
                   std::uint64_t* out, std::uint32_t* dval) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(cg, 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = kFactorSmemBytes;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cg;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  MB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kFactorSmemBytes));
  MB_CHECK(cudaLaunchKernelEx(&cfg, kernel, out, dval, reps, n, seed, params));
  MB_CHECK(cudaGetLastError());
  MB_CHECK(cudaDeviceSynchronize());
  return !mb_take_err();
}

float decode_d(std::uint32_t raw, bool f16) {
  if (!f16) {
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
  }
  const std::uint16_t low = static_cast<std::uint16_t>(raw & 0xffffu);
  __half h;
  std::memcpy(&h, &low, sizeof(low));
  return __half2float(h);
}

int main(int argc, char** argv) {
  std::uint32_t seed_base = 0;
  int forwarded = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--seed-base") != 0) {
      argv[forwarded++] = argv[i];
      continue;
    }
    if (++i == argc) return 2;
    char* end = nullptr;
    const unsigned long value = std::strtoul(argv[i], &end, 10);
    if (argv[i][0] < '0' || argv[i][0] > '9' || *end ||
        value > 0xfffffffful) return 2;
    seed_base = static_cast<std::uint32_t>(value);
  }
  argc = forwarded;
  argv[argc] = nullptr;
  const MbArgs args = mb_parse(argc, argv);
  if (seed_base > 0xffffffffu - static_cast<std::uint32_t>(args.launches - 1))
    return 2;

  std::vector<FactorCase> cases;
  for (const auto& form : kFactorForms) append_factor_form(cases, form);
  MbWriter writer(args, "umma_factors");
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * args.reps));
  MB_CHECK(cudaMalloc(
      &d_val, sizeof(std::uint32_t) * 2 * kFactorTelemetryWords));

  std::unordered_map<std::string, std::pair<bool, float>> checks;
  for (const auto& c : cases) {
    const auto& f = *c.form;
    const auto name = variant_name(c);
    if (!mb_selected(args, name)) continue;
    if (!checks.count(f.form_id)) {
      FactorParams params{
          f.idesc, f.smem_a_control, f.smem_b_control,
          f.start_a, f.start_b, f.a_one, f.b_one, f.scale_one,
          f.a_bits, f.b_bits, f.scale_bits, f.sf_id_a, f.sf_id_b,
          f.d_f16, f.sign, f.reuse, f.lbo_abs_a, f.lbo_abs_b,
          f.b_zero, f.sparse, f.sparsity_version, f.overlap,
          f.overlap_bytes,
      };
      if (!launch_factor(c.check_kernel, f.cta_group, f.n, 1,
                         seed_base, params, d_out, d_val)) {
        checks[f.form_id] = {false, -1.0f};
      } else {
        std::uint32_t raw[2 * kFactorTelemetryWords] = {};
        MB_CHECK(cudaMemcpy(raw, d_val, sizeof(raw), cudaMemcpyDeviceToHost));
        const float expected = f.b_zero ? 0.0f :
            static_cast<float>(f.sign * (f.sparse ? f.k / 2 : f.k));
        bool ok = true;
        float first = -1.0f;
        for (int rank = 0; rank < f.cta_group; ++rank) {
          const float actual = decode_d(
              raw[rank * kFactorTelemetryWords], f.d_f16 != 0);
          if (rank == 0) first = actual;
          ok = ok && actual == expected;
        }
        checks[f.form_id] = {ok, first};
      }
    }

    std::vector<double> samples;
    bool failed = false;
    bool timed_d_ok = true;
    double timed_d_first = 0.0;
    std::uint32_t timed_idesc = 0;
    std::uint64_t timed_smem_a = 0, timed_smem_b = 0;
    for (int launch = 0; launch < args.launches; ++launch) {
      FactorParams params{
          f.idesc, f.smem_a_control, f.smem_b_control,
          f.start_a, f.start_b, f.a_one, f.b_one, f.scale_one,
          f.a_bits, f.b_bits, f.scale_bits, f.sf_id_a, f.sf_id_b,
          f.d_f16, f.sign, f.reuse, f.lbo_abs_a, f.lbo_abs_b,
          f.b_zero, f.sparse, f.sparsity_version, f.overlap,
          f.overlap_bytes,
      };
      if (!launch_factor(c.kernel, f.cta_group, f.n, args.reps,
                         seed_base + static_cast<std::uint32_t>(launch),
                         params, d_out, d_val)) {
        failed = true;
        break;
      }
      std::vector<std::uint64_t> raw(
          2 * static_cast<std::size_t>(args.reps));
      MB_CHECK(cudaMemcpy(raw.data(), d_out, raw.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
      std::uint32_t telemetry[2 * kFactorTelemetryWords] = {};
      MB_CHECK(cudaMemcpy(telemetry, d_val, sizeof(telemetry),
                          cudaMemcpyDeviceToHost));
      const bool check_timed_d = c.mode == std::string("xw") && c.u == 1;
      const double timed_expected = f.b_zero ? 0.0 :
          static_cast<double>(
              f.sign * (f.sparse ? f.k / 2 : f.k) * c.u * args.reps);
      for (int rank = 0; rank < f.cta_group; ++rank) {
        const auto index = rank * kFactorTelemetryWords;
        const double actual = decode_d(telemetry[index], f.d_f16 != 0);
        if (rank == 0) {
          timed_d_first = actual;
          timed_idesc = telemetry[index + 1];
          timed_smem_a = static_cast<std::uint64_t>(telemetry[index + 2]) |
              (static_cast<std::uint64_t>(telemetry[index + 3]) << 32);
          timed_smem_b = static_cast<std::uint64_t>(telemetry[index + 4]) |
              (static_cast<std::uint64_t>(telemetry[index + 5]) << 32);
        }
        if (check_timed_d)
          timed_d_ok = timed_d_ok && actual == timed_expected;
      }
      for (int rep = 1; rep < args.reps; ++rep) {
        const auto value = static_cast<std::uint64_t>(
            raw[2 * rep + 1] - raw[2 * rep]);
        samples.push_back(static_cast<double>(value));
      }
    }
    if (failed) {
      writer.fail(name, c.u, "\"factor_id\":\"" + std::string(f.form_id) + "\"",
                  "mbarrier wait timed out");
      continue;
    }
    const auto [d_ok, d_first] = checks[f.form_id];
    const double d_expected = f.b_zero ? 0.0 :
        static_cast<double>(f.sign * (f.sparse ? f.k / 2 : f.k));
    char extra[768];
    std::snprintf(extra, sizeof(extra),
                  "\"d_expect\":%.1f,\"d_first\":%.1f,\"d_check_u\":1,"
                  "\"d_check_reps\":1,\"d_ok\":%s,"
                  "\"d_timed_expect\":%.1f,\"d_timed_first\":%.1f,"
                  "\"d_timed_ok\":%s,\"timed_idesc\":\"0x%08x\","
                  "\"timed_smem_a\":\"0x%016llx\","
                  "\"timed_smem_b\":\"0x%016llx\"",
                  d_expected, static_cast<double>(d_first),
                  d_ok ? "true" : "false",
                  f.b_zero ? 0.0 :
                      static_cast<double>(
                          f.sign * (f.sparse ? f.k / 2 : f.k) *
                          c.u * args.reps),
                  timed_d_first,
                  (c.mode != std::string("xw") || c.u != 1)
                      ? "null" : timed_d_ok ? "true" : "false",
                  timed_idesc,
                  static_cast<unsigned long long>(timed_smem_a),
                  static_cast<unsigned long long>(timed_smem_b));
    char descriptor_params[320];
    std::snprintf(
        descriptor_params, sizeof(descriptor_params),
        ",\"idesc_input\":\"0x%08x\",\"smem_b_control\":\"0x%016llx\","
        "\"start_b\":%u,\"sparse\":%s,\"sparsity_version\":%u,"
        "\"overlap\":%u,\"overlap_bytes\":%u",
        f.idesc, static_cast<unsigned long long>(f.smem_b_control),
        f.start_b, f.sparse ? "true" : "false", f.sparsity_version,
        f.overlap, f.overlap_bytes);
    const std::string params =
        "\"factor_id\":\"" + std::string(f.form_id) +
        "\",\"cell_id\":\"" + f.cell_id +
        "\",\"factor\":\"" + f.factor +
        "\",\"level\":\"" + f.level +
        "\",\"kind\":\"" + f.kind +
        "\",\"operation\":\"" + f.operation +
        "\",\"mode\":\"" + c.mode +
        "\",\"cta_group\":" + std::to_string(f.cta_group) +
        ",\"a_src\":\"" + (f.a_src ? "tmem" : "smem") +
        "\",\"b_zero\":" + (f.b_zero ? "true" : "false") +
        ",\"m\":" + std::to_string(f.m) +
        ",\"n\":" + std::to_string(f.n) +
        ",\"k\":" + std::to_string(f.k) +
        ",\"acc\":" + std::to_string(f.acc) +
        ",\"arot\":" + std::to_string(f.arot) +
        ",\"brot\":" + std::to_string(f.brot) +
        ",\"grp\":" + std::to_string(f.grp) +
        ",\"issuers\":" + std::to_string(f.issuers) +
        ",\"kalu\":" + std::to_string(f.kalu) +
        ",\"elected\":" + std::to_string(f.elected) +
        ",\"u\":" + std::to_string(c.u) + descriptor_params;
    writer.row(name, c.u, params, samples, extra);
  }
  return writer.failures ? 1 : 0;
}
