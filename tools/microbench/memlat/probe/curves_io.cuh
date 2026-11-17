// JSONL writer for SM103 mechanism curves. Host only.
#pragma once

#include "common.cuh"

#include <cmath>

struct CurvePoint {
  const char* family = "";
  const char* quantity = "";  // latency | throughput | capacity
  int cta_group = 1;
  const char* x_name = "";
  double x = 0;
  const char* y_name = "";
  const char* y_unit = "cycles";
  std::vector<double> samples;
  int holdout = 0;
  std::string extra_json;  // already-quoted fields, no wrapping braces
  std::string error;
};

inline double mean_of(const std::vector<double>& xs) {
  if (xs.empty()) return 0.0;
  double s = 0;
  for (double v : xs) s += v;
  return s / static_cast<double>(xs.size());
}

inline double stddev_of(const std::vector<double>& xs) {
  if (xs.size() < 2) return 0.0;
  const double m = mean_of(xs);
  double s = 0;
  for (double v : xs) s += (v - m) * (v - m);
  return std::sqrt(s / static_cast<double>(xs.size() - 1));
}

inline void append_jsonl(const char* path, const CurvePoint& p) {
  FILE* f = std::fopen(path, "a");
  if (!f) {
    std::fprintf(stderr, "cannot append %s\n", path);
    std::exit(1);
  }
  const double med = median_of(p.samples);
  const double p10 = percentile(p.samples, 0.10);
  const double p90 = percentile(p.samples, 0.90);
  const double sd = stddev_of(p.samples);
  const double cv = (med == 0.0) ? 0.0 : sd / std::fabs(med);
  std::fprintf(f,
               "{\"schema\":\"blackwell.mechanism_curve_point.v1\","
               "\"family\":\"%s\",\"quantity\":\"%s\",\"cta_group\":%d,"
               "\"x_name\":\"%s\",\"x\":%.6f,\"y_name\":\"%s\",\"y_unit\":\"%s\","
               "\"median\":%.6f,\"p10\":%.6f,\"p90\":%.6f,\"mean\":%.6f,"
               "\"stddev\":%.6f,\"cv\":%.6f,\"n\":%zu,\"holdout\":%s,"
               "\"samples\":",
               p.family, p.quantity, p.cta_group, p.x_name, p.x, p.y_name, p.y_unit,
               med, p10, p90, mean_of(p.samples), sd, cv, p.samples.size(),
               p.holdout ? "true" : "false");
  print_json_array(f, p.samples);
  if (!p.extra_json.empty()) std::fprintf(f, ",%s", p.extra_json.c_str());
  if (!p.error.empty()) {
    std::fprintf(f, ",\"error\":\"");
    json_escape(f, p.error.c_str());
    std::fprintf(f, "\"");
  }
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("  %s %s x=%.0f med=%.3f n=%zu holdout=%d%s\n", p.family, p.quantity,
              p.x, med, p.samples.size(), p.holdout,
              p.error.empty() ? "" : " ERR");
}
