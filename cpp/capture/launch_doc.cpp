// _launch_doc verbatim port + Python-compatible JSON dumps.
#include "capture/launch_doc.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>

namespace cyclescope_convert::capture {

static void emit(const Json &v, int depth, std::string *out) {
  const std::string pad(depth + 1, ' ');
  const std::string pad0(depth, ' ');
  if (std::holds_alternative<std::nullptr_t>(v.v)) {
    *out += "null";
  } else if (const auto b = std::get_if<bool>(&v.v)) {
    *out += *b ? "true" : "false";
  } else if (const auto i = std::get_if<long long>(&v.v)) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lld", *i);
    *out += buf;
  } else if (const auto d = std::get_if<double>(&v.v)) {
    // Python float repr: shortest round-trip; matches %.17g trimmed
    if (*d == std::floor(*d) && std::fabs(*d) < 1e16) {
      char buf[40];
      std::snprintf(buf, sizeof buf, "%.1f", *d);
      *out += buf;
    } else {
      char buf[40];
      std::snprintf(buf, sizeof buf, "%.17g", *d);
      *out += buf;
    }
  } else if (const auto s = std::get_if<std::string>(&v.v)) {
    *out += '"';
    for (const char c : *s) {
      switch (c) {
        case '"': *out += "\\\""; break;
        case '\\': *out += "\\\\"; break;
        case '\n': *out += "\\n"; break;
        case '\t': *out += "\\t"; break;
        case '\r': *out += "\\r"; break;
        default:
          if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            *out += buf;
          } else {
            *out += c;
          }
      }
    }
    *out += '"';
  } else if (const auto a = std::get_if<Json::Arr>(&v.v)) {
    if (a->empty()) { *out += "[]"; return; }
    *out += "[\n";
    for (std::size_t i = 0; i < a->size(); ++i) {
      *out += pad;
      emit((*a)[i], depth + 1, out);
      *out += i + 1 < a->size() ? ",\n" : "\n";
    }
    *out += pad0 + "]";
  } else {
    const auto &o = std::get<Json::Obj>(v.v);
    if (o.empty()) { *out += "{}"; return; }
    *out += "{\n";
    std::size_t i = 0;
    for (const auto &[k, val] : o) {
      *out += pad;
      emit(Json(k), depth + 1, out);
      *out += ": ";
      emit(val, depth + 1, out);
      *out += ++i < o.size() ? ",\n" : "\n";
    }
    *out += pad0 + "}";
  }
}

std::string json_dumps(const Json &v) {
  std::string out;
  emit(v, 0, &out);
  return out;
}

std::string hex_sha256(const std::string &text) {
  std::uint8_t d[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(text.data()),
         text.size(), d);
  static const char *hex = "0123456789abcdef";
  std::string out;
  for (const auto b : d) {
    out += hex[b >> 4];
    out += hex[b & 15];
  }
  return out;
}
void sha256_digest(const std::string &text, std::uint8_t out[32]) {
  SHA256(reinterpret_cast<const unsigned char *>(text.data()),
         text.size(), out);
}
std::string file_sha256(const std::string &path) {
  SHA256_CTX ctx;
  SHA256_Init(&ctx);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  char buf[1 << 20];
  while (in) {
    in.read(buf, sizeof buf);
    if (in.gcount() > 0)
      SHA256_Update(&ctx, buf, static_cast<std::size_t>(in.gcount()));
  }
  std::uint8_t d[SHA256_DIGEST_LENGTH];
  SHA256_Final(d, &ctx);
  static const char *hex = "0123456789abcdef";
  std::string out;
  for (const auto b : d) { out += hex[b >> 4]; out += hex[b & 15]; }
  return out;
}

std::string launch_doc_json(const LaunchDocInput &in) {
  Json::Obj d;
  d["device"] = in.device_key;
  d["sm_arch"] = in.sm_arch.empty()
      ? (in.device_key == "b300" ? "sm_103a" : "sm_90a") : in.sm_arch;
  d["launch"] = (long long)in.launch;
  d["function"] = in.function_name;
  Json::Arr grid;
  for (const auto g : in.grid) grid.push_back(Json(g));
  d["grid"] = grid;
  Json::Arr block;
  for (const auto g : in.block) block.push_back(Json(g));
  d["block"] = block;
  Json::Arr cl;
  for (const auto c : in.dims) cl.push_back(Json((long long)c));
  d["cluster"] = cl;
  d["cluster_source"] = in.cta_info_present ? "hardware" : "attribute";
  if (in.preferred_cluster.empty()) {
    d["preferred_cluster"] = nullptr;
  } else {
    Json::Arr pc;
    for (const auto c : in.preferred_cluster) pc.push_back(Json(c));
    d["preferred_cluster"] = pc;
  }
  if (in.keep) {
    Json::Arr k;
    for (const auto c : *in.keep) k.push_back(Json((long long)c));
    d["cluster_ids"] = k;
  } else {
    d["cluster_ids"] = nullptr;
  }
  if (in.sampled_clusters) {
    Json::Arr sc;
    for (const auto &row : *in.sampled_clusters) {
      Json::Arr r;
      for (const auto x : row) r.push_back(Json(x));
      sc.push_back(r);
    }
    d["sampled_clusters"] = sc;
  } else {
    d["sampled_clusters"] = nullptr;
  }
  d["shared_static_bytes"] = in.smem_static;
  d["shared_dynamic_bytes"] = in.smem_dynamic;
  d["registers_per_thread"] = in.regs;
  d["sm_count"] = in.sm_count ? Json(*in.sm_count) : Json();
  d["l2_bytes"] = in.l2_bytes ? Json(*in.l2_bytes) : Json();
  d["occupancy_blocks_per_sm"] =
      in.occupancy_ncu ? Json(*in.occupancy_ncu)
      : in.occupancy_blocks_per_sm ? Json(*in.occupancy_blocks_per_sm)
                                 : Json();
  d["occupancy_blocks_per_sm_tracer"] =
      in.occupancy_blocks_per_sm_tracer
          ? Json(*in.occupancy_blocks_per_sm_tracer) : Json();
  d["occupancy_source"] = in.occupancy_ncu ? "ncu" : "tracer";
  d["occupancy_active_clusters"] =
      in.occupancy_active_clusters ? Json(*in.occupancy_active_clusters)
                                   : Json();
  d["function_prior_launches"] =
      in.function_prior_launches ? Json(*in.function_prior_launches)
                                 : Json();
  d["capture"] = in.case_dir;
  d["trace_sha256"] = in.trace_sha;
  d["dropped_polls"] = (long long)in.dropped_polls;
  {
    Json::Obj m;
    for (const auto &[k, v] : in.missing) m[k] = v;
    d["missing_facts"] = m;
    Json::Obj u;
    for (const auto &[k, v] : in.unsupported) u[k] = v;
    d["unsupported"] = u;
    Json::Obj pl;
    for (const auto &[k, v] : in.placements) {
      Json::Obj e;
      for (const auto &[ek, ev] : v) e[ek] = ev;
      pl[k] = e;
    }
    d["placements"] = pl;
  }
  if (in.sm_topology_path) d["sm_topology"] = *in.sm_topology_path;
  if (in.reference_sm_clock_khz)
    d["reference_sm_clock_khz"] = (long long)*in.reference_sm_clock_khz;
  if (in.reference_cache_control)
    d["reference_cache_control"] = (long long)*in.reference_cache_control;
  if (in.reference_source) d["reference_source"] = *in.reference_source;
  return json_dumps(Json(d));
}

}  // namespace cyclescope_convert::capture
