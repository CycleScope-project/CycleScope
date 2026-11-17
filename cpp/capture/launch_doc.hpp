// launch.json document + META digest (capture_pipeline._launch_doc,
// _file_sha256, sha256(trace_sha|launch|cl) digest).
// JSON serialization replicates Python json.dumps(sort_keys=True,
// indent=1) byte-for-byte for the value types we emit.
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyclescope_convert::capture {

struct Json {
  using Arr = std::vector<Json>;
  using Obj = std::map<std::string, Json>;
  std::variant<std::nullptr_t, bool, long long, double, std::string,
               Arr, Obj> v;
  Json() : v(nullptr) {}
  Json(std::nullptr_t) : v(nullptr) {}
  Json(bool b) : v(b) {}
  Json(long long x) : v(x) {}
  Json(int x) : v((long long)x) {}
  Json(std::uint32_t x) : v((long long)x) {}
  Json(std::int64_t x) : v((long long)x) {}
  Json(std::uint64_t x) : v((long long)x) {}
  Json(double x) : v(x) {}
  Json(const char *s) : v(std::string(s)) {}
  Json(std::string s) : v(std::move(s)) {}
  Json(Arr a) : v(std::move(a)) {}
  Json(Obj o) : v(std::move(o)) {}
  static Json arr(std::initializer_list<Json> l) { return Json(Arr(l)); }
};

// json.dumps(value, sort_keys=True, indent=1) exact bytes
std::string json_dumps(const Json &v);

// sha256 (openssl) helpers
std::string file_sha256(const std::string &path);
std::string hex_sha256(const std::string &text);
void sha256_digest(const std::string &text, std::uint8_t out[32]);

// launch.json fields mirrored from Python _launch_doc
struct LaunchDocInput {
  std::string case_dir, device_key;
  int launch = 0;
  // launches.jsonl row
  std::string function_name;
  std::vector<std::int64_t> grid, block;
  std::vector<int> dims;                    // effective_cluster
  std::optional<std::vector<int>> keep;     // cluster filter
  bool cta_info_present = false;            // kind-3 records seen
  std::vector<std::int64_t> preferred_cluster;  // launches.jsonl
  std::optional<std::vector<std::vector<int64_t>>> sampled_clusters;
  std::int64_t smem_static = 0, smem_dynamic = 0, regs = 0;
  std::optional<std::int64_t> occupancy_blocks_per_sm,
      occupancy_active_clusters, function_prior_launches;
  // device.json
  std::optional<std::int64_t> sm_count, l2_bytes;
  std::optional<std::int64_t> occupancy_ncu;  // min ncu limit_* override
  std::optional<std::int64_t> occupancy_blocks_per_sm_tracer;
  std::string sm_arch;
  std::string trace_sha;
  std::uint64_t dropped_polls = 0;
  std::map<std::string, std::int64_t> missing, unsupported;
  // placements: cta_linear -> {sm_id, warp_slot_base, source}
  std::map<std::string, std::map<std::string, std::int64_t>> placements;
  std::optional<std::string> sm_topology_path;
  // reference.tsv facts (sort_keys JSON; position-free)
  std::optional<std::int64_t> reference_sm_clock_khz;
  std::optional<int> reference_cache_control;
  std::optional<std::string> reference_source;
  bool occupancy_block_has = false;   // whether row had occupancy_max_blocks_per_sm
};

std::string launch_doc_json(const LaunchDocInput &in);

}  // namespace cyclescope_convert::capture
