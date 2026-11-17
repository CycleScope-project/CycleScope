// cyclescope-capture-facts — minimal path: statics.gsc + per-cluster
// <cl:06d>.gsc with CTAS/WARP/DYNI filled; MREF/OPS/GSYN rows are
// emitted empty (DYN ops/gsync indices = N32) pending the operand /
// memory / gsync ports.
//
// Mirrors python/cyclescope/convert/capture_pipeline.py convert_capture:
// _cluster_dims, _cluster_linear, _cluster_filter_spec, rec_keep,
// static_index assignment, cluster_rank_list (z,y,x) ordering,
// ctas_f / begin_warp / DYN field order.
#include "capture/facts_writer.hpp"
#include "capture/poll_folding.hpp"
#include "capture/operand_facts.hpp"
#include "capture/control_flow.hpp"
#include "capture/memory_refs.hpp"
#include "capture/tensor_map.hpp"
#include "capture/global_sync.hpp"
#include "capture/launch_doc.hpp"
#include "capture/static_code.hpp"
#include "capture/trace_reader.hpp"
#include "convert/convert_rules.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <regex>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace cyclescope_convert;
using namespace cyclescope_convert::capture;

namespace {

[[noreturn]] void die(const std::string &m) {
  std::fprintf(stderr, "cyclescope-capture-facts: %s\n", m.c_str());
  std::exit(1);
}
[[noreturn]] void missing(const std::string &w, const std::string &where) {
  std::fprintf(stderr, "missing_fact:%s (%s)\n", w.c_str(),
               where.c_str());
  std::exit(1);
}

std::string json_str(const std::string &line, const char *k) {
  const std::string pat = std::string("\"") + k + "\"";
  auto p = line.find(pat);
  if (p == std::string::npos) return "";
  p = line.find(':', p + pat.size());
  if (p == std::string::npos) return "";
  ++p;
  while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
  if (p >= line.size()) return "";
  if (line[p] == '"') {
    const auto e = line.find('"', p + 1);
    return line.substr(p + 1, e - p - 1);
  }
  std::size_t e = p;
  while (e < line.size() &&
         (std::isdigit(static_cast<unsigned char>(line[e])) ||
          line[e] == '-' || line[e] == 'x' || line[e] == 'X' ||
          std::isxdigit(static_cast<unsigned char>(line[e]))))
    ++e;
  return line.substr(p, e - p);
}
// "grid": [a,b,c] or "effective_cluster": [a,b,c]
std::vector<int> json_arr3(const std::string &line, const char *k) {
  const std::string pat = std::string("\"") + k + "\"";
  auto p = line.find(pat);
  if (p == std::string::npos) return {};
  p = line.find('[', p);
  if (p == std::string::npos) return {};
  const auto e = line.find(']', p);
  std::vector<int> out;
  std::string num;
  for (std::size_t i = p + 1; i < e; ++i) {
    if (std::isdigit(static_cast<unsigned char>(line[i])) ||
        line[i] == '-') {
      num += line[i];
    } else if (!num.empty()) {
      out.push_back(std::stoi(num));
      num.clear();
    }
  }
  if (!num.empty()) out.push_back(std::stoi(num));
  return out;
}

long long json_num(const std::string &line, const char *k,
                   long long deflt) {
  const std::string t = json_str(line, k);
  return t.empty() ? deflt : std::stoll(t);
}

struct LaunchRow {
  std::vector<int> grid{1, 1, 1};
  std::vector<int> dims{1, 1, 1};
  std::vector<std::vector<int>> sampled;   // [[x,y,z],...]
  std::string raw;                          // launches.jsonl line
  bool found = false;
};

// ncu csv for the traced launch: the capture's own census
// (<case>/ncu.csv, old flow) or the reference run's csv next to the
// capture (<captures>/reference/<case>/<case>/ncu_r*.csv, capture_batch
// layout) -- same source order as capture_pipeline._ncu_occupancy.
// A capture root may be a farm of per-case symlinks: the reference/
// sibling lives next to the resolved real case dir, so the canonical
// path is tried as a second candidate.
fs::path ncu_csv_in(const fs::path &dir) {
  const fs::path own = dir / "ncu.csv";
  if (fs::is_regular_file(own)) return own;
  const auto name = dir.filename();
  const fs::path ref = dir.parent_path() / "reference" / name / name;
  std::vector<fs::path> found;
  std::error_code ec;
  for (const auto &e : fs::directory_iterator(ref, ec)) {
    const auto fn = e.path().filename().string();
    if (e.is_regular_file() && fn.rfind("ncu_r", 0) == 0 &&
        fn.size() > 4 && fn.compare(fn.size() - 4, 4, ".csv") == 0)
      found.push_back(e.path());
  }
  if (found.empty()) return {};
  std::sort(found.begin(), found.end());
  return found.front();
}

fs::path ncu_csv_path(const fs::path &case_dir) {
  if (const fs::path p = ncu_csv_in(case_dir); !p.empty()) return p;
  std::error_code ec;
  const fs::path canon = fs::weakly_canonical(case_dir, ec);
  if (!ec && canon != case_dir) {
    if (const fs::path p = ncu_csv_in(canon); !p.empty()) return p;
  }
  return {};
}

// ncu csv row for the traced launch (kernel name + occurrence +
// block_size); returns min(launch__occupancy_limit_*) or -1.
std::int64_t ncu_occupancy(const fs::path &case_dir,
                         const LaunchRow &lr) {
  const fs::path p = ncu_csv_path(case_dir);
  if (p.empty()) return -1;
  std::ifstream in(p);
  if (!in) return -1;
  std::string line;
  std::vector<std::string> cols;
  while (std::getline(in, line)) {
    if (line.rfind("\"ID\"", 0) != 0) continue;
    std::istringstream hs(line);
    // simple csv: quoted fields, no embedded commas in headers
    for (std::string t; std::getline(hs, t, ',');) {
      if (t.size() >= 2 && t.front() == '"' && t.back() == '"')
        t = t.substr(1, t.size() - 2);
      cols.push_back(t);
    }
    break;
  }
  if (cols.empty()) return -1;
  auto col = [&](const char *k) -> int {
    for (size_t i = 0; i < cols.size(); ++i)
      if (cols[i] == k) return (int)i;
    return -1;
  };
  const int i_fn = col("Kernel Name"), i_blk = col("launch__block_size");
  const int i_reg = col("launch__occupancy_limit_registers"),
            i_smem = col("launch__occupancy_limit_shared_mem"),
            i_blk2 = col("launch__occupancy_limit_blocks"),
            i_warps = col("launch__occupancy_limit_warps");
  if (i_fn < 0) return -1;
  const std::string fname = json_str(lr.raw, "mangled");
  std::vector<std::vector<std::string>> same;
  std::string row_line;
  while (std::getline(in, row_line)) {
    if (row_line.empty() || row_line[0] != '"') continue;
    std::vector<std::string> f;
    std::string cur; bool inq = false;
    for (char ch : row_line) {
      if (ch == '"') inq = !inq;
      else if (ch == ',' && !inq) { f.push_back(cur); cur.clear(); }
      else cur += ch;
    }
    f.push_back(cur);
    if ((int)f.size() > i_fn && f[i_fn] == fname) same.push_back(f);
  }
  const long long nth = json_num(lr.raw, "function_prior_launches", 0);
  if (nth < 0 || (size_t)nth >= same.size()) return -1;
  const auto &cand = same[(size_t)nth];
  auto num = [&](int i) -> long long {
    if (i < 0 || (size_t)i >= cand.size() || cand[i].empty()) return -1;
    try { return std::stoll(cand[i]); }
    catch (...) { return -1; }
  };
  const long long blk_ncu = num(i_blk);
  const auto blk = json_arr3(lr.raw, "block");
  const long long blk_want =
      blk.size() == 3 ? (long long)blk[0] * blk[1] * blk[2] : -1;
  if (blk_ncu > 0 && blk_want > 0 && blk_ncu != blk_want) return -1;
  long long best = -1;
  for (int i : {i_reg, i_smem, i_blk2, i_warps}) {
    const long long v = num(i);
    if (v > 0 && (best < 0 || v < best)) best = v;
  }
  return best;
}

LaunchRow find_launch(const fs::path &case_dir, int launch) {
  const fs::path lj = case_dir / "launches.jsonl";
  if (!fs::is_regular_file(lj)) missing("launches.jsonl", case_dir.string());
  std::ifstream fh(lj);
  std::string line;
  LaunchRow row;
  while (std::getline(fh, line)) {
    if (line.empty()) continue;
    const std::string l = json_str(line, "launch");
    if (l.empty() || std::stoi(l) != launch) continue;
    row.found = true;
    row.raw = line;
    auto g = json_arr3(line, "grid");
    if (!g.empty()) row.grid = g;
    auto d = json_arr3(line, "effective_cluster");
    if (d.empty()) d = json_arr3(line, "cluster");
    if (!d.empty()) row.dims = d;
    // sampled_clusters: [{x,y,z},...] or [[x,y,z],...]
    const std::string key = "\"sampled_clusters\"";
    auto p = line.find(key);
    if (p != std::string::npos) {
      p = line.find('[', p);
      if (p != std::string::npos) {
        const std::string rest = line.substr(p);
        std::size_t i = 0, depth = 0;
        while (i < rest.size()) {
          const char c = rest[i];
          if (c == '{') {              // dict form
            const auto e = rest.find('}', i);
            std::vector<int> v;
            for (const char *k : {"x", "y", "z"}) {
              const std::string j = json_str(
                  rest.substr(i, e - i), (std::string("\"") + k + "\"").c_str());
              v.push_back(j.empty() ? 0 : std::stoi(j));
            }
            row.sampled.push_back(v);
            i = e;
          } else if (c == '[' && depth > 0) {
            const auto e = rest.find(']', i);
            std::vector<int> v;
            std::string num;
            for (std::size_t j = i + 1; j < e; ++j) {
              if (std::isdigit(static_cast<unsigned char>(rest[j]))) {
                num += rest[j];
              } else if (!num.empty()) {
                v.push_back(std::stoi(num));
                num.clear();
              }
            }
            if (!num.empty()) v.push_back(std::stoi(num));
            if (v.size() >= 3) row.sampled.push_back(v);
            i = e;
          }
          if (c == '[') ++depth;
          if (c == ']' && --depth == 0) break;
          ++i;
        }
      }
    }
    break;
  }
  return row;
}

int cdiv(int a, int b) { return b > 0 ? (a + b - 1) / b : a; }

int cta_linear(int x, int y, int z, const std::vector<int> &grid) {
  const int gx = std::max(1, grid[0]), gy = std::max(1, grid[1]);
  return x + y * gx + z * gx * gy;
}
int cluster_linear(int x, int y, int z, const std::vector<int> &dims,
                   const std::vector<int> &grid) {
  const int cx = std::max(1, dims[0]), cy = std::max(1, dims[1]),
            cz = std::max(1, dims[2]);
  const int ncx = cdiv(grid[0], cx), ncy = cdiv(grid[1], cy);
  return x + y * ncx + z * ncx * ncy;
}

std::uint32_t tmem_columns(const std::string &arch,
                           const std::string &configs) {
  if (arch != "blackwell") return 0;
  const fs::path p = fs::path(configs) / "hardware" / "b300" / "device.json";
  std::ifstream fh(p);
  if (!fh) return 0;
  std::string text((std::istreambuf_iterator<char>(fh)),
                   std::istreambuf_iterator<char>());
  const std::string v = json_str(text, "tmem_columns");
  return v.empty() ? 0 : static_cast<std::uint32_t>(std::stoul(v));
}

}  // namespace

int main(int argc, char **argv) {
  std::string case_dir, arch, out_dir, clusters = "all",
      split_dir, configs = "configs",
      reference_tsv, placement_path;
  int launch = -1, shard_k = -1, shard_n = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) die("missing value for " + a);
      return argv[i];
    };
    if (a == "--case") case_dir = next();
    else if (a == "--launch") launch = std::stoi(next());
    else if (a == "--arch") arch = next();
    else if (a == "--out") out_dir = next();
    else if (a == "--clusters") clusters = next();
    else if (a == "--split-dir") split_dir = next();
    else if (a == "--configs") configs = next();
    else if (a == "--reference") reference_tsv = next();
    else if (a == "--placement") placement_path = next();
    else if (a == "--shard") {
      const std::string s = next();
      const auto sl = s.find('/');
      if (sl == std::string::npos) die("--shard expects k/n");
      shard_k = std::stoi(s.substr(0, sl));
      shard_n = std::stoi(s.substr(sl + 1));
    } else die("unknown argument " + a);
  }
  if (launch < 0) {
    // default: the single traced launch in launches.jsonl
    std::ifstream lf(case_dir + "/launches.jsonl");
    std::string line;
    while (std::getline(lf, line)) {
      if (line.find("\"traced\": true") != std::string::npos ||
          line.find("\"traced\":true") != std::string::npos) {
        const std::string l = json_str(line, "launch");
        if (!l.empty()) launch = std::stoi(l);
        break;
      }
    }
  }
  if (case_dir.empty() || launch < 0 || arch.empty() || out_dir.empty())
    die("usage: cyclescope-capture-facts --case D --launch N --arch A "
        "--out D [--clusters spec] [--split-dir D --shard k/n] "
        "[--configs D]");

  {
    const std::string device_key =
        (arch == "hopper") ? "h100" : "b300";
    const fs::path hw = fs::path(configs) / "hardware" / device_key;
    for (const char *f : {"device.json", "sm_topology.json"}) {
      const fs::path p = hw / f;
      if (!fs::is_regular_file(p))
        die("unsupported:configs:" + device_key + "/" + f + " (" +
            p.string() + ")");
    }
    if (arch == "blackwell") {
      const fs::path p = hw / "preinitialized_mbarriers.json";
      if (!fs::is_regular_file(p))
        die("unsupported:configs:" + device_key +
            "/preinitialized_mbarriers.json (" + p.string() + ")");
    }
  }
  const ConvertRules *rules = convert_rules_for(arch);
  if (!rules) die("no convert rules for arch " + arch);
  const CaptureRules &cap_rules = capture_rules_for(arch);
  TmState tm_state;
  tm_state.entries = load_tensor_maps(case_dir);
  std::map<std::string, int> missing_n, unsupported_n;
  std::string trace_sha_global;
  std::uint64_t dropped_total = 0;
  std::map<std::int64_t, std::pair<std::int64_t, std::int64_t>>
      placement_map;
  const LaunchRow lr = find_launch(case_dir, launch);
  if (!lr.found) missing("launch_row", case_dir);
  for (auto &d : const_cast<std::vector<int> &>(lr.dims))
    d = std::max(1, d);

  // _cluster_filter_spec
  std::set<int> keep_ids;
  bool keep_all = clusters == "all";
  if (!keep_all) {
    if (clusters.rfind("sampled:", 0) == 0) {
      const int n = std::stoi(clusters.substr(8));
      for (int i = 0; i < n && i < int(lr.sampled.size()); ++i)
        keep_ids.insert(cluster_linear(lr.sampled[i][0], lr.sampled[i][1],
                                       lr.sampled[i][2], lr.dims, lr.grid));
    } else {
      std::size_t b = 0;
      while (true) {
        const auto c = clusters.find(',', b);
        const std::string t = clusters.substr(
            b, c == std::string::npos ? c : c - b);
        if (!t.empty()) keep_ids.insert(std::stoi(t));
        if (c == std::string::npos) break;
        b = c + 1;
      }
    }
  }
  if (shard_k >= 0) {
    if (keep_all)
      for (const auto &c : lr.sampled)
        keep_ids.insert(cluster_linear(c[0], c[1], c[2], lr.dims, lr.grid));
    for (auto it = keep_ids.begin(); it != keep_ids.end();)
      it = (*it % shard_n == shard_k) ? ++it : keep_ids.erase(it);
    keep_all = false;
  }

  auto tables = load_case(case_dir, *rules);
  // static_index global ordinal; statics in (fid, idx) order
  std::vector<StatRow> statics;
  std::map<std::pair<int, int>, int> sindex;
  // flat (function_id, idx) -> row lookup for the per-record hot path
  std::vector<std::vector<const StaticRow *>> row_by_fid;
  for (auto &[fid, t] : tables)
    for (auto &[idx, r] : t.rows) {
      r.static_index = static_cast<int>(statics.size());
      statics.push_back(r.inst);
      sindex[{fid, idx}] = r.static_index;
      prepare_static_row(r, cap_rules);
      if (fid >= 0 && idx >= 0) {
        if (row_by_fid.size() <= std::size_t(fid))
          row_by_fid.resize(std::size_t(fid) + 1);
        auto &v = row_by_fid[std::size_t(fid)];
        if (v.size() <= std::size_t(idx)) v.resize(std::size_t(idx) + 1);
        v[std::size_t(idx)] = &r;
      }
    }
  const auto row_of = [&](std::uint32_t fid,
                          std::uint32_t idx) -> const StaticRow * {
    if (fid >= row_by_fid.size()) return nullptr;
    const auto &v = row_by_fid[fid];
    return idx < v.size() ? v[idx] : nullptr;
  };
  bool cta_group2 = false;
  for (const auto &s : statics)
    if (s.opcode.find(".2CTA") != std::string::npos) {
      cta_group2 = true;
      break;
    }
  const std::uint32_t tmem = tmem_columns(arch, configs);

  // shard trace path + gsync_seq sidecar
  std::string trace_path;
  std::vector<std::uint64_t> seq;
  std::vector<std::uint64_t> *seq_p = nullptr;
  if (!split_dir.empty() && shard_k >= 0) {
    const std::string base = split_dir + "/trace-" +
        std::to_string(launch) + ".shard" + std::to_string(shard_k) +
        "of" + std::to_string(shard_n);
    trace_path = base + ".bin.zst";
    if (!fs::is_regular_file(trace_path))
      missing("split_trace", trace_path);
    const std::string sf = base + ".gsync_seq";
    std::ifstream sh(sf, std::ios::binary);
    if (!sh) missing("split_gsync_sequence", sf);
    std::vector<char> bytes((std::istreambuf_iterator<char>(sh)),
                            std::istreambuf_iterator<char>());
    for (std::size_t i = 0; i + 8 <= bytes.size(); i += 8) {
      std::uint64_t v;
      std::memcpy(&v, bytes.data() + i, 8);
      seq.push_back(v);
    }
    seq_p = &seq;
  }

  const int cx = std::max(1, lr.dims[0]), cy = std::max(1, lr.dims[1]),
            cz = std::max(1, lr.dims[2]);
  const int ncx = cdiv(lr.grid[0], cx), ncy = cdiv(lr.grid[1], cy);
  const bool split_shard = !split_dir.empty() && shard_k >= 0;
  auto rec_keep = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    if (split_shard) return true;   // split file already routed
    const int cl = int(x / cx) + int(y / cy) * ncx +
        int(z / cz) * ncx * ncy;
    return keep_all || keep_ids.count(cl);
  };

  // buffer records per (cluster -> (cta,warp) -> recs), insertion order
  struct WarpRecs {
    std::uint32_t x = 0, y = 0, z = 0, warp = 0, sm = 0, wslot = 0;
    std::vector<DynRecord> recs;
  };
  const bool profile = std::getenv("CYCLESCOPE_CAPTURE_PROFILE") != nullptr;
  const int capsi = std::getenv("CAPSI") ? std::atoi(std::getenv("CAPSI"))
                                         : -1;   // debug: dump one static
  auto t0 = profile ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
  auto lap = [&](const char *tag) {
    if (!profile) return;
    auto n = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[profile] %s %.3fs\n", tag,
                 std::chrono::duration<double>(n - t0).count());
    t0 = n;
  };
  lap("statics");
  // hardware path: cl = lin*4096 + nctaid.x*256 + .y*16 + .z — unique
  // under mixed-shape launches (must match capture_pipeline.py).
  std::map<int, std::vector<WarpRecs>> by_cluster;
  std::map<int, std::map<std::pair<std::array<int, 3>, int>, int>> order;
  std::map<std::array<std::uint32_t, 3>, std::array<std::uint32_t, 10>>
      cta_info;   // kind-3 records: cta -> 10 u32 cluster facts
  std::uint32_t last_x = ~0u, last_y = ~0u, last_z = ~0u, last_warp = ~0u;
  std::vector<WarpRecs> *last_vec = nullptr;
  int last_wi = 0;
  std::size_t last_cta_n = 0;
  iter_records(
      case_dir, launch, trace_path,
      [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return rec_keep(x, y, z);
      },
      &cta_info,
      seq_p,
      [&](DynRecord &&r) {
        // records of one warp arrive in runs: reuse the last bucket
        if (r.cta_x == last_x && r.cta_y == last_y && r.cta_z == last_z &&
            r.warp == last_warp && last_vec &&
            cta_info.size() == last_cta_n) {
          (*last_vec)[last_wi].recs.push_back(std::move(r));
          return;
        }
        const auto ci = cta_info.find(
            {r.cta_x, r.cta_y, r.cta_z});
        int cl;
        if (ci != cta_info.end()) {
          const auto &v = ci->second;
          const int lin = int(v[4]) + int(v[5]) * int(v[7]) +
              int(v[6]) * int(v[7]) * int(v[8]);
          cl = lin * 4096 + int(v[1]) * 256 + int(v[2]) * 16 +
              int(v[3]);
        } else {
          cl = int(r.cta_x / cx) + int(r.cta_y / cy) * ncx +
              int(r.cta_z / cz) * ncx * ncy;
        }
        const std::array<int, 3> cta{int(r.cta_x), int(r.cta_y),
                                     int(r.cta_z)};
        auto &ord = order[cl];
        const std::pair<std::array<int, 3>, int> key{cta, int(r.warp)};
        int wi;
        auto it = ord.find(key);
        if (it == ord.end()) {
          wi = static_cast<int>(by_cluster[cl].size());
          ord.emplace(key, wi);
          WarpRecs w;
          w.x = r.cta_x; w.y = r.cta_y; w.z = r.cta_z;
          w.warp = r.warp; w.sm = r.sm; w.wslot = r.warp_slot;
          by_cluster[cl].push_back(std::move(w));
        } else {
          wi = it->second;
        }
        auto &vec = by_cluster[cl];
        last_x = r.cta_x; last_y = r.cta_y; last_z = r.cta_z;
        last_warp = r.warp; last_vec = &vec; last_wi = wi;
        last_cta_n = cta_info.size();
        vec[wi].recs.push_back(std::move(r));
      });

  fs::create_directories(out_dir);
  lap("read_trace");
  double t_instr = 0, t_write = 0, t_prep = 0;
  auto sub = [&](double &acc, auto &&f) {
    if (!profile) { f(); return; }
    auto a = std::chrono::steady_clock::now();
    f();
    acc += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - a).count();
  };

  // split_capture writes the digest of the compressed trace once per
  // case (<split_dir>/trace-<launch>.sha256); hashing the whole file in
  // every shard process cost ~2.5 s per shard
  if (!split_dir.empty()) {
    std::ifstream sf(split_dir + "/trace-" + std::to_string(launch) +
                     ".sha256");
    std::string hex;
    if (sf >> hex && hex.size() == 64 &&
        hex.find_first_not_of("0123456789abcdef") == std::string::npos)
      trace_sha_global = hex;
  }
  if (trace_sha_global.empty())
    trace_sha_global = file_sha256(
        case_dir + "/trace-" + std::to_string(launch) + ".bin.zst");
  const auto run_cluster = [&](int cl, std::vector<WarpRecs> &warps,
                               std::map<std::string, int> &missing_l,
                               std::map<std::string, int> &unsupported_l,
                               std::map<std::int64_t,
                                        std::pair<std::int64_t,
                                                  std::int64_t>> &placement_l,
                               std::uint64_t &dropped_l,
                               double &l_instr, double &l_write,
                               double &l_prep) {
    TmState tms;
    tms.entries = tm_state.entries;

    // warps in sorted (cta,warp) order — like warps_done after sort
    std::sort(warps.begin(), warps.end(), [](const WarpRecs &a,
                                           const WarpRecs &b) {
      return std::tie(a.x, a.y, a.z, a.warp) <
             std::tie(b.x, b.y, b.z, b.warp);
    });
    // cluster_rank_list = sorted ctas by (z,y,x)
    std::map<std::array<int, 3>, int> rank;
    for (const auto &w : warps)
      rank.emplace(std::array<int, 3>{int(w.x), int(w.y), int(w.z)}, 0);
    {
      std::vector<std::array<int, 3>> ctas;
      for (auto &[c, r] : rank) ctas.push_back(c);
      std::sort(ctas.begin(), ctas.end(),
                [](const std::array<int, 3> &a,
                   const std::array<int, 3> &b) {
                  return std::tie(a[2], a[1], a[0]) <
                         std::tie(b[2], b[1], b[0]);
                });
      for (std::size_t i = 0; i < ctas.size(); ++i) rank[ctas[i]] = int(i);
    }
    // hardware rank: %cluster_ctarank when kind-3 records exist;
    // multicast over product(%cluster_nctaid)
    int n_cta = static_cast<int>(rank.size());
    if (!cta_info.empty()) {
      int max_cg = 1;
      for (const auto &[k, v] : cta_info)
        max_cg = std::max(max_cg, int(v[1]) * int(v[2]) * int(v[3]));
      n_cta = std::max(n_cta, max_cg);
      for (const auto &w : warps) {
        const auto ci = cta_info.find(
            {w.x, w.y, w.z});
        if (ci != cta_info.end())
          rank[{int(w.x), int(w.y), int(w.z)}] = int(ci->second[0]);
      }
    }
    std::vector<CtaRow> ctas;
    std::map<int, int> rank_to_cta;
    std::vector<CtaRow> seen;
    std::set<std::array<int, 3>> added;
    for (const auto &w : warps) {
      const std::array<int, 3> c{int(w.x), int(w.y), int(w.z)};
      if (added.count(c)) continue;
      added.insert(c);
      CtaRow cr;
      cr.cta_linear = cta_linear(int(w.x), int(w.y), int(w.z), lr.grid);
      cr.x = w.x; cr.y = w.y; cr.z = w.z;
      cr.cluster_rank = static_cast<std::uint32_t>(rank[c]);
      cr.sm_id = w.sm;
      cr.cta_group = cta_group2 ? 2 : 1;
      cr.pair_leader = cta_group2 && rank[c] % 2 == 0 ? 1 : 0;
      cr.multicast_mask = n_cta >= 32 ? 0xFFFFFFFFu
                                    : ((1u << n_cta) - 1);
      cr.tmem_columns = tmem;
      rank_to_cta[cr.cluster_rank] = static_cast<int>(ctas.size());
      ctas.push_back(cr);
    }
    // peer_sm_id from rank partner
    for (auto &c : ctas) {
      const auto it = rank_to_cta.find(int(c.cluster_rank) ^ 1);
      c.peer_sm_id =
          (c.cta_group == 2 && it != rank_to_cta.end())
              ? ctas[it->second].sm_id
              : c.sm_id;
    }

    StrIntern strtab;
    std::vector<WarpRow> warps_rows;
    std::vector<std::vector<DynRow>> dyns;
    std::uint64_t dropped_polls = 0;
    tms.used.clear();
    std::size_t t_ukv = 0, t_lane = 0, t_lval = 0, t_cord = 0,
        t_desc = 0, t_ops = 0, t_opsx = 0;
    std::size_t t_addr = 0;
    // pass 1: per-warp folded positions + control-flow patches
    std::vector<std::pair<std::set<int>, int>> foldeds;
    std::vector<Patches> warp_patches;
    std::vector<Patches *> patch_ptrs;
    const StaticOf static_of = [&](const DynRecord &r) {
      return row_of(r.function_id, r.idx);
    };
    std::map<int, std::map<std::uint64_t, const StaticRow *>> pc_maps;
    const ByPc by_pc = [&](int fid) -> const
        std::map<std::uint64_t, const StaticRow *> & {
      auto &m = pc_maps[fid];
      if (m.empty())
        for (const auto &[idx, r] : tables[fid].rows) m[r.inst.pc] = &r;
      return m;
    };
    for (const auto &w : warps) {
      foldeds.push_back(fold_warp(w.recs, static_of));
      dropped_polls += foldeds.back().second;
      dropped_l += foldeds.back().second;
      warp_patches.push_back(flow_patches(
          w.recs, static_of, by_pc,
          std::uint64_t(w.x) << 40 | std::uint64_t(w.y) << 20 |
              std::uint64_t(w.z)));
    }
    for (auto &p : warp_patches) patch_ptrs.push_back(&p);
    fill_participating(patch_ptrs);
    std::size_t wi = 0;
    for (const auto &w : warps) {
      const auto &folded = foldeds[wi];
      const auto &patches = warp_patches[wi];
      ++wi;
      if (std::getenv("CAPDBG")) {
        int ph=0, ar=0;
        for (const auto &r : w.recs) {
          if (static_of(r) && static_of(r)->inst.opcode.find("PHASECHK")!=std::string::npos) ++ph;
          if (r.has_after_ready) ++ar;
        }
        std::fprintf(stderr, "warp %u recs=%zu phasechk=%d after=%d folded=%zu\n",
                     w.warp, w.recs.size(), ph, ar, folded.first.size());
      }
      WarpRow wr;
      wr.cta_linear =
          static_cast<std::uint32_t>(cta_linear(int(w.x), int(w.y),
                                              int(w.z), lr.grid));
      wr.x = w.x; wr.y = w.y; wr.z = w.z;
      wr.sm_id = w.sm;
      wr.warp_slot = w.wslot;
      wr.warp_in_cta = static_cast<std::int32_t>(w.warp);
      if (w.warp == 0)
        placement_l[wr.cta_linear] = {w.sm, w.wslot};
      warps_rows.push_back(wr);
      std::vector<DynRow> ds;
      ds.reserve(w.recs.size());
      const int crank = rank[{int(w.x), int(w.y), int(w.z)}];
      int pos = 0;
      for (const auto &r : w.recs) {
        if (folded.first.count(pos++)) continue;
        const StaticRow *srow = row_of(r.function_id, r.idx);
        if (!srow) continue;   // missing static_row
        // patch position key uses 0-based record pos before fold-skip
        DynRow d;
        const auto _t0i = profile ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        d.static_index = static_cast<std::uint32_t>(srow->static_index);
        d.active_mask = r.active_mask;
        d.predicate_mask = r.exec_mask;
        {
          const auto ranks = remote_ranks(r.ops, crank);
          auto ops = operand_facts(r, *srow, cap_rules, crank, &ranks);
          const auto mrefs = memory_refs(
              r, *srow, cap_rules, tms, missing_l,
              unsupported_l, ranks, ops ? &*ops : nullptr);
          const auto pit = patches.find(pos - 1);
          if (pit != patches.end() && !ops)
            ops = SemanticOperands{};
          if (pit != patches.end()) {
            for (const auto &[k, v, is_int] : pit->second.entries) {
              if (!k.empty() && k[0] == '_') continue;
              if (is_int) uput_wrap(*ops, k, std::stoll(v));
              else uput_str(*ops, k, v);
            }
          }
          const auto gso = observations(r, *srow);
          if (!gso.lane.empty() && !ops) ops = SemanticOperands{};
          if (ops)
            for (const auto &k : gso.lane_order) {
              if (!ops->lane.count(k)) ops->lane_order.push_back(k);
              ops->lane[k] = gso.lane.at(k);
            }
          if (ops) {
            d.has_ops = true;
            auto put4 = [](std::vector<std::uint8_t> &v, std::uint32_t x) {
              for (int i = 0; i < 4; ++i) v.push_back(x >> (8 * i));
            };
            auto put8 = [](std::vector<std::uint8_t> &v, std::uint64_t x) {
              for (int i = 0; i < 8; ++i) v.push_back(x >> (8 * i));
            };
            const std::uint32_t kb = static_cast<std::uint32_t>(t_ukv);
            for (const auto &k : ops->uniform_order) {
              const auto sit = ops->str_uniform.find(k);
              put4(d.ukv, strtab.intern(k));
              if (sit != ops->str_uniform.end()) {
                d.ukv.push_back(0);
                put8(d.ukv, strtab.intern(sit->second));
              } else {
                d.ukv.push_back(1);
                put8(d.ukv,
                     static_cast<std::uint64_t>(ops->uniform[k]));
              }
            }
            const std::uint32_t lb = static_cast<std::uint32_t>(t_lane);
            for (const auto &k : ops->lane_order) {
              const auto &vs = ops->lane[k];
              const std::uint32_t vb = static_cast<std::uint32_t>(
                  t_lval + d.lval.size() / 8);
              for (const auto x : vs)
                put8(d.lval, static_cast<std::uint64_t>(x));
              put4(d.lane, strtab.intern(k));
              put4(d.lane, vb);
              put4(d.lane, static_cast<std::uint32_t>(vs.size()));
            }
            const std::uint32_t rb =
                lb + static_cast<std::uint32_t>(d.lane.size() / 12);
            const std::uint32_t lane_cnt =
                static_cast<std::uint32_t>(d.lane.size() / 12);
            for (const int k : ops->ranks_order) {
              const auto &vs = ops->ranks[k];
              const std::uint32_t vb = static_cast<std::uint32_t>(
                  t_lval + d.lval.size() / 8);
              for (const auto x : vs)
                put8(d.lval, static_cast<std::uint64_t>(x));
              put4(d.lane, 0x80000000u | std::uint32_t(k));
              put4(d.lane, vb);
              put4(d.lane, static_cast<std::uint32_t>(vs.size()));
            }
            const std::uint32_t rank_cnt =
                static_cast<std::uint32_t>(d.lane.size() / 12) - lane_cnt;
            const std::uint32_t cb = static_cast<std::uint32_t>(t_cord);
            for (const auto v : ops->tensor_coordinates)
              put8(d.cord, static_cast<std::uint64_t>(v));
            const std::uint32_t db = static_cast<std::uint32_t>(t_desc);
            for (const auto v : ops->descriptor_words)
              put8(d.desc, static_cast<std::uint64_t>(v));
            for (int i = 0; i < 8; ++i) put4(d.ops, 0);
            auto *op = d.ops.data();
            auto w32 = [&](int i, std::uint32_t x) {
              std::memcpy(op + 4 * i, &x, 4);
            };
            w32(0, kb);
            w32(1, static_cast<std::uint32_t>(d.ukv.size() / 13));
            w32(2, lb); w32(3, lane_cnt);
            w32(4, rb); w32(5, rank_cnt);
            w32(6, ops->after_ready_mask ? *ops->after_ready_mask : 0);
            w32(7, ops->after_ready_mask ? 1 : 0);
            put4(d.opsx, ops->tensor_map ? *ops->tensor_map : 0);
            put4(d.opsx, cb);
            put4(d.opsx, static_cast<std::uint32_t>(d.cord.size() / 8));
            put4(d.opsx, db);
            put4(d.opsx, static_cast<std::uint32_t>(d.desc.size() / 8));
            t_ukv += d.ukv.size() / 13;
            t_lane += d.lane.size() / 12;
            t_lval += d.lval.size() / 8;
            t_cord += d.cord.size() / 8;
            t_desc += d.desc.size() / 8;
          }
          // MREF/MREX/ADDR rows (cluster-level addr offsets)
          {
          static const std::map<std::string, int> kSpace = {
              {"global", 0}, {"shared", 1}, {"local", 2},
              {"constant", 3}, {"generic", 4}, {"param", 5},
              {"shared_cluster", 6}, {"global_system", 7}};
          static const std::map<std::string, int> kAk = {
              {"load", 0}, {"store", 1}, {"atomic", 2},
              {"async_copy", 3}, {"async_reduce", 4}, {"prefetch", 5},
              {"reduction", 6}, {"other", 7}};
          for (const auto &mr : mrefs) {
            const std::uint32_t ab = static_cast<std::uint32_t>(
                t_addr + d.addr.size() / 8);
            for (const auto a : mr.lane_addresses) {
              for (int i = 0; i < 8; ++i)
                d.addr.push_back(std::uint8_t(std::uint64_t(a) >> (8 * i)));
            }
            auto puti = [&](std::vector<std::uint8_t> &v, std::int32_t x) {
              for (int i = 0; i < 4; ++i)
                v.push_back(std::uint8_t(std::uint32_t(x) >> (8 * i)));
            };
            puti(d.mref, kSpace.count(mr.space) ? kSpace.at(mr.space) : 0);
            puti(d.mref, kAk.count(mr.access_kind)
                            ? kAk.at(mr.access_kind) : 7);
            puti(d.mref, mr.width_bytes);
            puti(d.mref, mr.remote_rank ? *mr.remote_rank : -1);
            puti(d.mref, std::int32_t(mr.lane_mask));
            puti(d.mref, mr.cache_hint);
            puti(d.mref,
                 mr.destination_count ? *mr.destination_count : -1);
            const std::int64_t tb =
                mr.transfer_bytes ? *mr.transfer_bytes : -1;
            for (int i = 0; i < 8; ++i)
              d.mrefx.push_back(std::uint8_t(std::uint64_t(tb) >> (8 * i)));
            for (int i = 0; i < 4; ++i)
              d.mrefx.push_back(std::uint8_t(ab >> (8 * i)));
            const std::uint32_t cnt = static_cast<std::uint32_t>(
                d.addr.size() / 8) - (ab - static_cast<std::uint32_t>(t_addr));
            for (int i = 0; i < 4; ++i)
              d.mrefx.push_back(std::uint8_t(cnt >> (8 * i)));
            const std::int32_t cr =
                mr.cut_rows ? *mr.cut_rows : -1;
            for (int i = 0; i < 4; ++i)
              d.mrefx.push_back(std::uint8_t(std::uint32_t(cr) >> (8 * i)));
          }
          d.mref_count = static_cast<std::uint32_t>(mrefs.size());
          }
          for (const auto &o : gso.obs) {
            d.has_gsync = true;
            auto p4 = [&](std::vector<std::uint8_t> &v, std::uint32_t x) {
              for (int i = 0; i < 4; ++i)
                v.push_back(std::uint8_t(x >> (8 * i)));
            };
            auto p8 = [&](std::vector<std::uint8_t> &v, std::uint64_t x) {
              for (int i = 0; i < 8; ++i)
                v.push_back(std::uint8_t(x >> (8 * i)));
            };
            p4(d.gsyn, std::uint32_t(o.kind));
            p4(d.gsyn, std::uint32_t(o.lane));
            p8(d.gsyn, std::uint64_t(o.address));
            p8(d.gsyn, std::uint64_t(o.before));
            p8(d.gsyn, std::uint64_t(o.value));
            p8(d.gsyn, o.sequence);
            p4(d.gsyx, std::uint32_t(o.acquire_mode));
            p4(d.gsyx, std::uint32_t(o.post_known));
          }
        }
        t_addr += d.addr.size() / 8;
        if (capsi >= 0 && srow->static_index == capsi) {
          fprintf(stderr, "si=%d layout:", srow->static_index);
          for (const auto &o : srow->ops_layout)
            fprintf(stderr, " %s", o.c_str());
          fprintf(stderr, "\nops:");
          for (const auto &[k, v] : r.ops) {
            fprintf(stderr, " %s=[", k.c_str());
            for (std::size_t i = 0; i < v.size() && i < 4; ++i)
              fprintf(stderr, "%lld,", (long long)v[i]);
            fprintf(stderr, "]");
          }
          fprintf(stderr, "\n");
        }
        if (profile)
          l_instr += std::chrono::duration<double>(
              std::chrono::steady_clock::now() - _t0i).count();
        ds.push_back(std::move(d));
      }
      dyns.push_back(std::move(ds));
    }
    char name[16];
    std::snprintf(name, sizeof name, "%06d.gsc", cl);
    const std::uint8_t zeros[32] = {0};
    // META digest = sha256(trace_sha|launch|cl)
    std::uint8_t digest32[32] = {0};
    {
      sha256_digest(trace_sha_global + "|" +
                    std::to_string(launch) + "|" +
                    std::to_string(cl), digest32);
    }
    std::vector<std::vector<std::uint8_t>> mbpi_rows;
    if (arch == "blackwell" && !ctas.empty()) {
      const fs::path mbp = fs::path(configs) / "hardware" / "b300" /
                           "preinitialized_mbarriers.json";
      std::ifstream jf(mbp);
      if (!jf)
        die("unsupported:configs:preinitialized_mbarriers.json (" +
            mbp.string() + ")");
      std::ostringstream js;
      js << jf.rdbuf();
      const std::string j = js.str();
      // minimal parse: barrier entries carry "offset": N and
      // "expected_arrivals": N | {"<cg>": N, "default": N}
      const std::regex ore(R"("offset"\s*:\s*(-?[0-9]+))");
      const std::regex numre(R"("expected_arrivals"\s*:\s*(-?[0-9]+))");
      const std::regex cgre("\"" + std::to_string(ctas[0].cta_group) +
                            "\"\\s*:\\s*(-?[0-9]+)");
      const std::regex dre(R"("default"\s*:\s*(-?[0-9]+))");
      const auto op = j.find("\"offset\"");
      if (op == std::string::npos)
        die("unsupported:configs:preinitialized_mbarriers.json:offset (" +
            mbp.string() + ")");
      {
        std::smatch om;
        const std::string tail = j.substr(op);
        std::int64_t offset = -1;
        if (std::regex_search(tail, om, ore))
          offset = std::stoll(om[1]);
        std::int64_t arrivals = -1;
        const auto ap = tail.find("\"expected_arrivals\"");
        if (ap != std::string::npos) {
          const std::string atail = tail.substr(ap);
          std::smatch m2;
          if (std::regex_search(atail, m2, numre))
            arrivals = std::stoll(m2[1]);
          else if (std::regex_search(atail, m2, cgre))
            arrivals = std::stoll(m2[1]);
          else if (std::regex_search(atail, m2, dre))
            arrivals = std::stoll(m2[1]);
        }
        if (offset >= 0 && arrivals >= 0)
          for (const auto &c : ctas) {
            mbpi_rows.emplace_back(32, 0);
            auto *p = mbpi_rows.back().data();
            std::int32_t v;
            v = c.cta_linear; std::memcpy(p, &v, 4);
            v = std::int32_t(offset); std::memcpy(p + 4, &v, 4);
            v = std::int32_t(arrivals); std::memcpy(p + 8, &v, 4);
          }
      }
    }
    std::vector<std::vector<std::uint8_t>> tmap_rows, tmdi_rows;
    for (const int tm_id : tms.used) {
      const auto mit = tms.models.find(tm_id);
      if (mit == tms.models.end()) continue;
      const auto &t = mit->second;
      const std::uint32_t dd = tmdi_rows.size(), ss = dd + t.global_dim.size();
      for (const auto v : t.global_dim) {
        tmdi_rows.emplace_back();
        for (int i = 0; i < 8; ++i)
          tmdi_rows.back().push_back(std::uint8_t(std::uint64_t(v) >> (8 * i)));
      }
      for (const auto v : t.stride_bytes) {
        tmdi_rows.emplace_back();
        for (int i = 0; i < 8; ++i)
          tmdi_rows.back().push_back(std::uint8_t(v >> (8 * i)));
      }
      const std::uint32_t bb = tmdi_rows.size();
      for (const auto v : t.box_dim) {
        tmdi_rows.emplace_back();
        for (int i = 0; i < 8; ++i)
          tmdi_rows.back().push_back(std::uint8_t(std::uint64_t(v) >> (8 * i)));
      }
      tmap_rows.emplace_back();
      auto p4 = [&](std::uint32_t x) {
        for (int i = 0; i < 4; ++i) tmap_rows.back().push_back(std::uint8_t(x >> (8 * i)));
      };
      auto p8 = [&](std::uint64_t x) {
        for (int i = 0; i < 8; ++i) tmap_rows.back().push_back(std::uint8_t(x >> (8 * i)));
      };
      p4(std::uint32_t(tm_id)); p4(std::uint32_t(t.element_bytes));
      p4(std::uint32_t(t.rank)); p8(t.global_address);
      p8(dd); p8(ss); p8(bb);
    }
    std::vector<std::uint8_t> bytes;
    sub(l_write, [&] {
      bytes = write_cluster_file(
          strtab, static_cast<std::uint32_t>(cl), ctas, warps_rows, dyns,
          tmap_rows, tmdi_rows, mbpi_rows,
          static_cast<std::uint32_t>(statics.size()), digest32);
      const fs::path p = fs::path(out_dir) / name;
      std::ofstream oh(p, std::ios::binary);
      oh.write(reinterpret_cast<const char *>(bytes.data()),
               bytes.size());
    });
  };

  // clusters run on a bounded pool (CYCLESCOPE_CAPTURE_THREADS, default =
  // hardware threads; 1 = serial in this thread); per-cluster results
  // are merged in by_cluster order, so the pool size does not affect
  // the output
  struct ClusterResult {
    std::map<std::string, int> ml, ul;
    std::map<std::int64_t, std::pair<std::int64_t, std::int64_t>> pl;
    std::uint64_t dl = 0;
  };
  std::vector<std::pair<int, std::vector<WarpRecs> *>> work;
  for (auto &[cl, warps] : by_cluster) work.emplace_back(cl, &warps);
  std::vector<ClusterResult> results(work.size());
  std::atomic<std::size_t> next_work{0};
  const auto worker = [&]() {
    for (;;) {
      const std::size_t i = next_work.fetch_add(1);
      if (i >= work.size()) break;
      auto &r = results[i];
      double li = 0, lw = 0, lp = 0;
      run_cluster(work[i].first, *work[i].second, r.ml, r.ul, r.pl, r.dl,
                  li, lw, lp);
    }
  };
  {
    unsigned nthreads = std::thread::hardware_concurrency();
    if (const char *e = std::getenv("CYCLESCOPE_CAPTURE_THREADS"))
      nthreads = static_cast<unsigned>(std::max(1, std::atoi(e)));
    nthreads = std::max(1u, std::min<unsigned>(
        nthreads, static_cast<unsigned>(work.size())));
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < nthreads; ++t) pool.emplace_back(worker);
    worker();
    for (auto &t : pool) t.join();
  }
  for (auto &r : results) {
    for (const auto &[k, v] : r.ml) missing_n[k] += v;
    for (const auto &[k, v] : r.ul) unsupported_n[k] += v;
    for (const auto &[k, v] : r.pl) placement_map[k] = v;
    dropped_total += r.dl;
  }
  if (profile) {
    std::fprintf(stderr,
        "[profile] instr=%.3f write=%.3f prep=%.3f\n",
        t_instr, t_write, t_prep);
  }
  {
    LaunchDocInput li;
    li.case_dir = case_dir;
    li.device_key = (arch == "hopper") ? "h100" : "b300";
    li.launch = launch;
    const std::string fn1 = json_str(lr.raw, "mangled");
    const std::string fn2 = json_str(lr.raw, "name");
    li.function_name = !fn1.empty() ? fn1 : fn2;
    for (const auto g : lr.grid) li.grid.push_back(g);
    {
      const auto bl = json_arr3(lr.raw, "block");
      for (const auto b : bl) li.block.push_back(b);
    }
    li.dims = lr.dims;
    li.cta_info_present = !cta_info.empty();
    {
      const std::string p1 = (lr.raw.find("\"preferred_cluster\"") !=
                              std::string::npos)
                                 ? "1" : "";
      if (!p1.empty()) {
        const auto pc = json_arr3(lr.raw, "preferred_cluster");
        for (const auto c : pc) li.preferred_cluster.push_back(c);
      }
    }
    if (!keep_all) {
      li.keep = std::vector<int>(keep_ids.begin(), keep_ids.end());
    }
    if (!lr.sampled.empty()) {
      li.sampled_clusters =
          std::vector<std::vector<std::int64_t>>();
      for (const auto &c : lr.sampled) {
        std::vector<std::int64_t> r(c.begin(), c.end());
        li.sampled_clusters->push_back(r);
      }
    }
    li.smem_static = json_num(lr.raw, "smem_static", 0);
    li.smem_dynamic = json_num(lr.raw, "smem_dynamic", 0);
    li.regs = json_num(lr.raw, "regs", 0);
    {
      const std::string ob = json_str(lr.raw,
                                      "occupancy_max_blocks_per_sm");
      if (!ob.empty()) li.occupancy_blocks_per_sm = std::stoll(ob);
      li.occupancy_blocks_per_sm_tracer = li.occupancy_blocks_per_sm;
      {
        const auto occ = ncu_occupancy(case_dir, lr);
        if (occ > 0) li.occupancy_ncu = occ;
      }
      const std::string oc = json_str(
          lr.raw, "occupancy_max_active_clusters");
      if (!oc.empty()) li.occupancy_active_clusters = std::stoll(oc);
      const std::string fp = json_str(lr.raw, "function_prior_launches");
      if (!fp.empty()) li.function_prior_launches = std::stoll(fp);
    }
    {
      std::ifstream dj(case_dir + "/device.json");
      std::ostringstream ds;
      ds << dj.rdbuf();
      const std::string dline = ds.str();
      const std::string sc = json_str(dline, "sm_count");
      if (!sc.empty()) li.sm_count = std::stoll(sc);
      const std::string l2 = json_str(dline, "l2_cache_bytes");
      if (!l2.empty()) li.l2_bytes = std::stoll(l2);
    }
    li.sm_arch = arch == "blackwell" ? "sm_103a" : "sm_90a";
    li.trace_sha = trace_sha_global;
    li.dropped_polls = dropped_total;
    for (const auto &[k, v] : missing_n) li.missing[k] = v;
    for (const auto &[k, v] : unsupported_n) li.unsupported[k] = v;
    for (const auto &[cta, pl] : placement_map)
      li.placements[std::to_string(cta)] = {
          {"sm_id", pl.first}, {"warp_slot_base", pl.second},
          {"source", 1}};
    // probe placement.jsonl (--placement): source=2, first wave only --
    // non-cluster: linear < sm_count*occ_blocks; cluster launch:
    // cluster_id < occupancy_active_clusters (matches Python
    // capture_pipeline._placements).
    if (!placement_path.empty()) {
      fs::path pj = fs::path(placement_path);
      if (fs::is_directory(pj)) pj /= "placement.jsonl";
      if (fs::is_regular_file(pj)) {
        const bool cluster_launch =
            (lr.dims[0] > 1 || lr.dims[1] > 1 || lr.dims[2] > 1);
        const long long smc = li.sm_count.value_or(0);
        const long long occb = li.occupancy_blocks_per_sm.value_or(0);
        const long long ncx_ = (lr.grid[0] + lr.dims[0] - 1) / lr.dims[0];
        const long long ncy_ = (lr.grid[1] + lr.dims[1] - 1) / lr.dims[1];
        std::map<long long, std::map<std::string, std::int64_t>> probe;
        std::map<long long, long long> pwarp;
        std::ifstream pf(pj);
        std::string line;
        while (std::getline(pf, line)) {
          if (line.find("\"entry\"") == std::string::npos) continue;
          const auto c3 = json_arr3(line, "cta");
          if (c3.size() != 3) continue;
          const long long key = (long long)c3[0] +
              (long long)c3[1] * lr.grid[0] +
              (long long)c3[2] * lr.grid[0] * lr.grid[1];
          if (cluster_launch) {
            if (!li.occupancy_active_clusters.has_value()) continue;
            const long long cl = (c3[0] / lr.dims[0]) +
                (c3[1] / lr.dims[1]) * ncx_ +
                (c3[2] / lr.dims[2]) * ncx_ * ncy_;
            if (cl >= *li.occupancy_active_clusters) continue;
          } else if (key >= smc * occb) {
            continue;
          }
          const long long w = (long long)json_num(line, "warp", 0);
          auto it = pwarp.find(key);
          if (it == pwarp.end() || w < it->second) {
            pwarp[key] = w;
            probe[key] = {
                {"sm_id", (std::int64_t)json_num(line, "smid", 0)},
                {"warp_slot_base",
                 (std::int64_t)json_num(line, "warpid", 0)},
                {"source", 2}};
          }
        }
        if (!probe.empty()) {
          li.placements.clear();
          for (auto &[k, e] : probe)
            li.placements[std::to_string(k)] = e;
        }
      }
    }
    {
      const fs::path topo = fs::path(configs) / "hardware" /
          li.device_key / "sm_topology.json";
      if (fs::is_regular_file(topo)) li.sm_topology_path = topo.string();
      if (!reference_tsv.empty()) {
        std::ifstream rf(reference_tsv);
        std::string hdr;
        const std::string cname = fs::path(case_dir).filename().string();
        if (std::getline(rf, hdr)) {
          std::istringstream hh(hdr);
          std::vector<std::string> cols;
          for (std::string t; std::getline(hh, t, '\t');) cols.push_back(t);
          for (std::string line; std::getline(rf, line);) {
            if (line.rfind(cname + "\t", 0) != 0) continue;
            std::istringstream ll(line);
            std::map<std::string, std::string> row;
            size_t i = 0;
            for (std::string t; std::getline(ll, t, '\t') && i < cols.size();)
              row[cols[i++]] = t;
            auto it = row.find("sm_clock_khz");
            if (it != row.end() && !it->second.empty())
              li.reference_sm_clock_khz =
                  (std::int64_t)std::stod(it->second);
            auto ic = row.find("cache_control");
            if (ic != row.end() && !ic->second.empty()) {
              const std::string &v = ic->second;
              li.reference_cache_control =
                  v == "none" ? 1 : (v == "all" ? 2 : 0);
            }
            li.reference_source = reference_tsv;
            break;
          }
        }
      }
    }
    const std::string doc = launch_doc_json(li);
    std::ofstream oh(fs::path(out_dir) / "launch.json");
    oh << doc;
  }
  {
    const auto bytes = write_statics_file(statics);
    std::ofstream oh(fs::path(out_dir) / "statics.gsc",
                     std::ios::binary);
    oh.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  }
  std::printf("convert_capture launch=%d clusters=%zu statics=%zu\n",
              launch, by_cluster.size(), statics.size());
  lap("clusters+launch");
  return 0;
}
