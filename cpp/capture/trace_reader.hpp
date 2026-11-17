// Streaming reader of trace-<launch>.bin.zst record streams.
// Mirrors python/cyclescope/convert/trace_reader.py (iter_records) and
// shares the V5 layout with cpp/convert/trace_split.cpp.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cyclescope_convert::capture {

// HEAD <IHHHBBIIIII (32B V5Head); kinds: 0 INSTR, 1 AFTER, 2 GSYNC
struct Head {
  std::uint32_t cta_x;
  std::uint16_t cta_y, cta_z, warp;
  std::uint8_t sm, kind;
  std::uint32_t function, idx, active_mask, exec_mask, warp_slot;
};

struct DynRecord {
  int launch = 0;
  std::uint32_t cta_x = 0, cta_y = 0, cta_z = 0;
  std::uint32_t warp = 0, sm = 0, warp_slot = 0;
  std::uint32_t function_id = 0, idx = 0;
  std::uint32_t active_mask = 0, exec_mask = 0;
  // mrefs[m] = exec_count u64 lane addresses
  std::vector<std::vector<std::uint64_t>> mrefs;
  // ops[name] = values in stream order (u32 scalars / per-lane u32,
  // RANK*/BRANK* signed i32 values kept sign-extended)
  std::map<std::string, std::vector<std::int64_t>> ops;
  std::vector<std::string> ops_order;   // dict insertion order (Python)
  bool has_after_ready = false;
  std::uint32_t after_ready = 0;
  // gsync entries: (phase, nwords, flags, words[exec_count][nwords], seq)
  struct GSync {
    std::uint8_t phase = 0, nwords = 0, flags = 0;
    std::vector<std::vector<std::uint32_t>> words;
    std::uint64_t sequence = 0;
  };
  std::vector<GSync> gsync;

  int exec_count() const {
    int n = 0;
    for (std::uint32_t m = exec_mask; m; m &= m - 1) ++n;
    return n;
  }
};

// Layouts: function_id -> idx -> (ops, mref_count); same tsv columns as
// trace_split.load_layouts / trace_reader.load_static.
using Layouts = std::map<std::uint32_t,
                         std::map<std::uint32_t,
                                  std::pair<std::vector<std::string>,
                                            std::uint32_t>>>;
Layouts load_layouts(const std::string &functions_dir);

std::uint64_t op_entry_len(const std::string &op, std::uint64_t exec);
std::uint64_t record_tail_len(const std::vector<std::string> &ops,
                              std::uint32_t mref_count,
                              std::uint64_t exec);

// iter_records equivalent: streams the trace for `launch`, calls `emit`
// for each completed DynRecord (attachments resolved lazily exactly as
// kind-3 CTA records (v6.1): when cta_info is non-null they fill it as
// cta_info[{x,y,z}] = {ctarank, nctaid[3], clusterid[3], nclusterid[3]};
// one row per CTA suffices, later warps are ignored.
// the Python version).  cluster_filter(cta_x,cta_y,cta_z)->bool or
// nullptr = keep all.  gsync_seq: for pre-split streams, u64 ordinals
// read from the .gsync_seq sidecar (pass empty for normal traces).
void iter_records(
    const std::string &case_dir, int launch,
    const std::string &trace_path,      // "" = auto from case_dir
    const std::function<bool(std::uint32_t, std::uint32_t,
                             std::uint32_t)> &cluster_filter,
    std::map<std::array<std::uint32_t, 3>,
             std::array<std::uint32_t, 10>> *cta_info,
    std::vector<std::uint64_t> *gsync_seq,
    const std::function<void(DynRecord &&)> &emit);

}  // namespace cyclescope_convert::capture
