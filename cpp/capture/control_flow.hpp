// Control-flow facts for one warp's stream
// (python/cyclescope/convert/control_flow.py flow_patches /
// fill_participating / bssy_close_pairs / _btoken).
#pragma once
#include "capture/trace_reader.hpp"
#include "capture/static_code.hpp"
#include "capture/poll_folding.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace cyclescope_convert::capture {

// patch entry: position -> (uniform facts + internal "_"-keys)
struct PatchEntry {
  // insertion-ordered (Python dict order): (key, value, is_int)
  std::vector<std::tuple<std::string, std::string, bool>> entries;
  void put(const std::string &k, const std::string &v, bool is_int) {
    for (auto &[ek, ev, ei] : entries)
      if (ek == k) { ev = v; ei = is_int; return; }
    entries.emplace_back(k, v, is_int);
  }
  void puti(const std::string &k, std::int64_t v) {
    put(k, std::to_string(v), true);
  }
  std::string pop(const std::string &k) {
    for (auto it = entries.begin(); it != entries.end(); ++it)
      if (std::get<0>(*it) == k) {
        const std::string v = std::get<1>(*it);
        entries.erase(it);
        return v;
      }
    return "";
  }
};
using Patches = std::map<int, PatchEntry>;   // record pos -> patch

// static_by_pc_of(fid) -> {pc: const StaticRow*}
using ByPc = std::function<const std::map<std::uint64_t,
                                          const StaticRow *> &(int fid)>;

Patches flow_patches(const std::vector<DynRecord> &records,
                     const StaticOf &static_of, const ByPc &by_pc,
                     std::uint64_t cta_key);

// fills control_participating_warps into each patch; keys are removed
// (like Python's pop) — call after all warps of the CTA are done.
void fill_participating(std::vector<Patches *> &patches_by_warp);

}  // namespace cyclescope_convert::capture
