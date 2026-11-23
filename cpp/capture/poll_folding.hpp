// Polling-fold rules for one warp's dynamic record stream.
// Mirrors python/cyclescope/convert/poll_folding.py fold_warp /
// poll_ready / _poll_key / _gap_free (whitelists verbatim).
#pragma once
#include "capture/trace_reader.hpp"
#include "capture/static_code.hpp"
#include <functional>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace cyclescope_convert::capture {

// static_of(rec) -> const StaticRow* for (function_id, idx)
using StaticOf = std::function<const StaticRow *(const DynRecord &)>;

// (folded positions, dropped poll count)
// folded positions, dropped poll count, wait origins (survivor
// position -> absolute code address; placement v4 rule P')
struct FoldResult {
  std::set<int> folded;
  int dropped = 0;
  std::map<int, std::uint64_t> wait_origin;
};

FoldResult fold_warp(const std::vector<DynRecord> &records,
                     const StaticOf &static_of);

}  // namespace cyclescope_convert::capture
