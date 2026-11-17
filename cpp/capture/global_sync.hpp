// kind-2 (GSYNC) records -> GlobalSyncObservation + lane facts
// (python/cyclescope/convert/global_sync_observation.py verbatim port).
#pragma once
#include "capture/static_code.hpp"
#include "capture/trace_reader.hpp"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyclescope_convert::capture {

struct GlobalSyncObservation {
  int kind = 0;                 // 1 release / 2 acquire / 3 atomic
  std::int64_t address = 0;
  std::int64_t value = 0;
  int lane = 0;
  int acquire_mode = 0;
  int post_known = 0;
  std::int64_t before = 0;
  std::uint64_t sequence = 0;
};

struct GSyncObsResult {
  std::vector<GlobalSyncObservation> obs;
  std::map<std::string, std::vector<std::int64_t>> lane;  // glane
  std::vector<std::string> lane_order;
};

int observation_kind(const std::string &opcode_root, int gsync_flags);
GSyncObsResult observations(const DynRecord &rec, const StaticRow &row);

}  // namespace cyclescope_convert::capture
