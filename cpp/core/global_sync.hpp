#pragma once
// §13.10 global-sync pairing: GSYN observation rows -> an acquire
// plan the engine evaluates at run time.  Row classes: 1 atom,
// 2 red, 3 ld_strong, 4 st_strong; flags bit0 = after valid,
// bit1 = before valid.

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "job_reader.hpp"

namespace cyclescope {

struct GlobalSyncPlan {
  struct Acquire {
    // 1 from | 2 any | 3 count | 4 initial
    std::uint8_t mode = 4;
    std::vector<std::uint64_t> producers;  // from/any candidates
    std::uint64_t count = 0;               // count: kth completion
    std::uint64_t address = 0;
  };
  // acquire instance -> pairing
  std::unordered_map<std::uint64_t, Acquire> acquires;
  // producer instance -> address (completion bookkeeping)
  std::unordered_map<std::uint64_t, std::uint64_t> producer_address;
  // failed polls of a poll sequence (rule 3): every ld_strong of the
  // sequence except the winner; they issue but carry no load latency
  // and no hierarchy traffic (the paired winner models the wait)
  std::unordered_set<std::uint64_t> failed_polls;
  // poll-region folding (rule 3b): winner instance -> first failed poll
  // of its sequence; region_members lists the failed polls belonging
  // to that sequence (used to tell own polls from foreign GSYN rows)
  std::unordered_map<std::uint64_t, std::uint64_t> region_first;
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>
      region_members;
  struct Stats {
    std::uint64_t dropped_polls = 0, chains_unique = 0,
        chains_non_unique = 0, update_unknown_post = 0;
    std::uint64_t acq_from = 0, acq_any = 0, acq_count = 0,
        acq_initial = 0, acq_no_writer = 0;
  } stats;
};

GlobalSyncPlan pair_global_sync(const Job& job);

}  // namespace cyclescope
