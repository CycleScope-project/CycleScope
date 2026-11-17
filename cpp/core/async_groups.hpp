#pragma once
// Per-warp asynchronous operation groups (docs/en/design/engine.md): each warp
// owns two FIFO lanes — bulk (cp.async.bulk group ops) and cp.async.
// A lane holds an open group collecting op completions; commit closes
// it into a FIFO of group completions; wait allows at most N newer
// groups to still be pending.  Pure state machine — the Simulator
// feeds issue/commit, polls readiness, and owns wake timing.

#include <array>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

namespace cyclescope {

class AsyncGroups {
 public:
  static constexpr int kBulk = 0;
  static constexpr int kCpAsync = 1;
  static constexpr int kWgmma = 2;

  // record an op into the lane's open group; cp.async issues also
  // remember (completion, bytes) for the next ARRIVES commit
  void issue(int lane, std::uint32_t warp, std::uint64_t completion,
             std::uint64_t bytes);

  // close the open group (possibly empty -> completion = commit cycle)
  void commit(int lane, std::uint32_t warp, std::uint64_t cycle);

  // cycle at which the wait would pass: the latest completion among
  // committed groups beyond the newest `pending_allowed` (0 = ready)
  std::uint64_t earliest_ready(int lane, std::uint32_t warp,
                               std::uint64_t pending_allowed) const;

  // all committed groups except the newest `pending_allowed` have
  // completion <= cycle; on success the finished prefix is popped.
  // earliest_ready (out) = cycle at which the wait would pass.
  bool wait_ready(int lane, std::uint32_t warp,
                  std::uint64_t pending_allowed, std::uint64_t cycle,
                  std::uint64_t* earliest_ready);

  // cp.async copies recorded since the last ARRIVES (or since start)
  const std::vector<std::pair<std::uint64_t, std::uint64_t>>&
      copies_since_arrives(std::uint32_t warp) const;
  void clear_copies(std::uint32_t warp);

  // issue cycle of the first copy in the lane's OPEN group
  // (UINT64_MAX when the group is empty); recorded on first issue,
  // cleared on commit
  std::uint64_t open_first_issue(int lane, std::uint32_t warp) const;
  void set_open_first_issue(int lane, std::uint32_t warp,
                            std::uint64_t cycle);

  // latest completion among open ops and committed groups (both lanes)
  std::uint64_t drain_until(std::uint32_t warp) const;

  // groups of one lane not yet complete at `cycle`: committed groups
  // whose completion is still ahead plus the open group when it holds
  // an unfinished op (engine axis group_depth)
  std::uint64_t pending_groups(int lane, std::uint32_t warp,
                               std::uint64_t cycle) const;

 private:
  struct LaneState {
    std::vector<std::uint64_t> open_ops;
    std::deque<std::uint64_t> committed;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> copies;
    std::uint64_t first_issue = ~0ull;
  };
  LaneState& lane_of(int lane, std::uint32_t warp);
  const LaneState* find(int lane, std::uint32_t warp) const;
  // flat per-(lane, warp) table; an out-of-range warp reads as
  // never-created and every find() consumer treats an empty state
  // identically to a miss, so no created/empty distinction is needed
  std::array<std::vector<LaneState>, 3> lanes_;
};

}  // namespace cyclescope
