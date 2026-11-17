// Per-warp asynchronous operation groups (docs/en/design/engine.md).

#include "async_groups.hpp"

#include <algorithm>

namespace cyclescope {

AsyncGroups::LaneState& AsyncGroups::lane_of(int lane,
                                             std::uint32_t warp) {
  auto& v = lanes_[lane];
  if (warp >= v.size()) v.resize(warp + 1);
  return v[warp];
}

const AsyncGroups::LaneState* AsyncGroups::find(
    int lane, std::uint32_t warp) const {
  const auto& v = lanes_[lane];
  return warp < v.size() ? &v[warp] : nullptr;
}

void AsyncGroups::issue(int lane, std::uint32_t warp,
                        std::uint64_t completion, std::uint64_t bytes) {
  auto& l = lane_of(lane, warp);
  l.open_ops.push_back(completion);
  if (lane == kCpAsync) l.copies.push_back({completion, bytes});
}

void AsyncGroups::commit(int lane, std::uint32_t warp,
                         std::uint64_t cycle) {
  auto& l = lane_of(lane, warp);
  std::uint64_t completion = cycle;
  for (std::uint64_t c : l.open_ops) completion = std::max(completion, c);
  l.committed.push_back(completion);
  l.open_ops.clear();
  l.first_issue = ~0ull;
}

std::uint64_t AsyncGroups::earliest_ready(
    int lane, std::uint32_t warp, std::uint64_t pending_allowed) const {
  const auto* l = find(lane, warp);
  std::uint64_t earliest = 0;
  if (l && l->committed.size() > pending_allowed) {
    const std::size_t must_done = l->committed.size() - pending_allowed;
    for (std::size_t i = 0; i < must_done; ++i)
      earliest = std::max(earliest, l->committed[i]);
  }
  return earliest;
}

bool AsyncGroups::wait_ready(int lane, std::uint32_t warp,
                             std::uint64_t pending_allowed,
                             std::uint64_t cycle,
                             std::uint64_t* earliest_ready) {
  const std::uint64_t earliest =
      this->earliest_ready(lane, warp, pending_allowed);
  if (earliest_ready) *earliest_ready = earliest;
  if (earliest > cycle) return false;
  auto& m = lane_of(lane, warp);
  while (m.committed.size() > pending_allowed) m.committed.pop_front();
  return true;
}

const std::vector<std::pair<std::uint64_t, std::uint64_t>>&
    AsyncGroups::copies_since_arrives(std::uint32_t warp) const {
  static const std::vector<std::pair<std::uint64_t, std::uint64_t>>
      empty;
  const auto* l = find(kCpAsync, warp);
  return l ? l->copies : empty;
}

void AsyncGroups::clear_copies(std::uint32_t warp) {
  lane_of(kCpAsync, warp).copies.clear();
}

std::uint64_t AsyncGroups::pending_groups(int lane, std::uint32_t warp,
                                          std::uint64_t cycle) const {
  const auto* l = find(lane, warp);
  if (!l) return 0;
  std::uint64_t n = 0;
  for (std::uint64_t c : l->committed)
    if (c > cycle) ++n;
  for (std::uint64_t c : l->open_ops)
    if (c > cycle) { ++n; break; }
  return n;
}

std::uint64_t AsyncGroups::drain_until(std::uint32_t warp) const {
  std::uint64_t until = 0;
  for (int lane = kBulk; lane <= kCpAsync; ++lane) {
    const auto* l = find(lane, warp);
    if (!l) continue;
    for (std::uint64_t c : l->open_ops) until = std::max(until, c);
    for (std::uint64_t c : l->committed) until = std::max(until, c);
  }
  return until;
}

std::uint64_t AsyncGroups::open_first_issue(int lane,
                                            std::uint32_t warp) const {
  const LaneState* l = find(lane, warp);
  return l ? l->first_issue : ~0ull;
}

void AsyncGroups::set_open_first_issue(int lane, std::uint32_t warp,
                                       std::uint64_t cycle) {
  lane_of(lane, warp).first_issue = cycle;
}

}  // namespace cyclescope
