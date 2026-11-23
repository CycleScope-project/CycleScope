// TMEM: per-SM column budget (schema-31 columns_live else 512);
// alloc blocks when out of budget, dealloc frees and wakes waiters.

#include "arch/blackwell/module.hpp"

#include <cstdio>
#include <cstdlib>

#include "core/mechanism_runtime.hpp"
#include "core/semantics_kinds.hpp"
#include "core/simulator.hpp"

namespace cyclescope {

namespace {
// CYCLESCOPE_DEBUG_TMEM=1 traces every alloc attempt / dealloc to stderr
// (diagnostic only)
bool tmem_dbg() {
  static const bool on = std::getenv("CYCLESCOPE_DEBUG_TMEM") != nullptr;
  return on;
}
}  // namespace

bool BlackwellModule::tmem_alloc_ready(
    Simulator& s, WarpState& w, const InstanceSemantics& sema) {
  const std::uint64_t cols = sema.field[0];
  if (cols == kSemaNoObject)
    throw EngineError("missing_fact:tmem_columns (warp " +
                      std::to_string(w.warp_index) + ")");
  auto& free = tmem_free_.emplace(w.sm, tmem_budget_).first->second;
  if (tmem_dbg())
    std::fprintf(stderr,
                 "TMEM alloc cyc=%llu sm=%u cta=%u warp=%u inst=%llu "
                 "cols=%llu free=%llu hold=%llu %s\n",
                 (unsigned long long)s.now(), w.sm, w.cta, w.warp_index,
                 (unsigned long long)w.next_instance,
                 (unsigned long long)cols, (unsigned long long)free,
                 (unsigned long long)tmem_hold_[w.cta],
                 free < cols ? "BLOCK" : "ok");
  if (free < cols) {
    tmem_waiters_.push_back(w.warp_index);
    w.last_wait = WaitBucket::kResource;
    w.retry_at = 0;  // woken by dealloc on this SM
    ++tele_[kTmemBlocks];
    stall_detail_ = "tmem_alloc";
    return false;
  }
  free -= cols;
  tmem_hold_[w.cta] += cols;
  ++tele_[kTmemAllocs];
  return true;
}

void BlackwellModule::tmem_dealloc(
    Simulator& s, WarpState& w, const InstanceSemantics& sema) {
  const std::uint64_t cols = sema.field[0];
  std::uint64_t freed = 0;
  if (cols == kSemaNoObject) {
    freed = tmem_hold_[w.cta];
    tmem_hold_[w.cta] = 0;
  } else {
    freed = std::min(cols, tmem_hold_[w.cta]);
    tmem_hold_[w.cta] -= freed;
  }
  // an SM whose first TMEM event is a dealloc (the non-allocating CTA
  // of a cta_group::2 pair frees 0) must start from the budget, not 0
  auto& free = tmem_free_.emplace(w.sm, tmem_budget_).first->second;
  free += freed;
  if (tmem_dbg())
    std::fprintf(stderr,
                 "TMEM dealloc cyc=%llu sm=%u cta=%u warp=%u inst=%llu "
                 "cols=%llu freed=%llu free=%llu\n",
                 (unsigned long long)s.now(), w.sm, w.cta, w.warp_index,
                 (unsigned long long)w.next_instance,
                 (unsigned long long)cols, (unsigned long long)freed,
                 (unsigned long long)free);
  // wake warps parked on an exhausted budget in this SM
  for (const auto wi : tmem_waiters_) {
    auto& w2 = s.warp(wi);
    if (!w2.done && w2.sm == w.sm) s.mark_warp_for_eval(w2);
  }
}

}  // namespace cyclescope
