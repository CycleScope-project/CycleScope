#include "global_sync.hpp"
#include "job_format.hpp"
#include "mechanism_runtime.hpp"

#include <algorithm>
#include <map>
#include <set>

namespace cyclescope {

namespace {
constexpr std::uint8_t kAtom = 1, kRed = 2, kLdStrong = 3,
    kStStrong = 4;
}  // namespace

GlobalSyncPlan pair_global_sync(const Job& job) {
  GlobalSyncPlan plan;
  // coverage: grid 0 means the launch geometry is unknown -> treat as
  // not covered (fail closed)
  const std::uint64_t grid = static_cast<std::uint64_t>(
      job.launch.grid_x) * job.launch.grid_y * job.launch.grid_z;
  const bool full_coverage = grid > 0 && job.ctas.size() == grid;
  // rows grouped per address, ordered by sequence
  std::map<std::uint64_t, std::vector<const GlobalSyncRow*>> by_addr;
  for (const auto& g : job.global_sync) {
    if (g.segment >= job.segment_instance_base.size() ||
        g.instance >= job.instances.size())
      continue;  // validate_job reports these as errors
    by_addr[g.address].push_back(&g);
  }
  // instance -> warp index, built lazily on the first address that
  // needs it (was rebuilt per address: O(#addresses x #instances))
  std::vector<std::pair<std::uint64_t, std::uint32_t>> inst_warp;
  bool inst_warp_built = false;
  for (auto& [addr, rows] : by_addr) {
    std::sort(rows.begin(), rows.end(),
              [](const GlobalSyncRow* a, const GlobalSyncRow* b) {
                return a->sequence < b->sequence;
              });
    const auto after_v = [](const GlobalSyncRow* r) {
      return (r->flags & 1) != 0;
    };
    const auto before_v = [](const GlobalSyncRow* r) {
      return (r->flags & 2) != 0;
    };
    // producers: update rows with a *changing* observed after; a
    // before==after update is a read (failed CAS / ADD 0)
    std::vector<const GlobalSyncRow*> producers;
    std::vector<const GlobalSyncRow*> updates;
    bool unknown_writer = false;   // update row with invalid after
    for (const auto* r : rows) {
      if (r->class_ == kLdStrong) continue;
      if (!after_v(r)) {
        ++plan.stats.update_unknown_post;
        unknown_writer = true;
        continue;
      }
      if (before_v(r) && r->before == r->after) continue;
      producers.push_back(r);
    }
    // no writer on this address: no producer and no update whose
    // after is unknown (before==after no-ops are not writers)
    const bool no_writer = producers.empty() && !unknown_writer;
    // chain: a producer's before links it to the unique row whose
    // after matches; unique chain -> exactly one start (no
    // predecessor), every before consumed once, walk covers all
    std::unordered_map<std::uint64_t, std::vector<const GlobalSyncRow*>>
        by_before, by_after;
    for (const auto* p : producers) {
      if (before_v(p)) by_before[p->before].push_back(p);
      by_after[p->after].push_back(p);
    }
    bool unique = true;
    const GlobalSyncRow* start = nullptr;
    {
      for (const auto* p : producers) {
        // has a predecessor?  its before equals some producer's after
        const auto it = by_after.find(p->before);
        const bool has_pred =
            before_v(p) && it != by_after.end() &&
            !(it->second.size() == 1 && it->second[0] == p);
        if (!has_pred) {
          if (start) { unique = false; break; }
          start = p;
        }
      }
      if (unique) {
        if (!start) unique = false;
        else {
          std::set<const GlobalSyncRow*> seen;
          const GlobalSyncRow* cur = start;
          while (cur) {
            if (!seen.insert(cur).second) { unique = false; break; }
            const auto it = by_before.find(cur->after);
            cur = it == by_before.end() || it->second.size() != 1
                      ? nullptr
                      : it->second[0];
          }
          if (seen.size() != producers.size()) unique = false;
        }
      }
    }
    if (unique && start) ++plan.stats.chains_unique;
    else if (!producers.empty()) ++plan.stats.chains_non_unique;
    for (const auto* p : producers)
      plan.producer_address[p->instance] = addr;

    // poll sequences: same (warp,lane) program-ordered ld_strong rows
    // with equal (template, address); all but the last are dropped
    std::vector<const GlobalSyncRow*> reads;
    {
      std::map<std::pair<std::uint32_t, std::uint32_t>,
               std::vector<const GlobalSyncRow*>> lane_reads;
      // map instances to warps once per job (not per address):
      // sorted (first_instance, warp) index + binary search
      auto warp_of = [&](std::uint64_t i) -> std::uint32_t {
        if (!inst_warp_built) {
          inst_warp_built = true;
          inst_warp.reserve(job.warps.size());
          for (std::uint32_t w = 0; w < job.warps.size(); ++w)
            inst_warp.emplace_back(job.warps[w].first_instance, w);
          std::sort(inst_warp.begin(), inst_warp.end());
        }
        auto it = std::upper_bound(
            inst_warp.begin(), inst_warp.end(),
            std::pair<std::uint64_t, std::uint32_t>{i, ~0u});
        if (it == inst_warp.begin()) return kNone32;
        --it;
        const auto& w = job.warps[it->second];
        return i < w.first_instance + w.instance_count ? it->second
                                                      : kNone32;
      };
      for (const auto* r : rows)
        if (r->class_ == kLdStrong) {
          const std::uint32_t w = warp_of(r->instance);
          if (w != kNone32) lane_reads[{w, r->lane}].push_back(r);
        }
      for (auto& [lane, seq] : lane_reads) {
        std::sort(seq.begin(), seq.end(),
                  [](const GlobalSyncRow* a, const GlobalSyncRow* b) {
                    return a->instance < b->instance;
                  });
        std::size_t run = 0;
        auto flush = [&](std::size_t end) {
          if (end == run + 1) reads.push_back(seq[run]);
          else {
            plan.stats.dropped_polls += end - run - 1;
            auto& members = plan.region_members[seq[end - 1]->instance];
            for (std::size_t k = run; k + 1 < end; ++k) {
              plan.failed_polls.insert(seq[k]->instance);
              members.push_back(seq[k]->instance);
            }
            plan.region_first[seq[end - 1]->instance] =
                seq[run]->instance;
            reads.push_back(seq[end - 1]);  // last is the winner
          }
          run = end;
        };
        for (std::size_t i = run + 1; i <= seq.size(); ++i) {
          const bool same = i < seq.size() &&
              job.instances[seq[i]->instance].template_index ==
                  job.instances[seq[run]->instance].template_index;
          if (!same) flush(i);
        }
      }
    }

    // pairing per successful read
    for (const auto* r : reads) {
      GlobalSyncPlan::Acquire acq;
      acq.address = addr;
      const std::uint64_t v = after_v(r) ? r->after : 0;
      std::vector<const GlobalSyncRow*> cands, later;
      for (const auto* p : producers)
        if (p->after == v)
          (p->sequence < r->sequence ? cands : later).push_back(p);
      if (cands.size() == 1) {
        acq.mode = 1;
        acq.producers = {cands[0]->instance};
        ++plan.stats.acq_from;
      } else if (cands.size() >= 2) {
        acq.mode = 2;
        for (const auto* p : cands) acq.producers.push_back(p->instance);
        ++plan.stats.acq_any;
      } else if (!later.empty()) {
        acq.mode = 4;
        ++plan.stats.acq_initial;
      } else if (unique && start && start->before == v) {
        acq.mode = 4;  // read the pre-write value == chain init
        ++plan.stats.acq_initial;
      } else {
        // red-only counter with a unique stride -> count acquire
        const bool red_only = !producers.empty() &&
            std::all_of(producers.begin(), producers.end(),
                        [](const GlobalSyncRow* p) {
                          return p->class_ == kRed;
                        });
        bool counted = false;
        if (red_only) {
          std::uint64_t stride = 0; bool stride_ok = true;
          for (const auto* p : producers) {
            if (!before_v(p) || p->after <= p->before) {
              stride_ok = false;
              break;
            }
            const std::uint64_t d = p->after - p->before;
            if (!stride) stride = d;
            else if (stride != d) { stride_ok = false; break; }
          }
          if (stride_ok && stride > 0) {
            // counter init = the smallest before among the reds
            // (the value no producer produced)
            bool any_before = false;
            std::uint64_t init = 0;
            for (const auto* p : producers)
              if (before_v(p)) {
                if (!any_before || p->before < init) init = p->before;
                any_before = true;
              }
            if (!any_before)
              throw EngineError(
                  "missing_fact:global_counter_initial_value (address " +
                  std::to_string(addr) + ")");
            const std::int64_t diff =
                static_cast<std::int64_t>(v) -
                static_cast<std::int64_t>(init);
            const std::uint64_t k =
                diff >= 0 ? static_cast<std::uint64_t>(diff) / stride
                          : ~0ull;
            if (diff < 0 || diff % stride ||
                k > producers.size())
              throw EngineError(
                  "unsupported:engine:inter_cta_global_sync:"
                  "counter_shape (address " +
                  std::to_string(addr) + ")");
            acq.mode = 3;
            acq.count = k;
            ++plan.stats.acq_count;
            counted = true;
          }
        }
        if (!counted) {
          if (!unique && !producers.empty()) {
            acq.mode = 4;
            ++plan.stats.acq_initial;
          } else if (no_writer && full_coverage) {
            // strong read with no writer on the address and full CTA
            // coverage: value came from pre-launch memory or a weak
            // store; ordering is carried by other sync objects
            acq.mode = 4;
            ++plan.stats.acq_no_writer;
          } else {
            throw EngineError(
                "unsupported:engine:inter_cta_global_sync:"
                "unpaired_read (address " +
                std::to_string(addr) + ")");
          }
        }
      }
      plan.acquires[r->instance] = std::move(acq);
    }
  }
  return plan;
}

}  // namespace cyclescope
