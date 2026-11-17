#pragma once
// HopperModule — Hopper (sm_90a) arch semantics (docs/en/design/engine.md):
// wgmma collective launch + group FIFO, shared-operand occupancy,
// register-file issue steal, accumulator handoff, ldgsts completion
// adjustment, proxy_fence drain.

#include <array>
#include <deque>
#include <unordered_map>
#include <vector>

#include "arch/architecture_module.hpp"
#include "core/job_format.hpp"
#include "core/mechanism_runtime.hpp"

namespace cyclescope {

// group-3 tele counters emitted by this module
enum HopperTele : std::uint32_t {
  kWgmmaLaunches = 0, kWarpgroupWaitCycles, kOccupancyStalls,
  kStealBlocks, kHandoffWaits, kFenceBlocks, kLdgstsAdjusts,
  kProxyFenceDisabled, kProxyFenceBackpressure,
  kWgmmaPendingHolds, kHopperTeleCount
};

class HopperModule : public ArchitectureModule {
 public:
  void on_load(Simulator&) override;
  bool instance_ready(Simulator&, WarpState&,
                      const InstructionInstance&,
                      const InstanceSemantics&) override;
  void on_instance_issue(Simulator&, WarpState&,
                         const InstructionInstance&,
                         const InstanceSemantics&,
                         std::uint64_t finish_cycle) override;
  void on_event(Simulator&, const Event&) override;
  void collect_tele(std::vector<TelemetryRow>&) override;

 private:
  // one rendezvous per (cta, warpgroup): members arrive at their
  // wgmma instance; the last arrival is the collective launch cycle
  struct Rendezvous {
    std::uint32_t members[4] = {};
    std::uint32_t count = 0, issued = 0;
    std::uint64_t launch = 0, op_start = 0, beat = 0;
    bool launched = false;
  };
  // keyed by ((cta << 4 | warpgroup) << 32 | op sequence): the n-th
  // wgmma of each member warp rendezvouses with the others' n-th
  std::unordered_map<std::uint64_t, Rendezvous> rendezvous_;
  std::unordered_map<std::uint32_t, std::uint32_t> wg_seq_;
  std::vector<std::uint32_t> wg_total_;
  // wgmma commit-group admission + memory-queue hold
  // (wgmma_overlap.h100.v6, Q'/H'): per (cta, warpgroup) key
  // (cta << 4 | wg), pipe ends (op_start + beat) of launched ops,
  // ends of CLOSED commit groups, and the still-open group's max op
  // end (the open group never counts).  A launch is held while
  // pending_commit_groups_limit closed groups are pending; a held
  // launch parks the SMSPs' memory-queue classes until it launches.
  struct WgQueue {
    std::deque<std::uint64_t> op_end;
    std::deque<std::uint64_t> group_end;
    std::uint64_t open_group_end = 0;
    bool group_open = false;
    bool held_launch = false;
    bool registered = false;
    std::uint64_t mq_hold_above = 0;
  };
  // mutable: memory_queue_retry lazily pops ended op ends
  mutable std::unordered_map<std::uint64_t, WgQueue> wgq_;
  // warpgroup keys registered per SMSP (key (sm << 3) | smsp) for the
  // memory-queue hold; a key is added on the SMSP of each member warp
  // at the warpgroup's first launch
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>
      mq_keys_;
  // most recent wgmma's accumulator window per warp (handoff check)
  struct AccWindow {
    std::uint64_t begin = 0, count = 0, packed = 0, complete = 0;
    bool valid = false;
    // the producing wgmma was a QGMMA op — the fp8 fmnmx handoff
    // leaf is bound to QGMMA producers only (hopper.28c53171 origin
    // data is QGMMA n136)
    bool producer_is_qgmma = false;
    // producer-side capture facts (operation_class / datatype /
    // source_mode / tensor_*) from the wgmma template's kind-14 USES
    // row — the consumer relation program is keyed by them
    EngineFacts producer_facts;
  };
  std::vector<AccWindow> acc_;
  // open wgmma group marker per warp (GROUP_OPENER closes the
  // previous open group before recording this op)
  std::unordered_map<std::uint32_t, bool> wg_open_;
  // proxy_fence (schema 29): synchronous shared stores drain through a
  // per-SM FIFO (per-SMSP sub-queue); op-5 fences wait for the drain
  bool pf_enabled_ = false;
  double pf_latency_ = 0, pf_sm_drain_ = 0, pf_smsp_drain_ = 0;
  std::uint64_t pf_capacity_ = 0;
  std::vector<double> pf_sm_until_;
  std::vector<std::array<double, 4>> pf_smsp_until_;
  std::uint64_t tele_[kHopperTeleCount] = {};
  // wgmma register-file issue steal (protocol-v3 C3/C4): per-warp
  // issue floor CYCLES while the warp's warpgroup has ops in flight
  // (async_groups kWgmma pending), plus a per-SM in-flight op table
  // for the co-resident dependent penalty (field
  // coresident_dependent_half_cycles)
  std::vector<std::uint64_t> steal_cycles_, steal_reuse_floor_;
  std::vector<bool> steal_coresident_, steal_rs_;
  struct Inflight {
    std::uint64_t launch = 0, complete = 0;
    bool coresident = false;
  };
  std::vector<std::vector<Inflight>> sm_inflight_;
  mutable std::uint64_t steal_applied_ = 0;
  IssueSteal issue_steal(const Simulator&,
                         const WarpState&) const override;
  bool coresident_dependent_penalty(
      const Simulator&, const WarpState&) const override;

  std::uint64_t memory_queue_retry(std::uint32_t sm,
                                   std::uint32_t smsp,
                                   std::uint64_t now) const override;
  std::uint32_t warpgroup_id(const Simulator& s,
                             const WarpState& w) const;
  bool wgmma_ready(Simulator&, WarpState&,
                   const InstructionInstance&,
                   const InstanceSemantics&);
  void wgmma_issue(Simulator&, WarpState&,
                   const InstructionInstance&,
                   const InstanceSemantics&);
  bool warpgroup_wait_ready(Simulator&, WarpState&,
                            const InstanceSemantics&);
  bool handoff_ready(Simulator&, WarpState&,
                     const InstructionInstance&);
  bool sync_shared_store(const Simulator&,
                         const InstructionInstance&,
                         const InstanceSemantics&,
                         std::uint64_t* bytes) const;
  std::uint64_t release_floor(std::uint32_t sm,
                              std::uint64_t now) const override {
    std::uint64_t floor = now;
    if (sm < pf_sm_until_.size())
      floor = std::max(floor, (std::uint64_t)pf_sm_until_[sm]);
    if (sm < pf_smsp_until_.size())
      for (double u : pf_smsp_until_[sm])
        floor = std::max(floor, (std::uint64_t)u);
    return floor;
  }
};

}  // namespace cyclescope
