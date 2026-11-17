#pragma once
// ArchitectureModule — the single dispatch point for arch-specific
// semantics (docs/en/design/engine.md).  One stateful instance per
// simulation; core never branches on device names.

#include <memory>
#include <vector>

#include "core/job_format.hpp"

namespace cyclescope {

class Simulator;
struct WarpState;
struct Event;
struct TelemetryRow;

class ArchitectureModule {
 public:
  virtual ~ArchitectureModule() = default;
  // floor for barrier-release / shared-visibility waits: engines
  // with per-SM drain queues (proxy fence) return the drain cycle
  virtual std::uint64_t release_floor(std::uint32_t /*sm*/,
                                      std::uint64_t now) const {
    return now;
  }
  // once, after load_setup/place_warps
  virtual void on_load(Simulator&) {}
  // readiness of an arch-owned semantic (wgmma collective launch,
  // warpgroup wait, tmem budget, ...); may set retry_at/last_wait
  virtual bool instance_ready(Simulator&, WarpState&,
                              const InstructionInstance&,
                              const InstanceSemantics&) {
    return true;
  }
  // every issued instance, after the core commit
  virtual void on_instance_issue(Simulator&, WarpState&,
                                 const InstructionInstance&,
                                 const InstanceSemantics&,
                                 std::uint64_t /*finish_cycle*/) {}
  // per-warp issue-steal parameters while the warp's group has an
  // arch async op in flight (Hopper wgmma register-file issue steal):
  // floor for ordinary issues, reuse_floor for issues carrying a
  // .reuse source flag, rs = the in-flight op reads A from registers
  // (every other issue then pays an extra cycle)
  struct IssueSteal {
    std::uint64_t floor = 0, reuse_floor = 0;
    bool rs = false;
  };
  virtual IssueSteal issue_steal(const Simulator&,
                                 const WarpState&) const {
    return {};
  }
  // co-resident dependent-issue penalty on the same SM (Hopper wgmma
  // C4): true while another group's async op is in flight
  virtual bool coresident_dependent_penalty(const Simulator&,
                                            const WarpState&) const {
    return false;
  }
  // memory-queue hold (Hopper wgmma_overlap.h100.v6, H'): an
  // instance of a memory-queue class (MUFU/SHFL/LDS) on this SMSP
  // waits while a warpgroup with a member warp on it exceeds the
  // outstanding-op threshold or has a commit-group-held wgmma launch.
  // Returns the finite retry cycle (0 = not held, ~0 = park until the
  // SMSP is marked)
  virtual std::uint64_t memory_queue_retry(std::uint32_t /*sm*/,
                                           std::uint32_t /*smsp*/,
                                           std::uint64_t /*now*/) const {
    return 0;
  }
  // all events incl. Event::kArch (payload is module-defined)
  virtual void on_event(Simulator&, const Event&) {}
  virtual void on_warp_exit(Simulator&, WarpState&) {}
  // group-3 telemetry rows appended at result write
  virtual void collect_tele(std::vector<TelemetryRow>&) {}
};

class NullArchitectureModule : public ArchitectureModule {};

// one stateful module per simulation; unknown devices get the null
// module so cores stays arch-free
std::unique_ptr<ArchitectureModule> make_architecture(
    const char* device);

}  // namespace cyclescope
