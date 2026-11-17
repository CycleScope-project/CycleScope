#include "arch/hopper/module.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "core/semantics_kinds.hpp"
#include "core/simulator.hpp"
#include "core/generated/parameter_schema.hpp"

namespace TR = cyclescope::core::parameter_fields::TIMING_ROW;
namespace PF = cyclescope::core::parameter_fields::HOPPER_PROXY_FENCE;

namespace cyclescope {

void HopperModule::on_load(Simulator& s) {
  acc_.assign(s.job().warps.size(), AccWindow{});
  // wgmma ops per warp: a live warp with no wgmma left is not an
  // expected rendezvous member
  wg_total_.assign(s.job().warps.size(), 0);
  for (std::size_t wi = 0; wi < s.job().warps.size(); ++wi) {
    const auto& info = s.job().warps[wi];
    for (std::uint64_t i = info.first_instance;
         i < info.first_instance + info.instance_count; ++i) {
      const auto& inst = s.job().instances[i];
      if (inst.semantics != kNone32 &&
          s.job().semantics[inst.semantics].kind == kSemaWgmma)
        ++wg_total_[wi];
    }
  }
  pf_sm_until_.assign(s.job().launch.sm_count, 0.0);
  pf_smsp_until_.assign(s.job().launch.sm_count, {});
  steal_cycles_.assign(s.job().warps.size(), 0);
  steal_reuse_floor_.assign(s.job().warps.size(), 0);
  steal_coresident_.assign(s.job().warps.size(), false);
  steal_rs_.assign(s.job().warps.size(), false);
  sm_inflight_.assign(s.job().launch.sm_count, {});
  // schema-29 proxy_fence leaf (initialization USES)
  for (std::uint64_t ui = 0; ui < s.job().uses.size(); ++ui) {
    const auto& u = s.job().uses[ui];
    if (u.schema_id != 29) continue;
    const auto& f = s.rt().resolve(ui, s.engine_facts(), nullptr);
    pf_enabled_ = f.count(PF::ENABLED) &&
        MechanismRuntime::field(f, PF::ENABLED, 0.0) != 0.0;
    pf_latency_ = MechanismRuntime::field(f, PF::LATENCY_CYCLES, 0.0);
    pf_sm_drain_ = MechanismRuntime::field(
        f, PF::SM_DRAIN_BYTES_PER_CYCLE, 0.0);
    pf_smsp_drain_ = MechanismRuntime::field(
        f, PF::SMSP_DRAIN_BYTES_PER_CYCLE, 0.0);
    pf_capacity_ = static_cast<std::uint64_t>(MechanismRuntime::field(
        f, PF::QUEUE_CAPACITY_BYTES, 0.0));
  }
  if (!pf_enabled_) ++tele_[kProxyFenceDisabled];
}

// proxy_fence inflight object = a synchronous shared store: MACC
// shared space (flags&7 == 1) + store (0x40), excluding async stores
// (STAS/TMA/bulk); bytes = sum of sector byte_mask popcounts
bool HopperModule::sync_shared_store(
    const Simulator& s, const InstructionInstance& inst,
    const InstanceSemantics& sema, std::uint64_t* bytes) const {
  return s.sync_shared_store(inst, sema, bytes);
}

bool HopperModule::instance_ready(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema) {
  if (sema.kind == kSemaWgmma) return wgmma_ready(s, w, inst, sema);
  if (sema.kind == kSemaWarpgroupWait)
    return warpgroup_wait_ready(s, w, sema);
  // proxy_fence queue-capacity backpressure on shared stores
  if (pf_enabled_ && pf_capacity_ && pf_sm_drain_ > 0 &&
      w.sm < pf_sm_until_.size()) {
    std::uint64_t bytes = 0;
    if (sync_shared_store(s, inst, sema, &bytes)) {
      const double pending =
          (pf_sm_until_[w.sm] - static_cast<double>(s.now())) *
          pf_sm_drain_;
      if (pending > static_cast<double>(pf_capacity_)) {
        // cycle the backlog drains back below the capacity
        const std::uint64_t retry = static_cast<std::uint64_t>(
            std::ceil(pf_sm_until_[w.sm] -
                      static_cast<double>(pf_capacity_) /
                          pf_sm_drain_));
        w.retry_at = retry;
        ++tele_[kProxyFenceBackpressure];
        return false;
      }
    }
  }
  return handoff_ready(s, w, inst);
}

// wgmma issue steal (protocol-v5 C3+C7+R): the warp issues under a
// per-instruction floor while its own warpgroup has a wgmma op in
// flight; a .reuse-flagged issue uses reuse_floor, and rs ops add an
// alternating extra cycle handled by the caller
ArchitectureModule::IssueSteal HopperModule::issue_steal(
    const Simulator& s, const WarpState& w) const {
  IssueSteal out;
  if (w.warp_index >= steal_cycles_.size() ||
      steal_cycles_[w.warp_index] == 0 ||
      s.async_groups().pending_groups(AsyncGroups::kWgmma,
                                      w.warp_index, s.now()) == 0)
    return out;
  ++steal_applied_;
  out.floor = steal_cycles_[w.warp_index];
  out.reuse_floor = steal_reuse_floor_[w.warp_index];
  out.rs = steal_rs_[w.warp_index];
  return out;
}

// wgmma co-resident dependent penalty (protocol-v3 C4): warps of
// OTHER warpgroups pay +half-cycle on a dependency-triggered issue
// while an op marked coresident is in flight on the SM
bool HopperModule::coresident_dependent_penalty(
    const Simulator& s, const WarpState& w) const {
  if (w.sm >= sm_inflight_.size()) return false;
  if (s.async_groups().pending_groups(AsyncGroups::kWgmma,
                                      w.warp_index, s.now()) != 0)
    return false;
  for (const auto& e : sm_inflight_[w.sm])
    if (e.coresident && e.complete > s.now()) return true;
  return false;
}

void HopperModule::on_instance_issue(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema, std::uint64_t finish_cycle) {
  switch (sema.kind) {
    case kSemaWgmma:
      wgmma_issue(s, w, inst, sema);
      return;
    case kSemaWarpgroupArrive:
      wg_open_[w.warp_index] = true;
      return;
    case kSemaDescriptorVisibility:
      if (sema.field[0] == 5 && pf_enabled_) {
        // FENCE.VIEW.ASYNC.S / proxy fence: block until the SM's
        // (and this SMSP's) queued shared stores have drained
        double drain = static_cast<double>(s.now());
        if (w.sm < pf_sm_until_.size()) {
          drain = std::max(drain, pf_sm_until_[w.sm]);
          drain = std::max(drain, pf_smsp_until_[w.sm][w.smsp % 4]);
        }
        s.block_warp(w, static_cast<std::uint64_t>(
            std::ceil(drain + pf_latency_)),
                     WaitBucket::kCollective);
        ++tele_[kFenceBlocks];
      }
      return;
    default:
      break;
  }
  // synchronous shared stores enter the proxy_fence drain FIFO
  if (pf_enabled_ && w.sm < pf_sm_until_.size()) {
    std::uint64_t bytes = 0;
    if (sync_shared_store(s, inst, sema, &bytes) && bytes) {
      const double now = static_cast<double>(s.now());
      if (pf_sm_drain_ > 0)
        pf_sm_until_[w.sm] = std::max(now, pf_sm_until_[w.sm]) +
            static_cast<double>(bytes) / pf_sm_drain_;
      if (pf_smsp_drain_ > 0) {
        auto& q = pf_smsp_until_[w.sm][w.smsp % 4];
        q = std::max(now, q) +
            static_cast<double>(bytes) / pf_smsp_drain_;
      }
    }
  }
}

void HopperModule::on_event(Simulator&, const Event&) {}

void HopperModule::collect_tele(std::vector<TelemetryRow>& tele) {
  tele_[kStealBlocks] = steal_applied_;
  for (std::uint32_t c = 0; c < kHopperTeleCount; ++c)
    if (tele_[c]) tele.push_back({3, c, tele_[c]});
}

}  // namespace cyclescope
