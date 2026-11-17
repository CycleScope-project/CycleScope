#include "arch/blackwell/module.hpp"

#include "core/mechanism_runtime.hpp"
#include "core/semantics_kinds.hpp"
#include "core/simulator.hpp"
#include "core/generated/parameter_schema.hpp"

namespace TR = cyclescope::core::parameter_fields::TIMING_ROW;

namespace cyclescope {

double BlackwellModule::kind_value(
    Simulator& s, const InstructionInstance& inst,
    std::uint32_t kind) const {
  const auto* f = s.template_timing_fields(
      inst.template_index, kind, s.engine_facts());
  if (!f || MechanismRuntime::field(*f, TR::ENABLED, 1.0) == 0.0)
    return 0;
  return MechanismRuntime::field(*f, TR::CYCLES, 0.0);
}

std::uint64_t BlackwellModule::kind_cycles(
    Simulator& s, const InstructionInstance& inst,
    std::uint32_t kind) const {
  return static_cast<std::uint64_t>(kind_value(s, inst, kind));
}

void BlackwellModule::on_load(Simulator& s) {
  for (std::uint64_t ui = 0; ui < s.job().uses.size(); ++ui) {
    const auto& u = s.job().uses[ui];
    if (u.schema_id != 31) continue;
    const auto& f = s.rt().resolve(ui, s.engine_facts(), nullptr);
    if (f.count(0))
      tmem_budget_ = static_cast<std::uint64_t>(
          MechanismRuntime::field(f, 0, 512.0));
  }
}

bool BlackwellModule::instance_ready(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema) {
  switch (sema.kind) {
    case kSemaTmemAlloc:
      return tmem_alloc_ready(s, w, sema);
    case kSemaTmemWaitStore: {
      const auto it = sttm_until_.find(w.warp_index);
      if (it != sttm_until_.end() && it->second > s.now()) {
        w.last_wait = WaitBucket::kCollective;
        w.retry_at = it->second;
        ++tele_[kTmemWaitStalls];
        return false;
      }
      return true;
    }
    default:
      return true;
  }
}

void BlackwellModule::on_instance_issue(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema, std::uint64_t finish_cycle) {
  switch (sema.kind) {
    case kSemaUmmaIssue:
      umma_issue(s, w, inst, sema, finish_cycle);
      break;
    case kSemaUmmaCommit:
      umma_commit(s, w, inst, sema);
      break;
    case kSemaTmemDealloc:
      tmem_dealloc(s, w, sema);
      break;
    case kSemaTmemRelinquish:
      break;  // alloc permission only; no time semantics yet
    case kSemaTmemStore:
    case kSemaTmemCopy: {
      const bool copy = sema.kind == kSemaTmemCopy;
      const std::uint64_t done = copy
          ? std::max(finish_cycle, s.tensor_operand_path_finish(
                w.sm, s.now(), kind_value(s, inst, 55)))
          : finish_cycle + kind_cycles(s, inst, 53);
      if (copy) umma_done_[w.warp_index].push_back(done);
      auto& u = sttm_until_[w.warp_index];
      u = std::max(u, done);
      break;
    }
    case kSemaTmemLoad: {
      const std::uint64_t done =
          finish_cycle + kind_cycles(s, inst, 52);
      auto& u = sttm_until_[w.warp_index];
      u = std::max(u, done);
      break;
    }
    case kSemaClcQuery:
      clc_query(s, w, inst, sema);
      break;
    default:
      break;
  }
}

void BlackwellModule::on_event(Simulator& s, const Event& e) {
  if (e.kind != Event::kArch) return;
  const std::uint32_t object =
      static_cast<std::uint32_t>(e.payload & 0xFFFFFFFFull);
  const std::uint32_t mask =
      static_cast<std::uint32_t>((e.payload >> 32) & 0xFFFFFFull);
  const std::uint8_t op = static_cast<std::uint8_t>(e.payload >> 56);
  const auto& w = s.warp(e.warp);
  const std::uint32_t seg = s.seg_of(w);
  const std::uint32_t issuer_rank =
      s.job().ctas[w.cta].cluster_rank;
  auto fire = [&](std::uint32_t rank) {
    const auto [sg, ob] =
        s.resolve_destination(seg, object, rank, issuer_rank);
    if (op == kArrive1) {
      s.mbarrier_arrive(sg, ob, 1, 0, e.warp);
    } else if (op == kTx16) {
      s.mbarrier_complete_tx(sg, ob, 16);
    } else {
      // STAS: payload mask field carries the rank; bytes are in
      // pending_ keyed by payload
      const auto it = pending_.find(e.payload);
      const std::uint64_t bytes =
          it == pending_.end() ? 0 : it->second.second;
      s.mbarrier_complete_tx(sg, ob, bytes);
      if (it != pending_.end()) pending_.erase(it);
    }
  };
  if (mask == 0 || op == kTxN) {
    fire(op == kTxN ? mask : issuer_rank);
    return;
  }
  for (std::uint32_t r = 0; r < 24; ++r)
    if (mask >> r & 1) fire(r);
}

void BlackwellModule::collect_tele(std::vector<TelemetryRow>& tele) {
  for (std::uint32_t c = 0; c < kBlackwellTeleCount; ++c)
    if (tele_[c]) tele.push_back({3, 16 + c, tele_[c]});
}

}  // namespace cyclescope
