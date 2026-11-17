// wgmma: collective launch rendezvous, shared per-SM tensor pipe
// (tensor.pipe / tensor.completion), group FIFO (AsyncGroups kWgmma
// lane), register-file issue steal, accumulator handoff consumer.

#include "arch/hopper/module.hpp"

#include "core/semantics_kinds.hpp"
#include "core/simulator.hpp"
#include "core/generated/parameter_schema.hpp"

namespace TR = cyclescope::core::parameter_fields::TIMING_ROW;
namespace IT =
    cyclescope::core::parameter_fields::INSTRUCTION_TEMPLATE;

namespace cyclescope {

namespace {
constexpr std::uint32_t kKindSteal = 44;
constexpr std::uint32_t kKindHandoffConsumer = 13;
}

std::uint32_t HopperModule::warpgroup_id(const Simulator& s,
                                         const WarpState& w) const {
  return s.job().warps[w.warp_index].warp_in_cta / 4;
}



bool HopperModule::wgmma_ready(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema) {
  const std::uint32_t wg = warpgroup_id(s, w);
  const std::uint64_t key =
      ((static_cast<std::uint64_t>(w.cta) << 4 | wg) << 32) |
      wg_seq_[w.warp_index];
  auto& r = rendezvous_[key];
  if (r.launched) return true;
  // expected members: live warps of this warpgroup in the CTA that
  // still have a wgmma to issue
  const auto& cw = s.warps_of_cta(w.cta);
  std::uint32_t expected = 0;
  for (const auto wi : cw) {
    const auto& m = s.warp(wi);
    if (!m.done && s.job().warps[wi].warp_in_cta / 4 == wg &&
        (wi >= wg_total_.size() || wg_seq_[wi] < wg_total_[wi]))
      ++expected;
  }
  // register this warp's arrival once
  bool seen = false;
  for (std::uint32_t i = 0; i < r.count; ++i)
    if (r.members[i] == w.warp_index) seen = true;
  if (!seen && r.count < 4) r.members[r.count++] = w.warp_index;
  if (r.count < expected) {
    w.last_wait = WaitBucket::kCollective;
    w.retry_at = 0;  // woken when the last member arrives
    return false;
  }
  // kind-44 admission fields (wgmma_overlap.h100.v6): absent or 0
  // keeps the pre-v6 behaviour
  std::uint64_t group_limit = 0;
  bool mq_block = false;
  std::uint64_t mq_hold_above = 0;
  if (const auto* f44 = s.template_timing_fields(
          inst.template_index, kKindSteal, s.engine_facts())) {
    group_limit = static_cast<std::uint64_t>(MechanismRuntime::field(
        *f44, TR::PENDING_COMMIT_GROUPS_LIMIT, 0.0));
    mq_block = MechanismRuntime::field(
        *f44, TR::LAUNCH_HOLD_BLOCKS_MEMORY_QUEUE, 0.0) != 0.0;
    mq_hold_above = static_cast<std::uint64_t>(MechanismRuntime::field(
        *f44, TR::MEMORY_QUEUE_HOLD_ABOVE_PENDING_OPS, 0.0));
  }
  // commit-group admission (Q'): a launch waits while group_limit
  // CLOSED commit groups of this warpgroup are pending (latest pipe
  // end of their ops still ahead), retrying at the earliest pending
  // group's end.  The still-open group never counts.
  auto& q = wgq_[(std::uint64_t(w.cta) << 4) | wg];
  if (mq_block && mq_hold_above > q.mq_hold_above)
    q.mq_hold_above = mq_hold_above;
  while (!q.group_end.empty() && q.group_end.front() <= s.now())
    q.group_end.pop_front();
  if (group_limit && q.group_end.size() >= group_limit) {
    w.last_wait = WaitBucket::kResource;
    w.retry_at = q.group_end.front();
    q.held_launch = true;
    ++tele_[kWgmmaPendingHolds];
    return false;
  }
  // collective launch: this cycle is the launch for all members; the
  // op then takes the SM's shared tensor pipe (tensor.pipe, kind 63)
  r.launched = true;
  r.launch = s.now();
  r.op_start = s.now();
  if (const auto* f = s.template_timing_fields(
          inst.template_index, kTimingTensorPipeBeat, s.engine_facts())) {
    if (MechanismRuntime::field(*f, TR::ENABLED, 1.0) != 0.0) {
      r.beat = static_cast<std::uint64_t>(
          MechanismRuntime::field(*f, TR::CYCLES, 0.0));
      r.op_start = s.tensor_pipe_start(w.sm, r.launch, r.beat);
      tele_[kOccupancyStalls] += r.op_start - r.launch;
    }
  }
  // once per collective op: push the op's pipe end, fold the group
  // bookkeeping (the same boundaries wgmma_issue commits:
  // GROUP_OPENER closes the still-open group, GROUP_COMMIT closes
  // this one), register the warpgroup on the members' SMSPs for the
  // memory-queue hold, and clear a held launch (marking the SMSPs
  // wakes the parked queue-class warps)
  if (group_limit || (mq_block && mq_hold_above)) {
    const std::uint64_t end = r.op_start + r.beat;
    if ((sema.flags & kSemaFlagGroupOpener) && q.group_open) {
      q.group_end.push_back(q.open_group_end);
      q.group_open = false;
      q.open_group_end = 0;
    }
    q.group_open = true;
    q.open_group_end = std::max(q.open_group_end, end);
    q.op_end.push_back(end);
    if (sema.flags & kSemaFlagGroupCommit) {
      q.group_end.push_back(q.open_group_end);
      q.group_open = false;
      q.open_group_end = 0;
    }
    if (mq_block && !q.registered) {
      q.registered = true;
      for (std::uint32_t i = 0; i < r.count; ++i) {
        const auto& mw = s.warp(r.members[i]);
        auto& keys =
            mq_keys_[(std::uint64_t(mw.sm) << 3) | mw.smsp];
        const std::uint64_t wgk = (std::uint64_t(w.cta) << 4) | wg;
        if (std::find(keys.begin(), keys.end(), wgk) == keys.end())
          keys.push_back(wgk);
      }
    }
  }
  q.held_launch = false;
  ++tele_[kWgmmaLaunches];
  for (std::uint32_t i = 0; i < r.count; ++i)
    if (r.members[i] != w.warp_index)
      s.mark_warp_for_eval(s.warp(r.members[i]));
  return true;
}

void HopperModule::wgmma_issue(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema) {
  const std::uint64_t key =
      ((static_cast<std::uint64_t>(w.cta) << 4 | warpgroup_id(s, w))
       << 32) | wg_seq_[w.warp_index];
  auto& r = rendezvous_[key];
  const auto& t = s.job().templates[inst.template_index];
  // op completion = pipe start + beat + tensor.completion (kind 64);
  // without a pipe row the template's service completion latency
  std::uint64_t latency = r.beat;
  if (const auto* f = s.template_timing_fields(
          inst.template_index, kTimingTensorResultLatency,
          s.engine_facts())) {
    if (MechanismRuntime::field(*f, TR::ENABLED, 1.0) != 0.0)
      latency += static_cast<std::uint64_t>(
          MechanismRuntime::field(*f, TR::CYCLES, 0.0));
  }
  if (latency == 0)
    latency = static_cast<std::uint64_t>(MechanismRuntime::field(
        s.template_service_fields(t), IT::COMPLETION_LATENCY_CYCLES, 4.0));
  // GROUP_OPENER closes the still-open previous group first
  if ((sema.flags & kSemaFlagGroupOpener) && wg_open_[w.warp_index]) {
    s.async_groups().commit(AsyncGroups::kWgmma, w.warp_index,
                            s.now());
    wg_open_[w.warp_index] = false;
  }
  wg_open_[w.warp_index] = true;
  const std::uint64_t complete = r.op_start + latency;
  // register_file_issue_steal (kind 44, protocol-v3): per-warp issue
  // floor while this warpgroup has ops in flight; the first member
  // also registers the op into the SM's in-flight table for the
  // co-resident dependent penalty
  if (const auto* f = s.template_timing_fields(
          inst.template_index, kKindSteal, s.engine_facts())) {
    if (MechanismRuntime::field(*f, TR::ENABLED, 0.0) != 0.0) {
      if (w.warp_index < steal_cycles_.size()) {
        steal_cycles_[w.warp_index] = static_cast<std::uint64_t>(
            MechanismRuntime::field(*f, TR::CYCLES, 0.0));
        steal_reuse_floor_[w.warp_index] = static_cast<std::uint64_t>(
            MechanismRuntime::field(*f, TR::REUSE_FLAG_FLOOR_CYCLES,
                                    0.0));
        steal_coresident_[w.warp_index] =
            MechanismRuntime::field(
                *f, TR::CORESIDENT_DEPENDENT_HALF_CYCLES, 0.0) != 0.0;
        bool rs = false;
        if (MechanismRuntime::field(*f, TR::RS_EXTRA_HALF_CYCLES,
                                    0.0) != 0.0) {
          if (const auto sid = s.rt().axis_id("source_mode")) {
            const EngineFacts cf = s.template_timing_capture_facts(
                inst.template_index, kKindSteal);
            for (const auto& kv : cf)
              if (kv.first == *sid &&
                  std::strcmp(s.job().strings.c_str() + kv.second,
                              "rs") == 0)
                rs = true;
          }
        }
        steal_rs_[w.warp_index] = rs;
      }
      if (r.issued == 0 && w.sm < sm_inflight_.size()) {
        auto& v = sm_inflight_[w.sm];
        v.erase(std::remove_if(v.begin(), v.end(),
                               [&](const Inflight& e) {
                                 return e.complete <= s.now();
                               }),
                v.end());
        v.push_back(Inflight{r.launch, complete,
                             w.warp_index < steal_coresident_.size() &&
                                 steal_coresident_[w.warp_index]});
      }
    }
  }
  s.async_groups().issue(AsyncGroups::kWgmma, w.warp_index,
                         complete, 0);
  if (sema.flags & kSemaFlagGroupCommit) {
    s.async_groups().commit(AsyncGroups::kWgmma, w.warp_index,
                            s.now());
    wg_open_[w.warp_index] = false;
  }
  // remember the accumulator window for consumer-side handoff, with
  // the producer's capture facts (the tensor.pipe row carries
  // operation_class / tensor_n)
  auto& a = acc_[w.warp_index];
  a.begin = sema.field[1] >> 32;
  a.count = sema.field[1] & 0xFFFFFFFFull;
  a.packed = sema.field[2];
  a.complete = complete;
  a.valid = a.count != 0;
  a.producer_facts = s.template_timing_capture_facts(
      inst.template_index, kTimingTensorPipeBeat);
  a.producer_is_qgmma =
      std::strncmp(s.job().strings.c_str() + t.operation_class,
                   "QGMMA", 5) == 0;
  // one rendezvous per collective op (per-warp op sequence): the entry
  // is dropped once every member issued.  Before this the first launch
  // stayed valid forever and every later op of the warpgroup skipped
  // the rendezvous and the tensor pipe, completing relative to the
  // first op's start.
  ++wg_seq_[w.warp_index];
  if (++r.issued >= r.count) rendezvous_.erase(key);
}

bool HopperModule::warpgroup_wait_ready(
    Simulator& s, WarpState& w, const InstanceSemantics& sema) {
  std::uint64_t earliest = 0;
  if (s.async_groups().wait_ready(AsyncGroups::kWgmma, w.warp_index,
                                  sema.field[1], s.now(), &earliest))
    return true;
  w.last_wait = WaitBucket::kCollective;
  if (earliest > s.now()) w.retry_at = earliest;
  tele_[kWarpgroupWaitCycles] += 1;
  return false;
}

bool HopperModule::handoff_ready(
    Simulator& s, WarpState& w, const InstructionInstance& inst) {
  // accumulator_handoff.consumer (kind 13): an inst reading the last
  // wgmma's accumulator window waits until complete + leaf cycles
  if (w.warp_index >= acc_.size() || !acc_[w.warp_index].valid)
    return true;
  const auto& a = acc_[w.warp_index];
  const auto& t = s.job().templates[inst.template_index];
  bool touches = false;
  for (std::uint32_t i = 0; i < t.source_count; ++i) {
    const std::uint64_t reg =
        s.job().register_refs[t.source_begin + i].reg
            & 0x00FFFFFFu;
    if (reg >= a.begin && reg < a.begin + a.count) {
      touches = true;
      break;
    }
  }
  if (!touches) return true;
  // consumer facts: engine axes (group_depth = this warp's wgmma groups
  // still in flight), the producer shape, then the producer's other
  // capture facts; all precede the FFMA's own FACT values
  EngineFacts ef;
  const auto ax = [&](const char* n, std::uint64_t v) {
    if (auto id = s.rt().axis_id(n)) ef.emplace_back(*id, v);
  };
  ax("group_depth", s.async_groups().pending_groups(
                        AsyncGroups::kWgmma, w.warp_index, s.now()));
  ax("tensor_m", a.packed >> 32);
  ax("tensor_n", (a.packed >> 16) & 0xFFFF);
  ax("tensor_k", a.packed & 0xFFFF);
  for (const auto& kv : a.producer_facts) {
    bool have = false;
    for (const auto& e : ef) if (e.first == kv.first) { have = true; break; }
    if (!have) ef.push_back(kv);
  }
  // P5 fp8_fmnmx_handoff.consumer (kind 29, issue event): bound to
  // QGMMA producer x FMNMX consumer in the predecessor engine (the
  // leaf's origin data is QGMMA n136) — HGMMA producers and
  // non-FMNMX consumers do not pay it.  Before 2026-11-05 it was added
  // unguarded to every first consumer that carried a kind-13 row
  // (i.e. every FFMA reading any wgmma accumulator).
  const bool consumer_fmnmx =
      std::strncmp(s.job().strings.c_str() + t.operation_class,
                   "FMNMX", 5) == 0;
  const std::uint64_t fp8_extra =
      (a.producer_is_qgmma && consumer_fmnmx)
          ? static_cast<std::uint64_t>(s.program_cycles_by_name(
                "hopper.wgmma.fp8_fmnmx_handoff.consumer", ef))
          : 0;
  // kind-13 accumulator_handoff.consumer row (FFMA consumers)
  const auto* f = s.template_timing_fields(
      inst.template_index, kKindHandoffConsumer, ef);
  if (!f && fp8_extra == 0) return true;
  std::uint64_t ready = a.complete + fp8_extra;
  if (f)
    ready += static_cast<std::uint64_t>(
        MechanismRuntime::field(*f, TR::CYCLES, 0.0));
  if (s.now() < ready) {
    w.last_wait = WaitBucket::kSemantic;
    w.retry_at = ready;
    ++tele_[kHandoffWaits];
    return false;
  }
  acc_[w.warp_index].valid = false;  // consumed once per wgmma
  return true;
}

std::uint64_t HopperModule::memory_queue_retry(
    std::uint32_t sm, std::uint32_t smsp, std::uint64_t now) const {
  // memory-queue hold (H'): for each warpgroup registered on this
  // SMSP, count its outstanding ops (pipe ends > now); above
  // memory_queue_hold_above_pending_ops the queue waits until the
  // outstanding count reaches the threshold (the end of the
  // (count - threshold)-th earliest outstanding op).  A warpgroup
  // with a launch held by Q' parks the queue until the launch marks
  // the SMSP.
  const auto it = mq_keys_.find((std::uint64_t(sm) << 3) | smsp);
  if (it == mq_keys_.end()) return 0;
  std::uint64_t retry = 0;
  bool parked = false;
  for (const std::uint64_t key : it->second) {
    auto q = wgq_.find(key);
    if (q == wgq_.end()) continue;
    auto& ends = q->second.op_end;
    while (!ends.empty() && ends.front() <= now) ends.pop_front();
    const std::size_t out = ends.size();
    const std::uint64_t above = q->second.mq_hold_above;
    if (above && out > above) {
      const std::uint64_t r = ends[out - above - 1];
      if (!retry || r < retry) retry = r;
    } else if (q->second.held_launch) {
      parked = true;
    }
  }
  return retry ? retry : (parked ? ~0ull : 0ull);
}

}  // namespace cyclescope
