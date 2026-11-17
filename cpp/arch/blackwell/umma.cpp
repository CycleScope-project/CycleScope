// UMMA: shared per-SM tensor pipe (tensor.pipe kind 63 beat +
// tensor.completion kind 64), UTCBAR commit arrive
// (kind-51 + kind-59 for cta_group 2).  cta_group-2 non-leader CTAs
// are simplified to an immediate completion + telemetry (docs/en/design/engine.md).

#include "arch/blackwell/module.hpp"

#include <algorithm>

#include "core/semantics_kinds.hpp"
#include "core/simulator.hpp"
#include "core/generated/parameter_schema.hpp"

namespace TR = cyclescope::core::parameter_fields::TIMING_ROW;

namespace cyclescope {

void BlackwellModule::umma_issue(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema, std::uint64_t finish) {
  const bool pair_follower = sema.field[1] == 2 &&
      !s.job().ctas[w.cta].is_pair_leader;
  if (pair_follower) {
    // batch-2 simplification: leader issues on behalf; the follower
    // op completes immediately and only counts telemetry
    umma_done_[w.warp_index].push_back(s.now());
    ++tele_[kUmmaPairFollower];
    return;
  }
  // beat and operand-read rate come off the same resolve (the Fields
  // pointer dies with the next one): read both before touching it
  const auto* bf = s.template_timing_fields(
      inst.template_index, kTimingTensorPipeBeat, s.engine_facts());
  double beat = 0.0, rate = 0.0;
  if (bf && MechanismRuntime::field(*bf, TR::ENABLED, 1.0) != 0.0) {
    beat = MechanismRuntime::field(*bf, TR::CYCLES, 0.0);
    rate = MechanismRuntime::field(
        *bf, TR::OPERAND_READ_BYTES_PER_CYCLE, 0.0);
  }
  const std::uint64_t beat_cycles = static_cast<std::uint64_t>(beat);
  const std::uint64_t start = s.tensor_mma_start(
      w.sm, s.now(), beat_cycles,
      umma_operand_read(s, inst, sema,
                        static_cast<double>(beat_cycles), rate));
  umma_done_[w.warp_index].push_back(
      start + beat_cycles + kind_cycles(s, inst, kTimingTensorResultLatency));
  ++tele_[kUmmaIssues];
}

// cycles the MMA holds the operand path shared with tcgen05.cp:
// its shared-memory operand bytes at the leaf's read rate, capped by
// the beat.  rate <= 0 (field absent) -> the whole beat (old rule).
double BlackwellModule::umma_operand_read(
    Simulator& s, const InstructionInstance& inst,
    const InstanceSemantics& sema, double beat, double rate) {
  if (rate <= 0.0) return beat;
  static const double kBits[] = {0.0, 16.0, 32.0, 8.0, 8.0, 4.0, 4.0};
  const std::uint64_t kind = sema.field[0];
  if (kind == 0 || kind >= sizeof(kBits) / sizeof(kBits[0]) ||
      kBits[kind] == 0.0)
    throw EngineError("unsupported:umma_kind (umma_issue kind " +
                      std::to_string(kind) + ")");
  auto& tk = umma_tensor_k_[inst.template_index];
  if (!tk) {
    tk = kNone64;
    if (const auto ax = s.rt().axis_id("tensor_k"))
      for (const auto& [k, v] :
           s.template_capture_facts(inst.template_index))
        if (k == *ax) { tk = v; break; }
  }
  if (tk == kNone64)
    throw EngineError("missing_fact:axis:tensor_k (umma operand read, "
                      "template " +
                      std::to_string(inst.template_index) + ")");
  const double cg = sema.field[1] == 2 ? 2.0 : 1.0;
  const double m = static_cast<double>(sema.field[3] >> 32);
  const double n =
      static_cast<double>(sema.field[3] & 0xFFFFFFFFull);
  // a_source 1 = shared memory, 2 = tensor memory (convert_rules
  // umma_a_src); only the smem A operand crosses this path, B always
  // does
  const double a = sema.field[2] == 1 ? m : 0.0;
  const double bytes = (a + n) / cg * static_cast<double>(tk) *
                       kBits[kind] / 8.0;
  return std::min(beat, bytes / rate);
}

void BlackwellModule::umma_commit(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema) {
  // t = max(this warp's in-flight UMMA completions) + kind-51
  // (+ kind-59 pair skew when cta_group 2)
  std::uint64_t base = s.tensor_pipe_start(w.sm, s.now(), 0);
  for (const auto c : umma_done_[w.warp_index])
    base = std::max(base, c);
  std::uint64_t t = base + kind_cycles(s, inst, 51);
  if (sema.field[1] == 2) t += kind_cycles(s, inst, 59);
  const std::uint64_t object = sema.field[0];
  const std::uint64_t mask = sema.field[2];
  s.enqueue_arch(t, w.warp_index,
                 (std::uint64_t(kArrive1) << 56) |
                     ((mask & 0xFFFFFFull) << 32) |
                     (object & 0xFFFFFFFFull));
  ++tele_[kUtcbarArrives];
}

}  // namespace cyclescope
