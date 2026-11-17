// CLC: clc_query -> now + kind-56 -> complete_tx(16) on object /
// each destination_mask rank (docs/en/design/engine.md).

#include "arch/blackwell/module.hpp"

#include "core/semantics_kinds.hpp"
#include "core/simulator.hpp"

namespace cyclescope {

void BlackwellModule::clc_query(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema) {
  const std::uint64_t t = s.now() + kind_cycles(s, inst, 56);
  const std::uint64_t object = sema.field[0];
  const std::uint64_t mask = sema.field[2];
  s.enqueue_arch(t, w.warp_index,
                 (std::uint64_t(kTx16) << 56) |
                     ((mask & 0xFFFFFFull) << 32) |
                     (object & 0xFFFFFFFFull));
  ++tele_[kClcQueries];
}

}  // namespace cyclescope
