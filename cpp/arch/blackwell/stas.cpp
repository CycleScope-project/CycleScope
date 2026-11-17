// STAS (shared_store_async kind 33): finish + kind-21 cycles ->
// complete_tx(bytes f1) on object f0 resolved via destination_rank
// f2 (MBOB).  docs/en/design/engine.md.

#include "arch/blackwell/module.hpp"

#include "core/semantics_kinds.hpp"
#include "core/simulator.hpp"

namespace cyclescope {

void BlackwellModule::stas(
    Simulator& s, WarpState& w, const InstructionInstance& inst,
    const InstanceSemantics& sema, std::uint64_t finish) {
  const std::uint64_t t =
      finish + kind_cycles(s, inst, 21);
  const std::uint64_t object = sema.field[0];
  const std::uint64_t rank = sema.field[2];
  // reuse the kTxN op: bytes ride in the event sequence low bits
  const std::uint64_t payload =
      (std::uint64_t(kTxN) << 56) |
      (rank << 32) | (object & 0xFFFFFFFFull);
  s.enqueue_arch(t, w.warp_index, payload);
  pending_[payload] = {w.warp_index, sema.field[1]};
  ++tele_[kStasStores];
}

}  // namespace cyclescope
