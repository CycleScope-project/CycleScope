#pragma once
// BlackwellModule — sm_103a arch semantics (docs/en/design/engine.md):
// UMMA tensor pipe + UTCBAR, TMEM column budget, CLC, STAS.

#include <unordered_map>
#include <vector>

#include "arch/architecture_module.hpp"
#include "core/job_format.hpp"

namespace cyclescope {

enum BlackwellTele : std::uint32_t {
  kUmmaIssues = 0, kUmmaPairFollower, kUtcbarArrives, kTmemAllocs,
  kTmemBlocks, kTmemWaitStalls, kClcQueries, kStasStores,
  kBlackwellTeleCount
};

class BlackwellModule : public ArchitectureModule {
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
  // per-SM UMMA pipe next-free cycle
  // per-warp recorded UMMA op completions (UTCbar baseline)
  std::unordered_map<std::uint32_t, std::vector<std::uint64_t>>
      umma_done_;
  // TMEM: remaining columns per SM + per-CTA holdings + waiters
  std::unordered_map<std::uint32_t, std::uint64_t> tmem_free_;
  std::unordered_map<std::uint32_t, std::uint64_t> tmem_hold_;
  std::vector<std::uint32_t> tmem_waiters_;
  std::uint64_t tmem_budget_ = 512;
  // per-warp in-flight STTM (tmem_store/copy) completions
  std::unordered_map<std::uint32_t, std::uint64_t> sttm_until_;
  // template_index -> tensor_k capture fact (kNone64 = known absent;
  // template_capture_facts allocates, so look it up once)
  std::unordered_map<std::uint32_t, std::uint64_t> umma_tensor_k_;
  // deferred arch events: payload packs object | mask<<32 | op<<56
  enum ArchOp : std::uint8_t { kArrive1 = 0, kTx16 = 1, kTxN = 2 };
  std::unordered_map<std::uint64_t, std::pair<std::uint32_t, std::uint64_t>>
      pending_;  // seq -> (issuer warp, extra bytes/rank)
  std::uint64_t seq_ = 0;
  std::uint64_t tele_[kBlackwellTeleCount] = {};

  void umma_issue(Simulator&, WarpState&, const InstructionInstance&,
                  const InstanceSemantics&, std::uint64_t finish);
  double umma_operand_read(Simulator&, const InstructionInstance&,
                           const InstanceSemantics&, double beat,
                           double rate);
  void umma_commit(Simulator&, WarpState&,
                   const InstructionInstance&,
                   const InstanceSemantics&);
  bool tmem_alloc_ready(Simulator&, WarpState&,
                        const InstanceSemantics&);
  void tmem_dealloc(Simulator&, WarpState&,
                    const InstanceSemantics&);
  void clc_query(Simulator&, WarpState&, const InstructionInstance&,
                 const InstanceSemantics&);
  void stas(Simulator&, WarpState&, const InstructionInstance&,
            const InstanceSemantics&, std::uint64_t finish);
  uint64_t kind_cycles(Simulator&, const InstructionInstance&,
                       uint32_t kind) const;
  double kind_value(Simulator&, const InstructionInstance&,
                    std::uint32_t kind) const;
};

}  // namespace cyclescope
