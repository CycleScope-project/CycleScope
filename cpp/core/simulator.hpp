#pragma once
// Simulator + Scheduler — 3-1 skeleton + 3-2 memory + 3-3 step a
// synchronization (docs/en/design/engine.md).
// Per-warp program-order issue.  Readiness order (§13.2): dispatch ->
// collective wait (suspended / deferred / barrier waits) -> semantic ->
// resource -> register/scoreboard/control -> scheduler arbitration.
// Time advance: now = max(now+1, min(next_attempt, next event)) —
// next_attempt is the earliest cycle a non-semantically-blocked
// resident warp can be retried.

#include <cstdint>
#include <deque>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "async_groups.hpp"
#include "dispatch.hpp"
#include "global_sync.hpp"
#include "instruction_fetch.hpp"
#include "job_reader.hpp"
#include "lifecycle.hpp"
#include "mechanism_runtime.hpp"
#include "result_writer.hpp"
#include "memory_hierarchy.hpp"
#include "semantics_kinds.hpp"
#include "sync_objects.hpp"
#include "arch/architecture_module.hpp"

namespace cyclescope {

enum class WaitBucket : std::uint8_t {
  kDispatch = 0, kFetch, kCollective, kSemantic, kResource, kRegister,
  kScheduler, kCount
};

// TELE counter ids (group 1 = sync)
enum SyncTele : std::uint32_t {
  kTeleMbarrierPhases = 0,
  kTeleMbarrierArrivals,
  kTeleMbarrierTxBytes,
  kTeleMbarrierInvals,
  kTeleMbarrierRemoteArrivals,
  kTeleCtaBarrierGenerations,
  kTeleClusterBarrierGenerations,
  kTeleFailedPolls,
  kTelePollSuspendCycles,
  kTeleNanosleeps,
  kTeleVisibilitySettled,
  kTeleMbarrierInvalInflight,
  kTeleTmaCompletions,
  kTeleTmaBytes,
  kTeleDescriptorAcquireUnpublished,
  kTeleGsyncAcquiresFrom,
  kTeleGsyncAcquiresAny,
  kTeleGsyncAcquiresCount,
  kTeleGsyncAcquiresInitial,
  kTeleGsyncDroppedPolls,
  kTeleGsyncChainsUnique,
  kTeleGsyncChainsNonUnique,
  kTeleGsyncUpdateUnknownPost,
  kTeleGsyncWaitCycles,
  kTeleStalePolls,
  kTeleSuspendedWaits,
  kTeleStoreBufferStalls,
  kTeleStoreBufferStallCycles,
  kTeleAbandonedPolls,
  kTeleGsyncAcquiresNoWriter,
  kTeleGsyncRegionsRetired,
  kTeleGsyncRegionInstances,
  kTeleFenceDrainHolds,
  kTeleFenceDrainCycles,
  kTeleCount
};

// TELE group 2 counters (dispatch / lifecycle / fetch)
enum EngineTele : std::uint32_t {
  kTele2DispatchEntries = 0,
  kTele2DispatchWaves,
  kTele2DispatchWaitCycles,
  kTele2LifecycleEnvelope,
  kTele2FetchUnavailable,
  kTele2FetchL0Hits,
  kTele2FetchL1Lines,
  kTele2FetchL2Lines,
  kTele2FetchColdLines,
  kTele2FetchWaitCycles,
  // per-CTA issue span (first issue -> last issue): mean over CTAs that
  // issued, and the maximum; max / mean >> 1 flags a trace whose
  // dynamic tile scheduler loaded one CTA far more than the others
  // (full-gpu-chain-report.md §19)
  kTele2CtaSpanMean,
  kTele2CtaSpanMax,
  kTele2Count
};

struct Event {
  enum Kind : std::uint8_t { kResultReady, kReadBarrierFree,
                             kResourceFree, kWarpDone, kWake,
                             kBarrierRelease, kDeferredArrive,
                             kTransferComplete, kDispatch,
                             kArch } kind;
  std::uint64_t cycle;
  std::uint32_t warp;
  std::uint32_t instance;
  std::uint64_t payload;
  std::uint64_t sequence;
  bool operator>(const Event& o) const {
    return cycle != o.cycle ? cycle > o.cycle : sequence > o.sequence;
  }
};

struct WarpState {
  std::uint32_t warp_index = 0, cta = 0, sm = 0, smsp = 0, slot = 0;
  std::uint64_t next_instance = 0, end_instance = 0;
  bool done = false;
  bool resident = true;  // dispatch lands in a later step
  // flat per-warp reg->ready-cycle table; out-of-range == unset (0),
  // grown lazily on first destination write (to max_reg_+1)
  std::vector<std::uint64_t> reg_ready;
  // diagnostic: writer instance of each reg_ready / barrier_free
  std::unordered_map<std::uint32_t, std::uint32_t> reg_writer;
  std::uint32_t barrier_writer[8] = {};
  std::uint64_t barrier_free[8] = {};
  // cached "register + scoreboard ready" horizon for the CURRENT
  // next_instance: reg_ready/barrier_free only change at issue time,
  // so this stays valid until the warp issues again
  std::uint64_t ready_at_ = 0;
  std::uint64_t stall_since = 0;
  WaitBucket last_wait = WaitBucket::kScheduler;
  std::uint64_t issue_stamp = ~0ull;
  // collective suspension (nanosleep / failed-poll trywait)
  std::uint64_t blocked_until = 0;  // 0 = not suspended
  // cycle the warp first blocked on the current mbarrier wait (0 =
  // none): a suspended wait resumes at suspended_release_cycle; a wait
  // that is already phase-ready at its first evaluation is not gated
  std::uint64_t suspended_mbar_since = 0;
  WaitBucket block_bucket = WaitBucket::kCollective;
  // Hopper deferred wait: issue wait, block at instance block_index
  struct Deferred {
    bool active = false;
    std::uint32_t object = 0, parity = 0;
    std::uint64_t block_index = 0;
  } deferred;
  struct CtaWait {
    bool active = false;
    std::uint32_t barrier = 0;
    std::uint64_t generation = 0;
  } cta_wait;
  struct ClusterWait {
    bool active = false;
    std::uint64_t generation = 0;
  } cluster_wait;
  std::uint64_t cluster_arrive_gen = 0;
  // most recent mbarrier_wait object / parity (NANOSLEEP.SYNCS wake)
  std::uint32_t last_wait_object = kNone32;
  std::uint32_t last_wait_parity = 0;
  // kernel-exit drain (§13.6): barrier exit deduction happened but
  // async work is still in flight
  bool exit_signaled = false, draining = false;
  std::uint64_t inflight_until = 0;
  // last object evaluated by a global_sync acquire (deadlock report)
  std::uint64_t wait_detail = 0;
  // next-issue floor: control envelope and no-destination service results
  std::uint64_t control_floor = 0;
  // co-resident dependent penalty toggle (arch C4 rule): flips on
  // every dependency-triggered issue while the penalty applies
  bool dep_penalty_toggle = false;
  // rs extra half cycle toggle (arch C7+R rule): flips on every
  // issue while an rs-sourced op is in flight
  bool rs_toggle = false;
  // instance for which the group-wait implicit commit already ran
  std::uint64_t implicit_committed_instance = ~0ull;
  // dispatch entry floor (§13.11): 0 = admitted at cycle 0
  std::uint64_t entry_cycle = 0;
  // cached fetch verdict for (fetch_instance) — one ready() per inst
  std::uint64_t fetch_instance = ~0ull, fetch_ready = 0;
  // earliest cycle this warp can be retried (0 = semantic, event-driven)
  std::uint64_t retry_at = 0;
  // bucket accounting mark: cycles [wait_mark, eval) accrue to
  // last_wait lazily instead of per-cycle accumulation
  std::uint64_t wait_mark = 0;
  // gsync acquire wait: cycle the current blocked span began; the
  // wait telemetry accrues cycles (previously per-cycle eval counts)
  std::uint64_t gsync_wait_mark = ~0ull;
  // rule 3b region wait: first blocked cycle at a region head (~0 = no
  // wait in flight); used for kTeleGsyncWaitCycles accrual and the
  // journal wait span
  std::uint64_t region_wait_mark = ~0ull;
  // longest-waiter-first unit arbitration (scheduler
  // unit_grant_policy=1): key (res << 32 | owner) of the execution
  // unit this warp is parked on, kNone64 when not a waiter; the last
  // try_issue verdict was a unit busy/outbid verdict when unit_blocked
  std::uint64_t unit_wait_key = ~0ull;
  bool unit_blocked = false;
};

struct SmspState {
  std::vector<std::uint32_t> warps;
  // scheduler cursor (candidate_order): last issued warp + yield flag
  std::uint32_t last_issued = 0;
  bool has_last = false, last_yield = true;
  std::uint64_t first_active = 0xFFFFFFFFFFFFFFFFull;
  std::uint64_t last_active = 0;
  // event/wake scheduling: step only when a warp in this SMSP has a
  // retry due (next_attempt <= now) or an event woke it (needs_eval)
  bool needs_eval = true;
  std::uint64_t next_attempt = ~0ull;
  // cursor position of last_issued in `warps` when still valid
  std::size_t last_pos = 0;
  // register_file.operand_read (collector 2, pipelined): next free
  // read slot, warp owning the open cycle, service start of the
  // entry; the SMSP stalls while the last n > 0 entry has not started
  std::uint64_t coll_slot_end = 0;
  std::uint32_t coll_warp = 0xFFFFFFFFu;
  std::uint64_t coll_svc_start = 0;
};

struct ResourceState {
  bool seen = false;
  std::uint32_t resource = 0, copy = 0;
  // explicit in-flight capacity (0 = unlimited): the unit is modelled
  // by its initiation interval (service leaf issue_interval_cycles,
  // fractional residual) plus an optional capacity cap
  std::uint64_t max_inflight = 0;
  std::uint64_t next_free = 0;             // next admission cycle
  std::vector<std::uint64_t> completions;  // per-copy completion cycles
  double spacing_residual = 0.0, occupancy_residual = 0.0;
  std::uint64_t wait_cycles = 0;
};

struct ResultStats {
  std::uint64_t cycles = 0, issued = 0, completed = 0;
  std::uint64_t hard_failures = 0;
  std::vector<SmspSpanRow> smsp_spans;
  std::vector<std::pair<std::uint32_t, std::uint64_t>> wait;
  std::vector<std::pair<std::uint32_t, std::uint64_t>> rwait;
  std::vector<TelemetryRow> tele;
  std::vector<JournalRow> journal;
  std::vector<WaitLedgerRow> wait_ledger;
};

class Simulator {
 public:
  struct Options {
    bool journal = false, wait_ledger = false, progress = false;
    std::uint64_t max_cycles = 0;
  };
  Simulator(const Job& job, MechanismRuntime& rt, Options opt);

  ResultStats run();

  std::uint64_t now() const { return now_; }
  const Job& job() const { return job_; }
  // synchronous shared store predicate shared by the dirty proxy-fence
  // rule and Hopper's proxy_fence drain: MACC shared space
  // (flags&7 == 1) + synchronous store (flags&0x1C0 == 0x40), excluding
  // the async-store semantics; bytes = sector byte_mask popcounts
  // (mask 0 counts 32)
  bool sync_shared_store(const InstructionInstance& inst,
                         const InstanceSemantics& sema,
                         std::uint64_t* bytes) const;

  // arch-module hooks (docs/en/design/engine.md): the module sees warp tables
  // and enqueue/arrive helpers but not core containers
  WarpState& warp(std::uint32_t i) { return warps_[i]; }
  const WarpState& warp(std::uint32_t i) const { return warps_[i]; }
  const std::vector<std::uint32_t>& warps_of_cta(
      std::uint32_t cta) const { return cta_warps_[cta]; }
  void enqueue_arch(std::uint64_t cycle, std::uint32_t warp,
                    std::uint64_t payload) {
    enqueue(Event::kArch, cycle, warp, 0, payload);
  }
  bool mbarrier_arrive(std::uint32_t segment, std::uint32_t object,
                       std::uint64_t credit, std::int64_t tx,
                       std::uint32_t warp);
  bool mbarrier_complete_tx(std::uint32_t segment,
                            std::uint32_t object,
                            std::uint64_t bytes);
  std::pair<std::uint32_t, std::uint32_t> resolve_destination(
      std::uint32_t segment, std::uint64_t object, std::uint32_t rank,
      std::uint32_t issuer_rank) const;
  // resolved leaf of the first TIME row of `kind` on a template
  // (nullptr if none); the pointer is invalidated by the next
  // resolve — read fields immediately
  const Fields* template_timing_fields(
      std::uint32_t template_index, std::uint32_t kind,
      const EngineFacts& extra);
  const Fields& template_service_fields(
      const InstructionTemplate& t) { return service_fields(t); }
  // capture FACT pairs of the USES row behind the first TIME row of
  // `kind` on a template (empty if none) — lets a module hand a
  // producer's facts to a consumer-side relation program
  EngineFacts template_timing_capture_facts(std::uint32_t template_index,
                                            std::uint32_t kind) const;
  // union of every capture FACT row recorded on this static
  // instruction: its instruction.service USES row (target 0) plus
  // each of its TIME rows' uses; on an axis-id collision the first
  // occurrence wins
  EngineFacts template_capture_facts(std::uint32_t template_index) const;
  double program_cycles_by_name(const char* name,
                                const EngineFacts& ef);
  void block_warp(WarpState& w, std::uint64_t until,
                  WaitBucket bucket) {
    w.blocked_until = until;
    w.block_bucket = bucket;
    if (until > w.retry_at) w.retry_at = until;
  }
  void mark_warp_for_eval(WarpState& w) { mark_smsp_of(w); }
  // issue gate shared by all warps on one SM (wgmma RF steal)
  void block_sm_issue(std::uint32_t sm, std::uint64_t until) {
    auto& u = sm_issue_block_until_[sm];
    u = std::max(u, until);
  }
  // shared per-SM tensor frontiers (tensor.pipe, both architectures):
  // the compute beat frontier (tensor_pipe_next_) serializes MMA
  // beats; the operand-path frontier (tensor_operand_path_next_) is
  // the per-SM path a Blackwell MMA shares with tcgen05.cp — an MMA
  // holds it only for `operand_read` cycles of its beat while a copy
  // reserves it for its service.  tensor_pipe_start keeps the
  // single-frontier behaviour for Hopper wgmma and for UTCBAR commit.
  std::uint64_t tensor_pipe_start(std::uint32_t sm, std::uint64_t launch,
                                  std::uint64_t beat) {
    auto& next = tensor_pipe_next_[sm];
    const double start = std::max(static_cast<double>(launch), next);
    next = start + beat;
    return static_cast<std::uint64_t>(std::ceil(start));
  }
  std::uint64_t tensor_mma_start(std::uint32_t sm, std::uint64_t launch,
                                 std::uint64_t beat, double operand_read) {
    auto& pipe = tensor_pipe_next_[sm];
    auto& path = tensor_operand_path_next_[sm];
    const double start =
        std::max({static_cast<double>(launch), pipe, path});
    pipe = start + beat;
    path = start + operand_read;
    return static_cast<std::uint64_t>(std::ceil(start));
  }
  std::uint64_t tensor_operand_path_finish(std::uint32_t sm,
                                           std::uint64_t launch,
                                           double service) {
    auto& path = tensor_operand_path_next_[sm];
    path = std::max(static_cast<double>(launch), path) + service;
    return static_cast<std::uint64_t>(std::ceil(path));
  }
  std::uint32_t seg_of(const WarpState& w) const;
  std::uint32_t cta_segment(std::uint32_t cta) const {
    return job_.cta_segment[cta];
  }
  MechanismRuntime& rt() { return rt_; }
  EngineFacts& engine_facts() { return engine_facts_; }
  AsyncGroups& async_groups() { return async_; }
  const AsyncGroups& async_groups() const { return async_; }

 private:
  struct ResDef {
    std::uint32_t scope = 0;
    std::uint64_t copies = 1;
    double max_inflight = 0.0, interval = 1.0;
  };
  struct Policy {
    std::uint32_t issue_slots = 1;
    bool cursor_commit = true, try_next_on_block = true;
    // unit_grant_policy: 0 = cursor re-serves the last issuer,
    // 1 = a free execution unit goes to its longest live waiter
    bool unit_grant_lwf = false;
  };
  // pending store-visibility entry (§13.8): the TIME row's USES is
  // resolved when a release event settles it
  struct VisibilityEntry {
    std::uint64_t time_index, issue_cycle, accept_cycle;
  };

  void place_warps();
  void load_setup();
  void schedule(const Event& e);
  void step_smsp(std::uint32_t sm, std::uint32_t smsp);
  std::vector<std::size_t> candidate_order(const SmspState& s) const;
  bool eligible(const WarpState& w) const;
  void mark_smsp_of(const WarpState& w);
  void flush_wait(WarpState& w, std::uint64_t now);
  bool try_issue(WarpState& w);
  bool resource_ready(WarpState& w, const InstructionTemplate& t,
                      const Fields& s);
  bool register_ready(const WarpState& w,
                      const InstructionTemplate& t) const;
  bool scoreboard_ready(const WarpState& w,
                        const InstructionTemplate& t) const;
  // earliest unmet register/scoreboard/control ready cycle (kNever ok)
  std::uint64_t max_source_reg_ready(const WarpState& w,
                                     const InstructionTemplate& t) const;
  std::uint64_t register_retry(const WarpState& w,
                               const InstructionTemplate& t) const;
  std::uint64_t capacity_ready(
      const std::vector<std::uint64_t>& completions,
      std::uint64_t max_inflight, std::uint64_t candidate);
  void enqueue(Event::Kind k, std::uint64_t cycle, std::uint32_t warp,
               std::uint32_t inst, std::uint64_t payload);
  std::uint32_t service_resource(const InstructionTemplate& t,
                                 const Fields& svc,
                                 std::uint32_t which) const;
  static std::uint64_t resource_cap(const ResDef& d, const Fields& svc);
  const Fields& service_fields(const InstructionTemplate& t);

  // ---- 3-3 sync ----
  bool semantic_ready(WarpState& w, const InstructionInstance& inst,
                      const InstanceSemantics& sema);
  bool failed_poll(std::uint64_t inst_index,
                   const InstanceSemantics& sema) const;
  // next instance is a blocking mbarrier wait (no side effects; used
  // to exclude it from the ready_warps counts)
  bool semantic_blocked(const WarpState& w) const;
  void on_semantic_issue(WarpState& w, const InstructionInstance& inst,
                         const InstanceSemantics& sema,
                         const InstructionTemplate& t,
                         std::uint64_t finish_cycle);
  void on_warp_exit(WarpState& w);
  void mbarrier_phase_complete(std::uint32_t segment,
                               std::uint32_t object);
  void cta_generation_close(std::uint32_t cta, std::uint32_t barrier,
                            const CtaBarrierTable::ArriveResult& r);
  void cluster_generation_close(
      std::uint32_t cluster, const ClusterBarrierTable::ArriveResult& r);
  std::uint64_t settle_visibility(std::uint32_t cta,
                                  std::uint64_t upto_cycle,
                                  std::uint64_t participant_warps);
  EngineFacts event_facts(
      std::initializer_list<std::pair<const char*, std::uint64_t>> axes)
      const;
  double timing_cycles(std::uint64_t time_index,
                       const EngineFacts* extra = nullptr);
  double program_cycles(std::optional<std::uint32_t> prog,
                        const EngineFacts& ef, const char* where) const;
  // cycle at which this warp's pending gsync acquire becomes ready;
  // nullopt when it cannot be determined yet (producer not recorded)
  std::optional<std::uint64_t> gsync_ready_cycle(
      const WarpState& w) const;
  std::optional<std::uint64_t> gsync_ready_cycle_of(
      std::uint64_t acquire_instance) const;
  std::uint64_t ns_to_cycles(std::uint64_t ns) const;
  [[noreturn]] void deadlock_report() const;

  static constexpr std::uint64_t kNever = ~0ull;

  const Job& job_;
  MechanismRuntime& rt_;
  std::unique_ptr<ArchitectureModule> arch_;
  // per-SM issue gate written by the arch module (wgmma RF steal)
  std::unordered_map<std::uint32_t, std::uint64_t>
      sm_issue_block_until_;
  // per-template memory-queue class flag (MUFU/SHFL/LDS op classes),
  // filled lazily on the first issue
  std::vector<bool> mq_class_;
  // distinct general-purpose source registers per template (id < 256,
  // a 64-bit pair counts the second register too); lazy, sized like
  // mq_class_.  Used by IT::SINGLE_SOURCE_ISSUE_INTERVAL_CYCLES.
  std::vector<std::uint32_t> gpr_source_count_;
  std::unordered_map<std::uint32_t, double> tensor_pipe_next_;
  std::unordered_map<std::uint32_t, double> tensor_operand_path_next_;
  Options opt_;
  std::uint64_t now_ = 0, seq_ = 0;
  std::priority_queue<Event, std::vector<Event>, std::greater<>> queue_;
  std::vector<WarpState> warps_;
  std::vector<std::uint32_t> active_;
  // warp -> index in active_ (maintained on push and swap-pop)
  std::vector<std::uint32_t> active_pos_;
  std::uint64_t pending_warps_ = 0;
  std::vector<std::vector<SmspState>> smsps_;
  std::unordered_map<std::uint64_t, ResourceState> resources_;
  // flat lookup cache into resources_ (node pointers are stable):
  // res_ptr_[res][copy * owner_span + owner]; inner vectors grow on
  // demand as smsps_ / ctas grow
  std::vector<std::vector<ResourceState*>> res_ptr_;
  ResourceState& resource_state(std::uint32_t res, std::uint64_t copy,
                                std::uint64_t owner, const ResDef& d);
  std::unordered_map<std::uint32_t, ResDef> res_def_;
  std::vector<std::uint64_t> tmpl_use_;   // template_index -> use index (kNone64)
  // leaf-resolve caches: nullptr entry = depends on engine axes,
  // must resolve every call (§14 perf2); *_cached_ marks "visited"
  std::vector<const Fields*> svc_cache_;
  std::vector<std::uint8_t> svc_cached_;
  std::vector<const Fields*> time_cache_;
  std::vector<std::uint8_t> time_cached_;
  void update_ready_at(WarpState& w);
  // unit_grant_policy=1 arbitration state: per (res << 32 | owner)
  // execution unit, the warps parked on it and the cycle each first
  // waited, plus the warp the unit last went to (ties yield to it)
  struct UnitGrant {
    std::unordered_map<std::uint32_t, std::uint64_t> waiters;
    std::uint32_t last_grant = kNone32;
  };
  std::unordered_map<std::uint64_t, UnitGrant> unit_waiters_;
  void clear_unit_waiter(WarpState& w);
  std::uint64_t sched_use_ = kNone64;
  Policy policy_;
  // register_file.operand_read leaf (schema 33): enabled 0 = off.
  // collector 0 / 1 are declared but not implemented (init throws)
  struct { std::uint32_t enabled = 0, reads_per_cycle = 2,
           collector = 2; } operand_cfg_;
  std::uint64_t operand_use_ = kNone64;
  MemoryHierarchy mem_{MemoryConfig{}};
  MemoryConfig mem_cfg_{};
  // memory-hierarchy return of the instance being issued (kNone64 when
  // it has no load path): the async (TMA / bulk) completion adds only
  // the hierarchy's excess over an L2 hit on top of its own leaf
  std::uint64_t mem_return_ = kNone64;
  // the same request's return with the GPU-shared L2 / DRAM services
  // free (MemoryHierarchy::load `shared_free`); return - shared_free is
  // the queueing the leaf-timed TMA / bulk completion adds on top of
  // its leaf (hot + cold leg)
  std::uint64_t mem_shared_free_ = kNone64;
  // sectors of that load and how many the L2 residency model sent to
  // DRAM (cold_fraction for the TMA cold leg, tma_cold protocol)
  std::uint64_t mem_sectors_ = 0, mem_dram_sectors_ = 0;
  std::vector<TimingRow> timing_;
  std::vector<std::uint64_t> timing_use_;  // time_index -> use index (kNone64)
  std::uint32_t max_reg_ = 0;              // max reg id in job_.register_refs
  const Fields& timing_fields(std::uint64_t time_index,
                              const EngineFacts* extra = nullptr);
  const Fields& try_resolve(std::uint64_t use_index);
  Fields empty_fields_;
  std::vector<std::uint64_t> pending_uses_;
  // engine axis values in load-time slot order (P2)
  EngineFacts engine_facts_;
  std::unordered_map<std::string, std::uint64_t> engine_facts_names_;
  std::uint32_t ax_ready_smsp_ = 0, ax_ready_sm_ = 0,
      ax_cache_hit_ = 0;
  std::uint64_t candidate_ready = 0;
  std::uint64_t wait_[static_cast<int>(WaitBucket::kCount)] = {};
  // --wait-ledger: (warp << 8 | bucket) -> {cycles, count}
  std::unordered_map<std::uint64_t,
                     std::pair<std::uint64_t, std::uint64_t>>
      wait_ledger_;
  ResultStats out_;
  std::uint32_t smsps_per_sm_ = 4;

  // ---- 3-3 sync state ----
  MbarrierTable mbar_;
  CtaBarrierTable cta_bar_{std::vector<std::uint32_t>{}};
  ClusterBarrierTable cluster_bar_{std::vector<std::uint32_t>{}};
  std::optional<std::uint32_t> prog_mbarrier_release_;
  std::optional<std::uint32_t> prog_trywait_;
  std::optional<std::uint32_t> prog_cluster_release_;
  // instance index -> next mbarrier_wait's (object, parity) in the same
  // warp (failed-poll lookahead, §13.3.6)
  std::unordered_map<std::uint64_t,
                     std::pair<std::uint64_t, std::uint64_t>> next_wait_;
  // instances whose (object, parity) is never checked again later in
  // the same warp: abandoned polls pay trywait_timeout, retried ones
  // cost nothing beyond completion (§13.3.6)
  std::unordered_set<std::uint64_t> abandoned_poll_;
  // NANOSLEEP.SYNCS poll regions (§13.9): instance -> [begin,end)
  // into nanosleep_objects_, the region's (object, parity) set.
  std::vector<std::pair<std::uint32_t, std::uint32_t>>
      nanosleep_objects_;
  std::unordered_map<std::uint64_t,
                     std::pair<std::uint32_t, std::uint32_t>>
      nanosleep_region_;
  std::unordered_map<std::uint32_t, std::vector<VisibilityEntry>>
      pending_visibility_;

  // ---- step b: async groups / TMA / descriptors / gsync / segments --
  AsyncGroups async_;
  // per-segment MBOB: object -> (cta, address) and the reverse lookup
  std::vector<std::vector<MbarrierObjectRow>> mbob_rows_;
  std::vector<std::map<std::pair<std::uint32_t, std::uint64_t>,
                       std::uint32_t>> mbob_rev_;
  // per segment: cluster_rank -> cta_index
  std::vector<std::unordered_map<std::uint32_t, std::uint32_t>>
      rank_cta_;
  // per segment: CTA count (cluster release leaf axis)
  std::vector<std::uint32_t> segment_cta_count_;
  // SPLT rows keyed by semantics index then rank
  std::unordered_map<std::uint64_t,
                     std::unordered_map<std::uint32_t, std::uint32_t>>
      split_credit_;
  // deferred cp.async ARRIVES / TMA completion events (payload = index)
  struct DeferredArrive {
    std::uint32_t segment, object, warp;
    std::uint64_t credit, bytes;
  };
  std::vector<DeferredArrive> deferred_arrivals_;
  struct TransferCompletion {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> dests;
    std::uint64_t bytes;
  };
  std::vector<TransferCompletion> transfer_completions_;
  // TMA service-start serialization per SM + descriptor refetch state
  std::unordered_map<std::uint32_t, std::uint64_t> sm_async_last_start_;
  // tma_oob v2: the service interval owed by the PREVIOUS TMA on this
  // SM (max of its kind-21 leaf and its cut_service_cycles)
  std::unordered_map<std::uint32_t, std::uint64_t> sm_async_prev_gap_;
  // template-level interval rows (shared issue per SMSP, shared
  // acceptance / atomic issue per SM): next admission per unit
  std::unordered_map<std::uint64_t, std::uint64_t> shared_issue_next_;
  // kind 65 global_store_issue_interval_smsp: next admission per SMSP
  // (the issuing SMSP's next global / local store waits for it)
  std::unordered_map<std::uint64_t, std::uint64_t> store_issue_next_;
  std::unordered_map<std::uint32_t, std::uint64_t> shared_accept_next_;
  // kind 80 shared_store_acceptance_smsp: next admission per SMSP
  // (a synchronous shared store occupies its own SMSP's acceptance);
  // double so the measured fractional rate is not truncated
  std::unordered_map<std::uint64_t, double> shared_accept_smsp_next_;
  std::unordered_map<std::uint32_t, std::uint64_t> atomic_issue_next_;
  // storage.shared.fence (kind 66): bytes and acceptance end of the
  // synchronous shared stores a CTA wrote since its last proxy fence
  std::unordered_map<std::uint32_t, std::uint64_t> fence_dirty_bytes_;
  std::unordered_map<std::uint32_t, std::uint64_t> fence_dirty_until_;
  std::unordered_map<std::uint32_t, bool> sm_desc_acquired_;
  std::unordered_map<std::uint32_t,
                     std::unordered_set<std::uint64_t>> sm_seen_maps_;
  // descriptor publish records (descriptor address -> cycle)
  std::unordered_map<std::uint64_t, std::uint64_t> desc_publish_;
  std::optional<std::uint32_t> prog_desc_visibility_;
  // global sync (GSYN): pairing plan + completion bookkeeping
  GlobalSyncPlan gsync_plan_;
  std::unordered_map<std::uint64_t, std::uint64_t> gsync_done_;
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>
      gsync_by_addr_;
  // InstanceTiming control-envelope bindings (USES target 2, schema 5)
  std::unordered_map<std::uint64_t, std::uint64_t> inst_envelope_;
  // SMSPs containing a warp parked on a global-sync acquire; marked
  // on every producer completion so they re-judge
  std::unordered_set<std::uint64_t> gsync_wait_smsps_;
  // rule 3b foldable poll regions: first failed poll -> winner
  std::unordered_map<std::uint64_t, std::uint64_t> gsync_region_;
  // warps that exhausted their instances this round; handled at the
  // original post-sweep point to preserve ordering
  std::vector<std::uint32_t> just_exhausted_;
  // lazy ready-warp stats cache per SM (computed at most once per
  // cycle and only when the axes exist)
  std::vector<std::uint64_t> sm_ready_stamp_;
  std::vector<std::uint64_t> ready_smsp_scratch_;
  std::uint64_t tele_[kTeleCount] = {};
  // dispatch state (§13.11): schema-24 params, dispatcher, queued
  // clusters (segment), entry records behind Event::kDispatch
  DispatchParams dispatch_;
  Dispatcher dispatcher_;
  std::deque<std::pair<std::uint32_t, std::uint64_t>> dispatch_queue_;
  // rule S fairness (store_flood replay): per-SM FIFO of warps waiting
  // at the store-buffer-full gate, in arrival order (global warp index)
  std::vector<std::deque<std::uint32_t>> store_gate_q_;
  struct DispatchEntry {
    std::uint32_t segment;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> slots;
  };
  std::vector<DispatchEntry> dispatch_entries_;
  std::vector<std::uint32_t> cta_live_warps_, cta_sm_, cta_slot_;
  // CTA executed griddepcontrol.launch_dependents (SEMA kind 40):
  // its completion holds the dispatch serializer longer
  std::vector<std::uint8_t> cta_dependents_launch_;
  // per-segment/per-CTA lookup tables (built once in place_warps)
  std::vector<std::vector<std::uint32_t>> segment_ctas_, cta_warps_;
  std::vector<std::uint32_t> segment_wpc_;
  double dispatch_last_entry_ = 0.0;
  std::uint64_t tele2_[16] = {};
  // per-CTA first / last issue cycle (kNone64 = never issued)
  std::vector<std::uint64_t> cta_first_issue_, cta_last_issue_;
  // instruction fetch (§13.13)
  InstructionFetch fetch_;
  bool fetch_enabled_ = false;
  std::unordered_map<std::uint64_t, std::uint64_t> code_addr_;
  // dense form when job_.code is an identity instance->row mapping
  std::vector<std::uint64_t> code_addr_dense_;
  bool code_addr_is_dense_ = false;
  const std::uint64_t* code_address_of(std::uint64_t inst) const {
    if (code_addr_is_dense_)
      return inst < code_addr_dense_.size() ? &code_addr_dense_[inst]
                                            : nullptr;
    const auto it = code_addr_.find(inst);
    return it == code_addr_.end() ? nullptr : &it->second;
  }
  // gsync_plan_.failed_polls as a per-instance bitmap
  std::vector<bool> gsync_failed_poll_;
  std::unordered_set<std::uint32_t> sm_prefetched_, sm_used_;
  std::unordered_map<std::uint32_t, std::uint32_t> sm_entries_;
  // use indexes for schemas 24 (dispatch), 25 (lifecycle), 26 (fetch)
  std::uint64_t dispatch_use_ = kNone64, lifecycle_use_ = kNone64,
      fetch_use_ = kNone64;
  void try_dispatch_queue(std::uint64_t now);
  void admit_cluster(std::uint32_t entry_index);
};

}  // namespace cyclescope
