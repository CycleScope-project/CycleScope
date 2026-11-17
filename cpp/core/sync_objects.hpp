#pragma once
// Synchronization state machines (docs/en/design/engine.md): mbarrier
// phases, CTA named barriers, cluster barriers.  Pure state machines —
// the Simulator feeds issue events, reads readiness, and owns release
// timing (release_cycle is written back here once the mechanism leaf
// is resolved).

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "job_format.hpp"

namespace cyclescope {

class MbarrierTable {
 public:
  struct Object {
    bool initialized = false;
    bool cluster_scope = false;
    std::uint32_t owner_cta = kNone32;
    std::uint64_t expected_arrivals = 0;
    std::int64_t pending_arrivals = 0;
    std::int64_t tx_count = 0;  // signed: PTX allows complete before expect
    std::uint64_t phase = 0;    // completed phase count
    std::uint64_t phase_complete_cycle = 0;
    std::uint64_t release_cycle = 0;
    // phase completion + suspended_wake_cycles (or release_cycle when
    // the field is not installed): suspended try_wait/deferred waiters
    // resume at this cycle, not at release_cycle (trywait_wake
    // protocol, full-gpu-chain-report §28)
    std::uint64_t suspended_release_cycle = 0;
    // evidence accumulated during the phase that last completed
    std::unordered_set<std::uint32_t> phase_warps;
    std::uint64_t phase_expect_bytes = 0;
    std::uint64_t phase_complete_bytes = 0;
    // snapshot of the evidence at the last phase completion (leaf axes)
    std::uint64_t release_participant_warps = 0;
    std::uint64_t release_expected_tx_bytes = 0;
    std::uint64_t release_completed_tx_bytes = 0;
    std::unordered_set<std::uint32_t> waiters;
    // warps suspended until this object's next release (failed-poll /
    // NANOSLEEP.SYNCS wakes); blocked_until tags the suspension so
    // stale entries cannot shorten a later unrelated suspension
    struct Sleeper { std::uint32_t warp; std::uint64_t blocked_until;
                     std::uint64_t wake_latency = 0; };
    std::vector<Sleeper> sleepers;
  };

  // object index spaces are segment-local (multi-segment containers)
  static constexpr std::uint64_t key(std::uint32_t segment,
                                     std::uint64_t object) {
    return (std::uint64_t(segment) << 32) | (object & 0xFFFFFFFFull);
  }

  // expected == 0 -> missing_fact; init on a live object -> reinit error
  void init(std::uint32_t segment, std::uint32_t object,
            std::uint64_t expected, bool cluster_scope,
            std::uint32_t cta, std::uint64_t cycle,
            std::uint32_t initial_phase = 0);
  // returns true when the arrival completes a phase
  bool arrive(std::uint32_t segment, std::uint32_t object,
              std::uint64_t credit, std::uint64_t tx_bytes,
              std::uint32_t warp, std::uint64_t cycle);
  bool expect_tx(std::uint32_t segment, std::uint32_t object,
                 std::uint64_t bytes, std::uint64_t cycle);
  bool complete_tx(std::uint32_t segment, std::uint32_t object,
                   std::uint64_t bytes, std::uint64_t cycle);
  bool wait_ready(std::uint32_t segment, std::uint32_t object,
                  std::uint32_t parity, std::uint64_t cycle) const;
  // CYCLESCOPE_DEBUG_MBARRIER=<segment>:<object> state trace (diagnostic)
  void debug(const char* op, std::uint32_t segment, std::uint32_t object,
             std::int64_t delta, std::uint64_t cycle) const;
  // returns the discarded in-flight (pending, tx) counts
  std::pair<std::int64_t, std::int64_t> inval(std::uint32_t segment,
                                              std::uint32_t object);

  const Object* find(std::uint32_t segment, std::uint32_t object) const;
  Object* find(std::uint32_t segment, std::uint32_t object);
  void add_waiter(std::uint32_t segment, std::uint32_t object,
                  std::uint32_t warp);
  void remove_waiter(std::uint32_t segment, std::uint32_t object,
                     std::uint32_t warp);
  void add_sleeper(std::uint32_t segment, std::uint32_t object,
                   std::uint32_t warp, std::uint64_t blocked_until,
                   std::uint64_t wake_latency = 0);

 private:
  bool check_phase(Object& o, std::uint32_t object, std::uint64_t cycle);
  // keyed by (segment << 32) | object
  std::unordered_map<std::uint64_t, Object> objects_;
};

class CtaBarrierTable {
 public:
  struct ArriveResult {
    bool full = false;
    std::uint64_t generation = 0;
    std::uint32_t participant_warps = 0;
    // last arriving instance (for the kind-17 TIME row on release)
    std::uint64_t last_arrive_instance = kNone64;
  };

  // warps_per_cta[i] = warp count of CTA i
  explicit CtaBarrierTable(std::vector<std::uint32_t> warps_per_cta);

  ArriveResult arrive(std::uint32_t cta, std::uint32_t barrier,
                      std::uint32_t warp, std::uint64_t expected_field,
                      std::uint64_t instance, std::uint64_t cycle);
  bool wait_ready(std::uint32_t cta, std::uint32_t barrier,
                  std::uint64_t generation, std::uint64_t cycle) const;
  void set_release(std::uint32_t cta, std::uint32_t barrier,
                   std::uint64_t generation, std::uint64_t cycle);
  // re-judges open generations of `cta` after a warp exits; returns
  // (barrier, result) for generations that close
  std::vector<std::pair<std::uint32_t, ArriveResult>> warp_exited(
      std::uint32_t cta, std::uint32_t warp, std::uint64_t cycle);

 private:
  struct Gen {
    std::uint64_t expected = 0;
    std::uint64_t arrived = 0;
    std::unordered_set<std::uint32_t> warps;
    std::uint64_t last_arrive_instance = kNone64;
  };
  struct Barrier {
    std::uint64_t expected_field = 0;
    std::uint64_t open_gen = 0;
    Gen open;
    // closed generation -> release cycle
    std::unordered_map<std::uint64_t, std::uint64_t> closed_release;
  };
  std::uint64_t expected_for(const Barrier& b, std::uint32_t cta) const;
  Barrier& get_or_create(std::uint32_t cta, std::uint32_t barrier);
  // key = cta << 32 | barrier
  std::unordered_map<std::uint64_t, Barrier> barriers_;
  // cta -> its barrier keys in creation order (warp_exited only
  // re-judges this CTA's barriers)
  std::vector<std::vector<std::uint64_t>> cta_keys_;
  std::vector<std::uint32_t> warps_per_cta_;
  std::vector<std::unordered_set<std::uint32_t>> exited_;
};

class ClusterBarrierTable {
 public:
  struct ArriveResult {
    bool full = false;
    std::uint64_t generation = 0;
    std::uint32_t participant_warps = 0;
    std::uint32_t participant_ctas = 0;
    std::uint64_t last_arrive_instance = kNone64;
  };

  // warps_per_cluster[i] = warp count of cluster i
  explicit ClusterBarrierTable(std::vector<std::uint32_t> warps_per_cluster);

  // `generation` is the warp's own arrive generation (program order)
  ArriveResult arrive(std::uint32_t cluster, std::uint64_t generation,
                      std::uint32_t cta, std::uint32_t warp,
                      std::uint64_t instance, std::uint64_t cycle);
  bool wait_ready(std::uint32_t cluster, std::uint64_t generation,
                  std::uint64_t cycle) const;
  void set_release(std::uint32_t cluster, std::uint64_t generation,
                   std::uint64_t cycle);
  std::vector<ArriveResult> warp_exited(std::uint32_t cluster,
                                        std::uint32_t warp,
                                        std::uint64_t cycle);

 private:
  struct Gen {
    std::unordered_set<std::uint32_t> warps;
    std::unordered_set<std::uint32_t> ctas;
    std::uint64_t last_arrive_instance = kNone64;
    bool closed = false;
    std::uint64_t release_cycle = 0;
  };
  struct Cluster {
    std::uint32_t total_warps = 0;
    std::unordered_set<std::uint32_t> exited;
    std::unordered_map<std::uint64_t, Gen> gens;
  };
  std::uint64_t expected_for(const Cluster& c) const;
  bool maybe_close(Cluster& c, std::uint64_t gen,
                   ArriveResult& out) const;
  std::unordered_map<std::uint32_t, Cluster> clusters_;
};

}  // namespace cyclescope
