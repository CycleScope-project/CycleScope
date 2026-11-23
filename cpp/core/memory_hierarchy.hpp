#pragma once
// MemoryHierarchy — 3-2 scope (docs/en/design/engine.md).
// L1 (per SM) -> L2 (two partitions, sector cache, reader-side
// replica + far hit, weighted global LRU, dirty write-back) -> DRAM.
// Store path: store_l2/store_dram queues.  GPU shared service in
// bytes/ns.  Parameters arrive already resolved (leaf field maps)
// via MechanismRuntime per schema; no device branches.

#include <array>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <unordered_map>
#include <vector>

#include "mechanism_runtime.hpp"
#include "job_format.hpp"

namespace cyclescope {

struct MemoryConfig {
  // memory_level schema fields (schema 3), all cycles/bytes
  double l1_latency = 0, l1_issue_interval = 1, l1_bytes_per_cycle = 0;
  double l2_latency = 0, l2_issue_interval = 1, l2_bytes_per_cycle = 0;
  double dram_latency = 0, dram_issue_interval = 1,
      dram_bytes_per_cycle = 0;
  double store_l2_latency = 0, store_l2_issue_interval = 1,
      store_dram_latency = 0,
      store_l2_bytes_per_cycle = 0, store_dram_bytes_per_cycle = 0;
  // l1.local_*: stack (local space) lines are ordinary L1 residents
  // with their own hit latency (load_after_copies v3)
  double local_hit_latency = 0;
  std::uint64_t local_frame_interleaved = 0;
  // store_flood protocol (lane-store port extras): a lane store's
  // request occupies this SM's store port
  // store_l2_partial_line_cycles longer per touched line that holds a
  // partially written granule (store_l2_partial_granule_bytes each,
  // dividing the sector); issue_interval is the per-request floor for
  // lane stores only.  partial_line_cycles == 0 -> pre-measurement
  // behaviour
  double store_l2_partial_line_cycles = 0;
  std::uint64_t store_l2_partial_granule_bytes = 0;
  std::uint64_t l1_outstanding = 0, l2_outstanding = 0,
      dram_outstanding = 0, store_l2_outstanding = 0,
      store_dram_outstanding = 0;
  std::uint64_t l2_sets = 0, l2_ways = 0, sector_bytes = 32,
      line_bytes = 128, l2_resident = 0;
  // L1 residency (data array budget minus reservation and launch
  // shared-memory usage; data_array_bytes == 0 -> L1 not modeled)
  std::uint64_t l1_data_array_bytes = 0, l1_reserved_bytes = 0,
      shared_static_bytes = 0, shared_dynamic_bytes = 0;
  // memory.l1.residency l1.shared_configuration_<i>_bytes: the carveout
  // configurations the driver can select (ascending); empty -> the
  // launch's own shared bytes are subtracted (jobs built before the field)
  std::vector<std::uint64_t> shared_configurations;
  std::uint64_t launch_warps_per_cta = 0;
  // L2 capacity in weighted-line units (home copy 1 + reader replica
  // 1 + dirty 1); 0 -> resident_lines_lower_bound; both 0 -> no cap
  std::uint64_t l2_resident_lines = 0,
      l2_resident_lines_lower_bound = 0;
  // memory.l2_partition (schema 20)
  double far_hit_cycles = 0;
  std::uint64_t pool_count = 2;
  // per-SM L2 partition from TOPO (empty -> sm % pool_count)
  std::vector<std::uint32_t> sm_partition;
  bool replicate_read_lines = true;
  bool dirty_lines_near_both_sides = false;  // schema field 3
  // memory.shared_service (21) — bytes/cycle already converted by
  // MechanismRuntime (registry bytes/ns -> /GHz)
  double l2_read_bpc = 0, l2_write_bpc = 0, l2_max_inflight_bytes = 0;
  double l2_read_gpc_bpc = 0, l2_write_gpc_bpc = 0;
  // SM topology gpc (0xFFFFFFFF = no gpc): the per-GPC services below,
  // in series before the GPU-wide ones
  std::vector<std::uint32_t> sm_gpc;
  double dram_read_bpc = 0, dram_write_bpc = 0,
      dram_max_inflight_bytes = 0;
  // memory.l2_merge (22)
  double merge_window = 0;
  double merge_window_async = 0;
  bool merge_window_async_from_service = false;
  std::uint64_t merge_max_requests = 0;
  std::uint64_t merge_domain_sms = 0;
  std::uint32_t merge_domain_gpc = 0;
  std::vector<std::uint32_t> sm_merge_domain;
  double merge_bytes_per_cycle = 0;
  // memory.global_atomic (23) — ns fields converted
  double atomic_same_addr_service = 0, atomic_dependent_rt = 0,
      release_to_acquire = 0;
  std::uint64_t atomic_ops_per_window = 0;
  // memory.constant_cache (27)
  bool const_enabled = false;
  double const_same_line = 0, const_hit = 0, const_miss = 0;
  // optional: when >= 0 every bank-0 access returns now + this value;
  // uniform-path (ULDC/LDCU) accesses use const_bank0_uniform instead
  // when that field is present (per-path rule)
  double const_bank0 = -1.0, const_bank0_uniform = -1.0;
  // uniform-path cold hold (kernel_start v2): a uniform constant
  // access on a non-resident / still-filling line holds its warp at
  // issue until the fill completes; -1 = rule off (fields absent)
  double const_ubank0_hold = -1.0, const_uhold = -1.0;
  // kernel_start v3: a uniform load of a resident (incl. same-line)
  // line uses this result latency instead of const_same_line /
  // const_hit; active only when both cold-hold fields are set too
  double const_uresident_hold = -1.0;
  // kernel_start v4: a vector load of a bank-0 line not resident in
  // the SM's constant cache starts a fill and gets its result after
  // this latency; resident / in-flight lines keep const_bank0;
  // -1 = rule off (field absent), flat const_bank0 for every access
  double const_vbank0_cold = -1.0;
  // constant_paths v1: two per-SM resident sets (vector path 0,
  // uniform path 1), 64 B lines, per-path first-touch / re-read /
  // fill interval and a per-GPC 256 B block layer for banks != 0.
  // The model is on when both first-touch fields are present;
  // absent -> every rule above keeps today's behaviour byte for
  // byte (the old fields are unused while the model is on).
  double const_ufirst = -1.0, const_ureread = 0.0,
      const_ufill_interval = 0.0;
  double const_vfirst = -1.0, const_vreread = 0.0,
      const_vfill_interval = 0.0;
  double const_block_fetch = 0.0;
  std::uint64_t const_block_bytes = 0;
  // h100 few-warps step (superseded C5): uniform first touch uses
  // this value when at most few_warps_limit warps are resident on
  // the SM at fill start; both fields absent -> always const_ufirst
  double const_ufirst_few = -1.0;
  std::uint64_t const_few_warps_limit = 0;
  // constant_paths v2 C5 (h100): uniform first touch =
  // const_ufirst x max(min_smsps, n) / smsps_per_sm, n = SMSPs of
  // the SM holding a resident warp at fill start (0 = absent)
  std::uint64_t const_ufirst_min_smsps = 0;
  std::uint32_t const_smsps_per_sm = 0;
  bool const_paths() const {
    return const_ufirst >= 0.0 && const_vfirst >= 0.0;
  }
  std::uint64_t const_line_bytes = 0, const_capacity = 0;
  // memory.dram_port (32)
  std::uint64_t dram_port_inflight = 0;
  // memory_level dram/l2.loaded_latency_extra_cycles: piecewise-linear
  // (u, cycles) points over shared_utilization; empty -> no extra
  // (single cluster and unleveled jobs see none).  u = shared-service
  // bytes accepted in the last 1024 cycles / (rate * 1024).
  std::vector<std::pair<double, double>> dram_loaded_extra,
      l2_loaded_extra;
  // l2.tma_sector_bytes_per_cycle (field 62, tma_loaded protocol):
  // per-SM TMA/bulk delivery cap on the L2 and DRAM ports; 0 -> not
  // installed (bit-identical behaviour)
  double tma_sector_bpc = 0;
  // memory.load_port line_granular (schema 34): a request occupies
  // each load port once per distinct line it routes there (first
  // sector pays line_bytes / bpc, later same-line sectors 0); 0 ->
  // per-sector charging (bit-identical behaviour)
  bool load_port_line_granular = false;
  // memory.load_port lane_order (schema 34 field 1, gather_hit
  // protocol): a lane load's port occupancy is line switches in lane
  // order (per lane the line of its lowest-addressed sector) instead
  // of distinct lines; only under line_granular; 0 -> unchanged
  bool load_port_lane_order = false;
  // store_l2.shares_fill_port (field 63, tma_store_loaded protocol):
  // the per-SM memory port is shared by fills and store sectors;
  // absent -> independent directions (bit-identical behaviour)
  bool store_shares_fill_port = false;
  // store_l2.shared_store_shares_fill_port (field 6, R2): synchronous
  // shared-store acceptance and the dirty proxy-fence drain also take
  // turns with fills on the same per-SM port
  bool shared_store_shares_fill_port = false;
  // store_l2.orders_loads (field 64, load_after_stores protocol):
  // the per-SM load/store unit serves in issue order -- a plain load
  // starts only after this SM's store port has drained earlier store
  // sectors; absent -> unchanged
  bool store_orders_loads = false;
  // memory.dram_partition near/far split (row_gather run 3): a DRAM
  // miss costs dram_latency when the 4-KiB page's home side equals
  // the SM's gpc side, dram_latency + dram_far_extra otherwise; the
  // page's side is the parity of sector_address /
  // dram_home_page_bytes, the SM's side is bit sm_gpc[sm] of
  // dram_far_side_gpc_mask (an SM without a gpc is on side 0).  All
  // three 0 -> flat dram_latency (bit-identical behaviour)
  double dram_far_extra = 0;
  std::uint64_t dram_home_page_bytes = 0;
  std::uint64_t dram_far_side_gpc_mask = 0;
};

class MemoryHierarchy {
 public:
  bool dbg_ = false;  // CYCLESCOPE_DEBUG_INST pass-through
  struct Telemetry {
    std::uint64_t l1_sectors = 0, l2_sectors = 0, dram_sectors = 0,
        l1_bytes = 0, l2_bytes = 0, dram_bytes = 0;
    std::uint64_t l2_far_hits = 0, l2_replica_fills = 0,
        dirty_writebacks = 0, shared_waits = 0, atomics = 0,
        merge_merges = 0, l2_misses = 0, l1_port_wait = 0,
        l1_misses = 0, l2_evictions = 0;
    // L2 hits on sectors whose fill was still in flight, and the cycles
    // those hits were held for the fill
    std::uint64_t l2_inflight_hits = 0, l2_inflight_wait = 0;
    // atomics / reductions whose sector was not resident: DRAM fills
    // the L2 slice issued for the read-modify-write
    std::uint64_t atomic_fills = 0;
    std::uint64_t wait_cycles = 0;
    // memory.constant_cache: per-warp same-line hits, per-SM cache hits,
    // and capacity misses (see constant()).
    std::uint64_t const_same_line = 0, const_hits = 0, const_misses = 0;
    // cycles added by the loaded-latency extra (cycles added / sectors
    // charged), per level
    std::uint64_t loaded_extra_dram_cycles = 0, loaded_extra_dram_count = 0,
        loaded_extra_l2_cycles = 0, loaded_extra_l2_count = 0;
    // port-share serialization (store_l2.shares_fill_port): sectors
    // that had to wait behind the other direction's port occupancy,
    // and the cycles they waited
    std::uint64_t port_share_waits = 0, port_share_wait_cycles = 0;
    // load/store in-order unit (store_l2.orders_loads): plain loads
    // deferred behind this SM's outstanding store-port occupancy
    std::uint64_t ordered_load_waits = 0, ordered_load_wait_cycles = 0;
    // out-of-bounds (phantom) tiled-TMA sectors carried through the
    // load services (tma_oob v2)
    std::uint64_t phantom_sectors = 0;
  };

  enum class HitLevel { l1, l2, dram, store, constant, mixed };
  static constexpr std::uint32_t kSectorsPerLine = 4;  // 128 B / 32 B

  explicit MemoryHierarchy(MemoryConfig cfg = {});

  // A load: returns the cycle the last sector is back at the SM.
  // `sectors` are MSEC addresses.  Sets `level` to the dominant hit.
  // use_l1=false bypasses L1 lookup and allocation (async copies and
  // prefetches bypass L1 on both predecessors)
  // When `accept` is non-null it receives the cycle the SM-side
  // pipeline accepted the request = the first sector's L1-port
  // service start when the L1 port is modeled (l1.sector_bytes_per_cycle
  // live and use_l1), else `now`.  The far-side L2 / DRAM ports never
  // define acceptance: the warp's read barrier is an SM-local event.
  // `shared_free` (optional) receives the cycle the same request would
  // have returned at with the GPU-wide shared L2 / DRAM services
  // (memory.shared_service) free: every sector's per-SM port start plus
  // the level latency (far-hit extra included), skipping the
  // service_shared stage.  return - shared_free is the queueing the
  // request suffered in the GPU-shared services; leaf-timed transfers
  // (TMA / bulk) add exactly that on top of their leaf -- the per-SM
  // side (port, TMA service start) is already in their cluster-local
  // leaves (Simulator::on_semantic_issue).
  std::uint64_t load(std::uint32_t sm, std::uint64_t now,
                     const MemorySector* sectors, std::size_t n,
                     bool use_l1, HitLevel* level, std::uint32_t* l1_out,
                     std::uint32_t* l2_out, std::uint32_t* dram_out,
                     std::uint64_t* accept = nullptr,
                     std::uint64_t* shared_free = nullptr,
                     bool tma = false, bool l1_allocate = false,
                     bool ordered = false, bool async_copy = false,
                     std::size_t phantom_sectors = 0,
                     bool local = false);
  // insert a local-space sector into an SM's L1 residency, usable at
  // `ready` (local stores leave their lines resident)
  void l1_insert(std::uint32_t sm, std::uint64_t sector,
                 std::uint64_t ready);
  // A store: returns the acceptance cycle (store-L2 port, used for
  // completion / visibility; it does not gate the read barrier).
  // `fill_done` (optional, atomics) receives the cycle the last RMW
  // fill of a non-resident sector lands at the slice (0 when none):
  // the value an atomic returns is not available before it; a
  // reduction without a destination does not hold its warp for it
  // oldest undrained store-request drain end when the per-SM store
  // buffer (store_l2.outstanding_requests) is full, else now
  std::uint64_t store_issue_block(std::uint32_t sm, std::uint64_t now);
  // lane_store: the request is a lane-masked global STG (not TMA /
  // bulk / reduce / atomic): the partial-line port term and the
  // issue-interval floor apply.  merged: the request is folded into
  // the pending write of its line (warp_queue S4 fast path): the L2
  // line state is updated but no write-service, store-port or store
  // buffer occupancy is added
  std::uint64_t store(std::uint32_t sm, std::uint64_t now,
                      const MemorySector* sectors, std::size_t n,
                      bool atomic, std::uint64_t address0,
                      std::uint64_t* fill_done = nullptr,
                      double cost_ratio = 1.0, bool lane_store = false,
                      bool merged = false);
  // Constant-cache access.  `warp` is the issuing warp's index; the
  // same-line fast path tracks the last line each warp touched.
  // uniform cold-hold readiness: the earliest cycle the access may
  // issue (> now = held).  A missing line starts its fill here once
  // (same capacity / LRU logic as the constant() miss branch) and is
  // in flight on every later evaluation; a resident, fill-complete
  // line returns now with no state touched (caller runs constant()).
  std::uint64_t constant_uniform_hold(std::uint32_t sm,
                                      std::uint32_t warp,
                                      std::uint64_t now,
                                      std::uint64_t address,
                                      std::uint8_t* hold_class,
                                      bool count,
                                      // constant_paths v1 (only when
                                      // const_paths()): the block
                                      // layer's GPC key and the SM's
                                      // resident-warp count for C5
                                      std::uint64_t gpc = 0,
                                      std::uint64_t sm_warps = ~0ull,
                                      std::uint64_t sm_smsps = ~0ull);
  // issue-time bookkeeping of a held load: LRU tick + last-line, no
  // latency and no telemetry recount
  void constant_post_hold(std::uint32_t sm, std::uint32_t warp,
                          std::uint64_t address);
  // constant_paths v1 miss path (C2/C3) shared by constant() and
  // constant_uniform_hold()
  std::uint64_t const_paths_fill(std::uint32_t sm, std::uint32_t path,
                                 std::uint64_t now,
                                 std::uint64_t address,
                                 std::uint64_t line,
                                 std::uint64_t gpc,
                                 std::uint64_t sm_warps,
                                 std::uint64_t sm_smsps,
                                 std::uint8_t* hit_kind, bool count);
  std::uint64_t constant(std::uint32_t sm, std::uint32_t warp,
                         std::uint64_t now, std::uint64_t address,
                         std::size_t bytes, bool uniform = false,
                         // optional diagnostic (stall ledger):
                         // old model: 0 same-line, 1 hit, 2 miss;
                         // const_paths: 3 resident, 4 inflight,
                         // 5 first, 6 first+block
                         std::uint8_t* hit_kind = nullptr,
                         std::uint64_t gpc = 0,
                         std::uint64_t sm_warps = ~0ull,
                         std::uint64_t sm_smsps = ~0ull,
                         // mio_basic protocol v3 M1: cycle the
                         // const_paths re-read terms count from
                         // (default: now)
                         std::uint64_t reread_from = ~0ull);

  // architectural limits of sm_90 / sm_100 class SMs; the dispatcher's
  // 64-slot warp bitmap is the same limit
  static constexpr std::uint64_t kMaxCtasPerSm = 32;
  static constexpr std::uint64_t kMaxWarpsPerSm = 64;
  static constexpr std::uint64_t kReservedSharedPerCtaBytes = 1024;
  // the launch's shared-memory carveout: smallest listed configuration
  // that fits n_res CTAs' per-CTA need, else the largest; empty list ->
  // the launch's own shared bytes
  std::uint64_t shared_configuration_bytes() const;
  // L1 data budget = data_array - reserved - configuration
  std::uint64_t l1_capacity_bytes() const;

  std::uint32_t sm_side(std::uint32_t sm) const {
    return sm < cfg_.sm_partition.size()
               ? cfg_.sm_partition[sm]
               : static_cast<std::uint32_t>(sm % cfg_.pool_count);
  }
  // l2_merge domain: merge entries are keyed per merge domain. With
  // merge_domain_gpc the kinds it selects (gpc_kind: 1 = TMA / bulk
  // loads, 2 = plus async copies) key on the SM topology gpc
  // (sm_merge_domain, filled by the simulator; a missing table or a
  // 0xFFFFFFFF entry is a missing fact) in a separate key space from
  // the other kinds, which keep floor(sm / merge_domain_sms); domain 0
  // keeps the sector key alone (former rule)
  std::uint64_t merge_key_(std::uint32_t sm, std::uint64_t sector,
                           bool gpc_kind) const {
    if (gpc_kind) {
      if (sm >= cfg_.sm_merge_domain.size() ||
          cfg_.sm_merge_domain[sm] == 0xFFFFFFFFu)
        throw EngineError("missing_fact:sm_topology (sm " +
                          std::to_string(sm) + ")");
      return (sector << 9) | 0x100 | cfg_.sm_merge_domain[sm];
    }
    if (cfg_.merge_domain_sms == 0)
      return sector;
    const std::uint64_t dom = sm / cfg_.merge_domain_sms;
    if (dom >= 256)
      throw EngineError("unsupported:l2_merge_domain ()");
    if (cfg_.merge_domain_gpc != 0)
      return (sector << 9) | dom;
    return (sector << 8) | dom;
  }
  // per-SM memory port shared by fills and store sectors: place this
  // sector's port occupancy at its per-direction port start; returns the
  // extra wait the other direction's traffic imposes (0 when off).
  // Only the measured exclusion is modeled: fills (either level) vs
  // store sectors; same-direction fills never wait on each other
  enum class PortUser { fill_l2, fill_dram, store, shared_store };
  // book=false only affects the shared_store branch: the same wait is
  // computed from the fill ledgers but ps.shared_store is not written
  // (the caller serializes the same occupancy on its own frontier, so
  // booking it here as well would charge it twice) and no telemetry
  // is bumped
  double port_share_wait(std::uint32_t sm, PortUser who, double start,
                         double occupy, bool book = true);
  // shared-store acceptance port occupancy (R2): gated on
  // shared_store_shares_fill_port; waits only on fills and books
  // occupancy so fills see it (self-serialization stays with the SM
  // acceptance frontier)
  double shared_store_port_wait(std::uint32_t sm, double start,
                                double occupy, bool book = true) {
    if (!cfg_.shared_store_shares_fill_port) return 0.0;
    return port_share_wait(sm, PortUser::shared_store, start, occupy,
                           book);
  }
  Telemetry tele;

 private:
  struct L2Line {
    std::uint64_t tag = 0;
    // sector-granular presence per side (bit k = sector k of the line
    // is resident on that side): an L2 miss fills only the missed
    // sector, the line's other sectors still come from DRAM on first
    // touch.  A line is "present" on a side when any sector is.
    std::uint32_t sectors[2] = {};
    // cycle the sector's data is in that side's array (a DRAM fill or
    // a replica install completes later than the request that started
    // it): a hit on a sector whose fill is still in flight returns no
    // earlier than this, like the L1 fill-ready rule
    std::uint64_t ready[2][kSectorsPerLine] = {};
    bool dirty = false;
    std::uint64_t last_access = 0;
    std::uint32_t weight = 0;      // copies + dirty, LRU accounting
    bool in_lru = false;
    std::list<std::uint64_t>::iterator lru_it;
  };
  // per-(level, SM) port: fractional busy frontier + a ring of
  // in-flight request completion times (capacity = outstanding;
  // 0 -> none). outstanding counts warp-level requests, not sectors
  // (hopper-prototype LevelQueue charges 1 credit per access; the b300 w4
  // probe's "32 warps x 8 independent loads" is the same unit): one
  // inflight entry per load()/store() call routed to this port, each
  // sector still paying sector_bytes/bpc of port bandwidth
  struct Port {
    double busy_until = 0.0;
    std::deque<double> inflight;
  };
  // L2 merge window bookkeeping per sector
  struct MergeEntry {
    double expiry = 0.0;
    std::uint64_t completion = 0;
    std::uint64_t count = 0;
  };
  // per-SM L1 residency: sector-granular LRU; a fill only hits once
  // its recorded completion cycle has passed
  struct L1Set {
    std::list<std::uint64_t> order;
    std::unordered_map<std::uint64_t,
                       std::pair<std::list<std::uint64_t>::iterator,
                                 std::uint64_t>> pos;  // -> {it,ready}
    std::uint64_t cap = 0;
    bool has(std::uint64_t s, std::uint64_t now) const {
      auto it = pos.find(s);
      return it != pos.end() && now >= it->second.second;
    }
    void touch(std::uint64_t s);
    void insert(std::uint64_t s, std::uint64_t ready);
  };

  // GPU-shared bandwidth as capacity per time bucket: a request
  // arriving at `when` takes the earliest free capacity at or after
  // `when`, so requests arriving out of time order (per-SM ports push
  // their starts to different futures) never queue behind later ones
  struct SharedService {
    std::map<std::uint64_t, double> used;  // bucket -> bytes placed
    std::uint64_t floor = 0;               // buckets below are full
    // union-find style skip list: b -> first bucket >= b that may
    // still have free capacity; entries below floor are dead.  A
    // bucket is only linked when saturated at mark_bpc (the largest
    // cap the queue can see — shared_dram_ is served at both read
    // and write bpc), so a link is a safe skip for every caller.
    std::unordered_map<std::uint64_t, std::uint64_t> next_free;
  };
  static constexpr std::uint64_t kServiceBucketCycles = 32;

  MemoryConfig cfg_;
  std::unordered_map<std::uint64_t, L2Line> l2_;
  std::list<std::uint64_t> l2_lru_;
  std::uint64_t l2_weight_total_ = 0;
  std::unordered_map<std::uint32_t, L1Set> l1_sets_;
  // per-SM constant cache: line -> {last-use tick (LRU), fill-ready
  // cycle}; a missed line is inserted at issue time, so an entry can be
  // resident while its fill is still in flight — hits before `ready`
  // return the fill-ready cycle, not the hit latency.  Separate from
  // the data-L1 residency table so line spaces cannot collide.
  struct ConstLine {
    std::uint64_t tick;
    std::uint64_t ready;
  };
  std::unordered_map<std::uint32_t,
                   std::unordered_map<std::uint64_t, ConstLine>>
      const_resident_;
  // per-warp last constant line for the same-line fast path
  std::unordered_map<std::uint32_t,
                   std::unordered_map<std::uint64_t, std::uint64_t>>
      const_last_line_;
  // constant_paths v1: per-SM per-path resident sets (key sm*2+path,
  // path 0 = vector, 1 = uniform); per-set previous fill-ready for
  // the fill interval; per-GPC block fetch state (no eviction — a
  // fetched block stays)
  std::unordered_map<std::uint64_t,
                   std::unordered_map<std::uint64_t, ConstLine>>
      const_resident_p_;
  std::unordered_map<std::uint64_t, std::uint64_t> const_last_fill_;
  std::unordered_map<std::uint64_t,
                   std::unordered_map<std::uint64_t, std::uint64_t>>
      const_blocks_;
  std::uint64_t const_tick_ = 0;
  std::unordered_map<std::uint64_t, MergeEntry> merge_last_;
  // GPU-level fractional delivery queue for merged sectors
  // (delivered_bytes_per_cycle leaf)
  double merge_delivery_busy_ = 0.0;
  // GPU shared service queues: L2 read / L2 write / DRAM
  SharedService shared_l2_read_, shared_l2_write_, shared_dram_;
  // per-GPC L2 read / store services, in series before the GPU-wide
  // ones (sized to the largest gpc + 1 by the ctor)
  std::vector<SharedService> gpc_l2_read_, gpc_l2_write_;

  std::uint64_t l1_cap_lines() const;
  bool l1_modeled() const { return cfg_.l1_data_array_bytes != 0; }
  std::uint32_t l2_weight(const L2Line& l) const {
    return (l.sectors[0] ? 1u : 0u) + (l.sectors[1] ? 1u : 0u) +
           (l.dirty ? 1u : 0u);
  }
  // bit of `sector` (a sector address) within its line
  std::uint32_t sector_index(std::uint64_t sector) const {
    const std::uint64_t sb = cfg_.sector_bytes ? cfg_.sector_bytes : 32;
    return static_cast<std::uint32_t>((sector % cfg_.line_bytes) / sb);
  }
  std::uint32_t sector_bit(std::uint64_t sector) const {
    return 1u << sector_index(sector);
  }
  void l2_touch(std::uint64_t line, L2Line& l, std::uint64_t now);
  void l2_account(L2Line& l);
  void l2_evict(std::uint64_t line, std::uint64_t now);
  void l2_ensure_capacity(std::uint64_t self, std::uint64_t now);
  // sector occupies the port for sector_bytes/bpc cycles (fallback
  // max(1, issue_interval)); returns the service-start cycle.
  // new_request=true pushes one inflight entry (unit: warp-level
  // request — see Port); =false only extends the current request's
  // entry so it lives until its last sector returns
  double port_acquire(Port& p, double now, std::uint64_t outstanding,
                      double occupy, double completion_after,
                      bool new_request);
  // bucket-placement half of service_shared: occupies capacity in
  // the shared queue and returns the fractional service-end
  // (max(end, when + bytes / bpc)); no telemetry
  double place_shared(SharedService& q, double when, double bytes,
                      double bpc, double mark_bpc);
  double service_shared(SharedService& q, double when, double bytes,
                        double bpc, double mark_bpc);
  // fraction of `bpc` capacity the queue committed in the 32 buckets
  // (1024 cycles) ending before bucket(when); buckets below the
  // compacted floor count as full
  double utilization(const SharedService& q, double when,
                     double bpc) const;
  // piecewise-linear (x, y) evaluation; x outside the points clamps
  static double eval_linear(
      const std::vector<std::pair<double, double>>& pts, double x);
  // service-end `bytes` would get at `when` without placing them (no
  // capacity consumed): the earliest bucket at or after `when` with
  // room, served at bpc
  double peek_shared(const SharedService& q, double when, double bytes,
                     double bpc) const;
  double dram_bpc(bool write) const {
    return write ? cfg_.dram_write_bpc : cfg_.dram_read_bpc;
  }
  // DRAM-miss latency of `sector_address` fetched for `sm`: near on
  // the SM's side of the page split, near + far extra across it
  // (MemoryConfig dram_far_* fields; all 0 -> dram_latency)
  double dram_latency_at(std::uint32_t sm,
                         std::uint64_t sector_address) const {
    if (cfg_.dram_far_extra <= 0.0 || cfg_.dram_home_page_bytes == 0)
      return cfg_.dram_latency;
    const std::uint64_t side =
        (sm < cfg_.sm_gpc.size() && cfg_.sm_gpc[sm] != 0xFFFFFFFFu &&
         ((cfg_.dram_far_side_gpc_mask >> cfg_.sm_gpc[sm]) & 1u))
            ? 1 : 0;
    const std::uint64_t home =
        (sector_address / cfg_.dram_home_page_bytes) & 1u;
    return cfg_.dram_latency +
           (side != home ? cfg_.dram_far_extra : 0.0);
  }
  std::unordered_map<std::uint64_t, Port> port_l1_, port_l2_,
      port_dram_, port_sl2_, port_sdram_;
  // rule S (lsu_accept addendum): per-SM deque of store-request drain
  // ends (the request's shared-L2-write service end); a global/generic
  // STG issued while >= store_l2_outstanding entries are undrained
  // stalls the issuing warp until the oldest drains
  std::unordered_map<std::uint32_t, std::deque<std::uint64_t>>
      store_drain_;
  // rule R1 (store_l2.shares_fill_port): per-SM port occupancy in
  // 32-cycle buckets, one ledger per direction; fill ledgers do not
  // queue on each other, only fills vs store sectors alternate on
  // the port
  struct PortShare {
    std::vector<double> fill_l2, fill_dram, store, shared_store;
  };
  std::unordered_map<std::uint64_t, PortShare> port_share_;
  // per-direction virtual finish under sector-by-sector alternation
  // (0 = fills, 1 = store); bounds the wait returned to a sector
  std::unordered_map<std::uint64_t, std::array<double, 2>> turn_finish_;
};

}  // namespace cyclescope
