#include "simulator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <optional>

#include "generated/parameter_schema.hpp"

namespace cyclescope {

namespace ML = core::parameter_fields::MEMORY_LEVEL;
namespace LP = core::parameter_fields::MEMORY_L2_PARTITION;
namespace SS = core::parameter_fields::MEMORY_SHARED_SERVICE;
namespace MG = core::parameter_fields::MEMORY_L2_MERGE;
namespace GA = core::parameter_fields::MEMORY_GLOBAL_ATOMIC;
namespace CC = core::parameter_fields::MEMORY_CONSTANT_CACHE;
namespace DP = core::parameter_fields::MEMORY_DRAM_PORT;
namespace LOADP = core::parameter_fields::MEMORY_LOAD_PORT;
namespace MDP = core::parameter_fields::MEMORY_DRAM_PARTITION;
namespace IT = core::parameter_fields::INSTRUCTION_TEMPLATE;
namespace RS = core::parameter_fields::RESOURCE;
namespace SC = core::parameter_fields::SCHEDULER;
namespace TR = core::parameter_fields::TIMING_ROW;

namespace {

constexpr std::uint8_t SCOPE_SMSP = 0, SCOPE_SM = 1, SCOPE_CLUSTER = 2;
// SM hardware fact for sm_90a/sm_103a; moves to LNCH device facts in
// step 4 (spec decision, documented in the 3-1 report).
constexpr std::uint32_t SMSPS_PER_SM = 4;
// SASS barrier index 7 (and 0xFF) means "none" (ControlWord default).
constexpr std::uint8_t NO_BARRIER = 7;

std::uint64_t accumulate_spacing(double interval, double& residual) {
  // predecessor accumulate_measured_spacing: fractional intervals
  // carry a residual so long-run cadence matches the measured mean.
  if (!std::isfinite(interval) || interval <= 0.0) {
    residual = 0.0;
    return 1;
  }
  if (interval <= 1.0) {
    residual = 0.0;
    return 1;
  }
  const double total = interval + residual;
  const double spacing = std::max(1.0, std::floor(total + 1.0e-12));
  residual = total - spacing;
  if (std::abs(residual) < 1.0e-12) residual = 0.0;
  return static_cast<std::uint64_t>(spacing);
}

}  // namespace

Simulator::Simulator(const Job& job, MechanismRuntime& rt,
                     Options opt)
    : job_(job), rt_(rt),
      arch_(make_architecture(job.launch.device)), opt_(opt) {
  smsps_per_sm_ = SMSPS_PER_SM;
}

void Simulator::enqueue(Event::Kind k, std::uint64_t cycle,
                        std::uint32_t warp, std::uint32_t inst,
                        std::uint64_t payload) {
  queue_.push(Event{k, cycle, warp, inst, payload, seq_++});
}

void Simulator::place_warps() {
  warps_.resize(job_.warps.size());
  pending_warps_ = job_.warps.size();
  std::uint32_t max_sm = 0;
  for (const auto& c : job_.ctas)
    if (c.sm_id != kNone32) max_sm = std::max(max_sm, c.sm_id + 1);
  for (const auto& p : job_.placements)
    if (p.placement == 1) max_sm = std::max(max_sm, p.sm_id + 1);
  if (max_sm == 0) max_sm = 1;
  std::unordered_map<std::uint32_t, const PlacementRow*> pl;
  for (const auto& p : job_.placements)
    if (p.placement == 1) pl[p.cta] = &p;
  // a CTA is first-wave placed only when a PLCE row names it; the
  // CTAS.sm_id trace-placement fallback applies only when the PLCE
  // table is empty (probe placements cover just the first wave —
  // CTAs without a PLCE row queue for dispatch)
  const bool have_plce = !job_.placements.empty();
  const auto cta_placed = [&](const CtaInfo& c) {
    if (have_plce) return pl.count(c.cta_index) != 0;
    return c.sm_id != kNone32;
  };
  std::vector<std::uint32_t> unplaced;
  for (std::uint32_t s = 0; s < job_.segment_instance_base.size(); ++s)
    for (const auto& c : job_.ctas)
      if (job_.cta_segment[c.cta_index] == s && !cta_placed(c)) {
        unplaced.push_back(s);
        break;
      }
  if (dispatch_.enabled && !unplaced.empty()) {
    if (!job_.launch.occupancy_blocks_per_sm)
      throw EngineError("missing_fact:occupancy_blocks_per_sm ()");
    if (!job_.launch.sm_count)
      throw EngineError("missing_fact:sm_count ()");
    for (const auto s : unplaced)
      if (segment_cta_count_[s] > 1 && job_.topology.empty())
        throw EngineError("missing_fact:sm_topology ()");
  }
  const std::uint32_t dsm =
      dispatch_.enabled ? std::max<std::uint32_t>(
          max_sm, job_.launch.sm_count ? job_.launch.sm_count : 1)
                        : max_sm;
  dispatcher_.configure(dsm, job_.launch.occupancy_blocks_per_sm,
                        job_.topology);
  smsps_.assign(std::max<std::uint32_t>(max_sm, 1),
                std::vector<SmspState>(smsps_per_sm_));
  // first wave: PLCE rows use (sm_id, warp_slot_base); trace-placed
  // CTAs use (CTAS.sm_id, the CTA's smallest WARP.warp_slot); entry =
  // rank * first_wave_interval in (sm, slot) order; slots are claimed
  // up front when dispatch is enabled
  struct WaveCta { std::uint32_t cta, sm, base; };
  std::vector<WaveCta> wave;
  for (const auto& c : job_.ctas) {
    if (auto it = pl.find(c.cta_index); it != pl.end())
      wave.push_back({c.cta_index, it->second->sm_id,
                      it->second->warp_slot_base});
    else if (!have_plce && c.sm_id != kNone32) {
      std::uint32_t base = kNone32;
      for (const auto& wi : job_.warps)
        if (wi.cta_index == c.cta_index)
          base = std::min(base, wi.warp_slot);
      wave.push_back({c.cta_index, c.sm_id,
                      base == kNone32 ? 0 : base});
    }
  }
  std::sort(wave.begin(), wave.end(), [](const WaveCta& a,
                                         const WaveCta& b) {
    return std::tie(a.sm, a.base) < std::tie(b.sm, b.base);
  });
  std::unordered_map<std::uint32_t, std::uint64_t> entry_of;
  for (std::size_t k = 0; k < wave.size(); ++k) {
    const std::uint64_t e =
        dispatch_.enabled ? k * dispatch_.first_wave_interval : 0;
    entry_of[wave[k].cta] = e;
    dispatch_last_entry_ =
        std::max(dispatch_last_entry_, static_cast<double>(e));
    if (dispatch_.enabled) {
      dispatcher_.occupy(wave[k].sm, wave[k].base,
                         job_.ctas[wave[k].cta].warp_count);
      tele2_[kTele2DispatchWaves] = std::max<std::uint64_t>(
          tele2_[kTele2DispatchWaves], ++sm_entries_[wave[k].sm]);
    }
  }
  for (const auto s : unplaced) dispatch_queue_.push_back({s, 0});
  cta_live_warps_.assign(job_.ctas.size(), 0);
  cta_dependents_launch_.assign(job_.ctas.size(), 0);
  cta_sm_.assign(job_.ctas.size(), 0);
  cta_slot_.assign(job_.ctas.size(), 0);
  // per-segment/per-CTA lookup tables for dispatch hot paths
  segment_ctas_.assign(job_.segment_instance_base.size(), {});
  segment_wpc_.assign(job_.segment_instance_base.size(), 0);
  cta_warps_.assign(job_.ctas.size(), {});
  for (const auto& c : job_.ctas) {
    const auto s = job_.cta_segment[c.cta_index];
    segment_ctas_[s].push_back(c.cta_index);
    segment_wpc_[s] = std::max(segment_wpc_[s], c.warp_count);
  }
  for (std::uint32_t i = 0; i < job_.warps.size(); ++i)
    cta_warps_[job_.warps[i].cta_index].push_back(i);
  std::uint32_t rr_sm_ = 0;  // dispatch-disabled CTA round-robin
  std::unordered_map<std::uint32_t, std::uint32_t> rr_cta_sm;
  for (std::uint32_t i = 0; i < job_.warps.size(); ++i) {
    const auto& wi = job_.warps[i];
    auto& w = warps_[i];
    w.warp_index = i;
    w.cta = wi.cta_index;
    const auto& cta = job_.ctas[wi.cta_index];
    ++cta_live_warps_[w.cta];
    w.next_instance = wi.first_instance;
    w.end_instance = wi.first_instance + wi.instance_count;
    update_ready_at(w);
    w.sm = cta.sm_id;
    w.slot = wi.warp_slot;
    if (auto it = pl.find(w.cta); it != pl.end()) {
      w.sm = it->second->sm_id;
      w.slot = it->second->warp_slot_base + wi.warp_in_cta;
      cta_sm_[w.cta] = w.sm;
      cta_slot_[w.cta] = it->second->warp_slot_base;
    } else if (!have_plce && w.sm != kNone32) {
      // trace placement: CTAS.sm_id + WARP.warp_slot
      cta_sm_[w.cta] = w.sm;
      cta_slot_[w.cta] = wi.warp_slot;
    }
    if (cta_placed(cta)) {
      w.entry_cycle = entry_of[w.cta];
    } else if (dispatch_.enabled) {
      // queued cluster: stays inert until Event::kDispatch lands
      w.resident = false;
      w.sm = 0; w.slot = 0; w.smsp = 0;
      if (i >= active_pos_.size()) active_pos_.resize(i + 1);
      active_pos_[i] = active_.size();
      active_.push_back(i);
      continue;
    } else {
      // disabled: whole CTA round-robins onto one SM at cycle 0
      if (!rr_cta_sm.count(w.cta))
        rr_cta_sm[w.cta] =
            rr_sm_++ % std::max<std::uint32_t>(1, max_sm);
      w.sm = rr_cta_sm[w.cta];
      w.slot = wi.warp_slot;
    }
    smsps_.resize(std::max<std::size_t>(smsps_.size(), w.sm + 1),
                  std::vector<SmspState>(smsps_per_sm_));
    w.smsp = w.slot % smsps_per_sm_;
    w.resident = true;
    sm_used_.insert(w.sm);
    smsps_[w.sm][w.smsp].warps.push_back(i);
    if (i >= active_pos_.size()) active_pos_.resize(i + 1);
    active_pos_[i] = active_.size();
    active_.push_back(i);
  }
  // launch prefetch: the first resident warp on each SM
  for (const auto i : active_) {
    const auto& w = warps_[i];
    if (!w.resident || !sm_prefetched_.insert(w.sm).second) continue;
    const std::uint64_t* ca = code_address_of(w.next_instance);
    if (ca) {
      const std::uint64_t p = fetch_.parallelism();
      const std::uint32_t ch = static_cast<std::uint32_t>(
          (p <= 4 ? w.smsp : w.slot) % p);
      fetch_.launch_prefetch(w.sm, w.smsp, ch, *ca,
                             w.entry_cycle);
    }
  }
  // initial queue pass: anything placeable enters after the last
  // first-wave entry at serial_per_cta spacing
  try_dispatch_queue(
      static_cast<std::uint64_t>(dispatch_last_entry_));
}

void Simulator::try_dispatch_queue(std::uint64_t now) {
  while (!dispatch_queue_.empty()) {
    const auto [seg, since] = dispatch_queue_.front();
    const std::uint32_t size = segment_cta_count_[seg];
    auto placed = dispatcher_.place_cluster(size, segment_wpc_[seg]);
    if (!placed) break;  // wait for the next capacity release
    // consecutive entries keep >= serial_per_cta * size spacing off
    // the previous cluster's entry cycle; the accumulator keeps the
    // fraction (serial_per_cta is not a whole number of cycles)
    const double t = std::max(
        static_cast<double>(now),
        dispatch_last_entry_ + dispatch_.serial_per_cta * size);
    dispatch_last_entry_ = t;
    const std::uint64_t entry =
        static_cast<std::uint64_t>(std::ceil(t - 1e-9));
    for (const auto& [sm, base] : *placed)
      dispatcher_.occupy(sm, base, segment_wpc_[seg]);
    dispatch_entries_.push_back(DispatchEntry{seg, *placed});
    enqueue(Event::kDispatch, entry, kNone32, 0,
            dispatch_entries_.size() - 1);
    tele2_[kTele2DispatchEntries] += size;
    tele2_[kTele2DispatchWaitCycles] += entry - since;
    dispatch_queue_.pop_front();
  }
}

void Simulator::admit_cluster(std::uint32_t entry_index) {
  const auto& e = dispatch_entries_[entry_index];
  std::size_t slot = 0;
  std::vector<std::uint32_t> newly_resident;
  for (const auto c : segment_ctas_[e.segment]) {
    const auto [sm, base] = e.slots[slot++];
    cta_sm_[c] = sm;
    cta_slot_[c] = base;
    for (const auto i : cta_warps_[c]) {
      auto& w = warps_[i];
      if (w.resident) continue;
      const auto& wi = job_.warps[i];
      w.sm = sm;
      w.slot = base + wi.warp_in_cta;
      w.smsp = w.slot % smsps_per_sm_;
      smsps_.resize(std::max<std::size_t>(smsps_.size(), w.sm + 1),
                    std::vector<SmspState>(smsps_per_sm_));
      w.entry_cycle = now_;
      w.resident = true;
      sm_used_.insert(w.sm);
      smsps_[w.sm][w.smsp].warps.push_back(i);
      smsps_[w.sm][w.smsp].needs_eval = true;
      newly_resident.push_back(i);
    }
    ++sm_entries_[sm];
    tele2_[kTele2DispatchWaves] =
        std::max<std::uint64_t>(tele2_[kTele2DispatchWaves],
                                sm_entries_[sm]);
  }
  // launch prefetch for a newly occupied SM: only warps made resident
  // above can hit an un-prefetched SM (every resident warp's SM was
  // inserted by the same loop when it became resident); sorting by
  // active_ position reproduces the original first-in-active_ pick
  std::sort(newly_resident.begin(), newly_resident.end(),
            [&](std::uint32_t a, std::uint32_t b) {
              return active_pos_[a] < active_pos_[b];
            });
  for (const auto i : newly_resident) {
    const auto& w = warps_[i];
    if (!w.resident || !sm_prefetched_.insert(w.sm).second) continue;
    const std::uint64_t* ca = code_address_of(w.next_instance);
    if (ca) {
      const std::uint64_t p = fetch_.parallelism();
      const std::uint32_t ch = static_cast<std::uint32_t>(
          (p <= 4 ? w.smsp : w.slot) % p);
      fetch_.launch_prefetch(w.sm, w.smsp, ch, *ca, now_);
    }
  }
}

// ---- load-time setup: resource defs, template->USES, policy ----

namespace {
struct ResolvGuard {};  // marker
}

const Fields& Simulator::try_resolve(std::uint64_t use_index) {
  try {
    return rt_.resolve(use_index, engine_facts_, nullptr);
  } catch (const EngineError&) {
    // leaf guards on engine axes resolved at issue; retry lazily.
    pending_uses_.push_back(use_index);
    return empty_fields_;
  }
}

void Simulator::load_setup() {
  // engine-axis slots by name from the AXIS table (P2); value 0 at
  // load so guard evaluation at load is deterministic.
  static const Fields kEmpty;
  empty_fields_ = kEmpty;
  tmpl_use_.assign(job_.templates.size(), kNone64);
  svc_cache_.assign(job_.templates.size(), nullptr);
  svc_cached_.assign(job_.templates.size(), 0);
  timing_use_.assign(job_.timing.size(), kNone64);
  time_cache_.assign(job_.timing.size(), nullptr);
  time_cached_.assign(job_.timing.size(), 0);
  std::unordered_map<std::string, std::uint32_t> name_to_id;
  for (std::uint32_t i = 0; i < job_.axis_names.size(); ++i)
    name_to_id[std::string(
        job_.strings.c_str() + job_.axis_names[i].name)] = i;
  for (const char* nm : {"ready_warps_smsp", "ready_warps_sm",
                         "cache_hit_level"}) {
    auto it = name_to_id.find(nm);
    if (it != name_to_id.end()) {
      std::uint32_t slot = engine_facts_.size();
      engine_facts_.emplace_back(it->second, 0u);
      if (!std::strcmp(nm, "ready_warps_smsp")) ax_ready_smsp_ = slot;
      else if (!std::strcmp(nm, "ready_warps_sm")) ax_ready_sm_ = slot;
      else ax_cache_hit_ = slot;
    } else {
      engine_facts_.emplace_back(0u, 0u);  // unused slot
      if (!std::strcmp(nm, "ready_warps_smsp")) ax_ready_smsp_ =
          engine_facts_.size() - 1;
      else ax_ready_sm_ = engine_facts_.size() - 1;
    }
  }
  // resource definitions from RESOURCE-schema USES rows
  for (std::uint64_t i = 0; i < job_.uses.size(); ++i) {
    const auto& u = job_.uses[i];
    if (u.schema_id == 2) {
      ResDef d;
      const auto& f = rt_.resolve(i, engine_facts_, nullptr);
      d.scope = static_cast<std::uint32_t>(
          MechanismRuntime::field(f, RS::SCOPE, 0.0));
      d.copies = std::max<std::uint64_t>(
          1, static_cast<std::uint64_t>(
              MechanismRuntime::field(f, RS::COPY_COUNT, 1.0)));
      // 0 = no explicit in-flight cap (the initiation interval models
      // the unit); a positive value is a measured in-flight limit
      d.max_inflight = MechanismRuntime::field(
          f, RS::MAX_INFLIGHT_PER_COPY, 0.0);
      d.interval = MechanismRuntime::field(
          f, RS::INITIATION_INTERVAL_CYCLES, 1.0);
      res_def_[u.target_index] = d;
    }
    if (u.schema_id == 1 && u.target == 0) {
      if (u.target_index >= tmpl_use_.size())
        tmpl_use_.resize(u.target_index + 1, kNone64);
      tmpl_use_[u.target_index] = i;
    }
    if (u.schema_id == 7)
      sched_use_ = i;
    if (u.schema_id == 24) dispatch_use_ = i;
    if (u.schema_id == 25) lifecycle_use_ = i;
    if (u.schema_id == 26) fetch_use_ = i;
    if (u.schema_id == 33) operand_use_ = i;
    if (u.target == 1) {  // TIME-row binding: target_index = row id
      if (u.target_index >= timing_use_.size())
        timing_use_.resize(u.target_index + 1, kNone64);
      timing_use_[u.target_index] = i;
    }
    // memory parameters (3-2): schema-id -> MemoryConfig fields
    if (u.schema_id == 20) {
      const auto& f = try_resolve(i);
      mem_cfg_.pool_count = std::max<std::uint64_t>(1,
          (std::uint64_t)MechanismRuntime::field(
              f, LP::POOL_COUNT, 2.0));
      mem_cfg_.replicate_read_lines =
          MechanismRuntime::field(f, LP::REPLICATE_READ_LINES, 1.0) != 0;
      mem_cfg_.far_hit_cycles =
          MechanismRuntime::field(f, LP::FAR_HIT_CYCLES, 0.0);
      mem_cfg_.dirty_lines_near_both_sides =
          MechanismRuntime::field(
              f, LP::DIRTY_LINES_NEAR_BOTH_SIDES, 0.0) != 0;
    }
    if (u.schema_id == 21) {
      const auto& f = try_resolve(i);
      mem_cfg_.l2_read_bpc =
          MechanismRuntime::field(f, SS::L2_READ_BYTES_PER_NS, 0.0);
      mem_cfg_.l2_write_bpc =
          MechanismRuntime::field(f, SS::L2_WRITE_BYTES_PER_NS, 0.0);
      mem_cfg_.l2_max_inflight_bytes =
          MechanismRuntime::field(f, SS::L2_MAX_INFLIGHT_BYTES, 0.0);
      mem_cfg_.dram_read_bpc =
          MechanismRuntime::field(f, SS::DRAM_READ_BYTES_PER_NS, 0.0);
      mem_cfg_.dram_write_bpc =
          MechanismRuntime::field(f, SS::DRAM_WRITE_BYTES_PER_NS, 0.0);
      mem_cfg_.l2_read_gpc_bpc = MechanismRuntime::field(
          f, SS::L2_READ_GPC_BYTES_PER_NS, 0.0);
      mem_cfg_.l2_write_gpc_bpc = MechanismRuntime::field(
          f, SS::L2_WRITE_GPC_BYTES_PER_NS, 0.0);
    }
    if (u.schema_id == 22) {
      const auto& f = try_resolve(i);
      mem_cfg_.merge_window =
          MechanismRuntime::field(f, MG::WINDOW_CYCLES, 0.0);
      mem_cfg_.merge_bytes_per_cycle = MechanismRuntime::field(
          f, MG::DELIVERED_BYTES_PER_CYCLE, 0.0);
      mem_cfg_.merge_max_requests = (std::uint64_t)
          MechanismRuntime::field(f, MG::MAX_REQUESTS_PER_READ, 0.0);
      mem_cfg_.merge_domain_sms =
          (std::uint64_t)MechanismRuntime::field(f, MG::DOMAIN_SMS, 0.0);
      mem_cfg_.merge_window_async = MechanismRuntime::field(
          f, MG::ASYNC_COPY_WINDOW_CYCLES, 0.0);
      mem_cfg_.merge_window_async_from_service =
          MechanismRuntime::field(
              f, MG::ASYNC_COPY_WINDOW_FROM_SERVICE, 0.0) != 0;
      mem_cfg_.merge_domain_gpc =
          (std::uint32_t)MechanismRuntime::field(f, MG::DOMAIN_GPC, 0.0);
      if (mem_cfg_.merge_domain_gpc > 2)
        throw EngineError("unsupported:l2_merge_domain_gpc ()");
    }
    if (u.schema_id == 23) {
      const auto& f = try_resolve(i);
      mem_cfg_.atomic_same_addr_service = MechanismRuntime::field(
          f, GA::SAME_ADDRESS_SERVICE_NS, 0.0);
      mem_cfg_.atomic_dependent_rt = MechanismRuntime::field(
          f, GA::DEPENDENT_ROUND_TRIP_NS, 0.0);
      mem_cfg_.release_to_acquire =
          MechanismRuntime::field(f, GA::RELEASE_TO_ACQUIRE_NS, 0.0);
    }
    if (u.schema_id == 27) {
      const auto& f = try_resolve(i);
      mem_cfg_.const_enabled =
          MechanismRuntime::field(f, CC::ENABLED, 0.0) != 0;
      mem_cfg_.const_line_bytes = (std::uint64_t)
          MechanismRuntime::field(f, CC::LINE_BYTES, 64.0);
      mem_cfg_.const_capacity = (std::uint64_t)
          MechanismRuntime::field(f, CC::CAPACITY_BYTES, 0.0);
      mem_cfg_.const_same_line =
          MechanismRuntime::field(f, CC::SAME_LINE_LATENCY_CYCLES, 0.0);
      mem_cfg_.const_hit =
          MechanismRuntime::field(f, CC::HIT_LATENCY_CYCLES, 0.0);
      mem_cfg_.const_miss =
          MechanismRuntime::field(f, CC::MISS_LATENCY_CYCLES, 0.0);
      {
        const auto itb0 = f.find(CC::BANK0_ACCESS_CYCLES);
        mem_cfg_.const_bank0 =
            itb0 == f.end() ? -1.0 : itb0->second;
        const auto itbu = f.find(CC::BANK0_UNIFORM_ACCESS_CYCLES);
        mem_cfg_.const_bank0_uniform =
            itbu == f.end() ? -1.0 : itbu->second;
      }
    }
    if (u.schema_id == 32) {
      const auto& f = try_resolve(i);
      mem_cfg_.dram_port_inflight = (std::uint64_t)
          MechanismRuntime::field(f, DP::MAX_INFLIGHT_BYTES, 0.0);
    }
    if (u.schema_id == 34) {
      const auto& f = try_resolve(i);
      mem_cfg_.load_port_line_granular =
          MechanismRuntime::field(f, LOADP::LINE_GRANULAR, 0.0) != 0;
      mem_cfg_.load_port_lane_order =
          MechanismRuntime::field(f, LOADP::LANE_ORDER, 0.0) != 0;
    }
    if (u.schema_id == 35) {
      const auto& f = try_resolve(i);
      mem_cfg_.dram_far_extra =
          MechanismRuntime::field(f, MDP::FAR_LATENCY_EXTRA_CYCLES,
                                  0.0);
      mem_cfg_.dram_far_side_gpc_mask = (std::uint64_t)
          MechanismRuntime::field(f, MDP::FAR_SIDE_GPC_MASK, 0.0);
      mem_cfg_.dram_home_page_bytes = (std::uint64_t)
          MechanismRuntime::field(f, MDP::HOME_PAGE_BYTES, 0.0);
    }
    // memory_level leaves (schema 3): the memory.<level>.{service,
    // capacity,residency} programs each carry a subset of the
    // level-tagged fields; every field present is applied (the row's
    // target_index is not a level id)
    if (u.schema_id == 3) {
      std::uint32_t mem_leaf = 0;
      const auto& f = rt_.resolve(i, engine_facts_, &mem_leaf);
      auto set = [&](std::uint32_t id, auto& dst) {
        if (auto it = f.find(id); it != f.end())
          dst = static_cast<std::decay_t<decltype(dst)>>(it->second);
      };
      set(ML::L1_LATENCY_CYCLES, mem_cfg_.l1_latency);
      set(ML::L1_LOCAL_FRAME_INTERLEAVED,
          mem_cfg_.local_frame_interleaved);
      set(ML::L1_LOCAL_HIT_LATENCY_CYCLES, mem_cfg_.local_hit_latency);
      set(ML::L1_OUTSTANDING_REQUESTS, mem_cfg_.l1_outstanding);
      set(ML::L1_ISSUE_INTERVAL_CYCLES, mem_cfg_.l1_issue_interval);
      set(ML::L1_SECTOR_BYTES_PER_CYCLE, mem_cfg_.l1_bytes_per_cycle);
      set(ML::L1_DATA_ARRAY_BYTES, mem_cfg_.l1_data_array_bytes);
      set(ML::L1_RESERVED_BYTES, mem_cfg_.l1_reserved_bytes);
      set(ML::L2_LATENCY_CYCLES, mem_cfg_.l2_latency);
      set(ML::L2_OUTSTANDING_REQUESTS, mem_cfg_.l2_outstanding);
      set(ML::L2_ISSUE_INTERVAL_CYCLES, mem_cfg_.l2_issue_interval);
      set(ML::L2_SECTOR_BYTES_PER_CYCLE, mem_cfg_.l2_bytes_per_cycle);
      set(ML::L2_RESIDENT_LINES, mem_cfg_.l2_resident_lines);
      set(ML::L2_RESIDENT_LINES_LOWER_BOUND,
          mem_cfg_.l2_resident_lines_lower_bound);
      set(ML::DRAM_LATENCY_CYCLES, mem_cfg_.dram_latency);
      set(ML::DRAM_OUTSTANDING_REQUESTS, mem_cfg_.dram_outstanding);
      set(ML::DRAM_ISSUE_INTERVAL_CYCLES, mem_cfg_.dram_issue_interval);
      set(ML::DRAM_SECTOR_BYTES_PER_CYCLE, mem_cfg_.dram_bytes_per_cycle);
      set(ML::STORE_L2_LATENCY_CYCLES, mem_cfg_.store_l2_latency);
      set(ML::STORE_L2_OUTSTANDING_REQUESTS, mem_cfg_.store_l2_outstanding);
      set(ML::STORE_L2_ISSUE_INTERVAL_CYCLES,
          mem_cfg_.store_l2_issue_interval);
      set(ML::STORE_L2_PARTIAL_LINE_CYCLES,
          mem_cfg_.store_l2_partial_line_cycles);
      set(ML::STORE_L2_PARTIAL_GRANULE_BYTES,
          mem_cfg_.store_l2_partial_granule_bytes);
      set(ML::STORE_L2_SECTOR_BYTES_PER_CYCLE,
          mem_cfg_.store_l2_bytes_per_cycle);
      set(ML::STORE_L2_SHARES_FILL_PORT,
          mem_cfg_.store_shares_fill_port);
      set(ML::STORE_L2_SHARED_STORE_SHARES_FILL_PORT,
          mem_cfg_.shared_store_shares_fill_port);
      set(ML::STORE_L2_ORDERS_LOADS,
          mem_cfg_.store_orders_loads);
      set(ML::STORE_DRAM_LATENCY_CYCLES, mem_cfg_.store_dram_latency);
      set(ML::STORE_DRAM_OUTSTANDING_REQUESTS,
          mem_cfg_.store_dram_outstanding);
      set(ML::STORE_DRAM_SECTOR_BYTES_PER_CYCLE,
          mem_cfg_.store_dram_bytes_per_cycle);
      if (auto it = f.find(ML::L2_TMA_SECTOR_BYTES_PER_CYCLE);
          it != f.end() && it->second > 0.0)
        mem_cfg_.tma_sector_bpc = it->second;
      // loaded-latency tables: piecewise-linear points over
      // shared_utilization (evaluated per request in MemoryHierarchy
      // against the shared-service bucket ledger); absent -> empty ->
      // extra = 0
      // memory.*.service leaves each carry a subset of the level
      // fields: only a leaf that actually declares the field may set
      // its table (otherwise a later leaf clears it again)
      if (auto dp = rt_.linear_points(
              i, engine_facts_, ML::DRAM_LOADED_LATENCY_EXTRA_CYCLES);
          !dp.empty())
        mem_cfg_.dram_loaded_extra = std::move(dp);
      if (auto lp = rt_.linear_points(
              i, engine_facts_, ML::L2_LOADED_LATENCY_EXTRA_CYCLES);
          !lp.empty())
        mem_cfg_.l2_loaded_extra = std::move(lp);
      static_assert(ML::L1_SHARED_CONFIGURATION_9_BYTES ==
                    ML::L1_SHARED_CONFIGURATION_0_BYTES + 9);
      for (std::uint32_t i = 0; i < 10; ++i)
        if (auto it = f.find(ML::L1_SHARED_CONFIGURATION_0_BYTES + i);
            it != f.end())
          mem_cfg_.shared_configurations.push_back(
              static_cast<std::uint64_t>(it->second));
    }
  }
  mem_cfg_.shared_static_bytes = job_.launch.shared_static_bytes;
  mem_cfg_.shared_dynamic_bytes = job_.launch.shared_dynamic_bytes;
  const std::uint64_t threads =
      std::uint64_t(job_.launch.block_x) * job_.launch.block_y *
      job_.launch.block_z;
  mem_cfg_.launch_warps_per_cta = (threads + 31) / 32;
  if (!mem_cfg_.shared_configurations.empty() &&
      mem_cfg_.l1_data_array_bytes && threads == 0)
    throw EngineError("missing_fact:launch_block_dims (LNCH)");
  // TOPO L2 partition map (fallback sm % pool_count handled inside)
  if (!job_.topology.empty()) {
    std::uint32_t nsm = 0;
    for (const auto& t : job_.topology) nsm = std::max(nsm, t.sm + 1);
    mem_cfg_.sm_partition.assign(nsm, 0xFFFFFFFFu);
    for (const auto& t : job_.topology) {
      // NONE_SENTINEL partitions fall back to sm % pool_count
      if (t.l2_partition < 0xFFFFFFFFu &&
          t.l2_partition < mem_cfg_.pool_count)
        mem_cfg_.sm_partition[t.sm] = t.l2_partition;
      else
        mem_cfg_.sm_partition[t.sm] = t.sm % mem_cfg_.pool_count;
    }
  }
  // TOPO gpc -> per-GPC shared services (an SM row without a gpc
  // skips the layer)
  if (mem_cfg_.l2_read_gpc_bpc > 0 || mem_cfg_.l2_write_gpc_bpc > 0) {
    if (job_.topology.empty())
      throw EngineError("missing_fact:sm_topology ()");
    mem_cfg_.sm_gpc.assign(mem_cfg_.sm_partition.size(),
                           0xFFFFFFFFu);
    for (const auto& t : job_.topology)
      mem_cfg_.sm_gpc[t.sm] = t.gpc;
  }
  // TOPO gpc -> l2_merge domain map (gpc domains 0..127; an SM row
  // without a gpc falls back to pair domain 128 + sm / domain_sms)
  if (mem_cfg_.merge_domain_gpc) {
    if (job_.topology.empty())
      throw EngineError("missing_fact:sm_topology ()");
    mem_cfg_.sm_merge_domain.assign(mem_cfg_.sm_partition.size(),
                                    0xFFFFFFFFu);
    for (const auto& t : job_.topology) {
      if (t.gpc != 0xFFFFFFFFu) {
        if (t.gpc >= 128)
          throw EngineError("unsupported:l2_merge_domain ()");
        mem_cfg_.sm_merge_domain[t.sm] = t.gpc;
      } else {
        if (mem_cfg_.merge_domain_sms == 0)
          throw EngineError("missing_fact:sm_topology.gpc (sm " +
                            std::to_string(t.sm) + ")");
        const std::uint64_t fb = 128 + t.sm / mem_cfg_.merge_domain_sms;
        if (fb >= 256)
          throw EngineError("unsupported:l2_merge_domain ()");
        mem_cfg_.sm_merge_domain[t.sm] = (std::uint32_t)fb;
      }
    }
  }
  mem_ = MemoryHierarchy(mem_cfg_);
  timing_ = job_.timing;
  if (sched_use_ != kNone64) {
    const auto& f = rt_.resolve(sched_use_, engine_facts_, nullptr);
    policy_.issue_slots = static_cast<std::uint32_t>(
        MechanismRuntime::field(f, SC::ISSUE_SLOTS, 1.0));
    policy_.cursor_commit =
        MechanismRuntime::field(f, SC::CURSOR_COMMIT, 1.0) != 0.0;
    policy_.try_next_on_block =
        MechanismRuntime::field(f, SC::TRY_NEXT_ON_BLOCK, 1.0) != 0.0;
    policy_.unit_grant_lwf =
        MechanismRuntime::field(f, SC::UNIT_GRANT_POLICY, 0.0) != 0.0;
  }
  if (operand_use_ != kNone64) {
    namespace OPR = core::parameter_fields::OPERAND_READ;
    const auto& f = rt_.resolve(operand_use_, engine_facts_, nullptr);
    operand_cfg_.enabled = static_cast<std::uint32_t>(
        MechanismRuntime::field(f, OPR::ENABLED, 0.0));
    operand_cfg_.reads_per_cycle = static_cast<std::uint32_t>(
        MechanismRuntime::field(f, OPR::READS_PER_CYCLE, 2.0));
    operand_cfg_.collector = static_cast<std::uint32_t>(
        MechanismRuntime::field(f, OPR::COLLECTOR, 0.0));
    if (operand_cfg_.enabled && operand_cfg_.collector != 2) {
      char buf[96];
      std::snprintf(buf, sizeof buf,
                    "unsupported:engine:operand_read_collector (%u)",
                    operand_cfg_.collector);
      throw std::runtime_error(buf);
    }
  }

  // launch_warps_per_cta engine fact: schema-25 warm model axis
  if (auto id = rt_.axis_id("launch_warps_per_cta")) {
    const auto& L = job_.launch;
    const std::uint64_t threads =
        static_cast<std::uint64_t>(L.block_x) * L.block_y * L.block_z;
    engine_facts_.emplace_back(*id, (threads + 31) / 32);
  }

  // dispatch params (schema 24): ns fields arrive as cycles already
  if (dispatch_use_ != kNone64) {
    const auto& f = rt_.resolve(dispatch_use_, engine_facts_, nullptr);
    dispatch_.first_wave_interval =
        static_cast<std::uint64_t>(MechanismRuntime::field(f, 0, 0.0));
    dispatch_.slot_free_to_entry = MechanismRuntime::field(f, 1, 0.0);
    // enabled absent -> dispatch on (Blackwell leaf predates it)
    dispatch_.enabled = MechanismRuntime::field(f, 2, 1.0) != 0.0;
    dispatch_.serial_per_cta = MechanismRuntime::field(f, 3, 0.0);
    // slow gpcs / dependents-launch serializer charge (absent ->
    // single class / no charge)
    dispatch_.slot_free_to_entry_slow =
        MechanismRuntime::field(f, 4, dispatch_.slot_free_to_entry);
    dispatch_.slow_gpc_mask =
        static_cast<std::uint64_t>(MechanismRuntime::field(f, 5, 0.0));
    dispatch_.dependents_launch_serial =
        MechanismRuntime::field(f, 6, 0.0);
  }

  // instruction fetch (schema 26): enabled 0 / missing / no CODE -> off
  {
    FetchConfig fc;
    if (fetch_use_ != kNone64) {
      const auto& f = rt_.resolve(fetch_use_, engine_facts_, nullptr);
      // `enabled` absent -> model on (Blackwell leaf predates the field)
      fc.enabled = MechanismRuntime::field(f, 0, 1.0) != 0.0;
      fc.line_bytes =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 1, 0.0));
      fc.prefetch_depth =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 3, 0.0));
      fc.capacity_l0 =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 6, 0.0));
      fc.capacity_l1 =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 7, 0.0));
      fc.stream_l1 =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 8, 0.0));
      fc.stream_l2 =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 9, 0.0));
      fc.stream_cold =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 10, 0.0));
      fc.launch_prefetch =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 11, 0.0));
      fc.sharing =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 12, 0.0));
      fc.parallelism =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 13, 0.0));
      fc.l0_ways =
          static_cast<std::uint64_t>(MechanismRuntime::field(f, 14, 0.0));
    }
    if (fc.enabled && fc.line_bytes == 0)
      throw EngineError("unsupported:engine:fetch_line_bytes ()");
    if (job_.code.empty()) fc.enabled = false;
    fetch_.configure(fc);
    fetch_enabled_ = fc.enabled;
    if (!fetch_enabled_) tele2_[kTele2FetchUnavailable] = 1;
  }
  {
    bool dense = job_.code.size() == job_.instances.size();
    for (std::size_t i = 0; dense && i < job_.code.size(); ++i)
      dense = job_.code[i].instance == i;
    code_addr_is_dense_ = dense;
    if (dense) {
      code_addr_dense_.resize(job_.code.size());
      for (std::size_t i = 0; i < job_.code.size(); ++i)
        code_addr_dense_[i] = job_.code[i].code_address;
    } else {
      for (const auto& c : job_.code)
        code_addr_[c.instance] = c.code_address;
    }
  }

  for (const auto& r : job_.register_refs)
    max_reg_ = std::max(max_reg_, reg_id(r));

  // ---- 3-3 sync state ----
  const std::uint32_t nseg =
      static_cast<std::uint32_t>(job_.segment_instance_base.size());
  for (const auto& mb : job_.mbarrier_preinit)
    mbar_.init(job_.cta_segment[mb.cta_index], mb.object,
               mb.arrival_count, false, mb.cta_index, 0);
  {
    std::vector<std::uint32_t> wpc(job_.ctas.size(), 0);
    for (const auto& c : job_.ctas) wpc[c.cta_index] = c.warp_count;
    cta_bar_ = CtaBarrierTable(wpc);
    // cluster barrier table: one cluster per segment
    std::vector<std::uint32_t> wps(nseg, 0);
    segment_cta_count_.assign(nseg, 0);
    for (const auto& c : job_.ctas) {
      const std::uint32_t s = job_.cta_segment[c.cta_index];
      wps[s] += c.warp_count;
      ++segment_cta_count_[s];
    }
    cluster_bar_ = ClusterBarrierTable(std::move(wps));
  }
  // MBOB: per-segment object -> (cta, address) and reverse map;
  // cluster_rank -> cta for destination resolution
  mbob_rows_.assign(nseg, {});
  mbob_rev_.assign(nseg, {});
  rank_cta_.assign(nseg, {});
  for (const auto& c : job_.ctas)
    rank_cta_[job_.cta_segment[c.cta_index]][c.cluster_rank] =
        c.cta_index;
  for (std::uint32_t s = 0; s < nseg; ++s) {
    const std::uint64_t begin = job_.segment_mbob_base[s];
    const std::uint64_t end =
        s + 1 < nseg ? job_.segment_mbob_base[s + 1]
                     : job_.mbarrier_owners.size();
    mbob_rows_[s].assign(job_.mbarrier_owners.begin() + begin,
                         job_.mbarrier_owners.begin() + end);
    for (std::uint64_t i = 0; i < mbob_rows_[s].size(); ++i)
      mbob_rev_[s][{mbob_rows_[s][i].cta_index,
                    mbob_rows_[s][i].address}] =
          static_cast<std::uint32_t>(i);
  }
  // SPLT credits keyed by (semantics index, rank)
  for (const auto& sp : job_.splits)
    split_credit_[sp.semantics][sp.rank] = sp.credit;
  // GSYN: pair observations into acquire plans at load (§13.10)
  gsync_plan_ = pair_global_sync(job_);
  tele_[kTeleGsyncDroppedPolls] = gsync_plan_.stats.dropped_polls;
  tele_[kTeleGsyncChainsUnique] = gsync_plan_.stats.chains_unique;
  tele_[kTeleGsyncChainsNonUnique] =
      gsync_plan_.stats.chains_non_unique;
  tele_[kTeleGsyncUpdateUnknownPost] =
      gsync_plan_.stats.update_unknown_post;
  tele_[kTeleGsyncAcquiresFrom] = gsync_plan_.stats.acq_from;
  tele_[kTeleGsyncAcquiresAny] = gsync_plan_.stats.acq_any;
  tele_[kTeleGsyncAcquiresCount] = gsync_plan_.stats.acq_count;
  tele_[kTeleGsyncAcquiresInitial] = gsync_plan_.stats.acq_initial;
  tele_[kTeleGsyncAcquiresNoWriter] =
      gsync_plan_.stats.acq_no_writer;
  gsync_failed_poll_.assign(job_.instances.size(), false);
  for (const auto inst : gsync_plan_.failed_polls)
    if (inst < gsync_failed_poll_.size()) gsync_failed_poll_[inst] = true;
  // rule 3b (§13.10): prescan each poll region [first, winner); it may
  // fold only when every member is a no-semantics instance or one of
  // the sequence's own failed polls — any other SEMA/GSYN row (foreign
  // polls included) blocks folding and the region issues per rule 3
  for (const auto& [winner, first] : gsync_plan_.region_first) {
    auto members = gsync_plan_.region_members.find(winner);
    bool fold = first <= winner &&
                members != gsync_plan_.region_members.end();
    for (std::uint64_t i = first; fold && i < winner; ++i) {
      if (i < gsync_failed_poll_.size() && gsync_failed_poll_[i]) {
        if (std::find(members->second.begin(), members->second.end(), i) !=
            members->second.end())
          continue;
        fold = false;  // foreign failed poll: another GSYN row
        break;
      }
      // any other data-memory access in the region means it is not a
      // pure spin: same-(template, address) polls of separate wait
      // episodes merge into one sequence, and the work between the
      // episodes (loads / stores / TMA) must not be deferred to the
      // last episode's producer (stream-K fixup regressed 6x
      // otherwise).  A constant-space access (LDC / ULDC of a loop
      // bound or a table entry inside the spin body) carries no such
      // work and does not block folding.
      const auto ma = job_.instances[i].memory_access;
      const bool data_access =
          ma != kNone32 && (job_.accesses[ma].flags & 0x7) != 3;
      if (job_.instances[i].semantics != kNone32 || data_access ||
          gsync_plan_.acquires.count(i) ||
          gsync_plan_.producer_address.count(i))
        fold = false;
    }
    if (fold) gsync_region_[first] = winner;
  }
  // InstanceTiming control envelopes (USES target 2, schema 5)
  for (std::uint64_t i = 0; i < job_.uses.size(); ++i)
    if (job_.uses[i].target == 2 && job_.uses[i].schema_id == 5)
      inst_envelope_[job_.uses[i].target_index] = i;
  prog_mbarrier_release_ =
      rt_.program_by_name("synchronization.mbarrier.release");
  prog_trywait_ =
      rt_.program_by_name("synchronization.mbarrier.trywait_timeout");
  prog_cluster_release_ =
      rt_.program_by_name("synchronization.cluster.release");
  prog_desc_visibility_ =
      rt_.program_by_name("synchronization.descriptor.visibility");
  // failed-poll lookahead: map each mbarrier_wait instance to the next
  // wait's (object, parity) within the same warp (§13.3.6).
  for (const auto& wi : job_.warps) {
    std::uint64_t prev = kNone64;
    for (std::uint64_t i = wi.first_instance;
         i < wi.first_instance + wi.instance_count; ++i) {
      const auto& inst = job_.instances[i];
      if (inst.semantics == kNone32) continue;
      const auto& sm = job_.semantics[inst.semantics];
      if (sm.kind != kSemaMbarrierWait) continue;
      if (prev != kNone64)
        next_wait_[prev] = {sm.field[0], sm.field[1] & 1u};
      prev = i;
    }
    // abandoned-poll set: reverse scan; a wait whose (object, parity)
    // has no later same-domain check is abandoned
    std::unordered_set<std::uint64_t> seen;
    for (std::uint64_t i = wi.first_instance + wi.instance_count;
         i-- > wi.first_instance;) {
      const auto& inst = job_.instances[i];
      if (inst.semantics == kNone32) continue;
      const auto& sm = job_.semantics[inst.semantics];
      if (sm.kind != kSemaMbarrierWait) continue;
      const std::uint64_t key =
          (static_cast<std::uint64_t>(sm.field[0]) << 32) |
          (sm.field[1] & 1u);
      if (!seen.count(key)) abandoned_poll_.insert(i);
      seen.insert(key);
    }
    // NANOSLEEP.SYNCS poll regions (§13.9): a region runs from a
    // non-failed mbarrier wait (or warp start) to the next non-failed
    // wait (or warp end); the closing wait belongs to the region.
    // Depends on next_wait_ via failed_poll, so it runs after the
    // scans above.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> region;
    std::vector<std::uint64_t> sleeps;
    auto close_region = [&]() {
      if (sleeps.empty()) {
        region.clear();
        return;
      }
      const std::uint32_t begin =
          static_cast<std::uint32_t>(nanosleep_objects_.size());
      for (const auto& e : region) nanosleep_objects_.push_back(e);
      const std::uint32_t end =
          static_cast<std::uint32_t>(nanosleep_objects_.size());
      for (std::uint64_t s : sleeps)
        nanosleep_region_[s] = {begin, end};
      region.clear();
      sleeps.clear();
    };
    for (std::uint64_t i = wi.first_instance;
         i < wi.first_instance + wi.instance_count; ++i) {
      const auto& inst = job_.instances[i];
      if (inst.semantics == kNone32) continue;
      const auto& sm = job_.semantics[inst.semantics];
      if (sm.kind == kSemaMbarrierWait) {
        const std::pair<std::uint32_t, std::uint32_t> key =
            {sm.field[0], sm.field[1] & 1u};
        bool dup = false;
        for (const auto& e : region)
          if (e == key) { dup = true; break; }
        if (!dup) region.push_back(key);
        if (!failed_poll(i, sm)) close_region();
        continue;
      }
      if (sm.kind == kSemaNanosleep && (sm.flags & kSemaFlagSyncs))
        sleeps.push_back(i);
    }
    close_region();
  }
}

const Fields& Simulator::service_fields(const InstructionTemplate& t) {
  auto idx = std::uint32_t(&t - job_.templates.data());
  if (idx < svc_cached_.size() && svc_cached_[idx] && svc_cache_[idx])
    return *svc_cache_[idx];
  if (idx >= tmpl_use_.size() || tmpl_use_[idx] == kNone64)
    throw EngineError(
        "missing_fact:instruction.service (" +
        std::string(job_.strings.c_str() + t.operation_class) + ")");
  bool dep = false;
  const auto& f = rt_.resolve(tmpl_use_[idx], engine_facts_, nullptr,
                              &dep);
  if (idx >= svc_cached_.size()) {
    svc_cached_.resize(idx + 1, 0);
    svc_cache_.resize(idx + 1, nullptr);
  }
  if (!svc_cached_[idx]) {
    svc_cache_[idx] = dep ? nullptr : &f;
    svc_cached_[idx] = 1;
  }
  return f;
}

void Simulator::update_ready_at(WarpState& w) {
  // ready_at_ = max(source reg_ready, masked barrier_free) for the
  // current next_instance — both only change at issue time, so the
  // value is exact until this warp's next issue
  w.ready_at_ = 0;
  if (w.next_instance >= w.end_instance) return;
  const auto& inst = job_.instances[w.next_instance];
  const auto& t = job_.templates[inst.template_index];
  for (std::uint32_t i = 0; i < t.source_count; ++i) {
    const auto reg = reg_id(job_.register_refs[t.source_begin + i]);
    const std::uint64_t v =
        reg < w.reg_ready.size() ? w.reg_ready[reg] : 0;
    w.ready_at_ = std::max(w.ready_at_, v);
  }
  for (std::uint32_t b = 0; b < NO_BARRIER; ++b)
    if (t.scoreboard_wait_mask & (1u << b))
      w.ready_at_ = std::max(w.ready_at_, w.barrier_free[b]);
  w.ready_at_ = std::max(w.ready_at_, w.control_floor);
}

// ---- readiness (fixed order; buckets per spec) ----------------------

// synchronous shared store: MACC shared space (flags&7 == 1) + store
// (0x40), excluding async stores (STAS/TMA/bulk); bytes = sum of
// sector byte_mask popcounts
bool Simulator::sync_shared_store(
    const InstructionInstance& inst, const InstanceSemantics& sema,
    std::uint64_t* bytes) const {
  switch (sema.kind) {
    case kSemaSharedStoreAsync:
    case kSemaTmaLoad:
    case kSemaBulkCopy:
    case kSemaBulkReduce:
    case kSemaBulkGroupIssue:
      return false;
    default:
      break;
  }
  if (inst.memory_access == kNone32) return false;
  const auto& ma = job_.accesses[inst.memory_access];
  // access_kind is a 3-bit field: exactly 0x40 = synchronous store
  if ((ma.flags & 0x7) != 1 || (ma.flags & 0x1C0) != 0x40)
    return false;
  std::uint64_t b = 0;
  for (std::uint32_t i = 0; i < ma.sector_count; ++i) {
    const std::uint32_t m =
        job_.sectors[ma.sector_begin + i].byte_mask;
    b += m ? static_cast<std::uint64_t>(__builtin_popcount(m)) : 32;
  }
  *bytes = b;
  return true;
}

bool Simulator::register_ready(const WarpState& w,
                               const InstructionTemplate& t) const {
  for (std::uint32_t i = 0; i < t.source_count; ++i) {
    std::uint32_t reg = reg_id(job_.register_refs[t.source_begin + i]);
    if (reg < w.reg_ready.size() && w.reg_ready[reg] > now_)
      return false;
  }
  return true;
}

bool Simulator::scoreboard_ready(const WarpState& w,
                                 const InstructionTemplate& t) const {
  for (std::uint32_t b = 0; b < NO_BARRIER; ++b) {
    if (t.scoreboard_wait_mask & (1u << b)) {
      if (w.barrier_free[b] > now_)
        return false;
    }
  }
  return true;
}

std::uint64_t Simulator::max_source_reg_ready(
    const WarpState& w, const InstructionTemplate& t) const {
  std::uint64_t m = 0;
  for (std::uint32_t i = 0; i < t.source_count; ++i) {
    const std::uint32_t reg =
        reg_id(job_.register_refs[t.source_begin + i]);
    if (reg < w.reg_ready.size()) m = std::max(m, w.reg_ready[reg]);
  }
  return m;
}

std::uint64_t Simulator::register_retry(
    const WarpState& w, const InstructionTemplate& t) const {
  std::uint64_t r = kNever;
  for (std::uint32_t i = 0; i < t.source_count; ++i) {
    std::uint32_t reg = reg_id(job_.register_refs[t.source_begin + i]);
    if (reg < w.reg_ready.size() && w.reg_ready[reg] > now_)
      r = std::min(r, w.reg_ready[reg]);
  }
  for (std::uint32_t b = 0; b < NO_BARRIER; ++b)
    if ((t.scoreboard_wait_mask & (1u << b)) &&
        w.barrier_free[b] > now_)
      r = std::min(r, w.barrier_free[b]);
  if (w.control_floor > now_) r = std::min(r, w.control_floor);
  return r;
}

std::uint64_t Simulator::capacity_ready(
    const std::vector<std::uint64_t>& completions,
    std::uint64_t max_inflight, std::uint64_t candidate) {
  // predecessor capacity_ready_cycle: find earliest candidate where
  // fewer than max_inflight completions are still pending.
  for (;;) {
    std::size_t active = 0;
    std::uint64_t earliest = ~0ull;
    for (auto c : completions)
      if (c > candidate) { ++active; earliest = std::min(earliest, c); }
    if (active < max_inflight) return candidate;
    if (earliest == ~0ull) throw EngineError(
        "unsupported:engine:resource_state ()");
    candidate = earliest;
  }
}

std::uint32_t Simulator::service_resource(
    const InstructionTemplate& t, const Fields& svc,
    std::uint32_t which) const {
  std::uint32_t res = which ? t.issue_group_resource : t.resource;
  if (res != kNone32) return res;
  // segment builders keep TMPL.resource at NONE; the binding lives on
  // the service leaf (fields 10/11, 0xFFFFFFFF or absent = none)
  const double v = MechanismRuntime::field(
      svc, which ? IT::ISSUE_GROUP_RESOURCE : IT::RESOURCE, -1.0);
  if (v < 0 || v >= kNone32) return kNone32;
  return static_cast<std::uint32_t>(v);
}

std::uint64_t Simulator::resource_cap(const ResDef& d,
                                      const Fields& svc) {
  // capacity: an explicit resource cap (0 = none) further limited by
  // the service leaf's max_inflight (0 = none); recomputed per issue
  // so ready and commit agree regardless of which class saw the unit
  // first
  std::uint64_t cap = static_cast<std::uint64_t>(
      std::max(0.0, d.max_inflight));
  const double svc_max =
      MechanismRuntime::field(svc, IT::MAX_INFLIGHT, 0.0);
  if (svc_max > 0.0)
    cap = cap ? std::min(cap, static_cast<std::uint64_t>(svc_max))
              : static_cast<std::uint64_t>(svc_max);
  return cap;
}

ResourceState& Simulator::resource_state(std::uint32_t res,
                                         std::uint64_t copy,
                                         std::uint64_t owner,
                                         const ResDef& d) {
  if (res >= res_ptr_.size()) res_ptr_.resize(res + 1);
  auto& v = res_ptr_[res];
  std::uint64_t span;
  switch (d.scope) {
    case SCOPE_SMSP: span = smsps_.size() * 16; break;
    case SCOPE_SM:   span = smsps_.size(); break;
    case SCOPE_CLUSTER: span = job_.ctas.size(); break;
    default:         span = 1; break;
  }
  const std::uint64_t idx = copy * span + owner;
  if (idx >= v.size()) v.resize(idx + 1, nullptr);
  if (!v[idx])
    v[idx] = &resources_[(std::uint64_t(res) << 40) |
                         (copy << 32) | owner];
  return *v[idx];
}

bool Simulator::resource_ready(
    WarpState& w, const InstructionTemplate& t,
    const Fields& svc) {
  for (std::uint32_t which = 0; which < 2; ++which) {
    const std::uint32_t res = service_resource(t, svc, which);
    if (res == kNone32) continue;
    const auto dit = res_def_.find(res);
    if (dit == res_def_.end()) continue;
    const auto& d = dit->second;
    const std::uint64_t cap = resource_cap(d, svc);
    std::uint64_t owner;
    switch (d.scope) {
      case SCOPE_SMSP: owner = (std::uint64_t(w.sm) << 4) | w.smsp; break;
      case SCOPE_SM:   owner = w.sm; break;
      case SCOPE_CLUSTER: owner = w.cta; break;
      default:         owner = 0; break;
    }
    bool any_ready = false;
    std::uint64_t earliest = kNever;
    for (std::uint64_t copy = 0; copy < d.copies; ++copy) {
      auto& r = resource_state(res, copy, owner, d);
      if (!r.seen) {
        r.resource = res; r.copy = copy;
        r.seen = true;
      }
      std::uint64_t ready = std::max(now_, r.next_free);
      if (cap) ready = capacity_ready(r.completions, cap, ready);
      if (ready <= now_) { any_ready = true; break; }
      r.wait_cycles += ready - now_;
      earliest = std::min(earliest, ready);
      candidate_ready = std::max(candidate_ready, ready);
    }
    const std::uint64_t ukey = (std::uint64_t(res) << 32) | owner;
    if (!any_ready) {
      if (policy_.unit_grant_lwf) {
        // park on the unit; the wait starts at the first blocked
        // evaluation and survives while the verdict stays "unit busy"
        if (w.unit_wait_key != ukey) {
          clear_unit_waiter(w);
          unit_waiters_[ukey].waiters.emplace(w.warp_index, now_);
          w.unit_wait_key = ukey;
        }
        w.unit_blocked = true;
      }
      w.last_wait = WaitBucket::kResource;
      w.retry_at = earliest;
      return false;
    }
    if (policy_.unit_grant_lwf) {
      // unit free: it goes to the longest live waiter; a warp parked
      // since the same cycle as the unit's most recent grantee wins
      // the tie, so the last issuer cannot re-take it while peers wait
      if (auto git = unit_waiters_.find(ukey);
          git != unit_waiters_.end()) {
        auto& g = git->second;
        const auto mine = g.waiters.find(w.warp_index);
        const std::uint64_t my_since =
            mine == g.waiters.end() ? now_ : mine->second;
        bool outbid = false;
        for (const auto& [wi, since] : g.waiters) {
          if (wi == w.warp_index) continue;
          const auto& ow = warps_[wi];
          if (ow.done || ow.next_instance >= ow.end_instance) continue;
          if (since < my_since ||
              (since == my_since && g.last_grant == w.warp_index)) {
            outbid = true;
            break;
          }
        }
        if (outbid) {
          if (w.unit_wait_key != ukey) {
            clear_unit_waiter(w);
            g.waiters.emplace(w.warp_index, now_);
            w.unit_wait_key = ukey;
          }
          w.unit_blocked = true;
          w.last_wait = WaitBucket::kResource;
          w.retry_at = now_ + 1;
          return false;
        }
      }
    }
  }
  return true;
}

void Simulator::clear_unit_waiter(WarpState& w) {
  if (w.unit_wait_key == kNone64) return;
  const auto it = unit_waiters_.find(w.unit_wait_key);
  if (it != unit_waiters_.end())
    it->second.waiters.erase(w.warp_index);
  w.unit_wait_key = kNone64;
}

bool Simulator::try_issue(WarpState& w) {
  w.unit_blocked = false;
  if (w.next_instance >= w.end_instance) return false;
  // rule 3b (§13.10): the head of a foldable poll region waits on the
  // paired winner's acquire readiness; once ready every instance in
  // [first, W) retires in this cycle (no latency, resources, traffic,
  // fetch) and the winner issues right after through the normal path
  if (auto rg = gsync_region_.find(w.next_instance);
      rg != gsync_region_.end()) {
    const std::uint64_t first = w.next_instance;
    const std::uint64_t winner = rg->second;
    const auto cyc = gsync_ready_cycle_of(winner);
    if (!cyc || *cyc > now_) {
      if (w.region_wait_mark != ~0ull)
        tele_[kTeleGsyncWaitCycles] += now_ - w.region_wait_mark;
      w.region_wait_mark = now_;
      if (cyc && *cyc > now_) w.retry_at = *cyc;
      gsync_wait_smsps_.insert(
          (std::uint64_t(w.sm) << 3) | w.smsp);
      w.last_wait = WaitBucket::kSemantic;
      return false;
    }
    if (w.region_wait_mark != ~0ull) {
      tele_[kTeleGsyncWaitCycles] += now_ - w.region_wait_mark;
      w.region_wait_mark = ~0ull;
    }
    const std::uint64_t span =
        w.stall_since ? now_ - w.stall_since : 0;
    std::uint64_t object = kNone64;
    if (auto a = gsync_plan_.acquires.find(winner);
        a != gsync_plan_.acquires.end())
      object = a->second.address;
    for (std::uint64_t i = first; i < winner; ++i) {
      ++out_.issued;
      ++out_.completed;
      if (opt_.journal) {
        const auto& ri = job_.instances[i];
        out_.journal.push_back(JournalRow{
            w.warp_index, w.smsp, static_cast<std::uint32_t>(i), now_,
            job_.templates[ri.template_index].operation_class,
            ri.template_index,
            span ? static_cast<std::uint32_t>(WaitBucket::kSemantic)
                 : kNone32,
            span, object});
      }
    }
    ++tele_[kTeleGsyncRegionsRetired];
    tele_[kTeleGsyncRegionInstances] += winner - first;
    w.next_instance = winner;
    w.stall_since = 0;
    if (w.next_instance >= w.end_instance) return true;
  }
  const auto& inst = job_.instances[w.next_instance];
  const auto& t = job_.templates[inst.template_index];
  const auto& svc = service_fields(t);
  static const InstanceSemantics kNoSema{};
  const auto& sema = inst.semantics != kNone32
      ? job_.semantics[inst.semantics] : kNoSema;
  w.retry_at = 0;
  candidate_ready = now_;
  // CYCLESCOPE_DEBUG_INST=<instance> diagnostics (default off): kDebugInst
  // guards every writer-table update and DBG print so the hot path pays
  // nothing when the variable is unset
  static const bool kDebugInst =
      std::getenv("CYCLESCOPE_DEBUG_INST") != nullptr;
  static const std::uint64_t dbg_inst = kDebugInst
      ? std::strtoull(std::getenv("CYCLESCOPE_DEBUG_INST"), nullptr, 0)
      : kNone64;
  // readiness order (§13.2): semantic -> resource ->
  // register/scoreboard/control -> arch
  if (!semantic_ready(w, inst, sema)) {
    // suspensions set blocked_until + a bucket; pure semantic gates
    // (mbarrier waits, gsync) stay in the semantic bucket
    if (w.blocked_until > now_) {
      w.last_wait = w.block_bucket;
      w.retry_at = w.blocked_until;
    } else {
      w.last_wait = WaitBucket::kSemantic;
    }
    return false;
  }
  if (!resource_ready(w, t, svc)) return false;
  // storage.global.store_issue (kind 65): a global / local store cannot
  // issue before its SMSP's previous store's interval has elapsed
  // (lsu_accept protocol v5: STG.32 / .64 / .128 every 3 / 6 / 23
  // cycles per SMSP); shared-resolved ST / STL are vetoed like kind 5
  if (t.timing_count && inst.memory_access != kNone32 &&
      (job_.accesses[inst.memory_access].flags & 0x7) != 1) {
    for (std::uint64_t ti = t.timing_begin;
         ti < t.timing_begin + t.timing_count; ++ti) {
      if (timing_[ti].kind != kTimingGlobalStoreIssueInterval ||
          !timing_[ti].enabled)
        continue;
      auto it = store_issue_next_.find((std::uint64_t(w.sm) << 4) | w.smsp);
      if (it != store_issue_next_.end() && it->second > now_) {
        w.last_wait = WaitBucket::kResource;
        w.retry_at = it->second;
        return false;
      }
      break;
    }
  }
  // register_file.operand_read: while the most recently issued
  // instruction with n > 0 unserved GPR reads has not started its
  // collector service, the SMSP issues nothing further
  if (operand_cfg_.enabled) {
    auto& s = smsps_[w.sm][w.smsp];
    if (s.coll_svc_start > now_) {
      w.last_wait = WaitBucket::kResource;
      w.retry_at = s.coll_svc_start;
      return false;
    }
  }
  const std::uint64_t reg_r = register_retry(w, t);
  if (reg_r != kNever) {
    if (kDebugInst && dbg_inst == w.next_instance) {
      for (std::uint32_t i = 0; i < t.source_count; ++i) {
        const std::uint32_t reg =
            reg_id(job_.register_refs[t.source_begin + i]);
        auto iw = w.reg_writer.find(reg);
        fprintf(stderr,
                "DBG_INST %llu fail@ %llu reg_ready r%u=%llu wr=%u\n",
                (unsigned long long)w.next_instance,
                (unsigned long long)now_, reg,
                reg < w.reg_ready.size()
                    ? (unsigned long long)w.reg_ready[reg] : 0,
                iw != w.reg_writer.end() ? iw->second : 0);
      }
      for (std::uint32_t b = 0; b < NO_BARRIER; ++b)
        if ((t.scoreboard_wait_mask & (1u << b)) &&
            w.barrier_free[b] > now_)
          fprintf(stderr,
                  "DBG_INST %llu fail@ %llu barrier_free sb%u=%llu "
                  "wr=%u\n",
                  (unsigned long long)w.next_instance,
                  (unsigned long long)now_, b,
                  (unsigned long long)w.barrier_free[b],
                  w.barrier_writer[b]);
      fprintf(stderr,
              "DBG_INST %llu fail@ %llu control_floor=%llu "
              "blocked_until=%llu block_bucket=%u retry_at=%llu\n",
              (unsigned long long)w.next_instance,
              (unsigned long long)now_,
              (unsigned long long)w.control_floor,
              (unsigned long long)w.blocked_until,
              (unsigned)w.block_bucket,
              (unsigned long long)w.retry_at);
    }
    w.last_wait = WaitBucket::kRegister;
    w.retry_at = reg_r;
    return false;
  }
  if (auto it = sm_issue_block_until_.find(w.sm);
      it != sm_issue_block_until_.end() && it->second > now_) {
    w.last_wait = WaitBucket::kResource;
    w.retry_at = it->second;
    return false;
  }
  // memory-queue hold (arch decides; base 0 = not held): while a
  // warpgroup with a member on this SMSP exceeds its outstanding-op
  // threshold or has a commit-group-held wgmma launch, MUFU/SHFL/LDS-
  // class instances wait.  A finite return is the retry cycle; ~0
  // parks the warp like the collective-wait path — it is re-judged
  // whenever the SMSP is marked (a held launch that finally launches
  // marks the members' SMSPs)
  if (mq_class_.size() != job_.templates.size()) {
    mq_class_.assign(job_.templates.size(), false);
    for (std::uint64_t i = 0; i < job_.templates.size(); ++i) {
      const std::string_view cls =
          job_.strings.c_str() + job_.templates[i].operation_class;
      mq_class_[i] = cls.rfind("MUFU", 0) == 0 ||
                     cls.rfind("SHFL", 0) == 0 ||
                     cls.rfind("LDS", 0) == 0;
    }
  }
  if (inst.template_index < mq_class_.size() &&
      mq_class_[inst.template_index]) {
    const std::uint64_t mq_r =
        arch_->memory_queue_retry(w.sm, w.smsp, now_);
    if (mq_r) {
      w.last_wait = WaitBucket::kResource;
      w.retry_at = (mq_r == ~0ull ? 0 : mq_r);
      return false;
    }
  }
  if (!arch_->instance_ready(*this, w, inst, sema)) {
    w.last_wait = WaitBucket::kSemantic;
    return false;
  }
  // co-resident dependent penalty (hopper wgmma C4): while another
  // warpgroup's op is in flight on the SM, an issue triggered exactly
  // by a register dependency resolving this cycle pays +1 every other
  // time (coresident_dependent_half_cycles)
  if (arch_->coresident_dependent_penalty(*this, w) &&
      max_source_reg_ready(w, t) == now_ &&
      (w.dep_penalty_toggle ^= 1)) {
    w.last_wait = WaitBucket::kRegister;
    w.retry_at = now_ + 1;
    return false;
  }

  std::uint64_t result_lat = static_cast<std::uint64_t>(
      std::max(0.0, MechanismRuntime::field(
          svc, IT::RESULT_LATENCY_CYCLES, 1.0)));
  // failed polls (§13.3.6) carry no result latency — their predicate
  // is consumed by the retry whose timing is the suspension / wake
  // model; a stale one (phase already released on the simulated
  // timeline) additionally counts as stale.
  if (sema.kind == kSemaMbarrierWait &&
      failed_poll(w.next_instance, sema)) {
    result_lat = 0;
    if (mbar_.wait_ready(seg_of(w),
                         static_cast<std::uint32_t>(sema.field[0]),
                         static_cast<std::uint32_t>(sema.field[1] & 1u),
                         now_))
      ++tele_[kTeleStalePolls];
  }
  // failed strong-load polls of a global-sync poll sequence (§13.10
  // rule 3) likewise: their count is the instrumented run's spin, the
  // wait itself is the paired winner's acquire; no result latency, no
  // hierarchy traffic
  const bool gsync_failed_poll =
      (w.next_instance < gsync_failed_poll_.size()
           ? gsync_failed_poll_[w.next_instance]
           : gsync_plan_.failed_polls.count(w.next_instance) != 0);
  if (gsync_failed_poll) result_lat = 0;
  // fence-family ops have no destination register: their service result
  // is the warp's own next-issue floor (MEMBAR/FENCE block the warp in
  // order until complete; k_fence definition).  Other no-destination
  // classes keep the register/barrier-only path.
  if (t.destination_count == 0 && result_lat > 1) {
    const std::string_view cls = job_.strings.c_str() + t.operation_class;
    if (cls.rfind("MEMBAR", 0) == 0 || cls.rfind("FENCE", 0) == 0 ||
        cls == "YIELD")
      w.control_floor = std::max(w.control_floor, now_ + result_lat);
  }
  const std::uint64_t completion_lat = static_cast<std::uint64_t>(
      std::max<double>(result_lat,
          MechanismRuntime::field(svc, IT::COMPLETION_LATENCY_CYCLES,
                                  std::max<std::uint64_t>(result_lat, 1))));
  // distinct GPR source registers per template (lazy; RZ / uniform /
  // predicate operands carry no REGS row, so every ref with id < 256
  // is a general-purpose register; a pair flag counts id+1 too)
  if (gpr_source_count_.size() != job_.templates.size()) {
    gpr_source_count_.assign(job_.templates.size(), 0);
    for (std::uint64_t i = 0; i < job_.templates.size(); ++i) {
      const auto& tt = job_.templates[i];
      std::uint64_t bits[4] = {0, 0, 0, 0};
      for (std::uint32_t j = 0; j < tt.source_count; ++j) {
        const auto& ref = job_.register_refs[tt.source_begin + j];
        const std::uint32_t id = reg_id(ref);
        if (id >= 256) continue;
        bits[id >> 6] |= 1ull << (id & 63);
        if ((reg_flags(ref) & 2) && id + 1 < 256)
          bits[(id + 1) >> 6] |= 1ull << ((id + 1) & 63);
      }
      gpr_source_count_[i] =
          static_cast<std::uint32_t>(__builtin_popcountll(bits[0]) +
                                     __builtin_popcountll(bits[1]) +
                                     __builtin_popcountll(bits[2]) +
                                     __builtin_popcountll(bits[3]));
    }
  }
  double issue_interval =
      MechanismRuntime::field(svc, IT::ISSUE_INTERVAL_CYCLES, 1.0);
  {
    const double single_source = MechanismRuntime::field(
        svc, IT::SINGLE_SOURCE_ISSUE_INTERVAL_CYCLES, 0.0);
    if (single_source > 0.0 &&
        inst.template_index < gpr_source_count_.size() &&
        gpr_source_count_[inst.template_index] <= 1)
      issue_interval = single_source;
  }

  // journal wait columns: stall window [stall_since, now_) accrued
  // by flush_wait against w.last_wait; object = sync semantic's
  // field[0] for mbarrier / named-barrier ops, kNone64 otherwise
  const std::uint64_t inst_wait_cycles =
      w.stall_since ? now_ - w.stall_since : 0;
  const std::uint32_t inst_wait_bucket =
      inst_wait_cycles ? static_cast<std::uint32_t>(w.last_wait)
                       : kNone32;
  const std::uint64_t inst_wait_object =
      (sema.kind >= kSemaMbarrierInit &&
       sema.kind <= kSemaClusterBarrierWait)
          ? sema.field[0] : kNone64;
  // rule S (lsu_accept addendum): a global/generic STG issued while
  // this SM's store buffer (store_l2.outstanding_requests undrained
  // requests) is full stalls until the oldest request drains —
  // lg_throttle; STL / RED / ATOM / UTMASTG are not gated
  if (inst.memory_access == kNone32 && !gsync_failed_poll &&
      sema.kind == kSemaTmaLoad) {
    // fully out-of-bounds tiled TMA: no access was captured; the whole
    // box still traverses the load path as phantom sectors (tma_oob v2)
    bool tile = false;
    for (std::uint64_t ti = t.timing_begin;
         ti < t.timing_begin + t.timing_count; ++ti) {
      if (timing_[ti].kind != kTimingTmaCompletion) continue;
      tile = timing_fields(ti, nullptr)
                 .find(TR::CUT_EXTRA_CYCLES) != timing_fields(ti, nullptr).end();
      break;
    }
    if (tile) {
      const std::uint64_t sb =
          mem_cfg_.sector_bytes ? mem_cfg_.sector_bytes : 32;
      const std::uint64_t total = (sema.field[3] + sb - 1) / sb;
      if (total) {
        const std::uint64_t ret = mem_.load(
            w.sm, now_, nullptr, 0, false, nullptr, nullptr, nullptr,
            nullptr, nullptr, &mem_shared_free_, true, false, false,
            false, static_cast<std::size_t>(total));
        mem_return_ = ret;
      }
    }
  }
  if (inst.memory_access != kNone32 && !gsync_failed_poll) {
    const auto& sma = job_.accesses[inst.memory_access];
    const std::uint32_t sspace = sma.flags & 0x7;
    const std::uint32_t sakind = sma.flags & 0x1C0;
    if (sspace == 0 && sakind == 0x40) {
      const std::uint64_t block =
          mem_.store_issue_block(w.sm, now_);
      if (store_gate_q_.size() <= w.sm) store_gate_q_.resize(w.sm + 1);
      auto& q = store_gate_q_[w.sm];
      const bool full = block > now_;
      const bool head = !q.empty() && q.front() == w.warp_index;
      if (full || (!q.empty() && !head)) {
        const bool queued =
            std::find(q.begin(), q.end(), w.warp_index) != q.end();
        if (!queued) {
          q.push_back(w.warp_index);
          ++tele_[kTeleStoreBufferStalls];
        }
        tele_[kTeleStoreBufferStallCycles] += full ? block - now_ : 1;
        w.last_wait = WaitBucket::kResource;
        w.retry_at = full ? block : now_ + 1;
        return false;
      }
      if (head) q.pop_front();
    }
  }
  // commit issue
  ++out_.issued;
  clear_unit_waiter(w);
  if (cta_first_issue_.size() != job_.ctas.size()) {
    cta_first_issue_.assign(job_.ctas.size(), kNone64);
    cta_last_issue_.assign(job_.ctas.size(), 0);
  }
  if (cta_first_issue_[w.cta] == kNone64) cta_first_issue_[w.cta] = now_;
  cta_last_issue_[w.cta] = now_;
  // PREEXIT executed anywhere in the CTA's life charges the dispatch
  // serializer at completion (§13.11); the instance itself is an
  // ordinary issue
  if (sema.kind == kSemaDependentsLaunch)
    cta_dependents_launch_[w.cta] = 1;
  w.issue_stamp = now_;
  w.stall_since = 0;
  if (smsps_[w.sm][w.smsp].first_active == kNone64)
    smsps_[w.sm][w.smsp].first_active = now_;
  // the read barrier is released at issue; for rows queued through
  // the MIO pipes (kinds 5 / 6 / 15) it is released at pipe
  // acceptance instead — see below.  7/0xFF = none.  Destination
  // registers and the write barrier are released below at
  // data_ready (§data-ready).
  if (t.scoreboard_read != 0xFF && t.scoreboard_read < NO_BARRIER)
    w.barrier_free[t.scoreboard_read] = now_;
  // resource instance: initiation interval per copy + inflight record
  for (std::uint32_t which = 0; which < 2; ++which) {
    const std::uint32_t res = service_resource(t, svc, which);
    if (res == kNone32) continue;
    const auto dit = res_def_.find(res);
    if (dit == res_def_.end()) continue;
    const auto& d = dit->second;
    const std::uint64_t cap = resource_cap(d, svc);
    std::uint64_t owner;
    switch (d.scope) {
      case SCOPE_SMSP: owner = (std::uint64_t(w.sm) << 4) | w.smsp; break;
      case SCOPE_SM:   owner = w.sm; break;
      case SCOPE_CLUSTER: owner = w.cta; break;
      default:         owner = 0; break;
    }
    // pick the copy that admits earliest
    ResourceState* best = nullptr;
    std::uint64_t best_ready = ~0ull;
    for (std::uint64_t copy = 0; copy < d.copies; ++copy) {
      auto& r = resource_state(res, copy, owner, d);
      std::uint64_t ready = std::max(now_, r.next_free);
      if (cap)
        ready = capacity_ready(r.completions, cap, ready);
      if (ready < best_ready) { best_ready = ready; best = &r; }
    }
    // initiation interval (fractional, residual accumulation): the
    // unit copy admits its next instruction issue_interval later
    best->next_free = now_ + accumulate_spacing(issue_interval,
                                                best->spacing_residual);
    if (policy_.unit_grant_lwf)
      unit_waiters_[(std::uint64_t(res) << 32) | owner].last_grant =
          w.warp_index;
    if (cap) {
      // drop completed entries, record this one
      best->completions.erase(
          std::remove_if(best->completions.begin(),
                         best->completions.end(),
                         [&](std::uint64_t c) { return c <= now_; }),
          best->completions.end());
      std::uint64_t done = now_ + completion_lat;
      best->completions.push_back(done);
    }
  }
  // register_file.operand_read commit (pipelined collector): n =
  // distinct unserved GPR source registers (64-bit pairs count both);
  // served rows count only when the SMSP's last issue was this warp.
  // The instruction takes the next n read slots: it continues in the
  // current cycle's spare slot when the collector's last entry was
  // this warp's previous instruction and this instruction is not
  // reuse-served; otherwise it starts on a fresh cycle.
  // wgmma register-file issue steal (hopper C3+C7+R): while the
  // warp's warpgroup has a wgmma op in flight the instruction takes
  // one read port exclusively (reuse flags do not serve); a
  // .reuse-flagged issue uses the reuse floor, and while the op
  // sources A from registers (rs) every other issue pays an extra
  // read cycle and issue cycle
  const auto sp = arch_->issue_steal(*this, w);
  const std::uint64_t steal = sp.floor;
  std::uint64_t floor_eff = steal;
  if (operand_cfg_.enabled) {
    auto& s = smsps_[w.sm][w.smsp];
    const std::uint64_t R = operand_cfg_.reads_per_cycle;
    const bool served_ok =
        !steal && s.has_last && s.last_issued == w.warp_index;
    bool has_reuse = false;
    bool any_served = false;
    std::uint64_t n = 0;
    for (std::uint32_t i = 0; i < t.source_count; ++i) {
      const auto& ref = job_.register_refs[t.source_begin + i];
      const std::uint32_t rid = reg_id(ref), rfl = reg_flags(ref);
      if (rid >= 256) continue;
      if (rfl & 4) has_reuse = true;  // bit2 reuse_carrier (v5a1)
      const bool served = (rfl & 1) && served_ok;
      if (served) { any_served = true; continue; }
      // distinct-register count via pairwise scan (n <= 3 typical)
      for (std::uint32_t rr = rid; rr < rid + ((rfl & 2) ? 2u : 1u);
           ++rr) {
        bool dup = false;
        for (std::uint32_t j = 0; j < i; ++j) {
          const auto& q = job_.register_refs[t.source_begin + j];
          const std::uint32_t qid = reg_id(q);
          if (qid >= 256) continue;
          const bool qs = (reg_flags(q) & 1) && served_ok;
          if (!qs && rr >= qid &&
              rr < qid + ((reg_flags(q) & 2) ? 2u : 1u)) {
            dup = true;
            break;
          }
        }
        if (!dup) ++n;
      }
    }
    if (n > 0) {
      const bool fresh = steal ||
                         (s.coll_warp != w.warp_index) || any_served ||
                         (s.coll_slot_end % R == 0);
      const std::uint64_t begin =
          std::max(R * now_,
                   fresh ? ((s.coll_slot_end + R - 1) / R) * R
                         : s.coll_slot_end);
      std::uint64_t extra = 0;
      if (steal) {
        if (sp.rs && (w.rs_toggle ^= 1)) extra = 1;
        floor_eff = has_reuse ? sp.reuse_floor : sp.floor;
        floor_eff += extra;
      }
      s.coll_svc_start = begin / R;
      s.coll_slot_end = steal ? begin + (n + extra) * R : begin + n;
      s.coll_warp = w.warp_index;
    }
  }
  std::uint64_t finish_cycle = now_ + completion_lat;
  // data_ready: when dependents can consume this instruction's result.
  // Starts at result latency; a data-producing return (load, constant,
  // atomic with destination) or bound TIME row completion raises it.
  std::uint64_t data_ready = now_ + result_lat;
  const bool dbg = kDebugInst && dbg_inst == w.next_instance;
  if (dbg)
    fprintf(stderr,
            "DBG_INST %llu issue=%llu warp=%u result_lat=%llu "
            "completion_lat=%llu\n",
            (unsigned long long)w.next_instance,
            (unsigned long long)now_, w.warp_index,
            (unsigned long long)result_lat,
            (unsigned long long)completion_lat);
  if (dbg) {
    auto it = gsync_plan_.acquires.find(w.next_instance);
    if (it != gsync_plan_.acquires.end())
      fprintf(stderr, "DBG_INST %llu gsync mode=%u nprod=%zu addr=%llx\n",
              (unsigned long long)w.next_instance, it->second.mode,
              it->second.producers.size(),
              (unsigned long long)it->second.address);
    else
      fprintf(stderr, "DBG_INST %llu gsync none\n",
              (unsigned long long)w.next_instance);
  }
  mem_return_ = mem_shared_free_ = kNone64;
  mem_sectors_ = mem_dram_sectors_ = 0;
  // read-barrier release floor: MIO-queued rows (kinds 5/6/15) and
  // L1-path loads release their read barrier at SM-side pipe
  // acceptance (MIO queue / L1 port); stores, L1-bypassing copies
  // and TMA / bulk transfers release at issue -- the far-side L2 /
  // DRAM / store-L2 ports never gate the barrier (docs/en/design/engine.md)
  std::uint64_t accept_start = now_, req_accept = now_;
  // memory hierarchy (3-2): MACC/MSEC -> L1/L2/DRAM; result-ready =
  // last sector return; store completes at acceptance.
  if (inst.memory_access != kNone32 && !gsync_failed_poll) {
    const auto& ma = job_.accesses[inst.memory_access];
    const auto* secs = job_.sectors.data() + ma.sector_begin;
    const std::size_t ns = ma.sector_count;
    // MACC flags: low 3 bits = space (1 shared, 3 constant, 0x20
    // remote); access_kind in bits 6-8 (segment_builder.py):
    // 0x0 load, 0x40 store, 0x80 atomic, 0xC0 async_copy,
    // 0x100 async_reduce, 0x140 prefetch.
    const std::uint32_t space = ma.flags & 0x7;
    const std::uint32_t akind = ma.flags & 0x1C0;
    // local space (2) with l1.local_frame_interleaved: expand the
    // trace sectors into this warp's own lane-interleaved frame --
    // every set byte of a trace sector is one byte offset o, each
    // distinct o a separate local sector keyed under a marker bit so
    // it can never collide with a global / constant sector number
    // (load_after_copies v3)
    constexpr std::uint64_t kLocalSectorMark = 1ull << 62;
    // stack buffer; spills to the vector only past 64 sectors
    std::array<MemorySector, 64> loc_small;
    std::vector<MemorySector> loc_big;
    const MemorySector* loc_secs = loc_small.data();
    std::size_t loc_n = 0;
    const bool local_expand =
        space == 2 && mem_cfg_.local_frame_interleaved != 0;
    if (local_expand) {
      for (std::size_t i = 0; i < ns; ++i)
        for (unsigned b = 0; b < 32; ++b) {
          if (!((secs[i].byte_mask >> b) & 1u)) continue;
          const std::uint64_t o = secs[i].address * 32 + b;
          const std::uint64_t key =
              kLocalSectorMark |
              (static_cast<std::uint64_t>(w.warp_index) << 24) |
              (o & 0xFFFFFF);
          bool dup = false;
          for (std::size_t d = 0; d < loc_n; ++d)
            if (loc_secs[d].address == key) { dup = true; break; }
          if (dup) continue;
          const MemorySector s{key, 0xFFFFFFFFu, 0xFFFFFFFFu};
          if (loc_n < loc_small.size()) {
            loc_small[loc_n] = s;
          } else {
            if (loc_big.empty()) {
              loc_big.assign(loc_small.begin(), loc_small.end());
              loc_secs = loc_big.data();
            }
            loc_big.push_back(s);
          }
          ++loc_n;
        }
    }
    const bool store_op = akind == 0x40 || akind == 0x80 ||
                          akind == 0x100;
    const bool atomic = akind == 0x80 || akind == 0x100;
    if (space == 1) {
      // shared-space: bank facts + TIME rows only (no cache path);
      // applied below via timing rows.
    } else if (space == 3) {
      const std::uint64_t cret =
          mem_.constant(w.sm, w.warp_index, now_,
                        ns ? secs[0].address : 0,
                        ns * mem_cfg_.sector_bytes,
                        (ma.flags & 0x10) != 0);
      finish_cycle = std::max(finish_cycle, cret);
      data_ready = std::max(data_ready, cret);
    } else if (store_op) {
      // storage.tma.reduce (kind 81): a reduce's sectors occupy the
      // store port and the shared L2-write service sector_cost_ratio
      // times as long
      double cost_ratio = 1.0;
      {
        const auto* rf = template_timing_fields(
            inst.template_index, kTimingTmaReduceStore, engine_facts_);
        if (rf && MechanismRuntime::field(
                *rf, core::parameter_fields::TIMING_ROW::ENABLED,
                1.0) != 0.0) {
          const double r = MechanismRuntime::field(
              *rf, core::parameter_fields::TIMING_ROW::SECTOR_COST_RATIO,
              0.0);
          if (r > 0.0) cost_ratio = r;
        }
      }
      // lane store (store_flood): a synchronous global STG whose
      // sectors carry per-lane masks; TMA / bulk stores emit sectors
      // with lane_mask 0 and keep their present cost.  (rule S above
      // uses space + kind only: UTMASTG enters as global store and is
      // NOT excluded by it -- the lane_mask test is what separates
      // lane STG from TMA)
      bool lane_store = akind == 0x40 && space == 0;
      if (lane_store) {
        lane_store = false;
        for (std::size_t si = 0; si < ns; ++si)
          if (secs[si].lane_mask != 0) { lane_store = true; break; }
      }
      std::uint64_t fill_done = 0;
      const std::uint64_t sret =
          mem_.store(w.sm, now_, secs, ns, atomic,
                     ns ? secs[0].address : 0, &fill_done,
                     cost_ratio, lane_store);
      finish_cycle = std::max(finish_cycle, sret);
      // a local store leaves its expanded lines resident in the SM's
      // L1 (usable this cycle); the store's cost above stays on the
      // unexpanded trace sectors
      if (local_expand)
        for (std::size_t d = 0; d < loc_n; ++d)
          mem_.l1_insert(w.sm, loc_secs[d].address, now_);
      // an atomic that writes a destination register returns data --
      // not before the slice has fetched a non-resident sector (RMW)
      if (atomic && t.destination_count)
        data_ready = std::max(data_ready, std::max(sret, fill_done));
    } else {
      MemoryHierarchy::HitLevel hl;
      std::uint32_t l1o = 0, l2o = 0, dramo = 0;
      // async_copy / prefetch bypass L1 entirely (P1)
      const bool use_l1 = akind != 0xC0 && akind != 0x140;
      // an async copy without .BYPASS allocates L1 (ldgsts_l1)
      const bool l1_allocate = akind == 0xC0 && (ma.flags & 0x200);
      if (kDebugInst) mem_.dbg_ = dbg;
      // tma_oob v2: a tiled UTMALDG's box is delivered whole even
      // where it is out of bounds -- the excess over the captured
      // (in-bounds) sectors traverses the load path as phantom
      // sectors.  The tile partition of the kind-38 leaf carries
      // CUT_EXTRA_CYCLES; its absence (im2col, bulk, stores) keeps
      // phantom at 0.
      std::uint32_t phantom = 0;
      if (sema.kind == kSemaTmaLoad) {
        for (std::uint64_t ti = t.timing_begin;
             ti < t.timing_begin + t.timing_count; ++ti) {
          if (timing_[ti].kind != kTimingTmaCompletion) continue;
          const auto& fl = timing_fields(ti, nullptr);
          if (fl.find(TR::CUT_EXTRA_CYCLES) != fl.end()) {
            const std::uint64_t sb =
                mem_cfg_.sector_bytes ? mem_cfg_.sector_bytes : 32;
            const std::uint64_t total =
                (sema.field[3] + sb - 1) / sb;
            if (total > ns)
              phantom = static_cast<std::uint32_t>(total - ns);
          }
          break;
        }
      }
      const std::uint64_t ret = mem_.load(
          w.sm, now_, local_expand ? loc_secs : secs,
          local_expand ? loc_n : ns, use_l1, &hl, &l1o,
          &l2o, &dramo, &req_accept, &mem_shared_free_,
          sema.kind == kSemaTmaLoad || sema.kind == kSemaBulkCopy ||
              sema.kind == kSemaBulkReduce, l1_allocate,
              akind == 0x0 && space != 1 && space != 3,
              akind == 0xC0, phantom, local_expand);
      if (kDebugInst) mem_.dbg_ = false;
      mem_return_ = ret;
      if (dbg)
        fprintf(stderr,
                "DBG_INST %llu load ret=%llu hl=%u l1=%u l2=%u dram=%u "
                "ns=%zu akind=%x space=%u accept=%llu\n",
                (unsigned long long)w.next_instance,
                (unsigned long long)ret, hl, l1o, l2o, dramo,
                (size_t)ns, akind, space,
                (unsigned long long)req_accept);
      // cold_fraction = sectors the L2 residency model sent to DRAM /
      // sectors of the access (L2 presence is sector-granular)
      mem_sectors_ = ns;
      mem_dram_sectors_ = dramo;
      // TMA / bulk transfers complete by their own leaf (an L2-hot
      // issue -> complete_tx total) plus the hierarchy's excess over an
      // L2 hit; the level latencies are LDG load-to-use totals and must
      // not be stacked under the leaf (on_semantic_issue)
      const bool leaf_timed_transfer =
          sema.kind == kSemaTmaLoad || sema.kind == kSemaBulkCopy ||
          sema.kind == kSemaBulkReduce;
      if (!leaf_timed_transfer) {
        finish_cycle = std::max(finish_cycle, ret);
        // plain loads raise data_ready; async_copy / prefetch return
        // through their own semantic objects, not the scoreboard
        if (akind != 0xC0 && akind != 0x140)
          data_ready = std::max(data_ready, ret);
      }
      if (ax_cache_hit_ < engine_facts_.size())
        engine_facts_[ax_cache_hit_].second =
            static_cast<std::uint64_t>(hl);
    }
    // TIME rows (3-2 bound kinds): resolve once via the row's USES
    // (FACT + engine axes), write cycles/enabled back, then apply.
    // kinds 3/4 (visibility) are recorded only (3-3).
    for (std::uint64_t ti = ma.timing_begin;
         ti < ma.timing_begin + ma.timing_count; ++ti) {
      auto& tr = timing_[ti];
      switch (tr.kind) {
        case 1:   // global_load_dependency
        case 2:   // global_store_acceptance
        case 5:   // shared_issue_interval_smsp
        case 6:   // shared_store_acceptance_sm
        case 15:  // atomic_issue_interval_sm
          break;
        default:
          continue;  // visibility/barrier rows -> 3-3, record only
      }
      if (!tr.enabled) continue;
      const auto& f = timing_fields(ti);
      if (MechanismRuntime::field(f, core::parameter_fields::
              TIMING_ROW::ENABLED, 1.0) == 0.0) {
        tr.enabled = 0;
        continue;
      }
      const double cyc = MechanismRuntime::field(
          f, core::parameter_fields::TIMING_ROW::CYCLES, 0.0);
      tr.cycles = cyc;  // write-back at first use
      finish_cycle = std::max(finish_cycle,
          now_ + static_cast<std::uint64_t>(cyc) + completion_lat);
      data_ready = std::max(data_ready,
          now_ + static_cast<std::uint64_t>(cyc) + completion_lat);
      if (dbg)
        fprintf(stderr, "DBG_INST %llu acc_time kind=%u cyc=%.1f\n",
                (unsigned long long)w.next_instance, tr.kind, cyc);
    }
  }
  // template-level TIME rows (issue-event programs: storage.shared.*,
  // storage.atomic.issue): interval kinds serialize on their unit —
  // kind 5 per SMSP, kinds 6 / 15 per SM — and the instance completes
  // no earlier than its admission plus the row's cycles; kinds 1 / 2
  // add latency.  (Before 2026-11-04 only access-level rows were
  // applied, so these leaves were inert on both devices.)
  // shared-issue / shared-acceptance rows (kinds 5/6) apply only when
  // this instance's MACC resolves to shared space — a generic LD/ST
  // bound to storage.shared.* by class must not be charged shared
  // timing when its pointer actually reaches global memory.  Every
  // memory-touching instance carries a MACC (zero-sector ones too),
  // so only a present MACC with non-shared space may veto the
  // class-level binding; instances without a MACC keep it.
  const bool shared_space =
      inst.memory_access == kNone32 ||
      (job_.accesses[inst.memory_access].flags & 0x7) == 1;
  accept_start = std::max(accept_start, req_accept);
  std::uint64_t shared_accept_end = now_;
  double fence_drain_cycles = 0.0, fence_drain_rate = 0.0;
  bool fence_drain_row = false;
  for (std::uint64_t ti = t.timing_begin;
       ti < t.timing_begin + t.timing_count; ++ti) {
    auto& tr = timing_[ti];
    if (tr.kind != 1 && tr.kind != 2 && tr.kind != 5 && tr.kind != 6 &&
        tr.kind != 15 && tr.kind != kTimingGlobalStoreIssueInterval &&
        tr.kind != kTimingSharedFenceDrain &&
        tr.kind != kTimingSharedStoreAcceptanceSmsp)
      continue;
    if ((tr.kind == 5 || tr.kind == 6 ||
         tr.kind == kTimingSharedStoreAcceptanceSmsp) && !shared_space)
      continue;
    // kind 65 is the global / local counterpart of kind 5: a present
    // MACC in shared space vetoes it (storage.shared.issue owns those)
    if (tr.kind == kTimingGlobalStoreIssueInterval &&
        inst.memory_access != kNone32 && shared_space)
      continue;
    if (!tr.enabled) continue;
    const auto& f = timing_fields(ti);
    if (MechanismRuntime::field(f, core::parameter_fields::
            TIMING_ROW::ENABLED, 1.0) == 0.0) {
      tr.enabled = 0;
      continue;
    }
    const double cyc = MechanismRuntime::field(
        f, core::parameter_fields::TIMING_ROW::CYCLES, 0.0);
    tr.cycles = cyc;
    // storage.shared.fence (kind 66): no frontier of its own — the row
    // only prices the drain of the CTA's dirty shared stores when the
    // instance is a proxy fence; applied below at issue end
    if (tr.kind == kTimingSharedFenceDrain) {
      fence_drain_row = true;
      fence_drain_cycles = cyc;
      fence_drain_rate = MechanismRuntime::field(
          f, core::parameter_fields::TIMING_ROW::DRAIN_BYTES_PER_CYCLE,
          0.0);
      continue;
    }
    const std::uint64_t c = static_cast<std::uint64_t>(cyc);
    std::uint64_t start = now_;
    std::uint64_t fin = kNone64;  // kind 80 supplies its own end
    if (tr.kind == 5) {
      auto& nx = shared_issue_next_[(std::uint64_t(w.sm) << 4) | w.smsp];
      start = std::max(now_, nx); nx = start + c;
    } else if (tr.kind == 6) {
      auto& nx = shared_accept_next_[w.sm];
      start = std::max(now_, nx); nx = start + c;
      // R2: a synchronous shared store's acceptance shares the per-SM
      // port with fills (storage.shared.fence / R2); its booked end
      // includes the port wait
      std::uint64_t stb = 0;
      if (c > 0 && sync_shared_store(inst, sema, &stb)) {
        const double extra =
            mem_.shared_store_port_wait(w.sm, start, c);
        nx += static_cast<std::uint64_t>(std::ceil(extra));
      }
      shared_accept_end = std::max(shared_accept_end, nx);
    } else if (tr.kind == kTimingSharedStoreAcceptanceSmsp) {
      // per-SMSP acceptance of a synchronous shared store
      // (storage.shared.acceptance_smsp): stores of one SMSP are
      // accepted one after another; the issuing warp is not held.
      // The frontier keeps the fractional part of the measured rate
      // (kinds 5/6/15/65 stay integer); integer views ceil at the
      // boundaries.  Same R2 fill-port stretch as kind 6 but without
      // booking the port occupancy a second time (kind 6 already
      // books it)
      const double cyc80 = cyc;
      auto& nx = shared_accept_smsp_next_[(std::uint64_t(w.sm) << 4) |
                                          w.smsp];
      const double start_d =
          std::max(static_cast<double>(now_), nx);
      nx = start_d + cyc80;
      std::uint64_t stb = 0;
      if (cyc80 > 0 && sync_shared_store(inst, sema, &stb))
        nx += mem_.shared_store_port_wait(w.sm, start_d, cyc80, false);
      start = static_cast<std::uint64_t>(std::ceil(start_d - 1e-9));
      fin = static_cast<std::uint64_t>(std::ceil(nx - 1e-9));
      shared_accept_end = std::max(shared_accept_end, fin);
    } else if (tr.kind == 15) {
      auto& nx = atomic_issue_next_[w.sm];
      start = std::max(now_, nx); nx = start + c;
    } else if (tr.kind == kTimingGlobalStoreIssueInterval) {
      // try_issue gated on the previous admission, so start == now_;
      // the interval is an issue-side occupancy, not a completion term
      auto& nx = store_issue_next_[(std::uint64_t(w.sm) << 4) | w.smsp];
      start = std::max(now_, nx); nx = start + c;
      continue;
    }
    if (tr.kind == 5 || tr.kind == 6 || tr.kind == 15 ||
        tr.kind == kTimingSharedStoreAcceptanceSmsp)
      accept_start = std::max(accept_start, start);
    const std::uint64_t fin_base = fin != kNone64 ? fin : start + c;
    finish_cycle = std::max(finish_cycle, fin_base + completion_lat);
    data_ready = std::max(data_ready, fin_base + completion_lat);
    if (dbg)
      fprintf(stderr, "DBG_INST %llu tmpl_time kind=%u cyc=%.1f "
                      "start=%llu\n",
              (unsigned long long)w.next_instance, tr.kind, cyc,
              (unsigned long long)start);
  }
  // storage.shared.fence: a synchronous shared store charges its CTA
  // the written bytes and the end of its own SM acceptance; a proxy
  // fence issued while that CTA is dirty then holds its warp until
  // acceptance end + cycles + bytes/drain, and clears the ledger
  {
    std::uint64_t st_bytes = 0;
    if (sync_shared_store(inst, sema, &st_bytes)) {
      auto& b = fence_dirty_bytes_[w.cta];
      auto& u = fence_dirty_until_[w.cta];
      b += st_bytes;
      u = std::max(u, shared_accept_end);
    }
    if (fence_drain_row && fence_dirty_bytes_[w.cta] > 0 &&
        !(sema.kind == kSemaDescriptorVisibility &&
          sema.field[0] == 5)) {
      // the CTA memory barrier (MEMBAR.ALL.CTA): an enabled kind-66
      // row without the proxy-fence semantic.  It waits for the dirty
      // stores' acceptance end and then takes only its own service
      // latency -- no `cycles`, no port wait, and the CTA's record
      // stays dirty for the following proxy fence.
      const std::uint64_t floor = static_cast<std::uint64_t>(std::ceil(
          std::max(static_cast<double>(now_),
                   static_cast<double>(fence_dirty_until_[w.cta])))) +
          result_lat;
      w.control_floor = std::max(w.control_floor, floor);
      ++tele_[kTeleFenceDrainHolds];
      tele_[kTeleFenceDrainCycles] += floor - now_;
    }
    if (sema.kind == kSemaDescriptorVisibility && sema.field[0] == 5 &&
        fence_drain_row && fence_dirty_bytes_[w.cta] > 0) {
      const double begin = std::max(
          static_cast<double>(now_),
          static_cast<double>(fence_dirty_until_[w.cta]));
      const double occupy = fence_drain_cycles +
          (fence_drain_rate > 0
               ? static_cast<double>(fence_dirty_bytes_[w.cta]) /
                     fence_drain_rate
               : 0.0);
      // R2: the drain itself also takes turns with fills on the port;
      // the fence then adds its own service result latency on top
      const std::uint64_t done = static_cast<std::uint64_t>(std::ceil(
          begin + result_lat + occupy +
          (occupy > 0.0
               ? mem_.shared_store_port_wait(w.sm, begin, occupy)
               : 0.0)));
      w.control_floor = std::max(w.control_floor, done);
      ++tele_[kTeleFenceDrainHolds];
      tele_[kTeleFenceDrainCycles] += done - now_;
      fence_dirty_bytes_[w.cta] = 0;
      fence_dirty_until_[w.cta] = 0;
    }
  }
  if (dbg)
    fprintf(stderr, "DBG_INST %llu final data_ready=%llu finish=%llu "
                    "mem_return=%llu\n",
            (unsigned long long)w.next_instance,
            (unsigned long long)data_ready,
            (unsigned long long)finish_cycle,
            (unsigned long long)mem_return_);
  // the read barrier protects the sources until the instruction has
  // read them: MIO-queued rows (kinds 5/6/15) and L1-path loads
  // (L1 port start) release at SM-side acceptance, not the issue
  // cycle; nothing downstream of the SM (L2 / DRAM / store-L2 port
  // queues) may hold it (2026-11-08, full-gpu-chain-report §11.3)
  if (t.scoreboard_read != 0xFF && t.scoreboard_read < NO_BARRIER &&
      accept_start > now_)
    w.barrier_free[t.scoreboard_read] = accept_start;
  // dependents consume the result at data_ready (result latency raised
  // by every data-producing return above)
  for (std::uint32_t i = 0; i < t.destination_count; ++i) {
    std::uint32_t reg =
        reg_id(job_.register_refs[t.destination_begin + i]);
    if (reg >= w.reg_ready.size())
      w.reg_ready.resize(
          std::max<std::uint32_t>(max_reg_ + 1, reg + 1), 0);
    w.reg_ready[reg] = data_ready;
    if (kDebugInst)
      w.reg_writer[reg] = static_cast<std::uint32_t>(w.next_instance);
  }
  if (t.scoreboard_write < NO_BARRIER) {
    w.barrier_free[t.scoreboard_write] = data_ready;
    if (kDebugInst)
      w.barrier_writer[t.scoreboard_write] =
          static_cast<std::uint32_t>(w.next_instance);
  }
  // store visibility (§13.8): kind 3/4 TIME rows are settled at the
  // next barrier release.  Template-level rows have no store-accept
  // path — their accept cycle is the issue cycle.
  auto record_visibility = [&](std::uint64_t begin, std::uint32_t count,
                               std::uint64_t accept) {
    for (std::uint64_t ti = begin; ti < begin + count; ++ti) {
      const auto k = timing_[ti].kind;
      if (k == kTimingSharedStoreVisibility ||
          k == kTimingGlobalStoreVisibility)
        pending_visibility_[w.cta].push_back(
            VisibilityEntry{ti, now_, accept});
    }
  };
  record_visibility(t.timing_begin, t.timing_count, now_);
  if (inst.memory_access != kNone32) {
    const auto& ma = job_.accesses[inst.memory_access];
    record_visibility(ma.timing_begin, ma.timing_count, finish_cycle);
  }
  enqueue(Event::kResultReady, finish_cycle, w.warp_index,
          static_cast<std::uint32_t>(w.next_instance), w.smsp);
  on_semantic_issue(w, inst, sema, t, finish_cycle);
  // InstanceTiming control envelope (§13.14): a bound instance pushes
  // the warp's control floor forward
  if (auto ei = inst_envelope_.find(w.next_instance);
      ei != inst_envelope_.end()) {
    const auto& ef = rt_.resolve(ei->second, engine_facts_, nullptr);
    w.control_floor = now_ + static_cast<std::uint64_t>(
        MechanismRuntime::field(ef, 4, 0.0));
  }
  arch_->on_instance_issue(*this, w, inst, sema, finish_cycle);
  // control word stall count (TMPL.control bit0-3): the warp's next
  // instruction issues no earlier than `stall` cycles after this one —
  // the hardware honours the compiler-encoded fixed-latency / hazard
  // distance regardless of register tracking.  max() with the other
  // floors (fence result, control envelope): never additive.
  const std::uint64_t stall = t.control & 0xF;
  if (stall > 1)
    w.control_floor = std::max(w.control_floor, now_ + stall);
  // issue-steal floor (hopper wgmma C3+C7+R): max() like the stall —
  // dependent chains whose own stall >= floor_eff do not move
  if (floor_eff)
    w.control_floor = std::max(w.control_floor, now_ + floor_eff);
  if (opt_.journal)
    out_.journal.push_back(
        JournalRow{w.warp_index, w.smsp,
                   static_cast<std::uint32_t>(w.next_instance), now_,
                   t.operation_class, inst.template_index,
                   inst_wait_bucket, inst_wait_cycles,
                   inst_wait_object});
  ++w.next_instance;
  update_ready_at(w);
  // the issue cycle itself is not a wait cycle
  w.wait_mark = now_ + 1;
  // issued warps with more instances may issue again next cycle
  // (blocked/suspended warps are event-driven and get no retry slot)
  if (w.next_instance < w.end_instance && w.blocked_until <= now_ &&
      !w.deferred.active && !w.cta_wait.active &&
      !w.cluster_wait.active)
    w.retry_at = std::max(now_ + 1, w.control_floor);
  // exhausted warps are settled post-sweep, same as the old active_
  // scan — queue them now so the settle cycle matches
  if (w.next_instance >= w.end_instance)
    just_exhausted_.push_back(w.warp_index);
  return true;
}

// scheduler policy: candidate_order from the predecessor — a cursor
// with commit / yield semantics.
std::vector<std::size_t> Simulator::candidate_order(
    const SmspState& s) const {
  std::vector<std::size_t> order;
  const std::size_t n = s.warps.size();
  order.reserve(n);
  std::size_t start = 0;
  if (s.has_last) {
    std::size_t pos;
    if (s.last_pos < n && s.warps[s.last_pos] == s.last_issued) {
      pos = s.last_pos;
    } else {
      auto it = std::find(s.warps.begin(), s.warps.end(), s.last_issued);
      pos = it == s.warps.end() ? 0 :
          static_cast<std::size_t>(it - s.warps.begin());
    }
    start = s.last_yield ? (pos + 1) % n : pos;
  }
  for (std::size_t o = 0; o < n; ++o)
    order.push_back((start + o) % n);
  return order;
}

bool Simulator::eligible(const WarpState& w) const {
  return !w.done && w.next_instance < w.end_instance;
}

static const Fields kEmptyFields;
const Fields& Simulator::timing_fields(std::uint64_t time_index,
                                       const EngineFacts* extra) {
  if (time_index >= timing_use_.size() ||
      timing_use_[time_index] == kNone64)
    return kEmptyFields;
  // engine-axis-independent rows resolve once and stay cached
  if (!extra) {
    if (time_index < time_cached_.size() && time_cached_[time_index] &&
        time_cache_[time_index])
      return *time_cache_[time_index];
  }
  EngineFacts ef = engine_facts_;
  if (extra) ef.insert(ef.end(), extra->begin(), extra->end());
  std::uint32_t leaf = kNone32;
  bool dep = false;
  const auto& f = rt_.resolve(timing_use_[time_index], ef, &leaf, &dep);
  if (!extra) {
    if (time_index >= time_cached_.size()) {
      time_cached_.resize(time_index + 1, 0);
      time_cache_.resize(time_index + 1, nullptr);
    }
    if (!time_cached_[time_index]) {
      time_cache_[time_index] = dep ? nullptr : &f;
      time_cached_[time_index] = 1;
    }
  }
  // TIME selector write-back (§13.15, diagnostic only)
  auto& tr = timing_[time_index];
  tr.selector_a = leaf;
  tr.selector_b += 1;
  std::uint64_t h = 1469598103934665603ull;
  for (const auto& [k, v] : ef) {
    h ^= k;
    h *= 1099511628211ull;
    h ^= v;
    h *= 1099511628211ull;
  }
  tr.selector_c = h;
  tr.ready_warps = static_cast<std::uint32_t>(
      engine_facts_[ax_ready_sm_].second);
  return f;
}

double Simulator::timing_cycles(std::uint64_t time_index,
                                const EngineFacts* extra) {
  auto& tr = timing_[time_index];
  const auto& f = timing_fields(time_index, extra);
  if (MechanismRuntime::field(f, TR::ENABLED, 1.0) == 0.0) {
    tr.enabled = 0;
    return 0.0;
  }
  const double cyc = MechanismRuntime::field(f, TR::CYCLES, 0.0);
  tr.cycles = cyc;
  return cyc;
}

EngineFacts Simulator::event_facts(
    std::initializer_list<std::pair<const char*, std::uint64_t>> axes)
    const {
  EngineFacts ef = engine_facts_;
  for (const auto& [name, v] : axes)
    if (auto id = rt_.axis_id(name)) ef.emplace_back(*id, v);
  return ef;
}

double Simulator::program_cycles(std::optional<std::uint32_t> prog,
                                 const EngineFacts& ef,
                                 const char* where) const {
  if (!prog) return 0.0;
  bool na = false;
  const auto& f =
      rt_.resolve_node(rt_.program(*prog).root_node, ef, where,
                       nullptr, &na);
  if (na) return 0.0;
  if (MechanismRuntime::field(f, TR::ENABLED, 1.0) == 0.0) return 0.0;
  return MechanismRuntime::field(f, TR::CYCLES, 0.0);
}

std::uint64_t Simulator::ns_to_cycles(std::uint64_t ns) const {
  const std::uint64_t khz = job_.launch.reference_sm_clock_khz;
  if (!khz)
    throw EngineError("unsupported:engine:no_reference_clock ()");
  return ns * khz / 1000000;
}

// ---- semantic readiness & issue-time dispatch (§13.2/§13.3) -------

bool Simulator::failed_poll(std::uint64_t inst_index,
                            const InstanceSemantics& sema) const {
  if (sema.flags & kSemaFlagAfterReady)
    return sema.field[5] == 0;  // known: empty after_ready_mask
  const auto it = next_wait_.find(inst_index);
  return it != next_wait_.end() &&
         it->second.first == sema.field[0] &&
         it->second.second == (sema.field[1] & 1u);
}

std::optional<std::uint64_t> Simulator::gsync_ready_cycle(
    const WarpState& w) const {
  return gsync_ready_cycle_of(w.next_instance);
}

std::optional<std::uint64_t> Simulator::gsync_ready_cycle_of(
    std::uint64_t acquire_instance) const {
  // atom/red instances carry no issue gate: same-address updates are
  // ordered by the L2 atomic unit in arrival order (same_address_service
  // on the store path), not by the issuing SM (§13.10-5)
  auto it = gsync_plan_.acquires.find(acquire_instance);
  if (it == gsync_plan_.acquires.end()) return now_;
  const auto& a = it->second;
  const std::uint64_t r2a = mem_cfg_.release_to_acquire;
  switch (a.mode) {
    case 4:
      return now_;
    case 2: {  // any: earliest completed candidate + r2a
      std::uint64_t best = kNever;
      for (const auto p : a.producers) {
        auto d = gsync_done_.find(p);
        if (d != gsync_done_.end()) best = std::min(best, d->second);
      }
      if (best == kNever) return std::nullopt;
      return best + r2a;
    }
    case 3: {  // count: kth producer completion on this address
      auto v = gsync_by_addr_.find(a.address);
      if (v == gsync_by_addr_.end() ||
          v->second.size() < a.count)
        return std::nullopt;
      auto cyc = v->second;
      std::sort(cyc.begin(), cyc.end());
      return cyc[a.count - 1] + r2a;
    }
    default: {  // from: the single paired producer + r2a
      if (a.producers.empty()) return now_;
      auto d = gsync_done_.find(a.producers[0]);
      if (d == gsync_done_.end()) return std::nullopt;
      return d->second + r2a;
    }
  }
}

bool Simulator::semantic_blocked(const WarpState& w) const {
  if (w.next_instance >= w.end_instance) return false;
  if (auto rg = gsync_region_.find(w.next_instance);
      rg != gsync_region_.end()) {
    const auto cyc = gsync_ready_cycle_of(rg->second);
    return cyc && *cyc > now_;
  }
  const auto& inst = job_.instances[w.next_instance];
  if (inst.semantics == kNone32) return false;
  const auto& sema = job_.semantics[inst.semantics];
  const auto f = sema.field;
  if (sema.kind == kSemaBulkGroupWait ||
      sema.kind == kSemaAsyncCopyWait) {
    const int lane =
        sema.kind == kSemaBulkGroupWait ? AsyncGroups::kBulk
                                        : AsyncGroups::kCpAsync;
    return async_.earliest_ready(lane, w.warp_index, f[1]) > now_;
  }
  if (sema.kind == kSemaDescriptorVisibility && f[0] == 4) {
    auto it = desc_publish_.find(f[2]);
    if (it == desc_publish_.end()) return false;
    const double cyc = program_cycles(prog_desc_visibility_,
                                      engine_facts_,
                                      "descriptor.visibility");
    return now_ < it->second + static_cast<std::uint64_t>(cyc);
  }
  if (sema.kind == kSemaGlobalSync) {
    const auto cyc = gsync_ready_cycle(w);
    return cyc && *cyc > now_;
  }
  if (sema.kind != kSemaMbarrierWait || f[2] > 0) return false;
  if (failed_poll(w.next_instance, sema)) return false;
  const std::uint32_t sob_obj = static_cast<std::uint32_t>(sema.field[0]);
  if (!mbar_.wait_ready(
      job_.cta_segment[w.cta], sob_obj,
      static_cast<std::uint32_t>(sema.field[1] & 1u), now_)) {
    // issued while the phase was incomplete -> suspended waiter
    auto& wm = const_cast<WarpState&>(w);
    if (!wm.suspended_mbar_since) wm.suspended_mbar_since = now_ + 1;
    return true;
  }
  const auto* sob = mbar_.find(job_.cta_segment[w.cta], sob_obj);
  // only a wait that was already blocked before the object's release
  // resumes at suspended_release_cycle; a wait first evaluated ready
  // (issued after completion) is not gated
  const bool gated = sob && w.suspended_mbar_since &&
      w.suspended_mbar_since - 1 < sob->release_cycle &&
      now_ < sob->suspended_release_cycle;
  if (!gated) const_cast<WarpState&>(w).suspended_mbar_since = 0;
  return gated;
}

bool Simulator::semantic_ready(WarpState& w,
                               const InstructionInstance& inst,
                               const InstanceSemantics& sema) {
  static const bool dbg_cpasync =
      std::getenv("CYCLESCOPE_DEBUG_CPASYNC") != nullptr;
  if (inst.semantics == kNone32) return true;
  const auto f = sema.field;
  const std::uint32_t seg = seg_of(w);
  if (sema.kind == kSemaBulkGroupWait || sema.kind == kSemaAsyncCopyWait) {
    const int lane =
        sema.kind == kSemaBulkGroupWait ? AsyncGroups::kBulk
                                        : AsyncGroups::kCpAsync;
    // flags bit2: close the open group exactly once for this instance
    if ((sema.flags & kSemaFlagImplicitCommit) &&
        w.implicit_committed_instance != w.next_instance) {
      async_.commit(lane, w.warp_index, now_);
      w.implicit_committed_instance = w.next_instance;
    }
    // the wait blocks the warp on the wait instruction itself until
    // at most `pending_allowed` groups remain
    if (dbg_cpasync)
      fprintf(stderr,
              "DBG_CPASYNC wait w=%u inst=%llu now=%llu n_copies=%zu\n",
              w.warp_index, (unsigned long long)w.next_instance,
              (unsigned long long)now_,
              async_.copies_since_arrives(w.warp_index).size());
    std::uint64_t earliest = 0;
    if (!async_.wait_ready(lane, w.warp_index, f[1], now_, &earliest)) {
      w.last_wait = WaitBucket::kCollective;
      w.retry_at = earliest;
      if (dbg_cpasync)
        fprintf(stderr, "DBG_CPASYNC wait w=%u wake=%llu\n",
                w.warp_index, (unsigned long long)earliest);
      return false;
    }
    if (dbg_cpasync)
      fprintf(stderr, "DBG_CPASYNC wait w=%u pass now=%llu\n",
              w.warp_index, (unsigned long long)now_);
    return true;
  }
  if (sema.kind == kSemaDescriptorVisibility && f[0] == 4) {
    // acquire: ready once the publish has had time to become visible
    auto it = desc_publish_.find(f[2]);
    if (it == desc_publish_.end()) {
      ++tele_[kTeleDescriptorAcquireUnpublished];
      return true;
    }
    const double cyc = program_cycles(prog_desc_visibility_,
                                      engine_facts_,
                                      "descriptor.visibility");
    const std::uint64_t ready =
        it->second + static_cast<std::uint64_t>(cyc);
    if (now_ < ready) {
      // readiness cycle is known: suspend until it (semantic bucket)
      w.blocked_until = ready;
      w.block_bucket = WaitBucket::kSemantic;
      return false;
    }
    return true;
  }
  if (sema.kind == kSemaGlobalSync) {
    if (auto a = gsync_plan_.acquires.find(w.next_instance);
        a != gsync_plan_.acquires.end())
      w.wait_detail = a->second.address;
    const auto cyc = gsync_ready_cycle(w);
    const bool ready = cyc && now_ >= *cyc;
    if (!ready) {
      // blocked cycles accrue lazily: first failure starts the span,
      // later verdicts (retry or producer completion) extend it;
      // identical to counting every cycle the wait failed
      if (w.gsync_wait_mark != ~0ull)
        tele_[kTeleGsyncWaitCycles] += now_ - w.gsync_wait_mark;
      w.gsync_wait_mark = now_;
      // the ready cycle may already be determined — let the loop
      // land on it instead of waiting for an unrelated event
      if (cyc && *cyc > now_) w.retry_at = *cyc;
      gsync_wait_smsps_.insert(
          (std::uint64_t(w.sm) << 3) | w.smsp);
      w.last_wait = WaitBucket::kSemantic;
    } else if (w.gsync_wait_mark != ~0ull) {
      tele_[kTeleGsyncWaitCycles] += now_ - w.gsync_wait_mark;
      w.gsync_wait_mark = ~0ull;
    }
    return ready;
  }
  if (sema.kind != kSemaMbarrierWait) return true;
  const std::uint32_t object = static_cast<std::uint32_t>(f[0]);
  const std::uint32_t parity =
      static_cast<std::uint32_t>(f[1] & 1u);
  if (failed_poll(w.next_instance, sema)) {
    mbar_.remove_waiter(seg, object, w.warp_index);
    return true;  // polls issue; the warp suspends after issue
  }
  if (mbar_.wait_ready(seg, object, parity, now_)) {
    mbar_.remove_waiter(seg, object, w.warp_index);
    return true;
  }
  if (f[2] > 0) return true;  // deferred wait block
  mbar_.add_waiter(seg, object, w.warp_index);
  return false;
}

void Simulator::on_semantic_issue(WarpState& w,
                                  const InstructionInstance& inst,
                                  const InstanceSemantics& sema,
                                  const InstructionTemplate& t,
                                  std::uint64_t finish_cycle) {
  if (inst.semantics == kNone32) return;
  const auto f = sema.field;
  const std::uint64_t inst_index = w.next_instance;
  const std::uint32_t seg = seg_of(w);
  auto tmpl_kind_cycles = [&](std::uint32_t kind, bool* found) {
    if (found) *found = false;
    double cyc = 0.0;
    for (std::uint64_t ti = t.timing_begin;
         ti < t.timing_begin + t.timing_count; ++ti)
      if (timing_[ti].kind == kind) {
        cyc = timing_cycles(ti);
        if (found) *found = true;
        break;
      }
    return cyc;
  };
  // an optional extra field of the template's row of `kind` (present
  // only when the leaf carries it)
  auto tmpl_kind_field = [&](std::uint32_t kind, std::uint32_t field,
                             bool* found) {
    *found = false;
    for (std::uint64_t ti = t.timing_begin;
         ti < t.timing_begin + t.timing_count; ++ti)
      if (timing_[ti].kind == kind) {
        const auto& fl = timing_fields(ti, nullptr);
        auto it = fl.find(field);
        if (it == fl.end()) return 0.0;
        *found = true;
        return it->second;
      }
    return 0.0;
  };
  switch (sema.kind) {
    case kSemaMbarrierInit: {
      auto do_init = [&](std::uint32_t object) {
        mbar_.init(seg, object, f[1], f[2] == 1, w.cta, now_,
                   static_cast<std::uint32_t>(f[3] & 1));
      };
      // MBOL: one init row covering several stage barriers (a
      // lane-parallel store promoted to init): initialise every listed
      // object; otherwise the single f[0] object
      const std::uint64_t list_begin =
          job_.segment_mbol_base[seg] + (f[5] >> 32),
          list_count = f[5] & 0xFFFFFFFFu;
      if (list_count) {
        for (std::uint64_t i = 0; i < list_count; ++i)
          do_init(job_.mbarrier_objects[list_begin + i].object);
      } else {
        do_init(static_cast<std::uint32_t>(f[0]));
      }
      break;
    }
    case kSemaMbarrierArrive: {
      const std::uint64_t count = f[3], tx = f[2];
      if (count == 0 && tx == 0)
        throw EngineError(
            "missing_fact:mbarrier_arrive_count (object " +
            std::to_string(f[0]) + ")");
      auto do_arrive = [&](std::uint32_t s, std::uint32_t object,
                           std::uint64_t credit) {
        ++tele_[kTeleMbarrierArrivals];
        tele_[kTeleMbarrierTxBytes] += tx;
        if (mbar_.arrive(s, object, credit, tx, w.warp_index, now_))
          mbarrier_phase_complete(s, object);
      };
      // .A<n>TX: the row completes the transaction bytes (0 arrivals)
      const bool complete_tx = (sema.flags & kSemaFlagCompleteTx) != 0;
      auto do_dest = [&](std::uint32_t s, std::uint32_t object,
                         std::uint64_t credit) {
        if (complete_tx) {
          tele_[kTeleMbarrierTxBytes] += tx;
          mbarrier_complete_tx(s, object, tx);
        } else {
          do_arrive(s, object, credit);
        }
      };
      const auto* splt = split_credit_.count(inst.semantics)
          ? &split_credit_.at(inst.semantics) : nullptr;
      if (f[4] == 0) {
        do_dest(seg, static_cast<std::uint32_t>(f[0]), count);
      } else {
        // destination_mask: one arrive per rank via MBOB (SPLT
        // overrides the per-rank credit when present)
        const std::uint32_t issuer_rank =
            job_.ctas[w.cta].cluster_rank;
        for (std::uint32_t rank = 0; rank < 32; ++rank) {
          if (!(f[4] >> rank & 1)) continue;
          auto [ds, dob] = resolve_destination(seg, f[0], rank,
                                               issuer_rank);
          const std::uint64_t credit =
              splt && splt->count(rank) ? splt->at(rank) : count;
          if (dob != f[0] || ds != seg)
            ++tele_[kTeleMbarrierRemoteArrivals];
          do_dest(ds, dob, credit);
        }
      }
      const std::uint64_t list_begin =
          job_.segment_mbol_base[seg] + (f[5] >> 32),
          list_count = f[5] & 0xFFFFFFFFu;
      for (std::uint64_t i = 0; i < list_count; ++i)
        do_dest(seg, job_.mbarrier_objects[list_begin + i].object,
                count);
      break;
    }
    case kSemaMbarrierExpectTx:
      tele_[kTeleMbarrierTxBytes] += f[2];
      if (mbar_.expect_tx(seg, static_cast<std::uint32_t>(f[0]), f[2],
                          now_))
        mbarrier_phase_complete(seg, static_cast<std::uint32_t>(f[0]));
      break;
    case kSemaMbarrierWait: {
      const std::uint32_t object = static_cast<std::uint32_t>(f[0]);
      const std::uint32_t parity = static_cast<std::uint32_t>(f[1] & 1u);
      w.last_wait_object = object;
      w.last_wait_parity = parity;
      if (failed_poll(inst_index, sema)) {
        ++tele_[kTeleFailedPolls];
        // retried polls cost nothing beyond completion; only a poll
        // abandoned by the warp holds trywait_timeout (§13.3.6)
        const bool abandoned = abandoned_poll_.count(inst_index) != 0;
        if (abandoned) ++tele_[kTeleAbandonedPolls];
        double timeout =
            abandoned ? program_cycles(prog_trywait_, engine_facts_,
                                       "trywait_timeout")
                      : 0.0;
        if (f[4] > 0)
          timeout = timeout > 0.0
              ? std::min(timeout, static_cast<double>(ns_to_cycles(f[4])))
              : static_cast<double>(ns_to_cycles(f[4]));
        if (timeout > 0.0) {
          // suspend until min(release if known, now + timeout)
          const auto* o = mbar_.find(seg, object);
          std::uint64_t wake = now_ + static_cast<std::uint64_t>(timeout);
          bool release_known = false;
          if (o && o->initialized && (o->phase % 2) != parity) {
            wake = std::min(wake, o->release_cycle);
            release_known = true;
          }
          if (wake > now_) {
            w.blocked_until = wake;
            w.block_bucket = WaitBucket::kCollective;
            tele_[kTelePollSuspendCycles] += wake - now_;
            if (!release_known && o && o->initialized)
              mbar_.add_sleeper(seg, object, w.warp_index, wake);
            enqueue(Event::kWake, wake, w.warp_index, 0, 0);
          }
        }
      } else if (!mbar_.wait_ready(seg, object, parity, now_) && f[2] > 0) {
        // successful wait with a deferred block (§13.3.6)
        const std::uint64_t block_index = inst_index + f[2];
        if (block_index >= w.end_instance)
          throw EngineError(
              "unsupported:engine:wait_block_offset (warp " +
              std::to_string(w.warp_index) + ")");
        w.deferred = WarpState::Deferred{true, object, parity,
                                         block_index};
      }
      break;
    }
    case kSemaMbarrierInval: {
      auto inflight = mbar_.inval(seg, static_cast<std::uint32_t>(f[0]));
      ++tele_[kTeleMbarrierInvals];
      tele_[kTeleMbarrierInvalInflight] +=
          static_cast<std::uint64_t>(std::max<std::int64_t>(
              0, inflight.first + inflight.second));
      break;
    }
    case kSemaTmaLoad:
    case kSemaBulkCopy:
    case kSemaBulkReduce: {
      // TMA / bulk transfer completion -> complete_tx (§13.7)
      const bool is_tma = sema.kind == kSemaTmaLoad;
      const double cyc21 = tmpl_kind_cycles(kTimingTmaServiceStart,
                                            nullptr);
      // tma_oob v2: a box cut along dimension 0 owes this SM
      // cut_service_cycles (= 2.02 x cut_rows on h100) of service
      // before the next TMA may start; the gap belongs to the
      // predecessor box
      bool have_cut_svc = false;
      const double cut_svc = tmpl_kind_field(
          kTimingTmaServiceStart, TR::CUT_SERVICE_CYCLES, &have_cut_svc);
      // service starts serialize per SM; the first TMA has no
      // predecessor and starts at its issue cycle
      auto it = sm_async_last_start_.find(w.sm);
      std::uint64_t start =
          it == sm_async_last_start_.end()
              ? now_
              : std::max(now_,
                         it->second + sm_async_prev_gap_[w.sm]);
      sm_async_last_start_[w.sm] = start;
      sm_async_prev_gap_[w.sm] = std::max<std::uint64_t>(
          static_cast<std::uint64_t>(cyc21),
          have_cut_svc ? static_cast<std::uint64_t>(
                             std::llround(cut_svc))
                       : 0);
      // Hopper descriptor refetch: first tensor_map on this SM, or a
      // descriptor acquire happened since the last UTMALDG
      bool have43 = false;
      const double cyc43 = tmpl_kind_cycles(kTimingDescriptorRefetch,
                                            &have43);
      if (have43) {
        const std::uint64_t map = f[5];
        if (!sm_seen_maps_[w.sm].count(map) || sm_desc_acquired_[w.sm])
          start += static_cast<std::uint64_t>(cyc43);
        sm_seen_maps_[w.sm].insert(map);
        sm_desc_acquired_[w.sm] = false;
      }
      bool have38 = false;
      const double cyc38 = tmpl_kind_cycles(kTimingTmaCompletion,
                                            &have38);
      std::uint64_t finish = std::max(
          finish_cycle, start + static_cast<std::uint64_t>(cyc38));
      bool have_cold = false;
      const double cold_extra = tmpl_kind_field(
          kTimingTmaCompletion, TR::COLD_EXTRA_CYCLES, &have_cold);
      if (have38 && have_cold) {
        // tma_cold protocol rule: completion = service start + hot leaf
        // + cold_fraction x cold_extra, cold_fraction = sectors the L2
        // residency model sent to DRAM / sectors of the access; the
        // hierarchy's latency does not enter -- the leaves (and the
        // kind-21 service start) are single-SM totals.  What no
        // cluster-local leaf can hold is the queueing this transfer's
        // sectors suffered in the GPU-shared L2 / DRAM services
        // (memory.shared_service): return - shared_free is added
        // (2026-11-08: cublas bigK / skinny ran at 2.4-2.6x the DRAM
        // leaf's bandwidth, full-gpu-chain-report.md §12).  The per-SM
        // port is not counted here: TMA-TMA serialization on one SM is
        // the kind-21 service-start interval (tma_forms protocol)
        const double frac = mem_sectors_
            ? static_cast<double>(mem_dram_sectors_) / mem_sectors_ : 0.0;
        finish = std::max(finish_cycle, start +
            static_cast<std::uint64_t>(cyc38 + frac * cold_extra));
        if (mem_return_ != kNone64 && mem_shared_free_ != kNone64 &&
            mem_return_ > mem_shared_free_)
          finish += mem_return_ - mem_shared_free_;
      } else if (mem_return_ != kNone64) {
        // no cold leg on this device: leaf total (L2-hot) + hierarchy
        // excess over an L2 hit (DRAM miss, port / shared-bandwidth
        // waits); without a leaf the hierarchy return stands alone
        const std::uint64_t l2_hit = now_ + static_cast<std::uint64_t>(
            mem_cfg_.l2_latency);
        const std::uint64_t excess =
            have38 ? (mem_return_ > l2_hit ? mem_return_ - l2_hit : 0)
                   : (mem_return_ > now_ ? mem_return_ - now_ : 0);
        finish = std::max(finish, start +
                          static_cast<std::uint64_t>(cyc38) + excess);
      }
      // tma_oob v2: a cut box completes +178 after the leaf /
      // hierarchy computation (0 on whole-row-outside and in-bounds
      // boxes -> no change there)
      bool have_cut_extra = false;
      const double cut_extra = tmpl_kind_field(
          kTimingTmaCompletion, TR::CUT_EXTRA_CYCLES, &have_cut_extra);
      if (have_cut_extra && cut_extra > 0)
        finish += static_cast<std::uint64_t>(cut_extra);
      const std::uint64_t bytes = is_tma ? f[3] : f[2];
      static const bool kDebugTma =
          std::getenv("CYCLESCOPE_DEBUG_TMA") != nullptr;
      if (kDebugTma) {
        std::uint64_t sectors = 0;
        if (inst.memory_access != kNone32)
          sectors = job_.accesses[inst.memory_access].sector_count;
        const double frac = mem_sectors_
            ? static_cast<double>(mem_dram_sectors_) / mem_sectors_ : 0.0;
        fprintf(stderr,
                "TMA issue=%llu complete=%llu bytes=%llu sectors=%llu "
                "hot=%llu cold_extra=%llu cold_fraction=%.3f "
                "hier_return=%llu dram_sectors=%llu l2_sectors=%llu "
                "sm=%u warp=%u\n",
                (unsigned long long)now_, (unsigned long long)finish,
                (unsigned long long)bytes, (unsigned long long)sectors,
                have38 ? (unsigned long long)cyc38 : 0ull,
                have_cold ? (unsigned long long)cold_extra : 0ull, frac,
                (unsigned long long)mem_return_,
                (unsigned long long)mem_dram_sectors_,
                (unsigned long long)(mem_sectors_ - mem_dram_sectors_),
                w.sm, w.warp_index);
      }
      if (bytes == 0 && inst.memory_access == kNone32)
        throw EngineError(
            "missing_fact:tma_transfer_bytes (instance " +
            std::to_string(inst_index) + ")");
      const std::uint64_t mask = is_tma ? f[2] : f[3];
      // cta_group::2 (.2CTA): a destination rank's bytes post on its
      // pair leader's barrier (rank & ~1)
      const bool pair = (sema.flags & kSemaFlagCtaPair) != 0;
      TransferCompletion tc;
      if (mask == 0) {
        tc.dests.push_back({seg, static_cast<std::uint32_t>(f[0])});
      } else {
        const std::uint32_t issuer_rank =
            job_.ctas[w.cta].cluster_rank;
        for (std::uint32_t rank = 0; rank < 32; ++rank) {
          if (!(mask >> rank & 1)) continue;
          // two ranks of one pair each count their own bytes
          const std::uint32_t target = pair ? (rank & ~1u) : rank;
          tc.dests.push_back(
              resolve_destination(seg, f[0], target, issuer_rank));
        }
      }
      tc.bytes = bytes;
      transfer_completions_.push_back(std::move(tc));
      w.inflight_until = std::max(w.inflight_until, finish);
      enqueue(Event::kTransferComplete, finish, w.warp_index, 0,
              transfer_completions_.size() - 1);
      break;
    }
    case kSemaBulkGroupIssue: {
      // UTMASTG & friends: op completion = max(memory finish, kind-38)
      bool have38 = false;
      const double cyc38 = tmpl_kind_cycles(kTimingTmaCompletion,
                                            &have38);
      const std::uint64_t completion = std::max(
          finish_cycle, now_ + static_cast<std::uint64_t>(cyc38));
      async_.issue(AsyncGroups::kBulk, w.warp_index, completion, f[2]);
      break;
    }
    case kSemaBulkGroupCommit:
      async_.commit(AsyncGroups::kBulk, w.warp_index, now_);
      break;
    case kSemaBulkGroupWait:
    case kSemaAsyncCopyWait:
      break;  // readiness + implicit commit handled in semantic_ready
    case kSemaAsyncCopyIssue: {
      std::uint64_t bytes = 0;
      if (inst.memory_access != kNone32) {
        const auto& ma = job_.accesses[inst.memory_access];
        for (std::uint32_t i = 0; i < ma.sector_count; ++i) {
          const std::uint32_t m =
              job_.sectors[ma.sector_begin + i].byte_mask;
          bytes += m ? static_cast<std::uint64_t>(
                           __builtin_popcount(m))
                     : mem_cfg_.sector_bytes;
        }
      }
      // storage.ldgsts.completion (kind 39): the async-copy op
      // completes at max(finish, issue + cycles); enabled=0 disables.
      {
        std::uint64_t completion = finish_cycle;
        double leaf_cyc = -1.0;
        const auto* gf = template_timing_fields(
            inst.template_index, /*kind*/ 39, engine_facts_);
        if (gf && MechanismRuntime::field(*gf, TR::ENABLED, 1.0) != 0.0)
          leaf_cyc = MechanismRuntime::field(*gf, TR::CYCLES, 0.0);
        // the group leaf models a whole group's latency from its
        // FIRST copy's issue: anchor the per-copy completion on the
        // open group's first issue, not on this copy's issue (was
        // issue_k + cycles -> group completed at last_issue + cycles)
        if (async_.open_first_issue(AsyncGroups::kCpAsync,
                                    w.warp_index) == ~0ull)
          async_.set_open_first_issue(AsyncGroups::kCpAsync,
                                      w.warp_index, now_);
        if (leaf_cyc >= 0.0)
          completion = std::max(
              completion,
              async_.open_first_issue(AsyncGroups::kCpAsync,
                                      w.warp_index) +
                  static_cast<std::uint64_t>(leaf_cyc));
        static const bool dbg_cpasync =
            std::getenv("CYCLESCOPE_DEBUG_CPASYNC") != nullptr;
        async_.issue(AsyncGroups::kCpAsync, w.warp_index, completion,
                     bytes);
        if (dbg_cpasync)
          fprintf(stderr,
                  "DBG_CPASYNC w=%u inst=%llu issue=%llu finish=%llu "
                  "leaf=%.1f first=%llu completion=%llu\n",
                  w.warp_index, (unsigned long long)w.next_instance,
                  (unsigned long long)now_,
                  (unsigned long long)finish_cycle, leaf_cyc,
                  (unsigned long long)async_.open_first_issue(
                      AsyncGroups::kCpAsync, w.warp_index),
                  (unsigned long long)completion);
      }
      break;
    }
    case kSemaAsyncCopyCommit: {
      const std::uint32_t object = static_cast<std::uint32_t>(f[2]);
      if (object == kSemaNoObject) {
        async_.commit(AsyncGroups::kCpAsync, w.warp_index, now_);
        break;
      }
      // ARRIVES: copies complete, then a delayed arrive/complete_tx
      std::uint64_t base = now_;
      std::uint64_t bytes = 0;
      for (const auto& [comp, b] :
           async_.copies_since_arrives(w.warp_index)) {
        base = std::max(base, comp);
        bytes += b;
      }
      bool have18 = false;
      const double cyc18 = tmpl_kind_cycles(
          kTimingAsyncCopyMbarrierArrival, &have18);
      const std::uint64_t at =
          base + static_cast<std::uint64_t>(cyc18);
      // TRANSCNT: complete the lane transactions the paired
      // SYNCS.ARRIVE.RED.A0T1 expected (f[4]); legacy rows without the
      // field fall back to the copied bytes
      deferred_arrivals_.push_back(
          DeferredArrive{seg, object, w.warp_index, f[3],
                         f[4] > 0 ? f[4] : bytes});
      w.inflight_until = std::max(w.inflight_until, at);
      enqueue(Event::kDeferredArrive, at, w.warp_index, 0,
              deferred_arrivals_.size() - 1);
      async_.clear_copies(w.warp_index);
      break;
    }
    case kSemaDescriptorVisibility:
      if (f[0] == 1) {
        // publish: record the addresses pushed visible
        for (std::uint64_t i = 0; i < f[1] && i < 4; ++i)
          desc_publish_[f[2 + i]] = now_;
      } else if (f[0] == 4) {
        // acquire: marks this SM for the next UTMALDG refetch check
        sm_desc_acquired_[w.sm] = true;
      }
      break;  // operations 3/5 carry no engine timing this step
    case kSemaGlobalSync:
      break;  // acquire counters are incremented at plan time
    case kSemaCtaBarrierArrive:
    case kSemaCtaBarrierWait: {
      auto r = cta_bar_.arrive(w.cta, static_cast<std::uint32_t>(f[0]),
                               w.warp_index, f[1], inst_index, now_);
      if (sema.kind == kSemaCtaBarrierWait)
        w.cta_wait = WarpState::CtaWait{
            true, static_cast<std::uint32_t>(f[0]), r.generation};
      if (r.full)
        cta_generation_close(w.cta, static_cast<std::uint32_t>(f[0]), r);
      break;
    }
    case kSemaClusterBarrierArrive: {
      ++w.cluster_arrive_gen;
      auto r = cluster_bar_.arrive(seg, w.cluster_arrive_gen, w.cta,
                                   w.warp_index, inst_index, now_);
      if (r.full) cluster_generation_close(seg, r);
      break;
    }
    case kSemaClusterBarrierWait:
      w.cluster_wait = WarpState::ClusterWait{true, w.cluster_arrive_gen};
      break;
    case kSemaSharedStoreAsync: {
      // st.async (STAS, sm_90 and sm_100): the converter already resolved
      // the destination CTA's object (owner rank from the DSMEM tag of
      // the [Rn.64] pair's barrier word); the store completes
      // transaction bytes f[1] on it at finish + kind-21 service start
      bool have21 = false;
      const double cyc21 = tmpl_kind_cycles(kTimingTmaServiceStart, &have21);
      TransferCompletion tc;
      // f[3] = object_list_begin|count: per-lane destinations (each
      // lane stored into its own rank's barrier, duplicates kept);
      // absent -> the single resolved object in f[0]
      const std::uint32_t list_count = f[3] & 0xFFFFFFFFu;
      if (list_count) {
        const std::uint64_t base =
            job_.segment_mbol_base[seg] + (f[3] >> 32);
        for (std::uint64_t i = 0; i < list_count; ++i)
          tc.dests.push_back({seg, job_.mbarrier_objects[base + i].object});
      } else {
        tc.dests.push_back({seg, static_cast<std::uint32_t>(f[0])});
      }
      tc.bytes = f[1];
      transfer_completions_.push_back(std::move(tc));
      const std::uint64_t at =
          finish_cycle + static_cast<std::uint64_t>(cyc21);
      w.inflight_until = std::max(w.inflight_until, at);
      enqueue(Event::kTransferComplete, at, w.warp_index, 0,
              transfer_completions_.size() - 1);
      break;
    }
    case kSemaNanosleep: {
      double cyc = 0.0;
      double syncs_wake = 0.0;
      bool have_row = false;
      for (std::uint64_t ti = t.timing_begin;
           ti < t.timing_begin + t.timing_count; ++ti) {
        if (timing_[ti].kind != kTimingNanosleepWarpDelay) continue;
        cyc = timing_cycles(ti);
        const auto& tf = timing_fields(ti);
        syncs_wake = MechanismRuntime::field(
            tf, core::parameter_fields::TIMING_ROW::SYNCS_WAKE_CYCLES,
            0.0);
        have_row = true;
        break;
      }
      if (!have_row) cyc = static_cast<double>(ns_to_cycles(f[0]));
      // PTX ISA caps the nanosleep duration at 1 ms
      // (= reference_sm_clock_khz cycles); skipped when the launch
      // carries no reference clock (preserves the no-clock path).
      if (job_.launch.reference_sm_clock_khz)
        cyc = std::min(cyc, static_cast<double>(
                              job_.launch.reference_sm_clock_khz));
      std::uint64_t wake = now_ + static_cast<std::uint64_t>(cyc);
      if (sema.flags & kSemaFlagSyncs) {
        // NANOSLEEP.SYNCS is the suspended try_wait: it ends when the
        // polled objects' awaited phases is released.
        // Already released (the simulated timeline ran ahead of the
        // captured failed poll) -> no sleep; release pending -> at the
        // release; otherwise the sleeper registration covers the phase
        // completion.
        // Poll regions (§13.9): the warp's instance stream is split
        // at every non-failed mbarrier wait; a NANOSLEEP.SYNCS awaits
        // every known-failed poll object in its region plus the
        // region-closing wait's (object, parity).  Divergent warps
        // interleave polls on several objects between two successful
        // waits, so a bounded distance scan both misses objects
        // behind an intervening arrive/other sync op and sees only
        // one branch of the alternation.  The last wait is the
        // fallback for a poll-free region.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> objects;
        auto collect = [&](std::uint32_t o, std::uint32_t p) {
          for (const auto& e : objects)
            if (e.first == o && e.second == p) return;
          objects.emplace_back(o, p);
        };
        const auto rit = nanosleep_region_.find(inst_index);
        if (rit != nanosleep_region_.end())
          for (std::uint32_t k = rit->second.first;
               k < rit->second.second; ++k)
            collect(nanosleep_objects_[k].first,
                    nanosleep_objects_[k].second);
        if (objects.empty())
          collect(w.last_wait_object, w.last_wait_parity);
        bool ready = false;
        for (const auto& [object, parity] : objects) {
          if (object == kNone32) continue;
          auto* o = mbar_.find(seg, object);
          if (o && o->initialized &&
              mbar_.wait_ready(seg, object, parity, now_)) {
            ready = true;
            break;
          }
        }
        if (ready) {
          wake = now_;
        } else {
          for (const auto& [object, parity] : objects) {
            if (object == kNone32) continue;
            auto* o = mbar_.find(seg, object);
            if (!o || !o->initialized) continue;
            if ((o->phase % 2) != parity && o->release_cycle > now_)
              wake = std::min(wake,
                              o->release_cycle +
                                  static_cast<std::uint64_t>(
                                      syncs_wake));
            mbar_.add_sleeper(seg, object, w.warp_index, wake,
                              static_cast<std::uint64_t>(syncs_wake));
          }
        }
      }
      ++tele_[kTeleNanosleeps];
      if (wake > now_) {
        w.blocked_until = wake;
        w.block_bucket = WaitBucket::kCollective;
        enqueue(Event::kWake, wake, w.warp_index, 0, 0);
      }
      break;
    }
    default:
      break;  // kinds bound in later steps stay non-blocking
  }
}

void Simulator::on_warp_exit(WarpState& w) {
  for (auto& [barrier, r] :
       cta_bar_.warp_exited(w.cta, w.warp_index, now_))
    cta_generation_close(w.cta, barrier, r);
  for (auto& r : cluster_bar_.warp_exited(seg_of(w), w.warp_index,
                                          now_))
    cluster_generation_close(seg_of(w), r);
  // an exiting warp can never be stalled at the rule S store gate
  // (it must issue the STG before exhausting), but keep the per-SM
  // FIFO free of stale entries in case a future gate moves earlier
  if (w.sm < store_gate_q_.size()) {
    auto& q = store_gate_q_[w.sm];
    q.erase(std::remove(q.begin(), q.end(), w.warp_index), q.end());
  }
  arch_->on_warp_exit(*this, w);
}

std::uint64_t Simulator::settle_visibility(std::uint32_t cta,
                                           std::uint64_t upto_cycle,
                                           std::uint64_t warps) {
  auto it = pending_visibility_.find(cta);
  if (it == pending_visibility_.end()) return 0;
  std::uint64_t rel = 0;
  auto& list = it->second;
  for (auto e = list.begin(); e != list.end();) {
    if (e->issue_cycle > upto_cycle) { ++e; continue; }
    EngineFacts ef = event_facts({{"actual_participant_warps", warps}});
    const double cyc = timing_cycles(e->time_index, &ef);
    rel = std::max<std::uint64_t>(
        rel, e->accept_cycle + static_cast<std::uint64_t>(cyc));
    ++tele_[kTeleVisibilitySettled];
    e = list.erase(e);
  }
  return rel;
}

std::uint32_t Simulator::seg_of(const WarpState& w) const {
  return job_.cta_segment[w.cta];
}

std::pair<std::uint32_t, std::uint32_t>
Simulator::resolve_destination(std::uint32_t segment,
                               std::uint64_t object, std::uint32_t rank,
                               std::uint32_t issuer_rank) const {
  const std::uint32_t ob = static_cast<std::uint32_t>(object);
  if (segment >= mbob_rows_.size() || mbob_rows_[segment].empty()) {
    // no MBOB table: only a self-rank destination is resolvable
    if (rank != issuer_rank)
      throw EngineError(
          "missing_fact:mbarrier_object_table (object " +
          std::to_string(object) + ")");
    return {segment, ob};
  }
  if (ob >= mbob_rows_[segment].size())
    throw EngineError(
        "missing_fact:mbarrier_object_table (object " +
        std::to_string(object) + ")");
  const auto& row = mbob_rows_[segment][ob];
  auto ci = rank_cta_[segment].find(rank);
  if (ci == rank_cta_[segment].end())
    throw EngineError(
        "missing_fact:mbarrier_object_table (object " +
        std::to_string(object) + ")");
  auto it = mbob_rev_[segment].find({ci->second, row.address});
  if (it == mbob_rev_[segment].end())
    throw EngineError(
        "missing_fact:mbarrier_object_table (object " +
        std::to_string(object) + ")");
  return {segment, it->second};
}

void Simulator::mbarrier_phase_complete(std::uint32_t segment,
                                        std::uint32_t object) {
  auto* o = mbar_.find(segment, object);
  ++tele_[kTeleMbarrierPhases];
  EngineFacts ef = event_facts({
      {"expected_tx_bytes", o->release_expected_tx_bytes},
      {"completed_tx_bytes", o->release_completed_tx_bytes},
      {"actual_participant_warps", o->release_participant_warps},
      {"same_parity", 0}});
  const std::uint64_t release_cyc = static_cast<std::uint64_t>(
      program_cycles(prog_mbarrier_release_, ef, "mbarrier.release"));
  std::uint64_t rel = o->phase_complete_cycle + release_cyc;
  // a wait that was issued while the phase was incomplete suspends and
  // wakes at completion + suspended_wake_cycles (not + cycles); waits
  // issued after completion keep release_cyc (trywait_wake protocol).
  // The field is read off the same resolved leaf as cycles; absent -> 0
  // -> bit-identical old behaviour.
  double susp = 0.0;
  if (prog_mbarrier_release_) {
    bool na = false;
    const auto& rf = rt_.resolve_node(
        rt_.program(*prog_mbarrier_release_).root_node, ef,
        "mbarrier.release.suspended_wake", nullptr, &na);
    if (!na)
      susp = MechanismRuntime::field(
          rf, TR::SUSPENDED_WAKE_CYCLES, 0.0);
  }
  if (susp > 0.0)
    o->suspended_release_cycle = o->phase_complete_cycle +
        static_cast<std::uint64_t>(std::llround(susp));
  else
    o->suspended_release_cycle = rel;
  if (o->owner_cta != kNone32)
    rel = std::max(rel, settle_visibility(o->owner_cta, now_,
                                          o->release_participant_warps));
  o->release_cycle = rel;
  enqueue(Event::kBarrierRelease, rel, kNone32, 0,
          (std::uint64_t(object) & 0xFFFFFFFFull));
  // wake warps suspended until this object's release (failed-poll
  // trywait suspensions and NANOSLEEP.SYNCS sleepers)
  const std::uint64_t wake_base = o->suspended_release_cycle;
  if (susp > 0.0)
    tele_[kTeleSuspendedWaits] += o->sleepers.size();
  for (const auto& s : o->sleepers) {
    auto& w = warps_[s.warp];
    const std::uint64_t wk = wake_base + s.wake_latency;
    if (!w.done && w.blocked_until == s.blocked_until &&
        w.blocked_until > wk) {
      w.blocked_until = wk;
      enqueue(Event::kWake, wk, s.warp, 0, 0);
    }
  }
  o->sleepers.clear();
}

void Simulator::cta_generation_close(
    std::uint32_t cta, std::uint32_t barrier,
    const CtaBarrierTable::ArriveResult& r) {
  ++tele_[kTeleCtaBarrierGenerations];
  double cyc = 0.0;
  // kind-17 TIME row on the last arriving instance's template
  if (r.last_arrive_instance != kNone64) {
    const auto& lt = job_.templates[
        job_.instances[r.last_arrive_instance].template_index];
    for (std::uint64_t ti = lt.timing_begin;
         ti < lt.timing_begin + lt.timing_count; ++ti) {
      if (timing_[ti].kind != kTimingCtaBarrierRelease) continue;
      EngineFacts ef = event_facts(
          {{"actual_participant_warps", r.participant_warps}});
      cyc = timing_cycles(ti, &ef);
      break;
    }
  }
  std::uint64_t rel = now_ + static_cast<std::uint64_t>(cyc);
  rel = std::max(rel, settle_visibility(cta, now_, r.participant_warps));
  // P3: drain-queued shared stores must be visible before release
  if (arch_ && cta < cta_sm_.size())
    rel = std::max(rel, arch_->release_floor(cta_sm_[cta], rel));
  cta_bar_.set_release(cta, barrier, r.generation, rel);
  enqueue(Event::kBarrierRelease, rel, kNone32, 0,
          (1ull << 40) | (std::uint64_t(cta) << 16) | barrier);
}

void Simulator::cluster_generation_close(
    std::uint32_t cluster, const ClusterBarrierTable::ArriveResult& r) {
  ++tele_[kTeleClusterBarrierGenerations];
  double cyc = 0.0;
  bool row_found = false;
  if (r.last_arrive_instance != kNone64) {
    const auto& lt = job_.templates[
        job_.instances[r.last_arrive_instance].template_index];
    for (std::uint64_t ti = lt.timing_begin;
         ti < lt.timing_begin + lt.timing_count; ++ti) {
      if (timing_[ti].kind != kTimingClusterBarrierRelease) continue;
      EngineFacts ef = event_facts({
          {"expected_arrivals",
           static_cast<std::uint64_t>(segment_cta_count_[cluster])},
          {"actual_participant_ctas", r.participant_ctas},
          {"actual_participant_warps", r.participant_warps}});
      cyc = timing_cycles(ti, &ef);
      row_found = true;
      break;
    }
  }
  if (!row_found) {
    EngineFacts ef = event_facts({
        {"expected_arrivals",
         static_cast<std::uint64_t>(segment_cta_count_[cluster])},
        {"actual_participant_ctas", r.participant_ctas},
        {"actual_participant_warps", r.participant_warps}});
    cyc = program_cycles(prog_cluster_release_, ef, "cluster.release");
  }
  const std::uint64_t rel = now_ + static_cast<std::uint64_t>(cyc);
  cluster_bar_.set_release(cluster, r.generation, rel);
  enqueue(Event::kBarrierRelease, rel, kNone32, 0, (2ull << 40) | cluster);
}

void Simulator::mark_smsp_of(const WarpState& w) {
  if (w.resident && w.sm < smsps_.size())
    smsps_[w.sm][w.smsp].needs_eval = true;
}

void Simulator::flush_wait(WarpState& w, std::uint64_t now) {
  if (now > w.wait_mark) {
    const std::uint64_t delta = now - w.wait_mark;
    wait_[static_cast<int>(w.last_wait)] += delta;
    if (opt_.wait_ledger) {
      auto& e = wait_ledger_[(std::uint64_t(w.warp_index) << 8) |
                             std::uint64_t(w.last_wait)];
      e.first += delta;
      ++e.second;
    }
    if (w.stall_since == 0) w.stall_since = w.wait_mark ? w.wait_mark
                                                      : (now ? now : 1);
    w.wait_mark = now;
  }
}

void Simulator::step_smsp(std::uint32_t sm, std::uint32_t smsp) {
  auto& s = smsps_[sm][smsp];
  if (s.warps.empty()) { s.needs_eval = false; return; }
  s.needs_eval = false;
  std::uint32_t issued_slots = 0;
  // candidate order inline: cursor after last issued (or at it when
  // it did not yield), wrapping the whole list once
  const std::size_t n = s.warps.size();
  std::size_t start = 0;
  if (s.has_last) {
    std::size_t pos;
    if (s.last_pos < n && s.warps[s.last_pos] == s.last_issued) {
      pos = s.last_pos;
    } else {
      auto it = std::find(s.warps.begin(), s.warps.end(), s.last_issued);
      pos = it == s.warps.end() ? 0 :
          static_cast<std::size_t>(it - s.warps.begin());
    }
    start = s.last_yield ? (pos + 1) % n : pos;
  }
  std::size_t cursor = 0;
  for (; cursor < n; ++cursor) {
    if (issued_slots >= policy_.issue_slots) break;
    auto& w = warps_[s.warps[(start + cursor) % n]];
    if (!w.done && w.next_instance >= w.end_instance) {
      // exhaustion is settled post-sweep to keep the original
      // ordering (barrier exit deductions happen after all steps)
      just_exhausted_.push_back(s.warps[(start + cursor) % n]);
      continue;
    }
    if (!eligible(w) || w.issue_stamp == now_) continue;
    w.retry_at = 0;
    // account the cycles since the last verdict into the old bucket
    // (per-cycle counting replaced by interval accounting)
    flush_wait(w, now_);
    // readiness order (§13.2): dispatch -> collective waits -> try_issue
    if (!w.resident || w.entry_cycle > now_) {
      w.last_wait = WaitBucket::kDispatch;
      if (w.resident) w.retry_at = w.entry_cycle;
      continue;
    }
    // instruction fetch (§13.13): one ready() per (warp, instance)
    if (fetch_enabled_) {
      const std::uint64_t* ca = code_address_of(w.next_instance);
      if (ca) {
        if (w.fetch_instance != w.next_instance) {
          w.fetch_instance = w.next_instance;
          const std::uint64_t p = fetch_.parallelism();
          const std::uint32_t ch = static_cast<std::uint32_t>(
              (p <= 4 ? w.smsp : w.slot) % p);
          w.fetch_ready =
              fetch_.ready(w.sm, w.smsp, ch, *ca, now_,
                           w.warp_index);
          if (w.fetch_ready > now_)
            tele2_[kTele2FetchWaitCycles] += w.fetch_ready - now_;
        }
        if (w.fetch_ready > now_) {
          clear_unit_waiter(w);
          w.last_wait = WaitBucket::kFetch;
          w.retry_at = w.fetch_ready;
          continue;
        }
      }
    }
    if (w.blocked_until > now_) {
      clear_unit_waiter(w);
      w.last_wait = w.block_bucket;
      w.retry_at = w.blocked_until;
      continue;
    }
    if (w.deferred.active && w.next_instance == w.deferred.block_index) {
      const auto* dob = mbar_.find(seg_of(w), w.deferred.object);
      if (mbar_.wait_ready(seg_of(w), w.deferred.object,
                           w.deferred.parity, now_) &&
          now_ >= (dob ? dob->suspended_release_cycle : 0)) {
        if (dob && dob->suspended_release_cycle > dob->release_cycle)
          ++tele_[kTeleSuspendedWaits];
        w.deferred = WarpState::Deferred{};
      } else {
        clear_unit_waiter(w);
        w.last_wait = WaitBucket::kCollective;
        continue;
      }
    }
    if (w.cta_wait.active) {
      if (cta_bar_.wait_ready(w.cta, w.cta_wait.barrier,
                              w.cta_wait.generation, now_)) {
        w.cta_wait = WarpState::CtaWait{};
      } else {
        clear_unit_waiter(w);
        w.last_wait = WaitBucket::kCollective;
        continue;
      }
    }
    if (w.cluster_wait.active) {
      if (cluster_bar_.wait_ready(seg_of(w), w.cluster_wait.generation,
                                  now_)) {
        w.cluster_wait = WarpState::ClusterWait{};
      } else {
        clear_unit_waiter(w);
        w.last_wait = WaitBucket::kCollective;
        continue;
      }
    }
    if (try_issue(w)) {
      s.last_issued = w.warp_index;
      s.last_pos = (start + cursor) % n;
      s.has_last = true;
      s.last_yield = (job_.templates[
          job_.instances[w.next_instance - 1].template_index].control >>
          4) & 1;
      ++issued_slots;
    } else {
      // a verdict other than "unit busy / outbid" ends the waitership
      if (!w.unit_blocked) clear_unit_waiter(w);
      if (!policy_.try_next_on_block) break;
    }
  }
  // candidates never tried because the slots ran out may be re-attempted
  // next cycle (they were not proven blocked)
  for (; cursor < n; ++cursor) {
    auto& w = warps_[s.warps[(start + cursor) % n]];
    // retry slots from an earlier round that already fell due are
    // stale — the earliest legal re-attempt is next cycle
    if (eligible(w) && w.resident && w.retry_at <= now_)
      w.retry_at = now_ + 1;
  }
  // this SMSP's next due retry: warps with retry_at == 0 are
  // semantic/event waits — they contribute nothing to advance and
  // are re-judged whenever this SMSP steps for another reason
  s.next_attempt = kNever;
  for (const auto wi : s.warps) {
    const auto& w = warps_[wi];
    if (eligible(w) && w.retry_at)
      s.next_attempt = std::min(s.next_attempt, w.retry_at);
  }
}

void Simulator::schedule(const Event& e) {
  // wakes: any event carrying a warp marks that warp's SMSP for
  // evaluation at this cycle (register waits sleep via kWake /
  // kResultReady already carry retry_at, but the mark is harmless)
  if (e.warp != kNone32 && e.warp < warps_.size())
    mark_smsp_of(warps_[e.warp]);
  switch (e.kind) {
    case Event::kResultReady:
      ++out_.completed;
      {
        auto& s = smsps_[warps_[e.warp].sm][e.payload];
        s.last_active = e.cycle;
      }
      // GSYN producer: a producer instance's completion is the
      // cycle acquires may compare against
      if (auto it = gsync_plan_.producer_address.find(e.instance);
          it != gsync_plan_.producer_address.end()) {
        gsync_done_[e.instance] = e.cycle;
        gsync_by_addr_[it->second].push_back(e.cycle);
        // acquire waiters on this producer re-judge now
        for (const auto key : gsync_wait_smsps_)
          smsps_[key >> 3][key & 7].needs_eval = true;
      }
      break;
    case Event::kDeferredArrive: {
      const auto& da = deferred_arrivals_[e.payload];
      if (da.credit > 0) {
        ++tele_[kTeleMbarrierArrivals];
        if (mbar_.arrive(da.segment, da.object, da.credit, 0, da.warp,
                         e.cycle))
          mbarrier_phase_complete(da.segment, da.object);
      } else if (mbar_.complete_tx(da.segment, da.object, da.bytes,
                                   e.cycle)) {
        mbarrier_phase_complete(da.segment, da.object);
      }
      break;
    }
    case Event::kTransferComplete: {
      const auto& tc = transfer_completions_[e.payload];
      ++tele_[kTeleTmaCompletions];
      tele_[kTeleTmaBytes] += tc.bytes;
      for (const auto& [s, ob] : tc.dests)
        if (mbar_.complete_tx(s, ob, tc.bytes, e.cycle))
          mbarrier_phase_complete(s, ob);
      break;
    }
    case Event::kDispatch:
      if (e.payload == ~0ull)
        try_dispatch_queue(e.cycle);  // capacity-release tick
      else if (e.payload & kDispatchReleaseTag) {
        // a slot becomes placeable slot_free_to_entry after the
        // completion of the CTA that held it (§13.11)
        const std::uint32_t c = static_cast<std::uint32_t>(
            e.payload & ~kDispatchReleaseTag);
        dispatcher_.release(cta_sm_[c], cta_slot_[c],
                            job_.ctas[c].warp_count);
        try_dispatch_queue(e.cycle);
      } else
        admit_cluster(static_cast<std::uint32_t>(e.payload));
      break;
    case Event::kReadBarrierFree:
      break;
    case Event::kResourceFree:
      break;
    case Event::kWarpDone:
      warps_[e.warp].done = true;
      break;
    case Event::kWake:
      break;  // exists only to land `now` on the wake cycle
    case Event::kBarrierRelease:
      // waiters re-check readiness when now reaches this cycle —
      // mark every SMSP (cheap relative to how rarely barriers close,
      // and covers deferred waits, trywait suspensions and CTA /
      // cluster barriers without per-object registries)
      for (auto& smsp_set : smsps_)
        for (auto& s : smsp_set) s.needs_eval = true;
      break;
    case Event::kArch:
      break;  // module-defined payload; handled via on_event below
  }
  arch_->on_event(*this, e);
}

void Simulator::deadlock_report() const {
  std::string first;
  std::uint32_t lines = 0;
  for (const auto& w : warps_) {
    if (w.done) continue;
    std::uint32_t object = kNone32;
    std::string pending_sema;
    if (w.deferred.active) object = w.deferred.object;
    std::string line = "warp " + std::to_string(w.warp_index) +
        " cta " + std::to_string(w.cta) +
        " inst " + std::to_string(w.next_instance) +
        " bucket " + std::to_string(static_cast<int>(w.last_wait));
    if (w.next_instance < w.end_instance) {
      // the instance the warp is parked on: class + semantic row
      const auto& inst = job_.instances[w.next_instance];
      line += " op " + std::string(job_.strings.c_str() +
          job_.templates[inst.template_index].operation_class);
      if (inst.semantics != kNone32) {
        const auto& s = job_.semantics[inst.semantics];
        pending_sema = " sema kind " + std::to_string(s.kind) +
            " f0 " + std::to_string(s.field[0]) +
            " f1 " + std::to_string(s.field[1]) +
            " f2 " + std::to_string(s.field[2]) +
            " f3 " + std::to_string(s.field[3]);
        if (s.kind == kSemaMbarrierWait && object == kNone32)
          object = static_cast<std::uint32_t>(s.field[0]);
      }
    }
    if (object == kNone32 && w.last_wait_object != kNone32)
      object = w.last_wait_object;
    line += pending_sema;
    if (object != kNone32) {
      line += " object " + std::to_string(object);
      if (const auto* o = mbar_.find(seg_of(w), object))
        line += (o->initialized ? " init" : " uninit") +
            std::string(" expected ") +
            std::to_string(o->expected_arrivals) +
            " phase " + std::to_string(o->phase) +
            " pending " + std::to_string(o->pending_arrivals) +
            " tx " + std::to_string(o->tx_count) +
            " release " + std::to_string(o->release_cycle) +
            " waiters " + std::to_string(o->waiters.size());
      else
        line += " (no state)";
    }
    if (w.cta_wait.active)
      line += " cta_barrier " + std::to_string(w.cta_wait.barrier) +
          " gen " + std::to_string(w.cta_wait.generation);
    if (w.cluster_wait.active)
      line += " cluster_barrier gen " +
          std::to_string(w.cluster_wait.generation);
    if (w.wait_detail)
      line += " gsync address " + std::to_string(w.wait_detail);
    if (!w.resident)
      line += " resident=0";
    if (w.entry_cycle > now_)
      line += " entry=" + std::to_string(w.entry_cycle);
    if (w.fetch_ready > now_)
      line += " fetch_ready=" + std::to_string(w.fetch_ready);
    if (w.blocked_until > now_)
      line += " blocked_until " + std::to_string(w.blocked_until);
    if (first.empty()) first = line;
    if (lines < 20)
      std::fprintf(stderr, "deadlock: %s\n", line.c_str());
    ++lines;
  }
  throw EngineError("unsupported:engine:deadlock (" + first + ")");
}

ResultStats Simulator::run() {
  load_setup();
  place_warps();
  arch_->on_load(*this);
  sm_ready_stamp_.assign(smsps_.size(), kNever);
  for (;;) {
    while (!queue_.empty() && queue_.top().cycle <= now_) {
      Event e = queue_.top(); queue_.pop();
      schedule(e);
    }
    const bool want_ready_stats = ax_ready_smsp_ != kNone64;
    for (std::uint32_t sm = 0; sm < smsps_.size(); ++sm) {
      // only SMSPs with a due retry or an event wake are stepped;
      // ready-warp stats are computed lazily once per (sm, cycle)
      // and only when the axes exist in the job at all
      std::uint64_t ready_sm = 0;
      bool stats_done = false;
      for (std::uint32_t sp = 0; sp < smsps_per_sm_; ++sp) {
        auto& s = smsps_[sm][sp];
        if (s.warps.empty()) { s.needs_eval = false; continue; }
        if (!s.needs_eval && s.next_attempt > now_) continue;
        if (want_ready_stats && !stats_done) {
          stats_done = true;
          if (sm_ready_stamp_[sm] != now_) {
            sm_ready_stamp_[sm] = now_;
            ready_smsp_scratch_.assign(smsps_per_sm_, 0);
            for (std::uint32_t q = 0; q < smsps_per_sm_; ++q)
              for (std::uint32_t wi : smsps_[sm][q].warps) {
                const auto& w = warps_[wi];
                if (!eligible(w) || !w.resident) continue;
                if (w.blocked_until > now_ || w.deferred.active ||
                    w.cta_wait.active || w.cluster_wait.active ||
                    w.ready_at_ > now_ ||
                    semantic_blocked(w))
                  continue;
                ++ready_smsp_scratch_[q];
              }
          }
          for (const auto c : ready_smsp_scratch_) ready_sm += c;
        }
        if (want_ready_stats) {
          engine_facts_[ax_ready_smsp_].second =
              ready_smsp_scratch_[sp];
          engine_facts_[ax_ready_sm_].second = ready_sm;
        }
        step_smsp(sm, sp);
      }
    }
    for (std::size_t i = 0; i < just_exhausted_.size(); ++i) {
      WarpState& w = warps_[just_exhausted_[i]];
      if (w.next_instance >= w.end_instance && !w.done) {
        // barrier participation ends at EXIT, but the warp stays
        // resident until async work in flight has drained (§13.6)
        if (!w.exit_signaled) {
          w.exit_signaled = true;
          on_warp_exit(w);
        }
        const std::uint64_t drain = std::max(
            async_.drain_until(w.warp_index), w.inflight_until);
        if (drain <= now_) {
          flush_wait(w, now_);
          w.done = true;
          // CTA slot frees only when its last warp is done (§13.11);
          // with CTAs queued, the slot becomes placeable only
          // slot_free_to_entry after ITS completion: a release event
          // tagged with the CTA fires then, releases the slot and
          // retries the queue
          if (--cta_live_warps_[w.cta] == 0 && dispatch_.enabled) {
            // an executed launch_dependents keeps the GPU-wide
            // dispatch serializer busy dependents_launch_serial
            // past this CTA's completion (§13.11)
            if (cta_dependents_launch_[w.cta] &&
                dispatch_.dependents_launch_serial > 0)
              dispatch_last_entry_ =
                  std::max(static_cast<double>(now_) -
                               dispatch_.serial_per_cta,
                           dispatch_last_entry_) +
                  dispatch_.dependents_launch_serial;
            if (!dispatch_queue_.empty()) {
              // SMs in slow gpcs hold the empty slot longer
              const auto* tr = dispatcher_.topo(cta_sm_[w.cta]);
              const double slot =
                  (tr && tr->gpc < 64 &&
                   (dispatch_.slow_gpc_mask >> tr->gpc) & 1)
                      ? dispatch_.slot_free_to_entry_slow
                      : dispatch_.slot_free_to_entry;
              enqueue(Event::kDispatch,
                      now_ + static_cast<std::uint64_t>(
                          std::llround(slot)),
                      kNone32, 0, kDispatchReleaseTag | w.cta);
            } else {
              dispatcher_.release(cta_sm_[w.cta], cta_slot_[w.cta],
                                  job_.ctas[w.cta].warp_count);
            }
          }
        } else if (!w.draining) {
          w.draining = true;
          w.blocked_until = drain;
          w.block_bucket = WaitBucket::kCollective;
          enqueue(Event::kWake, drain, w.warp_index, 0, 0);
        }
      }
      if (w.done) {
        const std::uint32_t jx = just_exhausted_[i];
        if (jx < active_pos_.size() &&
            active_pos_[jx] < active_.size() &&
            active_[active_pos_[jx]] == jx) {
          const std::uint32_t pos = active_pos_[jx];
          std::swap(active_[pos], active_.back());
          active_pos_[active_[pos]] = pos;
          active_.pop_back();
        } else {
          auto it = std::find(active_.begin(), active_.end(), jx);
          if (it != active_.end()) {
            std::swap(*it, active_.back());
            active_pos_[*it] =
                static_cast<std::uint32_t>(it - active_.begin());
            active_.pop_back();
          }
        }
        --pending_warps_;
      }
    }
    just_exhausted_.clear();
    const bool pending = pending_warps_ > 0;
    // earliest due retry across SMSPs (semantic waits contribute
    // nothing — events wake their SMSP)
    std::uint64_t next_attempt = kNever;
    for (const auto& smsp_set : smsps_)
      for (const auto& s : smsp_set)
        next_attempt = std::min(next_attempt, s.next_attempt);
    // stale wake-ups (the warp was woken earlier by a release / sleeper
    // registration, or has exited) must not move time: a 10 ms
    // NANOSLEEP hint otherwise stretches the kernel to its full hint
    while (!queue_.empty() && queue_.top().kind == Event::kWake) {
      const auto& e = queue_.top();
      const auto& sw = warps_[e.warp];
      if (sw.done || sw.blocked_until != e.cycle) queue_.pop();
      else break;
    }
    std::uint64_t advance = next_attempt;
    if (!queue_.empty())
      advance = std::min(advance, queue_.top().cycle);
    if (advance != kNever) {
      // events enqueued during this round for the current cycle are
      // processed before time moves on (release lands at its cycle)
      if (advance <= now_) continue;
      now_ = advance;
    } else {
      if (!pending) break;
      // no retry candidate and no events: no progress is possible
      deadlock_report();
    }
    if (opt_.max_cycles && now_ > opt_.max_cycles) {
      ++out_.hard_failures;
      break;
    }
    if (opt_.progress && (now_ & 0xFFFFF) == 0)
      std::fprintf(stderr, "cycle %llu issued %llu\n",
                   (unsigned long long)now_,
                   (unsigned long long)out_.issued);
  }
  // kernel.lifecycle envelope (§13.12): applied once past the last
  // completion; field presence decides the Blackwell/Hopper shape
  std::uint64_t envelope = 0;
  if (lifecycle_use_ != kNone64) {
    const auto& f = rt_.resolve(lifecycle_use_, engine_facts_, nullptr);
    envelope = lifecycle_envelope(f, sm_used_.size());
  }
  tele2_[kTele2LifecycleEnvelope] = envelope;
  for (auto& w : warps_)
    if (!w.done) flush_wait(w, now_);
  tele2_[kTele2FetchL0Hits] = fetch_.l0_hits;
  tele2_[kTele2FetchL1Lines] = fetch_.l1_lines;
  tele2_[kTele2FetchL2Lines] = fetch_.l2_lines;
  tele2_[kTele2FetchColdLines] = fetch_.cold_lines;
  arch_->collect_tele(out_.tele);
  out_.cycles = now_ + envelope;
  for (std::uint32_t sm = 0; sm < smsps_.size(); ++sm)
    for (std::uint32_t sp = 0; sp < smsps_per_sm_; ++sp) {
      const auto& s = smsps_[sm][sp];
      out_.smsp_spans.push_back(SmspSpanRow{
          sm, sp,
          s.first_active == kNone64 ? 0 : s.first_active,
          s.last_active});
    }
  for (int i = 0; i < (int)WaitBucket::kCount; ++i)
    if (wait_[i])
      out_.wait.push_back({static_cast<std::uint32_t>(i), wait_[i]});
  for (const auto& [k, r] : resources_)
    if (r.wait_cycles)
      out_.rwait.push_back({r.resource, r.wait_cycles});
  if (opt_.wait_ledger) {
    std::vector<std::pair<std::uint64_t,
        std::pair<std::uint64_t, std::uint64_t>>> rows(
        wait_ledger_.begin(), wait_ledger_.end());
    std::sort(rows.begin(), rows.end());
    for (const auto& [k, v] : rows)
      out_.wait_ledger.push_back(WaitLedgerRow{
          static_cast<std::uint32_t>(k >> 8),
          static_cast<std::uint32_t>(k & 0xFF), v.first,
          static_cast<std::uint32_t>(v.second)});
  }
  out_.tele.push_back({0, 0, out_.issued});
  for (std::uint32_t c = 0; c < kTeleCount; ++c)
    if (tele_[c]) out_.tele.push_back({1, c, tele_[c]});
  {
    std::uint64_t n = 0, sum = 0, mx = 0;
    for (std::size_t c = 0; c < cta_first_issue_.size(); ++c) {
      if (cta_first_issue_[c] == kNone64) continue;
      const std::uint64_t span = cta_last_issue_[c] - cta_first_issue_[c];
      ++n; sum += span; mx = std::max(mx, span);
    }
    if (n) { tele2_[kTele2CtaSpanMean] = sum / n; tele2_[kTele2CtaSpanMax] = mx; }
  }
  for (std::uint32_t c = 0; c < kTele2Count; ++c)
    if (tele2_[c]) out_.tele.push_back({2, c, tele2_[c]});
  // group 4: memory hierarchy counters (fixed order, docs/en/engine.md)
  {
    const auto& mt = mem_.tele;
    const std::uint64_t g4[] = {
        mt.l1_sectors, mt.l1_misses, mt.l2_sectors, mt.dram_sectors,
        mt.dram_bytes, mt.merge_merges, mt.l2_far_hits,
        mt.l2_replica_fills, mt.dirty_writebacks, mt.shared_waits,
        mt.wait_cycles, mt.l2_evictions, mt.const_same_line,
        mt.const_hits, mt.const_misses, mt.l2_inflight_hits,
        mt.l2_inflight_wait, mt.atomic_fills,
        mt.loaded_extra_dram_cycles, mt.loaded_extra_dram_count,
        mt.loaded_extra_l2_cycles, mt.loaded_extra_l2_count,
        mt.port_share_waits, mt.port_share_wait_cycles,
        mt.ordered_load_waits, mt.ordered_load_wait_cycles,
        mt.phantom_sectors};
    for (std::uint32_t c = 0; c < std::size(g4); ++c)
      if (g4[c]) out_.tele.push_back({4, c, g4[c]});
  }
  return out_;
}

bool Simulator::mbarrier_arrive(std::uint32_t segment,
                                std::uint32_t object,
                                std::uint64_t credit, std::int64_t tx,
                                std::uint32_t warp) {
  if (mbar_.arrive(segment, object, credit, tx, warp, now_)) {
    mbarrier_phase_complete(segment, object);
    return true;
  }
  return false;
}

bool Simulator::mbarrier_complete_tx(std::uint32_t segment,
                                     std::uint32_t object,
                                     std::uint64_t bytes) {
  if (mbar_.complete_tx(segment, object, bytes, now_)) {
    mbarrier_phase_complete(segment, object);
    return true;
  }
  return false;
}

const Fields* Simulator::template_timing_fields(
    std::uint32_t template_index, std::uint32_t kind,
    const EngineFacts& extra) {
  const auto& t = job_.templates[template_index];
  for (std::uint64_t ti = t.timing_begin;
       ti < t.timing_begin + t.timing_count; ++ti) {
    if (timing_[ti].kind != kind) continue;
    return &timing_fields(ti, &extra);
  }
  return nullptr;
}

EngineFacts Simulator::template_timing_capture_facts(
    std::uint32_t template_index, std::uint32_t kind) const {
  const auto& t = job_.templates[template_index];
  for (std::uint64_t ti = t.timing_begin;
       ti < t.timing_begin + t.timing_count; ++ti) {
    if (timing_[ti].kind != kind) continue;
    if (ti >= timing_use_.size() || timing_use_[ti] == kNone64)
      return {};
    return rt_.use_capture_facts(timing_use_[ti]);
  }
  return {};
}

EngineFacts Simulator::template_capture_facts(
    std::uint32_t template_index) const {
  EngineFacts out;
  auto add = [&](const EngineFacts& src) {
    for (const auto& [k, v] : src) {
      bool have = false;
      for (const auto& e : out)
        if (e.first == k) { have = true; break; }
      if (!have) out.emplace_back(k, v);
    }
  };
  if (template_index < tmpl_use_.size() &&
      tmpl_use_[template_index] != kNone64)
    add(rt_.use_capture_facts(tmpl_use_[template_index]));
  const auto& t = job_.templates[template_index];
  for (std::uint64_t ti = t.timing_begin;
       ti < t.timing_begin + t.timing_count; ++ti) {
    if (ti >= timing_use_.size() || timing_use_[ti] == kNone64)
      continue;
    add(rt_.use_capture_facts(timing_use_[ti]));
  }
  return out;
}

double Simulator::program_cycles_by_name(const char* name,
                                         const EngineFacts& ef) {
  const auto prog = rt_.program_by_name(name);
  return program_cycles(prog, ef, name);
}

}  // namespace cyclescope
