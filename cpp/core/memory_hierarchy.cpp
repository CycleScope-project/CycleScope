#include "memory_hierarchy.hpp"

#include <algorithm>
#include <cmath>

namespace cyclescope {

MemoryHierarchy::MemoryHierarchy(MemoryConfig cfg) : cfg_(cfg) {
  if (cfg_.sector_bytes == 0) cfg_.sector_bytes = 32;
  if (cfg_.line_bytes == 0) cfg_.line_bytes = 128;
  if (cfg_.pool_count == 0) cfg_.pool_count = 2;
  std::sort(cfg_.shared_configurations.begin(),
            cfg_.shared_configurations.end());
  std::uint32_t gpc_max = 0;
  bool gpc_any = false;
  for (const std::uint32_t g : cfg_.sm_gpc)
    if (g != 0xFFFFFFFFu) {
      gpc_any = true;
      gpc_max = std::max(gpc_max, g);
    }
  if (gpc_any) {
    gpc_l2_read_.assign(gpc_max + 1, SharedService{});
    gpc_l2_write_.assign(gpc_max + 1, SharedService{});
  }
}

void MemoryHierarchy::L1Set::touch(std::uint64_t s) {
  auto it = pos.find(s);
  if (it == pos.end()) return;
  order.splice(order.end(), order, it->second.first);
}
void MemoryHierarchy::L1Set::insert(std::uint64_t s, std::uint64_t ready) {
  if (pos.count(s)) { pos[s].second = ready; touch(s); return; }
  order.push_back(s);
  pos.emplace(s, std::make_pair(std::prev(order.end()), ready));
  while (cap && order.size() > cap) {
    pos.erase(order.front());
    order.pop_front();
  }
}

std::uint64_t MemoryHierarchy::shared_configuration_bytes() const {
  const std::uint64_t own = cfg_.shared_static_bytes + cfg_.shared_dynamic_bytes;
  if (cfg_.shared_configurations.empty()) return own;
  const std::uint64_t need = own + kReservedSharedPerCtaBytes;
  const std::uint64_t largest = cfg_.shared_configurations.back();
  const std::uint64_t warps = std::max<std::uint64_t>(1, cfg_.launch_warps_per_cta);
  std::uint64_t n_res = std::min({kMaxCtasPerSm, kMaxWarpsPerSm / warps, largest / need});
  if (n_res == 0) n_res = 1;   // a CTA that does not fit the largest carveout still runs alone
  const std::uint64_t demand = n_res * need;
  for (const auto c : cfg_.shared_configurations) if (c >= demand) return c;
  return largest;
}
std::uint64_t MemoryHierarchy::l1_capacity_bytes() const {
  const std::uint64_t used = cfg_.l1_reserved_bytes + shared_configuration_bytes();
  return cfg_.l1_data_array_bytes > used ? cfg_.l1_data_array_bytes - used : 0;
}
std::uint64_t MemoryHierarchy::l1_cap_lines() const { return l1_capacity_bytes() / cfg_.line_bytes; }

double MemoryHierarchy::port_acquire(Port& p, double now,
                                     std::uint64_t outstanding,
                                     double occupy,
                                     double completion_after,
                                     bool new_request) {
  double start = now > p.busy_until ? now : p.busy_until;
  if (new_request) {
    if (outstanding) {
      while (p.inflight.size() >= outstanding &&
             !p.inflight.empty() && p.inflight.front() <= start)
        p.inflight.pop_front();
      if (p.inflight.size() >= outstanding)
        start = p.inflight.front() > start ? p.inflight.front() : start;
      // the request's inflight slot lives until its last sector
      // returns (follower sectors only extend it via back() below)
      p.inflight.push_back(start + completion_after);
    }
  } else if (outstanding && !p.inflight.empty()) {
    // follower sector of the same request: occupies port bandwidth
    // only; the request's slot extends to this sector's completion
    p.inflight.back() =
        std::max(p.inflight.back(), start + completion_after);
  }
  p.busy_until = start + (occupy > 0 ? occupy : 1.0);
  return start;
}

double MemoryHierarchy::utilization(const SharedService& q, double when,
                                    double bpc) const {
  if (bpc <= 0.0) return 0.0;
  const std::uint64_t B = static_cast<std::uint64_t>(when / kServiceBucketCycles);
  const std::uint64_t lo = B > 32 ? B - 32 : 0;
  double sum = 0.0;
  for (std::uint64_t b = lo; b < B; ++b) {
    if (b < q.floor) sum += bpc * kServiceBucketCycles;
    else {
      const auto it = q.used.find(b);
      if (it != q.used.end()) sum += it->second;
    }
  }
  return sum / (bpc * kServiceBucketCycles * 32);
}

// piecewise-linear table lookup; the loaded-latency tables are keyed in
// permille of shared utilization (axis shared_utilization, int)
double MemoryHierarchy::eval_linear(
    const std::vector<std::pair<double, double>>& pts, double x) {
  if (pts.empty()) return 0.0;
  if (x <= pts.front().first) return pts.front().second;
  if (x >= pts.back().first) return pts.back().second;
  for (std::size_t i = 1; i < pts.size(); ++i) {
    if (x <= pts[i].first) {
      const auto& p1 = pts[i - 1];
      const auto& p2 = pts[i];
      const double t = (x - p1.first) / (p2.first - p1.first);
      return p1.second + t * (p2.second - p1.second);
    }
  }
  return pts.back().second;
}

double MemoryHierarchy::place_shared(SharedService& q, double when,
                                     double bytes, double bpc,
                                     double mark_bpc) {
  if (bpc <= 0.0 || bytes <= 0.0) return when;
  const double cap = bpc * kServiceBucketCycles;
  // mark threshold >= this call's cap by construction (callers pass
  // the largest bpc the queue is served at)
  const double mark_cap = mark_bpc * kServiceBucketCycles;
  std::uint64_t b = static_cast<std::uint64_t>(when) / kServiceBucketCycles;
  if (b < q.floor) b = q.floor;
  double remaining = bytes, end = when;
  for (;;) {
    // skip buckets already known full (take would be 0 for them)
    {
      std::uint64_t root = b;
      auto it = q.next_free.find(root);
      while (it != q.next_free.end()) {
        root = it->second;
        it = q.next_free.find(root);
      }
      while (b != root) {
        auto pit = q.next_free.find(b);
        if (pit == q.next_free.end()) break;
        const std::uint64_t nxt = pit->second;
        pit->second = root;
        b = nxt;
      }
      b = root;
    }
    double& u = q.used[b];
    const double take = std::min(remaining, cap - u);
    if (take > 0.0) {
      u += take;
      remaining -= take;
      end = static_cast<double>(b * kServiceBucketCycles) + u / bpc;
    }
    if (u >= mark_cap) q.next_free[b] = b + 1;
    if (remaining <= 0.0) break;
    if (u >= cap && b == q.floor) {
      q.used.erase(b);
      q.floor = b + 1;
    }
    ++b;
  }
  const double isolated = when + bytes / bpc;
  return std::max(end, isolated);  // fractional service-end
}

double MemoryHierarchy::service_shared(SharedService& q, double when,
                                       double bytes, double bpc,
                                       double mark_bpc) {
  if (bpc <= 0.0 || bytes <= 0.0) return when;
  const double end = place_shared(q, when, bytes, bpc, mark_bpc);
  const double isolated = when + bytes / bpc;
  if (end > isolated + 1e-9) {
    tele.shared_waits += 1;
    tele.wait_cycles += static_cast<std::uint64_t>(
        std::ceil(end - isolated));
  }
  return end;
}

double MemoryHierarchy::port_share_wait(std::uint32_t sm,
                                        PortUser who, double start,
                                        double occupy, bool book) {
  if (!cfg_.store_shares_fill_port || occupy <= 0.0) return 0.0;
  PortShare& ps = port_share_[sm];
  if (who == PortUser::shared_store) {
    // shared stores already serialize among themselves through the SM
    // acceptance frontier; on the port they only wait on fills and
    // book occupancy so that fills see them -- their own ledger never
    // caps their capacity (fill-free buckets are free at full rate).
    // book=false computes the same wait without writing the ledger or
    // the telemetry (kind 80: the SMSP frontier already books this
    // occupancy once at the SM level through kind 6)
    const double bucket = static_cast<double>(kServiceBucketCycles);
    std::vector<double>& own = ps.shared_store;
    const auto atf = [](const std::vector<double>& v, std::uint64_t i) {
      return i < v.size() ? v[i] : 0.0;
    };
    double t = start, remaining = occupy;
    for (std::uint64_t b = static_cast<std::uint64_t>(start) /
                            kServiceBucketCycles;; ++b) {
      if (book && b >= own.size()) own.resize(b + 1, 0.0);
      const double fill = atf(ps.fill_l2, b) + atf(ps.fill_dram, b);
      const double free_rate = std::max(0.0, bucket - fill) / bucket;
      const double t0 = std::max(t, static_cast<double>(b) * bucket);
      const double span = (b + 1) * bucket - t0;
      if (free_rate > 0.0) {
        const double take = std::min(remaining, span * free_rate);
        if (book)
          own[b] = std::min(bucket, own[b] + take);
        remaining -= take;
        t = t0 + take / free_rate;
      } else {
        t = (b + 1) * bucket;
      }
      if (remaining <= 1e-9) break;
    }
    const double end = std::max(t, start + occupy);
    const double extra = end - occupy - start;
    if (extra <= 1e-9) return 0.0;
    if (book) {
      ++tele.port_share_waits;
      tele.port_share_wait_cycles +=
          static_cast<std::uint64_t>(std::ceil(extra));
    }
    return extra;
  }
  std::vector<double>& own =
      who == PortUser::fill_l2 ? ps.fill_l2
      : who == PortUser::fill_dram ? ps.fill_dram : ps.store;
  const auto other = [&](std::uint64_t b) -> double {
    const auto at = [](const std::vector<double>& v, std::uint64_t i) {
      return i < v.size() ? v[i] : 0.0;
    };
    // fills are opposed by both store directions; each store
    // direction is opposed only by fills (stores do not wait on
    // each other)
    if (who == PortUser::store || who == PortUser::shared_store)
      return at(ps.fill_l2, b) + at(ps.fill_dram, b);
    return at(ps.store, b) + at(ps.shared_store, b);
  };
  const double bucket = static_cast<double>(kServiceBucketCycles);
  double remaining = occupy, end = start;
  for (std::uint64_t b = static_cast<std::uint64_t>(start) /
                            kServiceBucketCycles;; ++b) {
    if (b >= own.size()) own.resize(b + 1, 0.0);
    const double cap =
        std::max(0.0, bucket - other(b));
    const double take = std::min(remaining, cap - own[b]);
    if (take > 0.0) {
      own[b] += take;
      remaining -= take;
      end = static_cast<double>(b) * bucket + other(b) + own[b];
    }
    if (remaining <= 0.0) break;
  }
  end = std::max(end, start + occupy);   // first-placed-first-served end
  // alternation bound: with the two directions taking turns sector by
  // sector, this direction's sectors finish no later than its own
  // virtual finish time (previous finish or this start, + occupy/share)
  const double o_f = cfg_.l2_bytes_per_cycle > 0
      ? cfg_.sector_bytes / cfg_.l2_bytes_per_cycle : 0.0;
  const double o_s = cfg_.store_l2_bytes_per_cycle > 0
      ? cfg_.sector_bytes / cfg_.store_l2_bytes_per_cycle : 0.0;
  if (o_f > 0.0 && o_s > 0.0) {
    const double share =
        who == PortUser::store
            ? o_s / (o_s + o_f)
            : o_f / (o_s + o_f);
    double& vf = turn_finish_[sm][who == PortUser::store ? 1 : 0];
    vf = std::max(start, vf) + occupy / share;
    end = std::min(end, std::max(vf, start + occupy));
    vf = end;   // chain on the finish actually returned, not on the virtual one
  }
  const double extra = end - occupy - start;
  if (extra <= 1e-9) return 0.0;
  ++tele.port_share_waits;
  tele.port_share_wait_cycles +=
      static_cast<std::uint64_t>(std::ceil(extra));
  return extra;
}

double MemoryHierarchy::peek_shared(const SharedService& q, double when,
                                    double bytes, double bpc) const {
  if (bpc <= 0.0 || bytes <= 0.0) return when;
  const double cap = bpc * kServiceBucketCycles;
  std::uint64_t b = static_cast<std::uint64_t>(when) / kServiceBucketCycles;
  if (b < q.floor) b = q.floor;
  double remaining = bytes, end = when;
  for (;;) {
    auto nit = q.next_free.find(b);
    if (nit != q.next_free.end() && nit->second > b) { b = nit->second; continue; }
    const auto uit = q.used.find(b);
    const double u = uit == q.used.end() ? 0.0 : uit->second;
    const double take = std::min(remaining, cap - u);
    if (take > 0.0) {
      remaining -= take;
      end = static_cast<double>(b * kServiceBucketCycles) + (u + take) / bpc;
    }
    if (remaining <= 0.0) break;
    ++b;
  }
  return std::max(end, when + bytes / bpc);
}

void MemoryHierarchy::l2_touch(std::uint64_t line, L2Line& l,
                               std::uint64_t now) {
  l.last_access = now;
  if (l.in_lru) {
    l2_lru_.splice(l2_lru_.end(), l2_lru_, l.lru_it);
  } else {
    l2_lru_.push_back(line);
    l.lru_it = std::prev(l2_lru_.end());
    l.in_lru = true;
  }
}

void MemoryHierarchy::l2_account(L2Line& l) {
  l2_weight_total_ += l2_weight(l) - l.weight;
  l.weight = l2_weight(l);
}

void MemoryHierarchy::l2_evict(std::uint64_t line, std::uint64_t now) {
  auto it = l2_.find(line);
  if (it == l2_.end()) return;
  auto& l = it->second;
  l2_weight_total_ -= l.weight;
  l2_lru_.erase(l.lru_it);
  if (l.dirty) {
    // write-back: whole line through the shared DRAM write queue
    tele.dram_bytes += cfg_.line_bytes;
    service_shared(shared_dram_, now, cfg_.line_bytes,
                   dram_bpc(true),
                   std::max(cfg_.dram_read_bpc, cfg_.dram_write_bpc));
    ++tele.dirty_writebacks;
  }
  ++tele.l2_evictions;
  l2_.erase(it);
}

void MemoryHierarchy::l2_ensure_capacity(std::uint64_t self,
                                         std::uint64_t now) {
  const std::uint64_t cap = cfg_.l2_resident_lines
      ? cfg_.l2_resident_lines : cfg_.l2_resident_lines_lower_bound;
  if (!cap) return;
  while (l2_weight_total_ > cap && l2_lru_.size() > 1 &&
         l2_lru_.front() != self)
    l2_evict(l2_lru_.front(), now);
}

std::uint64_t MemoryHierarchy::load(std::uint32_t sm, std::uint64_t now,
                                    const MemorySector* sectors,
                                    std::size_t n, bool use_l1,
                                    HitLevel* level,
                                    std::uint32_t* l1o,
                                    std::uint32_t* l2o,
                                    std::uint32_t* dramo,
                                    std::uint64_t* accept,
                                    std::uint64_t* shared_free,
                                    bool tma, bool l1_allocate,
                                    bool ordered, bool async_copy,
                                    std::size_t phantom_sectors,
                                    bool local) {
  if (cfg_.store_orders_loads && ordered) {
    // in-order SM load/store unit: this load starts only after the
    // store port has drained the store sectors issued before it
    const std::uint64_t drain =
        static_cast<std::uint64_t>(std::ceil(port_sl2_[sm].busy_until));
    if (drain > now) {
      ++tele.ordered_load_waits;
      tele.ordered_load_wait_cycles += drain - now;
      now = drain;
    }
  }
  std::uint64_t acc = now;
  // return the request would have had with the GPU-wide shared L2 /
  // DRAM services free: every sector's port start + level latency,
  // skipping the service_shared stage (see the header)
  std::uint64_t shared_free_last = now;
  bool acc_set = false;
  std::uint64_t last = now;
  std::uint32_t n_l1 = 0, n_l2 = 0, n_dram = 0;
  // one warp-level request per port per load() call: the first sector
  // routed to the L2 port / DRAM port takes the inflight slot, later
  // sectors only extend it (see port_acquire)
  bool l2_new = true, dram_new = true;
  // the request traverses the L1 pipeline on hits and misses alike:
  // when the l1.sector_bytes_per_cycle leaf is live every sector
  // acquires the per-SM L1 port (same pattern as the L2 port below),
  // and the read barrier releases at the first sector's L1 start.
  // That is the only acceptance point reported through `accept`: the
  // L2 / DRAM ports below are far-side queues whose congestion under
  // full-GPU load must not hold the warp's register read barrier
  // (2026-11-08: cublas big_square epilogue UIADD3 waited ~400
  // cycles per UTMASTG on the store-L2 port, CUF-DSL-RMSNORM 1.37)
  const bool l1_port_on = use_l1 && l1_modeled() &&
                          cfg_.l1_bytes_per_cycle > 0;
  bool l1_new = true;
  // summed per-sector port occupation of this warp-level request, per
  // port; topped up to the per-request floor after the loop
  double occ_l1 = 0.0, occ_l2 = 0.0, occ_dram = 0.0;
  // loaded-latency extra: one utilization sample per request per
  // level, taken at this request's first shared-service submission
  // (before its bytes are added); -1 = not sampled
  double u_dram = -1.0, u_l2 = -1.0;
  // memory.load_port line_granular: on each load port the first
  // sector a request routes from a line pays line_bytes / bpc and
  // later same-line sectors pay 0 (the port acquire still runs so
  // start / inflight accounting is unchanged); TMA-path requests keep
  // the per-sector charge
  const bool line_gran =
      cfg_.load_port_line_granular && !(tma && cfg_.tma_sector_bpc > 0);
  std::vector<std::uint64_t> port_lines_l1, port_lines_l2,
      port_lines_dram;
  const auto line_occ = [this](std::vector<std::uint64_t>& lines,
                               std::uint64_t line, double bpc,
                               double fallback) {
    if (std::find(lines.begin(), lines.end(), line) != lines.end())
      return 0.0;
    lines.push_back(line);
    return bpc > 0 ? static_cast<double>(cfg_.line_bytes) / bpc
                   : fallback;
  };
  // memory.load_port lane_order (gather_hit protocol): under
  // line-granular charging a lane load occupies one line slot per
  // line switch between consecutive active lanes -- lanes in lane
  // order, each on the line of the lowest-addressed sector whose
  // lane_mask holds it -- instead of one per distinct line.  The
  // difference over distinct lines is added to the first sector's
  // port occupation; a request whose sectors carry no lane masks
  // (TMA / bulk) keeps the distinct-line count.
  const auto slot_cost = [this](double bpc, double fallback) {
    return bpc > 0 ? static_cast<double>(cfg_.line_bytes) / bpc
                   : fallback;
  };
  std::size_t lane_extra = 0;
  if (line_gran && cfg_.load_port_lane_order && cfg_.line_bytes) {
    bool masked = false;
    for (std::size_t i = 0; i < n; ++i)
      if (sectors[i].lane_mask) { masked = true; break; }
    if (masked) {
      std::vector<std::uint64_t> dlines;
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t l = sectors[i].address / cfg_.line_bytes;
        if (std::find(dlines.begin(), dlines.end(), l) == dlines.end())
          dlines.push_back(l);
      }
      std::size_t switches = 0;
      std::uint64_t prev = 0;
      bool first = true;
      for (std::uint32_t lane = 0; lane < 32; ++lane) {
        const MemorySector* best = nullptr;
        for (std::size_t i = 0; i < n; ++i)
          if ((sectors[i].lane_mask >> lane) & 1u &&
              (!best || sectors[i].address < best->address))
            best = &sectors[i];
        if (!best) continue;
        const std::uint64_t l = best->address / cfg_.line_bytes;
        if (first || l != prev) { ++switches; prev = l; first = false; }
      }
      if (switches > dlines.size())
        lane_extra = switches - dlines.size();
    }
  }
  // merge domain key space for this request kind: gpc for the kinds
  // merge_domain_gpc selects, the pair domain for the rest
  const bool gpc_kind =
      cfg_.merge_domain_gpc != 0 &&
      (tma || (cfg_.merge_domain_gpc == 2 && async_copy));
  // a read opened by an async copy keeps the shorter measured window;
  // whether a follower merges is judged against the opener's window
  const double merge_window =
      (async_copy && cfg_.merge_window_async > 0)
          ? cfg_.merge_window_async
          : cfg_.merge_window;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t sector = sectors[i].address;
    ++tele.l1_sectors;
    tele.l1_bytes += sectors[i].byte_mask ? __builtin_popcount(
        sectors[i].byte_mask) : cfg_.sector_bytes;
    const std::uint64_t line = sector / cfg_.line_bytes;
    // L1 port acquisition (every sector, hits and misses); downstream
    // stages use max(now, l1 port start)
    double now_eff = now;
    if (l1_port_on) {
      double occupy =
          line_gran ? line_occ(port_lines_l1, line,
                               cfg_.l1_bytes_per_cycle,
                               cfg_.sector_bytes / cfg_.l1_bytes_per_cycle)
                    : cfg_.sector_bytes / cfg_.l1_bytes_per_cycle;
      if (i == 0 && lane_extra && use_l1 && l1_modeled() &&
          l1_sets_[sm].has(sector, now))
        occupy += lane_extra *
                  slot_cost(cfg_.l1_bytes_per_cycle,
                            cfg_.sector_bytes / cfg_.l1_bytes_per_cycle);
      const double busy0 = port_l1_[sm].busy_until;
      const double when = port_acquire(port_l1_[sm], now,
                                       cfg_.l1_outstanding, occupy,
                                       cfg_.l1_latency, l1_new);
      if (occupy == 0.0) port_l1_[sm].busy_until = busy0;
      l1_new = false;
      if (!acc_set) {
        acc = static_cast<std::uint64_t>(std::ceil(when));
        acc_set = true;
      }
      occ_l1 += occupy;
      now_eff = std::max(now_eff, when);
      if (dbg_)
        fprintf(stderr,
                "DBG_MEM sm=%u i=%zu l1 port busy=%.0f->%.0f "
                "start=%.0f inflight=%zu\n",
                sm, i, busy0, port_l1_[sm].busy_until, when,
                port_l1_[sm].inflight.size());
      tele.l1_port_wait += when - now;
    }
    // L1 hit? sector-granular: hits only once the fill completed.
    // Without a memory.l1.residency data array the SM has no modeled
    // L1 (every sector goes to the merge window / L2).
    if (use_l1 && l1_modeled()) {
      auto& set = l1_sets_[sm];
      if (set.cap == 0 && cfg_.line_bytes)
        set.cap = l1_cap_lines() * (cfg_.line_bytes /
                   (cfg_.sector_bytes ? cfg_.sector_bytes : 32));
      if (set.has(sector, now)) {
        set.touch(sector);
        ++n_l1;
        const double hit_lat =
            local && cfg_.local_hit_latency > 0
                ? cfg_.local_hit_latency : cfg_.l1_latency;
        last = std::max(last, static_cast<std::uint64_t>(
            std::ceil(now_eff)) + static_cast<std::uint64_t>(
            hit_lat));
        shared_free_last = std::max(shared_free_last, last);
        continue;
      }
    }
    ++tele.l1_misses;
    // local-space miss (load_after_copies v3): the line is refilled
    // from L2 -- the request pays the ordinary L2-hit path (SM port,
    // shared read services, L2 hit latency) but consults no residency
    // table, inserts nothing into L2 and joins no merge window; the
    // arriving line then takes its place in the SM's L1.
    if (local) {
      ++n_l2;
      ++tele.l2_sectors;
      const double bpc_l2 = cfg_.l2_bytes_per_cycle;
      double occupy =
          line_gran ? line_occ(port_lines_l2, line, bpc_l2,
                               bpc_l2 > 0 ? cfg_.sector_bytes / bpc_l2
                                          : std::max(
                                                1.0,
                                                cfg_.l2_issue_interval))
                    : (bpc_l2 > 0 ? cfg_.sector_bytes / bpc_l2
                                  : std::max(1.0,
                                             cfg_.l2_issue_interval));
      if (i == 0 && lane_extra)
        occupy += lane_extra *
                  slot_cost(bpc_l2, std::max(1.0,
                                             cfg_.l2_issue_interval));
      const double busy0 = port_l2_[sm].busy_until;
      double when = port_acquire(port_l2_[sm], now_eff,
                                 cfg_.l2_outstanding, occupy,
                                 cfg_.l2_latency, l2_new);
      if (occupy == 0.0) port_l2_[sm].busy_until = busy0;
      shared_free_last = std::max(shared_free_last,
          static_cast<std::uint64_t>(std::ceil(when)) +
              static_cast<std::uint64_t>(cfg_.l2_latency));
      when += port_share_wait(sm, PortUser::fill_l2, when, occupy);
      l2_new = false;
      occ_l2 += occupy;
      if (cfg_.l2_read_gpc_bpc > 0 &&
          sm < cfg_.sm_gpc.size() &&
          cfg_.sm_gpc[sm] != 0xFFFFFFFFu)
        when = service_shared(gpc_l2_read_[cfg_.sm_gpc[sm]], when,
                              cfg_.sector_bytes,
                              cfg_.l2_read_gpc_bpc,
                              cfg_.l2_read_gpc_bpc);
      when = service_shared(shared_l2_read_, when,
                            cfg_.sector_bytes, cfg_.l2_read_bpc,
                            cfg_.l2_read_bpc);
      const std::uint64_t completion =
          static_cast<std::uint64_t>(std::ceil(when)) +
          static_cast<std::uint64_t>(cfg_.l2_latency);
      last = std::max(last, completion);
      l1_insert(sm, sector, completion);
      continue;
    }
    // merge window: coalesce with a recent same-sector request
    auto m = merge_last_.find(merge_key_(sm, sector, gpc_kind));
    if (merge_window > 0 && m != merge_last_.end() &&
        m->second.expiry > now &&
        (cfg_.merge_max_requests == 0 ||
         m->second.count < cfg_.merge_max_requests)) {
      ++m->second.count;
      ++tele.merge_merges;
      double delivered = m->second.completion;
      if (cfg_.merge_bytes_per_cycle > 0) {
        merge_delivery_busy_ =
            std::max(merge_delivery_busy_, now_eff) +
            cfg_.sector_bytes / cfg_.merge_bytes_per_cycle;
        delivered = std::max(delivered, merge_delivery_busy_);
      }
      // a merged follower is still this SM's own request to the slice:
      // it cannot complete before its own L2 hit path (issue + hit
      // latency, + the far-side extra when the reader side holds no
      // replica), whatever the leader's completion was
      double floor_cycles = cfg_.l2_latency;
      if (auto li = l2_.find(line); li != l2_.end()) {
        const std::uint32_t home =
            static_cast<std::uint32_t>(line % cfg_.pool_count);
        const std::uint32_t side = sm_side(sm);
        const std::uint32_t bit = sector_bit(sector);
        if (!(li->second.sectors[side] & bit) &&
            (li->second.sectors[home] & bit) && side != home)
          floor_cycles += cfg_.far_hit_cycles;
      }
      delivered = std::max(delivered, now_eff + floor_cycles);
      last = std::max(last,
                      static_cast<std::uint64_t>(std::ceil(delivered)));
      // a merged follower never enters the shared services: its
      // delivery is part of the shared-free return too (otherwise a
      // fully merged TMA would count the leader's whole latency as
      // queueing: single-cluster cutlass_stream_k_splitk_05 +37 %)
      shared_free_last = std::max(shared_free_last, last);
      continue;
    }
    // L2 (two partitions + replica)
    const std::uint32_t home =
        static_cast<std::uint32_t>(line % cfg_.pool_count);
    const std::uint32_t side = sm_side(sm);
    auto& l2 = l2_[line];
    if (l2.tag == 0) l2.tag = line;
    const std::uint32_t si = sector_index(sector);
    const std::uint32_t bit = 1u << si;
    const bool hit_home = (l2.sectors[home] & bit) != 0;
    const bool hit_local = (l2.sectors[side] & bit) != 0 || hit_home;
    std::uint64_t completion = now;
    // when this sector leaves the shared read service (issue by
    // default; used by merge_window_async_from_service)
    double served = now;
    if (hit_local) {
      ++n_l2;
      ++tele.l2_sectors;
      // far hit only when the reader side holds no replica; the hit
      // then installs the reader-side replica (replicate_read_lines)
      const bool far = !(l2.sectors[side] & bit) && hit_home &&
                       side != home;
      const double extra = far ? cfg_.far_hit_cycles : 0.0;
      // the array the data comes from: the side replica when present,
      // else home; its fill may still be in flight
      const std::uint64_t ready = si < kSectorsPerLine
          ? l2.ready[(l2.sectors[side] & bit) ? side : home][si] : 0;
      if (far) {
        ++tele.l2_far_hits;
        if (cfg_.replicate_read_lines) {
          l2.sectors[side] |= bit;
          ++tele.l2_replica_fills;
        }
      }
      const double bpc_l2 =
          (tma && cfg_.tma_sector_bpc > 0) ? cfg_.tma_sector_bpc
                                          : cfg_.l2_bytes_per_cycle;
      double occupy =
          line_gran ? line_occ(port_lines_l2, line, bpc_l2,
                               bpc_l2 > 0 ? cfg_.sector_bytes / bpc_l2
                                          : std::max(1.0, cfg_.l2_issue_interval))
                    : (bpc_l2 > 0 ? cfg_.sector_bytes / bpc_l2
                                  : std::max(1.0, cfg_.l2_issue_interval));
      if (i == 0 && lane_extra)
        occupy += lane_extra *
                  slot_cost(bpc_l2, std::max(1.0, cfg_.l2_issue_interval));
      const double busy0 = port_l2_[sm].busy_until;
      double when = port_acquire(port_l2_[sm], now_eff,
                                 cfg_.l2_outstanding, occupy,
                                 cfg_.l2_latency + extra, l2_new);
      if (occupy == 0.0) port_l2_[sm].busy_until = busy0;
      shared_free_last = std::max(shared_free_last,
          static_cast<std::uint64_t>(std::ceil(when)) +
              static_cast<std::uint64_t>(cfg_.l2_latency + extra));
      when += port_share_wait(sm, PortUser::fill_l2, when, occupy);
      l2_new = false;
      occ_l2 += occupy;
      if (dbg_)
        fprintf(stderr,
                "DBG_MEM sm=%u i=%zu l2hit far=%d port busy=%.0f->%.0f "
                "start=%.0f inflight=%zu\n",
                sm, i, (int)far, busy0, port_l2_[sm].busy_until, when,
                port_l2_[sm].inflight.size());
      if (!cfg_.l2_loaded_extra.empty() && u_l2 < 0.0)
        u_l2 = utilization(shared_l2_read_, when, cfg_.l2_read_bpc);
      if (cfg_.l2_read_gpc_bpc > 0 &&
          sm < cfg_.sm_gpc.size() &&
          cfg_.sm_gpc[sm] != 0xFFFFFFFFu)
        when = service_shared(gpc_l2_read_[cfg_.sm_gpc[sm]], when,
                              cfg_.sector_bytes,
                              cfg_.l2_read_gpc_bpc,
                              cfg_.l2_read_gpc_bpc);
      when = service_shared(shared_l2_read_, when,
                            cfg_.sector_bytes, cfg_.l2_read_bpc,
                            cfg_.l2_read_bpc);
      served = when;
      if (dbg_)
        fprintf(stderr, "DBG_MEM sm=%u i=%zu shared_l2_read end=%.0f\n",
                sm, i, when);
      completion = static_cast<std::uint64_t>(std::ceil(when)) +
          static_cast<std::uint64_t>(cfg_.l2_latency + extra);
      if (u_l2 >= 0.0) {
        const std::uint64_t ex = static_cast<std::uint64_t>(std::llround(
            eval_linear(cfg_.l2_loaded_extra, u_l2 * 1000.0)));
        completion += ex;
        tele.loaded_extra_l2_cycles += ex;
        ++tele.loaded_extra_l2_count;
      }
      // fill in flight: held at the slice until the data is there, then
      // returned like the request that fetched it (the merge window
      // already gives followers inside its 240 cycles the leader's
      // completion; this covers the rest of the fill latency)
      if (ready > completion) {
        // the recorded fill may still be queued at its own SM's DRAM
        // port (the leader was first in simulation order, not at the
        // slice): this request cannot be held past what its own miss
        // would have taken -- it would reach the slice first and start
        // the fill itself (flashinfer decode: 11.6 k such hits held
        // 3.4 k cycles each, 0.99 -> 1.17, full-gpu-chain-report.md §18)
        // the hypothetical miss must pay everything a real miss pays
        // at this cycle: this SM's DRAM port backlog (peek only), the
        // shared service queue, the base latency and the load-dependent
        // extra -- otherwise followers complete before their leader
        // (full-gpu-chain-report §27)
        const double miss_start =
            when > port_dram_[sm].busy_until ? when
                                            : port_dram_[sm].busy_until;
        const double bpc = dram_bpc(false);
        const double extra =
            cfg_.dram_loaded_extra.empty()
                ? 0.0
                : eval_linear(cfg_.dram_loaded_extra,
                              utilization(shared_dram_, miss_start, bpc) *
                                  1000.0);
        const std::uint64_t own_miss = static_cast<std::uint64_t>(
            std::ceil(peek_shared(shared_dram_, miss_start,
                                  cfg_.sector_bytes, bpc))) +
            static_cast<std::uint64_t>(dram_latency_at(sm, sector)) +
            static_cast<std::uint64_t>(std::llround(extra));
        const std::uint64_t held = std::min(ready, own_miss);
        if (held > completion) {
          ++tele.l2_inflight_hits;
          tele.l2_inflight_wait += held - completion;
          completion = held;
        }
        if (own_miss < ready && si < kSectorsPerLine) {
          // this request becomes the effective fill: data is there when
          // it returns
          l2.ready[home][si] = std::min(l2.ready[home][si], completion);
          l2.ready[side][si] = std::min(l2.ready[side][si], completion);
        }
      }
      if (far && cfg_.replicate_read_lines && si < kSectorsPerLine)
        l2.ready[side][si] = completion;
      last = std::max(last, completion);
    } else {
      ++tele.l2_misses;
      ++n_dram;
      const double bpc_dram =
          (tma && cfg_.tma_sector_bpc > 0) ? cfg_.tma_sector_bpc
                                          : cfg_.dram_bytes_per_cycle;
      double occupy =
          line_gran ? line_occ(port_lines_dram, line, bpc_dram,
                               bpc_dram > 0 ? cfg_.sector_bytes / bpc_dram
                                            : std::max(1.0, cfg_.dram_issue_interval))
                    : (bpc_dram > 0 ? cfg_.sector_bytes / bpc_dram
                                    : std::max(1.0, cfg_.dram_issue_interval));
      if (i == 0 && lane_extra)
        occupy += lane_extra *
                  slot_cost(bpc_dram,
                            std::max(1.0, cfg_.dram_issue_interval));
      // level latencies are load-to-use totals for a hit at that level
      // (memlat convention): a DRAM miss completes at start +
      // dram_latency_at (near page / far page of this SM's side)
      const double dram_lat = dram_latency_at(sm, sector);
      const double dbusy0 = port_dram_[sm].busy_until;
      double when = port_acquire(port_dram_[sm], now_eff,
                                 cfg_.dram_outstanding, occupy,
                                 dram_lat, dram_new);
      if (occupy == 0.0) port_dram_[sm].busy_until = dbusy0;
      shared_free_last = std::max(shared_free_last,
          static_cast<std::uint64_t>(std::ceil(when)) +
              static_cast<std::uint64_t>(dram_lat));
      when += port_share_wait(sm, PortUser::fill_dram, when, occupy);
      dram_new = false;
      occ_dram += occupy;
      if (dbg_)
        fprintf(stderr,
                "DBG_MEM sm=%u i=%zu dram port busy=%.0f->%.0f start=%.0f\n",
                sm, i, dbusy0, port_dram_[sm].busy_until, when);
      if (!cfg_.dram_loaded_extra.empty() && u_dram < 0.0)
        u_dram = utilization(shared_dram_, when, dram_bpc(false));
      when = service_shared(shared_dram_, when,
                            cfg_.sector_bytes, dram_bpc(false),
                            std::max(cfg_.dram_read_bpc,
                                     cfg_.dram_write_bpc));
      served = when;
      completion = static_cast<std::uint64_t>(std::ceil(when)) +
          static_cast<std::uint64_t>(dram_lat);
      if (u_dram >= 0.0) {
        const std::uint64_t ex = static_cast<std::uint64_t>(std::llround(
            eval_linear(cfg_.dram_loaded_extra, u_dram * 1000.0)));
        completion += ex;
        tele.loaded_extra_dram_cycles += ex;
        ++tele.loaded_extra_dram_count;
      }
      last = std::max(last, completion);
      ++tele.dram_sectors;
      tele.dram_bytes += cfg_.sector_bytes;
      // fill: home side always; reader side per replicate_read_lines;
      // the data is in the arrays when this request completes
      l2.sectors[home] |= bit;
      if (si < kSectorsPerLine) l2.ready[home][si] = completion;
      if (cfg_.replicate_read_lines && side != home) {
        l2.sectors[side] |= bit;
        if (si < kSectorsPerLine) l2.ready[side][si] = completion;
        ++tele.l2_replica_fills;
      }
      if (l2.dirty && cfg_.dirty_lines_near_both_sides)
        ++tele.dirty_writebacks;
    }
    l2_touch(line, l2, last);
    l2_account(l2);
    l2_ensure_capacity(line, now);
    if (merge_window > 0) {
      // with merge_window_async_from_service an async copy's window
      // starts when its sector leaves the shared read service, so a
      // read queued at the read limit stays joinable while it waits
      const bool from_service =
          async_copy && cfg_.merge_window_async > 0 &&
          cfg_.merge_window_async_from_service;
      const double expiry =
          from_service
              ? std::max<double>(now, served) +
                    cfg_.merge_window_async
              : now + merge_window;
      merge_last_[merge_key_(sm, sector, gpc_kind)] =
          MergeEntry{expiry, completion, 1};
    }
    if (l1_modeled() && (use_l1 || l1_allocate)) {
      auto& set = l1_sets_[sm];
      if (set.cap == 0 && cfg_.line_bytes)
        set.cap = l1_cap_lines() * (cfg_.line_bytes /
                   (cfg_.sector_bytes ? cfg_.sector_bytes : 32));
      l1_sets_[sm].insert(sector, completion);
    }
  }
  // tma_oob v2: phantom sectors are the out-of-bounds part of a tiled
  // TMA box (hardware delivers the whole box).  They touch no cache
  // state -- no L2 table, no fill, no merge entry, no DRAM, no
  // cold_fraction bookkeeping -- but take the timing path of an
  // unmerged L2-hit sector: per-SM L1 port (when live), per-SM L2
  // port, per-GPC and GPU-wide shared read services, and the same
  // completion expression feeding `last` / `shared_free_last`.
  for (std::size_t i = 0; i < phantom_sectors; ++i) {
    double now_eff = now;
    if (l1_port_on) {
      const double occupy = cfg_.sector_bytes / cfg_.l1_bytes_per_cycle;
      const double when = port_acquire(port_l1_[sm], now,
                                       cfg_.l1_outstanding, occupy,
                                       cfg_.l1_latency, l1_new);
      l1_new = false;
      if (!acc_set) {
        acc = static_cast<std::uint64_t>(std::ceil(when));
        acc_set = true;
      }
      occ_l1 += occupy;
      now_eff = std::max(now_eff, when);
      tele.l1_port_wait += when - now;
    }
    const double bpc_l2 =
        (tma && cfg_.tma_sector_bpc > 0) ? cfg_.tma_sector_bpc
                                        : cfg_.l2_bytes_per_cycle;
    const double occupy =
        bpc_l2 > 0 ? cfg_.sector_bytes / bpc_l2
                   : std::max(1.0, cfg_.l2_issue_interval);
    double when = port_acquire(port_l2_[sm], now_eff,
                               cfg_.l2_outstanding, occupy,
                               cfg_.l2_latency, l2_new);
    shared_free_last = std::max(shared_free_last,
        static_cast<std::uint64_t>(std::ceil(when)) +
            static_cast<std::uint64_t>(cfg_.l2_latency));
    when += port_share_wait(sm, PortUser::fill_l2, when, occupy);
    l2_new = false;
    occ_l2 += occupy;
    if (!cfg_.l2_loaded_extra.empty() && u_l2 < 0.0)
      u_l2 = utilization(shared_l2_read_, when, cfg_.l2_read_bpc);
    if (cfg_.l2_read_gpc_bpc > 0 &&
        sm < cfg_.sm_gpc.size() &&
        cfg_.sm_gpc[sm] != 0xFFFFFFFFu)
      when = service_shared(gpc_l2_read_[cfg_.sm_gpc[sm]], when,
                            cfg_.sector_bytes,
                            cfg_.l2_read_gpc_bpc,
                            cfg_.l2_read_gpc_bpc);
    when = service_shared(shared_l2_read_, when,
                          cfg_.sector_bytes, cfg_.l2_read_bpc,
                          cfg_.l2_read_bpc);
    std::uint64_t completion =
        static_cast<std::uint64_t>(std::ceil(when)) +
        static_cast<std::uint64_t>(cfg_.l2_latency);
    if (u_l2 >= 0.0) {
      const std::uint64_t ex = static_cast<std::uint64_t>(std::llround(
          eval_linear(cfg_.l2_loaded_extra, u_l2 * 1000.0)));
      completion += ex;
      tele.loaded_extra_l2_cycles += ex;
      ++tele.loaded_extra_l2_count;
    }
    last = std::max(last, completion);
  }
  tele.phantom_sectors += phantom_sectors;
  // a port accepts at most one warp-level request per cycle (b300 memq
  // measured ~1.0 request/cycle/SM at the L2 port): when this request's
  // sectors sum to less than the issue-interval floor, the port keeps
  // the difference -- only the shortfall is charged, so a multi-sector
  // request pays exactly its summed sector occupation
  const double l1_floor = std::max(1.0, cfg_.l1_issue_interval);
  if (occ_l1 > 0.0 && occ_l1 < l1_floor)
    port_l1_[sm].busy_until += l1_floor - occ_l1;
  const double l2_floor = std::max(1.0, cfg_.l2_issue_interval);
  if (occ_l2 > 0.0 && occ_l2 < l2_floor)
    port_l2_[sm].busy_until += l2_floor - occ_l2;
  const double dram_floor = std::max(1.0, cfg_.dram_issue_interval);
  if (occ_dram > 0.0 && occ_dram < dram_floor)
    port_dram_[sm].busy_until += dram_floor - occ_dram;
  if (l1o) *l1o = n_l1;
  if (l2o) *l2o = n_l2;
  if (dramo) *dramo = n_dram;
  if (level)
    *level = n_dram ? HitLevel::dram
                    : (n_l2 ? HitLevel::l2 : HitLevel::l1);
  if (accept) *accept = acc_set ? acc : now;
  if (shared_free) *shared_free = std::min(shared_free_last, last);
  return last;
}

std::uint64_t MemoryHierarchy::store(std::uint32_t sm, std::uint64_t now,
                                     const MemorySector* sectors,
                                     std::size_t n, bool atomic,
                                     std::uint64_t address0,
                                     std::uint64_t* fill_done_out,
                                     double cost_ratio, bool lane_store) {
  double when = static_cast<double>(now);
  if (atomic) {
    ++tele.atomics;
    // same-address serialization via same_addr_service
    when += cfg_.atomic_same_addr_service;
    (void)address0;
  }
  // every sector takes its own turn on this SM's store port (the same
  // per-sector occupancy the load path charges); the shared L2-write
  // queue is chained sector after sector
  // a reduce (storage.tma.reduce, kind 81) occupies the port and the
  // shared write service cost_ratio times as long per sector
  const double occupy = (cfg_.store_l2_bytes_per_cycle > 0
      ? cfg_.sector_bytes / cfg_.store_l2_bytes_per_cycle
      : std::max(1.0, cfg_.store_l2_issue_interval)) * cost_ratio;
  // store_flood: for lane stores, every touched 128-B line holding a
  // partially written granule occupies the port
  // partial_line_cycles longer, charged on the line's first sector of
  // this request.  Granule = partial_granule_bytes; a granule is
  // partially written when the request writes 0 < k < granule bytes
  // of it (byte_mask chunk neither empty nor full)
  std::vector<double> extra;
  const bool partial =
      lane_store && cfg_.store_l2_partial_line_cycles > 0 &&
      cfg_.store_l2_partial_granule_bytes > 0;
  if (partial) {
    const std::uint64_t g = cfg_.store_l2_partial_granule_bytes;
    const std::uint32_t gmask =
        g >= 32 ? 0xFFFFFFFFu : ((1u << g) - 1u);
    std::unordered_map<std::uint64_t, std::pair<std::size_t, bool>>
        lines;
    extra.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t line = sectors[i].address / cfg_.line_bytes;
      auto& e = lines.try_emplace(line, std::pair{i, false})
                    .first->second;
      const std::uint32_t m = sectors[i].byte_mask;
      bool part = false;
      if (g >= 32) {
        part = (m != 0xFFFFFFFFu);
      } else {
        for (std::uint32_t b = 0; b < 32; b += g) {
          const std::uint32_t gm = (m >> b) & gmask;
          if (gm != 0 && gm != gmask) { part = true; break; }
        }
      }
      e.second = e.second || part;
    }
    for (const auto& kv : lines)
      if (kv.second.second)
        extra[kv.second.first] = cfg_.store_l2_partial_line_cycles;
  }
  double accept = when;
  double occ = 0.0;
  double x_max = 0.0;
  double port_end = 0.0;
  double u_af_dram = -1.0;
  std::uint64_t fill_done = 0;
  for (std::size_t i = 0; i < n; ++i) {
    tele.l2_bytes += cfg_.sector_bytes;
    // dirty presence counts the home side only unless the A/B
    // field is on (spec §3 correction).
    const std::uint64_t line = sectors[i].address / cfg_.line_bytes;
    const std::uint32_t home =
        static_cast<std::uint32_t>(line % cfg_.pool_count);
    auto& l2 = l2_[line];
    if (l2.tag == 0) l2.tag = line;
    const std::uint32_t si = sector_index(sectors[i].address);
    const bool present = (l2.sectors[home] & (1u << si)) != 0;
    if (atomic && !present) {
      // an atomic / reduction is a read-modify-write at the L2 slice:
      // a sector not resident there is first fetched from DRAM (ncu:
      // lts__d_sectors_fill_device == dram__bytes_read on reduce-heavy
      // kernels, full-gpu-chain-report.md §15.2).  The fetch is the
      // slice's own request -- no SM port -- through the shared DRAM
      // read service; the reduced value is in the array when it lands
      if (!cfg_.dram_loaded_extra.empty() && u_af_dram < 0.0)
        u_af_dram = utilization(shared_dram_, when, dram_bpc(false));
      const double w = service_shared(shared_dram_, when, cfg_.sector_bytes,
                                      dram_bpc(false),
                                      std::max(cfg_.dram_read_bpc,
                                               cfg_.dram_write_bpc));
      const std::uint64_t done = static_cast<std::uint64_t>(std::ceil(w)) +
          static_cast<std::uint64_t>(cfg_.dram_latency) +
          (u_af_dram >= 0.0
               ? static_cast<std::uint64_t>(std::llround(
                     eval_linear(cfg_.dram_loaded_extra, u_af_dram * 1000.0)))
               : 0);
      if (u_af_dram >= 0.0) {
        tele.loaded_extra_dram_cycles += static_cast<std::uint64_t>(
            std::llround(eval_linear(cfg_.dram_loaded_extra, u_af_dram * 1000.0)));
        ++tele.loaded_extra_dram_count;
      }
      ++tele.l2_misses;
      ++tele.dram_sectors;
      tele.dram_bytes += cfg_.sector_bytes;
      ++tele.atomic_fills;
      if (si < kSectorsPerLine) l2.ready[home][si] = done;
      fill_done = std::max(fill_done, done);
    } else if (si < kSectorsPerLine && !atomic) {
      // written data is in the home array once accepted (a store onto a
      // sector whose fill is in flight does not wait for the fill)
      l2.ready[home][si] = 0;
    }
    l2.sectors[home] |= 1u << si;
    l2.dirty = true;
    l2_touch(line, l2, static_cast<std::uint64_t>(when));
    l2_account(l2);
    l2_ensure_capacity(line, static_cast<std::uint64_t>(when));
    if (cfg_.l2_write_gpc_bpc > 0 &&
        sm < cfg_.sm_gpc.size() &&
        cfg_.sm_gpc[sm] != 0xFFFFFFFFu)
      when = service_shared(gpc_l2_write_[cfg_.sm_gpc[sm]], when,
                            cfg_.sector_bytes * cost_ratio,
                            cfg_.l2_write_gpc_bpc,
                            cfg_.l2_write_gpc_bpc);
    // fractional accumulation through the shared L2-write queue
    when = service_shared(shared_l2_write_, when,
                          cfg_.sector_bytes * cost_ratio, cfg_.l2_write_bpc,
                          cfg_.l2_write_bpc);
    const double occ_i =
        occupy + (extra.empty() ? 0.0 : extra[i]);
    const double start = port_acquire(
        port_sl2_[sm], when, cfg_.store_l2_outstanding, occ_i,
        cfg_.store_l2_latency, i == 0);
    const double x = port_share_wait(sm, PortUser::store, start, occ_i);
    accept = std::max(accept, start + x);
    x_max = std::max(x_max, x);
    port_end = std::max(port_end, start + x + occ_i);
    occ += occ_i;
  }
  // same per-request floor as the load path, on lane stores only:
  // at most one warp-level lane-store request per issue_interval on
  // the store port (other store kinds keep the 1-cycle floor)
  const double s_floor =
      lane_store ? std::max(1.0, cfg_.store_l2_issue_interval) : 1.0;
  if (occ > 0.0 && occ < s_floor)
    port_sl2_[sm].busy_until += s_floor - occ;
  // rule S: the request drains from this SM's store buffer when its
  // last sector has left the shared L2-write service and this SM's
  // store port (whichever is later)
  if (!atomic)
    store_drain_[sm].push_back(static_cast<std::uint64_t>(
        std::ceil(std::max(when + x_max, port_end))));
  if (fill_done_out) *fill_done_out = fill_done;
  return static_cast<std::uint64_t>(std::ceil(accept)) +
      static_cast<std::uint64_t>(cfg_.store_l2_latency);
}

std::uint64_t MemoryHierarchy::store_issue_block(
    std::uint32_t sm, std::uint64_t now) {
  if (cfg_.store_l2_outstanding == 0) return now;
  auto& q = store_drain_[sm];
  while (!q.empty() && q.front() <= now) q.pop_front();
  return q.size() >= cfg_.store_l2_outstanding ? q.front() : now;
}

std::uint64_t MemoryHierarchy::constant(std::uint32_t sm,
                                        std::uint32_t warp,
                                        std::uint64_t now,
                                        std::uint64_t address,
                                        std::size_t bytes,
                                        bool uniform) {
  if (!cfg_.const_enabled) return now;
  const std::uint64_t line = cfg_.const_line_bytes
      ? address / cfg_.const_line_bytes : address;
  // bank-0 rules (optional fields, per path): flat access latency,
  // no resident / same-line state touched; counted as a hit
  if ((address >> 32) == 0) {
    const double f = uniform ? cfg_.const_bank0_uniform : cfg_.const_bank0;
    if (f >= 0.0) {
      ++tele.const_hits;
      return now + static_cast<std::uint64_t>(f);
    }
  }
  auto& last = const_last_line_[sm];
  const auto lit = last.find(warp);
  auto& set = const_resident_[sm];
  const auto hit = set.find(line);
  // 1) this warp's previous constant access was the same line; if the
  // line is still resident its fill may not have completed yet
  if (lit != last.end() && lit->second == line) {
    ++tele.const_same_line;
    const std::uint64_t ready = hit != set.end() ? hit->second.ready : now;
    return std::max(now + static_cast<std::uint64_t>(cfg_.const_same_line),
                    ready);
  }
  // 2) resident in this SM's constant cache (possibly still in flight)
  if (hit != set.end()) {
    ++tele.const_hits;
    hit->second.tick = ++const_tick_;
    last[warp] = line;
    return std::max(now + static_cast<std::uint64_t>(cfg_.const_hit),
                    hit->second.ready);
  }
  // 3) miss: fill the line (LRU when at capacity); capacity 0 keeps
  // `enabled` semantics but never retains a line
  ++tele.const_misses;
  const std::size_t cap = cfg_.const_line_bytes
      ? static_cast<std::size_t>(cfg_.const_capacity /
                                 cfg_.const_line_bytes)
      : 0;
  if (cap) {
    if (set.size() >= cap) {
      auto victim = set.begin();
      for (auto it = set.begin(); it != set.end(); ++it)
        if (it->second.tick < victim->second.tick) victim = it;
      set.erase(victim);
    }
    set[line] = {++const_tick_,
                 now + static_cast<std::uint64_t>(cfg_.const_miss)};
  }
  last[warp] = line;
  return now + static_cast<std::uint64_t>(cfg_.const_miss);
}

void MemoryHierarchy::l1_insert(std::uint32_t sm, std::uint64_t sector,
                                std::uint64_t ready) {
  if (!l1_modeled()) return;
  auto& set = l1_sets_[sm];
  if (set.cap == 0 && cfg_.line_bytes)
    set.cap = l1_cap_lines() * (cfg_.line_bytes /
               (cfg_.sector_bytes ? cfg_.sector_bytes : 32));
  set.insert(sector, ready);
}

}  // namespace cyclescope
