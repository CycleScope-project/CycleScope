#include "instruction_fetch.hpp"

namespace cyclescope {

void InstructionFetch::Lru::touch(std::uint64_t l) {
  auto it = pos.find(l);
  if (it == pos.end()) return;
  order.splice(order.end(), order, it->second);
}
void InstructionFetch::Lru::insert(std::uint64_t l) {
  if (pos.count(l)) { touch(l); return; }
  order.push_back(l);
  pos.emplace(l, std::prev(order.end()));
  while (cap && order.size() > cap) {
    pos.erase(order.front());
    order.pop_front();
  }
}

void InstructionFetch::configure(FetchConfig c) {
  cfg_ = c;
  // one field set for both devices (ifetch protocol): per-level
  // stream cadences, no separate demand head latency
  if (!cfg_.prefetch_depth) cfg_.prefetch_depth = 4;
  if (!cfg_.parallelism) cfg_.parallelism = 1;
  streams_[0] = cfg_.stream_l1;
  streams_[1] = cfg_.stream_l2;
  streams_[2] = cfg_.stream_cold;
  sms_.clear();
  l0_hits = l1_lines = l2_lines = cold_lines = staged_hits = 0;
}

void InstructionFetch::ensure_sm(std::uint32_t sm) {
  if (!cfg_.enabled) return;
  if (sms_.size() <= sm) {
    sms_.resize(sm + 1);
    for (auto& s : sms_)
      if (s.l0.empty()) {
        s.l0.resize(cfg_.sharing ? 4 : 1);
        const std::uint64_t cap_lines = cfg_.capacity_l0 / cfg_.line_bytes;
        const std::uint64_t sets =
            cfg_.l0_ways && cap_lines ? cap_lines / cfg_.l0_ways : 1;
        for (auto& o : s.l0) {
          o.resize(sets ? sets : 1);
          for (auto& l : o)
            l.cap = cfg_.l0_ways && cap_lines ? cfg_.l0_ways : cap_lines;
        }
        s.l1.cap = cfg_.capacity_l1 / cfg_.line_bytes;
        s.last_arrival.assign(
            3, std::vector<std::uint64_t>(cfg_.parallelism, 0));
      }
  }
}

void InstructionFetch::settle(Sm& s, std::uint64_t now) {
  for (auto it = s.inflight.begin(); it != s.inflight.end();) {
    if (it->second.arrival > now) { ++it; continue; }
    const std::uint64_t line = it->first;
    if (it->second.sequential) {
      // sequential prefetch arrival: parks in the staging set
      // (L1-resident); enters the L0 only when demanded
      s.staged.insert(line);
    } else if (!it->second.prefetch_only) {
      // demand fetch (or a sequential line claimed by a demand):
      // lands in its L0
      l0(s, it->second.smsp, line).insert(line);
    }
    // launch prefetch stays L1 + l2code; every filled line is also
    // L1/l2code-resident
    s.l1.insert(line);
    s.l2code.insert(line);
    it = s.inflight.erase(it);
  }
}

void InstructionFetch::launch_prefetch(std::uint32_t sm,
                                       std::uint32_t smsp,
                                       std::uint32_t channel,
                                       std::uint64_t code_addr,
                                       std::uint64_t now) {
  if (!cfg_.enabled) return;
  ensure_sm(sm);
  auto& s = sms_[sm];
  (void)channel;
  const std::uint64_t first = code_addr / cfg_.line_bytes;
  // launch prefetch: line j arrives at entry + (j+1)*stream_l1 as
  // prefetch_only and does NOT advance any channel's last_arrival;
  // with launch_first_line (kernel_boundary K3) the first line pays
  // the measured cold latency and the chain streams from that arrival
  const std::uint64_t n = cfg_.launch_prefetch / cfg_.line_bytes;
  for (std::uint64_t j = 0; j < n; ++j) {
    const std::uint64_t line = first + j;
    if (s.inflight.count(line) || s.l2code.count(line)) continue;
    const std::uint64_t arrival =
        cfg_.launch_first_line
            ? now + cfg_.launch_first_line + j * cfg_.stream_l1
            : now + (j + 1) * cfg_.stream_l1;
    s.inflight.emplace(line, Inflight{arrival, true, false, smsp});
    if (j == 0) s.launch_head_arrival = arrival;
  }
}

std::uint64_t InstructionFetch::ready(std::uint32_t sm,
                                      std::uint32_t smsp,
                                      std::uint32_t channel,
                                      std::uint64_t code_addr,
                                      std::uint64_t now,
                                      std::uint32_t warp) {
  ensure_sm(sm);
  auto& s = sms_[sm];
  settle(s, now);
  const std::uint64_t line = code_addr / cfg_.line_bytes;
  const std::uint32_t ch = channel % cfg_.parallelism;
  if (warps_.size() <= warp) warps_.resize(std::size_t(warp) + 1);
  auto& wf = warps_[warp];
  // (stream base / demand times were recorded but never read)
  wf.front = static_cast<std::int64_t>(line);
  if (l0(s, smsp, line).has(line)) {
    l0(s, smsp, line).touch(line);
    ++l0_hits;
    return now;
  }
  // an arrived sequential-prefetch line is free: it moves into the L0
  if (s.staged.erase(line)) {
    l0(s, smsp, line).insert(line);
    ++staged_hits;
    return now;
  }
  // a demand on an inflight non-prefetch line claims it for the L0
  auto it = s.inflight.find(line);
  if (it != s.inflight.end() && !it->second.prefetch_only) {
    it->second.sequential = false;
    it->second.smsp = smsp;
    return it->second.arrival;
  }
  // level of a line: an inflight launch-prefetch line counts as
  // L1-resident (the launch window streams at the L1 cadence)
  auto level_of = [&](std::uint64_t l, const Inflight* v) {
    if (v != nullptr) return 0;
    if (s.l1.has(l)) return 0;
    return s.l2code.count(l) ? 1 : 2;
  };
  const Inflight* pv0 =
      it != s.inflight.end() ? &it->second : nullptr;
  const int level = level_of(line, pv0);
  if (level == 0) ++l1_lines;
  else if (level == 1) ++l2_lines;
  else ++cold_lines;
  // demand miss: the line pays its level's stream slot on the channel
  std::uint64_t arrival =
      std::max(now, s.last_arrival[level][ch]) + stream(level);
  // K3: until the launch chain's head line lands, no demand on this
  // SM can arrive earlier; the chain's later lines keep the original
  // "chain first wins" comparison and can be out-raced by a demand
  if (cfg_.launch_first_line && arrival < s.launch_head_arrival)
    arrival = s.launch_head_arrival;
  if (pv0 != nullptr && pv0->arrival <= arrival) {
    // the launch-prefetch chain gets there first: take its arrival
    // and land the line in the L0 without spending a channel slot
    arrival = pv0->arrival;
    it->second = Inflight{arrival, false, false, smsp};
  } else {
    s.last_arrival[level][ch] = arrival;
    s.inflight[line] = Inflight{arrival, false, false, smsp};
  }
  // sequential prefetch: subsequent lines at their own stream level,
  // skipping lines already in L0/staged or claimed by a demand;
  // arrivals park in `staged`, not the L0
  std::uint64_t prev = arrival;
  for (std::uint64_t i = 1; i <= cfg_.prefetch_depth; ++i) {
    const std::uint64_t pl = line + i;
    if (l0(s, smsp, pl).has(pl) || s.staged.count(pl)) continue;
    auto pit = s.inflight.find(pl);
    if (pit != s.inflight.end() && !pit->second.prefetch_only) continue;
    const Inflight* pv =
        pit != s.inflight.end() ? &pit->second : nullptr;
    const int lv = level_of(pl, pv);
    const std::uint64_t a = prev + stream(lv);
    if (pv != nullptr && pv->arrival <= a) {
      // the launch chain beats the slot: mark it sequential so its
      // arrival parks in staged instead of the L0
      pit->second = Inflight{pv->arrival, false, true, smsp};
      prev = pv->arrival;
      continue;
    }
    s.inflight[pl] = Inflight{a, false, true, smsp};
    s.last_arrival[lv][ch] = std::max(s.last_arrival[lv][ch], a);
    prev = a;
  }
  return arrival;
}

}  // namespace cyclescope
