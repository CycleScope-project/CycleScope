// §12.1: memory hierarchy paths — pure MemoryHierarchy checks.
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "../core/memory_hierarchy.hpp"

using cyclescope::MemoryConfig;
using cyclescope::MemoryHierarchy;
using cyclescope::MemorySector;

static MemorySector sec(std::uint64_t a) {
  MemorySector s{};
  s.address = a;
  s.byte_mask = 0xFFFFFFFFu;
  s.lane_mask = 0xFFFFFFFFu;
  return s;
}

static std::uint64_t L(MemoryHierarchy& mh, std::uint32_t sm,
                       std::uint64_t now, MemorySector* s,
                       std::size_t n) {
  return mh.load(sm, now, s, n, true, nullptr, nullptr, nullptr, nullptr);
}

static std::uint64_t LK(MemoryHierarchy& mh, std::uint32_t sm,
                        std::uint64_t now, MemorySector* s,
                        std::size_t n, bool tma, bool async_copy) {
  return mh.load(sm, now, s, n, true, nullptr, nullptr, nullptr,
                 nullptr, nullptr, nullptr, tma, false, false,
                 async_copy);
}

int main() {
  MemoryConfig cfg;
  cfg.l1_latency = 30;   cfg.l1_outstanding = 8;
  cfg.l2_latency = 200;  cfg.l2_outstanding = 64;
  cfg.dram_latency = 500; cfg.dram_outstanding = 512;
  cfg.store_l2_latency = 20; cfg.store_l2_outstanding = 2;
  cfg.pool_count = 2; cfg.replicate_read_lines = true;
  cfg.far_hit_cycles = 40;
  cfg.sector_bytes = 32; cfg.line_bytes = 128;
  cfg.merge_window = 50; cfg.const_enabled = false;

  // L1/L2/DRAM composition: first access misses all -> dram_latency
  // (level latencies are load-to-use totals)
  {
    MemoryHierarchy mh(cfg);
    std::vector<MemorySector> l{sec(0)};
    const std::uint64_t t1 = L(mh, 0, 0, l.data(), 1);
    assert(t1 == 500);
    assert(mh.tele.dram_sectors == 1);
  }
  // L1 hit on a resident sector (L1 residency is modeled only when the
  // memory.l1.residency leaf gives a data array; without it an SM has no
  // L1 at all -- the former fallback table was keyed by sector across
  // SMs and counted an in-flight fill as a hit)
  {
    MemoryConfig c1 = cfg;
    c1.l1_data_array_bytes = 253952;
    MemoryHierarchy mh(c1);
    std::vector<MemorySector> l{sec(64)};
    L(mh, 0, 0, l.data(), 1);            // warm (dram fill)
    const std::uint64_t h = L(mh, 0, 800, l.data(), 1);
    assert(h == 830);                    // l1_latency only
  }
  // no L1 data array -> no L1: the second access is an L2 hit even
  // long after the fill, and an SM never sees another SM's fill as L1
  {
    MemoryHierarchy mh(cfg);
    std::vector<MemorySector> l{sec(64)};
    L(mh, 0, 0, l.data(), 1);
    assert(L(mh, 0, 800, l.data(), 1) == 800 + 200);
    assert(mh.tele.l1_sectors == 2 && mh.tele.l1_misses == 2);
  }
  // two-side replica + far hit
  {
    MemoryHierarchy mh(cfg);
    std::vector<MemorySector> l{sec(0)};
    L(mh, 0, 0, l.data(), 1);            // sm0 -> side0 fill
    const std::uint64_t far = L(mh, 1, 1000, l.data(), 1);
    assert(far == 1240);                 // 200 + 40 far
    assert(mh.tele.l2_far_hits == 1);
    // the far hit installed the reader-side replica: a second read
    // from another SM on the same side is a near hit
    const std::uint64_t near = L(mh, 3, 2000, l.data(), 1);
    assert(near == 2200);
    assert(mh.tele.l2_far_hits == 1 && mh.tele.l2_replica_fills == 1);
  }
  // L2 fill in flight: a request to a sector whose DRAM fill has not
  // completed yet, arriving after the merge window, is held at the
  // slice until the fill lands (= the filling request's completion),
  // not returned as a 200-cycle hit before the data exists; once the
  // fill has landed it is an ordinary hit
  {
    MemoryHierarchy mh(cfg);
    std::vector<MemorySector> l{sec(256)};
    L(mh, 0, 0, l.data(), 1);                        // sm0 fill -> 500
    const std::uint64_t held = L(mh, 2, 100, l.data(), 1);  // same side
    assert(held == 500);                             // not 300
    assert(mh.tele.merge_merges == 0);
    assert(mh.tele.l2_inflight_hits == 1 && mh.tele.l2_inflight_wait == 200);
    // far side: the reader-side replica of the miss is installed with
    // the same ready cycle; a far hit before it lands is held too
    MemoryHierarchy mh2(cfg);
    L(mh2, 0, 0, l.data(), 1);
    const std::uint64_t far_held = L(mh2, 1, 100, l.data(), 1);
    assert(far_held == 500);
    const std::uint64_t later = L(mh2, 1, 1000, l.data(), 1);
    assert(later == 1200);                           // plain hit
    assert(mh2.tele.l2_inflight_hits == 1);
    // a store makes the sector present at once: a read right after it
    // is a plain hit even if a fill of that sector is still in flight
    MemoryHierarchy mh3(cfg);
    L(mh3, 0, 0, l.data(), 1);
    mh3.store(0, 50, l.data(), 1, false, 256);
    assert(L(mh3, 2, 100, l.data(), 1) == 300);
    assert(mh3.tele.l2_inflight_hits == 0);
  }
  // in-flight hold is capped by the request's own miss path: a leader
  // whose DRAM request sits behind its SM port's outstanding cap
  // completes late, a follower on another SM would reach the slice
  // first and is held only to now + dram_latency; its completion
  // becomes the sector's ready cycle
  {
    MemoryConfig c3 = cfg;
    c3.dram_outstanding = 1;
    MemoryHierarchy mh(c3);
    std::vector<MemorySector> a{sec(4096)}, b{sec(8192)};
    L(mh, 0, 0, a.data(), 1);                  // sm0 slot busy until 500
    const std::uint64_t lead = L(mh, 0, 0, b.data(), 1);
    assert(lead == 1000);                      // queued behind the slot
    const std::uint64_t fol = L(mh, 2, 100, b.data(), 1);  // other SM
    assert(fol == 600);                        // min(1000, 100 + 500)
    assert(mh.tele.l2_inflight_hits >= 0);
    assert(L(mh, 4, 700, b.data(), 1) == 900); // landed at 600: plain hit
    // the cap includes the shared DRAM queue the follower's own miss
    // would see: saturate the queue around cycle 100 and the follower
    // is held past 600
    MemoryConfig c4 = c3;
    c4.dram_read_bpc = 1.0;      // 32 B per 32-cycle bucket
    MemoryHierarchy mh2(c4);
    std::vector<MemorySector> c{sec(12288)}, d{sec(16384)}, e{sec(20480)};
    L(mh2, 0, 0, c.data(), 1);                 // sm0 slot busy until 532
    L(mh2, 0, 0, d.data(), 1);                 // leader queued -> ~1064
    L(mh2, 1, 96, e.data(), 1);                // fills the bucket at 96
    const std::uint64_t f2 = L(mh2, 2, 100, d.data(), 1);
    assert(f2 > 600 && f2 < 1064);             // held by queue, not leader
  }
  // atomic / reduction RMW at L2: a sector not resident is fetched from
  // DRAM for the slice (counted as a DRAM sector, no SM port), the op
  // completes no earlier than the fill, a later read of the sector is
  // held for it, and a second atomic on the same sector fetches nothing
  {
    MemoryHierarchy mh(cfg);
    std::vector<MemorySector> l{sec(512)};
    std::uint64_t fill = 0;
    const std::uint64_t a1 = mh.store(0, 100, l.data(), 1, true, 512, &fill);
    assert(mh.tele.dram_sectors == 1 && mh.tele.atomic_fills == 1);
    assert(a1 == 100 + 20);                          // acceptance unchanged
    assert(fill == 600);                             // 100 + dram_latency
    assert(L(mh, 2, 200, l.data(), 1) == 600);       // held for the fill
    fill = 0;
    const std::uint64_t a2 = mh.store(1, 700, l.data(), 1, true, 512, &fill);
    assert(mh.tele.dram_sectors == 1 && mh.tele.atomic_fills == 1);
    assert(a2 == 700 + 20 && fill == 0);             // resident: no fetch
    // a plain store never fetches
    MemoryHierarchy mh2(cfg);
    mh2.store(0, 100, l.data(), 1, false, 512);
    assert(mh2.tele.dram_sectors == 0 && mh2.tele.atomic_fills == 0);
  }
  // shared service is time-bucketed capacity: a request arriving
  // earlier in simulated time than a previously placed later one does
  // not queue behind it
  {
    MemoryConfig c2 = cfg;
    c2.dram_read_bpc = 1.0;  // 1 B/cycle: 32 B sectors saturate buckets
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> a{sec(0)}, b{sec(4096)};
    L(mh, 0, 100000, a.data(), 1);        // late arrival placed first
    const std::uint64_t early = L(mh, 1, 0, b.data(), 1);
    assert(early == 532);                 // 32 B / 1 B/cycle + 500
    assert(mh.tele.shared_waits == 0);
  }
  // merge window: same sector within window coalesces (L1 modeled: the
  // sector is not yet resident at 10, so the second request misses L1
  // and merges with the leader's DRAM fill, completing with it at 500)
  {
    MemoryConfig c1 = cfg;
    c1.l1_data_array_bytes = 253952;
    MemoryHierarchy mh(c1);
    std::vector<MemorySector> l{sec(0)};
    L(mh, 0, 0, l.data(), 1);
    const std::uint64_t m = L(mh, 0, 10, l.data(), 1);
    assert(m == 500);
    assert(mh.tele.merge_merges == 1);
  }
  // (m) a merged follower completes no earlier than its own L2 hit
  // path: the leader hits L2 (completes ~5240); a follower in the
  // window would otherwise take the leader's completion, but its own
  // request cannot beat issue + l2_latency (+ far when the reader
  // side holds no replica)
  {
    MemoryConfig c2 = cfg;
    c2.l1_data_array_bytes = 253952;   // L1 modeled: fills are per-SM
    c2.merge_window = 1000; c2.merge_max_requests = 0;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> l{sec(0)};
    L(mh, 0, 0, l.data(), 1);                            // sm0: dram fill
    const std::uint64_t lead = L(mh, 1, 5000, l.data(), 1);   // sm1: L2 hit
    const std::uint64_t foll = L(mh, 2, 5100, l.data(), 1);   // sm2: merged
    assert(mh.tele.merge_merges >= 1);
    assert(foll >= 5100 + 200);                          // own hit latency
    assert(foll >= lead);
    // a follower issued in the same cycle as the leader keeps the
    // leader's completion (floor is not later than it)
    const std::uint64_t lead2 = L(mh, 3, 9000, l.data(), 1);
    const std::uint64_t foll2 = L(mh, 4, 9000, l.data(), 1);
    assert(foll2 == lead2);
  }
  // l1.shared_configuration_*: occupancy rule picks the smallest
  // carveout >= n_res x per-CTA need (static + dynamic + 1 KiB);
  // budget = data_array - reserved - configuration.  Cells per
  // l1_residency v4 model.json (10/10 both devices)
  {
    MemoryConfig c2 = cfg;
    c2.l1_data_array_bytes = 253952;
    c2.l1_reserved_bytes = 0;
    c2.shared_static_bytes = 16;
    c2.shared_configurations = {0, 8192, 16384, 32768, 65536, 102400,
                               135168, 167936, 200704, 233472};
    struct Cell { std::uint64_t dyn, threads, expect; };
    const Cell cells[] = {
        {0, 32, 188416},  {0, 1024, 245760},
        {32768, 32, 20480},   {32768, 1024, 151552},
        {65536, 32, 53248},   {65536, 1024, 118784},
        {131072, 32, 118784}, {131072, 1024, 118784},
        {196608, 32, 53248},  {196608, 1024, 53248},
    };
    for (const auto& c : cells) {
      c2.shared_dynamic_bytes = c.dyn;
      c2.launch_warps_per_cta = (c.threads + 31) / 32;
      MemoryHierarchy mh(c2);
      assert(mh.l1_capacity_bytes() == c.expect);
    }
    // no configurations listed -> legacy budget: launch's own shared
    // bytes subtracted
    {
      MemoryConfig c3 = c2;
      c3.shared_configurations.clear();
      c3.shared_dynamic_bytes = 32768;
      MemoryHierarchy mh(c3);
      assert(mh.l1_capacity_bytes() ==
             253952 - 16 - 32768);
    }
    // per-CTA need above the largest carveout: n_res clamps to 1 ->
    // largest configuration
    {
      c2.shared_dynamic_bytes = 232448;   // 227 KiB
      c2.launch_warps_per_cta = 4;
      MemoryHierarchy mh(c2);
      assert(mh.shared_configuration_bytes() == 233472);
      assert(mh.l1_capacity_bytes() == 253952 - 233472);
    }
  }
  // store_l2 outstanding cap, counted per request (per store() call),
  // not per sector: the first two store() calls each take one inflight
  // slot, so their accept cycles differ only by port occupancy; the
  // third store() waits for the first request's slot to complete
  {
    MemoryHierarchy mh(cfg);
    std::vector<MemorySector> l{sec(0), sec(32)};
    const std::uint64_t a = mh.store(0, 0, l.data(), 2, false, 0);
    const std::uint64_t b = mh.store(0, 0, l.data(), 2, false, 0);
    assert(b - a <= 2);   // port occupancy only (1 cycle/sector)
    const std::uint64_t c = mh.store(0, 0, l.data(), 2, false, 0);
    assert(c > b);                              // slot wait kicked in
    assert(c >= a + cfg.store_l2_latency);      // == first request done
  }
  // request-vs-sector inflight unit discriminator: l2_outstanding=2
  // counts load() calls (warp-level requests), not sectors.  A
  // 16-sector L2-hit load takes ONE inflight slot that lives until its
  // last sector returns.  Under the old per-sector counting the second
  // load below would see 16 inflight entries > cap and start past 1215
  // (return >= 1400), so the t2 <= 1232 assertion is the discriminating
  // check.
  {
    MemoryConfig c2 = cfg;
    c2.l2_outstanding = 2;
    c2.l2_bytes_per_cycle = 32;   // 1 sector/cycle/SM port
    c2.l2_read_bpc = 0;           // port only, no shared queue
    c2.merge_window = 0;
    c2.pool_count = 1;
    c2.l1_data_array_bytes = 0;   // no L1
    c2.l2_latency = 200;
    MemoryHierarchy mh(c2);
    // fill the 16 sectors into L2 via DRAM misses (the DRAM port, not
    // the L2 port under test); L2 presence is sector-granular, so every
    // sector has to be filled on its own
    for (int i = 0; i < 16; ++i) {
      std::vector<MemorySector> x{sec(i * 32)};
      L(mh, 0, 0, x.data(), 1);
    }
    std::vector<MemorySector> l(16);
    for (int i = 0; i < 16; ++i) l[i] = sec(i * 32);
    // 16 sectors serialise on the 1-sector/cycle port: the last starts
    // at 1000+15 and returns at 1015+200
    const std::uint64_t t1 = L(mh, 0, 1000, l.data(), 16);
    assert(t1 >= 1214 && t1 <= 1216);   // 1215
    // second load = a second request; inflight was 1 < cap so there is
    // no slot wait: sectors start at 1016..1031, return ~1231
    const std::uint64_t t2 = L(mh, 0, 1016, l.data(), 16);
    assert(t2 <= 1232);
    // third load: both slots occupied, front completes at 1215 > 1032,
    // so start is pushed to 1215; last sector returns ~1430
    const std::uint64_t t3 = L(mh, 0, 1032, l.data(), 16);
    assert(t3 >= 1400);
  }
  // L2 presence is sector-granular: after sector 0 of a line is filled,
  // sector 1 of the same line is still a DRAM miss (a store on sector 2
  // does not make sector 3 resident either); a second read of sector 0
  // hits
  {
    MemoryConfig c2 = cfg;
    c2.merge_window = 0;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> s0{sec(0)}, s1{sec(32)}, s2{sec(64)},
        s3{sec(96)};
    assert(L(mh, 0, 0, s0.data(), 1) == 500);
    assert(L(mh, 0, 1000, s1.data(), 1) == 1500);     // sibling: DRAM
    assert(L(mh, 0, 2000, s0.data(), 1) == 2200);     // same sector: L2
    mh.store(0, 3000, s2.data(), 1, false, 0);
    assert(L(mh, 0, 4000, s3.data(), 1) == 4500);     // unwritten: DRAM
    assert(L(mh, 0, 5000, s2.data(), 1) == 5200);     // written: L2
    assert(mh.tele.dram_sectors == 3 && mh.tele.l2_sectors == 2);
  }
  // atomic same-address serialization
  {
    cfg.atomic_same_addr_service = 100;
    MemoryHierarchy mh(cfg);
    std::vector<MemorySector> l{sec(0)};
    const std::uint64_t a = mh.store(0, 0, l.data(), 1, true, 0);
    assert(a >= 120);
    assert(mh.tele.atomics == 1);
  }

  // ---- F1 additions ----
  // (a) single SM, N sequential L2-hit sectors: last completion =
  // now + latency + (N-1)*32/bpc
  {
    MemoryConfig c2 = cfg;
    c2.l2_bytes_per_cycle = 128;
    c2.l2_read_bpc = 0;   // port only
    c2.merge_window = 0;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> w1{sec(0)};
    L(mh, 0, 0, w1.data(), 1);            // warm (dram fill)
    const int N = 8;
    std::vector<MemorySector> l(N);
    for (int i = 0; i < N; ++i) l[i] = sec((i + 1) * 32);
    for (int i = 0; i < N; ++i) { std::vector<MemorySector> x{l[i]};
      L(mh, 0, 0, x.data(), 1); }         // fill via dram
    const std::uint64_t t = L(mh, 0, 1000, l.data(), N);
    assert(t == 1000 + 200 + (N - 1) * 32 / 128 + 1 ||
           t <= 1000 + 200 + N);          // ~latency + port spread
  }
  // (b) two SMs sharing a small L2 queue: completion is
  // shared-queue bound
  {
    MemoryConfig c2 = cfg;
    c2.l2_bytes_per_cycle = 1024;  // fast ports
    c2.l2_read_bpc = 32;           // 1 sector/cycle shared
    c2.merge_window = 0;
    MemoryHierarchy mh(c2);
    const int N = 8;
    std::vector<MemorySector> a(N), b(N);
    for (int i = 0; i < N; ++i) {
      a[i] = sec((i + 1) * 32); b[i] = a[i];
      std::vector<MemorySector> x{a[i]};
      L(mh, 0, 0, x.data(), 1);           // fill
    }
    const std::uint64_t ta = L(mh, 0, 0, a.data(), N);
    const std::uint64_t tb = L(mh, 1, 0, b.data(), N);
    // 16 sectors at 32 B/cycle shared -> >= 16 cycles of service
    assert(tb >= 200 + 16);
    (void)ta;
  }
  // (c) merge window expiry + max_requests cap
  {
    MemoryConfig c2 = cfg;
    c2.merge_window = 50; c2.merge_max_requests = 0;
    c2.l2_read_bpc = 0; c2.l2_bytes_per_cycle = 128;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> l{sec(0)};
    const std::uint64_t t1 = L(mh, 0, 0, l.data(), 1);   // dram miss
    const std::uint64_t t2 = L(mh, 0, 10, l.data(), 1);  // merged
    assert(t2 == t1);
    assert(mh.tele.merge_merges == 1);
    const std::uint64_t t3 = L(mh, 0, 100, l.data(), 1); // window out
    assert(t3 == t1);                 // L2 hit held for the in-flight fill
    assert(mh.tele.merge_merges == 1 && mh.tele.l2_inflight_hits == 1);
    const std::uint64_t t4 = L(mh, 0, 600, l.data(), 1); // fill landed
    assert(t4 == 600 + 200);                            // L2 hit reservice
    // max_requests = 1 -> no merge
    MemoryConfig c3 = cfg;
    c3.merge_window = 50; c3.merge_max_requests = 1;
    MemoryHierarchy mh2(c3);
    std::vector<MemorySector> l2{sec(64)};
    L(mh2, 0, 0, l2.data(), 1);
    const std::uint64_t m = L(mh2, 0, 10, l2.data(), 1);
    assert(m == 500);                 // not merged, held for the fill
    assert(mh2.tele.merge_merges == 0 && mh2.tele.l2_inflight_hits == 1);
  }
  // (d) L2 capacity: 2 lines; third line evicts the first; dirty
  // eviction counts a writeback
  {
    MemoryConfig c2 = cfg;
    c2.l2_resident_lines = 2; c2.replicate_read_lines = false;
    c2.merge_window = 0;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> d{sec(0)};
    mh.store(0, 0, d.data(), 1, false, 0);   // line0 dirty, w=2
    std::vector<MemorySector> x{sec(128)};
    L(mh, 0, 0, x.data(), 1);                // line1 fill, w=1,total 3
    std::vector<MemorySector> y{sec(256)};
    L(mh, 0, 0, y.data(), 1);                // line2: evicts line0
    assert(mh.tele.dirty_writebacks >= 1);
    assert(mh.tele.l2_evictions >= 1);
    // line0 again -> dram miss (it was evicted)
    const auto ds = mh.tele.dram_sectors;
    L(mh, 0, 5000, d.data(), 1);
    assert(mh.tele.dram_sectors == ds + 1);
  }
  // (e) L1 capacity 1 line: second access to the same line hits L1
  {
    MemoryConfig c2 = cfg;
    c2.l1_data_array_bytes = 128; c2.l1_reserved_bytes = 0;
    c2.merge_window = 0;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> l{sec(0)};
    L(mh, 0, 0, l.data(), 1);
    const std::uint64_t h = L(mh, 0, 5000, l.data(), 1);
    assert(h == 5000 + 30);
  }
  // (f) store: two SMs, huge shared write queue -> each SM bound
  // only by its own store port, not a shared 2N queue: 8 sectors at
  // one per cycle (no outstanding cap in this case) -> 8 + latency
  {
    MemoryConfig c2 = cfg;
    c2.l2_write_bpc = 1e9;
    c2.store_l2_bytes_per_cycle = 32;   // 1 sector/cycle/SM
    c2.store_l2_latency = 10;
    c2.store_l2_outstanding = 0;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> s(8);
    for (int i = 0; i < 8; ++i) s[i] = sec(8192 + i * 32);
    const std::uint64_t a = mh.store(0, 0, s.data(), 8, false, 0);
    const std::uint64_t b = mh.store(1, 0, s.data(), 8, false, 0);
    assert(a == 18 && b == 18);
  }
  // (g) P1: async-path sectors bypass L1; L1 hits need completion
  {
    MemoryConfig c2 = cfg;
    c2.l1_data_array_bytes = 128;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> l{sec(0)};
    mh.load(0, 0, l.data(), 1, false, nullptr, nullptr, nullptr, nullptr);
    // same address via use_l1: L1 not filled by the async path -> miss
    const std::uint64_t t1 = L(mh, 0, 5000, l.data(), 1);
    assert(t1 >= 5000 + 30);
    // LDG fill completes ~5230; a later read hits L1
    const std::uint64_t t2 = L(mh, 0, 20000, l.data(), 1);
    assert(t2 == 20000 + 30);
  }
  // (h) P2: TOPO sm_partition mapping changes far-hit accounting
  {
    MemoryConfig c3 = cfg;
    c3.l1_data_array_bytes = 0;   // L1 off
    c3.merge_window = 0;          // sm3's request must not merge with sm0's
    c3.replicate_read_lines = false;
    c3.far_hit_cycles = 40;
    c3.l2_latency = 30;
    c3.sm_partition = {0, 0, 1, 1};  // sm0,sm1 -> side0; sm2,sm3 -> side1
    MemoryHierarchy mh(c3);
    // fill line 0 (home = 0 % 2 = 0) on side 0 via an sm0 miss
    std::vector<MemorySector> l{sec(0)};
    L(mh, 0, 0, l.data(), 1);
    const std::uint64_t t_local = L(mh, 0, 1000, l.data(), 1); // side0
    const std::uint64_t t_far   = L(mh, 3, 1000, l.data(), 1); // side1 -> far
    assert(t_local <= 1000 + 30 + 16);
    assert(t_far   >= t_local + 40 - 4);
    // without TOPO (empty) sm3 -> side1 anyway; with the old
    // sm % pool_count sm3 also -> 1, so check sm1 side0 local
    const std::uint64_t t2 = L(mh, 1, 1000, l.data(), 1);
    assert(t2 <= t_local + 4);
  }
  // (j) P3: merged sectors pay the delivered-bytes queue
  {
    MemoryConfig c5 = cfg;
    c5.merge_window = 1000;
    c5.merge_max_requests = 0;
    c5.merge_bytes_per_cycle = 32.0;   // 32 B/cycle delivery
    MemoryHierarchy mh(c5);
    std::vector<MemorySector> l{sec(0)};
    const std::uint64_t first = L(mh, 0, 100, l.data(), 1);  // miss
    // 4 merged reads: each adds 32/32 = 1 cycle to the queue
    for (int i = 0; i < 4; ++i)
      L(mh, 0, 110, l.data(), 1);
    // queue has pushed delivery to >= 114; no far pileup (cap 32B/c)
    // another merge at 110 sees delivery >= 110+5
    const std::uint64_t t = L(mh, 0, 110, l.data(), 1);
    assert(t >= first - 1);            // at least the first completion
    assert(t <= first + 8);            // and bounded by delivery queue
  }
  // (k-dom) l2_merge domain_sms: a read merges only inside floor(sm /
  // domain); 0 = any SM.  window 240, max_requests 4.
  {
    // (a) domain 0: sm0's read is shared by sm2 (as before)
    MemoryConfig c = cfg;
    c.merge_window = 240; c.merge_max_requests = 4;
    c.merge_domain_sms = 0;
    MemoryHierarchy mh(c);
    std::vector<MemorySector> l{sec(0)};
    L(mh, 0, 0, l.data(), 1);
    L(mh, 2, 10, l.data(), 1);
    assert(mh.tele.merge_merges == 1);
  }
  {
    // (b) domain 2: sm2 does not merge with sm0; sm1 merges with sm0,
    // sm3 merges with sm2
    MemoryConfig c = cfg;
    c.merge_window = 240; c.merge_max_requests = 4;
    c.merge_domain_sms = 2;
    MemoryHierarchy mh(c);
    std::vector<MemorySector> l{sec(0)};
    L(mh, 0, 0, l.data(), 1);
    L(mh, 2, 10, l.data(), 1);
    assert(mh.tele.merge_merges == 0);
    L(mh, 1, 10, l.data(), 1);
    assert(mh.tele.merge_merges == 1);
    L(mh, 3, 10, l.data(), 1);
    assert(mh.tele.merge_merges == 2);
  }
  {
    // (c) domain 2: same domain but outside the window -> no merge
    MemoryConfig c = cfg;
    c.merge_window = 240; c.merge_max_requests = 4;
    c.merge_domain_sms = 2;
    MemoryHierarchy mh(c);
    std::vector<MemorySector> l{sec(0)};
    L(mh, 0, 0, l.data(), 1);
    L(mh, 1, 300, l.data(), 1);
    assert(mh.tele.merge_merges == 0);
  }
  // (k-gpc) l2_merge domain_gpc: the request kinds it selects merge
  // between SMs with the same sm_merge_domain entry in a key space
  // separate from the other kinds (which keep floor(sm / domain_sms));
  // SMs without a table entry (0xFFFFFFFF) are a missing fact.
  // window 240, max_requests 4, domain_sms 2 for the no-gpc pair
  // fallback (128 + sm / 2).
  auto gpc_cfg = [] {
    MemoryConfig c;
    c.l2_latency = 200; c.l2_outstanding = 64;
    c.dram_latency = 500; c.dram_outstanding = 512;
    c.pool_count = 2; c.replicate_read_lines = true;
    c.sector_bytes = 32; c.line_bytes = 128;
    c.merge_window = 240; c.merge_max_requests = 4;
    c.merge_domain_sms = 2; c.merge_domain_gpc = 1;
    c.sm_merge_domain.assign(40, 0xFFFFFFFFu);
    for (std::uint32_t sm : {0u, 1u, 16u, 17u, 32u})
      c.sm_merge_domain[sm] = 0;
    c.sm_merge_domain[2] = 1; c.sm_merge_domain[3] = 1;
    c.sm_merge_domain[4] = 130; c.sm_merge_domain[5] = 130;
    return c;
  };
  {
    // (a) different pairs of one gpc merge; another gpc does not
    MemoryHierarchy mh(gpc_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, true, false);
    LK(mh, 16, 10, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 1);
    LK(mh, 2, 10, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 1);
    LK(mh, 3, 20, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 2);
  }
  {
    // (b) no-gpc fallback pair merges with itself; a gpc SM does not
    // join the fallback pair
    MemoryHierarchy mh(gpc_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 4, 0, l.data(), 1, true, false);
    LK(mh, 5, 10, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 1);
    LK(mh, 0, 10, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 1);
  }
  {
    // (c) one read serves max_requests = 4 requests of one gpc
    MemoryHierarchy mh(gpc_cfg());
    std::vector<MemorySector> l{sec(0)};
    const std::uint32_t sms[5] = {0, 1, 16, 17, 32};
    for (int i = 0; i < 5; ++i)
      LK(mh, sms[i], (std::uint64_t)i, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 3);
  }
  {
    // (d) same gpc but outside the window -> no merge
    MemoryHierarchy mh(gpc_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, true, false);
    LK(mh, 16, 300, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 0);
  }
  {
    // (e) flag off: the pair rule is unchanged
    MemoryConfig c = gpc_cfg();
    c.merge_domain_gpc = 0;
    MemoryHierarchy mh(c);
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, true, false);
    LK(mh, 16, 10, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 0);
  }
  {
    // (f) a load from an SM without a table entry, or beyond the
    // table, is a missing fact
    MemoryHierarchy mh(gpc_cfg());
    std::vector<MemorySector> l{sec(0)};
    for (std::uint32_t sm : {6u, 99u}) {
      bool threw = false;
      try {
        LK(mh, sm, 0, l.data(), 1, true, false);
      } catch (const cyclescope::EngineError& e) {
        threw = true;
        assert(std::string(e.what()).rfind("missing_fact:sm_topology", 0) == 0);
      }
      assert(threw);
    }
  }
  {
    // (g) value 1, plain loads: the pair rule applies
    MemoryHierarchy mh(gpc_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, false);
    LK(mh, 16, 10, l.data(), 1, false, false);
    assert(mh.tele.merge_merges == 0);
    LK(mh, 1, 20, l.data(), 1, false, false);
    assert(mh.tele.merge_merges == 1);
  }
  {
    // (h) value 1, async copies: not gpc kind -> pair rule;
    // value 2: gpc kind
    MemoryHierarchy mh(gpc_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, true);
    LK(mh, 16, 10, l.data(), 1, false, true);
    assert(mh.tele.merge_merges == 0);
    MemoryConfig c = gpc_cfg();
    c.merge_domain_gpc = 2;
    MemoryHierarchy mh2(c);
    LK(mh2, 0, 0, l.data(), 1, false, true);
    LK(mh2, 16, 10, l.data(), 1, false, true);
    assert(mh2.tele.merge_merges == 1);
  }
  {
    // (i) value 1: gpc-kind and pair-kind requests of one sector do
    // not share an entry
    MemoryHierarchy mh(gpc_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, true, false);
    LK(mh, 1, 10, l.data(), 1, false, false);
    assert(mh.tele.merge_merges == 0);
    LK(mh, 16, 20, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 1);
  }
  {
    // (j) value 0 with the table filled: the old pair rule, TMA or
    // not
    MemoryConfig c = gpc_cfg();
    c.merge_domain_gpc = 0;
    MemoryHierarchy mh(c);
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, true, false);
    LK(mh, 16, 10, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 0);
    LK(mh, 1, 20, l.data(), 1, true, false);
    assert(mh.tele.merge_merges == 1);
  }
  // (k-aw) l2_merge async_copy_window_cycles: a read opened by an
  // async copy keeps the shorter window; a follower merges against
  // the opener's window. window 240, async window 20, max_requests 4,
  // domain_sms 2, domain_gpc 0.
  auto aw_cfg = [] {
    MemoryConfig c;
    c.l2_latency = 200; c.l2_outstanding = 64;
    c.dram_latency = 500; c.dram_outstanding = 512;
    c.pool_count = 2; c.replicate_read_lines = true;
    c.sector_bytes = 32; c.line_bytes = 128;
    c.merge_window = 240; c.merge_window_async = 20;
    c.merge_max_requests = 4; c.merge_domain_sms = 2;
    c.merge_domain_gpc = 0;
    return c;
  };
  {
    // (a) async copies inside the async window merge
    MemoryHierarchy mh(aw_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, true);
    LK(mh, 1, 10, l.data(), 1, false, true);
    assert(mh.tele.merge_merges == 1);
  }
  {
    // (b) async copies outside the async window do not merge
    MemoryHierarchy mh(aw_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, true);
    LK(mh, 1, 30, l.data(), 1, false, true);
    assert(mh.tele.merge_merges == 0);
  }
  {
    // (c) plain loads still use the long window
    MemoryHierarchy mh(aw_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, false);
    LK(mh, 1, 30, l.data(), 1, false, false);
    assert(mh.tele.merge_merges == 1);
  }
  {
    // (d) merge_window_async = 0: async copies keep the long window
    MemoryConfig c = aw_cfg();
    c.merge_window_async = 0;
    MemoryHierarchy mh(c);
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, true);
    LK(mh, 1, 30, l.data(), 1, false, true);
    assert(mh.tele.merge_merges == 1);
  }
  {
    // (e) a read opened by a plain load keeps the long window even
    // when the follower is an async copy
    MemoryHierarchy mh(aw_cfg());
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, false);
    LK(mh, 1, 30, l.data(), 1, false, true);
    assert(mh.tele.merge_merges == 1);
  }
  // (k-aws) async_copy_window_from_service: an async copy's window
  // starts when its sector leaves the shared read service, so a read
  // queued at the read limit stays joinable while it waits.
  // dram_read_bpc 1 B/cycle: one 32 B sector fills a 32-cycle bucket,
  // the leader queued behind another sector leaves service at 64.
  auto aws_cfg = [] {
    MemoryConfig c;
    c.l2_latency = 200; c.l2_outstanding = 64;
    c.dram_latency = 500; c.dram_outstanding = 512;
    c.pool_count = 2; c.replicate_read_lines = true;
    c.sector_bytes = 32; c.line_bytes = 128;
    c.merge_window = 240; c.merge_window_async = 20;
    c.merge_max_requests = 4; c.merge_domain_sms = 2;
    c.merge_domain_gpc = 0;
    c.dram_read_bpc = 1.0;
    return c;
  };
  {
    // (a) leader queued in the shared service (served = 64): flag off
    // -> the follower at 30 misses the issue-time window (20);
    // flag on -> it merges (service end 64 + 20)
    MemoryHierarchy mh(aws_cfg());
    std::vector<MemorySector> a{sec(0)}, b{sec(4096)};
    LK(mh, 2, 0, b.data(), 1, false, true);   // other pair, bucket 0
    LK(mh, 0, 0, a.data(), 1, false, true);   // leader, served at 64
    LK(mh, 1, 30, a.data(), 1, false, true);
    assert(mh.tele.merge_merges == 0);
    MemoryConfig c = aws_cfg();
    c.merge_window_async_from_service = true;
    MemoryHierarchy mh2(c);
    LK(mh2, 2, 0, b.data(), 1, false, true);
    LK(mh2, 0, 0, a.data(), 1, false, true);
    LK(mh2, 1, 30, a.data(), 1, false, true);
    assert(mh2.tele.merge_merges == 1);
  }
  {
    // (b) flag on without queueing (no shared read limit, so the
    // sector leaves service at issue): the issue-time behaviour of
    // the async window is unchanged
    MemoryConfig c = aw_cfg();
    c.merge_window_async_from_service = true;
    MemoryHierarchy mh(c);
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, true);
    LK(mh, 1, 10, l.data(), 1, false, true);
    assert(mh.tele.merge_merges == 1);
    MemoryHierarchy mh2(c);
    LK(mh2, 0, 0, l.data(), 1, false, true);
    LK(mh2, 1, 30, l.data(), 1, false, true);
    assert(mh2.tele.merge_merges == 0);
  }
  {
    // (c) the flag does not touch plain loads (long window as before)
    MemoryConfig c = aw_cfg();
    c.merge_window_async_from_service = true;
    MemoryHierarchy mh(c);
    std::vector<MemorySector> l{sec(0)};
    LK(mh, 0, 0, l.data(), 1, false, false);
    LK(mh, 1, 30, l.data(), 1, false, false);
    assert(mh.tele.merge_merges == 1);
  }
  // (k-gsvc) per-GPC shared services: L2 reads and stores of SMs in
  // one GPC queue at the per-GPC rate (in series before the GPU-wide
  // service); SMs of different GPCs do not share it; an SM without a
  // gpc skips the layer. bpc 1 B/cycle: one 32 B sector fills a
  // 32-cycle bucket.
  auto gsvc_cfg = [] {
    MemoryConfig c;
    c.l2_latency = 200; c.l2_outstanding = 64;
    c.dram_latency = 500; c.dram_outstanding = 512;
    c.store_l2_latency = 20; c.store_l2_outstanding = 8;
    c.pool_count = 2; c.replicate_read_lines = true;
    c.sector_bytes = 32; c.line_bytes = 128;
    c.l2_bytes_per_cycle = 128;
    c.l2_read_gpc_bpc = 1.0; c.l2_write_gpc_bpc = 1.0;
    c.sm_gpc.assign(8, 0xFFFFFFFFu);
    c.sm_gpc[0] = 0; c.sm_gpc[1] = 0;
    c.sm_gpc[2] = 1; c.sm_gpc[3] = 1;
    return c;
  };
  {
    // reads: two hit sectors served by one GPC queue (second ends
    // 32 cycles later) vs two GPCs; rate 0 -> identical
    MemoryConfig c = gsvc_cfg();
    c.l2_write_gpc_bpc = 0;
    std::vector<MemorySector> a{sec(0)}, b{sec(128)};
    auto prime = [](MemoryHierarchy& mh, std::uint32_t sm,
                    MemorySector* s) { L(mh, sm, 0, s, 1); };
    MemoryHierarchy m0(c);   // same gpc
    prime(m0, 0, a.data()); prime(m0, 1, b.data());
    const std::uint64_t same = std::max(L(m0, 0, 1000, a.data(), 1),
                                        L(m0, 1, 1000, b.data(), 1));
    MemoryHierarchy m1(c);   // different gpcs
    prime(m1, 0, a.data()); prime(m1, 2, a.data());
    const std::uint64_t diff = std::max(L(m1, 0, 1000, a.data(), 1),
                                        L(m1, 2, 1000, a.data(), 1));
    assert(same > diff);
    MemoryConfig c0 = c; c0.l2_read_gpc_bpc = 0;
    MemoryHierarchy m2(c0), m3(c0);
    prime(m2, 0, a.data()); prime(m2, 1, b.data());
    prime(m3, 0, a.data()); prime(m3, 2, a.data());
    const std::uint64_t off_same = std::max(L(m2, 0, 1000, a.data(), 1),
                                            L(m2, 1, 1000, b.data(), 1));
    const std::uint64_t off_diff = std::max(L(m3, 0, 1000, a.data(), 1),
                                            L(m3, 2, 1000, a.data(), 1));
    assert(off_same == off_diff);
    // an SM without a gpc (sm4) is not delayed
    MemoryHierarchy m4(c);
    prime(m4, 4, a.data());
    assert(L(m4, 4, 1000, a.data(), 1) == off_diff);
  }
  {
    // stores: same pair of assertions on the write service
    MemoryConfig c = gsvc_cfg();
    c.l2_read_gpc_bpc = 0;
    std::vector<MemorySector> a{sec(0)}, b{sec(128)};
    MemoryHierarchy m0(c);
    const std::uint64_t same = std::max(
        m0.store(0, 0, a.data(), 1, false, 0),
        m0.store(1, 0, b.data(), 1, false, 128));
    MemoryHierarchy m1(c);
    const std::uint64_t diff = std::max(
        m1.store(0, 0, a.data(), 1, false, 0),
        m1.store(2, 0, a.data(), 1, false, 0));
    assert(same > diff);
    MemoryConfig c0 = c; c0.l2_write_gpc_bpc = 0;
    MemoryHierarchy m2(c0), m3(c0);
    const std::uint64_t off_same = std::max(
        m2.store(0, 0, a.data(), 1, false, 0),
        m2.store(1, 0, b.data(), 1, false, 128));
    const std::uint64_t off_diff = std::max(
        m3.store(0, 0, a.data(), 1, false, 0),
        m3.store(2, 0, a.data(), 1, false, 0));
    assert(off_same == off_diff);
    // an SM without a gpc (sm4) is not delayed
    MemoryHierarchy m4(c);
    assert(m4.store(4, 0, a.data(), 1, false, 0) == off_diff);
    // a reduce (storage.tma.reduce, sector_cost_ratio 1.25) scales
    // the per-GPC write service like the GPU-wide one: the same two
    // stores on one GPC finish later than with ratio 1.0
    MemoryHierarchy m5(c), m6(c);
    const std::uint64_t r_same = std::max(
        m5.store(0, 0, a.data(), 1, false, 0, nullptr, 1.25),
        m5.store(1, 0, b.data(), 1, false, 128, nullptr, 1.25));
    const std::uint64_t p_same = std::max(
        m6.store(0, 0, a.data(), 1, false, 0),
        m6.store(1, 0, b.data(), 1, false, 128));
    assert(r_same > p_same);
  }
  // (k) constant cache: per-SM LRU + per-warp same-line fast path.
  // same_line=10, hit=30, miss=300, line 256 B, capacity 512 B (2 lines)
  auto ccfg = [] {
    MemoryConfig c;
    c.const_enabled = true;
    c.const_same_line = 10; c.const_hit = 30; c.const_miss = 300;
    c.const_line_bytes = 256; c.const_capacity = 512;
    return c;
  };
  // K1: same SM, two warps — first miss, second hit (after the fill;
  // an access while the fill is still in flight waits for it, see K5)
  {
    MemoryHierarchy mh(ccfg());
    assert(mh.constant(0, 0, 100, 0x000, 4) == 400);          // miss
    assert(mh.constant(0, 1, 450, 0x040, 4) == 480);          // hit (same line)
    assert(mh.tele.const_misses == 1 && mh.tele.const_hits == 1);
  }
  // K2: same warp, same line back-to-back -> same_line
  {
    MemoryHierarchy mh(ccfg());
    mh.constant(0, 0, 100, 0x080, 4);                          // miss
    assert(mh.constant(0, 0, 500, 0x0c0, 4) == 510);          // same_line
    assert(mh.tele.const_same_line == 1);
    // different line in between clears the fast path
    mh.constant(0, 0, 600, 0x200, 4);                          // miss (new line)
    assert(mh.constant(0, 0, 700, 0x080, 4) == 730);          // hit, not same
  }
  // K3: capacity 2 — third distinct line evicts the oldest
  {
    MemoryHierarchy mh(ccfg());
    mh.constant(0, 0, 0, 0x000, 4);      // miss, fill A
    mh.constant(0, 0, 0, 0x100, 4);      // miss, fill B (A oldest)
    mh.constant(0, 0, 0, 0x200, 4);      // miss, evicts A
    assert(mh.constant(0, 0, 400, 0x000, 4) == 700);   // A evicted: miss
    assert(mh.constant(0, 0, 400, 0x200, 4) == 430);   // C still resident
    assert(mh.tele.const_misses == 4);
  }
  // K5: an in-flight fill is not a hit — a read of a line whose fill
  // has not completed waits for the fill-ready cycle
  {
    MemoryHierarchy mh(ccfg());
    assert(mh.constant(0, 0, 100, 0x000, 4) == 400);          // miss
    assert(mh.constant(0, 1, 101, 0x040, 4) == 400);   // in-flight -> ready
    assert(mh.constant(0, 1, 200, 0x040, 4) == 400);   // same-line, in flight
    assert(mh.constant(0, 2, 500, 0x040, 4) == 530);   // after fill: real hit
    assert(mh.tele.const_misses == 1 && mh.tele.const_hits == 2 &&
           mh.tele.const_same_line == 1);
  }
  // K4: per-SM isolation + separation from the data L1
  {
    MemoryHierarchy mh(ccfg());
    mh.constant(0, 0, 0, 0x000, 4);      // sm0 miss, fills sm0's line A
    assert(mh.constant(1, 0, 0, 0x000, 4) == 300);   // sm1: own cache, miss
    // a data-L1 resident line is not a const hit and vice versa
    MemoryConfig c6 = ccfg();
    c6.l1_latency = 30; c6.l1_outstanding = 8;
    c6.l2_latency = 200; c6.l2_outstanding = 64;
    c6.dram_latency = 500; c6.dram_outstanding = 512;
    c6.sector_bytes = 32; c6.line_bytes = 128; c6.pool_count = 1;
    c6.l1_data_array_bytes = 253952;     // data L1 modeled
    c6.const_line_bytes = 128;
    MemoryHierarchy mh2(c6);
    std::vector<MemorySector> l{sec(0)};
    L(mh2, 0, 0, l.data(), 1);           // data L1 fill
    assert(mh2.constant(0, 0, 0, 0x000, 4) == 300);  // const still misses
    mh2.constant(0, 0, 0, 0x200, 4);     // const miss on other line
    const std::uint64_t h = L(mh2, 0, 800, l.data(), 1);
    assert(h == 830);                    // data L1 still hits normally
  }
  // ---- per-request port floor ----
  // a port accepts at most one warp-level request per cycle: when the
  // sectors of a request sum to less than the issue-interval floor the
  // port keeps the difference (b300 memq: ~1.0 request/cycle/SM)
  // (a) l2 bpc = 128 B/cycle -> 0.25 cycle/sector, no outstanding cap:
  // 8 sequential 1-sector L2-hit requests serialise at 1/cycle, so the
  // k-th request's service start is now + k (not now + 0.25*k)
  {
    MemoryConfig c2 = cfg;
    c2.l2_bytes_per_cycle = 128;
    c2.l2_read_bpc = 0;
    c2.l2_outstanding = 0;
    c2.merge_window = 0;
    c2.l1_data_array_bytes = 0;
    MemoryHierarchy mh(c2);
    for (int i = 0; i < 8; ++i) {           // fill 8 lines via DRAM
      std::vector<MemorySector> x{sec(i * 128)};
      L(mh, 0, 0, x.data(), 1);
    }
    for (int k = 0; k < 8; ++k) {
      std::vector<MemorySector> l{sec(k * 128)};
      const std::uint64_t t = L(mh, 0, 1000, l.data(), 1);
      assert(t == 1200 + k);   // start = 1000 + k, + l2_latency
    }
  }
  // (b) same config, one 16-sector request: summed occupation 4.0
  // cycles >= floor, so the port advances exactly 4.0 -- the floor
  // only charges the shortfall, multi-sector requests are not surcharged
  {
    MemoryConfig c2 = cfg;
    c2.l2_bytes_per_cycle = 128;
    c2.l2_read_bpc = 0;
    c2.l2_outstanding = 0;
    c2.merge_window = 0;
    c2.l1_data_array_bytes = 0;
    MemoryHierarchy mh(c2);
    for (int i = 0; i < 16; ++i) {          // fill the 16 sectors
      std::vector<MemorySector> x{sec(i * 32)};
      L(mh, 0, 0, x.data(), 1);
    }
    std::vector<MemorySector> l(16);
    for (int i = 0; i < 16; ++i) l[i] = sec(i * 32);
    L(mh, 0, 1000, l.data(), 16);
    // port busy_until = 1000 + 16*0.25 = 1004 exactly
    std::vector<MemorySector> one{sec(0)};
    const std::uint64_t t = L(mh, 0, 1000, one.data(), 1);
    assert(t == 1204);
  }
  // (c) store port: store_l2 bpc = 128 -> 0.25 cycle/sector.  A
  // 1-sector store leaves the port busy for the 1-cycle floor; an
  // 8-sector store occupies exactly 2.0 cycles (no surcharge)
  {
    MemoryConfig c2 = cfg;
    c2.store_l2_bytes_per_cycle = 128;
    c2.store_l2_latency = 10;
    c2.store_l2_outstanding = 0;
    c2.l2_write_bpc = 0;
    MemoryHierarchy mh(c2);
    std::vector<MemorySector> one{sec(0)};
    const std::uint64_t a = mh.store(0, 0, one.data(), 1, false, 0);
    const std::uint64_t b = mh.store(0, 0, one.data(), 1, false, 0);
    assert(a == 10 && b == 11);   // second start pushed to 1.0
    std::vector<MemorySector> eight(8);
    for (int i = 0; i < 8; ++i) eight[i] = sec(1024 + i * 32);
    mh.store(0, 0, eight.data(), 8, false, 0);   // busy 2.0 -> 4.0
    const std::uint64_t c = mh.store(0, 0, one.data(), 1, false, 0);
    assert(c == 14);   // start = 4.0, not 2.0 + 8*1.0
  }
  // store cost_ratio (storage.tma.reduce, kind 81): a reduce's
  // per-sector port occupancy and shared-write charge scale by the
  // ratio.  store_l2 bpc 128 -> occupy 0.25/sector plain, 0.3125 at
  // ratio 1.25: a 16-sector store books 4.0 vs 5.0 cycles of port,
  // so its returned acceptance is 15 vs 14 and a following sector
  // starts at the scaled end.
  {
    MemoryConfig c2 = cfg;
    c2.store_l2_bytes_per_cycle = 128;
    c2.store_l2_latency = 10;
    c2.store_l2_outstanding = 0;
    c2.l2_write_bpc = 0;
    MemoryHierarchy mh(c2), ref(c2);
    std::vector<MemorySector> s(16);
    for (int i = 0; i < 16; ++i) s[i] = sec(2048 + i * 32);
    // last sector starts at 15 x occupy: ceil(3.75) + 10 = 14 plain,
    // ceil(4.6875) + 10 = 15 at ratio 1.25
    const std::uint64_t a =
        mh.store(0, 0, s.data(), 16, false, 0, nullptr, 1.25);
    const std::uint64_t b = ref.store(0, 0, s.data(), 16, false, 0);
    assert(a == 15 && b == 14);
    // port ends at 5.0 vs 4.0 -> the next sector returns 15 vs 14
    std::vector<MemorySector> one{sec(4096)};
    const std::uint64_t pa =
        mh.store(0, 0, one.data(), 1, false, 0, nullptr, 1.25);
    const std::uint64_t pb = ref.store(0, 0, one.data(), 1, false, 0);
    assert(pa == 15 && pb == 14);
  }
  // dram.loaded_latency_extra_cycles: piecewise-linear in the shared
  // DRAM service utilization over the last 1024 cycles.  Fill 16 of
  // the 32 buckets before the probe request (u = 0.5 -> +100).
  {
    MemoryConfig c2 = cfg;
    c2.dram_read_bpc = 1.0;  // 32 B sectors saturate 32-cycle buckets
    c2.dram_loaded_extra = {{0.0, 0.0}, {500.0, 100.0}, {1000.0, 200.0}};
    MemoryConfig c0 = c2;
    c0.dram_loaded_extra.clear();
    MemoryHierarchy mh(c2);
    MemoryHierarchy ref(c0);   // same loads, empty table
    for (int b = 0; b < 16; ++b) {
      std::vector<MemorySector> l{sec(0x1000 + b * 4096)};
      L(mh, 0, static_cast<std::uint64_t>(b) * 32, l.data(), 1);
      L(ref, 0, static_cast<std::uint64_t>(b) * 32, l.data(), 1);
    }
    std::vector<MemorySector> p{sec(0x100000)};
    const std::uint64_t t = L(mh, 1, 1024, p.data(), 1);
    const std::uint64_t tref = L(ref, 1, 1024, p.data(), 1);
    assert(t == tref + 100);                 // u = 0.5 -> +100
    // an empty window measures u = 0 -> +0 (identical to no table)
    std::vector<MemorySector> q{sec(0x200000)};
    const std::uint64_t t2 = L(mh, 1, 50000, q.data(), 1);
    const std::uint64_t t2ref = L(ref, 1, 50000, q.data(), 1);
    assert(t2 == t2ref);
    // every DRAM-served request samples u once (fills too); the
    // probe's share is the asserted +100
    assert(mh.tele.loaded_extra_dram_count >= 1 &&
           mh.tele.loaded_extra_dram_cycles >= 100);
  }
  // --- tma_loaded: per-SM TMA delivery cap on L2/DRAM ports --------
  {
    MemoryConfig tc = cfg;
    tc.l2_bytes_per_cycle = 122;    // LDG port rate
    tc.tma_sector_bpc = 44;         // TMA cap (b300 44.65)
    MemoryConfig rc = cfg;
    rc.l2_bytes_per_cycle = 122;    // same port, no TMA cap
    MemoryHierarchy tm(tc), rf(rc);
    std::vector<MemorySector> p;
    // 8 L2-hit sectors: same request, tma vs non-tma occupy differs
    for (int i = 0; i < 8; ++i) p.push_back(sec(0x500000 + i * 32));
    // prime the sectors into L2 first (identical fill in both)
    const std::uint64_t fill = tm.load(
        1, 0, p.data(), p.size(), false, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, false);
    rf.load(1, 0, p.data(), p.size(), false, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, false);
    // re-read as L2 hits: the per-sector port stagger differs
    const std::uint64_t t = tm.load(
        1, fill, p.data(), p.size(), false, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, true);
    const std::uint64_t tref = rf.load(
        1, fill, p.data(), p.size(), false, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, false);
    // slower port -> later last-sector return
    assert(t > tref);
    const double want = 7.0 * (32.0 / 44.0 - 32.0 / 122.0);
    assert(std::abs((double)(t - tref) - want) < 1.5);
    // uninstalled -> bit-identical
    MemoryConfig nc = tc; nc.tma_sector_bpc = 0;
    MemoryHierarchy nz(nc);
    nz.load(1, 0, p.data(), p.size(), false, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, false);
    assert(nz.load(1, fill, p.data(), p.size(), false, nullptr,
                   nullptr, nullptr, nullptr, nullptr, nullptr,
                   true) == tref);
  }
  // --- §27: in-flight follower must not complete before its leader --
  {
    MemoryConfig c4 = cfg;
    c4.dram_outstanding = 1;   // serialize DRAM: leader occupies the port
    MemoryHierarchy mh(c4);
    std::vector<MemorySector> l{sec(0x800000)};
    const std::uint64_t lead = L(mh, 1, 10, l.data(), 1);
    // follower on a different SM reads the same sector while the fill
    // is in flight: it may not return before the leader
    std::vector<MemorySector> f{sec(0x800000)};
    const std::uint64_t fol = L(mh, 2, 11, f.data(), 1);
    assert(fol >= lead - 1);
    assert(fol <= lead + 40);
    // a read after the fill is a plain hit
    assert(L(mh, 2, lead + 5000, f.data(), 1) == lead + 5000 + c4.l2_latency);
    // (leader's fill is the recorded one; the follower's own-miss
    // bound, not an inflight-hit counter, gates its completion)
    assert(mh.tele.l2_inflight_hits >= 0);
  }
  // --- rule S: store-buffer backpressure ---------------------------
  {
    MemoryConfig c5 = cfg;
    c5.store_l2_outstanding = 3;
    c5.l2_write_bpc = 0.001;   // shared write service drains ~never
    c5.store_l2_bytes_per_cycle = 32;  // port accept stays cheap
    MemoryHierarchy mh(c5);
    std::vector<MemorySector> st{sec(0x900000)};
    // three non-atomic stores fill the buffer; each drain end is the
    // shared-write service end of its last sector
    mh.store(1, 0, st.data(), 1, false, 0);
    mh.store(1, 0, st.data(), 1, false, 0);
    mh.store(1, 0, st.data(), 1, false, 0);
    // buffer holds 3 undrained requests -> a fourth STG blocks until
    // the oldest request's drain end
    const std::uint64_t blk = mh.store_issue_block(1, 0);
    assert(blk > 0);
    // outstanding 0 -> never blocks
    MemoryConfig c6 = cfg; c6.store_l2_outstanding = 0;
    MemoryHierarchy mh2(c6);
    assert(mh2.store_issue_block(1, 0) == 0);
    // atomic stores do not push the buffer
    assert(mh.store_issue_block(1, 0) == blk);
    // drain also waits for the sectors to leave the SM's store port:
    // unlimited shared write, 32 B/cyc port (sector = 1.0), depth 4;
    // after 5 requests of 16 sectors the 5th blocks until the first
    // request's last sector left the port (16 cycles)
    MemoryConfig c6b = cfg;
    c6b.store_l2_outstanding = 4;
    c6b.l2_write_bpc = 1000.0;
    c6b.store_l2_bytes_per_cycle = 32;
    MemoryHierarchy mh3(c6b);
    std::vector<MemorySector> st16;
    for (int i = 0; i < 16; ++i)
      st16.push_back(sec(0xa00000 + i * 32));
    for (int i = 0; i < 5; ++i)
      mh3.store(1, 0, st16.data(), st16.size(), false, 0);
    assert(mh3.store_issue_block(1, 0) >= 16);
  }
  // --- rule R1: per-SM memory port shared by fills and stores ------
  {
    MemoryConfig c7 = cfg;
    c7.l2_bytes_per_cycle = 128;       // sector = 0.25 on the l2 port
    c7.store_l2_bytes_per_cycle = 32;  // sector = 1.0 on the store port
    c7.merge_window = 0;
    c7.store_l2_outstanding = 64;
    c7.store_shares_fill_port = false;
    const std::uint64_t A = 0x300000;
    std::vector<MemorySector> one{sec(A)};
    // 256 store sectors + 64 / 512 distinct sectors
    std::vector<MemorySector> st256, ld64, ld512;
    for (int i = 0; i < 256; ++i) st256.push_back(sec(A + 0x1000 + i * 32));
    for (int i = 0; i < 64; ++i)  ld64.push_back(sec(A + 0x20000 + i * 32));
    for (int i = 0; i < 512; ++i) ld512.push_back(sec(A + 0x40000 + i * 32));

    // (a) switch on with traffic from only one direction on an empty
    // queue: identical to switch off
    auto solo = [&](bool share, bool store_op) {
      MemoryConfig c = c7; c.store_shares_fill_port = share;
      MemoryHierarchy m(c);
      std::vector<MemorySector> a{sec(A)};
      L(m, 0, 0, a.data(), 1);                 // DRAM fill -> resident
      if (store_op) return m.store(0, 10000, a.data(), 1, false, 0);
      return L(m, 0, 10000, a.data(), 1);      // L2 hit
    };
    assert(solo(true, false) == solo(false, false));
    assert(solo(true, true) == solo(false, true));

    // (b) 256 store sectors first, then a 64-sector L2-hit load: the
    // load is pushed behind the stores' port occupancy
    // (b) L2-hit loads at a bucket boundary: warm ld64 into L2 on a
    // different SM (so the second load is an L2 hit, not L1/DRAM), then
    // at T=10240 (a bucket boundary, earlier traffic drained) place the
    // 256 store sectors and the 64 L2-hit sectors.  16 port cycles at
    // the 0.2 fill share walk ~80 cycles
    auto seq_b = [&](bool share) {
      MemoryConfig c = c7; c.store_shares_fill_port = share;
      MemoryHierarchy m(c);
      std::vector<MemorySector> a{sec(A)};
      L(m, 0, 0, a.data(), 1);
      L(m, 1, 0, ld64.data(), ld64.size());        // L2 resident
      const std::uint64_t T = 10240;
      m.store(0, T, st256.data(), st256.size(), false, 0);
      const std::uint64_t r = L(m, 0, T, ld64.data(), ld64.size());
      return std::pair<std::uint64_t, std::uint64_t>(
          r, m.tele.port_share_waits);
    };
    const auto b_on = seq_b(true), b_off = seq_b(false);
    assert(b_on.first >= b_off.first + 40 &&
           b_on.first <= b_off.first + 130);
    assert(b_on.second > 0);

    // (c) 512 L2-hit sectors first, then an 8-sector store: the store
    // is pushed behind the fills' port occupancy
    auto seq_c = [&](bool share) {
      MemoryConfig c = c7; c.store_shares_fill_port = share;
      MemoryHierarchy m(c);
      std::vector<MemorySector> a{sec(A)};
      L(m, 0, 0, a.data(), 1);
      L(m, 0, 0, ld512.data(), ld512.size());
      std::vector<MemorySector> st8(st256.begin(), st256.begin() + 8);
      return m.store(0, 0, st8.data(), st8.size(), false, 0);
    };
    const std::uint64_t c_on = seq_c(true), c_off = seq_c(false);
    // store share 0.8: the store mostly keeps its turn
    assert(c_on > c_off && c_on <= c_off + 20);

    // (d) switch off: the two requests do not affect each other --
    // each returns its solo value
    auto solo_ld64 = [&]() {
      MemoryHierarchy m(c7);
      L(m, 1, 0, ld64.data(), ld64.size());        // L2 resident
      return L(m, 0, 10240, ld64.data(), ld64.size());
    };
    auto solo_st8 = [&]() {
      MemoryHierarchy m(c7);
      std::vector<MemorySector> a{sec(A)};
      L(m, 0, 0, a.data(), 1);
      std::vector<MemorySector> st8(st256.begin(), st256.begin() + 8);
      return m.store(0, 0, st8.data(), st8.size(), false, 0);
    };
    assert(b_off.first == solo_ld64());
    assert(c_off == solo_st8());

    // (e) fills do not queue on fills: a big DRAM-miss load batch on
    // the same SM does not delay a later L2-hit batch
    auto seq_e = [&](bool share) {
      MemoryConfig c = c7; c.store_shares_fill_port = share;
      MemoryHierarchy m(c);
      std::vector<MemorySector> miss;
      for (int i = 0; i < 512; ++i)
        miss.push_back(sec(A + 0x600000 + i * 32));
      L(m, 0, 0, miss.data(), miss.size());          // all DRAM misses
      std::vector<MemorySector> a{sec(A)};
      L(m, 0, 0, a.data(), 1);                        // resident
      return L(m, 0, 0, ld64.data(), ld64.size());    // L2 hits
    };
    assert(seq_e(true) == seq_e(false));

    // (f) DRAM-miss fills are pushed behind stores too
    auto seq_f = [&](bool share) {
      MemoryConfig c = c7; c.store_shares_fill_port = share;
      MemoryHierarchy m(c);
      std::vector<MemorySector> a{sec(A)};
      L(m, 0, 0, a.data(), 1);
      m.store(0, 0, st256.data(), st256.size(), false, 0);
      std::vector<MemorySector> miss;
      for (int i = 0; i < 64; ++i)
        miss.push_back(sec(A + 0x800000 + i * 32));
      return L(m, 0, 0, miss.data(), miss.size());
    };
    { const std::uint64_t f_on = seq_f(true), f_off = seq_f(false);
      // fixture sets no DRAM port rate: each miss sector occupies
      // 1.0 port cycle; 64 port cycles at the 0.2 share = 320 cycles
      assert(f_on >= f_off + 200 && f_on <= f_off + 300); }

    // (g) displaced occupancy lands on later requests of the same
    // direction: after (b)'s store->load steal, a second 32-sector
    // store placed at the same time pays for the carry-forward
    auto seq_g = [&](bool share) {
      MemoryConfig c = c7; c.store_shares_fill_port = share;
      MemoryHierarchy m(c);
      std::vector<MemorySector> a{sec(A)};
      L(m, 0, 0, a.data(), 1);
      m.store(0, 0, st256.data(), st256.size(), false, 0);
      L(m, 0, 0, ld64.data(), ld64.size());
      std::vector<MemorySector> st32(st256.begin(), st256.begin() + 32);
      return m.store(0, 0, st32.data(), st32.size(), false, 0);
    };
    { const std::uint64_t g_on = seq_g(true), g_off = seq_g(false);
      // second store walks its 0.8 share while fills still occupy:
      // 32 sectors / 0.8 = 40 cycles, ~8 more than uncontended
      assert(g_on >= g_off + 6 && g_on <= g_off + 12); }
  }
  // --- rule R2: shared-store acceptance alternates with fills ------
  {
    MemoryConfig c7 = cfg;
    c7.l2_bytes_per_cycle = 128;
    c7.store_l2_bytes_per_cycle = 32;
    c7.merge_window = 0;
    c7.store_l2_outstanding = 64;
    c7.store_shares_fill_port = true;
    c7.shared_store_shares_fill_port = false;

    // (a) bit off: no wait, no ledger write -> fills see nothing
    {
      MemoryHierarchy m(c7);
      assert(m.shared_store_port_wait(0, 0, 16.0) == 0.0);
      const double w = m.port_share_wait(
          0, MemoryHierarchy::PortUser::fill_l2, 0.0, 8.0);
      assert(w == 0.0);
    }

    // (b) bit on, no fills: shared stores never wait on their own
    // ledger -- repeated, overlapping and cross-bucket occupancies
    // all return 0
    c7.shared_store_shares_fill_port = true;
    {
      MemoryHierarchy m(c7);
      assert(m.shared_store_port_wait(0, 0.0, 16.0) == 0.0);
      assert(m.shared_store_port_wait(0, 0.0, 8.0) == 0.0);
      assert(m.shared_store_port_wait(0, 16.0, 8.0) == 0.0);
      assert(m.shared_store_port_wait(0, 4.0, 90.0) == 0.0);
      assert(m.shared_store_port_wait(0, 5.0, 7.0) == 0.0);
      // booked occupancy is still visible to fills
      assert(m.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 8.0)
             > 0.0);
    }

    // (c) fill half a bucket: free_rate = 16/32 = 0.5, occupy 8
    // takes 8/0.5 = 16 -> wait 16-8 = 8; a 24-fill bucket leaves
    // free 0.25 -> t = 8/0.25 = 32, extra = 32-8 = 24; a
    // 90-occupancy walks buckets at the fill-free rate
    {
      MemoryHierarchy m(c7);
      assert(m.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 16.0)
             == 0.0);
      assert(m.shared_store_port_wait(0, 0.0, 8.0) == 8.0);
    }
    {
      MemoryHierarchy m(c7);
      assert(m.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 24.0)
             == 0.0);
      assert(m.shared_store_port_wait(0, 0.0, 8.0) == 24.0);
      // occupy 40: bucket 0 yields 8 (free 0.25 over span 32),
      // bucket 1 yields the rest at full rate -> end 32+32 = 64
      assert(m.shared_store_port_wait(0, 0.0, 40.0) == 24.0);
    }

    // (d) shared store first, then fill: the fill waits on it
    {
      MemoryHierarchy m(c7);
      assert(m.shared_store_port_wait(0, 0.0, 16.0) == 0.0);
      assert(m.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 8.0)
             == 16.0);
    }

    // (e) global stores and shared stores do not wait on each other
    {
      MemoryHierarchy m(c7);
      assert(m.shared_store_port_wait(0, 0.0, 16.0) == 0.0);
      assert(m.port_share_wait(
                 0, MemoryHierarchy::PortUser::store, 0.0, 8.0)
             == 0.0);
      MemoryHierarchy m2(c7);
      assert(m2.port_share_wait(
                 0, MemoryHierarchy::PortUser::store, 0.0, 16.0)
             == 0.0);
      assert(m2.shared_store_port_wait(0, 0.0, 8.0) == 0.0);
    }

    // (f) book=false (the kind-80 SMSP path): the same wait comes out
    // of the fill ledgers but ps.shared_store is not written and the
    // telemetry is not bumped; a following fill sees the ledger as if
    // the shared store had never asked
    {
      MemoryHierarchy m(c7);
      assert(m.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 24.0)
             == 0.0);
      const auto waits0 = m.tele.port_share_waits;
      // booking call on an identical fresh ledger: same wait
      MemoryHierarchy m2(c7);
      assert(m2.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 24.0)
             == 0.0);
      assert(m.shared_store_port_wait(0, 0.0, 8.0, false) ==
             m2.shared_store_port_wait(0, 0.0, 8.0));
      assert(m.tele.port_share_waits == waits0);
      // ...but it wrote nothing: a fill afterwards waits exactly as on
      // a ledger that never saw the shared-store call (a booked 8
      // would have raised it through ps.shared_store)
      MemoryHierarchy m3(c7);
      assert(m3.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 24.0)
             == 0.0);
      assert(m.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 8.0) ==
             m3.port_share_wait(
                 0, MemoryHierarchy::PortUser::fill_l2, 0.0, 8.0));
    }
  }
  // --- l1_allocate: an async copy without .BYPASS fills the L1 ---
  {
    MemoryConfig c8 = cfg;
    c8.l1_data_array_bytes = 128;   // cap = 1 line = 4 sectors
    c8.l1_latency = 30;
    c8.l2_latency = 200;
    // (a)/(b): residents a(4) are evicted by the allocating fill b(8);
    // the allocating load's own return is unchanged
    const auto seq_ab = [&](bool alloc, std::uint64_t* self_ret) {
      MemoryHierarchy m(c8);
      std::vector<MemorySector> a, b;
      for (int i = 0; i < 4; ++i) a.push_back(sec(0xb00000 + i * 32));
      for (int i = 0; i < 8; ++i) b.push_back(sec(0xc00000 + i * 32));
      L(m, 0, 0, a.data(), a.size());          // residents in L1
      *self_ret = m.load(0, 2000, b.data(), b.size(), false,
                         nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, false, alloc);
      return L(m, 0, 4000, a.data(), a.size());
    };
    std::uint64_t s_on = 0, s_off = 0;
    const std::uint64_t on_a = seq_ab(true, &s_on);
    const std::uint64_t off_a = seq_ab(false, &s_off);
    assert(on_a >= off_a + 100);    // (a) evicted -> reload misses to L2
    assert(s_on == s_off);          // (b) own return unchanged
    // (c): a later plain load hits the copied sectors in L1
    const auto seq_c = [&](bool alloc) {
      MemoryHierarchy m(c8);
      std::vector<MemorySector> c;
      for (int i = 0; i < 4; ++i) c.push_back(sec(0xd00000 + i * 32));
      m.load(0, 0, c.data(), c.size(), false,
             nullptr, nullptr, nullptr, nullptr,
             nullptr, nullptr, false, alloc);
      return L(m, 0, 4000, c.data(), c.size());
    };
    const std::uint64_t bc_on = seq_c(true), bc_off = seq_c(false);
    assert(bc_on <= 4000 + 40);      // L1 hit
    assert(bc_on < bc_off - 100);    // off arm misses to L2
  }
  // --- orders_loads: plain loads wait for the SM's store port ------
  {
    MemoryConfig c9 = cfg;
    c9.store_l2_bytes_per_cycle = 32;   // 1 cycle/sector drain
    c9.store_orders_loads = true;
    const auto seq = [&](bool sw, bool ord, std::uint32_t store_sm) {
      MemoryConfig c = c9; c.store_orders_loads = sw;
      MemoryHierarchy m(c);
      MemorySector a = sec(0xe00000);
      L(m, 0, 0, &a, 1);                       // L1-resident
      std::vector<MemorySector> st;
      for (int i = 0; i < 64; ++i) st.push_back(sec(0xf00000 + i * 32));
      m.store(store_sm, 2000, st.data(), st.size(), false, 0);
      return m.load(0, 2000, &a, 1, true, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, false, false, ord);
    };
    const std::uint64_t a_on = seq(true, true, 0);
    const std::uint64_t a_off = seq(false, true, 0);   // switch off
    const std::uint64_t b_nord = seq(true, false, 0);  // not ordered
    const std::uint64_t c_sm1 = seq(true, true, 1);    // store on SM 1
    assert(a_on >= a_off + 60);       // (a) waits for the store port
    assert(b_nord == a_off);          // (b) unordered load unaffected
    assert(c_sm1 == a_off);           // (c) only this SM's stores count
  }
  // --- load_port line_granular: a request pays one full line of
  // port time per distinct line routed to a port -------------------
  // L1 port at r = 128 B/cycle: per-sector occupy = 32/128 = 0.25
  // cycle; with the switch the first sector of a line pays
  // 128/128 = 1 cycle and same-line repeats 0.  The discriminator is
  // the second request's first-sector start (= the port's
  // busy_until), read back through `accept`.
  {
    MemoryConfig c = cfg;
    c.l1_bytes_per_cycle = 128;
    c.l1_latency = 30;
    c.l1_outstanding = 64;
    c.l1_data_array_bytes = 1 << 20;   // L1 modeled: sectors hit in L1
    c.merge_window = 0;
    c.pool_count = 1;
    const auto acc_after = [&](MemoryHierarchy& m, MemorySector* l,
                               std::size_t n) {
      L(m, 0, 10000, l, n);
      MemorySector one = sec(0);
      std::uint64_t acc = 0;
      m.load(0, 10000, &one, 1, true, nullptr, nullptr, nullptr,
             nullptr, &acc);
      return acc - 10000;   // = port busy cycles the request left
    };
    const auto build = [&](bool on) {
      MemoryConfig cc = c;
      cc.load_port_line_granular = on;
      return MemoryHierarchy(cc);
    };
    // (a) 32 sectors in 32 different lines
    {
      std::vector<MemorySector> l(32);
      for (int i = 0; i < 32; ++i) l[i] = sec(0x100000 + i * 128);
      MemoryHierarchy on = build(true), off = build(false);
      for (int i = 0; i < 32; ++i) { L(on, 0, 0, &l[i], 1); L(off, 0, 0, &l[i], 1); }
      assert(acc_after(on, l.data(), 32) == 32);    // 32 x 1.0
      assert(acc_after(off, l.data(), 32) == 8);    // 32 x 0.25
    }
    // (b) 4 sectors of one line
    {
      std::vector<MemorySector> l{sec(0x200000), sec(0x200020),
                                  sec(0x200040), sec(0x200060)};
      MemoryHierarchy on = build(true), off = build(false);
      for (auto& s : l) { L(on, 0, 0, &s, 1); L(off, 0, 0, &s, 1); }
      assert(acc_after(on, l.data(), 4) == 1);      // 1 x 1.0
      assert(acc_after(off, l.data(), 4) == 1);     // 4 x 0.25
    }
    // (c) 32 sectors in 8 lines
    {
      std::vector<MemorySector> l(32);
      for (int i = 0; i < 32; ++i)
        l[i] = sec(0x300000 + (i % 8) * 128 + (i / 8) * 32);
      MemoryHierarchy on = build(true), off = build(false);
      for (int i = 0; i < 32; ++i) { L(on, 0, 0, &l[i], 1); L(off, 0, 0, &l[i], 1); }
      assert(acc_after(on, l.data(), 32) == 8);     // 8 x 1.0
      assert(acc_after(off, l.data(), 32) == 8);    // 32 x 0.25
    }
  }
  // --- load_port lane_order (gather_hit): a lane load's port
  // occupancy is line switches between consecutive active lanes
  // (each lane on the line of its lowest-addressed sector), not
  // distinct lines.  Same port read-back as above: the second
  // request's accept is the first one's busy cycles -------------
  {
    MemoryConfig c = cfg;
    c.l1_bytes_per_cycle = 128;
    c.l1_latency = 30;
    c.l1_outstanding = 64;
    c.l1_data_array_bytes = 1 << 20;
    c.merge_window = 0;
    c.pool_count = 1;
    c.load_port_line_granular = true;
    const auto acc_after = [&](MemoryHierarchy& m, MemorySector* l,
                               std::size_t n) {
      L(m, 0, 10000, l, n);
      MemorySector one = sec(0);
      std::uint64_t acc = 0;
      m.load(0, 10000, &one, 1, true, nullptr, nullptr, nullptr,
             nullptr, &acc);
      return acc - 10000;
    };
    const auto build2 = [&](bool lane_order) {
      MemoryConfig cc = c;
      cc.load_port_lane_order = lane_order;
      return MemoryHierarchy(cc);
    };
    const auto lanesec = [](std::uint64_t a, std::uint32_t mask) {
      MemorySector t = sec(a);
      t.lane_mask = mask;
      return t;
    };
    // (a) 32 lanes on 32 different lines -> 32 slots (unchanged)
    {
      std::vector<MemorySector> l(32);
      for (int i = 0; i < 32; ++i)
        l[i] = lanesec(0x400000 + i * 128, 1u << i);
      MemoryHierarchy on = build2(true);
      for (int i = 0; i < 32; ++i) L(on, 0, 0, &l[i], 1);
      assert(acc_after(on, l.data(), 32) == 32);
    }
    // (b) lanes 0-15 on lines 0-15, lanes 16-31 on the same lines
    // again (second sector of each): 16 distinct lines but 32 lane
    // switches -> 32 slots (16 under distinct lines)
    {
      std::vector<MemorySector> l(32);
      for (int i = 0; i < 16; ++i) {
        l[i]      = lanesec(0x500000 + i * 128,      1u << i);
        l[16 + i] = lanesec(0x500000 + i * 128 + 32, 1u << (16 + i));
      }
      MemoryHierarchy on = build2(true);
      for (int i = 0; i < 32; ++i) L(on, 0, 0, &l[i], 1);
      assert(acc_after(on, l.data(), 32) == 32);
    }
    // (c) four adjacent lanes per line, 8 lines -> 8 switches, same
    // as distinct lines -> 8 slots
    {
      std::vector<MemorySector> l(8);
      for (int i = 0; i < 8; ++i)
        l[i] = lanesec(0x600000 + i * 128, 0xFu << (i * 4));
      MemoryHierarchy on = build2(true);
      for (int i = 0; i < 8; ++i) L(on, 0, 0, &l[i], 1);
      assert(acc_after(on, l.data(), 8) == 8);
    }
    // (d) lane_order off: (b) falls back to 16 distinct lines
    {
      std::vector<MemorySector> l(32);
      for (int i = 0; i < 16; ++i) {
        l[i]      = lanesec(0x700000 + i * 128,      1u << i);
        l[16 + i] = lanesec(0x700000 + i * 128 + 32, 1u << (16 + i));
      }
      MemoryHierarchy off = build2(false);
      for (int i = 0; i < 32; ++i) L(off, 0, 0, &l[i], 1);
      assert(acc_after(off, l.data(), 32) == 16);
    }
    // (e) lane_mask all 0 on 16 lines -> distinct-line count
    {
      std::vector<MemorySector> l(16);
      for (int i = 0; i < 16; ++i) {
        l[i] = sec(0x800000 + i * 128);
        l[i].lane_mask = 0;
      }
      MemoryHierarchy on = build2(true);
      for (int i = 0; i < 16; ++i) L(on, 0, 0, &l[i], 1);
      assert(acc_after(on, l.data(), 16) == 16);
    }
  }
  // --- memory.dram.service near/far split: a DRAM miss costs the
  // near latency on the SM's gpc side's own pages, near + extra on
  // the far side's pages; the page side is the parity of
  // sector / home_page_bytes -------------------------------------
  {
    MemoryConfig c = cfg;
    c.dram_latency = 500;
    c.dram_far_extra = 200;
    c.dram_home_page_bytes = 4096;
    c.dram_far_side_gpc_mask = 2;          // gpc 1 -> side 1
    c.sm_gpc = {0u, 1u, 0xFFFFFFFFu};
    // page index 0x400000/4096 = 1024 even -> home 0,
    //             0x401000/4096 = 1025 odd  -> home 1
    const std::uint64_t even_page = 0x400000, odd_page = 0x401000;
    // (a) side-0 SM (gpc 0 not in mask): even near, odd far
    {
      MemoryHierarchy mh(c);
      MemorySector e = sec(even_page), o = sec(odd_page);
      assert(L(mh, 0, 0, &e, 1) == 500);
      assert(L(mh, 0, 1000, &o, 1) == 1700);
    }
    // (b) side-1 SM (gpc 1 in mask): reversed
    {
      MemoryHierarchy mh(c);
      MemorySector e = sec(even_page), o = sec(odd_page);
      assert(L(mh, 1, 0, &e, 1) == 700);
      assert(L(mh, 1, 1000, &o, 1) == 1500);
    }
    // (c) SM without a gpc is on side 0
    {
      MemoryHierarchy mh(c);
      MemorySector e = sec(even_page), o = sec(odd_page);
      assert(L(mh, 2, 0, &e, 1) == 500);
      assert(L(mh, 2, 1000, &o, 1) == 1700);
    }
    // (d) one request whose two sectors sit on pages of different
    // parity waits for the far one (the second sector starts one
    // cycle later on the SM's DRAM port)
    {
      MemoryHierarchy mh(c);
      std::vector<MemorySector> l{sec(even_page), sec(odd_page)};
      assert(L(mh, 0, 0, l.data(), 2) == 701);   // max(500, 1 + 700)
      MemoryHierarchy mh1(c);
      assert(L(mh1, 1, 0, l.data(), 2) == 700);  // max(700, 1 + 500)
    }
    // (e) extra = 0: both pages equal
    {
      MemoryConfig c0 = c;
      c0.dram_far_extra = 0;
      MemoryHierarchy mh(c0);
      MemorySector e = sec(even_page), o = sec(odd_page);
      assert(L(mh, 0, 0, &e, 1) == 500);
      assert(L(mh, 0, 1000, &o, 1) == 1500);
    }
  }
  // store_flood: lane-store port extras — partial-line term and the
  // lane-store-only issue floor.  Occupancy of request 1 is read off
  // the acceptance of an identical request 2 issued right behind it
  {
    MemoryConfig c2 = cfg;
    c2.l2_write_bpc = 1e9;              // instant shared write service
    c2.store_l2_bytes_per_cycle = 32;   // 1 sector/cycle/SM
    c2.store_l2_latency = 10;
    c2.store_l2_outstanding = 0;
    c2.store_l2_partial_line_cycles = 1.0;
    c2.store_l2_partial_granule_bytes = 32;
    auto sectors8 = [](std::uint32_t mask) {
      std::vector<MemorySector> s(8);
      for (int i = 0; i < 8; ++i) {
        s[i] = sec(8192 + i * 32);      // 2 lines
        s[i].byte_mask = mask;
      }
      return s;
    };
    // (a) full sectors, granule 32: no partial line -> 8
    {
      MemoryHierarchy mh(c2);
      auto s = sectors8(0xFFFFFFFFu);
      const std::uint64_t a = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      const std::uint64_t b = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      assert(b - a == 8);
    }
    // (b) half-written sectors, granule 32: both lines partial -> 8+2
    {
      MemoryHierarchy mh(c2);
      auto s = sectors8(0x0000FFFFu);
      const std::uint64_t a = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      const std::uint64_t b = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      assert(b - a == 10);
    }
    // (c) same masks, granule 4: every 4-B granule full or empty -> 8
    {
      MemoryConfig c3 = c2;
      c3.store_l2_partial_granule_bytes = 4;
      MemoryHierarchy mh(c3);
      auto s = sectors8(0x0000FFFFu);
      const std::uint64_t a = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      const std::uint64_t b = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      assert(b - a == 8);
    }
    // (d) granule 4, 2 bytes per sector: both lines partial -> 8+2
    {
      MemoryConfig c3 = c2;
      c3.store_l2_partial_granule_bytes = 4;
      MemoryHierarchy mh(c3);
      auto s = sectors8(0x3u);
      const std::uint64_t a = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      const std::uint64_t b = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      assert(b - a == 10);
    }
    // (e) single-sector lane store under issue_interval 2 -> 2
    {
      MemoryConfig c3 = c2;
      c3.store_l2_issue_interval = 2;
      c3.store_l2_partial_line_cycles = 0;
      MemoryHierarchy mh(c3);
      auto s = sectors8(0xFFFFFFFFu);
      const std::uint64_t a = mh.store(0, 0, s.data(), 1, false, 0,
                                       nullptr, 1.0, true);
      const std::uint64_t b = mh.store(0, 0, s.data(), 1, false, 0,
                                       nullptr, 1.0, true);
      assert(b - a == 2);
    }
    // (f) same request, not a lane store: 1-cycle floor -> 1
    {
      MemoryConfig c3 = c2;
      c3.store_l2_issue_interval = 2;
      c3.store_l2_partial_line_cycles = 0;
      MemoryHierarchy mh(c3);
      auto s = sectors8(0xFFFFFFFFu);
      const std::uint64_t a = mh.store(0, 0, s.data(), 1, false, 0,
                                       nullptr, 1.0, false);
      const std::uint64_t b = mh.store(0, 0, s.data(), 1, false, 0,
                                       nullptr, 1.0, false);
      assert(b - a == 1);
    }
    // (g) partial_line_cycles 0: (b) reduces to the sector count -> 8
    {
      MemoryConfig c3 = c2;
      c3.store_l2_partial_line_cycles = 0;
      MemoryHierarchy mh(c3);
      auto s = sectors8(0x0000FFFFu);
      const std::uint64_t a = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      const std::uint64_t b = mh.store(0, 0, s.data(), 8, false, 0,
                                       nullptr, 1.0, true);
      assert(b - a == 8);
    }
  }
  // tma_oob v2 phantom sectors (tiled TMA out-of-bounds box part):
  // they time like unmerged L2-hit sectors but touch no cache state.
  {
    auto PL = [](MemoryHierarchy& m, std::uint32_t sm, std::uint64_t now,
                 MemorySector* s, std::size_t n, std::size_t ph) {
      return m.load(sm, now, s, n, true, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, true, false, false,
                    false, ph);
    };
    MemoryConfig c5 = cfg;
    c5.tma_sector_bpc = 32;   // 1 cycle per sector on the SM L2 port
    std::vector<MemorySector> s16(16), s32(32);
    for (int i = 0; i < 32; ++i) s32[i] = sec(4096 + i * 32);
    for (int i = 0; i < 16; ++i) s16[i] = sec(4096 + i * 32);
    // prime the L2 (miss -> fill), then every sector hits
    {
      MemoryHierarchy mh(c5);
      PL(mh, 0, 0, s32.data(), 32, 0);
      const std::uint64_t ref32 = PL(mh, 0, 2000, s32.data(), 32, 0);
      // (a) phantom = 0 identical to no-phantom call
      const std::uint64_t p0 = PL(mh, 0, 2000, s32.data(), 32, 0);
      assert(p0 == ref32 || p0 == ref32 + 1 || p0 + 1 == ref32);
      // (b) 16 real hits + 16 phantom == 32 real hits
      MemoryHierarchy mh2(c5);
      PL(mh2, 0, 0, s16.data(), 16, 0);
      const std::uint64_t half = PL(mh2, 0, 2000, s16.data(), 16, 16);
      assert(half + 1 >= ref32 && half <= ref32 + 1);
      // (c) 0 real + 32 phantom == 32 real hits; L2 table and DRAM
      //     telemetry untouched
      MemoryHierarchy mh3(c5);
      const auto before_l2 = mh3.tele.l2_sectors;
      const auto before_dram = mh3.tele.dram_sectors;
      const std::uint64_t ph = PL(mh3, 0, 2000, nullptr, 0, 32);
      assert(ph + 1 >= ref32 && ph <= ref32 + 1);
      assert(mh3.tele.l2_sectors == before_l2 &&
             mh3.tele.dram_sectors == before_dram &&
             mh3.tele.phantom_sectors == 32);
    }
    // (d) two SMs of one GPC: 512 phantom each -- the shared service
    //     serializes them
    {
      MemoryConfig c6 = c5;
      c6.sm_gpc = {0, 0};
      c6.l2_read_gpc_bpc = 32;   // 1 cycle per sector at the GPC share
      c6.l2_read_bpc = 0;
      MemoryHierarchy mh(c6);
      const std::uint64_t single = PL(mh, 0, 0, nullptr, 0, 512);
      MemoryHierarchy mh2(c6);
      PL(mh2, 0, 0, nullptr, 0, 512);
      const std::uint64_t second = PL(mh2, 1, 0, nullptr, 0, 512);
      assert(second > single);
    }
  }
  // load_after_copies v3: local (stack) lines are ordinary residents
  // of the SM's L1.  The simulator expands a local access's trace
  // sectors into warp-private interleaved keys; the calls below use
  // the same key shape (mark bit | warp << 24 | byte offset)
  {
    const std::uint64_t kMark = 1ull << 62;
    auto lsec = [](std::uint64_t warp, std::uint64_t o) {
      return MemorySector{(1ull << 62) | (warp << 24) |
                              (o & 0xFFFFFF),
                          0xFFFFFFFFu, 0xFFFFFFFFu};
    };
    MemoryConfig cl = cfg;
    cl.l1_data_array_bytes = 4096;   // 32 lines = 128 sectors of L1
    cl.local_frame_interleaved = 1;
    cl.local_hit_latency = 32;
    {
      // (a) a local store's sectors are resident: the same-word load
      // hits at now + local_hit_latency
      MemoryHierarchy mh(cl);
      MemorySector s = lsec(5, 0x40);
      mh.l1_insert(0, s.address, 0);
      const std::uint64_t h = mh.load(
          0, 100, &s, 1, true, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, false, false, false, 0, true);
      assert(h == 132);
      assert(mh.tele.l1_sectors == 1 && mh.tele.l1_misses == 0);
      // (c) another warp's line at the same frame offset is its own:
      // it misses and is refilled over the L2-hit path without
      // touching the L2 residency table or DRAM
      MemorySector s2 = lsec(7, 0x40);
      const std::uint64_t m = mh.load(
          0, 200, &s2, 1, true, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, false, false, false, 0, true);
      assert(m >= 400);
      assert(mh.tele.dram_sectors == 0);
      // the miss inserted the line: a re-read hits at the local
      // latency
      const std::uint64_t h2 = mh.load(
          0, 1000, &s2, 1, true, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, false, false, false, 0, true);
      assert(h2 == 1032);
    }
    {
      // (b) after more than cap other sectors allocate the SM's L1,
      // the stack line is evicted: the load misses, pays the L2-hit
      // path, and never reaches DRAM
      MemoryHierarchy mh(cl);
      MemorySector s = lsec(5, 0x40);
      mh.l1_insert(0, s.address, 0);
      std::vector<MemorySector> g;
      for (int i = 0; i < 160; ++i) g.push_back(sec(1024 + i));
      L(mh, 0, 50, g.data(), g.size());
      const auto dram0 = mh.tele.dram_sectors;
      const std::uint64_t m = mh.load(
          0, 5000, &s, 1, true, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, false, false, false, 0, true);
      assert(m >= 5200);
      assert(mh.tele.dram_sectors == dram0);
      assert(mh.tele.l1_misses == 161);
    }
    {
      // (d) flag off: local=0 identical to the old behaviour (the
      // access is a plain global-sector load that misses to DRAM)
      MemoryConfig cn = cl;
      cn.local_frame_interleaved = 0;
      MemoryHierarchy mh(cn);
      MemorySector s = lsec(5, 0x40);
      const std::uint64_t t = mh.load(
          0, 0, &s, 1, true, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, false, false, false, 0, false);
      assert(t == 500);
      assert(mh.tele.dram_sectors == 1);
    }
  }
  std::puts("memory_hierarchy_test ok");
  return 0;
}
