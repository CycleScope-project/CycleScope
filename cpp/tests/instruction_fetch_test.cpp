// instruction fetch model — docs/en/design/engine.md (one field set for both
// devices: per-level stream cadences, launch prefetch chain, staged
// sequential prefetch).
#include <cassert>
#include <cstdio>
#include "core/instruction_fetch.hpp"

using cyclescope::FetchConfig;
using cyclescope::InstructionFetch;

// drive one warp at 1 IPC over `nlines` lines starting at `base`
// (8 instructions x 16 B per line); when ready() returns a future
// cycle the warp waits for it, else it issues next cycle
static std::uint64_t run_lines(InstructionFetch& f, std::uint64_t base,
                               std::uint64_t nlines, std::uint64_t now,
                               std::uint32_t warp = 0) {
  for (std::uint64_t i = 0; i < nlines; ++i)
    for (int j = 0; j < 8; ++j) {
      const std::uint64_t r =
          f.ready(0, 0, 0, base + (i * 8 + j) * 16, now, warp);
      if (r > now) now = r;
      ++now;
    }
  return now;
}

int main() {
  // (a) launch prefetch chain: line j arrives at (j+1)*stream_l1 as
  // prefetch_only. A demand at now=0 pays one L1 slot (16) and beats
  // the chain, so the demand claims the chain arrival (<= one slot);
  // the sequential prefetch in turn claims the remaining chain lines
  // as staged at their chain arrivals 32/48/64, so demands on them
  // still return 32/48/64.
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128; c.launch_prefetch = 512;
    c.stream_l1 = 16; c.stream_cold = 100; c.parallelism = 1;
    InstructionFetch f;
    f.configure(c);
    f.launch_prefetch(0, 0, 0, 4096, 0);
    for (std::uint64_t j = 0; j < 4; ++j)
      assert(f.ready(0, 0, 0, 4096 + j * 128, 0, 0) == (j + 1) * 16);
    // line 32's sequential prefetch claimed line 36 (outside the
    // launch window) at 64 + stream_cold = 164 on the cold channel, so
    // a far cold demand at now=20 lands at 164 + 100 = 264
    assert(f.ready(0, 0, 0, 100 * 128, 20, 0) == 264);
  }
  // (b) a demand on a launch-prefetch line counts it as L1-resident:
  // it pays one L1 slot (20 + 100 = 120), beating the chain arrival
  // 8*100 = 800, so the line becomes a demand fetch at 120
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128; c.launch_prefetch = 8 * 128;
    c.stream_l1 = 100; c.stream_cold = 30; c.parallelism = 1;
    InstructionFetch f;
    f.configure(c);
    f.launch_prefetch(0, 0, 0, 0, 0);
    assert(f.ready(0, 0, 0, 7 * 128, 20, 0) == 120);
  }
  // (c) sequential cold stream of 8 lines on one channel with
  // prefetch depth 4 = 8 x stream_cold (every line pays its slot):
  // demands on inflight prefetch lines claim their slot, line 4->5
  // misses again at last + 60
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128; c.stream_cold = 60;
    c.stream_l1 = 10; c.stream_l2 = 20; c.prefetch_depth = 4;
    InstructionFetch f;
    f.configure(c);
    std::uint64_t t = 0;
    for (std::uint64_t i = 0; i < 8; ++i)
      t = f.ready(0, 0, 0, i * 128, 0, 0);
    assert(t == 8 * 60);
  }
  // (d) L1 plateau overlaps execution: 384 L1-resident lines (L0 224
  // lines < body, L1 896 lines > body) at stream_l1 = 20 with
  // sequential prefetch enabled cost ~20 cycles/line on the second
  // pass, not 20 + 8 serial. Replica (fetch_replica_v3) gives 8192;
  // the old skip rule gave ~384*28 = 10752.
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128;
    c.capacity_l0 = 224 * 128; c.capacity_l1 = 896 * 128;
    c.stream_l1 = 20; c.stream_l2 = 40; c.stream_cold = 120;
    c.prefetch_depth = 5; c.parallelism = 1;
    InstructionFetch f;
    f.configure(c);
    const std::uint64_t pass1 = run_lines(f, 0, 384, 0);
    const std::uint64_t pass2 = run_lines(f, 0, 384, pass1) - pass1;
    assert(pass2 >= 384 * 20 && pass2 <= (std::uint64_t)(384 * 20 * 1.1));
  }
  // (e) a loop body exactly filling the L0 (224 lines) does not
  // thrash: prefetched lines 224..228 park in staged instead of
  // evicting body lines, so the second pass is all L0 hits
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128;
    c.capacity_l0 = 224 * 128; c.capacity_l1 = 896 * 128;
    c.stream_l1 = 20; c.stream_l2 = 40; c.stream_cold = 120;
    c.prefetch_depth = 5; c.parallelism = 1;
    InstructionFetch f;
    f.configure(c);
    std::uint64_t now = run_lines(f, 0, 224, 0);
    for (std::uint64_t i = 0; i < 224; ++i)
      for (int j = 0; j < 8; ++j) {
        assert(f.ready(0, 0, 0, (i * 8 + j) * 16, now, 0) == now);
        ++now;
      }
  }
  // (f) a warp starting 48 lines into the launch-prefetch window gets
  // its first line at one L1 slot (20), not the chain arrival 49*20 =
  // 980; 16 lines stream at the L1 cadence. Replica gives 344 total;
  // the old rule waited >= 980 before starting.
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128;
    c.capacity_l0 = 224 * 128; c.capacity_l1 = 896 * 128;
    c.launch_prefetch = 64 * 128;
    c.stream_l1 = 20; c.stream_l2 = 40; c.stream_cold = 120;
    c.prefetch_depth = 5; c.parallelism = 1;
    InstructionFetch f;
    f.configure(c);
    f.launch_prefetch(0, 0, 0, 0, 0);
    assert(f.ready(0, 0, 0, 48 * 128, 0, 0) <= 20);
    assert(run_lines(f, 48 * 128, 16, 0) <= 16 * 20 + 40);
  }
  // (g) prefetched-but-undemanded lines do not enter the L0: after the
  // prefetch slots for X+1..X+5 arrive they sit in `staged`; demanding
  // X+1 takes the staged path (free, counts neither l0_hits nor
  // l1_lines) and only then lands in the L0
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128; c.stream_cold = 60;
    c.stream_l1 = 10; c.stream_l2 = 20; c.prefetch_depth = 5;
    InstructionFetch f;
    f.configure(c);
    assert(f.ready(0, 0, 0, 0, 0, 0) == 60);  // X = line 0, cold
    // X+1..X+5 arrive at 120..360 and park in staged
    assert(f.ready(0, 0, 0, 128, 400, 0) == 400);
    assert(f.staged_hits == 1 && f.l0_hits == 0 && f.l1_lines == 0);
    // now L0-resident: a second demand is a plain L0 hit
    assert(f.ready(0, 0, 0, 128, 401, 0) == 401);
    assert(f.l0_hits == 1);
  }
  // (i) l0_ways = 4, capacity 32768 B -> 64 sets of 4 ways (ifetch v3):
  // a cyclic walk over 256 + 8 contiguous lines puts 5 lines into sets
  // 0..7 and 4 into sets 8..63. Sets 0..7 evict one member per insert,
  // so on the second pass every one of the 5 lines per thrashing set is
  // re-fetched: 5 x 8 = 40 non-hit first accesses (a miss, an inflight
  // claim, or a free staged re-entry -- all three bypass l0_hits), the
  // other 224 lines hit. 264 lines x 8 instructions: l0_hits delta =
  // 2112 - 40 = 2072, and (r > now calls) + (staged_hits delta) = 40.
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128;
    c.capacity_l0 = 32768; c.capacity_l1 = 896 * 128;
    c.stream_l1 = 20; c.stream_l2 = 40; c.stream_cold = 120;
    c.prefetch_depth = 5; c.parallelism = 1; c.l0_ways = 4;
    InstructionFetch f;
    f.configure(c);
    std::uint64_t now = run_lines(f, 0, 264, 0);
    const std::uint64_t h0 = f.l0_hits, s0 = f.staged_hits;
    std::uint64_t misses = 0;
    for (std::uint64_t i = 0; i < 264; ++i)
      for (int j = 0; j < 8; ++j) {
        const std::uint64_t r =
            f.ready(0, 0, 0, (i * 8 + j) * 16, now, 0);
        if (r > now) { now = r; ++misses; }
        ++now;
      }
    assert(misses + (f.staged_hits - s0) == 40);
    assert(f.l0_hits - h0 == 264 * 8 - 40);
  }
  // (j) same cache, walk exactly fills all 64 sets x 4 ways = 256
  // lines: nothing is evicted, the second pass is all L0 hits
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128;
    c.capacity_l0 = 32768; c.capacity_l1 = 896 * 128;
    c.stream_l1 = 20; c.stream_l2 = 40; c.stream_cold = 120;
    c.prefetch_depth = 5; c.parallelism = 1; c.l0_ways = 4;
    InstructionFetch f;
    f.configure(c);
    std::uint64_t now = run_lines(f, 0, 256, 0);
    for (std::uint64_t i = 0; i < 256; ++i)
      for (int j = 0; j < 8; ++j) {
        assert(f.ready(0, 0, 0, (i * 8 + j) * 16, now, 0) == now);
        ++now;
      }
  }
  // (k) l0_ways absent (0): the old single 256-line LRU thrashes on the
  // same 264-line cyclic walk -- every line misses on every pass
  // (2112 - 264 = 1848 l0_hits on the second pass)
  {
    FetchConfig c;
    c.enabled = true; c.line_bytes = 128;
    c.capacity_l0 = 32768; c.capacity_l1 = 896 * 128;
    c.stream_l1 = 20; c.stream_l2 = 40; c.stream_cold = 120;
    c.prefetch_depth = 5; c.parallelism = 1;
    InstructionFetch f;
    f.configure(c);
    std::uint64_t now = run_lines(f, 0, 264, 0);
    const std::uint64_t h0 = f.l0_hits, s0 = f.staged_hits;
    std::uint64_t misses = 0;
    for (std::uint64_t i = 0; i < 264; ++i)
      for (int j = 0; j < 8; ++j) {
        const std::uint64_t r =
            f.ready(0, 0, 0, (i * 8 + j) * 16, now, 0);
        if (r > now) { now = r; ++misses; }
        ++now;
      }
    assert(misses + (f.staged_hits - s0) == 264);
    assert(f.l0_hits - h0 == 264 * 8 - 264);
  }
  // (h) disabled config: configure leaves sms_ empty, engine never
  // calls ready()
  {
    FetchConfig c;
    c.enabled = false;
    InstructionFetch f;
    f.configure(c);
    (void)f;
  }
  std::puts("instruction_fetch_test ok");
  return 0;
}
