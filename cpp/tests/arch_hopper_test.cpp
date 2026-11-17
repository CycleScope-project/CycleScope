// arch_hopper_test — HopperModule fixture assertions (docs/en/design/engine.md).

#include <cassert>
#include <cstdio>
#include <map>
#include <string>
#include <utility>

#include "core/job_reader.hpp"
#include "core/mechanism_runtime.hpp"
#include "core/simulator.hpp"

using namespace cyclescope;

namespace {

struct Run {
  Job job;
  ResultStats stats;
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t>
      issue;
};

Run run(const std::string& path) {
  Run r;
  r.job = cyclescope::read_container_file(path);
  MechanismRuntime::Options ro;
  ro.reference_sm_clock_khz =
      r.job.launch.reference_sm_clock_khz;
  MechanismRuntime rt(r.job, r.job.launch.device, ro);
  Simulator::Options so;
  so.journal = true;
  Simulator sim(r.job, rt, so);
  r.stats = sim.run();
  for (const auto& j : r.stats.journal)
    r.issue[{j.warp, j.instance}] = j.issue_cycle;
  return r;
}

// warp-local instance offset -> issue cycle
std::uint64_t at(const Run& r, std::uint32_t w,
                 std::uint64_t inst_offset) {
  const std::uint64_t gi =
      r.job.warps[w].first_instance + inst_offset;
  auto it = r.issue.find({w, gi});
  if (it == r.issue.end()) {
    std::fprintf(stderr, "missing issue for warp %u inst %llu\n", w,
                 (unsigned long long)gi);
    std::exit(1);
  }
  return it->second;
}

int fail = 0;

#define CHECK(cond, ...)                                  \
  do {                                                    \
    if (!(cond)) {                                        \
      std::fprintf(stderr, "FAIL %s: ", #cond);           \
      std::fprintf(stderr, __VA_ARGS__);                  \
      std::fprintf(stderr, "\n");                         \
      ++fail;                                             \
    }                                                     \
  } while (0)

void expect_engine_error(const std::string& path,
                         const char* prefix) {
  try {
    (void)run(path);
  } catch (const cyclescope::EngineError& e) {
    CHECK(std::string(e.what()).rfind(prefix, 0) == 0,
          "expected EngineError %s, got %s", prefix, e.what());
    return;
  }
  CHECK(false, "expected EngineError %s, run finished", prefix);
}

// post-wait fill cycle of each warpgroup (h3 shape: 8 warps,
// wgmma + wait + fill)
std::pair<std::uint64_t, std::uint64_t> wg_fill(const Run& r) {
  std::uint64_t g0w = ~0ull, g1w = ~0ull;
  for (std::uint32_t w = 0; w < 8; ++w) {
    const std::uint64_t c = at(r, w, 2);
    if (w < 4) g0w = std::min(g0w, c); else g1w = std::min(g1w, c);
  }
  return {g0w, g1w};
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argv[1];

  {  // H1: four-warp collective launch + wait N=0
    auto r = run(dir + "/h1_collective.job");
    std::uint64_t launch = 0, lo = ~0ull;
    for (std::uint32_t w = 0; w < 4; ++w) {
      const std::uint64_t c = at(r, w, 0 + 3);  // wgmma
      launch = std::max(launch, c);
      lo = std::min(lo, c);
      CHECK(c >= 3, "wgmma w%u issued too early %llu", w,
            (unsigned long long)c);
    }
    CHECK(launch - lo <= 1, "wgmma issues spread %llu",
          (unsigned long long)(launch - lo));
    for (std::uint32_t w = 0; w < 4; ++w)
      CHECK(at(r, w, 5) == lo + 51,
            "post-wait fill w%u at %llu want %llu", w,
            (unsigned long long)at(r, w, 5),
            (unsigned long long)(lo + 51));
  }

  {  // H2: wait N=1 tolerates one pending group
    auto r = run(dir + "/h2_wait_n1.job");
    // both groups committed back-to-back; N=1 releases the older
    // once group0 completes at launch+50
    for (std::uint32_t w = 0; w < 4; ++w)
      CHECK(at(r, w, 3) >= at(r, w, 0) + 50 ||
                at(r, w, 3) >= at(r, w, 1) + 1,
            "h2 fill w%u at %llu", w,
            (unsigned long long)at(r, w, 3));
  }

  {  // H3: occupancy queue serializes the two warpgroups
    auto r = run(dir + "/h3_occupancy.job");
    std::uint64_t g0w = ~0ull, g1w = ~0ull;
    for (std::uint32_t w = 0; w < 8; ++w) {
      const std::uint64_t c = at(r, w, 2);  // post-wait fill
      if (w < 4) g0w = std::min(g0w, c); else g1w = std::min(g1w, c);
    }
    CHECK(g1w >= g0w + 10, "wg1 release %llu want >= %llu",
          (unsigned long long)g1w,
          (unsigned long long)(g0w + 10));
  }

  {  // H4: register-file steal does NOT delay the other warpgroup's
     // independent ALU stream (co-resident scope is issue-floor +
     // dependent penalty only)
    auto r = run(dir + "/h4_steal.job");
    // warp 4's 60 fills issue at ~1/cycle regardless of the op
    const std::uint64_t c0 = at(r, 4, 0), c59 = at(r, 4, 59);
    const double cad = double(c59 - c0) / 59.0;
    CHECK(cad <= 1.05, "co-resident stream cadence %f want ~1",
          cad);
  }

  {  // H11 (a): issuing warpgroup 1-read stream -> floor 2/inst
    auto r = run(dir + "/h11_steal_1read.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const double cad = double(at(r, w, 32) - at(r, w, 1)) / 31.0;
      CHECK(cad > 1.9 && cad < 2.1, "h11 w%u cadence %f want ~2", w,
            cad);
    }
  }

  {  // H12 (b): 3-read FFMA stream -> 3/inst
    auto r = run(dir + "/h12_steal_3read.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const double cad = double(at(r, w, 32) - at(r, w, 1)) / 31.0;
      CHECK(cad > 2.9 && cad < 3.1, "h12 w%u cadence %f want ~3", w,
            cad);
    }
  }

  {  // H13 (c): reuse flags do not serve under steal -> still 3
    auto r = run(dir + "/h13_steal_reuse.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const double cad = double(at(r, w, 32) - at(r, w, 1)) / 31.0;
      CHECK(cad > 2.9 && cad < 3.1, "h13 w%u cadence %f want ~3", w,
            cad);
    }
  }

  {  // H14 (d): issuing warpgroup's own dependent chain stays 4
    auto r = run(dir + "/h14_steal_chain.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const double cad = double(at(r, w, 32) - at(r, w, 1)) / 31.0;
      CHECK(cad > 3.9 && cad < 4.1, "h14 w%u cadence %f want ~4", w,
            cad);
    }
  }

  {  // H15 (e): after the op completes the 1-read stream is 1
    auto r = run(dir + "/h15_steal_done1.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const double cad = double(at(r, w, 32) - at(r, w, 24)) / 8.0;
      CHECK(cad <= 1.05, "h15 w%u tail cadence %f want ~1", w, cad);
    }
  }

  {  // H16 (e2): after the op completes the 3-read stream is 1.5
    auto r = run(dir + "/h16_steal_done3.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const double cad = double(at(r, w, 32) - at(r, w, 24)) / 8.0;
      CHECK(cad > 1.4 && cad < 1.6, "h16 w%u tail cadence %f want ~1.5",
            w, cad);
    }
  }

  {  // H17 (f): co-resident warpgroup's stream unaffected -> 1
    auto r = run(dir + "/h17_coresident_stream.job");
    for (std::uint32_t w = 4; w < 8; ++w) {
      const double cad = double(at(r, w, 31) - at(r, w, 0)) / 31.0;
      CHECK(cad <= 1.05, "h17 w%u cadence %f want ~1", w, cad);
    }
  }

  {  // H18 (g): co-resident dependent chain pays +0.5 -> ~4.5 avg
    auto r = run(dir + "/h18_coresident_chain.job");
    for (std::uint32_t w = 4; w < 8; ++w) {
      const std::uint64_t span = at(r, w, 31) - at(r, w, 0);
      const double cad = double(span) / 31.0;
      CHECK(cad > 4.4 && cad < 4.6, "h18 w%u cadence %f want ~4.5",
            w, cad);
    }
  }

  {  // H19 (v5): reuse-flagged stream pays the reuse floor -> ~4
    auto r = run(dir + "/h19_reuse_floor.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const double cad = double(at(r, w, 32) - at(r, w, 1)) / 31.0;
      CHECK(cad > 3.9 && cad < 4.1, "h19 w%u cadence %f want ~4", w,
            cad);
    }
  }

  {  // H20 (v5): unflagged stream pays the base floor -> ~2
    auto r = run(dir + "/h20_noflag_floor.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const double cad = double(at(r, w, 32) - at(r, w, 1)) / 31.0;
      CHECK(cad > 1.9 && cad < 2.1, "h20 w%u cadence %f want ~2", w,
            cad);
    }
  }

  {  // H21 (v5): rs op + unflagged stream -> alternating +1 -> ~2.5
    auto r = run(dir + "/h21_rs_floor.job");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const std::uint64_t span = at(r, w, 32) - at(r, w, 1);
      const double cad = double(span) / 31.0;
      CHECK(cad > 2.4 && cad < 2.6, "h21 w%u cadence %f want ~2.5",
            w, cad);
    }
  }

  {  // W1: pending_commit_groups_limit 6, beat 50, 10 back-to-back
    // ops, one op per commit group.  Ops 0-5 launch ~1/cycle.  Pipe
    // starts serialize: start_k = L0 + 50k, end_k = L0 + 50(k+1).
    // Each launch closes its own group, so at op 6's evaluation six
    // closed groups are pending -> held until the earliest group end,
    // i.e. launch_k = end_{k-6} = L0 + 50(k-5): launches land at
    // L0+50, L0+100, L0+150, L0+200.
    auto r = run(dir + "/w1_pending_ops.job");
    const std::uint64_t l0 = at(r, 0, 0);
    for (std::uint32_t k = 1; k <= 5; ++k)
      CHECK(at(r, 0, k) <= l0 + 10,
            "w1 op%u unheld at %llu want <= %llu", k,
            (unsigned long long)at(r, 0, k),
            (unsigned long long)(l0 + 10));
    CHECK(at(r, 0, 6) >= l0 + 50, "w1 op6 held at %llu want >= %llu",
          (unsigned long long)at(r, 0, 6),
          (unsigned long long)(l0 + 50));
    for (std::uint32_t k = 7; k <= 9; ++k) {
      const std::uint64_t gap = at(r, 0, k) - at(r, 0, k - 1);
      CHECK(gap >= 45 && gap <= 55,
            "w1 op%u gap %llu want ~50", k, (unsigned long long)gap);
    }
  }

  {  // W1 control: fields 11-13 absent -> launches stay back-to-back
    auto r = run(dir + "/w1b_pending_off.job");
    CHECK(at(r, 0, 9) - at(r, 0, 0) <= 20,
          "w1b op9 at +%llu want <= 20",
          (unsigned long long)(at(r, 0, 9) - at(r, 0, 0)));
  }

  {  // W1c: 12 ops in commit groups of 3 (commit on ops 2/5/8/11):
    // four closed groups total < limit 6 -> no launch is held,
    // launches stay ~1/cycle like the unheld control
    auto r = run(dir + "/w1c_groups3.job");
    CHECK(at(r, 0, 11) - at(r, 0, 0) <= 20,
          "w1c op11 at +%llu want <= 20",
          (unsigned long long)(at(r, 0, 11) - at(r, 0, 0)));
  }

  {  // W2: outstanding-op threshold.  Owner warps 0-3 launch 12 ops
    // of ONE still-open group (~1/cycle, no launch held); ends
    // serialize at end_k = L0 + 50(k+1).  Warp 4 (second warpgroup,
    // SMSP 0) reaches its first MUFU ~cycle 40 with all 12 ends
    // outstanding: 12 > 6 -> waits until the outstanding count
    // reaches 6, i.e. the end of the (12 - 6) = 6th earliest op
    // (index 5, L0 + 300); it must issue before op 6's end (L0+350)
    // plus a small scheduling bound
    auto r = run(dir + "/w2_mufu_hold.job");
    const std::uint64_t l0 = at(r, 0, 0);
    CHECK(at(r, 4, 1) >= l0 + 295,
          "w2 first MUFU at +%llu want >= +295 (end of op 5)",
          (unsigned long long)(at(r, 4, 1) - l0));
    CHECK(at(r, 4, 1) < l0 + 360,
          "w2 first MUFU at +%llu want < +360 (end of op 6 + 10)",
          (unsigned long long)(at(r, 4, 1) - l0));
  }

  {  // W2 twin: FFMA is not a memory-queue class -> issues right
    // after its ~38-cycle producer, long before the threshold retry
    auto r = run(dir + "/w2_ffma_twin.job");
    const std::uint64_t l0 = at(r, 0, 0);
    CHECK(at(r, 4, 1) < l0 + 100,
          "w2f first FFMA at +%llu want < +100",
          (unsigned long long)(at(r, 4, 1) - l0));
  }

  {  // W2 hold off: launch_hold_blocks_memory_queue = 0 keeps the
    // commit-group admission but not the memory-queue hold
    auto r = run(dir + "/w2_mufu_nohold.job");
    const std::uint64_t l0 = at(r, 0, 0);
    CHECK(at(r, 4, 1) < l0 + 100,
          "w2o first MUFU at +%llu want < +100",
          (unsigned long long)(at(r, 4, 1) - l0));
  }

  {  // W3: the owner's own MUFU.  Warp 0 issues the same 12 one-group
    // ops, then 8 MUFU; its first MUFU evaluation (~cycle 15) is
    // blocked by the same threshold until the outstanding count
    // reaches 6 (end of op index 5, L0 + 300)
    auto r = run(dir + "/w3_owner_mufu.job");
    const std::uint64_t l0 = at(r, 0, 0);
    CHECK(at(r, 0, 12) >= l0 + 295,
          "w3 owner first MUFU at +%llu want >= +295 (end of op 5)",
          (unsigned long long)(at(r, 0, 12) - l0));
    CHECK(at(r, 0, 12) < l0 + 360,
          "w3 owner first MUFU at +%llu want < +360",
          (unsigned long long)(at(r, 0, 12) - l0));
  }

  {  // H5: consumer FFMA waits for accumulator completion + leaf
    auto r = run(dir + "/h5_handoff.job");
    std::uint64_t lo = ~0ull;
    for (std::uint32_t w = 0; w < 4; ++w)
      lo = std::min(lo, at(r, w, 0));
    for (std::uint32_t w = 0; w < 4; ++w) {
      const std::uint64_t want = lo + 20 + 7 + 9;
      CHECK(at(r, w, 2) >= want, "ffma w%u at %llu want >= %llu",
            w, (unsigned long long)at(r, w, 2),
            (unsigned long long)want);
    }
  }

  {  // H10: fp8_fmnmx handoff guard: only QGMMA->FMNMX pays 112
    auto r = run(dir + "/h10_handoff_guard.job");
    const std::uint64_t base = at(r, 0, 0);   // wgmma issue
    const std::uint64_t want0 = base + 20 + 7 + 112;      // qgmma->fmnmx
    const std::uint64_t want12 = base + 20 + 7;           // acc completion
    CHECK(at(r, 0, 2) >= want0, "qgmma->fmnmx at %llu want >= %llu",
          (unsigned long long)at(r, 0, 2), (unsigned long long)want0);
    CHECK(at(r, 1, 2) >= want12 && at(r, 1, 2) < want0,
          "qgmma->fadd at %llu want [%llu,%llu)",
          (unsigned long long)at(r, 1, 2),
          (unsigned long long)want12, (unsigned long long)want0);
    CHECK(at(r, 2, 2) >= want12 && at(r, 2, 2) < want0,
          "hgmma->fmnmx at %llu want [%llu,%llu)",
          (unsigned long long)at(r, 2, 2),
          (unsigned long long)want12, (unsigned long long)want0);
  }

  {  // H6: ldgsts completion = issue + kind-39 (40)
    auto r = run(dir + "/h6_ldgsts.job");
    CHECK(at(r, 0, 3) >= 41, "post-wait fill at %llu want >= 41",
          (unsigned long long)at(r, 0, 3));
  }

  {  // H7: proxy_fence drain: 4 STS x 1024 B / 128 Bpc + latency 10
    auto r = run(dir + "/h7_proxy_fence.job");
    const std::uint64_t fence = at(r, 0, 4);
    CHECK(at(r, 0, 5) >= fence + 10 + 32,
          "post-fence fill at %llu want >= %llu",
          (unsigned long long)at(r, 0, 5),
          (unsigned long long)(fence + 42));
  }

  {  // H8: UTMALDG is not a drain object: fence blocks latency only
    auto r = run(dir + "/h8_fence_tma.job");
    const std::uint64_t fence = at(r, 0, 2);
    CHECK(at(r, 0, 3) >= fence + 10 &&
              at(r, 0, 3) <= fence + 14,
          "tma fence fill at %llu want ~%llu",
          (unsigned long long)at(r, 0, 3),
          (unsigned long long)(fence + 10));
  }

  {  // H9: queue capacity 1500 B backpressures the 3rd STS
    auto r = run(dir + "/h9_backpressure.job");
    CHECK(at(r, 0, 2) >= at(r, 0, 0) + 4,
          "3rd STS at %llu want >= %llu",
          (unsigned long long)at(r, 0, 2),
          (unsigned long long)(at(r, 0, 0) + 4));
  }

  std::fprintf(stderr, "arch_hopper: %d failures\n", fail);
  return fail ? 1 : 0;
}
