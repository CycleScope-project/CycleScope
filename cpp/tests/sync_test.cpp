// sync_test — engine 3-3 step a: synthetic sync fixtures driven
// through Simulator with a journal.  argv[1] = fixtures directory.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>

#include "core/job_reader.hpp"
#include "core/mechanism_runtime.hpp"
#include "core/simulator.hpp"
#include "arch/architecture_module.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what, std::uint64_t expect,
           std::uint64_t got) {
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL %s: expected %llu got %llu\n", what,
                 (unsigned long long)expect,
                 (unsigned long long)got);
  }
}

struct Run {
  cyclescope::Job job;
  cyclescope::ResultStats stats;
  // (warp, instance) -> issue cycle
  std::map<std::pair<std::uint32_t, std::uint64_t>, std::uint64_t> issue;
};

Run run_fixture(const std::string& path) {
  Run r;
  r.job = cyclescope::read_container_file(path);
  cyclescope::MechanismRuntime::Options ro;
  ro.reference_sm_clock_khz = r.job.launch.reference_sm_clock_khz;
  cyclescope::MechanismRuntime rt(r.job, r.job.launch.device, ro);
  cyclescope::Simulator::Options so;
  so.journal = true;
  cyclescope::Simulator sim(r.job, rt, so);
  r.stats = sim.run();
  for (const auto& j : r.stats.journal)
    r.issue[{j.warp, j.instance}] = j.issue_cycle;
  return r;
}

// issue cycle of warp-local instance k
std::uint64_t at(const Run& r, std::uint32_t w, std::uint64_t k) {
  const std::uint64_t gi = r.job.warps[w].first_instance + k;
  auto it = r.issue.find({w, gi});
  if (it == r.issue.end()) {
    std::fprintf(stderr, "missing journal entry warp %u inst %llu\n", w,
                 (unsigned long long)gi);
    ++failures;
    return ~0ull;
  }
  return it->second;
}

std::uint64_t tele(const Run& r, std::uint32_t counter) {
  for (const auto& t : r.stats.tele)
    if (t.group == 1 && t.counter == counter) return t.value;
  return 0;
}

std::uint64_t tele2(const Run& r, std::uint32_t counter) {
  for (const auto& t : r.stats.tele)
    if (t.group == 2 && t.counter == counter) return t.value;
  return 0;
}

// group 4 = memory hierarchy counters (fixed order, docs/en/engine.md):
// 12 same_line, 13 const_hits, 14 const_misses
std::uint64_t tele4(const Run& r, std::uint32_t counter) {
  for (const auto& t : r.stats.tele)
    if (t.group == 4 && t.counter == counter) return t.value;
  return 0;
}

void expect_engine_error(const std::string& path, const char* prefix,
                         const char* what) {
  try {
    run_fixture(path);
  } catch (const cyclescope::EngineError& e) {
    const std::string msg = e.what();
    check(msg.rfind(prefix, 0) == 0, what, 0, msg.empty() ? 1 : 0);
    if (msg.rfind(prefix, 0) != 0)
      std::fprintf(stderr, "  got error text: %s\n", msg.c_str());
    return;
  }
  ++failures;
  std::fprintf(stderr, "FAIL %s: expected EngineError %s, run finished\n",
               what, prefix);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  auto J = [&](const char* name) {
    return run_fixture(dir + "/" + name + ".job");
  };

  // ---- L1: per-cycle advance --------------------------------------
  {
    auto r = J("l1_one_warp");
    for (std::uint64_t i = 0; i < 8; ++i)
      check(at(r, 0, i) == i, "l1_one_warp issue", i, at(r, 0, i));
  }
  {
    auto r = J("l1_two_warps");
    bool covered = true, alternate = true;
    for (std::uint64_t i = 0; i < 4; ++i) {
      if (at(r, 0, i) != 2 * i) alternate = false;
      if (at(r, 1, i) != 2 * i + 1) alternate = false;
    }
    for (const auto& [k, v] : r.issue)
      if (v > 7) covered = false;
    check(alternate, "l1_two_warps alternate", 1, alternate);
    check(covered && r.issue.size() == 8, "l1_two_warps window", 8,
          r.issue.size());
  }

  // ---- A1: phase completes at last arrive; wait releases after leaf -
  {
    auto r = J("a1_basic");
    const std::uint64_t expect =
        std::max(at(r, 0, 5), at(r, 1, 0)) + 10;
    check(at(r, 0, 6) == expect, "a1 wait release", expect, at(r, 0, 6));
    check(at(r, 0, 7) == expect + 1, "a1 post-wait", expect + 1,
          at(r, 0, 7));
  }

  // ---- A3: parity 1 ready on a fresh object -------------------------
  {
    auto r = J("a3_parity1");
    check(at(r, 0, 0) == 0, "a3 wait", 0, at(r, 0, 0));
    check(at(r, 0, 1) == 1, "a3 post", 1, at(r, 0, 1));
  }

  // ---- A4/A5: failed poll suspends, success wait blocks -------------
  for (const char* name : {"a4_poll", "a4_poll_notimeout",
                           "a5_lookahead"}) {
    auto r = J(name);
    const std::uint64_t arrive = at(r, 1, 200);
    check(at(r, 0, 0) == 0, name, 0, at(r, 0, 0));
    check(at(r, 0, 1) == arrive + 10, name, arrive + 10, at(r, 0, 1));
    check(tele(r, cyclescope::kTeleFailedPolls) == 1, name, 1,
          tele(r, cyclescope::kTeleFailedPolls));
    // the failed poll is retried by a later same-(object,parity)
    // check -> not abandoned -> no trywait_timeout hold
    check(tele(r, cyclescope::kTelePollSuspendCycles) == 0, name, 0,
          tele(r, cyclescope::kTelePollSuspendCycles));
    check(tele(r, cyclescope::kTeleAbandonedPolls) == 0, name, 0,
          tele(r, cyclescope::kTeleAbandonedPolls));
  }

  // ---- A6: deferred wait block --------------------------------------
  {
    auto r = J("a6_deferred");
    check(at(r, 0, 1) == 1, "a6 inst1", 1, at(r, 0, 1));
    check(at(r, 0, 2) == 2, "a6 inst2", 2, at(r, 0, 2));
    check(at(r, 0, 3) == at(r, 1, 50) + 10, "a6 block", at(r, 1, 50) + 10,
          at(r, 0, 3));
  }

  // ---- A7: rejections ------------------------------------------------
  expect_engine_error(dir + "/a7_init_zero.job",
                      "missing_fact:mbarrier_init_count", "a7 init zero");
  expect_engine_error(dir + "/a7_arrive_zero.job",
                      "missing_fact:mbarrier_arrive_count",
                      "a7 arrive zero");

  // ---- A8: CTA named barrier -----------------------------------------
  {
    auto r = J("a8_one_gen");
    for (std::uint32_t w = 0; w < 4; ++w)
      check(at(r, w, 10 * w + 1) == 37, "a8 release", 37,
            at(r, w, 10 * w + 1));
  }
  {
    auto r = J("a8_two_gens");
    for (std::uint32_t w = 0; w < 4; ++w)
      check(at(r, w, 10 * w + 7) == 49, "a8 gen2", 49,
            at(r, w, 10 * w + 7));
  }
  {
    auto r = J("a8_exit");
    for (std::uint32_t w = 0; w < 3; ++w)
      check(at(r, w, 10 * w + 1) == 27, "a8 exit release", 27,
            at(r, w, 10 * w + 1));
  }

  // ---- A9: cluster barrier -------------------------------------------
  {
    auto r = J("a9_cluster");
    for (std::uint32_t w = 0; w < 4; ++w) {
      const std::uint64_t fill = 10 * w;  // warp order = (2c+k)
      check(at(r, w, fill + 7) == 50, "a9 wait release", 50,
            at(r, w, fill + 7));
    }
  }

  // ---- A10: nanosleep -------------------------------------------------
  {
    auto r = J("a10_nanosleep");
    check(at(r, 0, 0) == 0, "a10 issue", 0, at(r, 0, 0));
    check(at(r, 0, 1) == 100, "a10 wake", 100, at(r, 0, 1));
  }
  {
    auto r = J("a10_syncs");
    check(at(r, 0, 0) == 0, "a10s wait", 0, at(r, 0, 0));
    check(at(r, 0, 1) == 1, "a10s nanosleep", 1, at(r, 0, 1));
    check(at(r, 0, 2) == 40, "a10s syncs wake", 40, at(r, 0, 2));
  }
  {
    // SYNCS sleep after an already-released awaited phase: no sleep
    auto r = J("a10_past_release");
    check(at(r, 0, 20) == 20, "a10p wait", 20, at(r, 0, 20));
    check(at(r, 0, 21) == 21, "a10p nanosleep", 21, at(r, 0, 21));
    check(at(r, 0, 22) == 22, "a10p no sleep", 22, at(r, 0, 22));
  }
  {
    // plain NANOSLEEP ignores the object: full duration
    auto r = J("a10_plain_after_wait");
    const std::uint64_t expect = at(r, 0, 21) + 100;
    check(at(r, 0, 21) == 21, "a10q nanosleep", 21, at(r, 0, 21));
    check(at(r, 0, 22) == expect, "a10q full sleep", expect,
          at(r, 0, 22));
  }
  {
    // PTX 1 ms cap: 10 ms hint -> FILL issues at 1e6 (1 GHz clock)
    auto r = J("a10_cap");
    check(at(r, 0, 1) == 1000000, "a10cap wake", 1000000,
          at(r, 0, 1));
  }

  // ---- A12: unreachable wait reports immediately ----------------------
  {
    const auto t0 = std::chrono::steady_clock::now();
    try {
      run_fixture(dir + "/a12_deadlock.job");
    { Run r = run_fixture(dir + "/a24_branch_envelope.job");
      // fills 0,1 then BRA@2 with envelope 20 -> fill@3 at 22+svc
      check(at(r, 0, 3) >= at(r, 0, 2) + 20,
            "a24 branch envelope", at(r, 0, 2) + 20, at(r, 0, 3)); }
      ++failures;
      std::fprintf(stderr, "FAIL a12: run finished without deadlock\n");
    } catch (const cyclescope::EngineError& e) {
      const std::string msg = e.what();
      check(msg.rfind("unsupported:engine:deadlock", 0) == 0,
            "a12 deadlock", 0, msg.empty() ? 1 : 0);
      if (msg.rfind("unsupported:engine:deadlock", 0) != 0)
        std::fprintf(stderr, "  got error text: %s\n", msg.c_str());
    }
    const double secs =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    check(secs < 1.0, "a12 wallclock s", 0, secs >= 1.0 ? 1 : 0);
    std::fprintf(stderr, "a12 deadlock after %.4f s\n", secs);
  }

  // ---- A11: store visibility settlement -------------------------------
  {
    auto r = J("a11_visibility");
    check(at(r, 0, 2) == 100, "a11 w0 post-bar", 100, at(r, 0, 2));
    check(at(r, 1, 51) == 100, "a11 w1 post-bar", 100, at(r, 1, 51));
    check(tele(r, cyclescope::kTeleVisibilitySettled) == 1, "a11 settled", 1,
          tele(r, cyclescope::kTeleVisibilitySettled));
  }

  {
    // initial_phase=1: the barrier's first phase is already complete,
    // so a parity-0 wait is ready the cycle after init
    auto r = J("a13_initial_phase");
    check(at(r, 0, 1) == 1, "a13 wait", 1, at(r, 0, 1));
    check(at(r, 0, 2) == 2, "a13 post", 2, at(r, 0, 2));
  }
  expect_engine_error(dir + "/a13_initial_phase0.job",
                      "unsupported:engine:deadlock", "a13 phase0");

  // ---- A21: one init row initialises every MBOL object --------------
  {
    auto r = J("a21_init_mbol");
    const std::uint64_t expect = at(r, 2, 20) + 10;
    check(at(r, 0, 1) == expect, "a21 wait obj2", expect, at(r, 0, 1));
    check(at(r, 0, 2) == expect + 1, "a21 post", expect + 1, at(r, 0, 2));
  }

  // ---- A22: re-init of an idle barrier wins -------------------------
  {
    auto r = J("a22_reinit_idle");
    // wait on object 0 (expected 2 after re-init) releases at the
    // second arrive + release leaf
    const std::uint64_t expect = at(r, 2, 20) + 10;
    check(at(r, 0, 2) == expect, "a22 wait", expect, at(r, 0, 2));
    check(at(r, 0, 3) == expect + 1, "a22 post", expect + 1,
          at(r, 0, 3));
  }
  expect_engine_error(dir + "/a22b_reinit_busy.job",
                      "unsupported:engine:mbarrier_reinit", "a22b busy");

  // ---- A23: complete_tx closes a tx-blocked phase -------------------
  {
    auto r = J("a23_complete_tx");
    // waiter releases at the complete_tx instance + release leaf
    const std::uint64_t expect = at(r, 1, 6) + 10;
    check(at(r, 0, 1) == expect, "a23 wait", expect, at(r, 0, 1));
    check(at(r, 0, 2) == expect + 1, "a23 post", expect + 1,
          at(r, 0, 2));
  }
  expect_engine_error(dir + "/a23b_complete_tx_zero.job",
                      "missing_fact:mbarrier_arrive_count",
                      "a23b zero tx");

  // ---- A25: control-word stall floor --------------------------------
  {
    auto r = J("a25_control_stall");
    // stall=6 after inst0 pushes inst1 by 6; stall=1 adds nothing;
    // inst3 (stall=2, result latency 4 -> R5) lets inst4 wait on the
    // register floor instead (max, not additive)
    check(at(r, 0, 1) == at(r, 0, 0) + 6, "a25 stall6",
          at(r, 0, 0) + 6, at(r, 0, 1));
    check(at(r, 0, 2) == at(r, 0, 1) + 1, "a25 stall1",
          at(r, 0, 1) + 1, at(r, 0, 2));
    check(at(r, 0, 4) == at(r, 0, 3) + 4, "a25 reg-max",
          at(r, 0, 3) + 4, at(r, 0, 4));
  }

  // ---- A26: failed polls carry no result latency ---------------------
  {
    auto r = J("a26_failed_poll_result");
    // failed poll (inst1): dependent BRA at +1, not +38; the later
    // successful wait (inst3) still holds its predicate for 38
    check(at(r, 0, 2) == at(r, 0, 1) + 1, "a26 failed-poll bra",
          at(r, 0, 1) + 1, at(r, 0, 2));
    check(at(r, 0, 4) == at(r, 0, 3) + 38, "a26 success-wait bra",
          at(r, 0, 3) + 38, at(r, 0, 4));
  }

  // ---- A27: read barrier frees at MIO acceptance ---------------------
  {
    auto r = J("a27_read_barrier_accept");
    // 8 LDS x kind-6 acceptance (10 cycles): the waiting FILL issues
    // no earlier than the last LDS's acceptance (first issue + 70)
    const std::uint64_t lo = at(r, 0, 0) + 70;
    check(at(r, 0, 8) >= lo && at(r, 0, 8) <= lo + 2,
          "a27 barrier-at-accept", lo, at(r, 0, 8));
  }
  {
    auto r = J("a27b_read_barrier_issue");
    // no acceptance row: the same barrier frees at issue
    check(at(r, 0, 8) == at(r, 0, 7) + 1, "a27b barrier-at-issue",
          at(r, 0, 7) + 1, at(r, 0, 8));
  }

  // ---- step b ----
  {
    auto r = J("b1_bulk_group");
    check(at(r, 0, 3) == 51, "b1 wait", 51, at(r, 0, 3));
    check(at(r, 0, 4) == 52, "b1 post", 52, at(r, 0, 4));
  }
  {
    auto r = J("b1_bulk_group_n1");
    check(at(r, 0, 4) == 50, "b1 n1 wait", 50, at(r, 0, 4));
    check(at(r, 0, 5) == 51, "b1 n1 post", 51, at(r, 0, 5));
  }
  {
    auto r = J("b2_cpasync_arrives");
    check(at(r, 0, 3) == 22, "b2 wait", 22, at(r, 0, 3));
    check(at(r, 0, 4) == 23, "b2 post", 23, at(r, 0, 4));
  }
  {
    // TRANSCNT pairing: tx +32 at 0, copy issued at 1, arrives at 2
    // (commit) completes the 32 tx at copy-done + 20, the real arrive
    // at 3 drops pending to 0 -> phase completes when both are done
    auto r = J("b2b_cpasync_transcnt");
    check(at(r, 0, 4) > at(r, 0, 3), "b2b wait blocked", at(r, 0, 3) + 1,
          at(r, 0, 4));
    check(at(r, 0, 5) == at(r, 0, 4) + 1, "b2b post", at(r, 0, 4) + 1,
          at(r, 0, 5));
  }
  {
    auto r = J("b3_multicast_tma");
    check(at(r, 0, 2) == 51, "b3 w0 wait", 51, at(r, 0, 2));
    check(at(r, 1, 1) == 51, "b3 w1 wait", 51, at(r, 1, 1));
  }
  expect_engine_error(dir + "/b3b_no_mbob.job",
                      "missing_fact:mbarrier_object_table", "b3b no mbob");
  {
    auto r = J("b3c_late_arrive");
    check(at(r, 1, 61) == 61, "b3c w1 wait", 61, at(r, 1, 61));
  }
  {
    auto r = J("b4_service_start");
    const std::uint64_t got[3] = {at(r, 0, 2), at(r, 1, 2), at(r, 2, 2)};
    const bool ok = (got[0] == 51 && got[1] == 61 && got[2] == 71) ||
        (got[0] == 51 && got[1] == 71 && got[2] == 61) ||
        (got[0] == 61 && got[1] == 51 && got[2] == 71) ||
        (got[0] == 61 && got[1] == 71 && got[2] == 51) ||
        (got[0] == 71 && got[1] == 51 && got[2] == 61) ||
        (got[0] == 71 && got[1] == 61 && got[2] == 51);
    check(ok, "b4 wait set", 1, ok);
  }
  {
    auto r = J("b5_descriptor");
    check(at(r, 1, 10) == 724, "b5 acquire", 724, at(r, 1, 10));
    check(at(r, 1, 11) == 725, "b5 post", 725, at(r, 1, 11));
  }
  {
    auto r = J("b6a_from");
    check(at(r, 1, 0) == at(r, 0, 5) + 1, "b6a from", at(r, 0, 5) + 1,
          at(r, 1, 0));
  }
  {
    auto r = J("b6b_initial");
    check(at(r, 1, 0) == 0, "b6b initial", 0, at(r, 1, 0));
  }
  {
    auto r = J("b6c_any");
    const std::uint64_t e = std::min(at(r, 0, 5), at(r, 2, 8));
    check(at(r, 1, 0) == e + 1, "b6c any", e + 1, at(r, 1, 0));
  }
  {
    auto r = J("b6d_chain");
    check(at(r, 1, 0) == at(r, 2, 8) + 1, "b6d read", at(r, 2, 8) + 1,
          at(r, 1, 0));
    // chain links carry no issue gate: w2's atom issues at its own
    // program position regardless of w0's atom
    check(at(r, 2, 8) == at(r, 2, 7) + 1, "b6d no pred gate",
          at(r, 2, 7) + 1, at(r, 2, 8));
    check(tele(r, cyclescope::kTeleGsyncChainsUnique) >= 1, "b6d chain",
          1, tele(r, cyclescope::kTeleGsyncChainsUnique));
  }
  {
    auto r = J("b6e_count");
    const std::uint64_t e = std::max(at(r, 0, 5), at(r, 2, 8));
    check(at(r, 1, 0) == e + 1, "b6e count", e + 1, at(r, 1, 0));
  }
  {
    auto r = J("b6f_poll");
    // rule 3b: [poll0, poll1) is a clean poll region -> both retire
    // in the winner's cycle instead of issuing at 0 / 1
    check(at(r, 1, 0) == at(r, 1, 2), "b6f poll0",
          at(r, 1, 2), at(r, 1, 0));
    check(at(r, 1, 1) == at(r, 1, 2), "b6f poll1",
          at(r, 1, 2), at(r, 1, 1));
    check(at(r, 1, 2) == at(r, 0, 5) + 1, "b6f winner",
          at(r, 0, 5) + 1, at(r, 1, 2));
    check(tele(r, cyclescope::kTeleGsyncDroppedPolls) == 2, "b6f drop",
          2, tele(r, cyclescope::kTeleGsyncDroppedPolls));
  }
  {
    // failed polls of a poll sequence: no load latency (USE after the
    // 400-cycle poll issues within a few cycles), winner pairs `from`
    auto r = J("b6l_poll_free");
    check(at(r, 1, 1) < at(r, 1, 0) + 10, "b6l poll0 use",
          at(r, 1, 0) + 10, at(r, 1, 1));
    check(at(r, 1, 3) < at(r, 1, 2) + 10, "b6l poll1 use",
          at(r, 1, 2) + 10, at(r, 1, 3));
    check(at(r, 1, 4) == at(r, 0, 5) + 1, "b6l winner",
          at(r, 0, 5) + 1, at(r, 1, 4));
    check(at(r, 1, 5) >= at(r, 1, 4) + 400, "b6l winner use",
          at(r, 1, 4) + 400, at(r, 1, 5));
  }
  {
    // b6m_poll_region: [poll,use50x3,poll,use50x3] folds and retires
    // in the winner's issue cycle; the winner still pairs the release
    auto r = J("b6m_poll_region");
    check(at(r, 1, 8) == at(r, 0, 5) + 1, "b6m winner",
          at(r, 0, 5) + 1, at(r, 1, 8));
    for (int k = 0; k < 8; ++k)
      check(at(r, 1, k) == at(r, 1, 8), "b6m region cycle",
            at(r, 1, 8), at(r, 1, k));
    check(tele(r, cyclescope::kTeleGsyncRegionsRetired) == 1,
          "b6m regions", 1,
          tele(r, cyclescope::kTeleGsyncRegionsRetired));
    check(tele(r, cyclescope::kTeleGsyncRegionInstances) == 8,
          "b6m region insts", 8,
          tele(r, cyclescope::kTeleGsyncRegionInstances));
  }
  {
    // b6n_poll_region_mixed: an mbarrier arrive inside the region
    // blocks folding; the winner still pairs the release and the
    // 50-cycle use chain issues per instance
    auto r = J("b6n_poll_region_mixed");
    check(at(r, 1, 8) >= at(r, 0, 5) + 1, "b6n winner",
          at(r, 0, 5) + 1, at(r, 1, 8));
    check(tele(r, cyclescope::kTeleGsyncRegionsRetired) == 0,
          "b6n no fold", 0,
          tele(r, cyclescope::kTeleGsyncRegionsRetired));
    check(at(r, 1, 3) >= at(r, 1, 2) + 50, "b6n use gap",
          at(r, 1, 2) + 50, at(r, 1, 3));
  }
  {
    // b6o_poll_region_macc: a memory access inside the region blocks
    // folding; winner still pairs the release
    auto r = J("b6o_poll_region_macc");
    check(tele(r, cyclescope::kTeleGsyncRegionsRetired) == 0,
          "b6o no fold", 0,
          tele(r, cyclescope::kTeleGsyncRegionsRetired));
    check(at(r, 1, 8) >= at(r, 0, 5) + 1, "b6o winner",
          at(r, 0, 5) + 1, at(r, 1, 8));
  }
  {
    // b6p_poll_region_constant: a constant-space access (LDC) inside
    // the region is not work between episodes; the region still folds
    auto r = J("b6p_poll_region_constant");
    check(tele(r, cyclescope::kTeleGsyncRegionsRetired) == 1,
          "b6p fold", 1,
          tele(r, cyclescope::kTeleGsyncRegionsRetired));
  }
  expect_engine_error(dir + "/b6g_unpaired.job",
                      "unsupported:engine:inter_cta_global_sync",
                      "b6g unpaired");
  {
    auto r = J("b6h_noop");
    check(at(r, 1, 0) == at(r, 2, 8) + 1, "b6h noop", at(r, 2, 8) + 1,
          at(r, 1, 0));
  }
  {
    auto r = J("b6i_no_writer");
    check(at(r, 1, 0) == 0, "b6i no-writer read", 0, at(r, 1, 0));
    check(tele(r, cyclescope::kTeleGsyncAcquiresNoWriter) == 1,
          "b6i no-writer tele", 1,
          tele(r, cyclescope::kTeleGsyncAcquiresNoWriter));
    check(tele(r, cyclescope::kTeleGsyncAcquiresInitial) == 0,
          "b6i initial tele", 0,
          tele(r, cyclescope::kTeleGsyncAcquiresInitial));
  }
  expect_engine_error(dir + "/b6j_no_writer_partial.job",
                      "unsupported:engine:inter_cta_global_sync",
                      "b6j partial coverage");
  expect_engine_error(dir + "/b6k_unknown_writer.job",
                      "unsupported:engine:inter_cta_global_sync",
                      "b6k unknown writer");
  {
    auto r = J("b7_envelope");
    check(at(r, 0, 4) == at(r, 0, 3) + 30, "b7 floor",
          at(r, 0, 3) + 30, at(r, 0, 4));
  }
  {
    auto r = J("b8_two_segments");
    check(at(r, 0, 1) == 1, "b8 seg0 wait", 1, at(r, 0, 1));
    check(at(r, 1, 1) == 1, "b8 seg1 wait", 1, at(r, 1, 1));
  }
  {
    auto r = J("b10_fence_floor");
    check(at(r, 0, 1) == at(r, 0, 0) + 50, "b10 fence floor",
          at(r, 0, 0) + 50, at(r, 0, 1));
  }
  {
    auto r = J("b10b_dest_result");
    check(at(r, 0, 1) == at(r, 0, 0) + 1, "b10b dest result",
          at(r, 0, 0) + 1, at(r, 0, 1));
  }
  {
    auto r = J("b10c_store_result");
    check(at(r, 0, 1) == at(r, 0, 0) + 1, "b10c store result",
          at(r, 0, 0) + 1, at(r, 0, 1));
  }
  {
    auto r = J("b9_drain");
    check(r.stats.cycles >= 50, "b9 cycles", 50,
          r.stats.cycles >= 50 ? 50 : r.stats.cycles);
    check(r.stats.issued == 2 && r.stats.completed == 2,
          "b9 issued", 2, r.stats.issued);
  }
  {  // (a) dirty proxy fence: 4 stores x512 B accepted 10/20/30/40,
     // fence at 4 -> next inst at ceil(40 + 9 + 75.26 + 2048/569) = 128
    auto r = J("b16_fence_drain");
    check(at(r, 0, 5) == 128, "b16 drain release", 128, at(r, 0, 5));
  }
  {  // (b) clean fence: row present vs absent release identically
    auto r = J("b17_fence_drain_clean");
    check(at(r, 0, 1) == at(r, 1, 1), "b17 clean equal",
          at(r, 0, 1), at(r, 1, 1));
    check(at(r, 0, 1) == 9, "b17 clean svc", 9, at(r, 0, 1));
  }
  {  // (c) second warp's fence after the barrier, nothing new written:
     // only the svc result floor applies
    auto r = J("b18_fence_drain_second");
    check(at(r, 1, 2) == at(r, 1, 1) + 9, "b18 second fence svc",
          at(r, 1, 1) + 9, at(r, 1, 2));
  }
  {  // (d) second write round re-drains new bytes only: fence1 128;
     // stores 5/6 issue 128/129, accepted 138/148 on the SM frontier ->
     // fence2 ceil(148 + 9 + 75.26 + 1024/569) = 235
    auto r = J("b19_fence_drain_twice");
    check(at(r, 0, 8) == 235, "b19 second drain", 235, at(r, 0, 8));
  }
  {  // (e) per-SMSP acceptance (kind 80): 4 stores x 32 cycles on one
     // SMSP bind over the SM frontier (4 x 16 = 64), so the CTA's
     // dirty-until is 0 + 128; the fence at 4 holds the warp until
     // 128 + 9 + 5 = 142
    auto r = J("b20_smsp_accept");
    check(at(r, 0, 5) == 142, "b20 smsp drain", 142, at(r, 0, 5));
  }
  {  // (f) four warps on four SMSPs, 4 stores each: each SMSP
     // accepts 4 x 32 = 128 while the SM frontier serialises the 16
     // stores to 256; dirty-until = 0 + 256, fence holds until
     // 256 + 9 + 5 = 270
    auto r = J("b21_smsp_accept_sm");
    check(at(r, 0, 5) == 270, "b21 sm drain", 270, at(r, 0, 5));
  }
  {  // (g) two warps on the SAME SMSP, 2 stores each: the SMSP
     // frontier 4 x 32 = 128 binds over SM 4 x 16 = 64; warp1's
     // fence (after all four stores) holds until 128 + 9 + 5 = 142
    auto r = J("b22_smsp_accept_same");
    check(at(r, 1, 3) == 142, "b22 same-smsp", 142, at(r, 1, 3));
  }
  {  // (h) kind-80 row on a non-shared MACC is vetoed: the following
     // shared store (issued at 1) still opens the SMSP frontier
     // 1 -> 33; fence holds until 33 + 9 + 5 = 47
    auto r = J("b23_smsp_accept_nonshared");
    check(at(r, 0, 3) == 47, "b23 nonshared veto", 47, at(r, 0, 3));
  }
  {  // (i) CTA memory barrier on a dirty CTA: the MEMBAR after the
     // two stores is held until dirty-until (64) + its own service
     // latency (7) = 71; the proxy fence then waits 71 + 9 + 5 = 85
     // and clears the record, so the second MEMBAR issues at 85 held
     // only by its own 7 and FILL lands at 92
    auto r = J("b24_smsp_membar");
    check(at(r, 0, 3) == 71, "b24 membar->fence", 71, at(r, 0, 3));
    check(at(r, 0, 4) == 85, "b24 fence->membar2", 85, at(r, 0, 4));
    check(at(r, 0, 5) == 92, "b24 fill", 92, at(r, 0, 5));
  }
  {  // (j) MEMBAR with a clean CTA: only its own latency (7) floors
     // the warp
    auto r = J("b25_membar_clean");
    check(at(r, 0, 1) == 7, "b25 clean membar", 7, at(r, 0, 1));
  }
  {  // (k) fractional SMSP acceptance: 4 x 31.98 ends at 127.92 ->
     // dirty-until 128 (truncating would give 124 -> 138); fence
     // floor 128 + 9 + 5 = 142
    auto r = J("b26_smsp_accept_frac");
    check(at(r, 0, 5) == 142, "b26 frac drain", 142, at(r, 0, 5));
  }
  {  // (l) kind-81 reduce store: the sector cost ratio only scales
     // the store services -- the bulk group does NOT wait on the
     // DRAM fetch of the destination sectors (the fetch-hold rule of
     // line B is not taken over); both fixtures complete at
     // max(store finish, now + cycles(kind 38)) = 51 like b1.
    auto r = J("b27_reduce_store");
    check(at(r, 0, 3) == 51, "b27 reduce no hold", 51, at(r, 0, 3));
    auto r2 = J("b27b_reduce_store_norow");
    check(at(r2, 0, 3) == 51, "b27b reduce fill", 51, at(r2, 0, 3));
  }

  // ---- step c ----
  {
    auto r = J("c1_waves");
    for (int i = 0; i < 10; ++i) {
      check(at(r, 0, i) == (std::uint64_t)i, "c1 w0", i, at(r, 0, i));
      check(at(r, 1, i) == (std::uint64_t)i, "c1 w1", i, at(r, 1, i));
    }
    check(at(r, 2, 0) == 29, "c1 w2 entry", 29, at(r, 2, 0));
    check(at(r, 3, 0) == 29, "c1 w3 entry", 29, at(r, 3, 0));
  }
  {
    auto r = J("c1b_serial");
    check(at(r, 2, 0) == 29, "c1b w2 entry", 29, at(r, 2, 0));
    check(at(r, 3, 0) == 34, "c1b w3 entry", 34, at(r, 3, 0));
  }
  {
    auto r = J("c1c_first_wave");
    check(at(r, 0, 0) == 0, "c1c w0 entry", 0, at(r, 0, 0));
    check(at(r, 1, 0) == 3, "c1c w1 entry", 3, at(r, 1, 0));
  }
  {
    auto r = J("c1d_disabled");
    check(at(r, 0, 0) == 0 && at(r, 1, 0) == 0 && at(r, 2, 0) == 0 &&
          at(r, 3, 0) == 0, "c1d all at 0", 0, at(r, 3, 0));
  }
  expect_engine_error(dir + "/c1e_no_occupancy.job",
                      "missing_fact:occupancy_blocks_per_sm",
                      "c1e occupancy");
  {
    // trace-placed CTAs (no PLCE, CTAS.sm_id set) are first wave:
    // entry = rank * first_wave_interval in (sm, slot) order
    auto r = J("c1f_trace_placed");
    check(at(r, 0, 0) == 0, "c1f w0 entry", 0, at(r, 0, 0));
    check(at(r, 2, 0) == 3, "c1f w2 entry", 3, at(r, 2, 0));
    check(at(r, 1, 0) == 6, "c1f w1 entry", 6, at(r, 1, 0));
    check(at(r, 3, 0) == 9, "c1f w3 entry", 9, at(r, 3, 0));
  }
  {
    auto r = J("c2_cluster");
    check(at(r, 2, 0) == 29, "c2 w2 entry", 29, at(r, 2, 0));
    check(at(r, 3, 0) == 29, "c2 w3 entry", 29, at(r, 3, 0));
  }
  expect_engine_error(dir + "/c2b_no_topo.job",
                      "missing_fact:sm_topology", "c2b topo");
  expect_engine_error(dir + "/c2c_gpc_no_topo.job",
                      "missing_fact:sm_topology", "c2c gpc no topo");
  expect_engine_error(dir + "/c2d_gpc_row_no_gpc.job",
                      "missing_fact:sm_topology.gpc", "c2d gpc row");
  {
    // envelope_cycles linear in launch_warps_per_cta: block 128 = 4
    // warps -> 645, plus the 10-cycle fill timeline
    auto r = J("c3_envelope_linear");
    check(r.stats.cycles == 655, "c3 envelope linear", 655, r.stats.cycles);
  }
  {
    auto r = J("c3_blackwell");
    check(r.stats.cycles == 60, "c3 envelope", 60, r.stats.cycles);
  }
  {
    auto r = J("c3_dual_sm");
    check(r.stats.cycles == 40, "c3 dual_sm", 40, r.stats.cycles);
  }
  {
    auto r = J("c4_fetch");
    check(at(r, 0, 0) == 100, "c4 inst0 cold", 100, at(r, 0, 0));
    for (int i = 1; i < 8; ++i)
      check(at(r, 0, i) == (std::uint64_t)(100 + i), "c4 line0",
            100 + i, at(r, 0, i));
    check(at(r, 0, 8) == 200, "c4 inst8", 200, at(r, 0, 8));
    check(at(r, 0, 16) == 300, "c4 inst16", 300, at(r, 0, 16));
  }
  {
    // NOTE: capacity_l0=256 B = 2 lines < the 3 demand lines w0
    // filled, so line 0 is evicted at cycle 300; w1's per-line first
    // instance streams from L1 (stream_l1=10, serialized on the same
    // channel): 410, 428, 446 — the spec's "all L0 hits" reading
    // would need capacity_l0 >= 3 lines. Line 2 lands 2 earlier than
    // the pre-staging rule: w1's line-1 demand prefetches line 3 at
    // its own (cold) level, so it no longer spends an L1 slot
    auto r = J("c4b_shared_l0");
    for (int i = 0; i < 8; ++i) {
      check(at(r, 1, 400 + i) == (std::uint64_t)(410 + i),
            "c4b line0", 410 + i, at(r, 1, 400 + i));
      check(at(r, 1, 408 + i) == (std::uint64_t)(428 + i),
            "c4b line1", 428 + i, at(r, 1, 408 + i));
      check(at(r, 1, 416 + i) == (std::uint64_t)(446 + i),
            "c4b line2", 446 + i, at(r, 1, 416 + i));
    }
  }
  {
    // PLCE non-empty: CTAs 2/3 carry CTAS.sm_id but no PLCE row ->
    // they dispatch in wave 2 (F5)
    auto r = J("c5_plce_partial");
    check(at(r, 0, 0) == 0, "c5 w0 entry", 0, at(r, 0, 0));
    check(at(r, 1, 0) == 0, "c5 w1 entry", 0, at(r, 1, 0));
    check(tele2(r, cyclescope::kTele2DispatchWaves) == 2,
          "c5 waves", 2, tele2(r, cyclescope::kTele2DispatchWaves));
    check(at(r, 2, 0) >= 29, "c5 w2 entry", 29, at(r, 2, 0));
    check(at(r, 3, 0) >= 29, "c5 w3 entry", 29, at(r, 3, 0));
  }
  {
    // per-slot refill: CTA0 completes at 9, CTA1 at 14, stagger <
    // slot_free_to_entry (20); each queued CTA enters 20 after the
    // release of ITS slot -> w2 at 29, w3 at 34 (a shared tick that
    // refills both slots at 29 would give w3 at 30 = 29 + serial)
    auto r = J("c6_staggered_release");
    check(at(r, 2, 0) == 29, "c6 w2 entry", 29, at(r, 2, 0));
    check(at(r, 3, 0) == 34, "c6 w3 entry", 34, at(r, 3, 0));
  }
  {
    // fractional serial_per_cta (1.5): four slots released on the
    // same cycle refill at 20 + ceil(k * 1.5) = 29, 31, 32, 34
    // (truncating serial to 1 would give 29, 30, 31, 32)
    auto r = J("c7_frac_serial");
    check(at(r, 4, 0) == 29, "c7 w4 entry", 29, at(r, 4, 0));
    check(at(r, 5, 0) == 31, "c7 w5 entry", 31, at(r, 5, 0));
    check(at(r, 6, 0) == 32, "c7 w6 entry", 32, at(r, 6, 0));
    check(at(r, 7, 0) == 34, "c7 w7 entry", 34, at(r, 7, 0));
  }
  {
    // dependents_launch (kind 40): all 8 CTAs carry it; the four
    // first-wave CTAs complete together at 10 (the PREEXIT instance
    // adds one instruction) and each completion charges the
    // serializer
    //   last = max(now - serial, last) + dependents_launch_serial
    // -> last = 10, 11, 12, 13.  The slots release at 10 + slot 2 =
    // 12; refills enter at max(12, last + 1) = 14, 15, 16, 17
    // (without the marker they would enter at 11, 12, 13, 14)
    auto r = J("c8_dependents_launch");
    check(at(r, 4, 0) == 14, "c8 w4 entry", 14, at(r, 4, 0));
    check(at(r, 5, 0) == 15, "c8 w5 entry", 15, at(r, 5, 0));
    check(at(r, 6, 0) == 16, "c8 w6 entry", 16, at(r, 6, 0));
    check(at(r, 7, 0) == 17, "c8 w7 entry", 17, at(r, 7, 0));
  }
  {
    // control: same fixture without the kind-40 instances -> no
    // serializer charge, refills at release + serial = 11..14
    auto r = J("c8_no_marker");
    check(at(r, 4, 0) == 11, "c8n w4 entry", 11, at(r, 4, 0));
    check(at(r, 5, 0) == 12, "c8n w5 entry", 12, at(r, 5, 0));
    check(at(r, 6, 0) == 13, "c8n w6 entry", 13, at(r, 6, 0));
    check(at(r, 7, 0) == 14, "c8n w7 entry", 14, at(r, 7, 0));
  }
  {
    // slow gpc: SM 1 (gpc 1, in slow_gpc_mask) refills at its CTA's
    // completion (~9) + slot_slow 50 = 59; SM 0 (gpc 0) at its CTA's
    // completion (~79) + slot 20 = 99
    auto r = J("c9_slow_gpc");
    check(at(r, 2, 0) == 59, "c9 w2 entry", 59, at(r, 2, 0));
    check(at(r, 3, 0) == 99, "c9 w3 entry", 99, at(r, 3, 0));
  }
  {
    // stale failed poll: phase released before the poll issues ->
    // result latency 0, dependent issues one cycle later
    auto r = J("a14_stale_poll");
    check(tele(r, cyclescope::kTeleStalePolls) == 1,
          "a14 stale", 1, tele(r, cyclescope::kTeleStalePolls));
    check(at(r, 0, 3) == at(r, 0, 2) + 1, "a14 dep", at(r, 0, 2) + 1,
          at(r, 0, 3));
  }
  {
    // access_kind 0xC0 (async copy): load path, 64 cycles of DRAM
    // port (128 cold sectors x 0.5, every sector of a cold line is a
    // DRAM sector since L2 presence is sector-granular) + the
    // 200-cycle DRAM total latency, not ~128 on the 1-sector/cycle
    // store port
    auto r = J("a15_macc_kind");
    check(r.stats.cycles >= 256 && r.stats.cycles <= 272,
          "a15 cycles", 264, r.stats.cycles);
  }
  {
    // TMA load with a completion leaf (50) over 128 cold sectors: the
    // mbarrier phase completes at leaf + hierarchy excess over an L2
    // hit (hierarchy return ~264 as in a15, minus the 30-cycle L2 hit),
    // i.e. ~20 later than a15's max(leaf, hierarchy) would give
    auto r = J("a15b_tma_leaf_excess");
    check(r.stats.cycles >= 280 && r.stats.cycles <= 296,
          "a15b cycles", 287, r.stats.cycles);
  }
  {
    // tma_cold rule: leaf with cold_extra 300 over 128 cold sectors ->
    // start + 50 + 1.0 x 300 (~352); the hierarchy return is ignored
    auto r = J("a15c_tma_cold_leg");
    check(r.stats.cycles >= 345 && r.stats.cycles <= 365,
          "a15c cycles", 352, r.stats.cycles);
  }
  {
    // two cold-leg TMAs in the same cycle behind a 32 B/cycle shared
    // DRAM service: each completion = leaf (350) + its queueing in the
    // shared service (64 / 128; the second transfer's per-SM port wait
    // is not charged) -> the second phase releases at ~478; without
    // the excess both released at ~352
    auto r = J("a15d_tma_shared_queue");
    check(r.stats.cycles >= 468 && r.stats.cycles <= 495,
          "a15d cycles", 480, r.stats.cycles);
  }
  {
    // kind 65 (storage.global.store_issue): 4 STG of 16 sectors with a
    // 23-cycle per-SMSP interval -> the 4th issues >= 69 after the
    // first; the second warp on SMSP 1 is not delayed by SMSP 0
    // (kernel end ~128: the 8 stores' 128 sectors drain through the
    // one SM store port at its default 1 sector / cycle)
    auto r = J("a30_store_issue_interval");
    check(r.stats.cycles >= 120 && r.stats.cycles <= 140,
          "a30 cycles", 128, r.stats.cycles);
    check(at(r, 0, 3) >= at(r, 0, 0) + 69, "a30 4th store",
          at(r, 0, 0) + 69, at(r, 0, 3));
    check(at(r, 1, 3) >= at(r, 1, 0) + 69 &&
              at(r, 1, 3) <= at(r, 1, 0) + 72,
          "a30 smsp independence", at(r, 1, 0) + 69, at(r, 1, 3));
  }
  {
    // rule S fairness (store_flood replay): two warps on SMSPs 0/1
    // each with 12 single-sector lane STG, store_l2 holding 4
    // outstanding requests at 4 cycles of port service each.  Once
    // the buffer fills the grants must rotate across the queued
    // warps (the pre-change engine let one warp drain its whole
    // stream first); ordering changes but throughput does not:
    // total cycles equal the pre-change value 102 (+/- 1).
    auto r = J("a31_store_gate_fair");
    check(r.stats.cycles >= 101 && r.stats.cycles <= 103,
          "a31 cycles", 102, r.stats.cycles);
    // Each warp's insts 0..11 are STG (inst 12 is the trailing FILL).
    // Collect them in issue order and require strict alternation once
    // the buffer is full: entries at cycle >= 4 (all 4 outstanding
    // slots taken) must not repeat a warp consecutively.
    std::vector<std::pair<std::uint64_t, std::uint32_t>> stg;
    std::uint64_t last0 = 0, last1 = 0;
    for (const auto& j : r.stats.journal) {
      if (j.warp > 1 || j.instance >=
              r.job.warps[j.warp].first_instance + 12)
        continue;
      stg.push_back({j.issue_cycle, j.warp});
      (j.warp == 0 ? last0 : last1) = j.issue_cycle;
    }
    std::stable_sort(stg.begin(), stg.end());
    check(stg.size() == 24, "a31 stg count", 24, stg.size());
    unsigned viol = 0;
    for (unsigned i = 1; i < stg.size(); ++i)
      if (stg[i].first >= 4 && stg[i].second == stg[i - 1].second)
        ++viol;
    check(viol == 0, "a31 alternation", 0, viol);
    const std::uint64_t spread =
        last0 > last1 ? last0 - last1 : last1 - last0;
    check(spread <= 8, "a31 last-issue spread", 8, spread);
  }
  {
    // 4-warp version on SMSPs 0..3: only the last-issue proximity is
    // asserted (all four last STG within 4 port-service intervals).
    auto r = J("a31b_store_gate_fair_w4");
    check(r.stats.cycles >= 197 && r.stats.cycles <= 199,
          "a31b cycles", 198, r.stats.cycles);
    std::uint64_t last[4] = {0, 0, 0, 0};
    unsigned seen[4] = {0, 0, 0, 0};
    for (const auto& j : r.stats.journal) {
      if (j.warp > 3 || j.instance >=
              r.job.warps[j.warp].first_instance + 12)
        continue;
      last[j.warp] = j.issue_cycle;
      ++seen[j.warp];
    }
    check(seen[0] + seen[1] + seen[2] + seen[3] == 48,
          "a31b stg count", 48, seen[0] + seen[1] + seen[2] + seen[3]);
    std::uint64_t lo = last[0], hi = last[0];
    for (int i = 1; i < 4; ++i) {
      lo = std::min(lo, last[i]);
      hi = std::max(hi, last[i]);
    }
    check(hi - lo <= 16, "a31b last-issue spread", 16, hi - lo);
  }
  {
    // tma_oob v2 cut-row rule: (a) no cut_rows fact -> identical to
    // the pre-change timing; (b) cut_rows = 128 adds the 178-cycle
    // cut_extra to the completion; (c) 4 cut TMAs on one SM start
    // their service 259 cycles apart (the cut_service of the
    // predecessor bounds the next start), an uncut stream stays at
    // the 10-cycle kind-21 interval; (d) cut_rows = 32 ->
    // max(cyc21, 65) = 65.
    auto base1 = J("a32_tma_uncut");
    check(base1.stats.cycles >= 52 && base1.stats.cycles <= 54,
          "a32 cycles", 53, base1.stats.cycles);
    auto cut = J("a32b_tma_cut_extra");
    check(cut.stats.cycles >= 230 && cut.stats.cycles <= 232,
          "a32b cycles", 231, cut.stats.cycles);
    check(cut.stats.cycles == base1.stats.cycles + 178,
          "a32b +178", base1.stats.cycles + 178, cut.stats.cycles);
    auto ctrl4 = J("a32e_tma_uncut4");
    check(ctrl4.stats.cycles >= 82 && ctrl4.stats.cycles <= 84,
          "a32e cycles", 83, ctrl4.stats.cycles);
    for (unsigned k = 0; k < 3; ++k)
      check(at(ctrl4, k + 1, 2) - at(ctrl4, k, 2) == 10,
            "a32e service gap", 10,
            at(ctrl4, k + 1, 2) - at(ctrl4, k, 2));
    auto c = J("a32c_tma_cut_service");
    check(c.stats.cycles >= 829 && c.stats.cycles <= 831,
          "a32c cycles", 830, c.stats.cycles);
    for (unsigned k = 0; k < 3; ++k)
      check(at(c, k + 1, 2) - at(c, k, 2) >= 258 &&
                at(c, k + 1, 2) - at(c, k, 2) <= 260,
            "a32c service gap", 259,
            at(c, k + 1, 2) - at(c, k, 2));
    auto d = J("a32d_tma_cut_service32");
    check(d.stats.cycles >= 247 && d.stats.cycles <= 249,
          "a32d cycles", 248, d.stats.cycles);
    for (unsigned k = 0; k < 3; ++k)
      check(at(d, k + 1, 2) - at(d, k, 2) >= 64 &&
                at(d, k + 1, 2) - at(d, k, 2) <= 66,
            "a32d service gap", 65,
            at(d, k + 1, 2) - at(d, k, 2));
  }
  {
    // template-level shared acceptance rows (kind 6) serialize per SM:
    // 4 LDS x 40 cycles -> >= 160
    auto r = J("a18_shared_rows");
    check(r.stats.cycles >= 160 && r.stats.cycles <= 200,
          "a18 cycles", 165, r.stats.cycles);
  }
  {
    // class-bound LD template (generic pointer) whose MACCs carry
    // shared space (flags 0x1): kind-6 shared acceptance rows apply
    // exactly like a18's explicit LDS -> >= 160
    auto r = J("a18b_shared_rows_generic");
    check(r.stats.cycles >= 160 && r.stats.cycles <= 200,
          "a18b cycles", 165, r.stats.cycles);
  }
  {
    // same class-bound LD template but the MACCs resolve to global
    // space (flags 0x0): a non-shared MACC vetoes the kind-6 rows,
    // so only the per-instruction service latency remains
    auto r = J("a18c_shared_rows_global");
    check(r.stats.cycles < 60, "a18c cycles", 60, r.stats.cycles);
  }
  {
    // access_kind 0x100 (async reduce): store path acceptance
    auto r = J("a16_macc_reduce");
    check(r.stats.cycles < 60, "a16 cycles", 60, r.stats.cycles);
  }
  {
    // divergent failed polls: NANOSLEEP.SYNCS skips the known-failed
    // object-1 poll, sleeps on {0, 1} and wakes at object 0's release
    // (~warp1's arrive after 40 fills), not the 1e6-cycle hint
    auto r = J("a17_divergent_poll");
    check(r.stats.cycles < 500, "a17 cycles", 500, r.stats.cycles);
    check(at(r, 0, 5) <= at(r, 1, 40) + 4, "a17 wake", at(r, 1, 40) + 4,
          at(r, 0, 5));
  }
  {
    // divergent poll region spanning 40+ instances with an arrive in
    // the middle: NANOSLEEP.SYNCS still sees {obj0, obj1} plus the
    // region-closing obj0 wait, and wakes at obj0's release
    auto r = J("a17b_divergent_far");
    check(r.stats.cycles < 500, "a17b cycles", 500, r.stats.cycles);
    check(at(r, 0, 46) <= at(r, 1, 60) + 4, "a17b wake",
          at(r, 1, 60) + 4, at(r, 0, 46));
  }
  {
    // retried failed poll (same (object,parity) check follows): no
    // trywait_timeout hold — next FILL issues within a few cycles
    auto r = J("a18_poll_retry_free");
    check(tele(r, cyclescope::kTeleAbandonedPolls) == 0, "a18 abandoned", 0,
          tele(r, cyclescope::kTeleAbandonedPolls));
    check(at(r, 0, 1) < at(r, 0, 0) + 10, "a18 fill", at(r, 0, 0) + 10,
          at(r, 0, 1));
    check(at(r, 0, 2) >= at(r, 1, 50), "a18 release", at(r, 1, 50),
          at(r, 0, 2));
  }
  {
    // abandoned failed poll (no later check): holds trywait_timeout
    auto r = J("a19_poll_abandoned");
    check(tele(r, cyclescope::kTeleAbandonedPolls) == 1, "a19 abandoned", 1,
          tele(r, cyclescope::kTeleAbandonedPolls));
    check(at(r, 0, 1) >= at(r, 0, 0) + 7892, "a19 fill",
          at(r, 0, 0) + 7892, at(r, 0, 1));
  }
  {
    // STAS object_list: each destination CTA's barrier gets its 4 tx
    // bytes -> both waits release, no mbarrier_overrun
    auto r = J("a20_stas_lane_objects");
    check(r.stats.cycles < 500, "a20 cycles", 500, r.stats.cycles);
    check(r.stats.completed == r.stats.issued, "a20 issued==completed",
          r.stats.issued, r.stats.completed);
    check(at(r, 1, 3) > 0, "a20 cta1 wait release", 1, at(r, 1, 3));
    check(at(r, 2, 3) > 0, "a20 cta2 wait release", 1, at(r, 2, 3));
  }
  {
    // schema-24 leaf without enabled -> dispatch on: wave 2 enters
    // after first-wave completion + slot_free_to_entry
    auto r = J("c7_dispatch_no_enabled");
    check(tele2(r, cyclescope::kTele2DispatchWaves) == 2,
          "c7 waves", 2, tele2(r, cyclescope::kTele2DispatchWaves));
    check(at(r, 2, 0) >= 29, "c7 w2 entry", 29, at(r, 2, 0));
  }
  {
    auto r = J("c4c_no_code");
    check(tele2(r, cyclescope::kTele2FetchUnavailable) == 1,
          "c4c unavailable", 1,
          tele2(r, cyclescope::kTele2FetchUnavailable));
  }
  {
    // sharing=1: L0 is per SMSP — w1's own L0 is empty even though
    // w0 filled the SM's L1, so its per-line first instance streams
    // from L1 (10c, serialized on the shared channel). Under the
    // staging rule w1's line-0 demand sequentially prefetches the
    // L1-resident lines 1 and 2 at 420/430, and those slots beat the
    // old serialized demands
    auto r = J("c4d_sharing_smsp");
    for (int i = 0; i < 8; ++i) {
      check(at(r, 1, 400 + i) == (std::uint64_t)(410 + i),
            "c4d line0", 410 + i, at(r, 1, 400 + i));
      check(at(r, 1, 408 + i) == (std::uint64_t)(420 + i),
            "c4d line1", 420 + i, at(r, 1, 408 + i));
      check(at(r, 1, 416 + i) == (std::uint64_t)(430 + i),
            "c4d line2", 430 + i, at(r, 1, 416 + i));
    }
  }
  {
    // schema-27 capacity read: LDC 0x200 fills line 8 (miss), 0x300
    // misses line 12, 0x210 hits the resident line 8 -> 1 hit / 2
    // misses; the dropped CAPACITY_BYTES field made all three miss
    auto r = J("c8_const_capacity");
    check(tele4(r, 13) == 1, "c8 const hits", 1, tele4(r, 13));
    check(tele4(r, 14) == 2, "c8 const misses", 2, tele4(r, 14));
  }
  {
    // in-flight fill is not a hit: w1's two LDCs on the missed line
    // (one hit-path, one same-line) must wait for the fill ready at
    // ~t+406 instead of paying the 107 / 37-cycle latencies, so w1's
    // dependent DEP cannot issue before the fill completes; w0's
    // re-read after the fill is a real hit -> 2 hits / 2 misses /
    // 1 same_line.  Either warp can be the misser, so the bound is on
    // the earlier LDC issue.
    auto r = J("c8b_const_inflight");
    check(tele4(r, 13) == 2, "c8b const hits", 2, tele4(r, 13));
    check(tele4(r, 14) == 2, "c8b const misses", 2, tele4(r, 14));
    check(tele4(r, 12) == 1, "c8b const same_line", 1, tele4(r, 12));
    const std::uint64_t t0 = std::min(at(r, 0, 0), at(r, 1, 0));
    check(at(r, 1, 2) >= t0 + 400, "c8b w1 waits for fill",
          t0 + 400, at(r, 1, 2));
  }

  {
    // bank0_access_cycles field present (F = 64): both bank-0
    // accesses return issue + 64 (counted as hits, no resident /
    // same-line state); bank-3 still miss then hit.
    auto r = J("c8c_bank0");
    check(tele4(r, 13) == 2, "c8c const hits", 2, tele4(r, 13));
    check(tele4(r, 14) == 1, "c8c const misses", 1, tele4(r, 14));
    check(tele4(r, 12) == 1, "c8c const same_line", 1, tele4(r, 12));
    // w0: two bank-0 LDCs, both charged F=64 (issue+64 ready) --
    // the second is not a 406 miss nor does it wait on a fill.
    check(at(r, 0, 1) <= at(r, 0, 0) + 80,
          "c8c w0 repeated bank-0 stays ~F",
          at(r, 0, 0) + 80, at(r, 0, 1));
    // w1 bank-3: first access misses (406), DEP waits on it.
    check(at(r, 1, 2) >= at(r, 1, 0) + 400,
          "c8c w1 bank-3 miss latency kept",
          at(r, 1, 0) + 400, at(r, 1, 2));
  }

  {
    // field 6 (vector=64) + field 7 (uniform=96): w0 uniform bank-0
    // accesses return ~96, w1 vector bank-0 ~64, w2 bank-3 misses.
    auto r = J("c8d_bank0_uniform");
    check(tele4(r, 13) == 4, "c8d const hits", 4, tele4(r, 13));
    check(tele4(r, 14) == 1, "c8d const misses", 1, tele4(r, 14));
    check(tele4(r, 12) == 1, "c8d const same_line", 1, tele4(r, 12));
    check(at(r, 0, 2) >= at(r, 0, 0) + 90,
          "c8d w0 uniform bank-0 ~96",
          at(r, 0, 0) + 90, at(r, 0, 2));
    check(at(r, 1, 2) >= at(r, 1, 0) + 60 &&
              at(r, 1, 2) < at(r, 0, 0) + 90,
          "c8d w1 vector bank-0 ~64",
          at(r, 1, 0) + 64, at(r, 1, 2));
    check(at(r, 2, 2) >= at(r, 2, 0) + 400,
          "c8d w2 bank-3 miss latency kept",
          at(r, 2, 0) + 400, at(r, 2, 2));
  }

  {
    // register_file.operand_read (OP2p): one warp, 16 3-read FFMAs,
    // 2 reads/cycle pipelined -> issue = floor(1.5 * i)
    auto r = J("op1_stream3");
    // OP2p issue pattern 0,1,2,3,4,6,7,9,10,...: steady state is
    // 3 cycles per 2 instructions (1.5/inst average)
    for (std::uint64_t i = 4; i + 2 < 16; ++i)
      check(at(r, 0, i + 2) - at(r, 0, i) == 3,
            "op1 stream3 3/2inst", 3, at(r, 0, i + 2) - at(r, 0, i));
    check(at(r, 0, 15) - at(r, 0, 0) <= 24 && at(r, 0, 15) > 15,
          "op1 stream3 span", 24, at(r, 0, 15) - at(r, 0, 0));
  }
  {
    // W warps' 3-read dependent chains on one SMSP -> 2*W per step
    for (int w = 2; w <= 4; ++w) {
      char name[32];
      std::snprintf(name, sizeof name, "op2_dep3_w%d", w);
      auto r = J(name);
      for (int k = 0; k < w; ++k) {
        const std::uint64_t span = at(r, k, 15) - at(r, k, 0);
        check(span >= 15ull * 2 * w - 4 && span <= 15ull * 2 * w + 2,
              "op2 dep3 ~2W/step", 15ull * 2 * w, span);
      }
    }
  }
  {
    // every source row reuse-served -> the collector is bypassed
    auto r = J("op3_served");
    check(at(r, 0, 15) - at(r, 0, 0) == 15, "op3 served 1/inst",
          15, at(r, 0, 15) - at(r, 0, 0));
  }
  {
    // 2 reads fit one cycle -> 1 per instruction
    auto r = J("op4_reads2");
    check(at(r, 0, 15) - at(r, 0, 0) == 15, "op4 reads2 1/inst",
          15, at(r, 0, 15) - at(r, 0, 0));
  }
  {
    // served consumer after a 3-read producer: the chain is
    // latency-bound (8 cycles per producer/consumer pair); W=2
    // interleaves on the collector with bounded extra delay
    auto r1 = J("op5_pair_w1");
    const std::uint64_t s1 = at(r1, 0, 14) - at(r1, 0, 0);
    check(s1 >= 7 * 8 - 4 && s1 <= 7 * 8 + 4, "op5 w1 pair ~8", 56,
          s1);
    auto r2 = J("op5_pair_w2");
    for (int k = 0; k < 2; ++k) {
      const std::uint64_t s2 = at(r2, k, 14) - at(r2, k, 0);
      check(s2 >= 7 * 8 - 4 && s2 <= 7 * 8 + 24, "op5 w2 pair ~8+",
            56, s2);
    }
  }

  {
    // far-side ports never floor the read barrier: 8 x 16-sector
    // misses queue ~4 cycles each on the L2 port, but with no L1 port
    // in the leaf the barrier frees at issue -> FILL = last + 1
    // (until 2026-11-08 the L2-port start held it ~7x4 later)
    auto r = J("a28_read_barrier_global");
    check(at(r, 0, 8) == at(r, 0, 7) + 1, "a28 fill at last+1",
          at(r, 0, 7) + 1, at(r, 0, 8));
  }
  {
    // merged followers never reach a port -> rbar frees at issue
    auto r = J("a28b_read_barrier_merged");
    check(at(r, 0, 8) == at(r, 0, 7) + 1, "a28b fill at last+1",
          at(r, 0, 7) + 1, at(r, 0, 8));
  }
  {
    // stores: 8 x 16-sector STGs take 16 cycles each on the store-L2
    // port (queue ~100 cycles deep by the 8th), the read barrier
    // still frees at issue -> FILL = last + 1
    auto r = J("a28c_read_barrier_store");
    check(at(r, 0, 8) == at(r, 0, 7) + 1, "a28c fill at last+1",
          at(r, 0, 7) + 1, at(r, 0, 8));
  }
  {
    // m_l1_port: 8 hit loads of 16 sectors, l1 leaf 128 B/cy -> ~4
    // cycles on the L1 port each; rbar serializes issue on
    // acceptance -> FILL lands ~7*4+4 after the first hit load
    auto r = J("m_l1_port");
    const std::uint64_t first = at(r, 0, 2), last = at(r, 0, 9);
    const std::uint64_t fill = at(r, 0, 10);
    check(fill > last, "l1port fill waits rbar", last + 1, fill);
    check(fill - first >= 26 && fill - first <= 60,
          "l1port fill ~7*4 later", 32, fill - first);
  }
  {
    // NANOSLEEP.SYNCS wake = release + syncs_wake_cycles (row f4=30):
    // a10 shape wakes at 40 -> here 70
    auto r = J("m_syncs_wake");
    check(at(r, 0, 2) == at(r, 1, 40) + 30, "syncs wake +30",
          at(r, 1, 40) + 30, at(r, 0, 2));
  }
  {
    // cp.async group completion anchored at first copy issue:
    // 4 copies ~10 apart, leaf 300 -> wait at first+300 (not 330)
    auto r = J("m_cpasync_anchor");
    const std::uint64_t first = at(r, 0, 0), wait = at(r, 0, 8);
    const std::uint64_t nxt = at(r, 0, 9);
    check(nxt - first >= 295 && nxt - first <= 320,
          "cpasync anchor first+300", first + 300, nxt - first);
  }
  {
    // YIELD result 40 -> next issue at +40 (no-destination floor)
    auto r = J("m_control_yield");
    const std::uint64_t y = at(r, 0, 2), nxt = at(r, 0, 3);
    check(nxt >= y + 40 && nxt <= y + 44, "yield floor +40",
          y + 40, nxt);
  }
  {
    // leaf 0 -> L1 port off -> acceptance at issue -> fill = last+1
    auto r = J("m_l1_port_off");
    check(at(r, 0, 10) == at(r, 0, 9) + 1, "l1port_off last+1",
          at(r, 0, 9) + 1, at(r, 0, 10));
  }

  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
