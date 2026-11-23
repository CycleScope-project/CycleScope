// sync_test — engine 3-3 step a: synthetic sync fixtures driven
// through Simulator with a journal.  argv[1] = fixtures directory.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

// TELE group 2 last_completion_cycle (kernel_boundary K2: the old
// cycles definition = last completion + envelope)
std::uint64_t last_completion(const Run& r) {
  for (const auto& t : r.stats.tele)
    if (t.group == 2 &&
        t.counter == cyclescope::kTele2LastCompletionCycle)
      return t.value;
  return 0;
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
    // G1: the wait issues at once; the release lives on its result —
    // the no-barrier template holds the next issue until it
    check(at(r, 0, 6) == at(r, 0, 5) + 1, "a1 wait issues at once",
          at(r, 0, 5) + 1, at(r, 0, 6));
    check(at(r, 0, 7) == expect + 1, "a1 post-wait at result",
          expect + 1, at(r, 0, 7));
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
    // G1: the success wait issues at once; the release lives on its
    // result, which holds the following FILL at completion + release
    check(at(r, 0, 1) == 1, name, 1, at(r, 0, 1));
    check(at(r, 0, 2) == arrive + 11, name, arrive + 11, at(r, 0, 2));
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
    check(at(r, 0, 1) == 1, "a21 wait obj2 issues at once", 1,
          at(r, 0, 1));
    check(at(r, 0, 2) == expect + 1, "a21 post at result", expect + 1,
          at(r, 0, 2));
  }

  // ---- A22: re-init of an idle barrier wins -------------------------
  {
    auto r = J("a22_reinit_idle");
    // wait on object 0 (expected 2 after re-init) issues at once; the
    // FILL behind it lands at the second arrive + release leaf
    const std::uint64_t expect = at(r, 2, 20) + 10;
    check(at(r, 0, 2) == 2, "a22 wait issues at once", 2, at(r, 0, 2));
    check(at(r, 0, 3) == expect + 1, "a22 post at result", expect + 1,
          at(r, 0, 3));
  }
  expect_engine_error(dir + "/a22b_reinit_busy.job",
                      "unsupported:engine:mbarrier_reinit", "a22b busy");

  // ---- A23: complete_tx closes a tx-blocked phase -------------------
  {
    auto r = J("a23_complete_tx");
    // the wait issues at once; the FILL behind it lands at the
    // complete_tx instance + release leaf
    const std::uint64_t expect = at(r, 1, 6) + 10;
    check(at(r, 0, 1) == 1, "a23 wait issues at once", 1, at(r, 0, 1));
    check(at(r, 0, 2) == expect + 1, "a23 post at result", expect + 1,
          at(r, 0, 2));
  }
  expect_engine_error(dir + "/a23b_complete_tx_zero.job",
                      "missing_fact:mbarrier_arrive_count",
                      "a23b zero tx");

  // ---- issue_rules v1: R1-R3 ----------------------------------------
  {
    // R1: source-register readiness is not an issue term — the
    // reader issues the cycle after its producer (stall floor 1)
    // even though the producer's result is only ready at +10
    auto r = J("ir1_no_register_tracking");
    check(at(r, 0, 1) == at(r, 0, 0) + 1, "ir1 reader next cycle",
          at(r, 0, 0) + 1, at(r, 0, 1));
  }
  {
    // R2: barrier releases max — the reader waits the later of the
    // two producers' data_ready (+40), not the first (+20)
    auto r = J("ir2_barrier_latest");
    check(at(r, 0, 2) >= at(r, 0, 0) + 40, "ir2 latest release",
          at(r, 0, 0) + 40, at(r, 0, 2));
  }
  {
    // R3: the raise at t is invisible to the t+1 issue — with a
    // stall-1 load the reader issues at t+1, unheld
    auto r = J("ir3_load_s1");
    check(at(r, 0, 1) == at(r, 0, 0) + 1, "ir3 s1 reader at t+1",
          at(r, 0, 0) + 1, at(r, 0, 1));
  }
  {
    // R3: with a stall-2 load the reader is evaluated at t+2, sees
    // the raise, and waits until data_ready (t+50)
    auto r = J("ir3_load_s2");
    check(at(r, 0, 1) >= at(r, 0, 0) + 50, "ir3 s2 reader waits",
          at(r, 0, 0) + 50, at(r, 0, 1));
  }
  {
    // R3: one stall-1 instruction between load and reader — the
    // reader is again evaluated at t+2 and waits data_ready
    auto r = J("ir3_load_s1g1");
    check(at(r, 0, 2) >= at(r, 0, 0) + 50, "ir3 s1g1 reader waits",
          at(r, 0, 0) + 50, at(r, 0, 2));
  }

  // ---- A25: control-word stall floor --------------------------------
  {
    auto r = J("a25_control_stall");
    // stall=6 after inst0 pushes inst1 by 6; stall=1 adds nothing;
    // inst3 (stall=2, result latency 4 -> R5): issue_rules v1 R1
    // removed the register floor, so inst4 waits the stall floor
    // (+2) instead of the register floor (+4)
    check(at(r, 0, 1) == at(r, 0, 0) + 6, "a25 stall6",
          at(r, 0, 0) + 6, at(r, 0, 1));
    check(at(r, 0, 2) == at(r, 0, 1) + 1, "a25 stall1",
          at(r, 0, 1) + 1, at(r, 0, 2));
    check(at(r, 0, 4) == at(r, 0, 3) + 2, "a25 stall-floor-only",
          at(r, 0, 3) + 2, at(r, 0, 4));
  }

  // ---- issue_rules v2 Y1: yield hint + stall 1 -----------------------
  {
    auto r = J("ir4_yield_stall");
    check(at(r, 0, 1) == at(r, 0, 0) + 2, "y1 yield+stall1",
          at(r, 0, 0) + 2, at(r, 0, 1));
    check(at(r, 0, 2) == at(r, 0, 1) + 1, "y1 stall1 no yield",
          at(r, 0, 1) + 1, at(r, 0, 2));
    check(at(r, 0, 3) == at(r, 0, 2) + 4, "y1 yield+stall4",
          at(r, 0, 2) + 4, at(r, 0, 3));
    check(at(r, 0, 4) == at(r, 0, 3) + 1, "y1 tail stall1",
          at(r, 0, 3) + 1, at(r, 0, 4));
  }

  // ---- A26: failed polls carry no result latency ---------------------
  {
    auto r = J("a26_failed_poll_result");
    // failed poll (inst1): dependent BRA at +1, not +38.  The
    // successful wait's 38-cycle result latency still lands in
    // reg_ready but issue_rules v1 R1 no longer holds issue on it
    check(at(r, 0, 2) == at(r, 0, 1) + 1, "a26 failed-poll bra",
          at(r, 0, 1) + 1, at(r, 0, 2));
    // G1: the successful wait issues at once but raises no dependency
    // barrier, so the warp's next issue is held until its result —
    // arrive issue + result latency (release.cycles is 0 here)
    check(at(r, 0, 4) == at(r, 1, 20) + 38, "a26 success-wait result",
          at(r, 1, 20) + 38, at(r, 0, 4));
  }

  // ---- constant_paths v1 (schema-27 fields 12-21) ------------------
  {
    // C4: uniform no-write-barrier cold line is held at the load for
    // the first touch; constant_paths v2 C5: one occupied SMSP ->
    // 129 x max(2, 1) / 4 = 64. The same-line second load and the
    // reader are free
    auto r = J("cp1_uhold");
    const std::uint64_t t0 = at(r, 0, 0);
    check(t0 >= 60, "cp1 held uniform fill", 64, t0);
    check(at(r, 0, 1) <= t0 + 3, "cp1 same-line load free",
          t0 + 3, at(r, 0, 1));
    check(at(r, 0, 2) <= at(r, 0, 1) + 2, "cp1 reader free",
          at(r, 0, 1) + 2, at(r, 0, 2));
    // the constant_hold reason itself is checked with the ledger
    // fixtures below
  }
  {
    // C2/C4 vector path: the reader waits the barrier for
    // first_touch (~92); the resident reread waits reread (~22)
    auto r = J("cp2_vbarrier");
    const std::uint64_t t0 = at(r, 0, 0);
    check(at(r, 0, 2) >= t0 + 90, "cp2 first touch at reader",
          t0 + 90, at(r, 0, 2));
    check(at(r, 0, 5) >= at(r, 0, 3) + 20,
          "cp2 reread at reader", at(r, 0, 3) + 20, at(r, 0, 5));
    check(at(r, 0, 5) <= at(r, 0, 3) + 60,
          "cp2 reread upper", at(r, 0, 3) + 60, at(r, 0, 5));
  }
  {
    // C1: separate resident sets — w0's uniform fill of A does not
    // make A resident for the vector path (vector_first_touch 100);
    // w1's vector fill of B does not help the held uniform read of B
    // (uniform_first_touch 69)
    auto r = J("cp3_sets");
    check(at(r, 0, 0) >= 60, "cp3 w0 uniform fill held", 69,
          at(r, 0, 0));
    check(at(r, 0, 3) >= at(r, 0, 1) + 95,
          "cp3 vector still first touch",
          at(r, 0, 1) + 95, at(r, 0, 3));
    check(at(r, 1, 1) >= at(r, 1, 0) + 60,
          "cp3 uniform still held",
          at(r, 1, 0) + 60, at(r, 1, 1));
  }
  {
    // C2: four adjacent vector cold loads on one barrier complete at
    // t0+92, +119, +146, +173 (fill interval 27); the reader waits
    // the last
    auto r = J("cp4_batch");
    const std::uint64_t t0 = at(r, 0, 0);
    check(at(r, 0, 5) >= t0 + 170, "cp4 batch last fill",
          t0 + 170, at(r, 0, 5));
  }
  {
    // C3: the first line of an unfetched GPC block pays
    // block_fetch + first_touch (~695); the cross-SM access joins
    // the in-flight block; the second line of the fetched block
    // pays first_touch only (~100)
    auto r = J("cp5_block");
    const std::uint64_t t0 = at(r, 0, 0);
    check(at(r, 0, 2) >= t0 + 680, "cp5 block fetch at reader",
          t0 + 680, at(r, 0, 2));
    check(at(r, 0, 5) >= at(r, 0, 3) + 95 &&
              at(r, 0, 5) <= at(r, 0, 3) + 150,
          "cp5 second line first touch",
          at(r, 0, 3) + 100, at(r, 0, 5));
    check(at(r, 1, 2) >= t0 + 680,
          "cp5 cross-SM in-flight block", t0 + 680, at(r, 1, 2));
  }
  {
    // C5 (v2): uniform first touch = 129 x max(2, occupied SMSPs)
    // / 4 -> 64 / 64 / 96 / 129 for 1 / 2 / 3 / 4 occupied SMSPs
    auto r1 = J("cp6_smsp1");
    const std::uint64_t t1 = at(r1, 0, 0);
    check(t1 >= 60 && t1 <= 70, "cp6 smsp1 hold", 64, t1);
    auto r2 = J("cp6_smsp2");
    const std::uint64_t t2 = at(r2, 0, 0);
    check(t2 >= 60 && t2 <= 70, "cp6 smsp2 hold", 64, t2);
    auto r3 = J("cp6_smsp3");
    const std::uint64_t t3 = at(r3, 0, 0);
    check(t3 >= 90 && t3 <= 105, "cp6 smsp3 hold", 96, t3);
    auto r4 = J("cp6_smsp4");
    const std::uint64_t t4 = at(r4, 0, 0);
    check(t4 >= 120, "cp6 smsp4 hold", 129, t4);
    // w1..w3 done before w0's fill: n = 1 resident SMSP -> hold 64
    // (a mask that never cleared would give n = 4 -> 129)
    auto r5 = J("cp6_smsp_retire");
    const std::uint64_t t5 = at(r5, 0, 24);
    check(t5 >= 80 && t5 <= 96, "cp6 retire hold", 88, t5);
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
    // G1: the wait issues at once; the phase release lands on the
    // FILL behind it (completion + release, no result latency here)
    check(at(r, 0, 3) == at(r, 0, 2) + 1, "b2 wait issues at once",
          at(r, 0, 2) + 1, at(r, 0, 3));
    check(at(r, 0, 4) == 23, "b2 post", 23, at(r, 0, 4));
  }
  {
    // TRANSCNT pairing: tx +32 at 0, copy issued at 1, arrives at 2
    // (commit) completes the 32 tx at copy-done + 20, the real arrive
    // at 3 drops pending to 0 -> phase completes when both are done
    auto r = J("b2b_cpasync_transcnt");
    check(at(r, 0, 4) == at(r, 0, 3) + 1, "b2b wait issues at once",
          at(r, 0, 3) + 1, at(r, 0, 4));
    // G1: the result lands on the FILL behind the wait — completion
    // (commit arrive + copy done) + release
    check(at(r, 0, 5) >= 20 && at(r, 0, 5) <= 26, "b2b post at result",
          23, at(r, 0, 5));
  }
  {
    auto r = J("b3_multicast_tma");
    // G1: the waits issue at once; the TMA completion lands on the
    // FILL behind each wait
    check(at(r, 0, 2) <= 3, "b3 w0 wait issues at once", 3, at(r, 0, 2));
    check(at(r, 0, 3) == 52, "b3 w0 post at result", 52, at(r, 0, 3));
    check(at(r, 1, 1) <= 2, "b3 w1 wait issues at once", 2, at(r, 1, 1));
    check(at(r, 1, 2) == 52, "b3 w1 post at result", 52, at(r, 1, 2));
  }
  expect_engine_error(dir + "/b3b_no_mbob.job",
                      "missing_fact:mbarrier_object_table", "b3b no mbob");
  {
    auto r = J("b3c_late_arrive");
    check(at(r, 1, 61) == 61, "b3c w1 wait", 61, at(r, 1, 61));
  }
  {
    auto r = J("b4_service_start");
    // G1: the waits all issue at once; the service-start spacing
    // lands on the FILL behind each wait (inst 3)
    const std::uint64_t got[3] = {at(r, 0, 3), at(r, 1, 3), at(r, 2, 3)};
    const bool ok = (got[0] == 52 && got[1] == 62 && got[2] == 72) ||
        (got[0] == 52 && got[1] == 72 && got[2] == 62) ||
        (got[0] == 62 && got[1] == 52 && got[2] == 72) ||
        (got[0] == 62 && got[1] == 72 && got[2] == 52) ||
        (got[0] == 72 && got[1] == 52 && got[2] == 62) ||
        (got[0] == 72 && got[1] == 62 && got[2] == 52);
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
    check(at(r, 1, 5) == at(r, 1, 4) + 1, "b6l winner use (R1)",
          at(r, 1, 4) + 1, at(r, 1, 5));
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
    check(at(r, 1, 3) == at(r, 1, 2) + 1, "b6n use gap (R1)",
          at(r, 1, 2) + 1, at(r, 1, 3));
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
    check(last_completion(r) == 655, "c3 envelope linear", 655,
          last_completion(r));
  }
  {
    auto r = J("c3_blackwell");
    check(last_completion(r) == 60, "c3 envelope", 60,
          last_completion(r));
  }
  {
    auto r = J("c3_dual_sm");
    check(last_completion(r) == 40, "c3 dual_sm", 40,
          last_completion(r));
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
    check(last_completion(r) >= 256 && last_completion(r) <= 272,
          "a15 cycles", 264, last_completion(r));
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
    check(last_completion(r) >= 120 && last_completion(r) <= 140,
          "a30 cycles", 128, last_completion(r));
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
    check(last_completion(r) >= 101 && last_completion(r) <= 103,
          "a31 cycles", 102, last_completion(r));
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
    check(last_completion(r) >= 197 && last_completion(r) <= 199,
          "a31b cycles", 198, last_completion(r));
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
    // G1: the waits all issue at once; the service interval lands on
    // the FILL behind each wait (inst 3)
    for (unsigned k = 0; k < 3; ++k)
      check(at(ctrl4, k + 1, 3) - at(ctrl4, k, 3) == 10,
            "a32e service gap", 10,
            at(ctrl4, k + 1, 3) - at(ctrl4, k, 3));
    auto c = J("a32c_tma_cut_service");
    check(c.stats.cycles >= 829 && c.stats.cycles <= 831,
          "a32c cycles", 830, c.stats.cycles);
    for (unsigned k = 0; k < 3; ++k)
      check(at(c, k + 1, 3) - at(c, k, 3) >= 258 &&
                at(c, k + 1, 3) - at(c, k, 3) <= 260,
            "a32c service gap", 259,
            at(c, k + 1, 3) - at(c, k, 3));
    auto d = J("a32d_tma_cut_service32");
    check(d.stats.cycles >= 247 && d.stats.cycles <= 249,
          "a32d cycles", 248, d.stats.cycles);
    for (unsigned k = 0; k < 3; ++k)
      check(at(d, k + 1, 3) - at(d, k, 3) >= 64 &&
                at(d, k + 1, 3) - at(d, k, 3) <= 66,
            "a32d service gap", 65,
            at(d, k + 1, 3) - at(d, k, 3));
  }
  {
    // template-level shared acceptance rows (kind 6) serialize per SM:
    // 4 LDS x 40 cycles -> >= 160
    auto r = J("a18_shared_rows");
    check(last_completion(r) >= 160 && last_completion(r) <= 200,
          "a18 cycles", 165, last_completion(r));
  }
  {
    // class-bound LD template (generic pointer) whose MACCs carry
    // shared space (flags 0x1): kind-6 shared acceptance rows apply
    // exactly like a18's explicit LDS -> >= 160
    auto r = J("a18b_shared_rows_generic");
    check(last_completion(r) >= 160 && last_completion(r) <= 200,
          "a18b cycles", 165, last_completion(r));
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
    // G1: the retried check issues at once; its result (release) holds
    // the FILL behind it
    check(at(r, 0, 3) >= at(r, 1, 50) + 11, "a18 release",
          at(r, 1, 50) + 11, at(r, 0, 3));
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
    check(at(r, 1, 3) >= t0 + 400, "c8b w1 waits for fill",
          t0 + 400, at(r, 1, 3));
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
    check(at(r, 1, 3) >= at(r, 1, 0) + 400,
          "c8c w1 bank-3 miss latency kept",
          at(r, 1, 0) + 400, at(r, 1, 3));
  }

  {
    // field 6 (vector=64) + field 7 (uniform=96): w0 uniform bank-0
    // accesses return ~96, w1 vector bank-0 ~64, w2 bank-3 misses.
    auto r = J("c8d_bank0_uniform");
    check(tele4(r, 13) == 4, "c8d const hits", 4, tele4(r, 13));
    check(tele4(r, 14) == 1, "c8d const misses", 1, tele4(r, 14));
    check(tele4(r, 12) == 1, "c8d const same_line", 1, tele4(r, 12));
    check(at(r, 0, 3) >= at(r, 0, 0) + 90,
          "c8d w0 uniform bank-0 ~96",
          at(r, 0, 0) + 90, at(r, 0, 3));
    check(at(r, 1, 3) >= at(r, 1, 0) + 60 &&
              at(r, 1, 3) < at(r, 0, 0) + 90,
          "c8d w1 vector bank-0 ~64",
          at(r, 1, 0) + 64, at(r, 1, 3));
    check(at(r, 2, 3) >= at(r, 2, 0) + 400,
          "c8d w2 bank-3 miss latency kept",
          at(r, 2, 0) + 400, at(r, 2, 3));
  }

  {
    // uniform cold-hold (kernel_start v2, schema-27 fields 8=119 /
    // 9=463) + resident hold (kernel_start v3, field 10 = 0): a
    // uniform LDC on a cold or still-filling line holds at issue
    // until fill-ready; its result is ready at issue.  A uniform
    // re-read of a resident line is charged the resident hold (0:
    // result at issue) instead of hit / same-line latency.
    //  (i) w0 and w1 read the same cold bank-0 line: both issue at
    //      the fill-ready cycle and their DEPs pay no hit latency.
    //  (ii) w0 then reads a cold bank-3 line: the holds serialize
    //      (463 more cycles).
    //  (iii) w0 re-reads the resident line: ready at issue; the
    //      trailing vector LDC on the same line keeps same-line 37.
    auto r = J("c8e_const_hold");
    check(tele4(r, 14) == 3, "c8e const misses", 3, tele4(r, 14));
    check(tele4(r, 13) == 2, "c8e const hits", 2, tele4(r, 13));
    check(tele4(r, 12) == 2, "c8e const same_line", 2,
          tele4(r, 12));
    // (i) both warps issue within a cycle of the shared fill-ready
    const std::uint64_t d01 = at(r, 0, 0) > at(r, 1, 0)
        ? at(r, 0, 0) - at(r, 1, 0) : at(r, 1, 0) - at(r, 0, 0);
    check(d01 <= 1, "c8e both warps at fill-ready", 1, d01);
    check(at(r, 1, 2) <= at(r, 1, 0) + 5,
          "c8e held load result at issue",
          at(r, 1, 0) + 5, at(r, 1, 2));
    // (ii) the cold bank-3 line holds ~463 more cycles serially
    check(at(r, 0, 1) >= at(r, 0, 0) + 460,
          "c8e serial cold holds add",
          at(r, 0, 0) + 460, at(r, 0, 1));
    // (iii) resident line: no hold, result at issue (field 10 = 0)
    check(at(r, 0, 2) <= at(r, 0, 1) + 5,
          "c8e resident line not held",
          at(r, 0, 1) + 5, at(r, 0, 2));
    check(at(r, 0, 3) <= at(r, 0, 2) + 5,
          "c8e second re-read not held",
          at(r, 0, 2) + 5, at(r, 0, 3));
    check(at(r, 0, 5) <= at(r, 0, 3) + 5,
          "c8e uniform re-read ready at issue",
          at(r, 0, 3) + 5, at(r, 0, 5));
    // vector LDC on the same resident line keeps same-line 37
    check(at(r, 0, 8) >= at(r, 0, 6) + 30,
          "c8e vector re-read latency kept",
          at(r, 0, 6) + 30, at(r, 0, 8));
    // w2 cold bank-3 line: ~463
    check(at(r, 2, 2) <= at(r, 2, 0) + 5,
          "c8e w2 held load result at issue",
          at(r, 2, 0) + 5, at(r, 2, 2));
  }

  {
    // field 10 dropped, fields 8/9 kept (kernel_start v2 behaviour):
    // cold lines are still held at issue, uniform re-reads of a
    // resident line are charged hit (107) / same-line (37) again.
    auto r = J("c8g_const_hold_noresident");
    check(tele4(r, 14) == 3, "c8g const misses", 3, tele4(r, 14));
    check(tele4(r, 12) == 2, "c8g const same_line", 2,
          tele4(r, 12));
    const std::uint64_t g01 = at(r, 0, 0) > at(r, 1, 0)
        ? at(r, 0, 0) - at(r, 1, 0) : at(r, 1, 0) - at(r, 0, 0);
    check(g01 <= 1, "c8g both warps at fill-ready", 1, g01);
    check(at(r, 0, 1) >= at(r, 0, 0) + 460,
          "c8g serial cold holds add",
          at(r, 0, 0) + 460, at(r, 0, 1));
    check(at(r, 0, 5) >= at(r, 0, 3) + 30,
          "c8g uniform re-read same-line latency",
          at(r, 0, 3) + 30, at(r, 0, 5));
    check(at(r, 0, 8) >= at(r, 0, 6) + 30,
          "c8g vector re-read latency kept",
          at(r, 0, 6) + 30, at(r, 0, 8));
  }

  {
    // rule off without fields 8/9/10: bit-identical to today's
    // engine -- cold accesses are misses that issue at once and only
    // make the consumer wait for the fill, no issue hold.
    auto r = J("c8f_const_hold_off");
    // w0/w1 race on the shared cold line: the misser and the
    // in-flight reader are fixed by evaluation order, so only the
    // miss+hit total is stable
    check(tele4(r, 14) + tele4(r, 13) == 5, "c8f miss+hit", 5,
          tele4(r, 14) + tele4(r, 13));
    check(tele4(r, 12) == 2, "c8f const same_line", 2,
          tele4(r, 12));
    // the second warp's LDC issues at once (no hold), unlike c8e
    check(at(r, 1, 0) <= at(r, 0, 0) + 30,
          "c8f no hold at issue",
          at(r, 0, 0) + 30, at(r, 1, 0));
    // but its consumer still waits for the fill (~406)
    check(at(r, 1, 2) >= at(r, 0, 0) + 390,
          "c8f consumer waits for fill",
          at(r, 0, 0) + 390, at(r, 1, 2));
  }

  {
    // kernel_start v4 (field 11, fields 6/8/9/10 on): a vector LDC
    // on a bank-0 line not resident in the SM's constant cache
    // starts a fill whose cold latency the readers wait for;
    // resident and in-flight lines keep bank0_access_cycles.  The
    // resident set is shared with the uniform path.
    auto r = J("c8h_vbank0_cold");
    // w1 races w0's fill: which of them is the misser is fixed by
    // evaluation order, so only the miss+hit total is stable; the
    // shared fill-ready cycle is checked by timing below.  w0's
    // re-read and w2's vector read must hit deterministically
    check(tele4(r, 14) + tele4(r, 13) == 5, "c8h const miss+hit", 5,
          tele4(r, 14) + tele4(r, 13));
    check(tele4(r, 13) >= 2, "c8h const hits", 2, tele4(r, 13));
    check(tele4(r, 12) == 0, "c8h const same_line", 0,
          tele4(r, 12));
    // (i) the cold vector load's reader waits ~153 for the fill
    check(at(r, 0, 2) >= at(r, 0, 0) + 140,
          "c8h cold vector fill latency",
          at(r, 0, 0) + 140, at(r, 0, 2));
    check(at(r, 0, 2) <= at(r, 0, 0) + 180,
          "c8h cold vector fill latency upper",
          at(r, 0, 0) + 180, at(r, 0, 2));
    // (iii) w1 reads the same line while the fill is in flight: its
    // reader is ready at the same cycle as w0's
    const std::uint64_t h01 = at(r, 0, 2) > at(r, 1, 2)
        ? at(r, 0, 2) - at(r, 1, 2) : at(r, 1, 2) - at(r, 0, 2);
    check(h01 <= 5, "c8h in-flight reader same ready", 5, h01);
    // (ii) same warp re-reads the line: ~30
    check(at(r, 0, 5) >= at(r, 0, 3) + 25,
          "c8h resident re-read latency",
          at(r, 0, 3) + 25, at(r, 0, 5));
    check(at(r, 0, 5) <= at(r, 0, 3) + 60,
          "c8h resident re-read latency upper",
          at(r, 0, 3) + 60, at(r, 0, 5));
    // (iv) the line a uniform load filled is resident for the
    // vector path too: ~30
    check(at(r, 2, 5) >= at(r, 2, 3) + 25,
          "c8h vector after uniform fill",
          at(r, 2, 3) + 25, at(r, 2, 5));
    check(at(r, 2, 5) <= at(r, 2, 3) + 60,
          "c8h vector after uniform fill upper",
          at(r, 2, 3) + 60, at(r, 2, 5));
  }

  {
    // field 11 dropped (6/8/9/10 kept): vector bank-0 reads keep the
    // flat 30 and do not touch the resident set, so the uniform read
    // of the same line right after is still cold-held (~119)
    auto r = J("c8i_vbank0_cold_off");
    check(tele4(r, 14) == 1, "c8i const misses", 1, tele4(r, 14));
    check(tele4(r, 13) == 1, "c8i const hits", 1, tele4(r, 13));
    check(tele4(r, 12) == 0, "c8i const same_line", 0,
          tele4(r, 12));
    check(at(r, 0, 2) >= at(r, 0, 0) + 25,
          "c8i flat vector latency",
          at(r, 0, 0) + 25, at(r, 0, 2));
    check(at(r, 0, 2) <= at(r, 0, 0) + 60,
          "c8i flat vector latency upper",
          at(r, 0, 0) + 60, at(r, 0, 2));
    check(at(r, 0, 3) >= at(r, 0, 2) + 100,
          "c8i line not made resident",
          at(r, 0, 2) + 100, at(r, 0, 3));
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
    // served consumer after a 3-read producer: issue_rules v1 R1
    // dropped the register floor, so a pair costs the control floors
    // (1 + 3 = 4); W=2 interleaves on the collector with bounded
    // extra delay
    auto r1 = J("op5_pair_w1");
    const std::uint64_t s1 = at(r1, 0, 14) - at(r1, 0, 0);
    check(s1 >= 7 * 4 - 4 && s1 <= 7 * 4 + 4, "op5 w1 pair ~4", 28,
          s1);
    auto r2 = J("op5_pair_w2");
    for (int k = 0; k < 2; ++k) {
      const std::uint64_t s2 = at(r2, k, 14) - at(r2, k, 0);
      check(s2 >= 7 * 4 - 4 && s2 <= 7 * 4 + 24, "op5 w2 pair ~4+",
            28, s2);
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
    // branch_basic v2 R2: BSYNC result 13 -> next issue at +13
    auto r = J("r2_bsync_floor");
    check(at(r, 0, 1) == at(r, 0, 0) + 13, "bsync floor +13",
          at(r, 0, 0) + 13, at(r, 0, 1));
    // stall 5 does not add to the floor
    auto r2 = J("r2_bsync_floor_stall");
    check(at(r2, 0, 1) == at(r2, 0, 0) + 13, "bsync floor stall +13",
          at(r2, 0, 0) + 13, at(r2, 0, 1));
  }
  {
    // branch_basic v2 R1: BSSY is not in the floor family -> +1
    auto r = J("r2_bssy_no_floor");
    check(at(r, 0, 1) == at(r, 0, 0) + 1, "bssy no floor +1",
          at(r, 0, 0) + 1, at(r, 0, 1));
  }
  {
    // leaf 0 -> L1 port off -> acceptance at issue -> fill = last+1
    auto r = J("m_l1_port_off");
    check(at(r, 0, 10) == at(r, 0, 9) + 1, "l1port_off last+1",
          at(r, 0, 9) + 1, at(r, 0, 10));
  }

  // ---- stall ledger -------------------------------------------------
  // --stall-ledger is statistics-only: the same fixture run with the
  // option must produce identical cycles / issued / completed /
  // telemetry / wait buckets / journal as the plain run.  The file is
  // the cyclescope.stall-ledger.v1 schema (residency + transition rows);
  // interval accounting is the same flush_wait machinery as the wait
  // buckets, so the ledger total equals the bucket total.
  auto run_ledger = [&](const char* name, const std::string& tsv,
                        std::vector<cyclescope::WarpState>* warps,
                        cyclescope::ResultStats* st) {
    cyclescope::Job job =
        cyclescope::read_container_file(dir + "/" + name + ".job");
    cyclescope::MechanismRuntime::Options ro;
    ro.reference_sm_clock_khz = job.launch.reference_sm_clock_khz;
    cyclescope::MechanismRuntime rt(job, job.launch.device, ro);
    cyclescope::Simulator::Options so;
    so.stall_ledger = tsv;
    so.journal = true;
    cyclescope::Simulator sim(job, rt, so);
    *st = sim.run();
    for (std::uint32_t i = 0; i < sim.warp_count(); ++i)
      warps->push_back(sim.warp(i));
  };
  auto same_stats = [](const cyclescope::ResultStats& a,
                       const cyclescope::ResultStats& b) {
    if (a.cycles != b.cycles || a.issued != b.issued ||
        a.completed != b.completed ||
        a.tele.size() != b.tele.size() ||
        a.wait.size() != b.wait.size() ||
        a.rwait.size() != b.rwait.size() ||
        a.journal.size() != b.journal.size())
      return false;
    for (std::size_t i = 0; i < a.tele.size(); ++i)
      if (a.tele[i].group != b.tele[i].group ||
          a.tele[i].counter != b.tele[i].counter ||
          a.tele[i].value != b.tele[i].value)
        return false;
    for (std::size_t i = 0; i < a.wait.size(); ++i)
      if (a.wait[i] != b.wait[i]) return false;
    for (std::size_t i = 0; i < a.rwait.size(); ++i)
      if (a.rwait[i] != b.rwait[i]) return false;
    for (std::size_t i = 0; i < a.journal.size(); ++i) {
      const auto& x = a.journal[i];
      const auto& y = b.journal[i];
      if (x.warp != y.warp || x.instance != y.instance ||
          x.issue_cycle != y.issue_cycle ||
          x.wait_bucket != y.wait_bucket ||
          x.wait_cycles != y.wait_cycles ||
          x.object != y.object)
        return false;
    }
    return true;
  };
  struct Ledger {
    std::map<std::pair<std::uint64_t, std::string>, std::uint64_t> res;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t>
        trans;
    std::set<std::string> reasons;
    std::uint64_t warps = 0, cycles = 0, res_total = 0;
    // residency rows with reason "issued" whose cycle count is not
    // exactly their issue-event count
    std::uint64_t issued_bad = 0;
  };
  auto read_ledger = [](const std::string& path) {
    Ledger l;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("#warps\t", 0) == 0) {
        l.warps = std::strtoull(line.c_str() + 7, nullptr, 10);
        continue;
      }
      if (line.rfind("#cycles\t", 0) == 0) {
        l.cycles = std::strtoull(line.c_str() + 8, nullptr, 10);
        continue;
      }
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      std::string kind;
      ss >> kind;
      if (kind == "residency") {
        std::string pc, op, reason;
        std::uint64_t cycles = 0, events = 0;
        ss >> pc >> op >> reason >> cycles >> events;
        const std::uint64_t a = std::strtoull(pc.c_str(), nullptr, 16);
        l.res[{a, reason}] += cycles;
        l.res_total += cycles;
        l.reasons.insert(reason);
        if (reason == "issued" && cycles != events) ++l.issued_bad;
      } else if (kind == "transition") {
        std::string from, to;
        std::uint64_t n = 0;
        ss >> from >> to >> n;
        l.trans[{std::strtoull(from.c_str(), nullptr, 16),
                 std::strtoull(to.c_str(), nullptr, 16)}] += n;
      }
    }
    return l;
  };
  // (a) statistics-only: identical result with and without the option
  for (const char* fx : {"a25_control_stall", "m_cpasync_anchor"}) {
    Run plain = J(fx);
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats with;
    const std::string tsv = dir + "/" + fx + ".stall.tsv";
    run_ledger(fx, tsv, &warps, &with);
    check(same_stats(plain.stats, with), "stall ledger neutral", 1, 1);
    const Ledger l = read_ledger(tsv);
    check(l.cycles == with.cycles, "stall ledger #cycles",
          with.cycles, l.cycles);
    check(l.warps == warps.size(), "stall ledger #warps",
          warps.size(), l.warps);
    // conservation: per warp the non-dispatch accrual is exactly
    // (last issue - first resident cycle + 1) + exit-drain cycles
    // (first resident cycle = first cycle charged a non-dispatch
    // reason; the issue cycle itself is charged "issued"); file
    // level the rows sum to the same total
    std::uint64_t per_warp = 0;
    bool cons = true;
    for (const auto& w : warps) {
      const std::uint64_t exp =
          w.stall_has_issue
              ? w.stall_last_issue - w.stall_first_resident + 1 +
                    w.stall_drain_cycles
              : 0;
      if (w.stall_resident_cycles != exp) cons = false;
      per_warp += w.stall_dispatch_cycles + w.stall_resident_cycles;
    }
    check(cons, "stall ledger conservation", 1, cons ? 1 : 0);
    check(l.res_total == per_warp, "stall ledger file total",
          per_warp, l.res_total);
    // the ledger covers every accrued cycle; the wait buckets skip
    // the issue cycle itself (wait_mark = issue+1), so the ledger
    // total is the bucket total plus the issue count
    std::uint64_t buckets = 0;
    for (const auto& [b, c] : with.wait) buckets += c;
    check(l.res_total == buckets + with.issued,
          "stall ledger vs buckets", buckets + with.issued,
          l.res_total);
    // only the issue cycle itself is "issued": every issued row's
    // cycle count equals its issue-event count
    check(l.issued_bad == 0, "stall ledger issued==events", 0,
          l.issued_bad);
    bool no_other = true;
    for (const auto& r : l.reasons)
      if (r.rfind("other:", 0) == 0) no_other = false;
    check(no_other, "stall ledger no other:*", 1, no_other ? 1 : 0);
  }
  // (b)+(d) conservation and reason names on the mixed-wait fixture:
  // register wait behind a load, a unit wait, a nanosleep, an
  // mbarrier wait
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/sl_waits.stall.tsv";
    run_ledger("sl_waits", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    bool cons = true;
    for (const auto& w : warps) {
      const std::uint64_t exp =
          w.stall_has_issue
              ? w.stall_last_issue - w.stall_first_resident + 1 +
                    w.stall_drain_cycles
              : 0;
      if (w.stall_resident_cycles != exp) cons = false;
    }
    check(cons, "sl_waits conservation", 1, cons ? 1 : 0);
    check(l.issued_bad == 0, "sl_waits issued==events", 0,
          l.issued_bad);
    auto has = [&](const char* prefix) {
      for (const auto& r : l.reasons)
        if (r.find(prefix) != std::string::npos) return true;
      return false;
    };
    check(has("barrier:LDG"), "sl_waits barrier:LDG", 1,
          has("barrier:LDG") ? 1 : 0);
    check(!has("reg:"), "sl_waits no reg:", 0,
          has("reg:") ? 1 : 0);
    check(has("unit:xu_smsp"), "sl_waits unit", 1,
          has("unit:xu_smsp") ? 1 : 0);
    check(has("suspended:sleep"), "sl_waits sleep", 1,
          has("suspended:sleep") ? 1 : 0);
    check(has("mbarrier"), "sl_waits mbarrier", 1,
          has("mbarrier") ? 1 : 0);
    // a nanosleep suspension is keyed at the instruction after
    // NANOSLEEP (0x1070), never at the NANOSLEEP pc (0x1060)
    const auto ns = l.res.find({0x1080, "suspended:sleep"});
    check(ns != l.res.end() && ns->second == 99,
          "sl_waits sleep at next pc", 99,
          ns == l.res.end() ? 0 : ns->second);
    check(l.res.count({0x1070, "suspended:sleep"}) == 0,
          "sl_waits sleep not at own pc", 0,
          l.res.count({0x1070, "suspended:sleep"}));
  }
  // constant_paths v1: the held uniform load accrues constant_hold
  // at its own address while it fills
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/cp1_uhold.stall.tsv";
    run_ledger("cp1_uhold", tsv, &warps, &st);
    const Ledger l2 = read_ledger(tsv);
    check(l2.res.count({0x1000, "constant_hold"}) > 0,
          "cp1 constant_hold at load", 1,
          l2.res.count({0x1000, "constant_hold"}));
    auto has2 = [&](const char* prefix) {
      for (const auto& r : l2.reasons)
        if (r.find(prefix) != std::string::npos) return true;
      return false;
    };
    check(!has2("reg:"), "cp1 no reg:", 0, has2("reg:") ? 1 : 0);
  }
  // (c) transitions on the loop fixture: a 3-instruction body run
  // four times per warp gives back edge count 3 per warp (2 warps)
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/sl_loop.stall.tsv";
    run_ledger("sl_loop", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    auto tr = [&](std::uint64_t a, std::uint64_t b) -> std::uint64_t {
      const auto it = l.trans.find({a, b});
      return it == l.trans.end() ? 0 : it->second;
    };
    check(tr(0x1020, 0x1000) == 6, "sl_loop back edge", 6,
          tr(0x1020, 0x1000));
    check(tr(0x1000, 0x1010) == 8, "sl_loop fallthrough 1", 8,
          tr(0x1000, 0x1010));
    check(tr(0x1010, 0x1020) == 8, "sl_loop fallthrough 2", 8,
          tr(0x1010, 0x1020));
    check(l.issued_bad == 0, "sl_loop issued==events", 0,
          l.issued_bad);
  }
  // (e) two always-ready warps on one SMSP: while one issues the
  // other accrues issue_slot at its next instruction's address, and
  // per-warp conservation still holds
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/sl_issue_slot.stall.tsv";
    run_ledger("sl_issue_slot", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    check(l.reasons.count("issue_slot") == 1, "sl_issue_slot reason",
          1, l.reasons.count("issue_slot") ? 1 : 0);
    std::uint64_t slot_cyc = 0, issued_cyc = 0;
    for (const auto& [k, c] : l.res) {
      if (k.second == "issue_slot") slot_cyc += c;
      if (k.second == "issued") issued_cyc += c;
    }
    check(slot_cyc > 0, "sl_issue_slot cycles", 1, slot_cyc ? 1 : 0);
    bool cons = true;
    for (const auto& w : warps) {
      const std::uint64_t exp =
          w.stall_has_issue
              ? w.stall_last_issue - w.stall_first_resident + 1 +
                    w.stall_drain_cycles
              : 0;
      if (w.stall_resident_cycles != exp) cons = false;
    }
    check(cons, "sl_issue_slot conservation", 1, cons ? 1 : 0);
    check(l.issued_bad == 0, "sl_issue_slot issued==events", 0,
          l.issued_bad);
  }
  // (f) control-word stall count 4: the span after the first issue is
  // control_floor at the following instruction's address for 3
  // cycles
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/sl_control_floor.stall.tsv";
    run_ledger("sl_control_floor", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    const auto it = l.res.find({0x1010, "control_floor"});
    const std::uint64_t got = it == l.res.end() ? 0 : it->second;
    check(got == 3, "sl_control_floor span", 3, got);
    check(l.issued_bad == 0, "sl_control_floor issued==events", 0,
          l.issued_bad);
  }
  // (h) issue_rules v2 Y1: the cycle skipped by yield+stall1
  // accrues "yield" (one cycle at the next instruction), not
  // control_floor
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/ir4_yield_stall.stall.tsv";
    run_ledger("ir4_yield_stall", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    std::uint64_t yc = 0;
    for (const auto& [k, c] : l.res)
      if (k.second == "yield") yc += c;
    check(yc == 1, "ir4 yield span", 1, yc);
    // yield+stall4 stays a plain control_floor (3 cycles)
    std::uint64_t cf = 0;
    for (const auto& [k, c] : l.res)
      if (k.second == "control_floor") cf += c;
    check(cf == 3, "ir4 stall4 control_floor", 3, cf);
    check(l.issued_bad == 0, "ir4 issued==events", 0, l.issued_bad);
  }
  // control:* labels: the part of a binding floor beyond the stall
  // field carries the mechanism's name; the stall-field part stays
  // control_floor at the same pc (totals unchanged)
  {
    auto reason_at = [](const Ledger& l, std::uint64_t pc,
                        const char* r) {
      const auto it = l.res.find({pc, r});
      return it == l.res.end() ? 0 : it->second;
    };
    auto ledger_of = [&](const char* fx) {
      std::vector<cyclescope::WarpState> warps;
      cyclescope::ResultStats st;
      const std::string tsv = std::string(dir) + "/" + fx + ".stall.tsv";
      run_ledger(fx, tsv, &warps, &st);
      return read_ledger(tsv);
    };
    // taken BRA (envelope 10, stall 5): FILL at +10 -> 4 control_floor
    // + 5 control:branch at 0x1010
    {
      const Ledger l = ledger_of("sl_control_branch");
      check(reason_at(l, 0x1010, "control_floor") == 4,
            "sl_control_branch stall part", 4,
            reason_at(l, 0x1010, "control_floor"));
      check(reason_at(l, 0x1010, "control:branch") == 5,
            "sl_control_branch mech part", 5,
            reason_at(l, 0x1010, "control:branch"));
    }
    // YIELD (floor 50, stall 5): FILL at +50 -> 4 + 45
    {
      const Ledger l = ledger_of("sl_control_yield");
      check(reason_at(l, 0x1010, "control_floor") == 4,
            "sl_control_yield stall part", 4,
            reason_at(l, 0x1010, "control_floor"));
      check(reason_at(l, 0x1010, "control:yield") == 45,
            "sl_control_yield mech part", 45,
            reason_at(l, 0x1010, "control:yield"));
    }
    // BSYNC (floor 13, stall 5): FILL at +13 -> 4 + 8
    {
      const Ledger l = ledger_of("r2_bsync_floor_stall");
      std::uint64_t cf = 0, cb = 0;
      for (const auto& [k, c] : l.res) {
        if (k.second == "control_floor") cf += c;
        if (k.second == "control:bsync") cb += c;
      }
      check(cf == 4, "r2_bsync_floor_stall stall part", 4, cf);
      check(cb == 8, "r2_bsync_floor_stall mech part", 8, cb);
    }
    // stall 5 only: stays control_floor, no control:* reason
    {
      const Ledger l = ledger_of("sl_control_stall");
      check(reason_at(l, 0x1010, "control_floor") == 4,
            "sl_control_stall stall span", 4,
            reason_at(l, 0x1010, "control_floor"));
      bool none = true;
      for (const auto& r : l.reasons)
        if (r.rfind("control:", 0) == 0) none = false;
      check(none, "sl_control_stall no control:*", 1, none ? 1 : 0);
    }
  }
  // placement v1 (rules T, D, B, M, F): the ledger books the same
  // cycles on other pcs / under other names; every fixture runs with
  // and without --stall-ledger to identical statistics, and its cycle
  // count is the one the engine had before the change (old = the
  // build of commit 03e3ce08 on the same job)
  {
    auto reason_at = [](const Ledger& l, std::uint64_t pc,
                        const char* r) {
      const auto it = l.res.find({pc, r});
      return it == l.res.end() ? 0 : it->second;
    };
    auto sum_reason = [](const Ledger& l, const char* r) {
      std::uint64_t n = 0;
      for (const auto& [k, c] : l.res)
        if (k.second == r) n += c;
      return n;
    };
    auto placed = [&](const char* fx, std::uint64_t old_cycles) {
      std::vector<cyclescope::WarpState> warps;
      cyclescope::ResultStats st;
      const std::string tsv = std::string(dir) + "/" + fx + ".stall.tsv";
      run_ledger(fx, tsv, &warps, &st);
      const auto plain = J(fx);
      check(same_stats(plain.stats, st), fx, 1, 0);
      check(st.cycles == old_cycles, fx, old_cycles, st.cycles);
      return read_ledger(tsv);
    };
    {  // T: 4 + 3 + 2 of the nine cycles behind a taken BRA (old 10)
      const Ledger l = placed("pl_branch_pieces", 10);
      check(reason_at(l, 0x1010, "control:branch_fallthrough") == 4,
            "T fallthrough", 4,
            reason_at(l, 0x1010, "control:branch_fallthrough"));
      check(reason_at(l, 0x1040, "control:branch") == 3, "T resolve", 3,
            reason_at(l, 0x1040, "control:branch"));
      check(reason_at(l, 0x1040, "control:branch_refetch") == 2,
            "T refetch", 2,
            reason_at(l, 0x1040, "control:branch_refetch"));
      check(sum_reason(l, "control_floor") == 0, "T no control_floor", 0,
            sum_reason(l, "control_floor"));
    }
    {  // D: the group wait is booked behind the DEPBAR (old 301)
      const Ledger l = placed("pl_depbar_wait", 301);
      check(reason_at(l, 0x1030, "async_group_wait") == 298,
            "D behind the DEPBAR", 298,
            reason_at(l, 0x1030, "async_group_wait"));
      check(reason_at(l, 0x1020, "async_group_wait") == 0,
            "D not on the DEPBAR", 0,
            reason_at(l, 0x1020, "async_group_wait"));
    }
    {  // B: Dmin 5, stall fields 1; old cycles 10 / 8 / 8 / 11 / 8
      const Ledger a = placed("pl_defer_d4", 10);
      check(reason_at(a, 0x1030, "cta_barrier") == 6,
            "B d4: S2R at dist 4 passed, IMAD runs through, LDC stops",
            6, reason_at(a, 0x1030, "cta_barrier"));
      const Ledger b = placed("pl_defer_d5", 8);
      check(reason_at(b, 0x1010, "cta_barrier") == 6,
            "B d5: S2R stops", 6, reason_at(b, 0x1010, "cta_barrier"));
      const Ledger u = placed("pl_defer_umov", 8);
      check(reason_at(u, 0x1010, "cta_barrier") == 6,
            "B UMOV stops", 6, reason_at(u, 0x1010, "cta_barrier"));
      const Ledger p = placed("pl_defer_pass", 11);
      check(reason_at(p, 0x1040, "cta_barrier") == 6,
            "B IMAD / UIADD3 / FFMA run through", 6,
            reason_at(p, 0x1040, "cta_barrier"));
      const Ledger n = placed("pl_bar_plain", 8);
      check(reason_at(n, 0x1010, "cta_barrier") == 6,
            "B plain BAR unchanged", 6,
            reason_at(n, 0x1010, "cta_barrier"));
    }
    {  // M: names by operation class; the stall-field part stays
       // control_floor (old 62)
      const Ledger l = placed("pl_membar_names", 62);
      check(reason_at(l, 0x1010, "control_floor") == 4 &&
                reason_at(l, 0x1010, "control:membar:MEMBAR.SC.GPU") == 25,
            "M MEMBAR.SC.GPU", 25,
            reason_at(l, 0x1010, "control:membar:MEMBAR.SC.GPU"));
      check(reason_at(l, 0x1020, "control_floor") == 2 &&
                reason_at(l, 0x1020, "control:membar:MEMBAR.ALL.CTA") == 9,
            "M MEMBAR.ALL.CTA", 9,
            reason_at(l, 0x1020, "control:membar:MEMBAR.ALL.CTA"));
      check(reason_at(l, 0x1030, "control_floor") == 1 &&
                reason_at(l, 0x1030,
                          "control:fence:FENCE.VIEW.ASYNC.S") == 18,
            "M FENCE.VIEW.ASYNC.S", 18,
            reason_at(l, 0x1030, "control:fence:FENCE.VIEW.ASYNC.S"));
    }
    {  // M, drain floors (old 92; b24: 71 / 85 / 92)
      const Ledger l = placed("pl_fence_drain", 92);
      check(reason_at(l, 0x1030, "control:membar:MEMBAR.ALL.CTA") == 68,
            "M membar drain", 68,
            reason_at(l, 0x1030, "control:membar:MEMBAR.ALL.CTA"));
      check(reason_at(l, 0x1040,
                      "control:fence:FENCE.VIEW.ASYNC.S") == 13,
            "M proxy fence drain", 13,
            reason_at(l, 0x1040, "control:fence:FENCE.VIEW.ASYNC.S"));
      check(reason_at(l, 0x1050, "control:membar:MEMBAR.ALL.CTA") == 6,
            "M clean membar", 6,
            reason_at(l, 0x1050, "control:membar:MEMBAR.ALL.CTA"));
    }
    {  // F: the suspension is booked on the first waiter of PRE's
       // write barrier (0x1070), not on PRE (0x1060) (old 726)
      const Ledger l = placed("pl_fence_waiter", 726);
      check(reason_at(l, 0x1070, "suspended:descriptor_visibility") ==
                720,
            "F on the waiter", 720,
            reason_at(l, 0x1070, "suspended:descriptor_visibility"));
      check(reason_at(l, 0x1060, "suspended:descriptor_visibility") ==
                0,
            "F not on PRE", 0,
            reason_at(l, 0x1060, "suspended:descriptor_visibility"));
    }
  }
  // placement v3 rule F (extended), service-floor form: the part of the
  // fence's floor beyond its stall field is booked on the first waiter
  // of its write barrier (the BAR, 0x1030), two instances behind the
  // next one; the stall-field part stays control_floor on 0x1010.
  // Cycles as before the change: 23.
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv =
        std::string(dir) + "/pl_fence_service_waiter.stall.tsv";
    run_ledger("pl_fence_service_waiter", tsv, &warps, &st);
    const auto plain = J("pl_fence_service_waiter");
    check(same_stats(plain.stats, st), "F service neutral", 1, 0);
    check(st.cycles == 23, "F service cycles", 23, st.cycles);
    const Ledger l = read_ledger(tsv);
    auto at_pc = [&](std::uint64_t pc, const char* r) {
      const auto it = l.res.find({pc, r});
      return it == l.res.end() ? 0 : it->second;
    };
    check(at_pc(0x1030, "control:fence:FENCE.VIEW.ASYNC.S") == 18,
          "F service on the waiter", 18,
          at_pc(0x1030, "control:fence:FENCE.VIEW.ASYNC.S"));
    check(at_pc(0x1010, "control:fence:FENCE.VIEW.ASYNC.S") == 0,
          "F service not on the next instance", 0,
          at_pc(0x1010, "control:fence:FENCE.VIEW.ASYNC.S"));
    check(at_pc(0x1010, "control_floor") == 1, "F service stall part", 1,
          at_pc(0x1010, "control_floor"));
  }
  // placement v5 rule P'' (ledger): a wait held by the pending result
  // of a poll that carries the wait origin books the cycles earlier
  // than issue(poll) + origin_hold_cycles on the origin address
  // (0x1ff0), the rest where it is booked today.  The poll issues at
  // cycle 1 and its reader (on 0x10b0) is held from cycle 12 to 112;
  // statistics and cycles are those of the same fixture without the
  // flag (112)
  {
    const char* kReason = "barrier:SYNCS.PHASECHK";
    auto run_one = [&](const char* fx, Ledger* out) {
      std::vector<cyclescope::WarpState> warps;
      cyclescope::ResultStats st;
      const std::string tsv = std::string(dir) + "/" + fx + ".stall.tsv";
      run_ledger(fx, tsv, &warps, &st);
      check(same_stats(J(fx).stats, st), fx, 1, 0);
      check(st.cycles == 112, fx, 112, st.cycles);
      *out = read_ledger(tsv);
    };
    auto at_pc = [](const Ledger& l, std::uint64_t pc, const char* r) {
      const auto it = l.res.find({pc, r});
      return it == l.res.end() ? 0 : it->second;
    };
    Ledger all, cap, off;
    run_one("pl_origin_all", &all);
    run_one("pl_origin_cap", &cap);
    run_one("pl_origin_off", &off);
    check(at_pc(off, 0x10b0, kReason) == 100, "P'' off: today's booking",
          100, at_pc(off, 0x10b0, kReason));
    check(at_pc(all, 0x1ff0, kReason) == 100 &&
              at_pc(all, 0x10b0, kReason) == 0,
          "P'' wait shorter than T: all on the origin", 100,
          at_pc(all, 0x1ff0, kReason));
    check(at_pc(cap, 0x1ff0, kReason) == 19 &&
              at_pc(cap, 0x10b0, kReason) == 81,
          "P'' T = 30 from the issue at 1: cycles before 31 on the origin "
          "(held from 12: 19), the rest as today", 19,
          at_pc(cap, 0x1ff0, kReason));
  }
  // (g) a finite retry that fell due while the warp was never
  // reached: [1, 8) stays under unit:xu_smsp, the rest goes to
  // issue_slot at the same address
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/sl_due.stall.tsv";
    run_ledger("sl_due", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    const auto iu = l.res.find({0x1010, "unit:xu_smsp"});
    check(iu != l.res.end() && iu->second == 7,
          "sl_due unit span", 7, iu == l.res.end() ? 0 : iu->second);
    const auto sl = l.res.find({0x1010, "issue_slot"});
    check(sl != l.res.end() && sl->second == 13,
          "sl_due issue_slot span", 13,
          sl == l.res.end() ? 0 : sl->second);
    check(l.issued_bad == 0, "sl_due issued==events", 0,
          l.issued_bad);
  }
  // (h) cta_barrier waits are keyed at the instruction after the
  // barrier (0x1010), not at the barrier instruction (0x1000)
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/sl_cta_barrier.stall.tsv";
    run_ledger("sl_cta_barrier", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    const auto cb = l.res.find({0x1010, "cta_barrier"});
    check(cb != l.res.end() && cb->second > 0,
          "sl_cta_barrier at next pc", 1,
          cb != l.res.end() && cb->second ? 1 : 0);
    check(l.res.count({0x1000, "cta_barrier"}) == 0,
          "sl_cta_barrier not at own pc", 0,
          l.res.count({0x1000, "cta_barrier"}));
    check(l.issued_bad == 0, "sl_cta_barrier issued==events", 0,
          l.issued_bad);
  }
  // (i) cluster_barrier waits are keyed at the wait instruction's own
  // address (0x1060), not at the following instruction (0x1070)
  {
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/sl_cluster_barrier.stall.tsv";
    run_ledger("sl_cluster_barrier", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    const auto cb = l.res.find({0x1060, "cluster_barrier"});
    check(cb != l.res.end() && cb->second > 0,
          "sl_cluster_barrier at own pc", 1,
          cb != l.res.end() && cb->second ? 1 : 0);
    check(l.res.count({0x1070, "cluster_barrier"}) == 0,
          "sl_cluster_barrier not at next pc", 0,
          l.res.count({0x1070, "cluster_barrier"}));
    check(l.issued_bad == 0, "sl_cluster_barrier issued==events", 0,
          l.issued_bad);
  }
  {
    // uniform cold-hold keying: the hold cycles are charged to
    // "constant_hold" at the held load's own pc (0x1000 / 0x1010
    // for w0's two cold lines, 0x1070 w1, 0x1090 w2)
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/c8e_const_hold.stall.tsv";
    run_ledger("c8e_const_hold", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    for (const std::uint64_t pc : {0x1000ull, 0x1010ull, 0x1090ull,
                                   0x10c0ull}) {
      const auto it = l.res.find({pc, "constant_hold"});
      check(it != l.res.end() && it->second > 0,
            "c8e constant_hold at load pc", 1,
            it != l.res.end() && it->second ? 1 : 0);
    }
    check(l.res.count({0x1020, "constant_hold"}) == 0 &&
              l.res.count({0x1030, "constant_hold"}) == 0,
          "c8e resident line no hold", 0,
          l.res.count({0x1020, "constant_hold"}));
    check(l.issued_bad == 0, "c8e issued==events", 0,
          l.issued_bad);
  }
  // ---- kernel_boundary K2/K3/K4 ----
  {
    // K2: cycles is the busiest SM's active-cycle union, not the
    // last completion; the ~15-cycle idle gap between the first
    // wave and the refilled CTAs is excluded (last completion
    // would be ~40)
    const Run r = J("kb1_active_cycles");
    check(r.stats.cycles >= 20 && r.stats.cycles <= 30,
          "kb1 active-union cycles", 26, r.stats.cycles);
  }
  {
    // K3: first launch line at entry + 200 (entries 0 / 120 / 240
    // from first_wave_interval); the entrant at 120 shares the
    // in-flight arrival; the entrant at 240 pays only its own
    // demand (~entry + 10); the next 128 B line is entry + 210
    const Run r = J("kb2_first_line");
    check(at(r, 0, 0) == 200, "kb2 first line entry+first", 200,
          at(r, 0, 0));
    check(at(r, 1, 0) == 200, "kb2 in-flight entrant same arrival",
          200, at(r, 1, 0));
    check(at(r, 2, 0) >= 240 && at(r, 2, 0) <= 290,
          "kb2 post-arrival entrant own demand", 250, at(r, 2, 0));
    // chain line 20 lands at 200 + 20 x 10 = 400; the entrant's
    // demand at 240 uses its own L1 slot (250), not the chain's
    check(at(r, 2, 0) == 250,
          "kb2 in-window line out-races the chain", 250,
          at(r, 2, 0));
    check(at(r, 0, 8) == 210,
          "kb2 second line entry+first+stream", 210, at(r, 0, 8));
  }
  {
    // K4: EXIT (control stall 5) issued at t -> done = t + 5 + 8 +
    // 7; ledger control_floor 5 / exit_branch 8 / exit_drain 7
    const Run r = J("kb3_exit");
    const std::uint64_t t_exit = at(r, 0, 3);
    check(r.stats.cycles == t_exit + 20,
          "kb3 done = EXIT+stall+branch+tail", t_exit + 20,
          r.stats.cycles);
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/kb3_exit.stall.tsv";
    run_ledger("kb3_exit", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    std::uint64_t cf = 0, eb = 0, ed = 0;
    for (const auto& kv : l.res) {
      if (kv.first.second == "control_floor") cf += kv.second;
      if (kv.first.second == "exit_branch") eb += kv.second;
      if (kv.first.second == "exit_drain") ed += kv.second;
    }
    check(cf == 4, "kb3 ledger control_floor", 4, cf);
    check(eb == 8, "kb3 ledger exit_branch", 8, eb);
    check(ed == 7, "kb3 ledger exit_drain", 7, ed);
    check(l.issued_bad == 0, "kb3 issued==events", 0, l.issued_bad);
  }
  {
    // K4 drain path: a cp.async group completing at ~50 outlives
    // EXIT's own floor (~13): done = ~50 + 7
    const Run r = J("kb3b_exit_drain");
    check(r.stats.cycles >= 52 && r.stats.cycles <= 62,
          "kb3b done = drain+tail", 57, r.stats.cycles);
  }
  {
    // drain-settle starvation (CUF-C84B-SPARSE-S05): warp 1 drains
    // to ~issue+50 while the non-yielding sibling spends the issue
    // slot ahead of it each sweep; the exhausted warp must still
    // reach the settle pass (before the fix this deadlocked once
    // the sibling exhausted and the queue emptied)
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    run_ledger("kb4_drain_starve", dir + "/kb4.stall.tsv", &warps,
               &st);
    check(warps.size() == 2 && warps[0].done && warps[1].done,
          "kb4 both warps done", 2, warps.size());
    check(warps[1].done_cycle >= 48 && warps[1].done_cycle <= 62,
          "kb4 drainer done at drain target", 55,
          warps[1].done_cycle);
    check(warps[0].done_cycle >= 110 && warps[0].done_cycle <= 135,
          "kb4 sibling done after its span", 124,
          warps[0].done_cycle);
    check(st.issued == st.completed,
          "kb4 issued==completed", st.issued, st.completed);
  }
  // ---- protocol-v5 G1 wait result / R2 SYNCS service -----------------
  {
    // (a) one warp: SYNCS arrive completes its own phase; the wait
    // issues at once (no semantic hold) and the reader lands
    // wait-issue + 38 on reason barrier:SYNCS.PHASECHK
    const Run r = J("g1_self");
    check(at(r, 0, 1) == at(r, 0, 0) + 1, "g1_self wait issues at once",
          at(r, 0, 0) + 1, at(r, 0, 1));
    check(at(r, 0, 3) == at(r, 0, 1) + 38,
          "g1_self reader = wait + result_lat", at(r, 0, 1) + 38,
          at(r, 0, 3));
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    const std::string tsv = dir + "/g1_self.stall.tsv";
    run_ledger("g1_self", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    check(l.reasons.count("barrier:SYNCS.PHASECHK") == 1,
          "g1_self reader reason barrier:SYNCS.PHASECHK", 1,
          l.reasons.count("barrier:SYNCS.PHASECHK"));
    check(l.reasons.count("semantic:mbarrier_wait") == 0,
          "g1_self no semantic:mbarrier_wait span", 0,
          l.reasons.count("semantic:mbarrier_wait"));
  }
  {
    // (b) the wait issues long before the other warp's SYNCS arrive:
    // reader at arrive-issue + 38, independent of 88 / 228
    const Run r = J("g1_handoff");
    check(at(r, 0, 0) <= 3, "g1_handoff wait not held", 3, at(r, 0, 0));
    check(at(r, 0, 2) == at(r, 1, 40) + 38,
          "g1_handoff reader = arrive + result_lat", at(r, 1, 40) + 38,
          at(r, 0, 2));
  }
  {
    // (c) the wait issues after the other warp's SYNCS arrive:
    // reader at wait-issue + 38
    const Run r = J("g1_late_wait");
    check(at(r, 0, 41) == at(r, 0, 40) + 1,
          "g1_late_wait next free", at(r, 0, 40) + 1, at(r, 0, 41));
    check(at(r, 0, 42) == at(r, 0, 40) + 38,
          "g1_late_wait reader = wait + result_lat", at(r, 0, 40) + 38,
          at(r, 0, 42));
  }
  {
    // (d) non-SYNCS completion: pending wait -> reader at
    // completion + suspended_wake(228) + 38
    const Run r = J("g1_nonsyncs_pend");
    check(at(r, 0, 0) <= 3, "g1_nonsyncs_pend wait not held", 3,
          at(r, 0, 0));
    check(at(r, 0, 2) == at(r, 1, 40) + 228 + 38,
          "g1_nonsyncs_pend reader = comp+susp+lat",
          at(r, 1, 40) + 266, at(r, 0, 2));
  }
  {
    // (d) non-SYNCS completion, wait issued at/after release:
    // reader at wait-issue + 38
    const Run r = J("g1_nonsyncs_late");
    check(at(r, 0, 201) == at(r, 0, 200) + 1,
          "g1_nonsyncs_late next free", at(r, 0, 200) + 1,
          at(r, 0, 201));
    check(at(r, 0, 202) == at(r, 0, 200) + 38,
          "g1_nonsyncs_late reader = wait + lat", at(r, 0, 200) + 38,
          at(r, 0, 202));
  }
  {
    // (e) NANOSLEEP.SYNCS sleeper on a SYNCS-completed phase wakes at
    // the completion date (no syncs_wake / wake latency)
    const Run r = J("g1_sleep_syncs");
    check(at(r, 0, 2) == at(r, 1, 40),
          "g1_sleep_syncs wake at completion", at(r, 1, 40),
          at(r, 0, 2));
  }
  {
    // (e) same sleeper on a non-SYNCS-completed phase keeps today's
    // dating: completion + suspended_wake(228) + syncs_wake(136)
    const Run r = J("g1_sleep_nonsyncs");
    check(at(r, 0, 2) == at(r, 1, 40) + 228 + 136,
          "g1_sleep_nonsyncs wake at susp+syncs_wake",
          at(r, 1, 40) + 364, at(r, 0, 2));
  }
  {
    // (f) wait without a dependency barrier: the next issue is held
    // until the result time (result floor)
    const Run r = J("g1_nobarrier");
    check(at(r, 0, 2) == at(r, 0, 1) + 38,
          "g1_nobarrier next issue at result", at(r, 0, 1) + 38,
          at(r, 0, 2));
  }
  {
    // (f) no-barrier wait pending at completion: next issue at the
    // resolved result time (completion + 38 for a SYNCS arrive)
    const Run r = J("g1_nobarrier_pend");
    check(at(r, 0, 1) == at(r, 1, 40) + 38,
          "g1_nobarrier_pend next issue at resolved result",
          at(r, 1, 40) + 38, at(r, 0, 1));
  }
  {
    // (g) a wait on a phase that never completes ends in the deadlock
    // report naming the mbarrier object
    try {
      run_fixture(dir + "/g1_deadlock.job");
      ++failures;
      std::fprintf(stderr,
                   "FAIL g1 deadlock: run finished without error\n");
    } catch (const cyclescope::EngineError& e) {
      const std::string msg = e.what();
      check(msg.rfind("unsupported:engine:deadlock", 0) == 0,
            "g1 deadlock prefix", 0, msg.empty() ? 1 : 0);
      check(msg.find("object 0") != std::string::npos,
            "g1 deadlock names mbarrier object", 0,
            msg.find("object 0") == std::string::npos ? 1 : 0);
      if (msg.find("object 0") == std::string::npos)
        std::fprintf(stderr, "  got error text: %s\n", msg.c_str());
    }
  }
  {
    // (h) service interval 5 on one SM: two warps issuing a SYNCS
    // arrive in the same cycle are served 5 apart; their readers land
    // 5 apart
    const Run r = J("g1_service");
    const std::uint64_t d = at(r, 1, 3) - at(r, 0, 3);
    check(d == 5 || d == static_cast<std::uint64_t>(-5),
          "g1_service readers 5 apart", 5,
          d == static_cast<std::uint64_t>(-5) ? 5 : d);
    check(at(r, 0, 0) == at(r, 1, 0),
          "g1_service arrives same cycle", at(r, 0, 0), at(r, 1, 0));
  }
  {
    // (h1) two pending waits on the same dependency barrier: the
    // barrier stays pending until the later result is fixed; the
    // reader lands on the second arrive + 38
    const Run r = J("g1_two_pending");
    check(at(r, 0, 0) <= 4, "g1_two_pending wait0 issued", 4,
          at(r, 0, 0));
    check(at(r, 0, 1) <= 4, "g1_two_pending wait1 issued", 4,
          at(r, 0, 1));
    check(at(r, 0, 3) == at(r, 1, 241) + 38,
          "g1_two_pending reader = late arrive + result_lat",
          at(r, 1, 241) + 38, at(r, 0, 3));
  }
  {
    // (h1') same with the arrive order swapped: the reader follows
    // the second (object-0) arrive
    const Run r = J("g1_two_pending_rev");
    check(at(r, 0, 0) <= 4, "g1_two_pending_rev wait0 issued", 4,
          at(r, 0, 0));
    check(at(r, 0, 1) <= 4, "g1_two_pending_rev wait1 issued", 4,
          at(r, 0, 1));
    check(at(r, 0, 3) == at(r, 1, 241) + 38,
          "g1_two_pending_rev reader = late arrive + result_lat",
          at(r, 1, 241) + 38, at(r, 0, 3));
  }
  {
    // (h2) fractional service interval 5.25: the frontier keeps the
    // fraction, service starts 0, 5.25, 10.5, 15.75 land on cycles
    // 0, 6, 11, 16 -> the two readers land one cycle later than with
    // the integer interval
    const Run ri = J("g1_service");
    const Run rf = J("g1_service_frac");
    const auto lo = [](const Run& r) {
      return std::min(at(r, 0, 3), at(r, 1, 3));
    };
    const auto hi = [](const Run& r) {
      return std::max(at(r, 0, 3), at(r, 1, 3));
    };
    check(lo(rf) == lo(ri) + 1, "g1_service_frac lo +1", lo(ri) + 1,
          lo(rf));
    check(hi(rf) == hi(ri) + 1, "g1_service_frac hi +1", hi(ri) + 1,
          hi(rf));
  }
  {
    // (h) SYNCS on different SMs do not interact: identical timing
    const Run r = J("g1_service_2sm");
    check(at(r, 1, 0) == at(r, 0, 0), "g1_service_2sm arrive same cycle",
          at(r, 0, 0), at(r, 1, 0));
    check(at(r, 1, 3) == at(r, 0, 3), "g1_service_2sm readers equal",
          at(r, 0, 3), at(r, 1, 3));
  }
  // ---- mio_basic protocol v3 (M1/M2/M3) ------------------------------
  {
    // M1: a warp's producers return one per return interval on chain
    // 1; the reader lands at the last turn start + result latency
    const Run r = J("mio_return_chain");
    const std::uint64_t t0 = at(r, 0, 0);
    check(at(r, 0, 4) == t0 + 6 + 23, "mio_return_chain R",
          t0 + 29, at(r, 0, 4));
    const Run r4 = J("mio_return_chain_r4");
    const std::uint64_t t4 = at(r4, 0, 0);
    check(at(r4, 0, 4) == t4 + 12 + 23, "mio_return_chain r4 R",
          t4 + 35, at(r4, 0, 4));
  }
  {
    // M1 chains: A and B on one chain serialize on it; B on chain 3
    // is independent
    const Run r = J("mio_chain_sharing");
    const std::uint64_t t0 = at(r, 0, 0);
    check(at(r, 0, 8) == t0 + 14 + 23, "mio_chain_sharing R",
          t0 + 37, at(r, 0, 8));
    const Run r3 = J("mio_chain_sharing_b3");
    const std::uint64_t t3 = at(r3, 0, 0);
    check(at(r3, 0, 8) == t3 + 30, "mio_chain_sharing b3 R",
          t3 + 30, at(r3, 0, 8));
  }
  {
    // M2: one server per SM and operation class; 4 warps x P P R,
    // the 8 turns land 2 apart in issue order
    const Run r = J("mio_sm_service");
    const std::uint64_t t0 = at(r, 0, 0);
    std::uint64_t latest = 0;
    for (std::uint32_t w = 0; w < 4; ++w)
      latest = std::max(latest, at(r, w, 2));
    check(latest == t0 + 14 + 23, "mio_sm_service latest reader",
          t0 + 37, latest);
    // two operation classes -> two independent servers
    const Run r2 = J("mio_sm_service_2cls");
    const std::uint64_t u0 = at(r2, 0, 0);
    latest = 0;
    for (std::uint32_t w = 0; w < 4; ++w)
      latest = std::max(latest, at(r2, w, 2));
    check(latest == u0 + 6 + 23, "mio_sm_service_2cls latest reader",
          u0 + 29, latest);
    // protocol v4: the cycles between one warp's turns serve other
    // warps -- B (two NOPs behind A) takes turns t0+3 .. t0+9
    const Run rg = J("mio_sm_service_gaps");
    const std::uint64_t g0 = at(rg, 0, 0);
    check(at(rg, 0, 4) == g0 + 6 + 23, "mio_gaps A reader",
          g0 + 29, at(rg, 0, 4));
    check(at(rg, 1, 2) == g0 + 2, "mio_gaps B first producer",
          g0 + 2, at(rg, 1, 2));
    check(at(rg, 1, 6) == g0 + 9 + 23, "mio_gaps B reader",
          g0 + 32, at(rg, 1, 6));
  }
  {
    // M3: the in-flight group of four waits at issue; 8 producers
    // land t0..t0+3 and t0+8..t0+11
    const Run r = J("mio_inflight_group");
    const std::uint64_t t0 = at(r, 0, 0);
    for (std::uint64_t i = 0; i < 4; ++i) {
      check(at(r, 0, i) == t0 + i, "mio_group first burst",
            t0 + i, at(r, 0, i));
      check(at(r, 0, i + 4) == t0 + 8 + i, "mio_group second burst",
            t0 + 8 + i, at(r, 0, i + 4));
    }
    check(at(r, 0, 8) == t0 + 14 + 26, "mio_group R",
          t0 + 40, at(r, 0, 8));
    // 6 producers: 5th lands t0+8, 6th t0+9
    const Run r6 = J("mio_inflight_group6");
    const std::uint64_t t6 = at(r6, 0, 0);
    check(at(r6, 0, 4) == t6 + 8, "mio_group6 5th", t6 + 8,
          at(r6, 0, 4));
    check(at(r6, 0, 5) == t6 + 9, "mio_group6 6th", t6 + 9,
          at(r6, 0, 5));
    check(at(r6, 0, 6) == t6 + 10 + 26, "mio_group6 R",
          t6 + 36, at(r6, 0, 6));
    // two bursts >= 100 cycles apart: the count does not carry
    const Run r2 = J("mio_inflight_group_2x6");
    const std::uint64_t u0 = at(r2, 0, 0);
    const std::uint64_t u5 = at(r2, 0, 5);
    const std::uint64_t u6 = at(r2, 0, 6);  // filler
    const std::uint64_t u8 = at(r2, 0, 8);  // second burst start
    check(u5 == u0 + 9, "mio_2x6 first burst end", u0 + 9, u5);
    check(u8 >= u6 + 100, "mio_2x6 burst gap", u6 + 100, u8);
    check(at(r2, 0, 12) == u8 + 8, "mio_2x6 burst2 5th", u8 + 8,
          at(r2, 0, 12));
    check(at(r2, 0, 13) == u8 + 9, "mio_2x6 burst2 6th", u8 + 9,
          at(r2, 0, 13));
    // mio_inflight ledger: 4 cycles per burst
    const std::string tsv = dir + "/mio_inflight_group_2x6.stall.tsv";
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    run_ledger("mio_inflight_group_2x6", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    std::uint64_t mi = 0;
    for (const auto& [k, c] : l.res)
      if (k.second == "mio_inflight") mi += c;
    check(mi == 8, "mio_inflight ledger cycles", 8, mi);
  }
  {
    // a 2-cycle unit: producers and the reader take turns 2.0 apart;
    // the reader's wait is the unit's, no barrier:* reason on it
    const Run r = J("mio_unit_two_cycle");
    const std::uint64_t t0 = at(r, 0, 0);
    for (std::uint64_t i = 0; i < 4; ++i)
      check(at(r, 0, i) == t0 + 2 * i, "unit_two_cycle C",
            t0 + 2 * i, at(r, 0, i));
    check(at(r, 0, 4) == t0 + 8, "unit_two_cycle R", t0 + 8,
          at(r, 0, 4));
    const std::string tsv = dir + "/mio_unit_two_cycle.stall.tsv";
    std::vector<cyclescope::WarpState> warps;
    cyclescope::ResultStats st;
    run_ledger("mio_unit_two_cycle", tsv, &warps, &st);
    const Ledger l = read_ledger(tsv);
    std::uint64_t bar = 0;
    for (const auto& [k, c] : l.res)
      if (k.first == 0x1040 && k.second.rfind("barrier", 0) == 0)
        bar += c;
    check(bar == 0, "unit_two_cycle no barrier on R", 0, bar);
  }
  {
    // M1 on the kind-5 storage.shared.issue row: result =
    // turn_start + cycles + completion latency
    const Run r = J("mio_lds_turn");
    const std::uint64_t t0 = at(r, 0, 0);
    check(at(r, 0, 4) == t0 + 6 + 2 + 21, "lds_turn R", t0 + 29,
          at(r, 0, 4));
    // the SMSP frontier is not advanced by the LDS turns: the two
    // turn-less kind-5 stores behind them start at their own issue
    // and the reader lands at the second store's end
    const Run rf = J("mio_lds_turn_frontier");
    const std::uint64_t f0 = at(rf, 0, 0);
    check(at(rf, 0, 6) == f0 + 14 + 10 + 30, "lds_turn frontier R",
          f0 + 54, at(rf, 0, 6));
  }
  {
    // M1 on constant_paths C2: the resident-line re-read counts from
    // the turn start; R(with) - R(without) = 3
    const Run r = J("mio_constant_turn");
    const Run ro = J("mio_constant_turn_off");
    check(at(r, 0, 6) == at(ro, 0, 6) + 3, "constant_turn R delta",
          at(ro, 0, 6) + 3, at(r, 0, 6));
  }
  {
    // ---- instruction.warp_queue (warp_queue protocol Q1-Q4) --------
    // Independent model of the rule: one warp issues one instruction
    // per cycle; a member needs fewer than 10 unreleased entries (whole
    // cycles: released at or before the cycle), then
    // start = max(F, issue + 5.5 + 0.5 r), release = start + r.
    struct WqOp { double r; bool member; };
    auto model = [](const std::vector<WqOp>& ops) {
      std::vector<std::uint64_t> out;
      std::vector<double> q;
      double free_at = 0.0;
      std::uint64_t t = 0;
      for (const auto& op : ops) {
        if (op.member) {
          for (;;) {
            std::vector<double> keep;
            for (double x : q)
              if (x > static_cast<double>(t) + 1e-9) keep.push_back(x);
            q = keep;
            if (q.size() < 10) break;
            t = static_cast<std::uint64_t>(std::ceil(q.front() - 1e-9));
          }
        }
        out.push_back(t);
        if (op.member) {
          const double st =
              std::max(free_at, static_cast<double>(t) + 5.5 +
                                    0.5 * op.r);
          free_at = st + op.r;
          q.push_back(free_at);
        }
        ++t;
      }
      return out;
    };
    auto rep = [](double r, int n, bool member = true) {
      return std::vector<WqOp>(n, WqOp{r, member});
    };
    auto cat = [](std::vector<WqOp> a, const std::vector<WqOp>& b) {
      a.insert(a.end(), b.begin(), b.end());
      return a;
    };
    auto compare_model = [&](const char* name,
                             const std::vector<WqOp>& ops) {
      const Run r = J(name);
      const auto m = model(ops);
      const std::uint64_t t0 = at(r, 0, 0);
      for (std::size_t i = 0; i < m.size(); ++i)
        check(at(r, 0, i) == t0 + m[i], name, t0 + m[i], at(r, 0, i));
      return r;
    };
    auto reason_cycles = [&](const char* name, const char* reason) {
      const std::string tsv = dir + "/" + name + ".stall.tsv";
      std::vector<cyclescope::WarpState> warps;
      cyclescope::ResultStats st;
      run_ledger(name, tsv, &warps, &st);
      const Ledger l = read_ledger(tsv);
      std::uint64_t sum = 0;
      for (const auto& [k, c] : l.res)
        if (k.second == reason) sum += c;
      return sum;
    };

    // a. 16 STS: 0-9 free, then +12, 16, 20, 24, 28, 32; the waits
    // book 2 + 5 x 3 = 17 cycles of warp_queue:mio
    {
      const Run r = compare_model("wq_sts16", rep(4.0, 16));
      const std::uint64_t t0 = at(r, 0, 0);
      for (std::uint64_t i = 0; i < 10; ++i)
        check(at(r, 0, i) == t0 + i, "wq a free", t0 + i, at(r, 0, i));
      const std::uint64_t want[6] = {12, 16, 20, 24, 28, 32};
      for (int i = 0; i < 6; ++i)
        check(at(r, 0, 10 + i) == t0 + want[i], "wq a waits",
              t0 + want[i], at(r, 0, 10 + i));
      check(reason_cycles("wq_sts16", "warp_queue:mio") == 17,
            "wq a ledger mio", 17,
            reason_cycles("wq_sts16", "warp_queue:mio"));
      check(reason_cycles("wq_sts16", "warp_queue:lg") == 0,
            "wq a ledger lg", 0,
            reason_cycles("wq_sts16", "warp_queue:lg"));
      // the queue row disabled: ten and more stores issue one apart
      const Run off = J("wq_sts16_off");
      for (std::uint64_t i = 0; i < 16; ++i)
        check(at(off, 0, i) == at(off, 0, 0) + i, "wq a disabled",
              at(off, 0, 0) + i, at(off, 0, i));
    }
    // b. 16 STG (4 B): 0-9 free, then +15 (ceil 14.5), 21, ... 45;
    // lg because every entry and the waiter are global
    {
      const Run r = compare_model("wq_stg16", rep(6.0, 16));
      const std::uint64_t t0 = at(r, 0, 0);
      for (std::uint64_t i = 0; i < 10; ++i)
        check(at(r, 0, i) == t0 + i, "wq b free", t0 + i, at(r, 0, i));
      const std::uint64_t want[6] = {15, 21, 27, 33, 39, 45};
      for (int i = 0; i < 6; ++i)
        check(at(r, 0, 10 + i) == t0 + want[i], "wq b waits",
              t0 + want[i], at(r, 0, 10 + i));
      check(reason_cycles("wq_stg16", "warp_queue:lg") == 30,
            "wq b ledger lg", 30,
            reason_cycles("wq_stg16", "warp_queue:lg"));
      check(reason_cycles("wq_stg16", "warp_queue:mio") == 0,
            "wq b ledger mio", 0,
            reason_cycles("wq_stg16", "warp_queue:mio"));
    }
    // c. 24 LDS: releases 8.5 + 2 j; producers 0-11 are free, the 13th
    // (index 12: 12 issued, 10 unreleased at cycle 12) is the first to
    // wait (issues at 13), every later one issues 2 after the one
    // before
    {
      const Run r = compare_model("wq_lds24", rep(2.0, 24));
      const std::uint64_t t0 = at(r, 0, 0);
      for (std::uint64_t i = 0; i < 12; ++i)
        check(at(r, 0, i) == t0 + i, "wq c free", t0 + i, at(r, 0, i));
      check(at(r, 0, 12) == t0 + 13, "wq c first wait", t0 + 13,
            at(r, 0, 12));
      for (std::uint64_t i = 13; i < 24; ++i)
        check(at(r, 0, i) == at(r, 0, i - 1) + 2, "wq c pace",
              at(r, 0, i - 1) + 2, at(r, 0, i));
    }
    // d. 6 STS then 18 LDS: indices 11-15 four apart, 16 on two apart
    {
      const Run r = compare_model("wq_sts6_lds18",
                                  cat(rep(4.0, 6), rep(2.0, 18)));
      for (std::uint64_t i = 11; i <= 15; ++i)
        check(at(r, 0, i) == at(r, 0, i - 1) + 4, "wq d four apart",
              at(r, 0, i - 1) + 4, at(r, 0, i));
      for (std::uint64_t i = 16; i < 24; ++i)
        check(at(r, 0, i) == at(r, 0, i - 1) + 2, "wq d two apart",
              at(r, 0, i - 1) + 2, at(r, 0, i));
    }
    // e. 12 LDS then 12 STS: the pace changes from 2 to 4 at index 22
    // and 23
    {
      const Run r = compare_model("wq_lds12_sts12",
                                  cat(rep(2.0, 12), rep(4.0, 12)));
      for (std::uint64_t i = 12; i <= 21; ++i)
        check(at(r, 0, i) == at(r, 0, i - 1) + 2, "wq e two apart",
              at(r, 0, i - 1) + 2, at(r, 0, i));
      for (std::uint64_t i = 22; i <= 23; ++i)
        check(at(r, 0, i) == at(r, 0, i - 1) + 4, "wq e four apart",
              at(r, 0, i - 1) + 4, at(r, 0, i));
    }
    // f. class of the wait
    {
      compare_model("wq_sts10_stg6", cat(rep(4.0, 10), rep(6.0, 6)));
      check(reason_cycles("wq_sts10_stg6", "warp_queue:mio") > 0 &&
                reason_cycles("wq_sts10_stg6", "warp_queue:lg") == 0,
            "wq f sts10_stg6 mio", 1, 0);
      compare_model("wq_stg6_ldg20", cat(rep(6.0, 6), rep(4.0, 20)));
      check(reason_cycles("wq_stg6_ldg20", "warp_queue:lg") > 0 &&
                reason_cycles("wq_stg6_ldg20", "warp_queue:mio") == 0,
            "wq f stg6_ldg lg", 1, 0);
      compare_model("wq_stg10_sts6", cat(rep(6.0, 10), rep(4.0, 6)));
      check(reason_cycles("wq_stg10_sts6", "warp_queue:mio") > 0 &&
                reason_cycles("wq_stg10_sts6", "warp_queue:lg") == 0,
            "wq f stg10_sts6 mio", 1, 0);
    }
    // g. Q4: M stores take chain-1 turns of 4 at t0 + 0, 4, ...; the
    // two LDS (issued t0 + M, t0 + M + 1) turn at the chain frontier
    // 4 M and 4 M + 2; the reader lands at the last turn start + the
    // LDS result latency of the fixture (21): t0 + 4 M + 2 + 21 = t0 + 39
    // for M = 4.  Control M = 0: the chain is free, turns t0 and
    // t0 + 2, reader t0 + 2 + 21 = t0 + 23 (unchanged from before).
    {
      const Run r = J("wq_fwd4");
      const std::uint64_t t0 = at(r, 0, 0);
      check(at(r, 0, 6) == t0 + 4 * 4 + 2 + 21, "wq g reader M=4",
            t0 + 39, at(r, 0, 6));
      const Run c = J("wq_fwd0");
      const std::uint64_t c0 = at(c, 0, 0);
      check(at(c, 0, 2) == c0 + 2 + 21, "wq g reader M=0", c0 + 23,
            at(c, 0, 2));
    }
    // h. non-members: S2R and MUFU runs issue as without the queue
    // (rows disabled), one per cycle; a remote shared store and a
    // predicated-off store (no row) are no entries
    {
      for (const char* name : {"wq_s2r28", "wq_mufu12"}) {
        const std::string off = std::string(name) + "_off";
        const Run a = J(name);
        const Run b = J(off.c_str());
        const std::size_t n = std::string(name) == "wq_s2r28" ? 28 : 12;
        for (std::size_t i = 0; i < n; ++i)
          check(at(a, 0, i) - at(a, 0, 0) == at(b, 0, i) - at(b, 0, 0),
                name, at(b, 0, i) - at(b, 0, 0),
                at(a, 0, i) - at(a, 0, 0));
        for (std::size_t i = 0; i < n; ++i)
          check(at(a, 0, i) == at(a, 0, 0) + i, name, at(a, 0, 0) + i,
                at(a, 0, i));
      }
      const Run rem = J("wq_remote16");
      for (std::uint64_t i = 0; i < 16; ++i)
        check(at(rem, 0, i) == at(rem, 0, 0) + i, "wq h remote",
              at(rem, 0, 0) + i, at(rem, 0, i));
      std::vector<WqOp> ops = cat(cat(rep(4.0, 6), rep(0.0, 6, false)),
                                  rep(4.0, 10));
      compare_model("wq_pred_off", ops);
    }
    // i. per warp: two warps on one SMSP each issuing 10 STS never wait
    // on the queue
    {
      const Run r = J("wq_two_warps_smsp");
      (void)r;
      check(reason_cycles("wq_two_warps_smsp", "warp_queue:mio") == 0,
            "wq i two warps", 0,
            reason_cycles("wq_two_warps_smsp", "warp_queue:mio"));
    }
  }
  {
    // ---- instruction.warp_queue v2 (S1-S4) --------------------------
    auto reason_at = [&](const char* name, const char* reason,
                         const std::set<std::uint64_t>* pcs) {
      const std::string tsv = dir + "/" + name + ".stall.tsv";
      std::vector<cyclescope::WarpState> warps;
      cyclescope::ResultStats st;
      run_ledger(name, tsv, &warps, &st);
      const Ledger l = read_ledger(tsv);
      std::uint64_t sum = 0;
      for (const auto& [k, c] : l.res)
        if (k.second == reason && (!pcs || pcs->count(k.first)))
          sum += c;
      return sum;
    };
    auto readers = [&](const Run& r, std::uint32_t lo, std::uint32_t hi) {
      std::vector<std::uint64_t> v;
      for (std::uint32_t w = lo; w < hi; ++w) v.push_back(at(r, w, 2));
      std::sort(v.begin(), v.end());
      return v;
    };
    // S1: 16 warps (4 per SMSP) each 2 LDS + reader (a reader behind a
    // single producer does not see its barrier).  The class's SM server
    // starts the 32 releases at least 1.0 apart; the reader of a warp
    // (floored at the second release + 14.7, above the 21-cycle LDS
    // result) issues at the ceiling: 16 distinct cycles spanning at
    // least 15
    {
      const Run r = J("wq2_s1_lds16");
      const auto v = readers(r, 0, 16);
      bool strict = true;
      for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] <= v[i - 1]) strict = false;
      check(strict, "wq2 s1 distinct releases", 1, strict);
      check(v.back() - v.front() >= 15, "wq2 s1 span", 15,
            v.back() - v.front());
    }
    // two classes (8 warps x 2 LDS, 8 warps x 2 SHFL.IDX): 16 releases
    // per class from about 6.5, the last at most 6.5 + 15 + 3 (issue
    // skew) + r + 14.7: readers by t0 + 46; one shared server would
    // need 32 slots (t0 + 54 and later)
    {
      const Run r = J("wq2_s1_two_classes");
      for (int g = 0; g < 2; ++g) {
        const auto v = readers(r, 8 * g, 8 * g + 8);
        check(v.back() <= at(r, 0, 0) + 46, "wq2 s1 class alone", 46,
              v.back() - at(r, 0, 0));
        check(v.back() >= at(r, 0, 0) + 30, "wq2 s1 class releases", 30,
              v.back() - at(r, 0, 0));
      }
    }
    // S2: 16 warps x 10 STS, no SM slots: the releases follow from
    // each warp's own issue cycles (v1 rule, r = 4); at no cycle are
    // more than 46 entries unreleased, and the bound binds
    {
      const Run r = J("wq2_s2_sts");
      struct E { std::uint64_t t; double rel; };
      std::vector<E> all;
      std::uint64_t tmax = 0;
      for (std::uint32_t w = 0; w < 16; ++w) {
        double free_at = 0.0;
        for (std::uint64_t k = 0; k < 10; ++k) {
          const std::uint64_t t = at(r, w, k);
          const double st =
              std::max(free_at, static_cast<double>(t) + 5.5 + 2.0);
          free_at = st + 4.0;
          all.push_back({t, free_at});
          tmax = std::max(tmax, t);
        }
      }
      std::size_t peak = 0;
      for (std::uint64_t t = 0; t <= tmax; ++t) {
        std::size_t n = 0;
        for (const auto& e : all)
          if (e.t <= t && e.rel > static_cast<double>(t) + 1e-9) ++n;
        peak = std::max(peak, n);
      }
      check(peak <= 46, "wq2 s2 bound", 46, peak);
      check(peak == 46, "wq2 s2 binds", 46, peak);
    }
    // S2': waiters pass in the order they began to wait: the warps of
    // 16 x 128 STS finish within 10 % of the longest
    {
      const Run r = J("wq2_s2_fair");
      std::uint64_t lo = ~0ull, hi = 0;
      for (std::uint32_t w = 0; w < 16; ++w) {
        const std::uint64_t t = at(r, w, 127);
        lo = std::min(lo, t);
        hi = std::max(hi, t);
      }
      check((hi - lo) * 10 <= hi, "wq2 s2' fairness spread", hi / 10,
            hi - lo);
    }
    // 32 warps (8 per SMSP) x 3 STS against the SM bound: every
    // instance issues (the head of the waiting list is reached even
    // when earlier candidates of its SMSP wait behind it)
    {
      const Run r = J("wq2_s2_burst");
      check(r.issue.size() == 96, "wq2 s2' burst all issued", 96,
            r.issue.size());
    }
    // the same with global stores held by the store buffer's own
    // waiting list: the two lists must not wait for each other
    {
      const Run r = J("wq2_s2_stg_burst");
      check(r.issue.size() == 96, "wq2 s2' stg burst all issued", 96,
            r.issue.size());
    }
    // a warp that exits while the others wait does not stall the SM:
    // every instance of every warp issues
    {
      const Run r = J("wq2_s2_exit");
      std::uint64_t n = 0;
      for (std::uint32_t w = 0; w < 16; ++w)
        n += r.job.warps[w].instance_count;
      check(r.issue.size() == n, "wq2 s2' exit all issued", n,
            r.issue.size());
    }
    // a late warp with no entries of its own that waits on the SM
    // bound: STS -> warp_queue:mio, STG -> warp_queue:lg
    for (const char* name : {"wq2_s2_late_stg", "wq2_s2_late_sts"}) {
      const Run r = J(name);
      std::set<std::uint64_t> pcs;
      for (std::uint32_t w = 12; w < 16; ++w)
        pcs.insert(0x1000 + 16 * (r.job.warps[w].first_instance + 12));
      const bool stg = std::string(name) == "wq2_s2_late_stg";
      const std::uint64_t want =
          reason_at(name, stg ? "warp_queue:lg" : "warp_queue:mio", &pcs);
      const std::uint64_t other =
          reason_at(name, stg ? "warp_queue:mio" : "warp_queue:lg", &pcs);
      check(want > 0, name, 1, want);
      check(other == 0, name, 0, other);
    }
    // S3: 2 LDS then the reader: releases 8.5, 10.5 -> floor
    // ceil(10.5 + 14.7) = 26 (the fixture's own result is later than
    // t0 + 23 only by nothing: turns t0, t0 + 2, latency 21)
    {
      const Run r = J("wq2_ld2");
      check(at(r, 0, 2) == at(r, 0, 0) + 26, "wq2 s3 lds", 26,
            at(r, 0, 2) - at(r, 0, 0));
      // 4 STS + 2 LDS (no Q4 turn): STS releases 11.5 .. 23.5, LDS
      // 25.5, 27.5 -> reader at ceil(27.5 + 14.7) = t0 + 43
      const Run r4 = J("wq2_sts4_lds2");
      check(at(r4, 0, 6) == at(r4, 0, 0) + 43, "wq2 s3 sts4 lds2", 43,
            at(r4, 0, 6) - at(r4, 0, 0));
      // LDG, c = 28.3: releases 11.5, 15.5 -> floor ceil(43.8) = 44;
      // the reader is the later of that and the same fixture with
      // c = 0 (the fixture's own memory return)
      const Run g = J("wq2_ldg2");
      const Run g0 = J("wq2_ldg2_c0");
      const std::uint64_t own = at(g0, 0, 2) - at(g0, 0, 0);
      check(at(g, 0, 2) - at(g, 0, 0) == std::max<std::uint64_t>(own, 44),
            "wq2 s3 ldg", std::max<std::uint64_t>(own, 44),
            at(g, 0, 2) - at(g, 0, 0));
    }
    // R1: two member LDG on a line made resident by an earlier load:
    // releases 11.5, 15.5 -> result at ceil(15.5 + 28.3) = t0 + 44
    // exactly, whatever the L1 latency leaf (60 or 5)
    for (const char* name : {"wq3_l1_res60", "wq3_l1_res5"}) {
      const Run r = J(name);
      check(at(r, 0, 4) == at(r, 0, 2) + 44, name, 44,
            at(r, 0, 4) - at(r, 0, 2));
    }
    // not resident (both miss): v2's rule, the later of the floor and
    // the hierarchy's own return (the same fixture with c = 0)
    {
      const Run r = J("wq3_l1_cold");
      const Run r0 = J("wq3_l1_cold_c0");
      const std::uint64_t own = at(r0, 0, 2) - at(r0, 0, 0);
      check(at(r, 0, 2) - at(r, 0, 0) == std::max<std::uint64_t>(own, 44),
            "wq3 cold", std::max<std::uint64_t>(own, 44),
            at(r, 0, 2) - at(r, 0, 0));
    }
    // S4: store fast path (store-L2 leaf with two outstanding
    // requests): the same 4-byte line again and again, and two warps
    // each rewriting their own line, never wait on the store buffer;
    // alternating two lines, 2-byte stores and 16-byte lanes over four
    // lines do
    check(reason_at("wq4_same_line", "store_buffer", nullptr) == 0,
          "wq2 s4 same line", 0,
          reason_at("wq4_same_line", "store_buffer", nullptr));
    check(reason_at("wq4_two_warps", "store_buffer", nullptr) == 0,
          "wq2 s4 two warps", 0,
          reason_at("wq4_two_warps", "store_buffer", nullptr));
    for (const char* name :
         {"wq4_alternate", "wq4_two_byte", "wq4_wide16"})
      check(reason_at(name, "store_buffer", nullptr) > 0, name, 1, 0);
  }
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
