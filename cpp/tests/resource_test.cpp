// Resource binding + memory data-ready fixtures.
// argv[1] = fixtures directory with res_f{1,2,2b,3,4}.seg
//
// Each fixture runs to completion with a journal; the per-instance
// issue cycles are asserted against the bound mechanism values.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include "core/job_reader.hpp"
#include "core/mechanism_runtime.hpp"
#include "core/simulator.hpp"
#include "arch/architecture_module.hpp"

static int failures = 0;

static void check(bool ok, const char* what, long long got,
                  long long want) {
  if (ok)
    std::printf("  ok   %-52s got=%lld\n", what, got);
  else {
    ++failures;
    std::printf("  FAIL %-52s got=%lld want>=%lld\n", what, got, want);
  }
}

static std::map<std::uint64_t, std::uint64_t> issue_cycles(
    const std::string& path) {
  auto job = cyclescope::read_segment_file(path.c_str());
  cyclescope::MechanismRuntime::Options ro;
  cyclescope::MechanismRuntime rt(job, "h100", ro);
  cyclescope::Simulator::Options so;
  so.journal = true;
  cyclescope::Simulator sim(job, rt, so);
  auto st = sim.run();
  if (st.issued != st.completed || st.hard_failures)
    std::printf("  run anomaly issued=%llu completed=%llu hard=%llu\n",
                (unsigned long long)st.issued,
                (unsigned long long)st.completed,
                (unsigned long long)st.hard_failures);
  std::map<std::uint64_t, std::uint64_t> m;
  for (const auto& j : st.journal) m[j.instance] = j.issue_cycle;
  return m;
}

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  {  // F1: two MUFU on xu_smsp, issue_interval 8
    std::printf("F1 service issue_interval=8 resource=xu_smsp\n");
    auto m = issue_cycles(dir + "/res_f1.seg");
    check(m.count(0) && m.count(1) && m[1] - m[0] >= 8,
          "second MUFU issue - first >= 8", (long long)(m[1] - m[0]), 8);
  }
  {  // F2: two warps on one SM, DFMA on fp64_sm, issue_interval 16
    std::printf("F2 service issue_interval=16 resource=fp64_sm\n");
    auto m = issue_cycles(dir + "/res_f2.seg");
    long long gap = m.count(0) && m.count(1)
        ? std::llabs((long long)m[1] - (long long)m[0]) : -1;
    check(m.count(0) && m.count(1) && gap >= 16,
          "DFMA issues on fp64_sm >= 16 apart", gap, 16);
  }
  {  // F2b: max_inflight=1 completion=64 -> second waits for completion
    std::printf("F2b max_inflight=1 completion=64\n");
    auto m = issue_cycles(dir + "/res_f2b.seg");
    long long gap = m.count(0) && m.count(1)
        ? std::llabs((long long)m[1] - (long long)m[0]) : -1;
    check(m.count(0) && m.count(1) && gap >= 64,
          "second DFMA waits for first completion (+64)", gap, 64);
  }
  {  // F3: LDG level latency 300 -> dependents wait for the return.
    // issue_rules v1 R1: the register-dependent consumer issues the
    // next cycle; the barrier consumer (gap NOP for R3) still waits
    // for the return
    std::printf("F3 LDG return constrains dependents (300)\n");
    auto m = issue_cycles(dir + "/res_f3.seg");
    check(m.count(0) && m.count(1) && m[1] - m[0] == 1,
          "FFMA(reg dep) issue - LDG == 1 (R1)",
          (long long)(m[1] - m[0]), 1);
    check(m.count(2) && m.count(4) && m[4] - m[2] >= 300,
          "FFMA(barrier dep) issue - LDG >= 300",
          (long long)(m[4] - m[2]), 300);
  }
  {  // F4: LDS kind-6 TIME row cycles=200 -> the write barrier it
    // raises frees the dependent when the access finishes
    std::printf("F4 LDS kind-6 TIME row (200) constrains dependent\n");
    auto m = issue_cycles(dir + "/res_f4.seg");
    check(m.count(0) && m.count(2) && m[2] - m[0] >= 200,
          "FFMA issue - LDS >= 200", (long long)(m[2] - m[0]), 200);
  }
  {  // F5: F1 through the real program-table path (tree resources ->
     // compiled NODE/LEAF/PARM/USES rows)
    std::printf("F5 resource def via compiled program tables\n");
    auto m = issue_cycles(dir + "/res_f5.seg");
    check(m.count(0) && m.count(1) && m[1] - m[0] >= 8,
          "second MUFU issue - first >= 8", (long long)(m[1] - m[0]), 8);
  }
  {  // F6: alu_forms rule — one-GPR-source VIADD on an SMSP unit with
     // issue_interval 2 and single_source_issue_interval 1: 2 warps x
     // 16 instances = 32 issues on the unit, spaced 1 -> span 31.
    std::printf("F6 single-source interval 1 (srcs=1)\n");
    auto m = issue_cycles(dir + "/res_f6.seg");
    long long span = (m.count(0) && m.count(31))
        ? (long long)(m[31] - m[0]) : -1;
    check(m.size() == 32 && span == 31, "one-source issue span == 31",
          span, 31);
  }
  {  // F6b: same fixture, two distinct GPR sources -> the class
     // interval 2 applies -> span 62.
    std::printf("F6b two-source interval 2\n");
    auto m = issue_cycles(dir + "/res_f6b.seg");
    long long span = (m.count(0) && m.count(31))
        ? (long long)(m[31] - m[0]) : -1;
    check(m.size() == 32 && span == 62, "two-source issue span == 62",
          span, 62);
  }
  {  // F6c: field absent -> byte-identical legacy behaviour, one
     // source still pays the class interval -> span 62.
    std::printf("F6c field absent, interval 2\n");
    auto m = issue_cycles(dir + "/res_f6c.seg");
    long long span = (m.count(0) && m.count(31))
        ? (long long)(m[31] - m[0]) : -1;
    check(m.size() == 32 && span == 62,
          "absent field keeps interval 2 span == 62", span, 62);
  }
  {  // F7: unit_grant_policy=0 — 3 warps x 8 VIADD, unit interval 4;
     // the cursor re-serves warp 0 at every free cycle, so it issues
     // all 8 (cycles 0,4,...,28) before warp 1 (32..60) and warp 2
     // (64..92) get the unit at their turn.
    std::printf("F7 unit_grant_policy=0 warp0 drains first\n");
    auto m = issue_cycles(dir + "/res_f7.seg");
    check(m.count(23) == 1, "24 issues", (long long)m.size(), 24);
    check(m[7] == 28, "warp0 last issue == 28", (long long)m[7], 28);
    check(m[7] < m[8] && m[7] < m[16],
          "warp0 drains before warps 1/2 start", (long long)m[7],
          (long long)std::min(m[8], m[16]));
    check(m[15] == 60 && m[23] == 92,
          "warp1 last == 60, warp2 last == 92",
          (long long)m[15] * 1000 + m[23], 60092);
  }
  {  // F7b: unit_grant_policy=1 — the free unit goes to its longest
     // live waiter and the last grantee loses equal-age ties, so the
     // warps alternate w0@0, w1@4, w2@8, w0@12, ...; last issues
     // 84/88/92 — within two intervals of one another.
    std::printf("F7b unit_grant_policy=1 warps alternate\n");
    auto m = issue_cycles(dir + "/res_f7b.seg");
    check(m.count(23) == 1, "24 issues", (long long)m.size(), 24);
    check(m[7] == 84 && m[15] == 88 && m[23] == 92,
          "last issues 84/88/92",
          (long long)m[7] * 1000000 + m[15] * 1000 + m[23], 84088092);
    const long long hi = std::max(m[7], std::max(m[15], m[23]));
    const long long lo = std::min(m[7], std::min(m[15], m[23]));
    check(hi - lo <= 8, "last issues within 2 intervals",
          hi - lo, 8);
  }
  std::printf("%s\n", failures ? "FAILURES" : "all ok");
  return failures ? 1 : 0;
}
