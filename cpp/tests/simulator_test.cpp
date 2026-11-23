// Simulator: mini job runs to completion; issued == completed.
#include <cassert>
#include <cstdio>
#include "core/job_reader.hpp"
#include "core/mechanism_runtime.hpp"
#include "core/simulator.hpp"
#include "arch/architecture_module.hpp"

// fixture engine.seg: template 0 bound by USES row 3 (target 0/index 0
// -> node 1 leaf: result_latency=3, completion affine 2*x+1@x=1? axis
// value 0 -> intercept 1; issue_interval linear pts: x=0 -> 10).
int main(int argc, char** argv) {
  auto job = cyclescope::read_segment_file(argv[1]);
  cyclescope::MechanismRuntime::Options ro;
  cyclescope::MechanismRuntime rt(job, "h100", ro);
  cyclescope::Simulator::Options so;
  cyclescope::Simulator sim(job, rt, so);
  auto st = sim.run();
  assert(st.issued == st.completed);
  assert(st.issued > 0);
  assert(st.hard_failures == 0);
  // S1 (kernel_boundary K2): cycles is the busiest SM's active-span
  // union — the fixture's single warp is resident only until its
  // done cycle, so the 99-cycle completion no longer extends
  // cycles; the last completion stays in telemetry group 2
  // last_completion_cycle.
  assert(st.cycles >= 1);
  bool last_completion_99 = false;
  for (const auto& t : st.tele)
    if (t.group == 2 && t.counter == cyclescope::kTele2LastCompletionCycle &&
        t.value >= 99)
      last_completion_99 = true;
  assert(last_completion_99);
  std::printf("ok issued=%llu cycles=%llu\n",
              (unsigned long long)st.issued,
              (unsigned long long)st.cycles);
  return 0;
}
