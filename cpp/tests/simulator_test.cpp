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
  // S1: completed arrives at the completion event; cycles must reach
  // the last completion (fixture completion latency = 99).
  assert(st.cycles >= 99);
  std::printf("ok issued=%llu cycles=%llu\n",
              (unsigned long long)st.issued,
              (unsigned long long)st.cycles);
  return 0;
}
