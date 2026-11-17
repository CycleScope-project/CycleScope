// arch_blackwell_test — BlackwellModule fixture assertions (§14.3).

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
  std::map<std::pair<std::uint32_t, std::uint64_t>, std::uint64_t>
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

std::uint64_t at(const Run& r, std::uint32_t w,
                 std::uint64_t inst_offset) {
  const std::uint64_t gi =
      r.job.warps[w].first_instance + inst_offset;
  auto it = r.issue.find({w, gi});
  if (it == r.issue.end()) {
    std::fprintf(stderr, "missing issue warp %u inst %llu\n", w,
                 (unsigned long long)gi);
    std::exit(1);
  }
  return it->second;
}

#define CHECK(cond, ...)                                  \
  do {                                                    \
    if (!(cond)) {                                        \
      std::fprintf(stderr, "FAIL %s: ", #cond);           \
      std::fprintf(stderr, __VA_ARGS__);                  \
      std::fprintf(stderr, "\n");                         \
      ++fail;                                             \
    }                                                     \
  } while (0)

void expect_engine_error(int& fail, const std::string& path,
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

}  // namespace

int main(int argc, char** argv) {
  int fail = 0;
  const std::string dir = argv[1];

  {  // B1: UMMA pipe 0/20/40 + UTCBAR arrive at 65
    auto r = run(dir + "/b1_umma.job");
    // UMMA issues are not delayed by the pipe (pipe only moves the
    // completion beat); the release of the object happens at
    // max(completion)=60 + kind51 5 = 65 -> waiter fill at >= 66
    CHECK(at(r, 1, 1) >= 66, "waiter fill at %llu want >= 66",
          (unsigned long long)at(r, 1, 1));
  }

  {  // B2: second CTA's alloc blocked until first deallocs
    auto r = run(dir + "/b2_tmem.job");
    const std::uint64_t dealloc = at(r, 0, 11);
    CHECK(at(r, 1, 0) >= dealloc,
          "cta2 alloc at %llu want >= dealloc %llu",
          (unsigned long long)at(r, 1, 0),
          (unsigned long long)dealloc);
  }

  {  // B2b: a dealloc on an SM without a prior alloc leaves the budget
     // intact -> the second CTA's 512-column alloc issues at once
    auto r = run(dir + "/b2b_tmem_dealloc_first.job");
    CHECK(r.stats.issued == r.stats.completed && r.stats.cycles < 200,
          "cycles %llu issued %llu completed %llu want < 200",
          (unsigned long long)r.stats.cycles,
          (unsigned long long)r.stats.issued,
          (unsigned long long)r.stats.completed);
  }

  {  // B3: CLC completes 16 B at clc + 8
    auto r = run(dir + "/b3_clc.job");
    const std::uint64_t clc = at(r, 0, 2);
    CHECK(at(r, 0, 5) >= clc + 8,
          "post-wait fill at %llu want >= %llu",
          (unsigned long long)at(r, 0, 4),
          (unsigned long long)(clc + 8));
  }

  {  // B4: STAS completes 32 B at stas-finish + kind-21 (6)
    auto r = run(dir + "/b4_stas.job");
    const std::uint64_t stas = at(r, 0, 3);
    CHECK(at(r, 0, 5) >= stas + 7,
          "post-wait fill at %llu want >= %llu",
          (unsigned long long)at(r, 0, 5),
          (unsigned long long)(stas + 7));
  }

  {
    auto r = run(dir + "/b5_copy_commit.job");
    const std::uint64_t delta = at(r, 1, 1) - at(r, 0, 1);
    CHECK(delta >= 46 && delta <= 50,
          "copy->waiter delta %llu want [46,50]",
          (unsigned long long)delta);
  }

  {
    auto r = run(dir + "/b6_copy_mma.job");
    const std::uint64_t delta = at(r, 1, 1) - at(r, 0, 1);
    CHECK(delta >= 73 && delta <= 77,
          "copy+mma->waiter delta %llu want [73,77]",
          (unsigned long long)delta);
  }

  {
    auto r = run(dir + "/b7_mma_copy.job");
    const std::uint64_t delta = at(r, 1, 1) - at(r, 0, 1);
    CHECK(delta >= 66 && delta <= 70,
          "mma+copy->waiter delta %llu want [66,70]",
          (unsigned long long)delta);
  }

  {
    auto r = run(dir + "/b8_fractional_copy.job");
    const std::uint64_t delta = at(r, 1, 1) - at(r, 0, 1);
    CHECK(delta >= 31 && delta <= 34,
          "10x2.5 copy delta %llu want [31,34]",
          (unsigned long long)delta);
  }

  {
    auto r = run(dir + "/b9_foreign_commit.job");
    const std::uint64_t delta = at(r, 2, 1) - at(r, 1, 1);
    CHECK(delta >= 86 && delta < 100,
          "foreign commit delta %llu want [86,100)",
          (unsigned long long)delta);
  }

  {  // B10: A in TMEM -> operand_read 32 of beat 64; the copy hides
     // in the slack so MMA2 starts at the beat frontier (commit
     // releases the waiter ~141 after the first MMA's issue; the old
     // serial rule would land ~150)
    auto r = run(dir + "/b10_operand_tmem.job");
    const std::uint64_t delta = at(r, 1, 1) - at(r, 0, 1);
    CHECK(delta >= 139 && delta <= 144,
          "tmem-A operand path delta %llu want [139,144]",
          (unsigned long long)delta);
  }

  {  // B11: A in shared memory -> operand_read 64 == beat: serial
    auto r = run(dir + "/b11_operand_smem.job");
    const std::uint64_t delta = at(r, 1, 1) - at(r, 0, 1);
    CHECK(delta >= 147 && delta <= 152,
          "smem-A serial delta %llu want [147,152]",
          (unsigned long long)delta);
  }

  {  // B12: cta_group 2, m256 n256, A smem -> operand_read 64 of a
     // 128 beat; MMA2 still starts on the beat frontier
    auto r = run(dir + "/b12_operand_cg2.job");
    const std::uint64_t delta = at(r, 1, 1) - at(r, 0, 1);
    CHECK(delta >= 266 && delta <= 272,
          "cg2 operand path delta %llu want [266,272]",
          (unsigned long long)delta);
  }

  {  // B13: rate present but no tensor_k fact -> refuse
    expect_engine_error(fail, dir + "/b13_operand_no_k.job",
                        "missing_fact:axis:tensor_k");
  }

  {  // B14/B15: a fractional beat must cap the operand read at the
     // truncated integer beat — the rate leaf and the absent field
     // must release the waiter on the same cycle
    auto r14 = run(dir + "/b14_beat_fraction_rate.job");
    auto r15 = run(dir + "/b15_beat_fraction_nofield.job");
    CHECK(at(r14, 1, 1) == 93,
          "fractional beat release %llu want 93",
          (unsigned long long)at(r14, 1, 1));
    CHECK(at(r14, 1, 1) == at(r15, 1, 1),
          "fractional beat rate-vs-absent release %llu != %llu",
          (unsigned long long)at(r14, 1, 1),
          (unsigned long long)at(r15, 1, 1));
  }

  {
    Run r;
    r.job = cyclescope::read_container_file(dir + "/b1_umma.job");
    MechanismRuntime::Options ro;
    ro.reference_sm_clock_khz = r.job.launch.reference_sm_clock_khz;
    MechanismRuntime rt(r.job, r.job.launch.device, ro);
    Simulator::Options so;
    Simulator sim(r.job, rt, so);
    // two frontiers: an MMA holds the operand path only for
    // operand_read of its beat; copies hide in the remaining slack
    CHECK(sim.tensor_mma_start(0, 0, 64, 32.0) == 0,
          "sm0 mma start");
    CHECK(sim.tensor_operand_path_finish(0, 0, 8.5) == 41,
          "sm0 copy1 want 41");
    CHECK(sim.tensor_operand_path_finish(0, 0, 8.5) == 49,
          "sm0 copy2 want 49");
    CHECK(sim.tensor_mma_start(0, 0, 64, 32.0) == 64,
          "sm0 mma2 start want 64");
    CHECK(sim.tensor_operand_path_finish(0, 0, 8.5) == 105,
          "sm0 copy after mma2 want 105");
    CHECK(sim.tensor_pipe_start(0, 0, 0) == 128,
          "sm0 pipe frontier want 128");
    // operand_read == beat reproduces the old serial rule
    CHECK(sim.tensor_mma_start(1, 0, 64, 64.0) == 0,
          "sm1 mma start");
    CHECK(sim.tensor_operand_path_finish(1, 0, 8.5) == 73,
          "sm1 copy want 73");
    CHECK(sim.tensor_mma_start(1, 0, 64, 64.0) == 73,
          "sm1 mma2 start want 73");
    // copies that outrun the slack push the next MMA start
    CHECK(sim.tensor_mma_start(2, 0, 64, 32.0) == 0,
          "sm2 mma start");
    for (int i = 0; i < 4; ++i)
      sim.tensor_operand_path_finish(2, 0, 8.5);
    CHECK(sim.tensor_operand_path_finish(2, 0, 8.5) == 75,
          "sm2 copy5 want 75");
    CHECK(sim.tensor_mma_start(2, 0, 64, 32.0) == 75,
          "sm2 mma2 start want 75");
  }

  std::fprintf(stderr, "arch_blackwell: %d failures\n", fail);
  return fail ? 1 : 0;
}
