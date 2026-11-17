// Reads the Python-produced container fixture and checks merged tables.

#include <cstdio>
#include <string>

#include "core/job_reader.hpp"

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  try {
    cyclescope::Job job = cyclescope::read_container_file(argv[1]);
    if (job.ctas.size() != 3 || job.warps.size() != 3) {
      std::fprintf(stderr, "cta/warp counts wrong\n");
      return 1;
    }
    if (job.instances.size() != 6) {
      std::fprintf(stderr, "instance count wrong\n");
      return 1;
    }
    if (job.warps[1].first_instance != 1 ||
        job.warps[2].first_instance != 3) {
      std::fprintf(stderr, "first_instance rebase wrong\n");
      return 1;
    }
    // per-segment STRS / FACT tails: templates name OP0/OP1/OP2, the
    // per-instance USES rows see FACT values 100/101/102 after the shared
    // init row (value 7) kept once
    if (job.templates.size() != 3 || job.facts.size() != 4 ||
        job.facts[0].value != 7 || job.facts[1].value != 100 ||
        job.facts[2].value != 101 || job.facts[3].value != 102) {
      std::fprintf(stderr, "FACT merge wrong (%zu facts)\n",
                   job.facts.size());
      return 1;
    }
    for (int k = 0; k < 3; ++k) {
      const char* op = job.strings.c_str() + job.templates[k].operation_class;
      if (std::string(op) != "OP" + std::to_string(k)) {
        std::fprintf(stderr, "STRS merge wrong: template %d -> %s\n", k, op);
        return 1;
      }
    }
    int per_inst = 0;
    for (const auto& u : job.uses) {
      if (u.target != 0) continue;
      if (u.fact_count != 1 || u.fact_begin != 1u + per_inst ||
          job.facts[u.fact_begin].value != 100u + per_inst) {
        std::fprintf(stderr, "USES fact_begin rebase wrong\n");
        return 1;
      }
      ++per_inst;
    }
    if (per_inst != 3) {
      std::fprintf(stderr, "per-instance USES count %d\n", per_inst);
      return 1;
    }
    // dense CTA ids (segments carried linear ids 7/3/9), WARP follows,
    // PLCE translated and filtered to CTAs present in the job
    for (int k = 0; k < 3; ++k)
      if (job.ctas[k].cta_index != (std::uint32_t)k ||
          job.warps[k].cta_index != (std::uint32_t)k) {
        std::fprintf(stderr, "CTA renumbering wrong\n");
        return 1;
      }
    if (job.placements.size() != 1 || job.placements[0].cta != 1 ||
        job.placements[0].sm_id != 0) {
      std::fprintf(stderr, "PLCE translation wrong (%zu rows)\n",
                   job.placements.size());
      return 1;
    }
    std::printf("job_reader_test ok\n");
  } catch (const cyclescope::FormatError& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
