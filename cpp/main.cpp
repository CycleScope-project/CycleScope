// cyclescope-engine — step 2a: job file inspection and structural
// validation; step 3-1: simulate.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "core/job_format.hpp"
#include "core/job_reader.hpp"
#include "core/mechanism_runtime.hpp"
#include "core/result_writer.hpp"
#include "core/simulator.hpp"
#include "arch/architecture_module.hpp"

namespace {

int usage();
std::uint64_t fnv(const void* p, std::size_t n) {
  std::uint64_t h = 1469598103934665603ull;
  const auto* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
template <typename V>
void dump_vec(const char* name, const V& v) {
  std::printf("%-22s %8zu %016llx\n", name, v.size(),
              (unsigned long long)fnv(v.data(), v.size() * sizeof(*v.data())));
}
// --warp N: one line per instance of warp N (index, code_line, class,
// semantics kind/flags/fields) — diagnostic listing
void dump_warp(const cyclescope::Job& job, std::uint32_t warp) {
  if (warp >= job.warps.size()) {
    std::fprintf(stderr, "warp %u out of range\n", warp);
    return;
  }
  const auto& wi = job.warps[warp];
  std::printf("warp %u cta %u warp_in_cta %u instances %llu..%llu\n", warp,
              wi.cta_index, wi.warp_in_cta,
              (unsigned long long)wi.first_instance,
              (unsigned long long)(wi.first_instance + wi.instance_count));
  for (std::uint64_t i = wi.first_instance;
       i < wi.first_instance + wi.instance_count; ++i) {
    const auto& in = job.instances[i];
    const auto& t = job.templates[in.template_index];
    std::printf("%llu line %u %s", (unsigned long long)i, in.code_line,
                job.strings.c_str() + t.operation_class);
    if (in.semantics != 0xFFFFFFFFu) {
      const auto& s = job.semantics[in.semantics];
      std::printf(" | sema kind %u flags 0x%x f", s.kind, s.flags);
      for (int k = 0; k < 6; ++k)
        std::printf(" %llu", (unsigned long long)s.field[k]);
    }
    std::printf("\n");
  }
}
int dump(int argc, char** argv) {
  std::string job_path;
  long warp = -1;
  for (int i = 0; i < argc; ++i) {
    if (std::string(argv[i]) == "--job" && i + 1 < argc)
      job_path = argv[++i];
    else if (std::string(argv[i]) == "--warp" && i + 1 < argc)
      warp = std::strtol(argv[++i], nullptr, 0);
  }
  if (job_path.empty()) return usage();
  cyclescope::Job job = cyclescope::BinaryFile::peek_kind(job_path) == 0
      ? cyclescope::read_segment_file(job_path)
      : cyclescope::read_container_file(job_path);
  if (warp >= 0) {
    dump_warp(job, static_cast<std::uint32_t>(warp));
    return 0;
  }
  dump_vec("placements", job.placements);
  dump_vec("global_sync", job.global_sync);
  dump_vec("topology", job.topology);
  std::printf("%-22s %8d %016llx\n", "meta", 1,
              (unsigned long long)fnv(&job.meta, sizeof job.meta));
  std::printf("%-22s %8d %016llx\n", "launch", 1,
              (unsigned long long)fnv(&job.launch, sizeof job.launch));
  dump_vec("ctas", job.ctas);
  dump_vec("warps", job.warps);
  dump_vec("templates", job.templates);
  dump_vec("instances", job.instances);
  dump_vec("register_refs", job.register_refs);
  dump_vec("semantics", job.semantics);
  dump_vec("accesses", job.accesses);
  dump_vec("sectors", job.sectors);
  dump_vec("timing", job.timing);
  dump_vec("programs", job.programs);
  dump_vec("nodes", job.nodes);
  dump_vec("guards", job.guards);
  dump_vec("leaves", job.leaves);
  dump_vec("fields", job.fields);
  dump_vec("points", job.points);
  dump_vec("facts", job.facts);
  dump_vec("uses", job.uses);
  dump_vec("axis_names", job.axis_names);
  dump_vec("splits", job.splits);
  dump_vec("mbarrier_preinit", job.mbarrier_preinit);
  dump_vec("mbarrier_objects", job.mbarrier_objects);
  dump_vec("mbarrier_owners", job.mbarrier_owners);
  dump_vec("code", job.code);
  dump_vec("clock_markers", job.clock_markers);
  std::printf("%-22s %8zu %016llx\n", "strings", job.strings.size(),
              (unsigned long long)fnv(job.strings.data(), job.strings.size()));
  dump_vec("cta_segment", job.cta_segment);
  dump_vec("segment_mbol_base", job.segment_mbol_base);
  dump_vec("segment_instance_base", job.segment_instance_base);
  dump_vec("segment_mbob_base", job.segment_mbob_base);
  return 0;
}

int usage() {
  std::fprintf(stderr,
               "usage: cyclescope-engine <info|validate> --job <file>\n"
               "       cyclescope-engine simulate --job <job.bin> --out "
               "<result.bin> [--journal f] [--wait-ledger f] "
               "[--stall-ledger f] "
               "[--telemetry off] [--progress]\n");
  return 2;
}

int simulate(int argc, char** argv) {
  std::string job_path, out_path, journal_path, ledger_path;
  cyclescope::Simulator::Options opt;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--job" && i + 1 < argc) job_path = argv[++i];
    else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (a == "--journal" && i + 1 < argc)
      journal_path = argv[++i], opt.journal = true;
    else if (a == "--wait-ledger" && i + 1 < argc)
      ledger_path = argv[++i], opt.wait_ledger = true;
    else if (a == "--stall-ledger" && i + 1 < argc)
      opt.stall_ledger = argv[++i];
    else if (a == "--telemetry" && i + 1 < argc) { ++i; }
    else if (a == "--progress") opt.progress = true;
  }
  if (job_path.empty() || out_path.empty()) return usage();
  const auto t0 = std::chrono::steady_clock::now();
  // header-only kind probe: the full file is read once, by the loader
  cyclescope::Job job = cyclescope::BinaryFile::peek_kind(job_path) == 0
      ? cyclescope::read_segment_file(job_path)
      : cyclescope::read_container_file(job_path);
  const auto t1 = std::chrono::steady_clock::now();
  cyclescope::MechanismRuntime::Options ropt;
  ropt.reference_sm_clock_khz = job.launch.reference_sm_clock_khz;
  cyclescope::MechanismRuntime rt(job, job.launch.device, ropt);
  cyclescope::Simulator::Options sopt = opt;
  cyclescope::Simulator sim(job, rt, sopt);
  cyclescope::ResultStats stats;
  try {
    stats = sim.run();
  } catch (const cyclescope::EngineError&) {
    // journal rows collected so far stay available on a deadlock exit
    if (opt.journal)
      cyclescope::write_journal(sim.partial_stats().journal, journal_path);
    throw;
  }
  const auto t2 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "LOAD=%.3f RUN=%.3f\n",
      std::chrono::duration<double>(t1 - t0).count(),
      std::chrono::duration<double>(t2 - t1).count());
  cyclescope::SimulationResult res;
  res.cycles = stats.cycles;
  res.issued = stats.issued;
  res.completed = stats.completed;
  res.hard_constraint_failures = stats.hard_failures;
  res.smsp_spans = stats.smsp_spans;
  for (auto [b, c] : stats.wait)
    res.wait_buckets.push_back({0u, b, c});
  for (auto [r, c] : stats.rwait)
    res.resource_waits.push_back({r, 0u, c});
  for (auto& t : stats.tele)
    res.telemetry.push_back(t);
  cyclescope::write_result(res, out_path);
  if (opt.journal)
    cyclescope::write_journal(stats.journal, journal_path);
  if (opt.wait_ledger)
    cyclescope::write_wait_ledger(stats.wait_ledger, ledger_path);
  std::printf("cycles=%llu issued=%llu completed=%llu hard=%llu\n",
              (unsigned long long)stats.cycles,
              (unsigned long long)stats.issued,
              (unsigned long long)stats.completed,
              (unsigned long long)stats.hard_failures);
  return stats.hard_failures ? 1 : 0;
}

int info(const std::string& path) {
  const auto kind = cyclescope::BinaryFile::peek_kind(path);
  static const char* kinds[] = {"segment", "container", "result",
                                "journal", "wait_ledger", "programs"};
  cyclescope::Job job = kind == 0 ? cyclescope::read_segment_file(path)
                              : cyclescope::read_container_file(path);
  std::printf("kind: %s\n", kinds[kind]);
  std::printf("ctas: %zu  warps: %zu  instances: %zu  templates: %zu\n",
              job.ctas.size(), job.warps.size(), job.instances.size(),
              job.templates.size());
  std::printf("programs: %zu  nodes: %zu  leaves: %zu  uses: %zu\n",
              job.programs.size(), job.nodes.size(), job.leaves.size(),
              job.uses.size());
  return 0;
}

int validate(const std::string& path) {
  cyclescope::Job job = cyclescope::BinaryFile::peek_kind(path) == 0
      ? cyclescope::read_segment_file(path)
      : cyclescope::read_container_file(path);
  auto errors = cyclescope::validate_job(job);
  for (const auto& e : errors) std::fprintf(stderr, "error: %s\n", e.c_str());
  if (errors.empty()) {
    std::printf("ok\n");
    return 0;
  }
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && !std::strcmp(argv[1], "simulate")) {
    try {
      return simulate(argc - 2, argv + 2);
    } catch (const cyclescope::FormatError& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 2;
    } catch (const cyclescope::EngineError& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
  if (argc < 4 || std::strcmp(argv[2], "--job")) return usage();
  try {
    if (!std::strcmp(argv[1], "info")) return info(argv[3]);
    if (!std::strcmp(argv[1], "validate")) return validate(argv[3]);
    if (!std::strcmp(argv[1], "dump"))
      return dump(argc - 1, argv + 1);
  } catch (const cyclescope::FormatError& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return usage();
}
