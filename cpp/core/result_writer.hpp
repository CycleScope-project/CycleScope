#pragma once
// Result-file writers: GSIMRES / GSIMJNL / GSIMWLG (spec §8).

#include <cstdint>
#include <string>
#include <vector>

#include "job_format.hpp"

namespace cyclescope {

#pragma pack(push, 1)
struct SmspSpanRow {
  std::uint32_t sm, smsp;
  std::uint64_t first_active, last_active;
};
struct WaitBucketRow {
  std::uint32_t scope, cause_kind;
  std::uint64_t cycles;
};
struct ResourceWaitRow {
  std::uint32_t resource, cause_kind;
  std::uint64_t cycles;
};
struct TelemetryRow {
  std::uint32_t group, counter;
  std::uint64_t value;
};
struct JournalRow {
  std::uint32_t warp, smsp, instance;
  std::uint64_t issue_cycle;
  std::uint32_t operation_class, template_index;
  // wait attribution of this issue: bucket of the last failed
  // readiness check (kNone32 = no wait), stall window length in
  // cycles (issue - stall_since), and the barrier/mbarrier object
  // number of sync semantics (kNone64 = not a sync op)
  std::uint32_t wait_bucket;
  std::uint64_t wait_cycles;
  std::uint64_t object;
};
struct WaitLedgerRow {
  std::uint32_t warp, cause_kind;
  std::uint64_t cycles;
  std::uint32_t count;
};
#pragma pack(pop)

struct SimulationResult {
  std::uint64_t cycles = 0, issued = 0, completed = 0,
      hard_constraint_failures = 0;
  std::uint32_t flags = 0;
  std::vector<SmspSpanRow> smsp_spans;
  std::vector<WaitBucketRow> wait_buckets;
  std::vector<ResourceWaitRow> resource_waits;
  std::vector<TelemetryRow> telemetry;
};

void write_result(const SimulationResult& result, const std::string& path);
void write_journal(const std::vector<JournalRow>& journal,
                   const std::string& path);
void write_wait_ledger(const std::vector<WaitLedgerRow>& ledger,
                       const std::string& path);

}  // namespace cyclescope
