// Result/journal/wait-ledger writers (spec §8). The engine fills
// SimulationResult in step 3; writers only serialize.

#include "result_writer.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace cyclescope {

namespace {

std::uint64_t align8(std::uint64_t v) { return (v + 7) / 8 * 8; }

struct TableSpec {
  std::uint32_t tag;
  std::uint32_t record_bytes;
  const void* data;
  std::uint64_t count;
};

}  // namespace

static void write_packed(const char magic[8],
                         std::vector<TableSpec> specs,
                         const std::string& path) {
  std::uint64_t dir_end =
      align8(kHeaderBytes + specs.size() * kDirectoryEntryBytes);
  FileHeader h{};
  std::memcpy(h.magic, magic, 8);
  h.format_version = kFormatVersion;
  h.header_bytes = kHeaderBytes;
  h.directory_offset = kHeaderBytes;
  h.table_count = static_cast<std::uint32_t>(specs.size());
  std::vector<TableEntry> entries;
  std::uint64_t off = dir_end;
  for (const auto& s : specs) {
    TableEntry e{};
    e.tag = s.tag;
    e.record_bytes = s.record_bytes;
    e.record_count = s.count;
    e.offset = off;
    off = align8(off + s.count * s.record_bytes);
    entries.push_back(e);
  }
  h.file_bytes = off;
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&h), sizeof h);
  for (const auto& e : entries)
    out.write(reinterpret_cast<const char*>(&e), sizeof e);
  std::uint64_t pos = dir_end;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    std::uint64_t want = entries[i].offset;
    while (pos < want) { out.put(0); ++pos; }
    out.write(reinterpret_cast<const char*>(specs[i].data),
              specs[i].count * specs[i].record_bytes);
    pos += specs[i].count * specs[i].record_bytes;
    // body rows are 8-byte aligned like the directory promised —
    // file_bytes includes this trailing pad
    while (pos < align8(pos)) { out.put(0); ++pos; }
  }
}

void write_result(const SimulationResult& result,
                  const std::string& path) {
  std::vector<ResultRow> rslt(1);
  rslt[0].cycles = result.cycles;
  rslt[0].issued = result.issued;
  rslt[0].completed = result.completed;
  rslt[0].hard_constraint_failures = result.hard_constraint_failures;
  rslt[0].flags = result.flags;
  write_packed("GSIMRES", {
      {tag("RSLT"), sizeof(ResultRow), rslt.data(), rslt.size()},
      {tag("SMSP"), sizeof(SmspSpanRow), result.smsp_spans.data(),
       result.smsp_spans.size()},
      {tag("WAIT"), sizeof(WaitBucketRow), result.wait_buckets.data(),
       result.wait_buckets.size()},
      {tag("RWAT"), sizeof(ResourceWaitRow), result.resource_waits.data(),
       result.resource_waits.size()},
      {tag("TELE"), sizeof(TelemetryRow), result.telemetry.data(),
       result.telemetry.size()}}, path);
}

void write_journal(const std::vector<JournalRow>& journal,
                   const std::string& path) {
  write_packed("GSIMJNL", {
      {tag("JRNL"), sizeof(JournalRow), journal.data(), journal.size()}},
      path);
}

void write_wait_ledger(const std::vector<WaitLedgerRow>& ledger,
                       const std::string& path) {
  write_packed("GSIMWLG", {
      {tag("WLDG"), sizeof(WaitLedgerRow), ledger.data(), ledger.size()}},
      path);
}

}  // namespace cyclescope
