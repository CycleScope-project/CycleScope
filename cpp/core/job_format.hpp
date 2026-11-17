#pragma once
// Binary job format records (docs/en/design/job-format.md).
// Layouts are packed little-endian PODs; readers memcpy them directly.

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cyclescope {

constexpr std::uint32_t kFormatVersion = 1;
// segment/container v2: shared program tables stored once per
// segments dir (program.tables) or once at container level
constexpr std::uint32_t kSplitProgramVersion = 2;
constexpr std::size_t kHeaderBytes = 72;  // spec lists fields summing to 72
constexpr std::size_t kDirectoryEntryBytes = 32;
constexpr std::uint32_t kNone32 = 0xFFFFFFFFu;
constexpr std::uint64_t kNone64 = 0xFFFFFFFFFFFFFFFFull;

#pragma pack(push, 1)
struct FileHeader {
  char magic[8];
  std::uint32_t format_version;
  std::uint32_t header_bytes;
  std::uint64_t file_bytes;
  std::uint64_t directory_offset;
  std::uint32_t table_count;
  std::uint32_t flags;
  std::uint8_t payload_sha256[32];
};
struct TableEntry {
  std::uint32_t tag;
  std::uint32_t record_bytes;
  std::uint64_t record_count;
  std::uint64_t offset;
  std::uint64_t reserved;
};

// ---- segment tables (§2) ----
struct MetaRow {
  std::uint8_t tree_digest[32], program_digest[32], converter_digest[32];
  std::uint32_t cluster_id, cta_count, warp_count;
  std::uint64_t instruction_count;
  std::uint32_t flags;
};
struct CtaInfo {
  std::uint32_t cta_index, cta_x, cta_y, cta_z, cluster_rank;
  std::uint32_t cluster_shape_x, cluster_shape_y, cluster_shape_z;
  std::uint32_t sm_id, peer_sm_id, multicast_mask;
  std::uint8_t cta_group, is_pair_leader;
  std::uint16_t tmem_columns;
  std::uint32_t first_warp, warp_count;
};
struct WarpInfo {
  std::uint32_t cta_index, warp_in_cta, warp_slot, smsp;
  std::uint64_t first_instance, instance_count;
};
struct InstructionTemplate {
  std::uint32_t operation_class, flags, resource, issue_group_resource;
  std::uint64_t source_begin;
  std::uint32_t source_count;
  std::uint64_t destination_begin;
  std::uint32_t destination_count;
  std::uint8_t scoreboard_read, scoreboard_write, scoreboard_wait_mask,
      control;  // bit0-3 stall, bit4 yield
  std::uint64_t timing_begin;
  std::uint32_t timing_count, semantic_kind;
  std::uint64_t semantic_u64;
};
struct InstructionInstance {
  std::uint32_t template_index, semantics, memory_access, code_line;
};
struct RegisterRef { std::uint32_t reg; };
// REGS row: bits 0-23 register namespace id, bits 24-31 flags
// (bit0 reuse_served, bit1 64-bit pair; destinations carry 0)
inline std::uint32_t reg_id(const RegisterRef& r) {
  return r.reg & 0x00FFFFFFu;
}
inline std::uint32_t reg_flags(const RegisterRef& r) {
  return r.reg >> 24;
}
struct InstanceSemantics {
  std::uint16_t kind, flags;
  std::uint64_t earliest_issue;
  std::uint64_t field[6];
};
struct MemoryAccess {
  std::uint64_t sector_begin;
  std::uint32_t sector_count, flags;
  std::uint64_t sequence, shape;
  std::uint64_t timing_begin;
  std::uint32_t timing_count, cache_hint;
};
struct MemorySector {
  std::uint64_t address;
  std::uint32_t byte_mask, lane_mask;
};
struct TimingRow {
  std::uint32_t kind, selector_a, selector_b, ready_warps;
  std::uint64_t selector_c;
  double cycles;
  std::uint8_t enabled, pad[7];
};
struct MechanismProgram {
  std::uint32_t mechanism;
  std::uint8_t event;
  std::uint16_t schema_id;
  std::uint8_t timing_kind;
  std::uint32_t root_node;
  std::uint64_t field_mask, operation_class_begin;
  std::uint32_t operation_class_count;
};
struct MechanismNode {
  std::uint32_t parent, child_begin, child_count, guard_begin,
      guard_count, leaf;
};
struct MechanismGuard {
  std::uint32_t axis;
  std::uint8_t op, pad[3];
  std::uint64_t low, high;
};
struct MechanismLeaf {
  std::uint32_t parameter;
  std::uint8_t status, pad[3];
};
struct ParameterField {
  std::uint16_t schema_id, field_id;
  std::uint8_t model_kind, pad[3];
  std::uint32_t axis;
  double value, slope, intercept;
  std::uint64_t points_begin;
  std::uint32_t points_count;
};
struct PointEntry { double axis_value, value; };
struct FactRow { std::uint32_t axis; std::uint64_t value; };
struct AxisRow { std::uint32_t name; };
struct ParameterUse {
  std::uint32_t node;
  std::uint8_t target, pad[3];
  std::uint64_t target_index;
  std::uint16_t schema_id;
  std::uint64_t field_mask, fact_begin;
  std::uint32_t fact_count;
};
struct SplitArrive {
  std::uint32_t semantics, rank, credit, pad;
};
struct MbarrierPreinit {
  std::uint32_t cta_index, object;
  std::uint64_t address;
  std::uint32_t arrival_count, pad;
};
struct MbarrierObject { std::uint32_t object; };
// optional MBOB table: one row per segment-local mbarrier object,
// naming its owning CTA and its CTA-local shared address (used to
// resolve cross-CTA destinations within a cluster segment)
struct MbarrierObjectRow {
  std::uint32_t cta_index;
  std::uint64_t address;
};
struct CodeRow {
  std::uint32_t instance, pad;
  std::uint64_t code_address;
};
struct ClockMarker {
  std::uint32_t warp, kind;
  std::uint64_t instance;
};

// ---- container tables (§3) ----
struct LaunchInfo {
  std::uint32_t grid_x, grid_y, grid_z, block_x, block_y, block_z,
      cluster_x, cluster_y, cluster_z;
  std::uint32_t shared_static_bytes, shared_dynamic_bytes,
      registers_per_thread, sm_count;
  std::uint64_t l2_bytes;
  std::uint32_t occupancy_blocks_per_sm, occupancy_active_clusters,
      function_prior_launches,
      reference_sm_clock_khz;
  std::uint8_t reference_cache_control, device_facts_present;
  std::uint16_t pad;
  std::uint32_t flags, native_ctas_launched;
  char device[16];
};
struct SegmentDirectoryRow {
  std::uint32_t segment, cluster_id;
  std::uint64_t byte_offset, byte_length;
  std::uint32_t cta_base, cta_count;
  std::uint8_t segment_sha256[32];
};
struct PlacementRow {
  std::uint32_t cta, segment, cta_in_segment;
  std::uint8_t placement, pad[3];
  std::uint32_t sm_id, warp_slot_base;
  std::uint8_t source, pad2[3];
};
struct GlobalSyncRow {
  std::uint32_t segment, instance;
  std::uint8_t lane, kind, class_, flags;
  std::uint64_t address, after, before, sequence;
};
struct SmTopologyRow {
  std::uint32_t sm, placement_domain, l2_partition, tpc_pair, gpc, pad;
};

// ---- result (§8) ----
struct ResultRow {
  std::uint64_t cycles, issued, completed, hard_constraint_failures;
  std::uint32_t flags;
};
#pragma pack(pop)

constexpr std::uint32_t tag(const char t[5]) {
  return std::uint32_t(t[0]) | (std::uint32_t(t[1]) << 8) |
         (std::uint32_t(t[2]) << 16) | (std::uint32_t(t[3]) << 24);
}

struct Table {
  std::uint32_t record_bytes = 0;
  std::uint64_t record_count = 0;
  std::uint64_t offset = 0;
};

class FormatError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Directory-driven view of one binary file.
class BinaryFile {
 public:
  struct InMemory {};
  struct View {};
  explicit BinaryFile(const std::string& path);
  BinaryFile(std::string data, InMemory);
  // non-owning view over bytes kept alive by the caller (a segment
  // image inside a loaded container): no copy of the image
  BinaryFile(std::string_view data, View);
  void init(std::string data);
  void init_view(std::string_view data);
  // file kind from the header only (no full read): 0 segment,
  // 1 container, 2 result, 3 journal, 4 wait ledger, 5 programs
  static std::uint8_t peek_kind(const std::string& path);
  const Table* table(std::uint32_t tag_value) const;
  const Table& require(std::uint32_t tag_value) const;
  std::uint32_t format_version() const { return version_; }
  template <typename R>
  std::vector<R> rows(std::uint32_t tag_value) const {
    const Table& t = require(tag_value);
    if (t.record_bytes != sizeof(R))
      throw FormatError("record size mismatch for table");
    std::vector<R> out(t.record_count);
    if (t.record_count)
      std::memcpy(out.data(), view_data_.data() + t.offset,
                  t.record_count * sizeof(R));
    return out;
  }
  std::string blob(std::uint32_t tag_value) const {
    const Table& t = require(tag_value);
    std::size_t n = t.record_bytes ? t.record_bytes * t.record_count
                                   : t.record_count;
    return std::string(view_data_.substr(t.offset, n));
  }
  std::string_view data() const { return view_data_; }
  std::uint8_t kind() const { return kind_; }

 private:
  std::string data_;            // owned bytes (empty in view mode)
  std::string_view view_data_;  // the bytes in use (owned or external)
  std::vector<TableEntry> entries_;
  mutable std::optional<Table> view_;
  std::uint8_t kind_ = 0;
  std::uint32_t version_ = 0;
};

}  // namespace cyclescope
