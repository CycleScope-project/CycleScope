#pragma once
// Merged in-memory job produced by the container/segment reader (spec §3).

#include <string>
#include <vector>

#include "job_format.hpp"

namespace cyclescope {

struct Job {
  LaunchInfo launch{};
  std::vector<PlacementRow> placements;
  std::vector<GlobalSyncRow> global_sync;
  std::vector<SmTopologyRow> topology;

  MetaRow meta{};
  std::vector<CtaInfo> ctas;
  std::vector<WarpInfo> warps;
  std::vector<InstructionTemplate> templates;
  std::vector<InstructionInstance> instances;
  std::vector<RegisterRef> register_refs;
  std::vector<InstanceSemantics> semantics;
  std::vector<MemoryAccess> accesses;
  std::vector<MemorySector> sectors;
  std::vector<TimingRow> timing;
  std::vector<MechanismProgram> programs;
  std::vector<MechanismNode> nodes;
  std::vector<MechanismGuard> guards;
  std::vector<MechanismLeaf> leaves;
  std::vector<ParameterField> fields;
  std::vector<PointEntry> points;
  std::vector<FactRow> facts;
  std::vector<ParameterUse> uses;
  std::vector<AxisRow> axis_names;
  std::vector<SplitArrive> splits;
  std::vector<MbarrierPreinit> mbarrier_preinit;
  std::vector<MbarrierObject> mbarrier_objects;
  // optional MBOB rows, segment-local object order (empty when absent)
  std::vector<MbarrierObjectRow> mbarrier_owners;
  std::vector<CodeRow> code;
  std::vector<ClockMarker> clock_markers;
  std::string strings;

  // segment (cluster) partitioning, filled by merge_parts: every CTA's
  // segment index and per-segment bases for the segment-local tables
  // (mbarrier object index spaces, MBOL lists, instance indices)
  std::vector<std::uint32_t> cta_segment;
  std::vector<std::uint64_t> segment_mbol_base;
  std::vector<std::uint64_t> segment_instance_base;
  std::vector<std::uint64_t> segment_mbob_base;
};

Job read_segment_file(const std::string& path);
Job read_container_file(const std::string& path);
Job read_segment_file_data(const BinaryFile& file,
                           const BinaryFile* prog = nullptr,
                           bool program_tables = true);
void merge_parts(Job& job, std::vector<Job> parts);
std::vector<std::string> validate_job(const Job& job);

}  // namespace cyclescope
