// DynRecord.ops -> SemanticOperands (operand_facts.py), plus the arch
// capture rules (arch/<arch>/capture_rules.py) as a per-arch table.
#pragma once
#include "capture/static_code.hpp"
#include "capture/trace_reader.hpp"
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace cyclescope_convert::capture {

struct SemanticOperands {
  std::map<std::string, std::int64_t> uniform;   // insertion order kept
  std::vector<std::string> uniform_order;
  // lane values: key is a register name or (ranks) negative-encoded int
  std::map<std::string, std::vector<std::int64_t>> lane;
  std::vector<std::string> lane_order;
  std::map<int, std::vector<std::int64_t>> ranks;
  std::vector<int> ranks_order;
  std::optional<std::uint32_t> after_ready_mask;
  std::map<std::string, std::string> str_uniform;  // str-valued (ukv type0)
  std::optional<std::uint32_t> tensor_map;
  std::vector<std::int64_t> tensor_coordinates;
  std::vector<std::int64_t> descriptor_words;
};

// arch capture rules (capture_rules.py per-arch table)
struct CaptureRules {
  std::set<std::string> mbarrier_companion, mbarrier_self,
      mbarrier_bracket;
  // owner_rank_entry / mbarrier_ur_word / tma_coordinate_offset —
  // implemented per arch in cpp/arch/<arch>/capture_rules.cpp
  std::optional<int> (*owner_rank_entry)(
      const std::map<std::string, std::vector<std::int64_t>> &ops,
      const std::string &operands_text) = nullptr;
  std::optional<int> (*mbarrier_ur_word)(
      const std::map<std::string, std::vector<std::int64_t>> &ops,
      const std::string &operands_text, const std::string &root) = nullptr;
  int (*tma_coordinate_offset)(const std::string &root,
                               int bracket_count) = nullptr;
};
const CaptureRules &capture_rules_for(const std::string &arch);
const CaptureRules &capture_rules_hopper();
const CaptureRules &capture_rules_blackwell();

// {mref_index: per-lane ranks} for RANK/BRANK ops naming remote CTAs
std::map<int, std::vector<std::int64_t>>
remote_ranks(const std::map<std::string, std::vector<std::int64_t>> &ops,
             int cluster_rank);

// operand_facts() — may throw missing_fact / unsupported like Python.
std::optional<SemanticOperands>
operand_facts(const DynRecord &rec, const StaticRow &row,
              const CaptureRules &rules, int cluster_rank,
              const std::map<int, std::vector<std::int64_t>> *ranks);

int opcode_width_bytes_public(const std::string &opcode);

// Fill row.derived (static-only values shared by every dynamic record
// of the row).  Call for every row before the cluster threads start;
// derived_of() falls back to a thread-local computation for rows that
// were not prepared.
void prepare_static_row(StaticRow &row, const CaptureRules &rules);
const StaticDerived &derived_of(const StaticRow &row,
                                const CaptureRules &rules);
void uput_wrap(SemanticOperands &o, const std::string &k,
               std::int64_t v);
void uput_str(SemanticOperands &o, const std::string &k,
              const std::string &v);

}  // namespace cyclescope_convert::capture
