// Static instruction tables (functions.jsonl + functions/<fid>.tsv).
// Mirrors python/cyclescope/convert/static_code.py:
//   load_case / _parse_operands / _register_roles / _SPACE_BY_ROOT.
#pragma once
#include "capture/facts_writer.hpp"
#include "convert/convert_rules.hpp"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyclescope_convert::capture {

// Per-static-row values that operand_facts / memory_refs otherwise
// recomputed for every dynamic record (upper-cased opcode, root, operand
// tokens, root-class flags, decoded width); filled once by
// prepare_static_row (operand_facts.hpp) before the cluster threads run.
struct StaticDerived {
  bool ready = false;
  std::string up, root;                 // upper(opcode_text), its root
  std::vector<std::string> tokens;      // operands_text split on commas
  bool op_root = false;                 // root starts with an _OP_ROOTS entry
  bool trivial = false;                 // operand_facts can only yield after_ready
  bool is_tma = false, is_bulk = false, no_mref_root = false,
       const_root = false;
  std::vector<std::pair<std::string, std::int64_t>> op_consts;  // op<N>
  std::optional<int> width;             // opcode_width_bytes
  std::string width_error;              // its missing_fact message, if any
  std::string access_kind;
};

struct StaticRow {
  int idx = 0;
  std::string sass;
  std::vector<std::string> ops_layout;
  std::uint32_t mref_count = 0;
  std::string opcode_text;     // opcode with modifiers (no guard)
  std::string operands_text;
  std::int32_t branch_target = -1;   // -1 = None
  int static_index = -1;
  StatRow inst;                // row fields for write_statics_file
  StaticDerived derived;
};

struct FunctionStatic {
  int function_id = 0;
  std::string name, mangled;
  std::uint64_t addr = 0;
  std::map<int, StaticRow> rows;   // idx -> row
};

// sass_decode helpers used by load_case (opcode_root,
// control_word_from_bytes/control_schedule_fields,
// numeric_control_target).
std::string opcode_root(const std::string &opcode);
struct ControlWord {
  int stall = 0, yield_hint = 0, read_barrier = 7, write_barrier = 7,
      wait_mask = 0, reuse_mask = 0;
};
ControlWord control_word_from_bytes(const std::string &bytes_hex);
// returns false when not exactly one numeric token (MissingFact)
bool numeric_control_target(const std::string &operands,
                            std::int32_t *target);

// load_case: MissingFact surfaces as std::runtime_error
// ("missing_fact:<path> (<where>)").
std::map<int, FunctionStatic> load_case(const std::string &case_dir,
                                        const ConvertRules &rules);

}  // namespace cyclescope_convert::capture
