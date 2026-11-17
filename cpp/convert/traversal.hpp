// TreeRuntime-equivalent over GSIMPRG01 + traverse/program_facts
// (mechanism_uses.py port).
#pragma once

#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <vector>

#include "convert/axes.hpp"
#include "convert/facts_reader.hpp"

namespace cyclescope_convert {

// decoded program-table runtime
struct ProgRow {
  std::string name;
  int event = 0, schema_id = 0, timing_kind = 0, root = 0;
  std::uint64_t field_mask = 0, op_begin = 0;
  int class_count = 0;
  std::vector<std::string> classes;
};

struct NodeRow {
  int parent = -1, children_begin = 0, children_count = 0;
  int guard_begin = -1, guard_count = 0, leaf_row = -1;
  std::vector<int> kids;   // reconstructed via parent_row
};

struct GuardRow {
  int axis = 0, op = 0;      // op: 0 eq, 1 range
  std::uint64_t lo = 0, hi = 0;
};

struct AxisDef {
  std::string name;
  int kind = 0;              // 0 int, 1 enum
  int owner = 0;             // 0 capture, 1 engine
  std::vector<std::string> values;
  std::unordered_map<std::string, std::uint64_t> value_index;  // enum
  // position of `v` in values, or false (first match, like the scan)
  bool index_of(const std::string &v, std::uint64_t &out) const {
    const auto it = value_index.find(v);
    if (it == value_index.end()) return false;
    out = it->second;
    return true;
  }
};

struct TreeRt {
  explicit TreeRt(FactsFile &programs);
  std::vector<AxisDef> axes;             // sorted-name order
  std::unordered_map<std::string, int> axis_index;
  std::vector<NodeRow> nodes;
  std::vector<GuardRow> guards;
  std::vector<ProgRow> progs;         // sorted-name order
  std::unordered_map<std::string, int> prog_row;
  std::unordered_map<std::string, std::vector<int>> by_class;   // cls -> prog rows
  std::set<int> predicated_programs;     // prog rows
  // eq_index[nid] -> (axis_index, {value->child_row}, fallback_rows)
  struct Eq { int axis; std::map<std::uint64_t, int> by_value;
              std::vector<int> fallback; };
  std::map<int, Eq> eq_index;
  std::map<int, std::vector<std::pair<int, int>>> plan;  // (kid,gidx|-1)
  std::set<int> engine_child;
  std::map<int, std::vector<std::string>> required_axes;
  // leaf rows: LEAF.parm rows? needed for leaf facts ranges? FACT rows
  // in gsp FACT are leaf-guard-path facts (shared, appended to segment
  // FACT already) — per-node fact ranges not needed for traversal.
  int row_of_node(int nid_row) const { return nid_row; }
};

// traverse result
struct UseRow {
  bool missing = false;
  std::string missing_key;             // "axis:name"
  std::string program;
  int prog = -1, node = -1;
  std::map<std::string, AxVal> facts;
  std::map<std::string, AxVal> cache;
  bool skip = false;
};

std::vector<UseRow> program_facts(const TreeRt &rt,
                                  const StaticInfo &st,
                                  const DynOps *dyn, AxisCtx &ctx,
                                  const std::string &cls,
                                  const std::vector<std::string>
                                      &classes);

}  // namespace cyclescope_convert
