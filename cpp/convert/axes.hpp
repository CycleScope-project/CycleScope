// Axis functions — port of convert/axes.py.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "convert/convert_rules.hpp"
#include "convert/semantics.hpp"
#include <unordered_map>

namespace cyclescope_convert {

// Value tri-state: python int | str | None | NOT_APPLICABLE.
struct AxVal {
  enum Tag : int { kNone, kNa, kInt, kStr } tag = kNone;
  long long i = 0;
  std::string s;
  bool is_na() const { return tag == kNa; }
  bool is_none() const { return tag == kNone; }
  static AxVal none() { return {}; }
  static AxVal na() { AxVal v; v.tag = kNa; return v; }
  static AxVal iv(long long x) { AxVal v; v.tag = kInt; v.i = x; return v; }
  static AxVal sv(std::string x) { AxVal v; v.tag = kStr; v.s = std::move(x); return v; }
};

// per-warp flow state (axes.FlowContext)
struct FlowCtx {
  // inst idx -> (reader static_index, role|none)
  std::map<std::uint64_t, std::pair<std::uint64_t, std::string>> readers;
  std::pair<bool, std::uint64_t> reader{false, 0};  // has reader, static idx
  std::string reader_role;
  bool reader_role_none = false;
  // executing LDGSTS since last ARRIVES or LDGDEPBAR (group commit)
  int cpasync_count = 0;
};

struct AxisCtx {
  int n_ctas = 0;
  // tensor_maps rank -> rank value (tensor_rank axis needs tm.rank only;
  // SemanticsContext::tensor_maps stores element_bytes+box)
  const SemanticsContext *sem = nullptr;
  // launch facts
  std::array<int, 3> block{0, 0, 0};
  int registers_per_thread = -1;
  int declared_math_warpgroups = -1;   // -1 = absent
  FlowCtx *flow = nullptr;
  const ConvertRules *rules = nullptr;
  std::string arch;
  bool arch_bw = false;  // arch == "blackwell"; avoids hot-path compares
  // tensor rank lookup: tensor_map id -> rank
  const std::map<int, std::pair<std::uint32_t, std::vector<std::uint64_t>>>
      *tmaps = nullptr;
  // statics opcode strings for first-reader class
  const std::vector<StaticInfo> *statics = nullptr;
  // branch_taken axis: static pc of this warp's next dynamic instance
  std::uint64_t next_static_pc = 0;
  bool has_next_static_pc = false;
  // memoized values of axes whose result depends only on the static
  // row (no DynOps, no flow/next_static_pc state); keyed by st.pc
  std::unordered_map<std::string,
                     std::unordered_map<std::uint64_t, AxVal>>
      own_static_axis_cache;
  // the memo in use (shared across segments of one facts dir by the
  // builder; defaults to own_static_axis_cache)
  std::unordered_map<std::string,
                     std::unordered_map<std::uint64_t, AxVal>>
      *static_axis_cache = &own_static_axis_cache;
};

// static operand tokens accessor shared with builder
AxVal axis_value(const std::string &name, const StaticInfo &st,
                 const DynOps *dyn, AxisCtx &ctx);

// blackwell idesc helper used by axes + decode
std::map<std::string, long long> umma_idesc_facts(const std::string &opc,
                                                  std::uint64_t v);
std::string umma_kind_of_axis(const std::string &opc);
std::string umma_kind_of_idesc(const std::string &opc,
                             std::uint64_t v);

}  // namespace cyclescope_convert
