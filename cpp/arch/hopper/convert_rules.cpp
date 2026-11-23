// Hopper convert rules — port of python/cyclescope/arch/hopper/rules.py
#include "convert/convert_rules.hpp"
#include "convert/str_util.hpp"
#include "convert/semantics.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>

namespace {

std::string upper(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::toupper((unsigned char)c));
  return s;
}
std::vector<std::string> split_dots(const std::string &s) {
  return cyclescope_convert::split_delim(s, '.');
}
bool in(const std::string &s, std::initializer_list<const char *> l) {
  return std::find(l.begin(), l.end(), s) != l.end();
}

constexpr int FLAG_MEMORY = 1, FLAG_ATOMIC = 2, FLAG_BARRIER = 4,
              FLAG_BRANCH = 8, FLAG_ASYNC = 16, FLAG_TENSOR = 32,
              FLAG_PREDICATED = 64, FLAG_DEFER_BLOCKING = 512;

const std::set<std::string> kSemanticRoots = {
    "BAR",  "MBAR",   "SYNCS",   "ARRIVES",  "DEPBAR",  "LDGDEPBAR",
    "WARPGROUP", "UCGABAR", "UCGABAR_ARV", "UCGABAR_WAIT", "MEMBAR",
    "FENCE", "WARPSYNC", "NOP", "YIELD", "USETMAXREG", "UTMACMDFLUSH",
    "USETSHMSZ", "UTMACCTL", "NANOSLEEP", "BRA", "BRX", "BRXU", "CALL",
    "BREAK", "BSSY", "BSYNC", "EXIT", "RET", "ENDCOLLECTIVE", "ERRBAR",
    "CGAERRBAR", "CCTL", "HGMMA", "QGMMA", "WGMMA", "IGMMA", "HMMA",
    "IMMA", "BMMA"};
const std::set<std::string> kBranchRoots = {"BRA", "BRX", "BRXU", "CALL",
                                            "BREAK", "BSSY", "BSYNC",
                                            "RET"};
const std::set<std::string> kMatrixRoots = {"HGMMA", "QGMMA", "WGMMA",
                                            "IGMMA", "HMMA", "IMMA",
                                            "BMMA"};
const std::set<std::string> kAtomicRoots = {"ATOM", "ATOMG", "ATOMS",
                                            "RED", "REDG", "REDS"};
const std::set<std::string> kAsyncRoots = {"LDGSTS", "UTMALDG", "UTMASTG",
                                           "UTMAREDG", "UBLKCP",
                                           "UBLKRED"};
const std::set<std::string> kBarrierSubset = {
    "BAR",  "MBAR",   "SYNCS",  "ARRIVES",  "DEPBAR",
    "LDGDEPBAR", "MEMBAR", "FENCE", "WARPSYNC", "UCGABAR",
    "UCGABAR_ARV", "UCGABAR_WAIT", "ERRBAR", "CGAERRBAR"};

struct HopperRules final : cyclescope_convert::ConvertRules {
  std::string memory_space(const cyclescope_convert::StaticInfo &s) const override {
    if (!s.memory_space.empty()) return s.memory_space;
    auto toks = split_dots(upper(s.opcode));
    const auto &root = toks[0];
    if (in(root, {"LDG", "STG", "ATOMG", "REDG", "LDGSTS", "UTMALDG",
                  "UTMASTG", "UTMAREDG", "UBLKCP", "UBLKRED"}))
      return "global";
    if (in(root, {"LDS", "STS", "LDSM", "STSM", "ATOMS", "REDS"}))
      return "shared";
    if (in(root, {"LDL", "STL"})) return "local";
    if (in(root, {"LDC", "ULDC"})) return "constant";
    if (in(root, {"LD", "ST"})) return "generic";
    return "";
  }

  std::string operation_class(const cyclescope_convert::StaticInfo &s) const override {
    auto toks = split_dots(upper(s.opcode));
    const auto &root = toks[0];
    if (root == "CS2R" && !s.operands.empty()) {
      std::string ops;
      for (auto &t : s.operands) ops += upper(t) + " ";
      if (ops.find("GLOBALTIMER") != std::string::npos ||
          ops.find("GLOBALTIM") != std::string::npos)
        return "CS2R.GLOBALTIMER";
      if (ops.find("CLOCK") != std::string::npos) return "CS2R.CLOCK";
      return "CS2R.ZERO";
    }
    if ((root == "LD" || root == "ST")) {
      if (memory_space(s) == "shared")
        return root == "LD" ? "LDS" : "STS";
    }
    if (root == "MUFU") {
      for (const char* op : {"EX2", "RSQ", "LG2", "TANH"})
        if (std::find(toks.begin(), toks.end(), op) != toks.end())
          return std::string("MUFU.") + op;
    }
    if (root == "DEPBAR" &&
        std::find(toks.begin(), toks.end(), "LE") != toks.end())
      return "DEPBAR.FIFO";
    if (root == "IMAD" &&
        std::find(toks.begin(), toks.end(), "HI") != toks.end())
      return "IMAD.HI";
    if (root == "USETSHMSZ" &&
        std::find(toks.begin(), toks.end(), "FLUSH") != toks.end())
      return "USETSHMSZ.FLUSH";
    if (root == "MEMBAR") {
      std::string sem, scope;
      for (auto &t : toks) {
        if (t == "ALL" || t == "SC") sem = t;
        if (t == "CTA" || t == "GPU" || t == "SYS") scope = t;
      }
      if (!sem.empty() && !scope.empty())
        return "MEMBAR." + sem + "." + scope;
    }
    if (root == "SYNCS" &&
        std::find(toks.begin(), toks.end(), "PHASECHK") != toks.end())
      return "SYNCS.PHASECHK";
    if (root == "SYNCS" &&
        std::find(toks.begin(), toks.end(), "ARRIVE") != toks.end() &&
        !s.operands.empty()) {
      const std::string dst = upper(s.operands[0]);
      const bool is_reg = dst.size() > 1 && dst[0] == 'R' &&
          std::all_of(dst.begin() + 1, dst.end(),
                      [](unsigned char c) { return std::isdigit(c); });
      if (is_reg)
        return std::find(toks.begin(), toks.end(), "ART0") != toks.end()
                   ? "SYNCS.ARRIVE.STATE.COUNT"
                   : "SYNCS.ARRIVE.STATE";
    }
    if (root == "FENCE" &&
        std::find(toks.begin(), toks.end(), "VIEW") != toks.end() &&
        std::find(toks.begin(), toks.end(), "ASYNC") != toks.end()) {
      for (std::size_t i = 1; i < toks.size(); ++i)
        if (toks[i] == "S" || toks[i] == "G")
          return "FENCE.VIEW.ASYNC." + toks[i];
    }
    if (in(root, {"HMMA", "IMMA", "BMMA"}) &&
        std::find(toks.begin(), toks.end(), "SP") != toks.end())
      return root + ".SPARSE";
    if (root == "SHFL") {
      for (std::size_t i = 1; i < toks.size(); ++i)
        if (in(toks[i], {"IDX", "UP", "DOWN", "BFLY"}))
          return "SHFL." + toks[i];
    }
    return root;
  }

  std::vector<std::string>
  operation_classes(const cyclescope_convert::StaticInfo &s) const override {
    auto cls = operation_class(s);
    auto root = split_dots(upper(s.opcode))[0];
    if (cls == root) return {cls};
    return {cls, root};
  }

  int instruction_flags(const cyclescope_convert::StaticInfo &s) const override {
    auto toks = split_dots(upper(s.opcode));
    const auto &root = toks[0];
    int flags = 0;
    if (!memory_space(s).empty()) flags |= FLAG_MEMORY;
    if (kAtomicRoots.count(root)) flags |= FLAG_ATOMIC;
    if (kSemanticRoots.count(root) && kBarrierSubset.count(root))
      flags |= FLAG_BARRIER;
    if (kBranchRoots.count(root)) flags |= FLAG_BRANCH;
    if (kAsyncRoots.count(root)) flags |= FLAG_ASYNC;
    if (kMatrixRoots.count(root)) flags |= FLAG_TENSOR;
    if (s.has_predicate) flags |= FLAG_PREDICATED;
    if (root == "BAR" &&
        std::find(toks.begin() + 1, toks.end(), "DEFER_BLOCKING") !=
            toks.end())
      flags |= FLAG_DEFER_BLOCKING;
    return flags;
  }

  cyclescope_convert::SemOp promote_operation(const cyclescope_convert::StaticInfo &s,
                                const cyclescope_convert::DynOps &d,
                                const cyclescope_convert::SemanticsContext &ctx,
                                cyclescope_convert::SemOp op) const override {
    // STS / STS.64 onto a barrier word named by explicit barrier ops
    // (prescan) decodes as mbarrier_init (or is a capture gap).
    return cyclescope_convert::promote_store_to_mbarrier_init(s, d, ctx, op);
  }
};

}  // namespace

namespace cyclescope_convert {
const ConvertRules *hopper_convert_rules() {
  static const HopperRules kRules;
  return &kRules;
}
}  // namespace cyclescope_convert

// ---- 3b: semantic_operation table + arch decoders --------------------
#include "convert/semantics.hpp"

#include <map>
#include <regex>

namespace {

const std::map<std::string, cyclescope_convert::SemOp> kSemOps = {
    {"SYNCS.EXCH", cyclescope_convert::SemOp::mbarrier_init},
    {"SYNCS.ARRIVE", cyclescope_convert::SemOp::mbarrier_arrive},
    {"SYNCS.PHASECHK", cyclescope_convert::SemOp::mbarrier_wait},
    {"MBAR.TRY_WAIT", cyclescope_convert::SemOp::mbarrier_wait},
    {"MBAR.TEST_WAIT", cyclescope_convert::SemOp::mbarrier_wait},
    {"BAR.ARV", cyclescope_convert::SemOp::cta_barrier_arrive},
    {"BAR.SYNC", cyclescope_convert::SemOp::cta_barrier_wait},
    {"BAR.RED", cyclescope_convert::SemOp::cta_barrier_wait},
    {"UCGABAR_ARV", cyclescope_convert::SemOp::cluster_barrier_arrive},
    {"UCGABAR_WAIT", cyclescope_convert::SemOp::cluster_barrier_wait},
    {"UTMALDG", cyclescope_convert::SemOp::tma_load},
    {"UTMASTG", cyclescope_convert::SemOp::bulk_group_issue},
    {"UTMAREDG", cyclescope_convert::SemOp::bulk_group_issue},
    // .G.S = destination global, source shared: a bulk store (bulk group)
    {"UBLKCP.G.S", cyclescope_convert::SemOp::bulk_group_issue},
    {"UBLKRED.G.S", cyclescope_convert::SemOp::bulk_group_issue},
    {"UTMACMDFLUSH", cyclescope_convert::SemOp::bulk_group_commit},
    {"UTMACCTL.COMMIT", cyclescope_convert::SemOp::bulk_group_commit},
    {"UTMACCTL.IV", cyclescope_convert::SemOp::descriptor_visibility_acquire},
    {"MEMBAR.ALL.GPU", cyclescope_convert::SemOp::descriptor_visibility_publish},
    {"ERRBAR", cyclescope_convert::SemOp::descriptor_visibility_error_check},
    {"CGAERRBAR", cyclescope_convert::SemOp::descriptor_visibility_error_check},
    {"STG", cyclescope_convert::SemOp::descriptor_visibility_store},
    {"UBLKCP", cyclescope_convert::SemOp::bulk_copy},
    {"UBLKRED", cyclescope_convert::SemOp::bulk_reduce},
    {"LDGSTS", cyclescope_convert::SemOp::async_copy_issue},
    {"LDGDEPBAR", cyclescope_convert::SemOp::async_copy_commit},
    {"ARRIVES.LDGSTSBAR", cyclescope_convert::SemOp::async_copy_commit},
    {"DEPBAR", cyclescope_convert::SemOp::depbar_wait},
    {"STAS", cyclescope_convert::SemOp::shared_store_async},
    {"FENCE.VIEW.ASYNC.S", cyclescope_convert::SemOp::descriptor_visibility_proxy_fence},
    {"FENCE.VIEW.ASYNC.G", cyclescope_convert::SemOp::none},
    {"SYNCS.CCTL.IV", cyclescope_convert::SemOp::mbarrier_inval},
    {"SYNCS.CCTL.IVALL", cyclescope_convert::SemOp::none},
    {"NANOSLEEP", cyclescope_convert::SemOp::nanosleep},
    {"PREEXIT", cyclescope_convert::SemOp::dependents_launch},
    {"WARPGROUP.ARRIVE", cyclescope_convert::SemOp::warpgroup_arrive},
    {"WARPGROUP.DEPBAR.LE", cyclescope_convert::SemOp::warpgroup_wait},
    {"WARPGROUP.WAIT", cyclescope_convert::SemOp::warpgroup_wait},
    {"HGMMA", cyclescope_convert::SemOp::wgmma},
    {"WGMMA", cyclescope_convert::SemOp::wgmma},
    {"QGMMA", cyclescope_convert::SemOp::wgmma},
    {"IGMMA", cyclescope_convert::SemOp::wgmma},
};

std::vector<std::string> toks_of(const std::string &s) {
  return cyclescope_convert::split_delim(s, '.');
}

// _dtype_fields
std::pair<int, int> dtype_fields(const std::string &opc) {
  static const std::regex dt_re(R"((?:[BUSF]\d+|BF16|TF32|E4M3|E5M2))",
                                std::regex::icase);
  static const std::map<std::string, int> codes = {
      {"", 0}, {"f16", 1}, {"bf16", 2}, {"tf32", 3}, {"f32", 4},
      {"s32", 5}, {"e4m3", 6}, {"e5m2", 7}, {"f8", 8}, {"u8", 9},
      {"s8", 10}, {"f64", 11}};
  auto toks = toks_of(opc);
  std::vector<std::string> types;
  for (std::size_t i = 1; i < toks.size(); ++i) {
    if (std::regex_match(toks[i], dt_re)) {
      std::string l = toks[i];
      for (auto &c : l) c = std::tolower((unsigned char)c);
      types.push_back(l);
    }
  }
  const bool is_igmma = opc.rfind("IGMMA", 0) == 0;
  std::string acc = is_igmma ? "s32" : (types.empty() ? "f32" : types[0]);
  std::string inp;
  if (types.size() > 1) inp = types[1];
  else if (is_igmma) inp = "";
  else if (opc.rfind("HGMMA", 0) == 0) inp = "f16";
  else inp = "";
  auto at = codes.count(acc) ? codes.at(acc) : 0;
  auto it = codes.count(inp) ? codes.at(inp) : 0;
  return {it, at};
}

std::uint64_t shape_packed(const std::string &opc) {
  std::smatch m;
  static const std::regex re(R"((?:^|\.)(\d+)X(\d+)X(\d+)(?:\.|$))");
  if (!std::regex_search(opc, m, re)) return 0;
  const auto M = std::stoull(m[1]), N = std::stoull(m[2]),
             K = std::stoull(m[3]);
  return (M << 32) | (N << 16) | K;
}

const std::regex kRegTok(R"(^(?:RZ|R\d+|URZ|UR\d+|P\d+|PT|UP\d+)$)",
                         std::regex::icase);
const std::regex kNumTok(R"(^(0x[0-9a-fA-F]+|\d+)$)");

int source_mode(const cyclescope_convert::StaticInfo &st, const std::string &where) {
  const auto &t = st.operands;
  if (t.size() < 2)
    throw cyclescope_convert::ConversionError("missing_fact:wgmma_source_operands (" +
                                where + ")");
  std::string first = t[1], second = t.size() > 2 ? t[2] : "";
  for (auto &c : first) c = std::tolower((unsigned char)c);
  for (auto &c : second) c = std::tolower((unsigned char)c);
  if (first.rfind("gdesc[", 0) == 0) return 1;
  if (std::regex_match(first, kRegTok) &&
      second.rfind("gdesc[", 0) == 0)
    return 2;
  throw cyclescope_convert::ConversionError("unsupported:wgmma_source_operands (" +
                              where + ")");
}

std::pair<int, int> acc_registers(const cyclescope_convert::StaticInfo &st,
                                  const std::string &opc, int acc_dtype,
                                  const std::string &where) {
  static const std::regex n_re(R"(64x(\d+)x)", std::regex::icase);
  std::smatch m;
  int n = 2;
  if (std::regex_search(opc, m, n_re)) n = std::stoi(m[1]);
  const int cnt = std::max(1, n / ((acc_dtype == 4 || acc_dtype == 5) ? 2 : 4));
  int begin = -1;
  static const std::regex r_re(R"(^R(\d+)$)", std::regex::icase);
  for (auto &name : st.dst_names) {
    if (std::regex_match(name, m, r_re)) {
      begin = std::stoi(m[1]);
      break;
    }
  }
  if (begin < 0 && !st.operands.empty() &&
      std::regex_match(st.operands[0], m, r_re))
    begin = std::stoi(m[1]);
  if (begin < 0 && !st.operands.empty() && st.operands[0] == "RZ")
    return {0, 0};   // RZ writes no accumulator registers: a fact
  if (begin < 0)
    throw cyclescope_convert::ConversionError("missing_fact:wgmma_accumulator_register (" +
                                where + ")");
  return {begin, cnt};
}

bool gsb0(const cyclescope_convert::StaticInfo &st) {
  static const std::regex g_re(R"(^gsb0$)", std::regex::icase);
  for (auto &t : st.operands)
    if (std::regex_match(t, g_re)) return true;
  return false;
}

}  // namespace

namespace cyclescope_convert {

SemOp semantic_operation_hopper(const StaticInfo &st) {
  const auto opc = upper(st.opcode);
  auto toks = toks_of(opc);
  // two-word roots first
  for (int w = 4; w >= 2; --w) {
    if ((int)toks.size() >= w) {
      std::string head = toks[0];
      for (int i = 1; i < w; ++i) head += "." + toks[i];
      auto it = kSemOps.find(head);
      if (it != kSemOps.end()) return it->second;
    }
  }
  auto it = kSemOps.find(toks[0]);
  if (it != kSemOps.end()) return it->second;
  // supported root without a mapped modifier -> raise
  // UTMACCTL variants other than COMMIT/IV and MEMBAR variants carry
  // no semantics (hopper rules.semantic_operation).
  if (toks[0] == "UTMACCTL" || toks[0] == "MEMBAR") return SemOp::none;
  {
    static const std::set<std::string> supported = [] {
      std::set<std::string> s;
      for (auto &[k, v] : kSemOps)
        s.insert(k.substr(0, k.find('.')));
      return s;
    }();
    if (supported.count(toks[0]))
      throw cyclescope_convert::ConversionError(
          "unsupported:semantic_modifier (" + where_str(st) + ")");
  }
  return SemOp::none;
}

SemRow decode_hopper_semantics(const StaticInfo &st, const DynOps &d,
                               SemanticsContext &ctx, SemOp op, int,
                               int wslot) {
  const auto opc = upper(st.opcode);
  const std::string where = where_str(st);
  if (op == SemOp::warpgroup_arrive) {
    const int group = ctx.wg_counts.count(wslot) ? ctx.wg_counts[wslot] : 0;
    ctx.wg_counts[wslot] = group + 1;
    ctx.wg_groups[wslot] = group;
    SemRow r;
    r.kind = SemKind::warpgroup_arrive;
    r.fields["group"] = group;
    r.fields["pending_allowed"] = 0;
    return r;
  }
  if (op == SemOp::wgmma) {
    int flags = 0;
    int group;
    if (ctx.wg_groups.count(wslot) && ctx.wg_groups[wslot] >= 0) {
      group = ctx.wg_groups[wslot];
    } else {
      group = ctx.wg_counts.count(wslot) ? ctx.wg_counts[wslot] : 0;
      ctx.wg_counts[wslot] = group + 1;
      ctx.wg_groups[wslot] = group;
      flags = SEMA_FLAG_GROUP_OPENER;
    }
    auto [inp, acc] = dtype_fields(opc);
    auto [ab, ac2] = acc_registers(st, opc, acc, where);
    SemRow r;
    r.kind = SemKind::wgmma;
    r.flags = flags;
    r.fields["group"] = group;
    r.fields["accumulator_registers_begin"] = ab;
    r.fields["accumulator_registers_count"] = ac2;
    r.fields["shape_packed"] = shape_packed(opc);
    r.fields["input_dtype"] = inp;
    r.fields["accumulator_dtype"] = acc;
    r.fields["source_mode"] = source_mode(st, where);
    if (gsb0(st)) {
      r.flags |= SEMA_FLAG_GROUP_COMMIT;
      ctx.wg_groups[wslot] = -1;
      ctx.wg_commits[wslot] = ctx.wg_commits.count(wslot)
                                  ? ctx.wg_commits[wslot] + 1
                                  : 1;
    }
    return r;
  }
  if (op == SemOp::warpgroup_wait) {
    static const std::regex sb_re(R"(gsb\d+)", std::regex::icase);
    bool has_sb = false;
    for (auto &t : st.operands)
      if (std::regex_match(t, sb_re)) has_sb = true;
    if (has_sb && !gsb0(st))
      throw cyclescope_convert::ConversionError("unsupported:warpgroup_scoreboard (" +
                            where + ")");
    long keep;
    if (opc.rfind("WARPGROUP.WAIT", 0) == 0) {
      keep = 0;
    } else {
      std::vector<long> nums;
      for (auto &t : st.operands)
        if (std::regex_match(t, kNumTok))
          nums.push_back(std::stol(t, nullptr, 0));
      if (nums.empty())
        throw cyclescope_convert::ConversionError("missing_fact:warpgroup_wait_keep (" +
                              where + ")");
      keep = nums.back();
    }
    if (keep > 7)
      throw cyclescope_convert::ConversionError("unsupported:warpgroup_wait_keep (" +
                            where + ")");
    SemRow r;
    r.kind = SemKind::warpgroup_wait;
    r.fields["group"] = ctx.wg_commits.count(wslot)
                            ? ctx.wg_commits[wslot]
                            : 0;
    r.fields["pending_allowed"] = keep;
    return r;
  }
  return {};
}

}  // namespace cyclescope_convert

// ---- opcode_facts + helper exports for axes --------------------------
namespace {
const std::regex kMatrixRe(
    R"((?:LDSM|STSM)\.(\d+)\.M(T)?(\d)(\d)(?:\.(\d+))?(?:\..*)?)");
const std::regex kDtRe(R"(^[BUSF]\d+$|^BF16$|^TF32$|^E4M3$|^E5M2$)");
}

namespace cyclescope_convert {

std::pair<int, int> hopper_dtype_fields(const std::string &opc) {
  auto p = dtype_fields(opc);
  return {p.first, p.second};
}

int hopper_source_mode(const std::vector<std::string> &ops, bool &ok) {
  StaticInfo st;
  st.operands = ops;
  try {
    ok = true;
    return source_mode(st, "axis");
  } catch (...) {
    ok = false;
    return 0;
  }
}

std::map<std::string, long long>
hopper_opcode_facts(const std::string &opcode) {
  std::map<std::string, long long> f;
  const auto op = upper(opcode);
  StaticInfo si;
  si.opcode = op;
  f["operation_class"] = 0;  // string value — axes uses rules->operation_class
  std::smatch m;
  const auto root = op.substr(0, op.find('.'));
  if (root == "LDSM" || root == "STSM") {
    if (std::regex_match(op, m, kMatrixRe)) {
      f["width_bytes"] = std::stoll(m[1]) / 8;
      f["tensor_m"] = std::stoll(m[3]);
      f["tensor_n"] = std::stoll(m[4]);
      f["matrix_count"] = m[5].matched ? std::stoll(m[5]) : 1;
      f["matrix_transposed"] = m[2].matched ? 1 : 0;
    }
  }
  // sparse GMMA forms carry a .SP token in the opcode
  {
    const auto toks = split_dots(op);
    f["tensor_sparse"] =
        std::find(toks.begin(), toks.end(), "SP") != toks.end() ? 1 : 0;
  }
  // datatype: non-integer — axes reads it via str; encode marker
  // (axis f_datatype does not use opcode_facts for datatype)
  return f;
}

}  // namespace cyclescope_convert
