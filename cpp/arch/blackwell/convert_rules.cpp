// Blackwell convert rules — port of python/cyclescope/arch/blackwell/rules.py
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
              FLAG_PREDICATED = 64, FLAG_TMEM = 128, FLAG_CLC = 256;

const std::set<std::string> kBranchRoots = {"BRA", "BRX", "BRXU", "CALL",
                                            "BREAK", "BSSY", "BSYNC",
                                            "RET"};
const std::set<std::string> kMatrixRoots = {
    "HGMMA", "QGMMA", "WGMMA", "HMMA", "IMMA", "BMMA", "UTCHMMA",
    "UTCQMMA", "UTCOMMA"};
const std::set<std::string> kTmemRoots = {"LDTM", "STTM"};
const std::set<std::string> kAtomicRoots = {"ATOM", "ATOMG", "ATOMS",
                                            "RED", "REDG", "REDS"};
const std::set<std::string> kAsyncRoots = {"LDGSTS", "UTMALDG", "UTMASTG",
                                           "UTMAREDG", "UBLKCP",
                                           "UBLKRED"};
const std::set<std::string> kBarrierRoots = {
    "BAR",  "MBAR",   "SYNCS",  "ARRIVES", "DEPBAR",
    "LDGDEPBAR", "MEMBAR", "FENCE", "WARPSYNC", "UCGABAR",
    "UCGABAR_ARV", "UCGABAR_WAIT", "UTCBAR", "ERRBAR", "CGAERRBAR"};

namespace {
const std::set<std::string> kNamedBarrierRoots = {
    "SYNCS", "UTCBAR", "ARRIVES", "UCGABAR"};
}

struct BlackwellRules final : cyclescope_convert::ConvertRules {
  std::string memory_space(const cyclescope_convert::StaticInfo &s) const override {
    if (!s.memory_space.empty()) return s.memory_space;
    auto toks = split_dots(upper(s.opcode));
    const auto &root = toks[0];
    if (in(root, {"LDG", "STG", "ATOMG", "REDG", "LDGSTS", "UTMALDG",
                  "UTMASTG", "UTMAREDG", "UBLKCP", "UBLKRED"}))
      return "global";
    if (in(root, {"LDS", "STS", "LDSM", "STSM", "ATOMS", "REDS", "STAS"}))
      return "shared";
    if (in(root, {"LDL", "STL"})) return "local";
    if (in(root, {"LDC", "ULDC", "LDCU"})) return "constant";
    if (in(root, {"LD", "ST"})) return "generic";
    return "";
  }

  std::string operation_class(const cyclescope_convert::StaticInfo &s) const override {
    auto toks = split_dots(upper(s.opcode));
    const auto &root = toks[0];
    auto has = [&](const char *t) {
      return std::find(toks.begin(), toks.end(), t) != toks.end();
    };
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
    if (root == "HADD2" && toks.size() == 2 && toks[1] == "F32")
      return "HADD2.F32";
    if (root == "MUFU") {
      for (const char* op : {"EX2", "RSQ", "LG2", "TANH"})
        if (has(op)) return std::string("MUFU.") + op;
    }
    if (root == "DEPBAR" && has("LE")) return "DEPBAR.FIFO";
    if (root == "IMAD" && has("HI")) return "IMAD.HI";
    if (root == "MEMBAR") {
      std::string sem, scope;
      for (auto &t : toks) {
        if (t == "ALL" || t == "SC") sem = t;
        if (t == "CTA" || t == "GPU" || t == "SYS") scope = t;
      }
      if (!sem.empty() && !scope.empty())
        return "MEMBAR." + sem + "." + scope;
    }
    if (root == "SYNCS" && has("PHASECHK")) return "SYNCS.PHASECHK";
    if (root == "SYNCS" && has("ARRIVE") && !s.operands.empty()) {
      const std::string dst = upper(s.operands[0]);
      const bool is_reg = dst.size() > 1 && dst[0] == 'R' &&
          std::all_of(dst.begin() + 1, dst.end(),
                      [](unsigned char c) { return std::isdigit(c); });
      if (is_reg)
        return has("ART0") ? "SYNCS.ARRIVE.STATE.COUNT"
                           : "SYNCS.ARRIVE.STATE";
    }
    if (root == "CCTL" && has("IVALL")) return "CCTL.IVALL";
    if (root == "SYNCS" && toks.size() == 3 && toks[1] == "CCTL" &&
        toks[2] == "IVALL")
      return "CCTL.IVALL";
    if (root == "FENCE" && has("VIEW") && has("ASYNC")) {
      for (std::size_t i = 1; i < toks.size(); ++i)
        if (toks[i] == "S" || toks[i] == "G")
          return "FENCE.VIEW.ASYNC." + toks[i];
    }
    if (in(root, {"HMMA", "IMMA", "BMMA"}) && has("SP"))
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
    auto opcode = upper(s.opcode);
    auto root = opcode.substr(0, opcode.find('.'));
    std::vector<std::string> out =
        (cls == root) ? std::vector<std::string>{cls}
                      : std::vector<std::string>{cls, root};
    auto mods = split_dots(opcode);
    if (root == "LDTM") {
      for (std::size_t i = 1; i < mods.size(); ++i)
        if (mods[i].rfind("STAT", 0) == 0) {
          out.push_back("tmem_load_stat");
          break;
        }
    } else if (cls == "FENCE.VIEW.ASYNC.T") {
      out.push_back("tmem_wait_store");
    }
    return out;
  }

  int instruction_flags(const cyclescope_convert::StaticInfo &s) const override {
    auto toks = split_dots(upper(s.opcode));
    const auto &root = toks[0];
    int flags = 0;
    if (!memory_space(s).empty()) flags |= FLAG_MEMORY;
    if (kAtomicRoots.count(root)) flags |= FLAG_ATOMIC;
    if (kBarrierRoots.count(root)) flags |= FLAG_BARRIER;
    if (kBranchRoots.count(root)) flags |= FLAG_BRANCH;
    if (kAsyncRoots.count(root)) flags |= FLAG_ASYNC;
    if (kMatrixRoots.count(root)) flags |= FLAG_TENSOR;
    if (kTmemRoots.count(root)) flags |= FLAG_TMEM;
    if (root == "UGETNEXTWORKID") flags |= FLAG_CLC;
    if (s.has_predicate) flags |= FLAG_PREDICATED;
    return flags;
  }

  const std::set<std::string> &named_barrier_roots() const override {
    return kNamedBarrierRoots;
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
const ConvertRules *blackwell_convert_rules() {
  static const BlackwellRules kRules;
  return &kRules;
}
}  // namespace cyclescope_convert

// ---- 3b: semantic_operation table + arch decoders --------------------

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
    {"ARRIVES", cyclescope_convert::SemOp::async_copy_commit},
    {"DEPBAR", cyclescope_convert::SemOp::depbar_wait},
    {"STAS", cyclescope_convert::SemOp::shared_store_async},
    {"FENCE.VIEW.ASYNC.S", cyclescope_convert::SemOp::descriptor_visibility_proxy_fence},
    {"FENCE.VIEW.ASYNC.G", cyclescope_convert::SemOp::none},
    {"SYNCS.CCTL.IV", cyclescope_convert::SemOp::mbarrier_inval},
    {"SYNCS.CCTL.IVALL", cyclescope_convert::SemOp::none},
    {"FENCE.VIEW.ASYNC.T", cyclescope_convert::SemOp::tmem_wait_store},
    {"NANOSLEEP", cyclescope_convert::SemOp::nanosleep},
    {"PREEXIT", cyclescope_convert::SemOp::dependents_launch},
    {"UTCHMMA", cyclescope_convert::SemOp::umma_issue},
    {"UTCQMMA", cyclescope_convert::SemOp::umma_issue},
    {"UTCOMMA", cyclescope_convert::SemOp::umma_issue},
    {"UTCBAR", cyclescope_convert::SemOp::umma_commit},
    {"UTCATOMSWS", cyclescope_convert::SemOp::tmem_alloc},
    {"LDTM", cyclescope_convert::SemOp::tmem_load},
    {"STTM", cyclescope_convert::SemOp::tmem_store},
    {"UTCCP", cyclescope_convert::SemOp::tmem_copy},
    {"UGETNEXTWORKID", cyclescope_convert::SemOp::clc_query},
};

std::vector<std::string> toks_of(const std::string &s) {
  return cyclescope_convert::split_delim(s, '.');
}
std::vector<std::string> mods_of(const std::string &opc) {
  auto t = toks_of(opc);
  return {t.begin() + 1, t.end()};
}

// ---- umma helpers ----------------------------------------------------
const std::map<std::string, int> kUmmaKind = {
    {"f16", 1}, {"tf32", 2}, {"f8f6f4", 3}, {"mxf8f6f4", 4},
    {"mxf4", 5}, {"mxf4nvf4", 6}};

std::string umma_kind_of(const std::string &opc) {
  const auto root = toks_of(opc)[0];
  auto mods = mods_of(opc);
  auto has = [&](const char *m) {
    return std::find(mods.begin(), mods.end(), m) != mods.end();
  };
  if (root == "UTCHMMA") return has("TF32") ? "tf32" : "f16";
  if (root == "UTCQMMA") {
    if (has("BLOCK16") ||
        std::any_of(mods.begin(), mods.end(),
                    [](const std::string &m) {
                      return m.rfind("MX", 0) == 0;
                    }))
      return "mxf8f6f4";
    return "f8f6f4";
  }
  if (root == "UTCOMMA") return has("BLOCK16") ? "mxf4nvf4" : "mxf4";
  return "";
}

const std::map<std::string, int> kBaseK = {
    {"f16", 16}, {"tf32", 8}, {"f8f6f4", 32}, {"mxf8f6f4", 32},
    {"mxf4", 64}, {"mxf4nvf4", 64}};

struct UmmaFacts {
  std::string kind;
  int tensor_m = 0, tensor_n = 0, tensor_k = 0;
};

UmmaFacts umma_descriptor_facts(const std::string &opc,
                                std::uint64_t v) {
  const auto root = toks_of(opc)[0];
  auto mods = mods_of(opc);
  std::string kind = umma_kind_of(opc);
  if (root == "UTCHMMA") {
    const int t1 = (v >> 7) & 7, t2 = (v >> 10) & 7;
    kind = (t1 == 2 && t2 == 2) ? "tf32"
           : (t1 <= 1 && t2 <= 1) ? "f16" : "";
  } else if (root == "UTCQMMA") {
    kind = (v & (1 << 23)) ? "mxf8f6f4" : "f8f6f4";
  } else if (root == "UTCOMMA") {
    const bool b16 = std::find(mods.begin(), mods.end(), "BLOCK16") !=
                     mods.end();
    kind = (b16 || !(v & (1 << 23))) ? "mxf4nvf4" : "mxf4";
  }
  const bool sparse = v & (1 << 2);
  int tk = 0;
  if (kBaseK.count(kind)) tk = kBaseK.at(kind) * (sparse ? 2 : 1);
  if ((kind == "mxf4" || kind == "mxf4nvf4") && (v & (1ull << 31)))
    tk = sparse ? 192 : 96;
  else if (kind == "mxf8f6f4" && (v & (1ull << 31)) && tk)
    tk *= 2;
  UmmaFacts f;
  f.kind = kind;
  f.tensor_m = ((v >> 24) & 0x1F) << 4;
  f.tensor_n = ((v >> 17) & 0x3F) << 3;
  f.tensor_k = tk;
  return f;
}

int umma_a_src(const cyclescope_convert::StaticInfo &st, const std::string &where) {
  std::string text;
  for (auto &t : st.operands) text += t + " ";
  static const std::regex tmem_re(R"(tmem\[[^\]]+\])", std::regex::icase);
  static const std::regex gdesc_re(R"(gdesc\[[^\]]+\])",
                                   std::regex::icase);
  std::smatch mt, mg;
  const bool has_t = std::regex_search(text, mt, tmem_re);
  const bool has_g = std::regex_search(text, mg, gdesc_re);
  if (!has_t && !has_g)
    throw cyclescope_convert::ConversionError("unsupported:umma_a_source (" + where +
                                ")");
  if (!has_t) return 1;
  if (!has_g) return 2;
  return (mt.position() < mg.position()) ? 2 : 1;
}

std::pair<int, int> tmem_shape_repeat(const std::string &opc,
                                      const std::string &where) {
  static const std::map<std::string, std::string> known = {
      {"16X64B", "16x64b"}, {"16X128B", "16x128b"},
      {"16X256B", "16x256b"}, {"32X32B", "32x32b"},
      {"16DP64BIT", "16x64b"}, {"16DP128BIT", "16x128b"},
      {"16DP256BIT", "16x256b"}, {"32DP32BIT", "32x32b"}};
  static const std::map<std::string, int> codes = {
      {"16x64b", 1}, {"16x128b", 2}, {"16x256b", 3}, {"32x32b", 4}};
  static const std::regex xn_re(R"(^x(\d+)$)", std::regex::icase);
  static const std::regex utccp_re(R"(^(\d+)x32dp(\d+)bit$)");
  static const std::regex utccp_sb_re(R"(^(\d+)dp(\d+)bit$)");
  static const std::regex f_re(R"(^F\d+$)");
  const auto root = toks_of(opc)[0];
  if (root != "LDTM" && root != "STTM" && root != "UTCCP") return {0, 0};
  std::vector<std::string> parts;
  for (auto &m : mods_of(opc)) {
    if (m == "2CTA" || m == "STAT" || m == "T" || m == "S") continue;
    if (m.rfind("STAT", 0) == 0 || m.rfind("MAX", 0) == 0) continue;
    if (std::regex_match(m, f_re)) continue;
    parts.push_back(m);
  }
  if (parts.empty()) {
    // tcgen05.cp without an explicit Nx32dpMbit shape (UTCCP.T.S:
    // transposed default-shape copy): shape unknown, one copy;
    // bare LDTM/STTM is 32x32b.x1 (the tmem2 microbench kernels the
    // measured x1 leaf was gated on compile to bare LDTM/STTM)
    if (root == "UTCCP") return {0, 1};
    return {4, 1};
  }
  if (root == "UTCCP") {
    std::string j;
    for (auto &p : parts) {
      std::string l = p;
      for (auto &c : l) c = std::tolower((unsigned char)c);
      j += l;
    }
    if (j == "2x64dp128bit_lw02_lw13")
      return {(2 << 16) | 128, 1};
    std::smatch m;
    if (!std::regex_match(j, m, utccp_re)) {
      // single-block spelling <D>dp<M>bit (tcgen05.cp .128x128b etc):
      // D >= 32 vs N <= 4 in Nx32dpMbit keeps them distinguishable
      std::smatch sm;
      if (std::regex_match(j, sm, utccp_sb_re))
        return {(std::stoi(sm[1]) << 16) | std::stoi(sm[2]), 1};
      throw cyclescope_convert::ConversionError("unsupported:tmem_shape (" + where +
                                  ")");
    }
    return {(std::stoi(m[1]) << 16) | std::stoi(m[2]), 1};
  }
  const auto shape_text = parts[0];
  int repeat = 1;
  std::smatch xm;
  for (auto &m : parts)
    if (std::regex_match(m, xm, xn_re)) repeat = std::stoi(xm[1]);
  if (std::regex_match(shape_text, xn_re))
    return {4, repeat};
  // SASS halves of tcgen05.ld/st .16x32bx2: 16dp32bit_t0_t15 and
  // 16dp32bit_t16_t31 (source mechtree.tmem.16x32bx2) -> code 5
  static const std::regex half_re(R"(^16DP32BIT_T\d+_T\d+$)");
  if (std::regex_match(shape_text, half_re)) return {5, repeat};
  auto kit = known.count(shape_text) ? known.at(shape_text)
                                     : shape_text;
  std::string low = kit;
  for (auto &c : low) c = std::tolower((unsigned char)c);
  if (!codes.count(low))
    throw cyclescope_convert::ConversionError("unsupported:tmem_shape (" + where + ")");
  if (repeat == 1 && parts.size() > 1) {
    for (std::size_t i = 1; i < parts.size(); ++i)
      if (std::regex_match(parts[i], xm, xn_re)) {
        repeat = std::stoi(xm[1]);
        break;
      }
  }
  return {codes.at(low), repeat};
}

std::string tmem_alloc_op(const std::string &opc) {
  if (toks_of(opc)[0] != "UTCATOMSWS") return "";
  std::string j;
  for (auto &m : mods_of(opc)) j += "." + m;
  if (j.find("FIND_AND_SET") != std::string::npos) return "alloc";
  if (j.find("RELINQUISH") != std::string::npos) return "relinquish";
  if (j.find("AND") != std::string::npos) return "dealloc";
  return "";
}

}  // namespace

namespace cyclescope_convert {

SemOp semantic_operation_blackwell(const StaticInfo &st) {
  const auto opc = upper(st.opcode);
  auto toks = toks_of(opc);
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
  // no semantics (blackwell rules.semantic_operation).
  if (toks[0] == "UTMACCTL" || toks[0] == "MEMBAR") return SemOp::none;
  {
    static const std::set<std::string> supported = [] {
      std::set<std::string> s;
      for (auto &[k, v] : kSemOps)
        s.insert(k.substr(0, k.find('.')));
      return s;
    }();
    if (supported.count(toks[0]))
      throw cyclescope_convert::ConversionError("unsupported:semantic_modifier (" +
                            where_str(st) + ")");
  }
  return SemOp::none;
}

SemRow decode_blackwell_semantics(const StaticInfo &st, const DynOps &d,
                                  SemanticsContext &ctx, SemOp op,
                                  int cta, int wslot) {
  const auto opc = upper(st.opcode);
  const std::string where = where_str(st);
  auto mods = mods_of(opc);
  auto has = [&](const char *m) {
    return std::find(mods.begin(), mods.end(), m) != mods.end();
  };
  if (op == SemOp::umma_issue) {
    auto it = d.uniform.find("idesc_value");
    if (it == d.uniform.end())
      throw cyclescope_convert::ConversionError("missing_fact:umma_descriptor (" + where +
                            ")");
    const auto idesc = it->second;
    auto f = umma_descriptor_facts(opc, idesc);
    if (!kUmmaKind.count(f.kind))
      throw cyclescope_convert::ConversionError("unsupported:umma_kind (" + where + ")");
    SemRow r;
    r.kind = SemKind::umma_issue;
    r.fields["kind"] = kUmmaKind.at(f.kind);
    r.fields["cta_group"] = has("2CTA") ? 2 : 1;
    r.fields["a_source"] = umma_a_src(st, where);
    r.fields["tensor_m"] = f.tensor_m;
    r.fields["tensor_n"] = f.tensor_n;
    r.fields["descriptor"] = idesc;
    return r;
  }
  if (op == SemOp::umma_commit) {
    std::uint64_t addr = 0;
    if (auto *v = [&] {
          const char *names[] = {"mbarrier_address", "mbarrier",
                                 "object", "mbar"};
          for (auto *n : names) {
            auto i = d.uniform.find(n);
            if (i != d.uniform.end()) return &i->second;
          }
          return (const std::uint64_t *)nullptr;
        }())
      addr = *v;
    else if (!d.memory.empty() && !d.memory[0].addrs.empty())
      addr = d.memory[0].addrs[0];
    else
      throw cyclescope_convert::ConversionError("missing_fact:mbarrier_address (" + where +
                            ")");
    std::uint64_t mask = 0;
    if (has("MULTICAST")) {
      auto it = d.uniform.find("multicast_mask");
      if (it == d.uniform.end())
        throw cyclescope_convert::ConversionError("missing_fact:umma_commit_mask (" +
                              where + ")");
      mask = it->second;
    }
    SemRow r;
    r.kind = SemKind::umma_commit;
    r.fields["object"] = cyclescope_convert::resolve_mbarrier(d, ctx, cta, addr, where.c_str());
    r.fields["cta_group"] = has("2CTA") ? 2 : 1;
    r.fields["commit_mask"] = mask;
    r.fields["scope"] = (has("2CTA") || has("MULTICAST")) ? 2 : 1;
    return r;
  }
  if (op == SemOp::tmem_alloc || op == SemOp::tmem_dealloc ||
      op == SemOp::tmem_relinquish) {
    const auto aop = tmem_alloc_op(opc);
    if (aop.empty())
      throw cyclescope_convert::ConversionError("unsupported:tmem_alloc_op (" + where + ")");
    std::uint64_t columns = 0;
    if (aop == "alloc" || aop == "dealloc") {
      auto it = d.uniform.find("tmem_columns");
      if (it != d.uniform.end()) {
        columns = it->second;
      } else {
        static const std::set<std::uint64_t> colset = {32, 64, 128,
                                                       256, 512};
        auto iu = d.uniform.find("tmem_alloc_ur");
        if (iu != d.uniform.end()) {
          if (colset.count(iu->second)) columns = iu->second;
          else if (colset.count(iu->second * 32))
            columns = iu->second * 32;
          else columns = UINT64_MAX;
        } else columns = UINT64_MAX;
      }
      if (columns == UINT64_MAX) {
        if (aop == "dealloc") columns = SEMA_UNKNOWN;
        else
          throw cyclescope_convert::ConversionError("missing_fact:tmem_columns (" + where +
                                ")");
      }
    }
    SemRow r;
    r.kind = (aop == "alloc" ? SemKind::tmem_alloc
             : aop == "dealloc" ? SemKind::tmem_dealloc
                               : SemKind::tmem_relinquish);
    r.fields["columns"] = columns;
    return r;
  }
  if (op == SemOp::tmem_load || op == SemOp::tmem_store ||
      op == SemOp::tmem_copy) {
    auto [shape, repeat] = tmem_shape_repeat(opc, where);
    SemRow r;
    r.kind = (op == SemOp::tmem_load ? SemKind::tmem_load
             : op == SemOp::tmem_store ? SemKind::tmem_store
                                      : SemKind::tmem_copy);
    if (has("STAT")) r.flags = SEMA_FLAG_STAT;
    r.fields["shape"] = shape;
    r.fields["repeat"] = repeat;
    return r;
  }
  if (op == SemOp::tmem_wait_store) {
    SemRow r;
    r.kind = SemKind::tmem_wait_store;
    return r;
  }
  if (op == SemOp::clc_query) {
    std::uint64_t addr = 0;
    bool found = false;
    if (auto *v = [&] {
          const char *names[] = {"mbarrier_address", "mbarrier",
                                 "object", "mbar"};
          for (auto *n : names) {
            auto i = d.uniform.find(n);
            if (i != d.uniform.end()) return &i->second;
          }
          return (const std::uint64_t *)nullptr;
        }()) {
      addr = *v;
      found = true;
    } else if (d.memory.size() > 1 && !d.memory[1].addrs.empty()) {
      addr = d.memory[1].addrs[0];
      found = true;
    } else if (!d.memory.empty() && !d.memory[0].addrs.empty()) {
      addr = d.memory[0].addrs[0];
      found = true;
    }
    if (!found)
      throw cyclescope_convert::ConversionError("missing_fact:clc_mbarrier_address (" +
                            where + ")");
    std::uint64_t mask;
    if (opc.find("BROADCAST") != std::string::npos)
      mask = ctx.n_ctas ? (1ull << ctx.n_ctas) - 1 : 0;
    else if (opc.find("SELFCAST") != std::string::npos)
      mask = 1;
    else
      throw cyclescope_convert::ConversionError("missing_fact:clc_multicast_class (" +
                            where + ")");
    SemRow r;
    r.kind = SemKind::clc_query;
    r.fields["object"] = cyclescope_convert::resolve_mbarrier(d, ctx, cta, addr, where.c_str());
    r.fields["generation"] = GEN_UNRESOLVED;
    r.fields["destination_mask"] = mask;
    r.fields["response_identity"] = 0;
    r.fields["outcome"] = 0;
    return r;
  }
  return {};
}

}  // namespace cyclescope_convert

// ---- opcode_facts + helper exports for axes --------------------------
namespace {
const std::regex kMatrixRe(
    R"((?:LDSM|STSM)\.(\d+)\.M(T)?(\d)(\d)(?:\.(\d+))?(?:\..*)?)");
}

namespace cyclescope_convert {

std::map<std::string, long long> umma_idesc_facts(
    const std::string &opc, std::uint64_t v) {
  const auto f = umma_descriptor_facts(upper(opc), v);
  auto mods = mods_of(upper(opc));
  const bool ws = std::find(mods.begin(), mods.end(), "WS") != mods.end();
  const bool sparse = v & (1 << 2);
  const bool b16 = std::find(mods.begin(), mods.end(), "BLOCK16") !=
                   mods.end();
  const auto root = toks_of(upper(opc))[0];
  std::string operation;
  if (ws && sparse) operation = "ws_sp";
  else if (ws) operation = "ws";
  else if (sparse) operation = "sp";
  else operation = "normal";
  const int trans_a = (v >> 15) & 1, trans_b = (v >> 16) & 1;
  const int major = 2 * trans_a + trans_b;
  const int max_shift = ((v >> 30) & 3) *
      ((f.kind == "f16" || f.kind == "tf32" || f.kind == "f8f6f4") ? 1
                                                                  : 0);
  return {{"tensor_m", f.tensor_m}, {"tensor_n", f.tensor_n},
          {"tensor_k", f.tensor_k}, {"umma_operand_major", major},
          {"umma_max_shift", max_shift}, {"umma_operation",
                                          operation == "ws_sp" ? 3
                                             : operation == "ws" ? 2
                                             : operation == "sp" ? 1
                                                                 : 0}};
}

std::string umma_kind_of_axis(const std::string &opc) {
  return umma_kind_of(upper(opc));
}

std::string umma_kind_of_idesc(const std::string &opc,
                               std::uint64_t v) {
  return umma_descriptor_facts(upper(opc), v).kind;
}

std::string umma_operation_of_str(
    const std::map<std::string, long long> &f) {
  static const char *names[] = {"normal", "sp", "ws", "ws_sp"};
  auto it = f.find("umma_operation");
  return it == f.end() ? "normal" : names[it->second & 3];
}

std::pair<int, int> blackwell_tmem_shape_repeat(const std::string &opc,
                                                bool &ok) {
  try {
    ok = true;
    return tmem_shape_repeat(upper(opc), "axis");
  } catch (...) {
    ok = false;
    return {0, 0};
  }
}

std::map<std::string, long long>
blackwell_opcode_facts(const std::string &opcode) {
  std::map<std::string, long long> f;
  const auto op = upper(opcode);
  const auto root = op.substr(0, op.find('.'));
  std::smatch m;
  if (root == "LDSM" || root == "STSM") {
    if (std::regex_match(op, m, kMatrixRe)) {
      f["width_bytes"] = std::stoll(m[1]) / 8;
      f["tensor_m"] = std::stoll(m[3]);
      f["tensor_n"] = std::stoll(m[4]);
      f["matrix_count"] = m[5].matched ? std::stoll(m[5]) : 1;
      f["matrix_transposed"] = m[2].matched ? 1 : 0;
    }
  }
  if (root == "UTMALDG" || root == "UTMASTG" || root == "UTMAREDG" ||
      root == "UTMAPF") {
    static const std::regex d_re(R"(\.(\d+)D(?:\.|$))");
    if (std::regex_search(op, m, d_re))
      f["tma_instruction_rank"] = std::stoll(m[1]);
  }
  static const std::regex s_re(R"((?:^|\.)(\d+)X(\d+)X(\d+)(?:\.|$))");
  if (std::regex_search(op, m, s_re)) {
    f["tensor_m"] = std::stoll(m[1]);
    f["tensor_n"] = std::stoll(m[2]);
    f["tensor_k"] = std::stoll(m[3]);
  }
    {
    const auto toks = split_dots(upper(opcode));
    f["tensor_sparse"] =
        std::find(toks.begin(), toks.end(), "SP") != toks.end() ? 1 : 0;
  }
return f;
}

}  // namespace cyclescope_convert
