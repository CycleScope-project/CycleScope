// Axis functions — 1:1 port of convert/axes.py.
#include "convert/axes.hpp"
#include "convert/str_util.hpp"

namespace cyclescope_convert {
std::pair<int, int> hopper_dtype_fields(const std::string &);
int hopper_source_mode(const std::vector<std::string> &, bool &);
std::pair<int, int> blackwell_tmem_shape_repeat(const std::string &,
                                                bool &);
std::string umma_operation_of_str(
    const std::map<std::string, long long> &);
std::map<std::string, long long> hopper_opcode_facts(const std::string &);
std::map<std::string, long long> blackwell_opcode_facts(
    const std::string &);
}

#include <algorithm>
#include <cctype>
#include <cstring>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>

namespace cyclescope_convert {
namespace {

std::string upper(std::string s) {
  for (auto &c : s) c = (char)std::toupper((unsigned char)c);
  return s;
}
std::string lower(std::string s) {
  for (auto &c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

std::vector<std::string> toks(const StaticInfo &st) {
  return st.operands;
}

int lanes_of(const DynOps *d) {
  return d ? __builtin_popcount(d->active_mask) : 32;
}

const std::uint64_t *uniform(const DynOps *d, const char *k) {
  if (!d) return nullptr;
  auto it = d->uniform.find(k);
  return it == d->uniform.end() ? nullptr : &it->second;
}

const std::vector<DynOps::Ref> empty_refs;
const std::vector<DynOps::Ref> &mem(const DynOps *d) {
  return d ? d->memory : empty_refs;
}
// references the memory hierarchy sees: for a mixed-space instruction
// (LDGSTS: global source + shared destination) only the non-shared side;
// otherwise all references (space codes: 1 shared, 6 shared_cluster)
std::vector<DynOps::Ref> hier(const DynOps *d) {
  const auto &m = mem(d);
  std::vector<DynOps::Ref> h;
  for (auto &r : m)
    if (r.space != 1 && r.space != 6) h.push_back(r);
  return (!h.empty() && h.size() != m.size()) ? h : m;
}

const std::regex kTensorRe(R"((?:^|\.)(\d+)X(\d+)X(\d+)(?:\.|$))");
// warp-level mma.sync shapes are packed digits: HMMA.16816 = m16n8k16,
// HMMA.1688 = m16n8k8, HMMA.884 = m8n8k4, IMMA.16832 = m16n8k32,
// BMMA.168256 = m16n8k256 (M in {8,16}, N always 8, K the remainder)
const std::regex kPackedMmaRe(R"(^(?:HMMA|IMMA|DMMA|BMMA)\.(\d{3,6})(?:\.|$))");

bool tensor_dims(const std::string &opc, long long dims[3]) {
  std::smatch m;
  if (std::regex_search(opc, m, kTensorRe)) {
    dims[0] = std::stoll(m[1]); dims[1] = std::stoll(m[2]);
    dims[2] = std::stoll(m[3]);
    return true;
  }
  if (!std::regex_search(opc, m, kPackedMmaRe)) return false;
  const std::string digits = m[1];
  const bool m16 = digits.rfind("16", 0) == 0;
  const std::string rest = digits.substr(m16 ? 2 : 1);
  if (rest.size() < 2 || rest[0] != '8') return false;
  dims[0] = m16 ? 16 : 8; dims[1] = 8; dims[2] = std::stoll(rest.substr(1));
  return true;
}
const std::regex kDtypeRe(R"(^(?:[BUSF]\d+|BF16|TF32|E4M3|E5M2)$)");
const std::regex kXnRe(R"(^x(\d+)$)", std::regex::icase);

std::vector<std::string> dtype_segments(const std::string &opc) {
  std::vector<std::string> out;
  const auto toks = split_delim(opc, '.');
  for (std::size_t i = 1; i < toks.size(); ++i) {
    const std::string &t = toks[i];
    if (std::regex_match(t, kDtypeRe)) out.push_back(lower(t));
  }
  return out;
}

// opcode_facts equivalents — arch rules return sparse maps
std::map<std::string, long long> opcode_facts_of(const StaticInfo &st,
                                                 AxisCtx &ctx) {
  if (!ctx.arch_bw) {
    if (ctx.arch == "hopper")
      return hopper_opcode_facts(st.opcode);
  } else return blackwell_opcode_facts(st.opcode);
  if (ctx.rules) return ctx.rules->opcode_facts(st.opcode);
  return {};
}

const std::set<std::string> kGmmaRoots = {"HGMMA", "WGMMA", "QGMMA",
                                        "IGMMA"};

std::pair<AxVal, AxVal> gmma_dtypes(const StaticInfo &st, AxisCtx &c) {
  const auto root = upper(st.opcode).substr(
      0, st.opcode.find('.'));
  if (!kGmmaRoots.count(root) || c.arch_bw)
    return {AxVal::none(), AxVal::none()};
  // hopper dtype_fields (semantics port)

  static const char *names[] = {"", "f16", "bf16", "tf32", "f32",
                                "s32", "e4m3", "e5m2", "f8", "u8",
                                "s8", "f64"};
  auto [inp, acc] = hopper_dtype_fields(upper(st.opcode));
  return {AxVal::sv(names[inp]), AxVal::sv(names[acc])};
}

const std::set<std::string> kSrcStorage = {
    "LD","ST","LDG","STG","LDL","STL","LDC","ULDC","LDCU","LDS","STS",
    "LDSM","STSM","LDGSTS","UTMALDG","UTMASTG","UTMAREDG","UTMAPF",
    "UBLKCP","UBLKRED","ATOM","ATOMG","ATOMS","RED","REDG","REDS",
    "STAS","LDTM","STTM","UTCATOMSWS","UTCCP","UVIRTCOUNT"};
const std::set<std::string> kSrcBranch = {"BRA","BRX","BRXU","CALL",
                                        "BSSY","BSYNC","BREAK","EXIT",
                                        "RET"};
const std::set<std::string> kTwoDest = {"FSETP","DSETP","ISETP",
                                        "UISETP"};

std::string source_class(std::string token) {
  // .REUSE strip, uppercase, lstrip("-+|~") (sign / abs / bitwise
  // not modifiers), rstrip("|")
  auto pos = token.find(".REUSE");
  if (pos != std::string::npos) token.erase(pos, 6);
  token = upper(token);
  while (!token.empty() &&
         (token.front() == '-' || token.front() == '+' ||
          token.front() == '|' || token.front() == '~'))
    token.erase(token.begin());
  while (!token.empty() && token.back() == '|') token.pop_back();
  static const std::regex rz(R"(^(?:RZ|URZ|SRZ)(?:\.[A-Z0-9_]+)*$)");
  static const std::regex rg(R"(^R\d+(?:\.[A-Z0-9_]+)*$)");
  static const std::regex ur(R"(^UR\d+(?:\.[A-Z0-9_]+)*$)");
  static const std::regex cx(R"(^C\[.*\](?:\[.*\])?$)");
  static const std::regex im(
      R"(^(?:0X[0-9A-F]+|(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)$)");
  static const std::regex pr(R"(^!?U?P(?:T|[0-6])$)");
  if (std::regex_match(token, rz)) return "zero";
  if (std::regex_match(token, rg)) return "gpr";
  if (std::regex_match(token, ur)) return "uniform";
  if (std::regex_match(token, cx)) return "constant";
  if (std::regex_match(token, im)) return "immediate";
  if (std::regex_match(token, pr)) return "predicate";
  return "";
}

struct SrcForm { std::map<std::string, int> counts; std::string mode; };
// -> SrcForm | na | none(unclassifiable)
bool source_form(const StaticInfo &s, SrcForm &out, AxVal &bad) {
  const auto root = upper(s.opcode).substr(0, s.opcode.find('.'));
  if (kSrcStorage.count(root) || kSrcBranch.count(root)) {
    bad = AxVal::na();
    return false;
  }
  const auto t = toks(s);
  const int dest = kTwoDest.count(root) ? 2 : 1;
  const char *names[] = {"gpr", "uniform", "immediate", "zero",
                         "constant"};
  std::map<std::string, int> counts;
  for (auto n : names) counts[n] = 0;
  for (std::size_t i = dest; i < t.size(); ++i) {
    auto cls = source_class(t[i]);
    if (cls.empty()) { bad = AxVal::none(); return false; }
    if (cls == "predicate") continue;
    counts[cls]++;
  }
  bool any = false;
  for (auto &[k, v] : counts) any |= v > 0;
  if (!any) { bad = AxVal::none(); return false; }
  std::vector<std::string> present;
  for (auto n : names)
    if (counts[n]) present.push_back(n);
  out.counts = counts;
  out.mode = present.size() == 1 ? present[0] : "mixed";
  return true;
}

AxVal source_form_part(const StaticInfo &s, const std::string &part) {
  SrcForm f;
  AxVal bad;
  if (!source_form(s, f, bad)) return bad;
  if (part == "mode") return AxVal::sv(f.mode);
  return AxVal::iv(f.counts[part]);
}

// ---- individual axes -------------------------------------------------

AxVal f_operation_class(const StaticInfo &s, const DynOps *, AxisCtx &c) {
  if (c.rules) return AxVal::sv(c.rules->operation_class(s));
  return AxVal::sv(upper(s.opcode).substr(0, s.opcode.find('.')));
}

AxVal f_datatype(const StaticInfo &s, const DynOps *, AxisCtx &) {
  auto t = dtype_segments(upper(s.opcode));
  if (t.empty()) return AxVal::sv("unspecified");
  std::string j = upper(t[0]);
  for (std::size_t i = 1; i < t.size(); ++i) j += "." + upper(t[i]);
  return AxVal::sv(j);
}

AxVal f_input_dtype(const StaticInfo &s, const DynOps *, AxisCtx &c) {
  auto p = gmma_dtypes(s, c);
  if (p.first.is_none()) return AxVal::na();
  return p.first;
}
AxVal f_accumulator_dtype(const StaticInfo &s, const DynOps *,
                          AxisCtx &c) {
  auto p = gmma_dtypes(s, c);
  if (p.second.is_none()) return AxVal::na();
  return p.second;
}



AxVal f_source_mode(const StaticInfo &s, const DynOps *, AxisCtx &c) {
  const auto root = upper(s.opcode).substr(0, s.opcode.find('.'));
  if (kGmmaRoots.count(root)) {
    if (c.arch_bw) return AxVal::na();
    bool ok = false;
    int m = hopper_source_mode(s.operands, ok);
    if (!ok) return AxVal::none();
    return AxVal::sv(m == 1 ? "ss" : "rs");
  }
  return source_form_part(s, "mode");
}

AxVal f_tensor_dim(const StaticInfo &s, const DynOps *d, AxisCtx &c,
                   int which) {
  const auto opc = upper(s.opcode);
  long long dims[3];
  if (tensor_dims(opc, dims))
    return AxVal::iv(dims[which - 1]);
  const char *key = which == 1 ? "tensor_m" : "tensor_n";
  auto f = opcode_facts_of(s, c);
  if (f.count(key)) return AxVal::iv(f[key]);
  if (auto *v = uniform(d, key)) return AxVal::iv((long long)*v);
  const auto root = opc.substr(0, opc.find('.'));
  if (root == "UTCHMMA" || root == "UTCQMMA" || root == "UTCOMMA") {
    if (auto *v = uniform(d, "idesc_value")) {
      auto uf = umma_idesc_facts(opc, *v);
      return AxVal::iv(uf[key]);
    }
    return AxVal::none();
  }
  return AxVal::none();
}

const std::map<std::string, int> kUmmaK = {
    {"f16", 16}, {"bf16", 16}, {"tf32", 8}, {"f8f6f4", 32},
    {"mxf8f6f4", 32}, {"mxf4", 64}, {"mxf4nvf4", 64}};

AxVal f_tensor_k(const StaticInfo &s, const DynOps *d, AxisCtx &) {
  const auto opc = upper(s.opcode);
  long long dims[3];
  if (tensor_dims(opc, dims))
    return AxVal::iv(dims[2]);
  const auto root = opc.substr(0, opc.find('.'));
  if (root != "UTCHMMA" && root != "UTCQMMA" && root != "UTCOMMA")
    return AxVal::none();
  if (auto *v = uniform(d, "idesc_value")) {
    auto uf = umma_idesc_facts(opc, *v);
    return AxVal::iv(uf["tensor_k"]);
  }
  auto kind = umma_kind_of_axis(opc);
  if (!kind.empty() && kUmmaK.count(kind))
    return AxVal::iv(kUmmaK.at(kind));
  return AxVal::none();
}

AxVal f_tensor_rank(const StaticInfo &s, const DynOps *d, AxisCtx &c) {
  const auto opc = upper(s.opcode);
  std::smatch m;
  static const std::regex re(R"(\.(\d)D)");
  if (std::regex_search(opc, m, re)) return AxVal::iv(std::stoll(m[1]));
  long long tm_id = -1;
  if (d && d->has_tensor_map) tm_id = d->tensor_map;
  else if (auto *v = uniform(d, "tensor_map")) tm_id = *v;
  if (tm_id >= 0 && c.tmaps) {
    auto it = c.tmaps->find((int)tm_id);
    if (it != c.tmaps->end())
      return AxVal::iv((long long)it->second.second.size());
  }
  return AxVal::none();
}

AxVal f_tma_form(const StaticInfo &s, const DynOps *, AxisCtx &) {
  const auto op = upper(s.opcode);
  if (op.find("IM2COL") != std::string::npos) return AxVal::sv("im2col");
  for (const char *f : {"GATHER4", "SCATTER4"})
    if (op.find(f) != std::string::npos) return AxVal::sv(lower(f));
  if (op.find("TILE") != std::string::npos) return AxVal::sv("tile");
  for (const char *r : {"1D", "2D", "3D", "4D", "5D"})
    if (op.find(r) != std::string::npos) return AxVal::sv("tile");
  return AxVal::none();
}

AxVal f_matrix_count(const StaticInfo &s, const DynOps *, AxisCtx &c) {
  auto f = opcode_facts_of(s, c);
  if (f.count("matrix_count")) return AxVal::iv(f["matrix_count"]);
  return AxVal::iv(upper(s.opcode).find("MMA") != std::string::npos);
}

AxVal f_matrix_transposed(const StaticInfo &s, const DynOps *,
                          AxisCtx &c) {
  auto f = opcode_facts_of(s, c);
  if (f.count("matrix_transposed"))
    return AxVal::iv(f["matrix_transposed"]);
  const auto op = upper(s.opcode);
  const auto tail = op.substr(op.find("MMA") == std::string::npos
                                  ? op.size() : op.find("MMA") + 3);
  return AxVal::iv(op.find("TN") != std::string::npos ||
                   tail.find(".T") != std::string::npos ||
                   op.find("TNSP") != std::string::npos);
}

AxVal f_execution_unit(const StaticInfo &s, const DynOps *,
                       AxisCtx &c) {
  auto f = opcode_facts_of(s, c);
  if (f.count("execution_unit")) return AxVal::iv(f["execution_unit"]);
  return AxVal::none();
}

AxVal f_branch_direction(const StaticInfo &s, const DynOps *,
                         AxisCtx &) {
  static const std::regex re(R"(^0x[0-9a-fA-F]+$)");
  for (auto it = s.operands.rbegin(); it != s.operands.rend(); ++it)
    if (std::regex_match(*it, re))
      return AxVal::sv(std::stoull(*it, nullptr, 16) < s.pc
                           ? "backward" : "forward");
  return AxVal::none();
}

AxVal f_branch_taken(const StaticInfo &s, const DynOps *,
                     AxisCtx &ctx) {
  if (!ctx.has_next_static_pc) return AxVal::iv(0);
  return AxVal::iv(ctx.next_static_pc == s.pc + 16 ? 0 : 1);
}

AxVal f_umma_kind(const StaticInfo &s, const DynOps *d, AxisCtx &) {
  const auto opc = upper(s.opcode);
  const auto root = opc.substr(0, opc.find('.'));
  if (root != "UTCHMMA" && root != "UTCQMMA" && root != "UTCOMMA")
    return AxVal::none();
  if (auto *v = uniform(d, "idesc_value")) {
    auto k = umma_kind_of_idesc(opc, *v);
    return k.empty() ? AxVal::none() : AxVal::sv(k);
  }
  auto k = umma_kind_of_axis(opc);
  return k.empty() ? AxVal::none() : AxVal::sv(k);
}

AxVal f_cta_group(const StaticInfo &s, const DynOps *d, AxisCtx &) {
  if (auto *v = uniform(d, "cta_group")) return AxVal::iv((long long)*v);
  return AxVal::iv(upper(s.opcode).find("2CTA") != std::string::npos
                       ? 2 : 1);
}

AxVal f_a_src(const StaticInfo &s, const DynOps *, AxisCtx &) {
  std::string text;
  for (auto &t : s.operands) text += lower(t) + " ";
  if (text.find("tmem[") != std::string::npos &&
      text.find("gdesc[") == std::string::npos)
    return AxVal::sv("tmem");
  if (text.find("gdesc[") != std::string::npos)
    return AxVal::sv("smem");
  return AxVal::none();
}

const std::set<std::string> kTmemShapeDomain = [] {
  std::set<std::string> v;
  for (int i : {1, 2, 4, 8, 16, 32, 64})
    v.insert("x" + std::to_string(i));
  v.insert("16dp256bit");
  v.insert("16dp32bit");
  v.insert("x128");
  return v;
}();



AxVal f_tmem_shape(const StaticInfo &s, const DynOps *, AxisCtx &c) {
  const auto opc = upper(s.opcode);
  const auto root = opc.substr(0, opc.find('.'));
  if (root != "LDTM" && root != "STTM" && root != "UTCCP")
    return AxVal::na();
  if (root == "UTCCP") {
    static const std::map<std::string, std::string> kTmemCopyShapes = [] {
      std::map<std::string, std::string> m;
      for (const std::string group : {"", ".2CTA"})
        for (const auto &[suffix, shape] :
             std::vector<std::pair<std::string, std::string>>{
                 {"", "copy_default"},
                 {".4DP256BIT", "4dp256bit"},
                 {".128DP128BIT", "128dp128bit"},
                 {".2X64DP128BIT_LW02_LW13", "2x64dp128bit_lw02_lw13"},
                 {".4X32DP128BIT", "4x32dp128bit"}})
          m["UTCCP.T.S" + group + suffix] = shape;
      return m;
    }();
    const auto it = kTmemCopyShapes.find(opc);
    return it == kTmemCopyShapes.end() ? AxVal::none()
                                       : AxVal::sv(it->second);
  }
  const auto toks = split_delim(opc, '.');
  for (std::size_t i = 1; i < toks.size(); ++i) {
    const std::string &t = toks[i];
    auto lt = lower(t);
    if (kTmemShapeDomain.count(lt)) return AxVal::sv(lt);
    {
      static const std::regex half_t_re(R"(_t\d+_t\d+$)");
      auto base = std::regex_replace(lt, half_t_re, std::string());
      if (kTmemShapeDomain.count(base)) return AxVal::sv(base);
    }
  }
  if (c.arch_bw) {
    bool ok = false;
    auto [code, rep] = blackwell_tmem_shape_repeat(opc, ok);
    if (ok && kTmemShapeDomain.count("x" + std::to_string(rep)))
      return AxVal::sv("x" + std::to_string(rep));
  }
  return AxVal::none();
}

AxVal f_tmem_repeat(const StaticInfo &s, const DynOps *, AxisCtx &c) {
  const auto opc = upper(s.opcode);
  const auto root = opc.substr(0, opc.find('.'));
  if (root != "LDTM" && root != "STTM" && root != "UTCCP")
    return AxVal::na();
  if (c.arch_bw) {
    bool ok = false;
    auto [code, rep] = blackwell_tmem_shape_repeat(opc, ok);
    if (ok) return AxVal::sv("x" + std::to_string(rep));
    return AxVal::none();
  }
  const auto toks = split_delim(opc, '.');
  for (std::size_t i = 1; i < toks.size(); ++i) {
    const std::string &t = toks[i];
    std::smatch m;
    if (std::regex_match(t, m, kXnRe))
      return AxVal::sv(lower(t));
  }
  return AxVal::sv("x1");
}

AxVal f_cache_policy(const StaticInfo &s, const DynOps *, AxisCtx &) {
  const auto opc = upper(s.opcode);
  const auto toks = split_delim(opc, '.');
  for (std::size_t i = 1; i < toks.size(); ++i) {
    const std::string &t = toks[i];
    if (t == "CA" || t == "CG") return AxVal::sv(lower(t));
  }
  return AxVal::none();
}

// umma_idesc axis value: Na when not UMMA, none when idesc missing
bool idesc_val(const StaticInfo &s, const DynOps *d,
               std::map<std::string, long long> &out, AxVal &bad) {
  const auto opc = upper(s.opcode);
  const auto root = opc.substr(0, opc.find('.'));
  if (root != "UTCHMMA" && root != "UTCQMMA" && root != "UTCOMMA") {
    bad = AxVal::na();
    return false;
  }
  auto *v = uniform(d, "idesc_value");
  if (!v) { bad = AxVal::none(); return false; }
  out = umma_idesc_facts(opc, *v);
  return true;
}

// ---- dynamic ---------------------------------------------------------

AxVal f_active_lanes(const StaticInfo &, const DynOps *d, AxisCtx &) {
  if (!d) return AxVal::none();
  return AxVal::iv(
      __builtin_popcount(d->active_mask & d->predicate_mask));
}
AxVal f_predicated_on(const StaticInfo &, const DynOps *d, AxisCtx &) {
  if (!d) return AxVal::none();
  return AxVal::iv((d->active_mask & d->predicate_mask) != 0);
}
AxVal f_memory_space(const StaticInfo &s, const DynOps *, AxisCtx &c) {
  if (c.rules)
    return AxVal::sv(c.rules->memory_space(s));
  return AxVal::none();
}
AxVal f_access_kind(const StaticInfo &, const DynOps *d, AxisCtx &) {
  if (!mem(d).empty()) {
    static const char *names[] = {"load", "store", "atomic",
                                  "async_copy", "async_reduce",
                                  "prefetch"};
    const int k = mem(d)[0].access_kind;
    return AxVal::sv(k >= 0 && k < 6 ? names[k] : "none");
  }
  return AxVal::sv("none");
}
AxVal f_memory_width(const StaticInfo &, const DynOps *d, AxisCtx &) {
  long long w = 0;
  for (auto &r : mem(d)) w = std::max(w, (long long)r.width);
  return mem(d).empty() ? AxVal::none() : AxVal::iv(w);
}
AxVal f_sector_count(const StaticInfo &, const DynOps *d, AxisCtx &) {
  const auto refs = hier(d);
  if (refs.empty()) return AxVal::none();
  long long n = 0;
  for (auto &r : refs) {
    std::set<long long> sec;
    for (auto a : r.addrs) sec.insert(a / 32);
    n += sec.size();
  }
  return AxVal::iv(n);
}
AxVal f_cache_line_count(const StaticInfo &, const DynOps *d,
                         AxisCtx &) {
  if (mem(d).empty()) return AxVal::none();
  std::set<std::pair<long long, int>> lines;
  for (auto &r : mem(d))
    for (auto a : r.addrs) lines.emplace(a / 128, r.space);
  return AxVal::iv((long long)lines.size());
}
AxVal f_distinct_addresses(const StaticInfo &, const DynOps *d,
                           AxisCtx &) {
  const auto refs = hier(d);
  if (refs.empty()) return AxVal::none();
  std::set<long long> a;
  for (auto &r : refs) a.insert(r.addrs.begin(), r.addrs.end());
  return AxVal::iv((long long)a.size());
}
AxVal f_address_stride(const StaticInfo &, const DynOps *d, AxisCtx &) {
  if (mem(d).empty() || mem(d)[0].addrs.empty()) return AxVal::none();
  auto a = mem(d)[0].addrs;
  std::sort(a.begin(), a.end());
  if (a.size() < 2) return AxVal::iv(0);
  return AxVal::iv(a[1] - a[0]);
}
AxVal f_address_pattern(const StaticInfo &, const DynOps *d,
                        AxisCtx &) {
  if (mem(d).empty() || mem(d)[0].addrs.empty()) return AxVal::none();
  const auto &a = mem(d)[0].addrs;
  if (std::set<long long>(a.begin(), a.end()).size() == 1)
    return AxVal::sv("same");
  std::set<long long> diffs;
  for (std::size_t i = 1; i < a.size(); ++i)
    diffs.insert(a[i] - a[i - 1]);
  return AxVal::sv(diffs.size() == 1 ? "strided" : "pseudo_random");
}
AxVal f_working_set(const StaticInfo &, const DynOps *d, AxisCtx &) {
  const auto refs = hier(d);
  if (refs.empty()) return AxVal::none();
  std::set<long long> lines;
  for (auto &r : refs)
    for (auto a : r.addrs) lines.insert(a / 128);
  return AxVal::iv((long long)lines.size() * 128);
}

const std::regex kBankMatrixRe(
    R"((?:LDSM|STSM)\.\d+\.MT?\d\d(?:\.(\d+))?(?:\..*)?)");

int bank_degree_group(const std::string &opc, bool is_load, int raw_w,
                      const std::vector<int> &ids,
                      const std::vector<long long> &addrs) {
  std::smatch m;
  if (std::regex_match(opc, m, kBankMatrixRe)) {
    const int count = m[1].matched ? std::stoi(m[1]) : 1;
    std::map<int, std::map<int, int>> trans;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] >= 8 * count) continue;
      const int fw = (int)(addrs[i] / 4);
      auto &cnt = trans[ids[i] / 8];
      for (int w = fw; w < fw + 4; ++w) cnt[w % 32]++;
    }
    int best = 0;
    for (auto &[_, c] : trans)
      for (auto &[_, n] : c) best = std::max(best, n);
    return best;
  }
  std::vector<int> banks;
  for (auto a : addrs) banks.push_back((int)((a / 4) % 32));
  if (is_load && raw_w <= 4) {
    std::map<int, std::set<long long>> wb;
    for (std::size_t i = 0; i < addrs.size(); ++i)
      wb[banks[i]].insert(addrs[i] / 4);
    int best = 0;
    for (auto &[_, w] : wb) best = std::max(best, (int)w.size());
    return best;
  }
  std::map<int, int> cnt;
  for (auto b : banks) cnt[b]++;
  int best = 0;
  for (auto &[_, n] : cnt) best = std::max(best, n);
  return best;
}

AxVal f_bank_conflict(const StaticInfo &s, const DynOps *d,
                      AxisCtx &) {
  std::vector<const DynOps::Ref *> refs;
  bool has_shared = false;
  for (auto &r : mem(d)) {
    if (r.space != 1 && r.space != 6) continue;
    has_shared = true;
    if (!r.addrs.empty()) refs.push_back(&r);
  }
  if (!mem(d).empty() && !has_shared) return AxVal::na();
  if (refs.empty()) return AxVal::none();
  const auto opc = upper(s.opcode);
  int worst = 0;
  for (auto *ref : refs) {
    const int raw_w = ref->width ? ref->width : 4;
    const bool is_load = ref->access_kind == 0;
    std::vector<int> ids;
    if (ref->lane_mask) {
      for (int i = 0; i < 32; ++i)
        if ((ref->lane_mask >> i) & 1) ids.push_back(i);
    } else {
      for (std::size_t i = 0; i < ref->addrs.size(); ++i)
        ids.push_back((int)i);
    }
    if (std::set<long long>(ref->addrs.begin(),
                            ref->addrs.end()).size() == 1 &&
        is_load) {
      worst = std::max(worst, 1);
      continue;
    }
    std::map<int, std::vector<std::pair<int, long long>>> by_rank;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const int rank = (int)((ref->addrs[i] >> 24) & 0xFF);
      by_rank[rank].emplace_back(ids[i], ref->addrs[i]);
    }
    for (auto &[_, rows] : by_rank) {
      std::vector<int> ri;
      std::vector<long long> ra;
      for (auto &[id, a] : rows) {
        ri.push_back(id);
        ra.push_back(a & 0xFFFFFF);
      }
      worst = std::max(worst,
                       bank_degree_group(opc, is_load, raw_w, ri, ra));
    }
  }
  return AxVal::iv(worst);
}

// 128 B wavefronts of one rank group (mirror of axes.py
// _wavefronts_group): words touched by every lane mapped to 32 x 4 B
// banks, wavefronts = max over banks of the distinct words on that
// bank; LDSM/STSM per 8-lane transaction (4 words per row), summed.
int wavefronts_group(const std::string &opc, int raw_w,
                     const std::vector<int> &ids,
                     const std::vector<long long> &addrs) {
  static const std::regex matrix_re(
      R"((?:LDSM|STSM)\.\d+\.MT?\d\d(?:\.(\d+))?(?:\..*)?)");
  std::smatch m;
  const int words_per_lane = std::max(1, raw_w / 4);
  if (std::regex_match(opc, m, matrix_re)) {
    const int count = m[1].matched ? std::stoi(m[1].str()) : 1;
    std::map<int, std::map<int, std::set<long long>>> per_tx;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] >= 8 * count) continue;
      const long long first = addrs[i] / 4;
      auto &banks = per_tx[ids[i] / 8];
      for (long long w = first; w < first + 4; ++w)
        banks[(int)(w % 32)].insert(w);
    }
    int total = 0;
    for (auto &[_, banks] : per_tx) {
      int mx = 0;
      for (auto &[__, words] : banks) mx = std::max(mx, (int)words.size());
      total += mx;
    }
    return total;
  }
  std::map<int, std::set<long long>> banks;
  for (auto a : addrs) {
    const long long first = a / 4;
    for (long long w = first; w < first + words_per_lane; ++w)
      banks[(int)(w % 32)].insert(w);
  }
  int mx = 0;
  for (auto &[_, words] : banks) mx = std::max(mx, (int)words.size());
  return mx;
}

AxVal f_shared_wavefronts(const StaticInfo &s, const DynOps *d,
                          AxisCtx &) {
  std::vector<const DynOps::Ref *> refs;
  bool has_shared = false;
  for (auto &r : mem(d)) {
    if (r.space != 1 && r.space != 6) continue;
    has_shared = true;
    if (!r.addrs.empty()) refs.push_back(&r);
  }
  if (!mem(d).empty() && !has_shared) return AxVal::na();
  if (refs.empty()) return AxVal::none();
  const auto opc = upper(s.opcode);
  int worst = 0;
  for (auto *ref : refs) {
    const int raw_w = ref->width ? ref->width : 4;
    std::vector<int> ids;
    if (ref->lane_mask) {
      for (int i = 0; i < 32; ++i)
        if ((ref->lane_mask >> i) & 1) ids.push_back(i);
    } else {
      for (std::size_t i = 0; i < ref->addrs.size(); ++i)
        ids.push_back((int)i);
    }
    std::map<int, std::vector<std::pair<int, long long>>> by_rank;
    for (std::size_t i = 0; i < ids.size(); ++i)
      by_rank[(int)((ref->addrs[i] >> 24) & 0xFF)].emplace_back(
          ids[i], ref->addrs[i] & 0xFFFFFF);
    for (auto &[_, rows] : by_rank) {
      std::vector<int> ri;
      std::vector<long long> ra;
      for (auto &[id, a] : rows) { ri.push_back(id); ra.push_back(a); }
      worst = std::max(worst, wavefronts_group(opc, raw_w, ri, ra));
    }
  }
  return AxVal::iv(worst);
}

// atomic / async_reduce: largest number of lanes on one address
AxVal f_atomic_same_address_lanes(const StaticInfo &, const DynOps *d,
                                  AxisCtx &) {
  int worst = 0;
  bool any = false;
  for (auto &r : mem(d)) {
    if (r.addrs.empty() || (r.access_kind != 2 && r.access_kind != 4))
      continue;
    any = true;
    std::map<long long, int> counts;
    for (auto a : r.addrs) ++counts[a];
    for (auto &[_, c] : counts) worst = std::max(worst, c);
  }
  return any ? AxVal::iv(worst) : AxVal::none();
}

AxVal f_remote_smem(const StaticInfo &, const DynOps *d, AxisCtx &) {
  if (mem(d).empty()) return AxVal::iv(0);
  for (auto &r : mem(d))
    if (r.remote_rank >= 0) return AxVal::iv(1);
  return AxVal::iv(0);
}

AxVal f_bulk_direction(const StaticInfo &s, const DynOps *d,
                          AxisCtx &) {
  (void)d;
  const auto op = upper(s.opcode);
  const auto root = op.substr(0, op.find('.'));
  if (root != "UBLKCP" && root != "UBLKRED") return AxVal::none();
  const auto mods = op.substr(root.size() + 1);
  if (mods.rfind("G", 0) == 0) return AxVal::sv("store");
  if (mods.rfind("S", 0) == 0) return AxVal::sv("load");
  return AxVal::none();
}

AxVal f_destination_count(const StaticInfo &s, const DynOps *d,
                          AxisCtx &) {
  for (auto &r : mem(d)) {
    if (r.destination_count >= 0)
      return AxVal::iv(r.destination_count);
    if (r.multicast_mask)
      return AxVal::iv(__builtin_popcount(r.multicast_mask));
  }
  if (auto *v = uniform(d, "multicast_mask"))
    return AxVal::iv(__builtin_popcountll(*v));
  const auto root = upper(s.opcode).substr(0, s.opcode.find('.'));
  static const std::set<std::string> r1 = {"LDGSTS","UTMALDG","UTMASTG",
      "UBLKCP","UBLKRED","UTMAREDG","STAS","ST","STG","STS","LD","LDS"};
  if (r1.count(root)) return AxVal::iv(1);
  return AxVal::none();
}

AxVal f_transfer_bytes(const StaticInfo &, const DynOps *d, AxisCtx &) {
  if (mem(d).empty()) {
    if (auto *v = uniform(d, "copy_bytes"))
      return AxVal::iv((long long)*v);
    return AxVal::none();
  }
  long long t = 0;
  for (auto &r : mem(d))
    t += r.transfer_bytes >= 0 ? r.transfer_bytes
                              : (long long)r.width * r.addrs.size();
  return AxVal::iv(t);
}

AxVal f_cut_rows(const StaticInfo &, const DynOps *d, AxisCtx &) {
  // tma_oob v2: rows of the box left in bounds when a tiled TMA load
  // is cut along dimension 0; 0 otherwise
  long long t = 0;
  for (auto &r : mem(d))
    if (r.cut_rows > 0) t += r.cut_rows;
  return AxVal::iv(t);
}

AxVal f_expected_arrivals(const StaticInfo &, const DynOps *d,
                          AxisCtx &) {
  if (auto *v = uniform(d, "expected_arrivals"))
    return AxVal::iv((long long)*v);
  if (auto *v = uniform(d, "arrival_count"))
    return AxVal::iv((long long)*v);
  return AxVal::na();
}
AxVal f_expected_tx(const StaticInfo &, const DynOps *d, AxisCtx &) {
  if (auto *v = uniform(d, "expected_tx_bytes"))
    return AxVal::iv((long long)*v);
  return AxVal::none();
}

AxVal f_wait_group(const StaticInfo &s, const DynOps *d, AxisCtx &c) {
  const auto root = upper(s.opcode).substr(0, s.opcode.find('.'));
  if (root == "DEPBAR") {
    // depbar keep = first numeric/{..0..} token else uniform N
    long long keep = -1;
    for (auto &t : s.operands) {
      if (t.size() > 2 && t.front() == '{' && t.back() == '}' &&
          t.find('0') != std::string::npos) {
        keep = 0;
        break;
      }
      char *e = nullptr;
      auto v = std::strtoll(t.c_str(), &e, 0);
      if (e && *e == 0) keep = v;
    }
    if (keep < 0) {
      if (auto *v = uniform(d, "pending_allowed"))
        keep = *v;
      else if (auto *v = uniform(d, "N"))
        keep = *v;
    }
    return keep < 0 ? AxVal::none() : AxVal::iv(keep);
  }
  if (root == "WARPGROUP") {
    if (auto *v = uniform(d, "depbar_keep"))
      return AxVal::iv((long long)*v);
    return AxVal::none();
  }
  return AxVal::na();
}

AxVal f_copies_per_group(const StaticInfo &, const DynOps *d,
                         AxisCtx &c) {
  if (auto *v = uniform(d, "copies_per_group"))
    return AxVal::iv((long long)*v);
  if (c.flow) return AxVal::iv(c.flow->cpasync_count);
  return AxVal::none();
}

AxVal f_requested_sleep(const StaticInfo &s, const DynOps *d,
                        AxisCtx &) {
  if (auto *v = uniform(d, "requested_ns"))
    return AxVal::iv((long long)*v);
  for (auto &t : s.operands) {
    char *e = nullptr;
    auto v = std::strtoll(t.c_str(), &e, 0);
    if (e && *e == 0) return AxVal::iv(v);
  }
  return AxVal::none();
}

AxVal f_match_distinct(const StaticInfo &, const DynOps *d, AxisCtx &) {
  if (!d) return AxVal::none();
  for (auto &[k, vals] : d->lane_values) {
    std::set<long long> u(vals.begin(), vals.end());
    return AxVal::iv((long long)u.size());
  }
  return AxVal::none();
}
AxVal f_same_body_reuse(const StaticInfo &, const DynOps *d,
                        AxisCtx &) {
  if (auto *v = uniform(d, "same_body_reuse"))
    return AxVal::iv((long long)*v);
  return AxVal::none();
}
AxVal f_transaction_count(const StaticInfo &, const DynOps *d,
                          AxisCtx &) {
  if (auto *v = uniform(d, "transaction_count"))
    return AxVal::iv((long long)*v);
  return AxVal::none();
}
AxVal f_operand_facts(const StaticInfo &, const DynOps *d, AxisCtx &) {
  return AxVal::iv(d && !d->uniform.empty());
}

// ---- launch ----------------------------------------------------------

AxVal f_launch_warps(const StaticInfo &, const DynOps *, AxisCtx &c) {
  if (c.block[0]) {
    const long long t = (long long)c.block[0] * c.block[1] * c.block[2];
    return AxVal::iv(t / 32);
  }
  return AxVal::none();
}
AxVal f_regs_per_thread(const StaticInfo &, const DynOps *,
                        AxisCtx &c) {
  return c.registers_per_thread >= 0
             ? AxVal::iv(c.registers_per_thread)
             : AxVal::none();
}
AxVal f_declared_warpgroups(const StaticInfo &, const DynOps *,
                            AxisCtx &c) {
  return c.declared_math_warpgroups >= 0
             ? AxVal::iv(c.declared_math_warpgroups)
             : AxVal::none();
}

// ---- flow ctx --------------------------------------------------------

const std::map<std::string, std::string> kConsumerClass = {
    {"FMUL", "fp32"}, {"FADD", "fp32"}, {"FFMA", "fp32"},
    {"PRMT", "integer"}, {"IMAD", "integer"},
    {"MUFU", "special_function"}};

AxVal f_dependency_consumer_class(const StaticInfo &, const DynOps *,
                                  AxisCtx &c) {
  if (!c.flow || !c.flow->reader.first || !c.statics)
    return AxVal::na();
  const auto &rs = (*c.statics)[c.flow->reader.second];
  const auto root = upper(rs.opcode).substr(0, rs.opcode.find('.'));
  auto it = kConsumerClass.find(root);
  return it == kConsumerClass.end() ? AxVal::na()
                                  : AxVal::sv(it->second);
}
AxVal f_dependency_source_role(const StaticInfo &, const DynOps *,
                               AxisCtx &c) {
  if (!c.flow || !c.flow->reader.first) return AxVal::na();
  if (c.flow->reader_role_none) return AxVal::none();
  return AxVal::sv(c.flow->reader_role);
}
AxVal f_shared_data_domain(const StaticInfo &, const DynOps *,
                           AxisCtx &c) {
  if (!c.flow || !c.flow->reader.first) return AxVal::na();
  if (c.flow->reader_role_none) return AxVal::none();
  return AxVal::iv(c.flow->reader_role == "data");
}

AxVal f_group_member_count(const StaticInfo &, const DynOps *d,
                           AxisCtx &) {
  if (auto *v = uniform(d, "group_member_count"))
    return AxVal::iv((long long)*v);
  return AxVal::none();
}

AxVal f_control_path_count(const StaticInfo &, const DynOps *d,
                           AxisCtx &) {
  if (d) {
    if (uniform(d, "control_selector_present")) {
      if (auto *v = uniform(d, "sequential_region_count"))
        return AxVal::iv((long long)*v);
      return AxVal::none();
    }
    if (auto *v = uniform(d, "control_path_count"))
      return AxVal::iv((long long)*v);
  }
  return AxVal::na();
}
AxVal f_control_key(const std::string &name, const DynOps *d,
                    AxisCtx &) {
  if (auto *v = uniform(d, name.c_str()))
    return AxVal::iv((long long)*v);
  return AxVal::none();
}

}  // namespace

// dispatch table — same key order semantics as python dict
using AxFn = AxVal (*)(const StaticInfo &, const DynOps *, AxisCtx &);

AxVal axis_value(const std::string &name, const StaticInfo &st,
                 const DynOps *d, AxisCtx &ctx) {
  static const std::map<std::string, AxFn> fns = {
      {"a_src", f_a_src},
      {"access_kind", f_access_kind},
      {"accumulator_dtype", f_accumulator_dtype},
      {"active_lanes", f_active_lanes},
      {"active_lanes_per_path", f_active_lanes},
      {"address_pattern", f_address_pattern},
      {"address_stride_bytes", f_address_stride},
      {"backward_loop_class",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         if (d) {
           if (d) {
             auto it2 = d->uniform_str.find("backward_loop_class_s");
             if (it2 != d->uniform_str.end())
               return AxVal::sv(it2->second);
           }
         }
         return f_control_key("backward_loop_class", d, c);
       }},
      {"bank_conflict_degree", f_bank_conflict},
      {"shared_wavefronts", f_shared_wavefronts},
      {"atomic_same_address_lanes", f_atomic_same_address_lanes},
      {"branch_direction", f_branch_direction},
      {"branch_taken", f_branch_taken},
      {"cache_line_count", f_cache_line_count},
      {"cache_policy", f_cache_policy},
      {"constant_source_count",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return source_form_part(s, "constant");
       }},
      {"control_participating_warps",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return f_control_key("control_participating_warps", d, c);
       }},
      {"control_path_count", f_control_path_count},
      {"copies_per_group", f_copies_per_group},
      {"cta_group", f_cta_group},
      {"datatype", f_datatype},
      {"dependency_consumer_class", f_dependency_consumer_class},
      {"dependency_source_role", f_dependency_source_role},
      {"bulk_direction", f_bulk_direction},
      {"destination_count", f_destination_count},
      {"distinct_addresses", f_distinct_addresses},
      {"execution_unit", f_execution_unit},
      {"expected_arrivals", f_expected_arrivals},
      {"expected_tx_bytes", f_expected_tx},
      {"gpr_source_count",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return source_form_part(s, "gpr");
       }},
      {"group_member_count", f_group_member_count},
      {"immediate_source_count",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return source_form_part(s, "immediate");
       }},
      {"input_dtype", f_input_dtype},
      {"launch_warps_per_cta", f_launch_warps},
      {"loop_body_bytes",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return f_control_key("loop_body_bytes", d, c);
       }},
      {"loop_instruction_count",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return f_control_key("loop_instruction_count", d, c);
       }},
      {"match_distinct_values", f_match_distinct},
      {"matrix_count", f_matrix_count},
      {"matrix_transposed", f_matrix_transposed},
      {"memory_space", f_memory_space},
      {"memory_width_bytes", f_memory_width},
      {"operand_facts_available", f_operand_facts},
      {"operation_class", f_operation_class},
      {"predicated_on", f_predicated_on},
      {"registers_per_thread", f_regs_per_thread},
      {"remote_smem", f_remote_smem},
      {"requested_sleep_cycles", f_requested_sleep},
      {"same_body_reuse", f_same_body_reuse},
      {"sector_count", f_sector_count},
      {"shared_data_result_domain", f_shared_data_domain},
      {"source_mode", f_source_mode},
      {"tensor_k", f_tensor_k},
      {"tensor_m",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return f_tensor_dim(s, d, c, 1);
       }},
      {"tensor_n",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return f_tensor_dim(s, d, c, 2);
       }},
      {"tensor_sparse",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         auto f = opcode_facts_of(s, c);
         if (f.count("tensor_sparse"))
           return AxVal::iv(f["tensor_sparse"]);
         return AxVal::iv(0);
       }},
      {"tensor_rank", f_tensor_rank},
      {"tma_box_planes", f_tensor_rank},
      {"tma_form", f_tma_form},
      {"tma_instruction_rank",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         auto f = opcode_facts_of(s, c);
         if (f.count("tma_instruction_rank"))
           return AxVal::iv(f["tma_instruction_rank"]);
         return f_tensor_rank(s, d, c);
       }},
      {"tmem_repeat", f_tmem_repeat},
      {"tmem_shape", f_tmem_shape},
      {"transaction_count", f_transaction_count},
      {"cut_rows", f_cut_rows},
      {"transfer_bytes", f_transfer_bytes},
      {"umma_kind", f_umma_kind},
      {"umma_max_shift",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         std::map<std::string, long long> f;
         AxVal bad;
         if (!idesc_val(s, d, f, bad)) return bad;
         return AxVal::iv(f["umma_max_shift"]);
       }},
      {"umma_operand_major",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         std::map<std::string, long long> f;
         AxVal bad;
         if (!idesc_val(s, d, f, bad)) return bad;
         return AxVal::iv(f["umma_operand_major"]);
       }},
      {"umma_operation",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         std::map<std::string, long long> f;
         AxVal bad;
         if (!idesc_val(s, d, f, bad)) return bad;
         return AxVal::sv(umma_operation_of_str(f));
       }},
      {"uniform_source_count",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return source_form_part(s, "uniform");
       }},
      {"wait_group", f_wait_group},
      {"width_bytes", f_memory_width},
      {"working_set_bytes", f_working_set},
      {"zero_source_count",
       +[](const StaticInfo &s, const DynOps *d, AxisCtx &c) -> AxVal {
         return source_form_part(s, "zero");
       }},
  };
  static const std::set<std::string> exec_axes = {
      "access_kind", "address_pattern", "address_stride_bytes",
      "atomic_same_address_lanes", "bank_conflict_degree",
      "cache_line_count", "copies_per_group", "shared_wavefronts",
      "dependency_consumer_class", "dependency_source_role",
      "bulk_direction", "destination_count", "distinct_addresses",
      "match_distinct_values", "memory_width_bytes", "remote_smem",
      "cut_rows", "sector_count",
      "shared_data_result_domain", "transfer_bytes",
      "width_bytes", "working_set_bytes"};
  // axes whose value depends only on the static row: none read DynOps
  // and none read flow/next_static_pc/tensor-map state (verified per
  // body); ctx reads are limited to ctor-constant fields (rules, arch,
  // block, registers_per_thread, declared_math_warpgroups)
  static const std::set<std::string> static_axes = {
      "a_src",
      "accumulator_dtype",
      "branch_direction",
      "cache_policy",
      "constant_source_count",
      "datatype",
      "declared_math_warpgroups",
      "execution_unit",
      "gpr_source_count",
      "immediate_source_count",
      "input_dtype",
      "launch_warps_per_cta",
      "matrix_count",
      "matrix_transposed",
      "memory_space",
      "operation_class",
      "registers_per_thread",
      "source_mode",
      "tma_form",
      "tmem_repeat",
      "tmem_shape",
      "uniform_source_count",
      "zero_source_count"};
  // one lookup per call: (fn, exec-only, static-memoised) per axis name
  struct Entry { AxFn fn; bool exec; bool is_static; };
  static const std::unordered_map<std::string, Entry> table = [] {
    std::unordered_map<std::string, Entry> t;
    for (const auto &[n, fn] : fns)
      t.emplace(n, Entry{fn, exec_axes.count(n) != 0,
                         static_axes.count(n) != 0});
    return t;
  }();
  const auto it = table.find(name);
  if (it == table.end())
    throw std::runtime_error("axis function not registered: " + name);
  const Entry &e = it->second;
  if (d && e.exec && (d->active_mask & d->predicate_mask) == 0)
    return AxVal::na();
  if (e.is_static) {
    auto& m = (*ctx.static_axis_cache)[name];
    auto ci = m.find(st.pc);
    if (ci != m.end()) return ci->second;
    return m[st.pc] = e.fn(st, d, ctx);
  }
  return e.fn(st, d, ctx);
}

}  // namespace cyclescope_convert
