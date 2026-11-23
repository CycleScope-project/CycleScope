// Mirrors python/cyclescope/convert/operand_facts.py + the sass_decode
// helpers it calls (decode_mbarrier_init_expected_arrivals,
// arrival_and_tx_factors, is_complete_tx_form, opcode_width_bytes,
// tmem_columns_from_operand, captured_arrival_count).
#include "capture/operand_facts.hpp"

#include <cctype>
#include <cstring>
#include <regex>
#include <stdexcept>

namespace cyclescope_convert::capture {
namespace {

const std::set<std::string> kOpRoots = {
    "SYNCS", "ARRIVES", "UTMA", "UBLKCP", "UBLKRED", "UTC", "UCGABAR",
    "BAR", "STAS", "REDAS", "UGETNEXTWORKID", "STS", "UTCBAR",
    "DEPBAR", "WARPGROUP", "NANOSLEEP", "UTMACMDFLUSH", "UTMACCTL",
    "LDGDEPBAR", "MEMBAR", "FENCE", "CCTL", "ERRBAR", "CGAERRBAR",
    "MATCH",
};
const std::set<std::string> kTmaRoots = {"UTMALDG", "UTMASTG",
                                         "UTMAREDG", "UTMAPF"};
const std::set<std::string> kBulkRoots = {"UBLKCP", "UBLKRED"};
const std::set<std::string> kArriveRoots = {"ARRIVES", "UTCBAR"};

[[noreturn]] void missing(const std::string &where, const std::string &w) {
  throw std::runtime_error("missing_fact:" + w + " (" + where + ")");
}
[[noreturn]] void unsupp(const std::string &w) {
  throw std::runtime_error("unsupported:" + w);
}

bool is_ur(const std::string &op) {
  return op.rfind("UR", 0) == 0 && op.rfind("URANK", 0) != 0;
}
std::string upper(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::toupper(c));
  return s;
}
std::vector<std::string> tokens_of(const std::string &s) {
  std::vector<std::string> out;
  std::string t = s;
  while (!t.empty() && (t.back() == ' ' || t.back() == ';')) t.pop_back();
  std::size_t b = 0;
  while (true) {
    const auto c = t.find(',', b);
    std::string tok = t.substr(b, c == std::string::npos ? c : c - b);
    const auto a = tok.find_first_not_of(" \t");
    if (a != std::string::npos) {
      tok = tok.substr(a, tok.find_last_not_of(" \t") - a + 1);
      out.push_back(tok);
    }
    if (c == std::string::npos) break;
    b = c + 1;
  }
  return out;
}
std::vector<int> bare_ur_tokens(const std::vector<std::string> &toks) {
  static const std::regex re("^UR([0-9]+)$");
  std::vector<int> out;
  for (const auto &t : toks) {
    if (t.find('[') != std::string::npos) continue;
    std::smatch m;
    if (std::regex_match(t, m, re)) out.push_back(std::stoi(m[1]));
  }
  return out;
}
std::int64_t str_to_i64(const std::string &s, bool *ok) {
  try {
    std::size_t idx = 0;
    const auto v = std::stoll(s, &idx, 0);
    *ok = idx == s.size();
    return v;
  } catch (...) { *ok = false; return 0; }
}
const std::vector<std::int64_t> *
lane_get(const SemanticOperands &o, const std::string &k) {
  auto it = o.lane.find(k);
  return it == o.lane.end() ? nullptr : &it->second;
}
std::optional<std::int64_t> uni_get(const SemanticOperands &o,
                                    const std::string &k) {
  auto it = o.uniform.find(k);
  if (it == o.uniform.end()) return std::nullopt;
  return it->second;
}

// _bracket_word / _bracket_words_per_lane / _bracket_displacement
std::optional<std::int64_t> bracket_word(
    const std::string &operands_text, const SemanticOperands &o) {
  static const std::regex br("\\[([^\\]]+)\\]");
  std::smatch m;
  if (!std::regex_search(operands_text, m, br)) return std::nullopt;
  std::int64_t total = 0;
  const std::string inner = m[1];
  std::size_t b = 0;
  while (true) {
    const auto c = inner.find('+', b);
    std::string t = inner.substr(b, c == std::string::npos ? c : c - b);
    const auto a = t.find_first_not_of(" \t");
    t = a == std::string::npos ? "" :
        t.substr(a, t.find_last_not_of(" \t") - a + 1);
    t = t.substr(0, t.find('.'));
    b = c == std::string::npos ? c : c + 1;
    if (t == "RZ" || t == "URZ" || t.empty()) {
      if (c == std::string::npos) break;
      continue;
    }
    static const std::regex rr("^R([0-9]+)$"), ur("^UR([0-9]+)$");
    std::smatch mm;
    if (std::regex_match(t, mm, rr)) {
      const auto *v = lane_get(o, t);
      if (!v || v->empty()) return std::nullopt;
      total += (*v)[0];
    } else if (std::regex_match(t, mm, ur)) {
      const auto v = uni_get(o, t);
      if (!v) return std::nullopt;
      total += *v;
    } else {
      bool ok;
      total += str_to_i64(t, &ok);
      if (!ok) return std::nullopt;
    }
    if (c == std::string::npos) break;
  }
  return total;
}
std::optional<std::vector<std::int64_t>> bracket_words_per_lane(
    const std::string &operands_text, const SemanticOperands &o) {
  static const std::regex br("\\[([^\\]]+)\\]");
  std::smatch m;
  if (!std::regex_search(operands_text, m, br)) return std::nullopt;
  std::int64_t base = 0;
  std::vector<const std::vector<std::int64_t> *> regs;
  const std::string inner = m[1];
  std::size_t b = 0;
  while (true) {
    const auto c = inner.find('+', b);
    std::string t = inner.substr(b, c == std::string::npos ? c : c - b);
    const auto a = t.find_first_not_of(" \t");
    t = a == std::string::npos ? "" :
        t.substr(a, t.find_last_not_of(" \t") - a + 1);
    t = t.substr(0, t.find('.'));
    b = c == std::string::npos ? c : c + 1;
    if (t == "RZ" || t == "URZ" || t.empty()) {
      if (c == std::string::npos) break;
      continue;
    }
    static const std::regex rr("^R([0-9]+)$"), ur("^UR([0-9]+)$");
    std::smatch mm;
    if (std::regex_match(t, mm, rr)) {
      const auto *v = lane_get(o, t);
      if (!v || v->empty()) return std::nullopt;
      regs.push_back(v);
    } else if (std::regex_match(t, mm, ur)) {
      const auto v = uni_get(o, t);
      if (!v) return std::nullopt;
      base += *v;
    } else {
      bool ok;
      base += str_to_i64(t, &ok);
      if (!ok) return std::nullopt;
    }
    if (c == std::string::npos) break;
  }
  if (regs.empty()) return std::nullopt;
  std::size_t n = regs[0]->size();
  for (const auto *r : regs) n = std::min(n, r->size());
  std::vector<std::int64_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::int64_t v = base;
    for (const auto *r : regs) v += (*r)[i];
    out[i] = v;
  }
  return out;
}
int bracket_displacement(const std::string &operands_text, int ur) {
  static const std::regex re("\\[UR([0-9]+)\\+(0[xX][0-9a-fA-F]+|[0-9]+)\\]");
  for (std::sregex_iterator it(operands_text.begin(), operands_text.end(),
                             re), end; it != end; ++it)
    if (std::stoi((*it)[1]) == ur)
      return static_cast<int>(std::stoll((*it)[2], nullptr, 0));
  return 0;
}

// sass_decode helpers ----------------------------------------------------
std::optional<int64_t> decode_init_expected(std::uint64_t raw_state) {
  const std::uint64_t raw = raw_state;
  if (raw >= (1ull << 32)) {
    const std::uint64_t low = raw & 0xFFFFFFFFull, high = raw >> 32;
    if ((low & 1) || !(low > 0 && low < (1 << 21))) return std::nullopt;
    const std::int64_t count = (1 << 20) - (low >> 1);
    if (count > 0 && count < (1 << 20) &&
        (high & 0x7FFFFFFF) ==
            ((0x80000000ull - (count << 11)) & 0x7FFFFFFF))
      return count;
    return std::nullopt;
  }
  const std::int64_t bound = 1 << 16;
  std::optional<std::int64_t> hopper, sm103;
  if (!(raw & 1) && raw > 0 && raw < (1 << 21)) {
    const std::int64_t e = (1 << 20) - (raw >> 1);
    if (e > 0 && e <= bound) hopper = e;
  }
  if (raw > 0 && raw < (1 << 20)) {
    const std::int64_t e = (1 << 20) - raw;
    if (e > 0 && e <= bound) sm103 = e;
  }
  if (hopper && !sm103) return *hopper;
  if (sm103 && !hopper) return *sm103;
  if (!hopper) return std::nullopt;
  if (*hopper <= 1024) return *hopper;
  if (sm103 && *sm103 <= 1024) return *sm103;
  return *hopper;
}
bool is_complete_tx_form(const std::string &opcode) {
  static const std::regex re("(?:^|\\.)A[0-9]+TX(?:\\.|$)");
  return std::regex_search(upper(opcode), re);
}
std::pair<int, int> arrival_tx_factors(const std::string &opcode) {
  const std::string up = upper(opcode);
  const auto parts0 = up.substr(0, up.find('.'));
  if (parts0 == "UTCBAR") return {1, 0};
  std::smatch m;
  static const std::regex atx("(?:^|\\.)A([0-9]+)TX(?:\\.|$)"),
      at("(?:^|\\.)A([0-9]+)T([0-9]+)(?:\\.|$)"),
      atr("(?:^|\\.)A([0-9]+)TR[0-9]*(?:\\.|$)"),
      art("(?:^|\\.)ART([0-9]+)(?:\\.|$)");
  if (std::regex_search(up, m, atx)) return {std::stoi(m[1]), 1};
  if (std::regex_search(up, m, at)) return {std::stoi(m[1]), std::stoi(m[2])};
  if (std::regex_search(up, m, atr)) return {std::stoi(m[1]), 1};
  if (std::regex_search(up, m, art)) return {1, std::stoi(m[1])};
  auto has_part = [&](const char *p) {
    const std::string dot = std::string(".") + p;
    return up.find(dot) != std::string::npos ||
           up.rfind(p) == 0 && up.size() == strlen(p);
  };
  if (has_part("RED")) return has_part("TRANS64") ? std::pair{1, 1}
                                                  : std::pair{1, 0};
  if (has_part("ARVCNT")) return {1, 0};
  if (has_part("TRANSCNT")) return {0, 1};
  if (up == "SYNCS.ARRIVE.TRANS64") return {1, 1};
  unsupp("unrecognized arrive spelling " + opcode);
}
std::optional<int> opcode_width_bytes(const std::string &opcode) {
  const std::string up = upper(opcode);
  std::smatch m;
  static const std::regex vec("(?:^|\\.)(?:F|S|U)(8|16|32|64)X([0-9]+)(?:\\.|$)"),
      typ("(?:^|\\.)(?:B|F|S|U)(8|16|32|64)(?:\\.|$)");
  if (std::regex_search(up, m, vec))
    return (std::stoi(m[1]) / 8) * std::stoi(m[2]);
  if (std::regex_search(up, m, typ)) return std::stoi(m[1]) / 8;
  static const std::vector<std::pair<std::regex, int>> bit_widths = [] {
    std::vector<std::pair<std::regex, int>> v;
    for (const auto &[bits, w] : std::vector<std::pair<int, int>>{
             {256, 32}, {128, 16}, {64, 8}, {32, 4}, {16, 2}, {8, 1}})
      v.emplace_back(std::regex("(?:^|\\.)" + std::to_string(bits) +
                                "(?:\\.|$)"), w);
    return v;
  }();
  for (const auto &[r, w] : bit_widths)
    if (std::regex_search(up, r)) return w;
  const std::string root = up.substr(0, up.find('.'));
  if (root == "LDSM" || root == "STSM") {
    static const std::regex cnt("\\.X([0-9])(?:\\.|$)");
    return 4 * (std::regex_search(up, m, cnt) ? std::stoi(m[1]) : 1);
  }
  static const std::set<std::string> nodata = {"CCTL", "CCTLL",
                                             "PREFETCH", "UTMACCTL"};
  if (nodata.count(root)) return std::nullopt;
  static const std::set<std::string> four = {
      "LDC", "LDCU", "ULDC", "LDG", "LD", "LDL", "LDS", "STG", "ST",
      "STL", "STS", "STAS", "ATOM", "ATOMS", "ATOMG", "RED", "REDG",
      "REDS", "REDAS", "LDGSTS", "QSPC", "MEMBAR"};
  if (four.count(root)) return 4;
  static const std::set<std::string> mb = {"SYNCS", "MBAR", "ARRIVES",
                                         "UTCBAR", "UCGABAR"};
  if (mb.count(root)) return 8;
  missing("width", "memory width is undecodable: " + opcode);
}
std::optional<int> tmem_columns_from(std::int64_t v) {
  static const std::set<int> s = {32, 64, 128, 256, 512};
  if (s.count(int(v))) return int(v);
  if (s.count(int(v * 32))) return int(v * 32);
  return std::nullopt;
}
std::int64_t captured_arrival(const std::string &opcode, int lanes) {
  const auto [a, t] = arrival_tx_factors(opcode);
  if (a <= 0 || lanes <= 0) return 0;
  if (upper(opcode).substr(0, opcode.find('.')) == "UTCBAR") return a;
  return lanes * a;
}

std::pair<std::optional<std::int64_t>, std::optional<std::uint64_t>>
init_expected(const std::vector<std::string> &ops_layout,
              const SemanticOperands &o) {
  std::vector<std::pair<int, std::string>> urs;
  static const std::regex re("^UR([0-9]+)$");
  for (const auto &op : ops_layout) {
    std::smatch m;
    if (std::regex_match(op, m, re))
      urs.emplace_back(std::stoi(m[1]), op);
  }
  for (int i = int(urs.size()) - 2; i >= 0; --i) {
    auto [n0, lo] = urs[i];
    auto [n1, hi] = urs[i + 1];
    if (n1 == n0 + 1) {
      const auto vl = uni_get(o, lo), vh = uni_get(o, hi);
      if (vl && vh) {
        const std::uint64_t v =
            (std::uint64_t(*vh) << 32) |
            (std::uint64_t(*vl) & 0xFFFFFFFFull);
        if (const auto c = decode_init_expected(v))
          return {*c, v};
      }
    }
  }
  if (!ops_layout.empty()) {
    const auto &last = ops_layout.back();
    if (is_ur(last)) {
      if (const auto v = uni_get(o, last))
        if (const auto c = decode_init_expected(
                static_cast<std::uint64_t>(*v)))
          return {*c, static_cast<std::uint64_t>(*v)};
    }
    if (!last.empty() && last[0] == 'R' && last.rfind("RANK", 0) != 0) {
      const auto *v = lane_get(o, last);
      if (v && !v->empty())
        if (const auto c = decode_init_expected(
                static_cast<std::uint64_t>((*v)[0]) & 0xFFFFFFFFull))
          return {*c, static_cast<std::uint64_t>((*v)[0])};
    }
  }
  return {std::nullopt, std::nullopt};
}

void uput(SemanticOperands &o, const std::string &k, std::int64_t v) {
  if (!o.uniform.count(k)) o.uniform_order.push_back(k);
  o.uniform[k] = v;
}
void lput(SemanticOperands &o, const std::string &k,
          const std::vector<std::int64_t> &v) {
  if (!o.lane.count(k)) o.lane_order.push_back(k);
  o.lane[k] = v;
}

std::optional<int> opcode_width_bytes(const std::string &opcode);

}  // namespace

// public wrapper (memory_refs uses it for constant/LDG widths)
int opcode_width_bytes_public(const std::string &opcode) {
  const auto w = opcode_width_bytes(opcode);
  return w.value_or(0);
}

namespace {
const std::set<std::string> kNoMrefRootsD = {
    "LDTM", "STTM", "UTCCP", "UTCHMMA", "UTCQMMA", "UTCOMMA", "UTCBMMA",
    "UTCIMMA", "UTCMMA", "UTCFMMA", "UTCDMMA", "UTCBFMMA", "UTCATOMSWS",
    "UGETNEXTWORKID", "UTMACCTL", "UTMACMDFLUSH", "UVIRTCOUNT",
};
const std::set<std::string> kConstRootsD = {"LDC", "ULDC", "LDCU"};
const std::map<std::string, std::string> kAccessRootsD = {
    {"LDG", "load"}, {"STG", "store"}, {"LD", "load"}, {"ST", "store"},
    {"LDS", "load"}, {"STS", "store"}, {"LDL", "load"}, {"STL", "store"},
    {"LDC", "load"}, {"ULDC", "load"}, {"LDCU", "load"},
    {"LDSM", "load"}, {"STSM", "store"},
    {"LDGSTS", "async_copy"}, {"UTMALDG", "async_copy"},
    {"UTMASTG", "store"}, {"UTMAREDG", "async_reduce"},
    {"UTMAPF", "prefetch"}, {"UBLKCP", "async_copy"},
    {"UBLKRED", "async_reduce"}, {"ATOM", "atomic"}, {"ATOMG", "atomic"},
    {"ATOMS", "atomic"}, {"RED", "atomic"}, {"REDG", "atomic"},
    {"REDS", "atomic"}, {"STAS", "store"}, {"STTM", "store"},
    {"LDTM", "load"}, {"ATOMICS", "atomic"},
};

bool has_part_of(const std::string &up, const char *p) {
  std::size_t pos = 0;
  while ((pos = up.find('.', pos)) != std::string::npos) {
    if (up.compare(pos + 1, strlen(p), p) == 0 &&
        (pos + 1 + strlen(p) == up.size() || up[pos + 1 + strlen(p)] == '.'))
      return true;
    ++pos;
  }
  return false;
}

void fill_derived(StaticDerived &d, const StaticRow &row,
                  const CaptureRules &rules) {
  d = StaticDerived{};
  d.up = upper(row.opcode_text);
  d.root = d.up.substr(0, d.up.find('.'));
  d.tokens = tokens_of(row.operands_text);
  for (const auto &r : kOpRoots)
    if (d.root.rfind(r, 0) == 0) { d.op_root = true; break; }
  if (d.op_root) {
    int index = 0;
    for (const auto &tok : row.inst.operands) {
      const std::string t = upper(tok);
      if (t == "RZ" || t == "URZ") {
        d.op_consts.emplace_back("op" + std::to_string(index), 0);
      } else {
        bool ok;
        const auto v = str_to_i64(t, &ok);
        if (ok) d.op_consts.emplace_back("op" + std::to_string(index), v);
      }
      ++index;
    }
  }
  d.is_tma = kTmaRoots.count(d.root) != 0;
  d.is_bulk = kBulkRoots.count(d.root) != 0;
  d.no_mref_root = kNoMrefRootsD.count(d.root) != 0;
  d.const_root = kConstRootsD.count(d.root) != 0;
  {
    const auto it = kAccessRootsD.find(d.root);
    d.access_kind = it == kAccessRootsD.end() ? "other" : it->second;
  }
  try {
    d.width = opcode_width_bytes(row.opcode_text);
  } catch (const std::runtime_error &e) {
    d.width_error = e.what();
  }
  // trivial: no dynamic operand entry can reach operand_facts (layout
  // holds only PRED / BRVAL* / SPACE), the root is not an _OP_ROOTS,
  // arrive, phase-token or mbarrier-rule root -> the only possible
  // fact is after_ready_mask (see operand_facts)
  bool dyn_ops = false;
  for (const auto &op : row.ops_layout)
    if (op != "PRED" && op.rfind("BRVAL", 0) != 0 && op != "SPACE")
      dyn_ops = true;
  bool rule_root = rules.mbarrier_companion.count(d.root) ||
      rules.mbarrier_self.count(d.root) ||
      rules.mbarrier_bracket.count(d.root);
  if (!rule_root && rules.mbarrier_ur_word) {
    const std::map<std::string, std::vector<std::int64_t>> no_ops;
    if (rules.mbarrier_ur_word(no_ops, row.operands_text, d.root))
      rule_root = true;
  }
  d.trivial = !d.op_root && !dyn_ops && !rule_root &&
      !kArriveRoots.count(d.root) && !d.is_tma && !d.is_bulk &&
      !has_part_of(d.up, "PHASECHK") && !has_part_of(d.up, "TRYWAIT") &&
      !has_part_of(d.up, "TEST_WAIT") && !has_part_of(d.up, "ARRIVE") &&
      d.up.rfind("MBAR.INIT", 0) != 0 && d.root != "STS" &&
      d.root != "NANOSLEEP" && d.root != "UTMACCTL" &&
      d.root != "UTCATOMSWS" && d.root != "STAS";
  d.ready = true;
}
}  // namespace

void prepare_static_row(StaticRow &row, const CaptureRules &rules) {
  fill_derived(row.derived, row, rules);
}

const StaticDerived &derived_of(const StaticRow &row,
                                const CaptureRules &rules) {
  if (row.derived.ready) return row.derived;
  thread_local StaticDerived tmp;
  fill_derived(tmp, row, rules);
  return tmp;
}

std::map<int, std::vector<std::int64_t>>
remote_ranks(const std::map<std::string, std::vector<std::int64_t>> &ops,
             int cluster_rank) {
  std::map<int, std::vector<std::int64_t>> out;
  static const std::regex tail("([0-9]+)$");
  for (const auto &[name, value] : ops) {
    if (name.rfind("RANK", 0) != 0 && name.rfind("BRANK", 0) != 0)
      continue;
    std::smatch m;
    if (!std::regex_search(name, m, tail)) continue;
    bool remote = false;
    for (const auto v : value)
      if (v >= 0 && v != cluster_rank) remote = true;
    if (remote) out[std::stoi(m[0])] = value;
  }
  return out;
}

std::optional<SemanticOperands>
operand_facts(const DynRecord &rec, const StaticRow &row,
              const CaptureRules &rules, int cluster_rank,
              const std::map<int, std::vector<std::int64_t>> *ranks_in) {
  const auto &ops = rec.ops;
  const StaticDerived &d = derived_of(row, rules);
  // rows without any fact-bearing operand (the bulk of ALU records):
  // only the after_ready mask can be reported
  if (d.trivial) {
    if (!rec.has_after_ready) return std::nullopt;
    SemanticOperands o;
    o.after_ready_mask = rec.after_ready;
    return o;
  }
  const std::string &opcode = row.opcode_text;
  const std::string &up = d.up;
  const std::string &root = d.root;
  const auto &tokens = d.tokens;
  SemanticOperands o;

  auto has_part = [&](const char *p) { return has_part_of(up, p); };

  // op<N> static operand tokens for _OP_ROOTS (startswith semantics)
  for (const auto &[k, v] : d.op_consts) uput(o, k, v);

  for (const auto &name : rec.ops_order) {
    const auto &value = ops.at(name);
    if (is_ur(name)) {
      uput(o, name, value.empty() ? 0 : value[0]);
    } else if (name == "IDESC") {
      uput(o, "idesc_value", value[0]);
      o.descriptor_words.push_back(value[0]);
    } else if (name == "GDESC") {
      uput(o, "gdesc_value_lo", value[0]);
      o.descriptor_words.push_back(value[0]);
    } else if (name == "GDESCH") {
      uput(o, "gdesc_value_hi", value[0]);
      o.descriptor_words.push_back(value[0]);
    } else if (name == "TMEMD") {
      uput(o, "tmem_d_value", value[0]);
      o.descriptor_words.push_back(value[0]);
    } else if (name == "CPOL") {
      uput(o, "cache_policy_lo", value[0]);
    } else if (name == "CPOLH") {
      uput(o, "cache_policy_hi", value[0]);
    } else if (name == "DESC") {
      o.tensor_map = static_cast<std::uint32_t>(value[0]);
    } else if (name[0] == 'R' && name.rfind("RANK", 0) != 0) {
      lput(o, name, value);
    }
  }

  // TMA box coordinates
  if (kTmaRoots.count(root) || kBulkRoots.count(root)) {
    std::smatch rm;
    static const std::regex dre("\\.([0-9])D(?:\\.|$)");
    int rank = 0;
    if (std::regex_search(up, rm, dre)) rank = std::stoi(rm[1]);
    if (has_part("IM2COL") && rank) rank += 2;
    std::vector<int> brackets;
    static const std::regex bu("\\[UR([0-9]+)[^\\]]*\\]");
    for (std::sregex_iterator it(row.operands_text.begin(),
                               row.operands_text.end(), bu), e;
         it != e; ++it)
      brackets.push_back(std::stoi((*it)[1]));
    const int offset = !brackets.empty() && rank
        ? rules.tma_coordinate_offset(root, int(brackets.size())) : -1;
    if (offset >= 0) {
      const int base = brackets[0] + offset;
      bool all = true;
      std::vector<std::int64_t> vals;
      for (int k = 0; k < rank; ++k) {
        const auto v = uni_get(o, "UR" + std::to_string(base + k));
        if (!v) { all = false; break; }
        vals.push_back(*v);
      }
      if (all) o.tensor_coordinates = vals;
    }
    if (o.tensor_coordinates.empty()) {
      std::vector<std::pair<int, std::int64_t>> legacy;
      for (const auto &[n, v] : ops)
        if (n.rfind("TCOORD", 0) == 0 && !v.empty())
          legacy.emplace_back(
              n.size() > 6 ? std::stoi(n.substr(6)) : 0, v[0]);
      std::sort(legacy.begin(), legacy.end());
      if (int(legacy.size()) >= rank && rank > 0) {
        for (int i = 0; i < rank; ++i)
          o.tensor_coordinates.push_back(legacy[i].second);
      }
    }
  }

  std::map<int, std::vector<std::int64_t>> ranks_storage;
  const std::map<int, std::vector<std::int64_t>> &ranks =
      ranks_in ? *ranks_in
               : (ranks_storage = remote_ranks(ops, cluster_rank),
                  ranks_storage);

  const bool executing = rec.exec_mask != 0;

  // mbarrier word
  std::optional<int> mbar;
  if (rules.mbarrier_ur_word)
    mbar = rules.mbarrier_ur_word(ops, row.operands_text, root);
  if (!executing) {
    mbar = std::nullopt;
  } else if (root == "UBLKCP" && up.find(".G.S") != std::string::npos) {
    mbar = std::nullopt;
  } else if (root == "STAS") {
    static const std::regex pr("\\[R([0-9]+)\\.64");
    std::smatch pm;
    if (std::regex_search(row.operands_text, pm, pr)) {
      const auto *hi = lane_get(o, "R" + std::to_string(std::stoi(pm[1]) + 1));
      if (hi && !hi->empty()) {
        const std::int64_t word = (*hi)[0];
        uput(o, "mbarrier_address", word);
        if ((std::uint64_t(word) >> 24) != 0)
          uput(o, "owner_rank", std::int64_t(std::uint64_t(word) >> 24));
        std::set<std::int64_t> uniq(hi->begin(), hi->end());
        if (uniq.size() > 1) lput(o, "mbarrier_words", *hi);
      }
    }
    if (!d.width_error.empty()) throw std::runtime_error(d.width_error);
    if (d.width) uput(o, "bytes", std::int64_t(*d.width) * rec.exec_count());
  } else if (mbar && uni_get(o, "UR" + std::to_string(*mbar))) {
    const bool self = rules.mbarrier_self.count(root) != 0;
    const int disp = self ? bracket_displacement(row.operands_text, *mbar) : 0;
    uput(o, "mbarrier_address",
         *uni_get(o, "UR" + std::to_string(*mbar)) + disp);
  } else if (executing && rules.mbarrier_bracket.count(root)) {
    auto word = bracket_word(row.operands_text, o);
    if (!word && !rec.mrefs.empty() && !rec.mrefs[0].empty())
      word = static_cast<std::int64_t>(rec.mrefs[0][0]);
    if (word) uput(o, "mbarrier_address", *word);
    auto words = bracket_words_per_lane(row.operands_text, o);
    if (!words && !rec.mrefs.empty() && rec.mrefs[0].size() > 1) {
      words = std::vector<std::int64_t>(rec.mrefs[0].begin(),
                                        rec.mrefs[0].end());
    }
    if (words) {
      std::set<std::int64_t> uniq(words->begin(), words->end());
      if (uniq.size() > 1) lput(o, "mbarrier_words", *words);
    }
  }
  if (o.uniform.count("mbarrier_address")) {
    const auto rank = rules.owner_rank_entry
        ? rules.owner_rank_entry(ops, row.operands_text) : std::nullopt;
    if (rank) {
      uput(o, "owner_rank", *rank);
    } else if (std::uint64_t(o.uniform["mbarrier_address"]) >> 24) {
      uput(o, "owner_rank",
           std::int64_t(
               std::uint64_t(o.uniform["mbarrier_address"]) >> 24));
    } else if (kTmaRoots.count(root) || kBulkRoots.count(root) ||
               root == "UTCBAR" || root == "UGETNEXTWORKID" ||
               (root == "SYNCS" && has_part("RED"))) {
      uput(o, "owner_rank", 0);
    }
  }
  if (kTmaRoots.count(root) && !rec.mrefs.empty() && !rec.mrefs[0].empty())
    uput(o, "shared_destination",
         static_cast<std::int64_t>(rec.mrefs[0][0]));

  // mbarrier init state / STS.64
  if (executing) {
    if ((root == "SYNCS" && has_part("EXCH")) ||
        up.rfind("MBAR.INIT", 0) == 0) {
      const auto [count, value] = init_expected(row.ops_layout, o);
      if (count) {
        uput(o, "expected_arrivals", *count);
        uput(o, "initial_phase", std::int64_t((*value >> 63) & 1));
      }
    } else if (up == "STS.64" || up == "STS") {
      std::vector<std::int64_t> values;
      if (up == "STS.64") {
        if (!tokens.empty()) {
          static const std::regex rr("^R([0-9]+)$");
          std::smatch m;
          if (std::regex_match(tokens.back(), m, rr)) {
            const int b = std::stoi(m[1]);
            const auto *lo = lane_get(o, "R" + std::to_string(b));
            const auto *hi = lane_get(o, "R" + std::to_string(b + 1));
            if (lo && hi) {
              const auto n = std::min(lo->size(), hi->size());
              for (std::size_t i = 0; i < n; ++i)
                values.push_back(
                    (std::int64_t((*hi)[i] & 0xFFFFFFFFull) << 32) |
                    ((*lo)[i] & 0xFFFFFFFFull));
            }
          }
        }
      } else {
        if (!tokens.empty()) {
          static const std::regex rr("^R([0-9]+)$");
          std::smatch m;
          if (std::regex_match(tokens.back(), m, rr)) {
            const auto *v = lane_get(o, tokens.back());
            if (v) values = *v;
          }
        }
      }
      if (!values.empty()) {
        const auto count = decode_init_expected(
            static_cast<std::uint64_t>(values[0]));
        uput(o, "store_value_captured", 1);
        if (count) {
          uput(o, "expected_arrivals", *count);
          uput(o, "initial_phase",
               std::int64_t((std::uint64_t(values[0]) >> 63) & 1));
          lput(o, "sts64_value", values);
        }
      }
    }
  }

  // phase-token parity
  if (has_part("PHASECHK") || has_part("TRYWAIT") ||
      has_part("TEST_WAIT")) {
    std::optional<std::int64_t> parity;
    for (auto it = row.ops_layout.rbegin(); it != row.ops_layout.rend();
         ++it) {
      const auto &op = *it;
      if (!op.empty() && op[0] == 'R' && op.rfind("RANK", 0) != 0) {
        const auto *v = lane_get(o, op);
        if (v && !v->empty()) {
          parity = (std::uint64_t((*v)[0]) >> 31) & 1;
          break;
        }
      }
    }
    if (!parity && !tokens.empty() &&
        (tokens.back() == "RZ" || tokens.back() == "URZ"))
      parity = 0;
    if (parity) uput(o, "parity", *parity);
  }

  // arrive / expect-tx facts
  if (executing && ((root == "SYNCS" && has_part("ARRIVE")) ||
                    kArriveRoots.count(root))) {
    const auto [a_factor, t_factor] = arrival_tx_factors(opcode);
    const int lanes = rec.exec_count();
    std::smatch art;
    static const std::regex artre("(?:^|\\.)ART([0-9]+)(?:\\.|$)");
    if (std::regex_search(up, art, artre)) {
      const std::string tok = tokens.empty() ? "" : tokens.back();
      if (tok == "RZ" || tok == "URZ") {
        uput(o, "arrival_count", 0);
      } else {
        static const std::regex rr("^R([0-9]+)$"), ur("^UR[0-9]+$");
        if (std::regex_match(tok, rr)) {
          const auto *v = lane_get(o, tok);
          if (!v) missing("pc", tok + " per-lane arrival count");
          std::int64_t s = 0;
          for (const auto x : *v) s += x;
          uput(o, "arrival_count", s);
        } else if (std::regex_match(tok, ur)) {
          const auto uv = uni_get(o, tok);
          if (!uv) missing("pc", tok + " uniform arrival count");
          uput(o, "arrival_count", *uv * lanes);
        } else {
          missing("pc", "arrival count operand");
        }
      }
      uput(o, "transaction_bytes", std::int64_t(std::stoi(art[1])) * lanes);
    } else {
      uput(o, "arrival_count", captured_arrival(opcode, lanes));
    }
    if (t_factor && !std::regex_search(up, artre)) {
      std::smatch imm;
      static const std::regex aim("(?:^|\\.)A([0-9]+)T([0-9]+)(?:\\.|$)");
      if (std::regex_search(up, imm, aim) && std::stoi(imm[2]) > 0) {
        uput(o, "transaction_bytes", std::int64_t(std::stoi(imm[2])) * lanes);
      } else if (!tokens.empty()) {
        const std::string tok = tokens.back();
        std::optional<std::int64_t> tx;
        if (tok == "RZ" || tok == "URZ") {
          tx = 0;
        } else {
          static const std::regex rr("^R([0-9]+)$"), ur("^UR[0-9]+$");
          if (std::regex_match(tok, rr)) {
            const auto *v = lane_get(o, tok);
            if (v && !v->empty()) tx = (*v)[0];
          } else if (std::regex_match(tok, ur)) {
            tx = uni_get(o, tok);
          } else {
            bool ok;
            tx = str_to_i64(tok, &ok);
            if (!ok) tx = std::nullopt;
          }
        }
        if (tx) uput(o, "transaction_bytes", *tx);
      }
    }
    if (is_complete_tx_form(opcode)) uput(o, "transaction_complete", 1);
  }

  // UTMACCTL.IV [URn]
  if (executing && root == "UTMACCTL" && has_part("IV")) {
    static const std::regex brk("\\[UR([0-9]+)(?:\\+(0x[0-9a-fA-F]+|[0-9]+))?\\]");
    std::smatch bm;
    if (std::regex_search(row.operands_text, bm, brk)) {
      const auto lo = uni_get(o, "UR" + bm[1].str());
      const auto hi = uni_get(o, "UR" + std::to_string(std::stoi(bm[1]) + 1));
      if (lo && hi) {
        std::uint64_t base = (std::uint64_t(*hi) << 32) |
            (std::uint64_t(*lo) & 0xFFFFFFFFull);
        if (bm[2].matched)
          base += std::stoull(bm[2].str(), nullptr, 0);
        uput(o, "descriptor_address", std::int64_t(base));
      }
    }
  }

  // multicast mask
  {
    std::optional<std::int64_t> mask;
    if (kTmaRoots.count(root) && up.find("MULTICAST") != std::string::npos &&
        up.find("MULTICAST") > up.find('.')) {
      std::set<int> bracket;
      // brackets over the joined tokens (capture_rules parity)
      static const std::regex bu(R"(\[UR([0-9]+)[^\]]*\])");
      const std::string joined = [&] {
        std::string s;
        for (const auto &t : tokens) s += t + " ";
        return s;
      }();
      for (std::sregex_iterator it(joined.begin(), joined.end(), bu), e;
           it != e; ++it) {
        bracket.insert(std::stoi((*it)[1]));
        bracket.insert(std::stoi((*it)[1]) + 1);
      }
      const auto bare = bare_ur_tokens(tokens);
      for (auto it = bare.rbegin(); it != bare.rend(); ++it) {
        if (bracket.count(*it)) continue;
        const auto v = uni_get(o, "UR" + std::to_string(*it));
        if (v) { mask = (*v >> 16) & 0xFFFF; break; }
      }
    } else if (root == "UBLKCP" && up.find("MULTICAST") != std::string::npos) {
      const auto bare = bare_ur_tokens(tokens);
      if (bare.size() == 1) {
        const auto v = uni_get(o, "UR" + std::to_string(bare[0]));
        if (v) mask = (*v >> 16) & 0xFFFF;
      }
    } else if (root == "UTCBAR" &&
               up.find("MULTICAST") != std::string::npos &&
               up.find("MULTICAST") > up.find('.')) {
      const auto bare = bare_ur_tokens(tokens);
      if (!bare.empty()) {
        const auto v = uni_get(o, "UR" + std::to_string(bare.back()));
        if (v) mask = *v;
      }
    }
    if (mask) uput(o, "multicast_mask", *mask);
  }

  if (kBulkRoots.count(root)) {
    const auto bare = bare_ur_tokens(tokens);
    if (bare.size() == 1) {
      const auto v = uni_get(o, "UR" + std::to_string(bare[0]));
      if (v) {
        const std::int64_t units = *v & 0xFFFF;
        if (units > 0) uput(o, "bytes", units * 16);
      }
    }
  }

  // tmem columns / dealloc mask
  {
    std::optional<std::int64_t> columns;
    const bool fas = root == "UTCATOMSWS" && has_part("FIND_AND_SET");
    if (fas) {
      const auto bare = bare_ur_tokens(tokens);
      if (!bare.empty()) {
        const auto v = uni_get(o, "UR" + std::to_string(bare.back()));
        if (v)
          if (const auto c = tmem_columns_from(*v))
            columns = *c;
      }
    }
    if (columns) {
      uput(o, "tmem_columns", *columns);
    } else if (root == "UTCATOMSWS" && has_part("AND")) {
      const auto bare = bare_ur_tokens(tokens);
      if (!bare.empty()) {
        const auto v = uni_get(o, "UR" + std::to_string(bare.back()));
        if (v) uput(o, "tmem_dealloc_mask", *v);
      }
    }
  }

  if (root == "NANOSLEEP") {
    for (const auto &op : row.ops_layout) {
      if (!op.empty() && op[0] == 'R' && op.rfind("RANK", 0) != 0) {
        const auto *v = lane_get(o, op);
        if (v && !v->empty()) {
          uput(o, "requested_ns", (*v)[0]);
          break;
        }
      }
    }
  }

  o.ranks = ranks;
  for (const auto &[k, _v] : ranks) o.ranks_order.push_back(k);
  if (rec.has_after_ready) o.after_ready_mask = rec.after_ready;

  if (o.uniform.empty() && o.lane.empty() && ranks.empty() &&
      o.tensor_coordinates.empty() && o.descriptor_words.empty() &&
      !o.tensor_map && !o.after_ready_mask)
    return std::nullopt;
  return o;
}

void uput_wrap(SemanticOperands &o, const std::string &k,
               std::int64_t v) {
  if (!o.uniform.count(k) && !o.str_uniform.count(k))
    o.uniform_order.push_back(k);
  o.str_uniform.erase(k);
  o.uniform[k] = v;
}
void uput_str(SemanticOperands &o, const std::string &k,
              const std::string &v) {
  if (!o.uniform.count(k) && !o.str_uniform.count(k))
    o.uniform_order.push_back(k);
  o.uniform.erase(k);
  o.str_uniform[k] = v;
}

// the barrier word a poll addresses: the bracket operand's register sum
// of the first executing lane plus the displacement (as the converter
// resolves the mbarrier object), the tracer's mref address where a
// register value is not captured
std::optional<std::int64_t> poll_barrier_word(const DynRecord &rec,
                                              const StaticRow &row) {
  if (!rec.exec_mask) return std::nullopt;
  SemanticOperands o;
  for (const auto &name : rec.ops_order) {
    const auto &value = rec.ops.at(name);
    if (is_ur(name))
      uput(o, name, value.empty() ? 0 : value[0]);
    else if (name[0] == 'R' && name.rfind("RANK", 0) != 0)
      lput(o, name, value);
  }
  auto word = bracket_word(row.operands_text, o);
  if (!word && !rec.mrefs.empty() && !rec.mrefs[0].empty())
    word = static_cast<std::int64_t>(rec.mrefs[0][0]);
  return word;
}

}  // namespace cyclescope_convert::capture
