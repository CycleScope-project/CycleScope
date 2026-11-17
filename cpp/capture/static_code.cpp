#include "capture/static_code.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace fs = std::filesystem;

namespace cyclescope_convert::capture {
namespace {

[[noreturn]] void missing(const std::string &where, const std::string &w) {
  throw std::runtime_error("missing_fact:" + w + " (" + where + ")");
}

const std::regex &reg_token() {
  static const std::regex r(
      "\\b(UR[0-9]+|R[0-9]+|UP[0-9]+|P[0-9]+|SR_[A-Z_]+|B[0-9]+|RZ|URZ|PT|UPT)\\b");
  return r;
}
const std::regex &addr_bracket() {
  static const std::regex r("\\[([^\\]]+)\\]");
  return r;
}

std::vector<std::string> regs_in(const std::string &token) {
  std::vector<std::string> out;
  for (std::sregex_iterator it(token.begin(), token.end(), reg_token()),
       end; it != end; ++it)
    out.push_back((*it)[1].str());
  return out;
}

std::string strip(const std::string &s) {
  std::size_t a = s.find_first_not_of(" \t");
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(" \t") - a + 1);
}
std::string upper(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::toupper(c));
  return s;
}
std::vector<std::string> split_commas(const std::string &s) {
  std::vector<std::string> out;
  std::size_t b = 0;
  while (true) {
    std::size_t c = s.find(',', b);
    std::string t = strip(s.substr(b, c == std::string::npos ? c : c - b));
    if (!t.empty()) out.push_back(t);
    if (c == std::string::npos) break;
    b = c + 1;
  }
  return out;
}

// _parse_operands: guard, opcode, operands_text, dests, srcs
struct Parsed {
  std::string guard;    // without '@' (may include '!'); empty = none
  std::string opcode, operands;
  std::vector<std::string> dests, srcs;
};
Parsed parse_operands(const std::string &sass) {
  Parsed p;
  std::string text = strip(sass);
  if (!text.empty() && text[0] == '@') {
    const std::size_t sp = text.find(' ');
    p.guard = text.substr(1, sp == std::string::npos ? sp : sp - 1);
    text = sp == std::string::npos ? "" : strip(text.substr(sp + 1));
  }
  const std::size_t sp = text.find(' ');
  std::string head = sp == std::string::npos ? text : text.substr(0, sp);
  std::string ops = sp == std::string::npos ? "" : text.substr(sp + 1);
  if (!head.empty() && head.back() == ';') head.pop_back();
  const std::size_t semi = ops.find(';');
  ops = strip(ops.substr(0, semi));
  p.opcode = head;
  p.operands = ops;
  const auto tokens = split_commas(ops);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto regs = regs_in(tokens[i]);
    if (i == 0 && tokens[i].find('[') == std::string::npos)
      p.dests.insert(p.dests.end(), regs.begin(), regs.end());
    else
      p.srcs.insert(p.srcs.end(), regs.begin(), regs.end());
  }
  return p;
}

// _register_roles: registers inside [] are addresses
std::vector<std::pair<std::string, bool>>
register_roles(const std::string &operands,
               const std::vector<std::string> &srcs,
               const std::vector<std::string> &dests) {
  std::vector<std::pair<std::string, bool>> out;
  std::set<std::string> addr;
  for (std::sregex_iterator it(operands.begin(), operands.end(),
                             addr_bracket()), end; it != end; ++it)
    for (const auto &r : regs_in((*it)[1].str())) addr.insert(r);
  // dict{reg: role for reg in srcs+dests}: first occurrence wins
  std::set<std::string> seen;
  auto emit = [&](const std::vector<std::string> &v) {
    for (const auto &r : v)
      if (seen.insert(r).second)
        out.emplace_back(r, addr.count(r) != 0);
  };
  emit(srcs);
  emit(dests);
  return out;
}

// _SPACE_BY_ROOT / _GENERIC_ROOTS
std::string space_for(const std::string &root) {
  static const std::map<std::string, std::string> m = {
      {"LDS","shared"},{"STS","shared"},{"LDSM","shared"},{"STSM","shared"},
      {"LDTM","shared"},{"STTM","shared"},{"ATOMS","shared"},{"REDS","shared"},
      {"STAS","shared"},{"SYNCS","shared"},{"ARRIVES","shared"},
      {"UTCBAR","shared"},{"UCGABAR","shared"},{"MBAR","shared"},
      {"LDL","local"},{"STL","local"},
      {"LDC","constant"},{"ULDC","constant"},{"LDCU","constant"},
      {"LDG","global"},{"STG","global"},{"ATOMG","global"},{"REDG","global"},
      {"LDGSTS","global"},{"UTMALDG","shared"},{"UTMASTG","global"},
      {"UTMAREDG","global"},{"UTMAPF","global"},{"UBLKCP","global"},
      {"UBLKRED","global"},
  };
  auto it = m.find(root);
  if (it != m.end()) return it->second;
  static const std::set<std::string> gen = {"LD","ST","ATOM","RED"};
  return gen.count(root) ? "generic" : "";
}

}  // namespace

std::string opcode_root(const std::string &opcode) {
  std::string text = strip(opcode);
  if (!text.empty() && text[0] == '@') {
    const std::size_t sp = text.find(' ');
    text = sp == std::string::npos ? "" : strip(text.substr(sp + 1));
  }
  const std::size_t sp = text.find_first_of(" \t");
  if (sp != std::string::npos) text = text.substr(0, sp);
  return upper(text.substr(0, text.find('.')));
}

ControlWord control_word_from_bytes(const std::string &bytes_hex) {
  ControlWord c;
  if (bytes_hex.size() < 32) return c;
  std::uint64_t word = 0;
  for (int i = 0; i < 8; ++i) {
    const auto hi = bytes_hex.substr(16 + 2 * i, 2);
    word |= std::uint64_t(std::stoul(hi, nullptr, 16)) << (8 * i);
  }
  c.stall = int(word >> 41) & 0xF;
  c.yield_hint = 1 - (int(word >> 45) & 1);
  c.write_barrier = int(word >> 46) & 7;
  c.read_barrier = int(word >> 49) & 7;
  c.wait_mask = int(word >> 52) & 0x3F;
  c.reuse_mask = int(word >> 58) & 0xF;
  return c;
}

bool numeric_control_target(const std::string &operands,
                            std::int32_t *target) {
  std::vector<long long> numeric;
  static const std::regex num("0[xX][0-9a-fA-F]+|[0-9]+");
  // sass_tokens: split on ';' then ',' fallback
  std::vector<std::string> parts;
  {
    std::size_t b = 0;
    while (true) {
      std::size_t c = operands.find(';', b);
      parts.push_back(strip(
          operands.substr(b, c == std::string::npos ? c : c - b)));
      if (c == std::string::npos) break;
      b = c + 1;
    }
    parts.erase(std::remove_if(parts.begin(), parts.end(),
                               [](const std::string &s) {
                                 return s.empty();
                               }),
                parts.end());
    if (parts.size() == 1 && parts[0].find(',') != std::string::npos)
      parts = split_commas(parts[0]);
  }
  for (auto &t : parts) {
    if (std::regex_match(t, num))
      numeric.push_back(std::stoll(t, nullptr, 0));
  }
  if (numeric.size() != 1) return false;
  *target = static_cast<std::int32_t>(numeric[0]);
  return true;
}

std::map<int, FunctionStatic> load_case(const std::string &case_dir,
                                        const ConvertRules &rules) {
  const fs::path dir(case_dir);
  const fs::path fj = dir / "functions.jsonl";
  if (!fs::is_regular_file(fj))
    missing(case_dir, "functions.jsonl");
  std::map<int, std::string> names, mangleds;
  std::map<int, std::uint64_t> addrs;
  {
    std::ifstream fh(fj);
    std::string line;
    while (std::getline(fh, line)) {
      if (line.empty()) continue;
      // minimal JSON field extraction (flat object, string/number values)
      auto field = [&](const char *k) -> std::string {
        const std::regex r(std::string("\"") + k + "\"\\s*:\\s*(\"[^\"]*\"|[0-9a-fA-FxX]+)");
        std::smatch m;
        if (!std::regex_search(line, m, r)) return "";
        std::string v = m[1].str();
        if (v.front() == '"') v = v.substr(1, v.size() - 2);
        return v;
      };
      std::string id = field("function_id");
      if (id.empty()) id = field("reg");
      if (id.empty()) continue;
      const int fid = std::stoi(id);
      const std::string nm = field("name");
      const std::string mg = field("mangled");
      names[fid] = nm;
      mangleds[fid] = mg.empty() ? nm : mg;
      const std::string ad = field("addr");
      if (!ad.empty()) addrs[fid] = std::stoull(ad, nullptr, 16);
    }
  }
  std::map<int, FunctionStatic> tables;
  const fs::path fdir = dir / "functions";
  std::vector<fs::path> tsvs;
  if (fs::is_directory(fdir))
    for (const auto &e : fs::directory_iterator(fdir))
      if (e.path().extension() == ".tsv") tsvs.push_back(e.path());
  std::sort(tsvs.begin(), tsvs.end());
  for (const auto &tsv : tsvs) {
    const int fid = std::stoi(tsv.stem().string());
    auto &table = tables[fid];
    table.function_id = fid;
    const std::string dflt = "function" + std::to_string(fid);
    table.name = names.count(fid) ? names[fid] : dflt;
    table.mangled = mangleds.count(fid) ? mangleds[fid] : table.name;
    table.addr = addrs.count(fid) ? addrs[fid] : 0;
    std::ifstream fh(tsv);
    std::string line;
    while (std::getline(fh, line)) {
      if (line.empty()) continue;
      std::vector<std::string> parts;
      std::size_t b = 0;
      while (true) {
        std::size_t t = line.find('\t', b);
        parts.push_back(line.substr(b, t == std::string::npos ? t : t - b));
        if (t == std::string::npos) break;
        b = t + 1;
      }
      if (parts.size() < 6)
        missing(tsv.string(), "malformed row: " + line.substr(0, 80));
      StaticRow row;
      row.idx = std::stoi(parts[0]);
      const std::uint64_t pc = std::stoull(parts[1], nullptr, 16);
      const std::string &bytes_hex = parts[2];
      row.sass = strip(parts[3]);
      row.ops_layout = parts[4].empty() ? std::vector<std::string>{}
                                      : split_commas(parts[4]);
      row.mref_count = static_cast<std::uint32_t>(std::stoul(parts[5]));
      const Parsed p = parse_operands(row.sass);
      row.opcode_text = p.opcode;
      row.operands_text = p.operands;
      const std::string root = opcode_root(p.opcode);
      if (root == "BRA" || root == "BSSY") {
        std::int32_t t;
        if (numeric_control_target(p.operands, &t)) row.branch_target = t;
      }
      StatRow &s = row.inst;
      s.pc = pc;
      s.code_address = table.addr ? table.addr + pc : 0;
      s.opcode = p.opcode;
      s.predicate = p.guard.empty() ? ""
          : (p.guard[0] == '!' ? p.guard.substr(1) : p.guard);
      s.has_predicate = !p.guard.empty();
      s.predicate_inverted = !p.guard.empty() && p.guard[0] == '!';
      s.operands = split_commas(p.operands);
      s.source_registers = p.srcs;
      s.destination_registers = p.dests;
      s.register_roles = register_roles(p.operands, p.srcs, p.dests);
      const ControlWord cw = control_word_from_bytes(bytes_hex);
      s.wait_mask = static_cast<std::uint32_t>(cw.wait_mask);
      s.reuse_mask = static_cast<std::uint32_t>(cw.reuse_mask);
      s.packed_ctrl = (cw.stall & 0xF) | ((cw.yield_hint & 1) << 4) |
          ((cw.read_barrier & 0xF) << 5) | ((cw.write_barrier & 0xF) << 9);
      const std::string sp = space_for(root);
      s.memory_space = sp;
      s.has_memory_space = !sp.empty();
      s.branch_target = row.branch_target;
      // operation_class via arch rules (StaticInfo view)
      StaticInfo info;
      info.opcode = s.opcode;
      info.operands = s.operands;
      info.memory_space = s.memory_space;
      info.has_predicate = s.has_predicate;
      info.pc = s.pc;
      info.dst_names = s.destination_registers;
      info.src_names = s.source_registers;
      s.operation_class = rules.operation_class(info);
      table.rows[row.idx] = std::move(row);
    }
  }
  return tables;
}

}  // namespace cyclescope_convert::capture
