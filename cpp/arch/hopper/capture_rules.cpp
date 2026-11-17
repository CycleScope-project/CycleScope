// Hopper operand-fact extraction rules
// (python/cyclescope/arch/hopper/capture_rules.py).
#include "capture/operand_facts.hpp"
#include <regex>

namespace cyclescope_convert::capture {
namespace {

std::optional<int> owner_rank_entry(
    const std::map<std::string, std::vector<std::int64_t>> &ops,
    const std::string &operands_text) {
  static const std::regex br("\\[UR([0-9]+)");
  std::smatch m;
  if (std::regex_search(operands_text, m, br)) {
    const auto it = ops.find("URANK" + m[1].str());
    if (it != ops.end() && !it->second.empty()) {
      const auto v = it->second[0];
      return v >= 0 ? std::optional<int>(int(v)) : std::nullopt;
    }
  }
  for (const auto &[name, v] : ops) {
    if (name.rfind("BRANK", 0) != 0) continue;
    bool all = !v.empty();
    for (const auto x : v) all &= x >= 0;
    if (all) return int(v[0]);
    return std::nullopt;
  }
  for (const auto &[name, v] : ops) {
    if (name.rfind("RANK", 0) != 0) continue;
    bool all = !v.empty();
    for (const auto x : v) all &= x >= 0;
    if (!all) return std::nullopt;
    for (const auto x : v) if (x != v[0]) return std::nullopt;
    return int(v[0]);
  }
  return std::nullopt;
}

std::optional<int> mbarrier_ur_word(
    const std::map<std::string, std::vector<std::int64_t>> &,
    const std::string &operands_text, const std::string &root) {
  static const std::regex br("\\[UR([0-9]+)");
  static const std::set<std::string> self = {"UTCBAR", "UCGABAR",
                                           "ARRIVES"};
  static const std::set<std::string> comp = {"UTMALDG", "UBLKCP"};
  std::smatch m;
  if (self.count(root)) {
    if (std::regex_search(operands_text, m, br))
      return std::stoi(m[1]);
    return std::nullopt;
  }
  if (comp.count(root)) {
    for (std::sregex_iterator it(operands_text.begin(),
                               operands_text.end(), br), e; it != e; ++it)
      return std::stoi((*it)[1]) + 1;
    return std::nullopt;
  }
  return std::nullopt;
}

int tma_coordinate_offset(const std::string &root, int) {
  if (root == "UTMALDG") return 2;
  if (root == "UTMASTG" || root == "UTMAREDG") return 1;
  if (root == "UTMAPF") return 0;
  return -1;
}

}  // namespace

const CaptureRules &capture_rules_hopper() {
  static const CaptureRules r{
      {"UTMALDG", "UBLKCP"}, {"UTCBAR", "UCGABAR", "ARRIVES"},
      {"SYNCS", "ARRIVES"},
      owner_rank_entry, mbarrier_ur_word, tma_coordinate_offset};
  return r;
}

}  // namespace cyclescope_convert::capture
