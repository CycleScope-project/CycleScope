#include "capture/poll_folding.hpp"

#include <optional>
#include <string>

namespace cyclescope_convert::capture {
namespace {

const std::set<std::string> kNoSideEffect = {
    "BRA", "BRX", "BSSY", "BSYNC", "NANOSLEEP", "VOTEU", "VOTE",
    "ISETP", "FSETP", "DSETP", "PLOP3", "LOP3", "SEL", "FSEL", "MOV",
    "UMOV", "IMAD", "IADD3", "VIADD", "PRMT", "FLO", "BREV", "POPC",
    "UPOPC", "S2R", "S2UR", "CS2R", "LEA", "ULEA", "UMUF", "SHF", "USHF",
    "I2I", "F2F", "R2UR", "NOP", "USCHED", "WARPSYNC", "YIELD",
    "UIADD3", "UISETP", "ULOP3", "USEL", "UPRMT", "UIMAD", "UFLO",
};
const std::set<std::string> kGsyncGapExtra = {"LDC", "ULDC", "LDCU"};

std::optional<bool> poll_ready(const DynRecord &rec) {
  if (!rec.has_after_ready) return std::nullopt;
  return (rec.after_ready & rec.exec_mask) == rec.exec_mask;
}

std::int64_t r_value(const DynRecord &rec, const std::string &op) {
  const auto it = rec.ops.find(op);
  if (it == rec.ops.end() || it->second.empty()) return 0;
  return it->second[0];
}

std::pair<std::uint64_t, int> poll_key(const DynRecord &rec,
                                       const StaticRow &row) {
  const std::uint64_t addr =
      (!rec.mrefs.empty() && !rec.mrefs[0].empty()) ? rec.mrefs[0][0] : 0;
  int parity = 0;
  for (auto it = row.ops_layout.rbegin();
       it != row.ops_layout.rend(); ++it) {
    const std::string &op = *it;
    if (op.size() && op[0] == 'R' && op.rfind("UR", 0) != 0 &&
        op.rfind("RANK", 0) != 0) {
      parity = int((static_cast<std::uint64_t>(r_value(rec, op)) >> 31) & 1);
      break;
    }
  }
  return {addr, parity};
}

bool gap_free(const std::string &opcode, const std::string &root,
              bool gsync) {
  if (kNoSideEffect.count(root)) return true;
  if (gsync) {
    if (kGsyncGapExtra.count(root)) return true;
    if (root == "CCTL") {
      const std::string up = opcode;
      std::size_t dot = up.find('.');
      if (dot != std::string::npos &&
          up.find("IVALL", dot) != std::string::npos)
        return true;
    }
  }
  return false;
}

std::string root_of(const std::string &opcode) {
  std::string t = opcode;
  const auto dot = t.find('.');
  if (dot != std::string::npos) t = t.substr(0, dot);
  for (auto &c : t) c = static_cast<char>(std::toupper(c));
  return t;
}
std::string upper(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::toupper(c));
  return s;
}

}  // namespace

std::pair<std::set<int>, int>
fold_warp(const std::vector<DynRecord> &records,
          const StaticOf &static_of) {
  std::set<int> folded;
  std::vector<int> polls;
  for (int i = 0; i < int(records.size()); ++i)
    if (upper(static_of(records[i])->inst.opcode).find("PHASECHK") !=
        std::string::npos)
      polls.push_back(i);
  int dropped = 0;
  std::size_t pos = 0;
  while (pos < polls.size()) {
    const int i0 = polls[pos];
    const auto key = poll_key(records[i0], *static_of(records[i0]));
    if (poll_ready(records[i0]) != false) { ++pos; continue; }
    std::vector<int> run{i0};
    std::size_t j = pos + 1;
    while (j < polls.size() &&
           poll_key(records[polls[j]], *static_of(records[polls[j]])) ==
               key &&
           poll_ready(records[polls[j]]) == false) {
      run.push_back(polls[j]);
      ++j;
    }
    const auto ready_next =
        j < polls.size() &&
        poll_key(records[polls[j]], *static_of(records[polls[j]])) == key &&
        poll_ready(records[polls[j]]) == true;
    if (!ready_next) { pos = j; continue; }
    const int end_pos = polls[j];
    // maximal clean suffix: scan gaps backwards from the ready poll
    std::vector<int> seq = run;
    seq.push_back(polls[j]);
    int start = -1;
    for (int q = int(seq.size()) - 1; q > 0; --q) {
      bool clean = true;
      for (int k = seq[q - 1] + 1; k < seq[q]; ++k) {
        const auto *sr = static_of(records[k]);
        if (!gap_free(sr->opcode_text, root_of(sr->inst.opcode), false)) {
          clean = false;
          break;
        }
      }
      if (!clean) break;
      start = q - 1;
    }
    const bool fold_ok = start >= 0;
    if (fold_ok) {
      for (std::size_t u = start; u < run.size(); ++u) {
        folded.insert(run[u]);
        ++dropped;
      }
      for (int k = run[start] + 1; k < end_pos; ++k) folded.insert(k);
    }
    pos = fold_ok ? j + 1 : j;
  }
  return {folded, dropped};
}

}  // namespace cyclescope_convert::capture
