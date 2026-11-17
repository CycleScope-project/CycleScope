#include "capture/control_flow.hpp"
#include <functional>
#include <memory>
#include <optional>
#include <set>

namespace cyclescope_convert::capture {
namespace {

const std::set<std::string> kPureRoots = {
    "BRA", "NOP", "MOV", "UMOV", "IADD3", "UIADD3", "VIADD",
    "IMAD", "UIMAD", "LOP3", "ULOP3", "ISETP", "UISETP",
    "PLOP3", "UPLOP3", "SEL", "USEL", "LEA", "ULEA", "SHF",
    "USHF", "PRMT", "UPRMT", "BFE", "UBFE", "BMSK", "POPC",
    "FLO", "IMNMX", "UIMNMX",
};
constexpr int kMaxParamRegions = 2;

std::string root_of(const std::string &op) {
  std::string t = op.substr(0, op.find('.'));
  for (auto &c : t) c = static_cast<char>(std::toupper(c));
  return t;
}
std::string upper(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::toupper(c));
  return s;
}
std::string strip(const std::string &s) {
  const auto a = s.find_first_not_of(" \t");
  return a == std::string::npos ? "" :
      s.substr(a, s.find_last_not_of(" \t") - a + 1);
}

// _btoken: first operand token that is B<n> (lstrip +- and .suffix)
std::string btoken(const StaticRow &row) {
  for (const auto &tok : row.inst.operands) {
    std::string t = strip(tok);
    while (!t.empty() && (t[0] == '+' || t[0] == '-')) t = t.substr(1);
    t = upper(t.substr(0, t.find('.')));
    if (t.size() >= 2 && t[0] == 'B') {
      bool digits = true;
      for (std::size_t i = 1; i < t.size(); ++i)
        if (!std::isdigit(static_cast<unsigned char>(t[i]))) digits = false;
      if (digits) return t;
    }
  }
  return "";
}

// bssy_close_pairs
std::map<std::uint64_t, std::uint64_t> bssy_close_pairs(
    const std::map<std::uint64_t, const StaticRow *> &rows_by_pc) {
  std::map<std::uint64_t, std::uint64_t> pairs;
  std::vector<std::pair<std::uint64_t, std::string>> stack;
  for (const auto &[pc, row] : rows_by_pc) {
    const std::string root = root_of(row->inst.opcode);
    if (root == "BSSY") {
      stack.emplace_back(pc, btoken(*row));
    } else if (root == "BSYNC") {
      const std::string tok = btoken(*row);
      for (int i = int(stack.size()) - 1; i >= 0; --i)
        if (stack[i].second == tok) {
          pairs[stack[i].first] = pc;
          stack.erase(stack.begin() + i);
          break;
        }
    }
  }
  return pairs;
}

struct Region {
  int pos = 0;
  std::string token;
  std::uint64_t close_pc = 0;
  std::uint32_t open_mask = 0, resolved = 0;
  std::vector<std::uint32_t> path_masks;
};

}  // namespace

Patches flow_patches(const std::vector<DynRecord> &records,
                     const StaticOf &static_of, const ByPc &by_pc,
                     std::uint64_t cta_key) {
  Patches patch;
  std::map<std::tuple<std::uint64_t, std::uint32_t, std::uint64_t,
                      std::int64_t>, int> loop_ordinals;
  std::vector<Region *> opened;
  struct Completed {
    int open_pos, close_pos, lanes, path_count;
  };
  std::vector<Completed> completed;
  std::vector<Region *> lane_stacks[32];
  std::map<int, std::map<std::uint64_t, std::uint64_t>> pairs_cache;
  std::vector<std::unique_ptr<Region>> region_store;

  for (int pos = 0; pos < int(records.size()); ++pos) {
    const auto &rec = records[pos];
    const StaticRow *row = static_of(rec);
    const std::string root = root_of(row->inst.opcode);
    if (root != "BRA" && root != "BSSY" && root != "BSYNC" &&
        root != "BREAK" && root != "EXIT")
      continue;
    const std::uint32_t mask = rec.exec_mask;
    if (root == "BRA") {
      const auto target = row->branch_target;
      if (target < 0 || std::uint64_t(target) >= row->inst.pc) continue;
      const auto &by_pc_map = by_pc(int(rec.function_id));
      std::map<std::string, int> counts;
      int span = 0;
      for (const auto &[pc, r] : by_pc_map)
        if (std::uint64_t(target) <= pc && pc <= row->inst.pc) {
          counts[root_of(r->inst.opcode)] += 1;
          ++span;
        }
      bool pure = true;
      for (const auto &[k, _n] : counts)
        if (!kPureRoots.count(k)) { pure = false; break; }
      const auto key = std::make_tuple(cta_key, rec.function_id,
                                       row->inst.pc,
                                       std::int64_t(target));
      const int ordinal = ++loop_ordinals[key];
      PatchEntry &e = patch[pos];
      e.puti("same_body_reuse", ordinal > 1 ? 1 : 0);
      e.puti("loop_body_bytes",
             std::int64_t(row->inst.pc) - target + 16);
      e.puti("loop_instruction_count", span);
      e.put("backward_loop_class_s",
            pure ? "pure_integer_register_control" : "mixed_or_stateful",
            false);
      const std::string lk =
          std::to_string(cta_key) + ":" + std::to_string(rec.function_id) +
          ":" + std::to_string(row->inst.pc) + ":" +
          std::to_string(target);
      e.put("_participants_key", lk, false);
      e.put("_loop_key", lk, false);
      continue;
    }
    if (root == "EXIT") {
      if (!mask) continue;
      for (int l = 0; l < 32; ++l)
        if (mask & (1u << l)) lane_stacks[l].clear();
      continue;
    }
    const std::string token = btoken(*row);
    if (token.empty()) continue;
    if (root == "BSSY") {
      if (!pairs_cache.count(int(rec.function_id)))
        pairs_cache[int(rec.function_id)] =
            bssy_close_pairs(by_pc(int(rec.function_id)));
      if (!mask) continue;
      region_store.push_back(std::make_unique<Region>());
      Region *reg = region_store.back().get();
      reg->pos = pos;
      reg->token = token;
      const auto it =
          pairs_cache[int(rec.function_id)].find(row->inst.pc);
      reg->close_pc = it == pairs_cache[int(rec.function_id)].end()
                          ? 0 : it->second;
      reg->open_mask = mask;
      for (int l = 0; l < 32; ++l)
        if (mask & (1u << l)) lane_stacks[l].push_back(reg);
      opened.push_back(reg);
      continue;
    }
    // BSYNC / BREAK
    if (!mask) continue;
    std::map<int, int> matched;
    bool ok = true;
    for (int l = 0; l < 32; ++l) {
      if (!(mask & (1u << l))) continue;
      int hits = -1, nhits = 0;
      for (int i = 0; i < int(lane_stacks[l].size()); ++i)
        if (lane_stacks[l][i]->token == token) { hits = i; ++nhits; }
      if (nhits != 1) { ok = false; break; }
      matched[l] = hits;
    }
    if (!ok || matched.empty()) continue;
    Region *region = lane_stacks[matched.begin()->first]
                                   [matched.begin()->second];
    for (const auto &[l, i] : matched)
      if (lane_stacks[l][i] != region) { region = nullptr; break; }
    if (!region) continue;
    region->resolved |= mask;
    if (root == "BSYNC") region->path_masks.push_back(mask);
    for (const auto &[l, i] : matched)
      lane_stacks[l].erase(lane_stacks[l].begin() + i);
    if (region->resolved == region->open_mask) {
      std::vector<int> lanes;
      for (const auto m : region->path_masks) {
        int n = 0;
        for (std::uint32_t x = m; x; x &= x - 1) ++n;
        lanes.push_back(n);
      }
      bool eq = !lanes.empty() &&
          std::all_of(lanes.begin(), lanes.end(),
                      [&](int v) { return v == lanes[0]; });
      if (region->path_masks.size() >= 2 && eq) {
        completed.push_back({region->pos, pos, lanes.empty() ? 0
                                                           : lanes[0],
                             int(region->path_masks.size())});
      } else {
        patch[region->pos].puti("control_path_count",
                                std::max<size_t>(1, region->path_masks.size()));
      }
    }
  }
  // compound selector: adjacent completed regions with equal lanes
  std::vector<std::vector<Completed>> runs;
  for (const auto &r : completed) {
    if (!runs.empty() && r.open_pos == runs.back().back().close_pos + 1 &&
        r.lanes == runs.back().back().lanes)
      runs.back().push_back(r);
    else
      runs.push_back({r});
  }
  for (auto &run : runs)
    for (std::size_t off = 0; off < run.size(); off += kMaxParamRegions) {
      const std::size_t end =
          std::min(run.size(), off + size_t(kMaxParamRegions));
      for (std::size_t i = off; i < end; ++i) {
        auto &e = patch[run[i].open_pos];
        e.puti("control_selector_present", 1);
        e.puti("sequential_region_count", std::int64_t(end - off));
        e.puti("control_path_count", run[i].path_count);
      }
    }
  return patch;
}

void fill_participating(std::vector<Patches *> &patches_by_warp) {
  std::map<std::string, std::set<int>> participants;
  for (std::size_t w = 0; w < patches_by_warp.size(); ++w)
    for (auto &[pos, e] : *patches_by_warp[w]) {
      const std::string k = e.pop("_participants_key");
      if (!k.empty()) participants[k].insert(int(w));
    }
  for (auto *p : patches_by_warp)
    for (auto &[pos, e] : *p) {
      const std::string k = e.pop("_loop_key");
      if (!k.empty())
        e.puti("control_participating_warps",
               std::int64_t(participants[k].size()));
    }
}

}  // namespace cyclescope_convert::capture
