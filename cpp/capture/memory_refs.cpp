// capture_pipeline._memory_refs verbatim port.
#include "capture/memory_refs.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <set>

namespace cyclescope_convert::capture {
namespace {

// root classes / access kinds / decoded width live in StaticDerived
// (operand_facts.cpp prepare_static_row)
constexpr int kSector = 32;

std::string space_of(const DynRecord &rec, const StaticRow &srow,
                     const std::string &static_space) {
  if (static_space == "generic") {
    const auto it = rec.ops.find("SPACE");
    if (it != rec.ops.end() && it->second.size() >= 3) {
      const auto &v = it->second;
      if (v[2]) return "shared_cluster";
      if (v[1]) return "shared";
      if (v[0]) return "global";
    }
  }
  return static_space;
}
std::vector<std::int64_t> bare_ur_tokens(const StaticRow &srow) {
  static const std::regex re("^UR([0-9]+)$");
  std::vector<std::int64_t> out;
  for (const auto &t : srow.inst.operands) {
    if (t.find('[') != std::string::npos) continue;
    std::string s = t.substr(t.find_first_not_of(" \t"));
    std::smatch m;
    if (std::regex_match(s, m, re))
      out.push_back(std::stoll(m[1]));
  }
  return out;
}

std::vector<MemoryReference> constant_refs(
    const DynRecord &rec, const StaticRow &srow,
    const SemanticOperands *ops, std::map<std::string, int> &missing,
    const StaticDerived &d) {
  if (rec.exec_mask == 0) return {};
  std::string token;
  for (const auto &t : srow.inst.operands) {
    const auto s = t.find_first_not_of(" \t");
    if (t.compare(s, 2, "c[") == 0) { token = t.substr(s); break; }
  }
  static const std::regex cre(
      R"(^c\[(0x[0-9a-fA-F]+)\]\[(?:(UR[0-9]+|R[0-9]+|RZ|URZ)\+?)?(0x[0-9a-fA-F]+)?\]$)");
  std::smatch m;
  if (token.empty() || !std::regex_match(token, m, cre)) {
    missing["constant_operand"] += 1;
    return {};
  }
  const std::uint64_t bank = std::stoull(m[1].str(), nullptr, 16);
  const std::int64_t imm =
      m[3].matched ? std::stoll(m[3].str(), nullptr, 16) : 0;
  const std::string reg = m[2].matched ? m[2].str() : "";
  const auto addr = [&](std::int64_t off) {
    return std::int64_t((bank << 32) |
                        (std::uint64_t(off) & 0xFFFFFFFFull));
  };
  const int lanes = __builtin_popcount(rec.exec_mask);
  std::vector<std::int64_t> addrs;
  if (reg.empty() || reg == "RZ" || reg == "URZ") {
    for (int i = 0; i < lanes; ++i) addrs.push_back(addr(imm));
  } else if (reg[0] == 'U') {
    if (!ops || !ops->uniform.count(reg)) {
      missing["constant_address"] += 1;
      return {};
    }
    for (int i = 0; i < lanes; ++i)
      addrs.push_back(addr(ops->uniform.at(reg) + imm));
  } else {
    const auto it = ops ? ops->lane.find(reg) : ops->lane.end();
    if (!ops || it == ops->lane.end() || it->second.empty()) {
      missing["constant_address"] += 1;
      return {};
    }
    for (const auto v : it->second) addrs.push_back(addr(v + imm));
  }
  int width = 4;
  // opcode_width_bytes lives in operand_facts.cpp's anonymous
  // namespace; replicate the MissingFact surface via a helper
  // (kept in operand_facts.hpp? falls back to 4 as Python does)
  if (!d.width_error.empty()) missing["opcode_width"] += 1;
  else width = d.width.value_or(0);
  return {{.space = "constant", .access_kind = "load",
           .width_bytes = width,
           .lane_addresses = addrs, .lane_mask = rec.exec_mask}};
}

}  // namespace

std::vector<MemoryReference>
memory_refs(const DynRecord &rec, const StaticRow &srow,
            const CaptureRules &cap_rules, TmState &tmaps,
            std::map<std::string, int> &missing,
            std::map<std::string, int> &unsupported,
            const std::map<int, std::vector<std::int64_t>> &ranks,
            const SemanticOperands *ops) {
  const StaticDerived &d = derived_of(srow, cap_rules);
  if (d.no_mref_root) return {};
  if (d.const_root)
    return constant_refs(rec, srow, ops, missing, d);
  if (!srow.mref_count && !d.is_tma && !d.is_bulk) return {};
  if (rec.exec_mask == 0 && (d.is_tma || d.is_bulk)) return {};
  const std::string &root = d.root;
  const std::string &where = srow.inst.opcode;
  const std::string static_space = srow.inst.memory_space;
  std::optional<std::int64_t> multicast;
  if (ops) {
    const auto it = ops->uniform.find("multicast_mask");
    if (it != ops->uniform.end()) multicast = it->second;
  }
  std::optional<int> destination_count;
  if (multicast)
    destination_count = __builtin_popcountll(std::uint64_t(*multicast));
  std::vector<MemoryReference> out;

  if (d.is_tma) {
    const std::optional<std::uint32_t> tm_id =
        ops ? ops->tensor_map : std::nullopt;
    std::vector<std::int64_t> coords =
        ops ? ops->tensor_coordinates : std::vector<std::int64_t>{};
    const TmEntry *entry = nullptr;
    if (tm_id) {
      tmaps.used.insert(*tm_id);
      const auto eit = tmaps.entries.find(*tm_id);
      if (eit != tmaps.entries.end()) entry = &eit->second;
      if (!tmaps.facts.count(*tm_id) && entry == nullptr) {
        missing["tensor_map_row"] += 1;
        tmaps.facts[*tm_id] = TmResult{};
      }
      if (!tmaps.facts.count(*tm_id)) {
        try {
          const auto r = tensor_map_entry(*entry, srow.opcode_text, where);
          tmaps.models[*tm_id] = r;
          tmaps.facts[*tm_id] = r;
        } catch (const std::runtime_error &) {
          missing["tensor_map_decode"] += 1;
          tmaps.facts[*tm_id] = TmResult{};
        }
      }
    }
    const TmResult *facts =
        tm_id && tmaps.facts.count(*tm_id) ? &tmaps.facts[*tm_id]
                                           : nullptr;
    if (!tm_id) missing["tensor_map"] += 1;
    std::vector<std::int64_t> sectors;
    int r_cut_rows = 0;
    std::optional<std::int64_t> transfer;
    if (facts && facts->global_bytes) transfer = facts->global_bytes;
    const bool have_geo = facts && facts->geometry.rank > 0;
    if (have_geo && !coords.empty()) {
      if (facts->api == "im2col") {
        if (facts->geometry.element_strides.empty() ||
            facts->geometry.element_strides[0] != 1) {
          unsupported["tma_im2col_element_stride"] += 1;
        } else {
          const auto bare = bare_ur_tokens(srow);
          const int want = root == "UTMALDG" ? 1 : 0;
          std::vector<std::int64_t> offs;
          bool offs_ok = false;
          if (int(bare.size()) != want) {
            unsupported["tma_im2col_operands"] += 1;
          } else if (!bare.empty()) {
            if (!ops || !ops->uniform.count(
                            "UR" + std::to_string(bare[0]))) {
              missing["im2col_offsets"] += 1;
            } else {
              offs = im2col_offsets(
                  facts->geometry.rank,
                  std::uint64_t(
                      ops->uniform.at("UR" + std::to_string(bare[0]))));
              offs_ok = true;
            }
          } else {
            offs = std::vector<std::int64_t>(facts->geometry.rank - 2, 0);
            offs_ok = true;
          }
          if (offs_ok) {
            try {
              sectors = im2col_box_sectors(
                  *facts,
                  std::vector<std::int64_t>(
                      coords.begin(),
                      coords.begin() + facts->geometry.rank),
                  offs);
            } catch (const std::runtime_error &) {
              unsupported["tma_im2col_coordinate"] += 1;
            }
          }
        }
      } else {
        try {
          int cut_rows = 0;
          if (root == "UTMALDG") {
            // tma_oob v2: partial cut along dimension 0 costs the
            // in-bounds rows; whole rows outside along other dims
            // and fully inside / outside boxes cost nothing.
            const auto rect =
                tiled_box_rectangle(facts->geometry, coords);
            const auto &es = facts->geometry.element_strides;
            const std::int64_t n0 =
                (facts->geometry.box_dim[0] + es[0] - 1) / es[0];
            if (0 < rect.counts[0] && rect.counts[0] < n0) {
              std::int64_t rows = 1;
              for (std::size_t k = 1; k < rect.counts.size(); ++k)
                rows *= rect.counts[k];
              cut_rows = int(rows);
            }
          }
          sectors = tma_box_sectors(facts->geometry, coords);
          r_cut_rows = cut_rows;
        } catch (const std::runtime_error &) {
          unsupported["tma_box"] += 1;
        }
      }
    } else if (have_geo) {
      missing["tensor_coordinates"] += 1;
    } else if (facts && facts->api == "im2col") {
      unsupported["tma_im2col_box"] += 1;
    }
    MemoryReference r;
    r.space = "global";
    r.access_kind = d.access_kind;
    r.width_bytes = kSector;
    r.lane_addresses = sectors;
    r.lane_mask = 0;
    r.destination_count = destination_count;
    r.transfer_bytes = transfer;
    r.cut_rows = r_cut_rows;
    out.push_back(std::move(r));
    return out;
  }

  if (d.is_bulk) {
    {
      // shared -> shared (.S.S) cluster copy: no global side, no ref
      std::string up = srow.opcode_text;
      for (auto &c : up) c = char(std::toupper(c));
      std::vector<std::string> parts;
      std::size_t st = 0;
      while (true) {
        const std::size_t dot = up.find('.', st);
        parts.push_back(up.substr(st, dot == std::string::npos ? std::string::npos : dot - st));
        if (dot == std::string::npos) break;
        st = dot + 1;
      }
      std::optional<std::int64_t> size0;
      if (ops) {
        const auto it = ops->uniform.find("bytes");
        if (it != ops->uniform.end()) size0 = it->second;
      }
      if (!(parts.size() >= 3 && (parts[1] == "G" || parts[2] == "G"))) {
        if (!size0) missing["bulk_copy_bytes"] += 1;
        MemoryReference r;
        r.space = "shared";
        r.access_kind = d.access_kind;
        r.width_bytes = kSector;
        r.lane_mask = 0;
        r.destination_count = destination_count;
        r.transfer_bytes = size0;
        out.push_back(std::move(r));
        return out;
      }
    }
    std::optional<std::int64_t> size;
    if (ops) {
      const auto it = ops->uniform.find("bytes");
      if (it != ops->uniform.end()) size = it->second;
    }
    if (!size) missing["bulk_copy_bytes"] += 1;
    // global-side base = the .G bracket's UR pair (operand_facts.
    // bulk_global_address); older traces without it fall back to the
    // NVBit mref and are reported as missing
    std::optional<std::uint64_t> base;
    {
      std::string up = srow.opcode_text;
      for (auto &c : up) c = char(std::toupper(c));
      std::vector<std::string> parts;
      std::size_t st = 0;
      while (true) {
        const std::size_t dot = up.find('.', st);
        parts.push_back(up.substr(st, dot == std::string::npos ? std::string::npos : dot - st));
        if (dot == std::string::npos) break;
        st = dot + 1;
      }
      int index = -1;
      if (parts.size() >= 3) {
        if (parts[1] == "G") index = 0;
        else if (parts[2] == "G") index = 1;
      }
      if (index >= 0 && ops) {
        std::vector<int> brackets;
        static const std::regex bre("\\[UR([0-9]+)");
        for (const auto &t : srow.inst.operands) {
          std::smatch m;
          if (std::regex_search(t, m, bre)) brackets.push_back(std::stoi(m[1]));
        }
        if (index < int(brackets.size())) {
          const auto lo = ops->uniform.find("UR" + std::to_string(brackets[index]));
          const auto hi = ops->uniform.find("UR" + std::to_string(brackets[index] + 1));
          if (lo != ops->uniform.end() && hi != ops->uniform.end())
            base = (std::uint64_t(hi->second & 0xFFFFFFFF) << 32) |
                   std::uint64_t(lo->second & 0xFFFFFFFF);
        }
      }
    }
    if (!base) {
      missing["bulk_global_address"] += 1;
      if (!rec.mrefs.empty() && !rec.mrefs[0].empty()) base = rec.mrefs[0][0];
    }
    const auto sectors =
        base ? bulk_sectors(*base, std::uint64_t(size.value_or(0)))
             : std::vector<std::int64_t>{};
    MemoryReference r;
    r.space = "global";
    r.access_kind = d.access_kind;
    r.width_bytes = kSector;
    r.lane_addresses = sectors;
    r.lane_mask = 0;
    r.destination_count = destination_count;
    r.transfer_bytes = size;
    out.push_back(std::move(r));
    return out;
  }

  int width = 0;
  if (!d.width_error.empty()) missing["opcode_width"] += 1;
  else width = d.width.value_or(0);
  const std::string &kind = d.access_kind;
  // LDGSTS: mref0 shared dest, mref1 global source (global first)
  std::vector<int> order;
  if (root == "LDGSTS" && srow.mref_count == 2) {
    order = {1, 0};
  } else {
    for (int i = 0; i < int(srow.mref_count); ++i) order.push_back(i);
  }
  for (const int mi : order) {
    const std::vector<std::uint64_t> *addrs =
        mi < int(rec.mrefs.size()) ? &rec.mrefs[mi] : nullptr;
    std::string space;
    if (root == "LDGSTS") {
      space = mi == 1 ? "global" : "shared";
    } else if (mi == 0) {
      space = space_of(rec, srow, static_space);
    } else {
      space = static_space;
    }
    std::optional<int> remote;
    const auto rit = ranks.find(mi);
    if (rit != ranks.end()) {
      space = "shared_cluster";
      for (const auto v : rit->second)
        if (v >= 0) { remote = int(v); break; }
    }
    auto dcount = destination_count;
    if (root == "STAS" && rit != ranks.end()) {
      std::set<std::int64_t> s(rit->second.begin(), rit->second.end());
      s.erase(-1);
      // count of distinct v>=0
      int n = 0;
      for (const auto v : s) if (v >= 0) ++n;
      dcount = n;
    }
    MemoryReference r;
    r.space = space.empty() ? "global" : space;
    r.access_kind = kind;
    r.width_bytes = width;
    if (addrs)
      for (const auto a : *addrs)
        r.lane_addresses.push_back(std::int64_t(a));
    r.lane_mask = rec.exec_mask;
    r.remote_rank = remote;
    r.destination_count = dcount;
    out.push_back(std::move(r));
  }
  return out;
}

}  // namespace cyclescope_convert::capture
