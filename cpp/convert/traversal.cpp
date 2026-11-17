// TreeRuntime over GSIMPRG01 + traverse/program_facts port.
#include "convert/traversal.hpp"

#include <cstring>

namespace cyclescope_convert {
namespace {

std::uint32_t rd32(const std::uint8_t *p) {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
std::uint64_t rd64(const std::uint8_t *p) {
  std::uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

std::string str_at(const std::uint8_t *strs, std::size_t n,
                   std::uint32_t off) {
  if (off >= n) return "";
  return reinterpret_cast<const char *>(strs + off);
}

}  // namespace

TreeRt::TreeRt(FactsFile &programs) {
  std::size_t n = 0;
  const auto *strs = programs.table("STRS", &n);
  const std::size_t strs_n = n;

  // AXIS + AXV + AXVN
  const auto *ax = programs.table("AXIS", &n);
  const auto *axv = programs.table("AXV ", &n);
  std::size_t axv_n = 0;
  programs.table("AXV ", &axv_n);
  const auto *axvn = programs.table("AXVN", &n);
  std::size_t axvn_n = 0;
  programs.table("AXVN", &axvn_n);
  const int n_ax = axv_n / 16;
  axes.resize(n_ax);
  for (int i = 0; i < n_ax; ++i) {
    const auto *r = axv + i * 16;
    const auto *an = ax + i * 4;
    axes[i].name = str_at(strs, strs_n, rd32(an));
    axes[i].kind = rd32(r);
    axes[i].owner = r[4];
    const auto vb = rd32(r + 8), vc = rd32(r + 12);
    for (std::uint32_t j = 0; j < vc; ++j) {
      const char *p =
          reinterpret_cast<const char *>(axvn + vb);
      // walk to j-th NUL string
      std::size_t off = vb;
      for (std::uint32_t k = 0; k < j; ++k) {
        while (off < axvn_n && axvn[off]) ++off;
        ++off;
      }
      axes[i].values.emplace_back(
          reinterpret_cast<const char *>(axvn + off));
    }
    for (std::size_t j = 0; j < axes[i].values.size(); ++j)
      axes[i].value_index.emplace(axes[i].values[j], j);  // first wins
    axis_index[axes[i].name] = i;
  }

  // NODE/GRD
  const auto *nd = programs.table("NODE", &n);
  const std::size_t n_nodes = n / 24;
  const auto *gd = programs.table("GRD ", &n);
  nodes.resize(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const auto *r = nd + i * 24;
    nodes[i].parent = (int)rd32(r);
    nodes[i].children_begin = (int)rd32(r + 4);
    nodes[i].children_count = (int)rd32(r + 8);
    nodes[i].guard_begin = (int)rd32(r + 12);
    nodes[i].guard_count = (int)rd32(r + 16);
    nodes[i].leaf_row = (int)rd32(r + 20);
  }
  const std::size_t n_g = n / 24;
  guards.resize(n_g);
  for (std::size_t i = 0; i < n_g; ++i) {
    const auto *r = gd + i * 24;
    guards[i].axis = rd32(r);
    guards[i].op = r[4];
    guards[i].lo = rd64(r + 8);
    guards[i].hi = rd64(r + 16);
  }

  // PROG
  const auto *pg = programs.table("PROG", &n);
  const std::size_t n_pg = n / 32;
  progs.reserve(n_pg);
  for (std::size_t i = 0; i < n_pg; ++i) {
    const auto *r = pg + i * 32;
    ProgRow p;
    p.name = str_at(strs, strs_n, rd32(r));
    p.event = r[4];
    p.schema_id = *(const std::uint16_t *)(r + 5);
    p.timing_kind = r[7];
    p.root = rd32(r + 8);
    p.field_mask = rd64(r + 12);
    p.op_begin = rd64(r + 20);
    p.class_count = rd32(r + 28);
    for (int j = 0; j < p.class_count; ++j) {
      std::size_t off = p.op_begin;
      for (int k = 0; k < j; ++k) {
        while (off < strs_n && strs[off]) ++off;
        ++off;
      }
      p.classes.emplace_back(
          reinterpret_cast<const char *>(strs + off));
    }
    prog_row[p.name] = (int)i;
    progs.push_back(std::move(p));
  }

  // by_class (skip event initialization=0)
  for (std::size_t i = 0; i < progs.size(); ++i) {
    if (progs[i].event == 0) continue;
    for (auto &c : progs[i].classes)
      by_class[c].push_back((int)i);
  }

  // kids are not contiguous in DFS order — rebuild via parent_row
  for (std::size_t i = 0; i < n_nodes; ++i)
    if (nodes[i].parent >= 0)
      nodes[nodes[i].parent].kids.push_back((int)i);

  // plan + eq_index + engine_child
  for (std::size_t nid = 0; nid < n_nodes; ++nid) {
    const auto &nr = nodes[nid];
    if (nr.kids.empty()) continue;
    std::vector<std::pair<int, int>> pl;
    bool eng = false;
    std::vector<std::pair<int, int>> cap;   // (kid, guard_idx)
    for (int kid : nr.kids) {
      const auto &kr = nodes[kid];
      const int gi = kr.guard_count ? kr.guard_begin : -1;
      pl.emplace_back(kid, gi);
      if (gi >= 0) {
        if (axes[guards[gi].axis].owner == 1) eng = true;
        else cap.emplace_back(kid, gi);
      }
    }
    plan[(int)nid] = pl;
    if (eng) {
      engine_child.insert((int)nid);
      continue;
    }
    if (!cap.empty()) {
      const int ax0 = guards[cap[0].second].axis;
      bool all_eq = true;
      Eq eq;
      eq.axis = ax0;
      for (auto &[kid, gi] : cap) {
        const auto &g = guards[gi];
        if (g.axis != ax0 || g.op != 0) {
          all_eq = false;
          break;
        }
        eq.by_value[g.lo] = kid;
      }
      for (auto &[kid, gi] : pl)
        if (gi < 0) eq.fallback.push_back(kid);
      if (all_eq) eq_index[(int)nid] = std::move(eq);
    }
  }

  // required axes from RAX/RAXV (per node row)
  const auto *rx = programs.table("RAX ", &n);
  const auto *rxv = programs.table("RAXV", &n);
  if (rx && rxv) {
    for (std::size_t i = 0; i < n_nodes; ++i) {
      const auto b = rd32(rx + i * 8), c = rd32(rx + i * 8 + 4);
      auto &out = required_axes[(int)i];
      for (std::uint32_t j = 0; j < c; ++j)
        out.push_back(axes[rd32(rxv + (b + j) * 4)].name);
    }
  }

  // predicated_programs: subtree contains a predicated_on guard
  const int po = axis_index.count("predicated_on")
                     ? axis_index.at("predicated_on")
                     : -1;
  for (std::size_t i = 0; i < progs.size(); ++i) {
    if (po < 0) break;
    std::vector<int> st{progs[i].root};
    while (!st.empty()) {
      const int nid = st.back();
      st.pop_back();
      const auto &nr = nodes[nid];
      if (nr.guard_count && guards[nr.guard_begin].axis == po) {
        predicated_programs.insert((int)i);
        break;
      }
      for (int k2 : nr.kids) st.push_back(k2);
    }
  }
}

namespace {

// axis value -> comparable u64 (enum: index in values; int: value)
bool ax_to_u64(const AxisDef &ax, const AxVal &v, std::uint64_t &out) {
  if (ax.kind == 1) {  // enum
    if (v.tag != AxVal::kStr) return false;
    return ax.index_of(v.s, out);
  }
  if (v.tag == AxVal::kInt) {
    out = (std::uint64_t)v.i;
    return true;
  }
  return false;
}

bool guard_match(const AxisDef &ax, const AxVal &v, const GuardRow &g) {
  std::uint64_t x = 0;
  if (!ax_to_u64(ax, v, x)) return false;
  if (g.op == 0) return x == g.lo;
  return g.lo <= x && x <= g.hi;
}

AxVal cached_axis(const TreeRt &rt, const std::string &name,
                  const StaticInfo &st, const DynOps *dyn, AxisCtx &ctx,
                  std::map<std::string, AxVal> &cache) {
  auto it = cache.find(name);
  if (it != cache.end()) return it->second;
  auto v = axis_value(name, st, dyn, ctx);
  cache[name] = v;
  return v;
}

}  // namespace

std::vector<UseRow> program_facts(
    const TreeRt &rt, const StaticInfo &st, const DynOps *dyn,
    AxisCtx &ctx, const std::string &cls,
    const std::vector<std::string> &classes) {
  const std::vector<std::string> cls_only{cls};
  const std::vector<std::string> &cls_set =
      classes.empty() ? cls_only : classes;
  const bool executing =
      !dyn || (dyn->active_mask & dyn->predicate_mask) != 0;
  std::vector<int> cand;
  std::set<int> seen;
  for (auto &m : cls_set) {
    auto it = rt.by_class.find(m);
    if (it == rt.by_class.end()) continue;
    for (auto p : it->second)
      if (seen.insert(p).second) cand.push_back(p);
  }
  if (!executing) {
    std::vector<int> keep;
    for (auto p : cand)
      if (rt.predicated_programs.count(p)) keep.push_back(p);
    cand = std::move(keep);
  }
  std::vector<UseRow> out;
  const int po_axis = rt.axis_index.count("operation_class")
                          ? rt.axis_index.at("operation_class")
                          : -1;
  for (auto pi : cand) {
    const auto &prog = rt.progs[pi];
    std::string member = cls;
    if (!prog.classes.empty()) {
      bool found = false;
      for (auto &c : prog.classes)
        if (c == cls) found = true;
      if (!found) {
        member = cls;
        for (auto &c : cls_set) {
          for (auto &pc : prog.classes)
            if (pc == c) member = c;
          if (member != cls) break;
        }
      }
    }
    std::map<std::string, AxVal> facts, cache;
    int nid = prog.root;
    int stop_node = -1;
    bool applicable = true;
    std::string lost;
    while (true) {
      const auto &nr = rt.nodes[nid];
      if (nr.kids.empty()) {
        stop_node = nid;
        break;
      }
      int nxt = -1;
      auto eit = rt.eq_index.find(nid);
      if (eit != rt.eq_index.end()) {
        const auto &eq = eit->second;
        const auto &an = rt.axes[eq.axis].name;
        AxVal val;
        if (an == "operation_class") {
          val = AxVal::sv(member);
          for (auto &cand2 : cls_set) {
            std::uint64_t idx = 0;
            if (ax_to_u64(rt.axes[eq.axis], AxVal::sv(cand2), idx) &&
                eq.by_value.count(idx)) {
              nxt = eq.by_value.at(idx);
              break;
            }
          }
        } else {
          val = cached_axis(rt, an, st, dyn, ctx, cache);
          if (val.is_na()) {
            applicable = false;
            break;
          }
          if (val.is_none()) {
            lost = "axis:" + an;
            break;
          }
          std::uint64_t x = 0;
          if (ax_to_u64(rt.axes[eq.axis], val, x) &&
              eq.by_value.count(x))
            nxt = eq.by_value.at(x);
        }
        if (nxt < 0 && !eq.fallback.empty()) nxt = eq.fallback.back();
        facts[an] = val;
      } else {
        for (auto &[kid, gi] : rt.plan.at(nid)) {
          if (gi < 0) {
            nxt = kid;
            continue;
          }
          const auto &ax = rt.axes[rt.guards[gi].axis];
          const auto &g = rt.guards[gi];
          if (ax.owner == 1) {
            stop_node = nid;
            break;
          }
          if (ax.name == "operation_class") {
            const auto val = AxVal::sv(member);
            // guard_set_match: any classes member satisfies
            bool ok = false;
            if (g.op == 0) {
              std::uint64_t x;
              if (ax_to_u64(ax, val, x)) ok = (x == g.lo);
              else {
                for (auto &c2 : cls_set) {
                  std::uint64_t x2;
                  if (ax_to_u64(ax, AxVal::sv(c2), x2) && x2 == g.lo)
                    ok = true;
                }
              }
            } else {
              for (auto &c2 : cls_set) {
                std::uint64_t x2;
                if (ax_to_u64(ax, AxVal::sv(c2), x2) &&
                    g.lo <= x2 && x2 <= g.hi)
                  ok = true;
              }
            }
            if (!ok) continue;
            facts[ax.name] = val;
            nxt = kid;
            break;
          }
          auto val = cached_axis(rt, ax.name, st, dyn, ctx, cache);
          if (val.is_na()) {
            applicable = false;
            break;
          }
          if (val.is_none()) {
            lost = "axis:" + ax.name;
            break;
          }
          if (!guard_match(ax, val, g)) continue;
          facts[ax.name] = val;
          nxt = kid;
          break;
        }
      }
      if (!lost.empty() || !applicable || stop_node >= 0) break;
      if (nxt < 0) {
        applicable = false;
        break;
      }
      nid = nxt;
    }
    UseRow row;
    row.program = prog.name;
    row.prog = pi;
    if (!lost.empty()) {
      row.missing = true;
      row.missing_key = lost;
      out.push_back(std::move(row));
      continue;
    }
    if (!applicable) continue;
    row.node = stop_node;
    row.cache = std::move(cache);
    // required leaf axes pass
    bool skip = false;
    std::string lost2;
    static const std::vector<std::string> kNoRequired;
    const auto &req =
        rt.required_axes.count(stop_node) ? rt.required_axes.at(stop_node)
                                          : kNoRequired;
    for (auto &an : req) {
      if (facts.count(an)) continue;
      auto val = cached_axis(rt, an, st, dyn, ctx, row.cache);
      if (val.is_na()) {
        skip = true;
        break;
      }
      if (val.is_none()) {
        lost2 = "axis:" + an;
        break;
      }
      facts[an] = val;
    }
    if (!lost2.empty()) {
      UseRow m;
      m.missing = true;
      m.missing_key = lost2;
      m.program = prog.name;
      m.prog = pi;
      out.push_back(std::move(m));
      continue;
    }
    if (skip) continue;
    row.facts = std::move(facts);
    out.push_back(std::move(row));
  }
  return out;
}

}  // namespace cyclescope_convert
