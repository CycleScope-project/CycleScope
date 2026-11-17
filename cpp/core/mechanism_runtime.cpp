#include "mechanism_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string_view>

#include "generated/parameter_schema.hpp"

namespace cyclescope {

namespace {

struct UnitNS {};  // marker

}  // namespace

MechanismRuntime::MechanismRuntime(const Job& job, std::string_view device,
                                   Options options)
    : job_(job), options_(options) {
  axis_names_.reserve(job_.axis_names.size());
  for (const auto& a : job_.axis_names)
    axis_names_.push_back(str_at(a.name));
  for (std::uint32_t i = 0; i < axis_names_.size(); ++i)
    axis_ids_[axis_names_[i]] = i;
  // cache FACT rows per USES row (fact_begin/fact_count) as numeric
  // (axis id, value) pairs: same last-wins dedup as the old map, sorted
  // by axis id so the resolve-order argument is deterministic
  use_facts_num_.resize(job_.uses.size());
  for (std::uint64_t i = 0; i < job_.uses.size(); ++i) {
    const auto& u = job_.uses[i];
    if (!u.fact_count) continue;
    std::unordered_map<std::string_view, std::uint64_t> m;
    for (std::uint32_t f = 0; f < u.fact_count; ++f) {
      const auto& fr = job_.facts[u.fact_begin + f];
      m[str_at(fr.axis)] = fr.value;
    }
    auto& out = use_facts_num_[i];
    out.reserve(m.size());
    for (const auto& [name, v] : m) {
      const auto it = axis_ids_.find(name);
      if (it == axis_ids_.end())
        // unknown axis: stash the STRS offset; the error is raised
        // lazily at use sites, same as the old lookup-on-use order
        out.emplace_back(kNone32,
            static_cast<std::uint64_t>(name.data() -
                                       job_.strings.data()));
      else
        out.emplace_back(it->second, v);
    }
    std::sort(out.begin(), out.end());
  }
  for (const auto& l : job_.leaves)
    if (l.parameter != kNone32) parm_begins_.push_back(l.parameter);
  std::sort(parm_begins_.begin(), parm_begins_.end());
  parm_begins_.erase(
      std::unique(parm_begins_.begin(), parm_begins_.end()),
      parm_begins_.end());
}

std::string_view MechanismRuntime::str_at(std::uint32_t off) const {
  if (off >= job_.strings.size())
    throw FormatError("STRS offset out of range");
  const char* p = job_.strings.data() + off;
  return {p, std::strlen(p)};
}

std::uint32_t MechanismRuntime::axis_numeric_id(
    std::string_view name) const {
  auto it = axis_ids_.find(name);
  if (it == axis_ids_.end())
    throw EngineError("unsupported:engine:unknown_axis (" +
                      std::string(name) + ")");
  return it->second;
}

std::optional<std::uint64_t> MechanismRuntime::axis_value(
    std::uint32_t axis_id, const EngineFacts& ef) const {
  for (const auto& [id, v] : ef)
    if (id == axis_id) return v;
  return std::nullopt;
}

double MechanismRuntime::eval_field(const ParameterField& f,
                                    const EngineFacts& ef,
                                    const Where& where, bool* dep) const {
  double v = 0.0;
  switch (f.model_kind) {
    case 0:  // constant
      v = f.value;
      break;
    case 1: {  // discrete: PTS row layout = (width axis values + value)
      const std::uint32_t width = static_cast<std::uint32_t>(f.slope);
      const bool fail_closed = f.intercept != 0.0;
      v = f.value;  // placeholder_default
      bool found = false;
      for (std::uint32_t i = 0; i < f.points_count; i += width + 1) {
        bool match = true;
        for (std::uint32_t a = 0; a < width; ++a) {
          const auto& row = job_.points[f.points_begin + i + a];
          const double want = row.axis_value;
          // every axis row carries its numeric axis id + 1 in the
          // value column (program_tables.py); a row without one is a
          // pre-fix segment whose axes beyond the first cannot be
          // identified -> fail closed
          std::uint32_t axis = f.axis;
          if (row.value > 0.0) {
            axis = static_cast<std::uint32_t>(row.value) - 1;
          } else if (a > 0) {
            throw EngineError(std::string(
                "unsupported:engine:discrete_axis_id (") + where.str() + ")");
          }
          auto av = axis_value(axis, ef);
          if (av) dep_axis(axis, dep);
          // an axis without a value cannot match a point on it
          if (!av || *av != static_cast<std::uint64_t>(want)) {
            match = false;
            break;
          }
        }
        if (match) {
          v = job_.points[f.points_begin + i + width].axis_value;
          found = true;
          break;
        }
      }
      if (!found && fail_closed) {
        // locate the miss: list each model axis with its fact value
        // (axes the facts don't carry print as `axis=?`)
        std::string axes;
        for (std::uint32_t a = 0; a < width; ++a) {
          const auto& row = job_.points[f.points_begin + a];
          std::uint32_t axis = f.axis;
          bool known = true;
          if (row.value > 0.0) {
            axis = static_cast<std::uint32_t>(row.value) - 1;
          } else if (a > 0) {
            known = false;
          }
          if (a) axes += ',';
          if (!known || axis >= axis_names_.size()) {
            axes += "?=?";
            continue;
          }
          axes += axis_names_[axis];
          axes += '=';
          auto av = axis_value(axis, ef);
          if (av) axes += std::to_string(*av);
          else axes += '?';
        }
        throw EngineError(std::string("unsupported:engine:discrete_miss (") +
                          where.str() + "; " + axes + ")");
      }
      break;
    }
    case 2: {  // linear via points (piecewise interpolation)
      auto av = axis_value(f.axis, ef);
      if (av) dep_axis(f.axis, dep);
      if (!av) {
        v = 0.0;
        break;
      }
      const double x = static_cast<double>(*av);
      // points sorted by axis value; linear interpolate/clamp
      const auto& pts = job_.points;
      const std::uint64_t n = f.points_count;
      if (n == 0) { v = 0.0; break; }
      if (x <= pts[f.points_begin].axis_value) {
        v = pts[f.points_begin].value;
        break;
      }
      if (x >= pts[f.points_begin + n - 1].axis_value) {
        v = pts[f.points_begin + n - 1].value;
        break;
      }
      for (std::uint64_t i = 1; i < n; ++i) {
        const auto& p1 = pts[f.points_begin + i - 1];
        const auto& p2 = pts[f.points_begin + i];
        if (x <= p2.axis_value) {
          const double t = (x - p1.axis_value) /
                           (p2.axis_value - p1.axis_value);
          v = p1.value + t * (p2.value - p1.value);
          break;
        }
      }
      break;
    }
    case 3: {  // affine: slope*x + intercept, value=domain lo/hi
      auto av = axis_value(f.axis, ef);
      if (av) dep_axis(f.axis, dep);
      if (!av) { v = f.intercept; break; }
      double x = static_cast<double>(*av);
      const double lo = f.value, hi = f.points_count ? 0.0 : f.value;
      (void)hi;
      v = f.slope * x + f.intercept;
      (void)lo;
      break;
    }
    default:
      throw EngineError(std::string("unsupported:engine:parm_model (") +
                        where.str() + ")");
  }
  // unit conversion from the registry (generated table): ns fields
  // scale to cycles (x GHz); per-ns rates scale to per-cycle (/ GHz).
  const std::uint32_t conv =
      core::field_unit_conversion(f.schema_id, f.field_id);
  if (conv) {
    if (!options_.reference_sm_clock_khz)
      throw EngineError("unsupported:engine:no_reference_clock ()");
    const double ghz = options_.reference_sm_clock_khz / 1e6;
    v = conv == 1 ? v * ghz : v / ghz;
  }
  return v;
}

const Fields& MechanismRuntime::resolve(std::uint64_t use_index,
                                        const EngineFacts& engine_facts,
                                        std::uint32_t* leaf_node,
                                        bool* depends_on_engine) {
  if (use_index >= job_.uses.size())
    throw EngineError("unsupported:engine:use_index ()");
  const auto& use = job_.uses[use_index];
  if (depends_on_engine) *depends_on_engine = false;
  // engine axis ids = the caller's set before USES capture facts
  engine_axis_ids_.assign(axis_names_.size(), 0);
  for (const auto& [k, v] : engine_facts) {
    if (k >= engine_axis_ids_.size())
      engine_axis_ids_.resize(k + 1, 0);
    engine_axis_ids_[k] = 1;
  }

  // fold the USES row's capture facts (STRS-named) into the numeric
  // axis-value set, then continue from the pinned node.
  EngineFacts ef = engine_facts;
  for (const auto& [k, v] : use_facts_num_[use_index]) {
    if (k == kNone32)
      (void)axis_numeric_id(str_at(static_cast<std::uint32_t>(v)));
    else
      ef.emplace_back(k, v);
  }
  // error location is packed and only formatted on throw paths
  Where where;
  where.packed = true;
  where.use_index = use_index;
  where.schema = use.schema_id;
  where.node = use.node;
  return walk(use.node, ef, where, leaf_node, depends_on_engine);
}

const Fields& MechanismRuntime::resolve_node(
    std::uint32_t node, const EngineFacts& facts, const char* where,
    std::uint32_t* leaf_node, bool* not_applicable) {
  Where w;
  w.lit = where;
  return walk(node, facts, w, leaf_node, nullptr, not_applicable);
}

EngineFacts MechanismRuntime::use_capture_facts(
    std::uint64_t use_index) const {
  EngineFacts out;
  if (use_index >= use_facts_num_.size()) return out;
  for (const auto& [k, v] : use_facts_num_[use_index]) {
    if (k == kNone32)
      out.emplace_back(axis_numeric_id(
          str_at(static_cast<std::uint32_t>(v))), 0);
    else
      out.emplace_back(k, v);
  }
  return out;
}

std::optional<std::uint32_t> MechanismRuntime::axis_id(
    std::string_view name) const {
  auto it = axis_ids_.find(name);
  if (it == axis_ids_.end()) return std::nullopt;
  return it->second;
}

std::optional<std::uint32_t> MechanismRuntime::program_by_name(
    std::string_view mechanism) const {
  for (std::uint32_t i = 0; i < job_.programs.size(); ++i)
    if (str_at(job_.programs[i].mechanism) == mechanism) return i;
  return std::nullopt;
}

const Fields& MechanismRuntime::walk(std::uint32_t node,
                                     const EngineFacts& ef,
                                     const Where& where,
                                     std::uint32_t* leaf_node,
                                     bool* dep,
                                     bool* not_applicable) {
  // cache key = (node, sorted (axis, value) pairs of the full fact set)
  CacheKey key{node, {}};
  key.axis_values.reserve(ef.size() * 2);
  {
    key_sort_buf_.assign(ef.begin(), ef.end());
    std::sort(key_sort_buf_.begin(), key_sort_buf_.end());
    for (const auto& [k, v] : key_sort_buf_) {
      key.axis_values.push_back(k);
      key.axis_values.push_back(v);
    }
  }
  auto cached = cache_.find(key);
  if (cached != cache_.end()) {
    ++cache_hits_;
    if (leaf_node) *leaf_node = cache_leaf_[key];
    return cached->second;
  }
  ++cache_misses_;

  // descend from `node` to a leaf
  while (job_.nodes[node].leaf == kNone32 ||
         job_.leaves[job_.nodes[node].leaf].parameter == kNone32) {
    const auto& n = job_.nodes[node];
    if (n.child_count == 0)
      throw EngineError(std::string("missing_fact:mechanism_leaf (") +
                        where.str() + ")");
    std::uint32_t next = kNone32;
    for (std::uint32_t c = 0; c < n.child_count; ++c) {
      const std::uint32_t child = n.child_begin + c;
      const auto& cn = job_.nodes[child];
      if (cn.guard_count == 0) { next = child; break; }
      const auto& g = job_.guards[cn.guard_begin];
      auto av = axis_value(g.axis, ef);
      if (!av) continue;
      dep_axis(g.axis, dep);
      bool ok = g.op == 0 ? (*av == g.low)
                          : (*av >= g.low && *av <= g.high);
      if (ok) { next = child; break; }
    }
    if (next == kNone32) {
      const auto& g0 = job_.guards[job_.nodes[n.child_begin].guard_begin];
      if (not_applicable && axis_value(g0.axis, ef)) {
        *not_applicable = true;
        static const Fields kEmpty;
        return kEmpty;
      }
      throw EngineError("missing_fact:axis:" +
          std::string(g0.axis < axis_names_.size()
              ? axis_names_[g0.axis] : std::string_view("?")) +
          " (" + where.str() + ")");
    }
    node = next;
  }
  const auto& leaf = job_.leaves[job_.nodes[node].leaf];
  auto& out = cache_[key];
  const std::uint64_t prm_begin = leaf.parameter;
  // PARM rows for this leaf: the contiguous run starting at parameter,
  // bounded by the next leaf's parameter begin
  const auto next = std::upper_bound(parm_begins_.begin(),
                                     parm_begins_.end(), prm_begin);
  const std::uint64_t prm_end =
      next == parm_begins_.end() ? job_.fields.size() : *next;
  for (std::uint64_t i = prm_begin;
       i < prm_end && i < job_.fields.size(); ++i) {
    const auto& f = job_.fields[i];
    // stop when the schema changes away from the leaf's first row
    if (f.schema_id != job_.fields[prm_begin].schema_id) break;
    out[f.field_id] = eval_field(f, ef, where, dep);
  }
  cache_leaf_[key] = node;
  if (leaf_node) *leaf_node = node;
  return out;
}

std::vector<std::pair<double, double>> MechanismRuntime::linear_points(
    std::uint64_t use_index, const EngineFacts& ef,
    std::uint32_t field_id) const {
  std::uint32_t node = 0;
  (void)const_cast<MechanismRuntime*>(this)->resolve(use_index, ef, &node);
  const auto& leaf = job_.leaves[job_.nodes[node].leaf];
  const std::uint64_t prm_begin = leaf.parameter;
  const auto next = std::upper_bound(parm_begins_.begin(),
                                     parm_begins_.end(), prm_begin);
  const std::uint64_t prm_end =
      next == parm_begins_.end() ? job_.fields.size() : *next;
  std::vector<std::pair<double, double>> pts;
  for (std::uint64_t i = prm_begin;
       i < prm_end && i < job_.fields.size(); ++i) {
    const auto& f = job_.fields[i];
    if (f.schema_id != job_.fields[prm_begin].schema_id) break;
    if (f.field_id != field_id || !f.points_count) continue;
    for (std::uint32_t j = 0; j < f.points_count; ++j) {
      const auto& p = job_.points[f.points_begin + j];
      pts.emplace_back(p.axis_value, p.value);
    }
    break;
  }
  return pts;
}

}  // namespace cyclescope
