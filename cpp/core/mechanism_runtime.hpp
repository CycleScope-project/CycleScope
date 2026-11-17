#pragma once
// MechanismRuntime — runtime leaf selection over the compiled
// program tables (docs/en/design/engine.md).
//
// A USES row pins the walk at a NODE row (where conversion stopped).
// resolve() continues descending: each child is picked by its GRD
// row whose axis value comes from (a) the USES row's FACT entries
// (capture axes, named via STRS) or (b) caller-supplied engine facts
// (engine-owned axes by name; the axis's numeric id comes from the
// baked per-device sorted axis list, see generated/tree_axes.hpp).
// At the leaf every PARM field model is evaluated
// (constant/discrete(PTS)/linear/affine) and converted to cycles
// when the registry unit is ns-based.  Results are cached per
// (node, engine-axis value tuple).

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "job_format.hpp"
#include "job_reader.hpp"

namespace cyclescope {

class EngineError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Engine-supplied axis values, keyed by numeric axis id (GRD/AXIS row
// order).  Capture axes come from the USES row's FACT entries.
using EngineFacts = std::vector<std::pair<std::uint32_t, std::uint64_t>>;
using Fields = std::unordered_map<std::uint32_t, double>;

class MechanismRuntime {
 public:
  struct Options {
    std::uint32_t reference_sm_clock_khz = 0;
  };

  explicit MechanismRuntime(const Job& job,
                            std::string_view device,
                            Options options);

  // Resolve the leaf under `use` (a USES row index).  Returns
  // field_id -> value (cycles where the unit needs conversion) and
  // the resolved leaf node row via `leaf_node`.
  const Fields& resolve(std::uint64_t use_index,
                        const EngineFacts& engine_facts,
                        std::uint32_t* leaf_node = nullptr,
                        bool* depends_on_engine = nullptr);

  // Continue the walk from an arbitrary NODE row using `facts` as the
  // complete axis-value set (numeric axis id -> value).  A guard axis
  // with no value and no sibling match fails closed.  With
  // `not_applicable`, a guard whose axis has a value that matches no
  // child reports the program as not applicable (empty Fields,
  // *not_applicable = true) instead of throwing missing_fact.
  const Fields& resolve_node(std::uint32_t node, const EngineFacts& facts,
                             const char* where,
                             std::uint32_t* leaf_node = nullptr,
                             bool* not_applicable = nullptr);

  // Capture FACT pairs (numeric axis id, value) recorded on a USES row.
  EngineFacts use_capture_facts(std::uint64_t use_index) const;

  // Numeric axis id by name; nullopt when the AXIS table lacks it.
  std::optional<std::uint32_t> axis_id(std::string_view name) const;

  // PROG row lookup by mechanism name (STRS compared).
  std::optional<std::uint32_t> program_by_name(
      std::string_view mechanism) const;
  const MechanismProgram& program(std::uint32_t i) const {
    return job_.programs[i];
  }

  // Points (axis_value, value) of a linear field in the leaf resolved
  // for `use_index`: lets callers evaluate a field against a
  // runtime-varying axis (e.g. shared_utilization) that USES facts
  // cannot supply.  Empty when the field is absent or carries no
  // points.
  std::vector<std::pair<double, double>> linear_points(
      std::uint64_t use_index, const EngineFacts& engine_facts,
      std::uint32_t field_id) const;

  // Field value lookup helper.
  static double field(const Fields& f,
                      std::uint32_t id, double fallback = 0.0) {
    auto it = f.find(id);
    return it == f.end() ? fallback : it->second;
  }

  const Job& job() const { return job_; }
  std::uint64_t cache_hits() const { return cache_hits_; }
  std::uint64_t cache_misses() const { return cache_misses_; }

 private:
  struct CacheKey {
    std::uint32_t node;
    std::vector<std::uint64_t> axis_values;  // sorted (axis << 32 | value)
    bool operator==(const CacheKey&) const = default;
  };
  struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const {
      std::size_t h = k.node;
      for (auto v : k.axis_values) h = h * 1315423911u + v;
      return h;
    }
  };

  // value for the axis named `name` / numeric id `axis` used by a
  // guard row: capture FACT rows first, then engine facts.
  std::optional<std::uint64_t> axis_value(std::uint32_t axis_id,
                                        const EngineFacts& ef) const;
  // axis ids that count as "engine axes" for the current resolve
  // (populated from the caller's fact set before USES facts fold in);
  // flat flags indexed by axis id, grown for out-of-range ids
  std::vector<std::uint8_t> engine_axis_ids_;
  void dep_axis(std::uint32_t axis_id, bool* dep) const {
    if (dep && !*dep && axis_id < engine_axis_ids_.size() &&
        engine_axis_ids_[axis_id])
      *dep = true;
  }
  // lazily-formatted error location: either a literal (resolve_node
  // callers) or the packed (use_index, schema, node) resolve() form;
  // str() is only called on throw paths
  struct Where {
    const char* lit = nullptr;
    std::uint64_t use_index = 0;
    std::uint32_t schema = 0, node = 0;
    bool packed = false;
    std::string str() const {
      if (packed)
        return "use" + std::to_string(use_index) + " schema" +
               std::to_string(schema) + " node" + std::to_string(node);
      return lit ? std::string(lit) : std::string();
    }
  };
  double eval_field(const ParameterField& f, const EngineFacts& ef,
                    const Where& where, bool* dep = nullptr) const;
  const Fields& walk(std::uint32_t node, const EngineFacts& ef,
                     const Where& where,
                     std::uint32_t* leaf_node,
                     bool* dep = nullptr,
                     bool* not_applicable = nullptr);
  std::string_view str_at(std::uint32_t off) const;
  std::uint32_t axis_numeric_id(std::string_view name) const;
  bool field_needs_ns(std::uint16_t schema, std::uint16_t field) const = delete;

  const Job& job_;
  Options options_;
  // numeric axis id -> axis name (from the device's sorted list)
  std::vector<std::string_view> axis_names_;
  std::unordered_map<std::string_view, std::uint32_t> axis_ids_;
  // axes present in FACT rows keyed by STRS name are read directly.
  mutable std::unordered_map<CacheKey, Fields, CacheKeyHash> cache_;
  mutable std::unordered_map<CacheKey, std::uint32_t, CacheKeyHash>
      cache_leaf_;
  mutable std::uint64_t cache_hits_ = 0, cache_misses_ = 0;
  // per-USES capture facts as numeric (axis id, value) pairs sorted by
  // axis id, deduplicated last-wins like the old name-keyed map; a
  // kNone32 "axis id" stores the STRS offset of an unknown axis name
  // so the lookup still throws only when the row is actually used
  std::vector<EngineFacts> use_facts_num_;
  // reusable sort buffer for walk()'s cache-key build
  mutable EngineFacts key_sort_buf_;
  // sorted unique LEAF.parameter values; a leaf's PARM run ends at the
  // next begin (PARM rows are contiguous across leaves of one schema)
  std::vector<std::uint64_t> parm_begins_;
};

}  // namespace cyclescope
