// _memory_refs / _space_of / _access_kind / _CONSTANT_ROOTS /
// NO_MEMORY_REFERENCE_ROOTS (capture_pipeline.py) -> MemoryReference
// rows for MREF/MREX/ADDR.
#pragma once
#include "capture/operand_facts.hpp"
#include "capture/static_code.hpp"
#include "capture/tensor_map.hpp"
#include "capture/trace_reader.hpp"
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace cyclescope_convert::capture {

struct MemoryReference {
  std::string space;           // global/shared/local/constant/...
  std::string access_kind;     // load/store/atomic/async_copy/...
  int width_bytes = 0;
  std::vector<std::int64_t> lane_addresses;
  std::uint32_t lane_mask = 0;
  std::optional<int> remote_rank;
  std::optional<int> destination_count;
  std::optional<std::int64_t> transfer_bytes;
  std::optional<int> cut_rows;   // tiled TMA: in-bounds rows on dim-0 cut
  int cache_hint = 0;
};

// per-launch tensor map state shared across records
struct TmState {
  std::map<int, TmEntry> entries;                    // load_tensor_maps
  std::map<int, TmResult> models;                    // decoded TensorMap
  std::map<int, TmResult> facts;                     // geometry facts
  std::set<int> used;                                // used_maps
};

// may throw missing_fact/unsupported like Python.
std::vector<MemoryReference>
memory_refs(const DynRecord &rec, const StaticRow &srow,
            const CaptureRules &cap_rules, TmState &tmaps,
            std::map<std::string, int> &missing,
            std::map<std::string, int> &unsupported,
            const std::map<int, std::vector<std::int64_t>> &ranks,
            const SemanticOperands *ops);

}  // namespace cyclescope_convert::capture
