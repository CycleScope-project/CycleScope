#pragma once
// §13.12 kernel.lifecycle: a single envelope applied once at the end
// of the core timeline.  Field presence is decided by the Fields map
// (a 0.0 value still counts as present).

#include <cstdint>

#include "mechanism_runtime.hpp"

namespace cyclescope {

// f: resolved schema-25 leaf fields: 2 envelope_cycles (linear in
// launch_warps_per_cta), 3 dual_sm_envelope_cycles.  Every launch is
// cold under the cache-control-all reference, so there is no
// first-launch extra (the predecessor warm/cold split, fields 0/1, is
// gone from both trees).
inline std::uint64_t lifecycle_envelope(const Fields& f,
                                 std::uint64_t sm_used) {
  std::uint64_t e = 0;
  if (const auto env = f.find(2); env != f.end()) {
    const auto dual = f.find(3);
    e += static_cast<std::uint64_t>(
        sm_used <= 2 && dual != f.end() ? dual->second : env->second);
  }
  return e;
}

}  // namespace cyclescope
