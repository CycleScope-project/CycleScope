#pragma once
// §13.11 launch dispatch: first-wave entries from PLCE rows, queued
// clusters for unplaced CTAs, per-SM occupancy + 64-slot warp bitmap,
// placement-domain-aware cluster placement, serial entry spacing.

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

#include "job_reader.hpp"

namespace cyclescope {

struct DispatchParams {
  std::uint64_t first_wave_interval = 0;  // cycles
  double slot_free_to_entry = 0.0;       // cycles, fraction kept
  double serial_per_cta = 0.0;           // cycles/CTA, fraction kept
  bool enabled = false;
  double slot_free_to_entry_slow = 0.0;  // cycles; slow-gpc SMs
  std::uint64_t slow_gpc_mask = 0;       // bit g = topology gpc g
  double dependents_launch_serial = 0.0; // cycles/CTA on the serializer
};

// Event::kDispatch payload: ~0ull = retry tick, bit 63 set = release
// the slot of the CTA in the low bits, otherwise an index into
// dispatch_entries_ (admit_cluster)
constexpr std::uint64_t kDispatchReleaseTag = 1ull << 63;

struct DispatchState {
  std::deque<std::uint32_t> queue;  // pending cluster (segment) order
  std::uint64_t last_entry = 0;
};

class Dispatcher {
 public:
  void configure(std::uint32_t sm_count, std::uint32_t occupancy,
                 const std::vector<SmTopologyRow>& topo);
  // first free contiguous run of `n` warp slots on `sm`; -1 if none
  int slot_base(std::uint32_t sm, std::uint32_t n) const;
  bool has_capacity(std::uint32_t sm) const;
  // cluster placement of `size` CTAs (same warp_count): returns
  // chosen (sm, slot_base) per CTA in cluster order, or nullopt;
  // TOPO: all CTAs land in one placement_domain, pairs of 2 prefer
  // the same tpc_pair; SMs iterate in TOPO row order (or 0..n-1)
  std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>>
  place_cluster(std::uint32_t size, std::uint32_t warp_count_per_cta)
      const;
  void occupy(std::uint32_t sm, std::uint32_t slot_base, std::uint32_t n);
  void release(std::uint32_t sm, std::uint32_t slot_base,
               std::uint32_t n);

  std::uint32_t sm_count() const { return nsm_; }
  const SmTopologyRow* topo(std::uint32_t sm) const {
    return sm < nsm_ ? topo_[sm] : nullptr;
  }

 private:
  std::vector<std::uint64_t> sm_;        // warp-slot bitmaps
  std::vector<std::uint32_t> ctas_;      // resident CTA count
  std::vector<std::uint32_t> order_;     // SM iteration order (TOPO)
  std::vector<const SmTopologyRow*> topo_;  // per SM or null
  std::uint32_t nsm_ = 0, occupancy_ = 0;
};

}  // namespace cyclescope
