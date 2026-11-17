#pragma once
// CachePolicy — weighted global LRU + cache_hint behaviour
// (docs/en/design/engine.md).  Weights: home 1 + reader-side replica 1 + dirty 1.
// Hints (MACC.cache_hint): L1 NA/EF/EL/EU policies and L2 EF share —
// hint 0 means "default" (Hopper jobs); the Blackwell EF share
// parameter arrives via the cache_hint leaf.
#include <cstdint>

namespace cyclescope {

class CachePolicy {
 public:
  struct L2EvictFirst {
    double share = 0.0;
    std::uint64_t resident_lines = 0;
  };
  // weight for global-LRU pressure accounting
  static std::uint32_t lru_weight(bool home, bool replica, bool dirty) {
    return (home ? 1u : 0u) + (replica ? 1u : 0u) + (dirty ? 1u : 0u);
  }
  // L1 hint -> allocate policy (MACC.cache_hint low nibble)
  enum class L1Hint : std::uint8_t { allocate = 0, evict_first,
      evict_last, evict_unchanged, no_allocate };
  static L1Hint l1_hint(std::uint32_t cache_hint) {
    switch (cache_hint & 0xF) {
      case 1: return L1Hint::evict_first;
      case 2: return L1Hint::evict_last;
      case 4: return L1Hint::evict_unchanged;
      case 12: return L1Hint::no_allocate;
      default: return L1Hint::allocate;
    }
  }
  static bool l2_evict_first(std::uint32_t cache_hint) {
    return ((cache_hint >> 4) & 0xF) == 1;
  }
};

}  // namespace cyclescope
