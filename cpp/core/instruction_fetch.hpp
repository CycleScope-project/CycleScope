#pragma once
// §13.13 instruction fetch: per-SM L0/L1 code-cache model with
// chained cold streaming, launch prefetch, and sequential prefetch.
// Demand lines and sequential-prefetch lines claimed by a demand land
// in the L0; sequential prefetch arrivals park in a per-SM staging set
// (L1-resident) and enter the L0 only when demanded. Launch-prefetch
// lines stay L1 + l2code and are served at the L1 cadence while
// inflight.

#include <cstdint>
#include <list>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cyclescope {

struct FetchConfig {
  bool enabled = false;
  std::uint64_t line_bytes = 0;
  std::uint64_t prefetch_depth = 4;    // sequential prefetch lines
  std::uint64_t capacity_l0 = 0;       // bytes; 0 -> unbounded
  std::uint64_t capacity_l1 = 0;
  std::uint64_t stream_l1 = 0, stream_l2 = 0, stream_cold = 0;
  std::uint64_t launch_prefetch = 0;   // bytes streamed at stream_l1
  std::uint64_t sharing = 0;           // 0: L0 per SM, 1: L0 per SMSP
  std::uint64_t parallelism = 1;       // arrival channels per SM
  // 0 -> one LRU of capacity_l0 lines per owner; >0 -> that many ways,
  // sets = capacity lines / ways, set = line % sets
  std::uint64_t l0_ways = 0;
};

class InstructionFetch {
 public:
  void configure(FetchConfig c);
  bool enabled() const { return cfg_.enabled; }
  std::uint64_t parallelism() const { return cfg_.parallelism; }
  // grow the SM table when the dispatcher lands a warp on a new SM
  void ensure_sm(std::uint32_t sm);

  // launch prefetch chain starting at the warp's first code line;
  // arrivals land in L1 + l2code (prefetch_only)
  void launch_prefetch(std::uint32_t sm, std::uint32_t smsp,
                       std::uint32_t channel,
                       std::uint64_t code_addr, std::uint64_t now);

  // cycle at which `code_addr` is resident in the demand L0; the L0
  // set is selected by `smsp` when sharing=1, else per SM
  std::uint64_t ready(std::uint32_t sm, std::uint32_t smsp,
                      std::uint32_t channel,
                      std::uint64_t code_addr, std::uint64_t now,
                      std::uint32_t warp);

  std::uint64_t l0_hits = 0, l1_lines = 0, l2_lines = 0, cold_lines = 0;
  std::uint64_t staged_hits = 0;  // diagnostics only, not in TELE

 private:
  struct Lru {
    std::list<std::uint64_t> order;
    std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator>
        pos;
    std::uint64_t cap = 0;  // lines; 0 -> unbounded
    bool has(std::uint64_t l) const { return pos.count(l); }
    void touch(std::uint64_t l);
    void insert(std::uint64_t l);
  };
  struct Inflight {
    std::uint64_t arrival;
    bool prefetch_only;
    bool sequential;  // sequential prefetch, not yet demanded
    std::uint32_t smsp;  // L0 owner on demand settle
  };
  struct Sm {
    // outer: 1 per SM or 1 per SMSP; inner: sets (1 when l0_ways=0)
    std::vector<std::vector<Lru>> l0;
    Lru l1;
    std::unordered_set<std::uint64_t> l2code;
    std::unordered_map<std::uint64_t, Inflight> inflight;
    // arrived sequential-prefetch lines not yet demanded (L1-resident,
    // enter L0 on demand)
    std::unordered_set<std::uint64_t> staged;
    // last_arrival[level][channel]: level 0=l1, 1=l2, 2=cold
    std::vector<std::vector<std::uint64_t>> last_arrival;
  };

  // per-warp demand stream state: a non-sequential demand opens a
  // new stream; sequential rows arrive at prev + stream(level)
  struct WarpFetch {
    std::int64_t front = -1;
  };
  std::vector<WarpFetch> warps_;   // indexed by warp (grown on demand)

  FetchConfig cfg_;
  std::vector<Sm> sms_;
  std::uint64_t streams_[3] = {0, 0, 0};  // l1, l2, cold

  Lru& l0(Sm& s, std::uint32_t smsp, std::uint64_t line) {
    auto& owner = s.l0[cfg_.sharing ? smsp % s.l0.size() : 0];
    return owner[line % owner.size()];
  }
  void settle(Sm& s, std::uint64_t now);
  std::uint64_t stream(int level) const { return streams_[level]; }
};

}  // namespace cyclescope
