#include "dispatch.hpp"

namespace cyclescope {

void Dispatcher::configure(std::uint32_t sm_count,
                           std::uint32_t occupancy,
                           const std::vector<SmTopologyRow>& topo) {
  nsm_ = sm_count;
  occupancy_ = occupancy;
  sm_.assign(nsm_, 0);
  ctas_.assign(nsm_, 0);
  topo_.assign(nsm_, nullptr);
  order_.clear();
  for (const auto& t : topo) {
    if (t.sm < nsm_) {
      topo_[t.sm] = &t;
      order_.push_back(t.sm);
    }
  }
  if (order_.empty())
    for (std::uint32_t i = 0; i < nsm_; ++i) order_.push_back(i);
}

int Dispatcher::slot_base(std::uint32_t sm, std::uint32_t n) const {
  if (sm >= nsm_ || !n || n > 64) return -1;
  const std::uint64_t b = sm_[sm];
  for (std::uint32_t i = 0; i + n <= 64; ++i) {
    const std::uint64_t mask = n == 64 ? ~0ull : ((1ull << n) - 1) << i;
    if ((b & mask) == 0) return static_cast<int>(i);
  }
  return -1;
}

bool Dispatcher::has_capacity(std::uint32_t sm) const {
  return sm < nsm_ && ctas_[sm] < occupancy_;
}

std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>>
Dispatcher::place_cluster(std::uint32_t size,
                          std::uint32_t warp_count_per_cta) const {
  // candidate SMs in iteration order with both CTA capacity and a
  // free contiguous warp-slot run
  std::vector<std::pair<std::uint32_t, std::uint32_t>> cand;
  for (const std::uint32_t sm : order_) {
    if (!has_capacity(sm)) continue;
    const int base = slot_base(sm, warp_count_per_cta);
    if (base >= 0) cand.emplace_back(sm, static_cast<std::uint32_t>(base));
  }
  if (cand.size() < size) return std::nullopt;
  if (size == 1) return std::vector{cand.front()};
  const auto domain_of = [&](std::uint32_t sm) {
    return topo_[sm] ? topo_[sm]->placement_domain : sm;
  };
  if (size == 2) {
    // prefer a pair within one tpc_pair, else any same-domain pair
    for (std::size_t i = 0; i < cand.size(); ++i)
      for (std::size_t j = i + 1; j < cand.size(); ++j)
        if (topo_[cand[i].first] && topo_[cand[j].first] &&
            topo_[cand[i].first]->tpc_pair ==
                topo_[cand[j].first]->tpc_pair)
          return std::vector{cand[i], cand[j]};
    for (std::size_t i = 0; i < cand.size(); ++i)
      for (std::size_t j = i + 1; j < cand.size(); ++j)
        if (domain_of(cand[i].first) == domain_of(cand[j].first))
          return std::vector{cand[i], cand[j]};
    return std::nullopt;
  }
  // larger clusters: first `size` candidates in one domain
  for (std::size_t i = 0; i < cand.size(); ++i) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> out;
    for (std::size_t j = i; j < cand.size(); ++j)
      if (domain_of(cand[j].first) == domain_of(cand[i].first)) {
        out.push_back(cand[j]);
        if (out.size() == size) return out;
      }
  }
  return std::nullopt;
}

void Dispatcher::occupy(std::uint32_t sm, std::uint32_t base,
                        std::uint32_t n) {
  const std::uint64_t mask = n == 64 ? ~0ull : ((1ull << n) - 1) << base;
  sm_[sm] |= mask;
  ++ctas_[sm];
}

void Dispatcher::release(std::uint32_t sm, std::uint32_t base,
                         std::uint32_t n) {
  const std::uint64_t mask = n == 64 ? ~0ull : ((1ull << n) - 1) << base;
  sm_[sm] &= ~mask;
  if (ctas_[sm]) --ctas_[sm];
}

}  // namespace cyclescope
