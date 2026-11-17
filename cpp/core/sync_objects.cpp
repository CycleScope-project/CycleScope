// Synchronization state machines (docs/en/design/engine.md).

#include "sync_objects.hpp"

#include <cstdio>
#include <cstdlib>

#include <algorithm>

#include "mechanism_runtime.hpp"  // EngineError

namespace cyclescope {

// ---- mbarrier --------------------------------------------------------

const MbarrierTable::Object* MbarrierTable::find(
    std::uint32_t segment, std::uint32_t object) const {
  auto it = objects_.find(key(segment, object));
  return it == objects_.end() ? nullptr : &it->second;
}

MbarrierTable::Object* MbarrierTable::find(std::uint32_t segment,
                                           std::uint32_t object) {
  auto it = objects_.find(key(segment, object));
  return it == objects_.end() ? nullptr : &it->second;
}

namespace {
// CYCLESCOPE_DEBUG_MBARRIER=<segment>:<object> traces every state change of
// one object to stderr (diagnostic only)
bool dbg_match(std::uint32_t segment, std::uint32_t object) {
  static int seg = -2, obj = -2;
  if (seg == -2) {
    const char* v = std::getenv("CYCLESCOPE_DEBUG_MBARRIER");
    seg = obj = -1;
    if (v && std::sscanf(v, "%d:%d", &seg, &obj) != 2) seg = obj = -1;
  }
  return seg >= 0 && (std::uint32_t)seg == segment &&
         (std::uint32_t)obj == object;
}
// CYCLESCOPE_DEBUG_CTABAR=<cta>:<barrier> traces every state change of
// one CTA named barrier to stderr (diagnostic only)
bool ctabar_dbg_match(std::uint32_t cta, std::uint32_t barrier) {
  static int c = -2, b = -2;
  if (c == -2) {
    const char* v = std::getenv("CYCLESCOPE_DEBUG_CTABAR");
    c = b = -1;
    if (v && std::sscanf(v, "%d:%d", &c, &b) != 2) c = b = -1;
  }
  return c >= 0 && (std::uint32_t)c == cta && (std::uint32_t)b == barrier;
}
#define CTABAR_DBG(cta, barrier, cycle, ...)                             \
  do {                                                                   \
    if (ctabar_dbg_match(cta, barrier))                                  \
      std::fprintf(stderr, "CTABAR cta=%u id=%u cyc=%llu ", cta,         \
                   barrier, (unsigned long long)(cycle));                \
    if (ctabar_dbg_match(cta, barrier))                                  \
      std::fprintf(stderr, __VA_ARGS__);                                 \
  } while (0)
}  // namespace

void MbarrierTable::debug(const char* op, std::uint32_t segment,
                          std::uint32_t object, std::int64_t delta,
                          std::uint64_t cycle) const {
  if (!dbg_match(segment, object)) return;
  const auto* o = find(segment, object);
  std::fprintf(stderr, "mbarrier %u:%u cycle %llu %s %lld -> pending %lld tx %lld phase %llu\n",
               segment, object, (unsigned long long)cycle, op,
               (long long)delta, o ? (long long)o->pending_arrivals : -1,
               o ? (long long)o->tx_count : -1,
               o ? (unsigned long long)o->phase : 0ull);
}

void MbarrierTable::init(std::uint32_t segment, std::uint32_t object,
                         std::uint64_t expected, bool cluster_scope,
                         std::uint32_t cta, std::uint64_t cycle,
                         std::uint32_t initial_phase) {
  if (expected == 0)
    throw EngineError("missing_fact:mbarrier_init_count (object " +
                      std::to_string(object) + ")");
  auto& o = objects_[key(segment, object)];
  // re-init of an idle barrier (no arrival of the current phase, no
  // pending tx, nobody waiting) overwrites its state, as the hardware
  // does: Triton warp-specialised kernels init the same word once per
  // partition prologue.  A re-init while a phase is in progress is still
  // the converter-numbering error it always signalled.
  if (o.initialized) {
    const bool idle =
        o.pending_arrivals == static_cast<std::int64_t>(o.expected_arrivals)
        && o.tx_count == 0 && o.waiters.empty() && o.phase_warps.empty();
    if (!idle)
      throw EngineError("unsupported:engine:mbarrier_reinit (object " +
                        std::to_string(object) + ")");
  }
  o = Object{};
  o.initialized = true;
  o.cluster_scope = cluster_scope;
  o.owner_cta = cta;
  o.expected_arrivals = expected;
  o.pending_arrivals = static_cast<std::int64_t>(expected);
  o.phase = initial_phase;  // converter-decoded init parity (§13.3)
  debug("init", segment, object, (std::int64_t)expected, cycle);
}

bool MbarrierTable::check_phase(Object& o, std::uint32_t object,
                                std::uint64_t cycle) {
  if (o.pending_arrivals < 0 || (o.pending_arrivals == 0 && o.tx_count < 0))
    throw EngineError("unsupported:engine:mbarrier_overrun (object " +
                      std::to_string(object) + ")");
  if (o.pending_arrivals != 0 || o.tx_count != 0) return false;
  o.phase += 1;
  o.pending_arrivals = static_cast<std::int64_t>(o.expected_arrivals);
  o.phase_complete_cycle = cycle;
  o.release_participant_warps = o.phase_warps.size();
  o.release_expected_tx_bytes = o.phase_expect_bytes;
  o.release_completed_tx_bytes = o.phase_complete_bytes;
  o.phase_warps.clear();
  o.phase_expect_bytes = 0;
  o.phase_complete_bytes = 0;
  return true;
}

bool MbarrierTable::arrive(std::uint32_t segment, std::uint32_t object,
                           std::uint64_t credit, std::uint64_t tx_bytes,
                           std::uint32_t warp, std::uint64_t cycle) {
  auto* o = find(segment, object);
  if (!o || !o->initialized)
    throw EngineError(
        "unsupported:engine:mbarrier_uninitialized (object " +
        std::to_string(object) + ")");
  o->pending_arrivals -= static_cast<std::int64_t>(credit);
  o->tx_count += static_cast<std::int64_t>(tx_bytes);
  o->phase_warps.insert(warp);
  o->waiters.erase(warp);
  debug("arrive", segment, object, (std::int64_t)tx_bytes, cycle);
  return check_phase(*o, object, cycle);
}

bool MbarrierTable::expect_tx(std::uint32_t segment, std::uint32_t object,
                              std::uint64_t bytes, std::uint64_t cycle) {
  auto* o = find(segment, object);
  if (!o || !o->initialized)
    throw EngineError(
        "unsupported:engine:mbarrier_uninitialized (object " +
        std::to_string(object) + ")");
  o->tx_count += static_cast<std::int64_t>(bytes);
  o->phase_expect_bytes += bytes;
  debug("expect_tx", segment, object, (std::int64_t)bytes, cycle);
  return check_phase(*o, object, cycle);
}

bool MbarrierTable::complete_tx(std::uint32_t segment, std::uint32_t object,
                                std::uint64_t bytes, std::uint64_t cycle) {
  auto* o = find(segment, object);
  if (!o || !o->initialized)
    throw EngineError(
        "unsupported:engine:mbarrier_uninitialized (object " +
        std::to_string(object) + ")");
  o->tx_count -= static_cast<std::int64_t>(bytes);
  o->phase_complete_bytes += bytes;
  debug("complete_tx", segment, object, -(std::int64_t)bytes, cycle);
  return check_phase(*o, object, cycle);
}

bool MbarrierTable::wait_ready(std::uint32_t segment, std::uint32_t object,
                               std::uint32_t parity,
                               std::uint64_t cycle) const {
  const auto* o = find(segment, object);
  if (!o || !o->initialized) return false;
  // phase p is complete once phase count has odd/even parity flipped
  return (o->phase % 2) != (parity & 1) && cycle >= o->release_cycle;
}

std::pair<std::int64_t, std::int64_t> MbarrierTable::inval(
    std::uint32_t segment, std::uint32_t object) {
  auto it = objects_.find(key(segment, object));
  if (it == objects_.end()) return {0, 0};
  auto inflight = std::make_pair(it->second.pending_arrivals,
                                 it->second.tx_count);
  it->second = Object{};
  return inflight;
}

void MbarrierTable::add_waiter(std::uint32_t segment,
                               std::uint32_t object,
                               std::uint32_t warp) {
  objects_[key(segment, object)].waiters.insert(warp);
}

void MbarrierTable::remove_waiter(std::uint32_t segment,
                                  std::uint32_t object,
                                  std::uint32_t warp) {
  auto it = objects_.find(key(segment, object));
  if (it != objects_.end()) it->second.waiters.erase(warp);
}

void MbarrierTable::add_sleeper(std::uint32_t segment,
                                std::uint32_t object, std::uint32_t warp,
                                std::uint64_t blocked_until,
                                std::uint64_t wake_latency) {
  objects_[key(segment, object)].sleepers.push_back({warp, blocked_until, wake_latency});
}

// ---- CTA named barrier -------------------------------------------------

CtaBarrierTable::CtaBarrierTable(std::vector<std::uint32_t> warps_per_cta)
    : warps_per_cta_(std::move(warps_per_cta)) {
  exited_.resize(warps_per_cta_.size());
}

CtaBarrierTable::Barrier& CtaBarrierTable::get_or_create(
    std::uint32_t cta, std::uint32_t barrier) {
  const std::uint64_t key = (std::uint64_t(cta) << 32) | barrier;
  auto [it, added] = barriers_.try_emplace(key);
  if (added) {
    if (cta >= cta_keys_.size()) cta_keys_.resize(cta + 1);
    cta_keys_[cta].push_back(key);
  }
  return it->second;
}

std::uint64_t CtaBarrierTable::expected_for(const Barrier& b,
                                            std::uint32_t cta) const {
  if (b.expected_field) return b.expected_field;
  const std::uint64_t live =
      warps_per_cta_[cta] - exited_[cta].size();
  return live * 32;
}

CtaBarrierTable::ArriveResult CtaBarrierTable::arrive(
    std::uint32_t cta, std::uint32_t barrier, std::uint32_t warp,
    std::uint64_t expected_field, std::uint64_t instance,
    std::uint64_t cycle) {
  auto& b = get_or_create(cta, barrier);
  b.expected_field = expected_field;
  if (b.open.expected == 0) b.open.expected = expected_for(b, cta);
  b.open.arrived += 32;
  b.open.warps.insert(warp);
  b.open.last_arrive_instance = instance;
  ArriveResult r;
  r.generation = b.open_gen;
  CTABAR_DBG(cta, barrier, cycle,
             "arrive warp=%u ef=%llu exp=%llu arrived=%llu gen=%llu\n",
             warp, (unsigned long long)expected_field,
             (unsigned long long)b.open.expected,
             (unsigned long long)b.open.arrived,
             (unsigned long long)b.open_gen);
  if (b.open.arrived > b.open.expected)
    throw EngineError("unsupported:engine:cta_barrier_overrun (cta " +
                      std::to_string(cta) + " barrier " +
                      std::to_string(barrier) + ")");
  if (b.open.arrived == b.open.expected) {
    r.full = true;
    r.participant_warps = b.open.warps.size();
    r.last_arrive_instance = b.open.last_arrive_instance;
    // close: the caller resolves the release leaf and writes it back
    CTABAR_DBG(cta, barrier, cycle,
               "full gen=%llu warps=%zu arrived=%llu exp=%llu\n",
               (unsigned long long)b.open_gen, b.open.warps.size(),
               (unsigned long long)b.open.arrived,
               (unsigned long long)b.open.expected);
    b.closed_release[b.open_gen] = kNone64;
    ++b.open_gen;
    b.open = Gen{};
    (void)cycle;
  }
  return r;
}

bool CtaBarrierTable::wait_ready(std::uint32_t cta, std::uint32_t barrier,
                                 std::uint64_t generation,
                                 std::uint64_t cycle) const {
  auto it = barriers_.find((std::uint64_t(cta) << 32) | barrier);
  if (it == barriers_.end()) return false;
  const auto& b = it->second;
  if (generation >= b.open_gen) return false;
  auto r = b.closed_release.find(generation);
  const bool ok = r != b.closed_release.end() &&
                  r->second != kNone64 && cycle >= r->second;
  if (ok)
    CTABAR_DBG(cta, barrier, cycle,
               "wait_ready gen=%llu release=%llu open_gen=%llu\n",
               (unsigned long long)generation,
               (unsigned long long)r->second,
               (unsigned long long)b.open_gen);
  return ok;
}

void CtaBarrierTable::set_release(std::uint32_t cta, std::uint32_t barrier,
                                  std::uint64_t generation,
                                  std::uint64_t cycle) {
  get_or_create(cta, barrier).closed_release[generation] = cycle;
  CTABAR_DBG(cta, barrier, cycle, "set_release gen=%llu\n",
             (unsigned long long)generation);
}

std::vector<std::pair<std::uint32_t, CtaBarrierTable::ArriveResult>>
CtaBarrierTable::warp_exited(std::uint32_t cta, std::uint32_t warp,
                             std::uint64_t cycle) {
  std::vector<std::pair<std::uint32_t, ArriveResult>> out;
  if (cta >= exited_.size()) return out;
  if (!exited_[cta].insert(warp).second) return out;
  if (cta >= cta_keys_.size()) return out;
  for (const std::uint64_t key : cta_keys_[cta]) {
    auto& b = barriers_.find(key)->second;
    if (b.expected_field != 0) continue;
    b.open.expected = expected_for(b, cta);
    if (b.open.arrived > 0 && b.open.arrived >= b.open.expected) {
      ArriveResult r;
      r.full = true;
      r.generation = b.open_gen;
      r.participant_warps = b.open.warps.size();
      r.last_arrive_instance = b.open.last_arrive_instance;
      CTABAR_DBG(cta, static_cast<std::uint32_t>(key & 0xFFFFFFFF),
                 cycle,
                 "exit_close warp=%u gen=%llu arrived=%llu exp=%llu\n",
                 warp, (unsigned long long)b.open_gen,
                 (unsigned long long)b.open.arrived,
                 (unsigned long long)b.open.expected);
      b.closed_release[b.open_gen] = kNone64;
      ++b.open_gen;
      b.open = Gen{};
      out.emplace_back(static_cast<std::uint32_t>(key & 0xFFFFFFFF), r);
    }
    (void)cycle;
  }
  return out;
}

// ---- cluster barrier ---------------------------------------------------

ClusterBarrierTable::ClusterBarrierTable(
    std::vector<std::uint32_t> warps_per_cluster) {
  for (std::uint32_t i = 0; i < warps_per_cluster.size(); ++i)
    clusters_[i].total_warps = warps_per_cluster[i];
}

std::uint64_t ClusterBarrierTable::expected_for(const Cluster& c) const {
  return c.total_warps - c.exited.size();
}

bool ClusterBarrierTable::maybe_close(Cluster& c, std::uint64_t gen,
                                      ArriveResult& out) const {
  auto it = c.gens.find(gen);
  if (it == c.gens.end() || it->second.closed) return false;
  if (it->second.warps.size() < expected_for(c)) return false;
  it->second.closed = true;
  it->second.release_cycle = kNone64;
  out.full = true;
  out.generation = gen;
  out.participant_warps = it->second.warps.size();
  out.participant_ctas = it->second.ctas.size();
  out.last_arrive_instance = it->second.last_arrive_instance;
  return true;
}

ClusterBarrierTable::ArriveResult ClusterBarrierTable::arrive(
    std::uint32_t cluster, std::uint64_t generation, std::uint32_t cta,
    std::uint32_t warp, std::uint64_t instance, std::uint64_t cycle) {
  auto& g = clusters_[cluster].gens[generation];
  g.warps.insert(warp);
  g.ctas.insert(cta);
  g.last_arrive_instance = instance;
  ArriveResult r;
  r.generation = generation;
  maybe_close(clusters_[cluster], generation, r);
  (void)cycle;
  return r;
}

bool ClusterBarrierTable::wait_ready(std::uint32_t cluster,
                                     std::uint64_t generation,
                                     std::uint64_t cycle) const {
  auto ci = clusters_.find(cluster);
  if (ci == clusters_.end()) return false;
  auto it = ci->second.gens.find(generation);
  if (it == ci->second.gens.end() || !it->second.closed) return false;
  return it->second.release_cycle != kNone64 &&
         cycle >= it->second.release_cycle;
}

void ClusterBarrierTable::set_release(std::uint32_t cluster,
                                      std::uint64_t generation,
                                      std::uint64_t cycle) {
  clusters_[cluster].gens[generation].release_cycle = cycle;
}

std::vector<ClusterBarrierTable::ArriveResult>
ClusterBarrierTable::warp_exited(std::uint32_t cluster, std::uint32_t warp,
                                 std::uint64_t cycle) {
  std::vector<ArriveResult> out;
  auto& c = clusters_[cluster];
  if (!c.exited.insert(warp).second) return out;
  for (auto& [gen, g] : c.gens) {
    if (g.closed || g.warps.empty()) continue;
    ArriveResult r;
    if (maybe_close(c, gen, r)) out.push_back(r);
  }
  (void)cycle;
  return out;
}

}  // namespace cyclescope
