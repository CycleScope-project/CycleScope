// Semantics decode — port of convert/semantics_common.py.
#include "convert/semantics.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <sstream>

namespace cyclescope_convert {

namespace {

std::string upper(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::toupper((unsigned char)c));
  return s;
}

std::string where_of(const StaticInfo &st) {
  char buf[64];
  // python: f"{opcode} @{pc:#x}" — we don't carry pc in StaticInfo; the
  // caller passes it via operands? keep "opcode @pc" form using a member
  return st.opcode;
}

// ---- operand helpers -------------------------------------------------

const std::uint64_t *uniform(const DynOps &d,
                             std::initializer_list<const char *> names) {
  for (auto *n : names) {
    auto it = d.uniform.find(n);
    if (it != d.uniform.end()) return &it->second;
  }
  return nullptr;
}

const std::uint64_t *last_op(const DynOps &d) {
  for (int i = 7; i >= 0; --i) {
    auto it = d.uniform.find("op" + std::to_string(i));
    if (it != d.uniform.end()) return &it->second;
  }
  return nullptr;
}

std::uint64_t first_op(const DynOps &d,
                       std::initializer_list<const char *> names,
                       std::uint64_t dflt) {
  if (auto *v = uniform(d, names)) return *v;
  if (auto *v = last_op(d)) return *v;
  return dflt;
}

std::uint64_t lane_address(const DynOps &d) {
  if (d.memory.empty() || d.memory[0].addrs.empty()) return 0;
  return d.memory[0].addrs[0];
}

std::uint64_t dst_address(const DynOps &d, int idx = 0) {
  if ((int)d.memory.size() <= idx || d.memory[idx].addrs.empty())
    return 0;
  return d.memory[idx].addrs[0];
}

std::uint64_t object_address(const DynOps &d, int idx = 0) {
  if (auto *v = uniform(d, {"mbarrier_address", "mbarrier", "object",
                            "mbar"}))
    return *v;
  return dst_address(d, idx);
}

std::uint64_t ranks_mask(const DynOps &d, int idx = 0) {
  auto it = d.ranks.find(idx);
  if (it == d.ranks.end()) return 0;
  std::uint64_t m = 0;
  for (auto r : it->second) m |= 1ull << r;
  return m;
}

std::uint64_t multicast_mask(const DynOps &d, int idx = 0) {
  if (auto *v = uniform(d, {"multicast_mask"})) return *v;
  return ranks_mask(d, idx);
}

}  // namespace

std::uint32_t SemanticsContext::mbarrier(int cta, std::uint64_t addr,
                                         const std::string &where) {
  auto key = std::make_pair(cta, addr);
  auto it = mbarrier_objects.find(key);
  if (it != mbarrier_objects.end()) return it->second;
  auto idx = static_cast<std::uint32_t>(mbarrier_objects.size());
  mbarrier_objects[key] = idx;
  return idx;
}
int SemanticsContext::next_group(std::map<int, int> &t, int w) {
  int g = t.count(w) ? t[w] : 0;
  t[w] = g + 1;
  return g;
}
int SemanticsContext::current_group(std::map<int, int> &t, int w) {
  return t.count(w) ? t[w] : 0;
}

namespace {

SemRow row(SemKind k, int fl = 0) {
  SemRow r;
  r.kind = k;
  r.flags = fl;
  return r;
}

}  // namespace

std::uint32_t resolve_mbarrier(const DynOps &d, SemanticsContext &c,
                               int cta, std::uint64_t addr,
                               const char *where) {
  // object keys are local shared offsets; strip the DSMEM rank tag
  // (rank << 24) in every case, not just remote (bug 4-5.1)
  if (c.local_shared) addr = c.local_shared(addr);
  auto it = d.uniform.find("owner_rank");
  if (it == d.uniform.end()) return c.mbarrier(cta, addr, where);
  const int rank = (int)it->second;
  auto iit = c.cta_rank.find(cta);
  const int issuing = iit == c.cta_rank.end() ? 0 : iit->second;
  if (rank == issuing) return c.mbarrier(cta, addr, where);
  for (auto &[lin, rk] : c.cta_rank) {
    if (rk == rank) {
      if (c.local_shared) addr = c.local_shared(addr);
      return c.mbarrier(lin, addr, where);
    }
  }
  throw ConversionError(std::string("missing_fact:owner_rank_cta (") +
                        where + ")");
}

namespace {

// SYNCS.EXCH / STS.64 init state word -> PTX mbarrier.init count, or -1
// (sass_decode.decode_mbarrier_init_expected_arrivals).
std::int64_t decode_mbarrier_init_count(std::uint64_t raw) {
  if (raw >= (1ULL << 32)) {
    const std::uint64_t low = raw & 0xFFFFFFFFu, high = raw >> 32;
    if (low & 1 || !(0 < low && low < (1ULL << 21))) return -1;
    const std::int64_t count = (1 << 20) - (std::int64_t)(low >> 1);
    // bit 31 of the high word is the initial phase; the rest of it is
    // the pending-count mirror
    if (0 < count && count < (1 << 20) &&
        (high & 0x7FFFFFFFu) ==
            ((0x80000000u - ((std::uint64_t)count << 11)) & 0x7FFFFFFFu))
      return count;
    return -1;
  }
  // reject decoded counts that cannot be an arrival count
  const std::int64_t bound = 1 << 16;
  std::int64_t hopper = -1, sm103 = -1;
  if (!(raw & 1) && 0 < raw && raw < (1ULL << 21)) {
    const std::int64_t expected = (1 << 20) - (std::int64_t)(raw >> 1);
    if (0 < expected && expected <= bound) hopper = expected;
  }
  if (0 < raw && raw < (1ULL << 20)) {
    const std::int64_t expected = (1 << 20) - (std::int64_t)raw;
    if (0 < expected && expected <= bound) sm103 = expected;
  }
  if (hopper >= 0 && sm103 < 0) return hopper;
  if (sm103 >= 0 && hopper < 0) return sm103;
  if (hopper < 0) return -1;
  if (hopper <= 1024) return hopper;
  if (sm103 >= 0 && sm103 <= 1024) return sm103;
  return hopper;
}

// per-lane arrays in the facts are compacted to the executing lanes
// (exec_mask.bit_count() entries, lane order); a 32-entry form is
// full-width and indexed by lane id.  Never silently drops entries.
std::vector<std::uint64_t> executing_entries(
    const std::vector<std::uint64_t> &values, std::uint64_t mask) {
  if ((int)values.size() == __builtin_popcountll(mask)) return values;
  if (values.size() == 32) {
    std::vector<std::uint64_t> out;
    out.reserve(values.size());
    for (int i = 0; i < 32; ++i)
      if ((mask >> i) & 1) out.push_back(values[i]);
    return out;
  }
  return values;
}

SemRow d_mbarrier_init(const DynOps &d, SemanticsContext &c, int cta,
                       const StaticInfo &st) {
  const std::string where = where_str(st);
  // distinct barrier objects over the active lanes' mref addresses (one
  // predicated store initialising several stage barriers); a captured
  // barrier-word fact names the object when present
  std::vector<std::uint32_t> objs;
  if (!d.memory.empty() && !d.uniform.count("mbarrier_address")) {
    const auto &ref = d.memory[0];
    // lane_mask is the record's own executing mask; absent -> caller's
    const std::uint64_t mask =
        ref.lane_mask ? ref.lane_mask : (d.active_mask & d.predicate_mask);
    std::set<std::uint32_t> seen;
    for (auto addr : executing_entries(ref.addrs, mask)) {
      auto o = resolve_mbarrier(d, c, cta, addr, "mbarrier_init");
      if (seen.insert(o).second) objs.push_back(o);
    }
  }
  auto r = row(SemKind::mbarrier_init);
  if (objs.size() > 1) {
    r.fields["object"] = objs[0];
    r.object_list = objs;
  } else if (!objs.empty()) {
    r.fields["object"] = objs[0];
  } else {
    r.fields["object"] = resolve_mbarrier(d, c, cta, object_address(d),
                                          "mbarrier_init");
  }
  // a promoted store's per-lane init words must decode to the same
  // (expected_arrivals, initial_phase); divergent lanes are not a legal
  // multi-barrier init
  auto lv = d.lane_values.find("sts64_value");
  if (lv != d.lane_values.end() && !lv->second.empty()) {
    const std::uint64_t mask = d.active_mask & d.predicate_mask;
    std::int64_t ref_count = -1, ref_phase = -1;
    bool have = false;
    for (auto value : executing_entries(lv->second, mask)) {
      const std::int64_t count = decode_mbarrier_init_count(value);
      const std::int64_t phase = (value >> 63) & 1;
      if (count < 0 || (have && (count != ref_count ||
                                 phase != ref_phase)))
        throw ConversionError(
            "unsupported:mbarrier_init:divergent_lane_values (" + where +
            ")");
      ref_count = count; ref_phase = phase; have = true;
    }
  }
  r.fields["expected_arrivals"] =
      first_op(d, {"expected_arrivals", "count"}, 0);
  r.fields["scope"] = ranks_mask(d) ? SCOPE_CLUSTER : SCOPE_CTA;
  r.fields["initial_phase"] = first_op(d, {"initial_phase"}, 0);
  return r;
}

// per-lane barrier words (operand fact mbarrier_words) naming different
// objects: same CTA-local offset on several ranks (CLC response arrive:
// lane i -> CTA i) -> issuer object + destination_mask + SPLT credit
// (lanes per rank); different offsets -> MBOL list.  false: lanes agree.
bool per_lane_destinations(const DynOps &d, SemanticsContext &c, int cta,
                           SemRow &r) {
  auto it = d.lane_values.find("mbarrier_words");
  if (it == d.lane_values.end() || it->second.size() < 2) return false;
  const auto &words = it->second;
  std::set<std::uint64_t> distinct(words.begin(), words.end());
  if (distinct.size() < 2) return false;
  bool tagged = false;
  for (auto w : words) if (w >> 24) tagged = true;
  auto iit = c.cta_rank.find(cta);
  const int issuer = iit == c.cta_rank.end() ? 0 : iit->second;
  auto rank_of = [&](std::uint64_t w) {
    return tagged ? (int)(w >> 24) : issuer;
  };
  auto strip = [&](std::uint64_t w) {
    return c.local_shared ? c.local_shared(w) : w;
  };
  std::set<std::uint64_t> offsets;
  for (auto w : words) offsets.insert(strip(w));
  if (offsets.size() == 1) {
    std::map<int, std::uint32_t> per_rank;
    for (auto w : words) per_rank[rank_of(w)] += 1;
    std::uint64_t mask = 0;
    for (auto &[rank, n] : per_rank) {
      mask |= 1ull << rank;
      r.split.push_back({(std::uint32_t)rank, n, 0u});
    }
    r.fields["object"] = c.mbarrier(cta, *offsets.begin(), "mbarrier_arrive");
    r.fields["destination_mask"] = mask;
    return true;
  }
  std::vector<std::uint32_t> objs;
  for (auto w : words) {
    const int rank = rank_of(w);
    int target = cta;
    for (auto &[lin, rk] : c.cta_rank)
      if (rk == rank) { target = lin; break; }
    const auto o = c.mbarrier(target, strip(w), "mbarrier_arrive");
    if (std::find(objs.begin(), objs.end(), o) == objs.end())
      objs.push_back(o);
  }
  r.fields["object"] = objs[0];
  r.object_list = objs;
  r.fields["destination_mask"] = 0;
  return true;
}

SemRow d_mbarrier_arrive(const DynOps &d, SemanticsContext &c, int cta,
                         int) {
  auto r = row(SemKind::mbarrier_arrive);
  if (per_lane_destinations(d, c, cta, r)) {
    r.fields["generation"] = GEN_UNRESOLVED;
    r.fields["transaction_bytes"] =
        first_op(d, {"transaction_bytes", "expect_tx_bytes"}, 0);
    const auto count = first_op(d, {"arrival_count", "count"}, 1);
    r.fields["arrival_count"] = count;
    if (!r.split.empty()) {
      // per-lane arrival factor: the uniform count is lanes x factor
      std::uint64_t lanes = 0;
      for (auto &sp : r.split) lanes += sp[1];
      const std::uint64_t factor =
          std::max<std::uint64_t>(1, count / std::max<std::uint64_t>(1, lanes));
      for (auto &sp : r.split) sp[1] = (std::uint32_t)(sp[1] * factor);
    }
    if (first_op(d, {"transaction_complete"}, 0))
      r.flags |= SEMA_FLAG_COMPLETE_TX;
    return r;
  }
  // distinct objects over active lanes: mref fallback only for facts
  // without a captured barrier word (sm_103 NVBit SYNCS mref +0x7FFF)
  std::vector<std::uint32_t> objs;
  std::set<std::uint32_t> seen;
  if (!d.memory.empty() && !d.uniform.count("mbarrier_address")) {
    const auto &ref = d.memory[0];
    const std::uint64_t mask = ref.lane_mask ? ref.lane_mask : d.active_mask;
    for (auto addr : executing_entries(ref.addrs, mask)) {
      auto o = resolve_mbarrier(d, c, cta, addr, "mbarrier_arrive");
      if (seen.insert(o).second) objs.push_back(o);
    }
  }
  if (objs.size() > 1) {
    r.fields["object"] = objs[0];
    r.object_list = objs;
  } else if (!objs.empty()) {
    r.fields["object"] = objs[0];
  } else {
    r.fields["object"] = resolve_mbarrier(d, c, cta, object_address(d),
                                    "mbarrier_arrive");
  }
  r.fields["generation"] = GEN_UNRESOLVED;
  r.fields["transaction_bytes"] =
      first_op(d, {"transaction_bytes", "expect_tx_bytes"}, 0);
  r.fields["arrival_count"] =
      first_op(d, {"arrival_count", "count"}, 1);
  r.fields["destination_mask"] = ranks_mask(d);
  // .A<n>TX: complete the captured transaction bytes instead of
  // arriving / expecting (nvjet CLC-response arrive when CLC is unused)
  if (first_op(d, {"transaction_complete"}, 0))
    r.flags |= SEMA_FLAG_COMPLETE_TX;
  return r;
}

SemRow d_expect_tx(const DynOps &d, SemanticsContext &c, int cta, int) {
  auto r = row(SemKind::mbarrier_expect_tx);
  r.fields["object"] = resolve_mbarrier(d, c, cta, object_address(d), "expect_tx");
  r.fields["generation"] = GEN_UNRESOLVED;
  r.fields["transaction_bytes"] =
      first_op(d, {"transaction_bytes"}, 0);
  r.fields["scope"] = ranks_mask(d) ? SCOPE_CLUSTER : SCOPE_CTA;
  return r;
}

SemRow d_mbarrier_wait(const DynOps &d, SemanticsContext &c, int cta,
                       int) {
  auto *parity = uniform(d, {"parity", "phase", "UR0", "R0"});
  if (!parity) throw ConversionError("missing_fact:parity");
  auto r = row(SemKind::mbarrier_wait);
  r.fields["object"] = resolve_mbarrier(d, c, cta, object_address(d),
                                  "mbarrier_wait");
  r.fields["generation"] = GEN_UNRESOLVED;
  r.fields["parity"] = *parity & 1;
  if (d.has_after_ready) {
    r.fields["after_ready_mask"] = d.after_ready_mask;
    r.flags |= SEMA_FLAG_AFTER_READY;
  }
  if (auto *b = uniform(d, {"block_offset", "offset"}))
    r.fields["block_offset"] = *b;
  if (auto *t = uniform(d, {"timeout_hint", "suspend_time_hint"}))
    r.fields["timeout_hint"] = *t;
  return r;
}

SemRow d_mbarrier_inval(const DynOps &d, SemanticsContext &c, int cta,
                        int) {
  auto r = row(SemKind::mbarrier_inval);
  r.fields["object"] = resolve_mbarrier(d, c, cta, object_address(d),
                                  "mbarrier_inval");
  return r;
}

// BAR operand `index`: immediate as-is; a register from its captured
// first-lane value.  ptxas packs (count << 16) | id into one register
// for the register form (BAR.SYNC R0, R0): with a register id the id is
// the low 16 bits and the count the high 16; a register count next to
// an immediate id is the raw count.  shift: 0 low half, 16 high half,
// -1 raw.  Returns false when the operand is absent.
bool bar_operand(const StaticInfo &st, const DynOps &d, std::size_t index,
                 int shift, std::uint64_t &out) {
  std::vector<std::string> toks;
  for (auto &t : st.operands) {
    std::string x = t;
    while (!x.empty() && x.front() == ' ') x.erase(x.begin());
    while (!x.empty() && x.back() == ' ') x.pop_back();
    if (!x.empty() && (x[0] == 'P' || (x[0] == '!' && x.size() > 1 &&
                                       x[1] == 'P')))
      continue;   // BAR.RED source predicate
    toks.push_back(x);
  }
  if (index >= toks.size()) return false;
  const std::string &tok = toks[index];
  if (!tok.empty() && std::isdigit((unsigned char)tok[0])) {
    out = std::stoull(tok, nullptr, 0);
    return true;
  }
  if (tok.size() > 1 && tok[0] == 'R' && std::isdigit((unsigned char)tok[1])) {
    auto it = d.lane_values.find(tok);
    if (it == d.lane_values.end() || it->second.empty())
      throw ConversionError("missing_fact:cta_barrier_operand " + tok +
                            " (" + st.opcode + ")");
    const std::uint64_t v = it->second[0];
    out = shift < 0 ? v : shift == 0 ? (v & 0xFFFF) : (v >> 16);
    return true;
  }
  throw ConversionError("missing_fact:cta_barrier_operand " + tok + " (" +
                        st.opcode + ")");
}

SemRow d_cta_barrier(const StaticInfo &st, const DynOps &d,
                     SemanticsContext &c, int cta, int, bool arrives) {
  std::uint64_t barrier = 0, threads = 0;
  bool have_b = false, have_t = false;
  // immediates arrive as op0/op1 facts; register forms need the lane
  // values of the static operand tokens
  if (auto *v = uniform(d, {"barrier", "barrier_id", "op0"})) {
    barrier = *v; have_b = true;
  }
  if (auto *v = uniform(d, {"expected_threads", "count", "op1"})) {
    threads = *v; have_t = true;
  }
  if (!have_b && !bar_operand(st, d, 0, 0, barrier))
    throw ConversionError("missing_fact:cta_barrier_id (" + st.opcode + ")");
  if (!have_t) {
    const bool id_is_register =
        !st.operands.empty() && st.operands[0].size() > 1 &&
        st.operands[0][0] == 'R' &&
        std::isdigit((unsigned char)st.operands[0][1]);
    if (!bar_operand(st, d, 1, id_is_register ? 16 : -1, threads)) {
      // PTX bar.sync without a count: every non-exited CTA warp takes
      // part; 0 lets the engine deduct warps that already exited
      threads = 0;
    }
  }
  if (barrier > 15)
    throw ConversionError("unsupported:cta_barrier_id " +
                          std::to_string(barrier) + " (" + st.opcode + ")");
  auto r = row(arrives ? SemKind::cta_barrier_arrive
                       : SemKind::cta_barrier_wait);
  r.fields["barrier"] = barrier;
  r.fields["expected_threads"] = threads;
  r.fields["generation"] = GEN_UNRESOLVED;
  if (!arrives) r.flags = SEMA_FLAG_ALSO_ARRIVES;
  return r;
}

SemRow d_cluster_barrier(const DynOps &d, SemanticsContext &, int, int,
                         bool arrives) {
  auto r = row(arrives ? SemKind::cluster_barrier_arrive
                       : SemKind::cluster_barrier_wait);
  r.fields["generation"] = GEN_UNRESOLVED;
  auto m = ranks_mask(d);
  if (m) r.fields["destination_mask"] = m;
  return r;
}

SemRow d_tma_load(const DynOps &d, SemanticsContext &c, int cta, int) {
  auto r = row(SemKind::tma_load);
  r.fields["object"] = resolve_mbarrier(d, c, cta, object_address(d), "tma_load");
  r.fields["generation"] = GEN_UNRESOLVED;
  r.fields["multicast_mask"] = multicast_mask(d);
  if (d.has_tensor_map) {
    r.fields["tensor_map"] = d.tensor_map;
    auto it = c.tensor_maps.find(d.tensor_map);
    if (it != c.tensor_maps.end()) {
      std::uint64_t box = 1;
      for (auto dim : it->second.second) box *= dim;
      r.fields["shared_bytes"] = box * it->second.first;
      r.fields["global_bytes"] = box * it->second.first;
    }
  }
  return r;
}

SemRow d_bulk_group_issue(const DynOps &d, SemanticsContext &c, int,
                          int w) {
  c.bulk_open_issues[w] += 1;
  auto r = row(SemKind::bulk_group_issue);
  r.fields["group"] = c.current_group(c.bulk_group, w);
  r.fields["pending_allowed"] = 0;
  r.fields["bytes"] = first_op(d, {"bytes", "size"}, 0);
  return r;
}

SemRow d_bulk_group_commit(const DynOps &, SemanticsContext &c, int,
                           int w) {
  c.saw_utmaflush.insert(w);
  c.bulk_open_issues[w] = 0;
  auto r = row(SemKind::bulk_group_commit);
  r.fields["group"] = c.next_group(c.bulk_group, w);
  r.fields["pending_allowed"] = 0;
  return r;
}

SemRow d_bulk_group_wait(const DynOps &d, SemanticsContext &c, int,
                         int w) {
  auto r = row(SemKind::bulk_group_wait);
  r.fields["group"] = c.current_group(c.bulk_group, w);
  r.fields["pending_allowed"] =
      first_op(d, {"pending_allowed", "N"}, 0);
  return r;
}

SemRow d_async_copy_issue(const DynOps &, SemanticsContext &c, int,
                          int w) {
  auto r = row(SemKind::async_copy_issue);
  r.fields["group"] = c.current_group(c.ldgsts_group, w);
  r.fields["pending_allowed"] = 0;
  return r;
}

SemRow d_async_copy_commit(const DynOps &d, SemanticsContext &c, int cta,
                           int w, const std::string &opcode) {
  if (opcode.rfind("ARRIVES", 0) == 0) {
    // no pending cp.async in the thread -> the arrive triggers at once
    // (PTX cp.async.mbarrier.arrive); predicated-off LDGSTS records
    // still mark issue, so no guard is needed
    auto addr = object_address(d);
    auto r = row(SemKind::async_copy_commit);
    r.fields["group"] = c.current_group(c.ldgsts_group, w);
    r.fields["pending_allowed"] = 0;
    r.fields["object"] = resolve_mbarrier(d, c, cta, addr, "async_copy_commit");
    // .TRANSCNT: the paired SYNCS.ARRIVE.RED.A0T1 raised the tx count by
    // one per lane, the deferred completion lowers it — no arrival;
    // without it (.noinc) each lane arrives once
    const auto lanes = __builtin_popcount(d.active_mask & d.predicate_mask);
    const bool transcnt = opcode.find(".TRANSCNT") != std::string::npos;
    r.fields["arrival_count"] = transcnt ? 0 : lanes;
    r.fields["transaction_count"] = transcnt ? lanes : 0;
    return r;
  }
  c.saw_lgdepbar.insert(w);
  auto r = row(SemKind::async_copy_commit);
  r.fields["group"] = c.next_group(c.ldgsts_group, w);
  r.fields["pending_allowed"] = 0;
  return r;
}

SemRow d_async_copy_wait(const DynOps &d, SemanticsContext &c, int,
                         int w) {
  auto r = row(SemKind::async_copy_wait);
  r.fields["group"] = c.current_group(c.ldgsts_group, w);
  r.fields["pending_allowed"] =
      first_op(d, {"pending_allowed", "N"}, 0);
  return r;
}

SemRow d_depbar(const StaticInfo &st, const DynOps &d,
                SemanticsContext &c, int, int w, bool *emit) {
  // depbar_operands: keep/sb0 from operand tokens
  auto toks = st.operands;
  std::int64_t keep = -1;
  bool sb0 = false;
  const auto op = upper(st.opcode);
  if (op.rfind("DEPBAR.LE", 0) == 0) {
    for (auto &t : toks) {
      // SB|GSB\d+
      auto tu = upper(t);
      if ((tu.rfind("SB", 0) == 0 || tu.rfind("GSB", 0) == 0)) {
        const char *p = tu.c_str() + (tu[0] == 'G' ? 3 : 2);
        bool digits = *p;
        for (const char *q = p; *q; ++q) digits &= isdigit(*q) != 0;
        if (digits) {
          sb0 = std::atoi(p) == 0;
          continue;
        }
      }
      char *end = nullptr;
      auto v = std::strtoll(t.c_str(), &end, 0);
      if (end && *end == 0) keep = v;
    }
    if (keep < 0) {
      auto *v = uniform(d, {"pending_allowed", "N"});
      keep = v ? (std::int64_t)*v : -1;
    }
    if (keep < 0) throw ConversionError("missing_fact:depbar_keep");
  } else {
    for (auto &t : toks) {
      if (t.size() > 2 && t.front() == '{' && t.back() == '}' &&
          t.find('0') != std::string::npos) {
        keep = 0;
        sb0 = true;
      }
    }
  }
  *emit = true;
  if (!sb0 || keep < 0) {
    *emit = false;
    return {};
  }
  bool saw_cp = c.saw_lgdepbar.count(w), saw_bl = c.saw_utmaflush.count(w);
  bool group_wait =
      (0 <= keep && keep <= 7) || (keep <= 63 && (saw_cp || saw_bl));
  if (!group_wait) {
    *emit = false;
    return {};
  }
  if (saw_cp && saw_bl)
    throw ConversionError("unsupported:mixed_async_groups");
  if (saw_cp) {
    auto r = row(SemKind::async_copy_wait);
    r.fields["group"] = c.current_group(c.ldgsts_group, w);
    r.fields["pending_allowed"] = keep;
    return r;
  }
  auto r = row(SemKind::bulk_group_wait);
  r.fields["group"] = c.current_group(c.bulk_group, w);
  if (c.bulk_open_issues.count(w) && c.bulk_open_issues[w] > 0) {
    r.fields["group"] = c.next_group(c.bulk_group, w);
    c.bulk_open_issues[w] = 0;
    r.flags = SEMA_FLAG_IMPLICIT_COMMIT;
  }
  r.fields["pending_allowed"] = keep;
  return r;
}

SemRow d_bulk_copy(const DynOps &d, SemanticsContext &c, int cta, int,
                   SemKind k) {
  auto r = row(k);
  r.fields["object"] = resolve_mbarrier(d, c, cta, object_address(d), "bulk_copy");
  r.fields["generation"] = GEN_UNRESOLVED;
  r.fields["bytes"] = first_op(d, {"bytes", "size"}, 0);
  auto m = multicast_mask(d);
  if (m) r.fields["multicast_mask"] = m;
  return r;
}

SemRow d_shared_store_async(const DynOps &d, SemanticsContext &c, int cta,
                            int) {
  auto r = row(SemKind::shared_store_async);
  r.fields["bytes"] = first_op(d, {"bytes", "size"}, 0);
  auto addr = object_address(d);
  if (addr) r.fields["object"] = resolve_mbarrier(d, c, cta, addr, "ssa");
  auto it = d.ranks.find(0);
  if (it != d.ranks.end() && !it->second.empty())
    r.fields["destination_rank"] = it->second[0];
  auto lw = d.lane_values.find("mbarrier_words");
  if (lw != d.lane_values.end() && lw->second.size() > 1) {
    std::set<std::uint64_t> distinct(lw->second.begin(), lw->second.end());
    if (distinct.size() > 1) {
      // lanes store into different ranks' barriers: each lane completes
      // its bytes on its own rank's object (duplicates kept: two lanes
      // on the same CTA complete twice)
      std::vector<std::uint32_t> objs;
      for (auto w : lw->second) {
        const int rank = (int)(w >> 24);
        int target = cta;
        for (auto &[lin, rk] : c.cta_rank)
          if (rk == rank) { target = lin; break; }
        objs.push_back(c.mbarrier(
            target, c.local_shared ? c.local_shared(w) : w, "ssa"));
      }
      const std::uint64_t total = r.fields["bytes"];
      if (total % objs.size())
        throw ConversionError("missing_fact:stas_lane_bytes (ssa)");
      r.fields["bytes"] = total / objs.size();
      r.fields["object"] = objs[0];
      r.object_list = objs;
      return r;
    }
  }
  return r;
}

SemRow d_desc_vis(const DynOps &d, SemanticsContext &, int, int, int op) {
  auto r = row(SemKind::descriptor_visibility);
  r.fields["operation"] = op;
  if (auto *a = uniform(d, {"descriptor_address", "address"}))
    r.fields["descriptor_address"] = *a;
  return r;
}

SemRow d_nanosleep(const StaticInfo &st, const DynOps &d,
                   SemanticsContext &, int, int) {
  // NANOSLEEP.SYNCS is the suspended try_wait: it also wakes when the
  // last polled mbarrier's awaited phase completes
  const auto opc = upper(st.opcode);
  const bool syncs = opc.find(".SYNCS") != std::string::npos;
  auto r = row(SemKind::nanosleep, syncs ? SEMA_FLAG_SYNCS : 0);
  r.fields["requested_ns"] = first_op(d, {"requested_ns", "ns"}, 0);
  return r;
}

}  // namespace

// arch decode entry points (in cpp/arch/<arch>/convert_rules.cpp)
SemRow decode_hopper_semantics(const StaticInfo &, const DynOps &,
                               SemanticsContext &, SemOp, int, int);
SemRow decode_blackwell_semantics(const StaticInfo &, const DynOps &,
                                  SemanticsContext &, SemOp, int, int);

SemRow decode_semantics(const StaticInfo &st, const DynOps &d,
                        SemanticsContext &ctx, SemOp op, int cta,
                        int wslot, int arch_kind) {
  if ((d.active_mask & d.predicate_mask) == 0) return {};
  const auto opc = upper(st.opcode);
  switch (op) {
    case SemOp::mbarrier_init:
      return d_mbarrier_init(d, ctx, cta, st);
    case SemOp::mbarrier_arrive:
      return d_mbarrier_arrive(d, ctx, cta, wslot);
    case SemOp::mbarrier_expect_tx: return d_expect_tx(d, ctx, cta, wslot);
    case SemOp::mbarrier_wait: return d_mbarrier_wait(d, ctx, cta, wslot);
    case SemOp::mbarrier_inval:
      return d_mbarrier_inval(d, ctx, cta, wslot);
    case SemOp::cta_barrier_arrive:
      return d_cta_barrier(st, d, ctx, cta, wslot, true);
    case SemOp::cta_barrier_wait:
      return d_cta_barrier(st, d, ctx, cta, wslot, false);
    case SemOp::cluster_barrier_arrive:
      return d_cluster_barrier(d, ctx, cta, wslot, true);
    case SemOp::cluster_barrier_wait:
      return d_cluster_barrier(d, ctx, cta, wslot, false);
    case SemOp::tma_load: {
      auto r = d_tma_load(d, ctx, cta, wslot);
      // cta_group::2: every destination's bytes post on its pair leader
      if (opc.find(".2CTA") != std::string::npos)
        r.flags |= SEMA_FLAG_CTA_PAIR;
      // tile::gather4 / scatter4: four row coordinates, each a box row
      if (opc.find("GATHER4") != std::string::npos ||
          opc.find("SCATTER4") != std::string::npos) {
        for (const char *k : {"shared_bytes", "global_bytes"}) {
          auto it = r.fields.find(k);
          if (it != r.fields.end()) it->second *= 4;
        }
      }
      return r;
    }
    case SemOp::bulk_group_issue:
      return d_bulk_group_issue(d, ctx, cta, wslot);
    case SemOp::bulk_group_commit:
      return d_bulk_group_commit(d, ctx, cta, wslot);
    case SemOp::bulk_group_wait:
      return d_bulk_group_wait(d, ctx, cta, wslot);
    case SemOp::async_copy_issue:
      return d_async_copy_issue(d, ctx, cta, wslot);
    case SemOp::async_copy_commit:
      return d_async_copy_commit(d, ctx, cta, wslot, opc);
    case SemOp::async_copy_wait:
      return d_async_copy_wait(d, ctx, cta, wslot);
    case SemOp::bulk_copy:
      return d_bulk_copy(d, ctx, cta, wslot, SemKind::bulk_copy);
    case SemOp::bulk_reduce:
      return d_bulk_copy(d, ctx, cta, wslot, SemKind::bulk_reduce);
    case SemOp::shared_store_async:
      return d_shared_store_async(d, ctx, cta, wslot);
    case SemOp::descriptor_visibility_publish: {
      if (opc.rfind("MEMBAR.ALL.GPU", 0) != 0) return {};
      auto stores = ctx.desc_stores.count(wslot)
                        ? ctx.desc_stores[wslot]
                        : std::vector<std::pair<std::uint64_t, int>>{};
      ctx.desc_stores.erase(wslot);
      std::map<std::uint64_t, std::set<int>> chunks;
      for (auto &e : stores) chunks[e.first].insert(e.second);
      std::vector<std::uint64_t> complete;
      for (auto &[b, cs] : chunks)
        if (cs.size() == 8) complete.push_back(b);
      if (complete.empty()) return {};
      auto r = row(SemKind::descriptor_visibility);
      r.fields["operation"] = DV_PUBLISH;
      r.fields["descriptor_count"] = complete.size();
      for (std::size_t i = 0; i < complete.size(); ++i)
        r.fields["descriptor_address_" + std::to_string(i)] = complete[i];
      ctx.desc_pending[wslot] = {complete, {}, 0};
      return r;
    }
    case SemOp::descriptor_visibility_store: {
      if (d.memory.empty() ||
          __builtin_popcount(d.active_mask & d.predicate_mask) != 1)
        return {};
      const auto &ref = d.memory[0];
      if (ref.width != 16 || ref.addrs.empty() || ref.addrs[0] % 16)
        return {};
      auto addr = ref.addrs[0];
      ctx.desc_stores[wslot].push_back({addr & ~0x7Full,
                                        (int)((addr & 0x7F) / 16)});
      return {};
    }
    case SemOp::descriptor_visibility_error_check: {
      auto it = ctx.desc_pending.find(wslot);
      if (it == ctx.desc_pending.end()) return {};
      std::string scope = opc.rfind("CGAERRBAR", 0) == 0 ? "cluster" : "sm";
      std::string expected = it->second.checks.empty() ? "sm" : "cluster";
      if (scope != expected || it->second.checks.count(scope))
        throw ConversionError("unsupported:descriptor_check_order");
      it->second.checks.insert(scope);
      return d_desc_vis(d, ctx, cta, wslot, DV_ERROR_CHECK);
    }
    case SemOp::descriptor_visibility_acquire: {
      auto *base = uniform(d, {"descriptor_address", "address", "op1",
                               "op0"});
      if (!base) throw ConversionError("missing_fact:descriptor_address");
      // acquire is defined across warps (engine resolves any warp's
      // publish); same-warp pending bookkeeping only when it exists
      auto it = ctx.desc_pending.find(wslot);
      if (it != ctx.desc_pending.end() &&
          std::find(it->second.bases.begin(), it->second.bases.end(),
                    *base) != it->second.bases.end())
        it->second.acquired += 1;
      auto r = row(SemKind::descriptor_visibility);
      r.fields["operation"] = DV_ACQUIRE;
      r.fields["descriptor_count"] = 1;
      r.fields["descriptor_address_0"] = *base;
      return r;
    }
    case SemOp::descriptor_visibility_proxy_fence:
      return d_desc_vis(d, ctx, cta, wslot, DV_PROXY_FENCE);
    case SemOp::depbar_wait: {
      bool emit = false;
      auto r = d_depbar(st, d, ctx, cta, wslot, &emit);
      return emit ? r : SemRow{};
    }
    case SemOp::nanosleep: return d_nanosleep(st, d, ctx, cta, wslot);
    case SemOp::dependents_launch:
      // PREEXIT executed (griddepcontrol.launch_dependents): marker
      // only; the engine charges the dispatch serializer at CTA
      // completion
      return row(SemKind::dependents_launch);
    default: break;
  }
  if (arch_kind == 1)
    return decode_hopper_semantics(st, d, ctx, op, cta, wslot);
  if (arch_kind == 2)
    return decode_blackwell_semantics(st, d, ctx, op, cta, wslot);
  return {};
}

// ---- pack_fields -----------------------------------------------------
namespace {
const char *const *fields_for(SemKind k) {
  static const char *const m_init[] = {"object", "expected_arrivals",
                                       "scope", "initial_phase", nullptr,
                                       "object_list_begin|"
                                       "object_list_count"};
  static const char *const m_arrive[] = {
      "object", "generation", "transaction_bytes", "arrival_count",
      "destination_mask", "object_list_begin|object_list_count"};
  static const char *const m_exptx[] = {"object", "generation",
                                        "transaction_bytes", "scope",
                                        nullptr, nullptr};
  static const char *const m_wait[] = {
      "object", "generation|parity", "block_offset", "scope",
      "timeout_hint", "after_ready_mask"};
  static const char *const m_inval[] = {"object", nullptr, nullptr,
                                        nullptr, nullptr, nullptr};
  static const char *const cta_b[] = {"barrier", "expected_threads",
                                      "generation", nullptr, nullptr,
                                      nullptr};
  static const char *const cl_b[] = {"generation", "destination_mask",
                                     nullptr, nullptr, nullptr, nullptr};
  static const char *const tma[] = {"object", "generation",
                                    "multicast_mask", "shared_bytes",
                                    "global_bytes", "tensor_map"};
  static const char *const gissue[] = {"group", "pending_allowed",
                                       "bytes", nullptr, nullptr,
                                       nullptr};
  static const char *const g2[] = {"group", "pending_allowed",
                                   nullptr, nullptr, nullptr, nullptr};
  static const char *const ac_issue[] = {"group", "pending_allowed",
                                         "object", nullptr, nullptr,
                                         nullptr};
  static const char *const ac_commit[] = {"group", "pending_allowed",
                                          "object", "arrival_count",
                                          "transaction_count", nullptr};
  static const char *const bc[] = {"object", "generation", "bytes",
                                   "multicast_mask", nullptr, nullptr};
  static const char *const wgmma_f[] = {
      "group", "accumulator_registers_begin|accumulator_registers_count",
      "shape_packed", "input_dtype", "accumulator_dtype", "source_mode"};
  static const char *const umma_i[] = {"kind", "cta_group", "a_source",
                                       "tensor_m|tensor_n",
                                       "descriptor"};
  static const char *const umma_c[] = {"object", "cta_group",
                                       "commit_mask", "scope", nullptr,
                                       nullptr};
  static const char *const cols[] = {"columns", nullptr, nullptr,
                                       nullptr, nullptr, nullptr};
  static const char *const tmem_x[] = {"shape", "repeat", nullptr,
                                       nullptr, nullptr, nullptr};
  static const char *const empty[] = {nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr};
  static const char *const clc[] = {"object", "generation",
                                    "destination_mask",
                                    "response_identity", "outcome",
                                    nullptr};
  static const char *const ssa[] = {"object", "bytes",
                                    "destination_rank",
                                    "object_list_begin|object_list_count",
                                    nullptr, nullptr};
  static const char *const ns[] = {"requested_ns", nullptr, nullptr,
                                   nullptr, nullptr, nullptr};
  static const char *const dv[] = {"operation", "descriptor_count",
                                   "descriptor_address_0",
                                   "descriptor_address_1",
                                   "descriptor_address_2",
                                   "descriptor_address_3"};
  static const char *const gsync[] = {"observation_kind", "class",
                                       nullptr, nullptr, nullptr,
                                       nullptr};
  static const char *const rec[] = {"region", "envelope_cycles",
                                     nullptr, nullptr, nullptr,
                                     nullptr};
  static const char *const clk[] = {"kind", nullptr, nullptr,
                                     nullptr, nullptr, nullptr};
  switch (k) {
    case SemKind::mbarrier_init: return m_init;
    case SemKind::mbarrier_arrive: return m_arrive;
    case SemKind::mbarrier_expect_tx: return m_exptx;
    case SemKind::mbarrier_wait: return m_wait;
    case SemKind::mbarrier_inval: return m_inval;
    case SemKind::cta_barrier_arrive:
    case SemKind::cta_barrier_wait: return cta_b;
    case SemKind::cluster_barrier_arrive:
    case SemKind::cluster_barrier_wait: return cl_b;
    case SemKind::tma_load: return tma;
    case SemKind::bulk_group_issue: return gissue;
    case SemKind::bulk_group_commit:
    case SemKind::bulk_group_wait:
    case SemKind::warpgroup_arrive:
    case SemKind::warpgroup_wait: return g2;
    case SemKind::async_copy_issue: return ac_issue;
    case SemKind::async_copy_commit: return ac_commit;
    case SemKind::async_copy_wait: return ac_issue;
    case SemKind::bulk_copy:
    case SemKind::bulk_reduce: return bc;
    case SemKind::wgmma: return wgmma_f;
    case SemKind::umma_issue: return umma_i;
    case SemKind::umma_commit: return umma_c;
    case SemKind::tmem_alloc:
    case SemKind::tmem_dealloc:
    case SemKind::tmem_relinquish: return cols;
    case SemKind::tmem_load:
    case SemKind::tmem_store:
    case SemKind::tmem_copy: return tmem_x;
    case SemKind::tmem_wait_store: return empty;
    case SemKind::clc_query: return clc;
    case SemKind::shared_store_async: return ssa;
    case SemKind::global_sync: return gsync;
    case SemKind::reconvergence_start:
    case SemKind::reconvergence_complete: return rec;
    case SemKind::nanosleep: return ns;
    case SemKind::descriptor_visibility: return dv;
    case SemKind::clock_marker: return clk;
    default: return empty;
  }
}
}  // namespace

std::array<std::uint64_t, 6>
pack_fields(SemKind kind,
            const std::map<std::string, std::uint64_t> &fields,
            std::uint32_t mbol_begin, std::uint32_t mbol_count) {
  std::array<std::uint64_t, 6> slots{};
  const char *const *names = fields_for(kind);
  auto get = [&](const std::string &n) -> std::uint64_t {
    if (n == "object_list_begin") return mbol_begin;
    if (n == "object_list_count") return mbol_count;
    if (n == "generation" && !fields.count("generation") &&
        fields.count("parity"))
      return GEN_UNRESOLVED | fields.at("parity");
    auto it = fields.find(n);
    if (it == fields.end()) {
      if (n == "object") return SEMA_NO_OBJECT;
      if (n == "generation") return GEN_UNRESOLVED;
      return 0;
    }
    if (n == "object" && it->second == SEMA_NO_OBJECT)
      return SEMA_NO_OBJECT;
    return it->second;
  };
  for (int i = 0; i < 6; ++i) {
    const char *n = names[i];
    if (!n || !*n) continue;   // unused slot (mbarrier_init has one at 4)
    const std::string name(n);
    if (name == "object_list_begin|object_list_count") {
      if (fields.count("object") && !fields.count("_has_list")) {
        // python: object_list written only when list present; slot
        // then holds begin|count packed
      }
      if (fields.count("_has_list")) {
        slots[i] = ((std::uint64_t)mbol_begin << 32) | mbol_count;
      }
      continue;
    }
    const auto bar = name.find('|');
    if (bar != std::string::npos) {
      const auto hi = get(name.substr(0, bar)) & 0xFFFFFFFFull;
      const auto lo = get(name.substr(bar + 1)) & 0xFFFFFFFFull;
      slots[i] = (hi << 32) | lo;
      continue;
    }
    slots[i] = get(name);
  }
  return slots;
}

}  // namespace cyclescope_convert

namespace cyclescope_convert {

std::string where_str(const StaticInfo &st) {
  char buf[32];
  std::snprintf(buf, sizeof buf, " @0x%llx",
                (unsigned long long)st.pc);
  return st.opcode + buf;
}

SemOp promote_store_to_mbarrier_init(const StaticInfo &s, const DynOps &d,
                                     const SemanticsContext &ctx, SemOp op) {
  // only the 32-bit STS and the STS.64 pair carry an init state
  // (STS.128 / sub-word forms / STSM are data stores)
  if (s.opcode != "STS" && s.opcode != "STS.64") return op;
  if (d.memory.empty() || ctx.mbarrier_addresses.empty()) return op;
  for (auto a : d.memory[0].addrs) {
    if (ctx.local_shared) a = ctx.local_shared(a);
    if (!ctx.mbarrier_addresses.count(a)) continue;
    // a store onto a named barrier word is an init; a missing captured
    // init value is a capture gap, not a silently skipped op
    if (d.uniform.count("expected_arrivals"))
      return SemOp::mbarrier_init;
    if (d.uniform.count("store_value_captured"))
      return op;   // captured data word that is not a legal init
                   // state: a data store, not an init
    throw ConversionError("missing_fact:mbarrier_init_value (" +
                          where_str(s) + ")");
  }
  return op;
}

int global_sync_class(const std::string &opcode) {
  static const char *const roots[] = {"ATOM", "RED", "LD", "ST"};
  const auto root = upper(opcode.substr(0, opcode.find('.')));
  for (int code = 1; code <= 4; ++code)
    if (root.rfind(roots[code - 1], 0) == 0) return code;
  return 0;
}

SemRow global_sync_row(const StaticInfo &st, const DynOps &d) {
  SemRow row;
  if (!d.has_gsyn || (d.active_mask & d.predicate_mask) == 0) return row;
  const int cls = global_sync_class(st.opcode);
  if (cls < 1 || cls > 3) return row;
  row.kind = SemKind::global_sync;
  row.fields["observation_kind"] = static_cast<std::uint64_t>(d.gsyn_kind);
  row.fields["class"] = static_cast<std::uint64_t>(cls);
  return row;
}

SemOp semantic_operation_hopper(const StaticInfo &);
SemOp semantic_operation_blackwell(const StaticInfo &);

SemOp semantic_operation_of(int arch_kind,
                            const StaticInfo &st) {
  if (arch_kind == 1) return semantic_operation_hopper(st);
  if (arch_kind == 2) return semantic_operation_blackwell(st);
  return SemOp::none;
}

}  // namespace cyclescope_convert
