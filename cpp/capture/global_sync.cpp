// global_sync_observation.py verbatim port.
#include "capture/global_sync.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <vector>

namespace cyclescope_convert::capture {
namespace {

const char *kAtomicRoots[] = {"ATOM", "ATOMG", "ATOMS"};
const char *kReleaseRoots[] = {"STG", "ST", "STS", "STAS", "RED",
                               "REDG", "REDS", "MEMBAR", "FENCE",
                               "ARRIVES", "UTCBAR"};
const char *kAtomicOps[] = {"ADD", "EXCH", "CAS", "MIN", "MAX", "INC",
                            "DEC", "AND", "OR", "XOR"};
bool in(const std::string &s, const char *const *arr, int n) {
  for (int i = 0; i < n; ++i) if (s == arr[i]) return true;
  return false;
}
std::vector<std::string> parts_of(const std::string &op) {
  std::string u = op;
  for (auto &c : u) c = char(std::toupper(c));
  std::vector<std::string> out;
  std::size_t b = 0;
  while (true) {
    const auto c = u.find('.', b);
    out.push_back(u.substr(b, c == std::string::npos ? c : c - b));
    if (c == std::string::npos) break;
    b = c + 1;
  }
  return out;
}
std::int64_t signed_of(std::uint64_t v, int bits) {
  return (v >> (bits - 1)) & 1 ? std::int64_t(v) - (1ll << bits)
                             : std::int64_t(v);
}
std::uint64_t join(const std::uint32_t *w, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v |= std::uint64_t(w[i]) << (32 * i);
  return v;
}

// _shape
std::tuple<std::string, std::string, int> shape(const std::string &op) {
  const auto parts = parts_of(op);
  std::string aop, kind = "uint";
  int bits = 32;
  for (std::size_t i = 1; i < parts.size(); ++i)
    if (in(parts[i], kAtomicOps, 10)) { aop = parts[i]; break; }
  for (std::size_t i = 1; i < parts.size(); ++i) {
    const auto &p = parts[i];
    if (p == "S32" || p == "S64") kind = "sint";
    else if (p == "F32" || p == "F64" || p == "F16" || p == "BF16")
      kind = "float";
    if (p == "64" || p == "S64" || p == "U64" || p == "F64") bits = 64;
  }
  return {aop, kind, bits};
}

// _post — post-operation value (nullopt = unknown)
std::optional<std::int64_t> post(const std::string &op,
                                 const std::string &kind, int bits,
                                 std::uint64_t old,
                                 const std::vector<std::uint64_t> &ops) {
  const std::uint64_t mask = bits == 64 ? ~0ull : (1ull << bits) - 1;
  if (op.empty() || ops.empty()) return std::nullopt;
  const std::uint64_t a = ops[0] & mask;
  old &= mask;
  if (op == "EXCH") return std::int64_t(a);
  if (op == "CAS") {
    if (ops.size() < 2) return std::nullopt;
    return std::int64_t(old == a ? ops[1] & mask : old);
  }
  if (kind == "float") {
    if (op != "ADD") return std::nullopt;
    if (bits == 32) {
      std::uint32_t oi = std::uint32_t(old), ai = std::uint32_t(a);
      float fo, fa;
      std::memcpy(&fo, &oi, 4); std::memcpy(&fa, &ai, 4);
      const float r = fo + fa;
      std::uint32_t ri;
      std::memcpy(&ri, &r, 4);
      return std::int64_t(ri);
    }
    std::uint64_t ri;
    double fo, fa;
    std::memcpy(&fo, &old, 8); std::memcpy(&fa, &a, 8);
    const double r = fo + fa;
    std::memcpy(&ri, &r, 8);
    return std::int64_t(ri);
  }
  if (op == "ADD") return std::int64_t((old + a) & mask);
  if (op == "AND" || op == "OR" || op == "XOR") {
    const std::uint64_t r =
        op == "AND" ? old & a : op == "OR" ? old | a : old ^ a;
    return std::int64_t(r & mask);
  }
  if (op == "MIN" || op == "MAX") {
    const bool take_min = op == "MIN";
    if (kind == "sint") {
      const auto so = signed_of(old, bits), sa = signed_of(a, bits);
      return std::int64_t(take_min ? std::min(so, sa)
                                   : std::max(so, sa));
    }
    return std::int64_t(take_min ? std::min(old, a)
                                 : std::max(old, a));
  }
  if (op == "INC")
    return std::int64_t(old >= a ? 0 : (old + 1) & mask);
  if (op == "DEC")
    return std::int64_t(old == 0 || old > a ? a : (old - 1) & mask);
  return std::nullopt;
}

}  // namespace

int observation_kind(const std::string &root, int) {
  if (in(root, kAtomicRoots, 3)) return 3;
  if (in(root, kReleaseRoots, 11)) return 1;
  return 2;
}

GSyncObsResult observations(const DynRecord &rec, const StaticRow &row) {
  GSyncObsResult res;
  if (rec.gsync.empty()) return res;
  std::string root = row.inst.opcode.substr(
      0, row.inst.opcode.find('.'));
  for (auto &c : root) c = char(std::toupper(c));
  const auto [op, kind, bits] = shape(row.inst.opcode);
  const int wpv = bits / 32;
  const int kind_code = observation_kind(root, 0);
  const std::int64_t address =
      !rec.mrefs.empty() && !rec.mrefs[0].empty()
          ? std::int64_t(rec.mrefs[0][0]) : 0;
  const DynRecord::GSync *phase0 = nullptr, *phase1 = nullptr;
  std::uint64_t seq = ~0ull;
  for (const auto &g : rec.gsync) {
    if (g.phase == 0) phase0 = &g;
    if (g.phase == 1) phase1 = &g;
    seq = std::min(seq, g.sequence);
  }
  const std::size_t n = std::max(
      phase0 ? phase0->words.size() : 0,
      phase1 ? phase1->words.size() : 0);
  std::vector<std::int64_t> befores, afters, flags_per_lane;
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::uint64_t> operands;  // u32 words joined
    std::optional<std::int64_t> pre, dst;
    if (phase0 && i < phase0->words.size()) {
      auto lw = phase0->words[i];
      lw.resize(phase0->nwords);
      if ((phase0->flags & 1) && int(lw.size()) >= wpv) {
        pre = std::int64_t(
            join(lw.data() + lw.size() - wpv, wpv));
        lw.resize(lw.size() - wpv);
      }
      for (std::size_t j = 0; j + wpv <= lw.size(); j += wpv)
        operands.push_back(join(lw.data() + j, wpv));
    }
    if (phase1 && i < phase1->words.size())
      dst = std::int64_t(join(phase1->words[i].data(),
                              phase1->nwords));
    std::optional<std::int64_t> before, after;
    if (in(root, kAtomicRoots, 3)) {
      before = dst ? dst : pre;
      if (before)
        after = post(op, kind, bits, std::uint64_t(*before), operands);
    } else if (root.rfind("RED", 0) == 0) {
      before = pre;
      if (pre) after = post(op, kind, bits, std::uint64_t(*pre), operands);
    } else if (root.rfind("ST", 0) == 0) {
      if (!operands.empty()) after = std::int64_t(operands[0]);
    } else {
      after = dst;
    }
    befores.push_back(before.value_or(0));
    afters.push_back(after.value_or(0));
    flags_per_lane.push_back((after ? 1 : 0) | (before ? 2 : 0));
  }
  GlobalSyncObservation o;
  o.kind = kind_code;
  o.address = address;
  o.value = afters.empty() ? 0 : afters[0];
  o.lane = 0;
  o.acquire_mode = 0;
  o.post_known = flags_per_lane.empty() ? 0 : int(flags_per_lane[0]);
  o.before = befores.empty() ? 0 : befores[0];
  o.sequence = seq;
  res.obs.push_back(o);
  res.lane["gsync_before"] = befores;
  res.lane["gsync_after"] = afters;
  res.lane_order = {"gsync_before", "gsync_after"};
  return res;
}

}  // namespace cyclescope_convert::capture
