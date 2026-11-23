// poll_folding_test: placement v5 rule P'' fixtures (wait origin by the
// chain of failed polls on the same barrier word); mirrors
// tests/common/test_poll_folding.py.
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "capture/poll_folding.hpp"

using namespace cyclescope_convert::capture;

namespace {
int failures = 0;
const char *kPoll = "SYNCS.PHASECHK.TRANS64.TRYWAIT";

struct Spec {
  std::string op;
  std::uint64_t pc;
  int ready = -1;   // -1 unknown, 0 failed, 1 ready
  int wbar = 7;
  unsigned wait = 0;
  std::string text;
  std::vector<std::pair<std::string, std::int64_t>> ops;
  std::uint64_t mref = 0x1000;
};
Spec P(std::uint64_t pc, int ready, int wbar = 0, unsigned wait = 0,
       std::string text = "", std::vector<std::pair<std::string,
       std::int64_t>> ops = {}, std::uint64_t mref = 0x1000) {
  return {kPoll, pc, ready, wbar, wait, text, ops, mref};
}
Spec X(const std::string &op, std::uint64_t pc, int wbar = 7,
       unsigned wait = 0) {
  return {op, pc, -1, wbar, wait};
}

struct Fx {
  std::vector<DynRecord> recs;
  std::vector<StaticRow> rows;
  StaticOf static_of() const {
    return [this](const DynRecord &r) { return &rows[r.idx]; };
  }
};
Fx make(const std::vector<Spec> &spec) {
  Fx f;
  f.rows.resize(spec.size());
  for (std::size_t n = 0; n < spec.size(); ++n) {
    const Spec &s = spec[n];
    DynRecord r;
    r.idx = std::uint32_t(n);
    r.exec_mask = 0xFFFFFFFFu;
    r.mrefs = {{s.mref}};
    r.has_after_ready = s.ready >= 0;
    r.after_ready = s.ready == 1 ? 0xFFFFFFFFu : 0;
    for (const auto &[k, v] : s.ops) {
      r.ops[k] = {v};
      r.ops_order.push_back(k);
    }
    f.recs.push_back(r);
    StaticRow &row = f.rows[n];
    row.idx = int(n);
    row.inst.opcode = s.op;
    row.opcode_text = s.op;
    row.operands_text = s.text;
    row.ops_layout = {"R2"};
    row.inst.pc = s.pc;
    row.inst.code_address = 0x7000 + s.pc;
    row.inst.wait_mask = s.wait;
    row.inst.packed_ctrl = std::uint32_t(s.wbar) << 9;
  }
  return f;
}
using Map = std::map<int, std::uint64_t>;
void check(bool ok, const char *what) {
  if (ok) return;
  ++failures;
  std::fprintf(stderr, "FAIL %s\n", what);
}
FoldResult run(const Fx &f) { return fold_warp(f.recs, f.static_of()); }
}  // namespace

int main() {
  {  // (i)
    Fx f = make({P(0x100, 0), X("BRA", 0x110, 7, 1), P(0x100, 0),
                 X("BRA", 0x110, 7, 1), P(0x100, 1)});
    auto r = run(f);
    check(r.folded == std::set<int>({0, 1, 2, 3}) && r.dropped == 2 &&
              r.wait_origin.empty(), "same address");
  }
  {  // (ii)
    Fx f = make({P(0x100, 0), X("ISETP", 0x110), X("BRA", 0x120, 7, 1),
                 P(0x200, 0), X("NANOSLEEP", 0x210), P(0x200, 1)});
    auto r = run(f);
    check(r.folded == std::set<int>({0, 1, 2, 3, 4}) && r.dropped == 2 &&
              r.wait_origin == Map({{5, 0x7120}}), "pollcute");
  }
  for (int behind : {10, 14, 16, 17}) {  // (iii) (iv)
    std::vector<Spec> sp = {P(0x100, 0)};
    for (int n = 0; n < behind - 1; ++n) sp.push_back(X("SHF", 0x110 + 16 * n));
    sp.push_back(X("BRA", 0x110 + 16 * (behind - 1), 7, 1));
    sp.push_back(P(0x400, 0));
    sp.push_back(P(0x400, 1));
    Fx f = make(sp);
    auto r = run(f);
    const Map want = behind == 17 ? Map()
        : Map({{int(sp.size()) - 1, 0x7000 + 0x110 + 16u * (behind - 1)}});
    check(r.wait_origin == want, "reach");
  }
  {  // (v) LDS between inline poll and its BRA
    const std::string t = "P0, [R8+URZ], R3";
    Fx f = make({P(0x100, 0, 0, 0, t, {{"R8", 0x40}}), X("LDS", 0x110, 1),
                 X("IMAD", 0x120, 7, 2), X("BRA", 0x130, 7, 1),
                 P(0x200, 0, 0, 0, t, {{"R8", 0x40}}),
                 P(0x200, 1, 0, 0, t, {{"R8", 0x40}})});
    auto r = run(f);
    check(!r.folded.count(0) && r.folded.count(4) &&
              r.wait_origin == Map({{5, 0x7130}}), "lds");
  }
  {  // (vi) barrier word from registers, not the mref
    const std::string ti = "P0, [UR4+0x28040], R2",
                      tl = "P0, [R3+URZ+0x28040], R2";
    auto inl = [&] { return P(0x100, 0, 0, 0, ti,
                              {{"UR4", 0x100}, {"R2", 0}}, 0x5000); };
    auto loop = [&](int ready, std::int64_t r3, std::int64_t r2) {
      return P(0x200, ready, 0, 0, tl, {{"R3", r3}, {"R2", r2}},
               0x5000 + 32649);
    };
    Fx f = make({inl(), X("BRA", 0x110, 7, 1), loop(0, 0x100, 0),
                 loop(1, 0x100, 0)});
    check(run(f).wait_origin == Map({{3, 0x7110}}), "word equal");
    Fx g = make({inl(), X("BRA", 0x110, 7, 1), loop(0, 0x100, 0),
                 loop(1, 0x140, 0)});
    check(run(g).wait_origin.empty(), "other word");
    Fx h = make({inl(), X("BRA", 0x110, 7, 1), loop(0, 0x100, 0),
                 loop(1, 0x100, std::int64_t(1) << 31)});
    check(run(h).wait_origin.empty(), "other parity");
  }
  {  // (vii) a record writing b before the waiter
    Fx f = make({P(0x100, 0), X("LDS", 0x110, 0), X("BRA", 0x120, 7, 1),
                 P(0x200, 0), P(0x200, 1)});
    check(run(f).wait_origin.empty(), "rewritten");
    Fx g = make({P(0x100, 0), X("LDS", 0x110, 0, 1), P(0x200, 0),
                 P(0x200, 1)});
    check(run(g).wait_origin == Map({{3, 0x7110}}), "wait and write");
  }
  {  // (viii) unknown result ends the chain
    Fx f = make({P(0x100, 0), X("BRA", 0x110, 7, 1), P(0x300, -1),
                 P(0x200, 0), P(0x200, 1)});
    check(run(f).wait_origin.empty(), "unknown ends chain");
  }
  {  // (ix) cublas layout
    Fx f = make({P(0x100, 0, 2), X("ISETP", 0x110), P(0x200, 0, 0, 4),
                 X("NANOSLEEP", 0x210), P(0x200, 1, 0, 4)});
    auto r = run(f);
    check(r.dropped == 2 && r.wait_origin == Map({{4, 0x7200}}), "cublas");
  }
  if (failures) return 1;
  std::puts("poll_folding_test ok");
  return 0;
}
