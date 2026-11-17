// MechanismRuntime: leaf walk, PARM models, engine-axis guard, cache.
#include <cassert>
#include <cstdio>
#include <cmath>
#include "core/job_reader.hpp"
#include "core/mechanism_runtime.hpp"

int main(int argc, char** argv) {
  auto job = cyclescope::read_segment_file(argv[1]);
  cyclescope::MechanismRuntime::Options o;
  o.reference_sm_clock_khz = 1980000;   // 1.98 GHz
  cyclescope::MechanismRuntime rt(job, "h100", o);
  cyclescope::EngineFacts ef;
  // fixture AXIS row 0 = ready_warps_smsp (numeric axis id 0)
  ef.emplace_back(0u, 0u);
  auto& f1 = rt.resolve(1, ef);
  assert(f1.at(1) == 3.0);           // constant
  assert(f1.at(2) == 99.0);          // discrete miss -> default
  assert(f1.at(8) == 1.0);           // affine 2*0+1
  auto& f1b = rt.resolve(1, ef);
  assert(&f1b == &f1);               // cached
  ef[0].second = 5;
  auto& f2 = rt.resolve(1, ef);
  assert(f2.at(2) == 42.0);          // discrete hit
  // linear = piecewise interpolation between knots: x=5 lies between
  // (4, 18) and (8, 26) -> 20; the knot itself reproduces exactly
  assert(f2.at(7) == 20.0);
  ef[0].second = 4;
  assert(rt.resolve(1, ef).at(7) == 18.0);
  ef[0].second = 5;
  // two-axis discrete (op_axis = AXIS row 1, ready_warps_smsp = row 0):
  // both axes must match the point; a missing axis is a miss
  assert(f2.at(5) == 11.0);          // op_axis absent -> default
  ef.emplace_back(1u, 3u);
  auto& f2b = rt.resolve(1, ef);
  assert(f2b.at(5) == 77.0);         // (3, 5) hit
  ef[1].second = 4;
  auto& f2c = rt.resolve(1, ef);
  assert(f2c.at(5) == 11.0);         // op_axis mismatch -> default
  ef.pop_back();
  // single-point linear clamps: below/at/above 512 all -> 33.6
  for (double x : {400.0, 512.0, 700.0}) {
    ef[0].second = static_cast<std::uint64_t>(x);
    auto& fp = rt.resolve(1, ef);
    assert(fp.at(9) == 33.6);
  }
  ef[0].second = 1;             // walk to second child leaf
  auto& f3 = rt.resolve(2, ef);
  assert(f3.at(1) == 9.0);
  // schema-21 bytes/ns -> bytes/cycle (/GHz); bytes field untouched
  auto& f4 = rt.resolve(4, ef);
  assert(std::abs(f4.at(0) - 7602.0 / 1.98) < 1e-6);
  assert(f4.at(2) == 1000.0);

  // not_applicable: fixture node 4 has children guarded by
  // ready_warps_smsp == 7 / == 8 only
  {
    // (i) axis value present but matching no child -> NA
    cyclescope::EngineFacts m;
    m.emplace_back(0u, 0u);
    bool na = false, threw = false;
    const cyclescope::Fields* nf = nullptr;
    try { nf = &rt.resolve_node(4, m, "t", nullptr, &na); }
    catch (const cyclescope::EngineError&) { threw = true; }
    assert(!threw && na && nf && nf->empty());
    // (ii) same facts without not_applicable -> missing_fact throw
    threw = false;
    try { (void)rt.resolve_node(4, m, "t"); assert(false); }
    catch (const cyclescope::EngineError& e) {
      threw = true;
      assert(std::string(e.what()).rfind("missing_fact:axis:", 0) == 0);
    }
    assert(threw);
    // (iii) axis absent -> still missing_fact, na stays false
    cyclescope::EngineFacts empty;
    na = false; threw = false;
    try { (void)rt.resolve_node(4, empty, "t", nullptr, &na); }
    catch (const cyclescope::EngineError& e) {
      threw = true;
      assert(std::string(e.what()).rfind("missing_fact:axis:", 0) == 0);
    }
    assert(threw && !na);
    // matching value resolves the leaf normally
    cyclescope::EngineFacts hit;
    hit.emplace_back(0u, 7u);
    na = false;
    auto& fh = rt.resolve_node(4, hit, "t", nullptr, &na);
    assert(!na && fh.at(1) == 9.0);
  }
  // fail-closed discrete (leaf3 / node 7, same (op_axis=3, ready=5)
  // point but intercept = 1): a matching fact resolves to 77; a fact
  // miss throws unsupported:engine:discrete_miss
  {
    cyclescope::EngineFacts hit;
    hit.emplace_back(0u, 5u);
    hit.emplace_back(1u, 3u);
    assert(rt.resolve_node(7, hit, "t").at(10) == 77.0);
    bool threw = false;
    cyclescope::EngineFacts miss;
    miss.emplace_back(0u, 5u);
    miss.emplace_back(1u, 4u);
    try { (void)rt.resolve_node(7, miss, "t"); assert(false); }
    catch (const cyclescope::EngineError& e) {
      threw = true;
      assert(std::string(e.what()).rfind(
          "unsupported:engine:discrete_miss (", 0) == 0);
    }
    assert(threw);
    // an absent axis is also a miss, not a silent default
    threw = false;
    cyclescope::EngineFacts partial;
    partial.emplace_back(0u, 5u);
    try { (void)rt.resolve_node(7, partial, "t"); assert(false); }
    catch (const cyclescope::EngineError& e) {
      threw = true;
      assert(std::string(e.what()).rfind(
          "unsupported:engine:discrete_miss (", 0) == 0);
    }
    assert(threw);
  }
  std::printf("ok\n");
  return 0;
}
