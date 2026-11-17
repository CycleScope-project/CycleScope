#!/usr/bin/env python3
"""atomic_issue derivation (tools/microbench/atomic_issue/protocol.json).

  model.py --device h100|b300 --run-a <dir with atomic.jsonl>
           --run-b <dir> --out <model.json> [--checks <checks.json>]

slope(op, L, W) = OLS slope over U of the per-U median of (t_e - t_s);
cycles(op, L) = slope(op, L, W = 4) / 4.  Diagnostic: if slope(W = 1)
>= 0.9 x slope(W = 4) the bound is per SMSP, not per SM -> open engine
question (kind 15 scope); the install still uses W = 4.  Thresholds are
copied from the frozen protocol text.

r2 addendum (variants atomic.<op>.L<L>.W<W>.indep, U in {16, 32, 64,
128}): cycles_indep(op, L) = (med(U=128) - med(U=64)) / 64 / 4 at W = 4;
steady = relative gap between that increment and the U=32->64 one;
verdict measured iff run agreement <= 5 % and steady <= 10 %.  Control:
the same-run same-address atomg / redg L=1 W=4 slope must reproduce the
r1 install within 10 % or nothing is installed.

r3 addendum (variants atomic.<op>.L<L>.W<W>.S<S>, block stride S bytes;
the .indep variant is S = 128): same per-cell rule as r2.  Lines sweep
L = 1, S in {4, 8, 16, 32, 64, 128} -> cache_line_count knots {1, 2, 4, 8,
16, 32}; same-word sweep S = 4, L in {1..32} (spread > 10 % -> open
item).  Controls: r1 same-address cell within 10 % of r1 and the S = 128
cell within 10 % of r2 where r2 was measured.

r4 addendum (W = 16 cells, U in {16, 32, 64}): cycles = (med(U=64) -
med(U=32)) / 32 / 16, steady = relative gap to the 16->32 increment;
same per-cell gate.  Control: the W = 16 redg lines knots reproduce the
r3 W = 4 install within 10 %.  Install ATOMG / ATOM on cache_line_count
from the W = 16 lines cells when all six knots are measured.

r5 addendum (same W = 16 cells, fresh run): cycles = mean over U in
{32, 64} of T(U) / U / 16, flat = |T(64)/64 - T(32)/32| / mean <= 10 %,
run agreement <= 5 %; controls as r4 (r1 cell; W = 16 redg knots vs the
r3 install under this derivation).  Install ATOMG / ATOM on
cache_line_count when all six knots are measured.
"""
import argparse
import json
import statistics
from pathlib import Path

# --- thresholds copied verbatim from protocol.json ------------------------
FIT_RESID = 0.05       # "fit max relative residual <= 5 %"
RUN_AGREE = 0.05       # "run agreement <= 5 %"
PERSMSP = 0.9          # "slope(W = 1) >= 0.9 x slope(W = 4) -> per SMSP"
# r2 addendum
R2_STEADY = 0.10       # "steady <= 10 %"
R2_CONTROL = 0.10      # "control reproduces r1 within 10 %"
R2_U_HI, R2_U_MID, R2_U_LO = 128, 64, 32
R2_OPEN_ITEM = 0.20    # "differs from r1 by > 20 % -> open item"
R2_OPS = ("redg", "atomg", "atom")
R1_CONTROL = {"h100": {"atomg": 61.63, "redg": 31.61},
              "b300": {"atomg": 54.50, "redg": 34.13}}
# r3 addendum
R3_S_KNOTS = (4, 8, 16, 32, 64, 128)       # block stride -> lines S / 4
R3_LINES = {4: 1, 8: 2, 16: 4, 32: 8, 64: 16, 128: 32}
R3_SAME_WORD_SPREAD = 0.10                 # "spread > 10 % -> open item"
R2_MEASURED = {"h100": {"redg": 32.0},     # r2 install (atomic_issue.<dev>.r2)
               "b300": {"redg": 31.99, "atomg": 58.09, "atom": 55.31}}
# r5 addendum
R5_FLAT = 0.10         # "flat <= 10 %"
R5_US = (32, 64)
# r4 addendum
R4_W = 16
R4_U_HI, R4_U_MID, R4_U_LO = 64, 32, 16
R4_CONTROL = 0.10      # "W = 16 redg knots within 10 % of the r3 install"
R3_REDG_INSTALL = {"h100": {1: 5.0, 2: 10.0, 4: 20.0, 8: 40.0, 16: 48.0, 32: 32.0},
                   "b300": {1: 5.08, 2: 8.07, 4: 16.06, 8: 31.99, 16: 31.99, 32: 31.99}}

OPS = {"redg": 0, "atomg": 1, "atoms": 2, "reds": 3, "atom": 4}
L_KNOTS = (1, 2, 4, 8, 16, 32)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def ols(xs, ys):
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return b, my - b * mx


def slope_fit(series):
    us = sorted(series)
    if len(us) < 2:
        return None
    meds = [statistics.median(series[u]) for u in us]
    b, a = ols(us, meds)
    resid = max(abs(m - (a + b * u)) / max(1.0, abs(m)) for u, m in zip(us, meds))
    return {"slope": b, "max_rel_resid": resid, "us": us, "medians": meds}


def rel(a, b):
    m = (a + b) / 2
    return abs(a - b) / m if m else float("inf")


def load(run_dir):
    """(op, l, w) -> u -> [samples], merged over jsonl files."""
    rows = {}
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "samples" not in r:
                continue
            v = r["variant"].split(".")
            if len(v) not in (4, 5) or v[0] != "atomic":
                continue
            if len(v) == 5 and v[4] != "indep" and not v[4].startswith("S"):
                continue
            # r2 independent-line variants are keyed by op "<op>.indep",
            # r3 stride variants by "<op>.S<bytes>"
            op = v[1] + ("." + v[4] if len(v) == 5 else "")
            key = (op, int(v[2][1:]), int(v[3][1:]))
            rows.setdefault(key, {}).setdefault(int(r["u"]), []).extend(
                float(x) for x in r["samples"])
    return rows


def gate_ok(checks, op, u, indep=False):
    if not checks:
        return True
    key = f"atomic/{op}/{u}" + ("/indep" if indep else "")
    k = checks.get("atomic", {}).get("kernels", {}).get(key)
    return k is not None and k.get("ok")


def r2_fit(series, us=(R2_U_LO, R2_U_MID, R2_U_HI)):
    """r2 addendum estimate from the per-U medians of one (op, L, W);
    r4 passes its own (lo, mid, hi) U points."""
    u_lo, u_mid, u_hi = us
    if not all(u in series for u in us):
        return None
    med = {u: statistics.median(series[u]) for u in series}
    hi = (med[u_hi] - med[u_mid]) / (u_hi - u_mid)
    lo = (med[u_mid] - med[u_lo]) / (u_mid - u_lo)
    steady = abs(hi - lo) / abs(hi) if hi else float("inf")
    f = slope_fit(series)
    return {"increment": hi, "increment_lo": lo, "steady": steady,
            "ols_slope": f["slope"], "ols_max_rel_resid": f["max_rel_resid"],
            "us": f["us"], "medians": f["medians"]}


def r3_fits(rows, checks):
    """(op, l, w, sbytes) -> r2_fit over the r2 / r3 independent-line cells."""
    out = {}
    for (op, l, w), us in rows.items():
        if "." not in op:
            continue
        base, tag = op.split(".", 1)
        sb = 128 if tag == "indep" else int(tag[1:])
        series = {u: s for u, s in us.items()
                  if gate_ok(checks, OPS[base], u, indep=True)}
        f = r2_fit(series, (R4_U_LO, R4_U_MID, R4_U_HI) if w == R4_W
                   else (R2_U_LO, R2_U_MID, R2_U_HI))
        if f:
            out[(base, l, w, sb)] = f
    return out


def r2_fits(rows, checks):
    out = {}
    for (op, l, w), us in rows.items():
        if not op.endswith(".indep"):
            continue
        base = op[:-len(".indep")]
        series = {u: s for u, s in us.items()
                  if gate_ok(checks, OPS[base], u, indep=True)}
        f = r2_fit(series)
        if f:
            out[(base, l, w)] = f
    return out


def fits(rows, checks):
    """(op, l, w) -> slope_fit dict, gate-filtered."""
    out = {}
    for (op, l, w), us in rows.items():
        if "." in op:
            continue
        series = {u: s for u, s in us.items() if gate_ok(checks, OPS[op], u)}
        f = slope_fit(series)
        if f:
            out[(op, l, w)] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--checks", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    checks = json.loads(Path(a.checks).read_text()) if a.checks else {}
    fa, fb = fits(load(a.run_a), checks), fits(load(a.run_b), checks)

    res = {"schema": "cyclescope.microbench-model.v1",
           "protocol": "tools/microbench/atomic_issue/protocol.json",
           "device": a.device, "runs": {"a": a.run_a, "b": a.run_b},
           "cells": {}, "open_questions": []}
    for op in OPS:
        cells = {}
        for l in L_KNOTS:
            e4a, e4b = fa.get((op, l, 4)), fb.get((op, l, 4))
            e1a, e1b = fa.get((op, l, 1)), fb.get((op, l, 1))
            ent = {}
            if e4a and e4b:
                cyc = mean([e4a["slope"] / 4, e4b["slope"] / 4])
                resid = max(e4a["max_rel_resid"], e4b["max_rel_resid"])
                agree = rel(e4a["slope"], e4b["slope"])
                ent = {"cycles": cyc, "resid": resid, "agree": agree,
                       "verdict": "measured" if resid <= FIT_RESID and agree <= RUN_AGREE
                                   else "experimental"}
            if e1a and e4a:
                ent["diag_w1_over_w4"] = e1a["slope"] / e4a["slope"] if e4a["slope"] else None
                if ent["diag_w1_over_w4"] is not None and ent["diag_w1_over_w4"] >= PERSMSP:
                    res["open_questions"].append(
                        f"{op}.L{l}: slope(W=1) >= {PERSMSP} x slope(W=4); the bound looks per SMSP, kind 15 scope")
            ent["slope_w4"] = [e4a and e4a["slope"], e4b and e4b["slope"]]
            ent["slope_w1"] = [e1a and e1a["slope"], e1b and e1b["slope"]]
            cells[str(l)] = ent
        res["cells"][op] = cells
    # install knots: cycles linear(atomic_same_address_lanes) per operation class
    # protocol install mapping: ATOM (generic) <- ATOM.E cell; ATOMG / RED /
    # REDG <- the global cells (atomg, redg); ATOMS / REDS <- shared cells.
    res["install"] = {
        "knots_l": list(L_KNOTS),
        "op_class_from_cell": {"ATOM": "atom", "ATOMG": "atomg", "RED": "redg",
                               "REDG": "redg", "ATOMS": "atoms", "REDS": "reds"},
        "cycles": {op: {l: res["cells"][op][str(l)].get("cycles") for l in L_KNOTS} for op in OPS}}
    # ---- r2 addendum: independent-line cells, control, install -----------
    ra, rb = r2_fits(load(a.run_a), checks), r2_fits(load(a.run_b), checks)
    if ra or rb:
        r2 = {"cells": {}, "control": {}, "open_items": [], "install": {}}
        ctrl_ok = True
        for op in ("atomg", "redg"):
            want = R1_CONTROL[a.device][op]
            ent = {"r1": want, "runs": [], "ok": False}
            for f in (fa, fb):
                e = f.get((op, 1, 4))
                ent["runs"].append(e["slope"] / 4 if e else None)
            got = [x for x in ent["runs"] if x is not None]
            ent["ok"] = len(got) == 2 and all(rel(x, want) <= R2_CONTROL for x in got)
            ctrl_ok = ctrl_ok and ent["ok"]
            r2["control"][op] = ent
        r2["control"]["ok"] = ctrl_ok
        for op in R2_OPS:
            cells = {}
            for l in L_KNOTS:
                e4a, e4b = ra.get((op, l, 4)), rb.get((op, l, 4))
                e1a, e1b = ra.get((op, l, 1)), rb.get((op, l, 1))
                ent = {}
                if e4a and e4b:
                    cyc = mean([e4a["increment"] / 4, e4b["increment"] / 4])
                    agree = rel(e4a["increment"], e4b["increment"])
                    steady = max(e4a["steady"], e4b["steady"])
                    ent = {"cycles": cyc, "agree": agree, "steady": steady,
                           "verdict": "measured" if agree <= RUN_AGREE and steady <= R2_STEADY
                                       else "experimental",
                           "ols_max_rel_resid": max(e4a["ols_max_rel_resid"],
                                                    e4b["ols_max_rel_resid"]),
                           "medians_w4": [e4a["medians"], e4b["medians"]]}
                    r1_cell = res["cells"].get(op, {}).get(str(l), {}).get("cycles")
                    ent["r1_same_address_cycles"] = r1_cell
                    if r1_cell and rel(cyc, r1_cell) > R2_OPEN_ITEM:
                        r2["open_items"].append(
                            f"{op}.L{l}: independent-line {cyc:.2f} vs same-address {r1_cell:.2f}: "
                            "consecutive same-line atomics of one SM serialize at L2; kind 15 has no "
                            "per-line state (candidate axis atomic_line_reuse, not introduced)")
                if e1a and e4a:
                    ent["diag_w1_over_w4"] = (e1a["increment"] / e4a["increment"]
                                              if e4a["increment"] else None)
                ent["increment_w4"] = [e4a and e4a["increment"], e4b and e4b["increment"]]
                ent["increment_w1"] = [e1a and e1a["increment"], e1b and e1b["increment"]]
                cells[str(l)] = ent
            r2["cells"][op] = cells
        r2["install"] = {
            "knots_l": list(L_KNOTS),
            "op_class_from_cell": {"ATOM": "atom", "ATOMG": "atomg", "RED": "redg", "REDG": "redg"},
            "control_ok": ctrl_ok,
            "cycles": {op: {l: (r2["cells"][op][str(l)].get("cycles")
                               if ctrl_ok and r2["cells"][op][str(l)].get("verdict") == "measured"
                               else None) for l in L_KNOTS} for op in R2_OPS}}
        res["r2"] = r2
    # ---- r3 addendum: lines sweep (cache_line_count knots) + same-word sweep
    ta, tb = r3_fits(load(a.run_a), checks), r3_fits(load(a.run_b), checks)
    if any(k[3] != 128 for k in ta) or any(k[3] != 128 for k in tb):
        def cell(op, l, sb):
            e4a, e4b = ta.get((op, l, 4, sb)), tb.get((op, l, 4, sb))
            e1a = ta.get((op, l, 1, sb))
            if not (e4a and e4b):
                return None
            agree = rel(e4a["increment"], e4b["increment"])
            steady = max(e4a["steady"], e4b["steady"])
            return {"cycles": mean([e4a["increment"] / 4, e4b["increment"] / 4]),
                    "agree": agree, "steady": steady,
                    "verdict": "measured" if agree <= RUN_AGREE and steady <= R2_STEADY
                               else "experimental",
                    "ols_max_rel_resid": max(e4a["ols_max_rel_resid"], e4b["ols_max_rel_resid"]),
                    "diag_w1_over_w4": (e1a["increment"] / e4a["increment"]
                                        if e1a and e4a["increment"] else None),
                    "medians_w4": [e4a["medians"], e4b["medians"]]}
        r3 = {"lines": {}, "same_word": {}, "control": {}, "open_items": [], "install": {}}
        ctrl_ok = bool(res.get("r2", {}).get("control", {}).get("ok", False))
        r3["control"]["r1_same_address"] = res.get("r2", {}).get("control")
        for op, want in R2_MEASURED[a.device].items():
            c = cell(op, 1, 128)
            ok = c is not None and rel(c["cycles"], want) <= R2_CONTROL
            r3["control"][f"r2_{op}"] = {"r2": want, "got": c and c["cycles"], "ok": ok}
            ctrl_ok = ctrl_ok and ok
        r3["control"]["ok"] = ctrl_ok
        for op in R2_OPS:
            r3["lines"][op] = {str(R3_LINES[sb]): cell(op, 1, sb) for sb in R3_S_KNOTS}
            r3["same_word"][op] = {str(l): cell(op, l, 4) for l in L_KNOTS}
            sw = {l: c["cycles"] for l, c in r3["same_word"][op].items() if c}
            if "1" in sw and len(sw) > 1:
                spread = (max(sw.values()) - min(sw.values())) / sw["1"]
                r3["same_word"][op]["spread"] = spread
                if spread > R3_SAME_WORD_SPREAD:
                    r3["open_items"].append(
                        f"{op}: one-line cycles over same-word lanes L spread {spread:.2f} "
                        f"({ {k: round(v, 1) for k, v in sw.items()} }); leaf stays on cache_line_count")
            knots = r3["lines"][op]
            full = all(knots[str(n)] and knots[str(n)]["verdict"] == "measured"
                       for n in (1, 2, 4, 8, 16, 32))
            r3["install"][op] = ({str(n): knots[str(n)]["cycles"] for n in (1, 2, 4, 8, 16, 32)}
                                 if ctrl_ok and full else None)
        r3["install"]["axis"] = "cache_line_count"
        r3["install"]["op_class_from_cell"] = {"ATOM": "atom", "ATOMG": "atomg",
                                               "RED": "redg", "REDG": "redg"}
        res["r3"] = r3
    # ---- r4 addendum: W = 16 cells ---------------------------------------
    if any(k[2] == R4_W for k in ta) or any(k[2] == R4_W for k in tb):
        def cell16(op, l, sb):
            e4a, e4b = ta.get((op, l, R4_W, sb)), tb.get((op, l, R4_W, sb))
            if not (e4a and e4b):
                return None
            agree = rel(e4a["increment"], e4b["increment"])
            steady = max(e4a["steady"], e4b["steady"])
            return {"cycles": mean([e4a["increment"] / R4_W, e4b["increment"] / R4_W]),
                    "agree": agree, "steady": steady,
                    "verdict": "measured" if agree <= RUN_AGREE and steady <= R2_STEADY
                               else "experimental",
                    "medians_w16": [e4a["medians"], e4b["medians"]]}
        r4 = {"lines": {}, "same_word": {}, "control": {}, "open_items": [], "install": {}}
        ctrl_ok = bool(res.get("r2", {}).get("control", {}).get("ok", False))
        r4["control"]["r1_same_address"] = res.get("r2", {}).get("control", {}).get("ok")
        for sb in R3_S_KNOTS:
            n = R3_LINES[sb]
            c = cell16("redg", 1, sb)
            want = R3_REDG_INSTALL[a.device][n]
            ok = c is not None and rel(c["cycles"], want) <= R4_CONTROL
            r4["control"][f"redg_lines_{n}"] = {"r3": want, "got": c and c["cycles"], "ok": ok}
            ctrl_ok = ctrl_ok and ok
        r4["control"]["ok"] = ctrl_ok
        for op in R2_OPS:
            r4["lines"][op] = {str(R3_LINES[sb]): cell16(op, 1, sb) for sb in R3_S_KNOTS}
            r4["same_word"][op] = {str(l): cell16(op, l, 4) for l in L_KNOTS}
            knots = r4["lines"][op]
            full = all(knots[str(n)] and knots[str(n)]["verdict"] == "measured"
                       for n in (1, 2, 4, 8, 16, 32))
            r4["install"][op] = ({str(n): knots[str(n)]["cycles"] for n in (1, 2, 4, 8, 16, 32)}
                                 if ctrl_ok and full and op != "redg" else None)
        r4["install"]["axis"] = "cache_line_count"
        r4["install"]["op_class_from_cell"] = {"ATOM": "atom", "ATOMG": "atomg"}
        res["r4"] = r4
        # ---- r5 addendum: saturated-throughput derivation on the W = 16 cells
        def r5_cell(op, l, sb):
            vals = []
            for f in (ta, tb):
                e = f.get((op, l, R4_W, sb))
                if not e:
                    return None
                med = dict(zip(e["us"], e["medians"]))
                if not all(u in med for u in R5_US):
                    return None
                per = [med[u] / u for u in R5_US]
                vals.append({"cycles": mean(per) / R4_W,
                             "flat": abs(per[1] - per[0]) / mean(per),
                             "t_over_u": per})
            agree = rel(vals[0]["cycles"], vals[1]["cycles"])
            flat = max(v["flat"] for v in vals)
            return {"cycles": mean([v["cycles"] for v in vals]), "flat": flat,
                    "agree": agree,
                    "verdict": "measured" if flat <= R5_FLAT and agree <= RUN_AGREE
                               else "experimental",
                    "t_over_u": [v["t_over_u"] for v in vals]}
        r5 = {"lines": {}, "same_word": {}, "control": {}, "open_items": [], "install": {}}
        ctrl_ok = bool(res.get("r2", {}).get("control", {}).get("ok", False))
        r5["control"]["r1_same_address"] = res.get("r2", {}).get("control", {}).get("ok")
        for sb in R3_S_KNOTS:
            n = R3_LINES[sb]
            c = r5_cell("redg", 1, sb)
            want = R3_REDG_INSTALL[a.device][n]
            ok = c is not None and rel(c["cycles"], want) <= R4_CONTROL
            r5["control"][f"redg_lines_{n}"] = {"r3": want, "got": c and c["cycles"], "ok": ok}
            ctrl_ok = ctrl_ok and ok
        r5["control"]["ok"] = ctrl_ok
        for op in R2_OPS:
            r5["lines"][op] = {str(R3_LINES[sb]): r5_cell(op, 1, sb) for sb in R3_S_KNOTS}
            r5["same_word"][op] = {str(l): r5_cell(op, l, 4) for l in L_KNOTS}
            sw = {l: c["cycles"] for l, c in r5["same_word"][op].items() if c}
            if "1" in sw and len(sw) > 1:
                spread = (max(sw.values()) - min(sw.values())) / sw["1"]
                r5["same_word"][op]["spread"] = spread
                if spread > R3_SAME_WORD_SPREAD:
                    r5["open_items"].append(
                        f"{op}: one-line cycles over same-word lanes L spread {spread:.2f} "
                        f"({ {k: round(v, 1) for k, v in sw.items()} }); leaf stays on cache_line_count")
            knots = r5["lines"][op]
            full = all(knots[str(n)] and knots[str(n)]["verdict"] == "measured"
                       for n in (1, 2, 4, 8, 16, 32))
            r5["install"][op] = ({str(n): knots[str(n)]["cycles"] for n in (1, 2, 4, 8, 16, 32)}
                                 if ctrl_ok and full and op != "redg" else None)
        r5["install"]["axis"] = "cache_line_count"
        r5["install"]["op_class_from_cell"] = {"ATOM": "atom", "ATOMG": "atomg"}
        res["r5"] = r5
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({op: {l: res["cells"][op][str(l)].get("cycles") for l in L_KNOTS} for op in OPS},
                     indent=1, default=str))
    if "r2" in res:
        print(json.dumps({"r2_control": res["r2"]["control"],
                          "r2_install": res["r2"]["install"]["cycles"],
                          "r2_verdicts": {op: {l: res["r2"]["cells"][op][str(l)].get("verdict")
                                               for l in L_KNOTS} for op in R2_OPS}},
                         indent=1, default=str))
    if "r3" in res:
        print(json.dumps({"r3_control": res["r3"]["control"],
                          "r3_install": {op: res["r3"]["install"][op] for op in R2_OPS},
                          "r3_lines_verdicts": {op: {n: (c and c["verdict"]) for n, c in res["r3"]["lines"][op].items()}
                                                for op in R2_OPS},
                          "r3_open_items": res["r3"]["open_items"]},
                         indent=1, default=str))
    if "r5" in res:
        print(json.dumps({"r5_control": res["r5"]["control"],
                          "r5_install": {op: res["r5"]["install"][op] for op in R2_OPS},
                          "r5_lines": {op: {n: c and (round(c["cycles"], 2), c["verdict"], round(c["flat"], 3), round(c["agree"], 4))
                                            for n, c in res["r5"]["lines"][op].items()} for op in R2_OPS},
                          "r5_open_items": res["r5"]["open_items"]},
                         indent=1, default=str))
    if "r4" in res:
        print(json.dumps({"r4_control": res["r4"]["control"],
                          "r4_install": {op: res["r4"]["install"][op] for op in R2_OPS},
                          "r4_lines": {op: {n: c and (round(c["cycles"], 2), c["verdict"], round(c["steady"], 3))
                                            for n, c in res["r4"]["lines"][op].items()} for op in R2_OPS}},
                         indent=1, default=str))


if __name__ == "__main__":
    main()
