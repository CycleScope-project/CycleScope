#!/usr/bin/env python3
"""lsu_accept protocol v1 (frozen): per-SM LSU acceptance queue?

Rows <variant>.w<W> (params variant/w/d) from
$A/microbench/lsu_accept/<dev>-<run>/run-{a,b}/*.jsonl; the installed
storage.shared.acceptance_sm slope comes from configs/trees/<dev>.json
(node id prefix storage.shared.acceptance_sm.root/operation_class=LDS).
Fit T over D in {1,2,4,8,16} per (variant, W): slope s, intercept c,
residual, run agreement; D = 0 = overhead cell.  Wavefronts per the
protocol: ldgsts16/ldg16/stg16 = 4, ldgsts8 = 2, ldgsts4/ldg4 = 1.
Decision per protocol "decision": baseline iff ldgsts16 W=1 slope
<= 1.5, else Q / Q-shared (mix cell decides) / none.

Usage: model.py <A> <dev> <run> [--tree <clone root>]
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = ("ldgsts16", "ldgsts8", "ldgsts4", "ldg16", "ldg4",
            "stg16", "mix", "lds")
WAVEFRONTS = {"ldgsts16": 4, "ldgsts8": 2, "ldgsts4": 1, "ldg16": 4,
              "ldg4": 1, "stg16": 4, "mix": 4, "lds": 4}
FIT_D = (1, 2, 4, 8, 16)
WS = (1, 2, 4, 8)


def load_rows(run: Path) -> dict:
    cells = defaultdict(list)
    for d in ("run-a", "run-b"):
        for f in (run / d).glob("*.jsonl"):
            for ln in f.open():
                r = json.loads(ln)
                p = r["params"]
                cells[(d, p["variant"], p["w"], p["d"])].extend(
                    r["samples"])
    return cells


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    s = sxy / sxx
    c = my - s * mx
    resid = max(abs(y - (s * x + c)) for x, y in zip(xs, ys))
    denom = max(1e-9, max(ys))
    return s, c, resid / denom * 100


def tree_slope(clone: Path, dev: str) -> float:
    t = json.loads((clone / "configs" / "trees" /
                    f"{dev}.json").read_text())
    stack = [t]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            nid = str(n.get("id", ""))
            if nid.startswith(
                    "storage.shared.acceptance_sm.root/"
                    "operation_class=LDS/remote_smem=0"):
                fm = (n.get("parameter", {}).get("fields", {})
                      .get("cycles", {}).get("model", {}))
                if fm.get("kind") == "affine":
                    return float(fm["slope"])
            stack += list(n.values())
        elif isinstance(n, list):
            stack += n
    raise SystemExit("storage.shared.acceptance_sm LDS slope not found")


def main() -> int:
    a, dev, run = sys.argv[1], sys.argv[2], sys.argv[3]
    v2 = "--v2" in sys.argv
    v3 = "--v3" in sys.argv
    clone = Path(sys.argv[sys.argv.index("--tree") + 1]) \
        if "--tree" in sys.argv else Path.cwd()
    base = Path(a) / "microbench" / "lsu_accept" / f"{dev}-{run}"
    cells = load_rows(base)
    med = {}
    agree = {}
    for (rn, v, w, d), ss in cells.items():
        med[(rn, v, w, d)] = statistics.median(ss)
    fit = {}
    for v in VARIANTS:
        for w in WS:
            for rn in ("run-a", "run-b"):
                xs = [d for d in FIT_D if (rn, v, w, d) in med]
                if len(xs) < 3:
                    continue
                ys = [med[(rn, v, w, d)] for d in xs]
                s, c, r = ols(xs, ys)
                fit[(rn, v, w)] = (s, c, r)
            if all((rn, v, w) in fit for rn in ("run-a", "run-b")):
                sa = fit[("run-a", v, w)][0]
                sb = fit[("run-b", v, w)][0]
                agree[(v, w)] = abs(sa - sb) <= max(
                    1.0, 0.05 * max(sa, sb))
    b_lds = tree_slope(clone, dev)
    out = {"device": dev, "run": run, "cells": {}, "b_lds_leaf": b_lds}
    print(f"{dev}-{run}: installed shared acceptance slope "
          f"b_lds={b_lds:.3f}")
    print(f"{'variant':<10} {'W':>2} {'slope':>8} {'icept':>8} "
          f"{'resid%':>7} {'agree':>6} {'b/wf':>6}")
    for v in VARIANTS:
        for w in WS:
            if (v, w) not in agree:
                continue
            s = sum(fit[(rn, v, w)][0]
                    for rn in ("run-a", "run-b")) / 2
            c = sum(fit[(rn, v, w)][1]
                    for rn in ("run-a", "run-b")) / 2
            r = max(fit[(rn, v, w)][2] for rn in ("run-a", "run-b"))
            b = s / WAVEFRONTS[v]
            ok = r <= 5 and agree[(v, w)]
            out["cells"][f"{v}.w{w}"] = {
                "slope": s, "intercept": c, "resid_pct": r,
                "agree": agree[(v, w)], "b": b, "usable": ok,
                "d0": {rn: med.get((rn, v, w, 0))
                       for rn in ("run-a", "run-b")}}
            print(f"{v:<10} {w:>2} {s:8.3f} {c:8.3f} {r:6.1f}% "
                  f"{'ok' if agree[(v, w)] else 'BAD':>6} {b:6.3f}")
    # decision per protocol
    s1 = out["cells"].get("ldgsts16.w1", {}).get("slope")
    if v3:
        import math as _m
        r_ = sum(out["cells"].get(f"{v}.w1", {}).get("slope", 0)
                 for v in ("ldgsts4", "ldgsts8")) / 2
        cvals = [out["cells"][f"ldgsts16.w{w}"]["slope"] / (w * 4)
                 for w in (2, 4)
                 if out["cells"].get(f"ldgsts16.w{w}", {})
                    .get("usable")]
        c_ = statistics.median(cvals) if cvals else 0.0
        s4 = out["cells"].get("ldg4.w4", {}).get("slope", 0.0)
        c_wb = s4 / 4 if s4 > 1.15 * r_ else 0.0
        q_ = s1 or 0.0
        WF = {"ldgsts16": 4, "ldgsts8": 2, "ldgsts4": 1, "ldg16": 4,
              "ldg4": 1, "stg16": 4, "mix": 4, "lds": 4}
        WFR = {"ldg16": 4, "ldg4": 1}
        print(f"r={r_:.3f} c={c_:.3f} c_wb={c_wb:.3f} q={q_:.3f}")
        print(f"{'cell':<14}{'meas':>8}{'T3':>8}{'T2':>8}")
        res = {"T3": 0, "T2": 0}
        n = 0
        gate_v = {"ldgsts4", "ldgsts8", "ldgsts16", "ldg4", "ldg16"}
        for v in VARIANTS:
            for w in WS:
                key = f"{v}.w{w}"
                if key not in out["cells"]:
                    continue
                m = out["cells"][key]["slope"]
                wf = WF[v]
                t2 = max(r_ * _m.ceil(w / 4), w * wf * c_)
                t3 = t2
                if v in WFR:
                    t3 = max(t2, w * WFR[v] * c_wb)
                print(f"{key:<14}{m:8.2f}{t3:8.2f}{t2:8.2f}"
                      + ("" if (v in gate_v and w <= 4)
                         else "  (record)"))
                if v in gate_v and w <= 4:
                    ok = lambda p: abs(m - p) <= max(1.0, 0.15 * m)
                    res["T3"] += ok(t3)
                    res["T2"] += ok(t2)
                    n += 1
        decision = "none"
        for cand in ("T3", "T2"):
            if res[cand] == n:
                decision = cand
                break
        out["v3"] = {"r": r_, "c": c_, "c_wb": c_wb,
                     "pass": res, "n": n}
        out["decision"] = decision
        out["reason"] = f"{res} of {n} gating cells within 15%/1c"
        mdir = Path(a) / "microbench" / "lsu_accept-model" / f"{dev}-{run}"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "model.json").write_text(json.dumps(out, indent=1))
        print(f"decision {decision}: {out['reason']}")
        return 0
    if v2:
        r_ = sum(out["cells"].get(f"{v}.w1", {}).get("slope", 0)
                 for v in ("ldgsts4", "ldg4")) / 2
        cvals = [out["cells"][f"{v}.w{w}"]["slope"] / (w * 4)
                 for v in ("ldgsts16", "ldg16") for w in (2, 4, 8)
                 if f"{v}.w{w}" in out["cells"]
                 and out["cells"][f"{v}.w{w}"]["usable"]]
        c_ = statistics.median(cvals) if cvals else 0.0
        q_ = s1 or 0.0
        s_lds = out["cells"].get("lds.w1", {}).get("slope", 0.0)
        WF = {"ldgsts16": 4, "ldgsts8": 2, "ldgsts4": 1, "ldg16": 4,
              "ldg4": 1, "stg16": 4, "mix": 4, "lds": 4}
        print(f"r={r_:.3f} c={c_:.3f} q={q_:.3f} s_lds={s_lds:.3f}")
        print(f"{'cell':<14}{'meas':>8}{'T2':>8}{'T2sh':>8}{'Q':>8}")
        res = {"T2": 0, "T2-shared": 0, "Q": 0}
        n = 0
        for v in VARIANTS:
            for w in WS:
                key = f"{v}.w{w}"
                if key not in out["cells"] or v == "stg16":
                    continue
                m = out["cells"][key]["slope"]
                wf = WF[v]
                import math as _m
                t2 = max(r_ * _m.ceil(w / 4), w * wf * c_)
                qq = w * q_
                t2s = t2
                if v == "mix":
                    t2 = max(r_ * _m.ceil(w / 4), w * wf * c_) + s_lds
                    t2s = max(2 * r_, 4 * c_ + 4 * 1.0112)
                ok = lambda p: abs(m - p) <= max(1.0, 0.15 * m)
                for cand, p in (("T2", t2), ("T2-shared", t2s),
                                ("Q", qq)):
                    res[cand] += ok(p)
                n += 1
                print(f"{key:<14}{m:8.2f}{t2:8.2f}{t2s:8.2f}{qq:8.2f}")
        decision = "none"
        for cand in ("baseline", "T2", "T2-shared", "Q"):
            if cand == "baseline":
                continue
            if res[cand] == n:
                decision = cand
                break
        reason = f"{res} of {n} usable cells within 15%/1c"
        out["v2"] = {"r": r_, "c": c_, "q": q_, "pass": res, "n": n}
        out["decision"] = decision
        out["reason"] = reason
        mdir = Path(a) / "microbench" / "lsu_accept-model" / f"{dev}-{run}"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "model.json").write_text(json.dumps(out, indent=1))
        print(f"decision {decision}: {reason}")
        return 0
    decision, reason = "none", ""
    if s1 is not None and s1 <= 1.5:
        decision, reason = "baseline", \
            f"ldgsts16 W=1 slope {s1:.2f} <= 1.5 (port model suffices)"
    else:
        s_ldg16 = out["cells"].get("ldg16.w1", {}).get("slope")
        s_lds = out["cells"].get("lds.w1", {}).get("slope")
        wscale_ok = all(
            0.8 * w <= out["cells"].get(f"ldgsts16.w{w}", {})
            .get("slope", 1e9) / max(s1, 1e-9) <= 1.2 * w
            for w in (2, 4))
        lds_ok = s_lds is not None and abs(
            s_lds - 4 * b_lds) <= 0.15 * 4 * b_lds
        s_mix = out["cells"].get("mix.w1", {}).get("slope")
        if not lds_ok:
            reason = "lds control off installed slope -> " \
                     "mix decision unusable"
        shared = None
        if lds_ok and s_mix is not None and s1:
            if abs(s_mix - s1) <= 0.15 * s1:
                shared = False
            elif abs(s_mix - (s1 + b_lds * 4)) <= \
                    0.15 * (s1 + b_lds * 4):
                shared = True
        byte_bs = {v: out["cells"][f"{v}.w1"]["b"] for v in
                   ("ldgsts16", "ldgsts8", "ldgsts4")
                   if f"{v}.w1" in out["cells"]}
        bmin, bmax = (min(byte_bs.values()), max(byte_bs.values())) \
            if byte_bs else (0, 0)
        affine = bmax <= 1.15 * bmin if byte_bs else False
        out["byte_b"] = byte_bs
        out["wscale"] = {w: out["cells"].get(f"ldgsts16.w{w}", {})
                         .get("slope", 0) / max(s1 or 1, 1e-9)
                         for w in WS}
        if wscale_ok and (shared is not None):
            decision = "Q-shared" if shared else "Q"
            reason = f"mix slope {s_mix:.2f} vs ldgsts16 {s1:.2f} " \
                     f"(+lds 4*b={s1 + b_lds * 4:.2f}); " \
                     f"{'affine' if affine else 'knots'} b={byte_bs}"
        elif wscale_ok:
            reason = "Q candidate; mix undecidable: " + reason
            decision = "Q"
        else:
            reason = f"W-scaling {out['wscale']} not ~W -> report only"
        if s_ldg16 is not None and s1 and \
                abs(s_ldg16 - s1) > 0.15 * s1:
            reason += ("; ldg16/ldgsts16 disagree "
                       f"({s_ldg16:.2f}/{s1:.2f}) -> partition by akind")
    out["decision"] = decision
    out["reason"] = reason
    mdir = Path(a) / "microbench" / "lsu_accept-model" / f"{dev}-{run}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.json").write_text(json.dumps(out, indent=1))
    print(f"decision {decision}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
