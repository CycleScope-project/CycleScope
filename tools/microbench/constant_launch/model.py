#!/usr/bin/env python3
"""constant_launch derivation (tools/microbench/constant_launch/protocol.json).

Per launch (launches 0 and 1 of every variant dropped):
    first = t1 - t0, second = t2 - t1
Per (BANK, S) and run: OLS over U of the per-U launch means -> slope
(per-load latency) and intercept, for the first and the second traversal;
residual = max |mean - fit| / slope.
Device value F = mean over the two runs of the bank-0 S = 256 first slope.
Acceptance: run agreement <= 5 % on every slope, residual <= 5 % of the
slope -> measured; else experimental (not installed).

Usage: model.py --run-a <dir> --run-b <dir> --checks <checks.json>
                --tree configs/trees/<dev>.json --out model.json
"""
import argparse, json, re, statistics
from pathlib import Path

LAUNCH_RE = re.compile(r"^LAUNCH k_cl variant=cl\.b(\d+)\.s(\d+)\.u(\d+) launch=(\d+) "
                       r"t0=(\d+) t1=(\d+) t2=(\d+) idx=(\d+)")
DROP = {0, 1}
RUN_AGREE = 0.05
RESID = 0.05


def per_variant(run_dir, gated):
    acc = {}
    for line in (Path(run_dir) / "launches.log").read_text().splitlines():
        m = LAUNCH_RE.match(line.strip())
        if not m:
            continue
        b, s, u, launch, t0, t1, t2, _idx = map(int, m.groups())
        if launch in DROP or (gated is not None and f"cl.b{b}.s{s}.u{u}" not in gated):
            continue
        acc.setdefault((b, s, u), {"first": [], "second": []})
        acc[(b, s, u)]["first"].append(t1 - t0)
        acc[(b, s, u)]["second"].append(t2 - t1)
    return {k: {t: statistics.fmean(v[t]) for t in v} | {"n": len(v["first"])} for k, v in acc.items()}


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icpt = my - slope * mx
    resid = max(abs(y - (icpt + slope * x)) for x, y in zip(xs, ys))
    return slope, icpt, resid


def fits(cells):
    out = {}
    for b, s in sorted({(b, s) for b, s, _ in cells}):
        us = sorted(u for bb, ss, u in cells if (bb, ss) == (b, s))
        if len(us) < 3:
            out[f"b{b}.s{s}"] = {"missing_u": us}
            continue
        rec = {"u": us}
        for t in ("first", "second"):
            slope, icpt, resid = ols(us, [cells[(b, s, u)][t] for u in us])
            rec[t] = {"slope": slope, "intercept": icpt, "max_resid": resid,
                      "resid_rel": resid / slope if slope > 0 else None,
                      "means": {str(u): cells[(b, s, u)][t] for u in us}}
        out[f"b{b}.s{s}"] = rec
    return out


def constant_leaf(tree_path):
    t = json.loads(Path(tree_path).read_text())
    for n in t["nodes"]:
        if n["id"] == "memory.constant_cache.root":
            f = n["parameter"]["fields"]
            return {k: f[k]["model"]["value"] for k in f if "value" in f[k]["model"]}
    raise SystemExit("memory.constant_cache.root not in tree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--checks", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    checks = json.loads(Path(a.checks).read_text())
    gated = {v for v, c in checks.get("variants", {}).items() if c.get("ok")}
    runs = {"a": fits(per_variant(a.run_a, gated)), "b": fits(per_variant(a.run_b, gated))}
    leaf = constant_leaf(a.tree)
    out = {"protocol": "tools/microbench/constant_launch/protocol.json", "runs": {"a": a.run_a, "b": a.run_b},
           "gated_variants": sorted(gated), "installed_leaf": leaf, "fits": runs, "cells": {}}
    for key in sorted(set(runs["a"]) | set(runs["b"])):
        ra, rb = runs["a"].get(key, {}), runs["b"].get(key, {})
        cell = {}
        for t in ("first", "second"):
            if t not in ra or t not in rb:
                cell[t] = {"verdict": "unrealised", "reason": "fit missing in a run"}
                continue
            sa, sb = ra[t]["slope"], rb[t]["slope"]
            m = (sa + sb) / 2
            agree = abs(sa - sb) / m if m > 0 else None
            rmax = max(ra[t]["resid_rel"] or 1e9, rb[t]["resid_rel"] or 1e9)
            ok = agree is not None and agree <= RUN_AGREE and rmax <= RESID
            cell[t] = {"slope": m, "run_agree": agree, "resid_rel_max": rmax,
                       "intercept": (ra[t]["intercept"] + rb[t]["intercept"]) / 2,
                       "verdict": "measured" if ok else "experimental"}
        out["cells"][key] = cell
    f = out["cells"].get("b0.s256", {}).get("first", {})
    out["install"] = ({"bank0_first_access_cycles": round(f["slope"], 2), "verdict": f["verdict"]}
                      if f.get("verdict") == "measured" else
                      {"bank0_first_access_cycles": None, "verdict": f.get("verdict", "unrealised"),
                       "reported_slope": f.get("slope")})
    b3 = out["cells"].get("b3.s256", {}).get("first", {})
    if b3.get("slope") and leaf.get("miss_latency_cycles"):
        out["bank3_vs_installed_miss"] = {"bank3_first": b3["slope"], "miss": leaf["miss_latency_cycles"],
                                          "rel": b3["slope"] / leaf["miss_latency_cycles"] - 1}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for key, cell in out["cells"].items():
        print(key, {t: (round(c.get("slope", 0) or 0, 2), c.get("verdict")) for t, c in cell.items()})
    print("install", out["install"], "bank3 vs miss", out.get("bank3_vs_installed_miss"))


if __name__ == "__main__":
    main()
