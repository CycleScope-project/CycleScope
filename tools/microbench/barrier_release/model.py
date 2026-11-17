#!/usr/bin/env python3
"""barrier_release derivation (tools/microbench/barrier_release/protocol.json).

Part 1: per-variant median + P10/P90 over the jsonl 'samples' arrays
written by the sync probes (MbWriter) -- the raw cells.
Part 2 (--checks): the protocol's release definition, blackwell-prototype wave1
form:  release = E[t_w - t_s | waiter blocked, last arriver jittered]
                 - o_s - o_w - r_sync
  o_s     stall-count sum (SASS control words) from the t_s clock read to
          the signal op; o_w the same from the try_wait exit to the t_w
          clock read (checks.json from sass_check.py)
  r_sync  per-TRYWAIT iteration cost = slope over U of the tw_only chain
          medians minus the slope of its SASS stall sums
  E[.]    mean of the sync_xw samples at the settled delays (>= 4000),
          try_wait waiter, pooled over the arrive (notx) and complete_tx
          (tx) producers for mbarrier.release; c1 / c2 for cluster.release
  cp_async.arrival = (median noinc - its .issue) - (median arrive - its .issue)
"""
import argparse, json, math, statistics
from pathlib import Path

def q(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, round(p * (len(s) - 1))))]

def ols_slope(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def derive(rows, checks, installed):
    """rows: list of jsonl dicts with 'probe' (file stem) added."""
    res = {}
    kern_mb = checks.get("sync_mb", {}).get("kernels", {})
    kern_xw = checks.get("sync_xw", {}).get("kernels", {})
    # r_sync from the tw_only chain: slope of medians over U minus the
    # slope of the SASS stall sums over U (ok kernels only)
    pts, stalls = [], []
    for r in rows:
        if r["probe"] == "sync_mb" and r.get("variant") == "tw_only_nowait":
            c = kern_mb.get(f"sync_mb/tw_only/{r['u']}")
            if c and c.get("ok"):
                pts.append((r["u"], statistics.median(r["samples"])))
                stalls.append((r["u"], c["stall_sum"]))
    r_sync = None
    if len(pts) >= 2:
        s1 = ols_slope([p[0] for p in pts], [p[1] for p in pts])
        s2 = ols_slope([p[0] for p in stalls], [p[1] for p in stalls])
        r_sync = s1 - s2
        res["r_sync"] = {"value": r_sync, "chain_slope": s1, "stall_slope": s2,
                         "points": pts, "eq": "slope(tw_only chain medians) - slope(SASS stall sums)"}
    else:
        res["r_sync"] = {"value": None, "why": "tw_only chain kernels not ok / missing"}

    def pooled(op, waiter):
        xs = []
        for r in rows:
            if r["probe"] != "sync_xw" or "samples" not in r:
                continue
            p = r.get("params", {})
            if (p.get("op") == op and p.get("waiter") == waiter
                    and p.get("delay", 0) >= 4000 and "peerflag" not in r["variant"]):
                xs += [float(x) for x in r["samples"]]
        if not xs:
            return None
        return {"mean": statistics.fmean(xs), "se": statistics.pstdev(xs) / math.sqrt(len(xs)),
                "n": len(xs), "median": statistics.median(xs)}

    def release(op, waiter="try"):
        k = pooled(op, waiter)
        c = kern_xw.get(f"sync_xw/{op}")
        if not k or not c or not c.get("ok") or r_sync is None:
            return {"value": None, "pooled": k, "check": c and {"ok": c.get("ok"), "why": c.get("why")}}
        o_w_list = c["o_w"] if waiter == "try" else c.get("o_w_spin", [])
        if not o_w_list:
            return {"value": None, "pooled": k, "why": "no o_w"}
        o_w = statistics.median(o_w_list)
        v = k["mean"] - c["o_s"] - o_w - r_sync
        return {"value": v, "se": k["se"], "se_rel": k["se"] / v if v else None, "n": k["n"],
                "o_s": c["o_s"], "o_w": o_w, "r_sync": r_sync, "E": k["mean"],
                "eq": f"E[t_w-t_s | {op}, {waiter}] - o_s - o_w - r_sync"}
    for op in ("arr", "tx", "c1", "c2"):
        res[f"release/{op}"] = release(op)
    arr, tx = res["release/arr"], res["release/tx"]
    if arr.get("value") is not None and tx.get("value") is not None:
        n = arr["n"] + tx["n"]
        res["synchronization.mbarrier.release"] = {
            "value": (arr["value"] * arr["n"] + tx["value"] * tx["n"]) / n,
            "notx": arr["value"], "tx": tx["value"], "n": n,
            "se": math.sqrt(arr["se"] ** 2 + tx["se"] ** 2) / 2,
            "eq": "pooled over arrive and complete_tx (protocol derivation)"}
    for op, name in (("c1", "synchronization.cluster.release"), ("c2", "synchronization.cluster.release.cta_group_2")):
        if res[f"release/{op}"].get("value") is not None:
            res[name] = {"value": res[f"release/{op}"]["value"], "se": res[f"release/{op}"]["se"]}
    # cp_async.arrival from the async probe (medians, issue overheads removed)
    med = {}
    for r in rows:
        if r["probe"] == "async" and "samples" in r:
            med[r["variant"]] = statistics.median([float(x) for x in r["samples"]])
    if all(k in med for k in ("arr.arrive", "arr.arrive.issue", "arr.cpasync_noinc", "arr.cpasync_noinc.issue")):
        res["synchronization.cp_async.arrival"] = {
            "value": (med["arr.cpasync_noinc"] - med["arr.cpasync_noinc.issue"]) - (med["arr.arrive"] - med["arr.arrive.issue"]),
            "eq": "(median noinc - noinc.issue) - (median arrive - arrive.issue)", "inputs": med}
    # reproduction of the installed values (10 % rule)
    for name, ref in installed.items():
        e = res.get(name)
        if e and e.get("value") is not None and ref:
            e["installed"] = ref
            e["reproduces_within_10pct"] = abs(e["value"] - ref) / ref <= 0.10
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", help="checks.json from sass_check.py (enables the release derivation)")
    ap.add_argument("--device", choices=["h100", "b300"])
    a = ap.parse_args()
    cells = {}
    rows = []
    for jl in Path(a.run).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = d.get("variant")
            samples = d.get("samples") or []
            if not v or not samples:
                continue
            d["probe"] = jl.stem
            rows.append(d)
            key = f"{jl.stem}:{v}"
            e = cells.setdefault(key, [])
            e.extend(float(x) for x in samples)
    out = {"run": a.run, "cells": {}}
    if a.checks:
        checks = json.loads(Path(a.checks).read_text())
        installed = ({"synchronization.mbarrier.release": 87.78, "synchronization.cluster.release": 87.0,
                      "synchronization.cp_async.arrival": 12.7} if a.device == "b300"
                     else {"synchronization.mbarrier.release": None})
        out["derived"] = derive(rows, checks, installed)
    for k, s in sorted(cells.items()):
        med = statistics.median(s)
        p10, p90 = q(s, 0.10), q(s, 0.90)
        out["cells"][k] = {"median": med, "p10": p10, "p90": p90,
                           "spread_rel": (p90 - p10) / med if med else None,
                           "n": len(s)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, c in sorted(out["cells"].items()):
        print(f"  {k}: {c['median']:.1f} [{c['p10']:.0f},{c['p90']:.0f}] n={c['n']}")
    for k, e in sorted(out.get("derived", {}).items()):
        if isinstance(e, dict) and e.get("value") is not None:
            print(f"  derived {k}: {e['value']:.2f}" + (f" (se {e['se']:.2f})" if e.get("se") is not None else "")
                  + (f" installed {e['installed']} repro={e['reproduces_within_10pct']}" if "installed" in e else ""))

if __name__ == "__main__":
    main()
