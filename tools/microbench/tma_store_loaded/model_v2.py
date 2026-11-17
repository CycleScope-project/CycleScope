#!/usr/bin/env python3
"""store_round derivation (protocol-v2.json).

  model_v2.py --run <dir with run-a.jsonl run-b.jsonl> --leaf-2k <c> --leaf-8k <c> --leaf-16k <c> \
              --loader-ref <B/cycle> --out <json>

Per cell: period (STAMPS 0 and 1), per-warp mean timeline (offsets of the
stamps A..F from warp 0's A), and from warp 0: fence = B - A, bar1 = C - B,
hold = D - C (UTMASTG + commit), wait = E - D, bar2 = F - E; completion =
E - C at STAGES 1.
"""
import argparse, json, statistics
from pathlib import Path


def load(path):
    rows = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[d["family"]] = d
    return rows


def med(xs):
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--leaf-2k", type=float, required=True)
    ap.add_argument("--leaf-8k", type=float, required=True)
    ap.add_argument("--leaf-16k", type=float, required=True)
    ap.add_argument("--loader-ref", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {r: load(Path(a.run) / f"run-{r}.jsonl") for r in ("a", "b")}
    fams = sorted(set(runs["a"]) | set(runs["b"]))
    cells = {}
    for f in fams:
        cs = [runs[r][f] for r in ("a", "b") if f in runs[r]]
        c = {k: cs[0][k] for k in ("mode", "row_bytes", "swizzle", "promo", "stride_bytes", "box_bytes", "writers",
                                   "ctas", "loaders", "stages", "stamps", "epi")}
        c["family"] = f
        c["period"] = med([x["period_median"] for x in cs])
        c["load_bpc"] = med([x["load_bytes_per_cycle_median"] for x in cs])
        c["spread"] = max(((x["period_p90"] - x["period_p10"]) / x["period_median"]) if x["period_median"] else 0.0
                          for x in cs)
        c["run_delta"] = (abs(cs[0]["period_median"] - cs[1]["period_median"]) /
                          max(cs[0]["period_median"], cs[1]["period_median"], 1e-9)) if len(cs) == 2 else None
        c["timeline"] = [[med([x["timeline"][w][k] for x in cs]) for k in range(6)] for w in range(4)]
        t0 = c["timeline"][0]
        c["steps"] = {"fence": t0[1] - t0[0], "bar1": t0[2] - t0[1], "hold": t0[3] - t0[2], "wait": t0[4] - t0[3],
                      "bar2": t0[5] - t0[4]}
        c["errors"] = [x["error"] for x in cs if x.get("error")]
        cells[f] = c
    ck = {}
    leaf = {2048: a.leaf_2k, 8192: a.leaf_8k, 16384: a.leaf_16k}
    ck["control_completion"] = {}
    for bb, lv in leaf.items():
        c = cells.get(f"store_round.store.r128s128p128w128.b{bb}.W4.N1.L0.D1.s1")
        if c:
            m = c["steps"]["hold"] + c["steps"]["wait"]
            ck["control_completion"][bb] = {"measured": m, "leaf": lv, "ratio": m / lv, "ok": abs(m - lv) <= 0.2 * lv}
    c = cells.get("store_round.store.r128s128p128w128.b8192.W4.N1.L4.D4.s0.epi0")
    if c:
        ck["control_loader"] = {"measured": c["load_bpc"], "ref": a.loader_ref, "ratio": c["load_bpc"] / a.loader_ref,
                                "ok": abs(c["load_bpc"] - a.loader_ref) <= 0.15 * a.loader_ref}
    ck["controls_ok"] = bool(ck["control_completion"]) and all(v["ok"] for v in ck["control_completion"].values()) \
        and bool(ck.get("control_loader", {}).get("ok"))
    ck["stamp_ratio"] = {f: c["period"] / cells[f[:-1] + "0"]["period"] for f, c in cells.items()
                         if c["stamps"] == 1 and c["epi"] and f[:-1] + "0" in cells and cells[f[:-1] + "0"]["period"]}
    ck["runs_agree"] = all(c["run_delta"] is None or c["run_delta"] <= 0.05 for c in cells.values())
    ck["errors"] = [c["family"] for c in cells.values() if c["errors"]]
    Path(a.out).write_text(json.dumps({"run": a.run, "inputs": vars(a), "cells": list(cells.values()), "checks": ck},
                                      indent=1))
    print(json.dumps({k: v for k, v in ck.items() if k != "stamp_ratio"}, indent=1))
    print("%-62s %7s %6s | %6s %6s %6s %6s %6s | warp B,C,F offsets (w1 w2 w3)" %
          ("family", "period", "load", "fence", "bar1", "hold", "wait", "bar2"))
    for f in fams:
        c = cells[f]
        s = c["steps"]
        tl = c["timeline"]
        o = " ".join("%d:%.0f/%.0f/%.0f/%.0f" % (w, tl[w][0], tl[w][1], tl[w][2], tl[w][5]) for w in (1, 2, 3)) \
            if c["stamps"] else ""
        print("%-62s %7.1f %6.1f | %6.1f %6.1f %6.1f %6.1f %6.1f | %s" % (
            f.replace("store_round.", ""), c["period"], c["load_bpc"], s["fence"], s["bar1"], s["hold"], s["wait"],
            s["bar2"], o))


if __name__ == "__main__":
    main()
