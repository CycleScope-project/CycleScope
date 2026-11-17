#!/usr/bin/env python3
"""ldgsts_loaded derivation (protocol.json): per-SM TMA delivery cap (N=1, S=8),
aggregate TMA delivery (N=SMs, S=8), latency table vs aggregate utilization.

  model.py --run <dir with run-a.jsonl run-b.jsonl> --l2-port <bytes/cycle> \
           --l2-rate <bytes/ns> --ldgsts-unloaded <cycles for the 16 KiB hot tile> --out <json>
"""
import argparse, json, statistics
from pathlib import Path


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[d["family"]] = d
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--l2-port", type=float, required=True)
    ap.add_argument("--l2-rate", type=float, required=True)
    ap.add_argument("--ldgsts-unloaded", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {r: load(Path(a.run) / f"run-{r}.jsonl") for r in ("a", "b")}
    fams = sorted(set(runs["a"]) | set(runs["b"]),
                  key=lambda f: (int(f.split(".N")[1].split(".")[0]), int(f.split(".W")[1])))
    out = {"run": a.run, "cells": [], "checks": {}}
    nmax = max(int(f.split(".N")[1].split(".")[0]) for f in fams)
    agree = True
    for f in fams:
        cs = [runs[r][f] for r in ("a", "b") if f in runs[r]]
        N = int(f.split(".N")[1].split(".")[0]); S = int(f.split(".W")[1])
        bpc = statistics.median(c["median"] for c in cs)
        agg = statistics.median(c["aggregate_bytes_per_ns_median"] for c in cs)
        p50 = statistics.median(c["latency_p50"] for c in cs)
        p90 = statistics.median(c["latency_p90"] for c in cs)
        spread = max((c["p90"] - c["p10"]) / c["median"] if c["median"] else 9 for c in cs)
        if len(cs) == 2:
            aa, ab = (c["aggregate_bytes_per_ns_median"] for c in cs)
            if abs(aa - ab) > 0.05 * max(aa, ab):
                agree = False
                out["checks"].setdefault("disagree", []).append(f)
        out["cells"].append({"family": f, "N": N, "S": S, "bpc_per_cta": bpc, "aggregate_bns": agg,
                             "u_l2": agg / a.l2_rate, "lat_p50": p50, "lat_p90": p90,
                             "spread_bpc": spread, "spread_ok": (N < 16 or spread < 0.10),
                             "error": [c.get("error") for c in cs if c.get("error")]})
    cell = {(c["N"], c["S"]): c for c in out["cells"]}
    c1 = cell.get((1, 8)); c4 = cell.get((1, 4)); cmax = cell.get((nmax, 8)); c11 = cell.get((1, 1))
    out["checks"]["runs_agree"] = agree
    out["checks"]["all_spread_ok"] = all(c["spread_ok"] for c in out["cells"])
    if c11:
        out["checks"]["unloaded_latency_vs_leaf"] = c11["lat_p50"] / a.ldgsts_unloaded
        out["checks"]["unloaded_ok"] = abs(c11["lat_p50"] - a.ldgsts_unloaded) <= 0.20 * a.ldgsts_unloaded
    if c1 and c4:
        out["per_sm_cap_bpc"] = c1["bpc_per_cta"]
        out["per_sm_saturated"] = abs(c1["bpc_per_cta"] - c4["bpc_per_cta"]) <= 0.05 * c1["bpc_per_cta"]
        out["rule_A_applies"] = c1["bpc_per_cta"] < 0.9 * a.l2_port
    if cmax:
        out["aggregate_cap_bns"] = cmax["aggregate_bns"]
        out["aggregate_per_sm_bpc"] = cmax["bpc_per_cta"]
        out["rule_B_applies"] = cmax["aggregate_bns"] < 0.9 * a.l2_rate
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))
    for c in out["cells"]:
        print(f"N={c['N']:3d} S={c['S']} bpc/cta={c['bpc_per_cta']:7.2f} agg={c['aggregate_bns']:8.1f} B/ns u={c['u_l2']:.2f} lat p50={c['lat_p50']:6.0f} p90={c['lat_p90']:6.0f} spread={c['spread_bpc']:.3f}")


if __name__ == "__main__":
    main()
