#!/usr/bin/env python3
"""memlat_loaded derivation (protocol.json): per level, base = N=0 cell,
extra(u) = median cycles/hop - base, u = loader B/ns / installed shared rate.

  model.py --run <dir with run-a.jsonl run-b.jsonl> --tree <configs/trees/<dev>.json or
           mechanism_tree/<arch>/leaves dir> --memlat l2=<cycles>,dram=<cycles> --out <json>
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


def spread(xs):
    xs = sorted(xs)
    if not xs:
        return None
    med = statistics.median(xs)
    p10 = xs[int(0.10 * (len(xs) - 1))]
    p90 = xs[int(0.90 * (len(xs) - 1))]
    return (p90 - p10) / med if med else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rates", required=True,
                    help="dram=<B/ns>,l2=<B/ns> installed shared_service rates")
    ap.add_argument("--memlat", required=True, help="dram=<cycles>,l2=<cycles>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rates = {k: float(v) for k, v in (x.split("=") for x in a.rates.split(","))}
    memlat = {k: float(v) for k, v in (x.split("=") for x in a.memlat.split(","))}
    runs = {r: load(Path(a.run) / f"run-{r}.jsonl") for r in ("a", "b")}
    out = {"run": a.run, "levels": {}}
    for level in ("dram", "l2"):
        lv = {"cells": [], "checks": {}, "points": []}
        fails = []
        base = {}
        for r, rows in runs.items():
            cell = rows.get(f"memlat_loaded.{level}.N0.W16")
            if cell is None:
                fails.append(f"run {r}: no N0 cell")
                continue
            base[r] = cell["median"]
        if fails:
            lv["checks"]["fail"] = fails
            out["levels"][level] = lv
            continue
        b = statistics.median(base.values())
        lv["base_cycles"] = b
        lv["checks"]["memlat_reproduced"] = abs(b - memlat[level]) / memlat[level] <= 0.05
        lv["checks"]["memlat_installed"] = memlat[level]
        pts = {}
        for r, rows in runs.items():
            for fam, d in rows.items():
                if not fam.startswith(f"memlat_loaded.{level}."):
                    continue
                med = d["median"]
                bw = d.get("loader_bytes_per_ns") or []
                bwm = statistics.median(bw) if bw else 0.0
                u = bwm / rates[level]
                cell = {"run": r, "family": fam, "median_cycles": med,
                        "extra": med - base[r], "u": u, "bw_bns": bwm,
                        "spread_cycles": spread(d.get("samples", [])),
                        "spread_bw": spread(bw) if bw else None,
                        "error": d.get("error")}
                cell["ok_spread"] = (cell["spread_cycles"] is not None and cell["spread_cycles"] < 0.10
                                     and (cell["spread_bw"] is None or cell["spread_bw"] < 0.10 or u == 0))
                lv["cells"].append(cell)
                pts.setdefault(fam, {})[r] = cell
        # run agreement + points
        agree = True
        for fam, rr in pts.items():
            if "a" in rr and "b" in rr:
                ea, eb = rr["a"]["extra"], rr["b"]["extra"]
                tol = max(0.10 * max(abs(ea), abs(eb)), 0.05 * b)
                if abs(ea - eb) > tol:
                    agree = False
                    lv["checks"].setdefault("disagree", []).append(fam)
        lv["checks"]["runs_agree"] = agree
        lv["checks"]["all_spread_ok"] = all(c["ok_spread"] for c in lv["cells"])
        merged = []
        for fam, rr in pts.items():
            cs = list(rr.values())
            u = statistics.median(c["u"] for c in cs)
            e = statistics.median(c["extra"] for c in cs)
            merged.append((u, max(0.0, e) if abs(e) < 0.05 * b else e, fam))
        merged.sort()
        points = [[0.0, 0.0]] + [[round(u, 4), round(e, 1)] for u, e, _ in merged if u > 0]
        mono = all(points[i + 1][1] >= points[i][1] - 0.05 * b for i in range(len(points) - 1))
        lv["checks"]["monotone"] = mono
        lv["points"] = points
        lv["families_sorted"] = [f for _, _, f in merged]
        ok = lv["checks"]["memlat_reproduced"] and lv["checks"]["all_spread_ok"]
        lv["evidence"] = ("measured" if ok and mono and agree else
                          "experimental" if ok else "not_installed")
        out["levels"][level] = lv
    Path(a.out).write_text(json.dumps(out, indent=1))
    for level, lv in out["levels"].items():
        print(level, lv.get("evidence"), lv.get("checks"), lv.get("points"))


if __name__ == "__main__":
    main()
