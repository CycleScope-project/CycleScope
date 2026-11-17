#!/usr/bin/env python3
"""table_v2.py <run dir>: per cell run a/b medians of u=0 (t1-t0) per N,
least-squares slope over N and slope/(D+1) = cycles per store.
List only."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
cells = {}
for run in ("a", "b"):
    f = root / f"run-{run}" / "store_hold.jsonl"
    for line in f.read_text().splitlines():
        r = json.loads(line)
        v = r["variant"]
        # loop.<X>.<G>.d<D>.n<N>
        parts = v.split(".")
        n = int(parts[-1][1:])
        key = ".".join(parts[:-1])
        cells.setdefault(key, {}).setdefault(
            n, {})[run] = statistics.median(r["samples"])


def slope(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if den else float("nan")


print("cell                    n4      n8      n16     n24   slope  per_store")
for key in sorted(cells):
    d = int(key.rsplit(".d", 1)[1])
    byn = cells[key]
    ns = sorted(byn)
    meds = {n: statistics.median(byn[n].values()) for n in ns}
    s = slope(ns, [meds[n] for n in ns])
    print("%-22s %7.0f %7.0f %7.0f %7.0f %7.2f %9.2f" % (
        key, meds.get(4, float("nan")), meds.get(8, float("nan")),
        meds.get(16, float("nan")), meds.get(24, float("nan")),
        s, s / (d + 1)))
    for run in ("a", "b"):
        print("   run %s: " % run + "  ".join(
            "n%d=%.0f" % (n, byn[n][run]) for n in ns))
