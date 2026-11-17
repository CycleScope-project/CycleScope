#!/usr/bin/env python3
"""table.py <run dir>: median per cell for runs a/b, least-squares
slope over U per (form, cell, w), residual share. List only."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])


def slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sxy / sxx if sxx else 0.0
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    return b, (sum(r * r for r in resid) ** 0.5) / n


for run in ("a", "b"):
    rows = {}
    for line in (root / f"run-{run}.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["variant"]] = statistics.median(r["samples"])
    print(f"== run {run}")
    print("cell med")
    for v in sorted(rows):
        print("%-46s %8.0f" % (v, rows[v]))
    print("== slopes")
    groups = {}
    for v, m in rows.items():
        p = v.split(".")
        key = ".".join(p[1:-1])
        u = int(p[-1][1:])
        groups.setdefault(key, []).append((u, m))
    for key in sorted(groups):
        pts = sorted(groups[key])
        b, res = slope([u for u, _ in pts], [m for _, m in pts])
        share = res / b if b else float("nan")
        print("%-40s slope=%8.2f resid=%.2f share=%.3f" %
              (key, b, res, share))
