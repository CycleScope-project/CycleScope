#!/usr/bin/env python3
"""table.py <run dir>: per cell run a/b medians of u=0 (ts-t0) and u=1
(t1-ts); hold = u1 median - same-(G,K,W) control u1 median.  List only."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
for run in ("a", "b"):
    f = root / f"run-{run}" / "store_hold.jsonl"
    rows = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["variant"], {})[r["u"]] = \
            statistics.median(r["samples"])
    print(f"== run {run}")
    print("cell t0_med x_med hold")
    for v in sorted(rows):
        x, rest = v.split(".", 1)
        ctl = rows.get("none." + rest, {})
        u1 = rows[v].get(1, float("nan"))
        c1 = ctl.get(1, float("nan"))
        hold = u1 - c1
        print("%-34s %8.0f %8.0f %7.0f" %
              (v, rows[v].get(0, float("nan")), u1, hold))
