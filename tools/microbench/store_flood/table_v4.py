#!/usr/bin/env python3
"""table_v4.py <run dir>: v4 bisection table (list only).  Per cell and
run: steady group gap (groups 33..48 median), T = (steady - 2) / 8 / W,
and the first 6 group gaps."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
W = 4
for run in ("a", "b"):
    f = root / f"run-{run}" / "store_flood.jsonl"
    rows = {}
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows.setdefault(r["variant"], {})[r["u"]] = \
                statistics.median(r["samples"])
    print(f"== run {run}")
    print("cell steady T/req g1 g2 g3 g4 g5 g6")
    for v in sorted(rows):
        g = rows[v]
        if not g:
            continue
        steady = statistics.median(g[k] for k in range(33, 49)
                                   if k in g)
        T = (steady - 2) / 8 / W
        first = " ".join(f"{g[k]:.0f}" if k in g else "-" for k in
                         range(1, 7))
        print("%-18s %7.1f %6.2f %s" % (v, steady, T, first))
