#!/usr/bin/env python3
"""table_v5.py <run dir>: v5 smem/siblings table (list only).  Per cell
and run: steady group gap (groups 33..48 median), T = (steady - 2) / 8 / W,
and the actual smem_bytes / participant_sm recorded in params."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
W = 4
for run in ("a", "b"):
    f = root / f"run-{run}" / "store_flood.jsonl"
    rows, info = {}, {}
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows.setdefault(r["variant"], {})[r["u"]] = \
                statistics.median(r["samples"])
            info[r["variant"]] = r.get("params", {})
    print(f"== run {run}")
    print("cell steady T/req smem part_sm")
    for v in sorted(rows):
        g = rows[v]
        if not g:
            continue
        steady = statistics.median(g[k] for k in range(33, 49)
                                   if k in g)
        T = (steady - 2) / 8 / W
        p = info[v]
        print("%-34s %7.1f %6.2f %8s %6s" % (
            v, steady, T, p.get("smem_bytes", "?"),
            p.get("participant_sm", "?")))
