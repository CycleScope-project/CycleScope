#!/usr/bin/env python3
"""table_bridge.py <run dir>: store_flood bridge cells table (list only).
Per cell and run: total window cycles (u=0 median), steady group gap
(groups 33..48 median for 48-group cells, all groups for 16-group
cells), T = (steady - control) / 8 / W, and the per-group median gap
sequence."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])

ctl = {}
rows = {}
for run in ("a", "b"):
    f = root / f"run-{run}" / "store_flood.jsonl"
    if not f.exists():
        continue
    for line in f.read_text().splitlines():
        r = json.loads(line)
        v = r["variant"]
        cell = rows.setdefault((run, v), {"gap": {}, "u0": []})
        if r["u"] == 0:
            cell["u0"] = r["samples"]
        else:
            cell["gap"][r["u"]] = statistics.median(r["samples"])
    # control = none cell's per-group median (like v3's none.x4)
for run in ("a", "b"):
    print(f"== run {run}")
    print("cell total_med steady T/req gaps")
    for (rn, v), c in sorted(rows.items()):
        if rn != run:
            continue
        g = c["gap"]
        ng = max(g) if g else 0
        steady = (statistics.median(g[k] for k in range(33, 49))
                  if ng == 48 else
                  statistics.median(g[k] for k in range(1, ng + 1))
                  if ng else float("nan"))
        tot = statistics.median(c["u0"]) if c["u0"] else float("nan")
        T = steady / 8 / 4
        seq = " ".join(f"{g[k]:.0f}" for k in sorted(g))
        print("%-16s %10.0f %7.1f %6.2f %s" % (v, tot, steady, T, seq))
