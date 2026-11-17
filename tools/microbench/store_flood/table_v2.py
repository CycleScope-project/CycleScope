#!/usr/bin/env python3
"""table_v2.py <run dir>: store_flood protocol v2 table (list only).
Per cell (variant = geometry.set.xW): steady group gap (median of groups
33..48 p50s), T = (steady - control) / 8 / W, early gap (groups 2..3),
saturated (v1 rule), participants, cross-SM steady p10 / p90.
Controls: none.sm1.x4 for single-SM cells, none.<set>.x4 when present else
none.ekall.x4 for multi-SM cells."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])


def cell_median(samples, extra):
    q = extra.get("q")
    if q:
        return q["p50"]
    return statistics.median(samples)


res = {}
for run in ("a", "b"):
    cells = {}
    f = root / f"run-{run}" / "store_flood.jsonl"
    for line in f.read_text().splitlines():
        r = json.loads(line)
        assert "error" not in r, r.get("error")
        c = cells.setdefault(r["variant"], {
            "params": r["params"], "gap": {}, "extra": {}})
        g = r["u"]
        med = cell_median(r["samples"], r)
        c["gap"][g] = med
        c["extra"] = r
    ctl = {}
    for name, c in cells.items():
        if c["params"]["geometry"] == "none":
            ctl[c["params"]["set"]] = statistics.median(
                c["gap"][g] for g in range(2, 49))
    print(f"== run {run}: controls {ctl}")
    print("cell set geo W parts early steady sat T/req sm_p10 sm_p90 discarded")
    for name, c in sorted(cells.items()):
        p = c["params"]; g = c["gap"]
        if p["geometry"] == "none":
            continue
        setname = p["set"]
        over = ctl.get(setname, ctl.get("ekall", ctl.get("sm1", 0.0)))
        early = statistics.median(g[k] for k in (2, 3))
        s1 = statistics.median(g[k] for k in range(33, 41))
        s2 = statistics.median(g[k] for k in range(41, 49))
        steady = statistics.median(g[k] for k in range(33, 49))
        sat = steady > 1.5 * early and abs(s1 - s2) <= 0.10 * max(s1, s2)
        W = p["warps"]
        T = (steady - over) / 8 / W
        sm = c["extra"].get("sm_steady", {})
        print("%-26s %-6s %-11s %d %3d %7.1f %7.1f %-5s %7.2f %6.1f %6.1f %d" % (
            name, setname, p["geometry"], W, p["participants"],
            early, steady, str(sat), T,
            sm.get("p10", 0.0), sm.get("p90", 0.0),
            c["extra"].get("discarded_launches", 0)))
