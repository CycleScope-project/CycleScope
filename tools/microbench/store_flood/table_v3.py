#!/usr/bin/env python3
"""table_v3.py <run dir>: store_flood protocol v3 table (list only).
Per cell (variant = geometry.dest.xW): early gap (groups 2..3), steady
group gap (median of groups 33..48 p50s), saturated (v1 rule),
T = (steady - control) / 8 / W.  Control = the none.*.x4 cell.
Also prints the geometry self-check (controls.geometry): sectors, lines,
half-written sectors and lane-order line runs per request, recomputed
from the cell definitions."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])


def cell_median(samples, extra):
    q = extra.get("q")
    if q:
        return q["p50"]
    return statistics.median(samples)


def lane_off(geo, lane, k):
    """Byte offset of (lane, request k) relative to the warp region."""
    if geo == "lace":
        return k * 16384 + (lane % 4) * 4096 + 2 * (lane // 4)
    if geo in ("epi", "epi_seq"):
        b, s, h, p = (k // 4) % 4, k % 2, (k // 2) % 2, k // 16
        base = (8 * b + s) * 8192 + 32 * p + 16 * h
        if geo == "epi":
            return base + (lane % 4) * 16384 + 2 * (lane // 4)
        return base + (lane // 8) * 16384 + 2 * (lane % 8)
    return k * 16384 + (lane // 8) * 4096 + 2 * (lane % 8)   # row4


def geo_stats(geo, k):
    offs = [lane_off(geo, l, k) for l in range(32)]
    sectors = {o // 32 for o in offs}
    lines = {o // 128 for o in offs}
    half = {s for s in sectors
            if not all(b in {o // 1 for o in offs}
                       for b in (32 * s, 32 * s + 16))}
    # half-written sector: only one of its two 16 B halves touched
    half = set()
    for s in sectors:
        touched = {o - 32 * s for o in offs if o // 32 == s}
        if not ({i for i in range(0, 16, 2)} <= touched and
                {i for i in range(16, 32, 2)} <= touched):
            half.add(s)
    lane_lines = [o // 128 for o in offs]
    runs = 1 + sum(1 for a, b in zip(lane_lines, lane_lines[1:]) if a != b)
    return len(sectors), len(lines), len(half), runs


print("== geometry self-check (recomputed from cell definitions)")
print("geo sectors lines half_written_sectors lane_line_runs")
for geo in ("row4", "lace", "epi", "epi_seq"):
    vals = {geo_stats(geo, k) for k in range(384)}
    if len(vals) == 1:
        s_, l_, h_, r_ = next(iter(vals))
        print(f"{geo:8s} {s_:3d} {l_:3d} {h_:3d} {r_:3d}")
    else:
        print(f"{geo:8s} varies: {sorted(vals)[:4]} ...")

for run in ("a", "b"):
    cells = {}
    f = root / f"run-{run}" / "store_flood.jsonl"
    for line in f.read_text().splitlines():
        r = json.loads(line)
        assert "error" not in r, r.get("error")
        c = cells.setdefault(r["variant"], {"params": r["params"],
                                            "gap": {}, "extra": {}})
        c["gap"][r["u"]] = cell_median(r["samples"], r)
        c["extra"] = r
    ctl = statistics.median(
        c["gap"][g] for name, c in cells.items()
        if c["params"]["geometry"] == "none" for g in range(2, 49))
    print(f"== run {run}: control {ctl}")
    print("cell geo dest W parts early steady sat T/req discarded")
    for name, c in sorted(cells.items()):
        p = c["params"]
        g = c["gap"]
        if p["geometry"] == "none":
            continue
        early = statistics.median(g[k] for k in (2, 3))
        s1 = statistics.median(g[k] for k in range(33, 41))
        s2 = statistics.median(g[k] for k in range(41, 49))
        steady = statistics.median(g[k] for k in range(33, 49))
        sat = steady > 1.5 * early and abs(s1 - s2) <= 0.10 * max(s1, s2)
        W = p["warps"]
        T = (steady - ctl) / 8 / W
        print("%-24s %-8s %-6s %d %3d %7.1f %7.1f %-5s %7.2f %d" % (
            name, p["geometry"], p.get("dest", "-"), W, p["participants"],
            early, steady, str(sat), T,
            c["extra"].get("discarded_launches", 0)))
