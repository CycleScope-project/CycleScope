#!/usr/bin/env python3
"""table_v2.py <run dir>: v2 cells grouped by the SASS mnemonic the
form actually emitted (build/check.json.sass-map.json); per cell the
run a/b medians and the least-squares slope over U. List only."""
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


smap = json.loads(
    (root / "build/check.json.sass-map.json").read_text())

runs = {}
for run in ("a", "b"):
    rows = {}
    for line in (root / f"run-{run}.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["variant"]] = statistics.median(r["samples"])
    runs[run] = rows

cells = {}
for v in runs["a"]:
    p = v.split(".")
    # arrive_status.v2.<form>.c<N>.w<W>.u<U>
    form, cn, w, u = p[2], p[3], p[4], int(p[5][1:])
    cells[(form, cn, w)] = cells.get((form, cn, w), []) + [u]

print("form           cn w  sass")
for (form, cn, w) in sorted(cells):
    mn = smap.get(f"{form}.u{max(cells[(form, cn, w)])}")
    print("%-14s %-2s %-2s %s" % (form, cn[1:], w[1:], mn))

for run in ("a", "b"):
    print(f"== run {run}")
    print("cell med")
    for v in sorted(runs[run]):
        print("%-52s %8.0f" % (v, runs[run][v]))

print("== slopes by mnemonic")
groups = {}
for (form, cn, w), us in cells.items():
    mn = smap.get(f"{form}.u{max(us)}") or "?"
    key = (mn, form, cn, w)
    pts = [(u, runs["a"].get(
        f"arrive_status.v2.{form}.{cn}.{w}.u{u}"),
        runs["b"].get(f"arrive_status.v2.{form}.{cn}.{w}.u{u}"))
        for u in us]
    groups[key] = pts
for (mn, form, cn, w) in sorted(groups):
    pts = sorted(groups[(mn, form, cn, w)])
    for run_i, run in enumerate(("a", "b")):
        ys = [p[1 + run_i] for p in pts]
        if any(y is None for y in ys):
            continue
        b, res = slope([u for u, _, _ in pts], ys)
        share = res / b if b else float("nan")
        print("%-34s %-24s %s %s run%s slope=%8.2f resid=%.2f "
              "share=%.3f" % (mn, form, cn, w, run, b, res, share))
