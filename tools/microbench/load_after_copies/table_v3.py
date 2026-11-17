#!/usr/bin/env python3
"""table_v3.py <run dir>: per cell run a/b medians C of the chain cycles;
extra = C(K) - C(0) for the same (n, a, timing, form, issuer, smem).
List only."""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
for run in ("a", "b"):
    f = root / f"run-{run}.jsonl"
    rows = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        rows[r["variant"]] = statistics.median(r["samples"])
    print(f"== run {run}")
    print("cell med extra")
    for v in sorted(rows):
        if not v.startswith("v3."):
            continue
        base = v.rsplit(".k", 1)[0]
        ctl = rows.get(base + ".k0", float("nan"))
        print("%-44s %8.0f %8.0f" % (v, rows[v], rows[v] - ctl))
