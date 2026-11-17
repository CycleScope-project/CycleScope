#!/usr/bin/env python3
"""table.py <run dir>: per cell run a/b medians C of the chain cycles;
extra = C(K) - C(0) for the same victim / form / issuer.  List only."""
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
        victim, form, issuer, k = v.split(".")
        ctl = rows.get(f"{victim}.{form}.{issuer}.k0", float("nan"))
        print("%-34s %8.0f %8.0f" % (v, rows[v], rows[v] - ctl))
