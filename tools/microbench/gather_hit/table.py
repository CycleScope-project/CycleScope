#!/usr/bin/env python3
"""table.py <run dir>: per (mode,N,W,K,L) run-a p50, run-b p50,
T = p50 - control (K=0 same mode+N, nearest W), and W*K*L."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {}
for run in ("a", "b"):
    for line in (root / f"run-{run}.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[(run, r["mode"], r["ctas"], r["warps"], r["loads"],
              r["lines"])] = r["p50"]
def ctl(mode, N, W):
    for w in (W, 1, 4, 2):
        k = ("a", mode, N, w, 0, 32)
        if k in rows: return rows[k]
    return 0
print("cell\trun_a_p50\trun_b_p50\tT\tWxKxL")
seen = set()
for (run, mode, N, W, K, L), v in sorted(rows.items()):
    if run != "a" or K == 0: continue
    b = rows.get(("b", mode, N, W, K, L))
    t = v - ctl(mode, N, W)
    print(f"{mode}.N{N}.W{W}.K{K}.L{L}\t{v}\t{b}\t{t}\t{W*K*L}")
