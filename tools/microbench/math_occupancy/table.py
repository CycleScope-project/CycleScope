#!/usr/bin/env python3
"""table.py <run dir>: per unit run-a p50, run-b p50, c = p50 / 256,
participants / warps_per_smsp_actual / discarded_launches.  Lists only."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for run in ("a", "b"):
    for line in (root / f"run-{run}.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (run, r["stream"], r["placement"], r["warps_per_smsp"],
               r.get("ctas", 1))
        rows[key] = r
print("unit\trun_a_p50\trun_b_p50\tc\tparticipants\twarps_per_smsp_actual\tdiscarded_launches")
for (run, s, p, w, cta), r in sorted(rows.items()):
    if run != "a":
        continue
    b = rows.get(("b", s, p, w, cta))
    print(f"{s}.{p}.W{w}.C{cta}\t{r['p50']}\t{b and b['p50']}\t"
          f"{r['p50'] / 256:.4f}\t{r.get('participants', '')}\t"
          f"{r.get('warps_per_smsp_actual', '')}\t"
          f"{r.get('discarded_launches', '')}")
