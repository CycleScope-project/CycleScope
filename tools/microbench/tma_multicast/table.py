#!/usr/bin/env python3
"""table.py <run dir>: tma_multicast v1 table (list only).  Per cell
(mode, N, S) and run: issued boxes / 1000 cycles per issuing CTA,
received boxes / 1000 cycles per SM, issue->complete latency p10/p50/p90,
aggregate bytes/ns."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
print("run family issued/1k received/1k lat_p10 lat_p50 lat_p90 agg_Bns error")
for run in ("a", "b"):
    for line in (root / f"run-{run}.jsonl").read_text().splitlines():
        r = json.loads(line)
        print("%s %-38s %9.3f %9.3f %7.0f %7.0f %7.0f %8.2f %s" % (
            run, r["family"], r["issued_boxes_per_1k"],
            r["received_boxes_per_1k"], r["latency_p10"], r["latency_p50"],
            r["latency_p90"], r["aggregate_bytes_per_ns_median"],
            r.get("error", "")))
