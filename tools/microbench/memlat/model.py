#!/usr/bin/env python3
"""memlat leaf derivation per protocol: read every cell dir's probe jsonl,
emit medians + spread per field."""
import argparse, json, sys
from pathlib import Path

FIELDS = {
    "stage27_memlat.l1.chain": "l1.latency_cycles",
    "stage27_memlat.l2.chain": "l2.latency_cycles",
    "stage27_memlat.dram.chain.cg": "dram.latency_cycles",
    "stage27_memlat.dram.chain.cv": "dram.latency_cycles.cv_check",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    for jl in Path(a.run).glob("*/stage27_memlat_*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            fam = d.get("family")
            if fam and fam not in rows:
                rows[fam] = d
    out = {"run": a.run, "cells": {}, "fields": {}}
    for fam, d in sorted(rows.items()):
        med = d.get("median")
        p10, p90 = d.get("p10"), d.get("p90")
        spread = (p90 - p10) / med if med else None
        out["cells"][fam] = {"median": med, "p10": p10, "p90": p90,
                             "spread_rel": spread, "n": d.get("n"),
                             "cell_dir": jl.parent.name}
        f = FIELDS.get(fam)
        if f:
            out["fields"][f] = {"value": med, "spread_rel": spread}
    # L2 near/far: protocol wants the two latency clusters of the 2MiB chase
    # across 8 SMs.  The stage27 l2.cell reports one distribution per run;
    # report the raw samples so the far/near split is judged downstream.
    l2 = rows.get("stage27_memlat.l2.chain")
    if l2 and l2.get("samples"):
        s = sorted(l2["samples"])
        out["l2_samples_sorted"] = s
    # constant cache: no ldc/const op exists in stage27_memlat_ops (kOpNames
    # has no const load) -> unrealised on this probe, listed per protocol.
    out["fields"]["constant_cache"] = {"value": None,
        "note": "no const-load op in stage27_memlat_ops; unrealised"}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for f, v in out["fields"].items():
        print(f"  {f}: {v.get('value')} spread={v.get('spread_rel')}")

if __name__ == "__main__":
    main()
