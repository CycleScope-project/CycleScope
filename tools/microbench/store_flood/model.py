#!/usr/bin/env python3
"""model.py <run dir> <out.json>: store_flood protocol v1 derivation.
Reads <run dir>/run-{a,b}/store_flood.jsonl (rows: variant, u = group 1..48, params width_bytes / stride_bytes /
lanes / warps, samples = group gaps in cycles, one per (repetition, warp)).  Per run and cell: early gap (groups
2..3), steady gap (groups 33..48), saturation, knee group, service per request, buffer estimate; per run a least
squares fit T = a * sectors + b * lines + c over the saturated cells."""
import json, statistics, sys
from pathlib import Path
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
G = 8  # stores per group
res = {"runs": {}}
for run in ("a", "b"):
    cells = {}
    for line in (root / f"run-{run}" / "store_flood.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert "error" not in r and r["samples"], r.get("error")
        c = cells.setdefault(r["variant"], {"params": {k: r["params"][k] for k in ("width_bytes", "stride_bytes", "lanes", "warps")}, "gap": {}})
        assert r["u"] not in c["gap"]
        c["gap"][r["u"]] = statistics.median(r["samples"])
    ctl = {c["params"]["warps"]: statistics.median(c["gap"][g] for g in range(2, 49)) for n, c in cells.items() if c["params"]["width_bytes"] == 0}
    rows = []
    for name, c in sorted(cells.items()):
        p = c["params"]; g = c["gap"]
        assert set(g) == set(range(1, 49)), (name, sorted(g))
        if p["width_bytes"] == 0: continue
        over = ctl.get(p["warps"], min(ctl.values()))        # control group gap (loop + clock)
        early = statistics.median(g[k] for k in (2, 3)); s1 = statistics.median(g[k] for k in range(33, 41)); s2 = statistics.median(g[k] for k in range(41, 49))
        steady = statistics.median(g[k] for k in range(33, 49))
        sat = steady > 1.5 * early and abs(s1 - s2) <= 0.10 * max(s1, s2)
        knee = next((k for k in range(2, 49) if g[k] > 1.5 * early), None)
        w, st, L, W = p["width_bytes"], p["stride_bytes"], p["lanes"], p["warps"]
        sectors = len({(l * st + b) // 32 for l in range(L) for b in range(w)}); lines = len({(l * st + b) // 128 for l in range(L) for b in range(w)})
        I = (steady - over) / G; T = I / W
        buf = None
        if sat and knee:
            elapsed = sum(g[k] for k in range(1, knee))      # cycles until the knee group starts
            buf = W * G * (knee - 1) - elapsed / T
        c.update(sectors=sectors, lines=lines, early_gap=early, early_issue=(early - over) / G, steady_gap=steady, saturated=sat, knee_group=knee,
                 interval_per_request=I, service_per_request=T, buffer_requests=buf)
        rows.append((name, w, st, L, W, sectors, lines, early, steady, sat, knee, I, T, buf))
    fit = {}
    S = [r for r in rows if r[9]]
    def lsq(sel, cols):
        if len(sel) < len(cols) + 1: return None
        A = np.array([[{"s": r[5], "l": r[6], "1": 1.0}[k] for k in cols] for r in sel], float); y = np.array([r[12] for r in sel], float)
        x, *_ = np.linalg.lstsq(A, y, rcond=None); pred = A @ x
        return {"coef": dict(zip(cols, map(float, x))), "n": len(sel), "max_rel_err": float(np.max(np.abs(pred - y) / y)), "max_abs_err": float(np.max(np.abs(pred - y)))}
    fit["all: a*sectors+b*lines+c"] = lsq(S, "sl1"); fit["all: a*sectors+b*lines"] = lsq(S, "sl"); fit["all: a*sectors+c"] = lsq(S, "s1")
    for w in sorted({r[1] for r in S}): fit[f"width {w}: a*sectors+b*lines"] = lsq([r for r in S if r[1] == w], "sl")
    for W in sorted({r[4] for r in S}): fit[f"warps {W}: a*sectors+b*lines"] = lsq([r for r in S if r[4] == W], "sl")
    res["runs"][run] = {"control_group_gap": ctl, "cells": cells, "fit": fit}
    print(f"== run {run}: control group gap {ctl}")
    print("cell width stride lanes warps sectors lines | early_gap steady_gap saturated knee | interval/request service/request(T) T/(sectors+lines) buffer")
    for r in rows:
        print("%-22s %2d %3d %2d %d %3d %3d | %7.1f %7.1f %-5s %4s | %7.2f %7.2f %5.2f %s" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12],
              r[12] / (r[5] + r[6]), "%.1f" % r[13] if r[13] is not None else "-"))
    for k, v in fit.items(): print("fit", k, json.dumps(v))
out.write_text(json.dumps(res, indent=1, default=str) + "\n")
