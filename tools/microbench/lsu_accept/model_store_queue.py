#!/usr/bin/env python3
"""Derivation for lsu_accept protocol v4 (store_queue probe), frozen in
protocol-v4.json "derivation".

    model_store_queue.py <run dir with run-a run-b> <model dir>
        [--port-bpc <installed store_l2.sector_bytes_per_cycle>]

Per cell (variant, W): median per-store issue gap g_k (k = 1..24), g0 =
median(g_1..g_3), knee k* = smallest k >= 2 with g_k and g_{k+1} both
>= max(2 g0, g0 + 6); Q_cell = k* - 1; drain r_cell = median(g_k, k >
k*).  Decisions (a) scope, (b) unit, (c) repeatability, (d) drain per
the protocol; writes model.json + table.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

D = 24
WS = (1, 2, 4, 8)


def load_run(path: Path) -> dict:
    cells: dict = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or not r.get("samples"):
            continue
        p = r["params"]
        cells[(p["variant"], p["w"])][p["k"]] = statistics.median(r["samples"])
    return cells


def analyse_cell(g: dict) -> dict:
    gaps = [g.get(k) for k in range(1, D + 1)]
    if any(x is None for x in gaps):
        return {"usable": False, "why": "missing k rows"}
    g0 = statistics.median(gaps[:3])
    thr = max(2 * g0, g0 + 6)
    knee = None
    for k in range(2, D):            # need g_k and g_{k+1}
        if gaps[k - 1] >= thr and gaps[k] >= thr:
            knee = k
            break
    out = {"usable": True, "g0": g0, "threshold": thr, "gaps": gaps,
           "total": g.get(0)}
    if knee is None:
        out.update({"knee": None, "q": None, "drain": None,
                    "note": "no knee up to k = 24 (queue deeper than 24)"})
        return out
    tail = gaps[knee:]               # k > knee
    out.update({"knee": knee, "q": knee - 1,
                "drain": statistics.median(tail) if tail else None})
    return out


def within(a: float, b: float, pct: float) -> bool:
    return a is not None and b is not None and abs(a - b) <= pct / 100.0 * max(abs(a), abs(b), 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--port-bpc", type=float, default=None,
                    help="installed store_l2.sector_bytes_per_cycle (B/cycle per SM)")
    a = ap.parse_args()
    a.model_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for rn in ("run-a", "run-b"):
        f = a.run_dir / rn / "store_queue.jsonl"
        if f.exists():
            runs[rn] = {k: analyse_cell(v) for k, v in load_run(f).items()}
    if len(runs) < 2:
        (a.model_dir / "model.json").write_text(json.dumps(
            {"decision": "none", "why": "fewer than two runs"}, indent=1))
        print("model_store_queue: fewer than two runs -> none")
        return 1

    def cell(rn, v, w):
        return runs[rn].get((v, w), {"usable": False})

    report = {"runs": {rn: {f"{v}.w{w}": c for (v, w), c in cs.items()}
                       for rn, cs in runs.items()}}
    # (c) repeatability on stg16 / stg4, every W
    repeat_ok, repeat_notes = True, []
    for v in ("stg16", "stg4"):
        for w in WS:
            ca, cb = cell("run-a", v, w), cell("run-b", v, w)
            if not (ca.get("usable") and cb.get("usable")):
                repeat_notes.append(f"{v}.w{w}: unusable"); continue
            if ca.get("q") != cb.get("q"):
                repeat_ok = False
                repeat_notes.append(f"{v}.w{w}: q {ca.get('q')} vs {cb.get('q')}")
            if ca.get("drain") and cb.get("drain") and not within(ca["drain"], cb["drain"], 15):
                repeat_ok = False
                repeat_notes.append(f"{v}.w{w}: drain {ca['drain']:.1f} vs {cb['drain']:.1f}")
    # (a) scope on stg16 W in {1,2,4}, run-a (run-b must agree per (c))
    q = {w: cell("run-a", "stg16", w).get("q") for w in WS}
    scope, q_inst = "none", None
    if all(q[w] is not None for w in (1, 2, 4)):
        qs = [q[1], q[2], q[4]]
        if max(qs) - min(qs) <= 1:
            scope, q_inst = "QS", int(statistics.median(qs))
        elif all(abs(q[w] - q[1] / w) <= 1 for w in (2, 4)):
            scope, q_inst = "QM", q[1]
    w8_check = None
    if scope == "QS" and q[8] is not None:
        w8_check = {"predicted": q_inst / 2, "measured": q[8],
                    "ok": abs(q[8] - q_inst / 2) <= 1}
    elif scope == "QM" and q[8] is not None:
        w8_check = {"predicted": q_inst / 8, "measured": q[8],
                    "ok": abs(q[8] - q_inst / 8) <= 1}
    # (b) unit
    q4 = cell("run-a", "stg4", 1).get("q")
    unit = "none"
    if q4 is not None and q[1] is not None:
        if abs(q4 - q[1]) <= 1:
            unit = "requests"
        elif within(q4, 4 * q[1], 25):
            unit = "sectors"
    # (d) drain vs installed port
    drain = {}
    for v, bpl in (("stg16", 16), ("stg4", 4)):
        for w in WS:
            c = cell("run-a", v, w)
            r = c.get("drain")
            if r:
                bw = w * 32 * bpl / r          # B/cycle per SM if the SM port is the drain
                drain[f"{v}.w{w}"] = {"cycles_per_request": r, "sm_bandwidth_B_per_cycle": bw,
                                      "vs_port": (bw / a.port_bpc) if a.port_bpc else None}
    drain_ok = None
    if a.port_bpc:
        checks = [drain[f"stg16.w{w}"]["vs_port"] for w in (1, 2, 4) if f"stg16.w{w}" in drain]
        drain_ok = bool(checks) and all(abs(x - 1.0) <= 0.15 for x in checks)
    decision = "none"
    if repeat_ok and scope in ("QS", "QM") and unit in ("requests", "sectors"):
        decision = scope + ("-sectors" if unit == "sectors" else "")
    elif repeat_ok and scope in ("QS", "QM"):
        decision = scope + " (unit undecided: requests from stg16)"
    fields = {}
    if decision != "none":
        fields["store_l2.queue_scope_smsp"] = 1 if scope == "QS" else 0
        if unit == "sectors":
            fields["store_l2.queue_sectors"] = q_inst * 16
        else:
            fields["store_l2.queue_requests"] = q_inst
    model = {"decision": decision, "scope": scope, "unit": unit,
             "q_by_w_stg16": q, "q_stg4_w1": q4, "w8_check": w8_check,
             "repeatable": repeat_ok, "repeat_notes": repeat_notes,
             "drain": drain, "drain_matches_installed_port": drain_ok,
             "installed_port_bpc": a.port_bpc, "fields": fields}
    (a.model_dir / "model.json").write_text(json.dumps(model, indent=1))
    (a.model_dir / "cells.json").write_text(json.dumps(report, indent=1))
    lines = ["| cell | g0 | knee | Q | drain (cyc/req) | SM B/cycle | gaps k=1..24 |", "|---|---|---|---|---|---|---|"]
    for rn in runs:
        for (v, w), c in sorted(runs[rn].items()):
            if not c.get("usable"):
                lines.append(f"| {rn} {v}.w{w} | unusable | | | | | |"); continue
            bw = (w * 32 * (16 if v == "stg16" else 4) / c["drain"]) if c.get("drain") else None
            lines.append(f"| {rn} {v}.w{w} | {c['g0']:.1f} | {c['knee']} | {c['q']} | "
                         f"{c['drain']:.1f} | {bw:.1f} | " if c.get("drain") else
                         f"| {rn} {v}.w{w} | {c['g0']:.1f} | - | >=24 | - | - | ")
            lines[-1] += " ".join(f"{x:.0f}" for x in c["gaps"]) + " |"
    (a.model_dir / "table.md").write_text("\n".join(lines) + "\n")
    print(f"model_store_queue: decision={decision} scope={scope} unit={unit} q={q} q4={q4} "
          f"repeatable={repeat_ok} drain_ok={drain_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
