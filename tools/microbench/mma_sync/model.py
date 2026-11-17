#!/usr/bin/env python3
"""mma_sync protocol v1 derivation.

    model.py --device h100|b300 --root <run dir> --out model.json

<run dir> holds run-a/ and run-b/ mma.jsonl and build/sass/checks.json.
Per cell: lat slope = OLS over U of median windows (result_latency =
completion_latency); issue_interval = thr slope / 8.  Acceptance per
protocol: residual <= 5 % and runs within 5 % -> measured; residual or
spread within 15 % -> experimental; beyond / gate fail / missing ->
unrealised.  HMMA installs hmma_f16 with hmma_bf16 as control (> 5 % ->
experimental); other classes install their own cell.
"""
import argparse
import json
import statistics
from pathlib import Path

CELLS = ["dmma", "hmma_f16", "hmma_bf16", "imma_s8", "hmma_sp", "imma_sp"]
# class -> (install cell, control cell or None)
CLASSES = {
    "DMMA": ("dmma", None),
    "HMMA": ("hmma_f16", "hmma_bf16"),
    "IMMA": ("imma_s8", None),
    "HMMA.SPARSE": ("hmma_sp", None),
    "IMMA.SPARSE": ("imma_sp", None),
}
RESID = 0.05
AGREE = 0.05
EXPERIMENTAL = 0.15
CONTROL = 0.05


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return b, my - b * mx


def slope_fit(series):
    us = sorted(series)
    if len(us) < 2:
        return None
    meds = [statistics.median(series[u]) for u in us]
    b, a = ols(us, meds)
    resid = max(abs(m - (a + b * u)) / max(1.0, abs(m))
                for u, m in zip(us, meds))
    return {"slope": b, "intercept": a, "us": us, "medians": meds,
            "max_rel_resid": resid}


def load(run_dir):
    rows = []
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "samples" in d:
                rows.append(d)
    return rows


def cell_series(rows, cell, mode):
    out = {}
    for r in rows:
        p = r.get("params", {})
        if p.get("mode") != mode or \
                str(p.get("op", "")).lower() != cell:
            continue
        if mode == "thr" and int(p.get("wps", 0)) != 1:
            continue
        out.setdefault(int(r["u"]), []).extend(
            float(x) for x in r["samples"])
    return out


def gate_ok(checks, cell):
    """A cell passes iff every gated kernel for it is ok and at least
    one kernel was gated."""
    ks = [v for k, v in checks.get("kernels", {}).items()
          if str(v.get("cell", "")).lower() == cell.lower()]
    return bool(ks) and all(k.get("ok") for k in ks)


def sparse_call_subroutine(checks, cell):
    """sm_103a ptxas wraps sparse mma.sync in a CALL metadata subroutine
    per instance: measured -> experimental cap (addendum 2026-11-05)."""
    return any(v.get("sparse_call_subroutine")
               for v in checks.get("kernels", {}).values()
               if str(v.get("cell", "")).lower() == cell.lower())


def cell_result(rows_a, rows_b, checks, cell):
    la = slope_fit(cell_series(rows_a, cell, "lat"))
    lb = slope_fit(cell_series(rows_b, cell, "lat"))
    ta = slope_fit(cell_series(rows_a, cell, "thr"))
    tb = slope_fit(cell_series(rows_b, cell, "thr"))
    gated = gate_ok(checks, cell)
    r = {"gated": gated,
         "sparse_call_subroutine": sparse_call_subroutine(checks, cell)}
    if la and lb:
        va, vb = la["slope"], lb["slope"]
        m = (va + vb) / 2
        agree = abs(va - vb) / m if m else float("inf")
        resid = max(la["max_rel_resid"], lb["max_rel_resid"])
        r["result_latency"] = m
        r["lat_runs"] = [va, vb]
        r["lat_resid"] = resid
        r["lat_agree"] = agree
    if ta and tb:
        va, vb = ta["slope"] / 8.0, tb["slope"] / 8.0
        m = (va + vb) / 2
        agree = abs(va - vb) / m if m else float("inf")
        resid = max(ta["max_rel_resid"], tb["max_rel_resid"])
        r["issue_interval"] = m
        r["thr_runs"] = [va, vb]
        r["thr_resid"] = resid
        r["thr_agree"] = agree
    return r


def status_of(r):
    if not r["gated"] or "result_latency" not in r:
        return "unrealised"
    resid = r["lat_resid"]
    agree = r["lat_agree"]
    if resid <= RESID and agree <= AGREE:
        return "measured"
    if resid <= EXPERIMENTAL and agree <= EXPERIMENTAL:
        return "experimental"
    return "unrealised"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    rows_a = load(root / "run-a")
    rows_b = load(root / "run-b")
    checks_path = root / "build" / "sass" / "checks.json"
    checks = (json.loads(checks_path.read_text())
              if checks_path.is_file() else {"kernels": {}})
    cells = {c: cell_result(rows_a, rows_b, checks, c) for c in CELLS}
    classes = {}
    for cls, (cell, control) in CLASSES.items():
        r = cells[cell]
        status = status_of(r)
        if status == "measured" and r.get("sparse_call_subroutine"):
            status = "experimental"
        entry = {"cell": cell, "status": status,
                 "sparse_call_subroutine": r["sparse_call_subroutine"],
                 "result_latency": r.get("result_latency"),
                 "completion_latency": r.get("result_latency"),
                 "issue_interval": r.get("issue_interval"),
                 "lat_runs": r.get("lat_runs"),
                 "thr_runs": r.get("thr_runs"),
                 "lat_resid": r.get("lat_resid"),
                 "lat_agree": r.get("lat_agree"),
                 "thr_resid": r.get("thr_resid"),
                 "thr_agree": r.get("thr_agree"),
                 "gate": r["gated"]}
        if control is not None:
            c = cells[control]
            entry["control_cell"] = control
            entry["control_result_latency"] = c.get("result_latency")
            entry["control_runs"] = c.get("lat_runs")
            v, cv = r.get("result_latency"), c.get("result_latency")
            if v is not None and cv is not None and v:
                delta = abs(cv - v) / v
                entry["control_delta"] = delta
                if delta > CONTROL and status == "measured":
                    entry["status"] = "experimental"
                    entry["status_why"] = (
                        f"control {control} differs {delta:.3f} > 5 %")
        classes[cls] = entry
    out = {"protocol": "tools/microbench/mma_sync/protocol.json",
           "device": a.device,
           "runs": {"a": str(root / "run-a"), "b": str(root / "run-b")},
           "cells": cells, "classes": classes}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for cls, e in classes.items():
        print(f"  {cls:12s} lat {e.get('result_latency')} "
              f"ii {e.get('issue_interval')} -> {e['status']}"
              + ("" if not e.get("control_delta") else
                 f" (control delta {e['control_delta']:.3f})"))


if __name__ == "__main__":
    main()
