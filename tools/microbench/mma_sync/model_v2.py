#!/usr/bin/env python3
"""mma_sync protocol v2 derivation.

    model_v2.py --device h100|b300 --root <run dir> --out model.json

v2: thr slopes are taken at each wps in {1, 2, 4}; per wps the
quotient is thr slope / (8 x wps).  per_warp_blocking = (slope w=4) /
(slope w=1) < 1.25 -> issue_interval = quotient at w=4, experimental.
Otherwise issue_interval = min quotient.  BMMA gets v1 lat + thr
cells; BMMA.SPARSE is ISA-absent -> unrealised:isa_absent.  All other
cell rules are v1 (5 % / 15 %).
"""
import argparse
import json
from pathlib import Path

import model as v1

_MECH_ROOT = (Path(__file__).resolve().parents[3] /
              "configs" / "mechanism_tree")


def _scope_map(source_root):
    """class -> (resource_name, scope, divisor) via
    common/class_resource.json + common/resources.json (v3 scope rule:
    sm-scoped copy-1 units are shared by the 4 SMSPs, so the per-warp
    quotient is 4 x the unit's admission interval)."""
    cr = json.loads((Path(source_root) / "common" /
                     "class_resource.json").read_text())["classes"]
    res = {r["name"]: r for r in json.loads(
        (Path(source_root) / "common" / "resources.json").read_text())}
    out = {}
    for cls, name in cr.items():
        r = res.get(name, {})
        divisor = 4 if (r.get("scope") == "sm" and
                        r.get("copy_count", 1) == 1) else 1
        out[cls] = {"resource": name, "scope": r.get("scope"),
                    "scope_divisor": divisor}
    return out

CELLS = v1.CELLS + ["bmma"]
CLASSES = {
    "DMMA": ("dmma", None),
    "HMMA": ("hmma_f16", "hmma_bf16"),
    "IMMA": ("imma_s8", None),
    "HMMA.SPARSE": ("hmma_sp", None),
    "IMMA.SPARSE": ("imma_sp", None),
    "BMMA": ("bmma", None),
    "BMMA.SPARSE": (None, None),   # ISA-absent
}
WPS = (1, 2, 4)
BLOCK_RATIO = 1.25


def cell_series_wps(rows, cell, mode, wps):
    out = {}
    for r in rows:
        p = r.get("params", {})
        if p.get("mode") != mode or \
                str(p.get("op", "")).lower() != cell:
            continue
        want = 0 if mode == "lat" else wps
        if int(p.get("wps", 0)) != want:
            continue
        out.setdefault(int(r["u"]), []).extend(
            float(x) for x in r["samples"])
    return out


def run_slope(rows, cell, wps):
    f = v1.slope_fit(cell_series_wps(rows, cell, "thr", wps))
    return f["slope"] if f else None


def emulated_cell(checks, cell):
    """b300 rule: gate fails AND every kernel has n_hits == 0 AND a
    CALL* opcode present -> the toolchain emulates the class."""
    ks = [v for v in checks.get("kernels", {}).values()
          if v.get("cell") == cell]
    return bool(ks) and all(
        v.get("n_hits", 0) == 0 and
        any(op.startswith("CALL") for op in v.get("sass_counts", {}))
        for v in ks)


def cell_result(rows_a, rows_b, checks, cell):
    """v1 lat result + per-wps thr quotients (mean of the two runs)."""
    la = v1.slope_fit(v1.cell_series(rows_a, cell, "lat"))
    lb = v1.slope_fit(v1.cell_series(rows_b, cell, "lat"))
    gated = v1.gate_ok(checks, cell)
    r = {"gated": gated,
         "sparse_call_subroutine": v1.sparse_call_subroutine(checks,
                                                           cell)}
    if la and lb:
        va, vb = la["slope"], lb["slope"]
        m = (va + vb) / 2
        r["result_latency"] = m
        r["lat_runs"] = [va, vb]
        r["lat_resid"] = max(la["max_rel_resid"], lb["max_rel_resid"])
        r["lat_agree"] = abs(va - vb) / m if m else float("inf")
    quots, slopes, resids, agrees = {}, {}, {}, {}
    for w in WPS:
        sa, sb = (run_slope(rows_a, cell, w),
                  run_slope(rows_b, cell, w))
        if sa is None or sb is None:
            continue
        m = (sa + sb) / 2
        slopes[w] = m
        quots[w] = m / (8.0 * w)
        resids[w] = max(
            (v1.slope_fit(cell_series_wps(rows_a, cell, "thr", w))
             or {}).get("max_rel_resid", 0),
            (v1.slope_fit(cell_series_wps(rows_b, cell, "thr", w))
             or {}).get("max_rel_resid", 0))
        agrees[w] = abs(sa - sb) / m if m else float("inf")
    r["thr_slopes"] = {str(k): v for k, v in slopes.items()}
    r["quotients"] = {str(k): v for k, v in quots.items()}
    r["thr_resids"] = {str(k): v for k, v in resids.items()}
    r["thr_agrees"] = {str(k): v for k, v in agrees.items()}
    if len(slopes) == len(WPS):
        ratio = slopes[4] / slopes[1] if slopes[1] else float("inf")
        r["wps4_wps1_ratio"] = ratio
        r["per_warp_blocking"] = ratio < BLOCK_RATIO
        r["raw_issue_interval"] = (quots[4] if r["per_warp_blocking"]
                                 else min(quots.values()))
        r["thr_resid"] = max(resids.values())
        r["thr_agree"] = max(agrees.values())
    return r


def status_of(r):
    if not r["gated"] or "result_latency" not in r:
        return "unrealised"
    resid = r["lat_resid"]
    agree = r["lat_agree"]
    if resid <= v1.RESID and agree <= v1.AGREE:
        return "measured"
    if resid <= v1.EXPERIMENTAL and agree <= v1.EXPERIMENTAL:
        return "experimental"
    return "unrealised"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    rows_a = v1.load(root / "run-a")
    rows_b = v1.load(root / "run-b")
    checks_path = root / "build" / "sass" / "checks.json"
    checks = (json.loads(checks_path.read_text())
              if checks_path.is_file() else {"kernels": {}})
    scopes = _scope_map(_MECH_ROOT)
    cells = {c: cell_result(rows_a, rows_b, checks, c) for c in CELLS}
    classes = {}
    for cls, (cell, control) in CLASSES.items():
        scope = scopes.get(cls, {"resource": None, "scope": None,
                                 "scope_divisor": 1})
        if cell is None:
            classes[cls] = {"status": "unrealised:isa_absent", **scope}
            continue
        r = cells[cell]
        if not r["gated"] and emulated_cell(checks, cell):
            classes[cls] = {"status": "unrealised:emulated", "cell": cell,
                            **scope}
            continue
        status = status_of(r)
        if status == "measured":
            if r.get("sparse_call_subroutine") or \
                    r.get("per_warp_blocking"):
                status = "experimental"
        raw_ii = r.get("raw_issue_interval")
        entry = {"cell": cell, "status": status, **scope,
                 "sparse_call_subroutine": r["sparse_call_subroutine"],
                 "per_warp_blocking": r.get("per_warp_blocking"),
                 "wps4_wps1_ratio": r.get("wps4_wps1_ratio"),
                 "result_latency": r.get("result_latency"),
                 "completion_latency": r.get("result_latency"),
                 "issue_interval":
                     (raw_ii / scope["scope_divisor"]
                      if raw_ii is not None else None),
                 "raw_issue_interval": raw_ii,
                 "lat_runs": r.get("lat_runs"),
                 "thr_slopes": r.get("thr_slopes"),
                 "quotients": r.get("quotients"),
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
                if delta > v1.CONTROL and status == "measured":
                    entry["status"] = "experimental"
                    entry["status_why"] = (
                        f"control {control} differs {delta:.3f} > 5 %")
        classes[cls] = entry
    out = {"protocol": "tools/microbench/mma_sync/protocol-v3.json",
           "device": a.device,
           "runs": {"a": str(root / "run-a"), "b": str(root / "run-b")},
           "cells": cells, "classes": classes}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for cls, e in classes.items():
        print(f"  {cls:12s} lat {e.get('result_latency')} "
              f"ii {e.get('issue_interval')} -> {e['status']}"
              + (f" blocking(r={e['wps4_wps1_ratio']:.2f})"
                 if e.get("per_warp_blocking") else ""))


if __name__ == "__main__":
    main()
