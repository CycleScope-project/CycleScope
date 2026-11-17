#!/usr/bin/env python3
"""ffma2_operands model (protocol v1): interval = OLS slope over K / 32
(one op per chain per K, 8 chains x 4 warps = 32 ops per warp-instr step;
written into the output per cell), v3 acceptance; decision (a)/(b)/(c).
"""
import argparse
import json
import sys
from pathlib import Path


def load(run_dir):
    rows = []
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def series(rows, cell):
    out = {}
    for r in rows:
        v = str(r.get("variant") or "")
        cell_v = (r.get("params") or {}).get("cell") or v.rsplit("_k",
                                                              1)[0]
        if cell_v != cell and v != f"{cell}_k{r['u']}":
            continue
        out.setdefault(int(r["u"]), []).extend(
            float(x) for x in r["samples"])
    return out


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) \
        / sum((x - mx) ** 2 for x in xs)
    return b, my - b * mx


def fit(s):
    ks = sorted(s)
    if len(ks) < 2:
        return None
    meds = [sorted(s[k])[len(s[k]) // 2] for k in ks]
    b, a = ols(ks, meds)
    resid = max(abs(m - (a + b * k)) / max(1.0, abs(m))
                for k, m in zip(ks, meds))
    return {"slope": b, "interval": b / 32, "ops_per_K": 1,
            "chains": 8, "warps": 16, "divisor": "slope/32",
            "ks": ks, "medians": meds, "max_rel_resid": resid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--checks", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    checks = None
    cp = Path(a.checks) if a.checks else None
    if cp and cp.is_file():
        checks = json.loads(cp.read_text())
    out = {"protocol": "ffma2_operands v1", "gate": checks,
           "cells": {}, "decision": {}}
    I = {}
    for cell in ("F3", "F2a", "F2b", "F1", "A2", "S3"):
        e = {}
        for run in ("run-a", "run-b"):
            e[run] = fit(series(load(root / run), f"thr.{cell}"))
        if e["run-a"] and e["run-b"]:
            i1, i2 = e["run-a"]["interval"], e["run-b"]["interval"]
            e["interval"] = (i1 + i2) / 2
            e["agree_rel"] = abs(i1 - i2) / e["interval"]
            e["status"] = ("measured" if e["agree_rel"] <= 0.05
                           and max(e["run-a"]["max_rel_resid"],
                                   e["run-b"]["max_rel_resid"]) <= 0.05
                           else "experimental"
                           if e["agree_rel"] <= 0.15
                           else "unrealised")
        out["cells"][cell] = e
        I[cell] = e.get("interval")

    def rel(x, y):
        return abs(x - y) / y if x is not None and y else None

    dec = {"F3_vs_2.921": rel(I.get("F3"), 2.921),
           "A2_vs_1.991": rel(I.get("A2"), 1.991),
           "S3_vs_1.007": rel(I.get("S3"), 1.007)}
    f3, a2 = I.get("F3"), I.get("A2")
    dec["controls_ok"] = (
        dec["A2_vs_1.991"] is not None and dec["A2_vs_1.991"] <= 0.05
        and dec["S3_vs_1.007"] is not None
        and dec["S3_vs_1.007"] <= 0.10)
    f2s = [I.get(c) for c in ("F2a", "F2b", "F1")]
    if not dec["controls_ok"]:
        dec["decision"] = "c: controls failed"
    elif (dec["F3_vs_2.921"] is not None
          and dec["F3_vs_2.921"] <= 0.05 and a2):
        if all(f is not None and rel(f, a2) <= 0.05 for f in f2s):
            dec["decision"] = ("a: operand-read limit; FFMA2 interval = "
                               "F2/F1 value; 3-distinct-pair cost is an "
                               "unmodelled register-operand mechanism")
        elif all(f is not None and rel(f, f3) <= 0.05 for f in f2s):
            dec["decision"] = "b: class property; install F3 value"
        else:
            dec["decision"] = "c: ordered/mixed intervals"
    else:
        dec["decision"] = "c"
    out["decision"] = dec
    Path(a.out).write_text(json.dumps(out, indent=1))
    for c, e in out["cells"].items():
        print(c, e.get("interval"), e.get("status"))
    print(dec["decision"])


if __name__ == "__main__":
    sys.exit(main())
