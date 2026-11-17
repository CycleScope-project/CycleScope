#!/usr/bin/env python3
"""dispatch_contention model (protocol v1): T = median over
warps x launches x reps; slope = OLS over U; P from the device's
installed instruction.service leaves; excess = slope - P;
d = slope / W for W >= 2 cells with positive excess; decision
A / B / C per the frozen rule.  Writes
$A/microbench/dispatch_contention-model/<dev>-<run>/model.json.

Usage: model.py <artifact_root> <device> <run>
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

VARIANT_OP = {"ffma": "FFMA", "fmnmx": "FMNMX", "mufu": "MUFU.EX2",
              "imad": "IMAD", "mixed": None, "ilp8": "FFMA"}
US = (64, 128, 256)
WS = (1, 2, 3, 4)


def load_rows(path: Path) -> dict:
    rows = {}
    for f in path.glob("*.jsonl"):
        for line in f.open():
            r = json.loads(line)
            # variant field form "<variant>.w<W>"; u field is the window
            v = r["variant"].rsplit(".w", 1)[0]
            wn = int(r["variant"].rsplit(".w", 1)[1])
            u = r["u"]
            rows[(v, wn, u)] = r["samples"]
    return rows


def ols(xs, ys):
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    sxx = sum((x - xm) ** 2 for x in xs)
    b = sxy / sxx
    a = ym - b * xm
    resid = max(abs(y - (a + b * x)) / y for x, y in zip(xs, ys))
    return b, a, resid


def service_leaf(tree: dict, cls: str) -> dict:
    want = (f"instruction.service.root/predicated_on=1/"
            f"operation_class={cls}")
    for node in tree["nodes"]:
        if node.get("id") == want:
            return node.get("parameter", {}).get("fields", {})
    return {}


def field(fields: dict, name: str, default: float) -> float:
    f = fields.get(name)
    if not f:
        return default
    m = f.get("model", {})
    if m.get("kind") == "constant":
        return float(m.get("value", default))
    return default


def P_of(tree, variant: str, W: int, stall_map: dict) -> float:
    """P(variant,W) per protocol: dependent chains
    max(L_class, W * ii_class); L = installed result_latency, which
    must equal the window's control-word stall count."""
    def LI(cls):
        f = service_leaf(tree, cls)
        if not f and "." in cls:  # e.g. MUFU.EX2 has no leaf; the
            f = service_leaf(tree, cls.split(".")[0])  # MUFU root does
        return (field(f, "result_latency_cycles", 1.0),
                field(f, "issue_interval_cycles", 0.0))
    if variant == "mixed":
        lf, ii_f = LI("FFMA")
        lm, ii_m = LI("FMNMX")
        return max((lf + lm) / 2, W * max(ii_f, ii_m) / 2)
    if variant == "ilp8":
        lf, ii_f = LI("FFMA")
        return max(1.0, W * ii_f)
    cls = VARIANT_OP[variant]
    l, ii = LI(cls)
    return max(l, W * ii)


def main() -> int:
    A, dev, run = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    base = A / "microbench" / "dispatch_contention" / f"{dev}-{run}"
    runs = {r: load_rows(base / r) for r in ("run-a", "run-b")
            if (base / r).is_dir()}
    tree = json.load(open(
        Path(__file__).resolve().parents[3] / "configs" / "trees"
        / f"{dev}.json"))
    chk = (A / "microbench" / "dispatch_contention-model" / f"{dev}-{run}"
           / "checks.json")
    checks = json.load(open(chk)) if chk.exists() else {}
    cells = {}
    excluded = []
    for v in VARIANT_OP:
        for w in WS:
            slopes, resids, agrees = {}, {}, {}
            for rname, rows in runs.items():
                Ts = []
                for u in US:
                    s = rows.get((v, w, u), [])
                    Ts.append(statistics.median(s) if s else None)
                if any(t is None for t in Ts):
                    slopes[rname] = None
                    continue
                b, a, resid = ols(US, Ts)
                slopes[rname] = b
                resids[rname] = resid
            if len([b for b in slopes.values() if b]) < 2:
                excluded.append(f"{v}/w{w}")
                continue
            sa, sb = slopes["run-a"], slopes["run-b"]
            agree = abs(sa - sb) / ((sa + sb) / 2)
            resid = max(resids.get("run-a", 0), resids.get("run-b", 0))
            key = f"{v}/w{w}"
            gate = checks.get(f"{v}/u{US[-1]}", {})
            usable = resid <= 0.02 and agree <= 0.03 \
                and gate.get("ok", False)
            slope = (sa + sb) / 2
            P = P_of(tree, v, w, {})
            excess = slope - P
            entry = {"slope": slope, "resid": resid, "agree": agree,
                     "P": P, "excess": excess, "gate_ok": gate.get("ok"),
                     "stalls": gate.get("stalls"), "usable": usable}
            if w >= 2 and excess > 0:
                entry["d"] = slope / w
            cells[key] = entry
            if not usable:
                excluded.append(key)
    # decision (protocol verbatim): C if all excess <= max(2% slope,1)
    small = [k for k, c in cells.items()
             if c["usable"] and c["excess"] <= max(0.02 * c["slope"], 1.0)]
    big = [k for k, c in cells.items()
           if c["usable"]
           and not c["excess"] <= max(0.02 * c["slope"], 1.0)]
    ds = [cells[k]["d"] for k in big
          if int(k.split("w")[1]) >= 2 and cells[k]["excess"] > 0
          and "d" in cells[k]]
    w1_excess = max([cells[k]["excess"] for k in cells
                     if k.endswith("w1") and cells[k]["usable"]] or [0])
    decision = "C"
    reason = "all excess <= max(2% of slope, 1 cycle)"
    med_d = None
    if big and ds:
        med_d = statistics.median(ds)
        spread = (max(ds) - min(ds)) / med_d if med_d else 1e9
        if spread <= 0.10 and w1_excess <= 1.0:
            decision = "A"
            reason = (f"d spread {spread:.3f} <= 10% and W=1 excess "
                      f"{w1_excess:.2f} <= 1")
            # ilp8 consistency: slope ~= max(1, W*max(ii_ffma, d)) in 5%
            for w in WS[1:]:
                c = cells.get(f"ilp8/w{w}")
                if not c or not c["usable"]:
                    continue
                ii = service_leaf(tree, "FFMA").get(
                    "issue_interval_cycles", {}).get(
                        "model", {}).get("value", 0.0)
                want = max(1.0, w * max(float(ii), med_d))
                if abs(c["slope"] - want) / want > 0.05:
                    reason += ("; ilp8 inconsistency w"
                               f"{w}: slope {c['slope']:.2f} vs {want:.2f}"
                               " -> downgraded experimental")
                    decision = "A*"
        else:
            decision = "B"
            reason = f"d spread {spread:.3f} > 10% or W=1 excess {w1_excess:.2f}"
    out = {"device": dev, "run": run, "cells": cells,
           "excluded": excluded, "d_median": med_d,
           "decision": decision, "reason": reason}
    mdir = A / "microbench" / "dispatch_contention-model" / f"{dev}-{run}"
    mdir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(mdir / "model.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
