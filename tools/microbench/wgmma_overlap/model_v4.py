#!/usr/bin/env python3
"""wgmma_overlap protocol v4 decision (C3 / C3+C5 / C3+C6 / C3+C5+C6),
written before the r6 data.

    model_v4.py --root <run dir> --out model.json

Variants as v2/v3 (`<form>.g8.<kind>.u<U>.<self|other>.w<W>`, ctrl); the
per-cell derivation is model_v2's.  n accounting per kind:
  gpr   = distinct GPR sources, flags = .reuse-flagged sources,
  uni   = uniform / constant-bank sources.
C3: n = gpr; C5: +0.5 * flags; C6: +uni.  t = max(t_ctrl, 2, n).
Tolerance max(15 %, 0.5); install measured at 100 % of gating cells,
experimental at >= 80 % with no cell beyond 1.0.
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL_REL, TOL_ABS, HARD_ABS = 0.15, 0.5, 1.0
KINDS = {  # kind: (gpr, flags, uni)
    "ilp8": (3, 2, 0), "ilp8n": (3, 0, 0), "fadd1": (1, 0, 0), "fmul2": (2, 0, 0),
    "fmul_r": (2, 1, 0), "fadd2n": (2, 0, 0), "ffma_ur": (2, 0, 1),
    "ffma_ur_r": (2, 1, 1), "fmul_ur": (1, 0, 1), "f2fp": (2, 0, 0), "mufu8": (1, 0, 0),
}
FORMS = ["hgmma_bf16_n256_ss", "hgmma_bf16_n128_ss", "hgmma_bf16_n256_rs"]
CANDS = {"C3": (False, False), "C3+C5": (True, False), "C3+C6": (False, True),
         "C3+C5+C6": (True, True)}


def within(p, t, rel=TOL_REL, ab=TOL_ABS):
    return abs(p - t) <= max(rel * abs(t), ab)


def pred(kind, t_ctrl, c5, c6):
    gpr, flags, uni = KINDS[kind]
    n = gpr + (0.5 * flags if c5 else 0) + (uni if c6 else 0)
    return max(t_ctrl, 2, n)


def verdict(rows):
    if not rows:
        return {"n": 0, "install": None}
    ok = [within(p, t) for _, p, t, _ in rows]
    hard = all(abs(p - t) <= HARD_ABS for _, p, t, _ in rows)
    frac = sum(ok) / len(ok)
    level = ("measured" if all(ok) and all(a for *_, a in rows)
             else "experimental" if frac >= 0.8 and hard else None)
    return {"n": len(rows), "pass_frac": frac,
            "max_abs_err": max(abs(p - t) for _, p, t, _ in rows), "install": level,
            "fails": [{"cell": nm, "pred": p, "t": t} for (nm, p, t, _), o in zip(rows, ok) if not o]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "m2.json"
        subprocess.run([sys.executable, str(HERE / "model_v2.py"), "--root", a.root,
                        "--out", str(tmp)], check=True, stdout=subprocess.DEVNULL)
        C = json.loads(tmp.read_text())["cells"]
    out = {"protocol": "wgmma_overlap v4", "root": a.root, "cells": C, "candidates": {}}
    table = {}
    for cand, (c5, c6) in CANDS.items():
        rows = []
        for form in FORMS:
            for kind in KINDS:
                for u in (32, 64, 128):
                    c = C.get(f"{form}.g8.{kind}.u{u}.self.w1")
                    if c and c.get("usable"):
                        p = pred(kind, c["t_ctrl"], c5, c6)
                        rows.append((f"{form}.{kind}.self.w1", p, c["t"], c["accepted"]))
                        table.setdefault(f"{form}.{kind}", {"t_ctrl": c["t_ctrl"], "t": c["t"]})[cand] = p
                    o = C.get(f"{form}.g8.{kind}.u{u}.other.w2")
                    if o and o.get("usable"):
                        rows.append((f"{form}.{kind}.other.w2", o["t_ctrl"], o["t"], o["accepted"]))
        out["candidates"][cand] = verdict(rows)
    chosen = next((cand for cand in CANDS if out["candidates"][cand]["install"]), None)
    out["decision"] = {"chosen": chosen,
                       "level": out["candidates"][chosen]["install"] if chosen else None,
                       "summary": {k: (v["install"], round(v.get("pass_frac", 0), 3), round(v.get("max_abs_err", 0), 3))
                                   for k, v in out["candidates"].items()}}
    out["table"] = table
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out["decision"], indent=1))
    print(f"{'cell':40s} ctrl    t     " + " ".join(f"{c:>9s}" for c in CANDS))
    for k, v in sorted(table.items()):
        print(f"{k:40s} {v['t_ctrl']:5.2f} {v['t']:5.2f} " + " ".join(f"{v.get(c, float('nan')):9.2f}" for c in CANDS))


if __name__ == "__main__":
    main()
