#!/usr/bin/env python3
"""wgmma_overlap protocol v5 decision (C3 / C3+C7 / C3+C7+R), before r7 data.
    model_v5.py --root <run dir> --out model.json
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
TOL_REL, TOL_ABS, HARD_ABS = 0.15, 0.5, 1.0
# kind: (gpr, flagged)
KINDS = {"ilp8": (3, 1), "ilp8n": (3, 0), "fadd1": (1, 0), "fmul2": (2, 0), "fmul_r": (2, 1),
         "fadd2n": (2, 0), "ffma_ur": (2, 0), "ffma_ur_r": (2, 1), "fmul_ur": (1, 0)}
FORMS = {"hgmma_bf16_n256_ss": "ss", "hgmma_bf16_n128_ss": "ss", "hgmma_bf16_n256_rs": "rs"}
CANDS = {"C3": (False, False), "C3+C7": (True, False), "C3+C7+R": (True, True)}

def within(p, t): return abs(p - t) <= max(TOL_REL * abs(t), TOL_ABS)
def pred(kind, form, t_ctrl, c7, r):
    gpr, fl = KINDS[kind]
    floor = 4 if (c7 and fl) else 2
    if r and FORMS[form] == "rs": floor += 0.5
    return max(t_ctrl, floor, gpr + (0.5 if (r and FORMS[form] == "rs") else 0))
def verdict(rows):
    if not rows: return {"n": 0, "install": None}
    ok = [within(p, t) for _, p, t, _ in rows]; frac = sum(ok) / len(ok)
    hard = all(abs(p - t) <= HARD_ABS for _, p, t, _ in rows)
    level = "measured" if all(ok) and all(a for *_, a in rows) else "experimental" if frac >= 0.8 and hard else None
    return {"n": len(rows), "pass_frac": frac, "max_abs_err": max(abs(p - t) for _, p, t, _ in rows), "install": level,
            "fails": [{"cell": n, "pred": p, "t": t} for (n, p, t, _), o in zip(rows, ok) if not o]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "m2.json"
        subprocess.run([sys.executable, str(HERE / "model_v2.py"), "--root", a.root, "--out", str(tmp)], check=True, stdout=subprocess.DEVNULL)
        C = json.loads(tmp.read_text())["cells"]
    out = {"protocol": "wgmma_overlap v5", "root": a.root, "cells": C, "candidates": {}, "table": {}}
    for cand, (c7, r) in CANDS.items():
        rows = []
        for form in FORMS:
            for kind in KINDS:
                for u in (64, 128):
                    c = C.get(f"{form}.g8.{kind}.u{u}.self.w1")
                    if c and c.get("usable"):
                        p = pred(kind, form, c["t_ctrl"], c7, r)
                        rows.append((f"{form}.{kind}.self.w1", p, c["t"], c["accepted"]))
                        out["table"].setdefault(f"{form}.{kind}", {"t_ctrl": c["t_ctrl"], "t": c["t"]})[cand] = p
                    o = C.get(f"{form}.g8.{kind}.u{u}.other.w2")
                    if o and o.get("usable"):
                        rows.append((f"{form}.{kind}.other.w2", o["t_ctrl"], o["t"], o["accepted"]))
            for u in (64, 128):
                c = C.get(f"{form}.g8.dep3r.u{u}.self.w1")
                if c and c.get("usable"):
                    rows.append((f"{form}.dep3r.self.w1", c["t_ctrl"], c["t"], c["accepted"]))
        out["candidates"][cand] = verdict(rows)
    chosen = next((c for c in CANDS if out["candidates"][c]["install"]), None)
    out["decision"] = {"chosen": chosen, "level": out["candidates"][chosen]["install"] if chosen else None,
                       "summary": {k: (v["install"], round(v.get("pass_frac", 0), 3), round(v.get("max_abs_err", 0), 3)) for k, v in out["candidates"].items()},
                       "fails": out["candidates"][chosen]["fails"] if chosen else None}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out["decision"], indent=1))
    for k, v in sorted(out["table"].items()):
        print(f"{k:38s} ctrl {v['t_ctrl']:5.2f} t {v['t']:5.2f} " + " ".join(f"{c}={v[c]:.1f}" for c in CANDS if c in v))

if __name__ == "__main__":
    main()
