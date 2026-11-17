#!/usr/bin/env python3
"""wgmma_overlap protocol v3 decision (C3 / C4), written before the r5 data.

    model_v3.py --root <run dir> --out model.json

Cells, loading and per-cell T / excess / usable come from model_v2.py
(same probe, same variants).  C3: t = max(t_ctrl, 2, n) for the issuing
warpgroup's streams (n = distinct GPR sources: fadd1 1, fmul2 2, ilp8n 3,
ilp8 3); chains and other warpgroups unchanged.  C4: other.w2 chains
t_ctrl + 0.5.  Tolerance max(15 %, 0.5); install measured at 100 % of
gating cells, experimental at >= 80 % with no cell beyond 1.0 cycle.
"""
import argparse
import json
import statistics
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("model_v2", HERE / "model_v2.py")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)

TOL_REL, TOL_ABS, HARD_ABS = 0.15, 0.5, 1.0
SEC_REL, SEC_ABS = 0.25, 0.75
N_READS = {"fadd1": 1, "fmul2": 2, "ilp8n": 3, "ilp8": 3}
STREAMS = list(N_READS)
CHAINS = ["dep3r", "dep2", "fmnmx", "imad"]
SS_GATING = {"hgmma_bf16_n256_ss": (2, 4, 8), "hgmma_bf16_n128_ss": (4, 8),
             "hgmma_bf16_n64_ss": (8,)}
ALL_FORMS = list(m2.N_OF)
L4 = 0.5


def within(p, t, rel=TOL_REL, ab=TOL_ABS):
    return abs(p - t) <= max(rel * abs(t), ab)


def cells_of(root):
    """Reuse model_v2's per-cell derivation by running its main into a
    temporary json and reading the cells back."""
    import subprocess, sys, tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "m2.json"
        subprocess.run([sys.executable, str(HERE / "model_v2.py"), "--root",
                        str(root), "--out", str(out)], check=True,
                       stdout=subprocess.DEVNULL)
        return json.loads(out.read_text())


def verdict(rows):
    """rows: list of (name, pred, t, accepted) -> dict with install level."""
    if not rows:
        return {"n": 0, "install": None}
    ok = [within(p, t) for _, p, t, _ in rows]
    hard = [abs(p - t) <= HARD_ABS for _, p, t, _ in rows]
    frac = sum(ok) / len(ok)
    if all(ok) and all(a for *_, a in rows):
        level = "measured"
    elif frac >= 0.8 and all(hard):
        level = "experimental"
    else:
        level = None
    return {"n": len(rows), "pass_frac": frac, "max_abs_err": max(abs(p - t) for _, p, t, _ in rows),
            "install": level,
            "fails": [{"cell": n, "pred": p, "t": t} for (n, p, t, _), o in zip(rows, ok) if not o]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = cells_of(a.root)
    C = m["cells"]
    out = {"protocol": "wgmma_overlap v3", "root": a.root, "cells": C,
           "issue_phase": m.get("issue_phase")}

    def g(form, G, kind, u, mode, w):
        c = C.get(f"{form}.g{G}.{kind}.u{u}.{mode}.w{w}")
        return c if c and c.get("usable") else None

    # ---- C3 ----
    rows, report = [], {}
    for form in ALL_FORMS:
        for G in (2, 4, 8):
            for kind in STREAMS:
                for u in (32, 64, 128):
                    c = g(form, G, kind, u, "self", 1)
                    if not c:
                        continue
                    p = max(c["t_ctrl"], 2, N_READS[kind])
                    report[f"{form}.g{G}.{kind}"] = {"t_ctrl": c["t_ctrl"], "t": c["t"], "pred": p, "err": c["t"] - p}
                    if form in SS_GATING and G in SS_GATING[form]:
                        rows.append((f"{form}.g{G}.{kind}.self.w1", p, c["t"], c["accepted"]))
            # scope: other.w2 streams unchanged; floor: self.w1 chains unchanged
            for kind in STREAMS:
                for u in (32, 64, 128):
                    c = g(form, G, kind, u, "other", 2)
                    if c:
                        rows.append((f"{form}.g{G}.{kind}.other.w2", c["t_ctrl"], c["t"], c["accepted"]))
            for kind in CHAINS:
                for u in (32, 64, 128):
                    c = g(form, G, kind, u, "self", 1)
                    if c:
                        rows.append((f"{form}.g{G}.{kind}.self.w1", c["t_ctrl"], c["t"], c["accepted"]))
    c3 = verdict(rows)
    c3["stream_report"] = report
    c3["rs_qgmma_residual"] = {k: v["err"] for k, v in report.items()
                               if k.startswith("hgmma_bf16_n256_rs") or k.startswith("qgmma")}
    out["C3"] = c3

    # ---- C4 ----
    rows4, rows4_n256, park = [], [], {}
    for form in ALL_FORMS:
        for G in (2, 4, 8):
            for kind in CHAINS:
                for u in (32, 64, 128):
                    c = g(form, G, kind, u, "other", 2)
                    if not c:
                        continue
                    r = (f"{form}.g{G}.{kind}.other.w2", c["t_ctrl"] + L4, c["t"], c["accepted"])
                    rows4.append(r)
                    if m2.N_OF[form] == 256:
                        rows4_n256.append(r)
    park_ok = True
    for kind in CHAINS:
        for u in (32, 64, 128):
            c = C.get(f"park.{kind}.u{u}.w2")
            if c and c.get("usable"):
                park[f"{kind}.u{u}"] = c["excess"]
                park_ok &= abs(c["excess"]) <= TOL_ABS
    v_all = verdict(rows4); v_256 = verdict(rows4_n256)
    c4 = {"all": v_all, "n256": v_256, "park": park, "park_ok": park_ok}
    if not park_ok or v_256["install"] is None:
        c4["install"] = None
    else:
        c4["install"] = "measured" if v_all["install"] == "measured" else "experimental"
    out["C4"] = c4

    # ---- secondary: self.w2 streams ----
    sec = {}
    for v, c in C.items():
        if c.get("usable") and c["mode"] == "self" and c["w"] == 2 and c["kind"] in STREAMS:
            p = max(c["t_ctrl"], 2, N_READS[c["kind"]])
            sec[v] = {"pred": p, "t": c["t"], "ok": within(p, c["t"], SEC_REL, SEC_ABS)}
    out["secondary"] = sec
    # ---- C0 ----
    all_e = [abs(c["excess"]) for c in C.values() if c.get("usable") and c["mode"] in ("self", "other")]
    out["decision"] = {
        "C0": bool(all_e) and max(all_e) <= TOL_ABS,
        "C3": c3["install"], "C3_pass_frac": c3.get("pass_frac"), "C3_max_err": c3.get("max_abs_err"),
        "C3_fails": c3.get("fails"),
        "C4": c4["install"], "C4_n256": v_256.get("install"), "C4_all": v_all.get("install"),
        "C4_park_ok": park_ok,
        "secondary_ok": all(s["ok"] for s in sec.values()) if sec else None,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out["decision"], indent=1))
    print("C3 stream report (t vs pred):")
    for k, v in sorted(report.items()):
        print(f"  {k:36s} ctrl {v['t_ctrl']:.2f} t {v['t']:.2f} pred {v['pred']:.0f} err {v['err']:+.2f}")


if __name__ == "__main__":
    main()
