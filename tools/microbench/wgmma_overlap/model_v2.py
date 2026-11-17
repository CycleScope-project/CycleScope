#!/usr/bin/env python3
"""wgmma_overlap protocol v2 derivation + decision (written before data).

    model_v2.py --root <run dir> --out model.json

Variants: <form>.g<G>.<kind>.u<U>.<self|other>.w<W>, park.<kind>.u<U>.w2,
ctrl.<kind>.u<U>.w<W>.  Rows: one per (variant, launch) with samples
(active warps x reps, t_w - t_s), "tails" (t_d - max active t_w) and
"issue" (t_i1 - t_i0 of warpgroup 0).  Gate: build/sass/checks.json keyed
<variant without .w>.wb<W>.
"""
import argparse
import json
import statistics
from pathlib import Path

TOL_REL, TOL_ABS = 0.05, 0.25
SEC_REL, SEC_ABS = 0.25, 0.5
AGREE_REL, AGREE_ABS = 0.03, 0.25
OVERLAP_FAIL_MAX = 0.10
STREAMS = ["ilp8", "ilp8n", "fadd1", "fmul2"]
CHAINS = ["dep3r", "dep2", "fmnmx", "imad"]
N_OF = {"hgmma_bf16_n256_ss": 256, "hgmma_bf16_n128_ss": 128,
        "hgmma_bf16_n64_ss": 64, "hgmma_bf16_n256_rs": 256,
        "qgmma_e4m3_n256_ss": 256}
SS_FORMS = ["hgmma_bf16_n64_ss", "hgmma_bf16_n128_ss", "hgmma_bf16_n256_ss"]


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


def parse_variant(v):
    parts = v.split(".")
    w = int(parts[-1][1:])
    if parts[0] == "ctrl":
        return dict(form=None, g=0, kind=".".join(parts[1:-2]),
                    u=int(parts[-2][1:]), mode="ctrl", w=w)
    if parts[0] == "park":
        return dict(form=None, g=0, kind=".".join(parts[1:-2]),
                    u=int(parts[-2][1:]), mode="park", w=w)
    form = parts[0]; g = int(parts[1][1:]); mode = parts[-2]
    u = int(parts[-3][1:]); kind = ".".join(parts[2:-3])
    return dict(form=form, g=g, kind=kind, u=u, mode=mode, w=w)


def window(launches, u):
    meds, fails, total, issue = [], 0, 0, []
    for l, (s, tails, iss) in sorted(launches.items()):
        if l == 0 or not s:
            continue
        if tails:
            bad = sum(1 for t in tails if t <= 0)
            fails += bad; total += len(tails)
            if bad / len(tails) > OVERLAP_FAIL_MAX:
                continue
        meds.append(statistics.median(float(x) for x in s) / u)
        if iss:
            issue.append(statistics.median(float(x) for x in iss))
    if not meds:
        return None, (fails / total if total else 0.0), None
    return (statistics.median(meds), (fails / total if total else 0.0),
            statistics.median(issue) if issue else None)


def derive_run(run_dir):
    pv = {}
    for r in load(run_dir):
        l = int(r["params"].get("launch", 0))
        pv.setdefault(r["variant"], {})[l] = (
            r["samples"], r.get("tails") or [], r.get("issue") or [])
    cells = {}
    for v, launches in pv.items():
        meta = parse_variant(v)
        t, ff, iss = window(launches, meta["u"])
        cells[v] = dict(meta, t=t, overlap_fail=ff, issue=iss)
    return cells


def within(a, b, rel=TOL_REL, ab=TOL_ABS):
    return abs(a - b) <= max(rel * abs(b), ab)


def agree(vals, rel=TOL_REL, ab=TOL_ABS):
    if not vals:
        return False
    m = statistics.median(vals)
    return all(within(v, m, rel, ab) for v in vals)


def gate_key(meta):
    if meta["mode"] == "ctrl":
        return f"ctrl.{meta['kind']}.u{meta['u']}.wb{meta['w']}"
    if meta["mode"] == "park":
        return f"park.{meta['kind']}.u{meta['u']}.wb{meta['w']}"
    return (f"{meta['form']}.g{meta['g']}.{meta['kind']}.u{meta['u']}."
            f"{meta['mode']}.wb{meta['w']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    checks = {}
    cj = root / "build" / "sass" / "checks.json"
    if cj.is_file():
        checks = json.loads(cj.read_text()).get("kernels", {})
    runs = {r: derive_run(root / r) for r in ("run-a", "run-b")
            if (root / r).is_dir()}
    names = sorted(set().union(*[set(c) for c in runs.values()]))
    out = {"protocol": "wgmma_overlap v2", "root": str(root), "cells": {},
           "issue_phase": {}}

    def ctrl_w(meta):
        return meta["w"] if meta["mode"] == "self" else meta["w"] - 1

    for v in names:
        meta = next(runs[r][v] for r in runs if v in runs[r])
        if meta["mode"] == "ctrl":
            continue
        ck = f"ctrl.{meta['kind']}.u{meta['u']}.w{ctrl_w(meta)}"
        ts, tc, ff = [], [], []
        for r in runs:
            c = runs[r].get(v); cc = runs[r].get(ck)
            if c and c["t"] is not None and cc and cc["t"] is not None:
                ts.append(c["t"]); tc.append(cc["t"]); ff.append(c["overlap_fail"])
            if c and c.get("issue") is not None:
                out["issue_phase"].setdefault(v, []).append(c["issue"])
        gk, gck = gate_key(meta), f"ctrl.{meta['kind']}.u{meta['u']}.wb{ctrl_w(meta)}"
        gate_ok = True
        for kk in (gk, gck):
            if kk in checks:
                gate_ok &= bool(checks[kk].get("ok", False))
            elif checks:
                gate_ok = False
        usable = bool(ts) and gate_ok and all(f <= OVERLAP_FAIL_MAX for f in ff) \
            or (meta["mode"] == "park" and bool(ts) and gate_ok)
        cell = dict(meta, t_runs=ts, t_ctrl_runs=tc, overlap_fail=ff,
                    gate_ok=gate_ok, usable=usable,
                    accepted=len(ts) == 2 and within(ts[0], ts[1], AGREE_REL, AGREE_ABS))
        if usable:
            cell["t"] = statistics.mean(ts); cell["t_ctrl"] = statistics.mean(tc)
            cell["excess"] = cell["t"] - cell["t_ctrl"]
        out["cells"][v] = cell
    C = out["cells"]

    def get(form, g, kind, u, mode, w):
        c = C.get(f"{form}.g{g}.{kind}.u{u}.{mode}.w{w}")
        return c if c and c.get("usable") else None

    # ---- C1: issuing-warpgroup issue floor from self.w1 streams ----
    c1 = {"F_cell": {}, "kinds_agree": {}, "F_formG": {}, "refuted": [],
          "accepted_all": True}
    for form in N_OF:
        for g in (2, 4, 8):
            vals = {}
            for kind in STREAMS:
                for u in (32, 64, 128):
                    c = get(form, g, kind, u, "self", 1)
                    if c:
                        vals[kind] = c["t"]
                        c1["accepted_all"] &= c["accepted"]
            if len(vals) < 2:
                continue
            F = statistics.median(vals.values())
            ok = all(within(t, F) for t in vals.values())
            c1["F_cell"][f"{form}.g{g}"] = vals
            c1["kinds_agree"][f"{form}.g{g}"] = ok
            c1["F_formG"][f"{form}.g{g}"] = F
            if not ok:
                c1["refuted"].append(f"{form}.g{g}: stream kinds disagree {vals}")
    # G independence per form
    c1["F_form"] = {}
    for form in N_OF:
        fs = [F for k, F in c1["F_formG"].items() if k.startswith(form + ".")]
        if fs:
            c1["F_form"][form] = statistics.median(fs)
            if not agree(fs):
                c1["refuted"].append(f"{form}: F depends on G {fs}")
    # rs / qgmma vs ss at N=256
    for form in ("hgmma_bf16_n256_rs", "qgmma_e4m3_n256_ss"):
        if form in c1["F_form"] and "hgmma_bf16_n256_ss" in c1["F_form"]:
            if not within(c1["F_form"][form], c1["F_form"]["hgmma_bf16_n256_ss"]):
                c1["refuted"].append(
                    f"{form} F {c1['F_form'][form]:.2f} != ss {c1['F_form']['hgmma_bf16_n256_ss']:.2f}: source_mode / class partition needed")
    fN = {N_OF[f]: c1["F_form"][f] for f in SS_FORMS if f in c1["F_form"]}
    c1["F_N"] = fN
    c1["constant"] = agree(list(fN.values())) if fN else None
    c1["moves"] = any(C[v]["excess"] > TOL_ABS for v in C
                      if C[v].get("usable") and C[v]["mode"] == "self"
                      and C[v]["w"] == 1 and C[v]["kind"] in STREAMS)
    c1["install"] = bool(fN) and not c1["refuted"] and c1["moves"]
    c1["field"] = ("constant" if c1["constant"] else "knots(tensor_n)") if c1["install"] else None
    c1["value"] = (statistics.median(fN.values()) if c1["constant"] else fN) if c1["install"] else None
    c1["evidence"] = ("measured" if c1["accepted_all"] else "experimental") if c1["install"] else None
    out["C1"] = c1

    # ---- C2: co-resident dependent latency from other.w2 chains ----
    c2 = {"L_cell": {}, "park": {}, "refuted": []}
    for form in N_OF:
        for g in (2, 4, 8):
            vals = {}
            for kind in CHAINS:
                for u in (32, 64, 128):
                    c = get(form, g, kind, u, "other", 2)
                    if c:
                        vals[kind] = c["excess"]
            if len(vals) < 2:
                continue
            L = statistics.median(vals.values())
            c2["L_cell"][f"{form}.g{g}"] = {"L": L, "kinds": vals,
                                            "agree": all(within(e, L) for e in vals.values())}
            if not c2["L_cell"][f"{form}.g{g}"]["agree"]:
                c2["refuted"].append(f"{form}.g{g}: chain kinds disagree {vals}")
    Ls = [x["L"] for x in c2["L_cell"].values()]
    c2["L"] = statistics.median(Ls) if Ls else None
    if Ls and not agree(Ls):
        c2["refuted"].append(f"L depends on form / G: {Ls}")
    park_ok = True
    for kind in CHAINS:
        for u in (32, 64, 128):
            c = C.get(f"park.{kind}.u{u}.w2")
            if c and c.get("usable"):
                c2["park"][f"{kind}.u{u}"] = c["excess"]
                park_ok &= abs(c["excess"]) <= TOL_ABS
    c2["park_ok"] = park_ok if c2["park"] else None
    c2["moves"] = c2["L"] is not None and c2["L"] > TOL_ABS
    c2["install"] = bool(Ls) and not c2["refuted"] and c2["moves"] and bool(c2["park_ok"])
    out["C2"] = c2

    # ---- secondary: self.w2 streams vs max(t_ctrl(2), F, 2) ----
    sec = {}
    for v, c in C.items():
        if c.get("usable") and c["mode"] == "self" and c["w"] == 2 and c["kind"] in STREAMS:
            F = c1["F_form"].get(c["form"])
            if F is None:
                continue
            p = max(c["t_ctrl"], F, 2)
            sec[v] = {"pred": p, "t": c["t"], "ok": within(p, c["t"], SEC_REL, SEC_ABS)}
    out["secondary"] = sec
    out["decision"] = {
        "C0": not c1["moves"] and not c2["moves"],
        "C1": c1["install"], "C1_field": c1["field"], "C1_value": c1["value"],
        "C1_evidence": c1["evidence"], "C1_refuted": c1["refuted"],
        "C2": c2["install"], "C2_L": c2["L"], "C2_park_ok": c2["park_ok"],
        "C2_refuted": c2["refuted"],
        "secondary_ok": all(s["ok"] for s in sec.values()) if sec else None,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    d = out["decision"]
    print(json.dumps(d, indent=1))
    print("F(form,G):", {k: round(v, 2) for k, v in c1["F_formG"].items()})
    print("F(N):", {k: round(v, 2) for k, v in fN.items()})
    print("L cells:", {k: round(x["L"], 2) for k, x in c2["L_cell"].items()}, "park:", {k: round(v, 2) for k, v in c2["park"].items()})


if __name__ == "__main__":
    main()
