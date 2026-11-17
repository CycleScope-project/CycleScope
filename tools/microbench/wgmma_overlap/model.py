#!/usr/bin/env python3
"""wgmma_overlap protocol v1 derivation + decision (written before data).

    model.py --root <run dir> --out model.json [--tree <h100.json>]

<run dir> holds run-a/ and run-b/ overlap.jsonl (one row per variant and
launch; samples = per active warp per rep t_w - t_s; extra "tails" =
warpgroup 0's t_d - t_w) and build/sass/checks.json.

Protocol text is authoritative; the closed forms below are copied from
engine_rules_declared_before_data:
  R0 : t = t_ctrl
  R-F: t = max(t_ctrl(W), 1 + k, W)                 k in {1, 2, 3}
  R-P: t = max(t_ctrl(W), W * n / r)                r in {1, 0.5}
  R-S: t = max(t_ctrl(W), 2 W)
with n = unserved GPR reads of the window kind (dep3r 1, dep2 2, fmnmx 2,
ilp8n 3, ilp8 1, imad 3, mufu.ex2 1, uadd 0).  Gating cells: self.w1 (all
U) and other.w2; tolerance max(5 %, 0.25 cycles per instruction).
Secondary cells (self.w2 / self.w3 / other.w3): direction + 25 % or 0.5.
"""
import argparse
import json
import statistics
from pathlib import Path

TOL_REL, TOL_ABS = 0.05, 0.25
SEC_REL, SEC_ABS = 0.25, 0.5
AGREE_REL, AGREE_ABS = 0.03, 0.25
OVERLAP_FAIL_MAX = 0.10
CTRL_REPRO_REL = 0.10
# dispatch_contention v4 baselines (h100) for the shared kinds at W = 1..3
CTRL_REF = {("dep3r", 1): 4.0, ("dep3r", 2): 5.0, ("dep3r", 3): 6.0,
            ("ilp8n", 1): 1.5, ("ilp8", 1): 1.0, ("imad", 1): 4.0}
UNSERVED = {"dep3r": 1, "dep2": 2, "fmnmx": 2, "ilp8n": 3, "ilp8": 1,
            "imad": 3, "mufu.ex2": 1, "uadd": 0}
FORMS = ["hgmma_bf16_n256_ss", "hgmma_bf16_n256_rs", "hgmma_bf16_n64_ss",
         "qgmma_e4m3_n256_ss"]
KINDS = list(UNSERVED)


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
    """-> dict(form, kind, u, mode, w) ; ctrl rows have form None."""
    parts = v.split(".")
    if parts[0] == "ctrl":
        # ctrl.<kind>.u<U>.w<W>  (kind may contain a dot: mufu.ex2)
        w = int(parts[-1][1:]); u = int(parts[-2][1:])
        kind = ".".join(parts[1:-2])
        return dict(form=None, kind=kind, u=u, mode="ctrl", w=w)
    # <form>.<kind>.u<U>.<mode>.w<W>
    w = int(parts[-1][1:]); mode = parts[-2]; u = int(parts[-3][1:])
    form = parts[0]
    kind = ".".join(parts[1:-3])
    return dict(form=form, kind=kind, u=u, mode=mode, w=w)


def per_variant(rows):
    """variant -> {launch: (samples, tails)}"""
    out = {}
    for r in rows:
        v = r["variant"]; l = int(r["params"].get("launch", 0))
        out.setdefault(v, {})[l] = (r["samples"], r.get("tails") or [])
    return out


def window(launches, u):
    """T per instruction = median over launches>=1 of per-launch median
    of samples / u; overlap-failure fraction from tails (<= 0)."""
    meds, fails, total = [], 0, 0
    for l, (s, tails) in sorted(launches.items()):
        if l == 0 or not s:
            continue
        if tails:
            bad = sum(1 for t in tails if t <= 0)
            fails += bad; total += len(tails)
            if bad / len(tails) > OVERLAP_FAIL_MAX:
                continue  # unusable launch
        meds.append(statistics.median(float(x) for x in s) / u)
    if not meds:
        return None, (fails / total if total else 0.0)
    return statistics.median(meds), (fails / total if total else 0.0)


def derive_run(run_dir):
    pv = per_variant(load(run_dir))
    cells = {}
    for v, launches in pv.items():
        meta = parse_variant(v)
        t, ff = window(launches, meta["u"])
        cells[v] = dict(meta, t=t, overlap_fail=ff)
    return cells


def ctrl_key(kind, u, w):
    return f"ctrl.{kind}.u{u}.w{w}"


def predictions(kind, w, t_ctrl):
    n = UNSERVED[kind]
    p = {"R0": t_ctrl}
    for k in (1, 2, 3):
        p[f"R-F(k={k})"] = max(t_ctrl, 1 + k, w)
    for r in (1.0, 0.5):
        p[f"R-P(r={r:g})"] = max(t_ctrl, w * n / r)
    p["R-S"] = max(t_ctrl, 2 * w)
    return p


def within(pred, meas, rel, ab):
    return abs(pred - meas) <= max(rel * meas, ab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    checks = {}
    cj = root / "build" / "sass" / "checks.json"
    if cj.is_file():
        checks = json.loads(cj.read_text())
    runs = {r: derive_run(root / r) for r in ("run-a", "run-b")
            if (root / r).is_dir()}
    out = {"protocol": "wgmma_overlap v1", "root": str(root),
           "cells": {}, "ctrl_reproduction": {}, "decision": None}
    # control reproduction (harness fault check)
    repro_ok = True
    for (kind, w), ref in CTRL_REF.items():
        key = ctrl_key(kind, 128, w)
        vals = [runs[r][key]["t"] for r in runs if key in runs[r]
                and runs[r][key]["t"] is not None]
        if not vals:
            continue
        ok = all(abs(v - ref) <= CTRL_REPRO_REL * ref for v in vals)
        out["ctrl_reproduction"][key] = {"ref": ref, "runs": vals, "ok": ok}
        repro_ok &= ok
    out["ctrl_reproduction_ok"] = repro_ok
    # per overlap cell: t per run, excess per run, accepted, usable
    gating, secondary = [], []
    for v in sorted(set().union(*[set(c) for c in runs.values()])):
        meta = next(runs[r][v] for r in runs if v in runs[r])
        if meta["mode"] == "ctrl":
            continue
        w_ctrl = meta["w"] - 1 if meta["mode"] == "other" else meta["w"]
        ck = ctrl_key(meta["kind"], meta["u"], w_ctrl)
        ts, es, tc, ff = [], [], [], []
        for r in runs:
            c = runs[r].get(v); cc = runs[r].get(ck)
            if not c or c["t"] is None or not cc or cc["t"] is None:
                continue
            ts.append(c["t"]); tc.append(cc["t"]); es.append(c["t"] - cc["t"])
            ff.append(c["overlap_fail"])
        gate_ok = True
        kkey = f"{meta['form']}.{meta['kind']}.u{meta['u']}.{meta['mode']}.wb{meta['w']}"
        ckkey = f"ctrl.{meta['kind']}.u{meta['u']}.wb{w_ctrl}"
        kern = checks.get("kernels", {})
        for kk in (kkey, ckkey):
            if kk in kern:
                gate_ok &= bool(kern[kk].get("ok", False))
            elif kern:
                gate_ok = False  # kernel missing from the gate output
        usable = bool(ts) and gate_ok and all(f <= OVERLAP_FAIL_MAX for f in ff)
        accepted = len(ts) == 2 and within(ts[0], ts[1], AGREE_REL, AGREE_ABS)
        cell = dict(meta, t_runs=ts, t_ctrl_runs=tc, excess_runs=es,
                    overlap_fail=ff, gate_ok=gate_ok, usable=usable,
                    accepted=accepted)
        if usable:
            t = statistics.mean(ts); t0 = statistics.mean(tc)
            cell["t"] = t; cell["t_ctrl"] = t0; cell["excess"] = t - t0
            cell["pred"] = predictions(meta["kind"], w_ctrl, t0)
            cell["err"] = {k: p - t for k, p in cell["pred"].items()}
            is_gating = (meta["mode"] == "self" and meta["w"] == 1) or \
                        (meta["mode"] == "other" and meta["w"] == 2)
            cell["gating"] = is_gating
            (gating if is_gating else secondary).append(v)
        out["cells"][v] = cell
    # decision: first candidate in order fitting every usable gating cell
    order = ["R0"] + [f"R-F(k={k})" for k in (1, 2, 3)] + \
            [f"R-P(r={r:g})" for r in (1.0, 0.5)] + ["R-S"]
    family_order = ["R0", "R-F", "R-P", "R-S"]
    fits = {}
    for cand in order:
        errs = [(v, out["cells"][v]["pred"][cand], out["cells"][v]["t"])
                for v in gating]
        ok = all(within(p, t, TOL_REL, TOL_ABS) for _, p, t in errs)
        maxerr = max((abs(p - t) for _, p, t in errs), default=0.0)
        fits[cand] = {"gating_ok": ok, "max_abs_err": maxerr,
                      "misses": [v for v, p, t in errs
                                 if not within(p, t, TOL_REL, TOL_ABS)]}
    chosen = None
    for fam in family_order:
        cands = [c for c in order if c.split("(")[0] == fam and fits[c]["gating_ok"]]
        if cands:
            chosen = min(cands, key=lambda c: fits[c]["max_abs_err"])
            break
    sec = {}
    if chosen:
        for v in secondary:
            c = out["cells"][v]
            p = c["pred"][chosen]; t = c["t"]; t0 = c["t_ctrl"]
            direction_ok = (p - t0) * (t - t0) >= 0 or abs(t - t0) <= TOL_ABS
            sec[v] = {"pred": p, "t": t, "ok": direction_ok and
                      within(p, t, SEC_REL, SEC_ABS)}
    # scope: other.w2 vs self.w1 agreement is embedded in gating (other.w2
    # cells are gating); report explicitly
    scope = {}
    for v in gating:
        c = out["cells"][v]
        if c["mode"] == "other":
            s1 = f"{c['form']}.{c['kind']}.u{c['u']}.self.w1"
            if s1 in out["cells"] and out["cells"][s1].get("usable"):
                e_other, e_self = c["excess"], out["cells"][s1]["excess"]
                scope[v] = {"excess_other": e_other, "excess_self": e_self,
                            "sm_scope": within(e_other, e_self, TOL_REL, TOL_ABS)
                            or (abs(e_other) <= TOL_ABS and abs(e_self) <= TOL_ABS)}
    all_accepted = all(out["cells"][v]["accepted"] for v in gating) if gating else False
    out["fits"] = fits
    out["decision"] = {
        "gating_cells": len(gating), "secondary_cells": len(secondary),
        "chosen": chosen,
        "evidence": None if not chosen else ("measured" if all_accepted else "experimental"),
        "secondary": sec,
        "secondary_ok": all(s["ok"] for s in sec.values()) if sec else None,
        "scope": scope,
        "sm_scope": all(s["sm_scope"] for s in scope.values()) if scope else None,
        "note": "no candidate fits every gating cell: record only, protocol v2 before any further GPU run" if not chosen else None,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    d = out["decision"]
    print(f"gating {d['gating_cells']} secondary {d['secondary_cells']} "
          f"ctrl_repro_ok={repro_ok} chosen={chosen} evidence={d['evidence']} "
          f"sm_scope={d['sm_scope']} secondary_ok={d['secondary_ok']}")
    for cand in order:
        f = fits[cand]
        print(f"  {cand:12s} gating_ok={f['gating_ok']} max_err={f['max_abs_err']:.2f} misses={len(f['misses'])}")


if __name__ == "__main__":
    main()
