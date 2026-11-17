#!/usr/bin/env python3
"""tmem protocol v1 derivation.

    model.py --root <run dir> --out model.json

<run dir> holds run-a/ and run-b/ ldtm.jsonl and build/sass/checks.json.
Per (shape, N): E = median over launches (launch 0 dropped) of the
per-launch median window (rep 0 dropped on the device side).  Slope =
OLS of E over N in {1, 2, 4, 8}.

Acceptance (protocol.json): x64 slope must reproduce 32.0 and x32 slope
16.0 within 10 % on both runs; else nothing is realised
(unrealised:control).  With controls passing: x128 OLS max relative
residual <= 5 % and run agreement <= 5 % -> measured; run agreement
<= 10 % -> experimental; else unrealised.
"""
import argparse
import json
import statistics
from pathlib import Path

NS = [1, 2, 4, 8]
CONTROL = {"32x32b.x64": 32.0, "32x32b.x32": 16.0}
RESID = 0.05
AGREE_MEASURED = 0.05
AGREE_EXPERIMENTAL = 0.10
CONTROL_TOL = 0.10


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return b, my - b * mx


def load_run(run_dir):
    """shape -> n -> {launch_idx -> median window}; errors noted."""
    per = {}
    errs = []
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = d.get("params", {})
            shape = p.get("shape")
            n = int(p.get("n", d.get("u", 0)))
            if "error" in d:
                errs.append({"variant": d.get("variant"), "n": n,
                             "launch": p.get("launch"),
                             "error": d["error"]})
                continue
            if "samples" not in d or shape is None:
                continue
            if d["samples"]:
                per.setdefault(shape, {}).setdefault(n, {})[
                    int(p.get("launch", -1))] = statistics.median(
                        float(x) for x in d["samples"])
    return per, errs


def slope_for(shape_map):
    """E per N (median over launches, launch 0 dropped) + OLS."""
    es = {}
    for n, launches in shape_map.items():
        meds = [v for k, v in sorted(launches.items()) if k != 0]
        if meds:
            es[n] = statistics.median(meds)
    xs = [n for n in NS if n in es]
    if len(xs) < 2:
        return {"E": es, "n_points": len(xs)}
    ys = [es[n] for n in xs]
    b, a = ols(xs, ys)
    resid = max(abs(m - (a + b * x)) / max(1.0, abs(m))
                for x, m in zip(xs, ys))
    return {"E": es, "n_points": len(xs), "slope": b, "intercept": a,
            "max_rel_resid": resid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", default=None,
                    help="override path to sass gate json")
    a = ap.parse_args()
    checks = {}
    cands = ([Path(a.checks)] if a.checks else []) + [
        Path(a.root) / "build" / "sass" / cn
        for cn in ("checks-local.json", "checks.json")]
    for cpath in cands:
        if cpath.exists():
            checks = json.loads(cpath.read_text())
            break
    runs = {}
    errors = {}
    for run in ("run-a", "run-b"):
        per, errs = load_run(Path(a.root) / run)
        errors[run] = errs
        runs[run] = {s: slope_for(m) for s, m in per.items()}
    out = {"schema": "mechtree.model.v1", "probe": "tmem",
           "root": a.root, "sass_gate": checks.get("kernels", {}),
           "sass_gate_ok": checks.get("n_ok", 0),
           "sass_gate_fail": checks.get("n_fail", 0),
           "errors": errors, "runs": {}, "controls": {}, "verdict": {}}
    # per-shape table
    shapes = sorted({s for r in runs.values() for s in r})
    for s in shapes:
        out["runs"][s] = {run: runs[run].get(s) for run in runs}
    # controls must pass on both runs
    ctl_ok = True
    for shape, want in CONTROL.items():
        c = {"want": want}
        for run in runs:
            fit = runs[run].get(shape) or {}
            slope = fit.get("slope")
            c[run] = {"slope": slope,
                      "E": fit.get("E"),
                      "rel_err": (abs(slope - want) / want
                                  if slope else None)}
            if slope is None or abs(slope - want) / want > CONTROL_TOL:
                ctl_ok = False
        c["ok"] = all(c[run]["rel_err"] is not None and
                      c[run]["rel_err"] <= CONTROL_TOL for run in runs)
        out["controls"][shape] = c
    # x128 verdict
    xa = runs["run-a"].get("32x32b.x128") or {}
    xb = runs["run-b"].get("32x32b.x128") or {}
    sa, sb = xa.get("slope"), xb.get("slope")
    verdict = {"slope_run_a": sa, "slope_run_b": sb,
               "max_rel_resid_a": xa.get("max_rel_resid"),
               "max_rel_resid_b": xb.get("max_rel_resid"),
               "n_points_a": xa.get("n_points"), "n_points_b": xb.get("n_points")}
    if not ctl_ok:
        verdict["grade"] = "unrealised:control"
    elif sa is None or sb is None:
        verdict["grade"] = "unrealised:missing"
    else:
        agree = abs(sa - sb) / max(1.0, (abs(sa) + abs(sb)) / 2)
        verdict["run_agreement"] = agree
        resid = max(xa.get("max_rel_resid") or 1,
                    xb.get("max_rel_resid") or 1)
        if resid <= RESID and agree <= AGREE_MEASURED:
            verdict["grade"] = "measured"
            verdict["value"] = (sa + sb) / 2
        elif agree <= AGREE_EXPERIMENTAL:
            verdict["grade"] = "experimental"
            verdict["value"] = (sa + sb) / 2
        else:
            verdict["grade"] = "unrealised"
    out["verdict"]["ldtm_x128_issue_interval"] = verdict

    # ---- r2 x64 cell (protocol addendum): S = ld_wc slope over U,
    # T = ld_thru slope over K, I = T/2, C = S - T/2 ------------------
    x64 = {}
    if any("16x32bx2" in s for r in runs.values() for s in r):
        wc = {"run-a": {}, "run-b": {}}
        th = {"run-a": {}, "run-b": {}}
        for run in runs:
            for sh, fit in runs[run].items():
                for kind, mp in (("ld_wc", wc), ("ld_thru", th)):
                    pass
        # load_run keyed shape only; rerun with variant-aware split
        def split(run_dir):
            per = {}
            for jl in Path(run_dir).glob("*.jsonl"):
                for line in jl.read_text().splitlines():
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    p = d.get("params", {})
                    shape = p.get("shape")
                    if shape is None or "samples" not in d or not d["samples"]:
                        continue
                    n = int(p.get("n", d.get("u", 0)))
                    key = (d.get("variant", ""), shape)
                    per.setdefault(key, {}).setdefault(n, {})[
                        int(p.get("launch", -1))] = statistics.median(
                            float(x) for x in d["samples"])
            return per
        def slope_map(m, kind, shape, xs_all):
            es = {}
            for n, launches in m.get((kind, shape), {}).items():
                meds = [v for k, v in sorted(launches.items()) if k != 0]
                if meds:
                    es[n] = statistics.median(meds)
            xs = [x for x in xs_all if x in es]
            if len(xs) < 2:
                return {"E": es, "n_points": len(xs)}
            ys = [es[n] for n in xs]
            b, a_ = ols(xs, ys)
            resid = max(abs(y - (a_ + b * x)) / max(1.0, abs(y))
                        for x, y in zip(xs, ys))
            return {"E": es, "n_points": len(xs), "slope": b,
                    "max_rel_resid": resid}
        X64K = [1, 2, 3]
        X_U = [1, 2, 4, 8]
        controls = {
            "16x32bx2.x16": 8.17857, "16x32bx2.x32": 16.0,
            "32x32b.x16": 8.357, "32x32b.x32": 16.0}
        for run in ("run-a", "run-b"):
            m = split(Path(a.root) / run)
            x64[run] = {}
            for sh in controls | {"16x32bx2.x64": None}:
                wc_fit = slope_map(m, "ld_wc", sh, X_U)
                ks = X64K if sh == "16x32bx2.x64" else X_U
                th_fit = slope_map(m, "ld_thru", sh, ks)
                S, T = wc_fit.get("slope"), th_fit.get("slope")
                # SASS instructions per PTX tcgen05.ld: 16x32bx2 splits
                # into t0_t15 + t16_t31 (2), 32x32b is one
                div = 2 if sh.startswith("16x32bx2") else 1
                I = T / div if T is not None else None
                C = (S - T / 2) if (S is not None and T is not None) else None
                x64[run][sh] = {"S": wc_fit, "T": th_fit,
                                "I": I, "C": C}
        ctl_ok_x = True
        for sh, want in controls.items():
            c = {"want": want}
            # installed blackwell.tmem.load cycles for the C control:
            # 16x32bx2 -> C = S - T/2 vs the 16dp32bit leaf; 32x32b -> S
            # vs the top-level leaf (that family's installed cycles are
            # the chain slope itself)
            want_c = {"16x32bx2.x16": 18.284472049689438,
                      "16x32bx2.x32": 27.002173913043478,
                      "32x32b.x16": 18.284472049689438,
                      "32x32b.x32": 27.002173913043478}[sh]
            for run in ("run-a", "run-b"):
                I = x64[run][sh]["I"]
                rel = abs(I - want) / want if I is not None else None
                cyc = (x64[run][sh]["C"] if sh.startswith("16x32bx2")
                       else x64[run][sh]["S"].get("slope"))
                c_rel = (abs(cyc - want_c) / want_c
                         if cyc is not None else None)
                c[run] = {"I": I, "C": x64[run][sh]["C"],
                          "cyc_vs_installed": cyc, "c_rel_err": c_rel,
                          "rel_err": rel}
                if rel is None or rel > CONTROL_TOL or                         c_rel is None or c_rel > CONTROL_TOL:
                    ctl_ok_x = False
            c["ok"] = all(c[r]["rel_err"] is not None and
                          c[r]["rel_err"] <= CONTROL_TOL and
                          c[r]["c_rel_err"] is not None and
                          c[r]["c_rel_err"] <= CONTROL_TOL
                          for r in ("run-a", "run-b"))
            x64.setdefault("controls", {})[sh] = c
        va, vb = x64["run-a"]["16x32bx2.x64"], x64["run-b"]["16x32bx2.x64"]
        v = {"I_a": va["I"], "I_b": vb["I"],
             "C_a": va["C"], "C_b": vb["C"],
             "S_a": va["S"].get("slope"), "T_a": va["T"].get("slope"),
             "resid_a": va["T"].get("max_rel_resid"),
             "resid_b": vb["T"].get("max_rel_resid")}
        if not ctl_ok_x:
            v["grade"] = "unrealised:control"
        elif va["I"] is None or vb["I"] is None or va["C"] is None                 or vb["C"] is None:
            v["grade"] = "unrealised:missing"
        else:
            i_agree = abs(va["I"] - vb["I"]) / max(
                1.0, (va["I"] + vb["I"]) / 2)
            c_agree = abs(va["C"] - vb["C"]) / max(
                1.0, (va["C"] + vb["C"]) / 2)
            resid = max(v["resid_a"] or 1, v["resid_b"] or 1)
            s_ok = va["S"].get("slope") and va["T"].get("slope")                 and va["S"]["slope"] > va["T"]["slope"]
            v["I_agree"] = i_agree
            v["C_agree"] = c_agree
            v["S_gt_T"] = bool(s_ok)
            if s_ok and resid <= RESID and                     i_agree <= AGREE_MEASURED and                     c_agree <= AGREE_MEASURED:
                v["grade"] = "measured"
            elif s_ok and i_agree <= AGREE_EXPERIMENTAL and                     c_agree <= AGREE_EXPERIMENTAL:
                v["grade"] = "experimental"
            else:
                v["grade"] = "unrealised"
            if v["grade"] in ("measured", "experimental"):
                v["issue_interval_cycles"] = (va["I"] + vb["I"]) / 2
                v["load_cycles"] = (va["C"] + vb["C"]) / 2
        x64["verdict"] = v
        out["x64"] = x64
        # install payload (install.py consumes model json like r5)
        if v.get("grade") in ("measured", "experimental"):
            out["install"] = out.get("install", {})
            out["install"]["x64"] = {
                "instruction.service.LDTM.16dp32bit.tmem_repeat.x64.issue_interval": v["issue_interval_cycles"],
                "blackwell.tmem.load.16dp32bit.x64.cycles": v["load_cycles"],
                "source": "tmem.b300.r2.x64",
                "evidence": v["grade"]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(verdict, indent=1))
    if "x64" in out:
        print(json.dumps(out["x64"]["verdict"], indent=1))
    for shape, c in out["controls"].items():
        print(f"control {shape}: want {c['want']} "
              f"a={c['run-a']['slope']} b={c['run-b']['slope']} ok={c['ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
