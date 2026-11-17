#!/usr/bin/env python3
"""memory_request v2 model (protocol-v2.json derivation/acceptance/install).

  model_v2.py --dir $A/microbench/memory_request/<device>-r2 \
              --tree configs/trees/<device>.json --device h100|b300 \
              --out $A/microbench/memory_request-model/<device>-r2/model.json

v2: k_ld is a dependent chase -> cycles_per_step = OLS slope of the
per-rep body cycles over K (steps); rate = W*D/cycles_per_step;
latency_unloaded = cycles_per_step at (W1,D1); ceiling for the cap
rule = the maximum rate_saturated over the three levels of the run.
Installs nothing.
"""
import argparse
import json
import statistics
from pathlib import Path

LEVELS = ("l1", "l2", "dram")
WD = [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1),
      (32, 2), (32, 4), (32, 8), (32, 16), (32, 32)]
REPS_PER_LAUNCH = 32
LAUNCHES_USED = 7
DRAM_REPS = 16  # reps 1..16 of 17
DRAM_LAUNCHES = 6  # launches 1..6 of 7
BANDS = {"l1": 0.15, "l2": 0.10, "dram": 0.15}


def load_rows(path):
    rows = []
    for l in Path(path).read_text().splitlines():
        if '"probe":"memq"' in l or '"probe": "memq"' in l:
            rows.append(json.loads(l))
    return rows


def samples(row):
    s = row["samples"]
    if row["variant"].startswith("ld.dram."):
        return s[DRAM_REPS:DRAM_REPS * (DRAM_LAUNCHES + 1)]
    return s[REPS_PER_LAUNCH:REPS_PER_LAUNCH * (LAUNCHES_USED + 1)]


def ols(points):
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    if not sxx:
        return None
    b = sxy / sxx
    a = my - b * mx
    res = max(abs(y - (a + b * x)) for x, y in points)
    ymax = max(y for _, y in points)
    return b, a, res, ymax


def cv(v):
    if len(v) < 2:
        return 0.0
    m = statistics.mean(v)
    return statistics.stdev(v) / m if m else 0.0


def tree_field(tree, node_suffix, field):
    for n in tree["nodes"]:
        if n["id"] == node_suffix + ".root":
            f = n["parameter"]["fields"].get(field)
            if f is not None:
                return f["model"].get("value")
    return None


def analyze_ld(rows):
    """{(lvl,W,D): cycles_per_step, rate, linear_ok, max_row_cv}"""
    out = {}
    for lvl in LEVELS:
        for w, d in WD:
            pts, cvs = [], []
            for r in rows:
                p = r.get("params", {})
                if (r["variant"].startswith(f"ld.{lvl}.")
                        and p.get("d") == d and p.get("warps") == w):
                    s = samples(r)
                    cvs.append(cv(s))
                    for y in s:
                        pts.append((r["u"], y))  # x = K steps
            if not pts:
                continue
            fit = ols(pts)
            if fit is None:
                continue
            b, a, res, ymax = fit
            out[(lvl, w, d)] = {
                "cycles_per_step": b,
                "rate": w * d / b,
                "residual_pct": res / ymax * 100 if ymax else 0,
                "linear_ok": res <= 0.02 * ymax,
                "max_row_cv": max(cvs), "n_samples": len(pts)}
    return out


def analyze_st(rows):
    out = {}
    for form in ("dist", "same"):
        for w in (1, 4, 16, 32):
            pts, cvs = [], []
            for r in rows:
                p = r.get("params", {})
                if r["variant"] == f"st.{form}.w{w}":
                    s = samples(r)
                    cvs.append(cv(s))
                    for y in s:
                        pts.append((r["u"] * w, y))
            if not pts:
                continue
            fit = ols(pts)
            if fit is None:
                continue
            b, a, res, ymax = fit
            out[(form, w)] = {
                "cycles_per_request": b, "rate": 1.0 / b,
                "residual_pct": res / ymax * 100 if ymax else 0,
                "linear_ok": res <= 0.02 * ymax,
                "max_row_cv": max(cvs), "n_samples": len(pts)}
    return out


def analyze_ack(rows):
    out = {}
    for form in ("cold", "warm"):
        s = []
        for r in rows:
            if r["variant"] == f"ack.{form}":
                s += r["samples"]
        if s:
            out[form] = {"median": statistics.median(s), "n": len(s)}
    return out


def level_metrics(ld):
    res = {}
    for lvl in LEVELS:
        pts = [(w * d, v) for (l, w, d), v in ld.items() if l == lvl]
        if not pts:
            continue
        pts.sort()
        lat = ld.get((lvl, 1, 1), {}).get("cycles_per_step")
        rates = [(n, v["rate"], v["linear_ok"], v["max_row_cv"])
                 for n, v in pts if v["linear_ok"]]
        rmax = max(r for _, r, _, _ in rates) if rates else None
        near = [r for _, r, _, _ in rates if rmax and r >= 0.97 * rmax]
        sat = statistics.median(near) if len(near) >= 2 else None
        res[lvl] = {"points": rates, "latency_unloaded": lat,
                    "rate_saturated": sat,
                    "rate_saturated_points": len(near),
                    "issue_interval": (1.0 / sat if sat else None),
                    "cap": {}}
    return res


def cap_resolve(metrics):
    """v2: ceiling = max rate_saturated over the levels of the run."""
    ceiling = max((m["rate_saturated"] for m in metrics.values()
                   if m["rate_saturated"]), default=None)
    for lvl, m in metrics.items():
        lat = m["latency_unloaded"]
        seq = sorted((n, r) for n, r, _, _ in m["points"])
        growing = [seq[i][0] for i in range(1, len(seq))
                   if seq[i][1] > 1.03 * seq[i - 1][1]]
        nstar = None
        for i, (n, r) in enumerate(seq[:-1]):
            if seq[i + 1][1] <= 1.03 * r:
                nstar = n
                break
        if nstar is None:
            nstar = seq[-1][0] if seq else None
        cap = {"verdict": "UNRESOLVED",
               "lower_bound": (max(growing) if growing
                               else (seq[0][0] if seq else None)),
               "N_star": nstar, "ceiling": ceiling}
        if lat is None or nstar is None or ceiling is None:
            m["cap"] = cap
            continue
        rate_n = next((r for n, r in seq if n == nstar), None)
        conc = rate_n * lat if rate_n else None
        cond_i = conc is not None and nstar >= 1.2 * conc
        cond_ii = rate_n is not None and rate_n <= 0.9 * ceiling
        cap.update({"concurrency_eff": conc,
                    "cond_i_offered_20pct": cond_i,
                    "cond_ii_below_ceiling": cond_ii})
        if cond_i and cond_ii:
            cap["verdict"] = "RESOLVED"
            cap["outstanding"] = round(conc)
        m["cap"] = cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tree = json.loads(Path(a.tree).read_text())
    memlat = {lvl: tree_field(tree, f"memory.{lvl}.service",
                              f"{lvl}.latency_cycles") for lvl in LEVELS}

    runs = {}
    for r in ("run-a", "run-b"):
        p = Path(a.dir) / r / "memq.jsonl"
        if not p.is_file():
            continue
        rows = load_rows(p)
        ld = analyze_ld(rows)
        m = level_metrics(ld)
        runs[r] = {"ld": ld, "metrics": m,
                   "st": analyze_st(rows), "ack": analyze_ack(rows)}
    for r in runs.values():
        cap_resolve(r["metrics"])

    out = {"device": a.device, "run_dir": a.dir,
           "memlat_installed": memlat,
           "runs": {}, "acceptance": {}, "install": {}}
    for r, q in runs.items():
        out["runs"][r] = {
            "ld_points": {f"{lvl}.w{w}.d{d}": v
                          for (lvl, w, d), v in q["ld"].items()},
            "metrics": q["metrics"],
            "st": {f"{form}.w{w}": v
                   for (form, w), v in q["st"].items()},
            "ack": q["ack"]}

    acc = {}
    lin_bad, cv_bad = [], []
    for r, q in runs.items():
        for (lvl, w, d), v in q["ld"].items():
            if not v["linear_ok"]:
                lin_bad.append(f"{r}:ld.{lvl}.d{d}.w{w}")
            if v["max_row_cv"] > 0.03:
                cv_bad.append(f"{r}:ld.{lvl}.d{d}.w{w} cv={v['max_row_cv']:.3f}")
        for (form, w), v in q["st"].items():
            if not v["linear_ok"]:
                lin_bad.append(f"{r}:st.{form}.w{w}")
            if v["max_row_cv"] > 0.03:
                cv_bad.append(f"{r}:st.{form}.w{w} cv={v['max_row_cv']:.3f}")
    acc["C2_linearity"] = {"pass": not lin_bad, "failures": lin_bad}
    acc["C1_rep_cv"] = {"pass": not cv_bad, "failures": cv_bad}
    agree = {}
    ok = True
    for lvl in LEVELS:
        for key in ("rate_saturated", "latency_unloaded"):
            va = runs.get("run-a", {}).get("metrics", {}).get(lvl, {}).get(key)
            vb = runs.get("run-b", {}).get("metrics", {}).get(lvl, {}).get(key)
            if va and vb:
                diff = abs(va - vb) / ((va + vb) / 2) * 100
                agree[f"{lvl}.{key}"] = diff
                if diff > 5:
                    ok = False
            else:
                agree[f"{lvl}.{key}"] = "missing"
                ok = False
        ca = runs.get("run-a", {}).get("metrics", {}).get(lvl, {}).get("cap", {})
        cb = runs.get("run-b", {}).get("metrics", {}).get(lvl, {}).get("cap", {})
        if ca.get("verdict") == "RESOLVED" and cb.get("verdict") == "RESOLVED":
            va, vb = ca["outstanding"], cb["outstanding"]
            diff = abs(va - vb) / ((va + vb) / 2) * 100
            agree[f"{lvl}.outstanding"] = diff
            if diff > 5:
                ok = False
        else:
            agree[f"{lvl}.outstanding"] = "not resolved on both runs"
    st_a = runs.get("run-a", {}).get("st", {})
    st_b = runs.get("run-b", {}).get("st", {})
    for key in set(st_a) & set(st_b):
        va, vb = st_a[key]["cycles_per_request"], st_b[key]["cycles_per_request"]
        diff = abs(va - vb) / ((va + vb) / 2) * 100
        agree[f"st.{key[0]}.w{key[1]}"] = diff
        if diff > 5:
            ok = False
    acc["C1_runs"] = {"pass": ok and acc["C1_rep_cv"]["pass"],
                      "diff_pct": agree}
    c3 = {}
    ok = True
    for lvl in LEVELS:
        for r in ("run-a", "run-b"):
            v = runs.get(r, {}).get("metrics", {}).get(lvl, {}).get(
                "latency_unloaded")
            inst = memlat[lvl]
            good = (v is not None and inst
                    and abs(v / inst - 1) <= BANDS[lvl])
            c3[f"{lvl}.{r}"] = {"measured": v, "installed": inst,
                                "within_band": good}
            if not good:
                ok = False
    acc["C3_control"] = {"pass": ok, "detail": c3}
    c4 = {}
    ok = True
    for r in ("run-a", "run-b"):
        v = runs.get(r, {}).get("metrics", {}).get("l1", {}).get(
            "rate_saturated")
        c4[r] = v
        if v is None or v < 0.9:
            ok = False
    acc["C4_l1_sanity"] = {"pass": ok, "l1_rate_saturated": c4}
    out["acceptance"] = acc

    gates = acc["C1_runs"]["pass"] and acc["C4_l1_sanity"]["pass"]
    inst = {}
    for lvl in LEVELS:
        m_a = runs.get("run-a", {}).get("metrics", {}).get(lvl, {})
        m_b = runs.get("run-b", {}).get("metrics", {}).get(lvl, {})
        sat = [m["rate_saturated"] for m in (m_a, m_b)
               if m.get("rate_saturated")]
        interval = statistics.mean([1.0 / s for s in sat]) if sat else None
        c3_lvl = all(acc["C3_control"]["detail"].get(f"{lvl}.{r}", {})
                     .get("within_band") for r in ("run-a", "run-b"))
        # C2 for the points feeding this level: all its ld points linear
        c2_lvl = not any(f"ld.{lvl}." in f_ for f_ in lin_bad)
        inst[f"{lvl}.issue_interval_cycles"] = {
            "value": interval,
            "install": bool(gates and c3_lvl and c2_lvl and interval),
            "evidence": "measured",
            "per_run": {r: runs.get(r, {}).get("metrics", {})
                        .get(lvl, {}).get("issue_interval")
                        for r in ("run-a", "run-b")}}
        ca, cb = m_a.get("cap", {}), m_b.get("cap", {})
        both_res = (ca.get("verdict") == "RESOLVED"
                    and cb.get("verdict") == "RESOLVED")
        agreed = (both_res and
                  isinstance(agree.get(f"{lvl}.outstanding"), float)
                  and agree[f"{lvl}.outstanding"] <= 5)
        inst[f"{lvl}.outstanding_requests"] = {
            "verdicts": {"run-a": ca, "run-b": cb},
            "install": bool(gates and c3_lvl and c2_lvl and agreed),
            "evidence": "measured" if agreed else "experimental",
            "value": (round(statistics.mean(
                [ca["outstanding"], cb["outstanding"]]))
                if both_res else None),
            "lower_bound": cb.get("lower_bound") or ca.get("lower_bound"),
            "note": None if agreed else
                    "not installed; if UNRESOLVED keep the installed "
                    "value and record 'no cap up to N in flight' in the "
                    "source (experimental)"}
    st_sat = []
    st_lin_ok = True
    for r in ("run-a", "run-b"):
        pts = [(k, v["rate"]) for k, v in runs.get(r, {})
               .get("st", {}).items() if k[0] == "dist"]
        if pts:
            rmax = max(rate for _, rate in pts)
            near = [rate for _, rate in pts if rate >= 0.97 * rmax]
            st_sat.append(statistics.median(near))
        st_lin_ok = st_lin_ok and not any(
            f"st.dist" in f_ for f_ in lin_bad)
    st_interval = (statistics.mean([1.0 / s for s in st_sat])
                   if st_sat else None)
    inst["store_l2.issue_interval_cycles"] = {
        "value": st_interval,
        "install": bool(acc["C1_runs"]["pass"] and st_lin_ok
                        and st_interval),
        "evidence": "measured",
        "per_run_rate_saturated": st_sat}
    inst["store_l2.outstanding_requests"] = {
        "install": False,
        "note": "not installable here (reported only)"}
    inst["store_l2.latency_cycles"] = {
        "install": False,
        "note": "not installed (ack is fence-covered); ack cycles "
                "reported for the store_accept protocol",
        "ack": {r: runs.get(r, {}).get("ack")
                for r in ("run-a", "run-b")}}
    out["install"] = inst
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: v["pass"] for k, v in acc.items()}, indent=1))
    print(json.dumps({k: (v.get("value"), v.get("install"))
                      for k, v in inst.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
