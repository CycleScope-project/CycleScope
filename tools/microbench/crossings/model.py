#!/usr/bin/env python3
"""crossings protocol-v2 derivation
(tools/microbench/crossings/protocol-v2.json).

  model.py --device h100|b300 --root <$A/microbench/crossings/<device>-<run>>
           --out <model.json>

<root>/<cell>/{run-a,run-b}/*.jsonl are the two runs of every cell and
<root>/<cell>/checks.json its SASS gate (crossings/run.sh layout); smport
rows come from store_dram_port/<run>/ (and l1_bandwidth/<run>/smport/ if
present).
Thresholds are copied from the frozen protocol text; do not edit them
after data exists.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# --- thresholds copied verbatim from protocol-v2.json ---------------------
FIT_RESID = 0.05       # "fit residual <= 5 %" (trywait, constant_cache)
RUN_AGREE_5 = 0.05     # "two runs within 5 %"
RUN_AGREE_3 = 0.03     # l1_bandwidth / store_dram_port "runs within 3 %"
SE_3 = 0.03            # descriptor_refetch "se <= 3 %"
KNEE_TOL = 0.10        # l1_residency "within 10 % of the L1 hit latency"
LINE_TOL = 0.05        # constant_cache "equals the hit plateau within 5 %"
SMPORT_PLATEAU = 0.03  # "plateau confirmed by the two alternative cells within 3 %"
B300_L1_BW = 127.7     # protocol acceptance control values
B300_L1_INTERVAL = 0.97
B300_CTRL_TOL = 0.10   # "b300 control within 10 %"
FLOOR_TOL = 0.05       # memq_rules.FLOOR_TOL: period within 5 % of 1 -> 0
MEMLAT_L1 = {"h100": 52.5, "b300": 38.46875}  # installed memory.l1.service latency (memlat protocol)

CELLS = ("trywait_timeout", "constant_cache", "l1_residency",
         "descriptor_refetch", "l1_bandwidth", "store_dram_port")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def se(xs):
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("inf")


def ols(xs, ys):
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0,
            my - (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0) * mx)


def slope_fit(series):
    """series: u -> [samples].  OLS of the per-u medians; returns
    slope/intercept/max relative residual, or None."""
    us = sorted(series)
    if len(us) < 2:
        return None
    meds = [statistics.median(series[u]) for u in us]
    b, a = ols(us, meds)
    resid = max(abs(m - (a + b * u)) / max(1.0, abs(m)) for u, m in zip(us, meds))
    return {"slope": b, "intercept": a, "max_rel_resid": resid,
            "us": us, "medians": meds}


def rel(a, b):
    m = (a + b) / 2
    return abs(a - b) / m if m else float("inf")


def load_rows(cell_dir):
    """variant -> u -> [samples] merged over every jsonl under cell_dir."""
    rows = {}
    d = Path(cell_dir)
    if not d.exists():
        return rows
    for jl in sorted(d.glob("*.jsonl")):
        for line in jl.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "samples" not in r or "variant" not in r:
                continue
            rows.setdefault(r["variant"], {}).setdefault(int(r["u"]), []).extend(
                float(x) for x in r["samples"])
    return rows


def load_cell(run_dir, cell, run_tag):
    """rows of one cell for one run: <root>/<cell>/<run_tag>/*.jsonl."""
    return load_rows(Path(run_dir) / cell / run_tag)


def load_smport(run_dir, run_tag):
    """mem --cells smport rows: l1_bandwidth/<run>/smport/ plus
    store_dram_port/<run>/."""
    rows = {}
    for d in (Path(run_dir) / "l1_bandwidth" / run_tag / "smport",
              Path(run_dir) / "store_dram_port" / run_tag):
        for v, us in load_rows(d).items():
            for u, s in us.items():
                rows.setdefault(v, {}).setdefault(u, []).extend(s)
    return rows


def load_checks(checks_root, cell):
    if not checks_root:
        return {}
    p = Path(checks_root) / cell / "checks.json"
    return json.loads(p.read_text()) if p.exists() else {}


def gate_ok(checks, probe, key):
    k = checks.get(probe, {}).get("kernels", {}).get(key)
    return k is not None and k.get("ok")


# ---------------------------------------------------------------- cells

def derive_trywait(rows, checks):
    """period(variant) = OLS slope over N of the median loop cycles.
    cycles = period(try.h0) - period(test) + iter_stall (issue cycles of
    one test_wait iteration from the SASS control words)."""
    out = {"periods": {}, "release": {}}
    fits = {}
    for v in ("test", "try.h0", "try.h1000", "try.h100000"):
        f = slope_fit(rows.get(v, {}))
        if f:
            fits[v] = f
            out["periods"][v] = {"slope": f["slope"], "resid": f["max_rel_resid"],
                                 "medians": dict(zip(f["us"], f["medians"]))}
    stalls = [k["iter_stall"] for k in checks.get("trywait", {}).get("kernels", {}).values()
              if k.get("mode") == 0 and k.get("iter_stall") is not None]
    iter_stall = mean(stalls) if stalls else None
    out["iter_stall_test_wait"] = iter_stall
    for v in ("release.d2000", "release.d20000"):
        s = rows.get(v, {}).get(1)
        s0 = rows.get(v + ".minus_t_s0", {}).get(1)
        if s:
            out["release"][v] = {"mean_tw_minus_ts1": mean(s),
                                 "mean_tw_minus_ts0_plus_d": mean(s0) if s0 else None}
    if "try.h0" in fits and "test" in fits:
        out["cycles"] = fits["try.h0"]["slope"] - fits["test"]["slope"] + (iter_stall or 0.0)
        out["suspend"] = fits["try.h0"]["slope"] - fits["test"]["slope"]
    return out, fits


def cell_trywait(a, b, ca, cb):
    res = {"a": derive_trywait(a, ca), "b": derive_trywait(b, cb)}
    fa, fb = res["a"][1], res["b"][1]
    if "try.h0" not in fa or "test" not in fa:
        return {"cell": "trywait_timeout", "verdict": "unrealised",
                "why": "missing loop variants", "detail": res}
    resid = max([f["max_rel_resid"] for f in fa.values()] +
                [f["max_rel_resid"] for f in fb.values()])
    agree = rel(fa["try.h0"]["slope"], fb["try.h0"]["slope"])
    # "if the no-hint period equals the test_wait period within 5 % the
    # leaf is 0 (the hardware does not suspend beyond its issue)"
    susp_a = fa["try.h0"]["slope"] - fa["test"]["slope"]
    susp_b = fb["try.h0"]["slope"] - fb["test"]["slope"]
    susp_rel = abs(susp_a) / mean([fa["try.h0"]["slope"], fa["test"]["slope"]])
    it = res["a"][0]["iter_stall_test_wait"] or res["b"][0]["iter_stall_test_wait"] or 0.0
    if susp_rel <= RUN_AGREE_5:
        value, note = 0.0, "no-hint period within 5 % of test_wait: no suspension beyond issue"
    else:
        value = mean([susp_a, susp_b]) + it
        note = "period(try.h0) - period(test) + iter_stall"
    verdict = "measured" if resid <= FIT_RESID and agree <= RUN_AGREE_5 else "experimental"
    return {"cell": "trywait_timeout", "cycles": value, "note": note,
            "resid": resid, "agree": agree, "verdict": verdict,
            "periods": {r: res[r][0]["periods"] for r in "ab"},
            "iter_stall": it, "release": {"a": res["a"][0]["release"], "b": res["b"][0]["release"]}}


def parse_ldc(v):
    # ldc.s<S>.f<F>.<warm|cold>
    parts = v.split(".")
    if len(parts) == 4 and parts[0] == "ldc":
        return int(parts[1][1:]), int(parts[2][1:]), parts[3]
    return None


def derive_ldc(rows, checks):
    """per-load latency per cell = OLS slope over U (two points 64/256)."""
    lat, fits = {}, {}
    gated = checks.get("ldc_chain", {}).get("kernels", {})
    gate_failed = any(not v.get("ok") for v in gated.values()) if gated else False
    for v, us in rows.items():
        p = parse_ldc(v)
        if not p:
            continue
        s, f, heat = p
        f_ = slope_fit(us)
        if f_:
            lat[(s, f, heat)] = f_["slope"]
            fits[(s, f, heat)] = f_
    return {"per_load": lat, "fits": fits, "gate_failed": gate_failed}


def cell_ldc(a, b, ca, cb):
    da, db = derive_ldc(a, ca), derive_ldc(b, cb)
    common = sorted(set(da["per_load"]) & set(db["per_load"]))
    agree = {k: rel(da["per_load"][k], db["per_load"][k]) for k in common}
    lat = {k: (da["per_load"][k] + db["per_load"][k]) / 2 for k in common}
    warm = {k: v for k, v in lat.items() if k[2] == "warm"}
    cold = {k: v for k, v in lat.items() if k[2] == "cold"}
    out = {"cell": "constant_cache", "agree_max": max(agree.values(), default=0.0),
           "per_load": {".".join(map(str, k)): lat[k] for k in common}}
    if not warm:
        return out | {"verdict": "unrealised", "why": "no cells"}
    # same_line = warm S = 4, F = 256 B (protocol derivation text)
    same_line = warm.get((4, 256, "warm"))
    # Warm plateaus per stride over F (5 % groups).  The hit plateau is the
    # plateau holding the largest F at the largest stride (256 B >= any
    # line): the level that holds the whole 64 KiB bank.  The F where the
    # level below it ends is the lower (L0) knee; the capacity knee of the
    # hit level is not reachable inside the 64 KiB __constant__ bank and is
    # reported as >= the largest F.  (b300-r2: 37 up to 2 KiB, 105 beyond.)
    def plateaus(s):
        cells_s = sorted((f, v) for (ss, f, _), v in warm.items() if ss == s)
        groups = []
        for f, v in cells_s:
            if groups and abs(v - groups[-1][0][1]) <= LINE_TOL * groups[-1][0][1]:
                groups[-1].append((f, v))
            else:
                groups.append([(f, v)])
        return groups
    s_max = max(k[0] for k in warm)
    pl = plateaus(s_max)
    hit_group = pl[-1]
    hit_f_min = hit_group[0][0]
    hit_ref = mean([v for _, v in hit_group])
    low_knee = pl[-2][-1][0] if len(pl) > 1 else None
    out["warm_plateaus_s_max"] = [[[f, round(v, 2)] for f, v in g] for g in pl]
    out["l0_knee_bytes"] = low_knee
    out["capacity_knee_bytes"] = f">= {hit_group[-1][0]} (plateau flat to the largest footprint)"
    out["per_stride_plateau"] = {str(s): mean([v for (ss, f, _), v in warm.items() if ss == s and f >= hit_f_min])
                                 for s in sorted({k[0] for k in warm})}
    # line_bytes (protocol): the smallest S at which the warm S-stride
    # latency equals the hit plateau within 5 % of the (S = 4 -> plateau)
    # step, over the F range of the hit plateau
    line_bytes, hit = None, None
    if same_line is not None:
        step = hit_ref - same_line
        for s in sorted({k[0] for k in warm}):
            pts = [v for (ss, f, _), v in warm.items() if ss == s and f >= hit_f_min]
            if pts and all(abs(v - hit_ref) <= LINE_TOL * abs(step) for v in pts):
                line_bytes = s
                break
        if line_bytes is not None:
            hit = mean([v for (ss, f, _), v in warm.items() if ss >= line_bytes and f >= hit_f_min])
    # miss = cold first traversal: the cell where every one of the U loads
    # touches a new line and none wraps (F / S >= U_max): the largest
    # stride at the largest footprint; the other cold cells (mixed
    # first-touch / re-touch within the chain) are reported
    knees = {str(s): low_knee for s in sorted({k[0] for k in warm})}
    u_max = max(max(f_.get("us", [0])) for f_ in da["fits"].values()) if da.get("fits") else 256
    miss = None
    cand = [(s, f, v) for (s, f, _), v in cold.items() if s == s_max and f // s >= u_max and f >= hit_f_min]
    if cand:
        s_, f_, miss = max(cand)
        out["miss_cell"] = {"stride_bytes": s_, "footprint_bytes": f_}
    out["cold_table"] = {f"{s}.{f}": round(v, 1) for (s, f, _), v in sorted(cold.items())}
    out["warm_table"] = {f"{s}.{f}": round(v, 1) for (s, f, _), v in sorted(warm.items())}
    resid = max([f["max_rel_resid"] for f in da["fits"].values()] + [f["max_rel_resid"] for f in db["fits"].values()], default=0.0)
    out["fit_resid_max"] = resid
    ok = (out["agree_max"] <= RUN_AGREE_5 and resid <= FIT_RESID and same_line is not None and hit is not None
          and miss is not None and not (da["gate_failed"] or db["gate_failed"]))
    out |= {"same_line": same_line, "hit": hit, "miss": miss,
            "line_bytes": line_bytes,
            "verdict": "measured" if ok else "experimental"}
    return out


def derive_l1cap(rows):
    """per-load latency per (S,F) = median traversal cycles / hops (u)."""
    lat = {}
    for v, us in rows.items():
        p = v.split(".")
        if len(p) != 3 or p[0] != "l1cap":
            continue
        s_k = int(p[1][1:-1]) * 1024
        f_k = int(p[2][1:-1]) * 1024
        for u, samples in us.items():
            lat[(s_k, f_k)] = statistics.median(samples) / u
    return lat


def knees_of(lat, l1_hit):
    """per S: the largest F whose warm per-load latency is within KNEE_TOL
    of the L1 hit latency (memlat l1); None when even the smallest F is not."""
    out = {}
    for s in sorted({k[0] for k in lat}):
        cells_s = sorted((f, v) for (ss, f), v in lat.items() if ss == s)
        inside = [f for f, v in cells_s if v <= (1 + KNEE_TOL) * l1_hit]
        out[s] = max(inside) if inside else None
    return out


def cell_l1cap(a, b, device):
    la, lb = derive_l1cap(a), derive_l1cap(b)
    common = sorted(set(la) & set(lb))
    lat = {k: (la[k] + lb[k]) / 2 for k in common}
    l1_hit = MEMLAT_L1[device]
    ka, kb = knees_of(la, l1_hit), knees_of(lb, l1_hit)
    knees = knees_of(lat, l1_hit)
    table = {}
    for s in sorted({k[0] for k in lat}):
        cells_s = sorted((f, v) for (ss, f), v in lat.items() if ss == s)
        table[str(s)] = {"smallest_f_latency": cells_s[0][1], "knee_bytes": knees[s],
                         "knee_a": ka.get(s), "knee_b": kb.get(s),
                         "points": {str(f): round(v, 2) for f, v in cells_s}}
    # 16 KiB-reference form (reported): knee relative to the smallest-F point of the same S
    alt = {}
    for s in sorted({k[0] for k in lat}):
        cells_s = sorted((f, v) for (ss, f), v in lat.items() if ss == s)
        alt[str(s)] = max(f for f, v in cells_s if v <= (1 + KNEE_TOL) * cells_s[0][1])
    good = {s: k for s, k in knees.items() if k is not None}
    sums = [good[s] + s for s in good]
    spread = (max(sums) - min(sums)) if sums else float("inf")
    run_equal = bool(good) and all(ka.get(s) == kb.get(s) for s in good)
    verdict = ("unrealised" if not good or len(good) < len(knees)
               else "measured" if spread <= 8192 and run_equal else "experimental")
    return {"cell": "l1_residency", "l1_hit_reference": l1_hit,
            "data_array_bytes": mean(sums) if sums else None, "reserved_bytes": 0,
            "knee_plus_s": {str(s): good[s] + s for s in good}, "knee_table": table,
            "knee_relative_to_16k": alt, "spread": spread, "run_equal": run_equal, "verdict": verdict,
            "note": "engine uses data_array - reserved; installed as data_array_bytes=A, reserved_bytes=0"}


def derive_tmap(rows):
    out = {}
    for c in ("first", "reuse", "acquire", "param"):
        s = rows.get(f"tmap.{c}", {}).get(1)
        i = rows.get(f"tmap.{c}.issue", {}).get(1)
        if s:
            out[c] = {"mean": mean(s), "se": se(s), "n": len(s),
                      "issue_mean": mean(i) if i else None}
    return out


def cell_tmap(a, b):
    da, db = derive_tmap(a), derive_tmap(b)
    if "reuse" not in da or "reuse" not in db:
        return {"cell": "descriptor_refetch", "verdict": "unrealised"}
    refetch = {}
    for c in ("first", "acquire", "param"):
        if c in da and c in db:
            refetch[c] = mean([da[c]["mean"] - da["reuse"]["mean"],
                               db[c]["mean"] - db["reuse"]["mean"]])
            refetch[c + "_runs"] = [da[c]["mean"] - da["reuse"]["mean"],
                                    db[c]["mean"] - db["reuse"]["mean"]]
    se_rel = max(da["first"]["se"] / da["first"]["mean"],
                 db["first"]["se"] / db["first"]["mean"]) if "first" in da else float("inf")
    agree = rel(*refetch.get("first_runs", [0, float("inf")])) if "first" in refetch else float("inf")
    verdict = "measured" if se_rel <= SE_3 and agree <= RUN_AGREE_5 else "experimental"
    return {"cell": "descriptor_refetch", "refetch_cycles": refetch.get("first"),
            "refetch_reported": {c: refetch[c] for c in ("acquire", "param") if c in refetch},
            "cells": {"a": da, "b": db}, "se_rel": se_rel, "agree": agree,
            "verdict": verdict,
            "note": "installed cycles = refetch(first); engine's first-use case"}


def mem_bytes_per_cycle(rows, checks):
    """stream/store cells: bytes/cycle/SM = W x 32 x vec / slope over K.
    (blackwell-prototype wave3_leaves.build_mem equation.)"""
    cells = {}
    for v, us in rows.items():
        p = v.split(".")
        if len(p) < 5 or p[0] not in ("stream", "store"):
            continue
        lvl, vec, wn = p[1], int(p[2][1:]), int(p[3][1:])
        d = next((int(x[1:]) for x in p if x.startswith("d") and x[1:].isdigit()), 8)
        all_sm = 1 if p[-1] == "all" else 0
        st = 1 if p[0] == "store" else 0
        lvl_i = {"l1": 0, "l2": 1, "dram": 2}[lvl]
        gkey = f"mem/{'s' if st else ''}{lvl_i}/{vec}/"
        series = {}
        for u, samples in us.items():
            gk = gkey + f"{u}" + ("" if d == 8 else f"/d{d}")
            if checks and not gate_ok(checks, "mem", gk):
                continue
            series[u] = samples
        f = slope_fit(series)
        if f and f["slope"] > 0:
            cells[(lvl, st, vec, wn, d, all_sm)] = {
                "bytes_per_cycle": wn * 32 * vec / f["slope"],
                "slope": f["slope"], "resid": f["max_rel_resid"], "variant": v}
    return cells


def cell_l1bw(a, b, ca, cb, device):
    ca_cells, cb_cells = mem_bytes_per_cycle(a, ca), mem_bytes_per_cycle(b, cb)
    res = {"cell": "l1_bandwidth"}
    key = ("l1", 0, 16, 32, 8, 0)  # wave3_leaves MEM_BW "l1" analysis key
    va, vb = ca_cells.get(key, {}).get("bytes_per_cycle"), cb_cells.get(key, {}).get("bytes_per_cycle")
    res["l1_cells_curve"] = {"a": {str(k): round(v["bytes_per_cycle"], 2) for k, v in ca_cells.items() if k[0] == "l1" and not k[1]},
                             "b": {str(k): round(v["bytes_per_cycle"], 2) for k, v in cb_cells.items() if k[0] == "l1" and not k[1]}}
    if va and vb:
        agree = rel(va, vb)
        val = (va + vb) / 2
        ctrl_ok = device != "b300" or rel(val, B300_L1_BW) <= B300_CTRL_TOL
        res["sector_bytes_per_cycle"] = {"value": val, "a": va, "b": vb, "agree": agree,
                                         "b300_control_ok": ctrl_ok,
                                         "verdict": "measured" if agree <= RUN_AGREE_3 else "experimental"}
    # issue interval: saturated l1 1-sector request rate from memq rows
    # (memq_rules.request_rate / issue_interval verbatim)
    def l1_rates(rows, checks):
        rates = []
        for v, us in rows.items():
            p = v.split(".")
            if len(p) == 4 and p[0] == "ld" and p[1] == "l1":  # ld.l1.d<D>.w<W>
                d, wn = int(p[2][1:]), int(p[3][1:])
                series = {u: s for u, s in us.items()
                          if not checks or gate_ok(checks, "memq", f"memq/ld0/d{d}/{u}")}
                f = slope_fit(series)
                if f and f["slope"] > 0:
                    rates.append((v, wn / f["slope"], wn * d))
        return rates
    ra, rb = l1_rates(a, ca), l1_rates(b, cb)
    for tag, rates in (("a", ra), ("b", rb)):
        res.setdefault("issue_interval_series", {})[tag] = sorted(
            (v, round(r, 4), n) for v, r, n in rates)
    if ra and rb:
        rate_max = max(max(r for _, r, _ in ra), max(r for _, r, _ in rb))
        period = 1.0 / rate_max
        # memq_rules.issue_interval: period <= 1 + FLOOR_TOL -> 0 (at or below the engine's 1-cycle floor)
        value = 0 if period <= 1.0 + FLOOR_TOL else period
        ctrl_ok = device != "b300" or rel(period, B300_L1_INTERVAL) <= B300_CTRL_TOL
        res["issue_interval_cycles"] = {"value": value, "period": period,
                                        "rate_max": rate_max, "b300_control_ok": ctrl_ok,
                                        "verdict": "measured",
                                        "note": "memq_rules.issue_interval: period <= 1 + 5 % -> 0 (engine 1-cycle floor)"}
    return res


def cell_smport(a, b, ca, cb):
    """store_dram_port: dram store one-SM cells, plateau per the frozen
    sm_dram_port derivation (best cell, confirmed by two alternative
    (W, D) cells within 3 %)."""
    ca_cells, cb_cells = mem_bytes_per_cycle(a, ca), mem_bytes_per_cycle(b, cb)
    table = {}
    for c in sorted(set(ca_cells) & set(cb_cells)):
        if c[0] != "dram" or c[1] != 1 or c[5] != 0:
            continue
        va, vb = ca_cells[c]["bytes_per_cycle"], cb_cells[c]["bytes_per_cycle"]
        table[c] = {"a": va, "b": vb, "mean": (va + vb) / 2, "agree": rel(va, vb)}
    res = {"cell": "store_dram_port",
           "cells": {str(c): {k: round(v, 3) for k, v in t.items()} for c, t in table.items()}}
    # blackwell-prototype sm_dram_port write definition: primary cell 32 warps x
    # depth 8 (STG.E.128, one SM), plateau confirmed by 8 x 32 and 16 x 16
    prim = ("dram", 1, 16, 32, 8, 0)
    alts = [("dram", 1, 16, 8, 32, 0), ("dram", 1, 16, 16, 16, 0)]
    if prim not in table:
        return res | {"verdict": "unrealised", "why": "no store.dram.v16.w32.d8.one cell"}
    best_v = table[prim]["mean"]
    others = [table[c]["mean"] for c in alts if c in table]
    plateau_ok = len(others) == 2 and all(rel(best_v, v) <= SMPORT_PLATEAU for v in others)
    agree = table[prim]["agree"]
    verdict = "measured" if agree <= RUN_AGREE_3 and plateau_ok else "experimental"
    return res | {"sector_bytes_per_cycle": best_v, "best_cell": str(prim),
                  "run_agree": agree, "plateau_ok": plateau_ok,
                  "alternatives": [round(v, 3) for v in others],
                  "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    checks_a = {c: load_checks(a.root, c) for c in CELLS}
    rows_a = {c: load_cell(a.root, c, "run-a") for c in CELLS}
    rows_b = {c: load_cell(a.root, c, "run-b") for c in CELLS}
    smport_a = load_smport(a.root, "run-a")
    smport_b = load_smport(a.root, "run-b")

    res = {"schema": "cyclescope.microbench-model.v1",
           "protocol": "tools/microbench/crossings/protocol-v2.json",
           "device": a.device, "root": a.root,
           "cells": {}}
    res["cells"]["trywait_timeout"] = cell_trywait(
        rows_a["trywait_timeout"], rows_b["trywait_timeout"],
        checks_a["trywait_timeout"], checks_a["trywait_timeout"])
    res["cells"]["constant_cache"] = cell_ldc(
        rows_a["constant_cache"], rows_b["constant_cache"],
        checks_a["constant_cache"], checks_a["constant_cache"])
    res["cells"]["l1_residency"] = cell_l1cap(rows_a["l1_residency"], rows_b["l1_residency"], a.device)
    res["cells"]["descriptor_refetch"] = cell_tmap(rows_a["descriptor_refetch"],
                                                 rows_b["descriptor_refetch"])
    res["cells"]["l1_bandwidth"] = cell_l1bw(rows_a["l1_bandwidth"], rows_b["l1_bandwidth"],
                                           checks_a["l1_bandwidth"], checks_a["l1_bandwidth"],
                                           a.device)
    res["cells"]["store_dram_port"] = cell_smport(smport_a, smport_b,
                                                  checks_a["l1_bandwidth"], checks_a["l1_bandwidth"])
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: {kk: v[kk] for kk in ("verdict", "cycles", "sector_bytes_per_cycle",
                                               "issue_interval_cycles", "data_array_bytes",
                                               "refetch_cycles") if kk in v}
                      for k, v in res["cells"].items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
