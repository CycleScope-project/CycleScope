#!/usr/bin/env python3
"""tma_store_loaded derivation (protocol.json).

  model.py --run <dir with run-a.jsonl run-b.jsonl> --store-port <B/cycle> \
           --l2-write <B/ns> --tma-store-leaf <cycles, 8 KiB UTMASTG tile> \
           --loader-ref <B/cycle, tma_loaded N1.S4 of the device> --out <json>
"""
import argparse, json, statistics
from pathlib import Path

BOX = 8192


def load(path):
    rows = {}
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[d["family"]] = d
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--store-port", type=float, required=True)
    ap.add_argument("--l2-write", type=float, required=True)
    ap.add_argument("--tma-store-leaf", type=float, required=True)
    ap.add_argument("--loader-ref", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {r: load(Path(a.run) / f"run-{r}.jsonl") for r in ("a", "b")}
    fams = sorted(set(runs["a"]) | set(runs["b"]))
    nmax = max(runs[r][f]["ctas"] for r in runs for f in runs[r])
    keys = ["period_median", "aggregate_store_bytes_per_ns_median", "cycles_per_ns_median",
            "load_bytes_per_cycle_median", "seg_fence", "seg_bar1", "seg_issue", "seg_wait", "seg_bar2",
            "other_fence", "other_bar1", "other_rest", "wait_p50", "wait_p90", "round_p50", "round_p90"]
    cells = {}
    for f in fams:
        cs = [runs[r][f] for r in ("a", "b") if f in runs[r]]
        c = {k: statistics.median(x[k] for x in cs) for k in keys}
        c.update({k: cs[0][k] for k in ("kind", "ctas", "loaders", "stages", "dst", "state", "stamps", "epi")})
        c["family"] = f
        c["errors"] = [x["error"] for x in cs if x.get("error")]
        c["spread"] = max(((x["period_p90"] - x["period_p10"]) / x["period_median"]) if x["period_median"] else 0.0
                          for x in cs)
        if len(cs) == 2:
            pa, pb = (x["period_median"] or x["load_bytes_per_cycle_median"] for x in cs)
            c["run_delta"] = abs(pa - pb) / max(pa, pb) if max(pa, pb) else 0.0
        else:
            c["run_delta"] = None
        cells[f] = c

    def cell(kind, n, l, d, dst, state="warm", stamps=0, epi=1):
        return cells.get(f"tma_store_loaded.{kind}.N{n}.L{l}.D{d}.{dst}.{state}.s{stamps}" + ("" if epi else ".epi0"))

    def P(*k, **kw):
        c = cell(*k, **kw)
        return c["period_median"] if c else None

    out = {"run": a.run, "nmax": nmax, "inputs": vars(a), "cells": list(cells.values()), "checks": {}}
    ck = out["checks"]
    cpn = statistics.median(c["cycles_per_ns_median"] for c in cells.values() if c["cycles_per_ns_median"])
    ck["cycles_per_ns"] = cpn
    # controls
    c1 = cell("tma", 1, 0, 1, "ring", stamps=1)
    if c1:
        lat = c1["seg_issue"] + c1["seg_wait"]
        ck["control_store_completion"] = {"measured": lat, "leaf": a.tma_store_leaf, "ratio": lat / a.tma_store_leaf,
                                          "ok": abs(lat - a.tma_store_leaf) <= 0.20 * a.tma_store_leaf}
    c2 = cell("tma", 1, 4, 4, "ring", epi=0)
    if c2:
        r = c2["load_bytes_per_cycle_median"]
        ck["control_loader"] = {"measured": r, "ref": a.loader_ref, "ratio": r / a.loader_ref,
                                "ok": abs(r - a.loader_ref) <= 0.15 * a.loader_ref}
    ck["controls_ok"] = bool(ck.get("control_store_completion", {}).get("ok") and ck.get("control_loader", {}).get("ok"))
    # stamp perturbation
    ck["stamp_ratio"] = {}
    for f, c in cells.items():
        if c["stamps"] == 1 and c["epi"]:
            c0 = cells.get(f[:-1] + "0")
            if c0 and c0["period_median"]:
                ck["stamp_ratio"][f] = c["period_median"] / c0["period_median"]

    def pred(n):  # installed-leaf prediction of the saturated ring period
        return BOX / min(a.store_port, a.l2_write / (n * cpn))

    for dst in ("ring", "fresh"):
        o = {}
        p1 = P("tma", 1, 0, 4, dst)
        p1l = P("tma", 1, 4, 4, dst)
        pn = P("tma", nmax, 0, 4, dst)
        pnl = P("tma", nmax, 4, 4, dst)
        if p1:
            o["O1_rho1"] = p1 / pred(1)
        l0 = cell("tma", 1, 4, 4, "ring", epi=0)
        l1 = cell("tma", 1, 4, 4, dst)
        if p1 and p1l and l0 and l1:
            kappa = p1l / p1
            lam = l1["load_bytes_per_cycle_median"] / l0["load_bytes_per_cycle_median"]
            o["O2"] = {"kappa": kappa, "lambda": lam, "U": 1.0 / kappa + lam}
        cn = cell("tma", nmax, 0, 4, dst)
        sn = cell("stg", nmax, 0, 4, dst)
        s1 = cell("stg", 1, 0, 4, dst)
        if cn and pn:
            o["O3"] = {"period_nmax": pn, "pred_nmax": pred(nmax), "ratio": pn / pred(nmax),
                       "aggregate_tma": cn["aggregate_store_bytes_per_ns_median"],
                       "aggregate_stg": sn["aggregate_store_bytes_per_ns_median"] if sn else None,
                       "stg_period_1": s1["period_median"] if s1 else None,
                       "stg_period_nmax": sn["period_median"] if sn else None,
                       "l2_write_leaf": a.l2_write}
        ln0 = cell("tma", nmax, 4, 4, "ring", epi=0)
        ln1 = cell("tma", nmax, 4, 4, dst)
        if pn and pnl and ln0 and ln1:
            o["O4"] = {"kappa_n": pnl / pn,
                       "lambda_n": ln1["load_bytes_per_cycle_median"] / ln0["load_bytes_per_cycle_median"],
                       "aggregate_store": ln1["aggregate_store_bytes_per_ns_median"],
                       "aggregate_load": ln1["load_bytes_per_cycle_median"] * nmax * cpn}
        if dst == "fresh":
            cold = P("tma", nmax, 0, 4, "fresh", state="cold")
            if cold and pn:
                o["cold_over_warm_nmax"] = cold / pn
            cold1 = P("tma", 1, 0, 4, "fresh", state="cold")
            if cold1 and p1:
                o["cold_over_warm_1"] = cold1 / p1
            if pnl:
                o["analog_period"] = pnl
        ck[dst] = o
    s = cell("tma", 1, 0, 4, "ring", stamps=1)
    if s:
        ck["O5_steps_n1"] = {k: s[k] for k in ("seg_fence", "seg_bar1", "seg_issue", "seg_wait", "seg_bar2",
                                                "other_fence", "other_bar1", "other_rest")}
    ck["all_spread_ok"] = all(c["spread"] < 0.10 for c in cells.values() if c["ctas"] >= 16 and c["epi"])
    ck["runs_agree"] = all(c["run_delta"] is None or c["run_delta"] <= 0.05 for c in cells.values())
    ck["errors"] = [c["family"] for c in cells.values() if c["errors"]]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(ck, indent=1))
    print("%-58s %8s %9s %8s %7s %7s %7s %7s %7s" % ("family", "period", "agg B/ns", "load", "fence", "bar1", "issue", "wait", "bar2"))
    for f in fams:
        c = cells[f]
        print("%-58s %8.1f %9.1f %8.1f %7.1f %7.1f %7.1f %7.1f %7.1f" % (
            f.replace("tma_store_loaded.", ""), c["period_median"], c["aggregate_store_bytes_per_ns_median"],
            c["load_bytes_per_cycle_median"], c["seg_fence"], c["seg_bar1"], c["seg_issue"], c["seg_wait"], c["seg_bar2"]))


if __name__ == "__main__":
    main()
