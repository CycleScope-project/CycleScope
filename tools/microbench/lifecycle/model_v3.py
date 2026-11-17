#!/usr/bin/env python3
"""kernel.lifecycle envelope derivation, protocol version 3
(tools/microbench/lifecycle/protocol-v3.json).

Per hardware launch (launches 0 and 1 of every variant dropped):
    out_hw = sm__cycles_active.max - marker span   (no fetch term)
Per cell and run:
    env = mean(out_hw) - (core_sim - span_sim)
core_sim / span_sim come from the engine's own run of the captured probe
launch (probe_sim.py -> probe-sim.tsv).  envelope(W) = mean over the two
runs of env(c1.w<W>.s0.l0).

Acceptance per W (protocol):
  hardware: se(mean out_hw) / mean(out_hw) <= 3 % per run, run agreement
            |a - b| / mean <= 5 % (on out_hw), L = 0 cell
  body_independence: |env(L=1024) - env(L=0)| <= max(0.10 env(L=0), 64)
  measured = both; experimental = hardware only; unrealised = hardware
  fails; env(L=0) < 0 for any W -> nothing installed on the device.

Usage: model_v3.py --run-a <dir> --run-b <dir> --sim <probe-sim.tsv>
                   --out model.json
"""
import argparse, csv, json, math, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import parse_counters, parse_launches  # noqa: E402

DROP = {0, 1}
SE_REL = 0.03
RUN_AGREE = 0.05
BODY_REL, BODY_ABS = 0.10, 64.0
WS = (1, 4, 8, 16, 32)
LS = (0, 1024, 4096)


def hw_cells(run_dir):
    launches = parse_launches(Path(run_dir) / "launches.log")
    act, _ = parse_counters(Path(run_dir) / "counters.csv")
    if len(launches) != len(act):
        raise SystemExit(f"{run_dir}: {len(launches)} launches vs {len(act)} counter rows")
    cells = {}
    for (v, launch, span, _smids), a in zip(launches, act):
        if launch in DROP:
            continue
        cells.setdefault(v, []).append(a - span)
    out = {}
    for v, xs in cells.items():
        n = len(xs)
        if n < 4:
            continue
        mean = statistics.fmean(xs)
        se = statistics.stdev(xs) / math.sqrt(n)
        srt = sorted(xs)
        g, i = max((srt[k + 1] - srt[k], k) for k in range(n - 1))
        lo, hi = srt[:i + 1], srt[i + 1:]
        out[v] = {"mean_out_hw": mean, "se": se, "se_rel": se / mean if mean else None, "n": n,
                  "median": statistics.median(xs),
                  "deciles": [srt[min(n - 1, round(q * (n - 1) / 10))] for q in range(11)],
                  "modes": {"gap": g, "low": {"n": len(lo), "mean": statistics.fmean(lo)},
                            "high": {"n": len(hi), "mean": statistics.fmean(hi)} if hi else None}}
    return out


def sim_table(path):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[r["cell"].removeprefix("life.")] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {"a": hw_cells(a.run_a), "b": hw_cells(a.run_b)}
    sim = sim_table(a.sim)
    out = {"protocol": "tools/microbench/lifecycle/protocol-v3.json",
           "runs": {"a": a.run_a, "b": a.run_b}, "sim": a.sim,
           "tree_digest": sorted({r.get("tree_digest", "") for r in sim.values()}),
           "engine": sorted({r.get("engine", "") for r in sim.values()}),
           "cells": {}, "hardware_only_cells": {}, "install": {}}
    for v in sorted(set(runs["a"]) | set(runs["b"])):
        ca, cb = runs["a"].get(v), runs["b"].get(v)
        if v not in sim:
            out["hardware_only_cells"][v] = {"run_a": ca, "run_b": cb}
            continue
        s = sim[v]
        if s["status"] != "ok" or s["issued"] != s["completed"]:
            out["cells"][v] = {"run_a": ca, "run_b": cb, "sim": s, "error": "sim not ok"}
            continue
        core, span = float(s["core_sim"]), float(s["span_sim"])
        sim_out = core - span
        cell = {"run_a": ca, "run_b": cb, "core_sim": core, "span_sim": span, "sim_outside": sim_out}
        for tag, c in (("a", ca), ("b", cb)):
            if c:
                cell[f"env_{tag}"] = c["mean_out_hw"] - sim_out
        if ca and cb:
            m = (ca["mean_out_hw"] + cb["mean_out_hw"]) / 2
            cell["mean_out_hw"] = m
            cell["run_agree"] = abs(ca["mean_out_hw"] - cb["mean_out_hw"]) / m
            cell["se_rel_max"] = max(ca["se_rel"], cb["se_rel"])
            cell["env"] = m - sim_out
        out["cells"][v] = cell
    negative = []
    for w in WS:
        c0 = out["cells"].get(f"c1.w{w}.s0.l0", {})
        c1 = out["cells"].get(f"c1.w{w}.s0.l1024", {})
        c4 = out["cells"].get(f"c1.w{w}.s0.l4096", {})
        if "env" not in c0:
            out["install"][str(w)] = {"verdict": "unrealised", "reason": "L = 0 cell incomplete"}
            continue
        hw_ok = c0["se_rel_max"] <= SE_REL and c0["run_agree"] <= RUN_AGREE
        e0 = c0["env"]
        rec = {"envelope_cycles": e0, "env_l0": e0,
               "env_l1024": c1.get("env"), "env_l4096_reported": c4.get("env"),
               "se_rel_max": c0["se_rel_max"], "run_agree": c0["run_agree"], "hardware_ok": hw_ok}
        if "env" in c1:
            lim = max(BODY_REL * e0, BODY_ABS)
            rec["body_diff"] = abs(c1["env"] - e0)
            rec["body_limit"] = lim
            body_ok = rec["body_diff"] <= lim
        else:
            body_ok = False
            rec["body_diff"] = None
        rec["body_ok"] = body_ok
        rec["verdict"] = ("measured" if hw_ok and body_ok else "experimental" if hw_ok else "unrealised")
        if e0 < 0:
            negative.append(w)
        out["install"][str(w)] = rec
    out["negative_env_w"] = negative
    if negative:
        out["points"] = []
        out["verdict"] = "not_installed:negative_envelope"
    else:
        out["points"] = [[w, round(out["install"][str(w)]["envelope_cycles"], 1)] for w in WS
                         if out["install"].get(str(w), {}).get("verdict") in ("measured", "experimental")]
        vs = [out["install"][str(w)]["verdict"] for w in WS]
        out["verdict"] = ("measured" if all(v == "measured" for v in vs)
                          else "experimental" if out["points"] else "unrealised")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for w in WS:
        r = out["install"].get(str(w), {})
        print(f"W={w:2d} env L0={r.get('env_l0')!s:>10.10} L1024={r.get('env_l1024')!s:>10.10} "
              f"L4096={r.get('env_l4096_reported')!s:>10.10} se={r.get('se_rel_max')!s:.6} "
              f"agree={r.get('run_agree')!s:.6} body_ok={r.get('body_ok')} -> {r.get('verdict')}")
    print("points", out["points"], "verdict", out["verdict"])


if __name__ == "__main__":
    main()
