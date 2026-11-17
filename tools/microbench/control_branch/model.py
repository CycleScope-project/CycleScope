#!/usr/bin/env python3
"""control_branch model: per (variant,W) slope over U in {16,32,64,128}
of E = mean(t_w - t_s); branch = slope - body; body = installed FFMA
result_latency + stalls of non-branch insts (sass_ctrl o_s convention,
one gap containing the unit's non-branch instructions).

    model.py --device h100|b300 --run-dir <run> --sass <branch.sass> \
        --ffma-latency <cycles> --out <model dir>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tensor_pipe"))
import sass_ctrl as sc  # noqa: E402

US = (16, 32, 64, 128)


def ols(xs, ys):
    mx = statistics.fmean(xs)
    b = sum((x - mx) * y for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = statistics.fmean(ys) - b * mx
    resid = max(abs(a + b * x - y) / y for x, y in zip(xs, ys))
    return b, a, resid


def load_run(run_dir):
    out = defaultdict(dict)
    for f in sorted(Path(run_dir).glob("branch.jsonl")):
        for line in f.open():
            r = json.loads(line)
            s = r["samples"]
            v = r["variant"].rsplit(".u", 1)[0]
            out[(v, r["params"]["w"])][r["u"]] = {
                "mean": statistics.fmean(s), "sd": statistics.pstdev(s),
                "n": len(s)}
    return out


def body_cost(kernel: sc.Kernel, variant: str, ffma_lat: float):
    """non-branch stall sum per EXECUTED unit + FFMA result latency.
    For taken/uncond/taken_dep the block between a BRA and its target
    is skipped (never executed) and does not count; for nottaken the
    fall-through chain FFMA does count; for loop the whole loop body
    counts."""
    ins = kernel.insts
    bra = [i for i, x in enumerate(ins) if x.opcode.startswith("BRA")]
    if not bra:
        return None
    import re as _re
    stalls = []
    if variant == "loop":
        seg = ins[bra[0] - 8:bra[0]]
        stalls = [x.stall for x in seg if not x.opcode.startswith("BRA")]
    else:
        addr2idx = {x.addr: i for i, x in enumerate(ins)}
        for a, b in zip(bra, bra[1:] + [len(ins)]):
            seg = ins[a + 1:b]
            if variant != "nottaken":
                m = _re.search(r"0x[0-9a-f]+", ins[a].body)
                tgt = addr2idx.get(int(m.group(0), 16) if m else -1)
                if tgt is not None and a < tgt <= b:
                    seg = ins[tgt:b]
            stalls.append(sum(x.stall for x in seg))
    per_unit = statistics.fmean(stalls) if stalls else 0.0
    return ffma_lat + per_unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--sass", required=True, type=Path)
    ap.add_argument("--ffma-latency", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    kern_text = a.sass.read_text(errors="replace")
    kernels = {k.name: k for k in sc.parse(kern_text).values()}
    runs = {}
    for run in ("run-a", "run-b"):
        recs = load_run(a.run_dir / run)
        res = {}
        for (v, w), g in sorted(recs.items()):
            us = [u for u in US if u in g]
            if len(us) != 4:
                res[f"{v}.w{w}"] = {"error": "incomplete U set"}
                continue
            slope, icpt, resid = ols(us, [g[u]["mean"] for u in us])
            res[f"{v}.w{w}"] = {"slope": slope, "resid": resid,
                                "means": {str(u): g[u]["mean"] for u in us}}
        runs[run] = res
    a.out.mkdir(parents=True, exist_ok=True)
    verdict = {}
    print(f"{'cell':16s} {'slope':>8s} {'body':>6s} {'branch':>8s} "
          f"{'resid%':>6s} {'agree%':>6s} status")
    for key in sorted(runs["run-a"]):
        x, y = runs["run-a"].get(key, {}), runs["run-b"].get(key, {})
        if not x or not y or "slope" not in x or "slope" not in y:
            verdict[key] = {"status": "unrealised", "reason": "missing"}
            continue
        slope = (x["slope"] + y["slope"]) / 2
        agree = abs(x["slope"] - y["slope"]) / slope
        resid = max(x["resid"], y["resid"])
        variant = key.split(".")[0]
        body = None
        vnum = {"taken": 0, "nottaken": 1, "uncond": 2,
                "taken_dep": 3, "loop": 4}[variant]
        kname = next((n for n in kernels
                      if f"k_branchILi{vnum}E" in n and "Li64E" in n), None)
        if kname:
            body = body_cost(kernels[kname], variant, a.ffma_latency)
        branch = None if body is None else slope - body
        ok = resid <= 0.02 and agree <= 0.03
        stat = ("measured" if ok else
                "experimental" if resid <= 0.05 or agree <= 0.08
                else "unrealised")
        verdict[key] = {"variant": variant, "w": key.rsplit("w", 1)[1],
                        "slope": slope, "body": body, "branch": branch,
                        "resid": resid, "agree": agree,
                        "status": stat}
        print(f"{key:16s} {slope:8.2f} {body if body is not None else -1:6.1f} "
              f"{branch if branch is not None else -1:8.2f} "
              f"{resid*100:6.2f} {agree*100:6.2f} {stat}")
    (a.out / "model.json").write_text(json.dumps(
        {"protocol": "tools/microbench/control_branch/protocol.json",
         "device": a.device, "run_dir": str(a.run_dir),
         "ffma_latency": a.ffma_latency, "runs": runs,
         "cells": verdict}, indent=1))


if __name__ == "__main__":
    main()
