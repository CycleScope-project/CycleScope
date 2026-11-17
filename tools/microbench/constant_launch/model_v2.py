#!/usr/bin/env python3
"""constant_launch derivation, protocol version 2
(tools/microbench/constant_launch/protocol-v2.json).

Per launch (launches 0 and 1 dropped): clocks t_0..t_32, d_j = t_{j+1} - t_j.
d_j (j >= 1) measures load_{j-1}: first traversal loads 0..15 = d_1..d_16,
second traversal loads 0'..14' = d_17..d_31.  x = d - a, a = installed
instruction.service result latency of the in-between ALU class (gate's
between_op; 0 when none).
Per variant and run: first = median x over d_2..d_16, load0 = d_1,
second = median x over d_17..d_31 (all kept launches pooled); p10 / p90.
Cell measured: run agreement <= 5 %, (p90 - p10) / median <= 30 % per run.
Decision (declared): bank0_access_cycles = mean of the two runs' bank-0
S = 256 first, installed if first and second are measured and second is
within 10 % of first; miss_latency_cycles = mean of the bank-3 S = 256
first when measured.

Usage: model_v2.py --run-a <dir> --run-b <dir> --checks <checks.json>
                   --tree configs/trees/<dev>.json --out model.json
"""
import argparse, json, re, statistics
from pathlib import Path

LAUNCH_RE = re.compile(r"^LAUNCH k_cl variant=cl\.b(\d+)\.s(\d+) launch=(\d+) t=([\d,]+) idx=(\d+)")
DROP = {0, 1}
RUN_AGREE, SPREAD, SECOND_VS_FIRST = 0.05, 0.30, 0.10


def q(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(p * (len(s) - 1))))]


def stats(xs):
    med = statistics.median(xs)
    return {"median": med, "p10": q(xs, 0.1), "p90": q(xs, 0.9), "n": len(xs),
            "spread": (q(xs, 0.9) - q(xs, 0.1)) / med if med else None}


def result_latency(tree_path, sass_class):
    """installed instruction.service result latency of one class (root constant
    or the class point); None when the class has no point."""
    if not sass_class:
        return 0.0
    from types import SimpleNamespace
    import cyclescope.arch.blackwell.rules as bw_rules
    import cyclescope.arch.hopper.rules as hp_rules
    t = json.loads(Path(tree_path).read_text())
    rules = hp_rules if "h100" in Path(tree_path).name else bw_rules
    cls = rules.operation_class(SimpleNamespace(opcode=sass_class, operands=[], text=sass_class))
    nid = f"instruction.service.root/predicated_on=1/operation_class={cls}"
    hits = [n for n in t["nodes"] if n["id"] == nid]
    if len(hits) != 1:
        raise SystemExit(f"result latency of {sass_class} (class {cls}): node {nid} not in tree")
    return float(hits[0]["parameter"]["fields"]["result_latency_cycles"]["model"]["value"])


def run_cells(run_dir, gated, sub):
    acc = {}
    for line in (Path(run_dir) / "launches.log").read_text().splitlines():
        m = LAUNCH_RE.match(line.strip())
        if not m:
            continue
        b, s, launch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        v = f"cl.b{b}.s{s}"
        if launch in DROP or v not in gated:
            continue
        t = [int(x) for x in m.group(4).split(",")]
        if len(t) != 33:
            raise SystemExit(f"{v} launch {launch}: {len(t)} clocks")
        d = [t[j + 1] - t[j] for j in range(32)]
        a = sub[v]
        c = acc.setdefault(v, {"first": [], "load0": [], "second": []})
        c["load0"].append(d[1] - a)
        c["first"].extend(x - a for x in d[2:17])
        c["second"].extend(x - a for x in d[17:32])
    return {v: {k: stats(xs) for k, xs in c.items()} for v, c in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    for k in ("--run-a", "--run-b", "--checks", "--tree", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    checks = json.loads(Path(a.checks).read_text())["variants"]
    gated = {v for v, c in checks.items() if c.get("ok")}
    sub = {v: result_latency(a.tree, checks[v].get("between_op", "")) for v in gated}
    runs = {"a": run_cells(a.run_a, gated, sub), "b": run_cells(a.run_b, gated, sub)}
    out = {"protocol": "tools/microbench/constant_launch/protocol-v2.json", "runs": {"a": a.run_a, "b": a.run_b},
           "gated": sorted(gated), "subtracted": sub, "per_run": runs, "cells": {}}
    for v in sorted(gated):
        cell = {}
        for t in ("first", "load0", "second"):
            ra, rb = runs["a"].get(v, {}).get(t), runs["b"].get(v, {}).get(t)
            if not ra or not rb:
                cell[t] = {"verdict": "unrealised"}
                continue
            m = (ra["median"] + rb["median"]) / 2
            agree = abs(ra["median"] - rb["median"]) / m if m else None
            ok = agree is not None and agree <= RUN_AGREE and max(ra["spread"], rb["spread"]) <= SPREAD
            cell[t] = {"value": m, "run_agree": agree, "spread_max": max(ra["spread"], rb["spread"]),
                       "verdict": "measured" if ok else "experimental"}
        out["cells"][v] = cell
    inst = {}
    b0 = out["cells"].get("cl.b0.s256", {})
    f, s = b0.get("first", {}), b0.get("second", {})
    if f.get("verdict") == "measured" and s.get("verdict") == "measured" \
            and abs(s["value"] - f["value"]) / f["value"] <= SECOND_VS_FIRST:
        inst["bank0_access_cycles"] = round(f["value"], 2)
    else:
        inst["bank0_access_cycles"] = None
        inst["bank0_reason"] = {"first": f, "second": s}
    b3 = out["cells"].get("cl.b3.s256", {}).get("first", {})
    inst["miss_latency_cycles"] = round(b3["value"], 2) if b3.get("verdict") == "measured" else None
    out["install"] = inst
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for v, c in out["cells"].items():
        print(v, {t: (round(x.get("value", 0) or 0, 2), x.get("verdict"),
                      round(x.get("spread_max") or 0, 3), round(x.get("run_agree") or 0, 3)) for t, x in c.items()})
    print("subtracted", sub, "install", inst)


if __name__ == "__main__":
    main()
