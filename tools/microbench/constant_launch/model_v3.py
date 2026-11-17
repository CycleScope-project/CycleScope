#!/usr/bin/env python3
"""constant_launch derivation, protocol version 3
(tools/microbench/constant_launch/protocol-v3.json).

Per variant the v2 derivation (model_v2.py) unchanged; variants
cl.b<B>.s256 (vector, control) and cl.u.b<B>.s256 (uniform).
Control: vector cells reproduce the v2 first / second values within 5 %.
Decision (declared): uniform bank-0 rule when cl.u.b0.s256 first and second
are measured and second within 10 % of first -> bank0_uniform_access_cycles
= first; vector rule bank0_access_cycles = v2 value (needs the v2 cells
measured, they are); cold miss replaced by the mean of vector and uniform
bank-3 first only if both measured and within 10 % of each other.

Usage: model_v3.py --run-a <dir> --run-b <dir> --checks <checks.json>
                   --tree configs/trees/<dev>.json --v2 <v2 model.json>
                   --out model.json
"""
import argparse, json, re, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_v2  # noqa: E402

model_v2.LAUNCH_RE = re.compile(
    r"^LAUNCH k_cl variant=(cl\.(?:u\.)?b\d+\.s\d+) launch=(\d+) t=([\d,]+) idx=(\d+)")
CONTROL = 0.05


def run_cells(run_dir, gated, sub):
    acc = {}
    for line in (Path(run_dir) / "launches.log").read_text().splitlines():
        m = model_v2.LAUNCH_RE.match(line.strip())
        if not m:
            continue
        v, launch = m.group(1), int(m.group(2))
        if launch in model_v2.DROP or v not in gated:
            continue
        t = [int(x) for x in m.group(3).split(",")]
        if len(t) != 33:
            raise SystemExit(f"{v} launch {launch}: {len(t)} clocks")
        d = [t[j + 1] - t[j] for j in range(32)]
        a = sub[v]
        c = acc.setdefault(v, {"first": [], "load0": [], "second": []})
        c["load0"].append(d[1] - a)
        c["first"].extend(x - a for x in d[2:17])
        c["second"].extend(x - a for x in d[17:32])
    return {v: {k: model_v2.stats(xs) for k, xs in c.items()} for v, c in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    for k in ("--run-a", "--run-b", "--checks", "--tree", "--v2", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    checks = json.loads(Path(a.checks).read_text())["variants"]
    gated = {v for v, c in checks.items() if c.get("ok")}
    sub = {v: model_v2.result_latency(a.tree, checks[v].get("between_op", "")) for v in gated}
    runs = {"a": run_cells(a.run_a, gated, sub), "b": run_cells(a.run_b, gated, sub)}
    out = {"protocol": "tools/microbench/constant_launch/protocol-v3.json", "runs": {"a": a.run_a, "b": a.run_b},
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
            ok = agree is not None and agree <= model_v2.RUN_AGREE and max(ra["spread"], rb["spread"]) <= model_v2.SPREAD
            cell[t] = {"value": m, "run_agree": agree, "spread_max": max(ra["spread"], rb["spread"]),
                       "verdict": "measured" if ok else "experimental"}
        out["cells"][v] = cell
    v2 = json.loads(Path(a.v2).read_text())["cells"]
    control = {}
    for v in ("cl.b0.s256", "cl.b3.s256"):
        for t in ("first", "second"):
            x3, x2 = out["cells"].get(v, {}).get(t, {}).get("value"), v2.get(v, {}).get(t, {}).get("value")
            control[f"{v}.{t}"] = {"v3": x3, "v2": x2, "rel": abs(x3 - x2) / x2 if x3 and x2 else None}
    out["control"] = control
    control_ok = all(c["rel"] is not None and c["rel"] <= CONTROL for c in control.values())
    out["control_ok"] = control_ok
    inst = {}
    if control_ok:
        u = out["cells"].get("cl.u.b0.s256", {})
        f, s = u.get("first", {}), u.get("second", {})
        ok = f.get("verdict") == "measured" and s.get("verdict") == "measured" \
            and abs(s["value"] - f["value"]) / f["value"] <= model_v2.SECOND_VS_FIRST
        inst["bank0_uniform_access_cycles"] = round(f["value"], 2) if ok else None
        if not ok:
            inst["uniform_reason"] = {"first": f, "second": s}
        vf, uf = v2.get("cl.b3.s256", {}).get("first", {}), out["cells"].get("cl.u.b3.s256", {}).get("first", {})
        vf3 = out["cells"].get("cl.b3.s256", {}).get("first", {})
        both = vf3.get("verdict") == "measured" and uf.get("verdict") == "measured"
        close = both and abs(vf3["value"] - uf["value"]) / ((vf3["value"] + uf["value"]) / 2) <= 0.10
        inst["miss_latency_cycles"] = round((vf3["value"] + uf["value"]) / 2, 2) if close else None
        inst["miss_cells"] = {"vector_b3_first": vf3, "uniform_b3_first": uf, "v2_vector_b3_first": vf}
    else:
        inst["reason"] = "control failed: run not used"
    out["install"] = inst
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for v, c in out["cells"].items():
        print(v, {t: (round(x.get("value", 0) or 0, 2), x.get("verdict"),
                      round(x.get("spread_max") or 0, 3), round(x.get("run_agree") or 0, 3)) for t, x in c.items()})
    print("subtracted", sub)
    print("control", {k: round(c["rel"], 4) if c["rel"] is not None else None for k, c in control.items()}, "ok", control_ok)
    print("install", {k: v for k, v in inst.items() if k != "miss_cells"})


if __name__ == "__main__":
    main()
