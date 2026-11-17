#!/usr/bin/env python3
"""l1_residency protocol-v3 derivation
(tools/microbench/crossings/l1_residency/protocol-v3.json).

  model.py --device h100|b300 --root $A/microbench/crossings/l1_residency/<device>-<run>
           --out <model.json>

<root>/{run-a,run-b}/l1cap.jsonl hold the two runs (variant
`l1cap.<cell>.f<F>k`, u = hops, params carveout_kib / smem_bytes / block /
requested_carveout_pct / static_shared_bytes / occupancy_blocks);
<root>/checks.json the SASS gate.  Thresholds are copied from the frozen
protocol text; do not edit them after data exists.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# --- thresholds copied verbatim from protocol-v3.json ----------------------
KNEE_TOL = 0.10          # lat(F') <= 1.10 * hit(cell)
HIT_AGREE = 0.05         # hit(cell) within 5 % of the smallest hit of the run
STEP = 4096              # F step; measured: spread <= 1 step, experimental: <= 2 steps
SLOPE_RANGE = (-1.10, -0.90)
CONFIGS = [0, 8, 16, 32, 64, 100, 132, 164, 196, 228]       # KiB, cc 9.0 / 10.x
ARRAY = 256 * 1024
MEMLAT_L1 = {"h100": 52.5, "b300": 38.46875}                # reported only
MAX_CTAS_PER_SM, MAX_WARPS_PER_SM, RESERVED_PER_CTA = 32, 64, 1024


def load(run_dir: Path):
    """cell -> {"params": {...}, "lat": {F: cycles per hop}}"""
    cells: dict[str, dict] = {}
    p = run_dir / "l1cap.jsonl"
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "samples" not in r:
            continue
        v = r["variant"].split(".")          # l1cap.<cell...>.f<F>k
        cell = ".".join(v[1:-1])
        f = int(v[-1][1:-1]) * 1024
        c = cells.setdefault(cell, {"params": r.get("params", {}), "lat": {}})
        c["lat"][f] = statistics.median(r["samples"]) / int(r["u"])
    return cells


def knee(lat: dict[int, float], hit: float):
    """largest F such that every F' <= F is within KNEE_TOL of hit (first crossing)"""
    e = None
    for f in sorted(lat):
        if lat[f] <= (1 + KNEE_TOL) * hit:
            e = f
        else:
            break
    return e


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")
    return b, my - b * mx


def config_of(l1_bytes: float) -> int:
    """supported configuration (KiB) closest to 256 KiB - l1_bytes"""
    want = (ARRAY - l1_bytes) / 1024
    return min(CONFIGS, key=lambda c: abs(c - want))


def hypothesis_config(smem: int, block: int, static: int) -> int:
    n_max = min(MAX_CTAS_PER_SM, MAX_WARPS_PER_SM // max(1, block // 32))
    need = n_max * (smem + static + RESERVED_PER_CTA) / 1024
    return next((c for c in CONFIGS if c >= need), CONFIGS[-1])


def derive(run_dir: Path):
    cells = load(run_dir)
    hits = {c: d["lat"].get(STEP) for c, d in cells.items()}
    hit_min = min(h for h in hits.values() if h is not None)
    out = {}
    for c, d in cells.items():
        h = hits[c]
        valid = h is not None and abs(h - hit_min) <= HIT_AGREE * hit_min
        e = knee(d["lat"], h) if valid else None
        out[c] = {"params": d["params"], "hit": h, "valid": valid, "E": e,
                  "points": {str(f): round(v, 2) for f, v in sorted(d["lat"].items())}}
    return out, hit_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    runs = {r: derive(a.root / r) for r in ("run-a", "run-b")}
    checks = json.loads((a.root / "checks.json").read_text()) if (a.root / "checks.json").exists() else {}
    gate = {k: v.get("ok") for k, v in checks.get("l1cap", {}).get("kernels", {}).items()}
    gate_ok = bool(gate) and all(gate.values())

    def pk(p):  # cell key for the carveout family
        return (int(p.get("carveout_kib", -1)), int(p.get("smem_bytes", 0)), int(p.get("block", 32)))

    cells_a, cells_b = runs["run-a"][0], runs["run-b"][0]
    carve = {}      # C -> per-run E for the S = 0, block 32 pinned cells
    smem_vs = {}
    default = {}
    for c, da in cells_a.items():
        db = cells_b.get(c)
        p = da["params"]
        pct = int(p.get("requested_carveout_pct", -1))
        rec = {"E_a": da["E"], "E_b": db["E"] if db else None, "hit_a": da["hit"],
               "hit_b": db["hit"] if db else None, "valid": da["valid"] and bool(db and db["valid"]),
               "params": p, "points_a": da["points"], "points_b": db["points"] if db else None}
        C, S, B = pk(p)
        if pct >= 0 and S == 0 and B == 32:
            carve[C] = rec
        if pct >= 0:
            smem_vs[f"c{C}.s{S // 1024}k"] = {"C_kib": C, "S_bytes": S, "E_a": da["E"], "E_b": rec["E_b"]}
        if pct < 0:
            default[c] = rec
    # K(C) over C >= 8
    ks = {}
    for C, r in sorted(carve.items()):
        if C >= 8 and r["valid"] and r["E_a"] is not None and r["E_b"] is not None:
            ks[C] = {"E_a": r["E_a"], "E_b": r["E_b"], "K_a": r["E_a"] + C * 1024, "K_b": r["E_b"] + C * 1024}
    all_valid = all(r["valid"] and r["E_a"] is not None and r["E_b"] is not None
                    for C, r in carve.items() if C >= 8) and len([C for C in carve if C >= 8]) == len(CONFIGS) - 1
    k_vals = [v["K_a"] for v in ks.values()] + [v["K_b"] for v in ks.values()]
    spread = (max(k_vals) - min(k_vals)) if k_vals else None
    runs_equal = all(v["E_a"] == v["E_b"] for v in ks.values()) if ks else False
    one_step = sum(1 for v in ks.values() if v["E_a"] != v["E_b"])
    slope = icpt = None
    if len(ks) >= 2:
        xs = [C * 1024 for C in ks] + [C * 1024 for C in ks]
        ys = [ks[C]["E_a"] for C in ks] + [ks[C]["E_b"] for C in ks]
        slope, icpt = ols(xs, ys)
    slope_ok = slope is not None and SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]
    if not gate_ok or not all_valid or not ks:
        verdict = "unrealised"
    elif spread <= STEP and runs_equal and slope_ok:
        verdict = "measured"
    elif spread <= 2 * STEP and one_step <= 1 and slope_ok:
        verdict = "experimental"
    else:
        verdict = "unrealised"
    data_array = round(statistics.fmean(k_vals) / 1024) * 1024 if k_vals else None
    # diagnostics
    for c, r in default.items():
        p = r["params"]
        if r["E_a"] is not None and data_array:
            l1 = r["E_a"] + (ARRAY - data_array)
            r["implied_config_kib"] = config_of(l1)
            r["hypothesis_config_kib"] = hypothesis_config(int(p.get("smem_bytes", 0)), int(p.get("block", 32)),
                                                           int(p.get("static_shared_bytes", 0)))
            r["agree"] = r["implied_config_kib"] == r["hypothesis_config_kib"]
        for k in ("points_a", "points_b"):
            r.pop(k, None)
    for C, r in carve.items():
        r.pop("points_a", None); r.pop("points_b", None)
    model = {
        "cell": "l1_residency", "protocol": "tools/microbench/crossings/l1_residency/protocol-v3.json",
        "device": a.device, "root": str(a.root), "sass_gate_ok": gate_ok, "sass_gate": gate,
        "hit_min": {"run-a": runs["run-a"][1], "run-b": runs["run-b"][1]}, "memlat_l1_reference": MEMLAT_L1[a.device],
        "carveout_cells": {str(C): r for C, r in sorted(carve.items())},
        "K": {str(C): v for C, v in ks.items()}, "spread_bytes": spread, "runs_equal": runs_equal,
        "fit_E_vs_C": {"slope": slope, "intercept": icpt},
        "data_array_bytes": data_array, "reserved_bytes": 0,
        "unusable_bytes": (ARRAY - data_array) if data_array else None,
        "smem_vs_carveout": smem_vs, "default_hint_cells": default,
        "verdict": verdict,
        "note": "engine uses data_array - reserved - launch shared bytes; installed as data_array_bytes = mean K(C), reserved_bytes = 0",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(model, indent=1) + "\n")
    print(a.device, verdict, "data_array", data_array, "spread", spread, "slope", slope)


if __name__ == "__main__":
    main()
