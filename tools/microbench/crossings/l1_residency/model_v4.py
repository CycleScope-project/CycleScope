#!/usr/bin/env python3
"""l1_residency protocol-v4 derivation
(tools/microbench/crossings/l1_residency/protocol-v4.json): the install
uses the default-hint cells with the occupancy-rule configuration; the
pinned-hint cells are reported.  Same probe / run layout as version 3.

  model_v4.py --device h100|b300 --root <run dir> --out <model.json>
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from model import ARRAY, CONFIGS, STEP, derive  # version-3 loader / knee rules

MAX_CTAS_PER_SM, MAX_WARPS_PER_SM, RESERVED_PER_CTA = 32, 64, 1024
MAX_SHARED = 228 * 1024


def config_rule(smem: int, block: int, static: int):
    need = smem + static + RESERVED_PER_CTA
    n_res = min(MAX_CTAS_PER_SM, MAX_WARPS_PER_SM // max(1, block // 32), MAX_SHARED // need)
    cfg = next((c for c in CONFIGS if c * 1024 >= n_res * need), CONFIGS[-1])
    return n_res, cfg


def implied_config(e: int, k_mean: float):
    return min(CONFIGS, key=lambda c: abs(c * 1024 - (k_mean - e)))


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
    ca, cb = runs["run-a"][0], runs["run-b"][0]
    default, pinned = {}, {}
    for name, da in ca.items():
        db = cb.get(name)
        p = da["params"]
        rec = {"params": p, "E_a": da["E"], "E_b": db["E"] if db else None,
               "valid": da["valid"] and bool(db and db["valid"]), "hit_a": da["hit"], "hit_b": db["hit"] if db else None}
        if int(p.get("requested_carveout_pct", -1)) < 0:
            n_res, cfg = config_rule(int(p["smem_bytes"]), int(p["block"]), int(p.get("static_shared_bytes", 0)))
            rec |= {"n_res_rule": n_res, "occupancy_api": p.get("occupancy_blocks"), "config_kib": cfg}
            if rec["E_a"] is not None and rec["E_b"] is not None:
                rec |= {"K_a": rec["E_a"] + cfg * 1024, "K_b": rec["E_b"] + cfg * 1024}
            default[name] = rec
        else:
            pinned[name] = rec
    ks = [v["K_a"] for v in default.values() if "K_a" in v] + [v["K_b"] for v in default.values() if "K_b" in v]
    all_valid = len(default) == 10 and all(v["valid"] and "K_a" in v for v in default.values())
    spread = (max(ks) - min(ks)) if ks else None
    diff_cells = sum(1 for v in default.values() if v.get("E_a") != v.get("E_b"))
    if not gate_ok or not all_valid:
        verdict = "unrealised"
    elif spread <= STEP and diff_cells == 0:
        verdict = "measured"
    elif spread <= 2 * STEP and diff_cells <= 1:
        verdict = "experimental"
    else:
        verdict = "unrealised"
    k_mean = statistics.fmean(ks) if ks else None
    data_array = round(k_mean / 1024) * 1024 if k_mean else None
    for v in pinned.values():
        if v["E_a"] is not None and k_mean:
            v["implied_config_kib"] = implied_config(v["E_a"], k_mean)
    model = {
        "cell": "l1_residency", "protocol": "tools/microbench/crossings/l1_residency/protocol-v4.json",
        "device": a.device, "root": str(a.root), "sass_gate_ok": gate_ok,
        "hit_min": {"run-a": runs["run-a"][1], "run-b": runs["run-b"][1]},
        "default_hint_cells": default, "K_values": ks, "spread_bytes": spread, "cells_differing_between_runs": diff_cells,
        "data_array_bytes": data_array, "reserved_bytes": 0, "unusable_bytes": (ARRAY - data_array) if data_array else None,
        "pinned_hint_cells": pinned, "verdict": verdict,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(model, indent=1) + "\n")
    print(a.device, verdict, "data_array", data_array, "spread", spread, "K", sorted(set(ks)))


if __name__ == "__main__":
    main()
