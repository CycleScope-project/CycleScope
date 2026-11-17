#!/usr/bin/env python3
"""fetch_parallelism from the k_mix variants of an ifetch version-1 run
(protocol-v2.json addendum 2026-11-05, rules F7 / F8 of blackwell-prototype).

Per mode (all / none = two independent runs of the same binary), warm
passes only:
  a0           level-0 intercept of k_line W=1 (least squares over the
               three smallest footprints), cpi_c(N) = (T - a0) / N
  cpi_line_L2  median warm cpi_c of k_line W=1 on its L2 plateau
  q(W, N)      (cpi_mix(W, N) - 1) / (cpi_line_L2 - 1) for every k_mix
               point whose total footprint W * N * 16 B > capacity_l1
  p = 4 / q(4); streams per SMSP = round(4 / q(16));
  fetch_parallelism = 4 * streams per SMSP
Acceptance: modes agree within 5 % on q(4), q(8), q(16); L2 plateau of
k_line has >= 3 footprints.  Verdict experimental when accepted, else
unrealised (value 1 kept).

Usage: model_parallelism.py --run <run dir> --capacity-l1 <bytes>
                            --out <model.json>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as m1  # noqa: E402

INST_BYTES = 16
MODES = ("all", "none")
AGREE = 0.05


def mode_table(run: Path, mode: str, capacity_l1: int) -> dict:
    records, stats = m1.parse_launches(run / mode / "launches.log")
    cells = m1.build_cells(records)
    line_ns = sorted({r["n"] for r in records
                      if r["kind"] == "line" and r["w"] == 1})
    t_line = {n: m1.warm_T(cells, f"line.n{n}.w1") for n in line_ns}
    t_line = {n: t for n, t in t_line.items() if t is not None}
    ns = sorted(t_line)
    if len(ns) < 3:
        raise SystemExit(f"{mode}: fewer than 3 stable k_line footprints")
    a0, _, _ = m1._linfit([(n, t_line[n]) for n in ns[:3]])
    cpi_c = {n: (t_line[n] - a0) / n for n in ns}
    plats = m1.plateaus(cpi_c, ns)
    l2 = plats["L2"]
    out = {"parse_stats": stats, "a0": a0,
           "line_cpi_c": {str(n): cpi_c[n] for n in ns},
           "plateaus": plats, "l2_plateau_ns": l2}
    if len(l2) < 3:
        out["error"] = "k_line L2 plateau has fewer than 3 footprints"
        return out
    cpi_l2 = statistics.median(cpi_c[n] for n in l2)
    out["cpi_line_l2"] = cpi_l2
    q = {}
    skipped = []
    for r in records:
        if r["kind"] != "mix":
            continue
        key = (r["w"], r["n"])
        if key in q or key in skipped:
            continue
        if r["w"] * r["n"] * INST_BYTES <= capacity_l1:
            skipped.append(key)
            continue
        t = m1.warm_T(cells, r["key"])
        if t is None:
            skipped.append(key)
            continue
        cpi_mix = (t - a0) / r["n"]
        q[key] = {"T_warm": t, "cpi_mix": cpi_mix,
                  "q": (cpi_mix - 1.0) / (cpi_l2 - 1.0),
                  "total_bytes": r["w"] * r["n"] * INST_BYTES}
    out["mix"] = {f"w{w}.n{n}": v for (w, n), v in sorted(q.items())}
    out["mix_skipped"] = [f"w{w}.n{n}" for w, n in sorted(skipped)]
    qw = {}
    for (w, _), v in q.items():
        qw.setdefault(w, []).append(v["q"])
    out["q_by_w"] = {str(w): {"median": statistics.median(v), "n": len(v),
                              "values": sorted(v)}
                     for w, v in sorted(qw.items())}
    if 4 in qw:
        out["p_w4"] = 4.0 / statistics.median(qw[4])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--capacity-l1", type=int, required=True,
                    help="installed capacity_l1_bytes of the device")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    out = {"protocol": "tools/microbench/ifetch/protocol-v2.json"
                       " (addendum 2026-11-05, fetch_parallelism)",
           "run": str(run), "capacity_l1_bytes": a.capacity_l1, "modes": {}}
    for mode in MODES:
        if (run / mode / "launches.log").exists():
            out["modes"][mode] = mode_table(run, mode, a.capacity_l1)
    modes = [k for k, v in out["modes"].items() if "error" not in v]
    field = {"value": 1, "verdict": "unrealised"}
    reasons = []
    if len(modes) < 2:
        reasons.append("fewer than two modes with a k_line L2 plateau")
    else:
        agree = {}
        for w in (4, 8, 16):
            vals = [out["modes"][k]["q_by_w"].get(str(w), {}).get("median")
                    for k in modes]
            if any(v is None for v in vals):
                reasons.append(f"q({w}) missing in a mode")
                continue
            rel = abs(vals[0] - vals[1]) / statistics.mean(vals)
            agree[f"q{w}"] = {"values": vals, "rel": rel, "ok": rel <= AGREE}
            if rel > AGREE:
                reasons.append(f"q({w}) modes differ by {100 * rel:.1f} %")
        field["agreement"] = agree
        if not reasons:
            q16 = statistics.mean(agree["q16"]["values"])
            q8 = statistics.mean(agree["q8"]["values"])
            q4 = statistics.mean(agree["q4"]["values"])
            streams = max(1, round(4.0 / q16))
            field.update({
                "q4": q4, "p_w4": 4.0 / q4, "q8": q8, "q16": q16,
                "streams_per_smsp": streams, "value": 4 * streams,
                "verdict": "experimental",
                "note": "F8 intermediate verdict (q(8) and q(16) between 1 "
                        "and W/4): channel form with 4 x round(4 / q(16)) "
                        "arrival channels per SM",
            })
    if reasons:
        field["reasons"] = reasons
    out["fields"] = {"fetch_parallelism": field}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(field, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
