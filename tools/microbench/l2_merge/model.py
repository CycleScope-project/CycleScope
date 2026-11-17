#!/usr/bin/env python3
"""l2_merge protocol derivation (tools/microbench/l2_merge/protocol.json).

  model.py --device h100|b300 --root $A/microbench/l2_merge/<device>-<run> --out <model.json>

<root>/{run-a,run-b}/results.json (run_scan.py set m6r2): per config the
probe's TRIAL / CTA / RESULT lines and the ncu row (lts__t_bytes.sum).
Quantities per the frozen protocol:
  r(G)   = lts bytes / SM-side bytes of the ncu launch (aligned lockstep G)
  m(D)   = clamp(2 (1 - r(D)), 0, 1) for the G = 2 shift sweep
  window = D at the 0.5 crossing of m(D) (linear interpolation)
  N*     = argmin_N mean_G |r(G) - ceil(G/N)/G|, N in 1..8
  delivered = max over aligned G >= 2 of agg_B_per_ns * 1000 / mhz
Thresholds copied from the protocol; do not edit after data exists.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

WINDOW_AGREE = 0.10
DELIVERED_AGREE = 0.05
MAE_MAX = 0.06
REF_RANGE = (0.95, 1.05)
SPREAD_NS = 64.0
N_RANGE = range(1, 9)
SHIFTS_G2 = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096)
ALIGNED_G = (2, 3, 4, 6, 8, 11, 12, 16, 22, 32, 44, 132)


def fnum(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def load(run_dir: Path):
    d = json.loads((run_dir / "results.json").read_text())
    ncu = {row["name"]: row for row in d.get("ncu", [])}
    cfgs = {}
    for r in d["results"]:
        name = r["config"]["name"]
        kv = r["config"].get("kv", {})
        group = int(kv.get("group", 1))
        sm_bytes = next((t["total_bytes"] for t in r["trials"] if t.get("total_bytes")), None)
        row = ncu.get(name, {})
        lts = fnum(row.get("metrics", {}).get("lts__t_bytes.sum", {}).get("value"))
        m0 = [c.get("start_ns", 0.0) for c in r.get("ctas", []) if c.get("trial") == 1 and int(c["cta"]) % group == 0]
        res = r.get("result", {})
        mhz = res.get("mhz") or 0
        cfgs[name] = {
            "group": group, "shift": int(kv.get("shift_cycles", 0)), "sync": int(kv.get("sync_start", 0)) or int(kv.get("shift_cycles", 0)) > 0,
            "rot": int(kv.get("group_rot", 0)), "nsm": r["config"]["nsm"], "active_sms": r.get("active_sms"),
            "rc": r["rc"], "ncu_rc": row.get("rc"), "sm_bytes": sm_bytes, "lts_bytes": lts,
            "r": (lts / sm_bytes) if lts and sm_bytes else None,
            "m0_spread_ns": max(m0) if m0 else None,
            "agg_B_per_ns": res.get("agg_B_per_ns"), "mhz": mhz,
            "delivered_B_per_cycle": (res.get("agg_B_per_ns", 0) * 1000 / mhz) if mhz else None,
            "per_sm_B_per_cycle": res.get("per_sm_B_per_cycle"),
        }
    return cfgs


def valid(c):
    return c["rc"] == 0 and c["ncu_rc"] == 0 and c["r"] is not None and c["active_sms"] == c["nsm"] \
        and (c["m0_spread_ns"] is None or c["m0_spread_ns"] <= SPREAD_NS)


def derive(cfgs):
    out = {"aligned": {}, "shift_g2": {}, "shift_g4": {}, "references": {}, "invalid": []}
    for name, c in cfgs.items():
        m = re.match(r"tma_l2g_24mb_g(\d+)_(sync|sync128|shift(\d+)|lock|rot)$", name)
        if name.startswith("tma_l2s_"):
            out["references"][name] = c
            continue
        if not m:
            continue
        g, tag = int(m.group(1)), m.group(2)
        if not valid(c) and not tag.endswith(("lock", "rot")):
            out["invalid"].append(name)
        if tag.startswith("sync"):
            out["aligned"][g] = c
        elif tag.startswith("shift"):
            (out["shift_g2"] if g == 2 else out["shift_g4"])[int(m.group(3))] = c
        else:
            out["references"][name] = c
    # window
    pts = {0: out["aligned"].get(2)}
    pts.update(out["shift_g2"])
    curve = {d: max(0.0, min(1.0, 2 * (1 - c["r"]))) for d, c in sorted(pts.items()) if c and valid(c)}
    window = None
    ds = sorted(curve)
    for d0, d1 in zip(ds, ds[1:]):
        m0, m1 = curve[d0], curve[d1]
        if m0 >= 0.5 > m1:
            window = d0 + (m0 - 0.5) / (m0 - m1) * (d1 - d0)
            break
    # max requests
    rs = {g: c["r"] for g, c in out["aligned"].items() if valid(c)}
    mae = {n: statistics.fmean(abs(rs[g] - math.ceil(g / n) / g) for g in rs) for n in N_RANGE} if rs else {}
    n_star = min(mae, key=mae.get) if mae else None
    # delivered
    deliv = {g: c["delivered_B_per_cycle"] for g, c in out["aligned"].items() if valid(c) and g >= 2 and c["delivered_B_per_cycle"]}
    g_max = max(deliv, key=deliv.get) if deliv else None
    refs_ok = all(REF_RANGE[0] <= c["r"] <= REF_RANGE[1] for n, c in out["references"].items()
                  if c["r"] is not None and (n.startswith("tma_l2s_") or n.endswith("_rot")))
    # addendum 2026-11-05: the aggregate of this TMA probe equals nsm x the
    # per-SM stream rate of the disjoint reference at every G (the 4-stage
    # 16 KiB ring is latency-bound per SM), so its maximum is the probe's own
    # stream rate, not the delivery cap; flagged, not installed
    disjoint = next((c for n, c in out["references"].items() if n.startswith("tma_l2s_")), None)
    bound = None
    if deliv and disjoint and disjoint.get("per_sm_B_per_cycle") and disjoint.get("active_sms"):
        bound = disjoint["per_sm_B_per_cycle"] * disjoint["active_sms"]
    deliv_probe_bound = bool(deliv) and bound is not None and \
        (max(deliv.values()) - min(deliv.values())) / max(deliv.values()) <= 0.15 and \
        abs(max(deliv.values()) - bound) / bound <= 0.05
    return {
        "r_aligned": {str(g): rs.get(g) for g in sorted(out["aligned"])},
        "merged_fraction_g2": {str(d): round(v, 4) for d, v in curve.items()},
        "window_cycles": window, "mae_by_N": {str(n): round(v, 4) for n, v in mae.items()}, "N_star": n_star,
        "delivered_B_per_cycle": deliv.get(g_max), "delivered_at_G": g_max,
        "delivered_by_G": {str(g): round(v, 1) for g, v in deliv.items()},
        "delivered_probe_bound": deliv_probe_bound, "per_sm_stream_bound_B_per_cycle": bound,
        "shift_g4": {str(d): c["r"] for d, c in sorted(out["shift_g4"].items())},
        "references": {n: {"r": c["r"], "per_sm_B_per_cycle": c["per_sm_B_per_cycle"], "active_sms": c["active_sms"]}
                       for n, c in out["references"].items()},
        "references_ok": refs_ok, "invalid": out["invalid"],
        "ncu_rc": {n: c["ncu_rc"] for n, c in cfgs.items()},
    }


def rel(a, b):
    m = (a + b) / 2
    return abs(a - b) / m if m else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    runs = {r: derive(load(a.root / r)) for r in ("run-a", "run-b")}
    ra, rb = runs["run-a"], runs["run-b"]
    out = {"protocol": "tools/microbench/l2_merge/protocol.json", "device": a.device, "root": str(a.root), "runs": runs}
    unreal = []
    for r, d in runs.items():
        if d["window_cycles"] is None:
            unreal.append(f"{r}: no 0.5 crossing")
        if d["N_star"] is None or d["delivered_B_per_cycle"] is None:
            unreal.append(f"{r}: missing aligned cells")
        if not d["references_ok"]:
            unreal.append(f"{r}: references outside 0.95..1.05")
        if len(d["invalid"]) > 1:
            unreal.append(f"{r}: {len(d['invalid'])} invalid aligned/shift configs")
        if any(rc not in (0, None) for rc in d["ncu_rc"].values()):
            unreal.append(f"{r}: ncu rc != 0")
    if unreal:
        out |= {"verdict": "unrealised", "why": unreal}
    else:
        w = (ra["window_cycles"] + rb["window_cycles"]) / 2
        w_ag = rel(ra["window_cycles"], rb["window_cycles"])
        n_ok = ra["N_star"] == rb["N_star"] and max(ra["mae_by_N"][str(ra["N_star"])], rb["mae_by_N"][str(rb["N_star"])]) <= MAE_MAX
        dv = (ra["delivered_B_per_cycle"] + rb["delivered_B_per_cycle"]) / 2
        d_ag = rel(ra["delivered_B_per_cycle"], rb["delivered_B_per_cycle"])
        measured = w_ag <= WINDOW_AGREE and n_ok and d_ag <= DELIVERED_AGREE
        probe_bound = ra["delivered_probe_bound"] and rb["delivered_probe_bound"]
        out |= {"window_cycles": w, "window_agree": w_ag, "max_requests_per_read": ra["N_star"],
                "max_requests_runs": [ra["N_star"], rb["N_star"]], "delivered_bytes_per_cycle": dv,
                "delivered_agree": d_ag, "delivered_probe_bound": probe_bound,
                "delivered_verdict": "unrealised (probe stream-rate bound, addendum 2026-11-05)" if probe_bound
                else ("measured" if d_ag <= DELIVERED_AGREE else "experimental"),
                "window_verdict": "measured" if w_ag <= WINDOW_AGREE else "experimental",
                "max_requests_verdict": "measured" if n_ok else "experimental",
                "verdict": "measured" if measured else "experimental"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(a.device, out["verdict"], {k: out.get(k) for k in ("window_cycles", "max_requests_per_read", "delivered_bytes_per_cycle", "why")})


if __name__ == "__main__":
    main()
