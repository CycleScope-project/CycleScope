#!/usr/bin/env python3
"""l2_merge protocol v2 derivation (tools/microbench/l2_merge/protocol-v2.json):
delivered_bytes_per_cycle from the LDG lockstep set (run_scan.py --set m1r4).

  model_v2.py --device h100|b300 --root $A/microbench/l2_merge/<device>-<run> --out <model.json>

<root>/{run-a,run-b}/results.json: per config the probe's TRIAL / CTA /
RESULT lines and the ncu row (lts__t_bytes.sum, l1tex__t_bytes.sum,
dram__bytes_read.sum).  <root>/build/sass_check.txt: the sass gate
(first line PASS / FAIL).  Quantities per the frozen protocol:
  r         = lts__t_bytes / l1tex__t_bytes of the ncu launch
  delivered = max over valid lock configs of agg_B_per_ns * 1000 / mhz,
              attaining config must have r <= R_MAX and agg >= (1 + EXCESS_MIN)
              x the disjoint reference aggregate
Thresholds copied from the protocol; do not edit after data exists.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

AGREE_MEASURED = 0.05
AGREE_EXPERIMENTAL = 0.15
R_MAX = 0.75
EXCESS_MIN = 0.20
LOCK_RE = re.compile(r"^ld_l2g_24mb_g(\d+)_lock(_stride2mb)?$")
ROT_RE = re.compile(r"^ld_l2g_24mb_g(\d+)_rot$")


def fnum(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def load(run_dir: Path) -> dict:
    d = json.loads((run_dir / "results.json").read_text())
    ncu = {row["name"]: row for row in d.get("ncu", [])}
    cfgs = {}
    for r in d["results"]:
        name = r["config"]["name"]
        row = ncu.get(name, {})
        met = row.get("metrics", {})
        lts = fnum(met.get("lts__t_bytes.sum", {}).get("value"))
        l1 = fnum(met.get("l1tex__t_bytes.sum", {}).get("value"))
        res = r.get("result", {})
        mhz = res.get("mhz") or 0
        pl = r.get("placement") or {}
        cfgs[name] = {
            "nsm": r["config"]["nsm"], "group": int(r["config"].get("kv", {}).get("group", 1)),
            "rc": r["rc"], "ncu_rc": row.get("rc"),
            "distinct_sm": pl.get("distinct_sm"), "active_sms": r.get("active_sms"),
            "lts_bytes": lts, "l1tex_bytes": l1,
            "dram_read_bytes": fnum(met.get("dram__bytes_read.sum", {}).get("value")),
            "r": (lts / l1) if lts and l1 else None,
            "agg_B_per_ns": res.get("agg_B_per_ns"), "mhz": mhz,
            "agg_B_per_cycle": (res.get("agg_B_per_ns", 0) * 1000 / mhz) if mhz else None,
            "per_sm_B_per_cycle": res.get("per_sm_B_per_cycle"),
            "agg_cv_pct": r.get("agg_cv_pct"),
        }
    return cfgs


def valid(c) -> bool:
    return c["rc"] == 0 and c["ncu_rc"] == 0 and c["r"] is not None \
        and c["active_sms"] == c["nsm"] and c["agg_B_per_cycle"] is not None


def derive(cfgs: dict) -> dict:
    lock, rot, invalid = {}, {}, []
    disjoint = None
    for name, c in cfgs.items():
        if LOCK_RE.match(name):
            lock[name] = c
            if not valid(c):
                invalid.append(name)
        elif ROT_RE.match(name):
            rot[name] = c
        elif name.startswith("ld_l2s_24mb_t1024_n") and name.endswith("_base"):
            disjoint = c
            if not valid(c):
                invalid.append(name)
    cand = {n: c for n, c in lock.items() if valid(c)}
    best = max(cand, key=lambda n: cand[n]["agg_B_per_cycle"]) if cand else None
    ref_agg = disjoint["agg_B_per_cycle"] if disjoint and valid(disjoint) else None
    why = []
    if best is None:
        why.append("no valid lock config")
    else:
        b = cand[best]
        if b["r"] > R_MAX:
            why.append(f"attaining config {best} r {b['r']:.3f} > {R_MAX}")
        if ref_agg is None:
            why.append("disjoint reference invalid")
        elif b["agg_B_per_cycle"] < (1 + EXCESS_MIN) * ref_agg:
            why.append(f"attaining config {best} aggregate {b['agg_B_per_cycle']:.0f} < "
                       f"{1 + EXCESS_MIN:.2f} x disjoint {ref_agg:.0f}")
    if len(invalid) > 1:
        why.append(f"{len(invalid)} invalid lock/reference configs")
    b = cand.get(best) if best else None
    return {
        "delivered_B_per_cycle": b["agg_B_per_cycle"] if b else None,
        "attaining": best,
        "attaining_r": b["r"] if b else None,
        "merged_share_B_per_cycle": (b["agg_B_per_cycle"] * (1 - b["r"])) if b else None,
        "disjoint_B_per_cycle": ref_agg,
        "disjoint_per_sm_B_per_cycle": disjoint["per_sm_B_per_cycle"] if disjoint else None,
        "lock": {n: {k: c[k] for k in ("nsm", "active_sms", "agg_B_per_cycle", "per_sm_B_per_cycle",
                                      "r", "dram_read_bytes", "agg_cv_pct", "rc", "ncu_rc")}
                 for n, c in sorted(lock.items(), key=lambda kv: (kv[1]["group"], kv[0]))},
        "rot": {n: {k: c[k] for k in ("nsm", "active_sms", "agg_B_per_cycle", "per_sm_B_per_cycle", "r")}
                for n, c in sorted(rot.items(), key=lambda kv: kv[1]["group"])},
        "invalid": invalid, "why": why,
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
    sass = (a.root / "build" / "sass_check.txt")
    sass_first = sass.read_text().splitlines()[0] if sass.exists() and sass.read_text().strip() else "MISSING"
    out = {"protocol": "tools/microbench/l2_merge/protocol-v2.json", "device": a.device,
           "root": str(a.root), "sass_gate": sass_first, "runs": runs}
    unreal = [f"{r}: {w}" for r, d in runs.items() for w in d["why"]]
    if not sass_first.startswith("PASS"):
        unreal.append(f"sass gate: {sass_first}")
    ra, rb = runs["run-a"], runs["run-b"]
    if not unreal:
        va, vb = ra["delivered_B_per_cycle"], rb["delivered_B_per_cycle"]
        agree = rel(va, vb)
        if agree > AGREE_EXPERIMENTAL:
            unreal.append(f"runs differ by {agree:.1%} > {AGREE_EXPERIMENTAL:.0%}")
        else:
            out |= {"delivered_bytes_per_cycle": (va + vb) / 2, "runs_values": [va, vb], "agree": agree,
                    "attaining": [ra["attaining"], rb["attaining"]],
                    "verdict": "measured" if agree <= AGREE_MEASURED else "experimental"}
    if unreal:
        out |= {"verdict": "unrealised", "why": unreal}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(a.device, out["verdict"], {k: out.get(k) for k in ("delivered_bytes_per_cycle", "agree", "attaining", "why")})


if __name__ == "__main__":
    main()
