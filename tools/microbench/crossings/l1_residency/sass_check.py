#!/usr/bin/env python3
"""SASS gate for the l1_residency probe (crossings protocol-v2).

k_l1cap: lane 0 loops {warm pass; t0; timed pass; t1; out[r] = t1 - t0}, so
CS2R windows alternate between the timed chase (LDG + address arithmetic +
loop control only) and the inter-rep window (warm-pass LDGs + the out[r]
STG).  Gate: every LDG anywhere is plain LDG.E; no memory-class
instruction other than LDG / STG anywhere; at least one window contains
LDGs and no STG (the timed chase).
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc

MEM_ROOTS = ("LD", "ST", "LDG", "STG", "LDS", "STS", "LDC", "ATOM", "ATOMG",
             "ATOMS", "RED", "REDG", "REDS", "LDGSTS", "LDSM", "STSM",
             "LDL", "STL")


def check(name, k):
    if not re.match(r"_Z\d+k_l1cap", name):
        return None
    res = {"key": "l1cap/chase", "u": 0}
    ws = [w for w in sc.clock_windows(k) if w]
    if not ws:
        return res | {"ok": False, "why": ["no clock window"]}
    why = []
    per_window = []
    n_timed = 0
    for w in ws:
        ldg = [i for i in w if i.root == "LDG"]
        non_e = [i.text for i in ldg if not i.opcode.startswith("LDG.E")]
        other_mem = sorted({i.opcode for i in w if i.root in MEM_ROOTS
                            and i.root not in ("LDG", "STG")})
        stg = sum(1 for i in w if i.root == "STG")
        per_window.append({"ldg": len(ldg), "stg": stg, "other_mem": other_mem})
        if non_e:
            why.append(f"non-LDG.E chase loads: {non_e[:3]}")
        if other_mem:
            why.append(f"unexpected memory instructions in window: {other_mem}")
        if ldg and stg == 0 and not other_mem:
            n_timed += 1
    if n_timed == 0:
        why.append("no clean timed chase window (LDG only)")
    return res | {"ok": not why, "why": why, "windows": per_window}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True, help="dir with l1cap.sass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = {}
    p = Path(a.sass) / "l1cap.sass"
    if p.exists():
        for name, k in sc.parse(p.read_text()).items():
            res = check(name, k)
            if res is not None:
                kernels[res["key"]] = res
    Path(a.out).write_text(json.dumps({"l1cap": {"sass": str(p), "kernels": kernels}}, indent=1))


if __name__ == "__main__":
    main()
