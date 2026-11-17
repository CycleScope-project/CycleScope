#!/usr/bin/env python3
"""SASS gate for the descriptor_refetch probe (crossings protocol-v2).

k_tmap: every CS2R window that contains a UTMALDG must contain exactly one
UTMALDG.2D (t_s -> t_e issue window of a cell).  o_s = the control-word
stall sum from the window start to the UTMALDG.  The tensormap acquire
fence is issued before t_s and must not appear inside a timed window.
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc


def check(name, k):
    if not name.startswith("_Z6k_tmap"):
        return None
    res = {"key": "tmap/cells", "u": 1}
    ws = [w for w in sc.clock_windows(k) if any(i.root == "UTMALDG" for i in w)]
    if not ws:
        return res | {"ok": False, "why": ["no UTMALDG clock window"]}
    why = []
    o_s = []
    for w in ws:
        ops = [i for i in w if i.root == "UTMALDG"]
        if len(ops) != 1:
            why.append(f"{len(ops)} UTMALDG in a window, want 1")
            continue
        if not ops[0].opcode.startswith("UTMALDG.2D"):
            why.append(f"unexpected form {ops[0].opcode}")
        o_s.append(sum(i.stall for i in w[:w.index(ops[0])]))
    return res | {"ok": not why, "why": why, "o_s": o_s, "n_windows": len(ws)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True, help="dir with tmap.sass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = {}
    p = Path(a.sass) / "tmap.sass"
    if p.exists():
        for name, k in sc.parse(p.read_text()).items():
            res = check(name, k)
            if res is not None:
                kernels[res["key"]] = res
    Path(a.out).write_text(json.dumps({"tmap": {"sass": str(p), "kernels": kernels}}, indent=1))


if __name__ == "__main__":
    main()
