#!/usr/bin/env python3
"""SASS gate for the atomic_issue probe.

k_atomic<OP,U,IND>: the CS2R window must contain exactly U atomic
instructions of the expected opcode family and no other memory-class
instruction (IND = 1: r2 independent-line variant, key suffix /indep).
Expected families (the compiled form may differ by arch; the gate accepts
the documented spellings):
  OP 0 red.global   RED.E.ADD / REDG.E.ADD
  OP 1 atom.global  ATOMG.E.ADD
  OP 2 atom.shared  ATOMS.ADD
  OP 3 red.shared   REDS / ATOMS as compiled
  OP 4 atom generic ATOM.E.ADD
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc

MEM_ROOTS = ("LD", "ST", "LDG", "STG", "LDS", "STS", "LDC", "ATOM", "ATOMG",
             "ATOMS", "RED", "REDG", "REDS", "LDGSTS", "LDSM", "STSM",
             "LDL", "STL")
ATOMIC_ROOTS = ("ATOM", "ATOMG", "ATOMS", "RED", "REDG", "REDS")
EXPECTED = {
    0: ("RED", "REDG"),
    1: ("ATOMG",),
    2: ("ATOMS",),
    3: ("REDS", "ATOMS"),
    4: ("ATOM",),
}


def check(name, k):
    m = re.match(r"_Z\d+k_atomicILi(\d)ELi(\d+)ELi(\d)EEv\w+$", name)
    if not m:
        return None
    op, u, ind = int(m.group(1)), int(m.group(2)), int(m.group(3))
    res = {"key": f"atomic/{op}/{u}" + ("/indep" if ind else ""), "u": u}
    ws = sc.clock_windows(k)
    w = max(ws, key=len) if ws else None
    if not w:
        return res | {"ok": False, "why": ["no clock window"]}
    why = []
    ops = [i for i in w if i.root in ATOMIC_ROOTS]
    if len(ops) != u:
        why.append(f"{len(ops)} atomic instructions in window, want {u}")
    wrong = [i.opcode for i in ops if i.root not in EXPECTED[op] or "ADD" not in i.opcode]
    if wrong:
        why.append(f"unexpected atomic opcodes: {sorted(set(wrong))[:4]}")
    stray = sorted({i.opcode for i in w if i.root in MEM_ROOTS and i.root not in ATOMIC_ROOTS})
    if stray:
        why.append(f"other memory instructions in window: {stray}")
    return res | {"ok": not why, "why": why, "opcodes": sorted({i.opcode for i in ops})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True, help="dir with atomic.sass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = {}
    p = Path(a.sass) / "atomic.sass"
    if p.exists():
        for name, k in sc.parse(p.read_text()).items():
            res = check(name, k)
            if res is not None:
                kernels[res["key"]] = res
    Path(a.out).write_text(json.dumps({"atomic": {"sass": str(p), "kernels": kernels}}, indent=1))


if __name__ == "__main__":
    main()
