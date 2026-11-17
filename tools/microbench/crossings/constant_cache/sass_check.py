#!/usr/bin/env python3
"""SASS gate for the constant_cache probe (crossings protocol-v2).

k_ldc<U>: the window between the two CS2R reads must contain exactly U
LDC instructions whose address operand is register-indexed (c[0x3][R#] or
c[0x3][UR#]) and no other memory-class instruction.
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc

MEM_ROOTS = ("LD", "ST", "LDG", "STG", "LDS", "STS", "ATOM", "ATOMG", "ATOMS",
             "RED", "REDG", "REDS", "LDGSTS", "LDSM", "STSM", "LDL", "STL")
LDC_REG = re.compile(r"c\[0x3\]\[(?:U?R)\d+")


def check_ldc(name, k):
    m = re.match(r"_Z\d+k_ldcILi(\d+)EEv\w+$", name)
    if not m:
        return None
    u = int(m.group(1))
    res = {"key": f"ldc/{u}", "u": u}
    ws = sc.clock_windows(k)
    w = ws[0] if ws else None
    if not w:
        return res | {"ok": False, "why": ["no clock window"]}
    why = []
    lds = [i for i in w if i.root == "LDC"]
    if len(lds) != u:
        why.append(f"{len(lds)} LDC in window, want {u}")
    bad = [i.text for i in lds if not LDC_REG.search(i.body)]
    if bad:
        why.append(f"LDC without a register index: {bad[:3]}")
    stray = sorted({i.opcode for i in w if i.root in MEM_ROOTS})
    if stray:
        why.append(f"other memory instructions in window: {stray}")
    return res | {"ok": not why, "why": why, "ldc_count": len(lds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True, help="dir with ldc_chain.sass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = {}
    p = Path(a.sass) / "ldc_chain.sass"
    if p.exists():
        for name, k in sc.parse(p.read_text()).items():
            res = check_ldc(name, k)
            if res is not None:
                kernels[res["key"]] = res
    Path(a.out).write_text(json.dumps({"ldc_chain": {"sass": str(p), "kernels": kernels}}, indent=1))


if __name__ == "__main__":
    main()
