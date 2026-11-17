#!/usr/bin/env python3
"""SASS gate for the trywait_timeout probe (crossings protocol-v2).

For every k_loop<N,MODE,HINT> kernel: the window between the two CS2R reads
must contain exactly one SYNCS.PHASECHK (.TRANS64.TRYWAIT for try_wait
variants, plain .TRANS64 for the test_wait control), one backward BRA, and
no other memory-class instruction.  iter_stall = control-word stall sum
from the PHASECHK through the backward BRA inclusive (the derivation adds
it back to the test_wait period to get one poll's issue cycles).
k_release<D>: informational -- one TRYWAIT + backward BRA loop.
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc

MEM_ROOTS = ("LD", "ST", "LDG", "STG", "LDS", "STS", "LDC", "ATOM", "ATOMG",
             "ATOMS", "RED", "REDG", "REDS", "LDGSTS", "LDSM", "STSM",
             "UBLKCP", "UTMALDG", "UTMASTG", "LDL", "STL", "TMA")


def stall_sum(insts):
    return sum(i.stall for i in insts)


def bra_target(i):
    last = i.body.split()[-1]
    return int(last, 16) if last.startswith("0x") else None


def check_loop(name, k):
    m = re.match(r"_Z\d+k_loopILi(\d+)ELj(\d+)EEv\w+$", name)
    if not m:
        return None
    mode, hint = (int(g) for g in m.groups())
    key = f"trywait/loop.m{mode}.h{hint}"
    res = {"key": key, "mode": mode, "hint": hint}
    ws = sc.clock_windows(k)
    w = ws[0] if ws else None
    if not w:
        return res | {"ok": False, "why": ["no clock window"]}
    why = []
    tryw = [i for i in w if i.opcode.startswith("SYNCS.PHASECHK.TRANS64.TRYWAIT")]
    plain = [i for i in w
             if i.opcode.startswith("SYNCS.PHASECHK") and not i.opcode.startswith("SYNCS.PHASECHK.TRANS64.TRYWAIT")]
    if mode == 0:
        if len(plain) != 1:
            why.append(f"{len(plain)} test_wait PHASECHK in window, want 1")
        if tryw:
            why.append(f"{len(tryw)} TRYWAIT in the test_wait window")
    else:
        if len(tryw) != 1:
            why.append(f"{len(tryw)} TRYWAIT PHASECHK in window, want 1")
        if plain:
            why.append(f"{len(plain)} non-TRYWAIT PHASECHK in window")
    sync = (plain or tryw)
    back = [i for i in w if i.root == "BRA" and bra_target(i) is not None and bra_target(i) < i.addr]
    if len(back) != 1:
        why.append(f"{len(back)} backward BRA in window, want 1")
    stray = sorted({i.opcode for i in w if i.root in MEM_ROOTS})
    if stray:
        why.append(f"memory instructions in window: {stray}")
    iter_stall = None
    if len(sync) == 1 and len(back) == 1:
        i0, i1 = w.index(sync[0]), w.index(back[0])
        if i0 <= i1:
            iter_stall = stall_sum(w[i0:i1 + 1])
        else:
            why.append("backward BRA before PHASECHK")
    return res | {"ok": not why, "why": why, "iter_stall": iter_stall,
                  "window": [[i.addr, i.text] for i in w]}


def check_release(name, k):
    m = re.match(r"_Z\d+k_releaseILj(\d+)EEv\w+$", name)
    if not m:
        return None
    res = {"key": f"trywait/release.d{m.group(1)}", "delay": int(m.group(1))}
    tryw = [i for i in k.insts if i.opcode.startswith("SYNCS.PHASECHK.TRANS64.TRYWAIT")]
    back = [i for i in k.insts if i.root == "BRA" and bra_target(i) is not None and bra_target(i) < i.addr]
    why = []
    if not tryw:
        why.append("no TRYWAIT loop in the waiter")
    if not back:
        why.append("no backward BRA")
    return res | {"ok": not why, "why": why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True, help="dir with trywait.sass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = {}
    p = Path(a.sass) / "trywait.sass"
    if p.exists():
        for name, k in sc.parse(p.read_text()).items():
            for fn in (check_loop, check_release):
                res = fn(name, k)
                if res is not None:
                    kernels[res["key"]] = res
                    break
    Path(a.out).write_text(json.dumps({"trywait": {"sass": str(p), "kernels": kernels}}, indent=1))


if __name__ == "__main__":
    main()
