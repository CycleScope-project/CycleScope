#!/usr/bin/env python3
"""SASS gate for the l1_bandwidth probes (mem.cu / memq.cu, verbatim
blackwell-prototype copies).  Trimmed port of blackwell-prototype sass_check.py's
check_mem / check_memq covering the kernel forms the crossings cells
instantiate: k_stream, k_store (wave-3 default cells + --cells smport),
k_ld, k_st, k_ack.  Keys match blackwell-prototype so the derivation can reuse
them: mem/<lvl>/<vec>/<K>[/d<D>] , mem/s<lvl>/<vec>/<K>[/d<D>] ,
memq/ld<lvl>/d<D>/<K>, memq/st<SAME>/<K>, memq/ack.  Kernels of cells we do
not run (k_chain / k_cpchain / k_cpstage) are ignored.
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc


def check_mem(name, k):
    m = re.match(r"_Z8k_streamILi(\d)ELi(\d+)ELi(\d+)E(?:Li(\d+)E)?(?:Li(\d+)E)?Ev", name)
    ms = re.match(r"_Z7k_storeILi(\d)ELi(\d+)ELi(\d+)E(?:Li(\d+)E)?Ev", name)
    if not (m or ms):
        return None
    mm = m or ms
    lvl, vec, kk = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
    d = mm.group(4)
    depth_key = "" if d in (None, "8") else f"/d{d}"
    res = {"key": f"mem/{'s' if ms else ''}{lvl}/{vec}/{kk}{depth_key}", "u": kk}
    ws = sc.clock_windows(k)
    w = max(ws, key=len) if ws else None
    if not w:
        return res | {"ok": False, "why": ["no clock window"]}
    root = "STG" if ms else "LDG"
    ops = [i for i in w if i.root == root]
    why = []
    if len(ops) != kk:
        why.append(f"{len(ops)} {root} in window, want {kk}")
    if any((".128" in i.opcode) != (vec == 16) for i in ops):
        why.append(f"unexpected {root} width")
    bars = [x for x, i in enumerate(w) if i.root == "BAR"]
    pos = [x for x, i in enumerate(w) if i.root == root]
    if len(bars) != 1 or (pos and max(pos) > bars[0]):
        why.append(f"{root} after the closing bar.sync")
    stray = sorted({i.opcode for i in w if i.root in ("LDL", "STL")})
    if stray:
        why.append(f"local memory in window: {stray}")
    return res | {"ok": not why, "why": why, "opcodes": sorted({i.opcode for i in ops})}


def check_memq(name, k):
    m = re.match(r"_Z4k_ldILi(\d)ELi(\d+)ELi(\d+)EEv", name)
    ms = re.match(r"_Z4k_stILi(\d)ELi(\d+)EEv", name)
    if name.startswith("_Z5k_ack"):
        res = {"key": "memq/ack", "u": 0}
        ws = [w for w in sc.clock_windows(k) if any(i.root == "MEMBAR" for i in w)]
        why = []
        if not ws or len(ws) % 2:
            why.append(f"{len(ws)} fence windows, want pairs (cold, warm)")
        for w in ws:
            roots = [i.root for i in w]
            if roots.count("STG") != 1 or roots.count("MEMBAR") != 1:
                why.append(f"window roots {roots}")
        return res | {"ok": not why, "why": why}
    if ms:
        same, kk = int(ms.group(1)), int(ms.group(2))
        res = {"key": f"memq/st{same}/{kk}", "u": kk}
        ws = sc.clock_windows(k)
        w = max(ws, key=len) if ws else None
        if not w:
            return res | {"ok": False, "why": ["no clock window"]}
        st = [i for i in w if i.root == "STG"]
        why = [] if len(st) == kk else [f"{len(st)} STG in window, want {kk}"]
        if any(".128" in i.opcode or ".64" in i.opcode for i in st):
            why.append("unexpected STG width")
        return res | {"ok": not why, "why": why, "opcodes": sorted({i.opcode for i in st})}
    if not m:
        return None
    lvl, d, kk = int(m.group(1)), int(m.group(2)), int(m.group(3))
    res = {"key": f"memq/ld{lvl}/d{d}/{kk}", "u": kk}
    ws = sc.clock_windows(k)
    w = max(ws, key=len) if ws else None
    if not w:
        return res | {"ok": False, "why": ["no clock window"]}
    lds = [i for i in w if i.root == "LDG"]
    why = [] if len(lds) == kk else [f"{len(lds)} LDG in window, want {kk}"]
    if any(".128" in i.opcode or ".64" in i.opcode for i in lds):
        why.append("unexpected LDG width")
    bars = [x for x, i in enumerate(w) if i.root == "BAR"]
    pos = [x for x, i in enumerate(w) if i.root == "LDG"]
    if len(bars) != 1 or (pos and max(pos) > bars[0]):
        why.append("loads after the closing bar.sync")
    return res | {"ok": not why, "why": why, "opcodes": sorted({i.opcode for i in lds})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True, help="dir with mem.sass / memq.sass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    checks = {"mem": check_mem, "memq": check_memq}
    out = {}
    for probe, fn in checks.items():
        kernels = {}
        p = Path(a.sass) / f"{probe}.sass"
        if p.exists():
            for name, k in sc.parse(p.read_text()).items():
                res = fn(name, k)
                if res is not None:
                    kernels[res["key"]] = res
        out[probe] = {"sass": str(p), "kernels": kernels}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
