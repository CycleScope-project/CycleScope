#!/usr/bin/env python3
"""SASS gate for the memory_request v2 probe (protocol-v2 sass_gate).

  k_ld<LVL,D,K>: the timed window has exactly K x D LDG
      (LDG.E.STRONG.SM for l1 / .STRONG.GPU for l2 and dram) and no
      other global memory instruction; every LDG's address register
      must be produced from an earlier LDG's destination (the
      dependent chase) except the first D chain seeds.
  k_st / k_ack: as v1.
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc
import sass_check as v1

REG = re.compile(r"(?<![A-Z.])(R\d+|UR\d+)")


def ld_kernel(kernels, lvl, d, kk):
    pat = re.compile(r"k_ldILi%dELi%dELi%dE" % (lvl, d, kk))
    for n, k in kernels.items():
        if pat.search(n):
            return k
    return None


def windows(k):
    return [w for w in sc.clock_windows(k) if w]


def check_ld_v2(k, lvl, kk, d):
    why, hit = [], set()
    ws = windows(k)
    if not ws:
        return ["no clock window"], []
    want = "SM" if lvl == 0 else "GPU"
    want_ldg = kk * d
    for wi, w in enumerate(ws):
        glob = [i for i in w if v1.GLOBAL_RE.match(i.opcode)]
        ldg = [i for i in glob if i.opcode.split(".")[0] == "LDG"]
        other = sorted({i.opcode for i in glob
                        if i.opcode.split(".")[0] != "LDG"})
        if len(ldg) != want_ldg:
            # ptxas may partially roll the K loop: a backward branch
            # inside the window iterating a K*D / m body also yields
            # K*D executed loads -- accept m dividing K*D
            back = [i for i in w if i.opcode.split(".")[0] == "BRA"
                    and re.search(r"0x([0-9a-f]+)", i.text)
                    and int(re.search(r"0x([0-9a-f]+)", i.text)
                            .group(1), 16) < i.addr]
            if not (back and ldg and want_ldg % len(ldg) == 0):
                why.append(f"window {wi}: {len(ldg)} LDG, "
                           f"want {want_ldg}")
        bad = sorted({i.opcode for i in ldg
                      if "STRONG" not in i.opcode or want not in i.opcode})
        if bad:
            why.append(f"window {wi}: LDG not STRONG.{want}: {bad}")
        if other:
            why.append(f"window {wi}: other global ops {other}")
        hit.update(i.opcode for i in ldg)
        # dependency: register provenance; each LDG's address regs must
        # trace to an earlier LDG destination except the first D seeds
        prov = {}
        seeds = d
        for ins in w:
            toks = REG.findall(ins.text)
            if ins.opcode.split(".")[0] == "LDG":
                # every bracket of the address expression
                # (desc[URx][Ry.64+off] -> URx and Ry)
                aregs = [r for grp in re.findall(r"\[([^\]]*)\]",
                                                 ins.text)
                         for r in REG.findall(grp)]
                dep = any(prov.get(a, set()) & {"LDG"} for a in aregs)
                if not dep:
                    if seeds > 0:
                        seeds -= 1
                    else:
                        why.append(f"window {wi}: {ins.opcode} address "
                                   "not chain-dependent")
                if toks:
                    prov[toks[0]] = {"LDG"}
            elif toks:
                dst = toks[0]
                p = set()
                for t in toks[1:]:
                    p |= prov.get(t, set())
                prov[dst] = p
    return why, sorted(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    checks = {}
    for lvl, lname in ((0, "l1"), (1, "l2"), (2, "dram")):
        for d in (1, 2, 4, 8, 16, 32):
            for kk in (64, 128, 256):
                key = f"ld.{lname}.d{d}.k{kk}"
                k = ld_kernel(kernels, lvl, d, kk)
                if k is None:
                    checks[key] = {"pass": False,
                                   "why": ["kernel not found"],
                                   "opcodes": []}
                    continue
                why, ops = check_ld_v2(k, lvl, kk, d)
                checks[key] = {"pass": not why, "why": why,
                               "opcodes": ops}
    for same in (0, 1):
        for kk in (16, 32, 64):
            key = f"st.{'same' if same else 'dist'}.k{kk}"
            k = v1.st_kernel(kernels, same, kk)
            if k is None:
                checks[key] = {"pass": False, "why": ["kernel not found"],
                               "opcodes": []}
                continue
            why, ops = v1.check_st(k, kk)
            checks[key] = {"pass": not why, "why": why, "opcodes": ops}
    ack = next((k for n, k in kernels.items() if "k_ack" in n), None)
    if ack is None:
        checks["ack"] = {"pass": False, "why": ["kernel not found"],
                         "opcodes": []}
    else:
        why, ops = v1.check_ack(ack)
        checks["ack"] = {"pass": not why, "why": why, "opcodes": ops}
    out = {"checks": checks,
           "pass": all(c["pass"] for c in checks.values())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    fails = {k: v["why"] for k, v in checks.items() if not v["pass"]}
    print(f"pass={out['pass']} kernels={len(checks)} fails={len(fails)}")
    for k, w in fails.items():
        print(f"  {k}: {w[:2]}")


if __name__ == "__main__":
    main()
