#!/usr/bin/env python3
"""SASS gate for the memory_request probe (protocol.json sass_gate).

  k_ld<LVL,D,K>: the timed window between the two SR_CLOCKLO reads has
      exactly K LDG (STRONG.SM for l1 / STRONG.GPU for l2 and dram, one
      per request) and no other global memory instruction.
  k_st<SAME,K>: the timed window has exactly K STG.E.
  k_ack: one STG.E + MEMBAR.SC.GPU per pass.

A gate failure voids the variant.  Writes checks.json with pass/fail
and the opcodes hit per kernel.
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc

GLOBAL_RE = re.compile(
    r"^(LDG|STG|ATOM|ATOMG|RED|REDG|LDGSTS|UTMALDG|UTMASTG|UBLKCP|LDCU)")


def root(o):
    return o.split(".")[0]


def ld_kernel(kernels, lvl, d, kk):
    pat = re.compile(r"k_ldILi%dELi%dELi%dE" % (lvl, d, kk))
    for n, k in kernels.items():
        if pat.search(n):
            return k
    return None


def st_kernel(kernels, same, kk):
    pat = re.compile(r"k_stILi%dELi%dE" % (same, kk))
    for n, k in kernels.items():
        if pat.search(n):
            return k
    return None


def windows(k):
    ws = sc.clock_windows(k)
    return [w for w in ws if w]


def check_ld(k, lvl, kk):
    why, hit = [], set()
    ws = windows(k)
    if not ws:
        return ["no clock window"], []
    want = "SM" if lvl == 0 else "GPU"
    for wi, w in enumerate(ws):
        ldg = [i for i in w if root(i.opcode) == "LDG"]
        other = sorted({i.opcode for i in w
                        if GLOBAL_RE.match(i.opcode)
                        and root(i.opcode) != "LDG"})
        if len(ldg) != kk:
            why.append(f"window {wi}: {len(ldg)} LDG, want {kk}")
        bad = sorted({i.opcode for i in ldg
                      if "STRONG" not in i.opcode or want not in i.opcode})
        if bad:
            why.append(f"window {wi}: LDG not STRONG.{want}: {bad}")
        if other:
            why.append(f"window {wi}: other global ops {other}")
        hit.update(i.opcode for i in ldg)
    return why, sorted(hit)


def check_st(k, kk):
    why, hit = [], set()
    ws = windows(k)
    if not ws:
        return ["no clock window"], []
    for wi, w in enumerate(ws):
        stg = [i for i in w if root(i.opcode) == "STG"]
        other = sorted({i.opcode for i in w
                        if GLOBAL_RE.match(i.opcode)
                        and root(i.opcode) != "STG"})
        if len(stg) != kk:
            why.append(f"window {wi}: {len(stg)} STG, want {kk}")
        if other:
            why.append(f"window {wi}: other global ops {other}")
        hit.update(i.opcode for i in stg)
    return why, sorted(hit)


def loop_body(k, anchor):
    """Backward-branch loop body containing insts[anchor] (Rec/out
    bookkeeping stores may sit outside it); falls back to all insts."""
    insts = k.insts
    for j in range(anchor, len(insts)):
        ins = insts[j]
        if ins.opcode.split(".")[0] != "BRA":
            continue
        m = re.search(r"0x([0-9a-f]+)", ins.text)
        if m and int(m.group(1), 16) < ins.addr:
            tgt = int(m.group(1), 16)
            lo = next(i for i, x in enumerate(insts) if x.addr >= tgt)
            return insts[lo:j + 1]
    return insts


def check_ack(k):
    ai = next((i for i, x in enumerate(k.insts)
               if x.opcode.split(".")[0] == "MEMBAR"), 0)
    ops = [i.opcode for i in loop_body(k, ai)]
    stg = [o for o in ops if root(o) == "STG"]
    membar = [o for o in ops if root(o) == "MEMBAR"]
    why = []
    if len(stg) != 2:
        why.append(f"{len(stg)} STG, want 2 (one per pass)")
    gpu_sc = [o for o in membar if "SC" in o and "GPU" in o]
    if len(gpu_sc) != 2:
        why.append(f"{len(gpu_sc)} MEMBAR.SC.GPU, want 2; "
                   f"membar ops: {sorted(set(membar))}")
    return why, sorted(set(ops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    checks = {}
    for lvl, lname in ((0, "l1"), (1, "l2"), (2, "dram")):
        ks = [64, 128, 256] + ([512, 1024] if lvl == 1 else [])
        for d in (1, 2, 4, 8, 16, 32):
            for kk in ks:
                key = f"ld.{lname}.d{d}.k{kk}"
                k = ld_kernel(kernels, lvl, d, kk)
                if k is None:
                    checks[key] = {"pass": False,
                                   "why": ["kernel not found"],
                                   "opcodes": []}
                    continue
                why, ops = check_ld(k, lvl, kk)
                checks[key] = {"pass": not why, "why": why,
                               "opcodes": ops}
    for same in (0, 1):
        for kk in (16, 32, 64):
            key = f"st.{'same' if same else 'dist'}.k{kk}"
            k = st_kernel(kernels, same, kk)
            if k is None:
                checks[key] = {"pass": False, "why": ["kernel not found"],
                               "opcodes": []}
                continue
            why, ops = check_st(k, kk)
            checks[key] = {"pass": not why, "why": why, "opcodes": ops}
    ack = next((k for n, k in kernels.items() if "k_ack" in n), None)
    if ack is None:
        checks["ack"] = {"pass": False, "why": ["kernel not found"],
                         "opcodes": []}
    else:
        why, ops = check_ack(ack)
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
