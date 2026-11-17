#!/usr/bin/env python3
"""SASS gate for math_occupancy (v1 + v2): in every kernel, between the
rep's two consecutive SR_CLOCKLO reads the declared opcodes occur the
declared number of times -- single-class streams: 256 of one opcode;
shfl_fadd: 128 SHFL + 128 FADD; mixed streams: 128 + 128 of the two
opcodes strictly alternating -- with no memory and no branch
instructions in the window.  For the chain kernels the destination
register must equal one source register.  Nonzero exit on any
mismatch; a per-kernel mnemonic summary is printed (caller tees it to
$OUT/build/sass-counts.txt).

Usage: check.py <sass file>
"""
import re
import sys

# stream id -> {opcode: count}; opcode is the full mnemonic
OPS = {0: {"IMAD": 256}, 1: {"FADD": 256}, 2: {"FMNMX": 256},
       3: {"SHFL": 128, "FADD": 128}, 4: {"LOP3": 256},
       5: {"IMAD": 128, "LOP3": 128}, 6: {"IMAD": 128, "FADD": 128},
       7: {"LOP3": 128, "FADD": 128},
       8: {"IMAD": 128, "IMAD.HI": 128},
       9: {"FADD": 128, "FMUL": 128},
       10: {"FADD": 256}, 11: {"IMAD.HI": 256},
       12: {"IMAD": 256}, 13: {"LOP3": 256},
       14: {"IMAD.HI": 256}, 15: {"FMUL": 256}}
ALT = {5: ("IMAD", "LOP3"), 6: ("IMAD", "FADD"), 7: ("LOP3", "FADD"),
       8: ("IMAD", "IMAD.HI"), 9: ("FADD", "FMUL")}
CHAIN = {1, 2}  # dest must equal a source
# serial-chain streams (v2): each counted op's destination register
# must appear among the next counted op's source registers
SCHAIN = {5, 6, 7, 8, 9, 12, 13, 14, 15}
MEM_BR_BASE = {"LDG", "STG", "LDS", "STS", "LDL", "STL", "ATOM", "RED",
               "UTMALDG", "UTMASTG", "LD", "ST", "BRA", "BSSY", "BSYNC",
               "BRX", "JMP", "CALL", "RET", "EXIT", "WARPSYNC", "B2R",
               "BREAK", "SSY", "BAR", "UBLKCP", "ELECT"}


def opcode(line):
    seg = line.split("*/", 1)[1] if "*/" in line else ""
    m = re.match(r"\s*(?:@?!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9.]*)", seg)
    return m.group(1) if m else ""


def window_check(fn_lines, s, name):
    idx = [i for i, l in enumerate(fn_lines) if "SR_CLOCKLO" in l]
    expect = OPS[s]
    bad = []
    nclk = 0
    for a, b in zip(idx, idx[1:]):
        seg = fn_lines[a + 1:b]
        ops = [opcode(l) for l in seg]
        ops = [o for o in ops if o]

        def isop(o, key):
            # "IMAD" matches IMAD / IMAD.WIDE / IMAD.MOV base classes;
            # "IMAD.HI" matches only that variant
            return o.split(".")[0] == key if "." not in key \
                else o == key or o.startswith(key + ".")
        cnt = {}
        keys = sorted(expect, key=len, reverse=True)  # IMAD.HI before IMAD
        for o in ops:
            for k in keys:
                if isop(o, k):
                    cnt[k] = cnt.get(k, 0) + 1
                    break
            else:
                cnt[o] = cnt.get(o, 0) + 1
        bad_mem = [o for o in ops if o.split(".")[0] in MEM_BR_BASE]
        nclk += 1
        match = all(cnt.get(k, 0) == v for k, v in expect.items())
        if not match or bad_mem:
            bad.append((nclk, cnt, bad_mem[:4]))
            continue
        if s in ALT:
            a_op, b_op = ALT[s]
            # classify each op by longest matching key first (same
            # rule as the count loop) so IMAD.HI is not seen as IMAD
            seq = []
            for o in ops:
                for k in keys:
                    if isop(o, k):
                        seq.append(k)
                        break
            seq = [k for k in seq if k in (a_op, b_op)]
            ok = (len(seq) == 256 and
                  all(k == (a_op if i % 2 == 0 else b_op)
                      for i, k in enumerate(seq)))
            if not ok:
                bad.append((nclk, cnt, ["not strictly alternating"]))
        if s in CHAIN:
            ok = True
            for l in seg:
                if re.search(r"\b(FADD|FMNMX)\b", l):
                    m = re.search(r"\bR(\d+),\s*R(\d+)", l)
                    if not m or m.group(1) != m.group(2):
                        ok = False
            if not ok:
                bad.append((nclk, cnt, ["chain dest != src"]))
        if s in SCHAIN:
            seq = []
            for l in seg:
                o = opcode(l)
                if not o:
                    continue
                if any(isop(o, k) for k in keys):
                    seg_txt = l.split("*/", 1)[1]
                    # registers after the mnemonic; first is the dest
                    regs = re.findall(r"\bR(\d+)\b", seg_txt)
                    if regs:
                        seq.append(regs)
            ok = len(seq) >= 2 and all(
                seq[i][0] in seq[i + 1][1:] for i in range(len(seq) - 1))
            if not ok:
                bad.append((nclk, cnt,
                            ["chain dest not a source of next op"]))
    return nclk, bad


def main():
    sass = open(sys.argv[1]).read()
    fns = re.split(r"Function : ", sass)
    nfn = 0
    fail = False
    for f in fns:
        m = re.match(r"(_Z\d*k_math_occILi(\d+)E[^\s]*)", f)
        if not m:
            continue
        nfn += 1
        name, s = m.group(1), int(m.group(2))
        lines = f.splitlines()
        nclk, bad = window_check(lines, s, name)
        print(f"{name}: stream={s} clock_windows={nclk} ok={not bad}")
        for w, cnt, mem in bad:
            print(f"  window {w}: counts={cnt} mem/br={mem}")
        if nclk == 0 or bad:
            fail = True
    if nfn == 0:
        print("no k_math_occ kernels found")
        fail = True
    sys.exit(1 if fail else 0)


main()
