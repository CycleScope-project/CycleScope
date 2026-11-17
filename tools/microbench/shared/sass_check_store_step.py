#!/usr/bin/env python3
"""store_step (protocol-v3) SASS gate: sass_check_store_step.py <sass dump>

For each of the 38 k_step<WIDE, K, TAIL, NFILL> instantiations find the round
loop = instructions from the target of the backward branch that closes the
round up to that branch (the loop containing the STS).  Inside the loop:
  - exactly K STS (STS.128 for WIDE 1, STS.64 for WIDE 2, a 32-bit STS for
    WIDE 0) and no STS of any other width
  - BAR.SYNC 0x1, 0x80 count 0 / 1 / 2 for TAIL 0,3 / 1 / 2,4,5,6
  - exactly NFILL IMAD for TAIL 3/4/5, zero IMAD-family otherwise
    (register moves spelled IMAD.MOV* are allowed, counted separately)
  - TAIL 6: one MEMBAR and one FENCE.VIEW.ASYNC; zero of both otherwise
  - no CS2R / S2R, no LD* inside the loop
Prints per kernel: OK/BAD, the counts, loop length, whether the stores are
predicated (@P STS) or branched over, and the full loop text.  Exit 0 only if
all 38 kernels are found and pass.
"""
import re
import sys

KERNELS = [
    (1, k, t, n)
    for t, ks, ns in [(0, (1, 2, 4, 8), (0,)), (1, (1, 2, 4, 8), (0,)),
                      (2, (1, 2, 4, 8), (0,)), (6, (1, 2, 4, 8), (0,)),
                      (3, (1, 2, 4, 8), (8, 24, 64))]
    for n in ns for k in ks
] + [(1, 4, 4, 24), (1, 4, 4, 64), (1, 4, 5, 24), (1, 4, 5, 64),
     (0, 4, 0, 0), (0, 8, 0, 0), (0, 4, 2, 0), (0, 8, 2, 0),
     (2, 4, 0, 0), (2, 8, 0, 0)]
# = 4*4 + 4*4 + 4 + 4 + 2 = 38

BAR_BY_TAIL = {0: 0, 3: 0, 1: 1, 2: 2, 4: 2, 5: 2, 6: 2}

INS_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")
LABEL_RE = re.compile(r"^\s*(\.L_\w+|\$\w+):|^\s*(\.L_x_\d+):")
SEC_RE = re.compile(r"\.text\.(\S+)|Function : (\S+)")
KSTEP_RE = re.compile(r"k_stepILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E")


def functions(text):
    """yield (mangled_name, [ (lineno, addr, text), ... ]) per function block"""
    cur, name = None, None
    for ln, line in enumerate(text.splitlines()):
        m = SEC_RE.search(line)
        if m:
            if name is not None:
                yield name, cur
            name = m.group(1) or m.group(2)
            cur = []
            continue
        if cur is not None:
            i = INS_RE.search(line)
            if i:
                cur.append((ln, int(i.group(1), 16), i.group(2)))
            else:
                lm = LABEL_RE.match(line)
                if lm:
                    cur.append((ln, None, lm.group(1) or lm.group(2)))
    if name is not None:
        yield name, cur


def find_loop(ins):
    """outermost backward branch whose span contains an STS -> (start, end) indices"""
    pos_of_label = {}
    for i, (ln, addr, txt) in enumerate(ins):
        if addr is None:
            pos_of_label[txt] = i
    best = None
    for i, (ln, addr, txt) in enumerate(ins):
        if addr is None or "BRA" not in txt:
            continue
        m = re.search(r"`\((\.\w+)\)", txt)
        if m:
            tgt = pos_of_label.get(m.group(1))
        else:
            h = re.search(r"\bBRA\b.*?0x([0-9a-f]+)", txt)
            tgt = None
            if h:
                taddr = int(h.group(1), 16)
                for j, (ln2, a2, t2) in enumerate(ins):
                    if a2 == taddr:
                        tgt = j
                        break
        if tgt is None or tgt >= i:
            continue
        span = ins[tgt:i + 1]
        if any(a is not None and re.search(r"\bSTS\b", t) for _, a, t in span):
            if best is None or tgt < best[0]:
                best = (tgt, i)
    return best


def opcode(txt):
    m = re.match(r"\s*(@!?U?P\d+\s+)?([A-Z0-9@][A-Za-z0-9.]*)", txt)
    return m.group(2) if m else ""


def main():
    text = open(sys.argv[1], errors="replace").read()
    found = {}
    for name, ins in functions(text):
        m = KSTEP_RE.search(name)
        if m:
            found[tuple(int(x) for x in m.groups())] = (name, ins)
    ok_all, n = True, 0
    for key in KERNELS:
        if key not in found:
            print(f"BAD k_step<{key[0]},{key[1]},{key[2]},{key[3]}>: kernel not found")
            ok_all = False
            continue
        n += 1
        wide, k, tail, nfill = key
        name, ins = found[key]
        loop = find_loop(ins)
        if loop is None:
            print(f"BAD k_step<{wide},{k},{tail},{nfill}> ({name}): no backward-branch loop containing STS")
            ok_all = False
            continue
        body = [(a, t) for _, a, t in ins[loop[0]:loop[1] + 1] if a is not None]
        ops = [opcode(t) for _, t in body]
        sts_lines = [(a, t) for (a, t), o in zip(body, ops) if o.startswith("STS")]
        if wide == 1:
            want_sts = "STS.128"
        elif wide == 2:
            want_sts = "STS.64"
        else:
            want_sts = None
        if want_sts:
            sts = [t for _, t in sts_lines if want_sts in t.split("[")[0]]
            sts_other = len(sts_lines) - len(sts)
        else:
            sts = [t for _, t in sts_lines
                   if not re.search(r"\bSTS\.(128|64)", t)]
            sts_other = len(sts_lines) - len(sts)
        bar = sum(1 for o, (a, t) in zip(ops, body)
                  if re.search(r"\bBAR\.SYNC\b", t) and re.search(r"0x1,\s*0x80", t))
        imad = [o for o in ops if o.startswith("IMAD")]
        imad_mov = [o for o in imad if o.startswith("IMAD.MOV")]
        imad_arith = len(imad) - len(imad_mov)
        membar = sum(1 for o in ops if o.startswith("MEMBAR"))
        fence = sum(1 for o in ops if o.startswith("FENCE.VIEW.ASYNC"))
        cs2r = sum(1 for o in ops if o.startswith(("CS2R", "S2R")))
        loads = sum(1 for o in ops if o.startswith("LD"))
        pred = all(re.match(r"\s*@!?U?P", t) for _, t in sts_lines) if sts_lines else False
        want = {
            "sts": k,
            "bar": BAR_BY_TAIL[tail],
            "imad": nfill if tail in (3, 4, 5) else 0,
            "membar": 1 if tail == 6 else 0,
            "fence": 1 if tail == 6 else 0,
            "cs2r_s2r": 0,
            "loads": 0,
            "sts_other": 0,
        }
        have = {"sts": len(sts), "bar": bar, "imad": imad_arith, "membar": membar,
                "fence": fence, "cs2r_s2r": cs2r, "loads": loads,
                "sts_other": sts_other}
        bad = {kk: (have[kk], v) for kk, v in want.items() if have[kk] != v}
        style = "predicated" if pred else "branched-over" if sts_lines else "none"
        verdict = "OK " if not bad else "BAD"
        print(f"{verdict} k_step<{wide},{k},{tail},{nfill}> loop_len={len(body)} stores={style} "
              f"sts={have['sts']} bar={bar} imad={imad_arith} imad_mov={len(imad_mov)} "
              f"membar={membar} fence={fence} cs2r_s2r={cs2r} loads={loads}" +
              (f" mismatch(found,want)={bad}" if bad else ""))
        for a, t in body:
            print(f"    /*{a:04x}*/ {t}")
        ok_all = ok_all and not bad
    print(f"kernels={n}/38")
    sys.exit(0 if ok_all and n == 38 else 1)


main()
