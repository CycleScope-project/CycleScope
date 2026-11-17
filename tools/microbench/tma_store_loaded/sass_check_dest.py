#!/usr/bin/env python3
"""store_dest (protocol-v5) SASS gate: sass_check_dest.py <sass dump>

For each of the four k_dest<RED, STAGES> instantiations find the round loop =
instructions from the target of the backward branch that closes the round up
to that branch (the loop containing the UTMASTG / UTMAREDG).  Inside the loop:
  - RED 0: exactly one UTMASTG and zero UTMAREDG; RED 1: the reverse
  - exactly one UTMACMDFLUSH
  - exactly one DEPBAR.LE SB0, 0x<STAGES-1>
  - exactly two BAR.SYNC 0x1, 0x80
  - no STS inside the loop
  - at most one clock CS2R inside the loop and it must be predicated
Prints per kernel: OK/BAD, the counts, loop length, and the full loop text.
Exit 0 only if all 4 kernels are found and pass.
"""
import re
import sys

KERNELS = [(r, s) for r in (0, 1) for s in (2, 4)]

INS_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")
LABEL_RE = re.compile(r"^\s*(\.L_\w+|\$\w+):|^\s*(\.L_x_\d+):")
SEC_RE = re.compile(r"\.text\.(\S+)|Function : (\S+)")
KDEST_RE = re.compile(r"k_destILi(\d+)ELi(\d+)E")


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
    """outermost backward branch whose span contains UTMASTG/UTMAREDG"""
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
        if any(a is not None and re.search(r"\bUTMA(STG|REDG)\b", t)
               for _, a, t in span):
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
        m = KDEST_RE.search(name)
        if m:
            found[tuple(int(x) for x in m.groups())] = (name, ins)
    ok_all, n = True, 0
    for key in KERNELS:
        red, stages = key
        if key not in found:
            print(f"BAD k_dest<{red},{stages}>: kernel not found")
            ok_all = False
            continue
        n += 1
        name, ins = found[key]
        loop = find_loop(ins)
        if loop is None:
            print(f"BAD k_dest<{red},{stages}> ({name}): "
                  "no backward-branch loop containing UTMASTG/UTMAREDG")
            ok_all = False
            continue
        body = [(a, t) for _, a, t in ins[loop[0]:loop[1] + 1]
                if a is not None]
        ops = [opcode(t) for _, t in body]
        utmastg = sum(1 for o in ops if o.startswith("UTMASTG"))
        utmaredg = sum(1 for o in ops if o.startswith("UTMAREDG"))
        flush = sum(1 for o in ops if o.startswith("UTMACMDFLUSH"))
        depbar = sum(1 for a, t in body
                     if re.search(r"\bDEPBAR\.LE\b", t) and
                     re.search(r"SB0,\s*0x%x\b" % (stages - 1), t))
        bar = sum(1 for a, t in body
                  if re.search(r"\bBAR\.SYNC\b", t) and
                  re.search(r"0x1,\s*0x80", t))
        sts = sum(1 for o in ops if o.startswith("STS"))
        cs2r_lines = [(a, t) for (a, t), o in zip(body, ops)
                      if o.startswith(("CS2R", "S2R"))]
        cs2r = len(cs2r_lines)
        cs2r_pred = all(re.match(r"\s*@!?U?P", t) for _, t in cs2r_lines)
        want = {"utmastg": 0 if red else 1, "utmaredg": red,
                "flush": 1, "depbar": 1, "bar": 2, "sts": 0}
        have = {"utmastg": utmastg, "utmaredg": utmaredg, "flush": flush,
                "depbar": depbar, "bar": bar, "sts": sts}
        bad = {kk: (have[kk], v) for kk, v in want.items()
               if have[kk] != v}
        if cs2r > 1 or (cs2r == 1 and not cs2r_pred):
            bad["cs2r"] = (cs2r, "<=1 predicated")
        verdict = "OK " if not bad else "BAD"
        print(f"{verdict} k_dest<{red},{stages}> loop_len={len(body)} "
              f"utmastg={utmastg} utmaredg={utmaredg} flush={flush} "
              f"depbar={depbar} bar={bar} sts={sts} "
              f"cs2r={cs2r} cs2r_pred={cs2r_pred}" +
              (f" mismatch(found,want)={bad}" if bad else ""))
        for a, t in body:
            print(f"    /*{a:04x}*/ {t}")
        ok_all = ok_all and not bad
    print(f"kernels={n}/4")
    sys.exit(0 if ok_all and n == 4 else 1)


main()
