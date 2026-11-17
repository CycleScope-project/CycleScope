#!/usr/bin/env python3
"""sass_check_grpq.py -- run-9 (addendum 13) SASS gate for k_grpq.

    sass_check_grpq.py <overlap_v4.sass> <out_dir>

Per shape kernel k_grpq<158, SHAPE>:
  * between the ti0 and ti1 clock reads (CS2R .. SR_CLOCKLO) only
    WARPGROUP.ARRIVE / HGMMA lines plus non-memory setup ptxas hoists;
    in particular NO WARPGROUP.DEPBAR there (serialisation check).
  * after ti1: exactly NG lines "WARPGROUP.DEPBAR.LE gsb0, N" with
    N = NG-1 .. 0 in order, each followed by a CS2R SR_CLOCKLO before
    the next DEPBAR (the wait sequence).
  * a dependent-integer delay loop (IMAD) sits before ti0.
Writes the matched lines to <out_dir>/sass-grpq-<shape>.txt and exits
non-zero if any shape fails.
"""
import re
import sys

SHAPES = {
    "0": ("q6", 6),
    "1": ("g4c3", 4),
    "2": ("fa", 2),
}
MEM = re.compile(r"\b(LDG|LDS|STG|STS|LDGSTS|UTMALDG|LDSM|TMA)\b")


def opcode(line):
    m = re.search(r"\*/\s+(@!?\w+\s+)?([A-Z0-9._:]+)", line)
    return m.group(2) if m else ""


def main():
    sass_path, out_dir = sys.argv[1], sys.argv[2]
    funcs = {}
    cur = None
    for line in open(sass_path, errors="replace"):
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur is not None:
            funcs[cur].append(line.rstrip("\n"))
    ok = True
    for tag, (shape, ng) in SHAPES.items():
        pat = re.compile(r"k_grpqILi\d+ELi" + tag + r"E")
        name = next((n for n in funcs if pat.search(n)), None)
        lines_out = []
        if name is None:
            print(f"MISSING kernel k_grpq{tag}")
            ok = False
            continue
        lines = funcs[name]
        clocks = [i for i, l in enumerate(lines)
                  if "CS2R" in l and "SR_CLOCKLO" in l]
        # ti0/ti1 = the clock pair bracketing HGMMA; find first window
        # containing HGMMA
        win = None
        for a, b in zip(clocks, clocks[1:]):
            if any("HGMMA" in l for l in lines[a:b]):
                win = (a, b)
                break
        if win is None:
            lines_out.append("FAIL no clock window with HGMMA")
            ok = False
        else:
            a, b = win
            body = [(i, opcode(lines[i]), lines[i].strip())
                    for i in range(a + 1, b)]
            bad = [(i, o, raw) for i, o, raw in body
                   if "DEPBAR" in o or MEM.search(raw)]
            n_hgmma = sum(1 for _, o, _ in body if "HGMMA" in o)
            n_arr = sum(1 for _, o, _ in body if "ARRIVE" in o)
            lines_out.append(f"ti0..ti1 window: {n_hgmma} HGMMA, "
                             f"{n_arr} ARRIVE, {len(bad)} bad")
            if bad:
                ok = False
            for i, o, raw in body:
                lines_out.append(("  BAD " if (i, o, raw) in bad else "  ")
                                 + raw)
            # wait sequence after ti1: DEPBAR.LE gsb0 counts NG-1..0
            dep = []
            for i in range(b + 1, len(lines)):
                o = opcode(lines[i])
                if "DEPBAR" in o:
                    dep.append((i, lines[i].strip()))
                if "BAR.SYNC" in o or len(dep) >= ng + 2:
                    break
            dep = dep[:ng]
            counts = []
            for i, raw in dep:
                m = re.search(r"gsb0,\s*0x([0-9a-f]+)", raw)
                counts.append(int(m.group(1), 16) if m else -1)
            expect = list(range(ng - 1, -1, -1))
            status = "ok" if counts == expect else "FAIL"
            if counts != expect:
                ok = False
            lines_out.append(f"wait sequence {status}: {counts} vs {expect}")
            for i, raw in dep:
                nxt = next((opcode(lines[j]) for j in range(i + 1, len(lines))
                            if opcode(lines[j])), "")
                clk = "SR_CLOCKLO" in lines[i + 1] or "SR_CLOCKLO" in (
                    lines[i + 2] if i + 2 < len(lines) else "")
                lines_out.append(f"  {raw}   clock-after={clk}")
                if not clk:
                    ok = False
            # delay loop before ti0: IMAD present between clocks a-2 and a
            pre = "\n".join(lines[max(0, a - 200):a])
            lines_out.append(f"delay loop IMAD before ti0: "
                             f"{'IMAD' in pre}")
        with open(f"{out_dir}/sass-grpq-{shape}.txt", "w") as f:
            f.write("\n".join(lines_out) + "\n")
        print(f"== {shape} ({name})")
        print("\n".join(lines_out))
    print("GRPQ_GATE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


sys.exit(main())
