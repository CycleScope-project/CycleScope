#!/usr/bin/env python3
"""sass_gate.py <row_gather.sass> <gate.txt>

SASS gate for the row_gather probe (protocol.json).  Input is the full
cuobjdump --dump-sass output of the probe binary.  Per probe kernel a timed
window is the SASS between two consecutive clock reads (CS2R .. SR_CLOCKLO).
The gate requires every timed window's LDG opcode sequence to equal the
expected 128-bit forms:

  k_single_plain  [LDG.E.128]
  k_single_hint   [LDG.E.EL.LTC256B.128.CONSTANT]
  k_block_plain   [LDG.E.128] x 7
  k_block_hint    [LDG.E.EL.LTC128B.128.CONSTANT] + [LDG.E.EL.LTC256B.128.CONSTANT] x 6

Loads outside timed windows (e.g. the L1 spoiler) are ignored.  Writes the
matched lines to gate.txt and exits non-zero on failure.
"""
import re
import sys

EXPECT = {
    "k_single_plain": ["LDG.E.128"],
    "k_single_hint": ["LDG.E.EL.LTC256B.128.CONSTANT"],
    "k_block_plain": ["LDG.E.128"] * 7,
    "k_block_hint": (["LDG.E.EL.LTC128B.128.CONSTANT"]
                     + ["LDG.E.EL.LTC256B.128.CONSTANT"] * 6),
    "k_pair": ["LDG.E.128"],
}
PLAIN_OK = re.compile(r"^LDG\.E\.128$")


def opcode(line):
    m = re.search(r"\*/\s+([A-Z0-9._:]+)", line)
    if m:
        return m.group(1)
    p = line.split()
    return p[1] if len(p) > 1 else ""


def main():
    sass, out = sys.argv[1], sys.argv[2]
    funcs = {}
    cur = None
    for line in open(sass, errors="replace"):
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur is not None:
            funcs[cur].append(line.rstrip("\n"))
    log = []
    ok = True
    for want, expected in EXPECT.items():
        name = next((n for n in funcs if want in n), None)
        if name is None:
            log.append(f"MISSING kernel {want}")
            ok = False
            continue
        lines = funcs[name]
        clock = [i for i, l in enumerate(lines)
                 if "CS2R" in l and "SR_CLOCKLO" in l]
        windows = []
        for a, b in zip(clock, clock[1:]):
            ldg = [(i, opcode(lines[i]), lines[i].strip())
                   for i in range(a + 1, b) if "LDG" in lines[i]]
            if ldg:
                windows.append((a, ldg))
        log.append(f"== {want} ({name}): {len(clock)} clock reads, "
                   f"{len(windows)} timed windows with LDG")
        if not windows:
            log.append(f"FAIL {want}: no timed window with LDG")
            ok = False
            continue
        for a, ldg in windows:
            got = [o for _, o, _ in ldg]
            status = "ok" if sorted(got) == sorted(expected) else "FAIL"
            if status == "FAIL":
                ok = False
            log.append(f"window after line {a}: {status} got={got}")
            for i, o, raw in ldg:
                log.append(f"  {raw}")
        stray = [(i, opcode(l), l.strip()) for i, l in enumerate(lines)
                 if "LDG" in l and i not in
                 {i for _, ldg in windows for i, _, _ in ldg}]
        for i, o, raw in stray:
            if o.startswith("LDG") and ("128" in o or "CONSTANT" in o):
                ok = False
                log.append(f"FAIL stray timed-form LDG outside window: {raw}")
            else:
                log.append(f"  (outside window) {raw}")
    log.append("GATE " + ("PASS" if ok else "FAIL"))
    with open(out, "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


sys.exit(main())
