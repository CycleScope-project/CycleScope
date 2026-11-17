#!/usr/bin/env python3
"""fp32_mix sass gate: window = exactly the variant's ops, emits the
window program (opcode/dst/srcs/reuse_bits) for the Replica.
Usage: sass_check.py <build_dir> <checks_out.json>"""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "crossings" / "trywait_timeout"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "dispatch_contention"))
from sass_ctrl import parse  # noqa
from sass_check_v2 import prog_entry, clock_spans  # noqa

K = re.compile(r"k_cellILi(\d+)ELi(\d+)E")
VAR = {0: "quartet", 1: "trio", 2: "mufu", 3: "fmnmx"}
WANT = {"quartet": {"FMNMX": 1, "FFMA": 1, "MUFU.EX2": 1, "FADD": 1},
        "trio": {"FMNMX": 1, "FFMA": 1, "FADD": 1},
        "mufu": {"MUFU.EX2": 4}, "fmnmx": {"FMNMX": 4}}


def window_ok(win, vname, u):
    cnt = Counter(i.opcode for i in win)
    for opc, m in WANT[vname].items():
        got = sum(n for k, n in cnt.items() if k == opc or
                  k.startswith(opc + "."))
        if got != u * m:
            return False
    return len(win) <= u * sum(WANT[vname].values()) + 4 and all(
        i.opcode in WANT[vname] or i.opcode.startswith("LDG")
        or any(i.opcode.startswith(o + ".") for o in WANT[vname])
        for i in win)


def main():
    build, out = sys.argv[1], Path(sys.argv[2])
    sass = Path(build) / "sass" / "mix.sass"
    kernels = parse(sass.read_text())
    res, fails = {}, 0
    for name, kern in kernels.items():
        m = K.search(name)
        if not m:
            continue
        v, u = int(m.group(1)), int(m.group(2))
        vn = VAR[v]
        good = [s for s in clock_spans(kern)
                if window_ok(kern.insts[s[0] + 1:s[1]], vn, u)]
        ok = bool(good)
        ent = {"u": u, "ok": ok, "windows": len(good), "kernel": name}
        if ok:
            a, b = good[0]
            prog = kern.insts[a:b + 1]
            ent["program"] = [prog_entry(i) for i in prog]
        res[f"{vn}/u{u}"] = ent
        fails += 0 if ok else 1
        if not ok:
            print(f"FAIL {name}: no clean {vn} u={u} window")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"sass_check: {len(res)} kernels, {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
