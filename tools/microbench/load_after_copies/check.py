#!/usr/bin/env python3
"""SASS gate for load_after_copies (protocol.json).  Per kernel:
 - the timed window (one adjacent pair of SR_CLOCKLO reads that contains
   loads) holds exactly 8 loads of the declared kind (LDG for k_lac,
   LDL for k_lacl), no LDGSTS / store / barrier;
 - the copy loop holds LDGSTS of the declared form (ca: no .BYPASS,
   cg: .BYPASS).

Usage: check.py <sass> <checks.json out>
"""
import json, re, sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

FORBIDDEN = ("LDGSTS", "STG", "STS", "STL", "LDL", "ATOM", "RED",
             "MEMBAR", "BAR", "ERRBAR", "SYNCS", "BSSY", "BSYNC",
             "BRA", "JMP", "EXIT")


def check(kern, form, want_load, nldl=0):
    out = {"kernel": kern.name, "ok": True, "fails": []}

    def need(cond, msg):
        if not cond:
            out["ok"] = False
            out["fails"].append(msg)

    clk = [i for i, ins in enumerate(kern.insts)
           if "SR_CLOCKLO" in ins.text]
    need(len(clk) >= 2, f"clock_reads={len(clk)} (want >=2)")
    chains = []
    for a, b in zip(clk, clk[1:]):
        between = kern.insts[a + 1:b]
        loads = [i for i in between
                 if (i.opcode.startswith("LDG") and
                     not i.opcode.startswith("LDGSTS"))
                 or i.opcode.startswith("LDL")]
        if loads:
            chains.append((a, between, loads))
    if want_load == "LDG":
        good = [c for c in chains
                if all(i.opcode.startswith("LDG") for i in c[2])]
        need(len(good) == 1,
             f"load windows={len(chains)} good={len(good)}")
        if good:
            _, between, loads = good[0]
            need(len(loads) == 8,
                 f"window {len(loads)} LDG (want 8): "
                 + ",".join(i.opcode for i in between))
            for i in between:
                if i.opcode.startswith("LDG") or \
                        i.opcode.startswith("LDC"):
                    continue
                if any(i.opcode.startswith(p) for p in FORBIDDEN):
                    need(False, f"foreign {i.opcode} @{i.addr:#x}")
                if i.opcode.startswith("LD") or \
                        i.opcode.startswith("ST"):
                    need(False, f"foreign mem {i.opcode} @{i.addr:#x}")
    else:   # LDL
        good = [c for c in chains
                if all(i.opcode.startswith("LDL") for i in c[2])]
        need(len(good) == 1,
             f"LDL windows={len(chains)} good={len(good)}")
        if good:
            _, between, loads = good[0]
            need(len(loads) == (nldl or 8),
                 f"window {len(loads)} LDL (want {nldl or 8})")
            for i in between:
                if i.opcode.startswith("LDL") or \
                        i.opcode.startswith("LDC"):
                    continue
                if any(i.opcode.startswith(p) for p in FORBIDDEN) or \
                        "DEPBAR" in i.opcode:
                    need(False, f"foreign {i.opcode} @{i.addr:#x}")
                if i.opcode.startswith(("LD", "ST")):
                    need(False, f"foreign mem {i.opcode} @{i.addr:#x}")
    gsts = [i for i in kern.insts if i.opcode.startswith("LDGSTS")]
    need(len(gsts) >= 1, "no LDGSTS copy found")
    for i in gsts:
        bypass = "BYPASS" in i.opcode
        if form == 0 and bypass:
            need(False, f"ca copy has BYPASS: {i.opcode}")
        if form == 1 and not bypass:
            need(False, f"cg copy lacks BYPASS: {i.opcode}")
    return out


def main() -> int:
    kerns = parse(Path(sys.argv[1]).read_text())
    results = []
    for kern in kerns.values():
        m = re.search(r"k_lacILi(\d+)ELi(\d+)E", kern.name)
        if m:
            results.append(check(kern, int(m.group(1)), "LDG"))
            continue
        m = re.search(r"k_lacl3ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", kern.name)
        if m:
            results.append(check(kern, int(m.group(3)), "LDL",
                                 int(m.group(1))))
            continue
        m = re.search(r"k_lacl2ILi(\d+)ELi(\d+)ELi(\d+)E", kern.name)
        if m:
            results.append(check(kern, int(m.group(3)), "LDL",
                                 int(m.group(1))))
            continue
        m = re.search(r"k_laclILi(\d+)E", kern.name)
        if m:
            results.append(check(kern, int(m.group(1)), "LDL"))
    bad = [r for r in results if not r["ok"]]
    passed = len(results) == 37 and not bad
    print(f"load_after_copies check: {len(results)} kernels, "
          f"passed={passed}")
    for r in bad:
        print(r["kernel"], r["fails"][:8])
    Path(sys.argv[2]).write_text(json.dumps(
        {"passed": passed, "kernels": results}, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
