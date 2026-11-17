#!/usr/bin/env python3
"""SASS gate for the store_queue probe (lsu_accept protocol v4, frozen).
Per kernel k_cell<V> the timed window is the run of SR_CLOCKLO reads
inside the repetition loop: there must be exactly 25 reads, and between
two consecutive reads exactly one STG.* of the variant's width (none
for the `none` control) and no other memory / barrier / fence
instruction (address arithmetic, MOV and predicate ops are allowed).
The 24 STGs must use distinct address registers.  Control words of
every window instruction are recorded.

Usage: sass_check_store_queue.py <build_dir> <checks.json out>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

D = 24
WIDTH = {0: "STG.E.128", 1: "STG.E", 2: "STG.E.64", 3: None}  # V_STG16, V_STG4 (STG.E = 32-bit), V_STG8, V_NONE
MEM = ("LDG", "STG", "LDS", "STS", "LD.", "ST.", "LDL", "STL", "ATOM", "RED",
       "MEMBAR", "BAR", "ERRBAR", "LDGSTS", "UTMA", "UBLK", "SYNCS", "CCTL",
       "FENCE", "DEPBAR", "EXIT", "BRA")
_ADDR = re.compile(r"\[(R\d+)(?:\.64)?")


def check_kernel(kern, v: int) -> dict:
    out = {"kernel": kern.name, "variant": v, "ok": True, "fails": [],
           "program": []}
    clk = [i for i, ins in enumerate(kern.insts) if "SR_CLOCKLO" in ins.text]
    # the loop body holds D + 1 consecutive reads; find the longest run
    # of reads with exactly one STG between neighbours
    runs, cur = [], [clk[0]] if clk else []
    for a, b in zip(clk, clk[1:]):
        between = kern.insts[a + 1:b]
        stg = [x for x in between if x.body.startswith("STG")]
        if len(stg) == (0 if WIDTH[v] is None else 1):
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    run = max(runs, key=len) if runs else []
    if len(run) != D + 1:
        out["ok"] = False
        out["fails"].append(f"clock_reads={len(run)} (want {D + 1})")
        return out
    addrs = []
    for a, b in zip(run, run[1:]):
        for ins in kern.insts[a + 1:b]:
            body = ins.body
            out["program"].append({"addr": ins.addr, "text": ins.text,
                                   "ctrl": getattr(ins, "ctrl", None)})
            if body.startswith("STG"):
                if WIDTH[v] is None or not body.startswith(WIDTH[v]) or (
                        v == 1 and body.startswith("STG.E.")):
                    out["ok"] = False
                    out["fails"].append(f"width {body.split()[0]}")
                m = _ADDR.search(body)
                addrs.append(m.group(1) if m else body)
            elif any(body.startswith(k) for k in MEM):
                out["ok"] = False
                out["fails"].append(f"foreign {body.split()[0]} @{ins.addr:#x}")
    if WIDTH[v] is not None and len(set(addrs)) != D:
        out["ok"] = False
        out["fails"].append(f"distinct_address_regs={len(set(addrs))}")
    return out


def main() -> int:
    build, outp = Path(sys.argv[1]), Path(sys.argv[2])
    sass = build / "sass" / "store_queue.sass"
    kernels = parse(sass.read_text())
    results, fails = [], 0
    for name, kern in sorted(kernels.items()):
        m = re.search(r"k_cellILi(\d+)E", name)
        if not m:
            continue
        r = check_kernel(kern, int(m.group(1)))
        fails += 0 if r["ok"] else 1
        results.append(r)
    outp.write_text(json.dumps({"kernels": results, "n_fail": fails},
                               indent=1))
    print(f"sass_check_store_queue: {len(results)} kernels, {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
