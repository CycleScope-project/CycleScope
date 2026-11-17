#!/usr/bin/env python3
"""SASS gate for the store_flood probe (protocol v1, frozen).  Per kernel
k_flood<V> the timed window is the run of SR_CLOCKLO reads inside the
repetition loop: exactly 49 reads; between two neighbouring reads exactly
8 STG of the variant's width (0 for the control) and no other memory,
barrier, fence or branch instruction between the first and the last read
(address arithmetic, MOV and integer ops are allowed).

Usage: check.py <sass file> <checks.json out>
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

STAMPS = 49
STORES = 8
# V -> expected STG opcode (None = control, no stores)
WIDTH = {2: "STG.E.U16", 4: "STG.E", 8: "STG.E.64", 16: "STG.E.128", 0: None}
# v2 kernel k_floodx<GEO>: 0 none, 1 full16 (w16), 2 line32 (w4),
# 3 row4, 4 row4_refill.  Alignment / arrival instructions are only
# allowed outside the clock-read window (the gate checks between the
# first and the last read only).
GEO_WIDTH = {0: None, 1: "STG.E.128", 2: "STG.E", 3: "STG.E.U16",
             4: "STG.E.U16", 5: "STG.E.U16", 6: "STG.E.U16",
             7: "STG.E.U16"}
MEM = ("LDG", "STG", "LDS", "STS", "LD.", "ST.", "LDL", "STL", "ATOM", "RED",
       "MEMBAR", "BAR", "ERRBAR", "LDGSTS", "UTMA", "UBLK", "SYNCS", "CCTL",
       "FENCE", "DEPBAR", "EXIT", "BRA", "BRX", "BRXU", "JMP", "JMX",
       "CALL", "RET", "SSY", "BSSY", "BSYNC", "WARPSYNC")


def check_kernel(kern, v: int, want="derive") -> dict:
    out = {"kernel": kern.name, "variant": v, "ok": True, "fails": []}
    clk = [i for i, ins in enumerate(kern.insts) if "SR_CLOCKLO" in ins.text]
    if len(clk) != STAMPS:
        out["ok"] = False
        out["fails"].append(f"clock_reads={len(clk)} (want {STAMPS})")
        return out
    if want == "derive":
        want = WIDTH[v]
    n_stg = 0
    for a, b in zip(clk, clk[1:]):
        between = kern.insts[a + 1:b]
        stg = [x for x in between if x.opcode.startswith("STG")]
        n_stg += len(stg)
        if len(stg) != (0 if want is None else STORES):
            out["ok"] = False
            out["fails"].append(
                f"group @{kern.insts[a].addr:#x}: stg={len(stg)} "
                f"(want {0 if want is None else STORES})")
        for ins in between:
            op = ins.opcode
            if op.startswith("STG"):
                if want is None or op != want:
                    out["ok"] = False
                    out["fails"].append(f"width {op} @{ins.addr:#x}")
            elif any(op.startswith(k) for k in MEM):
                out["ok"] = False
                out["fails"].append(f"foreign {op} @{ins.addr:#x}")
    if want is not None and n_stg != STORES * (STAMPS - 1):
        out["ok"] = False
        out["fails"].append(f"total_stg={n_stg} (want {STORES * (STAMPS - 1)})")
    return out


def check_bridge(kern, flags: int, groups: int) -> dict:
    """Bridge kernels: `groups` clock-window intervals of exactly 8
    STG.E.U16; with the filler bit each interval also carries one FMUL,
    one F2FP and one ISETP per store; no other memory/branch op inside
    the window.  stamps = groups + 1 clock reads."""
    out = {"kernel": kern.name, "variant": f"bridge{flags}x{groups}",
           "ok": True, "fails": []}
    clk = [i for i, ins in enumerate(kern.insts) if "SR_CLOCKLO" in ins.text]
    if len(clk) != groups + 1:
        out["ok"] = False
        out["fails"].append(f"clock_reads={len(clk)} (want {groups + 1})")
        return out
    for a, b in zip(clk, clk[1:]):
        between = kern.insts[a + 1:b]
        stg = [x for x in between if x.opcode.startswith("STG")]
        if len(stg) != STORES or any(x.opcode != "STG.E.U16" for x in stg):
            out["ok"] = False
            out["fails"].append(
                f"group @{kern.insts[a].addr:#x}: " +
                ",".join(x.opcode for x in stg))
        if flags & 1:   # filler: FMUL + F2FP + ISETP per store
            for want, cnt in (("FMUL", 8), ("F2F", 8), ("ISETP", 8)):
                n = sum(1 for x in between if x.opcode.startswith(want))
                if n != cnt:
                    out["ok"] = False
                    out["fails"].append(
                        f"{want}={n} (want {cnt}) @{kern.insts[a].addr:#x}")
        for ins in between:
            op = ins.opcode
            if op.startswith("STG"):
                continue
            if any(op.startswith(k) for k in MEM):
                out["ok"] = False
                out["fails"].append(f"foreign {op} @{ins.addr:#x}")
    return out


def main() -> int:
    kernels = parse(Path(sys.argv[1]).read_text())
    # optional arg: variants matched by this regex must pass; others are
    # still reported but do not affect the gate
    focus = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
    results = []
    for name, kern in sorted(kernels.items()):
        m = re.search(r"k_floodILi(\d+)E", name)
        if m:
            results.append(check_kernel(kern, int(m.group(1))))
            continue
        m = re.search(r"k_flood5ILi(\d+)E", name)
        if m:
            r = check_kernel(kern, int(m.group(1)))
            r["variant"] = f"v5k{m.group(1)}"
            results.append(r)
            continue
        m = re.search(r"k_bisect_[bn]ILi(\d+)E", name)
        if m:
            r = check_kernel(kern, f"bisect{int(m.group(1))}",
                             want="STG.E")
            results.append(r)
            continue
        m = re.search(r"k_floodxILi(\d+)E", name)
        if m:
            g = int(m.group(1))
            r = check_kernel(kern, f"geo{g}", want=GEO_WIDTH[g])
            results.append(r)
            continue
        m = re.search(r"k_bridgeILi(\d+)ELi(\d+)E", name)
        if m:
            results.append(check_bridge(kern, int(m.group(1)),
                                        int(m.group(2))))
    geo = [r for r in results if str(r["variant"]).startswith("geo")]
    br = [r for r in results if str(r["variant"]).startswith("bridge")]
    bs = [r for r in results if str(r["variant"]).startswith("bisect")]
    v1 = [r for r in results
          if not str(r["variant"]).startswith(
              ("geo", "bridge", "bisect", "v5k"))]
    gated = [r for r in results
             if focus is None or focus.search(str(r["variant"]))]
    passed = (len(geo) == len(GEO_WIDTH) and len(v1) == len(WIDTH)
              and len(br) == 6 and len(bs) == 24
              and all(r["ok"] for r in gated))
    Path(sys.argv[2]).write_text(
        json.dumps({"passed": passed, "kernels": results}, indent=1) + "\n")
    print(f"store_flood check: {len(results)} kernels, passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
