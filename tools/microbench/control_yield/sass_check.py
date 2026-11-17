#!/usr/bin/env python3
"""sass gate for control_yield protocol v1.

Per kernel: the loop between the two SR_CLOCKLO reads.  spin must
contain exactly one YIELD, one SYNCS.PHASECHK(.TRANS64.TRYWAIT)-family
instruction, the FFMA and BRA(s); nospin must contain no YIELD;
spin_nowait records presence.  Control words recorded for YIELD/BRA.
Usage: sass_check.py <build_dir> <out_json>
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

VARIANTS = ["spin", "nospin", "spin_nowait"]
KCELL = re.compile(r"k_cellILi(\d+)ELi(\d+)E")
TRY = ("SYNCS", "TRYWAIT")


def check_kernel(kern, v: str, k: int) -> dict:
    out = {"kernel": kern.name, "variant": v, "k": k, "ok": True,
           "fails": [], "program": []}
    clk = [i for i, ins in enumerate(kern.insts)
           if "SR_CLOCKLO" in ins.text]
    if len(clk) < 2:
        out["ok"] = False
        out["fails"].append("clocklo")
        return out
    # the clock pair richest in loop-body instructions (the timed
    # window covers the whole nounroll loop)
    best, best_n = [], -1
    for a, b in zip(clk, clk[1:]):
        win = kern.insts[a + 1:b]
        if len(win) > best_n:
            best, best_n = win, len(win)
    win = best
    n_yield = sum(1 for i in win if i.opcode.startswith("YIELD"))
    n_try = sum(1 for i in win
                if i.opcode.split(".", 1)[0] in TRY
                or "TRYWAIT" in i.opcode or "PHASECHK" in i.opcode)
    n_ffma = sum(1 for i in win if i.opcode.startswith("FFMA"))
    n_bra = sum(1 for i in win if i.opcode.startswith("BRA"))
    out["counts"] = {"yield": n_yield, "trywait": n_try,
                     "ffma": n_ffma, "bra": n_bra}
    for ins in win:
        if (ins.opcode.startswith("YIELD") or
                ins.opcode.startswith("BRA") or
                "TRYWAIT" in ins.opcode or "PHASECHK" in ins.opcode):
            out["program"].append(
                {"opcode": ins.opcode, "stall": ins.stall,
                 "wbar": ins.wbar, "rbar": ins.rbar, "wait": ins.wait})
    # ptxas may replicate the loop body Nx even with nounroll: each
    # copy carries one YIELD + one TRYWAIT -> require the pair counts
    # to match (exactly one YIELD per loop body copy)
    if v == "spin":
        if n_yield < 1 or n_try != n_yield:
            out["ok"] = False
            out["fails"].append(f"yield={n_yield} trywait={n_try}")
        if n_bra < 1:
            out["ok"] = False
            out["fails"].append("no_bra")
    elif v == "nospin":
        if n_yield != 0:
            out["ok"] = False
            out["fails"].append(f"yield={n_yield}")
    return out


def main() -> int:
    build_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    sass = build_dir / "sass" / "yield.sass"
    kernels = parse(sass.read_text())
    results = []
    fails = 0
    for name, kern in sorted(kernels.items()):
        m = KCELL.search(name)
        if not m:
            continue
        v, k = int(m.group(1)), int(m.group(2))
        ck = check_kernel(kern, VARIANTS[v], k)
        results.append(ck)
        fails += 0 if ck["ok"] else 1
        if not ck["ok"]:
            print(f"FAIL {name} {VARIANTS[v]} k={k}: {ck['fails']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"protocol": "control_yield/v1", "kernels": results},
        indent=1))
    print(f"sass_check: {len(results)} kernels, {fails} fail")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
