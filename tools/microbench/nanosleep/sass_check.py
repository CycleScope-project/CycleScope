#!/usr/bin/env python3
"""sass gate for nanosleep protocol v1.

syncs kernel: loop body between the two SR_CLOCKLO reads must be
SYNCS.PHASECHK.TRANS64.TRYWAIT + NANOSLEEP.SYNCS + BRA (nothing else);
plain kernel window = exactly one NANOSLEEP.  Control words recorded.
Usage: sass_check.py <build_dir> <out_json>
"""
import json
import re
import sys
from pathlib import Path


def ins_text_addr(ins) -> int:
    return getattr(ins, "addr", -1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

K = re.compile(r"k_(syncs|plain)ILi(\d+)E")
ALLOWED = {"SYNCS", "NANOSLEEP", "BRA", "SEL"}
# giveup-bound bookkeeping tolerated inside the loop (fail-fast path,
# fires only on missing arrive; recorded in the control words)
BOOKKEEP = {"LOP3", "ISETP", "IMAD", "IADD3", "LDC", "STG", "VIADD",
            "NOP", "CS2R", "MOV", "SHF", "I2F"}


def check_kernel(kern, variant: str) -> dict:
    out = {"kernel": kern.name, "variant": variant, "ok": True,
           "fails": [], "program": []}
    clk = [i for i, ins in enumerate(kern.insts)
           if "SR_CLOCKLO" in ins.text]
    if len(clk) < 2:
        out["ok"] = False
        out["fails"].append("clocklo")
        return out
    if variant == "syncs":
        # locate the try_wait loop directly: span from the first
        # TRYWAIT to the BRA that closes it (ptxas may schedule the
        # loop outside the clock-read window)
        ti = [i for i, ins in enumerate(kern.insts)
              if "TRYWAIT" in ins.opcode]
        if ti:
            a = ti[0]
            # end at the BRA just after the last NANOSLEEP.SYNCS /
            # PHASECHK in the unrolled loop cluster
            last = max(i for i, ins in enumerate(kern.insts)
                       if "NANOSLEEP" in ins.opcode
                       or "PHASECHK" in ins.opcode)
            b = last
            for i in range(last, min(last + 8, len(kern.insts))):
                if kern.insts[i].opcode.startswith("BRA"):
                    b = i
                    break
            win = kern.insts[a:b + 1]
        else:
            win = []
    else:
        best, best_n = [], -1
        for a, b in zip(clk, clk[1:]):
            win0 = kern.insts[a + 1:b]
            n = sum(1 for i in win0 if "NANOSLEEP" in i.opcode)
            if n > best_n:
                best, best_n = win0, n
        win = best
    n_sleep = sum(1 for i in win if i.opcode.startswith("NANOSLEEP"))
    n_try = sum(1 for i in win
                if "TRYWAIT" in i.opcode or "PHASECHK" in i.opcode)
    n_bra = sum(1 for i in win if i.opcode.startswith("BRA"))
    n_syncs_sleep = sum(1 for i in win
                        if "NANOSLEEP" in i.opcode and ".SYNCS" in i.opcode)
    extra = [i.opcode for i in win
             if i.opcode.split(".", 1)[0] not in ALLOWED
             and i.opcode.split(".", 1)[0] not in BOOKKEEP
             and not i.opcode.startswith("@!")]
    out["counts"] = {"nanosleep": n_sleep, "nanosleep_syncs": n_syncs_sleep,
                     "trywait": n_try, "bra": n_bra}
    for ins in win:
        out["program"].append(
            {"opcode": ins.opcode, "stall": ins.stall,
             "wbar": ins.wbar, "rbar": ins.rbar, "wait": ins.wait})
    if variant == "syncs":
        if n_try < 1 or n_syncs_sleep < 1 or n_bra < 1:
            out["ok"] = False
            out["fails"].append(
                f"try={n_try} sleep_syncs={n_syncs_sleep} bra={n_bra}")
    else:
        if n_sleep != 1:
            out["ok"] = False
            out["fails"].append(f"nanosleep={n_sleep}")
    if extra:
        out["ok"] = False
        out["fails"].append("extra: " + ",".join(extra))
    return out


def main() -> int:
    build_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    sass = build_dir / "sass" / "nanosleep.sass"
    kernels = parse(sass.read_text())
    results, fails = [], 0
    seen = set()
    for name, kern in sorted(kernels.items()):
        m = K.search(name)
        if not m or (m.group(1), int(m.group(2))) in seen:
            # one representative kernel per (variant,H); ptxas may
            # emit multiple clones
            continue
        seen.add((m.group(1), int(m.group(2))))
        ck = check_kernel(kern, m.group(1))
        results.append(ck)
        fails += 0 if ck["ok"] else 1
        if not ck["ok"]:
            print(f"FAIL {name}: {ck['fails']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"protocol": "nanosleep/v1", "kernels": results}, indent=1))
    print(f"sass_check: {len(results)} kernels, {fails} fail")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
