#!/usr/bin/env python3
"""register_operands sass gate + records (protocol v1).

Gate: target-class count == 8*K in the clock window and no non-harness
class in window.  Records: per kernel, for every target instruction in
the window -> reads (distinct general source registers; a 64-bit pair
counts as 2; RZ/immediates/constant-bank/uniform/.reuse operands do not
count) and bank_max (max reads in one bank = reg index mod 4).  A kernel
whose target instructions do not share one (reads, bank_max) is marked
mixed and excluded downstream.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

CELL_CLASS = {
    "FFMA_xxx": "FFMA", "FFMA_xkk": "FFMA", "FFMA_xyk": "FFMA",
    "FFMA_xyz": "FFMA",
    "FFMA2_xxx": "FFMA2", "FFMA2_xkk": "FFMA2", "FFMA2_xyk": "FFMA2",
    "FFMA2_xyz": "FFMA2",
    "FADD2_xx": "FADD2", "FADD2_xk": "FADD2", "FADD2_xy": "FADD2",
    "IMAD_xxx": "IMAD", "IMAD_xkk": "IMAD", "IMAD_xyk": "IMAD",
    "IMAD_xyz": "IMAD",
    "FADD_xx": "FADD", "FADD_xk": "FADD", "FADD_xy": "FADD",
}
HARNESS = {"LDS", "STS", "LD", "ST", "LDC", "ULDC", "LDCU", "BAR", "BRA",
           "NOP", "EXIT", "WARPSYNC", "DEPBAR", "BSSY", "BSYNC", "IADD3",
           "IMAD", "MOV", "UMOV", "LOP3", "ISETP", "SEL", "P2R", "R2P",
           "ULOP3", "UIADD3", "VIADD", "RET", "CALL", "CS2R", "S2R",
           "S2UR", "SYNCS", "CSYNC", "LDG", "STG", "PRMT", "SHF",
           "PLOP3", "IABS", "IADD", "VIADDMNMX", "LEA", "ULEA", "USHF",
           "F2I", "I2F", "F2F", "ELECT", "VOTEU", "VOTE", "R2UR",
           "UP2UR", "UPLOP3", "VIADDMNMX"}

# operand token -> list of general register indices it reads (0/1/2)
REG = re.compile(r"R(\d+)")
NON_REG = re.compile(r"^(RZ|UR|PT|P\d|SRZ|UZ|c\[|0x|#|-?\d)")


def reads_banks(ins):
    toks = ins.body.split()[1:]
    reads, banks = [], []
    for t in toks[1:]:  # drop destination
        t = t.strip(",")
        if t.endswith("F32x2.HI_LO") or t.endswith("F32.HI_LO"):
            # 64-bit pair: R2n + R2n+1
            m = REG.match(t)
            if m and not t.startswith("RZ"):
                n = int(m.group(1))
                if ".reuse" in t:
                    continue
                reads += [n, n + 1]
                banks += [n % 4, (n + 1) % 4]
            continue
        m = REG.match(t)
        if m and not NON_REG.match(t) and ".reuse" not in t:
            n = int(m.group(1))
            reads.append(n)
            banks.append(n % 4)
    bc = Counter(banks)
    return len(set(reads)), (max(bc.values()) if bc else 0)


def demangle(name):
    m = re.match(r"^_Z\d+k_thrI(\d+)([A-Za-z0-9_]+)Li(\d+)ELi(\d+)E.*$",
                 name)
    if m:
        return m.group(2).lstrip("0123456789"), int(m.group(3)), \
            int(m.group(4))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v1", "probe": "regops",
           "kernels": {}, "n_ok": 0, "n_fail": 0}
    for name, k in kernels.items():
        d = demangle(name)
        if not d:
            continue
        opname, kk, pert = d
        cell = opname.replace("_", ".", 1) + (".p1" if pert else ".p0")
        w = sc.clock_windows(k)
        win = max(w, key=len) if w else []
        cnt = Counter(i.root for i in win)
        target = CELL_CLASS[opname]
        tin = [i for i in win if i.root == target]
        rb = [reads_banks(i) for i in tin]
        rbd = Counter(rb)
        mixed = len(rbd) > 1
        recs = [{"addr": hex(i.addr), "text": i.text,
                 "reads": r[0], "bank_max": r[1]}
                for i, r in zip(tin, rb)]
        res = {"cell": cell, "k": kk, "target": target,
               "count": cnt.get(target, 0), "expect": 8 * kk,
               "sass_counts": dict(cnt),
               "non_harness": {r: n for r, n in cnt.items()
                               if r != target and r not in HARNESS},
               "reads_bank_dist": {f"{r}|{b}": n
                                   for (r, b), n in rbd.items()},
               "reads": (list(rbd)[0][0] if len(rbd) == 1 else None),
               "bank_max": (list(rbd)[0][1] if len(rbd) == 1 else None),
               "mixed": mixed, "n_target": len(recs),
               "records": recs[:6]}
        res["ok"] = (cnt.get(target, 0) == 8 * kk
                     and not res["non_harness"])
        if not res["ok"]:
            res["why"] = []
            if cnt.get(target, 0) != 8 * kk:
                res["why"].append(
                    f"{target} count {cnt.get(target,0)} != 8x{kk}")
            if res["non_harness"]:
                res["why"].append(f"non-harness {res['non_harness']}")
        out["kernels"][f"{cell}/{kk}"] = res
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = len(out["kernels"]) - out["n_ok"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check regops: {out['n_ok']}/{len(out['kernels'])} ok")
    for key, r in out["kernels"].items():
        if not r["ok"]:
            print(" FAIL", key, r.get("why"), r.get("reads_bank_dist"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
