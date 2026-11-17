#!/usr/bin/env python3
"""ffma2_operands sass gate + sass_record (protocol v1).

Gate: composition-exact (target class count == 8*K in the clock window)
and harness counts constant across K.  Record: per kernel, for every
target instruction in the window — source register numbers, reuse flag
of the control word, register bank (index mod 4) sharing count.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

CELL_OP = {"F3": "FFMA2", "F2a": "FFMA2", "F2b": "FFMA2", "F1": "FFMA2",
           "A2": "FADD2", "S3": "FFMA"}
HARNESS = {"LDS", "STS", "LD", "ST", "LDC", "ULDC", "LDCU", "BAR", "BRA",
           "NOP", "EXIT", "WARPSYNC", "DEPBAR", "BSSY", "BSYNC", "IADD3",
           "IMAD", "MOV", "UMOV", "LOP3", "ISETP", "SEL", "P2R", "R2P",
           "ULOP3", "UIADD3", "VIADD", "RET", "CALL", "CS2R", "S2R",
           "S2UR", "SYNCS", "CSYNC", "LDG", "STG", "PRMT", "SHF",
           "PLOP3", "IABS", "IADD", "VIADDMNMX", "LEA", "ULEA", "USHF",
           "F2I", "I2F", "F2F", "ELECT", "VOTEU", "VOTE", "R2UR",
           "UP2UR", "UPLOP3", "UMOV"}
REG = re.compile(r"R(\d+)")

NAME = re.compile(r"^_Z\d+k_thrx?2?I?(\d*)([A-Z0-9_]+)Li(\d+)E.*$")


def demangle(n):
    m = re.match(r"^_Z\d+k_thrx2I(\d+)([A-Z0-9_]+)Li(\d+)E.*$", n)
    if m:
        return m.group(2).lstrip("0123456789"), int(m.group(3))
    for kn, cell in (("k_thr_f1", "F1"), ("k_thr_f3", "F3"),
                     ("k_thr_s3", "S3")):
        m = re.match(rf"^_Z\d+{kn}ILi(\d+)E.*$", n)
        if m:
            return cell, int(m.group(1))
    return None


def regs_of(ins):
    toks = [t.strip(",") for t in ins.body.split()[1:]]
    return [t for t in toks if re.fullmatch(r"R\d+", t)]


def reuse(ins):
    # reuse flag = ctrl[17:21]; nonzero means operand marked for reuse
    return (ins.ctrl >> 17) & 0xF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v1", "probe": "ffma2",
           "kernels": {}, "n_ok": 0, "n_fail": 0}
    for name, k in kernels.items():
        d = demangle(name)
        if not d:
            continue
        cell, kk = d
        w = sc.clock_windows(k)
        win = max(w, key=len) if w else []
        target = CELL_OP[cell]
        cnt = Counter(i.root for i in win)
        tinsts = [i for i in win if i.root == target]
        recs = []
        for i in tinsts:
            rs = regs_of(i)
            srcs = rs[1:]
            banks = [int(r[1:]) % 4 for r in srcs]
            share = len(banks) - len(set(banks))
            recs.append({"addr": hex(i.addr), "text": i.text,
                         "srcs": srcs, "reuse": reuse(i),
                         "banks": banks, "bank_share": share})
        res = {"cell": cell, "k": kk, "target": target,
               "count": cnt.get(target, 0), "expect": 8 * kk,
               "sass_counts": dict(cnt),
               "non_harness": {r: n for r, n in cnt.items()
                               if r != target and r not in HARNESS},
               "records": recs[:4], "n_target": len(recs),
               "ok": not w or (cnt.get(target, 0) == 8 * kk
                               and not {r: n for r, n in cnt.items()
                                        if r != target
                                        and r not in HARNESS})}
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
    print(f"sass_check ffma2: {out['n_ok']}/{len(out['kernels'])} ok")
    for key, r in out["kernels"].items():
        if not r["ok"]:
            print(" FAIL", key, r.get("why"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
