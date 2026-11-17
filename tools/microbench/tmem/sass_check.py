#!/usr/bin/env python3
"""SASS gate for the tmem probe (protocol.json sass_gate).

    sass_check.py --sass <ldtm_warps.sass> --out <checks.json>

Per kernel k_ldtm<Tm_32x32b_xN> (one kernel per shape; N is a runtime
blockDim): every run between two consecutive SR_CLOCKLO reads must hold
exactly 1 LDTM whose printed shape is 32x32b.x<shape> plus the
tcgen05.wait::ld form, and nothing outside the allow list.  0 classified
kernels = gate failure.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

# pin traffic, clock reads, loop control and the tcgen05 wait form are the
# only other instructions a window may hold (mma_sync's allow list +
# STG/R2UR is not needed: the t_s / t_e stores land after t_e)
ALLOWED = {"LDS", "STS", "LD", "ST", "STG", "BAR", "BRA", "ISETP", "IADD3",
           "UIADD3", "VIADD", "NOP", "BSSY", "BSYNC", "CS2R", "S2R",
           "EXIT", "WARPSYNC", "DEPBAR", "MOV", "IMAD", "R2UR", "IMNMX",
           "SEL", "LEA", "SHF", "LOP3", "I2F", "F2I", "LDG", "UMOV",
           "USHF", "ULOP3", "UISETP", "PRMT", "VOTEU", "IABS", "FADD",
           "S2UR", "PLOP3", "R2P", "POPC", "FLO", "BREV"}

_NAME = re.compile(r"Tm_(32x32b|16x32bx2)_x(\d+)")
_SHAPE = re.compile(r"\.x(\d+)")
_WAIT = re.compile(r"WAIT|WARPSYNC", re.IGNORECASE)


def demangle(name):
    m = _NAME.search(name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def is_wait_form(i):
    # tcgen05.wait::ld prints as WARPSYNC.ALL / a *WAIT* op on sm_103a
    return bool(_WAIT.search(i.opcode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", default="ldtm_warps")
    a = ap.parse_args()
    x64 = a.probe == "x64"
    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v1", "probe": a.probe,
           "sass": a.sass, "kernels": {}, "unclassified": []}
    for name, k in kernels.items():
        fam, shape = demangle(name)
        if shape is None or (x64 and "k_tm" not in name):
            out["unclassified"].append(name)
            continue
        res = {"key": f"ldtm/{fam}.x{shape}" if x64 else f"ldtm/x{shape}",
               "shape": shape, "kernel": name,
               "ok": False, "why": [], "n_windows": 0}
        ws = sc.clock_windows(k)
        if not ws:
            res["why"].append("no clock window")
            out["kernels"][res["key"]] = res
            continue
        res["n_windows"] = len(ws)
        bad = 0
        n_ld = 0
        for w in ws:
            if not any(i.root == "LDTM" for i in w):
                continue   # inter-rep overhead window (t_e -> next t_s)
            n_ld += 1
            if x64:
                # spelling 16dp32bit_t0_t15/_t16_t31.x<N> (two SASS per
                # PTX op) or 32x32b.x<N> controls
                hits = [i for i in w if i.root == "LDTM" and
                        _SHAPE.search(i.text) and
                        int(_SHAPE.search(i.text).group(1)) == shape]
            else:
                hits = [i for i in w if i.root == "LDTM" and
                        _SHAPE.search(i.text) and
                        int(_SHAPE.search(i.text).group(1)) == shape]
            waits = [i for i in w if is_wait_form(i)]
            other = [i.opcode for i in w
                     if i not in hits and i not in waits
                     and i.root not in ALLOWED]
            counts = Counter(i.opcode for i in w)
            res.setdefault("window_sass_counts", dict(counts))
            want = 2 if not x64 else (
                # x64 kernel names hold U as template arg; windows with
                # LDTM carry 2*U (each PTX op lowers to t0_t15 + t16_t31)
                # for 16x32bx2 shapes, 1*U for 32x32b
                2)
            if x64:
                um = re.search(r"k_tm<[^,]+,\s*\d+,\s*(\d+)", name)
                want = 0
                if um:
                    per = 2 if fam == "16x32bx2" else 1
                    want = per * int(um.group(1))
                if want and len(hits) != want:
                    bad += 1
                    res["why"].append(
                        f"window with {len(hits)} {fam}.x{shape} LDTM "
                        f"(want {want}; opcodes {sorted(counts)})")
            elif len(hits) != want:
                bad += 1
                res["why"].append(
                    f"window with {len(hits)} 32x32b.x{shape} LDTM "
                    f"(opcodes {sorted(counts)})")
            if not waits:
                bad += 1
                res["why"].append(
                    f"window without tcgen05.wait::ld "
                    f"(opcodes {sorted(counts)})")
            if other:
                bad += 1
                res["why"].append(
                    f"other instructions in a window: {sorted(set(other))}")
        res["n_ldtm_windows"] = n_ld
        if n_ld == 0:
            res["why"].append("no LDTM window")
        res["n_bad_windows"] = bad
        res["ok"] = not res["why"]
        out["kernels"][res["key"]] = res
    n_match = len(out["kernels"])
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = n_match - out["n_ok"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check: {out['n_ok']}/{n_match} gated kernels ok")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"][:3])
    return 0 if n_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
