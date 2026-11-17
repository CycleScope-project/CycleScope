#!/usr/bin/env python3
"""SASS gate for the mma_sync probe (protocol.json sass_gate).

    sass_check.py --sass <mma.sass> --out <checks.json>

One record per kernel, key mma/<variant>/<u>.  Between the two
SR_CLOCKLO reads the window must hold exactly U (lat) or 8K (thr)
instructions of the cell's class root with the cell's shape token
(DMMA 8x8x4 / HMMA 16816 / IMMA 16832; sparse HMMA 16832 / sparse IMMA
16864) and nothing else but the allowed window noise.  .SP is recorded;
when the toolchain prints no .SP the sparse cells are gated on the
shape token, which only the sparse forms carry.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

CELLS = {
    "DMMA":      {"root": "DMMA", "shape": "8x8x4", "sparse": False},
    "HMMA_F16":  {"root": "HMMA", "shape": "16816", "sparse": False},
    "HMMA_BF16": {"root": "HMMA", "shape": "16816", "sparse": False},
    "IMMA_S8":   {"root": "IMMA", "shape": "16832", "sparse": False},
    "HMMA_SP":   {"root": "HMMA", "shape": "16832", "sparse": True},
    "IMMA_SP":   {"root": "IMMA", "shape": "16864", "sparse": True},
    # protocol v2: BMMA gated on count only; observed shape tokens
    # are recorded (b1 mma -> BMMA.168256.AND.POPC or similar)
    "BMMA":      {"root": "BMMA", "shape": "", "sparse": False},
}
# pin traffic, clock reads and loop control are the only other
# instructions a window may hold: service_latency alu3's allow list plus
# UIADD3 / VIADD (rep-loop induction, protocol addendum 2026-11-05)
ALLOWED = {"LDS", "STS", "LD", "ST", "BAR", "BRA", "ISETP", "IADD3",
           "UIADD3", "VIADD", "NOP", "BSSY", "BSYNC", "CS2R", "S2R",
           "EXIT", "WARPSYNC", "DEPBAR", "MOV", "IMAD"}
# sm_103a ptxas wraps each sparse mma.sync in MOV + CALL.REL.NOINC +
# WARPSYNC per instance (real kernels carry the same sequence): sparse
# cells additionally allow the call sequence and record it
SPARSE_CALL_ALLOWED = {"CALL", "RET", "WARPSYNC", "MOV"}

_NAME = re.compile(
    r"^_Z\d+(k_lat|k_thr)I(\d+)([A-Z0-9_]+)Li(\d+)E(?:Li(\d+)E)?.*$")


def demangle(name):
    m = _NAME.match(name)
    if not m:
        return None
    kind, op, u = m.group(1), m.group(3), int(m.group(4))
    wps = int(m.group(5)) if m.group(5) else 1
    return kind, op, u, wps


def widest_window(k):
    ws = sc.clock_windows(k)
    return max(ws, key=len) if ws else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v1", "probe": "mma",
           "sass": a.sass, "kernels": {}, "unclassified": []}
    for name, k in kernels.items():
        d = demangle(name)
        if d is None:
            out["unclassified"].append(name)
            continue
        kind, cell, u, wps = d
        spec = CELLS[cell]
        mode = "lat" if kind == "k_lat" else "thr"
        variant = (f"{mode}.{cell}" if kind == "k_lat"
                   else f"{mode}.{cell}.wps{wps}")
        want = u if kind == "k_lat" else 8 * u
        res = {"key": f"mma/{variant}/{u}", "op": variant, "mode": mode,
               "u": u, "wps": wps, "cell": cell, "kernel": name,
               "root": spec["root"], "shape": spec["shape"],
               "sparse": spec["sparse"], "ok": False, "why": []}
        w = widest_window(k)
        if not w:
            res["why"].append("no clock window")
            out["kernels"][res["key"]] = res
            continue
        hits = [i for i in w if i.root == spec["root"] and
                spec["shape"] in i.text]
        if not spec["shape"]:
            res["shape_tokens"] = sorted({i.opcode for i in hits})
        sp_seen = any(".SP" in i.opcode.upper() or " SP" in
                      i.opcode.upper() for i in hits)
        res["sp_printed"] = sp_seen
        allowed = set(ALLOWED)
        has_call = any(i.root == "CALL" for i in w)
        if spec["sparse"]:
            allowed |= SPARSE_CALL_ALLOWED
            res["sparse_call_subroutine"] = has_call
        other = [i.opcode for i in w
                 if i not in hits and i.root not in allowed]
        counts = Counter(i.opcode for i in w)
        res.update({"sass_counts": dict(counts), "n_hits": len(hits),
                    "want": want, "n_other": len(other),
                    "window_pcs": [hex(w[0].addr), hex(w[-1].addr)]})
        if len(hits) != want:
            res["why"].append(
                f"{spec['root']}.{spec['shape']} count {len(hits)} "
                f"!= {want}")
        # sparse cells: gate on the unique shape token; record whether
        # the toolchain printed .SP (either suffices per the protocol)
        if spec["sparse"]:
            res["sparse_gate"] = (
                "probe .SP" if sp_seen
                else f"shape {spec['shape']} (no .SP printed)")
        elif sp_seen:
            res["why"].append(".SP modifier on a dense cell")
        if other:
            res["why"].append(
                f"other instructions in the window: {sorted(set(other))}")
        res["ok"] = not res["why"]
        out["kernels"][res["key"]] = res
    n_match = len(out["kernels"])
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = n_match - out["n_ok"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    ok = out["n_ok"]
    print(f"sass_check: {ok}/{n_match} gated kernels ok")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"])
    # 0 matched kernels = gate failure of the binary
    return 0 if n_match else 2


if __name__ == "__main__":
    sys.exit(main())
