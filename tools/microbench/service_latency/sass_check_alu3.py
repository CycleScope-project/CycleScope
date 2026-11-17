#!/usr/bin/env python3
"""SASS gate for the alu3 probe (service_latency protocol v2).

    sass_check_alu3.py --sass <alu3.sass> --out <checks.json>

One record per kernel, key alu3/<variant>/<u> (the jsonl variant names).
A cell passes when its rep-loop window (the widest run between two
SR_CLOCKLO reads) holds exactly the expected number of instructions of
the class under test (lat: U; thr: 8 chains x K; mma thr: 4 x K) and
nothing else but pin traffic / loop control.  Output layout follows
blackwell-prototype tools/microbench/sass_check.py (mechtree.sass_check.v1) so
model_v2.py reads both with one loader.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

# probe struct name -> jsonl variant op, expected SASS opcode prefix
# probe struct name -> jsonl variant op, expected SASS opcode prefix,
# instructions of that class per chain step (F2F: f32->f64->f32)
OPS = {
    "IMADHI": ("IMADHI", "IMAD.HI", 1),
    "F2F_": ("F2F_", "F2F", 2),
}
# pin traffic, clock reads and the rep-loop control are the only other
# instructions a window may hold
ALLOWED = {"LDS", "STS", "LD", "ST", "BAR", "BRA", "ISETP", "IADD3", "NOP",
           "BSSY", "BSYNC", "CS2R", "S2R", "EXIT", "WARPSYNC", "DEPBAR",
           "MOV" }

_NAME = re.compile(r"^_Z(\d+)(k_lat|k_thr|k_mma_f16_thr|k_mma_f16|k_mma_s8)I(.*)$")


def demangle(name):
    m = _NAME.match(name)
    if not m:
        return None
    kind, rest = m.group(2), m.group(3)
    if kind in ("k_lat", "k_thr"):
        n = int(re.match(r"\d+", rest).group(0))
        rest = rest[len(str(n)):]
        op = rest[:n]
        u = int(re.match(r"Li(\d+)E", rest[n:]).group(1))
        return kind, op, u
    u = int(re.match(r"Li(\d+)E", rest).group(1))
    return kind, None, u


def widest_window(k):
    ws = sc.clock_windows(k)
    return max(ws, key=len) if ws else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v1", "probe": "alu3", "sass": a.sass,
           "kernels": {}, "unclassified": []}
    for name, k in kernels.items():
        d = demangle(name)
        if d is None:
            out["unclassified"].append(name)
            continue
        kind, op, u = d
        if kind == "k_lat":
            variant, mode, expect, want = f"lat.{OPS[op][0]}", "lat", OPS[op][1], u * OPS[op][2]
        elif kind == "k_thr":
            variant, mode, expect, want = f"thr.{OPS[op][0]}", "thr", OPS[op][1], 8 * u * OPS[op][2]
        elif kind == "k_mma_f16":
            variant, mode, expect, want = "lat.mma.HMMA", "lat", "HMMA", u
        elif kind == "k_mma_f16_thr":
            variant, mode, expect, want = "thr.mma.HMMA", "thr", "HMMA", 4 * u
        else:
            variant, mode, expect, want = "lat.mma.IMMA", "lat", "IMMA", u
        w = widest_window(k)
        res = {"key": f"alu3/{variant}/{u}", "op": variant, "mode": mode, "u": u,
               "kernel": name, "expected_sass": [expect], "ok": False, "why": []}
        if not w:
            res["why"].append("no clock window")
            out["kernels"][res["key"]] = res
            continue
        hits = [i for i in w if i.opcode == expect or i.opcode.startswith(expect + ".")]
        # MOV cells: the op itself is MOV, so MOV is not an allowed extra there
        allowed = ALLOWED - ({"MOV"} if expect == "MOV" else set())
        other = [i.opcode for i in w if i not in hits and i.root not in allowed]
        counts = Counter(i.opcode for i in w)
        res.update({"sass_counts": dict(counts), "n_hits": len(hits), "want": want,
                    "stall_sum": sum(i.stall for i in w), "n_other": len(other),
                    "window_pcs": [hex(w[0].addr), hex(w[-1].addr)]})
        if len(hits) != want:
            res["why"].append(f"{expect} count {len(hits)} != {want}")
        if other:
            res["why"].append(f"other instructions in the window: {sorted(set(other))}")
        res["ok"] = not res["why"]
        out["kernels"][res["key"]] = res
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = sum(1 for r in out["kernels"].values() if not r["ok"])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check alu3: {out['n_ok']} ok, {out['n_fail']} fail, "
          f"{len(out['unclassified'])} unclassified ({len(kernels)} kernels)")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"])
    return 0 if out["kernels"] else 2


if __name__ == "__main__":
    sys.exit(main())
