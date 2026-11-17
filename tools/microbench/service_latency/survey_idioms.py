#!/usr/bin/env python3
"""survey_idioms.py — decompose the clock window of every idioms.cu kernel.

Reads cuobjdump -sass output (or an already-dumped .sass file), finds the
widest window between two clock reads per kernel, counts instructions by
full opcode (with modifiers), and reports the per-step composition
(count / U, or /K*8 for thr kernels). Anything not an exact multiple is
flagged.
"""
import json
import re
import subprocess
import sys

_U = 8          # lat-chain steps per window
_K, _CH = 2, 8  # thr: K iters x 8 chains; issue: K instances

_FUNC = re.compile(r"Function\s*:\s*(\S+)")
_ADDR = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")

def _demangle(names):
    try:
        out = subprocess.run(["c++filt"] + names, capture_output=True,
                             text=True, check=True).stdout.splitlines()
        return dict(zip(names, out))
    except Exception:
        return {n: n for n in names}

def _cells(dem):
    """k_<harness><Op, N> -> (harness, cell, steps)."""
    m = re.search(r"k_(lat|ulat|plat|thr|dthr|dlat|issue)<([A-Za-z0-9_]+),\s*(\d+)>", dem)
    if not m:
        return None
    harness, cell, n = m.group(1), m.group(2), int(m.group(3))
    steps = {"lat": n, "ulat": n, "plat": n,
             "thr": n * _CH, "dthr": n * _CH, "dlat": 2 * n, "issue": n}[harness]
    return harness, cell, steps

def parse(text):
    kernels = {}   # func -> [ (addr, text) ]
    func = None
    for line in text.splitlines():
        f = _FUNC.search(line)
        if f:
            func = f.group(1); kernels[func] = []
            continue
        a = _ADDR.search(line)
        if a and func:
            kernels[func].append((int(a.group(1), 16), a.group(2).strip()))
    dem = _demangle(list(kernels))
    out = {}
    for fn, insns in kernels.items():
        info = _cells(dem[fn])
        if info is None:
            continue
        harness, cell, steps = info
        # clock reads: CS2R x, SR_CLOCKLO/HI / SRZ / CS2R.*CLK pairs
        clock_addrs = [a for a, t in insns
                       if t.startswith("CS2R") or "SR_CLOCK" in t]
        # window = widest gap between consecutive clock instructions
        pairs = []
        for i in range(len(clock_addrs) - 1):
            pairs.append((clock_addrs[i], clock_addrs[i + 1]))
        if not pairs:
            out[fn] = {"harness": harness, "cell": cell, "steps": steps,
                       "error": "no clock pair"}
            continue
        lo, hi = max(pairs, key=lambda p: p[1] - p[0])
        win = [t for a, t in insns if lo < a < hi]
        counts = {}
        for t in win:
            op = t.split()[0].rstrip(",")
            counts[op] = counts.get(op, 0) + 1
        per, rem = {}, {}
        for op, c in sorted(counts.items()):
            if c % steps == 0:
                per[op] = c // steps
            else:
                rem[op] = c
        out[fn] = {"harness": harness, "cell": cell, "steps": steps,
                   "window_insns": len(win), "per_step": per,
                   "remainder": rem}
    return out

def main():
    src = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else "-"
    if src.endswith(".cubin") or not src.endswith((".sass", ".txt")):
        text = subprocess.run(["cuobjdump", "-sass", src],
                              capture_output=True, text=True,
                              check=True).stdout
    else:
        text = open(src).read()
    res = parse(text)
    by_cell = {}
    for fn, r in sorted(res.items(), key=lambda kv: kv[1].get("cell", "")):
        by_cell.setdefault(r["harness"], []).append(r)
    lines = []
    for harness in ("lat", "ulat", "plat", "thr", "dthr", "issue"):
        for r in by_cell.get(harness, []):
            if "error" in r:
                lines.append(f"{r['harness']:5} {r['cell']:24} ERROR {r['error']}")
                continue
            comp = ", ".join(f"{o}x{c}" for o, c in r["per_step"].items())
            rem = (" REM=" + ",".join(f"{o}x{c}" for o, c in
                                       r["remainder"].items())
                   if r["remainder"] else "")
            lines.append(f"{r['harness']:5} {r['cell']:24} "
                         f"[{r['window_insns']:3}/{r['steps']:2}] {comp}{rem}")
    report = "\n".join(lines)
    js = json.dumps(res, indent=1, sort_keys=True)
    if out_json == "-":
        print(report)
    else:
        open(out_json, "w").write(js)
        open(out_json.replace(".json", ".txt"), "w").write(report + "\n")
        print(report)

if __name__ == "__main__":
    main()
