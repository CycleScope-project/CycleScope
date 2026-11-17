#!/usr/bin/env python3
"""SASS gate for shfl_scope v1: in every kernel the window between two
consecutive SR_CLOCKLO reads must be exactly U = 128 instructions, all
SHFL.BFLY (kernels V in {0, 2}) or all SHFL.IDX (V in {1, 3}), and
nothing else.  Usage: sass_check.py <build_dir> <checks.json out>"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

KIND = {0: "SHFL.BFLY", 1: "SHFL.IDX", 2: "SHFL.BFLY", 3: "SHFL.IDX"}


def windows(kern):
    idx = [i for i, ins in enumerate(kern.insts)
           if "SR_CLOCKLO" in ins.text]
    return [kern.insts[idx[i] + 1:idx[i + 1]]
            for i in range(len(idx) - 1)]


def main() -> int:
    build, out = sys.argv[1], sys.argv[2]
    kernels = parse(open(Path(build) / "sass" / "shfl_scope.sass").read())
    res = {}
    for name, kern in kernels.items():
        m = re.search(r"k_cellILi(\d)ELi(\d)E", name)
        if not m:
            continue
        v, n = int(m.group(1)), int(m.group(2))
        want = KIND[v]
        win = windows(kern)[0] if windows(kern) else []
        bad = [i.text.strip()[:60] for i in win
               if not i.text.strip().startswith(want)]
        key = f"{('chain.' if v >= 2 else '')}{'bfly' if v % 2 == 0 else 'idx'}.n{n}"
        res[key] = {
            "ok": len(win) == 128 and not bad,
            "u": len(win), "windows": len(windows(kern)),
            "kernel": name,
            "unexpected": bad[:6],
            "stalls": dict(Counter(i.stall for i in win)),
        }
    json.dump({"schema": "shfl_scope.sass_gate.v1",
               "probe": "shfl_scope", "kernels": res},
              open(out, "w"), indent=1)
    for k, v in sorted(res.items()):
        print(k, "OK" if v["ok"] else "FAIL", v["u"], v["unexpected"][:2])
    return 0 if res and all(v["ok"] for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
