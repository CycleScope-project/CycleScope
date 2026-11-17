#!/usr/bin/env python3
"""Census of tensor-pipe (wgmma) op shapes in the h100 SASS captures.

Usage: census.py <capture-root>   # e.g. $A/captures/cold4/hp-sc

Reads <capture-root>/<case>/functions/*.tsv for EVERY case (static SASS
dumps) and prints two tables:
  1) raw per-shape counts over all dumps,
  2) the same table with per-case dedup: each (case, shape) counts once
     with its intra-case total.

Key: (class, input_dtype, accumulator, ss|rs, n).

SASS operand form (verified against cold4/hp-sc):
  ss: HGMMA.64x128x16.F32.BF16 R24, gdesc[UR16].tnspA, R24 ;
      A operand is a gdesc[...] descriptor reference.
  rs: the first source operand is a register instead
      (PTX wgmma.mma_async.rs form) — detected by operand position;
      an explicit .RS suffix is also honoured when present.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import Counter

OP_RE = re.compile(
    r"\b(HGMMA|QGMMA|IGMMA)\.(\d+)x(\d+)x(\d+)"
    r"((?:\.[A-Z0-9]+)+)"
)


def classify(line: str):
    m = OP_RE.search(line)
    if not m:
        return None
    cls, _m, n, _k = m.group(1), m.group(2), m.group(3), m.group(4)
    toks = [t for t in m.group(5).split(".") if t]
    if cls == "IGMMA":
        # IGMMA.64xNx32.<dta>.<dtb>: accumulator is S32 (implied)
        acc = "S32"
        dtype = ".".join(toks) if toks else "-"
    else:
        # HGMMA/QGMMA.64xNxK.<acc>[.<dta>[.<dtb>]]
        acc = toks[0] if toks else "-"
        rest = [t for t in toks[1:] if t != acc]
        dtype = ".".join(toks[1:]) if len(toks) > 1 else "-"
        _ = rest
    tail = line[m.end():]
    ops = [o.strip() for o in tail.split(";")[0].split(",")]
    src = "ss"
    if len(ops) > 1 and "gdesc" not in ops[1]:
        src = "rs"
    if ".RS." in line.upper():
        src = "rs"
    return (cls, dtype or "-", acc, src, int(n))


def collect(root: str):
    raw: Counter = Counter()
    per_case: dict[str, Counter] = {}
    nfiles = 0
    for case_dir in sorted(glob.glob(f"{root}/*")):
        case = os.path.basename(case_dir)
        for path in glob.glob(case_dir + "/functions/*.tsv"):
            nfiles += 1
            for line in open(path, errors="replace"):
                r = classify(line)
                if r:
                    raw[r] += 1
                    per_case.setdefault(case, Counter())[r] += 1
    return raw, per_case, nfiles


def dump(title: str, table: Counter) -> None:
    print(title)
    print(f"{'class':<7} {'dtype':<14} {'acc':<4} {'src':<3} {'n':>4} count")
    for (cls, dt, acc, src, n), c in sorted(table.items()):
        print(f"{cls:<7} {dt:<14} {acc:<4} {src:<3} {n:>4} {c}")


def main() -> None:
    root = sys.argv[1]
    raw, per_case, nfiles = collect(root)
    print(f"{len(per_case)} cases, {nfiles} function dumps under {root}")
    dump("=== raw totals ===", raw)
    dedup = Counter()
    for cnt in per_case.values():
        dedup.update(cnt)
    # per-case dedup: same shape within one case counts once
    dedup2 = Counter({k: sum(1 for c in per_case.values() if k in c)
                      for k in raw})
    dump("=== cases-containing-shape (dedup) ===", dedup2)


if __name__ == "__main__":
    main()
