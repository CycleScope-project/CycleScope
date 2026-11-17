#!/usr/bin/env python3
"""Protocol v3 per-cell reference TSV for `cyclescope run --reference-tsv`.
One row per cell; convert matches the row whose `case` column equals
the capture dir name (cyclescope/convert/capture_pipeline.py).

Per cell, over kept launches (launch 0 and 1 dropped) of run-a:
    sm_clock_khz  = median sm__cycles_elapsed.avg.per_second / 1000
    cycles_median = median sm__cycles_active.max
cache_control all, clock_control none, status success,
source `lifecycle v3 run-a`.

    python3 probe_reference.py --run-a <dir> --out <tsv>
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import statistics
from pathlib import Path

LAUNCH_RE = re.compile(
    r"^LAUNCH k_life variant=life\.c(\d+)\.w(\d+)\.s(\d+)\.l(\d+)"
    r" launch=(\d+) smid=")
DROP = {0, 1}
WS = (1, 4, 8, 16, 32)
LS = (0, 1024, 4096)

HEADER = ["case", "kernel", "launch_ordinal", "grid", "block", "cluster",
          "cycles_median", "cycles_min", "cycles_max", "cycles_active_avg",
          "sm_clock_khz", "sm_clock_source", "cache_control",
          "clock_control", "repeats", "status", "source"]


def launches(path: Path) -> list[tuple[str, int]]:
    out = []
    for line in path.read_text().splitlines():
        m = LAUNCH_RE.match(line.strip())
        if m:
            c, w, s, l, n = map(int, m.groups())
            out.append((f"life.c{c}.w{w}.s{s}.l{l}", n))
    return out


def counter_rows(path: Path) -> list[dict]:
    text = path.read_text()
    start = re.search(r'^(?:"ID"|ID),', text, re.MULTILINE)
    if not start:
        return []
    rows = list(csv.reader(io.StringIO(text[start.start():])))
    names = rows[0]
    out = []
    for r in rows[2:]:
        if len(r) == len(names) and r[0].isdigit():
            out.append(dict(zip(names, r)))
    return out


def num(row: dict, key: str) -> float:
    return float((row.get(key) or "0").replace(",", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    seq = launches(args.run_a / "launches.log")
    rows = counter_rows(args.run_a / "counters.csv")
    if len(seq) != len(rows):
        print(f"probe_reference: {len(seq)} launches vs "
              f"{len(rows)} counter rows")
        return 2
    per_cell: dict[str, list[tuple[float, float]]] = {}
    for (cell, n), row in zip(seq, rows):
        if n in DROP:
            continue
        per_cell.setdefault(cell, []).append(
            (num(row, "sm__cycles_active.max"),
             num(row, "sm__cycles_elapsed.avg.per_second")))
    lines = ["\t".join(HEADER)]
    missing = []
    for w in WS:
        for l in LS:
            cell = f"life.c1.w{w}.s0.l{l}"
            vals = per_cell.get(cell)
            if not vals:
                missing.append(cell)
                continue
            act = [v[0] for v in vals]
            clk = [v[1] / 1000.0 for v in vals]
            lines.append("\t".join([
                cell, f"_Z6k_lifeILi{l}EEvPmf", "3",
                "1x1x1", f"{32 * w}x1x1", "1x1x1",
                str(int(statistics.median(act))),
                str(int(min(act))), str(int(max(act))), "",
                str(int(statistics.median(clk))),
                "ncu", "all", "none", str(len(vals)),
                "success", "lifecycle v3 run-a"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    if missing:
        print(f"probe_reference: missing cells {missing}")
        return 2
    print(f"probe_reference: {len(lines) - 1} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
