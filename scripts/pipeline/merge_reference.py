#!/usr/bin/env python3
"""Merge per-case `cyclescope reference` tables into one suite reference table.

    merge_reference.py --captures <dir> --cases <list> --out <reference.tsv>
        [--align-to <capture root>]

<dir>/reference/<case>/reference.tsv (one row per case, written by
`cyclescope reference`) are concatenated in <list> order under the shared
header.  Cases without a table, or whose row is not `status=success`, are
listed on stderr and left out; the exit code is 1 when any are missing so
a partial batch cannot silently become the suite reference.

--align-to <root> makes every row time the launch <root>/<case> traced:
when the reference picked another launch, the row is rebuilt from the
case's recorded ncu repeats for the traced launch (same kernel name and
occurrence; no new measurement).  A case whose capture is missing, or
whose traced launch is absent from any recorded repeat, is left out and
listed like a missing table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from cyclescope.capture.launch_selection import traced_launch_key  # noqa: E402
from cyclescope.reference.ncu_runner import (TIMING_COLUMNS,  # noqa: E402
                                         realigned_row,
                                         reference_launch_key)


def case_ids(spec: str) -> list[str]:
    p = Path(spec)
    if p.suffix == ".tsv":
        return [r["case_id"] for r in
                csv.DictReader(p.open(), delimiter="\t")]
    return [l.split()[-1] for l in p.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--cases", required=True,
                    help="cases.tsv (case_id column) or a one-id-per-line file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--align-to", type=Path, default=None,
                    help="capture root (<root>/<case>): rows time the traced "
                         "launch")
    args = ap.parse_args()

    header, rows, missing, bad = None, [], [], []
    realigned, unaligned = [], []
    for cid in case_ids(args.cases):
        tsv = args.captures / "reference" / cid / "reference.tsv"
        if not tsv.is_file():
            missing.append(cid)
            continue
        lines = tsv.read_text().splitlines()
        hdr = lines[0].split("\t")
        header = header or hdr
        if hdr != header:
            print(f"error: header mismatch in {tsv}", file=sys.stderr)
            return 2
        row = next((l.split("\t") for l in lines[1:]
                    if l.split("\t")[0] == cid), None)
        if row is None or row[header.index("status")] != "success":
            bad.append((cid, row[header.index("status")] if row else "no_row"))
            continue
        if args.align_to is not None:
            if header != TIMING_COLUMNS:
                print(f"error: {tsv} header is not TIMING_COLUMNS",
                      file=sys.stderr)
                return 2
            try:
                key = traced_launch_key(args.align_to / cid)
            except (OSError, ValueError) as e:
                unaligned.append((cid, f"no traced launch: {e}"))
                continue
            case_dir = args.captures / "reference" / cid / cid
            if reference_launch_key(case_dir) != key:
                new = realigned_row(case_dir, cid, *key)
                if new is None:
                    unaligned.append((cid, f"{key[0]}@{key[1]} not in every "
                                           f"recorded repeat"))
                    continue
                realigned.append((cid, row[header.index("launch_ordinal")],
                                  str(new[header.index("launch_ordinal")]),
                                  row[header.index("cycles_median")],
                                  str(new[header.index("cycles_median")])))
                row = [str(c) for c in new]
        rows.append(row)

    if header is None:
        print("error: no reference tables found", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    cc = {r[header.index("cache_control")] for r in rows}
    print(f"{args.out}: {len(rows)} rows, cache_control={sorted(cc)}, "
          f"missing={len(missing)}, not_success={len(bad)}"
          + (f", realigned={len(realigned)}, unaligned={len(unaligned)}"
             if args.align_to is not None else ""))
    for cid, o_old, o_new, c_old, c_new in realigned:
        print(f"realigned {cid} ordinal {o_old}->{o_new} "
              f"cycles_median {c_old}->{c_new}")
    for cid in missing:
        print(f"missing {cid}", file=sys.stderr)
    for cid, st in bad:
        print(f"not_success {cid} {st}", file=sys.stderr)
    for cid, why in unaligned:
        print(f"unaligned {cid} {why}", file=sys.stderr)
    return 1 if (missing or bad or unaligned) else 0


if __name__ == "__main__":
    sys.exit(main())
