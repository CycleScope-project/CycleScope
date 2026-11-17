#!/usr/bin/env python3
"""Collect the traced launch keys of completed captures into the suite's
fixed launch table suites/<arch>/<set>/launches.tsv.

    launch_table.py --suite <arch>/<set> --captures <root>
        [--cases <file>] [--table <path>]

For every completed capture under <root>/<case> (`.done` plus a
`trace-*.bin*`) the traced launch's (mangled, occurrence) key and its
grid/block/cluster are written as one table row; `source` records the
captures root relative to CYCLESCOPE_ARTIFACT_ROOT.  New rows are appended;
rows identical to an existing entry are skipped; a row conflicting with
an existing entry is never rewritten -- it is listed on stderr and the
exit code is 1.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from cyclescope.capture.launch_selection import (  # noqa: E402
    load_launch_table, traced_launch_key)

COLUMNS = ["case", "kernel", "occurrence", "grid", "block", "cluster",
           "source"]


def _dims(v) -> str:
    return "x".join(str(x) for x in (v or []))


def _traced_row(capture_dir: Path) -> dict | None:
    """geometry of the traced launch from launches.jsonl."""
    launches = Path(capture_dir) / "launches.jsonl"
    if not launches.is_file():
        return None
    for line in launches.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("traced"):
            return row
    return None


def _source(captures: Path) -> str:
    try:
        from cyclescope import site
        root = site.artifact_root()
    except Exception:
        root = Path("/nonexistent/artifact-root")
    try:
        rel = captures.resolve().relative_to(root.resolve())
        return f"trace:{rel}"
    except ValueError:
        return f"trace:{captures}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--captures", required=True, type=Path)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--table", type=Path)
    args = ap.parse_args()
    arch, sset = args.suite.split("/", 1)
    table = (args.table or Path(__file__).resolve().parents[2]
             / "suites" / arch / sset / "launches.tsv")
    if args.cases:
        cases = [l.strip().split()[-1] for l in
                 args.cases.read_text().splitlines() if l.strip()]
    else:
        cases = sorted(p.name for p in args.captures.iterdir()
                       if p.is_dir())
    src = _source(args.captures)
    existing = load_launch_table(table)
    rows, appended, identical, conflicts = [], 0, 0, 0
    for cid in cases:
        d = args.captures / cid
        if not (d / ".done").exists() or not glob.glob(
                str(d / "trace-*.bin*")):
            continue
        try:
            kernel, occurrence = traced_launch_key(d)
        except (OSError, ValueError) as e:
            print(f"skip {cid}: {e}", file=sys.stderr)
            continue
        traced = _traced_row(d) or {}
        row = {"case": cid, "kernel": kernel,
               "occurrence": str(occurrence),
               "grid": _dims(traced.get("grid")),
               "block": _dims(traced.get("block")),
               "cluster": _dims(traced.get("cluster")),
               "source": src}
        old = existing.get(cid)
        if old is not None:
            same = all(old.get(k) == row[k] for k in COLUMNS[1:])
            if same:
                identical += 1
            else:
                conflicts += 1
                print(f"conflict {cid}: table {old} vs capture {row}",
                      file=sys.stderr)
            row = old  # never rewrite an existing row
        else:
            appended += 1
        rows.append(row)
    for cid, old in sorted(existing.items()):
        if all(r["case"] != cid for r in rows):
            rows.append(old)
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in sorted(rows, key=lambda r: r["case"]):
            fh.write("\t".join(r[c] for c in COLUMNS) + "\n")
    print(f"appended={appended} identical={identical} "
          f"conflicts={conflicts}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
