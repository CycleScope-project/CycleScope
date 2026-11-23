#!/usr/bin/env python3
"""Resolve every suite case and dump the concrete argv/cwd/env.

Used to self-check capture reproducibility on a new machine: run with
``PYTHONPATH=<repo>/python`` and ``CYCLESCOPE_ARTIFACT_ROOT`` set; emits one
JSON object per case (suite, case, argv, cwd, env values that look like
paths, resolved existence flags).  Resolution is pure — nothing is
executed, missing files only lower ``exists`` markers.

    python3 scripts/pipeline/resolve_cases.py [--suites hp-fg,bw-fg] [--out out.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from cyclescope.capture.driver import (  # noqa: E402
    REPO as DRIVER_REPO, case_command, resolve_case, tracer_env)

SUITES = {
    "hp-fg": "hopper/full_gpu",
    "bw-fg": "blackwell/full_gpu",
}

PATH_CHARS = frozenset("/")


def _pathish(v) -> bool:
    return isinstance(v, str) and "/" in v and not v.startswith("-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--out", default="-")
    ap.add_argument("--cases", default=None,
                    help="comma-separated case ids to limit to")
    args = ap.parse_args()
    out = (open(args.out, "w") if args.out != "-" else sys.stdout)
    limit = set(args.cases.split(",")) if args.cases else None
    fake_tracer = Path("/nonexistent/tracer.so")
    for short in args.suites.split(","):
        suite = SUITES[short]
        arch, sset = suite.split("/", 1)
        import csv
        cases_tsv = DRIVER_REPO / "suites" / arch / sset / "cases.tsv"
        if arch == "hopper" or cases_tsv.is_file():
            rows = [r["case_id"] for r in csv.DictReader(
                cases_tsv.open(), delimiter="\t")]
        else:
            rows = [l.split()[-1] for l in
                    (DRIVER_REPO / "suites" / arch / sset / "cases.txt")
                    .read_text().splitlines() if l.strip()]
        for case_id in rows:
            if limit and case_id not in limit:
                continue
            rec = {"suite": suite, "set": short, "case": case_id}
            try:
                res = resolve_case(suite, case_id, Path("/tmp/capture"))
                argv, run_env = case_command(res, Path("/CAPTURE_DIR"),
                                             dict(res.env))
                run_env = tracer_env(run_env, fake_tracer,
                                     Path("/CAPTURE_DIR"), "k@1")
                rec["argv"] = argv
                rec["cwd"] = str(res.cwd or "")
                rec["env"] = {k: v for k, v in run_env.items()
                              if _pathish(v)}
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            out.write(json.dumps(rec, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
