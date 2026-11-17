#!/usr/bin/env python3
"""Protocol v3 engine-probe captures (protocol-v3.json engine_runs.capture).

Per cell life.c1.w<W>.s0.l<L> (W in {1,4,8,16,32}, L in {0,1024,4096}):
one traced run of `lifecyc --only =<cell> --launches 3 --reps 1` under the
corpus tracer, selection = the mangled k_life<L> at its third launch
(CYCLESCOPE_CAPTURE_LAUNCHES occurrence is 1-based; the model drops launches
0 and 1, so the kept launch is occurrence 3) — once in trace mode and
once in placement mode (CYCLESCOPE_CAPTURE_MODE=placement into
<cell>/placement), then trace_check --quick must pass.  The capture dir
layout is what `cyclescope convert --capture` accepts.

    PYTHONPATH=<repo>/python python3 capture_probe.py \
        --build <run dir>/build --out <run dir>/capture \
        --sm-arch sm_90a|sm_103a --tracer <tracer-*.so>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cyclescope.capture.driver import tracer_env
from cyclescope.capture.launch_selection import selection_for

WS = (1, 4, 8, 16, 32)
LS = (0, 1024, 4096)
LAUNCHES = 3
# tracer occurrences are 1-based: launches 0 and 1 are dropped by the
# model, so the captured launch is the third = occurrence 3
OCCURRENCE = 3


def mangled(l: int) -> str:
    return f"_Z6k_lifeILi{l}EEvPmf"


def cells() -> list[tuple[str, int, int]]:
    return [(f"life.c1.w{w}.s0.l{l}", w, l) for w in WS for l in LS]


def traced_run(binary: Path, cell: str, target: Path, tracer: Path,
               selection: str, mode: str, timeout_s: int) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    env = tracer_env(os.environ, tracer, target, selection)
    if mode == "placement":
        env["CYCLESCOPE_CAPTURE_MODE"] = "placement"
    argv = [str(binary), "--only", f"={cell}", "--launches", str(LAUNCHES),
            "--reps", "1", "--out", str(target)]
    (target / "run_env.json").write_text(json.dumps(
        {k: str(v) for k, v in sorted(env.items())}, indent=1))
    t0 = time.time()
    with (target / "app.log").open("w") as fh:
        fh.write("command " + json.dumps(argv) + "\n")
        rc = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            timeout=timeout_s).returncode
    return {"rc": rc, "wall_s": round(time.time() - t0, 2),
            "dir": str(target), "mode": mode}


def check_cell(case_dir: Path, cell: str, w: int, l: int) -> list[str]:
    """Exactly one traced launch: k_life<L>, block 32*W, grid 1."""
    fails = []
    rows = [json.loads(x) for x in
            (case_dir / "launches.jsonl").read_text().splitlines() if x.strip()]
    traced = [r for r in rows if r.get("traced")]
    if len(traced) != 1:
        fails.append(f"traced={len(traced)}")
    else:
        r = traced[0]
        if r.get("mangled") != mangled(l):
            fails.append(f"kernel={r.get('mangled')}")
        if r.get("block") != [32 * w, 1, 1]:
            fails.append(f"block={r.get('block')}")
        if r.get("grid") != [1, 1, 1]:
            fails.append(f"grid={r.get('grid')}")
        launch = int(r["launch"])
        if not ((case_dir / f"trace-{launch}.bin.zst").is_file()
                or (case_dir / f"trace-{launch}.bin").is_file()):
            fails.append("no_trace_file")
    if not (case_dir / "placement" / "placement.jsonl").is_file():
        fails.append("no_placement")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True,
                    help="run dir build/ containing bin/lifecyc")
    ap.add_argument("--out", type=Path, required=True,
                    help="capture root ($A/microbench/lifecycle/<dev>-v3r1/capture)")
    ap.add_argument("--sm-arch", required=True, choices=["sm_90a", "sm_103a"])
    ap.add_argument("--tracer", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    binary = args.build / "bin" / "lifecyc"
    if not binary.is_file():
        print(f"capture_probe: no binary {binary}")
        return 2
    if not args.tracer.is_file():
        print(f"capture_probe: no tracer {args.tracer}")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for cell, w, l in cells():
        case_dir = args.out / cell
        sel = selection_for(mangled(l), OCCURRENCE)
        meta = {"cell": cell, "selection": sel,
                "tracer": str(args.tracer), "argv_only": f"={cell}",
                "launches": LAUNCHES}
        tr = traced_run(binary, cell, case_dir, args.tracer, sel,
                        "trace", args.timeout)
        pl = traced_run(binary, cell, case_dir / "placement", args.tracer,
                        sel, "placement", args.timeout)
        meta.update({"run_trace": tr, "run_placement": pl})
        fails = check_cell(case_dir, cell, w, l)
        if tr["rc"] != 0:
            fails.append(f"trace_rc={tr['rc']}")
        if pl["rc"] != 0:
            fails.append(f"placement_rc={pl['rc']}")
        if not fails:
            qc = subprocess.run(
                [sys.executable, "-m", "cyclescope.capture.trace_check",
                 str(case_dir), "--quick"],
                capture_output=True, text=True)
            (case_dir / "trace_check.txt").write_text(
                qc.stdout + qc.stderr)
            if qc.returncode != 0:
                fails.append("trace_check")
        meta["status"] = "ok" if not fails else "fail:" + ",".join(fails)
        (case_dir / "probe.json").write_text(json.dumps(meta, indent=1))
        if not fails:
            (case_dir / ".done").write_text("ok\n")
        else:
            failures += 1
        print(f"{cell}: {meta['status']}", flush=True)
    print(f"capture_probe: {15 - failures}/15 cells ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
