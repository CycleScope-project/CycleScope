#!/usr/bin/env python3
"""Stage27 MemLat per-shard point runner (crossed grid).

Every shard runs every probe point serially on its single GPU (stage27
gpu_camp convention), so each quantity gets 4 independent GPUs of
replication.  Raw per-sample values land in one jsonl per binary; per-point
rows are appended with shard tags.  Fail-closed per point: failures are
recorded and make the shard exit nonzero, never silently dropped.  For the
ops binary, ops whose SASS qualification failed are skipped with a recorded
reason (the gate already failed the job for `mem`-kind expectations).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def op_qualified(bin_dir: Path) -> dict:
    """Read the ops sass_check verdict produced by the job script."""
    p = bin_dir.parent / "stage27_memlat_ops.sass_check.json"
    try:
        rep = json.loads(p.read_text())
        return rep.get("op_qualification", {})
    except Exception:
        return {}


def grid(qual: dict) -> list[dict]:
    pts: list[dict] = []
    B = "{bin}"

    def add(name: str, binname: str, args: list[str], needs_op: str | None = None):
        if needs_op and qual and not qual.get(needs_op, False):
            pts.append({"name": name, "bin": binname, "args": None,
                        "skip": f"sass gate: op {needs_op} unqualified"})
            return
        pts.append({"name": name, "bin": binname, "args": args})

    # A: dram chain + width + MLP curve (P0-1)
    add("dram.chain.cg", "stage27_memlat_memory",
        ["--mode", "dram", "--variant", "0", "--samples", "32", "--warmup", "4"])
    add("dram.chain.cv", "stage27_memlat_memory",
        ["--mode", "dram", "--variant", "1", "--samples", "32", "--warmup", "4"])
    for w in (32, 64, 128):
        add(f"dram.width.{w}", "stage27_memlat_memory",
            ["--mode", "dram_width", "--width", str(w), "--samples", "16",
             "--warmup", "4"])
    for c in (1, 2, 4, 8, 16, 32):
        add(f"dram.mlp.{c}", "stage27_memlat_memory",
            ["--mode", "dram_mlp", "--chains", str(c), "--samples", "16",
             "--warmup", "4"])

    # B: L2 improved residency + L1 sanity (P1-4)
    add("l2.chain", "stage27_memlat_memory",
        ["--mode", "l2", "--samples", "64", "--warmup", "8"])
    add("l1.chain", "stage27_memlat_memory",
        ["--mode", "l1", "--samples", "32", "--warmup", "4"])

    # C: per-opcode chains (P0-2); depth sweep 4/16 for constancy
    for op in ("fadd", "fmul", "ffma", "imad", "lop3", "shf"):
        for d in (4, 16):
            add(f"ops.{op}.d{d}", "stage27_memlat_ops",
                ["--op", op, "--depth", str(d), "--rounds", "8", "--launches", "8"],
                needs_op=op)
    for op in ("mov", "mufu", "ex2", "prmt", "iadd3"):
        for d in (4, 16):
            add(f"ops.{op}.d{d}", "stage27_memlat_ops",
                ["--op", op, "--depth", str(d), "--rounds", "8", "--launches", "8"],
                needs_op=op)
    add("ops.sel.d16", "stage27_memlat_ops",
        ["--op", "sel", "--depth", "16", "--rounds", "8", "--launches", "8"],
        needs_op="sel")
    add("ops.isetp.d16", "stage27_memlat_ops",
        ["--op", "isetp", "--depth", "16", "--rounds", "8", "--launches", "8"],
        needs_op="isetp")
    # context: memory-op chains (not installed to instruction.service)
    add("ops.ldg.d16", "stage27_memlat_ops",
        ["--op", "ldg", "--depth", "16", "--rounds", "8", "--launches", "8"],
        needs_op="ldg")
    add("ops.lds.d16", "stage27_memlat_ops",
        ["--op", "lds", "--depth", "16", "--rounds", "8", "--launches", "8"],
        needs_op="lds")
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qual = op_qualified(Path(args.bin_dir))
    rows = []
    failed = []
    t_start = time.time()
    for pt in grid(qual):
        jsonl = out / f"{pt['bin']}.jsonl"
        if pt.get("args") is None:
            rows.append({"point": pt["name"], "shard": args.shard, "ok": False,
                         "skipped": pt["skip"], "rc": None, "seconds": 0.0})
            print(f"[{args.shard}] {pt['name']:32s} SKIPPED {pt['skip']}", flush=True)
            continue
        binp = Path(args.bin_dir) / pt["bin"]
        cmd = [str(binp)] + pt["args"] + ["--out", str(jsonl)]
        t0 = time.time()
        env = dict(os.environ)
        env.setdefault("CUDA_MODULE_LOADING", "EAGER")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0
        ok = r.returncode == 0 and jsonl.exists()
        rows.append({
            "point": pt["name"], "shard": args.shard, "ok": ok,
            "rc": r.returncode, "seconds": round(dt, 3),
            "stderr_tail": (r.stderr or "")[-400:],
        })
        print(f"[{args.shard}] {pt['name']:32s} rc={r.returncode} {dt:.1f}s",
              flush=True)
        if not ok:
            failed.append(pt["name"])
    summary = {
        "schema": "blackwell.stage27.shard_summary.v1",
        "line": "stage27.memlat",
        "shard": args.shard,
        "points_total": len(rows),
        "points_failed": len(failed),
        "points_skipped": sum(1 for r in rows if r.get("skipped")),
        "failed_points": failed,
        "wall_seconds": round(time.time() - t_start, 1),
        "rows": rows,
    }
    (out / f"shard_summary_{args.shard}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"[{args.shard}] summary: {len(rows)} points, {len(failed)} failed, "
          f"{summary['points_skipped']} skipped, {summary['wall_seconds']}s",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
