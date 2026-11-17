#!/usr/bin/env python3
"""Protocol v3 per-cell probe sim summary (protocol-v3.json
engine_runs.per_cell_quantities).

Reads the `cyclescope run` output of the 15 engine cells (--sim is the
run's sim/ root: <run out>/sim/<cell>/out-cpp/{result.bin,journal.bin,
segments/...}; run each cell with `cyclescope run --ledgers --keep` so both
journal.bin and segments/ survive) and writes probe-sim.tsv:

    cell  status  issued  completed  cycles  lifecycle_envelope
    core_sim  cs2r_first_min  cs2r_second_max  span_sim  n_warps
    tree_digest  engine

lifecycle_envelope = TELE group 2 counter kTele2LifecycleEnvelope (3) of
result.bin; core_sim = cycles - envelope.  cs2r_* come from journal.bin:
instance -> pc via the segment INST/CODE tables (job-level instance
numbers accumulate INST row counts across .seg files; pc offsets are
relative to the lowest CODE address), the two CS2R ... SR_CLOCKLO pcs
taken from the traced kernel's functions/<fid>.tsv SASS dump.
cs2r_first_min = min over warps of the first CS2R's issue cycle,
cs2r_second_max = max over warps of the second's, span_sim = difference.

Also dumps warp 0's journal (pc, SASS, issue cycle, wait) per cell to
<sim>/<cell>/journal-w0.txt.

    PYTHONPATH=<repo>/python python3 probe_sim.py \
        --sim <sim root> --captures <capture root> \
        --tree <tree.json> --engine <cyclescope-engine> --out <tsv>
"""
from __future__ import annotations

import argparse
import glob
import json
import struct
import sys
from pathlib import Path

from cyclescope.io.file_header import FileReader
from cyclescope.job.result import read_journal, read_result

WS = (1, 4, 8, 16, 32)
LS = (0, 1024, 4096)
TELE2_LIFECYCLE_ENVELOPE = 3  # cpp/core/simulator.hpp kTele2LifecycleEnvelope
BUCKETS = {0: "none", 1: "register", 2: "resource", 3: "semantic",
           4: "collective", 5: "barrier", 6: "fetch", 7: "scheduler",
           8: "memory", 9: "arch", 10: "other", 0xFFFFFFFF: "none"}


def pcmap(seg_dir: Path) -> dict[int, int]:
    """instance (job-level) -> code address, across .seg files."""
    code = {}
    off = 0
    for sp in sorted(glob.glob(str(seg_dir / "segments" / "*.seg"))):
        f = FileReader(Path(sp).read_bytes())
        iv = f.require_table("INST")
        cv = f.require_table("CODE")
        for i in range(0, len(cv), 16):
            inst, _pad, addr = struct.unpack_from("<IIQ", cv, i)
            code[off + inst] = addr
        off += len(iv) // 16
    return code


def function_sass(case_dir: Path) -> tuple[dict[int, str], int]:
    """addr -> SASS text for the traced launch's function; (map, fid)."""
    rows = [json.loads(x) for x in
            (case_dir / "launches.jsonl").read_text().splitlines()
            if x.strip()]
    traced = [r for r in rows if r.get("traced")]
    fid = traced[0]["function_id"] if traced else None
    out = {}
    tsvs = [case_dir / "functions" / f"{fid}.tsv"] if fid is not None \
        else sorted((case_dir / "functions").glob("*.tsv"))
    for tsv in tsvs:
        if not tsv.is_file():
            continue
        for line in tsv.read_text().splitlines():
            p = line.split("\t")
            if len(p) >= 4:
                try:
                    out[int(p[1], 16)] = p[3].strip()
                except ValueError:
                    pass
        if out:
            break
    return out, fid


def envelope(result: dict) -> int:
    for row in result["telemetry"]:
        if row["group"] == 2 and row["counter"] == TELE2_LIFECYCLE_ENVELOPE:
            return row["value"]
    return 0


def tree_digest(path: Path) -> str:
    tree = json.loads(path.read_text())
    d = tree.get("content_digest")
    if d:
        return d
    from cyclescope.tree.validate import compute_content_digest
    return compute_content_digest(tree)


def one_cell(cell: str, sim_root: Path, cap_root: Path) -> tuple[dict, str]:
    out = {"cell": cell, "status": "ok", "issued": 0, "completed": 0,
           "cycles": 0, "lifecycle_envelope": 0, "core_sim": 0,
           "cs2r_first_min": "", "cs2r_second_max": "", "span_sim": "",
           "n_warps": 0}
    case_out = sim_root / cell / "out-cpp"
    res_path = case_out / "result.bin"
    jnl_path = case_out / "journal.bin"
    w0_lines = []
    if not res_path.is_file():
        out["status"] = "no_result"
        return out, ""
    r = read_result(res_path.read_bytes())
    env = envelope(r)
    out.update(issued=r["issued"], completed=r["completed"],
               cycles=r["cycles"], lifecycle_envelope=env,
               core_sim=r["cycles"] - env)
    if not jnl_path.is_file():
        out["status"] = "no_journal"
        return out, ""
    code = pcmap(case_out)
    sass, fid = function_sass(cap_root / cell)
    base = min(code.values()) if code else 0
    cs2r = sorted(a for a, s in sass.items()
                  if "CS2R" in s and "SR_CLOCKLO" in s)
    journal = read_journal(jnl_path.read_bytes())["JRNL"]
    out["n_warps"] = len({e["warp"] for e in journal})
    if len(cs2r) >= 2 and code:
        a0, a1 = cs2r[0], cs2r[1]
        first = {}
        second = {}
        for e in journal:
            off = code.get(e["instance"], 0) - base
            w = e["warp"]
            if off == a0:
                first[w] = min(first.get(w, 1 << 62), e["issue_cycle"])
            elif off == a1:
                second[w] = max(second.get(w, -1), e["issue_cycle"])
        if first and second:
            out["cs2r_first_min"] = min(first.values())
            out["cs2r_second_max"] = max(second.values())
            out["span_sim"] = out["cs2r_second_max"] - out["cs2r_first_min"]
        else:
            out["status"] = "no_cs2r_in_journal"
    else:
        out["status"] = f"cs2r_pcs={len(cs2r)} code={len(code)}"
    for e in journal:
        if e["warp"] != 0:
            continue
        off = code.get(e["instance"], 0) - base
        w0_lines.append(
            f"0x{off:05x}\t{sass.get(off, '?'):48s}\twarp={e['warp']}"
            f"\tissue={e['issue_cycle']}"
            f"\twait={BUCKETS.get(e['wait_bucket'], e['wait_bucket'])}"
            f"+{e['wait_cycles']}")
    return out, "\n".join(w0_lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    digest = tree_digest(args.tree)
    header = ["cell", "status", "issued", "completed", "cycles",
              "lifecycle_envelope", "core_sim", "cs2r_first_min",
              "cs2r_second_max", "span_sim", "n_warps", "tree_digest",
              "engine"]
    lines = ["\t".join(header)]
    bad = 0
    for w in WS:
        for l in LS:
            cell = f"life.c1.w{w}.s0.l{l}"
            row, w0 = one_cell(cell, args.sim, args.captures)
            if row["status"] != "ok" or row["issued"] != row["completed"]:
                bad += 1
            if w0:
                d = args.sim / cell
                d.mkdir(parents=True, exist_ok=True)
                (d / "journal-w0.txt").write_text(w0)
            lines.append("\t".join(str(row[k]) for k in header[:-2])
                         + f"\t{digest}\t{args.engine}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"probe_sim: {15 - bad}/15 ok -> {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
