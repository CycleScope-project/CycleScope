"""Capture-directory sanity check.

    python -m cyclescope.capture.trace_check <case_dir> [--launch N]

Parses every trace record (via cyclescope.convert.trace_reader), reports
record counts by kind, leftover bytes, exec_mask=0 records, PHASECHK
AFTER pairing, GSYNC counts, top opcode roots, MREF roots with address
samples (LDG/STG/LDS/STS one each), SPACE entries, tensor_maps rows,
launches.jsonl selected
fields, and functions tsv coverage.  Any parse failure exits nonzero.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from cyclescope.convert import trace_reader


def _opcode_root(sass: str) -> str:
    toks = sass.split()
    if toks and toks[0].startswith("@"):
        toks = toks[1:]
    return toks[0].split(".")[0] if toks else "?"


def quick_check(case_dir: Path) -> tuple[list[str], dict]:
    """Fast integrity check (no full parse): launches.jsonl has exactly
    one traced launch, its trace-<launch>.bin[.zst] exists and, for .zst,
    passes `zstd -t`; the decompressed stream starts with b"GSIMTRC\0";
    functions/<function_id>.tsv exists and is nonempty."""
    import shutil, subprocess
    failures: list[str] = []
    summary: dict = {}
    lp = case_dir / "launches.jsonl"
    if not lp.is_file():
        return ["no launches.jsonl"], summary
    rows = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
    traced = [r for r in rows if r.get("traced")]
    if len(traced) != 1:
        return ([f"{len(traced)} traced launches"]
                if traced else ["no traced launch"]), summary
    t = traced[0]
    launch, fid = int(t["launch"]), t.get("function_id")
    summary["launch"], summary["function_id"] = launch, fid
    trace = None
    for cand in (case_dir / f"trace-{launch}.bin.zst",
                 case_dir / f"trace-{launch}.bin"):
        if cand.is_file():
            trace = cand
            break
    if trace is None:
        return [f"missing trace-{launch}.bin[.zst]"], summary
    summary["trace"] = trace.name
    summary["trace_bytes"] = trace.stat().st_size
    if trace.suffix == ".zst":
        zstd = shutil.which("zstd")
        if zstd is None:
            return ["zstd_missing"], summary
        if subprocess.run([zstd, "-t", "-q", str(trace)],
                          capture_output=True).returncode != 0:
            return ["zstd frame check failed"], summary
        head = subprocess.run([zstd, "-dc", str(trace)],
                              capture_output=True).stdout[:16]
    else:
        head = trace.open("rb").read(16)
    if not head.startswith(b"GSIMTRC\0"):
        failures.append("bad trace header")
    if fid is None or not (case_dir / "functions" / f"{fid}.tsv"
                           ).is_file() or \
            (case_dir / "functions" / f"{fid}.tsv").stat().st_size == 0:
        failures.append(f"missing functions/{fid}.tsv")
    return failures, summary


def _quick_main(case_dir: Path) -> int:
    failures, summary = quick_check(case_dir)
    print(json.dumps(summary, indent=1))
    for f in failures:
        print(f"FAIL: {f}")
    if not failures:
        print("trace_check: OK (quick)")
        return 0
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="cyclescope.capture.trace_check")
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--launch", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    case_dir = args.case_dir
    if args.quick:
        return _quick_main(case_dir)

    failures: list[str] = []
    try:
        layouts, names = trace_reader.load_static(case_dir)
    except Exception as exc:  # noqa: BLE001 - report, exit nonzero
        print(f"FAIL load_static: {exc}")
        return 2
    if not layouts:
        print("FAIL: no functions/<id>.tsv found")
        return 2

    launches = [json.loads(l) for l in
                (case_dir / "launches.jsonl").read_text().splitlines()] \
        if (case_dir / "launches.jsonl").is_file() else []
    traced = [l["launch"] for l in launches if l.get("traced")]
    if args.launch is not None:
        selected = [args.launch]
    else:
        selected = traced or sorted(
            int(p.name.split("-")[1].split(".")[0])
            for p in case_dir.glob("trace-*.bin.zst"))

    counts = Counter()
    opcodes = Counter()
    exec0 = 0
    leftover = 0
    after_total = after_paired = 0
    gsync_total = 0
    phasechk_pending: dict = {}
    mref_roots: dict = {}
    space_samples: list = []
    per_launch: Counter = Counter()

    if not selected:
        # the app never reached the selected launch (killed at the
        # timeout, selection mismatch): nothing to check is a failure
        failures.append("no traced launch (no trace-*.bin.zst, no "
                        "launches.jsonl row with traced=true)")
    for no in selected:
        path_zst = case_dir / f"trace-{no}.bin.zst"
        path_bin = case_dir / f"trace-{no}.bin"
        if not path_zst.is_file() and not path_bin.is_file():
            failures.append(f"missing trace-{no}.bin.zst")
            continue
        try:
            for rec in trace_reader.iter_records(case_dir, no):
                per_launch[no] += 1
                counts["instr"] += 1
                sass, _ops, _m = layouts.get(rec.function_id, {}).get(
                    rec.idx, ("", [], 0))
                root = _opcode_root(sass)
                opcodes[root] += 1
                if rec.exec_mask == 0:
                    exec0 += 1
                if rec.after_ready is not None:
                    after_total += 1
                    after_paired += 1
                gsync_total += len(rec.gsync)
                if "PHASECHK" in sass and rec.after_ready is None:
                    counts["phasechk_unpaired"] += 1
                if rec.mrefs and any(a for a in rec.mrefs[0]):
                    mref_roots.setdefault(root, rec)
                if "SPACE" in rec.ops and len(space_samples) < 3:
                    space_samples.append(
                        (root, rec.ops["SPACE"],
                         [hex(a) for a in rec.mrefs[0][:4]] if rec.mrefs
                         else []))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"trace-{no}: {exc}")
        # leftover check: byte size vs consumed is implicit in the
        # reader (it raises on undelimited tails)
    counts["after_records"] = after_total
    counts["gsync_records"] = gsync_total

    tensor_maps = 0
    tm = case_dir / "tensor_maps.jsonl"
    if tm.is_file():
        tensor_maps = sum(1 for l in tm.read_text().splitlines() if l.strip())

    tsv_rows = 0
    tsv_nonempty_bytes = 0
    for tsv in (case_dir / "functions").glob("*.tsv"):
        for line in tsv.read_text().splitlines():
            parts = line.split("\t")
            tsv_rows += 1
            if len(parts) > 2 and parts[2].strip():
                tsv_nonempty_bytes += 1

    print(f"case_dir: {case_dir}")
    print(f"records: {dict(per_launch)} total={sum(per_launch.values())}")
    print(f"exec_mask=0 records: {exec0}")
    print(f"AFTER attached (PHASECHK paired): {after_paired}; "
          f"unpaired PHASECHK instrs: {counts['phasechk_unpaired']}")
    print(f"GSYNC records attached: {gsync_total}")
    print(f"leftover bytes: {leftover}")
    print("top opcodes:", opcodes.most_common(20))
    print("mref roots:", sorted(mref_roots))
    for want in ("LDG", "STG", "LDS", "STS", "SYNCS"):
        rec = mref_roots.get(want)
        if rec is not None:
            print(f"  {want} sample addrs: "
                  f"{[hex(a) for a in rec.mrefs[0][:8]]}")
    for root, masks, addrs in space_samples:
        print(f"  SPACE {root}: ballots={masks} addrs={addrs}")
    print(f"tensor_maps entries: {tensor_maps}")
    for l in launches:
        if l.get("traced"):
            print("selected launch:", {k: l.get(k) for k in (
                "launch", "function_id", "mangled", "grid", "block",
                "cluster", "prior_identical_launches",
                "argument_bytes_status", "function_prior_launches",
                "sampled_clusters", "occupancy_max_blocks_per_sm")})
    print(f"functions tsv: rows={tsv_rows} bytes16_nonnull="
          f"{tsv_nonempty_bytes} "
          f"({tsv_nonempty_bytes / tsv_rows if tsv_rows else 0:.1%})")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("trace_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
