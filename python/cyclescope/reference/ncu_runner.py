"""``cyclescope reference`` — ncu timing repeats per suite case.

For each case: resolve argv/env via the capture driver, run ncu
``repeats`` times (same metrics/switches as the capture census), pick
the target launch with the suite's selection rule, and emit
``<out>/<case>/timing.json`` plus a ``reference.tsv`` summary row.
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

from ..capture.driver import (NCU_LAUNCH_METRICS, Resolved, resolve_case,
                              _gpu_info, case_command, prepare_case)
from ..capture.launch_selection import census_row, traced_launch_key
from ..capture.ncu_tools import (CACHE_CONTROL, CLOCK_CONTROL, TIMEOUT_RC,
                                 ncu_binary as _ncu_binary, run_group)

TIMING_COLUMNS = ["case", "kernel", "launch_ordinal", "grid", "block",
                  "cluster", "cycles_median", "cycles_min", "cycles_max",
                  "cycles_active_avg", "sm_clock_khz", "sm_clock_source",
                  "cache_control", "clock_control", "repeats", "status",
                  "source"]


def _ncu_csv(text: str) -> list[dict]:
    start = text.find('"ID"')
    if start < 0:
        start = text.find("ID,")
    if start < 0:
        return []
    parsed = list(csv.reader(text[start:].splitlines()))
    if not parsed:
        return []
    hdr = parsed[0]
    return [dict(zip(hdr, r)) for r in parsed[1:]
            if len(r) == len(hdr) and str(r[0]).isdigit()]


def _num(v):
    v = (v or "").strip().replace(",", "")
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return None


def _ncu_version() -> str | None:
    try:
        out = subprocess.run([_ncu_binary(), "--version"],
                             capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return lines[-1] if lines else None


def _run_once(argv, env, cwd, log: Path, timeout_s: int,
              metrics: str | None = None) -> tuple[int, list[dict]]:
    ncu = _ncu_binary()
    names = list(NCU_LAUNCH_METRICS)
    if metrics:
        names += [m.strip() for m in metrics.split(",") if m.strip()]
    cmd = [ncu, "--csv", "--page", "raw", "--print-units", "base",
           "--print-kernel-base", "mangled", "--target-processes", "all",
           "--clock-control", CLOCK_CONTROL,
           "--cache-control", CACHE_CONTROL,
           "--replay-mode", "kernel", "--metrics",
           ",".join(names), "--log-file", str(log),
           "--", *argv]
    # app stdout/stderr next to the ncu csv (ncu_r<n>.app.log)
    with log.with_suffix(".app.log").open("w") as fh:
        rc = run_group(cmd, env=env, cwd=cwd, stdout=fh,
                       stderr=subprocess.STDOUT, timeout_s=timeout_s)
    if rc == TIMEOUT_RC:
        return rc, []
    text = log.read_text(errors="replace") if log.is_file() else ""
    return rc, _ncu_csv(text)


def _run_source(argv, env, cwd, log: Path, timeout_s: int,
                ordinal: int) -> int:
    """SourceCounters stall-sampling pass for the picked launch: one
    ncu --page source run restricted to the launch at census
    ``ordinal``; raw SASS-level csv written to log."""
    ncu = _ncu_binary()
    cmd = [ncu, "--section", "SourceCounters", "--page", "source", "--csv",
           "--print-source", "sass", "--cache-control", "all",
           "--clock-control", "none", "--target-processes", "all",
           "--launch-skip", str(ordinal), "--launch-count", "1",
           "--log-file", str(log)]
    # CYCLESCOPE_NCU_SAMPLING_INTERVAL=<0..31>: warp-stall sampling period
    # 2^(5+n) cycles (ncu --warp-sampling-interval); unset keeps ncu's auto
    # choice.  Small kernels need 0 to collect enough samples per PC.
    interval = os.environ.get("CYCLESCOPE_NCU_SAMPLING_INTERVAL")
    if interval:
        cmd += ["--warp-sampling-interval", interval]
    cmd += ["--", *argv]
    with log.with_suffix(".app.log").open("w") as fh:
        return run_group(cmd, env=env, cwd=cwd, stdout=fh,
                         stderr=subprocess.STDOUT, timeout_s=timeout_s)


def _pick(rows: list[dict], resolved: Resolved) -> dict | None:
    """Choose the target launch row by suite rule."""
    if resolved.argv_row:
        # same rule as the tracer's `regex@N`: the Nth (1-based) census
        # row whose mangled name matches kernel_regex
        matches = [r for r in rows
                   if resolved.kernel_regex is None
                   or re.search(resolved.kernel_regex, r["mangled"])]
        if not matches:
            return None
        idx = (resolved.occurrence or 1) - 1
        return matches[idx] if 0 <= idx < len(matches) else None
    from ..capture.launch_selection import (CYCLES_METRIC, demangle_many,
                                            main_kernel_matches,
                                            pick_repeat)
    if not resolved.calls:
        return None
    pick_rows = [{"mangled": r["mangled"], "ordinal": r["ordinal"],
                  CYCLES_METRIC: r.get(CYCLES_METRIC) or 0,
                  "grid": r.get("grid"), "block": r.get("block")}
                 for r in rows]
    if resolved.main_regex is not None:
        dem = demangle_many(sorted({p["mangled"] for p in pick_rows}))
        pick_rows = [p for p in pick_rows if main_kernel_matches(
            resolved.main_regex.pattern, p["mangled"],
            dem.get(p["mangled"]))]
    pick = pick_repeat(pick_rows, resolved.calls, resolved.family,
                       aux_regexes=[])
    if pick.get("error"):
        return None
    for r in rows:
        if int(r["ordinal"]) == pick["ordinal"]:
            return r
    return None


def _rows(recs: list[dict], rep: int) -> list[dict]:
    rows = []
    for i, rec in enumerate(recs):
        row = {"ordinal": i, "repeat": rep,
               "mangled": (rec.get("Kernel Name") or "").strip(),
               "grid": [_num(rec.get("launch__grid_dim_x")),
                        _num(rec.get("launch__grid_dim_y")),
                        _num(rec.get("launch__grid_dim_z"))],
               "block": [_num(rec.get("launch__block_dim_x")),
                         _num(rec.get("launch__block_dim_y")),
                         _num(rec.get("launch__block_dim_z"))],
               "cluster": [_num(rec.get("launch__cluster_dim_x")),
                           _num(rec.get("launch__cluster_dim_y")),
                           _num(rec.get("launch__cluster_dim_z"))],
               }
        for m in NCU_LAUNCH_METRICS:
            row[m] = _num(rec.get(m))
        rows.append(row)
    return rows


def _launch_stats(all_rows: list[list[dict]],
                  ordinal: int) -> tuple[list, list]:
    cycles, clocks = [], []
    for rows in all_rows:
        for r in rows:
            if r["ordinal"] == ordinal:
                if r.get("sm__cycles_active.max"):
                    cycles.append(r["sm__cycles_active.max"])
                hz = r.get("sm__cycles_elapsed.avg.per_second")
                if hz:
                    clocks.append(int(hz / 1000))
    return cycles, clocks


def _recorded_rows(case_dir: Path, repeats: int) -> list[list[dict]]:
    return [_rows(_ncu_csv((case_dir / f"ncu_r{rep}.csv").read_text(
                errors="replace")), rep) for rep in range(repeats)]


def reference_launch_key(case_dir: Path) -> tuple[str, int] | None:
    """(kernel, occurrence) of the launch a `cyclescope reference` case
    directory timed: its pick ordinal located in repeat 0's ncu rows."""
    timing = json.loads((case_dir / "timing.json").read_text())
    ordinal = (timing.get("pick") or {}).get("ordinal")
    if ordinal is None:
        return None
    rows = _recorded_rows(case_dir, 1)[0]
    if not 0 <= ordinal < len(rows):
        return None
    kernel = rows[ordinal]["mangled"]
    return kernel, sum(1 for r in rows[:ordinal + 1]
                       if r["mangled"] == kernel)


def realigned_row(case_dir: Path, cid: str, kernel: str,
                  occurrence: int) -> list | None:
    """reference.tsv row for the occurrence-th launch of kernel, from the
    case's recorded ncu repeats (no new measurement).  None when any
    repeat lacks that launch or the repeats place it at different
    ordinals."""
    timing = json.loads((case_dir / "timing.json").read_text())
    repeats = int(timing.get("repeats") or 0)
    all_rows = _recorded_rows(case_dir, repeats)
    picks = [census_row(rows, kernel, occurrence) for rows in all_rows]
    if not picks or any(p is None for p in picks) or \
            len({p["ordinal"] for p in picks}) != 1:
        return None
    cycles, clocks = _launch_stats(all_rows, picks[0]["ordinal"])
    row = _tsv_row(cid, None, picks[0], repeats,
                   "success" if cycles else "no_cycles",
                   cycles=cycles, clocks=clocks)
    row[-1] = "cyclescope reference realigned to traced launch"
    return row


def run_reference(suite: str, case_ids: list[str], out: Path, *,
                  repeats: int = 3, timeout_s: int = 1800,
                  triton_autotune: Path | None = None,
                  captures: Path | None = None,
                  metrics: str | None = None,
                  source_stalls: bool = False) -> dict:
    """``captures`` (<root>/<case> capture directories): time the launch
    each capture traced instead of re-picking with the suite rule."""
    from .. import site
    site.export_environment()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for cid in case_ids:
        cdir = out / cid
        cdir.mkdir(parents=True, exist_ok=True)
        work = cdir / "work"
        work.mkdir(parents=True, exist_ok=True)
        resolved = resolve_case(suite, cid, work)
        prep = prepare_case(resolved, work, cdir / "triton_prepass.log",
                            autotune_from=triton_autotune,
                            timeout_s=min(timeout_s, 1200))
        if prep.get("rc", 0) != 0:
            timing = {"case": cid, "suite": suite,
                      "status": "app_failed:triton_prepass",
                      "prepass": prep}
            (cdir / "timing.json").write_text(
                json.dumps(timing, indent=1))
            summary.append(_tsv_row(cid, resolved, None, repeats,
                                    "app_failed:triton_prepass"))
            continue
        env = dict(os.environ)
        env.update({k: str(v) for k, v in resolved.env.items()})
        env.pop("CUDA_INJECTION64_PATH", None)
        try:
            ncu = _ncu_binary()
        except FileNotFoundError:
            ncu = None
        timing = {"case": cid, "suite": suite, "argv": resolved.argv,
                  "repeats": repeats, "nvidia_smi": _gpu_info(),
                  "ncu": ncu, "ncu_version": _ncu_version(),
                  "cache_control": CACHE_CONTROL,
                  "clock_control": CLOCK_CONTROL,
                  "start": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime())}
        # same argv/env as the traced run: <CAPTURE> substitution and
        # env_profile wrapping (a literal <CAPTURE> reached the shell
        # as a redirection in DeepGEMM's nvcc invocation; far cases
        # produced an empty census)
        # with --captures the app runs with <CAPTURE> = the capture's own
        # case dir: its scratch/ (JIT caches, app result files) is reused
        # instead of recompiling every JIT kernel under ncu in a fresh
        # work dir (flashinfer MoE: hours per repeat; ninja build files
        # hold absolute paths, so a copied cache would rebuild anyway).
        # No tracer is attached, so no capture fact file is touched.
        scratch_root = work
        if captures is not None and (captures / cid).is_dir():
            scratch_root = captures / cid
        argv, run_env = case_command(resolved, scratch_root, env)
        timing["scratch_root"] = str(scratch_root)
        all_rows = []
        rcs = []
        for rep in range(repeats):
            log = cdir / f"ncu_r{rep}.csv"
            rc, recs = _run_once(argv, run_env, resolved.cwd, log,
                                 timeout_s, metrics)
            if metrics and rep == 0:
                (cdir / "ncu_metrics.csv").write_text(
                    log.read_text(errors="replace"))
            rcs.append(rc)
            all_rows.append(_rows(recs, rep))
        timing["repeat_rcs"] = rcs
        # repeat-consistency: same (ordinal, mangled) sequence each repeat
        seqs = [[(r["ordinal"], r["mangled"]) for r in rows]
                for rows in all_rows]
        if seqs and any(s != seqs[0] for s in seqs[1:]):
            timing["status"] = "sequence_error"
            timing["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime())
            (cdir / "timing.json").write_text(json.dumps(timing, indent=1))
            summary.append(_tsv_row(cid, resolved, None, repeats,
                                    "sequence_error"))
            continue
        census = all_rows[0] if all_rows else []
        if not census and any(rcs):
            # ncu never profiled a launch: environment error (driver /
            # ncu mismatch, timeout), not a selection miss
            status = ("ncu_timeout" if rcs[0] == TIMEOUT_RC
                      else f"ncu_failed:rc={rcs[0]}")
            timing["status"] = status
            timing["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime())
            (cdir / "timing.json").write_text(json.dumps(timing, indent=1))
            summary.append(_tsv_row(cid, resolved, None, repeats, status))
            continue
        if captures is not None:
            try:
                kernel, occurrence = traced_launch_key(captures / cid)
            except (OSError, ValueError) as e:
                pick, status = None, "missing_traced_launch"
                timing["launch_from_error"] = str(e)
            else:
                pick = census_row(census, kernel, occurrence)
                status = "traced_launch_not_in_census"
            timing["launch_from"] = str(captures / cid)
            timing["launch_source"] = "captures"
        elif resolved.launch_key is not None:
            pick = census_row(census, *resolved.launch_key)
            status = "table_launch_not_in_census"
            timing["launch_source"] = "table"
        else:
            pick, status = _pick(census, resolved), "pick_failed"
            timing["launch_source"] = "rule"
        if pick is None:
            timing["status"] = status
            summary.append(_tsv_row(cid, resolved, None, repeats, status))
        else:
            # gather the same ordinal across repeats for cycles stats
            ordn = pick["ordinal"]
            cycles, clocks = _launch_stats(all_rows, ordn)
            timing["pick"] = {"ordinal": ordn,
                              "kernel": pick["mangled"],
                              "occurrence": sum(
                                  1 for r in census[:ordn + 1]
                                  if r["mangled"] == pick["mangled"]),
                              "grid": pick["grid"],
                              "block": pick["block"],
                              "cluster": pick["cluster"],
                              "cycles": cycles}
            timing["status"] = "success" if cycles else "no_cycles"
            summary.append(_tsv_row(
                cid, resolved, pick, repeats, timing["status"],
                cycles=cycles, clocks=clocks))
        if source_stalls and pick is not None:
            src_log = cdir / "ncu_source.csv"
            src_rc = _run_source(argv, run_env, resolved.cwd, src_log,
                                 timeout_s, pick["ordinal"])
            timing["source_stalls_rc"] = src_rc
        timing["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (cdir / "timing.json").write_text(json.dumps(timing, indent=1))

    tsv = out / "reference.tsv"
    with tsv.open("w") as fh:
        fh.write("\t".join(TIMING_COLUMNS) + "\n")
        for row in summary:
            fh.write("\t".join(str(c) for c in row) + "\n")
    return {"cases": len(case_ids), "out": str(out),
            "statuses": [r[-2] for r in summary]}


def _dims(dims) -> str:
    if not dims or any(d is None for d in dims):
        return ""
    return "x".join(str(d) for d in dims)


def _tsv_row(cid, resolved, pick, repeats, status, cycles=(), clocks=()):
    kernel = pick["mangled"] if pick else (resolved.kernel_regex or "")
    ordn = pick["ordinal"] if pick else (resolved.occurrence or "")
    grid = _dims(pick["grid"]) if pick else ""
    block = _dims(pick["block"]) if pick else ""
    cluster = _dims(pick["cluster"]) if pick else ""
    med = int(statistics.median(cycles)) if cycles else ""
    return [cid, kernel, ordn, grid, block, cluster,
            med, min(cycles) if cycles else "",
            max(cycles) if cycles else "", "",
            int(statistics.median(clocks)) if clocks else "",
            "ncu", CACHE_CONTROL, CLOCK_CONTROL, repeats, status,
            "cyclescope reference"]
