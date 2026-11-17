"""Single-case driver: capture -> facts -> segments -> job -> simulate.

A faithful port of ``scripts/pipeline/capture_batch.sh`` (GPU stages)
and ``scripts/pipeline/convert_capture_batch.sh``'s ``one_case`` /
``classify`` (CPU stages): same commands, flags, order, layout and
failure classification.  Layout:

    <C>/<case>/                 capture (C = --captures, else <out>/captures)
    <C>/reference/<case>/       hardware reference (cyclescope reference)
    <out>/sim/<case>/           convert.log segments.log build.log sim.log
        stages.tsv status.json facts/launch.json out-<builder>/

Exit code of ``cyclescope run`` is 0 only for stage ``done`` + class ``ok``.
"""
from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cyclescope import site
from cyclescope.pipeline import shards as _shards

REPO = Path(__file__).resolve().parents[3]

_STAGES = ("convert", "segments", "build", "sim")


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "python")
    env["PATH"] = os.pathsep.join(
        [str(site.build_dir()),
         str(site.artifact_root() / "tools" / "bin"),
         env.get("PATH", "")])
    env.setdefault("CYCLESCOPE_ARTIFACT_ROOT", str(site.artifact_root()))
    return env


class _Stages:
    """stage_rss.py equivalent: one line per stage run, appended to
    stages.tsv as ``<stage>\\t<wall_s>\\t<max_rss_kb>\\t<rc>\\t<cpu_s>``
    (RSS/CPU are cumulative RUSAGE_CHILDREN maxima, as the script's)."""

    def __init__(self, path: Path):
        self.path = path
        path.write_text("")

    def run(self, stage: str, cmd: list[str], log: Path, env=None,
            nice: int = 0) -> int:
        cmd = list(cmd)
        if nice:
            cmd = ["nice", "-n", str(nice)] + cmd
        t0 = time.monotonic()
        with log.open("a") as fh:
            rc = subprocess.call(cmd, stdout=fh,
                                 stderr=subprocess.STDOUT, env=env)
        wall = time.monotonic() - t0
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        with self.path.open("a") as fh:
            fh.write(f"{stage}\t{wall:.1f}\t{ru.ru_maxrss}\t{rc}\t"
                     f"{ru.ru_utime + ru.ru_stime:.1f}\n")
        return rc

    def aggregate(self, prefix: str, stage: str) -> None:
        """The awk line the script appended after sharded stages:
        ``<stage>\\t<max wall>\\t<max rss>\\t0``."""
        wall, rss = 0.0, 0
        for line in self.path.read_text().splitlines():
            p = line.split("\t")
            if p[0].startswith(prefix):
                wall = max(wall, float(p[1]))
                rss = max(rss, int(p[2]))
        if rss:
            with self.path.open("a") as fh:
                fh.write(f"{stage}\t{wall:.1f}\t{rss}\t0\n")

    def max_rss(self) -> dict:
        out = {}
        for line in self.path.read_text().splitlines():
            p = line.split("\t")
            if p[0] in _STAGES:
                out[p[0]] = int(p[2])
        return out


def _cli(*args: object) -> list[str]:
    return [sys.executable, "-m", "cyclescope.cli"] + [str(a) for a in args]


def _classify(log: Path) -> str:
    """first missing_fact:/unsupported: token in a log, else the last
    non-empty line's first word, else "error"."""
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return "error"
    m = re.search(r"missing_fact:[^ ]*|unsupported:[^ ]*", text)
    if m:
        return m.group(0)
    if re.search(r"deadlock", text, re.I):
        return "deadlock"
    if re.search(r"killed|Segmentation|Aborted|std::bad_alloc",
                 text, re.I):
        return "crash"
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].split()[0] if lines else "error"


def _read_launch(launch: Path, convert_log: Path):
    """(records, dropped_polls, missing, unsupported) of a facts dir."""
    try:
        d = json.loads(launch.read_text())
    except Exception:
        d = {}
    n = d.get("dynamic_records") or 0
    if not n:
        try:
            n = sum(int(x) for x in re.findall(
                r"convert_capture [^:\n]*: (\d+) records",
                convert_log.read_text(errors="replace")))
        except OSError:
            n = 0
    return (n, d.get("dropped_polls") or 0,
            len(d.get("missing_facts") or {}),
            len(d.get("unsupported") or {}))


def _flag_class(launch: Path) -> str:
    d = json.loads(launch.read_text())
    keys = (list((d.get("missing_facts") or {}).keys())
            + list((d.get("unsupported") or {}).keys()))
    return keys[0] if keys else "launch_json_flags"


def _seg_flag_class(seg_launch: Path) -> tuple[int, int, str]:
    d = json.loads(seg_launch.read_text())
    both = dict(d.get("missing_facts") or {})
    both.update(d.get("unsupported") or {})
    key = max(both, key=lambda k: (both[k], -list(both).index(k))) \
        if both else "ok"
    return (len(d.get("missing_facts") or {}),
            len(d.get("unsupported") or {}), key)


def _read_result(result_bin: Path) -> tuple[int, int, int, int]:
    from cyclescope.job.result import read_result
    r = read_result(result_bin.read_bytes())
    return (r.get("issued", 0), r.get("completed", 0),
            r.get("cycles", 0), r.get("hard_constraint_failures", 0))


def _reference_row(tsv: Path, case: str) -> dict | None:
    import csv
    try:
        with tsv.open() as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r.get("case", "").strip() == case:
                    return r
    except OSError:
        pass
    return None


def _gpu_stages(suite: str, case: str, cap_root: Path, *,
                repeats: int, timeout_s: int, tracer,
                skip_reference: bool, env: dict) -> tuple[str, str]:
    """capture_batch.sh's per-case GPU part; returns (stage, class)."""
    case_dir = cap_root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    crc = subprocess.call(
        _cli("capture", "--suite", suite, "--case", case,
             "--out", case_dir, "--mode", "both",
             "--timeout", timeout_s)
        + (["--tracer", str(tracer)] if tracer else []),
        env=env)
    rrc = 0
    if not skip_reference and crc == 0:
        autotune = case_dir / ".work" / "triton_autotune.json"
        rrc = subprocess.call(
            _cli("reference", "--suite", suite, "--cases", case,
                 "--repeats", repeats, "--out", cap_root / "reference" / case,
                 "--captures", cap_root, "--timeout", timeout_s)
            + (["--triton-autotune", str(autotune)]
               if autotune.is_file() else []),
            env=env)
    try:
        run = json.loads((case_dir / "run.json").read_text())
        tcrc = run.get("trace_check_rc", 99)
    except Exception:
        tcrc = 98
    if crc == 0 and tcrc == 0 and (skip_reference or rrc == 0):
        (case_dir / ".done").touch()
        return "done", "ok"
    if crc != 0 or tcrc != 0:
        return "capture", f"capture_rc{crc}_tc{tcrc}"
    return "reference", f"reference_rc{rrc}"


def _tree_default(suite: str) -> Path:
    arch = suite.split("/", 1)[0]
    return (REPO / "configs" / "trees"
            / ("h100.json" if arch == "hopper" else "b300.json"))


def run_case(suite: str, case: str, out: Path, *,
             captures: Path | None = None,
             reference_tsv: Path | None = None,
             skip_reference: bool = False,
             tree: Path | None = None,
             engine: str | None = None,
             builder: str = "cpp",
             jobs: int | None = None,
             shard_records: int = 1_500_000,
             shard_max: int = 16,
             placement: bool = True,
             keep: bool = False,
             ledgers: bool = False,
             repeats: int = 3,
             timeout_s: int = 5400,
             tracer=None,
             scratch: Path | None = None) -> dict:
    out = Path(out)
    arch = suite.split("/", 1)[0]
    tree = Path(tree) if tree else _tree_default(suite)
    cap_root = Path(captures) if captures else Path(out) / "captures"
    o = Path(out) / "sim" / case
    o.mkdir(parents=True, exist_ok=True)
    env = _env()
    engine = engine or shutil.which("cyclescope-engine", path=env["PATH"]) \
        or "cyclescope-engine"

    if captures is None:
        stage, cls = _gpu_stages(suite, case, cap_root, repeats=repeats,
                                 timeout_s=timeout_s, tracer=tracer,
                                 skip_reference=skip_reference, env=env)
        if cls != "ok":
            status = _status(suite, case, stage, cls)
            _write_status(o, status)
            return status

    cap_dir = cap_root / case
    w = Path(scratch) / case if scratch else o
    if scratch:
        shutil.rmtree(w, ignore_errors=True)
    w.mkdir(parents=True, exist_ok=True)
    facts, seg = w / "facts", w / f"out-{builder}"
    ref = Path(reference_tsv) if reference_tsv else None
    if ref is None and (cap_root / "reference" / case
                        / "reference.tsv").is_file():
        ref = cap_root / "reference" / case / "reference.tsv"
    place = cap_dir / "placement"
    use_placement = placement and place.is_dir()

    stats = _Stages(o / "stages.tsv")
    t0 = time.monotonic()
    stage, cls = "convert", "ok"
    conv_log, seg_log = o / "convert.log", o / "segments.log"
    conv_log.write_text("")
    seg_log.write_text("")

    rec_est, rec_src = _shards.record_estimate(cap_dir)
    nsh = _shards.shard_count(rec_est, shard_records, shard_max)
    par = jobs if jobs else min(nsh, os.cpu_count() or 1)

    if nsh <= 1:
        if stats.run("convert", _cli(
                "convert", "--capture", cap_dir, "--arch", arch,
                "--tree", tree, "--out", facts, "--no-cache",
                *(["--reference", ref] if ref else []),
                *(["--placement", place] if use_placement else [])),
                conv_log, env) != 0:
            cls = _classify(conv_log)
    else:
        conv_log.write_text(
            f"sharded convert: {nsh} shards for {rec_est} records "
            f"({rec_src})\n")
        shutil.rmtree(w / "shards", ignore_errors=True)
        shutil.rmtree(w / "split", ignore_errors=True)
        (w / "shards").mkdir(parents=True)
        split_dir = None
        if stats.run("split", _cli(
                "convert", "--capture", cap_dir, "--arch", arch,
                "--split-shards", nsh, "--out", w / "split"),
                conv_log, env) != 0:
            cls = _classify(conv_log)
        if cls == "ok":
            if list((w / "split").glob(
                    f"trace-*.shard0of{nsh}.bin.zst")):
                split_dir = w / "split"
            else:
                cls = "split_failed"

        def shard(k: int) -> int:
            senv = dict(env)
            senv["CYCLESCOPE_CAPTURE_THREADS"] = "1"
            if stats.run(f"convert-shard{k}", _cli(
                    "convert", "--capture", cap_dir, "--arch", arch,
                    "--tree", tree, "--out", w / "shards" / str(k),
                    "--no-cache", "--shard", f"{k}/{nsh}",
                    *(["--split-dir", split_dir] if split_dir else []),
                    *(["--reference", ref] if ref else []),
                    *(["--placement", place] if use_placement else [])),
                    w / "shards" / f"convert-{k}.log", senv,
                    nice=5) != 0:
                return 1
            return stats.run(f"segments-shard{k}", _cli(
                "convert", "--facts", w / "shards" / str(k),
                "--builder", builder, "--arch", arch, "--tree", tree,
                "--out", w / "shards" / f"seg-{k}", "--no-cache"),
                w / "shards" / f"segments-{k}.log", env, nice=5)

        if cls == "ok":
            with ThreadPoolExecutor(max_workers=par) as ex:
                list(ex.map(shard, range(nsh)))
        for k in range(nsh):
            slog = w / "shards" / f"convert-{k}.log"
            if slog.is_file():
                with conv_log.open("a") as fh:
                    fh.write(slog.read_text(errors="replace"))
            if cls == "ok" and not (w / "shards" / str(k)
                                    / "launch.json").is_file():
                cls = _classify(slog)
        if cls == "ok":
            try:
                _shards.merge_shards(
                    facts, sorted((w / "shards").glob("[0-9]*"),
                                  key=lambda p: int(p.name)))
            except SystemExit as e:
                with conv_log.open("a") as fh:
                    fh.write(f"{e}\n")
                cls = "merge_failed"
        stats.aggregate("convert-shard", "convert")

    t1 = time.monotonic()
    issued = completed = cycles = hard = 0
    records, dropped, miss, unsup = _read_launch(facts / "launch.json",
                                               conv_log)
    if cls == "ok" and (miss or unsup):
        cls = _flag_class(facts / "launch.json")

    if cls == "ok":
        stage = "segments"
        if nsh <= 1:
            if stats.run("segments", _cli(
                    "convert", "--facts", facts, "--builder", builder,
                    "--arch", arch, "--tree", tree, "--out", seg,
                    "--no-cache"), seg_log, env) != 0:
                cls = _classify(seg_log)
        else:
            for k in range(nsh):
                slog = w / "shards" / f"segments-{k}.log"
                if slog.is_file():
                    with seg_log.open("a") as fh:
                        fh.write(slog.read_text(errors="replace"))
                if cls == "ok" and not (w / "shards" / f"seg-{k}"
                                        / "launch.json").is_file():
                    cls = _classify(slog)
            if cls == "ok":
                try:
                    _shards.merge_shards(
                        seg,
                        sorted((w / "shards").glob("seg-*"),
                               key=lambda p: int(p.name.split("-")[1])),
                        segments=True)
                except SystemExit as e:
                    with seg_log.open("a") as fh:
                        fh.write(f"{e}\n")
                    cls = "merge_failed"
            stats.aggregate("segments-shard", "segments")
            if cls == "ok":
                shutil.rmtree(w / "shards", ignore_errors=True)
                shutil.rmtree(w / "split", ignore_errors=True)
        t2 = time.monotonic()
    else:
        t2 = time.monotonic()

    if cls == "ok" and (seg / "launch.json").is_file():
        _m, _u, cls_seg = _seg_flag_class(seg / "launch.json")
        if cls_seg != "ok":
            cls = cls_seg

    if cls == "ok":
        stage = "build"
        if not keep:
            shutil.rmtree(facts, ignore_errors=True)
        if stats.run("build",
                     _cli("build", seg, "--out", seg / "job.bin",
                          "--keep-segments"),
                     o / "build.log", env) != 0:
            cls = _classify(o / "build.log")
        t3 = time.monotonic()
        if cls == "ok" and not keep:
            shutil.rmtree(seg / "segments", ignore_errors=True)
    else:
        t3 = time.monotonic()

    if cls == "ok":
        stage = "sim"
        led = []
        if ledgers:
            led = ["--journal", str(seg / "journal.bin"),
                   "--wait-ledger", str(seg / "wait.bin")]
        if stats.run("sim",
                     [engine, "simulate", "--job", str(seg / "job.bin"),
                      "--out", str(seg / "result.bin")] + led,
                     o / "sim.log", env) != 0:
            cls = _classify(o / "sim.log")
        t4 = time.monotonic()
        if cls == "ok" and not keep:
            (seg / "job.bin").unlink(missing_ok=True)
        if (seg / "result.bin").is_file():
            issued, completed, cycles, hard = _read_result(
                seg / "result.bin")
            if cls == "ok" and issued != completed:
                cls = "issued_ne_completed"
    else:
        t4 = time.monotonic()

    if cls == "ok":
        stage = "done"

    # copy durable outputs back from scratch, then drop it
    if w != o:
        if keep:
            for item in w.iterdir():
                dst = o / item.name
                if dst.exists():
                    shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
                shutil.move(str(item), str(dst))
        else:
            (o / f"out-{builder}").mkdir(exist_ok=True)
            (o / "facts").mkdir(exist_ok=True)
            for src, dst in ((seg / "result.bin",
                              o / f"out-{builder}" / "result.bin"),
                             (seg / "launch.json",
                              o / f"out-{builder}" / "launch.json"),
                             (facts / "launch.json",
                              o / "facts" / "launch.json")):
                if src.is_file():
                    shutil.copy2(src, dst)
        shutil.rmtree(w, ignore_errors=True)

    ref_row = _reference_row(ref, case) if ref else None
    try:
        seg_doc = json.loads((o / f"out-{builder}" / "launch.json"
                              if w != o else seg / "launch.json")
                             .read_text())
    except Exception:
        seg_doc = {}
    from cyclescope.pipeline.report import placement_digest
    rss = stats.max_rss()
    status = {
        "suite": suite, "case": case, "stage": stage, "class": cls,
        "records": records, "dropped_polls": dropped,
        "missing": miss, "unsupported": unsup,
        "issued": issued, "completed": completed, "cycles": cycles,
        "hard": hard,
        "walls": {"convert": round(t1 - t0, 1),
                  "segments": round(t2 - t1, 1),
                  "build": round(t3 - t2, 1),
                  "sim": round(t4 - t3, 1),
                  "total": round(t4 - t0, 1)},
        "rss_kb": {s: rss.get(s, 0) for s in _STAGES},
        "hw_cycles": (float(ref_row["cycles_median"])
                      if ref_row and ref_row.get("cycles_median")
                      else None),
        "reference_cache_control": (ref_row.get("cache_control")
                                    if ref_row else None),
        "placement_digest": placement_digest(seg_doc),
        "shards": nsh,
    }
    _write_status(o, status)
    return status


def _status(suite: str, case: str, stage: str, cls: str) -> dict:
    return {"suite": suite, "case": case, "stage": stage, "class": cls,
            "records": 0, "dropped_polls": 0, "missing": 0,
            "unsupported": 0, "issued": 0, "completed": 0, "cycles": 0,
            "hard": 0,
            "walls": {"convert": 0, "segments": 0, "build": 0,
                      "sim": 0, "total": 0},
            "rss_kb": {s: 0 for s in _STAGES},
            "hw_cycles": None, "reference_cache_control": None,
            "placement_digest": "", "shards": 0}


def _write_status(o: Path, status: dict) -> None:
    (o / "status.json").write_text(json.dumps(status, indent=1))
