"""Fixed launch table: load/resolve, capture selection priority,
post-trace geometry check, reference pick priority, table collector."""
import csv
import json
import runpy
import sys
from pathlib import Path

import pytest

from cyclescope.capture import driver
from cyclescope.capture.driver import (Resolved, _capture_selection,
                                   _check_traced_against_table)
from cyclescope.capture.launch_selection import (load_launch_table,
                                             selection_for)
from cyclescope.reference import ncu_runner

REPO = Path(__file__).resolve().parents[2]
HEADER = "case\tkernel\toccurrence\tgrid\tblock\tcluster\tsource\n"


def test_load_launch_table(tmp_path):
    t = tmp_path / "launches.tsv"
    t.write_text(HEADER + "c1\tK1\t2\t1x1x1\t128x1x1\t1x1x1\ttrace:x\n")
    rows = load_launch_table(t)
    assert rows["c1"]["kernel"] == "K1" and rows["c1"]["occurrence"] == "2"
    assert load_launch_table(tmp_path / "missing.tsv") == {}


def test_capture_selection_priority(tmp_path, monkeypatch):
    out = tmp_path / "o"
    out.mkdir()
    calls = {"n": 0}

    def census():
        calls["n"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(driver, "_reference_selection",
                        lambda d, o: "REFSEL")
    r = Resolved(argv=[], env={}, cwd=None, arch="hopper",
                 kernel_regex="K1", occurrence=2,
                 launch_key=("K1", 3), launch_row={"kernel": "K1"})
    sel, src = _capture_selection(r, out, tmp_path / "refcase", census)
    assert src == "reference" and sel == "REFSEL" and calls["n"] == 0
    sel, src = _capture_selection(r, out, None, census)
    assert src == "table" and calls["n"] == 0
    assert sel == selection_for("K1", 3)


def _bw_resolved(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    # ncu census the main-last picker resolves: K1 twice, K2 three times
    # (calls=1 -> eligible name needs count%1==0; heaviest total wins)
    rows = [{"function": "K1", "ordinal": i,
             "cycles_active_max": c, "grid": "1", "block": "1"}
            for i, c in [(0, 5), (1, 10), (2, 20), (3, 30), (4, 40)]]
    names = ["K1", "K1", "K2", "K2", "K2"]
    for r, n in zip(rows, names):
        r["function"] = n
    with (out / "ncu_launches.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return out


def test_capture_selection_census_bw(tmp_path):
    out = _bw_resolved(tmp_path)
    calls = {"n": 0}

    def census():
        calls["n"] += 1
        return {"status": "ok"}

    import re
    r = Resolved(argv=[], env={}, cwd=None, arch="blackwell", calls=1,
                 family="cutlass")
    sel, src = _capture_selection(r, out, None, census)
    assert src == "census" and calls["n"] == 1
    assert sel.startswith(re.escape("K2") + "@")


def test_check_traced_against_table(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    row = {"kernel": "K1", "occurrence": "1", "grid": "2x1x1",
           "block": "128x1x1", "cluster": "2x1x1"}
    rec = {"launch": 1, "mangled": "K1", "traced": True,
           "grid": [2, 1, 1], "block": [128, 1, 1], "cluster": [2, 1, 1]}
    (out / "launches.jsonl").write_text(json.dumps(rec) + "\n")
    assert _check_traced_against_table(out, row) is None
    bad = dict(row, grid="4x1x1")
    assert "grid 2x1x1 != 4x1x1" in _check_traced_against_table(out, bad)
    bad2 = dict(row, kernel="K9")
    assert "kernel K1 != K9" in _check_traced_against_table(out, bad2)


def _ncu_recs():
    return [{"Kernel Name": "K1", "sm__cycles_active.max": "5"},
            {"Kernel Name": "K2", "sm__cycles_active.max": "10"},
            {"Kernel Name": "K2", "sm__cycles_active.max": "30"},
            {"Kernel Name": "K1", "sm__cycles_active.max": "7"}]


def test_run_reference_table_and_missing(monkeypatch, tmp_path):
    rows = _ncu_recs()
    monkeypatch.setattr(ncu_runner, "_run_once",
                        lambda *a, **k: (0, rows))
    monkeypatch.setattr(ncu_runner, "_gpu_info", lambda: {"gpu": "x"})
    monkeypatch.setattr(ncu_runner, "_ncu_version", lambda: "ncu v1")
    monkeypatch.setattr(ncu_runner, "_ncu_binary", lambda: "/x/ncu")
    monkeypatch.setattr(ncu_runner, "resolve_case",
                        lambda *a, **k: Resolved(
                            argv=[], env={}, cwd=None,
                            arch="blackwell", calls=1,
                            launch_key=("K2", 2), launch_row={}))
    out = ncu_runner.run_reference("blackwell/full_gpu", ["c1"],
                                   tmp_path / "ref", repeats=1)
    t = json.loads((tmp_path / "ref" / "c1" / "timing.json").read_text())
    assert t["launch_source"] == "table"
    assert t["pick"]["kernel"] == "K2" and t["pick"]["ordinal"] == 2
    assert t["pick"]["occurrence"] == 2 and t["status"] == "success"

    monkeypatch.setattr(ncu_runner, "resolve_case",
                        lambda *a, **k: Resolved(
                            argv=[], env={}, cwd=None,
                            arch="blackwell", calls=1,
                            launch_key=("K9", 1), launch_row={}))
    out = ncu_runner.run_reference("blackwell/full_gpu", ["c2"],
                                   tmp_path / "ref2", repeats=1)
    t2 = json.loads((tmp_path / "ref2" / "c2" / "timing.json").read_text())
    assert t2["status"] == "table_launch_not_in_census"


def _mk_capture(d: Path, kernel: str, occurrence_launch: int,
                grid=(2, 1, 1)):
    d.mkdir(parents=True, exist_ok=True)
    (d / ".done").touch()
    (d / "trace-1.bin.zst").touch()
    rows = []
    for i in range(occurrence_launch):
        rows.append({"launch": i + 1, "mangled": kernel,
                     "traced": i + 1 == occurrence_launch,
                     "grid": list(grid), "block": [128, 1, 1],
                     "cluster": [2, 1, 1]})
    (d / "launches.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


def test_launch_table_script(tmp_path):
    caps = tmp_path / "caps"
    _mk_capture(caps / "b", "K2", 1)
    _mk_capture(caps / "a", "K1", 2)
    table = tmp_path / "launches.tsv"
    argv = ["x", "--suite", "hopper/full_gpu",
            "--captures", str(caps), "--table", str(table)]
    sys.argv = argv
    import subprocess
    r = subprocess.run([sys.executable,
                        str(REPO / "scripts/pipeline/launch_table.py"),
                        *argv[1:]], capture_output=True, text=True)
    assert r.returncode == 0 and "appended=2" in r.stdout
    rows = load_launch_table(table)
    assert rows["a"]["kernel"] == "K1" and rows["a"]["occurrence"] == "2"
    # rerun: identical
    r = subprocess.run([sys.executable,
                        str(REPO / "scripts/pipeline/launch_table.py"),
                        *argv[1:]], capture_output=True, text=True)
    assert r.returncode == 0 and "identical=2" in r.stdout
    # conflict: rewrite capture a with a different kernel
    _mk_capture(caps / "a", "K9", 1)
    r = subprocess.run([sys.executable,
                        str(REPO / "scripts/pipeline/launch_table.py"),
                        *argv[1:]], capture_output=True, text=True)
    assert r.returncode == 1 and "conflicts=1" in r.stdout
    rows = load_launch_table(table)
    assert rows["a"]["kernel"] == "K1"  # never rewritten
