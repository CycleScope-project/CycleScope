"""One launch for capture and reference: traced/timed launch keys,
reference realignment from recorded ncu repeats, merge --align-to."""
import csv
import json
import re
import runpy
import sys
from pathlib import Path

import pytest

from cyclescope.capture import driver
from cyclescope.capture.driver import Resolved
from cyclescope.capture.launch_selection import (census_row, selection_for,
                                             traced_launch_key)
from cyclescope.reference import ncu_runner

REPO = Path(__file__).resolve().parents[2]


def _launches(d: Path, names: list[str], traced: int) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "launches.jsonl").open("w") as fh:
        for i, n in enumerate(names, 1):
            fh.write(json.dumps({"launch": i, "mangled": n,
                                 "traced": i == traced}) + "\n")
    return d


def _ncu_csv(path: Path, kernels: list[tuple[str, int]]) -> None:
    fields = ["ID", "Kernel Name", *driver.NCU_LAUNCH_METRICS]
    with path.open("w", newline="") as fh:
        fh.write("==PROF== Connected to process 1\n")
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow(fields)
        for i, (name, cycles) in enumerate(kernels):
            rec = {m: "1" for m in driver.NCU_LAUNCH_METRICS}
            rec.update({"sm__cycles_active.max": str(cycles),
                        "sm__cycles_elapsed.avg.per_second": "2000000000",
                        "launch__grid_dim_x": "2",
                        "launch__cluster_dim_x": "0",
                        "launch__cluster_dim_y": "0",
                        "launch__cluster_dim_z": "0"})
            w.writerow([str(i), name] + [rec[m] for m in
                                         driver.NCU_LAUNCH_METRICS])


def _reference_case(root: Path, cid: str, repeats: list[list], ordinal: int):
    case_dir = root / "reference" / cid / cid
    case_dir.mkdir(parents=True)
    for rep, kernels in enumerate(repeats):
        _ncu_csv(case_dir / f"ncu_r{rep}.csv", kernels)
    (case_dir / "timing.json").write_text(json.dumps(
        {"repeats": len(repeats), "pick": {"ordinal": ordinal}}))
    all_rows = ncu_runner._recorded_rows(case_dir, len(repeats))
    cycles, clocks = ncu_runner._launch_stats(all_rows, ordinal)
    row = ncu_runner._tsv_row(cid, None, all_rows[0][ordinal], len(repeats),
                              "success", cycles=cycles, clocks=clocks)
    (root / "reference" / cid / "reference.tsv").write_text(
        "\t".join(ncu_runner.TIMING_COLUMNS) + "\n"
        + "\t".join(str(c) for c in row) + "\n")
    return case_dir


TWO = [[("K1", 100), ("K2", 90)], [("K1", 95), ("K2", 120)],
       [("K1", 99), ("K2", 110)]]


def test_traced_launch_key_counts_exact_name_occurrences(tmp_path):
    d = _launches(tmp_path / "c", ["A", "B", "A", "Bx", "B"], traced=5)
    assert traced_launch_key(d) == ("B", 2)
    _launches(tmp_path / "n", ["A", "B"], traced=0)
    with pytest.raises(ValueError):
        traced_launch_key(tmp_path / "n")


def test_census_row_and_selection_for():
    rows = [{"function": "k$1.x", "ordinal": 0},
            {"function": "k$1yx", "ordinal": 1},
            {"function": "k$1.x", "ordinal": 2}]
    assert census_row(rows, "k$1.x", 2, name_key="function")["ordinal"] == 2
    assert census_row(rows, "k$1.x", 3, name_key="function") is None
    pattern, occurrence = selection_for("k$1.x", 2).rsplit("@", 1)
    assert occurrence == "2"
    assert re.fullmatch(pattern, "k$1.x")
    assert not re.fullmatch(pattern, "k$1yx")


def test_reference_launch_key_and_realigned_row(tmp_path):
    case_dir = _reference_case(tmp_path, "c", TWO, ordinal=0)
    assert ncu_runner.reference_launch_key(case_dir) == ("K1", 1)
    row = ncu_runner.realigned_row(case_dir, "c", "K2", 1)
    col = dict(zip(ncu_runner.TIMING_COLUMNS, row))
    assert col["kernel"] == "K2" and col["launch_ordinal"] == 1
    assert (col["cycles_median"], col["cycles_min"], col["cycles_max"]) \
        == (110, 90, 120)
    assert col["sm_clock_khz"] == 2000000 and col["status"] == "success"
    assert col["source"] == "cyclescope reference realigned to traced launch"
    assert ncu_runner.realigned_row(case_dir, "c", "K3", 1) is None
    assert ncu_runner.realigned_row(case_dir, "c", "K2", 2) is None


def test_realigned_row_requires_one_ordinal_across_repeats(tmp_path):
    moved = [TWO[0], TWO[1], [("K2", 110), ("K1", 99)]]
    case_dir = _reference_case(tmp_path, "c", moved, ordinal=0)
    assert ncu_runner.realigned_row(case_dir, "c", "K2", 1) is None


def test_merge_reference_align_to(tmp_path, monkeypatch, capsys):
    refs, caps = tmp_path / "refs", tmp_path / "caps"
    for cid in ("same", "flip", "absent"):
        _reference_case(refs, cid, TWO, ordinal=0)
    _launches(caps / "same", ["K1", "K2"], traced=1)
    _launches(caps / "flip", ["K1", "K2"], traced=2)
    _launches(caps / "absent", ["K1", "K3"], traced=2)
    cases = tmp_path / "cases.txt"
    cases.write_text("same\nflip\nabsent\n")
    out = tmp_path / "reference.tsv"
    monkeypatch.setattr(sys, "argv", [
        "merge_reference.py", "--captures", str(refs), "--cases", str(cases),
        "--out", str(out), "--align-to", str(caps)])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(REPO / "scripts/pipeline/merge_reference.py"),
                       run_name="__main__")
    assert e.value.code == 1
    rows = {r["case"]: r for r in
            csv.DictReader(out.open(), delimiter="\t")}
    assert sorted(rows) == ["flip", "same"]
    assert (rows["same"]["kernel"], rows["same"]["launch_ordinal"],
            rows["same"]["cycles_median"], rows["same"]["source"]) == \
        ("K1", "0", "99", "cyclescope reference")
    assert (rows["flip"]["kernel"], rows["flip"]["launch_ordinal"],
            rows["flip"]["cycles_median"]) == ("K2", "1", "110")
    io = capsys.readouterr()
    assert "realigned flip ordinal 0->1 cycles_median 99->110" in io.out
    assert "unaligned absent" in io.err


def test_run_reference_times_the_traced_launch(tmp_path, monkeypatch):
    def fake_run_once(argv, env, cwd, log, timeout_s, metrics=None):
        return 0, [{"Kernel Name": "K1", "sm__cycles_active.max": "100"},
                   {"Kernel Name": "K2", "sm__cycles_active.max": "90"},
                   {"Kernel Name": "K2", "sm__cycles_active.max": "80"}]

    bw = Resolved(argv=[], env={}, cwd=None, arch="blackwell",
                  family="cutlass", calls=1)
    monkeypatch.setattr(ncu_runner, "resolve_case", lambda s, c, w: bw)
    monkeypatch.setattr(ncu_runner, "_run_once", fake_run_once)
    monkeypatch.setattr(ncu_runner, "_gpu_info", lambda: {"gpu": "x"})
    monkeypatch.setattr(ncu_runner, "_ncu_version", lambda: "ncu v1")
    monkeypatch.setattr(ncu_runner, "_ncu_binary", lambda: "/x/ncu")
    caps = tmp_path / "caps"
    _launches(caps / "caseX", ["K1", "K2", "K2"], traced=3)
    rec = ncu_runner.run_reference("blackwell/full_gpu", ["caseX"],
                                   tmp_path / "ref", repeats=2,
                                   captures=caps)
    assert rec["statuses"] == ["success"]
    pick = json.loads(
        (tmp_path / "ref" / "caseX" / "timing.json").read_text())["pick"]
    assert (pick["ordinal"], pick["kernel"], pick["occurrence"],
            pick["cycles"]) == (2, "K2", 2, [80, 80])
    _launches(caps / "caseX", ["K1", "K3"], traced=2)
    rec = ncu_runner.run_reference("blackwell/full_gpu", ["caseX"],
                                   tmp_path / "ref2", repeats=1,
                                   captures=caps)
    assert rec["statuses"] == ["traced_launch_not_in_census"]


def test_capture_reference_selection_checks_census(tmp_path):
    case_dir = _reference_case(tmp_path / "refs", "c", TWO, ordinal=1)
    out = tmp_path / "cap"
    out.mkdir()
    (out / "ncu_launches.jsonl").write_text(
        json.dumps({"ordinal": 0, "function": "K1"}) + "\n"
        + json.dumps({"ordinal": 1, "function": "K2"}) + "\n")
    assert driver._reference_selection(case_dir, out) == \
        selection_for("K2", 1)
    (out / "ncu_launches.jsonl").write_text(
        json.dumps({"ordinal": 0, "function": "K1"}) + "\n")
    with pytest.raises(KeyError):
        driver._reference_selection(case_dir, out)
