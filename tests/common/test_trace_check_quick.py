"""Quick trace_check integrity: single traced launch, trace file frame
check + GSIMTRC header, functions tsv presence."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cyclescope.capture.trace_check import quick_check

zstd = shutil.which("zstd")
pytestmark = pytest.mark.skipif(zstd is None, reason="zstd not in PATH")


def _mk(d: Path, traced_rows=1, fid=7, trace=True, zst=True,
        good_head=True, functions=True, trunc=False):
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(3):
        rows.append({"launch": i + 1, "function_id": fid,
                     "traced": i < traced_rows})
    (d / "launches.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    if trace:
        raw = (b"GSIMTRC\0" if good_head else b"BADTREC\0") + b"payload"
        if zst:
            p = subprocess.run([zstd, "-q", "-f", "-o",
                                str(d / "trace-1.bin.zst"), "-"],
                               input=raw, capture_output=True)
            if trunc:
                f = d / "trace-1.bin.zst"
                f.write_bytes(f.read_bytes()[:-2])
        else:
            (d / "trace-1.bin").write_bytes(raw)
    if functions:
        (d / "functions").mkdir(exist_ok=True)
        (d / "functions" / f"{fid}.tsv").write_text("x\ty\n")


def test_ok(tmp_path):
    _mk(tmp_path)
    fails, summ = quick_check(tmp_path)
    assert not fails and summ["launch"] == 1


def test_no_traced(tmp_path):
    _mk(tmp_path, traced_rows=0)
    assert quick_check(tmp_path)[0] == ["no traced launch"]


def test_two_traced(tmp_path):
    _mk(tmp_path, traced_rows=2)
    assert quick_check(tmp_path)[0] == ["2 traced launches"]


def test_missing_trace(tmp_path):
    _mk(tmp_path, trace=False)
    assert "missing trace" in quick_check(tmp_path)[0][0]


def test_truncated_zst(tmp_path):
    _mk(tmp_path, trunc=True)
    assert quick_check(tmp_path)[0] == ["zstd frame check failed"]


def test_bad_header(tmp_path):
    _mk(tmp_path, good_head=False)
    assert "bad trace header" in quick_check(tmp_path)[0]


def test_missing_function_tsv(tmp_path):
    _mk(tmp_path, functions=False)
    assert quick_check(tmp_path)[0] == ["missing functions/7.tsv"]


def test_uncompressed_bin(tmp_path):
    _mk(tmp_path, zst=False)
    fails, _ = quick_check(tmp_path)
    assert not fails
