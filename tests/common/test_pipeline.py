"""cyclescope.pipeline: shard sizing, record estimate, placement digest,
verdict rules, baseline accuracy, CLI parsing, toolchain record."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cyclescope.pipeline import report as rep
from cyclescope.pipeline import shards

REPO = Path(__file__).resolve().parents[2]


# ---- shard_count / record_estimate --------------------------------------

def test_shard_count_shell_formula():
    cases = [(0, 100, 16, 1), (-5, 100, 16, 1), (100, 0, 16, 1),
             (1, 100, 16, 1), (100, 100, 16, 1), (101, 100, 16, 2),
             (1600, 100, 16, 16), (1700, 100, 16, 16),
             (10, 3, 4, 4)]
    for rec, per, mx, want in cases:
        assert shards.shard_count(rec, per, mx) == want, (rec, per, mx)


def test_record_estimate(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    assert shards.record_estimate(cap) == (0, "none")
    (cap / "trace-0.bin.zst").write_bytes(b"x" * 400)
    n, src = shards.record_estimate(cap)
    assert (n, src) == (400 // shards.BPR, "bytes")
    (cap / "trace_check.txt").write_text("records: ... total=12345")
    assert shards.record_estimate(cap) == (12345, "total")


# ---- placement_digest / verdict / accuracy -------------------------------

def _doc(sms):
    return {"placements": {str(i): {"sm_id": s} for i, s in
                           enumerate(sms)}}


def test_placement_digest():
    assert rep.placement_digest({}) == ""
    assert rep.placement_digest({"placements": {}}) == ""
    doc = _doc([0, 5, 5, 9])
    text = "\n".join(f"{c}:{s}" for c, s in
                     ((0, 0), (1, 5), (2, 5), (3, 9)))
    want = hashlib.sha256(text.encode()).hexdigest()[:16]
    assert rep.placement_digest(doc) == want
    # ordering is by int key, not string: "10" sorts after "2"
    doc2 = {"placements": {str(i): {"sm_id": i} for i in range(11)}}
    text2 = "\n".join(f"{i}:{i}" for i in range(11))
    assert rep.placement_digest(doc2) == \
        hashlib.sha256(text2.encode()).hexdigest()[:16]


def _status(cls="ok", stage="done", cycles=1000, digest="aa"):
    return {"suite": "hopper/full_gpu", "case": "c", "stage": stage,
            "class": cls, "cycles": cycles, "issued": 10,
            "placement_digest": digest, "hw_cycles": 1100}


def _erow(cycles=1000, digest="aa"):
    return {"case": "c", "class": "ok", "sim_cycles": str(cycles),
            "hw_cycles": "1100", "placement_digest": digest}


def test_verdict():
    # same placement within tolerance -> ok
    assert rep.verdict(_status(cycles=1020), _erow()) == "ok"
    # same placement beyond tolerance -> fail
    assert rep.verdict(_status(cycles=1100), _erow()) == "fail"
    # different placement -> listed
    assert rep.verdict(_status(digest="bb"), _erow()) == "listed"
    assert rep.verdict(_status(digest=""), _erow()) == "listed"
    # non-ok -> fail
    assert rep.verdict(_status(cls="crash"), _erow()) == "fail"
    assert rep.verdict(_status(stage="sim"), _erow()) == "fail"
    # no expected row
    assert rep.verdict(_status(), None) == "n/a"


def test_accuracy_shipped_tables():
    for suite, mape, p90, med in (("hopper", 7.93, 17.2, 0.947),
                                  ("blackwell", 7.64, 14.8, 0.952)):
        rows = rep.read_expected(REPO / "suites" / suite / "full_gpu"
                                 / "expected.tsv")
        vals = [{"sim_cycles": r["sim_cycles"],
                 "hw_cycles": r["hw_cycles"]}
                for r in rows.values()
                if r.get("class") == "ok" and r.get("hw_cycles")]
        a = rep.accuracy(vals)
        assert a["cases"] == 500
        assert a["mape_pct"] == pytest.approx(mape, abs=0.02)
        assert a["p90_abs_err_pct"] == pytest.approx(p90, abs=0.05)
        assert a["median_ratio"] == pytest.approx(med, abs=0.002)


# ---- CLI parsing ----------------------------------------------------------

def _cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "cyclescope.cli"] + list(argv),
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "python")})


def test_cli_run_help():
    rc = _cli("run", "--help")
    assert rc.returncode == 0
    for opt in ("--captures", "--reference-tsv", "--skip-reference",
                "--shard-records", "--no-placement", "--ledgers",
                "--keep", "--jobs", "--scratch"):
        assert opt in rc.stdout


def test_cli_report_help():
    rc = _cli("report", "--help")
    assert rc.returncode == 0
    for opt in ("--expected", "--tolerance", "--baseline", "--out"):
        assert opt in rc.stdout


def test_cli_report_run(tmp_path):
    st = _status()
    d = tmp_path / "sim" / "c"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps(st))
    rc = _cli("report", str(tmp_path))
    assert rc.returncode == 0
    assert "c\t" in rc.stdout or "\tc\t" in rc.stdout


# ---- toolchain record -----------------------------------------------------

def test_toolchain_record(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    ptxas = fake / "ptxas"
    ptxas.write_text("#!/bin/sh\necho 'ptxas V13.0.0'\n")
    ptxas.chmod(0o755)
    from cyclescope.capture.driver import _toolchain_record
    rec = _toolchain_record({"PATH": str(fake),
                             "TRITON_PTXAS_PATH": str(ptxas)})
    assert rec["which"]["ptxas"] == str(ptxas)
    assert rec["which"]["TRITON_PTXAS_PATH"] == str(ptxas)
    assert rec["versions"]["ptxas"] == "ptxas V13.0.0"
    assert rec["python"] == sys.version
    assert "packages" in rec
    # missing tools are simply absent, never a failure
    rec2 = _toolchain_record({"PATH": str(tmp_path)})
    assert rec2["which"] == {}
    assert rec2["versions"] == {}
