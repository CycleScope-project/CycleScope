import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))


def test_triton_spec_and_kind_map():
    specs = json.loads(
        (REPO / "suites/hopper/full_gpu/cases_triton.json").read_text())
    sys.path.insert(0, str(REPO / "suites/hopper/runners/triton"))
    import run as trunner
    for s in specs:
        assert s["kind"] in trunner.KINDS, s["kind"]
        stem, case = trunner.KINDS[s["kind"]](s)
        assert stem and case["kind"] and case["shape"]


def test_cublas_adapter_argv(monkeypatch):
    sys.path.insert(0, str(REPO / "suites/blackwell/runners/common"))
    monkeypatch.setenv("CYCLESCOPE_RUNNER_ROOT",
                       str(REPO / "suites/blackwell/runners"))
    monkeypatch.setenv("BW_CASES_DIR", str(REPO / "suites/blackwell/full_gpu"))
    import capture_adapters as ca
    case = ca.load_case("BLF-BF16-C05", root=REPO)
    assert case.suite == "cublas"
    ad = ca.get_adapter("cublas")
    res = ad.resolve(case, Path("/tmp/x"))
    assert res.argv[0].endswith("cublaslt_case")
    assert "--dtype" in res.argv and "bf16" in res.argv
    assert res.main_regex.pattern == ".*(nvjet|cutlass|xmma|gemm).*"


def test_cases_txt_and_spec_counts():
    ids = {l.split()[-1] for l in (REPO / "suites/blackwell/full_gpu/cases.txt")
           .read_text().splitlines() if l.split() and l.split()[0] == "cublas"}
    specs = {c["id"] for c in json.loads(
        (REPO / "suites/blackwell/full_gpu/cases_cublas.json")
        .read_text())["cases"]}
    assert ids == specs


def test_vendor_exclusion_suite_scoped():
    from cyclescope.capture.launch_selection import pick_repeat
    row = {"mangled": "nvjet_sm103_tst_x", "ordinal": 0,
           "sm__cycles_active.max": 5.0, "grid": [1], "block": [1]}
    rows = [dict(row) for _ in range(4)]
    assert "error" not in pick_repeat(rows, 4, "cublas")
    res = pick_repeat(rows, 4, "cutlass")
    assert res.get("error") == "no kernel launched once per call"


def _tsv_ids(path: Path, family: str) -> set:
    return {l.split("\t")[1] for l in path.read_text().splitlines()[1:]
            if l.split("\t") and l.split("\t")[0] == family}


def test_hopper_triton_full_gpu_specs_have_tsv_ids():
    fg = {s["id"] for s in json.loads(
        (REPO / "suites/hopper/full_gpu/cases_triton.json").read_text())}
    assert fg == _tsv_ids(REPO / "suites/hopper/full_gpu/cases.tsv", "triton")


def test_hopper_triton_num_sms_no_override_in_full_gpu():
    sys.path.insert(0, str(REPO / "suites/hopper/runners/triton"))
    import run as trunner
    for s in json.loads(
            (REPO / "suites/hopper/full_gpu/cases_triton.json")
            .read_text()):
        _, case = trunner.KINDS[s["kind"]](s)
        assert s["params"].get("num_sms") is None, s["id"]
        assert not case.get("single_cluster"), s["id"]
        assert case.get("mode") != "full", s["id"]
        if s["kind"] == "tutorial04_seeded_dropout":
            assert s["config"]["kwargs"]["BLOCK_SIZE"] == 1024, s["id"]
