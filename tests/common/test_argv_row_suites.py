"""argv-row suites (cases.tsv) resolve the same way on both arches."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


_HEADER = ("set\tcase_id\tkernel_regex\toccurrence\targv_json\tcwd\t"
           "env_json\tenv_profile\ttma_recorder\tbaseline\tnote\n")


def _write_suite(repo: Path, arch: str, sset: str = "microbench") -> Path:
    d = repo / "suites" / arch / sset
    d.mkdir(parents=True)
    row = ("mb\tcaseX\tk_start_v3\t2\t"
           '[""/bin/true"", ""--cell"", ""b0.first""]\t-\t{}\t\t\t-\t\n')
    (d / "cases.tsv").write_text(_HEADER + row)
    (d / "launches.tsv").write_text(
        "case\tkernel\toccurrence\tgrid\tblock\tcluster\tsource\n"
        "caseX\t_Z9k_start_v312CellParamsPd\t2\t132x1x1\t128x1x1\t"
        "1x1x1\tprobe:kstart_v3\n")
    return d


def test_blackwell_cases_tsv_is_argv_row(tmp_path, monkeypatch):
    """suites/blackwell/<set>/cases.tsv resolves through the argv-row
    path: kernel_regex@occurrence selection, no hopper harness env."""
    from cyclescope.capture import driver
    repo = tmp_path / "r"
    _write_suite(repo, "blackwell")
    monkeypatch.setattr(driver, "REPO", repo)
    res = driver.resolve_case("blackwell/microbench", "caseX",
                              tmp_path / "w")
    assert res.arch == "blackwell"
    assert res.argv_row is True
    assert res.kernel_regex == "k_start_v3"
    assert res.occurrence == 2
    assert res.launch_key == ("_Z9k_start_v312CellParamsPd", 2)
    assert res.launch_row["grid"] == "132x1x1"
    assert res.argv == ["/bin/true", "--cell", "b0.first"]
    # the hopper harness PYTHONPATH rewrite is hopper-only
    assert "harness" not in res.env.get("PYTHONPATH", "")
    assert "PYTHONPATH" not in res.env


def test_hopper_still_argv_row(tmp_path, monkeypatch):
    """hopper resolves as before: argv_row=True, kernel_regex path."""
    from cyclescope.capture import driver
    repo = tmp_path / "r"
    _write_suite(repo, "hopper")
    monkeypatch.setattr(driver, "REPO", repo)
    res = driver.resolve_case("hopper/microbench", "caseX",
                              tmp_path / "w")
    assert res.arch == "hopper"
    assert res.argv_row is True
    assert res.kernel_regex == "k_start_v3"
    assert res.occurrence == 2
    assert res.launch_key == ("_Z9k_start_v312CellParamsPd", 2)


def test_blackwell_adapter_without_cases_tsv(tmp_path, monkeypatch):
    """a blackwell set with no cases.tsv keeps the adapter path
    (argv_row=False) — smoke: it must not read cases.tsv."""
    from cyclescope.capture import driver
    repo = tmp_path / "r"
    (repo / "suites" / "blackwell" / "single_cluster").mkdir(
        parents=True)
    monkeypatch.setattr(driver, "REPO", repo)
    try:
        driver.resolve_case("blackwell/single_cluster", "caseX",
                            tmp_path / "w")
    except Exception as e:
        # adapter path fails on the missing runners/cases.txt, never
        # on a cases.tsv lookup
        assert "cases.tsv" not in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected resolve to fail without adapters")
