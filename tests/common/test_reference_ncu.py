"""Cold-reference capture side: ncu constants, hopper pick, tracer env."""
import json
import shutil
from pathlib import Path

import pytest

from cyclescope.capture import driver
from cyclescope.capture.driver import Resolved
from cyclescope.reference import ncu_runner


def _hopper(kernel_regex="nvjet", occurrence=2):
    return Resolved(argv=[], env={}, cwd=None, arch="hopper",
                    kernel_regex=kernel_regex, occurrence=occurrence,
                    argv_row=True)


def _row(ordinal, mangled, cycles=None):
    return {"ordinal": ordinal, "mangled": mangled,
            "sm__cycles_active.max": cycles,
            "grid": [1, 1, 1], "block": [1, 1, 1], "cluster": [1, 1, 1]}


def test_pick_hopper_occurrence_counts_matching_rows():
    rows = [_row(0, "nvjet_a"), _row(1, "other_a"), _row(2, "nvjet_b"),
            _row(3, "other_b"), _row(4, "nvjet_c"), _row(5, "other_c")]
    pick = ncu_runner._pick(rows, _hopper(occurrence=2))
    assert pick["mangled"] == "nvjet_b"
    pick = ncu_runner._pick(rows, _hopper(occurrence=None))
    assert pick["mangled"] == "nvjet_a"
    assert ncu_runner._pick(rows, _hopper(occurrence=4)) is None


def _ncu_rec(name, cycles):
    rec = {"Kernel Name": name, "sm__cycles_active.max": str(cycles),
           "sm__cycles_elapsed.avg.per_second": "1980000000"}
    for m in driver.NCU_LAUNCH_METRICS:
        rec.setdefault(m, "1")
    return rec


def test_run_reference_writes_cold_columns(tmp_path, monkeypatch):
    cycles_by_rep = [100, 300, 200]
    calls = {"n": 0}

    def fake_run_once(argv, env, cwd, log, timeout_s, metrics=None):
        c = cycles_by_rep[calls["n"] % len(cycles_by_rep)]
        calls["n"] += 1
        return 0, [_ncu_rec("nvjet_a", 10), _ncu_rec("other", 20),
                   _ncu_rec("nvjet_b", c)]

    monkeypatch.setattr(ncu_runner, "resolve_case",
                        lambda suite, cid, work: _hopper(occurrence=2))
    monkeypatch.setattr(ncu_runner, "_run_once", fake_run_once)
    monkeypatch.setattr(ncu_runner, "_gpu_info", lambda: {"gpu": "x"})
    monkeypatch.setattr(ncu_runner, "_ncu_version", lambda: "ncu v1")
    monkeypatch.setattr(ncu_runner, "_ncu_binary", lambda: "/x/ncu")

    rec = ncu_runner.run_reference("hopper/full_gpu", ["caseX"],
                                   tmp_path, repeats=3)
    assert rec["statuses"] == ["success"]

    lines = (tmp_path / "reference.tsv").read_text().splitlines()
    cols = lines[1].split("\t")
    assert cols[12] == "all"    # cache_control
    assert cols[13] == "none"   # clock_control
    assert cols[6] == "200"     # median of 100/300/200
    assert cols[2] == "2"       # census ordinal of the 2nd nvjet

    timing = json.loads((tmp_path / "caseX" / "timing.json").read_text())
    assert timing["cache_control"] == "all"
    assert timing["clock_control"] == "none"
    assert timing["ncu"] == "/x/ncu"
    assert timing["ncu_version"] == "ncu v1"
    assert timing["pick"]["cycles"] == [100, 300, 200]


@pytest.mark.parametrize("rc,status", [(9, "ncu_failed:rc=9"),
                                       (124, "ncu_timeout")])
def test_run_reference_names_ncu_failure(tmp_path, monkeypatch, rc,
                                         status):
    # empty census from failed ncu repeats (driver/ncu mismatch, hang)
    # is an environment error, not a traced_launch_not_in_census miss
    monkeypatch.setattr(ncu_runner, "resolve_case",
                        lambda suite, cid, work: _hopper(occurrence=1))
    monkeypatch.setattr(ncu_runner, "_run_once",
                        lambda *a, **k: (rc, []))
    monkeypatch.setattr(ncu_runner, "_gpu_info", lambda: {"gpu": "x"})
    monkeypatch.setattr(ncu_runner, "_ncu_version", lambda: "ncu v1")
    monkeypatch.setattr(ncu_runner, "_ncu_binary", lambda: "/x/ncu")
    cap = tmp_path / "captures"
    (cap / "caseX").mkdir(parents=True)
    rec = ncu_runner.run_reference("hopper/full_gpu", ["caseX"],
                                   tmp_path / "ref", repeats=3,
                                   captures=cap)
    assert rec["statuses"] == [status]
    timing = json.loads((tmp_path / "ref" / "caseX" / "timing.json")
                        .read_text())
    assert timing["status"] == status
    assert timing["repeat_rcs"] == [rc, rc, rc]


def test_run_reference_reuses_capture_scratch(tmp_path, monkeypatch):
    seen = {}

    def fake_run_once(argv, env, cwd, log, timeout_s, metrics=None):
        seen["argv"] = argv
        return 0, [_ncu_rec("nvjet_a", 10)]

    r = Resolved(argv=["app", "--out", "<CAPTURE>/scratch/r.json"], env={},
                 cwd=None, arch="hopper", kernel_regex="nvjet", occurrence=1,
                 argv_row=True)
    monkeypatch.setattr(ncu_runner, "resolve_case", lambda suite, cid, work: r)
    monkeypatch.setattr(ncu_runner, "_run_once", fake_run_once)
    monkeypatch.setattr(ncu_runner, "_gpu_info", lambda: {"gpu": "x"})
    monkeypatch.setattr(ncu_runner, "_ncu_version", lambda: "ncu v1")
    monkeypatch.setattr(ncu_runner, "_ncu_binary", lambda: "/x/ncu")
    monkeypatch.setattr(ncu_runner, "traced_launch_key",
                        lambda d: ("nvjet_a", 1))
    cap = tmp_path / "captures" / "caseX"
    (cap / "scratch").mkdir(parents=True)
    rec = ncu_runner.run_reference("hopper/full_gpu", ["caseX"],
                                   tmp_path / "ref", repeats=1,
                                   captures=tmp_path / "captures")
    assert rec["statuses"] == ["success"]
    # <CAPTURE> resolves to the capture dir (JIT caches reused), not work
    assert seen["argv"][-1] == str(cap / "scratch" / "r.json")
    timing = json.loads((tmp_path / "ref" / "caseX" / "timing.json")
                        .read_text())
    assert timing["scratch_root"] == str(cap)


def test_run_group_kills_process_group(tmp_path):
    from cyclescope.capture.ncu_tools import TIMEOUT_RC, run_group
    import subprocess
    pidfile = tmp_path / "pid"
    rc = run_group(["bash", "-c", f"sleep 60 & echo $! > {pidfile}; wait"],
                   timeout_s=1)
    assert rc == TIMEOUT_RC
    pid = int(pidfile.read_text())
    assert subprocess.run(["kill", "-0", str(pid)],
                          capture_output=True).returncode != 0
    assert run_group(["bash", "-c", "exit 3"], timeout_s=5) == 3


def test_case_command_capture_and_far(tmp_path):
    r = Resolved(argv=["app", "--out", "<CAPTURE>/res.json"], env={},
                 cwd=None, arch="hopper", kernel_regex="k",
                 occurrence=1, argv_row=True)
    argv, env = driver.case_command(r, tmp_path / "t", {"A": "1"})
    assert argv == ["app", "--out", str(tmp_path / "t") + "/res.json"]
    assert env == {"A": "1", "CYCLESCOPE_RUNNER_ROOT": env["CYCLESCOPE_RUNNER_ROOT"]}
    assert env["CYCLESCOPE_RUNNER_ROOT"].endswith("suites/hopper/runners")

    far = Resolved(argv=["app", "x"], env={}, cwd=None, arch="hopper",
                   env_profile="far")
    argv, env = driver.case_command(far, tmp_path, {})
    assert argv[0] == "bash" and argv[1] == "-c"
    assert "environment_far.sh" in argv[2] and 'exec "$@"' in argv[2]
    assert argv[3] == "_" and argv[4:] == ["app", "x"]
    assert env["CYCLESCOPE_RUNNER_ROOT"].endswith(
        "suites/hopper/runners")


def test_expand_colon_segments(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLESCOPE_ARTIFACT_ROOT", str(tmp_path))
    v = ("$CYCLESCOPE_ROOT/x:$CYCLESCOPE_ARTIFACT_ROOT/tools/y:/plain/path")
    out = driver._expand(v)
    A = str(tmp_path)
    assert out == (f"{driver.REPO}/x:{A}/tools/y:/plain/path")
    assert driver._expand("/plain/path") == "/plain/path"


def test_resolve_case_mqa_pythonpath(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLESCOPE_ARTIFACT_ROOT", str(tmp_path))
    r = driver.resolve_case("hopper/full_gpu",
                            "deepgemm_mqa_logits_02", tmp_path)
    A = str(tmp_path)
    harness = str(driver.REPO / "suites" / "hopper" / "runners" / "harness")
    assert r.env["PYTHONPATH"].split(":") == [
        harness,
        f"{harness}/suites/deepgemm/harness",
        harness]


def test_ncu_binary_env_and_missing(monkeypatch):
    monkeypatch.setenv("CYCLESCOPE_NCU", "/opt/ncu/bin/ncu")
    assert ncu_runner._ncu_binary() == "/opt/ncu/bin/ncu"
    monkeypatch.delenv("CYCLESCOPE_NCU")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(FileNotFoundError):
        ncu_runner._ncu_binary()


def test_tracer_env_pins_l2_footprint_off(monkeypatch):
    monkeypatch.setenv("CYCLESCOPE_L2_FOOTPRINT", "1")
    env = driver.tracer_env({"PATH": "/bin", "CYCLESCOPE_L2_FOOTPRINT": "1"},
                            Path("/t/tracer.so"), Path("/out"),
                            "nvjet@2")
    assert env["CYCLESCOPE_L2_FOOTPRINT"] == "0"
    assert env["CUDA_INJECTION64_PATH"] == "/t/tracer.so"
    assert env["CYCLESCOPE_CAPTURE_LAUNCHES"] == "nvjet@2"


def test_tracer_env_leaves_module_loading_to_nvbit():
    env = driver.tracer_env({"PATH": "/bin"}, Path("/t/tracer.so"),
                            Path("/out"), "k@1")
    assert "NO_EAGER_LOAD" not in env
    env = driver.tracer_env({"PATH": "/bin", "NO_EAGER_LOAD": "0"},
                            Path("/t/tracer.so"), Path("/out"), "k@1")
    assert env["NO_EAGER_LOAD"] == "0"


def test_run_env_json_redacts_secret_keys(tmp_path):
    out = tmp_path / "cap"
    r = Resolved(argv=["true"], env={}, cwd=None, arch="hopper",
                 kernel_regex="k", argv_row=True)
    env = {"PATH": "/usr/bin:/bin",
           "AWS_ACCESS_KEY_ID": "AKIA-SECRET",
           "AWS_SECRET_ACCESS_KEY": "shhh",
           "MY_API_KEY": "k1", "GITHUB_TOKEN": "t",
           "CUDA_PRIVATE_THING": "p", "NORMAL_VAR": "visible"}
    driver._traced_run(out, r, env, Path("/t/tracer.so"), "k", "trace",
                       30)
    recorded = json.loads((out / "run_env.json").read_text())
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "MY_API_KEY", "GITHUB_TOKEN", "CUDA_PRIVATE_THING"):
        assert recorded[key] == "<redacted>", key
    assert recorded["NORMAL_VAR"] == "visible"


def test_run_env_json_skips_unrelated_parent_var(tmp_path, monkeypatch):
    out = tmp_path / "cap"
    monkeypatch.setenv("UNRELATED_PARENT_VAR", "same-value")
    r = Resolved(argv=["true"], env={}, cwd=None, arch="hopper",
                 kernel_regex="k")
    env = {"UNRELATED_PARENT_VAR": "same-value",
           "CYCLESCOPE_EXTRA": "set-by-run"}
    driver._traced_run(out, r, env, Path("/t/tracer.so"), "k", "trace",
                       30)
    recorded = json.loads((out / "run_env.json").read_text())
    assert "UNRELATED_PARENT_VAR" not in recorded
    assert recorded["CYCLESCOPE_EXTRA"] == "set-by-run"


def test_placement_failure_fails_capture(tmp_path, monkeypatch):
    r = Resolved(argv=["true"], env={}, cwd=None, arch="hopper",
                 argv_row=True)
    monkeypatch.setattr(driver, "resolve_case", lambda *a, **k: r)
    monkeypatch.setattr(driver, "prepare_case",
                        lambda *a, **k: {"status": "not_needed", "rc": 0})
    monkeypatch.setattr(driver, "_ncu_launch_pass",
                        lambda *a, **k: {"status": "ok", "launches": 0})

    def fake_run(out, resolved, env, tracer, selection, mode, timeout_s):
        return {"rc": -6 if mode == "placement" else 0,
                "wall_s": 0.0, "mode": mode}

    monkeypatch.setattr(driver, "_traced_run", fake_run)
    rec = driver.capture_case("hopper/full_gpu", "caseX",
                              tmp_path / "cap", mode="both")
    assert rec["status"] == "placement_failed:rc-6"
    assert rec["rc"] == 1
    saved = json.loads((tmp_path / "cap" / "run.json").read_text())
    assert saved["status"] == "placement_failed:rc-6"
