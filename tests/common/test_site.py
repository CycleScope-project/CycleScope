"""cyclescope.site: env > site file > built-in default resolution."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cyclescope import site

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_env(monkeypatch, tmp_path):
    for k in ("CYCLESCOPE_ARTIFACT_ROOT", "CYCLESCOPE_SITE_CONFIG",
              "CYCLESCOPE_BUILD_DIR", "CYCLESCOPE_TRACER", "CYCLESCOPE_NCU",
              "NVDISASM"):
        monkeypatch.delenv(k, raising=False)
    site._cache.clear()
    yield tmp_path
    site._cache.clear()


def _site_file(tmp_path, doc):
    f = tmp_path / "site.json"
    f.write_text(json.dumps(doc))
    return f


def test_artifact_root_default_is_repo_artifacts(no_env):
    assert site.artifact_root() == REPO / "artifacts"


def test_artifact_root_site_file(no_env, monkeypatch, tmp_path):
    f = _site_file(tmp_path, {"artifact_root": "/opt/site-root"})
    monkeypatch.setenv("CYCLESCOPE_SITE_CONFIG", str(f))
    assert site.artifact_root() == Path("/opt/site-root")


def test_artifact_root_env_wins(no_env, monkeypatch, tmp_path):
    f = _site_file(tmp_path, {"artifact_root": "/opt/site-root"})
    monkeypatch.setenv("CYCLESCOPE_SITE_CONFIG", str(f))
    monkeypatch.setenv("CYCLESCOPE_ARTIFACT_ROOT", "/opt/env-root")
    assert site.artifact_root() == Path("/opt/env-root")


def test_artifact_root_relative_to_repo(no_env, monkeypatch, tmp_path):
    f = _site_file(tmp_path, {"artifact_root": "rel/root"})
    monkeypatch.setenv("CYCLESCOPE_SITE_CONFIG", str(f))
    assert site.artifact_root() == (REPO / "rel/root").resolve()


def test_unknown_key_rejected_note_accepted(no_env, monkeypatch,
                                            tmp_path):
    f = _site_file(tmp_path, {"bogus": 1})
    monkeypatch.setenv("CYCLESCOPE_SITE_CONFIG", str(f))
    site._cache.clear()
    with pytest.raises(ValueError):
        site.artifact_root()
    f.write_text(json.dumps({"_note": "hi",
                             "artifact_root": "/opt/r"}))
    site._cache.clear()
    assert site.artifact_root() == Path("/opt/r")


def test_build_dir_default_and_overrides(no_env, monkeypatch, tmp_path):
    assert site.build_dir() == REPO / "build"
    f = _site_file(tmp_path, {"build_dir": "/site/build"})
    monkeypatch.setenv("CYCLESCOPE_SITE_CONFIG", str(f))
    assert site.build_dir() == Path("/site/build")
    monkeypatch.setenv("CYCLESCOPE_BUILD_DIR", "/env/build")
    assert site.build_dir() == Path("/env/build")


def test_tracer_default_and_overrides(no_env, monkeypatch, tmp_path):
    assert site.tracer("sm_90a") == (
        REPO / "artifacts" / "build" / "tracer" / "sm_90a" / "tracer.so")
    f = _site_file(tmp_path,
                   {"tracer": {"sm_90a": "/site/tracer.so"}})
    monkeypatch.setenv("CYCLESCOPE_SITE_CONFIG", str(f))
    assert site.tracer("sm_90a") == Path("/site/tracer.so")
    monkeypatch.setenv("CYCLESCOPE_TRACER", "/env/tracer.so")
    assert site.tracer("sm_90a") == Path("/env/tracer.so")


def test_get_cli(no_env):
    out = subprocess.run(
        [sys.executable, "-m", "cyclescope.site", "--get", "artifact_root"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO / "python"), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0
    assert out.stdout.strip() == str(REPO / "artifacts")
