"""--configs resolution: package-derived default, cwd-independent."""
import shutil
import subprocess
from pathlib import Path

import pytest

from cyclescope.cli import default_configs_dir

REPO = Path(__file__).resolve().parents[2]


def test_default_configs_dir_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("CYCLESCOPE_CONFIGS", raising=False)
    monkeypatch.chdir(tmp_path)
    d = Path(default_configs_dir())
    assert d.is_absolute() and d.is_dir()
    assert (d / "hardware" / "b300" /
            "preinitialized_mbarriers.json").is_file()


def test_default_configs_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CYCLESCOPE_CONFIGS", str(tmp_path / "cfg"))
    assert default_configs_dir() == str(tmp_path / "cfg")


def test_capture_facts_bad_configs_fails_closed(tmp_path):
    bin_ = shutil.which("cyclescope-capture-facts") or str(
        REPO / "build" / "cyclescope-capture-facts")
    if not Path(bin_).exists():
        pytest.skip("cyclescope-capture-facts not built")
    p = subprocess.run(
        [bin_, "--case", str(tmp_path), "--arch", "blackwell",
         "--out", str(tmp_path / "out"), "--launch", "0",
         "--configs", str(tmp_path / "nonexistent")],
        capture_output=True, text=True)
    assert p.returncode != 0
    assert "unsupported:configs" in p.stderr
