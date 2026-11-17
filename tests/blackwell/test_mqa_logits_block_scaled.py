"""The blackwell full_gpu suite set resolves through the deepgemm runner."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

RUN = REPO / "suites/blackwell/runners/deepgemm/run.py"


def _resolve(cases_dir, *ids):
    env = dict(os.environ)
    env["BW_CASES_DIR"] = str(cases_dir)
    env["CYCLESCOPE_RUNNER_ROOT"] = str(REPO / "suites/blackwell/runners")
    out = subprocess.run(
        [sys.executable, str(RUN), "--resolve", *ids],
        env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return [l for l in out.stdout.splitlines() if l.startswith("RESOLVE ")]


def test_default_sets_still_resolve():
    sset = "full_gpu"
    specs = json.loads(
        (REPO / "suites/blackwell" / sset / "cases_deepgemm.json")
        .read_text())["cases"]
    lines = _resolve(REPO / "suites/blackwell" / sset)
    assert len(lines) == len(specs)
