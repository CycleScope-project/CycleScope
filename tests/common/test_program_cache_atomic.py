"""Concurrent .gsp writers: file ends complete, no temp files left."""

import os
import secrets
import subprocess
import sys
from pathlib import Path


def _write_once(target: Path):
    """Mirror of pipeline.py's atomic gsp write."""
    data = b"x" * 4096
    t = target.with_name(
        f"{target.stem}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    t.write_bytes(data)
    try:
        os.link(t, target)
        t.unlink()
    except FileExistsError:
        t.unlink()
    except OSError:
        os.replace(t, target)


def test_concurrent_writes(tmp_path):
    worker = f"""
import sys
sys.path.insert(0, {os.getcwd()!r})
from pathlib import Path
from tests.common.test_program_cache_atomic import _write_once
for _ in range(50):
    _write_once(Path({str(tmp_path)!r}) / 'k.gsp')
"""
    procs = [subprocess.Popen([sys.executable, "-c", worker],
                              env={**os.environ, "PYTHONPATH": "python"})
             for _ in range(4)]
    for p in procs:
        assert p.wait() == 0
    g = tmp_path / "k.gsp"
    assert g.exists() and g.read_bytes() == b"x" * 4096
    assert not list(tmp_path.glob("*.tmp"))
