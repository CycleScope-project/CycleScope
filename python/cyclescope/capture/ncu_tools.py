"""ncu binary location + reference-condition constants.

Shared by the capture census (``capture.driver``) and the reference
runner (``reference.ncu_runner``) so both sides stay in one module and
cannot diverge.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

CACHE_CONTROL = "all"
CLOCK_CONTROL = "none"

TIMEOUT_RC = 124


def run_group(cmd: list, *, timeout_s: float, **popen_kw) -> int:
    """subprocess.run with a timeout that kills the whole process group.

    ``subprocess.run(timeout=)`` only kills the direct child; an ncu-
    injected target or JIT compiler children survive as orphans.  An
    orphaned ncu target keeps the GPU profiler lock, and every later
    ncu session in the pod blocks at "Connected to process" until its
    own timeout (hp-fg decode_paged/einsum after a moe_gdn timeout).
    Returns the exit code, or TIMEOUT_RC after SIGKILL of the group."""
    proc = subprocess.Popen(cmd, start_new_session=True, **popen_kw)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return TIMEOUT_RC
    return proc.returncode


def ncu_binary() -> str:
    from .. import site
    env = site.ncu()
    if env:
        return env
    found = shutil.which("ncu")
    if found:
        return found
    default = "/usr/local/cuda/bin/ncu"
    if Path(default).is_file():
        return default
    raise FileNotFoundError("ncu not found: set CYCLESCOPE_NCU")
