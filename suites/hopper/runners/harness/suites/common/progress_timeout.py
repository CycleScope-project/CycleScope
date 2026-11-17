"""Run a capture process with an inactivity lease and a hard absolute limit."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Iterable, Sequence


ProgressSnapshot = tuple[int, int, int]


def _is_capture_progress_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".log") or name.endswith(".progress.jsonl"):
        return False
    return (
        "trace" in name
        or "map" in name
        or path.suffix.lower() in {".csv", ".jsonl", ".json", ".bin", ".zip", ".gz", ".cubin", ".so", ".ptx", ".sass", ".cu"}
    )


def capture_progress_snapshot(paths: Iterable[Path]) -> ProgressSnapshot:
    count = 0
    size = 0
    newest_mtime_ns = 0
    seen: set[Path] = set()
    for raw in paths:
        root = Path(raw)
        candidates = [root] if root.is_file() else root.rglob("*") if root.exists() else []
        for path in candidates:
            try:
                if path in seen or not path.is_file() or path.is_symlink():
                    continue
                seen.add(path)
                if not _is_capture_progress_file(path):
                    continue
                metadata = path.stat()
            except FileNotFoundError:
                continue
            count += 1
            size += metadata.st_size
            newest_mtime_ns = max(newest_mtime_ns, metadata.st_mtime_ns)
    return count, size, newest_mtime_ns


@dataclass
class ProgressLease:
    started_at: float
    inactivity_timeout: float
    absolute_timeout: float
    last_progress_at: float | None = None
    snapshot: ProgressSnapshot | None = None

    def observe(self, now: float, snapshot: ProgressSnapshot) -> str | None:
        if self.last_progress_at is None:
            self.last_progress_at = self.started_at
            self.snapshot = snapshot
        elif snapshot != self.snapshot:
            self.snapshot = snapshot
            self.last_progress_at = now
        if now - self.started_at >= self.absolute_timeout:
            return "absolute_timeout"
        if now - self.last_progress_at >= self.inactivity_timeout:
            return "inactivity_timeout"
        return None


class ProgressTimeoutError(RuntimeError):
    def __init__(self, reason: str, command: Sequence[object], log_path: Path):
        super().__init__(f"capture process reached {reason}; log={log_path}")
        self.reason = reason
        self.command = tuple(str(item) for item in command)
        self.log_path = log_path


def run_with_progress(
    command: Sequence[object],
    *,
    watched_paths: Iterable[Path],
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    inactivity_timeout: float = 600,
    absolute_timeout: float = 21600,
    poll_interval: float = 10,
) -> subprocess.CompletedProcess[str]:
    if inactivity_timeout <= 0 or absolute_timeout <= 0 or poll_interval <= 0:
        raise ValueError("capture timeout values must be positive")
    if inactivity_timeout > absolute_timeout:
        raise ValueError("inactivity timeout cannot exceed absolute timeout")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = log_path.with_suffix(log_path.suffix + ".progress.jsonl")
    started = time.monotonic()
    lease = ProgressLease(started, inactivity_timeout, absolute_timeout)

    def record(status: str, now: float, snapshot: ProgressSnapshot, **extra: object) -> None:
        row = {
            "status": status,
            "elapsed_seconds": now - started,
            "file_count": snapshot[0],
            "total_bytes": snapshot[1],
            "newest_mtime_ns": snapshot[2],
            **extra,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    watched = [Path(path) for path in watched_paths]
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=cwd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        initial = capture_progress_snapshot(watched)
        lease.observe(started, initial)
        record("started", started, initial, pid=process.pid)
        reason: str | None = None
        while process.poll() is None:
            try:
                process.wait(timeout=min(poll_interval, max(0.05, inactivity_timeout)))
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            snapshot = capture_progress_snapshot(watched)
            previous = lease.snapshot
            reason = lease.observe(now, snapshot)
            if snapshot != previous:
                record("progress", now, snapshot)
            if reason is not None:
                record("timeout", now, snapshot, reason=reason)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                break
        returncode = process.wait()
    output = log_path.read_text(encoding="utf-8", errors="replace")
    finished = time.monotonic()
    final_snapshot = capture_progress_snapshot(watched)
    record("finished", finished, final_snapshot, returncode=returncode)
    if reason is not None:
        raise ProgressTimeoutError(reason, command, log_path)
    return subprocess.CompletedProcess(
        [str(item) for item in command], returncode, output, None
    )
