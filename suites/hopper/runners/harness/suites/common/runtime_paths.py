#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import tempfile


RUNTIME_VARIABLES = (
    "HOPPER_CAPTURE_HOME",
    "XDG_CACHE_HOME",
    "CUDA_CACHE_PATH",
    "TMPDIR",
    "PIP_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
)


def _reject_symlink_components(task_root: Path, candidate: Path) -> None:
    relative = candidate.relative_to(task_root)
    current = task_root
    for component in relative.parts:
        current = current / component
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"runtime path contains a symlink: {current}")


def validate_runtime_paths(
    *, task_root: Path, source_root: Path, environment: dict[str, str]
) -> dict[str, object]:
    task_root = task_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    runtime_root = (task_root / "runtime").resolve(strict=True)
    runtime_root.relative_to(task_root)
    rows = []
    seen: set[Path] = set()
    for name in RUNTIME_VARIABLES:
        raw = environment.get(name, "")
        if not raw or not Path(raw).is_absolute():
            raise ValueError(f"{name} must name a non-empty absolute directory")
        path = Path(raw)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{name} cannot be resolved as a runtime directory: {path}") from exc
        if not resolved.is_dir():
            raise ValueError(f"{name} is not a directory: {resolved}")
        try:
            resolved.relative_to(runtime_root)
        except ValueError as exc:
            raise ValueError(f"{name} escapes the task runtime root: {resolved}") from exc
        try:
            resolved.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise ValueError(f"{name} points into the immutable source: {resolved}")
        _reject_symlink_components(task_root, path.absolute())
        if resolved in seen:
            raise ValueError(f"runtime variables alias the same directory: {resolved}")
        seen.add(resolved)
        rows.append({"name": name, "path": str(resolved)})
    return {
        "schema_version": "hopper.capture-runtime-paths.v1",
        "status": "pass",
        "task_root": str(task_root),
        "runtime_root": str(runtime_root),
        "source_root": str(source_root),
        "paths": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = validate_runtime_paths(
        task_root=args.task_root, source_root=args.source_root, environment=dict(os.environ)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, args.output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
