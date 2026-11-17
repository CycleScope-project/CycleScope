#!/usr/bin/env python3
"""Safety checks for delivery builds that read frozen source trees."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat


def assert_separate_source_and_output(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.resolve()
    output = output.resolve(strict=False)
    if not source.is_dir():
        raise ValueError(f"delivery source is not a directory: {source}")
    if output == source or output.is_relative_to(source):
        raise ValueError(f"delivery output must be separate from source: source={source} output={output}")
    return source, output


def make_owner_writable_work_tree(tree: Path, *, source: Path) -> None:
    source, tree = assert_separate_source_and_output(source, tree)
    if not tree.is_dir() or tree.is_symlink():
        raise ValueError(f"work tree must be a real directory: {tree}")
    for directory, names, files in os.walk(tree):
        directory_path = Path(directory)
        directory_path.chmod(stat.S_IMODE(directory_path.stat().st_mode) | stat.S_IWUSR)
        for name in (*names, *files):
            path = directory_path / name
            if path.is_symlink():
                resolved = path.resolve()
                if not resolved.is_relative_to(tree) or not resolved.exists():
                    raise ValueError(f"work-tree symlink escapes or is missing: {path} -> {os.readlink(path)}")
                continue
            path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--source", required=True, type=Path)
    check.add_argument("--output", required=True, type=Path)
    writable = subparsers.add_parser("make-owner-writable")
    writable.add_argument("--source", required=True, type=Path)
    writable.add_argument("--tree", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "check":
        assert_separate_source_and_output(args.source, args.output)
    else:
        make_owner_writable_work_tree(args.tree, source=args.source)


if __name__ == "__main__":
    main()
