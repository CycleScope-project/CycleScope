"""Content hash of a directory tree (see docs/en/capture-tools.md).

tree_sha256(root): collect every entry under root (regular files and
symlinks, including symlinks pointing at directories), sort globally by
relative POSIX path, then feed "<relpath>\0<payload>\n" lines into one
sha256 and return its hexdigest.  Payload is the file sha256 for regular
files and "symlink:<target>" for symlinks; links are never followed.
__pycache__ directories and *.pyc files are skipped (regenerated
artifacts).  Missing paths return None.
"""
from __future__ import annotations

import hashlib
import os


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root) -> str | None:
    root = os.fspath(root)
    if not os.path.isdir(root):
        return None
    entries = []  # (relpath_posix, payload)
    for dirpath, dirnames, filenames in os.walk(
            root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        # symlinks to directories appear in dirnames: record, don't walk
        kept = []
        for d in dirnames:
            fp = os.path.join(dirpath, d)
            rel = os.path.normpath(os.path.join(rel_dir, d))
            if os.path.islink(fp):
                entries.append((rel.replace(os.sep, "/"),
                                "symlink:" + os.readlink(fp)))
            elif d != "__pycache__":
                kept.append(d)
        dirnames[:] = sorted(kept)
        for fn in filenames:
            if fn.endswith(".pyc"):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.normpath(os.path.join(rel_dir, fn))
            rel = rel.replace(os.sep, "/")
            if os.path.islink(fp):
                entries.append((rel, "symlink:" + os.readlink(fp)))
            else:
                entries.append((rel, _file_sha256(fp)))
    h = hashlib.sha256()
    for rel, payload in sorted(entries):
        h.update(rel.encode() + b"\0" + payload.encode() + b"\n")
    return h.hexdigest()
