"""Segment cache (docs/en/design/job-format.md).

Entry name: ``<sha256(capture|cluster|src)[:32]>-<tree[:16]>.seg`` so a
tree-only change can find the same-input sibling by prefix.  ``src`` is
the converter source digest (``python/cyclescope/{convert,job,arch}/**/*.py``
+ ``cpp/{convert,arch/*/convert_rules}`` sources), ``tree`` is the
generated tree's ``content_digest``.  Hits reuse bytes; a prefix-only
match relinks.  Writes are atomic (pid-tagged tmp + rename); every entry
carries a ``<name>.meta.json`` with the cleartext components.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def converter_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for sub in ("convert", "job", "arch"):
        for f in sorted((root / sub).rglob("*.py")):
            h.update(f.read_bytes())
    cpp = root.parents[1] / "cpp"
    for f in sorted((cpp / "convert").glob("*.cpp")) + \
            sorted((cpp / "convert").glob("*.hpp")) + \
            sorted(cpp.glob("arch/*/convert_rules.cpp")):
        h.update(f.read_bytes())
    return h.hexdigest()


def entry_prefix(capture_id: str, cluster_id: int, src_digest: str) -> str:
    return hashlib.sha256(
        f"{capture_id}|{cluster_id}|{src_digest}".encode()
    ).hexdigest()[:32]


def entry_name(capture_id: str, cluster_id: int, src_digest: str,
               tree_digest: str) -> str:
    return (f"{entry_prefix(capture_id, cluster_id, src_digest)}"
            f"-{tree_digest[:16]}.seg")


class SegmentCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, name: str) -> bytes | None:
        p = self.root / name
        if not p.exists():
            return None
        return p.read_bytes()

    def find_for_relink(self, capture_id: str, cluster_id: int,
                        src_digest: str) -> tuple[Path, bytes] | None:
        """Newest same-prefix entry under a different tree digest."""
        prefix = entry_prefix(capture_id, cluster_id, src_digest)
        best = None
        for p in self.root.glob(f"{prefix}-*.seg"):
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
        if best is None:
            return None
        return best, best.read_bytes()

    def put(self, name: str, data: bytes, meta: dict) -> Path:
        tmp = self.root / f".{name}.{os.getpid()}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, self.root / name)
        (self.root / f"{name}.meta.json").write_text(
            json.dumps(meta, indent=1))
        return self.root / name
