"""Merge per-shard converter outputs into one directory; shard sizing.

    merge(out, shard_dirs, segments=False)

facts (default): each shard dir holds ``launch.json`` + ``<cluster>.gsc``
(``cyclescope convert --capture --shard k/n``).  --segments: each shard dir
holds ``launch.json`` + ``segments/<cluster>.seg`` (``convert --facts`` on a
shard's facts).  The merged ``launch.json`` is shard 0's launch-level
document with the shard-dependent fields combined: ``dropped_polls`` /
``missing_facts`` / ``unsupported`` summed, ``placements`` united,
``cluster_ids`` = union (None when every shard covered the full sampled
set), ``segments`` = sorted union.  Files are hard-linked (or copied); a
cluster appearing in two shards is a format error.

``record_estimate`` is the trace record estimate used to size the
capture->facts split; ``shard_count`` is the ceil/cap formula applied to it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

BPR = int(os.environ.get("TRACE_BYTES_PER_RECORD", "4"))


def record_estimate(capture_dir: Path) -> tuple[int, str]:
    """Estimated trace record count of a capture and its source.

    Priority:
      1. `records: ... total=N` in trace_check.txt (exact, full check);
      2. else total bytes of trace-*.bin.zst // BPR (default 4).  470
         captures with full checks measured 3.5 / 4.4 / 7.7 compressed
         bytes per record (p10 / p50 / p90); 4 overestimates most traces
         (more shards) and underestimates only the most compressible
         ones (below 4 B per record);
      3. else 0 ("none").
    """
    d = Path(capture_dir)
    n, src = 0, "none"
    tc = d / "trace_check.txt"
    if tc.is_file():
        m = re.search(r"total=(\d+)", tc.read_text(errors="replace"))
        if m:
            n, src = int(m.group(1)), "total"
    if not n:
        b = sum(f.stat().st_size for f in d.glob("trace-*.bin*"))
        if b:
            n, src = b // BPR, "bytes"
    return n, src


def shard_count(records: int, shard_records: int, shard_max: int) -> int:
    """ceil(records / shard_records) capped at shard_max, min 1;
    1 when records or shard_records <= 0."""
    nsh = 1
    if records > 0 and shard_records > 0:
        nsh = (records + shard_records - 1) // shard_records
        if nsh > shard_max:
            nsh = shard_max
        if nsh < 1:
            nsh = 1
    return nsh


def merge(out: Path, shards: list[Path], segments: bool = False) -> dict:
    out = Path(out)
    shards = [Path(s) for s in shards]
    out.mkdir(parents=True, exist_ok=True)
    sub, suffix = ("segments", ".seg") if segments else ("", ".gsc")
    (out / sub).mkdir(parents=True, exist_ok=True)
    docs = [json.loads((s / "launch.json").read_text()) for s in shards]
    merged = dict(docs[0])
    merged["dropped_polls"] = sum(int(d.get("dropped_polls") or 0)
                                  for d in docs)
    for key in ("missing_facts", "unsupported"):
        acc: dict = {}
        for d in docs:
            for k, v in (d.get(key) or {}).items():
                acc[k] = acc.get(k, 0) + int(v)
        merged[key] = acc
    placements: dict = {}
    for d in docs:
        placements.update(d.get("placements") or {})
    merged["placements"] = placements
    ids: set = set()
    full = False
    for d in docs:
        if d.get("cluster_ids") is None:
            full = True
        else:
            ids.update(int(c) for c in d["cluster_ids"])
    merged["cluster_ids"] = None if full else sorted(ids)
    merged["shards"] = len(shards)
    if segments:
        merged["segments"] = sorted({n for d in docs
                                     for n in (d.get("segments") or [])})
    seen: set = set()
    for s in shards:
        for g in sorted((s / sub).glob("*" + suffix)):
            if g.name == "statics.gsc":
                continue   # shared per-dir bundle, merged below
            if g.name in seen:
                raise SystemExit(f"cluster {g.name} present in two shards")
            seen.add(g.name)
            dst = out / sub / g.name
            if dst.exists():
                dst.unlink()
            try:
                os.link(g, dst)
            except OSError:
                shutil.copy2(g, dst)
    (out / "launch.json").write_text(json.dumps(merged, indent=1,
                                                sort_keys=True))
    return merged


def merge_shards(out: Path, shards: list[Path],
                 segments: bool = False) -> dict:
    """``merge`` plus the cross-shard bundle checks the CLI driver did."""
    m = merge(out, shards, segments)
    if segments:
        # segment-v2 program tables: identical across shards
        ref = shards[0] / "segments" / "program.tables"
        if ref.is_file():
            for s_ in shards[1:]:
                o = s_ / "segments" / "program.tables"
                if not o.is_file() or o.read_bytes() != ref.read_bytes():
                    raise SystemExit(
                        f"program.tables mismatch: {o} vs {ref}")
            shutil.copy2(ref, Path(out) / "segments" / "program.tables")
    else:
        # GSIMCAP02 statics bundle: identical across shards by
        # construction; take shard 0's and verify
        ref = shards[0] / "statics.gsc"
        if ref.is_file():
            for s_ in shards[1:]:
                o = s_ / "statics.gsc"
                if not o.is_file() or o.read_bytes() != ref.read_bytes():
                    raise SystemExit(
                        f"statics.gsc mismatch: {o} vs {ref}")
            shutil.copy2(ref, Path(out) / "statics.gsc")
    return m
