"""Streaming reader of trace-<launch>.bin.zst record streams.

Decompression goes through a `zstd -dc` subprocess pipe read in chunks
(no whole-file buffer, no zstandard module dependency).  Record lengths
come from the static layout in functions/<function_id>.tsv
(captured_operands + mref_count), so records outside a cluster filter
are skipped by length only.  AFTER records attach to the most recent
unpaired same-idx instruction record of their (cta, warp); GSYNC records
attach by the same rule per phase.
"""
from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

HEAD = struct.Struct("<IHHHBBIIIII")  # 32 bytes, matches V5Head
FILE_HEADER_SIZE = 16
FILE_MAGIC = b"GSIMTRC\x00"
KIND_INSTR, KIND_AFTER, KIND_GSYNC, KIND_CTA = 0, 1, 2, 3
_CTA_BODY = struct.Struct("<10I")  # V5CtaBody: ctarank + nctaid +
                                  # clusterid + nclusterid

_DESCRIPTOR_OPS = frozenset({"IDESC", "GDESC", "GDESCH", "TMEMD"})

_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_TAIL4 = struct.Struct("<4B")


def is_ur_token(op: str) -> bool:
    return op.startswith("UR") and not op.startswith("URANK")


def _op_entry_len(op: str, exec_count: int) -> int:
    """Tail bytes of one operand entry (matches the tracer's writers)."""
    if op == "PRED" or op.startswith("BRVAL"):
        return 0
    if op == "SPACE":
        return 12
    if op.startswith(("URANK", "TCOORD")) or op == "DESC" or \
            op in ("CPOL", "CPOLH") or op.startswith("UR") or \
            op in _DESCRIPTOR_OPS:
        return 4
    # RANK<k>, BRANK<k>, R<n>: per executing lane 4 bytes
    return 4 * exec_count


def record_tail_len(ops: list[str], mref_count: int, exec_count: int) -> int:
    return 8 * mref_count * exec_count + sum(
        _op_entry_len(op, exec_count) for op in ops)


@dataclass(slots=True)
class DynRecord:
    launch: int
    cta: tuple[int, int, int]
    warp: int
    sm: int
    warp_slot: int
    function_id: int
    idx: int
    active_mask: int
    exec_mask: int
    mrefs: list[tuple[int, ...]] = field(default_factory=list)
    ops: dict = field(default_factory=dict)
    after_ready: int | None = None
    gsync: list[tuple[int, int, int, list[tuple[int, ...]], int]] = \
        field(default_factory=list)
    # (phase, nwords, flags, words, sequence)

    @property
    def exec_count(self) -> int:
        return self.exec_mask.bit_count()


def load_static(case_dir: Path):
    """function_id -> {idx -> (sass, ops, mref_count)}, id -> name."""
    case_dir = Path(case_dir)
    layouts: dict[int, dict[int, tuple[str, list[str], int]]] = {}
    names: dict[int, str] = {}
    fjsonl = case_dir / "functions.jsonl"
    if fjsonl.is_file():
        for line in fjsonl.read_text().splitlines():
            row = json.loads(line)
            fid = row.get("function_id", row.get("reg"))
            names[fid] = row["name"]
    for tsv in sorted((case_dir / "functions").glob("*.tsv")):
        fid = int(tsv.stem)
        layouts[fid] = {}
        for line in tsv.read_text().splitlines():
            parts = line.split("\t")
            idx = int(parts[0])
            ops = parts[4].split(",") if len(parts) > 4 and parts[4] else []
            layouts[fid][idx] = (parts[3], ops, int(parts[5]))
    return layouts, names


class _ZstStream:
    """Chunked reader over `zstd -dc` (or a raw .bin file)."""

    def __init__(self, path: Path, chunk_size: int = 1 << 20):
        self._proc = None
        if path.suffix == ".zst":
            self._proc = subprocess.Popen(
                ["zstd", "-dc", str(path)], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL)
            self._raw = self._proc.stdout
        else:
            self._raw = path.open("rb")
        self._chunk_size = chunk_size
        self._buf = bytearray()
        self.eof = False
        self.pos = 0  # bytes consumed from the stream head

    def _fill(self, need: int) -> bool:
        while len(self._buf) < need and not self.eof:
            data = self._raw.read(max(self._chunk_size, need - len(self._buf)))
            if not data:
                self.eof = True
                if self._proc is not None and self._proc.wait() != 0:
                    raise OSError(
                        f"zstd -dc failed (exit {self._proc.returncode}) "
                        f"on the trace stream")
                break
            self._buf += data
        return len(self._buf) >= need

    def read_into(self, out: memoryview, n: int) -> int:
        """Copy up to n bytes; returns bytes copied (< n at EOF)."""
        if not self._fill(n):
            n = len(self._buf)
        out[:n] = self._buf[:n]
        del self._buf[:n]
        self.pos += n
        return n

    def skip(self, n: int) -> int:
        skipped = 0
        view = memoryview(bytearray(min(1 << 16, n)))
        while skipped < n:
            got = self.read_into(view, min(len(view), n - skipped))
            if got == 0:
                break
            skipped += got
        return skipped

    def close(self):
        try:
            self._raw.close()
        finally:
            if self._proc is not None:
                self._proc.wait()


def iter_records(case_dir, launch: int, cluster_filter=None,
                 chunk_size: int = 1 << 20, trace_path=None,
                 gsync_sequences=None,
                 cta_info=None) -> Iterator[DynRecord]:
    """Streaming reader of trace-<launch>.bin.zst record streams.

Decompression goes through a `zstd -dc` subprocess pipe read in chunks
(no whole-file buffer, no zstandard module dependency).  Record lengths
come from the static layout in functions/<function_id>.tsv
(captured_operands + mref_count), so records outside a cluster filter
are skipped by length only.  AFTER records attach to the most recent
unpaired same-idx instruction record of their (cta, warp); GSYNC records
attach by the same rule per phase.  ``cta_info`` (optional dict):
kind-3 records fill it as ``cta_info[(x,y,z)] = dict(cluster_ctarank=...,
cluster_nctaid=[x,y,z], clusterid=[x,y,z], nclusterid=[x,y,z])``; one warp
record per CTA suffices, later duplicates are ignored.  ``cluster_filter``
is a callable
    on linear cluster id... actually on (cta_x,cta_y,cta_z): return True
    to keep the record.  Records outside the filter advance by computed
    length only and are never decoded.

    AFTER/GSYNC records are attached to their instruction record inside
    the stream; the yielded DynRecord carries after_ready/gsync once all
    attachments are known -- i.e. records are yielded when the next
    same-warp instruction head arrives or at stream end.
    """
    layouts, _names = load_static(case_dir)
    if trace_path is not None:
        path = Path(trace_path)
    else:
        path = Path(case_dir) / f"trace-{launch}.bin.zst"
        if not path.is_file():
            raw = Path(case_dir) / f"trace-{launch}.bin"
            path = raw if raw.is_file() else path
    stream = _ZstStream(path, chunk_size)
    try:
        header = memoryview(bytearray(FILE_HEADER_SIZE))
        if stream.read_into(header, FILE_HEADER_SIZE) != FILE_HEADER_SIZE:
            raise ValueError(f"{path}: truncated file header")
        if bytes(header[:8]) != FILE_MAGIC:
            raise ValueError(f"{path}: bad magic {bytes(header[:8])!r}")
        pending: dict[tuple, DynRecord] = {}  # (cta,warp) -> last instr
        gsync_seq = 0        # launch-wide capture-channel order of GSYNC
        head_buf = memoryview(bytearray(HEAD.size))
        while True:
            got = stream.read_into(head_buf, HEAD.size)
            if got == 0:
                break
            if got != HEAD.size:
                raise ValueError(f"{path}: {got}-byte trailing head")
            (cta_x, cta_y, cta_z, warp, sm, kind, function_id, idx,
             active_mask, exec_mask, warp_slot) = HEAD.unpack(head_buf)
            cta = (cta_x, cta_y, cta_z)
            key = (cta, warp)
            keep = cluster_filter is None or cluster_filter(cta)
            if kind == KIND_CTA:
                buf = memoryview(bytearray(_CTA_BODY.size))
                stream.read_into(buf, _CTA_BODY.size)
                if cta_info is not None and cta not in cta_info:
                    v = _CTA_BODY.unpack(buf)
                    cta_info[cta] = {
                        "cluster_ctarank": v[0],
                        "cluster_nctaid": list(v[1:4]),
                        "clusterid": list(v[4:7]),
                        "nclusterid": list(v[7:10])}
                continue
            if kind == KIND_AFTER:
                buf = memoryview(bytearray(4))
                stream.read_into(buf, 4)
                if not keep:
                    continue        # attachment of a skipped record
                (ready,) = _U32.unpack(buf)
                rec = pending.get(key)
                if rec is not None and rec.idx == idx and \
                        rec.after_ready is None:
                    rec.after_ready = int(ready)
                else:
                    yield from _flush_nonkey(pending, key)
                    if rec is None:
                        raise ValueError(
                            f"{path}: AFTER idx={idx} has no instruction")
                continue
            if kind == KIND_GSYNC:
                if gsync_sequences is not None:
                    # pre-split stream: seq ordinals come from the
                    # splitter's launch-wide count file; running out
                    # means splitter/reader disagree -- fail closed
                    seq = next(gsync_sequences, None)
                    if seq is None:
                        raise ValueError(
                            f"{path}: more GSYNC records than "
                            f"gsync_seq entries")
                else:
                    seq = gsync_seq
                gsync_seq += 1      # launch-wide order counts every GSYNC
                t4 = memoryview(bytearray(4))
                stream.read_into(t4, 4)
                phase, nwords, flags, _pad = _TAIL4.unpack(t4)
                count = exec_mask.bit_count()
                if not keep:
                    stream.skip(count * nwords * 4)
                    continue        # attachment of a skipped record
                body = memoryview(bytearray(count * nwords * 4))
                stream.read_into(body, count * nwords * 4)
                vals = struct.unpack_from(f"<{count * nwords}I", body, 0)
                words = [tuple(vals[i * nwords:(i + 1) * nwords])
                         for i in range(count)]
                rec = pending.get(key)
                if rec is None or rec.idx != idx:
                    yield from _flush_nonkey(pending, key)
                    if pending.get(key) is None:
                        raise ValueError(
                            f"{path}: GSYNC idx={idx} has no instruction")
                    rec = pending[key]
                rec.gsync.append((phase, nwords, flags, words, seq))
                continue
            # kind 0: new instruction record -- its predecessor in the
            # same warp is now complete
            if key in pending:
                yield pending.pop(key)
            layout = layouts.get(function_id, {}).get(idx)
            if layout is None:
                raise ValueError(
                    f"{path}: no layout function={function_id} idx={idx}")
            _sass, opnames, mref_count = layout
            exec_count = exec_mask.bit_count()
            if not keep:
                stream.skip(record_tail_len(opnames, mref_count,
                                            exec_count))
                continue
            mrefs: list[tuple[int, ...]] = []
            for _m in range(mref_count):
                buf = memoryview(bytearray(exec_count * 8))
                stream.read_into(buf, exec_count * 8)
                mrefs.append(struct.unpack_from(f"<{exec_count}Q", buf, 0))
            ops: dict = {}
            gdesc_index = -1
            tmemd_index = -1
            for op in opnames:
                n = _op_entry_len(op, exec_count)
                if n == 0:
                    continue
                buf = memoryview(bytearray(n))
                stream.read_into(buf, n)
                if op == "SPACE":
                    ops[op] = struct.unpack_from("<3I", buf, 0)
                elif op.startswith("BRANK") or op.startswith("RANK"):
                    ops[op] = struct.unpack_from(f"<{exec_count}i", buf, 0)
                elif (op.startswith("URANK") or op == "DESC" or
                      op in ("CPOL", "CPOLH") or is_ur_token(op) or
                      op in _DESCRIPTOR_OPS or op.startswith("TCOORD")):
                    (ops[op],) = _U32.unpack_from(buf, 0)
                    if op == "GDESC":
                        gdesc_index += 1
                        ops[f"GDESC{gdesc_index}"] = ops[op]
                    elif op == "GDESCH" and gdesc_index >= 0:
                        ops[f"GDESC{gdesc_index}H"] = ops[op]
                    elif op == "TMEMD":
                        tmemd_index += 1
                        ops[f"TMEMD{tmemd_index}"] = ops[op]
                else:  # R<n>
                    ops[op] = struct.unpack_from(f"<{exec_count}I", buf, 0)
            if gdesc_index >= 0:
                ops["GDESC_COUNT"] = gdesc_index + 1
            if tmemd_index >= 0:
                ops["TMEMD_COUNT"] = tmemd_index + 1
            rec = DynRecord(
                launch=launch, cta=cta, warp=warp, sm=sm,
                warp_slot=warp_slot, function_id=function_id, idx=idx,
                active_mask=active_mask, exec_mask=exec_mask,
                mrefs=mrefs, ops=ops)
            pending[key] = rec
        if gsync_sequences is not None and \
                next(gsync_sequences, None) is not None:
            raise ValueError(
                f"{path}: gsync_seq has entries beyond stream end")
        for rec in pending.values():
            yield rec
    finally:
        stream.close()


def _flush_nonkey(pending: dict, key) -> Iterator[DynRecord]:
    """Yield every pending record except `key`'s (an AFTER/GSYNC must
    attach only to its own warp's latest record)."""
    for k in [k for k in pending if k != key]:
        yield pending.pop(k)
