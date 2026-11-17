"""Segment table record layouts and segment writer/reader (spec §2).

A segment (``GSIMSEG``) carries all per-cluster facts plus the compiled
mechanism program tables.  Record layouts below are the single source of
truth shared with the C++ side.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..io.file_header import FileFormatError, FileReader, tag_value, write_file
from ..io.fixed_tables import Record, TableReader, TableWriter

META = Record("<32s32s32sIIIQI", (
    "tree_digest", "program_digest", "converter_digest", "cluster_id",
    "cta_count", "warp_count", "instruction_count", "flags"))
CTAS = Record("<11IBBHII", (
    "cta_index", "cta_x", "cta_y", "cta_z", "cluster_rank",
    "cluster_shape_x", "cluster_shape_y", "cluster_shape_z", "sm_id",
    "peer_sm_id", "multicast_mask", "cta_group", "is_pair_leader",
    "tmem_columns", "first_warp", "warp_count"))
WARP = Record("<IIIIQQ", (
    "cta_index", "warp_in_cta", "warp_slot", "smsp",
    "first_instance", "instance_count"))
TMPL = Record("<IIIIQIQIBBBBQIIQ", (
    "operation_class", "flags", "resource", "issue_group_resource",
    "source_begin", "source_count", "destination_begin",
    "destination_count", "scoreboard_read", "scoreboard_write",
    "scoreboard_wait_mask", "control", "timing_begin", "timing_count",
    "semantic_kind", "semantic_u64"))
INST = Record("<IIII", (
    "template", "semantics", "memory_access", "code_line"))
# register id namespace: R<n>->n, UR<n>->256+n, P<n>->512+n,
# UP<n>->768+n, B<n>->1024+n; RZ/URZ/PT/UPT/SR_* and other hard-wired
# names emit no row (see segment_builder._register_id)
REGS = Record("<I", ("register",))
SEMA = Record("<HHQ6Q", (
    "kind", "flags", "earliest_issue", "f0", "f1", "f2", "f3", "f4", "f5"))
MACC = Record("<QIIQQQII", (
    "sector_begin", "sector_count", "flags", "sequence", "shape",
    "timing_begin", "timing_count", "cache_hint"))
MSEC = Record("<QII", ("address", "byte_mask", "lane_mask"))
TIME = Record("<IIIIQdB7s", (
    "kind", "selector_a", "selector_b", "ready_warps", "selector_c",
    "cycles", "enabled", "pad"))
AXIS = Record("<I", ("name",))
AXV_ = Record("<IBB2sII", ("kind", "owner", "pad1", "pad2",
                        "values_begin", "values_count"))
RAX_ = Record("<II", ("begin", "count"))
RAXV = Record("<I", ("axis",))
PROG = Record("<IBHBIQQI", (
    "mechanism", "event", "schema_id", "timing_kind", "root_node",
    "field_mask",
    "operation_class_begin", "operation_class_count"))
NODE = Record("<IIIIII", (
    "parent", "child_begin", "child_count", "guard_begin",
    "guard_count", "leaf"))
GRD_ = Record("<IB3sQQ", ("axis", "operator", "pad", "low", "high"))
LEAF = Record("<IB3s", ("parameter", "status", "pad"))
PARM = Record("<HHB3sIdddQI", (
    "schema_id", "field_id", "model_kind", "pad", "axis", "value",
    "slope", "intercept", "points_begin", "points_count"))
PTS_ = Record("<dd", ("axis_value", "value"))
FACT = Record("<IQ", ("axis", "value"))
USES = Record("<IB3sQHQQI", (
    "node", "target", "pad", "target_index", "schema_id", "field_mask",
    "fact_begin", "fact_count"))
SPLT = Record("<IIII", ("semantics", "rank", "credit", "pad"))
MBPI = Record("<IIQII", (
    "cta_index", "object", "address", "arrival_count", "pad"))
MBOL = Record("<I", ("object",))
CODE = Record("<IIQ", ("instance", "pad", "code_address"))
CLKM = Record("<IIQ", ("warp", "kind", "instance"))

RECORDS: dict[str, Record | None] = {
    "META": META, "CTAS": CTAS, "WARP": WARP, "TMPL": TMPL,
    "INST": INST, "REGS": REGS, "SEMA": SEMA, "MACC": MACC,
    "MSEC": MSEC, "TIME": TIME, "PROG": PROG, "NODE": NODE,
    "GRD ": GRD_, "LEAF": LEAF, "PARM": PARM, "PTS ": PTS_,
    # "AXV "/"RAX "/"RAXV" are GSIMPRG01-only (not segment tables)
    "FACT": FACT, "USES": USES, "SPLT": SPLT, "MBPI": MBPI,
    "MBOL": MBOL, "CODE": CODE, "CLKM": CLKM, "STRS": None,
    "AXIS": AXIS,
}

REQUIRED_TABLES = ["META", "CTAS", "WARP", "TMPL", "INST", "REGS", "SEMA",
                   "MACC", "MSEC", "TIME", "PROG", "NODE", "GRD ", "LEAF",
                   "PARM", "PTS ", "FACT", "USES", "SPLT", "MBPI", "MBOL",
                   "CODE", "CLKM", "STRS", "AXIS"]

# optional segment tables: emitted only when present in `tables` (kept
# out of RECORDS/REQUIRED_TABLES so the SegmentWriter does not create
# an empty writer and break byte-parity with the C++ builder)
OPTIONAL_RECORDS: dict[str, Record] = {
    "MBOB": Record("<IQ", ("cta_index", "address")),
    # global-sync observation rows; same fields as the container GSYN
    # record minus the segment column (instance is segment-local)
    "GSYN": Record("<IBBBBQQQQ", (
        "instance", "lane", "kind", "class_", "flags",
        "address", "after", "before", "sequence")),
}

META_FLAG_MODEL_COMPLETE = 0x1

PROGRAM_TAGS = ("PROG", "NODE", "GRD ", "LEAF", "PARM", "PTS ", "AXIS")
SEGMENT_SPLIT_VERSION = 2


def field_offset(record: Record, name: str) -> int:
    """Byte offset of a named field inside a packed record."""
    import re
    import struct as _s
    pos = 0
    items = []
    for token in re.findall(r"\d*[a-zA-Z]", record.fmt.lstrip("<>=!")):
        count = int(token[:-1]) if len(token) > 1 else 1
        items.extend([token[-1]] * (count if token[-1] != "s" else 1))
    for token, fname in zip(items, record.fields):
        size = _s.calcsize("<" + token)
        if fname == name:
            return pos
        pos += size
    raise KeyError(name)


def write_segment(tables: Mapping[str, bytes],
                  *, split_program: bool = False,
                  payload_sha256: bool = False) -> bytes:
    """Assemble one ``GSIMSEG`` file; ``tables`` is tag -> payload bytes.

    ``split_program`` (version 2): the seven shared program tables are
    omitted; readers splice them from the sibling ``program.tables``.
    """
    required = REQUIRED_TABLES if not split_program else [
        t for t in REQUIRED_TABLES if t not in PROGRAM_TAGS]
    missing = [t for t in required if t not in tables]
    if missing:
        raise FileFormatError(f"segment missing tables: {missing}")
    order = list(required) + [
        t for t in OPTIONAL_RECORDS if t in tables]
    return write_file(
        "segment",
        [(tag, OPTIONAL_RECORDS[tag].size if tag in OPTIONAL_RECORDS
          else (RECORDS[tag].size if RECORDS[tag] else 0),
          tables[tag])
         for tag in order],
        version=SEGMENT_SPLIT_VERSION if split_program else 1,
        payload_sha256=payload_sha256)


def program_tables_payload(tables: Mapping[str, bytes]) -> bytes:
    """One ``program.tables`` file (GSIMPRG01) with the seven shared
    program tables."""
    return write_file(
        "GSIMPRG01",
        [(tag, RECORDS[tag].size, tables[tag]) for tag in PROGRAM_TAGS])


class _SplitReader:
    """v2 segment view: per-segment tables from the file, the seven
    program tables from the shared ``program.tables`` FileReader."""

    def __init__(self, fr, program):
        self._fr = fr
        self._program = program

    def table(self, tag):
        t = self._fr.table(tag)
        return t if t is not None else (
            self._program.table(tag) if self._program else None)

    def require_table(self, tag):
        t = self.table(tag)
        if t is None:
            raise FileFormatError(f"missing table {tag!r}")
        return t

    def table_layout(self, tag):
        lay = self._fr.table_layout(tag)
        return lay if lay is not None else (
            self._program.table_layout(tag) if self._program else None)

    def __getattr__(self, name):
        return getattr(self._fr, name)


def read_segment(data: bytes | memoryview,
                 program: "FileReader | bytes | memoryview | None" = None):
    """Read one segment image.  Version-2 images (GSIMSEG v2) carry no
    program tables; ``program`` supplies the shared tables — a FileReader,
    the raw program.tables bytes, or None to fail on access."""
    fr = FileReader(data, expect="segment")
    if fr.version < SEGMENT_SPLIT_VERSION:
        for tag in REQUIRED_TABLES:
            fr.require_table(tag)
            record = RECORDS.get(tag)
            if record is not None:
                payload = fr.require_table(tag)
                if len(payload) % record.size:
                    raise FileFormatError(
                        f"table {tag!r} payload is not a multiple of "
                        f"{record.size}")
        return fr
    if isinstance(program, (bytes, memoryview)):
        program = FileReader(program)
    return _SplitReader(fr, program)


def load_segment(path, program=None):
    """Read a .seg file; v2 images auto-load ``program.tables`` from the
    same directory when ``program`` is not given."""
    from pathlib import Path
    path = Path(path)
    data = path.read_bytes()
    if program is None:
        pt = path.parent / "program.tables"
        if pt.is_file():
            program = FileReader(memoryview(pt.read_bytes()))
    return read_segment(memoryview(data), program=program)


class SegmentWriter:
    """Convenience holder for one segment's table writers."""

    def __init__(self):
        self.writers = {
            tag: TableWriter(record)
            for tag, record in RECORDS.items() if record is not None}
        self.strings_payload = b""

    def table(self, tag: str) -> TableWriter:
        return self.writers[tag]

    def payload(self) -> bytes:
        tables = {tag: w.payload() for tag, w in self.writers.items()}
        tables["STRS"] = self.strings_payload
        return write_segment(tables)


def table_rows(reader: FileReader, tag: str) -> TableReader | None:
    record = RECORDS.get(tag) or OPTIONAL_RECORDS.get(tag)
    payload = reader.table(tag)
    if record is None or payload is None:
        return None
    return TableReader(record, payload)
