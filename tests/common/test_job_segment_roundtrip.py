"""Segment write/read round-trip tests."""

from __future__ import annotations

import pytest

from cyclescope.io.file_header import FileFormatError
from cyclescope.io.fixed_tables import TableReader
from cyclescope.job import segment_format as sf


def _segment(cluster_id: int = 0) -> bytes:
    sw = sf.SegmentWriter()
    tables = {t: w.payload() for t, w in sw.writers.items()}
    tables["META"] = sf.META.pack(
        tree_digest=b"\1" * 32, program_digest=b"\2" * 32,
        converter_digest=b"\3" * 32, cluster_id=cluster_id, cta_count=1,
        warp_count=1, instruction_count=1, flags=sf.META_FLAG_MODEL_COMPLETE)
    tables["STRS"] = b""
    tables["CTAS"] = sf.CTAS.pack_row(
        (0, 0, 0, 0, 0, 1, 1, 1, 0, 0xFFFFFFFF, 0, 1, 0, 0, 0, 1))
    tables["WARP"] = sf.WARP.pack_row((0, 0, 0, 0, 0, 1))
    tables["INST"] = sf.INST.pack_row((0, 0, 0xFFFFFFFF, 0))
    tables["TMPL"] = sf.TMPL.pack_row(
        (0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0, 0))
    tables["SEMA"] = sf.SEMA.pack_row((2, 0, 100, 5, 6, 7, 8, 9, 10))
    tables["MSEC"] = sf.MSEC.pack_row((0x1000, 0xFF, 0xFFFFFFFF))
    return sf.write_segment(tables, payload_sha256=True)


def test_segment_roundtrip() -> None:
    seg = _segment()
    reader = sf.read_segment(seg)
    meta = TableReader(sf.META, reader.require_table("META")).unpack(0)
    assert meta["cluster_id"] == 0 and meta["flags"] == 1
    sema = TableReader(sf.SEMA, reader.require_table("SEMA")).unpack(0)
    assert (sema["kind"], sema["earliest_issue"], sema["f4"]) == (2, 100, 9)
    assert reader.payload_sha256 is not None


def test_missing_table_rejected() -> None:
    seg = _segment()
    reader = sf.read_segment(seg)
    with pytest.raises(FileFormatError):
        reader.require_table("XXXX")


def test_table_tag_helpers() -> None:
    from cyclescope.job.semantics import (
        SemanticsKind, pack_fields, unpack_fields)
    slots = pack_fields(
        SemanticsKind.mbarrier_arrive, object=3, generation=9,
        transaction_bytes=128, arrival_count=1, destination_mask=0xFF,
        object_list_begin=4, object_list_count=2)
    back = unpack_fields(SemanticsKind.mbarrier_arrive, slots)
    assert back["object"] == 3 and back["generation"] == 9
    assert back["object_list_begin"] == 4
    assert back["object_list_count"] == 2
