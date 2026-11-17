"""Container write/read/merge and digest-mismatch tests."""

from __future__ import annotations

import hashlib

import pytest

from cyclescope.io.file_header import FileFormatError
from cyclescope.io.fixed_tables import TableReader
from cyclescope.job import container as cf, segment_format as sf


def _segment(cluster_id: int, n_inst: int = 1) -> bytes:
    sw = sf.SegmentWriter()
    tables = {t: w.payload() for t, w in sw.writers.items()}
    tables["META"] = sf.META.pack(
        tree_digest=b"\1" * 32, program_digest=b"\2" * 32,
        converter_digest=b"\3" * 32, cluster_id=cluster_id, cta_count=1,
        warp_count=1, instruction_count=n_inst, flags=0)
    tables["STRS"] = b""
    tables["CTAS"] = sf.CTAS.pack_row(
        (cluster_id, 0, 0, 0, 0, 1, 1, 1, 0, 0xFFFFFFFF, 0, 1, 0, 0, 0, 1))
    tables["WARP"] = sf.WARP.pack_row((cluster_id, 0, 0, 0, 0, n_inst))
    tables["INST"] = b"".join(
        sf.INST.pack_row((0, 0, 0xFFFFFFFF, i)) for i in range(n_inst))
    tables["TMPL"] = sf.TMPL.pack_row(
        (0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0, 0))
    return sf.write_segment(tables)


def test_container_three_segments_merge() -> None:
    segs = [_segment(i, n_inst=i + 1) for i in range(3)]
    cont = cf.write_container({"device": "h100"}, segs)
    reader = cf.read_container(cont)
    images = cf.segment_images(reader)
    assert len(images) == 3
    merged = cf.merge_segments(images)
    inst = merged["tables"]["INST"]
    assert len(inst) // sf.INST.size == 6
    warps = merged["tables"]["WARP"]
    # warp in segment k has first_instance rebased by earlier segments
    w1 = sf.WARP.unpack(warps, 1)["first_instance"]
    w2 = sf.WARP.unpack(warps, 2)["first_instance"]
    assert (w1, w2) == (1, 3)
    segd = TableReader(cf.SEGD, reader.require_table("SEGD"))
    assert [segd.unpack(i)["cta_base"] for i in range(3)] == [0, 1, 2]


def test_partial_launch_single_segment() -> None:
    seg = _segment(0, 2)
    cont = cf.write_container(
        {"device": "h100", "flags": cf.LNCH_PARTIAL_LAUNCH}, [seg])
    lnch = TableReader(
        cf.LNCH, cf.read_container(cont).require_table("LNCH")).unpack(0)
    assert lnch["flags"] & cf.LNCH_PARTIAL_LAUNCH
    merged = cf.merge_segments(cf.segment_images(cf.read_container(cont)))
    assert len(merged["tables"]["INST"]) // sf.INST.size == 2


def test_segment_digest_mismatch_rejected() -> None:
    segs = [_segment(i) for i in range(2)]
    cont = bytearray(cf.write_container({"device": "h100"}, segs))
    reader = cf.read_container(bytes(cont))
    segd = TableReader(cf.SEGD, reader.require_table("SEGD"))
    off = segd.unpack(1)["byte_offset"]
    cont[off + 200] ^= 0xFF
    reader = cf.read_container(bytes(cont))
    with pytest.raises(FileFormatError, match="sha256"):
        cf.segment_images(reader)


def test_meta_digest_mismatch_rejected() -> None:
    seg0 = _segment(0)
    bad1 = _segment(1).replace(b"\1" * 32, b"\7" * 32, 1)
    with pytest.raises(FileFormatError, match="differs"):
        cf.write_container({"device": "h100"}, [seg0, bad1])


def _segment_with_facts(cluster_id: int, tail_strings: list[bytes],
                        tail_facts: list[tuple[int, int]],
                        tmpl_string: int = 0) -> bytes:
    """Program prefix: STRS b"axis\\0mech\\0" + one init FACT row; the
    segment appends its own strings and per-instance FACT rows with one
    USES row (target 0 = template) referencing them."""
    sw = sf.SegmentWriter()
    tables = {t: w.payload() for t, w in sw.writers.items()}
    tables["META"] = sf.META.pack(
        tree_digest=b"\1" * 32, program_digest=b"\2" * 32,
        converter_digest=b"\3" * 32, cluster_id=cluster_id, cta_count=1,
        warp_count=1, instruction_count=1, flags=0)
    prefix = b"axis\0mech\0"
    strs = bytearray(prefix)
    offs = []
    for s in tail_strings:
        offs.append(len(strs))
        strs += s + b"\0"
    tables["STRS"] = bytes(strs)
    tables["AXIS"] = sf.AXIS.pack_row((0,))
    facts = bytearray(sf.FACT.pack_row((0, 7)))        # init-path row
    for axis_off, value in tail_facts:
        facts += sf.FACT.pack_row((axis_off, value))
    tables["FACT"] = bytes(facts)
    tables["USES"] = (sf.USES.pack_row((0, 9, b"\0\0\0", 0, 0, 0, 0, 1)) +
                      sf.USES.pack_row((0, 0, b"\0\0\0", 0, 0, 0, 1,
                                        len(tail_facts))))
    tables["CTAS"] = sf.CTAS.pack_row(
        (cluster_id, 0, 0, 0, 0, 1, 1, 1, 0, 0xFFFFFFFF, 0, 1, 0, 0, 0, 1))
    tables["WARP"] = sf.WARP.pack_row((cluster_id, 0, 0, 0, 0, 1))
    tables["INST"] = sf.INST.pack_row((0, 0, 0xFFFFFFFF, 0))
    tables["TMPL"] = sf.TMPL.pack_row(
        (offs[tmpl_string], 0, 0xFFFFFFFF, 0, 0, 0, 0, 0xFFFFFFFF, 0, 0, 0,
         0, 0, 0, 0, 0))
    return sf.write_segment(tables)


def test_merge_rebases_per_segment_strings_and_facts() -> None:
    seg0 = _segment_with_facts(0, [b"FFMA", b"LDG"], [(0, 11), (0, 12)])
    seg1 = _segment_with_facts(1, [b"FFMA", b"HMMA", b"BAR"], [(0, 21)],
                               tmpl_string=2)
    merged = cf.merge_segments([seg0, seg1])["tables"]
    strs = bytes(merged["STRS"])
    # shared prefix + seg0 tail once, seg1 tail after it (FFMA shared)
    assert strs == b"axis\0mech\0FFMA\0LDG\0HMMA\0BAR\0"
    tm = merged["TMPL"]
    op0 = sf.TMPL.unpack(tm, 0)["operation_class"]
    op1 = sf.TMPL.unpack(tm, 1)["operation_class"]
    assert strs[op0:].split(b"\0")[0] == b"FFMA"
    assert strs[op1:].split(b"\0")[0] == b"BAR"
    facts = merged["FACT"]
    assert [sf.FACT.unpack(facts, i)["value"] for i in range(4)] == \
        [7, 11, 12, 21]
    uses = TableReader(sf.USES, merged["USES"])
    rows = [uses.unpack(i) for i in range(len(uses))]
    per_inst = [r for r in rows if r["target"] == 0]
    assert [(r["fact_begin"], r["fact_count"]) for r in per_inst] == \
        [(1, 2), (3, 1)]
    assert len([r for r in rows if r["target"] == 9]) == 1


def test_merge_renumbers_ctas_densely() -> None:
    def seg(linear, n_inst=1):
        s = bytearray(_segment(0, n_inst))
        return s
    # segments carrying launch-wide CTA ids 7 and 3 -> dense 0 / 1
    sw = []
    for linear in (7, 3):
        tables = {t: w.payload() for t, w in sf.SegmentWriter().writers.items()}
        tables["META"] = sf.META.pack(
            tree_digest=b"\1" * 32, program_digest=b"\2" * 32,
            converter_digest=b"\3" * 32, cluster_id=linear, cta_count=1,
            warp_count=1, instruction_count=1, flags=0)
        tables["STRS"] = b""
        tables["CTAS"] = sf.CTAS.pack_row(
            (linear, linear, 0, 0, 0, 1, 1, 1, 0, 0xFFFFFFFF, 0, 1, 0, 0,
             0, 1))
        tables["WARP"] = sf.WARP.pack_row((linear, 0, 0, 0, 0, 1))
        tables["INST"] = sf.INST.pack_row((0, 0, 0xFFFFFFFF, 0))
        tables["TMPL"] = sf.TMPL.pack_row(
            (0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0,
             0))
        tables["MBPI"] = sf.MBPI.pack_row((linear, 0, 0x100, 1, 0))
        sw.append(sf.write_segment(tables))
    merged = cf.merge_segments(sw)["tables"]
    assert [sf.CTAS.unpack(merged["CTAS"], i)["cta_index"]
            for i in range(2)] == [0, 1]
    assert [sf.WARP.unpack(merged["WARP"], i)["cta_index"]
            for i in range(2)] == [0, 1]
    assert [sf.MBPI.unpack(merged["MBPI"], i)["cta_index"]
            for i in range(2)] == [0, 1]


def test_lnch_record_size() -> None:
    assert cf.LNCH.size == 104
    assert "warmup_launches" not in cf.LNCH.fields
    assert "L2FP" not in cf.CONTAINER_RECORDS


def test_build_gate_reference_cache_control(tmp_path) -> None:
    import json

    from cyclescope.convert.semantics_common import ConversionError

    seg_dir = tmp_path / "segments"
    seg_dir.mkdir()
    (seg_dir / "000000.seg").write_bytes(_segment(0))

    def build(v):
        doc = {"device": "h100", "segments": ["000000.seg"]}
        if v is not None:
            doc["reference_cache_control"] = v
        (tmp_path / "launch.json").write_text(json.dumps(doc))
        return cf.build_container_from_dir(tmp_path, tmp_path / "job.bin")

    with pytest.raises(ConversionError,
                       match="missing_fact:reference_cache_control"):
        build(None)
    with pytest.raises(ConversionError,
                       match="missing_fact:reference_cache_control"):
        build(0)
    with pytest.raises(ConversionError,
                       match="unsupported:reference_cache_control:none"):
        build(1)
    out = build(2)
    lnch = TableReader(
        cf.LNCH,
        cf.read_container(out.read_bytes()).require_table("LNCH")
    ).unpack(0)
    assert lnch["reference_cache_control"] == 2
