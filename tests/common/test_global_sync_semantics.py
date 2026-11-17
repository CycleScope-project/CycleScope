"""kind-34 global_sync SEMA rows: emitted for GSYN-observed ld_strong /
atom / red (never st_strong or unobserved loads), python == cpp bytes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cyclescope.arch.hopper import rules
from cyclescope.convert import model
from cyclescope.convert.facts_format import FactsWriter
from cyclescope.convert.pipeline import convert_facts
from cyclescope.io.fixed_tables import TableReader
from cyclescope.job import segment_builder as sb, segment_format as sf
from cyclescope.job.semantics import SemanticsKind, unpack_fields

from .test_segment_builder import _PROGRAM_TABLES

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / "configs" / "trees" / "h100.json"
NONE32 = 0xFFFFFFFF
FULL = 0xFFFFFFFF

# (opcode, access kind, GSYN observation kind) -> expected SEMA class
_OPS = [
    ("LDG.E.STRONG.GPU", "load", 2, 3),
    ("ATOMG.E.ADD.STRONG.GPU", "atomic", 3, 1),
    ("REDG.E.ADD.STRONG.GPU", "atomic", 1, 2),
    ("STG.E.STRONG.GPU", "store", 1, None),
    ("LDG.E", "load", None, None),
]


def _unit() -> model.ClusterUnit:
    statics, insts = [], []
    for i, (opcode, access, gkind, _cls) in enumerate(_OPS):
        statics.append(model.StaticInstruction(
            pc=0x10 * i, code_address=0x1000 + 0x10 * i, opcode=opcode,
            operation_class=rules.operation_class(opcode),
            source_registers=("R2", "R3"),
            destination_registers=() if access == "store" else ("R4",),
            memory_space="global"))
        gs = None if gkind is None else model.GlobalSyncObservation(
            kind=gkind, address=0x4000 + 0x40 * i, value=7 + i, lane=0,
            post_known=1 | (2 if gkind != 2 else 0), before=i,
            sequence=i)
        insts.append(model.DynamicInstruction(
            static_index=i, active_mask=FULL, predicate_mask=FULL,
            memory=(model.MemoryReference(
                space="global", access_kind=access, width_bytes=4,
                lane_addresses=tuple(0x4000 + 0x40 * i for _ in range(32)),
                lane_mask=FULL),),
            global_sync=gs))
    warp = model.WarpStream(cta_linear=0, cta=(0, 0, 0), warp_in_cta=0,
                            sm_id=0, warp_slot=0, instructions=insts)
    return model.ClusterUnit(
        cluster_id=0,
        ctas=[model.CtaFacts(cta_linear=0, cta=(0, 0, 0), sm_id=0)],
        warps=[warp], statics=statics)


def _rows(reader, tag):
    table = reader.require_table(tag)
    tr = TableReader(sf.RECORDS[tag], table)
    return [tr.unpack(i) for i in range(len(tr))]


def _sema_by_instance(reader) -> list[dict | None]:
    sema = _rows(reader, "SEMA")
    out = []
    for inst in _rows(reader, "INST"):
        if inst["semantics"] == NONE32:
            out.append(None)
            continue
        row = sema[inst["semantics"]]
        fields = unpack_fields(SemanticsKind(row["kind"]),
                               tuple(row[f"f{k}"] for k in range(6)))
        out.append({"kind": row["kind"], **fields})
    return out


def test_gsync_rows_mark_readers_and_updaters() -> None:
    image = sb.build_segment(_unit(), rules, _PROGRAM_TABLES, {})
    reader = sf.read_segment(image)
    got = _sema_by_instance(reader)
    assert got == [
        {"kind": 34, "observation_kind": 2, "class": 3},
        {"kind": 34, "observation_kind": 3, "class": 1},
        {"kind": 34, "observation_kind": 1, "class": 2},
        None,   # st_strong: producer only
        None,   # unobserved load
    ]
    gsyn = TableReader(sf.OPTIONAL_RECORDS["GSYN"],
                       reader.require_table("GSYN"))
    assert [gsyn.unpack(i)["class_"] for i in range(len(gsyn))] == \
        [3, 1, 2, 4]


def test_gsync_row_skips_predicated_off() -> None:
    unit = _unit()
    unit.warps[0].instructions[0].predicate_mask = 0
    image = sb.build_segment(unit, rules, _PROGRAM_TABLES, {})
    got = _sema_by_instance(sf.read_segment(image))
    assert got[0] is None and got[1] is not None


def test_gsync_rows_python_cpp_parity(tmp_path) -> None:
    bin_ = shutil.which("cyclescope-convert") or str(REPO / "build" /
                                               "cyclescope-convert")
    if not Path(bin_).exists():
        pytest.skip("cyclescope-convert not built")
    facts = tmp_path / "facts"
    facts.mkdir()
    FactsWriter().write_unit(_unit(), facts / "000000.gsc")
    (facts / "launch.json").write_text(json.dumps({
        "device": "h100", "sm_arch": "sm_90a",
        "grid": [1, 1, 1], "block": [32, 1, 1], "cluster": [1, 1, 1],
        "capture": "test_capture"}))
    outs = {}
    for builder in ("python", "cpp"):
        out = tmp_path / builder
        convert_facts(facts, "hopper", TREE, out, cache_dir=None,
                      builder=builder)
        outs[builder] = out
    segs = {b: sorted((o / "segments").glob("*.seg"))
            for b, o in outs.items()}
    assert [p.name for p in segs["python"]] == [p.name for p in segs["cpp"]]
    for pa, pb in zip(segs["python"], segs["cpp"]):
        a = sf.read_segment(pa.read_bytes())
        b = sf.read_segment(pb.read_bytes())
        for tag in ("SEMA", "INST", "GSYN"):
            assert bytes(a.table(tag)) == bytes(b.table(tag)), tag
        kinds = [r["kind"] for r in _rows(a, "SEMA")]
        assert kinds == [34, 34, 34]
    assert json.loads((outs["python"] / "launch.json").read_text()) == \
        json.loads((outs["cpp"] / "launch.json").read_text())
