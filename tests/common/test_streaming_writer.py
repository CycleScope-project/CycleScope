"""GSIMCAP02 split-statics format: cluster .gsc carries only dynamic
tables; statics.gsc carries the shared statics.  FactsReader must
reconstruct the identical model either way."""
import struct
from pathlib import Path

import pytest

from cyclescope.convert import model
from cyclescope.convert.facts_format import (
    StreamingFactsWriter, FactsReader, FactsWriter, write_statics_file)


def _statics():
    cw = model.ControlWord(stall=3, yield_hint=1, read_barrier=2,
                           write_barrier=4, wait_mask=7, reuse_mask=9)
    return [
        model.StaticInstruction(
            pc=0x10, code_address=0x100, opcode="LDG.E",
            operation_class="load", operands=("@P0", "R2", "[R4]"),
            predicate="P0", predicate_inverted=True,
            source_registers=("R2", "R4"), destination_registers=("R0",),
            register_roles={"R4": "address"}, control=cw,
            memory_space="global", branch_target=0x20),
        model.StaticInstruction(pc=0x14, code_address=None,
                                opcode="BAR.SYNC",
                                operation_class="barrier"),
    ]


def _cta():
    return model.CtaFacts(cta_linear=0, cta=(1, 2, 0), cluster_rank=2,
                          sm_id=5, peer_sm_id=6, multicast_mask=0xF,
                          cta_group=1, is_pair_leader=1, tmem_columns=64)


def _unit():
    u = model.ClusterUnit(cluster_id=7)
    u.ctas = [_cta()]
    ops = model.SemanticOperands(
        uniform={"UR4": 7}, lane={"R2": (1, 2, 3)}, ranks={0: (0, 1)})
    w = model.WarpStream(cta_linear=0, cta=(1, 2, 0), warp_in_cta=3,
                         sm_id=5, warp_slot=9)
    w.instructions = [
        model.DynamicInstruction(static_index=0, active_mask=3,
                                 predicate_mask=1, operands=ops),
        model.DynamicInstruction(static_index=1, active_mask=0xFF)]
    u.warps = [w]
    u.statics = _statics()
    return u


def test_split_statics_roundtrip(tmp_path):
    u = _unit()
    # v1: single-file facts
    p1 = tmp_path / "old.gsc"
    FactsWriter().write_unit(u, p1)
    # v2: cluster file + statics.gsc
    w = StreamingFactsWriter(split_statics=True)
    w.begin_warp(0, (1, 2, 0), 5, 9, 3)
    for d in u.warps[0].instructions:
        w.add_warp_instruction(d)
    w.end_warp()
    p2 = tmp_path / "000007.gsc"
    w.finish(7, u.ctas, u.statics, {}, [], p2)
    write_statics_file(u.statics, tmp_path / "statics.gsc")

    r1, r2 = FactsReader(p1), FactsReader(p2)
    assert r1.cluster_id == r2.cluster_id == 7
    assert r1.n_stat == r2.n_stat == 2
    s1, s2 = r1.statics(), r2.statics()
    assert [x.opcode for x in s1] == [x.opcode for x in s2]
    assert [x.operands for x in s1] == [x.operands for x in s2]
    w1, w2 = list(r1.warps()), list(r2.warps())
    assert len(w1) == len(w2) == 1
    d1, d2 = w1[0].instructions, w2[0].instructions
    assert len(d1) == len(d2) == 2
    assert d1[0].operands.uniform == d2[0].operands.uniform


def test_split_statics_missing_companion_fails(tmp_path):
    u = _unit()
    w = StreamingFactsWriter(split_statics=True)
    w.begin_warp(0, (1, 2, 0), 5, 9, 3)
    w.add_warp_instruction(u.warps[0].instructions[0])
    w.end_warp()
    w.finish(7, u.ctas, u.statics, {}, [], tmp_path / "000007.gsc")
    with pytest.raises(Exception, match="facts_statics"):
        FactsReader(tmp_path / "000007.gsc")


def test_strs_contains_no_static_strings(tmp_path):
    u = _unit()
    w = StreamingFactsWriter(split_statics=True)
    w.begin_warp(0, (1, 2, 0), 5, 9, 3)
    w.add_warp_instruction(u.warps[0].instructions[0])
    w.end_warp()
    p = tmp_path / "000007.gsc"
    w.finish(7, u.ctas, u.statics, {}, [], p)
    data = p.read_bytes()
    assert data.startswith(b"GSIMCP2\0")
    assert b"LDG.E" not in data.split(b"STRS")[-1]
    write_statics_file(u.statics, tmp_path / "statics.gsc")
    assert (tmp_path / "statics.gsc").read_bytes().startswith(b"GSIMCPS\0")
