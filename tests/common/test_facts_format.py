"""GSIMCAP01 facts file round-trip (model -> file -> model)."""
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                     "scripts" / "migration"))

from cyclescope.convert.facts_format import FactsReader, FactsWriter  # noqa
from cyclescope.convert import model


def _unit():
    cw = model.ControlWord(stall=3, yield_hint=1, read_barrier=2,
                           write_barrier=4, wait_mask=7, reuse_mask=9)
    st = model.StaticInstruction(
        pc=0x10, code_address=0x100, opcode="LDG.E",
        operation_class="load", operands=("@P0", "R2", "[R4]"),
        predicate="P0", predicate_inverted=True,
        source_registers=("R2", "R4"), destination_registers=("R0",),
        register_roles={"R4": "address"}, control=cw,
        memory_space="global", branch_target=0x20)
    st2 = model.StaticInstruction(
        pc=0x14, code_address=None, opcode="BAR.SYNC",
        operation_class="barrier")
    m = model.MemoryReference(
        space="global", access_kind="load", width_bytes=4,
        lane_addresses=(0x1000, 0x1004), lane_mask=3,
        remote_rank=1, cache_hint=5, destination_count=2,
        transfer_bytes=4096)
    ops = model.SemanticOperands(
        uniform={"UR4": 7, "role:R1": "data"},
        lane={"R2": (1, 2, 3)}, ranks={0: (0, 1)},
        after_ready_mask=0xF, tensor_map=1,
        tensor_coordinates=(64, 64), descriptor_words=(0xAB,))
    g = model.GlobalSyncObservation(kind=1, address=0x40, value=9,
                                    lane=2, acquire_mode=1,
                                    post_known=1)
    d1 = model.DynamicInstruction(static_index=0, active_mask=3,
                                  predicate_mask=1, memory=(m,),
                                  operands=ops, global_sync=g)
    d2 = model.DynamicInstruction(static_index=1, active_mask=0xFF)
    w = model.WarpStream(cta_linear=0, cta=(1, 2, 0), warp_in_cta=3,
                         sm_id=5, warp_slot=9,
                         instructions=[d1, d2])
    c = model.CtaFacts(cta_linear=0, cta=(1, 2, 0), cluster_rank=2,
                       sm_id=5, peer_sm_id=6, multicast_mask=0xF,
                       cta_group=1, is_pair_leader=1, tmem_columns=64)
    u = model.ClusterUnit(cluster_id=7)
    u.ctas = [c]
    u.warps = [w]
    u.statics = [st, st2]
    return u


def test_facts_round_trip(tmp_path):
    u = _unit()
    p = tmp_path / "u.gsc"
    FactsWriter().write_unit(u, p, digest=b"\xAA" * 32)
    r = FactsReader(p)
    assert r.cluster_id == 7
    assert r.digest == b"\xAA" * 32
    assert r.ctas() == u.ctas
    assert r.statics() == u.statics
    ws = list(r.warps())
    assert len(ws) == 1
    w = ws[0]
    assert (w.cta_linear, w.cta, w.warp_in_cta, w.sm_id,
            w.warp_slot) == (0, (1, 2, 0), 3, 5, 9)
    def _d(x):
        d = dataclasses.asdict(x)
        d.pop("raw_key", None)
        return d
    assert [_d(d) for d in w.instructions] == [
        _d(d) for d in u.warps[0].instructions]


def test_facts_streaming_no_materialization(tmp_path):
    u = _unit()
    p = tmp_path / "u.gsc"
    FactsWriter().write_unit(u, p)
    r = FactsReader(p)
    it = r.warps()
    first = next(it)          # one warp at a time
    assert first.instructions[0].static_index == 0
    with pytest.raises(StopIteration):
        next(it)


def test_after_ready_mask_zero_roundtrips(tmp_path):
    import dataclasses
    u = _unit()
    for mask in (0, None, 0xFFFFFFFF):
        d = dataclasses.replace(u.warps[0].instructions[0],
                                operands=dataclasses.replace(
                                    u.warps[0].instructions[0].operands,
                                    after_ready_mask=mask))
        w = dataclasses.replace(
            u.warps[0], instructions=[d, u.warps[0].instructions[1]])
        u.warps = [w]
        p = tmp_path / f"u{mask}.gsc"
        FactsWriter().write_unit(u, p, digest=b"\xAA" * 32)
        got = list(FactsReader(p).warps())[0].instructions[0].operands
        assert got.after_ready_mask == mask


def test_facts_tensor_map_im2col_box_roundtrips(tmp_path):
    u = _unit()
    u.tensor_maps = {1: model.TensorMap(rank=4, element_bytes=2,
                                        box_dim=(8, 8))}
    p = tmp_path / "u.gsc"
    FactsWriter().write_unit(u, p, digest=b"\xAA" * 32)
    tms = FactsReader(p).tensor_maps()
    assert tms[1].box_dim == (8, 8)
    assert tms[1].element_bytes == 2
    assert tms[1].rank == 4
