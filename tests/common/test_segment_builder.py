"""Segment builder: synthetic ClusterUnit -> GSIMSEG -> read back."""

from __future__ import annotations

import pytest

from cyclescope.convert import model
from cyclescope.arch.hopper import rules
from cyclescope.io.fixed_tables import TableReader
from cyclescope.job import segment_builder as sb, segment_format as sf

_PROGRAM_TABLES = {
    "PROG": b"", "NODE": b"", "GRD ": b"", "LEAF": b"",
    "PARM": b"", "PTS ": b"", "FACT": b"", "USES": b"",
    "STRS": b"",
}


def _rows(reader, tag):
    record = sf.RECORDS[tag]
    table = reader.require_table(tag)
    tr = TableReader(record, table)
    return [tr.unpack(i) for i in range(len(tr))]


def _unit() -> model.ClusterUnit:
    statics = [
        model.StaticInstruction(
            pc=0x0, code_address=0x1000, opcode="LDG.E.128",
            operation_class="LDG",
            source_registers=("R2", "R3"),
            destination_registers=("R4", "R5", "R6", "R7"),
            control=model.ControlWord(
                read_barrier=1, write_barrier=2, wait_mask=0x3F),
            memory_space="global"),
        model.StaticInstruction(
            pc=0x10, code_address=0x1010, opcode="STG.E.128",
            operation_class="STG",
            source_registers=("R8", "R9", "R10", "R11"),
            destination_registers=(),
            memory_space="global"),
        model.StaticInstruction(
            pc=0x20, code_address=0x1020, opcode="ATOMG.ADD",
            operation_class="ATOMG",
            memory_space="global"),
        model.StaticInstruction(
            pc=0x30, code_address=0x1030, opcode="UTMALDG.2D",
            operation_class="UTMALDG",
            memory_space="global"),
    ]
    lanes = tuple(0x200000 + i * 128 for i in range(32))
    store_lanes = tuple(0x300000 + i * 64 for i in range(32))
    warps = [
        model.WarpStream(
            cta_linear=0, cta=(0, 0, 0), warp_in_cta=0, sm_id=4,
            warp_slot=8,
            instructions=[
                model.DynamicInstruction(
                    static_index=0, active_mask=0xFFFFFFFF,
                    memory=(model.MemoryReference(
                        space="global", access_kind="load",
                        width_bytes=16, lane_addresses=lanes,
                        lane_mask=0xFFFFFFFF),)),
                model.DynamicInstruction(
                    static_index=1, active_mask=0xFFFFFFFF,
                    memory=(model.MemoryReference(
                        space="global", access_kind="store",
                        width_bytes=16, lane_addresses=store_lanes,
                        lane_mask=0xFFFFFFFF),)),
            ]),
        model.WarpStream(
            cta_linear=0, cta=(0, 0, 0), warp_in_cta=1, sm_id=4,
            warp_slot=9,
            instructions=[
                model.DynamicInstruction(
                    static_index=2, active_mask=0xFFFFFFFF,
                    memory=(model.MemoryReference(
                        space="global", access_kind="atomic",
                        width_bytes=4,
                        lane_addresses=tuple(
                            0x400000 + i * 4 for i in range(32)),
                        lane_mask=0xFFFFFFFF),)),
                model.DynamicInstruction(
                    static_index=3, active_mask=0xFFFFFFFF,
                    memory=(model.MemoryReference(
                        space="global", access_kind="async_copy",
                        width_bytes=32,
                        lane_addresses=tuple(
                            0x800000 + i * 32 for i in range(128)),
                        lane_mask=0),),
                    operands=model.SemanticOperands(
                        uniform={"mbarrier_address": 0x100},
                        tensor_map=0,
                        tensor_coordinates=(0, 0))),
            ]),
    ]
    return model.ClusterUnit(
        cluster_id=0,
        ctas=[model.CtaFacts(cta_linear=0, cta=(0, 0, 0), sm_id=4)],
        warps=warps, statics=statics,
        tensor_maps={0: model.TensorMap(
            rank=2, element_bytes=2, global_dim=(1024, 1024),
            stride_bytes=(2048,), box_dim=(32, 32),
            global_address=0x800000)})


def test_segment_builder_roundtrip() -> None:
    image = sb.build_segment(
        _unit(), rules, _PROGRAM_TABLES,
        {"cluster_shape": (1, 1, 1),
         "tree_digest": b"\1" * 32,
         "program_digest": b"\2" * 32,
         "converter_digest": b"\3" * 32})
    reader = sf.read_segment(image)
    meta = _rows(reader, "META")[0]
    assert meta["instruction_count"] == 4
    inst = _rows(reader, "INST")
    assert len(inst) == 4
    warps = _rows(reader, "WARP")
    assert [w["instance_count"] for w in warps] == [2, 2]
    assert [w["first_instance"] for w in warps] == [0, 2]
    tmpl = _rows(reader, "TMPL")
    assert len(tmpl) == 4
    assert tmpl[0]["scoreboard_read"] == 1
    assert tmpl[0]["scoreboard_write"] == 2
    assert tmpl[0]["scoreboard_wait_mask"] == 0x3F
    assert tmpl[0]["source_count"] == 2
    assert tmpl[0]["destination_count"] == 4
    macc = _rows(reader, "MACC")
    assert len(macc) == 4
    msec = _rows(reader, "MSEC")
    assert macc[0]["sector_count"] == 32
    store_rows = msec[macc[1]["sector_begin"]:
                      macc[1]["sector_begin"] + macc[1]["sector_count"]]
    assert all(r["byte_mask"] != 0 for r in store_rows)
    assert macc[3]["sector_count"] > 0
    code = _rows(reader, "CODE")
    assert len(code) == 4
    assert code[0]["code_address"] == 0x1000


def test_template_dedup() -> None:
    unit = _unit()
    unit.warps[1].instructions.insert(
        0, model.DynamicInstruction(
            static_index=0, active_mask=0xFFFFFFFF,
            memory=(model.MemoryReference(
                space="global", access_kind="load", width_bytes=16,
                lane_addresses=tuple(
                    0x500000 + i * 128 for i in range(32)),
                lane_mask=0xFFFFFFFF),)))
    image = sb.build_segment(unit, rules, _PROGRAM_TABLES, {})
    reader = sf.read_segment(image)
    assert len(_rows(reader, "TMPL")) == 4
    inst = _rows(reader, "INST")
    assert inst[2]["template"] == 0


def test_template_identity_registers_and_control() -> None:
    """Two static instructions of one class share a template only when
    their register operands and control word are identical: the TMPL
    row carries the register refs / scoreboard bits the engine tracks
    dependencies with."""
    unit = _unit()
    base = unit.statics[1]
    same = model.StaticInstruction(
        pc=0x40, code_address=0x1040, opcode="STG.E.128",
        operation_class="STG",
        source_registers=("R8", "R9", "R10", "R11"),
        destination_registers=(), memory_space="global")
    other_regs = model.StaticInstruction(
        pc=0x50, code_address=0x1050, opcode="STG.E.128",
        operation_class="STG",
        source_registers=("R12", "R13", "R14", "R15"),
        destination_registers=(), memory_space="global")
    other_ctrl = model.StaticInstruction(
        pc=0x60, code_address=0x1060, opcode="STG.E.128",
        operation_class="STG",
        source_registers=("R8", "R9", "R10", "R11"),
        destination_registers=(), memory_space="global",
        control=model.ControlWord(wait_mask=0x1, stall=4))
    unit.statics.extend([same, other_regs, other_ctrl])
    store = tuple(0x300000 + i * 64 for i in range(32))
    for si in (4, 5, 6):
        unit.warps[1].instructions.append(model.DynamicInstruction(
            static_index=si, active_mask=0xFFFFFFFF,
            memory=(model.MemoryReference(
                space="global", access_kind="store", width_bytes=16,
                lane_addresses=store, lane_mask=0xFFFFFFFF),)))
    image = sb.build_segment(unit, rules, _PROGRAM_TABLES, {})
    reader = sf.read_segment(image)
    tmpl = _rows(reader, "TMPL")
    inst = _rows(reader, "INST")
    stg = inst[1]["template"]
    assert base.operation_class == "STG"
    assert inst[-3]["template"] == stg            # identical -> shared
    assert inst[-2]["template"] != stg            # other registers
    assert inst[-1]["template"] not in (stg, inst[-2]["template"])
    assert len(tmpl) == 6
    regs = _rows(reader, "REGS")
    t2 = tmpl[inst[-2]["template"]]
    assert [regs[t2["source_begin"] + i]["register"]
            for i in range(t2["source_count"])] == [12, 13, 14, 15]
    t3 = tmpl[inst[-1]["template"]]
    assert t3["scoreboard_wait_mask"] == 0x1 and (t3["control"] & 0xF) == 4


def test_register_namespaces() -> None:
    """REGS rows use the register namespace (R/UR/P/UP/B blocks);
    RZ/PT and other hard-wired names emit no row and drop out of the
    template key, so UR11 and R11 no longer share a template."""
    unit = _unit()
    ur_load = model.StaticInstruction(
        pc=0x40, code_address=0x1040, opcode="LDG.E.128",
        operation_class="LDG",
        source_registers=("UR12", "R22", "RZ", "PT"),
        destination_registers=("P0",), memory_space="global")
    mov_ur = model.StaticInstruction(
        pc=0x50, code_address=0x1050, opcode="MOV",
        operation_class="MOV",
        source_registers=("UR11",), destination_registers=("R7",))
    mov_r = model.StaticInstruction(
        pc=0x60, code_address=0x1060, opcode="MOV",
        operation_class="MOV",
        source_registers=("R11",), destination_registers=("R7",))
    unit.statics.extend([ur_load, mov_ur, mov_r])
    for si in (4, 5, 6):
        unit.warps[0].instructions.append(model.DynamicInstruction(
            static_index=si, active_mask=0xFFFFFFFF))
    image = sb.build_segment(unit, rules, _PROGRAM_TABLES, {})
    reader = sf.read_segment(image)
    tmpl = _rows(reader, "TMPL")
    inst = _rows(reader, "INST")
    regs = _rows(reader, "REGS")
    # warps[0]'s original two instructions precede the appended three
    t_ur = tmpl[inst[2]["template"]]
    assert t_ur["source_count"] == 2
    assert t_ur["destination_count"] == 1
    assert [regs[t_ur["source_begin"] + i]["register"]
            for i in range(2)] == [268, 22]     # UR12, R22; RZ/PT skipped
    assert [regs[t_ur["destination_begin"] + i]["register"]
            for i in range(1)] == [512]         # P0
    # MOV R7, UR11 vs MOV R7, R11: different namespaces -> different
    # templates (digit extraction used to map both to 11)
    assert inst[3]["template"] != inst[4]["template"]


def test_sector_rows_are_per_access() -> None:
    """Lane->sector merge happens inside one access only: a second
    instruction reading the same lane addresses still emits its own
    MSEC rows.  The old segment-global dedup left the second MACC
    with sector_count 0 -> the engine returned zero load latency
    (and constant-space MACCs computed latency at address 0)."""
    unit = _unit()
    dup = unit.warps[0].instructions[0]      # LDG, 32 lanes 0x200000+
    unit.warps[0].instructions.append(model.DynamicInstruction(
        static_index=0, active_mask=0xFFFFFFFF,
        memory=dup.memory))
    # within one access, lanes 0 and 1 share a sector -> merged row
    merged = model.StaticInstruction(
        pc=0x40, code_address=0x1040, opcode="LDG.E",
        operation_class="LDG",
        destination_registers=("R5",), memory_space="global")
    unit.statics.append(merged)
    lanes = tuple([0x600000, 0x600010] +
                  [0x700000 + i * 128 for i in range(30)])
    unit.warps[1].instructions.append(model.DynamicInstruction(
        static_index=4, active_mask=0xFFFFFFFF,
        memory=(model.MemoryReference(
            space="global", access_kind="load", width_bytes=4,
            lane_addresses=lanes, lane_mask=0xFFFFFFFF),)))
    image = sb.build_segment(unit, rules, _PROGRAM_TABLES, {})
    reader = sf.read_segment(image)
    macc = _rows(reader, "MACC")
    msec = _rows(reader, "MSEC")
    first = macc[0]                          # warp0 inst0 LDG
    second = macc[2]                         # warp0 inst2 = the dup
    assert first["sector_count"] == 32
    assert second["sector_count"] == 32      # was 0 before the fix
    assert second["sector_begin"] == macc[1]["sector_begin"] + \
        macc[1]["sector_count"]
    assert msec[second["sector_begin"]]["address"] == 0x200000
    # merged: 32 lanes on 31 sectors -> 31 rows, first row both lanes
    last = macc[-1]
    assert last["sector_count"] == 31
    row0 = msec[last["sector_begin"]]
    assert row0["address"] == 0x600000 and row0["lane_mask"] == 0b11


def test_generic_ld_resolved_space_flags() -> None:
    """A generic LD whose pointer resolved to cluster shared memory
    carries MACC space code 0x1 (0x20 when the reference names a remote
    rank); global_system resolves to 0x0.  These flags drive the
    engine's shared-space check for class-bound storage.shared.* rows."""
    unit = _unit()
    unit.statics.append(model.StaticInstruction(
        pc=0x40, code_address=0x1040, opcode="LD.E.128",
        operation_class="LD",
        destination_registers=("R4", "R5", "R6", "R7"),
        memory_space="generic"))
    idx = len(unit.statics) - 1
    base = 0x7fa300001400
    for space, remote in (("shared_cluster", None),
                          ("shared_cluster", 1),
                          ("global_system", None)):
        unit.warps[0].instructions.append(model.DynamicInstruction(
            static_index=idx, active_mask=0xFFFFFFFF,
            memory=(model.MemoryReference(
                space=space, access_kind="load", width_bytes=16,
                lane_addresses=tuple(base + i * 16 for i in range(32)),
                lane_mask=0xFFFFFFFF, remote_rank=remote),)))
    image = sb.build_segment(unit, rules, _PROGRAM_TABLES, {})
    reader = sf.read_segment(image)
    macc = _rows(reader, "MACC")[2:5]   # warp0's three LD instances
    assert macc[0]["flags"] & 0x7 == 0x1 and not macc[0]["flags"] & 0x20
    assert macc[1]["flags"] == 0x21
    assert macc[2]["flags"] & 0x7 == 0x0 and not macc[2]["flags"] & 0x20


def test_compute_uses_cache_includes_next_static_pc(monkeypatch) -> None:
    """The uses memo key must include _next_static_pc: branch_taken
    reads it, so two instances equal in every other fact but followed
    by different statics must not share cached program rows."""
    from cyclescope.convert import mechanism_uses as muses
    calls = {"n": 0}

    def fake(runtime, static, dyn, context, where, cls=None):
        calls["n"] += 1
        return []
    monkeypatch.setattr(muses, "program_facts", fake)

    b = sb._Builder.__new__(sb._Builder)
    b.runtime = object()          # only checked for None
    b.unit = None
    b.rules = None
    b._launch_obj = None
    b._flow_by_warp = {}
    b._uses_cache = {}
    b._next_static_pc = 0x1000

    dyn = model.DynamicInstruction(
        static_index=0, active_mask=0xFFFFFFFF,
        memory=(model.MemoryReference(
            space="global", access_kind="load", width_bytes=4,
            lane_addresses=(0x100,), lane_mask=1),))
    st = model.StaticInstruction(
        pc=0x10, code_address=None, opcode="BRA", operation_class="BRA")
    import types
    warp = types.SimpleNamespace(warp_slot=0)
    b.compute_uses(st, dyn, warp)
    assert calls["n"] == 1
    # same instance, different next pc -> recomputed
    b._next_static_pc = 0x1030
    b.compute_uses(st, dyn, warp)
    assert calls["n"] == 2
    # same next pc again -> cache hit
    b.compute_uses(st, dyn, warp)
    assert calls["n"] == 2


def _mk_static(pc, opcode, operands, dsts, srcs, reuse=0):
    return model.StaticInstruction(
        pc=pc, code_address=pc + 0x1000, opcode=opcode,
        operation_class=opcode.split(".")[0], operands=tuple(operands),
        destination_registers=tuple(dsts), source_registers=tuple(srcs),
        control=model.ControlWord(reuse_mask=reuse))


def test_src_flags_slots() -> None:
    """REGS flags (operand_read): slot j counts every token after the
    register destination(s) and the plain secondary predicate
    destinations; served = predecessor at pc-16 carries the reuse bit
    on that slot with the same register."""
    b = sb._Builder.__new__(sb._Builder)
    b._src_flags_cache = {}
    b.regs_flag_fallbacks = 0
    # IADD3 R3, PT, PT, R21, -0x4498517b, RZ  reuse bit0 -> R21 served
    p0 = _mk_static(0, "IADD3", ["R3", "PT", "PT", "R21.reuse",
                                 "-0x4498517b", "RZ"],
                    ["R3"], ["PT", "PT", "R21", "RZ"], reuse=1)
    i0 = _mk_static(16, "IADD3", ["R4", "PT", "PT", "R21",
                                  "-0x4498517b", "RZ"],
                    ["R4"], ["PT", "PT", "R21", "RZ"])
    b._static_by_pc = {0: p0, 16: i0}
    assert b._src_flags(i0) == (1,)
    # ISETP P2, PT, R37, R28, PT  reuse bit1 -> R28 served
    p1 = _mk_static(32, "ISETP.GE.U32.AND",
                    ["P2", "PT", "R37", "R28.reuse", "PT"],
                    ["P2"], ["PT", "R37", "R28", "PT"], reuse=2)
    i1 = _mk_static(48, "ISETP.GE.U32.AND",
                    ["P3", "PT", "R37", "R28", "PT"],
                    ["P3"], ["PT", "R37", "R28", "PT"])
    b._static_by_pc.update({32: p1, 48: i1})
    assert b._src_flags(i1) == (0, 1)
    # FFMA d, a.reuse, b.reuse, d: 3 slots all served
    p2 = _mk_static(64, "FFMA", ["R10", "R10.reuse", "R2.reuse",
                                 "R3.reuse"],
                    ["R10"], ["R10", "R2", "R3"], reuse=7)
    i2 = _mk_static(80, "FFMA", ["R10", "R10", "R2", "R3"],
                    ["R10"], ["R10", "R2", "R3"])
    b._static_by_pc.update({64: p2, 80: i2})
    assert b._src_flags(i2) == (1, 1, 1)
    # FFMA2: ".F32x2" rows are pairs, a ".F32" scalar is not
    i3 = _mk_static(96, "FFMA2", ["R4", "R36.F32x2.HI_LO",
                                  "R40.F32", "R4.F32x2.HI_LO"],
                    ["R4"], ["R36", "R40", "R4"])
    b._static_by_pc.update({96: i3})
    assert b._src_flags(i3) == (2, 0, 2)
    # ".64" suffix sets bit1 on that operand only (non-pair class)
    i4 = _mk_static(112, "LDG", ["R2", "R4.64", "R6"],
                    ["R2"], ["R4.64", "R6"])
    b._static_by_pc.update({112: i4})
    assert b._src_flags(i4) == (2, 0)
    # FFMA2: ".F32x2" operands are pairs, a ".F32" broadcast scalar is
    # not (R68 served+pair=3, R75=0, R16 pair=2)
    p6 = _mk_static(144, "FFMA2",
                    ["R16", "R68.reuse.F32x2.HI_LO", "R75.F32",
                     "R16.F32x2.HI_LO"],
                    ["R16"], ["R68", "R75", "R16"], reuse=1)
    i6 = _mk_static(160, "FFMA2",
                    ["R16", "R68.F32x2.HI_LO", "R75.F32",
                     "R16.F32x2.HI_LO"],
                    ["R16"], ["R68", "R75", "R16"])
    b._static_by_pc.update({144: p6, 160: i6})
    assert b._src_flags(i6) == (3, 0, 2)
    # mismatch fallback: operands don't line up with source_registers
    i5 = _mk_static(128, "FFMA", ["R10", "R10", "R2"],
                    ["R10"], ["R10", "R2", "R3"])
    b._static_by_pc.update({128: i5})
    assert b._src_flags(i5) == (0, 0, 0)
    assert b.regs_flag_fallbacks == 1


def test_cpasync_count_resets_on_ldgdepbar() -> None:
    """copies_per_group: a cp.async group ends at ARRIVES or LDGDEPBAR
    (commit_group kernels have no ARRIVES); the LDGSTS after LDGDEPBAR
    must see count 0 again."""
    from cyclescope.convert.axes import FlowContext

    statics = [
        _mk_static(0x0, "LDGSTS.E.BYPASS", [], [], []),
        _mk_static(0x10, "LDGDEPBAR", [], [], []),
        _mk_static(0x20, "ARRIVES.LDGSTSBAR.64", [], [], []),
    ]
    unit = model.ClusterUnit(
        cluster_id=0,
        ctas=[model.CtaFacts(cta_linear=0, cta=(0, 0, 0), sm_id=4)],
        warps=[], statics=statics, tensor_maps={})
    b = sb._Builder(unit, _PROGRAM_TABLES, {})
    b.rules = rules
    flow = FlowContext()

    def emit(si):
        b.emit_instance(
            model.DynamicInstruction(
                static_index=si, active_mask=0xFFFFFFFF,
                predicate_mask=0xFFFFFFFF),
            None, flow)

    emit(0); emit(0)
    assert flow.cpasync_count == 2
    emit(1)                    # LDGDEPBAR ends the group
    assert flow.cpasync_count == 0
    emit(0)
    assert flow.cpasync_count == 1
    emit(2)                    # ARRIVES still resets
    assert flow.cpasync_count == 0


def test_local_load_precise_byte_mask() -> None:
    """load_after_copies v3: a 4-byte local load covers only its real
    bytes of a trace sector (byte_mask 0xF << offset); global loads
    keep the full-sector mask."""
    unit = _unit()
    unit.warps[0].instructions.append(
        model.DynamicInstruction(
            static_index=0, active_mask=0xFFFFFFFF,
            memory=(model.MemoryReference(
                space="local", access_kind="load", width_bytes=4,
                lane_addresses=tuple(0x1C + i * 0x80 for i in range(32)),
                lane_mask=0xFFFFFFFF),)))
    image = sb.build_segment(unit, rules, _PROGRAM_TABLES, {})
    reader = sf.read_segment(image)
    macc = _rows(reader, "MACC")
    msec = _rows(reader, "MSEC")
    lm = next(m for m in macc if (m["flags"] & 0x7) == 2)
    rows = msec[lm["sector_begin"]:
                lm["sector_begin"] + lm["sector_count"]]
    assert rows and all(
        r["byte_mask"] == (0xF << 28) & 0xFFFFFFFF for r in rows)
    gm = macc[0]
    grows = msec[gm["sector_begin"]:
                 gm["sector_begin"] + gm["sector_count"]]
    assert all(r["byte_mask"] == 0xFFFFFFFF for r in grows)
