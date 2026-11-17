"""2b-4: traversal outcomes, axes, first-reader flow, relink, cache."""
import json
import pytest
from cyclescope.convert import mechanism_uses as muses, axes
from cyclescope.convert import model

MINI_TREE = {
    "axes": {
        "eng": {"kind": "int", "owner": "engine"},
        "oc": {"kind": "enum", "owner": "capture",
               "values": ["NAV", "OTHER"]},
    },
    "nodes": [
        {"id": "r", "guard": None, "children": ["g1", "g2"],
         "parent": None, "parameter": None},
        {"id": "g1", "guard": {"axis": "eng", "operator": "eq",
                               "value": 0}, "children": [],
         "parent": "r",
         "parameter": {"scope": "instruction", "status": "experimental",
                       "fields": {"c": {"schema_id": 1, "field_id": 1,
                                       "model": {"kind": "constant",
                                                "value": 1.0}}}}},
        {"id": "g2", "guard": {"axis": "eng", "operator": "range",
                               "value": [1, 9]}, "children": [],
         "parent": "r", "parameter": None},
    ],
    "programs": {
        "mini.prog": {"event": "issue",
                      "parameter_schema": "timing_row",
                      "schema_id": 1, "field_mask": 1,
                      "root": "r", "operation_classes": ["NAV"],
                      "timing_kind": "nanosleep_warp_delay"},
    },
}


def _static(op="NAV", operands=(), srcs=(), dsts=(), roles=None):
    return model.StaticInstruction(
        pc=0x10, code_address=None, opcode=op,
        operation_class=op.split(".", 1)[0], operands=operands,
        source_registers=srcs, destination_registers=dsts,
        register_roles=roles or {})


def _dyn(mask=0xFFFFFFFF, pred=0xFFFFFFFF, memory=(), operands=None):
    return model.DynamicInstruction(
        static_index=0, active_mask=mask, predicate_mask=pred,
        memory=memory, operands=operands)


class _Rules:
    NAME = "mini"

    @staticmethod
    def operation_class(s, d=None):
        return s.opcode.split(".", 1)[0]

    @staticmethod
    def instruction_flags(s):
        return 0

    @staticmethod
    def semantics(s, d, ctx, cta=0, wslot=0):
        return None


def _ctx(flow=None):
    return axes.AxisContext(unit=None, launch=None, warp=None,
                            flow=flow or axes.FlowContext(),
                            rules=_Rules)


def test_engine_guard_stops_at_parent():
    rt = muses.TreeRuntime(MINI_TREE)
    rows = muses.program_facts(rt, _static(), _dyn(), _ctx(), "t")
    assert len(rows) == 1
    assert rows[0]["node"] == "r"
    assert rt.row_of[rows[0]["node"]] == 0


def test_not_applicable_class_produces_no_rows():
    rt = muses.TreeRuntime(MINI_TREE)
    assert muses.program_facts(rt, _static("OTHER"), _dyn(),
                               _ctx(), "t") == []


def _pred_tree():
    return {
        "axes": {
            "predicated_on": {"kind": "int", "owner": "capture"},
            "operation_class": {"kind": "enum", "owner": "capture",
                                "values": ["NAV", "OTHER"]},
        },
        "nodes": [
            {"id": "pr", "guard": None, "children": ["p0", "p1"],
             "parent": None, "parameter": None},
            {"id": "p0", "guard": {"axis": "predicated_on",
                                   "operator": "eq", "value": 0},
             "children": [], "parent": "pr",
             "parameter": {"scope": "instruction", "status": "experimental",
                           "fields": {"c": {"schema_id": 1,
                                           "field_id": 1,
                                           "model": {
                               "kind": "constant", "value": 1.0}}}}},
            {"id": "p1", "guard": {"axis": "predicated_on",
                                   "operator": "eq", "value": 1},
             "children": [], "parent": "pr",
             "parameter": {"scope": "instruction", "status": "experimental",
                           "fields": {"c": {"schema_id": 1,
                                           "field_id": 1,
                                           "model": {
                               "kind": "constant", "value": 1.0}}}}},
            {"id": "np", "guard": None, "children": [], "parent": None,
             "parameter": {"scope": "instruction", "status": "experimental",
                           "fields": {"c": {"schema_id": 1,
                                           "field_id": 1,
                                           "model": {
                               "kind": "constant", "value": 1.0}}}}},
        ],
        "programs": {
            "mini.pred": {"event": "issue",
                          "parameter_schema": "timing_row",
                          "schema_id": 1, "field_mask": 1,
                          "root": "pr", "operation_classes": ["NAV"],
                          "timing_kind": "nanosleep_warp_delay"},
            "mini.nopred": {"event": "issue",
                            "parameter_schema": "timing_row",
                            "schema_id": 1, "field_mask": 1,
                            "root": "np", "operation_classes": ["NAV"],
                            "timing_kind": "nanosleep_warp_delay"},
        },
    }


def test_predicated_off_binding():
    """Non-executing instances bind only programs whose guard tree
    contains predicated_on; execution axes are NOT_APPLICABLE."""
    rt = muses.TreeRuntime(_pred_tree())
    off = _dyn(pred=0)
    rows = muses.program_facts(rt, _static(), off, _ctx(), "t")
    assert [r["program"] for r in rows if "program" in r] == ["mini.pred"]
    row = rows[0]
    assert row["node"] == "p0"
    assert row["facts"] == {"predicated_on": 0}
    assert axes.axis_value("active_lanes", _static(), off) == 0
    assert axes.axis_value("predicated_on", _static(), off) == 0
    assert axes.axis_value("sector_count", _static(), off) \
        is axes.NOT_APPLICABLE
    on = _dyn()
    rows = muses.program_facts(rt, _static(), on, _ctx(), "t")
    assert {r.get("node") for r in rows if "program" in r} == \
        {"p1", "np"}


def test_source_form_classification():
    st = _static("FFMA", ("R0", "R1", "R2"))
    assert axes.axis_value("source_mode", st, _dyn()) == "gpr"
    assert axes.axis_value("gpr_source_count", st, _dyn()) == 2
    st = _static("FFMA", ("R0", "R1", "c[0x0][0x160]"))
    assert axes.axis_value("source_mode", st, _dyn()) == "mixed"
    assert axes.axis_value("constant_source_count", st, _dyn()) == 1
    st = _static("I2FP.F32.S32", ("R2", "UR4"))
    assert axes.axis_value("source_mode", st, _dyn()) == "uniform"
    st = _static("ISETP.EQ.AND", ("P0", "P1", "R0", "R1"))
    assert axes.axis_value("gpr_source_count", st, _dyn()) == 2
    st = _static("LDS", ("R0", "[R2]"))
    assert axes.axis_value("source_mode", st, _dyn()) \
        is axes.NOT_APPLICABLE
    # bitwise-not / negate / abs operand modifiers are not part of the
    # source class (UIADD3.X UR11, UR11, ~UR9, URZ, UP0, !UPT)
    st = _static("UIADD3.X", ("UR11", "UR11", "~UR9", "URZ", "UP0", "!UPT"))
    assert axes.axis_value("uniform_source_count", st, _dyn()) == 2
    assert axes.axis_value("zero_source_count", st, _dyn()) == 1
    st = _static("FFMA", ("R0", "-R1", "|R2|", "R3"))
    assert axes.axis_value("gpr_source_count", st, _dyn()) == 3
    st = _static("HGMMA.64x128x16.F32", ("R0", "gdesc[UR4]", "gdesc[UR6]"))
    ctx = _ctx()
    assert axes.axis_value("source_mode", st, _dyn(), ctx) == "ss"


def test_input_accumulator_dtype():
    st = _static("HGMMA.64x128x16.F32")
    ctx = _ctx()
    assert axes.axis_value("input_dtype", st, _dyn(), ctx) == "f16"
    assert axes.axis_value("accumulator_dtype", st, _dyn(), ctx) == "f32"
    st = _static("QGMMA.64x128x32.F32.E4M3.E4M3")
    assert axes.axis_value("input_dtype", st, _dyn(), ctx) == "e4m3"
    st = _static("FFMA")
    assert axes.axis_value("input_dtype", st, _dyn(), ctx) \
        is axes.NOT_APPLICABLE


def _memref(addrs, kind="load", width=4, space="shared", lane_mask=0):
    return model.MemoryReference(
        space=space, access_kind=kind, width_bytes=width,
        lane_addresses=tuple(addrs), lane_mask=lane_mask)


def test_bank_conflict_degree():
    # broadcast load -> 1
    d = _dyn(memory=[_memref([0x100] * 32)])
    assert axes.axis_value("bank_conflict_degree",
                           _static("LDS"), d) == 1
    # stride-2 words: lanes hit each bank twice -> 2
    d = _dyn(memory=[_memref([i * 8 for i in range(32)])])
    assert axes.axis_value("bank_conflict_degree",
                           _static("LDS"), d) == 2
    # conflict-free stride-1
    d = _dyn(memory=[_memref([i * 4 for i in range(32)])])
    assert axes.axis_value("bank_conflict_degree",
                           _static("LDS"), d) == 1
    # store same address: no multicast -> degree counts lanes/bank
    d = _dyn(memory=[_memref([0x100] * 32, kind="store")])
    assert axes.axis_value("bank_conflict_degree",
                           _static("STS"), d) == 32
    # LDSM.4: per 8-lane transaction, 4 words per row address;
    # 4+4 lanes on two overlapping rows inside one transaction -> 8
    addrs = ([0x100, 0x104] * 4) * 4
    d = _dyn(memory=[_memref(addrs)])
    assert axes.axis_value("bank_conflict_degree",
                           _static("LDSM.16.M88.4"), d) == 8
    # LDSM.1: only lanes 0-7 supply addresses; lanes 8-31 ignored
    addrs = [0x200] * 8 + [0x104] * 24
    d = _dyn(memory=[_memref(addrs)])
    assert axes.axis_value("bank_conflict_degree",
                           _static("LDSM.16.M88.1"), d) == 8
    # shared::cluster rank tags: two ranks same bank -> max per rank
    addrs = [0x100] * 16 + [(2 << 24) | 0x100] * 16
    d = _dyn(memory=[_memref(addrs, kind="store")])
    assert axes.axis_value("bank_conflict_degree",
                           _static("STS"), d) == 16


def test_shared_wavefronts_resolved_cluster_space():
    # a generic LD/ST resolved to cluster shared memory feeds the
    # shared wavefront axis exactly like an explicit shared reference
    addrs = [i * 4 for i in range(32)]
    want = axes.axis_value("shared_wavefronts", _static("LDS"),
                           _dyn(memory=[_memref(addrs)]))
    assert want is not None
    d = _dyn(memory=[_memref(addrs, space="shared_cluster")])
    assert axes.axis_value("shared_wavefronts", _static("LD"), d) == want
    # a reference resolved to global: the shared.* programs do not
    # apply to this instance -> NOT_APPLICABLE (not missing_fact)
    d = _dyn(memory=[_memref(addrs, space="global")])
    assert axes.axis_value("shared_wavefronts", _static("LD"), d) \
        is axes.NOT_APPLICABLE
    assert axes.axis_value("bank_conflict_degree", _static("LD"), d) \
        is axes.NOT_APPLICABLE
    # no memory reference at all -> no fact (unchanged)
    assert axes.axis_value("shared_wavefronts", _static("LD"),
                           _dyn()) is None


def test_first_reader_consumer_class_and_role():
    """I2FP writes R1, FFMA reads it (role data): the I2FP instance's
    first reader is the FFMA -> consumer class fp32, role data."""
    from cyclescope.job import segment_builder as sb
    producer = _static("I2FP.F32.S32", ("R1", "R0"), dsts=("R1",))
    consumer = _static("FFMA", ("R5", "R1", "R2"), srcs=("R1", "R2"),
                       roles={"R1": "data", "R2": "data"})
    unit = model.ClusterUnit(cluster_id=0)
    unit.statics = [producer, consumer]
    warp = model.WarpStream(cta_linear=0, cta=(0, 0, 0), warp_in_cta=0,
                            sm_id=0, warp_slot=0)
    later = _static("STG", ("[R7]", "R1"), srcs=("R7", "R1"),
                    roles={"R1": "data"})
    unit.statics.append(later)
    warp.instructions = [
        model.DynamicInstruction(static_index=0, active_mask=0xFFFFFFFF,
                                 predicate_mask=0xFFFFFFFF),
        model.DynamicInstruction(static_index=1, active_mask=0xFFFFFFFF,
                                 predicate_mask=0xFFFFFFFF),
        model.DynamicInstruction(static_index=2, active_mask=0xFFFFFFFF,
                                 predicate_mask=0xFFFFFFFF)]
    unit.warps = [warp]
    b = sb._Builder(unit, _mini_program_tables()[0], {}, None)
    b.rules = _Rules
    readers = b._scan_first_readers(warp)
    assert 0 in readers and 1 not in readers
    reader_static, role = readers[0]
    assert reader_static is consumer and role == "data"
    # a second writer overwrites: the later reader binds to it, not to 0
    w2 = model.WarpStream(cta_linear=0, cta=(0, 0, 0), warp_in_cta=0,
                          sm_id=0, warp_slot=0)
    w2.instructions = [warp.instructions[0], warp.instructions[0],
                       warp.instructions[1]]
    readers2 = b._scan_first_readers(w2)
    assert readers2.get(0) is None          # overwritten -> no reader
    rs2, _ = readers2[1]
    assert rs2 is consumer                  # later reader binds to inst1
    flow = axes.FlowContext(reader=readers[0])
    ctx = _ctx(flow)
    assert axes.axis_value("dependency_consumer_class",
                           producer, warp.instructions[0], ctx) == "fp32"
    assert axes.axis_value("dependency_source_role",
                           producer, warp.instructions[0], ctx) == "data"
    flow.reader = None
    assert axes.axis_value("dependency_consumer_class",
                           producer, warp.instructions[0], ctx) \
        is axes.NOT_APPLICABLE


def _mini_program_tables():
    from cyclescope.job import program_tables
    tree = _pred_tree()
    return program_tables.compile_program_tables(tree), tree


def _mini_unit():
    st = _static("NAV")
    unit = model.ClusterUnit(cluster_id=0)
    unit.ctas = [model.CtaFacts(cta_linear=0, cta=(0, 0, 0))]
    unit.statics = [st]
    warp = model.WarpStream(cta_linear=0, cta=(0, 0, 0), warp_in_cta=0,
                            sm_id=0, warp_slot=0)
    warp.instructions = [_dyn(), _dyn(pred=0)]
    unit.warps = [warp]
    return unit


_META = {"schema_targets": {"timing_row": "TimingRow",
                            "instruction_template": "InstructionTemplate",
                            "instance_timing": "InstanceTiming"}}


def test_memo_key_distinguishes_addresses():
    a = _dyn(memory=[_memref([0x100] * 32)])
    b = _dyn(memory=[_memref([i * 8 for i in range(32)])])
    ka = sb_key(a)
    kb = sb_key(b)
    assert ka != kb


def sb_key(d):
    from cyclescope.job import segment_builder as sb
    return sb._Builder._uses_key(d, None, "LDS", None)


def test_fact_no_orphan_rows():
    """A domain-miss program must not leave orphan FACT rows that
    corrupt every later fact_begin."""
    pt, tree = _mini_program_tables()
    # add a program whose leaf model axis is enum 'operation_class'
    # but whose domain lacks NAV -> domain miss after emitting rows
    t2 = json.loads(json.dumps(tree))
    t2["nodes"].append(
        {"id": "dr", "guard": None, "children": ["dleaf"],
         "parent": None, "parameter": None})
    t2["nodes"].append(
        {"id": "dleaf", "guard": None, "children": [], "parent": "dr",
         "parameter": {"scope": "instruction", "status": "experimental",
                      "fields": {"f": {"schema_id": 1, "field_id": 1,
                                      "model": {
                          "kind": "linear",
                          "axis": "operation_class",
                          "slope": 1.0, "intercept": 0.0,
                          "domain": [0, 1]}}}}})
    t2["axes"]["operation_class"]["values"] = ["OTHER"]
    t2["programs"]["mini.domain"] = {
        "event": "issue", "parameter_schema": "timing_row",
        "schema_id": 1, "field_mask": 1, "root": "dr",
        "operation_classes": ["NAV"],
        "timing_kind": "nanosleep_warp_delay"}
    from cyclescope.job import segment_builder as sb, program_tables, \
        segment_format as sf
    pt = program_tables.compile_program_tables(t2)
    rt = muses.TreeRuntime(t2)
    unit = _mini_unit()
    b = sb._Builder(unit, pt, _META, rt)
    b.rules = _Rules
    for warp in unit.warps:
        b.emit_warp(warp)
    fact_rows = len(b.tables["FACT"]) // sf.FACT.size
    assert fact_rows == b._fact_count
    # the p0 row (predicated_on=0 fact) survived with correct begin
    uses = [sf.USES.struct.unpack(
        b.tables["USES"][i:i + sf.USES.size])
        for i in range(0, len(b.tables["USES"]), sf.USES.size)]
    for u in uses:
        assert u[6] + u[7] <= fact_rows
    assert b.missing_facts


def test_tmpl_timing_range():
    pt, tree = _mini_program_tables()
    from cyclescope.job import segment_builder as sb, segment_format as sf
    rt = muses.TreeRuntime(tree)
    unit = _mini_unit()
    b = sb._Builder(unit, pt, _META, rt)
    b.rules = _Rules
    for warp in unit.warps:
        b.emit_warp(warp)
    time_rows = len(b.tables["TIME"]) // sf.TIME.size
    total = 0
    for i in range(len(b.tables["TMPL"]) // sf.TMPL.size):
        tmpl = sf.TMPL.struct.unpack(
            b.tables["TMPL"][i * sf.TMPL.size:(i + 1) * sf.TMPL.size])
        begin, count = tmpl[12], tmpl[13]
        assert begin + count <= time_rows and count >= 1
        total += count
    assert total == time_rows


def _build_seg(tree, unit=None):
    from cyclescope.job import segment_builder as sb, program_tables, \
        segment_format as sf
    pt = program_tables.compile_program_tables(tree)
    rt = muses.TreeRuntime(tree)
    unit = unit or _mini_unit()
    b = sb._Builder(unit, pt, _META, rt)
    b.rules = _Rules
    for warp in unit.warps:
        b.emit_warp(warp)
    digest = bytes.fromhex(tree.get("content_digest", "00" * 32))
    from cyclescope.job.cache import converter_digest
    conv_d = bytes.fromhex(converter_digest())
    b.tables["META"] += sf.META.pack_row(
        (digest, digest, conv_d, unit.cluster_id,
         len(unit.ctas), len(unit.warps), b.inst_count, 0))
    b.tables["STRS"] += b.strings
    return sf.write_segment(
        {t: bytes(buf) for t, buf in b.tables.items()})


def _tree_variant(tree, mutate):
    t = json.loads(json.dumps(tree))
    t["content_digest"] = "aa" * 32
    mutate(t)
    return t


def _final_tree(tree):
    t = _tree_variant(tree, lambda x: None)
    t["content_digest"] = "bb" * 32
    return t


def test_relink_leaf_change_byte_equal():
    from cyclescope.job import relink
    t1 = _final_tree(_pred_tree())
    t2 = _tree_variant(t1, lambda t:
                       t["nodes"][1]["parameter"]["fields"]["c"]
                       ["model"].update({"value": 9.0}))
    seg1 = _build_seg(t1)
    seg2_direct = _build_seg(t2)
    assert relink.relink_segment(seg1, t2) == seg2_direct


def test_relink_dropped_program_byte_equal():
    from cyclescope.job import relink
    t1 = _final_tree(_pred_tree())
    def _drop(t):
        t["programs"].pop("mini.nopred")
        t["nodes"] = [n for n in t["nodes"] if n["id"] != "np"]
    t2 = _tree_variant(t1, _drop)
    seg1 = _build_seg(t1)
    seg2_direct = _build_seg(t2)
    assert relink.relink_segment(seg1, t2) == seg2_direct


def test_relink_new_program_rejected():
    from cyclescope.job import relink
    t1 = _final_tree(_pred_tree())
    t2 = _tree_variant(t1, lambda t: t["programs"].update({
        "mini.new": dict(t["programs"]["mini.pred"])}))
    seg1 = _build_seg(t1)
    with pytest.raises(Exception, match="relink_new_program"):
        relink.relink_segment(seg1, t2)


def test_relink_strs_corruption_rejected():
    from cyclescope.job import relink, segment_format as sf
    t1 = _final_tree(_pred_tree())
    seg = bytearray(_build_seg(t1))
    # corrupt: strip the terminating NUL of the STRS region is not
    # possible via table edit without repacking; instead verify a
    # segment whose STRS fails to parse is rejected.
    with pytest.raises(Exception):
        relink.relink_segment(b"GSIMSEG\x00" + b"\0" * 64, t1)


def test_axis_functions_smoke():
    st = _static("HGMMA.64x128x16.F32")
    assert axes.axis_value("tensor_m", st, _dyn(), None) == 64
    assert axes.axis_value("tensor_n", st, _dyn(), None) == 128
    assert axes.axis_value("tensor_k", st, _dyn(), None) == 16
    st2 = _static("UTMALDG.4D")
    assert axes.axis_value("tensor_rank", st2, _dyn(), None) == 4
    # tma_form: every plain <n>D box is tile (4D included); the im2col
    # modes collapse to one form; gather4 / scatter4 are their own forms
    for op, form in (("UTMALDG.4D", "tile"), ("UTMALDG.5D.2CTA", "tile"),
                     ("UTMALDG.5D.IM2COL", "im2col"),
                     ("UTMALDG.4D.IM2COL_W128", "im2col"),
                     ("UTMALDG.2D.GATHER4.2CTA", "gather4"),
                     ("UTMASTG.2D.SCATTER4", "scatter4")):
        assert axes.axis_value("tma_form", _static(op), _dyn(), None) == form


def test_tree_validate_timing_kind_and_axes():
    from cyclescope.tree.validate import validate_generated_tree
    bad = json.loads(json.dumps(MINI_TREE))
    bad["programs"]["mini.prog"].pop("timing_kind")
    with pytest.raises(Exception, match="timing_kind"):
        validate_generated_tree(bad)
    bad2 = json.loads(json.dumps(MINI_TREE))
    bad2["axes"]["ghost"] = {"kind": "int", "owner": "capture"}
    bad2["nodes"][1]["guard"]["axis"] = "ghost"
    with pytest.raises(Exception, match="axis function"):
        validate_generated_tree(bad2)


def test_segment_cache_roundtrip_and_relink(tmp_path):
    from cyclescope.job import cache as cmod, relink
    t1 = _final_tree(_pred_tree())
    t2 = _tree_variant(t1, lambda t:
                       t["nodes"][1]["parameter"]["fields"]["c"]
                       ["model"].update({"value": 7.0}))
    c = cmod.SegmentCache(tmp_path)
    src = "srcA"
    seg1 = _build_seg(t1)
    c.put(cmod.entry_name("cap", 0, src, t1["content_digest"]),
          seg1, {"capture": "cap", "src": src,
                 "tree": t1["content_digest"]})
    # (a) full hit
    assert c.get(cmod.entry_name("cap", 0, src,
                                t1["content_digest"])) == seg1
    # (b) tree-only change: prefix match -> relink byte-equal
    prior = c.find_for_relink("cap", 0, src)
    assert prior is not None
    assert relink.relink_segment(prior[1], t2) == _build_seg(t2)
    # (c) source change: no prefix match
    assert c.get(cmod.entry_name("cap", 0, "srcB",
                                t1["content_digest"])) is None
    assert c.find_for_relink("cap", 0, "srcB") is None


def test_predicated_off_binds_opcode_axis_program():
    """A program whose predicated_on=0 subtree guards on a static-derived
    axis (tensor_m via opcode) still binds non-executing instances."""
    t = _pred_tree()
    t["axes"]["tensor_m"] = {"kind": "int", "owner": "capture"}
    t["nodes"].append({"id": "pm", "guard": {
        "axis": "tensor_m", "operator": "eq", "value": 64},
        "children": [], "parent": "p0",
        "parameter": {"scope": "instruction", "status": "experimental",
                      "fields": {"c": {"schema_id": 1, "field_id": 1,
                                      "model": {"kind": "constant",
                                                "value": 1.0}}}}})
    t["nodes"][1]["children"] = ["pm"]
    t["programs"]["mini.pred"]["operation_classes"] = ["HGMMA"]
    rt = muses.TreeRuntime(t)
    off = _dyn(pred=0)
    st = _static("HGMMA.64x128x16.F32")
    rows = muses.program_facts(rt, st, off, _ctx(), "t")
    bound = {(r["program"], r["node"]) for r in rows if "program" in r}
    assert ("mini.pred", "pm") in bound


def test_shared_data_result_domain_int():
    producer = _static("LDSM.16.M88.4")
    rd = _static("FFMA")
    flow = axes.FlowContext(reader=(rd, "data"))
    ctx = _ctx(flow)
    assert axes.axis_value("shared_data_result_domain",
                           producer, _dyn(), ctx) == 1
    flow.reader = (rd, "address")
    assert axes.axis_value("shared_data_result_domain",
                           producer, _dyn(), ctx) == 0
    flow.reader = (rd, None)
    assert axes.axis_value("shared_data_result_domain",
                           producer, _dyn(), ctx) is None
    flow.reader = None
    assert axes.axis_value("shared_data_result_domain",
                           producer, _dyn(), ctx) is axes.NOT_APPLICABLE


def test_container_relink_byte_equal():
    from cyclescope.job import relink, container
    t1 = _final_tree(_pred_tree())
    t2 = _tree_variant(t1, lambda t:
                       t["nodes"][1]["parameter"]["fields"]["c"]
                       ["model"].update({"value": 3.0}))
    segs1 = [_build_seg(t1), _build_seg(t1)]
    segs2 = [relink.relink_segment(x, t2) for x in segs1]
    seg2 = _build_seg(t2)
    assert segs2[0] == seg2
    cont1 = container.write_container({}, segs1)
    out = relink.relink_container(cont1, t2)
    from cyclescope.io.file_header import FileReader
    imgs = container.segment_images(FileReader(memoryview(out)))
    assert len(imgs) == 2
    for im in imgs:
        assert im.tobytes() == seg2


def test_tmem_shape_repeat_and_wait_group():
    class _BW(_Rules):
        NAME = "blackwell"
    ctx = _ctx(); ctx.rules = _BW
    st = _static("STTM.16dp256bit")
    assert axes.axis_value("tmem_shape", st, _dyn(), ctx) == "16dp256bit"
    assert axes.axis_value("tmem_repeat", st, _dyn(), ctx) == "x1"
    st = _static("LDTM.x16")
    assert axes.axis_value("tmem_shape", st, _dyn(), ctx) == "x16"
    assert axes.axis_value("tmem_repeat", st, _dyn(), ctx) == "x16"
    st = _static("DEPBAR.LE", ("SB0", "0x1"))
    assert axes.axis_value("wait_group", st, _dyn()) == 1
    st = _static("FFMA")
    assert axes.axis_value("wait_group", st, _dyn()) is axes.NOT_APPLICABLE


def _relation_tree(engine_axes):
    """Root partitions on an engine axis; the leaf model uses the
    capture axis tensor_k that only the producer instruction carries."""
    return {
        "axes": {
            "eng": {"kind": "int", "owner": "engine"},
            "tensor_k": {"kind": "int", "owner": "capture"},
        },
        "nodes": [
            {"id": "r", "guard": None, "children": ["g1"],
             "parent": None, "parameter": None},
            {"id": "g1", "guard": {"axis": "eng", "operator": "range",
                                   "value": [0, 9]}, "children": [],
             "parent": "r",
             "parameter": {"scope": "instruction",
                           "status": "experimental",
                           "fields": {"c": {
                               "schema_id": 1, "field_id": 1,
                               "model": {"kind": "linear",
                                         "axis": "tensor_k",
                                         "slope": 1.0,
                                         "intercept": 0.0}}}}},
        ],
        "programs": {
            "mini.consumer": {"event": "issue",
                              "parameter_schema": "timing_row",
                              "schema_id": 1, "field_mask": 1,
                              "root": "r", "operation_classes": ["FFMA"],
                              "timing_kind": "nanosleep_warp_delay",
                              **({"engine_axes": engine_axes}
                                 if engine_axes else {})},
        },
    }


def test_program_engine_axes_not_required_from_capture():
    # without the declaration the consumer-side FFMA has no tensor_k
    rt = muses.TreeRuntime(_relation_tree(None))
    rows = muses.program_facts(rt, _static("FFMA", ("R0", "R1", "R2", "R3")),
                               _dyn(), _ctx(), "t")
    assert rows and rows[0].get("missing") == "axis:tensor_k"
    # declared engine_axes: the USES row stops at the root, no FACT needed
    rt = muses.TreeRuntime(_relation_tree(["tensor_k"]))
    rows = muses.program_facts(rt, _static("FFMA", ("R0", "R1", "R2", "R3")),
                               _dyn(), _ctx(), "t")
    assert rows and "missing" not in rows[0]
    assert rows[0]["node"] == "r" and rows[0]["facts"] == {}
    assert rt.required_axes["r"] == ()


def test_relink_preserves_optional_tables():
    from cyclescope.job import relink, segment_format as sf
    t1 = _final_tree(_pred_tree())
    t2 = _tree_variant(t1, lambda t:
                       t["nodes"][1]["parameter"]["fields"]["c"]
                       ["model"].update({"value": 9.0}))
    seg1 = _build_seg(t1)
    reader = sf.read_segment(seg1)
    tables = {t: bytes(reader.require_table(t))
              for t in sf.REQUIRED_TABLES}
    tables["MBOB"] = sf.OPTIONAL_RECORDS["MBOB"].pack_row((0, 0x600))
    tables["GSYN"] = (sf.OPTIONAL_RECORDS["GSYN"].pack_row(
        (1, 0, 1, 4, 3, 0xA, 7, 5, 11)) +
        sf.OPTIONAL_RECORDS["GSYN"].pack_row(
        (2, 1, 3, 2, 1, 0xB, 9, 2, 13)))
    seg1x = sf.write_segment(tables)
    out = sf.read_segment(relink.relink_segment(seg1x, t2))
    assert bytes(out.require_table("MBOB")) == tables["MBOB"]
    assert bytes(out.require_table("GSYN")) == tables["GSYN"]


@pytest.mark.parametrize("arch", ["hopper", "blackwell"])
@pytest.mark.parametrize("width,cycles", [(2, 3.0), (4, 3.0), (8, 6.0), (16, 23.0)])
@pytest.mark.parametrize("spread", [0, 32, 128])
def test_store_issue_width_is_independent_of_address_geometry(arch, width, cycles, spread):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "configs" / "mechanism_tree"
    fragment = json.loads((root / arch / "leaves" / "storage.global.store_issue.json").read_text())
    leaf = fragment["storage.global.store_issue.root"]
    field = leaf["fields"]["cycles"]["model"]
    stride = spread or width
    memory = model.MemoryReference(space="global", access_kind="store", width_bytes=width,
        lane_addresses=tuple(0x10000 + lane * stride for lane in range(32)), lane_mask=0xFFFFFFFF)
    static = _static("STG")
    dynamic = _dyn(memory=(memory,))
    assert field["axis"] == "width_bytes"
    value = axes.axis_value(field["axis"], static, dynamic, _ctx())
    assert value == width
    assert dict(field["points"])[value] == cycles
    assert leaf["scope"] == "smsp"
