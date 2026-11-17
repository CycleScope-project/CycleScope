"""Table-driven tests for convert.semantics_common (2b-2)."""
import pytest

from cyclescope.convert import model
from cyclescope.arch.hopper import rules
from cyclescope.convert.semantics_common import (
    ConversionError, SemanticRow, SemanticsContext, decode,
    FLAG_AFTER_READY, FLAG_ALSO_ARRIVES, DV_PROXY_FENCE, DV_PUBLISH)
from cyclescope.job.semantics import (SemanticsKind, GENERATION_UNRESOLVED,
                                  SEMA_FLAG_COMPLETE_TX)


def _static(opcode, **kw):
    return model.StaticInstruction(pc=kw.pop("pc", 0x10), code_address=None, opcode=opcode,
                                   operation_class=opcode.split(".")[0],
                                   **kw)


def _dyn(mask=0xFFFFFFFF, memory=(), uniform=None, ranks=None,
         after_ready=None, pred=0xFFFFFFFF, tensor_map=None):
    ops = None
    if (uniform is not None or ranks is not None or after_ready is not None
            or tensor_map is not None):
        ops = model.SemanticOperands(uniform=uniform or {},
                                     ranks=ranks or {},
                                     after_ready_mask=after_ready,
                                     tensor_map=tensor_map)
    return model.DynamicInstruction(static_index=0, active_mask=mask,
                                    predicate_mask=pred, memory=memory,
                                    operands=ops)


def _memref(addr, lanes=1, kind="other"):
    lanes_tuple = tuple(addr for _ in range(lanes))
    return model.MemoryReference(space="shared", access_kind=kind,
                                 width_bytes=8, lane_addresses=lanes_tuple,
                                 lane_mask=(1 << lanes) - 1)


@pytest.fixture
def ctx():
    unit = model.ClusterUnit(cluster_id=0, ctas=[], warps=[], statics=[],
                             tensor_maps={0: model.TensorMap(
                                 rank=2, element_bytes=2,
                                 box_dim=(16, 8), stride_bytes=(512,),
                                 global_dim=(64, 64)),
                                 1: model.TensorMap(
                                 rank=4, element_bytes=2,
                                 box_dim=(8, 8))})
    return SemanticsContext(unit=unit)


CASES = [
    # opcode, operands kwargs, expected kind, expected field subset
    ("SYNCS.EXCH.64", dict(memory=(_memref(0x340),),
                               uniform={"op2": 4}),
     SemanticsKind.mbarrier_init, {"expected_arrivals": 4}),
    ("SYNCS.ARRIVE", dict(memory=(_memref(0x340),), uniform={"op2": 2}),
     SemanticsKind.mbarrier_arrive, {"arrival_count": 2}),
    ("SYNCS.ARRIVE.TRANS64.A0T8", dict(memory=(_memref(0x340),),
                                      uniform={"op2": 512}),
     SemanticsKind.mbarrier_arrive, {"transaction_bytes": 512}),

    ("SYNCS.PHASECHK.TRANS64", dict(memory=(_memref(0x340),),
                                   uniform={"parity": 1}),
     SemanticsKind.mbarrier_wait, {"parity": 1}),

    ("BAR.ARV", dict(uniform={"op0": 3, "op1": 128}),
     SemanticsKind.cta_barrier_arrive,
     {"barrier": 3, "expected_threads": 128}),
    ("BAR.SYNC", dict(uniform={"op0": 0, "op1": 256}),
     SemanticsKind.cta_barrier_wait,
     {"barrier": 0, "expected_threads": 256}),
    ("UCGABAR_ARV", dict(), SemanticsKind.cluster_barrier_arrive, {}),
    ("UCGABAR_WAIT", dict(), SemanticsKind.cluster_barrier_wait, {}),
    ("UTMALDG.2D", dict(memory=(_memref(0x900),),
                       uniform={"mbarrier_address": 0x900, "op2": 4096},
                       tensor_map=0),
     SemanticsKind.tma_load, {"shared_bytes": 256}),
    ("UTMALDG.4D.IM2COL", dict(memory=(_memref(0x900),),
                              uniform={"mbarrier_address": 0x900,
                                       "op2": 4096},
                              tensor_map=1),
     SemanticsKind.tma_load, {"shared_bytes": 128}),
    ("UTMASTG.2D", dict(uniform={"op2": 1024}),
     SemanticsKind.bulk_group_issue, {"bytes": 1024}),
    ("UBLKCP.G.S", dict(uniform={"op2": 64}),
     SemanticsKind.bulk_group_issue, {"bytes": 64}),
    ("UTMACMDFLUSH", dict(), SemanticsKind.bulk_group_commit, {}),

    ("UBLKCP.S.G", dict(memory=(_memref(0x500),), uniform={"op2": 128}),
     SemanticsKind.bulk_copy, {"bytes": 128}),
    ("UBLKRED.S.G", dict(memory=(_memref(0x500),), uniform={"op2": 32}),
     SemanticsKind.bulk_reduce, {"bytes": 32}),
    ("LDGSTS.BYPASS", dict(), SemanticsKind.async_copy_issue, {}),
    ("LDGDEPBAR", dict(), SemanticsKind.async_copy_commit, {}),
    ("ARRIVES.LDGSTSBAR", dict(mask=1, memory=(_memref(0x300),)),
     SemanticsKind.async_copy_commit, {"arrival_count": 1}),


    ("STAS", dict(memory=(_memref(0x200),), uniform={"op2": 16}),
     SemanticsKind.shared_store_async, {"bytes": 16}),

    ("FENCE.VIEW.ASYNC.S", dict(),
     SemanticsKind.descriptor_visibility,
     {"operation": DV_PROXY_FENCE}),
    ("NANOSLEEP", dict(uniform={"op0": 256}),
     SemanticsKind.nanosleep, {"requested_ns": 256}),
    ("PREEXIT", dict(), SemanticsKind.dependents_launch, {}),
]


@pytest.mark.parametrize("opcode,kwargs,kind,expect", CASES)
def test_decode_kinds(ctx, opcode, kwargs, kind, expect):
    if opcode.startswith("ARRIVES"):
        decode(_static("LDGSTS.E.LTC128B.128"),
               _dyn(mask=1, uniform={"op0": 1}), ctx,
               rules.semantic_operation(
                   _static("LDGSTS.E.LTC128B.128")), 0, 0)
    row = _dec(_static(opcode), _dyn(**kwargs), ctx, 0, 0)
    assert row is not None and row.kind == kind
    for name, value in expect.items():
        assert row.fields.get(name) == value, (name, row.fields)


def test_predicated_off_arrive(ctx):
    # real payload form: active_mask=ffffffff, predicate_mask=00000000
    row = _dec(_static("SYNCS.ARRIVE.TRANS64.RED.A1T0"),
                 _dyn(mask=0xFFFFFFFF, pred=0), ctx, 0, 0)
    assert row is None


def test_predicated_off_preexit(ctx):
    # a predicated-off PREEXIT carries no executed launch_dependents
    assert _dec(_static("PREEXIT"), _dyn(mask=0xFFFFFFFF, pred=0),
                ctx, 0, 0) is None
    assert _dec(_static("PREEXIT"), _dyn(), ctx, 0, 0).kind == \
        SemanticsKind.dependents_launch


def test_predicated_off(ctx):
    assert _dec(_static("BAR.SYNC"), _dyn(mask=0, pred=0),
                  ctx, 0, 0) is None


def test_missing_parity(ctx):
    with pytest.raises(ConversionError, match="missing_fact:parity"):
        _dec(_static("SYNCS.PHASECHK"), _dyn(memory=(_memref(0x40),)),
               ctx, 0, 0)


def test_missing_mbarrier_address(ctx):
    with pytest.raises(ConversionError, match="missing_fact:mbarrier"):
        _dec(_static("SYNCS.ARRIVE"), _dyn(uniform={"op2": 1}), ctx, 0, 0)


def test_after_ready_flag(ctx):
    row = _dec(_static("SYNCS.PHASECHK"),
                 _dyn(memory=(_memref(0x40),),
                      uniform={"parity": 0}, after_ready=0xFF), ctx, 0, 0)
    assert row.flags & FLAG_AFTER_READY
    assert row.fields["after_ready_mask"] == 0xFF


def test_cta_barrier_also_arrives(ctx):
    row = _dec(_static("BAR.SYNC"), _dyn(uniform={"op0": 1}), ctx, 0, 0)
    assert row.flags & FLAG_ALSO_ARRIVES


def test_divergent_arrive_mbol(ctx):
    mem = model.MemoryReference(space="shared", access_kind="other",
                                width_bytes=8,
                                lane_addresses=tuple(0x40 + i * 8
                                                     for i in range(4)),
                                lane_mask=0xF)
    row = _dec(_static("SYNCS.ARRIVE"), _dyn(mask=0xF, memory=(mem,)),
                 ctx, 0, 0)
    assert row.object_list and len(row.object_list) == 4


def test_bulk_group_numbering(ctx):
    issue = _dec(_static("UTMASTG"), _dyn(uniform={"op2": 1}), ctx, 0, 0)
    commit = _dec(_static("UTMACMDFLUSH"), _dyn(), ctx, 0, 0)
    issue2 = _dec(_static("UTMASTG"), _dyn(uniform={"op2": 1}), ctx, 0, 0)
    assert (issue.fields["group"], commit.fields["group"],
            issue2.fields["group"]) == (0, 0, 1)


def test_debar_wait_pending(ctx):
    st = _static("DEPBAR.LE", operands=("SB0", "0x2"))
    row = _dec(st, _dyn(uniform={"op1": 2}), ctx, 0, 0)
    assert row.kind in (SemanticsKind.async_copy_wait,
                        SemanticsKind.bulk_group_wait)
    assert row.fields["pending_allowed"] == 2


def test_generation_unresolved(ctx):
    row = _dec(_static("SYNCS.ARRIVE"), _dyn(memory=(_memref(0x40),)),
                 ctx, 0, 0)
    assert row.fields["generation"] == GENERATION_UNRESOLVED


def _dec(static, dyn, ctx, cta=0, wslot=0):
    operation = rules.semantic_operation(static)
    if operation is None:
        return None
    return decode(static, dyn, ctx, operation, cta, wslot)


def test_bulk_group_wait_direct(ctx):
    from cyclescope.convert.semantics_common import SemanticOperation
    row = decode(_static("X"), _dyn(uniform={"op0": 2}), ctx,
                 SemanticOperation.bulk_group_wait, 0, 0)
    assert row.kind == SemanticsKind.bulk_group_wait
    assert row.fields["pending_allowed"] == 2


def test_descriptor_publish_direct(ctx):
    from cyclescope.convert.semantics_common import SemanticOperation
    # 8 chunk stores then MEMBAR.ALL.GPU publishes (bw protocol)
    for chunk in range(8):
        mem = model.MemoryReference(
            space="global", access_kind="store", width_bytes=16,
            lane_addresses=(0x1000 + chunk * 16,), lane_mask=1)
        decode(_static("STG"), _dyn(mask=1, pred=1, memory=(mem,)),
               ctx, SemanticOperation.descriptor_visibility_store, 0, 0)
    row = decode(_static("MEMBAR.ALL.GPU"), _dyn(), ctx,
                 SemanticOperation.descriptor_visibility_publish, 0, 0)
    assert row.fields["operation"] == DV_PUBLISH
    # MEMBAR without the protocol -> None
    assert decode(_static("MEMBAR.ALL.CTA"), _dyn(), ctx,
                  SemanticOperation.descriptor_visibility_publish,
                  0, 0) is None


def test_unknown_opcode_none(ctx):
    assert rules.semantic_operation(_static("FADD")) is None


def test_unsupported_modifier(ctx):
    with pytest.raises(ConversionError, match="unsupported:semantic_modifier"):
        rules.semantic_operation(_static("BAR.UNKNOWN"))


def test_no_object_sentinel_pack_unpack():
    from cyclescope.job.semantics import (pack_fields, unpack_fields,
                                      SEMA_NO_OBJECT)
    slots = pack_fields(SemanticsKind.async_copy_commit, group=0,
                        pending_allowed=0)
    assert slots[2] == SEMA_NO_OBJECT           # object slot sentinel
    back = unpack_fields(SemanticsKind.async_copy_commit, slots)
    assert back["object"] is None
    slots2 = pack_fields(SemanticsKind.async_copy_commit, group=0,
                         pending_allowed=0, object=7)
    assert slots2[2] == 7


def test_lgdepbar_row_has_no_object():
    from cyclescope.convert.semantics_common import SemanticOperation
    from cyclescope.job.semantics import pack_fields, SEMA_NO_OBJECT
    row = decode(_static("LDGDEPBAR"), _dyn(uniform={"op0": 1}), None
                 or _mk_ctx(), SemanticOperation.async_copy_commit, 0, 0)
    assert row.kind == SemanticsKind.async_copy_commit
    assert "object" not in row.fields or row.fields["object"] is None
    slots = pack_fields(row.kind, **row.fields)
    assert slots[2] == SEMA_NO_OBJECT


def _mk_ctx():
    unit = model.ClusterUnit(cluster_id=0)
    return SemanticsContext(unit)


def test_arrives_without_ldgsts_decodes(ctx):
    # PTX cp.async.mbarrier.arrive with no pending copies triggers at
    # once: the row decodes with object + transaction_count = lanes
    from cyclescope.convert.semantics_common import SemanticOperation
    mem = _memref(0x200)
    row = decode(_static("ARRIVES.LDGSTSBAR.64.TRANSCNT"),
                 _dyn(mask=1, memory=(mem,), uniform={"op0": 1}), ctx,
                 SemanticOperation.async_copy_commit, 0, 0)
    assert row.kind == SemanticsKind.async_copy_commit
    assert row.fields["arrival_count"] == 0
    assert row.fields["transaction_count"] == 1
    # the .noinc spelling (no TRANSCNT) arrives once per lane
    row = decode(_static("ARRIVES.LDGSTSBAR.64"),
                 _dyn(mask=0b111, memory=(mem,), uniform={"op0": 1}), ctx,
                 SemanticOperation.async_copy_commit, 0, 0)
    assert row.fields["arrival_count"] == 3
    assert row.fields["transaction_count"] == 0


def test_stas_per_lane_object_list(ctx):
    # lanes 1..7 store 4 B each into a different rank's barrier:
    # one MBOL object per lane in lane order, bytes = total / lanes
    from cyclescope.convert.semantics_common import SemanticOperation
    words = tuple((r + 1) << 24 | 0x4b8 for r in range(7))
    dyn = _dyn(mask=0xFE, memory=(_memref(0x200, lanes=7),),
               uniform={"bytes": 28, "mbarrier_address": words[0]},
               )
    dyn.operands.lane["mbarrier_words"] = words
    row = decode(_static("STAS"), dyn, ctx,
                 SemanticOperation.shared_store_async, 0, 0)
    assert row.kind == SemanticsKind.shared_store_async
    assert row.object_list is not None and len(row.object_list) == 7
    assert row.fields["bytes"] == 4
    assert row.fields["object"] == row.object_list[0]
    # single destination: unchanged fields, no list
    row = decode(_static("STAS"),
                 _dyn(memory=(_memref(0x200),),
                      uniform={"bytes": 16}), ctx,
                 SemanticOperation.shared_store_async, 0, 0)
    assert row.object_list is None
    assert row.fields["bytes"] == 16
    assert "object_list" not in row.fields.values()


def test_depbar_keep_zero_fallback(ctx):
    # keep=0 arriving through the operand-fact fallback must stay 0
    # (not slip to the last op token).
    st = _static("DEPBAR.LE", operands=("SB0",))
    row = decode(st, _dyn(uniform={"N": 0, "op3": 7}), ctx,
                 rules.semantic_operation(st), 0, 0)
    assert row is not None
    assert row.fields["pending_allowed"] == 0


def test_cta_barrier_register_form_packs_count_and_id(ctx):
    # BAR.SYNC R0, R0 with R0 = (384 << 16) | 12 -> barrier 12, 384 threads
    st = _static("BAR.SYNC.DEFER_BLOCKING")
    st.operands = ("R0", "R0")
    dyn = _dyn(uniform={})
    dyn.operands.lane["R0"] = (0x180000C,)
    row = _dec(st, dyn, ctx, 0, 0)
    assert row.fields["barrier"] == 12 and row.fields["expected_threads"] == 384
    # immediate id + register count: raw count
    st = _static("BAR.SYNC.DEFER_BLOCKING")
    st.operands = ("0xa", "R112")
    dyn = _dyn(uniform={"op0": 10})
    dyn.operands.lane["R112"] = (64,)
    row = _dec(st, dyn, ctx, 0, 0)
    assert row.fields["barrier"] == 10 and row.fields["expected_threads"] == 64


def test_bar_sync_no_count_expected_zero(ctx):
    # count-less BAR.SYNC -> expected_threads 0 (engine deducts exited
    # warps); explicit count unchanged
    row = _dec(_static("BAR.SYNC"), _dyn(uniform={"op0": 0}), ctx, 0, 0)
    assert row.fields["expected_threads"] == 0
    row = _dec(_static("BAR.SYNC"), _dyn(uniform={"op0": 0, "op1": 256}),
               ctx, 0, 0)
    assert row.fields["expected_threads"] == 256


def test_fence_view_async_g_no_row(ctx):
    # FENCE.VIEW.ASYNC.G carries no engine sync state -> no row
    assert _dec(_static("FENCE.VIEW.ASYNC.G"), _dyn(), ctx, 0, 0) is None


def test_synx_cctl_iv_mbarrier_inval(ctx):
    # Blackwell SYNCS.CCTL.IV = mbarrier.inval
    from cyclescope.convert.semantics_common import SemanticOperation
    row = decode(_static("SYNCS.CCTL.IV"),
                 _dyn(mask=1, memory=(_memref(0x300),)), ctx,
                 SemanticOperation.mbarrier_inval, 0, 0)
    assert row.kind == SemanticsKind.mbarrier_inval


def test_ublkred_destination_count_axis():
    # UBLKRED resolves destination_count = 1 (single global dest)
    from cyclescope.convert.axes import _f_destination_count
    s = _static("UBLKRED.G.S.ADD.F32.RN")
    assert _f_destination_count(s, None, None) == 1


@pytest.mark.parametrize("mod,half", [("16dp32bit_t0_t15", None),
                                      ("16dp32bit_t16_t31", None)])
@pytest.mark.parametrize("op", ["LDTM", "STTM"])
def test_tmem_16x32bx2_halves(ctx, op, mod, half):
    # SASS halves of tcgen05.ld/st .16x32bx2 -> shape code 5 + xN repeat
    from cyclescope.arch.blackwell.semantics import _tmem_shape_repeat
    code, rep = _tmem_shape_repeat(f"{op}.{mod}.x32", f"{op} @0x0")
    assert (code, rep) == (5, 32)
    code, rep = _tmem_shape_repeat(f"{op}.{mod}", f"{op} @0x0")
    assert (code, rep) == (5, 1)


def test_tmem_shape_axis_16dp32bit_suffix():
    from cyclescope.convert.axes import _f_tmem_shape
    s = _static("LDTM.16dp32bit_t0_t15.x32")
    assert _f_tmem_shape(s, None, None) == "16dp32bit"


def test_complete_tx_arrive_flag(ctx):
    row = _dec(_static("SYNCS.ARRIVE.TRANS64.RED.A0TX"),
               _dyn(memory=(_memref(0x340),),
                    uniform={"transaction_complete": 1,
                             "transaction_bytes": 16,
                             "arrival_count": 0}), ctx, 0, 0)
    assert row is not None and row.kind == SemanticsKind.mbarrier_arrive
    assert row.flags & SEMA_FLAG_COMPLETE_TX
    assert row.fields["arrival_count"] == 0
    assert row.fields["transaction_bytes"] == 16
