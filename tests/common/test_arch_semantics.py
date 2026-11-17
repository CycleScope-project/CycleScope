"""2b-3 arch semantics: warpgroup/wgmma (hopper), umma/tmem/clc (bw)."""
import pytest

from cyclescope.convert import model
from cyclescope.convert.semantics_common import (
    ConversionError, SemanticsContext)
from cyclescope.arch.hopper import rules as hrules
from cyclescope.arch.hopper import semantics as hsem
from cyclescope.arch.blackwell import rules as brules
from cyclescope.arch.blackwell import semantics as bsem
from cyclescope.job.semantics import (SemanticsKind, SEMA_FLAG_GROUP_COMMIT,
                                  SEMA_FLAG_GROUP_OPENER, SEMA_FLAG_STAT)


def _static(opcode, operands=(), dsts=()):
    return model.StaticInstruction(
        pc=0x10, code_address=None, opcode=opcode,
        operation_class=opcode.split(".")[0], operands=operands,
        destination_registers=dsts)


def _dyn(uniform=None, memory=()):
    ops = model.SemanticOperands(uniform=uniform or {}) if uniform else None
    return model.DynamicInstruction(static_index=0, active_mask=1,
                                    predicate_mask=1, memory=memory,
                                    operands=ops)


def _mem(addr):
    return model.MemoryReference(space="shared", access_kind="other",
                                 width_bytes=8, lane_addresses=(addr,),
                                 lane_mask=1)


@pytest.fixture
def ctx():
    unit = model.ClusterUnit(cluster_id=0)
    unit.ctas = [model.CtaFacts(cta_linear=0, cta=(0, 0, 0))] * 8
    return SemanticsContext(unit)


def _h(st, dyn, ctx, w=0):
    return hsem.decode(st, dyn, ctx, hrules.semantic_operation(st), 0, w)


def _b(st, dyn, ctx, w=0):
    return bsem.decode(st, dyn, ctx, brules.semantic_operation(st), 0, w)


# ---- hopper ----

def test_warpgroup_arrive_groups(ctx):
    st = _static("WARPGROUP.ARRIVE")
    assert _h(st, _dyn(), ctx).fields["group"] == 0
    assert _h(st, _dyn(), ctx).fields["group"] == 1


def test_wgmma_folded_opener_and_commit(ctx):
    st = _static("HGMMA.64x128x16.F32.BF16",
                 operands=("R24", "gdesc[UR16].tnspA", "R24", "gsb0"),
                 dsts=("R24", "R25"))
    row = _h(st, _dyn(), ctx)
    assert row.kind == SemanticsKind.wgmma
    assert row.flags & SEMA_FLAG_GROUP_OPENER
    assert row.flags & SEMA_FLAG_GROUP_COMMIT
    assert row.fields["shape_packed"] == (64 << 32) | (128 << 16) | 16
    assert row.fields["input_dtype"] == 2          # bf16
    assert row.fields["accumulator_dtype"] == 4    # f32
    assert row.fields["source_mode"] == 1          # ss
    assert row.fields["accumulator_registers_count"] == 64


def test_wgmma_rs_mode(ctx):
    st = _static("HGMMA.64x64x16.F32.F16",
                 operands=("R24", "R20", "gdesc[UR16]", ""),
                 dsts=("R24",))
    row = _h(st, _dyn(), ctx)
    assert row.fields["source_mode"] == 2


def test_warpgroup_depbar_wait(ctx):
    st = _static("WARPGROUP.DEPBAR.LE", operands=("gsb0", "0x2"))
    row = _h(st, _dyn(), ctx)
    assert row.kind == SemanticsKind.warpgroup_wait
    assert row.fields["pending_allowed"] == 2


def test_warpgroup_depbar_fail_closed(ctx):
    with pytest.raises(ConversionError,
                       match="missing_fact:warpgroup_wait_keep"):
        _h(_static("WARPGROUP.DEPBAR.LE", operands=("gsb0",)), _dyn(), ctx)
    with pytest.raises(ConversionError,
                       match="unsupported:warpgroup_scoreboard"):
        _h(_static("WARPGROUP.DEPBAR.LE", operands=("gsb1", "0x0")),
           _dyn(), ctx)
    with pytest.raises(ConversionError,
                       match="unsupported:warpgroup_wait_keep"):
        _h(_static("WARPGROUP.DEPBAR.LE", operands=("gsb0", "0x8")),
           _dyn(), ctx)


def test_hopper_utmacctl_iv_acquire(ctx):
    # UTMACCTL.IV [UR8] with UR8/UR9 captured: operand_facts composes
    # the 64-bit descriptor_address, routed to the acquire row
    st = _static("UTMACCTL.IV", operands=("[UR8]",))
    row = hrules.semantics(
        st, _dyn(uniform={"descriptor_address": 0x1122334455667700}),
        ctx)
    assert row.kind == SemanticsKind.descriptor_visibility
    assert row.fields["operation"] == 4       # DV_ACQUIRE
    assert row.fields["descriptor_address_0"] == 0x1122334455667700


def test_hopper_utmacctl_membar_variants_none(ctx):
    # only COMMIT is a bulk-group commit; other UTMACCTL / MEMBAR
    # variants carry no semantics (same rule as blackwell)
    assert hrules.semantic_operation(_static("UTMACCTL.PF")) is None
    assert hrules.semantic_operation(_static("UTMACCTL")) is None
    assert hrules.semantic_operation(_static("MEMBAR.ALL.CTA")) is None
    assert hrules.semantic_operation(
        _static("UTMACCTL.COMMIT")).name == "bulk_group_commit"


def test_hopper_descriptor_publish_needs_chunks(ctx):
    # MEMBAR.ALL.GPU routes to publish but emits a row only after 8
    # complete 16B chunks of a 128B descriptor were stored
    assert hrules.semantics(
        _static("MEMBAR.ALL.GPU"), _dyn(), ctx) is None
    for chunk in range(8):
        mem = model.MemoryReference(
            space="global", access_kind="store", width_bytes=16,
            lane_addresses=(0x2000 + chunk * 16,), lane_mask=1)
        assert hrules.semantics(
            _static("STG.E.128"), _dyn(memory=(mem,)), ctx) is None
    row = hrules.semantics(_static("MEMBAR.ALL.GPU"), _dyn(), ctx)
    assert row.fields["operation"] == 1       # DV_PUBLISH
    assert row.fields["descriptor_count"] == 1
    assert row.fields["descriptor_address_0"] == 0x2000


# ---- blackwell ----

def test_umma_issue(ctx):
    idesc = (16 << 24) | (8 << 17) | (1 << 7) | (1 << 10)
    st = _static("UTCHMMA.2CTA",
                 operands=("gdesc[UR4]", "gdesc[UR6]", "tmem[UR11]",
                           "idesc[UR33]"))
    row = _b(st, _dyn(uniform={"idesc_value": idesc}), ctx)
    assert row.kind == SemanticsKind.umma_issue
    assert row.fields["kind"] == 1
    assert row.fields["cta_group"] == 2
    assert row.fields["a_source"] == 1
    assert row.fields["tensor_m"] == 256
    assert row.fields["tensor_n"] == 64
    assert row.fields["descriptor"] == idesc


def test_umma_issue_missing_idesc(ctx):
    with pytest.raises(ConversionError,
                       match="missing_fact:umma_descriptor"):
        _b(_static("UTCHMMA.2CTA"), _dyn(), ctx)


def test_utcbar(ctx):
    st = _static("UTCBAR.2CTA.MULTICAST")
    dyn = _dyn(uniform={"multicast_mask": 15}, memory=(_mem(0x440),))
    row = _b(st, dyn, ctx)
    assert row.kind == SemanticsKind.umma_commit
    assert row.fields["commit_mask"] == 15
    assert row.fields["scope"] == 2
    assert row.fields["cta_group"] == 2


def test_utcbar_missing_mask(ctx):
    st = _static("UTCBAR.2CTA.MULTICAST")
    with pytest.raises(ConversionError,
                       match="missing_fact:umma_commit_mask"):
        _b(st, _dyn(memory=(_mem(0x440),)), ctx)


def test_tmem_alloc_columns(ctx):
    st = _static("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN",
                 operands=("UR4", "UR4"))
    row = _b(st, _dyn(uniform={"tmem_columns": 512}), ctx)
    assert row.kind == SemanticsKind.tmem_alloc
    assert row.fields["columns"] == 512
    with pytest.raises(ConversionError, match="missing_fact:tmem_columns"):
        _b(st, _dyn(), ctx)


def test_tmem_load_shape(ctx):
    st = _static("LDTM.STAT.32X32B.x16")
    row = _b(st, _dyn(), ctx)
    assert row.kind == SemanticsKind.tmem_load
    assert row.flags & SEMA_FLAG_STAT
    assert row.fields["shape"] == 4
    assert row.fields["repeat"] == 16
    with pytest.raises(ConversionError, match="unsupported:tmem_shape"):
        _b(_static("LDTM.BOGUS.zz"), _dyn(), ctx)


def test_fence_async_t(ctx):
    st = _static("FENCE.VIEW.ASYNC.T")
    row = _b(st, _dyn(), ctx)
    assert row.kind == SemanticsKind.tmem_wait_store
    st2 = _static("FENCE.VIEW.ASYNC.S")
    op = brules.semantic_operation(st2)
    assert op is not None and op.name == "descriptor_visibility_proxy_fence"


def test_clc_query(ctx):
    st = _static("UGETNEXTWORKID.BROADCAST")
    dyn = _dyn(memory=(_mem(0x100), _mem(0x4d0)))
    row = _b(st, dyn, ctx)
    assert row.kind == SemanticsKind.clc_query
    assert row.fields["destination_mask"] == 0xFF


def test_uvirtcount_none(ctx):
    assert brules.semantic_operation(_static("UVIRTCOUNT")) is None


# ---- review revisions (H1/H2/B1/B2/SEMA_UNKNOWN) ----

def test_wgmma_acc_count_by_dtype(ctx):
    # 32-bit accumulator f32: N/2 regs
    st = _static("HGMMA.64x128x16.F32.BF16",
                 operands=("R24", "gdesc[UR16]", "R24"),
                 dsts=("R24", "R25"))
    assert _h(st, _dyn(), ctx).fields["accumulator_registers_count"] == 64
    # IGMMA acc dtype s32 (32-bit): N/2
    st = _static("IGMMA.64x32x32.S32.S8",
                 operands=("R24", "gdesc[UR16]", "R24"),
                 dsts=("R24", "R25"))
    row = _h(st, _dyn(), ctx)
    assert row.fields["accumulator_dtype"] == 5  # s32
    assert row.fields["accumulator_registers_count"] == 16


def test_wgmma_acc_begin_missing(ctx):
    # only UR dst + non-R operand head -> no begin -> missing
    st2 = _static("HGMMA.64x128x16.F32.BF16",
                  operands=("", "gdesc[UR16]", ""), dsts=("UR4",))
    with pytest.raises(ConversionError,
                       match="missing_fact:wgmma_accumulator_register"):
        _h(st2, _dyn(), ctx)


def test_warpgroup_wait_group_is_commit_count(ctx):
    w = _static("WARPGROUP.ARRIVE")
    g = _static("HGMMA.64x64x16.F32.F16",
                operands=("R24", "gdesc[UR16]", "R24", "gsb0"),
                dsts=("R24", "R25"))
    _h(w, _dyn(), ctx); _h(w, _dyn(), ctx)      # groups 0,1
    _h(g, _dyn(), ctx)                          # commit #1
    row = _h(_static("WARPGROUP.DEPBAR.LE",
                     operands=("gsb0", "0x1")), _dyn(), ctx)
    assert row.fields["group"] == 1             # commit count, not 2


def test_tmem_shape_codes(ctx):
    for op, want in (("LDTM.16x64b", 1), ("STTM.16dp128bit", 2),
                     ("STTM.16dp256bit", 3), ("STTM.32dp32bit", 4)):
        row = _b(_static(op), _dyn(), ctx)
        assert row.fields["shape"] == want, op
    row = _b(_static("LDTM.x128"), _dyn(), ctx)
    assert row.fields["shape"] == 4 and row.fields["repeat"] == 128
    row = _b(_static("LDTM.STAT.x32.MAX.F32"), _dyn(), ctx)
    assert row.fields["shape"] == 4 and row.fields["repeat"] == 32 \
        and row.flags & SEMA_FLAG_STAT
    row = _b(_static("UTCCP.T.S.4x32dp128bit"), _dyn(), ctx)
    assert row.kind == SemanticsKind.tmem_copy
    assert row.fields["shape"] == (4 << 16) | 128
    with pytest.raises(ConversionError, match="unsupported:tmem_shape"):
        _b(_static("STTM.bogusmod"), _dyn(), ctx)


def test_umma_a_source_fail(ctx):
    idesc = (16 << 24) | (8 << 17)
    st = _static("UTCHMMA", operands=("UR4", "UR6"))
    with pytest.raises(ConversionError, match="unsupported:umma_a_source"):
        _b(st, _dyn(uniform={"idesc_value": idesc}), ctx)


def test_tmem_dealloc_unknown_columns(ctx):
    st = _static("UTCATOMSWS.AND", operands=("URZ", "UR5"))
    row = _b(st, _dyn(), ctx)
    assert row.kind == SemanticsKind.tmem_dealloc
    assert row.fields["columns"] == 0xFFFFFFFF
    with pytest.raises(ConversionError, match="missing_fact:tmem_columns"):
        _b(_static("UTCATOMSWS.FIND_AND_SET"), _dyn(), ctx)


def test_mufu_per_op_class_routing():
    from cyclescope.arch.hopper.rules import operation_class as hop_class
    from cyclescope.arch.blackwell.rules import operation_class as blw_class
    import pytest
    _S = __import__("tests.common.test_arch_semantics", fromlist=["_S"])._S \
        if False else None

    class St:
        pass

    for op in ("EX2", "RSQ", "LG2", "TANH"):
        s = St()
        s.opcode = f"MUFU.{op}"
        s.operands = ["R2", "R4"]
        assert hop_class(s) == f"MUFU.{op}", ("hopper", op)
        assert blw_class(s) == f"MUFU.{op}", ("blackwell", op)
    s = St()
    s.opcode = "MUFU.RCP"
    s.operands = ["R2", "R4"]
    assert hop_class(s) == "MUFU"
    assert blw_class(s) == "MUFU"
