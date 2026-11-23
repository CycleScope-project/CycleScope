from cyclescope.arch.blackwell import rules
from cyclescope.convert import model


def _static(opcode: str) -> model.StaticInstruction:
    return model.StaticInstruction(
        pc=0, code_address=None, opcode=opcode,
        operation_class=rules.operation_class(opcode))


def test_operation_class() -> None:
    assert rules.operation_class("FFMA.F32") == "FFMA"
    assert rules.operation_class("HADD2.F32") == "HADD2.F32"
    assert rules.operation_class("IMAD.HI") == "IMAD.HI"
    assert rules.operation_class("MUFU.EX2") == "MUFU.EX2"
    assert rules.operation_class("MEMBAR.ALL.SYS") == "MEMBAR.ALL.SYS"
    assert rules.operation_class("UTCHMMA.MMA") == "UTCHMMA"


def test_whitelist() -> None:
    assert rules.is_compute_whitelisted("UTCHMMA")
    assert rules.is_compute_whitelisted("LDTM")
    assert rules.is_compute_whitelisted("UGETNEXTWORKID")
    assert not rules.is_compute_whitelisted("FAKEOP")


def test_flags() -> None:
    assert rules.instruction_flags(_static("LDTM.32")) & rules.FLAG_TMEM
    assert rules.instruction_flags(
        _static("UGETNEXTWORKID")) & rules.FLAG_CLC
    assert rules.instruction_flags(
        _static("UTCHMMA.MMA")) & rules.FLAG_TENSOR


def test_cache_hint() -> None:
    hint = rules.cache_hint(_static("LDG.E.L2::EF.L1::CV"))
    assert hint & rules.CACHE_HINT_L1_MASK
    assert hint & rules.CACHE_HINT_L2_MASK
    assert rules.cache_hint(_static("FFMA.F32")) == 0


def test_memory_space() -> None:
    assert rules.memory_space(_static("LDG.E")) == "global"
    assert rules.memory_space(_static("STAS")) == "shared"
    assert rules.memory_space(_static("LDCU")) == "constant"


def _static_ops(opcode, operands):
    return model.StaticInstruction(
        pc=0, code_address=None, opcode=opcode, operation_class="",
        operands=tuple(operands))


def test_operation_class_arrive_state() -> None:
    assert rules.operation_class(_static_ops(
        "SYNCS.ARRIVE.TRANS64.A1T0",
        ["R12", "[R0+UR4]", "RZ"])) == "SYNCS.ARRIVE.STATE"
    assert rules.operation_class(_static_ops(
        "SYNCS.ARRIVE.TRANS64.ART0",
        ["R8", "[R8+URZ]", "R13"])) == "SYNCS.ARRIVE.STATE.COUNT"
    assert rules.operation_class(_static_ops(
        "SYNCS.ARRIVE.TRANS64.A1T0",
        ["RZ", "[R5+URZ]", "RZ"])) == "SYNCS"
    assert rules.operation_class(_static_ops(
        "SYNCS.ARRIVE.TRANS64.RED.A1T0",
        ["RZ", "[R5+URZ]", "R2"])) == "SYNCS"


def test_defer_blocking_flag() -> None:
    """placement v2 rule B: only a BAR carrying the DEFER_BLOCKING
    modifier is marked; the operation class stays BAR."""
    for op in ("BAR.SYNC.DEFER_BLOCKING", "BAR.RED.POPC.DEFER_BLOCKING"):
        assert rules.instruction_flags(_static(op)) & rules.FLAG_DEFER_BLOCKING
        assert rules.operation_class(op) == "BAR"
    for op in ("BAR.SYNC", "BAR.ARV", "BAR.SYNC.DEFER", "LDG.DEFER_BLOCKING"):
        assert not rules.instruction_flags(_static(op)) \
            & rules.FLAG_DEFER_BLOCKING
