from cyclescope.arch.hopper import rules
from cyclescope.convert import model


def _static(opcode: str) -> model.StaticInstruction:
    return model.StaticInstruction(
        pc=0, code_address=None, opcode=opcode,
        operation_class=rules.operation_class(opcode))


def test_operation_class() -> None:
    assert rules.operation_class("FFMA.F32") == "FFMA"
    assert rules.operation_class("DEPBAR.LE") == "DEPBAR.FIFO"
    assert rules.operation_class("USETSHMSZ.FLUSH") == "USETSHMSZ.FLUSH"
    assert rules.operation_class("SHFL.IDX") == "SHFL.IDX"
    assert rules.operation_class("HMMA.16816.SP") == "HMMA.SPARSE"
    assert rules.operation_class("IMAD.HI") == "IMAD.HI"


def test_whitelist() -> None:
    assert rules.is_compute_whitelisted("FFMA")
    assert rules.is_compute_whitelisted("WGMMA")
    assert rules.is_compute_whitelisted("UTMALDG")
    assert not rules.is_compute_whitelisted("FAKEOP")


def test_flags() -> None:
    assert rules.instruction_flags(_static("LDG.E")) & rules.FLAG_MEMORY
    assert rules.instruction_flags(_static("ATOMG.ADD")) & rules.FLAG_ATOMIC
    assert rules.instruction_flags(_static("BRA")) & rules.FLAG_BRANCH
    assert rules.instruction_flags(_static("WGMMA.MMA")) & rules.FLAG_TENSOR
    assert not rules.instruction_flags(_static("FFMA.F32")) \
        & rules.FLAG_MEMORY


def test_memory_space() -> None:
    assert rules.memory_space(_static("LDG.E")) == "global"
    assert rules.memory_space(_static("LDS.128")) == "shared"
    assert rules.memory_space(_static("LDC.B")) == "constant"
    assert rules.memory_space(_static("LDL.B")) == "local"
    assert rules.memory_space(_static("FFMA.F32")) is None


def test_semantic_operation() -> None:
    from cyclescope.convert.semantics_common import SemanticOperation
    assert rules.semantic_operation(_static("SYNCS.PHASECHK")) == \
        SemanticOperation.mbarrier_wait
    assert rules.semantic_operation(_static("SYNCS.CCTL.IV")) == \
        SemanticOperation.mbarrier_inval
    # barrier-init fence carries no engine sync state -> no row
    assert rules.semantic_operation(_static("SYNCS.CCTL.IVALL")) is None


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
