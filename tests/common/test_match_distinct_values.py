import dataclasses

from cyclescope.convert import axes
from cyclescope.convert.model import (DynamicInstruction, SemanticOperands,
                                  StaticInstruction)


def _static(opcode, srcs, cls="MATCH"):
    return StaticInstruction(pc=0, code_address=0, opcode=opcode,
                             operation_class=cls,
                             source_registers=tuple(srcs),
                             destination_registers=("R2",))


def _dyn(lane):
    return DynamicInstruction(static_index=0, active_mask=0xffffffff,
                              operands=SemanticOperands(lane=lane))


def test_match_source_distinct():
    s = _static("MATCH.ANY.B32", ("R3",))
    d = _dyn({"R2": (1, 1, 1, 1), "R3": (1, 2, 3, 4, 5, 6, 7, 8)})
    assert axes._f_match_distinct_values(s, d, None) == 8


def test_match_u64_pair():
    s = _static("MATCH.ANY.U64", ("R4",))
    d = _dyn({"R4": (1, 1, 2, 2), "R5": (9, 9, 8, 8), "R2": (0, 0)})
    assert axes._f_match_distinct_values(s, d, None) == 2


def test_non_match_not_applicable():
    s = dataclasses.replace(_static("FADD", ("R3",)), operation_class="FADD")
    d = _dyn({"R3": (1, 2, 3)})
    assert axes._f_match_distinct_values(s, d, None) is axes.NOT_APPLICABLE


def test_missing_source_is_none():
    s = _static("MATCH.ANY.B32", ("R3",))
    d = _dyn({"R2": (1, 2)})
    assert axes._f_match_distinct_values(s, d, None) is None
