"""control.branch binds every BRA instance; branch_taken is a
converter axis = 1 when the warp's next dynamic instance's static pc
is not pc + 16 (16 B SASS), else 0 (last instance of a warp -> 0)."""
import json
from pathlib import Path

import pytest

from cyclescope.convert import mechanism_uses as muses, axes
from cyclescope.convert import model

TREES = Path(__file__).resolve().parents[2] / "configs" / "trees"


def _static(op, pc=0x10):
    return model.StaticInstruction(
        pc=pc, code_address=None, opcode=op,
        operation_class=op.split(".", 1)[0], operands=(),
        source_registers=(), destination_registers=(),
        register_roles={})


def _dyn():
    return model.DynamicInstruction(
        static_index=0, active_mask=0xFFFFFFFF,
        predicate_mask=0xFFFFFFFF, memory=(), operands=None)


class _Rules:
    NAME = "test"

    @staticmethod
    def operation_class(s, d=None):
        return s.opcode.split(".", 1)[0]

    @staticmethod
    def instruction_flags(s):
        return 0

    @staticmethod
    def semantics(s, d, ctx, cta=0, wslot=0):
        return None


def _ctx(next_pc=None):
    return axes.AxisContext(unit=None, launch=None, warp=None,
                            flow=axes.FlowContext(), rules=_Rules,
                            next_static_pc=next_pc)


@pytest.fixture(params=["h100", "b300"])
def tree(request):
    return json.loads(
        (TREES / f"{request.param}.json").read_text())


def test_program_binds_bra(tree):
    assert tree["programs"]["control.branch"][
        "operation_classes"] == ["BRA"]


def test_bra_produces_row_with_branch_taken(tree):
    rt = muses.TreeRuntime(tree)
    rows = muses.program_facts(
        rt, _static("BRA", pc=0x40), _dyn(), _ctx(next_pc=0x60),
        "control_transition")
    rec = [r for r in rows if r.get("program") == "control.branch"]
    assert rec, f"no control.branch row in {rows}"
    assert rec[0]["facts"]["branch_taken"] == 1


def test_fallthrough_bra_axis_is_zero(tree):
    assert axes.axis_value("branch_taken", _static("BRA", pc=0x40),
                           _dyn(), _ctx(next_pc=0x50)) == 0
    # last instance of the warp -> 0
    assert axes.axis_value("branch_taken", _static("BRA", pc=0x40),
                           _dyn(), _ctx(next_pc=None)) == 0


def test_non_bra_produces_no_row(tree):
    rt = muses.TreeRuntime(tree)
    rows = muses.program_facts(rt, _static("FADD"), _dyn(), _ctx(),
                               "control_transition")
    assert not [r for r in rows
                if r.get("program") == "control.branch"]
