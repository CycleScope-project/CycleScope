"""control.reconvergence binds to the BSSY that opens the region.

The converter attaches region facts (control_path_count,
sequential_region_count, control_selector_present) to the opening BSSY
record; BSYNC carries none of them.  The generated trees must therefore
bind the program to BSSY, and mechanism_uses must emit a
control.reconvergence row for a fact-carrying BSSY and none for a bare
BSYNC.
"""
import json
from pathlib import Path

import pytest

from cyclescope.convert import mechanism_uses as muses, axes
from cyclescope.convert import model

TREES = Path(__file__).resolve().parents[2] / "configs" / "trees"


def _static(op):
    return model.StaticInstruction(
        pc=0x10, code_address=None, opcode=op,
        operation_class=op.split(".", 1)[0], operands=(),
        source_registers=(), destination_registers=(),
        register_roles={})


def _dyn(uniform=None):
    ops = model.SemanticOperands(uniform=uniform or {}) \
        if uniform is not None else None
    return model.DynamicInstruction(
        static_index=0, active_mask=0xFFFFFFFF,
        predicate_mask=0xFFFFFFFF, memory=(), operands=ops)


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


def _ctx():
    return axes.AxisContext(unit=None, launch=None, warp=None,
                            flow=axes.FlowContext(), rules=_Rules)


@pytest.fixture(params=["h100", "b300"])
def tree(request):
    return json.loads(
        (TREES / f"{request.param}.json").read_text())


def test_program_binds_bssy(tree):
    assert tree["programs"]["control.reconvergence"][
        "operation_classes"] == ["BSSY"]


def test_bssy_with_region_facts_produces_row(tree):
    rt = muses.TreeRuntime(tree)
    rows = muses.program_facts(
        rt, _static("BSSY"),
        _dyn(uniform={"control_path_count": 2}), _ctx(), "issue")
    rec = [r for r in rows
           if r.get("program") == "control.reconvergence"]
    assert rec, f"no control.reconvergence row in {rows}"
    assert rec[0]["facts"]["control_path_count"] == 2


def test_bsync_produces_no_reconvergence_row(tree):
    rt = muses.TreeRuntime(tree)
    rows = muses.program_facts(rt, _static("BSYNC"), _dyn(), _ctx(),
                               "issue")
    assert not [r for r in rows
                if r.get("program") == "control.reconvergence"]
    assert not [r for r in rows if "missing" in r]
