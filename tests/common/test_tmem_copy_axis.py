import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyclescope.convert import mechanism_uses as muses, axes
from cyclescope.convert.axes import _f_tmem_shape
from cyclescope.convert import model

TREES = Path(__file__).resolve().parents[2] / "configs" / "trees"

_SHAPES = {
    "": "copy_default",
    ".4DP256BIT": "4dp256bit",
    ".128DP128BIT": "128dp128bit",
    ".2X64DP128BIT_LW02_LW13": "2x64dp128bit_lw02_lw13",
    ".4X32DP128BIT": "4x32dp128bit",
}


@pytest.mark.parametrize("group", ["", ".2CTA"])
@pytest.mark.parametrize("suffix,shape", list(_SHAPES.items()))
def test_utccp_copy_shapes(group, suffix, shape):
    s = SimpleNamespace(opcode=f"UTCCP.T.S{group}{suffix}", pc=0)
    assert _f_tmem_shape(s, None, None) == shape


@pytest.mark.parametrize("opcode", [
    "UTCCP",
    "UTCCP.T.S.BAD",
    "UTCCP.T.S.4x32dp128bit.BAD",
    "UTCCP.T.S.2CTA.2CTA",
])
def test_utccp_unknown_returns_none(opcode):
    s = SimpleNamespace(opcode=opcode, pc=0)
    assert _f_tmem_shape(s, None, None) is None


@pytest.mark.parametrize("opcode,want", [
    ("LDTM.x4", "x4"),
    ("STTM.x8", "x8"),
    ("LDTM.16dp32bit_t0_t15.x64", "16dp32bit"),
])
def test_ldtm_sttm_unchanged(opcode, want):
    s = SimpleNamespace(opcode=opcode, pc=0)
    assert _f_tmem_shape(s, None, None) == want


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


@pytest.fixture
def b300_tree():
    return json.loads((TREES / "b300.json").read_text())


def test_tmem_copy_children_are_five_point_discrete(b300_tree):
    rt = muses.TreeRuntime(b300_tree)
    nodes = {n["id"]: n for n in b300_tree["nodes"]}
    root = nodes[b300_tree["programs"]["blackwell.tmem.copy"]["root"]]
    assert len(root["children"]) == 2
    for cid, src in zip(root["children"], ("tmem_pipeline.b300.r5.cg1",
                                         "tmem_pipeline.b300.r5.cg2")):
        child = nodes[cid]
        model = child["parameter"]["fields"]["cycles"]["model"]
        assert model["kind"] == "discrete"
        assert model["axes"] == ["tmem_shape"]
        assert len(model["points"]) == 5
        assert {p[0] for p in model["points"]} == set(_SHAPES.values())
        assert child["parameter"]["fields"]["cycles"]["sources"] == [src]


@pytest.mark.parametrize("group", ["", ".2CTA"])
@pytest.mark.parametrize("suffix,shape", list(_SHAPES.items()))
def test_utccp_binds_copy_leaf(b300_tree, group, suffix, shape):
    rt = muses.TreeRuntime(b300_tree)
    cg = 2 if group else 1
    rows = muses.program_facts(
        rt, _static(f"UTCCP.T.S{group}{suffix}"),
        _dyn(uniform={"cta_group": cg}), _ctx(), "issue")
    rec = [r for r in rows if r.get("program") == "blackwell.tmem.copy"]
    assert rec, f"no blackwell.tmem.copy row in {rows}"
    assert "missing" not in rec[0]
    assert rec[0]["facts"]["tmem_shape"] == shape
