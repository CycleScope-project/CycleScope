import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cyclescope.convert import model
from cyclescope.convert.semantics_common import (SemanticsContext,
                                             ConversionError)
from cyclescope.arch.blackwell import rules
from cyclescope.arch.hopper import rules as hopper_rules


def _make(with_addr: bool, opcode="STS.64", uniform=None, width=8):
    unit = model.ClusterUnit(cluster_id=0)
    unit.ctas = [model.CtaFacts(cta_linear=0, cta=(0, 0, 0),
                                cluster_rank=0, sm_id=0)]
    ctx = SemanticsContext(unit=unit)
    ctx.extra["mbarrier_addresses"] = {0x40} if with_addr else set()
    st = model.StaticInstruction(pc=0, code_address=0, opcode=opcode,
                                 operation_class="storage",
                                 operands=("R0", "[R2]", "R4"))
    dyn = model.DynamicInstruction(
        static_index=0, active_mask=1, predicate_mask=1,
        memory=(model.MemoryReference(
            space="shared", access_kind="store", width_bytes=width,
            lane_addresses=(0x40,), lane_mask=1),),
        operands=model.SemanticOperands(
            uniform={"expected_arrivals": 1} if uniform is None
            else uniform))
    return st, dyn, ctx


def test_sts64_hit_is_mbarrier_init():
    st, dyn, ctx = _make(True)
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is not None
    assert row.kind.name == "mbarrier_init"
    assert row.fields["expected_arrivals"] == 1


def test_sts64_miss_stays_store():
    st, dyn, ctx = _make(False)
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is None or row.kind.name != "mbarrier_init"


@pytest.mark.parametrize("arch", [rules, hopper_rules])
def test_sts32_hit_is_mbarrier_init(arch):
    st, dyn, ctx = _make(True, opcode="STS", width=4)
    row = arch.semantics(st, dyn, ctx, 0, 0)
    assert row is not None and row.kind.name == "mbarrier_init"


@pytest.mark.parametrize("arch", [rules, hopper_rules])
def test_sts_hit_without_value_is_missing_fact(arch):
    st, dyn, ctx = _make(True, opcode="STS", uniform={}, width=4)
    with pytest.raises(ConversionError, match="missing_fact:mbarrier_init_value"):
        arch.semantics(st, dyn, ctx, 0, 0)


def test_sts128_never_promoted():
    st, dyn, ctx = _make(True, opcode="STS.128", uniform={}, width=16)
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is None or row.kind.name != "mbarrier_init"


def test_stsm_never_promoted():
    st, dyn, ctx = _make(True, opcode="STSM.16.M88.4", uniform={}, width=16)
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is None or row.kind.name != "mbarrier_init"


def test_rank_tagged_address_matches_local_set():
    st, dyn, ctx = _make(True, opcode="STS", width=4)
    ctx.local_shared = rules.local_shared_address
    dyn.memory[0].lane_addresses = ((1 << 24) | 0x40,)
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is not None and row.kind.name == "mbarrier_init"


@pytest.mark.parametrize("arch", [rules, hopper_rules])
def test_sts_captured_non_init_value_not_promoted(arch):
    # STS writing a data word (0) onto an address that later becomes a
    # barrier: captured but not a legal init state -> stays a store
    st, dyn, ctx = _make(True, opcode="STS", width=4,
                         uniform={"store_value_captured": 1})
    row = arch.semantics(st, dyn, ctx, 0, 0)
    assert row is None or row.kind.name != "mbarrier_init"


INIT_WORD = (0x7ffff800 << 32) | 0x1ffffe   # expected_arrivals 1, phase 0


def _make_multi(nlanes: int, values, opcode="STS.64", mask=None):
    unit = model.ClusterUnit(cluster_id=0)
    unit.ctas = [model.CtaFacts(cta_linear=0, cta=(0, 0, 0),
                                cluster_rank=0, sm_id=0)]
    ctx = SemanticsContext(unit=unit)
    addrs = tuple(0x400 + 8 * i for i in range(nlanes))
    ctx.extra["mbarrier_addresses"] = set(addrs)
    st = model.StaticInstruction(pc=0x9d0, code_address=0, opcode=opcode,
                                 operation_class="storage",
                                 operands=("R0", "[R2]", "R4"))
    m = (1 << nlanes) - 1 if mask is None else mask
    dyn = model.DynamicInstruction(
        static_index=0, active_mask=m, predicate_mask=m,
        memory=(model.MemoryReference(
            space="shared", access_kind="store", width_bytes=8,
            lane_addresses=addrs, lane_mask=m),),
        operands=model.SemanticOperands(
            uniform={"expected_arrivals": 1, "initial_phase": 0},
            lane={"sts64_value": tuple(values)}))
    return st, dyn, ctx


def test_sts64_multi_lane_init_mbol():
    st, dyn, ctx = _make_multi(6, [INIT_WORD] * 6)
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is not None and row.kind.name == "mbarrier_init"
    assert row.object_list is not None and len(row.object_list) == 6
    objs = {ctx.mbarrier(0, a, "t") for a in (0x400 + 8 * i for i in range(6))}
    assert set(row.object_list) == objs
    assert row.fields["object"] == row.object_list[0]


def test_sts64_divergent_lane_values_rejected():
    vals = [INIT_WORD] * 6
    vals[3] = INIT_WORD + 2          # different decoded count
    st, dyn, ctx = _make_multi(6, vals)
    with pytest.raises(ConversionError,
                       match="unsupported:mbarrier_init:divergent_lane_values"):
        rules.semantics(st, dyn, ctx, 0, 0)


def test_sts64_single_lane_no_object_list():
    st, dyn, ctx = _make_multi(1, [INIT_WORD])
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is not None and row.kind.name == "mbarrier_init"
    assert row.object_list is None


def test_sts64_compacted_exec_lanes():
    # exec mask 0x30 (lanes 4,5 only): the facts compact per-lane arrays
    # to the executing lanes -> 2 addresses, 2 init words, 2 objects
    st, dyn, ctx = _make_multi(2, [INIT_WORD] * 2, mask=0x30)
    dyn.memory[0].lane_mask = 0x30
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is not None and row.kind.name == "mbarrier_init"
    assert row.object_list is not None and len(row.object_list) == 2


def test_sts64_fullwidth_exec_lanes():
    # full-width 32-entry form: entries indexed by lane id, mask selects
    st, dyn, ctx = _make_multi(2, [INIT_WORD] * 32, mask=0x30)
    dyn.memory[0].lane_mask = 0
    dyn.operands.lane["sts64_value"] = tuple([INIT_WORD] * 32)
    row = rules.semantics(st, dyn, ctx, 0, 0)
    assert row is not None and row.kind.name == "mbarrier_init"
    assert row.object_list is not None and len(row.object_list) == 2
