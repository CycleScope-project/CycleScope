"""shared_cluster memory-space resolution (MACC flags) + storage.shared
binding for generic LD/ST."""

from cyclescope.job.segment_builder import _SPACE_CODES
from cyclescope.convert.facts_format import _SPACE


def test_space_codes():
    assert _SPACE_CODES["shared_cluster"] == 0x1
    assert _SPACE_CODES["shared"] == 0x1
    assert _SPACE_CODES["global_system"] == 0x0
    assert _SPACE_CODES.get("param", 0x4) == 0x4
    # facts enum ints used by the cpp path (facts_format._SPACE)
    assert _SPACE["shared_cluster"] == 6
    assert _SPACE["global_system"] == 7


import json
from pathlib import Path

import pytest

from cyclescope.convert import mechanism_uses as muses, axes, model

TREES = Path(__file__).resolve().parents[2] / "configs" / "trees"


@pytest.fixture(params=["h100", "b300"])
def tree(request):
    return json.loads((TREES / f"{request.param}.json").read_text())


def _ld():
    return model.StaticInstruction(
        pc=0x10, code_address=None, opcode="LD.E.128",
        operation_class="LD", memory_space="generic")


def _dyn_ld(space):
    addrs = tuple(0x1400 + i * 16 for i in range(32))
    return model.DynamicInstruction(
        static_index=0, active_mask=0xFFFFFFFF,
        predicate_mask=0xFFFFFFFF,
        memory=(model.MemoryReference(
            space=space, access_kind="load", width_bytes=16,
            lane_addresses=addrs, lane_mask=0xFFFFFFFF),))


def _ctx():
    return axes.AxisContext(unit=None, launch=None, warp=None,
                            flow=axes.FlowContext(), rules=None)


def test_generic_ld_global_skips_shared_programs(tree):
    """A generic LD resolved to global: storage.shared.* programs are
    not applicable (NOT_APPLICABLE skips the row) and must not produce
    a missing_fact."""
    rt = muses.TreeRuntime(tree)
    rows = muses.program_facts(rt, _ld(), _dyn_ld("global"), _ctx(),
                             "issue")
    # the leaf whose model needs shared_wavefronts (acceptance_sm) is
    # skipped as not applicable; the constant issue/visibility leaves
    # still bind and are vetoed engine-side by the MACC space
    assert not [r for r in rows if "missing" in r]
    assert not [r for r in rows if r.get("program")
                == "storage.shared.acceptance_sm"]


def test_generic_ld_shared_cluster_binds_shared_programs(tree):
    """The same LD resolved to cluster shared memory binds the
    storage.shared programs and records the wavefront fact."""
    rt = muses.TreeRuntime(tree)
    rows = muses.program_facts(rt, _ld(), _dyn_ld("shared_cluster"),
                             _ctx(), "issue")
    rec = [r for r in rows
           if r.get("program") == "storage.shared.acceptance_sm"]
    assert rec, f"no acceptance_sm row in {[r.get('program') for r in rows]}"
    assert "shared_wavefronts" in rec[0]["facts"]
