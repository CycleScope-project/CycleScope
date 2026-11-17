"""Guard partition and leaf status validation tests."""

from __future__ import annotations

import pytest

from cyclescope.tree.validate import (
    ValidateError, validate_guard_partition)

INT_AXIS = {"kind": "int", "owner": "capture", "unit": "bytes",
            "minimum": 0, "maximum": 8}
ENUM_AXIS = {"kind": "enum", "owner": "capture", "unit": "enum",
             "values": ["a", "b", "c"]}


def _child(operator: str, value) -> dict:
    return {"guard": {"axis": "x", "operator": operator, "value": value},
            "parameter": {}}


def test_int_partition_complete() -> None:
    children = [_child("range", [0, 4]), _child("range", [5, 8])]
    validate_guard_partition(children, INT_AXIS, (0, 8), label="t")


def test_int_partition_gap_rejected() -> None:
    children = [_child("range", [0, 3]), _child("range", [5, 8])]
    with pytest.raises(ValidateError, match="gap or overlap"):
        validate_guard_partition(children, INT_AXIS, (0, 8), label="t")


def test_int_partition_overlap_rejected() -> None:
    children = [_child("range", [0, 5]), _child("range", [4, 8])]
    with pytest.raises(ValidateError, match="gap or overlap"):
        validate_guard_partition(children, INT_AXIS, (0, 8), label="t")


def test_int_partition_tail_missing_rejected() -> None:
    children = [_child("range", [0, 4]), _child("range", [5, 7])]
    with pytest.raises(ValidateError, match="domain ends"):
        validate_guard_partition(children, INT_AXIS, (0, 8), label="t")


def test_enum_partition_and_range_slice() -> None:
    children = [_child("eq", "a"), _child("range", ["b", "c"])]
    validate_guard_partition(children, ENUM_AXIS, {"a", "b", "c"},
                             label="t")
    with pytest.raises(ValidateError, match="overlap"):
        validate_guard_partition(
            [_child("eq", "a"), _child("eq", "a")], ENUM_AXIS,
            {"a", "b", "c"}, label="t")


def test_enum_range_slice_covers_middle() -> None:
    # range is a contiguous slice of the declared values list.
    children = [_child("eq", "a"), _child("range", ["b", "c"])]
    validate_guard_partition(children, ENUM_AXIS, {"a", "b", "c"},
                             label="t")
    bad = [_child("range", ["a", "b"])]
    with pytest.raises(ValidateError, match="cover"):
        validate_guard_partition(bad, ENUM_AXIS, {"a", "b", "c"},
                                 label="t")


def test_leaf_conventions_report():
    """field_models declared for every common mechanism; the checker runs
    on the repository tree and reports the four violation classes."""
    from pathlib import Path
    from cyclescope.tree.conventions import check_conventions
    root = Path(__file__).resolve().parents[2] / "configs" / "mechanism_tree"
    viol = check_conventions(root)
    assert set(viol) == {"field_model", "discrete_axis", "unit", "evidence_crossing", "class_resource"}
    assert viol["class_resource"] == []
    # every common mechanism declares field models (no 'no declaration' lines)
    assert not [v for v in viol["field_model"] if "no field_models" in v]


def test_tree_validate_strict_gate(tmp_path):
    """`cyclescope tree validate --strict` is a gate: the repository tree has no
    leaf-convention violation of any class (exit 0), and an injected
    evidence crossing (one device measured, the other placeholder on a
    common mechanism) makes it exit 1."""
    import json
    import shutil
    from pathlib import Path
    from cyclescope.cli import main
    from cyclescope.tree.conventions import check_conventions
    repo = Path(__file__).resolve().parents[2]
    root = repo / "configs" / "mechanism_tree"
    viol = check_conventions(root)
    assert {k: v for k, v in viol.items() if v} == {}, viol
    assert main(["tree", "validate", "--strict", "--source", str(root)]) == 0
    # inject a crossing: b300 memory.store_dram.service sourced from a placeholder
    copy = tmp_path / "mechanism_tree"
    shutil.copytree(root, copy)
    sp = copy / "blackwell" / "sources.json"
    sources = json.loads(sp.read_text())
    sources["placeholder.test.crossing"] = {"description": "test", "evidence": "placeholder"}
    lp = copy / "blackwell" / "leaves" / "memory.store_dram.service.json"
    leaf = json.loads(lp.read_text())
    for field in leaf["memory.store_dram.service.root"]["fields"].values():
        for key in field["sources"]:
            sources.pop(key, None)  # the build requires every source to be cited
        field["sources"] = ["placeholder.test.crossing"]
    lp.write_text(json.dumps(leaf))
    sp.write_text(json.dumps(sources))
    assert check_conventions(copy)["evidence_crossing"] == [
        "memory.store_dram.service: hopper ['measured'] / blackwell ['placeholder']"]
    assert main(["tree", "validate", "--strict", "--source", str(copy)]) == 1
    assert main(["tree", "validate", "--source", str(copy)]) == 0


def test_evidence_crossing_ignores_structural_only_leaf(tmp_path):
    """A leaf sourced only from structural records (unreachable on that
    device) does not cross a measured leaf of the other device."""
    import json
    import shutil
    from pathlib import Path
    from cyclescope.tree.conventions import check_conventions
    repo = Path(__file__).resolve().parents[2]
    copy = tmp_path / "mechanism_tree"
    shutil.copytree(repo / "configs" / "mechanism_tree", copy)
    for arch, ev in (("hopper", "structural"), ("blackwell", "measured")):
        sp = copy / arch / "sources.json"
        sources = json.loads(sp.read_text())
        sources[f"test.{ev}"] = {"description": "test", "evidence": ev}
        lp = copy / arch / "leaves" / "memory.l2_merge.json"
        leaf = json.loads(lp.read_text())
        for field in leaf["memory.l2_merge.root"]["fields"].values():
            for key in field["sources"]:
                sources.pop(key, None)
            field["sources"] = [f"test.{ev}"]
        lp.write_text(json.dumps(leaf))
        sp.write_text(json.dumps(sources))
    assert check_conventions(copy)["evidence_crossing"] == []
