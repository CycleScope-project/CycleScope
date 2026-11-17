"""Fragment expansion tests (nested roots and the groups shorthand)."""

from __future__ import annotations

from cyclescope.tree.source import expand_fragment


def _leaf(value: float) -> dict:
    return {"parameter": {"scope": "instruction", "fields": {
        "cycles": {"model": {"kind": "constant", "value": value},
                   "sources": []}}}}


def test_expand_nested_fragment() -> None:
    fragment = {
        "program": {"event": "issue", "parameter_schema": "timing_row"},
        "root": {
            "partition_axis": "width_bytes",
            "partition_domain": [0, 8],
            "children": [
                {"guard": {"axis": "width_bytes", "operator": "range",
                           "value": [0, 4]},
                 **_leaf(1.0)},
                {"guard": {"axis": "width_bytes", "operator": "range",
                           "value": [5, 8]},
                 "partition_axis": "operation_class",
                 "partition_domain": ["X", "Y"],
                 "children": [
                     {"guard": {"axis": "operation_class", "operator": "eq",
                                "value": "X"}, **_leaf(2.0)},
                     {"guard": {"axis": "operation_class", "operator": "eq",
                                "value": "Y"}, **_leaf(3.0)},
                 ]},
            ],
        },
    }
    nodes = expand_fragment("m", fragment)
    by_id = {n["id"]: n for n in nodes}
    assert "m.root" in by_id
    assert by_id["m.root"]["children"] == [
        "m.root/width_bytes=0-4",
        "m.root/width_bytes=5-8",
    ]
    inner = by_id["m.root/width_bytes=5-8"]
    assert inner["partition_axis"] == "operation_class"
    assert set(inner["children"]) == {
        "m.root/width_bytes=5-8/operation_class=X",
        "m.root/width_bytes=5-8/operation_class=Y"}
    leaf = by_id["m.root/width_bytes=5-8/operation_class=Y"]
    assert leaf["parameter"]["fields"]["cycles"]["model"]["value"] == 3.0
    assert leaf["guard"] == {"axis": "operation_class", "operator": "eq",
                            "value": "Y"}


def test_expand_groups_shorthand() -> None:
    fragment = {
        "program": {"event": "issue", "parameter_schema": "timing_row",
                    "operation_classes": ["X", "Y"]},
        "groups": {
            "X,Y": {"fields": {"cycles": {
                "model": {"kind": "constant", "value": 7.0},
                "sources": ["s"]}}},
        },
    }
    nodes = expand_fragment("g", fragment)
    by_id = {n["id"]: n for n in nodes}
    assert by_id["g.root"]["partition_axis"] == "operation_class"
    assert sorted(by_id["g.root"]["children"]) == [
        "g.root/operation_class=X", "g.root/operation_class=Y"]
    leaf = by_id["g.root/operation_class=X"]
    assert leaf["parameter"]["fields"]["cycles"]["model"]["value"] == 7.0


# ---- common definitions with per-device leaves -------------------------

import pytest

from cyclescope.tree.source import SourceError, materialize_device_fragment


def _device_def() -> dict:
    return {
        "program": {"event": "issue", "parameter_schema": "timing_row",
                    "operation_classes": ["X", "Y", "Z"]},
        "root": {
            "partition_axis": "operation_class",
            "partition_domain": ["X", "Y", "Z"],
            "children": [
                {"classes": ["X", "Y"], "parameter": {"fields": "device"}},
                {"guard": {"axis": "operation_class", "operator": "eq",
                           "value": "Z"},
                 "parameter": {"scope": "sm", "fields": "device"}},
            ],
        },
    }


def _fields(value: float) -> dict:
    return {"cycles": {"model": {"kind": "constant", "value": value},
                       "sources": []}}


def test_device_leaves_fill_common_definition() -> None:
    leaves = {
        "m.root/operation_class=X": {"fields": _fields(1.0)},
        "m.root/operation_class=Y": {"unsupported": True},
        "m.root/operation_class=Z": {
            "partition_axis": "width_bytes", "partition_domain": [0, 8],
            "children": [
                {"guard": {"axis": "width_bytes", "operator": "range",
                           "value": [0, 8]}, **_leaf(5.0)}]},
    }
    stats: dict = {}
    fragment = materialize_device_fragment("m", _device_def(), leaves,
                                           stats=stats)
    assert stats == {"device_leaves": 1, "device_unsupported": 1,
                     "device_subtrees": 1}
    nodes = {n["id"]: n for n in expand_fragment("m", fragment)}
    assert nodes["m.root/operation_class=X"]["parameter"]["fields"][
        "cycles"]["model"]["value"] == 1.0
    assert nodes["m.root/operation_class=Y"]["unsupported"] is True
    assert nodes["m.root/operation_class=Y"]["parameter"] is None
    assert nodes["m.root/operation_class=Z/width_bytes=0-8"]["parameter"][
        "fields"]["cycles"]["model"]["value"] == 5.0
    # the common guard is kept on the materialised node
    assert nodes["m.root/operation_class=Z"]["guard"]["value"] == "Z"


def test_device_leaves_missing_rejected() -> None:
    leaves = {"m.root/operation_class=X": {"fields": _fields(1.0)}}
    with pytest.raises(SourceError, match="missing from leaves"):
        materialize_device_fragment("m", _device_def(), leaves)


def test_device_leaves_unknown_node_rejected() -> None:
    leaves = {f"m.root/operation_class={c}": {"fields": _fields(1.0)}
              for c in "XYZ"}
    leaves["m.root/operation_class=Q"] = {"fields": _fields(2.0)}
    with pytest.raises(SourceError, match="unknown nodes"):
        materialize_device_fragment("m", _device_def(), leaves)


def test_device_groups_shorthand() -> None:
    fragment = {
        "program": {"event": "issue", "parameter_schema": "timing_row",
                    "operation_classes": ["X", "Y"]},
        "groups": {"X,Y": {"fields": "device"}},
    }
    leaves = {"m.root/operation_class=X": {"fields": _fields(1.0)},
              "m.root/operation_class=Y": {"fields": _fields(2.0)}}
    out = materialize_device_fragment("m", fragment, leaves)
    assert "groups" not in out and out["root"]["partition_domain"] == ["X", "Y"]
    nodes = {n["id"]: n for n in expand_fragment("m", out)}
    assert nodes["m.root/operation_class=Y"]["parameter"]["fields"][
        "cycles"]["model"]["value"] == 2.0
