"""Generated-tree build (digest stability, status) and diff tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cyclescope.tree.build import assemble_tree, build_tree
from cyclescope.tree.diff import diff_trees
from cyclescope.tree.registry import load_registry
from cyclescope.tree.source import load_source_tree
from cyclescope.tree.validate import validate_generated_tree

DATA = Path(__file__).resolve().parents[1] / "data" / "mechanism_tree" / "mini"


@pytest.fixture(scope="module")
def mini_tree() -> dict:
    return build_tree(DATA, "miniarch", DATA / "registry.json")


def test_build_mini_tree(mini_tree: dict) -> None:
    validate_generated_tree(mini_tree)
    assert mini_tree["device"] == "mini0"
    assert mini_tree["leaf_count"] == 5
    assert mini_tree["status_counts"] == {
        "measured": 3, "experimental": 1, "placeholder": 1}
    # resource name resolved to id 0
    leaf = {n["id"]: n for n in mini_tree["nodes"]}["mini.leaf.root"]
    fields = leaf["parameter"]["fields"]
    assert fields["resource"]["model"]["value"] == 0
    assert fields["latency_cycles"]["field_id"] == 1
    assert leaf["parameter"]["status"] == "measured"
    # program field_mask = bits of fields 1 and 2
    program = mini_tree["programs"]["mini.leaf"]
    assert program["field_mask"] == 0b110
    # groups fragment expanded into an operation_class partition
    group_root = {n["id"]: n for n in mini_tree["nodes"]}[
        "mini.groups.root"]
    assert group_root["partition_axis"] == "operation_class"
    assert len(group_root["children"]) == 2


def test_digest_stable(mini_tree: dict) -> None:
    again = build_tree(DATA, "miniarch", DATA / "registry.json")
    assert again["content_digest"] == mini_tree["content_digest"]
    assert json.dumps(again, sort_keys=True) == json.dumps(
        mini_tree, sort_keys=True)


def test_status_derivation_placeholder(mini_tree: dict) -> None:
    leaf = {n["id"]: n for n in mini_tree["nodes"]}[
        "mini.partition.root/width_bytes=5-8"]
    assert leaf["parameter"]["status"] == "placeholder"


def test_diff_alignment(mini_tree: dict) -> None:
    changed = copy.deepcopy(mini_tree)
    for node in changed["nodes"]:
        if node["id"] == "mini.leaf.root":
            node["parameter"]["fields"]["latency_cycles"]["model"][
                "value"] = 9
            node["parameter"]["status"] = "experimental"
        if node["id"] == "mini.partition.root/width_bytes=0-4":
            del node["parameter"]  # leaf removed
    changed["nodes"].append({
        "id": "new.leaf.root", "parent": None, "guard": None,
        "children": [], "parameter": {
            "scope": "instruction", "status": "placeholder",
            "fields": {}}})
    diff = diff_trees(mini_tree, changed)
    assert diff.added_leaves == ["new.leaf.root"]
    assert diff.removed_leaves == ["mini.partition.root/width_bytes=0-4"]
    assert [c["leaf"] for c in diff.changed_fields] == ["mini.leaf.root"]
    assert diff.status_changes == [
        {"leaf": "mini.leaf.root", "from": "measured",
         "to": "experimental"}]


def test_fragment_rejects_unknown_source(mini_tree: dict) -> None:
    source_tree = load_source_tree(DATA, "miniarch")
    fragment = {
        "doc": "bad",
        "program": {"event": "issue",
                    "parameter_schema": "timing_row",
                    "timing_kind": "nanosleep_warp_delay"},
        "root": {"parameter": {"scope": "instruction", "fields": {
            "cycles": {"model": {"kind": "constant", "value": 1.0},
                       "sources": ["src.missing"]}}}},
    }
    source_tree.fragments["bad"] = fragment
    registry = load_registry(DATA / "registry.json")
    with pytest.raises(Exception, match="unknown source"):
        assemble_tree(source_tree, registry)
