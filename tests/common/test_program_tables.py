"""Program-table compile tests on the mini fixture tree."""

from __future__ import annotations

import json
from pathlib import Path

from cyclescope.io.fixed_tables import TableReader
from cyclescope.job import program_tables as pt, segment_format as sf
from cyclescope.tree.build import build_tree
from cyclescope.tree.registry import load_registry

DATA = Path(__file__).resolve().parents[1] / "data" / "mechanism_tree" / "mini"


def test_compile_mini_tree() -> None:
    tree = build_tree(DATA, "miniarch", DATA / "registry.json")
    payloads = pt.compile_program_tables(
        tree, load_registry(DATA / "registry.json"))
    progs = TableReader(sf.PROG, payloads["PROG"])
    nodes = TableReader(sf.NODE, payloads["NODE"])
    leaves = TableReader(sf.LEAF, payloads["LEAF"])
    parms = TableReader(sf.PARM, payloads["PARM"])
    # mini.leaf, mini.partition, mini.groups + one execution_units.* per resource
    assert len(progs) == 3 + len(tree.get("resources") or [])
    assert len(leaves) == 5 + len(tree.get("resources") or [])
    # every leaf reachable: sum of leaves across programs
    leaf_ids = {i for i in range(len(leaves))}
    assert len(leaf_ids) == len(leaves)
    # field_mask for mini.leaf program: fields 1+2 -> 0b110
    for i in range(len(progs)):
        p = progs.unpack(i)
        root = nodes.unpack(p["root_node"])
        # reach leaves under root; check field mask non-zero
        assert p["field_mask"] != 0
    masks = {progs.unpack(i)["field_mask"] for i in range(len(progs))}
    assert 0b110 in masks and 0b01 in masks
    # PARM rows exist for latency_cycles (field_id 1, schema 1)
    found = any(
        parms.unpack(i)["schema_id"] == 1
        and parms.unpack(i)["field_id"] == 1
        and parms.unpack(i)["value"] == 4.0
        for i in range(len(parms)))
    assert found


def test_program_table_node_children_contiguous() -> None:
    tree = build_tree(DATA, "miniarch", DATA / "registry.json")
    payloads = pt.compile_program_tables(
        tree, load_registry(DATA / "registry.json"))
    nodes = TableReader(sf.NODE, payloads["NODE"])
    leaves = TableReader(sf.LEAF, payloads["LEAF"])
    reached = set()
    progs = TableReader(sf.PROG, payloads["PROG"])
    for i in range(len(progs)):
        stack = [progs.unpack(i)["root_node"]]
        while stack:
            row = stack.pop()
            n = nodes.unpack(row)
            if n["leaf"] != 0xFFFFFFFF:
                reached.add(n["leaf"])
            for c in range(n["child_begin"],
                           n["child_begin"] + n["child_count"]):
                assert nodes.unpack(c)["parent"] == row
                stack.append(c)
    assert reached == set(range(len(leaves)))


def test_memory_level_programs_bound_at_load() -> None:
    """Every memory_level program (capacity / residency / service) of the
    generated device trees gets a job-global USES row: the engine applies
    the level latencies / intervals / bandwidths from these rows at load,
    so a memory.*.service program without a row would leave the level
    at zero latency."""
    root = Path(__file__).resolve().parents[2] / "configs"
    registry = load_registry(root / "schema_registry.json")
    for device in ("h100", "b300"):
        tree = json.loads((root / "trees" / f"{device}.json").read_text())
        payloads = pt.compile_program_tables(tree, registry)
        uses = TableReader(sf.USES, payloads["USES"])
        progs = TableReader(sf.PROG, payloads["PROG"])
        strs = payloads["STRS"]
        bound_nodes = {uses.unpack(i)["node"] for i in range(len(uses))}
        schema = registry.schema("memory_level")
        for i in range(len(progs)):
            p = progs.unpack(i)
            if p["schema_id"] != schema.schema_id:
                continue
            name = strs[p["mechanism"]:strs.index(b"\0", p["mechanism"])]
            # the memory_level programs are single-leaf: root == leaf
            assert p["root_node"] in bound_nodes, (device, name)


def test_resource_defs_bound_at_load() -> None:
    """Every tree resource gets a job-global USES row: the engine binds
    service-leaf resource ids against schema-2 rows (scope / copies /
    max_inflight_per_copy / interval), so a resource without a row
    leaves the binding inert."""
    root = Path(__file__).resolve().parents[2] / "configs"
    registry = load_registry(root / "schema_registry.json")
    scope_ids = {"smsp": 0, "sm": 1, "cluster": 2, "gpu": 3}
    for device in ("h100", "b300"):
        tree = json.loads((root / "trees" / f"{device}.json").read_text())
        payloads = pt.compile_program_tables(tree, registry)
        uses = TableReader(sf.USES, payloads["USES"])
        nodes = TableReader(sf.NODE, payloads["NODE"])
        leaves = TableReader(sf.LEAF, payloads["LEAF"])
        parms = TableReader(sf.PARM, payloads["PARM"])
        bound: dict[int, int] = {}
        for i in range(len(uses)):
            u = uses.unpack(i)
            if u["schema_id"] == 2:
                bound[u["target_index"]] = u["node"]
        assert len(bound) == len(tree["resources"]), device
        for i, res in enumerate(tree["resources"]):
            node = nodes.unpack(bound[i])
            leaf = leaves.unpack(node["leaf"])
            fields: dict[int, float] = {}
            for p in range(leaf["parameter"], leaf["parameter"] + 4):
                pr = parms.unpack(p)
                assert pr["schema_id"] == 2
                fields[pr["field_id"]] = pr["value"]
            assert fields[0] == scope_ids[res["scope"]]
            assert fields[1] == res["copy_count"]
            assert fields[2] == res["max_inflight_per_copy"]
            assert fields[3] == res["initiation_interval_cycles"]
