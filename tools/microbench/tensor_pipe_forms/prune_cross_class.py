#!/usr/bin/env python3
"""Prune unreachable (operation_class, umma_kind) pairs from the b300
tensor.pipe / tensor.completion leaves (tensor_pipe_forms protocol
addendum 2026-11-05 item 4).

A class can only execute the kinds its ISA opcode carries
(converter python/cyclescope/arch/blackwell/semantics.py _umma_kind_of):
UTCHMMA x {f16, tf32}, UTCQMMA x {f8f6f4, mxf8f6f4},
UTCOMMA x {mxf4, mxf4nvf4}.  For each operation_class node the children
partitioned on umma_kind outside the reachable set are dropped and
partition_domain is set to the reachable list.  Reports the leaf points
removed per class, then lists sources.json keys no longer referenced by
any field in blackwell/{leaves,resources,mechanisms}.

usage: python3 prune_cross_class.py [--source-root configs/mechanism_tree]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REACHABLE = {"UTCHMMA": ["f16", "tf32"],
            "UTCQMMA": ["f8f6f4", "mxf8f6f4"],
            "UTCOMMA": ["mxf4", "mxf4nvf4"]}
LEAVES = ("tensor.pipe", "tensor.completion")


def leaf_points(node: dict) -> int:
    n = sum(len(f.get("model", {}).get("points", []))
            for f in node.get("parameter", {}).get("fields", {}).values())
    return n + sum(leaf_points(c) for c in node.get("children", []))


def prune(path: Path) -> None:
    tree = json.loads(path.read_text())
    total = 0
    for cls, keep in REACHABLE.items():
        node = tree.get(f"{path.stem}.root/operation_class={cls}")
        if node is None:
            continue
        removed = [c for c in node.get("children", [])
                   if c.get("guard", {}).get("axis") == "umma_kind"
                   and c["guard"]["value"] not in keep]
        pts = sum(leaf_points(c) for c in removed)
        total += pts
        print(f"{path.name} {cls}: kept {keep}, removed "
              f"{[c['guard']['value'] for c in removed]} ({pts} points)")
        node["children"] = [c for c in node.get("children", [])
                            if c.get("guard", {}).get("axis") != "umma_kind"
                            or c["guard"]["value"] in keep]
        if node.get("partition_axis") == "umma_kind":
            node["partition_domain"] = [k for k in node["partition_domain"]
                                        if k in keep]
    path.write_text(json.dumps(tree, indent=1) + "\n")
    print(f"{path.name}: total {total} points removed")


def used_sources(root: Path) -> set:
    used = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "sources" and isinstance(v, list):
                    used.update(str(s) for s in v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for p in list(root.glob("blackwell/leaves/*.json")) + \
            [root / "blackwell/resources.json"] + \
            list(root.glob("blackwell/mechanisms/*.json")):
        if p.exists():
            walk(json.loads(p.read_text()))
    return used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path,
                    default=Path("configs/mechanism_tree"))
    a = ap.parse_args()
    for name in LEAVES:
        prune(a.source_root / "blackwell/leaves" / f"{name}.json")
    sources = json.loads(
        (a.source_root / "blackwell/sources.json").read_text())
    unref = sorted(set(sources) - used_sources(a.source_root))
    print(f"unreferenced sources ({len(unref)}):")
    for s in unref:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
