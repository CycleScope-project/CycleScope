"""Diff two generated mechanism trees (`cyclescope tree diff`).

Aligns leaves by (mechanism, leaf path, field name) and reports added /
removed / value-changed / status-changed entries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _leaf_index(tree: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Map leaf node id -> leaf record for one tree."""
    out: dict[str, dict[str, Any]] = {}
    for node in tree.get("nodes", []):
        if "parameter" in node or node.get("unsupported"):
            out[node["id"]] = node
    return out


@dataclass
class TreeDiff:
    added_leaves: list[str] = field(default_factory=list)
    removed_leaves: list[str] = field(default_factory=list)
    changed_fields: list[dict[str, Any]] = field(default_factory=list)
    status_changes: list[dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.added_leaves or self.removed_leaves
                    or self.changed_fields or self.status_changes)


def diff_trees(tree_a: Mapping[str, Any], tree_b: Mapping[str, Any],
               *, leaf_values: bool = True) -> TreeDiff:
    """Align leaves by node id; report adds/removes/value/status changes."""
    diff = TreeDiff()
    leaves_a = _leaf_index(tree_a)
    leaves_b = _leaf_index(tree_b)
    diff.added_leaves = sorted(set(leaves_b) - set(leaves_a))
    diff.removed_leaves = sorted(set(leaves_a) - set(leaves_b))
    for node_id in sorted(set(leaves_a) & set(leaves_b)):
        a, b = leaves_a[node_id], leaves_b[node_id]
        param_a = a.get("parameter") or {}
        param_b = b.get("parameter") or {}
        status_a, status_b = param_a.get("status"), param_b.get("status")
        if status_a != status_b:
            diff.status_changes.append({
                "leaf": node_id, "from": status_a, "to": status_b})
        if not leaf_values:
            continue
        fields_a = param_a.get("fields", {})
        fields_b = param_b.get("fields", {})
        for name in sorted(set(fields_a) | set(fields_b)):
            fa, fb = fields_a.get(name), fields_b.get(name)
            if fa is None or fb is None:
                diff.changed_fields.append({
                    "leaf": node_id, "field": name,
                    "from": fa and fa.get("model"),
                    "to": fb and fb.get("model"),
                    "kind": "added" if fb is not None else "removed"})
            elif fa["model"] != fb["model"] or fa.get("field_id") != fb.get(
                    "field_id") or fa.get("unit") != fb.get("unit"):
                diff.changed_fields.append({
                    "leaf": node_id, "field": name,
                    "from": fa["model"], "to": fb["model"],
                    "kind": "changed"})
    return diff


def diff_files(path_a: str | Path, path_b: str | Path,
               *, leaf_values: bool = True) -> TreeDiff:
    return diff_trees(json.loads(Path(path_a).read_text()),
                      json.loads(Path(path_b).read_text()),
                      leaf_values=leaf_values)


def format_diff(diff: TreeDiff) -> str:
    lines: list[str] = []
    for leaf in diff.added_leaves:
        lines.append(f"+ leaf {leaf}")
    for leaf in diff.removed_leaves:
        lines.append(f"- leaf {leaf}")
    for change in diff.changed_fields:
        lines.append(
            f"~ {change['leaf']} .{change['field']} [{change['kind']}] "
            f"{json.dumps(change['from'], sort_keys=True)} -> "
            f"{json.dumps(change['to'], sort_keys=True)}")
    for change in diff.status_changes:
        lines.append(
            f"~ {change['leaf']} status {change['from']} -> {change['to']}")
    if not lines:
        lines.append("(no differences)")
    return "\n".join(lines)
