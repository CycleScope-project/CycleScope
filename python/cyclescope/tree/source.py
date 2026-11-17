"""Mechanism tree source fragments (`configs/mechanism_tree/`).

Loads a source directory tree:

```
<source>/
  common/   axes.json  sources.json  mechanisms/<mechanism>.json
  <arch>/   device.json axes.json resources.json sources.json mechanisms/...
            leaves/<mechanism>.json
```

and expands mechanism fragments (nested ``root``/``children`` structures and
the ``groups`` shorthand) into flat node records for the builder.

A ``common/`` fragment is a mechanism *definition* shared by every device:
program, partition tree and axes.  Its leaves carry ``"fields": "device"``
instead of values; ``<arch>/leaves/<mechanism>.json`` supplies, per
generated node id, either the leaf (``{"fields": {...}, "scope"?: ...}``),
``{"unsupported": true}`` (the class does not exist in this ISA), or a
device-specific subtree (``{"partition_axis": ..., "children": [...]}``,
transitional: counted by ``cyclescope tree validate``).  Every device point
must be supplied -- nothing is defaulted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

EVIDENCE_LEVELS = ("measured", "experimental", "structural", "placeholder")

PHASE_EVENTS = (
    "initialization", "issue", "memory_issue", "barrier_release",
    "collective_launch", "control_transition",
)

MODEL_KINDS = ("constant", "discrete", "linear", "affine")


class SourceError(ValueError):
    """Malformed mechanism tree source."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise SourceError(message)


@dataclass
class SourceTree:
    """Loaded source fragments for one device."""

    device: dict[str, Any]
    axes: dict[str, Any]
    resources: list[dict[str, Any]]
    sources: dict[str, Any]
    # mechanism name -> fragment {"doc", "program", "root"|"groups", "sources"}
    fragments: dict[str, dict[str, Any]] = field(default_factory=dict)
    # common mechanism -> device point counts (leaves / unsupported /
    # subtrees) filled from <arch>/leaves/
    device_stats: dict[str, dict[str, int]] = field(default_factory=dict)


def _load_json(path: Path, what: str) -> Any:
    _fail(path.is_file(), f"missing {what}: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SourceError(f"invalid JSON in {path}: {exc}") from exc


def _load_dir(directory: Path, *, arch: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "axes": {}, "sources": {}, "resources": [], "fragments": {},
    }
    if not directory.is_dir():
        return out
    if arch:
        out["device"] = _load_json(directory / "device.json", "device.json")
    axes_path = directory / "axes.json"
    if axes_path.is_file():
        axes = _load_json(axes_path, "axes.json")
        _fail(isinstance(axes, Mapping), f"{axes_path}: axes must be an object")
        out["axes"] = dict(axes)
    resources_path = directory / "resources.json"
    if resources_path.is_file():
        resources = _load_json(resources_path, "resources.json")
        _fail(isinstance(resources, list), f"{resources_path}: must be a list")
        out["resources"] = list(resources)
    sources_path = directory / "sources.json"
    if sources_path.is_file():
        sources = _load_json(sources_path, "sources.json")
        _fail(isinstance(sources, Mapping),
              f"{sources_path}: sources must be an object")
        out["sources"] = dict(sources)
    mechanisms_dir = directory / "mechanisms"
    if mechanisms_dir.is_dir():
        for path in sorted(mechanisms_dir.glob("*.json")):
            fragment = _load_json(path, f"mechanism fragment {path.name}")
            _fail(isinstance(fragment, Mapping),
                  f"{path}: fragment must be an object")
            out["fragments"][path.stem] = dict(fragment)
    out["leaves"] = {}
    leaves_dir = directory / "leaves"
    if leaves_dir.is_dir():
        for path in sorted(leaves_dir.glob("*.json")):
            leaves = _load_json(path, f"device leaves {path.name}")
            _fail(isinstance(leaves, Mapping),
                  f"{path}: device leaves must be an object")
            out["leaves"][path.stem] = dict(leaves)
    return out


DEVICE_FIELDS = "device"


def _is_device_point(spec: Mapping[str, Any]) -> bool:
    if spec.get("fields") == DEVICE_FIELDS:
        return True
    parameter = spec.get("parameter")
    return isinstance(parameter, Mapping) and \
        parameter.get("fields") == DEVICE_FIELDS


def _device_spec(entry: Mapping[str, Any], guard: Mapping[str, Any] | None,
                 label: str) -> dict[str, Any]:
    """Concrete child spec for one device leaves entry."""
    _fail(isinstance(entry, Mapping), f"{label}: device entry must be an object")
    out: dict[str, Any] = {}
    if guard is not None:
        out["guard"] = guard
    if entry.get("unsupported"):
        out["unsupported"] = True
        return out
    if "fields" in entry:
        _fail(isinstance(entry["fields"], Mapping),
              f"{label}: device leaf fields must be an object")
        out["parameter"] = {
            "scope": entry.get("scope", "instruction"),
            "fields": entry["fields"]}
        return out
    _fail("partition_axis" in entry and "children" in entry,
          f"{label}: device entry needs fields, unsupported or a subtree")
    out.update({k: v for k, v in entry.items() if k != "guard"})
    return out


def materialize_device_fragment(mechanism: str, fragment: Mapping[str, Any],
                                leaves: Mapping[str, Any],
                                *, stats: dict[str, int] | None = None
                                ) -> dict[str, Any]:
    """Resolve a common definition's device points from ``leaves``.

    Returns a concrete fragment (``root`` form) with the same node ids
    :func:`expand_fragment` would generate.  Missing or unused device
    entries are errors.  ``stats`` (optional) collects ``device_leaves``,
    ``device_unsupported`` and ``device_subtrees`` counts.
    """
    used: set[str] = set()
    stats = stats if stats is not None else {}

    def resolve(spec: Mapping[str, Any], node_id: str,
                guard: Mapping[str, Any] | None) -> dict[str, Any]:
        if node_id in leaves:
            used.add(node_id)
            entry = leaves[node_id]
            key = ("device_unsupported" if entry.get("unsupported")
                   else "device_leaves" if "fields" in entry
                   else "device_subtrees")
            stats[key] = stats.get(key, 0) + 1
            return _device_spec(entry, guard, f"{mechanism} {node_id}")
        _fail(not _is_device_point(spec),
              f"{mechanism}: device leaf {node_id} missing from leaves/")
        out = dict(spec)
        if guard is not None:
            out["guard"] = guard
        if "parameter" in spec or spec.get("unsupported"):
            return out
        children = spec.get("children")
        _fail(isinstance(children, list) and children,
              f"{mechanism}: internal node {node_id} needs non-empty children")
        axis = spec.get("partition_axis")
        new_children = []
        for child in _expand_class_children(children):
            child_guard = child["guard"]
            child_axis = child_guard.get("axis", axis)
            child_id = f"{node_id}/{child_axis}={guard_label(child_guard)}"
            new_children.append(resolve(child, child_id, child_guard))
        out["children"] = new_children
        return out

    out_fragment = {k: v for k, v in fragment.items()
                    if k not in ("root", "groups")}
    groups = fragment.get("groups")
    if groups is not None:
        root: dict[str, Any] = {
            "partition_axis": "operation_class",
            "children": []}
        domain: list[str] = []
        for class_list, spec in groups.items():
            for op_class in [c.strip() for c in class_list.split(",") if c.strip()]:
                child_spec = dict(spec)
                if "fields" in child_spec and "parameter" not in child_spec:
                    child_spec["parameter"] = {
                        "scope": child_spec.get("scope", "instruction"),
                        "fields": child_spec.pop("fields")}
                    child_spec.pop("scope", None)
                guard = {"axis": "operation_class", "operator": "eq",
                         "value": op_class}
                root["children"].append(resolve(
                    child_spec, f"{mechanism}.root/operation_class={op_class}",
                    guard))
                domain.append(op_class)
        root["partition_domain"] = sorted(domain)
        out_fragment["root"] = root
    else:
        _fail(isinstance(fragment.get("root"), Mapping),
              f"{mechanism}: fragment needs a root or groups")
        out_fragment["root"] = resolve(fragment["root"], f"{mechanism}.root",
                                       None)
    unused = sorted(set(leaves) - used)
    _fail(not unused,
          f"{mechanism}: device leaves for unknown nodes: {unused[:5]}")
    return out_fragment


def load_source_tree(source_root: str | Path, arch: str) -> SourceTree:
    """Load `common/` plus `<arch>/` fragments for one architecture.

    A mechanism fragment present in both directories is replaced wholesale
    by the arch-level one (no deep merge).  Axes merge; a name defined in
    both must be identical.
    """
    root = Path(source_root)
    common = _load_dir(root / "common", arch=False)
    arch_dir = _load_dir(root / arch, arch=True)

    _fail(isinstance(arch_dir.get("device"), Mapping),
          f"{root / arch}: missing device.json")

    axes = dict(common["axes"])
    for name, axis in arch_dir["axes"].items():
        if name in axes:
            _fail(axes[name] == axis,
                  f"axis {name!r} differs between common/ and {arch}/")
        else:
            axes[name] = axis

    sources = dict(common["sources"])
    for key, record in arch_dir["sources"].items():
        _fail(key not in sources or sources[key] == record,
              f"source {key!r} defined twice with different content")
        sources[key] = record

    # common definitions take their leaves from <arch>/leaves/; a mechanism
    # must not be defined in both common/ and <arch>/mechanisms
    both = sorted(set(common["fragments"]) & set(arch_dir["fragments"]))
    _fail(not both,
          f"mechanisms defined in both common/ and {arch}/: {both}")
    device_stats: dict[str, dict[str, int]] = {}
    fragments: dict[str, dict[str, Any]] = {}
    for mech, fragment in common["fragments"].items():
        stats: dict[str, int] = {}
        fragments[mech] = materialize_device_fragment(
            mech, fragment, arch_dir["leaves"].get(mech, {}), stats=stats)
        device_stats[mech] = stats
    stray = sorted(set(arch_dir["leaves"]) - set(common["fragments"]))
    _fail(not stray,
          f"{arch}/leaves for mechanisms without a common definition: {stray}")
    fragments.update(arch_dir["fragments"])

    # Fragment-local sources merge into the dir-level table.
    for mech, fragment in fragments.items():
        local = fragment.get("sources", {})
        _fail(isinstance(local, Mapping),
              f"{mech}: fragment sources must be an object")
        for key, record in local.items():
            _fail(key not in sources or sources[key] == record,
                  f"source {key!r} of {mech} clashes with dir-level record")
            sources[key] = record

    # execution units: common/resources.json first (shared ids), then
    # architecture-only units
    resources = list(common["resources"]) + list(arch_dir["resources"])
    return SourceTree(
        device=arch_dir["device"], axes=axes, resources=resources,
        sources=sources, fragments=fragments, device_stats=device_stats)


def _expand_class_children(children: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Expand ``classes`` shorthand entries into per-class eq-guard children.

    A child ``{"classes": ["A", "B"], ...}`` stands for one child per class,
    each guarded by ``{"axis": "operation_class", "operator": "eq"}``.
    """
    out: list[Mapping[str, Any]] = []
    for child in children:
        classes = child.get("classes")
        if classes is None:
            out.append(child)
            continue
        _fail(isinstance(classes, list) and classes,
              "classes shorthand needs a non-empty list")
        for op_class in classes:
            expanded = dict(child)
            expanded.pop("classes")
            expanded["guard"] = {
                "axis": "operation_class", "operator": "eq",
                "value": op_class}
            out.append(expanded)
    return out


def guard_label(guard: Mapping[str, Any]) -> str:
    """Deterministic label used inside generated node ids."""
    value = guard["value"]
    if isinstance(value, list) and len(value) == 2 and all(
            isinstance(v, int) for v in value):
        return f"{value[0]}-{value[1]}"
    return str(value)


def expand_fragment(mechanism: str, fragment: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand one mechanism fragment into flat node records.

    Returns a list of dicts with keys ``id``, ``parent``, ``guard``,
    ``partition_axis``, ``partition_domain``, ``children`` (filled by the
    caller), ``parameter`` and ``unsupported``.  Node ids are generated as
    ``<mechanism>.root`` and ``<parent>/<axis>=<label>``.
    """
    nodes: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}

    def emit(node_id: str, parent: str | None, guard: Mapping[str, Any] | None,
             spec: Mapping[str, Any]) -> dict[str, Any]:
        _fail(node_id not in by_id, f"{mechanism}: duplicate node id {node_id}")
        node: dict[str, Any] = {
            "id": node_id, "parent": parent, "guard": guard,
            "partition_axis": spec.get("partition_axis"),
            "partition_domain": spec.get("partition_domain"),
            "children": [],
            "parameter": spec.get("parameter"),
            "unsupported": bool(spec.get("unsupported", False)),
        }
        nodes.append(node)
        by_id[node_id] = node
        return node

    def visit(spec: Mapping[str, Any], node_id: str, parent: str | None,
              guard: Mapping[str, Any] | None) -> None:
        node = emit(node_id, parent, guard, spec)
        if node["parameter"] is not None or node["unsupported"]:
            _fail("children" not in spec and "partition_axis" not in spec,
                  f"{mechanism}: leaf {node_id} cannot have children")
            return
        children = spec.get("children")
        _fail(isinstance(children, list) and children,
              f"{mechanism}: internal node {node_id} needs non-empty children")
        _fail(isinstance(node["partition_axis"], str),
              f"{mechanism}: internal node {node_id} needs partition_axis")
        for child in _expand_class_children(children):
            _fail(isinstance(child, Mapping) and isinstance(child.get("guard"), Mapping),
                  f"{mechanism}: child of {node_id} needs a guard")
            child_guard = child["guard"]
            axis = child_guard.get("axis", node["partition_axis"])
            child_id = f"{node_id}/{axis}={guard_label(child_guard)}"
            visit(child, child_id, node_id, child_guard)
            node["children"].append(child_id)

    root_spec = fragment.get("root")
    groups = fragment.get("groups")
    if groups is not None:
        _fail(root_spec is None,
              f"{mechanism}: cannot mix root and groups")
        _fail(isinstance(groups, Mapping) and groups,
              f"{mechanism}: groups must be a non-empty object")
        # Expand the groups shorthand into an operation_class partition.
        root_node = emit(f"{mechanism}.root", None, None, {
            "partition_axis": "operation_class",
            "partition_domain": sorted(_all_group_classes(groups)),
        })
        domain: set[str] = set()
        for class_list, spec in groups.items():
            classes = [c.strip() for c in class_list.split(",") if c.strip()]
            _fail(classes, f"{mechanism}: empty operation_class group")
            for op_class in classes:
                _fail(op_class not in domain,
                      f"{mechanism}: operation_class {op_class} in two groups")
                domain.add(op_class)
                child_spec = dict(spec)
                if "fields" in child_spec and "parameter" not in child_spec:
                    child_spec["parameter"] = {
                        "scope": child_spec.get("scope", "instruction"),
                        "fields": child_spec.pop("fields")}
                    child_spec.pop("scope", None)
                child_spec["guard"] = {
                    "axis": "operation_class", "operator": "eq",
                    "value": op_class}
                visit(child_spec,
                      f"{mechanism}.root/operation_class={op_class}",
                      root_node["id"], child_spec["guard"])
                root_node["children"].append(
                    f"{mechanism}.root/operation_class={op_class}")
        root_node["partition_domain"] = sorted(domain)
    else:
        _fail(isinstance(root_spec, Mapping),
              f"{mechanism}: fragment needs a root or groups")
        visit(root_spec, f"{mechanism}.root", None, None)
    return nodes


def _all_group_classes(groups: Mapping[str, Any]) -> Iterable[str]:
    for class_list in groups:
        for op_class in class_list.split(","):
            if op_class.strip():
                yield op_class.strip()


def field_sources_used(fragment: Mapping[str, Any]) -> set[str]:
    """Collect source keys referenced by a fragment's leaf fields."""
    used: set[str] = set()

    def visit(spec: Mapping[str, Any]) -> None:
        parameter = spec.get("parameter")
        if parameter:
            for fdef in parameter.get("fields", {}).values():
                used.update(fdef.get("sources", []))
        for child in spec.get("children", []) or []:
            visit(child)

    if "root" in fragment:
        visit(fragment["root"])
    for spec in fragment.get("groups", {}).values():
        visit(spec)
    return used
