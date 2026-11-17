"""Validation for mechanism tree sources and generated trees (spec §6)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .registry import Registry
from .source import (
    EVIDENCE_LEVELS, PHASE_EVENTS, MODEL_KINDS, SourceError,
    _expand_class_children)


class ValidateError(ValueError):
    """Mechanism tree failed validation."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValidateError(message)


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def compute_content_digest(tree: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON without the ``content_digest`` key."""
    payload = {k: v for k, v in tree.items() if k != "content_digest"}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _axis_domain(axis: Mapping[str, Any]) -> tuple[int, int] | set:
    if axis["kind"] == "int":
        return int(axis["minimum"]), int(axis["maximum"])
    return set(axis["values"])


def _guard_covers(guard: Mapping[str, Any], axis: Mapping[str, Any]) -> tuple[int, int] | set:
    """Coverage of a guard as an int interval or a set of enum values."""
    operator = guard["operator"]
    value = guard["value"]
    if axis["kind"] == "int":
        lo, hi = _axis_domain(axis)
        if operator == "eq":
            _fail(isinstance(value, int) and lo <= value <= hi,
                  f"guard eq {value} outside axis domain [{lo}, {hi}]")
            return value, value
        _fail(operator == "range",
              f"int axis guard must be eq or range, got {operator}")
        _fail(isinstance(value, list) and len(value) == 2
              and all(isinstance(v, int) for v in value)
              and lo <= value[0] <= value[1] <= hi,
              f"guard range {value} outside axis domain [{lo}, {hi}]")
        return value[0], value[1]
    # enum axis: `eq` is one value; `range` is a contiguous slice of the
    # declared values list (inclusive of both endpoints).
    values = list(axis["values"])
    if operator == "eq":
        _fail(value in values, f"guard eq {value!r} not in enum values")
        return {value}
    _fail(operator == "range",
          f"enum axis guard must be eq or range, got {operator}")
    _fail(isinstance(value, list) and len(value) == 2
          and value[0] in values and value[1] in values,
          f"enum range {value} endpoints must be declared values")
    lo_i, hi_i = values.index(value[0]), values.index(value[1])
    _fail(lo_i <= hi_i, f"enum range {value} is inverted")
    return set(values[lo_i:hi_i + 1])


def validate_guard_partition(children: list[Mapping[str, Any]],
                             axis: Mapping[str, Any],
                             domain: tuple[int, int] | set,
                             *, label: str) -> None:
    """Children guards must completely and disjointly cover ``domain``."""
    if axis["kind"] == "int":
        lo, hi = domain
        intervals = sorted(
            (_guard_covers(child["guard"], axis) for child in children),
            key=lambda pair: pair[0])
        cursor = lo
        for start, end in intervals:
            _fail(start == cursor,
                  f"{label}: guard partition gap or overlap at {cursor}")
            _fail(start <= end, f"{label}: empty guard interval")
            cursor = end + 1
        _fail(cursor == hi + 1,
              f"{label}: guard partition ends at {cursor - 1}, domain ends {hi}")
    else:
        covered: set = set()
        for child in children:
            part = _guard_covers(child["guard"], axis)
            _fail(not (covered & part),
                  f"{label}: enum guards overlap on {sorted(covered & part)}")
            covered |= part
        _fail(covered == domain,
              f"{label}: enum guards cover {sorted(covered)} but domain is "
              f"{sorted(domain)}")


def _check_model(model: Mapping[str, Any], axes: Mapping[str, Any],
                 *, label: str) -> None:
    kind = model.get("kind")
    _fail(kind in MODEL_KINDS, f"{label}: unknown model kind {kind!r}")

    def declared(axis_name: Any, want_int: bool) -> None:
        _fail(axis_name in axes, f"{label}: model axis {axis_name!r} undeclared")
        if want_int:
            _fail(axes[axis_name]["kind"] == "int",
                  f"{label}: model axis {axis_name!r} must be int")

    if kind == "constant":
        _fail(set(model) == {"kind", "value"}
              and isinstance(model["value"], (int, float, str)),
              f"{label}: invalid constant model")
    elif kind == "discrete":
        _fail(set(model) <= {"kind", "axes", "points", "placeholder_default",
                             "fail_closed"},
              f"{label}: invalid discrete model fields")
        axis_names = model.get("axes")
        _fail(isinstance(axis_names, list) and axis_names,
              f"{label}: discrete model needs axes")
        for axis_name in axis_names:
            declared(axis_name, want_int=False)
        points = model.get("points")
        _fail(isinstance(points, list) and points,
              f"{label}: discrete model needs points")
        if (not model.get("fail_closed")
                and "placeholder_default" not in model
                and len(axis_names) == 1):
            axis = axes[axis_names[0]]
            seen = set()
            for point in points:
                _fail(isinstance(point, list) and len(point) == 2,
                      f"{label}: discrete point must be [value, output]")
                seen.add(point[0])
            domain = _axis_domain(axis)
            if axis["kind"] == "enum":
                _fail(seen == domain,
                      f"{label}: discrete points do not cover enum axis "
                      f"{axis_names[0]!r} (declare fail_closed to exempt)")
    else:
        _fail("axis" in model, f"{label}: {kind} model needs an axis")
        declared(model["axis"], want_int=True)
        if kind == "linear":
            points = model.get("points")
            # 1 point is legal: the axis was measured at a single
            # value only; the engine clamps at the endpoint, so the
            # model extrapolates as a constant.
            _fail(isinstance(points, list) and len(points) >= 1,
                  f"{label}: linear model needs >= 1 point")
            xs = [p[0] for p in points]
            _fail(xs == sorted(xs) and len(set(xs)) == len(xs),
                  f"{label}: linear knots must be strictly increasing")
        if kind == "affine":
            _fail({"domain", "slope", "intercept"} <= set(model),
                  f"{label}: affine model needs domain, slope, intercept")


def validate_fragment(mechanism: str, fragment: Mapping[str, Any],
                      axes: Mapping[str, Any], sources: Mapping[str, Any],
                      resources: Mapping[str, int], registry: Registry,
                      *, resource_names: set[str] | None = None) -> None:
    """Validate one mechanism fragment against the shared context."""
    program = fragment.get("program")
    _fail(isinstance(program, Mapping), f"{mechanism}: missing program")
    _fail(program.get("event") in PHASE_EVENTS,
          f"{mechanism}: event {program.get('event')!r} not in "
          f"{PHASE_EVENTS}")
    schema = registry.schema(program.get("parameter_schema"))
    _fail(isinstance(program.get("operation_classes", []), list),
          f"{mechanism}: operation_classes must be a list")
    # engine_axes: capture axes the engine supplies at this program's
    # event point (relation programs bound on the consumer side, e.g.
    # the producer wgmma's tensor shape for accumulator_handoff.consumer);
    # the converter neither requires nor evaluates them
    engine_axes = program.get("engine_axes", [])
    _fail(isinstance(engine_axes, list) and all(
        a in axes and axes[a].get("owner") == "capture"
        for a in engine_axes),
        f"{mechanism}: engine_axes must list capture axes")

    resource_names = resource_names or set(resources)

    def check_parameter(parameter: Mapping[str, Any], label: str) -> None:
        fields = parameter.get("fields")
        _fail(isinstance(fields, Mapping) and fields,
              f"{label}: parameter needs fields")
        for name, fdef in fields.items():
            spec_field = schema.field(name)
            _fail(isinstance(fdef, Mapping),
                  f"{label}: field {name} must be an object")
            _check_model(fdef.get("model", {}), axes,
                         label=f"{label}.{name}")
            model = fdef.get("model", {})
            if spec_field.unit == "resource":
                values = []
                if model.get("kind") == "constant":
                    values = [model.get("value")]
                elif model.get("kind") == "discrete":
                    values = [p[1] for p in model.get("points", [])]
                    if "placeholder_default" in model:
                        values.append(model["placeholder_default"])
                _fail(values,
                      f"{label}.{name}: resource field model must be "
                      "constant or discrete")
                for value in values:
                    _fail(value in resource_names or value == "none",
                          f"{label}.{name}: unknown resource {value!r}")
            elif model.get("kind") == "constant":
                _fail(isinstance(model.get("value"), (int, float)),
                      f"{label}.{name}: constant value must be numeric")
            source_keys = fdef.get("sources", [])
            _fail(isinstance(source_keys, list),
                  f"{label}.{name}: sources must be a list")
            for key in source_keys:
                _fail(key in sources,
                      f"{label}.{name}: unknown source {key!r}")

    def visit(spec: Mapping[str, Any], label: str) -> None:
        if spec.get("unsupported"):
            _fail("parameter" not in spec,
                  f"{label}: unsupported leaf cannot carry a parameter")
            return
        if "parameter" in spec:
            check_parameter(spec["parameter"], label)
            return
        axis_name = spec.get("partition_axis")
        _fail(axis_name in axes,
              f"{label}: partition_axis {axis_name!r} undeclared")
        axis = axes[axis_name]
        domain_spec = spec.get("partition_domain")
        domain = _axis_domain(axis)
        if domain_spec is not None:
            if axis["kind"] == "int":
                _fail(isinstance(domain_spec, list) and len(domain_spec) == 2,
                      f"{label}: int partition_domain must be [lo, hi]")
                domain = (domain_spec[0], domain_spec[1])
            else:
                domain = set(domain_spec)
        children = spec.get("children", [])
        for child in children:
            classes = child.get("classes")
            if classes is not None:
                _fail(axis_name == "operation_class",
                      f"{label}: classes shorthand only under "
                      "operation_class partitions")
                _fail(isinstance(classes, list) and classes
                      and len(classes) == len(set(classes)),
                      f"{label}: classes needs a non-empty unique list")
                continue
            guard = child.get("guard", {})
            guard_axis = guard.get("axis", axis_name)
            _fail(guard_axis == axis_name,
                  f"{label}: child guard axis {guard_axis!r} does not match "
                  f"partition_axis {axis_name!r}")
            _fail(guard.get("operator") in ("eq", "range"),
                  f"{label}: bad guard operator {guard.get('operator')!r}")
        expanded = _expand_class_children(children)
        validate_guard_partition(expanded, axis, domain, label=label)
        for child in children:
            visit(child, f"{label}/{axis_name}")

    if "groups" in fragment:
        _fail("operation_class" in axes,
              f"{mechanism}: groups need an operation_class axis")
        covered: set = set()
        for class_list, spec in fragment["groups"].items():
            for op_class in class_list.split(","):
                op_class = op_class.strip()
                _fail(op_class not in covered,
                      f"{mechanism}: operation_class {op_class} grouped twice")
                covered.add(op_class)
            if "fields" in spec:
                check_parameter({"fields": spec["fields"]},
                                f"{mechanism}.groups[{class_list}]")
            elif "parameter" in spec:
                check_parameter(spec["parameter"],
                                f"{mechanism}.groups[{class_list}]")
            else:
                visit(spec, f"{mechanism}.groups[{class_list}]")
        declared = program.get("operation_classes", [])
        if declared:
            _fail(set(declared) == covered,
                  f"{mechanism}: groups cover {sorted(covered)} but program "
                  f"declares {sorted(declared)}")
    else:
        _fail(isinstance(fragment.get("root"), Mapping),
              f"{mechanism}: needs root or groups")
        visit(fragment["root"], f"{mechanism}.root")


def validate_sources(sources: Mapping[str, Any],
                     used_keys: set[str]) -> None:
    """Every source well-formed; every declared source referenced once."""
    for key, record in sources.items():
        _fail(isinstance(record, Mapping), f"source {key!r} must be an object")
        _fail(record.get("evidence") in EVIDENCE_LEVELS,
              f"source {key!r}: bad evidence {record.get('evidence')!r}")
        _fail(isinstance(record.get("description"), str),
              f"source {key!r}: needs a description")
        _fail(key in used_keys, f"source {key!r} is never referenced")


def validate_resources(resources: list[Mapping[str, Any]]) -> dict[str, int]:
    """Validate resources.json entries; return name -> generated id."""
    ids: dict[str, int] = {}
    for index, entry in enumerate(resources):
        name = entry.get("name")
        _fail(isinstance(name, str) and name and name not in ids,
              f"resources[{index}]: bad or duplicate name {name!r}")
        _fail(entry.get("scope") in ("smsp", "sm", "cluster"),
              f"resource {name}: bad scope {entry.get('scope')!r}")
        for key in ("copy_count", "max_inflight_per_copy",
                    "initiation_interval_cycles"):
            _fail(isinstance(entry.get(key), int) and entry[key] >= 0,
                  f"resource {name}: {key} must be a non-negative int")
        ids[name] = index
    return ids


def validate_generated_tree(tree: Mapping[str, Any]) -> None:
    """Check a generated tree's digest and internal counts, that every
    timing_row program names a known TimingKind, and that every capture
    axis has a registered axis function."""
    from ..job.timing_rows import TimingKind
    from ..convert.axes import AXIS_FUNCTIONS
    kinds = {k.name for k in TimingKind}
    for name, prog in tree.get("programs", {}).items():
        if prog.get("parameter_schema") != "timing_row":
            continue
        kind = prog.get("timing_kind")
        _fail(kind is not None,
              f"{name}: timing_row program missing timing_kind")
        _fail(kind in kinds,
              f"{name}: unknown timing_kind {kind}")
    for axis_name, axis in tree.get("axes", {}).items():
        _fail(axis.get("owner") != "capture"
              or axis_name in AXIS_FUNCTIONS,
              f"capture axis without axis function: {axis_name}")
    _fail(tree.get("schema_version") == "cyclescope.mechanism-tree.v1",
          "generated tree needs schema_version cyclescope.mechanism-tree.v1")
    _fail(tree.get("content_digest") == compute_content_digest(tree),
          "content_digest mismatch")
    leaf_count = sum(1 for n in tree.get("nodes", [])
                     if "parameter" in n or n.get("unsupported"))
    _fail(leaf_count == tree.get("leaf_count"), "leaf_count mismatch")


def validate_program_operation_classes(programs: Mapping[str, Any]) -> None:
    """Reject duplicated operation classes inside a single program.

    Note: the spec's stricter reading (no overlap between programs at one
    event) does not hold for the imported legacy trees, where several
    programs deliberately resolve on the same (event, operation_class) to
    fill different timing fields.  Cross-program overlap is therefore
    allowed; only a duplicate inside one program is an error.
    """
    for name, program in programs.items():
        classes = program.get("operation_classes", [])
        _fail(len(classes) == len(set(classes)),
              f"{name}: operation_class duplicated inside one program")


__all__ = [
    "ValidateError", "SourceError", "canonical_json",
    "compute_content_digest", "validate_fragment",
    "validate_guard_partition", "validate_resources",
    "validate_sources", "validate_generated_tree",
    "validate_program_operation_classes",
]
