"""Build a generated mechanism tree (`configs/trees/<device>.json`).

Reads source fragments via :mod:`cyclescope.tree.source`, validates them via
:mod:`cyclescope.tree.validate`, assigns resource ids / field masks / node ids
and writes the flat generated tree consumed by the job builder and engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .registry import Registry, load_registry
from .source import SourceTree, expand_fragment, load_source_tree
from .validate import (
    compute_content_digest, validate_fragment,
    validate_program_operation_classes, validate_resources,
    validate_sources,
)

GENERATED_SCHEMA_VERSION = "cyclescope.mechanism-tree.v1"


def _derive_status(field_sources: list[list[str]],
                   sources: Mapping[str, Any]) -> str:
    evidence = [
        sources[key]["evidence"]
        for keys in field_sources for key in keys
    ]
    if evidence and all(e == "structural" for e in evidence):
        return "structural"
    if evidence and all(e in ("measured", "structural") for e in evidence) \
            and "measured" in evidence:
        return "measured"
    if "experimental" in evidence:
        return "experimental"
    return "placeholder"


def assemble_tree(source_tree: SourceTree, registry: Registry) -> dict[str, Any]:
    """Assemble one generated tree dict (before digest) from sources."""
    resource_ids = validate_resources(source_tree.resources)

    nodes_out: list[dict[str, Any]] = []
    used_source_keys: set[str] = {
        key for entry in source_tree.resources
        for key in entry.get("sources", [])}
    programs: dict[str, Any] = {}
    used_schemas: set[str] = set()
    status_counts = {"measured": 0, "experimental": 0,
                     "structural": 0, "placeholder": 0, "unsupported": 0}
    leaf_count = 0

    for mechanism in sorted(source_tree.fragments):
        fragment = source_tree.fragments[mechanism]
        validate_fragment(mechanism, fragment, source_tree.axes,
                          source_tree.sources, resource_ids, registry)
        program_src = fragment["program"]
        schema = registry.schema(program_src["parameter_schema"])
        used_schemas.add(schema.name)
        expanded = expand_fragment(mechanism, fragment)
        field_mask = 0
        id_map: dict[str, dict[str, Any]] = {}
        for node in expanded:
            out: dict[str, Any] = {
                "id": node["id"],
                "parent": node["parent"],
                "guard": node["guard"],
                "children": list(node["children"]),
            }
            if node["partition_axis"] is not None:
                out["partition_axis"] = node["partition_axis"]
                out["partition_domain"] = node["partition_domain"]
            if node["unsupported"]:
                out["unsupported"] = True
                leaf_count += 1
                status_counts["unsupported"] += 1
            parameter = node["parameter"]
            if parameter is not None:
                leaf_count += 1
                fields_out: dict[str, Any] = {}
                field_sources: list[list[str]] = []
                for field_name, fdef in sorted(parameter["fields"].items()):
                    spec_field = schema.field(field_name)
                    field_mask |= 1 << spec_field.field_id
                    model = dict(fdef["model"])
                    if spec_field.unit == "resource":
                        def resolve_resource(value: Any) -> Any:
                            return (0xFFFFFFFF if value == "none"
                                    else resource_ids[value])
                        if model["kind"] == "constant":
                            model["value"] = resolve_resource(model["value"])
                        elif model["kind"] == "discrete":
                            model["points"] = [
                                [p[0], resolve_resource(p[1])]
                                for p in model["points"]]
                            if "placeholder_default" in model:
                                model["placeholder_default"] = (
                                    resolve_resource(
                                        model["placeholder_default"]))
                    fsrc = list(fdef.get("sources", []))
                    used_source_keys.update(fsrc)
                    field_sources.append(fsrc)
                    fields_out[field_name] = {
                        "schema_id": schema.schema_id,
                        "field_id": spec_field.field_id,
                        "type": spec_field.type,
                        "unit": spec_field.unit,
                        "model": model,
                        "sources": fsrc,
                    }
                status = _derive_status(field_sources, source_tree.sources)
                status_counts[status] += 1
                out["parameter"] = {
                    "scope": parameter.get("scope", "instruction"),
                    "status": status,
                    "fields": fields_out,
                }
            nodes_out.append(out)
            id_map[node["id"]] = out

        program: dict[str, Any] = {
            "event": program_src["event"],
            "parameter_schema": schema.name,
            "schema_id": schema.schema_id,
            "field_mask": field_mask,
            "operation_classes": list(
                program_src.get("operation_classes", [])),
            "root": f"{mechanism}.root",
        }
        if "target_index" in program_src:
            program["target_index"] = program_src["target_index"]
        if "target" in program_src:
            program["target"] = program_src["target"]
        if "timing_kind" in program_src:
            program["timing_kind"] = program_src["timing_kind"]
        if "engine_axes" in program_src:
            program["engine_axes"] = list(program_src["engine_axes"])
        programs[mechanism] = program

    validate_program_operation_classes(programs)
    validate_sources(source_tree.sources, used_source_keys)

    resources_out = []
    for index, entry in enumerate(source_tree.resources):
        resource = {
            "id": index,
            "name": entry["name"],
            "scope": entry["scope"],
            "copy_count": entry["copy_count"],
            "max_inflight_per_copy": entry["max_inflight_per_copy"],
            "initiation_interval_cycles": entry["initiation_interval_cycles"],
        }
        if entry.get("sources"):
            resource["sources"] = list(entry["sources"])
        resources_out.append(resource)

    schemas_out = {
        name: {
            "id": registry.schemas[name].schema_id,
            "fields": {
                fname: {"id": f.field_id, "type": f.type, "unit": f.unit}
                for fname, f in registry.schemas[name].fields.items()
            },
        }
        for name in sorted(used_schemas)
    }

    device = source_tree.device
    tree: dict[str, Any] = {
        "schema_version": GENERATED_SCHEMA_VERSION,
        "device": device["device"],
        "architecture": device["architecture"],
        "sm_arch": device["sm_arch"],
        "smsps_per_sm": device["smsps_per_sm"],
        "axes": source_tree.axes,
        "resources": resources_out,
        "schemas": schemas_out,
        "sources": source_tree.sources,
        "programs": programs,
        "nodes": nodes_out,
        "leaf_count": leaf_count,
        "status_counts": {k: v for k, v in status_counts.items() if v},
    }
    tree["content_digest"] = compute_content_digest(tree)
    return tree


def build_tree(source_root: str | Path, arch: str,
               registry_path: str | Path) -> dict[str, Any]:
    source_tree = load_source_tree(source_root, arch)
    registry = load_registry(registry_path)
    return assemble_tree(source_tree, registry)


def write_tree(tree: Mapping[str, Any], out_dir: str | Path) -> Path:
    out_path = Path(out_dir) / f"{tree['device']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tree, indent=1, sort_keys=True) + "\n")
    return out_path


def emit_cpp_header(registry: Registry, path: str | Path) -> Path:
    """Write `cpp/core/generated/parameter_schema.hpp` from the registry."""
    def const_name(name: str) -> str:
        out = []
        for ch in name:
            out.append("_" if not ch.isalnum() else ch.upper())
        return "".join(out)

    lines = [
        "// Generated by `cyclescope tree build --emit-cpp`. Do not edit.",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace cyclescope::core {",
        "",
        "enum class ParameterSchema : std::uint32_t {",
    ]
    for name in sorted(registry.schemas,
                       key=lambda n: registry.schemas[n].schema_id):
        schema = registry.schemas[name]
        lines.append(f"  {const_name(name)} = {schema.schema_id},")
    lines += ["};", ""]
    for name in sorted(registry.schemas,
                       key=lambda n: registry.schemas[n].schema_id):
        schema = registry.schemas[name]
        lines.append(f"namespace parameter_fields::{const_name(name)} {{")
        for fname in sorted(schema.fields,
                            key=lambda f: schema.fields[f].field_id):
            spec = schema.fields[fname]
            lines.append(
                f"inline constexpr std::uint32_t {const_name(fname)}"
                f" = {spec.field_id};")
        lines.append("}")
        lines.append("")
    # Per-field value conversion derived from the registry `unit`:
    # code 1 = ns -> cycles (x reference GHz);
    # code 2 = per-ns rate -> per cycle (/ reference GHz).
    conv = {}
    for name in sorted(registry.schemas,
                       key=lambda n: registry.schemas[n].schema_id):
        schema = registry.schemas[name]
        for fname, spec in schema.fields.items():
            unit = spec.unit or ""
            if unit == "ns":
                code = 1
            elif unit.endswith("/ns"):
                code = 2
            else:
                continue
            conv.setdefault(schema.schema_id, {}).setdefault(
                code, []).append(spec.field_id)
    lines += [
        "// PARM value conversion by registry field unit: 0 none,",
        "// 1 = ns -> cycles (x GHz), 2 = per-ns rate -> per cycle",
        "// (/ GHz).  Reference clock = LNCH.reference_sm_clock_khz.",
        "inline constexpr std::uint32_t field_unit_conversion(",
        "    std::uint32_t schema, std::uint32_t field) {",
        "  switch (schema) {",
    ]
    for sid, codes in conv.items():
        lines.append(f"    case {sid}:")
        lines.append("      switch (field) {")
        for code, fids in sorted(codes.items()):
            cases = " ".join(f"case {f}:" for f in sorted(fids))
            lines.append(f"        {cases} return {code};")
        lines.append("        default: return 0;")
        lines.append("      }")
    lines += [
        "    default: return 0;",
        "  }",
        "}",
        "",
        "}  // namespace cyclescope::core",
        ""]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def emit_python_schema(registry: Registry, path: str | Path) -> Path:
    """Write `python/cyclescope/tree/generated_schema.py` (name -> id tables)."""
    payload = {
        "schemas": {
            name: {
                "id": schema.schema_id,
                "fields": {fname: f.field_id
                           for fname, f in schema.fields.items()},
            }
            for name, schema in sorted(registry.schemas.items())
        }
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        '"""Generated by `cyclescope tree build --emit-cpp`. Do not edit."""\n\n'
        f"SCHEMAS = {json.dumps(payload['schemas'], indent=1, sort_keys=True)}\n")
    return out
