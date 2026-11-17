"""Parameter schema registry (`configs/schema_registry.json`).

Each schema describes the fields one engine binding object
(``ParameterTarget``) accepts.  Fragments reference fields by name only;
the registry assigns numeric ``field_id`` values used to build
``field_mask`` in generated trees and the C++ header.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REGISTRY_SCHEMA_VERSION = "cyclescope.schema-registry.v1"

FIELD_TYPES = {"uint64", "float64"}

# Controlled unit vocabulary for leaf parameter fields (spec §4).
FIELD_UNITS = {
    "cycles", "ns", "bytes", "bytes/cycle", "bytes/ns", "count",
    "ratio", "enum", "boolean", "resource", "ops/ns",
}


class RegistryError(ValueError):
    """Malformed schema registry."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    field_id: int
    type: str
    unit: str


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    schema_id: int
    target: str
    fields: Mapping[str, FieldSpec]

    def field(self, name: str) -> FieldSpec:
        try:
            return self.fields[name]
        except KeyError:
            raise RegistryError(
                f"schema {self.name!r} has no field {name!r}") from None


@dataclass(frozen=True)
class Registry:
    schemas: Mapping[str, SchemaSpec]

    def schema(self, name: str) -> SchemaSpec:
        try:
            return self.schemas[name]
        except KeyError:
            raise RegistryError(f"unknown parameter schema {name!r}") from None


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(f"schema registry: {message}")


def load_registry(path: str | Path) -> Registry:
    """Load and validate `configs/schema_registry.json`."""
    raw = json.loads(Path(path).read_text())
    _fail(isinstance(raw, Mapping), "registry must be a JSON object")
    _fail(raw.get("schema_version") == REGISTRY_SCHEMA_VERSION,
          "schema_version must be cyclescope.schema-registry.v1")
    schemas_raw = raw.get("schemas")
    _fail(isinstance(schemas_raw, Mapping) and schemas_raw, "schemas must be a non-empty object")

    seen_ids: dict[int, str] = {}
    schemas: dict[str, SchemaSpec] = {}
    for name, entry in schemas_raw.items():
        _fail(isinstance(entry, Mapping), f"schema {name!r} must be an object")
        schema_id = entry.get("id")
        _fail(isinstance(schema_id, int) and schema_id >= 0,
              f"schema {name!r} needs a non-negative integer id")
        if schema_id in seen_ids:
            raise RegistryError(
                f"schema id {schema_id} used by both "
                f"{seen_ids[schema_id]!r} and {name!r}")
        seen_ids[schema_id] = name
        _fail(isinstance(entry.get("target"), str) and entry["target"],
              f"schema {name!r} needs a target name")
        fields_raw = entry.get("fields")
        _fail(isinstance(fields_raw, Mapping) and fields_raw,
              f"schema {name!r} needs a non-empty fields object")
        seen_field_ids: set[int] = set()
        fields: dict[str, FieldSpec] = {}
        for field_name, fdef in fields_raw.items():
            _fail(isinstance(fdef, Mapping),
                  f"field {name}.{field_name} must be an object")
            field_id = fdef.get("id")
            _fail(isinstance(field_id, int) and 0 <= field_id < 64,
                  f"field {name}.{field_name} needs an id in [0, 64)")
            _fail(field_id not in seen_field_ids,
                  f"field id {field_id} repeated in schema {name!r}")
            seen_field_ids.add(field_id)
            _fail(fdef.get("type") in FIELD_TYPES,
                  f"field {name}.{field_name} has bad type {fdef.get('type')!r}")
            _fail(fdef.get("unit") in FIELD_UNITS,
                  f"field {name}.{field_name} has bad unit {fdef.get('unit')!r}")
            fields[field_name] = FieldSpec(
                name=field_name, field_id=field_id,
                type=fdef["type"], unit=fdef["unit"])
        schemas[name] = SchemaSpec(
            name=name, schema_id=schema_id, target=entry["target"],
            fields=fields)
    return Registry(schemas=schemas)
