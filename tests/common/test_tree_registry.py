"""Registry loading and field lookup tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyclescope.tree.registry import RegistryError, load_registry

MINI_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "data" / "mechanism_tree" / "mini" / "registry.json")


def test_load_registry_and_field_lookup() -> None:
    registry = load_registry(MINI_REGISTRY)
    schema = registry.schema("instruction_template")
    assert schema.schema_id == 1
    field = schema.field("latency_cycles")
    assert (field.field_id, field.type, field.unit) == (1, "uint64", "cycles")
    with pytest.raises(RegistryError, match="no field"):
        schema.field("missing_field")
    with pytest.raises(RegistryError, match="unknown parameter schema"):
        registry.schema("missing_schema")


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    bad = {
        "schema_version": "cyclescope.schema-registry.v1",
        "schemas": {
            "a": {"id": 1, "target": "T", "fields": {
                "f": {"id": 0, "type": "uint64", "unit": "cycles"}}},
            "b": {"id": 1, "target": "T", "fields": {
                "g": {"id": 0, "type": "uint64", "unit": "cycles"}}},
        },
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(RegistryError, match="schema id 1"):
        load_registry(path)


def test_registry_rejects_bad_unit(tmp_path: Path) -> None:
    bad = {
        "schema_version": "cyclescope.schema-registry.v1",
        "schemas": {"a": {"id": 1, "target": "T", "fields": {
            "f": {"id": 0, "type": "uint64", "unit": "furlongs"}}}},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(RegistryError, match="bad unit"):
        load_registry(path)
