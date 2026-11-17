"""Committed generated files must match `cyclescope tree build --emit-cpp` output."""

from __future__ import annotations

from pathlib import Path

from cyclescope.tree.build import emit_cpp_header, emit_python_schema
from cyclescope.tree.registry import load_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "schema_registry.json"
CPP_HEADER = REPO_ROOT / "cpp" / "core" / "generated" / "parameter_schema.hpp"
PY_SCHEMA = REPO_ROOT / "python" / "cyclescope" / "tree" / "generated_schema.py"


def test_cpp_parameter_schema_header_is_current(tmp_path: Path) -> None:
    out = emit_cpp_header(load_registry(REGISTRY), tmp_path / "parameter_schema.hpp")
    assert out.read_text() == CPP_HEADER.read_text(), (
        "cpp/core/generated/parameter_schema.hpp is stale: "
        "run `cyclescope tree build --emit-cpp`")


def test_python_generated_schema_is_current(tmp_path: Path) -> None:
    out = emit_python_schema(load_registry(REGISTRY), tmp_path / "generated_schema.py")
    assert out.read_text() == PY_SCHEMA.read_text(), (
        "python/cyclescope/tree/generated_schema.py is stale: "
        "run `cyclescope tree build --emit-cpp`")
