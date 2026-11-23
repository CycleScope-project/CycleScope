"""Pin the engine's stall-ledger unit-name table to
configs/mechanism_tree/common/resources.json.

cpp/core/simulator.cpp's kStallUnitNames maps resource ids to the
names of common/resources.json (the shared ids; arch-only units fall
back to the numeric id).  The table is a second copy by necessity —
this test keeps it honest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES = (
    REPO_ROOT / "configs" / "mechanism_tree" / "common" / "resources.json"
)
SIMULATOR = REPO_ROOT / "cpp" / "core" / "simulator.cpp"


def test_stall_unit_names_match_resources_json() -> None:
    names = [r["name"] for r in json.loads(RESOURCES.read_text())]
    src = SIMULATOR.read_text()
    m = re.search(
        r"kStallUnitNames\[\]\s*=\s*\{(.*?)\};", src, re.S)
    assert m, "kStallUnitNames table not found in simulator.cpp"
    table = re.findall(r'"([^"]+)"', m.group(1))
    assert table == names, (
        "kStallUnitNames drifted from common/resources.json: "
        f"table={table} resources={names}")
