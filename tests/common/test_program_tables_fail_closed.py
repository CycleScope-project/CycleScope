"""discrete PARM fail_closed emission: a multi-axis discrete model
without an explicit placeholder_default compiles to intercept=1 so the
engine rejects a point miss instead of silently taking 0."""

from __future__ import annotations

from cyclescope.io.fixed_tables import TableReader
from cyclescope.job import program_tables as pt, segment_format as sf


def _tree(models: list[dict]) -> dict:
    axes = {"ax": {"kind": "int", "owner": "capture", "unit": "count",
                   "minimum": 0, "maximum": 99}}
    nodes = [{"id": "r", "parent": None, "guard": None,
              "children": [f"l{i}" for i in range(len(models))],
              "parameter": None}]
    for i, model in enumerate(models):
        nodes.append({
            "id": f"l{i}", "parent": "r",
            "guard": {"axis": "ax", "operator": "eq", "value": i},
            "children": [],
            "parameter": {
                "status": "measured",
                "fields": {
                    "f": {"schema_id": 1, "field_id": 1,
                          "model": model, "sources": []},
                },
            },
        })
    return {
        "axes": axes,
        "nodes": nodes,
        "programs": {
            "p": {"event": "issue", "schema_id": 1, "root": "r",
                  "field_mask": 0, "operation_classes": []},
        },
    }


def _discrete(**kw) -> dict:
    return {"kind": "discrete", "axes": ["ax"],
            "points": [[0, 5.0]], **kw}


def test_discrete_fail_closed_intercept() -> None:
    models = [
        _discrete(fail_closed=True),            # explicit -> 1
        _discrete(placeholder_default=0.0),     # default present -> 0
        _discrete(),                            # neither -> 1 (reject)
        _discrete(placeholder_default=7.0,
                  fail_closed=False),           # explicit off -> 0
    ]
    payloads = pt.compile_program_tables(_tree(models))
    parms = TableReader(sf.PARM, payloads["PARM"])
    discrete = [parms.unpack(i) for i in range(len(parms))
                if parms.unpack(i)["model_kind"] == 1]
    assert [d["intercept"] for d in discrete] == [1.0, 0.0, 1.0, 0.0]
