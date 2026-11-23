"""instruction.warp_queue leaves and the disabled global store interval
(warp_queue protocol Q2 / Q5) in the generated device trees."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
R = {"STS": [4.0, 8.0, 10.0], "STG": [6.0, 8.0, 12.0], "LDS": [2.0],
     "LDG": [4.0], "SHFL.IDX": [4.0], "SHFL.BFLY": [2.0]}


SM_START = {"STS": [1.0, 4.0, 4.0], "STG": [2.0, 2.0, 2.0]}
RESULT = {"LDG": {"h100": 20.5, "b300": 20.5}}


def _nodes(device):
    tree = json.loads((ROOT / "configs" / "trees" / f"{device}.json")
                      .read_text())
    return {n["id"]: n for n in tree["nodes"]}


def _value(node, field):
    return node["parameter"]["fields"][field]["model"]["value"]


@pytest.mark.parametrize("device", ["h100", "b300"])
def test_warp_queue_leaves(device):
    nodes = _nodes(device)
    seen = 0
    for cls, rs in R.items():
        i = -1
        ids = sorted(i for i in nodes
                     if i.startswith(f"instruction.warp_queue.root/"
                                     f"operation_class={cls}")
                     and "parameter" in nodes[i])
        assert len(ids) == len(rs), (cls, ids)
        for node_id, r in zip(ids, rs):
            i += 1
            node = nodes[node_id]
            assert _value(node, "release_interval_cycles") == r, node_id
            assert _value(node, "queue_depth_count") == 10
            assert _value(node, "start_latency_cycles") == 5.5
            assert _value(node, "start_interval_ratio") == 0.5
            assert _value(node, "enabled") == 1
            assert _value(node, "is_global_entry") == int(
                cls in ("LDG", "STG"))
            assert _value(node, "has_chain_turn") == 0
            assert _value(node, "sm_depth_count") == 52
            assert _value(node, "sm_start_interval_cycles") == (
                (4.55 if (device == "b300" and cls == "STS" and i == 2) else SM_START[cls][i % 3]) if cls in SM_START else 1.0), node_id
            assert _value(node, "result_after_release_cycles") == (
                RESULT.get(cls, {}).get(device, RESULT.get(cls, 0.0))
                if cls in ("LDG",) else
                14.7 if cls in ("LDS", "SHFL.IDX", "SHFL.BFLY")
                else 0.0), node_id
            seen += 1
    assert seen == 10


@pytest.mark.parametrize("device", ["h100", "b300"])
def test_store_issue_interval_disabled(device):
    node = _nodes(device)["storage.global.store_issue.root"]
    assert _value(node, "enabled") == 0
