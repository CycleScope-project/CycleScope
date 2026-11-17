import json
import statistics
import sys
from pathlib import Path

root, out = map(Path, sys.argv[1:3])
result = {"runs": {}}
for run in ("a", "b"):
    cells = {}
    for line in (root / f"run-{run}" / "store_scatter.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert "error" not in row and row["samples"] and row["n"] == len(row["samples"])
        cell = cells.setdefault(row["variant"], {"params": row["params"], "gap_medians": {}})
        assert str(row["u"]) not in cell["gap_medians"]
        cell["gap_medians"][str(row["u"])] = statistics.median(row["samples"])
    assert len(cells) == 26, len(cells)
    c0 = statistics.median(cells["none.s16.w1"]["gap_medians"][str(k)] for k in range(2, 25))
    for name, cell in cells.items():
        gaps = cell["gap_medians"]
        assert set(gaps) == {str(k) for k in range(25)}
        cell["early_issue"] = max(0, statistics.median(gaps[str(k)] for k in range(2, 5)) - (c0 - 1))
        cell["tail_gap"] = statistics.median(gaps[str(k)] for k in range(17, 25))
        if not name.startswith("none"):
            width, stride = cell["params"]["width_bytes"], cell["params"]["stride_bytes"]
            cell["sectors"] = len({(lane * stride + byte) // 32 for lane in range(32) for byte in range(width)})
            cell["lines"] = len({(lane * stride + byte) // 128 for lane in range(32) for byte in range(width)})
        print(run, name, "early", cell["early_issue"], "tail", cell["tail_gap"],
              "gaps", [gaps[str(k)] for k in range(1, 25)])
    result["runs"][run] = {"clock_gap": c0, "cells": cells}
out.write_text(json.dumps(result, indent=2) + "\n")
