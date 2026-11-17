#!/usr/bin/env python3
"""install control_branch results into the device leaves
(protocol.json v1 install clause):

    install.py --model <model.json> --source-root configs/mechanism_tree \
        --device h100|b300 --run <h100-r1>

<arch>/leaves/control.branch.json gets
    control.branch.root/operation_class=BRA
with fields.control_envelope = discrete over [branch_taken] from the
measured taken (point 1) and nottaken (point 0) W=1 cells; uncond /
taken_dep / loop / W=4 are reported, not installed. Sources
control_branch.<device>.<run>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    model = json.loads(a.model.read_text())
    cells = model["cells"]
    src = f"control_branch.{a.device}.{a.run}"
    vals = {}
    for key, v in cells.items():
        if v["w"] != "1" or v.get("branch") is None:
            continue
        if v["status"] not in ("measured", "experimental"):
            continue
        vals[v["variant"]] = v["branch"]
    leaf = {
        "control.branch.root/operation_class=BRA": {
            "fields": {
                "control_envelope": {
                    "model": {
                        "kind": "discrete",
                        "axes": ["branch_taken"],
                        "points": [
                            [0, vals["nottaken"]],
                            [1, vals["taken"]],
                        ],
                    },
                    "sources": [src],
                },
                "enabled": {
                    "model": {"kind": "constant", "value": 1},
                    "sources": [src],
                },
            }
        }
    }
    path = (a.source_root /
            {"h100": "hopper", "b300": "blackwell"}[a.device] /
            "leaves" / "control.branch.json")
    path.write_text(json.dumps(leaf, indent=1) + "\n")
    print(f"installed {path}: nottaken={vals['nottaken']:.3f} "
          f"taken={vals['taken']:.3f} ({src})")


if __name__ == "__main__":
    main()
