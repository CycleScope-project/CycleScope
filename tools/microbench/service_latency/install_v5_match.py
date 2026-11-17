#!/usr/bin/env python3
"""MATCH leaf partition (protocol-v5 run v5r2): rebuild
instruction.service .../operation_class=MATCH as 32 children guarded
eq on match_distinct_values, per the run's match spec.

    python3 install_v5_match.py <repo root>
"""
import json
import sys
from pathlib import Path

HP_SRC = "service_latency.h100.v5r2.match"
BW_SRC = "service_latency.b300.v5r2.match"
BW_SRC_E = "service_latency.b300.v5r2.match.experimental"

HP_R_A, HP_R_B = 16.95982142857143, 11.0
BW_R_A, BW_R_B = 16.9225079957356, 10.9976012793177
HP_KNOTS = [(1, 1.0), (2, 1.0), (4, 1.25), (8, 2.0939), (16, 4.0870),
            (32, 8.0944)]
BW_KNOTS = [(1, 1.0), (2, 1.0), (4, 1.2504), (8, 2.0815), (16, 3.8152)]


def interp(knots, d):
    """Linear interpolation between knots; clamped outside."""
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    if d <= xs[0]:
        return ys[0]
    if d >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if d <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (d - x0) / (x1 - x0)
    return ys[-1]


def const(v):
    return {"kind": "constant", "value": v}


def child(fields, d, result, interval, rsrcs, isrcs):
    f = {k: dict(v) for k, v in fields.items()}
    f["result_latency_cycles"] = {
        "model": const(round(result, 4)), "sources": list(rsrcs)}
    f["completion_latency_cycles"] = {
        "model": const(round(result, 4)), "sources": list(rsrcs)}
    f["issue_interval_cycles"] = {
        "model": const(round(interval, 4)), "sources": list(isrcs)}
    return {"guard": {"axis": "match_distinct_values",
                      "operator": "range", "value": [d, d]},
            "parameter": {"fields": f}}


def install(path, ra, rb, knots, rsrcs, isrcs):
    p = Path(path)
    d = json.loads(p.read_text())
    key = ("instruction.service.root/predicated_on=1/"
           "operation_class=MATCH")
    node = d[key]
    fields = node["fields"]
    fields = node["fields"]
    del d[key]
    for dd in range(1, 33):
        c = child(fields, dd, ra + rb * dd, interp(knots, dd),
                  rsrcs, isrcs)
        d[key + f"/match_distinct_values={dd}-{dd}"] = c["parameter"]
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")


def main():
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    install(repo / "configs/mechanism_tree/hopper/leaves/"
                    "instruction.service.json",
            HP_R_A, HP_R_B, HP_KNOTS, [HP_SRC], [HP_SRC])
    install(repo / "configs/mechanism_tree/blackwell/leaves/"
                    "instruction.service.json",
            BW_R_A, BW_R_B, BW_KNOTS, [BW_SRC], [BW_SRC_E])
    print("installed 32-child MATCH partition (hopper + blackwell)")


if __name__ == "__main__":
    main()
