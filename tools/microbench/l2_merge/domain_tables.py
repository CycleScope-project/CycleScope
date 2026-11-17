#!/usr/bin/env python3
"""Slice tables by SM id for the l2_merge / ldgsts_loaded sharing-domain cells
(protocol-v2.json addendum "read_domain"), and the ratio each candidate
domain predicts for them.

    domain_tables.py <h100|b300> <sm_topology.json> <outdir>

Writes <outdir>/<cell>.tbl (`slices <n>` + one `<sm> <slice> <member>` line
per SM id; SMs with the same slice read the same data in lockstep) and
<outdir>/predictions.json; prints the prediction table.

Candidate domains (a read can be shared only inside one domain):
  pair  : SM 2k and 2k + 1 (installed rule, memory.l2_merge.domain_sms = 2)
  block : the groups in which a full grid of CTAs is dispatched (recorded
          SM ids of the cp.async placement cells): H100 ids 0..63 in 8
          blocks of 4 pairs {b, b+16, b+32, b+48}, ids 64..119 in 7 blocks
          of 4 pairs with id stride 14, the pairs 120 and 122 after the last
          two of those, 124..131 one block; B300 ids 0..79 in 8 blocks of 5
          pairs, 80..135 in 7 blocks of 4 pairs with stride 14, the pairs
          136 / 138 / 140 after the last three, 142..147 one block
  gpc   : sm_topology placement_domain (two blocks each on most ids);
          ids without one keep their pair
Predicted ratio of a cell = sum over (slice, domain) of ceil(members /
N) / SM count, N = requests one read can serve (installed 4 on h100, 3 on
b300); cells whose shared slices hold two SMs do not depend on N.
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

NONE = 0xFFFFFFFF
MASKS = (0, 1, 2, 4, 8, 16, 32, 64, 3, 7, 15, 31, 63, 255)


def block(dev: str, s: int) -> int:
    if dev == "h100":
        if s < 64: return (s >> 1) & 7
        if s < 120: return 8 + ((s - 64) % 14) // 2
        if s < 124: return 13 + (s - 120) // 2
        return 15
    if s < 80: return (s >> 1) & 7
    if s < 136: return 8 + ((s - 80) % 14) // 2
    if s < 142: return 12 + (s - 136) // 2
    return 15


def domains(dev: str, topo: dict) -> dict:
    n = topo["sm_count"]
    pd = {r["sm"]: r["placement_domain"] for r in topo["rows"]}
    return {
        "pair": [s >> 1 for s in range(n)],
        "block": [block(dev, s) for s in range(n)],
        "gpc": [pd[s] if pd[s] != NONE else 1000 + (s >> 1) for s in range(n)],
    }


def groups_to_table(n: int, groups: list) -> list:
    """groups: lists of SM ids sharing a slice; SMs not listed get their own slice."""
    rows = [None] * n
    sl = 0
    for g in groups:
        for m, s in enumerate(g):
            assert rows[s] is None, s
            rows[s] = (sl, m)
        sl += 1
    for s in range(n):
        if rows[s] is None:
            rows[s] = (sl, 0); sl += 1
    return rows


def cells(dev: str, topo: dict) -> dict:
    n = topo["sm_count"]
    dom = domains(dev, topo)
    out = {}
    for m in MASKS:
        g = defaultdict(list)
        for s in range(n): g[s & ~m].append(s)
        out[f"m{m}"] = groups_to_table(n, [v for v in g.values() if len(v) > 1])
    by_block = defaultdict(list)
    for s in range(n): by_block[dom["block"][s]].append(s)
    # halfx: inside every block the pairs are taken two at a time; SM e of the
    # first shares with SM e of the second (different pairs, same block)
    g = []
    for b in sorted(by_block):
        tp = sorted({s >> 1 for s in by_block[b]})
        for i in range(0, len(tp) - 1, 2):
            g += [[2 * tp[i], 2 * tp[i + 1]], [2 * tp[i] + 1, 2 * tp[i + 1] + 1]]
    out["halfx"] = groups_to_table(n, g)
    # gpcx: in every placement domain holding two blocks, SM i of the first
    # block shares with SM i of the second (same gpc, different blocks)
    by_gpc = defaultdict(lambda: defaultdict(list))
    for s in range(n): by_gpc[dom["gpc"][s]][dom["block"][s]].append(s)
    g = []
    for d in sorted(by_gpc):
        bl = sorted(by_gpc[d])
        if len(bl) == 2:
            g += [list(p) for p in zip(sorted(by_gpc[d][bl[0]]), sorted(by_gpc[d][bl[1]]))]
    out["gpcx"] = groups_to_table(n, g)
    # speca: the first block against the last one (is the block of the highest
    # ids the second half of the placement domain of SM 0)
    a = sorted(by_block[0]); z = sorted(by_block[15])
    out["speca"] = groups_to_table(n, [list(p) for p in zip(a, z)])
    out["blk"] = groups_to_table(n, [sorted(v) for v in by_block.values()])
    by = defaultdict(list)
    for s in range(n): by[dom["gpc"][s]].append(s)
    out["gpc"] = groups_to_table(n, [sorted(v) for v in by.values() if len(v) > 1])
    return out


IDLE = 65535   # member value of an SM whose CTA issues no loads


def mechanics_cells(topo: dict) -> dict:
    """Tables of the ldgsts_loaded merge_mechanics cells: SMs outside the named set are idle.
    The gpc is the topology table's gpc column (SMs without one keep their pair)."""
    n = topo["sm_count"]
    gpc = [r["gpc"] if r["gpc"] != NONE else 1000 + (r["sm"] >> 1) for r in sorted(topo["rows"], key=lambda r: r["sm"])]
    by = defaultdict(list)
    for s in range(n): by[gpc[s]].append(s)
    real = sorted(g for g in by if g < 1000)
    pairs = lambda sms: [[2 * t, 2 * t + 1] for t in sorted({s >> 1 for s in sms})]
    def table(active, groups):
        rows = groups_to_table(n, groups)
        return [(sl, m if s in active else IDLE) for s, (sl, m) in enumerate(rows)]
    g1 = set(by[real[1]]); g12 = g1 | set(by[real[2]])
    t1 = {s for g in real for s in pairs(by[g])[0]}
    t2 = {s for g in real for p in pairs(by[g])[:2] for s in p}
    every = set(range(n))
    out = {
        "all_m0": table(every, []), "all_m1": table(every, pairs(every)),
        "g1_m0": table(g1, []), "g1_m1": table(g1, pairs(g1)), "g1_all": table(g1, [sorted(g1)]),
        "t1_m0": table(t1, []), "t1_m1": table(t1, pairs(t1)),
        "g2_m0": table(g12, []), "t2_m0": table(t2, []),
    }
    # k<k>: in every gpc its first k SMs (pair by pair) read one slice, the rest idle
    for k in (2, 3, 4, 6, 8, 12, 16):
        gs = [sorted(by[g])[:k] for g in real]
        out[f"k{k}"] = table({s for g in gs for s in g}, gs)
    gs = [[pairs(by[g])[0][0], pairs(by[g])[1][0]] for g in real]
    out["k2x"] = table({s for g in gs for s in g}, gs)
    # x<d>: SM e of pair i with SM e of pair i + d of the same gpc (member 0 / 1), the rest idle
    first = {1: (0, 2, 4, 6), 2: (0, 1, 4, 5), 3: (0, 1, 2), 4: (0, 1, 2, 3)}
    for d, idx in first.items():
        gs = []
        for g in real:
            pp = pairs(by[g])
            for i in idx:
                if i + d < len(pp): gs += [[pp[i][0], pp[i + d][0]], [pp[i][1], pp[i + d][1]]]
        out[f"x{d}"] = table({s for g in gs for s in g}, gs)
    # every SM active: one pair per slice (m1) and SM e of a pair with SM e of the next pair of its gpc
    # (xp, member 0 / 1; xpr = the same slices with the members swapped)
    out["m0"] = table(every, []); out["m1"] = table(every, pairs(every))
    gs = []
    for g in real:
        pp = pairs(by[g])
        for i in range(0, len(pp) - 1, 2): gs += [[pp[i][0], pp[i + 1][0]], [pp[i][1], pp[i + 1][1]]]
    out["xp"] = table(every, gs); out["xpr"] = table(every, [g[::-1] for g in gs])
    return out


def predict(table: list, domain: list, cap: int) -> float:
    c = Counter((sl, domain[s]) for s, (sl, _) in enumerate(table))
    return sum(math.ceil(v / cap) for v in c.values()) / len(table)


def predict_from_slices(slices: list, smids: list, domain: list, cap: int) -> float:
    """Cells grouped by block index: slices[i] / smids[i] of CTA i as recorded."""
    c = Counter((sl, domain[s]) for sl, s in zip(slices, smids))
    return sum(math.ceil(v / cap) for v in c.values()) / len(slices)


def main() -> int:
    dev, topo_path, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    topo = json.loads(Path(topo_path).read_text())
    outdir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 4 and sys.argv[4] == "mechanics":
        for name, table in mechanics_cells(topo).items():
            with open(outdir / f"{name}.tbl", "w") as f:
                f.write(f"slices {max(sl for sl, _ in table) + 1}\n")
                for s, (sl, m) in enumerate(table): f.write(f"{s} {sl} {m}\n")
            act = [s for s, (_, m) in enumerate(table) if m != IDLE]
            sl = {table[s][0] for s in act}
            print("%-7s active SMs %3d slices among them %3d pair-only ratio %.3f" % (name, len(act), len(sl),
                  len({(table[s][0], s >> 1) for s in act}) / len(act)))
        return 0
    dom = domains(dev, topo)
    cap = {"h100": 4, "b300": 3}[dev]
    pred = {}
    print(f"{dev}: predicted ratio per cell (N = {cap}; in brackets N = 2 / 3 / 4)")
    print("%-6s %6s  %-26s %-26s %-26s" % ("cell", "slices", "pair", "block", "gpc"))
    for name, table in cells(dev, topo).items():
        ns = max(sl for sl, _ in table) + 1
        with open(outdir / f"{name}.tbl", "w") as f:
            f.write(f"slices {ns}\n")
            for s, (sl, m) in enumerate(table): f.write(f"{s} {sl} {m}\n")
        pred[name] = {d: {str(c): round(predict(table, dom[d], c), 4) for c in (2, 3, 4)} for d in dom}
        col = lambda d: "%.3f [%.3f %.3f %.3f]" % (pred[name][d][str(cap)], *(pred[name][d][str(c)] for c in (2, 3, 4)))
        print("%-6s %6d  %-26s %-26s %-26s" % (name, ns, col("pair"), col("block"), col("gpc")))
    (outdir / "predictions.json").write_text(json.dumps({"device": dev, "cap": cap, "cells": pred}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
