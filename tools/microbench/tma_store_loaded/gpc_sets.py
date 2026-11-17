#!/usr/bin/env python3
"""gpc_sets.py <sm_topology.json> <out dir>: SM-id sets for the store_dest placement cells (protocol-v5 addendum).
one     : every SM of the largest GPC                    spread  : the same number, taken evenly from all GPCs
halfgpc : every SM of whole GPCs, about half the SMs      halfeach: the same number, taken evenly from all GPCs
all     : every SM"""
import json, sys, os, collections
rows = json.load(open(sys.argv[1]))["rows"]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
NONE = 4294967295
g = collections.defaultdict(list)
for r in rows:
    if r["gpc"] != NONE: g[r["gpc"]].append(r["sm"])
for k in g: g[k].sort()
def evenly(n):
    pick, i = [], 0
    while len(pick) < n:
        for k in sorted(g):
            if i < len(g[k]) and len(pick) < n: pick.append(g[k][i])
        i += 1
    return sorted(pick)
big = max(sorted(g), key=lambda k: len(g[k]))
order = sorted(g, key=lambda k: (-len(g[k]), k))
half, tot = [], sum(len(v) for v in g.values())
for k in sorted(g):
    if k != big and len(half) + len(g[k]) <= tot // 2 + 2 and len([x for x in half]) < tot // 2: half += g[k]
sets = {"one": g[big], "spread": evenly(len(g[big])), "halfgpc": sorted(half), "halfeach": evenly(len(half)),
        "all": sorted(r["sm"] for r in rows)}
for name, v in sets.items():
    open(os.path.join(out, name + ".txt"), "w").write(" ".join(map(str, v)) + "\n")
    c = collections.Counter(next((k for k in g if s in g[k]), -1) for s in v)
    print(name, len(v), "per gpc", dict(sorted(c.items())))
