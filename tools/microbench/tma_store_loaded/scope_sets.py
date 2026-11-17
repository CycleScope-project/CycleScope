#!/usr/bin/env python3
"""scope_sets.py <sm_topology.json> <out dir>: SM sets for the gpc_scope cells (protocol-v5 addendum).
g1k<k>: the first k SMs of the largest GPC (g1kall = all of it); ek<k>: the first k SMs of every GPC
(ekall = every SM that has a GPC).  Writes <name>.txt (SM ids, for store_dest --active-file) and <name>.tbl
(ldgsts_loaded --table: every active SM its own slice, the other SMs idle)."""
import json, sys, os, collections
topo = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
NONE = 4294967295; n = topo["sm_count"]
g = collections.defaultdict(list)
for r in topo["rows"]:
    if r["gpc"] != NONE: g[r["gpc"]].append(r["sm"])
for k in g: g[k].sort()
big = max(sorted(g), key=lambda k: len(g[k]))
sets = {}
for k in (2, 4, 8, 12): sets[f"g1k{k}"] = g[big][:k]
sets["g1kall"] = g[big]
for k in (1, 2, 4, 8): sets[f"ek{k}"] = sorted(s for v in g.values() for s in v[:k])
sets["ekall"] = sorted(s for v in g.values() for s in v)
for name, v in sets.items():
    open(os.path.join(out, name + ".txt"), "w").write(" ".join(map(str, v)) + "\n")
    act = set(v)
    with open(os.path.join(out, name + ".tbl"), "w") as f:
        f.write(f"slices {n}\n")
        for s in range(n): f.write(f"{s} {s} {0 if s in act else 65535}\n")
    print(name, len(v), "gpcs", len({k for k in g for s in v if s in g[k]}))
