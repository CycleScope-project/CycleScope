#!/usr/bin/env python3
"""gpc_classes.py : per device, the two refill-gap classes in ns from the spin cells of dispatch_refill runs 1, 2, 4
(per-SM median steady gap, generations >= 4; class = topology gpc; slow gpcs = gpc median above the midpoint), and a
replay of the per-slot rule with the class value of each SM (M4) on the run-4 cells."""
import os, sys, json, heapq, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as M
REPO = Path(__file__).resolve().parents[4]
A = (sys.argv[1] if len(sys.argv) > 1
     else os.environ["CYCLESCOPE_ARTIFACT_ROOT"])
CELLS = [("r1", "spin.t128.o0.w12.h3000.j300"), ("r1", "spin.t128.o0.w12.h6000.j300"), ("r2", "spin.t128.o0.w6.h9000.j150"),
         ("r4", "p.spin.a"), ("r4", "p.spin.b"), ("r4", "g.h6000")]
def replay_m4(R, sm_order, occ, slot_of_sm, serial, pre_units):
    """M3 with a slot constant per SM; first wave on the SMs in sm_order (round robin)."""
    n = len(R); n_sm = len(sm_order); first = min(n, n_sm * occ)
    ev = []; free = collections.defaultdict(int); avail = []; q = first; busy = 0.0; end = 0.0
    for i in range(first):
        heapq.heappush(ev, (float(R[i]), 0, sm_order[i % n_sm], i))
    while ev:
        t, kind, s, i = heapq.heappop(ev)
        if kind == 0:
            end = max(end, t)
            if pre_units[i] > 0: busy = max(t, busy) + pre_units[i] * serial
            heapq.heappush(ev, (t + slot_of_sm[s], 1, s, i)); continue
        free[s] += 1; heapq.heappush(avail, s)
        while q < n and avail:
            s2 = heapq.heappop(avail)
            if free[s2] <= 0: continue
            free[s2] -= 1; e = max(t, busy); busy = e + serial
            heapq.heappush(ev, (e + float(R[q]), 0, s2, q)); q += 1
    return end
out = {}
for dev, serial_ns, units in (("b300", 0.513613, 1.0), ("h100", 0.617, 0.95)):
    topo = {r["sm"]: r["gpc"] for r in json.load(open(REPO / "configs" / "hardware" / dev / "sm_topology.json"))["rows"]}
    fast_ns, slow_ns, rows = [], [], []
    for run, cell in CELLS:
        meta, L = M.load(f"{A}/fg-line-cold/refill/{dev}-{run}/cells/{cell}.txt")
        f = M.cycles_per_ns({k: v for k, v in L.items() if k > 0}); occ = int(meta["occupancy_api"])
        per = collections.defaultdict(list)
        for la, a in L.items():
            if la == 0: continue
            for s, g in M.sm_steady_gap(a, occ, 3).items(): per[s].append(g)
        per = {s: float(np.median(v)) for s, v in per.items()}
        bg = collections.defaultdict(list)
        for s, g in per.items(): bg[topo[s]].append(g)
        gm = {g: float(np.median(v)) for g, v in bg.items()}
        cut = (min(gm.values()) + max(gm.values())) / 2
        slow = sorted(g for g, v in gm.items() if v > cut)
        fs = [per[s] for s in per if topo[s] not in slow]; ss = [per[s] for s in per if topo[s] in slow]
        fast_ns.append(np.median(fs) / f); slow_ns.append(np.median(ss) / f)
        rows.append((run, cell, round(f, 4), slow, len(fs), round(np.median(fs)), round(np.median(fs) / f, 1), len(ss), round(np.median(ss)), round(np.median(ss) / f, 1),
                     round(np.mean(list(per.values())) / f, 1)))
        print(dev, *rows[-1])
    F, S = float(np.mean(fast_ns)), float(np.mean(slow_ns))
    print(dev, "class values: fast %.1f ns (range %.1f-%.1f), slow %.1f ns (range %.1f-%.1f)" % (F, min(fast_ns), max(fast_ns), S, min(slow_ns), max(slow_ns)))
    out[dev] = {"fast_ns": F, "slow_ns": S, "rows": rows}
    # M4 replay on run-4 cells (slow set of that card)
    m4 = json.load(open(f"{A}/fg-line-cold/refill-model/{dev}-r4/model4.json")); f = m4["cycles_per_ns"]
    slow4 = rows[3][3]
    res = []
    for cell in ("p.spin.a", "p.rms", "p.cpasync", "g.none", "g.c500", "g.c1000", "g.h4500", "g.h6000", "g.t256.none", "g.t256.pdl", "g.t32.none", "g.t32.pdl", "g.t512.none", "g.t512.pdl", "g.w1", "g.early"):
        meta, L = M.load(f"{A}/fg-line-cold/refill/{dev}-r4/cells/{cell}.txt"); a = L[2]
        R = (a[:, 6] - a[:, 4]).astype(float); occ = int(meta["occupancy_api"]); n_sm = int(meta["sms"])
        order = [int(x) for x in a[:n_sm, 1]]
        slot = {s: (S if topo[s] in slow4 else F) * f for s in topo}
        pm = M.pre_mask(len(a), meta) * units
        end = replay_m4(R, order, occ, slot, serial_ns * f, pm)
        res.append("%s %.3f" % (cell, M.span_aligned(a, f) / end))
    print(dev, "span / M4 replay (launch 2):", "  ".join(res))
json.dump(out, open(A + "/fg-line-cold/refill-model/gpc_classes.json", "w"), indent=1)
