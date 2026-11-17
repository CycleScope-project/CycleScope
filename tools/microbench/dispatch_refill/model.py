#!/usr/bin/env python3
"""dispatch_refill derivation (protocol.json).

  model.py cells  <run dir> [--out <dir>] [--serial-ns .. --slot-ns ..] per cell: measured quantities + replay of
                                                                   the present engine rule (M0) on the measured
                                                                   residencies -> <run dir>/model.json + table
  model.py cells ... --m2-slot-ns <ns>                              adds M2 (addendum 1): M1 with untruncated
                                                                   constants and the given slot constant
  model.py predict2 <sms> <cycles_per_ns> --slot-ns <M2 slot> ..    M2 on nominal residencies for the stage-2 cells
  model.py cells4 <run dir> --f <cycles per ns> --slot-ns .. --serial-ns .. --units <u> --topo <sm_topology.json>
                  --out <dir>                                     run 4 (addendum 4): M3 replay, SM gap classes
  model.py predict4 <sms> <cycles_per_ns> --slot-ns .. --serial-ns .. --units <u>   M3 at nominal residencies
  model.py predict <sms> <cycles_per_ns> [--serial-ns .. --slot-ns .. --base 280]
                                                                   M0 on nominal residencies for the protocol's
                                                                   cells (written into the protocol before data)
  model.py ncu <cell txt> <ncu raw csv>                            method check: ncu's per-warp stall-latency
                                                                   sum against the in-kernel residency

Measured per launch (launch 0 dropped), all on the SM's own clock64 unless noted:
  R            clk_out - clk_in of a CTA
  first wave   the first `occ` entries of an SM (occ = largest number of CTAs resident on any SM)
  fw_step      median spacing of consecutive first-wave entries on an SM
  gap_k        k-th refill entry of an SM minus the k-th exit of that SM (exits sorted): the time one slot of the
               SM stayed empty before the k-th refill; generation of the refill = 2 + k // occ
  span         GPU-wide: per SM (first entry offset by %globaltimer) + (last exit - first entry on its clock), max
  fill         sum R / (sms * occ * span)
  sweep        dispatch order = CTA index; SM rank = position of the SM in the first `sms` CTAs; a sweep ends when
               the rank decreases; CTAs per sweep after the first wave
  rate         refill entries per cycle GPU-wide (per-SM clocks aligned by the median of gt_in * f - clk_in)
M0 (cpp/core/simulator.cpp today): the first wave enters at 0; a slot is released at its CTA's exit; a dispatch
event fires `slot` cycles later and fills every slot released by then (also slots released less than `slot`
cycles ago) in SM order, consecutive entries at least `serial` cycles apart GPU-wide; both constants are
truncated to whole cycles as the engine does.
M1: as M0, but a slot can be refilled only `slot` cycles after its own release (the leaf's literal meaning).
"""
import collections
import glob
import heapq
import json
import os
import statistics as st
import sys

import numpy as np


def load(path):
    meta = {}
    by = collections.defaultdict(list)
    for line in open(path):
        if line.startswith("META "):
            for t in line[5:].split():
                if "=" in t:
                    k, v = t.split("=", 1)
                    meta[k] = v
        elif line.startswith("CTA "):
            d = dict(t.split("=") for t in line[4:].split())
            by[int(d["launch"])].append((int(d["cta"]), int(d["smid"]), int(d["warpid"]), int(d["gt_in"]),
                                         int(d["clk_in"]), int(d["gt_out"]), int(d["clk_out"])))
    return meta, {L: np.array(sorted(v), dtype=np.int64) for L, v in by.items()}


def cycles_per_ns(launches):
    r = []
    for a in launches.values():
        d = a[:, 5] - a[:, 3]
        m = d >= 1000
        if m.any():
            r.append(np.median((a[m, 6] - a[m, 4]) / d[m]))
    return float(np.median(r)) if r else None


def per_sm_gaps(sm, t_in, t_out, occ):
    """FIFO slot gaps per SM: list of (generation, gap, free slots on the SM right after the matched exit)."""
    out = []
    fw_steps, fw_spread = [], []
    for s in np.unique(sm):
        m = sm == s
        ins = np.sort(t_in[m]); outs = np.sort(t_out[m])
        if len(ins) > 1:
            k = min(occ, len(ins))
            if k > 1:
                fw_steps.append(float(np.median(np.diff(ins[:k])))); fw_spread.append(float(ins[k - 1] - ins[0]))
        for k in range(max(0, len(ins) - occ)):
            e = ins[occ + k]; x = outs[k]
            n_in = int(np.searchsorted(ins, x, side="right")); n_out = k + 1
            out.append((2 + k // occ, float(e - x), occ - (n_in - n_out)))
    return out, fw_steps, fw_spread


def replay_m0(R, n_sm, occ, slot, serial, per_slot=False):
    """Entry / exit times and SM of every CTA under the present engine rule (per_slot False: a dispatch event
    fills every slot released so far) or under M1 (per_slot True: a slot can be filled only `slot` cycles after
    its own release). R in dispatch order."""
    n = len(R); first = min(n, n_sm * occ)
    entry = np.zeros(n); sm_of = np.zeros(n, dtype=np.int64)
    ev = []  # (time, kind, sm): kind 0 release, 1 dispatch
    for i in range(first):
        sm_of[i] = i % n_sm
        heapq.heappush(ev, (float(R[i]), 0, i % n_sm))
    free = [0] * n_sm; avail = []  # heap of SMs with a free slot
    q = first; last = 0.0
    while ev:
        t, kind, s = heapq.heappop(ev)
        if kind == 0:
            if not per_slot:
                free[s] += 1; heapq.heappush(avail, s)
            heapq.heappush(ev, (t + slot, 1, s))
            continue
        if per_slot:
            free[s] += 1; heapq.heappush(avail, s)
        while q < n and avail:
            s2 = heapq.heappop(avail)
            if free[s2] <= 0:
                continue
            free[s2] -= 1
            e = max(t, last + serial); last = e
            entry[q] = e; sm_of[q] = s2
            heapq.heappush(ev, (e + float(R[q]), 0, s2))
            q += 1
    return entry, entry + np.asarray(R, dtype=float), sm_of


def replay_m3(R, n_sm, occ, slot, serial, pre_units, early=False):
    """M3 (addendum 4): M2 plus the dependents-launch notification. The GPU-wide serializer is busy `serial`
    cycles per CTA entry; a CTA with pre_units[i] > 0 (it executed PREEXIT) occupies it for pre_units[i] * serial
    more, starting at its exit (early: at its entry). R, pre_units in dispatch order."""
    n = len(R); first = min(n, n_sm * occ)
    entry = np.zeros(n); sm_of = np.zeros(n, dtype=np.int64)
    ev = []  # (time, kind, sm, cta): kind 0 exit, 1 slot placeable, 2 early notification
    for i in range(first):
        sm_of[i] = i % n_sm
        heapq.heappush(ev, (float(R[i]), 0, i % n_sm, i))
        if early and pre_units[i] > 0:
            heapq.heappush(ev, (0.0, 2, i % n_sm, i))
    free = [0] * n_sm; avail = []
    q = first; busy = 0.0
    while ev:
        t, kind, s, i = heapq.heappop(ev)
        if kind == 0:
            if pre_units[i] > 0 and not early:
                busy = max(t, busy) + pre_units[i] * serial
            heapq.heappush(ev, (t + slot, 1, s, i))
            continue
        if kind == 2:
            busy = max(t, busy) + pre_units[i] * serial
            continue
        free[s] += 1; heapq.heappush(avail, s)
        while q < n and avail:
            s2 = heapq.heappop(avail)
            if free[s2] <= 0:
                continue
            free[s2] -= 1
            e = max(t, busy); busy = e + serial
            entry[q] = e; sm_of[q] = s2
            heapq.heappush(ev, (e + float(R[q]), 0, s2, q))
            if early and pre_units[q] > 0:
                heapq.heappush(ev, (e, 2, s2, q))
            q += 1
    return entry, entry + np.asarray(R, dtype=float), sm_of


def hash32(x):
    x = np.asarray(x, dtype=np.uint64) & np.uint64(0xffffffff)
    m = np.uint64(0xffffffff)
    x ^= x >> np.uint64(16); x = (x * np.uint64(0x7feb352d)) & m
    x ^= x >> np.uint64(15); x = (x * np.uint64(0x846ca68b)) & m
    x ^= x >> np.uint64(16)
    return x


def pre_mask(n, meta):
    """Which CTAs execute PREEXIT, from the probe's META (same hash as refill.cu)."""
    if not (int(meta.get("pdl", 0)) & 2):
        return np.zeros(n, dtype=bool)
    h = hash32(np.arange(n, dtype=np.uint64) ^ np.uint64(0x9e3779b9)) % np.uint64(1000)
    return h < np.uint64(int(meta.get("pdl_cta", 1000)))


def span_aligned(a, f):
    cta, sm, wid, gi, ci, go, co = a.T
    lo, hi = np.inf, -np.inf
    for s in np.unique(sm):
        m = sm == s; off = np.median(gi[m] * f - ci[m])
        lo = min(lo, (ci[m] + off).min()); hi = max(hi, (co[m] + off).max())
    return hi - lo


def sm_steady_gap(a, occ, skip_gen):
    cta, sm, wid, gi, ci, go, co = a.T; out = {}
    for s in np.unique(sm):
        m = sm == s; ins = np.sort(ci[m]); outs = np.sort(co[m])
        g = [ins[occ + k] - outs[k] for k in range(skip_gen * occ, len(ins) - occ)]
        if g:
            out[int(s)] = float(np.median(g))
    return out


def cmd_cells4(run, f, slot_ns, serial_ns, units, topo_path, outdir):
    """Run-4 table: span, M2 (no notification) and M3 replays on the measured residencies, largest refill rate,
    two-class split of the per-SM steady gap and the per-gpc medians."""
    topo = {r["sm"]: r["gpc"] for r in json.load(open(topo_path))["rows"]}
    print("cell\tR\tspan\tcyc/CTA\trate_max\tspan/m2\tspan/m3\tfast n\tfast gap\tslow n\tslow gap\tslow gpcs\tgpc medians")
    res = {}
    for p in sorted(glob.glob(os.path.join(run, "cells", "*.txt")), key=os.path.getmtime):
        meta, L = load(p)
        if "sms" not in meta:
            continue
        name = os.path.basename(p)[:-4]; n_sm = int(meta["sms"]); occ = int(meta["occupancy_api"])
        rows = []
        for la in sorted(L):
            if la == 0:
                continue
            a = L[la]; R = (a[:, 6] - a[:, 4]).astype(float); sp = span_aligned(a, f)
            pm = pre_mask(len(a), meta); early = int(meta.get("pdl_early", 0)) == 1
            _, x2, _ = replay_m0(R, n_sm, occ, slot_ns * f, serial_ns * f, per_slot=True)
            _, x3, _ = replay_m3(R, n_sm, occ, slot_ns * f, serial_ns * f, pm * units, early)
            cta, sm, wid, gi, ci, go, co = a.T
            t = np.zeros(len(a))
            for s_ in np.unique(sm):
                m = sm == s_; t[m] = ci[m] + np.median(gi[m] * f - ci[m])
            ref = np.sort(t[n_sm * occ:]); w = 512.0
            rate = float((np.searchsorted(ref, ref + w) - np.arange(len(ref))).max() / w) if len(ref) > 600 else float("nan")
            rows.append((np.median(R), sp, sp / len(a), rate, sp / x2.max(), sp / x3.max(), sm_steady_gap(a, occ, 2)))
        md = lambda k: float(np.median([r[k] for r in rows]))
        per = {s_: float(np.median([r[6][s_] for r in rows if s_ in r[6]])) for s_ in rows[0][6]}
        v = np.array(sorted(per.values())); cut = (v[0] + v[-1]) / 2 if v[-1] - v[0] > 80 else v[-1] + 1
        fast = [x for x in per.values() if x <= cut]; slow = [x for x in per.values() if x > cut]
        bg = collections.defaultdict(list)
        for s_, x in per.items():
            bg[topo.get(s_, -1)].append(x)
        gm = {g: float(np.median(x)) for g, x in sorted(bg.items())}
        slow_g = [g for g, x in gm.items() if x > cut]
        res[name] = {"R": md(0), "span": md(1), "rate_max": md(3), "span_over_m2": md(4), "span_over_m3": md(5),
                     "fast": [len(fast), float(np.median(fast)) if fast else None],
                     "slow": [len(slow), float(np.median(slow)) if slow else None], "gpc_median": gm, "per_sm": per}
        print("%s\t%.0f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%s\t%d\t%s\t%s\t%s" % (
            name, md(0), md(1), md(2), md(3), md(4), md(5), len(fast), "%.0f" % np.median(fast) if fast else "-",
            len(slow), "%.0f" % np.median(slow) if slow else "-", ",".join(str(g) if g < 1000 else "x" for g in slow_g),
            " ".join("%.0f" % x for x in gm.values())))
    os.makedirs(outdir, exist_ok=True)
    json.dump({"cycles_per_ns": f, "slot_ns": slot_ns, "serial_ns": serial_ns, "units": units, "cells": res},
              open(os.path.join(outdir, "model4.json"), "w"), indent=1)


def cmd_predict4(n_sm, f, slot_ns, serial_ns, units, base):
    """M3 on nominal residencies for the run-4 cells G (spin, jitter 150)."""
    rng = np.random.default_rng(1)
    print("M3 slot %.1f serial %.3f cycles, %.2f extra units per PREEXIT CTA" % (slot_ns * f, serial_ns * f, units))
    cells = [("t128 w6 h2200 pdl-cta %d" % c, 16, 6, 2200, c / 1000.0, False) for c in (0, 250, 500, 750, 1000)]
    cells += [("t128 w6 h2200 early", 16, 6, 2200, 1.0, True), ("t128 w6 h4500 pdl", 16, 6, 4500, 1.0, False),
              ("t128 w6 h6000 pdl", 16, 6, 6000, 1.0, False), ("t256 w12 h100 none", 8, 12, 800, 0.0, False),
              ("t256 w12 h100 pdl", 8, 12, 800, 1.0, False), ("t32 w6 h2200 none", 32, 6, 2200, 0.0, False),
              ("t32 w6 h2200 pdl", 32, 6, 2200, 1.0, False)]
    for name, occ, waves, hold, p, early in cells:
        n = n_sm * occ * waves
        R = base + hold * (1 + 0.15 * rng.uniform(-1, 1, n))
        pm = (rng.uniform(0, 1, n) < p).astype(float) * units
        _, x, _ = replay_m3(R, n_sm, occ, slot_ns * f, serial_ns * f, pm, early)
        print("%-28s span %6.0f cyc/CTA %.3f" % (name, x.max(), x.max() / n))


def summarize_gaps(g):
    if not g:
        return {"refills": 0}
    a = np.array([x[1] for x in g]); gen = np.array([x[0] for x in g]); fr = np.array([x[2] for x in g])
    o = {"refills": int(len(a)), "gap_mean": float(a.mean()), "gap_median": float(np.median(a)),
         "gap_p10": float(np.percentile(a, 10)), "gap_p90": float(np.percentile(a, 90)),
         "gap_negative": int((a < 0).sum()), "by_generation": {}, "by_free_slots_after_exit": {}}
    last = int(gen.max())
    for name, m in (("2", gen == 2), ("3", gen == 3), ("4+", (gen >= 4) & (gen < last)), ("last", gen == last)):
        if m.any():
            o["by_generation"][name] = {"n": int(m.sum()), "mean": float(a[m].mean()), "median": float(np.median(a[m]))}
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 64)):
        m = (fr >= lo) & (fr <= hi)
        if m.any():
            o["by_free_slots_after_exit"]["%d-%d" % (lo, hi)] = {"n": int(m.sum()), "mean": float(a[m].mean()),
                                                                  "median": float(np.median(a[m]))}
    return o


def analyse_launch(a, n_sm, f, slot_ns, serial_ns, m2_slot_ns=None):
    cta, sm, wid, gt_in, c_in, gt_out, c_out = a.T
    R = (c_out - c_in).astype(float)
    # occupancy = largest number of CTAs resident on an SM
    occ = 0
    for s in np.unique(sm):
        m = sm == s
        t = np.concatenate([c_in[m], c_out[m]]); d = np.concatenate([np.ones(m.sum()), -np.ones(m.sum())])
        o = np.lexsort((d, t))  # exits before entries at equal times
        occ = max(occ, int(np.cumsum(d[o]).max()))
    gaps, fw_steps, fw_spread = per_sm_gaps(sm, c_in.astype(float), c_out.astype(float), occ)
    # GPU-wide time: per-SM clock offset from the globaltimer
    t_al_in = np.zeros(len(a)); t_al_out = np.zeros(len(a))
    for s in np.unique(sm):
        m = sm == s
        off = np.median(gt_in[m] * f - c_in[m])
        t_al_in[m] = c_in[m] + off; t_al_out[m] = c_out[m] + off
    t0 = t_al_in.min()
    span = float(t_al_out.max() - t0)
    res = {"ctas": int(len(a)), "occ": occ, "R_median": float(np.median(R)), "R_mean": float(R.mean()),
           "R_p10": float(np.percentile(R, 10)), "R_p90": float(np.percentile(R, 90)),
           "fw_step_median": float(np.median(fw_steps)) if fw_steps else None,
           "fw_spread_median": float(np.median(fw_spread)) if fw_spread else None,
           "span": span, "cycles_per_cta": span / len(a), "fill": float(R.sum() / (n_sm * occ * span)),
           "gaps": summarize_gaps(gaps)}
    # dispatch order: sweeps after the first wave
    order = sm[:n_sm]
    rank = {int(s): i for i, s in enumerate(order)}
    if len(rank) == n_sm and len(a) > n_sm * occ:
        rk = np.array([rank[int(s)] for s in sm[n_sm * occ:]])
        cuts = np.flatnonzero(np.diff(rk) < 0)
        sizes = np.diff(np.concatenate([[-1], cuts, [len(rk) - 1]]))
        res["sweep_ctas_median"] = float(np.median(sizes)); res["sweep_ctas_mean"] = float(sizes.mean())
        res["sweeps"] = int(len(sizes))
    # refill entry rate GPU-wide
    ref = np.sort(t_al_in[n_sm * occ:] - t0) if len(a) > n_sm * occ else np.array([])
    if len(ref) > 600:
        w = 512.0
        cnt = np.searchsorted(ref, ref + w) - np.arange(len(ref))
        res["refill_rate_max_per_cycle"] = float(cnt.max() / w)
        res["refill_rate_overall_per_cycle"] = float(len(ref) / (ref[-1] - ref[0]))
    # M0 on the measured residencies (dispatch order = CTA index)
    slot = float(int(slot_ns * f)); serial = float(int(serial_ns * f))
    e, x, s0 = replay_m0(R, n_sm, occ, slot, serial)
    g0, _, _ = per_sm_gaps(s0, e, x, occ)
    res["m0"] = {"slot_cycles": slot, "serial_cycles": serial, "span": float(x.max()),
                 "fill": float(R.sum() / (n_sm * occ * x.max())), "gaps": summarize_gaps(g0)}
    res["span_over_m0"] = span / float(x.max())
    e, x, s1 = replay_m0(R, n_sm, occ, slot, serial, per_slot=True)
    g1, _, _ = per_sm_gaps(s1, e, x, occ)
    res["m1"] = {"span": float(x.max()), "fill": float(R.sum() / (n_sm * occ * x.max())), "gaps": summarize_gaps(g1)}
    res["span_over_m1"] = span / float(x.max())
    if m2_slot_ns is not None:
        # M2: M1 with the constants not truncated and the device's measured slot constant
        e, x, s2 = replay_m0(R, n_sm, occ, m2_slot_ns * f, serial_ns * f, per_slot=True)
        g2, _, _ = per_sm_gaps(s2, e, x, occ)
        res["m2"] = {"slot_cycles": m2_slot_ns * f, "serial_cycles": serial_ns * f, "span": float(x.max()),
                     "fill": float(R.sum() / (n_sm * occ * x.max())), "gaps": summarize_gaps(g2)}
        res["span_over_m2"] = span / float(x.max())
    return res


def med(rows, *path):
    v = []
    for r in rows:
        x = r
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
            if x is None:
                break
        if x is not None:
            v.append(x)
    return st.median(v) if v else None


def cmd_cells(run, slot_ns, serial_ns, m2_slot_ns=None):
    files = sorted(glob.glob(os.path.join(run, "cells", "*.txt")))
    loaded = {os.path.basename(p)[:-4]: load(p) for p in files}
    fs = [cycles_per_ns({L: a for L, a in l.items() if L > 0}) for _, l in loaded.values()]
    fs = [x for x in fs if x]
    f_dev = st.median(fs) if fs else None
    if "--f" in sys.argv:
        f_dev = opt("--f", f_dev)
    out = {"cycles_per_ns": f_dev, "slot_ns": slot_ns, "serial_ns": serial_ns, "m2_slot_ns": m2_slot_ns, "cells": {}}
    hdr = "cell\tocc\tR\tfw_step\tfw_spread\tspan\tcyc/CTA\tfill\tgap_mean\tgap_med\tgap_g2\tgap_g3\tgap_g4+\tsweep\trate_max\tm0_span\tm0_gap\tspan/m0\tm1_span\tm1_gap\tspan/m1\tm2_span\tm2_gap\tspan/m2"
    print(hdr)
    for name, (meta, launches) in loaded.items():
        if "sms" not in meta:
            continue
        n_sm = int(meta["sms"])
        rows = [analyse_launch(a, n_sm, f_dev, slot_ns, serial_ns, m2_slot_ns) for L, a in sorted(launches.items()) if L > 0]
        if not rows:
            continue
        c = {"meta": meta, "launches": rows}
        keys = [("occ",), ("R_median",), ("fw_step_median",), ("fw_spread_median",), ("span",), ("cycles_per_cta",),
                ("fill",), ("gaps", "gap_mean"), ("gaps", "gap_median"), ("gaps", "by_generation", "2", "mean"),
                ("gaps", "by_generation", "3", "mean"), ("gaps", "by_generation", "4+", "mean"),
                ("sweep_ctas_median",), ("refill_rate_max_per_cycle",), ("m0", "span"), ("m0", "gaps", "gap_mean"),
                ("span_over_m0",), ("m1", "span"), ("m1", "gaps", "gap_mean"), ("span_over_m1",),
                ("m2", "span"), ("m2", "gaps", "gap_mean"), ("span_over_m2",)]
        c["median"] = {"/".join(k): med(rows, *k) for k in keys}
        c["span_spread"] = (max(r["span"] for r in rows) - min(r["span"] for r in rows)) / st.median(r["span"] for r in rows)
        out["cells"][name] = c
        fmt = lambda x: "-" if x is None else ("%.3f" % x if abs(x) < 10 else "%.0f" % x)
        print(name + "\t" + "\t".join(fmt(c["median"]["/".join(k)]) for k in keys))
    o = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else run
    os.makedirs(o, exist_ok=True)
    json.dump(out, open(os.path.join(o, "model.json"), "w"), indent=1)


CELLS = [  # (mode, threads, occ request, waves, hold, jitter) -- run.sh carries the same list
    *[("spin", 128, 0, w, h, 0) for h in (0, 800, 1500, 3000, 6000, 12000) for w in (4, 12)],
    ("spin", 128, 0, 4, 3000, 300), ("spin", 128, 0, 12, 3000, 300), ("spin", 128, 0, 12, 6000, 300),
    ("spin", 128, 4, 12, 3000, 0), ("spin", 128, 8, 12, 3000, 0),
    ("spin", 256, 0, 12, 3000, 0), ("spin", 512, 0, 12, 3000, 0), ("spin", 32, 0, 12, 3000, 0),
    ("sleep", 128, 0, 4, 3000, 0), ("sleep", 128, 0, 12, 3000, 0),
    *[(m, 128, 0, w, 3000, 0) for m in ("ldg", "cpasync") for w in (4, 12)],
    ("ldg", 128, 0, 4, 0, 0), ("cpasync", 128, 0, 4, 0, 0),
]


CELLS2 = [  # stage 2 (addendum 1): confirmation cells, then the kernel-shaped cells (mode rms, explicit grid)
    ("spin", 128, 0, 6, 2200, 150), ("spin", 128, 0, 6, 4500, 150), ("spin", 128, 0, 6, 9000, 150),
    ("spin", 128, 0, 6, 4500, 0),
]
CELLS2_RMS = [(5612, 1400), (7552, 1400), (8192, 1400), (8192, 700), (8192, 2800)]  # (grid, hold)


def cell_name(c):
    return "%s.t%d.o%d.w%d.h%d.j%d" % c


def cmd_predict2(n_sm, f, slot_ns, serial_ns, base):
    print("M2 slot %.1f serial %.3f cycles; nominal R = hold + %d" % (slot_ns * f, serial_ns * f, base))
    print("cell\tocc\tR\tspan\tcyc/CTA\tfill\tgap_mean\tgap_g2\tgap_g3\tgap_g4+")
    for c in CELLS2:
        mode, thr, occ_req, waves, hold, jit = c
        occ = 16; n = n_sm * occ * waves
        rng = np.random.default_rng(1)
        R = base + hold * (1 + (jit / 1000.0) * rng.uniform(-1, 1, n)) if jit else np.full(n, float(hold + base))
        e, x, s0 = replay_m0(R, n_sm, occ, slot_ns * f, serial_ns * f, per_slot=True)
        g = summarize_gaps(per_sm_gaps(s0, e, x, occ)[0]); bg = g["by_generation"]
        print("%s\t%d\t%.0f\t%.0f\t%.3f\t%.3f\t%.0f\t%s" % (
            cell_name(c), occ, np.median(R), x.max(), x.max() / n, R.sum() / (n_sm * occ * x.max()), g["gap_mean"],
            "\t".join("%.0f" % bg[k]["mean"] if k in bg else "-" for k in ("2", "3", "4+"))))


def cmd_predict(n_sm, f, slot_ns, serial_ns, base):
    slot = float(int(slot_ns * f)); serial = float(int(serial_ns * f))
    print("M0 slot %.0f serial %.0f cycles; nominal R = hold + %d (spin cells)" % (slot, serial, base))
    print("cell\tocc\tR\tM0: span\tcyc/CTA\tfill\tgap_mean\tgap_g2\tgap_g3\tgap_g4+\tM1: span\tcyc/CTA\tfill\tgap_mean\tgap_g2\tgap_g3\tgap_g4+")
    nat = {128: min(16, 64 // 4), 256: 8, 512: 4, 32: 32}
    for c in CELLS:
        mode, thr, occ_req, waves, hold, jit = c
        if mode not in ("spin", "sleep"):
            continue
        occ = occ_req or nat[thr]
        n = n_sm * occ * waves
        rng = np.random.default_rng(1)
        R = np.full(n, float(hold + base))
        if jit:
            R = base + hold * (1 + (jit / 1000.0) * rng.uniform(-1, 1, n))
        cols = []
        for per_slot in (False, True):
            e, x, s0 = replay_m0(R, n_sm, occ, slot, serial, per_slot)
            g = summarize_gaps(per_sm_gaps(s0, e, x, occ)[0])
            bg = g["by_generation"]
            cols += ["%.0f" % x.max(), "%.3f" % (x.max() / n), "%.3f" % (R.sum() / (n_sm * occ * x.max())),
                     "%.0f" % g["gap_mean"]] + ["%.0f" % bg[k]["mean"] if k in bg else "-" for k in ("2", "3", "4+")]
        print("%s\t%d\t%.0f\t%s" % (cell_name(c), occ, np.median(R), "\t".join(cols)))


def cmd_ncu(cell, csvp):
    import csv, io
    meta, launches = load(cell)
    t = open(csvp, errors="replace").read(); i = t.find('"ID"')
    rows = [r for r in csv.DictReader(io.StringIO(t[i:])) if "cell_kernel" in r.get("Kernel Name", "")]
    num = lambda s: float(s.replace(",", ""))
    for k, r in enumerate(rows):
        lat = {c[len("smsp__average_warp_latency_issue_stalled_"):-len(".ratio")]: num(v) for c, v in r.items()
               if c.startswith("smsp__average_warp_latency_issue_stalled_") and c.endswith(".ratio") and v not in ("", "n/a")}
        a = launches.get(k)
        R = float(np.mean(a[:, 6] - a[:, 4])) if a is not None else float("nan")
        print("launch %d ncu sum_latency %.0f (selected %.1f) cycles_active max %s avg %s | in-kernel R mean %.0f | sum/R %.3f" % (
            k, sum(lat.values()), lat.get("selected", float("nan")), r.get("sm__cycles_active.max"), r.get("sm__cycles_active.avg"),
            R, sum(lat.values()) / R))


def opt(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


if __name__ == "__main__":
    slot_ns = opt("--slot-ns", 160.0); serial_ns = opt("--serial-ns", 0.513613)
    if sys.argv[1] == "cells":
        cmd_cells(sys.argv[2], slot_ns, serial_ns, opt("--m2-slot-ns", None))
    elif sys.argv[1] == "cells4":
        cmd_cells4(sys.argv[2], opt("--f", 0.0), slot_ns, serial_ns, opt("--units", 1.0),
                   sys.argv[sys.argv.index("--topo") + 1], sys.argv[sys.argv.index("--out") + 1])
    elif sys.argv[1] == "predict4":
        cmd_predict4(int(sys.argv[2]), float(sys.argv[3]), slot_ns, serial_ns, opt("--units", 1.0), int(opt("--base", 174)))
    elif sys.argv[1] == "predict2":
        cmd_predict2(int(sys.argv[2]), float(sys.argv[3]), slot_ns, serial_ns, int(opt("--base", 180)))
    elif sys.argv[1] == "predict":
        cmd_predict(int(sys.argv[2]), float(sys.argv[3]), slot_ns, serial_ns, int(opt("--base", 280)))
    elif sys.argv[1] == "ncu":
        cmd_ncu(sys.argv[2], sys.argv[3])
