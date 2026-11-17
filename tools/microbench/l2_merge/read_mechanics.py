#!/usr/bin/env python3
"""read_mechanics.py <run dir of run_mechanics.sh>: per cell and run the read ratio r = srcunit_tex L2 read sectors /
sectors delivered (retained launches), the per-CTA delivery, the aggregate and L2-side read rate, the group latency,
and for cp.async cells with shared slices the start-time difference between the members of a slice during the stream
(globaltimer stamps of thread 0, 32 ns resolution)."""
import csv, io, json, os, re, sys, statistics
from pathlib import Path
root = sys.argv[1]


def launches(path, kernel):
    t = open(path, errors="replace").read(); i = t.find('"ID"'); L = {}
    if i < 0: return []
    for r in csv.DictReader(io.StringIO(t[i:])):
        if kernel in r.get("Kernel Name", ""):
            try: L.setdefault(int(r["ID"]), {})[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
            except ValueError: pass
    return [L[k] for k in sorted(L)]


def ratio(ls):
    out = [m["lts__t_sectors_srcunit_tex_op_read.sum"] / m["l1tex__m_xbar2l1tex_read_sectors.sum"]
           for m in ls[1:] if m.get("l1tex__m_xbar2l1tex_read_sectors.sum")]
    return out


def table(path):
    rows = [l.split() for l in open(path).read().splitlines()[1:]]
    return {int(a): (int(b), int(c)) for a, b, c in rows}


def key(c):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", c)]


print("%-18s %-21s %-21s %s" % ("cell", "run-a r", "run-b r", "run-a: B/cyc/CTA, aggregate B/ns, L2 reads B/ns, latency p50/p90 | notes"))
for cell in sorted(os.listdir(Path(root, "run-a")), key=key):
    cols = []; note = ""
    for run in "ab":
        d = Path(root, f"run-{run}", cell); kind = cell.split(".")[0]
        rs = ratio(launches(d / "ncu.csv", "tma_scan_kernel" if kind == "tma" else "k_ldgsts_loaded")) if (d / "ncu.csv").exists() else []
        cols.append(" ".join("%.3f" % x for x in rs) or "missing")
        if run != "a": continue
        r = sum(rs) / len(rs) if rs else float("nan")
        if kind == "cpa" and (d / "probe.jsonl").exists():
            p = json.loads(open(d / "probe.jsonl").readline())
            note = "%6.1f %7.0f %7.0f  %4.0f/%4.0f" % (p["median"], p["aggregate_bytes_per_ns_median"], p["aggregate_bytes_per_ns_median"] * r, p["latency_p50"], p["latency_p90"])
            if p.get("pace_cycles"): note += " late %d" % p["late_rounds"]
            tb = table(p["table"]) if p.get("table") else {}
            by = {}
            for cta, sm in enumerate(p["smid_by_cta"]):
                sl, mem = tb.get(int(sm), (None, 65535))
                if mem != 65535: by.setdefault(sl, []).append((mem, cta))
            diffs = []
            for sl, ms in by.items():
                if len(ms) < 2: continue
                ms.sort(); a, b = ms[0][1], ms[1][1]
                sa, sb = p["stamps_ns_by_cta"][a], p["stamps_ns_by_cta"][b]
                diffs += [y - x for x, y in zip(sa[8:], sb[8:]) if x >= 0 and y >= 0]
            if diffs:
                ab = sorted(abs(x) for x in diffs)
                note += " | member1-member0 start: median %+.0f ns, |d| p50 %.0f p90 %.0f ns, >=64 ns %.2f" % (
                    statistics.median(diffs), ab[len(ab) // 2], ab[int(0.9 * len(ab))], sum(x >= 64 for x in ab) / len(ab))
        elif kind == "tma" and (d / "probe.txt").exists():
            for line in open(d / "probe.txt", errors="replace"):
                if line.startswith("RESULT "):
                    kv = dict(re.findall(r"(\w+)=([-\w.]+)", line)); note = "%6.1f B/cyc/SM %7.0f B/ns" % (float(kv["per_sm_B_per_cycle"]), float(kv["agg_B_per_ns"]))
    print("%-18s %-21s %-21s %s" % (cell, cols[0], cols[1], note))
