#!/usr/bin/env python3
"""read_domain.py <run dir of run_domain.sh> <sm_topology.json>

Per cell and run: r_all = lts__t_sectors_op_read.sum / l1tex__m_xbar2l1tex_read_sectors.sum and
r_tex = lts__t_sectors_srcunit_tex_op_read.sum / l1tex__m_xbar2l1tex_read_sectors.sum of the retained
launches (the first launch of every cell warms the L2 and is dropped), next to the ratio each candidate
domain predicts (domain_tables.py).  Cells grouped by block index are predicted from the recorded SM ids.
"""
import csv, io, json, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import domain_tables as dt

root, topo_path = sys.argv[1], sys.argv[2]
topo = json.loads(Path(topo_path).read_text())
dev = topo["device"]
dom = dt.domains(dev, topo)
pred = json.loads(Path(root, "tables", "predictions.json").read_text())
cap = pred["cap"]


def launches(path, kernel):
    t = open(path, errors="replace").read(); i = t.find('"ID"')
    L = {}
    if i < 0: return []
    for r in csv.DictReader(io.StringIO(t[i:])):
        if kernel in r.get("Kernel Name", ""):
            try: L.setdefault(int(r["ID"]), {})[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
            except ValueError: pass
    return [L[k] for k in sorted(L)]


def ratios(ls):
    out = []
    for m in ls[1:]:
        xb = m.get("l1tex__m_xbar2l1tex_read_sectors.sum", 0)
        if xb: out.append((m["lts__t_sectors_op_read.sum"] / xb, m["lts__t_sectors_srcunit_tex_op_read.sum"] / xb))
    return out


def tma_smids(path):
    sm = {}
    for line in open(path, errors="replace"):
        if line.startswith("CTA trial=1 "):
            kv = dict(re.findall(r"(\w+)=([-\w.]+)", line)); sm[int(kv["cta"])] = int(kv["smid"])
    return [sm[c] for c in sorted(sm)]


def tma_info(path):
    res = {}
    for line in open(path, errors="replace"):
        if line.startswith("RESULT "): res.update({k: float(v) for k, v in re.findall(r"(\w+)=([-\w.]+)", line)})
        if line.startswith("SYNC trial=1"): res.update({"sync_" + k: float(v) for k, v in re.findall(r"(\w+)=([-\w.]+)", line)})
    return res


cells = sorted(os.listdir(Path(root, "run-a")))
print("%-16s %-27s %-27s  %s" % ("cell", "run-a r_all (r_tex)", "run-b r_all (r_tex)", "predicted pair / block / gpc (N=%d)   notes" % cap))
for cell in cells:
    cols = []; note = ""
    for run in "ab":
        d = Path(root, f"run-{run}", cell)
        kind, name = cell.split(".", 1)
        rs = ratios(launches(d / "ncu.csv", "tma_scan_kernel" if kind == "tma" else "k_ldgsts_loaded")) if (d / "ncu.csv").exists() else []
        cols.append(" ".join("%.3f(%.3f)" % x for x in rs) or "missing")
        if run != "a": continue
        base = name.split(".s")[0] if re.search(r"\.s\d+$", name) else name
        if base in pred["cells"]:
            p = pred["cells"][base]; note = "%.3f / %.3f / %.3f" % tuple(p[k][str(cap)] for k in ("pair", "block", "gpc"))
        elif kind == "tma" and (d / "probe.txt").exists():
            sm = tma_smids(d / "probe.txt")
            m = re.match(r"g(\d+)\.(far[01]|rot)", name)
            if m and sm:
                g = int(m.group(1)); n = len(sm)
                sl = [c % (n // g) for c in range(n)] if m.group(2) == "far1" else [c // g for c in range(n)]
                if m.group(2) == "rot": note = "rotated control"
                else: note = "%.3f / %.3f / %.3f (recorded SM ids)" % tuple(dt.predict_from_slices(sl, sm, dom[k], cap) for k in ("pair", "block", "gpc"))
            elif name == "mc2": note = "0.500 if a multicast read is one read for two deliveries"
        if kind == "tma" and (d / "probe.txt").exists():
            i = tma_info(d / "probe.txt")
            note += "  | %.1f B/cyc/SM, start spread %.0f ns" % (i.get("per_sm_B_per_cycle", 0), i.get("sync_m0_max_start_ns", -1))
            sm = tma_smids(d / "probe.txt")
            if len(set(sm)) != len(sm): note += " DUPLICATE SMs"
        if kind == "cpa" and (d / "probe.jsonl").exists():
            p = json.loads(open(d / "probe.jsonl").readline())
            note += "  | %.1f B/cyc/CTA lat50 %.0f" % (p["median"], p["latency_p50"])
            if p.get("shift_cycles") and p.get("table"):
                tb = [l.split() for l in open(p["table"]).read().splitlines()[1:]]
                mem = {int(a): int(c) for a, b, c in tb}; slc = {int(a): int(b) for a, b, c in tb}
                shared = {s for s in slc if list(slc.values()).count(slc[s]) > 1}
                cy = dict(zip([int(x) for x in p["smid_by_cta"]], p["cycles_by_cta"]))
                lead = [cy[s] for s in shared if mem[s] == 0 and s in cy]; fol = [cy[s] for s in shared if mem[s] == 1 and s in cy]
                if lead and fol: note += " stream cycles member0 %.0f member1 %.0f" % (sum(lead) / len(lead), sum(fol) / len(fol))
    print("%-16s %-27s %-27s  %s" % (cell, cols[0], cols[1], note))
