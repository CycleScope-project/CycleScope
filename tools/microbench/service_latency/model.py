#!/usr/bin/env python3
"""service_latency derivation (tools/microbench/service_latency/protocol.json).

Inputs: two run directories (alu.jsonl / alu2.jsonl written by the probes)
and the sass checks.json of the build.  Output: per operation_class
result_latency (lat slope), issue_interval (thr slope / chains / wps=1),
fit residuals, run agreement, verdicts; the class map is the blackwell-prototype
wave3_leaves.py table (data, copied).
"""
import argparse, json, statistics
from pathlib import Path

LATENCY_CLASSES = {
    "FFMA": ["FFMA"], "FADD": ["FADD"], "FMUL": ["FMUL"], "FFMA2": ["FFMA2"], "FADD2": ["FADD2"],
    "FMUL2": ["FMUL2"], "HFMA2": ["HFMA2"], "HADD2": ["HADD2"], "HMUL2": ["HMUL2"], "F2FP": ["F2FP"],
    "I2FP": ["I2FP"], "IADD3": ["IADD3"], "IMAD": ["IMAD"], "LOP3": ["LOP3"], "PRMT": ["PRMT"], "SHF": ["SHF"],
    "VIMNMX": ["VIMNMX"], "FMNMX": ["FMNMX"], "POPC": ["POPC"], "FLO": ["FLO"], "BREV": ["BREV"], "F2I": ["F2I"],
    "I2F": ["I2F"], "SHFL.IDX": ["SHFL_IDX"], "SHFL.BFLY": ["SHFL_BFLY"], "SHFL.UP": ["SHFL_UP"],
    "SHFL.DOWN": ["SHFL_DOWN"], "DADD": ["DADD"], "DMUL": ["DMUL"], "DFMA": ["DFMA"],
    "MUFU": ["MUFU_RCP"], "IMAD.HI": ["IMADHI"], "HSETP2": ["HSETP2"], "IDP": ["IDP4A"],
    "BMSK": ["BMSK"], "SGXT": ["SGXT"], "HMNMX2": ["F16X2_MAX"],
}
ALU2_CLASSES = {
    "SEL": ("vlat", "SEL", None), "ISETP": ("vlat", "ISETP_SEL", "SEL"),
    "FSEL": ("vlat", "FSEL", None), "FSETP": ("vlat", "FSETP_FSEL", "FSEL"),
    "FMNMX3": ("vlat", "FMNMX3", None), "VIMNMX3": ("vlat", "VIMNMX3", None),
    "UIADD3": ("ulat", "IADD3", None), "UIMAD": ("ulat", "IMAD", None), "ULOP3": ("ulat", "LOP3", None),
    "USHF": ("ulat", "SHF", None), "UPRMT": ("ulat", "PRMT", None), "UPOPC": ("ulat", "POPC", None),
    "UFLO": ("ulat", "FLO", None), "USEL": ("ulat", "SEL", None), "UISETP": ("ulat", "ISETP_SEL", "SEL"),
}
RESID = 0.05
AGREE = 0.05
CONTROL = 0.10


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return b, my - b * mx


def slope_fit(series):
    us = sorted(series)
    if len(us) < 2:
        return None
    meds = [statistics.median(series[u]) for u in us]
    b, a = ols(us, meds)
    resid = max(abs(m - (a + b * u)) / max(1.0, abs(m)) for u, m in zip(us, meds))
    return {"slope": b, "intercept": a, "us": us, "medians": meds, "max_rel_resid": resid}


def load(run_dir):
    rows = []
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "samples" in d:
                d["probe"] = jl.stem
                rows.append(d)
    return rows


def series(rows, probe, pred):
    out = {}
    for r in rows:
        if r["probe"] == probe and pred(r):
            out.setdefault(int(r["u"]), []).extend(float(x) for x in r["samples"])
    return out


def ok(checks, probe, key):
    k = checks.get(probe, {}).get("kernels", {}).get(key)
    return k is not None and k.get("ok")


def derive(rows, checks):
    out = {}
    for cls, ops in LATENCY_CLASSES.items():
        op = ops[0]
        lat = series(rows, "alu", lambda r: r["params"].get("mode") == "lat" and r["params"].get("op") == op
                     and ok(checks, "alu", f"alu/lat/{op}/{r['u']}"))
        f = slope_fit(lat)
        e = {"op": op}
        if f:
            e["result_latency"] = f["slope"]; e["lat_resid"] = f["max_rel_resid"]; e["lat_medians"] = dict(zip(f["us"], f["medians"]))
        thr = series(rows, "alu", lambda r: r["params"].get("mode") == "thr" and r["params"].get("op") == op
                     and int(r["params"].get("wps", 0)) == 1 and ok(checks, "alu", f"alu/thr/{op}/{r['u']}"))
        g = slope_fit(thr)
        if g:
            e["issue_interval"] = g["slope"] / 8.0; e["thr_resid"] = g["max_rel_resid"]
        if len(e) > 1:
            out[cls] = e
    for cls, (mode, op, partner) in ALU2_CLASSES.items():
        s = series(rows, "alu2", lambda r: r["params"].get("mode") == mode and r["params"].get("op") == op
                   and ok(checks, "alu2", f"alu2/{mode}/{op}/{r['u']}"))
        f = slope_fit(s)
        if not f:
            continue
        v = f["slope"]; resid = f["max_rel_resid"]
        if partner:
            p = slope_fit(series(rows, "alu2", lambda r: r["params"].get("mode") == mode and r["params"].get("op") == partner
                                 and ok(checks, "alu2", f"alu2/{mode}/{partner}/{r['u']}")))
            if not p:
                continue
            v -= p["slope"]; resid = max(resid, p["max_rel_resid"])
        e = {"op": f"{mode}.{op}", "result_latency": v, "lat_resid": resid}
        if mode == "ulat":
            t = slope_fit(series(rows, "alu2", lambda r: r["params"].get("mode") == "u" and r["params"].get("op") == op
                                 and ok(checks, "alu2", f"alu2/uthr/{op}/{r['u']}")))
            if t:
                e["issue_interval"] = t["slope"] / 8.0; e["thr_resid"] = t["max_rel_resid"]
        out[cls] = e
    return out


def installed(tree_path):
    t = json.loads(Path(tree_path).read_text())
    out = {}
    for n in t["nodes"]:
        if n["id"].startswith("instruction.service") and n.get("parameter") and "predicated_on=1" in n["id"]:
            cls = n["id"].split("operation_class=")[1].split("/")[0]
            fl = n["parameter"]["fields"]
            def c(f):
                m = fl.get(f, {}).get("model", {}); return m.get("value") if m.get("kind") == "constant" else None
            out[cls] = {"result_latency": c("result_latency_cycles"), "issue_interval": c("issue_interval_cycles")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True); ap.add_argument("--run-b", required=True)
    ap.add_argument("--checks", required=True); ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--tree", help="generated tree of this device (control / current values)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    checks = json.loads(Path(a.checks).read_text())
    da, db = derive(load(a.run_a), checks), derive(load(a.run_b), checks)
    inst = installed(a.tree) if a.tree else {}
    classes = {}
    for cls in sorted(set(da) | set(db)):
        ea, eb = da.get(cls, {}), db.get(cls, {})
        c = {"run_a": ea, "run_b": eb, "verdict": {}}
        for field in ("result_latency", "issue_interval"):
            if field in ea and field in eb:
                va, vb = ea[field], eb[field]; m = (va + vb) / 2
                agree = abs(va - vb) / m if m else float("inf")
                resid = max(ea.get("lat_resid" if field == "result_latency" else "thr_resid", 0),
                            eb.get("lat_resid" if field == "result_latency" else "thr_resid", 0))
                c[field] = m; c[field + "_agree"] = agree; c[field + "_resid"] = resid
                c["verdict"][field] = "measured" if resid <= RESID and agree <= AGREE else "experimental"
                ref = (inst.get(cls) or {}).get(field)
                if ref not in (None, 0):
                    c[field + "_installed"] = ref
                    c[field + "_control_ok"] = abs(m - ref) / ref <= CONTROL
        classes[cls] = c
    out = {"protocol": "tools/microbench/service_latency/protocol.json", "device": a.device,
           "runs": {"a": a.run_a, "b": a.run_b}, "classes": classes}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for cls, c in classes.items():
        print(f"  {cls:10s} lat {c.get('result_latency', float('nan')):6.2f} ({c['verdict'].get('result_latency','-')}, inst {c.get('result_latency_installed')}) "
              f"ii {c.get('issue_interval', float('nan')):5.2f} ({c['verdict'].get('issue_interval','-')}, inst {c.get('issue_interval_installed')})")


if __name__ == "__main__":
    main()
