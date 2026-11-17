#!/usr/bin/env python3
"""shared derivation (tools/microbench/shared/protocol.json): LDS chain
latency, per-SMSP issue cost, per-SM acceptance per wavefront, STS
visibility, from two runs of smem.cu plus the SASS checks."""
import argparse, json, statistics, math
from pathlib import Path

RESID = 0.05
AGREE = 0.05


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
    return {"slope": b, "max_rel_resid": resid, "medians": dict(zip(us, meds))}


def load(run_dir):
    rows = []
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "samples" in d:
                rows.append(d)
    return rows


def ok(checks, key):
    k = checks.get("smem", {}).get("kernels", {}).get(key)
    return k if (k is not None and k.get("ok")) else None


def derive(rows, checks):
    out = {"lds_chain": {}, "issue": {}, "acceptance": {}, "vis": {},
           "ldsm_chain": {}, "mtx_issue": {}}
    chain, issue = {}, {}
    for r in rows:
        v, p, u = r["variant"], r.get("params", {}), int(r["u"])
        if v.startswith("ldsm_chain"):
            c = ok(checks, f"smem/ldsm_chain/x{p['x']}.mt{p['matrix_transposed']}/{u}")
            if c:
                out["ldsm_chain"].setdefault(
                    (p["x"], bool(p["matrix_transposed"])), {}
                    ).setdefault(u, []).extend(
                        float(x) for x in r["samples"])
            continue
        if v.startswith("matrix."):
            c = ok(checks, f"matrix/{p['op']}.x{p['x']}"
                           f".mt{p['matrix_transposed']}/{u}")
            if c:
                out["mtx_issue"].setdefault(
                    (p["op"], int(p["x"]), bool(p["matrix_transposed"]),
                     int(p["wps"]), bool(p["all_smsp"])), {}
                    ).setdefault(u, []).extend(
                        float(x) for x in r["samples"])
            continue
        if v.startswith("lds_chain") and ok(checks, f"smem/lds_chain/w{p['width_bytes']}/{u}"):
            chain.setdefault(p["width_bytes"], {}).setdefault(u, []).extend(float(x) for x in r["samples"])
        elif v.startswith(("lds.", "sts.", "lds_nv.")) and ok(checks, f"smem/issue/{p['op']}/{u}"):
            issue.setdefault((p["op"], int(p["degree_param"]), int(p["wps"]), bool(p["all_smsp"])), {}) \
                 .setdefault(u, []).extend(float(x) for x in r["samples"])
        elif v.startswith("vis.") and v.endswith(".warp1"):
            c = ok(checks, f"smem/vis/{p['stores']}")
            if c:
                xs = [float(x) for x in r["samples"]]
                out["vis"][int(p["stores"])] = {"mean": statistics.fmean(xs), "se": statistics.pstdev(xs) / math.sqrt(len(xs)),
                                               "o_s": c.get("o_s"),
                                               "o_w": (statistics.median(c["o_w"]) if isinstance(c.get("o_w"), list)
                                                       else c.get("o_w"))}
    for wb, s in chain.items():
        f = slope_fit(s)
        if f:
            out["lds_chain"][wb] = f
    for (op, d, wps, al), s in issue.items():
        f = slope_fit(s)
        if not f:
            continue
        if al:
            out["acceptance"].setdefault(op, {})[max(d, 1)] = {"cycles_per_op_per_sm": f["slope"] / (wps * 4), "resid": f["max_rel_resid"], "wps": wps}
        elif d == 1:
            out["issue"][op] = {"cycles_per_op_per_smsp": f["slope"] / wps, "resid": f["max_rel_resid"], "wps": wps}
    for key, s in out["ldsm_chain"].items():
        f = slope_fit(s)
        out["ldsm_chain"][key] = f
    mtx_issue = out["mtx_issue"]
    out["mtx_issue"] = {}
    for (op, x, tr, wps, al), s in mtx_issue.items():
        f = slope_fit(s)
        if f:
            out["mtx_issue"][(op, x, tr, wps, al)] = f
    if 1 in out["vis"]:
        v1 = out["vis"][1]
        if v1["o_s"] is not None and v1["o_w"] is not None:
            base = out["vis"].get(0, {}).get("mean", 0.0)
            out["visibility"] = {"value": v1["mean"] - v1["o_s"] - v1["o_w"] - base, "se": v1["se"],
                                 "eq": "E[t_w - t_s | S=1] - o_s - o_w - E[S=0]"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True); ap.add_argument("--run-b", required=True)
    ap.add_argument("--checks", required=True); ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    checks = json.loads(Path(a.checks).read_text())
    da, db = derive(load(a.run_a), checks), derive(load(a.run_b), checks)

    def agree(x, y):
        m = (x + y) / 2
        return abs(x - y) / m if m else float("inf")

    res = {"protocol": "tools/microbench/shared/protocol.json", "device": a.device, "runs": {"a": a.run_a, "b": a.run_b}}
    # LDS result latency
    if 4 in da["lds_chain"] and 4 in db["lds_chain"]:
        fa, fb = da["lds_chain"][4], db["lds_chain"][4]
        res["lds_result_latency"] = {"value": (fa["slope"] + fb["slope"]) / 2, "agree": agree(fa["slope"], fb["slope"]),
                                     "resid": max(fa["max_rel_resid"], fb["max_rel_resid"])}
        res["lds_result_latency"]["verdict"] = "measured" if res["lds_result_latency"]["resid"] <= RESID and res["lds_result_latency"]["agree"] <= AGREE else "experimental"
        res["lds_chain_by_width"] = {str(w): {"a": da["lds_chain"].get(w, {}).get("slope"), "b": db["lds_chain"].get(w, {}).get("slope")} for w in sorted(set(da["lds_chain"]) | set(db["lds_chain"]))}
    res["ldsm_result"] = {}
    for key in sorted(set(da["ldsm_chain"]) & set(db["ldsm_chain"])):
        fa, fb = da["ldsm_chain"][key], db["ldsm_chain"][key]
        if fa and fb:
            res["ldsm_result"][f"x{key[0]}.mt{int(key[1])}"] = {
                "value": (fa["slope"] + fb["slope"]) / 2,
                "agree": agree(fa["slope"], fb["slope"]),
                "resid": max(fa["max_rel_resid"],
                             fb["max_rel_resid"]),
                "verdict": "measured" if
                    max(fa["max_rel_resid"], fb["max_rel_resid"]) <= RESID
                    and agree(fa["slope"], fb["slope"]) <= AGREE
                    else "experimental"}
    res["mtx_issue"] = {}
    for key in sorted(set(da["mtx_issue"]) & set(db["mtx_issue"]),
                      key=str):
        fa, fb = da["mtx_issue"][key], db["mtx_issue"][key]
        if fa and fb:
            op, x, tr, wps, al = key
            res["mtx_issue"][f"{op}.x{x}.mt{int(tr)}."
                             f"wps{wps}.{int(al)}"] = {
                "value": (fa["slope"] + fb["slope"]) / 2,
                "per_smsp": (fa["slope"] + fb["slope"]) / 2 / wps,
                "per_sm": (fa["slope"] + fb["slope"]) / 2 / (wps * 4),
                "agree": agree(fa["slope"], fb["slope"]),
                "resid": max(fa["max_rel_resid"],
                             fb["max_rel_resid"])}
    res["issue"] = {}
    for op in sorted(set(da["issue"]) & set(db["issue"])):
        x, y = da["issue"][op]["cycles_per_op_per_smsp"], db["issue"][op]["cycles_per_op_per_smsp"]
        r = max(da["issue"][op]["resid"], db["issue"][op]["resid"])
        res["issue"][op] = {"value": (x + y) / 2, "agree": agree(x, y), "resid": r,
                            "verdict": "measured" if r <= RESID and agree(x, y) <= AGREE else "experimental"}
    res["acceptance"] = {}
    for op in sorted(set(da["acceptance"]) & set(db["acceptance"])):
        knots = []
        for d in sorted(set(da["acceptance"][op]) & set(db["acceptance"][op])):
            x, y = da["acceptance"][op][d]["cycles_per_op_per_sm"], db["acceptance"][op][d]["cycles_per_op_per_sm"]
            r = max(da["acceptance"][op][d]["resid"], db["acceptance"][op][d]["resid"])
            knots.append([d, round((x + y) / 2, 4), "measured" if r <= RESID and agree(x, y) <= AGREE else "experimental"])
        res["acceptance"][op] = knots
    if "visibility" in da and "visibility" in db:
        x, y = da["visibility"]["value"], db["visibility"]["value"]
        res["visibility"] = {"value": (x + y) / 2, "agree": agree(x, y), "a": da["visibility"], "b": db["visibility"],
                             "verdict": "measured" if agree(x, y) <= AGREE else "experimental"}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("runs", "protocol")}, indent=None)[:3000])


if __name__ == "__main__":
    main()
