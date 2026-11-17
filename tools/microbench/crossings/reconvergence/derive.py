#!/usr/bin/env python3
"""ctrl derivation, verbatim from blackwell-prototype ctrl_model.py (minimal)."""
import json, re, statistics, sys


def body(seg, kind):
    insts = [(s[1], s[2]) for s in seg]
    end = max(i for i, (t, _) in enumerate(insts) if t.startswith("BRA.U") and "!" not in t.split(",")[0])
    return insts[:end + 1]


def derive(run_dir, checks):
    chk = json.load(open(checks))["ctrl"]["kernels"]
    series, per = {}, {}
    for line in open(run_dir):
        r = json.loads(line)
        if chk.get(f"ctrl/{r['variant']}/{r['u']}", {}).get("ok"):
            series.setdefault(r["variant"], {})[r["u"]] = statistics.median(r["samples"])
    for v, d in series.items():
        ns = sorted(d)
        slope = (d[ns[-1]] - d[ns[0]]) / (ns[-1] - ns[0])
        per.setdefault(v, {"per_iteration": [], "segment": chk[f"ctrl/{v}/64"]["segment"]})["per_iteration"].append(slope)
    out = {"variants": {}}
    for v, e in sorted(per.items()):
        out["variants"][v] = {"per_iteration": statistics.fmean(e["per_iteration"]), "raw": e["per_iteration"]}
    for kind in ("uni", "div"):
        res = []
        for v, rec in out["variants"].items():
            if not v.startswith(kind + "."):
                continue
            loop = body(per[v]["segment"], kind)
            stalls = sum(s for _, s in loop)
            if kind == "uni":
                stalls -= sum(s for t, s in loop if "FFMA" in t) / 2
            rec.update(stalls=stalls, residual=rec["per_iteration"] - stalls)
            res.append(rec["residual"])
        out[f"{kind}_region_residual"] = {"value": statistics.fmean(res) if res else None, "raw": res}
    return out


if __name__ == "__main__":
    res = {r: derive(p, sys.argv[3]) for r, p in (("run-a", sys.argv[1]), ("run-b", sys.argv[2]))}
    json.dump(res, open(sys.argv[4], "w"), indent=1)
    for r, o in res.items():
        print(r, "div residual:", o["div_region_residual"], "uni:", o["uni_region_residual"])
