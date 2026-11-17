#!/usr/bin/env python3
"""trywait_wake derivation (protocol.json)."""
import argparse, json, statistics
from pathlib import Path

def load(p):
    rows = {}
    for line in Path(p).read_text().splitlines():
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        rows[d["family"]] = d
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--release", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {r: load(Path(a.run) / f"run-{r}.jsonl") for r in ("a", "b")}
    fams = sorted(set(runs["a"]) | set(runs["b"]))
    out = {"run": a.run, "cells": [], "checks": {}, "forms": {}}
    agree = True
    for f in fams:
        cs = [runs[r][f] for r in ("a", "b") if f in runs[r]]
        form, comp, D = f.split(".")[1], f.split(".")[2], int(f.split(".D")[1])
        med = statistics.median(c["median"] for c in cs)
        if len(cs) == 2 and abs(cs[0]["median"] - cs[1]["median"]) > max(0.10 * med, 20):
            agree = False; out["checks"].setdefault("disagree", []).append(f)
        out["cells"].append({"family": f, "form": form, "completion": comp, "D": D, "median": med,
                             "p10": statistics.median(c["p10"] for c in cs), "p90": statistics.median(c["p90"] for c in cs),
                             "error": [c.get("error") for c in cs if c.get("error")]})
    out["checks"]["runs_agree"] = agree
    ctrl = [c["median"] for c in out["cells"] if c["form"] == "testwait"]
    if ctrl:
        out["checks"]["control_median"] = statistics.median(ctrl)
        out["checks"]["control_ok"] = abs(statistics.median(ctrl) - a.release) <= 0.25 * a.release
    for form in ("trywait", "trywait_hint"):
        fo = {}
        for comp in ("arrive", "tx"):
            hi = [c["median"] for c in out["cells"] if c["form"] == form and c["completion"] == comp and c["D"] >= 200]
            lo = [c["median"] for c in out["cells"] if c["form"] == form and c["completion"] == comp and c["D"] < 200]
            if hi:
                m = statistics.mean(hi)
                fo[comp] = {"suspended_wake_mean": m, "cells_ge200": hi, "cells_lt200": lo,
                            "within_30pct": all(abs(x - m) <= 0.30 * m for x in hi)}
        if "arrive" in fo and "tx" in fo:
            fo["arrive_tx_agree"] = abs(fo["arrive"]["suspended_wake_mean"] - fo["tx"]["suspended_wake_mean"]) <= 0.15 * max(fo["arrive"]["suspended_wake_mean"], fo["tx"]["suspended_wake_mean"])
        out["forms"][form] = fo
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))
    for c in out["cells"]:
        print(f"{c['form']:13s} {c['completion']:6s} D={c['D']:5d} wake p10/50/90 = {c['p10']:6.0f} {c['median']:6.0f} {c['p90']:6.0f}")

if __name__ == "__main__":
    main()
