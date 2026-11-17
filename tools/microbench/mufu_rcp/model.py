#!/usr/bin/env python3
"""mufu_rcp model: slopes of pair chains; o from ex2/rsq calibration;
rcp = slope(rcp_mul) - o.  Usage: model.py <A> <dev> <run>"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

CAL = {"h100": {"ex2_mul": 17.00, "rsq_mul": 17.00},
       "b300": {"ex2_mul": 13.95, "rsq_mul": 17.95}}


def ols(xs, ys):
    n = len(xs); sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    s = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c = (sy - s * sx) / n
    r = statistics.pstdev([y - (c + s * x) for x, y in zip(xs, ys)])
    return s, c, r / max(1e-9, abs(sy / n))


def main():
    a, dev, run = sys.argv[1], sys.argv[2], sys.argv[3]
    base = Path(a) / "microbench" / "mufu_rcp" / f"{dev}-{run}"
    rows = defaultdict(dict)
    for rd in sorted(base.glob("run-*")):
        if not rd.is_dir():
            continue
        for ln in (rd / "mufu_rcp.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if "error" in r or not r.get("samples"):
                continue
            p = r["params"]
            rows[(rd.name, p["variant"])][p["u"]] = \
                statistics.median(r["samples"])
    slopes = {}
    print(f"{dev}-{run}: mufu_rcp")
    for v in ("rcp_mul", "ex2_mul", "rsq_mul", "sqrt_mul", "sin_mul"):
        for rn in ("run-a", "run-b"):
            d = rows.get((rn, v), {})
            if len(d) >= 2:
                us = sorted(d)
                s, c, resid = ols(us, [d[u] for u in us])
                slopes.setdefault(v, {})[rn] = (s, resid)
                print(f"  {v:9s} {rn} slope={s:8.3f} resid={resid:.2%}")
    cal = CAL[dev]
    o_a = slopes["ex2_mul"]["run-a"][0] - cal["ex2_mul"]
    o_b = slopes["ex2_mul"]["run-b"][0] - cal["ex2_mul"]
    q_a = slopes["rsq_mul"]["run-a"][0] - cal["rsq_mul"]
    q_b = slopes["rsq_mul"]["run-b"][0] - cal["rsq_mul"]
    o = statistics.median([o_a, o_b, q_a, q_b])
    agree = abs(o_a - q_a) <= 1.0 and abs(o_b - q_b) <= 1.0
    rcp_a = slopes["rcp_mul"]["run-a"][0] - o
    rcp_b = slopes["rcp_mul"]["run-b"][0] - o
    rcp = statistics.median([rcp_a, rcp_b])
    resid = max(slopes["rcp_mul"][rn][1] for rn in ("run-a", "run-b"))
    runs = abs(rcp_a - rcp_b) <= max(0.03 * abs(rcp), 1.0)
    ok = agree and resid <= 0.02 and runs
    print(f"  o={o:.3f} (ex2:{o_a:.2f}/{o_b:.2f} rsq:{q_a:.2f}/{q_b:.2f})"
          f" agree={agree}")
    print(f"  rcp={rcp:.3f} (a={rcp_a:.2f} b={rcp_b:.2f}) resid={resid:.2%}"
          f" runs_ok={runs}")
    decision = "install" if ok else "none"
    out = {"o": o, "agree": agree, "rcp": rcp, "resid": resid,
           "runs_ok": runs, "decision": decision,
           "slopes": {v: {rn: s for rn, (s, _) in d.items()}
                      for v, d in slopes.items()}}
    mdir = Path(a) / "microbench" / "mufu_rcp-model" / f"{dev}-{run}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.json").write_text(json.dumps(out, indent=1))
    print(f"decision {decision}")


if __name__ == "__main__":
    main()
