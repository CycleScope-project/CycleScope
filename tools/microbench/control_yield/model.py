#!/usr/bin/env python3
"""model for control_yield protocol v1.

slope over K per (variant, W) -> cost_yield(W) =
slope(spin,W) - slope(nospin,W) - (37.93 - 4); W-const within 20% ->
install mean; grows with W -> report only; <= 2 -> free.
Usage: model.py <A> <dev> <run>
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = ["spin", "nospin", "spin_nowait"]
KS = [64, 128, 256]
SYNCS_RT = 37.93   # installed SYNCS.PHASECHK result latency
ISETP_RT = 4.0


def ols(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    s = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c = (sy - s * sx) / n
    resid = statistics.pstdev([y - (c + s * x) for x, y in zip(xs, ys)])
    return s, c, resid / max(1e-9, abs(s))


def main() -> int:
    a, dev, run = sys.argv[1], sys.argv[2], sys.argv[3]
    base = Path(a) / "microbench" / "control_yield" / f"{dev}-{run}"
    rows = []
    for rd in sorted(base.glob("run-*")):
        if not rd.is_dir():
            continue
        for ln in (rd / "yield.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if "error" in r or not r.get("samples"):
                continue
            p = r["params"]
            rows.append((rd.name, p["variant"], p["w"], p["k"],
                         r["samples"]))
    med = defaultdict(list)
    for rn, v, w_, k, samples in rows:
        med[(rn, v, w_, k)].append(statistics.median(samples))
    fit = {}
    for rn in ("run-a", "run-b"):
        for v in VARIANTS:
            for w_ in (1, 2, 4):
                ys = [statistics.median(med[(rn, v, w_, k)])
                      for k in KS if (rn, v, w_, k) in med]
                if len(ys) >= 3:
                    fit[(rn, v, w_)] = ols(KS, ys)
    print(f"{dev}-{run}: control_yield v1 model")
    print(f"{'variant':<12}{'W':>2}{'slope':>9}{'icept':>8}"
          f"{'resid%':>8}{'agree':>7}")
    slopes = {}
    ok = True
    for v in VARIANTS:
        for w_ in (1, 2, 4):
            if (("run-a", v, w_) not in fit or
                    ("run-b", v, w_) not in fit):
                continue
            s = sum(fit[(rn, v, w_)][0]
                    for rn in ("run-a", "run-b")) / 2
            c = sum(fit[(rn, v, w_)][1]
                    for rn in ("run-a", "run-b")) / 2
            resid = max(fit[(rn, v, w_)][2]
                        for rn in ("run-a", "run-b"))
            sa, sb = fit[("run-a", v, w_)][0], fit[("run-b", v, w_)][0]
            agree = abs(sa - sb) <= 0.05 * max(1.0, abs(s))
            slopes[(v, w_)] = s
            ok = ok and resid <= 0.05 and agree
            print(f"{v:<12}{w_:>2}{s:9.3f}{c:8.2f}"
                  f"{resid*100:7.1f}%"
                  f"{'ok' if (resid <= 0.05 and agree) else 'FAIL':>7}")
    costs = {}
    for w_ in (1, 2, 4):
        if ("spin", w_) in slopes and ("nospin", w_) in slopes:
            costs[w_] = (slopes[("spin", w_)]
                         - slopes[("nospin", w_)]
                         - (SYNCS_RT - ISETP_RT))
    print("cost_yield(W):",
          {w: round(c, 2) for w, c in costs.items()})
    if not costs:
        decision, cost = "none", None
    else:
        mean_c = sum(costs.values()) / len(costs)
        spread = max(abs(c - mean_c)
                     for c in costs.values()) / max(1e-9, mean_c)
        if mean_c <= 2:
            decision, cost = "free", None
        elif spread <= 0.20:
            decision, cost = "install", round(mean_c, 2)
        else:
            decision, cost = "report_only", round(mean_c, 2)
    out = {"device": dev, "run": run,
           "slopes": {f"{v}.w{w}": s for (v, w), s in slopes.items()},
           "cost_yield": costs, "decision": decision,
           "install_value": cost, "fit_ok": ok}
    mdir = Path(a) / "microbench" / "control_yield-model" / f"{dev}-{run}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.json").write_text(json.dumps(out, indent=1))
    print(f"decision {decision}: cost_yield={cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
