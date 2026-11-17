#!/usr/bin/env python3
"""model for nanosleep protocol v1.

syncs_wake_cycles = median over D of delta per H; constant if per-H
medians within 20 % else knots over H; plain: OLS affine over H.
Usability: p10-p90 <= 30 % of median per cell, runs <= 10 %.
Usage: model.py <A> <dev> <run>
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HS = [0, 200, 1000, 10000, 100000, 1000000]
DS = [2000, 20000, 200000]


def ols(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    s = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c = (sy - s * sx) / n
    resid = statistics.pstdev([y - (c + s * x) for x, y in zip(xs, ys)])
    return s, c, resid / max(1e-9, abs(sy / n))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def bins16(xs):
    from collections import Counter
    c = Counter(int(x) // 16 for x in xs)
    return c.most_common(1)[0]


def main() -> int:
    v2 = "--v2" in sys.argv
    argv = [x for x in sys.argv if x != "--v2"]
    a, dev, run = argv[1], argv[2], argv[3]
    if v2:
        return main_v2(a, dev, run)


def main_v2(a, dev, run) -> int:
    base = Path(a) / "microbench" / "nanosleep" / f"{dev}-{run}"
    rows = defaultdict(list)
    khz = None
    for rd in sorted(base.glob("run-*")):
        if not rd.is_dir():
            continue
        for ln in (rd / "nanosleep.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if "error" in r or not r.get("samples"):
                continue
            if khz is None:
                khz = r.get("sm_clock_khz")
            p = r["params"]
            rows[(rd.name, p["variant"], p["h"],
                  p.get("d", -1))].extend(r["samples"])
    print(f"{dev}-{run}: nanosleep v2 (clock {khz} khz)")
    # per-cell dominant 16-cycle bin per run; usable = >=40% and
    # runs' modes within one bin
    cell_mode = {}
    unusable = []
    for h in HS:
        for d in DS:
            modes = {}
            for rn in ("run-a", "run-b"):
                s = rows.get((rn, "syncs", h, d), [])
                if s:
                    b, n = bins16(s)
                    modes[rn] = (b, n / len(s))
            if len(modes) != 2:
                unusable.append((h, d, "missing"))
                continue
            (ba, fa), (bb, fb) = modes["run-a"], modes["run-b"]
            if fa < 0.40 or fb < 0.40 or abs(ba - bb) > 1:
                unusable.append((h, d,
                    f"fa={fa:.2f} fb={fb:.2f} db={abs(ba-bb)}"))
                continue
            cell_mode[(h, d)] = statistics.median([ba, bb]) * 16 + 8
    print("syncs usable cells:")
    for (h, d), m in sorted(cell_mode.items()):
        print(f"  H={h:>8} D={d:>7} mode~{m:.0f}")
    for u in unusable:
        print(f"  unusable {u}")
    per_h = {}
    for h in HS:
        vs = [m for (hh, _), m in cell_mode.items() if hh == h]
        if vs:
            per_h[h] = statistics.median(vs)
    const = False
    syncs_val = None
    evidence = None
    if per_h:
        syncs_val = statistics.median(per_h.values())
        const = all(abs(v - syncs_val) <= 0.20 * max(1.0, syncs_val)
                    for v in per_h.values())
        evidence = "measured" if const else "experimental"
    # plain OLS over H (cycles vs H ns)
    xs, ys = [], []
    print("plain(H):")
    for h in HS:
        s = []
        for rn in ("run-a", "run-b"):
            s += rows.get((rn, "plain", h, -1), [])
        if s:
            m = statistics.median(s)
            # protocol: x = requested_sleep_cycles = H ns * clock
            xs.append(h * (khz / 1e6) if khz else h)
            ys.append(m)
            print(f"  H={h:>8} med={m:>9.1f} cycles")
    slope, icept, resid = (ols(xs, ys) if len(xs) >= 2
                           else (1.0, 0.0, 0.0))
    runs_plain = True
    for h in HS:
        a_ = statistics.median(rows[("run-a", "plain", h, -1)])             if rows.get(("run-a", "plain", h, -1)) else None
        b_ = statistics.median(rows[("run-b", "plain", h, -1)])             if rows.get(("run-b", "plain", h, -1)) else None
        if a_ is not None and b_ is not None and                 abs(a_ - b_) > 0.10 * max(1.0, max(a_, b_)):
            runs_plain = False
    plain_ok = resid <= 0.10 and runs_plain
    decision = "measured" if (syncs_val is not None and cell_mode
                            and plain_ok) else "none"
    out = {"device": dev, "run": run, "v": 2,
           "per_h_modes": per_h, "cell_modes": cell_mode and
           {f"{h}/{d}": v for (h, d), v in cell_mode.items()},
           "unusable": unusable, "const": const,
           "syncs_wake_cycles": syncs_val if const else None,
           "syncs_knots": None if const else per_h,
           "syncs_evidence": evidence,
           "plain": {"slope": slope, "intercept": icept,
                     "resid": resid, "runs_ok": runs_plain,
                     "ok": plain_ok},
           "decision": decision}
    mdir = Path(a) / "microbench" / "nanosleep-model" / f"{dev}-{run}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.json").write_text(json.dumps(out, indent=1))
    print(f"per-H modes: {per_h}")
    print(f"syncs_wake={syncs_val} const={const} knots={None if const else per_h} evidence={evidence}")
    print(f"plain {icept:.1f}+{slope:.3f}*H resid={resid:.2%} runs={runs_plain} ok={plain_ok}")
    print(f"decision {decision}")
    return 0
    base = Path(a) / "microbench" / "nanosleep" / f"{dev}-{run}"
    rows = defaultdict(list)
    khz = None
    for rd in sorted(base.glob("run-*")):
        if not rd.is_dir():
            continue
        for ln in (rd / "nanosleep.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if "error" in r or not r.get("samples"):
                continue
            if khz is None:
                khz = r.get("sm_clock_khz")
            p = r["params"]
            key = (rd.name, p["variant"], p["h"], p.get("d", -1))
            rows[key].extend(r["samples"])
    med = {}
    spread_ok = True
    for (rn, v, h, d), s in rows.items():
        m = statistics.median(s)
        med[(rn, v, h, d)] = m
        p10, p90 = pct(s, 0.10), pct(s, 0.90)
        if m > 0 and (p90 - p10) > 0.30 * m:
            spread_ok = False
    # syncs: per (H): median delta over D, then constant-vs-knots
    print(f"{dev}-{run}: nanosleep v1  (clock {khz} khz)")
    print(f"{'H':>8} {'D':>7} {'run-a':>9} {'run-b':>9} {'med':>9}")
    deltas = defaultdict(list)
    for h in HS:
        for d in DS:
            a_ = med.get(("run-a", "syncs", h, d))
            b_ = med.get(("run-b", "syncs", h, d))
            if a_ is None and b_ is None:
                continue
            m = statistics.median([x for x in (a_, b_)
                                   if x is not None])
            deltas[h].append(m)
            ag = "" if a_ is None or b_ is None else (
                "" if abs(a_ - b_) <= 0.10 * max(1.0, m) else " R>")
            print(f"{h:>8} {d:>7} "
                  f"{a_ if a_ is not None else float('nan'):>9.1f} "
                  f"{b_ if b_ is not None else float('nan'):>9.1f} "
                  f"{m:>9.1f}{ag}")
    per_h = {h: statistics.median(v) for h, v in deltas.items() if v}
    if per_h:
        cmed = statistics.median(per_h.values())
        const = all(abs(v - cmed) <= 0.20 * max(1.0, cmed)
                    for v in per_h.values())
    else:
        cmed, const = 0.0, False
    # plain: affine over H (cycles vs ns converted by sm clock)
    print("plain(H):")
    xs, ys = [], []
    for h in HS:
        a_ = med.get(("run-a", "plain", h, -1))
        b_ = med.get(("run-b", "plain", h, -1))
        if a_ is None and b_ is None:
            continue
        m = statistics.median([x for x in (a_, b_) if x is not None])
        xs.append(h / (khz / 1e6) if khz else h)
        ys.append(m)
        print(f"  H={h:>8} med={m:>9.1f} cycles")
    if len(xs) >= 2:
        slope, icept, resid = ols(xs, ys)
    else:
        slope, icept, resid = 1.0, 0.0, 0.0
    runs_agree = all(
        abs(med.get(("run-a", v, h, d), 0) -
            med.get(("run-b", v, h, d), 0)) <=
        0.10 * max(1.0, med.get(("run-a", v, h, d), 1))
        for v in ("syncs", "plain") for h in HS for d in DS)
    # knots view: per-H medians when not constant
    knots = None
    if per_h and not const:
        knots = {h: v for h, v in per_h.items()}
    decision = "measured" if (const and resid <= 0.10 and runs_agree
                            and spread_ok) else "none"
    if const and cmed <= 0:
        decision = "none"
    out = {"device": dev, "run": run, "per_h": per_h,
           "syncs_wake_cycles": cmed if const else None,
           "const": const, "plain": {"slope": slope,
                                     "intercept": icept,
                                     "resid": resid},
           "knots": knots,
           "runs_agree": runs_agree, "spread_ok": spread_ok,
           "decision": decision}
    mdir = Path(a) / "microbench" / "nanosleep-model" / f"{dev}-{run}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.json").write_text(json.dumps(out, indent=1))
    print(f"syncs_wake_cycles={cmed:.1f} const={const} knots={knots} "
          f"plain {icept:.1f}+{slope:.3f}*H resid={resid:.2%} "
          f"runs_agree={runs_agree} spread_ok={spread_ok}")
    print(f"decision {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
