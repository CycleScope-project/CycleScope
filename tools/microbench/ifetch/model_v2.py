#!/usr/bin/env python3
"""instruction.fetch derivation, protocol version 2
(tools/microbench/ifetch/protocol-v2.json).  Parsing, cell statistics and
the version-1 plateau rules are imported from model.py (version 1).

Run layout (run_ncu.sh v2): <run>/{launches.log,counters.csv} (cache-control
all only), two runs per device.

Per run:
  cells            (variant, pass) -> median over launches >= 2 of the
                   slowest warp's pass cycles (model.py build_cells)
  T_p(F)           cell(line.nN.w1, pass p) - cell(line.n256.w1, pass p):
                   the extra cycles of the extra code; F = 16 N bytes,
                   lines(F) = F / 128
  capacity_l0 / l1 version-1 plateau rules on the overhead-corrected warm
                   cpi (model.py plateaus / 2 % rule), line_bytes 128
  stream_l1        OLS slope of T_warm(F) over lines(F) for
                   capacity_l0 < F <= capacity_l1 (T_warm = median of the
                   pooled pass 1..3 samples)
  stream_l2        the same for F > capacity_l1
  launch_prefetch_bytes, stream_cold
                   two-segment fit of T_0(F) in the engine form
                   P(F) = stream_l1 * min(lines, K) + stream_cold * max(0, lines - K)
                   (instruction_fetch.cpp: launch-prefetched head at the
                   stream_l1 cadence, cold lines at stream_cold), fitted on
                   the differences D(F) = T_0(F) = P(F) - P(F_min) with
                   stream_l1 fixed, stream_cold free (closed-form OLS per
                   knot), K over every measured lines(F); the knot with the
                   least squared error is launch_prefetch_bytes = 128 K
  check            |D(F) - (P(F) - P(F_min))| / D(F) at every F >= 16 KiB
Acceptance per field: slope-fit max relative residual <= 5 %, runs within
5 %, and the engine-form check within 10 % at every F >= 16 KiB ->
measured; else experimental (per-F table recorded).  The installed value
is the mean of the two runs.  sharing keeps the version-1 rule (W = 4 warm
cpi within 10 % of W = 1 -> 0); fetch_parallelism stays unrealised.

Usage: model_v2.py --run-a <dir> --run-b <dir> [--gate checks.json] --out model.json
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as v1  # noqa: E402

LINE = 128
BYTES_PER_INSTR = 16
FIT_RESIDUAL = 0.05
RUN_AGREE = 0.05
CHECK_TOL = 0.10
CHECK_MIN_BYTES = 16 * 1024
WARM_PASSES = (1, 2, 3)


def load_run(run):
    records, stats = v1.parse_launches(Path(run) / "launches.log")
    counters = v1.parse_counters(Path(run) / "counters.csv")
    ci = 0
    for rec in records:
        if ci >= len(counters):
            break
        key = v1._row_kernel_key(counters[ci]["Kernel Name"])
        if key is None:
            continue
        want = (("jump", rec["m"], rec["perm"]) if rec["kind"] == "jump"
                else ("mix", rec["n"], rec["w"]) if rec["kind"] == "mix" else ("line", rec["n"]))
        if key != want:
            continue
        for f in ("sm__cycles_active.max",):
            if f in counters[ci]:
                rec[f] = v1._int(counters[ci][f])
        ci += 1
    return records, stats


def pooled_warm(records, vkey):
    """median of the pooled pass 1..3 slowest-warp cycles (launches >= 2)."""
    vals = []
    for rec in records:
        if rec["key"] != vkey or rec["launch"] in v1.DROPPED_LAUNCHES:
            continue
        for ps in WARM_PASSES:
            if ps in rec["passes"]:
                vals.append(max(rec["passes"][ps]))
    return v1._median(vals) if vals else None


def ols(points):
    """-> slope, intercept, max relative residual (|res| / |y|, y != 0)."""
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    xm, ym = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - xm) ** 2 for x in xs)
    if den == 0:
        return None
    b = sum((x - xm) * (y - ym) for x, y in points) / den
    a = ym - b * xm
    rel = [abs(y - (a + b * x)) / abs(y) for x, y in points if y]
    res = [abs(y - (a + b * x)) for x, y in points]
    span = max(ys) - min(ys)
    return {"slope": b, "intercept": a, "residual_max_rel": max(rel) if rel else 0.0,
            "residual_max_abs": max(res), "residual_max_over_span": (max(res) / span) if span else None,
            "n_points": len(points)}


def two_segment(D, lines_min, s_l1):
    """D: lines -> T_0 difference to the smallest footprint.  Engine form
    without intercept; stream_l1 fixed; knot K over every measured lines."""
    best = None
    table = []
    for K in sorted(D):
        xs, ys = [], []
        for L, d in D.items():
            a = s_l1 * (min(L, K) - min(lines_min, K))
            x = max(0, L - K) - max(0, lines_min - K)
            xs.append(x); ys.append(d - a)
        sxx = sum(x * x for x in xs)
        s_cold = sum(x * y for x, y in zip(xs, ys)) / sxx if sxx else None
        if s_cold is None:
            continue
        sse = sum((y - s_cold * x) ** 2 for x, y in zip(xs, ys))
        table.append({"K_lines": K, "launch_prefetch_bytes": K * LINE, "stream_cold": s_cold, "sse": sse})
        if best is None or sse < best["sse"]:
            best = table[-1]
    return best, table


def predict(L, L_min, K, s_l1, s_cold):
    P = lambda x: s_l1 * min(x, K) + s_cold * max(0, x - K)
    return P(L) - P(L_min)


def derive_run(run, void_keys):
    records, stats = load_run(run)
    cells = v1.build_cells(records)
    ns = sorted({r["n"] for r in records if r["kind"] == "line" and f"line.n{r['n']}.w1" not in void_keys})
    n_min = ns[0]
    lines = {n: n * BYTES_PER_INSTR // LINE for n in ns}
    # version-1 plateau rules (overhead-corrected warm cpi, passes 2/3)
    Tw = {n: v1.warm_T(cells, f"line.n{n}.w1") for n in ns}
    Tw = {n: t for n, t in Tw.items() if t is not None}
    ns1 = sorted(Tw)
    a0, b0, _ = v1._linfit([(n, Tw[n]) for n in ns1[:3]])
    cpi = {n: (Tw[n] - a0) / n for n in ns1}
    pl = v1.plateaus(cpi, ns1)
    base = pl["base"]
    cap_l0 = max(n for n in ns1 if cpi[n] <= base * 1.02) * BYTES_PER_INSTR
    cap_l1 = (max(pl["L1"]) * BYTES_PER_INSTR) if pl["L1"] else None
    # T_p(F) relative to the smallest footprint
    T = {}
    for ps in range(v1.K_PASSES):
        ref = v1.cell_median(cells, f"line.n{n_min}.w1", ps)
        T[ps] = {n: (v1.cell_median(cells, f"line.n{n}.w1", ps) - ref)
                 for n in ns if ref is not None and v1.cell_median(cells, f"line.n{n}.w1", ps) is not None}
    warm_ref = pooled_warm(records, f"line.n{n_min}.w1")
    T_warm = {n: pooled_warm(records, f"line.n{n}.w1") - warm_ref for n in ns
              if pooled_warm(records, f"line.n{n}.w1") is not None}
    out = {"run": str(run), "parse_stats": stats, "ns": ns, "lines": lines, "a0": a0, "cpi_w1": cpi,
           "plateaus": {k: v for k, v in pl.items()}, "T_pass": {str(p): t for p, t in T.items()},
           "T_warm": T_warm, "fields": {}}
    f = out["fields"]
    f["line_bytes"] = {"value": LINE, "note": "SASS gate: block starts 128 B apart (version 1)"}
    f["capacity_l0_bytes"] = {"value": cap_l0, "note": "version 1: largest F with warm cpi within 2 % of the smallest-N cpi"}
    f["capacity_l1_bytes"] = {"value": cap_l1, "l1_plateau_ns": pl["L1"], "l2_plateau_ns": pl["L2"],
                              "transition_ns": pl["transition"]}
    # stream_l1 / stream_l2: OLS slopes of T_warm over lines
    def seg(pred):
        pts = [(lines[n], T_warm[n]) for n in ns if n in T_warm and pred(n * BYTES_PER_INSTR)]
        return ols(pts) if len(pts) >= 2 else None, pts
    s1, p1 = seg(lambda F: cap_l0 < F <= (cap_l1 or 0))
    s2, p2 = seg(lambda F: cap_l1 is not None and F > cap_l1)
    f["stream_l1_cycles_per_line"] = {"value": s1["slope"] if s1 else None, "fit": s1, "points": p1}
    f["stream_l2_cycles_per_line"] = {"value": s2["slope"] if s2 else None, "fit": s2, "points": p2}
    # launch_prefetch_bytes / stream_cold: two-segment engine-form fit of T_0
    D = {lines[n]: T[0][n] for n in ns if n in T[0] and n != n_min}
    if s1 and D:
        best, table = two_segment(D, lines[n_min], s1["slope"])
        check = {}
        for n in ns:
            if n == n_min or n not in T[0]:
                continue
            L = lines[n]; d = T[0][n]
            p = predict(L, lines[n_min], best["K_lines"], s1["slope"], best["stream_cold"])
            check[str(n * BYTES_PER_INSTR)] = {"T0": d, "pred": p, "rel": abs(d - p) / d if d else None}
        worst = max((c["rel"] for F, c in check.items() if int(F) >= CHECK_MIN_BYTES and c["rel"] is not None), default=None)
        f["launch_prefetch_bytes"] = {"value": best["launch_prefetch_bytes"], "knot_table": table}
        f["stream_cold_cycles_per_line"] = {"value": best["stream_cold"], "knot_lines": best["K_lines"],
                                            "engine_form_check": check, "check_worst_rel": worst}
    else:
        f["launch_prefetch_bytes"] = {"value": None}
        f["stream_cold_cycles_per_line"] = {"value": None}
    # sharing (version-1 rule): W = 4 warm cpi within 10 % of W = 1
    ratios, ok_all = {}, True
    for n in ns1:
        t = v1.warm_T(cells, f"line.n{n}.w4")
        if t is None:
            continue
        rw = (t - a0) / n
        ok = abs(rw - cpi[n]) <= max(0.05, 0.10 * cpi[n])
        ratios[f"n{n}"] = {"cpi_w4": rw, "cpi_w1": cpi[n], "within_10pct": ok}
        ok_all = ok_all and ok
    f["sharing"] = {"value": 0 if ok_all else 1, "ratios": ratios}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True); ap.add_argument("--run-b", required=True)
    ap.add_argument("--gate"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    void = set()
    if a.gate and Path(a.gate).exists():
        for k, r in json.loads(Path(a.gate).read_text()).get("kernels", {}).items():
            if not r.get("ok"):
                parts = k.split("/")
                if parts[1] == "line":
                    void.add(f"line.n{parts[2]}.w1")
    runs = {"a": derive_run(a.run_a, void), "b": derive_run(a.run_b, void)}
    out = {"protocol": "tools/microbench/ifetch/protocol-v2.json", "runs": runs,
           "gate_void_variants": sorted(void), "fields": {}}
    for name in ("line_bytes", "capacity_l0_bytes", "capacity_l1_bytes", "stream_l1_cycles_per_line",
                 "stream_l2_cycles_per_line", "launch_prefetch_bytes", "stream_cold_cycles_per_line", "sharing"):
        va, vb = runs["a"]["fields"][name].get("value"), runs["b"]["fields"][name].get("value")
        e = {"run_a": va, "run_b": vb}
        if va is None or vb is None:
            e.update({"value": None, "verdict": "unrealised"})
        else:
            m = (va + vb) / 2
            agree = abs(va - vb) / m if m else 0.0
            e.update({"value": m, "run_agree": agree})
            if name in ("line_bytes", "capacity_l0_bytes", "capacity_l1_bytes", "launch_prefetch_bytes", "sharing"):
                e["verdict"] = "measured" if va == vb else "experimental"
                if va != vb:
                    e["value"] = va if name != "sharing" else max(va, vb)
            else:
                fit_ok, chk_ok = True, True
                for r in runs.values():
                    fr = r["fields"][name]
                    if fr.get("fit"):
                        fit_ok = fit_ok and fr["fit"]["residual_max_rel"] <= FIT_RESIDUAL
                    if name == "stream_cold_cycles_per_line":
                        w = fr.get("check_worst_rel")
                        chk_ok = chk_ok and w is not None and w <= CHECK_TOL
                e.update({"fit_ok": fit_ok, "check_ok": chk_ok,
                          "verdict": "measured" if fit_ok and chk_ok and agree <= RUN_AGREE else "experimental"})
        out["fields"][name] = e
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, e in out["fields"].items():
        extra = "".join(f" {x}={e[x]}" for x in ("run_agree", "fit_ok", "check_ok") if x in e)
        print(f"  {k}: {e.get('value')} (a {e['run_a']}, b {e['run_b']}){extra} -> {e.get('verdict')}")
    for tag, r in runs.items():
        sc = r["fields"]["stream_cold_cycles_per_line"]
        if sc.get("engine_form_check"):
            print(f"  run {tag} engine-form check (F: T0 / pred / rel):")
            for F, c in sc["engine_form_check"].items():
                print(f"    {int(F)//1024:4d} KiB: {c['T0']:9.0f} {c['pred']:9.0f} {100*(c['rel'] or 0):6.1f}%")


if __name__ == "__main__":
    main()
