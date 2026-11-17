#!/usr/bin/env python3
"""tma_cold derivation (tools/microbench/tma_cold/protocol.json).

Reads the probe jsonl of two runs (run a / run b) and writes model.json
with, per dc1 tile box: latency(hot), latency(cold), cold_extra,
service_hot, the per-cell acceptance verdicts and the knot lists ready
for the leaf.  The protocol text is authoritative; this file only
implements it.
"""
import argparse, json, statistics, math
from pathlib import Path

BOXES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
CELL_SE = 0.03         # se / latency
RUN_AGREE = 0.03       # |a - b| / mean
HOT_CONTROL = 0.10     # vs the installed knot
COLD_SANITY = 0.25     # cold_extra <= (1 + 0.25) x (dram - l2)


def size_tag(b):
    return f"b{b}" if b % 1024 else f"b{b // 1024}k"


def load_run(run_dir):
    """variant -> u -> list of samples (k rows and .issue rows)."""
    rows = {}
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            v, u, s = d.get("variant"), d.get("u"), d.get("samples")
            if v is None or u is None or not s:
                continue
            rows.setdefault(v, {}).setdefault(int(u), []).extend(
                float(x) for x in s)
    return rows


def mean(xs):
    return sum(xs) / len(xs)


def se(xs):
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("inf")


def slope(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def derive(rows):
    """per box: latency hot/cold, cold_extra, service_hot, se."""
    out = {}
    hot0 = rows.get("tma.dc1.b16k.hot", {}).get(0)
    hot0_issue = rows.get("tma.dc1.b16k.hot.issue", {}).get(0)
    if not hot0 or not hot0_issue:
        raise SystemExit("missing the U=0 hot control (tma.dc1.b16k.hot u=0)")
    e0 = mean(hot0) - mean(hot0_issue)
    for b in BOXES:
        tag = size_tag(b)
        cell = {"box_bytes": b}
        lat = {}
        for temp in ("hot", "cold"):
            v = f"tma.dc1.{tag}.{temp}"
            k1 = rows.get(v, {}).get(1); i1 = rows.get(v + ".issue", {}).get(1)
            if not k1 or not i1:
                cell[temp] = {"missing": True}
                continue
            l = (mean(k1) - mean(i1)) - e0
            lat[temp] = l
            cell[temp] = {"latency": l, "se": se(k1), "n": len(k1),
                          "se_rel": se(k1) / l if l > 0 else None,
                          "means_by_u": {str(u): mean(s) for u, s in sorted(rows[v].items())}}
        if "hot" in lat and "cold" in lat:
            cell["cold_extra"] = lat["cold"] - lat["hot"]
        hv = rows.get(f"tma.dc1.{tag}.hot", {})
        pts = [(u, mean(s)) for u, s in sorted(hv.items()) if u >= 1]
        cell["service_hot"] = slope(pts) if len(pts) >= 2 else None
        out[str(b)] = cell
    return out, e0


def installed_knots(tree_path, mechanism):
    """knots of the UTMALDG tile dc1 leaf of `mechanism` in a generated tree."""
    t = json.loads(Path(tree_path).read_text())
    for n in t["nodes"]:
        if (n["id"].startswith(mechanism) and "UTMALDG" in n["id"]
                and "tma_form=tile" in n["id"] and "destination_count=0-1" in n["id"]
                and n.get("parameter")):
            m = n["parameter"]["fields"]["cycles"]["model"]
            if m["kind"] == "linear":
                return [tuple(p) for p in m["points"]]
            if m["kind"] == "affine":
                return [(x, m["intercept"] + m["slope"] * x) for x in BOXES]
            if m["kind"] == "constant":
                return [(x, m["value"]) for x in BOXES]
    return None


def interp(knots, x):
    if not knots:
        return None
    knots = sorted(knots)
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--tree", help="generated tree for the hot control (b300)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dram_minus_l2 = {"b300": 874.9 - 320.726562, "h100": 697.9 - 305.5}[a.device]
    runs = {"a": load_run(a.run_a), "b": load_run(a.run_b)}
    der = {k: derive(r)[0] for k, r in runs.items()}
    inst_c = installed_knots(a.tree, "storage.tma.completion") if a.tree else None
    inst_s = installed_knots(a.tree, "storage.async.service_start") if a.tree else None
    cells, hot_knots, cold_knots = {}, [], []
    for b in BOXES:
        ca, cb = der["a"][str(b)], der["b"][str(b)]
        c = {"box_bytes": b, "run_a": ca, "run_b": cb, "verdict": {}}
        def agree(x, y):
            m = (x + y) / 2
            return abs(x - y) / m if m else float("inf")
        ok_hot = ("latency" in ca.get("hot", {}) and "latency" in cb.get("hot", {}))
        if ok_hot:
            la, lb = ca["hot"]["latency"], cb["hot"]["latency"]
            c["hot_latency"] = (la + lb) / 2
            c["hot_run_agree"] = agree(la, lb)
            c["verdict"]["hot"] = (
                "measured" if ca["hot"]["se_rel"] <= CELL_SE and cb["hot"]["se_rel"] <= CELL_SE
                and c["hot_run_agree"] <= RUN_AGREE else "experimental")
        if "cold_extra" in ca and "cold_extra" in cb:
            xa, xb = ca["cold_extra"], cb["cold_extra"]
            c["cold_extra"] = (xa + xb) / 2
            c["cold_run_agree"] = agree(xa, xb)
            v = ("measured" if ca["cold"]["se_rel"] <= CELL_SE and cb["cold"]["se_rel"] <= CELL_SE
                 and c["cold_run_agree"] <= RUN_AGREE else "experimental")
            sane = 0 < c["cold_extra"] <= (1 + COLD_SANITY) * dram_minus_l2
            c["cold_sanity_ok"] = sane
            if not sane:
                v = "experimental"
            c["verdict"]["cold_extra"] = v
            cold_knots.append([b, round(c["cold_extra"], 3), v])
        if ok_hot:
            sv = [x for x in (ca.get("service_hot"), cb.get("service_hot")) if x is not None]
            c["service_hot"] = mean(sv) if sv else None
            if inst_c is not None:
                ref = interp(inst_c, b)
                c["hot_control"] = {"installed": ref,
                                    "diff": abs(c["hot_latency"] - ref) / ref if ref else None}
                c["hot_control"]["ok"] = c["hot_control"]["diff"] is not None and c["hot_control"]["diff"] <= HOT_CONTROL
                if inst_s is not None and c["service_hot"] is not None:
                    refs = interp(inst_s, b)
                    c["service_control"] = {"installed": refs,
                                            "diff": abs(c["service_hot"] - refs) / refs if refs else None}
                    c["service_control"]["ok"] = c["service_control"]["diff"] is not None and c["service_control"]["diff"] <= HOT_CONTROL
                install_hot = c["hot_control"]["ok"]
            else:
                install_hot = True
            if install_hot:
                hot_knots.append([b, round(c["hot_latency"], 3), c["verdict"]["hot"]])
        cells[str(b)] = c
    out = {"protocol": "tools/microbench/tma_cold/protocol.json", "device": a.device,
           "runs": {"a": a.run_a, "b": a.run_b}, "dram_minus_l2": dram_minus_l2,
           "cells": cells,
           "hot_knots": hot_knots, "cold_extra_knots": cold_knots}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for b, c in cells.items():
        print(f"  {b:>6} B: hot {c.get('hot_latency', float('nan')):7.1f} "
              f"({c['verdict'].get('hot')}) cold_extra {c.get('cold_extra', float('nan')):7.1f} "
              f"({c['verdict'].get('cold_extra')}) svc {c.get('service_hot') or float('nan'):6.1f} "
              f"ctl {json.dumps(c.get('hot_control', {}).get('ok'))}")


if __name__ == "__main__":
    main()
