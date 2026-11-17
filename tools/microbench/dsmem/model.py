#!/usr/bin/env python3
"""dsmem model (protocol.json): report-only latencies.

  peer_load_latency: OLS slope over U for each of peer / own_window /
      lds chains, two runs; report-only.
  peer_store_visibility: E(S) per run; visibility excess =
      E(S) - E(0) - S x issue_per_store where issue_per_store comes from
      the S=1 / S=4 difference; recorded, not installed.

  model.py --root <run dir> --out model.json [--checks gate.json]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def load(run_dir):
    rows = []
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def series(rows, var):
    out = {}
    for r in rows:
        v = r.get("params", {}).get("variant") or r.get("variant") or ""
        v = str(v)
        if not v.startswith(f"lat.{var}") and v != var:
            continue
        out.setdefault(int(r["u"]), []).extend(
            float(x) for x in r["samples"])
    return out


def vis_series(rows):
    out = {}
    for r in rows:
        v = str(r.get("variant") or "")
        if v != "vis.store":
            continue
        out.setdefault(int(r["u"]), []).extend(
            float(x) for x in r["samples"])
    return out


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) \
        / sum((x - mx) ** 2 for x in xs)
    return b, my - b * mx


def slope(series):
    us = sorted(series)
    if len(us) < 2:
        return None
    meds = [statistics.median(series[u]) for u in us]
    b, a = ols(us, meds)
    resid = max(abs(m - (a + b * u)) / max(1.0, abs(m))
                for u, m in zip(us, meds))
    return {"slope": b, "intercept": a, "us": us, "medians": meds,
            "max_rel_resid": resid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    checks = None
    cp = (Path(a.checks) if a.checks else
          root / "build" / "sass" / "checks.json")
    if cp.is_file():
        checks = json.loads(cp.read_text())
    out = {"protocol": "tools/microbench/dsmem/protocol.json",
           "gate": checks, "chains": {}, "visibility": {}}
    for var in ("peer", "own_window", "lds"):
        e = {}
        for run in ("run-a", "run-b"):
            s = slope(series(load(root / run), var))
            e[run] = s
        if e["run-a"] and e["run-b"]:
            m = (e["run-a"]["slope"] + e["run-b"]["slope"]) / 2
            e["latency_cycles"] = m
            e["agree"] = abs(e["run-a"]["slope"] - e["run-b"]["slope"]) / m
        out["chains"][var] = e
    for run in ("run-a", "run-b"):
        vs = vis_series(load(root / run))
        E = {s: (statistics.median(v) if v else None)
             for s, v in vs.items()}
        e = {"E": E}
        if all(E.get(s) is not None for s in (0, 1, 4)):
            issue = (E[4] - E[0] - (E[1] - E[0])) / 3 \
                if E[1] != E[4] else 0.0
            issue = (E[4] - E[1]) / 3
            e["issue_per_store"] = issue
            e["excess"] = {s: E[s] - E[0] - s * issue
                           for s in (0, 1, 4)}
        out["visibility"][run] = e
    Path(a.out).write_text(json.dumps(out, indent=1))
    for var, e in out["chains"].items():
        print(var, e.get("latency_cycles"),
              "agree", e.get("agree"))
    print(json.dumps(out["visibility"], indent=1)[:800])


if __name__ == "__main__":
    sys.exit(main())
