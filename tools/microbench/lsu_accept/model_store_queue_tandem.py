#!/usr/bin/env python3
"""Derivation for lsu_accept protocol v5 (store_queue probe with stg8 /
none), frozen in protocol-v5.json "derivation".

    model_store_queue_tandem.py <run dir with run-a run-b> <model dir>
        --port-bpc <installed store_l2.sector_bytes_per_cycle>

(1) clock cost c0 = median gap (k = 2..24) of none.w1; issue interval
t(S) = median gap of stg4 / stg8 / stg16 at W = 1 minus (c0 - 1); W = 2
must agree within 15 % or 1 cycle.  (2) r_b = installed port.  (3) grid
search (C_a, C_b, r_a) replaying every stg cell at W in {2, 4, 8}
through I + Q2 (two-stage per-SM byte queue) and I + Q1 (single
stage); acceptance per the protocol; writes model.json + table.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

D = 24
WS = (1, 2, 4, 8)
SECTORS = {"stg4": 4, "stg8": 8, "stg16": 16}
INF = float("inf")


def load_run(path: Path) -> dict:
    cells: dict = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or not r.get("samples"):
            continue
        p = r["params"]
        cells[(p["variant"], p["w"])][p["k"]] = statistics.median(r["samples"])
    return cells


def med_gap(cell: dict) -> float | None:
    g = [cell.get(k) for k in range(2, D + 1)]
    if any(x is None for x in g):
        return None
    return statistics.median(g)


def replay(w: int, bytes_req: float, t_issue: float, c0: float,
           ca: float, cb: float, ra: float, rb: float) -> list[list[float]]:
    """Issue times per warp (k = 1..24) under I + Q2 (ra = inf -> Q1)."""
    step = t_issue + (c0 - 1.0)        # store interval + the clock read
    nxt = [0.0] * w                    # earliest issue of each warp's next request
    cnt = [0] * w
    issues = [[] for _ in range(w)]
    q: list[tuple[float, float, float]] = []   # (bytes, t_a, t_b) in SM issue order
    while any(c < D for c in cnt):
        # stage-A room for a request of bytes_req given the current queue
        def room_a() -> float:
            acc = bytes_req
            for b, ta, _tb in reversed(q):
                acc += b
                if acc > ca:
                    return ta
            return 0.0
        ra_room = room_a() if ra != INF else 0.0
        # arbitration: earliest actual issue; ties (several warps blocked
        # on the same room) go to the warp that has waited longest
        best, bw, best_key = None, -1, None
        for i in range(w):
            if cnt[i] >= D:
                continue
            t = max(nxt[i], ra_room)
            key = (t, nxt[i], i)
            if best_key is None or key < best_key:
                best, bw, best_key = t, i, key
        issue = best
        last_ta = q[-1][1] if q else 0.0
        last_tb = q[-1][2] if q else 0.0
        if ra == INF:
            ta = issue
            # single stage: issue itself waits for stage-B room
            acc = bytes_req
            for b, _ta, tb in reversed(q):
                acc += b
                if acc > cb:
                    issue = max(issue, tb)
                    break
            ta = issue
        else:
            ta = max(issue, last_ta) + bytes_req / ra
            acc = bytes_req
            for b, _ta, tb in reversed(q):
                acc += b
                if acc > cb:
                    ta = max(ta, tb)
                    break
        tb = max(ta, last_tb) + bytes_req / rb
        q.append((bytes_req, ta, tb))
        issues[bw].append(issue)
        cnt[bw] += 1
        nxt[bw] = issue + step
    return issues


def predicted_gaps(w, bytes_req, t_issue, c0, ca, cb, ra, rb) -> list[float]:
    iss = replay(w, bytes_req, t_issue, c0, ca, cb, ra, rb)
    gaps = []
    for k in range(1, D):
        gaps.append(statistics.median(iss[i][k] - iss[i][k - 1] for i in range(w)))
    return gaps                        # k = 2..24


def score(cells, tis, c0, ca, cb, ra, rb) -> tuple[float, dict]:
    total, per = 0.0, {}
    for (v, w), cell in cells.items():
        if v not in SECTORS or w == 1:
            continue
        meas = [cell.get(k) for k in range(2, D + 1)]
        if any(x is None for x in meas):
            continue
        pred = predicted_gaps(w, SECTORS[v] * 32, tis[v], c0, ca, cb, ra, rb)
        err = [abs(p - m) for p, m in zip(pred, meas)]
        ok = sum(1 for p, m in zip(pred, meas) if abs(p - m) <= max(0.15 * m, 4.0))
        total += sum(err)
        per[f"{v}.w{w}"] = {"pred": [round(x, 1) for x in pred], "meas": meas,
                            "within": ok, "of": len(meas)}
    return total, per


def fit(cells, tis, c0, rb, single: bool) -> dict:
    ras = [INF] if single else [32.0, 48.0, 64.0, 96.0, 128.0, 256.0]
    best = None
    def consider(ca, cb, ra):
        nonlocal best
        s, per = score(cells, tis, c0, ca, cb, ra, rb)
        if best is None or s < best["score"]:
            best = {"score": s, "ca": ca, "cb": cb, "ra": ra, "cells": per}
    coarse = [512 * 4 * i for i in range(1, 17)]            # 2 KB .. 32 KB
    for ra in ras:
        for ca in ([0.0] if single else coarse):
            for cb in coarse:
                consider(ca, cb, ra)
    ca0, cb0, ra0 = best["ca"], best["cb"], best["ra"]
    fine = lambda c: [x for x in (c + 512 * i for i in range(-4, 5)) if x >= 512]
    for ca in ([0.0] if single else fine(ca0)):
        for cb in fine(cb0):
            consider(ca, cb, ra0)
    tot_w = sum(c["within"] for c in best["cells"].values())
    tot_n = sum(c["of"] for c in best["cells"].values())
    best["pass"] = bool(best["cells"]) and tot_w >= 0.85 * tot_n and all(
        c["within"] >= 0.75 * c["of"] for c in best["cells"].values())
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--port-bpc", type=float, required=True)
    a = ap.parse_args()
    a.model_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for rn in ("run-a", "run-b"):
        f = a.run_dir / rn / "store_queue.jsonl"
        if f.exists():
            runs[rn] = load_run(f)
    if len(runs) < 2:
        (a.model_dir / "model.json").write_text(json.dumps({"decision": "none", "why": "fewer than two runs"}, indent=1))
        print("fewer than two runs -> none")
        return 1
    rb = a.port_bpc
    out = {"runs": {}}
    fits = {}
    for rn, cells in runs.items():
        c0 = med_gap(cells.get(("none", 1), {}))
        if c0 is None:
            out["runs"][rn] = {"error": "no none.w1 cell"}
            continue
        tis, ti_notes, ti_ok = {}, [], True
        for v in SECTORS:
            g1, g2 = med_gap(cells.get((v, 1), {})), med_gap(cells.get((v, 2), {}))
            if g1 is None:
                ti_ok = False; ti_notes.append(f"{v}: no W=1"); continue
            tis[v] = g1 - (c0 - 1.0)
            if g2 is None or abs(g2 - g1) > max(0.15 * g1, 1.0):
                ti_ok = False; ti_notes.append(f"{v}: W=2 {g2} vs W=1 {g1}")
        res = {"c0": c0, "issue_interval_by_sectors": tis, "issue_interval_ok": ti_ok,
               "issue_notes": ti_notes}
        if len(tis) == 3:
            q2 = fit(cells, tis, c0, rb, single=False)
            q1 = fit(cells, tis, c0, rb, single=True)
            res["q2"] = q2
            res["q1"] = q1
            fits[rn] = (q2, q1)
        out["runs"][rn] = res
    decision = "none"
    fields = {}
    if len(fits) == 2:
        (q2a, q1a), (q2b, q1b) = fits["run-a"], fits["run-b"]
        agree = lambda x, y: abs(x["ca"] - y["ca"]) <= 512 and abs(x["cb"] - y["cb"]) <= 512 and x["ra"] == y["ra"]
        if q2a["pass"] and q2b["pass"] and agree(q2a, q2b):
            decision = "I + Q2"
            fields = {"store_l2.accept_bytes_per_cycle": q2a["ra"],
                      "store_l2.accept_queue_bytes": (q2a["ca"] + q2b["ca"]) / 2,
                      "store_l2.drain_queue_bytes": (q2a["cb"] + q2b["cb"]) / 2}
        elif q1a["pass"] and q1b["pass"] and abs(q1a["cb"] - q1b["cb"]) <= 512:
            decision = "I + Q1"
            fields = {"store_l2.drain_queue_bytes": (q1a["cb"] + q1b["cb"]) / 2}
    ra_ = out["runs"].get("run-a", {})
    rb_ = out["runs"].get("run-b", {})
    issue_ok = ra_.get("issue_interval_ok") and rb_.get("issue_interval_ok")
    if issue_ok:
        ti = {v: (ra_["issue_interval_by_sectors"][v] + rb_["issue_interval_by_sectors"][v]) / 2 for v in SECTORS}
        if all(abs(ra_["issue_interval_by_sectors"][v] - rb_["issue_interval_by_sectors"][v]) <= max(0.15 * ti[v], 1.0) for v in SECTORS):
            fields["storage.global.store_issue.cycles_by_sectors"] = {SECTORS[v]: ti[v] for v in SECTORS}
        else:
            issue_ok = False
    out.update({"decision": decision, "issue_interval_installable": bool(issue_ok),
                "fields": fields, "installed_port_bpc": rb})
    (a.model_dir / "model.json").write_text(json.dumps(out, indent=1, default=str))
    lines = ["| run | cell | measured gaps k=2..24 | Q2 predicted |", "|---|---|---|---|"]
    for rn, res in out["runs"].items():
        for name, c in res.get("q2", {}).get("cells", {}).items():
            lines.append(f"| {rn} | {name} | " + " ".join(f"{x:.0f}" for x in c["meas"]) + " | " +
                         " ".join(f"{x:.0f}" for x in c["pred"]) + f" ({c['within']}/{c['of']}) |")
    (a.model_dir / "table.md").write_text("\n".join(lines) + "\n")
    print(f"decision={decision} fields={json.dumps(fields, default=str)} issue_ok={issue_ok}")
    for rn, res in out["runs"].items():
        q2 = res.get("q2"); q1 = res.get("q1")
        print(f"  {rn}: c0={res.get('c0')} t={res.get('issue_interval_by_sectors')} "
              f"Q2={(q2['ca'], q2['cb'], q2['ra'], round(q2['score']), q2['pass']) if q2 else None} "
              f"Q1={(q1['cb'], round(q1['score']), q1['pass']) if q1 else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
