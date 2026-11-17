#!/usr/bin/env python3
"""wgmma_handoff protocol v1 derivation.

    model.py --root <run dir> --out model.json

<run dir> holds run-a/ and run-b/ handoff.jsonl and
build/sass/checks.json.  E(form, D, cell) = median over launches of the
per-launch median of t_e - t_w (rep 0 and launch 0 dropped on the host
side via per-launch rows: each jsonl row is one launch).  handoff =
E(acc) - E(ctrl).  A value is accepted when both runs agree within 5 %
or 2 cycles; |handoff| <= 2 in both runs counts as 0.  Bucket install
rule per protocol acceptance.
"""
import argparse
import json
import statistics
from pathlib import Path

AGREE_REL = 0.05
AGREE_ABS = 2.0
BUCKET_TOL_ABS = 3.0
BUCKET_TOL_REL = 0.10


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


def e_values(rows):
    """(form, d, cell) -> median over all reps' samples (per protocol:
    median over launches of per-launch medians; our rows carry all
    launches' reps flat -> per-launch medians first)."""
    per_launch = {}
    for r in rows:
        p = r["params"]
        key = (p["form"], int(p["d"]), p["cell"])
        per_launch.setdefault(key, {})[
            int(r.get("params", {}).get("launch", 0))] = r["samples"]
    out = {}
    for key, launches in per_launch.items():
        meds = [statistics.median(float(x) for x in s)
                for l, s in sorted(launches.items()) if l != 0 and s]
        if meds:
            out[key] = statistics.median(meds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    checks = {}
    cpath = Path(a.root) / "build" / "sass" / "checks.json"
    if cpath.exists():
        checks = json.loads(cpath.read_text())
    ea, eb = e_values(load(Path(a.root) / "run-a")), \
        e_values(load(Path(a.root) / "run-b"))
    keys = sorted(set(ea) | set(eb))
    table = {}
    for (form, d, cell) in keys:
        va, vb = ea.get((form, d, cell)), eb.get((form, d, cell))
        table.setdefault(form, {}).setdefault(d, {})[cell] = \
            {"run-a": va, "run-b": vb}
    out = {"schema": "mechtree.model.v1", "probe": "wgmma_handoff",
           "root": a.root,
           "sass_gate_ok": checks.get("n_ok", 0),
           "sass_gate_fail": checks.get("n_fail", 0),
           "E": {}, "handoff": {}, "buckets": {}}
    for form, ds in table.items():
        for d, cells in ds.items():
            for cell, vv in cells.items():
                out["E"][f"{form}/d{d}/{cell}"] = vv
    # handoff values + acceptance
    hand = {}
    for form, ds in table.items():
        for d in ds:
            for run_i, run in enumerate(("run-a", "run-b")):
                va = ds[d].get("acc", {}).get(run)
                vc = ds[d].get("ctrl", {}).get(run)
                h = None if va is None or vc is None else va - vc
                hand.setdefault(form, {}).setdefault(d, {})[run] = h
            ha, hb = hand[form][d].get("run-a"), hand[form][d].get("run-b")
            rec = {"run-a": ha, "run-b": hb}
            if ha is not None and hb is not None:
                if abs(ha) <= AGREE_ABS and abs(hb) <= AGREE_ABS:
                    rec["accepted"] = 0.0
                elif abs(ha - hb) <= max(AGREE_ABS,
                                         AGREE_REL * max(abs(ha), abs(hb))):
                    rec["accepted"] = (ha + hb) / 2
                else:
                    rec["accepted"] = None
                    rec["why"] = f"runs disagree {ha:.2f} vs {hb:.2f}"
            hand[form][d].update(rec)
            out["handoff"][f"{form}/d{d}"] = rec
    # bucket install rule: depths {0,1} -> bucket01, {2,3} -> bucket23
    for bname, ds in (("d01", (0, 1)), ("d23", (2, 3))):
        vals = []
        spread_note = []
        for form, dmap in hand.items():
            for d in ds:
                acc = dmap.get(d, {}).get("accepted")
                if acc is not None:
                    vals.append((form, d, acc))
                else:
                    spread_note.append(f"{form}/d{d} unaccepted")
        flat = [v for _, _, v in vals]
        rec = {"n_values": len(flat), "unaccepted": spread_note}
        if flat:
            med = statistics.median(flat)
            spread = max(abs(v - med) for v in flat)
            rec["median"] = med
            rec["max_dev_from_median"] = spread
            ok = spread <= max(BUCKET_TOL_ABS, BUCKET_TOL_REL * abs(med))
            rec["agree_within_tol"] = ok
            # per-depth disagreement check
            for d in ds:
                dv = [v for _, dd, v in vals if dd == d]
                if dv:
                    dmed = statistics.median(dv)
                    rec[f"d{d}_median"] = dmed
            if len(ds) == 2:
                m0 = statistics.median([v for _, dd, v in vals if dd == ds[0]]) \
                    if any(dd == ds[0] for _, dd, _ in vals) else None
                m1 = statistics.median([v for _, dd, v in vals if dd == ds[1]]) \
                    if any(dd == ds[1] for _, dd, _ in vals) else None
                if m0 is not None and m1 is not None:
                    rec["depth_split_needed"] = abs(m0 - m1) > max(
                        BUCKET_TOL_ABS, BUCKET_TOL_REL * abs((m0 + m1) / 2))
            rec["verdict"] = ("measured" if ok and not spread_note
                              else "experimental")
            rec["enabled"] = any(abs(v) > 0 for v in flat)
        else:
            rec["verdict"] = "unrealised"
        out["buckets"][bname] = rec
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["buckets"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
