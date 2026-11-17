#!/usr/bin/env python3
"""global_atomic model (protocol.json derivation / acceptance / install).

  model.py --dir $A/microbench/global_atomic/<device>-r1 \
           --tree configs/trees/<device>.json --device h100|b300 \
           --out $A/microbench/global_atomic-model/<device>-r1/model.json

Reads run-a / run-b cell outputs, derives every quantity per run and
the two-run mean, applies C0-C4, and emits per-field installable flags
and evidence per the frozen install table.  Installs nothing.
"""
import argparse
import glob
import json
import re
import statistics
from pathlib import Path


def median(v):
    return statistics.median(v) if v else None


def cv(v):
    if len(v) < 2:
        return 0.0
    m = statistics.mean(v)
    return (statistics.stdev(v) / m) if m else 0.0


def trial_values(path, key):
    """TRIAL <t> ... key=<float> per trial."""
    vals = []
    for l in Path(path).read_text().splitlines():
        if l.startswith("TRIAL "):
            m = re.search(key + r"=([0-9.eE+-]+)", l)
            if m:
                vals.append(float(m.group(1)))
    return vals


def result_value(path, key):
    for l in Path(path).read_text().splitlines():
        if l.startswith("RESULT "):
            m = re.search(key + r"=([0-9.eE+-]+)", l)
            if m:
                return float(m.group(1))
    return None


def line_values(path, key):
    vals = []
    for l in Path(path).read_text().splitlines():
        if l.startswith("LINE "):
            m = re.search(key + r"=([0-9.eE+-]+)", l)
            if m:
                vals.append(float(m.group(1)))
    return vals


def placement_ok(path):
    for l in Path(path).read_text().splitlines():
        if l.startswith("PLACEMENT "):
            d = dict(t.split("=") for t in l[10:].split())
            return int(d["distinct_sm"]) == int(d["nsm"])
    return None  # no placement record emitted


def hist(v, binw):
    h = {}
    for x in v:
        b = int(x // binw) * binw
        h[b] = h.get(b, 0) + 1
    n = len(v)
    modes = sorted(((c / n, b) for b, c in h.items()), reverse=True)
    return {"bin": binw,
            "modes": [{"bin_start": b, "share": round(s, 4)}
                      for s, b in modes],
            "min": min(v) if v else None,
            "max": max(v) if v else None}


def tree_field(tree, node_suffix, field):
    for n in tree["nodes"]:
        if n["id"] == node_suffix + ".root":
            f = n["parameter"]["fields"].get(field)
            if f is not None:
                return f["model"].get("value")
    return None


def service_latency(tree, op_class):
    """instruction.service result_latency for operation_class under
    predicated_on=1 (addendum 2: the release form's fence leaf)."""
    for n in tree["nodes"]:
        if n["id"] == ("instruction.service.root/predicated_on=1"
                       "/operation_class=" + op_class):
            f = n["parameter"]["fields"].get("result_latency_cycles")
            if f is not None:
                return f["model"].get("value")
    return None


def run_quantities(rdir):
    q = {}
    # chase_l2 -> L_ld; CV over the per-trial cyc_per_op_med
    p = f"{rdir}/chase_l2.txt"
    q["L_ld"] = result_value(p, "cyc_per_op")
    q["chase_l2_trial_cv"] = cv(trial_values(p, "cyc_per_op_med"))
    q["chase_l2_trials"] = trial_values(p, "cyc_per_op_med")
    # pingpong_addr -> one_way median over the 16 placements + histogram
    p = f"{rdir}/pingpong_addr.txt"
    ow = line_values(p, "one_way_cycles")
    own = line_values(p, "one_way_ns")
    q["one_way_cycles"] = median(ow)
    q["one_way_hist"] = hist(ow, 100)
    q["f_sm_pingpong"] = median([c / n for c, n in zip(ow, own) if n])
    q["pingpong_placement_cv"] = None  # per-trial values not emitted
    # atom_chain_addr -> dependent RT median over 64 lines + histogram
    p = f"{rdir}/atom_chain_addr.txt"
    cy = line_values(p, "cycles_per_link")
    ns = line_values(p, "ns_per_link")
    q["chain_cycles_per_link"] = median(cy)
    q["chain_hist"] = hist(cy, 50)
    q["f_sm_chain"] = median([c / n for c, n in zip(cy, ns) if n])
    q["chain_line_cv"] = None  # per-trial values not emitted
    # red_same per nsm -> plateau
    red = {}
    for f in sorted(glob.glob(f"{rdir}/red_same.n*.txt"),
                    key=lambda s: int(re.search(r"\.n(\d+)", s).group(1))):
        nsm = int(re.search(r"\.n(\d+)", f).group(1))
        red[nsm] = {
            "agg_ops_per_ns": result_value(f, "agg_ops_per_ns"),
            "trial_cv": cv(trial_values(f, "agg_ops_per_ns")),
            "trials": trial_values(f, "agg_ops_per_ns"),
            "placement_ok": placement_ok(f),
        }
    q["red_same"] = red
    pts = [(n, r["agg_ops_per_ns"]) for n, r in red.items()
           if r["agg_ops_per_ns"]]
    if pts:
        mx = max(v for _, v in pts)
        near = [v for _, v in pts if v >= 0.97 * mx]
        q["same_address_plateau"] = median(near) if len(near) >= 3 else None
        q["red_same_max"] = mx
        q["red_same_full_nsm"] = max(n for n, _ in pts)
        q["red_same_at_full"] = dict(pts).get(q["red_same_full_nsm"])
    # atom_same at 1 and full nsm (diagnostic for C2)
    atom = {}
    for f in sorted(glob.glob(f"{rdir}/atom_same.n*.txt"),
                    key=lambda s: int(re.search(r"\.n(\d+)", s).group(1))):
        nsm = int(re.search(r"\.n(\d+)", f).group(1))
        atom[nsm] = {"agg_ops_per_ns": result_value(f, "agg_ops_per_ns"),
                     "trial_cv": cv(trial_values(f, "agg_ops_per_ns")),
                     "placement_ok": placement_ok(f)}
    q["atom_same"] = atom
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tree = json.loads(Path(a.tree).read_text())
    L_st = tree_field(tree, "memory.store_l2.service",
                      "store_l2.latency_cycles")
    # addendum 2: st.release.gpu compiles to MEMBAR.ALL.GPU;ERRBAR;
    # CGAERRBAR;STG.E.STRONG.GPU and the engine floors the issuing
    # warp's next issue by the MEMBAR result latency -> subtract it
    L_fence = service_latency(tree, "MEMBAR.ALL.GPU") or 0.0
    errbar = service_latency(tree, "ERRBAR")
    cgaerrbar = service_latency(tree, "CGAERRBAR")
    for v in (errbar, cgaerrbar):
        if v and v > 1:
            L_fence += v
    l2_lat = tree_field(tree, "memory.l2.service", "l2.latency_cycles")
    ga = {f: tree_field(tree, "memory.global_atomic", f) for f in
          ("release_to_acquire_ns", "dependent_round_trip_ns",
           "same_address_service_ns")}

    runs = {}
    for r in ("a", "b"):
        d = Path(a.dir) / f"run-{r}"
        runs[r] = run_quantities(d) if d.is_dir() else None

    out = {"device": a.device, "run_dir": a.dir,
           "L_st": L_st, "L_fence": L_fence,
           "l2_latency_cycles": l2_lat,
           "installed": ga, "runs": {}, "quantities": {},
           "acceptance": {}, "install": {}}
    for r, q in runs.items():
        if q is None:
            continue
        rq = dict(q)
        # derived quantities per the frozen derivation
        if rq["one_way_cycles"] is not None and rq["L_ld"] is not None:
            rq["store_visibility_cycles"] = \
                rq["one_way_cycles"] - rq["L_ld"]
            rq["release_to_acquire_ns"] = max(
                0.0, rq["one_way_cycles"] - L_fence - L_st
                - rq["L_ld"]) / rq["f_sm_pingpong"]
            rq["release_to_acquire_ns_superseded"] = max(
                0.0, rq["one_way_cycles"] - L_st - rq["L_ld"]) \
                / rq["f_sm_pingpong"]
        if rq["chain_cycles_per_link"] is not None:
            rq["dependent_round_trip_ns"] = \
                rq["chain_cycles_per_link"] / rq["f_sm_chain"]
        if rq.get("same_address_plateau"):
            rq["same_address_service_ns"] = \
                1.0 / rq["same_address_plateau"]
        out["runs"][r] = rq

    qa, qb = out["runs"].get("a"), out["runs"].get("b")
    key_q = ["L_ld", "one_way_cycles", "chain_cycles_per_link",
             "release_to_acquire_ns", "dependent_round_trip_ns",
             "same_address_service_ns", "same_address_plateau"]
    for k in key_q:
        vals = {r: out["runs"][r].get(k) for r in ("a", "b")
                if r in out["runs"] and out["runs"][r].get(k) is not None}
        e = {"per_run": vals}
        if len(vals) == 2:
            va, vb = vals["a"], vals["b"]
            e["mean"] = (va + vb) / 2
            e["run_diff_pct"] = abs(va - vb) / e["mean"] * 100 \
                if e["mean"] else 0.0
        out["quantities"][k] = e

    acc = {}
    # C0: placement records where the probe emits them
    c0 = {"cells": {}}
    ok = True
    for r in ("a", "b"):
        if r not in out["runs"]:
            continue
        for form in ("red_same", "atom_same"):
            for nsm, cell in out["runs"][r][form].items():
                p = cell["placement_ok"]
                c0["cells"][f"{form}.n{nsm}.{r}"] = p
                if p is False:
                    ok = False
    c0["pingpong_addr"] = "not emitted by probe (2-CTA form, no " \
        "PLACEMENT line); peer/cta0 smids not in stdout"
    acc["C0_placement"] = {"pass": ok, "detail": c0}
    # C1: two-run agreement <= 5 % on installable quantities; trial CV
    c1q = {}
    ok = True
    for k in ("release_to_acquire_ns", "dependent_round_trip_ns",
              "same_address_service_ns"):
        e = out["quantities"].get(k, {})
        if "run_diff_pct" in e:
            c1q[k] = e["run_diff_pct"]
            if e["run_diff_pct"] > 5:
                ok = False
        else:
            ok = False
            c1q[k] = "missing"
    cvs = {}
    cv_bad = []
    for r in ("a", "b"):
        if r not in out["runs"]:
            continue
        q = out["runs"][r]
        cvs[f"chase_l2.{r}"] = q["chase_l2_trial_cv"]
        if q["chase_l2_trial_cv"] > 0.03:
            cv_bad.append(f"chase_l2.{r}")
        for nsm, cell in q["red_same"].items():
            cvs[f"red_same.n{nsm}.{r}"] = cell["trial_cv"]
            if cell["trial_cv"] > 0.03:
                cv_bad.append(f"red_same.n{nsm}.{r}")
        cvs[f"pingpong_addr.{r}"] = "per-trial values not emitted " \
            "(probe prints per-placement medians only)"
        cvs[f"atom_chain_addr.{r}"] = "per-trial values not emitted " \
            "(probe prints per-line medians only)"
    acc["C1_runs"] = {"pass": ok and not cv_bad,
                      "run_diff_pct": c1q, "trial_cv": cvs,
                      "cv_violations": cv_bad,
                      "note": "pingpong_addr / atom_chain_addr per-trial "
                              "CV is not measurable: the probe emits only "
                              "per-placement / per-line medians"}
    # C2: plateau + atom_same within 10 % of red_same at full nsm
    c2 = {}
    for r in ("a", "b"):
        if r not in out["runs"]:
            continue
        q = out["runs"][r]
        plat = q.get("same_address_plateau")
        full = q.get("red_same_full_nsm")
        at = q["red_same"].get(full, {}).get("agg_ops_per_ns")
        aa = q["atom_same"].get(full, {}).get("agg_ops_per_ns")
        c2[r] = {"plateau": plat,
                 "plateau_points_ge_97pct_max":
                     sum(1 for n, c in q["red_same"].items()
                         if c["agg_ops_per_ns"]
                         and c["agg_ops_per_ns"] >= 0.97 * q["red_same_max"]),
                 "atom_full": aa, "red_full": at,
                 "atom_within_10pct": (
                     abs(aa / at - 1) <= 0.10 if aa and at else None)}
    acc["C2_plateau"] = {
        "pass": all(c2[r]["plateau"] is not None for r in c2),
        "atom_same_report": {r: c2[r]["atom_within_10pct"] for r in c2},
        "detail": c2}
    # C3: L_ld within 10 % of installed l2.latency_cycles, both runs
    c3 = {}
    ok = True
    for r in ("a", "b"):
        if r not in out["runs"]:
            continue
        v = out["runs"][r]["L_ld"]
        c3[r] = {"L_ld": v, "installed": l2_lat,
                 "within_10pct": (v is not None
                                  and abs(v / l2_lat - 1) <= 0.10)}
        if not c3[r]["within_10pct"]:
            ok = False
    acc["C3_control"] = {"pass": ok, "detail": c3}
    # C4: one_way max/min <= 1.25 over the 16 placements
    c4 = {}
    for r in ("a", "b"):
        if r not in out["runs"]:
            continue
        h = out["runs"][r]["one_way_hist"]
        c4[r] = {"max_over_min": (h["max"] / h["min"]
                                 if h["min"] else None),
                 "single_mode": (h["max"] / h["min"] <= 1.25
                                 if h["min"] else None)}
    acc["C4_placement_spread"] = {
        "pass": all(c["single_mode"] for c in c4.values()),
        "detail": c4,
        "note": "on failure the median is still installed; the source "
                "records 'home-partition bimodal, median installed, no "
                "address axis in the leaf'"}
    out["acceptance"] = acc

    def holds(*names):
        return all(acc[n]["pass"] for n in names)

    q = out["quantities"]
    inst = {}
    if a.device == "h100":
        inst["release_to_acquire_ns"] = {
            "value": q["release_to_acquire_ns"].get("mean"),
            "install": holds("C0_placement", "C1_runs", "C3_control"),
            "evidence": "experimental",
            "note": "residual over placeholder-grade store_l2.latency "
                    "42; bimodal by placement (see C4); replaces the 0 "
                    "placeholder hopper.placeholder.release_to_acquire"}
        inst["dependent_round_trip_ns"] = {
            "value": q["dependent_round_trip_ns"].get("mean"),
            "install": holds("C0_placement", "C1_runs"),
            "evidence": "experimental",
            "note": "address-bimodal median, no address axis; compares "
                    "to hopper-prototype 150.21"}
        c2ok = acc["C2_plateau"]["pass"]
        inst["same_address_service_ns"] = {
            "value": q["same_address_service_ns"].get("mean"),
            "install": holds("C0_placement", "C1_runs") and c2ok,
            "evidence": "measured" if c2ok else "experimental",
            "note": "plateau" if c2ok else
                    "plateau not reached -> full-nsm value, experimental"}
        if not c2ok:
            full = out["runs"]["a"].get("red_same_full_nsm")
            vals = {r: 1.0 / out["runs"][r]["red_same_at_full"]
                    for r in ("a", "b")
                    if out["runs"].get(r, {}).get("red_same_at_full")}
            if vals:
                inst["same_address_service_ns"]["value"] = \
                    sum(vals.values()) / len(vals)
    else:
        inst["release_to_acquire_ns"] = {
            "value": q["release_to_acquire_ns"].get("mean"),
            "install": holds("C0_placement", "C1_runs", "C3_control"),
            "evidence": "experimental",
            "note": "same-run residual; replaces "
                    "blackwell.derived.release_to_acquire_store_accept "
                    "(m3 one-way 1425.68 compared in the source text)"}
        for f, installed in (("dependent_round_trip_ns", 247.055),
                             ("same_address_service_ns", 0.019927454103336805)):
            v = q[f].get("mean")
            within = (v is not None and abs(v / installed - 1) <= 0.10)
            inst[f] = {"value": v, "install": False,
                       "control_within_10pct": within,
                       "note": "control: kept installed " + str(installed)
                       if within else
                       "control: outside 10% -> reported, not replaced "
                       "(installed value is a two-card measurement, this "
                       "run is one card)"}
    out["install"] = inst
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["pass"] for k, v in acc.items()}, indent=1))
    print(json.dumps({k: (v.get("value"), v.get("install"))
                      for k, v in inst.items()}, indent=1))


if __name__ == "__main__":
    main()
