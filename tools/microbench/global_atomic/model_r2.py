#!/usr/bin/env python3
"""global_atomic r2 model (protocol.json addendum 1, h100).

  model_r2.py --dir $A/microbench/global_atomic/h100-r2 \
              --tree configs/trees/h100.json --device h100 \
              [--r1-model $A/microbench/global_atomic-model/h100-r1/model.json] \
              --out $A/microbench/global_atomic-model/h100-r2/model.json

Six runs (a..f), three cells each (chase_l2, pingpong_addr,
atom_chain_addr).  Per the addendum:
  one_way_install_quantity = median over the pooled 96 placements;
  C1_r2: pooled medians of runs a-c vs d-f within 10 % (one_way) and
  within 5 % (dependent RT line medians); L_ld six-run medians within
  5 % and C3 unchanged;
  r2a = max(0, pooled_one_way - L_st 42 - six-run median L_ld) / f_sm.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

RUNS = list("abcdef")
L_ST = 42.0  # h100 placeholder-grade store_l2.latency (addendum text)


def median(v):
    return statistics.median(v) if v else None


def line_values(path, key):
    vals = []
    for l in Path(path).read_text().splitlines():
        if l.startswith("LINE "):
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


def hist(v, binw):
    h = {}
    for x in v:
        h[int(x // binw) * binw] = h.get(int(x // binw) * binw, 0) + 1
    n = len(v)
    modes = sorted(((c / n, b) for b, c in h.items()), reverse=True)
    return {"bin": binw, "min": min(v) if v else None,
            "max": max(v) if v else None,
            "modes": [{"bin_start": b, "share": round(s, 4)}
                      for s, b in modes]}


def service_latency(tree, op_class):
    for n in tree["nodes"]:
        if n["id"] == ("instruction.service.root/predicated_on=1"
                       "/operation_class=" + op_class):
            f = n["parameter"]["fields"].get("result_latency_cycles")
            if f is not None:
                return f["model"].get("value")
    return None


def tree_field(tree, node_suffix, field):
    for n in tree["nodes"]:
        if n["id"] == node_suffix + ".root":
            f = n["parameter"]["fields"].get(field)
            if f is not None:
                return f["model"].get("value")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--device", default="h100")
    ap.add_argument("--r1-model",
                    default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tree = json.loads(Path(a.tree).read_text())
    l2_lat = tree_field(tree, "memory.l2.service", "l2.latency_cycles")
    # addendum 2: subtract the MEMBAR.ALL.GPU floor the engine already
    # charges the issuing warp (ERRBAR/CGAERRBAR are 1 -> ignored)
    L_fence = service_latency(tree, "MEMBAR.ALL.GPU") or 0.0
    for oc in ("ERRBAR", "CGAERRBAR"):
        v = service_latency(tree, oc)
        if v and v > 1:
            L_fence += v

    runs = {}
    for r in RUNS:
        d = Path(a.dir) / f"run-{r}"
        if not d.is_dir():
            continue
        ow = line_values(d / "pingpong_addr.txt", "one_way_cycles")
        own = line_values(d / "pingpong_addr.txt", "one_way_ns")
        cy = line_values(d / "atom_chain_addr.txt", "cycles_per_link")
        ns = line_values(d / "atom_chain_addr.txt", "ns_per_link")
        runs[r] = {
            "L_ld": result_value(d / "chase_l2.txt", "cyc_per_op"),
            "one_way": ow, "one_way_ns": own,
            "one_way_median": median(ow),
            "one_way_min": min(ow) if ow else None,
            "one_way_max": max(ow) if ow else None,
            "f_sm": median([c / n for c, n in zip(ow, own) if n]),
            "chain": cy, "chain_ns": ns,
            "chain_median": median(cy),
            "f_sm_chain": median([c / n for c, n in zip(cy, ns) if n]),
        }

    pooled_ow = [x for r in runs.values() for x in r["one_way"]]
    pooled_ow_ns = [x for r in runs.values() for x in r["one_way_ns"]]
    pooled_chain = [x for r in runs.values() for x in r["chain"]]
    pooled_chain_ns = [x for r in runs.values() for x in r["chain_ns"]]
    one_way_med = median(pooled_ow)
    L_lds = [r["L_ld"] for r in runs.values() if r["L_ld"]]
    L_ld_med = median(L_lds)
    f_sm = median([r["f_sm"] for r in runs.values() if r["f_sm"]])
    f_sm_chain = median([r["f_sm_chain"]
                         for r in runs.values() if r["f_sm_chain"]])
    chain_med = median(pooled_chain)
    f_ow = median([c / n for c, n in zip(pooled_ow, pooled_ow_ns) if n])
    f_chain = median([c / n for c, n
                      in zip(pooled_chain, pooled_chain_ns) if n])
    f_sm = f_sm or f_ow
    f_sm_chain = f_sm_chain or f_chain

    r2a = (max(0.0, one_way_med - L_fence - L_ST - L_ld_med) / f_sm
           if one_way_med and L_ld_med and f_sm else None)
    r2a_sup = (max(0.0, one_way_med - L_ST - L_ld_med) / f_sm
               if one_way_med and L_ld_med and f_sm else None)
    dep_rt = chain_med / f_sm_chain if chain_med and f_sm_chain else None
    store_vis = one_way_med - L_ld_med \
        if one_way_med and L_ld_med else None
    store_vis_net = one_way_med - L_fence - L_ld_med \
        if one_way_med and L_ld_med else None

    abc = [x for r in "abc" if r in runs for x in runs[r]["one_way"]]
    def_ = [x for r in "def" if r in runs for x in runs[r]["one_way"]]
    mabc, mdef = median(abc), median(def_)
    c1_ow = (mabc and mdef and abs(mabc / mdef - 1) <= 0.10)
    cabc = [x for r in "abc" if r in runs for x in runs[r]["chain"]]
    cdef = [x for r in "def" if r in runs for x in runs[r]["chain"]]
    mcabc, mcdef = median(cabc), median(cdef)
    c1_chain = (mcabc and mcdef and abs(mcabc / mcdef - 1) <= 0.05)
    l_spread = (max(L_lds) / min(L_lds) - 1 <= 0.05) if L_lds else False
    c3 = (L_ld_med is not None and l2_lat
          and abs(L_ld_med / l2_lat - 1) <= 0.10)
    c1r2 = {"one_way_abc": mabc, "one_way_def": mdef,
            "one_way_diff_pct": (abs(mabc / mdef - 1) * 100
                                 if mabc and mdef else None),
            "one_way_pass": bool(c1_ow),
            "chain_abc": mcabc, "chain_def": mcdef,
            "chain_diff_pct": (abs(mcabc / mcdef - 1) * 100
                               if mcabc and mcdef else None),
            "chain_pass": bool(c1_chain),
            "L_ld_six_median": L_ld_med,
            "L_ld_spread_pct": ((max(L_lds) / min(L_lds) - 1) * 100
                                if L_lds else None),
            "L_ld_within_5pct": bool(l_spread),
            "C3_pass": bool(c3)}

    out = {"device": a.device, "run_dir": a.dir,
           "L_st": L_ST, "l2_latency_cycles": l2_lat,
           "per_run": {r: {k: v for k, v in q.items()
                           if k in ("L_ld", "one_way_median",
                                    "one_way_min", "one_way_max",
                                    "f_sm", "chain_median",
                                    "f_sm_chain")}
                       for r, q in runs.items()},
           "L_fence": L_fence,
           "pooled": {
               "one_way_median": one_way_med,
               "release_to_acquire_ns_superseded": r2a_sup,
               "store_visibility_cycles_net_of_fence": store_vis_net,
               "one_way_hist": hist(pooled_ow, 100),
               "store_visibility_cycles": store_vis,
               "L_ld_median": L_ld_med,
               "chain_median": chain_med,
               "chain_hist": hist(pooled_chain, 50),
               "f_sm_pingpong": f_sm, "f_sm_chain": f_sm_chain,
               "release_to_acquire_ns": r2a,
               "dependent_round_trip_ns": dep_rt},
           "C1_r2": c1r2,
           "install": {}}

    can_install = (c1r2["one_way_pass"] and c1r2["L_ld_within_5pct"]
                   and c1r2["C3_pass"])
    out["install"]["release_to_acquire_ns"] = {
        "value": r2a, "install": bool(can_install and r2a is not None),
        "evidence": "experimental",
        "note": "home-partition mixture (per-run medians "
                f"{[round(r['one_way_median'],1) for r in runs.values()]}), "
                "pooled median over 96 placements; residual over "
                "placeholder-grade store_l2.latency 42"}
    out["install"]["dependent_round_trip_ns"] = {
        "value": dep_rt,
        "install": bool(c1r2["chain_pass"] and dep_rt is not None),
        "evidence": "experimental",
        "note": "pooled 64x6 line median / f_sm, address-bimodal, "
                "no address axis"}
    sas = None
    if a.r1_model and Path(a.r1_model).is_file():
        m1 = json.loads(Path(a.r1_model).read_text())
        sas = m1["quantities"]["same_address_service_ns"].get("mean")
    out["install"]["same_address_service_ns"] = {
        "value": sas, "install": sas is not None,
        "evidence": "measured",
        "note": "from r1 (C0, C1, C2 passed there); two-run mean"}
    if not can_install:
        out["install_note"] = "C1_r2 or C3 fails: nothing installed; " \
            "the 0 placeholder stays with the r1+r2 facts in the source"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(c1r2, indent=1))
    print(json.dumps({"r2a": r2a, "dep_rt": dep_rt, "sas": sas,
                      "install": {k: v["install"]
                                  for k, v in out["install"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
