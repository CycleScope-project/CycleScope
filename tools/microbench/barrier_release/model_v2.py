#!/usr/bin/env python3
"""barrier_release derivation, protocol version 2
(tools/microbench/barrier_release/protocol-v2.json + addendum 2026-11-05).

Per run (model.py derive(): E - o_s - o_w - r_sync for mbarrier.release,
cp_async.arrival from the async probe) plus, from sync_bar + checks_bar.json:
  cta.release(W)     = E[bar.cta.wW.warplast] - o_s - o_w,   W in {2..32}
  cluster.release(S) = E[bar.cluster.cS.relaxed.arriver] - o_s - o_w,
                       S in {2, 4, 8}; the .release form is a diagnostic
                       (fence cost = release - relaxed)
sync_xw c1 / c2 are tcgen05.commit cells (tensor.umma.commit), reported
under that name.

Acceptance per cell: se(value) / value <= 3 % (or se <= 1 cycle) in both
runs and run agreement <= 5 % -> measured, else experimental.  b300
control: installed within 10 % -> kept; beyond with the cell passing ->
replaced.  Installed value = mean of the two runs.

Usage: model_v2.py --run-a <dir> --run-b <dir> --checks checks.json
         --checks-bar checks_bar.json --device h100|b300 --out model.json
"""
import argparse, json, math, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as v1  # noqa: E402

SE_REL = 0.03
SE_ABS = 1.0
RUN_AGREE = 0.05
CONTROL = 0.10
CTA_WS = (2, 4, 8, 12, 16, 32)
CTA_INSTALL_WS = (4, 8, 12, 16)
CLUSTER_SS = (2, 4, 8)
INSTALLED = {
    "b300": {"synchronization.mbarrier.release": 87.78346192794399,
             "synchronization.cp_async.arrival": 14.0,
             "synchronization.cta.release": {4: 16.0, 8: 17.0, 12: 18.333333333333332, 16: 19.875},
             "synchronization.cluster.release": {2: 87.0, 4: 87.0, 8: 87.0}},
    "h100": {"synchronization.mbarrier.release": 88.3,
             "synchronization.cp_async.arrival": 18.0,
             "synchronization.cta.release": {4: 16.0, 8: 16.0, 12: 16.0, 16: 16.0},
             "synchronization.cluster.release": {2: 166.378515625, 4: 170.370703125, 8: 182.069921875}},
}


def load_rows(run):
    rows = []
    for jl in Path(run).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            d["probe"] = jl.stem
            rows.append(d)
    return rows


def stat(samples):
    xs = [float(x) for x in samples]
    return {"mean": statistics.fmean(xs), "se": statistics.pstdev(xs) / math.sqrt(len(xs)), "n": len(xs),
            "median": statistics.median(xs)}


def bar_cells(rows, checks_bar):
    kern = checks_bar.get("sync_bar", {}).get("kernels", {})
    out = {"cta": {}, "cluster": {}, "cluster_release_form": {}}
    for r in rows:
        if r["probe"] != "sync_bar" or "samples" not in r:
            continue
        v = r["variant"]
        if v.startswith("bar.cta.w") and v.endswith(".warplast") and ".stg." not in v and ".all." not in v:
            w = int(v.split(".")[2][1:])
            c = kern.get(f"sync_bar/cta/w{w}")
            if c and c.get("ok"):
                s = stat(r["samples"])
                out["cta"][w] = {**s, "o_s": c["o_s"], "o_w": c["o_w"], "value": s["mean"] - c["o_s"] - c["o_w"],
                                 "neg_samples": sum(1 for x in r["samples"] if x < 0)}
        if v.startswith("bar.cluster.c") and v.endswith(".arriver"):
            parts = v.split(".")  # bar cluster cS form arriver
            S, form = int(parts[2][1:]), parts[3]
            c = kern.get(f"sync_bar/cluster/c{S}/{form}")
            if c and c.get("ok"):
                s = stat(r["samples"])
                key = "cluster" if form == "relaxed" else "cluster_release_form"
                out[key][S] = {**s, "o_s": c["o_s"], "o_w": c["o_w"], "value": s["mean"] - c["o_s"] - c["o_w"]}
    return out


def verdict(a, b):
    """a / b: {'value', 'se'} of the two runs."""
    if a is None or b is None or a.get("value") is None or b.get("value") is None:
        return None
    m = (a["value"] + b["value"]) / 2
    agree = abs(a["value"] - b["value"]) / abs(m) if m else float("inf")
    se_ok = all((x.get("se") is not None) and (x["se"] <= SE_ABS or (m and x["se"] / abs(m) <= SE_REL)) for x in (a, b))
    return {"value": m, "run_a": a["value"], "run_b": b["value"], "se_a": a.get("se"), "se_b": b.get("se"),
            "run_agree": agree, "se_ok": se_ok,
            "verdict": "measured" if se_ok and agree <= RUN_AGREE else "experimental"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True); ap.add_argument("--run-b", required=True)
    ap.add_argument("--checks", required=True); ap.add_argument("--checks-bar", required=True)
    ap.add_argument("--device", required=True, choices=["h100", "b300"]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    checks = json.loads(Path(a.checks).read_text())
    checks_bar = json.loads(Path(a.checks_bar).read_text())
    inst = INSTALLED[a.device]
    runs = {}
    for tag, d in (("a", a.run_a), ("b", a.run_b)):
        rows = load_rows(d)
        der = v1.derive(rows, checks, {})
        # relabel the version-1 c1 / c2 names (tcgen05.commit, not cluster barrier)
        for old, new in (("synchronization.cluster.release", "tensor.umma.commit.cta_group_1"),
                         ("synchronization.cluster.release.cta_group_2", "tensor.umma.commit.cta_group_2")):
            if old in der:
                der[new] = der.pop(old)
        der["sync_bar"] = bar_cells(rows, checks_bar)
        runs[tag] = der
    out = {"protocol": "tools/microbench/barrier_release/protocol-v2.json", "runs": {"a": a.run_a, "b": a.run_b},
           "derived": {"a": runs["a"], "b": runs["b"]}, "cells": {}}
    # scalar leaves
    for name in ("synchronization.mbarrier.release", "synchronization.cp_async.arrival",
                 "tensor.umma.commit.cta_group_1", "tensor.umma.commit.cta_group_2"):
        ea, eb = runs["a"].get(name), runs["b"].get(name)
        if name == "synchronization.cp_async.arrival":
            # version-1 form is a median difference without an se: use the
            # se of the noinc and arrive sample means (pooled) as the cell se
            for tag, e in (("a", ea), ("b", eb)):
                if e and e.get("value") is not None:
                    rows = load_rows(getattr(a, f"run_{tag}"))
                    s = {r["variant"]: stat(r["samples"]) for r in rows if r["probe"] == "async" and "samples" in r}
                    if "arr.arrive" in s and "arr.cpasync_noinc" in s:
                        e["se"] = math.sqrt(s["arr.arrive"]["se"] ** 2 + s["arr.cpasync_noinc"]["se"] ** 2)
                        e["mean_form"] = (s["arr.cpasync_noinc"]["mean"] - s["arr.cpasync_noinc.issue"]["mean"]) - \
                                         (s["arr.arrive"]["mean"] - s["arr.arrive.issue"]["mean"])
        v = verdict(ea, eb)
        if v and name in inst:
            v["installed"] = inst[name]
            v["control_within_10pct"] = abs(v["value"] - inst[name]) / inst[name] <= CONTROL
        out["cells"][name] = v or {"value": None, "verdict": "unrealised",
                                   "why": {"a": ea and ea.get("why"), "b": eb and eb.get("why")}}
    # cta.release(W) and cluster.release(S)
    for name, key, ws, install_ws in (("synchronization.cta.release", "cta", CTA_WS, CTA_INSTALL_WS),
                                      ("synchronization.cluster.release", "cluster", CLUSTER_SS, CLUSTER_SS),
                                      ("synchronization.cluster.release.release_form", "cluster_release_form", CLUSTER_SS, ())):
        per = {}
        for w in ws:
            v = verdict(runs["a"]["sync_bar"][key].get(w), runs["b"]["sync_bar"][key].get(w))
            if v:
                ref = inst.get(name, {}).get(w) if isinstance(inst.get(name), dict) else None
                if ref:
                    v["installed"] = ref
                    v["control_within_10pct"] = abs(v["value"] - ref) / ref <= CONTROL
                v["install"] = w in install_ws
            per[str(w)] = v
        pts = [[w, per[str(w)]["value"]] for w in install_ws if per.get(str(w))]
        out["cells"][name] = {"per_w": per, "points": pts,
                              "verdict": ("measured" if pts and all(per[str(w)]["verdict"] == "measured" for w in install_ws)
                                          else "experimental" if pts else "unrealised")}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, c in out["cells"].items():
        if "per_w" in c:
            print(f"  {k}: {c['verdict']} points {[(w, round(v, 2)) for w, v in c['points']]}")
            for w, v in c["per_w"].items():
                print(f"      W={w}: {v['value']:.2f} (a {v['run_a']:.2f} b {v['run_b']:.2f}, se {v['se_a']:.2f}/{v['se_b']:.2f}) {v['verdict']}"
                      + (f" installed {v['installed']} ctrl={v['control_within_10pct']}" if "installed" in v else ""))
        elif c.get("value") is not None:
            print(f"  {k}: {c['value']:.2f} (a {c['run_a']:.2f} b {c['run_b']:.2f}, se {c['se_a']:.2f}/{c['se_b']:.2f}, agree {100*c['run_agree']:.1f}%) {c['verdict']}"
                  + (f" installed {c['installed']} ctrl={c['control_within_10pct']}" if "installed" in c else ""))
        else:
            print(f"  {k}: unrealised {c.get('why')}")


if __name__ == "__main__":
    main()
