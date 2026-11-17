#!/usr/bin/env python3
"""service_latency protocol v2 derivation (protocol-v2.json + cells-v2.json).

    model_v2.py --device h100|b300 --run-a <dir> --run-b <dir>
        --checks <dir with alu.json alu2.json misc.json svc.json alu3.json>
        --tree configs/trees/<dev>.json --cells cells-v2.json
        --barrier-model <barrier_release-model/<dev>-r*/model.json>
        --out model.json

Per class of cells-v2 (this device's architecture):
  bucket            -> structural 1 / 1 / 0 (ownership rule), except the
                       classes listed in KEEP (owned by another protocol's
                       installed values / routing not verified)
  probe alu / alu2  -> version-1 derivation (model.py tables): result =
                       OLS slope over U of lat medians (pair chains minus
                       partner), issue_interval = thr slope / 8 at wps 1
  probe alu3        -> lat slope over U; thr slope / 32 (16 warps = 4 per
                       SMSP x 8 chains); mma thr slope / 16 (4 accumulators)
  probe misc        -> shot cells: mean(t1-t0) - window stall sum - t0 stall
                       + target stall (blackwell-prototype single-shot rule);
                       lat cells: slope over U
  probe svc         -> k_fence: result = slope over K of the w1 medians
                       (issue -> next dependent issue); issue_interval =
                       slope(w4s0) / 4 when four warps on one SMSP take
                       longer than one (> 5 %), else 0 with the /4 value
                       reported as the unresolved upper bound
  ERRBAR, CGAERRBAR -> barrier_release v2 cluster arrive.release minus
                       arrive.relaxed (per S, mean) minus result(MEMBAR.ALL.GPU)
Every numeric cell needs its SASS gate record ok for all its points; cell
verdict: fit residual <= 5 % and runs within 5 % -> measured, else
experimental.  b300 control: installed measured leaf within 10 % -> kept
(cited), else replaced.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as v1  # noqa: E402  (version-1 tables and fits)

RESID, AGREE, CONTROL, SERIAL = 0.05, 0.05, 0.10, 0.05
# classes whose service leaf is owned by another installed protocol or
# whose owning engine path is not verified: the bucket rule is not applied
KEEP = {
    "LDS": "shared protocol v2 installs result = chain slope - acceptance here",
    "LDSM": "shared protocol v2 installs result = chain slope - acceptance here",
    "STSM": "shared protocol v2 installs result = chain slope - acceptance here",
    "LDL": "local-memory latency path through memory.* not verified in the engine",
    "STL": "local-memory latency path through memory.* not verified in the engine",
}
ALU3 = {  # class -> (lat variant, thr variant, chains per SMSP step, ops per step)
    "IMAD.HI": ("lat.IMADHI", "thr.IMADHI", 32, 1),
    "F2F": ("lat.F2F_", "thr.F2F_", 32, 2),
    "HMMA": ("lat.mma.HMMA", "thr.mma.HMMA", 16, 1), "IMMA": ("lat.mma.IMMA", None, 16, 1),
}
MISC = {  # class -> (kind, variant)
    "S2R": ("shot", "shot.s2r", 3), "S2UR": ("shot", "shot.s2ur", 4),
    "LEA": ("lat", "lat.LEA64", None), "ULEA": ("lat", "ulat.ULEA", None),
}
FENCE = {
    "MEMBAR.ALL.CTA": "cta.acq_rel", "MEMBAR.SC.CTA": "cta.sc",
    "MEMBAR.ALL.GPU": "gpu.acq_rel", "MEMBAR.SC.GPU": "gpu.sc",
    "MEMBAR.ALL.SYS": "sys.acq_rel", "FENCE.VIEW.ASYNC.S": "async.shared",
    "FENCE.VIEW.ASYNC.G": "async.global",
}


def load_checks(d):
    out = {}
    for p in ("alu", "alu2", "misc", "svc", "alu3"):
        f = Path(d) / f"{p}.json"
        if f.exists():
            out[p] = json.loads(f.read_text())
    return out


def check(checks, probe, key):
    return checks.get(probe, {}).get("kernels", {}).get(key)


def gate_ok(checks, probe, key):
    k = check(checks, probe, key)
    return k is not None and k.get("ok")


def fit_series(rows, probe, variant, checks, key_of):
    s = {}
    gated = []
    for r in rows:
        if r["probe"] != probe or r["variant"] != variant:
            continue
        key = key_of(r)
        if not gate_ok(checks, probe, key):
            gated.append(key)
            continue
        s.setdefault(int(r["u"]), []).extend(float(x) for x in r["samples"])
    return v1.slope_fit(s), gated


def verdict(va, vb, resid):
    m = (va + vb) / 2
    agree = abs(va - vb) / abs(m) if m else float("inf")
    return m, agree, ("measured" if resid <= RESID and agree <= AGREE else "experimental")


def alu3_cell(rows, checks, cls):
    lat_v, thr_v, chains, per_step = ALU3[cls]
    key = lambda r: f"alu3/{r['variant']}/{r['u']}"  # noqa: E731
    e = {"probe": "alu3", "cells": {}}
    f, gated = fit_series(rows, "alu3", lat_v, checks, key)
    e["cells"]["lat"] = {"variant": lat_v, "gated_out": gated}
    if f:
        e["result_latency"] = f["slope"] / per_step; e["lat_resid"] = f["max_rel_resid"]
        e["lat_medians"] = dict(zip(f["us"], f["medians"])); e["ops_per_step"] = per_step
    if thr_v:
        g, gated2 = fit_series(rows, "alu3", thr_v, checks, key)
        e["cells"]["thr"] = {"variant": thr_v, "gated_out": gated2, "chains_per_smsp_step": chains}
        if g:
            e["issue_interval"] = g["slope"] / (chains * per_step); e["thr_resid"] = g["max_rel_resid"]
    return e


def misc_cell(rows, checks, cls):
    kind, variant, u = MISC[cls]
    e = {"probe": "misc", "cells": {}}
    if kind == "lat":
        f, gated = fit_series(rows, "misc", variant, checks,
                              lambda r: f"misc/{r['variant']}/{r['u']}")
        e["cells"]["lat"] = {"variant": variant, "gated_out": gated}
        if f:
            e["result_latency"] = f["slope"]; e["lat_resid"] = f["max_rel_resid"]
            e["lat_medians"] = dict(zip(f["us"], f["medians"]))
        return e
    c = check(checks, "misc", f"misc/{variant}/{u}")
    rr = [r for r in rows if r["probe"] == "misc" and r["variant"] == variant]
    e["cells"]["shot"] = {"variant": variant, "gate": bool(c and c.get("ok"))}
    if rr and c and c.get("ok"):
        samples = [float(x) for r in rr for x in r["samples"]]
        mean = statistics.fmean(samples)
        extra = mean - c["stall_sum"] - (c.get("t0_stall") or 0)
        e["result_latency"] = extra + c["target_stall"]
        e["lat_resid"] = 0.0 if len(set(samples)) == 1 else statistics.pstdev(samples) / mean
        e["shot"] = {"mean": mean, "stall_sum": c["stall_sum"], "t0_stall": c.get("t0_stall"),
                     "target_stall": c["target_stall"], "n": len(samples)}
    return e


def fence_cell(rows, checks, cls):
    fl = FENCE[cls]
    key = lambda r: f"svc/{r['variant']}/{r['u']}"  # noqa: E731
    e = {"probe": "svc", "cells": {}}
    out = {}
    for w in ("w1", "w4x1", "w4s0", "w16"):
        f, gated = fit_series(rows, "svc", f"fence.{fl}.{w}", checks, key)
        e["cells"][w] = {"variant": f"fence.{fl}.{w}", "gated_out": gated}
        if f:
            out[w] = f
            e["cells"][w].update(slope=f["slope"], resid=f["max_rel_resid"],
                                 medians=dict(zip(f["us"], f["medians"])))
    if "w1" in out:
        e["result_latency"] = out["w1"]["slope"]; e["lat_resid"] = out["w1"]["max_rel_resid"]
    if "w1" in out and "w4s0" in out:
        s1, s4 = out["w1"]["slope"], out["w4s0"]["slope"]
        e["serialisation_ratio"] = s4 / s1 if s1 else None
        e["issue_interval_upper_bound"] = s4 / 4.0
        e["issue_interval"] = s4 / 4.0 if s4 > (1 + SERIAL) * s1 else 0.0
        e["thr_resid"] = out["w4s0"]["max_rel_resid"]
        e["issue_interval_rule"] = ("slope(w4s0)/4: four warps on one SMSP take longer than one"
                                    if s4 > (1 + SERIAL) * s1 else
                                    "0: four warps on one SMSP finish in one warp's time (no "
                                    "serialisation resolved; /4 value is only an upper bound)")
    return e


def errbar(barrier_model, membar_gpu):
    b = json.loads(Path(barrier_model).read_text())["cells"]
    rel = b["synchronization.cluster.release.release_form"]["per_w"]
    rlx = b["synchronization.cluster.release"]["per_w"]
    diffs = {s: rel[s]["value"] - rlx[s]["value"] for s in rel if s in rlx}
    d = statistics.fmean(diffs.values())
    e = {"release_minus_relaxed_per_s": diffs, "release_minus_relaxed": d,
         "membar_all_gpu_result": membar_gpu, "barrier_model": barrier_model}
    if membar_gpu is not None:
        e["result_latency"] = d - membar_gpu
        e["consistent"] = e["result_latency"] > 0
        if not e["consistent"]:
            e["open_question"] = ("MEMBAR.ALL.GPU k_fence result (the fence.acq_rel.gpu expansion:"
                                  " MEMBAR + ERRBAR + CGAERRBAR + CCTL.IVALL) already exceeds the "
                                  "release-minus-relaxed difference; ERRBAR cannot be separated")
    return e


def installed(tree_path):
    t = json.loads(Path(tree_path).read_text())
    out = {}
    for n in t["nodes"]:
        if n["id"].startswith("instruction.service") and n.get("parameter") and \
                "predicated_on=1" in n["id"] and "operation_class=" in n["id"]:
            cls = n["id"].split("operation_class=")[1].split("/")[0]
            fl = n["parameter"]["fields"]

            def c(f):
                m = fl.get(f, {}).get("model", {})
                return m.get("value") if m.get("kind") == "constant" else None
            out[cls] = {"result_latency": c("result_latency_cycles"),
                        "completion_latency": c("completion_latency_cycles"),
                        "issue_interval": c("issue_interval_cycles"),
                        "status": n["parameter"].get("status")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--run-a", required=True); ap.add_argument("--run-b", required=True)
    ap.add_argument("--checks", required=True); ap.add_argument("--tree", required=True)
    ap.add_argument("--cells", required=True); ap.add_argument("--barrier-model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arch = "hopper" if a.device == "h100" else "blackwell"
    cells = json.loads(Path(a.cells).read_text())["cells"][arch]
    checks = load_checks(a.checks)
    ra, rb = v1.load(a.run_a), v1.load(a.run_b)
    inst = installed(a.tree)
    # version-1 derivation for alu / alu2
    v1a, v1b = v1.derive(ra, checks), v1.derive(rb, checks)
    classes, buckets, unrealised = {}, {}, {}
    for cls, cell in cells.items():
        kind = cell["kind"]
        if kind == "bucket":
            if cls in KEEP:
                unrealised[cls] = {"kind": "kept", "reason": KEEP[cls], "cell": cell}
            else:
                buckets[cls] = {"bucket": cell["bucket"], "install": {"result_latency": 1,
                                "completion_latency": 1, "issue_interval": 0},
                                "installed": inst.get(cls)}
            continue
        if kind in ("unrealised", "unsupported_on_device"):
            unrealised[cls] = {"kind": kind, "cell": cell}
            continue
        probe = cell["probe"]
        ea = eb = None
        if probe in ("alu", "alu2"):
            ea, eb = v1a.get(cls), v1b.get(cls)
            if ea is None and eb is None:
                known = cls in v1.LATENCY_CLASSES or cls in v1.ALU2_CLASSES
                unrealised[cls] = {"kind": "gate_or_fit_failed" if known else "no_derivation_table",
                                   "cell": cell,
                                   "note": "every point gated out or no fit" if known else
                                           "class not in the version-1 alu/alu2 class map"}
                continue
        elif probe == "alu3":
            if cls not in ALU3:
                unrealised[cls] = {"kind": "no_cell_in_probe", "cell": cell}
                continue
            ea, eb = alu3_cell(ra, checks, cls), alu3_cell(rb, checks, cls)
        elif probe == "misc":
            if cls not in MISC:
                unrealised[cls] = {"kind": "no_cell_in_probe", "cell": cell,
                                   "note": "misc.cu has no shot for this class (cells-v2 mapped it "
                                           "to another opcode's shot or to a missing cell)"}
                continue
            ea, eb = misc_cell(ra, checks, cls), misc_cell(rb, checks, cls)
        elif probe == "svc":
            if cls not in FENCE:
                unrealised[cls] = {"kind": "no_cell_in_probe", "cell": cell,
                                   "note": "k_fence has no variant for this class"}
                continue
            ea, eb = fence_cell(ra, checks, cls), fence_cell(rb, checks, cls)
        c = {"probe": probe, "cell": cell, "run_a": ea, "run_b": eb, "verdict": {},
             "installed": inst.get(cls)}
        ea, eb = ea or {}, eb or {}
        for field, rk in (("result_latency", "lat_resid"), ("issue_interval", "thr_resid")):
            if field in ea and field in eb:
                m, agree, vd = verdict(ea[field], eb[field], max(ea.get(rk, 0), eb.get(rk, 0)))
                c[field] = m; c[field + "_agree"] = agree
                c[field + "_resid"] = max(ea.get(rk, 0), eb.get(rk, 0)); c["verdict"][field] = vd
                if probe == "svc" and field == "issue_interval":
                    c["verdict"][field] = "experimental"  # rule fixed after the data
                if probe == "alu3" and field == "issue_interval":
                    c["verdict"][field] = "experimental"  # wps 4, not the protocol's wps 1
                ref = (inst.get(cls) or {}).get(field)
                if ref not in (None,):
                    c[field + "_installed"] = ref
                    c[field + "_control_ok"] = (abs(m - ref) / ref <= CONTROL) if ref else (m == 0)
        if "result_latency" not in c and "issue_interval" not in c:
            unrealised[cls] = {"kind": "gate_or_fit_failed", "cell": cell, "run_a": ea, "run_b": eb}
            continue
        classes[cls] = c
    mg = classes.get("MEMBAR.ALL.GPU", {}).get("result_latency")
    eb_ = errbar(a.barrier_model, mg)
    for cls in ("ERRBAR", "CGAERRBAR"):
        if cls in cells:
            e = dict(eb_); e["cell"] = cells[cls]; e["installed"] = inst.get(cls)
            if e.get("consistent"):
                e["verdict"] = {"result_latency": "experimental"}; e["issue_interval"] = 0
                classes[cls] = e
            else:
                unrealised[cls] = {"kind": "derivation_inconsistent", **e}
    out = {"protocol": "tools/microbench/service_latency/protocol-v2.json", "device": a.device,
           "runs": {"a": a.run_a, "b": a.run_b}, "checks": a.checks, "tree": a.tree,
           "classes": classes, "buckets": buckets, "unrealised": unrealised,
           "keep": KEEP}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{a.device}: {len(classes)} measured classes, {len(buckets)} bucket classes, "
          f"{len(unrealised)} unrealised / kept")
    for cls, c in sorted(classes.items()):
        rl = c.get("result_latency"); ii = c.get("issue_interval")
        print(f"  {cls:20s} lat {rl if rl is None else round(rl, 2)!s:>8} "
              f"({c['verdict'].get('result_latency', '-')}, inst {(c.get('installed') or {}).get('result_latency')}) "
              f"ii {ii if ii is None else round(ii, 2)!s:>6} ({c['verdict'].get('issue_interval', '-')}, "
              f"inst {(c.get('installed') or {}).get('issue_interval')})")
    for cls, u in sorted(unrealised.items()):
        print(f"  - {cls:20s} {u['kind']}")


if __name__ == "__main__":
    main()
