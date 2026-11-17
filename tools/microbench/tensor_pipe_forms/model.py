#!/usr/bin/env python3
"""tensor_pipe_forms derivation (tools/microbench/tensor_pipe_forms/protocol.json).

  model.py --root $A/microbench/tensor_pipe_forms/b300-<run> [--checks checks-v2.json]
           --tree configs/trees/b300.json --out <model.json>

Per form (factor_id): E_U = mean of the window samples (xw: t_w - t_s,
cad: t1 - t0); beat = OLS slope of E_U(cad) over U {32, 64, 128};
latency_1 = (E_xw1 - E_xw0) - (o_s1 - o_s0) with the same form's xw U = 0
kernel; completion = latency_1 - beat.  Gate / thresholds copied from the
frozen protocol (tensor_pipe v2 rules): beat residual <= 2 %, run
agreement <= 3 %; completion agreement <= 5 % or <= 2 cycles, negative
within 2 se -> 0, beyond -> not installed; d_ok and d_timed_ok required.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

V2_U, V2_RESID, BEAT_AGREE = (32, 64, 128), 0.02, 0.03
COMP_AGREE, COMP_AGREE_ABS = 0.05, 2.0
CONTROL = 0.10
KIND_CLASS = {"f16": "UTCHMMA", "tf32": "UTCHMMA", "f8f6f4": "UTCQMMA",
              "mxf8f6f4": "UTCQMMA", "mxf4": "UTCOMMA", "mxf4nvf4": "UTCOMMA"}
OP_NAME = {"normal": "normal", "sp": "sp", "ws": "ws", "ws_sp": "ws_sp"}
ACC_RE = re.compile(r"_Z8k_factorILi\d+ELi\d+ELi\d+ELi\d+ELi\d+ELi\d+ELb\dELb\dELb\dELi(\d+)E")


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    resid = max(abs(a + b * x - y) / y for x, y in zip(xs, ys))
    return b, a, resid


def coord(p):
    major = int(p["level"]) if p["factor"] == "umma_operand_major" else 0
    shift = int(p["level"]) if p["factor"] == "max_shift" else 0
    return [int(p["m"]), int(p["n"]), int(p["k"]), OP_NAME[p["operation"]], major, shift]


def load_run(run_dir: Path):
    forms = {}
    for f in sorted(run_dir.glob("*/umma_factors.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            p = r["params"]
            fm = forms.setdefault(p["factor_id"], {"params": p, "cells": {}})
            s = r["samples"]
            fm["cells"][(p["mode"], int(r["u"]))] = {
                "mean": statistics.fmean(s), "sd": statistics.pstdev(s), "n": len(s),
                "d_ok": bool(r.get("d_ok", True)), "d_timed_ok": r.get("d_timed_ok") in (True, None)}
    return forms


def load_checks(path: Path):
    """(kind, cg, asrc, mode, u, m_rows, ws, sparse) -> record (elected = 1, ACC = 1)"""
    c = json.loads(path.read_text())
    out = {}
    for name, k in c["umma_factors"]["kernels"].items():
        if not k.get("elected"):
            continue
        m = ACC_RE.match(name)
        if m and int(m.group(1)) != 1:
            continue
        key = (k["kind"], int(k["cg"]), int(k["asrc"]), k["mode"], int(k["u"]), int(k["m_rows"]),
               bool(k["ws"]), bool(k["sparse"]))
        out[key] = k
    return out


def kernel_key(p, mode, u):
    return (p["kind"], int(p["cta_group"]), 1 if p["a_src"] == "tmem" else 0, mode, u,
            int(p["m"]) // int(p["cta_group"]), p["operation"].startswith("ws"), bool(p["sparse"]))


def derive(forms, checks):
    res = {}
    for fid, fm in forms.items():
        p, cells = fm["params"], fm["cells"]
        if any(("cad", u) not in cells for u in V2_U) or ("xw", 1) not in cells or ("xw", 0) not in cells:
            res[fid] = {"params": p, "coord": coord(p), "unrealised": "missing cells"}
            continue
        us = list(V2_U)
        ys = [cells[("cad", u)]["mean"] for u in us]
        beat, icpt, resid = ols(us, ys)
        mx = statistics.fmean(us)
        den = sum((u - mx) ** 2 for u in us)
        beat_se = math.sqrt(sum(((u - mx) / den) ** 2 * (cells[("cad", u)]["sd"] ** 2 / cells[("cad", u)]["n"]) for u in us))
        gate = []
        o_s = {}
        for mode, u in [("cad", u) for u in us] + [("xw", 1), ("xw", 0)]:
            k = checks.get(kernel_key(p, mode, u))
            if k is None:
                gate.append(f"{mode} U{u}: no gated kernel")
                continue
            if not k.get("ok"):
                gate += [f"{mode} U{u}: {w}" for w in k.get("why", [])]
            o_s[(mode, u)] = k.get("o_s")
        d_ok = all(c["d_ok"] and c["d_timed_ok"] for c in cells.values())
        e1, e0 = cells[("xw", 1)], cells[("xw", 0)]
        lat1 = None
        if o_s.get(("xw", 1)) is not None and o_s.get(("xw", 0)) is not None:
            lat1 = (e1["mean"] - e0["mean"]) - (o_s[("xw", 1)] - o_s[("xw", 0)])
        lat_se = math.sqrt(e1["sd"] ** 2 / e1["n"] + e0["sd"] ** 2 / e0["n"])
        res[fid] = {"params": p, "coord": coord(p), "beat": beat, "beat_resid": resid, "beat_se": beat_se,
                    "latency_1": lat1, "latency_se": lat_se,
                    "completion": None if lat1 is None else lat1 - beat,
                    "completion_se": math.sqrt(lat_se ** 2 + beat_se ** 2),
                    "cad_means": {str(u): cells[("cad", u)]["mean"] for u in us},
                    "xw_means": {"1": e1["mean"], "0": e0["mean"]}, "o_s": {f"{m}{u}": v for (m, u), v in o_s.items()},
                    "d_ok": d_ok, "sass_gate": gate}
    return res


def installed_points(tree: dict):
    """(class, kind, cg, a_src, m, n, k, op, major, shift) -> installed pipe value"""
    out = {}
    for node in tree["nodes"]:
        nid = node["id"]
        m = re.match(r"tensor\.pipe\.root/operation_class=(\w+)/umma_kind=(\w+)/cta_group=(\d)/a_src=(\w+)$", nid)
        if not m or "parameter" not in node:
            continue
        for p in node["parameter"]["fields"]["cycles"]["model"]["points"]:
            out[(m.group(1), m.group(2), int(m.group(3)), m.group(4)) + tuple(p[:6])] = p[6]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--checks", default="checks-v2.json")
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rerun-of", type=Path, default=None,
                    help="model.json of the run this run re-runs (r3 vs r2)")
    a = ap.parse_args()
    checks = load_checks(a.root / a.checks)
    runs = {r: derive(load_run(a.root / r), checks) for r in ("run-a", "run-b")}
    tree = json.loads(a.tree.read_text())
    inst = installed_points(tree)
    verdict = {}
    counts = defaultdict(int)
    for fid in sorted(runs["run-a"]):
        x, y = runs["run-a"][fid], runs["run-b"].get(fid)
        p = x["params"]
        key = (KIND_CLASS[p["kind"]], p["kind"], int(p["cta_group"]), p["a_src"]) + tuple(x["coord"])
        installed = inst.get(key)
        if y is None or "unrealised" in x or "unrealised" in y:
            verdict[fid] = {"params": p, "coord": x["coord"], "installed": installed, "status": "unrealised",
                            "why": x.get("unrealised") or (y or {}).get("unrealised") or "missing in run-b"}
            counts["unrealised"] += 1
            continue
        beat = (x["beat"] + y["beat"]) / 2
        agree = abs(x["beat"] - y["beat"]) / beat
        gate_ok = not x["sass_gate"] and x["d_ok"] and y["d_ok"]
        beat_ok = max(x["beat_resid"], y["beat_resid"]) <= V2_RESID and agree <= BEAT_AGREE and gate_ok
        comp = comp_install = None
        comp_ok = False
        if x["completion"] is not None and y["completion"] is not None:
            comp = (x["completion"] + y["completion"]) / 2
            diff = abs(x["completion"] - y["completion"])
            comp_ok = diff <= max(COMP_AGREE * abs(comp), COMP_AGREE_ABS) and gate_ok
            se = max(x["completion_se"], y["completion_se"])
            if comp_ok:
                comp_install = comp if comp >= 0 else (0.0 if -comp <= 2 * se else None)
        lat = None if x["latency_1"] is None or y["latency_1"] is None else (x["latency_1"] + y["latency_1"]) / 2
        ctrl = None if installed is None or lat is None or not installed else (lat - installed) / installed
        status = "unrealised" if not gate_ok else ("measured" if beat_ok else "experimental")
        counts[status] += 1
        if status != "unrealised" and comp_install is None:
            counts["completion_not_installed"] += 1
        if installed is None:
            counts["coordinate_not_in_tree"] += 1
        verdict[fid] = {
            "params": {k: p[k] for k in ("kind", "cta_group", "a_src", "m", "n", "k", "operation", "factor", "level", "sparse")},
            "coord": x["coord"], "beat": beat, "beat_agree": agree, "beat_resid": max(x["beat_resid"], y["beat_resid"]),
            "beat_status": status, "completion": comp, "completion_ok": comp_ok, "completion_install": comp_install,
            "latency_1": lat, "installed_pipe": installed, "control_rel": ctrl,
            "control": None if ctrl is None else ("reproduced" if abs(ctrl) <= CONTROL else "differs"),
            "sass_gate": x["sass_gate"], "d_ok": x["d_ok"] and y["d_ok"],
            **({"sparse_gate": "probe D" if (x["d_ok"] and y["d_ok"]) else "FAIL probe D"}
               if p["sparse"] else {}),
            "cad_means": {"a": x["cad_means"], "b": y["cad_means"]}, "o_s": x["o_s"],
            "v2_coordinate": p["operation"] == "normal" and int(p["m"]) // int(p["cta_group"]) == 128
            and x["coord"][4] == 0 and x["coord"][5] == 0,
        }
    out = {"protocol": "tools/microbench/tensor_pipe_forms/protocol.json", "root": str(a.root), "checks": a.checks,
           "counts": dict(counts), "forms": verdict}
    if a.rerun_of is not None:
        r2 = json.loads(a.rerun_of.read_text())["forms"]
        eligible = [fid for fid, v in r2.items() if v.get("completion_install") is None]
        elig = set(eligible)
        control = {}
        for fid, v in r2.items():
            if fid in elig:
                continue
            b3 = (verdict.get(fid) or {}).get("beat")
            b2 = v.get("beat")
            control[fid] = {"beat_r2": b2, "beat_r3": b3,
                            "rel": None if not b2 or b3 is None else abs(b3 - b2) / abs(b2)}
        counts["eligible_pass"] = sum(
            1 for fid in eligible
            if verdict.get(fid, {}).get("beat_status") in ("measured", "experimental")
            and verdict.get(fid, {}).get("completion_install") is not None)
        out["rerun"] = {"of": str(a.rerun_of), "eligible": eligible, "control": control}
        out["counts"] = dict(counts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(dict(counts))


if __name__ == "__main__":
    main()
