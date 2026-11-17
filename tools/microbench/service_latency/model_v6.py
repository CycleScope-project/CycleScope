#!/usr/bin/env python3
"""service_latency protocol v6 model (probe alu7, cells-v6.json).

    model_v5.py --device h100|b300 --root <run dir> --out model.json \
        [--tree configs/trees/<dev>.json] [--checks <gate json>]

Rules (protocol-v5):
  single: result = completion = lat slope; issue_interval = thr slope
          / N_unit (v3).  lat and thr are credited independently.
  pair (DSETP): subtract measured partner result_latency (v3); interval
          via same-resource rule.
  issue cells (CS2R_*): interval = OLS slope of the window over K.
  HFMA2 thr: OLS over all seven K; residual > 5 % -> unrealised:
          nonlinear with per-K per-instruction rates + HFMA2 /
          HFMA2.MMA split from the gate sass_counts.
  NOP filler (fp64 cells, b300): lat credited iff slope >= S + 2 both
          runs; thr interval = per-instruction slope - n_NOP x 1, iff
          per-instruction slope >= S + 2; status cap experimental.
  MATCH: per d, result(d) = lat slope, interval(d) = thr slope / N_unit;
          MATCH_model: affine a + b x d over the six d with max rel
          residual <= 5 % on both runs -> affine, else knots.
Acceptance (v3): resid <= 5 % and run agreement <= 5 % -> measured;
<= 15 % -> experimental.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mma_sync"))
import model as v1  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_CELLS = Path(__file__).with_name("cells-v6.json")

PROBES = ("alu7",)
PROBE_NAME = {"DSETP": "DSETP_P"}
MODE_OF = {"lat": "lat", "thr": "thr", "dlat": "dlat", "dthr": "dthr",
           "mlat": "mlat", "mthr": "mthr"}


def probe(cell):
    return PROBE_NAME.get(cell, cell)


def cell_series(rows, pcell, mode):
    out = {}
    for r in rows:
        p = r.get("params", {})
        v = p.get("variant") or r.get("variant") or ""
        vmode = p.get("mode") or v.split(".")[0]
        vcell = ".".join(str(v).split(".")[1:])
        if (r.get("probe") or "") not in PROBES:
            continue
        if vmode != mode or vcell != pcell:
            continue
        out.setdefault(int(r["u"]), []).extend(
            float(x) for x in r["samples"])
    return out


def scope_divisor(repo, cls):
    try:
        cr = json.loads(
            (repo / "configs/mechanism_tree/common/class_resource.json")
            .read_text())
        res = json.loads(
            (repo / "configs/mechanism_tree/common/resources.json")
            .read_text())
    except OSError:
        return None, 1
    rname = (cr.get("classes") or cr).get(cls)
    if rname is None:
        return None, 1
    entries = res if isinstance(res, list) else res.get("resources", [])
    for e in entries:
        if e.get("name") == rname:
            return rname, (4 if e.get("scope") == "sm"
                           and e.get("copy_count", 1) == 1 else 1)
    return rname, 1


def _is_sm_scope(repo, rname):
    try:
        res = json.loads(
            (repo / "configs/mechanism_tree/common/resources.json")
            .read_text())
    except OSError:
        return False
    entries = res if isinstance(res, list) else res.get("resources", [])
    return any(e.get("name") == rname and e.get("scope") == "sm"
               and e.get("copy_count", 1) == 1 for e in entries)


def n_unit(repo, cls, n_instr):
    rname, _ = scope_divisor(repo, cls)
    return rname, (128 if rname is not None and _is_sm_scope(repo, rname)
                   else 32) * n_instr


def service_leaf(tree, cls):
    pref = ("instruction.service.root/predicated_on=1/operation_class="
            + cls)
    for n in tree["nodes"]:
        if n["id"] == pref:
            return n
    return None


def leaf_field(leaf, field):
    f = leaf.get("parameter", {}).get("fields", {}).get(field)
    return (f or {}).get("model", {}).get("value")


def leaf_measured(tree, leaf, field):
    f = leaf.get("parameter", {}).get("fields", {}).get(field)
    if not f:
        return False
    srcs = tree.get("sources", {})
    return any((srcs.get(k) or {}).get("evidence") == "measured"
               for k in f.get("sources", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks")
    ap.add_argument("--tree", default=str(
        _REPO / "configs" / "trees" / "b300.json"))
    a = ap.parse_args()
    if a.tree.endswith("b300.json") and a.device == "h100":
        a.tree = str(_REPO / "configs" / "trees" / "h100.json")

    root = Path(a.root)
    rows = {run: v1.load(root / run) for run in ("run-a", "run-b")}
    checks_path = (Path(a.checks) if a.checks else
                   root / "build" / "sass" / "checks_alu7.json")
    checks = (json.loads(checks_path.read_text())
              if checks_path.is_file() else {"kernels": {}})
    cells = json.loads(_CELLS.read_text())["cells"]
    tree = json.loads(Path(a.tree).read_text())

    gated = {}
    for v in checks.get("kernels", {}).values():
        gated.setdefault(v["cell"], {})[f"{v['mode']}/{v['u']}"] = v

    classes = {}
    for cell, spec in cells.items():
        if a.device not in spec["devices"]:
            classes[cell] = {"cell": cell, "status": "unrealised:device"}
            continue
        gk = gated.get(cell, {})
        all_ok = bool(gk) and all(v["ok"] for v in gk.values())
        e = {"cell": cell, "kind": spec["kind"],
             "composition": spec["composition"],
             "gate": all_ok,
             "gate_failures": {k: v["why"] for k, v in gk.items()
                               if not v["ok"]}}
        if not all_ok:
            e["status"] = "unrealised"
            classes[cell] = e
            continue
        fp64 = spec.get("fp64")
        # NOP filler stats (per run kernel, all kernels of this cell)
        nopS, nops = {}, {}
        for v in gk.values():
            if v.get("stall_S") is not None:
                nopS[v["mode"]] = v["stall_S"]
                nops[v["mode"]] = v["nop_per_target"][0]
        e["nop_filler"] = {"S": nopS, "n_per_target": nops}

        if spec["kind"] == "issue":
            sa = v1.slope_fit(
                cell_series(rows["run-a"], cell, "issue"))
            sb = v1.slope_fit(
                cell_series(rows["run-b"], cell, "issue"))
            if sa and sb:
                m = (sa["slope"] + sb["slope"]) / 2
                e["issue_interval"] = m
                e["resid"] = max(sa["max_rel_resid"],
                                 sb["max_rel_resid"])
                e["agree"] = abs(sa["slope"] - sb["slope"]) / m
                e["runs"] = [sa["slope"], sb["slope"]]
                e["status"] = ("measured" if e["resid"] <= 0.05
                               and e["agree"] <= 0.05 else
                               "experimental" if e["resid"] <= 0.15
                               and e["agree"] <= 0.15 else "unrealised")
            else:
                e["status"] = "unrealised"
            classes[cell] = e
            continue

        ra = v1.slope_fit(cell_series(rows["run-a"], probe(cell),
                                      "dlat" if spec.get("fp64") else "lat"))
        rb = v1.slope_fit(cell_series(rows["run-b"], probe(cell),
                                      "dlat" if spec.get("fp64") else "lat"))
        thr_mode = ("dthr" if spec.get("fp64") else "thr")
        ta = v1.slope_fit(cell_series(rows["run-a"], probe(cell),
                                      thr_mode))
        tb = v1.slope_fit(cell_series(rows["run-b"], probe(cell),
                                      thr_mode))
        n_instr = spec["composition"].get(spec.get("credits") or next(iter(spec["composition"])), 1)
        rname, nu = n_unit(_REPO, spec.get("credits") or cell, n_instr)
        e["resource"], e["n_unit"] = rname, nu

        # lat
        if ra and rb:
            la = ra["slope"]; lb = rb["slope"]
            e["lat_runs"] = [la, lb]
            e["lat_resid"] = max(ra["max_rel_resid"], rb["max_rel_resid"])
            m = (la + lb) / 2
            e["lat_agree"] = abs(la - lb) / m if m else float("inf")
            stall_ok = True
            if fp64 and nopS.get("dlat") is not None:
                S = nopS["dlat"]
                stall_ok = la >= S + 2 and lb >= S + 2
                e["lat_stall_S"] = S
            if not stall_ok:
                e["lat_status"] = "unrealised:stall_bound"
            elif spec["kind"] == "single":
                e["result_latency"] = e["completion_latency"] = m
                e["lat_status"] = (
                    "measured" if e["lat_resid"] <= 0.05
                    and e["lat_agree"] <= 0.05 else
                    "experimental" if e["lat_resid"] <= 0.15
                    and e["lat_agree"] <= 0.15 else "unrealised")
                if fp64 and nopS.get("dlat") is not None \
                        and e["lat_status"] == "measured":
                    e["lat_status"] = "experimental"  # NOP cap
            else:  # pair
                used, missing = {}, []
                for cls in spec.get("critical_path_partners", []):
                    leaf = service_leaf(tree, cls)
                    if leaf and leaf_measured(
                            tree, leaf, "result_latency_cycles"):
                        used[cls] = leaf_field(
                            leaf, "result_latency_cycles")
                    else:
                        missing.append(cls)
                if missing:
                    e["lat_status"] = "unrealised"
                    e["missing_partner_evidence"] = missing
                else:
                    e["partner_result_latency"] = used
                    e["result_latency"] = e["completion_latency"] = (
                        m - sum(used.values()))
                    e["lat_status"] = "experimental"  # pair cap
        # thr
        if ta and tb:
            sa_, sb_ = ta["slope"], tb["slope"]
            per_instr_a = sa_ / (2 * nu)   # slope over 8 chains
            per_instr_b = sb_ / (2 * nu)
            e["thr_resid"] = max(ta["max_rel_resid"],
                                 tb["max_rel_resid"])
            m = (per_instr_a + per_instr_b) / 2
            e["thr_agree"] = (abs(per_instr_a - per_instr_b) / m
                              if m else float("inf"))
            e["thr_runs"] = [per_instr_a, per_instr_b]
            if cell == "HFMA2":
                per_k = {}
                for v in gk.values():
                    K = int(v["key"].rsplit("/", 1)[1])
                    sc_ = v.get("sass_counts", {})
                    per_k[K] = {"hfma2": sc_.get("HFMA2", 0),
                                "hfma2_mma": sc_.get("HFMA2.MMA", 0)}
                e["hfma2_split"] = per_k
                if e["thr_resid"] > 0.05:
                    e["thr_status"] = "unrealised:nonlinear"
                    e["per_k_rate"] = {
                        str(K): [ta["medians"][i] / (2 * nu)
                                 for i, u_ in enumerate(ta["us"])
                                 if u_ == K] +
                                [tb["medians"][j] / (2 * nu)
                                 for j, u_ in enumerate(tb["us"])
                                 if u_ == K]
                        for K in sorted(per_k)}
                else:
                    e["issue_interval"] = m
                    e["thr_status"] = (
                        "measured" if e["thr_agree"] <= 0.05
                        else "experimental"
                        if e["thr_agree"] <= 0.15 else "unrealised")
            elif fp64 and nopS.get("dthr") is not None:
                S = nopS["dthr"]; n = nops["dthr"]
                # v6 thr rule: c=8 chains per warp; stall-bound iff the
                # per-instruction slope <= 1.05 x c x S, else interval =
                # slope / N_unit - 0 (NOPs issue on the SMSP front end)
                c = 8
                bound = 1.05 * c * S
                e["thr_stall_S"] = S
                e["thr_stall_bound"] = bound
                if per_instr_a <= bound and per_instr_b <= bound:
                    e["thr_status"] = "unrealised:stall_bound"
                else:
                    e["issue_interval"] = (sa_ + sb_) / 2 / nu
                    e["thr_status"] = "experimental"  # NOP cap
            else:
                e["issue_interval"] = m
                e["thr_status"] = (
                    "measured" if e["thr_resid"] <= 0.05
                    and e["thr_agree"] <= 0.05 else
                    "experimental" if e["thr_resid"] <= 0.15
                    and e["thr_agree"] <= 0.15 else "unrealised")
        # v6 records: ops per U/K step and the divisor used
        e["ops_per_lat_step"] = 1
        e["chains_thr"] = 8
        e["thr_divisor"] = "slope / (2 * n_unit)" if spec["kind"] == "single" else None
        # repro_rule (b300): within 10 % of installed -> reproduced
        credit = spec.get("credits") or next(iter(spec["composition"]))
        leaf = service_leaf(tree, credit)
        if leaf is not None:
            inst_r = leaf_field(leaf, "result_latency_cycles")
            inst_i = leaf_field(leaf, "issue_interval_cycles")
            def _val(f):
                return (f.get("model", {}).get("value")
                        if isinstance(f, dict) else f)
            e["installed_result"] = _val(inst_r)
            e["installed_result_sources"] = (
                inst_r.get("sources") if isinstance(inst_r, dict) else None)
            e["installed_interval"] = _val(inst_i)
            if (e.get("result_latency") is not None
                    and isinstance(e["installed_result"], (int, float))
                    and e["installed_result"]):
                rel = abs(e["result_latency"] - e["installed_result"]) / e["installed_result"]
                e["repro_result_rel"] = rel
                e["repro_result"] = "reproduced" if rel <= 0.10 else "replaces"
            if (e.get("issue_interval") is not None
                    and isinstance(e["installed_interval"], (int, float))
                    and e["installed_interval"]):
                rel = abs(e["issue_interval"] - e["installed_interval"]) / e["installed_interval"]
                e["repro_interval_rel"] = rel
                e["repro_interval"] = "reproduced" if rel <= 0.10 else "replaces"
        sts = [x for x in (e.get("lat_status"), e.get("thr_status"))
               if x]
        e["status"] = ("unrealised" if not sts else
                       "measured" if all(s == "measured" for s in sts)
                       else "experimental" if any(
                           s == "experimental" for s in sts)
                       else min(sts, key=lambda s: (
                           "measured" == s, "experimental" == s)) if any(
                           s in ("measured", "experimental") for s in sts)
                       else "unrealised")
        if any(s and s.startswith("unrealised") for s in sts) and not any(
                s in ("measured", "experimental") for s in sts):
            e["status"] = "unrealised"
        classes[cell] = e

    # MATCH_model: affine a + b x d over the six d
    match = {}
    for d_ in (1, 2, 4, 8, 16, 32):
        e = classes.get(f"MATCH_d{d_}")
        if e and e.get("result_latency") is not None:
            match[d_] = {"result": e["result_latency"],
                         "interval": e.get("issue_interval")}
    if match:
        def affine(xs, ys):
            n = len(xs)
            mx = sum(xs) / n; my = sum(ys) / n
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) \
                / sum((x - mx) ** 2 for x in xs)
            aa = my - b * mx
            resid = max(abs(aa + b * x - y) / abs(y)
                        for x, y in zip(xs, ys) if y) if all(ys) else 1
            return aa, b, resid
        mm = {"d": match}
        if a.device == "h100" and 32 in match:
            v = match[32].get("interval")
            mm["d32_repro_8.09"] = (None if v is None else
                                    {"value": v, "rel": abs(v - 8.09) / 8.09,
                                     "within_10pct": abs(v - 8.09) / 8.09 <= 0.10})
        if a.device == "b300" and 32 in match:
            mm["d32_replaces_clamp"] = match[32].get("interval")
        for fld in ("result", "interval"):
            vals = [match[d_].get(fld) for d_ in sorted(match)]
            if any(v is None for v in vals):
                mm[fld + "_model"] = "partial"
                continue
            aa, b, resid = affine(sorted(match), vals)
            mm[fld + "_affine"] = {"a": aa, "b": b,
                                   "max_rel_resid": resid,
                                   "model": "affine"
                                   if resid <= 0.05 else "knots"}
        classes["MATCH_model"] = mm
    out = {"protocol":
           "tools/microbench/service_latency/protocol-v5.json",
           "device": a.device,
           "runs": {"a": str(root / "run-a"), "b": str(root / "run-b")},
           "classes": classes}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for cls, e in classes.items():
        print(f"  {cls:14s} {e.get('result_latency')} / "
              f"{e.get('issue_interval')} -> {e.get('status')}")


if __name__ == "__main__":
    main()
