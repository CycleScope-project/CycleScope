#!/usr/bin/env python3
"""service_latency protocol v3 derivation (alu4 probe, addendum rules).

    model_v3.py --device h100|b300 --root <run dir> --out model.json \
        [--tree configs/trees/<dev>.json]

Per cell's declared composition (cells-v3.json):
  single: result = completion = lat slope; issue_interval = thr slope/8,
          /4 when the class_resource scope is sm (v3 resource rule).
  pair:   result(T) = lat slope - sum of installed partner
          result_latency (tree at branch head, measured evidence only,
          else unrealised); status capped at experimental.
          issue_interval only via the same-resource rule (target and
          partners share one class_resource, partner interval measured
          in this run or installed measured).
thr_only cells (DADD/DMUL dthr) install issue_interval only.
Acceptance: 5 % resid + 5 % run agreement -> measured; 15 % ->
experimental; else unrealised.  no_form classes -> unrealised:no_form.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mma_sync"))
import model as v1  # noqa: E402  (ols / slope_fit / thresholds)

_REPO = Path(__file__).resolve().parents[3]
_CELLS = Path(__file__).with_name("cells-v3.json")


def cell_series(rows, cell, mode):
    out = {}
    for r in rows:
        p = r.get("params", {})
        v = p.get("variant") or r.get("variant") or ""
        vmode = p.get("mode") or v.split(".")[0]
        vcell = str(v).split(".")[-1]
        if vmode != mode or vcell != cell:
            continue
        out.setdefault(int(r["u"]), []).extend(
            float(x) for x in r["samples"])
    return out


def n_unit(repo, cls, n_instr):
    """Instructions of the target class issued per K step on one copy of
    its unit: 32 x n for smsp-scoped units (4 warps x 8 chains), 128 x n
    for sm-scoped (16 warps x 8 chains)."""
    rname, _ = scope_divisor(repo, cls)
    return rname, (128 if rname is not None and _is_sm_scope(repo, rname)
                   else 32) * n_instr


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


def scope_divisor(repo, cls):
    """4 when the class's class_resource is sm-scoped with copy 1."""
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
    # resources.json: list of dicts with name/scope/copy_count
    entries = res if isinstance(res, list) else res.get("resources", [])
    for e in entries:
        if e.get("name") == rname:
            return rname, (4 if e.get("scope") == "sm"
                           and e.get("copy_count", 1) == 1 else 1)
    return rname, 1


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
    """Field carries >=1 source whose evidence is measured."""
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
    ap.add_argument("--tree", default=str(
        _REPO / "configs" / "trees" / "b300.json"))
    a = ap.parse_args()
    if a.tree.endswith("b300.json") and a.device == "h100":
        a.tree = str(_REPO / "configs" / "trees" / "h100.json")

    root = Path(a.root)
    rows = {run: v1.load(root / run) for run in ("run-a", "run-b")}
    checks_path = root / "build" / "sass" / "checks_alu4.json"
    checks = (json.loads(checks_path.read_text())
              if checks_path.is_file() else {"kernels": {}})
    cells_meta = json.loads(_CELLS.read_text())
    cells, no_form = cells_meta["cells"], cells_meta["no_form"]
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
        thr_mode = "dthr" if spec.get("thr_only") else "thr"
        ra = rb = None
        if not spec.get("thr_only"):
            ra = v1.slope_fit(cell_series(rows["run-a"], cell, "lat"))
            rb = v1.slope_fit(cell_series(rows["run-b"], cell, "lat"))
        ta = v1.slope_fit(cell_series(rows["run-a"], cell, thr_mode))
        tb = v1.slope_fit(cell_series(rows["run-b"], cell, thr_mode))
        rname, div = scope_divisor(_REPO, cell)
        e["resource"], e["scope_divisor"] = rname, div
        partners = spec.get("critical_path_partners", [])
        lat_slope = None
        if ra and rb:
            lat_slope = (ra["slope"] + rb["slope"]) / 2
            e["lat_resid"] = max(ra["max_rel_resid"],
                                 rb["max_rel_resid"])
            e["lat_agree"] = (abs(ra["slope"] - rb["slope"]) / lat_slope
                              if lat_slope else float("inf"))
            e["lat_runs"] = [ra["slope"], rb["slope"]]
        if spec["kind"] == "single" and lat_slope is not None:
            e["result_latency"] = lat_slope
            e["completion_latency"] = lat_slope
        elif spec["kind"] == "pair" and lat_slope is not None:
            used, missing = {}, []
            # partners not pinned in the composition (e.g. REDUX's UR->R
            # move): the gate-recorded counts decide — exactly one
            # partner class may appear at the per-step count; a mix of
            # two, or an unmeasured one, is unrealised
            extra = [c for c in partners if c not in spec["composition"]]
            if extra:
                steps_seen = Counter()
                for v in gk.values():
                    for c in extra:
                        steps_seen[c] = max(
                            steps_seen[c],
                            v.get("sass_counts", {}).get(c, 0))
                present = [c for c in extra if steps_seen[c] > 0]
                comp_steps = None
                for v in gk.values():
                    comp_steps = v.get("sass_counts", {}).get(
                        cell, 0)
                    break
                if len(present) != 1 or                         steps_seen[present[0]] != comp_steps:
                    e["status"] = "unrealised"
                    e["partner_mix"] = {c: steps_seen[c]
                                        for c in present}
                    classes[cell] = e
                    continue
                used_moves = {present[0]: 1}
            for cls, n in spec["composition"].items():
                if cls == cell:
                    continue
                leaf = service_leaf(tree, cls)
                ok = leaf and leaf_measured(
                    tree, leaf, "result_latency_cycles")
                if not ok:
                    missing.append(cls)
                else:
                    val = leaf_field(leaf, "result_latency_cycles")
                    used[cls] = val
            for cls, n in (used_moves if extra else {}).items():
                leaf = service_leaf(tree, cls)
                ok = leaf and leaf_measured(
                    tree, leaf, "result_latency_cycles")
                if not ok:
                    missing.append(cls)
                else:
                    used[cls] = leaf_field(
                        leaf, "result_latency_cycles")
            if missing:
                e["status"] = "unrealised"
                e["missing_partner_evidence"] = missing
                classes[cell] = e
                continue
            sub = sum(used[c] for c in used)
            e["partner_result_latency"] = used
            e["result_latency"] = lat_slope - sub
            e["completion_latency"] = lat_slope - sub
        if ta and tb:
            n_instr = spec["composition"].get(cell, 1)
            rname2, nu = n_unit(_REPO, cell, n_instr)
            e["n_unit"] = nu
            m = (ta["slope"] + tb["slope"]) / (2 * nu)
            e["thr_resid"] = max(ta["max_rel_resid"],
                                 tb["max_rel_resid"])
            e["thr_agree"] = abs(ta["slope"] - tb["slope"]) \
                / (m * 2 * nu) if m else float("inf")
            e["thr_runs"] = [ta["slope"] / nu, tb["slope"] / nu]
            if spec["kind"] == "pair":
                # same-resource rule: partner intervals must exist and
                # share the class_resource
                pres = {rname}
                pi, ok = {}, True
                for cls, n in spec["composition"].items():
                    if cls == cell:
                        continue
                    pr, _ = scope_divisor(_REPO, cls)
                    leaf = service_leaf(tree, cls)
                    iv = (leaf_field(leaf, "issue_interval_cycles")
                          if leaf else None)
                    if (pr != rname or iv is None
                            or not leaf_measured(
                                    tree, leaf,
                                    "issue_interval_cycles")):
                        ok = False
                    else:
                        pi[cls] = iv
                if ok:
                    e["partner_issue_interval"] = pi
                    e["issue_interval"] = \
                        m - sum(spec["composition"][c] * pi[c]
                                for c in pi)
            else:
                e["issue_interval"] = m
        resid = max(e.get("lat_resid", 0), e.get("thr_resid", 0))
        agree = max(e.get("lat_agree", 0), e.get("thr_agree", 0))
        if "result_latency" not in e and "issue_interval" not in e:
            e["status"] = "unrealised"
        elif resid <= 0.05 and agree <= 0.05:
            e["status"] = ("experimental" if spec["kind"] == "pair"
                           else "measured")
        elif resid <= 0.15 and agree <= 0.15:
            e["status"] = "experimental"
        else:
            e["status"] = "unrealised"
        classes[cell] = e
    # ---- addendum 2 rule 2: alu2 uniform cells (v2 derivation) ----
    ALU2 = {"SEL": "USEL", "IADD3": "UIADD3", "IMAD": "UIMAD",
            "LOP3": "ULOP3", "SHF": "USHF", "PRMT": "UPRMT",
            "POPC": "UPOPC", "FLO": "UFLO", "ISETP_SEL": "UISETP"}
    # alu2 cells that time a pair: cell -> partner class derived from
    # its own cell in the same run
    ALU2_PAIR = {"ISETP_SEL": "USEL"}
    checks2_path = root / "build" / "sass" / "checks_alu2.json"
    checks2 = (json.loads(checks2_path.read_text())
               if checks2_path.is_file() else {"kernels": {}})

    def a2_series(mode, op, wps=None):
        out = {}
        for run in ("run-a", "run-b"):
            for r in rows[run]:
                if r.get("probe") != "alu2":
                    continue
                v = r.get("variant", "")
                want = f"{mode}.{op}" + (f".wps{wps}" if wps else "")
                if v != want:
                    continue
                out.setdefault(run, {}).setdefault(
                    int(r["u"]), []).extend(
                    float(x) for x in r["samples"])
        return out

    for op, cls in ALU2.items():
        if cls in classes and classes[cls].get("status") in                 ("measured", "experimental"):
            continue
        lat_g = [v for v in checks2.get("kernels", {}).values()
                 if v["op"] == f"ulat.{op}"]
        thr_g = [v for v in checks2.get("kernels", {}).values()
                 if v["op"] == f"uthr.{op}"]
        lat_ok = bool(lat_g) and all(v["ok"] for v in lat_g)
        thr_ok = bool(thr_g) and all(v["ok"] for v in thr_g)
        e = {"cell": f"alu2.{op}", "kind": "alu2", "gate": lat_ok,
             "gate_failures": [v["why"] for v in lat_g + thr_g
                               if not v["ok"]]}
        la = v1.slope_fit(a2_series("ulat", op).get("run-a", {}))
        lb = v1.slope_fit(a2_series("ulat", op).get("run-b", {}))
        ta = v1.slope_fit(a2_series("uthr", op, 1).get("run-a", {}))
        tb = v1.slope_fit(a2_series("uthr", op, 1).get("run-b", {}))
        pair_partner = ALU2_PAIR.get(op)
        sub_r, sub_i = 0.0, 0.0
        if pair_partner:
            pe = classes.get(pair_partner, {})
            if pe.get("status") not in ("measured", "experimental"):
                e["status"] = "unrealised"
                e["missing_partner_evidence"] = [pair_partner]
                classes[cls] = e
                continue
            sub_r = pe.get("result_latency") or 0.0
            sub_i = pe.get("issue_interval") or 0.0
            e["partner"] = {pair_partner: {
                "result_latency": sub_r, "issue_interval": sub_i}}
        if lat_ok and la and lb:
            m = (la["slope"] + lb["slope"]) / 2
            e["result_latency"] = m - sub_r
            e["completion_latency"] = m - sub_r
            e["lat_resid"] = max(la["max_rel_resid"],
                                 lb["max_rel_resid"])
            e["lat_agree"] = (abs(la["slope"] - lb["slope"]) / m
                              if m else float("inf"))
            e["lat_runs"] = [la["slope"], lb["slope"]]
        if thr_ok and ta and tb:
            m = (ta["slope"] + tb["slope"]) / 16.0   # wps1: /8 chains, 2 runs
            e["issue_interval"] = m - sub_i
            e["thr_resid"] = max(ta["max_rel_resid"],
                                 tb["max_rel_resid"])
            e["thr_agree"] = abs(ta["slope"] - tb["slope"]) / (m * 16)                 if m else float("inf")
            e["thr_runs"] = [ta["slope"] / 8, tb["slope"] / 8]
        resid = max(e.get("lat_resid", 0), e.get("thr_resid", 0))
        agree = max(e.get("lat_agree", 0), e.get("thr_agree", 0))
        if "result_latency" not in e and "issue_interval" not in e:
            e["status"] = ("unrealised" if lat_ok or thr_ok
                           else "unrealised:no_form")
        elif resid <= 0.05 and agree <= 0.05:
            e["status"] = ("experimental" if pair_partner
                           else "measured")
        elif resid <= 0.15 and agree <= 0.15:
            e["status"] = "experimental"
        else:
            e["status"] = "unrealised"
        classes[cls] = e
    # ---- R2: SHFL leaf classes — group cells by leaf, 5 % agreement --
    shfl = {}
    for cell in ("SHFL_BFLY", "SHFL_UP", "SHFL_DOWN", "SHFL_IDX"):
        e = classes.get(cell)
        if not e or e.get("status") not in ("measured", "experimental"):
            continue
        leaf_cls = cell.replace("_", ".")
        if service_leaf(tree, leaf_cls) is None:
            leaf_cls = "SHFL"
        shfl.setdefault(leaf_cls, []).append((cell, e))
    for leaf_cls, grp in shfl.items():
        cells_g = [c for c, _ in grp]
        res = [e["result_latency"] for _, e in grp
               if e.get("result_latency") is not None]
        ivs = [e["issue_interval"] for _, e in grp
               if e.get("issue_interval") is not None]
        worst = max(e["status"] == "experimental" for _, e in grp)
        spread_r = (max(res) - min(res)) / (sum(res) / len(res)) \
            if res else 0
        spread_i = (max(ivs) - min(ivs)) / (sum(ivs) / len(ivs)) \
            if ivs else 0
        agree = max(spread_r, spread_i) <= 0.05
        m = {"cell": "+".join(cells_g), "kind": "single_group",
             "leaf_class": leaf_cls,
             "cell_results": {c: {"result_latency": e["result_latency"],
                                  "issue_interval":
                                  e["issue_interval"],
                                  "status": e["status"]}
                              for c, e in grp},
             "spread": {"result": spread_r, "interval": spread_i}}
        if res:
            m["result_latency"] = sum(res) / len(res)
            m["completion_latency"] = m["result_latency"]
        if ivs:
            m["issue_interval"] = sum(ivs) / len(ivs)
        if not agree:
            m["status"] = "experimental"
        elif worst:
            m["status"] = "experimental"
        else:
            m["status"] = "measured"
        classes[leaf_cls] = m
        for c in cells_g:
            del classes[c]
    for cls, why in no_form.items():
        classes.setdefault(cls, {"status": "unrealised:no_form",
                                 "why": why})
    out = {"protocol":
           "tools/microbench/service_latency/protocol-v3.json",
           "device": a.device,
           "runs": {"a": str(root / "run-a"), "b": str(root / "run-b")},
           "classes": classes}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for cls, e in classes.items():
        print(f"  {cls:12s} {e.get('result_latency')} / "
              f"{e.get('issue_interval')} -> {e['status']}")


if __name__ == "__main__":
    main()
