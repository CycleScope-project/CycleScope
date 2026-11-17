#!/usr/bin/env python3
"""Install tensor_pipe_forms results into the b300 tensor.pipe /
tensor.completion leaves (protocol install text + addendum 2026-11-05).

    python3 install.py --model $A/microbench/tensor_pipe_forms-model/b300-r2/model.json \
        --source-root configs/mechanism_tree --run b300-r2 [--report <md>] \
        [--only-forms <json list of form ids>] [--v2-points <json list of [node, coord]>]

--list-v2-remaining prints the [node, coord] pairs still holding
tensor_pipe v2 values per the r2 model (v2_coordinate forms whose
completion_install is None) and installs nothing:

    python3 install.py --list-v2-remaining --model <r2 model.json> \
        --source-root configs/mechanism_tree

A form is installed when its beat status is measured / experimental and
its completion is installable (completion_install not None): the point
(tensor_m, tensor_n, tensor_k, umma_operation, umma_operand_major,
umma_max_shift) under the form's (class, umma_kind, cta_group, a_src) node
gets tensor.pipe = beat and tensor.completion = completion.  Points whose
form failed, or that no form covers, keep their current values.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

KIND_CLASS = {"f16": "UTCHMMA", "tf32": "UTCHMMA", "f8f6f4": "UTCQMMA",
              "mxf8f6f4": "UTCQMMA", "mxf4": "UTCOMMA", "mxf4nvf4": "UTCOMMA"}


def leaves(tree: dict):
    for cls_key, cls_node in tree.items():
        if "children" not in cls_node:
            continue
        for kind_node in cls_node["children"]:
            for cg_node in kind_node["children"]:
                for leaf in cg_node["children"]:
                    yield (cls_key.split("=")[-1], kind_node["guard"]["value"], cg_node["guard"]["value"],
                           leaf["guard"]["value"]), leaf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--run", default="")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--only-forms", type=Path, default=None,
                    help="json list of form ids; install only these")
    ap.add_argument("--v2-points", type=Path, default=None,
                    help="json list of [node, coord] pairs still holding "
                         "tensor_pipe v2 values; when a node's listed points "
                         "are all replaced, drop tensor_pipe.b300.r2* sources")
    ap.add_argument("--list-v2-remaining", action="store_true")
    a = ap.parse_args()
    model = json.loads(a.model.read_text())
    if a.list_v2_remaining:
        out = []
        for fid, v in model["forms"].items():
            if v.get("v2_coordinate") and v.get("completion_install") is None:
                p = v["params"]
                out.append([[KIND_CLASS[p["kind"]], p["kind"], int(p["cta_group"]), p["a_src"]],
                            v["coord"]])
        print(json.dumps(out, indent=1))
        return
    if not a.run:
        ap.error("--run is required unless --list-v2-remaining")
    run = a.run.replace("-", ".")
    src_m, src_e = f"tensor_pipe_forms.{run}", f"tensor_pipe_forms.{run}.experimental"
    only = set(json.loads(a.only_forms.read_text())) if a.only_forms else None
    v2_left = defaultdict(set)
    if a.v2_points:
        for node, coord in json.loads(a.v2_points.read_text()):
            v2_left[tuple(node)].add(tuple(coord))
    by_node = defaultdict(dict)
    for fid, v in model["forms"].items():
        if only is not None and fid not in only:
            continue
        if v.get("status") == "unrealised" or v.get("beat_status") in (None, "unrealised"):
            continue
        if v.get("completion_install") is None:
            continue
        p = v["params"]
        node = (KIND_CLASS[p["kind"]], p["kind"], int(p["cta_group"]), p["a_src"])
        by_node[node][tuple(v["coord"])] = v
    pipe_path = a.source_root / "blackwell/leaves/tensor.pipe.json"
    comp_path = a.source_root / "blackwell/leaves/tensor.completion.json"
    pipe, comp = json.loads(pipe_path.read_text()), json.loads(comp_path.read_text())
    comp_leaves = dict(leaves(comp))
    stats = Counter()
    rows = []
    for node, leaf in leaves(pipe):
        cells = by_node.get(node, {})
        if not cells:
            continue
        pf = leaf["parameter"]["fields"]["cycles"]
        cl = comp_leaves[node]["parameter"]["fields"]["cycles"]
        cpts = {tuple(p[:6]): p for p in cl["model"]["points"]}
        used_exp = used = False
        replaced = set()
        for pt in pf["model"]["points"]:
            v = cells.get(tuple(pt[:6]))
            if v is None:
                stats["points_not_covered"] += 1
                continue
            old_pipe, old_comp = pt[6], cpts[tuple(pt[:6])][6] if tuple(pt[:6]) in cpts else None
            rows.append((node, tuple(pt[:6]), old_pipe, old_comp, v["beat"], v["completion_install"],
                         v["beat_status"], v.get("v2_coordinate")))
            pt[6] = round(v["beat"], 3)
            if tuple(pt[:6]) in cpts:
                cpts[tuple(pt[:6])][6] = round(v["completion_install"], 3)
            else:
                cl["model"]["points"].append(list(pt[:6]) + [round(v["completion_install"], 3)])
            used = True
            used_exp |= v["beat_status"] != "measured"
            replaced.add(tuple(pt[:6]))
            stats["points_replaced"] += 1
            stats["v2_points_replaced" if v.get("v2_coordinate") else "other_points_replaced"] += 1
        covered = {tuple(p[:6]) for p in pf["model"]["points"]}
        for c in cells:
            if c in covered:
                continue
            v = cells[c]
            pf["model"]["points"].append(list(c) + [round(v["beat"], 3)])
            if c in cpts:
                cpts[c][6] = round(v["completion_install"], 3)
            else:
                cl["model"]["points"].append(
                    list(c) + [round(v["completion_install"], 3)])
            used = True
            used_exp |= v["beat_status"] != "measured"
            replaced.add(c)
            stats["points_added"] += 1
        if used:
            pf["sources"] = sorted(set(pf["sources"]) | {src_m} | ({src_e} if used_exp else set()))
            cl["sources"] = sorted(set(cl["sources"]) | {src_m} | ({src_e} if used_exp else set()))
            for f in (pf, cl):
                f["model"]["points"].sort(
                    key=lambda p: (p[0], p[1], p[2], str(p[3]), p[4], p[5]))
        if a.v2_points and node in v2_left and not (v2_left[node] - replaced):
            # every v2-valued point of this node was replaced: the v2
            # sources no longer back any point here
            for f in (pf, cl):
                f["sources"] = [s for s in f.get("sources", [])
                                if s not in ("tensor_pipe.b300.r2",
                                             "tensor_pipe.b300.r2.experimental")]
            stats["nodes_v2_sources_dropped"] += 1
    pipe_path.write_text(json.dumps(pipe, indent=1) + "\n")
    comp_path.write_text(json.dumps(comp, indent=1) + "\n")
    print(dict(stats))
    if a.report:
        lines = ["| node | point (m, n, k, op, major, shift) | pipe old -> new | completion old -> new | status | v2 |", "|---|---|---|---|---|---|"]
        for node, c, op_, oc, nb, nc, st, v2 in rows:
            lines.append(f"| {'/'.join(map(str, node))} | {c} | {op_} -> {nb:.2f} | {oc} -> {nc:.2f} | {st} | {'yes' if v2 else ''} |")
        a.report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
