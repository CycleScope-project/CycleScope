#!/usr/bin/env python3
"""Install the b300 tensor_pipe protocol-v2 result into the device leaves
(tools/microbench/tensor_pipe/protocol-v2.json install.b300).

    python3 install.py --model $A/microbench/tensor_pipe-model/b300-r2/model.json \
        --source-root configs/mechanism_tree --run b300-r2

tensor.pipe: every normal-operation point (major 0, max shift 0) at a
measured (kind, cta_group, a_src, M, N, K) gets cycles = beat.
tensor.completion: re-partitioned like tensor.pipe (umma_kind / cta_group /
a_src) with the same discrete axes; measured points get the completion,
every other point of the pipe table gets 0 (those points keep the
predecessor's pipe = single-op latency semantics).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

AXES = ["tensor_m", "tensor_n", "tensor_k", "umma_operation", "umma_operand_major", "umma_max_shift"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path, nargs="+")
    ap.add_argument("--device", default="b300", choices=["b300", "h100"])
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--completion-ser", action="store_true",
                    help="h100-r4: patch tensor.completion points that "
                         "have a passing ser cell (addendum install)")
    a = ap.parse_args()
    if a.device == "h100" and a.completion_ser:
        install_h100_ser(a.model[0], a.source_root,
                         a.run.replace("h100-", ""))
        return
    if a.device == "h100":
        install_h100(a.model, a.source_root)
        return
    model = json.loads(a.model[0].read_text())
    cells = {}
    for v in model["cells"].values():
        if v["layout"] != "sw128":
            continue
        key = (v["kind"], v["cta_group"], v["a_src"], v["installed_m"], v["n"])
        cells[key] = v
    src_m, src_e = f"tensor_pipe.{a.run.replace('-', '.')}", f"tensor_pipe.{a.run.replace('-', '.')}.experimental"
    src_zero = "tensor_pipe.structural.completion_unmeasured_points"
    k_of = {"f16": 16, "tf32": 8, "f8f6f4": 32, "mxf8f6f4": 32, "mxf4": 64, "mxf4nvf4": 64}

    pipe_path = a.source_root / "blackwell/leaves/tensor.pipe.json"
    comp_path = a.source_root / "blackwell/leaves/tensor.completion.json"
    pipe = json.loads(pipe_path.read_text())
    comp = json.loads(comp_path.read_text())
    replaced = added = 0
    for cls_key, cls_node in pipe.items():
        if "children" not in cls_node:
            continue
        comp_cls = copy.deepcopy(cls_node)
        for kind_node, comp_kind in zip(cls_node["children"], comp_cls["children"]):
            kind = kind_node["guard"]["value"]
            for cg_node, comp_cg in zip(kind_node["children"], comp_kind["children"]):
                cg = cg_node["guard"]["value"]
                for leaf, comp_leaf in zip(cg_node["children"], comp_cg["children"]):
                    asrc = leaf["guard"]["value"]
                    f = leaf["parameter"]["fields"]["cycles"]
                    pts = f["model"]["points"]
                    have = {tuple(p[:6]): i for i, p in enumerate(pts)}
                    comp_pts = []
                    used_exp = used = False
                    for (kd, g, s, m, n), v in cells.items():
                        if (kd, g, s) != (kind, cg, asrc):
                            continue
                        coord = (m, n, k_of[kind], "normal", 0, 0)
                        if coord in have:
                            pts[have[coord]][6] = v["beat"]
                            replaced += 1
                        else:
                            pts.append(list(coord) + [v["beat"]])
                            have[coord] = len(pts) - 1
                            added += 1
                        used = True
                        used_exp |= v["beat_status"] != "measured"
                    pts.sort(key=lambda p: (p[0], p[1], p[2], str(p[3]), p[4], p[5]))
                    for p in pts:
                        coord = tuple(p[:6])
                        v = cells.get((kind, cg, asrc, p[0], p[1])) if p[2] == k_of[kind] and p[3:6] == ["normal", 0, 0] else None
                        comp_pts.append(list(coord) + [v["completion_install"] if v else 0.0])
                    if used:
                        f["sources"] = sorted(set(f["sources"]) | {src_m} | ({src_e} if used_exp else set()))
                    comp_leaf["parameter"] = {"fields": {
                        "cycles": {"model": dict({k: x for k, x in f["model"].items() if k != "points"},
                                                 points=comp_pts),
                                   "sources": [src_m, src_zero] + ([src_e] if used_exp else [])},
                        "enabled": {"model": {"kind": "constant", "value": 1},
                                    "sources": ["blackwell.structural.tensor_completion_zero"]}}}
        comp[cls_key.replace("tensor.pipe", "tensor.completion")] = comp_cls
    pipe_path.write_text(json.dumps(pipe, indent=1) + "\n")
    comp_path.write_text(json.dumps(comp, indent=1) + "\n")
    print(f"pipe points replaced {replaced}, added {added}")



def install_h100(models: list[Path], source_root: Path):
    """h100: per HGMMA / QGMMA / IGMMA, tensor.pipe and tensor.completion =
    discrete over [input_dtype, (accumulator_dtype for HGMMA), source_mode,
    tensor_n] at the measured points (protocol-v2 install.h100; HGMMA f16 F16-
    vs F32-accumulator beats differ > 10 %, so HGMMA carries the accumulator
    axis). ss cells from h100-r1h, rs cells from h100-r3 (addenda)."""
    cls_of = {"f16": "HGMMA", "f16a": "HGMMA", "bf16": "HGMMA", "tf32": "HGMMA",
              "e4m3e4m3": "QGMMA", "e5m2e5m2": "QGMMA", "s8s8": "IGMMA"}
    # e4m3e5m2 maps to input_dtype e4m3 like e4m3e4m3 (identical cells); not installed separately
    pts = {c: {"pipe": [], "comp": []} for c in ("HGMMA", "QGMMA", "IGMMA")}
    srcs = {c: set() for c in pts}
    for mp in models:
        m = json.loads(mp.read_text())
        run = Path(m["run_dir"]).name.replace("h100-", "")
        for v in m["cells"].values():
            if v["kind"] not in cls_of or v["beat_status"] == "unrealised":
                continue
            # r1h supplies ss only, r3 rs only (addendum)
            if (run == "r1h") != (v["src"] == "ss"):
                continue
            cls = cls_of[v["kind"]]
            key = [v["input_dtype"]] + ([v["acc"]] if cls == "HGMMA" else []) + [v["src"], v["n"]]
            pts[cls]["pipe"].append(key + [v["beat"]])
            ci = v["completion_install"]
            pts[cls]["comp"].append(key + [ci if ci is not None else 0.0])
            srcs[cls].add(f"tensor_pipe.h100.{run}" + ("" if v["beat_status"] == "measured" else ".experimental"))
    pipe_path = source_root / "hopper/leaves/tensor.pipe.json"
    comp_path = source_root / "hopper/leaves/tensor.completion.json"
    pipe = json.loads(pipe_path.read_text())
    comp = json.loads(comp_path.read_text())
    for cls in pts:
        axes = ["input_dtype"] + (["accumulator_dtype"] if cls == "HGMMA" else []) + ["source_mode", "tensor_n"]
        key = f"tensor.pipe.root/operation_class={cls}"
        pipe[key] = {"fields": {
            "cycles": {"model": {"kind": "discrete", "axes": axes, "points": sorted(pts[cls]["pipe"], key=str)},
                       "sources": sorted(srcs[cls])},
            "enabled": {"model": {"kind": "constant", "value": 1}, "sources": sorted(srcs[cls])}}}
        comp[key.replace("tensor.pipe", "tensor.completion")] = {"fields": {
            "cycles": {"model": {"kind": "discrete", "axes": axes, "points": sorted(pts[cls]["comp"], key=str)},
                       "sources": ["tensor_pipe.h100.completion.experimental"]},
            "enabled": {"model": {"kind": "constant", "value": 1},
                        "sources": ["tensor_pipe.h100.completion.experimental"]}}}
    pipe_path.write_text(json.dumps(pipe, indent=1) + "\n")
    comp_path.write_text(json.dumps(comp, indent=1) + "\n")
    print({c: len(p["pipe"]) for c, p in pts.items()})


def install_h100_ser(model_path: Path, source_root: Path, run: str):
    """Patch tensor.completion points that have a passing ser cell
    (protocol-v2 addendum install.h100): cycles = completion_ser at the
    cell's (input_dtype, accumulator_dtype, source_mode, tensor_n)
    point; every other point keeps its present value and sources."""
    cls_of = {"f16": "HGMMA", "f16a": "HGMMA", "bf16": "HGMMA",
              "tf32": "HGMMA", "e4m3e4m3": "QGMMA",
              "e4m3e5m2": "QGMMA", "e5m2e5m2": "QGMMA", "s8s8": "IGMMA"}
    m = json.loads(model_path.read_text())
    comp_path = source_root / "hopper/leaves/tensor.completion.json"
    comp = json.loads(comp_path.read_text())
    src = f"tensor_pipe.h100.completion.ser.{run}"
    src_e = src + ".experimental"
    npts = 0
    used_exp = False
    for v in m["cells"].values():
        st = v.get("ser_status")
        if st not in ("measured", "experimental"):
            continue
        ci = v.get("completion_ser_install")
        if ci is None or v["kind"] not in cls_of:
            continue
        cls = cls_of[v["kind"]]
        key = [v["input_dtype"]] +             ([v["acc"]] if cls == "HGMMA" else []) + [v["src"], v["n"]]
        fkey = f"tensor.completion.root/operation_class={cls}"
        flds = comp[fkey]["fields"]
        hit = False
        for pt in flds["cycles"]["model"]["points"]:
            if pt[:len(key)] == key:
                pt[len(key)] = ci
                hit = True
                npts += 1
        if hit:
            used_exp |= st != "measured"
            for sect in ("cycles", "enabled"):
                srcs = set(flds[sect].get("sources", []))
                srcs.add(src)
                if st != "measured":
                    srcs.add(src_e)
                flds[sect]["sources"] = sorted(srcs)
    comp_path.write_text(json.dumps(comp, indent=1) + "\n")
    print(f"ser completion points patched: {npts} "
          f"(experimental used: {used_exp})")


if __name__ == "__main__":
    main()
