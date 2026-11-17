#!/usr/bin/env python3
"""Expand umma_factor_matrix.json into one-factor-at-a-time encoded forms."""

import argparse
import json
import re
from pathlib import Path

from umma_factor_encode import (
    SWIZZLE_REPEAT,
    base_offset_for,
    canonical_lbo_sbo,
    descriptor_view_span,
    encode_idesc,
    encode_smem,
    one_encoding,
)

K_BY_KIND = {
    "f16": 16, "tf32": 8, "f8f6f4": 32, "mxf8f6f4": 32,
    "mxf4": 64, "mxf4nvf4": 64,
}
BASE_TYPES = {
    "f16": ("F16", "F16", "F32", "UE8M0"),
    "tf32": ("TF32", "TF32", "F32", "UE8M0"),
    "f8f6f4": ("E4M3", "E4M3", "F32", "UE8M0"),
    "mxf8f6f4": ("E4M3", "E4M3", "F32", "UE8M0"),
    "mxf4": ("E2M1", "E2M1", "F32", "UE8M0"),
    "mxf4nvf4": ("E2M1", "E2M1", "F32", "UE4M3"),
}
KIND_ID = {
    "f16": 0, "tf32": 1, "f8f6f4": 2, "mxf8f6f4": 3,
    "mxf4": 4, "mxf4nvf4": 5,
}
FACTOR_STAGES = 2
A_STAGE_BYTES = 52 * 1024
B_STAGE_BYTES = 52 * 1024
CELL_STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES
B_ZERO_BYTES = 16 * 1024
MAX_CTA_SMEM_BYTES = 227 * 1024

def values_in_braces(text, name):
    match = re.search(rf"{name}\s+in\s*\{{([^}}]+)\}}", text)
    if not match:
        raise ValueError(f"missing {name} levels in factor matrix")
    return tuple(int(x.strip()) for x in match.group(1).split(","))


def factor_transposes(cell_id, cell):
    if cell_id == "C4":
        return [("K", "K"), ("K", "MN")]
    return [("K", "K"), ("MN", "MN")]


def make_config(cell_id, cell, *, a_type=None, b_type=None, d_type=None,
                negate=("none", "none"), transpose=("K", "K"),
                swizzle="SW128", starts=(0, 0), lbo_mode="relative",
                lbo_override=None, sbo_override=None, base_offset=False,
                sf_id=(0, 0), scale_type=None, reuse=False,
                operation="normal", max_shift=0, b_zero=False, sparse=False,
                sparsity_version=0, overlap="none", acc=1, arot=0, brot=0, grp=0, issuers=1, kalu=0,
                elected=0):
    kind = cell["kind"]
    # r6 addendum: per-operand swizzle — an MN-major tf32 operand is
    # SWIZZLE_128B_BASE32B (the only accepted MN layout for 32-bit
    # types); every other operand keeps the form-level swizzle.
    sw_a = "SW128_32B_ATOM" if kind == "tf32" and transpose[0] == "MN" else swizzle
    sw_b = "SW128_32B_ATOM" if kind == "tf32" and transpose[1] == "MN" else swizzle
    default_a, default_b, default_d, default_scale = BASE_TYPES[kind]
    a_type = a_type or default_a
    b_type = b_type or default_b
    d_type = d_type or default_d
    scale_type = scale_type or default_scale
    m, n, cg = int(cell["M"]), int(cell["N"]), int(cell["cta_group"])
    k = int(cell.get("K", K_BY_KIND[kind]))
    is_mx = kind.startswith("mx")
    idesc = encode_idesc(
        kind, m, n, a_type, b_type, d_type,
        transpose=transpose, negate=negate, sf_id=sf_id,
        scale_type=scale_type, max_shift=max_shift, k=k, sparse=sparse,
        sparsity_version=sparsity_version)
    a_desc = None
    lbo_a = sbo_a = None
    a_abs = False
    if cell["a_src"] == "smem":
        lbo_a, sbo_a = canonical_lbo_sbo(m // cg, sw_a, transpose[0])
        if lbo_override and lbo_override[0] is not None:
            lbo_a = lbo_override[0]
        if sbo_override and sbo_override[0] is not None:
            sbo_a = sbo_override[0]
        a_abs = lbo_mode == "absolute"
        if a_abs:
            lbo_a = 0
        bo_a = base_offset_for(starts[0], sw_a) if base_offset else 0
        a_desc = encode_smem(
            0, lbo_a, sbo_a, sw_a, lbo_mode=lbo_mode,
            base_offset=bo_a)
    lbo_b, sbo_b = canonical_lbo_sbo(n // cg, sw_b, transpose[1])
    if lbo_override and lbo_override[1] is not None:
        lbo_b = lbo_override[1]
    if sbo_override and sbo_override[1] is not None:
        sbo_b = sbo_override[1]
    b_abs = lbo_mode == "absolute"
    if b_abs:
        lbo_b = 0
    bo_b = base_offset_for(starts[1], sw_b) if base_offset else 0
    b_desc = encode_smem(
        0, lbo_b, sbo_b, sw_b, lbo_mode=lbo_mode,
        base_offset=bo_b)
    if overlap == "same":
        if a_desc is None:
            raise ValueError("identical A/B views require an SMEM A operand")
        lbo_b, sbo_b, b_abs, bo_b = lbo_a, sbo_a, a_abs, bo_a
        b_desc = encode_smem(
            0, lbo_b, sbo_b, sw_b, lbo_mode=lbo_mode,
            base_offset=bo_b)
    a_bits, a_one = one_encoding(a_type)
    b_bits, b_one = one_encoding(b_type)
    if is_mx:
        sf_bits, sf_one = one_encoding(scale_type)
    else:
        sf_bits, sf_one = 8, 0
    sign = -1 if (negate[0] != negate[1]) else 1
    return {
        "cell_id": cell_id, "kind": kind, "cta_group": cg,
        "a_src": cell["a_src"], "M": m, "N": n, "K": k,
        "operation": operation,
        "ws": operation.lower() in {"ws", "ws_sp"},
        "sparse": sparse, "sparsity_version": sparsity_version,
        "overlap": overlap, "overlap_bytes": 0, "acc": acc,
        "arot": arot, "brot": brot, "grp": grp,
        "issuers": issuers, "kalu": kalu, "elected": elected,
        "factor_idesc": idesc, "smem_a": a_desc, "smem_b": b_desc,
        "start_a": starts[0] if cell["a_src"] == "smem" else 0,
        "start_b": starts[0] if overlap == "same" else starts[1],
        "a_type": a_type, "b_type": b_type, "d_type": d_type,
        "a_bits": a_bits, "b_bits": b_bits, "a_one": a_one, "b_one": b_one,
        "scale_type": scale_type, "scale_bits": sf_bits, "scale_one": sf_one,
        "sf_id_a": sf_id[0], "sf_id_b": sf_id[1],
        "sign": sign, "d_f16": d_type == "F16",
        "reuse": reuse, "lbo_abs_a": a_abs, "lbo_abs_b": b_abs,
        "b_zero": b_zero,
        "max_shift": max_shift, "swizzle": swizzle,
        "swizzle_a": sw_a, "swizzle_b": sw_b,
        "transpose": list(transpose), "negate": list(negate),
        "expected_d": {
            "formula": "sign * nonzero_K * MMAs",
            "sign": sign, "K": k // 2 if sparse else k,
            "d_type": d_type,
        },
        "lbo_sbo": {
            "a_lbo": lbo_a if not a_abs else None, "a_sbo": sbo_a if cell["a_src"] == "smem" else None,
            "b_lbo": lbo_b if not b_abs else None, "b_sbo": sbo_b,
        },
        "base_offset": base_offset,
    }


def add_form(forms, cell_id, cell, factor, level, **kwargs):
    kwargs.setdefault("swizzle", cell.get("swizzle", "SW128"))
    config = make_config(cell_id, cell, **kwargs)
    config.update({"factor": factor, "level": str(level)})
    config["form_id"] = f"{cell_id}.{factor}.{level}"
    a_span = 0
    if config["a_src"] == "smem":
        a_span = descriptor_view_span(
            config["M"] // config["cta_group"],
            config["K"] // 2 if config["sparse"] else config["K"],
            config["a_bits"], config["swizzle_a"], config["transpose"][0],
            config["lbo_sbo"]["a_lbo"] or 0,
            config["lbo_sbo"]["a_sbo"] or 0)
    b_span = descriptor_view_span(
        config["N"] // config["cta_group"], config["K"], config["b_bits"],
        config["swizzle_b"], config["transpose"][1],
        config["lbo_sbo"]["b_lbo"] or 0,
        config["lbo_sbo"]["b_sbo"] or 0)
    config["view_bytes"] = {"a": a_span, "b": b_span}
    reason = []
    if config["start_a"] + a_span > A_STAGE_BYTES:
        reason.append("A descriptor view exceeds stage A")
    if config["start_b"] + b_span > B_STAGE_BYTES:
        reason.append("B descriptor view exceeds stage B")
    if config["overlap"] == "partial":
        if config["a_src"] != "smem":
            reason.append("partial A/B overlap requires an SMEM A operand")
        else:
            config["overlap_bytes"] = max(
                16, (min(a_span, b_span) // 2 // 16) * 16)
    elif config["overlap"] == "same":
        if config["a_src"] != "smem":
            reason.append("identical A/B views require an SMEM A operand")
        else:
            config["overlap_bytes"] = 0
    required_a = max(A_STAGE_BYTES, config["start_a"] + a_span)
    required_b = max(B_STAGE_BYTES, config["start_b"] + b_span)
    config["required_stage_bytes"] = {
        "a": required_a, "b": required_b,
    }
    config["required_smem_bytes"] = (
        FACTOR_STAGES * (required_a + required_b) +
        B_ZERO_BYTES + 64 + 1024
    )
    config["required_stage_bytes"] = {
        "a": required_a, "b": required_b,
    }
    if config["required_smem_bytes"] > MAX_CTA_SMEM_BYTES:
        reason.append(
            f"requires {config['required_smem_bytes']} shared-memory bytes/CTA "
            f"(limit {MAX_CTA_SMEM_BYTES})")
    config["run_status"] = "unmeasurable" if reason else "candidate"
    config["skip_reason"] = reason
    forms.append(config)


def measured_base_leaves(profile_path):
    text = Path(profile_path).read_text()
    leaves = set()
    for suffix in re.findall(
            r"mechtree\.w6\.umma\.latency\.([a-z0-9.]+)", text):
        parts = suffix.split(".")
        if len(parts) == 4:
            leaves.add((parts[0], int(parts[1][2:]), parts[2],
                        int(parts[3][1:])))
    if leaves:
        return leaves

    profile = json.loads(text)
    tree = profile.get("mechanism_tree", {})
    root = tree.get("mechanisms", {}).get("tensor.umma.issue")
    nodes = {
        node["mechanism_id"]: node for node in tree.get("nodes", [])
    }
    if root not in nodes:
        return leaves

    def visit(node_id, state):
        node = nodes[node_id]
        current = dict(state)
        for guard in node.get("guards", []):
            if guard["operator"] == "eq":
                current[guard["axis"]] = guard["value"]
        if node.get("children"):
            for child in node["children"]:
                visit(child, current)
            return
        parameter = node.get("parameter", {})
        if parameter.get("status") == "unsupported":
            return
        field = next(
            (item for item in parameter.get("fields", [])
             if item["effect"] == "cycles"),
            None,
        )
        if field is None:
            return
        model = field.get("model", {})
        if model.get("kind") != "discrete":
            return
        kind = current.get("umma_kind")
        cta_group = current.get("cta_group")
        a_src = current.get("a_src")
        if kind is None or cta_group is None or a_src is None:
            return
        for point in model.get("points", []):
            if point:
                leaves.add((kind, cta_group, a_src, int(point[0])))

    visit(root, {})
    return leaves


def expand(matrix, profile_path):
    forms = []
    leaves = measured_base_leaves(profile_path)
    cells = matrix["cells"]
    for cell_id, cell in cells.items():
        add_form(forms, cell_id, cell, "base", "base")
    for cell_id, cell in matrix["operation_ws"]["cells"].items():
        add_form(forms, cell_id, cell, "base", "base")

    factors = matrix["factors"]
    for cell_id in factors["ab_type"]["cells"]:
        cell = cells[cell_id]
        for level in factors["ab_type"]["levels"][cell["kind"]]:
            a_type, b_type = level.split("x")
            if (a_type, b_type) == BASE_TYPES[cell["kind"]][:2]:
                continue
            add_form(forms, cell_id, cell, "ab_type", level,
                     a_type=a_type, b_type=b_type)

    for cell_id in factors["d_type"]["cells"]:
        cell = cells[cell_id]
        for level in factors["d_type"]["levels"]:
            if level == "F32":
                continue
            add_form(forms, cell_id, cell, "d_type", level, d_type=level)

    for cell_id in factors["negate"]["cells"]:
        cell = cells[cell_id]
        levels = factors["negate"]["levels"]
        for level in levels:
            if level == "none":
                continue
            neg = {"none": ("none", "none"), "A": ("A", "none"),
                   "B": ("none", "B")}[level]
            add_form(forms, cell_id, cell, "negate", level, negate=neg)

    for cell_id in factors["transpose"]["cells"]:
        cell = cells[cell_id]
        levels = factors["transpose"].get("levels_by_cell", {}).get(
            cell_id, factors["transpose"]["levels"][cell["a_src"]])
        for level in levels:
            pair = tuple(level.split(","))
            if pair == ("K", "K"):
                continue
            add_form(forms, cell_id, cell, "transpose", level,
                     transpose=pair)

    for cell_id in factors["swizzle"]["cells"]:
        cell = cells[cell_id]
        pairs = factor_transposes(cell_id, cell)
        for swizzle in factors["swizzle"]["levels"]:
            for pair in pairs:
                if swizzle == "SW128_32B_ATOM" and "K" in pair:
                    continue
                if swizzle == "SW128" and pair == ("K", "K"):
                    continue
                if cell["a_src"] == "tmem" and pair[0] != "K":
                    continue
                add_form(forms, cell_id, cell, "swizzle",
                         f"{swizzle}:{pair[0]},{pair[1]}",
                         swizzle=swizzle, transpose=pair)

    start_cells = factors["start"]["cells"]
    offpattern = (128, 160, 192, 224, 256, 288, 320, 352, 384, 512)
    for cell_id in start_cells:
        cell = cells[cell_id]
        swizzles = factors["swizzle"]["levels"]
        pairs = cell_transposes_for_start(cell_id, cell)
        k = K_BY_KIND[cell["kind"]]
        start_text = factors["start"]["levels"]
        offset_match = re.search(r"off-pattern offsets seen in c450\s*\(([^)]+)\)",
                                 start_text)
        offpattern = tuple(
            int(re.search(r"\d+", x).group(0))
            for x in offset_match.group(1).split(",")
        ) if offset_match else ()
        for swizzle in swizzles:
            step = k * one_encoding(BASE_TYPES[cell["kind"]][0])[0] // 8
            offsets = sorted(set(
                (0, step, 2 * step, 3 * step, SWIZZLE_REPEAT[swizzle],
                 *offpattern)))
            for pair in pairs:
                if swizzle == "SW128_32B_ATOM" and "K" in pair:
                    continue
                if cell["a_src"] == "tmem" and pair[0] != "K":
                    continue
                for offset in offsets:
                    if offset == 0:
                        continue
                    starts = (offset if cell["a_src"] == "smem" else 0, offset)
                    add_form(forms, cell_id, cell, "start",
                             f"{swizzle}:{pair[0]},{pair[1]}@{offset}",
                             swizzle=swizzle, transpose=pair, starts=starts)

    for cell_id in factors["lbo_sbo"]["cells"]:
        cell = cells[cell_id]
        pairs = cell_transposes_for_start(cell_id, cell)
        lbo_values = values_in_braces(
            factors["lbo_sbo"]["levels"], "LBO")
        sbo_values = tuple(sorted(set(
            (0,) + values_in_braces(factors["lbo_sbo"]["levels"], "SBO")
        )))
        for swizzle in factors["swizzle"]["levels"]:
            for pair in pairs:
                if swizzle == "SW128_32B_ATOM" and "K" in pair:
                    continue
                base = make_config(cell_id, cell, swizzle=swizzle, transpose=pair)
                for role, major in (("a", pair[0]), ("b", pair[1])):
                    if role == "a" and cell["a_src"] == "tmem":
                        continue
                    for value in lbo_values:
                        base_lbo = base["lbo_sbo"][f"{role}_lbo"]
                        if value == base_lbo:
                            continue
                        lbo_override = [None, None]
                        lbo_override[0 if role == "a" else 1] = value
                        add_form(forms, cell_id, cell, "lbo_sbo",
                                 f"{swizzle}:{pair[0]},{pair[1]}:{role}.LBO={value}",
                                 swizzle=swizzle, transpose=pair,
                                 lbo_override=lbo_override)
                    for value in sbo_values:
                        base_sbo = base["lbo_sbo"][f"{role}_sbo"]
                        if value == base_sbo:
                            continue
                        sbo_override = [None, None]
                        sbo_override[0 if role == "a" else 1] = value
                        add_form(forms, cell_id, cell, "lbo_sbo",
                                 f"{swizzle}:{pair[0]},{pair[1]}:{role}.SBO={value}",
                                 swizzle=swizzle, transpose=pair,
                                 sbo_override=sbo_override)

    for cell_id in factors["lbo_mode"]["cells"]:
        cell = cells[cell_id]
        add_form(forms, cell_id, cell, "lbo_mode", "absolute",
                 lbo_mode="absolute")

    for cell_id in factors["base_offset"]["cells"]:
        cell = cells[cell_id]
        add_form(forms, cell_id, cell, "base_offset", "pattern_start_128",
                 starts=(128, 128), base_offset=True)

    for cell_id in factors["sf_id"]["cells"]:
        cell = cells[cell_id]
        levels = factors["sf_id"]["levels"][cell["kind"]]
        for level in levels:
            pair = tuple(int(x) for x in level.split("/"))
            if pair == (0, 0):
                continue
            add_form(forms, cell_id, cell, "sf_id", level, sf_id=pair)

    for cell_id in factors["scale_type"]["cells"]:
        cell = cells[cell_id]
        for level in factors["scale_type"]["levels"]:
            if level == BASE_TYPES[cell["kind"]][3]:
                continue
            add_form(forms, cell_id, cell, "scale_type", level,
                     scale_type=level)

    for cell_id in factors["operand_reuse"]["cells"]:
        cell = cells[cell_id]
        add_form(forms, cell_id, cell, "operand_reuse", "same_descriptor",
                 reuse=True)

    for cell_id, cell in matrix["operation_ws"]["cells"].items():
        for level in matrix["operation_ws"]["levels"]:
            if level == "default":
                continue
            shift = int(level.rsplit(" ", 1)[1])
            add_form(forms, cell_id, cell, "operation_ws",
                     f"max_shift_{shift}", operation="WS", max_shift=shift)
    for form in forms:
        form["base_leaf"] = (
            form["kind"], form["cta_group"], form["a_src"], form["M"]
        ) in leaves
        if not form["base_leaf"]:
            form["run_status"] = "unmeasurable"
            form["skip_reason"].append("no measured base leaf for this cell")
    base_status = {
        form["cell_id"]: form["run_status"]
        for form in forms if form["factor"] == "base"
    }
    for form in forms:
        if (form["factor"] != "base" and
                base_status.get(form["cell_id"]) != "candidate"):
            form["run_status"] = "unmeasurable"
            form["skip_reason"].append("cell base form is not measurable")
        if form["swizzle"] == "SW128_32B_ATOM":
            form["run_status"] = "unmeasurable"
            form["skip_reason"].append(
                "functional smoke d_ok=false (C1 returned 0 for expected 16); "
                "canonical base32 descriptor stride encoding unresolved")
    base_forms = {form["form_id"]: form for form in forms
                  if form["factor"] == "base"}
    for cell_id, new_n in (("C2", 256), ("C3", 128)):
        base = base_forms[f"{cell_id}.base.base"]
        control = dict(base)
        control.update({
            "factor": "probe_control",
            "level": f"idesc_N={new_n}",
            "form_id": f"POSITIVE.{cell_id}.idesc_N{new_n}",
            "run_status": "control",
            "skip_reason": [],
            "N": new_n,
            "factor_idesc": (
                (base["factor_idesc"] & ~(0x3F << 17)) |
                ((new_n >> 3) << 17)
            ),
        })
        control["view_bytes"] = dict(base["view_bytes"])
        control["view_bytes"]["b"] = descriptor_view_span(
            new_n // control["cta_group"], control["K"],
            control["b_bits"], control["swizzle"], control["transpose"][1],
            control["lbo_sbo"]["b_lbo"] or 0,
            control["lbo_sbo"]["b_sbo"] or 0)
        control["required_smem_bytes"] = (
            FACTOR_STAGES *
            (max(A_STAGE_BYTES, control["start_a"] +
                 control["view_bytes"]["a"]) +
             max(B_STAGE_BYTES, control["start_b"] +
                 control["view_bytes"]["b"])) +
            B_ZERO_BYTES + 64 + 1024
        )
        control["required_stage_bytes"] = {
            "a": max(A_STAGE_BYTES, control["start_a"] +
                     control["view_bytes"]["a"]),
            "b": max(B_STAGE_BYTES, control["start_b"] +
                     control["view_bytes"]["b"]),
        }
        control["base_leaf"] = base["base_leaf"]
        forms.append(control)
    zero_control = dict(base_forms["C2.base.base"])
    zero_control.update({
        "factor": "probe_control", "level": "B_start_zero_region",
        "form_id": "POSITIVE.C2.B_start_zero_region",
        "run_status": "control", "skip_reason": [], "b_zero": True,
        "expected_d": {
            "formula": "B zero region -> D=0", "sign": 0,
            "K": zero_control["K"], "d_type": zero_control["d_type"],
        },
    })
    forms.append(zero_control)
    return forms


def cell_transposes_for_start(cell_id, cell):
    if cell["a_src"] == "tmem":
        return [("K", "K"), ("K", "MN")]
    return [("K", "K"), ("MN", "K"), ("K", "MN"), ("MN", "MN")]


def select_161(forms, *, block_c8_sf_id=True):
    ab_types = {"BF16xBF16", "E5M2xE5M2", "E2M1xE2M1"}
    transpose_levels = {"MN,K", "K,MN", "MN,MN"}
    swizzles = {"NONE", "SW64"}
    starts = {32, 64, 96, 256, 512}
    lbo_values = {0, 1024, 8192, 16384}
    sbo_values = {0, 128, 512}
    selected = []
    for form in forms:
        factor = form["factor"]
        include = factor in {"base", "probe_control"}
        if factor == "ab_type":
            include = form["level"] in ab_types
        elif factor == "transpose":
            include = form["level"] in transpose_levels
        elif factor == "swizzle":
            include = (form["swizzle"] in swizzles and
                       form["level"].split(":", 1)[1] in
                       {"K,K", "MN,MN", "K,MN"})
        elif factor == "start":
            offset = int(form["level"].rsplit("@", 1)[1])
            include = (form["swizzle"] in {"SW128", "SW64", "NONE"} and
                       offset in starts)
        elif factor == "lbo_sbo":
            match = re.search(r"\.(LBO|SBO)=(\d+)$", form["level"])
            include = (form["swizzle"] in {"SW128", "SW64", "NONE"} and
                       match is not None and int(match.group(2)) in
                       (lbo_values if match and match.group(1) == "LBO"
                        else sbo_values))
        elif factor == "sf_id":
            if form["kind"] == "mxf8f6f4":
                include = form["level"] in {"1/1", "2/2"}
            elif form["kind"] == "mxf4nvf4":
                include = form["level"] in {"0/2", "2/2"}
                if include and block_c8_sf_id:
                    form = dict(form)
                    form["run_status"] = "unmeasurable"
                    form["skip_reason"] = [
                        "cuda-gdb observed Warp Illegal Instruction Parameter "
                        "at block16 MMA for nonzero mxf4nvf4 SF ID; 45-minute "
                        "diagnosis limit reached"
                    ]
        elif factor == "d_type":
            include = form["level"] == "F16"
        elif factor == "negate":
            include = form["level"] == "A"
        elif factor == "operation_ws":
            include = form["level"] == "max_shift_0"
        if include:
            selected.append(form)
    return selected


def emit_header(forms, out_path, smoke=False):
    chosen = [f for f in forms if f["run_status"] in {"candidate", "control"}]
    if smoke:
        by_factor = {}
        for f in chosen:
            if f["factor"] not in {"base", "probe_control"}:
                by_factor.setdefault((f["cell_id"], f["factor"]), f)
        base = [f for f in chosen if f["factor"] == "base"]
        controls = [f for f in chosen if f["factor"] == "probe_control"]
        chosen = base + list(by_factor.values()) + controls
    lines = [
        "#pragma once",
        "// generated by UMMA factor form generator",
        "static const FactorForm kFactorForms[] = {",
    ]
    for i, f in enumerate(chosen):
        vals = [
            f["form_id"], f["cell_id"], f["factor"], f["level"],
            f["kind"], f["a_type"], f["b_type"], f["d_type"],
            f["scale_type"], f["operation"], f["swizzle"],
            f["transpose"][0], f["transpose"][1],
        ]
        s = ", ".join(json.dumps(v) for v in vals)
        nums = [
            KIND_ID[f["kind"]], f["cta_group"], int(f["a_src"] == "tmem"),
            f["M"], f["N"], f["K"], f["factor_idesc"],
            f["smem_a"] or 0, f["smem_b"], f["start_a"], f["start_b"],
            f["a_bits"], f["b_bits"], f["a_one"], f["b_one"],
            f["scale_bits"], f["scale_one"], f["sf_id_a"], f["sf_id_b"],
            int(f["d_f16"]), f["sign"], int(f["reuse"]), int(f["ws"]),
            f["max_shift"], int(f["lbo_abs_a"]), int(f["lbo_abs_b"]),
            int(f["b_zero"]), int(f["sparse"]), f["sparsity_version"],
            {"none": 0, "partial": 1, "same": 2}[f["overlap"]],
            f["overlap_bytes"], f.get("acc", 1), f.get("arot", 0),
            f.get("brot", 0), f.get("grp", 0),
            f.get("issuers", 1), f.get("kalu", 0), f.get("elected", 0),
        ]
        lines.append(
            "  {%s, %s},  // %s" %
            (s, ", ".join(str(n) for n in nums), f["form_id"]))
    lines += ["};", ""]
    Path(out_path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="tools/microbench/umma_factor_matrix.json")
    ap.add_argument("--out", default="tools/microbench/umma_factor_forms.json")
    ap.add_argument("--header", default="tools/microbench/probes/umma_factor_forms.hpp")
    ap.add_argument("--profile", default="configs/runtime_profiles/b300s_cluster_p0.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--c450-161", action="store_true")
    ap.add_argument("--measure-mxf4nvf4-sf-id", action="store_true")
    ap.add_argument("--cell", action="append")
    ap.add_argument("--factor", action="append")
    ap.add_argument("--form-id", action="append")
    args = ap.parse_args()
    matrix = json.load(open(args.matrix))
    forms = expand(matrix, args.profile)
    if args.c450_161:
        forms = select_161(
            forms, block_c8_sf_id=not args.measure_mxf4nvf4_sf_id)
    if args.cell:
        selected_cells = set(args.cell)
        forms = [form for form in forms if form["cell_id"] in selected_cells]
    if args.factor:
        selected_factors = set(args.factor)
        forms = [form for form in forms if form["factor"] in selected_factors]
    if args.form_id:
        selected_ids = set(args.form_id)
        forms = [form for form in forms if form["form_id"] in selected_ids]
    result = {
        "schema": ("mechtree.umma_factor_forms.161.v1" if args.c450_161
                   else "mechtree.umma_factor_forms.v1"),
        "matrix": args.matrix,
        "functional_check": matrix["functional_check"],
        "measurement": matrix["measurement"],
        "rules": "tools/microbench/umma_factor_rules.py",
        "descriptor_source": (
            "CUTLASS cute/arch/mma_sm100_desc.hpp: InstrDescriptor, "
            "InstrDescriptorBlockScaled, SmemDescriptor, make_umma_desc; "
            "PTX ISA 9.7.18.4.1/.2"),
        "smem_allocation": {
            "stages": FACTOR_STAGES,
            "stage_a_bytes": A_STAGE_BYTES,
            "stage_b_bytes": B_STAGE_BYTES,
            "b_zero_region_bytes": B_ZERO_BYTES,
            "maximum_cta_bytes": MAX_CTA_SMEM_BYTES,
        },
        "census_supplemental_levels": {"sbo_bytes": [0]},
        "scope": ("c450 corpus subset for umma-factors-161"
                  if args.c450_161 else
                  "selected factor forms"
                  if args.cell or args.factor or args.form_id else
                  "full factor matrix"),
        "selected_cells": sorted(set(args.cell)) if args.cell else None,
        "selected_factors": sorted(set(args.factor)) if args.factor else None,
        "selected_form_ids": sorted(set(args.form_id)) if args.form_id else None,
        "forms": forms,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    emit_header(forms, args.header, args.smoke)
    print("forms", len(forms),
          "candidate", sum(f["run_status"] == "candidate" for f in forms),
          "unmeasurable", sum(f["run_status"] == "unmeasurable" for f in forms),
          "controls", sum(f["run_status"] == "control" for f in forms))


if __name__ == "__main__":
    main()
