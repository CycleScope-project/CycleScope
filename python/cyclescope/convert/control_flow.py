"""Control-flow facts for one warp's stream (spec §4.2 key spellings).

Backward BRA -> loop facts (same_body_reuse over per-(cta,loop) ordinal,
loop_body_bytes, loop_instruction_count, backward_loop_class_s,
control_participating_warps filled in a second pass per CTA).
BSSY/BSYNC -> reconvergence region facts (sequential_region_count,
control_selector_present, control_path_count).
"""
from __future__ import annotations

import re
from collections import defaultdict

from cyclescope.convert.sass_decode import (PURE_INTEGER_REGISTER_CONTROL_ROOTS,
                                      numeric_control_target)

_MAX_PARAMETERIZED_REGIONS = 2


def _btoken(row) -> str | None:
    for token in row.inst.operands:
        name = token.strip().lstrip("+-").split(".", 1)[0].upper()
        if re.fullmatch(r"B\d+", name):
            return name
    return None


def bssy_close_pairs(rows_by_pc) -> dict:
    pairs: dict[int, int] = {}
    stack: list = []
    for pc in sorted(rows_by_pc):
        row = rows_by_pc[pc]
        root = row.inst.opcode.split(".", 1)[0].upper()
        if root == "BSSY":
            stack.append((pc, _btoken(row)))
        elif root == "BSYNC":
            token = _btoken(row)
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][1] == token:
                    pairs[stack[i][0]] = pc
                    del stack[i]
                    break
    return pairs


def flow_patches(records, static_of, static_by_pc_of, cta_key) -> dict:
    """{record position: uniform-dict patch} for one warp.

    static_of(rec) -> StaticRow; static_by_pc_of(function_id) -> {pc:
    StaticRow} for the whole function.  Loop ordinals are per (cta,
    function, source, target); participating warp counts are patched by
    the caller (``fill_participating``) after all warps are seen.
    """
    patch: dict[int, dict] = {}
    loop_ordinals: dict = defaultdict(int)
    opened: list = []
    completed: list = []
    lane_stacks: list[list] = [[] for _ in range(32)]
    next_region = 0
    pairs_cache: dict = {}

    for pos, rec in enumerate(records):
        row = static_of(rec)
        root = row.inst.opcode.split(".", 1)[0].upper()
        if root not in {"BRA", "BSSY", "BSYNC", "BREAK", "EXIT"}:
            continue
        mask = rec.exec_mask
        if root == "BRA":
            target = row.branch_target
            if target is None or target >= row.inst.pc:
                continue
            span_pcs = [pc for pc in static_by_pc_of(rec.function_id)
                        if target <= pc <= row.inst.pc]
            counts = defaultdict(int)
            for pc in span_pcs:
                counts[static_by_pc_of(rec.function_id)[pc].inst.opcode
                       .split(".", 1)[0].upper()] += 1
            disallowed = sorted(
                set(counts) - PURE_INTEGER_REGISTER_CONTROL_ROOTS)
            key = (cta_key, rec.function_id, row.inst.pc, target)
            loop_ordinals[key] += 1
            ordinal = loop_ordinals[key]
            patch[pos] = {
                "same_body_reuse": 1 if ordinal > 1 else 0,
                "loop_body_bytes": row.inst.pc - target + 16,
                "loop_instruction_count": len(span_pcs),
                "backward_loop_class_s": ("pure_integer_register_control"
                                          if not disallowed
                                          else "mixed_or_stateful"),
                "_participants_key": key,
                "_loop_key": key,
            }
            continue
        if root == "EXIT":
            if not mask:
                continue
            for lane in range(32):
                if mask & (1 << lane):
                    lane_stacks[lane].clear()
            continue
        token = _btoken(row)
        if token is None:
            continue
        if root == "BSSY":
            if rec.function_id not in pairs_cache:
                pairs_cache[rec.function_id] = bssy_close_pairs(
                    static_by_pc_of(rec.function_id))
            close_pc = pairs_cache[rec.function_id].get(row.inst.pc)
            if not mask:
                continue
            region = {"pos": pos, "token": token,
                      "close_pc": close_pc, "open_mask": mask,
                      "resolved": 0, "path_masks": []}
            for lane in range(32):
                if mask & (1 << lane):
                    lane_stacks[lane].append(region)
            opened.append(region)
            next_region += 1
            continue
        # BSYNC / BREAK resolves one path of the innermost matching region
        if not mask:
            continue
        matched: dict = {}
        ok = True
        for lane in range(32):
            if not mask & (1 << lane):
                continue
            hits = [i for i, r in enumerate(lane_stacks[lane])
                    if r["token"] == token]
            if len(hits) != 1:
                ok = False
                break
            matched[lane] = hits[0]
        if not ok:
            continue
        regions = {id(lane_stacks[l][matched[l]]): lane_stacks[l][matched[l]]
                   for l in matched}
        if len(regions) != 1:
            continue
        region = next(iter(regions.values()))
        region["resolved"] |= mask
        if root == "BSYNC":
            region["path_masks"].append(int(mask))
        for lane in matched:
            del lane_stacks[lane][matched[lane]]
        if region["resolved"] == region["open_mask"]:
            path_lanes = [m.bit_count() for m in region["path_masks"]]
            if len(region["path_masks"]) >= 2 \
                    and len(set(path_lanes)) == 1:
                completed.append({
                    "open_pos": region["pos"], "close_pos": pos,
                    "lanes": path_lanes[0],
                    "path_count": len(region["path_masks"])})
            else:
                patch.setdefault(region["pos"], {})[
                    "control_path_count"] = len(region["path_masks"]) or 1
    # compound selector: adjacent completed regions with equal lane counts
    runs: list[list] = []
    for region in completed:
        if runs and region["open_pos"] == runs[-1][-1]["close_pos"] + 1 \
                and region["lanes"] == runs[-1][-1]["lanes"]:
            runs[-1].append(region)
        else:
            runs.append([region])
    for run in runs:
        for off in range(0, len(run), _MAX_PARAMETERIZED_REGIONS):
            sel = run[off:off + _MAX_PARAMETERIZED_REGIONS]
            for i, region in enumerate(sel):
                patch.setdefault(region["open_pos"], {}).update({
                    "control_selector_present": 1,
                    "sequential_region_count": len(sel),
                    "control_path_count": region["path_count"]})
    return patch


def fill_participating(patches_by_warp: dict) -> None:
    """After all warps of a CTA are analyzed: control_participating_warps
    = number of warps that executed each loop key."""
    participants: dict = defaultdict(set)
    for warp_id, patch in patches_by_warp.items():
        for entry in patch.values():
            key = entry.pop("_participants_key", None)
            if key is not None:
                participants[key].add(warp_id)
    for patch in patches_by_warp.values():
        for entry in patch.values():
            key = entry.pop("_loop_key", None)
            if key is not None:
                entry["control_participating_warps"] = len(participants[key])
