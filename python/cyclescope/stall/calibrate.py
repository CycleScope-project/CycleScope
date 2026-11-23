"""Calibration table of the reason -> class mapping from probe rows
(tools/microbench/stall_class_mapping/protocol.json).

Input: two `cyclescope stall compare` outputs of the same rows (the
hardware pass is run twice).  A row is usable when it passes the
protocol's controls; the decision per mapping rule is A (calibrated to
the top class), B (split) or C (not exercised).
"""
from __future__ import annotations

from pathlib import Path

from .class_mapping import ClassMapping
from .summarize import hardware_total, load_results

EXECUTION_COUNT_ERROR_MAX = 0.05
WARPS_RATIO_BAND = 0.02
REFERENCE_CONSISTENCY_MAX = 0.10
SAMPLES_MIN = 2000
CYCLE_RATIO_BAND = (0.85, 1.15)
REPEAT_SHARE_POINTS = 0.03
TOP_SHARE_MIN = 0.70
POOLED_OVERLAP_MIN = 0.01
ROWS_MIN = 2


def hardware_shares(result: dict) -> dict[str, float]:
    total = hardware_total(result)
    out: dict[str, float] = {}
    for rows in result["table"].values():
        for cls, (hw, _) in rows.items():
            out[cls] = out.get(cls, 0.0) + hw / total
    return out


def row_status(first: dict, second: dict | None) -> str:
    """"usable" or the first failed control."""
    if first.get("status") != "ok":
        return f"failed:{first.get('status')}"
    alignment, hw = first["alignment"], first["hardware"]
    error = alignment["execution_count_error"]
    if alignment["mnemonic_mismatches"] or error is None or \
            error > EXECUTION_COUNT_ERROR_MAX or \
            abs(alignment["warps_ratio"] - 1) > WARPS_RATIO_BAND:
        return "same_launch"
    if hw["reference_consistency"] > REFERENCE_CONSISTENCY_MAX or \
            hw["samples"] < SAMPLES_MIN:
        return "thin"
    ratio = first["metrics"]["cycle_ratio"]
    if not CYCLE_RATIO_BAND[0] <= ratio <= CYCLE_RATIO_BAND[1]:
        return "timing"
    if second is None or second.get("status") != "ok":
        return "no_repeat"
    a, b = hardware_shares(first), hardware_shares(second)
    if any(abs(a.get(c, 0.0) - b.get(c, 0.0)) > REPEAT_SHARE_POINTS
           for c in set(a) | set(b)):
        return "unstable"
    return "usable"


def calibration_table(first: list[dict], second: list[dict],
                      mapping: ClassMapping) -> dict:
    repeats = {r["case"]: r for r in second}
    statuses = {r["case"]: row_status(r, repeats.get(r["case"]))
                for r in first}
    usable = [r for r in first if statuses[r["case"]] == "usable"]
    pooled: dict[int, dict[str, float]] = {}
    reasons: dict[int, dict[str, float]] = {}
    rows_reached: dict[int, int] = {}
    for result in usable:
        total = hardware_total(result)
        per_rule: dict[int, float] = {}
        for row in result.get("cooccurrence", []):
            index = mapping.rule_index(row["reason"])
            if index is None:
                continue
            share = row["cycles"] / total
            by_class = pooled.setdefault(index, {})
            by_class[row["class"]] = by_class.get(row["class"], 0.0) + share
            by_reason = reasons.setdefault(index, {})
            by_reason[row["reason"]] = by_reason.get(row["reason"], 0.0) + \
                share
            per_rule[index] = per_rule.get(index, 0.0) + share
        for index, share in per_rule.items():
            if share >= POOLED_OVERLAP_MIN:
                rows_reached[index] = rows_reached.get(index, 0) + 1
    rules = []
    for index, document in enumerate(mapping.documents):
        by_class = pooled.get(index, {})
        overlap = sum(by_class.values())
        mean_overlap = overlap / len(usable) if usable else 0.0
        shares = {c: v / overlap for c, v in
                  sorted(by_class.items(), key=lambda kv: -kv[1])} \
            if overlap > 0 else {}
        top = next(iter(shares), None)
        if mean_overlap < POOLED_OVERLAP_MIN or \
                rows_reached.get(index, 0) < ROWS_MIN:
            decision = "C"
        elif shares[top] >= TOP_SHARE_MIN:
            decision = "A"
        else:
            decision = "B"
        rules.append({
            "rule": document["reason"], "class": document["class"],
            "evidence": document["evidence"], "decision": decision,
            "calibrated_class": top if decision == "A" else None,
            "mean_overlap": mean_overlap,
            "rows": rows_reached.get(index, 0), "shares": shares,
            "reasons": dict(sorted(reasons.get(index, {}).items(),
                                   key=lambda kv: -kv[1])[:8]),
        })
    return {"rows": statuses, "usable": len(usable), "rules": rules}


def run_calibrate(first_root: Path, second_root: Path,
                  mapping: ClassMapping) -> dict:
    return calibration_table(load_results(Path(first_root) / "cases"),
                             load_results(Path(second_root) / "cases"),
                             mapping)
