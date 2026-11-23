"""Suite-level view of per-case stall comparisons (compare.py output)."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

CASE_COLUMNS = [
    "case", "family", "status", "cycle_ratio", "attribution_error",
    "attribution_error_loop", "attribution_error_outside",
    "residency_ratio", "residency_ratio_loop", "residency_ratio_outside",
    "hardware_loop_share", "shape_distance", "shape_distance_loop",
    "shape_distance_outside", "pc_residency_error",
    "block_residency_error", "site_residency_error",
    "class_error", "kind_class_error", "site_class_error",
    "block_class_error", "pc_class_error", "kind_residency_error",
    "reference_consistency", "selected_scale_ratio",
    "execution_count_error", "warps_ratio", "mnemonic_mismatches",
    "unplaced_share", "hardware_unissued_share",
    "hardware_unissued_unplaced_share", "samples",
]


def family_of(case: str) -> str:
    """Suite family: the token before the first `_` (hopper names) or
    `-` (blackwell names)."""
    for sep in ("_", "-"):
        if sep in case:
            head = case.split(sep, 1)[0]
            if sep == "_" and "-" in head:
                continue
            return head
    return case


def percentile_90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.9 * (len(ordered) - 1))]


def hardware_total(result: dict) -> float:
    """Hardware cycles per warp over all regions and classes (the
    denominator of attribution_error; includes `drain`)."""
    return sum(hw for rows in result["table"].values()
               for hw, _ in rows.values())


def case_row(result: dict) -> dict:
    row = {"case": result["case"], "family": family_of(result["case"]),
           "status": result.get("status", "")}
    if result.get("status") != "ok":
        return row
    row.update(result["metrics"])
    hw, sim, al = result["hardware"], result["engine"], result["alignment"]
    row.update({
        "reference_consistency": hw["reference_consistency"],
        "selected_scale_ratio": hw["selected_scale_ratio"],
        "samples": hw["samples"],
        "execution_count_error": al["execution_count_error"],
        "warps_ratio": al["warps_ratio"],
        "mnemonic_mismatches": al["mnemonic_mismatches"],
        "hardware_unissued_share": al.get("hardware_unissued_share"),
        "hardware_unissued_unplaced_share":
            al.get("hardware_unissued_unplaced_share"),
        "unplaced_share": sim["unplaced_per_warp"] / hardware_total(result),
    })
    return row


def class_gaps(results: list[dict]) -> list[dict]:
    """Per (region, class): signed and absolute mean of
    (sim - hw) / hardware residency over the ok cases, and the number
    of cases where the engine is above / below hardware by more than
    1 % of the residency."""
    acc: dict[tuple[str, str], list[float]] = {}
    count = 0
    for result in results:
        if result.get("status") != "ok":
            continue
        count += 1
        total = hardware_total(result)
        for region, rows in result["table"].items():
            for cls, (hw, sim) in rows.items():
                acc.setdefault((region, cls), []).append((sim - hw) / total)
    out = []
    for (region, cls), values in acc.items():
        out.append({
            "region": region, "class": cls,
            "mean_signed": sum(values) / count,
            "mean_absolute": sum(abs(v) for v in values) / count,
            "cases_above": sum(v > 0.01 for v in values),
            "cases_below": sum(v < -0.01 for v in values),
        })
    out.sort(key=lambda r: -r["mean_absolute"])
    return out


def site_gaps(results: list[dict]) -> list[dict]:
    """Per (region, site kind), without the class mapping: mean over
    the ok cases of the hardware and the engine residency and of
    (sim - hw), all / hardware residency of the case; the signed mean
    is also split into issue cycles and wait cycles."""
    acc: dict[tuple[str, str], list[list[float]]] = {}
    count = 0
    for result in results:
        if result.get("status") != "ok" or "sites" not in result:
            continue
        count += 1
        total = hardware_total(result)
        for s in result["sites"]:
            acc.setdefault((s["region"], s["kind"]), []).append(
                [s["hw_issue"] / total, s["hw_wait"] / total,
                 s["sim_issue"] / total, s["sim_wait"] / total])
    out = []
    for (region, kind), rows in acc.items():
        diffs = [r[2] + r[3] - r[0] - r[1] for r in rows]
        out.append({
            "region": region, "kind": kind, "cases": len(rows),
            "hw": sum(r[0] + r[1] for r in rows) / count,
            "sim": sum(r[2] + r[3] for r in rows) / count,
            "mean_signed": sum(diffs) / count,
            "mean_absolute": sum(abs(d) for d in diffs) / count,
            "signed_issue": sum(r[2] - r[0] for r in rows) / count,
            "signed_wait": sum(r[3] - r[1] for r in rows) / count,
            "cases_above": sum(d > 0.01 for d in diffs),
            "cases_below": sum(d < -0.01 for d in diffs),
        })
    out.sort(key=lambda r: -r["mean_absolute"])
    return out


def site_parts(results: list[dict]) -> list[dict]:
    """Per (region, site kind): the hardware classes and the engine
    reasons its wait cycles are made of, mean over the ok cases of
    cycles / hardware residency of the case (parts below 0.05 % of a
    case's residency are not in compare.py's rows)."""
    acc: dict[tuple[str, str, str, str], float] = {}
    count = 0
    for result in results:
        if result.get("status") != "ok" or "sites" not in result:
            continue
        count += 1
        total = hardware_total(result)
        for s in result["sites"]:
            for side, parts in (("hw", s.get("hw_classes", {})),
                                ("sim", s.get("sim_reasons", {}))):
                for name, v in parts.items():
                    key = (s["region"], s["kind"], side, name)
                    acc[key] = acc.get(key, 0.0) + v / total
    out = [{"region": r, "kind": k, "side": side, "name": name,
            "mean": v / count} for (r, k, side, name), v in acc.items()]
    out.sort(key=lambda row: (row["region"], row["kind"], row["side"],
                              -row["mean"]))
    return out


def kind_gaps(results: list[dict]) -> list[dict]:
    """Per (instruction kind of the pc a warp is parked on, class):
    mean over the ok cases of the hardware and the engine cycles and
    of (sim - hw), all / hardware residency of the case."""
    acc: dict[tuple[str, str], list[float]] = {}
    count = 0
    for result in results:
        if result.get("status") != "ok" or "kinds" not in result:
            continue
        count += 1
        total = hardware_total(result)
        for row in result["kinds"]:
            cell = acc.setdefault((row["kind"], row["class"]),
                                  [0.0, 0.0, 0.0])
            cell[0] += row["hw"] / total
            cell[1] += row["sim"] / total
            cell[2] += abs(row["sim"] - row["hw"]) / total
    out = [{"kind": kind, "class": cls, "hw": v[0] / count,
            "sim": v[1] / count, "mean_signed": (v[1] - v[0]) / count,
            "mean_absolute": v[2] / count}
           for (kind, cls), v in acc.items()] if count else []
    out.sort(key=lambda r: -r["mean_absolute"])
    return out


def site_pc_errors(results: list[dict]) -> list[dict]:
    """Per (region, site kind): mean over the ok cases of the per-pc
    and per-(pc, class) residency error inside the site, / hardware
    residency of the case.  Against site_gaps' mean_absolute (the
    error of the site's total) this separates `the time is in the
    wrong place inside the site` from `the site has the wrong time`."""
    acc: dict[tuple[str, str], list[float]] = {}
    count = 0
    for result in results:
        if result.get("status") != "ok" or "site_pc" not in result:
            continue
        count += 1
        total = hardware_total(result)
        for row in result["site_pc"]:
            cell = acc.setdefault((row["region"], row["kind"]), [0.0, 0.0])
            cell[0] += row["pc_error"] / total
            cell[1] += row["pc_class_error"] / total
    out = [{"region": r, "kind": kind, "pc_error": v[0] / count,
            "pc_class_error": v[1] / count}
           for (r, kind), v in acc.items()] if count else []
    out.sort(key=lambda r: -r["pc_class_error"])
    return out


def floor_levels(results: list[dict]) -> dict | None:
    """Mean over the ok cases that carry a repeat sampling of the
    hardware-against-hardware distance at every level, next to the
    engine's errors over the same cases."""
    rows = [r for r in results
            if r.get("status") == "ok" and r.get("floor")]
    if not rows:
        return None
    names = {"class": "class_error", "region_class": "attribution_error",
             "kind_class": "kind_class_error",
             "site_class": "site_class_error",
             "block_class": "block_class_error",
             "pc_class": "pc_class_error", "kind": "kind_residency_error",
             "site": "site_residency_error",
             "block": "block_residency_error", "pc": "pc_residency_error"}
    out = {"cases": len(rows), "levels": {}}
    for level, metric in names.items():
        out["levels"][level] = {
            "floor": sum(r["floor"][level] for r in rows) / len(rows),
            "engine": sum(r["metrics"][metric] for r in rows) / len(rows)}
    return out


def reason_classes(results: list[dict]) -> list[dict]:
    """Per engine reason: the hardware classes its waits overlap with
    (compare.py `cooccurrence`), every ok case weighted by its own
    hardware residency; shares are of the reason's overlapping cycles."""
    acc: dict[str, dict[str, float]] = {}
    cases: dict[str, set] = {}
    count = 0
    for result in results:
        if result.get("status") != "ok":
            continue
        count += 1
        total = hardware_total(result)
        for row in result.get("cooccurrence", []):
            by_class = acc.setdefault(row["reason"], {})
            by_class[row["class"]] = by_class.get(row["class"], 0.0) + \
                row["cycles"] / total
            cases.setdefault(row["reason"], set()).add(result["case"])
    out = []
    for reason, by_class in acc.items():
        overlap = sum(by_class.values())
        out.append({
            "reason": reason, "cases": len(cases[reason]),
            "mean_overlap": overlap / count,
            "classes": {c: v / overlap for c, v in
                        sorted(by_class.items(), key=lambda kv: -kv[1])},
        })
    out.sort(key=lambda r: -r["mean_overlap"])
    return out


def aggregate(rows: list[dict], key: str) -> dict | None:
    values = [r[key] for r in rows
              if r.get("status") == "ok" and r.get(key) is not None
              and r[key] == r[key]]
    if not values:
        return None
    return {"n": len(values), "mean": sum(values) / len(values),
            "median": statistics.median(values),
            "p90": percentile_90(values), "max": max(values)}


def load_results(directory: Path) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(Path(directory).glob("*.json"))]


def _format(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary(results: list[dict], out: Path, label: str) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [case_row(r) for r in results]
    with (out / "cases.tsv").open("w") as fh:
        fh.write("\t".join(CASE_COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(_format(row.get(c))
                               for c in CASE_COLUMNS) + "\n")
    gaps = class_gaps(results)
    with (out / "class_gaps.tsv").open("w") as fh:
        fh.write("region\tclass\tmean_signed\tmean_absolute\t"
                 "cases_above\tcases_below\n")
        for g in gaps:
            fh.write(f"{g['region']}\t{g['class']}\t{g['mean_signed']:.5f}"
                     f"\t{g['mean_absolute']:.5f}\t{g['cases_above']}"
                     f"\t{g['cases_below']}\n")
    with (out / "reason_classes.tsv").open("w") as fh:
        fh.write("reason\tcases\tmean_overlap\tclasses\n")
        for r in reason_classes(results):
            shares = " ".join(f"{c}={v:.3f}" for c, v in
                              list(r["classes"].items())[:5])
            fh.write(f"{r['reason']}\t{r['cases']}\t"
                     f"{r['mean_overlap']:.5f}\t{shares}\n")
    site_rows = site_gaps(results)
    with (out / "site_gaps.tsv").open("w") as fh:
        columns = ["region", "kind", "cases", "hw", "sim", "mean_signed",
                   "mean_absolute", "signed_issue", "signed_wait",
                   "cases_above", "cases_below"]
        fh.write("\t".join(columns) + "\n")
        for g in site_rows:
            fh.write("\t".join(
                f"{g[c]:.5f}" if isinstance(g[c], float) else str(g[c])
                for c in columns) + "\n")
    with (out / "site_parts.tsv").open("w") as fh:
        fh.write("region\tkind\tside\tname\tmean\n")
        for row in site_parts(results):
            fh.write(f"{row['region']}\t{row['kind']}\t{row['side']}\t"
                     f"{row['name']}\t{row['mean']:.5f}\n")
    kind_rows = kind_gaps(results)
    with (out / "kind_gaps.tsv").open("w") as fh:
        fh.write("kind\tclass\thw\tsim\tmean_signed\tmean_absolute\n")
        for g in kind_rows:
            if g["mean_absolute"] < 5e-6:
                continue
            fh.write(f"{g['kind']}\t{g['class']}\t{g['hw']:.5f}\t"
                     f"{g['sim']:.5f}\t{g['mean_signed']:.5f}\t"
                     f"{g['mean_absolute']:.5f}\n")
    site_pc_rows = site_pc_errors(results)
    with (out / "site_pc.tsv").open("w") as fh:
        fh.write("region\tkind\tpc_error\tpc_class_error\n")
        for g in site_pc_rows:
            fh.write(f"{g['region']}\t{g['kind']}\t{g['pc_error']:.5f}\t"
                     f"{g['pc_class_error']:.5f}\n")
    keys = ["attribution_error", "attribution_error_loop",
            "attribution_error_outside", "shape_distance",
            "pc_residency_error", "block_residency_error",
            "site_residency_error", "class_error", "kind_class_error",
            "site_class_error", "block_class_error", "pc_class_error",
            "kind_residency_error",
            "reference_consistency"]
    ok = [r for r in rows if r["status"] == "ok"]
    statuses: dict[str, int] = {}
    for r in rows:
        key = r["status"].split(" (")[0]
        statuses[key] = statuses.get(key, 0) + 1
    summary = {
        "label": label, "cases": len(rows), "statuses": statuses,
        "overall": {k: aggregate(ok, k) for k in keys},
        "cycle_error": aggregate(
            [dict(r, cycle_error=abs(r["cycle_ratio"] - 1)) for r in ok],
            "cycle_error"),
        "families": {},
        "class_gaps": gaps,
        "site_gaps": site_rows,
        "site_pc": site_pc_rows,
        "floor": floor_levels(results),
    }
    for name in sorted({r["family"] for r in ok}):
        members = [r for r in ok if r["family"] == name]
        summary["families"][name] = {k: aggregate(members, k) for k in keys}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
