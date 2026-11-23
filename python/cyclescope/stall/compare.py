"""One case: hardware stall samples against the engine stall ledger.

Everything is in cycles per warp.  Hardware: per-pc class samples
scaled per class by HardwareStalls.class_scales().  Engine: ledger warp
cycles / ledger warps, reasons mapped to classes.  Regions: `loop`,
`outside`, plus `unplaced` for engine cycles whose code address is not
inside the kernel function (hardware has nothing there).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .class_mapping import ClassMapping
from .hardware import HardwareStalls
from .ledger import StallLedger
from .regions import (basic_blocks, find_loops, instruction_kind,
                      mnemonic_root, place_unissued, poll_blocks,
                      region_of, sites)

INSTRUCTION_BYTES = 16
REGIONS = ("loop", "outside", "unplaced")


@dataclass
class KernelFunction:
    function_id: int
    name: str
    address: int
    instruction_count: int
    sass: dict[int, str]            # offset -> SASS text

    def offset(self, code_address: int) -> int | None:
        delta = code_address - self.address
        if 0 <= delta < self.instruction_count * INSTRUCTION_BYTES:
            return delta
        return None


def load_kernel_function(capture_dir: Path,
                         ledger: StallLedger) -> KernelFunction:
    """The capture function that holds the ledger's code addresses (the
    one with the most issues inside it)."""
    capture_dir = Path(capture_dir)
    functions = [json.loads(line) for line in
                 (capture_dir / "functions.jsonl").read_text().splitlines()
                 if line.strip()]
    best, best_events = None, 0
    for f in functions:
        if not f.get("addr"):
            continue
        base = int(f["addr"], 16)
        end = base + int(f["instructions"]) * INSTRUCTION_BYTES
        events = sum(entry[1] for (address, reason), entry
                     in ledger.residency.items()
                     if reason == "issued" and base <= address < end)
        if events > best_events:
            best, best_events = f, events
    if best is None:
        raise ValueError(
            f"missing_fact:kernel_function ({capture_dir}/functions.jsonl)")
    sass: dict[int, str] = {}
    table = capture_dir / "functions" / f"{best['function_id']}.tsv"
    for line in table.read_text().splitlines():
        cells = line.split("\t")
        if len(cells) >= 4:
            sass[int(cells[1], 16)] = cells[3].strip().rstrip(";").strip()
    return KernelFunction(function_id=int(best["function_id"]),
                          name=best.get("name") or "",
                          address=int(best["addr"], 16),
                          instruction_count=int(best["instructions"]),
                          sass=sass)


def _total_variation(a: dict[str, float], b: dict[str, float]) -> float | None:
    sa, sb = sum(a.values()), sum(b.values())
    if sa <= 0 or sb <= 0:
        return None
    return 0.5 * sum(abs(a.get(c, 0.0) / sa - b.get(c, 0.0) / sb)
                     for c in set(a) | set(b))

LEVELS = ("class", "region_class", "kind_class", "site_class",
          "block_class", "pc_class", "kind", "site", "block", "pc")


def level_errors(first: dict, second: dict, *, region, blocks: dict,
                 site_of: dict, kind_of, denominator: float) -> dict:
    """L1 distances between two per-pc class tables at several
    aggregation levels, each / `denominator`.

    `first`, `second`: offset (None = outside the function) ->
    {class: cycles per warp}.  Levels: the class alone; (region, class)
    = attribution_error; (instruction kind of the pc, class); (region,
    site kind, class); (straight-line run, class); (pc, class); and the
    same four placements without the class.  Returns {"errors": level
    -> distance, "kinds": (kind, class) -> [first, second], "site_pc":
    (region, site kind) -> [sum over its pcs of |second - first|, the
    same per (pc, class)]}."""
    acc = {level: {} for level in LEVELS}
    kinds: dict[tuple[str, str], list[float]] = {}
    for side, table in enumerate((first, second)):
        sign = 1.0 if side else -1.0
        for offset, row in table.items():
            if offset is None:
                reg, block, site, kind = "unplaced", None, \
                    ("unplaced", "-"), "<unplaced>"
            else:
                reg = region(offset)
                block = blocks.get(offset, offset)
                site = (reg, site_of.get(offset, (offset, "compute"))[1])
                kind = kind_of(offset)
            for cls, value in row.items():
                v = sign * value
                for level, key in (
                        ("class", cls), ("region_class", (reg, cls)),
                        ("kind_class", (kind, cls)),
                        ("site_class", (site, cls)),
                        ("block_class", (block, cls)),
                        ("pc_class", (offset, cls)), ("kind", kind),
                        ("site", site), ("block", block), ("pc", offset)):
                    bucket = acc[level]
                    bucket[key] = bucket.get(key, 0.0) + v
                kinds.setdefault((kind, cls), [0.0, 0.0])[side] += value
    site_pc: dict[tuple[str, str], list[float]] = {}
    for index, level in ((0, "pc"), (1, "pc_class")):
        for key, value in acc[level].items():
            offset = key[0] if level == "pc_class" else key
            if offset is None:
                site = ("unplaced", "-")
            else:
                site = (region(offset),
                        site_of.get(offset, (offset, "compute"))[1])
            site_pc.setdefault(site, [0.0, 0.0])[index] += abs(value)
    return {"errors": {level: sum(abs(v) for v in acc[level].values())
                       / denominator for level in LEVELS},
            "kinds": kinds, "site_pc": site_pc}


def compare_case(case: str, hardware: HardwareStalls, ledger: StallLedger,
                 function: KernelFunction, mapping: ClassMapping, *,
                 top: int = 20, repeat: HardwareStalls | None = None,
                 scope: str = "kernel") -> dict:
    """`repeat`: a second hardware sampling of the same launch; its
    distance from `hardware` at every level is reported as `floor`
    (the reference's own noise).  `scope` "loops": both sides are
    restricted to the pcs of the launch's loops (work loops and wait
    loops) plus the pc behind each loop, for probe kernels whose
    prologue is not the subject."""
    if scope not in ("kernel", "loops"):
        raise ValueError(f"unsupported:stall_compare_scope:{scope}")
    # --- engine: (offset | None, class) -> cycles per warp ---------------
    sim_pc: dict[int | None, dict[str, float]] = {}
    sim_issues: dict[int, float] = {}
    # waits only (no `issued`): offset -> reason -> cycles per warp
    sim_wait: dict[int, dict[str, float]] = {}
    sim_issue_cycles: dict[int, float] = {}
    reasons: dict[tuple[str, str], float] = {}
    excluded = 0.0
    for (address, reason), (cycles, events) in ledger.residency.items():
        cls = mapping.classify(reason,
                               ledger.operation_class.get(address))
        if cls is None:
            excluded += cycles / ledger.warps
            continue
        offset = function.offset(address)
        # where the cycles are booked: an instruction's issue cycle on
        # the pc `issue_pc_shift` instructions behind it when the
        # mapping says so (hardware samples the issue on the pc the
        # warp holds next) and that pc is inside the function
        booked = offset
        if reason == "issued" and offset is not None and \
                mapping.issue_pc_shift:
            shifted = offset + mapping.issue_pc_shift * INSTRUCTION_BYTES
            if shifted < function.instruction_count * INSTRUCTION_BYTES:
                booked = shifted
        row = sim_pc.setdefault(booked, {})
        row[cls] = row.get(cls, 0.0) + cycles / ledger.warps
        reasons[(reason, cls)] = reasons.get((reason, cls), 0.0) + \
            cycles / ledger.warps
        if reason == "issued" and offset is not None:
            sim_issues[offset] = sim_issues.get(offset, 0.0) + events
            sim_issue_cycles[booked] = sim_issue_cycles.get(booked, 0.0) + \
                cycles / ledger.warps
        elif offset is not None:
            waits = sim_wait.setdefault(offset, {})
            waits[reason] = waits.get(reason, 0.0) + cycles / ledger.warps

    # --- hardware: (offset, class) -> cycles per warp ---------------------
    # each class by its own counter (HardwareStalls.class_scales)
    scales = hardware.class_scales()
    hw_pc: dict[int, dict[str, float]] = {}
    for p in hardware.pcs:
        if p.samples:
            hw_pc[p.offset] = {c: n * scales[c]
                               for c, n in p.samples.items() if n}

    # --- regions ----------------------------------------------------------
    transitions: dict[tuple[int, int], int] = {}
    dropped_transitions = 0
    for (src, dst), count in ledger.transitions.items():
        a, b = function.offset(src), function.offset(dst)
        if a is None or b is None:
            dropped_transitions += count
            continue
        transitions[(a, b)] = transitions.get((a, b), 0) + count
    loops = find_loops(transitions, function.sass)
    repeat_pc: dict[int, dict[str, float]] | None = None
    if repeat is not None:
        if len(repeat.pcs) != len(hardware.pcs):
            raise ValueError("unsupported:stall_repeat_listing:"
                             f"{len(repeat.pcs)}!={len(hardware.pcs)}")
        repeat_scales = repeat.class_scales()
        repeat_pc = {p.offset: {c: n * repeat_scales[c]
                                for c, n in p.samples.items() if n}
                     for p in repeat.pcs if p.samples}
    if scope == "loops":
        keep: set[int] = set()
        for loop in loops:
            keep |= set(loop.pcs)
            keep.add(loop.last + INSTRUCTION_BYTES)
        if not keep:
            raise ValueError("missing_fact:stall_compare_loops (scope)")
        hw_pc = {o: r for o, r in hw_pc.items() if o in keep}
        sim_pc = {o: r for o, r in sim_pc.items() if o in keep}
        sim_wait = {o: r for o, r in sim_wait.items() if o in keep}
        sim_issue_cycles = {o: v for o, v in sim_issue_cycles.items()
                            if o in keep}
        if repeat_pc is not None:
            repeat_pc = {o: r for o, r in repeat_pc.items() if o in keep}
    # hardware pcs the replayed launch never issued take the region of
    # the code they belong to
    regions, far_pcs = place_unissued(
        region_of(loops), sim_issues, hw_pc, function.sass, transitions,
        INSTRUCTION_BYTES)

    def region(offset: int | None) -> str:
        if offset is None:
            return "unplaced"
        return regions.get(offset, "outside")

    table = {r: {} for r in REGIONS}       # region -> class -> [hw, sim]
    for offset, row in hw_pc.items():
        for cls, v in row.items():
            table[region(offset)].setdefault(cls, [0.0, 0.0])[0] += v
    for offset, row in sim_pc.items():
        for cls, v in row.items():
            table[region(offset)].setdefault(cls, [0.0, 0.0])[1] += v

    hw_total = sum(v[0] for r in table.values() for v in r.values())
    sim_total = sum(v[1] for r in table.values() for v in r.values())
    if hw_total <= 0:
        raise ValueError("missing_fact:hardware_residency (ncu_source.csv)")
    unissued = sum(sum(row.values()) for offset, row in hw_pc.items()
                   if offset not in sim_issues) / hw_total
    unissued_far = sum(sum(hw_pc[offset].values())
                       for offset in far_pcs) / hw_total

    def error(region_name: str) -> float:
        return sum(abs(v[1] - v[0])
                   for v in table[region_name].values()) / hw_total

    def totals(region_name: str) -> tuple[float, float]:
        rows = table[region_name].values()
        return sum(v[0] for v in rows), sum(v[1] for v in rows)

    def shape(region_names) -> float | None:
        hw, sim = {}, {}
        for name in region_names:
            for cls, v in table[name].items():
                hw[cls] = hw.get(cls, 0.0) + v[0]
                sim[cls] = sim.get(cls, 0.0) + v[1]
        return _total_variation(hw, sim)

    def ratio(region_name: str) -> float | None:
        hw, sim = totals(region_name)
        return sim / hw if hw > 0 else None

    # mapping-independent: residency per pc
    pc_error = 0.0
    pc_rows = []
    for offset in set(hw_pc) | {o for o in sim_pc if o is not None}:
        hw = sum(hw_pc.get(offset, {}).values())
        sim = sum(sim_pc.get(offset, {}).values())
        pc_error += abs(sim - hw)
        pc_rows.append((abs(sim - hw), offset, hw, sim))
    pc_error += sum(sim_pc.get(None, {}).values())
    pc_rows.sort(reverse=True)

    # straight-line runs: tolerant of the one-instruction difference
    # between where the engine and the hardware park a waiting warp
    blocks = basic_blocks(set(hw_pc) | {o for o in sim_pc if o is not None},
                          transitions, INSTRUCTION_BYTES)
    block_hw: dict[int, dict[str, float]] = {}
    block_sim: dict[int, dict[str, float]] = {}
    block_total: dict[int, list[float]] = {}
    for offset, row in hw_pc.items():
        total = block_total.setdefault(blocks[offset], [0.0, 0.0])
        total[0] += sum(row.values())
        waits = block_hw.setdefault(blocks[offset], {})
        for cls, v in row.items():
            if cls != "selected":
                waits[cls] = waits.get(cls, 0.0) + v
    for offset, row in sim_pc.items():
        if offset is not None:
            block_total.setdefault(blocks[offset], [0.0, 0.0])[1] += \
                sum(row.values())
    for offset, row in sim_wait.items():
        waits = block_sim.setdefault(blocks[offset], {})
        for reason, v in row.items():
            waits[reason] = waits.get(reason, 0.0) + v
    block_error = (sum(abs(sim - hw) for hw, sim in block_total.values())
                   + sum(sim_pc.get(None, {}).values())) / hw_total
    # which hardware class holds the waits where the engine charges a
    # reason: per block the overlapping wait cycles min(hw, sim) are
    # spread over (reason, class) by the two sides' shares
    cooccurrence: dict[tuple[str, str], float] = {}
    for leader, sim_row in block_sim.items():
        hw_row = block_hw.get(leader)
        if not hw_row:
            continue
        sim_sum, hw_sum = sum(sim_row.values()), sum(hw_row.values())
        if sim_sum <= 0 or hw_sum <= 0:
            continue
        overlap = min(sim_sum, hw_sum)
        for reason, s in sim_row.items():
            for cls, h in hw_row.items():
                key = (reason, cls)
                cooccurrence[key] = cooccurrence.get(key, 0.0) + \
                    overlap * (s / sim_sum) * (h / hw_sum)

    # sites: residency by the kind of instruction that holds the warp,
    # without the class mapping.  (region, kind) ->
    # [hw issue, hw wait, engine issue, engine wait]
    site_of = sites(set(hw_pc) | {o for o in sim_pc if o is not None},
                    transitions, function.sass, INSTRUCTION_BYTES)

    def kind_of(offset: int) -> str:
        return instruction_kind(function.sass.get(offset, ""))

    levels = level_errors(hw_pc, sim_pc, region=region, blocks=blocks,
                          site_of=site_of, kind_of=kind_of,
                          denominator=hw_total)
    # the reference's own noise: the repeat sampling against the first,
    # with regions / runs / sites laid out over the pcs of both
    floor = None
    if repeat_pc is not None:
        both = set(hw_pc) | set(repeat_pc)
        floor_regions, _ = place_unissued(
            region_of(loops), sim_issues, both, function.sass, transitions,
            INSTRUCTION_BYTES)
        floor = level_errors(
            hw_pc, repeat_pc,
            region=lambda o: floor_regions.get(o, "outside"),
            blocks=basic_blocks(both, transitions, INSTRUCTION_BYTES),
            site_of=sites(both, transitions, function.sass,
                          INSTRUCTION_BYTES),
            kind_of=kind_of, denominator=hw_total)["errors"]
    site_table: dict[tuple[str, str], list[float]] = {}
    site_total: dict[tuple[int, str], list[float]] = {}
    # what the waits of a (region, kind) are made of: hardware classes
    # and engine reasons, both unmapped
    site_hw: dict[tuple[str, str], dict[str, float]] = {}
    site_sim: dict[tuple[str, str], dict[str, float]] = {}
    for offset, row in hw_pc.items():
        first, kind = site_of[offset]
        key = (region(offset), kind)
        cell = site_table.setdefault(key, [0.0] * 4)
        issue = row.get("selected", 0.0)
        cell[0] += issue
        cell[1] += sum(row.values()) - issue
        site_total.setdefault((first, kind), [0.0, 0.0])[0] += \
            sum(row.values())
        parts = site_hw.setdefault(key, {})
        for cls, v in row.items():
            if cls != "selected":
                parts[cls] = parts.get(cls, 0.0) + v
    for offset, row in sim_pc.items():
        if offset is None:
            continue
        first, kind = site_of[offset]
        key = (region(offset), kind)
        cell = site_table.setdefault(key, [0.0] * 4)
        issue = sim_issue_cycles.get(offset, 0.0)
        cell[2] += issue
        cell[3] += sum(row.values()) - issue
        site_total.setdefault((first, kind), [0.0, 0.0])[1] += \
            sum(row.values())
        parts = site_sim.setdefault(key, {})
        for reason, v in sim_wait.get(offset, {}).items():
            parts[reason] = parts.get(reason, 0.0) + v
    site_error = (sum(abs(v[2] + v[3] - v[0] - v[1])
                      for v in site_table.values())
                  + sum(sim_pc.get(None, {}).values())) / hw_total

    # same launch on both sides: issues per pc.  Spin-wait runs are
    # left out of the gate (their iteration count depends on timing);
    # the count over all pcs is reported next to it
    executed = {p.offset: p.executed for p in hardware.pcs if p.executed}
    counted = set(executed) | set(sim_issues)
    count_blocks = basic_blocks(counted | set(blocks), transitions,
                                INSTRUCTION_BYTES)
    polls = poll_blocks(count_blocks, function.sass)

    def count_error(offsets) -> float | None:
        total = sum(executed.get(o, 0.0) for o in offsets)
        if total <= 0:
            return None
        return sum(abs(executed.get(o, 0.0) - sim_issues.get(o, 0.0))
                   for o in offsets) / total

    execution_error = count_error(
        [o for o in counted if count_blocks[o] not in polls])
    execution_error_all = count_error(counted)

    # static text agreement between the ncu listing and the capture
    mismatched = sum(
        1 for p in hardware.pcs
        if mnemonic_root(p.sass) != mnemonic_root(function.sass.get(
            p.offset, "")))

    def top_classes(row: dict[str, float]) -> dict[str, float]:
        return {c: round(v, 2) for c, v in
                sorted(row.items(), key=lambda kv: -kv[1])[:3]}

    loop_rows = []
    for loop in loops:
        hw = sum(sum(hw_pc.get(o, {}).values()) for o in loop.pcs)
        sim = sum(sum(sim_pc.get(o, {}).values()) for o in loop.pcs)
        loop_rows.append({"index": loop.index, "kind": loop.kind,
                          "first": loop.first, "last": loop.last,
                          "pcs": len(loop.pcs), "hw": round(hw, 3),
                          "sim": round(sim, 3)})

    return {
        "case": case, "status": "ok",
        "kernel": hardware.kernel, "function": function.name,
        "hardware": {
            "cycles": hardware.cycles, "warps": hardware.warps,
            "residency_per_warp": hardware.residency_per_warp,
            "samples": hardware.sample_total,
            "drain_samples": hardware.class_samples().get("drain", 0.0),
            "counters": hardware.counters,
            "reference_consistency": hardware.reference_consistency(),
            "selected_scale_ratio": hardware.selected_scale_ratio(),
            "class_scale_ratios": hardware.class_scale_ratios(),
            "unsampled_per_warp": hardware.unsampled(),
            "sampled_pcs": len(hw_pc), "listed_pcs": len(hardware.pcs),
        },
        "engine": {
            "cycles": ledger.cycles, "warps": ledger.warps,
            "residency_per_warp": sim_total,
            "excluded_per_warp": excluded,
            "unplaced_per_warp": sum(sim_pc.get(None, {}).values()),
            "dropped_transitions": dropped_transitions,
        },
        "alignment": {
            "function_instructions": function.instruction_count,
            "mnemonic_mismatches": mismatched,
            "execution_count_error": execution_error,
            "execution_count_error_all": execution_error_all,
            "warps_ratio": ledger.warps / hardware.warps,
            "hardware_unissued_share": unissued,
            "hardware_unissued_unplaced_share": unissued_far,
        },
        "metrics": {
            "cycle_ratio": ledger.cycles / hardware.cycles,
            "attribution_error": sum(error(r) for r in REGIONS),
            "attribution_error_loop": error("loop"),
            "attribution_error_outside": error("outside") + error("unplaced"),
            "residency_ratio": sim_total / hw_total,
            "residency_ratio_loop": ratio("loop"),
            "residency_ratio_outside": ratio("outside"),
            "hardware_loop_share": totals("loop")[0] / hw_total,
            "shape_distance": shape(REGIONS),
            "shape_distance_loop": shape(("loop",)),
            "shape_distance_outside": shape(("outside", "unplaced")),
            "pc_residency_error": pc_error / hw_total,
            "block_residency_error": block_error,
            "site_residency_error": site_error,
            "class_error": levels["errors"]["class"],
            "kind_class_error": levels["errors"]["kind_class"],
            "site_class_error": levels["errors"]["site_class"],
            "block_class_error": levels["errors"]["block_class"],
            "pc_class_error": levels["errors"]["pc_class"],
            "kind_residency_error": levels["errors"]["kind"],
        },
        "scope": scope,
        "floor": floor,
        "kinds": [{"kind": kind, "class": cls,
                   "hw": round(v[0], 4), "sim": round(v[1], 4)}
                  for (kind, cls), v in sorted(levels["kinds"].items())
                  if max(v) >= 1e-5 * hw_total],
        "site_pc": [{"region": r, "kind": kind,
                     "pc_error": round(v[0], 4),
                     "pc_class_error": round(v[1], 4)}
                    for (r, kind), v in sorted(levels["site_pc"].items())],
        "table": {r: {c: [round(v[0], 4), round(v[1], 4)]
                      for c, v in sorted(rows.items())}
                  for r, rows in table.items()},
        "loops": loop_rows,
        "sites": [{"region": r, "kind": kind,
                   "hw_issue": round(v[0], 4), "hw_wait": round(v[1], 4),
                   "sim_issue": round(v[2], 4), "sim_wait": round(v[3], 4),
                   "hw_classes": {c: round(x, 3) for c, x in sorted(
                       site_hw.get((r, kind), {}).items(),
                       key=lambda kv: -kv[1]) if x >= 0.0005 * hw_total},
                   "sim_reasons": {c: round(x, 3) for c, x in sorted(
                       site_sim.get((r, kind), {}).items(),
                       key=lambda kv: -kv[1]) if x >= 0.0005 * hw_total}}
                  for (r, kind), v in sorted(site_table.items())],
        "site_rows": [{"first": first, "kind": kind,
                       "region": region(first),
                       "sass": function.sass.get(first, ""),
                       "hw": round(hw, 3), "sim": round(sim, 3)}
                      for (first, kind), (hw, sim) in sorted(
                          site_total.items(),
                          key=lambda kv: -abs(kv[1][1] - kv[1][0]))[:top]],
        "cooccurrence": [
            {"reason": reason, "class": cls, "cycles": round(v, 3)}
            for (reason, cls), v in
            sorted(cooccurrence.items(), key=lambda kv: -kv[1])
            if v >= 0.0005 * hw_total],
        "reasons": [{"reason": reason, "class": cls, "cycles": round(v, 3)}
                    for (reason, cls), v in
                    sorted(reasons.items(), key=lambda kv: -kv[1])[:2 * top]],
        "pcs": [{"offset": offset, "region": region(offset),
                 "sass": function.sass.get(offset, ""),
                 "hw": round(hw, 3), "sim": round(sim, 3),
                 "hw_classes": top_classes(hw_pc.get(offset, {})),
                 "sim_classes": top_classes(sim_pc.get(offset, {}))}
                for _, offset, hw, sim in pc_rows[:top]],
    }
