#!/usr/bin/env python3
"""instruction.fetch leaf derivation per tools/microbench/ifetch/protocol.json.

Reads one probe run: <run>/{all,none}/{launches.log,counters.csv} (single
GPU; the `all` mode is --cache-control all, `none` is the reference run).
Cell value = median over launches >= 2 of the slowest warp's per-pass
cycles.  Overhead-corrected cpi_c(N) = (T(N) - a0)/N with a0 the intercept
of the least-squares line through the three smallest footprints (same
correction the predecessor's amendment (a) introduced); the warm value of a
cell is the median of passes 2 and 3 (pass 1 is the L1 transition pass).

Writes model.json: every intermediate table (per-N/W/pass cpi, jump
per-block cycles, cold table) plus the 9 protocol fields, each with the
literal protocol form and, where the predecessor used a different form,
that form as well (the derivation notes which was taken as `value`).

Usage: model.py --run <run dir> --out <model.json>
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import re
import statistics
import sys

LAUNCH_RE = re.compile(
    r"^LAUNCH ifetch variant=ifetch\.(line|jump|mix)\.(\S+)"
    r" launch=(\d+) smid=([0-9,]+) pass=(\d+) cycles=([0-9,]+)\s*$")
LAUNCH_RECOVER_RE = re.compile(
    r"^LAUNCH ifetch variant=ifetch\.(line|jump|mix)\.(\S+)"
    r" launch=(\d+) smid=(.*)$")

LINE_BYTES = 16    # bytes per instruction (16 B)
BLOCK_BYTES = 128  # jump block = one 128 B line
DROPPED_LAUNCHES = {0, 1}
K_PASSES = 4

COUNTER_COLUMNS = [
    "ID", "Kernel Name", "Block Size", "Grid Size",
    "sm__cycles_active.max",
    "smsp__warps_issue_stalled_no_instruction.sum",
    "smsp__warps_active.sum",
    "smsp__inst_executed.sum",
]


def _die(msg):
    raise SystemExit(f"model: {msg}")


def _median(v):
    return float(statistics.median(v))


def _iqr(v):
    if len(v) < 2:
        return 0.0
    q1, _, q3 = statistics.quantiles(v, n=4)
    return float(q3 - q1)


def _linfit(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in points) / den
    a = ym - b * xm
    res = max(abs(t - (a + b * n)) for n, t in points)
    return a, b, res


def _parse_variant(kind, rest):
    if kind == "line":
        m = re.fullmatch(r"n(\d+)\.w(\d+)", rest)
        if not m:
            _die(f"bad line variant {rest!r}")
        return {"kind": "line", "n": int(m.group(1)), "w": int(m.group(2)),
                "key": f"line.n{m.group(1)}.w{m.group(2)}"}
    if kind == "mix":
        m = re.fullmatch(r"n(\d+)\.w(\d+)", rest)
        if not m:
            _die(f"bad mix variant {rest!r}")
        return {"kind": "mix", "n": int(m.group(1)), "w": int(m.group(2)),
                "key": f"mix.n{m.group(1)}.w{m.group(2)}"}
    m = re.fullmatch(r"m(\d+)\.p([01])\.w(\d+)", rest)
    if not m:
        _die(f"bad jump variant {rest!r}")
    return {"kind": "jump", "m": int(m.group(1)), "perm": int(m.group(2)),
            "w": int(m.group(3)),
            "key": f"jump.m{m.group(1)}.p{m.group(2)}.w{m.group(3)}"}


def parse_launches(path):
    entries = []
    stats = {"intact": 0, "recovered": 0, "dropped_corrupt": 0}
    smids = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        m = LAUNCH_RE.match(s)
        if m:
            v = _parse_variant(m.group(1), m.group(2))
            cycles = [int(x) for x in m.group(6).split(",")]
            if len(cycles) != v["w"]:
                _die(f"{path}: {s} has {len(cycles)} warp values, want {v['w']}")
            entries.append((v, int(m.group(3)), m.group(4),
                            int(m.group(5)), cycles, None))
            smids.append(m.group(4))
            stats["intact"] += 1
            continue
        rm = LAUNCH_RECOVER_RE.match(s)
        if not rm:
            continue
        v = _parse_variant(rm.group(1), rm.group(2))
        groups = re.findall(r"\d+", rm.group(4))
        if len(groups) == v["w"] + 1:
            entries.append((v, int(rm.group(3)), groups[0], None,
                            [int(x) for x in groups[1:]], None))
            smids.append(groups[0])
            stats["recovered"] += 1
        elif len(groups) == v["w"]:
            entries.append((v, int(rm.group(3)), None, None, None, groups))
        else:
            stats["dropped_corrupt"] += 1
    smid = statistics.mode(smids) if smids else None
    records = []
    by_key = {}
    for v, launch, sm, ps, cycles, groups in entries:
        key = (v["key"], launch)
        if cycles is None:
            if smid is None or not groups[0].startswith(str(smid)) \
                    or len(groups[0]) <= len(str(smid)):
                stats["dropped_corrupt"] += 1
                continue
            cycles = [int(groups[0][len(str(smid)):])] + \
                     [int(x) for x in groups[1:]]
            stats["recovered"] += 1
        if ps is None:
            rec = by_key.get(key)
            ps = len(rec["passes"]) if rec else 0
        if key not in by_key:
            rec = {**v, "launch": launch, "passes": {}}
            by_key[key] = rec
            records.append(rec)
        by_key[key]["passes"][ps] = cycles
    if not records:
        _die(f"{path}: no LAUNCH lines")
    return records, stats


def parse_counters(path):
    text = Path(path).read_text()
    header = re.search(r'^(?:"ID"|ID),', text, re.MULTILINE)
    if header is None:
        return []
    reader = csv.reader(io.StringIO(text[header.start():]))
    rows = [r for r in reader if r]
    names = rows[0]
    have = [c for c in COUNTER_COLUMNS if c in names]
    idx = {c: names.index(c) for c in have}
    out = []
    for r in rows[2:]:
        if len(r) < len(names):
            continue
        out.append({c: r[idx[c]] for c in have})
    return out


def _int(value):
    d = Decimal(str(value).replace(",", "").strip())
    return int(d) if d.is_finite() else 0


def _row_kernel_key(name):
    mk = re.search(r"k_lineILi(\d+)E", name) or re.search(r"k_line<(\d+)>", name)
    if mk:
        return ("line", int(mk.group(1)))
    mk = (re.search(r"k_mixILi(\d+)ELi(\d+)E", name) or
          re.search(r"k_mix<(\d+),\s*(\d+)>", name))
    if mk:
        return ("mix", int(mk.group(1)), int(mk.group(2)))
    mk = (re.search(r"k_jumpILi(\d+)ELi(\d)E", name) or
          re.search(r"k_jump<(\d+),\s*(\d+)>", name))
    if mk:
        return ("jump", int(mk.group(1)), int(mk.group(2)))
    return None


def load_mode(run, mode):
    records, stats = parse_launches(Path(run) / mode / "launches.log")
    counters = parse_counters(Path(run) / mode / "counters.csv")
    ci = 0
    for rec in records:
        if ci >= len(counters):
            break
        key = _row_kernel_key(counters[ci]["Kernel Name"])
        if key is None:
            continue
        want = (("jump", rec["m"], rec["perm"]) if rec["kind"] == "jump"
                else ("mix", rec["n"], rec["w"]) if rec["kind"] == "mix"
                else ("line", rec["n"]))
        if key != want:
            continue
        row = counters[ci]
        ci += 1
        for f in ("sm__cycles_active.max",
                  "smsp__warps_issue_stalled_no_instruction.sum",
                  "smsp__warps_active.sum", "smsp__inst_executed.sum"):
            if f in row:
                rec[f] = _int(row[f])
    return records, stats


def build_cells(records):
    cells = {}
    for rec in records:
        for ps, cycles in rec["passes"].items():
            e = cells.setdefault((rec["key"], ps), {"values": [], "n_dropped": 0})
            if rec["launch"] in DROPPED_LAUNCHES:
                e["n_dropped"] += 1
                continue
            e["values"].append(max(cycles))
    out = {}
    for (vkey, ps), e in sorted(cells.items()):
        vals = e["values"]
        warm = ps >= 1
        cell = {"n": len(vals), "warm": warm}
        if vals:
            cell["median"] = _median(vals)
            cell["iqr"] = _iqr(vals)
            tol = (32.0, 0.05) if warm else (64.0, 0.15)
            cell["stable"] = cell["iqr"] <= max(tol[0], tol[1] * cell["median"])
        else:
            cell.update({"median": None, "iqr": None, "stable": False})
        out[(vkey, ps)] = cell
    return out


def cell_median(cells, vkey, ps):
    c = cells.get((vkey, ps))
    return c["median"] if c and c["stable"] else None


def warm_T(cells, vkey):
    m2, m3 = cell_median(cells, vkey, 2), cell_median(cells, vkey, 3)
    if m2 is None or m3 is None:
        return None
    if abs(m2 - m3) > max(16.0, 0.03 * _median([m2, m3])):
        return None
    return _median([m2, m3])


def plateaus(cpi, ns):
    """Assign each footprint L0 / L1 / L2 / transition per the predecessor's
    f2 logic (10 % plateau groups, skipping singleton transition points)."""
    base = cpi[ns[0]]
    tol0 = max(0.05, 0.05 * base)
    lvl0 = [n for n in ns if cpi[n] <= base + tol0]
    groups = []
    for n in ns:
        if n in lvl0:
            continue
        if groups and cpi[n] <= cpi[groups[-1][0]] * 1.10:
            groups[-1].append(n)
        else:
            groups.append([n])
    plats = [g for g in groups if len(g) >= 2]
    return {"base": base, "L0": lvl0,
            "L1": plats[0] if plats else [],
            "L2": plats[1] if len(plats) > 1 else [],
            "transition": [g[0] for g in groups if len(g) == 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", help="sass check.json; failed kernels are void")
    a = ap.parse_args()
    run = Path(a.run)
    void_keys = set()
    if a.gate and Path(a.gate).exists():
        gd = json.loads(Path(a.gate).read_text())
        for k, r in gd.get("kernels", {}).items():
            if not r.get("ok"):
                # 'ifetch/jump/4096/p1' -> jump.m4096.p1.w1 etc.
                parts = k.split("/")
                if parts[1] == "jump":
                    void_keys.add(f"jump.m{parts[2]}.p{parts[3][1:]}.w1")
                elif parts[1] == "line":
                    void_keys.add(f"line.n{parts[2]}.w1")
                elif parts[1] == "mix":
                    void_keys.add(f"mix.n{parts[2]}.w{parts[3][1:]}")
    out = {"run": str(run), "modes": {}, "fields": {},
           "gate_void_variants": sorted(void_keys)}

    per_mode = {}
    for mode in ("all", "none"):
        if not (run / mode / "launches.log").exists():
            continue
        records, stats = load_mode(run, mode)
        out["modes"][mode] = {"parse_stats": stats,
                              "n_records": len(records)}
        per_mode[mode] = (records, build_cells(records))
    if "all" not in per_mode:
        _die("missing <run>/all/launches.log")
    cells = per_mode["all"][1]      # the installed mode
    cells_none = per_mode.get("none", (None, {}))[1]

    # ---- line footprints (W=1): warm T(N), cpi_c, cold pass-0 -----------
    ns = sorted({rec["n"] for rec in per_mode["all"][0] if rec["kind"] == "line"})
    T = {n: warm_T(cells, f"line.n{n}.w1") for n in ns}
    T = {n: t for n, t in T.items() if t is not None}
    ns = sorted(T)
    if len(ns) < 4:
        _die(f"only {len(ns)} usable W=1 footprints")
    a0, b0, _ = _linfit([(n, T[n]) for n in ns[:3]])
    cpi = {n: (T[n] - a0) / n for n in ns}
    cold_T = {n: cell_median(cells, f"line.n{n}.w1", 0) for n in ns}
    cold_cpi = {n: (cold_T[n] - a0) / n for n in ns if cold_T[n] is not None}
    # per-pass cpi table (every N, W, pass) for the report
    cpi_table = {}
    ws = sorted({rec["w"] for rec in per_mode["all"][0] if rec["kind"] == "line"})
    for w in ws:
        for n in ns:
            v = {}
            for ps in range(K_PASSES):
                m = cell_median(cells, f"line.n{n}.w{w}", ps)
                v[ps] = ((m - a0) / n) if m is not None else None
            cpi_table[f"n{n}.w{w}"] = v
    out["line"] = {"ns": ns, "a0": a0, "b0": b0, "T_w1": T,
                   "cpi_c_w1": cpi, "cold_T": cold_T, "cold_cpi": cold_cpi,
                   "cpi_table": cpi_table}

    pl = plateaus(cpi, ns)
    base = pl["base"]
    out["plateaus"] = {k: v for k, v in pl.items() if k != "base"}
    out["plateaus"]["cpi_base"] = base
    fields = {}

    # capacity_l0: largest footprint whose warm cpi stays within 2 % of the
    # smallest-N cpi (protocol text).  The plateau form (5 % prefix rule)
    # is recorded alongside.
    l0_proto = max(n for n in ns if cpi[n] <= base * 1.02)
    fields["capacity_l0_bytes"] = {
        "value": l0_proto * LINE_BYTES,
        "plateau_form_bytes": max(pl["L0"]) * LINE_BYTES,
        "note": "protocol: within 2 % of smallest-N warm cpi"}
    # capacity_l1: largest footprint on the second plateau
    fields["capacity_l1_bytes"] = {
        "value": (max(pl["L1"]) * LINE_BYTES) if pl["L1"] else None,
        "l1_plateau_ns": pl["L1"], "l2_plateau_ns": pl["L2"],
        "transition_ns": pl["transition"]}

    # streams: cpi-difference form (plateau median - base) * 8, and the
    # predecessor's least-squares marginal-slope form for cross-check.
    def slope_form(level_ns, tmap):
        pts = [(n, tmap[n]) for n in level_ns if n in tmap]
        if len(pts) < 2:
            return None
        a, b, r = _linfit(pts)
        return {"slope_cpi": b, "stream_x8": 8.0 * b, "intercept": a,
                "residual_max": r, "n_points": len(pts)}

    l1_med = _median([cpi[n] for n in pl["L1"]]) if pl["L1"] else None
    l2_med = _median([cpi[n] for n in pl["L2"]]) if pl["L2"] else None
    fields["stream_l1_cycles_per_line"] = {
        "value": (l1_med - base) * 8 if l1_med is not None else None,
        "slope_form": slope_form(pl["L1"], T),
        "note": "value = (L1 plateau cpi - L0 cpi) * 8"}
    fields["stream_l2_cycles_per_line"] = {
        "value": (l2_med - base) * 8 if l2_med is not None else None,
        "slope_form": slope_form(pl["L2"], T)}

    # launch_prefetch_bytes: largest footprint whose cold pass-0 cpi_c is
    # within 10 % of the warm L1 plateau median.
    prefetch_n = None
    if l1_med is not None:
        inside = [n for n in ns if n in cold_cpi
                  and abs(cold_cpi[n] - l1_med) <= 0.10 * l1_med]
        prefetch_n = max(inside) if inside else None
    fields["launch_prefetch_bytes"] = {
        "value": prefetch_n * LINE_BYTES if prefetch_n is not None else None,
        "prefetch_n": prefetch_n}

    # stream_cold: protocol literal (pass-0 cpi - L0 cpi) * 8 per footprint
    # beyond prefetch (median), plus the predecessor's slope-of-T0 fit.
    beyond = [n for n in ns if prefetch_n is not None and n > prefetch_n
              and n in cold_cpi]
    per_n = {n: (cold_cpi[n] - base) * 8 for n in beyond}
    cold_slope = slope_form(beyond, {n: cold_T[n] for n in beyond})
    fields["stream_cold_cycles_per_line"] = {
        "value": _median(list(per_n.values())) if per_n else None,
        "per_footprint": per_n,
        "slope_form": cold_slope,
        "note": "value = median over beyond-prefetch N of (cold_cpi - cpi_base) * 8"}

    # fetch_parallelism from k_jump: (per-block cycles PERM=1 - PERM=0)
    # divided by stream_cold -> fills in flight; nearest power of two.
    jump = {}
    jump_void = set()
    cold_per_block = fields["stream_cold_cycles_per_line"]["value"]
    for rec in per_mode["all"][0]:
        if rec["kind"] != "jump":
            continue
        key = (rec["m"], rec["perm"])
        if key in jump:
            continue
        vkey = f"jump.m{rec['m']}.p{rec['perm']}.w1"
        if vkey in void_keys:
            jump_void.add(vkey)
            jump[key] = None
            continue
        t = warm_T(cells, vkey)
        jump[key] = t / rec["m"] if t is not None else None
    out["jump_per_block_cycles"] = {
        f"m{m}.p{p}": v for (m, p), v in sorted(jump.items())}
    out["jump_void"] = sorted(jump_void)
    diffs = {m: (jump.get((m, 1)) or 0) - (jump.get((m, 0)) or 0)
             for m in {m for m, _ in jump}
             if jump.get((m, 1)) is not None and jump.get((m, 0)) is not None
             and f"jump.m{m}.p1.w1" not in jump_void
             and f"jump.m{m}.p0.w1" not in jump_void}
    out["jump_perm_diff_per_block"] = diffs
    parallelism = None
    if cold_per_block:
        largest = max(diffs) if diffs else None
        # use the largest footprint (deepest level) as primary; all in table
        use_m = max(diffs, key=lambda m: m) if diffs else None
        if use_m is not None and diffs[use_m] > 0:
            raw = diffs[use_m] / cold_per_block
            p = 1
            while p * 2 <= raw * 1.42:
                p *= 2
            parallelism = p
            fields["fetch_parallelism"] = {
                "value": parallelism, "raw": raw, "from_m": use_m,
                "diff_per_block": diffs[use_m],
                "note": "(p1 - p0 per-block) / stream_cold, nearest power of two"}
    if "fetch_parallelism" not in fields:
        fields["fetch_parallelism"] = {"value": None, "diff_per_block": diffs}

    # sharing: 0 if k_line warm cpi flat in W (W <= 4), else 1
    sharing = {"verdict": None, "ratios": {}}
    ok_all = True
    for w in ws:
        if w == 1:
            continue
        for n in ns:
            t = warm_T(cells, f"line.n{n}.w{w}")
            if t is None:
                continue
            rw = (t - a0) / n
            ok = abs(rw - cpi[n]) <= max(0.05, 0.10 * cpi[n])
            sharing["ratios"][f"w{w}.n{n}"] = {
                "cpi": rw, "w1": cpi[n], "within_10pct": ok}
            if w <= 4:
                ok_all = ok_all and ok
    sharing["verdict"] = 0 if ok_all else 1
    fields["sharing"] = {"value": sharing["verdict"], "detail": sharing}
    fields["line_bytes"] = {"value": 128, "note": "SASS gate: block starts 128 B apart"}

    out["fields"] = fields
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"model: {a.out} fields:")
    for k, f in fields.items():
        print(f"  {k}: {f.get('value')}")


if __name__ == "__main__":
    main()
