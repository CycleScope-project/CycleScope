#!/usr/bin/env python3
"""Derive instruction.service result_latency for the new SYNCS.PHASECHK
class from the barrier_release v2 tw_only chains (protocol-v2 addendum
2026-11-06): value = OLS slope of the r_sync chain medians over U minus
the modal control-word stall of the unit's `@!P BRA` in the run's
sync_mb SASS (kernel k_sync1I Li6 E LiU, U=32 window).

Usage: model_tw_only.py <artifact_root> <device-run> [...]
Writes <artifact_root>/microbench/barrier_release-model/<run>/tw_only.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse, clock_windows  # noqa: E402

KERNEL = "k_sync1ILi6ELi32"
RESID_MAX = 0.02
AGREE_MAX = 0.03


def ols(points: list[list[float]], skip_first: bool = False
        ) -> tuple[float, float, float]:
    """OLS slope/intercept on all points; max relative residual of the
    fit.  With skip_first the residual skips the U=1 point (the window's
    first unit carries the entry boundary, same role as the last unit's
    loop-backedge BRA stall)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    xm = sum(xs) / n
    ym = sum(ys) / n
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    sxx = sum((x - xm) ** 2 for x in xs)
    slope = sxy / sxx
    intercept = ym - slope * xm
    pts = points[1:] if skip_first else points
    resid = max(abs(y - (slope * x + intercept)) / y for x, y in pts)
    return slope, intercept, resid


def unit_stalls(kernel) -> dict:
    """In the tw_only clock window each unit is TRYWAIT + `@!P BRA`;
    the BRA stall must be identical across units except the last."""
    win = None
    for w in clock_windows(kernel):
        if any("PHASECHK" in i.text and "TRYWAIT" in i.text for i in w):
            win = w
            break
    if win is None:
        raise SystemExit("no TRYWAIT clock window in " + kernel.name)
    units = []  # (trywait_stall, bra_stall)
    for i, ins in enumerate(win):
        if "PHASECHK" in ins.text and "TRYWAIT" in ins.text:
            bra = next(
                (j for j in win[i + 1:]
                 if j.opcode == "BRA" and j.pred and j.pred.startswith("@!")),
                None)
            units.append((ins.stall, bra.stall if bra else None))
    # literal protocol reading: every unit except the last must carry
    # the same BRA stall
    stalls = [b for _, b in units[:-1] if b is not None]
    modal = Counter(stalls).most_common(1)[0][0] if stalls else None
    identical = modal is not None and all(b == modal for b in stalls)
    return {
        "units": len(units),
        "trywait_stalls": sorted({t for t, _ in units}),
        "bra_stall_mode": modal,
        "bra_stall_identical": identical,
        "bra_stall_first": units[0][1] if units else None,
        "trywait_stall_first": units[0][0] if units else None,
        "bra_stall_last": units[-1][1] if units else None,
    }


def main() -> int:
    root = Path(sys.argv[1])
    out_all = {}
    for run in sys.argv[2:]:
        model = json.load(open(
            root / "microbench" / "barrier_release-model" / run / "model.json"))
        sass = open(
            root / "microbench" / "barrier_release" / run / "build" / "sass"
            / "sync_mb.sass").read()
        kernels = parse(sass)
        kern = next(k for k in kernels.values() if KERNEL in k.name)
        st = unit_stalls(kern)
        slopes = {}
        resids = {}
        for arm in ("a", "b"):
            pts = model["derived"][arm]["r_sync"]["points"]
            slope, intercept, resid = ols(pts)
            slopes[arm] = slope
            resids[arm] = resid
        resid_interior = {
            a: ols(model["derived"][a]["r_sync"]["points"],
                   skip_first=True)[2]
            for a in ("a", "b")}
        agree = abs(slopes["a"] - slopes["b"]) / ((slopes["a"] + slopes["b"]) / 2)
        value = (slopes["a"] + slopes["b"]) / 2 - (st["bra_stall_mode"] or 0)
        ok = (all(r <= RESID_MAX for r in resids.values())
              and agree <= AGREE_MAX and st["bra_stall_identical"])
        out = {
            "run": run,
            "kernel": kern.name,
            "slope": slopes,
            "intercept": {a: ols(model["derived"][a]["r_sync"]["points"])[1]
                          for a in ("a", "b")},
            "resid": resids,
            "resid_interior": resid_interior,
            "first_unit_stall": {
                "trywait": st["trywait_stall_first"],
                "bra": st["bra_stall_first"]},
            "agree": agree,
            "stall_BRA": st["bra_stall_mode"],
            "trywait_stalls": st["trywait_stalls"],
            "bra_stall_identical": st["bra_stall_identical"],
            "value": round(value, 2),
            "verdict": "measured" if ok else "experimental",
        }
        path = root / "microbench" / "barrier_release-model" / run / "tw_only.json"
        json.dump(out, open(path, "w"), indent=1)
        out_all[run] = out
        print(json.dumps(out, indent=1))
    return 0 if all(o["verdict"] == "measured" for o in out_all.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
