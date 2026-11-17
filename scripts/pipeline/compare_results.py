#!/usr/bin/env python3
"""Side-by-side result.bin comparison (cycles, WAIT buckets, TELE).

    compare_results.py <a.result.bin> <b.result.bin> [--a-label A --b-label B]
        [--a-g4 legacy]

WAIT buckets are named after `WaitBucket` (cpp/core/simulator.hpp);
TELE group 2 after `EngineTele`, group 4 after the memory-hierarchy
array in Simulator::run().  `--a-g4 legacy` / `--b-g4 legacy` names group
4 of that file with the pre-2026-11-03 14-entry layout (footprint_seeds
at 11, l2_evictions at 12, warmup_accesses at 13) so rounds produced
before the cold-start switch read correctly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cyclescope.job.result import read_result  # noqa: E402

WAIT = ["dispatch", "fetch", "collective", "semantic", "resource",
        "register", "scheduler"]
G1 = ["mbarrier_phases", "mbarrier_arrivals", "mbarrier_tx_bytes",
      "mbarrier_invals", "mbarrier_remote_arrivals",
      "cta_barrier_generations", "cluster_barrier_generations",
      "failed_polls", "poll_suspend_cycles", "nanosleeps",
      "visibility_settled", "mbarrier_inval_inflight", "tma_completions",
      "tma_bytes", "descriptor_acquire_unpublished",
      "gsync_acquires_from", "gsync_acquires_any", "gsync_acquires_count",
      "gsync_acquires_initial", "gsync_dropped_polls",
      "gsync_chains_unique", "gsync_chains_nonunique",
      "gsync_update_unknown_post", "gsync_wait_cycles", "stale_polls",
      "suspended_waits", "store_buffer_stalls",
      "store_buffer_stall_cycles", "abandoned_polls",
      "gsync_acquires_no_writer", "gsync_regions_retired",
      "gsync_region_instances", "fence_drain_holds",
      "fence_drain_cycles"]
G2 = ["dispatch_entries", "dispatch_waves", "dispatch_wait_cycles",
      "lifecycle_envelope", "fetch_unavailable", "fetch_l0_hits",
      "fetch_l1_lines", "fetch_l2_lines", "fetch_cold_lines",
      "fetch_wait_cycles", "cta_span_mean", "cta_span_max"]
G4 = ["l1_sectors", "l1_misses", "l2_sectors", "dram_sectors",
      "dram_bytes", "merge_merges", "l2_far_hits", "l2_replica_fills",
      "dirty_writebacks", "shared_waits", "shared_wait_cycles",
      "l2_evictions", "const_same_line", "const_hits", "const_misses",
      "l2_inflight_hits", "l2_inflight_wait", "atomic_fills",
      "loaded_extra_dram_cycles", "loaded_extra_dram_count",
      "loaded_extra_l2_cycles", "loaded_extra_l2_count",
      "port_share_waits", "port_share_wait_cycles",
      "ordered_load_waits", "ordered_load_wait_cycles",
      "phantom_sectors"]
G4_LEGACY = G4[:11] + ["footprint_seeds", "l2_evictions",
                       "warmup_accesses"]


def load(path: Path, g4_layout: str) -> dict:
    r = read_result(path.read_bytes())
    waits = {WAIT[w["cause_kind"]] if w["cause_kind"] < len(WAIT)
             else f"wait{w['cause_kind']}": w["cycles"]
             for w in r["wait_buckets"]}
    tele: dict[str, int] = {}
    names4 = G4_LEGACY if g4_layout == "legacy" else G4
    for t in r["telemetry"]:
        g, c, v = t["group"], t["counter"], t["value"]
        if g == 1 and c < len(G1):
            name = f"g1.{G1[c]}"
        elif g == 2 and c < len(G2):
            name = f"g2.{G2[c]}"
        elif g == 4 and c < len(names4):
            name = f"g4.{names4[c]}"
        else:
            name = f"g{g}.{c}"
        tele[name] = v
    return {"cycles": r["cycles"], "issued": r["issued"],
            "completed": r["completed"], "wait": waits, "tele": tele}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--a-g4", default="current", choices=["current", "legacy"])
    ap.add_argument("--b-g4", default="current", choices=["current", "legacy"])
    args = ap.parse_args()
    a = load(args.a, args.a_g4)
    b = load(args.b, args.b_g4)
    la, lb = args.a_label, args.b_label
    print(f"{'':28} {la:>14} {lb:>14} {'b/a':>8}")
    for k in ("cycles", "issued", "completed"):
        va, vb = a[k], b[k]
        print(f"{k:28} {va:>14} {vb:>14} "
              f"{(vb / va if va else 0):>8.2f}")
    print("WAIT buckets (cycles):")
    for k in WAIT:
        va, vb = a["wait"].get(k, 0), b["wait"].get(k, 0)
        if va or vb:
            print(f"  {k:26} {va:>14} {vb:>14} "
                  f"{(vb / va if va else 0):>8.2f}")
    print("TELE:")
    for k in sorted(set(a["tele"]) | set(b["tele"])):
        va, vb = a["tele"].get(k, 0), b["tele"].get(k, 0)
        if va != vb or k.startswith("g4.") or k.startswith("g2."):
            print(f"  {k:26} {va:>14} {vb:>14} "
                  f"{(vb / va if va else 0):>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
