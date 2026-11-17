#!/usr/bin/env python3
"""Generate the synthetic 3-segment container fixture for CTest.

Usage: make_fixtures.py <out_dir>   (run with PYTHONPATH=<repo>/python)
"""

from __future__ import annotations

import sys
from pathlib import Path

from cyclescope.job import container as cf, segment_format as sf


def make_segment(cluster_id: int, n_inst: int) -> bytes:
    """Segment k: shared program prefix (STRS "ready_warps_smsp", one
    init FACT row) + its own interned opcode string "OP<k>" and one
    per-instance FACT row (value 100 + k) referenced by a template USES
    row — exercises the merge's STRS / FACT tail rebasing."""
    sw = sf.SegmentWriter()
    tables = {t: w.payload() for t, w in sw.writers.items()}
    tables["META"] = sf.META.pack(
        tree_digest=b"\1" * 32, program_digest=b"\2" * 32,
        converter_digest=b"\3" * 32, cluster_id=cluster_id, cta_count=1,
        warp_count=1, instruction_count=n_inst, flags=0)
    prefix = b"ready_warps_smsp\0"
    op_off = len(prefix)
    tables["STRS"] = prefix + f"OP{cluster_id}\0".encode()
    tables["AXIS"] = sf.AXIS.pack_row((0,))
    tables["FACT"] = (sf.FACT.pack_row((0, 7)) +
                      sf.FACT.pack_row((0, 100 + cluster_id)))
    # launch-wide linear CTA id 7/3/9 (the merge renumbers densely)
    linear = {0: 7, 1: 3, 2: 9}.get(cluster_id, cluster_id)
    tables["CTAS"] = sf.CTAS.pack_row(
        (linear, linear, 0, 0, 0, 1, 1, 1, 0, 0xFFFFFFFF, 0, 1, 0, 0, 0,
         1))
    tables["WARP"] = sf.WARP.pack_row((linear, 0, 0, 0, 0, n_inst))
    tables["INST"] = b"".join(
        sf.INST.pack_row((0, 0xFFFFFFFF, 0xFFFFFFFF, i))
        for i in range(n_inst))
    tables["TMPL"] = sf.TMPL.pack_row(
        (op_off, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0))
    tables["NODE"] = sf.NODE.pack_row(
        (0xFFFFFFFF, 0, 0, 0, 0, 0))
    tables["LEAF"] = sf.LEAF.pack_row((0, 0, b"\0" * 3))
    tables["USES"] = (sf.USES.pack_row((0, 9, b"\0" * 3, 0, 1, 0x6, 0, 1)) +
                      sf.USES.pack_row((0, 0, b"\0" * 3, 0, 1, 0x6, 1, 1)))
    return sf.write_segment(tables)


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = [make_segment(i, i + 1) for i in range(3)]
    # PLCE rows by launch-wide id: 3 (present -> dense 1), 42 (absent)
    plce = (cf.PLCE.pack_row((3, 0, 0, 1, b"\0" * 3, 0, 0, 1, b"\0" * 3))
            + cf.PLCE.pack_row((42, 0, 0, 1, b"\0" * 3, 6, 0, 1,
                                b"\0" * 3)))
    container = cf.write_container(
        {"device": "h100", "sm_count": 132, "reference_cache_control": 2},
        segments, placements=plce)
    (out_dir / "job.bin").write_bytes(container)
    (out_dir / "seg0.seg").write_bytes(segments[0])
    (out_dir / "engine.seg").write_bytes(make_engine_segment())
    (out_dir / "engine-unplaced.seg").write_bytes(
        make_engine_segment(unplaced=True))


def make_engine_segment(unplaced: bool = False) -> bytes:
    """Segment with a real program subtree: constant/discrete/linear/
    affine fields + an engine-axis guard, exercising MechanismRuntime."""
    sw = sf.SegmentWriter()
    t = {tag: w.payload() for tag, w in sw.writers.items()}
    strings = bytearray(b"FFMA\0instruction.service\0op_axis\0"
                        b"ready_warps_smsp\0")
    # AXIS rows: 0 = ready_warps_smsp, 1 = op_axis
    fma, svc, ax, axis_name = 0, 5, 22, 40
    t["META"] = sf.META.pack(
        tree_digest=b"\1" * 32, program_digest=b"\2" * 32,
        converter_digest=b"\3" * 32, cluster_id=0, cta_count=1,
        warp_count=2, instruction_count=4, flags=0)
    t["CTAS"] = sf.CTAS.pack_row(
        (0, 0, 0, 0, 0, 1, 1, 1, 0xFFFFFFFF if unplaced else 0,
         0xFFFFFFFF, 0, 1, 0, 0, 0, 2))
    t["WARP"] = (sf.WARP.pack_row((0, 0, 0, 0, 0, 2)) +
                 sf.WARP.pack_row((0, 1, 1, 1, 2, 2)))
    t["TMPL"] = sf.TMPL.pack_row(
        (fma, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 1, 0, 1, 0xFF, 0xFF, 0, 0,
         0, 0, 0, 0))
    t["INST"] = b"".join(sf.INST.pack_row((0, 0xFFFFFFFF, 0xFFFFFFFF, i))
                         for i in range(4))
    t["REGS"] = (sf.REGS.pack_row((1,)) + sf.REGS.pack_row((0,)) +
                 sf.REGS.pack_row((2,)))
    t["PROG"] = sf.PROG.pack_row((svc, 1, 1, 0, 0, 0xF, fma, 0))
    # nodes: 0 root -> child 1 (guard ready_warps_smsp==0 -> leaf2) /
    # child 2 (range 1..max -> leaf3)
    t["NODE"] = (sf.NODE.pack_row((0xFFFFFFFF, 1, 2, 0, 0, 0xFFFFFFFF)) +
                 sf.NODE.pack_row((0, 0, 0, 0, 1, 0)) +
                 sf.NODE.pack_row((0, 0, 0, 1, 1, 1)))
    t["GRD "] = (sf.GRD_.pack_row((0, 0, b"\0" * 3, 0, 0)) +
                 sf.GRD_.pack_row((0, 1, b"\0" * 3, 1, 0xFFFFFFFFFFFFFFFF)))
    # leaves: leaf0 parm rows 0-5 (all models), leaf1 constant 9 (row
    # 6), leaf2 schema-21 shared-service units (rows 7-8), leaf3 the
    # fail-closed discrete field (row 9) reached via node 7 only
    t["LEAF"] = (sf.LEAF.pack_row((0, 0, b"\0" * 3)) +
                 sf.LEAF.pack_row((6, 0, b"\0" * 3)) +
                 sf.LEAF.pack_row((7, 0, b"\0" * 3)) +
                 sf.LEAF.pack_row((9, 0, b"\0" * 3)))
    # PARM: const(=3), discrete(axis0 pts), linear(points), affine
    t["NODE"] += sf.NODE.pack_row((3, 0, 0, 0, 0, 2))
    # NA-check root: node 4 -> child 5 (ready_warps_smsp==7 -> leaf1) /
    # child 6 (==8 -> leaf1); a present-but-unmatched axis value hits
    # the not_applicable path, an absent one still missing_fact
    t["NODE"] += (sf.NODE.pack_row((0xFFFFFFFF, 5, 2, 0, 0, 0xFFFFFFFF)) +
                  sf.NODE.pack_row((4, 0, 0, 2, 1, 1)) +
                  sf.NODE.pack_row((4, 0, 0, 3, 1, 1)))
    t["GRD "] += (sf.GRD_.pack_row((0, 0, b"\0" * 3, 7, 0)) +
                  sf.GRD_.pack_row((0, 0, b"\0" * 3, 8, 0)))
    # node 7: unguarded sibling leaf for the fail-closed discrete
    # (leaf index 3, PARM row 9); resolve_node(7) reaches it directly
    t["NODE"] += sf.NODE.pack_row((0, 0, 0, 0, 0, 3))
    t["PARM"] = (sf.PARM.pack_row((1, 1, 0, b"\0" * 3, 0xFFFFFFFF,
                                  3.0, 0.0, 0.0, 0, 0)) +
                 sf.PARM.pack_row((1, 2, 1, b"\0" * 3, 0,
                                  99.0, 1.0, 0.0, 0, 2)) +
                 sf.PARM.pack_row((1, 7, 2, b"\0" * 3, 0,
                                  0.0, 0.0, 0.0, 2, 3)) +
                 sf.PARM.pack_row((1, 8, 3, b"\0" * 3, 0,
                                  0.0, 2.0, 1.0, 0, 0)) +
                 # two-axis discrete over (op_axis, ready_warps_smsp):
                 # (3, 5) -> 77, default 11; PTS rows 5..7
                 sf.PARM.pack_row((1, 5, 1, b"\0" * 3, 1,
                                  11.0, 2.0, 0.0, 5, 3)) +
                 # single-point linear on axis 0: PTS row 8
                 # (512 -> 33.6); engine clamps -> constant eval.  A
                 # leaf's PARM rows are the contiguous run from its
                 # parameter index, so this row must stay inside leaf0
                 # (rows 0-5) and before leaf1 / leaf2 (rows 6 / 7-8)
                 sf.PARM.pack_row((1, 9, 2, b"\0" * 3, 0,
                                  0.0, 0.0, 0.0, 8, 1)) +
                 sf.PARM.pack_row((1, 1, 0, b"\0" * 3, 0xFFFFFFFF,
                                  9.0, 0.0, 0.0, 0, 0)) +
                 sf.PARM.pack_row((21, 0, 0, b"\0" * 3, 0xFFFFFFFF,
                                  7602.0, 0.0, 0.0, 0, 0)) +
                 sf.PARM.pack_row((21, 2, 0, b"\0" * 3, 0xFFFFFFFF,
                                  1000.0, 0.0, 0.0, 0, 0)) +
                 # leaf3, PARM row 9: fail-closed two-axis discrete over
                 # (op_axis, ready_warps_smsp), same point (3, 5) -> 77
                 # but intercept = 1: a fact miss throws
                 # unsupported:engine:discrete_miss instead of
                 # returning the default
                 sf.PARM.pack_row((1, 10, 1, b"\0" * 3, 1,
                                  0.0, 2.0, 1.0, 9, 3)))
    # PTS: discrete point (axis 0 == 5 -> 42) at row0 (axis row value
    # = axis id + 1); linear points rows 2-4; two-axis discrete point
    # (op_axis 3, ready_warps_smsp 5) -> 77 at rows 5-7
    t["PTS "] = (sf.PTS_.pack_row((5.0, 1.0)) +
                 sf.PTS_.pack_row((42.0, 0.0)) +
                 sf.PTS_.pack_row((0.0, 10.0)) +
                 sf.PTS_.pack_row((4.0, 18.0)) +
                 sf.PTS_.pack_row((8.0, 26.0)) +
                 sf.PTS_.pack_row((3.0, 2.0)) +
                 sf.PTS_.pack_row((5.0, 1.0)) +
                 sf.PTS_.pack_row((77.0, 0.0)) +
                 sf.PTS_.pack_row((512.0, 33.6)) +
                 # fail-closed discrete point rows 9-11: same (3, 5)
                 # point so a matching fact resolves and a mismatch
                 # throws
                 sf.PTS_.pack_row((3.0, 2.0)) +
                 sf.PTS_.pack_row((5.0, 1.0)) +
                 sf.PTS_.pack_row((77.0, 0.0)))
    t["STRS"] = bytes(strings)
    t["AXIS"] = sf.AXIS.pack_row((axis_name,)) + sf.AXIS.pack_row((ax,))
    t["USES"] = (sf.USES.pack_row((0, 15, b"\0" * 3, 0, 1, 0xF, 0, 0)) +
                 sf.USES.pack_row((1, 0, b"\0" * 3, 0, 1, 0xF, 0, 0)) +
                 sf.USES.pack_row((2, 0, b"\0" * 3, 0, 1, 0xF, 0, 0)) +
                 sf.USES.pack_row((1, 0, b"\0" * 3, 0, 1, 0xF, 0, 0)) +
                 sf.USES.pack_row((3, 9, b"\0" * 3, 0, 1, 0xF, 0, 0)))
    return sf.write_segment(t)


if __name__ == "__main__":
    main()
