#!/usr/bin/env python3
"""Generate the resource / data-ready fixture segments for CTest.

Usage: make_resource_fixtures.py <out_dir>
       (run with PYTHONPATH=<repo>/python)

Segments (all single cluster, CTA on SM 0):
  res_f1.seg  one warp, two independent MUFU ops; service leaf
              issue_interval=8, resource=xu_smsp -> 2nd issue >= 1st+8
  res_f2.seg  two warps on SM 0, one DFMA each; service leaf
              issue_interval=16, resource=fp64_sm (scope sm)
              -> issues >= 16 apart
  res_f2b.seg same as f2 plus max_inflight=1, completion=64
              -> 2nd issue >= 1st+64
  res_f3.seg  LDG (DRAM/L2 leaf latency 300) then FFMA dependent via
              destination register, and a second pair dependent via
              scoreboard wait mask -> consumer >= LDG+300
  res_f4.seg  LDS with a template-level kind-6 TIME row (cycles 200)
              -> dependent >= max(result, finish)
  res_f6.seg  two warps on one SMSP, 16 one-source VIADD each;
              single_source_issue_interval=1 over issue_interval=2
              -> 32 unit issues span 31 cycles
  res_f6b.seg same with two distinct sources -> span 62
  res_f6c.seg same as f6 without the field -> span 62
  res_f7.seg  three warps on one SMSP, 8 VIADD each, unit interval 4,
              scheduler unit_grant_policy=0 -> warp 0 drains first
  res_f7b.seg same with unit_grant_policy=1 -> the warps alternate
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_lib import NONE, Seg  # noqa: E402

from cyclescope.io.fixed_tables import TableReader  # noqa: E402
from cyclescope.job import program_tables as pt, segment_format as sf  # noqa: E402

# instruction.service field ids (ParameterSchema 1)
F_RESULT, F_COMPLETION, F_MAX_INFLIGHT, F_ISSUE_INT = 1, 2, 5, 7
F_RESOURCE, F_GROUP_RESOURCE = 10, 11
F_SINGLE_SRC_INT = 12
# RESOURCE leaf field ids (ParameterSchema 2)
R_SCOPE, R_COPIES, R_MAX_INFLIGHT, R_INTERVAL = 0, 1, 2, 3
SCOPE_SMSP, SCOPE_SM = 0, 1
# MEMORY_LEVEL leaf field ids (ParameterSchema 3)
ML_L2_LATENCY, ML_DRAM_LATENCY = 13, 17
# TIMING_ROW leaf field ids (ParameterSchema 4)
TR_CYCLES, TR_ENABLED = 0, 1

RES_XU, RES_FP64, RES_ALU = 100, 101, 102


def res_def(s: Seg, res_id: int, scope: int, copies: int = 1,
            max_inflight: float = 0.0, interval: float = 1.0) -> None:
    """Bind a RESOURCE-schema USES row defining resource `res_id`."""
    node = s.leaf_node(2, {R_SCOPE: float(scope), R_COPIES: float(copies),
                           R_MAX_INFLIGHT: max_inflight,
                           R_INTERVAL: interval})
    s.uses.append((node, 4, b"\0" * 3, res_id, 2, 0xF, 0, 0))


def mem_level(s: Seg, fields: dict[int, float],
              target_index: int = 0) -> None:
    node = s.leaf_node(3, fields)
    s.uses.append((node, 5, b"\0" * 3, target_index, 3, 0xFFFFFFFF,
                   0, 0))


def f1() -> bytes:
    s = Seg(1, 1)
    res_def(s, RES_XU, SCOPE_SMSP)
    t = s.template("MUFU", svc={F_RESULT: 18.0, F_ISSUE_INT: 8.0,
                                F_RESOURCE: float(RES_XU)})
    s.warp(0, 0, 0, [(t, None), (t, None)])
    s.cta(0, 0, 0, 1, 0)
    return s.build()


def f2(max_inflight: float = 0.0, completion: float = 2.0) -> bytes:
    s = Seg(1, 2)
    res_def(s, RES_FP64, SCOPE_SM, max_inflight=max_inflight)
    svc = {F_RESULT: 4.0, F_COMPLETION: completion, F_ISSUE_INT: 16.0,
           F_RESOURCE: float(RES_FP64)}
    if max_inflight:
        svc[F_MAX_INFLIGHT] = max_inflight
    t = s.template("DFMA", svc=svc)
    s.warp(0, 0, 0, [(t, None)])
    s.warp(0, 1, 4, [(t, None)])
    s.cta(0, 0, 0, 2, 0)
    return s.build()


def f3() -> bytes:
    s = Seg(1, 1)
    # memory level leaf: L2 latency 300; DRAM latency also 300 so a
    # cold line returns at least 300 either way
    mem_level(s, {ML_L2_LATENCY: 300.0, ML_DRAM_LATENCY: 300.0})
    # pair 1: LDG dst reg 1 -> FFMA src reg 1 (register dependency)
    ldg = s.template("LDG", svc={F_RESULT: 1.0, F_COMPLETION: 1.0},
                     dst=[1])
    fma = s.template("FFMA", svc={F_RESULT: 4.0}, src=[1], dst=[3])
    s.msec.append((0x1000, 0xFFFFFFFF, 0xFFFFFFFF))
    s.macc.append((0, 1, 0x0, 0, 0, 0, 0, 0))  # space 0, kind 0 = load
    s.instance(ldg, macc=0)
    s.instance(fma)
    # pair 2: LDG dst reg 5 write_barrier 1 -> FFMA wait_mask bit1
    ldg2 = s.template("LDG", svc={F_RESULT: 1.0, F_COMPLETION: 1.0},
                      dst=[5])
    s.tmpl[ldg2][9] = 1  # write_barrier
    fma2 = s.template("FFMA", svc={F_RESULT: 4.0}, src=[6], dst=[7])
    s.tmpl[fma2][10] = 2  # wait_mask = barrier 1
    s.msec.append((0x2000, 0xFFFFFFFF, 0xFFFFFFFF))
    s.macc.append((1, 1, 0x0, 0, 0, 0, 0, 0))
    s.instance(ldg2, macc=1)
    s.instance(fma2)
    s.warps.append((0, 0, 0, 0, 0, 4))
    s.cta(0, 0, 0, 1, 0)
    return s.build()


def f4() -> bytes:
    s = Seg(1, 1)
    # template-level kind-6 TIME row with cycles=200
    lds = s.template("LDS", timing=((6, {TR_CYCLES: 200.0,
                                        TR_ENABLED: 1.0}),),
                     svc={F_RESULT: 1.0, F_COMPLETION: 2.0}, dst=[1])
    fma = s.template("FFMA", svc={F_RESULT: 4.0}, src=[1], dst=[2])
    s.msec.append((0x0, 0xFFFFFFFF, 0xFFFFFFFF))
    s.macc.append((0, 1, 0x1, 0, 0, 0, 0, 0))  # space 1 = shared
    s.instance(lds, macc=0)
    s.instance(fma)
    s.warps.append((0, 0, 0, 0, 0, 2))
    s.cta(0, 0, 0, 1, 0)
    return s.build()


def f5() -> bytes:
    """Same shape as F1, but the resource definition reaches the segment
    through the real program-table path: a tree["resources"] entry is
    compiled by compile_program_tables into NODE/LEAF/PARM/USES rows and
    merged into the segment like cyclescope-convert's copied tables."""
    s = Seg(1, 1)
    tree = {"axes": {}, "nodes": [], "programs": {}, "resources": [
        {"id": 0, "name": "xu_smsp", "scope": "smsp", "copy_count": 1,
         "max_inflight_per_copy": 0, "initiation_interval_cycles": 1}]}
    pl = pt.compile_program_tables(tree)
    node_base, leaf_base = len(s.node), len(s.leaf)
    parm_base, use_base = len(s.parm), len(s.uses)
    for tbl, rows, conv in (
            ("NODE", s.node, lambda r: (r["parent"], r["child_begin"],
                                        r["child_count"], r["guard_begin"],
                                        r["guard_count"],
                                        NONE if r["leaf"] == NONE else
                                        r["leaf"] + leaf_base)),
            ("LEAF", s.leaf, lambda r: (r["parameter"] + parm_base,
                                        r["status"], r["pad"])),
            ("PARM", s.parm, lambda r: tuple(r[f] for f in sf.PARM.fields)),
            ("USES", s.uses, lambda r: (r["node"] + node_base, r["target"],
                                        r["pad"], r["target_index"],
                                        r["schema_id"], r["field_mask"],
                                        r["fact_begin"], r["fact_count"]))):
        reader = TableReader(getattr(sf, tbl if tbl != "GRD" else "GRD_"),
                             pl[tbl])
        for i in range(len(reader)):
            rows.append(conv(reader.unpack(i)))
    t = s.template("MUFU", svc={F_RESULT: 18.0, F_ISSUE_INT: 8.0,
                                F_RESOURCE: 0.0})
    s.warp(0, 0, 0, [(t, None), (t, None)])
    s.cta(0, 0, 0, 1, 0)
    return s.build()


def f6(srcs: list, single: float | None) -> bytes:
    """alu_forms addendum 2 (both devices): a VIADD/LOP3/LEA-class
    instruction with at most one GPR source occupies its SMSP unit for
    single_source_issue_interval_cycles (1.0) instead of the class
    issue_interval (2.0).  Two warps (slots 0 and 4, same SMSP 0), 16 instances each;
    spacing 1 -> issue span = 31 cycles over the unit, spacing 2
    (two-source template or field absent) -> 62."""
    s = Seg(1, 2)
    res_def(s, RES_ALU, SCOPE_SMSP)
    svc = {F_RESULT: 1.0, F_COMPLETION: 1.0, F_ISSUE_INT: 2.0,
           F_RESOURCE: float(RES_ALU)}
    if single is not None:
        svc[F_SINGLE_SRC_INT] = single
    t = s.template("VIADD", svc=svc, src=srcs)
    s.warp(0, 0, 0, [(t, None)] * 16)
    s.warp(0, 1, 4, [(t, None)] * 16)  # slot 4 -> same SMSP 0
    s.cta(0, 0, 0, 2, 0)
    return s.build()


# SCHEDULER leaf field ids (ParameterSchema 7)
S_CURSOR, S_SLOTS, S_TRYNEXT, S_GRANT = 1, 2, 3, 5


def sched_policy(s: Seg, grant: float) -> None:
    """Bind a SCHEDULER-schema USES row (cursor defaults + grant)."""
    node = s.leaf_node(7, {S_CURSOR: 1.0, S_SLOTS: 1.0,
                           S_TRYNEXT: 1.0, S_GRANT: grant})
    s.uses.append((node, 0, b"\0" * 3, 0, 7, 0x2E, 0, 0))


def f7(grant: float) -> bytes:
    """unit_grant_policy (fp_stream_mix addendum: execution-unit grants
    go to the longest waiter).  One SMSP, 3 warps (slots 0/4/8), 8
    independent two-source VIADD each on a unit with issue_interval 4.
    grant=0: the non-yield cursor re-serves warp 0 at every free cycle;
    it drains all 8 (first+28) before warp 1 (first+32..60) and warp 2
    (first+64..92).  grant=1: every free cycle the unit goes to its
    longest live waiter and the last grantee loses equal-age ties, so
    the warps alternate w0@first+0, w1@+4, w2@+8, w0@+12, ...; last
    issues first+84/88/92, all within 2 intervals."""
    s = Seg(1, 3)
    res_def(s, RES_ALU, SCOPE_SMSP)
    sched_policy(s, grant)
    t = s.template("VIADD", svc={F_RESULT: 1.0, F_COMPLETION: 1.0,
                                 F_ISSUE_INT: 4.0,
                                 F_RESOURCE: float(RES_ALU)},
                   src=[1, 2])
    s.warp(0, 0, 0, [(t, None)] * 8)
    s.warp(0, 1, 4, [(t, None)] * 8)
    s.warp(0, 2, 8, [(t, None)] * 8)
    s.cta(0, 0, 0, 3, 0)
    return s.build()


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "res_f1.seg").write_bytes(f1())
    (out / "res_f2.seg").write_bytes(f2())
    (out / "res_f2b.seg").write_bytes(f2(max_inflight=1.0, completion=64.0))
    (out / "res_f3.seg").write_bytes(f3())
    (out / "res_f4.seg").write_bytes(f4())
    (out / "res_f5.seg").write_bytes(f5())
    (out / "res_f6.seg").write_bytes(f6([1], 1.0))
    (out / "res_f6b.seg").write_bytes(f6([1, 2], 1.0))
    (out / "res_f6c.seg").write_bytes(f6([1], None))
    (out / "res_f7.seg").write_bytes(f7(0.0))
    (out / "res_f7b.seg").write_bytes(f7(1.0))


if __name__ == "__main__":
    main()
