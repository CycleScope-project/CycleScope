#!/usr/bin/env python3
"""Generate a large synthetic job for engine-loop perf measurement.

Usage: make_perf_job.py --sms N --warps-per-sm W --insts I
                        [--memory] --out job.bin
       (PYTHONPATH=<repo>/python:<repo>/tests/data/job)

Layout: each CTA is its own segment (cluster size 1) with 8 warps
(block 256); W/8 CTAs per SM carry trace placements (CTAS.sm_id,
WARP.warp_slot = cta_in_sm*8 + warp_in_cta) so no dispatch USES is
needed.  A warp holds I instances grouped into 64-instance blocks:
63 ALU instances on 8 fixed templates (RAW chains across 2-5
instances via destination R(k%8) / sources R((k+6)%8), R((k+3)%8))
plus one cta barrier wait every 64th instance (flags bit1 = also
arrives, barrier 0, TIME kind 17 bound to
synchronization.cta.release root = 20 cycles).  --memory turns the
block's 32nd instance into an LDG with a 4-sector MACC (distinct
addresses per warp) backed by schema-3 memory_level leaves for
l1/l2/dram and the schema-20 l2_partition leaf (no shared_service ->
all bandwidth figures stay 0).
"""

from __future__ import annotations

import argparse
import struct
import sys

from fixture_lib import NONE, Seg, container_multi, CTA_WAIT, \
    ALSO_ARRIVES, TK_CTA_REL


def build(sms: int, wps: int, insts: int, memory: bool,
          sleepers: bool = False) -> bytes:
    ctas_per_sm = wps // 8
    segs = []
    alu_svc = {1: 4.0, 2: 4.0, 7: 1.0}  # result/completion 4, issue 1
    for sm in range(sms):
        for csm in range(ctas_per_sm):
            s = Seg(1, 8)
            bar = s.template(
                "BAR.SYNC",
                timing=((TK_CTA_REL, {0: 20.0, 1: 1.0}),))
            alu = [s.template(f"ALU{k}",
                              svc=alu_svc,
                              dst=[k],
                              src=[(k + 6) % 8, (k + 3) % 8])
                   for k in range(8)]
            if sleepers:
                # scan-blockers spec: 7 of 8 warps just barrier+1
                # ALU; the 8th runs a strictly serial ALU chain
                # (src = prev dst, result_latency 4 -> ~4 c/inst)
                # for ~80k cycles while the rest sit blocked
                serial = s.template("ALU.SER", svc=alu_svc,
                                    dst=[0], src=[0, 0])
                bar_wait = s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                                      fields=(0, 0, 0))
                w0 = sm * ctas_per_sm * 8 + csm * 8
                for wi in range(8):
                    first = len(s.inst)
                    if wi == 7:
                        for _ in range(insts):
                            s.instance(serial)
                        s.instance(bar, bar_wait)
                        s.instance(serial)
                        s.warps.append((0, wi, csm * 8 + wi,
                                        (csm * 8 + wi) % 4, first,
                                        insts + 2))
                    else:
                        s.instance(bar, bar_wait)
                        s.instance(serial)
                        s.warps.append((0, wi, csm * 8 + wi,
                                        (csm * 8 + wi) % 4, first, 2))
                s.cta(0, 0, sm, 8, 0)
                segs.append(s.build())
                continue
            ldg = None
            if memory:
                ldg = s.template("LDG.E.128", svc=alu_svc,
                                 dst=[7], src=[0])
                # schema-3 memory_level leaves (target_index = level)
                # l1 30c/64out/1ii/32Bpc, l2 200/256/1/64,
                # dram 400/1024/1/64; schema-20 l2_partition pool 2
                for lvl, fdict in (
                        (1, {9: 30.0, 10: 64.0, 11: 1.0, 41: 32.0}),
                        (2, {13: 200.0, 14: 256.0, 15: 1.0, 42: 64.0}),
                        (3, {17: 400.0, 18: 1024.0, 19: 1.0, 43: 64.0})):
                    s.uses.append((s.leaf_node(3, fdict), 3, b"\0" * 3,
                                   lvl, 3, 0, 0, 0))
                s.uses.append(
                    (s.leaf_node(20, {0: 2.0, 2: 0.0}), 0, b"\0" * 3,
                     0, 20, 0, 0, 0))
            # 64-instance block: 63 ALU + barrier wait; inst 32 is LDG
            insts_t: list[tuple[int, int | None, int | None]] = []
            for i in range(insts):
                k = i % 64
                if k == 63:
                    insts_t.append(
                        (bar, s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                                         fields=(0, 0, 0)), None))
                elif memory and k == 32:
                    insts_t.append((ldg, None, -1))
                else:
                    insts_t.append((alu[k % 8], None, None))
            # macc rows appended lazily per warp so sector addresses
            # stay distinct across warps
            w = sm * ctas_per_sm * 8 + csm * 8
            for wi in range(8):
                wid = w + wi
                first = len(s.inst)
                for i, (t, sm_, ma) in enumerate(insts_t):
                    if ma == -1:
                        blk = i // 64
                        sb = len(s.msec)
                        for sec in range(4):
                            s.msec.append(
                                (wid * 4096 + blk * 128 + sec * 32,
                                 0, 0))
                        s.macc.append((sb, 4, 0, i, 0, 0, 0, 0))
                        s.instance(t, sm_, len(s.macc) - 1)
                    else:
                        s.instance(t, sm_)
                s.warps.append((0, wi, csm * 8 + wi,
                                (csm * 8 + wi) % 4, first,
                                insts))
            s.cta(0, 0, sm, 8, 0)
            segs.append(s.build())
    return container_multi(
        segs, sm_count=sms, occupancy_blocks_per_sm=ctas_per_sm,
        block_x=256, block_y=1, block_z=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sms", type=int, required=True)
    ap.add_argument("--warps-per-sm", type=int, required=True)
    ap.add_argument("--insts", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--sleepers", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = build(a.sms, a.warps_per_sm, a.insts, a.memory,
                 a.sleepers)
    with open(a.out, "wb") as fh:
        fh.write(data)


if __name__ == "__main__":
    main()
