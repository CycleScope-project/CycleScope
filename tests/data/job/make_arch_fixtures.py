#!/usr/bin/env python3
"""Synthetic arch-module (Hopper) fixtures for 3-4 batch 1."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fixture_lib import (NONE, AXID, Seg, container_multi, CTA_WAIT,
                         ALSO_ARRIVES, TK_CTA_REL)

K_WGMMA = 22; K_WG_ARRIVE = 19; K_WG_WAIT = 20
K_TMA_LOAD = 10; K_DESC_VIS = 38; K_ACP_ISSUE = 14
F_OPENER = 0x8; F_COMMIT = 0x10
SVC4 = {1: 1.0, 2: 1.0, 7: 1.0}   # fast filler latency 1


def new_seg(sm=0):
    return Seg(1, 8)


def h1():
    s = new_seg()
    fill = s.template("FILL", svc=SVC4)
    wg = s.template("WGMMA",
                    timing=((64, {0: 50.0, 1: 1.0}),), svc=SVC4)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    wt = s.semantic(K_WG_WAIT, 0, (0, 0))
    for wi in range(4):
        first = len(s.inst)
        for _ in range(3):
            s.instance(fill)
        s.instance(wg, wgm)
        s.instance(fill, wt)
        s.instance(fill)
        s.warps.append((0, wi, wi, wi % 4, first, 6))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h3():
    # two warpgroups on one SM; the tensor.pipe beat (kind 63, 10)
    # serializes the two ops' starts, tensor.completion (kind 64) 50
    s = new_seg()
    fill = s.template("FILL", svc=SVC4)
    wg = s.template("WGMMA",
                    timing=((64, {0: 50.0, 1: 1.0}),
                            (63, {0: 10.0, 1: 1.0})),
                    svc=SVC4)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    wt = s.semantic(K_WG_WAIT, 0, (0, 0))
    for wi in range(8):
        first = len(s.inst)
        s.instance(wg, wgm)
        s.instance(fill, wt)
        s.instance(fill)
        s.warps.append((0, wi, wi, wi % 4, first, 3))
    s.cta(0, 0, 0, 8, 0)
    return s.build()


def h4():
    # steal: warpgroup on sm0 blocks another warp's ALU issue
    s = new_seg()
    fill = s.template("FILL", svc=SVC4)
    wg = s.template("WGMMA",
                    timing=((64, {0: 50.0, 1: 1.0}),
                            (44, {0: 40.0, 1: 1.0})),
                    svc=SVC4)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    wt = s.semantic(K_WG_WAIT, 0, (0, 0))
    for wi in range(5):
        first = len(s.inst)
        if wi < 4:
            s.instance(wg, wgm)
            s.instance(fill, wt)
            s.instance(fill)
            s.warps.append((0, wi, wi, wi % 4, first, 3))
        else:
            for _ in range(60):
                s.instance(fill)
            s.warps.append((0, wi, wi, wi % 4, first, 60))
    s.cta(0, 0, 0, 5, 0)
    return s.build()


def h5():
    # handoff: FFMA reading the accumulator waits for completion
    # (pipe beat 20 + result latency 7) + the kind-13 consumer leaf 9
    s = new_seg()
    fill = s.template("FILL", svc=SVC4)
    wg = s.template("WGMMA",
                    timing=((63, {0: 20.0, 1: 1.0}),
                            (64, {0: 7.0, 1: 1.0})),
                    svc=SVC4)
    ff = s.template("FFMA", timing=((13, {0: 9.0, 1: 1.0}),),
                    svc=SVC4, dst=[16], src=[128, 0])
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT,
                     (0, (128 << 32) | 4, (64 << 32) | (32 << 16) | 16))
    wt = s.semantic(K_WG_WAIT, 0, (0, 0))
    for wi in range(4):
        first = len(s.inst)
        s.instance(wg, wgm)
        s.instance(fill, wt)
        s.instance(ff)
        s.warps.append((0, wi, wi, wi % 4, first, 3))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h6():
    # ldgsts: async copy completion = issue + kind-39 leaf
    s = new_seg()
    acp = s.template("LDGSTS",
                     timing=((39, {0: 40.0, 1: 1.0}),), svc=SVC4,
                     dst=[0], src=[1])
    accm = s.template("ACP.COMMIT", svc=SVC4)
    acw = s.template("ACP.WAIT", svc=SVC4)
    fill = s.template("FILL", svc=SVC4)
    si = s.semantic(K_ACP_ISSUE, 0, (0, 0, 0, 0))
    sc = s.semantic(15, 0, (0, 0, NONE))
    sw = s.semantic(16, 0, (0, 0))
    s.macc.append((0, 4, 0, 0, 0, 0, 0, 0))
    for sec in range(4):
        s.msec.append((sec * 32, 0xFFFFFFFF, 0))
    s.inst.append((acp, si, 0, 0))
    s.inst.append((accm, sc, NONE, 1))
    s.inst.append((acw, sw, NONE, 2))
    s.inst.append((fill, NONE, NONE, 3))
    s.warps.append((0, 0, 0, 0, 0, 4))
    s.cta(0, 0, 0, 1, 0)
    return s.build()


def _sts(s, tmpl, nsec):
    """one shared-store instance: MACC shared|store + nsec sectors."""
    macc = len(s.macc)
    s.macc.append((len(s.msec), nsec, 0x41, 0, 0, 0, 0, 0))
    for i in range(nsec):
        s.msec.append((i * 32, 0xFFFFFFFF, 0))
    s.inst.append((tmpl, NONE, macc, len(s.inst)))


def h7():
    # proxy_fence: 4 synchronous STS x 32 sectors (1024 B) then fence
    # blocks until 4096/128 + 10
    s = new_seg()
    sts = s.template("STS", svc=SVC4)
    fen = s.template("FENCE.PROXY", svc=SVC4)
    fill = s.template("FILL", svc=SVC4)
    sf = s.semantic(K_DESC_VIS, 0, (5, 0, 0))
    first = len(s.inst)
    for _ in range(4):
        _sts(s, sts, 32)
    s.inst.append((fen, sf, NONE, len(s.inst)))
    s.inst.append((fill, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, first, 6))
    s.cta(0, 0, 0, 1, 0)
    # code map for the stall-ledger pc assertion (fence at 0x1040)
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    # schema-29 leaf: latency 10, drains 128 / 51.2 B/c, cap 19405
    s.uses.append((s.leaf_node(
        29, {0: 1.0, 1: 10.0, 2: 128.0, 3: 51.2, 4: 19405.0}),
        0, b"\0" * 3, 0, 29, 0, 0, 0))
    return s.build()


def h8():
    # UTMALDG does not enter the FIFO: fence blocks only latency
    s = new_seg()
    tma = s.template("TMA.LOAD", svc=SVC4)
    fen = s.template("FENCE.PROXY", svc=SVC4)
    fill = s.template("FILL", svc=SVC4)
    si = s.semantic(1, 0, (0, 1, 0, 0))   # mbarrier_init obj 0
    st = s.semantic(K_TMA_LOAD, 0, (0, 0, 0, 4096))
    sf = s.semantic(K_DESC_VIS, 0, (5, 0, 0))
    first = len(s.inst)
    s.instance(fill, si)
    s.instance(tma, st)
    s.instance(fen, sf)
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 4))
    s.cta(0, 0, 0, 1, 0)
    s.uses.append((s.leaf_node(
        29, {0: 1.0, 1: 10.0, 2: 128.0, 3: 51.2, 4: 19405.0}),
        0, b"\0" * 3, 0, 29, 0, 0, 0))
    return s.build()


def h9():
    # backpressure: capacity 1500 B; 3rd STS waits for drain
    s = new_seg()
    sts = s.template("STS", svc=SVC4)
    fill = s.template("FILL", svc=SVC4)
    first = len(s.inst)
    for _ in range(3):
        _sts(s, sts, 32)
    s.inst.append((fill, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, first, 4))
    s.cta(0, 0, 0, 1, 0)
    s.uses.append((s.leaf_node(
        29, {0: 1.0, 1: 10.0, 2: 128.0, 3: 51.2, 4: 1500.0}),
        0, b"\0" * 3, 0, 29, 0, 0, 0))
    return s.build()


def h2():
    # wait N=1 across two committed groups
    s = new_seg()
    fill = s.template("FILL", svc=SVC4)
    wg = s.template("WGMMA",
                    timing=((64, {0: 50.0, 1: 1.0}),), svc=SVC4)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    wgm2 = s.semantic(K_WGMMA, F_COMMIT, (0, 0, 0))
    wt = s.semantic(K_WG_WAIT, 0, (0, 1))
    for wi in range(4):
        first = len(s.inst)
        s.instance(wg, wgm)
        s.instance(wg, wgm2)
        s.instance(fill, wt)
        s.instance(fill)
        s.warps.append((0, wi, wi, wi % 4, first, 4))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


# ---- Blackwell (b300) fixtures ----
K_UMMA = 23; K_UMMA_COMMIT = 24
K_TMEM_ALLOC = 25; K_TMEM_DEALLOC = 26; K_TMEM_WAIT = 31
K_CLC = 32; K_STAS = 33
K_MBI = 1; K_MBW = 4; K_MBETX = 3


def b1():
    # 3 UMMA on one SM (tensor.pipe kind-63 leaf 20) -> starts 0/20/40;
    # UTCBAR kind-51 leaf 5 -> arrive at 65, waiter released 65
    s = Seg(1, 8)
    fill = s.template("FILL", svc=SVC4)
    um = s.template("UMMA", timing=((63, {0: 20.0, 1: 1.0}),),
                    svc=SVC4)
    uc = s.template("UTCBAR", timing=((51, {0: 5.0, 1: 1.0}),),
                    svc=SVC4)
    si = s.semantic(K_MBI, 0, (0, 1, 0, 0))      # obj 0 expected 1
    su = s.semantic(K_UMMA, 0, (0, 1, 0, 0))     # cta_group 1
    sc = s.semantic(K_UMMA_COMMIT, 0, (0, 1, 0, 0))  # object 0
    sw = s.semantic(K_MBW, 0, (0, 0, 0, 0))      # wait obj 0 parity 0
    first = len(s.inst)
    s.instance(fill, si)
    for _ in range(3):
        s.instance(um, su)
    s.instance(uc, sc)
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 6))
    first2 = len(s.inst)
    s.instance(fill, sw)
    s.instance(fill)
    s.warps.append((0, 1, 1, 1, first2, 2))
    s.cta(0, 0, 0, 2, 0)
    return s.build()


def b2():
    # two CTAs each alloc 512 cols; second blocks until dealloc
    s = Seg(1, 8)
    fill = s.template("FILL", svc=SVC4)
    al = s.template("TMEM.ALLOC", svc=SVC4)
    de = s.template("TMEM.DEALLOC", svc=SVC4)
    sa = s.semantic(K_TMEM_ALLOC, 0, (512, 0, 0))
    sd = s.semantic(K_TMEM_DEALLOC, 0, (512, 0, 0))
    first = len(s.inst)
    s.instance(al, sa)
    for _ in range(10):
        s.instance(fill)
    s.instance(de, sd)
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 12))
    first2 = len(s.inst)
    s.instance(al, sa)
    s.instance(fill)
    s.warps.append((1, 0, 4, 1, first2, 2))
    s.cta(0, 0, 0, 1, 0)
    s.cta(1, 0, 0, 1, 0)
    s.uses.append((s.leaf_node(31, {0: 512.0}), 0, b"\0" * 3,
                   0, 31, 0, 0, 0))
    return s.build()


def b2b():
    # the non-allocating CTA of a cta_group::2 pair deallocs on an SM
    # that never saw an alloc (frees 0); a later CTA's 512-column alloc
    # on that SM must still succeed (budget, not 0)
    s = Seg(1, 8)
    fill = s.template("FILL", svc=SVC4)
    al = s.template("TMEM.ALLOC", svc=SVC4)
    de = s.template("TMEM.DEALLOC", svc=SVC4)
    sa = s.semantic(K_TMEM_ALLOC, 0, (512, 0, 0))
    sd = s.semantic(K_TMEM_DEALLOC, 0, (0xFFFFFFFF, 0, 0))
    first = len(s.inst)
    s.instance(de, sd)
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 2))
    first2 = len(s.inst)
    for _ in range(20):
        s.instance(fill)
    s.instance(al, sa)
    s.instance(fill)
    s.warps.append((1, 0, 4, 1, first2, 22))
    s.cta(0, 0, 0, 1, 0)
    s.cta(1, 0, 0, 1, 1)
    s.uses.append((s.leaf_node(31, {0: 512.0}), 0, b"\0" * 3,
                   0, 31, 0, 0, 0))
    return s.build()


def b3():
    # CLC: expect_tx 16 + clc_query -> completes 16 B at +56 leaf
    s = Seg(1, 8)
    fill = s.template("FILL", svc=SVC4)
    cl = s.template("CLC.Q", timing=((56, {0: 8.0, 1: 1.0}),),
                    svc=SVC4)
    si = s.semantic(K_MBI, 0, (0, 1, 0, 0))
    sx = s.semantic(K_MBETX, 0, (0, 0, 16, 0))  # expect 16 B
    sc = s.semantic(K_CLC, 0, (0, 0, 0, 0))
    sw = s.semantic(K_MBW, 0, (0, 0, 0, 0))
    sa = s.semantic(2, 0, (0, 0, 0, 1, 0, 0))   # arrive 1
    first = len(s.inst)
    s.instance(fill, si)
    s.instance(fill, sx)
    s.instance(cl, sc)
    s.instance(fill, sa)
    s.instance(fill, sw)
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 6))
    s.cta(0, 0, 0, 1, 0)
    return s.build()


def b4():
    # STAS: complete_tx(bytes) on object via destination_rank
    s = Seg(1, 4)
    fill = s.template("FILL", svc=SVC4)
    st = s.template("STAS", timing=((21, {0: 6.0, 1: 1.0}),),
                    svc=SVC4)
    si = s.semantic(K_MBI, 0, (0, 1, 0, 0))
    sx = s.semantic(K_MBETX, 0, (0, 0, 32, 0))
    ss = s.semantic(K_STAS, 0, (0, 32, 0))      # obj 0, 32 B, rank 0
    sw = s.semantic(K_MBW, 0, (0, 0, 0, 0))
    sa = s.semantic(2, 0, (0, 0, 0, 1, 0, 0))
    first = len(s.inst)
    s.instance(fill, si)
    s.instance(fill, sx)
    s.instance(fill, sa)
    s.instance(st, ss)
    s.instance(fill, sw)
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 6))
    s.cta(0, 0, 0, 1, 0)
    return s.build()


def h10():
    # fp8_fmnmx handoff guard: QGMMA->FMNMX pays kind-29 (112),
    # QGMMA->FADD and HGMMA->FMNMX do not.  Same kind-13 leaf 9 for all.
    s = new_seg()
    s.const_program("hopper.wgmma.fp8_fmnmx_handoff.consumer",
                    1, 4, 29, 112.0)
    fill = s.template("FILL", svc=SVC4)
    for name,cons in (("QGMMA","FMNMX"),("QGMMA","FADD"),("HGMMA","FMNMX")):
        wg = s.template(name,
                        timing=((63, {0: 20.0, 1: 1.0}),
                                (64, {0: 7.0, 1: 1.0})),
                        svc=SVC4)
        fc = s.template(cons, timing=((13, {0: 9.0, 1: 1.0}),),
                        svc=SVC4, dst=[16], src=[128, 0])
        wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT,
                         (0, (128 << 32) | 4, (64 << 32) | (32 << 16) | 16))
        wt = s.semantic(K_WG_WAIT, 0, (0, 0))
        first = len(s.inst)
        s.instance(wg, wgm)
        s.instance(fill, wt)
        s.instance(fc)
        s.warps.append((0, len(s.warps), len(s.warps), len(s.warps) % 4,
                        first, 3))
    s.cta(0, 0, 0, 3, 0)
    return s.build()





# ---- wgmma register-file issue steal (protocol-v3 C3 / C4) ----------
# kind-44 row on every WGMMA template below: cycles=2 (per-warpgroup
# issue floor while in flight), enabled=1, field 5
# coresident_dependent_half_cycles=1.  The window streams are
# per-template register layouts cycling over 8 register ids so that
# successive instructions do not depend on each other.

_STEAL = (44, {0: 2.0, 1: 1.0, 5: 1.0})


def _stream_templates(s, reads, reuse_two=False, svc=None):
    """8 round-robin stream templates: writes d[i], reads given
    sources; each instance of a template depends on the previous
    one through d[i], but 8 cycles of separation makes the
    dependency always satisfied."""
    ts = []
    for i in range(8):
        dst = [32 + i]
        if reads == 1:
            src = [32 + i]
        else:
            src = [32 + i, 64 + i, 96 + i]
            if reuse_two:
                src = [(32 + i, 1), (64 + i, 1), 96 + i]
        ts.append(s.template("FADD" if reads == 1 else "FFMA",
                             svc=svc or SVC4, dst=dst, src=src))
    return ts


def _chain_template(s):
    # dependent chain: dst feeds the next instruction, result
    # latency 4 -> every issue is triggered by reg_ready == now
    return s.template("FFMA", svc={1: 4.0, 2: 4.0, 7: 4.0},
                      dst=[16], src=[16, 17, 18])


def _issuer(s, sgm, complete_lat=500.0, svc=None):
    """wg0 issues one wgmma completing complete_lat after launch."""
    wg = s.template("WGMMA",
                    timing=((64, {0: complete_lat, 1: 1.0}), _STEAL),
                    svc=SVC4)
    first = len(s.inst)
    s.instance(wg, sgm)
    return first


def h11():
    # (a) issuing warpgroup: 1-read stream -> issue floor 2
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    ts = _stream_templates(s, 1)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 500.0, 1: 1.0}), _STEAL),
                        svc=SVC4)
        s.instance(wg, wgm)
        for k in range(32):
            s.instance(ts[k % 8])
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h12():
    # (b) issuing warpgroup: 3-read FFMA stream -> 3/inst
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    fill = s.template("FILL", svc=SVC4)
    ts = _stream_templates(s, 3)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 500.0, 1: 1.0}), _STEAL),
                        svc=SVC4)
        s.instance(wg, wgm)
        for k in range(32):
            s.instance(ts[k % 8])
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h13():
    # (c) as (b) with reuse flags on two sources: still 3/inst
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    ts = _stream_templates(s, 3, reuse_two=True)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 500.0, 1: 1.0}), _STEAL),
                        svc=SVC4)
        s.instance(wg, wgm)
        for k in range(32):
            s.instance(ts[k % 8])
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h14():
    # (d) issuing warpgroup: dependent chain latency 4 -> still 4
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    ch = _chain_template(s)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 500.0, 1: 1.0}), _STEAL),
                        svc=SVC4)
        s.instance(wg, wgm)
        for _ in range(32):
            s.instance(ch)
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h15():
    # (e) op completes early (latency 40): stream falls back to 1
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    ts = _stream_templates(s, 1)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 40.0, 1: 1.0}), _STEAL),
                        svc=SVC4)
        s.instance(wg, wgm)
        for k in range(32):
            s.instance(ts[k % 8])
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h16():
    # (e2) same, 3-read FFMA stream -> 1.5 after completion
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    ts = _stream_templates(s, 3)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 40.0, 1: 1.0}), _STEAL),
                        svc=SVC4)
        s.instance(wg, wgm)
        for k in range(32):
            s.instance(ts[k % 8])
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h17():
    # (f) co-resident warpgroup's 1-read stream: unaffected -> 1
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    fill = s.template("FILL", svc=SVC4)
    ts = _stream_templates(s, 1)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    wt = s.semantic(K_WG_WAIT, 0, (0, 0))
    for wi in range(8):
        first = len(s.inst)
        if wi < 4:
            wg = s.template("WGMMA",
                            timing=((64, {0: 500.0, 1: 1.0}), _STEAL),
                            svc=SVC4)
            s.instance(wg, wgm)
            s.instance(fill, wt)
            s.warps.append((0, wi, wi, wi % 4, first, 2))
        else:
            for k in range(32):
                s.instance(ts[k % 8])
            s.warps.append((0, wi, wi, wi % 4, first, 32))
    s.cta(0, 0, 0, 8, 0)
    return s.build()


def h18():
    # (g) co-resident warpgroup's dependent chain (latency 4):
    # +0.5 per issue while the op is in flight -> ~4.5
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    fill = s.template("FILL", svc=SVC4)
    ch = _chain_template(s)
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    wt = s.semantic(K_WG_WAIT, 0, (0, 0))
    for wi in range(8):
        first = len(s.inst)
        if wi < 4:
            wg = s.template("WGMMA",
                            timing=((64, {0: 500.0, 1: 1.0}), _STEAL),
                            svc=SVC4)
            s.instance(wg, wgm)
            s.instance(fill, wt)
            s.warps.append((0, wi, wi, wi % 4, first, 2))
        else:
            for _ in range(32):
                s.instance(ch)
            s.warps.append((0, wi, wi, wi % 4, first, 32))
    s.cta(0, 0, 0, 8, 0)
    return s.build()




def _rs_steal(s):
    """kind-44 row flagged source_mode=rs via capture FACT (v5)."""
    return (44, {0: 2.0, 1: 1.0, 5: 1.0, 6: 4.0, 7: 1.0},
            [("source_mode", s.so("rs"))])


def h19():
    # (v5) issuing warpgroup: 2-read stream with reuse_carrier (bit2)
    # on one source -> reuse_flag_floor 4
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    steal = (44, {0: 2.0, 1: 1.0, 5: 1.0, 6: 4.0, 7: 1.0})
    ts = []
    for i in range(8):
        ts.append(s.template("FMUL", svc=SVC4, dst=[32 + i],
                             src=[(32 + i, 4), 64 + i]))
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 500.0, 1: 1.0}), steal),
                        svc=SVC4)
        s.instance(wg, wgm)
        for k in range(32):
            s.instance(ts[k % 8])
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h20():
    # (v5) same shape, no reuse flag -> base floor 2
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    steal = (44, {0: 2.0, 1: 1.0, 5: 1.0, 6: 4.0, 7: 1.0})
    ts = []
    for i in range(8):
        ts.append(s.template("FMUL", svc=SVC4, dst=[32 + i],
                             src=[32 + i, 64 + i]))
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 500.0, 1: 1.0}), steal),
                        svc=SVC4)
        s.instance(wg, wgm)
        for k in range(32):
            s.instance(ts[k % 8])
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def h21():
    # (v5) rs op in flight + unflagged stream -> ~2.5 (alt +0/+1)
    s = new_seg()
    s.operand_read(1.0, 2.0, 2.0)
    ts = []
    for i in range(8):
        ts.append(s.template("FMUL", svc=SVC4, dst=[32 + i],
                             src=[32 + i, 64 + i]))
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        wg = s.template("WGMMA",
                        timing=((64, {0: 500.0, 1: 1.0}), _rs_steal(s)),
                        svc=SVC4)
        s.instance(wg, wgm)
        for k in range(32):
            s.instance(ts[k % 8])
        s.warps.append((0, wi, wi, wi % 4, first, 33))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def _pending_wgmma(s, fields44, beat=50.0):
    # kind-63 = tensor.pipe beat, kind-64 = tensor.completion,
    # kind-44 = register_file_issue_steal row carrying
    # pending_commit_groups_limit (field 11) /
    # launch_hold_blocks_memory_queue (field 12) /
    # memory_queue_hold_above_pending_ops (field 13); cycles (field
    # 0) absent so the v3 issue-floor part contributes nothing
    return s.template("WGMMA",
                      timing=((64, {0: 20.0, 1: 1.0}),
                              (63, {0: beat, 1: 1.0}),
                              (44, fields44)),
                      svc=SVC4)


def _owner_warps(s, wg, wgm, n_ops, gap_fills=0, tail_ops=0):
    for wi in range(4):
        first = len(s.inst)
        for _ in range(n_ops):
            s.instance(wg, wgm)
        if gap_fills:
            fill = s.template("FILL", svc=SVC4)
            for _ in range(gap_fills):
                s.instance(fill)
            for _ in range(tail_ops):
                s.instance(wg, wgm)
        s.warps.append((0, wi, wi, wi % 4, first,
                        n_ops + gap_fills + tail_ops))


def _queue_warp(s, mq_class, delay=38.0, count=8):
    # a second warpgroup's warp on SMSP 0: a single latency-`delay`
    # producer parks the warp's first queue-class evaluation inside
    # the owner's held window without consuming SMSP issue slots,
    # then `count` queue-class instances.  issue_rules v1: the
    # dependency rides write barrier 1 (R1 dropped register
    # tracking); the NOP puts the first evaluation at delay + 2 so
    # R3 sees the raise
    delay_t = s.template("FILL", svc={1: delay, 2: 1.0, 7: 1.0},
                         dst=[9], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    mqt = s.template(mq_class, svc=SVC4, src=[9],
                     scoreboard_wait_mask=0x2)
    first = len(s.inst)
    s.instance(delay_t)
    s.instance(nop)
    for _ in range(count):
        s.instance(mqt)
    s.warps.append((0, 4, 4, 0, first, 2 + count))


def w1():
    # limit 6, beat 50, 10 back-to-back ops, one op per commit group:
    # ops 0-5 launch ~1/cycle; op k >= 6 launches at the pipe end of
    # op k-6 (six pending closed groups)
    s = new_seg()
    wg = _pending_wgmma(s, {1: 1.0, 11: 6.0, 12: 1.0, 13: 6.0})
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    _owner_warps(s, wg, wgm, 10)
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def w1b():
    # same shape, fields 11-13 absent -> admission rule off
    s = new_seg()
    wg = _pending_wgmma(s, {1: 1.0})
    wgm = s.semantic(K_WGMMA, F_OPENER | F_COMMIT, (0, 0, 0))
    _owner_warps(s, wg, wgm, 10)
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def w1c():
    # 12 ops in commit groups of 3 (opener on op 0, commit on ops
    # 2/5/8/11): four closed groups total < limit 6 -> no launch held
    s = new_seg()
    wg = _pending_wgmma(s, {1: 1.0, 11: 6.0, 12: 1.0, 13: 6.0})
    wgm_open = s.semantic(K_WGMMA, F_OPENER, (0, 0, 0))
    wgm_plain = s.semantic(K_WGMMA, 0, (0, 0, 0))
    wgm_commit = s.semantic(K_WGMMA, F_COMMIT, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        for i in range(12):
            if i == 0:
                s.instance(wg, wgm_open)
            elif i % 3 == 2:
                s.instance(wg, wgm_commit)
            else:
                s.instance(wg, wgm_plain)
        s.warps.append((0, wi, wi, wi % 4, first, 12))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def _w2(mq_class, fields44):
    # owner: 12 ops of ONE still-open commit group (opener on op 0,
    # no commit): no launch is ever held, but 12 > 6 ops are
    # outstanding until the pipe ends drain
    s = new_seg()
    wg = _pending_wgmma(s, fields44)
    wgm_open = s.semantic(K_WGMMA, F_OPENER, (0, 0, 0))
    wgm_plain = s.semantic(K_WGMMA, 0, (0, 0, 0))
    for wi in range(4):
        first = len(s.inst)
        for i in range(12):
            s.instance(wg, wgm_open if i == 0 else wgm_plain)
        s.warps.append((0, wi, wi, wi % 4, first, 12))
    _queue_warp(s, mq_class)
    s.cta(0, 0, 0, 5, 0)
    return s.build()


def w2m():
    return _w2("MUFU.EX2", {1: 1.0, 11: 6.0, 12: 1.0, 13: 6.0})


def w2f():
    return _w2("FFMA", {1: 1.0, 11: 6.0, 12: 1.0, 13: 6.0})


def w2o():
    return _w2("MUFU.EX2", {1: 1.0, 11: 6.0, 12: 0.0, 13: 6.0})


def w3():
    # the owner's OWN MUFU after its 12 one-group ops: the memory-
    # queue hold applies to the owners' warps too (run 5)
    s = new_seg()
    wg = _pending_wgmma(s, {1: 1.0, 11: 6.0, 12: 1.0, 13: 6.0})
    wgm_open = s.semantic(K_WGMMA, F_OPENER, (0, 0, 0))
    wgm_plain = s.semantic(K_WGMMA, 0, (0, 0, 0))
    mqt = s.template("MUFU.EX2", svc=SVC4)
    for wi in range(4):
        first = len(s.inst)
        for i in range(12):
            s.instance(wg, wgm_open if i == 0 else wgm_plain)
        n_tail = 8 if wi == 0 else 0
        for _ in range(n_tail):
            s.instance(mqt)
        s.warps.append((0, wi, wi, wi % 4, first, 12 + n_tail))
    s.cta(0, 0, 0, 4, 0)
    return s.build()


def _copy_pipe(ops, service=40.0):
    s = Seg(1, 2)
    fill = s.template("FILL", svc=SVC4)
    cp = s.template("UTCCP", timing=((55, {0: service, 1: 1.0}),), svc=SVC4)
    mma = s.template("UMMA", timing=((63, {0: 20.0, 1: 1.0}),
                                      (64, {0: 7.0, 1: 1.0})), svc=SVC4)
    commit = s.template("UTCBAR", timing=((51, {0: 5.0, 1: 1.0}),), svc=SVC4)
    init = s.semantic(K_MBI, 0, (0, 1, 0, 0))
    copy = s.semantic(30, 0, ((4 << 16) | 128, 1))
    multiply = s.semantic(K_UMMA, 0, (0, 1, 0, 0))
    done = s.semantic(K_UMMA_COMMIT, 0, (0, 1, 0, 0))
    wait = s.semantic(K_MBW, 0, (0, 0, 0, 0))
    instructions = [(fill, init)] + [(cp, copy) if op == "copy" else (mma, multiply) for op in ops]
    s.warp(0, 0, 0, instructions + [(commit, done), (fill, None)])
    s.warp(0, 1, 1, [(fill, wait), (fill, None)])
    s.cta(0, 0, 0, 2, 0)
    return s.build()

def b5():
    return _copy_pipe(["copy"])

def b6():
    return _copy_pipe(["copy", "mma"])

def b7():
    return _copy_pipe(["mma", "copy"])

def b8():
    return _copy_pipe(["copy"] * 10, 2.5)

def b9():
    s = Seg(1, 3)
    fill = s.template("FILL", svc=SVC4)
    mma = s.template("UMMA", timing=((63, {0: 80.0, 1: 1.0}),
                                      (64, {0: 100.0, 1: 1.0})), svc=SVC4)
    commit = s.template("UTCBAR", timing=((51, {0: 5.0, 1: 1.0}),), svc=SVC4)
    init0 = s.semantic(K_MBI, 0, (0, 1, 0, 0))
    init1 = s.semantic(K_MBI, 0, (1, 1, 0, 0))
    multiply = s.semantic(K_UMMA, 0, (0, 1, 0, 0))
    done0 = s.semantic(K_UMMA_COMMIT, 0, (0, 1, 0, 0))
    done1 = s.semantic(K_UMMA_COMMIT, 0, (1, 1, 0, 0))
    wait0 = s.semantic(K_MBW, 0, (0, 0, 0, 0))
    wait1 = s.semantic(K_MBW, 0, (1, 0, 0, 0))
    s.warp(0, 0, 0, [(fill, init0)] + [(fill, None)] * 4 + [(commit, done0), (fill, None)])
    s.warp(0, 1, 1, [(fill, init1), (mma, multiply), (commit, done1), (fill, wait1), (fill, None)])
    s.warp(0, 2, 2, [(fill, wait0), (fill, None)])
    s.cta(0, 0, 0, 3, 0)
    return s.build()

def _operand_path(mma_fields, beat, copies=1, tensor_k=64,
                  cta_group=0, is_pair_leader=0):
    # UMMA beat carries operand_read_bytes_per_cycle=128 (field 9) and
    # a tensor_k capture fact; the MMA->copy->MMA sequence exposes the
    # operand-path frontier through the second MMA's start
    s = Seg(1, 2)
    fill = s.template("FILL", svc=SVC4)
    facts = () if tensor_k is None else (("tensor_k", tensor_k),)
    mma = s.template("UMMA", timing=((63, {0: beat, 1: 1.0, 9: 128.0}, facts),
                                      (64, {0: 7.0, 1: 1.0})), svc=SVC4)
    cp = s.template("UTCCP", timing=((55, {0: 8.5, 1: 1.0}),), svc=SVC4)
    commit = s.template("UTCBAR", timing=((51, {0: 5.0, 1: 1.0}),), svc=SVC4)
    init = s.semantic(K_MBI, 0, (0, 1, 0, 0))
    multiply = s.semantic(K_UMMA, 0, mma_fields)
    copy = s.semantic(30, 0, ((4 << 16) | 128, 1))
    done = s.semantic(K_UMMA_COMMIT, 0, (0, 1, 0, 0))
    wait = s.semantic(K_MBW, 0, (0, 0, 0, 0))
    s.warp(0, 0, 0, [(fill, init), (mma, multiply)] +
           [(cp, copy)] * copies + [(mma, multiply), (commit, done),
                                    (fill, None)])
    s.warp(0, 1, 1, [(fill, wait), (fill, None)])
    s.cta(0, 0, 0, 2, 0, cta_group, is_pair_leader)
    return s.build()

def b10():
    # mxf4nvf4 (4b) m128 n128 k64, A in tmem -> only B's 4096 B cross
    # the path -> operand_read 32 of a 64 beat: the copy hides in the
    # slack and the second MMA still starts at the beat frontier
    return _operand_path((6, 1, 2, (128 << 32) | 128, 0), 64.0)

def b11():
    # same shape with A in shared memory -> 8192 B -> operand_read 64
    # == beat: the old serial rule (copy done pushes the second MMA)
    return _operand_path((6, 1, 1, (128 << 32) | 128, 0), 64.0)

def b12():
    # cta_group 2, m256 n256 k64, A smem -> (128+128)*64*4/8 = 8192 B
    # -> operand_read 64 of a 128 beat
    return _operand_path((6, 2, 1, (256 << 32) | 256, 0), 128.0,
                         cta_group=2, is_pair_leader=1)

def b13():
    # the leaf rate is present but the template carries no tensor_k
    # fact -> the engine must refuse with missing_fact:axis:tensor_k
    return _operand_path((6, 1, 2, (128 << 32) | 128, 0), 64.0,
                         tensor_k=None)

def _serial_mma(beat, rate, tensor_k=64):
    # two back-to-back MMAs + commit + waiter, no copy: with a
    # fractional beat the operand path must end on the same integer
    # beat frontier the pipe uses, so a leaf with rate == a leaf
    # without the field must release the waiter identically
    s = Seg(1, 2)
    fill = s.template("FILL", svc=SVC4)
    fields = {0: beat, 1: 1.0}
    if rate is not None:
        fields[9] = rate
    facts = (("tensor_k", tensor_k),)
    mma = s.template("UMMA", timing=((63, fields, facts),
                                      (64, {0: 7.0, 1: 0.0})), svc=SVC4)
    commit = s.template("UTCBAR", timing=((51, {0: 5.0, 1: 1.0}),), svc=SVC4)
    init = s.semantic(K_MBI, 0, (0, 1, 0, 0))
    multiply = s.semantic(K_UMMA, 0, (6, 1, 1, (128 << 32) | 128, 0))
    done = s.semantic(K_UMMA_COMMIT, 0, (0, 1, 0, 0))
    wait = s.semantic(K_MBW, 0, (0, 0, 0, 0))
    s.warp(0, 0, 0, [(fill, init), (mma, multiply), (mma, multiply),
                     (commit, done), (fill, None)])
    s.warp(0, 1, 1, [(fill, wait), (fill, None)])
    s.cta(0, 0, 0, 2, 0)
    return s.build()

def b14():
    # beat 43.9 truncated to 43 on both frontiers (rate present)
    return _serial_mma(43.9, 128.0)

def b15():
    # same shape, leaf without the operand-read field (old rule)
    return _serial_mma(43.9, None)

FIXTURES = {
    "h1_collective": (h1, "h100"), "h2_wait_n1": (h2, "h100"),
    "h3_occupancy": (h3, "h100"), "h4_steal": (h4, "h100"),
    "h5_handoff": (h5, "h100"), "h6_ldgsts": (h6, "h100"),
    "h7_proxy_fence": (h7, "h100"), "h8_fence_tma": (h8, "h100"),
    "h9_backpressure": (h9, "h100"),
    "h10_handoff_guard": (h10, "h100"),
    "h11_steal_1read": (h11, "h100"), "h12_steal_3read": (h12, "h100"),
    "h13_steal_reuse": (h13, "h100"), "h14_steal_chain": (h14, "h100"),
    "h15_steal_done1": (h15, "h100"), "h16_steal_done3": (h16, "h100"),
    "h17_coresident_stream": (h17, "h100"),
    "h18_coresident_chain": (h18, "h100"),
    "h19_reuse_floor": (h19, "h100"), "h20_noflag_floor": (h20, "h100"),
    "h21_rs_floor": (h21, "h100"),
    "w1_pending_ops": (w1, "h100"),
    "w1b_pending_off": (w1b, "h100"),
    "w1c_groups3": (w1c, "h100"),
    "w2_mufu_hold": (w2m, "h100"),
    "w2_ffma_twin": (w2f, "h100"),
    "w2_mufu_nohold": (w2o, "h100"),
    "w3_owner_mufu": (w3, "h100"),
    "b1_umma": (b1, "b300"), "b2_tmem": (b2, "b300"),
    "b2b_tmem_dealloc_first": (b2b, "b300"),
    "b3_clc": (b3, "b300"), "b4_stas": (b4, "b300"),
    "b5_copy_commit": (b5, "b300"), "b6_copy_mma": (b6, "b300"),
    "b7_mma_copy": (b7, "b300"), "b8_fractional_copy": (b8, "b300"),
    "b9_foreign_commit": (b9, "b300"),
    "b10_operand_tmem": (b10, "b300"), "b11_operand_smem": (b11, "b300"),
    "b12_operand_cg2": (b12, "b300"), "b13_operand_no_k": (b13, "b300"),
    "b14_beat_fraction_rate": (b14, "b300"),
    "b15_beat_fraction_nofield": (b15, "b300"),
}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, (fn, device) in FIXTURES.items():
        segs = [fn()]
        data = container_multi(
            segs, device=device, sm_count=1,
            occupancy_blocks_per_sm=8, block_x=256, block_y=1,
            block_z=1, grid_x=len(segs), grid_y=1, grid_z=1,
            cluster_x=1)
        with open(os.path.join(out, name + ".job"), "wb") as fh:
            fh.write(data)
    print(f"wrote {len(FIXTURES)} arch fixtures to {out}")


if __name__ == "__main__":
    main()
