#!/usr/bin/env python3
"""Generate the synthetic sync-behavior fixtures for cpp/tests/sync_test.

Usage: make_sync_fixtures.py <out_dir>   (PYTHONPATH=<repo>/python)

Each scenario is containerised (job.bin) with a 1 GHz reference clock so
1 ns == 1 cycle.  Every template binds an ``instruction.service`` leaf
(result/completion/issue_interval = 1) through a shared USES node; there
is no scheduler USES so the default policy applies.  A warp's
``warp_slot`` equals its index in the segment so warps spread across the
four SMSPs unless a scenario pins them.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fixture_lib import *  # noqa: F401,F403

def l1_one_warp() -> bytes:
    s = Seg(1, 1)
    # latency-4 service: per-cycle issue must come from retry_at, not
    # from jumping to the completion event
    f = s.template("FILL", svc={1: 4.0, 2: 4.0, 7: 1.0})
    s.warp(0, 0, 0, fills(f, 8))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def l1_two_warps() -> bytes:
    s = Seg(1, 2)
    f = s.template("FILL", control=0x10,  # yield bit -> alternation
                   svc={1: 4.0, 2: 4.0, 7: 1.0})
    s.warp(0, 0, 0, fills(f, 4))
    s.warp(0, 1, 0, fills(f, 4))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a1_basic() -> bytes:
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 2, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    f = s.template("FILL")
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           fills(f, 5)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
              (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
              (f, None)])
    s.warp(0, 1, 1,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))]
           + fills(f, 20))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a3_parity1() -> bytes:
    s = Seg(1, 1)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 1))), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _a4(with_trywait: bool, lookahead: bool) -> bytes:
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    if with_trywait:
        s.const_program("synchronization.mbarrier.trywait_timeout",
                        1, 4, TK_TRYWAIT, 100.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    arr = s.template("MBAR.ARRIVE")
    flag = 0 if lookahead else AFTER_READY
    mask = 0 if lookahead else 0
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, flags=flag,
                            fields=(0, 0, 0, 0, 0, mask))),
             (wt, s.semantic(
                 MBAR_WAIT,
                 flags=0 if lookahead else AFTER_READY,
                 fields=(0, 0, 0, 0, 0,
                         0 if lookahead else 0xFFFFFFFF))),
             (f, None)])
    s.warp(0, 1, 1, fills(f, 200)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a6_deferred() -> bytes:
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0, 3)))]
           + fills(f, 4))
    s.warp(0, 1, 1, fills(f, 50)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a7_init_zero() -> bytes:
    s = Seg(1, 1)
    f = s.template("FILL")
    ini = s.template("MBAR.INIT")
    s.warp(0, 0, 0,
           [(ini, s.semantic(MBAR_INIT, fields=(0, 0, 0))), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a7_arrive_zero() -> bytes:
    s = Seg(1, 1)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    f = s.template("FILL")
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 0))),
             (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _cta_bar_template(s: Seg) -> int:
    return s.template("BAR.SYNC",
                      timing=((TK_CTA_REL, {0: 7.0, 1: 1.0}),))


def a8_one_gen() -> bytes:
    s = Seg(1, 4)
    f = s.template("FILL")
    bar = _cta_bar_template(s)
    for k in range(4):
        s.warp(0, k, k,
               fills(f, 10 * k)
               + [(bar, s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                                   fields=(0, 0, 0))),
                  (f, None)])
    s.cta(0, 0, 0, 4, 0)
    return container(s.build())


def a8_two_gens() -> bytes:
    s = Seg(1, 4)
    f = s.template("FILL")
    bar = _cta_bar_template(s)
    for k in range(4):
        bar_sema = s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                              fields=(0, 0, 0))
        s.warp(0, k, k,
               fills(f, 10 * k)
               + [(bar, bar_sema)] + fills(f, 5)
               + [(bar, bar_sema), (f, None)])
    s.cta(0, 0, 0, 4, 0)
    return container(s.build())


def a8_exit() -> bytes:
    s = Seg(1, 4)
    f = s.template("FILL")
    bar = _cta_bar_template(s)
    for k in range(3):
        s.warp(0, k, k,
               fills(f, 10 * k)
               + [(bar, s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                                   fields=(0, 0, 0))),
                  (f, None)])
    s.warp(0, 3, 3, fills(f, 3))
    s.cta(0, 0, 0, 4, 0)
    return container(s.build())


def a9_cluster() -> bytes:
    s = Seg(2, 4, cluster_shape_x=2)
    s.guarded_program("synchronization.cluster.release", 3, 4,
                      TK_CL_REL,
                      [(AXID["expected_arrivals"], 0, 2, 2, 20.0),
                       (AXID["expected_arrivals"], 0, 1, 1, 5.0)])
    f = s.template("FILL")
    arr = s.template("CLUSTER.ARRIVE")
    wt = s.template("CLUSTER.WAIT")
    wi = 0
    for c in range(2):
        for k in range(2):
            s.warp(c, k, k,
                   fills(f, 10 * (2 * c + k))
                   + [(arr, s.semantic(CL_ARRIVE, fields=(0, 0)))]
                   + fills(f, 5)
                   + [(wt, s.semantic(CL_WAIT, fields=(0, 0))),
                      (f, None)])
            wi += 1
    s.cta(0, 0, 0, 2, 0)
    s.cta(1, 1, 1, 2, 2)
    return container(s.build(), cluster_x=2)


def a10_nanosleep() -> bytes:
    s = Seg(1, 1)
    f = s.template("FILL")
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 100.0, 1: 1.0}),))
    s.warp(0, 0, 0,
           [(ns, s.semantic(NANOSLEEP, fields=(100,))), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a10_syncs() -> bytes:
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 100.0, 1: 1.0}),))
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(0, 0, 0, 0, 0, 0))),
             (ns, s.semantic(NANOSLEEP, flags=SYNCS, fields=(100,))),
             (f, None)])
    s.warp(0, 1, 1, fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def m_syncs_wake() -> bytes:
    """NANOSLEEP.SYNCS wake latency: the sleeper wakes at release +
    syncs_wake_cycles (field 4 on the nanosleep timing row), not at
    the release itself."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP,
                             {0: 100.0, 1: 1.0, 4: 30.0}),))
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(0, 0, 0, 0, 0, 0))),
            (ns, s.semantic(NANOSLEEP, flags=SYNCS, fields=(100,))),
            (f, None)])
    s.warp(0, 1, 1, fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a10_past_release() -> bytes:
    """NANOSLEEP.SYNCS after a poll whose awaited phase is already
    released: the suspended try_wait ends at once (no sleep) — the
    captured failed poll ran ahead of the simulated timeline."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 100.0, 1: 1.0}),))
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0, fills(f, 20)
           + [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
              (ns, s.semantic(NANOSLEEP, flags=SYNCS, fields=(100,))),
              (f, None)])
    s.warp(0, 1, 1,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))]
           + fills(f, 5))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a10_plain_after_wait() -> bytes:
    """Plain NANOSLEEP (no .SYNCS) after a wait sleeps its full
    duration even though the object has been released."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 100.0, 1: 1.0}),))
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0, fills(f, 20)
           + [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
              (ns, s.semantic(NANOSLEEP, fields=(100,))),
              (f, None)])
    s.warp(0, 1, 1,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))]
           + fills(f, 5))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a10_cap() -> bytes:
    """PTX nanosleep cap: a 10 ms hint sleeps only 1 ms
    (1e6 cycles at the 1 GHz fixture clock)."""
    s = Seg(1, 1)
    f = s.template("FILL")
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 10000000.0, 1: 1.0}),))
    s.warp(0, 0, 0,
           [(ns, s.semantic(NANOSLEEP, fields=(10000000,))),
            (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a24_branch_envelope() -> bytes:
    """A BRA bound to an InstanceTiming control_envelope (USES target
    2, schema 5, field 4) pushes the warp's next issue to at least
    BRA issue + envelope (control.branch application path)."""
    s = Seg(1, 1)
    f = s.template("FILL")
    bra = s.template("BRA")
    s.warp(0, 0, 0, fills(f, 2) + [(bra, None)] + fills(f, 2))
    node = s.leaf_node(5, {4: 20.0})
    s.uses.append((node, 2, b"\0" * 3, 2, 5, 0x1, 0, 0))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())




def _mem_leaf(s):
    """memory.hierarchy leaf: 32 B sectors / 128 B lines, L2 port
    128 B/cycle -> 16 sectors = 4 cycles per request, L2 lat 100;
    store-L2 port 32 B/cycle (16 cycles per 16-sector store), lat 50,
    8 outstanding.  No L1 port."""
    node = s.leaf_node(3, {2: 32.0, 3: 128.0, 13: 100.0, 14: 8.0,
                           21: 50.0, 22: 8.0, 42: 128.0, 44: 32.0})
    s.uses.append((node, 5, b"\0" * 3, 0, 3, 0xFFFFFFFF, 0, 0))


def _a28(mode: str) -> bytes:
    """far-side ports never floor the read barrier: 8 global requests,
    each scoreboard_read=1, then a wait_mask=0x02 FILL, with the
    memory leaf carrying an L2 port (128 B/cycle -> 4 cycles per
    16-sector request) but no L1 port.  a28 ("load"): 16 distinct-line
    misses per load queue on the L2 port, yet the barrier frees at
    issue -> FILL = last + 1 (before 2026-11-08 the L2-port start held
    it to ~7x4 after the first).  a28b ("merged"): one repeated sector
    -> merged followers, FILL = last + 1.  a28c ("store"): 16-sector
    STGs through the store-L2 port (32 B/cycle -> 16 cycles each, so
    the port queue reaches ~100 cycles) -> FILL = last + 1 too."""
    s = Seg(1, 1)
    _mem_leaf(s)
    if mode == "store":
        req = s.template("STG", scoreboard_read=1)
    else:
        req = s.template("LDG", scoreboard_read=1)
    wt = s.template("FILL", scoreboard_wait_mask=0x02)
    first = len(s.inst)
    for i in range(8):
        macc = len(s.macc)
        nsec = 1 if mode == "merged" else 16
        s.macc.append((len(s.msec), nsec, 0x40 if mode == "store" else 0,
                       0, 0, 0, 0, 0))
        for j in range(nsec):
            s.msec.append((0x10000 if mode == "merged"
                           else i * 0x800 + j * 128, 0xFFFFFFFF, 0))
        s.instance(req, None, macc)
    s.instance(wt)
    s.warps.append((0, 0, 0, 0, first, 9))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a28_read_barrier_global() -> bytes:
    return _a28("load")


def a28b_read_barrier_merged() -> bytes:
    return _a28("merged")


def a28c_read_barrier_store() -> bytes:
    return _a28("store")


def _m_l1_port(l1_bpc: float) -> bytes:
    """memory.l1.service.sector_bytes_per_cycle leaf live: one warp,
    8 loads of 16 sectors that all hit L1 (pre-warmed by a first
    miss pass + a NANOSLEEP that lets the fill complete).  With
    l1_bpc=128 (32 B sectors) each request holds the L1 port 4
    cycles -> the 8th request's acceptance (read barrier) is
    ~7*4 after the first hit load issues.  With l1_bpc=0 the leaf
    is off -> acceptance at issue -> FILL = last + 1."""
    s = Seg(1, 1)
    node = s.leaf_node(3, {2: 32.0, 3: 128.0, 9: 20.0, 10: 16.0,
                           11: 1.0, 13: 100.0, 14: 8.0, 41: l1_bpc,
                           42: 128.0, 47: 262144.0})
    s.uses.append((node, 5, b"\0" * 3, 0, 3, 0xFFFFFFFF, 0, 0))
    ldg = s.template("LDG", scoreboard_read=1)
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 300.0, 1: 1.0}),))
    wt = s.template("FILL", scoreboard_wait_mask=0x02)
    f = s.template("FILL")
    first = len(s.inst)
    # warmup: same 16 sectors, misses L1 -> fills at completion
    macc = len(s.macc)
    s.macc.append((len(s.msec), 16, 0, 0, 0, 0, 0, 0))
    for j in range(16):
        s.msec.append((j * 128, 0xFFFFFFFF, 0))
    s.instance(ldg, None, macc)
    s.instance(ns, s.semantic(NANOSLEEP, fields=(300,)))
    # 8 hit loads on the pre-warmed sectors
    for _ in range(8):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 16, 0, 0, 0, 0, 0, 0))
        for j in range(16):
            s.msec.append((j * 128, 0xFFFFFFFF, 0))
        s.instance(ldg, None, macc)
    s.instance(wt)
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 11))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def m_l1_port() -> bytes:
    return _m_l1_port(128.0)


def m_l1_port_off() -> bytes:
    return _m_l1_port(0.0)


def m_control_yield() -> bytes:
    """control_yield v1 install: a YIELD bound to an instruction.service
    leaf (result 40) pushes the warp's next issue to YIELD+40 via the
    no-destination floor family (MEMBAR/FENCE)."""
    s = Seg(1, 1)
    f = s.template("FILL")
    y = s.template("YIELD", svc={1: 40.0, 2: 40.0})
    s.warp(0, 0, 0, fills(f, 2) + [(y, None)] + fills(f, 2))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def m_cpasync_anchor() -> bytes:
    """kind-39 group completion anchored at the group's FIRST copy
    issue: 4 CP.ASYNC copies issued ~10 cycles apart, leaf 300,
    commit+wait(pending=0) -> the wait passes at first+300, not
    last+300 (330+)."""
    s = Seg(1, 1)
    _mem_leaf(s)
    iss = s.template("CP.ASYNC", timing=((39, {0: 300.0, 1: 1.0}),))
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 10.0, 1: 1.0}),))
    com = s.template("CP.COMMIT")
    wt = s.template("CP.WAIT")
    f = s.template("FILL")
    first = len(s.inst)
    for i in range(4):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 16, 0, 0, 0, 0, 0, 0))
        for j in range(16):
            s.msec.append((i * 0x800 + j * 128, 0xFFFFFFFF, 0))
        s.instance(iss, s.semantic(CP_ISSUE, fields=(0, 0, NONE)),
                   macc)
        if i < 3:
            s.instance(ns, s.semantic(NANOSLEEP, fields=(10,)))
    s.instance(com, s.semantic(CP_COMMIT, fields=(0, 0, NONE, 0)))
    s.instance(wt, s.semantic(CP_WAIT, fields=(0, 0)))
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 10))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a25_control_stall() -> bytes:
    """TMPL.control bit0-3 stall count is the warp's next-issue floor:
    inst1 waits 6 after inst0 (stall=6), inst2 keeps +1 (stall=1),
    inst4 reads R5 written by inst3 whose result latency (4) exceeds
    its stall (2) -> the register floor wins, no additive +6."""
    s = Seg(1, 1)
    f6 = s.template("FILL", control=6)
    f1 = s.template("FILL", control=1)
    fd = s.template("FILL", control=2,
                    svc={1: 4.0, 2: 4.0, 7: 1.0}, dst=[5])
    fu = s.template("USE", src=[5])
    s.warp(0, 0, 0,
           [(f6, None), (f1, None), (f1, None),
            (fd, None), (fu, None), (f1, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a26_failed_poll_result() -> bytes:
    """A known failed poll (AFTER_READY, after_ready_mask=0) carries no
    service result latency: the dependent BRA issues wait+1 even though
    the phase is not yet released.  A successful wait on the same
    template keeps its full 38-cycle result latency."""
    s = Seg(1, 2)
    init = s.template("INIT")
    arr = s.template("ARRIVE")
    f = s.template("FILL")
    wt = s.template("TRYWAIT", svc={1: 38.0, 2: 38.0, 7: 1.0}, dst=[512])
    bra = s.template("BRA", src=[512])
    s.warp(0, 0, 0,
           [(init, s.semantic(MBAR_INIT, fields=(0, 1))),
            (wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(0, 0))),
            (bra, None),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (bra, None),
            (f, None)])
    s.warp(0, 1, 1,
           fills(f, 20)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def _a27(with_accept_row: bool) -> bytes:
    # 8 LDS each read-barrier 1; a wait_mask=0x02 FILL must wait until
    # the last LDS's read barrier frees.  With a kind-6 shared
    # acceptance row (10 cycles each, per SM) the last acceptance is
    # 7 x 10 after the first issue; without the row the barrier frees
    # at issue and the FILL runs last LDS + 1.
    s = Seg(1, 1)
    timing = ((6, {0: 10.0, 1: 1.0}),) if with_accept_row else ()
    lds = s.template("LDS", timing=timing, scoreboard_read=1)
    wt = s.template("FILL", scoreboard_wait_mask=0x02)
    f = s.template("FILL")
    first = len(s.inst)
    for _ in range(8):
        s.instance(lds)
    s.instance(wt)
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 10))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a27_read_barrier_accept() -> bytes:
    return _a27(True)


def a27b_read_barrier_issue() -> bytes:
    return _a27(False)


def a12_deadlock() -> bytes:
    s = Seg(1, 1)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a11_visibility() -> bytes:
    s = Seg(1, 2)
    f = s.template("FILL")
    stg = s.template("STG",
                     timing=((TK_VIS_GLOBAL, {0: 100.0, 1: 1.0}),))
    bar = _cta_bar_template(s)
    s.warp(0, 0, 0,
           [(stg, None),
            (bar, s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                             fields=(0, 0, 0))),
            (f, None)])
    s.warp(0, 1, 1, fills(f, 50)
           + [(bar, s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                               fields=(0, 0, 0))),
              (f, None)])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def _a13(initial_phase: int) -> bytes:
    s = Seg(1, 1)
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    ini = s.template("MBAR.INIT")
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           [(ini, s.semantic(MBAR_INIT,
                             fields=(0, 1, 0, initial_phase))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a21_init_mbol() -> bytes:
    """One init row covering an MBOL list of 3 objects (lane-parallel
    store promoted to init): every listed object is initialised, so
    arrives on objects 1 and 2 do not hit mbarrier_uninitialized and the
    wait on object 2 releases."""
    s = Seg(1, 3)
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    f = s.template("FILL")
    ini = s.template("MBAR.INIT")
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT")
    s.mbol.extend([(0,), (1,), (2,)])
    s.warp(0, 0, 0,
           [(ini, s.semantic(MBAR_INIT,
                             fields=(0, 1, 0, 0, 0, 3))),
            (wt, s.semantic(MBAR_WAIT, fields=(2, 0))),
            (f, None)])
    s.warp(0, 1, 1,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(1, 0, 0, 1))),
            (f, None)])
    s.warp(0, 2, 2,
           fills(f, 20)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(2, 0, 0, 1)))])
    s.cta(0, 0, 0, 3, 0)
    return container(s.build())


def a22_reinit_idle() -> bytes:
    """Re-init of an idle barrier: init expected 1, then init again
    expected 2 before any arrive (second init wins, as hardware) -> the
    wait releases only after two arrives."""
    s = Seg(1, 3)
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    f = s.template("FILL")
    ini = s.template("MBAR.INIT")
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           [(ini, s.semantic(MBAR_INIT, fields=(0, 1, 0, 0))),
            (ini, s.semantic(MBAR_INIT, fields=(0, 2, 0, 0))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    # both arrives delayed past the re-init (a cycle-0 arrive erased by
    # the second init would leave expected 2 with one real arrive)
    s.warp(0, 1, 1,
           fills(f, 5)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.warp(0, 2, 2,
           fills(f, 20)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 3, 0)
    return container(s.build())


def a22b_reinit_busy() -> bytes:
    """Re-init after an arrival of the in-progress phase is the
    converter-numbering error it always signalled."""
    s = Seg(1, 2)
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    f = s.template("FILL")
    ini = s.template("MBAR.INIT")
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(ini, s.semantic(MBAR_INIT, fields=(0, 2, 0, 0)))])
    # the arrive lands after warp 0's init, then the second init throws
    s.warp(0, 1, 1,
           fills(f, 5)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
              (ini, s.semantic(MBAR_INIT, fields=(0, 2, 0, 0)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a23_complete_tx() -> bytes:
    """flags 0x100 arrive: 16 expected tx complete with no arrival."""
    s = Seg(1, 2)
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    f = s.template("FILL")
    ini = s.template("MBAR.INIT")
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           [(ini, s.semantic(MBAR_INIT, fields=(0, 1, 0, 0))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    s.warp(0, 1, 1,
           fills(f, 5)
           # arrive +16 expected tx: pending 0 but tx outstanding, the
           # phase is still in progress; the complete_tx row closes it
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 16, 1))),
              (arr, s.semantic(MBAR_ARRIVE, flags=0x100,
                               fields=(0, 0, 16, 0))),
              (f, None)])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a23b_complete_tx_zero() -> bytes:
    """complete_tx row with 0 tx completes nothing -> still the missing
    arrive fact."""
    s = Seg(1, 1)
    ini = s.template("MBAR.INIT")
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(ini, s.semantic(MBAR_INIT, fields=(0, 1, 0, 0))),
            (arr, s.semantic(MBAR_ARRIVE, flags=0x100,
                             fields=(0, 0, 0, 0)))])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


# ---- step b: async groups / TMA / descriptors / gsync / segments ----
BULK_ISSUE, BULK_COMMIT, BULK_WAIT = 11, 12, 13
CP_ISSUE, CP_COMMIT, CP_WAIT = 14, 15, 16
TMA_LOAD, BULK_COPY = 10, 17
DESC_VIS, GSYNC_KIND = 38, 34
TK_TMA_COMPLETE, TK_TMA_START = 38, 21


def b1_bulk_group() -> bytes:
    s = Seg(1, 1)
    f = s.template("FILL")
    iss = s.template("UTMASTG",
                     timing=((TK_TMA_COMPLETE, {0: 50.0, 1: 1.0}),))
    com = s.template("BULK.COMMIT")
    wt = s.template("BULK.WAIT")
    s.warp(0, 0, 0,
           [(iss, s.semantic(BULK_ISSUE, fields=(0, 0, 0))),
            (iss, s.semantic(BULK_ISSUE, fields=(0, 0, 0))),
            (com, s.semantic(BULK_COMMIT, fields=(0, 0))),
            (wt, s.semantic(BULK_WAIT, fields=(0, 0))),
            (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b1_bulk_group_n1() -> bytes:
    s = Seg(1, 1)
    f = s.template("FILL")
    iss = s.template("UTMASTG",
                     timing=((TK_TMA_COMPLETE, {0: 50.0, 1: 1.0}),))
    com = s.template("BULK.COMMIT")
    wt = s.template("BULK.WAIT")
    s.warp(0, 0, 0,
           [(iss, s.semantic(BULK_ISSUE, fields=(0, 0, 0))),
            (com, s.semantic(BULK_COMMIT, fields=(0, 0))),
            (iss, s.semantic(BULK_ISSUE, fields=(0, 0, 0))),
            (com, s.semantic(BULK_COMMIT, fields=(0, 0))),
            (wt, s.semantic(BULK_WAIT, fields=(0, 1))),
            (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b2_cpasync_arrives() -> bytes:
    s = Seg(1, 1)
    s.mbpi.append((0, 0, 0x100, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    iss = s.template("CP.ASYNC")
    com = s.template("CP.COMMIT",
                     timing=((TK_ASYNC_ARRIVE, {0: 20.0, 1: 1.0}),))
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           [(iss, s.semantic(CP_ISSUE, fields=(0, 0, NONE))),
            (iss, s.semantic(CP_ISSUE, fields=(0, 0, NONE))),
            (com, s.semantic(CP_COMMIT, fields=(0, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b2b_cpasync_transcnt() -> bytes:
    """cutlass cp.async pipeline shape: init 1; SYNCS.ARRIVE.RED.A0T1 ->
    expect 32 tx (arrive 0), ARRIVES.TRANSCNT -> complete 32 tx (no
    arrival), then the real A1T0 arrive completes the phase."""
    s = Seg(1, 1)
    s.mbpi.append((0, 0, 0x100, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    iss = s.template("CP.ASYNC")
    exp = s.template("MBAR.EXPECT")
    com = s.template("CP.COMMIT",
                     timing=((TK_ASYNC_ARRIVE, {0: 20.0, 1: 1.0}),))
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           [(exp, s.semantic(MBAR_ARRIVE, fields=(0, 0, 32, 0))),
            (iss, s.semantic(CP_ISSUE, fields=(0, 0, NONE))),
            (com, s.semantic(CP_COMMIT, fields=(0, 0, 0, 0, 32))),
            (arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _b3(with_mbob: bool, late_arrive: bool) -> bytes:
    s = Seg(2, 2)
    # obj0 owned by cta0, obj1 by cta1 — same CTA-local address
    s.mbpi.append((0, 0, 0x100, 1, 0))
    s.mbpi.append((1, 1, 0x100, 1, 0))
    if with_mbob:
        s.mbob.append((0, 0x100))
        s.mbob.append((1, 0x100))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    arr = s.template("MBAR.ARRIVE")
    tma = s.template("UTMALDG",
                     timing=((TK_TMA_COMPLETE, {0: 50.0, 1: 1.0}),))
    wt = s.template("MBAR.WAIT")
    s.warp(0, 0, 0,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 1024, 1))),
            (tma, s.semantic(TMA_LOAD, fields=(0, 0, 3, 1024))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    if late_arrive:
        s.warp(1, 0, 0, fills(f, 60)
               + [(arr, s.semantic(MBAR_ARRIVE, fields=(1, 0, 1024, 1))),
                  (wt, s.semantic(MBAR_WAIT, fields=(1, 0))),
                  (f, None)])
    else:
        s.warp(1, 0, 0,
               [(arr, s.semantic(MBAR_ARRIVE, fields=(1, 0, 1024, 1))),
                (wt, s.semantic(MBAR_WAIT, fields=(1, 0))),
                (f, None)])
    s.cta(0, 0, 0, 1, 0)
    s.cta(1, 1, 1, 1, 1)
    return container(s.build(), cluster_x=2)


def b4_service_start() -> bytes:
    s = Seg(1, 3)
    for k in range(3):
        s.mbpi.append((0, k, 0x100 + k * 8, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    arr = s.template("MBAR.ARRIVE")
    tma = s.template("UTMALDG",
                     timing=((TK_TMA_START, {0: 10.0, 1: 1.0}),
                             (TK_TMA_COMPLETE, {0: 50.0, 1: 1.0})))
    wt = s.template("MBAR.WAIT")
    for k in range(3):
        s.warp(0, k, k,
               [(arr, s.semantic(MBAR_ARRIVE, fields=(k, 0, 1024, 1))),
                (tma, s.semantic(TMA_LOAD, fields=(k, 0, 0, 1024))),
                (wt, s.semantic(MBAR_WAIT, fields=(k, 0))),
                (f, None)])
    s.cta(0, 0, 0, 3, 0)
    return container(s.build())


CUT_AX = 13  # "cut_rows" in fixture_lib.AXES
CUT_EXTRA_PTS = [(0, 0), (1, 178), (4096, 178)]
CUT_SVC_PTS = [(0, 0), (128, 258.6), (4096, 8275)]


def _a32(cut_rows: int | None, n_tma: int, svc_pts, extra_pts) -> bytes:
    """tma_oob v2 cut-row rule: n_tma single-CTA UTMALDGs on one SM,
    each carrying a completion cut_extra / service-start cut_service
    linear leaf on capture axis cut_rows."""
    s = Seg(1, n_tma)
    for k in range(n_tma):
        s.mbpi.append((0, k, 0x100 + k * 8, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    arr = s.template("MBAR.ARRIVE")
    facts = None if cut_rows is None else [("cut_rows", cut_rows)]
    tma = s.template(
        "UTMALDG",
        timing=((TK_TMA_START, {0: 10.0, 1: 1.0}, facts,
                 [(15, CUT_AX, svc_pts)] if svc_pts else ()),
                (TK_TMA_COMPLETE, {0: 50.0, 1: 1.0}, facts,
                 [(14, CUT_AX, extra_pts)] if extra_pts else ())))
    wt = s.template("MBAR.WAIT")
    for k in range(n_tma):
        s.warp(0, k, k,
               [(arr, s.semantic(MBAR_ARRIVE, fields=(k, 0, 1024, 1))),
                (tma, s.semantic(TMA_LOAD, fields=(k, 0, 0, 1024))),
                (wt, s.semantic(MBAR_WAIT, fields=(k, 0))),
                (f, None)])
    s.cta(0, 0, 0, n_tma, 0)
    return container(s.build())


def a32_tma_uncut() -> bytes:
    return _a32(None, 1, None, None)


def a32b_tma_cut_extra() -> bytes:
    return _a32(128, 1, None, CUT_EXTRA_PTS)


def a32c_tma_cut_service() -> bytes:
    return _a32(128, 4, CUT_SVC_PTS, None)


def a32d_tma_cut_service32() -> bytes:
    return _a32(32, 4, CUT_SVC_PTS, None)


def a32e_tma_uncut4() -> bytes:
    return _a32(None, 4, None, None)


def b5_descriptor() -> bytes:
    s = Seg(1, 2)
    s.const_program("synchronization.descriptor.visibility",
                    3, 4, 23, 724.0)
    f = s.template("FILL")
    pub = s.template("DESC.PUBLISH")
    acq = s.template("DESC.ACQUIRE")
    s.warp(0, 0, 0,
           [(pub, s.semantic(DESC_VIS, fields=(1, 1, 0xABC))), (f, None)])
    s.warp(0, 1, 1, fills(f, 10)
           + [(acq, s.semantic(DESC_VIS, fields=(4, 0, 0xABC))),
               (f, None)])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def _b6(rows: list[tuple], **launch_kw) -> bytes:
    s = Seg(1, 3)
    f = s.template("FILL")
    rel = s.template("GSYNC.RELEASE")
    acq = s.template("GSYNC.ACQUIRE")
    # w0: release-ish row at inst5; w1: acquire at inst6 (first);
    # w2: second producer at inst16
    s.warp(0, 0, 0,
           fills(f, 5) + [(rel, s.semantic(GSYNC_KIND, fields=(0, 0)))])
    s.warp(0, 1, 1,
           [(acq, s.semantic(GSYNC_KIND, fields=(0, 0))),
            (f, None)])
    s.warp(0, 2, 2,
           fills(f, 8) + [(rel, s.semantic(GSYNC_KIND, fields=(0, 0)))])
    s.cta(0, 0, 0, 3, 0)
    return container(s.build(), global_sync=b"".join(
        cf.GSYN.pack_row(r) for r in rows), **launch_kw)


def b6a_from() -> bytes:
    return _b6([gsyn(0, 5, GS_ST, address=0xA, after=7, sequence=1),
                gsyn(0, 6, GS_LD, address=0xA, after=7, sequence=2)])


def b6b_initial() -> bytes:
    # the read's sequence predates the equal-value producer ->
    # it observed the pre-write value (initial acquire)
    return _b6([gsyn(0, 6, GS_LD, address=0xA, after=0, sequence=1),
                gsyn(0, 5, GS_ST, address=0xA, after=7, sequence=2)])


def b6c_any() -> bytes:
    return _b6([gsyn(0, 5, GS_ST, address=0xA, after=7, sequence=1),
                gsyn(0, 16, GS_ST, address=0xA, after=7, sequence=2),
                gsyn(0, 6, GS_LD, address=0xA, after=7, sequence=3)])


def b6d_chain() -> bytes:
    # atom chain 1->2->3; the read pairs with the last link; chain links
    # carry no issue gate (L2 orders same-address updates on arrival)
    return _b6([gsyn(0, 5, GS_ATOM, address=0xA, before=1, after=2,
                     sequence=1),
                gsyn(0, 16, GS_ATOM, address=0xA, before=2, after=3,
                     sequence=2),
                gsyn(0, 6, GS_LD, address=0xA, after=3, sequence=3)])


def b6e_count() -> bytes:
    # two red +1 updates, no chain (same valid before=0) -> count k=2
    return _b6([gsyn(0, 5, GS_RED, address=0xA, before=0, after=1,
                     sequence=1, flags=3),
                gsyn(0, 16, GS_RED, address=0xA, before=0, after=1,
                     sequence=2, flags=3),
                gsyn(0, 6, GS_LD, address=0xA, after=2, sequence=3)])


def b6f_poll() -> bytes:
    s = Seg(1, 2)
    f = s.template("FILL")
    acq = s.template("GSYNC.ACQUIRE")
    s.warp(0, 0, 0,
           fills(f, 5) + [(f, None)])
    s.warp(0, 1, 1,
           [(f, s.semantic(GSYNC_KIND, fields=(0, 0))),
            (f, s.semantic(GSYNC_KIND, fields=(0, 0))),
            (f, s.semantic(GSYNC_KIND, fields=(0, 0)))])
    s.cta(0, 0, 0, 2, 0)
    # three same-template reads on lane 1: the first two are
    # dropped polls; the winner at inst8 pairs `from` w0's inst5
    rows = [gsyn(0, 5, GS_ST, address=0xA, after=7, sequence=4),
            gsyn(0, 6, GS_LD, address=0xA, after=0, sequence=2,
                 lane=1),
            gsyn(0, 7, GS_LD, address=0xA, after=0, sequence=3,
                 lane=1),
            gsyn(0, 8, GS_LD, address=0xA, after=7, sequence=5,
                 lane=1)]
    return container(s.build(), global_sync=b"".join(
        cf.GSYN.pack_row(r) for r in rows))


def b6l_poll_free() -> bytes:
    """Failed polls of a poll sequence carry no load latency: the
    strong-load template has result latency 400 and a dependent USE
    follows each poll; the two failed polls' USEs issue right away,
    the winner still pairs `from` w0's release."""
    s = Seg(1, 2)
    f = s.template("FILL")
    poll = s.template("GSYNC.POLL", svc={1: 400.0, 2: 400.0, 7: 1.0},
                      dst=[5])
    use = s.template("USE", src=[5])
    s.warp(0, 0, 0,
           fills(f, 5) + [(f, s.semantic(GSYNC_KIND, fields=(0, 0)))])
    s.warp(0, 1, 1,
           [(poll, s.semantic(GSYNC_KIND, fields=(0, 0))), (use, None),
            (poll, s.semantic(GSYNC_KIND, fields=(0, 0))), (use, None),
            (poll, s.semantic(GSYNC_KIND, fields=(0, 0))), (use, None)])
    s.cta(0, 0, 0, 2, 0)
    rows = [gsyn(0, 5, GS_ST, address=0xA, after=7, sequence=4),
            gsyn(0, 6, GS_LD, address=0xA, after=0, sequence=2,
                 lane=1),
            gsyn(0, 8, GS_LD, address=0xA, after=0, sequence=3,
                 lane=1),
            gsyn(0, 10, GS_LD, address=0xA, after=7, sequence=5,
                 lane=1)]
    return container(s.build(), global_sync=b"".join(
        cf.GSYN.pack_row(r) for r in rows))


def _b6m(mode: int) -> bytes:
    """Rule 3b poll-region folding (b6m): warp 0 releases after 5 FILL;
    warp 1 spins [poll, use50 x3, poll, use50 x3, poll(winner), use].
    The use50 chain has 50-cycle result latency so per-instance issue
    would cost >= 300 cycles; a foldable region retires all 8 at the
    winner's cycle. b6n plants an mbarrier arrive in the region ->
    no fold."""
    s = Seg(1, 2)
    if mode == 1:
        s.mbpi.append((0, 0, 0x1000, 2, 0))
        s.const_program("synchronization.mbarrier.release", 3, 4, 22,
                        10.0)
    f = s.template("FILL")
    poll = s.template("GSYNC.POLL", svc={1: 400.0, 2: 400.0, 7: 1.0},
                      dst=[5])
    arr = s.template("MBAR.ARRIVE")
    u = []
    for i in range(6):
        u.append(s.template("USE", svc={1: 50.0, 2: 50.0, 7: 1.0},
                            src=[5 + i], dst=[6 + i]))
    use = s.template("USE", src=[10])
    g = s.semantic(GSYNC_KIND, fields=(0, 0))
    mid = [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))
           if mode == 1 else (u[0], None)]
    s.warp(0, 0, 0,
           fills(f, 5) + [(f, s.semantic(GSYNC_KIND, fields=(0, 0)))])
    s.warp(0, 1, 1,
           [(poll, g)] + mid + [(u[1], None), (u[2], None),
            (poll, g), (u[3], None), (u[4], None), (u[5], None),
            (poll, g), (use, None)])
    if mode in (2, 3):
        # b6o: give the region's first use row a global memory access —
        # a data-memory MACC inside the region blocks folding (2d40921f);
        # b6p: the same access in constant space (flags space 3, an LDC
        # inside the spin body) does not
        macc = len(s.macc)
        s.macc.append((0, 1, 0x03 if mode == 3 else 0x00, 0, 0, 0, 0, 0))
        s.msec.append((0, 0xFFFFFFFF, 0))
        gi = s.warps[-1][4] + 1  # warp1 local inst 1 (the first use)
        t, sm, _, ix = s.inst[gi]
        s.inst[gi] = (t, sm, macc, ix)
    s.cta(0, 0, 0, 2, 0)
    rows = [gsyn(0, 5, GS_ST, address=0xA, after=7, sequence=4),
            gsyn(0, 6, GS_LD, address=0xA, after=0, sequence=2,
                 lane=1),
            gsyn(0, 10, GS_LD, address=0xA, after=0, sequence=3,
                 lane=1),
            gsyn(0, 14, GS_LD, address=0xA, after=7, sequence=5,
                 lane=1)]
    return container(s.build(), global_sync=b"".join(
        cf.GSYN.pack_row(r) for r in rows))


def b6m_poll_region() -> bytes:
    return _b6m(0)


def b6n_poll_region_mixed() -> bytes:
    return _b6m(1)


def b6o_poll_region_macc() -> bytes:
    return _b6m(2)


def b6p_poll_region_constant() -> bytes:
    return _b6m(3)


def b6g_unpaired() -> bytes:
    # no grid on the launch -> coverage unknown, the no-writer rule
    # stays fail-closed and this still rejects
    return _b6([gsyn(0, 6, GS_LD, address=0xA, after=5, sequence=1)])


def b6i_no_writer() -> bytes:
    # full coverage (grid == CTA count) and no writer on the address:
    # the strong read resolved pre-launch memory -> plain load
    return _b6([gsyn(0, 6, GS_LD, address=0xA, after=0, sequence=1,
                     flags=1)],
               grid_x=1, grid_y=1, grid_z=1)


def b6j_no_writer_partial() -> bytes:
    # grid 2 CTAs but the job covers 1 -> the writer may sit in an
    # unsampled CTA, still unpaired_read
    return _b6([gsyn(0, 6, GS_LD, address=0xA, after=0, sequence=1,
                     flags=1)],
               grid_x=2, grid_y=1, grid_z=1)


def b6k_unknown_writer() -> bytes:
    # an update with invalid after is an unknown writer -> rejects
    return _b6([gsyn(0, 5, GS_ATOM, address=0xA, flags=0, sequence=1),
                gsyn(0, 6, GS_LD, address=0xA, after=0, sequence=2,
                     flags=1)],
               grid_x=1, grid_y=1, grid_z=1)


def b6h_noop() -> bytes:
    # failed CAS (before==after) is a reader, not a producer
    return _b6([gsyn(0, 5, GS_ATOM, address=0xA, before=3, after=3,
                     sequence=1),
                gsyn(0, 16, GS_ST, address=0xA, after=3, sequence=2),
                gsyn(0, 6, GS_LD, address=0xA, after=3, sequence=3)])


def b7_envelope() -> bytes:
    s = Seg(1, 1)
    f = s.template("FILL")
    s.warp(0, 0, 0, fills(f, 6))
    s.cta(0, 0, 0, 1, 0)
    # USES target 2 (InstanceTiming), schema 5 leaf {4: 30} on inst 3
    node = s.leaf_node(5, {4: 30.0})
    s.uses.append((node, 2, b"\0" * 3, 3, 5, 0x10, 0, 0))
    return container(s.build())


def b10_fence_floor() -> bytes:
    # no-destination service result holds the warp to issue + result
    s = Seg(1, 1)
    mb = s.template("MEMBAR.ALL.GPU", svc={1: 50.0, 2: 50.0})
    f = s.template("FILL")
    s.warp(0, 0, 0, [(mb, None), (f, None), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b10b_dest_result() -> bytes:
    # a destination register carries the same latency as a reg_ready
    # dependency, but the warp itself is not held
    s = Seg(1, 1)
    alu = s.template("IADD3", svc={1: 50.0, 2: 50.0}, dst=[5])
    f = s.template("FILL")
    s.warp(0, 0, 0, [(alu, None), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b10c_store_result() -> bytes:
    # non-fence no-destination results do not floor the warp
    s = Seg(1, 1)
    st = s.template("STSM", svc={1: 28.0, 2: 28.0})
    f = s.template("FILL")
    s.warp(0, 0, 0, [(st, None), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


# ---- storage.shared.fence (timing kind 66) --------------------------

TK_FENCE_DRAIN = 66


def _sts(s: Seg, nsec: int) -> int:
    """one synchronous shared store: shared-space MACC (flags&7==1) +
    store kind (0x40), nsec sectors with byte_mask 0 (-> 32 B each);
    kind-6 SM acceptance 10 cycles."""
    macc = len(s.macc)
    s.macc.append((len(s.msec), nsec, 0x41, 0, 0, 0, 0, 0))
    for _ in range(nsec):
        s.msec.append((0, 0, 0))
    return macc


def _fence(s: Seg, drain: bool) -> tuple:
    """FENCE.VIEW.ASYNC.S semantic + template; drain=True carries the
    kind-66 leaf {cycles 75.26, enabled 1, drain 569 B/cyc}."""
    timing = ((TK_FENCE_DRAIN, {0: 75.26, 1: 1.0, 10: 569.0}),) \
        if drain else ()
    return (s.template("FENCE.VIEW.ASYNC.S", timing=timing,
                       svc={1: 9.0, 2: 9.0}),
            None)


def b16_fence_drain() -> bytes:
    # 4 shared stores (4 x 512 B) accepted 10/20/30/40, fence at 4
    # drains: ceil(40 + 9 + 75.26 + 2048/569) = 128
    s = Seg(1, 1)
    st = s.template("STS", timing=((6, {0: 10.0, 1: 1.0}),))
    fence, _ = _fence(s, True)
    f = s.template("FILL")
    first = len(s.inst)
    for _ in range(4):
        s.instance(st, None, _sts(s, 16))
    s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 6))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b17_fence_drain_clean() -> bytes:
    # dirty-free fence: the kind-66 row and its absence release the
    # warp on the same cycle (service result floor only)
    s = Seg(1, 2)
    fd, _ = _fence(s, True)
    fn, _ = _fence(s, False)
    f = s.template("FILL")
    s.warp(0, 0, 0, [(fd, s.semantic(DESC_VIS, fields=(5,))), (f, None)])
    s.warp(0, 1, 1, [(fn, s.semantic(DESC_VIS, fields=(5,))), (f, None)])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def b18_fence_drain_second() -> bytes:
    # warp0 stores + fence + BAR; warp1 fences after the barrier with
    # nothing new written: its fence is not held by the rule
    s = Seg(1, 2)
    st = s.template("STS", timing=((6, {0: 10.0, 1: 1.0}),))
    fence, _ = _fence(s, True)
    bar = _cta_bar_template(s)
    f = s.template("FILL")
    bsema = s.semantic(CTA_WAIT, flags=ALSO_ARRIVES, fields=(0, 0, 0))
    first = len(s.inst)
    for _ in range(4):
        s.instance(st, None, _sts(s, 16))
    s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
    s.instance(bar, bsema)
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 7))
    first = len(s.inst)
    s.instance(bar, bsema)
    s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
    s.instance(f)
    s.warps.append((0, 1, 1, 1, first, 3))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


# ---- storage.shared.acceptance_smsp (timing kind 80) ----------------

TK_SMSP_ACCEPT = 80
TK_TMA_REDUCE = 81


def _st_rows() -> tuple:
    """shared-store rows: SMSP issue 4, SM acceptance 16, SMSP
    acceptance 32."""
    return ((5, {0: 4.0, 1: 1.0}), (6, {0: 16.0, 1: 1.0}),
            (TK_SMSP_ACCEPT, {0: 32.0, 1: 1.0}))


def _fence_cycles(s: Seg) -> tuple:
    """kind-66 leaf {cycles 5, enabled 1}, no drain field: occupy =
    cycles only, plus the fence's own service result latency 9."""
    return (s.template("FENCE.VIEW.ASYNC.S",
                       timing=((TK_FENCE_DRAIN, {0: 5.0, 1: 1.0}),),
                       svc={1: 9.0, 2: 9.0}), None)


def _global(s: Seg, nsec: int) -> int:
    """non-shared MACC: kind bits 0x40 (store) but space != shared, so
    the kinds 5/6/80 shared_space veto skips their rows."""
    macc = len(s.macc)
    s.macc.append((len(s.msec), nsec, 0x40, 0, 0, 0, 0, 0))
    for _ in range(nsec):
        s.msec.append((0, 0, 0))
    return macc


def b20_smsp_accept() -> bytes:
    # one warp, 4 stores x 512 B issued 0-3: the SMSP frontier
    # 4 x 32 = 128 (starts 0/32/64/96) binds over the SM frontier
    # 4 x 16 = 64, so the CTA's dirty-until is 0 + 128; the fence at 4
    # then holds the warp until 128 + 9 + 5 = 142
    s = Seg(1, 1)
    st = s.template("STS", timing=_st_rows())
    fence, _ = _fence_cycles(s)
    f = s.template("FILL")
    first = len(s.inst)
    for _ in range(4):
        s.instance(st, None, _sts(s, 16))
    s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 6))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b21_smsp_accept_sm() -> bytes:
    # four warps on four SMSPs, 4 stores each: each SMSP accepts its
    # own 4 x 32 = 128 while the SM frontier serialises the 16 stores
    # to 16 x 16 = 256, so dirty-until = 0 + 256; warp0's fence holds
    # until 256 + 9 + 5 = 270
    s = Seg(1, 1)
    st = s.template("STS", timing=_st_rows())
    fence, _ = _fence_cycles(s)
    f = s.template("FILL")
    for w in range(4):
        first = len(s.inst)
        for _ in range(4):
            s.instance(st, None, _sts(s, 16))
        if w == 0:
            s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
            s.instance(f)
            s.warps.append((0, w, w, w, first, 6))
        else:
            s.warps.append((0, w, w, w, first, 4))
    s.cta(0, 0, 0, 4, 0)
    return container(s.build())


def b22_smsp_accept_same() -> bytes:
    # two warps on the SAME SMSP (slots 0 and 4, both smsp 0), 2 stores
    # each: the SMSP frontier serialises all four stores to 4 x 32 = 128
    # against the SM frontier's 4 x 16 = 64.  Warps on one SMSP issue
    # non-interleaved here (warp0's stream drains first), so the fence
    # sits in warp1 and sees every store: dirty-until 0 + 128, hold
    # until 128 + 9 + 5 = 142
    s = Seg(1, 1)
    st = s.template("STS", timing=_st_rows())
    fence, _ = _fence_cycles(s)
    f = s.template("FILL")
    first = len(s.inst)
    for _ in range(2):
        s.instance(st, None, _sts(s, 16))
    s.warps.append((0, 0, 0, 0, first, 2))
    first = len(s.inst)
    for _ in range(2):
        s.instance(st, None, _sts(s, 16))
    s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
    s.instance(f)
    s.warps.append((0, 1, 4, 0, first, 4))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def b23_smsp_accept_nonshared() -> bytes:
    # a kind-80 row on an instance whose MACC is not shared space is
    # vetoed: the shared store issued at 1 still opens the SMSP
    # frontier 1 -> 33 (a non-vetoed row would push it to 32 -> 64 and
    # the fence to 78); the fence holds until 33 + 9 + 5 = 47
    s = Seg(1, 1)
    g = s.template("ST", timing=((TK_SMSP_ACCEPT, {0: 32.0, 1: 1.0}),))
    st = s.template("STS", timing=_st_rows())
    fence, _ = _fence_cycles(s)
    f = s.template("FILL")
    first = len(s.inst)
    s.instance(g, None, _global(s, 16))
    s.instance(st, None, _sts(s, 16))
    s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 4))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _membar(s: Seg, lat: float) -> tuple:
    """CTA memory barrier: kind-66 row (cycles 5, enabled) but no
    proxy-fence semantic -- the engine's "not the proxy fence" branch."""
    return (s.template("MEMBAR.ALL.CTA",
                       timing=((TK_FENCE_DRAIN, {0: 5.0, 1: 1.0}),),
                       svc={1: lat, 2: lat}), None)


def b24_smsp_membar() -> bytes:
    # one warp: 2 shared stores (accepted 0->32 and 32->64 on the SMSP
    # frontier, SM only 32), a MEMBAR (service latency 7), the proxy
    # fence (latency 9, cycles 5), a second MEMBAR, then FILL.
    # stores issue 0/1 -> dirty-until 64; the first MEMBAR issues at 2
    # and is held until 64 + 7 = 71; the fence issues at 71 and is
    # held until 71 + 9 + 5 = 85, clearing the ledger; the second
    # MEMBAR issues at 85 with nothing dirty -> only its own 7, so
    # FILL issues at 92.
    s = Seg(1, 1)
    st = s.template("STS", timing=_st_rows())
    mb, _ = _membar(s, 7.0)
    fence, _ = _fence_cycles(s)
    f = s.template("FILL")
    first = len(s.inst)
    for _ in range(2):
        s.instance(st, None, _sts(s, 16))
    s.instance(mb)
    s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
    s.instance(mb)
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 6))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b25_membar_clean() -> bytes:
    # a MEMBAR with a clean CTA is unaffected: issue at 0, its own
    # service latency 7 floors the warp, FILL issues at 7.
    s = Seg(1, 1)
    mb, _ = _membar(s, 7.0)
    f = s.template("FILL")
    first = len(s.inst)
    s.instance(mb)
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 2))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b26_smsp_accept_frac() -> bytes:
    # the SMSP frontier keeps fractional cycles: 4 stores x 31.98
    # booked fractionally end at 127.92 -> dirty-until 128 (integer
    # arithmetic would give 124); the fence at 4 holds until
    # 128 + 9 + 5 = 142
    s = Seg(1, 1)
    st = s.template("STS", timing=((5, {0: 4.0, 1: 1.0}),
                                   (6, {0: 16.0, 1: 1.0}),
                                   (TK_SMSP_ACCEPT,
                                    {0: 31.98, 1: 1.0})))
    fence, _ = _fence_cycles(s)
    f = s.template("FILL")
    first = len(s.inst)
    for _ in range(4):
        s.instance(st, None, _sts(s, 16))
    s.instance(fence, s.semantic(DESC_VIS, fields=(5,)))
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 6))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _b27_reduce_store(kind81: bool) -> bytes:
    """bulk-group UTMAREDG on 8 non-resident sectors (MACC access_kind
    0x100 -> store path, atomic): every sector is fetched from DRAM
    by the slice (no memory.shared_service leaf in the fixture -> the
    read service returns at once, done = 0 + dram_latency 200 ->
    fill_done = 200).  With the kind-81 row {enabled 1,
    sector_cost_ratio 1.25} the reduce's sectors cost 1.25x on the
    store services; the bulk group's completion stays
    max(finish, now + 50) with or without the row (no fetch hold)."""
    s = Seg(1, 1)
    _mem_uses(s)
    rows = [(TK_TMA_COMPLETE, {0: 50.0, 1: 1.0})]
    if kind81:
        rows.append((TK_TMA_REDUCE, {1: 1.0, 20: 1.25}))
    red = s.template("UTMAREDG", timing=tuple(rows))
    com = s.template("BULK.COMMIT")
    wt = s.template("BULK.WAIT")
    f = s.template("FILL")
    macc = len(s.macc)
    s.macc.append((0, 8, 0x100, 0, 0, 0, 0, 0))
    for i in range(8):
        s.msec.append((i * 32, 0xFFFFFFFF, 0))
    first = len(s.inst)
    s.inst.append((red, s.semantic(BULK_ISSUE, fields=(0, 0, 0)),
                   macc, len(s.inst)))
    s.inst.append((com, s.semantic(BULK_COMMIT, fields=(0, 0)),
                   NONE, len(s.inst)))
    s.inst.append((wt, s.semantic(BULK_WAIT, fields=(0, 0)),
                   NONE, len(s.inst)))
    s.inst.append((f, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, first, 4))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b19_fence_drain_twice() -> bytes:
    # second round charges only the new stores: fence1 drains
    # ceil(40 + 9 + 75.26 + 2048/569) = 128; the next stores issue
    # 128/129 and are accepted 138/148 (the accept frontier outlives
    # issue), fence2 drains ceil(148 + 9 + 75.26 + 1024/569) = 235
    s = Seg(1, 1)
    st = s.template("STS", timing=((6, {0: 10.0, 1: 1.0}),))
    fence, _ = _fence(s, True)
    f = s.template("FILL")
    fsema = s.semantic(DESC_VIS, fields=(5,))
    first = len(s.inst)
    for _ in range(4):
        s.instance(st, None, _sts(s, 16))
    s.instance(fence, fsema)
    for _ in range(2):
        s.instance(st, None, _sts(s, 16))
    s.instance(fence, fsema)
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 9))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def b8_two_segments() -> bytes:
    def seg(slot: int, sm: int) -> bytes:
        s = Seg(1, 1)
        s.mbpi.append((0, 0, 0x100, 1, 0))
        s.const_program("synchronization.mbarrier.release", 3, 4, 22,
                        0.0)
        f = s.template("FILL")
        arr = s.template("MBAR.ARRIVE")
        wt = s.template("MBAR.WAIT")
        s.warp(0, 0, slot,
               [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
                (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
                (f, None)])
        s.cta(0, 0, sm, 1, 0)
        return s.build()
    # distinct SMSPs: no scheduler contention between the segments
    return container_multi([seg(0, 0), seg(1, 0)])


def b9_drain() -> bytes:
    s = Seg(1, 1)
    iss = s.template("UTMASTG",
                     timing=((TK_TMA_COMPLETE, {0: 50.0, 1: 1.0}),))
    com = s.template("BULK.COMMIT")
    s.warp(0, 0, 0,
           [(iss, s.semantic(BULK_ISSUE, fields=(0, 0, 0))),
            (com, s.semantic(BULK_COMMIT, fields=(0, 0)))])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


# ---- step c: dispatch / lifecycle / fetch ----


def _c1(fwi: float, sfe: float, serial: float, enabled: float,
        occupancy: int) -> bytes:
    segs = []
    for i in range(4):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: fwi, 1: sfe, 2: enabled,
                                3: serial})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        # launch-wide CTA id i (segments carry linear ids; PLCE too)
        s.warp(i, 0, i, fills(f, 10))
        # cta0/1 are PLCE-placed; cta2/3 are truly unplaced (no sm_id)
        s.cta(i, 0, i % 2 if i < 2 else NONE, 1, 0)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 1, 0, 1, 1, 0),
                   plce(2, 2, 0, 0, 0, 0), plce(3, 3, 0, 0, 0, 0)])
    tp = b"".join(cf.TOPO.pack_row(r)
                  for r in [topo(0), topo(1)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=2, occupancy_blocks_per_sm=occupancy)


def c1f_trace_placed() -> bytes:
    segs = []
    # no PLCE at all; CTAS.sm_id 0/1/0/1 + warp_slot 0/0/1/1 ->
    # trace placements (0,0),(0,1),(1,0),(1,1): w0@0 w2@3 w1@6 w3@9
    for i, (sm, slot) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 3.0, 1: 20.0, 2: 1.0, 3: 0.0})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        s.warp(0, 0, slot, fills(f, 10))
        s.cta(0, 0, sm, 1, 0)
        segs.append(s.build())
    return container_multi(segs, sm_count=2)


def _c2(with_topo: bool) -> bytes:
    segs = []
    for i in range(2):
        s = Seg(2, 2)
        node = s.leaf_node(24, {0: 0.0, 1: 20.0, 2: 1.0, 3: 0.0})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        s.warp(0, 0, 0, fills(f, 10))
        s.warp(1, 0, 0, fills(f, 10))
        # seg0 placed via PLCE; seg1 unplaced (no CTAS.sm_id)
        s.cta(0, 0, i * 2 % 2 if i == 0 else NONE, 1, 0)
        s.cta(1, 1, (i * 2 + 1) % 2 if i == 0 else NONE, 1, 1)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 0, 1, 1, 1, 0)])
    tp = b"" if not with_topo else b"".join(cf.TOPO.pack_row(r)
          for r in [topo(0, 0, 0), topo(1, 0, 0)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=2, occupancy_blocks_per_sm=1)


def _gpc(topo_kind: str) -> bytes:
    # memory.l2_merge domain_gpc (schema 22 field 4): the engine builds
    # the merge domain table from TOPO at load_setup; no TOPO, or a row
    # without gpc while domain_sms is absent, is a missing fact
    s = Seg(1, 1)
    node = s.leaf_node(24, {0: 0.0, 1: 20.0, 2: 1.0, 3: 0.0})
    s.uses.append(init_use(node, 24))
    node = s.leaf_node(22, {4: 1.0})
    s.uses.append(init_use(node, 22))
    f = s.template("FILL")
    s.warp(0, 0, 0, fills(f, 10))
    s.cta(0, 0, 0, 1, 0)
    pl = b"".join(cf.PLCE.pack_row(r) for r in [plce(0, 0, 0, 1, 0, 0)])
    tp = (b"" if topo_kind == "none" else
          b"".join(cf.TOPO.pack_row((sm, 0, 0, NONE, NONE, 0))
                   for sm in (0, 1)))
    return container_multi([s.build()], placements=pl, topo=tp,
                           sm_count=2, occupancy_blocks_per_sm=1)


def _lifecycle_node(s: Seg, fields: dict[int, float],
                    warm_pts: list[tuple] | None) -> int:
    pb = len(s.parm)
    if warm_pts is not None:
        pts_begin = len(s.pts)
        s.pts.extend(warm_pts)
        # envelope_cycles (field 2) linear in launch_warps_per_cta
        s.parm.append((25, 2, 2, b"\0" * 3,
                       AXID["launch_warps_per_cta"], 0.0, 0.0, 0.0,
                       pts_begin, len(warm_pts)))
    for fid, v in sorted(fields.items()):
        s.parm.append((25, fid, 0, b"\0" * 3, NONE, float(v), 0.0,
                       0.0, 0, 0))
    li = len(s.leaf)
    s.leaf.append((pb, 0, b"\0" * 3))
    ni = len(s.node)
    s.node.append([NONE, 0, 0, 0, 0, li])
    return ni


def _c3(fields: dict[int, float], warm_pts, prior: int,
        block_x: int = 128) -> bytes:
    s = Seg(1, 1)
    node = _lifecycle_node(s, fields, warm_pts)
    s.uses.append(init_use(node, 25))
    f = s.template("FILL")
    s.warp(0, 0, 0, fills(f, 10))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build(), block_x=block_x, block_y=1, block_z=1,
                     function_prior_launches=prior)


def _c4(second_warp: bool, no_code: bool,
        sharing: float = 0.0) -> bytes:
    s = Seg(1, 2)
    node = s.leaf_node(26, {0: 1.0, 1: 128.0, 3: 2.0,
                            6: 256.0, 7: 1024.0, 8: 10.0, 9: 20.0,
                            10: 100.0, 11: 0.0, 12: sharing,
                            13: 1.0})
    s.uses.append(init_use(node, 26))
    f = s.template("FILL")
    s.warp(0, 0, 0, fills(f, 24))
    if second_warp:
        s.warp(0, 1, 1, fills(f, 400) + fills(f, 24))
    else:
        s.warp(0, 1, 1, [(f, None)])
    if not no_code:
        for i in range(24):
            s.code.append((i, 0, 16 * i))
        if second_warp:
            for i in range(24):
                s.code.append((400 + 24 + i, 0, 16 * i))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())




def c5_plce_partial() -> bytes:
    # 4 CTAs all carrying CTAS.sm_id (trace placement); the PLCE table
    # covers only CTAs 0/1 -> 2/3 queue for dispatch (F5)
    segs = []
    for i in range(4):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 0.0, 1: 20.0, 2: 1.0, 3: 0.0})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        s.warp(i, 0, i, fills(f, 10))
        s.cta(i, 0, i % 2, 1, 0)     # every CTA has a trace sm_id
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 1, 0, 1, 1, 0)])
    tp = b"".join(cf.TOPO.pack_row(r) for r in [topo(0), topo(1)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=2, occupancy_blocks_per_sm=1)


def c6_staggered_release() -> bytes:
    # 1 SM x 2 slots; CTA0 (10 fills, done ~9) and CTA1 (15 fills,
    # done ~14) exit staggered by less than slot_free_to_entry (20):
    # each queued CTA enters slot_free_to_entry after ITS slot's
    # release -> CTA2 at 29, CTA3 at 34 (a shared tick refilling
    # both slots at 29 would enter CTA3 at 30 = 29 + serial)
    segs = []
    for i, n in enumerate([10, 15, 10, 10]):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 0.0, 1: 20.0, 2: 1.0, 3: 1.0})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        s.warp(i, 0, i, fills(f, n))
        s.cta(i, 0, 0 if i < 2 else NONE, 1, 0)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 1, 0, 1, 0, 1),
                   plce(2, 2, 0, 0, 0, 0), plce(3, 3, 0, 0, 0, 0)])
    tp = b"".join(cf.TOPO.pack_row(r) for r in [topo(0)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=1, occupancy_blocks_per_sm=2)


def c7_frac_serial() -> bytes:
    # 1 SM x 4 slots, 8 equal CTAs: the four first-wave exits land on
    # the same cycle and refill at +20 with serial_per_cta = 1.5 —
    # the fractional accumulator gives entries at 29, 31, 32, 34
    # (ceil of 0, 1.5, 3.0, 4.5); truncating serial to 1 would give
    # 29, 30, 31, 32
    segs = []
    for i in range(8):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 0.0, 1: 20.0, 2: 1.0, 3: 1.5})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        s.warp(i, 0, i, fills(f, 10))
        s.cta(i, 0, 0 if i < 4 else NONE, 1, 0)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(i, i, 0, 1 if i < 4 else 0, 0, i if i < 4 else 0)
                   for i in range(8)])
    tp = b"".join(cf.TOPO.pack_row(r) for r in [topo(0)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=1, occupancy_blocks_per_sm=4)


def _c8(dependents: bool) -> bytes:
    # c7's shape with serial 1.0, slot_free_to_entry 2, and
    # dependents_launch_serial 1.0 (ns fields; the container
    # reference clock is 1 GHz so ns values are numerically
    # cycles here, as in c6 / c7); when `dependents` every CTA
    # carries a kind-40 (dependents_launch) instance (the extra
    # instruction pushes completion to C = 10).  The four first-wave
    # CTAs complete together: each completion charges the serializer
    #   last = max(C - serial, last) + dependents
    # so last becomes C, C+1, C+2, C+3 = 13.  The four slots release
    # at C+2 = 12; the refills enter at max(12, last+1) = 14, 15, 16,
    # 17.  Without the marker last stays 0 and the refills enter at
    # C+2..C+5 = 11..14 (C = 9 there).
    segs = []
    for i in range(8):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 0.0, 1: 2.0, 2: 1.0, 3: 1.0,
                                6: 1.0})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        if dependents:
            px = s.template("PREEXIT")
            s.warp(i, 0, i,
                   fills(f, 10)
                   + [(px, s.semantic(DEPENDENTS_LAUNCH))])
        else:
            s.warp(i, 0, i, fills(f, 10))
        s.cta(i, 0, 0 if i < 4 else NONE, 1, 0)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(i, i, 0, 1 if i < 4 else 0, 0, i if i < 4 else 0)
                   for i in range(8)])
    tp = b"".join(cf.TOPO.pack_row(r) for r in [topo(0)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=1, occupancy_blocks_per_sm=4)


def c9_slow_gpc() -> bytes:
    # 2 SMs x 1 slot, TOPO gpc 0 / 1, slow_gpc_mask = 2 (gpc 1),
    # slot 20 / slow 50, serial 1 (ns fields at the fixture's 1 GHz
    # reference clock, so numerically cycles): CTA 1 (gpc 1, 10 fills) completes
    # at ~9 and its slow slot refills CTA 2 at 9 + 50 = 59; CTA 0
    # (gpc 0, 80 fills) completes at ~79 and refills CTA 3 at
    # 79 + 20 = 99 (SM 0 stays occupied until then, so neither
    # queued CTA can take it earlier)
    segs = []
    for i, n in enumerate([80, 10, 10, 10]):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 0.0, 1: 20.0, 2: 1.0, 3: 1.0,
                                4: 50.0, 5: 2.0})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        s.warp(i, 0, i, fills(f, n))
        s.cta(i, 0, i if i < 2 else NONE, 1, 0)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 1, 0, 1, 1, 0),
                   plce(2, 2, 0, 0, 0, 0), plce(3, 3, 0, 0, 0, 0)])
    tp = b"".join(cf.TOPO.pack_row(r)
                  for r in [(0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 1, 0)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=2, occupancy_blocks_per_sm=1)


def a14_stale_poll() -> bytes:
    # failed poll issued after the phase released: its result latency
    # is 0 so the dependent instruction issues on the next cycle
    s = Seg(1, 1)
    init = s.template("INIT")
    arr = s.template("ARRIVE")
    poll = s.template("TRYWAIT", svc={1: 36.0, 2: 36.0, 7: 1.0},
                      dst=[5])
    dep = s.template("DEP", src=[5])
    s.warp(0, 0, 0,
           [(init, s.semantic(MBAR_INIT, fields=(0, 1))),
            (arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
            # wait until the phase released, then the stale poll
            (poll, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                              fields=(0, 0))),
            (dep, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())




def _mem_uses(s):
    """schema-3 leaves: l2/dram/store_l2 with byte-rate ports."""
    s.uses.append((s.leaf_node(3, {13: 30.0, 42: 128.0}),
                   0, b"\0" * 3, 2, 3, 0, 0, 0))
    s.uses.append((s.leaf_node(3, {17: 200.0, 43: 64.0}),
                   0, b"\0" * 3, 3, 3, 0, 0, 0))
    s.uses.append((s.leaf_node(3, {21: 10.0, 44: 32.0}),
                   0, b"\0" * 3, 4, 3, 0, 0, 0))


def a15_macc_kind() -> bytes:
    # access_kind=0xC0 (async copy) -> LOAD path: 128 sectors through
    # the 64 B/cycle DRAM port (~64) + the DRAM total latency, not the
    # 1-sector/cycle store port
    s = Seg(1, 1)
    _mem_uses(s)
    tma = s.template("UTMALDG")
    fill = s.template("FILL")
    macc = len(s.macc)
    s.macc.append((0, 128, 0xC0, 0, 0, 0, 0, 0))
    for i in range(128):
        s.msec.append((i * 32, 0xFFFFFFFF, 0))
    first = len(s.inst)
    s.inst.append((tma, NONE, macc, len(s.inst)))
    s.inst.append((fill, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, first, 2))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a15b_tma_leaf_excess() -> bytes:
    # UTMALDG with TMA_LOAD semantics, a completion leaf (50, L2-hot
    # issue -> complete_tx total) and 128 cold sectors: the phase
    # completes at leaf + hierarchy excess over an L2 hit (a15's ~214
    # return - L2 30), ~20 cycles after what max(leaf, hierarchy) gives
    s = Seg(1, 1)
    _mem_uses(s)
    s.mbpi.append((0, 0, 0x100, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    arr = s.template("MBAR.ARRIVE")
    tma = s.template("UTMALDG",
                     timing=((TK_TMA_COMPLETE, {0: 50.0, 1: 1.0}),))
    wt = s.template("MBAR.WAIT")
    fill = s.template("FILL")
    macc = len(s.macc)
    s.macc.append((0, 128, 0xC0, 0, 0, 0, 0, 0))
    for i in range(128):
        s.msec.append((i * 32, 0xFFFFFFFF, 0))
    first = len(s.inst)
    s.instance(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 4096, 1)))
    s.instance(tma, s.semantic(TMA_LOAD, fields=(0, 0, 0, 4096)), macc)
    s.instance(wt, s.semantic(MBAR_WAIT, fields=(0, 0)))
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 4))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a15c_tma_cold_leg() -> bytes:
    # tma_cold protocol rule: a completion leaf carrying cold_extra
    # (field 3) completes at start + hot (50) + cold_fraction x 300;
    # all 128 sectors are cold -> ~352, the hierarchy return (~214)
    # does not enter
    s = Seg(1, 1)
    _mem_uses(s)
    s.mbpi.append((0, 0, 0x100, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    arr = s.template("MBAR.ARRIVE")
    tma = s.template("UTMALDG",
                     timing=((TK_TMA_COMPLETE,
                              {0: 50.0, 1: 1.0, 3: 300.0}),))
    wt = s.template("MBAR.WAIT")
    fill = s.template("FILL")
    macc = len(s.macc)
    s.macc.append((0, 128, 0xC0, 0, 0, 0, 0, 0))
    for i in range(128):
        s.msec.append((i * 32, 0xFFFFFFFF, 0))
    first = len(s.inst)
    s.instance(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 4096, 1)))
    s.instance(tma, s.semantic(TMA_LOAD, fields=(0, 0, 0, 4096)), macc)
    s.instance(wt, s.semantic(MBAR_WAIT, fields=(0, 0)))
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 4))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a15d_tma_shared_queue() -> bytes:
    # two warps of one CTA each issue a cold-leg TMA (hot 50 + cold 300)
    # of 128 cold sectors in the same cycle; the GPU-wide DRAM read
    # service is 32 B/cycle (schema 21, 1 GHz fixture clock), so the
    # 8192 B take 256 cycles through the shared queue.  Shared-free
    # return (per-SM port start + 200): first transfer's last sector
    # leaves the port at 64 -> 264, second at 128 -> 328; actual: served
    # by the shared queue at 128 -> 328 (excess 64) and 256 -> 456
    # (excess 128).  Completions: 350 + 64 = 414 and 350 + 128 = 478 ->
    # kernel ~480.  Before the queueing excess both completed at 352
    # (a15c), whatever the shared queue did; the per-SM port wait of
    # the second transfer is not charged (it is the kind-21 service
    # start's job).
    s = Seg(1, 2)
    _mem_uses(s)
    s.uses.append((s.leaf_node(21, {3: 32.0, 4: 32.0}),
                   0, b"\0" * 3, 0, 21, 0, 0, 0))
    s.mbpi.append((0, 0, 0x100, 1, 0))
    s.mbpi.append((0, 1, 0x108, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    arr = s.template("MBAR.ARRIVE")
    tma = s.template("UTMALDG",
                     timing=((TK_TMA_COMPLETE,
                              {0: 50.0, 1: 1.0, 3: 300.0}),))
    wt = s.template("MBAR.WAIT")
    fill = s.template("FILL")
    for obj in range(2):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 128, 0xC0, 0, 0, 0, 0, 0))
        for i in range(128):
            s.msec.append((0x100000 * (obj + 1) + i * 32, 0xFFFFFFFF, 0))
        first = len(s.inst)
        s.instance(arr, s.semantic(MBAR_ARRIVE, fields=(obj, 0, 4096, 1)))
        s.instance(tma, s.semantic(TMA_LOAD, fields=(obj, 0, 0, 4096)),
                   macc)
        s.instance(wt, s.semantic(MBAR_WAIT, fields=(obj, 0)))
        s.instance(fill)
        s.warps.append((0, obj, obj, obj, first, 4))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def _a31(warps: int) -> bytes:
    """rule S fairness (store_flood replay): <warps> warps on distinct
    SMSPs of one SM each issue 12 single-sector lane STG; store_l2
    holds 4 outstanding, 8 B/cycle port -> each request drains after 4
    cycles of port service.  Once the buffer fills the grants must
    rotate across the queued warps instead of letting one warp issue
    its whole stream first."""
    s = Seg(1, warps)
    node = s.leaf_node(3, {2: 32.0, 3: 128.0, 13: 100.0, 14: 8.0,
                           21: 10.0, 22: 4.0, 42: 128.0, 44: 8.0})
    s.uses.append((node, 5, b"\0" * 3, 0, 3, 0xFFFFFFFF, 0, 0))
    stg = s.template("STG")
    fill = s.template("FILL")
    for wi in range(warps):
        first = len(s.inst)
        for _ in range(12):
            macc = len(s.macc)
            s.macc.append((len(s.msec), 1, 0x40, 0, 0, 0, 0, 0))
            s.msec.append((0x40000 * (wi + 1), 0xFFFFFFFF, wi + 1))
            s.instance(stg, None, macc)
        s.instance(fill)
        s.warps.append((0, wi, wi, wi, first, 13))
    s.cta(0, 0, 0, warps, 0)
    return container(s.build())


def a31_store_gate_fair() -> bytes:
    return _a31(2)


def a31b_store_gate_fair_w4() -> bytes:
    return _a31(4)


def a30_store_issue_interval() -> bytes:
    # storage.global.store_issue (kind 65): per-SMSP store issue interval
    # by sector count.  Two warps on SMSP 0 and 1 each issue 4 global
    # STG of 16 sectors carrying a 23-cycle row -> the 4th store of a
    # warp issues no earlier than 3 x 23 after its first, independently
    # of the other SMSP's warp; the kernel ends ~128 when the 8 stores'
    # 128 sectors have drained through the SM store port (default 1
    # sector / cycle) -- without the gate the 4th store issues at ~3
    s = Seg(1, 2)
    stg = s.template("STG", timing=((65, {0: 23.0, 1: 1.0}),))
    fill = s.template("FILL")
    for wi in range(2):
        first = len(s.inst)
        for _ in range(4):
            macc = len(s.macc)
            s.macc.append((len(s.msec), 16, 0x40, 0, 0, 0, 0, 0))
            for j in range(16):
                s.msec.append((0x40000 * (wi + 1) + j * 32, 0xFFFFFFFF, 0))
            s.instance(stg, None, macc)
        s.instance(fill)
        s.warps.append((0, wi, wi, wi, first, 5))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a18_shared_rows() -> bytes:
    # template-level interval rows: LDS with a shared acceptance row
    # (kind 6, 40 cycles, per SM) issued 4 times back-to-back by one warp
    # -> the 4th completes no earlier than 4 x 40; before the fix these
    # rows were never applied (cycles ~ 8)
    s = Seg(1, 1)
    lds = s.template("LDS", timing=((6, {0: 40.0, 1: 1.0}),))
    fill = s.template("FILL")
    first = len(s.inst)
    for _ in range(4):
        s.instance(lds)
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 5))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _a18_generic(space_flags: int) -> bytes:
    s = Seg(1, 1)
    ld = s.template("LD", timing=((6, {0: 40.0, 1: 1.0}),))
    fill = s.template("FILL")
    first = len(s.inst)
    for i in range(4):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, space_flags, 0, 0, 0, 0, 0))
        s.msec.append((i * 32, 0xFFFFFFFF, 0))
        s.instance(ld, None, macc)
    s.instance(fill)
    s.warps.append((0, 0, 0, 0, first, 5))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def a18b_shared_rows_generic() -> bytes:
    # class-bound LD (generic pointer) whose MACCs resolve to shared
    # space (flags 0x1): the kind-6 shared acceptance rows apply, as
    # for the explicit LDS template of a18 -> >= 160
    return _a18_generic(0x1)


def a18c_shared_rows_global() -> bytes:
    # same LD template but MACCs resolve to global space (flags 0x0):
    # the kind-6 rows must be skipped -> only the service latency
    return _a18_generic(0x0)


def a16_macc_reduce() -> bytes:
    # access_kind=0x100 (async reduce) -> STORE path: acceptance at
    # the 32 B/cycle store-l2 port
    s = Seg(1, 1)
    _mem_uses(s)
    red = s.template("UBLKRED")
    fill = s.template("FILL")
    macc = len(s.macc)
    s.macc.append((0, 8, 0x100, 0, 0, 0, 0, 0))
    for i in range(8):
        s.msec.append((i * 32, 0xFFFFFFFF, 0))
    first = len(s.inst)
    s.inst.append((red, NONE, macc, len(s.inst)))
    s.inst.append((fill, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, first, 2))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def c8_const_capacity() -> bytes:
    # schema-27 constant cache WITH capacity (65536 B / 64 B lines):
    # LDC c[0x0][0x200] misses + fills line 8, LDC c[0x0][0x300] misses
    # line 12 (different line kills the same-line fast path), LDC
    # c[0x0][0x210] hits resident line 8 -> 2 misses + 1 hit.  With the
    # capacity field dropped (cap=0) every access pays the miss.
    s = Seg(1, 1)
    s.uses.append((s.leaf_node(27, {0: 1.0, 1: 64.0, 2: 65536.0,
                                    3: 37.0, 4: 107.0, 5: 406.0}),
                   0, b"\0" * 3, 0, 27, 0, 0, 0))
    ldc = s.template("LDC")
    for addr in (0x200, 0x300, 0x210):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, 3))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def c8c_bank0() -> bytes:
    # optional bank0_access_cycles (schema-27 field 6 = 64): every
    # bank-0 access returns now + 64 (first and repeated, both warps);
    # bank 3 keeps miss -> hit.  Field absent elsewhere -> unchanged.
    s = Seg(1, 2)
    s.uses.append((s.leaf_node(27, {0: 1.0, 1: 64.0, 2: 65536.0,
                                    3: 37.0, 4: 107.0, 5: 406.0,
                                    6: 64.0}),
                   0, b"\0" * 3, 0, 27, 0, 0, 0))
    ldc = s.template("LDC", dst=[5])
    dep = s.template("DEP", src=[5])

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    # w0: two bank-0 accesses (first then repeated) -> both now+64
    ld(0x200); ld(0x200)
    s.warps.append((0, 0, 0, 0, 0, 2))
    # w1: bank-3 miss then in-flight re-read, DEP consumer on the
    # first value -> miss latency visible at inst 2
    b3 = 0x3 << 32
    i0 = len(s.inst)
    ld(b3 | 0x300); ld(b3 | 0x300)
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 1, 0, 0, i0, 3))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def c8d_bank0_uniform() -> bytes:
    # bank0_access_cycles (field 6 = 64, vector path) +
    # bank0_uniform_access_cycles (field 7 = 96, ULDC/LDCU path, MACC
    # flag 0x10): bank-0 vector accesses return now+64, bank-0 uniform
    # accesses return now+96, bank 3 keeps miss -> hit.
    s = Seg(1, 3)
    s.uses.append((s.leaf_node(27, {0: 1.0, 1: 64.0, 2: 65536.0,
                                    3: 37.0, 4: 107.0, 5: 406.0,
                                    6: 64.0, 7: 96.0}),
                   0, b"\0" * 3, 0, 27, 0, 0, 0))
    ldc = s.template("LDC", dst=[5])
    dep = s.template("DEP", src=[5])

    def ld(addr, uniform=False):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3 | (0x10 if uniform else 0),
                       0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    # w0: two bank-0 uniform accesses (both now+96) then a dependent
    # DEP so the latency is visible at inst 2
    ld(0x200, True); ld(0x200, True)
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, 3))
    # w1: two bank-0 vector accesses (both now+64) then dependent DEP
    i0 = len(s.inst)
    ld(0x200); ld(0x200)
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 1, 0, 0, i0, 3))
    # w2: bank-3 miss then DEP consumer -> miss latency kept
    b3 = 0x3 << 32
    i1 = len(s.inst)
    ld(b3 | 0x300); ld(b3 | 0x300)
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 2, 0, 0, i1, 3))
    s.cta(0, 0, 0, 3, 0)
    return container(s.build())


def c8b_const_inflight() -> bytes:
    # in-flight constant fill is not a hit (schema-27 leaf as c8):
    # w0 LDC 0x200 misses -> fill ready at t0+406.
    # w1 LDC 0x200 issues while the fill is still in flight: it waits
    #    for the fill ready, not just the 107-cycle hit latency.
    # w1 re-reads the same line (same-line fast path): still the fill
    #    ready, not +37.
    # w0 then misses a second line (0x300) and re-reads 0x200 after the
    #    first fill completed -> a real 107-cycle hit.
    s = Seg(1, 2)
    s.uses.append((s.leaf_node(27, {0: 1.0, 1: 64.0, 2: 65536.0,
                                    3: 37.0, 4: 107.0, 5: 406.0}),
                   0, b"\0" * 3, 0, 27, 0, 0, 0))
    ldc = s.template("LDC", dst=[5])
    dep = s.template("DEP", src=[5])

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    for a in (0x200, 0x300, 0x210):
        ld(a)
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    n0 = len(s.inst)
    ld(0x200)
    ld(0x200)
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, n0))
    s.warps.append((0, 1, 1, 1, n0, len(s.inst) - n0))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def c7_dispatch_no_enabled() -> bytes:
    # b300-style schema-24 leaf WITHOUT the enabled field -> dispatch
    # defaults to on; 4 CTAs / 2 SM / occupancy 1 / PLCE 2 -> 2 waves
    segs = []
    for i in range(4):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 0.0, 1: 20.0, 3: 0.0})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        s.warp(i, 0, i, fills(f, 10))
        s.cta(i, 0, i % 2, 1, 0)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 1, 0, 1, 1, 0)])
    tp = b"".join(cf.TOPO.pack_row(r) for r in [topo(0), topo(1)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=2, occupancy_blocks_per_sm=1)


def a17_divergent_poll() -> bytes:
    """Warp 0 alternates known-failed polls on object 0 (released by
    warp 1) and object 1 (initialized, never arrived).  The
    NANOSLEEP.SYNCS must register on both objects and wake at object
    0's release, not on object 1 (kernel-end) or the hint."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.mbpi.append((0, 1, 0x2000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 1000000.0, 1: 1.0}),))
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(0, 0, 0, 0, 0, 0))),
            (wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(1, 0, 0, 0, 0, 0))),
            (ns, s.semantic(NANOSLEEP, flags=SYNCS,
                            fields=(1000000,))),
            (wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(1, 0, 0, 0, 0, 0))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    s.warp(0, 1, 1, fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a17b_divergent_far() -> bytes:
    """Divergent poll region wider than the old bounded scans: two
    failed-poll objects 40+ instances before the NANOSLEEP.SYNCS, an
    MBAR.ARRIVE inside the region (does not close it), and more
    failed polls plus the region-closing successful wait after it.
    The wait set is {obj0, obj1} so the sleep ends at obj0's release,
    not obj1's kernel-end or the hint."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.mbpi.append((0, 1, 0x2000, 1, 0))
    s.mbpi.append((0, 2, 0x3000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 1000000.0, 1: 1.0}),))
    arr = s.template("MBAR.ARRIVE")
    poll0 = (wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(0, 0, 0, 0, 0, 0)))
    poll1 = (wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(1, 0, 0, 0, 0, 0)))
    s.warp(0, 0, 0,
           [poll0, poll0]
           + fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(2, 0, 0, 1))),
              poll1, poll1,
              (ns, s.semantic(NANOSLEEP, flags=SYNCS,
                              fields=(1000000,)))]
           + fills(f, 20) + [poll1]
           + fills(f, 25) + [poll1]
           + fills(f, 25) + [poll1]
           + [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
              (f, None)])
    s.warp(0, 1, 1, fills(f, 60)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a18_poll_retry_free() -> bytes:
    """Failed poll followed by a later same-(object,parity) check: the
    retried poll is not abandoned and pays no trywait_timeout."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    s.const_program("synchronization.mbarrier.trywait_timeout",
                    1, 4, TK_TRYWAIT, 7892.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(0, 0, 0, 0, 0, 0))),
            (f, None),
            (wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(0, 0, 0, 0, 0, 0xFFFFFFFF))),
            (f, None)])
    s.warp(0, 1, 1, fills(f, 50)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a19_poll_abandoned() -> bytes:
    """Failed poll with no later same-(object,parity) check: the
    abandoned poll holds trywait_timeout."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    s.const_program("synchronization.mbarrier.trywait_timeout",
                    1, 4, TK_TRYWAIT, 7892.0)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT")
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, flags=AFTER_READY,
                            fields=(0, 0, 0, 0, 0, 0))),
            (f, None)])
    # nobody arrives on object 0: release stays unknown, the abandoned
    # poll holds the full trywait_timeout
    s.warp(0, 1, 1, fills(f, 50))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def a20_stas_lane_objects() -> bytes:
    """STAS with a per-lane object list: the issuer's store completes
    bytes on each destination CTA's barrier (one per lane's rank)."""
    s = Seg(3, 3, cluster_shape_x=3)
    s.mbpi.append((1, 0, 0x200, 1, 0))   # obj0 on cta1: 1 arrival, 4 tx
    s.mbpi.append((2, 1, 0x200, 1, 0))   # obj1 on cta2
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 0.0)
    f = s.template("FILL")
    st = s.template("STAS")
    ex = s.template("MBAR.EXPTX")
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT")
    s.mbol.extend([(0,), (1,)])
    # object_list in slot 3: begin 0, count 2 -> (0 << 32) | 2
    s.warp(0, 0, 0,
           [(st, s.semantic(33, fields=(0, 4, 0, 2))), (f, None)])
    for c, obj in ((1, 0), (2, 1)):
        s.warp(c, 0, 0,
               [(ex, s.semantic(MBAR_EXPECT_TX,
                                fields=(obj, 0, 4, 0))),
                (arr, s.semantic(MBAR_ARRIVE,
                                 fields=(obj, 0, 0, 1))),
                (wt, s.semantic(MBAR_WAIT, fields=(obj, 0))),
                (f, None)])
    s.cta(0, 0, 0, 1, 0)
    s.cta(1, 1, 1, 1, 1)
    s.cta(2, 2, 2, 1, 2)
    return container(s.build(), cluster_x=3)




def op1_stream3() -> bytes:
    """operand_read: one warp, 16 independent 3-read FFMAs (stall 1,
    result 4) -> issue cycles 0,1,3,4,6,7,... (1.5 per inst)."""
    s = Seg(1, 1)
    s.operand_read()
    a = s.template("FFMA", control=1, svc={1: 4.0, 2: 4.0, 7: 1.0},
                   src=[10, 11, 12], dst=[20])
    b = s.template("FFMA", control=1, svc={1: 4.0, 2: 4.0, 7: 1.0},
                   src=[13, 14, 15], dst=[21])
    s.warp(0, 0, 0, [(a, None), (b, None)] * 8)
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _op2_dep3(w: int) -> bytes:
    """operand_read: W warps on one SMSP, 16-step 3-read dependent
    chains (stall 4, result 4) -> 2*W cycles per step."""
    s = Seg(1, w)
    s.operand_read()
    f = s.template("FFMA", control=4, svc={1: 4.0, 2: 4.0, 7: 1.0},
                   src=[10, 11, 12], dst=[10])
    for i in range(w):
        s.warp(0, i, 0, fills(f, 16))
    s.cta(0, 0, 0, w, 0)
    return container(s.build())


def op2_dep3_w2() -> bytes:
    return _op2_dep3(2)


def op2_dep3_w3() -> bytes:
    return _op2_dep3(3)


def op2_dep3_w4() -> bytes:
    return _op2_dep3(4)


def op3_served() -> bytes:
    """operand_read: one warp, 16 FFMAs whose a/b rows are all
    reuse-served (flags bit0) -> 1 per cycle."""
    s = Seg(1, 1)
    s.operand_read()
    a = s.template("FFMA", control=1, svc={1: 4.0, 2: 4.0, 7: 1.0},
                   src=[(10, 1), (11, 1), (12, 1)], dst=[20])
    b = s.template("FFMA", control=1, svc={1: 4.0, 2: 4.0, 7: 1.0},
                   src=[(13, 1), (14, 1), (15, 1)], dst=[21])
    s.warp(0, 0, 0, [(a, None), (b, None)] * 8)
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def op4_reads2() -> bytes:
    """operand_read: one warp, 16 2-read FFMAs -> 1 per cycle."""
    s = Seg(1, 1)
    s.operand_read()
    a = s.template("FFMA", control=1, svc={1: 4.0, 2: 4.0, 7: 1.0},
                   src=[10, 11], dst=[20])
    b = s.template("FFMA", control=1, svc={1: 4.0, 2: 4.0, 7: 1.0},
                   src=[12, 13], dst=[21])
    s.warp(0, 0, 0, [(a, None), (b, None)] * 8)
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _op5(w: int) -> bytes:
    """operand_read: [3-read producer, served consumer] pairs; the
    served consumer must start on a fresh cycle."""
    s = Seg(1, w)
    s.operand_read()
    prod = s.template("FFMA", control=1, svc={1: 4.0, 2: 4.0, 7: 1.0},
                      src=[10, 11, 12], dst=[10])
    cons = s.template("FFMA", control=3, svc={1: 4.0, 2: 4.0, 7: 1.0},
                      src=[(10, 1), (11, 1)], dst=[10])
    body = [(prod, None), (cons, None)] * 8
    for i in range(w):
        s.warp(0, i, 0, body)
    s.cta(0, 0, 0, w, 0)
    return container(s.build())


def op5_pair_w1() -> bytes:
    return _op5(1)


def op5_pair_w2() -> bytes:
    return _op5(2)


FIXTURES = {
    "l1_one_warp": l1_one_warp,
    "l1_two_warps": l1_two_warps,
    "a1_basic": a1_basic,
    "a3_parity1": a3_parity1,
    "a4_poll": lambda: _a4(True, False),
    "a4_poll_notimeout": lambda: _a4(False, False),
    "a5_lookahead": lambda: _a4(True, True),
    "a6_deferred": a6_deferred,
    "a7_init_zero": a7_init_zero,
    "a7_arrive_zero": a7_arrive_zero,
    "a8_one_gen": a8_one_gen,
    "a8_two_gens": a8_two_gens,
    "a8_exit": a8_exit,
    "a9_cluster": a9_cluster,
    "a10_nanosleep": a10_nanosleep,
    "a10_plain_after_wait": a10_plain_after_wait,
    "a10_syncs": a10_syncs,
    "a10_past_release": a10_past_release,
    "a10_cap": a10_cap,
    "a11_visibility": a11_visibility,
    "a12_deadlock": a12_deadlock,
    "a13_initial_phase": lambda: _a13(1),
    "a13_initial_phase0": lambda: _a13(0),
    "a21_init_mbol": a21_init_mbol,
    "a22_reinit_idle": a22_reinit_idle,
    "a22b_reinit_busy": a22b_reinit_busy,
    "a23_complete_tx": a23_complete_tx,
    "a23b_complete_tx_zero": a23b_complete_tx_zero,
    "b1_bulk_group": b1_bulk_group,
    "b1_bulk_group_n1": b1_bulk_group_n1,
    "b27_reduce_store": lambda: _b27_reduce_store(True),
    "b27b_reduce_store_norow": lambda: _b27_reduce_store(False),
    "b2_cpasync_arrives": b2_cpasync_arrives,
    "b2b_cpasync_transcnt": b2b_cpasync_transcnt,
    "b3_multicast_tma": lambda: _b3(True, False),
    "b3b_no_mbob": lambda: _b3(False, False),
    "b3c_late_arrive": lambda: _b3(True, True),
    "b4_service_start": b4_service_start,
    "b5_descriptor": b5_descriptor,
    "b6a_from": b6a_from,
    "b6b_initial": b6b_initial,
    "b6c_any": b6c_any,
    "b6d_chain": b6d_chain,
    "b6e_count": b6e_count,
    "b6f_poll": b6f_poll, "b6l_poll_free": b6l_poll_free,
    "b6m_poll_region": b6m_poll_region,
    "b6n_poll_region_mixed": b6n_poll_region_mixed,
    "b6o_poll_region_macc": b6o_poll_region_macc,
    "b6p_poll_region_constant": b6p_poll_region_constant,
    "b6g_unpaired": b6g_unpaired,
    "b6h_noop": b6h_noop,
    "b6i_no_writer": b6i_no_writer,
    "b6j_no_writer_partial": b6j_no_writer_partial,
    "b6k_unknown_writer": b6k_unknown_writer,
    "b7_envelope": b7_envelope,
    "b8_two_segments": b8_two_segments,
    "b10_fence_floor": b10_fence_floor,
    "b16_fence_drain": b16_fence_drain,
    "b17_fence_drain_clean": b17_fence_drain_clean,
    "b18_fence_drain_second": b18_fence_drain_second,
    "b19_fence_drain_twice": b19_fence_drain_twice,
    "b20_smsp_accept": b20_smsp_accept,
    "b21_smsp_accept_sm": b21_smsp_accept_sm,
    "b22_smsp_accept_same": b22_smsp_accept_same,
    "b23_smsp_accept_nonshared": b23_smsp_accept_nonshared,
    "b24_smsp_membar": b24_smsp_membar,
    "b25_membar_clean": b25_membar_clean,
    "b26_smsp_accept_frac": b26_smsp_accept_frac,
    "b10b_dest_result": b10b_dest_result,
    "b10c_store_result": b10c_store_result,
    "b9_drain": b9_drain,
    "c1_waves": lambda: _c1(0, 20, 0, 1, 1),
    "c1b_serial": lambda: _c1(0, 20, 5, 1, 1),
    "c1c_first_wave": lambda: _c1(3, 20, 0, 1, 1),
    "c1d_disabled": lambda: _c1(0, 20, 0, 0, 1),
    "c1e_no_occupancy": lambda: _c1(0, 20, 0, 1, 0),
    "c1f_trace_placed": c1f_trace_placed,
    "c2_cluster": lambda: _c2(True),
    "c2b_no_topo": lambda: _c2(False),
    "c2c_gpc_no_topo": lambda: _gpc("none"),
    "c2d_gpc_row_no_gpc": lambda: _gpc("rows"),
    "c3_envelope_linear": lambda: _c3(
        {}, [(1, 646), (4, 645), (12, 657)], 0),
    "c3_blackwell": lambda: _c3({2: 50.0}, None, 0),
    "c3_dual_sm": lambda: _c3({2: 50.0, 3: 30.0}, None, 0),
    "c4_fetch": lambda: _c4(False, False),
    "c5_plce_partial": c5_plce_partial,
    "c6_staggered_release": c6_staggered_release,
    "c7_frac_serial": c7_frac_serial,
    "c8_dependents_launch": lambda: _c8(True),
    "c8_no_marker": lambda: _c8(False),
    "c9_slow_gpc": c9_slow_gpc,
    "a14_stale_poll": a14_stale_poll,
    "a15_macc_kind": a15_macc_kind,
    "a15b_tma_leaf_excess": a15b_tma_leaf_excess,
    "a15c_tma_cold_leg": a15c_tma_cold_leg,
    "a15d_tma_shared_queue": a15d_tma_shared_queue,
    "a30_store_issue_interval": a30_store_issue_interval,
    "a31_store_gate_fair": a31_store_gate_fair,
    "a31b_store_gate_fair_w4": a31b_store_gate_fair_w4,
    "a32_tma_uncut": a32_tma_uncut,
    "a32b_tma_cut_extra": a32b_tma_cut_extra,
    "a32c_tma_cut_service": a32c_tma_cut_service,
    "a32d_tma_cut_service32": a32d_tma_cut_service32,
    "a32e_tma_uncut4": a32e_tma_uncut4,
    "a18_shared_rows": a18_shared_rows,
    "a18b_shared_rows_generic": a18b_shared_rows_generic,
    "a18c_shared_rows_global": a18c_shared_rows_global,
    "a16_macc_reduce": a16_macc_reduce,
    "a17_divergent_poll": a17_divergent_poll,
    "a17b_divergent_far": a17b_divergent_far,
    "a18_poll_retry_free": a18_poll_retry_free,
    "a19_poll_abandoned": a19_poll_abandoned,
    "a20_stas_lane_objects": a20_stas_lane_objects,
    "a24_branch_envelope": a24_branch_envelope,
    "a25_control_stall": a25_control_stall,
    "a26_failed_poll_result": a26_failed_poll_result,
    "a27_read_barrier_accept": a27_read_barrier_accept,
    "a27b_read_barrier_issue": a27b_read_barrier_issue,
    "a28_read_barrier_global": a28_read_barrier_global,
    "a28b_read_barrier_merged": a28b_read_barrier_merged,
    "a28c_read_barrier_store": a28c_read_barrier_store,
    "m_syncs_wake": m_syncs_wake,
    "m_cpasync_anchor": m_cpasync_anchor,
    "m_l1_port": m_l1_port,
    "m_l1_port_off": m_l1_port_off,
    "m_control_yield": m_control_yield,
    "c7_dispatch_no_enabled": c7_dispatch_no_enabled,
    "c8_const_capacity": c8_const_capacity,
    "c8b_const_inflight": c8b_const_inflight,
    "c8c_bank0": c8c_bank0,
    "c8d_bank0_uniform": c8d_bank0_uniform,
    "c4b_shared_l0": lambda: _c4(True, False),
    "c4c_no_code": lambda: _c4(False, True),
    "c4d_sharing_smsp": lambda: _c4(True, False, sharing=1.0),
    "op1_stream3": op1_stream3,
    "op2_dep3_w2": op2_dep3_w2,
    "op2_dep3_w3": op2_dep3_w3,
    "op2_dep3_w4": op2_dep3_w4,
    "op3_served": op3_served,
    "op4_reads2": op4_reads2,
    "op5_pair_w1": op5_pair_w1,
    "op5_pair_w2": op5_pair_w2,
}


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in FIXTURES.items():
        (out_dir / f"{name}.job").write_bytes(fn())


if __name__ == "__main__":
    main()

