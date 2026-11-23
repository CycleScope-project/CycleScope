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


def r2_bsync_floor() -> bytes:
    """branch_basic v2 R2: a BSYNC joins the floor family -- the FILL
    behind it issues at BSYNC issue + result latency (13).  The real
    template names its B register as a destination (B0 = 1024), so
    the floor must apply with destination_count = 1."""
    s = Seg(1, 1)
    b = s.template("BSYNC", svc={1: 13.0, 2: 13.0}, dst=[1024])
    f = s.template("FILL")
    s.warp(0, 0, 0, [(b, None), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def r2_bsync_floor_stall() -> bytes:
    """branch_basic v2 R2: the floor is a max with the stall, not a
    sum -- a stall-5 BSYNC with result 13 still lands the FILL at
    +13."""
    s = Seg(1, 1)
    b = s.template("BSYNC", control=5, svc={1: 13.0, 2: 13.0},
                   dst=[1024])
    f = s.template("FILL")
    s.warp(0, 0, 0, [(b, None), (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def sl_control_branch() -> bytes:
    """stall ledger: a BRA with stall 5 bound to a control_envelope
    10 -- the 9-cycle wait behind it splits into 4 cycles
    control_floor (the stall field) + 5 cycles control:branch."""
    s = Seg(1, 1)
    bra = s.template("BRA", control=5)
    f = s.template("FILL")
    s.warp(0, 0, 0, [(bra, None), (f, None)])
    node = s.leaf_node(5, {4: 10.0})
    s.uses.append((node, 2, b"\0" * 3, 0, 5, 0x1, 0, 0))
    for i in range(2):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def sl_control_yield() -> bytes:
    """stall ledger: a YIELD with stall 5 and service result 50 --
    the wait behind it splits into 4 cycles control_floor + 45
    cycles control:yield."""
    s = Seg(1, 1)
    y = s.template("YIELD", control=5, svc={1: 50.0, 2: 50.0})
    f = s.template("FILL")
    s.warp(0, 0, 0, [(y, None), (f, None)])
    for i in range(2):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def sl_control_stall() -> bytes:
    """stall ledger: a FILL with stall 5 only -- the wait behind it
    stays control_floor, no control:* reason appears."""
    s = Seg(1, 1)
    f5 = s.template("FILL", control=5)
    f = s.template("FILL")
    s.warp(0, 0, 0, [(f5, None), (f, None)])
    for i in range(2):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def r2_bssy_no_floor() -> bytes:
    """branch_basic v2 R1: a BSSY is not in the floor family -- the
    FILL behind it issues one cycle later whatever the service
    result latency."""
    s = Seg(1, 1)
    b = s.template("BSSY", svc={1: 13.0, 2: 13.0})
    f = s.template("FILL")
    s.warp(0, 0, 0, [(b, None), (f, None)])
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


def ir1_no_register_tracking() -> bytes:
    """issue_rules v1 R1: a producer without a write barrier (result
    latency 10) followed by a stall-1 reader of its destination
    register: the reader issues the next cycle (hardware computes
    with the stale value; the replay does not track registers)."""
    s = Seg(1, 1)
    prod = s.template("VIADD", dst=[5], control=1,
                      svc={1: 10.0, 2: 10.0, 7: 1.0})
    rd = s.template("VIADD", src=[5], control=1)
    s.warp(0, 0, 0, [(prod, None), (rd, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def ir2_barrier_latest() -> bytes:
    """issue_rules v1 R2: two producers on barrier 1 with data_ready
    20 / 40 apart; the wait-mask reader is held until the later of
    the two releases."""
    s = Seg(1, 1)
    p1 = s.template("LDG", scoreboard_write=1, control=1,
                    svc={1: 20.0, 2: 20.0, 7: 1.0})
    p2 = s.template("LDG", scoreboard_write=1, control=1,
                    svc={1: 40.0, 2: 40.0, 7: 1.0})
    rd = s.template("FILL", scoreboard_wait_mask=0x2, control=1)
    s.warp(0, 0, 0, [(p1, None), (p2, None), (rd, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _ir3(load_stall: int, gap: int) -> bytes:
    """issue_rules v1 R3: a write-barrier producer followed by a
    wait-mask reader.  gap = stall-1 instructions between them: the
    raise at t is invisible to an issue at t + 1 (gap 0 with the
    reader's own floor met at t + 1) and visible from t + 2."""
    s = Seg(1, 1)
    ld = s.template("LDC", dst=[5], scoreboard_write=1,
                    control=load_stall,
                    svc={1: 50.0, 2: 50.0, 7: 1.0})
    nop = s.template("NOP", control=1)
    rd = s.template("FILL", src=[5], scoreboard_wait_mask=0x2,
                    control=1)
    s.warp(0, 0, 0,
           [(ld, None)] + [(nop, None)] * gap + [(rd, None)])
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def ir3_load_s1() -> bytes:
    return _ir3(1, 0)


def ir4_yield_stall() -> bytes:
    """issue_rules v2 Y1: a yield hint (TMPL.control bit4) with a
    stall count of 1 holds the warp's next issue until issue + 2;
    yield + stall 4 keeps +4; stall 1 without the yield bit is +1."""
    s = Seg(1, 1)
    fy1 = s.template("FILL", control=0x11)
    f1 = s.template("FILL", control=1)
    fy4 = s.template("FILL", control=0x14)
    f0 = s.template("FILL")
    s.warp(0, 0, 0, [(fy1, None), (f1, None), (fy4, None),
                   (f1, None), (f0, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def ir3_load_s2() -> bytes:
    return _ir3(2, 0)


def ir3_load_s1g1() -> bytes:
    return _ir3(1, 1)


def sl_waits() -> bytes:
    """Stall-ledger fixture: a barrier wait behind a global load
    (barrier:LDG:*, issue_rules v1), an execution-unit wait on
    resource 4 (unit:xu_smsp in the common resource table), a
    NANOSLEEP suspension (suspended:sleep) and an mbarrier wait,
    plus code addresses for every instance."""
    s = Seg(1, 5)
    # L2/DRAM latency 300 (res_f3's level): the consumer waits on the
    # load's write barrier -> a barrier:LDG:* residency row (the NOP
    # between them is what R3 needs for the raise to be visible)
    node = s.leaf_node(3, {13: 300.0, 17: 300.0})
    s.uses.append((node, 5, b"\0" * 3, 0, 3, 0xFFFFFFFF, 0, 0))
    s.mbpi.append((0, 0, 0x1000, 2, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    # RESOURCE-schema row defining resource id 4 (smsp scope,
    # initiation interval 8) -> the second XU issue waits on the unit
    node = s.leaf_node(2, {0: 0.0, 1: 1.0, 2: 0.0, 3: 8.0})
    s.uses.append((node, 4, b"\0" * 3, 4, 2, 0xF, 0, 0))
    ldg = s.template("LDG", dst=[1], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    use = s.template("USE", src=[1], scoreboard_wait_mask=0x2)
    xu = s.template("MUFU", svc={1: 18.0, 7: 8.0, 10: 4.0})
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 100.0, 1: 1.0}),))
    f = s.template("FILL")
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT")
    # warp 0: global load -> dependent USE (barrier wait on the load)
    s.macc.append((0, 1, 0x0, 0, 0, 0, 0, 0))
    s.msec.append((0x4000, 0xFFFFFFFF, 0xFFFFFFFF))
    first = len(s.inst)
    s.instance(ldg, None, 0)
    s.instance(nop)
    s.instance(use)
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 4))
    # warp 1: two XU ops on one unit with interval 8 -> unit wait
    s.warp(0, 1, 1, [(xu, None), (xu, None), (f, None)])
    # warp 2: nanosleep then a fill -> suspended:sleep
    s.warp(0, 2, 2,
           [(ns, s.semantic(NANOSLEEP, fields=(100,))), (f, None)])
    # warps 3/4: mbarrier arrive/wait pair (a1_basic shape)
    s.warp(0, 3, 3,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    s.warp(0, 4, 0,
           fills(f, 20)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 5, 0)
    return container(s.build())


def sl_loop() -> bytes:
    """Stall-ledger transitions: a 3-instruction body replayed four
    times per warp -> the back edge (0x1020 -> 0x1000) counts 3 per
    warp."""
    s = Seg(1, 2)
    f = s.template("FILL")
    s.warp(0, 0, 0, fills(f, 12))
    s.warp(0, 1, 1, fills(f, 12))
    for i in range(24):
        s.code.append((i, 0, 0x1000 + (i % 3) * 0x10))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def sl_issue_slot() -> bytes:
    """Two always-ready warps pinned to one SMSP (slots 0 and 4 ->
    smsp 0): each cycle one issues and the other accrues issue_slot
    at its next instruction's address."""
    s = Seg(1, 2)
    fy = s.template("FILL", control=0x10)  # yield bit -> alternation
    s.warp(0, 0, 0, fills(fy, 6))
    s.warp(0, 1, 4, fills(fy, 6))
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def sl_control_floor() -> bytes:
    """A stall-count-4 control word: the span after its issue is
    control_floor at the following instruction's address for
    stall - 1 = 3 cycles."""
    s = Seg(1, 1)
    f4 = s.template("FILL", control=4)
    f1 = s.template("FILL", control=1)
    s.warp(0, 0, 0, [(f4, None), (f1, None), (f1, None), (f1, None)])
    for i in range(4):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def sl_due() -> bytes:
    """A unit wait with a finite retry time whose warp is never
    reached once due: warp0's second XU is blocked [1, 8) under
    unit:xu_smsp, then accrues issue_slot at the same address while
    warp1 (same smsp, never yields) keeps the issue slot."""
    s = Seg(1, 2)
    # RESOURCE-schema row: resource id 4 (smsp scope, initiation
    # interval 8) — the second XU issue waits on the unit until 8
    node = s.leaf_node(2, {0: 0.0, 1: 1.0, 2: 0.0, 3: 8.0})
    s.uses.append((node, 4, b"\0" * 3, 4, 2, 0xF, 0, 0))
    xu = s.template("MUFU", svc={1: 18.0, 7: 8.0, 10: 4.0})
    f = s.template("FILL")
    s.warp(0, 0, 0, [(xu, None), (xu, None), (f, None)])
    s.warp(0, 1, 4, fills(f, 20))
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def sl_cta_barrier() -> bytes:
    """CTA barrier key convention: warp0's BAR.SYNC issues at 0 and
    its wait cycles are charged at the next instruction's address
    (0x1010), not at the barrier instruction itself."""
    s = Seg(1, 2)
    bar = _cta_bar_template(s)
    f = s.template("FILL")
    s.warp(0, 0, 0,
           [(bar, s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                             fields=(0, 0, 0)))]
           + fills(f, 5))
    s.warp(0, 1, 1,
           fills(f, 10)
           + [(bar, s.semantic(CTA_WAIT, flags=ALSO_ARRIVES,
                               fields=(0, 0, 0))),
              (f, None)])
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def sl_cluster_barrier() -> bytes:
    """Cluster barrier key convention: warp0's CLUSTER.WAIT waits for
    the other CTA's arrive; the wait cycles are charged at the wait
    instruction's own address (0x1060), not the instruction after it."""
    s = Seg(2, 4, cluster_shape_x=2)
    s.guarded_program("synchronization.cluster.release", 3, 4,
                      TK_CL_REL,
                      [(AXID["expected_arrivals"], 0, 2, 2, 20.0),
                       (AXID["expected_arrivals"], 0, 1, 1, 5.0)])
    f = s.template("FILL")
    arr = s.template("CLUSTER.ARRIVE")
    wt = s.template("CLUSTER.WAIT")
    for c in range(2):
        for k in range(2):
            s.warp(c, k, k,
                   fills(f, 10 * (2 * c + k))
                   + [(arr, s.semantic(CL_ARRIVE, fields=(0, 0)))]
                   + fills(f, 5)
                   + [(wt, s.semantic(CL_WAIT, fields=(0, 0))),
                      (f, None)])
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 2, 0)
    s.cta(1, 1, 1, 2, 2)
    return container(s.build(), cluster_x=2)


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


# ---- kernel_boundary K2/K3/K4 ----


def kb1_active_cycles() -> bytes:
    # K2: 1 SM x 2 slots; CTAs 0 (10 fills) / 1 (15 fills) first wave,
    # queued CTAs 2/3 refill at +20 serial 1.  The SM is active
    # [0, ~14) and again [~29, ~44); the 15-cycle gap does not count,
    # so cycles ~= active union length (~29), not the last
    # completion (~44)
    segs = []
    for i, n in enumerate([10, 15, 10, 10]):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 0.0, 1: 20.0, 2: 1.0, 3: 1.0})
        s.uses.append(init_use(node, 24))
        f = s.template("FILL")
        s.warp(i, 0, i, fills(f, n))
        s.cta(i, 0, i if i < 2 else NONE, 1, 0)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 1, 0, 1, 0, 1),
                   plce(2, 2, 0, 0, 0, 0), plce(3, 3, 0, 0, 0, 0)])
    tp = b"".join(cf.TOPO.pack_row(r) for r in [topo(0)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=1, occupancy_blocks_per_sm=2)


def kb2_first_line() -> bytes:
    # K3: launch_first_line 200 + stream_l1 10 over a 32-line launch
    # window: the first line arrives at entry + 200; a warp entering
    # while it is in flight (CTA 1, entry 120) waits the same
    # arrival; a CTA entering after the arrival (CTA 2, entry 240)
    # pays only its own stream slot; its demand on launch line 20 —
    # inside the window but not yet streamed to by the chain —
    # arrives on its own L1 slot (entry + 10), not entry + 200 +
    # 20 x 10.  The next 128 B line of line 0 is entry + 200 + 10
    segs = []
    for i in range(3):
        s = Seg(1, 1)
        node = s.leaf_node(24, {0: 120.0, 1: 20.0, 2: 1.0, 3: 1.0})
        s.uses.append(init_use(node, 24))
        fn = s.leaf_node(26, {0: 1.0, 1: 128.0, 3: 2.0, 6: 4096.0,
                              7: 65536.0, 8: 10.0, 9: 20.0,
                              10: 100.0, 11: 4096.0, 12: 0.0,
                              13: 1.0, 15: 200.0})
        s.uses.append(init_use(fn, 26))
        f = s.template("FILL")
        s.warp(i, 0, i, fills(f, 10))
        base = 2560 if i == 2 else 0
        for j in range(10):
            s.code.append((j, 0, base + 16 * j))
        s.cta(i, 0, i if i < 3 else NONE, 1, 0)
        segs.append(s.build())
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 1, 0, 1, 0, 1),
                   plce(2, 2, 0, 1, 0, 2)])
    tp = b"".join(cf.TOPO.pack_row(r) for r in [topo(0)])
    return container_multi(segs, placements=pl, topo=tp,
                           sm_count=1, occupancy_blocks_per_sm=3)


def kb3_exit() -> bytes:
    # K4: lifecycle fields 4/5 (branch 8 / tail 7): the last
    # instruction is EXIT (control stall 5) issued at t ->
    # done = t + 5 + 8 + 7; ledger control_floor 4, exit_branch 8,
    # exit_drain 7
    s = Seg(1, 1)
    node = _lifecycle_node(s, {4: 8.0, 5: 7.0}, None)
    s.uses.append(init_use(node, 25))
    f = s.template("FILL")
    ex = s.template("EXIT", control=5)
    s.warp(0, 0, 0, fills(f, 3) + [(ex, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def kb3b_exit_drain() -> bytes:
    # same EXIT shape but a cp.async group completing at first
    # issue + 50 is still in flight when EXIT issues: done =
    # max(t + 5 + 8, ~50) + 7
    s = Seg(1, 1)
    node = _lifecycle_node(s, {4: 8.0, 5: 7.0}, None)
    s.uses.append(init_use(node, 25))
    _mem_leaf(s)
    iss = s.template("CP.ASYNC", timing=((39, {0: 50.0, 1: 1.0}),))
    com = s.template("CP.COMMIT")
    ex = s.template("EXIT", control=5)
    macc = len(s.macc)
    s.macc.append((len(s.msec), 16, 0, 0, 0, 0, 0, 0))
    for j in range(16):
        s.msec.append((j * 128, 0xFFFFFFFF, 0))
    s.warp(0, 0, 0, [(iss, s.semantic(CP_ISSUE, fields=(0, 0, NONE))),
                   (com, s.semantic(CP_COMMIT,
                                    fields=(0, 0, NONE, 0))),
                   (ex, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def kb4_drain_starve() -> bytes:
    # drain-settle starvation (CUF-C84B-SPARSE-S05): warp 1 (slot 0)
    # retires early into an async drain targeting ~issue+50; sibling
    # warp 0 (slot 4, same SMSP 0) yields only while warp 1 still has
    # instructions, then stops yielding so the single issue slot is
    # spent ahead of warp 1 in every sweep.  Without the fix the
    # exhausted warp is never queued for the post-sweep settle once
    # the sibling exhausts, the drain wake lands with nothing armed
    # to re-evaluate it, and the run deadlocks when the queue
    # empties; with the fix it is pushed every sweep and completes
    # at its drain target
    s = Seg(1, 1)
    _mem_leaf(s)
    fy = s.template("FILL", control=16)   # yield bit: rotate cursor
    fn = s.template("FILL")               # no yield: hold cursor first
    iss = s.template("CP.ASYNC", timing=((39, {0: 50.0, 1: 1.0}),))
    com = s.template("CP.COMMIT")
    ex = s.template("EXIT")
    macc = len(s.macc)
    s.macc.append((len(s.msec), 16, 0, 0, 0, 0, 0, 0))
    for j in range(16):
        s.msec.append((j * 128, 0xFFFFFFFF, 0))
    s.warp(0, 0, 4, fills(fy, 10) + fills(fn, 110) + [(ex, None)])
    s.warp(0, 1, 0, [(iss, s.semantic(CP_ISSUE, fields=(0, 0, NONE))),
                     (com, s.semantic(CP_COMMIT,
                                      fields=(0, 0, NONE, 0))),
                     (ex, None)])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


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
    ldc = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    # w0: two bank-0 accesses (first then repeated) -> both now+64
    ld(0x200); ld(0x200)
    s.warps.append((0, 0, 0, 0, 0, 2))
    # w1: bank-3 miss then in-flight re-read, DEP consumer on the
    # first value -> miss latency visible at the last instance
    b3 = 0x3 << 32
    i0 = len(s.inst)
    ld(b3 | 0x300); ld(b3 | 0x300)
    s.inst.append((nop, NONE, NONE, len(s.inst)))
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 1, 0, 0, i0, 4))
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
    ldc = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)

    def ld(addr, uniform=False):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3 | (0x10 if uniform else 0),
                       0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    # w0: two bank-0 uniform accesses (both now+96) then a dependent
    # DEP so the latency is visible at the last instance
    ld(0x200, True); ld(0x200, True)
    s.inst.append((nop, NONE, NONE, len(s.inst)))
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, 4))
    # w1: two bank-0 vector accesses (both now+64) then dependent DEP
    i0 = len(s.inst)
    ld(0x200); ld(0x200)
    s.inst.append((nop, NONE, NONE, len(s.inst)))
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 1, 0, 0, i0, 4))
    # w2: bank-3 miss then DEP consumer -> miss latency kept
    b3 = 0x3 << 32
    i1 = len(s.inst)
    ld(b3 | 0x300); ld(b3 | 0x300)
    s.inst.append((nop, NONE, NONE, len(s.inst)))
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 2, 0, 0, i1, 4))
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
    ldc = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    def dep_after():
        s.inst.append((nop, NONE, NONE, len(s.inst)))
        s.inst.append((dep, NONE, NONE, len(s.inst)))

    for a in (0x200, 0x300, 0x210):
        ld(a)
    dep_after()
    n0 = len(s.inst)
    ld(0x200)
    ld(0x200)
    dep_after()
    s.warps.append((0, 0, 0, 0, 0, n0))
    s.warps.append((0, 1, 1, 1, n0, len(s.inst) - n0))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def _c8e(with_hold: bool, with_resident: bool) -> bytes:
    # uniform constant cold-hold (kernel_start v2): schema-27 fields
    # 8 (bank-0 hold) / 9 (other banks); field 10 (resident hold,
    # kernel_start v3) replaces the same_line / hit latency on uniform
    # re-reads when the cold rule is enabled.  with_hold=False drops
    # 8/9/10 (rule off -> miss/in-flight behaviour identical to
    # today); with_resident=False drops only 10.
    # w0: cold bank-0 line (hold 119), then a cold bank-3 line
    #     (hold 463) serial; the bank-0 fill has completed by then,
    #     so the two uniform re-reads of it hit the resident rule
    #     (field 10 present: result at issue) or the old hit (107) /
    #     same-line (37) latencies; a trailing vector LDC on the same
    #     resident line keeps 37 either way.
    # w1: same cold bank-0 line as w0, different smsp -> held until
    #     the same fill-ready cycle.
    # w2: cold bank-3 line -> 463.
    fields = {0: 1.0, 1: 64.0, 2: 65536.0, 3: 37.0, 4: 107.0, 5: 406.0}
    if with_hold:
        fields[8] = 119.0
        fields[9] = 463.0
        if with_resident:
            fields[10] = 0.0
    s = Seg(1, 3)
    s.uses.append((s.leaf_node(27, fields), 0, b"\0" * 3, 0, 27,
                   0, 0, 0))
    ldc = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)
    b3 = 0x3 << 32

    def ld(addr, uniform):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1,
                       3 | (0x10 if uniform else 0), 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    def dep_after():
        s.inst.append((nop, NONE, NONE, len(s.inst)))
        s.inst.append((dep, NONE, NONE, len(s.inst)))

    ld(0x200, True)
    ld(b3 | 0x300, True)
    ld(0x210, True)
    ld(0x220, True)
    dep_after()
    ld(0x230, False)
    dep_after()
    s.warps.append((0, 0, 0, 0, 0, 9))
    i0 = len(s.inst)
    ld(0x200, True)
    dep_after()
    s.warps.append((0, 1, 0, 1, i0, 3))
    i1 = len(s.inst)
    ld(b3 | 0x400, True)
    dep_after()
    s.warps.append((0, 2, 0, 2, i1, 3))
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 3, 0)
    return container(s.build())


def c8e_const_hold() -> bytes:
    return _c8e(True, True)


def c8f_const_hold_off() -> bytes:
    # same job without any of the three fields: the engine must
    # produce exactly today's cycles (miss on every cold access,
    # in-flight hits wait for fill-ready as a data latency)
    return _c8e(False, False)


def c8g_const_hold_noresident() -> bytes:
    # fields 8/9 kept, field 10 dropped: cold lines are still held at
    # issue, but uniform re-reads of a resident line are charged the
    # pre-v3 hit (107) / same-line (37) latencies again
    return _c8e(True, False)


def c8h_vbank0_cold() -> bytes:
    # vector bank-0 cold latency (kernel_start v4): schema-27 field
    # 11 with field 6 (flat access 30) and the uniform cold-hold rule
    # (8/9/10) on, so the resident set is shared between paths.
    # w0: vector LDC on cold bank-0 line A -> fill ready after 153;
    #     a second vector read of line A hits (+30).
    # w1: vector LDC on line A while the fill is in flight -> reader
    #     ready at the same cycle as w0's.
    # w2: uniform LDC on cold bank-0 line B (held to fill-ready 119),
    #     then a vector LDC on line B -> resident hit (+30).
    fields = {0: 1.0, 1: 64.0, 2: 65536.0, 3: 37.0, 4: 107.0, 5: 406.0,
              6: 30.0, 8: 119.0, 9: 463.0, 10: 0.0, 11: 153.0}
    s = Seg(1, 3)
    s.uses.append((s.leaf_node(27, fields), 0, b"\0" * 3, 0, 27,
                   0, 0, 0))
    ldc = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)

    def ld(addr, uniform):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1,
                       3 | (0x10 if uniform else 0), 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    def dep_after():
        s.inst.append((nop, NONE, NONE, len(s.inst)))
        s.inst.append((dep, NONE, NONE, len(s.inst)))

    ld(0x200, False)
    dep_after()
    ld(0x220, False)
    dep_after()
    s.warps.append((0, 0, 0, 0, 0, 6))
    i0 = len(s.inst)
    ld(0x240, False)
    dep_after()
    s.warps.append((0, 1, 0, 1, i0, 3))
    i1 = len(s.inst)
    ld(0x400, True)
    dep_after()
    ld(0x420, False)
    dep_after()
    s.warps.append((0, 2, 0, 2, i1, 6))
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 3, 0)
    return container(s.build())


def c8i_vbank0_cold_off() -> bytes:
    # field 11 dropped (fields 6/8/9/10 kept): vector bank-0 reads
    # keep the flat 30 latency and leave the resident set alone --
    # the uniform read of the same line right after is still cold-
    # held (~119)
    fields = {0: 1.0, 1: 64.0, 2: 65536.0, 3: 37.0, 4: 107.0, 5: 406.0,
              6: 30.0, 8: 119.0, 9: 463.0, 10: 0.0}
    s = Seg(1, 1)
    s.uses.append((s.leaf_node(27, fields), 0, b"\0" * 3, 0, 27,
                   0, 0, 0))
    ldc = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)

    def ld(addr, uniform):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1,
                       3 | (0x10 if uniform else 0), 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    def dep_after():
        s.inst.append((nop, NONE, NONE, len(s.inst)))
        s.inst.append((dep, NONE, NONE, len(s.inst)))

    ld(0x200, False)
    dep_after()
    ld(0x220, True)
    dep_after()
    s.warps.append((0, 0, 0, 0, 0, 6))
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


# constant_paths leaf shapes (schema 27); h100 v2: field 22
# uniform_first_touch_min_smsps replaces the v1 few-warps fields
CP_H100 = {0: 1.0, 1: 64.0, 2: 65536.0, 12: 129.0, 13: 0.0, 14: 0.0,
           15: 92.15, 16: 22.75, 17: 27.0, 18: 256.0, 19: 491.1,
           22: 2.0}
CP_B300 = {0: 1.0, 1: 64.0, 2: 65536.0, 12: 69.05, 13: 19.0, 14: 20.9,
           15: 100.1, 16: 27.5, 17: 20.3, 18: 256.0, 19: 594.6}


def _cp_leaf(s: Seg, fields: dict) -> None:
    s.uses.append((s.leaf_node(27, fields), 0, b"\0" * 3, 0, 27,
                   0, 0, 0))


def cp1_uhold() -> bytes:
    """constant_paths C4: a uniform load without a write barrier is
    held at issue until its cold line's fill is ready (~129); the
    same-line second load does not hold; the reader is free."""
    s = Seg(1, 1)
    _cp_leaf(s, CP_H100)
    ldc = s.template("LDC", dst=[5])
    dep = s.template("DEP", src=[5])

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3 | 0x10, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    ld(0x200)
    ld(0x210)
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, 3))
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def cp2_vbarrier() -> bytes:
    """constant_paths C2/C4 vector path: the first reader waits the
    write barrier for first_touch (92); the reread of the resident
    line waits reread (22) again."""
    s = Seg(1, 1)
    _cp_leaf(s, CP_H100)
    ldc = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldc, NONE, macc, len(s.inst)))

    def dep_after():
        s.inst.append((nop, NONE, NONE, len(s.inst)))
        s.inst.append((dep, NONE, NONE, len(s.inst)))

    ld(0x200)
    dep_after()
    ld(0x200)
    dep_after()
    s.warps.append((0, 0, 0, 0, 0, 6))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def cp3_sets() -> bytes:
    """constant_paths C1: uniform and vector resident sets are
    separate -- w0 fills line A through the uniform path and the
    vector read of A still pays vector_first_touch (100); w1 fills
    line B through the vector path and the uniform read of B is
    held for uniform_first_touch (69)."""
    s = Seg(1, 2)
    _cp_leaf(s, CP_B300)
    ldu = s.template("LDC", dst=[5])
    ldv = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)

    def ld(addr, uniform):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1,
                       3 | (0x10 if uniform else 0), 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldu if uniform else ldv, NONE, macc,
                       len(s.inst)))

    # w0: uniform fill of A (held ~69), then vector first touch of A
    ld(0x200, True)
    ld(0x200, False)
    s.inst.append((nop, NONE, NONE, len(s.inst)))
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, 4))
    # w1: vector fill of B (in flight), then uniform first touch of B
    # (held ~69)
    i0 = len(s.inst)
    ld(0x300, False)
    ld(0x300, True)
    s.warps.append((0, 1, 0, 1, i0, 2))
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def cp4_batch() -> bytes:
    """constant_paths C2 pipelined fills: four adjacent (one cycle
    apart) vector cold loads on one barrier complete at t0+92,
    +119, +146, +173 (first touch then +27 fill interval each);
    the reader waits for the last."""
    s = Seg(1, 1)
    _cp_leaf(s, CP_H100)
    ldv = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldv, NONE, macc, len(s.inst)))

    for a in (0x200, 0x300, 0x400, 0x500):
        ld(a)
    s.inst.append((nop, NONE, NONE, len(s.inst)))
    s.inst.append((dep, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, 6))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def cp5_block() -> bytes:
    """constant_paths C3: a bank-3 line pays the GPC 256 B block
    fetch plus first touch on its first line; another SM of the
    same GPC joins the in-flight block (both readers ~block_fetch
    + first_touch); the second line of the fetched block pays
    first_touch only."""
    s = Seg(1, 2)
    _cp_leaf(s, CP_B300)
    ldv = s.template("LDC", dst=[5], scoreboard_write=1)
    nop = s.template("NOP", control=1)
    dep = s.template("DEP", src=[5], scoreboard_wait_mask=0x2)
    b3 = 0x3 << 32

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldv, NONE, macc, len(s.inst)))

    def dep_after():
        s.inst.append((nop, NONE, NONE, len(s.inst)))
        s.inst.append((dep, NONE, NONE, len(s.inst)))

    # w0 on sm0: first line of block 3 (pays the fetch), then a
    # second line of the same block after the reader waited
    ld(b3 | 0x300)
    dep_after()
    ld(b3 | 0x380)
    dep_after()
    s.warps.append((0, 0, 0, 0, 0, 6))
    # w1 on sm1 (same GPC): another line of block 3, evaluated while
    # the block is in flight
    i0 = len(s.inst)
    ld(b3 | 0x3C0)
    dep_after()
    s.warps.append((1, 0, 0, 0, i0, 3))
    s.cta(0, 0, 0, 1, 0)
    s.cta(1, 1, 1, 1, 1)
    pl = b"".join(cf.PLCE.pack_row(r) for r in
                  [plce(0, 0, 0, 1, 0, 0), plce(1, 0, 0, 1, 1, 0)])
    tp = b"".join(cf.TOPO.pack_row((sm, 0, 0, 0, 3, 0))
                  for sm in (0, 1))
    return container(s.build(), topo=tp)


def _cp6(warps: int) -> bytes:
    """constant_paths v2 C5 (h100): the uniform first touch costs
    uniform_first_touch x max(min_smsps, n) / smsps_per_sm with
    min_smsps = 2, n = SMSPs of the SM holding a resident warp at
    fill start (each extra warp below sits on its own SMSP)."""
    s = Seg(1, warps)
    _cp_leaf(s, CP_H100)
    ldu = s.template("LDC", dst=[5])
    f = s.template("FILL")

    def ld(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3 | 0x10, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ldu, NONE, macc, len(s.inst)))

    # w0's first uniform access starts its fill with `warps` warps
    # resident; w1.. park on fills so the SM's resident count is
    # `warps` at fill start
    ld(0x200)
    s.inst.append((f, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, 2))
    for wi in range(1, warps):
        first = len(s.inst)
        for _ in range(8):
            s.inst.append((f, NONE, NONE, len(s.inst)))
        s.warps.append((0, wi, wi, wi % 4, first, 8))
    s.cta(0, 0, 0, warps, 0)
    return container(s.build())


def cp6_smsp1() -> bytes:
    return _cp6(1)


def cp6_smsp2() -> bytes:
    return _cp6(2)


def cp6_smsp3() -> bytes:
    return _cp6(3)


def cp6_smsp4() -> bytes:
    return _cp6(4)


def cp6_smsp_retire() -> bytes:
    """C5 resident semantics: four warps on the SM's four SMSPs;
    w1..w3 exhaust their bodies and are set done before w0's first
    uniform access starts its fill, so n counts only the still
    resident SMSP (1 -> 64, not 129 for the four SMSPs that have
    held a warp)."""
    s = Seg(1, 4)
    _cp_leaf(s, CP_H100)
    ldu = s.template("LDC", dst=[5])
    f = s.template("FILL")
    # w0: 24 filler insts, then the cold uniform access, then a
    # filler; its fill starts ~cycle 24, after w1..w3 are done
    for _ in range(24):
        s.inst.append((f, NONE, NONE, len(s.inst)))
    macc = len(s.macc)
    s.macc.append((len(s.msec), 1, 3 | 0x10, 0, 0, 0, 0, 0))
    s.msec.append((0x200, 0xFFFFFFFF, 0))
    s.inst.append((ldu, NONE, macc, len(s.inst)))
    s.inst.append((f, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, 26))
    for wi in range(1, 4):
        first = len(s.inst)
        for _ in range(2):
            s.inst.append((f, NONE, NONE, len(s.inst)))
        s.warps.append((0, wi, wi, wi % 4, first, 2))
    s.cta(0, 0, 0, 4, 0)
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


# ---- protocol-v5 G1 wait-result fixtures ------------------------------
#
# Leaf values are set so every installed field is distinguishable:
# result latency 38 (svc {1,2}), release.cycles 88 (f0),
# suspended_wake_cycles 228 (f8), syncs_wake_cycles 136 (nanosleep f4),
# SYNCS service interval 5 where set (f9).
#
# A "reader" is a template with src=[512] (the wait's predicate
# register) and scoreboard_wait_mask=0x02 (wait's write barrier is 1).
# A FILL between the wait and the reader keeps the reader's first
# attempt at wait+2, outside the +1 visibility window of the raise
# (issue_rules); the assertions name the cycle the model produces.


def _g1_common(s: Seg, service: float | None = None) -> None:
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    extra = {8: 228.0}
    if service is not None:
        extra[16] = service
    s.const_program("synchronization.mbarrier.release",
                    3, 4, 22, 88.0, extra=extra)


def g1_self() -> bytes:
    """One warp: SYNCS arrive completes its own phase, the wait issues
    at once (no semantic hold), the reader lands wait-issue + 38 on
    reason barrier:SYNCS.PHASECHK."""
    s = Seg(1, 1)
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None),
            (rd, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def g1_handoff() -> bytes:
    """Two warps: w0's wait issues long before w1's SYNCS arrive; the
    reader lands arrive-issue + 38, independent of 88 / 228."""
    s = Seg(1, 2)
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None),
            (rd, None)])
    s.warp(0, 1, 1,
           fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_late_wait() -> bytes:
    """The wait issues after the other warp's SYNCS arrive: reader at
    wait-issue + 38."""
    s = Seg(1, 2)
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           fills(f, 40)
           + [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
              (f, None),
              (rd, None)])
    s.warp(0, 1, 1,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_nonsyncs_pend() -> bytes:
    """Phase completed by a non-SYNCS arrive while the wait is pending:
    reader at completion + suspended_wake_cycles + 38."""
    s = Seg(1, 2)
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None),
            (rd, None)])
    s.warp(0, 1, 1,
           fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_nonsyncs_late() -> bytes:
    """Wait issued after a non-SYNCS completion: reader at
    max(wait-issue, completion + 88) + 38."""
    s = Seg(1, 2)
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("MBAR.ARRIVE")
    wt = s.template("MBAR.WAIT", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           fills(f, 200)
           + [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
              (f, None),
              (rd, None)])
    s.warp(0, 1, 1,
           fills(f, 30)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_sleep_syncs() -> bytes:
    """NANOSLEEP.SYNCS sleeper on a SYNCS-completed phase wakes at the
    completion date — no syncs_wake_cycles / wake latency."""
    s = Seg(1, 2)
    _g1_common(s)
    f = s.template("FILL")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP,
                             {0: 10000.0, 1: 1.0, 4: 136.0}),))
    arr = s.template("SYNCS.ARRIVE")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (ns, s.semantic(NANOSLEEP, flags=SYNCS, fields=(10000,))),
            (f, None)])
    s.warp(0, 1, 1,
           fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_sleep_nonsyncs() -> bytes:
    """The same sleeper on a phase completed by a non-SYNCS arrive
    keeps today's dating: completion + suspended_wake + syncs_wake."""
    s = Seg(1, 2)
    _g1_common(s)
    f = s.template("FILL")
    wt = s.template("MBAR.WAIT", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP,
                             {0: 10000.0, 1: 1.0, 4: 136.0}),))
    arr = s.template("MBAR.ARRIVE")
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (ns, s.semantic(NANOSLEEP, flags=SYNCS, fields=(10000,))),
            (f, None)])
    s.warp(0, 1, 1,
           fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_nobarrier() -> bytes:
    """Wait whose template raises no dependency barrier: the warp's
    next issue is held until the result time instead (result floor)."""
    s = Seg(1, 1)
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512],
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    s.warp(0, 0, 0,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def g1_nobarrier_pend() -> bytes:
    """No-barrier wait on an incomplete phase: next issue at the
    resolved result time (completion + 38 for a SYNCS arrive)."""
    s = Seg(1, 2)
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512],
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None)])
    s.warp(0, 1, 1,
           fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_deadlock() -> bytes:
    """Wait on a phase that never completes: the reader parks on the
    wait's dependency barrier; the deadlock report names the
    mbarrier object."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 2, 0))  # expects two arrivals
    s.const_program("synchronization.mbarrier.release",
                    3, 4, 22, 88.0, extra={8: 228.0})
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None),
            (rd, None)])
    s.warp(0, 1, 1,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_two_pending() -> bytes:
    """Two pending waits on the same dependency barrier: object 0
    completes ~200 cycles before object 1; the reader stays held until
    the later result is fixed (arrive on object 1 + 38)."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.mbpi.append((0, 1, 0x2000, 1, 0))
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (wt, s.semantic(MBAR_WAIT, fields=(1, 0))),
            (f, None),
            (rd, None)])
    s.warp(0, 1, 1,
           fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))]
           + fills(f, 200)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(1, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_two_pending_rev() -> bytes:
    """g1_two_pending with the arrive order swapped: object 1 first,
    object 0 last; the reader follows the object-0 arrive + 38."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.mbpi.append((0, 1, 0x2000, 1, 0))
    _g1_common(s)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (wt, s.semantic(MBAR_WAIT, fields=(1, 0))),
            (f, None),
            (rd, None)])
    s.warp(0, 1, 1,
           fills(f, 40)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(1, 0, 0, 1)))]
           + fills(f, 200)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_service() -> bytes:
    """R2: service interval 5 on one SM.  Two warps issue a SYNCS
    arrive in the same cycle; the second is served 5 later and its
    reader lands 5 later than the first's."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.mbpi.append((0, 1, 0x2000, 1, 0))
    _g1_common(s, service=5.0)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None),
            (rd, None)])
    s.warp(0, 1, 1,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(1, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(1, 0))),
            (f, None),
            (rd, None)])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_service_frac() -> bytes:
    """Exact copy of g1_service with a fractional service interval
    (5.25): service starts 0, 5.25, 10.5, 15.75 land on cycles
    0, 6, 11, 16 instead of 0, 5, 10, 15."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.mbpi.append((0, 1, 0x2000, 1, 0))
    _g1_common(s, service=5.25)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None),
            (rd, None)])
    s.warp(0, 1, 1,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(1, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(1, 0))),
            (f, None),
            (rd, None)])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def g1_service_2sm() -> bytes:
    """SYNCS on different SMs do not interact: both arrives are served
    at their own issue cycle (one segment, one object per CTA)."""
    s = Seg(1, 1)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.mbpi.append((0, 1, 0x2000, 1, 0))
    _g1_common(s, service=5.0)
    f = s.template("FILL")
    arr = s.template("SYNCS.ARRIVE")
    wt = s.template("SYNCS.PHASECHK", dst=[512], scoreboard_write=1,
                    svc={1: 38.0, 2: 38.0, 7: 1.0})
    rd = s.template("USE", src=[512], scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(0, 0))),
            (f, None),
            (rd, None)])
    s.warp(1, 0, 0,
           [(arr, s.semantic(MBAR_ARRIVE, fields=(1, 0, 0, 1))),
            (wt, s.semantic(MBAR_WAIT, fields=(1, 0))),
            (f, None),
            (rd, None)])
    s.cta(0, 0, 0, 1, 0)
    s.cta(1, 1, 1, 1, 1)
    return container(s.build())


# ---- mio_basic protocol v3 (M1 return chains, M2 SM-class service,
# M3 in-flight group) ----------------------------------------------
# instruction.service field ids used below: 13 return_interval_cycles,
# 14 return_chain, 15 sm_service_interval_cycles, 16
# warp_inflight_group.  Fixture shape follows the real templates: each
# producer has one destination register, raises write barrier 0 and
# carries stall 1; the reader has a source and wait-mask bit 0.


def _mio_prod(s: Seg, op: str, svc: dict) -> int:
    fields = {7: 0.0}
    fields.update(svc)
    return s.template(op, svc=fields, dst=[1], control=1,
                      scoreboard_write=0)


def _mio_reader(s: Seg) -> int:
    return s.template("R", src=[1], control=1,
                      scoreboard_wait_mask=0x1)


def _mio_return_chain(r: float) -> bytes:
    """M1: a warp's producers return one per return interval r (chain
    1); the reader lands at the last turn start + result latency."""
    s = Seg(1, 1)
    p = _mio_prod(s, "P", {1: 23.0, 13: r, 14: 1.0})
    rd = _mio_reader(s)
    s.warp(0, 0, 0, [(p, None)] * 4 + [(rd, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def mio_return_chain() -> bytes:
    return _mio_return_chain(2.0)


def mio_return_chain_r4() -> bytes:
    return _mio_return_chain(4.0)


def _mio_sharing(chain_b: float) -> bytes:
    """M1 chains: alternating kinds on one chain serialize on it; a
    different chain is independent."""
    s = Seg(1, 1)
    a = _mio_prod(s, "A", {1: 23.0, 13: 2.0, 14: 1.0})
    b = _mio_prod(s, "B", {1: 23.0, 13: 2.0, 14: chain_b})
    rd = _mio_reader(s)
    s.warp(0, 0, 0,
           [(a, None), (b, None)] * 4 + [(rd, None)])
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def mio_chain_sharing() -> bytes:
    return _mio_sharing(1.0)


def mio_chain_sharing_b3() -> bytes:
    return _mio_sharing(3.0)


def _mio_sm_service(two_classes: bool) -> bytes:
    """M2: one server per SM and operation class.  Four warps on the
    four SMSPs of one SM, each P P R."""
    s = Seg(1, 4)
    pa = _mio_prod(s, "A", {1: 23.0, 15: 2.0})
    pb = _mio_prod(s, "B", {1: 23.0, 15: 2.0})
    rd = _mio_reader(s)
    for w in range(4):
        p = pb if (two_classes and w >= 2) else pa
        s.warp(0, w, w, [(p, None), (p, None), (rd, None)])
    s.cta(0, 0, 0, 4, 0)
    return container(s.build())


def mio_sm_service() -> bytes:
    return _mio_sm_service(False)


def mio_sm_service_2cls() -> bytes:
    return _mio_sm_service(True)


def mio_sm_service_gaps() -> bytes:
    """M2 (protocol v4): the granted intervals are a set, not a
    frontier -- warp B's turns take the cycles between warp A's.
    Two warps on two SMSPs of one SM; B runs two stall-1 NOPs
    first, so its producers issue t0+2 .. t0+5 (t0 = A's first).
    A's turns t0, t0+2, t0+4, t0+6 (reader t0+6+23); B's turns
    t0+3, t0+5, t0+7, t0+9 (reader t0+9+23)."""
    s = Seg(1, 2)
    p = _mio_prod(s, "P", {1: 23.0, 13: 2.0, 14: 1.0, 15: 1.0})
    rd = _mio_reader(s)
    nop = s.template("NOP", control=1)
    s.warp(0, 0, 0, [(p, None)] * 4 + [(rd, None)])
    s.warp(0, 1, 1, [(nop, None)] * 2 + [(p, None)] * 4 + [(rd, None)])
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def _mio_group(n: int, gap: bool) -> bytes:
    """M3: after every g = 4 issues of a run the next waits at issue
    until the fourth turn's end (mio_inflight), where the count
    clears.  gap inserts a >100-cycle filler between two bursts."""
    s = Seg(1, 1)
    p = _mio_prod(s, "P", {1: 26.0, 13: 2.0, 14: 2.0, 16: 4.0})
    rd = _mio_reader(s)
    insts = [(p, None)] * n
    if gap:
        f = s.template("FILL", svc={1: 120.0, 2: 120.0, 7: 0.0},
                       dst=[9], control=1, scoreboard_write=1)
        p2 = s.template("P",
                        svc={1: 26.0, 7: 0.0, 13: 2.0, 14: 2.0,
                             16: 4.0},
                        src=[9], dst=[1], control=1,
                        scoreboard_write=0, scoreboard_wait_mask=0x2)
        nop = s.template("NOP", control=1)
        # the NOP keeps p2 out of the cycle where a barrier raised
        # the cycle before is still judged by its previous release
        insts += [(f, None), (nop, None), (p2, None)] + \
            [(p, None)] * (n - 1)
    insts += [(rd, None)]
    s.warp(0, 0, 0, insts)
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def mio_inflight_group() -> bytes:
    return _mio_group(8, False)


def mio_inflight_group6() -> bytes:
    return _mio_group(6, False)


def mio_inflight_group_2x6() -> bytes:
    return _mio_group(6, True)


def mio_unit_two_cycle() -> bytes:
    """a 2-cycle unit: producers and the reader take turns 2.0 apart
    on one SMSP unit; the reader's wait is the unit's, no barrier
    reason lands on it."""
    s = Seg(1, 1)
    node = s.leaf_node(2, {0: 0.0, 1: 1.0, 2: 0.0, 3: 1.0})
    s.uses.append((node, 4, b"\0" * 3, 100, 2, 0xF, 0, 0))
    c = s.template("C", svc={1: 2.0, 7: 2.0, 10: 100.0}, dst=[1],
                   control=1, scoreboard_write=0)
    rd = s.template("R", svc={7: 2.0, 10: 100.0}, src=[1], control=1,
                    scoreboard_wait_mask=0x1)
    s.warp(0, 0, 0, [(c, None)] * 4 + [(rd, None)])
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _lds_macc(s: Seg) -> int:
    """one shared-space load access (flags&7 == 1)."""
    macc = len(s.macc)
    s.macc.append((len(s.msec), 1, 0x1, 0, 0, 0, 0, 0))
    s.msec.append((0x0, 0xFFFFFFFF, 0))
    return macc


def _mio_lds(stores: int) -> bytes:
    """M1 on the storage.shared.issue row: an LDS that takes a turn
    books result = turn_start + cycles + completion latency and
    neither reads nor advances the SMSP frontier -- with `stores` a
    pair of turn-less kind-5 stores behind the LDS run shows the
    frontier untouched."""
    s = Seg(1, 1)
    ld = s.template("LDS", timing=((5, {0: 2.0, 1: 1.0}),),
                    svc={1: 21.0, 2: 21.0, 7: 0.0, 13: 2.0, 14: 1.0,
                         15: 1.0},
                    dst=[1], control=1, scoreboard_write=0)
    st = s.template("STS", timing=((5, {0: 10.0, 1: 1.0}),),
                    svc={1: 30.0, 2: 30.0, 7: 0.0},
                    scoreboard_write=0)
    rd = _mio_reader(s)
    first = len(s.inst)
    for _ in range(4):
        s.inst.append((ld, NONE, _lds_macc(s), len(s.inst)))
    for _ in range(stores):
        s.inst.append((st, NONE, _sts(s, 1), len(s.inst)))
    s.inst.append((rd, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, first, len(s.inst) - first))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def mio_lds_turn() -> bytes:
    return _mio_lds(0)


def mio_lds_turn_frontier() -> bytes:
    return _mio_lds(2)


def _mio_const_turn(turn: bool) -> bytes:
    """M1 on constant_paths C2: a vector constant re-read of a
    resident filled line counts its reread from the turn start --
    R(with r 2.0 chain 2) - R(without the fields) = 3."""
    s = Seg(1, 1)
    # first-touch fill of 4 cycles: the priming load's line is
    # resident by the time the measured producers issue, so their
    # re-reads run the resident path
    cp = dict(CP_H100)
    cp[15] = 4.0
    _cp_leaf(s, cp)
    svc = {1: 1.0, 7: 0.0}
    if turn:
        svc.update({13: 2.0, 14: 2.0})
    ld = s.template("LDC", svc=svc, dst=[1], control=1,
                    scoreboard_write=0)
    rd = _mio_reader(s)

    def ldc(addr):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 3, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.inst.append((ld, NONE, macc, len(s.inst)))

    # cold fill of the line; the reader waits its fill so the four
    # measured producers all hit the resident filled path
    ldc(0x200)
    s.inst.append((rd, NONE, NONE, len(s.inst)))
    for _ in range(4):
        ldc(0x200)
    s.inst.append((rd, NONE, NONE, len(s.inst)))
    s.warps.append((0, 0, 0, 0, 0, len(s.inst)))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def mio_constant_turn() -> bytes:
    return _mio_const_turn(True)


def mio_constant_turn_off() -> bytes:
    return _mio_const_turn(False)


# ---- instruction.warp_queue (timing kind 67; warp_queue protocol) ----
# TIME row fields (timing_row schema ids): 1 enabled, 23
# queue_depth_count, 24 release_interval_cycles, 25
# start_latency_cycles, 26 start_interval_ratio, 27 is_global_entry,
# 28 has_chain_turn.  Fixtures follow the real templates: one stall-1
# instruction per slot, loads and S2R / MUFU carry a destination and
# write barrier 0, the reader a source and wait-mask bit 0, memory
# instances carry one access of their space (shared 1, global 0).

TK_WARP_QUEUE = 67


def _wq_row(r: float, is_global: int, turn: int, enabled: float = 1.0,
            extra: dict | None = None):
    fields = {1: enabled, 23: 10.0, 24: r, 25: 5.5, 26: 0.5,
              27: float(is_global), 28: float(turn)}
    fields.update(extra or {})
    return ((TK_WARP_QUEUE, fields),)


def _wq_access(s: Seg, flags: int, slot: int) -> int:
    macc = len(s.macc)
    s.macc.append((len(s.msec), 1, flags, 0, 0, 0, 0, 0))
    s.msec.append((0x1000 * (slot + 1), 0xFFFFFFFF, 0))
    return macc


# name -> (operation class, MACC flags | None, row (r, global, turn)
#          | None, service fields, has destination)
_WQ_OPS = {
    "STS": ("STS", 0x41, (4.0, 0, 1), {1: 1.0, 2: 1.0, 7: 0.0}, False),
    "STSrem": ("STS", 0x61, (4.0, 0, 1), {1: 1.0, 2: 1.0, 7: 0.0},
               False),
    "STSoff": ("STS", 0x41, None, {1: 1.0, 2: 1.0, 7: 0.0}, False),
    "STG": ("STG", 0x40, (6.0, 1, 0), {1: 1.0, 2: 1.0, 7: 0.0}, False),
    "LDS": ("LDS", 0x1, (2.0, 0, 0),
            {1: 21.0, 2: 21.0, 7: 0.0, 13: 2.0, 14: 1.0}, True),
    "LDG": ("LDG", 0x0, (4.0, 1, 0), {1: 30.0, 2: 30.0, 7: 0.0}, True),
    "S2R": ("S2R", None, None, {1: 23.0, 2: 23.0, 7: 0.0}, True),
    "MUFU": ("MUFU.EX2", None, None, {1: 20.0, 2: 20.0, 7: 0.0}, True),
    "NOP": ("NOP", None, None, {1: 1.0, 2: 1.0, 7: 0.0}, False),
    "SHFL": ("SHFL.IDX", None, (4.0, 0, 0),
             {1: 1.0, 2: 1.0, 7: 0.0, 13: 4.0, 14: 1.0}, True),
}
# warp_queue v2 fields per operation: (sm_start_interval_cycles, c);
# sm_depth_count 46 on every row.  The v1 Q4 turn is off in v2 fixtures.
_WQ_V2 = {"STS": (1.0, 0.0), "STG": (2.0, 0.0), "LDS": (1.0, 14.7),
          "LDG": (1.0, 28.3), "SHFL": (1.0, 14.7), "STSrem": (1.0, 0.0)}


def _wq(ops: list, reader: bool = False, row_enabled: float = 1.0,
        non_member_rows: bool = False, warps: int = 1,
        slots: tuple | None = None, v2: bool = False,
        a_override: dict | None = None, c_override: dict | None = None,
        warps_ops: list | None = None,
        try_next_on_block: bool = True) -> bytes:
    """each warp issues `ops` (names of _WQ_OPS) in consecutive
    instructions; `reader` appends a reader of the last destination;
    row_enabled 0 keeps the rows but disables them; non_member_rows
    attaches a disabled row to the templates that have none."""
    if warps_ops is not None:
        warps = len(warps_ops)
    s = Seg(1, warps)
    if not try_next_on_block:
        # scheduler row (schema 7): an SMSP's attempt ends at the first
        # candidate that cannot issue
        node = s.leaf_node(7, {1: 1.0, 2: 1.0, 3: 0.0})
        s.uses.append((node, 7, b"\0" * 3, 0, 7, 0xF, 0, 0))
    tmpl = {}
    names = list(dict.fromkeys(
        n for w_ops in (warps_ops or [ops]) for n in w_ops))
    for name in names:
        op, flags, row, svc, dst = _WQ_OPS[name]
        timing = ()
        if row is not None:
            extra = None
            if v2:
                a, c = _WQ_V2[name]
                a = (a_override or {}).get(name, a)
                c = (c_override or {}).get(name, c)
                extra = {29: a, 30: 46.0, 31: c}
                row = (row[0], row[1], 0)
            timing = _wq_row(*row, enabled=row_enabled, extra=extra)
        elif non_member_rows:
            timing = _wq_row(4.0, 0, 0, enabled=0.0)
        tmpl[name] = s.template(
            op, timing=timing, svc=dict(svc),
            dst=[1] if dst else None, control=1,
            scoreboard_write=0 if dst else 0xFF)
    rd = _mio_reader(s)
    for wi in range(warps):
        first = len(s.inst)
        w_ops = warps_ops[wi] if warps_ops is not None else ops
        for k, name in enumerate(w_ops):
            flags = _WQ_OPS[name][1]
            macc = (None if flags is None
                    else _wq_access(s, flags, wi * 64 + k))
            s.instance(tmpl[name], None, macc)
        if reader:
            s.instance(rd)
        slot = slots[wi] if slots else wi
        s.warps.append((0, wi, slot, slot % 4, first,
                        len(w_ops) + (1 if reader else 0)))
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, warps, 0)
    return container(s.build())


def _wq_fixtures() -> dict:
    def run(name, n):
        return [name] * n

    def make(ops, **kw):
        return lambda: _wq(ops, **kw)
    out = {
        "wq_sts16": make(run("STS", 16)),
        "wq_sts16_off": make(run("STS", 16), row_enabled=0.0),
        "wq_stg16": make(run("STG", 16)),
        "wq_lds24": make(run("LDS", 24)),
        "wq_sts6_lds18": make(run("STS", 6) + run("LDS", 18)),
        "wq_lds12_sts12": make(run("LDS", 12) + run("STS", 12)),
        "wq_sts10_stg6": make(run("STS", 10) + run("STG", 6)),
        "wq_stg6_ldg20": make(run("STG", 6) + run("LDG", 20)),
        "wq_stg10_sts6": make(run("STG", 10) + run("STS", 6)),
        "wq_fwd4": make(run("STS", 4) + run("LDS", 2), reader=True),
        "wq_fwd0": make(run("LDS", 2), reader=True),
        "wq_s2r28": make(run("S2R", 28)),
        "wq_s2r28_off": make(run("S2R", 28), non_member_rows=True),
        "wq_mufu12": make(run("MUFU", 12)),
        "wq_mufu12_off": make(run("MUFU", 12), non_member_rows=True),
        "wq_remote16": make(run("STSrem", 16)),
        "wq_pred_off": make(run("STS", 6) + run("STSoff", 6)
                            + run("STS", 10)),
        "wq_two_warps_smsp": make(run("STS", 10), warps=2,
                                  slots=(0, 4)),
        # S1: 16 warps (4 per SMSP) each LDS + reader; the reader's
        # issue exposes the release (S3 floor); one class / two classes
        "wq2_s1_lds16": make([], reader=True, v2=True,
                             warps_ops=[run("LDS", 2)] * 16),
        "wq2_s1_two_classes": make(
            [], reader=True, v2=True,
            warps_ops=[run("LDS", 2)] * 8 + [run("SHFL", 2)] * 8),
        # S2: sixteen warps, ten STS each, no SM slots (a = 0) so the
        # releases follow from the issue cycles alone
        "wq2_s2_sts": make([], v2=True, a_override={"STS": 0.0},
                           warps_ops=[run("STS", 10)] * 16),
        # late warps (12 NOPs first, no entries of their own) wait on
        # the SM bound: STS early warps; STG late warps -> lg
        # S2': sixteen warps, 128 STS each (steady state) with the SM bound; one
        # warp issues only 3 STS (exits early) while the others wait
        "wq2_s2_fair": make([], v2=True,
                            warps_ops=[run("STS", 128)] * 16),
        # S2' hang shape: a 32-warp CTA (8 warps per SMSP) issues 3 STS
        # each, 96 entries against the bound of 46; the head of the
        # waiting list must be reached although earlier candidates of
        # its SMSP wait behind it
        "wq2_s2_burst": make([], v2=True, try_next_on_block=False,
                             warps_ops=[run("STS", 3)] * 32),
        "wq2_s2_exit": make([], v2=True,
                            warps_ops=[run("STS", 3)]
                            + [run("STS", 16)] * 15),
        "wq2_s2_late_stg": make(
            [], v2=True, a_override={"STS": 0.0},
            warps_ops=[run("STS", 10)] * 12
            + [run("NOP", 3) + run("STG", 6)] * 4),
        "wq2_s2_late_sts": make(
            [], v2=True, a_override={"STS": 0.0},
            warps_ops=[run("STS", 10)] * 12
            + [run("NOP", 3) + run("STS", 6)] * 4),
        # S3: result floor release + c
        "wq2_ld2": make(run("LDS", 2), reader=True, v2=True),
        "wq2_sts4_lds2": make(run("STS", 4) + run("LDS", 2), reader=True,
                              v2=True),
        "wq2_ldg2": make(run("LDG", 2), reader=True, v2=True),
        "wq2_ldg2_c0": make(run("LDG", 2), reader=True, v2=True,
                            c_override={"LDG": 0.0}),
    }
    out.update(_wq_fast_fixtures())
    out.update({
        "wq3_l1_res60": lambda: _wq3(True, 60.0, 28.3),
        "wq3_l1_res5": lambda: _wq3(True, 5.0, 28.3),
        "wq3_l1_cold": lambda: _wq3(False, 60.0, 28.3),
        "wq3_l1_cold_c0": lambda: _wq3(False, 60.0, 0.0),
    })
    return out


def _wq_store_row(s: Seg, sectors: list) -> int:
    macc = len(s.macc)
    s.macc.append((len(s.msec), len(sectors), 0x40, 0, 0, 0, 0, 0))
    for a, bm, lm in sectors:
        s.msec.append((a, bm, lm))
    return macc


def _wq_fast(pattern: str, warps: int = 1, n: int = 16,
             try_next: bool = True) -> bytes:
    """S4: global stores with lane masks; the store-L2 leaf holds two
    requests and drains 8 B/cycle, so a store that passes the store
    buffer gate soon waits (store_buffer); the fast path does not."""
    s = Seg(1, warps)
    if not try_next:
        node = s.leaf_node(7, {1: 1.0, 2: 1.0, 3: 0.0})
        s.uses.append((node, 7, b"\0" * 3, 0, 7, 0xF, 0, 0))
    node = s.leaf_node(3, {2: 32.0, 3: 128.0, 13: 100.0, 14: 8.0,
                           21: 50.0, 22: 2.0, 42: 128.0, 44: 8.0})
    s.uses.append((node, 5, b"\0" * 3, 0, 3, 0xFFFFFFFF, 0, 0))
    stg = s.template(
        "STG", timing=_wq_row(6.0, 1, 0, extra={29: 2.0, 30: 46.0}),
        svc={1: 1.0, 2: 1.0, 7: 0.0}, control=1)

    def sectors(wi, i):
        base = 0x10000 + 0x4000 * wi
        if pattern == "alternate":
            base += 0x1000 * (i % 2)
        if pattern == "two_byte":
            return [(base, 0xFFFFFFFF, 0xFFFF),
                    (base + 32, 0xFFFFFFFF, 0xFFFF0000)]
        if pattern == "wide16":
            return [(base + 32 * k, 0xFFFFFFFF, 0x3 << (2 * k))
                    for k in range(16)]
        return [(base + 32 * k, 0xFFFFFFFF, 0xFF << (8 * k))
                for k in range(4)]
    for wi in range(warps):
        first = len(s.inst)
        for i in range(n):
            s.instance(stg, None, _wq_store_row(s, sectors(wi, i)))
        s.warps.append((0, wi, wi, wi % 4, first, n))
    for i in range(len(s.inst)):
        s.code.append((i, 0, 0x1000 + 16 * i))
    s.cta(0, 0, 0, warps, 0)
    return container(s.build())


def _wq3(resident: bool, l1_latency: float, c: float) -> bytes:
    """R1: two member LDG + reader on one 32 B sector; resident: an
    earlier load fills the sector (a NANOSLEEP lets the fill land), so
    both LDG hit L1; otherwise both miss (two different sectors).
    The L1 data array is on, the L1 latency is `l1_latency`."""
    s = Seg(1, 1)
    node = s.leaf_node(3, {2: 32.0, 3: 128.0, 9: l1_latency, 10: 16.0,
                           11: 1.0, 13: 100.0, 14: 8.0, 42: 128.0,
                           47: 262144.0})
    s.uses.append((node, 5, b"\0" * 3, 0, 3, 0xFFFFFFFF, 0, 0))
    ldg = s.template(
        "LDG", timing=_wq_row(4.0, 1, 0, extra={29: 1.0, 30: 46.0, 31: c}),
        svc={1: 30.0, 2: 30.0, 7: 0.0}, dst=[1], control=1,
        scoreboard_write=0)
    ns = s.template("NANOSLEEP",
                    timing=((TK_NANOSLEEP, {0: 300.0, 1: 1.0}),))
    rd = _mio_reader(s)

    def load(addr=0x1000):
        macc = len(s.macc)
        s.macc.append((len(s.msec), 1, 0, 0, 0, 0, 0, 0))
        s.msec.append((addr, 0xFFFFFFFF, 0))
        s.instance(ldg, None, macc)
    first = len(s.inst)
    if resident:
        load()
        s.instance(ns, s.semantic(NANOSLEEP, fields=(300,)))
    load()
    load(0x1000 if resident else 0x2000)
    s.instance(rd)
    s.warps.append((0, 0, 0, 0, first, len(s.inst) - first))
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _wq_fast_fixtures() -> dict:
    return {
        "wq4_same_line": lambda: _wq_fast("same"),
        "wq4_alternate": lambda: _wq_fast("alternate"),
        "wq4_two_byte": lambda: _wq_fast("two_byte"),
        "wq4_wide16": lambda: _wq_fast("wide16"),
        "wq4_two_warps": lambda: _wq_fast("same", warps=2),
        # S2' with the store buffer: 32 warps x 3 first-time stores (the
        # store buffer's own waiting list and the SM bound's list must not
        # wait for each other)
        "wq2_s2_stg_burst": lambda: _wq_fast("alternate", warps=32, n=3,
                                             try_next=False),
    }


# ---- placement v1 (ledger booking; the cycle counts do not move) ----


def _pl_code(s: Seg, addrs: dict | None = None) -> None:
    """code address of every instance: 0x1000 + 16 * index unless
    `addrs` gives one."""
    for i in range(len(s.inst)):
        s.code.append((i, 0, (addrs or {}).get(i, 0x1000 + 16 * i)))


def pl_branch_pieces() -> bytes:
    """rule T: a taken BRA (stall 5, control_envelope 10) with
    fallthrough_hold 4 and resolve 3 -- the nine cycles behind it are
    booked 4 on the fall-through address (0x1010) as
    control:branch_fallthrough, 3 control:branch and 2
    control:branch_refetch on the target (0x1040)."""
    s = Seg(1, 1)
    bra = s.template("BRA", control=5)
    f = s.template("FILL")
    s.warp(0, 0, 0, [(bra, None), (f, None)])
    node = s.leaf_node(5, {4: 10.0, 6: 4.0, 7: 3.0})
    s.uses.append((node, 2, b"\0" * 3, 0, 5, 0x1, 0, 0))
    _pl_code(s, {1: 0x1040})
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def pl_depbar_wait() -> bytes:
    """rule D: m_cpasync_anchor with the wait named DEPBAR.LE and a
    FILL behind it; async_group_wait is booked on the FILL's address."""
    s = Seg(1, 1)
    _mem_leaf(s)
    iss = s.template("CP.ASYNC", timing=((39, {0: 300.0, 1: 1.0}),))
    com = s.template("CP.COMMIT")
    wt = s.template("DEPBAR.LE")
    f = s.template("FILL")
    first = len(s.inst)
    macc = len(s.macc)
    s.macc.append((len(s.msec), 16, 0, 0, 0, 0, 0, 0))
    for j in range(16):
        s.msec.append((j * 128, 0xFFFFFFFF, 0))
    s.instance(iss, s.semantic(CP_ISSUE, fields=(0, 0, NONE)), macc)
    s.instance(com, s.semantic(CP_COMMIT, fields=(0, 0, NONE, 0)))
    s.instance(wt, s.semantic(CP_WAIT, fields=(0, 0)))
    s.instance(f)
    s.warps.append((0, 0, 0, 0, first, 4))
    _pl_code(s)
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _pl_defer(defer: bool, bar_stall: int, kinds: list[str]) -> bytes:
    """rule B: one warp, a CTA barrier wait (template flag DEFER_BLOCKING
    when `defer`, release 7, Dmin 5)
    followed by the given instruction kinds; instances 1.. carry the
    code addresses 0x1010...  Stall field of every kind: 1."""
    s = Seg(1, 1)
    bar = s.template("BAR", control=bar_stall,
                     flags=512 if defer else 0,
                     timing=((TK_CTA_REL, {0: 7.0, 1: 1.0, 21: 5.0}),))
    t = {
        "IMAD": s.template("IMAD", control=1,
                           svc={1: 1.0, 2: 1.0, 10: 1.0}),
        "FFMA": s.template("FFMA", control=1,
                           svc={1: 1.0, 2: 1.0, 10: 2.0}),
        "UIADD3": s.template("UIADD3", control=1,
                             svc={1: 1.0, 2: 1.0, 10: 5.0}),
        "UMOV": s.template("UMOV", control=1,
                           svc={1: 1.0, 2: 1.0, 10: 5.0}),
        "S2R": s.template("S2R", control=1,
                          svc={1: 1.0, 2: 1.0, 13: 2.0, 14: 1.0}),
        "LDC": s.template("LDC", control=1, svc={1: 1.0, 2: 1.0}),
    }
    sema = s.semantic(CTA_WAIT, flags=ALSO_ARRIVES, fields=(0, 0, 0))
    s.warp(0, 0, 0, [(bar, sema)] + [(t[k], None) for k in kinds])
    _pl_code(s)
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def pl_defer_d4() -> bytes:
    # S2R at dist 4 is passed, IMAD passes, LDC at dist 6 stops (inst 3)
    return _pl_defer(True, 4,
                     ["S2R", "IMAD", "LDC", "LDC"])


def pl_defer_d5() -> bytes:
    # S2R at dist 5 stops (inst 1)
    return _pl_defer(True, 5, ["S2R", "LDC"])


def pl_defer_umov() -> bytes:
    # UMOV (uniform unit, no MIO turn) is not a pass kind (inst 1)
    return _pl_defer(True, 6, ["UMOV", "LDC"])


def pl_defer_pass() -> bytes:
    # IMAD, UIADD3, FFMA run through, LDC at dist 9 stops (inst 4)
    return _pl_defer(True, 6,
                     ["IMAD", "UIADD3", "FFMA", "LDC", "LDC"])


def pl_bar_plain() -> bytes:
    # a BAR without DEFER_BLOCKING keeps the next instance (inst 1)
    return _pl_defer(False, 6, ["IMAD", "LDC"])


def pl_membar_names() -> bytes:
    """rule M, service-result floors: MEMBAR.SC.GPU (stall 5, result
    30), MEMBAR.ALL.CTA (stall 3, result 12), FENCE.VIEW.ASYNC.S
    (stall 2, result 20); the stall-field part stays control_floor."""
    s = Seg(1, 1)
    m1 = s.template("MEMBAR.SC.GPU", control=5, svc={1: 30.0, 2: 30.0})
    m2 = s.template("MEMBAR.ALL.CTA", control=3, svc={1: 12.0, 2: 12.0})
    m3 = s.template("FENCE.VIEW.ASYNC.S", control=2,
                    svc={1: 20.0, 2: 20.0})
    f = s.template("FILL")
    s.warp(0, 0, 0, [(m1, None), (m2, None), (m3, None), (f, None)])
    _pl_code(s)
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def pl_fence_drain() -> bytes:
    """rule M, drain floors: b24_smsp_membar (MEMBAR.ALL.CTA held until
    71, proxy fence until 85, clean MEMBAR until 92) with code
    addresses."""
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
    _pl_code(s)
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def pl_fence_waiter() -> bytes:
    """rule F: warp 1 issues PRE (write barrier 1), then an acquire
    that is suspended until the publish is visible (b5_descriptor);
    the suspension is booked on the first instruction behind PRE
    whose wait mask holds barrier 1 (WAITER, 0x1050), not on PRE."""
    s = Seg(1, 2)
    s.const_program("synchronization.descriptor.visibility",
                    3, 4, 23, 724.0)
    f = s.template("FILL")
    pub = s.template("DESC.PUBLISH")
    pre = s.template("PRE", scoreboard_write=1)
    acq = s.template("DESC.ACQUIRE")
    waiter = s.template("WAITER", scoreboard_wait_mask=0x2)
    s.warp(0, 0, 0,
           [(pub, s.semantic(DESC_VIS, fields=(1, 1, 0xABC))), (f, None)])
    s.warp(0, 1, 1, fills(f, 3)
           + [(pre, None),
              (acq, s.semantic(DESC_VIS, fields=(4, 0, 0xABC))),
              (waiter, None), (f, None)])
    _pl_code(s)
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def pl_fence_service_waiter() -> bytes:
    """rule F (extended), service-floor form (b300 s.fence_bar, first
    copy): FENCE.VIEW.ASYNC.S (stall 2, result 20, write barrier 2),
    two VIADD, then a BAR whose wait mask holds barrier 2.  The part of
    the floor beyond the stall field is booked on the BAR (0x1030); the
    stall-field part stays control_floor on the next instance (0x1010)."""
    s = Seg(1, 1)
    fen = s.template("FENCE.VIEW.ASYNC.S", control=2, scoreboard_write=2,
                     svc={1: 20.0, 2: 20.0})
    add = s.template("VIADD")
    bar = s.template("BAR.SYNC", scoreboard_wait_mask=0x4)
    f = s.template("FILL")
    s.warp(0, 0, 0, [(fen, None), (add, None), (add, None), (bar, None),
                     (f, None)])
    _pl_code(s)
    s.cta(0, 0, 0, 1, 0)
    return container(s.build())


def _pl_origin(flagged: bool, cap: float) -> bytes:
    """rule P'' (ledger): warp 0 polls object 0 (pre-initialised, phase 0
    pending) with a TRYWAIT whose pending result gates the reader (a
    BRA with wait mask 0x1); warp 1 arrives after ~100 fills.  With the
    wait-origin flag (earliest_issue 0x1ff0) the cycles earlier than
    issue(poll) + cap are booked on 0x1ff0; the reader is held from ten
    issues after the poll."""
    s = Seg(1, 2)
    s.mbpi.append((0, 0, 0x1000, 1, 0))
    s.const_program("synchronization.mbarrier.release", 3, 4, 22, 10.0)
    s.const_program("synchronization.mbarrier.trywait_timeout",
                    1, 4, TK_TRYWAIT, 0.0, extra={22: cap})
    f = s.template("FILL")
    wt = s.template("SYNCS.PHASECHK", scoreboard_write=0)
    rd = s.template("BRA", scoreboard_wait_mask=0x1)
    arr = s.template("MBAR.ARRIVE")
    sw = s.semantic(MBAR_WAIT, flags=0x200 if flagged else 0, fields=(0, 0))
    if flagged:
        k, fl, _e, *slots = s.sema[sw]
        s.sema[sw] = (k, fl, 0x1ff0, *slots)
    s.warp(0, 0, 0, [(wt, sw)] + fills(f, 10) + [(rd, None), (f, None)])
    s.warp(0, 1, 1, fills(f, 100)
           + [(arr, s.semantic(MBAR_ARRIVE, fields=(0, 0, 0, 1)))])
    _pl_code(s)
    s.cta(0, 0, 0, 2, 0)
    return container(s.build())


def pl_origin_all() -> bytes:
    return _pl_origin(True, 100000.0)


def pl_origin_cap() -> bytes:
    return _pl_origin(True, 30.0)


def pl_origin_off() -> bytes:
    return _pl_origin(False, 100000.0)


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
    "kb1_active_cycles": kb1_active_cycles,
    "kb2_first_line": kb2_first_line,
    "kb3_exit": kb3_exit,
    "kb3b_exit_drain": kb3b_exit_drain,
    "kb4_drain_starve": kb4_drain_starve,
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
    "sl_waits": sl_waits,
    "sl_loop": sl_loop,
    "sl_issue_slot": sl_issue_slot,
    "sl_control_floor": sl_control_floor,
    "sl_due": sl_due,
    "sl_cta_barrier": sl_cta_barrier,
    "sl_cluster_barrier": sl_cluster_barrier,
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
    "r2_bsync_floor": r2_bsync_floor,
    "r2_bsync_floor_stall": r2_bsync_floor_stall,
    "sl_control_branch": sl_control_branch,
    "sl_control_yield": sl_control_yield,
    "sl_control_stall": sl_control_stall,
    "r2_bssy_no_floor": r2_bssy_no_floor,
    "c7_dispatch_no_enabled": c7_dispatch_no_enabled,
    "cp1_uhold": cp1_uhold,
    "cp2_vbarrier": cp2_vbarrier,
    "cp3_sets": cp3_sets,
    "cp4_batch": cp4_batch,
    "cp5_block": cp5_block,
    "cp6_smsp1": cp6_smsp1,
    "cp6_smsp2": cp6_smsp2,
    "cp6_smsp3": cp6_smsp3,
    "cp6_smsp4": cp6_smsp4,
    "cp6_smsp_retire": cp6_smsp_retire,
    "c8_const_capacity": c8_const_capacity,
    "c8b_const_inflight": c8b_const_inflight,
    "c8c_bank0": c8c_bank0,
    "c8d_bank0_uniform": c8d_bank0_uniform,
    "c8e_const_hold": c8e_const_hold,
    "c8f_const_hold_off": c8f_const_hold_off,
    "c8g_const_hold_noresident": c8g_const_hold_noresident,
    "c8h_vbank0_cold": c8h_vbank0_cold,
    "c8i_vbank0_cold_off": c8i_vbank0_cold_off,
    "c4b_shared_l0": lambda: _c4(True, False),
    "c4c_no_code": lambda: _c4(False, True),
    "c4d_sharing_smsp": lambda: _c4(True, False, sharing=1.0),
    "ir1_no_register_tracking": ir1_no_register_tracking,
    "ir2_barrier_latest": ir2_barrier_latest,
    "ir4_yield_stall": ir4_yield_stall,
    "ir3_load_s1": ir3_load_s1,
    "ir3_load_s2": ir3_load_s2,
    "ir3_load_s1g1": ir3_load_s1g1,
    "op1_stream3": op1_stream3,
    "op2_dep3_w2": op2_dep3_w2,
    "op2_dep3_w3": op2_dep3_w3,
    "op2_dep3_w4": op2_dep3_w4,
    "op3_served": op3_served,
    "op4_reads2": op4_reads2,
    "op5_pair_w1": op5_pair_w1,
    "op5_pair_w2": op5_pair_w2,
    "g1_self": g1_self,
    "g1_handoff": g1_handoff,
    "g1_late_wait": g1_late_wait,
    "g1_nonsyncs_pend": g1_nonsyncs_pend,
    "g1_nonsyncs_late": g1_nonsyncs_late,
    "g1_sleep_syncs": g1_sleep_syncs,
    "g1_sleep_nonsyncs": g1_sleep_nonsyncs,
    "g1_nobarrier": g1_nobarrier,
    "g1_nobarrier_pend": g1_nobarrier_pend,
    "g1_deadlock": g1_deadlock,
    "g1_two_pending": g1_two_pending,
    "g1_two_pending_rev": g1_two_pending_rev,
    "g1_service": g1_service,
    "g1_service_frac": g1_service_frac,
    "g1_service_2sm": g1_service_2sm,
    "mio_return_chain": mio_return_chain,
    "mio_return_chain_r4": mio_return_chain_r4,
    "mio_chain_sharing": mio_chain_sharing,
    "mio_chain_sharing_b3": mio_chain_sharing_b3,
    "mio_sm_service": mio_sm_service,
    "mio_sm_service_2cls": mio_sm_service_2cls,
    "mio_sm_service_gaps": mio_sm_service_gaps,
    "mio_inflight_group": mio_inflight_group,
    "mio_inflight_group6": mio_inflight_group6,
    "mio_inflight_group_2x6": mio_inflight_group_2x6,
    "mio_unit_two_cycle": mio_unit_two_cycle,
    "mio_lds_turn": mio_lds_turn,
    "mio_lds_turn_frontier": mio_lds_turn_frontier,
    "mio_constant_turn": mio_constant_turn,
    "mio_constant_turn_off": mio_constant_turn_off,
    "pl_branch_pieces": pl_branch_pieces,
    "pl_depbar_wait": pl_depbar_wait,
    "pl_defer_d4": pl_defer_d4,
    "pl_defer_d5": pl_defer_d5,
    "pl_defer_umov": pl_defer_umov,
    "pl_defer_pass": pl_defer_pass,
    "pl_bar_plain": pl_bar_plain,
    "pl_membar_names": pl_membar_names,
    "pl_fence_drain": pl_fence_drain,
    "pl_fence_waiter": pl_fence_waiter,
    "pl_fence_service_waiter": pl_fence_service_waiter,
    "pl_origin_all": pl_origin_all,
    "pl_origin_cap": pl_origin_cap,
    "pl_origin_off": pl_origin_off,
}


FIXTURES.update(_wq_fixtures())


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in FIXTURES.items():
        (out_dir / f"{name}.job").write_bytes(fn())


if __name__ == "__main__":
    main()

