#!/usr/bin/env python3
"""SASS overhead checks for the barrier_release probes (sync_mb, sync_xw).

Functions copied verbatim from blackwell-prototype tools/microbench/sass_check.py
(check_sync_mb / check_sync_xw and their helpers); blackwellsim imports
removed. CLI: --sass <dir with sync_mb.sass sync_xw.sass> --out checks.json.
"""

import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc

TRYWAIT = "SYNCS.PHASECHK.TRANS64.TRYWAIT"

def inst_rec(i: sc.Inst) -> list:
    return [i.addr, i.text, i.stall, i.wbar, i.rbar, i.wait]


def stall_sum(insts, start=0, stop=None) -> int:
    return sum(i.stall for i in insts[start:stop])


def gap(insts, a_pred, b_pred, occurrence=0):
    """Stall cycles from the occurrence-th instruction matching a_pred up to
    (excluding) the next instruction matching b_pred."""
    idx = [k for k, i in enumerate(insts) if a_pred(i)]
    if len(idx) <= occurrence:
        return None
    a = idx[occurrence]
    for b in range(a + 1, len(insts)):
        if b_pred(insts[b]):
            return stall_sum(insts, a, b)
    return None


def executed_path(w):
    """Walk a window along the success path: unconditional BRA and BRA.U UPx
    (tcgen05.alloc succeeded) are taken; '@Px BRA' right after a TRYWAIT is
    taken (phase complete) and '@!Px BRA' after it falls through; every other
    predicated branch (relinquished-permit check, ELECT loop, retry/timeout
    tails) falls through.  Instructions outside the window end the walk."""
    by_addr = {i.addr: n for n, i in enumerate(w)}
    path, n, guard = [], 0, 0
    while 0 <= n < len(w) and guard < 4 * len(w):
        guard += 1
        i = w[n]
        path.append(i)
        if i.root in ("CALL", "EXIT", "RET"):
            break
        if i.root == "BRA":
            last = i.body.split()[-1]
            target = int(last, 16) if last.startswith("0x") else None
            after_wait = n > 0 and is_trywait(w[n - 1])
            if i.pred is None:
                taken = "!UP" not in i.body
            else:
                taken = after_wait and not i.pred.startswith("@!")
            if taken and target is not None:
                if target not in by_addr:
                    break
                n = by_addr[target]
                continue
        n += 1
    return path


# CS2R window a checker actually selected, kernel name -> [start pc, end pc]
_SELECTED: dict[str, list[str]] = {}


def record_pcs(k, n_s: int, n_e: int) -> None:
    _SELECTED[k.name] = [hex(k.insts[n_s].addr), hex(k.insts[n_e].addr)]


def record_window(k: sc.Kernel, w) -> None:
    """Store the bounding CS2R pcs of the checker's chosen window."""
    if not w:
        return
    i0 = k.insts.index(w[0])
    i1 = k.insts.index(w[-1])
    n_s = max((j for j in range(i0) if "SR_CLOCKLO" in k.insts[j].text), default=None)
    n_e = min((j for j in range(i1 + 1, len(k.insts)) if "SR_CLOCKLO" in k.insts[j].text),
              default=None)
    if n_s is not None and n_e is not None:
        record_pcs(k, n_s, n_e)


def single_window(k: sc.Kernel):
    ws = sc.clock_windows(k)
    w = ws[0] if len(ws) >= 1 else None
    record_window(k, w)
    return w


def all_windows(k: sc.Kernel):
    clock_positions = [
        n for n, inst in enumerate(k.insts) if "SR_CLOCKLO" in inst.text
    ]
    return [
        k.insts[left + 1:right]
        for left, right in zip(clock_positions[::2], clock_positions[1::2])
    ]


def no_spill(w) -> bool:
    return not any(i.root in ("LDL", "STL") for i in w)


def is_trywait(i):
    return i.opcode.startswith(TRYWAIT)


def is_testwait(i):
    return i.opcode == "SYNCS.PHASECHK.TRANS64"


def waiter_exits(ins, pred):
    """stall cycles from a <wait> whose predicate came back true to the t_w
    CS2R: '@P' branches/BREAKs taken, '@!P' fall through, at most 12 steps"""
    by_addr = {i.addr: n for n, i in enumerate(ins)}
    out = []
    for n, i in enumerate(ins):
        if not pred(i):
            continue
        pos, total = n, 0
        for _ in range(12):
            x = ins[pos]
            if pos != n and "SR_CLOCKLO" in x.text:
                out.append(total)
                break
            total += x.stall
            if x.root == "BRA" and (x.pred is None or not x.pred.startswith("@!")):
                last = x.body.split()[-1]
                if not last.startswith("0x") or int(last, 16) not in by_addr:
                    break
                pos = by_addr[int(last, 16)]
                continue
            if x.root in ("EXIT", "RET", "CALL"):
                break
            pos += 1
            if pos >= len(ins):
                break
    return out


def check_sync_mb(name, k):
    m = re.match(r"_Z7k_sync([12])ILi(\d+)ELi(\d+)EEvPmiijj$", name)
    if not m:
        return None
    op = ["arr", "tx", "c1", "c2", "pair_skew", "flag_rtt", "tw_only", "arr_only"][int(m.group(2))]
    u = int(m.group(3))
    w = single_window(k)
    res = {"key": f"sync_mb/{op}/{u}", "op": op, "u": u}
    if w is None:
        return res | {"ok": False, "why": "no clock window"}
    why = []
    waits = [i for i in w if is_trywait(i)]
    if op == "tw_only":
        if len(waits) != u:
            why.append(f"{len(waits)} try_wait, want {u}")
    elif op == "arr_only":
        n_arr = sum(1 for i in w if i.opcode.startswith("SYNCS.ARRIVE.TRANS64.A1T0"))
        if n_arr != u or waits:
            why.append(f"{n_arr} arrives / {len(waits)} try_wait, want {u} / 0")
    elif op in ("arr", "tx", "c1", "c2"):
        if op == "arr":
            issue = lambda i: i.opcode.startswith("SYNCS.ARRIVE.TRANS64.A1T0")
        elif op == "tx":
            issue = lambda i: i.opcode.startswith("SYNCS.ARRIVE.TRANS64.RED.A0TX")
            expects = [i for i in w if i.opcode == "SYNCS.ARRIVE.TRANS64"]
            if len(expects) != u:
                why.append(f"{len(expects)} expect_tx arrives, want {u}")
        elif op == "c1":
            issue = lambda i: i.opcode == "UTCBAR"
        else:
            issue = lambda i: i.opcode == "UTCBAR.2CTA.MULTICAST"
        n_issue = sum(1 for i in w if issue(i))
        if n_issue != u:
            why.append(f"{n_issue} issue ops, want {u}")
        if len(waits) != u:
            why.append(f"{len(waits)} try_wait, want {u}")
        gaps = [gap(executed_path(w), issue, is_trywait, j) for j in range(u)]
        res["gap_issue_to_wait"] = gaps
    elif op == "pair_skew":
        if not any(i.opcode == "UTCBAR.2CTA.MULTICAST" for i in w):
            why.append("no multicast commit")
    if not no_spill(w):
        why.append("local memory in window")
    return res | {"ok": not why, "why": why, "stall_sum": stall_sum(executed_path(w)), "n_window": len(w)}


# ----------------------------------------------------------------- sync_xw

XW_OPS = {0: ("arr", lambda i: i.opcode == "SYNCS.ARRIVE.TRANS64.A1T0"),
          1: ("tx", lambda i: i.opcode == "SYNCS.ARRIVE.TRANS64.RED.A0TX"),
          2: ("c1", lambda i: i.opcode == "UTCBAR"),
          3: ("c2", lambda i: i.opcode == "UTCBAR.2CTA.MULTICAST")}


def check_sync_xw(name, k):
    m = re.match(r"_Z4k_xwILi(\d)EEv\w+$", name)
    if not m:
        return None
    op, is_op = XW_OPS[int(m.group(1))]
    res = {"key": f"sync_xw/{op}", "op": op}
    ins = k.insts
    why = []
    ops = [n for n, i in enumerate(ins) if is_op(i)]
    if len(ops) != 1:
        return res | {"ok": False, "why": [f"{len(ops)} signal ops"]}
    n_op = ops[0]
    n_clk = max((n for n in range(n_op) if "SR_CLOCKLO" in ins[n].text), default=None)
    if n_clk is None:
        return res | {"ok": False, "why": ["no t_s clock read before the signal op"]}
    seg = ins[n_clk:n_op]
    o_s = stall_sum(seg)
    # scoreboard waits inside CS2R..op on barriers set inside the segment are
    # latency the stall sum does not cover
    set_in = {i.wbar for i in seg if i.wbar != 7}
    dep = [f"{i.addr:x}:{i.text}" for i in ins[n_clk + 1:n_op + 1] if any(sc.waits_on(i, b) for b in set_in)]
    if dep:
        why.append(f"dependent waits between t_s and the signal op: {dep}")
    # waiter exit: TRYWAIT / test_wait PHASECHK ; @P BRA <CS2R>
    o_w = waiter_exits(ins, is_trywait)
    o_w_spin = waiter_exits(ins, is_testwait)
    if not o_w or not o_w_spin:
        why.append("no try_wait / test_wait -> t_w exit path found")
    return res | {"ok": not why, "why": why, "o_s": o_s, "o_w": o_w, "o_w_spin": o_w_spin,
                  "signal_segment": [inst_rec(i) for i in seg + [ins[n_op]]]}



def xw_signal(k, n_issue_pred, want_issue):
    """Signaler segment of a cross-warp kernel: the single UTCBAR, the t_s
    clock read before it, the issued ops in between (executed path), the
    stall cycles t_s->first op and t_s->UTCBAR, and the waiter exit o_w."""
    ins = k.insts
    bars = [n for n, i in enumerate(ins) if i.root == "UTCBAR"]
    if len(bars) != 1:
        return None, [f"{len(bars)} UTCBAR in kernel, want 1"]
    nb = bars[0]
    nclk = max((n for n in range(nb) if "SR_CLOCKLO" in ins[n].text), default=None)
    if nclk is None:
        return None, ["no t_s clock read before the commit"]
    seg = executed_path(ins[nclk:nb + 1])
    if not seg or seg[-1].root != "UTCBAR":
        return None, ["executed path from t_s does not reach the commit"]
    ops = [i for i in seg if n_issue_pred(i)]
    why = []
    set_w = {}
    for x, i in enumerate(seg):
        for b, src in list(set_w.items()):
            if sc.waits_on(i, b):
                why.append(f"{i.addr:x} waits on result SB{b} of {src} inside t_s..commit")
        if i.wbar != 7:
            set_w[i.wbar] = i.text
    if len(ops) != want_issue:
        why.append(f"{len(ops)} issued ops between t_s and commit, want {want_issue}")
    first = next((x for x, i in enumerate(seg) if n_issue_pred(i)), None)
    o_s_first = stall_sum(seg, 0, first) if first is not None else None
    o_s_bar = stall_sum(seg, 0, len(seg) - 1)
    o_w = waiter_exits(ins, is_testwait)
    if not o_w:
        why.append("no test_wait spin -> t_w exit path found")
    return {"o_s_first": o_s_first, "o_s_bar": o_s_bar, "o_w": o_w, "ops": ops,
            "commit_opcode": seg[-1].opcode, "segment": [inst_rec(i) for i in seg]}, why



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True, help="dir with <probe>.sass files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    checks = {"sync_mb": check_sync_mb, "sync_xw": check_sync_xw}
    out = {}
    for probe, fn in checks.items():
        kernels = {}
        p = Path(a.sass) / f"{probe}.sass"
        if p.exists():
            for name, k in sc.parse(p.read_text()).items():
                res = fn(name, k)
                if res is not None:
                    kernels[res["key"]] = res
        out[probe] = {"sass": str(p), "kernels": kernels}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
