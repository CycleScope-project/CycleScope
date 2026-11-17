#!/usr/bin/env python3
"""SASS checks for the shared (smem) probe.

check_smem and its helpers copied verbatim from blackwell-prototype
tools/microbench/sass_check.py; blackwellsim imports are optional.
"""

import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc

try:
    from blackwellsim.sass_isa import opcode_facts  # noqa: E402
except Exception:
    opcode_facts = None


def mtx_facts(opcode: str):
    """Local LDSM/STSM matrix facts (sass_isa may lack STSM rows).
    Encoding: <root>.16.M{8,88}.N[.T]; T in M... = transposed."""
    f = opcode_facts(opcode) if opcode_facts else {}
    if f:
        return f
    m = re.search(r"M(T?)8+(?:\.(\d))?", opcode)
    if not m:
        return {}
    return {"width_bytes": 2, "tensor_m": 8, "tensor_n": 8,
            "matrix_count": int(m.group(2) or 1),
            "matrix_transposed": m.group(1) == "T"}


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


def check_smem(name, k):
    mm = re.match(
        r"_Z5k_mtxILi(\d)ELi([124])ELi(\d+)ELb([01])EEv\w+$", name)
    if mm:
        op, x, kk, transposed = (int(v) for v in mm.groups())
        root = "LDSM" if op == 0 else "STSM"
        res = {
            "key": f"matrix/{root.lower()}.x{x}.mt{transposed}/{kk}",
            "u": kk, "matrix_count": x,
            "matrix_transposed": transposed,
        }
        w = single_window(k)
        if w is None:
            return res | {"ok": False, "why": ["no clock window"]}
        ops = [i for i in w if i.root == root]
        why = []
        if len(ops) != kk:
            why.append(f"{len(ops)} {root}, want {kk}")
        for ins in ops:
            facts = mtx_facts(ins.opcode)
            actual = (facts.get("width_bytes"), facts.get("tensor_m"),
                      facts.get("tensor_n"), facts.get("matrix_count"),
                      facts.get("matrix_transposed"))
            expected = (2, 8, 8, x, transposed)
            if actual != expected:
                why.append(f"{ins.opcode} matrix facts {actual} != {expected}")
        # rep-loop counter updates (VIADD/UIADD3/UISETP/ISETP) and pin moves are constant per window
        other = sorted({i.opcode for i in w if i.root not in (root, "NOP", "BAR", "BRA", "BSSY", "BSYNC", "ISETP",
                                                                   "LOP3", "WARPSYNC", "VIADD", "IADD3", "UIADD3",
                                                                   "UISETP", "MOV", "IMAD")})
        if other:
            why.append(f"other instructions in window: {other}")
        return res | {"ok": not why, "why": why, "opcodes": sorted({i.opcode for i in ops})}
    m = re.match(
        r"_Z12k_ldsm_chainILi(\d)ELi(\d+)ELb([01])EEv\w+$", name)
    if m:
        x, u, transposed = (int(v) for v in m.groups())
        res = {"key": f"smem/ldsm_chain/x{x}.mt{transposed}/{u}",
               "u": u, "matrix_count": x,
               "matrix_transposed": transposed}
        w = single_window(k)
        if w is None:
            return res | {"ok": False, "why": ["no clock window"]}
        ops = [i for i in w if i.root == "LDSM"]
        why = []
        if len(ops) != u:
            why.append(f"{len(ops)} LDSM, want {u}")
        for ins in ops:
            facts = mtx_facts(ins.opcode)
            actual = (facts.get("matrix_count"),
                      facts.get("matrix_transposed"))
            if actual != (x, transposed):
                why.append(
                    f"{ins.opcode} matrix facts {actual} != "
                    f"{(x, transposed)}")
        return res | {"ok": not why, "why": why,
                      "opcodes": sorted({i.opcode for i in ops})}
    m = re.match(r"_Z11k_lds_chainILi(\d+)ELi(\d+)EEv\w+$", name)
    if m:
        wb, u = int(m.group(1)), int(m.group(2))
        res = {"key": f"smem/lds_chain/w{wb}/{u}", "u": u}
        w = single_window(k)
        if w is None:
            return res | {"ok": False, "why": ["no clock window"]}
        ops = [i for i in w if i.root == "LDS"]
        why = []
        if len(ops) != u:
            why.append(f"{len(ops)} LDS, want {u}")
        pos = [w.index(i) for i in ops]
        for a, b in zip(pos, pos[1:]):
            if not any(sc.waits_on(x, w[a].wbar) for x in w[a + 1:b + 1]):
                why.append(f"LDS@{w[b].addr:x} not ordered after LDS@{w[a].addr:x}")
        if not no_spill(w):
            why.append("local memory in window")
        return res | {"ok": not why, "why": why, "opcode": ops[0].opcode if ops else None,
                      "stall_sum": stall_sum(w)}
    m = re.match(r"_Z7k_issueILi(\d)ELi(\d+)EEv\w+$", name)
    if m:
        op, kk = int(m.group(1)), int(m.group(2))
        root = "STS" if op == 1 else "LDS"
        name_ = {0: "lds", 1: "sts", 2: "lds_nv"}[op]
        res = {"key": f"smem/issue/{name_}/{kk}", "u": kk}
        w = single_window(k)
        if w is None:
            return res | {"ok": False, "why": ["no clock window"]}
        ops = [i for i in w if i.root == root]
        why = []
        if len(ops) != kk:
            why.append(f"{len(ops)} {root} in window, want {kk}")
        bars = {i.wbar for i in ops if i.wbar != 7}
        for n, i in enumerate(ops[1:], 1):
            if any(sc.waits_on(i, prev.wbar) for prev in ops[:n] if prev.wbar != 7):
                why.append(f"{root}@{i.addr:x} waits on an earlier in-window {root}")
        if not any(i.opcode.startswith("BAR.SYNC") for i in w):
            why.append("no bar.sync closing the window")
        return res | {"ok": not why, "why": why, "opcode": ops[0].opcode if ops else None}
    m = re.match(r"_Z5k_visILi(\d)EEv\w+$", name)
    if m:
        sn = int(m.group(1))
        res = {"key": f"smem/vis/{sn}", "u": sn}
        ins = k.insts
        bars = [n for n, i in enumerate(ins) if i.opcode.startswith("BAR.SYNC")]
        if len(bars) != 1:
            return res | {"ok": False, "why": [f"{len(bars)} BAR.SYNC"]}
        nb = bars[0]
        n_s = max((n for n in range(nb) if "SR_CLOCKLO" in ins[n].text), default=None)
        n_w = min((n for n in range(nb + 1, len(ins)) if "SR_CLOCKLO" in ins[n].text), default=None)
        if n_s is None or n_w is None:
            return res | {"ok": False, "why": ["clock reads around BAR.SYNC not found"]}
        record_pcs(k, n_s, n_w)
        seg = executed_path(ins[n_s:nb + 1])
        sts = [x for x, i in enumerate(seg) if i.root == "STS"]
        why = []
        if len(sts) != sn:
            why.append(f"{len(sts)} STS between t_s and BAR, want {sn}")
        first = sts[0] if sts else len(seg) - 1
        return res | {"ok": not why, "why": why, "o_s": stall_sum(seg, 0, first),
                      "o_s_bar": stall_sum(seg, 0, len(seg) - 1), "o_w": stall_sum(ins, nb, n_w),
                      "segment": [inst_rec(i) for i in seg]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True, help="dir with <probe>.sass files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    p = Path(a.sass) / "smem.sass"
    kernels = {}
    if p.exists():
        for name, k in sc.parse(p.read_text()).items():
            res = check_smem(name, k)
            if res is not None:
                kernels[res["key"]] = res
    out["smem"] = {"sass": str(p), "kernels": kernels}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
