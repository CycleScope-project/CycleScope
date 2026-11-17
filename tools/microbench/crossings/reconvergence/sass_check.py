#!/usr/bin/env python3
import argparse, json, re
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


# protocol-v2 gate: the single CS2R window of every kernel must contain only
# this set; any LDC/ULDC/LDCU/S2R/load/store/LDL/STL fails by exclusion.
ALLOWED = {"FFMA", "BSSY", "BSYNC", "BRA", "WARPSYNC", "ISETP", "IADD3", "VIADD",
           "IMAD", "LOP3", "PLOP3", "MOV", "SEL", "BMOV", "NOP", "YIELD"}
FFMA_PATHS = {"div": 2, "uni": 2, "div2": 4, "div3": 3}
MIN_SYNC = {"div": (1, 1), "div2": (2, 2), "div3": (1, 1), "uni": (0, 0)}


def check_ctrl(name, k):
    m = re.match(r"_Z\d+k_(div|uni|div2|div3)ILi(\d+)ELi(\d+)EEv", name)
    if not m:
        return None
    kind, b, n = m.group(1), int(m.group(2)), int(m.group(3))
    res = {"key": f"ctrl/{kind}.b{b}/{n}", "b": b, "n": n}
    w = single_window(k)
    if w is None:
        return res | {"ok": False, "why": ["no clock window"]}
    why = []
    note = []
    ff = [i for i in w if i.root == "FFMA"]
    other = sorted({i.opcode for i in w if i.root not in ALLOWED})
    if other:
        why.append(f"other instructions in window: {other}")
    want = FFMA_PATHS[kind] * b
    if len(ff) != want:
        why.append(f"{len(ff)} FFMA in the window, want {FFMA_PATHS[kind]} paths of {b}")
    if any(i.pred is not None for i in ff):
        why.append("predicated FFMA: ptxas if-converted the branch")
    n_bssy = sum(1 for i in w if i.root == "BSSY")
    n_bsync = sum(1 for i in w if i.root == "BSYNC")
    want_bssy, want_bsync = MIN_SYNC[kind]
    if n_bssy < want_bssy:
        why.append(f"{n_bssy} BSSY in the window, want >= {want_bssy}")
    if n_bsync < want_bsync:
        why.append(f"{n_bsync} BSYNC in the window, want >= {want_bsync}")
    if kind == "uni" and n_bssy:
        note.append(f"uni kernel contains {n_bssy} BSSY (reported, not a failure)")
    return res | {"ok": not why, "why": why, "note": note, "n_ffma_static": len(ff),
                  "n_bssy": n_bssy, "n_bsync": n_bsync, "stall_sum": stall_sum(w),
                  "segment": [inst_rec(i) for i in w]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"ctrl": {"kernels": {}}}
    p = Path(a.sass) / "ctrl2.sass"
    for name, k in sc.parse(p.read_text()).items():
        r = check_ctrl(name, k)
        if r is not None:
            out["ctrl"]["kernels"][r["key"]] = r
    n = len(out["ctrl"]["kernels"])
    # 0 kernels matched = gate failure
    out["ctrl"]["gate"] = {"matched": n, "ok": n > 0}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
