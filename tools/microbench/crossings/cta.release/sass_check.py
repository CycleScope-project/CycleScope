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


def check_sync_bar(name, k):
    ins = k.insts
    m = (re.match(r"_Z5k_barILi(\d+)ELi(\d)EEv\w+$", name) or re.match(r"_Z5k_barILi(\d+)EEv\w+$", name)
         or re.match(r"_Z8k_barallILi(\d+)EEv\w+$", name))
    if m:
        nw = int(m.group(1))
        allw = name.startswith("_Z8k_barall")
        stg = len(m.groups()) > 1 and m.group(2) == "1"
        res = {"key": f"sync_bar/{'cta_all' if allw else 'cta_stg' if stg else 'cta'}/w{nw}", "warps": nw}
        bars = [n for n, i in enumerate(ins) if i.opcode.startswith("BAR.SYNC")]
        if allw:
            # the timed barrier is the first one (the second follows the rep loop)
            bars = bars[:1]
        if len(bars) != 1:
            return res | {"ok": False, "why": [f"{len(bars)} BAR.SYNC"]}
        nb = bars[0]
        n_s = max((n for n in range(nb) if "SR_CLOCKLO" in ins[n].text), default=None)
        n_w = min((n for n in range(nb + 1, len(ins)) if "SR_CLOCKLO" in ins[n].text), default=None)
        if n_s is None or n_w is None:
            return res | {"ok": False, "why": ["clock reads around BAR.SYNC not found"]}
        record_pcs(k, n_s, n_w)
        seg = executed_path(ins[n_s:nb + 1])
        return res | {"ok": seg[-1] is ins[nb], "why": [] if seg[-1] is ins[nb] else ["t_s path does not reach BAR"],
                      "o_s": stall_sum(seg, 0, len(seg) - 1), "o_w": stall_sum(ins, nb, n_w),
                      "segment": [inst_rec(i) for i in seg + ins[nb + 1:n_w + 1]]}
    m = re.match(r"_Z6k_cbarILi(\d+)ELi(\d)EEv\w+$", name)
    if not m:
        return None
    cs, relaxed = int(m.group(1)), int(m.group(2))
    res = {"key": f"sync_bar/cluster/c{cs}/{'relaxed' if relaxed else 'release'}", "cluster": cs}
    arv = [n for n, i in enumerate(ins) if i.root.startswith("UCGABAR") and "ARV" in i.opcode]
    wt = [n for n, i in enumerate(ins) if i.root.startswith("UCGABAR") and "WAIT" in i.opcode]
    if len(arv) != 1 or len(wt) != 1:
        return res | {"ok": False, "why": [f"{len(arv)} UCGABAR_ARV / {len(wt)} UCGABAR_WAIT"],
                      "opcodes": sorted({i.opcode for i in ins if i.root.startswith("UCGABAR")})}
    na, nw_ = arv[0], wt[0]
    n_s = max((n for n in range(na) if "SR_CLOCKLO" in ins[n].text), default=None)
    n_w = min((n for n in range(nw_ + 1, len(ins)) if "SR_CLOCKLO" in ins[n].text), default=None)
    if n_s is None or n_w is None:
        return res | {"ok": False, "why": ["clock reads around the cluster barrier not found"]}
    record_pcs(k, n_s, n_w)
    seg = executed_path(ins[n_s:na + 1])
    return res | {"ok": True, "why": [], "o_s": stall_sum(seg, 0, len(seg) - 1), "o_w": stall_sum(ins, nw_, n_w),
                  "arv_to_wait": stall_sum(ins, na, nw_), "opcodes": [ins[na].opcode, ins[nw_].opcode],
                  "segment": [inst_rec(i) for i in seg + ins[na + 1:n_w + 1]]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"sync_bar": {"kernels": {}}}
    p = Path(a.sass) / "sync_bar.sass"
    if p.exists():
        for name, k in sc.parse(p.read_text()).items():
            r = check_sync_bar(name, k)
            if r is not None: out["sync_bar"]["kernels"][r["key"]] = r
    Path(a.out).write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
