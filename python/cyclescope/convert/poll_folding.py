"""Polling-fold rules for one warp's dynamic record stream.

Ported from blackwellsim convert/fold.py + gsync._instruction_drops:
a PHASECHK run of same (object, parity) whose final sample reports ready
folds to the surviving ready poll when every gap instruction is
side-effect-free; a global-sync poll run (same address, value unchanged)
folds its no-op-reader repeats by the same gap rule extended with
``CCTL.*IVALL*`` and constant-bank loads.
"""
from __future__ import annotations

_NO_SIDE_EFFECT = {
    "BRA", "BRX", "BSSY", "BSYNC", "NANOSLEEP", "VOTEU", "VOTE",
    "ISETP", "FSETP", "DSETP", "PLOP3", "LOP3", "SEL", "FSEL", "MOV",
    "UMOV", "IMAD", "IADD3", "VIADD", "PRMT", "FLO", "BREV", "POPC",
    "UPOPC", "S2R", "S2UR", "CS2R", "LEA", "ULEA", "UMUF", "SHF", "USHF",
    "I2I", "F2F", "R2UR", "NOP", "USCHED", "WARPSYNC", "YIELD",
    # uniform-datapath ALU (same side-effect class as the vector ones)
    "UIADD3", "UISETP", "ULOP3", "USEL", "UPRMT", "UIMAD", "UFLO",
}

_GSYNC_GAP_EXTRA = {"LDC", "ULDC", "LDCU"}


def poll_ready(rec) -> bool | None:
    if rec.after_ready is None:
        return None
    return (rec.after_ready & rec.exec_mask) == rec.exec_mask


def _r_value(rec, op: str):
    value = rec.ops.get(op)
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        return int(value[0]) if value else 0
    return int(value)


def _poll_key(rec, row) -> tuple:
    addr = rec.mrefs[0][0] if rec.mrefs and rec.mrefs[0] else 0
    parity = 0
    for op in reversed(row.ops_layout):
        if op.startswith("R") and not op.startswith(("UR", "RANK")):
            parity = (_r_value(rec, op) >> 31) & 1
            break
    return (addr, parity)


def _gap_free(opcode: str, root: str, gsync: bool) -> bool:
    if root in _NO_SIDE_EFFECT:
        return True
    if gsync:
        if root in _GSYNC_GAP_EXTRA:
            return True
        if root == "CCTL" and "IVALL" in opcode.upper().split(".")[1:]:
            return True
    return False


def fold_warp(records, static_of) -> tuple[set, int]:
    """records: per-warp DynRecord list in issue order.

    ``static_of(rec)`` -> StaticRow for the record's (function_id, idx).
    Returns (folded positions, dropped_poll_count).
    """
    folded: set = set()
    polls = [(i, r) for i, r in enumerate(records)
             if "PHASECHK" in static_of(r).inst.opcode.upper()]
    dropped = 0
    pos = 0
    while pos < len(polls):
        i0, first = polls[pos]
        key = _poll_key(first, static_of(first))
        if poll_ready(first) is not False:
            pos += 1
            continue
        run = [(i0, first)]
        j = pos + 1
        while j < len(polls) \
                and _poll_key(polls[j][1], static_of(polls[j][1])) == key \
                and poll_ready(polls[j][1]) is False:
            run.append(polls[j])
            j += 1
        if j >= len(polls) \
                or _poll_key(polls[j][1], static_of(polls[j][1])) != key \
                or poll_ready(polls[j][1]) is not True:
            pos = j
            continue
        end_pos = polls[j][0]
        # fold the maximal clean suffix of the run — scan gaps from
        # the ready poll backwards; stop at the first dirty gap
        seq = run + [polls[j]]
        start = None
        for q in range(len(seq) - 1, 0, -1):
            lo, hi = seq[q - 1][0], seq[q][0]
            clean = all(
                _gap_free(static_of(records[k]).opcode_text,
                          static_of(records[k]).inst.opcode
                          .split(".", 1)[0].upper(), False)
                for k in range(lo + 1, hi))
            if not clean:
                break
            start = q - 1
        fold_ok = start is not None
        if fold_ok:
            for posn, _r in run[start:]:
                folded.add(posn)
                dropped += 1
            folded.update(range(run[start][0] + 1, end_pos))
        pos = j + 1 if fold_ok else j
    return folded, dropped
