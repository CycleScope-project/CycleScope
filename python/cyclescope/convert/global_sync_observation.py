"""kind-2 (GSYNC) records -> model.GlobalSyncObservation + lane facts.

One observation per instruction record: the first executing lane is the
representative; every executing lane's before/after words are kept under
``gsync_before``/``gsync_after`` lane keys.  Records are observed, never
paired.
"""
from __future__ import annotations

from cyclescope.convert import model

# opcode class -> observation kind (blackwellsim gsync opcode_class)
_RELEASE_ROOTS = {"STG", "ST", "STS", "STAS", "RED", "REDG", "REDS",
                  "MEMBAR", "FENCE", "ARRIVES", "UTCBAR"}
_ATOMIC_ROOTS = {"ATOM", "ATOMG", "ATOMS"}


def observation_kind(opcode_root: str, gsync_flags: int) -> int:
    if opcode_root in _ATOMIC_ROOTS:
        return 3
    if opcode_root in _RELEASE_ROOTS:
        return 1
    return 2


_ATOMIC_OPS = ("ADD", "EXCH", "CAS", "MIN", "MAX", "INC", "DEC", "AND",
               "OR", "XOR")


def _shape(opcode: str):
    """(op, kind, bits) of a global atomic / reduction / store / load."""
    parts = opcode.upper().split(".")
    op = next((p for p in parts[1:] if p in _ATOMIC_OPS), None)
    kind, bits = "uint", 32
    for p in parts[1:]:
        if p in ("S32", "S64"):
            kind = "sint"
        elif p in ("F32", "F64", "F16", "BF16"):
            kind = "float"
        if p in ("64", "S64", "U64", "F64"):
            bits = 64
    return op, kind, bits


def _signed(value, bits):
    return value - (1 << bits) if value >> (bits - 1) & 1 else value


def _post(op, kind, bits, old, operands):
    """Post-operation value of an atomic / reduction (predecessor
    gsync_pairing._post); None when it cannot be computed."""
    import struct as _st
    mask = (1 << bits) - 1
    if op is None or not operands:
        return None
    a = operands[0] & mask
    if op == "EXCH":
        return a
    if op == "CAS":
        if len(operands) < 2:
            return None
        return operands[1] & mask if old == a else old
    if kind == "float":
        if op != "ADD":
            return None
        f, i = ("<f", "<I") if bits == 32 else ("<d", "<Q")
        fo = _st.unpack(f, _st.pack(i, old & mask))[0]
        fa = _st.unpack(f, _st.pack(i, a))[0]
        return _st.unpack(i, _st.pack(f, fo + fa))[0]
    if op == "ADD":
        return (old + a) & mask
    if op in ("AND", "OR", "XOR"):
        return {"AND": old & a, "OR": old | a, "XOR": old ^ a}[op] & mask
    if op in ("MIN", "MAX"):
        conv = (lambda v: _signed(v, bits)) if kind == "sint" else (
            lambda v: v)
        return (min if op == "MIN" else max)(old, a, key=conv)
    if op == "INC":
        return 0 if old >= a else (old + 1) & mask
    if op == "DEC":
        return a if old == 0 or old > a else (old - 1) & mask
    return None


def _join(words) -> int:
    value = 0
    for i, w in enumerate(words):
        value |= int(w) << (32 * i)
    return value


def observations(rec, row) -> tuple[list, dict]:
    """(list[GlobalSyncObservation], lane dict entries).

    Tracer layout (inject.cu instrument_gsync): the phase-0 record carries
    each executing lane's source operand words followed by the optional
    pre_value words (flags bit 0; the lane's mref0 word read right before
    the instruction); the phase-1 record carries the destination register
    words.  before = pre_value (RED) or the returned old value (ATOM);
    after = post(op, before, operands) for RED/ATOM, the stored operand
    for ST, the destination word for LD.  ``post_known`` carries the GSYN
    flags: bit 0 after known, bit 1 before known."""
    out: list = []
    lane: dict = {}
    if not rec.gsync:
        return out, lane
    root = row.inst.opcode.split(".", 1)[0].upper()
    op, kind, bits = _shape(row.inst.opcode)
    wpv = bits // 32
    kind_code = observation_kind(root, 0)
    address = rec.mrefs[0][0] if rec.mrefs and rec.mrefs[0] else 0
    phase0 = next((g for g in rec.gsync if g[0] == 0), None)
    phase1 = next((g for g in rec.gsync if g[0] == 1), None)
    seq = min(g[4] for g in rec.gsync)
    n = max(len(phase0[3]) if phase0 else 0, len(phase1[3]) if phase1 else 0)
    befores, afters, flags_per_lane = [], [], []
    for i in range(n):
        operands, pre, dst = [], None, None
        if phase0 is not None and i < len(phase0[3]):
            _ph, nwords, gflags, words, _seq = phase0
            lane_words = list(words[i][:nwords])
            if gflags & 1 and len(lane_words) >= wpv:
                pre = _join(lane_words[-wpv:])
                lane_words = lane_words[:-wpv]
            operands = [_join(lane_words[j:j + wpv])
                        for j in range(0, len(lane_words), wpv)]
        if phase1 is not None and i < len(phase1[3]):
            dst = _join(phase1[3][i][:phase1[1]])
        before = after = None
        if root in _ATOMIC_ROOTS:
            before = dst if dst is not None else pre
            after = _post(op, kind, bits, before, operands) \
                if before is not None else None
        elif root.startswith("RED"):
            before = pre
            after = _post(op, kind, bits, pre, operands) \
                if pre is not None else None
        elif root.startswith("ST"):
            after = operands[0] if operands else None
        else:
            after = dst
        befores.append(before if before is not None else 0)
        afters.append(after if after is not None else 0)
        flags_per_lane.append((1 if after is not None else 0)
                              | (2 if before is not None else 0))
    out.append(model.GlobalSyncObservation(
        kind=kind_code, address=int(address),
        value=int(afters[0]) if afters else 0, lane=0, acquire_mode=0,
        post_known=flags_per_lane[0] if flags_per_lane else 0,
        before=int(befores[0]) if befores else 0, sequence=int(seq)))
    lane["gsync_before"] = tuple(befores)
    lane["gsync_after"] = tuple(afters)
    return out, lane
