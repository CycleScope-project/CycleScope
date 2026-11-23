"""DynRecord.ops -> ``model.SemanticOperands``.

Every value is read from the instruction's own captured operand block
(names from the tracer's ``captured_operands`` layout) or from its own
static operand tokens; the decode rules are the predecessors' exact ones
(blackwell ``convert/semantics.py`` / ``facts.py``, hopper
``capture_v3.py``).  Arch-dependent operand positions enter through
``arch/<arch>/capture_rules.py``.  An instruction with no usable fact
keeps ``operands=None``.
"""
from __future__ import annotations

import re

from cyclescope.convert import model
from cyclescope.convert.sass_decode import (
    MissingFact, Unsupported, arrival_and_tx_factors,
    captured_arrival_count, decode_mbarrier_init_expected_arrivals,
    is_complete_tx_form, opcode_width_bytes, tmem_columns_from_operand)

# roots whose op<N> static operand tokens become uniform facts
_OP_ROOTS = (
    "SYNCS", "ARRIVES", "UTMA", "UBLKCP", "UBLKRED", "UTC", "UCGABAR",
    "BAR", "STAS", "REDAS", "UGETNEXTWORKID", "STS", "UTCBAR",
    "DEPBAR", "WARPGROUP", "NANOSLEEP", "UTMACMDFLUSH", "UTMACCTL",
    "LDGDEPBAR", "MEMBAR", "FENCE", "CCTL", "ERRBAR", "CGAERRBAR",
    "MATCH",
)
_TMA_ROOTS = frozenset({"UTMALDG", "UTMASTG", "UTMAREDG", "UTMAPF"})
_BULK_ROOTS = frozenset({"UBLKCP", "UBLKRED"})
_ARRIVE_ROOTS = frozenset({"ARRIVES", "UTCBAR"})
_UR_RE = re.compile(r"^UR(\d+)$")
_R_RE = re.compile(r"^R(\d+)$")
_BRACKET_UR_RE = re.compile(r"\[UR(\d+)[^\]]*\]")
_BRACKET_UR_DISP_RE = re.compile(r"\[UR(\d+)\+(0[xX][0-9a-fA-F]+|\d+)\]")


_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


def _bracket_word(operands_text: str, uniform: dict, lane: dict):
    """Address of the first bracket operand from captured register values:
    sum of R<n> (first executing lane), UR<n>, RZ/URZ (0) and immediate
    terms; None when a register value was not captured."""
    m = _BRACKET_RE.search(operands_text)
    if not m:
        return None
    total = 0
    for term in m.group(1).split("+"):
        t = term.strip().split(".")[0]     # [R4.64] -> R4
        if t in ("RZ", "URZ"):
            continue
        if _R_RE.match(t):
            vals = lane.get(t)
            if not vals:
                return None
            total += int(vals[0])
        elif _UR_RE.match(t):
            v = uniform.get(t)
            if v is None:
                return None
            total += int(v)
        else:
            try:
                total += int(t, 0)
            except ValueError:
                return None
    return total


def poll_barrier_word(rec, row):
    """The barrier word a poll addresses: bracket register sum of the
    first executing lane + displacement (what the converter resolves the
    mbarrier object from); the tracer's mref address where a register
    value is not captured; None for a predicated-off record."""
    if not rec.exec_mask:
        return None
    uniform: dict = {}
    lane: dict = {}
    for name, value in (rec.ops or {}).items():
        if _is_ur_token(name):
            uniform[name] = int(_first(value))
        elif name.startswith("R") and not name.startswith("RANK"):
            lane[name] = _lane_values(value)
    word = _bracket_word(row.operands_text, uniform, lane)
    if word is None and rec.mrefs and rec.mrefs[0]:
        word = int(rec.mrefs[0][0])
    return word


def _bracket_words_per_lane(operands_text: str, uniform: dict,
                            lane: dict):
    """Per-executing-lane bracket addresses (None when the bracket has
    no per-lane register or a value is missing)."""
    m = _BRACKET_RE.search(operands_text)
    if not m:
        return None
    base = 0
    regs = []
    for term in m.group(1).split("+"):
        t = term.strip().split(".")[0]
        if t in ("RZ", "URZ"):
            continue
        if _R_RE.match(t):
            vals = lane.get(t)
            if not vals:
                return None
            regs.append(vals)
        elif _UR_RE.match(t):
            v = uniform.get(t)
            if v is None:
                return None
            base += int(v)
        else:
            try:
                base += int(t, 0)
            except ValueError:
                return None
    if not regs:
        return None
    n = min(len(v) for v in regs)
    return tuple(base + sum(int(v[i]) for v in regs) for i in range(n))


def _bracket_displacement(operands_text: str, ur: int) -> int:
    """Immediate displacement of the ``[UR<ur>+imm]`` bracket operand
    (0 when the bracket has none)."""
    for m in _BRACKET_UR_DISP_RE.finditer(operands_text):
        if int(m.group(1)) == ur:
            return int(m.group(2), 0)
    return 0


def _is_ur_token(op: str) -> bool:
    return op.startswith("UR") and not op.startswith("URANK")


def _lane_values(value) -> tuple:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),)


def _first(value):
    vals = _lane_values(value)
    return vals[0] if vals else None


def _tokens(operands_text: str) -> list[str]:
    return [t.strip() for t in operands_text.rstrip(" ;").split(",")
            if t.strip()]


def _bare_ur_tokens(tokens) -> list[int]:
    """UR<n> operand tokens outside brackets / desc[] forms."""
    out = []
    for token in tokens:
        if "[" in token:
            continue
        m = _UR_RE.match(token)
        if m:
            out.append(int(m.group(1)))
    return out


def _ur(uniform: dict, n: int):
    return uniform.get(f"UR{n}")


def _trailing_operand(tokens, uniform: dict, lane: dict):
    """(token, value) of the last SASS operand: RZ -> 0, R<n> -> captured
    lane-0 value, UR<n> -> captured value, immediate -> int; None when
    the register was not captured."""
    if not tokens:
        return None, None
    token = tokens[-1]
    if token in ("RZ", "URZ"):
        return token, 0
    if _R_RE.match(token):
        vals = lane.get(token)
        return token, (vals[0] if vals else None)
    if _UR_RE.match(token):
        return token, uniform.get(token)
    try:
        return token, int(token, 0)
    except ValueError:
        return token, None


def _init_expected_arrivals(ops_layout, uniform: dict, lane: dict):
    """mbarrier init state word: last captured UR pair that decodes
    (UR<n>,UR<n+1> in ops_layout order, low UR first), else a single UR,
    else the data register's lane-0 value (blackwell ``_init_count``).
    Returns ``(count, value)`` so callers can derive the initial phase
    from the value's high-word bit 31."""
    ur_nums = []
    for op in ops_layout:
        m = re.fullmatch(r"UR(\d+)", op)
        if m:
            ur_nums.append((int(m.group(1)), op))
    for i in reversed(range(len(ur_nums) - 1)):
        (n0, lo), (n1, hi) = ur_nums[i], ur_nums[i + 1]
        if n1 == n0 + 1 and lo in uniform and hi in uniform:
            value = (uniform[hi] << 32) | (uniform[lo] & 0xFFFFFFFF)
            count = decode_mbarrier_init_expected_arrivals(value)
            if count is not None:
                return count, value
    # single-operand fallback: only the data operand (the last layout
    # token) can carry the init state; bracket base URs must not be
    # tried (their values are addresses, not init words)
    if ops_layout:
        last = ops_layout[-1]
        if _is_ur_token(last) and last in uniform:
            count = decode_mbarrier_init_expected_arrivals(uniform[last])
            if count is not None:
                return count, uniform[last]
        if last.startswith("R") and not last.startswith("RANK")                 and lane.get(last):
            count = decode_mbarrier_init_expected_arrivals(lane[last][0])
            if count is not None:
                return count, lane[last][0]
    return None, None


def _phase_token_parity(ops_layout, lane: dict):
    for op in reversed(ops_layout):
        if op.startswith("R") and not op.startswith("RANK") and lane.get(op):
            return (lane[op][0] >> 31) & 1
    return None


def _sts64_value(tokens, lane: dict):
    """STS.64 data pair (R<n>, R<n+1>) lane values -> 64-bit values."""
    if not tokens:
        return None
    m = _R_RE.match(tokens[-1])
    if m is None:
        return None
    base = int(m.group(1))
    lo = lane.get(f"R{base}")
    hi = lane.get(f"R{base + 1}")
    if not lo or not hi:
        return None
    return tuple(((h & 0xFFFFFFFF) << 32) | (l & 0xFFFFFFFF)
                 for l, h in zip(lo, hi))


def _multicast_mask(root: str, opcode: str, tokens, ops_layout,
                    uniform: dict):
    """ctaMask of a multicast form (both archs pack it in the high 16
    bits of the mask/size operand; UTCBAR carries it whole)."""
    upper = opcode.upper()
    if root in _TMA_ROOTS and "MULTICAST" in upper.split("."):
        bracket = set()
        for m in _BRACKET_UR_RE.finditer(" ".join(tokens)):
            bracket.add(int(m.group(1)))
            bracket.add(int(m.group(1)) + 1)
        for n in reversed(_bare_ur_tokens(tokens)):
            if n in bracket:
                continue
            value = _ur(uniform, n)
            if value is not None:
                return (value >> 16) & 0xFFFF
        return None
    if root == "UBLKCP" and "MULTICAST" in upper:
        bare = _bare_ur_tokens(tokens)
        if len(bare) == 1 and _ur(uniform, bare[0]) is not None:
            return (_ur(uniform, bare[0]) >> 16) & 0xFFFF
        return None
    if root == "UTCBAR" and "MULTICAST" in upper.split("."):
        bare = _bare_ur_tokens(tokens)
        if bare and _ur(uniform, bare[-1]) is not None:
            return _ur(uniform, bare[-1])
    return None


def bulk_size_bytes(tokens, uniform: dict):
    """UBLKCP/UBLKRED size operand: the single bare UR token, low 16
    bits in 16-byte units (blackwell ``_bulk_copy_size_bytes``)."""
    bare = _bare_ur_tokens(tokens)
    if len(bare) != 1:
        return None
    value = _ur(uniform, bare[0])
    if value is None:
        return None
    units = value & 0xFFFF
    return units * 16 if units > 0 else None


_BRACKET_UR_RE = re.compile(r"\[UR(\d+)")


def bulk_has_global_side(opcode: str) -> bool:
    """UBLKCP / UBLKRED whose .X.Y modifiers name a global side; the
    shared -> shared (cluster DSMEM) form `.S.S` has none and carries no
    global memory reference."""
    parts = opcode.upper().split(".")
    return len(parts) >= 3 and "G" in (parts[1], parts[2])


def bulk_global_address(opcode: str, tokens, uniform: dict):
    """UBLKCP / UBLKRED global-side base address: the ``.G.S`` / ``.S.G``
    modifiers name the destination / source spaces in bracket order, the
    global bracket's UR pair (lo, lo+1) is the 64-bit address.  None when
    the modifiers or the pair are not available."""
    parts = opcode.upper().split(".")
    if len(parts) < 3 or parts[0] not in ("UBLKCP", "UBLKRED"):
        return None
    if parts[1] == "G":
        index = 0
    elif parts[2] == "G":
        index = 1
    else:
        return None
    brackets = [int(m.group(1)) for m in
                (_BRACKET_UR_RE.search(t) for t in tokens) if m]
    if index >= len(brackets):
        return None
    lo, hi = _ur(uniform, brackets[index]), _ur(uniform, brackets[index] + 1)
    if lo is None or hi is None:
        return None
    return ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)


def _tmem_columns(opcode: str, tokens, uniform: dict):
    """tcgen05.alloc (UTCATOMSWS[.2CTA].FIND_AND_SET[.ALIGN]) column
    count from its last bare UR (count or 32-column granules).  The
    dealloc form (UTCATOMSWS.AND) carries a column *mask*, not a count:
    no column fact (engine frees the CTA's allocation, PTX semantics)."""
    parts = opcode.upper().split(".")
    if parts[0] != "UTCATOMSWS" or "FIND_AND_SET" not in parts:
        return None
    bare = _bare_ur_tokens(tokens)
    if not bare:
        return None
    return tmem_columns_from_operand(_ur(uniform, bare[-1]))


def remote_ranks(ops: dict, cluster_rank: int) -> dict:
    """{mref_index: per-lane ranks} for RANK/BRANK entries where some lane
    names a CTA other than the issuing one (predecessor ``remote`` rule);
    local-only entries are dropped."""
    out = {}
    for name, value in ops.items():
        if not (name.startswith("RANK") or name.startswith("BRANK")):
            continue
        m = re.search(r"(\d+)$", name)
        if m is None:
            continue
        vals = _lane_values(value)
        if any(v >= 0 and v != cluster_rank for v in vals):
            out[int(m.group(1))] = vals
    return out


def operand_facts(rec, row, rules, cluster_rank: int = 0,
                  ranks: dict | None = None
                  ) -> model.SemanticOperands | None:
    """Convert one DynRecord + its StaticRow into SemanticOperands."""
    ops = rec.ops or {}
    opcode = row.opcode_text
    upper = opcode.upper()
    parts = upper.split(".")
    root = parts[0]
    tokens = _tokens(row.operands_text)
    uniform: dict = {}
    lane: dict = {}
    coordinates: list = []
    descriptor: list = []
    tensor_map = None
    after = rec.after_ready

    if root.startswith(_OP_ROOTS):
        for index, token in enumerate(row.inst.operands):
            text = token.strip()
            if text.upper() in ("RZ", "URZ"):
                uniform[f"op{index}"] = 0
                continue
            try:
                uniform[f"op{index}"] = int(text, 0)
            except ValueError:
                pass

    for name, value in ops.items():
        if _is_ur_token(name):
            uniform[name] = int(value)
        elif name == "IDESC":
            uniform["idesc_value"] = int(value)
            descriptor.append(int(value))
        elif name == "GDESC":
            uniform["gdesc_value_lo"] = int(value)
            descriptor.append(int(value))
        elif name == "GDESCH":
            uniform["gdesc_value_hi"] = int(value)
            descriptor.append(int(value))
        elif name == "TMEMD":
            uniform["tmem_d_value"] = int(value)
            descriptor.append(int(value))
        elif name == "CPOL":
            uniform["cache_policy_lo"] = int(value)
        elif name == "CPOLH":
            uniform["cache_policy_hi"] = int(value)
        elif name == "DESC":
            tensor_map = int(value)
        elif name.startswith("R") and not name.startswith("RANK"):
            lane[name] = _lane_values(value)

    # TMA box coordinates: UR(first bracket base + arch offset + k)
    if root in _TMA_ROOTS or root in _BULK_ROOTS:
        rank_m = re.search(r"\.(\d)D(?:\.|$)", upper)
        rank = int(rank_m.group(1)) if rank_m else 0
        if "IM2COL" in parts and rank:
            rank += 2
        brackets = [int(m.group(1)) for m in
                    _BRACKET_UR_RE.finditer(row.operands_text)]
        offset = rules.tma_coordinate_offset(root, len(brackets)) \
            if brackets and rank else -1
        if offset >= 0:
            base = brackets[0] + offset
            vals = [_ur(uniform, base + k) for k in range(rank)]
            if all(v is not None for v in vals):
                coordinates = [(k, v) for k, v in enumerate(vals)]
        if not coordinates:
            # traces from the first tracer build named the bank TCOORD<i>
            legacy = sorted((int(n[6:] or 0), int(v)) for n, v in ops.items()
                            if n.startswith("TCOORD"))
            if legacy and len(legacy) >= rank:
                coordinates = legacy[:rank]

    if ranks is None:
        ranks = remote_ranks(ops, cluster_rank)

    # synchronisation facts only for executing instances: a predicated-off
    # instance's registers hold whatever the warp left there (CU3-C70
    # smoke: exec 0x0 SYNCS.EXCH.64 "named" barrier words at 0x4150 ...
    # and an epilogue STS over them was then mis-promoted)
    executing = rec.exec_mask != 0

    # mbarrier word: UR forms (companion / self) or the bracket word of
    # mref0 at the first executing lane; STAS ([Rn.64] = data destination
    # in Rn, mbarrier word in R(n+1)) names the barrier through the pair's
    # high register, its owner through the DSMEM rank tag (bits 24+)
    mbar = rules.mbarrier_ur_word(ops, row.operands_text, root, _is_ur_token)
    if not executing:
        mbar = None
    elif root == "UBLKCP" and ".G.S" in upper:
        mbar = None     # bulk store (destination global): no mbarrier
    elif root == "STAS":
        pair = re.search(r"\[R(\d+)\.64", row.operands_text)   # [Rn.64+disp]
        hi = lane.get(f"R{int(pair.group(1)) + 1}") if pair else None
        if hi:
            word = int(hi[0])
            uniform["mbarrier_address"] = word
            if word >> 24:
                uniform["owner_rank"] = word >> 24
            # per-lane barrier words: lanes may store into different
            # ranks' barriers (same key as the CLC per-lane arrive)
            if len(set(int(v) for v in hi)) > 1:
                lane["mbarrier_words"] = tuple(hi)
        try:
            width = opcode_width_bytes(opcode, f"{row.inst.pc:#x}")
        except MissingFact:
            width = None
        if width:
            uniform["bytes"] = width * rec.exec_mask.bit_count()
    elif mbar is not None and f"UR{mbar}" in uniform:
        # self roots address the word as [URx+imm]: the barrier is at
        # UR + displacement (ARRIVES.LDGSTSBAR [UR7+0x38000]); companion
        # roots keep the companion UR value whole
        disp = _bracket_displacement(row.operands_text, mbar) \
            if root in rules.MBARRIER_SELF_ROOTS else 0
        uniform["mbarrier_address"] = uniform[f"UR{mbar}"] + disp
    elif executing and root in rules.MBARRIER_BRACKET_ROOTS:
        # bracket forms [Rx+URy+disp] / [Rx+disp] / [URy+disp]: the word is
        # the register sum + displacement from the captured values
        # (predecessor rule); NVBit's SYNCS mref address on sm_103 is off
        # by +0x7FFF, so the mref is only the fallback
        word = _bracket_word(row.operands_text, uniform, lane)
        if word is None and rec.mrefs and rec.mrefs[0]:
            word = int(rec.mrefs[0][0])
        if word is not None:
            uniform["mbarrier_address"] = word
        # per-lane words when the bracket register differs across lanes
        # (CLC response arrive: lane i targets CTA i's barrier)
        words = _bracket_words_per_lane(row.operands_text, uniform, lane)
        if words is None and rec.mrefs and len(rec.mrefs[0]) > 1:
            # register values not captured: the per-lane mref addresses
            # carry the same words (rank tag in bits 24+ for mapa
            # results), the same fallback the uniform word uses
            words = tuple(int(a) for a in rec.mrefs[0])
        if words and len(set(words)) > 1:
            lane["mbarrier_words"] = words
    if "mbarrier_address" in uniform:
        rank = rules.owner_rank_entry(ops, row.operands_text)
        if rank is not None:
            uniform["owner_rank"] = int(rank)
        elif int(uniform["mbarrier_address"]) >> 24:
            # a rank-tagged word (mapa result) names its owner itself
            uniform["owner_rank"] = int(uniform["mbarrier_address"]) >> 24
        elif root in _TMA_ROOTS or root in _BULK_ROOTS or \
                root in ("UTCBAR", "UGETNEXTWORKID") or \
                (root == "SYNCS" and "RED" in parts):
            # shared::cluster operands (TMA / bulk / tcgen05.commit / CLC /
            # the .RED remote-capable arrive): untagged means rank 0
            # (cta_group::2 peers and nvjet's cluster barrier
            # `[URZ+0x34488]` name rank 0's barrier that way); non-RED
            # SYNCS words are shared::cta, so they stay the issuer's own
            uniform["owner_rank"] = 0
    if root in _TMA_ROOTS and rec.mrefs and rec.mrefs[0]:
        uniform["shared_destination"] = int(rec.mrefs[0][0])

    # mbarrier init state (SYNCS.EXCH.64 / MBAR.INIT); STS.64 data pair
    if not executing:
        pass
    elif (root == "SYNCS" and "EXCH" in parts) or upper.startswith("MBAR.INIT"):
        count, value = _init_expected_arrivals(row.ops_layout, uniform,
                                               lane)
        if count is not None:
            uniform["expected_arrivals"] = count
            uniform["initial_phase"] = (value >> 63) & 1
    elif upper in ("STS.64", "STS"):
        # STS.64 data pair -> 64-bit init state; plain STS carries a
        # 32-bit init word in a single data register.  Only a value that
        # decodes as a legal init state produces expected_arrivals
        # (promotion stays gated on a named barrier address downstream)
        if upper == "STS.64":
            values = _sts64_value(tokens, lane)
        else:
            vals = lane.get(tokens[-1]) if tokens and \
                _R_RE.match(tokens[-1]) else None
            values = tuple(vals) if vals else None
        if values:
            count = decode_mbarrier_init_expected_arrivals(values[0])
            if values:
                uniform["store_value_captured"] = 1
            if count is not None:
                uniform["expected_arrivals"] = count
                uniform["initial_phase"] = (values[0] >> 63) & 1
                lane["sts64_value"] = values

    # phase-token parity on wait forms
    if {"PHASECHK", "TRYWAIT", "TEST_WAIT"} & set(parts):
        parity = _phase_token_parity(row.ops_layout, lane)
        if parity is None and tokens and tokens[-1] in ("RZ", "URZ"):
            parity = 0      # literal zero phase token
        if parity is not None:
            uniform["parity"] = parity

    # arrival / expect-tx facts on arrive forms (executing instances only:
    # a predicated-off arrive carries no lane values and arrives nowhere)
    if executing and ((root == "SYNCS" and "ARRIVE" in parts)
                      or root in _ARRIVE_ROOTS):
        try:
            a_factor, t_factor = arrival_and_tx_factors(opcode)
        except Unsupported:
            a_factor = t_factor = None
        if a_factor is not None:
            lanes = rec.exec_mask.bit_count()
            art = re.search(r"(?:^|\.)ART(\d+)(?:\.|$)", upper)
            if art is not None:
                # .ART<n>: the trailing register carries each executing
                # lane's arrival count; tx = n per lane
                tok = tokens[-1] if tokens else None
                if tok == "RZ" or tok == "URZ":
                    uniform["arrival_count"] = 0
                elif tok and _R_RE.match(tok):
                    vals = lane.get(tok)
                    if vals is None:
                        raise MissingFact(
                            f"{row.inst.pc:#x}",
                            f"{tok} per-lane arrival count")
                    uniform["arrival_count"] = sum(int(v) for v in vals)
                elif tok and re.fullmatch(r"UR\d+", tok):
                    uv = uniform.get(tok)
                    if uv is None:
                        raise MissingFact(
                            f"{row.inst.pc:#x}",
                            f"{tok} uniform arrival count")
                    uniform["arrival_count"] = int(uv) * lanes
                else:
                    raise MissingFact(
                        f"{row.inst.pc:#x}",
                        "arrival count operand")
                uniform["transaction_bytes"] = \
                    int(art.group(1)) * lanes
            else:
                uniform["arrival_count"] = \
                    captured_arrival_count(opcode, lanes)
            if t_factor and art is None:
                imm = re.search(r"(?:^|\.)A(\d+)T(\d+)(?:\.|$)", upper)
                if imm is not None and int(imm.group(2)) > 0:
                    uniform["transaction_bytes"] = int(imm.group(2)) * lanes
                else:
                    _tok, tx = _trailing_operand(tokens, uniform, lane)
                    if tx is not None:
                        uniform["transaction_bytes"] = int(tx)
            if is_complete_tx_form(opcode):
                # .A<n>TX: complete_tx bytes (0 arrivals), not expect_tx
                uniform["transaction_complete"] = 1

    # UTMACCTL.IV [URn]: the acquire's base is the 64-bit generic
    # address in the UR pair (publish side records STG chunk bases as
    # generic addresses); leave unset when either UR was not captured
    if executing and root == "UTMACCTL" and "IV" in parts:
        brk = re.search(r"\[UR(\d+)(?:\+(0x[0-9a-fA-F]+|\d+))?\]",
                        row.operands_text)
        if brk:
            lo = uniform.get(f"UR{brk.group(1)}")
            hi = uniform.get(f"UR{int(brk.group(1)) + 1}")
            if lo is not None and hi is not None:
                base = (int(hi) << 32) | int(lo)
                if brk.group(2):
                    base += int(brk.group(2), 0)
                uniform["descriptor_address"] = base

    mask = _multicast_mask(root, opcode, tokens, row.ops_layout, uniform)
    if mask is not None:
        uniform["multicast_mask"] = mask

    if root in _BULK_ROOTS:
        size = bulk_size_bytes(tokens, uniform)
        if size is not None:
            uniform["bytes"] = size

    columns = _tmem_columns(opcode, tokens, uniform)
    if columns is not None:
        uniform["tmem_columns"] = columns
    elif root == "UTCATOMSWS" and "AND" in parts:
        bare = _bare_ur_tokens(tokens)
        mask = _ur(uniform, bare[-1]) if bare else None
        if mask is not None:
            uniform["tmem_dealloc_mask"] = mask

    if root == "NANOSLEEP":
        for op in row.ops_layout:
            if op.startswith("R") and not op.startswith("RANK") \
                    and lane.get(op):
                uniform["requested_ns"] = lane[op][0]
                break

    if not (uniform or lane or ranks or coordinates or descriptor or
            tensor_map is not None or after is not None):
        return None
    return model.SemanticOperands(
        uniform=uniform, lane=lane, ranks=dict(ranks),
        after_ready_mask=after, tensor_map=tensor_map,
        tensor_coordinates=tuple(v for _i, v in coordinates) or None,
        descriptor_words=tuple(descriptor) or None)
