"""Blackwell operand-fact extraction rules (arch table mirror).

All arch-varying answers for ``operand_facts`` live here -- which SASS
operand is the mbarrier word, owner-rank entry names, coordinate rank,
multicast-mask operand position, init-state decode bit fields.
"""

# Opcode roots whose first [URx] operand packs dst|mbarrier: the barrier
# word is URx+1 (tracer arch_table mbarrier_companion_roots).
# only the global->shared forms complete on an mbarrier; TMA stores /
# reductions (UTMASTG, UTMAREDG, UBLKRED) are bulk-group tracked and
# their [URx] pair has no barrier companion word
MBARRIER_COMPANION_ROOTS = frozenset({"UTMALDG", "UBLKCP"})
# Roots whose [URx] operand IS the mbarrier word itself.
MBARRIER_SELF_ROOTS = frozenset({"UTCBAR", "UCGABAR", "ARRIVES"})

# Roots whose mref0 bracket operand carries the barrier word
# ([Rx+URy+disp]): rank fact is BRANK<mref>, word = R + UR + disp.
MBARRIER_BRACKET_ROOTS = frozenset({"SYNCS", "ARRIVES"})

# URANK<n> names: rank of the [UR<n>] barrier operand.  BRANK<m>: rank
# of the bracket word of mref m.  RANK<m>: per-lane rank of mref m.
def owner_rank_entry(ops: dict, operands_text: str) -> int | None:
    import re
    match = re.search(r"\[UR(\d+)", operands_text)
    if match is not None:
        value = ops.get(f"URANK{match.group(1)}")
        if value is not None:
            v = int(value[0] if isinstance(value, (list, tuple)) else value)
            return v if v >= 0 else None
    for name in sorted(ops):
        if name.startswith("BRANK"):
            value = ops[name]
            vals = value if isinstance(value, (list, tuple)) else [value]
            if vals and all(int(v) >= 0 for v in vals):
                return int(vals[0])
            return None
    for name in sorted(ops):
        if name.startswith("RANK"):
            value = ops[name]
            vals = value if isinstance(value, (list, tuple)) else [value]
            if vals and all(int(v) >= 0 for v in vals):
                return int(vals[0]) if len(set(vals)) == 1 else None
            return None
    return None


def mbarrier_ur_word(ops: dict, operands_text: str, root: str,
                     is_ur_token):
    """(kind, ur_base) -> mbarrier word key ``UR<base>``.

    companion roots: first [URx] operand's companion is UR(x+1).
    self roots (UTCBAR): the word is UR(x)."""
    import re
    if root == "UGETNEXTWORKID":
        # CLC query: [URa] = response buffer, [URb] = response mbarrier
        # (the last bracket; the SELFCAST form has only the mbarrier)
        brackets = re.findall(r"\[UR(\d+)", operands_text)
        return int(brackets[-1]) if brackets else None
    if root in MBARRIER_SELF_ROOTS:
        match = re.search(r"\[UR(\d+)", operands_text)
        if match is not None:
            return int(match.group(1))
        return None
    if root in MBARRIER_COMPANION_ROOTS:
        for token in re.finditer(r"\[UR(\d+)", operands_text):
            return int(token.group(1)) + 1
        return None
    return None


def tma_coordinate_offset(root: str, bracket_count: int) -> int:
    """UR offset of the coordinate bank from the first [URx] bracket's low
    register (mirrors tools/tracer/arch_blackwell.h); -1 = none."""
    if root in ("UTMALDG", "UBLKCP"):
        return 2
    if root in ("UTMASTG", "UTMAREDG"):
        return 1
    if root == "UTMAPF":
        return 0 if bracket_count >= 2 else 2
    return -1
