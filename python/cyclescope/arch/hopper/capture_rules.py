"""Hopper operand-fact extraction rules (arch table mirror)."""

# only the global->shared forms complete on an mbarrier; TMA stores /
# reductions (UTMASTG, UTMAREDG, UBLKRED) are bulk-group tracked and
# their [URx] pair has no barrier companion word
MBARRIER_COMPANION_ROOTS = frozenset({"UTMALDG", "UBLKCP"})
MBARRIER_SELF_ROOTS = frozenset({"UTCBAR", "UCGABAR", "ARRIVES"})
MBARRIER_BRACKET_ROOTS = frozenset({"SYNCS", "ARRIVES"})


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
    import re
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




def tcoord_rank(opcode: str) -> int:
    import re
    m = re.search(r"\.(\d)D\b", opcode.upper())
    return int(m.group(1)) if m else 0


def tma_coordinate_offset(root: str, bracket_count: int) -> int:
    """UR offset of the coordinate bank from the first [URx] bracket's low
    register (mirrors tools/tracer/arch_hopper.h); -1 = none."""
    return {"UTMALDG": 2, "UTMASTG": 1, "UTMAREDG": 1,
            "UTMAPF": 0}.get(root, -1)
