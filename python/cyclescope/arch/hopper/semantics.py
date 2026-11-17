"""Hopper arch semantics (2b-3): WARPGROUP plan + WGMMA rows.

Facts only: opcode text, static operand tokens, register names.  The
warpgroup FIFO itself (tracked units, drained/retained lists) is the
engine's job — the converter emits per-warp group ordinals, opener and
commit flags, and the keep count.

Predecessor sources: hoppersim/convert.py _plan_warpgroups (L3553-3640),
_depbar_keep (~L3648), _gmma_has_gsb0; strict_timing.opcode_facts
(datatype/shape); profile.tensor_source_mode_from_payload (~L4993);
convert.py accumulator span (64xN -> N/2 GPRs, ~L1110).
"""

from __future__ import annotations

import re

from ...convert.model import DynamicInstruction, StaticInstruction
from ...convert.semantics_common import (
    ConversionError, SemanticOperation, SemanticRow, SemanticsContext,
    missing, unsupported, _where)
from ...job.semantics import (SemanticsKind, SEMA_FLAG_GROUP_COMMIT,
                              SEMA_FLAG_GROUP_OPENER)

_GMMA_ROOTS = frozenset({"HGMMA", "WGMMA", "QGMMA", "IGMMA"})
_WG_CTX_GROUPS = "hopper_wg_groups"      # wslot -> current group ordinal|None
_WG_CTX_COUNT = "hopper_wg_count"        # wslot -> groups opened so far
_WG_CTX_COMMITS = "hopper_wg_commits"    # wslot -> GROUP_COMMIT count

_TENSOR_N_RE = re.compile(r"64x(\d+)x", re.IGNORECASE)
_DTYPE_TOKEN = re.compile(r"(?:[BUSF]\d+|BF16|TF32|E4M3|E5M2)")
_GSB0_TOKEN = re.compile(r"^gsb0$", re.IGNORECASE)
_REG_TOKEN = re.compile(r"^(?:RZ|R\d+|URZ|UR\d+|P\d+|PT|UP\d+)$",
                        re.IGNORECASE)
_NUMBER_TOKEN = re.compile(r"^(0x[0-9a-fA-F]+|\d+)$")

# dtype -> u32 code (conversion-local contract; see formats.md §4)
DTYPE_CODES = {
    "": 0, "f16": 1, "bf16": 2, "tf32": 3, "f32": 4, "s32": 5,
    "e4m3": 6, "e5m2": 7, "f8": 8, "u8": 9, "s8": 10, "f64": 11,
}


def _dtype_fields(opcode: str):
    """(input_dtype, accumulator_dtype) from the GMMA type segments."""
    tokens = opcode.upper().split(".")
    types = [t.lower() for t in tokens[1:] if _DTYPE_TOKEN.fullmatch(t)]
    is_igmma = opcode.upper().startswith("IGMMA")
    acc = "s32" if is_igmma else (types[0] if types else "f32")
    if len(types) > 1:
        inp = types[1]
    elif is_igmma:
        inp = ""
    elif opcode.upper().startswith("HGMMA"):
        inp = "f16"
    else:
        inp = ""
    return DTYPE_CODES.get(inp, 0), DTYPE_CODES.get(acc, 0)


DTYPE_NAMES = {v: k for k, v in DTYPE_CODES.items()}


def _shape_packed(opcode: str) -> int:
    match = re.search(r"(?:^|\.)(\d+)X(\d+)X(\d+)(?:\.|$)",
                      opcode.upper())
    if match is None:
        return 0
    m, n, k = (int(g) for g in match.groups())
    return (m << 32) | (n << 16) | k


def _operand_tokens(static: StaticInstruction) -> list[str]:
    tokens = list(static.operands or ())
    if tokens:
        return [str(t).strip() for t in tokens if str(t).strip()]
    return []


def _source_mode(static: StaticInstruction, where: str) -> int:
    """1 ss / 2 rs (tensor_source_mode_from_payload)."""
    tokens = _operand_tokens(static)
    if len(tokens) < 2:
        raise missing("wgmma_source_operands", where)
    first_source = tokens[1]
    second_source = tokens[2] if len(tokens) > 2 else ""
    if first_source.lower().startswith("gdesc["):
        return 1
    if (_REG_TOKEN.fullmatch(first_source) is not None
            and second_source.lower().startswith("gdesc[")):
        return 2
    raise unsupported("wgmma_source_operands", where)


def _acc_registers(static: StaticInstruction, opcode: str,
                   acc_dtype: int, where: str):
    """(begin, count) of the accumulator destination register span.
    32-bit accumulators take N/2 GPRs, 16-bit (f16/bf16) take N/4."""
    n_match = _TENSOR_N_RE.search(opcode)
    n = int(n_match.group(1)) if n_match else 2
    count = max(1, n // (2 if acc_dtype in (4, 5) else 4))
    dsts = static.destination_registers or ()
    begin = None
    for name in dsts:
        m = re.fullmatch(r"R(\d+)", str(name), re.IGNORECASE)
        if m:
            begin = int(m.group(1))
            break
    if begin is None:
        tokens = _operand_tokens(static)
        m = re.fullmatch(r"R(\d+)", tokens[0], re.IGNORECASE) \
            if tokens else None
        if m:
            begin = int(m.group(1))
    if begin is None and tokens and tokens[0].strip() == "RZ":
        # RZ destination writes no accumulator registers: a fact, count 0
        return 0, 0
    if begin is None:
        raise missing("wgmma_accumulator_register", where)
    return begin, count


def _gsb0(static: StaticInstruction) -> bool:
    return any(_GSB0_TOKEN.fullmatch(str(t).strip())
               for t in static.operands or ())


def _wg_groups(ctx: SemanticsContext) -> dict:
    return ctx.extra.setdefault(_WG_CTX_GROUPS, {})


def _wg_counts(ctx: SemanticsContext) -> dict:
    return ctx.extra.setdefault(_WG_CTX_COUNT, {})


def _wg_commits(ctx: SemanticsContext) -> dict:
    return ctx.extra.setdefault(_WG_CTX_COMMITS, {})


def _open_group(ctx: SemanticsContext, wslot: int,
                folded: bool) -> tuple[int, int]:
    counts = _wg_counts(ctx)
    group = counts.get(wslot, 0)
    counts[wslot] = group + 1
    _wg_groups(ctx)[wslot] = group
    return group, SEMA_FLAG_GROUP_OPENER if folded else 0


def decode(static: StaticInstruction, dyn: DynamicInstruction,
           ctx: SemanticsContext, operation: SemanticOperation,
           cta_index: int, warp_slot: int):
    where = _where(static)
    opcode = static.opcode.upper()
    root = opcode.split(".", 1)[0]
    groups = _wg_groups(ctx)

    if operation is SemanticOperation.warpgroup_arrive:
        group, _ = _open_group(ctx, warp_slot, folded=False)
        return SemanticRow(SemanticsKind.warpgroup_arrive,
                           fields={"group": group, "pending_allowed": 0})

    if operation is SemanticOperation.wgmma:
        current = groups.get(warp_slot)
        if current is None:
            current, flags = _open_group(ctx, warp_slot, folded=True)
        else:
            flags = 0
        input_dtype, acc_dtype = _dtype_fields(opcode)
        acc_begin, acc_count = _acc_registers(static, opcode,
                                              acc_dtype, where)
        fields = {"group": current,
                  "accumulator_registers_begin": acc_begin,
                  "accumulator_registers_count": acc_count,
                  "shape_packed": _shape_packed(opcode),
                  "input_dtype": input_dtype,
                  "accumulator_dtype": acc_dtype,
                  "source_mode": _source_mode(static, where)}
        if _gsb0(static):
            flags |= SEMA_FLAG_GROUP_COMMIT
            groups[warp_slot] = None
            commits = _wg_commits(ctx)
            commits[warp_slot] = commits.get(warp_slot, 0) + 1
        return SemanticRow(SemanticsKind.wgmma, flags=flags,
                           fields=fields)

    if operation is SemanticOperation.warpgroup_wait:
        tokens = _operand_tokens(static)
        has_sb_token = any(re.fullmatch(r"gsb\d+", str(t).strip(),
                                        re.IGNORECASE)
                           for t in tokens)
        if has_sb_token and not _gsb0(static):
            raise unsupported("warpgroup_scoreboard", where)
        numbers = [int(t, 0) for t in tokens
                   if _NUMBER_TOKEN.fullmatch(str(t).strip())]
        if opcode.startswith("WARPGROUP.WAIT"):
            keep = 0              # predecessor returns 0 for non-DEPBAR
        elif not numbers:
            raise missing("warpgroup_wait_keep", where)
        else:
            keep = numbers[-1]
        if keep > 7:
            raise unsupported("warpgroup_wait_keep", where)
        return SemanticRow(SemanticsKind.warpgroup_wait,
                           fields={"group": _wg_commits(ctx)
                                   .get(warp_slot, 0),
                                   "pending_allowed": keep})

    return None


def local_shared_address(address: int) -> int:
    """Strip the DSMEM cluster-rank tag (rank << 24) from a shared
    address, leaving the CTA-local offset (S32 window)."""
    return int(address) & 0xFFFFFF
