"""Blackwell arch semantics (2b-3): UMMA / TMEM / CLC rows.

Facts only: opcode text, static operand tokens, captured operand values
(idesc, UR words, multicast mask), memory-reference lane addresses.
Generation/phase resolution stays with the engine.

Predecessor sources: blackwellsim/sass_isa.py (umma_descriptor_facts,
umma_kind_of, tmem_shape_and_repeat, tmem_columns_from_immediate,
infer_tmem_alloc_columns, tmem_alloc_op, umma_a_src_of),
convert/semantics.py _utcbar (L1568) / _clc (L1509), payload.py isa_row.
"""

from __future__ import annotations

import re

from ...convert import sass_decode
from ...convert.model import DynamicInstruction, StaticInstruction
from ...convert.semantics_common import (
    _mbarrier_object,
    SemanticOperation, SemanticRow, SemanticsContext,
    missing, unsupported, _first_op, _object_address, _where)
from ...job.semantics import (SemanticsKind, GENERATION_UNRESOLVED,
                              SEMA_FLAG_STAT, SEMA_UNKNOWN)

UMMA_ISSUE_ROOTS = frozenset({"UTCHMMA", "UTCQMMA", "UTCOMMA"})

_UMMA_KIND_CODES = {
    "f16": 1, "tf32": 2, "f8f6f4": 3, "mxf8f6f4": 4,
    "mxf4": 5, "mxf4nvf4": 6,
}
_TMEM_SHAPE_CODES = {
    "16x64b": 1, "16x128b": 2, "16x256b": 3, "32x32b": 4,
}
# UTCCP .T/.S direction modifiers carry Nx32dpMbit shapes; encode
# (N << 16) | M so the engine sees both dimensions (e.g.
# 4x32dp128bit -> 262272).
_UTCCP_SHAPE_RE = re.compile(r"^(\d+)x32dp(\d+)bit$")
_UTCCP_SB_SHAPE_RE = re.compile(r"^(\d+)dp(\d+)bit$")
_KNOWN_TMEM_SHAPES = {
    "16X64B": "16x64b", "16X128B": "16x128b",
    "16X256B": "16x256b", "32X32B": "32x32b",
    "16DP64BIT": "16x64b", "16DP128BIT": "16x128b",
    "16DP256BIT": "16x256b", "32DP32BIT": "32x32b",
}
_TMEM_COLUMN_SET = sass_decode.TMEM_COLUMN_SET
_TMEM_GRANULE = sass_decode.TMEM_GRANULE

_XN = re.compile(r"^x(\d+)$", re.IGNORECASE)
_TMEM_OP = re.compile(r"tmem\[[^\]]+\]", re.IGNORECASE)
_GDESC_OP = re.compile(r"gdesc\[[^\]]+\]", re.IGNORECASE)
_BRACKET_UR = re.compile(r"\[UR(\d+)[^\]]*\]", re.IGNORECASE)
_BARE_UR = re.compile(r"^UR(\d+)$", re.IGNORECASE)


def _mods(opcode: str) -> list[str]:
    return opcode.upper().split(".")[1:]


def _umma_kind_of(opcode: str):
    root = opcode.split(".", 1)[0].upper()
    mods = _mods(opcode)
    if root == "UTCHMMA":
        return "tf32" if "TF32" in mods else "f16"
    if root == "UTCQMMA":
        if "BLOCK16" in mods or any(m.startswith("MX") for m in mods):
            return "mxf8f6f4"
        return "f8f6f4"
    if root == "UTCOMMA":
        return "mxf4nvf4" if "BLOCK16" in mods else "mxf4"
    return None


_UMMA_BASE_K = {"f16": 16, "tf32": 8, "f8f6f4": 32, "mxf8f6f4": 32,
                "mxf4": 64, "mxf4nvf4": 64}


def _umma_descriptor_facts(opcode: str, value: int) -> dict:
    """Port of sass_isa.umma_descriptor_facts + umma_layout
    decode_umma_axes (idesc bit fields).  ``value`` is the captured
    idesc control word."""
    root = opcode.split(".", 1)[0].upper()
    kind = _umma_kind_of(opcode)
    if root == "UTCHMMA":
        types = ((value >> 7) & 7, (value >> 10) & 7)
        kind = "tf32" if types == (2, 2) else (
            "f16" if all(t in (0, 1) for t in types) else None)
    elif root == "UTCQMMA":
        kind = "mxf8f6f4" if value & (1 << 23) else "f8f6f4"
    elif root == "UTCOMMA":
        kind = ("mxf4nvf4" if "BLOCK16" in _mods(opcode)
                or not (value & (1 << 23)) else "mxf4")
    control = value
    sparse = bool(control & (1 << 2))
    base_k = _UMMA_BASE_K.get(kind)
    tensor_k = base_k * (2 if sparse else 1) if base_k else None
    if kind in {"mxf4", "mxf4nvf4"} and control & (1 << 31):
        tensor_k = 192 if sparse else 96
    elif kind == "mxf8f6f4" and control & (1 << 31):
        tensor_k = tensor_k * 2 if tensor_k is not None else None
    ws = "WS" in opcode.upper().split(".")[1:]
    operation = ("ws_sp" if ws and sparse else "ws" if ws else
                 "sp" if sparse else "normal")
    trans_a, trans_b = (control >> 15) & 1, (control >> 16) & 1
    operand_major = 2 * trans_a + trans_b
    max_shift = ((control >> 30) & 0x3
                 if kind in {"f16", "tf32", "f8f6f4"} else 0)
    return {"umma_kind": kind,
            "tensor_m": ((value >> 24) & 0x1F) << 4,
            "tensor_n": ((value >> 17) & 0x3F) << 3,
            "tensor_k": tensor_k,
            "umma_operation": operation,
            "umma_operand_major": operand_major,
            "umma_max_shift": max_shift}


def _umma_a_src(static: StaticInstruction, where: str):
    """1 smem / 2 tmem (umma_a_src_of)."""
    text = " ".join(str(t) for t in static.operands or ())
    tmem = _TMEM_OP.search(text)
    gdesc = _GDESC_OP.search(text)
    if tmem is None and gdesc is None:
        raise unsupported("umma_a_source", where)
    if tmem is None:
        return 1
    if gdesc is None:
        return 2
    return 2 if tmem.start() < gdesc.start() else 1


def _tmem_shape_repeat(opcode: str, where: str):
    """(shape_code, repeat) from opcode modifiers.

    Shape enum (loads/stores): 1 16x64b, 2 16x128b, 3 16x256b,
    4 32x32b.  The dp-bit spellings are the same four shapes
    (16dp64bit==16x64b, 16dp128bit==16x128b, 16dp256bit==16x256b,
    32dp32bit==32x32b).  A bare ``xN`` (and ``xN.MAX.*``) is shape 4
    with repeat N -- bare LDTM/STTM default to 32dp32bit with the w1
    field holding the repeat (predecessor strict_timing note; the
    reduction operator is irrelevant).  UTCCP keeps ``Nx32dpMbit``
    and packs ``(N<<16)|M`` instead -- a different encoding for
    tmem_copy, documented in formats.md.
    """
    root = opcode.split(".", 1)[0].upper()
    if root not in ("LDTM", "STTM", "UTCCP"):
        return 0, 0
    drop = {"2CTA", "STAT", "T", "S"}
    parts = [m for m in _mods(opcode) if m not in drop
             and not m.startswith("STAT") and not m.startswith("MAX")
             and not re.fullmatch(r"F\d+", m)]
    if not parts:
        if root == "UTCCP":
            # tcgen05.cp without an explicit Nx32dpMbit shape (UTCCP.T.S:
            # transposed default-shape copy): shape unknown, one copy
            return 0, 1
        # bare LDTM/STTM is 32x32b.x1 (the tmem2 microbench kernels
        # the measured x1 leaf was gated on compile to bare LDTM/STTM)
        return _TMEM_SHAPE_CODES["32x32b"], 1
    if root == "UTCCP":
        joined = "".join(p.lower() for p in parts)
        if joined == "2x64dp128bit_lw02_lw13":
            return (2 << 16) | 128, 1
        m = _UTCCP_SHAPE_RE.fullmatch(joined)
        if m is None:
            # single-block spelling <D>dp<M>bit (tcgen05.cp
            # .128x128b etc); D >= 32 vs N <= 4 in Nx32dpMbit
            m = _UTCCP_SB_SHAPE_RE.fullmatch(joined)
            if m is None:
                raise unsupported("tmem_shape", where)
        return (int(m.group(1)) << 16) | int(m.group(2)), 1
    shape_text = parts[0]
    rep_mod = next((m for m in parts if _XN.fullmatch(m)), None)
    repeat = int(rep_mod[1:]) if rep_mod else 1
    if _XN.fullmatch(shape_text):
        return _TMEM_SHAPE_CODES["32x32b"], repeat
    # SASS halves of tcgen05.ld/st .16x32bx2: 16dp32bit_t0_t15 and
    # 16dp32bit_t16_t31 (source mechtree.tmem.16x32bx2) -> code 5
    if re.fullmatch(r"16DP32BIT_T\d+_T\d+", shape_text):
        return 5, repeat
    code = _TMEM_SHAPE_CODES.get(
        _KNOWN_TMEM_SHAPES.get(shape_text, shape_text.lower()))
    if code is None:
        raise unsupported("tmem_shape", where)
    if repeat == 1:
        rep2 = next((m for m in parts[1:] if _XN.fullmatch(m)), None)
        if rep2:
            repeat = int(rep2[1:])
    return code, repeat


def _tmem_alloc_op(opcode: str):
    root = opcode.split(".", 1)[0].upper()
    if root != "UTCATOMSWS":
        return None
    joined = ".".join(_mods(opcode))
    if "FIND_AND_SET" in joined:
        return "alloc"
    if "RELINQUISH" in joined:
        return "relinquish"
    if "AND" in joined:
        return "dealloc"
    return None


def _tmem_columns_from_immediate(value):
    if not isinstance(value, int):
        return None
    if value in _TMEM_COLUMN_SET:
        return value
    scaled = value * _TMEM_GRANULE
    return scaled if scaled in _TMEM_COLUMN_SET else None


def _d_umma_issue(static, dyn, ctx, cta, wslot):
    where = _where(static)
    opcode = static.opcode.upper()
    idesc = dyn.operands.uniform.get("idesc_value") \
        if dyn.operands else None
    if idesc is None:
        raise missing("umma_descriptor", where)
    facts = _umma_descriptor_facts(opcode, int(idesc))
    kind_name = facts["umma_kind"]
    kind = _UMMA_KIND_CODES.get(kind_name)
    if kind is None:
        raise unsupported("umma_kind", where)
    return SemanticRow(
        SemanticsKind.umma_issue,
        fields={"kind": kind,
                "cta_group": 2 if "2CTA" in _mods(opcode) else 1,
                "a_source": _umma_a_src(static, where),
                "tensor_m": facts["tensor_m"],
                "tensor_n": facts["tensor_n"],
                "descriptor": int(idesc)})


def _d_umma_commit(static, dyn, ctx, cta, wslot):
    """UTCBAR = tcgen05.commit (predecessor _utcbar)."""
    where = _where(static)
    opcode = static.opcode.upper()
    addr = _object_address(dyn)
    if addr is None:
        raise missing("mbarrier_address", where)
    mask = 0
    if "MULTICAST" in _mods(opcode):
        mask = dyn.operands.uniform.get("multicast_mask") \
            if dyn.operands else None
        if mask is None:
            raise missing("umma_commit_mask", where)
        mask = int(mask)
    return SemanticRow(
        SemanticsKind.umma_commit,
        fields={"object": _mbarrier_object(dyn, ctx, cta, addr, where),
                "cta_group": 2 if "2CTA" in _mods(opcode) else 1,
                "commit_mask": mask,
                "scope": (2 if "2CTA" in _mods(opcode)
                          or "MULTICAST" in _mods(opcode) else 1)})


def _d_tmem_alloc(static, dyn, ctx, cta, wslot):
    where = _where(static)
    op = _tmem_alloc_op(static.opcode)
    if op is None:
        raise unsupported("tmem_alloc_op", where)
    columns = 0
    if op in ("alloc", "dealloc"):
        captured = dyn.operands.uniform.get("tmem_columns") \
            if dyn.operands else None
        if captured is not None:
            columns = int(captured)
        else:
            columns = _tmem_columns_from_immediate(
                dyn.operands.uniform.get("tmem_alloc_ur")
                if dyn.operands else None)
        if columns is None:
            if op == "dealloc":
                # PTX: tcgen05.dealloc frees the CTA's held columns;
                # unknown -> engine resolves (SEMA_UNKNOWN).
                columns = SEMA_UNKNOWN
            else:
                raise missing("tmem_columns", where)
    kind = {"alloc": SemanticsKind.tmem_alloc,
            "dealloc": SemanticsKind.tmem_dealloc,
            "relinquish": SemanticsKind.tmem_relinquish}[op]
    return SemanticRow(kind, fields={"columns": columns})


def _d_tmem_xfer(static, dyn, ctx, cta, wslot, kind):
    where = _where(static)
    opcode = static.opcode.upper()
    shape, repeat = _tmem_shape_repeat(opcode, where)
    flags = SEMA_FLAG_STAT if "STAT" in _mods(opcode) else 0
    return SemanticRow(kind, flags=flags,
                       fields={"shape": shape, "repeat": repeat})


def _d_clc_query(static, dyn, ctx, cta, wslot):
    """UGETNEXTWORKID (predecessor _clc): response mbarrier + scope."""
    where = _where(static)
    opcode = static.opcode.upper()
    addr = _object_address(dyn, ref_index=1)
    if addr is None:
        addr = _object_address(dyn)
    if addr is None:
        raise missing("clc_mbarrier_address", where)
    if "BROADCAST" in opcode:
        mask = (1 << len(ctx.unit.ctas)) - 1 if ctx.unit.ctas else 0
    elif "SELFCAST" in opcode:
        mask = 1
    else:
        raise missing("clc_multicast_class", where)
    return SemanticRow(
        SemanticsKind.clc_query,
        fields={"object": _mbarrier_object(dyn, ctx, cta, addr, where),
                "generation": GENERATION_UNRESOLVED,
                "destination_mask": mask,
                "response_identity": 0,
                "outcome": 0})


def decode(static: StaticInstruction, dyn: DynamicInstruction,
           ctx: SemanticsContext, operation: SemanticOperation,
           cta_index: int, warp_slot: int):
    if (dyn.active_mask & dyn.predicate_mask) == 0:
        return None
    if operation is SemanticOperation.umma_issue:
        return _d_umma_issue(static, dyn, ctx, cta_index, warp_slot)
    if operation is SemanticOperation.umma_commit:
        return _d_umma_commit(static, dyn, ctx, cta_index, warp_slot)
    if operation in (SemanticOperation.tmem_alloc,
                     SemanticOperation.tmem_dealloc,
                     SemanticOperation.tmem_relinquish):
        return _d_tmem_alloc(static, dyn, ctx, cta_index, warp_slot)
    if operation is SemanticOperation.tmem_load:
        return _d_tmem_xfer(static, dyn, ctx, cta_index, warp_slot,
                            SemanticsKind.tmem_load)
    if operation is SemanticOperation.tmem_store:
        return _d_tmem_xfer(static, dyn, ctx, cta_index, warp_slot,
                            SemanticsKind.tmem_store)
    if operation is SemanticOperation.tmem_copy:
        return _d_tmem_xfer(static, dyn, ctx, cta_index, warp_slot,
                            SemanticsKind.tmem_copy)
    if operation is SemanticOperation.tmem_wait_store:
        return SemanticRow(SemanticsKind.tmem_wait_store, fields={})
    if operation is SemanticOperation.clc_query:
        return _d_clc_query(static, dyn, ctx, cta_index, warp_slot)
    return None


def local_shared_address(address: int) -> int:
    """Strip the DSMEM cluster-rank tag (rank << 24) from a shared
    address, leaving the CTA-local offset (S32 window)."""
    return int(address) & 0xFFFFFF
