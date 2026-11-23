"""Blackwell architecture rules (docs/en/design/convert.md).

Adds sm_103a tensor/TMEM/CLC roots and the L1/L2 cache-hint bit field.
"""

from __future__ import annotations

import re
from ...convert import semantics_common

NAME = "blackwell"

_STORAGE_ROOTS = frozenset(
    "LD ST LDG STG LDL STL LDC ULDC LDCU LDS STS LDSM STSM LDGSTS "
    "UTMALDG UTMASTG UTMAREDG UTMAPF UBLKCP UBLKRED ATOM ATOMG ATOMS "
    "RED REDG REDS".split())

_SEMANTIC_ROOTS = frozenset(
    "BAR MBAR SYNCS ARRIVES DEPBAR LDGDEPBAR WARPGROUP UCGABAR "
    "UCGABAR_ARV UCGABAR_WAIT MEMBAR FENCE WARPSYNC NOP YIELD USETMAXREG "
    "UTMACMDFLUSH UTMACCTL NANOSLEEP BRA BRX BRXU BREAK BSSY BSYNC EXIT "
    "RET CALL ENDCOLLECTIVE ERRBAR CGAERRBAR CCTL HGMMA QGMMA WGMMA HMMA "
    "IMMA BMMA UTCHMMA UTCQMMA UTCOMMA UTCBAR LDTM STTM UTCATOMSWS UTCCP "
    "UGETNEXTWORKID STAS UVIRTCOUNT".split())

_COMPUTE_ROOTS = frozenset(
    "ACQBULK BMSK BREV CREDUX CS2R DADD DFMA DMUL DSETP ELECT F2F F2FP "
    "F2I FADD FADD2 FCHK FFMA FFMA2 FHFMA FHADD FLO FMNMX FMNMX3 FMUL "
    "FMUL2 FRND FSEL FSETP HADD2 HFMA2 HMNMX2 HMNMX2 HMUL2 HMNMX2 HSETP2 "
    "I2F I2FP IABS IADD IADD3 IDP IMAD ISETP LEA LEPC LOP3 MATCH MOV MUFU "
    "P2R PLOP3 POPC PREEXIT PRMT R2P R2UR REDUX RPCMOV S2R S2UR SEL SGXT "
    "SHF SHFL UFLO UIADD3 UIMAD UISETP ULEA ULOP3 UMOV UP2UR UPLOP3 UPOPC "
    "UPRMT USEL USGXT USHF VHMNMX VIADD VIADDMNMX VIMNMX VIMNMX3 VOTE "
    "VOTEU".split())

SUPPORTED_ROOTS = frozenset(_STORAGE_ROOTS | _SEMANTIC_ROOTS | _COMPUTE_ROOTS)

_BRANCH_ROOTS = frozenset("BRA BRX BRXU CALL BREAK BSSY BSYNC RET".split())
_MATRIX_ROOTS = frozenset(
    "HGMMA QGMMA WGMMA HMMA IMMA BMMA UTCHMMA UTCQMMA UTCOMMA".split())
_TMEM_ROOTS = frozenset("LDTM STTM".split())
_ATOMIC_ROOTS = frozenset("ATOM ATOMG ATOMS RED REDG REDS".split())
_ASYNC_ROOTS = frozenset("LDGSTS UTMALDG UTMASTG UTMAREDG UBLKCP UBLKRED".split())
_BARRIER_ROOTS = frozenset(
    "BAR MBAR SYNCS ARRIVES DEPBAR LDGDEPBAR MEMBAR FENCE WARPSYNC "
    "UCGABAR UCGABAR_ARV UCGABAR_WAIT UTCBAR ERRBAR CGAERRBAR".split())

FLAG_MEMORY = 1 << 0
FLAG_ATOMIC = 1 << 1
FLAG_BARRIER = 1 << 2
FLAG_BRANCH = 1 << 3
FLAG_ASYNC = 1 << 4
FLAG_TENSOR = 1 << 5
FLAG_PREDICATED = 1 << 6
FLAG_DEFER_BLOCKING = 1 << 9    # BAR.*.DEFER_BLOCKING (placement v2 rule B)
FLAG_TMEM = 1 << 7
FLAG_CLC = 1 << 8

# cache_hint bits (MACC.cache_hint; Blackwell cache-control hints)
CACHE_HINT_L1_SHIFT = 0
CACHE_HINT_L1_MASK = 0xF
CACHE_HINT_L2_SHIFT = 4
CACHE_HINT_L2_MASK = 0xF0
_CACHE_OPERATOR_BITS = {
    "EF": 1, "EL": 2, "LU": 3, "EU": 4, "CV": 5, "WB": 6, "WT": 7,
    "IV": 8, "CG": 9, "CS": 10, "CA": 11, "NA": 12,
}


def operation_class(static, dynamic=None) -> str:
    """Effective class (docs/en/design/convert.md)."""
    opcode = (static.opcode if hasattr(static, "opcode")
              else str(static))
    tokens = opcode.upper().split(".")
    root = tokens[0]
    if root == "CS2R" and hasattr(static, "operands"):
        ops = " ".join(str(t).upper()
                       for t in (static.operands or ()))
        if "GLOBALTIMER" in ops or "GLOBALTIM" in ops:
            return "CS2R.GLOBALTIMER"
        if "CLOCK" in ops:
            return "CS2R.CLOCK"
        if "SRZ" in ops or "RZ" in ops:
            return "CS2R.ZERO"
        return "CS2R.ZERO"
    if root in ("LD", "ST") and hasattr(static, "opcode"):
        if memory_space(static) == "shared":
            return "LDS" if root == "LD" else "STS"
    if root == "HADD2" and tokens[1:] == ["F32"]:
        return "HADD2.F32"
    if root == "DEPBAR" and "LE" in tokens:
        return "DEPBAR.FIFO"
    if root == "IMAD" and "HI" in tokens:
        return "IMAD.HI"
    if root == "MUFU" and tokens[1:] and tokens[1] in {
            "EX2", "RSQ", "LG2", "TANH"}:
        return f"MUFU.{tokens[1]}"
    if root == "MEMBAR":
        sem = next((t for t in ("ALL", "SC") if t in tokens), None)
        scope = next((t for t in ("CTA", "GPU", "SYS") if t in tokens), None)
        if sem and scope:
            return f"MEMBAR.{sem}.{scope}"
    if root == "SYNCS" and "PHASECHK" in tokens:
        return "SYNCS.PHASECHK"
    if (root == "SYNCS" and "ARRIVE" in tokens
            and hasattr(static, "operands")):
        ops = static.operands or ()
        first = str(ops[0]).upper() if ops else ""
        if re.fullmatch(r"R\d+", first):
            return ("SYNCS.ARRIVE.STATE.COUNT" if "ART0" in tokens
                    else "SYNCS.ARRIVE.STATE")
    if root == "CCTL" and "IVALL" in tokens:
        return "CCTL.IVALL"
    if root == "SYNCS" and tokens[1:] == ["CCTL", "IVALL"]:
        return "CCTL.IVALL"
    if root == "FENCE" and {"VIEW", "ASYNC"} <= set(tokens):
        view = next((t for t in tokens[1:] if t in {"S", "G"}), None)
        if view:
            return f"FENCE.VIEW.ASYNC.{view}"
    if root in {"HMMA", "IMMA", "BMMA"} and "SP" in tokens:
        return root + ".SPARSE"
    if root == "SHFL":
        mode = next(
            (t for t in tokens[1:] if t in {"IDX", "UP", "DOWN", "BFLY"}),
            None)
        if mode:
            return "SHFL." + mode
    return root


def opcode_root(opcode: str) -> str:
    return opcode.upper().split(".", 1)[0]


def is_compute_whitelisted(cls: str) -> bool:
    return opcode_root(cls) in SUPPORTED_ROOTS


def memory_space(static, dynamic=None) -> str | None:
    if static.memory_space:
        return static.memory_space
    root = opcode_root(static.opcode)
    if root in {"LDG", "STG", "ATOMG", "REDG", "LDGSTS", "UTMALDG",
                "UTMASTG", "UTMAREDG", "UBLKCP", "UBLKRED"}:
        return "global"
    if root in {"LDS", "STS", "LDSM", "STSM", "ATOMS", "REDS", "STAS"}:
        return "shared"
    if root in {"LDL", "STL"}:
        return "local"
    if root in {"LDC", "ULDC", "LDCU"}:
        return "constant"
    if root in {"LD", "ST"}:
        return "generic"
    return None


def cache_hint(static) -> int:
    """Encode .L1::/.L2:: hint tokens into the MACC.cache_hint bit field."""
    opcode = static.opcode.upper()
    hint = 0
    for token in opcode.split(".")[1:]:
        m = re.fullmatch(r"L([12])::([A-Z]+)", token)
        if m:
            level, op = int(m.group(1)), m.group(2)
            bits = _CACHE_OPERATOR_BITS.get(op)
            if bits is None:
                continue
            if level == 1:
                hint = (hint & ~CACHE_HINT_L1_MASK) | (
                    bits << CACHE_HINT_L1_SHIFT)
            else:
                hint = (hint & ~CACHE_HINT_L2_MASK) | (
                    bits << CACHE_HINT_L2_SHIFT)
    return hint


def instruction_flags(static) -> int:
    root = opcode_root(static.opcode)
    flags = 0
    if memory_space(static) is not None:
        flags |= FLAG_MEMORY
    if root in _ATOMIC_ROOTS:
        flags |= FLAG_ATOMIC
    if root in _BARRIER_ROOTS:
        flags |= FLAG_BARRIER
    if root in _BRANCH_ROOTS:
        flags |= FLAG_BRANCH
    if root in _ASYNC_ROOTS:
        flags |= FLAG_ASYNC
    if root in _MATRIX_ROOTS:
        flags |= FLAG_TENSOR
    if root in _TMEM_ROOTS:
        flags |= FLAG_TMEM
    if root == "UGETNEXTWORKID":
        flags |= FLAG_CLC
    if static.predicate is not None:
        flags |= FLAG_PREDICATED
    if root == "BAR" and "DEFER_BLOCKING" in static.opcode.upper().split(".")[1:]:
        flags |= FLAG_DEFER_BLOCKING
    return flags


# exact SASS spellings -> SemanticOperation (blackwell-prototype
# convert/semantics.py dispatch + corpus-checked bundle opcodes).
_SEMANTIC_OPERATIONS = {
    "SYNCS.EXCH": semantics_common.SemanticOperation.mbarrier_init,
    "SYNCS.ARRIVE": semantics_common.SemanticOperation.mbarrier_arrive,
    "SYNCS.PHASECHK": semantics_common.SemanticOperation.mbarrier_wait,
    "SYNCS.CCTL.IV": semantics_common.SemanticOperation.mbarrier_inval,
    "MBAR.TRY_WAIT": semantics_common.SemanticOperation.mbarrier_wait,
    "MBAR.TEST_WAIT": semantics_common.SemanticOperation.mbarrier_wait,
    "BAR.ARV": semantics_common.SemanticOperation.cta_barrier_arrive,
    "BAR.SYNC": semantics_common.SemanticOperation.cta_barrier_wait,
    "BAR.RED": semantics_common.SemanticOperation.cta_barrier_wait,
    "UCGABAR_ARV": semantics_common.SemanticOperation.cluster_barrier_arrive,
    "UCGABAR_WAIT": semantics_common.SemanticOperation.cluster_barrier_wait,
    "UTMALDG": semantics_common.SemanticOperation.tma_load,
    "UTMASTG": semantics_common.SemanticOperation.bulk_group_issue,
    "UTMAREDG": semantics_common.SemanticOperation.bulk_group_issue,
    # .G.S = destination global, source shared: a bulk store (bulk group,
    # no mbarrier); .S.G / .S.S complete on an mbarrier (bulk_copy)
    "UBLKCP.G.S": semantics_common.SemanticOperation.bulk_group_issue,
    "UBLKRED.G.S": semantics_common.SemanticOperation.bulk_group_issue,
    "UTMACMDFLUSH": semantics_common.SemanticOperation.bulk_group_commit,
    "UTMACCTL.COMMIT": semantics_common.SemanticOperation.bulk_group_commit,
    "UTMACCTL.IV":
        semantics_common.SemanticOperation.descriptor_visibility_acquire,
    "MEMBAR.ALL.GPU":
        semantics_common.SemanticOperation.descriptor_visibility_publish,
    "ERRBAR":
        semantics_common.SemanticOperation.descriptor_visibility_error_check,
    "CGAERRBAR":
        semantics_common.SemanticOperation.descriptor_visibility_error_check,
    "STG": semantics_common.SemanticOperation.descriptor_visibility_store,
    "UBLKCP": semantics_common.SemanticOperation.bulk_copy,
    "UBLKRED": semantics_common.SemanticOperation.bulk_reduce,
    "LDGSTS": semantics_common.SemanticOperation.async_copy_issue,
    "LDGDEPBAR": semantics_common.SemanticOperation.async_copy_commit,
    "ARRIVES": semantics_common.SemanticOperation.async_copy_commit,
    "DEPBAR": semantics_common.SemanticOperation.depbar_wait,
    "STAS": semantics_common.SemanticOperation.shared_store_async,
    "FENCE.VIEW.ASYNC.S":
        semantics_common.SemanticOperation.descriptor_visibility_proxy_fence,
    "FENCE.VIEW.ASYNC.T":
        semantics_common.SemanticOperation.tmem_wait_store,
    "NANOSLEEP": semantics_common.SemanticOperation.nanosleep,
    "PREEXIT": semantics_common.SemanticOperation.dependents_launch,
    # 2b-3 arch semantics
    "UTCHMMA": semantics_common.SemanticOperation.umma_issue,
    "UTCQMMA": semantics_common.SemanticOperation.umma_issue,
    "UTCOMMA": semantics_common.SemanticOperation.umma_issue,
    "UTCBAR": semantics_common.SemanticOperation.umma_commit,
    "UTCATOMSWS": semantics_common.SemanticOperation.tmem_alloc,
    "LDTM": semantics_common.SemanticOperation.tmem_load,
    "STTM": semantics_common.SemanticOperation.tmem_store,
    "UTCCP": semantics_common.SemanticOperation.tmem_copy,
    "UGETNEXTWORKID": semantics_common.SemanticOperation.clc_query,
}

_SUPPORTED_ROOTS = frozenset(k.split(".", 1)[0]
                             for k in _SEMANTIC_OPERATIONS)


def semantic_operation(static):
    """Exact SASS opcode-root -> SemanticOperation."""
    opcode = static.opcode.upper()
    # global-memory fences carry no engine sync state (like MEMBAR);
    # SYNCS.CCTL.IVALL is the barrier-init fence, also no state
    if opcode.startswith("FENCE.VIEW.ASYNC.G") or \
            opcode.startswith("SYNCS.CCTL.IVALL"):
        return None
    for width in (4, 3, 2):
        head = ".".join(opcode.split(".")[:width])
        op = _SEMANTIC_OPERATIONS.get(head)
        if op is not None:
            return op
    root = opcode.split(".", 1)[0]
    op = _SEMANTIC_OPERATIONS.get(root)
    if op is None:
        # UTMACCTL variants other than COMMIT/IV carry no semantics in
        # the predecessor (semantics.py:381,466).
        if root in ("UTMACCTL", "MEMBAR"):
            return None
        if root in _SUPPORTED_ROOTS:
            raise semantics_common.unsupported(
                "semantic_modifier", f"{static.opcode} @{static.pc:#x}")
        return None
    return op


def semantics(static, dynamic, context, cta_index=0, warp_slot=0):
    # STS / STS.64 onto a barrier word named by explicit barrier
    # instructions in the unit is an mbarrier init (or a capture gap).
    promoted = semantics_common.promote_store_to_mbarrier_init(
        static, dynamic, context, cta_index, warp_slot)
    if promoted is not None:
        return promoted
    operation = semantic_operation(static)
    if operation is None:
        return None
    if operation in semantics_common.ARCH_OPERATIONS:
        from . import semantics as arch_semantics
        return arch_semantics.decode(static, dynamic, context, operation,
                                     cta_index, warp_slot)
    return semantics_common.decode(static, dynamic, context, operation,
                                   cta_index, warp_slot)


_MATRIX_RE = re.compile(
    r"(?:LDSM|STSM)\.(\d+)\.M(T)?(\d)(\d)(?:\.(\d+))?(?:\..*)?")


def opcode_facts(opcode: str) -> dict:
    tokens = opcode.upper().split(".")
    facts = {"operation_class": operation_class(opcode)}
    root = opcode_root(opcode)
    if root in {"LDSM", "STSM"}:
        m = _MATRIX_RE.match(opcode.upper())
        if m:
            bits, transposed, rows, cols, count = m.groups()
            facts.update(
                width_bytes=int(bits) // 8, tensor_m=int(rows),
                tensor_n=int(cols), matrix_count=int(count or 1),
                matrix_transposed=int(transposed is not None))
    if root in {"UTMALDG", "UTMASTG", "UTMAREDG", "UTMAPF"}:
        dim = re.search(r"\.(\d+)D(?:\.|$)", opcode.upper())
        if dim:
            facts["tma_instruction_rank"] = int(dim.group(1))
    types = [t for t in tokens[1:]
             if re.fullmatch(r"(?:[BUSF]\d+|BF16|TF32|E4M3|E5M2)", t)]
    facts["datatype"] = ".".join(types) if types else "unspecified"
    shape = re.search(r"(?:^|\.)(\d+)X(\d+)X(\d+)(?:\.|$)", opcode.upper())
    if shape:
        facts.update(zip(("tensor_m", "tensor_n", "tensor_k"),
                         map(int, shape.groups())))
    return facts


def operation_classes(static, dynamic=None):
    """Class set (spec §3.1): effective class, opcode root, arch
    pseudo-class (predecessor sass_isa.instruction_class_of)."""
    cls = operation_class(static, dynamic)
    opcode = (static.opcode if hasattr(static, "opcode")
              else str(static)).upper()
    root = opcode.split(".", 1)[0]
    out = [cls] if cls == root else [cls, root]
    mods = opcode.split(".")[1:]
    if root == "LDTM" and any(m.startswith("STAT") for m in mods):
        out.append("tmem_load_stat")
    elif cls == "FENCE.VIEW.ASYNC.T":
        out.append("tmem_wait_store")
    return out


# re-exported for segment_builder remote-object resolution
from .semantics import local_shared_address  # noqa: E402
