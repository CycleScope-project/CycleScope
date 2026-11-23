"""Hopper architecture rules (docs/en/design/convert.md).

operation_class spellings are the mechanism-tree axis values — the former
opcode_root routing, minus payload-dependent extras (which 2b-4 handles).
"""

from __future__ import annotations

import re
from ...convert import semantics_common

NAME = "hopper"

_STORAGE_ROOTS = frozenset(
    "LD ST LDG STG LDL STL LDC ULDC LDS STS LDSM STSM LDGSTS UTMALDG "
    "UTMASTG UTMAREDG UTMAPF UBLKCP UBLKRED ATOM ATOMG ATOMS RED REDG "
    "REDS STAS".split())

_SEMANTIC_ROOTS = frozenset(
    "BAR MBAR SYNCS ARRIVES DEPBAR LDGDEPBAR WARPGROUP UCGABAR "
    "UCGABAR_ARV UCGABAR_WAIT MEMBAR FENCE WARPSYNC NOP YIELD USETMAXREG "
    "UTMACMDFLUSH USETSHMSZ UTMACCTL NANOSLEEP BRA BRX BRXU CALL BREAK "
    "BSSY BSYNC EXIT RET ENDCOLLECTIVE ERRBAR CGAERRBAR CCTL HGMMA QGMMA "
    "WGMMA IGMMA HMMA IMMA BMMA".split())

_COMPUTE_ROOTS = frozenset(
    "ACQBULK BREV CS2R DADD DMUL DSETP ELECT F2F F2FP F2I FADD FCHK FFMA "
    "FMNMX FMUL FSEL FSETP HADD2 HFMA2 HMUL2 I2F I2FP IABS IADD IADD3 IDP "
    "IMAD ISETP LEA LOP3 MATCH MOV MUFU P2R PLOP3 POPC PREEXIT PRMT R2P "
    "R2UR REDUX S2R S2UR SEL SGXT SHF SHFL FLO UFLO UIADD3 UIMAD UISETP "
    "ULEA ULOP3 UMOV UP2UR UPLOP3 UPOPC UPRMT USEL USGXT USHF VIADD "
    "VIADDMNMX VIMNMX VIMNMX3 VOTE VOTEU".split())

SUPPORTED_ROOTS = frozenset(_STORAGE_ROOTS | _SEMANTIC_ROOTS | _COMPUTE_ROOTS)

_BRANCH_ROOTS = frozenset("BRA BRX BRXU CALL BREAK BSSY BSYNC RET".split())
_MATRIX_ROOTS = frozenset("HGMMA QGMMA WGMMA IGMMA HMMA IMMA BMMA".split())
_ATOMIC_ROOTS = frozenset("ATOM ATOMG ATOMS RED REDG REDS".split())
_ASYNC_ROOTS = frozenset("LDGSTS UTMALDG UTMASTG UTMAREDG UBLKCP UBLKRED".split())

# TMPL.flags bits (docs/en/formats.md §2)
FLAG_MEMORY = 1 << 0
FLAG_ATOMIC = 1 << 1
FLAG_BARRIER = 1 << 2
FLAG_BRANCH = 1 << 3
FLAG_ASYNC = 1 << 4
FLAG_TENSOR = 1 << 5
FLAG_PREDICATED = 1 << 6
FLAG_DEFER_BLOCKING = 1 << 9    # BAR.*.DEFER_BLOCKING (placement v2 rule B)


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
    if root == "MUFU" and tokens[1:] and tokens[1] in {
            "EX2", "RSQ", "LG2", "TANH"}:
        return f"MUFU.{tokens[1]}"
    if root == "DEPBAR" and "LE" in tokens:
        return "DEPBAR.FIFO"
    if root == "IMAD" and "HI" in tokens:
        return "IMAD.HI"
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
    if root == "FENCE" and {"VIEW", "ASYNC"} <= set(tokens):
        view = next((t for t in tokens[1:] if t in {"S", "G"}), None)
        if view:
            return f"FENCE.VIEW.ASYNC.{view}"
    if root == "USETSHMSZ" and "FLUSH" in tokens:
        return "USETSHMSZ.FLUSH"
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
    if root in {"LDS", "STS", "LDSM", "STSM", "ATOMS", "REDS"}:
        return "shared"
    if root in {"LDL", "STL"}:
        return "local"
    if root in {"LDC", "ULDC"}:
        return "constant"
    if root in {"LD", "ST"}:
        return "generic"
    return None


def instruction_flags(static) -> int:
    root = opcode_root(static.opcode)
    flags = 0
    if memory_space(static) is not None:
        flags |= FLAG_MEMORY
    if root in _ATOMIC_ROOTS:
        flags |= FLAG_ATOMIC
    if root in _SEMANTIC_ROOTS and root in {
            "BAR", "MBAR", "SYNCS", "ARRIVES", "DEPBAR", "LDGDEPBAR",
            "MEMBAR", "FENCE", "WARPSYNC", "UCGABAR", "UCGABAR_ARV",
            "UCGABAR_WAIT", "ERRBAR", "CGAERRBAR"}:
        flags |= FLAG_BARRIER
    if root in _BRANCH_ROOTS:
        flags |= FLAG_BRANCH
    if root in _ASYNC_ROOTS:
        flags |= FLAG_ASYNC
    if root in _MATRIX_ROOTS:
        flags |= FLAG_TENSOR
    if static.predicate is not None:
        flags |= FLAG_PREDICATED
    if root == "BAR" and "DEFER_BLOCKING" in static.opcode.upper().split(".")[1:]:
        flags |= FLAG_DEFER_BLOCKING
    return flags


def cache_hint(static) -> int:
    return 0  # Hopper has no hint bit field


# exact SASS spellings -> SemanticOperation (hopper-prototype convert.py /
# docs/en/formats.md table "instruction payload 顶层键"; corpus-checked
# against the 2b-1 bundles).
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
    "ARRIVES.LDGSTSBAR": semantics_common.SemanticOperation.async_copy_commit,
    "DEPBAR": semantics_common.SemanticOperation.depbar_wait,
    "STAS": semantics_common.SemanticOperation.shared_store_async,
    "FENCE.VIEW.ASYNC.S":
        semantics_common.SemanticOperation.descriptor_visibility_proxy_fence,
    "NANOSLEEP": semantics_common.SemanticOperation.nanosleep,
    "PREEXIT": semantics_common.SemanticOperation.dependents_launch,
    # 2b-3 arch semantics
    "WARPGROUP.ARRIVE":
        semantics_common.SemanticOperation.warpgroup_arrive,
    "WARPGROUP.DEPBAR.LE":
        semantics_common.SemanticOperation.warpgroup_wait,
    "WARPGROUP.WAIT":
        semantics_common.SemanticOperation.warpgroup_wait,
    "HGMMA": semantics_common.SemanticOperation.wgmma,
    "WGMMA": semantics_common.SemanticOperation.wgmma,
    "QGMMA": semantics_common.SemanticOperation.wgmma,
    "IGMMA": semantics_common.SemanticOperation.wgmma,
}

_SUPPORTED_ROOTS = frozenset(k.split(".", 1)[0]
                             for k in _SEMANTIC_OPERATIONS)


_SEMOP_CACHE: dict = {}


def semantic_operation(static):
    """Exact SASS opcode-root -> SemanticOperation (modifier-aware)."""
    opcode = static.opcode.upper()
    # global-memory fences carry no engine sync state (like MEMBAR);
    # SYNCS.CCTL.IVALL is the barrier-init fence, also no state
    if opcode.startswith("FENCE.VIEW.ASYNC.G") or \
            opcode.startswith("SYNCS.CCTL.IVALL"):
        _SEMOP_CACHE[opcode] = None
        return None
    hit = _SEMOP_CACHE.get(opcode, 0)
    if hit != 0:
        return hit
    # multi-word roots first (UBLKCP.G.S vs UBLKCP)
    for width in (4, 3, 2):
        head = ".".join(opcode.split(".")[:width])
        op = _SEMANTIC_OPERATIONS.get(head)
        if op is not None:
            _SEMOP_CACHE[opcode] = op
            return op
    root = opcode.split(".", 1)[0]
    op = _SEMANTIC_OPERATIONS.get(root)
    if op is None:
        # UTMACCTL variants other than COMMIT/IV carry no semantics
        if root in ("UTMACCTL", "MEMBAR"):
            _SEMOP_CACHE[opcode] = None
            return None
        if root in _SUPPORTED_ROOTS:
            raise semantics_common.unsupported(
                "semantic_modifier", f"{static.opcode} @{static.pc:#x}")
        _SEMOP_CACHE[opcode] = None
        return None
    _SEMOP_CACHE[opcode] = op
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
    """Static opcode facts for the fact axis functions (2b-4 expands)."""
    tokens = opcode.upper().split(".")
    facts = {"operation_class": operation_class(opcode)}
    if opcode_root(opcode) in {"LDSM", "STSM"}:
        m = _MATRIX_RE.match(opcode.upper())
        if m:
            bits, transposed, rows, cols, count = m.groups()
            facts.update(
                width_bytes=int(bits) // 8, tensor_m=int(rows),
                tensor_n=int(cols), matrix_count=int(count or 1),
                matrix_transposed=int(transposed is not None))
    types = [t for t in tokens[1:]
             if re.fullmatch(r"(?:[BUSF]\d+|BF16|TF32|E4M3|E5M2)", t)]
    facts["datatype"] = ".".join(types) if types else "unspecified"
    return facts


def operation_classes(static, dynamic=None):
    """Class set (spec §3.1): effective class, opcode root."""
    cls = operation_class(static, dynamic)
    root = opcode_root(static.opcode if hasattr(static, "opcode")
                       else str(static))
    return [cls] if cls == root else [cls, root]


# re-exported for segment_builder remote-object resolution
from .semantics import local_shared_address  # noqa: E402
