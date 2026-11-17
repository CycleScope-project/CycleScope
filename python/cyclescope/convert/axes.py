"""Capture-axis functions (docs/en/design/convert.md).

Each function ``fn(static, dynamic, context) -> value | None`` is pure:
values come only from this instruction's static/dynamic facts, the
per-warp flow context, or launch facts.  ``None`` = fact absent.
``context`` is an ``AxisContext`` (unit, launch, warp, flow, rules).
Enum axes return the enum *value* (string); the traversal encodes to
index.  Int axes return ints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

NOT_APPLICABLE = object()      # axis undefined for this instruction


@dataclass(slots=True)
class FlowContext:
    """Per-warp stream state (spec §3.4).

    ``readers`` maps instruction index -> (reader_static, role) from the
    per-warp pre-scan (first later executing reader of a destination
    register, before overwrite).  ``reader`` is the current instance's
    entry; ``cpasync_count`` is the LDGSTS count since the last ARRIVES or LDGDEPBAR (group commit).
    """
    readers: dict = field(default_factory=dict)   # inst idx -> (static, role|None)
    reader: object = None        # current instance's (static, role)|None
    cpasync_count: int = 0       # executing LDGSTS since last ARRIVES or LDGDEPBAR (group commit)


@dataclass(slots=True)
class AxisContext:
    unit: object = None              # ClusterUnit
    launch: object = None            # LaunchFacts
    warp: object = None              # WarpStream
    flow: FlowContext = None         # per-warp flow context
    rules: object = None             # arch rules module
    next_static_pc: object = None    # static pc of this warp's next
                                     # dynamic instance (branch_taken)


# ---------------------------------------------------------------- helpers

_TENSOR_RE = re.compile(r"(?:^|\.)(\d+)X(\d+)X(\d+)(?:\.|$)")
# warp-level mma.sync shapes are packed as digits: HMMA.16816 = m16n8k16,
# HMMA.1688 = m16n8k8, HMMA.884 = m8n8k4, IMMA.16832 = m16n8k32,
# BMMA.168256 = m16n8k256 (M in {8,16}, N always 8, K the remainder)
_PACKED_MMA_RE = re.compile(r"^(?:HMMA|IMMA|DMMA|BMMA)\.(\d{3,6})(?:\.|$)")


def _packed_mma_shape(opcode: str):
    m = _PACKED_MMA_RE.match(opcode.upper())
    if m is None:
        return None
    digits = m.group(1)
    rest = digits[2:] if digits.startswith("16") else digits[1:]
    if not rest.startswith("8") or len(rest) < 2:
        return None
    return (16 if digits.startswith("16") else 8, 8, int(rest[1:]))


def _tensor_dims(opcode: str):
    m = _TENSOR_RE.search(opcode.upper())
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return _packed_mma_shape(opcode)
_DTYPE_RE = re.compile(r"(?:[BUSF]\d+|BF16|TF32|E4M3|E5M2)")
_XN_RE = re.compile(r"^x(\d+)$", re.IGNORECASE)
_REG_OP = re.compile(r"^(R\d+|UR\d+|RZ|URZ)$")


def _tokens(static):
    return [str(t).strip() for t in (static.operands or ())
            if str(t).strip()]


def _lanes(dynamic):
    return bin(dynamic.active_mask).count("1") if dynamic is not None \
        else 32


def _uniform(dynamic, key):
    ops = dynamic.operands if dynamic is not None else None
    if ops is None:
        return None
    v = ops.uniform.get(key)
    return int(v) if v is not None else None


def _mem(dynamic):
    return dynamic.memory if dynamic is not None else ()


def _hier(dynamic):
    """References the memory hierarchy sees: for a mixed-space
    instruction (LDGSTS: global source + shared destination) only the
    non-shared side; otherwise all references."""
    m = _mem(dynamic)
    h = [r for r in m if r.space not in ("shared", "shared_cluster")]
    return h if h and len(h) != len(m) else m


def _dtype_segments(opcode):
    return [t.lower() for t in opcode.upper().split(".")[1:]
            if _DTYPE_RE.fullmatch(t)]


def _opcode_facts(static, dynamic, context):
    rules = context.rules if context else None
    if rules is not None and hasattr(rules, "opcode_facts"):
        return rules.opcode_facts(static.opcode)
    return {}


# ---------------------------------------------------------------- static

def _f_operation_class(s, d, c):
    rules = c.rules if c else None
    if rules is not None:
        return rules.operation_class(s, d)
    return s.opcode.upper().split(".", 1)[0]


def _f_datatype(s, d, c):
    types = _dtype_segments(s.opcode)
    return ".".join(types).upper() if types else "unspecified"


_GMMA_ROOTS = frozenset({"HGMMA", "WGMMA", "QGMMA", "IGMMA"})


def _gmma_dtypes(s, c):
    """(input, accumulator) dtype names via the arch decoder."""
    root = s.opcode.upper().split(".", 1)[0]
    if root not in _GMMA_ROOTS:
        return None
    rules = getattr(c, "rules", None) if c else None
    if getattr(rules, "NAME", "hopper") == "blackwell":
        return None
    from ..arch.hopper import semantics as hsem
    inp, acc = hsem._dtype_fields(s.opcode.upper())
    return hsem.DTYPE_NAMES.get(inp), hsem.DTYPE_NAMES.get(acc)


def _f_input_dtype(s, d, c):
    pair = _gmma_dtypes(s, c)
    if pair is None:
        return NOT_APPLICABLE
    return pair[0]


def _f_accumulator_dtype(s, d, c):
    pair = _gmma_dtypes(s, c)
    if pair is None:
        return NOT_APPLICABLE
    return pair[1]


# Source-form classification (predecessor strict_timing.opcode_facts):
# operand tokens after the destination operands classify as
# gpr/uniform/immediate/zero/constant.  Storage and branch roots carry
# no such facts -> NOT_APPLICABLE; an unclassifiable token -> None.
_SOURCE_FORM_STORAGE = frozenset(
    "LD ST LDG STG LDL STL LDC ULDC LDCU LDS STS LDSM STSM LDGSTS "
    "UTMALDG UTMASTG UTMAREDG UTMAPF UBLKCP UBLKRED ATOM ATOMG ATOMS "
    "RED REDG REDS STAS LDTM STTM UTCATOMSWS UTCCP UVIRTCOUNT".split())
_SOURCE_FORM_BRANCH = frozenset(
    "BRA BRX BRXU CALL BSSY BSYNC BREAK EXIT RET".split())
_TWO_DEST_ROOTS = frozenset({"FSETP", "DSETP", "ISETP", "UISETP"})
_SOURCE_CLASSES = ("gpr", "uniform", "immediate", "zero", "constant")


def _source_class(token: str):
    """SASS operand token -> source class or None."""
    # operand modifiers: sign (-/+), absolute (|x|), bitwise not (~)
    token = re.sub(r"\.REUSE$", "",
                   token.strip().upper().lstrip("-+|~").rstrip("|"))
    if re.fullmatch(r"(?:RZ|URZ|SRZ)(?:\.[A-Z0-9_]+)*", token):
        return "zero"
    if re.fullmatch(r"R\d+(?:\.[A-Z0-9_]+)*", token):
        return "gpr"
    if re.fullmatch(r"UR\d+(?:\.[A-Z0-9_]+)*", token):
        return "uniform"
    if re.fullmatch(r"C\[.*\](?:\[.*\])?", token):
        return "constant"
    if re.fullmatch(
            r"(?:0X[0-9A-F]+|(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)",
            token):
        return "immediate"
    if re.fullmatch(r"!?U?P(?:T|[0-6])", token):
        return "predicate"
    return None


def _source_form(s):
    """-> (counts dict, mode) | NOT_APPLICABLE | None (unclassifiable)."""
    root = s.opcode.upper().split(".", 1)[0]
    if root in _SOURCE_FORM_STORAGE or root in _SOURCE_FORM_BRANCH:
        return NOT_APPLICABLE
    tokens = _tokens(s)
    dest = 2 if root in _TWO_DEST_ROOTS else 1
    counts = {name: 0 for name in _SOURCE_CLASSES}
    for raw in tokens[dest:]:
        cls = _source_class(raw)
        if cls is None:
            return None
        if cls == "predicate":
            continue
        counts[cls] += 1
    if not any(counts.values()):
        return None
    present = [name for name, n in counts.items() if n]
    return counts, present[0] if len(present) == 1 else "mixed"


def _source_form_part(s, part):
    form = _source_form(s)
    if form is NOT_APPLICABLE or form is None:
        return form
    counts, mode = form
    return mode if part == "mode" else counts[part]


def _f_source_mode(s, d, c):
    root = s.opcode.upper().split(".", 1)[0]
    if root in _GMMA_ROOTS:
        rules = getattr(c, "rules", None) if c else None
        if getattr(rules, "NAME", "hopper") == "blackwell":
            return NOT_APPLICABLE
        from ..arch.hopper import semantics as hsem
        where = f"{s.opcode} @{hex(s.pc or 0)}"
        return {1: "ss", 2: "rs"}[hsem._source_mode(s, where)]
    return _source_form_part(s, "mode")


def _f_tensor_sparse(s, d, c):
    # HGMMA/QGMMA/IGMMA .SP token -> 1 sparse, else 0 (dense)
    return 1 if ".SP" in s.opcode.upper() else 0


def _f_tensor_m(s, d, c):
    dims = _tensor_dims(s.opcode)
    if dims:
        return dims[0]
    v = _opcode_facts(s, d, c).get("tensor_m")
    if v is not None:
        return v
    v = _uniform(d, "tensor_m")
    if v is not None:
        return v
    if s.opcode.split(".", 1)[0].upper() in (
            "UTCHMMA", "UTCQMMA", "UTCOMMA"):
        idesc = _uniform(d, "idesc_value")
        if idesc is None:
            return None
        from ..arch.blackwell import semantics as bsem
        return bsem._umma_descriptor_facts(
            s.opcode.upper(), int(idesc)).get("tensor_m")
    return None


def _f_tensor_n(s, d, c):
    dims = _tensor_dims(s.opcode)
    if dims:
        return dims[1]
    v = _opcode_facts(s, d, c).get("tensor_n")
    if v is not None:
        return v
    v = _uniform(d, "tensor_n")
    if v is not None:
        return v
    if s.opcode.split(".", 1)[0].upper() in (
            "UTCHMMA", "UTCQMMA", "UTCOMMA"):
        idesc = _uniform(d, "idesc_value")
        if idesc is None:
            return None
        from ..arch.blackwell import semantics as bsem
        return bsem._umma_descriptor_facts(
            s.opcode.upper(), int(idesc)).get("tensor_n")
    return None


_UMMA_K = {"f16": 16, "bf16": 16, "tf32": 8, "f8f6f4": 32,
         "mxf8f6f4": 32, "mxf4": 64, "mxf4nvf4": 64}


def _f_tensor_k(s, d, c):
    dims = _tensor_dims(s.opcode)
    if dims:
        return dims[2]
    f = _umma_idesc_facts(s, d)
    if f is None:
        return None
    if f is not NOT_APPLICABLE:
        return f["tensor_k"]
    kind = _f_umma_kind(s, d, c)
    if kind is not None:
        return _UMMA_K.get(kind)
    return None


def _f_tensor_rank(s, d, c):
    m = re.search(r"\.(\d)D", s.opcode.upper())
    if m:
        return int(m.group(1))
    ops = d.operands if d else None
    tm_id = getattr(ops, "tensor_map", None) if ops else None
    if tm_id is None:
        tm_id = _uniform(d, "tensor_map")
    tm = (c.unit.tensor_maps.get(tm_id)
          if c and c.unit and tm_id is not None else None)
    return tm.rank if tm else None


def _f_tma_instruction_rank(s, d, c):
    v = _opcode_facts(s, d, c).get("tma_instruction_rank")
    if v is not None:
        return v
    return _f_tensor_rank(s, d, c)


def _f_tma_box_planes(s, d, c):
    rank = _f_tensor_rank(s, d, c)
    return rank


def _f_tma_form(s, d, c):
    op = s.opcode.upper()
    if "IM2COL" in op:
        return "im2col"
    for form in ("GATHER4", "SCATTER4"):
        if form in op:
            return form.lower()
    if "TILE" in op or any(f"{r}D" in op for r in "12345"):
        return "tile"
    return None


def _f_matrix_count(s, d, c):
    v = _opcode_facts(s, d, c).get("matrix_count")
    if v is not None:
        return v
    return 1 if "MMA" in s.opcode.upper() else 0


def _f_matrix_transposed(s, d, c):
    v = _opcode_facts(s, d, c).get("matrix_transposed")
    if v is not None:
        return v
    op = s.opcode.upper()
    return int("TN" in op or ".T" in op.split("MMA", 1)[-1]
               or "TNSP" in op)


def _f_gpr_source_count(s, d, c):
    return _source_form_part(s, "gpr")


def _f_uniform_source_count(s, d, c):
    return _source_form_part(s, "uniform")


def _f_immediate_source_count(s, d, c):
    return _source_form_part(s, "immediate")


def _f_zero_source_count(s, d, c):
    return _source_form_part(s, "zero")


def _f_constant_source_count(s, d, c):
    return _source_form_part(s, "constant")


def _f_execution_unit(s, d, c):
    facts = _opcode_facts(s, d, c)
    return facts.get("execution_unit")


def _f_branch_taken(s, d, c):
    """1 when this warp's next dynamic instance's static pc is not
    pc + 16 (16 B SASS); the last instance of a warp -> 0."""
    npc = getattr(c, "next_static_pc", None)
    if s.pc is None or npc is None:
        return 0
    return 0 if npc == s.pc + 16 else 1


def _f_branch_direction(s, d, c):
    if s.pc is None:
        return None
    toks = _tokens(s)
    for t in reversed(toks):
        if re.fullmatch(r"0x[0-9a-fA-F]+", t):
            return "backward" if int(t, 16) < s.pc else "forward"
    return None


def _f_umma_kind(s, d, c):
    """kind from opcode+idesc (arch decoder rules; umma_descriptor_facts)."""
    if s.opcode.split(".", 1)[0].upper() not in (
            "UTCHMMA", "UTCQMMA", "UTCOMMA"):
        return None
    try:
        from ..arch.blackwell import semantics as bsem
    except ImportError:
        return None
    idesc = _uniform(d, "idesc_value")
    if idesc is not None:
        return bsem._umma_descriptor_facts(
            s.opcode.upper(), int(idesc)).get("umma_kind")
    return bsem._umma_kind_of(s.opcode.upper())


def _f_cta_group(s, d, c):
    v = _uniform(d, "cta_group")
    if v is None:
        v = 2 if "2CTA" in s.opcode.upper() else 1
    return v


def _f_a_src(s, d, c):
    text = " ".join(_tokens(s)).lower()
    if "tmem[" in text and "gdesc[" not in text:
        return "tmem"
    if "gdesc[" in text:
        return "smem"
    return None


_TMEM_SHAPE_DOMAIN = frozenset(
    {"x%d" % i for i in (1, 2, 4, 8, 16, 32, 64)}
    | {"16dp256bit", "16dp32bit", "x128"})

_TMEM_COPY_SHAPES = {
    f"UTCCP.T.S{group}{suffix}": shape
    for group in ("", ".2CTA")
    for suffix, shape in (
        ("", "copy_default"),
        (".4DP256BIT", "4dp256bit"),
        (".128DP128BIT", "128dp128bit"),
        (".2X64DP128BIT_LW02_LW13", "2x64dp128bit_lw02_lw13"),
        (".4X32DP128BIT", "4x32dp128bit"),
    )
}


def _f_tmem_shape(s, d, c):
    root = s.opcode.upper().split(".", 1)[0]
    if root not in ("LDTM", "STTM", "UTCCP"):
        return NOT_APPLICABLE
    if root == "UTCCP":
        return _TMEM_COPY_SHAPES.get(s.opcode.upper())
    for m in s.opcode.split(".")[1:]:
        ml = m.lower()
        if ml in _TMEM_SHAPE_DOMAIN:
            return ml
        base = re.sub(r"_t\d+_t\d+$", "", ml)
        if base in _TMEM_SHAPE_DOMAIN:
            return base
    # fall back to the arch decoder: repeat-bearing .xN form
    rules = getattr(c, "rules", None) if c else None
    if rules and getattr(rules, "NAME", "") == "blackwell":
        from ..arch.blackwell import semantics as bsem
        _code, rep = bsem._tmem_shape_repeat(
            s.opcode.upper(), f"{s.opcode} @{hex(s.pc or 0)}")
        if f"x{rep}" in _TMEM_SHAPE_DOMAIN:
            return f"x{rep}"
    return None


def _f_tmem_repeat(s, d, c):
    root = s.opcode.upper().split(".", 1)[0]
    if root not in ("LDTM", "STTM", "UTCCP"):
        return NOT_APPLICABLE
    rules = getattr(c, "rules", None) if c else None
    if rules and getattr(rules, "NAME", "") == "blackwell":
        from ..arch.blackwell import semantics as bsem
        _code, rep = bsem._tmem_shape_repeat(
            s.opcode.upper(), f"{s.opcode} @{hex(s.pc or 0)}")
        return f"x{rep}"
    for m in s.opcode.upper().split(".")[1:]:
        if _XN_RE.fullmatch(m):
            return m.lower()
    return "x1"


def _f_cache_policy(s, d, c):
    for m in s.opcode.upper().split(".")[1:]:
        if m in ("CA", "CG"):
            return m.lower()
    return None


def _umma_idesc_facts(s, d):
    """All idesc-decoded UMMA axes; None without an idesc (missing)."""
    if s.opcode.split(".", 1)[0].upper() not in (
            "UTCHMMA", "UTCQMMA", "UTCOMMA"):
        return NOT_APPLICABLE
    idesc = _uniform(d, "idesc_value")
    if idesc is None:
        return None
    from ..arch.blackwell import semantics as bsem
    return bsem._umma_descriptor_facts(s.opcode.upper(), int(idesc))


def _f_umma_operation(s, d, c):
    f = _umma_idesc_facts(s, d)
    return None if f is None else         NOT_APPLICABLE if f is NOT_APPLICABLE else         f["umma_operation"]


def _f_umma_max_shift(s, d, c):
    f = _umma_idesc_facts(s, d)
    return None if f is None else         NOT_APPLICABLE if f is NOT_APPLICABLE else         f["umma_max_shift"]


def _f_umma_operand_major(s, d, c):
    f = _umma_idesc_facts(s, d)
    return None if f is None else         NOT_APPLICABLE if f is NOT_APPLICABLE else         f["umma_operand_major"]


# ---------------------------------------------------------------- dynamic

def _f_active_lanes(s, d, c):
    if d is None:
        return None
    return bin(d.active_mask & d.predicate_mask).count("1")


def _f_active_lanes_per_path(s, d, c):
    return _f_active_lanes(s, d, c)


def _f_predicated_on(s, d, c):
    """Whether the instance executes (active & predicate), not whether
    a static predicate token exists."""
    if d is None:
        return None
    return int((d.active_mask & d.predicate_mask) != 0)


def _f_memory_space(s, d, c):
    rules = c.rules if c else None
    if rules is not None:
        return rules.memory_space(s)
    return None


def _f_access_kind(s, d, c):
    m = _mem(d)
    if m:
        return m[0].access_kind
    return "none"


def _f_memory_width_bytes(s, d, c):
    m = _mem(d)
    return max(x.width_bytes for x in m) if m else None


def _f_width_bytes(s, d, c):
    return _f_memory_width_bytes(s, d, c)


def _f_sector_count(s, d, c):
    m = _hier(d)
    if not m:
        return None
    sectors = 0
    for ref in m:
        sectors += len({a // 32 for a in ref.lane_addresses})
    return sectors


def _f_cache_line_count(s, d, c):
    m = _mem(d)
    if not m:
        return None
    return len({(a // 128, r.space)
                for r in m for a in r.lane_addresses})


def _f_distinct_addresses(s, d, c):
    m = _hier(d)
    if not m:
        return None
    return len({a for r in m for a in r.lane_addresses})


def _f_address_stride_bytes(s, d, c):
    m = _mem(d)
    if not m or not m[0].lane_addresses:
        return None
    addrs = sorted(m[0].lane_addresses)
    if len(addrs) < 2:
        return 0
    return addrs[1] - addrs[0]


def _f_address_pattern(s, d, c):
    m = _mem(d)
    if not m or not m[0].lane_addresses:
        return None
    addrs = m[0].lane_addresses
    if len(set(addrs)) == 1:
        return "same"
    diffs = {addrs[i + 1] - addrs[i] for i in range(len(addrs) - 1)}
    return "strided" if len(diffs) == 1 else "pseudo_random"


def _f_working_set_bytes(s, d, c):
    m = _hier(d)
    if not m:
        return None
    lines = {a // 128 for r in m for a in r.lane_addresses}
    return len(lines) * 128


_BANK_MATRIX_RE = re.compile(
    r"(?:LDSM|STSM)\.\d+\.MT?\d\d(?:\.(\d+))?(?:\..*)?")


def _ref_lane_ids(ref):
    if ref.lane_mask:
        return [i for i in range(32) if (ref.lane_mask >> i) & 1]
    return list(range(len(ref.lane_addresses)))


def _bank_degree_group(opcode, is_load, raw_width, lane_ids,
                       addresses):
    """shared_bank_summary degree rule for one rank group."""
    matrix = _BANK_MATRIX_RE.fullmatch(opcode)
    if matrix:
        count = int(matrix.group(1) or 1)
        transactions: dict = {}
        for lane_id, address in zip(lane_ids, addresses):
            if int(lane_id) >= 8 * count:
                continue
            first_word = int(address) // 4
            counts = transactions.setdefault(int(lane_id) // 8, {})
            for word in range(first_word, first_word + 4):
                b = word % 32
                counts[b] = counts.get(b, 0) + 1
        return max((max(c.values()) for c in transactions.values()),
                   default=0)
    words = [a // 4 for a in addresses]
    banks = [w % 32 for w in words]
    if is_load and raw_width <= 4:
        words_by_bank: dict = {}
        for bank, word in zip(banks, words):
            words_by_bank.setdefault(bank, set()).add(word)
        return max((len(w) for w in words_by_bank.values()), default=0)
    counts: dict = {}
    for b in banks:
        counts[b] = counts.get(b, 0) + 1
    return max(counts.values(), default=0)


def _f_bank_conflict_degree(s, d, c):
    """hopper-prototype shared_bank.shared_bank_summary degree: 32 banks x 4 B;
    loads width<=4 count distinct words per bank (broadcast -> 1);
    LDSM/STSM per 8-lane transaction, 4 words per row, lanes >= 8*count
    ignored; other accesses count lanes per bank; shared::cluster
    addresses drop rank tag bits 24+ and group by rank (max)."""
    m = _mem(d)
    shared = [r for r in m if r.space in ("shared", "shared_cluster")]
    if m and not shared:
        return NOT_APPLICABLE
    refs = [r for r in shared if r.lane_addresses]
    if not refs:
        return None
    opcode = s.opcode.upper()
    worst = 0
    for ref in refs:
        raw_width = int(ref.width_bytes or 0) or 4
        is_load = ref.access_kind == "load"
        lane_ids = _ref_lane_ids(ref)
        if len(set(ref.lane_addresses)) == 1 and is_load:
            worst = max(worst, 1)
            continue
        by_rank: dict = {}
        for lane_id, addr in zip(lane_ids, ref.lane_addresses):
            rank = (int(addr) >> 24) & 0xFF
            by_rank.setdefault(rank, []).append((lane_id, int(addr)))
        for rows in by_rank.values():
            ids = [r[0] for r in rows]
            addrs = [r[1] & 0xFFFFFF for r in rows]
            worst = max(worst, _bank_degree_group(
                opcode, is_load, raw_width, ids, addrs))
    return worst


def _wavefronts_group(opcode, raw_width, lane_ids, addresses):
    """128 B wavefronts of one rank group: words touched by every lane
    (width/4 words each) mapped to 32 x 4 B banks; wavefronts = max over
    banks of the distinct words on that bank (a broadcast word counts
    once).  LDSM/STSM: per 8-lane transaction (4 words per row), summed."""
    matrix = _BANK_MATRIX_RE.fullmatch(opcode)
    words_per_lane = max(1, int(raw_width) // 4)
    if matrix:
        count = int(matrix.group(1) or 1)
        per_tx: dict = {}
        for lane_id, address in zip(lane_ids, addresses):
            if int(lane_id) >= 8 * count:
                continue
            first_word = int(address) // 4
            banks = per_tx.setdefault(int(lane_id) // 8, {})
            for word in range(first_word, first_word + 4):
                banks.setdefault(word % 32, set()).add(word)
        return sum(max((len(w) for w in banks.values()), default=0)
                   for banks in per_tx.values())
    banks: dict = {}
    for address in addresses:
        first_word = int(address) // 4
        for word in range(first_word, first_word + words_per_lane):
            banks.setdefault(word % 32, set()).add(word)
    return max((len(w) for w in banks.values()), default=0)


def _f_shared_wavefronts(s, d, c):
    """Shared-memory wavefronts of the instruction (max over rank
    groups of _wavefronts_group); 0 lanes -> 0."""
    m = _mem(d)
    shared = [r for r in m if r.space in ("shared", "shared_cluster")]
    if m and not shared:
        return NOT_APPLICABLE
    refs = [r for r in shared if r.lane_addresses]
    if not refs:
        return None
    opcode = s.opcode.upper()
    worst = 0
    for ref in refs:
        raw_width = int(ref.width_bytes or 0) or 4
        lane_ids = _ref_lane_ids(ref)
        by_rank: dict = {}
        for lane_id, addr in zip(lane_ids, ref.lane_addresses):
            by_rank.setdefault((int(addr) >> 24) & 0xFF, []).append(
                (lane_id, int(addr) & 0xFFFFFF))
        for rows in by_rank.values():
            worst = max(worst, _wavefronts_group(
                opcode, raw_width, [r[0] for r in rows],
                [r[1] for r in rows]))
    return worst


def _f_atomic_same_address_lanes(s, d, c):
    """Largest number of active lanes of an atomic / reduction hitting
    one address (same-address serialisation in the atomic unit)."""
    m = _mem(d)
    refs = [r for r in m if r.lane_addresses and
            r.access_kind in ("atomic", "async_reduce")]
    if not refs:
        return None
    worst = 0
    for ref in refs:
        counts: dict = {}
        for a in ref.lane_addresses:
            counts[int(a)] = counts.get(int(a), 0) + 1
        worst = max(worst, max(counts.values(), default=0))
    return worst


def _f_remote_smem(s, d, c):
    m = _mem(d)
    return int(any(r.remote_rank is not None for r in m)) if m else 0


def _f_bulk_direction(s, d, c):
    """UBLKCP / UBLKRED direction from the SASS dst.src modifier:
    .S.G / .S.S -> load (destination shared), .G.S -> store."""
    root = s.opcode.upper()
    if not (root.startswith("UBLKCP") or root.startswith("UBLKRED")):
        return None
    mods = root.split(".")[1:]
    if mods and mods[0].startswith("G"):
        return "store"
    if mods and mods[0].startswith("S"):
        return "load"
    return None


def _f_destination_count(s, d, c):
    for r in _mem(d):
        if r.destination_count is not None:
            return r.destination_count
        mm = getattr(r, "multicast_mask", None)
        if mm:
            return bin(mm).count("1")
    mm = _uniform(d, "multicast_mask")
    if mm:
        return bin(mm).count("1")
    root = s.opcode.upper().split(".", 1)[0]
    if root in ("LDGSTS", "UTMALDG", "UTMASTG", "UBLKCP", "UBLKRED",
                "UTMAREDG", "STAS", "ST", "STG", "STS", "LD", "LDS"):
        return 1
    return None


def _f_transfer_bytes(s, d, c):
    m = _mem(d)
    if not m:
        return _uniform(d, "copy_bytes")
    total = 0
    for r in m:
        if r.transfer_bytes is not None:
            total += r.transfer_bytes
        else:
            total += r.width_bytes * len(r.lane_addresses)
    return total


def _f_cut_rows(s, d, c):
    # tma_oob v2: rows of the box left in bounds when a tiled TMA load
    # is cut along dimension 0; 0 otherwise
    total = 0
    for r in _mem(d):
        if r.cut_rows is not None:
            total += r.cut_rows
    return total


def _f_expected_arrivals(s, d, c):
    v = _uniform(d, "expected_arrivals")
    if v is None:
        v = _uniform(d, "arrival_count")
    if v is None:
        return NOT_APPLICABLE
    return v


def _f_expected_tx_bytes(s, d, c):
    return _uniform(d, "expected_tx_bytes")


def _f_wait_group(s, d, c):
    root = s.opcode.upper().split(".", 1)[0]
    if root == "DEPBAR":
        from .semantics_common import depbar_operands, ConversionError
        try:
            keep, _sb0 = depbar_operands(s, d)
        except ConversionError:
            return None
        return keep
    if root == "WARPGROUP":
        rules = getattr(c, "rules", None) if c else None
        if rules is not None:
            fn = getattr(rules, "depbar_keep", None)
            if fn is not None:
                return fn(s, d)
        return _uniform(d, "depbar_keep")
    return NOT_APPLICABLE


def _f_copies_per_group(s, d, c):
    """cp.async copies committed in the warp's current group
    (predecessor _arrives len(copies))."""
    v = _uniform(d, "copies_per_group")
    if v is None and c and c.flow:
        v = getattr(c.flow, "cpasync_count", None)
    return v


def _f_requested_sleep_cycles(s, d, c):
    v = _uniform(d, "requested_ns")
    if v is not None:
        return v
    for t in _tokens(s):
        if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", t):
            return int(t, 0)
    return None


def _f_match_distinct_values(s, d, c):
    rules = c.rules if c else None
    cls = (rules.operation_class(s, d) if rules is not None
           else s.opcode.upper().split(".", 1)[0])
    if cls != "MATCH":
        return NOT_APPLICABLE
    if not s.source_registers:
        return None
    ops = d.operands if d else None
    if ops is None or not ops.lane:
        return None
    src = s.source_registers[0]
    if "U64" in s.opcode.upper():
        m = re.match(r"^R(\d+)$", src)
        if not m:
            return None
        lo, hi = "R" + m.group(1), "R" + str(int(m.group(1)) + 1)
        lv, hv = ops.lane.get(lo), ops.lane.get(hi)
        if lv is None or hv is None:
            return None
        return len(set(zip(lv, hv)))
    lv = ops.lane.get(src)
    if lv is None:
        return None
    return len(set(lv))


def _f_same_body_reuse(s, d, c):
    return _uniform(d, "same_body_reuse")


def _f_transaction_count(s, d, c):
    return _uniform(d, "transaction_count")


def _f_operand_facts_available(s, d, c):
    ops = d.operands if d else None
    return int(ops is not None and bool(ops.uniform))


# ---------------------------------------------------------------- launch

def _f_launch_warps_per_cta(s, d, c):
    if c and c.launch and c.launch.block[0]:
        threads = c.launch.block[0] * c.launch.block[1] \
            * c.launch.block[2]
        return threads // 32
    return None


def _f_registers_per_thread(s, d, c):
    return c.launch.registers_per_thread if c and c.launch else None


def _f_declared_math_warpgroups(s, d, c):
    if c and c.launch:
        return getattr(c.launch, "declared_math_warpgroups", None)
    return None


# ---------------------------------------------------------------- flow ctx

_CONSUMER_CLASS = {"FMUL": "fp32", "FADD": "fp32", "FFMA": "fp32",
                   "PRMT": "integer", "IMAD": "integer",
                   "MUFU": "special_function",
                   "MUFU.EX2": "special_function",
                   "MUFU.RSQ": "special_function",
                   "MUFU.LG2": "special_function",
                   "MUFU.TANH": "special_function"}


def _first_reader(c):
    """(reader_static, role|None) for this instance, or None."""
    if c is None or c.flow is None:
        return None
    return c.flow.reader


def _f_dependency_consumer_class(s, d, c):
    """Consumer class of the subject's first reader
    (job_v10 _COMPUTE_CONSUMER_CLASS); no reader or unmapped reader
    root -> NOT_APPLICABLE."""
    reader = _first_reader(c)
    if reader is None:
        return NOT_APPLICABLE
    reader_static, _role = reader
    root = reader_static.opcode.upper().split(".", 1)[0]
    if root not in _CONSUMER_CLASS:
        return NOT_APPLICABLE
    return _CONSUMER_CLASS[root]


def _f_dependency_source_role(s, d, c):
    """The first reader's register role (address/data); no reader ->
    NOT_APPLICABLE, reader without a recorded role -> missing."""
    reader = _first_reader(c)
    if reader is None:
        return NOT_APPLICABLE
    _reader_static, role = reader
    return role        # None -> missing_fact


def _f_shared_data_result_domain(s, d, c):
    reader = _first_reader(c)
    if reader is None:
        return NOT_APPLICABLE
    _reader_static, role = reader
    if role is None:
        return None        # reader exists but no recorded role
    return int(role == "data")


def _f_group_member_count(s, d, c):
    return _uniform(d, "group_member_count")


def _control(s, d, c, name):
    v = _uniform(d, name)
    if v is not None:
        return v
    if c and c.flow and hasattr(c.flow, name):
        return getattr(c.flow, name)
    return None


def _f_control_path_count(s, d, c):
    """Only defined for rows carrying reconvergence_timing_selector
    (predecessor _control_semantics); other sync ops ->
    NOT_APPLICABLE.  Selector present but count absent -> missing."""
    ops = d.operands if d else None
    if ops is not None:
        if ops.uniform.get("control_selector_present"):
            v = ops.uniform.get("sequential_region_count")
            return v
        v = ops.uniform.get("control_path_count")
        if v is not None:
            return v
    return NOT_APPLICABLE


def _f_control_participating_warps(s, d, c):
    return _control(s, d, c, "control_participating_warps")


def _f_loop_body_bytes(s, d, c):
    return _control(s, d, c, "loop_body_bytes")


def _f_loop_instruction_count(s, d, c):
    return _control(s, d, c, "loop_instruction_count")


def _f_backward_loop_class(s, d, c):
    ops = d.operands if d else None
    if ops is not None:
        v = ops.uniform.get("backward_loop_class_s")
        if v is not None:
            return v
    return _control(s, d, c, "backward_loop_class")


AXIS_FUNCTIONS = {
    "shared_wavefronts": _f_shared_wavefronts,
    "atomic_same_address_lanes": _f_atomic_same_address_lanes,
    "a_src": _f_a_src,
    "access_kind": _f_access_kind,
    "accumulator_dtype": _f_accumulator_dtype,
    "active_lanes": _f_active_lanes,
    "active_lanes_per_path": _f_active_lanes_per_path,
    "address_pattern": _f_address_pattern,
    "address_stride_bytes": _f_address_stride_bytes,
    "backward_loop_class": _f_backward_loop_class,
    "bank_conflict_degree": _f_bank_conflict_degree,
    "branch_direction": _f_branch_direction,
    "branch_taken": _f_branch_taken,
    "cache_line_count": _f_cache_line_count,
    "cache_policy": _f_cache_policy,
    "constant_source_count": _f_constant_source_count,
    "control_participating_warps": _f_control_participating_warps,
    "control_path_count": _f_control_path_count,
    "copies_per_group": _f_copies_per_group,
    "cta_group": _f_cta_group,
    "datatype": _f_datatype,
    "dependency_consumer_class": _f_dependency_consumer_class,
    "dependency_source_role": _f_dependency_source_role,
    "bulk_direction": _f_bulk_direction,
    "destination_count": _f_destination_count,
    "distinct_addresses": _f_distinct_addresses,
    "execution_unit": _f_execution_unit,
    "expected_arrivals": _f_expected_arrivals,
    "expected_tx_bytes": _f_expected_tx_bytes,
    "gpr_source_count": _f_gpr_source_count,
    "group_member_count": _f_group_member_count,
    "immediate_source_count": _f_immediate_source_count,
    "input_dtype": _f_input_dtype,
    "launch_warps_per_cta": _f_launch_warps_per_cta,
    "loop_body_bytes": _f_loop_body_bytes,
    "loop_instruction_count": _f_loop_instruction_count,
    "match_distinct_values": _f_match_distinct_values,
    "matrix_count": _f_matrix_count,
    "matrix_transposed": _f_matrix_transposed,
    "memory_space": _f_memory_space,
    "memory_width_bytes": _f_memory_width_bytes,
    "operand_facts_available": _f_operand_facts_available,
    "operation_class": _f_operation_class,
    "predicated_on": _f_predicated_on,
    "registers_per_thread": _f_registers_per_thread,
    "remote_smem": _f_remote_smem,
    "requested_sleep_cycles": _f_requested_sleep_cycles,
    "same_body_reuse": _f_same_body_reuse,
    "sector_count": _f_sector_count,
    "shared_data_result_domain": _f_shared_data_result_domain,
    "source_mode": _f_source_mode,
    "tensor_sparse": _f_tensor_sparse,
    "tensor_k": _f_tensor_k,
    "tensor_m": _f_tensor_m,
    "tensor_n": _f_tensor_n,
    "tensor_rank": _f_tensor_rank,
    "tma_box_planes": _f_tma_box_planes,
    "tma_form": _f_tma_form,
    "tma_instruction_rank": _f_tma_instruction_rank,
    "tmem_repeat": _f_tmem_repeat,
    "tmem_shape": _f_tmem_shape,
    "transaction_count": _f_transaction_count,
    "cut_rows": _f_cut_rows,
    "transfer_bytes": _f_transfer_bytes,
    "umma_kind": _f_umma_kind,
    "umma_max_shift": _f_umma_max_shift,
    "umma_operand_major": _f_umma_operand_major,
    "umma_operation": _f_umma_operation,
    "uniform_source_count": _f_uniform_source_count,
    "wait_group": _f_wait_group,
    "width_bytes": _f_width_bytes,
    "working_set_bytes": _f_working_set_bytes,
    "zero_source_count": _f_zero_source_count,
}


# Axes whose value derives from the instance's memory references,
# operand values, or flow context: undefined when the instance does not
# execute (predicated_on == 0).  Declarative; independent of tree
# validate.  predicated_on/active_lanes are deliberately absent — they
# report 0 for non-executing instances.
EXECUTION_AXES = frozenset({
    "access_kind",
    "address_pattern",
    "address_stride_bytes",
    "atomic_same_address_lanes",
    "bank_conflict_degree",
    "shared_wavefronts",
    "cache_line_count",
    "copies_per_group",
    "dependency_consumer_class",
    "dependency_source_role",
    "bulk_direction",
    "destination_count",
    "distinct_addresses",
    "match_distinct_values",
    "memory_width_bytes",
    "remote_smem",
    "sector_count",
    "shared_data_result_domain",
    "transfer_bytes",
    "width_bytes",
    "working_set_bytes",
})


def axis_value(name, static, dynamic=None, context=None):
    fn = AXIS_FUNCTIONS.get(name)
    if fn is None:
        raise KeyError(f"axis function not registered: {name}")
    if (dynamic is not None and name in EXECUTION_AXES
            and (dynamic.active_mask & dynamic.predicate_mask) == 0):
        return NOT_APPLICABLE
    return fn(static, dynamic, context)
