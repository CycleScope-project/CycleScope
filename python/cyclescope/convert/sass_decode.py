"""Stateless SASS decoders shared by the capture converter.

Ported from blackwellsim convert/facts.py + sass_isa.py -- mechanical
decoders over a static row or a captured record, no cross-warp
inference, no case conditions.
"""
from __future__ import annotations

import re

from cyclescope.convert import model


class MissingFact(Exception):
    def __init__(self, where: str, fact: str):
        super().__init__(f"missing_fact:{fact} ({where})")
        self.fact = fact


class Unsupported(Exception):
    def __init__(self, where: str, reason: str):
        super().__init__(f"unsupported:{reason} ({where})")
        self.reason = reason


PURE_INTEGER_REGISTER_CONTROL_ROOTS = frozenset({
    "BRA", "NOP", "MOV", "UMOV", "IADD3", "UIADD3", "VIADD",
    "IMAD", "UIMAD", "LOP3", "ULOP3", "ISETP", "UISETP",
    "PLOP3", "UPLOP3", "SEL", "USEL", "LEA", "ULEA", "SHF",
    "USHF", "PRMT", "UPRMT", "BFE", "UBFE", "BMSK", "POPC",
    "FLO", "IMNMX", "UIMNMX",
})


def opcode_root(opcode: object) -> str:
    text = str(opcode or "").strip()
    if text.startswith("@"):
        _head, _, text = text.partition(" ")
    text = text.split(None, 1)[0] if text else ""
    return text.split(".", 1)[0].upper()


def sass_tokens(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = [token.strip() for token in text.split(";") if token.strip()]
    if len(parts) == 1 and "," in parts[0]:
        parts = [token.strip() for token in parts[0].split(",")
                 if token.strip()]
    return parts


def numeric_control_target(operands: object, operation: str) -> int:
    numeric: list[int] = []
    for token in sass_tokens(operands):
        value = token.strip()
        if re.fullmatch(r"0[xX][0-9a-fA-F]+|[0-9]+", value):
            numeric.append(int(value, 0))
    if len(numeric) != 1:
        raise MissingFact(
            str(operands), f"{operation} needs one numeric static target")
    return numeric[0]


def register_name(token: str) -> str:
    return token.strip().lstrip("+-").split(".", 1)[0].upper()


def cluster_rank_of(x: int, y: int, z: int, cx: int, cy: int,
                    cz: int) -> int:
    cx, cy, cz = max(1, cx), max(1, cy), max(1, cz)
    return (x % cx) + (y % cy) * cx + (z % cz) * cx * cy


# cache-control/prefetch ops have a memory operand but no data width
_NO_DATA_WIDTH = {"CCTL", "CCTLL", "PREFETCH", "UTMACCTL"}

_MEMORY_DEFAULT_FOUR = {"LDC", "LDCU", "ULDC", "LDG", "LD", "LDL", "LDS",
                        "STG", "ST", "STL", "STS", "STAS", "ATOM", "ATOMS",
                        "ATOMG", "RED", "REDG", "REDS", "REDAS", "LDGSTS",
                        "QSPC", "MEMBAR"}


def opcode_width_bytes(opcode: object, where: str) -> int | None:
    """Memory access width in bytes decoded from the mnemonic."""
    text = str(opcode or "").upper()
    vector = re.search(r"(?:^|\.)(?:F|S|U)(8|16|32|64)X(\d+)(?:\.|$)",
                       text)
    if vector is not None:
        return (int(vector.group(1)) // 8) * int(vector.group(2))
    typed = re.search(r"(?:^|\.)(?:B|F|S|U)(8|16|32|64)(?:\.|$)", text)
    if typed is not None:
        return int(typed.group(1)) // 8
    for bits, width in (("256", 32), ("128", 16), ("64", 8), ("32", 4),
                        ("16", 2), ("8", 1)):
        if re.search(rf"(?:^|\.){bits}(?:\.|$)", text):
            return width
    root = text.split(".", 1)[0].split(None, 1)[0]
    matrix = re.search(r"(?:^|\.)M(?:88|16816)(?:\.|$)", text)
    if root in {"LDSM", "STSM"}:
        # per-lane bytes: one 32-bit register per 8x8 b16 matrix
        count = re.search(r"\.X(\d)(?:\.|$)", text)
        return 4 * (int(count.group(1)) if count else 1)
    if root in _NO_DATA_WIDTH:
        return None
    if root in _MEMORY_DEFAULT_FOUR:
        return 4
    # mbarrier-family ops address an 8-byte object; no width token.
    if root in {"SYNCS", "MBAR", "ARRIVES", "UTCBAR", "UCGABAR"}:
        return 8
    raise MissingFact(where, f"memory width is undecodable: {opcode!r}")


# tcgen05.alloc column counts (PTX: powers of two, 32..512); the SASS
# alloc operand carries either the count or 32-column granules
TMEM_COLUMN_SET = frozenset({32, 64, 128, 256, 512})
TMEM_GRANULE = 32


def tmem_columns_from_operand(value) -> int | None:
    """UTCATOMSWS FIND_AND_SET operand -> column count (None: not a
    legal count in either unit)."""
    if not isinstance(value, int):
        return None
    if value in TMEM_COLUMN_SET:
        return value
    scaled = value * TMEM_GRANULE
    return scaled if scaled in TMEM_COLUMN_SET else None


def decode_mbarrier_init_expected_arrivals(raw_state: int) -> int | None:
    """SYNCS.EXCH operand -> PTX mbarrier.init count.

    Hopper lowers the count into the low 21 bits as ``(2**20-count)<<1``;
    SM103 stores ``2**20 - count`` without the shift.
    """
    raw = int(raw_state)
    if raw >= 1 << 32:
        low = raw & 0xFFFFFFFF
        high = raw >> 32
        if low & 1 or not 0 < low < 1 << 21:
            return None
        count = (1 << 20) - (low >> 1)
        # bit 31 of the high word is the initial phase (0 or 1); the rest
        # of the high word is the pending-count mirror
        if (0 < count < 1 << 20 and (high & 0x7FFFFFFF) ==
                ((0x80000000 - (count << 11)) & 0x7FFFFFFF)):
            return count
        return None
    # a real mbarrier.init count is small; reject decoded counts that
    # cannot be an arrival count (e.g. a stray non-init operand value)
    bound = 1 << 16
    hopper: int | None = None
    if not (raw & 1) and 0 < raw < (1 << 21):
        expected = (1 << 20) - (raw >> 1)
        if 0 < expected <= bound:
            hopper = expected
    sm103: int | None = None
    if 0 < raw < (1 << 20):
        expected = (1 << 20) - raw
        if 0 < expected <= bound:
            sm103 = expected
    if hopper is not None and sm103 is None:
        return hopper
    if sm103 is not None and hopper is None:
        return sm103
    if hopper is None:
        return None
    if hopper <= 1024:
        return hopper
    if sm103 is not None and sm103 <= 1024:
        return sm103
    return hopper


def is_complete_tx_form(opcode: str) -> bool:
    """`.A<n>TX` arrive spelling: tx flag is COMPLETE transaction bytes
    (mbarrier.complete_tx), not expect_tx — nvjet emits it for the CLC
    response bytes when CLC is not queried.  `.A<n>TR` keeps its
    expect_tx meaning."""
    return re.search(r"(?:^|\.)A\d+TX(?:\.|$)", opcode.upper()) is not None


def arrival_and_tx_factors(opcode: str) -> tuple[int, int]:
    """(arrival factor per executing lane, expect-tx flag).  `.A<n>TX`
    (complete_tx) returns the tx flag like `.A<n>TR`; callers split on
    is_complete_tx_form."""
    upper = opcode.upper()
    if upper.split(".", 1)[0] == "UTCBAR":
        return 1, 0
    match = re.search(r"(?:^|\.)A(\d+)TX(?:\.|$)", upper)
    if match:
        return int(match.group(1)), 1
    match = re.search(r"(?:^|\.)A(\d+)T(\d+)(?:\.|$)", upper)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(?:^|\.)A(\d+)TR\d*(?:\.|$)", upper)
    if match:
        return int(match.group(1)), 1
    match = re.search(r"(?:^|\.)ART(\d+)(?:\.|$)", upper)
    if match:
        return 1, int(match.group(1))
    if "RED" in upper.split("."):
        if "TRANS64" in upper.split("."):
            return 1, 1
        return 1, 0
    if "ARVCNT" in upper.split("."):
        return 1, 0
    if "TRANSCNT" in upper.split("."):
        return 0, 1     # completes one transaction per lane, no arrival
    if upper.split(".") == ["SYNCS", "ARRIVE", "TRANS64"]:
        return 1, 1
    raise Unsupported(opcode,
                      f"unrecognized arrive spelling {opcode}")


def control_schedule_fields(word: int) -> dict:
    """Schedule fields of one 64-bit SASS control word."""
    return {
        "stall": (word >> 41) & 0xF,
        "yield_hint": 1 - ((word >> 45) & 1),
        "write_barrier": (word >> 46) & 7,
        "read_barrier": (word >> 49) & 7,
        "wait_mask": (word >> 52) & 0x3F,
        "reuse_mask": (word >> 58) & 0xF,
    }


def control_word_from_bytes(bytes_hex: str) -> model.ControlWord:
    """model.ControlWord from the high 64 bits of the 16-byte encoding."""
    if len(bytes_hex) < 32:
        return model.ControlWord()
    word = int.from_bytes(bytes.fromhex(bytes_hex[16:32]), "little")
    f = control_schedule_fields(word)
    return model.ControlWord(**f)


TENSOR_DTYPE_BYTES: dict[int, float] = {
    0: 1, 1: 2, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2, 10: 4,
    11: 4, 12: 4,
}


def captured_arrival_count(opcode: str, lanes: int) -> int:
    """Lane-weighted arrival count: UTCBAR is a uniform elect-thread
    commit (one arrival); other arrive forms arrive once per executing
    lane times the opcode's A factor."""
    factor = arrival_and_tx_factors(opcode)[0]
    if factor <= 0 or lanes <= 0:
        return 0
    if opcode.upper().split(".", 1)[0] == "UTCBAR":
        return factor
    return lanes * factor
