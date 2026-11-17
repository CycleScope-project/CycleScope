"""Parse `cuobjdump -sass` output into per-kernel instruction lists with the
Volta+ control word decoded from the high 64 bits of each 128-bit
instruction (bits 105..127 of the instruction):

  stall  = ctrl[0:4]   cycles the issuing warp waits before its next issue
  yield  = ctrl[4]
  wbar   = ctrl[5:8]   scoreboard set on result write (7 = none)
  rbar   = ctrl[8:11]  scoreboard set on operand read  (7 = none)
  wait   = ctrl[11:17] scoreboards this instruction waits on before issue
  reuse  = ctrl[17:21]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FUNC = re.compile(r"^\s*Function : (\S+)")
_INST = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\* (0x[0-9a-f]{16}) \*/")
_HIGH = re.compile(r"^\s*/\* (0x[0-9a-f]{16}) \*/\s*$")


@dataclass
class Inst:
    addr: int
    text: str
    low: int
    high: int = 0

    @property
    def pred(self) -> str | None:
        m = re.match(r"^(@!?U?P[T0-9]+)\s+", self.text)
        return m.group(1) if m else None

    @property
    def body(self) -> str:
        return re.sub(r"^@!?U?P[T0-9]+\s+", "", self.text).strip()

    @property
    def opcode(self) -> str:
        return self.body.split()[0] if self.body else ""

    @property
    def root(self) -> str:
        return self.opcode.split(".")[0]

    @property
    def ctrl(self) -> int:
        return (self.high >> 41) & ((1 << 23) - 1)

    @property
    def stall(self) -> int:
        return self.ctrl & 0xF

    @property
    def yld(self) -> int:
        return (self.ctrl >> 4) & 1

    @property
    def wbar(self) -> int:
        return (self.ctrl >> 5) & 7

    @property
    def rbar(self) -> int:
        return (self.ctrl >> 8) & 7

    @property
    def wait(self) -> int:
        return (self.ctrl >> 11) & 0x3F

    def ctrl_str(self) -> str:
        w = "-" if self.wbar == 7 else str(self.wbar)
        r = "-" if self.rbar == 7 else str(self.rbar)
        return f"S{self.stall:02d} W{w} R{r} wait{self.wait:06b}"


@dataclass
class Kernel:
    name: str
    insts: list[Inst] = field(default_factory=list)


def parse(text: str) -> dict[str, Kernel]:
    kernels: dict[str, Kernel] = {}
    cur: Kernel | None = None
    pending: Inst | None = None
    for line in text.splitlines():
        m = _FUNC.match(line)
        if m:
            cur = kernels.setdefault(m.group(1), Kernel(m.group(1)))
            pending = None
            continue
        if cur is None:
            continue
        m = _INST.match(line)
        if m:
            pending = Inst(int(m.group(1), 16), m.group(2).strip(), int(m.group(3), 16))
            cur.insts.append(pending)
            continue
        m = _HIGH.match(line)
        if m and pending is not None and pending.high == 0:
            pending.high = int(m.group(1), 16)
            pending = None
    return kernels


def clock_windows(k: Kernel) -> list[list[Inst]]:
    """Instruction runs strictly between consecutive SR_CLOCKLO reads."""
    idx = [i for i, ins in enumerate(k.insts) if "SR_CLOCKLO" in ins.text]
    return [k.insts[a + 1:b] for a, b in zip(idx, idx[1:])]


def waits_on(ins: Inst, bar: int) -> bool:
    return bar != 7 and bool(ins.wait & (1 << bar))
