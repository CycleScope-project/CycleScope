"""Engine stall ledger (``cyclescope-engine simulate --stall-ledger``).

Tab-separated; schema cyclescope.stall-ledger.v1 (docs/engine.md):

    #schema  cyclescope.stall-ledger.v1
    #warps   <warps that became resident>
    #cycles  <kernel cycles>
    residency   <code_address hex> <operation_class> <reason> <warp_cycles> <events>
    transition  <from code_address hex> <to code_address hex> <count>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "cyclescope.stall-ledger.v1"


@dataclass
class StallLedger:
    warps: int
    cycles: int
    # (code address, reason) -> [warp cycles, events]
    residency: dict[tuple[int, str], list[int]] = field(default_factory=dict)
    operation_class: dict[int, str] = field(default_factory=dict)
    # (from code address, to code address) -> count
    transitions: dict[tuple[int, int], int] = field(default_factory=dict)


def load_ledger(path: Path) -> StallLedger:
    header: dict[str, str] = {}
    ledger = StallLedger(warps=0, cycles=0)
    with Path(path).open() as fh:
        for number, line in enumerate(fh, 1):
            cells = line.rstrip("\n").split("\t")
            if not cells or cells == [""]:
                continue
            kind = cells[0]
            if kind.startswith("#"):
                header[kind[1:]] = cells[1] if len(cells) > 1 else ""
            elif kind == "residency" and len(cells) == 6:
                address = int(cells[1], 16)
                entry = ledger.residency.setdefault((address, cells[3]),
                                                    [0, 0])
                entry[0] += int(cells[4])
                entry[1] += int(cells[5])
                ledger.operation_class.setdefault(address, cells[2])
            elif kind == "transition" and len(cells) == 4:
                key = (int(cells[1], 16), int(cells[2], 16))
                ledger.transitions[key] = \
                    ledger.transitions.get(key, 0) + int(cells[3])
            else:
                raise ValueError(
                    f"unsupported:stall_ledger_row ({path}:{number})")
    if header.get("schema") != SCHEMA:
        raise ValueError(f"unsupported:stall_ledger_schema:"
                         f"{header.get('schema')} ({path})")
    for name in ("warps", "cycles"):
        if name not in header:
            raise ValueError(f"missing_fact:stall_ledger_{name} ({path})")
    ledger.warps = int(header["warps"])
    ledger.cycles = int(header["cycles"])
    if ledger.warps <= 0:
        raise ValueError(f"missing_fact:stall_ledger_warps ({path})")
    return ledger
