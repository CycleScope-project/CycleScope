"""Static instruction tables from a capture case directory.

Reads functions.jsonl + functions/<function_id>.tsv into per
(function_id, idx) ``model.StaticInstruction`` objects.  Ported from
blackwellsim convert/static.py; arch-specific bits enter through the
rules module (operation_class, memory_space).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from cyclescope.convert import model
from cyclescope.convert.sass_decode import (MissingFact, control_word_from_bytes,
                                      numeric_control_target, opcode_root)

_REG_TOKEN = re.compile(
    r"\b(UR\d+|R\d+|UP\d+|P\d+|SR_[A-Z_]+|B\d+|RZ|URZ|PT|UPT)\b")
_ADDR_BRACKET = re.compile(r"\[([^\]]+)\]")

# opcode root -> static memory space; GENERIC resolves per record from
# the SPACE ballot at dynamic time
_SPACE_BY_ROOT = {
    "LDS": "shared", "STS": "shared", "LDSM": "shared", "STSM": "shared",
    "LDTM": "shared", "STTM": "shared",
    "ATOMS": "shared", "REDS": "shared", "STAS": "shared",
    "SYNCS": "shared", "ARRIVES": "shared", "UTCBAR": "shared",
    "UCGABAR": "shared", "MBAR": "shared",
    "LDL": "local", "STL": "local",
    "LDC": "constant", "ULDC": "constant", "LDCU": "constant",
    "LDG": "global", "STG": "global", "ATOMG": "global",
    "REDG": "global", "LDGSTS": "global",
    "UTMALDG": "shared", "UTMASTG": "global", "UTMAREDG": "global",
    "UTMAPF": "global", "UBLKCP": "global", "UBLKRED": "global",
}
_GENERIC_ROOTS = {"LD", "ST", "ATOM", "RED"}


@dataclass
class StaticRow:
    """Per-idx decoded row: model view + converter-private fields."""
    idx: int
    sass: str
    ops_layout: tuple
    mref_count: int
    opcode_text: str          # opcode with modifiers (no guard)
    operands_text: str
    branch_target: int | None
    static_index: int
    inst: model.StaticInstruction


@dataclass
class FunctionStatic:
    function_id: int
    name: str
    mangled: str
    addr: int
    rows: dict = field(default_factory=dict)  # idx -> StaticRow


def _parse_operands(sass: str):
    """(guard, opcode, operands_text, dests, srcs) from a SASS line."""
    text = sass.strip()
    guard = None
    if text.startswith("@"):
        head, _, rest = text.partition(" ")
        guard = head[1:]
        text = rest.strip()
    head, _, operands = text.partition(" ")
    opcode = head.strip().rstrip(";")
    operands = operands.split(";")[0].strip()
    tokens = [t.strip() for t in operands.split(",") if t.strip()]
    dests: list[str] = []
    srcs: list[str] = []
    for index, token in enumerate(tokens):
        regs = _REG_TOKEN.findall(token)
        if index == 0 and token and "[" not in token:
            dests.extend(regs)
        else:
            srcs.extend(regs)
    return guard, opcode, operands, dests, srcs


def _register_roles(operands_text: str, srcs, dests):
    """Registers inside [brackets] are addresses; the rest are data."""
    address_regs = set()
    for token in _ADDR_BRACKET.findall(operands_text):
        address_regs.update(_REG_TOKEN.findall(token))
    return {reg: ("address" if reg in address_regs else "data")
            for reg in list(srcs) + list(dests)}


def load_case(case_dir, rules) -> dict[int, FunctionStatic]:
    """{function_id: FunctionStatic} for the whole case."""
    case_dir = Path(case_dir)
    functions_path = case_dir / "functions.jsonl"
    if not functions_path.is_file():
        raise MissingFact(str(case_dir), "functions.jsonl")
    names: dict[int, str] = {}
    mangleds: dict[int, str] = {}
    addrs: dict[int, int] = {}
    for line in functions_path.read_text().splitlines():
        row = json.loads(line)
        fid = int(row.get("function_id", row.get("reg")))
        names[fid] = str(row.get("name") or "")
        mangleds[fid] = str(row.get("mangled") or names[fid])
        if "addr" in row:
            addrs[fid] = int(str(row["addr"]), 16)
    tables: dict[int, FunctionStatic] = {}
    for tsv in sorted((case_dir / "functions").glob("*.tsv")):
        fid = int(tsv.stem)
        table = tables.setdefault(fid, FunctionStatic(
            function_id=fid, name=names.get(fid, f"function{fid}"),
            mangled=mangleds.get(fid, names.get(fid, f"function{fid}")),
            addr=addrs.get(fid, 0)))
        for line in tsv.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 6:
                raise MissingFact(str(tsv),
                                  f"malformed row: {line[:80]!r}")
            idx = int(parts[0])
            pc = int(parts[1], 16)
            bytes_hex = parts[2]
            sass = parts[3].strip()
            ops_layout = tuple(parts[4].split(",")) if parts[4] else ()
            mref_count = int(parts[5])
            guard, opcode, operands, dests, srcs = _parse_operands(sass)
            root = opcode_root(opcode)
            branch_target = None
            if root in ("BRA", "BSSY"):
                try:
                    branch_target = numeric_control_target(
                        operands, root)
                except MissingFact:
                    branch_target = None
            static = model.StaticInstruction(
                pc=pc,
                code_address=table.addr + pc if table.addr else None,
                opcode=opcode,
                operation_class="",
                operands=tuple(
                    t.strip() for t in operands.split(",") if t.strip()),
                predicate=guard.lstrip("!") if guard else None,
                predicate_inverted=bool(guard and guard.startswith("!")),
                source_registers=tuple(srcs),
                destination_registers=tuple(dests),
                register_roles=_register_roles(operands, srcs, dests),
                control=control_word_from_bytes(bytes_hex),
                memory_space=_SPACE_BY_ROOT.get(
                    root, "generic" if root in _GENERIC_ROOTS else None),
                branch_target=branch_target)
            static.operation_class = rules.operation_class(static, None)
            table.rows[idx] = StaticRow(
                idx=idx, sass=sass, ops_layout=ops_layout,
                mref_count=mref_count, opcode_text=opcode,
                operands_text=operands,
                branch_target=branch_target,
                static_index=-1, inst=static)
    return tables
