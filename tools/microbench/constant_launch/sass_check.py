#!/usr/bin/env python3
"""constant_launch sass_gate (protocol v1): for every probe kernel in the
build SASS dump, the window between CS2R SR_CLOCKLO #1 and #2 must hold
exactly U register-indexed `LDC R, c[0x<BANK>][R(+imm)]` and no other
memory instruction; the window between #2 and #3 likewise.

Writes checks.json:
  {"variants": {"cl.b0.s256.u4": {"ok": true, "window1_ldc": 4,
                                  "window2_ldc": 4, "other_mem": 0,
                                  "reason": ""}, ...}}

    python3 sass_check.py --sass <constlaunch.sass> --out <checks.json>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FUNC_RE = re.compile(r"Function : (_Z5k_cl(\d)ILi(\d+)ELi(\d+)EE\S*)")
INSN_RE = re.compile(r"\s*/\*([0-9a-f]+)\*/\s+(.*?;)")
CS2R_RE = re.compile(r"CS2R\s+\S+,\s*SR_CLOCKLO")
MEM_BASE = {"LDC", "ULDC", "LDCU", "LDS", "LDG", "LDL", "LDSM", "STG", "STS", "STL",
            "LD", "ST", "ATOM", "RED", "LDGSTS", "UTMALDG", "UBLKCP",
            "LDTM", "STTM", "TMA"}


def insn_op(text: str) -> str:
    """opcode token, dropping an @Pn guard."""
    t = text.strip()
    if t.startswith("@"):
        t = t.split(None, 1)[1]
    return t.split(None, 1)[0].rstrip(";")


def is_mem(text: str) -> bool:
    return insn_op(text).split(".")[0] in MEM_BASE


def is_chain_ldc(text: str, bank: int) -> bool:
    if insn_op(text).split(".")[0] != "LDC":
        return False
    m = re.search(r"c\[0x%x\]\[(.*?)\]" % bank, text)
    if not m:
        return False
    # register-indexed: an R/UR register inside the address brackets
    return bool(re.search(r"(?<![A-Za-z0-9_])[RU]*R\d+", m.group(1)))


def windows(insns: list[str]) -> list[list[str]]:
    idx = [i for i, t in enumerate(insns) if CS2R_RE.search(t)]
    out = []
    for a, b in zip(idx, idx[1:]):
        out.append(insns[a + 1:b])
    return out[:2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    variants: dict[str, dict] = {}
    cur = None
    insns: list[str] = []
    bank = u = s = 0

    def flush():
        if cur is None:
            return
        w = windows(insns)
        v = {"ok": False, "window1_ldc": 0, "window2_ldc": 0,
             "other_mem": 0, "reason": ""}
        if len(w) < 2:
            v["reason"] = f"only {len(w)} CS2R windows"
        else:
            for i, win in enumerate(w):
                ldc = sum(1 for t in win if is_chain_ldc(t, bank))
                other = sum(1 for t in win
                            if is_mem(t) and not is_chain_ldc(t, bank))
                v[f"window{i + 1}_ldc"] = ldc
                v["other_mem"] += other
            if (v["window1_ldc"] == u and v["window2_ldc"] == u
                    and v["other_mem"] == 0):
                v["ok"] = True
            else:
                v["reason"] = (
                    f"ldc {v['window1_ldc']}/{v['window2_ldc']} "
                    f"!= u{u} or other_mem {v['other_mem']}")
        variants[cur] = v

    for line in args.sass.read_text().splitlines():
        f = FUNC_RE.search(line)
        if f:
            flush()
            bank, u, s = int(f.group(2)), int(f.group(3)), int(f.group(4))
            cur = f"cl.b{bank}.s{s}.u{u}"
            insns = []
            continue
        m = INSN_RE.match(line)
        if cur is not None and m:
            insns.append(m.group(2))
    flush()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"variants": variants}, indent=1) + "\n")
    ok = sum(1 for v in variants.values() if v["ok"])
    print(f"sass_check: {ok}/{len(variants)} variants ok")
    for name, v in sorted(variants.items()):
        if not v["ok"]:
            print(f"  {name}: {v['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
