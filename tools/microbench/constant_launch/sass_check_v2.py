#!/usr/bin/env python3
"""constant_launch v2 sass_gate: per variant kernel, exactly 33 CS2R
SR_CLOCKLO and exactly 32 register-indexed LDC R, c[0x<BANK>][R(+imm)]
alternating with them in program order (t, load, t, load, ...), no
other memory instruction between the first and last CS2R, and between
consecutive LDCs either nothing but the CS2R or the same single ALU op
(class recorded in between_op).

checks.json:
  {"variants": {"cl.b0.s256": {"ok": true, "ldc": 32, "cs2r": 33,
   "between_op": "IMAD.WIDE.U32", "reason": ""}}}

    python3 sass_check_v2.py --sass <constlaunch_v2.sass> --out checks.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FUNC_RE = re.compile(r"Function : (_Z5k_cl(\d)ILi(\d+)EE\S*)")
INSN_RE = re.compile(r"\s*/\*([0-9a-f]+)\*/\s+(.*?;)")
CS2R_RE = re.compile(r"CS2R\s+\S+,\s*SR_CLOCKLO")
MEM_BASE = {"LDC", "ULDC", "LDCU", "LDS", "LDG", "LDL", "LDSM", "STG",
            "STS", "STL", "LD", "ST", "ATOM", "RED", "LDGSTS",
            "UTMALDG", "UBLKCP", "LDTM", "STTM", "TMA"}
ALU_BASE = {"IMAD", "IADD", "IADD3", "VIADD", "LEA", "SHF", "USHF",
            "SGXT", "MOV", "UMOV", "IMNMX", "LOP3", "SEL"}


def insn_op(text: str) -> str:
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
    return bool(m and re.search(r"(?<![A-Za-z0-9_])[RU]*R\d+", m.group(1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    variants: dict[str, dict] = {}
    cur = None
    insns: list[str] = []
    bank = s = 0

    def flush():
        if cur is None:
            return
        v = {"ok": False, "ldc": 0, "cs2r": 0, "between_op": "", "reason": ""}
        kinds = []
        for t in insns:
            if CS2R_RE.search(t):
                kinds.append("cs2r")
            elif is_chain_ldc(t, bank):
                kinds.append("ldc")
            elif is_mem(t):
                kinds.append("mem:" + insn_op(t))
            else:
                kinds.append("alu:" + insn_op(t))
        ci = [i for i, k in enumerate(kinds) if k == "cs2r"]
        # only the loads inside the first..last CS2R span are chain
        # loads; a ptxas prefetch of a chain-shaped LDC outside the span
        # does not count (and is harmless to the measurement)
        span = (ci[0], ci[-1]) if ci else (0, -1)
        li = [i for i, k in enumerate(kinds)
              if k == "ldc" and span[0] <= i <= span[1]]
        v["cs2r"] = len(ci)
        v["ldc"] = len(li)
        why = []
        if len(ci) != 33:
            why.append(f"cs2r {len(ci)} != 33")
        if len(li) != 32:
            why.append(f"ldc {len(li)} != 32")
        # alternating: each ldc sits between two cs2r
        ok_order = len(ci) == 33 and len(li) == 32 and \
            all(li[j] > ci[j] and li[j] < ci[j + 1] for j in range(32))
        if not ok_order and len(ci) == 33 and len(li) == 32:
            why.append("loads not alternating with clocks")
        # no other memory inside the first..last cs2r span
        if ci:
            other = [k for k in kinds[ci[0]:ci[-1] + 1]
                     if k.startswith("mem:")]
            if other:
                why.append(f"other memory: {sorted(set(other))[:3]}")
        # ops between consecutive LDCs (excluding the CS2R)
        ops = set()
        bad = False
        for a, b in zip(li, li[1:]):
            gap = [kinds[i] for i in range(a + 1, b) if kinds[i] != "cs2r"]
            if len(gap) > 1:
                bad = True
            ops.update(gap)
        if bad:
            why.append("more than one op between a load pair")
        elif len(ops) > 1:
            why.append(f"mixed between-ops: {sorted(ops)}")
        elif len(ops) == 1:
            v["between_op"] = sorted(ops)[0].split(":", 1)[1]
            if v["between_op"].split(".")[0] not in ALU_BASE:
                why.append(f"non-ALU between-op {v['between_op']}")
        v["ok"] = not why
        v["reason"] = "; ".join(why)
        variants[cur] = v

    for line in args.sass.read_text().splitlines():
        f = FUNC_RE.search(line)
        if f:
            flush()
            bank, s = int(f.group(2)), int(f.group(3))
            cur = f"cl.b{bank}.s{s}"
            insns = []
            continue
        m = INSN_RE.match(line)
        if cur is not None and m:
            insns.append(m.group(2))
    flush()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"variants": variants}, indent=1) + "\n")
    ok = sum(1 for v in variants.values() if v["ok"])
    print(f"sass_check_v2: {ok}/{len(variants)} variants ok")
    for name, v in sorted(variants.items()):
        print(f"  {name}: ok={v['ok']} ldc={v['ldc']} cs2r={v['cs2r']} "
              f"between_op={v['between_op'] or '-'} {v['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
