#!/usr/bin/env python3
"""SASS gate for the h100 wgmma probe (protocol-v2).

For every k_wgmma<CID, U> kernel in the dumped SASS:
  * exactly U GMMA instructions of the cell's shape appear between the
    two SR_CLOCKLO reads of each rep window (static count == U);
  * rs cells: the A operand is a register, never gdesc[...];
  * between t_s (first clock read) and the first GMMA only
    WARPGROUP.ARRIVE-class instructions are allowed.

Usage: sass_check.py <wgmma.sass> <wgmma_cells.json>
Exit 0 = pass; prints FAIL lines otherwise.
"""

from __future__ import annotations

import json
import re
import sys

FUNC_RE = re.compile(r"Function\s*:\s*(\S+)")
CID_RE = re.compile(r"k_wgmmaILi(\d+)ELi(\d+)E")
SERGRP_RE = re.compile(r"k_sergrpILi(\d+)ELi(\d+)ELi(\d+)E")
DEPBAR0_RE = re.compile(r"WARPGROUP\.DEPBAR\.LE\s+gsb0\s*,\s*0x0")
OP_RE = re.compile(r"\b(HGMMA|QGMMA|IGMMA)(\.SP)?\.(\d+)x(\d+)x(\d+)\b|UTC[HQO]MMA")
ALLOWED = re.compile(
    r"^\s*(/\*[0-9a-f]+\*/\s*)?(@!?U?P\d+\s+)?"
    r"(NOP|WARPGROUP|DEPBAR|UTRANCB|UTRANS|UTCB|ELECT|BAR|"
    r"R2UR|UMOV|S2R|S2UR|IMAD|MOV|LOP3|SHF|LEA|ULDC|LDCU|VOTEU|"
    r"CS2R|PRMT|SEL|UIADD3|USHF|UMOV|ISETP|UISETP|FENCE|"
    r"LD\.|ST\.|LDS|STS|UTMALDG|LDSM|STS|LDGDEPBAR|"
    r"WGMMA|GMMA|MBAR|ARRIVES|USCHED|USCHED|BSSY|BSYNC)\b")


def shapes(cell):
    n, k = cell["n"], cell["k"]
    return cell["class"], n, k


_SEEN = 0
_CIDS = []
_CIDS_SEEN = set()


def check(sass_path: str, cells_path: str) -> int:
    global _SEEN
    cells = {c["cid"]: c for c in json.load(open(cells_path))}
    fails = []
    cur = None
    ops = []
    clocklos = 0
    body = []
    for line in open(sass_path, errors="replace"):
        fm = FUNC_RE.search(line)
        if fm:
            if cur is not None:
                fails += finish(cur, ops, body, cells)
            cur = fm.group(1)
            _m = CID_RE.search(cur)
            if _m and cur not in _CIDS_SEEN:
                _SEEN += 1
                _CIDS.append(int(_m.group(1)))
                _CIDS_SEEN.add(cur)
            _sm = SERGRP_RE.search(cur)
            if _sm and cur not in _CIDS_SEEN:
                _SEEN += 1
                _CIDS.append(int(_sm.group(1)))
                _CIDS_SEEN.add(cur)
            ops, body, clocklos = [], [], 0
            continue
        if cur is None:
            continue
        body.append(line)
        m = OP_RE.search(line)
        if m:
            # ptxas emits a predicated-off marker op (!UPT, dest RZ)
            # for the empty U=0 group -- not a real cell op
            if "RZ," in line and "!UPT" in line:
                continue
            ops.append((m, line))
    if cur is not None:
        fails += finish(cur, ops, body, cells)
    if not _SEEN:
        fails.append("no kernels parsed (empty or header-only SASS)")
    else:
        # expected = 6 instantiations x cells of this file's kind
        kinds = {cells[c]["kind"] for c in _CIDS if c in cells}
        # 2 lat + 4 cad + 4 ser per cell; +4 grp for bf16/e4m3e4m3 ss
        # n in {64,128,256}
        want = 0
        for c in cells.values():
            if c["kind"] not in kinds:
                continue
            want += 10
            if (c["src"] == "ss" and
                    c["kind"] in ("bf16", "e4m3e4m3") and
                    c["n"] in (64, 128, 256)):
                want += 4
        if _SEEN != want:
            fails.append(f"{_SEEN} kernels parsed, expected {want}")
    for f in fails:
        print("FAIL", f)
    print(f"sass_check: {_SEEN} kernels checked, {len(fails)} failures")
    return 1 if fails else 0


def between_target_fails(name, ops, body):
    pos = {i: l for i, l in enumerate(body)}
    idxs = [i for i, l in enumerate(body)
            if any(l is ol for _, ol in ops)]
    fails = []
    for a, b in zip(idxs, idxs[1:]):
        for l in body[a + 1:b]:
            if re.search(r"\bBRA\b|\bELECT\b|\bWARPSYNC\b", l):
                fails.append(
                    f"{name}: BRA/ELECT/WARPSYNC between target ops: "
                    f"{l.strip()[:80]}")
                break
    return fails


def finish_sergrp(name, ops, body, cells):
    """ser/grp gate (h100-r4 addendum): ser = exactly U target ops and
    U DEPBAR.LE gsb0,0x0; grp = 4U ops and U DEPBARs; no
    BRA/ELECT/WARPSYNC between target ops; the null HGMMA pattern must
    not appear; ptxas C7511 is not a rejection here."""
    fails = between_target_fails(name, ops, body)
    m = SERGRP_RE.search(name)
    if not m:
        return fails
    cid, mode, u = int(m.group(1)), int(m.group(2)), int(m.group(3))
    cell = cells.get(cid)
    if cell is None:
        return fails + [f"{name}: no cell for cid {cid}"]
    want_ops = u * (4 if mode else 1)
    if len(ops) != want_ops:
        fails.append(f"{name}: {len(ops)} GMMA ops != {want_ops}")
    depbars = [l for l in body if "DEPBAR" in l]
    if len(depbars) != u or any(not DEPBAR0_RE.search(l)
                                for l in depbars):
        fails.append(f"{name}: DEPBAR.LE gsb0,0x0 count "
                     f"{len(depbars)} != {u}")
    for l in body:
        if re.match(r"^\s*@?!?U?P?T?\d*\s*HGMMA.*RZ,", l) or                 ("GMMA" in l and "RZ," in l and "!UPT" in l):
            fails.append(f"{name}: null GMMA present: {l.strip()[:80]}")
            break
    return fails


def finish(name, ops, body, cells):
    if SERGRP_RE.search(name):
        return finish_sergrp(name, ops, body, cells)
    fails0 = between_target_fails(name, ops, body)
    m = CID_RE.search(name)
    if not m:
        return fails0
    cid, u = int(m.group(1)), int(m.group(2))
    cell = cells.get(cid)
    if cell is None:
        return [f"{name}: no cell for cid {cid}"]
    cls, n, k = shapes(cell)
    fails = fails0
    want_ops = 16 if u > 16 else u   # cad U>16 = runtime loop of 16 ops
    if len(ops) != want_ops:
        fails.append(f"{name}: {len(ops)} GMMA ops != {want_ops}")
    else:
        idxs = [body.index(l) for _, l in ops]
        for i in range(len(idxs) - 1):
            for l in body[idxs[i] + 1:idxs[i + 1]]:
                if "DEPBAR" in l or "ARRIVE" in l or re.search(
                        r"\bBRA\b|\bBRA\.U\.ANY\b|\bELECT\b|\bWARPSYNC\b", l):
                    fails.append(
                        f"{name}: DEPBAR/ARRIVE/BRA/ELECT/WARPSYNC between ops: "
                        f"{l.strip()[:80]}")
                    break
            else:
                continue
            break
    for (om, line) in ops:
        if cell.get("sp") and ".SP" not in line:
            fails.append(f"{name}: sparse cell op missing .SP: {line.strip()[:80]}")
            break
        if (om.group(1), int(om.group(4)), int(om.group(5))) != (cls, n, k):
            fails.append(f"{name}: unexpected op shape {om.group(0)} "
                         f"(want {cls}.64x{n}x{k})")
            break
        if cell["src"] == "rs" and "gdesc" in line.split(",", 2)[1]:
            fails.append(f"{name}: rs op has gdesc A operand: {line.strip()}")
            break
    # t_s..first op: only allowed opcodes
    first_op_idx = next(
        (i for i, l in enumerate(body) if OP_RE.search(l)), None)
    clock_idx = next(
        (i for i, l in enumerate(body) if "SR_CLOCKLO" in l or
         "CS2R" in l and "RZ" in l), None)
    if first_op_idx is not None and clock_idx is not None:
        for l in body[clock_idx + 1:first_op_idx]:
            t = l.split(";")[0].strip()
            if not t or t.startswith(("/*", "//")):
                continue
            if not ALLOWED.search(t):
                fails.append(
                    f"{name}: non-arrive op before first GMMA: {t}")
                break
    return fails


if __name__ == "__main__":
    rc = check(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        n7511 = open(sys.argv[3], errors="replace").read().count("C7511")
        if n7511:
            print(f"FAIL ptxas C7511 serialized kernels: {n7511}")
            rc = 1
    sys.exit(rc)
