#!/usr/bin/env python3
"""SASS gate for the ifetch probe (cut down from blackwell-prototype
tools/microbench/sass_check.py check_ifetch + sass_ctrl.py).

Decodes the SR_CLOCKLO window of every timed kernel in `cuobjdump -sass`
output and asserts the protocol shape:
  k_line<N>:      one clock window, exactly N add-class instructions, no
                  memory instructions, no BRA inside, zero spills.
  k_mix<N, W>:    one clock window containing W bodies of ~N add-class
                  instructions at W disjoint address ranges.
  k_jump<M,PERM>: exactly M BRA + 7*M add-class inside the window, every
                  BRA target on a single 128 B grid, block starts exactly
                  128 B apart, exactly one exit target just past the last
                  block; zero spills.
Exit 0 when every kernel was classified (failed kernels are ok=false),
2 on parse failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- sass_ctrl
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
    def body(self) -> str:
        return re.sub(r"^@!?U?P[T0-9]+\s+", "", self.text).strip()

    @property
    def opcode(self) -> str:
        return self.body.split()[0] if self.body else ""

    @property
    def root(self) -> str:
        return self.opcode.split(".")[0]


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
            pending = Inst(int(m.group(1), 16), m.group(2).strip(),
                           int(m.group(3), 16))
            cur.insts.append(pending)
            continue
        m = _HIGH.match(line)
        if m and pending is not None and pending.high == 0:
            pending.high = int(m.group(1), 16)
            pending = None
    return kernels


def clock_windows(k: Kernel) -> list[list[Inst]]:
    idx = [i for i, ins in enumerate(k.insts) if "SR_CLOCKLO" in ins.text]
    return [k.insts[a + 1:b] for a, b in zip(idx, idx[1:])]


def no_spill(w) -> bool:
    return not any(i.root in ("LDL", "STL") for i in w)


# ---------------------------------------------------------------- check_ifetch
# ptxas spells the probe's add.u32 as IADD3, IMAD.IADD or VIADD at its own
# discretion; all are single-issue integer adds, counted as one class.
_IFETCH_MEM_ROOTS = ("LD", "ST", "LDG", "STG", "LDS", "STS")


def _is_iadd3_class(i: Inst) -> bool:
    return i.root in ("IADD3", "VIADD") or i.opcode == "IMAD.IADD"


def _ifetch_bra_target(i: Inst):
    last = i.body.split()[-1]
    return int(last, 16) if last.startswith("0x") else None


def check_ifetch(name: str, k: Kernel):
    m = re.match(r"_Z6k_lineILi(\d+)EEvPmi$", name)
    if m:
        n = int(m.group(1))
        res = {"key": f"ifetch/line/{n}", "u": n}
        ws = clock_windows(k)
        if len(ws) != 1:
            return res | {"ok": False, "why": [f"{len(ws)} clock windows, want 1"]}
        w = ws[0]
        why = []
        n_iadd = sum(1 for i in w if _is_iadd3_class(i))
        if n_iadd != n:
            why.append(f"{n_iadd} IADD3 in window, want {n}")
        mem = [i.text for i in w if i.root in _IFETCH_MEM_ROOTS]
        if mem:
            why.append(f"memory instructions in window: {mem}")
        bra = [i.text for i in w if i.root == "BRA"]
        if bra:
            why.append(f"{len(bra)} BRA inside window")
        if not no_spill(w):
            why.append("local memory in window")
        return res | {"ok": not why, "why": why, "n_window": len(w)}
    m = re.match(r"_Z5k_mixILi(\d+)ELi(\d+)EEvPmi$", name)
    if m:
        # One clock window per pass (t0 before the warp-id dispatch, t1
        # after): the W bodies live inside it as W long add-class runs at
        # disjoint addresses, separated by the dispatch branches.
        n = int(m.group(1))
        w_cnt = int(m.group(2))
        res = {"key": f"ifetch/mix/{n}/w{w_cnt}", "u": n}
        ws = clock_windows(k)
        why = []
        if len(ws) != 1:
            return res | {"ok": False, "why": [f"{len(ws)} clock windows, want 1"]}
        w = ws[0]
        total = sum(1 for i in w if _is_iadd3_class(i))
        if total != w_cnt * n:
            why.append(f"{total} IADD3 in window, want {w_cnt * n}")
        runs, cur = [], []
        for i in w:
            if _is_iadd3_class(i):
                cur.append(i)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        big = sorted(runs, key=len, reverse=True)[:w_cnt]
        if len(big) != w_cnt or any(len(r) < n - 16 for r in big):
            why.append(f"body runs {[len(r) for r in big]}, want {w_cnt} runs of ~{n}")
        ranges = [(r[0].addr, r[-1].addr) for r in big]
        for a in range(len(ranges)):
            for b in range(a + 1, len(ranges)):
                if ranges[a][0] <= ranges[b][1] and ranges[b][0] <= ranges[a][1]:
                    why.append(f"bodies {a} and {b} overlap")
        if any(i.root in _IFETCH_MEM_ROOTS for i in w):
            why.append("memory instructions in window")
        if not no_spill(w):
            why.append("local memory in window")
        res["body_runs"] = [len(r) for r in big]
        return res | {"ok": not why, "why": why}
    m = re.match(r"_Z6k_jumpILi(\d+)ELi(\d)EEvPmi$", name)
    if not m:
        return None
    mm, perm = int(m.group(1)), int(m.group(2))
    res = {"key": f"ifetch/jump/{mm}/p{perm}", "u": mm}
    ws = clock_windows(k)
    if len(ws) != 1:
        return res | {"ok": False, "why": [f"{len(ws)} clock windows, want 1"]}
    w = ws[0]
    why = []
    n_bra = sum(1 for i in w if i.root == "BRA")
    if n_bra != mm:
        why.append(f"{n_bra} BRA in window, want {mm}")
    n_iadd = sum(1 for i in w if _is_iadd3_class(i))
    if n_iadd != 7 * mm:
        why.append(f"{n_iadd} IADD3 in window, want {7 * mm}")
    # chain blocks sit on a 128 B grid: every BRA target is a block start
    # (or the exit just past the last block); the entry block is the one
    # grid position with an add-class instruction and no incoming BRA.
    bras = [i for i in w if i.root == "BRA"]
    targets = [_ifetch_bra_target(i) for i in bras]
    starts = []
    diffs = []
    if any(t is None for t in targets):
        why.append("BRA with unparseable target")
    offs = {t % 128 for t in targets if t is not None}
    if len(offs) != 1:
        why.append(f"BRA targets not on a single 128B grid: {sorted(offs)}")
    else:
        grid = offs.pop()
        start_set = {a.addr for a in w
                     if a.addr % 128 == grid and _is_iadd3_class(a)}
        starts = sorted(start_set)
        if len(starts) != mm:
            why.append(f"{len(starts)} block starts on the grid, want {mm}")
        diffs = [b - a for a, b in zip(starts, starts[1:])]
        if any(d != 128 for d in diffs):
            why.append(f"block start address deltas {diffs}, want all 128")
        by_addr = {i.addr: i for i in w}
        bad_shape = []
        for s in starts:
            blk = [by_addr.get(s + 16 * j) for j in range(8)]
            if (any(i is None for i in blk) or
                    sum(1 for i in blk if _is_iadd3_class(i)) != 7 or
                    blk[6].root != "BRA" or blk[7].root == "BRA"):
                bad_shape.append(hex(s))
        if bad_shape:
            why.append(f"blocks not 6xadd+BRA+add at {bad_shape}")
        exits = {t for t in targets if t not in start_set}
        if len(exits) != 1:
            why.append(f"{len(exits)} non-block-start BRA targets, want 1 (exit)")
        elif starts and exits.pop() != starts[-1] + 128:
            why.append("exit BRA target is not just past the last block")
    if not no_spill(w):
        why.append("local memory in window")
    return res | {"ok": not why, "why": why,
                  "block_starts": [hex(s) for s in starts],
                  "block_deltas": diffs, "n_window": len(w)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--ptxas")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = Path(a.sass).read_text()
    kernels = parse(text)
    out = {"schema": "cyclescope.sass_check_ifetch.v1", "sass": a.sass,
           "kernels": {}}
    if not kernels:
        out["error"] = "no kernels parsed"
        Path(a.out).write_text(json.dumps(out, indent=1))
        return 2
    spills = {}
    if a.ptxas and Path(a.ptxas).exists():
        cur = None
        for line in Path(a.ptxas).read_text().splitlines():
            m = re.search(r"Compiling entry function '(\S+)'", line)
            if m:
                cur = m.group(1)
            m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if m and cur:
                spills[cur] = int(m.group(1)) + int(m.group(2))
    unclassified = []
    for name, k in kernels.items():
        r = check_ifetch(name, k)
        if r is None:
            unclassified.append(name)
            continue
        r["kernel"] = name
        r["spill_bytes"] = spills.get(name)
        clk = [i for i, ins in enumerate(k.insts) if "SR_CLOCKLO" in ins.text]
        r["clock_pcs"] = [hex(k.insts[i].addr) for i in clk]
        out["kernels"][r["key"]] = r
    out["unclassified"] = unclassified
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = sum(1 for r in out["kernels"].values() if not r["ok"])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check_ifetch: {out['n_ok']} ok, {out['n_fail']} fail, "
          f"{len(unclassified)} unclassified")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"])
    return 0 if out["kernels"] else 2


if __name__ == "__main__":
    sys.exit(main())
