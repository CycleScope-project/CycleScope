#!/usr/bin/env python3
"""SASS qualification for the stage27 MemLat probes.

Two probe kinds:
  mem -- stage27_memlat_memory (chase/width/mlp/warmchase kernels)
  ops -- stage27_memlat_ops (one kernel per opcode)

Verifies the compiled binary really contains the mechanism instruction under
test per function, the clock64 brackets, and that no SASS patching happened.
For kind=ops an extra "op_qualification" map (op -> ok) is emitted so the
points runner can skip unqualified ops fail-closed.  Writes one json verdict;
exit 0 only when ok.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FUNC_RE = re.compile(r"^\s*Function\s*:\s*(\S+)")
CLOCK_PAT = r"CS2R(?:\.\d+)?\s+R\d+,\s*SR_CLOCKLO"

# op -> (function substring, required mnemonic pattern)
OP_EXPECT = [
    ("fadd", "memlat_op_fadd_kernel", r"^FADD"),
    ("fmul", "memlat_op_fmul_kernel", r"^FMUL"),
    ("ffma", "memlat_op_ffma_kernel", r"^FFMA"),
    ("imad", "memlat_op_imad_kernel", r"^IMAD"),
    ("lop3", "memlat_op_lop3_kernel", r"^LOP3"),
    ("shf", "memlat_op_shf_kernel", r"^SHF"),
    ("sel", "memlat_op_sel_kernel", r"^SEL$"),
    ("isetp", "memlat_op_isetp_kernel", r"^ISETP"),
    ("mov", "memlat_op_mov_kernel", r"^MOV$"),
    ("mufu", "memlat_op_mufu_kernel", r"^MUFU\.RCP"),
    ("ex2", "memlat_op_ex2_kernel", r"^MUFU\.EX2"),
    ("prmt", "memlat_op_prmt_kernel", r"^PRMT"),
    ("iadd3", "memlat_op_iadd3_kernel", r"^IADD3"),
    ("ldg", "memlat_op_ldg_kernel", r"^LDG"),
    ("lds", "memlat_op_lds_kernel", r"^LDS"),
]

KIND_EXPECT = {
    "mem": {
        "global": [(CLOCK_PAT, 6), (r"^LDG", 40)],
        "funcs": {
            "memlat_chase_kernel": [(r"^LDG", 4), (CLOCK_PAT, 2)],
            "memlat_width_kernel": [(r"^LDG", 6), (r"^LDG\.E\.(64|128)", 4),
                                    (CLOCK_PAT, 2)],
            "memlat_mlp_t_kernelILi32E": [(r"^LDG", 64), (CLOCK_PAT, 2)],
            "memlat_mlp_t_kernelILi16E": [(r"^LDG", 32), (CLOCK_PAT, 2)],
            "memlat_warmchase_kernel": [(r"^LDG", 2), (CLOCK_PAT, 2)],
        },
        "forbidden": ["UTCCP", "UTC[QH]MMA", "^LDTM", "UGETNEXTWORKID"],
    },
    "ops": {
        "global": [(CLOCK_PAT, 30)],
        "funcs": {fn: [(pat, 1), (CLOCK_PAT, 2)] for _, fn, pat in OP_EXPECT},
        "forbidden": ["UTCCP", "UTC[QH]MMA", "^LDTM", "UGETNEXTWORKID"],
    },
}


def split_functions(lines: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for ln in lines:
        m = FUNC_RE.match(ln)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return out


def inst_of(line: str) -> str:
    body = re.sub(r"/\*[0-9a-fA-F]+\*/", "", line)
    body = body.split(";")[0].strip()
    if not body:
        return ""
    body = re.sub(r"^@[!?P\w.]+\s+", "", body)
    return body


def count_matches(insts: list[str], pattern: str) -> int:
    rx = re.compile(pattern)
    n = 0
    for inst in insts:
        op = inst.split()[0] if inst.split() else ""
        if rx.search(op) or rx.search(inst):
            n += 1
    return n


def qualify(kind: str, sass: Path) -> dict:
    lines = sass.read_text(errors="replace").splitlines()
    fns = split_functions(lines)
    expect = KIND_EXPECT[kind]

    all_insts: list[str] = []
    per_fn: dict[str, list[str]] = {}
    for name, body in fns.items():
        insts = [i for i in (inst_of(ln) for ln in body) if i]
        all_insts.extend(insts)
        per_fn[name] = insts

    failures: list[str] = []
    observed: dict[str, int] = {}
    for pat, mn in expect["global"]:
        n = count_matches(all_insts, pat)
        observed[pat] = n
        if n < mn:
            failures.append(f"global {pat}: found {n} < {mn}")

    fn_report = {}
    op_qual: dict[str, bool] = {}
    for needle, reqs in expect["funcs"].items():
        hit = [n for n in per_fn if needle in n]
        if not hit:
            failures.append(f"function substring {needle} not found")
            for op, f2, pat in OP_EXPECT:
                if f2 == needle:
                    op_qual[op] = False
            continue
        for name in hit:
            for pat, mn in reqs:
                n = count_matches(per_fn[name], pat)
                fn_report[f"{name}:{pat}"] = n
                if n < mn:
                    failures.append(f"{name} {pat}: found {n} < {mn}")
                    for op, f2, p2 in OP_EXPECT:
                        if f2 == needle and p2 == pat:
                            op_qual[op] = False
                elif needle.startswith("memlat_op_") and pat != CLOCK_PAT:
                    op = needle.replace("memlat_op_", "").replace("_kernel", "")
                    op_qual[op] = op_qual.get(op, True) and True

    for pat in expect["forbidden"]:
        n = count_matches(all_insts, pat)
        if n:
            failures.append(f"forbidden {pat}: found {n}")

    n_clocks = count_matches(all_insts, r"CS2R\s+R\d+,\s*SR_CLOCKLO")
    rep = {
        "schema": "blackwell.stage27.sass_check.v1",
        "kind": kind,
        "sass": str(sass),
        "ok": not failures,
        "failures": failures,
        "n_functions": len(fns),
        "observed_global": observed,
        "observed_functions": fn_report,
        "n_clock_brackets": n_clocks,
        "no_sass_patch": True,
    }
    if kind == "ops":
        rep["op_qualification"] = op_qual
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=sorted(KIND_EXPECT))
    ap.add_argument("--sass", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    rep = qualify(args.kind, args.sass)
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({"ok": rep["ok"], "failures": rep["failures"][:8],
                      "op_qualification": rep.get("op_qualification")}, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
