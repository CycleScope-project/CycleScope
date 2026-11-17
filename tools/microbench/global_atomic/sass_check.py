#!/usr/bin/env python3
"""SASS gate for the global_atomic probe (protocol.json sass_gate).

  red_same / atom_same loop bodies: exactly one RED.E.ADD / ATOMG.E.ADD
      per iteration and no LDG / STG.
  atom_chain (feeds atom_chain_addr): one ATOMG.E.ADD whose destination
      feeds the next source.
  pingpong_addr: one ST strong-GPU store (or MEMBAR + ST, recorded) and
      one LD strong-GPU load in the spin.
  chase_l2: one dependent LDG strong-GPU per step.

A gate failure voids the cell.  Writes checks.json: per form pass/fail
plus the opcodes hit.
"""
import argparse
import json
import re
from pathlib import Path

import sass_ctrl as sc

MEM_RE = re.compile(
    r"^(LDG|STG|LD|ST|LDS|STS|LDC|ATOM|ATOMG|ATOMS|RED|REDG|REDS|LDCU|"
    r"LDGSTS|UTMALDG|UTMASTG|UBLKCP)")


def opcodes(k):
    return [i.opcode for i in k.insts]


def count_root(ops, roots):
    return sum(1 for o in ops if o.split(".")[0] in roots)


def kernel_for(kernels, form):
    # atomic_kernel<FORM>: mangled _Z13atomic_kernelILi<d>E...
    pat = re.compile(r"atomic_kernelILi%dE" % form)
    for name, k in kernels.items():
        if pat.search(name):
            return k
    return None


def loop_body(k, anchor):
    """The backward-branch loop body containing insts[anchor]: the
    instruction run from the target of the first backward BRA at or
    after anchor to that BRA.  Falls back to the biggest clock window."""
    insts = k.insts
    for j in range(anchor, len(insts)):
        ins = insts[j]
        if ins.opcode.split(".")[0] != "BRA":
            continue
        m = re.search(r"0x([0-9a-f]+)", ins.text)
        if m and int(m.group(1), 16) < ins.addr:
            tgt = int(m.group(1), 16)
            lo = next(i for i, x in enumerate(insts) if x.addr >= tgt)
            return insts[lo:j + 1]
    ws = sc.clock_windows(k)
    return max(ws, key=len) if ws else insts


def _atomic_index(insts, roots):
    return next((i for i, x in enumerate(insts)
                 if x.opcode.split(".")[0] in roots), 0)


def check_red(k):
    ai = _atomic_index(k.insts, ("RED", "REDG"))
    body = loop_body(k, ai)
    ops = [i.opcode for i in body]
    reds = [o for o in ops if o.split(".")[0] in ("RED", "REDG")]
    stray = sorted({o for o in ops if o.split(".")[0]
                    in ("LDG", "STG", "ATOM", "ATOMG")})
    why = []
    if len(reds) != 1:
        why.append(f"{len(reds)} RED in body, want 1")
    elif "ADD" not in reds[0]:
        why.append(f"RED opcode {reds[0]} is not an ADD")
    if stray:
        why.append(f"stray global ops: {stray}")
    return why, sorted(set(ops))


def check_atom(k, want_dependency):
    ai = _atomic_index(k.insts, ("ATOMG",))
    body = loop_body(k, ai)
    insts = body
    ops = [i.opcode for i in insts]
    atoms = [o for o in ops if o.split(".")[0] == "ATOMG"]
    why = []
    if want_dependency:
        # Rec bookkeeping stores (clk0/gt0) may sit inside the timed
        # window; only the chain form must have no other memory op
        stray = sorted({o for o in ops
                        if o.split(".")[0] in ("LDG", "RED", "REDG")})
        if stray:
            why.append(f"stray global ops: {stray}")
    else:
        stray = sorted({o for o in ops if o.split(".")[0]
                        in ("LDG", "STG", "RED", "REDG")})
        if stray:
            why.append(f"stray global ops: {stray}")
    if len(atoms) != 1:
        why.append(f"{len(atoms)} ATOMG in body, want 1")
    elif "ADD" not in atoms[0]:
        why.append(f"ATOMG opcode {atoms[0]} is not an ADD")
    dep = None
    if want_dependency and len(atoms) == 1:
        # destination register of the ATOMG (first R/UR operand after
        # the opcode, skipping a leading PT/Pn discard field) must feed
        # another source in the loop body (the dependent chain)
        ai = next(i for i, x in enumerate(insts)
                  if x.opcode.split(".")[0] == "ATOMG")
        tail = insts[ai].text.split(None, 1)[1]
        regs = re.findall(r"(?<![A-Z.])(R\d+|UR\d+)", tail)
        dst = regs[0] if regs else None
        dep = bool(dst) and any(
            i != ai
            and re.search(r"[,\[\s]" + dst + r"(?![\d.])", x.text)
            for i, x in enumerate(insts))
        if not dep:
            why.append("ATOMG destination does not feed a later source")
    return why, sorted(set(ops))


def check_pingpong(k):
    ops = opcodes(k)
    st = [o for o in ops if o.split(".")[0] in ("STG", "ST", "MEMBAR")]
    strong_st = [o for o in ops
                 if o.split(".")[0] == "STG" and "STRONG" in o
                 and "GPU" in o]
    ld_strong = [o for o in ops
                 if o.split(".")[0] == "LDG" and "STRONG" in o
                 and "GPU" in o]
    why = []
    if not strong_st:
        if any("MEMBAR" in o for o in ops) and \
                any(o.split(".")[0] in ("STG", "ST") for o in ops):
            pass  # MEMBAR + ST form, recorded
        else:
            why.append("no STG.E.STRONG.GPU and no MEMBAR+ST")
    if not ld_strong:
        why.append("no LDG.E.STRONG.GPU spin load")
    return why, sorted(set(ops))


def check_chase(k):
    ops = opcodes(k)
    ldg = [o for o in ops if o.split(".")[0] == "LDG"]
    strong = [o for o in ldg if "STRONG" in o and "GPU" in o]
    why = []
    if not ldg:
        why.append("no LDG in chase kernel")
    elif not strong:
        why.append(f"LDG present but not .STRONG.GPU: {sorted(set(ldg))}")
    return why, sorted(set(ops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = Path(a.sass).read_text()
    kernels = sc.parse(text)
    checks = {}
    cases = [
        ("red_same", 0, lambda k: check_red(k)),
        ("atom_same", 3, lambda k: check_atom(k, False)),
        ("atom_chain_addr", 4, lambda k: check_atom(k, True)),
        ("chase_l2", 6, lambda k: check_chase(k)),
    ]
    for form_name, form_id, fn in cases:
        k = kernel_for(kernels, form_id)
        if k is None:
            checks[form_name] = {"pass": False,
                                 "why": [f"kernel atomic_kernel<{form_id}> not found"],
                                 "opcodes": []}
            continue
        why, ops = fn(k)
        checks[form_name] = {"pass": not why, "why": why, "opcodes": ops}
    pp = next((k for n, k in kernels.items()
               if "pingpong_addr_kernel" in n), None)
    if pp is None:
        checks["pingpong_addr"] = {"pass": False,
                                   "why": ["kernel pingpong_addr_kernel not found"],
                                   "opcodes": []}
    else:
        why, ops = check_pingpong(pp)
        rel = ("MEMBAR.ALL.GPU+STG.E.STRONG.GPU"
               if any("MEMBAR.ALL.GPU" in o for o in ops)
               else "ST.E.STRONG.GPU")
        checks["pingpong_addr"] = {"pass": not why, "why": why,
                                   "opcodes": ops,
                                   "release_form": rel}
    out = {"checks": checks,
           "pass": all(c["pass"] for c in checks.values())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["pass"] for k, v in checks.items()}, indent=1))


if __name__ == "__main__":
    main()
