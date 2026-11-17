#!/usr/bin/env python3
"""SASS gate for the tensor_pipe_forms umma_factors probes.

Per k_factor<KIND, CG, ASRC, MODE, U, M_ROWS, WS, REUSE, SPARSE, ACC, ...>
kernel instantiation in the dumped SASS, over the CS2R window that holds
the measured ops:

  * exactly U UTC{H,Q,O}MMA of the kind's class (HMMA f16 / tf32,
    QMMA f8f6f4 / mxf8f6f4, OMMA mxf4 / mxf4nvf4), counted ops only --
    a predicated-off null op (!UPT + RZ) is the compiler's empty-group
    marker, not a measured op;
  * .2CTA iff CG = 2, .WS iff WS (opcode modifiers as printed by
    cuobjdump); sparse ops print no .SP token on this toolchain, so the
    printed-modifier set is recorded but the SP check cannot fail -- the
    sparse identity of a SPARSE form is established by the probe's D
    check instead (d_expect = sign x K / 2 for sparse, sign x K for
    dense); recorded as sparse_check per kernel (addendum 2026-11-05);
  * only kernels that correspond to a launched case are gated: ELECTED
    must be 1 (the probes' if/else instantiates both forms; ELECTED = 0
    instantiations are dead code and are skipped) and (CG, M_ROWS, WS,
    SPARSE) must match a candidate form of the binary's kind (dead
    template instantiations like mx-kind WS variants are skipped);
  * between the first and the last UMMA no BRA, no ELECT and no
    WARPSYNC (the wave-serialisation loop the ELECTED form avoids);
  * o_s = control-word stall sum from the last SR_CLOCKLO before the
    first UMMA (or, U = 0, before the first UTCBAR) to that target --
    tensor_pipe v2 o_s_and_gate.

Usage: sass_check.py --sass DIR --out checks.json [--bin NAME ...]
Exit nonzero and print if zero kernels matched.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import sass_ctrl as sc

KIND_NAME = {0: "f16", 1: "tf32", 2: "f8f6f4", 3: "mxf8f6f4",
             4: "mxf4", 5: "mxf4nvf4"}
KIND_CLASS = {"f16": "UTCHMMA", "tf32": "UTCHMMA", "f8f6f4": "UTCQMMA",
              "mxf8f6f4": "UTCQMMA", "mxf4": "UTCOMMA", "mxf4nvf4": "UTCOMMA"}
MODE_NAME = {0: "xw", 2: "cad"}
MMA = re.compile(r"^UTC[HQO]MMA")

KERNEL_RE = re.compile(
    r"_Z8k_factorILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E"
    r"Lb(\d)ELb(\d)ELb(\d)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)"
    r"ELi(\d+)EEv")


def is_null_op(i: sc.Inst) -> bool:
    # ptxas' empty-commit-group marker: UTCxMMA ... RZ, ... !UPT
    return "!UPT" in i.text and "RZ" in i.text.split(";", 1)[0]


def modifiers(opcode: str) -> set:
    return set(opcode.split(".")[1:])


def clock_reads(ins) -> list:
    return [n for n, x in enumerate(ins) if "SR_CLOCKLO" in x.text]


def check_kernel(name, k: sc.Kernel):
    m = KERNEL_RE.match(name)
    if not m:
        return None
    g = m.groups()
    kind_i, cg, asrc, mode, u, m_rows = (int(g[i]) for i in range(6))
    ws, _reuse, sparse = bool(int(g[6])), bool(int(g[7])), bool(int(g[8]))
    elected = int(g[15])
    kind = KIND_NAME.get(kind_i, f"k{kind_i}")
    res = {"key": name, "kind": kind, "cg": cg, "asrc": asrc,
           "mode": MODE_NAME.get(mode, mode), "u": u, "m_rows": m_rows,
           "ws": ws, "sparse": sparse, "elected": elected,
           "sparse_check": "probe D (d_expect = sign*K/2)" if sparse
           else "dense (no .SP printed by cuobjdump; identity via probe D)"}
    if elected != 1:
        return res | {"ok": True, "skipped": "elected=0 (dead instantiation)",
                      "why": [], "o_s": None, "n_mma": None}
    if form_keys is not None and (cg, m_rows, ws, sparse) not in form_keys.get(kind, set()):
        return res | {"ok": True, "skipped": "no launched form with this "
                      "(cg, m_rows, ws, sparse)", "why": [], "o_s": None,
                      "n_mma": None}
    ins = k.insts
    ops = [i for i, x in enumerate(ins) if MMA.match(x.opcode)
           and not is_null_op(x)]
    bars = [i for i, x in enumerate(ins) if x.opcode.startswith("UTCBAR")]
    why = []
    if len(ops) != u:
        why.append(f"{len(ops)} MMA != U {u}")
    want = KIND_CLASS[kind]
    for i in ops:
        op = ins[i].opcode
        if not op.startswith(want):
            why.append(f"MMA opcode {op} not {want}")
            break
    for i in ops:
        mods = modifiers(ins[i].opcode)
        if (".2CTA" in ins[i].opcode or "2CTA" in mods) != (cg == 2):
            why.append(f"2CTA modifier mismatch: {ins[i].opcode}")
            break
    for i in ops:
        mods = modifiers(ins[i].opcode)
        if ("WS" in mods) != ws:
            why.append(f"WS modifier mismatch: {ins[i].opcode}")
            break
    op_mods = sorted({m for i in ops for m in modifiers(ins[i].opcode)})
    if len(ops) >= 2:
        between = ins[ops[0] + 1:ops[-1]]
        bad = [x.opcode for x in between
               if x.root in ("BRA", "ELECT", "WARPSYNC")]
        if bad:
            why.append(f"wave-serialisation instructions between first and "
                       f"last UMMA: {sorted(set(bad))}")
    # o_s: stalls from the last SR_CLOCKLO before the first UMMA (or, U = 0,
    # the first UTCBAR) to that target -- tensor_pipe v2 o_s_and_gate.
    target = ops[0] if ops else (bars[0] if bars else None)
    o_s = None
    if target is not None:
        clocks = [i for i in clock_reads(ins) if i < target]
        if clocks:
            o_s = sum(x.stall for x in ins[clocks[-1]:target])
        else:
            why.append("no t_s clock read before the target")
    else:
        why.append("no UMMA and no UTCBAR")
    return res | {"ok": not why, "why": why, "o_s": o_s,
                  "n_mma": len(ops), "op_mods": op_mods}


form_keys = None


def load_form_keys(gen_dir):
    """(kind -> {(cta_group, m_rows, ws, sparse)}) of candidate forms."""
    global form_keys
    form_keys = {}
    for p in Path(gen_dir).glob("*.json"):
        for f in json.load(open(p)).get("forms", []):
            if f["run_status"] not in ("candidate", "control"):
                continue
            form_keys.setdefault(f["kind"], set()).add(
                (f["cta_group"], f["M"] // f["cta_group"],
                 f["operation"] in ("ws", "ws_sp"), f["sparse"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", default=None,
                    help="dir with gen/<kind>.json (launched-form filter)")
    ap.add_argument("--bin", nargs="*", default=None,
                    help="restrict to these sass files (default: all *.sass)")
    a = ap.parse_args()
    if a.gen:
        load_form_keys(a.gen)
    sdir = Path(a.sass)
    files = ([sdir / f"{b}.sass" for b in a.bin] if a.bin
             else sorted(sdir.glob("*.sass")))
    kernels = {}
    for p in files:
        if not p.exists():
            continue
        for name, k in sc.parse(p.read_text(errors="replace")).items():
            res = check_kernel(name, k)
            if res is not None:
                kernels[res["key"]] = res
                kernels[res["key"]]["sass"] = p.name
    n = len(kernels)
    gated = [v for v in kernels.values() if "skipped" not in v]
    n_ok = sum(1 for v in gated if v["ok"])
    out = {"umma_factors": {"kernels": kernels,
                            "gate": {"matched": n, "gated": len(gated),
                                     "ok": n_ok, "pass": len(gated) > 0}}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check: {n_ok}/{len(gated)} gated kernels ok "
          f"({n - len(gated)} skipped dead instantiations)")
    if n == 0:
        print("sass_check: FAIL no kernels matched")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
