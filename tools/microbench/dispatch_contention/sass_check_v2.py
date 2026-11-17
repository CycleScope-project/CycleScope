#!/usr/bin/env python3
"""SASS gate for dispatch_contention (protocol v2): in every kernel
window between the rep's two CS2R SR_CLOCKLO the variant's class occurs
exactly U times (dep*/ilp*: U FFMA; imad: U IMAD; mixed: U/2 FFMA +
U/2 FMNMX*; mufu.<op>: U MUFU.<OP>) and no other instruction occurs --
a mufu op that ptxas expands to more than one instruction fails the
gate and is unrealised.  The chain dependence is real where the variant
is a chain (everything but ilp*).

For every kernel the full ordered window instruction list INCLUDING
the two bounding CS2R reads is recorded as "program" (opcode, dst,
per-source register/immediate/constant classification, `.reuse` suffix,
control word).  The `.reuse` suffixes must agree with the control-word
reuse bits (bit s = source slot s) else the kernel fails with
"reuse_bits_mismatch".  The U=64/128 lists must be prefixes of the
U=256 list in opcode + operand kind/reuse/pair pattern (register
numbers may differ between kernels); recorded as "prefix_ok".

Usage: sass_check_v2.py <build_dir> <checks.json out> [--v1|--v3]
       (--v1 reads sass/contention.sass, --v3 sass/contention_v3.sass
        with the 14-variant list)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

VARIANTS = ["dep3r", "dep3n", "dep2", "dep1", "ilp8", "ilp8n", "ilp8n2",
            "ilp2", "imad", "mixed", "mufu.ex2", "mufu.rcp", "mufu.rsqrt",
            "mufu.lg2", "mufu.sin", "mufu.tanh"]
# protocol v3: mufu.rcp / mufu.sin dropped, indices 11..13 shift
VARIANTS_V3 = ["dep3r", "dep3n", "dep2", "dep1", "ilp8", "ilp8n", "ilp8n2",
               "ilp2", "imad", "mixed", "mufu.ex2", "mufu.rsqrt",
               "mufu.lg2", "mufu.tanh"]
MUFU_OP = {"ex2": "EX2", "rcp": "RCP", "rsqrt": "RSQ", "lg2": "LG2",
           "sin": "SIN", "tanh": "TANH"}
CHAIN = set(range(16)) - {4, 5, 6, 7}  # ilp* are parallel chains
US = (64, 128, 256)

_GPR = re.compile(r"R(\d+)$")
_UR = re.compile(r"UR(\d+)$")
_PRED = re.compile(r"U?P[T0-9]+$")
_IMM = re.compile(r"[-+]?0[xX][0-9a-fA-F]+$|[-+]?0[fF][0-9a-fA-F]+$|"
                  r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def operand(tok: str) -> dict:
    # operand modifiers (negation / abs bars / bit-not) are not part of
    # the register name
    t = tok.strip("|").lstrip("-~!")
    reuse = False
    pair = False
    # suffixes may appear in either order (R3.64.reuse)
    while True:
        if t.endswith(".reuse"):
            reuse = True
            t = t[:-6]
        elif t.endswith(".64"):
            pair = True
            t = t[:-3]
        else:
            break
    reg, kind = None, "other"
    m = _GPR.match(t)
    if m:
        reg, kind = int(m.group(1)), "gpr"
    else:
        m = _UR.match(t)
        if m:
            reg, kind = int(m.group(1)), "ur"
        elif t in ("RZ", "URZ"):
            kind = "rz"
        elif _PRED.match(t):
            kind = "pred"
        elif t.startswith("c["):
            kind = "const"
        elif _IMM.match(t):
            kind = "imm"
    d = {"text": tok, "reg": reg, "kind": kind, "reuse": reuse}
    if pair:
        d["pair"] = True
    return d


def dst_srcs(ins) -> tuple:
    """(dst token or None, [src dicts]) from the instruction text."""
    body = ins.body.split(";", 1)[0]
    parts = body.split(None, 1)
    toks = [t.strip() for t in parts[1].split(",")] if len(parts) > 1 else []
    if not toks:
        return None, []
    return toks[0], [operand(t) for t in toks[1:]]


def prog_entry(ins) -> dict:
    dst, srcs = dst_srcs(ins)
    return {"addr": ins.addr, "opcode": ins.opcode, "dst": dst,
            "srcs": srcs, "stall": ins.stall, "yield_raw": ins.yld,
            "wbar": ins.wbar, "rbar": ins.rbar, "wait": ins.wait,
            "reuse_bits": (ins.ctrl >> 17) & 0xF}


def clock_spans(kern) -> list:
    """(a, b) index pairs of consecutive SR_CLOCKLO reads."""
    idx = [i for i, ins in enumerate(kern.insts)
           if "SR_CLOCKLO" in ins.text]
    return list(zip(idx, idx[1:]))


def window_ok(win: list, v: int, u: int) -> bool:
    codes = [i.opcode for i in win]
    cnt = Counter(codes)
    if v in range(0, 8):  # dep*, ilp*: pure FFMA window
        return cnt.get("FFMA", 0) == u and len(win) == u
    if v == 8:
        return cnt.get("IMAD", 0) == u and len(win) == u
    if v == 9:
        n_ffma = sum(n for k, n in cnt.items() if k.startswith("FFMA"))
        n_fmnmx = sum(n for k, n in cnt.items() if k.startswith("FMNMX"))
        return n_ffma == u // 2 and n_fmnmx == u // 2 and len(win) == u
    op = MUFU_OP[VARIANTS[v].split(".", 1)[1]]
    return all(c.startswith(f"MUFU.{op}") for c in codes) and len(win) == u


def chained(win: list) -> bool:
    for a, b in zip(win, win[1:]):
        da, _ = dst_srcs(a)
        _, sb = dst_srcs(b)
        base = [s["text"].split(".")[0] for s in sb]
        if da is None or da.split(".")[0] not in base:
            return False
    return True


def check_kernel(kern, v: int, u: int) -> dict:
    good = [s for s in clock_spans(kern)
            if window_ok(kern.insts[s[0] + 1:s[1]], v, u)]
    if not good:
        return {"u": u, "ok": False, "windows": 0}
    a, b = good[0]
    win = kern.insts[a + 1:b]
    prog = kern.insts[a:b + 1]
    res = {"u": u, "ok": True, "windows": len(good),
           "stalls": dict(Counter(i.stall for i in win)),
           "yields": dict(Counter(i.yld for i in win)),
           "wbar": dict(Counter(i.wbar for i in win)),
           "rbar": dict(Counter(i.rbar for i in win)),
           "wait": dict(Counter(i.wait for i in win))}
    if v in CHAIN:
        res["chained"] = chained(win)
    entries = [prog_entry(i) for i in prog]
    res["program"] = entries
    bad = [{"addr": e["addr"],
            "expected": sum(1 << s for s, src in enumerate(e["srcs"])
                            if src["reuse"]),
            "actual": e["reuse_bits"]}
           for e in entries
           if e["reuse_bits"] != sum(1 << s for s, src in
                                     enumerate(e["srcs"])
                                     if src["reuse"])]
    if bad:
        res["ok"] = False
        res["reason"] = "reuse_bits_mismatch"
        res["reuse_mismatches"] = bad
    return res


def prog_form(entry: dict) -> tuple:
    """(opcode, operand kind/reuse/pair pattern) for prefix compare --
    register numbers legitimately differ between kernels, so the form
    compares operand kinds, `.reuse` flags and 64-bit pairing only."""
    dst = operand(entry["dst"]) if entry["dst"] else None
    ops = [entry["opcode"],
           (dst["kind"], dst.get("pair", False)) if dst else None]
    ops += [(s["kind"], s["reuse"], s.get("pair", False))
            for s in entry["srcs"]]
    return tuple(ops)


def main() -> int:
    global VARIANTS, CHAIN
    build, out = sys.argv[1], sys.argv[2]
    flags = set(sys.argv[3:])
    if "--v3" in flags:
        VARIANTS = VARIANTS_V3
    CHAIN = set(range(len(VARIANTS))) - {4, 5, 6, 7}
    name = {"--v1": "contention.sass", "--v3": "contention_v3.sass"}.get(
        next(iter(flags & {"--v1", "--v3"}), ""), "contention_v2.sass")
    sass = Path(build) / "sass" / name
    kernels = parse(sass.read_text())
    res = {}
    for name, kern in kernels.items():
        m = re.search(r"k_cellILi(\d+)ELi(\d+)E", name)
        if not m:
            continue
        v, u = int(m.group(1)), int(m.group(2))
        if v >= len(VARIANTS):
            continue
        key = f"{VARIANTS[v]}/u{u}"
        res[key] = check_kernel(kern, v, u)
        res[key]["kernel"] = name
    # U=64/128 program forms must be prefixes of the U=256 form
    for vname in VARIANTS:
        big = res.get(f"{vname}/u256")
        if not big or "program" not in big:
            continue
        # opening CS2R + units (the closing CS2R has no counterpart in
        # a prefix of the longer program)
        form256 = [prog_form(e) for e in big["program"][:-1]]
        ok = True
        for u in (64, 128):
            small = res.get(f"{vname}/u{u}")
            if not small or "program" not in small:
                continue
            form = [prog_form(e) for e in small["program"][:-1]]
            if form256[:len(form)] != form:
                ok = False
        big["prefix_ok"] = ok
    json.dump(res, open(out, "w"), indent=1)
    for k, v in sorted(res.items()):
        print(k, "OK" if v["ok"] else "FAIL",
              "chained" if v.get("chained") else "",
              "" if v["ok"] else v.get("reason", ""))
    return 0 if all(v["ok"] and v.get("chained", True)
                    for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
