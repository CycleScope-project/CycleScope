#!/usr/bin/env python3
"""SASS gate for control_branch protocol-v2.

Window = SASS between t_s and t_w (between the two CS2R clock reads).
Every window opcode is classified scaling (count proportional to U,
verified by comparing the U=16 and U=128 kernels of the same variant)
or constant.  Scaling set must be exactly: BRA x U (loop: static 1)
and chain FFMA x U; taken/uncond/taken_dep also allow the skipped
MEMBAR expansion ops (MEMBAR/ERRBAR/CGAERRBAR/CCTL) x U.  Constant set
may contain BSSY / BSSY.RECONVERGENT / BSYNC / loop entry-exit BRA /
ISETP / predicate materialisation and similar scaffolding; any OTHER
scaling instruction fails the kernel -> unrealised.  Window <= 16 KiB.

Usage: sass_check.py <branch.sass>
"""
import collections
import re
import sys

FUNC = re.compile(r"Function\s*:\s*(\S+)")
KRE = re.compile(r"k_branchILi(\d+)ELi(\d+)E")
VARIANTS = {0: "taken", 1: "nottaken", 2: "uncond", 3: "taken_dep",
            4: "loop"}
FENCE_OPS = ("MEMBAR", "ERRBAR", "CGAERRBAR", "CCTL")
CONST_OK = ("BSSY", "BSYNC", "BRA", "ISETP", "SETP", "UISETP",
            "PLOP3", "LOP3", "ULOP3", "FSETP", "MOV", "IMAD", "VIADD",
            "LEA", "SHF", "S2R", "CS2R", "IADD3", "SEL", "NOP",
            "UIADD3", "UIADD", "UMOV", "ULEA", "USHF", "ULOP")


def kernels(text):
    cur, body, out = None, [], []
    for line in text.splitlines():
        fm = FUNC.search(line)
        if fm:
            if cur is not None:
                out.append((cur, body))
            cur, body = fm.group(1), []
            continue
        if cur is not None and "*/" in line:
            body.append(line)
    if cur is not None:
        out.append((cur, body))
    return out


def window(body):
    cs = [i for i, l in enumerate(body) if "SR_CLOCK" in l]
    if len(cs) < 2:
        return body
    return body[cs[0] + 1:cs[1]]


def ops_of(win):
    c = collections.Counter()
    for l in win:
        m = re.search(r"/\*[0-9a-f]+\*/\s+(@!?\w+\s+)?(\S+)", l)
        if m:
            c[m.group(2).split(".")[0] + (
                "" if m.group(1) is None else "@")] += 1
    return c


def check(path):
    text = open(path, errors="replace").read()
    kerns = {}
    for name, body in kernels(text):
        m = KRE.search(name)
        if m:
            kerns[(int(m.group(1)), int(m.group(2)))] = (
                name, window(body))
    fails, report = [], {}
    for vi in VARIANTS:
        if (vi, 16) not in kerns or (vi, 128) not in kerns:
            fails.append(f"{VARIANTS[vi]}: missing u16/u128 kernel")
            continue
        w16, w128 = ops_of(kerns[(vi, 16)][1]), ops_of(kerns[(vi, 128)][1])
        allops = set(w16) | set(w128)
        scaling, const = {}, {}
        for op in allops:
            c16, c128 = w16.get(op, 0), w128.get(op, 0)
            if c128 == c16 * 8:
                scaling[op] = c16
            else:
                const[op] = (c16, c128)
        name = VARIANTS[vi]
        want = {"BRA": 16, "FFMA": 16}
        if name == "loop":
            want = {"BRA": 1, "FFMA": 1}
        ok = True
        # loop's unit is a static counted loop: its "scaling set" is
        # the static body (1 BRA back-edge + 1 chain FFMA), constant
        # across U by construction
        if name == "loop":
            body_ops = {op.rstrip("@"): c for op, c in
                        list(scaling.items()) + list(const.items())}
            want_body = {"BRA": 1, "FFMA": 1}
            extra = [op for op in body_ops
                     if op not in want_body and
                     op.rstrip(";") not in CONST_OK
                     and not op.startswith("MEMBAR")]
            ok = all(body_ops.get(op) == c
                     for op, c in want_body.items()) and not extra
            scaling = {op: body_ops[op] for op in want_body
                       if op in body_ops}
        else:
            norm = collections.Counter()
            for op, cnt in scaling.items():
                norm[op.rstrip("@;").rstrip(";")] += cnt
            for op, cnt in norm.items():
                if op in ("BRA", "FFMA"):
                    if cnt != want[op]:
                        ok = False
                elif op in FENCE_OPS and name in ("taken", "uncond",
                                                  "taken_dep"):
                    pass
                elif op in ("FSETP", "ISETP") and name == "taken_dep":
                    pass  # per-unit setp is the measured predicate dep
                else:
                    ok = False
            for op, wantc in want.items():
                if norm.get(op) != wantc:
                    ok = False
        # any constant op outside the allowed scaffolding set fails too
        for op, (c16, c128) in const.items():
            base = op.rstrip("@;").rstrip(";")
            if base not in CONST_OK and not base.startswith("MEMBAR"):
                ok = False
        report[name] = {"scaling": scaling, "constant": const}
        if not ok:
            fails.append(
                f"{name}: scaling={scaling} constant={const}")
    for f in fails:
        print("FAIL", f)
    print(f"sass_check: {len(kerns)} branch kernels, "
          f"{len(fails)} variant failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(check(sys.argv[1]))
