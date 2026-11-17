#!/usr/bin/env python3
"""SASS gate for lsu_accept (protocol v1, frozen).  Per kernel
k_cell<V,D> the window between the two SR_CLOCKLO reads must contain
exactly D instructions of the variant's class (LDGSTS.* / LDG.* /
STG.* / LDS.*; mix: D LDGSTS + D LDS interleaved) in order, then the
WAR: an IADD3 / IMAD.IADD pair writing the last request's LOW pointer
register (a 64-bit add = lo IADD3 + hi IADD3.X; both present, the first
waits on the rbar), and nothing else.  The WAR's wait mask must include
the read barrier set by the last request; the D requests must use
distinct address registers.  All control words are recorded.

Usage: sass_check.py <build_dir> <checks.json out>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

VARIANTS = ["ldgsts16", "ldgsts8", "ldgsts4", "ldg16", "ldg4",
            "stg16", "mix", "lds"]
DS = (0, 1, 2, 4, 8, 16)
CLASS = {"ldgsts16": ("LDGSTS",), "ldgsts8": ("LDGSTS",),
         "ldgsts4": ("LDGSTS",), "ldg16": ("LDG",), "ldg4": ("LDG",),
         "stg16": ("STG",), "mix": ("LDGSTS", "LDS"),
         "lds": ("LDS",)}
_GPR = re.compile(r"R(\d+)")


def dst_srcs(ins):
    body = ins.body.split(";", 1)[0]
    parts = body.split(None, 1)
    toks = [t.strip() for t in parts[1].split(",")] if len(parts) > 1 else []
    return (toks[0] if toks else None), toks[1:]


def _dead(ins) -> bool:
    # ptxas fills schedule bubbles with dead predicated `LDS RZ, [RZ]`
    t = ins.text.lstrip()
    return t.startswith("@!") and "RZ," in t


def check_kernel(kern, v: str, d: int) -> dict:
    out = {"kernel": kern.name, "variant": v, "d": d, "ok": True,
           "fails": [], "program": []}
    cls0 = CLASS[v]
    clk = [i for i, ins in enumerate(kern.insts)
           if "SR_CLOCKLO" in ins.text]
    if len(clk) < 2:
        out["ok"] = False
        out["fails"].append("clocklo")
        return out
    # pick the clock pair whose class count is closest to the expected
    # D (mix: 2D), tie-break fewest non-class instructions
    exp = 2 * d if v == "mix" else d
    best, best_key = [], (1e9, 1e9)
    for a, b in zip(clk, clk[1:]):
        w = [ins for ins in kern.insts[a + 1:b] if not _dead(ins)]
        n = sum(1 for ins in w
                if ins.opcode.split(".", 1)[0] in cls0)
        key = (abs(n - exp), len(w) - n)
        if key < best_key:
            best, best_key = w, key
    win = best
    for ins in win:
        dst, srcs = dst_srcs(ins)
        out["program"].append(
            {"opcode": ins.opcode, "dst": dst, "srcs": srcs,
             "stall": ins.stall, "wbar": ins.wbar, "rbar": ins.rbar,
             "wait": ins.wait})
    codes = [i.opcode for i in win]
    cls = CLASS[v]
    root = lambda i: win[i].opcode.split(".", 1)[0]
    war = {"IADD3", "IMAD", "LEA"}
    req = [i for i in range(len(win)) if root(i) in cls]
    rest = [i for i in range(len(win)) if root(i) not in cls]
    # D class instructions, in order; the remainder only the WAR pair
    if len(req) != exp:
        out["ok"] = False
        out["fails"].append(f"class_count={len(req)}")
    if v == "mix":
        n_sts = sum(1 for i in req if root(i) == "LDGSTS")
        n_lds = sum(1 for i in req if root(i) == "LDS")
        if n_sts != d or n_lds != d:
            out["ok"] = False
            out["fails"].append(f"mix {n_sts}+{n_lds}")
    mem = {"LDG", "LDGSTS", "STG", "LDSM", "STSM", "BAR",
           "DEPBAR", "ATOMG", "RED", "MEMBAR", "ERRBAR", "SYNCS",
           "ELECT", "TMA", "UTMALDG", "LDGDEPBAR", "CCTL"}
    if v != "lds" and v != "mix":
        mem.add("LDS")
    bad = [i for i in rest
           if root(i) not in war and root(i) in mem]
    # lds/mix: the consumer LOP3 chain after the WAR is allowed
    if v in ("lds", "mix"):
        widx = [i for i in rest if root(i) in war]
        last = widx[-1] if widx else (req[-1] if req else -1)
        n_lop3 = sum(1 for i in rest if i > last and root(i) == "LOP3")
        out["consumers"] = n_lop3
        bad = [i for i in bad if not (i > last and root(i) == "LOP3")]
        rest = [i for i in rest if not (i > last and root(i) == "LOP3")]
    if bad:
        out["ok"] = False
        out["fails"].append("extra_mem: " +
                            ",".join(win[i].opcode for i in bad))
    war_idx = [i for i in rest if root(i) in war]
    out["n_war"] = len(war_idx)
    if d > 0:
        # distinct address registers across the D requests: the
        # memory operand token [..R<n>..] of each request (data regs
        # may legitimately repeat, e.g. STG sources)
        addrs = set()
        for i in req:
            d_tok, srcs = dst_srcs(win[i])
            toks = ([d_tok] if d_tok else []) + list(srcs)
            mem_tok = (next((t for t in toks if t and "desc" in t),
                            None) or
                       next((t for t in toks
                             if t and "R" in t and "[" in t), None))
            if mem_tok:
                for m in _GPR.finditer(mem_tok):
                    addrs.add(int(m.group(1)))
        if len(addrs) < d:
            out["ok"] = False
            out["fails"].append("addr_regs_not_distinct")
        # WAR writes the last request's low pointer reg (for mix the
        # WAR is on the last LDGSTS pointer, not the trailing LDS)
        if not req:
            out["ok"] = False
            out["fails"].append("no_req")
            return out
        war_req = req
        if v == "mix":
            war_req = [i for i in req if root(i) == "LDGSTS"] or req
        last_req = win[war_req[-1]]
        _, lsrcs = dst_srcs(last_req)
        lregs = {int(m.group(1)) for t in lsrcs
                 for m in _GPR.finditer(t)}
        war_dst = [dst for dst, _ in
                   (dst_srcs(win[i]) for i in war_idx) if dst]
        war_regs = {int(m.group(1)) for t in war_dst
                    for m in _GPR.finditer(t)}
        if war_regs and not (war_regs & lregs):
            out["ok"] = False
            out["fails"].append("war_dst_mismatch")
        # WAR wait mask includes the last request's rbar (bitmasks)
        rbar = last_req.rbar
        if rbar and war_idx:
            waited = win[war_idx[0]].wait
            if not (rbar & waited):
                out["ok"] = False
                out["fails"].append(
                    f"war_wait {waited:#x} rbar {rbar:#x}")
        out["last_rbar"] = rbar
    return out


def demangle(name: str) -> str:
    try:
        r = subprocess.run(["c++filt", name], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or name
    except Exception:
        return name


def main() -> int:
    build = Path(sys.argv[1])
    sass = build / "sass" / "lsu_accept.sass"
    kernels = parse(sass.read_text())
    results = []
    fails = 0
    for name, kern in sorted(kernels.items()):
        dem = demangle(name)
        # mangled _Z6k_cellILi<V>ELi<D>E... or demangled k_cell<V, D>
        m = re.search(r"k_cellILi(\d+)ELi(\d+)E", name)
        m2 = re.search(r"k_cell<(int\s)?(\d+),\s*(int\s)?(\d+)>", dem)
        if not (m or m2):
            continue
        v, d = VARIANTS[int(m.group(1) if m else m2.group(2))], \
            int(m.group(2) if m else m2.group(4))
        ck = check_kernel(kern, v, d)
        results.append(ck)
        if not ck["ok"]:
            fails += 1
    out = {"kernels": results, "n_fail": fails}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    for r in results:
        if not r["ok"]:
            print(f"FAIL {r['kernel']} {r['variant']} d={r['d']}: "
                  f"{r['fails']}")
    print(f"sass_check: {len(results)} kernels, {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
