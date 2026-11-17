#!/usr/bin/env python3
"""SASS gate for store_hold (protocol v1): per kernel k_hold<GEO,X,K,W>
exactly 3 SR_CLOCKLO reads; between reads 1 and 2: exactly K+1 STG of the
geometry's width and nothing else; between reads 2 and 3: exactly the one
declared probe instruction (0 for none); no load / branch in the window.
For overwrite cells the X instruction's destination register must equal
the register the (test / first prior) STG used and must not be among X's
sources; for read cells the register must be among X's sources.  Dumps
the test STG and X line of every cell to <sass-cells.txt>.

Usage: check.py <sass> <checks.json out> <sass-cells.txt out>
"""
import json, re, sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa: E402

STG_OP = {0: "STG.E.128", 1: "STG.E"}
XNAME = ["none", "ovw_data", "ovw_addr", "ovw_prior_data",
         "ovw_prior_addr", "read_data", "read_addr"]
MEMBR = ("LDG", "LDS", "LD.", "LDL", "ATOM", "RED", "MEMBAR", "BAR",
         "ERRBAR", "LDGSTS", "UTMA", "UBLK", "SYNCS", "CCTL", "FENCE",
         "DEPBAR", "EXIT", "BRA", "BRX", "BRXU", "JMP", "JMX", "CALL",
         "RET", "SSY", "BSSY", "BSYNC", "WARPSYNC")


def regs(text):
    return [int(m) for m in re.findall(r"R(\d+)", text)]


def stg_regs(text):
    """addr lo reg and first data reg of a STG line."""
    m = re.search(r"\[R(\d+)\.64", text)
    addr = int(m.group(1)) if m else None
    tail = text.split("]", 1)[-1]
    ds = regs(tail.split(",", 1)[-1])
    data = ds[0] if ds else None
    return addr, data


def check(kern, g, x, k, w, cells):
    name = f"{XNAME[x]}.{('full16','line32')[g]}.k{k}.x{w}"
    out = {"kernel": kern.name, "variant": name, "ok": True, "fails": []}
    clk = [i for i, ins in enumerate(kern.insts)
           if "SR_CLOCKLO" in ins.text]
    if len(clk) != 3:
        out["ok"] = False
        out["fails"].append(f"clock_reads={len(clk)} (want 3)")
        return out
    win1 = kern.insts[clk[0] + 1:clk[1]]
    win2 = kern.insts[clk[1] + 1:clk[2]]
    want = STG_OP[g]
    stg = [i for i in win1 if i.opcode == want]
    if len(stg) != k + 1:
        out["ok"] = False
        ops = ",".join(i.opcode for i in win1)
        out["fails"].append(
            f"window1 {len(stg)}x {want} (want {k + 1}): {ops}")
    # declared X lowers to 1 SASS inst, or 2 for a 64-bit add pair
    # (IADD3 + IADD3.X); sm_90a/103a VIADD covers 64-bit in one
    nx = 0 if x == 0 else None
    if nx is None:
        if len(win2) not in (1, 2):
            out["ok"] = False
            out["fails"].append(
                f"window2 {len(win2)} insts (want 1-2): " +
                ",".join(i.opcode for i in win2))
        nx = len(win2)
    if x == 0 and len(win2) != 0:
        out["ok"] = False
        out["fails"].append(
            f"window2 {len(win2)} insts (want 0): " +
            ",".join(i.opcode for i in win2))
    for i in list(win1) + list(win2):
        if any(i.opcode.startswith(p) for p in MEMBR):
            out["ok"] = False
            out["fails"].append(f"foreign {i.opcode} @{i.addr:#x}")
        if i.opcode.startswith("STG") and i.opcode != want:
            out["ok"] = False
            out["fails"].append(f"foreign store {i.opcode} @{i.addr:#x}")
    # register-identity checks
    if len(stg) >= 1 and len(win2) == nx and nx >= 1:
        ta, td = stg_regs(stg[-1].text)     # test store = last STG
        pa, pd = stg_regs(stg[0].text)     # prior store #1 = first STG
        xi = win2[0]
        xr = regs(" ".join(i.text for i in win2))
        dest, srcs = (xr[0] if xr else None), xr[1:]
        def need(cond, msg):
            if not cond:
                out["ok"] = False
                out["fails"].append(msg)
        if x == 1:
            need(dest == td, f"ovw_data dest R{dest} != stg data R{td}")
        elif x == 2:
            need(dest == ta, f"ovw_addr dest R{dest} != stg addr R{ta}")
        elif x == 3:
            need(dest == pd, f"ovw_prior_data R{dest} != prior data R{pd}")
        elif x == 4:
            need(dest == pa, f"ovw_prior_addr R{dest} != prior addr R{pa}")
        elif x == 5:
            need(td in srcs and dest != td,
                 f"read_data {xi.text[:60]}")
        elif x == 6:
            need(ta in srcs and dest != ta,
                 f"read_addr {xi.text[:60]}")
        if x in (1, 2, 3, 4) and dest is not None:
            need(dest not in srcs, "overwrite dest among sources")
        cells.append(f"{name}: TEST {stg[-1].text.strip()}")
        cells.append(f"{name}: X    {xi.text.strip()}")
    elif len(stg) >= 1:
        cells.append(f"{name}: TEST {stg[-1].text.strip()}")
    return out


# ------------------------------------------------------------------
# protocol v2 (loop form): kernel k_loop<X, D, G>.  The loop body is the
# span of the innermost backward branch containing STGs.  Inside: exactly
# D + 1 STG of the declared width, D + 1 X operations (addr: add whose
# destination is one of the stores' address registers; data: add whose
# destination is a store data register; none: D + 1 dummy adds that touch
# no store operand), the loop-counter update and one backward branch;
# nothing else, no load.
LXN = ["none", "addr", "data"]
GN = ["full16", "line32"]
ADDOP = ("IADD", "VIADD", "LADD", "IMAD", "LEA", "SHF")
SETP = ("ISETP", "UISETP", "FSETP", "PSETP", "UPSETP")


def rregs(text):
    return [int(m) for m in re.findall(r"(?<![A-Z])R(\d+)", text)]


def loop_stg_regs(text):
    """addr lo reg and first data reg of a STG line (v4 braces aware)."""
    m = re.search(r"\[R(\d+)\.64", text)
    addr = int(m.group(1)) if m else None
    tail = text.split("]", 1)[-1].split(";", 1)[0]
    ds = rregs(tail)
    return addr, ds


def find_body(kern):
    best = None
    for ins in kern.insts:
        if not ins.opcode.startswith("BRA"):
            continue
        m = re.search(r"0x([0-9a-fA-F]+)", ins.text)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt >= ins.addr:
            continue
        body = [j for j in kern.insts if tgt <= j.addr <= ins.addr]
        if any(j.opcode.startswith("STG") for j in body) and (
                best is None or len(body) < len(best)):
            best = body
    return best


def check_loop(kern, x, d, g, cells):
    name = f"loop.{LXN[x]}.{GN[g]}.d{d}"
    out = {"kernel": kern.name, "variant": name, "ok": True, "fails": []}

    def need(cond, msg):
        if not cond:
            out["ok"] = False
            out["fails"].append(msg)

    body = find_body(kern)
    if body is None:
        need(False, "no backward-branch loop containing STG")
        return out
    want = STG_OP[g]
    stg = [i for i in body if i.opcode == want]
    need(len(stg) == d + 1,
         f"body {len(stg)}x {want} (want {d + 1})")
    addr_regs, data_regs = set(), set()
    for i in stg:
        a, dt = loop_stg_regs(i.text)
        if a is not None:
            addr_regs.add(a)
            addr_regs.add(a + 1)
        data_regs.update(dt)
    for i in body:
        cells.append(f"{name}: {i.text.strip()}")
    adds, others = [], []
    for i in body:
        op = i.opcode
        if op == want or op.startswith("BRA") or \
                op.startswith(SETP):
            continue
        # uniform-datapath instructions (UIADD3 UR4, ...) implement the
        # rolled loop counter: loop control, not X and not foreign
        first = i.text.split(" ", 1)[-1].lstrip("@P!0123456789 ")
        if op.startswith("U") or first.startswith(("UR", "UP")):
            continue
        if op.startswith(ADDOP) and ".MOV" not in op:
            adds.append(i)
        else:
            others.append(i)
    for i in others:
        need(False, f"foreign {i.opcode} @{i.addr:#x} {i.text.strip()[:70]}")
    nbra = sum(1 for i in body if i.opcode.startswith("BRA"))
    need(nbra == 1, f"body {nbra} branches (want 1)")
    need(not any(i.opcode.startswith(("LD", "STL", "LDS"))
                 for i in body), "load/local in body")
    op_regs = addr_regs | data_regs
    if x == 1:      # addr: X dest must be a STG address register
        hit = [i for i in adds
               if rregs(i.text) and rregs(i.text)[0] in addr_regs]
        need(len({rregs(i.text)[0] for i in hit}) >= d + 1,
             f"addr X: {len({rregs(i.text)[0] for i in hit})} dests "
             f"in stg addr regs (want {d + 1}): "
             + "; ".join(i.text.strip()[:60] for i in adds[:10]))
    elif x == 2:    # data: X dest must be a STG data register
        hit = [i for i in adds
               if rregs(i.text) and rregs(i.text)[0] in data_regs]
        need(len({rregs(i.text)[0] for i in hit}) >= d + 1,
             f"data X: {len({rregs(i.text)[0] for i in hit})} dests "
             f"in stg data regs (want {d + 1}): "
             + "; ".join(i.text.strip()[:60] for i in adds[:10]))
    else:           # none: no add may write a store operand register
        bad_a = [i for i in adds
                 if rregs(i.text) and rregs(i.text)[0] in op_regs]
        need(not bad_a, "none: add writes store reg " +
             "; ".join(i.text.strip()[:60] for i in bad_a[:6]))
    return out


def main() -> int:
    v2 = "--v2" in sys.argv
    kerns = parse(Path(sys.argv[1]).read_text())
    results, cells = [], []
    for kern in kerns.values():
        m = re.search(r"k_holdILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E",
                      kern.name)
        if m and not v2:
            g, x, k, w = (int(v) for v in m.groups())
            if k == 0 and x in (3, 4):
                continue  # instantiated via kpick but never launched
            results.append(check(kern, g, x, k, w, cells))
        m2 = re.search(r"k_loopILi(\d+)ELi(\d+)ELi(\d+)E", kern.name)
        if m2 and v2:
            x, d, g = (int(v) for v in m2.groups())
            results.append(check_loop(kern, x, d, g, cells))
    bad = [r for r in results if not r["ok"]]
    want = 18 if v2 else 132
    passed = len(results) == want and not bad
    print(f"store_hold check: {len(results)} kernels, passed={passed}")
    for r in bad:
        print(r["kernel"], r["variant"], r["fails"][:8])
    Path(sys.argv[2]).write_text(json.dumps(
        {"passed": passed, "kernels": results}, indent=1))
    Path(sys.argv[3]).write_text("\n".join(cells) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
