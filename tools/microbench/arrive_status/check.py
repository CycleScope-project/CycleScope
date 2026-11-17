#!/usr/bin/env python3
"""check.py <sass> <check.json>: per kernel, between the two SR_CLOCK
reads: exactly U ARRIVE-class instructions of the kernel's declared
form; no other memory/sync op and no branch inside the window.
Chain cells: each ARRIVE destination is not RZ and the register appears
as a source of an instruction between it and the next ARRIVE (address
derivation).  Discard cells: previous destination is not used before
the next ARRIVE.  Writes sass-cells.txt next to check.json."""
import json, re, sys
from pathlib import Path

CLK = re.compile(r"CS2R\s+R\d+,\s*SR_CLOCK")
ARRIVE = re.compile(r"\bSYNCS\.ARRIVE")
SYNCS = re.compile(r"\bSYNCS\.")
FORBID = re.compile(r"\b(LDG|LDS|LDL|LDC|STG|STS|STL|LDGSTS|BAR|MEMBAR|"
                    r"DEPBAR|ERRBAR|BSSY|BSYNC|BRA|JMP|EXIT|"
                    r"MBAR|ATOMS|ATOMG|RED|TRYWAIT|POLL)\b")


def regs_of(text):
    return set(re.findall(r"\bR(\d+)\b", text))


def check(kern, u, kind):
    out = {"kernel": kern["name"], "ok": True, "fails": [],
           "arrive_lines": []}

    def need(cond, msg):
        if not cond:
            out["ok"] = False
            out["fails"].append(msg)

    insts = kern["insts"]
    clks = [i for i, s in enumerate(insts) if CLK.search(s)]
    need(len(clks) >= 2 and len(clks) % 2 == 0,
         f"clock reads={len(clks)}")
    if len(clks) < 2:
        return out
    # rep loop may be unrolled: every clock pair is a window
    wins = [(clks[i] + 1, clks[i + 1])
            for i in range(0, len(clks) - 1, 2)]
    good = 0
    for a, b in wins:
        win = insts[a:b]
        arr = [(i, s) for i, s in enumerate(win) if ARRIVE.search(s)]
        if len(arr) != u:
            continue
        ok = True
        for i, s in enumerate(win):
            if ARRIVE.search(s):
                continue
            if FORBID.search(s):
                ok = False
        if not ok:
            continue
        dsts = []
        for _, s in arr:
            m = re.search(r"SYNCS\.ARRIVE[\w.]*\s+(\S+?),\s*\[", s)
            dsts.append(m.group(1) if m else None)
        dep_ok = True
        for j in range(len(arr) - 1):
            dst = dsts[j]
            between = win[arr[j][0] + 1:arr[j + 1][0]]
            srcs = set()
            for s in between:
                srcs |= regs_of(s.split(";")[0])
            dnum = dst[1:] if dst and dst[0] in "RU" else dst
            used = dst not in (None, "RZ") and dnum in srcs
            if kind == "chain" and not used:
                dep_ok = False
            if kind != "chain" and used:
                dep_ok = False
        if dep_ok:
            good += 1
            if not out["arrive_lines"]:
                out["arrive_lines"] = [s.split(";")[0].strip()
                                       for _, s in arr]
                out["dests"] = dsts
    need(good >= 1,
         f"windows={len(wins)} good={good}")
    return out


def check2(kern, u):
    """v2 discard cells: between the two clock reads, exactly U
    SYNCS-class instructions that all share one mnemonic, destination
    RZ when an explicit destination operand exists, nothing else."""
    out = {"kernel": kern["name"], "ok": True, "fails": [],
           "arrive_lines": []}

    def need(cond, msg):
        if not cond:
            out["ok"] = False
            out["fails"].append(msg)

    insts = kern["insts"]
    clks = [i for i, s in enumerate(insts) if CLK.search(s)]
    need(len(clks) >= 2 and len(clks) % 2 == 0,
         f"clock reads={len(clks)}")
    if len(clks) < 2:
        return out
    wins = [(clks[i] + 1, clks[i + 1])
            for i in range(0, len(clks) - 1, 2)]
    good = 0
    for a, b in wins:
        win = insts[a:b]
        syn = [(i, s) for i, s in enumerate(win) if SYNCS.search(s)]
        if len(syn) != u:
            continue
        mnems = {s.split(";")[0].split()[1].strip() for _, s in syn}
        ok = len(mnems) == 1
        for i, s in enumerate(win):
            if SYNCS.search(s):
                continue
            if FORBID.search(s):
                ok = False
        if not ok:
            continue
        # destination must be RZ when the form has an explicit one;
        # the last SYNCS may keep its state register to pin t1
        rz_ok = True
        for n, (_, s) in enumerate(syn):
            m = re.search(r"\bSYNCS\.\S+\s+(\S+?),\s*\[", s)
            if m and m.group(1) != "RZ" and n != len(syn) - 1:
                rz_ok = False
        if not rz_ok:
            continue
        good += 1
        if not out["arrive_lines"]:
            out["arrive_lines"] = [s.split(";")[0].strip()
                                   for _, s in syn]
            out["mnemonic"] = sorted(mnems)[0]
    need(good >= 1, f"windows={len(wins)} good={good}")
    return out


def main():
    sass = Path(sys.argv[1]).read_text()
    mode = sys.argv[3] if len(sys.argv) > 3 else "v1"
    kerns = []
    cur = None
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = {"name": m.group(1), "insts": []}
            kerns.append(cur)
        elif cur is not None and re.match(r"\s*/\*[0-9a-f]+\*/", line):
            cur["insts"].append(line)
    results = []
    cells_txt = []
    f2name = ["arrive", "arrive_cnt", "expect_tx", "expect_tx_only",
              "cl_arrive", "cl_arrive_expect_tx"]
    sass_map = {}
    for kern in kerns:
        m = re.search(r"k_arrILi(\d+)ELi(\d+)ELi(\d+)E", kern["name"])
        m2 = re.search(r"k_arr2ILi(\d+)ELi(\d+)E", kern["name"])
        if mode == "v2":
            if not m2:
                continue
            u, f2 = int(m2.group(1)), int(m2.group(2))
            r = check2(kern, u)
            r["u"], r["form"] = u, f2name[f2]
            sass_map[f"{f2name[f2]}.u{u}"] = r.get("mnemonic") or (
                r["arrive_lines"][0] if r["arrive_lines"] else None)
        else:
            if not m:
                continue
            u, form, kind = (int(m.group(1)), int(m.group(2)),
                             int(m.group(3)))
            kname = ["chain", "discard", "phase", "load_same",
                     "load_own"]
            r = check(kern, u, "chain" if kind != 1 else "discard")
            r["u"], r["form"], r["kind"] = u, form, kname[kind]
        results.append(r)
        cells_txt.append(f'== {kern["name"]} ok={r["ok"]}')
        cells_txt += ["  " + s for s in r["arrive_lines"]]
    bad = [r for r in results if not r["ok"]]
    want = 30 if mode == "v2" else 36
    passed = len(results) == want and not bad
    if mode == "v2":
        Path(str(sys.argv[2]) + ".sass-map.json").write_text(
            json.dumps(sass_map, indent=1))
    Path(sys.argv[2]).write_text(json.dumps(
        {"passed": passed, "kernels": results}, indent=1))
    Path(sys.argv[2]).with_name("sass-cells.txt").write_text(
        "\n".join(cells_txt) + "\n")
    print(f"arrive_status check: {len(results)} kernels, "
          f"passed={passed}")
    for r in bad:
        print(r["kernel"], r["fails"][:6])
    return 0 if passed else 1


sys.exit(main())
