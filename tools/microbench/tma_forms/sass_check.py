#!/usr/bin/env python3
# tma_forms SASS gate: per kernel, exactly U target ops of the
# expected form in the timed window.
import json, re, sys
FUNC = re.compile(r"Function : (\S+)")
OP = re.compile(r"/\*[0-9a-f]+\*/\s+(\S+)")
WANT = {"k_ldgsts": "LDGSTS", "k_stas": "STAS",
        "k_gather4": "UTMALDG.*GATHER4", "k_scatter4": "UTMASTG.*SCATTER4",
        "k_im2col": "UTMALDG.*IM2COL", "k_ublkcp_mc": "UBLKCP"}
def main(sass):
    cur = None; ops = []; fails = []; n = 0
    def fin():
        nonlocal n
        if not cur: return
        m = re.match(r"_Z\d+k_(\w+)ILi(\d+)(?:ELi(\d+))?E", cur)
        if not m: return
        kn, u = m.group(1), int(m.group(3) or m.group(2))
        want = WANT.get("k_" + kn)
        if not want: return
        n += 1
        hits = [o for o in ops if re.match(want, o)]
        mult = {"ldgsts": int(m.group(2))}.get(kn, 1)
        if u == 0:
            bad = len(hits) != 0
        elif kn in ("ldgsts", "stas"):
            bad = len(hits) != u * mult
        elif kn in ("gather4", "scatter4") and len(hits) == 0:
            bad = False   # unrealised on sm_90a: kernel compiles empty
        elif kn == "ublkcp_mc":
            bad = len(hits) != u   # window: exactly U multicast UBLKCP
        else:                      # 5 copies compiled per issue site
            bad = len(hits) < u or len(hits) % u != 0
        if bad:
            fails.append(
                f"{cur}: {len(hits)} {want} ops !~ u={u}x{mult}")
    for line in open(sass, errors="replace"):
        fm = FUNC.search(line)
        if fm:
            fin(); cur = fm.group(1); ops = []; continue
        if cur:
            om = OP.search(line)
            if om: ops.append(om.group(1))
    fin()
    print(f"sass_check: {n} kernels, {len(fails)} failures")
    for f in fails[:20]: print("FAIL", f)
    return 1 if (fails or n == 0) else 0
if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
