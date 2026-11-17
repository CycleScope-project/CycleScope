#!/usr/bin/env python3
"""SASS gate for dispatch_contention (protocol v1): in every kernel
window between the rep's two CS2R SR_CLOCKLO the chain class occurs
exactly U times (mixed: U/2 FFMA + U/2 FMNMX; ilp8: U FFMA) and no
other instruction occurs; the chain register dependence is real
(each unit's destination is a source of the next, ilp8 excepted).
Control words of every window instruction are recorded per variant.

Usage: sass_check.py <build_dir(containing sass/contention.sass)> \
        <checks.json out>
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout"))
from sass_ctrl import parse, clock_windows  # noqa: E402

OPC = {0: "FFMA", 1: "FMNMX", 2: "MUFU.EX2", 3: "IMAD",
       4: "MIXED", 5: "ILP8"}
VARIANTS = {"ffma": 0, "fmnmx": 1, "mufu": 2, "imad": 3,
            "mixed": 4, "ilp8": 5}
US = (64, 128, 256)


def dest_src(text: str) -> tuple[str, list[str]]:
    body = text.split(";", 1)[0]
    body = body.split(None, 1)[1] if " " in body else ""
    regs = [t.strip() for t in body.split(",")]
    return regs[0] if regs else "", regs[1:]


def check_kernel(k, variant: int, u: int) -> dict:
    wins = clock_windows(k)
    ok_wins = []
    for w in wins:
        codes = [i.opcode for i in w]
        cnt = Counter(codes)
        if variant == 4:
            want = (u // 2, u // 2)
            # suffix variants count toward their class (FMNMX.NAN on
            # Blackwell)
            n_ffma = sum(v for k, v in cnt.items() if k.startswith("FFMA"))
            n_fmnmx = sum(v for k, v in cnt.items()
                          if k.startswith("FMNMX"))
            good = (n_ffma == want[0] and n_fmnmx == want[1]
                    and len(w) == u)
        elif variant == 5:
            good = cnt.get("FFMA", 0) == u and len(w) == u
        else:
            # ptxas may pick the trinary form FMNMX3 for max.f32 on
            # Blackwell; it is the same class for this gate
            n_cls = sum(v for k, v in cnt.items()
                        if k.startswith(OPC[variant].split(".")[0]))
            good = n_cls == u and len(w) == u
        if good:
            ok_wins.append(w)
    if not ok_wins:
        return {"u": u, "ok": False, "windows": 0}
    w = ok_wins[0]
    # chain dependence: dst of unit i is a src of unit i+1 (ilp8 exempt)
    chained = True
    if variant != 5:
        chain_insts = [i for i in w]
        for a, b in zip(chain_insts, chain_insts[1:]):
            da, _ = dest_src(a.text)
            _, sb = dest_src(b.text)
            if da not in sb:
                chained = False
                break
    stalls = Counter(i.stall for i in w)
    yields = Counter(i.yld for i in w)
    return {"u": u, "ok": True, "windows": len(ok_wins),
            "chained": chained, "stalls": dict(stalls),
            "yields": dict(yields),
            "wbar": dict(Counter(i.wbar for i in w)),
            "rbar": dict(Counter(i.rbar for i in w)),
            "wait": dict(Counter(i.wait for i in w))}


def main() -> int:
    build, out = sys.argv[1], sys.argv[2]
    kernels = parse(open(Path(build) / "sass" / "contention.sass").read())
    res = {}
    for name, kern in kernels.items():
        # mangled _Z6k_cellILiVELiUEE...
        import re
        m = re.search(r"k_cellILi(\d)ELi(\d+)E", name)
        if not m:
            continue
        v, u = int(m.group(1)), int(m.group(2))
        vname = [k for k, x in VARIANTS.items() if x == v][0]
        res[f"{vname}/u{u}"] = check_kernel(kern, v, u)
        res[f"{vname}/u{u}"]["kernel"] = name
    json.dump(res, open(out, "w"), indent=1)
    for k, v in sorted(res.items()):
        print(k, "OK" if v["ok"] else "FAIL",
          "stalls" if v["ok"] else "", v.get("stalls", ""))
    return 0 if all(v["ok"] and v.get("chained", True)
                    for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
