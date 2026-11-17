#!/usr/bin/env python3
"""sass gate for mufu_rcp: window = U MUFU.<OP> + U FMUL alternating.
Usage: sass_check.py <build_dir> <out_json>
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "crossings" / "trywait_timeout"))
from sass_ctrl import parse  # noqa

K = re.compile(r"k_cellILi(\d+)ELi(\d+)E")
OPS = {0: "RCP", 1: "EX2", 2: "RSQ", 3: "SQRT", 4: "SIN"}


def main():
    build, outj = Path(sys.argv[1]), Path(sys.argv[2])
    sass = build / "sass" / "mufu_rcp.sass"
    kernels = parse(sass.read_text())
    res, fails = [], 0
    for name, kern in sorted(kernels.items()):
        m = K.search(name)
        if not m:
            continue
        opi, u = int(m.group(1)), int(m.group(2))
        clk = [i for i, x in enumerate(kern.insts)
               if "SR_CLOCKLO" in x.text]
        if len(clk) < 2:
            continue
        win = kern.insts[clk[-2] + 1:clk[-1]]
        n_mufu = sum(1 for x in win
                     if x.opcode.startswith("MUFU." + OPS[opi]))
        n_fmul = sum(1 for x in win if x.opcode.startswith("FMUL"))
        ok = (n_mufu == u and n_fmul == u)
        res.append({"kernel": name, "op": OPS[opi], "u": u,
                    "mufu": n_mufu, "fmul": n_fmul, "ok": ok})
        fails += 0 if ok else 1
        if not ok:
            print(f"FAIL {name}: mufu={n_mufu} fmul={n_fmul} u={u}")
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps({"kernels": res}, indent=1))
    print(f"sass_check: {len(res)} kernels, {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
