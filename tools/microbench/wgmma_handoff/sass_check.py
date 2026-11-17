#!/usr/bin/env python3
"""SASS gate for the wgmma_handoff probe (protocol.json sass_gate).

    sass_check.py --sass <handoff.sass> --out <checks.json>

Per kernel k_handoff<CID, D, CTRL>: before the t_w SR_CLOCKLO read
exactly 1 + D wgmma of the cell's SASS form (class root + 64xNxK shape
token) and one WARPGROUP.DEPBAR.LE of depth D; between the t_w and t_e
clock reads exactly one FFMA and one FADD, nothing else.  0 classified
kernels = gate failure.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

# cid -> (form, class, n, k, src) (handoff.cu's HC table)
CELLS = {
    142: ("hgmma_bf16_n64k16_ss",  "HGMMA",  64, 16, "ss"),
    158: ("hgmma_bf16_n128k16_ss", "HGMMA", 128, 16, "ss"),
    159: ("hgmma_bf16_n128k16_rs", "HGMMA", 128, 16, "rs"),
    46:  ("hgmma_f16_n192k16_ss",  "HGMMA", 192, 16, "ss"),
    222: ("hgmma_tf32_n128k8_ss",  "HGMMA", 128,  8, "ss"),
    286: ("qgmma_e4m3_n128k32_ss", "QGMMA", 128, 32, "ss"),
    288: ("qgmma_e4m3_n136k32_ss", "QGMMA", 136, 32, "ss"),
    302: ("qgmma_e4m3_n192k32_ss", "QGMMA", 192, 32, "ss"),
}
WGMMA_ROOTS = ("HGMMA", "QGMMA", "IGMMA", "BGMMA", "UTCHMMA")

_NAME = re.compile(r"k_handoffILi(\d+)ELi(\d+)ELi(\d+)E")


def demangle(name):
    m = _NAME.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v1", "probe": "handoff",
           "sass": a.sass, "kernels": {}, "unclassified": []}
    for name, k in kernels.items():
        d = demangle(name)
        if d is None:
            out["unclassified"].append(name)
            continue
        cid, dep, ctrl = d
        form, cls, n, kk, src = CELLS[cid]
        key = f"handoff/{form}/d{dep}/{'ctrl' if ctrl else 'acc'}"
        res = {"key": key, "cid": cid, "d": dep,
               "cell": "ctrl" if ctrl else "acc", "kernel": name,
               "ok": False, "why": []}
        clk = [i for i, ins in enumerate(k.insts) if "SR_CLOCKLO" in ins.text]
        if len(clk) < 2:
            res["why"].append("fewer than 2 SR_CLOCKLO reads")
            out["kernels"][key] = res
            continue
        i_tw, i_te = clk[0], clk[1]
        before, window = k.insts[:i_tw], k.insts[i_tw + 1:i_te]
        shape_tok = f"64x{n}x{kk}"
        hits = [i for i in before
                if i.root in WGMMA_ROOTS]
        good = [i for i in hits
                if i.root == cls and shape_tok.lower() in i.text.lower()]
        depbars = [i for i in before if "DEPBAR" in i.opcode.upper()]
        res["wgmma_count"] = len(hits)
        res["wgmma_opcodes"] = sorted({i.opcode for i in hits})
        res["depbar_opcodes"] = [i.opcode for i in depbars]
        res["window_opcodes"] = [i.opcode for i in window]
        # src mode evidence: record the operand text of the first hit
        if hits:
            res["wgmma_text"] = hits[0].text
        if len(good) != 1 + dep:
            res["why"].append(
                f"{cls} {shape_tok} count {len(good)} != {1 + dep} "
                f"(all-class wgmma {len(hits)})")
        if len(depbars) != 1:
            res["why"].append(f"{len(depbars)} DEPBAR before t_w")
        else:
            m = re.search(r"0x([0-9a-fA-F]+)", depbars[0].text)
            if not m or int(m.group(1), 16) != dep:
                res["why"].append(
                    f"DEPBAR text not .LE {dep}: {depbars[0].text}")
        wops = Counter(i.root for i in window)
        if not (wops.get("FFMA") == 1 and wops.get("FADD") == 1 and
                len(window) == 2):
            res["why"].append(
                f"window not exactly FFMA+FADD: {sorted(wops.items())}")
        res["ok"] = not res["why"]
        out["kernels"][key] = res
    n_match = len(out["kernels"])
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = n_match - out["n_ok"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check: {out['n_ok']}/{n_match} gated kernels ok")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"][:3])
    return 0 if n_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
