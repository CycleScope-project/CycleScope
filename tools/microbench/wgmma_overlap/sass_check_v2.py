#!/usr/bin/env python3
"""SASS gate for the wgmma_overlap v2 probe (protocol-v2.json sass_gate).

    sass_check_v2.py --sass <overlap_v2.sass> --out <checks.json>

Per kernel k_ov2<CID, G, V, U, MODE, WB>; checks.json keys as
model_v2.gate_key: <form>.g<G>.<kind>.u<U>.<self|other>.wb<W>,
ctrl.<kind>.u<U>.wb<W>, park.<kind>.u<U>.wb2.
  (1) between the window's two SR_CLOCKLO reads exactly U instructions
      of the kind's expected opcode and nothing else:
        dep3r/dep2/ilp8n/ilp8 -> FFMA;  fmnmx -> FMNMX (never FMNMX3);
        imad -> IMAD (never IMAD.MOV);  fadd1 -> FADD;  fmul2 -> FMUL.
  (2) operand sets: ilp8n FFMAs have three pairwise-distinct GPR sources
      and no .reuse anywhere in the window; ilp8 FFMAs carry two
      .reuse sources (a and b), except the final unit may have none;
      fadd1 FADDs have an immediate source; fmul2 FMULs have two
      distinct GPR sources and no .reuse.
  (3) MODE self / other: inside the block between the two BAR.SYNC of
      warpgroup 0's rep body, exactly G HGMMA/QGMMA of the form's shape
      (64xNxK) and source mode, zero WARPGROUP.DEPBAR, any
      WARPGROUP.ARRIVE.
  (4) MODE self / other: exactly one WARPGROUP.DEPBAR.LE gsb0, 0x0 after
      the wg0 rep body's second BAR.SYNC.
      MODE == ctrl: no wgmma-class instruction anywhere in the kernel.
      MODE == park: no wgmma-class instruction anywhere, and at least
      one BAR.SYNC on barrier 0x2 (the park barrier).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

# cid -> (form, class, n, k, src)
FORMS = {
    190: ("hgmma_bf16_n256_ss", "HGMMA", 256, 16, "ss"),
    191: ("hgmma_bf16_n256_rs", "HGMMA", 256, 16, "rs"),
    142: ("hgmma_bf16_n64_ss", "HGMMA", 64, 16, "ss"),
    158: ("hgmma_bf16_n128_ss", "HGMMA", 128, 16, "ss"),
    318: ("qgmma_e4m3_n256_ss", "QGMMA", 256, 32, "ss"),
}
KINDS = {0: "dep3r", 2: "dep2", 9: "fmnmx", 5: "ilp8n", 4: "ilp8",
         8: "imad", 15: "fadd1", 16: "fmul2"}
MODES = {0: "self", 1: "other", 2: "ctrl", 3: "park"}
WGMMA_ROOTS = ("HGMMA", "QGMMA", "IGMMA", "BGMMA", "UTCHMMA")

_NAME = re.compile(r"k_ov2ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E")
_GPR = re.compile(r"R(\d+)$")


def demangle(name):
    m = _NAME.search(name)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())


def kind_ok(ins, v):
    op, root = ins.opcode, ins.root
    if v in (0, 2, 4, 5):
        return root == "FFMA"
    if v == 9:
        return root == "FMNMX" and not op.startswith("FMNMX3")
    if v == 8:
        return root == "IMAD" and not op.startswith("IMAD.MOV")
    if v == 15:
        return root == "FADD"
    if v == 16:
        return root == "FMUL"
    return False


def src_regs(ins):
    """GPR register numbers among the instruction's sources."""
    body = ins.body.split(";", 1)[0]
    parts = body.split(None, 1)
    if len(parts) < 2:
        return []
    toks = [t.strip() for t in parts[1].split(",")][1:]
    regs = []
    for t in toks:
        t = t.split(".")[0].lstrip("-~!|")
        m = _GPR.match(t)
        if m:
            regs.append(int(m.group(1)))
    return regs


def n_reuse(ins):
    body = ins.body.split(";", 1)[0]
    parts = body.split(None, 1)
    if len(parts) < 2:
        return 0
    return sum(1 for t in parts[1].split(",")[1:] if ".reuse" in t)


_IMM = re.compile(r"^[-+]?\d+(\.\d+)?$|^0[xX][0-9a-fA-F]+$")


def has_imm_src(ins):
    """True when at least one source operand is an immediate literal."""
    body = ins.body.split(";", 1)[0]
    parts = body.split(None, 1)
    if len(parts) < 2:
        return False
    for t in parts[1].split(",")[1:]:
        t = t.strip().split(".")[0].lstrip("-~!|")
        if _IMM.match(t):
            return True
    return False


def wgmma_src(ins):
    """'ss' if the operand after the dst is a gdesc, 'rs' if a GPR."""
    body = ins.body.split(";", 1)[0]
    parts = body.split(None, 1)
    if len(parts) < 2:
        return "?"
    toks = [t.strip() for t in parts[1].split(",")]
    if len(toks) < 2:
        return "?"
    return "rs" if _GPR.match(toks[1].split(".")[0]) else "ss"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v1", "probe": "overlap",
           "sass": a.sass, "kernels": {}, "unclassified": []}
    for name, k in kernels.items():
        d = demangle(name)
        if d is None:
            out["unclassified"].append(name)
            continue
        cid, g, v, u, mode, wb = d
        form, cls, n, kk, src = FORMS[cid]
        kind = KINDS[v]
        mname = MODES[mode]
        key = (f"ctrl.{kind}.u{u}.wb{wb}" if mname == "ctrl"
               else f"park.{kind}.u{u}.wb{wb}" if mname == "park"
               else f"{form}.g{g}.{kind}.u{u}.{mname}.wb{wb}")
        res = {"key": key, "cid": cid, "g": g, "v": v, "u": u,
               "mode": mname, "kernel": name, "ok": False, "why": []}
        clk = [i for i, ins in enumerate(k.insts) if "SR_CLOCKLO" in ins.text]
        res["n_clock"] = len(clk)
        if len(clk) < 2:
            res["why"].append("fewer than 2 SR_CLOCKLO reads")
            out["kernels"][key] = res
            continue
        # window = the clock-to-clock span holding the expected kind ops
        # (in `other` mode wg0's t_d clock may precede the window span)
        spans = list(zip(clk, clk[1:]))
        best, window, i_ts, i_tw = -1, [], -1, -1
        for sa, sb in spans:
            w = k.insts[sa + 1:sb]
            n_exp = sum(1 for i in w if kind_ok(i, v))
            if len(w) == u and n_exp == u:
                best, window, i_ts, i_tw = u, w, sa, sb
                break
            if n_exp > best:
                best, window, i_ts, i_tw = n_exp, w, sa, sb
        before = k.insts[:i_ts]
        after = k.insts[i_tw + 1:]
        # (1) window: exactly U expected opcodes, nothing else
        bad = [i.text for i in window if not kind_ok(i, v)]
        res["window_len"] = len(window)
        res["window_opcodes"] = sorted(set(i.opcode for i in window))
        if len(window) != u or bad:
            res["why"].append(
                f"window {len(window)} insts (U={u}), "
                f"unexpected: {bad[:4] if bad else 'count'}")
        # (2) operand sets
        if v == 5:  # ilp8n: 3 distinct GPR srcs, no .reuse
            for i in window:
                r = src_regs(i)
                if len(r) != 3 or len(set(r)) != 3:
                    res["why"].append(f"ilp8n srcs not 3 distinct GPRs: {i.text}")
                    break
            reuses = [i.text for i in window if ".reuse" in i.text]
            if reuses:
                res["why"].append(f"ilp8n window has .reuse: {reuses[0]}")
        elif v == 4:  # ilp8: a / b reuse-served; the last unit of a
            # window has no later consumer and may drop .reuse (addendum 1)
            for j, i in enumerate(window):
                if n_reuse(i) != 2 and j != len(window) - 1:
                    res["why"].append(
                        f"ilp8 FFMA without 2 .reuse srcs: {i.text}")
                    break
        # (3)+(4) wgmma block (addendum 2): the check region is the block
        # between the two BAR.SYNC of warpgroup 0's rep body -- the
        # consecutive-BAR pair that brackets the wgmma ops
        hits_all = [i for i in k.insts if i.root in WGMMA_ROOTS]
        shape_tok = f"64x{n}x{kk}"
        if mode == 2:
            if hits_all:
                res["why"].append(f"ctrl kernel has {len(hits_all)} wgmma")
        elif mode == 3:
            if hits_all:
                res["why"].append(f"park kernel has {len(hits_all)} wgmma")
            bar2 = [i.text for i in k.insts
                    if i.opcode.startswith("BAR.SYNC") and "0x2" in i.text]
            res["park_bars"] = bar2
            if not bar2:
                res["why"].append("no BAR.SYNC on barrier 0x2")
        else:
            bars = [i for i, ins in enumerate(k.insts)
                    if ins.opcode.startswith("BAR.SYNC")]
            # the BAR..BAR block containing the wgmma ops
            block = None
            for ba_i, bb_i in zip(bars, bars[1:]):
                seg = k.insts[ba_i + 1:bb_i]
                if any(i.root in WGMMA_ROOTS for i in seg):
                    block = (ba_i, bb_i, seg)
                    break
            if block is None:
                res["why"].append(
                    f"no wgmma ops in any BAR..BAR block "
                    f"({len(hits_all)} total, {len(bars)} bars)")
            else:
                ba, bb, block_insts = block
                hits = [i for i in block_insts if i.root in WGMMA_ROOTS]
                good = [i for i in hits
                        if i.root == cls and shape_tok in i.opcode]
                arrives = [i for i in block_insts
                           if "ARRIVE" in i.opcode.upper()]
                pre_dep = [i for i in block_insts
                           if "DEPBAR" in i.opcode.upper()]
                res["wgmma_count"] = len(good)
                res["wgmma_opcodes"] = sorted({i.opcode for i in good})
                res["arrive_count"] = len(arrives)
                if good:
                    res["wgmma_text"] = good[0].text
                if len(good) != g:
                    res["why"].append(
                        f"{cls} {shape_tok} count {len(good)} != G={g} "
                        f"(all-class wgmma {len(hits)})")
                srcs = {wgmma_src(i) for i in good}
                res["wgmma_srcs"] = sorted(srcs)
                if srcs != {src}:
                    res["why"].append(
                        f"wgmma src mode {sorted(srcs)} != {src}")
                if pre_dep:
                    res["why"].append(
                        f"{len(pre_dep)} WARPGROUP.DEPBAR in wgmma block: "
                        f"{pre_dep[0].text}")
                # (4) exactly one DEPBAR.LE gsb0, 0x0 after the second
                # bar of the wg0 rep body (self: also after the window)
                le0 = [i for i in k.insts[bb:]
                       if "DEPBAR" in i.opcode.upper()
                       and "LE" in i.opcode and "gsb0, 0x0" in i.text]
                res["depbar_opcodes"] = [i.opcode for i in le0]
                if len(le0) != 1:
                    all_d = [i.text for i in k.insts
                             if "DEPBAR" in i.opcode.upper()]
                    res["why"].append(
                        f"{len(le0)} WARPGROUP.DEPBAR.LE gsb0, 0x0 after "
                        f"wgmma block (all depbars: {all_d[:4]})")
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
    if not n_match:
        return 2
    return 0 if out["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
