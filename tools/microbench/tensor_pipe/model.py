#!/usr/bin/env python3
"""tensor_pipe derivation (tools/microbench/tensor_pipe/protocol.json).

    python3 model.py --device b300 --run-dir $A/microbench/tensor_pipe/b300-r1 \
        --tree configs/trees/b300.json --out $A/microbench/tensor_pipe-model/b300-r1

Per cell: beat = OLS slope of the cad means over U >= 1; latency_1 =
(E_1(xw) - E_0(xw, commit-only of the same cta_group)) - (o_s(1) - o_s(0))
with o_s = SASS control-word stalls from the t_s clock read to the first
MMA (U >= 1) or to the commit (U = 0); completion = latency_1 - beat.
Both runs (run-a, run-b) are derived separately and compared.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sass_ctrl as sc  # noqa: E402

KIND_ID = {"f16": 0, "tf32": 1, "f8f6f4": 2, "mxf8f6f4": 3, "mxf4": 4, "mxf4nvf4": 5}
KIND_K = {"f16": 16, "tf32": 8, "f8f6f4": 32, "mxf8f6f4": 32, "mxf4": 64, "mxf4nvf4": 64}
KIND_CLASS = {"f16": "UTCHMMA", "tf32": "UTCHMMA", "f8f6f4": "UTCQMMA",
              "mxf8f6f4": "UTCQMMA", "mxf4": "UTCOMMA", "mxf4nvf4": "UTCOMMA"}
MODE_ID = {"xw": 0, "cad": 2}
MMA = re.compile(r"^UTC[HQO]MMA")

BEAT_RESID, BEAT_AGREE = 0.05, 0.03
# protocol-v2: steady-state slope over these U, residual bound 2 %
V2_U, V2_RESID = (32, 64, 128), 0.02
COMP_AGREE, COMP_AGREE_ABS = 0.05, 2.0
CONTROL = 0.10


def kernel_name(kind: str, cg: int, asrc: int, mode: int, u: int) -> str:
    return (f"_Z6k_ummaILi{KIND_ID[kind]}ELi{cg}ELi{asrc}ELi{mode}ELi{u}"
            "ELi0ELb0EEvPmPjiiijj2Fp")


def o_s_and_gate(k: sc.Kernel, u: int, kind: str, cg: int):
    ins = k.insts
    ops = [i for i, x in enumerate(ins) if MMA.match(x.opcode)]
    bars = [i for i, x in enumerate(ins) if x.opcode.startswith("UTCBAR")]
    why = []
    if len(ops) != u:
        why.append(f"{len(ops)} MMA != U {u}")
    want = KIND_CLASS[kind]
    if any(not ins[i].opcode.startswith(want) for i in ops):
        why.append(f"MMA opcode not {want}")
    if cg == 2 and any(".2CTA" not in ins[i].opcode for i in ops):
        why.append("cta_group 2 without .2CTA")
    target = ops[0] if u else (bars[0] if bars else None)
    if target is None:
        return None, why + ["no target instruction"]
    clocks = [i for i in range(target) if "SR_CLOCKLO" in ins[i].text]
    if not clocks:
        return None, why + ["no t_s clock read before the target"]
    ts = clocks[-1]
    return sum(x.stall for x in ins[ts:target]), why


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    resid = max(abs(a + b * x - y) / y for x, y in zip(xs, ys))
    return b, a, resid


def load_run(path: Path):
    cells = {}
    for line in path.open():
        r = json.loads(line)
        p = r["params"]
        s = r["samples"]
        key = (p["kind"], p["cta_group"], p["a_src"], p["layout"], p["n"], p["mode"], r["u"])
        cells[key] = {"mean": statistics.fmean(s), "sd": statistics.pstdev(s), "n": len(s),
                      "m": p["m"], "d_ok": r.get("d_ok", True)}
    return cells


def derive_b300(run_dir: Path, sass: dict[str, sc.Kernel], probe: str = "umma", fit_u=None):
    out = {}
    for run in ("run-a", "run-b"):
        cells = load_run(run_dir / run / f"{probe}.jsonl")
        # the commit-only kernel does not depend on N / layout: pool its cells
        pool = defaultdict(list)
        for (kind, cg, asrc, lay, n, mode, u), c in cells.items():
            if mode == "xw" and u == 0:
                pool[cg].append(c)
        base = {}
        for cg, cs in pool.items():
            tot = sum(c["n"] for c in cs)
            mean = sum(c["mean"] * c["n"] for c in cs) / tot
            var = sum(c["n"] * (c["sd"] ** 2 + (c["mean"] - mean) ** 2) for c in cs) / tot
            base[cg] = {"mean": mean, "sd": math.sqrt(var), "n": tot, "m": cs[0]["m"], "d_ok": True}
        groups = defaultdict(dict)
        for (kind, cg, asrc, lay, n, mode, u), c in cells.items():
            if u == 0:
                continue
            groups[(kind, cg, asrc, lay, n)][(mode, u)] = c
        res = {}
        for (kind, cg, asrc, lay, n), g in groups.items():
            us_all = sorted(u for (m, u) in g if m == "cad")
            us = [u for u in us_all if fit_u is None or u in fit_u]
            ys = [g[("cad", u)]["mean"] for u in us]
            beat, icpt, resid = ols(us, ys)
            lin_dev = {str(u): (g[("cad", u)]["mean"] - (icpt + beat * u)) / g[("cad", u)]["mean"]
                       for u in us_all if u not in us}
            # slope se from per-U sample se (independent means)
            mx = statistics.fmean(us)
            den = sum((u - mx) ** 2 for u in us)
            beat_se = math.sqrt(sum(((u - mx) / den) ** 2 *
                                    (g[("cad", u)]["sd"] ** 2 / g[("cad", u)]["n"]) for u in us))
            a = 1 if asrc == "tmem" else 0
            k1 = sass[kernel_name(kind, cg, a, 0, 1)]
            k0 = sass[kernel_name("f16", cg, 0, 0, 0)]
            os1, why1 = o_s_and_gate(k1, 1, kind, cg)
            os0, why0 = o_s_and_gate(k0, 0, "f16", cg)
            gate = []
            for u in us_all:
                _, w = o_s_and_gate(sass[kernel_name(kind, cg, a, 2, u)], u, kind, cg)
                gate += [f"cad U{u}: {x}" for x in w]
            gate += [f"xw U1: {x}" for x in why1] + [f"xw U0: {x}" for x in why0]
            e1, e0 = g[("xw", 1)], base[cg]
            lat1 = None if os1 is None or os0 is None else \
                (e1["mean"] - e0["mean"]) - (os1 - os0)
            lat_se = math.sqrt(e1["sd"] ** 2 / e1["n"] + e0["sd"] ** 2 / e0["n"])
            comp = None if lat1 is None else lat1 - beat
            res[f"{kind}.cg{cg}.{asrc}.{lay}.n{n}"] = {
                "kind": kind, "cta_group": cg, "a_src": asrc, "layout": lay, "n": n,
                "m": e1["m"], "k": KIND_K[kind],
                "beat": beat, "beat_intercept": icpt, "beat_resid": resid, "beat_se": beat_se,
                "latency_1": lat1, "latency_se": lat_se, "o_s1": os1, "o_s0": os0,
                "completion": comp, "completion_se": math.sqrt(lat_se ** 2 + beat_se ** 2),
                "d_ok": all(c["d_ok"] for c in g.values()), "sass_gate": gate,
                "cad_means": {str(u): g[("cad", u)]["mean"] for u in us_all},
                "linearity_dev": lin_dev,
            }
        out[run] = res
    return out


def installed_b300_beat(tree: dict, kind: str, cg: int, asrc: str, m: int, n: int, k: int):
    cls = KIND_CLASS[kind]
    nid = f"tensor.pipe.root/operation_class={cls}/umma_kind={kind}/cta_group={cg}/a_src={asrc}"
    node = next((x for x in tree["nodes"] if x["id"] == nid), None)
    if not node or "parameter" not in node:
        return None, None
    pts = node["parameter"]["fields"]["cycles"]["model"]["points"]
    for mm in ([m] if cg == 1 else [m, m // 2]):
        for p in pts:
            if p[:6] == [mm, n, k, "normal", 0, 0]:
                return p[6], mm
    return None, None


def judge(out: dict, tree: dict, version: int = 1):
    a, b = out["run-a"], out["run-b"]
    verdict = {}
    for key in sorted(a):
        x, y = a[key], b[key]
        beat = (x["beat"] + y["beat"]) / 2
        beat_agree = abs(x["beat"] - y["beat"]) / beat
        resid_bound = V2_RESID if version == 2 else BEAT_RESID
        beat_ok = (max(x["beat_resid"], y["beat_resid"]) <= resid_bound and beat_agree <= BEAT_AGREE
                   and not x["sass_gate"] and x["d_ok"] and y["d_ok"])
        comp = None
        comp_ok = False
        if x["completion"] is not None and y["completion"] is not None:
            comp = (x["completion"] + y["completion"]) / 2
            diff = abs(x["completion"] - y["completion"])
            comp_ok = diff <= max(COMP_AGREE * abs(comp), COMP_AGREE_ABS) and not x["sass_gate"]
        se = max(x["completion_se"], y["completion_se"])
        comp_install = None
        if comp is not None and comp_ok:
            if comp >= 0:
                comp_install = comp
            elif -comp <= 2 * se:
                comp_install = 0.0
        installed, m_used = installed_b300_beat(tree, x["kind"], x["cta_group"], x["a_src"],
                                                x["m"], x["n"], x["k"])
        if version == 2:
            lat = None if x["latency_1"] is None else (x["latency_1"] + y["latency_1"]) / 2
            ctrl = None if installed is None or lat is None else (lat - installed) / installed
        else:
            ctrl = None if installed is None else (beat - installed) / installed
        verdict[key] = {
            "kind": x["kind"], "cta_group": x["cta_group"], "a_src": x["a_src"], "layout": x["layout"],
            "n": x["n"], "beat": beat, "beat_agree": beat_agree,
            "beat_resid": max(x["beat_resid"], y["beat_resid"]),
            "beat_status": "measured" if beat_ok else "experimental",
            "completion": comp, "completion_se": se, "completion_ok": comp_ok,
            "completion_install": comp_install,
            "installed_beat": installed, "installed_m": m_used, "control_rel": ctrl,
            "latency_1": None if x["latency_1"] is None else (x["latency_1"] + y["latency_1"]) / 2,
            "control": None if ctrl is None else ("reproduced" if abs(ctrl) <= CONTROL else
                                                  ("replace" if beat_ok else "differs_not_installable")),
            "sass_gate": x["sass_gate"],
        }
    return verdict


def collapse_completion(verdict: dict):
    """Protocol install rule: per (kind, cta_group), mean over measured N and
    a_src of the sw128 cells when spread <= max(10 % of mean, 4 cycles)."""
    groups = defaultdict(list)
    for v in verdict.values():
        if v["layout"] != "sw128":
            continue
        groups[(v["kind"], v["cta_group"])].append(v)
    out = {}
    for (kind, cg), vs in sorted(groups.items()):
        vals = [v["completion_install"] for v in vs]
        ok = all(c is not None for c in vals)
        mean = statistics.fmean([c for c in vals if c is not None]) if any(c is not None for c in vals) else None
        spread = (max(c for c in vals if c is not None) - min(c for c in vals if c is not None)) \
            if mean is not None else None
        tight = mean is not None and spread <= max(0.10 * abs(mean), 4.0)
        out[f"{kind}.cg{cg}"] = {"kind": kind, "cta_group": cg, "cells": len(vs), "all_installable": ok,
                                 "mean": mean, "spread": spread, "collapse": tight and ok,
                                 "per_cell": {f"{v['a_src']}.n{v['n']}": v["completion_install"] for v in vs}}
    return out


# ---- h100 (wgmma probe) ---------------------------------------------------
GMMA = re.compile(r"^[HQI]GMMA\.")
H_DTYPE = {"f16": ("f16", "f32"), "f16a": ("f16", "f16"), "bf16": ("bf16", "f32"), "tf32": ("tf32", "f32"),
           "e4m3e4m3": ("e4m3", "f32"), "e4m3e5m2": ("e4m3", "f32"), "e5m2e5m2": ("e5m2", "f32"),
           "s8s8": ("s8", "s32"),
           "sp_f16": ("f16", "f32"), "sp_bf16": ("bf16", "f32"),
           "sp_e4m3e4m3": ("e4m3", "f32"),
           "spf16": ("f16", "f32"), "spbf16": ("bf16", "f32"),
           "spe4m3e4m3": ("e4m3", "f32")}


def h100_kernels(sass_texts: list[str]) -> dict[tuple[int, int], sc.Kernel]:
    out = {}
    for t in sass_texts:
        for name, k in sc.parse(t).items():
            m = re.search(r"k_wgmmaILi(\d+)ELi(\d+)E", name)
            if m:
                out[(int(m.group(1)), int(m.group(2)))] = k
    return out


def h100_sergrp_kernels(sass_texts: list[str]) -> dict[tuple[int, int, int], sc.Kernel]:
    """(cid, mode, u) -> kernel for k_sergrp (h100-r4 ser / grp)."""
    out = {}
    for t in sass_texts:
        for name, k in sc.parse(t).items():
            m = re.search(r"k_sergrpILi(\d+)ELi(\d+)ELi(\d+)E", name)
            if m:
                out[(int(m.group(1)), int(m.group(2)),
                     int(m.group(3)))] = k
    return out


DEPBAR = re.compile(r"DEPBAR")
BANNED = re.compile(r"\bBRA\b|\bELECT\b|\bWARPSYNC\b")


def h100_sergrp_gate(k: sc.Kernel, mode: int, u: int) -> list[str]:
    """h100-r4 SASS gate for one ser / grp kernel: mode 0 (ser) -> U
    target ops + U DEPBAR.LE gsb0,0x0; mode 1 (grp) -> 4U ops + U
    DEPBARs; no BRA/ELECT/WARPSYNC between target ops; no null GMMA."""
    g = []
    ops = [i for i, x in enumerate(k.insts)
           if GMMA.match(x.opcode) and not null_gmma(x)]
    want = u * (4 if mode else 1)
    if len(ops) != want:
        g.append(f"{len(ops)} GMMA ops != {want}")
    deps = [x for x in k.insts if "DEPBAR" in x.opcode]
    if len(deps) != u or any("0x0" not in x.text for x in deps):
        g.append(f"DEPBAR.LE gsb0,0x0 count {len(deps)} != {u}")
    for a, b in zip(ops, ops[1:]):
        if any(BANNED.search(x.text) for x in k.insts[a + 1:b]):
            g.append("BRA/ELECT/WARPSYNC between target ops")
            break
    if any(null_gmma(x) for x in k.insts):
        g.append("null GMMA present")
    return g


def h100_o_iter(k: sc.Kernel):
    """per-iteration non-target-instruction stall sum (o_s convention):
    the stall fields of the non-GMMA instructions in the gap between
    consecutive target ops that contains a DEPBAR (for ser every gap;
    for grp only the group boundary gap)."""
    w = sc.clock_windows(k)
    ins = w[0] if w else k.insts
    tgt = {i for i, x in enumerate(ins)
           if GMMA.match(x.opcode) and not null_gmma(x)}
    deps = [i for i, x in enumerate(ins) if "DEPBAR" in x.opcode]
    tpos = sorted(tgt)
    vals = []
    for a, b in zip(tpos, tpos[1:]):
        if any(a < d < b for d in deps):
            vals.append(sum(x.stall for j, x in enumerate(ins)
                            if a < j < b and j not in tgt))
    return statistics.fmean(vals) if vals else None


def null_gmma(x) -> bool:
    return bool(re.match(r"^[HQI]GMMA\S*\s+RZ,", x.body)) and "!UPT" in x.text


def h100_o_s(k: sc.Kernel, u: int):
    ins = k.insts
    if u:
        tgt = next((i for i, x in enumerate(ins) if GMMA.match(x.opcode) and not null_gmma(x)), None)
    else:
        tgt = next((i for i, x in enumerate(ins) if x.opcode.startswith("WARPGROUP.DEPBAR")), None)
    if tgt is None:
        return None
    clocks = [i for i in range(tgt) if "SR_CLOCKLO" in ins[i].text]
    return sum(x.stall for x in ins[clocks[-1]:tgt]) if clocks else None


def serialized_kernels(ptxas_dir: Path) -> set[tuple[int, int]]:
    """(cid, U) of kernels ptxas reports as C7511 wgmma-serialized."""
    out = set()
    for f in ptxas_dir.glob("*.ptxas.txt"):
        for m in re.finditer(r"C7511\).*?k_wgmmaILi(\d+)ELi(\d+)E", f.read_text()):
            out.add((int(m.group(1)), int(m.group(2))))
    return out


def derive_h100(run_dir: Path, cells_json: Path, kernels: dict,
                sergrp_kernels: dict | None = None,
                serialized=frozenset(), loop16=False):
    if sergrp_kernels is None:
        sergrp_kernels = {}
    cells = {c["cid"]: c for c in json.loads(cells_json.read_text())}
    by_coord = {(c["kind"], c["src"], c["n"]): cid for cid, c in cells.items()}
    out = {}
    for run in ("run-a", "run-b"):
        recs = defaultdict(dict)
        for f in sorted((run_dir / run).glob("wgmma_*.jsonl")):
            kind = f.stem.split("_", 1)[1]
            for line in f.open():
                r = json.loads(line)
                p = r["params"]
                mode = p["mode"] + (".wg2" if r["variant"].endswith(".wg2") else "")
                s = r["samples"]
                recs[(kind, p["src"], p["n"])][(mode, r["u"])] = {
                    "mean": statistics.fmean(s), "sd": statistics.pstdev(s), "n": len(s)}
        res = {}
        for (kind, src, n), g in sorted(recs.items()):
            us = [u for u in V2_U if ("cad", u) in g]
            beat, icpt, resid = ols(us, [g[("cad", u)]["mean"] for u in us])
            mx = statistics.fmean(us)
            den = sum((u - mx) ** 2 for u in us)
            beat_se = math.sqrt(sum(((u - mx) / den) ** 2 * g[("cad", u)]["sd"] ** 2 / g[("cad", u)]["n"]
                                    for u in us))
            lin16 = None
            if ("cad", 16) in g:
                lin16 = (g[("cad", 16)]["mean"] - (icpt + beat * 16)) / g[("cad", 16)]["mean"]
            wg2 = None
            us2 = [u for u in V2_U if ("cad.wg2", u) in g]
            if len(us2) >= 2:
                wg2 = ols(us2, [g[("cad.wg2", u)]["mean"] for u in us2])[0] / beat
            _kmap = {"spf16": "sp_f16", "spbf16": "sp_bf16",
                     "spe4m3e4m3": "sp_e4m3e4m3"}
            cid = by_coord.get((kind, src, n)) or by_coord.get(
                (_kmap.get(kind, kind), src, n))
            # ---- ser / grp (h100-r4 addendum) ----
            ser = {}
            ser_us = [u for u in (1, 2, 4, 8) if ("ser", u) in g]
            if len(ser_us) == 4:
                l_ser, i_ser, res_ser = ols(
                    ser_us, [g[("ser", u)]["mean"] for u in ser_us])
                mx_s = statistics.fmean(ser_us)
                den_s = sum((u - mx_s) ** 2 for u in ser_us)
                l_ser_se = math.sqrt(sum(
                    ((u - mx_s) / den_s) ** 2 *
                    g[("ser", u)]["sd"] ** 2 / g[("ser", u)]["n"]
                    for u in ser_us))
                o_ser = None
                sg = []
                for u in ser_us:
                    k = sergrp_kernels.get((cid, 0, u)) if cid else None
                    if k is None:
                        sg.append(f"no SASS ser kernel U{u}")
                    else:
                        sg += [f"ser U{u}: {x}"
                               for x in h100_sergrp_gate(k, 0, u)]
                        if u == ser_us[-1]:
                            o_ser = h100_o_iter(k)
                comp_ser = (None if o_ser is None else
                            l_ser - beat - o_ser)
                ser = {"l_ser": l_ser, "ser_resid": res_ser,
                       "l_ser_se": l_ser_se, "o_ser": o_ser,
                       "completion_ser": comp_ser,
                       "completion_ser_se": math.sqrt(
                           l_ser_se ** 2 + beat_se ** 2),
                       "ser_gate": sg}
            grp = {}
            grp_us = [u for u in (1, 2, 4) if ("grp", u) in g]
            if len(grp_us) == 3:
                l_grp, i_grp, res_grp = ols(
                    grp_us, [g[("grp", u)]["mean"] for u in grp_us])
                o_grp = None
                gg = []
                for u in grp_us:
                    k = sergrp_kernels.get((cid, 1, u)) if cid else None
                    if k is None:
                        gg.append(f"no SASS grp kernel U{u}")
                    else:
                        gg += [f"grp U{u}: {x}"
                               for x in h100_sergrp_gate(k, 1, u)]
                        if u == grp_us[-1]:
                            o_grp = h100_o_iter(k)
                grp = {"l_grp": l_grp, "grp_resid": res_grp,
                       "o_grp": o_grp,
                       "completion_grp": (None if o_grp is None else
                                          l_grp - 4 * beat - o_grp),
                       "grp_gate": gg}
            inp0, acc0 = H_DTYPE[kind]
            res[f"{kind}.{src}.n{n}"] = {
                "kind": kind, "input_dtype": inp0, "acc": acc0,
                "src": src, "n": n, "ser": ser, "grp": grp}
            os1 = os0 = None
            gate = []
            if cid is None:
                gate.append("cell not in wgmma_cells.json")
            else:
                for u in [0, 1] + list(us):
                    k = kernels.get((cid, u))
                    if k is None:
                        gate.append(f"no SASS kernel for cid {cid} U {u}")
                        continue
                    # the empty commit group of the U = 0 kernel compiles to a
                    # null op `HGMMA.64x8x16.F16 RZ, gdesc[URZ], RZ, !UPT, gsb0`
                    # (no destination); it is not one of the U measured ops
                    nops = sum(1 for x in k.insts if GMMA.match(x.opcode) and not null_gmma(x))
                    # protocol-v2 addendum h100-r3: rs cad U > 16 is a runtime loop
                    # over a 16-op body
                    want = u if (u <= 16 or not loop16) else 16
                    if nops != want:
                        gate.append(f"U {u}: {nops} GMMA in kernel (want {want})")
                    if (cid, u) in serialized:
                        gate.append(f"U {u}: ptxas C7511 serialized")
                k1, k0 = kernels.get((cid, 1)), kernels.get((cid, 0))
                os1 = h100_o_s(k1, 1) if k1 else None
                os0 = h100_o_s(k0, 0) if k0 else None
            e1, e0 = g[("lat", 1)], g[("lat", 0)]
            lat1 = None if os1 is None or os0 is None else (e1["mean"] - e0["mean"]) - (os1 - os0)
            lat_se = math.sqrt(e1["sd"] ** 2 / e1["n"] + e0["sd"] ** 2 / e0["n"])
            inp, acc = H_DTYPE[kind]
            res[f"{kind}.{src}.n{n}"].update({
                "kind": kind, "input_dtype": inp, "acc": acc, "src": src, "n": n,
                "beat": beat, "beat_resid": resid, "beat_se": beat_se, "linearity_u16": lin16,
                "latency_1": lat1, "latency_se": lat_se, "o_s1": os1, "o_s0": os0,
                "completion": None if lat1 is None else lat1 - beat,
                "completion_se": math.sqrt(lat_se ** 2 + beat_se ** 2), "wg2_ratio": wg2,
                "sass_gate": gate, "cad_means": {str(u): g[("cad", u)]["mean"] for u in us},
                "raw_lat1_minus_lat0": e1["mean"] - e0["mean"]})
        out[run] = res
    return out


def judge_h100(out: dict):
    a, b = out["run-a"], out["run-b"]
    v = {}
    for key in sorted(a):
        x, y = a[key], b[key]
        beat = (x["beat"] + y["beat"]) / 2
        agree = abs(x["beat"] - y["beat"]) / beat
        ok = max(x["beat_resid"], y["beat_resid"]) <= V2_RESID and agree <= BEAT_AGREE and not x["sass_gate"]
        comp = ci = None
        cok = False
        if x["completion"] is not None and y["completion"] is not None:
            comp = (x["completion"] + y["completion"]) / 2
            cok = abs(x["completion"] - y["completion"]) <= max(COMP_AGREE * abs(comp), COMP_AGREE_ABS) \
                and not x["sass_gate"]
            se = max(x["completion_se"], y["completion_se"])
            if cok:
                ci = comp if comp >= 0 else (0.0 if -comp <= 2 * se else None)
        v[key] = {k: x[k] for k in ("kind", "input_dtype", "acc", "src", "n")} | {
            "beat": beat, "beat_agree": agree, "beat_resid": max(x["beat_resid"], y["beat_resid"]),
            "beat_status": ("unrealised" if any("C7511" in g for g in x["sass_gate"]) else
                            "measured" if ok else "experimental"), "completion": comp,
            "completion_ok": cok, "completion_install": ci, "wg2_ratio": x["wg2_ratio"],
            "linearity_u16": x["linearity_u16"], "sass_gate": x["sass_gate"]}
        # ---- ser / grp verdicts (h100-r4 addendum acceptance) ----
        xs, ys = x.get("ser") or {}, y.get("ser") or {}
        if xs and ys and xs.get("completion_ser") is not None and \
                ys.get("completion_ser") is not None:
            comp_s = (xs["completion_ser"] + ys["completion_ser"]) / 2
            sdiff = abs(xs["completion_ser"] - ys["completion_ser"])
            sagree = sdiff <= max(0.03 * abs(comp_s), 2.0)
            sres = max(xs["ser_resid"], ys["ser_resid"])
            sgate = (xs.get("ser_gate") or []) + (ys.get("ser_gate") or [])
            if sres <= 0.02 and sagree and not sgate:
                sstat = "measured"
            elif sres <= 0.05 or sdiff <= max(0.08 * abs(comp_s), 2.0):
                sstat = "experimental"
            else:
                sstat = "unrealised"
            se_s = max(xs.get("completion_ser_se") or 0,
                       ys.get("completion_ser_se") or 0)
            sinst = None
            if sstat in ("measured", "experimental"):
                sinst = (comp_s if comp_s >= 0 else
                         (0.0 if -comp_s <= 2 * se_s else None))
            v[key] |= {"l_ser": (xs["l_ser"] + ys["l_ser"]) / 2,
                       "o_ser": xs.get("o_ser"),
                       "completion_ser": comp_s, "ser_resid": sres,
                       "ser_agree": sdiff, "ser_status": sstat,
                       "ser_gate": sgate,
                       "completion_ser_install": sinst}
        xg, yg = x.get("grp") or {}, y.get("grp") or {}
        if xg and yg and xg.get("completion_grp") is not None and \
                yg.get("completion_grp") is not None and \
                v[key].get("completion_ser") is not None:
            comp_g = (xg["completion_grp"] + yg["completion_grp"]) / 2
            cs = v[key]["completion_ser"]
            v[key] |= {"l_grp": (xg["l_grp"] + yg["l_grp"]) / 2,
                       "o_grp": xg.get("o_grp"),
                       "completion_grp": comp_g,
                       "grp_gate": (xg.get("grp_gate") or []) +
                                   (yg.get("grp_gate") or []),
                       "grp_vs_ser_pct": (None if not cs else
                                          (comp_g - cs) / cs)}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["b300", "h100"])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--version", type=int, default=1, choices=[1, 2])
    ap.add_argument("--loop16", action="store_true",
                    help="h100-r3 rs layout: cad U > 16 as a loop over 16 ops")
    ap.add_argument("--sass-dir", type=Path, default=None,
                    help="h100: SASS dumps (default <run-dir>/build/sass)")
    a = ap.parse_args()
    if a.device == "h100":
        sass_dir = a.sass_dir or (a.run_dir / "build" / "sass")
        texts = [p.read_text() for p in sorted(sass_dir.glob("wgmma_*.sass"))]
        kernels = h100_kernels(texts)
        sg_kernels = h100_sergrp_kernels(texts)
        runs = derive_h100(a.run_dir, a.run_dir / "build" / "wgmma_cells.json", kernels,
                           sg_kernels,
                           serialized_kernels(a.run_dir / "build" / "sass"), a.loop16)
        verdict = judge_h100(runs)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "model.json").write_text(json.dumps(
            {"protocol": "tools/microbench/tensor_pipe/protocol-v2.json", "device": "h100",
             "run_dir": str(a.run_dir), "kernels_parsed": len(kernels), "runs": runs, "cells": verdict}, indent=1))
        for key, v in verdict.items():
            _ser = v.get('completion_ser')
            print(f"{key:24s} beat {v['beat']:8.2f} {v['beat_status']:12s} res {v['beat_resid']*100:4.1f}% "
                  f"comp {v['completion']} -> {v['completion_install']} wg2 {v['wg2_ratio']} gate {len(v['sass_gate'])}"
                  + (f" | ser L={v.get('l_ser'):.1f} o={v.get('o_ser')} cs={_ser:.2f} "
                     f"->{v.get('completion_ser_install')} {v.get('ser_status')} "
                     f"cg={v.get('completion_grp')}" if _ser is not None else ""))
        print("kernels parsed", len(kernels))
        return
    probe = "umma" if a.version == 1 else "umma_cadence"
    sass = sc.parse((a.run_dir / "build" / "sass" / f"{probe}.sass").read_text())
    runs = derive_b300(a.run_dir, sass, probe, None if a.version == 1 else V2_U)
    tree = json.loads(a.tree.read_text())
    verdict = judge(runs, tree, a.version)
    comp = collapse_completion(verdict)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "model.json").write_text(json.dumps(
        {"protocol": "tools/microbench/tensor_pipe/" + ("protocol.json" if a.version == 1 else "protocol-v2.json"),
         "device": a.device,
         "run_dir": str(a.run_dir), "runs": runs, "cells": verdict, "completion_by_kind": comp},
        indent=1))
    for key, v in verdict.items():
        print(f"{key:34s} beat {v['beat']:8.2f} ({v['beat_status']:12s} agr {v['beat_agree']*100:4.1f}% "
              f"res {v['beat_resid']*100:4.1f}%) inst {v['installed_beat']} ctrl {v['control']} "
              f"comp {v['completion'] if v['completion'] is None else round(v['completion'], 2)} "
              f"-> {v['completion_install']} gate {len(v['sass_gate'])}")
    for k, v in comp.items():
        print(k, v["mean"], v["spread"], "collapse" if v["collapse"] else "NO-collapse", v["per_cell"])


if __name__ == "__main__":
    main()
