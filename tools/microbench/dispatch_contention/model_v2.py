#!/usr/bin/env python3
"""dispatch_contention protocol v2: measured per-warp / CTA window slopes
against a replica of the engine's SMSP issue model under each declared
candidate rule (baseline, P, O2x, O4x, O2p, O4p; protocol-v2.json
"engine_rules_declared_before_data" / "derivation" / "decision").

Inputs: $A/microbench/dispatch_contention/<dev>-<run>/run-{a,b}/*.jsonl
(rows "<variant>.w<W>" per-warp and "<variant>.cta.w<W>" CTA windows),
$A/microbench/dispatch_contention-model/<dev>-<run>/checks.json
(sass_check_v2.py: per kernel "program" = the window's instruction
list with the two CS2R bounds), configs/trees/<dev>.json (installed
instruction.service leaves and resources).  Writes model.json next to
checks.json.

Usage: model_v2.py <artifact_root> <device> <run> [--v3]
(--v3: protocol-v3.json candidate order baseline / OP2 / O2x / P,
greedy cells gate on the CTA slope when the per-warp window is bimodal,
form_mismatch report against the intended GPR read counts)
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

US = (64, 128, 256)
WS = (1, 2, 3, 4)
VARIANTS = ("dep3r", "dep3n", "dep2", "dep1", "ilp8", "ilp8n", "ilp8n2",
            "ilp2", "imad", "mixed", "mufu.ex2", "mufu.rcp", "mufu.rsqrt",
            "mufu.lg2", "mufu.sin", "mufu.tanh")
MUFU = tuple(v for v in VARIANTS if v.startswith("mufu."))
# gating set (protocol "decision" + addendum 2): dependent FP32 cells at
# W = 2, mixed at W >= 2 and all mufu cells are record only
GATE_W = {"dep3r": (1, 3, 4), "dep3n": (1, 3, 4), "dep2": (1, 3, 4),
          "dep1": (1, 3, 4), "mixed": (1,),
          "ilp8": WS, "ilp8n": WS, "ilp8n2": WS, "ilp2": WS, "imad": WS}
CANDIDATES_V2 = ("baseline", "P", "O2x", "O4x", "O2p", "O4p")
# protocol v3: operand-port candidate first; O4x / O2p / O4p dropped
CANDIDATES_V3 = ("baseline", "OP2q", "OP2x", "OP2s", "O2x", "P")
# protocol v4: pipelined operand fetch (same-warp packing, fresh cycle
# after a warp switch or for a reuse-served consumer) in front
CANDIDATES_V4 = ("baseline", "OP2p", "OP2q", "OP2x", "P")
CANDIDATES = CANDIDATES_V2
# intended GPR read counts per variant (v3 probe); reported as
# form_mismatch when the compiled window differs
INTENDED_READS_V3 = {"dep3r": 3, "dep3n": 3, "dep2": 2, "dep1": 1,
                     "ilp8": 3, "ilp8n": 3, "ilp8n2": 2, "ilp2": 3,
                     "imad": 3}
MEASURED_TOL = 0.05   # evidence: max error <= 5 % on both runs individually
ABS_TOL = 0.25        # cycles per instruction


# ----------------------------------------------------------------- data

def load_rows(path: Path) -> dict:
    rows = {}
    for f in path.glob("*.jsonl"):
        for line in f.open():
            r = json.loads(line)
            name, wn = r["variant"].rsplit(".w", 1)
            window = "cta" if name.endswith(".cta") else "warp"
            v = name[:-4] if window == "cta" else name
            rows[(v, window, int(wn), r["u"])] = r["samples"]
    return rows


def ols(xs, ys):
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    sxx = sum((x - xm) ** 2 for x in xs)
    b = sxy / sxx
    a = ym - b * xm
    resid = max(abs(y - (a + b * x)) / y for x, y in zip(xs, ys))
    return b, a, resid


# ----------------------------------------------------------------- tree

class Tree:
    def __init__(self, path: Path):
        t = json.load(open(path))
        self.res = {r["id"]: r for r in t["resources"]}
        self.res_by_name = {r["name"]: r["id"] for r in t["resources"]}
        self.leaves = {}
        pre = "instruction.service.root/predicated_on=1/operation_class="
        for n in t["nodes"]:
            i = n.get("id", "")
            if i.startswith(pre):
                self.leaves[i[len(pre):]] = n.get("parameter", {}).get(
                    "fields", {})

    @staticmethod
    def _f(fields, name, default):
        m = (fields.get(name) or {}).get("model", {})
        return m.get("value", default) if m.get("kind") == "constant" \
            else default

    def cls_of(self, inst: dict) -> str:
        op = inst["opcode"]
        if op == "CS2R":
            return "CS2R.CLOCK" if "CS2R.CLOCK" in self.leaves else "CS2R"
        # unsupported placeholder nodes (e.g. MUFU.EX2 on h100) carry no
        # fields: fall through to the root class
        for cand in (op, op.split(".")[0]):
            if self.leaves.get(cand):
                return cand
        raise KeyError(f"no instruction.service leaf for {op}")

    def service(self, inst: dict) -> dict:
        # the two CS2R clock reads are ordinary 1-cycle instructions in
        # the replica (protocol "derivation"), not the CS2R service leaf
        if inst["opcode"] == "CS2R":
            return {"L": 1.0, "ii": 1.0, "res": []}
        f = self.leaves[self.cls_of(inst)]
        res = self._f(f, "resource", None)
        ig = self._f(f, "issue_group_resource", None)
        return {"L": float(self._f(f, "result_latency_cycles", 1.0)),
                "ii": float(self._f(f, "issue_interval_cycles", 1.0)),
                "res": [r for r in (res, ig)
                        if isinstance(r, (int, float)) and r >= 0
                        and r != 0xFFFFFFFF]}


# -------------------------------------------------------------- replica

def spacing(interval: float, residual: float):
    """engine accumulate_spacing (fractional initiation intervals)."""
    if not math.isfinite(interval) or interval <= 1.0:
        return 1, 0.0
    total = interval + residual
    sp = max(1.0, math.floor(total + 1e-12))
    residual = total - sp
    if abs(residual) < 1e-12:
        residual = 0.0
    return int(sp), residual


def gpr_reads(inst: dict) -> list:
    """distinct general-purpose source registers (slot index kept for the
    reuse rule); a 64-bit pair contributes both registers."""
    seen, out = set(), []
    for s, src in enumerate(inst["srcs"]):
        if src.get("kind") != "gpr" or src.get("reg") is None:
            continue
        regs = (src["reg"], src["reg"] + 1) if src.get("pair") \
            else (src["reg"],)
        for r in regs:
            if r not in seen:
                seen.add(r)
                out.append((s, r, src["reg"]))
    return out


class Replica:
    def __init__(self, tree: Tree, rule: str, program: list, W: int):
        self.tree, self.rule, self.prog, self.W = tree, rule, program, W
        self.svc = [tree.service(i) for i in program]
        self.fma = tree.res_by_name.get("fma_smsp")
        self.nb = {"O2x": 2, "O4x": 4, "O2p": 2, "O4p": 4}.get(rule, 0)
        self.per_bank = rule in ("O2p", "O4p")
        # OP2x / OP2s: ceil(unserved GPR reads / 2) on an exclusive
        # collector; OP2s lets the same warp's next instruction issue in
        # the hold's last cycle when its reads fit the spare port slots
        self.ports = 2 if rule in ("OP2q", "OP2x", "OP2s", "OP2p") else 0
        self.share = rule == "OP2s"
        # OP2q: in-order collector behind the issue slot, one waiting
        self.queue = rule in ("OP2q", "OP2p")
        # OP2p: fetch slots at half-cycle resolution; the same warp's
        # next unserved instruction continues in the spare slot, a
        # served consumer or another warp starts on a fresh cycle
        self.pack = rule == "OP2p"
        # per-resource copies: (res, copy) -> [next_free, residual]
        self.units = defaultdict(lambda: [0, 0.0])

    def copies(self, res: int) -> int:
        if self.rule == "P" and res == self.fma:
            return 2
        return int(self.tree.res[res].get("copy_count", 1) or 1)

    def unit_ii(self, res: int, ii: float) -> float:
        if self.rule == "P" and res == self.fma:
            return 2.0
        # addendum 2026-11-06: intervals <= 1.1 are the thr harness's
        # rounding of a 1 / cycle rate; a fractional hole would switch
        # warps in the replica for no mechanism reason
        return 1.0 if ii <= 1.1 else ii

    def run(self):
        W, prog, svc = self.W, self.prog, self.svc
        n = len(prog)
        pc = [0] * W
        floor = [0] * W                 # previous issue + stall
        reg_ready = [defaultdict(int) for _ in range(W)]
        bar_free = [defaultdict(int) for _ in range(W)]
        first = [None] * W
        last = [None] * W
        cursor = 0
        last_issue = None               # (warp, pc) of the SMSP's last issue
        coll_busy = 0                   # exclusive collector
        coll_spare = (0, None)          # (cycle, warp) with free port slots
        svc_start = 0                   # OP2q: last entry's service start
        svc_end = 0                     # OP2q: last entry's service end
        slot_end = 0                    # OP2p: next free read slot (2/cycle)
        slot_warp = None                # OP2p: warp owning the open cycle
        bank_busy = defaultdict(int)    # per-bank collector
        t = 0
        done = 0
        guard = 0
        while done < W:
            issued = False
            for k in range(W):
                w = (cursor + k) % W
                i = pc[w]
                if i >= n:
                    continue
                inst, s = prog[i], svc[i]
                if floor[w] > t:
                    continue
                if any(reg_ready[w][r] > t for _, r, _ in gpr_reads(inst)):
                    continue
                if inst.get("wait") and any(
                        bar_free[w][b] > t for b in range(6)
                        if inst["wait"] >> b & 1):
                    continue
                # class units: earliest admitting copy (P: copy >= 1 only
                # right after this warp's own previous instruction)
                picks = []
                blocked = False
                for res in s["res"]:
                    best = None
                    for c in range(self.copies(res)):
                        if c >= 1 and self.rule == "P" and res == self.fma \
                                and last_issue != (w, i - 1):
                            continue
                        u = self.units[(res, c)]
                        if u[0] <= t and (best is None):
                            best = c
                    if best is None:
                        blocked = True
                        break
                    picks.append((res, best))
                if blocked:
                    continue
                # operand collector (O rules)
                cost, banks, served = 0, {}, set()
                if self.nb or self.ports:
                    prev = prog[i - 1] if i >= 1 else None
                    if prev is not None and last_issue == (w, i - 1):
                        for slot, src in enumerate(prev["srcs"]):
                            if src.get("reuse") and src.get("kind") == "gpr" \
                                    and slot < len(inst["srcs"]) \
                                    and inst["srcs"][slot].get("kind") == "gpr" \
                                    and inst["srcs"][slot].get("reg") == src.get("reg"):
                                served.add(slot)
                    nreads = 0
                    for slot, r, _ in gpr_reads(inst):
                        if slot in served:
                            continue
                        nreads += 1
                        if self.nb:
                            banks[r % self.nb] = banks.get(r % self.nb, 0) + 1
                    if self.ports:
                        cost = -(-nreads // self.ports)
                    else:
                        cost = max(banks.values()) if banks else 0
                    if self.per_bank:
                        if any(bank_busy[b] > t for b in banks):
                            continue
                    elif self.queue:
                        if svc_start > t:
                            continue
                    elif coll_busy > t:
                        # OP2s: own follower in the hold's last cycle
                        # using the spare slots (hold 2 for 3 reads
                        # leaves 1 slot in its second cycle)
                        if not (self.share and coll_busy == t + 1
                                and coll_spare[0] == t
                                and coll_spare[1] == w
                                and last_issue == (w, i - 1)
                                and nreads <= 1 and nreads > 0):
                            continue
                # ---- issue
                issued = True
                for res, c in picks:
                    u = self.units[(res, c)]
                    sp, u[1] = spacing(self.unit_ii(res, s["ii"]), u[1])
                    u[0] = t + sp
                if self.nb or self.ports:
                    if self.per_bank:
                        for b, kreads in banks.items():
                            bank_busy[b] = t + kreads
                    elif self.pack:
                        if nreads:
                            fresh = slot_warp != w or served or slot_end % 2 == 0
                            begin = max(2 * t, slot_end if not fresh
                                        else 2 * ((slot_end + 1) // 2))
                            svc_start = begin // 2
                            slot_end = begin + nreads
                            svc_end = (slot_end + 1) // 2
                            slot_warp = w
                    elif self.queue:
                        if cost:
                            svc_start = max(t, svc_end)
                            svc_end = svc_start + cost
                    elif cost:
                        if coll_busy > t:
                            # shared slot: the hold is unchanged
                            coll_spare = (0, None)
                        else:
                            coll_busy = t + cost
                            spare = cost * self.ports - nreads \
                                if self.ports else 0
                            coll_spare = (t + cost - 1, w) if spare > 0 \
                                else (0, None)
                floor[w] = t + max(1, inst["stall"])
                if inst.get("dst_reg") is not None:
                    reg_ready[w][inst["dst_reg"]] = t + int(s["L"])
                if inst.get("wbar", 7) != 7:
                    bar_free[w][inst["wbar"]] = t + int(s["L"])
                if inst.get("rbar", 7) != 7:
                    bar_free[w][inst["rbar"]] = t
                if first[w] is None:
                    first[w] = t
                last[w] = t
                last_issue = (w, i)
                pc[w] += 1
                if pc[w] >= n:
                    done += 1
                # yield_hint = 1 - raw bit 4 advances the cursor
                cursor = (w + 1) % W if not inst.get("yield_raw", 0) else w
                break
            t += 1
            guard += 1
            if guard > 10_000_000:
                raise RuntimeError("replica did not terminate")
        per_warp = [l - f for f, l in zip(first, last)]
        return statistics.median(per_warp), max(last) - min(first)


def program_for(checks: dict, variant: str, u: int) -> list:
    """the U kernel's own window (its register allocation decides the
    banks); CS2R bounds kept.  Falls back to the U = 256 prefix when the
    gate recorded only that kernel (gate checks prefix_ok)."""
    own = checks.get(f"{variant}/u{u}", {}).get("program")
    if own:
        return [dict(i) for i in own]
    full = checks[f"{variant}/u256"]["program"]
    return [dict(full[0])] + [dict(i) for i in full[1:1 + u]] + [dict(full[-1])]


def dst_reg(inst: dict):
    d = inst.get("dst")
    if not d or not d.startswith("R") or d.startswith("RZ"):
        return None
    num = "".join(ch for ch in d[1:].split(".")[0] if ch.isdigit())
    return int(num) if num else None


def predict(tree: Tree, checks: dict, variant: str, W: int, rule: str):
    tw, tc = [], []
    for u in US:
        prog = program_for(checks, variant, u)
        for inst in prog:
            inst["dst_reg"] = dst_reg(inst)
        a, b = Replica(tree, rule, prog, W).run()
        tw.append(a)
        tc.append(b)
    return ols(US, tw)[0], ols(US, tc)[0]


# ----------------------------------------------------------------- main

def main() -> int:
    global CANDIDATES
    v3 = "--v3" in sys.argv or "--v4" in sys.argv
    if "--v4" in sys.argv:
        CANDIDATES = CANDIDATES_V4
    elif v3:
        CANDIDATES = CANDIDATES_V3
    A, dev, run = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    base = A / "microbench" / "dispatch_contention" / f"{dev}-{run}"
    mdir = A / "microbench" / "dispatch_contention-model" / f"{dev}-{run}"
    runs = {r: load_rows(base / r) for r in ("run-a", "run-b")
            if (base / r).is_dir()}
    checks = json.load(open(mdir / "checks.json"))
    tree = Tree(Path(__file__).resolve().parents[3] / "configs" / "trees"
                / f"{dev}.json")
    cells, excluded = {}, []
    for v in VARIANTS:
        ws = (1,) if v in MUFU else WS
        gate = checks.get(f"{v}/u256", {})
        # prefix_ok is recorded, not gating: the replica simulates each
        # U kernel's own program, so a form difference between the U
        # kernels is reproduced rather than averaged away
        gate_ok = all(checks.get(f"{v}/u{u}", {}).get("ok") for u in US) \
            and all(checks.get(f"{v}/u{u}", {}).get("chained", True)
                    for u in US)
        for w in ws:
            key = f"{v}/w{w}"
            meas = {}
            for window in ("warp", "cta"):
                sl, rs = {}, {}
                for rname, rows in runs.items():
                    Ts = [statistics.median(rows[(v, window, w, u)])
                          if rows.get((v, window, w, u)) else None
                          for u in US]
                    if any(x is None for x in Ts):
                        continue
                    b, _, resid = ols(US, Ts)
                    sl[rname], rs[rname] = b, resid
                if len(sl) < 2:
                    meas[window] = None
                    continue
                sa, sb = sl["run-a"], sl["run-b"]
                agree = abs(sa - sb) / ((sa + sb) / 2) if sa + sb else 0.0
                # protocol: agreement <= 3 % or <= 1 cycle per window
                agree_ok = agree <= 0.03 or abs(sa - sb) * US[-1] <= 1.0
                meas[window] = {"slope": (sa + sb) / 2, "runs": [sa, sb],
                                "resid": max(rs.values()), "agree": agree,
                                "usable": max(rs.values()) <= 0.02
                                and agree_ok}
            # v3: a greedy cell (ilp8 / ilp8n / ilp8n2, W >= 2) whose
            # per-warp window is bimodal gates on its CTA slope alone
            cta_only = v3 and v in ("ilp8", "ilp8n", "ilp8n2") and w >= 2 \
                and meas.get("warp") and not meas["warp"]["usable"] \
                and meas["warp"]["resid"] > 0.02
            usable = gate_ok and all(m and m["usable"] for m in meas.values()) \
                or (cta_only and meas.get("cta") and meas["cta"]["usable"])
            entry = {"measured": meas, "gate_ok": gate_ok, "usable": usable,
                     "cta_only": bool(cta_only),
                     "gating": usable and w in GATE_W.get(v, ()),
                     "stalls": gate.get("stalls"), "pred": {}, "err": {},
                     "pass": {}}
            if v3 and v in INTENDED_READS_V3 and gate_ok:
                body = checks[f"{v}/u256"]["program"][1:-1]
                got = sorted({len(gpr_reads(i)) for i in body
                              if i["opcode"].split(".")[0] != "FMNMX"})
                if got != [INTENDED_READS_V3[v]]:
                    entry["form_mismatch"] = {"intended": INTENDED_READS_V3[v],
                                              "compiled": got}
            if gate_ok and meas.get("warp"):
                for rule in CANDIDATES:
                    try:
                        pw, pc_ = predict(tree, checks, v, w, rule)
                    except KeyError as e:
                        entry["pred"][rule] = str(e)
                        continue
                    entry["pred"][rule] = {"warp": pw, "cta": pc_}
                    errs, ok = {}, True
                    for window, p in (("warp", pw), ("cta", pc_)):
                        m = meas.get(window)
                        if not m:
                            ok = False
                            continue
                        if cta_only and window == "warp":
                            errs[window] = abs(p - m["slope"])
                            continue
                        e = abs(p - m["slope"])
                        errs[window] = e
                        tol = max(MEASURED_TOL * m["slope"], ABS_TOL)
                        ok = ok and e <= tol
                        # per-run strict 5 % for the evidence grade
                        errs[window + "_runs_within_5pct"] = all(
                            abs(p - r) <= MEASURED_TOL * r for r in m["runs"])
                    entry["err"][rule] = errs
                    entry["pass"][rule] = ok
            cells[key] = entry
            if not usable:
                excluded.append(key)
    gating = [k for k, c in cells.items() if c["gating"]]
    accepted = []
    for rule in CANDIDATES:
        if gating and all(cells[k]["pass"].get(rule) for k in gating):
            accepted.append(rule)
    decision = accepted[0] if accepted else None
    evidence = None
    if decision:
        strict = all((cells[k]["cta_only"]
                      or (cells[k]["err"][decision].get("warp_runs_within_5pct")
                          and cells[k]["measured"]["warp"]["agree"] <= 0.03))
                     and cells[k]["err"][decision].get("cta_runs_within_5pct")
                     and cells[k]["measured"]["cta"]["agree"] <= 0.03
                     for k in gating)
        evidence = "measured" if strict else "experimental"
    worst = {rule: max((max(cells[k]["err"][rule].get("warp", 0),
                            cells[k]["err"][rule].get("cta", 0))
                        for k in gating if rule in cells[k]["err"]),
                       default=None) for rule in CANDIDATES}
    out = {"protocol": "dispatch_contention v2", "device": dev, "run": run,
           "gating_cells": gating, "excluded": excluded,
           "accepted": accepted, "decision": decision,
           "evidence": evidence, "worst_error_cycles": worst,
           "cells": cells}
    json.dump(out, open(mdir / "model.json", "w"), indent=1)
    print(f"{dev}-{run}: decision {decision} ({evidence}); accepted "
          f"{accepted}; gating {len(gating)} cells; excluded {excluded}")
    hdr = f"{'cell':14s} {'meas':>6s} " + " ".join(f"{r:>8s}" for r in CANDIDATES)
    print(hdr)
    for k, c in cells.items():
        m = c["measured"].get("warp")
        ms = f"{m['slope']:6.2f}" if m else "   n/a"
        ps = " ".join(
            f"{c['pred'][r]['warp']:8.2f}" if isinstance(c["pred"].get(r), dict)
            else f"{'-':>8s}" for r in CANDIDATES)
        flag = "" if c["gating"] else ("  (record)" if c["usable"]
                                       else "  (excluded)")
        if c.get("cta_only"):
            flag += " cta-only"
        if c.get("form_mismatch"):
            flag += f" form_mismatch{c['form_mismatch']['compiled']}"
        print(f"{k:14s} {ms} {ps}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
