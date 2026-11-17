#!/usr/bin/env python3
"""sass_check_alu6 — protocol-v5 gate (v3 composition-exact + harness
scaling, CS2R special windows, b300 FP64 NOP filler rule).

Kernel naming (mangled template instantiations in probe/alu6.cu):
  k_lat<O,U>    lat.<O>          u32 chain (HFMA2)
  k_thr<O,K>    thr.<O>          u32 8-chain (HFMA2)
  k_dlat<O,U>   dlat.<O>         f64 dependent chain
  k_dthr<O,K>   dthr.<O>         f64 8-chain
  k_lat_d64<U>  dlat.DSETP_P     f64 pair
  k_clock<K>    issue.CS2R_CLOCK
  k_gt<K>       issue.CS2R_GT
  k_mlat<D,U>   mlat.MATCH_d<D>
  k_mthr<D,K>   mthr.MATCH_d<D>
"""
import argparse
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

_TREE = Path(__file__).resolve().parents[2].parent

HARNESS = {
    "LDS", "STS", "LD", "ST", "LDC", "ULDC", "LDCU", "BAR", "BRA",
    "NOP", "EXIT", "WARPSYNC", "DEPBAR", "DEPBAR.FIFO", "BSSY", "BSYNC",
    "CS2R", "CS2R.CLOCK", "CS2R.GLOBALTIMER", "CS2R.ZERO", "S2R", "S2UR",
    "IADD3", "UIADD3", "VIADD", "IMAD", "MOV", "UMOV", "LOP3",
    "ISETP", "UISETP", "SEL", "USEL", "P2R", "R2P", "ULOP3",
    "RET", "CALL", "PRET", "RPCMOV", "IMAD.WIDE", "IADD3.X",
    "LDG", "STG", "ULDCU", "PLOP3", "PRMT", "SHF",
}

_NAME = re.compile(
    r"^_Z\d+(k_latx2|k_thrx2|k_lat|k_thr|k_dlat|k_dthr|k_ulat)I(\d+)([A-Z0-9_]+)Li(\d+)E.*$")


def demangle(name):
    m = _NAME.match(name)
    if m:
        kind, cell, u = m.group(1), m.group(3), int(m.group(4))
        if kind == "k_ulat":
            cell = "U" + cell
        return kind, cell, u
    m = re.match(r"^_Z\d+k_lat_d64ILi(\d+)E.*$", name)
    if m:
        return "k_lat_d64", "DSETP_P", int(m.group(1))
    m = re.match(r"^_Z\d+k_clockILi(\d+)E.*$", name)
    if m:
        return "k_clock", "CS2R_CLOCK", int(m.group(1))
    m = re.match(r"^_Z\d+k_gtILi(\d+)E.*$", name)
    if m:
        return "k_gt", "CS2R_GT", int(m.group(1))
    m = re.match(r"^_Z\d+k_mlatILi(\d+)ELi(\d+)E.*$", name)
    if m:
        return "k_mlat", f"MATCH_d{m.group(1)}", int(m.group(2))
    m = re.match(r"^_Z\d+k_mthrILi(\d+)ELi(\d+)E.*$", name)
    if m:
        return "k_mthr", f"MATCH_d{m.group(1)}", int(m.group(2))
    return None



def loop_body(k):
    """Span of the rep loop: between a backward BRA and its target."""
    best = None
    for i, ins in enumerate(k.insts):
        if ins.opcode.split(".")[0] != "BRA":
            continue
        m = re.search(r"0x([0-9a-f]+)", ins.text)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt < ins.addr and (best is None or tgt < best[0]):
            best = (tgt, ins.addr)
    if best is None:
        return k.insts
    lo = next(i for i, x in enumerate(k.insts) if x.addr >= best[0])
    hi = next(i for i, x in enumerate(k.insts) if x.addr >= best[1])
    return k.insts[lo:hi + 1]

def widest_window(k):
    ws = sc.clock_windows(k)
    return max(ws, key=len) if ws else None


def service_classes(tree_path):
    t = json.loads(Path(tree_path).read_text())
    out = set()
    pref = "instruction.service.root/predicated_on=1/operation_class="
    for n in t["nodes"]:
        i = n["id"]
        if i.startswith(pref):
            out.add(i[len(pref):].split("/")[0])
    return out


def cls_of(rules, svc, inst):
    root = inst.opcode.split(".")[0]
    if root == "CS2R":
        t = inst.text.upper()
        if "GLOBALTIM" in t:
            return "CS2R.GLOBALTIMER"
        if "CLOCK" in t:
            return "CS2R.CLOCK"
        return "CS2R"
    try:
        classes = rules.operation_classes(inst)
    except Exception:
        classes = [root]
    for c in classes:
        if c in svc:
            return c
    return root


def nop_filler_stats(w, rules, svc, target_root):
    """Per target op in the window: NOPs that follow it (run length) and
    the stall sum of that run plus the target's own stall.  Returns
    (n_per_step_list, S, counts_per_target) or None if no targets."""
    nop_runs = []          # NOP counts after each target instruction
    S = None
    i = 0
    while i < len(w):
        ins = w[i]
        c = cls_of(rules, svc, ins)
        if c != target_root:
            i += 1
            continue
        n = 0
        s = ins.stall
        j = i + 1
        while j < len(w) and w[j].opcode.split(".")[0] == "NOP":
            n += 1
            s += w[j].stall
            j += 1
        nop_runs.append(n)
        S = s if S is None else S
        i = j
    if not nop_runs:
        return None
    return nop_runs, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["hopper", "blackwell"], required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--cells", default=str(
        Path(__file__).with_name("cells-v6.json")))
    a = ap.parse_args()

    rules = importlib.import_module(f"cyclescope.arch.{a.arch}.rules")
    cells = json.loads(Path(a.cells).read_text())["cells"]
    svc = service_classes(a.tree)
    kernels = sc.parse(Path(a.sass).read_text())

    out = {"schema": "mechtree.sass_check.v3", "probe": "alu7",
           "sass": a.sass, "arch": a.arch,
           "kernels": {}, "unclassified": []}
    for name, k in kernels.items():
        d = demangle(name)
        if d is None:
            out["unclassified"].append(name)
            continue
        kind, cell, u = d
        mode = {"k_lat": "lat", "k_thr": "thr", "k_dlat": "dlat",
                "k_dthr": "dthr", "k_lat_d64": "dlat",
                "k_latx2": "lat", "k_thrx2": "thr",
                "k_ulat": "lat",
                "k_clock": "issue", "k_gt": "issue",
                "k_mlat": "mlat", "k_mthr": "mthr"}[kind]
        spec = cells.get(cell) or cells.get(
            {"DSETP_P": "DSETP"}.get(cell, cell))
        res = {"key": f"alu7/{mode}.{cell}/{u}", "op": f"{mode}.{cell}",
               "mode": mode, "u": u, "cell": cell, "kernel": name,
               "credit": None, "ok": False, "why": []}
        if spec is None:
            res["why"].append(f"cell {cell} not in cells-v6.json")
            out["kernels"][res["key"]] = res
            continue
        comp = spec["composition"]
        res["composition"] = comp

        if kind == "k_clock":
            # rep loop body: exactly K ops of class CS2R.CLOCK
            body = loop_body(k)
            counts = Counter()
            for ins in body:
                counts[cls_of(rules, svc, ins)] += 1
            res["sass_counts"] = dict(counts)
            if counts.get("CS2R.CLOCK", 0) != u:
                res["why"].append(
                    f"CS2R.CLOCK count {counts.get('CS2R.CLOCK',0)} != K={u}")
            stray = {c: n for c, n in counts.items()
                     if c != "CS2R.CLOCK" and c not in HARNESS}
            if stray:
                res["why"].append(f"non-harness: {stray}")
            res["ok"] = not res["why"]
            res["credit"] = "CS2R.CLOCK" if res["ok"] else None
            out["kernels"][res["key"]] = res
            continue
        if kind == "k_gt":
            # between the two clock reads in the rep loop body: exactly
            # K CS2R.GLOBALTIMER
            body = loop_body(k)
            clk = [i for i, ins in enumerate(body)
                   if cls_of(rules, svc, ins) == "CS2R.CLOCK"]
            if len(clk) < 2:
                res["why"].append("fewer than 2 CS2R.CLOCK bounds")
            else:
                lo, hi = clk[0], clk[-1]
                inside = body[lo + 1:hi]
                counts = Counter(cls_of(rules, svc, ins)
                                 for ins in inside)
                res["sass_counts"] = dict(counts)
                if counts.get("CS2R.GLOBALTIMER", 0) != u:
                    res["why"].append(
                        "CS2R.GLOBALTIMER count "
                        f"{counts.get('CS2R.GLOBALTIMER',0)} != K={u}")
                stray = {c: n for c, n in counts.items()
                         if c != "CS2R.GLOBALTIMER" and c not in HARNESS}
                if stray:
                    res["why"].append(f"non-harness in window: {stray}")
            res["ok"] = not res["why"]
            res["credit"] = "CS2R.GLOBALTIMER" if res["ok"] else None
            out["kernels"][res["key"]] = res
            continue

        steps = u
        if kind in ("k_thr", "k_thrx2", "k_dthr", "k_mthr"):
            steps = 8 * u
        if kind in ("k_dlat", "k_lat_d64"):
            steps = 2 * u
        w = widest_window(k)
        if not w:
            res["why"].append("no clock window")
            out["kernels"][res["key"]] = res
            continue
        counts = Counter()
        raw = Counter()
        for ins in w:
            raw[ins.opcode] += 1
            counts[cls_of(rules, svc, ins)] += 1
        res["sass_counts"] = dict(counts)
        res["raw_counts"] = dict(raw)
        for cls, n in comp.items():
            got = counts.get(cls, 0)
            if got != n * steps:
                res["why"].append(f"{cls} count {got} != {n}x{steps}")
        stray = {c: n for c, n in counts.items()
                 if c not in comp and c not in HARNESS}
        if stray:
            res["why"].append(f"non-harness in window: {stray}")
        # FP64 NOP filler rule (b300 cells flagged fp64): NOPs allowed
        # only if the run length after each target op is constant
        if spec.get("fp64"):
            target = spec.get("credits") or next(iter(comp))
            st = nop_filler_stats(w, rules, svc, target)
            if st:
                runs, S = st
                res["nop_per_target"] = sorted(set(runs[:-1]
                                                 if len(runs) > 1
                                                 else runs))
                res["stall_S"] = S
                core = runs[:-1] if len(runs) > 1 else runs
                if len(set(core)) != 1:
                    res["why"].append(
                        f"NOP runs after {target} not constant: {runs}")
        res["window_pcs"] = [hex(w[0].addr), hex(w[-1].addr)]
        if not res["why"]:
            res["credit"] = spec.get("credits") or next(iter(comp))
        res["ok"] = not res["why"]
        out["kernels"][res["key"]] = res

    # harness scaling: for each (cell, mode), classes outside the comp
    # must have max-min <= 2 across U/K (NOP exempt on fp64 cells: its
    # constancy is checked per step above, but total NOP scales with
    # steps by design)
    groups = {}
    for r in out["kernels"].values():
        if r["mode"] in ("lat", "thr", "dlat", "dthr", "mlat", "mthr"):
            groups.setdefault((r["cell"], r["mode"]), []).append(r)
    for (cell, mode), rs in groups.items():
        comp = rs[0].get("composition") or {}
        fp64 = (cells.get(cell) or cells.get(
            {"DSETP_P": "DSETP"}.get(cell, cell), {})).get("fp64")
        extra = {}
        for r in rs:
            for cls, n in (r.get("sass_counts") or {}).items():
                if cls in comp:
                    continue
                if fp64 and cls == "NOP":
                    continue
                extra.setdefault(cls, []).append(n)
        bad = {c: sorted(v) for c, v in extra.items()
               if v and max(v) - min(v) > 2}
        if bad:
            for r in rs:
                if "harness scales" not in " ".join(r["why"]):
                    r["why"].append(f"harness scales with U: {bad}")
                    r["ok"] = False
                    r["credit"] = None
    # CS2R cells: scaling check across K for non-target classes
    for cell, root in (("CS2R_CLOCK", "CS2R.CLOCK"),
                       ("CS2R_GT", "CS2R.GLOBALTIMER")):
        rs = [r for r in out["kernels"].values() if r["cell"] == cell]
        extra = {}
        for r in rs:
            for cls, n in (r.get("sass_counts") or {}).items():
                if cls != root:
                    extra.setdefault(cls, []).append(n)
        bad = {c: sorted(v) for c, v in extra.items()
               if v and max(v) - min(v) > 2}
        if bad:
            for r in rs:
                r["why"].append(f"harness scales with K: {bad}")
                r["ok"] = False
                r["credit"] = None
    n = len(out["kernels"])
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = n - out["n_ok"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check alu6: {out['n_ok']}/{n} gated kernels ok")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"])
    return 0 if n else 2


if __name__ == "__main__":
    sys.exit(main())
