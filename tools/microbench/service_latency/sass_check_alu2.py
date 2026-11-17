#!/usr/bin/env python3
"""sass_check_alu2 — composition-exact window gate for the alu2 uniform
cells (protocol-v3 addendum 2, rule 2).

alu2 kernels: k_ulat<O,U> (one warp, U dependent uniform ops) and
k_uthr<O,K> (wps warps x 8 uniform chains x K steps).  The window must
hold exactly U / 8K x composition of the declared U-classes; harness
constants only otherwise.
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

from sass_check_alu4 import HARNESS, service_classes  # noqa: E402

# op name -> declared per-step composition (uniform-datapath classes)
COMP = {
    "IADD3": {"UIADD3": 1}, "IMAD": {"UIMAD": 1}, "LOP3": {"ULOP3": 1},
    "SHF": {"USHF": 1}, "PRMT": {"UPRMT": 1}, "POPC": {"UPOPC": 1},
    "FLO": {"UFLO": 1}, "SEL": {"USEL": 1},
    "ISETP_SEL": {"UISETP": 1, "USEL": 1},
    "LDC": {"ULDC": 1},
}

_NAME = re.compile(
    r"^_Z\d+(k_ulat|k_uthr)I(\d+)([A-Z0-9_]+)Li(\d+)E.*$")


def demangle(name):
    m = _NAME.match(name)
    if not m:
        return None
    return m.group(1), m.group(3), int(m.group(4))


def widest_window(k):
    ws = sc.clock_windows(k)
    return max(ws, key=len) if ws else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["hopper", "blackwell"], required=True)
    ap.add_argument("--tree", required=True)
    a = ap.parse_args()

    rules = importlib.import_module(f"cyclescope.arch.{a.arch}.rules")
    svc = service_classes(a.tree)

    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v3", "probe": "alu2",
           "sass": a.sass, "arch": a.arch,
           "kernels": {}, "unclassified": []}
    for name, k in kernels.items():
        d = demangle(name)
        if d is None:
            out["unclassified"].append(name)
            continue
        kind, cell, u = d
        mode = "ulat" if kind == "k_ulat" else "uthr"
        comp = COMP.get(cell)
        res = {"key": f"alu2/{mode}.{cell}/{u}", "op": f"{mode}.{cell}",
               "mode": mode, "u": u, "cell": cell, "kernel": name,
               "credit": None, "ok": False, "why": []}
        if comp is None:
            res["why"].append(f"no declared composition for {cell}")
            out["kernels"][res["key"]] = res
            continue
        steps = u if kind == "k_ulat" else 8 * u
        res["composition"] = comp
        w = widest_window(k)
        if not w:
            res["why"].append("no clock window")
            out["kernels"][res["key"]] = res
            continue
        counts = Counter()
        raw = Counter()
        for i in w:
            raw[i.opcode] += 1
            cred = None
            try:
                classes = rules.operation_classes(i)
            except Exception:
                classes = [i.opcode.split(".")[0]]
            for cls in classes:
                if cls in svc:
                    cred = cls
                    break
            counts[cred or i.opcode.split(".")[0]] += 1
        res["sass_counts"] = dict(counts)
        res["raw_counts"] = dict(raw)
        for cls, n in comp.items():
            got = counts.get(cls, 0)
            if got != n * steps:
                res["why"].append(f"{cls} count {got} != {n}x{steps}")
        stray = {c: n for c, n in counts.items()
                 if c not in comp and c not in HARNESS}
        if stray:
            res["why"].append(
                f"non-harness instructions in the window: {stray}")
        if not res["why"]:
            res["credit"] = next(iter(comp))
        res["ok"] = not res["why"]
        out["kernels"][res["key"]] = res
    # addendum: per-window constants must not scale with U/K — for each
    # (cell, mode), every class outside the composition must have
    # max-min count <= 2 across the U/K kernels, else all fail
    groups = {}
    for r in out["kernels"].values():
        groups.setdefault((r["cell"], r["mode"]), []).append(r)
    for (cell, mode), rs in groups.items():
        comp = rs[0].get("composition") or {}
        extra = {}
        for r in rs:
            for cls, n in (r.get("sass_counts") or {}).items():
                if cls not in comp:
                    extra.setdefault(cls, []).append(n)
        bad = {c: sorted(v) for c, v in extra.items()
               if v and max(v) - min(v) > 2}
        if bad:
            for r in rs:
                if "harness scales with U" not in " ".join(r["why"]):
                    r["why"].append(
                        f"harness scales with U: {bad}")
                    r["ok"] = False
                    r["credit"] = None
    n = len(out["kernels"])
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = n - out["n_ok"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check alu2: {out['n_ok']}/{n} gated kernels ok")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"])
    return 0 if n else 2


if __name__ == "__main__":
    sys.exit(main())
