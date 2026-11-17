#!/usr/bin/env python3
"""dsmem sass_check (protocol.json sass_gate).

Records the SASS opcode each cluster-window access lowers to in
k_sass_form (peer / own mapa ld and st), then checks the timed kernels:
  k_peer_load<T,U>:  exactly U loads of the recorded load form in the
      window and nothing else that scales with U (harness constants ok);
  k_peer_store<S>:   exactly S stores of the recorded store form before
      the cluster barrier in the window.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

HARNESS = {
    "LDS", "STS", "LD", "ST", "LDC", "ULDC", "LDCU", "BAR", "BRA",
    "NOP", "EXIT", "WARPSYNC", "DEPBAR", "BSSY", "BSYNC", "IADD3",
    "IMAD", "MOV", "UMOV", "LOP3", "ISETP", "SEL", "P2R", "R2P",
    "ULOP3", "UIADD3", "VIADD", "RET", "CALL", "CS2R", "S2R", "S2UR",
    "SYNCS", "CSYNC", "LDG", "STG", "MAPAL", "MAPA", "PRMT", "SHF", "IABS",
}

LOAD_ROOTS = {"LDS", "LD", "LDG", "LDSM", "LDLM", "UTMALDG"}
STORE_ROOTS = {"STS", "ST", "STG", "STSM", "UTMASTG"}


def root(i):
    return i.opcode.split(".")[0]


def windows(k):
    ws = sc.clock_windows(k)
    return ws


def widest(k):
    ws = windows(k)
    return max(ws, key=len) if ws else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kernels = sc.parse(Path(a.sass).read_text())
    out = {"schema": "mechtree.sass_check.v3", "probe": "dsmem",
           "sass": a.sass, "forms": {}, "kernels": {}, "n_ok": 0,
           "n_fail": 0}

    for name, k in kernels.items():
        if "k_sass_form" in name:
            forms = []
            for ins in k.insts:
                r = root(ins)
                if r in LOAD_ROOTS | STORE_ROOTS | {"LDS", "STS"}:
                    forms.append(ins.opcode)
            out["forms"]["k_sass_form"] = forms
            continue
        m = re.match(r"^_Z\d+k_peer_loadILi(\d+)ELi(\d+)E.*$", name)
        if m:
            target, u = int(m.group(1)), int(m.group(2))
            tname = {0: "peer", 1: "own_window", 2: "lds"}[target]
            w = widest(k)
            counts = Counter(root(i) for i in w)
            loads = sum(n for r, n in counts.items() if r in LOAD_ROOTS)
            stray = {r: n for r, n in counts.items()
                     if r not in LOAD_ROOTS | STORE_ROOTS
                     and r not in HARNESS}
            res = {"key": f"lat.{tname}/{u}", "mode": "lat",
                   "cell": f"lat.{tname}", "u": u,
                   "sass_counts": dict(counts),
                   "opcode_forms": sorted({i.opcode for i in w
                                           if root(i) in LOAD_ROOTS}),
                   "ok": True, "why": []}
            if loads != u:
                res["why"].append(f"{loads} loads != U={u}")
                res["ok"] = False
            if stray:
                res["why"].append(f"non-harness: {stray}")
                res["ok"] = False
            out["kernels"][res["key"]] = res
            continue
        m = re.match(r"^_Z\d+k_peer_storeILi(\d+)E.*$", name)
        if m:
            s = int(m.group(1))
            # stores sit before the first cluster barrier; window = all
            # insts of the kernel up to the barrier
            insts = k.insts
            bar = next((i for i, ins in enumerate(insts)
                        if "BARRIER.CLUSTER" in ins.opcode.upper()
                        or "BAR.CLUSTER" in ins.opcode.upper()
                        or "SYNCS.CLUSTER" in ins.opcode.upper()
                        or "CGA" in ins.opcode), len(insts))
            w = insts[:bar]
            counts = Counter(root(i) for i in w)
            peer_forms = [f for f in (out["forms"].get("k_sass_form")
                                      or ["ST.E"])
                          if root(sc.Inst(0, f, 0)) in STORE_ROOTS]
            stores = sum(1 for i in w if i.opcode in peer_forms)
            res = {"key": f"vis.store/{s}", "mode": "vis",
                   "cell": "vis.store", "u": s,
                   "sass_counts": dict(counts),
                   "opcode_forms": sorted({i.opcode for i in w
                                           if root(i) in STORE_ROOTS}),
                   "ok": True, "why": []}
            if stores != s:
                res["why"].append(f"{stores} stores != S={s}")
                res["ok"] = False
            out["kernels"][res["key"]] = res
            continue
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = len(out["kernels"]) - out["n_ok"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check dsmem: {out['n_ok']}/{len(out['kernels'])} ok")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"])
    print("  forms:", out["forms"])
    return 0 if out["kernels"] else 2


if __name__ == "__main__":
    sys.exit(main())
