#!/usr/bin/env python3
"""fp32_mix model: measured slope vs Replica(OP2p) -> excess; k and
decision per protocol.  Usage: model.py <A> <dev> <run>"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

MB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MB / "dispatch_contention"))
import model_v2  # noqa


def main():
    a, dev, run = sys.argv[1], sys.argv[2], sys.argv[3]
    base = Path(a) / "microbench" / "fp32_mix" / f"{dev}-{run}"
    ck = Path(a) / "microbench" / "fp32_mix-model" / f"{dev}-{run}"
    checks = json.load(open(ck / "checks.json"))
    tpath = Path(__file__).resolve().parents[3] / "configs" / "trees"
    tree = model_v2.Tree(tpath / f"{dev}.json")
    # measured slope per (variant, w) in cycles per quartet/unit-op
    per = {"quartet": 4, "trio": 3, "mufu": 4, "fmnmx": 4}
    meas = defaultdict(lambda: defaultdict(dict))
    for rd in sorted(base.glob("run-*")):
        if not rd.is_dir():
            continue
        for ln in (rd / "mix.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if "error" in r or not r.get("samples"):
                continue
            p = r["params"]
            meas[(rd.name, p["variant"], p["w"])][p["u"]] = \
                statistics.median(r["samples"])
    slope = {}
    for (rn, v, w), d in sorted(meas.items()):
        if len(d) < 2 or w != 1 and d is None:
            continue
        us = sorted(d)
        s, _ = model_v2.ols(us, [d[u] for u in us])[:2]
        slope[(v, w)] = s / per[v]
    # replica: run program once per W warps on one SMSP (W warps each
    # issuing the same program; take predicted cycles per unit op)
    print(f"{dev}-{run}: fp32_mix  (cycles per unit-op)")
    out = {"cells": {}, "k": None, "decision": "none"}
    for v in ("quartet", "trio", "mufu", "fmnmx"):
        for w in (1, 2, 3, 4):
            def span(u_):
                ck_ = checks.get(f"{v}/u{u_}")
                prog_ = (ck_ or {}).get("program")
                if not prog_:
                    return None
                insts = [{"opcode": e["opcode"], "dst": e["dst"],
                          "srcs": e["srcs"], "reuse_bits": e["reuse_bits"],
                          "wait": e.get("wait", 0),
                          "stall": e.get("stall", 0)}
                         for e in prog_]
                rep = model_v2.Replica(tree, "OP2p", insts, w)
                try:
                    med, _ = rep.run()
                    return med
                except Exception:
                    return None
            s64, s128 = span(64), span(128)
            pred = None if (s64 is None or s128 is None) \
                else (s128 - s64) / (per[v] * 64.0)
            m = slope.get((v, w))
            exc = (m - pred) if (m is not None and pred) else None
            out["cells"][f"{v}.w{w}"] = {"measured": m, "replica": pred,
                                        "excess": exc}
            if w == 1:
                print(f"  {v:8s} w1: meas={m}")
            print(f"  {v:8s} w{w}: meas={m and round(m,2)} "
                  f"rep={pred and round(pred,2)} "
                  f"excess={exc is not None and round(exc,2)}")
    # k and decision per protocol
    ex = out["cells"]
    ks = []
    cond = True
    for w in (2, 3, 4):
        eq, et = ex[f"quartet.w{w}"]["excess"], ex[f"trio.w{w}"]["excess"]
        mq, mt = ex[f"quartet.w{w}"]["measured"], ex[f"trio.w{w}"]["measured"]
        rq, rt = ex[f"quartet.w{w}"]["replica"], ex[f"trio.w{w}"]["replica"]
        if None in (eq, et, mq, mt, rq, rt):
            cond = False
            continue
        if eq <= max(0.05 * mq, 1.0):
            cond = False
        if et > max(0.10 * mt, 1.0):
            out["decision"] = "unrealised"
            break
        ks.append((eq - et) / (w * 1.0))
    else:
        if cond and len(ks) >= 3:
            med = statistics.median(ks)
            if all(abs(k - med) <= 0.2 * med for k in ks):
                out["k"] = med
                out["decision"] = "install"
    print(f"  k={out['k']} ks={[round(x,2) for x in ks]}"
          f" decision={out['decision']}")
    (ck / "model.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
