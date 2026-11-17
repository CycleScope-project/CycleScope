#!/usr/bin/env python3
"""port_pairs confirmation derivation (protocol-v4.json).

  model_v4.py --run <dir> --l2-port <B/cycle> --store-port <B/cycle> --loader-ref <B/cycle> --out <json>

Every cell with loaders and the round on is paired with its round-alone cell
(same mode, box bytes, CTA count) and its loader-alone cell (same kind, slot
count, CTA count, region): slow = P(with) / P(alone), keep = R(with) /
R(alone), U = 1 / slow + R(with) / l2_port.
"""
import argparse, json
from pathlib import Path


def load(path):
    rows = []
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--l2-port", type=float, required=True)
    ap.add_argument("--store-port", type=float, required=True)
    ap.add_argument("--loader-ref", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"run": a.run, "inputs": vars(a), "per_run": {}, "checks": {}}
    for r in ("a", "b"):
        rows = load(Path(a.run) / f"run-{r}.jsonl")
        alone = {(d["mode"], d["box_bytes"], d["ctas"]): d["period_median"] for d in rows
                 if d["loaders"] == 0 and d["epi"] == 1}
        lalone = {(d["lkind"], d["loaders"], d["ctas"], d["region_boxes"]): d["load_bytes_per_cycle_median"]
                  for d in rows if d["epi"] == 0}
        pairs = {}
        for d in rows:
            if d["loaders"] == 0 or d["epi"] == 0:
                continue
            pa = alone.get((d["mode"], d["box_bytes"], d["ctas"]))
            la = lalone.get((d["lkind"], d["loaders"], d["ctas"], d["region_boxes"]))
            if not pa or not la:
                continue
            slow = d["period_median"] / pa
            rate = d["load_bytes_per_cycle_median"]
            name = "%s%s+%s%d.N%d.R%d" % (d["mode"], "" if d["mode"] != "store" else str(d["box_bytes"] // 1024) + "k",
                                          d["lkind"], d["loaders"], d["ctas"], d["region_boxes"])
            pairs[name] = {"period": d["period_median"], "alone": pa, "slow": slow, "load": rate, "load_alone": la,
                           "keep": rate / la, "U": 1.0 / slow + rate / a.l2_port}
        out["per_run"][r] = {"alone": {"%s.b%d.N%d" % k: v for k, v in alone.items()},
                             "loader_alone": {"%s%d.N%d.R%d" % k: v for k, v in lalone.items()}, "pairs": pairs}
    ra, rb = out["per_run"]["a"], out["per_run"]["b"]
    ck = out["checks"]
    s8 = ra["alone"].get("store.b8192.N1")
    if s8:
        want = 8192.0 / a.store_port
        ck["control_store_drain"] = {"measured": s8, "pred": want, "ok": abs(s8 - want) <= 0.05 * want}
    l4 = ra["loader_alone"].get("tma4.N1.R64")
    if l4:
        ck["control_loader"] = {"measured": l4, "ref": a.loader_ref, "ok": abs(l4 - a.loader_ref) <= 0.15 * a.loader_ref}
    ck["controls_ok"] = bool(ck.get("control_store_drain", {}).get("ok") and ck.get("control_loader", {}).get("ok"))

    def judge(names):
        res = {}
        for n in names:
            xs = [x["pairs"].get(n) for x in (ra, rb)]
            if not all(xs):
                res[n] = "missing"
            elif min(x["slow"] for x in xs) < 1.03:
                res[n] = "uninformative"
            else:
                res[n] = "confirm" if all(abs(x["U"] - 1.0) <= 0.06 for x in xs) else "refute"
        return res

    # addendum 2026-11-10 (criterion adjusted after data, see protocol-v4.json): port occupancy
    # rho = (store port time + fill port time) / period, with the installed port rates
    def rho(name, x):
        head = name.split("+")[0]
        box = {"store2k": 2048, "store8k": 8192, "store16k": 16384, "stg": 8192}.get(head)
        return None if box is None else (box / a.store_port + x["load"] * x["period"] / a.l2_port) / x["period"]

    def judge_rho(names):
        res = {}
        for n in names:
            xs = [x["pairs"].get(n) for x in (ra, rb)]
            if not all(xs):
                res[n] = "missing"
                continue
            rs = [rho(n, x) for x in xs]
            slowed = min(x["slow"] for x in xs) >= 1.05
            ok = all(r <= 1.03 for r in rs) and (not slowed or all(r >= 0.97 for r in rs))
            res[n] = {"rho": [round(r, 3) for r in rs], "slow": [round(x["slow"], 3) for x in xs],
                      "verdict": ("confirm" if slowed else "consistent") if ok else "refute"}
        return res

    r1 = ["store%dk+tma%d.N1.R64" % (b, l) for b in (2, 8, 16) for l in (1, 2, 4)] + \
         ["stg+tma%d.N1.R64" % l for l in (1, 2, 4)] + ["store8k+ldg4.N1.R64", "stg+ldg4.N1.R64"]
    r2 = ["sts+tma%d.N1.R64" % l for l in (1, 2, 4)] + ["sts+ldg4.N1.R64"]
    ck["R1"] = judge(r1)
    ck["R1_confirmed"] = (sum(v == "confirm" for v in ck["R1"].values()) >= 8 and
                          not any(v in ("refute", "missing") for v in ck["R1"].values()))
    ck["R1_rho"] = judge_rho(r1)
    ck["R1_rho_confirmed"] = (not any(v == "missing" or v["verdict"] == "refute" for v in ck["R1_rho"].values()) and
                              sum(v["verdict"] == "confirm" for v in ck["R1_rho"].values()) >= 8)
    ck["R2"] = judge(r2)
    ck["R2_confirmed"] = (sum(v == "confirm" for v in ck["R2"].values()) >= 2 and
                          not any(v in ("refute", "missing") for v in ck["R2"].values()))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(ck, indent=1))
    for r in ("a", "b"):
        res = out["per_run"][r]
        print(f"run {r}: alone {res['alone']}\n       loader_alone {res['loader_alone']}")
        for p, x in res["pairs"].items():
            print("  %-26s period %7.1f slow %.3f load %6.1f keep %.3f U %.3f" % (p, x["period"], x["slow"], x["load"], x["keep"], x["U"]))


if __name__ == "__main__":
    main()
