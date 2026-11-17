#!/usr/bin/env python3
"""port_pairs derivation (protocol-v3.json).

  model_v3.py --run <dir> --l2-port <B/cycle> --store-port <B/cycle> --loader-ref <B/cycle> --out <json>

For every pair (B in store / stg / sts / fence / full, A in tma / ldgsts / ldg):
  slow = P_B(with A) / P_B(alone), keep = R_A(with B) / R_A(alone),
  U = 1 / slow + R_A(with B) / l2_port.
"""
import argparse, json, statistics
from pathlib import Path


def load(path):
    rows = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[d["family"]] = d
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--l2-port", type=float, required=True)
    ap.add_argument("--store-port", type=float, required=True)
    ap.add_argument("--loader-ref", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {r: load(Path(a.run) / f"run-{r}.jsonl") for r in ("a", "b")}
    out = {"run": a.run, "inputs": vars(a), "per_run": {}, "checks": {}}
    for r, rows in runs.items():
        def get(mode, kind, loaders, epi=1):
            for d in rows.values():
                if d["mode"] == mode and d["lkind"] == kind and d["loaders"] == loaders and d["epi"] == epi:
                    return d
            return None
        res = {"alone": {}, "loader_alone": {}, "pairs": {}}
        for m in ("store", "stg", "sts", "fence", "full"):
            d = get(m, "tma", 0)
            res["alone"][m] = d["period_median"] if d else None
        for k in ("tma", "ldgsts", "ldg"):
            d = get("store", k, 4, epi=0)
            res["loader_alone"][k] = d["load_bytes_per_cycle_median"] if d else None
        for m in ("store", "stg", "sts", "fence", "full"):
            for k in ("tma", "ldgsts", "ldg"):
                d = get(m, k, 4)
                if not d or not res["alone"][m] or not res["loader_alone"][k]:
                    continue
                slow = d["period_median"] / res["alone"][m]
                rate = d["load_bytes_per_cycle_median"]
                res["pairs"][f"{m}+{k}"] = {"period": d["period_median"], "slow": slow, "load": rate,
                                            "keep": rate / res["loader_alone"][k], "U": 1.0 / slow + rate / a.l2_port}
        out["per_run"][r] = res
    ra = out["per_run"]["a"]
    ck = out["checks"]
    if ra["alone"].get("store"):
        want = 8192.0 / a.store_port
        ck["control_store_drain"] = {"measured": ra["alone"]["store"], "pred": want,
                                     "ok": abs(ra["alone"]["store"] - want) <= 0.05 * want}
    if ra["loader_alone"].get("tma"):
        ck["control_loader"] = {"measured": ra["loader_alone"]["tma"], "ref": a.loader_ref,
                                "ok": abs(ra["loader_alone"]["tma"] - a.loader_ref) <= 0.15 * a.loader_ref}
    ck["controls_ok"] = bool(ck.get("control_store_drain", {}).get("ok") and ck.get("control_loader", {}).get("ok"))
    verdict = {}
    for p in ra["pairs"]:
        vs = []
        for r in ("a", "b"):
            x = out["per_run"][r]["pairs"].get(p)
            if not x:
                continue
            if x["slow"] >= 1.10 and 0.90 <= x["U"] <= 1.10:
                vs.append("share")
            elif x["slow"] <= 1.05 and x["keep"] >= 0.95:
                vs.append("independent")
            else:
                vs.append("partial")
        verdict[p] = vs[0] if len(set(vs)) == 1 else "disagree:" + "/".join(vs)
    ck["verdict"] = verdict
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(ck, indent=1))
    for r in ("a", "b"):
        res = out["per_run"][r]
        print(f"run {r}: alone {res['alone']} loader_alone {res['loader_alone']}")
        for p, x in res["pairs"].items():
            print("  %-14s period %7.1f slow %.3f load %6.1f keep %.3f U %.3f" % (p, x["period"], x["slow"], x["load"], x["keep"], x["U"]))


if __name__ == "__main__":
    main()
