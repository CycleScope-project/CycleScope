#!/usr/bin/env python3
"""register_operands model (protocol v1): interval = OLS slope/32
(one op per chain per K -> ops_per_K = 1, 8 chains x 16 warps / 16 warps
per SMSP group => divisor 32, recorded per kernel); per-class key
decision (reads, bank_max, or pair) with controls:
b300 FFMA (x,k1,k2) within 10% of installed 1.007;
FADD2 (x,y) within 5% of 2.01.
"""
import argparse
import json
from pathlib import Path


def load(run_dir):
    rows = []
    for jl in Path(run_dir).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def series(rows, cell):
    out = {}
    for r in rows:
        v = str(r.get("variant") or "")
        cell_v = (r.get("params") or {}).get("cell") or v
        if cell_v != cell:
            continue
        out.setdefault(int(r["u"]), []).extend(
            float(x) for x in r["samples"])
    return out


def ols(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) \
        / sum((x - mx) ** 2 for x in xs)
    return b, my - b * mx


def fit(s):
    ks = sorted(s)
    if len(ks) < 2:
        return None
    meds = [sorted(s[k])[len(s[k]) // 2] for k in ks]
    b, a = ols(ks, meds)
    resid = max(abs(m - (a + b * k)) / max(1.0, abs(m))
                for k, m in zip(ks, meds))
    return {"slope": b, "interval": b / 32, "ops_per_K": 1,
            "chains": 8, "warps": 16, "divisor": "slope/32",
            "ks": ks, "medians": meds, "max_rel_resid": resid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--checks", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    checks = json.loads(Path(a.checks).read_text())
    cells = {}
    for key, kr in checks["kernels"].items():
        cell = kr["cell"]
        e = cells.setdefault(cell, {"k": {}, "reads": kr["reads"],
                                    "bank_max": kr["bank_max"],
                                    "mixed": kr["mixed"],
                                    "gate_ok": kr["ok"]})
        e["k"][kr["k"]] = kr["ok"]
    ra = load(root / "run-a")
    rb = load(root / "run-b")
    table = {}
    for cell, e in sorted(cells.items()):
        for run, rows in (("run-a", ra), ("run-b", rb)):
            e[run] = fit(series(rows, f"thr.{cell}"))
        if e["run-a"] and e["run-b"]:
            i1, i2 = e["run-a"]["interval"], e["run-b"]["interval"]
            e["interval"] = (i1 + i2) / 2
            e["agree_rel"] = abs(i1 - i2) / e["interval"] if e["interval"] else None
        e["interval_runs"] = [e["run-a"]["interval"] if e["run-a"] else None,
                              e["run-b"]["interval"] if e["run-b"] else None]
        table[cell] = e

    out = {"protocol": "register_operands v1", "device": a.device,
           "gate": {"n_ok": checks["n_ok"], "n_fail": checks["n_fail"]},
           "kernels": {}, "classes": {}, "controls": {}}
    for cell, e in table.items():
        cls, _, form = cell.rpartition(".")
        cls, _, form2 = cls.partition(".")
        out["kernels"][cell] = {
            "class": cls, "form": form, "perturb": form2,
            "reads": e["reads"], "bank_max": e["bank_max"],
            "mixed": e["mixed"], "gate_ok": e["gate_ok"],
            "interval": e.get("interval"),
            "interval_runs": e["interval_runs"],
            "resid": max(r["max_rel_resid"] for r in
                         (e["run-a"], e["run-b"]) if r) if e.get(
                             "run-a") else None,
            "ops_per_K": 1, "divisor": "slope/32"}

    # per-class key decision
    for cls in ("FFMA", "FFMA2", "FADD2", "IMAD", "FADD"):
        kern = [(n, e) for n, e in out["kernels"].items()
                if e["class"] == cls and e["gate_ok"] and not e["mixed"]
                and e["interval"]]
        verdict = {}
        for key in ("reads", "bank_max"):
            groups = {}
            for n, e in kern:
                groups.setdefault(e[key], []).append(e["interval"])
            verdict[key] = {
                "groups": {str(k): v for k, v in groups.items()},
                "consistent": all(
                    (max(v) - min(v)) / max(v) <= 0.05
                    for v in groups.values() if len(v) > 1)}
        # combined key
        groups = {}
        for n, e in kern:
            groups.setdefault((e["reads"], e["bank_max"]),
                              []).append(e["interval"])
        verdict["pair"] = {
            "groups": {f"{r}|{b}": v for (r, b), v in groups.items()},
            "consistent": all(
                (max(v) - min(v)) / max(v) <= 0.05
                for v in groups.values() if len(v) > 1)}
        accepted = next(
            (k for k in ("reads", "bank_max", "pair")
             if verdict[k]["consistent"]), None)
        out["classes"][cls] = {"verdict": verdict, "accepted_key": accepted}
    # controls (b300)
    if a.device == "b300":
        f32xkk = out["kernels"].get("FFMA_xkk.p0") or out["kernels"].get(
            "FFMA.xkk.p0")
        ffma_ctrl = next((e for n, e in out["kernels"].items()
                          if n.startswith("FFMA.xkk")), None)
        fadd2_ctrl = next((e for n, e in out["kernels"].items()
                           if n.startswith("FADD2.xy")), None)
        c = {}
        if ffma_ctrl and ffma_ctrl["interval"]:
            c["FFMA_xkk_vs_1.007"] = abs(ffma_ctrl["interval"] - 1.007) \
                / 1.007
            c["FFMA_xkk_ok"] = c["FFMA_xkk_vs_1.007"] <= 0.10
        if fadd2_ctrl and fadd2_ctrl["interval"]:
            c["FADD2_xy_vs_2.01"] = abs(fadd2_ctrl["interval"] - 2.01) \
                / 2.01
            c["FADD2_xy_ok"] = c["FADD2_xy_vs_2.01"] <= 0.05
        c["controls_ok"] = c.get("FFMA_xkk_ok", False) and \
            c.get("FADD2_xy_ok", False)
        out["controls"] = c
    Path(a.out).write_text(json.dumps(out, indent=1))
    for cls, d in out["classes"].items():
        print(cls, "accepted:", d["accepted_key"])
        for k, v in d["verdict"].items():
            print("  ", k, v["consistent"], v["groups"])
    print("controls", out.get("controls"))


if __name__ == "__main__":
    main()
