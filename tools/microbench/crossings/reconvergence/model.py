#!/usr/bin/env python3
"""reconvergence protocol-v2 derivation
(tools/microbench/crossings/reconvergence/protocol-v2.json).

  model.py --device h100|b300 --root $A/microbench/crossings/reconvergence/<device>-<run>
           --out <model.json>

<root>/{run-a,run-b}/ctrl2.jsonl: variants div.bB / uni.bB / div2.bB /
div3.bB at N in {64, 128, 256}; <root>/checks.json: SASS gate keyed
`ctrl/<variant>/<n>` with the window's instruction segment and stalls.
body() / stalls / residual are the blackwell-prototype ctrl_model.py rules
(derive.py), verbatim.  Thresholds copied from the frozen protocol.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

B_SPREAD = 0.10        # D(div.bB) within 10 % of their mean
RUN_AGREE = 0.05       # two runs within 5 %
CONTROL = 0.10         # b300 installed 35.0 within 10 %
INSTALLED = {"b300": 35.0, "h100": None}
BS = (8, 16, 32)


def body(seg):
    """loop body = the instructions from the back-edge's target to the
    back-edge BRA inclusive (addendum 2026-11-05: ctrl2's loops compile to a
    guarded loop -- BSSY B0 / @P BRA guard before the body, back-edge @!P BRA
    to the body start, BSYNC B0 after -- so the predecessor's 'up to the last
    BRA.U' rule would count the once-per-rep guard as per-iteration stalls)."""
    insts = [(int(s[0]) if not isinstance(s[0], str) else int(s[0], 16), s[1], s[2]) for s in seg]
    back = None
    for addr, text, _ in insts:
        toks = text.replace(";", "").split()
        root = toks[1] if toks[0].startswith("@") else toks[0]
        if root.startswith("BRA") and toks[-1].startswith("0x") and int(toks[-1], 16) < addr:
            back = (int(toks[-1], 16), addr)
    if back is None:
        raise ValueError("no backward BRA in the window")
    lo, hi = back
    return [(t, st) for addr, t, st in insts if lo <= addr <= hi]


def body_predecessor(seg):
    """blackwell-prototype ctrl_model.py rule, reported for comparison."""
    insts = [(s[1], s[2]) for s in seg]
    ends = [i for i, (t, _) in enumerate(insts) if t.startswith("BRA.U") and "!" not in t.split(",")[0]]
    if not ends:
        ends = [i for i, (t, _) in enumerate(insts) if (t.split()[1] if t.startswith("@") else t.split()[0]).startswith("BRA")]
    end = max(ends)
    start = 0
    while start < end and insts[start][0].split()[0] in ("LDCU", "NOP"):
        start += 1
    return insts[start:end + 1]


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def derive(run_dir: Path, chk: dict):
    series: dict[str, dict[int, float]] = {}
    for line in (run_dir / ("ctrl2.jsonl" if (run_dir / "ctrl2.jsonl").exists() else "ctrl.jsonl")).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "samples" not in r:
            continue
        series.setdefault(r["variant"], {})[int(r["u"])] = statistics.median(r["samples"])
    out = {}
    for v, d in series.items():
        ns = sorted(d)
        gate = [chk.get(f"ctrl/{v}/{n}", {}) for n in ns]
        ok = bool(gate) and all(g.get("ok") for g in gate)
        per_it = ols(ns, [d[n] for n in ns]) if len(ns) >= 2 else None
        seg = chk.get(f"ctrl/{v}/{ns[0]}", {}).get("segment")
        stalls = resid = stalls_pred = resid_pred = None
        if seg:
            for fn in (body, body_predecessor):
                loop = fn(seg)
                st = sum(s for _, s in loop)
                if v.startswith("uni."):
                    st -= sum(s for t, s in loop if "FFMA" in t) / 2
                if fn is body:
                    stalls, resid = st, (per_it - st if per_it is not None else None)
                else:
                    stalls_pred, resid_pred = st, (per_it - st if per_it is not None else None)
        out[v] = {"per_iteration": per_it, "medians": {str(n): d[n] for n in ns}, "stalls": stalls,
                  "residual": resid, "stalls_predecessor_rule": stalls_pred, "residual_predecessor_rule": resid_pred,
                  "gate_ok": ok, "gate_why": [w for g in gate for w in g.get("why", [])]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    chk = json.loads((a.root / "checks.json").read_text())["ctrl"]["kernels"]
    runs = {r: derive(a.root / r, chk) for r in ("run-a", "run-b")}
    div = {}
    for b in BS:
        v = f"div.b{b}"
        xa, xb = runs["run-a"].get(v), runs["run-b"].get(v)
        if xa and xb and xa["residual"] is not None and xb["residual"] is not None:
            div[b] = {"a": xa["residual"], "b": xb["residual"], "mean": (xa["residual"] + xb["residual"]) / 2,
                      "gate_ok": xa["gate_ok"] and xb["gate_ok"]}
    gate_ok = len(div) == len(BS) and all(d["gate_ok"] for d in div.values())
    D = statistics.fmean(d["mean"] for d in div.values()) if div else None
    spread_ok = D is not None and all(abs(d["mean"] - D) <= B_SPREAD * abs(D) for d in div.values())
    Da = statistics.fmean(d["a"] for d in div.values()) if div else None
    Db = statistics.fmean(d["b"] for d in div.values()) if div else None
    agree = abs(Da - Db) / abs(D) if D else None
    agree_ok = agree is not None and agree <= RUN_AGREE
    verdict = "unrealised" if not gate_ok else "measured" if spread_ok and agree_ok else "experimental"

    def ratio(prefix):
        out = {}
        for b in BS:
            v = f"{prefix}.b{b}"
            xs = [runs[r].get(v, {}).get("residual") for r in ("run-a", "run-b")]
            if all(x is not None for x in xs) and b in div and div[b]["mean"]:
                out[str(b)] = {"residual": statistics.fmean(xs), "ratio_to_div": statistics.fmean(xs) / div[b]["mean"],
                               "gate_ok": all(runs[r][v]["gate_ok"] for r in ("run-a", "run-b"))}
        return out

    uni = {str(b): {"residual": statistics.fmean([runs[r][f"uni.b{b}"]["residual"] for r in ("run-a", "run-b")]),
                    "gate_ok": all(runs[r][f"uni.b{b}"]["gate_ok"] for r in ("run-a", "run-b"))}
           for b in BS if all(f"uni.b{b}" in runs[r] and runs[r][f"uni.b{b}"]["residual"] is not None for r in ("run-a", "run-b"))}
    inst = INSTALLED[a.device]
    control = None if inst is None or D is None else (D - inst) / inst
    model = {
        "cell": "reconvergence", "protocol": "tools/microbench/crossings/reconvergence/protocol-v2.json",
        "device": a.device, "root": str(a.root),
        "div_residual_per_B": {str(b): d for b, d in div.items()}, "slope": D, "intercept": 0.0,
        "run_values": {"a": Da, "b": Db}, "run_agree": agree, "spread_ok": spread_ok, "gate_ok": gate_ok,
        "verdict": verdict,
        "diagnostics": {"div2": ratio("div2"), "div3": ratio("div3"), "uni": uni},
        "installed": inst, "control_rel": control,
        "control": None if control is None else ("reproduced" if abs(control) <= CONTROL else "differs"),
        "variants": runs,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(model, indent=1) + "\n")
    print(a.device, verdict, "D", D, "runs", Da, Db, "div2", {k: round(v["ratio_to_div"], 2) for k, v in model["diagnostics"]["div2"].items()},
          "div3", {k: round(v["ratio_to_div"], 2) for k, v in model["diagnostics"]["div3"].items()},
          "uni", {k: round(v["residual"], 1) for k, v in uni.items()})


if __name__ == "__main__":
    main()
