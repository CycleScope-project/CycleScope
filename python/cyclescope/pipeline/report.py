"""Run report: status.json rows vs the shipped expected tables.

Acceptance rule for a re-captured case: simulated cycles within
``tolerance`` % of the shipped baseline when the CTA->SM placement is
the same as the baseline capture; cases whose placement differs are
listed separately, without a threshold.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

HEADER = ["suite", "case", "stage", "class", "issued", "sim_cycles",
          "hw_cycles", "ratio", "err_pct", "base_sim", "delta_pct",
          "placement", "verdict"]


def placement_digest(launch_doc: dict) -> str:
    """First 16 hex of sha256 over the cluster->sm placement lines."""
    placements = (launch_doc or {}).get("placements") or {}
    if not placements:
        return ""
    text = "\n".join(f"{c}:{placements[c]['sm_id']}"
                     for c in sorted(placements, key=int))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def accuracy(rows: list[dict]) -> dict:
    """Accuracy summary over done/ok rows carrying hardware cycles.

    rows: dicts with ``sim_cycles`` and ``hw_cycles``.  mape = mean
    |sim/hw - 1| in %, p90_abs_err_pct = the order statistic
    errs[int(0.9 * (n - 1))] of the same (the convention the baseline
    tables were reported with), median_ratio.
    """
    vals = [(float(r["sim_cycles"]) / float(r["hw_cycles"]))
            for r in rows
            if float(r.get("hw_cycles") or 0) > 0
            and float(r.get("sim_cycles") or 0) > 0]
    errs = sorted(abs(x - 1.0) for x in vals)
    n = len(errs)
    if not n:
        return {"cases": 0, "mape_pct": None,
                "p90_abs_err_pct": None, "median_ratio": None}
    p90 = errs[int(0.9 * (n - 1))]
    return {"cases": n, "mape_pct": sum(errs) / n * 100,
            "p90_abs_err_pct": p90 * 100,
            "median_ratio": statistics.median(vals)}


def read_expected(path: Path) -> dict:
    """{case: row} of a shipped expected.tsv."""
    out = {}
    with Path(path).open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["case"]] = r
    return out


def verdict(status: dict, expected_row: dict | None,
            tolerance_pct: float = 3.0) -> str:
    if not (status.get("stage") == "done" and status.get("class") == "ok"):
        return "fail"
    if expected_row is None:
        return "n/a"
    if status.get("placement_digest") and \
            status["placement_digest"] == \
            expected_row.get("placement_digest"):
        base = float(expected_row.get("sim_cycles") or 0)
        sim = float(status.get("cycles") or 0)
        if base > 0 and abs(sim - base) / base * 100 <= tolerance_pct:
            return "ok"
        return "fail"
    return "listed"


def _row(status: dict, expected_row: dict | None,
         tolerance: float) -> list[str]:
    sim = float(status.get("cycles") or 0)
    hw = status.get("hw_cycles") or None
    base = float(expected_row.get("sim_cycles") or 0) \
        if expected_row else 0
    ratio = f"{sim / hw:.3f}" if hw else "-"
    err = f"{abs(sim / hw - 1) * 100:.2f}" if hw else "-"
    delta = (f"{(sim - base) / base * 100:+.2f}" if base else "-")
    dig = status.get("placement_digest") or "-"
    v = verdict(status, expected_row, tolerance)
    return [status["suite"], status["case"], status["stage"],
            status["class"], str(status.get("issued", 0)),
            str(status.get("cycles", 0)),
            f"{hw:.0f}" if hw else "-", ratio, err,
            f"{base:.0f}" if base else "-", delta, dig, v]


def _accuracy_lines(statuses: list[dict]) -> list[str]:
    rows = [{"sim_cycles": st.get("cycles"),
             "hw_cycles": st.get("hw_cycles")}
            for st in statuses
            if st.get("stage") == "done" and st.get("class") == "ok"]
    a = accuracy(rows)
    if not a["cases"]:
        return ["accuracy: no comparable cases"]
    return [f"accuracy: cases={a['cases']} "
            f"mape={a['mape_pct']:.2f}% "
            f"p90_abs_err={a['p90_abs_err_pct']:.1f}% "
            f"median_ratio={a['median_ratio']:.3f}"]


def report(run_outs: list, expected=None, tolerance: float = 3.0,
           out=None) -> int:
    """``run_outs``: directories holding sim/<case>/status.json.
    ``expected``: None -> suites/<suite>/expected.tsv per status suite;
    a path or list of paths -> those tables.  Returns 1 on any fail."""
    if expected is not None and not isinstance(expected, (list, tuple)):
        expected = [expected]
    statuses = []
    for ro in run_outs:
        for p in sorted((Path(ro) / "sim").glob("*/status.json")):
            statuses.append(json.loads(p.read_text()))
    exp: dict[str, dict] = {}
    if expected:
        for f in expected:
            exp.update(read_expected(Path(f)))
    lines = ["\t".join(HEADER)]
    per_suite: dict[str, dict] = {}
    loaded_suites: set = set()
    any_fail = False
    for st in statuses:
        if expected is None and st["suite"] not in loaded_suites:
            loaded_suites.add(st["suite"])
            exp_file = REPO / "suites" / st["suite"] / "expected.tsv"
            if exp_file.is_file():
                exp.update(read_expected(exp_file))
        erow = exp.get(st["case"])
        v = verdict(st, erow, tolerance)
        row = _row(st, erow, tolerance)
        lines.append("\t".join(row))
        s = per_suite.setdefault(st["suite"], {"rows": [], "ok": 0,
                                               "beyond": 0,
                                               "listed": 0,
                                               "listed_max": 0.0})
        s["rows"].append(st)
        if v == "ok":
            s["ok"] += 1
        elif v == "listed":
            s["listed"] += 1
            if erow:
                base = float(erow.get("sim_cycles") or 0)
                if base:
                    s["listed_max"] = max(
                        s["listed_max"],
                        abs(st["cycles"] - base) / base * 100)
        elif v == "fail":
            any_fail = True
            if st.get("stage") == "done" and st.get("class") == "ok":
                s["beyond"] += 1
    for suite, s in per_suite.items():
        lines.append(f"# {suite}")
        lines += ["# " + l for l in _accuracy_lines(s["rows"])]
        lines.append(f"# vs baseline (tolerance {tolerance}%): "
                     f"{s['ok']} within, {s['beyond']} beyond, "
                     f"{s['listed']} listed (max |delta| "
                     f"{s['listed_max']:.2f}%)")
    text = "\n".join(lines) + "\n"
    if out:
        Path(out).write_text(text)
    else:
        print(text, end="")
    return 1 if any_fail else 0


def baseline_summary(expected_file) -> str:
    """Accuracy summary of a shipped expected.tsv alone."""
    rows = [r for r in read_expected(Path(expected_file)).values()
            if r.get("class") == "ok" and r.get("hw_cycles")]
    vals = [{"sim_cycles": r["sim_cycles"],
             "hw_cycles": r["hw_cycles"]} for r in rows]
    a = accuracy(vals)
    if not a["cases"]:
        return f"{expected_file}: no rows\n"
    return (f"{expected_file}: cases={a['cases']} "
            f"mape={a['mape_pct']:.2f}% "
            f"p90_abs_err={a['p90_abs_err_pct']:.1f}% "
            f"median_ratio={a['median_ratio']:.3f}\n")
