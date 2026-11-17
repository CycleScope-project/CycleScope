#!/usr/bin/env python3
"""Select the main per-call kernel from c450-full NCU timing rows."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SUITE_AUXILIARY_EXCLUSIONS = {
    "cutlass": [],
    "deepgemm": [],
    "flashattention": [],
    "flashmla": [],
    "flashinfer": [],
    "triton": [],
    "cublas": [],
}
# suites whose target kernel IS a vendor math-library kernel (nvjet,
# cutlass, xmma, gemm) -> the vendor exclusion must not apply
SUITES_TARGETING_VENDOR_KERNELS = frozenset({"cublas"})
FREQUENCY_METRIC = "sm__cycles_elapsed.avg.per_second"
CYCLES_METRIC = "sm__cycles_active.max"


def demangle_many(names: list[str]) -> dict[str, str]:
    """mangled -> demangled via c++filt (one process); names c++filt cannot
    demangle (or when c++filt is absent) map to themselves.  The family
    main-kernel regexes are written against demangled names; the tracer
    matches both spellings (launch_selected), so do the pickers."""
    import subprocess
    if not names:
        return {}
    try:
        proc = subprocess.run(["c++filt"], input="\n".join(names) + "\n",
                              capture_output=True, text=True, check=False,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return {n: n for n in names}
    out = proc.stdout.splitlines()
    if proc.returncode != 0 or len(out) != len(names):
        return {n: n for n in names}
    return dict(zip(names, out))


def main_kernel_matches(regex: str, mangled: str,
                        demangled: str | None = None) -> bool:
    """The ONE main-kernel predicate shared by the reference table and
    capture --launches main-last: fullmatch against the mangled or the
    demangled spelling (no loose search fallback)."""
    import re
    if not mangled:
        return False
    if re.fullmatch(regex, mangled):
        return True
    if demangled is not None and demangled != mangled and             re.fullmatch(regex, demangled):
        return True
    return False
VENDOR_NAMESPACE_PREFIXES = (
    "_ZN3cub", "_ZN6cublas", "_ZN8cublasLt", "_ZN5cudnn", "_ZN6curand",
)
VENDOR_UNMANGLED_PREFIXES = (
    "nvjet_", "cutlass3x_", "cutlass_80_", "ampere_", "volta_", "turing_",
    "gemv2T_kernel", "gemvNSP", "gemmSN_", "splitKreduce_kernel",
)
VENDOR_XMMA_PREFIX = re.compile(r"^sm\d+_xmma")


def is_framework_kernel(name: str) -> bool:
    if name.startswith(("_ZN2at", "_ZN14at_cuda_detail", "_ZN3c10", "_ZN5torch")):
        return True
    demangled = name[5:] if name.startswith("void ") else name
    return demangled.startswith(("at::", "at_cuda_detail::", "c10::", "torch::"))


def is_vendor_math_library_kernel(name: str) -> bool:
    if name.startswith(VENDOR_NAMESPACE_PREFIXES):
        return True
    lower = name.lower()
    if "cublas" in lower or "cudnn" in lower:
        return True
    demangled = name[5:] if name.startswith("void ") else name
    return (demangled.startswith(VENDOR_UNMANGLED_PREFIXES)
            or bool(VENDOR_XMMA_PREFIX.match(demangled)))


def load_launch_table(path: Path) -> dict[str, dict]:
    """{case: row} parsed from a suites/<arch>/<set>/launches.tsv fixed
    launch table (header `case kernel occurrence grid block cluster
    source`); {} when the file does not exist."""
    import csv
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(newline="") as fh:
        return {row["case"]: row for row in
                csv.DictReader(fh, delimiter="\t")}


def traced_launch_key(capture_dir: Path) -> tuple[str, int]:
    """(kernel, occurrence) of a capture's traced launch: its mangled name
    and how many launches of exactly that name the tracer saw up to and
    including it (1-based).  The capture must hold one traced launch."""
    rows = [json.loads(l) for l in
            (Path(capture_dir) / "launches.jsonl").read_text().splitlines()
            if l.strip()]
    traced = [r for r in rows if r.get("traced")]
    if len(traced) != 1:
        raise ValueError(f"{capture_dir}: {len(traced)} traced launches")
    t = traced[0]
    occurrence = sum(1 for r in rows if r.get("mangled") == t["mangled"]
                     and int(r["launch"]) <= int(t["launch"]))
    return t["mangled"], occurrence


def census_row(rows: list[dict], kernel: str, occurrence: int,
               name_key: str = "mangled") -> dict | None:
    """The occurrence-th (1-based) row, in launch order, named kernel."""
    seen = 0
    for row in rows:
        if row.get(name_key) == kernel:
            seen += 1
            if seen == occurrence:
                return row
    return None


def selection_for(kernel: str, occurrence: int) -> str:
    """Tracer CYCLESCOPE_CAPTURE_LAUNCHES value tracing the occurrence-th
    launch named exactly kernel."""
    return f"{re.escape(kernel)}@{occurrence}"


def pick_repeat(rows: list[dict], calls: int, suite: str,
                aux_regexes: list[str] | None = None) -> dict:
    counts = Counter(row["mangled"] for row in rows)
    excluded = set(SUITE_AUXILIARY_EXCLUSIONS[suite])
    compiled_aux = [re.compile(pattern) for pattern in (aux_regexes or [])]
    eligible_names = {
        name for name, count in counts.items()
        if count % calls == 0
        and not is_framework_kernel(name)
        and not (suite not in SUITES_TARGETING_VENDOR_KERNELS
                         and is_vendor_math_library_kernel(name))
        and name not in excluded
        and not any(pattern.fullmatch(name) for pattern in compiled_aux)
    }
    eligible_rows = [row for row in rows if row["mangled"] in eligible_names]
    if not eligible_rows:
        return {"error": "no kernel launched once per call"}

    totals = {}
    for row in eligible_rows:
        name = row["mangled"]
        totals[name] = totals.get(name, 0.0) + float(row.get(CYCLES_METRIC) or 0)
    name = max(totals, key=totals.get)
    match = [row for row in rows if row["mangled"] == name]
    index = (calls - 1) * (len(match) // calls)
    selected = match[index]
    return {
        "ordinal": int(selected["ordinal"]),
        "name": name,
        "cycles": float(selected.get(CYCLES_METRIC) or 0),
        "grid": selected.get("grid"),
        "block": selected.get("block"),
        "grid_dims": selected.get("grid_dims"),
        "block_dims": selected.get("block_dims"),
        "cluster": selected.get("cluster"),
    }


def build_pick(rows_by_repeat: dict[int, list[dict]], calls: int, suite: str,
               extra_metrics: list[str], expect_regex: str | None = None,
               aux_regexes: list[str] | None = None) -> dict:
    repeats = []
    for repeat in sorted(rows_by_repeat):
        result = pick_repeat(rows_by_repeat[repeat], calls, suite, aux_regexes)
        repeats.append({"repeat": repeat, "selected": result
                        if "error" not in result else None,
                        **({"error": result["error"]}
                           if "error" in result else {})})

    selected_pairs = [(row["repeat"], row["selected"]) for row in repeats
                      if row.get("selected")]
    selected = [row for _, row in selected_pairs]
    errors = [row.get("error") for row in repeats if row.get("error")]
    if errors:
        error = errors[0]
    elif selected and len({(row["ordinal"], row["name"])
                           for row in selected}) != 1:
        error = "repeats disagree on selected ordinal or kernel name"
    else:
        error = None

    cycles = [row["cycles"] for row in selected]
    result = {
        "suite": suite,
        "calls": calls,
        "repeats": repeats,
        "median": statistics.median(cycles) if cycles else None,
        "min": min(cycles) if cycles else None,
        "max": max(cycles) if cycles else None,
        "spread": ((max(cycles) - min(cycles)) / statistics.median(cycles)
                   if cycles and statistics.median(cycles) else None),
        "metrics": {},
        "kernel_mismatch": False,
    }
    if selected:
        first = selected[0]
        result.update({
            "ordinal": first["ordinal"],
            "name": first["name"],
            "grid": first["grid"],
            "block": first["block"],
            "grid_dims": first["grid_dims"],
            "block_dims": first["block_dims"],
            "cluster": first["cluster"],
        })
    metrics = list(dict.fromkeys([*extra_metrics, FREQUENCY_METRIC]))
    for metric in metrics:
        values = []
        for repeat, row in selected_pairs:
            matching = next(
                (candidate for candidate in rows_by_repeat[repeat]
                 if int(candidate["ordinal"]) == row["ordinal"]
                 and candidate["mangled"] == row["name"]),
                None,
            )
            value = matching.get(metric) if matching else None
            if value is not None:
                values.append(float(value))
        result["metrics"][metric] = statistics.median(values) if values else None
    if expect_regex and selected:
        result["kernel_mismatch"] = not re.fullmatch(expect_regex, selected[0]["name"])
    if error:
        result["error"] = error
    return result


def load_rows(path: Path) -> dict[int, list[dict]]:
    repeats: dict[int, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            repeats[int(row["repeat"])].append(row)
    for rows in repeats.values():
        rows.sort(key=lambda row: int(row["ordinal"]))
    return dict(repeats)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--calls", required=True, type=int)
    parser.add_argument("--suite", required=True,
                        choices=sorted(SUITE_AUXILIARY_EXCLUSIONS))
    parser.add_argument("--expect-regex")
    parser.add_argument("--aux-regex", action="append", default=[])
    args = parser.parse_args()
    if args.calls <= 0:
        parser.error("--calls must be positive")
    case_dir = args.case_dir
    timing_path = case_dir / "ncu_timing.jsonl"
    if timing_path.is_file():
        rows = load_rows(timing_path)
        timing_path_json = case_dir / "timing.json"
        timing = (json.loads(timing_path_json.read_text())
                  if timing_path_json.is_file() else {})
        extra_metrics = timing.get("ncu", {}).get("extra_metrics", [])
        result = build_pick(rows, args.calls, args.suite, extra_metrics,
                            args.expect_regex, args.aux_regex)
    else:
        result = {
            "suite": args.suite, "calls": args.calls, "repeats": [],
            "median": None, "min": None, "max": None, "spread": None,
            "metrics": {}, "kernel_mismatch": bool(args.expect_regex),
            "error": "missing ncu_timing.jsonl",
        }
    (case_dir / "pick.json").write_text(json.dumps(result, indent=1) + "\n")
    print(case_dir / "pick.json")
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
