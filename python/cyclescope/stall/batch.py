"""Batch driver of ``cyclescope stall compare`` / ``cyclescope stall summarize``."""
from __future__ import annotations

import json
from pathlib import Path

from .class_mapping import load_mapping
from .compare import compare_case, load_kernel_function
from .hardware import load_hardware
from .ledger import load_ledger
from .summarize import load_results, write_summary

LEDGER_RELATIVE = Path("out-cpp") / "stall.tsv"


def read_cases(spec: str) -> list[str]:
    """Case id, comma list, or a file with one case per line."""
    path = Path(spec)
    if path.is_file():
        return [line.strip() for line in path.read_text().splitlines()
                if line.strip()]
    return [c for c in spec.split(",") if c]


def run_compare(cases: list[str], ledger_root: Path, reference_root: Path,
                captures: Path, out: Path, mapping_path: Path | None = None,
                repeat_root: Path | None = None, scope: str = "kernel"
                ) -> dict[str, int]:
    """Writes <out>/cases/<case>.json per case.  A case whose inputs are
    missing or unusable is recorded with its failure string, never
    skipped silently."""
    mapping = load_mapping(mapping_path)
    target = Path(out) / "cases"
    target.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, int] = {}
    for case in cases:
        try:
            ledger_path = Path(ledger_root) / case / LEDGER_RELATIVE
            if not ledger_path.is_file():
                raise ValueError(f"missing_fact:stall_ledger ({ledger_path})")
            reference = Path(reference_root) / case
            for name in ("timing.json", "ncu_metrics.csv", "ncu_source.csv"):
                if not (reference / name).is_file():
                    raise ValueError(
                        f"missing_fact:reference:{name} ({reference})")
            ledger = load_ledger(ledger_path)
            hardware = load_hardware(reference)
            function = load_kernel_function(Path(captures) / case, ledger)
            repeat = None
            if repeat_root is not None and all(
                    (Path(repeat_root) / case / name).is_file()
                    for name in ("timing.json", "ncu_metrics.csv",
                                 "ncu_source.csv")):
                repeat = load_hardware(Path(repeat_root) / case)
                if repeat.kernel != hardware.kernel:
                    repeat = None
            result = compare_case(case, hardware, ledger, function, mapping,
                                  repeat=repeat, scope=scope)
        except ValueError as error:
            result = {"case": case, "status": str(error)}
        (target / f"{case}.json").write_text(json.dumps(result, indent=1))
        key = result["status"].split(" (")[0]
        statuses[key] = statuses.get(key, 0) + 1
    return statuses


def run_summarize(root: Path, label: str) -> dict:
    results = load_results(Path(root) / "cases")
    return write_summary(results, Path(root), label)
