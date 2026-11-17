#!/usr/bin/env python3
"""Resolve one qualified CUTLASS cluster-SM case; audit finished captures.

Stage-6 cutover: this module no longer orchestrates a capture.  The only
real capture entry is
``hoppersim capture via suites/common/capture_case.py``,
driven by ``suites/common/capture_case.py`` with the per-case
invocation that ``scripts/gen_invocations.py`` derives from the
qualification shards through this module's resolvers.  Two responsibilities
remain here:

- deterministic plan resolution: the qualified case's exact binary, its
  CUTLASS source and the per-case CLI (``find_invocation_record`` /
  ``resolve_case_inputs`` / ``plan_capture_arguments``) plus the pure
  run-once invocation object built from them (``build_invocation`` — argv
  is the qualification binary itself; the delivery env keeps the
  ``CUTLASS_DELIVERY_TRACE`` window protocol; no NVBit variable is set
  here);
- the independent dynamic clock-marker audit
  (``dynamic_clock_marker_audit``) over a finished capture directory,
  runnable via ``--audit-capture`` once the capture exists.

A missing input is never rebuilt here: capture identity must stay the exact
binary the qualification timed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = Path(__file__).resolve().parent
COMMON_ROOT = PROJECT_ROOT / "suites/common"
for path in (COMMON_ROOT, HARNESS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from group_inputs import case_has_forbidden_sm_limit, resolve_case_args  # noqa: E402


DEFAULT_QUALIFICATION_ROOT = Path(
    "/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920"
)
DENSE_RUNNER_DEFAULT_WARMUPS = 3
CASE_ID_SUFFIX = "_cluster_sm"
AUDIT_REPORT_NAME = "dynamic_clock_marker_audit.json"


def load_case(qualification_path: Path, case_id: str) -> dict:
    payload = json.loads(qualification_path.read_text(encoding="utf-8"))
    matches = [
        dict(row)
        for row in payload["cases"]
        if str(row.get("source_case_id") or "") == case_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one qualified CUTLASS case {case_id!r} in {qualification_path}, "
            f"found {len(matches)}"
        )
    case = matches[0]
    if case.get("status") != "qualified":
        raise RuntimeError(
            f"{case_id}: qualification status is {case.get('status')!r}, not 'qualified'"
        )
    if int(case.get("target_ctas") or 0) not in (1, 2):
        raise RuntimeError(f"{case_id}: unsupported target_ctas {case.get('target_ctas')!r}")
    return case


def find_invocation_record(artifact_root: Path, case: dict) -> dict:
    """Recover the per-case invocation row from the qualification shards."""

    wanted = f"{case['source_case_id']}_cluster_sm"
    matches: list[dict] = []
    unparsable: list[str] = []
    for attempt in case.get("attempts") or []:
        shard_dir = Path(str(attempt)).parent
        if not shard_dir.is_dir():
            raise RuntimeError(
                f"{case['source_case_id']}: qualification shard dir is missing: "
                f"{shard_dir}"
            )
        for summary_path in sorted(shard_dir.glob("summary_shard*.jsonl")):
            for line in summary_path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    # A shard writer artifact (e.g. a stray trailing brace)
                    # must not hide the wanted record, but it is reported
                    # if the record never shows up.
                    unparsable.append(str(summary_path))
                    continue
                if isinstance(record, dict) and str(record.get("case_id") or "") == wanted:
                    matches.append(record)
    unique: dict[str, dict] = {}
    for record in matches:
        key = json.dumps(record, sort_keys=True)
        unique.setdefault(key, record)
    if len(unique) != 1:
        detail = (
            f"summary lines failed to parse in: {sorted(set(unparsable))}"
            if unparsable and not matches
            else f"records disagree or are duplicated: {len(unique)}"
        )
        raise RuntimeError(
            f"{case['source_case_id']}: expected exactly one {wanted} shard record "
            f"({detail})"
        )
    record = next(iter(unique.values()))
    if str(record.get("delivery_mode") or "") != "cluster_sm":
        raise RuntimeError(
            f"{case['source_case_id']}: shard record delivery_mode is "
            f"{record.get('delivery_mode')!r}, not cluster_sm"
        )
    if case_has_forbidden_sm_limit(record):
        raise RuntimeError(
            f"{case['source_case_id']}: artificial SM limit is not allowed"
        )
    return record


def resolve_case_inputs(
    case_id: str, record: dict, *, artifact_root: Path, source_root: Path
) -> dict[str, Path]:
    """Resolve the qualified binary/SASS/source trio, or report the gap.

    A missing input is never rebuilt here: capture identity must stay the
    exact binary the qualification timed, so the caller is told to rebuild
    it with the same build script before capturing.
    """

    binary = Path(str(record.get("binary") or "")) if record.get("binary") else (
        artifact_root / "build" / str(record.get("binary_name") or "")
    )
    sass = binary.with_suffix(".sass")
    source = source_root / str(record.get("source") or "")
    missing = [
        (label, path, reason)
        for label, path, reason in (
            (
                "binary",
                binary,
                "rebuild it with the qualification build script "
                "(same source, same flags) before capturing",
            ),
            (
                "sass",
                sass,
                "dump it from the same qualified binary "
                "(cuobjdump -sass) before capturing",
            ),
            (
                "source",
                source,
                "restore the CUTLASS source the qualification built from",
            ),
        )
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        details = "\n".join(
            f"  {label}: {path} ({reason})" for label, path, reason in missing
        )
        raise RuntimeError(
            f"{case_id}: qualified capture inputs are missing on disk:\n{details}"
        )
    return {"binary": binary, "sass": sass, "source": source}


def plan_capture_arguments(
    case_id: str, record: dict
) -> tuple[list[str], dict[str, str]]:
    """Same-source CLI plus delivery env for exactly one profiled launch."""

    kind = str(record.get("kind") or "")
    if kind not in ("dense_harness", "upstream_example"):
        raise RuntimeError(f"{case_id}: unsupported CUTLASS case kind: {kind!r}")
    raw_args = [str(value) for value in record.get("args") or []]
    warmups: int | None = None
    kept: list[str] = []
    for argument in raw_args:
        if argument.startswith("--trials="):
            continue
        if argument.startswith("--warmups=") or argument.startswith("--warmup="):
            warmups = max(0, int(argument.split("=", 1)[1]))
            continue
        kept.append(argument)
    args = resolve_case_args(kept, source_root=SUITE_ROOT)
    if kind == "dense_harness":
        # CUTLASS_DELIVERY_TRACE requires exactly one measured trial; the
        # runner's own warmup loop precedes the single captured launch.
        args.extend(
            [
                f"--warmups={DENSE_RUNNER_DEFAULT_WARMUPS if warmups is None else warmups}",
                "--trials=1",
            ]
        )
    environment = {
        "CUTLASS_DELIVERY_TRACE": "1",
        "CUTLASS_DELIVERY_CASE_ID": case_id,
        "CUTLASS_DELIVERY_ALGORITHM": str(record.get("algorithm") or "unknown"),
        "CUTLASS_DELIVERY_SAMPLE_COUNT": "1",
        "CUTLASS_DELIVERY_MODE": "cluster_sm",
        "GEMM_DELIVERY_GPU": "h100",
    }
    if int(record.get("cluster_sm_count") or 0) > 0:
        environment["CUTLASS_CLUSTER_SM_COUNT"] = str(int(record["cluster_sm_count"]))
    if record.get("compiled_cluster"):
        environment["CUTLASS_CLUSTER_SHAPE"] = str(record["compiled_cluster"])
    if record.get("post_mode"):
        environment["CUTLASS_DELIVERY_POST_MODE"] = str(record["post_mode"])
    if kind == "dense_harness":
        # Dense binaries resolve cuTensorMapEncodeTiled through the PLT, so
        # the direct-driver preload records it; upstream examples reach
        # encode through dlsym/cuGetProcAddress and are covered by the
        # LD_PRELOAD recorder alone.
        environment["HOPPER_TENSOR_MAP_DIRECT_DRIVER_PRELOAD"] = "1"
    else:
        # The official examples launch their correctness/reference kernels;
        # the wrapper's host-only entry initializes and launches the exact
        # selected CUTLASS GEMM once inside the profiler range instead.
        environment["CUTLASS_DELIVERY_TRACE_ONLY"] = "1"
    return args, environment


def build_invocation(
    case_id: str,
    record: dict,
    inputs: dict[str, Path],
    *,
    tma_recorder: Path | None,
) -> dict:
    """The case's pure run-once invocation for the unified capture entry.

    argv is the qualified binary itself plus the same-source CLI; env is the
    delivery env of :func:`plan_capture_arguments` (the runner's
    ``CUTLASS_DELIVERY_TRACE`` window brackets exactly one target launch —
    warmup and correctness launches stay outside the profiler range).  No
    NVBit variable is declared here: the capture entry owns injection.
    """

    args, environment = plan_capture_arguments(case_id, record)
    invocation: dict = {
        "argv": [str(inputs["binary"]), *args],
        "program_file": str(inputs["binary"]),
        "source_file": str(inputs["source"]),
        "env": environment,
        "kernel_range": "",
        "carveout": "auto",
        "marker_contract": {"clock_marker_policy": "outermost_pair"},
    }
    if tma_recorder is not None:
        invocation["tma_recorder"] = str(tma_recorder)
    return invocation


def primary_from_capture_metadata(
    case_dir: Path, kernel_file: str | None
) -> dict:
    """The audited launch, selected explicitly from the capture's own
    inventory (``capture_metadata.json`` written by the capture adapter):
    exactly one traced launch, or the one named by ``--kernel-file``.
    Never by name tokens."""

    metadata = json.loads(
        (case_dir / "capture_metadata.json").read_text(encoding="utf-8")
    )
    launches = (metadata.get("manifest_case") or {}).get("traced_kernels") or []
    rows = [
        row
        for row in launches
        if isinstance(row, dict)
        and (kernel_file is None or str(row.get("kernel_file") or "") == kernel_file)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"{case_dir}: expected exactly one traced launch"
            f"{' matching kernel_file ' + kernel_file if kernel_file else ''}, "
            f"found {len(rows)} of {len(launches)}; pass --kernel-file to "
            "select explicitly"
        )
    return rows[0]


def dynamic_clock_marker_audit(case_dir: Path, primary: dict) -> dict:
    """Require trace and timing to execute the same outer marker interval."""

    static_path = case_dir / "simulation_static_sass.json.gz"
    with gzip.open(static_path, "rt", encoding="utf-8") as handle:
        static = json.load(handle)
    rows = static.get("static_sass_map.csv")
    if not isinstance(rows, list):
        raise RuntimeError(f"{case_dir.name}: static SASS table is missing")
    marker_pcs = sorted({
        int(str(row["pc_offset"]), 16)
        for row in rows
        if row.get("function_name") == primary["kernel_name"]
        and str(row.get("opcode_root") or row.get("opcode") or "").upper() == "CS2R"
        and "SR_CLOCKLO" in str(row.get("operands") or row.get("sass") or "")
    })
    if len(marker_pcs) < 2:
        raise RuntimeError(
            f"{case_dir.name}: selected CUTLASS kernel has fewer than two clock markers"
        )
    start_pc, end_pc = marker_pcs[0], marker_pcs[-1]
    trace_name = Path(str(primary["kernel_file"])).with_suffix(".traceg").name
    trace_path = case_dir / trace_name
    if not trace_path.is_file():
        raise RuntimeError(f"{case_dir.name}: selected dynamic trace is missing")
    warps: dict[tuple[str, int], list[int]] = {}
    cta = ""
    warp: int | None = None
    with trace_path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("thread block ="):
                cta = stripped.split("=", 1)[1].strip()
                warp = None
                continue
            if stripped.startswith("warp ="):
                warp = int(stripped.split("=", 1)[1])
                warps.setdefault((cta, warp), [])
                continue
            if warp is None or not stripped or stripped[0] not in "0123456789abcdefABCDEF":
                continue
            first = stripped.split(None, 1)[0]
            try:
                pc = int(first, 16)
            except ValueError:
                continue
            if pc in {start_pc, end_pc}:
                warps[(cta, warp)].append(pc)
    if not warps:
        raise RuntimeError(f"{case_dir.name}: selected dynamic trace has no warps")
    failures = {
        f"{cta}:warp{warp}": {
            "start_count": pcs.count(start_pc),
            "end_count": pcs.count(end_pc),
        }
        for (cta, warp), pcs in sorted(warps.items())
        if pcs.count(start_pc) != 1 or pcs.count(end_pc) != 1
    }
    if failures:
        raise RuntimeError(
            f"{case_dir.name}: trace/timing marker control flow differs: {failures}"
        )
    return {
        "schema_version": "delivery.dynamic-clock-marker-audit.v1",
        "status": "passed",
        "selection_policy": "outermost_pair",
        "kernel_file": primary["kernel_file"],
        "kernel_name": primary["kernel_name"],
        "static_clock_marker_pcs": [f"{pc:04x}" for pc in marker_pcs],
        "selected_start_pc": f"{start_pc:04x}",
        "selected_end_pc": f"{end_pc:04x}",
        "dynamic_warp_count": len(warps),
        "start_execution_count": len(warps),
        "end_execution_count": len(warps),
    }


def resolve_plan(
    *,
    case_id: str,
    qualification_root: Path,
    artifact_root: Path | None,
    source_root: Path,
) -> dict:
    qualification_path = qualification_root / "cutlass" / "qualification.json"
    source_case_id = (
        case_id[: -len(CASE_ID_SUFFIX)]
        if case_id.endswith(CASE_ID_SUFFIX)
        else case_id
    )
    case = load_case(qualification_path, source_case_id)
    artifact_root = (
        artifact_root.resolve() if artifact_root else qualification_root / "cutlass"
    )
    record = find_invocation_record(artifact_root, case)
    inputs = resolve_case_inputs(
        case_id, record, artifact_root=artifact_root, source_root=source_root
    )
    return {"case": case, "record": record, "inputs": inputs}


def audit_capture(case_dir: Path, kernel_file: str | None) -> int:
    primary = primary_from_capture_metadata(case_dir, kernel_file)
    report = dynamic_clock_marker_audit(case_dir, primary)
    (case_dir / AUDIT_REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve one qualified CUTLASS cluster-SM case into its pure "
            "run-once invocation, or audit a finished capture's dynamic "
            "clock markers"
        )
    )
    parser.add_argument("--case-id", default=None)
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=DEFAULT_QUALIFICATION_ROOT,
        help="cluster-SM qualification tree holding cutlass/qualification.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="CUTLASS qualification artifact root (default <qualification-root>/cutlass)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SUITE_ROOT,
        help="checkout root holding the CUTLASS harness/upstream sources",
    )
    parser.add_argument(
        "--tma-recorder",
        type=Path,
        default=None,
        help="TMA tensor-map recorder recorded in the planned invocation "
        "($HOPPER_TMA_RECORDER or the 296-flow artifact default)",
    )
    parser.add_argument(
        "--audit-capture",
        type=Path,
        default=None,
        help="audit this finished capture directory instead of resolving a plan",
    )
    parser.add_argument(
        "--kernel-file",
        default=None,
        help="explicit kernel_file to audit when the capture holds several "
        "traced launches",
    )
    args = parser.parse_args()

    if args.audit_capture is not None:
        try:
            return audit_capture(args.audit_capture.resolve(), args.kernel_file)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"capture_cluster_sm: {exc}", file=sys.stderr)
            return 4

    if not args.case_id:
        parser.error("--case-id is required (or pass --audit-capture DIR)")

    tma_recorder = args.tma_recorder
    if tma_recorder is None:
        override = os.environ.get("HOPPER_TMA_RECORDER", "")
        if override:
            tma_recorder = Path(override)
        else:
            tma_recorder = Path(
                os.environ.get(
                    "HOPPER_ARTIFACT_ROOT", "/path/to/hopper-prototype/artifacts"
                )
            ) / "cluster_sm_296_flow_20260921/common/libtma_tensor_map_recorder.so"
    if not tma_recorder.is_file():
        print(
            f"capture_cluster_sm: TMA recorder not found: {tma_recorder} "
            "(pass --tma-recorder or set HOPPER_TMA_RECORDER)",
            file=sys.stderr,
        )
        return 3
    try:
        plan = resolve_plan(
            case_id=args.case_id,
            qualification_root=args.qualification_root.resolve(),
            artifact_root=args.artifact_root,
            source_root=args.source_root.resolve(),
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"capture_cluster_sm: {exc}", file=sys.stderr)
        return 3
    record, inputs, case = plan["record"], plan["inputs"], plan["case"]
    binary_args, environment = plan_capture_arguments(args.case_id, record)
    print(
        json.dumps(
            {
                "status": "planned",
                "case_id": args.case_id,
                "binary": str(inputs["binary"]),
                "source": str(inputs["source"]),
                "sass": str(inputs["sass"]),
                "binary_args": binary_args,
                "env": environment,
                "invocation": build_invocation(
                    args.case_id, record, inputs, tma_recorder=tma_recorder
                ),
                "qualification_case": {
                    key: case.get(key)
                    for key in ("source_case_id", "target_ctas", "uses_cluster", "status")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
