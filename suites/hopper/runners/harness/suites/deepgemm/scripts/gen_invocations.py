#!/usr/bin/env python3
"""Generate per-case capture invocations for the DeepGEMM cluster-SM suite.

Deterministic and CPU-only: for every case row of the suite manifest
(``manifests/deepgemm_cluster_sm100.json``) the qualified case's final
formal attempt is resolved through ``harness/trace_cluster_sm_case.py``
(ENV/CMD header of the formal run.log) and turned into one
``hopper.capture-case-invocation.v1`` document whose argv is the pure
run-once program ``harness/run_once_case.py`` plus the case's identity
arguments.  The capture path is then exclusively::

    python3 suites/common/capture_case.py \\
        --suite-manifest manifests/deepgemm_cluster_sm100.json \\
        --case-id <case_id> --output <capture_dir>

(no library-name resolution anywhere: the invocation carries the real
program, and the formal ENV/CMD replay happens inside the run-once
program).

Outputs:
  - ``scripts/invocations/<case_id>.json``: standalone invocation documents
    (``--case-id`` restricts generation to one case);
  - ``--merge``: adds ONLY an ``invocation`` key to each case row of the
    suite manifest.  Existing fields are never touched; a pre-existing
    conflicting invocation object fails closed (regenerating an identical
    one is a no-op).

The qualification tree path recorded in argv is identity, not a rebuildable
input: replaying a case requires that tree to stay at its recorded path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = SUITE_ROOT / "harness"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from trace_cluster_sm_case import (  # noqa: E402
    DEFAULT_QUALIFICATION_ROOT,
    formal_attempt,
    load_case,
    parse_formal_invocation,
    plan_trace_invocation,
)

INVOCATION_SCHEMA = "hopper.capture-case-invocation.v1"
DEFAULT_MANIFEST = SUITE_ROOT / "manifests/deepgemm_cluster_sm100.json"
DEFAULT_OUT_DIR = SUITE_ROOT / "scripts/invocations"
RUN_ONCE_PROGRAM = HARNESS_ROOT / "run_once_case.py"
# Plan-time placeholder: plan_trace_invocation only builds redirect strings
# from it, so this validates the formal command without a capture directory.
PLAN_OUTPUT_DIR = Path("<capture-dir>")
DEFAULT_TMA_RECORDER_RELATIVE = (
    "cluster_sm_296_flow_20260921/common/libtma_tensor_map_recorder.so"
)


def default_tma_recorder() -> Path:
    override = os.environ.get("HOPPER_TMA_RECORDER", "") or os.environ.get(
        "CLUSTER_SM_TMA_RECORDER", ""
    )
    if override:
        return Path(override)
    artifacts = Path(
        os.environ.get(
            "HOPPER_ARTIFACT_ROOT", "/path/to/hopper-prototype/artifacts"
        )
    )
    return artifacts / DEFAULT_TMA_RECORDER_RELATIVE


def tracer_root_path() -> Path:
    """NVBit tracer root directory shared across all suites (overrides:
    ``HOPPER_TRACER_ROOT``, then the campaign-local
    ``CLUSTER_SM_TRACER_ROOT``).  The invocation stores the root directory
    itself — exactly what capture_binary's ``--tracer-root`` expects; the
    adapter validates the ``tracer_tool/`` layout under it at capture
    time.  Existence is a capture-time check."""
    override = os.environ.get("HOPPER_TRACER_ROOT", "") or os.environ.get(
        "CLUSTER_SM_TRACER_ROOT", ""
    )
    if override:
        return Path(override).resolve()
    artifacts = Path(
        os.environ.get(
            "HOPPER_ARTIFACT_ROOT", "/path/to/hopper-prototype/artifacts"
        )
    )
    return artifacts / "cluster_sm_296_flow_20260921/common/tracer_root"


def build_invocation(
    qualification_root: Path, case_id: str, *, tma_recorder: Path,
) -> dict:
    """One case's pure run-once invocation, resolved from the qualification."""
    qualification_path = qualification_root / "deepgemm" / "qualification.json"
    case = load_case(qualification_path, case_id)
    attempt = formal_attempt(case)
    formal_env, command = parse_formal_invocation(Path(attempt["log"]))
    # Validate the formal command (runner form, manifest fallback, package)
    # at generation time; the redirected argv itself is discarded — the
    # run-once program rebuilds it against its real work directory.
    plan_trace_invocation(
        case, command, formal_env, output_dir=PLAN_OUTPUT_DIR
    )
    env = dict(formal_env)
    env.setdefault("GEMM_DELIVERY_GPU", "h100")
    return {
        "argv": [
            str(RUN_ONCE_PROGRAM),
            "--case-id",
            case_id,
            "--qualification-root",
            str(qualification_root.resolve()),
        ],
        "program_file": str(RUN_ONCE_PROGRAM),
        "cwd": str(HARNESS_ROOT),
        "env": env,
        # The formal runner replays torch setup (randn/elementwise on
        # multi-thousand-CTA grids) before the deep_gemm launches; tracing
        # all of it produced a 22.9GB dynamic_mask sidecar in the r7 smoke
        # (~115M rows, hours of enrichment).  Default kernel_range is the
        # tracer's native name-pattern form ("@<regex>" = every launch whose
        # kernel name fully matches): the pattern selects the DeepGEMM
        # library symbol namespace, so all real deep_gemm launches
        # (warmups + measured) stay traced as complete units while torch
        # initialization kernels are excluded — no launch ordinals, no
        # grid/cluster changes.  HOPPER_DG_KERNEL_RANGE overrides for
        # special cases.
        "kernel_range": os.environ.get("HOPPER_DG_KERNEL_RANGE",
                                       "@.*deep_gemm.*"),
        "carveout": "auto",
        "allow_missing_lineinfo": True,
        "tma_recorder": str(tma_recorder),
        "tracer_root": str(tracer_root_path()),
        "marker_contract": {"clock_marker_policy": "outermost_pair"},
    }


def generate(
    *,
    manifest_path: Path,
    qualification_root: Path,
    out_dir: Path,
    case_id: str | None,
    tma_recorder: Path,
    merge: bool,
) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("cases") or []
        if isinstance(row, dict)
        and (case_id is None or str(row.get("case_id")) == case_id)
    ]
    if not rows:
        print(
            f"gen_invocations: no suite-manifest case matches {case_id!r} "
            f"in {manifest_path}",
            file=sys.stderr,
        )
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    invocations: dict[str, dict] = {}
    for row in rows:
        row_id = str(row["case_id"])
        try:
            invocations[row_id] = build_invocation(
                qualification_root, row_id, tma_recorder=tma_recorder
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"gen_invocations: {row_id}: {exc}", file=sys.stderr)
            return 3
        document = {
            "schema_version": INVOCATION_SCHEMA,
            "case_id": row_id,
            "invocation": invocations[row_id],
        }
        (out_dir / f"{row_id}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if merge:
        merged = 0
        for row in payload["cases"]:
            row_id = str(row.get("case_id"))
            if row_id not in invocations:
                continue
            existing = row.get("invocation")
            if existing is not None and existing != invocations[row_id]:
                print(
                    f"gen_invocations: {row_id}: suite manifest already "
                    "carries a different invocation object; refusing to "
                    "overwrite (only additive merges are allowed)",
                    file=sys.stderr,
                )
                return 3
            if existing is None:
                row["invocation"] = invocations[row_id]
                merged += 1
        manifest_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": "generated",
                "suite": "deepgemm",
                "manifest": str(manifest_path),
                "cases": sorted(invocations),
                "written": sorted(invocations),
                "out_dir": str(out_dir),
                "merged_invocation_keys": merged if merge else 0,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gen_invocations",
        description=(
            "Derive one hopper.capture-case-invocation.v1 per DeepGEMM "
            "cluster-SM suite-manifest case from its qualified formal "
            "invocation (deterministic, CPU-only)"
        ),
    )
    parser.add_argument("--suite-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=DEFAULT_QUALIFICATION_ROOT,
        help="cluster-SM qualification tree holding deepgemm/qualification.json",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--case-id", default=None, help="restrict generation to one case"
    )
    parser.add_argument(
        "--tma-recorder",
        type=Path,
        default=None,
        help="TMA tensor-map recorder library ($HOPPER_TMA_RECORDER or the "
        "296-flow artifact default)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="add the invocation object to each generated suite-manifest "
        "case row (only the invocation key; never any existing field)",
    )
    args = parser.parse_args(argv)
    tma_recorder = args.tma_recorder or default_tma_recorder()
    if not tma_recorder.is_file():
        print(
            f"gen_invocations: TMA recorder not found: {tma_recorder} "
            "(pass --tma-recorder or set HOPPER_TMA_RECORDER)",
            file=sys.stderr,
        )
        return 3
    return generate(
        manifest_path=args.suite_manifest.resolve(),
        qualification_root=args.qualification_root.resolve(),
        out_dir=args.out_dir.resolve(),
        case_id=args.case_id,
        tma_recorder=tma_recorder,
        merge=args.merge,
    )


if __name__ == "__main__":
    raise SystemExit(main())
