#!/usr/bin/env python3
"""Generate per-case capture invocations for the CUTLASS cluster-SM suite.

Deterministic and CPU-only: for every case row of the suite manifest
(``manifests/cutlass_cluster_sm100.json``) the qualified case's shard
record is resolved through ``harness/capture_cluster_sm.py``
(``find_invocation_record`` -> ``resolve_case_inputs`` ->
``plan_capture_arguments``) and turned into one
``hopper.capture-case-invocation.v1`` document whose argv is the qualified
binary itself plus its same-source CLI — a pure run-once invocation: the
delivery env keeps the ``CUTLASS_DELIVERY_TRACE`` window protocol (warmup
and correctness launches stay outside the single profiled launch) and no
NVBit variable is declared anywhere.  The capture path is then exclusively::

    python3 suites/common/capture_case.py \\
        --suite-manifest manifests/cutlass_cluster_sm100.json \\
        --case-id <case_id> --output <capture_dir>

Outputs:
  - ``scripts/invocations/<case_id>.json``: standalone invocation documents
    (``--case-id`` restricts generation to one case);
  - ``--merge``: adds ONLY an ``invocation`` key to each case row of the
    suite manifest.  Existing fields are never touched; a pre-existing
    conflicting invocation object fails closed (regenerating an identical
    one is a no-op).

The qualified binary/SASS/source paths recorded in the invocation are
identity, not rebuildable inputs: replaying a case requires them to stay at
their recorded paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = SUITE_ROOT / "harness"
for path in (HARNESS_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import capture_cluster_sm as harness  # noqa: E402

INVOCATION_SCHEMA = "hopper.capture-case-invocation.v1"
DEFAULT_MANIFEST = SUITE_ROOT / "manifests/cutlass_cluster_sm100.json"
DEFAULT_OUT_DIR = SUITE_ROOT / "scripts/invocations"
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
    qualification_root: Path,
    case_id: str,
    *,
    artifact_root: Path | None,
    source_root: Path,
    tma_recorder: Path,
) -> dict:
    plan = harness.resolve_plan(
        case_id=case_id,
        qualification_root=qualification_root,
        artifact_root=artifact_root,
        source_root=source_root,
    )
    invocation = harness.build_invocation(
        case_id, plan["record"], plan["inputs"], tma_recorder=tma_recorder
    )
    invocation["tracer_root"] = str(tracer_root_path())
    return invocation


def generate(
    *,
    manifest_path: Path,
    qualification_root: Path,
    artifact_root: Path | None,
    source_root: Path,
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
                qualification_root,
                row_id,
                artifact_root=artifact_root,
                source_root=source_root,
                tma_recorder=tma_recorder,
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
    merged = 0
    if merge:
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
                "suite": "cutlass",
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
            "Derive one hopper.capture-case-invocation.v1 per CUTLASS "
            "cluster-SM suite-manifest case from its qualification shard "
            "record (deterministic, CPU-only)"
        ),
    )
    parser.add_argument("--suite-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=harness.DEFAULT_QUALIFICATION_ROOT,
        help="cluster-SM qualification tree holding cutlass/qualification.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="CUTLASS qualification artifact root "
        "(default <qualification-root>/cutlass)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SUITE_ROOT,
        help="checkout root holding the CUTLASS harness/upstream sources",
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
        artifact_root=args.artifact_root.resolve() if args.artifact_root else None,
        source_root=args.source_root.resolve(),
        out_dir=args.out_dir.resolve(),
        case_id=args.case_id,
        tma_recorder=tma_recorder,
        merge=args.merge,
    )


if __name__ == "__main__":
    raise SystemExit(main())
