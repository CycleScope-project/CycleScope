#!/usr/bin/env python3
"""Deterministically generate capture invocations for the FlashAttention-3 suite.

One ``hopper.capture-case-invocation.v1`` document per case row of the
cluster-SM manifests: ``fa3_cluster_sm100.json`` and its near-holdout
``holdout_near.json``.  The invocation describes the pure run-once workload
(``harness/run_full_gpu.py --mode capture``: build the case through
full_gpu_ops on the pinned ``fa3_cluster_topo_marker`` build root, warm,
exactly one real op call) for the stage-6 unified capture entry —
``suites/common/capture_case.py`` ->
``hoppersim capture via suites/common/capture_case.py``.
The 2691-line in-harness NVBit capture chain and the shell wrapper that
injected its environment are gone; the tracer/recorder plumbing belongs to
the entry, never to the workload.

The build root resolves exactly like the (deleted) wrapper did:
``FA3_CAPTURE_BINARY`` -> the pinned campaign root
``/path/to/hopper-prototype/artifacts/fa3_cluster_topo_marker_20260921``.
The FA3 run-once needs no env delta: the harness loads the pinned
``flash_attn_3`` interface from the build root's own site-packages.

Outputs: ``scripts/invocations/<manifest_stem>/<case_id>.json`` (default)
and, with ``--merge``, the same invocation object appended to each case row
under a new ``invocation`` key.  ``--merge`` never touches any pre-existing
field (values, formatting and key order are round-trip preserved; an
existing ``invocation`` key is a hard error).

CPU-only and side-effect free (except the output files): paths are resolved
deterministically; nothing imports torch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SUITE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
HARNESS = SUITE_ROOT / "harness" / "run_full_gpu.py"
INVOCATION_SCHEMA = "hopper.capture-case-invocation.v1"
CLOCK_MARKER_POLICY = "outermost_pair"
DEFAULT_MANIFESTS = (
    SUITE_ROOT / "manifests" / "fa3_cluster_sm100.json",
    SUITE_ROOT / "manifests" / "holdout_near.json",
)
DEFAULT_BINARY_ROOT = Path(
    "/path/to/hopper-prototype/artifacts/fa3_cluster_topo_marker_20260921"
)


class GenerationError(RuntimeError):
    """Fail-closed manifest/contract violation."""


def _repo_artifact_root() -> Path:
    """Project root per scripts/paths.sh convention."""
    configured = os.environ.get("HOPPER_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    return PROJECT_ROOT


def pinned_build_root() -> Path:
    """fa3_cluster_topo_marker build root (the deleted wrapper's --binary
    contract: ``FA3_CAPTURE_BINARY`` or the campaign default).  Existence is
    a capture-time check."""
    value = os.environ.get("FA3_CAPTURE_BINARY")
    return Path(value).resolve() if value else DEFAULT_BINARY_ROOT


def tma_recorder_path() -> Path:
    """TMA tensor-map recorder shared across all suites (override:
    ``HOPPER_TMA_RECORDER``).  Existence is a capture-time check."""
    override = os.environ.get("HOPPER_TMA_RECORDER") or os.environ.get(
        "CLUSTER_SM_TMA_RECORDER"
    )
    if override:
        return Path(override).resolve()
    artifacts = Path(
        os.environ.get(
            "HOPPER_ARTIFACT_ROOT", "/path/to/hopper-prototype/artifacts"
        )
    )
    return artifacts / "cluster_sm_296_flow_20260921/common/libtma_tensor_map_recorder.so"


def tracer_root_path() -> Path:
    """NVBit tracer root directory (``HOPPER_TRACER_ROOT`` override; the
    invocation stores the root directory itself — what capture_binary's
    ``--tracer-root`` expects; the adapter validates the ``tracer_tool/``
    layout under it at capture time).  Existence is a capture-time check."""
    root = os.environ.get("HOPPER_TRACER_ROOT") or os.environ.get(
        "CLUSTER_SM_TRACER_ROOT"
    )
    root = Path(root).resolve() if root else (
        Path(os.environ.get(
            "HOPPER_ARTIFACT_ROOT", "/path/to/hopper-prototype/artifacts"
        )) / "cluster_sm_296_flow_20260921" / "common" / "tracer_root"
    )
    return root


def build_invocation(manifest: Path, case: dict) -> dict:
    case_id = str(case.get("case_id") or "")
    if not case_id:
        raise GenerationError("manifest case row without case_id")
    argv = [
        "python3",
        str(HARNESS),
        "--manifest",
        str(manifest),
        "--case-id",
        case_id,
        "--mode",
        "capture",
        "--build-root",
        str(pinned_build_root()),
    ]
    return {
        "argv": argv,
        "program_file": str(HARNESS),
        "cwd": str(SUITE_ROOT),
        "env": {},
        "kernel_range": "",
        "carveout": "auto",
        # The workload is a python harness: the declared program carries no
        # device code, so in-process lineinfo extraction records
        # unavailable/optional_diagnostic exactly like the deepgemm and
        # flashinfer harnesses; the conversion-side static control-word
        # source comes from the archived live dumped cubins (adapter
        # selected_runtime.sass), not from this program file.
        "allow_missing_lineinfo": True,
        "tma_recorder": str(tma_recorder_path()),
        "tracer_root": str(tracer_root_path()),
        "marker_contract": {"clock_marker_policy": CLOCK_MARKER_POLICY},
    }


def validate_invocation(invocation: dict, case_id: str) -> None:
    """Structural mirror of common/capture_case._validate_invocation.

    Artifact files (recorder/tracer/build root) are validated at capture
    time on the GPU host; here only repository-resident facts are checked so
    generation stays CPU-safe on hosts without the artifact tree.
    """
    argv = invocation.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise GenerationError(f"{case_id}: argv must be a non-empty string list")
    if not Path(invocation["program_file"]).is_file():
        raise GenerationError(f"{case_id}: program_file missing: {invocation['program_file']}")
    if not Path(invocation["cwd"]).is_dir():
        raise GenerationError(f"{case_id}: cwd is not a directory: {invocation['cwd']}")
    env = invocation.get("env") or {}
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise GenerationError(f"{case_id}: env must be str -> str")
    policy = invocation["marker_contract"]["clock_marker_policy"]
    if policy not in ("auto", "whole_kernel_pair", "outermost_pair",
                      "internal_measurement_reads"):
        raise GenerationError(f"{case_id}: invalid clock_marker_policy: {policy}")


def _manifest_serializer(text: str, payload: dict) -> bool:
    """Pick the serializer variant that reproduces the file byte-for-byte."""
    for ensure_ascii in (False, True):
        if json.dumps(payload, indent=2, ensure_ascii=ensure_ascii) + "\n" == text:
            return ensure_ascii
    raise GenerationError("manifest formatting is not json.dumps(indent=2); refusing to rewrite")


def merge_invocations(manifest: Path, invocations: dict[str, dict]) -> None:
    text = manifest.read_text(encoding="utf-8")
    payload = json.loads(text)
    ensure_ascii = _manifest_serializer(text, payload)
    for row in payload.get("cases") or []:
        case_id = str(row.get("case_id") or "")
        if case_id not in invocations:
            continue
        if "invocation" in row:
            raise GenerationError(
                f"{manifest.name}: case {case_id} already carries an invocation; "
                "--merge appends only"
            )
        row["invocation"] = invocations[case_id]
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )


def generate(
    manifest: Path, out_root: Path, case_filter: set[str] | None
) -> tuple[list[Path], dict[str, dict]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("cases") or []
    if not rows:
        raise GenerationError(f"{manifest}: no cases")
    written: list[Path] = []
    invocations: dict[str, dict] = {}
    out_dir = out_root / manifest.stem
    for case in rows:
        if not isinstance(case, dict):
            raise GenerationError(f"{manifest}: non-object case row")
        case_id = str(case.get("case_id") or "")
        if case_filter and case_id not in case_filter:
            continue
        invocation = build_invocation(manifest, case)
        validate_invocation(invocation, case_id)
        invocations[case_id] = invocation
        document = {
            "schema_version": INVOCATION_SCHEMA,
            "suite": "flashattention3",
            "manifest": str(manifest),
            "case_id": case_id,
            "generated_by": "suites/flashattention3/scripts/gen_invocations.py",
            "invocation": invocation,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{case_id}.json"
        target.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(target)
    return written, invocations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, action="append", default=None,
        help="manifest to generate for (repeatable; default: the cluster-sm100 "
             "manifest and the near holdout)",
    )
    parser.add_argument(
        "--out-root", type=Path, default=SUITE_ROOT / "scripts" / "invocations",
        help="output root; each manifest lands in <out-root>/<manifest_stem>/",
    )
    parser.add_argument(
        "--case-id", action="append", default=None,
        help="restrict generation to these case ids (repeatable)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="also append the invocation object to each manifest case row "
             "(new 'invocation' key only; existing keys are never touched)",
    )
    args = parser.parse_args(argv)

    manifests = [m.resolve() for m in (args.manifest or DEFAULT_MANIFESTS)]
    case_filter = set(args.case_id) if args.case_id else None
    total = 0
    for manifest in manifests:
        if not manifest.is_file():
            raise GenerationError(f"missing manifest: {manifest}")
        if args.merge and case_filter is not None:
            raise GenerationError("--merge always covers the whole manifest")
        written, invocations = generate(manifest, args.out_root.resolve(), case_filter)
        total += len(written)
        if args.merge:
            merge_invocations(manifest, invocations)
        print(json.dumps({
            "manifest": str(manifest),
            "invocations": len(written),
            "merged": bool(args.merge),
        }, sort_keys=True))
    print(json.dumps({"status": "ok", "total_invocations": total}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"gen_invocations: {error}", file=sys.stderr)
        raise SystemExit(2) from error
