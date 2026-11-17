#!/usr/bin/env python3
"""Deterministically generate capture invocations for the FlashInfer suite.

One ``hopper.capture-case-invocation.v1`` document per case row of the
cluster-SM manifests: ``holdout_far_dev.json``, ``holdout_far_sealed.json``
and ``flashinfer_cluster_sm100.json``.  A case id is resolved against these
manifests in exactly that order (the same three-choose-one the pre-stage-6
capture driver used), so far-holdout rows keep their manifest-local scaled
parameters — the argv names the manifest the scaled real call runs through.

The invocation describes the pure run-once workload
(``harness/run_case.py --capture-window``: build, warm, exactly one public
API call) for the stage-6 unified capture entry —
``suites/common/capture_case.py`` ->
``hoppersim capture via suites/common/capture_case.py``.
The 458-line in-suite capture chain and its NVBit window are gone; the
tracer/recorder plumbing belongs to the entry, never to the workload.

``invocation.env`` carries the pinned runtime the (deleted) drivers
exported: the 0.6.17 wheel site root + CuTe-DSL packages on PYTHONPATH, the
shared JIT workspace, the API-compat sitecustomize switch and the
JIT-parallelism caps.  A ``prep`` entry re-runs the same workload once
untraced before the capture so the op's JIT modules are materialized in the
shared workspace (the adapter records it verbatim; workload-only, never
traced).  Marker wheel patching, marker channels and the SM cap are
deliberately NOT part of the capture path.

Outputs: ``scripts/invocations/<manifest_stem>/<case_id>.json`` (default)
and, with ``--merge``, the same invocation object appended to each case row
under a new ``invocation`` key.  ``--merge`` never touches any pre-existing
field (values, formatting and key order are round-trip preserved; an
existing ``invocation`` key is a hard error).

CPU-only and side-effect free (except the output files): paths are resolved
deterministically; nothing imports torch/flashinfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
from hoppersim.kernel_symbols import kernel_symbol_contains  # noqa: E402

SUITE_ROOT = Path(__file__).resolve().parents[1]
HARNESS = SUITE_ROOT / "harness" / "run_case.py"
INVOCATION_SCHEMA = "hopper.capture-case-invocation.v1"
CLOCK_MARKER_POLICY = "outermost_pair"
FAR_MANIFESTS = (
    SUITE_ROOT / "manifests" / "holdout_far_dev.json",
    SUITE_ROOT / "manifests" / "holdout_far_sealed.json",
)
DEFAULT_MANIFESTS = FAR_MANIFESTS + (
    SUITE_ROOT / "manifests" / "flashinfer_cluster_sm100.json",
)
DEFAULT_BUILD_ROOT = Path(
    "/path/to/hopper-prototype/artifacts/cluster_sm_296_flow_20260921/flashinfer"
)
SITE_DIR_NAME = "build/site-packages-fi0617-cutlass470-tvmffi0113"


class GenerationError(RuntimeError):
    """Fail-closed manifest/contract violation."""


def _repo_artifact_root() -> Path:
    """Project root per scripts/paths.sh convention."""
    configured = os.environ.get("HOPPER_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    return PROJECT_ROOT


def pinned_build_root() -> Path:
    """Pinned 0.6.17 wheel build root (scripts/run_cluster_sm.sh contract)."""
    value = os.environ.get("FLASHINFER_CLUSTER_SM_BUILD_ROOT")
    return Path(value).resolve() if value else DEFAULT_BUILD_ROOT


def site_root() -> Path:
    return pinned_build_root() / SITE_DIR_NAME


def jit_workspace() -> Path:
    """Shared JIT workspace/cache (FLASHINFER_WORKSPACE_BASE in run_one)."""
    cluster_root = os.environ.get("FLASHINFER_CLUSTER_SM_ROOT")
    root = (
        Path(cluster_root).resolve()
        if cluster_root
        else Path("/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920/flashinfer")
    )
    return root / "jit-cache-shared"


def tma_recorder_path() -> Path:
    """TMA tensor-map recorder (override: ``HOPPER_TMA_RECORDER``).  The
    pre-stage-6 driver read ``CLUSTER_SM_TMA_RECORDER``; that campaign-local
    override is honored first.  Existence is a capture-time check."""
    for name in ("HOPPER_TMA_RECORDER", "CLUSTER_SM_TMA_RECORDER"):
        override = os.environ.get(name)
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


def workload_env() -> dict[str, str]:
    """Pinned runtime env for the run-once workload (run_cluster_sm.sh)."""
    site = site_root()
    return {
        "PYTHONPATH": ":".join([
            str(site / "nvidia_cutlass_dsl" / "dsl_packages"),
            str(site),
        ]),
        "FLASHINFER_DELIVERY_ARTIFACTS_ROOT": str(pinned_build_root()),
        "FLASHINFER_WORKSPACE_BASE": str(jit_workspace()),
        "FLASHINFER_CUBIN_DIR": str(pinned_build_root() / "build" / "cubins"),
        "FLASHINFER_CLUSTER_SM_MODE": "1",
        "HOPPER_FLASHINFER_API_COMPAT": "1",
        "MAX_JOBS": "2",
        "CMAKE_BUILD_PARALLEL_LEVEL": "2",
        "NINJAFLAGS": "-j2",
    }


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
        "--capture-window",
    ]
    return {
        "argv": argv,
        "program_file": str(HARNESS),
        "cwd": str(SUITE_ROOT),
        "env": workload_env(),
        # One untraced materialization run: guarantees the op's JIT modules
        # exist in the shared workspace before the traced child starts.
        "prep": [
            {
                "argv": list(argv),
                "cwd": str(SUITE_ROOT),
            }
        ],
        "kernel_range": "",
        "carveout": "auto",
        "allow_missing_lineinfo": True,
        "tma_recorder": str(tma_recorder_path()),
        "tracer_root": str(tracer_root_path()),
        "marker_contract": {"clock_marker_policy": CLOCK_MARKER_POLICY},
    }


MARKER_RUN_SCOPE = "outermost_in_kernel_clock_marker_interval"
DEFAULT_DISPOSITION = SUITE_ROOT / "far45" / "far45_disposition.json"


def disposition_fields(entry: dict) -> tuple[str | None, str | None]:
    """(override_field, sm_cap) of one far45 disposition entry.

    The original far45 suite recipe (run_flashinfer_suite.sh phase 3) is the
    only parameter source for the 45 dispositioned cases: gqa_probe wins over
    param_overrides as the --override-field; sm_cap becomes the
    FI_DELIVERY_SM_CAP env.  A case without an entry ran plain.
    """
    if not isinstance(entry, dict):
        return None, None
    field = "gqa_probe" if entry.get("gqa_probe") else (
        "param_overrides" if entry.get("param_overrides") else None)
    sm_cap = entry.get("sm_cap")
    return field, (str(sm_cap) if sm_cap is not None else None)


def marker_kernel_range_from_bare_result(path: Path, case_id: str) -> str:
    """The traced kernel-range fact from a bare ``--marker-timing`` run.

    The bare measurement run of the SAME instrumented binary is both the
    denominator producer and the only source of the resolved main-kernel
    name; the capture program traces exactly that kernel.  A bare result
    without a matched in-kernel marker scope is refused, never guessed.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(
            f"{case_id}: cannot read --from-bare-result {path}: {exc}"
        ) from exc
    timing = payload.get("timing") if isinstance(payload, dict) else None
    main_kernel = timing.get("main_kernel") if isinstance(timing, dict) else None
    if not isinstance(main_kernel, dict):
        raise GenerationError(
            f"{case_id}: {path} records no timing.main_kernel; run the bare "
            "--marker-timing measurement of the same instrumented binary "
            "first — it is the denominator producer and the kernel-name fact"
        )
    if str(main_kernel.get("scope") or "") != MARKER_RUN_SCOPE:
        raise GenerationError(
            f"{case_id}: {path} timing.main_kernel.scope is "
            f"{main_kernel.get('scope')!r}, not {MARKER_RUN_SCOPE!r}; the "
            "capture may only trace a marker producer run"
        )
    name = str(main_kernel.get("kernel_name_observed") or "")
    if not name:
        raise GenerationError(
            f"{case_id}: {path} records no kernel_name_observed"
        )
    return name


def build_marker_invocation(
    manifest: Path, case: dict, kernel_range: str,
    disposition: Path | None = None, disposition_entry: dict | None = None,
) -> dict:
    """The marker-timing program as the traced workload (same real binary).

    ``--marker-timing`` is the suite's own denominator producer run through
    the capture entry: the instrumented JIT executes real CS2R marker writes
    inside the traced process, and the formal 27->9 interval reduction stays
    with the bare measurement run.  No ``prep``: the bare measurement run of
    the same binary must have run first (``--from-bare-result`` carries its
    kernel fact), so the traced child's freshness audit finds instrumented
    modules already warm.

    A far45-dispositioned case MUST run its disposition entry (the original
    recipe's only parameter source): the override sidecar on the argv and
    FI_DELIVERY_SM_CAP in the env — the plain manifest shape is not the
    workload the far measurement contract describes.
    """
    case_id = str(case.get("case_id") or "")
    if not case_id:
        raise GenerationError("manifest case row without case_id")
    if case.get("marker_feasible") is not True:
        raise GenerationError(
            f"{case_id}: manifest row records marker_feasible="
            f"{case.get('marker_feasible')!r}; the marker-timing program is "
            "not a recorded fact for this case"
        )
    override_field, sm_cap = disposition_fields(disposition_entry or {})
    if (override_field or sm_cap is not None) and disposition is None:
        raise GenerationError(
            f"{case_id}: far45 disposition entry requires --disposition "
            "(the sidecar path is part of the original recipe)"
        )
    argv = [
        "python3",
        str(HARNESS),
        "--manifest",
        str(manifest),
        "--case-id",
        case_id,
        "--marker-timing",
        # capture-entry producer run: the marker pass only -- the whole-op
        # globaltimer protocol's five-second stream watch is calibrated for
        # untraced launches; the untraced bare measurement owns the timing.
        "--marker-pass-only",
    ]
    if override_field:
        argv += ["--case-overrides", str(disposition),
                 "--override-field", override_field]
    env = workload_env()
    # The producer brackets one real, marker-enabled operation; setup and
    # the untraced measurement protocol are outside the NVBit capture.
    env["ACTIVE_FROM_START"] = "0"
    if sm_cap is not None:
        env["FI_DELIVERY_SM_CAP"] = sm_cap
    return {
        "argv": argv,
        "program_file": str(HARNESS),
        "cwd": str(SUITE_ROOT),
        "env": env,
        "prep": [],
        "kernel_range": kernel_range,
        "carveout": "auto",
        "allow_missing_lineinfo": True,
        "tma_recorder": str(tma_recorder_path()),
        "tracer_root": str(tracer_root_path()),
        "marker_contract": {"clock_marker_policy": CLOCK_MARKER_POLICY},
    }


def marker_unit_from_capture(
    capture_dir: Path, case_id: str, selected_symbol: str | None
) -> dict:
    """Strict same-unit facts from the case's OWN capture and its recorded
    unit selection.

    The user scope is a unit INSIDE the original launch -- one complete CTA,
    a complete real hardware cluster of at most two members, or (type=launch)
    the whole 1/2-CTA launch -- never a whole-grid prediction.  The binding
    facts:

    - the producer kernel is UNIQUELY selected by the untraced
      measurement's resolved symbol (--from-bare-result); the sample
      protocol is checked on the bare result, while the capture may carry
      one trial launch, each physical launch in its own trace file;
    - ``ctas`` are that kernel's captured CTA coordinates, filtered to the
      recorded ``unit_selection`` (type=cta: exactly one complete CTA;
      type=cluster: exactly the selected real cluster members, at most two;
      type=launch: the whole launch, whose grid product must be <= 2);
    - a capture with no recorded unit selection only derives the
      complete-launch unit, and only within the bounded engine scope --
      missing identity is refused, never silently widened;
    - the original launch grid/block ride along as metadata; the bare
      measurement reduces ONLY the unit's own slots;
    - block shapes the device row format cannot map to actual warps (bx not
      a multiple of 32 with y/z > 1) and slot indexes beyond the device
      buffer (kMaxRows) refuse fail-closed.
    """
    capture_manifest = json.loads(
        (capture_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    launches = capture_manifest.get("launches") or []
    # The tracer writes EACH traced launch its own trace file
    # (kernel-<ctx_kernelid>-ctx_*.trace), so the producer protocol is
    # identified by KERNEL IDENTITY -- (mangled name, grid, block) -- with
    # its launches spread across many trace files, never by one file.
    per_kernel: dict[tuple, dict] = {}
    for launch in launches:
        kernel_file = str(launch.get("kernel_file") or "")
        name = str(launch.get("kernel_name_mangled") or "")
        grid_t = tuple(int(v) for v in launch.get("grid") or [])
        block_t = tuple(int(v) for v in launch.get("block") or [])
        if not kernel_file or not name:
            continue
        entry = per_kernel.setdefault(
            (name, grid_t, block_t),
            {"count": 0, "files": []})
        entry["count"] += 1
        entry["files"].append(kernel_file)
    if not per_kernel:
        raise GenerationError(f"{case_id}: capture records no launches")
    if not selected_symbol or not selected_symbol.strip():
        raise GenerationError(
            f"{case_id}: the producer kernel comes from the EXPLICIT "
            "resolved symbol of the untraced measurement "
            "(--from-bare-result), never from launch counts"
        )
    # The producer kernel is the resolved symbol's UNIQUE kernel identity
    # in the capture; several distinct identities matching one symbol stem
    # is an ambiguity that refuses (never "first match wins").
    selected_keys = [
        key for key in per_kernel
        if kernel_symbol_contains(key[0], selected_symbol.strip())
    ]
    if len(selected_keys) != 1:
        raise GenerationError(
            f"{case_id}: {len(selected_keys)} distinct kernel identities "
            f"match the resolved symbol {selected_symbol!r} "
            f"({[k[0][:60] for k in sorted(selected_keys)]}); the producer "
            "kernel is ambiguous and the unit refuses"
        )
    selected_key = selected_keys[0]
    dominant = per_kernel[selected_key]
    # The bare 27->9 sample protocol is verified on the bare result (the
    # pilot/consumer assert sample_count/status there).  The capture may
    # carry a single trial launch -- the capture file count is never the
    # sample protocol.
    grid = [int(v) for v in selected_key[1]]
    block = [int(v) for v in selected_key[2]]
    if len(grid) != 3 or len(block) != 3:
        raise GenerationError(f"{case_id}: launch grid/block facts incomplete")
    bx, by, bz = block
    if not (bx % 32 == 0 or (by <= 1 and bz <= 1)):
        raise GenerationError(
            f"{case_id}: block {block} is not supported by the device marker "
            "row format (CUDA warps follow the linear thread id; with bx not "
            "a multiple of 32 and y/z > 1 the recorded rows cannot be mapped "
            "to actual warps); no same-unit marker denominator"
        )
    # The complete unit coverage comes from the SELECTED physical launch's
    # own trace file (selected_launch: the first launch of the selected
    # kernel in recorded order) -- never a union across the 27 launches,
    # which could mask a per-launch coverage difference.
    selected_file = dominant["files"][0]
    logical_files = {
        name.removesuffix(".gz").removesuffix(".xz")
        for name in dominant["files"]
    }
    per_file_ctas: dict[str, set[tuple[int, int, int]]] = {}
    with (capture_dir / "cta_sm_map.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kernel_file = str(row.get("kernel_file") or "").removesuffix(".gz").removesuffix(".xz")
            cta = (int(row["cta_x"]), int(row["cta_y"]), int(row["cta_z"]))
            per_file_ctas.setdefault(kernel_file, set()).add(cta)
    logical_selected_file = selected_file.removesuffix(".gz").removesuffix(".xz")
    producer_ctas = per_file_ctas.get(logical_selected_file, set())
    if not producer_ctas:
        raise GenerationError(
            f"{case_id}: cta_sm_map.csv has no rows for the selected physical "
            f"launch file {selected_file!r}"
        )
    inconsistent = [
        f for f, ctas in per_file_ctas.items()
        if f in logical_files and ctas != producer_ctas
    ]
    if inconsistent:
        raise GenerationError(
            f"{case_id}: producer-kernel launches disagree on captured CTA "
            f"coverage: {sorted(inconsistent)}"
        )
    gx, gy, gz = grid

    meta = json.loads(
        (capture_dir / "capture_metadata.json").read_text(encoding="utf-8"))
    unit_selection = (meta.get("manifest_case") or {}).get("unit_selection")
    if isinstance(unit_selection, dict) and unit_selection.get("type"):
        utype = str(unit_selection.get("type"))
        selected = [
            (int(c[0]), int(c[1]), int(c[2]))
            for c in (unit_selection.get("ctas") or [])
        ]
        if not selected:
            raise GenerationError(
                f"{case_id}: recorded unit_selection.type={utype!r} lists no ctas"
            )
        missing = [c for c in selected if c not in producer_ctas]
        if missing:
            raise GenerationError(
                f"{case_id}: recorded unit selection names CTAs not in the "
                f"captured producer kernel: {missing}"
            )
        if utype == "cta" and len(selected) != 1:
            raise GenerationError(
                f"{case_id}: unit type cta must be exactly one complete CTA, "
                f"got {selected}"
            )
        if utype == "cluster" and len(selected) > 2:
            raise GenerationError(
                f"{case_id}: unit type cluster must be the complete real "
                f"cluster of at most two members, got {len(selected)}"
            )
        # Cluster COMPLETENESS (the selected set is the full real member set
        # with rank/peer facts) is enforced downstream by the converter's
        # cluster_launch_info gates; this side only binds membership and the
        # <=2 member bound.
        if utype == "launch":
            product = gx * gy * gz
            if len(selected) != product or product > 2:
                raise GenerationError(
                    f"{case_id}: unit type launch must enumerate the whole "
                    f"1/2-CTA launch (grid {grid}, selected {selected})"
                )
        unit = {"type": utype, "ctas": sorted(selected)}
    else:
        product = gx * gy * gz
        legal = {(x, y, z) for x in range(gx) for y in range(gy) for z in range(gz)}
        if product <= 2 and producer_ctas == legal:
            unit = {"type": "launch", "ctas": sorted(producer_ctas)}
        else:
            raise GenerationError(
                f"{case_id}: the capture records no unit selection and the "
                f"launch (grid {grid}, captured {len(producer_ctas)} CTAs) "
                "proves no bounded complete-launch unit; a partial unit needs "
                "an explicit recorded selection -- missing identity is never "
                "widened"
            )

    # Device buffer boundary: the slot index is the row position in a
    # kMaxRows-per-region table (fi_delivery_marker.h kMaxRows = 1 << 18).
    warp_x = (bx + 31) // 32
    warps_per_cta = warp_x * (by if by > 1 else 1) * (bz if bz > 1 else 1)
    max_cta_linear = max(
        cx + gx * (cy + gy * cz) for cx, cy, cz in unit["ctas"])
    max_slot = max_cta_linear * warps_per_cta + warps_per_cta - 1
    if max_slot >= (1 << 18):
        raise GenerationError(
            f"{case_id}: unit slot index {max_slot} exceeds the device "
            "marker buffer (kMaxRows = 262144)"
        )
    return {
        "type": unit["type"],
        "kernel_name_mangled": selected_key[0],
        "selected_launch_file": selected_file,
        "trace_kernel_files": sorted(dominant["files"]),
        "launch_count": dominant["count"],
        "ctas": unit["ctas"],
        "grid": grid,
        "block": block,
        "warps_per_cta": warps_per_cta,
        "unit_selection": unit_selection if isinstance(unit_selection, dict) else None,
    }


def validate_invocation(invocation: dict, case_id: str) -> None:
    """Structural mirror of common/capture_case._validate_invocation.

    Artifact files (recorder/tracer/pinned wheel) are validated at capture
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
    for index, item in enumerate(invocation.get("prep") or []):
        if not isinstance(item, dict) or not isinstance(item.get("argv"), list):
            raise GenerationError(f"{case_id}: prep[{index}] needs an argv list")
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
            "suite": "flashinfer",
            "manifest": str(manifest),
            "case_id": case_id,
            "generated_by": "suites/flashinfer/scripts/gen_invocations.py",
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


def _unit_companion(out: Path) -> Path:
    """Keep the entire output basename so distinct suffixes cannot alias.

    Regeneration without unit facts retires only this output's companion.
    """
    return out.with_name(out.name + ".unit.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, action="append", default=None,
        help="manifest to generate for (repeatable; default: holdout_far_dev, "
             "holdout_far_sealed, flashinfer_cluster_sm100)",
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
    parser.add_argument(
        "--marker-variant", action="store_true",
        help="emit the marker-timing program as the traced workload for ONE "
             "case: the suite's own denominator producer run through the "
             "capture entry, so the capture carries the real executed CS2R "
             "marker instructions of the same instrumented binary the bare "
             "measurement runs",
    )
    parser.add_argument(
        "--from-bare-result", type=Path, default=None,
        help="marker-variant only: bare --marker-timing result JSON; its "
             "timing.main_kernel.kernel_name_observed becomes kernel_range",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="marker-variant only: exact output document path",
    )
    parser.add_argument(
        "--disposition", default=DEFAULT_DISPOSITION,
        help="marker-variant only: far45 disposition sidecar (the original "
             "recipe's only parameter source for the 45 dispositioned cases); "
             "an empty --disposition '' asserts the case runs plain (kept as "
             "the raw string so '' stays empty instead of Path('.'))",
    )
    parser.add_argument(
        "--marker-unit-from-capture", type=Path, default=None,
        help="marker-variant only: the case's OWN capture directory; emits "
             "the out's own companion <out>.unit.json -- the flat {ctas, grid, block} "
             "unit facts (captured CTA set of the producer kernel + its "
             "recorded launch grid/block) that run_case --marker-unit-json "
             "reduces for the strict same-unit marker denominator; refuses "
             "when the capture proves no bounded unit",
    )
    args = parser.parse_args(argv)

    if args.marker_variant:
        if not args.manifest or len(args.manifest) != 1:
            raise GenerationError("--marker-variant needs exactly one --manifest")
        if not args.case_id or len(args.case_id) != 1:
            raise GenerationError("--marker-variant needs exactly one --case-id")
        if args.out is None:
            raise GenerationError("--marker-variant needs --out")
        if args.merge:
            raise GenerationError("--marker-variant never merges")
        manifest = args.manifest[0].resolve()
        if not manifest.is_file():
            raise GenerationError(f"missing manifest: {manifest}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        matches = [
            row for row in (payload.get("cases") or [])
            if isinstance(row, dict) and str(row.get("case_id") or "") == args.case_id[0]
        ]
        if len(matches) != 1:
            raise GenerationError(
                f"{args.case_id[0]!r} matches {len(matches)} rows in {manifest}"
            )
        disposition_raw = str(args.disposition or "").strip()
        disposition_path: Path | None = (
            Path(disposition_raw).resolve() if disposition_raw else None
        )
        disposition_entry: dict | None = None
        if disposition_path is not None:
            if not disposition_path.is_file():
                raise GenerationError(
                    f"missing disposition sidecar: {disposition_path}"
                )
            disposition_entry = json.loads(
                disposition_path.read_text(encoding="utf-8")
            ).get(args.case_id[0])
        marker_unit = None
        selected_symbol = None
        if args.from_bare_result is not None:
            bare_payload = json.loads(
                args.from_bare_result.resolve().read_text(encoding="utf-8"))
            selected_symbol = (
                (bare_payload.get("timing") or {}).get("main_kernel") or {}
            ).get("kernel_name_observed")
        if args.marker_unit_from_capture is not None:
            if selected_symbol is None:
                raise GenerationError(
                    "--marker-unit-from-capture needs --from-bare-result: "
                    "the producer kernel is the measurement's resolved "
                    "symbol, never a launch-count guess"
                )
            capture_dir = args.marker_unit_from_capture.resolve()
            if not capture_dir.is_dir():
                raise GenerationError(
                    f"missing capture directory: {capture_dir}")
            marker_unit = marker_unit_from_capture(
                capture_dir, args.case_id[0], selected_symbol
            )
        if args.from_bare_result is not None:
            # Legal DYNAMIC_KERNEL_RANGE grammar (tracer_tool.cu parse):
            # "@<regex>" = every kernel id, std::regex_match FULL match on
            # the demangled kernel name.  The observed stem is a C++
            # identifier (space/comma free), so the anchored pattern is
            # tokenizer-safe; a raw symbol string would crash the tracer's
            # std::stoull id parse.
            observed_stem = marker_kernel_range_from_bare_result(
                args.from_bare_result.resolve(), args.case_id[0]
            )
            kernel_range = "@.*" + re.escape(observed_stem) + ".*"
        else:
            # No bare fact yet: record every launch (the repeated-launch
            # identity mapping covers the capture side; the unit-scoped bare
            # measurement that follows supplies the denominator).  Heavy:
            # every launch's full per-warp CSVs land in the capture.
            kernel_range = ""
        invocation = build_marker_invocation(
            manifest, matches[0], kernel_range, disposition_path, disposition_entry
        )
        if args.from_bare_result is not None:
            bare_payload = json.loads(
                args.from_bare_result.resolve().read_text(encoding="utf-8"))
            observed = ((bare_payload.get("timing") or {}).get("main_kernel") or {}).get(
                "kernel_name_observed")
            if observed:
                invocation["marker_selected_kernel"] = observed
        # The research unit rides on the manifest's own target declaration:
        # a target_ctas == 1 non-cluster case is one complete CTA of the
        # original launch (CTA 0 -- deterministic in every launch).  It is
        # recorded ONLY when the bare kernel fact is in hand
        # (--from-bare-result): the adapter binds a capture-time unit's
        # kernel_file through marker_selected_kernel's selected launch, and
        # without that fact no selector may be guessed from file counts.
        # The capture-first flow records every launch (kernel_range="") and
        # the unit binds later from that capture (--marker-unit-from-capture).
        cluster_sm = matches[0].get("cluster_sm") or {}
        if (args.from_bare_result is not None
                and cluster_sm.get("target_ctas") == 1
                and not cluster_sm.get("uses_cluster")):
            invocation["unit_selection"] = {
                "type": "cta",
                "ctas": [[0, 0, 0]],
            }
        validate_invocation(invocation, args.case_id[0])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        unit_out = _unit_companion(args.out)
        if marker_unit is None:
            # This document derives no unit facts: its OWN companion (from a
            # previous unit-bearing generation of the same --out) is stale
            # for this lineage and is cleared.  Other outs' companions are
            # never touched.
            if unit_out.exists():
                unit_out.unlink()
        else:
            unit_out.write_text(
                json.dumps(marker_unit, indent=2, ensure_ascii=False,
                           sort_keys=True) + "\n",
                encoding="utf-8",
            )
        args.out.write_text(
            json.dumps(
                {
                    "schema_version": INVOCATION_SCHEMA,
                    "suite": "flashinfer",
                    "manifest": str(manifest),
                    "case_id": args.case_id[0],
                    "variant": "marker_timing_run",
                    "bare_result": (
                        str(args.from_bare_result.resolve())
                        if args.from_bare_result is not None else None),
                    # The unit-facts sidecar owned by THIS document (its own
                    # .unit.json companion); None = this invocation derives
                    # no unit facts, and consumers must refuse to guess a
                    # path.
                    "marker_unit_file": (
                        str(unit_out) if marker_unit is not None else None),
                    "generated_by":
                        "suites/flashinfer/scripts/gen_invocations.py",
                    "invocation": invocation,
                },
                indent=2, ensure_ascii=False, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": "ok",
            "variant": "marker_timing_run",
            "case_id": args.case_id[0],
            "kernel_range": kernel_range,
            "out": str(args.out),
        }, sort_keys=True))
        return 0

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
