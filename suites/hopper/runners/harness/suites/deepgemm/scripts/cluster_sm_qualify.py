#!/usr/bin/env python3
"""Compose cluster_sm qualification.json from per-case summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# User-authorized 2026-09-21: DeepGEMM may keep real short cases below 100k.
# Upper bound 500k unchanged. Other suites stay [100000, 500000].
# Existing qualified DG cases in [100k, 500k] are not retuned down.
MIN_CYCLES_EXCLUSIVE = 0.0
MAX_CYCLES = 500_000


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_only(correctness: dict[str, Any] | None) -> bool:
    if not correctness:
        return True
    if correctness.get("status") != "pass":
        return True
    if "symmetric_difference" in correctness or "max_abs_error" in correctness:
        return False
    for trial in correctness.get("trials") or []:
        if not isinstance(trial, dict):
            continue
        if "symmetric_difference" in trial or "max_abs_error" in trial:
            return False
    return True


def _grid_product(grid: Any) -> int | None:
    if isinstance(grid, dict):
        xs = [int(grid[k]) for k in ("x", "y", "z") if grid.get(k) is not None]
        if not xs:
            return None
        product = 1
        for value in xs:
            product *= value
        return product
    if isinstance(grid, (list, tuple)) and grid:
        product = 1
        for value in grid:
            product *= int(value)
        return product
    return None



def _xyz_list(value: Any) -> Any:
    """Emit [x, y, z] so scrapers that only product lists still see launched grid.

    Does not invent SM ids. Marker smids stay in formal/result.json.
    """
    if isinstance(value, dict):
        ordered: list[Any] = []
        for key in ("x", "y", "z"):
            if key in value:
                ordered.append(value[key])
        if len(ordered) != 3:
            return value
        try:
            return [int(item) for item in ordered]
        except (TypeError, ValueError):
            return value
    if isinstance(value, (list, tuple)) and value:
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            return list(value)
    return value


def _distinct_smids(sm_ids: Any) -> list[int]:
    if not isinstance(sm_ids, list):
        return []
    values: set[int] = set()
    for item in sm_ids:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            values.add(number)
    return sorted(values)


def _choose_observed_sms(*, target: int, sm_ids: Any, grid: Any, uses_cluster: bool = False) -> tuple[Any, str | None]:
    """Cluster 2CTA: two distinct marker %%smid values.
    Ordinary 2CTA: 1 or 2 marker smids (measured SM <= 2; same SM allowed).
    1CTA: one smid or grid product 1.
    Never copy observed_sm_count / requested_num_sms into observed_sms.
    """
    distinct = _distinct_smids(sm_ids)
    if target == 2:
        if uses_cluster:
            if len(distinct) == 2:
                return distinct, "marker_smid"
            return None, "missing_two_distinct_marker_smids"
        if 1 <= len(distinct) <= 2:
            return distinct, "marker_smid"
        return None, "missing_marker_smid_for_ordinary_2cta"
    if target == 1:
        if len(distinct) == 1:
            return distinct, "marker_smid"
        if _grid_product(grid) == 1:
            return 1, "inferred_from_grid_product_1"
        return None, "missing_marker_smid_or_grid_product_1"
    return None, None


def _payload_from_result(path: Any) -> dict[str, Any] | None:
    if not path:
        return None
    file = Path(str(path))
    if not file.is_file():
        return None
    try:
        payload = _load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _recover_payload(summary: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Pick the measured payload behind a summary row.

    The formal attempt is preferred, then any 9-sample pass attempt. A summary
    whose formal row is `topology_failed` because the old runner gate demanded a
    cluster from a cluster_dim=1 launch still carries the same-launch payload,
    so recover it from the attempt rows as well. Never re-run for this.
    """
    formal = summary.get("formal") or {}
    payload = formal.get("payload") if isinstance(formal.get("payload"), dict) else {}
    artifact = formal.get("result")
    candidates: list[tuple[dict[str, Any], Any, int]] = []
    if payload:
        samples = len(((payload.get("timing") or {}).get("samples_cycles") or []))
        candidates.append((payload, artifact, samples))
    for row in reversed(list(summary.get("attempts") or [])):
        if not row.get("result") or row.get("status") not in ("pass", "topology_failed"):
            continue
        recovered = _payload_from_result(row.get("result"))
        if not recovered:
            continue
        samples = len(((recovered.get("timing") or {}).get("samples_cycles") or []))
        candidates.append((recovered, str(row.get("result")), samples))
    if not candidates:
        return payload or {}, artifact
    with_nine = [row for row in candidates if row[2] >= 9]
    chosen = (with_nine or candidates)[0]
    return chosen[0], chosen[1]


def _kernel_name_from_cubin(cubin: str) -> str:
    parent = Path(cubin).parent.name
    parts = parent.split(".")
    if len(parts) >= 2 and parts[0] == "kernel":
        return parts[1]
    return ""


def _kernels(payload: dict[str, Any]) -> list[str]:
    topo = payload.get("topology") or {}
    timing = payload.get("timing") or {}
    values = [str(item) for item in (timing.get("kernels") or topo.get("kernels") or [])]
    kernel = payload.get("kernel") or {}
    for entry in kernel.get("entries") or []:
        cubin = str(entry.get("cubin") or "")
        name = str(entry.get("module") or entry.get("name") or "") or _kernel_name_from_cubin(cubin)
        for item in (name, cubin):
            if item and item not in values:
                values.append(item)
    return values


def _self_check() -> None:
    if _grid_product({"x": 1, "y": 1, "z": 1}) != 1 or _grid_product([1, 1, 1]) != 1:
        raise SystemExit("qualify self-check: 1CTA grid product must accept dict and list")
    if _xyz_list({"x": 2, "y": 1, "z": 1}) != [2, 1, 1] or _xyz_list([1, 1, 1]) != [1, 1, 1]:
        raise SystemExit("qualify self-check: launched grid/cluster_dim must emit [x,y,z] lists")
    sms, source = _choose_observed_sms(target=2, sm_ids=[], grid={"x": 2, "y": 1, "z": 1}, uses_cluster=True)
    if sms is not None or source != "missing_two_distinct_marker_smids":
        raise SystemExit("qualify self-check: cluster 2CTA must not infer observed_sms from count/grid")
    sms, source = _choose_observed_sms(target=2, sm_ids=[], grid={"x": 2, "y": 1, "z": 1}, uses_cluster=False)
    if sms is not None or source != "missing_marker_smid_for_ordinary_2cta":
        raise SystemExit("qualify self-check: ordinary 2CTA must not infer observed_sms from grid")
    sms, source = _choose_observed_sms(target=1, sm_ids=[], grid=[1, 1, 1])
    if sms != 1 or source != "inferred_from_grid_product_1":
        raise SystemExit("qualify self-check: 1CTA list grid product 1 must infer SM=1")
    sms, source = _choose_observed_sms(target=2, sm_ids=[124, 125], grid=None, uses_cluster=True)
    if sms != [124, 125] or source != "marker_smid":
        raise SystemExit("qualify self-check: cluster 2CTA observed_sms must be two marker smids")
    sms, source = _choose_observed_sms(target=2, sm_ids=[124], grid=None, uses_cluster=False)
    if sms != [124] or source != "marker_smid":
        raise SystemExit("qualify self-check: ordinary 2CTA same-SM marker is legal")
    sms, source = _choose_observed_sms(target=2, sm_ids=[124], grid=None, uses_cluster=True)
    if sms is not None:
        raise SystemExit("qualify self-check: cluster 2CTA still needs two distinct SMs")


def _flat_counts(counts: Any) -> list[int]:
    """Flatten cluster_cta_counts (list of ints or list of lists) to ints."""
    values: list[int] = []
    if not isinstance(counts, (list, tuple)):
        return values
    for row in counts:
        if isinstance(row, (list, tuple)):
            values.extend(int(v) for v in row)
        else:
            values.append(int(row))
    return values


def _dim_product(value: Any) -> int | None:
    if isinstance(value, dict):
        xs = [int(value[k]) for k in ("x", "y", "z") if value.get(k) is not None]
        if not xs:
            return None
        product = 1
        for item in xs:
            product *= item
        return product
    if isinstance(value, (list, tuple)) and value:
        product = 1
        for item in value:
            product *= int(item)
        return product
    return None


def _reconcile_cluster(topology: dict[str, Any]) -> tuple[Any, Any]:
    """Return (cluster_dim, multicast) consistent with the measured marker.

    cluster_cta_counts is the in-kernel cluster marker: the launch observation.
    multicast/cluster_dim may have been copied from a JIT cubin source parse,
    which a warm cache can leave describing a cubin this run never launched.
    The marker wins; the diagnostic is only used when the marker is absent.
    """
    counts = _flat_counts(topology.get("cluster_cta_counts"))
    multicast = topology.get("multicast")
    cluster_dim = topology.get("cluster_dim")
    if not counts:
        return cluster_dim, multicast
    measured = max(counts)
    recorded = multicast if isinstance(multicast, int) else None
    if recorded is None or measured < recorded:
        multicast = measured
        if measured == 1:
            cluster_dim = {"x": 1, "y": 1, "z": 1}
        elif cluster_dim is None:
            cluster_dim = {"x": 1, "y": measured, "z": 1}
    return cluster_dim, multicast


def _main_kernel_row(
    payload: dict[str, Any],
    *,
    cluster_dim: Any,
    grid: Any,
    cta_count: Any,
    source: dict[str, Any],
    artifact: Any,
) -> dict[str, Any] | None:
    """Target-kernel identity and launch shape, taken from the measured run.

    kind/name come from the loaded cubin module name (kernel.<module>.<hash> in
    the cache path), never from a generalized family label. cta_count/grid/
    cluster_dim are the launch the reported timing covers.
    """
    entries = (payload.get("kernel") or {}).get("entries") or []
    modules: list[str] = []
    for entry in entries:
        cubin = str(entry.get("cubin") or "")
        name = ""
        if cubin:
            parent = Path(cubin).parent.name
            parts = parent.split(".")
            if len(parts) >= 2 and parts[0] == "kernel":
                name = parts[1]
        if not name:
            name = str(entry.get("module") or "")
        if name and name not in modules:
            modules.append(name)
    timing = payload.get("timing") or {}
    bank_kind = (timing.get("simulator_comparison") or {}).get("selected_operation_kernel")
    if bank_kind and bank_kind not in modules:
        modules.insert(0, bank_kind)
    if not modules:
        return None
    return {
        "kind": modules[0],
        "name": modules[0],
        "all_marked_kernels_in_op": modules,
        "cta_count": cta_count,
        "grid": _xyz_list(grid),
        "cluster_dim": _xyz_list(cluster_dim),
        # One public op launches the target kernel once per measured trial; the
        # raw span count in timing.samples_cycles_raw is the launch evidence.
        "same_name_launch_count": len(
            (timing.get("simulator_comparison") or {}).get("samples_cycles_raw")
            or timing.get("samples_cycles_raw")
            or []
        ),
        "same_name_ordinal": 0,
        "operation_kernel_ordinal": modules.index(modules[0]),
        "selection_rule": (
            "attention: bank_0 MQA logits marker rows only"
            if bank_kind
            else "single marked core kernel of the public op"
        ),
        "implementation_entries": list(source.get("implementation_entries") or []),
        "evidence": artifact,
    }


def _observed_cluster(topology: dict[str, Any]) -> int:
    counts = topology.get("cluster_cta_counts") or []
    values = _flat_counts(counts)
    multicast = topology.get("multicast")
    if isinstance(multicast, int):
        values.append(multicast)
    return max(values) if values else 0


def _recorded_concurrency(work: Path) -> dict[str, Any]:
    concurrency = {
        "outer_shards": 8,
        "inner_workers": 4,
        "inner_workers_allowed": [4, 16],
        "gpu_section": "flock_serial",
        "prepare_queue_workers": 4,
    }
    exits: dict[str, int] = {}
    logs = work / "logs"
    if logs.is_dir():
        for path in sorted(logs.glob("run.shard*.exit")):
            shard = path.name[len("run.shard"):-len(".exit")]
            try:
                exits[shard] = int(path.read_text(encoding="utf-8").strip() or "0")
            except ValueError:
                continue
    if exits:
        concurrency["shard_exits"] = exits
    return concurrency


def qualify_case(source: dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
    case_id = source["source_case_id"]
    base = {
        "source_case_id": case_id,
        "target_ctas": source.get("target_ctas"),
        "uses_cluster": source.get("uses_cluster"),
        "parameters": None,
        "mechanisms_preserved": False,
        "mechanism_evidence": "",
        "topology": {
            "grid": None,
            "cluster_dim": None,
            "cta_count": None,
            "observed_sms": None,
            "evidence": None,
        },
        "correctness": {"status": "missing", "method": None, "artifact": None},
        "timing": {
            "cycles_median": None,
            "samples_cycles": [],
            "scope": None,
            "method": None,
            "artifact": None,
        },
        "reason": None,
        "attempts": [],
    }
    if summary is None:
        if source.get("vendor_legal_num_sms") is False and not source.get("plain_2cta_fallback"):
            base["status"] = "unresolved"
            base["reason"] = (
                "GemmDesc even-SM 1CTA is illegal, but skip probe was not executed; "
                + str(source.get("skip_reason") or "num_sms % 2 == 0")
            )
            return base
        base["status"] = "unresolved"
        base["reason"] = "no cluster_sm summary; case was not executed to completion"
        return base
    payload_early = (summary.get("formal") or {}).get("payload") or {}
    topology_early = payload_early.get("topology") or {}
    observed_early = _observed_cluster(topology_early)
    has_cluster_probe = bool(
        topology_early.get("cluster_cta_counts")
        or topology_early.get("sm_ids")
        or topology_early.get("cta_count")
    )
    unclusterable = (
        source.get("uses_cluster")
        and source.get("heuristic_cluster_at_2") is False
        and has_cluster_probe
        and observed_early != 2
    )
    if (
        summary.get("status") == "skipped"
        or (
            source.get("vendor_legal_num_sms") is False
            and not source.get("plain_2cta_fallback")
        )
        or unclusterable
    ):
        base["status"] = "skipped"
        base["parameters"] = summary.get("final_parameters") or source.get("shape")
        base["attempts"] = list(summary.get("attempts") or [])
        base["mechanisms_preserved"] = False
        base["mechanism_evidence"] = summary.get("mechanism_evidence") or source.get("skip_reason")
        payload = (summary.get("formal") or {}).get("payload") or {}
        topology = payload.get("topology") or {}
        base["reason"] = summary.get("reason") or source.get("skip_reason")
        if topology:
            sm_ids = _distinct_smids(topology.get("sm_ids"))
            base["topology"] = {
                "grid": _xyz_list(topology.get("grid")),
                "cluster_dim": _xyz_list(topology.get("cluster_dim")),
                "cta_count": topology.get("cta_count"),
                "observed_sms": sm_ids or None,
                "observed_sms_source": "marker_smid" if sm_ids else None,
                "kernels": topology.get("kernels"),
                "cluster_cta_counts": topology.get("cluster_cta_counts"),
                "requested_num_sms": topology.get("requested_num_sms"),
                "evidence": (summary.get("formal") or {}).get("result"),
                "status": topology.get("status"),
                "error": topology.get("error"),
                "jit_cluster": topology.get("jit_cluster"),
            }
            base["reason"] = (
                str(base["reason"])
                + f"; observed_cluster_cta_counts={topology.get('cluster_cta_counts')}; "
                + f"cta_count={topology.get('cta_count')}; sm_ids={topology.get('sm_ids')}"
            )
        base["correctness"] = {
            "status": (payload.get("correctness") or {}).get("status") or "not_run_illegal_config",
            "method": None,
            "artifact": (summary.get("formal") or {}).get("result"),
        }
        return base
    attempts = list(summary.get("attempts") or [])
    base["attempts"] = [
        {
            "tag": row.get("tag"),
            "log": row.get("log"),
            "result": row.get("result"),
            "exit_code": row.get("exit_code"),
            "status": row.get("status"),
            "median_cycles": row.get("median_cycles"),
        }
        for row in attempts
    ]
    base["parameters"] = summary.get("final_parameters") or source.get("shape")
    formal = summary.get("formal") or {}
    payload, artifact = _recover_payload(summary)
    topology = payload.get("topology") or {}
    correctness = payload.get("correctness") or {}
    timing = payload.get("timing") or {}
    samples = [float(v) for v in (timing.get("samples_cycles") or [])]
    cycles = timing.get("median_cycles")
    kernels = _kernels(payload)
    grid = topology.get("grid")
    cta_count = topology.get("cta_count") or topology.get("observed_cta_count") or topology.get("cta_count_min")
    target = int(source.get("target_ctas") or 0)
    uses_cluster = bool(source.get("uses_cluster"))
    # cluster_cta_counts comes from the in-kernel marker: it is the launch
    # observation. multicast/cluster_dim may come from a JIT cubin source parse,
    # which a warm cache can leave describing a cubin this run never launched.
    # Reconcile against the marker so a stale diagnostic cannot fabricate a
    # cluster (or hide one) on an already-measured run.
    cluster_dim, multicast = _reconcile_cluster(topology)
    observed_sms, observed_sms_source = _choose_observed_sms(
        target=target,
        sm_ids=topology.get("sm_ids"),
        grid=grid,
        uses_cluster=uses_cluster,
    )
    base["topology"] = {
        "grid": _xyz_list(grid),
        "cluster_dim": _xyz_list(cluster_dim),
        "cta_count": cta_count,
        "observed_sms": observed_sms,
        "observed_sms_source": observed_sms_source,
        "kernels": kernels,
        "multicast": multicast,
        "cluster_cta_counts": topology.get("cluster_cta_counts"),
        "requested_num_sms": topology.get("requested_num_sms"),
        "evidence": artifact,
        "status": topology.get("status"),
        "error": topology.get("error"),
        "jit_cluster": topology.get("jit_cluster"),
        # Target (main) kernel identity + launch shape, from the same measured
        # run. cta_count/grid/cluster_dim are the launch the timing above covers.
        "main_kernel": _main_kernel_row(
            payload,
            cluster_dim=cluster_dim,
            grid=grid,
            cta_count=cta_count,
            source=source,
            artifact=artifact,
        ),
    }
    method = "symmetric_difference"
    trial_max = None
    trial_tol = None
    for trial in correctness.get("trials") or []:
        if isinstance(trial, dict) and "max_abs_error" in trial:
            value = trial.get("max_abs_error")
            if isinstance(value, (int, float)):
                trial_max = value if trial_max is None else max(trial_max, float(value))
            if trial.get("tolerance") is not None:
                trial_tol = trial.get("tolerance")
            method = "attention_max_abs_error_plus_invalid_region"
    if "max_abs_error" in correctness:
        method = "attention_max_abs_error_plus_invalid_region"
    base["correctness"] = {
        "status": correctness.get("status") or "missing",
        "method": method,
        "artifact": artifact,
        "symmetric_difference": correctness.get("symmetric_difference"),
        "max_abs_error": correctness.get("max_abs_error") if "max_abs_error" in correctness else trial_max,
        "maximum_symmetric_difference": correctness.get("maximum_symmetric_difference"),
        "tolerance": correctness.get("tolerance") if "tolerance" in correctness else trial_tol,
        "invalid_region_negative_infinity": any(
            isinstance(trial, dict) and trial.get("invalid_region_negative_infinity")
            for trial in (correctness.get("trials") or [])
        ) or None,
    }
    # Timing source: the target (main) kernel, not the whole operation.
    # Attention ops run two marked kernels in one public op (bank_0 MQA logits
    # + bank_1 clean_logits); the runner already splits bank_0 into
    # timing.simulator_comparison. Prefer that when present so the reported
    # cycles are the main kernel's. Whole-operation cycles are kept as a
    # diagnostic. Core ops launch a single core kernel, so the two agree.
    main_timing = timing.get("simulator_comparison") or {}
    main_samples = [float(v) for v in (main_timing.get("samples_cycles") or [])]
    main_cycles = main_timing.get("median_cycles")
    use_main = (
        isinstance(main_cycles, (int, float))
        and len(main_samples) >= 9
        and main_timing.get("selected_operation_kernel")
    )
    base["timing"] = {
        "cycles_median": main_cycles if use_main else cycles,
        "samples_cycles": main_samples if use_main else samples,
        "scope": (
            main_timing.get("scope")
            or "outermost_in_kernel_clock_marker_interval"
            if use_main
            else (timing.get("scope") or "in_kernel_globaltimer_span_converted_by_clock64_ratio")
        ),
        "method": timing.get("method"),
        "artifact": artifact,
        "kernels": kernels,
        "cv_percent": main_timing.get("cv_percent") if use_main else timing.get("cv_percent"),
        "definition": timing.get("cycle_compatibility_definition") or timing.get("definition"),
        # Core ops launch exactly one marked kernel, so the outermost-marker
        # span already is the main kernel; record it explicitly instead of
        # relying on the reader to infer that. Attention ops point at the
        # runner's bank_0 split above.
        "main_kernel": (
            {
                "kind": ((_main_kernel_row(
                    payload,
                    cluster_dim=cluster_dim,
                    grid=grid,
                    cta_count=cta_count,
                    source=source,
                    artifact=artifact,
                ) or {}).get("kind")),
                "source": (
                    "timing.simulator_comparison"
                    if use_main
                    else "timing (single marked kernel of the public op)"
                ),
                "hardware_metric": main_timing.get("hardware_metric"),
                "hardware_value_field": main_timing.get("hardware_value_field"),
                "clock_marker_policy": main_timing.get("clock_marker_policy"),
                "marker_interval_definition": main_timing.get("marker_interval_definition"),
                "samples_cycles_raw": [float(v) for v in (main_timing.get("samples_cycles_raw") or [])]
                or [float(v) for v in (timing.get("samples_cycles_raw") or [])],
                "median_globaltimer": main_timing.get("median_globaltimer"),
            }
        ),
        "whole_operation_diagnostic": {
            "cycles_median": cycles,
            "samples_cycles": samples,
            "scope": timing.get("scope"),
            "note": "whole-operation span across all marked kernels; diagnostic only, not the main-kernel timing",
        },
    }
    modules = list(source.get("implementation_modules") or source.get("mechanisms") or [])
    counts = topology.get("cluster_cta_counts") or []
    flat = _flat_counts(counts)
    observed_cluster = max(flat) if flat else 0
    if isinstance(multicast, int):
        observed_cluster = max(observed_cluster, multicast)
    topo_measured_ordinary_2cta = (
        target == 2
        and not uses_cluster
        and observed_sms_source == "marker_smid"
        and isinstance(observed_sms, list)
        and 2 <= len(observed_sms) <= 2
        and _dim_product(base["topology"]["cluster_dim"]) in (None, 1)
        and cta_count == 2
        and (not counts or max(int(v) for v in _flat_counts(counts)) == 1)
    )
    if isinstance(multicast, int):
        observed_cluster = max(observed_cluster, multicast)
    expected_cluster = 2 if source.get("uses_cluster") else 1
    mechanisms_ok = (
        bool(modules)
        and observed_cluster == expected_cluster
        and (topology.get("status") == "pass" or topo_measured_ordinary_2cta)
    )
    base["mechanisms_preserved"] = mechanisms_ok
    base["mechanism_evidence"] = (
        f"source_modules={modules}; observed_cluster_cta_counts={topology.get('cluster_cta_counts')}; "
        f"multicast={multicast}; jit={topology.get('jit_cluster')}; "
        f"kernels={kernels}; result={artifact}"
    )
    smid_ok = (
        (
            target == 2
            and uses_cluster
            and observed_sms_source == "marker_smid"
            and isinstance(observed_sms, list)
            and len(observed_sms) == 2
        )
        or (
            target == 2
            and not uses_cluster
            and observed_sms_source == "marker_smid"
            and isinstance(observed_sms, list)
            and 1 <= len(observed_sms) <= 2
        )
        or (target == 1 and observed_sms_source in {"marker_smid", "inferred_from_grid_product_1"})
    )
    # Ordinary 2CTA measured content: cluster_dim product 1, cluster_cta_counts=[1],
    # cta_count 2, two distinct same-launch marker smids. The runner's
    # validate_cluster_sm_topology used to demand cluster_cta_count=[2] for any
    # 2CTA request, so an old run.log may carry status=fail with exactly this
    # measured content. Judge the payload, not the flag text (Main 2026-09-21).
    topo_ok = (
        (topology.get("status") == "pass" or topo_measured_ordinary_2cta)
        and base["topology"]["grid"] is not None
        and smid_ok
    )
    correct_ok = correctness.get("status") == "pass" and not _finite_only(correctness)
    timing_ok = (
        isinstance(base["timing"]["cycles_median"], (int, float))
        and MIN_CYCLES_EXCLUSIVE < float(base["timing"]["cycles_median"]) <= MAX_CYCLES
        and len(base["timing"]["samples_cycles"]) >= 9
    )
    if topo_ok and correct_ok and timing_ok and mechanisms_ok:
        base["status"] = "qualified"
        base["reason"] = None
        return base
    base["status"] = "unresolved"
    reasons = []
    if not topo_ok:
        reasons.append(
            f"topology={topology.get('status')} error={topology.get('error')} "
            f"smids={observed_sms} source={observed_sms_source}"
        )
    if not correct_ok:
        reasons.append(f"correctness={correctness.get('status')}")
    if not timing_ok:
        reasons.append(
            f"cycle window unmet at vendor-legal config; median={base['timing']['cycles_median']} "
            f"samples={len(base['timing']['samples_cycles'])} window=(0,{MAX_CYCLES}]; not padded"
        )
    if not mechanisms_ok:
        reasons.append(
            f"cluster mismatch expected={expected_cluster} observed={observed_cluster} modules={modules}"
        )
    if formal.get("exit_code"):
        reasons.append(f"exit_code={formal.get('exit_code')} log={formal.get('log')}")
    base["reason"] = "; ".join(reasons) or "formal run did not meet cluster_sm qualification"
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    _self_check()
    derived = _load(args.manifest)
    source_ids = [row["case_id"] for row in _load(args.source_manifest)["cases"]]
    if source_ids != [row["source_case_id"] for row in derived["cases"]]:
        raise SystemExit("derived source_case_id order does not match deepgemm_100.json")
    summaries = {}
    for path in sorted((args.work / "summaries").glob("*.json")):
        row = _load(path)
        summaries[row["source_case_id"]] = row
    # Retired summary dirs keep measured rows for cases whose pilot topology
    # failed before a summary was written under the ordinary-2CTA runner.
    # Read them as a fallback; never re-run for a missing summary file. Prefer
    # the candidate whose formal attempt actually produced a result artifact
    # (measured) over a retired skip probe.
    retired: dict[str, list[dict[str, Any]]] = {}
    for directory in sorted((args.work).glob("summaries.retired_*")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                row = _load(path)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(row, dict) and row.get("source_case_id"):
                retired.setdefault(row["source_case_id"], []).append(row)
    for case_id, candidates in retired.items():
        if case_id in summaries:
            continue
        measured = [row for row in candidates if (row.get("formal") or {}).get("result")]
        # A retired "skipped" row for a case the 2026-09-21 plain-2CTA
        # authorization covers is not a real skip: num_sms=1 is illegal for
        # GemmDesc, but legal num_sms=2 as two ordinary CTAs was not run yet.
        # Leave it without a summary so it lands as unresolved, not skipped.
        usable = [row for row in (measured or candidates) if row.get("status") != "skipped"]
        if usable:
            summaries[case_id] = (measured or usable)[0]
    cases = [qualify_case(row, summaries.get(row["source_case_id"])) for row in derived["cases"]]
    if args.require_all and sum(path.is_file() for path in (args.work / "summaries").glob("*.json")) != 100:
        print(json.dumps({
            "status": "partial",
            "summaries": len(summaries),
            "expected": 100,
        }, sort_keys=True))
        return 0
    marker = args.package / "deep_gemm/sm90_suite_marker.json"
    package_meta = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    if package_meta.get("cluster_sm_allow_num_sms_1") is True:
        raise SystemExit("qualification refuses illegally relaxed cluster_sm_allow_num_sms_1 package")
    concurrency = _recorded_concurrency(args.work)
    concurrency_path = args.work / "logs" / "concurrency.json"
    concurrency_path.parent.mkdir(parents=True, exist_ok=True)
    concurrency_path.write_text(json.dumps(concurrency, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = args.output or (args.work / "qualification.json")
    payload = {
        "suite": "deepgemm",
        "source_manifest": "manifests/deepgemm_100.json",
        "derived_manifest": "manifests/deepgemm_cluster_sm100.json",
        "runner": "scripts/run_cluster_sm.sh",
        "package": str(args.package),
        "package_meta": {
            "path": str(args.package),
            "cluster_sm_allow_num_sms_1": bool(package_meta.get("cluster_sm_allow_num_sms_1")),
            "kgroup_cluster_required": bool(package_meta.get("kgroup_cluster_required")),
        },
        "concurrency": concurrency,
        "work": str(args.work),
        "cycle_policy": {
            "user_authorized_lower_bound_exception": True,
            "authorization": "dg允许10万周期内",
            "when": "2026-09-21",
            "lower_bound": "positive GPU cycles; 100k floor waived for real short DG cases",
            "upper_bound_inclusive": MAX_CYCLES,
            "other_suites": [100_000, 500_000],
            "do_not_retune_existing_qualified_below_100k": True,
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in cases:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"output": str(output), "n": len(cases), "counts": counts, "concurrency": concurrency}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
