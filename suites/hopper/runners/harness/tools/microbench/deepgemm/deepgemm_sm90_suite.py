"""Manifest and timing helpers for the DeepGEMM full-GPU delivery suite."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3] / "suites" / "deepgemm"
DEFAULT_MANIFEST = ROOT / "manifests/deepgemm_core_runtime_manifest.json"
SCHEMA_VERSION = "deepgemm-sm90-suite-v1"
EXPECTED_REVISION = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
MIN_CYCLES = 100_000
MAX_CYCLES = 5_000_000
# cluster_sm: user-authorized 2026-09-21 DG short-case exception.
# Floor exclusive 0 (positive GPU cycles). Upper 500k unchanged.
CLUSTER_SM_MIN_CYCLES = 0
CLUSTER_SM_MAX_CYCLES = 500_000
CLUSTER_SM_MODE = "cluster_sm"
# Typical large-GEMM occupancy on H100; not a topology gate for legal small launches.
MIN_NORMAL_ACTIVE_SMS = 128
MODES = ("normal",)
ORIGINAL_CORE_CLASSES = (
    "bf16_dense_nt", "bf16_dense_nn", "bf16_dense_tn", "bf16_dense_tt",
    "bf16_dense_nt_acc_bf16", "bf16_dense_nt_acc_fp32",
    "bf16_dense_nn_acc_bf16", "bf16_dense_tn_acc_fp32",
    "bf16_dense_tt_acc_fp32", "fp8_dense_nt", "fp8_dense_nt_acc_fp32",
    "fp8_dense_nt_skip_head_mid", "bf16_mgroup_nt_contiguous",
    "bf16_mgroup_nn_contiguous", "bf16_mgroup_nt_contiguous_psum",
    "bf16_mgroup_nt_masked", "fp8_mgroup_nt_contiguous",
    "fp8_mgroup_nt_masked", "bf16_kgroup_tn_contiguous",
    "fp8_kgroup_nt_contiguous", "bf16_einsum_bmk_bnk_mn",
    "bf16_einsum_bhr_hdr_bhd", "bf16_einsum_bhd_hdr_bhr",
    "fp8_einsum_bhr_hdr_bhd", "tf32_hc_prenorm",
)
EXPECTED_CLASSES = ORIGINAL_CORE_CLASSES + (
    "fp8_dense_nn", "fp8_dense_tn", "fp8_mgroup_nt_contiguous_psum",
    "fp8_layout_transpose_fp32", "fp8_paged_mqa_metadata",
)
FORBIDDEN_MECHANISMS = frozenset({
    "fp8.kgroup.tn",
    "fp8.scale.ue8m0",
    "layout.pack_fp32_ue8m0",
    "mqa.full_attention",
    "mega_moe",
    "sm100.fp4",
})
_CLASS_MECHANISMS = {
    "bf16_dense_nn": ("bf16.dense.nn", "sm90.bf16_gemm"),
    "bf16_dense_nn_acc_bf16": ("bf16.dense.nn", "bf16.dense.acc.bf16", "sm90.bf16_gemm"),
    "bf16_dense_nt": ("bf16.dense.nt", "sm90.bf16_gemm"),
    "bf16_dense_nt_acc_bf16": ("bf16.dense.nt", "bf16.dense.acc.bf16", "sm90.bf16_gemm"),
    "bf16_dense_nt_acc_fp32": ("bf16.dense.nt", "bf16.dense.acc.fp32", "sm90.bf16_gemm"),
    "bf16_dense_tn": ("bf16.dense.tn", "sm90.bf16_gemm"),
    "bf16_dense_tn_acc_fp32": ("bf16.dense.tn", "bf16.dense.acc.fp32", "sm90.bf16_gemm"),
    "bf16_dense_tt": ("bf16.dense.tt", "sm90.bf16_gemm"),
    "bf16_dense_tt_acc_fp32": ("bf16.dense.tt", "bf16.dense.acc.fp32", "sm90.bf16_gemm"),
    "bf16_einsum_bhd_hdr_bhr": ("bf16.einsum.bhd_hdr_bhr", "sm90.bf16_bhd_hdr_bhr"),
    "bf16_einsum_bhr_hdr_bhd": ("bf16.einsum.bhr_hdr_bhd", "sm90.bf16_bhr_hdr_bhd"),
    "bf16_einsum_bmk_bnk_mn": ("bf16.einsum.bmk_bnk_mn", "sm90.bmn_bnk_mn_gemm"),
    "bf16_kgroup_tn_contiguous": ("bf16.kgroup.tn.contiguous", "sm90.bf16_k_grouped_gemm"),
    "bf16_mgroup_nn_contiguous": ("bf16.mgroup.nn.contiguous", "sm90.m_grouped_bf16_gemm_contiguous"),
    "bf16_mgroup_nt_contiguous": ("bf16.mgroup.nt.contiguous", "sm90.m_grouped_bf16_gemm_contiguous"),
    "bf16_mgroup_nt_contiguous_psum": (
        "bf16.mgroup.nt.contiguous", "bf16.mgroup.psum", "sm90.m_grouped_bf16_gemm_contiguous",
    ),
    "bf16_mgroup_nt_masked": ("bf16.mgroup.nt.masked", "sm90.bf16_m_grouped_gemm_masked"),
    "fp8_dense_nn": (
        "fp8.dense.nn.wrapper", "fp8.dense.nt", "fp8.scale.fp32.1d2d.identity", "sm90.fp8_gemm_1d2d",
    ),
    "fp8_dense_nt": ("fp8.dense.nt", "fp8.scale.fp32.1d2d.identity", "sm90.fp8_gemm_1d2d"),
    "fp8_dense_nt_acc_fp32": (
        "fp8.dense.acc.fp32", "fp8.dense.nt", "fp8.scale.fp32.1d1d.transpose", "sm90.fp8_gemm_1d1d",
    ),
    "fp8_dense_nt_skip_head_mid": (
        "fp8.dense.nt.skip_head_mid", "fp8.scale.fp32.1d2d.identity", "sm90.fp8_gemm_1d2d",
    ),
    "fp8_dense_tn": (
        "fp8.dense.nt", "fp8.dense.tn.wrapper", "fp8.scale.fp32.1d2d.identity", "sm90.fp8_gemm_1d2d",
    ),
    "fp8_einsum_bhr_hdr_bhd": ("fp8.einsum.bhr_hdr_bhd", "sm90.fp8_bmm"),
    "fp8_kgroup_nt_contiguous": (
        "fp8.kgroup.nt.contiguous", "fp8.scale.fp32.1d1d.transpose", "sm90.k_grouped_fp8_gemm_1d1d",
    ),
    "fp8_mgroup_nt_contiguous": (
        "fp8.mgroup.nt.contiguous", "fp8.scale.fp32.1d2d.identity",
        "sm90.m_grouped_fp8_gemm_contiguous_1d2d",
    ),
    "fp8_mgroup_nt_contiguous_psum": (
        "fp8.mgroup.nt.contiguous", "fp8.mgroup.psum", "fp8.scale.fp32.1d2d.identity",
        "sm90.m_grouped_fp8_gemm_contiguous_1d2d",
    ),
    "fp8_mgroup_nt_masked": (
        "fp8.mgroup.nt.masked", "fp8.scale.fp32.1d2d.identity",
        "sm90.fp8_m_grouped_gemm_masked_1d2d",
    ),
    "paged": ("mqa.logits.paged", "sm90.fp8_paged_mqa_logits"),
    "prefill_clean": (
        "mqa.clean_logits", "mqa.logits.prefill", "sm90.fp8_mqa_logits", "smxx.clean_logits",
    ),
    "prefill_compressed": ("mqa.compressed_window", "mqa.logits.prefill", "sm90.fp8_mqa_logits"),
    "tf32_hc_prenorm": ("sm90.tf32_hc_prenorm_gemm", "tf32.hc.prenorm"),
    "fp8_layout_transpose_fp32": ("layout.transpose_fp32", "smxx.layout.transpose_fp32"),
    "fp8_paged_mqa_metadata": ("mqa.paged.metadata", "sm90.paged_mqa_logits_metadata"),
}


class SuiteManifestError(ValueError):
    """Raised when a suite manifest is unsafe or incomplete."""


def parse_shape_json(value: Any) -> dict[str, Any]:
    """Parse a delivery shape object, including optional k-group `ks` lists."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SuiteManifestError(f"invalid shape JSON: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise SuiteManifestError("shape must be a non-empty object")
    parsed: dict[str, Any] = {}
    for key, item in value.items():
        if key == "ks":
            if not isinstance(item, list) or not item:
                raise SuiteManifestError("shape.ks must be a non-empty list")
            ks = []
            for entry in item:
                if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0 or entry % 128 != 0:
                    raise SuiteManifestError("shape.ks entries must be 128-aligned integers")
                ks.append(int(entry))
            parsed["ks"] = ks
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise SuiteManifestError(f"shape.{key} must be a positive integer")
        parsed[str(key)] = int(item)
    return parsed


def k_grouped_ks(shape: Mapping[str, Any]) -> list[int]:
    """Return per-group K sizes actually consumed by the k-grouped runner."""

    if "ks" in shape:
        ks = [int(entry) for entry in shape["ks"]]
        num_groups = int(shape["num_groups"])
        if len(ks) != num_groups:
            raise SuiteManifestError("shape.ks length must match num_groups")
        return ks
    return [int(shape["k_per_group"])] * int(shape["num_groups"])


def case_mechanisms(case: Mapping[str, Any]) -> list[str]:
    """Suite-local mechanism IDs for one delivery case."""

    class_id = case.get("class_id") or case.get("mode")
    if class_id not in _CLASS_MECHANISMS:
        raise SuiteManifestError(f"no mechanism map for {class_id}")
    ids = set(_CLASS_MECHANISMS[str(class_id)])
    shape = case.get("shape") or {}
    if isinstance(shape, Mapping) and "ks" in shape:
        ks = [int(entry) for entry in shape["ks"]]
        if len(set(ks)) > 1:
            ids.add("grouped.k.skew")
        # Small-K TMA in-place tensor-map update: many groups with K<=384.
        if ks and max(ks) <= 384 and len(ks) >= 8:
            ids.add("sm90.kgroup.tma_update.small_k")
    overlap = ids & FORBIDDEN_MECHANISMS
    if overlap:
        raise SuiteManifestError(f"invalid mechanism tags: {sorted(overlap)}")
    return sorted(ids)


def coverage_from_cases(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for case in cases:
        for mechanism in case_mechanisms(case):
            counts[mechanism] = counts.get(mechanism, 0) + 1
    return {
        "scope": "single_gpu_hopper",
        "selection": "mechanism_first",
        "mechanism_counts": dict(sorted(counts.items())),
        "notes": [
            "independent smxx_layout transpose_fp32 is launched via public get_mn_major_tma_aligned_tensor on contiguous non-MN-major FP32; it is a marked timed entry after packager rebuild",
            "independent sm90_paged_mqa_logits_metadata is launched via public get_paged_mqa_logits_metadata; paged logits cases still construct schedule_meta in prepare without timing that kernel and do not count as metadata coverage",
            "fp8 nn/tn wrappers and SM90 1d2d gran_mn=128 identity scales do not launch transpose_fp32; 1d1d may call it as a helper when SF is contiguous, but that helper is not independently timed",
            "pack_fp32_into_ue8m0 and UE8M0 GEMM scales are SM100-only on this revision and are not counted",
            "fp8 k_grouped tn, k-group psum, FP4, and MegaMoE are SM100 or out of single-GPU Hopper GEMM scope",
            "fp8 public nn/tn wrappers host-transpose into NT; they are not native SM90 MMA majors",
            "topology validates observed CTA/SM marker semantics; occupancy below 128 is valid for small launches; no artificial SM cap",
            "the 100k-5M cycle window is collection-level statistics, not a per-case pass/fail or exemption",
            "MQA cases are logits scoring (prefill clean/compressed and paged), not full attention",
            "reference_cycles/cycles_round*/cv/SASS on retained cases are prior collection history, not current qualification",
            "k-group cluster 1 or 2 is the pinned upstream SM90 heuristic result; the marker packager does not force cluster 2",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SuiteManifestError(f"manifest must be a JSON object: {path}")
    return value


def _validate_shape(shape: Mapping[str, Any], pool: str, index: int) -> None:
    if not isinstance(shape, Mapping):
        raise SuiteManifestError(f"shape_pools.{pool}[{index}] must be an object")
    for key, value in shape.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SuiteManifestError(
                f"shape_pools.{pool}[{index}].{key} must be a positive integer"
            )
    for key in ("k", "k_per_group", "alignment"):
        if key in shape and int(shape[key]) % 128 != 0:
            raise SuiteManifestError(
                f"shape_pools.{pool}[{index}].{key} must be a multiple of 128 on SM90"
            )


def _validate_family(family: Mapping[str, Any], pools: Mapping[str, Any]) -> None:
    class_id = family.get("class_id")
    if not isinstance(class_id, str) or not class_id:
        raise SuiteManifestError("every family needs a non-empty class_id")
    for key in ("api", "dtype", "layout", "shape_pool"):
        if not isinstance(family.get(key), str) or not family[key]:
            raise SuiteManifestError(f"{class_id}: missing {key}")
    pool = str(family["shape_pool"])
    if pool not in pools or not isinstance(pools[pool], list) or len(pools[pool]) < 4:
        raise SuiteManifestError(f"{class_id}: invalid shape pool {pool}")
    for index, shape in enumerate(pools[pool]):
        _validate_shape(shape, pool, index)


def validate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current full-GPU runner contract."""

    if config.get("schema_version") != SCHEMA_VERSION:
        raise SuiteManifestError("unsupported DeepGEMM suite schema")
    upstream = config.get("upstream")
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("name") != "DeepGEMM"
        or upstream.get("version") != "2.6.1"
        or upstream.get("revision") != EXPECTED_REVISION
    ):
        raise SuiteManifestError("manifest must pin the approved DeepGEMM revision")
    hardware = config.get("hardware")
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("compute_capability") != "9.0"
        or hardware.get("ptx_arch") != "sm_90a"
        or hardware.get("simulation_scope") not in (None, "full_gpu")
    ):
        raise SuiteManifestError("manifest must target the full-GPU SM90a domain")
    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise SuiteManifestError("selection is missing")
    if int(selection.get("minimum_cycles", -1)) != MIN_CYCLES:
        raise SuiteManifestError("minimum cycle bound must be 100000")
    if int(selection.get("maximum_cycles", -1)) != MAX_CYCLES:
        raise SuiteManifestError("maximum cycle bound must be 5000000")
    if (
        int(selection.get("warmups", -1)) != 3
        or int(selection.get("collection_trials", -1)) != 9
        or int(selection.get("verification_trials", -1)) != 9
    ):
        raise SuiteManifestError("warmup and trial counts must be 3/9/9")
    if selection.get("l2_cache_policy") != "warm_after_three_launches":
        raise SuiteManifestError("formal runs require three cache warmups")
    if (
        float(selection.get("maximum_cv_percent", -1)) != 3.0
        or float(selection.get("maximum_collection_verification_drift_percent", -1))
        != 3.0
    ):
        raise SuiteManifestError("stability limits must be CV=3% and drift=3%")
    modes = config.get("modes")
    if not isinstance(modes, Mapping) or set(modes) != {"normal"}:
        raise SuiteManifestError("manifest must expose only normal full-GPU mode")
    normal = modes["normal"]
    if (
        not isinstance(normal, Mapping)
        or normal.get("requested_num_sms") is not None
        or not bool(normal.get("representative_only"))
    ):
        raise SuiteManifestError("normal mode must use the full device SM count")
    pools = config.get("shape_pools")
    families = config.get("families")
    if not isinstance(pools, Mapping) or not isinstance(families, list):
        raise SuiteManifestError("families and shape_pools are required")
    ids = [row.get("class_id") for row in families if isinstance(row, Mapping)]
    if tuple(ids) != EXPECTED_CLASSES or len(set(ids)) != len(ids):
        raise SuiteManifestError("family order/count does not match the 25-class matrix")
    for family in families:
        if not isinstance(family, Mapping):
            raise SuiteManifestError("family rows must be objects")
        _validate_family(family, pools)
    reserves = config.get("reserves")
    if not isinstance(reserves, list) or not all(
        isinstance(row, str) and row for row in reserves
    ):
        raise SuiteManifestError("reserves must be a non-empty list of names")
    return dict(config)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return validate_manifest(_read_object(path.resolve()))


def family_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pools = config["shape_pools"]
    for family in config["families"]:
        for index, shape in enumerate(pools[family["shape_pool"]][:4]):
            row = dict(family)
            row.update(
                shape_index=index,
                shape=dict(shape),
                case_id=f"{family['class_id']}_s{index + 1}",
            )
            rows.append(row)
    return rows


def selected_case_rows(config: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode != "normal":
        raise SuiteManifestError(f"unsupported non-full-GPU mode: {mode}")
    return family_rows(config)


def coefficient_of_variation(values: Iterable[float]) -> float:
    samples = [float(value) for value in values]
    if not samples:
        return math.inf
    mean = statistics.fmean(samples)
    if len(samples) < 2 or mean == 0.0:
        return 0.0
    return statistics.stdev(samples) / mean * 100.0


# clock64 is a per-SM domain; max-min across SMs is not a duration.
def whole_kernel_span(marker_rows: Iterable[Mapping[str, Any]]) -> float:
    rows = list(marker_rows)
    if not rows:
        raise SuiteManifestError("marker sample is empty")
    by_sm: dict[int, list[tuple[int, int]]] = {}
    for row in rows:
        start = int(row["start_clock64"])
        end = int(row["end_clock64"])
        if start <= 0 or end <= start:
            raise SuiteManifestError("marker rows contain invalid clock64 values")
        by_sm.setdefault(int(row.get("sm_id", -1)), []).append((start, end))
    return float(
        max(
            max(end for _, end in pairs) - min(start for start, _ in pairs)
            for pairs in by_sm.values()
        )
    )


def summarize_marker_rows(marker_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in marker_rows]
    if not rows:
        raise SuiteManifestError("marker sample is empty")
    by_cta: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_cta.setdefault(int(row.get("cta_id", 0)), []).append(row)
    per_cta = []
    for cta_id, cta_rows in sorted(by_cta.items()):
        per_cta.append(
            {
                "cta_id": cta_id,
                "warp_count": len(cta_rows),
                "span_cycles": whole_kernel_span(cta_rows),
                "sm_ids": sorted({int(row.get("sm_id", -1)) for row in cta_rows}),
                "cluster_cta_ranks": sorted(
                    {int(row.get("cluster_cta_rank", 0)) for row in cta_rows}
                ),
                "cluster_cta_counts": sorted(
                    {int(row.get("cluster_cta_count", 1)) for row in cta_rows}
                ),
            }
        )
    spans = [float(row["span_cycles"]) for row in per_cta]
    return {
        "cta_count": len(per_cta),
        "warp_count_total": len(rows),
        "sm_ids": sorted({int(row.get("sm_id", -1)) for row in rows}),
        "cluster_cta_counts": sorted(
            {int(row.get("cluster_cta_count", 1)) for row in rows}
        ),
        "cluster_cta_ranks": sorted(
            {int(row.get("cluster_cta_rank", 0)) for row in rows}
        ),
        "whole_kernel_span": whole_kernel_span(rows),
        "cta_span_median": statistics.median(spans),
        "cta_span_p90": sorted(spans)[max(0, math.ceil(0.9 * len(spans)) - 1)],
        "per_cta": per_cta,
    }


def validate_cluster_sm_topology(
    target_ctas: int,
    topology: Mapping[str, Any],
    uses_cluster: bool | None = None,
) -> None:
    """1CTA, 2 ordinary CTAs (cluster_dim=1), or one 2-CTA cluster. Requested config is not enough."""

    sm_ids = sorted({int(sm) for sm in (topology.get("sm_ids") or [])})
    cta_count = int(topology.get("cta_count", 0))
    cluster_counts = [int(value) for value in (topology.get("cluster_cta_counts") or [])]
    ranks = [int(value) for value in (topology.get("cluster_cta_ranks") or [])]
    if not sm_ids or any(sm < 0 for sm in sm_ids):
        raise SuiteManifestError("cluster_sm produced no valid SM marker identities")
    if cta_count < 1:
        raise SuiteManifestError("cluster_sm produced no CTA markers")
    if target_ctas == 1:
        if cta_count != 1 or len(sm_ids) != 1:
            raise SuiteManifestError(
                f"1CTA topology expected 1 CTA on 1 SM, got cta={cta_count} sms={sm_ids}"
            )
        if cluster_counts and cluster_counts != [1]:
            raise SuiteManifestError(
                f"1CTA expected cluster_cta_count=[1], got {cluster_counts}"
            )
        return
    if target_ctas == 2:
        ordinary = cluster_counts == [1] and cta_count == 2 and 1 <= len(sm_ids) <= 2
        clustered = (
            cluster_counts == [2]
            and cta_count == 2
            and len(sm_ids) == 2
            and set(ranks) == {0, 1}
        )
        if uses_cluster is True:
            if not clustered:
                raise SuiteManifestError(
                    f"2CTA cluster expected cluster_cta_count=[2] ranks={{0,1}} "
                    f"2 SMs, got cta={cta_count} sms={sm_ids} counts={cluster_counts} ranks={ranks}"
                )
            return
        if uses_cluster is False:
            if clustered:
                raise SuiteManifestError(
                    f"noncluster 2 ordinary CTAs must not claim cluster/multicast, "
                    f"got counts={cluster_counts} ranks={ranks} sms={sm_ids}"
                )
            if not ordinary:
                raise SuiteManifestError(
                    f"2 ordinary CTAs expected grid=2 cluster_cta_count=[1] SM<=2, "
                    f"got cta={cta_count} sms={sm_ids} counts={cluster_counts} ranks={ranks}"
                )
            if set(ranks) - {0}:
                raise SuiteManifestError(
                    f"2 ordinary CTAs expected cluster ranks {{0}}, got {ranks}"
                )
            return
        if clustered or ordinary:
            return
        raise SuiteManifestError(
            f"2CTA expected a 2-CTA cluster or 2 ordinary CTAs, "
            f"got cta={cta_count} sms={sm_ids} counts={cluster_counts} ranks={ranks}"
        )
    raise SuiteManifestError(f"unsupported cluster_sm target_ctas={target_ctas}")

def validate_mode_topology(
    mode: str,
    topology: Mapping[str, Any],
    target_ctas: int | None = None,
    minimum_active_sms: int | None = None,
    uses_cluster: bool | None = None,
) -> None:
    """Check observed CTA/SM marker semantics. Do not require a 128-SM fill."""

    if mode == CLUSTER_SM_MODE:
        if target_ctas is None:
            raise SuiteManifestError("cluster_sm mode requires target_ctas")
        validate_cluster_sm_topology(int(target_ctas), topology, uses_cluster=uses_cluster)
        return
    if mode != "normal":
        raise SuiteManifestError(f"unsupported mode: {mode}")
    sm_ids = [int(sm) for sm in (topology.get("sm_ids") or [])]
    if not sm_ids or any(sm < 0 for sm in sm_ids):
        raise SuiteManifestError("normal mode produced no valid SM marker identities")
    cta_count = int(topology.get("cta_count", 0))
    if cta_count < 1:
        raise SuiteManifestError("normal mode produced no CTA markers")
    if cta_count < len(set(sm_ids)):
        raise SuiteManifestError("CTA count is below the number of observed SMs")
    if minimum_active_sms is not None and len(set(sm_ids)) < int(minimum_active_sms):
        raise SuiteManifestError(
            f"observed SM count {len(set(sm_ids))} is below launch occupancy {minimum_active_sms}"
        )
    if target_ctas is not None and cta_count < int(target_ctas):
        raise SuiteManifestError(f"CTA count {cta_count} is below launch target {target_ctas}")


# Archived dual-SM cluster shape helpers. They are not a delivery entry.
_CLUSTER_NUM_SMS = 2
_CLUSTER_IMPOSSIBLE = frozenset({
    "bf16_dense_nt", "bf16_dense_nn", "bf16_dense_tn", "bf16_dense_tt",
    "fp8_dense_nt", "fp8_dense_nt_skip_head_mid",
    "bf16_mgroup_nt_contiguous", "bf16_mgroup_nn_contiguous",
    "bf16_mgroup_nt_contiguous_psum", "bf16_mgroup_nt_masked",
    "fp8_mgroup_nt_contiguous", "fp8_mgroup_nt_masked",
    "bf16_einsum_bmk_bnk_mn", "bf16_einsum_bhr_hdr_bhd",
    "bf16_einsum_bhd_hdr_bhr", "fp8_einsum_bhr_hdr_bhd",
    "tf32_hc_prenorm",
})
_CLUSTER_ACC_DENSE = {
    "bf16_dense_nt_acc_bf16": (2, 2, True),
    "bf16_dense_nn_acc_bf16": (2, 2, True),
    "bf16_dense_nt_acc_fp32": (2, 4, False),
    "bf16_dense_tn_acc_fp32": (2, 4, False),
    "bf16_dense_tt_acc_fp32": (2, 4, False),
    "fp8_dense_nt_acc_fp32": (1, 4, False),
}
_BLOCK_NS_NOSF = tuple(range(16, 257, 16))
_BLOCK_NS_1D1D = (16, *range(24, 161, 16))
_CLUSTER_KGROUP = {
    "bf16_kgroup_tn_contiguous": (2, 4, False, _BLOCK_NS_NOSF),
    "fp8_kgroup_nt_contiguous": (1, 4, False, _BLOCK_NS_1D1D),
}
_DENSE_CLUSTER_SHAPES = (
    {"m": 128, "n": 2048, "k": 4096},
    {"m": 128, "n": 4096, "k": 8192},
    {"m": 256, "n": 2048, "k": 4096},
    {"m": 2048, "n": 128, "k": 4096},
    {"m": 1024, "n": 256, "k": 8192},
    {"m": 512, "n": 512, "k": 8192},
)
_KGROUP_CLUSTER_SHAPES = (
    {"m": 128, "n": 1024, "k_per_group": 4096, "num_groups": 4, "alignment": 128},
    {"m": 256, "n": 2048, "k_per_group": 4096, "num_groups": 4, "alignment": 128},
    {"m": 512, "n": 512, "k_per_group": 8192, "num_groups": 4, "alignment": 128},
    {"m": 1024, "n": 256, "k_per_group": 4096, "num_groups": 4, "alignment": 128},
    {"m": 256, "n": 2048, "k_per_group": 4096, "num_groups": 3, "alignment": 128},
    {"m": 256, "n": 2048, "k_per_group": 4096, "num_groups": 2, "alignment": 128},
)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _sm90_layout_cycles(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    cluster_m: int,
    cluster_n: int,
    *,
    acc: bool,
    es_ab: int,
    es_cd: int,
    groups: int,
    num_sms: int,
) -> float | None:
    cluster = cluster_m * cluster_n
    if cluster > 2 or block_m > 128 and block_n > 128:
        return None
    if num_sms % cluster != 0:
        return None
    num_blocks = _ceil_div(m, block_m) * _ceil_div(n, block_n) * groups
    num_waves = _ceil_div(num_blocks, num_sms)
    if cluster > 1 and num_waves <= 1:
        return math.inf
    l2_bw = int(min(64.0 * num_sms, 8e6 / 1.3e3))
    l1_bw = 128 * num_sms
    l2_ab = k * (block_m // cluster_n + block_n // cluster_m) * es_ab
    l1_ab = k * (block_m + block_n) * es_ab
    l1_tc = k * (max(64, block_m) + block_n) * es_ab + block_m * block_n * es_cd
    cd = block_m * block_n * es_cd * (2 if acc else 1)
    l2_cycles = (l2_ab + cd) * num_blocks // l2_bw
    l1_cycles = (l1_ab + l1_tc + cd) * num_blocks // l1_bw
    wave_efficiency = num_blocks / (num_waves * num_sms)
    return max(l1_cycles, l2_cycles) / wave_efficiency


def _predict_sm90_layout(
    m: int,
    n: int,
    k: int,
    *,
    acc: bool,
    es_ab: int,
    es_cd: int,
    groups: int = 1,
    num_sms: int = _CLUSTER_NUM_SMS,
    allow_bm256: bool = True,
    block_ns: tuple[int, ...] = _BLOCK_NS_NOSF,
) -> tuple[int, int, int, int] | None:
    block_ms = [64, 128] + ([256] if allow_bm256 and es_cd != 4 else [])
    if m <= 32:
        block_ms.append(32)
    if m <= 16:
        block_ms.append(16)
    best: tuple[float, int, int, int, int] | None = None
    for cluster_m, cluster_n in ((1, 1), (1, 2), (2, 1)):
        for block_m in block_ms:
            for block_n in block_ns:
                cycles = _sm90_layout_cycles(
                    m, n, k, block_m, block_n, cluster_m, cluster_n,
                    acc=acc, es_ab=es_ab, es_cd=es_cd, groups=groups, num_sms=num_sms,
                )
                if cycles is None:
                    continue
                if best is None or cycles < best[0]:
                    best = (cycles, cluster_m, cluster_n, block_m, block_n)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def predict_sm90_cluster(
    m: int,
    n: int,
    k: int,
    *,
    acc: bool,
    es_ab: int,
    es_cd: int,
    groups: int = 1,
    num_sms: int = _CLUSTER_NUM_SMS,
    allow_bm256: bool = True,
) -> tuple[int, int]:
    """Mirror SM90ArchSpec::get_best_config cluster_m/cluster_n at num_sms=2."""
    layout = _predict_sm90_layout(
        m, n, k, acc=acc, es_ab=es_ab, es_cd=es_cd, groups=groups,
        num_sms=num_sms, allow_bm256=allow_bm256,
    )
    if layout is None:
        return 1, 1
    return layout[0], layout[1]


def _kgroup_primary_tiles_even(
    m: int, n: int, cluster_m: int, cluster_n: int, block_m: int, block_n: int,
) -> bool:
    if cluster_n > 1:
        return _ceil_div(n, block_n) % 2 == 0
    if cluster_m > 1:
        return _ceil_div(m, block_m) % 2 == 0
    return True


def cluster_candidate_shapes(class_id: str) -> list[dict[str, Any]]:
    """Shapes expected to make SM90 get_best_config pick multicast=2 at num_sms=2.

    Empty list: class cannot produce a true cluster (do not emit cases).
    Each dict: {class_id, shape, cluster, multicast} with shape ready for --shape-json.
    """
    if class_id in _CLUSTER_IMPOSSIBLE or class_id not in EXPECTED_CLASSES:
        return []
    block_ns = _BLOCK_NS_NOSF
    if class_id in _CLUSTER_ACC_DENSE:
        es_ab, es_cd, allow_bm256 = _CLUSTER_ACC_DENSE[class_id]
        pool: tuple[dict[str, int], ...] = _DENSE_CLUSTER_SHAPES
        grouped = False
    elif class_id in _CLUSTER_KGROUP:
        es_ab, es_cd, allow_bm256, block_ns = _CLUSTER_KGROUP[class_id]
        pool = _KGROUP_CLUSTER_SHAPES
        grouped = True
    else:
        return []
    rows: list[dict[str, Any]] = []
    for shape in pool:
        groups = int(shape["num_groups"]) if grouped else 1
        if grouped and groups > 4:
            continue
        layout = _predict_sm90_layout(
            int(shape["m"]),
            int(shape["n"]),
            int(shape["k_per_group"] if grouped else shape["k"]),
            acc=True,
            es_ab=es_ab,
            es_cd=es_cd,
            groups=groups,
            allow_bm256=allow_bm256,
            block_ns=block_ns,
        )
        if layout is None:
            continue
        cluster_m, cluster_n, block_m, block_n = layout
        if cluster_m * cluster_n != 2:
            continue
        if grouped and not _kgroup_primary_tiles_even(
            int(shape["m"]), int(shape["n"]), cluster_m, cluster_n, block_m, block_n,
        ):
            continue
        rows.append({
            "class_id": class_id,
            "shape": dict(shape),
            "cluster": f"{cluster_m}x{cluster_n}x1",
            "multicast": 2,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DeepGEMM full-GPU manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    config = load_manifest(args.manifest)
    print(json.dumps({
        "status": "pass",
        "manifest": str(args.manifest.resolve()),
        "class_count": len(config["families"]),
        "normal_case_count": len(selected_case_rows(config, "normal")),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
