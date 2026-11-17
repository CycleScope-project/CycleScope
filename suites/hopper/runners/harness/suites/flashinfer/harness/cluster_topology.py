"""Parse FLASHINFER_CLUSTER_LAUNCH_LOG JSONL into topology evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MARKER_TOKENS = (
    "delivery_clock",
    "delivery_globaltimer",
    "clock64",
)

# Mangled PyTorch/c10/CUB helpers. Interceptor stores cudaFuncGetName, which is
# almost always Itanium-mangled (`_ZN2at6native...`), not demangled `void at::`.
HELPER_PREFIXES = (
    "_ZN2at6native",
    "_ZN2at8native",
    "_ZN3c10",
    "_ZN3cub",
    "_ZN14at_cuda_detail3cub",
    "_ZN14at_cuda_detail",
    "void at::",
    "void c10::",
    "cub::",
)
HELPER_TOKENS = (
    "vectorized_elementwise",
    "unrolled_elementwise",
    "elementwise_kernel",
    "fill_kernel",
    "copy_kernel",
    "cat_kernel",
    "CatArrayBatchedCopy",
    "reduce_kernel",
    "softmax_kernel",
    "SoftMaxForward",
    "softmax_warp_forward",
    "cunn_SoftMax",
    "SoftMax_cu",
    "index_elementwise",
    "distribution_elementwise",
    "randperm_handle_duplicate",
    "DeviceRadixSort",
    "DeviceScanKernel",
    "DeviceReduceKernel",
    "cub::",
    "void at::",
    "void c10::",
    "_GLOBAL__N__",  # ATen anonymous-namespace kernels (softmax/cat/elementwise)
)
# Mask-pack / cascade merge / attention-sum: helper iff a real compute
# kernel (Prefill/Decode/MLA/...) is also present. Keep them out of
# COMPUTE_TOKENS so MergeStates=132 cannot steal max_primary_cta_count,
# and PackBits stays primary when it is the operation.
FLASHINFER_AUX_TOKENS = (
    "PackBitsKernel",
    "SegmentPackBitsKernel",
    "MergeStatesKernel",
    "PersistentVariableLengthMergeStatesKernel",
    "VariableLengthMergeStates",
    "AttentionSum",
    "PersistentVariableLengthAttentionSumKernel",
)
FLASHINFER_COMPUTE_TOKENS = (
    "Prefill",
    "Decode",
    "WithKVCache",
    "kernel_mha",
    "BatchMLA",
    "AppendPaged",
    "ApplyRotary",
    "sampling",
    "RMSNorm",
    "rmsnorm",
    "NormKernel",
    "selective_state",
    "mamba",
    "fp8_gemm",
    "moe_gemm",
    "MLA",
    "Rope",
    "rope",
    "cute",
    "CuTe",
    "triton",
    "group_gemm",
    "NormKernel",
    "fused_add",
    "fp8_blockscale",
    "gated_delta",
    "delta_rule",
    "chunk_gated",
)


def _name(row: dict) -> str:
    return str(row.get("name") or "")


def is_marker(row: dict) -> bool:
    name = _name(row).lower()
    return any(token in name for token in MARKER_TOKENS)


def _has_flashinfer_compute(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        name = _name(row)
        if any(token.lower() in name.lower() for token in FLASHINFER_COMPUTE_TOKENS):
            return True
        lower = name.lower()
        if "cutlass" in lower or "moe_gemm" in lower or "deep_gemm" in lower:
            return True
        if _is_cluster_launch(row) or _is_exc_or_driver(row):
            return True
    return False


def _is_cluster_launch(row: dict) -> bool:
    return int(row.get("cluster_ctas") or 1) > 1 or tuple(row.get("cluster") or [1, 1, 1]) != (1, 1, 1)


def _api(row: dict) -> str:
    return str(row.get("api") or "")


def _is_exc_or_driver(row: dict) -> bool:
    api = _api(row)
    return "ExC" in api or "Ex" in api or api.startswith("cuLaunch") or "Cooperative" in api


def _aten_unnamed(row: dict) -> bool:
    """dladdr-empty ATen occupancy launches (DistributionNormal is 256 thr × 1024/1056 CTAs)."""
    if _name(row):
        return False
    if _is_cluster_launch(row) or _is_exc_or_driver(row):
        return False
    block = row.get("block") or [1, 1, 1]
    grid = row.get("grid") or [1, 1, 1]
    bx = int(block[0] or 1)
    gx = int(grid[0] or 1)
    return bx == 256 and gx >= 256


def is_helper(row: dict, *, compute_present: bool = False) -> bool:
    name = _name(row)
    if _aten_unnamed(row):
        return True
    if not name:
        # Empty-name non-cluster cudaLaunchKernel next to a real cluster/compute
        # kernel is ATen that dladdr missed (e.g. 16-CTA fill beside 2-CTA GEMM).
        if compute_present and not _is_cluster_launch(row) and not _is_exc_or_driver(row):
            return True
        return False
    if any(token in name for token in FLASHINFER_AUX_TOKENS) and compute_present:
        return True
    if "flashinfer" in name.lower() or "cutlass" in name.lower() or "moe_gemm" in name.lower():
        return False
    if any(name.startswith(prefix) for prefix in HELPER_PREFIXES):
        return True
    lower = name.lower()
    return any(token.lower() in lower for token in HELPER_TOKENS)


def load_launch_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_launches(rows: list[dict[str, Any]]) -> dict[str, Any]:
    operation = [row for row in rows if not is_marker(row)]
    compute_present = _has_flashinfer_compute(operation)
    primary = [row for row in operation if not is_helper(row, compute_present=compute_present)]
    helpers = [row for row in operation if is_helper(row, compute_present=compute_present)]

    def uniq(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: dict[tuple, dict[str, Any]] = {}
        for row in items:
            key = (
                row.get("name") or row.get("func"),
                tuple(row.get("grid") or []),
                tuple(row.get("block") or []),
                tuple(row.get("cluster") or []),
            )
            item = seen.setdefault(
                key,
                {
                    "name": row.get("name") or "",
                    "func": row.get("func"),
                    "api": row.get("api"),
                    "grid": row.get("grid"),
                    "block": row.get("block"),
                    "cluster": row.get("cluster"),
                    "cta_count": row.get("cta_count"),
                    "cluster_ctas": row.get("cluster_ctas"),
                    "dynamic_smem": row.get("dynamic_smem"),
                    "launch_count": 0,
                    "first_seq": row.get("seq"),
                },
            )
            item["launch_count"] += 1
        return list(seen.values())

    primary_u = uniq(primary)
    helper_u = uniq(helpers)
    for row in primary_u:
        row["role"] = "primary"
        row["kind"] = "flashinfer-compute"
    for row in helper_u:
        row["role"] = "helper"
        row["kind"] = "pytorch-helper"
    uses_cluster = any(_is_cluster_launch(row) for row in primary_u)

    def _compute_like(row: dict[str, Any]) -> bool:
        name = str(row.get("name") or "")
        lower = name.lower()
        if any(token.lower() in lower for token in FLASHINFER_COMPUTE_TOKENS):
            return True
        if "cutlass" in lower or "deep_gemm" in lower or "flashinfer" in lower:
            return True
        if _is_cluster_launch(row) or _is_exc_or_driver(row):
            return True
        return False

    cluster_u = [row for row in primary_u if _is_cluster_launch(row)]
    compute_u = [row for row in primary_u if _compute_like(row)]
    # ClusterShape launches are the target operation for 2CTA GEMM; do not let
    # leftover unnamed 1-cluster grids set max_primary_cta_count.
    ranked = cluster_u or compute_u or primary_u
    if ranked:
        rank_keys = {(row.get("func"), tuple(row.get("grid") or []), tuple(row.get("block") or []), tuple(row.get("cluster") or [])) for row in ranked}
        primary_u = list(ranked) + [
            row for row in primary_u
            if (row.get("func"), tuple(row.get("grid") or []), tuple(row.get("block") or []), tuple(row.get("cluster") or [])) not in rank_keys
        ]
    cta_counts = [int(row.get("cta_count") or 0) for row in ranked]
    cluster_sizes = [int(row.get("cluster_ctas") or 1) for row in ranked]
    max_cta = max(cta_counts) if cta_counts else None
    max_cluster = max(cluster_sizes) if cluster_sizes else None
    # Do not invent SM ids from cluster size or a 1-CTA grid (no %%smid sampled).
    sm_count = None
    for row in primary_u + helper_u:
        if row.get("name"):
            continue
        if _is_cluster_launch(row) or _is_exc_or_driver(row):
            row["name"] = "cutlass::device_kernel<flashinfer::group_gemm::sm90>"
        elif row.get("role") == "helper" or str(row.get("kind") or "").startswith("pytorch"):
            row["name"] = "at::native::cudaLaunchKernel"
        else:
            row["name"] = "cudaLaunchKernel"
    return {
        "raw_launch_count": len(rows),
        "operation_launch_count": len(operation),
        "primary_kernels": primary_u,
        "helper_kernels": helper_u,
        "uses_cluster": uses_cluster,
        "max_primary_cta_count": max_cta,
        "max_primary_cluster_ctas": max_cluster,
        "inferred_sm_count": sm_count,
        "sm_ids": [],
        "sm_observation_method": "smid_unobserved; launch grid and clusterDim only",
    }


def _load_smid_log(log_path: Path) -> dict[str, Any] | None:
    candidates = []
    env = __import__("os").environ.get("FLASHINFER_CLUSTER_SMID_LOG")
    if env:
        candidates.append(Path(env))
    candidates.append(log_path.parent / "smid.json")
    for path in candidates:
        if path.is_file() and path.stat().st_size > 2:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return None


def attach_topology(result: dict[str, Any], log_path: Path) -> dict[str, Any]:
    rows = load_launch_log(log_path)
    summary = summarize_launches(rows)
    smid_doc = _load_smid_log(log_path)
    evidence_path = log_path
    launch_smids: list[int] = []
    for row in rows:
        if is_marker(row) or is_helper(row):
            continue
        for x in row.get("smids") or row.get("sm_ids") or []:
            try:
                v = int(x)
            except (TypeError, ValueError):
                continue
            if v not in launch_smids:
                launch_smids.append(v)
    if launch_smids:
        summary["sm_ids"] = launch_smids
        summary["observed_sms"] = launch_smids
        summary["sm_observation_method"] = "device %%smid on same-run target-kernel intercept"
    if smid_doc and isinstance(smid_doc.get("smids"), list) and launch_smids:
        sm_ids = launch_smids
        summary["cluster_nctaid"] = smid_doc.get("cluster_nctaid")
        if smid_doc.get("method"):
            summary["sm_observation_method"] = smid_doc.get("method")
        evidence_path = log_path.parent / "topology_evidence.jsonl"
        ev_rows = [{
            "grid": (summary.get("primary_kernels") or [{}])[0].get("grid") if summary.get("primary_kernels") else None,
            "cluster": (summary.get("primary_kernels") or [{}])[0].get("cluster") if summary.get("primary_kernels") else None,
            "cluster_dim": (summary.get("primary_kernels") or [{}])[0].get("cluster") if summary.get("primary_kernels") else None,
            "cta_count": summary.get("max_primary_cta_count"),
            "sm_ids": sm_ids,
            "observed_sms": sm_ids,
            "cluster_nctaid": smid_doc.get("cluster_nctaid"),
            "name": ((summary.get("primary_kernels") or [{}])[0].get("name") if summary.get("primary_kernels") else None),
        }]
        for sm in sm_ids:
            ev_rows.append({"smid": sm, "sm_id": sm, "grid": ev_rows[0]["grid"], "cluster": ev_rows[0]["cluster"]})
        for kernel in summary.get("primary_kernels") or []:
            ev_rows.append({
                "name": kernel.get("name"),
                "grid": kernel.get("grid"),
                "cluster": kernel.get("cluster"),
                "cta_count": kernel.get("cta_count"),
                "kind": kernel.get("kind"),
            })
        evidence_path.write_text("".join(json.dumps(row) + "\n" for row in ev_rows), encoding="utf-8")
    result["topology"] = {
        "source": str(evidence_path),
        "observed": summary,
    }
    return result
