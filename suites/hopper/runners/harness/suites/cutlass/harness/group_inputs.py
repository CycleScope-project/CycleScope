#!/usr/bin/env python3
"""Resolve and validate every build input for a CUTLASS delivery group."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class GroupRequirement:
    binary_name: str
    source: str
    kind: str


SUPPORTED_TARGETS = {
    "delivery53_permute_full_gpu": (
        "harness/derived_53_hopper_gemm_permute.cu",
        "upstream_example",
    ),
    "dense_algorithms": ("harness/cutlass_full_gpu_runner.cu", "dense_harness"),
    "example49_collective_builder": (
        "upstream/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu",
        "upstream_example",
    ),
    "example50_epilogue_swizzle": (
        "upstream/examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle.cu",
        "upstream_example",
    ),
    "example52_gather_scatter": (
        "upstream/examples/52_hopper_gather_scatter_fusion/52_hopper_gather_scatter_fusion.cu",
        "upstream_example",
    ),
    "example54_fp8_fused": (
        "upstream/examples/54_hopper_fp8_warp_specialized_gemm/54_hopper_fp8_warp_specialized_gemm.cu",
        "upstream_example",
    ),
    "example55_mixed_f16_fp8": (
        "upstream/examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm.cu",
        "upstream_example",
    ),
    "example55_int4_bf16": (
        "upstream/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu",
        "upstream_example",
    ),
    "example55_int4_fp8": (
        "upstream/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu",
        "upstream_example",
    ),
    "example56_ptr_array": (
        "upstream/examples/56_hopper_ptr_array_batched_gemm/56_hopper_ptr_array_batched_gemm.cu",
        "upstream_example",
    ),
    "example57_grouped_fastacc": (
        "upstream/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu",
        "upstream_example",
    ),
    "example61_topk_softmax": (
        "upstream/examples/61_hopper_gemm_with_topk_and_softmax/61_hopper_gemm_with_topk_and_softmax.cu",
        "upstream_example",
    ),
    "example62_sparse": (
        "upstream/examples/62_hopper_sparse_gemm/62_hopper_sparse_gemm.cu",
        "upstream_example",
    ),
    "example63_weight_prefetch": (
        "upstream/examples/63_hopper_gemm_with_weight_prefetch/63_hopper_gemm_with_weight_prefetch.cu",
        "upstream_example",
    ),
    "example67_blockwise": (
        "upstream/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu",
        "upstream_example",
    ),
    "example67_groupwise_streamk": (
        "upstream/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu",
        "upstream_example",
    ),
    "example68_grouped_blockwise_coop": (
        "upstream/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling.cu",
        "upstream_example",
    ),
    "example68_grouped_blockwise_pingpong": (
        "upstream/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling_with_sparse_groups.cu",
        "upstream_example",
    ),
    "example69_grouped_mixed": (
        "upstream/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mixed_dtype_grouped_gemm.cu",
        "upstream_example",
    ),
    "example69_grouped_int4_bf16": (
        "upstream/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_bf16_grouped_gemm.cu",
        "upstream_example",
    ),
    "example69_grouped_int4_fp8": (
        "upstream/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu",
        "upstream_example",
    ),
}

SUITE_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SM_LIMIT_TOKENS = (
    "sm_count",
    "min_sm",
    "num_sm",
    "device_sms",
    "active_sm",
    "multiprocessor_limit",
)


def case_has_forbidden_sm_limit(case: dict) -> bool:
    joined = " ".join(str(arg).lower() for arg in case.get("args") or [])
    return any(token in joined for token in FORBIDDEN_SM_LIMIT_TOKENS)


def benchmark_path_for_arg(argument: str, source_root: Path = SUITE_ROOT) -> Path:
    relative = argument.split("=", 1)[1]
    path = Path(relative)
    if not path.is_absolute():
        path = (source_root / relative).resolve()
    return path


def resolve_case_args(args: list[str], source_root: Path = SUITE_ROOT) -> list[str]:
    resolved = []
    for argument in args:
        if argument.startswith("--benchmark="):
            resolved.append(f"--benchmark={benchmark_path_for_arg(argument, source_root)}")
        else:
            resolved.append(argument)
    return resolved


def load_group_requirements(manifest: Path, group: int) -> tuple[list[dict], list[GroupRequirement]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload.get("cases", []) if int(row["delivery_group"]) == group]
    if not rows:
        raise RuntimeError(f"CUTLASS manifest group {group} is empty")
    by_binary: dict[str, GroupRequirement] = {}
    for row in rows:
        if case_has_forbidden_sm_limit(row):
            raise RuntimeError(f"{row.get('case_id')}: artificial SM limit is not allowed")
        for argument in row.get("args") or []:
            if str(argument).startswith("--benchmark="):
                path = benchmark_path_for_arg(str(argument), SUITE_ROOT)
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RuntimeError(f"{row.get('case_id')}: missing group benchmark {path}")
        requirement = GroupRequirement(
            binary_name=str(row.get("binary_name") or ""),
            source=str(row.get("source") or ""),
            kind=str(row.get("kind") or ""),
        )
        if not all((requirement.binary_name, requirement.source, requirement.kind)):
            raise RuntimeError(f"{row.get('case_id')}: incomplete CUTLASS build input")
        expected = SUPPORTED_TARGETS.get(requirement.binary_name)
        if expected != (requirement.source, requirement.kind):
            raise RuntimeError(
                f"{row.get('case_id')}: unsupported or inconsistent CUTLASS target "
                f"{requirement.binary_name} source={requirement.source} kind={requirement.kind}"
            )
        previous = by_binary.setdefault(requirement.binary_name, requirement)
        if previous != requirement:
            raise RuntimeError(
                f"CUTLASS target {requirement.binary_name} has conflicting sources in group {group}"
            )
    return rows, sorted(by_binary.values(), key=lambda item: item.binary_name)


def _regular_nonempty(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def validate_group_inputs(
    rows: list[dict], *, artifact_root: Path, source_root: Path
) -> dict[str, dict[str, Path]]:
    artifact_root = artifact_root.resolve()
    source_root = source_root.resolve()
    checked: dict[str, dict[str, Path]] = {}
    errors: list[str] = []
    for row in rows:
        case_id = str(row["case_id"])
        if case_has_forbidden_sm_limit(row):
            errors.append(f"{case_id}: artificial SM limit is not allowed")
        binary = artifact_root / "build" / str(row["binary_name"])
        sass = binary.with_suffix(".sass")
        source = source_root / str(row["source"])
        try:
            source.resolve(strict=False).relative_to(source_root)
        except ValueError:
            errors.append(f"{case_id}: source escapes source root: {source}")
            continue
        if not (_regular_nonempty(binary) and os.access(binary, os.X_OK)):
            errors.append(f"{case_id}: binary is missing, empty, symlinked, or not executable: {binary}")
        if not _regular_nonempty(sass):
            errors.append(f"{case_id}: selected SASS is missing, empty, or symlinked: {sass}")
        if not _regular_nonempty(source):
            errors.append(f"{case_id}: source is missing, empty, or symlinked: {source}")
        for argument in row.get("args") or []:
            if str(argument).startswith("--benchmark="):
                bench = benchmark_path_for_arg(str(argument), source_root)
                if not _regular_nonempty(bench):
                    errors.append(f"{case_id}: group benchmark is missing or empty: {bench}")
        checked[case_id] = {"binary": binary, "sass": sass, "source": source}
    if errors:
        raise RuntimeError("CUTLASS group input preflight failed before capture:\n" + "\n".join(errors))
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--group", required=True, type=int, choices=range(4))
    args = parser.parse_args()
    _, requirements = load_group_requirements(args.manifest, args.group)
    for requirement in requirements:
        print(
            "\t".join(
                (requirement.binary_name, requirement.source, requirement.kind)
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
