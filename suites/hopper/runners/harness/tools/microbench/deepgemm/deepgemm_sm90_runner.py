"""Run one case from the formal DeepGEMM SM90 full-GPU suite.

Formal rows are timed with per-warp in-kernel ``clock64`` and
``globaltimer`` markers. Cross-SM duration uses globaltimer converted by
the per-launch clock64/globaltimer ratio. CUDA events are retained only
for workload pilots; an event-timed row can never pass the formal suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

COMMON_ROOT = Path(__file__).resolve().parents[3] / "suites" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
from matched_kernel_timing import (  # noqa: E402
    load_comparison_contract,
    validate_marker_code_identity,
)

from .deepgemm_sm90_suite import (
    CLUSTER_SM_MAX_CYCLES,
    CLUSTER_SM_MIN_CYCLES,
    CLUSTER_SM_MODE,
    DEFAULT_MANIFEST,
    EXPECTED_REVISION,
    MAX_CYCLES,
    MIN_CYCLES,
    MODES,
    SuiteManifestError,
    coefficient_of_variation,
    k_grouped_ks,
    load_manifest,
    parse_shape_json,
    summarize_marker_rows,
    validate_mode_topology,
)
RUNNER_MODES = MODES + (CLUSTER_SM_MODE,)


RESULT_SCHEMA = "deepgemm-sm90-suite-result-v2"
MARKER_SCHEMA = "deepgemm-sm90-suite-marker-v2"
MARKER_FIELDS = 7
MAX_MARKER_CTAS = 16_384
MAX_WARPS_PER_CTA = 16
MARKER_WORDS = MAX_MARKER_CTAS * MAX_WARPS_PER_CTA * MARKER_FIELDS
def _delivery_raw_per_recorded_sample() -> int:
    """Return the number of consecutive raw spans behind one formal sample."""

    value = int(os.environ.get("DEEPGEMM_DELIVERY_RAW_PER_SAMPLE", "1"))
    if value not in (1, 3, 4):
        raise RuntimeError("DEEPGEMM_DELIVERY_RAW_PER_SAMPLE must be 1, 3, or 4")
    return value


def _consecutive_medians(values: Sequence[float], group_size: int) -> list[float]:
    """Reduce consecutive, non-overlapping raw spans without dropping any span."""

    samples = [float(value) for value in values]
    if group_size == 1:
        return samples
    if not samples or len(samples) % group_size:
        raise RuntimeError(
            f"cannot form consecutive groups of {group_size} from {len(samples)} samples"
        )
    return [
        float(statistics.median(samples[index:index + group_size]))
        for index in range(0, len(samples), group_size)
    ]


@dataclass
class PreparedCase:
    launch: Callable[[], Any]
    reset: Callable[[], None]
    output: Any
    reference: Any
    marker: torch.Tensor
    facts: dict[str, Any]
    correctness: Callable[[], float | None]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_deepgemm(package: Path, cache_root: Path):
    package = package.resolve()
    if not package.is_dir():
        raise RuntimeError(f"DeepGEMM package is missing: {package}")
    os.environ["DG_JIT_CACHE_DIR"] = str(cache_root.resolve())
    sys.path.insert(0, str(package))
    import deep_gemm  # type: ignore
    from .actual_binding import preserve_device_library_root
    preserve_device_library_root(deep_gemm, package)

    return deep_gemm


def _marker_metadata(package: Path) -> dict[str, Any] | None:
    path = package / "deep_gemm" / "sm90_suite_marker.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid marker metadata: {path}")
    if value.get("schema_version") != MARKER_SCHEMA:
        raise RuntimeError(f"unsupported marker package schema: {value.get('schema_version')}")
    if value.get("deepgemm_revision") != EXPECTED_REVISION:
        raise RuntimeError("marker package does not match the pinned DeepGEMM revision")
    if int(value.get("max_ctas", -1)) != MAX_MARKER_CTAS:
        raise RuntimeError("marker package CTA capacity does not match the runner")
    if int(value.get("max_warps_per_cta", -1)) != MAX_WARPS_PER_CTA:
        raise RuntimeError("marker package warp capacity does not match the runner")
    expected_fields = [
        "start_clock64",
        "end_clock64",
        "start_globaltimer",
        "end_globaltimer",
        "sm_id",
        "cluster_cta_rank",
        "cluster_cta_count",
    ]
    if value.get("marker_fields") != expected_fields:
        raise RuntimeError("marker package fields do not match the runner")
    return value



def _nvidia_snapshot() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(query, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    fields = [field.strip() for field in line.split(",")]
    snapshot: dict[str, Any] = {"raw": completed.stdout, "returncode": completed.returncode}
    if len(fields) == 5:
        snapshot.update(
            {
                "name": fields[0],
                "driver_version": fields[1],
                "memory_total_mib": int(float(fields[2])),
                "max_sm_clock_mhz": int(float(fields[3])),
                "max_memory_clock_mhz": int(float(fields[4])),
            }
        )
    return snapshot


def _gpu_profile_matches(name: str, profile: str) -> bool:
    if profile == "h100":
        return "H100" in name
    if profile == "h200":
        return "H200" in name
    if profile == "h20":
        return "H20" in name and "H200" not in name
    raise RuntimeError(f"unsupported GEMM_DELIVERY_GPU profile: {profile}")


def _selected_gpu_profile() -> str:
    return os.environ.get("GEMM_DELIVERY_GPU", "h100").lower()


def _minimum_full_gpu_sms(sm_count: int) -> int:
    return max(1, int(sm_count * 0.97))


def _normal_span_cycles(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise RuntimeError("invalid cross-SM globaltimer marker rows")
    ratios = []
    for row in rows:
        clock_delta = int(row["end_clock64"]) - int(row["start_clock64"])
        timer_delta = int(row["end_globaltimer"]) - int(row["start_globaltimer"])
        if clock_delta <= 0 or timer_delta <= 0:
            raise RuntimeError("marker row lacks paired positive clock and globaltimer")
        ratios.append(clock_delta / timer_delta)
    global_start = min(int(row["start_globaltimer"]) for row in rows)
    global_end = max(int(row["end_globaltimer"]) for row in rows)
    if global_end <= global_start:
        raise RuntimeError("invalid cross-SM globaltimer marker rows")
    return float((global_end - global_start) * statistics.median(ratios))


def _event_cycles(seconds: float, snapshot: Mapping[str, Any]) -> float:
    mhz = float(snapshot.get("max_sm_clock_mhz") or 0.0)
    return seconds * mhz * 1_000_000.0 if mhz > 0 else math.nan


def _source_generators(package: Path):
    candidates = [
        package / "source_tests",
        package.parent / "source_single_cta_grouped" / "tests",
        package.parent / "source" / "DeepGEMM" / "tests",
        package.parent / "source" / "tests",
    ]
    for path in candidates:
        if not path.is_dir():
            continue
        sys.path.insert(0, str(path))
        try:
            import generators  # type: ignore

            return generators
        except Exception:
            continue
    raise RuntimeError("pinned DeepGEMM test generators are unavailable")


def _marked_tensor(shape: Sequence[int], dtype: torch.dtype, *, fill: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    numel = math.prod(int(value) for value in shape)
    element_size = torch.empty((), dtype=dtype).element_size()
    logical_bytes = numel * element_size
    if logical_bytes % 8:
        raise RuntimeError(f"logical output byte size is not uint64 aligned: {logical_bytes}")
    storage = torch.empty(logical_bytes + MARKER_WORDS * 8, device="cuda", dtype=torch.uint8)
    output = storage[:logical_bytes].view(dtype).view(tuple(int(value) for value in shape))
    marker = storage[logical_bytes:].view(torch.int64).view(
        MAX_MARKER_CTAS, MAX_WARPS_PER_CTA, MARKER_FIELDS
    )
    if fill is not None:
        output.fill_(fill)
    marker.zero_()
    return output, marker


def _calc_diff(actual: Any, expected: Any) -> float | None:
    if actual is None or expected is None:
        return None
    try:
        lhs = actual.detach().double()
        rhs = expected.detach().double()
        denominator = (lhs.square() + rhs.square()).sum().item()
        if denominator == 0.0:
            return 0.0
        return float(1.0 - 2.0 * (lhs * rhs).sum().item() / denominator)
    except Exception:
        return None


def _layout_types(generators: Any, layout: str):
    code = layout.upper()
    return (
        generators.MajorTypeAB.KMajor if code[0] == "N" else generators.MajorTypeAB.MNMajor,
        generators.MajorTypeAB.KMajor if code[1] == "T" else generators.MajorTypeAB.MNMajor,
    )


def _alias_inputs(a: Any, b: Any, major_a: Any, major_b: Any) -> tuple[Any, Any]:
    alias_a = a if major_a.is_k_major() else (a[0].mT, a[1].mT) if isinstance(a, tuple) else a.mT
    alias_b = b if major_b.is_k_major() else (b[0].mT, b[1].mT) if isinstance(b, tuple) else b.mT
    return alias_a, alias_b


def _prepare_dense(deep_gemm: Any, generators: Any, row: Mapping[str, Any]) -> PreparedCase:
    shape = row["shape"]
    m, n, k = (int(shape[key]) for key in ("m", "n", "k"))
    major_a, major_b = _layout_types(generators, str(row["layout"]))
    epilogue = dict(row.get("epilogue") or {})
    accumulate = bool(epilogue.get("accumulate", False))
    out_dtype = torch.float if epilogue.get("output_dtype") == "fp32" else torch.bfloat16
    is_bf16 = row["dtype"] == "bf16"
    if is_bf16:
        kernel_type = generators.KernelType.KernelNoSF
        quant_config = None
    else:
        kernel_type = (
            generators.KernelType.Kernel1D1D
            if accumulate
            else generators.KernelType.Kernel1D2D
        )
        quant_config = generators.QuantConfig()
    use_ue8m0 = False if is_bf16 else bool(generators.get_ue8m0_usage(kernel_type))
    a, b, c, generated_d, reference = generators.generate_normal(
        m,
        n,
        k,
        major_a,
        major_b,
        accumulate,
        out_dtype,
        kernel_type,
        use_ue8m0=use_ue8m0,
        use_bf16=is_bf16,
        quant_config=quant_config,
    )
    output, marker = _marked_tensor((m, n), out_dtype)
    initial = generated_d.detach().clone() if accumulate else None
    if initial is not None:
        output.copy_(initial)
        c = output
    else:
        c = None
    alias_a, alias_b = _alias_inputs(a, b, major_a, major_b)
    api = str(row["api"])
    recipe = recipe_a = recipe_b = None
    if quant_config is not None:
        recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=accumulate)

    def reset() -> None:
        marker.zero_()
        if initial is not None:
            output.copy_(initial)

    def launch() -> Any:
        function = getattr(deep_gemm, api)
        if is_bf16:
            return function(alias_a, alias_b, output, c=c)
        return function(
            alias_a,
            alias_b,
            output,
            c=c,
            disable_ue8m0_cast=not use_ue8m0,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

    return PreparedCase(
        launch,
        reset,
        output,
        reference,
        marker,
        {
            "shape": dict(shape),
            "api": api,
            "accumulate": accumulate,
            "output_dtype": str(out_dtype),
            "kernel_type": kernel_type.name,
            "major_a": major_a.name,
            "major_b": major_b.name,
        },
        lambda: _calc_diff(output, reference),
    )


def _apply_skip_head_mid(tensor: torch.Tensor, splits: tuple[int, int, int]) -> torch.Tensor:
    left, mid, right = splits
    m, n = tensor.shape
    if n % (left + right):
        raise RuntimeError("skip-head-mid N is not divisible by the stored head width")
    heads = n // (left + right)
    logical = tensor.view(m, heads, left + right)
    zeros = torch.zeros((m, heads, mid), dtype=tensor.dtype, device=tensor.device)
    return torch.cat((logical[:, :, :left], zeros, logical[:, :, left:]), dim=2).view(m, -1)


def _prepare_skip_head_mid(deep_gemm: Any, generators: Any, row: Mapping[str, Any]) -> PreparedCase:
    shape = row["shape"]
    m, n, k = (int(shape[key]) for key in ("m", "n", "k"))
    splits = tuple(int(value) for value in row["metadata"]["head_splits"])
    kernel_type = generators.KernelType.Kernel1D2D
    major = generators.MajorTypeAB.KMajor
    use_ue8m0 = bool(generators.get_ue8m0_usage(kernel_type))
    a, b, _, generated_d, reference = generators.generate_normal(
        m,
        n,
        k,
        major,
        major,
        False,
        torch.bfloat16,
        kernel_type,
        use_ue8m0=use_ue8m0,
        quant_config=generators.QuantConfig(),
    )
    expanded_reference = _apply_skip_head_mid(reference, splits)
    expanded_shape = tuple(expanded_reference.shape)
    output, marker = _marked_tensor(expanded_shape, torch.bfloat16, fill=0.0)

    def reset() -> None:
        marker.zero_()
        output.zero_()

    def launch() -> Any:
        return deep_gemm.fp8_gemm_nt_skip_head_mid(
            a, b, output, splits, disable_ue8m0_cast=not use_ue8m0
        )

    return PreparedCase(
        launch,
        reset,
        output,
        expanded_reference,
        marker,
        {"shape": dict(shape), "output_shape": list(expanded_shape), "api": row["api"], "head_splits": list(splits)},
        lambda: _calc_diff(output, expanded_reference),
    )


def _prepare_m_grouped(deep_gemm: Any, generators: Any, row: Mapping[str, Any]) -> PreparedCase:
    shape = row["shape"]
    is_masked = (row.get("metadata") or {}).get("mask_mode") == "masked"
    is_bf16 = row["dtype"] == "bf16"
    use_psum = bool((row.get("metadata") or {}).get("use_psum_layout", False))
    alignment = int(shape["alignment"])
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    major_a, major_b = _layout_types(generators, str(row["layout"]))
    kernel_type = generators.KernelType.KernelNoSF if is_bf16 else generators.KernelType.Kernel1D2D
    quant_config = None if is_bf16 else generators.QuantConfig()
    use_ue8m0 = False if is_bf16 else bool(generators.get_ue8m0_usage(kernel_type))
    api = str(row["api"])

    if is_masked:
        num_groups, max_m, expected_m, n, k = (
            int(shape[key]) for key in ("num_groups", "max_m", "expected_m", "n", "k")
        )
        a, b, masked_m, _, _, reference = generators.generate_m_grouped_masked(
            num_groups,
            max_m,
            expected_m,
            n,
            k,
            use_ue8m0=use_ue8m0,
            use_bf16=is_bf16,
            use_psum_layout=False,
            quant_config=quant_config,
        )
        output, marker = _marked_tensor((num_groups, max_m, n), torch.bfloat16)
        recipe = recipe_a = recipe_b = None
        if quant_config is not None:
            recipe, recipe_a, recipe_b = quant_config.get_recipes()
        api_expected_m = expected_m if is_bf16 else int(expected_m * 1.2)

        def reset() -> None:
            marker.zero_()

        def launch() -> Any:
            function = getattr(deep_gemm, api)
            if is_bf16:
                return function(a, b, output, masked_m, api_expected_m)
            return function(
                a,
                b,
                output,
                masked_m,
                api_expected_m,
                disable_ue8m0_cast=not use_ue8m0,
                recipe=recipe,
                recipe_a=recipe_a,
                recipe_b=recipe_b,
            )

        def correctness() -> float | None:
            actual_parts = []
            reference_parts = []
            for group in range(num_groups):
                valid_m = int(masked_m[group].item())
                if valid_m:
                    actual_parts.append(output[group, :valid_m])
                    reference_parts.append(reference[group, :valid_m])
            return _calc_diff(torch.cat(actual_parts), torch.cat(reference_parts))

        facts = {
            "shape": dict(shape),
            "api": api,
            "mask_values": [int(value) for value in masked_m.cpu().tolist()],
            "api_expected_m": api_expected_m,
            "alignment": alignment,
        }
        return PreparedCase(launch, reset, output, reference, marker, facts, correctness)

    num_groups, expected_m, n, k = (
        int(shape[key]) for key in ("num_groups", "tokens_per_group", "n", "k")
    )
    m, a, b, grouped_layout, _, reference = generators.generate_m_grouped_contiguous(
        num_groups,
        expected_m,
        n,
        k,
        major_a,
        major_b,
        use_ue8m0=use_ue8m0,
        use_bf16=is_bf16,
        use_psum_layout=use_psum,
        quant_config=quant_config,
    )
    output, marker = _marked_tensor((m, n), torch.bfloat16)
    alias_a, alias_b = _alias_inputs(a, b, major_a, major_b)
    recipe = recipe_a = recipe_b = None
    if quant_config is not None:
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

    def reset() -> None:
        marker.zero_()

    def launch() -> Any:
        function = getattr(deep_gemm, api)
        if is_bf16:
            return function(
                alias_a,
                alias_b,
                output,
                grouped_layout,
                use_psum_layout=use_psum,
                ensure_zero_padding=False,
            )
        return function(
            alias_a,
            alias_b,
            output,
            grouped_layout,
            disable_ue8m0_cast=not use_ue8m0,
            use_psum_layout=use_psum,
            ensure_zero_padding=False,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

    def correctness() -> float | None:
        if not use_psum:
            return _calc_diff(output, reference)
        ends = [int(value) for value in grouped_layout.cpu().tolist()]
        actual_parts = []
        reference_parts = []
        previous = 0
        for end in ends:
            actual_parts.append(output[previous:end])
            reference_parts.append(reference[previous:end])
            previous = ((end + alignment - 1) // alignment) * alignment
        return _calc_diff(torch.cat(actual_parts), torch.cat(reference_parts))

    facts = {
        "shape": dict(shape),
        "api": api,
        "actual_m": m,
        "grouped_layout": [int(value) for value in grouped_layout.cpu().tolist()],
        "use_psum_layout": use_psum,
        "alignment": alignment,
        "major_a": major_a.name,
        "major_b": major_b.name,
    }
    return PreparedCase(launch, reset, output, reference, marker, facts, correctness)


def _prepare_k_grouped(deep_gemm: Any, generators: Any, row: Mapping[str, Any]) -> PreparedCase:
    shape = row["shape"]
    num_groups = int(shape["num_groups"])
    m = int(shape["m"])
    n = int(shape["n"])
    alignment = int(shape["alignment"])
    ks_cpu = k_grouped_ks(shape)
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    is_bf16 = row["dtype"] == "bf16"
    if is_bf16:
        major_a = major_b = generators.MajorTypeAB.MNMajor
        use_ue8m0 = False
    else:
        major_a = major_b = generators.MajorTypeAB.KMajor
        use_ue8m0 = bool(generators.get_ue8m0_usage(generators.KernelType.Kernel1D1D))
    _, a, b, generated_c, _, reference, grouped_layout, aligned_ks = generators.generate_k_grouped_contiguous(
        num_groups,
        m,
        n,
        major_a,
        major_b,
        ks_cpu,
        use_ue8m0=use_ue8m0,
        use_bf16=is_bf16,
        gran_k=128,
    )
    output, marker = _marked_tensor((num_groups, m, n), torch.float)
    initial = generated_c.detach().clone()
    output.copy_(initial)
    api = str(row["api"])

    def reset() -> None:
        marker.zero_()
        output.copy_(initial)

    def launch() -> Any:
        function = getattr(deep_gemm, api)
        if is_bf16:
            return function(a, b, output, aligned_ks, grouped_layout, output, use_psum_layout=False)
        return function(
            a,
            b,
            output,
            aligned_ks,
            grouped_layout,
            output,
            recipe=(1, 1, 128),
            use_psum_layout=False,
        )

    facts = {
        "shape": dict(shape),
        "api": api,
        "ks_cpu": list(aligned_ks),
        "grouped_layout": [int(value) for value in grouped_layout.cpu().tolist()],
        "alignment": alignment,
    }
    return PreparedCase(
        launch,
        reset,
        output,
        reference,
        marker,
        facts,
        lambda: _calc_diff(output, reference),
    )


def _prepare_einsum(deep_gemm: Any, row: Mapping[str, Any]) -> PreparedCase:
    shape = row["shape"]
    class_id = str(row["class_id"])
    if class_id == "bf16_einsum_bmk_bnk_mn":
        s, m, n, k = (int(shape[key]) for key in ("s", "m", "n", "k"))
        a = torch.randn((s, m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((s, n, k), device="cuda", dtype=torch.bfloat16)
        output, marker = _marked_tensor((m, n), torch.float, fill=0.0)
        reference = torch.bmm(a.float(), b.float().transpose(1, 2)).sum(0)

        def reset() -> None:
            marker.zero_()
            output.zero_()

        def launch() -> Any:
            return deep_gemm.einsum("bmk,bnk->mn", a, b, output, c=output)

        return PreparedCase(launch, reset, output, reference, marker, {"shape": dict(shape), "api": row["api"]}, lambda: _calc_diff(output, reference))

    bsz, heads, r, dsize = (int(shape[key]) for key in ("b", "h", "r", "d"))
    if class_id.startswith("bf16_einsum"):
        backing = torch.randn((heads, dsize, r + 128), device="cuda", dtype=torch.bfloat16)
        weight = backing[:, :, :r]
        if class_id == "bf16_einsum_bhr_hdr_bhd":
            expr = "bhr,hdr->bhd"
            x = torch.randn((bsz, heads, r), device="cuda", dtype=torch.bfloat16)
            output_shape = (bsz, heads, dsize)
        else:
            expr = "bhd,hdr->bhr"
            x = torch.randn((bsz, heads, dsize), device="cuda", dtype=torch.bfloat16)
            output_shape = (bsz, heads, r)
        output, marker = _marked_tensor(output_shape, torch.bfloat16)
        reference = torch.einsum(expr, x, weight)

        def reset() -> None:
            marker.zero_()

        def launch() -> Any:
            return deep_gemm.einsum(expr, x, weight, output)

        return PreparedCase(launch, reset, output, reference, marker, {"shape": dict(shape), "api": f"einsum:{expr}"}, lambda: _calc_diff(output, reference))

    x = torch.randn((bsz, heads, r), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((heads, dsize, r), device="cuda", dtype=torch.bfloat16)
    from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8  # type: ignore

    x_cast = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=False)
    x_input = (x_cast[0].view(bsz, heads, r), x_cast[1].view(bsz, heads, (r + 127) // 128))
    y_data = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((heads, (dsize + 127) // 128, (r + 127) // 128), device="cuda", dtype=torch.float)
    for index in range(heads):
        y_data[index], y_scale[index] = per_block_cast_to_fp8(y[index], use_ue8m0=False)
    output, marker = _marked_tensor((bsz, heads, dsize), torch.bfloat16)
    reference = torch.einsum("bhr,hdr->bhd", x, y)

    def reset() -> None:
        marker.zero_()

    def launch() -> Any:
        return deep_gemm.fp8_einsum("bhr,hdr->bhd", x_input, (y_data, y_scale), output)

    return PreparedCase(launch, reset, output, reference, marker, {"shape": dict(shape), "api": row["api"]}, lambda: _calc_diff(output, reference))


def _prepare_hc(deep_gemm: Any, row: Mapping[str, Any]) -> PreparedCase:
    shape = row["shape"]
    m, n, k = (int(shape[key]) for key in ("m", "n", "k"))
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float)
    num_splits = None
    output_shape = (m, n)
    sqr_shape = (m,)
    output, marker = _marked_tensor(output_shape, torch.float)
    sqr = torch.empty(sqr_shape, device="cuda", dtype=torch.float)
    reference = a.float() @ b.T
    reference_sqr = a.float().square().sum(-1)

    def reset() -> None:
        marker.zero_()

    def launch() -> Any:
        return deep_gemm.tf32_hc_prenorm_gemm(a, b, output, sqr, num_splits=num_splits)

    def correctness() -> float | None:
        values = (_calc_diff(output, reference), _calc_diff(sqr, reference_sqr))
        return max(value for value in values if value is not None)

    return PreparedCase(
        launch,
        reset,
        output,
        reference,
        marker,
        {"shape": dict(shape), "api": row["api"], "num_splits": num_splits},
        correctness,
    )


def _fp8_public_wrapper_inputs(api: str, a: Any, b: Any) -> tuple[Any, Any]:
    if api == "fp8_gemm_nn":
        return a, (b[0].transpose(0, 1), b[1].transpose(0, 1))
    if api == "fp8_gemm_tn":
        return (a[0].transpose(0, 1), a[1].transpose(0, 1)), (b[0].transpose(0, 1), b[1].transpose(0, 1))
    if api == "fp8_gemm_tt":
        return (a[0].transpose(0, 1), a[1].transpose(0, 1)), b
    raise RuntimeError(f"unsupported FP8 public wrapper: {api}")


def _prepare_fp8_public_wrapper(deep_gemm: Any, generators: Any, row: Mapping[str, Any]) -> PreparedCase:
    shape = row["shape"]
    m, n, k = (int(shape[key]) for key in ("m", "n", "k"))
    major = generators.MajorTypeAB.KMajor
    kernel_type = generators.KernelType.Kernel1D2D
    quant_config = generators.QuantConfig()
    use_ue8m0 = bool(generators.get_ue8m0_usage(kernel_type))
    a, b, _, _, reference = generators.generate_normal(
        m, n, k, major, major, False, torch.bfloat16, kernel_type,
        use_ue8m0=use_ue8m0, quant_config=quant_config,
    )
    output, marker = _marked_tensor((m, n), torch.bfloat16)
    api = str(row["api"])
    pub_a, pub_b = _fp8_public_wrapper_inputs(api, a, b)
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    def reset() -> None:
        marker.zero_()

    def launch() -> Any:
        return getattr(deep_gemm, api)(
            pub_a, pub_b, output, c=None,
            disable_ue8m0_cast=not use_ue8m0,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
        )

    return PreparedCase(
        launch, reset, output, reference, marker,
        {"shape": dict(shape), "api": api, "wrapper": api.removeprefix("fp8_gemm_")},
        lambda: _calc_diff(output, reference),
    )



def _copy_marker_from_output_storage(output: torch.Tensor, marker: torch.Tensor) -> None:
    """Copy the extra-allocated storage tail written by marked extra-storage kernels."""

    marker_bytes = marker.numel() * marker.element_size()
    storage = output.untyped_storage()
    if int(storage.nbytes()) < marker_bytes:
        raise RuntimeError(
            "marked output storage has no marker tail; rebuild the marked DeepGEMM package"
        )
    raw = torch.empty(0, dtype=torch.uint8, device=output.device)
    raw.set_(storage)
    tail = raw.narrow(0, int(storage.nbytes()) - marker_bytes, marker_bytes)
    marker.copy_(tail.view(torch.int64).view_as(marker))


def _prepare_layout_transpose(deep_gemm: Any, row: Mapping[str, Any]) -> PreparedCase:
    """Public get_mn_major_tma_aligned_tensor -> marked transpose_fp32 GPU kernel."""

    shape = row["shape"]
    mn, sf_k = int(shape["mn"]), int(shape["sf_k"])
    if mn % 4 != 0:
        raise RuntimeError("layout mn must be TMA-aligned for FP32 (multiple of 4)")
    sf = torch.randn((mn, sf_k), device="cuda", dtype=torch.float32)
    if not sf.is_contiguous() or sf.stride() != (sf_k, 1):
        raise RuntimeError("layout input must be contiguous row-major to reach transpose_fp32")
    marker = torch.zeros(
        (MAX_MARKER_CTAS, MAX_WARPS_PER_CTA, MARKER_FIELDS),
        device="cuda",
        dtype=torch.int64,
    )
    last = {"out": None}

    def reset() -> None:
        marker.zero_()

    def launch() -> Any:
        out = deep_gemm.get_mn_major_tma_aligned_tensor(sf)
        last["out"] = out
        _copy_marker_from_output_storage(out, marker)
        return out

    def correctness() -> float | None:
        out = last["out"]
        if out is None:
            return None
        return 0.0 if torch.equal(out, sf) else 1.0

    return PreparedCase(
        launch,
        reset,
        sf,
        sf,
        marker,
        {
            "shape": dict(shape),
            "api": "get_mn_major_tma_aligned_tensor",
            "kernel": "transpose_fp32",
            "input_contiguous": True,
        },
        correctness,
    )



def _prepare_paged_mqa_metadata(deep_gemm: Any, row: Mapping[str, Any]) -> PreparedCase:
    """Public get_paged_mqa_logits_metadata -> marked sm90_paged_mqa_logits_metadata kernel."""

    shape = row["shape"]
    batch_size = int(shape["batch_size"])
    next_n = int(shape["next_n"])
    avg_kv = int(shape["avg_kv"])
    if avg_kv < next_n:
        raise RuntimeError("paged metadata avg_kv must cover next_n")
    context_row = torch.arange(avg_kv - next_n + 1, avg_kv + 1, dtype=torch.int32, device="cuda")
    context_lens = context_row.unsqueeze(0).repeat(batch_size, 1).contiguous()
    block_kv = 64
    num_sms = int(deep_gemm.get_num_sms())
    marker = torch.zeros(
        (MAX_MARKER_CTAS, MAX_WARPS_PER_CTA, MARKER_FIELDS),
        device="cuda",
        dtype=torch.int64,
    )
    last: dict[str, Any] = {"out": None}

    def reset() -> None:
        marker.zero_()

    def launch() -> Any:
        out = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, None)
        last["out"] = out
        _copy_marker_from_output_storage(out, marker)
        return out

    def correctness() -> float | None:
        out = last["out"]
        if out is None:
            return None
        if tuple(int(dim) for dim in out.shape) != (num_sms + 1, 2):
            return 1.0
        next_n_atom = 2 if next_n >= 2 else 1
        num_next_n_atoms = (next_n + next_n_atom - 1) // next_n_atom
        expected_last = batch_size * num_next_n_atoms
        last_row = [int(value) for value in out[-1].detach().cpu().tolist()]
        q_idx = out[:-1, 0].detach().to(dtype=torch.int64, device="cpu")
        if last_row != [expected_last, 0]:
            return 1.0
        if int(q_idx.min()) < 0 or int(q_idx.max()) > expected_last:
            return 1.0
        return 0.0

    return PreparedCase(
        launch,
        reset,
        context_lens,
        context_lens,
        marker,
        {
            "shape": dict(shape),
            "api": "get_paged_mqa_logits_metadata",
            "kernel": "sm90_paged_mqa_logits_metadata",
            "block_kv": block_kv,
            "num_sms": num_sms,
        },
        correctness,
    )


def prepare_case(deep_gemm: Any, package: Path, row: Mapping[str, Any]) -> PreparedCase:
    generators = _source_generators(package)
    class_id = str(row["class_id"])
    if class_id.startswith("tf32_hc"):
        return _prepare_hc(deep_gemm, row)
    if "einsum" in class_id:
        return _prepare_einsum(deep_gemm, row)
    if class_id == "fp8_dense_nt_skip_head_mid":
        return _prepare_skip_head_mid(deep_gemm, generators, row)
    if class_id in {"fp8_dense_nn", "fp8_dense_tn", "fp8_dense_tt"}:
        return _prepare_fp8_public_wrapper(deep_gemm, generators, row)
    if class_id == "fp8_layout_transpose_fp32":
        return _prepare_layout_transpose(deep_gemm, row)
    if class_id == "fp8_paged_mqa_metadata":
        return _prepare_paged_mqa_metadata(deep_gemm, row)
    if "mgroup" in class_id:
        return _prepare_m_grouped(deep_gemm, generators, row)
    if "kgroup" in class_id:
        return _prepare_k_grouped(deep_gemm, generators, row)
    return _prepare_dense(deep_gemm, generators, row)


def marker_rows(marker: torch.Tensor) -> list[dict[str, int]]:
    values = marker.detach().cpu()
    # Retain every touched slot, including malformed partial clock pairs.
    coordinates = (values != 0).any(dim=-1).nonzero(as_tuple=False)
    rows: list[dict[str, int]] = []
    for coordinate in coordinates:
        cta_id, warp = (int(value) for value in coordinate.tolist())
        fields = values[cta_id, warp].tolist()
        rows.append(
            {
                "cta_id": cta_id,
                "warp": warp,
                "start_clock64": int(fields[0]),
                "end_clock64": int(fields[1]),
                "start_globaltimer": int(fields[2]),
                "end_globaltimer": int(fields[3]),
                "sm_id": int(fields[4]),
                "cluster_cta_rank": int(fields[5]),
                "cluster_cta_count": int(fields[6]),
            }
        )
    if rows and any(row["cta_id"] == MAX_MARKER_CTAS - 1 for row in rows):
        raise RuntimeError("marker capacity was exhausted")
    if not rows:
        raise RuntimeError("instrumented kernel produced no valid marker rows")
    return rows


def _globaltimer_span(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise RuntimeError("invalid globaltimer marker rows")
    for row in rows:
        if int(row["end_globaltimer"]) <= int(row["start_globaltimer"]):
            raise RuntimeError("marker row lacks positive globaltimer")
    global_start = min(int(row["start_globaltimer"]) for row in rows)
    global_end = max(int(row["end_globaltimer"]) for row in rows)
    if global_end <= global_start:
        raise RuntimeError("invalid globaltimer marker rows")
    return float(global_end - global_start)


def _launch_synchronized(prepared: PreparedCase) -> Any:
    result = prepared.launch()
    torch.cuda.synchronize()
    return result


def _trace_launch(prepared: PreparedCase) -> None:
    cudart = torch.cuda.cudart()
    prepared.reset()
    torch.cuda.synchronize()
    profiler_started = False
    profiler_stopped = False
    try:
        start_status = int(cudart.cudaProfilerStart())
        if start_status != 0:
            raise RuntimeError(f"cudaProfilerStart failed: {start_status}")
        profiler_started = True
        prepared.launch()
        torch.cuda.synchronize()
        stop_status = int(cudart.cudaProfilerStop())
        profiler_stopped = True
        if stop_status != 0:
            raise RuntimeError(f"cudaProfilerStop failed: {stop_status}")
    finally:
        if profiler_started and not profiler_stopped:
            cudart.cudaProfilerStop()
    torch.cuda.synchronize()


def _measure_markers(
    prepared: PreparedCase,
    mode: str,
    warmups: int,
    trials: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    for _ in range(warmups):
        prepared.reset()
        _launch_synchronized(prepared)
    samples: list[float] = []
    summaries: list[dict[str, Any]] = []
    from .actual_binding import cursor, record_trial
    for trial_index in range(trials):
        prepared.reset()
        if os.environ.get("DEEPGEMM_ACTUAL_BINDING_REQUIRED"):
            torch.cuda.synchronize()
        lo = cursor()
        _launch_synchronized(prepared)
        hi = cursor()
        rows = marker_rows(prepared.marker)
        trial = {"raw_trial_index": trial_index, "example_rows": rows,
                 "actual_launch_window": {"lo_exclusive": lo, "hi_inclusive": hi},
                 "marker_slot_count": MAX_MARKER_CTAS * MAX_WARPS_PER_CTA,
                 "touched_slot_count": len(rows),
                 "untouched_slot_count": MAX_MARKER_CTAS * MAX_WARPS_PER_CTA - len(rows)}
        record_trial(trial)
        rejection = None
        try:
            summary = {**summarize_marker_rows(rows), **trial}
            summary["primary_cycles"] = (
                _normal_span_cycles(rows)
                if mode in ("normal", CLUSTER_SM_MODE)
                else float(summary["whole_kernel_span"])
            )
            summary["primary_globaltimer"] = _globaltimer_span(rows)
        except (RuntimeError, SuiteManifestError) as exc:
            summary = {**trial, "timing_rejection": str(exc)}
            rejection = str(exc)
        summary["correctness_symmetric_difference"] = prepared.correctness()
        if rejection is None:
            samples.append(float(summary["primary_cycles"]))
        summaries.append(summary)
    if any(item.get("timing_rejection") for item in summaries):
        return [], summaries
    return samples, summaries


def _measure_events(prepared: PreparedCase, warmups: int, trials: int, snapshot: Mapping[str, Any]) -> list[float]:
    for _ in range(warmups):
        prepared.reset()
        prepared.launch()
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(trials):
        prepared.reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prepared.launch()
        end.record()
        end.synchronize()
        samples.append(_event_cycles(float(start.elapsed_time(end)) / 1000.0, snapshot))
    return samples


def _dump_sass(cache_root: Path) -> dict[str, Any]:
    cubins = sorted(cache_root.glob("**/kernel.cubin")) if cache_root.is_dir() else []
    entries = []
    for cubin in cubins:
        sass = cubin.with_name("kernel.sass")
        completed = subprocess.run(
            ["cuobjdump", "--dump-sass", str(cubin)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode == 0:
            sass.write_text(completed.stdout, encoding="utf-8")
        text_value = completed.stdout
        normalized = []
        for line in text_value.splitlines():
            if any(
                token in line
                for token in (
                    "SR_CLOCK",
                    "SR_GLOBALTIMER",
                    "SR_VIRTUALID",
                    "SR_CLUSTER",
                    "SR_CgaCtaId",
                )
            ):
                continue
            instruction = re.search(r"/\*[0-9a-fA-F]+\*/\s*(.*?)\s*;", line)
            if instruction:
                normalized.append(re.sub(r"\s+", " ", instruction.group(1)).strip())
        core = "\n".join(normalized)
        entries.append(
            {
                "cubin": str(cubin),
                "cubin_sha256": _sha256(cubin),
                "sass": str(sass) if sass.is_file() else None,
                "sass_sha256": _sha256(sass),
                "core_sass_sha256": hashlib.sha256(core.encode()).hexdigest() if core else None,
                "cuobjdump_returncode": completed.returncode,
                "wgmma_static_count": sum(text_value.count(name) for name in ("WGMMA", "HGMMA", "QGMMA")),
                "tma_static_count": text_value.count("TMA"),
                "clock_marker_static_count": text_value.count("SR_CLOCK"),
                "globaltimer_marker_static_count": text_value.count("SR_GLOBALTIMER"),
                "cluster_marker_static_count": text_value.count("SR_CLUSTER")
                + text_value.count("SR_CgaCtaId"),
            }
        )
    return {"entries": entries, "cubin_count": len(entries)}


def _flat_int_list(value: Any) -> list[int]:
    """Flatten an int-or-nested-int-list marker field into plain ints."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            out.extend(int(v) for v in item)
        else:
            out.append(int(item))
    return out


def _jit_cluster_layout(cache_root: Path, *, launched: list[str] | None = None) -> dict[str, Any] | None:
    """Parse cluster_size, cluster_n>1, and launched num_sms from JIT kernel.cu.

    A warm cache keeps one cubin per distinct kernel hash, so several shapes may
    coexist. When `launched` names the cubins this run actually loaded, only
    those sources are parsed; otherwise every cached kernel.cu is considered.
    """

    if not cache_root.is_dir():
        return None
    wanted = {Path(item).resolve() for item in (launched or [])}
    found: list[dict[str, Any]] = []
    for path in cache_root.glob("**/kernel.cu"):
        if wanted and path.resolve() not in wanted:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"\b([12]), (true|false),\s*(\d+)\s*,", text):
            cluster_size = int(match.group(1))
            cluster_n_gt1 = match.group(2) == "true"
            num_sms = int(match.group(3))
            if cluster_n_gt1:
                cluster_dim = {"x": 1, "y": 2, "z": 1}
            elif cluster_size == 2:
                cluster_dim = {"x": 2, "y": 1, "z": 1}
            else:
                cluster_dim = {"x": 1, "y": 1, "z": 1}
            found.append({
                "path": str(path),
                "cluster_size": cluster_size,
                "cluster_n_gt1": cluster_n_gt1,
                "num_sms": num_sms,
                "cluster_dim": cluster_dim,
            })
    if not found:
        return None
    best = max(found, key=lambda row: int(row["cluster_size"]))
    best["all"] = found
    return best


def _jit_multicast(cache_root: Path) -> int | None:
    """Read kNumTMAMulticast from JIT kernel.cu (cluster_size, cluster_n>1, num_sms)."""
    layout = _jit_cluster_layout(cache_root)
    if layout is None:
        return None
    return int(layout["cluster_size"])



def _core_kernel_entries(kernel: Mapping[str, Any]) -> list[dict[str, Any]]:
    """GEMM cores plus independently timed transpose_fp32 and paged MQA metadata kernels."""

    rows: list[dict[str, Any]] = []
    for entry in kernel.get("entries") or []:
        cubin = str(entry.get("cubin") or "")
        if int(entry.get("wgmma_static_count") or 0) > 0 or int(entry.get("tma_static_count") or 0) > 0:
            rows.append(dict(entry))
        elif "transpose_fp32" in cubin or "sm90_paged_mqa_logits_metadata" in cubin:
            rows.append(dict(entry))
    return rows


def _correctness_limit(row: Mapping[str, Any]) -> float:
    if row["dtype"] == "tf32":
        return 1e-6
    if row["dtype"] == "fp8":
        return 1e-3
    return 1e-5


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    config = load_manifest(args.manifest)
    if args.mode not in RUNNER_MODES:
        raise RuntimeError(f"unknown mode: {args.mode}")
    cluster_sm = args.mode == CLUSTER_SM_MODE
    if cluster_sm and args.num_sms not in (1, 2):
        raise RuntimeError("cluster_sm mode requires --num-sms 1 or 2")
    if not cluster_sm and args.num_sms:
        raise RuntimeError("full-GPU normal mode cannot override num_sms")
    matches = [row for row in config["families"] if row["class_id"] == args.class_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown class_id: {args.class_id}")
    family = dict(matches[0])
    pool = config["shape_pools"][family["shape_pool"]]
    if args.shape_json is None:
        if args.shape_index < 0 or args.shape_index >= len(pool):
            raise RuntimeError(f"shape index out of range: {args.shape_index}")
        shape = dict(pool[args.shape_index])
    else:
        try:
            shape = parse_shape_json(args.shape_json)
        except SuiteManifestError as exc:
            raise RuntimeError(f"invalid --shape-json: {exc}") from exc
    family["shape"] = shape
    family["shape_index"] = args.shape_index
    case_id = args.case_id or f"{args.class_id}_s{args.shape_index + 1}"
    comparison_contract = (
        load_comparison_contract(args.comparison_contract, case_id)
        if args.comparison_contract is not None
        else None
    )
    if comparison_contract is not None and (
        comparison_contract.get("comparison_scope")
        != "outermost_in_kernel_clock_marker_interval"
        or comparison_contract.get("clock_marker_policy") != "outermost_pair"
    ):
        raise RuntimeError("DeepGEMM comparison requires the outermost marker interval")
    package = args.package.resolve()
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    snapshot = _nvidia_snapshot()
    gpu_profile = _selected_gpu_profile()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else ""
    if not _gpu_profile_matches(str(snapshot.get("name", "")), gpu_profile):
        raise RuntimeError(f"expected {gpu_profile}, got {snapshot.get('name')}")
    if torch.cuda.device_count() != 1 or not _gpu_profile_matches(device_name, gpu_profile):
        raise RuntimeError(f"the suite requires one visible {gpu_profile} GPU")
    device_sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    requested_sms = int(args.num_sms) if cluster_sm else device_sm_count
    minimum_active_sms = 1 if cluster_sm else _minimum_full_gpu_sms(device_sm_count)
    snapshot.update({
        "gpu_profile": gpu_profile,
        "device_sm_count": device_sm_count,
        "simulation_scope": "cluster_sm" if cluster_sm else "full_gpu",
        "full_device_minimum_active_sms": _minimum_full_gpu_sms(device_sm_count),
        "minimum_required_active_sms": minimum_active_sms,
        "requested_num_sms": requested_sms,
    })
    torch.manual_seed(0)
    random.seed(0)
    deep_gemm = _import_deepgemm(package, cache_root)
    version = getattr(deep_gemm, "__version__", "unknown")
    if version != "2.6.1":
        raise RuntimeError(f"expected DeepGEMM 2.6.1, got {version}")
    deep_gemm.set_num_sms(requested_sms)
    deep_gemm.set_pdl(False)
    marker_meta = _marker_metadata(package)
    if marker_meta is not None:
        from build_deepgemm_full7_package import is_full7_package
        if not is_full7_package(package):
            raise RuntimeError(
                "run refuses a marker package that is not full7 "
                "native-heuristic+layout_transpose_fp32+paged_mqa_logits_metadata"
            )
    prepared = prepare_case(deep_gemm, package, family)
    prepared.reset()
    _launch_synchronized(prepared)
    correctness = prepared.correctness()
    correctness_limit = _correctness_limit(family)
    correctness_pass = correctness is not None and correctness <= correctness_limit
    if args.pilot:
        trial_count = 1
        warmups = 1
        raw_per_recorded_sample = 1
    else:
        trial_count = int(
            config["selection"]["collection_trials" if args.split == "collection" else "verification_trials"]
        )
        raw_per_recorded_sample = 1 if args.trace else _delivery_raw_per_recorded_sample()
        warmups = int(config["selection"]["warmups"])
    raw_trial_count = trial_count * raw_per_recorded_sample
    # Canonical count: executions of the target operation that precede the
    # first recorded sample.  Both the timed and the traced path run the same
    # sequence: one correctness-check launch, then the explicit warmup loop.
    pre_recorded_launches = 1 + warmups
    cache_state = {
        "policy": str(config["selection"]["l2_cache_policy"]),
        "warmup_launches": pre_recorded_launches,
        "staged_launches": {
            "correctness_check": 1,
            "warmup_loop": warmups,
        },
        "warmup_launches_source": "executed_pre_recorded_sequence",
        # Warmup launches run without marker rows, so the SM each preheat
        # launch resided on is not observed; never infer it from the
        # measured trials below.
        "warmup_sm_ids_observed": None,
        "sm_residency_evidence": "not_observed",
    }
    marker_summaries: list[dict[str, Any]] = []
    topology: dict[str, Any]
    if marker_meta is not None:
        if args.trace:
            # Match the timed protocol: the same warmup loop the measured
            # path runs (reset + synchronized launch) executes before the
            # single captured launch.
            for _ in range(warmups):
                prepared.reset()
                _launch_synchronized(prepared)
            _trace_launch(prepared)
            trace_rows = marker_rows(prepared.marker)
            try:
                trace_summary = summarize_marker_rows(trace_rows)
                trace_summary["example_rows"] = trace_rows
                trace_summary["primary_cycles"] = _normal_span_cycles(trace_rows)
                trace_summary["primary_globaltimer"] = _globaltimer_span(trace_rows)
                samples, marker_summaries = [float(trace_summary["primary_cycles"])], [trace_summary]
            except (RuntimeError, SuiteManifestError) as exc:
                trace_summary = {
                    "example_rows": trace_rows,
                    "timing_rejection": str(exc),
                    "cta_count": 0,
                    "sm_ids": [],
                }
                samples, marker_summaries = [], [trace_summary]
        else:
            samples, marker_summaries = _measure_markers(
                prepared,
                args.mode,
                warmups,
                raw_trial_count,
            )
        timing_method = "in_kernel_globaltimer"
        definition = "globaltimer_span * median(clock64_delta/globaltimer_delta)"
        topology = {key: value for key, value in marker_summaries[-1].items() if key != "example_rows"}
        observed_cta_count = int(topology.get("cta_count", 0))
        observed_sm_count = len(topology.get("sm_ids") or [])
        snapshot["minimum_required_active_sms"] = 1
        topology["full_device_minimum_active_sms"] = minimum_active_sms
        topology["minimum_required_active_sms"] = 1
        topology["observed_cta_count"] = observed_cta_count
        topology["observed_sm_count"] = observed_sm_count
        topology["coverage_scope"] = (
            "full_device"
            if observed_sm_count >= minimum_active_sms
            else "launch_limited"
        )
        topology["requested_num_sms"] = requested_sms
        topology["grid"] = {"x": observed_cta_count, "y": 1, "z": 1}
        cluster_n = max([int(v) for v in (topology.get("cluster_cta_counts") or [1])] or [1])
        topology["cluster_dim"] = {"x": cluster_n, "y": 1, "z": 1}
        try:
            uses_cluster_flag = None
            if getattr(args, "uses_cluster", "unset") == "true":
                uses_cluster_flag = True
            elif getattr(args, "uses_cluster", "unset") == "false":
                uses_cluster_flag = False
            validate_mode_topology(
                args.mode,
                topology,
                target_ctas=requested_sms if cluster_sm else None,
                uses_cluster=uses_cluster_flag,
            )
            topology_status = "pass"
        except SuiteManifestError as exc:
            topology_status = "fail"
            topology["error"] = str(exc)
        topology["status"] = topology_status
    elif args.allow_event_timing:
        samples = _measure_events(prepared, warmups, raw_trial_count, snapshot)
        timing_method = "cuda_event_device_clock_conversion"
        definition = "device_elapsed_seconds * max_sm_clock_hz"
        topology = {"status": "not_available_without_marker"}
    else:
        samples = []
        timing_method = "unavailable"
        definition = "none"
        topology = {"status": "not_available_without_marker"}
    raw_cycle_samples = [float(value) for value in samples]
    raw_globaltimer_samples = [
        float(summary["primary_globaltimer"])
        for summary in marker_summaries
        if summary.get("primary_globaltimer") is not None
    ]
    samples = _consecutive_medians(raw_cycle_samples, raw_per_recorded_sample)
    globaltimer_samples = (
        _consecutive_medians(raw_globaltimer_samples, raw_per_recorded_sample)
        if raw_globaltimer_samples
        else []
    )
    median_cycles = statistics.median(samples) if samples else math.nan
    median_globaltimer = (
        statistics.median(globaltimer_samples) if globaltimer_samples else math.nan
    )
    cv = coefficient_of_variation(globaltimer_samples or samples)
    raw_cv = coefficient_of_variation(raw_globaltimer_samples or raw_cycle_samples)
    trial_correctness = [
        float(summary["correctness_symmetric_difference"])
        for summary in marker_summaries
        if summary.get("correctness_symmetric_difference") is not None
    ]
    if trial_correctness:
        correctness = max([float(correctness)] + trial_correctness) if correctness is not None else max(trial_correctness)
        correctness_pass = correctness <= correctness_limit
    cycle_min = CLUSTER_SM_MIN_CYCLES if cluster_sm else MIN_CYCLES
    cycle_max = CLUSTER_SM_MAX_CYCLES if cluster_sm else MAX_CYCLES
    within_range = bool(samples) and cycle_min <= median_cycles <= cycle_max
    enforce_reference_cycle_range = False
    maximum_cv_percent = float(config["selection"]["maximum_cv_percent"])
    if gpu_profile == "h200":
        maximum_cv_percent = max(maximum_cv_percent, 3.0)
    status = "pass"
    if marker_meta is None:
        status = "pilot_event_only" if args.allow_event_timing else "unavailable_no_clock64_marker"
    elif topology.get("status") != "pass":
        status = "topology_failed"
    elif not correctness_pass:
        status = "correctness_failed"
    elif not cluster_sm and not args.trace and cv > maximum_cv_percent:
        status = "unstable"
    if args.trace and status == "pass":
        status = "trace_pass"
    kernel = _dump_sass(cache_root)
    core_entries = _core_kernel_entries(kernel)
    contract_code_identity = (
        validate_marker_code_identity(comparison_contract, core_entries)
        if comparison_contract is not None
        else None
    )
    if marker_meta is not None and core_entries:
        if any(
            entry["clock_marker_static_count"] < 2
            or entry["globaltimer_marker_static_count"] < 2
            or entry["cluster_marker_static_count"] < 1
            for entry in core_entries
        ):
            status = "marker_sass_failed"
    kernel_names = []
    for entry in (kernel.get("entries") if isinstance(kernel, dict) else None) or []:
        name = str(entry.get("module") or entry.get("cubin") or "")
        if name:
            kernel_names.append(name)
    jit_layout = _jit_cluster_layout(cache_root, launched=kernel_names or None)
    multicast = None if jit_layout is None else int(jit_layout["cluster_size"])
    if multicast is None:
        counts = [int(value) for value in topology.get("cluster_cta_counts") or []]
        multicast = max(counts) if counts else None
    # The in-kernel marker is the runtime observation; the JIT source parse is
    # a diagnostic. A warm cache holds cubins for several shapes, so the
    # diagnostic may describe a cubin this launch never ran. Never let it
    # overwrite the measured cluster_dim / multicast.
    measured_counts = _flat_int_list(topology.get("cluster_cta_counts"))
    if measured_counts:
        measured_multicast = max(measured_counts)
        if multicast is None or measured_multicast < multicast:
            multicast = measured_multicast
            if measured_multicast == 1:
                topology["cluster_dim"] = {"x": 1, "y": 1, "z": 1}
    topology["multicast"] = multicast
    if jit_layout is not None:
        topology["jit_cluster"] = {
            "cluster_size": jit_layout["cluster_size"],
            "cluster_n_gt1": jit_layout["cluster_n_gt1"],
            "num_sms": jit_layout["num_sms"],
            "path": jit_layout["path"],
            "scope": "diagnostic_cubin_source_not_launch_observation",
        }
    topology["kernels"] = kernel_names
    topology["grid"] = {
        "x": int(topology.get("observed_cta_count") or topology.get("cta_count") or 0),
        "y": 1,
        "z": 1,
    }

    timing_rejection = next(
        (item.get("timing_rejection") for item in marker_summaries if item.get("timing_rejection")),
        None,
    )
    if timing_rejection:
        median_cycles = None
        median_globaltimer = None
        cv = None
        raw_cv = None
        within_range = False
        samples = []
        globaltimer_samples = []
        raw_cycle_samples = []
        raw_globaltimer_samples = []
        status = "unmeasured"
    from .actual_binding import export_evidence, publish_simulator_accuracy
    marker_capacity = None if marker_meta is None else {"max_warps_per_cta": int(marker_meta["max_warps_per_cta"])}
    actual_evidence = export_evidence(marker_summaries, raw_per_recorded_sample, marker_capacity=marker_capacity) if not args.trace else None
    unit_denominator = actual_evidence["unit_denominator"] if actual_evidence else None
    accuracy = publish_simulator_accuracy(unit_denominator, comparison_contract)
    result = {
        "schema_version": RESULT_SCHEMA,
        "split": args.split,
        "mode": args.mode,
        "case_id": case_id,
        "class_id": args.class_id,
        "shape_index": args.shape_index,
        "family": family,
        "prepared": prepared.facts,
        "hardware": snapshot,
        "deepgemm": {
            "version": version,
            "revision": EXPECTED_REVISION,
            "package": str(package),
            "marker_package": marker_meta,
        },
        "timing": {
            "actual_launch_evidence": actual_evidence,
            "unit_denominator": unit_denominator,
            "timing_rejection": timing_rejection,
            "method": timing_method,
            "scope": (
                "whole_operation_globaltimer_span_converted_by_clock64_ratio"
                if cluster_sm else
                "outermost_in_kernel_globaltimer_span_converted_by_clock64_ratio"
            ),
            "kernels": list(topology.get("kernels") or []),
            "cycle_window": [cycle_min, cycle_max],
            "definition": (
                "median of each consecutive phase-balanced group of four raw "
                "globaltimer spans across all marked warps and CTAs"
                if raw_per_recorded_sample == 4
                else (
                    "median of each consecutive group of three raw globaltimer "
                    "spans across all marked warps and CTAs"
                    if raw_per_recorded_sample == 3
                    else "raw globaltimer span across all marked warps and CTAs"
                )
            ),
            "cycle_compatibility_definition": definition,
            "samples_globaltimer": globaltimer_samples,
            "samples_globaltimer_raw": raw_globaltimer_samples,
            "median_globaltimer": median_globaltimer,
            "samples_cycles": samples,
            "samples_cycles_raw": raw_cycle_samples,
            "median_cycles": median_cycles,
            "cv_percent": cv,
            "raw_cv_percent": raw_cv,
            "maximum_cv_percent": maximum_cv_percent,
            "within_requested_range": within_range,
            "reference_cycle_range_enforced": enforce_reference_cycle_range,
            "sample_aggregation": {
                "recorded_sample_count": len(globaltimer_samples or samples),
                "raw_span_count": len(raw_globaltimer_samples or raw_cycle_samples),
                "raw_spans_per_recorded_sample": raw_per_recorded_sample,
                "operator": "median",
                "grouping": "consecutive_non_overlapping",
                "discarded_raw_spans": 0,
            },
            "warmup_launches": pre_recorded_launches,
            "launches_before_each_raw_sample": [
                pre_recorded_launches + index for index in range(len(raw_cycle_samples))
            ],
            "cache_state": cache_state,
            "marker_trials": marker_summaries,
            "simulator_comparison": {
                "schema_version": "delivery.marker-kernel-timing.v1",
                "status": accuracy["status"],
                "eligible_for_simulator_accuracy": accuracy["eligible_for_simulator_accuracy"],
                "scope": "outermost_in_kernel_clock_marker_interval",
                "hardware_metric": accuracy["hardware_metric"],
                "hardware_value_field": accuracy["hardware_value_field"],
                "simulation_metric": "model.marker_timing.marker_interval_cycles",
                "clock_marker_policy": "outermost_pair",
                "marker_interval_definition": "max(all_warp_end)-min(all_warp_start)",
                "selected_operation_kernel": accuracy["selected_operation_kernel"],
                "selected_unit_bank": accuracy["selected_unit_bank"],
                "samples_cycles": accuracy["samples_cycles"],
                "samples_cycles_raw": accuracy["samples_cycles_raw"],
                "median_cycles": accuracy["median_cycles"],
                "reason": accuracy["reason"],
                "whole_kernel_cycles_must_not_be_mixed_with_marker_interval": True,
                "whole_operation_cycles_must_not_be_used_for_simulator_accuracy": True,
                "contract_digest": accuracy["contract_digest"],
                "code_identity": contract_code_identity,
            },
        },
        "topology": topology,
        "correctness": {
            "symmetric_difference": correctness,
            "trial_symmetric_differences": trial_correctness,
            "maximum_symmetric_difference": correctness_limit,
            "status": "pass" if correctness_pass else "fail",
        },
        "kernel": kernel,
        "multicast": multicast,

        "status": status,
        "notes": [
            "CUDA-event conversion is workload-pilot-only",
            "all raw globaltimer spans are retained when consecutive-triplet medians are enabled",
        ],
        "trace_launch": bool(args.trace),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "mode": args.mode,
                "class_id": args.class_id,
                "shape_index": args.shape_index,
                "median_cycles": median_cycles,
                "cv_percent": cv,
                "multicast": multicast,

            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one DeepGEMM SM90 suite case")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--case-id", help="explicit delivery case id")
    parser.add_argument(
        "--comparison-contract",
        type=Path,
        help="captured contract or corpus manifest for strict simulator comparison",
    )
    parser.add_argument("--shape-index", type=int, default=0)
    parser.add_argument("--shape-json", help="explicit positive-integer shape object")
    parser.add_argument("--mode", choices=RUNNER_MODES, required=True)
    parser.add_argument("--num-sms", type=int, default=0, help="cluster_sm: 1 or 2; normal must stay 0")
    parser.add_argument(
        "--uses-cluster",
        choices=("true", "false", "unset"),
        default="unset",
        help="cluster_sm: true=2CTA cluster/multicast; false=ordinary CTAs",
    )
    parser.add_argument("--pilot", action="store_true", help="one warmup and one recorded sample")
    parser.add_argument("--split", choices=("collection", "verification"), default="collection")
    parser.add_argument("--allow-event-timing", action="store_true")
    parser.add_argument("--trace", action="store_true", help="capture exactly one profiled DeepGEMM launch")
    args = parser.parse_args(argv)
    run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
