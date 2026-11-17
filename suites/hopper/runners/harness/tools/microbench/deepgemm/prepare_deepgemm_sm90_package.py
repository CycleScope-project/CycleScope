"""Build the pinned DeepGEMM source with formal SM90 timing markers.

The transformation adds one marker pointer to each supported SM90 JIT launch.
The pointer is the storage tail immediately after the logical output tensor,
so public DeepGEMM Python APIs remain unchanged. The instrumented package
refuses outputs that do not reserve the complete marker tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .deepgemm_sm90_suite import EXPECTED_REVISION


MARKER_SCHEMA = "deepgemm-sm90-suite-marker-v2"
MAX_MARKER_CTAS = 16_384
MAX_WARPS_PER_CTA = 16
MARKER_FIELDS = 7
CUDA_HEADERS = (
    "sm90_bf16_gemm.cuh",
    "sm90_fp8_gemm_1d2d.cuh",
    "sm90_fp8_gemm_1d1d.cuh",
    "sm90_bmk_bnk_mn.cuh",
    "sm90_tf32_hc_prenorm_gemm.cuh",
)

HOST_HEADERS = (
    "sm90_bf16_gemm.hpp",
    "sm90_fp8_gemm_1d2d.hpp",
    "sm90_fp8_gemm_1d1d.hpp",
    "sm90_bmk_bnk_mn.hpp",
    "sm90_tf32_hc_prenorm_gemm.hpp",
)

RUNTIME_NAMES = {
    "sm90_bf16_gemm.hpp": "SM90BF16GemmRuntime",
    "sm90_fp8_gemm_1d2d.hpp": "SM90FP8Gemm1D2DRuntime",
    "sm90_fp8_gemm_1d1d.hpp": "SM90FP8Gemm1D1DRuntime",
    "sm90_bmk_bnk_mn.hpp": "SM90BmkBnkMnRuntime",
    "sm90_tf32_hc_prenorm_gemm.hpp": "SM90BF16HCPrenormGemmRuntime",
}

KERNEL_NAMES = {
    "sm90_bf16_gemm.cuh": "sm90_bf16_gemm_impl",
    "sm90_fp8_gemm_1d2d.cuh": "sm90_fp8_gemm_1d2d_impl",
    "sm90_fp8_gemm_1d1d.cuh": "sm90_fp8_gemm_1d1d_impl",
    "sm90_bmk_bnk_mn.cuh": "sm90_bmn_bnk_mn_gemm_impl",
    "sm90_tf32_hc_prenorm_gemm.cuh": "sm90_tf32_hc_prenorm_gemm_impl",
}

MARKER_HEADER = r'''#pragma once

#include <cstdint>

namespace deep_gemm::sm90_suite_marker {

static constexpr std::uint32_t kMaxCTAs = 16384;
static constexpr std::uint32_t kMaxWarpsPerCTA = 16;
static constexpr std::uint32_t kFields = 7;

__device__ __forceinline__ void begin(unsigned long long* markers) {
    const auto warp = static_cast<std::uint32_t>(threadIdx.x / 32);
    if (threadIdx.x % 32 != 0 or warp >= kMaxWarpsPerCTA)
        return;
    const auto row = (static_cast<unsigned long long>(blockIdx.x) * kMaxWarpsPerCTA + warp) * kFields;
    unsigned long long clock_value;
    unsigned long long timer_value;
    unsigned int sm_id;
    unsigned int cluster_cta_rank;
    unsigned int cluster_cta_count;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer_value) :: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock_value) :: "memory");
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id) :: "memory");
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cluster_cta_rank) :: "memory");
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_cta_count) :: "memory");
    markers[row + 0] = clock_value;
    markers[row + 2] = timer_value;
    markers[row + 4] = sm_id;
    markers[row + 5] = cluster_cta_rank;
    markers[row + 6] = cluster_cta_count;
}

__device__ __forceinline__ void end(unsigned long long* markers) {
    const auto warp = static_cast<std::uint32_t>(threadIdx.x / 32);
    if (threadIdx.x % 32 != 0 or warp >= kMaxWarpsPerCTA)
        return;
    const auto row = (static_cast<unsigned long long>(blockIdx.x) * kMaxWarpsPerCTA + warp) * kFields;
    unsigned long long clock_value;
    unsigned long long timer_value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer_value) :: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock_value) :: "memory");
    markers[row + 1] = clock_value;
    markers[row + 3] = timer_value;
}

}  // namespace deep_gemm::sm90_suite_marker
'''


HOST_HELPER = r'''
static unsigned long long* get_sm90_suite_marker_ptr(const torch::Tensor& output) {
    constexpr std::size_t kMarkerBytes = 16384ull * 16ull * 7ull * sizeof(unsigned long long);
    DG_HOST_ASSERT(output.storage_offset() == 0);
    const auto logical_bytes = output.nbytes();
    DG_HOST_ASSERT(logical_bytes % sizeof(unsigned long long) == 0);
    DG_HOST_ASSERT(output.storage().nbytes() >= logical_bytes + kMarkerBytes);
    return reinterpret_cast<unsigned long long*>(
        static_cast<std::uint8_t*>(output.data_ptr()) + logical_bytes);
}
'''


class PatchError(RuntimeError):
    pass


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def _output_name(prefix: str) -> str:
    function = prefix.rsplit("static void ", 1)[-1]
    signature = function.split("{", 1)[0]
    if "const torch::Tensor& tensor_d" in signature:
        return "tensor_d"
    if "const torch::Tensor &d" in signature or "const torch::Tensor& d" in signature:
        return "d"
    raise PatchError("could not identify output tensor for a runtime initializer")


def _patch_initializers(text: str, runtime: str, expected: int) -> str:
    anchor = f"const {runtime}::Args& args = {{"
    starts = [match.start() for match in re.finditer(re.escape(anchor), text)]
    if len(starts) != expected:
        raise PatchError(f"{runtime}: expected {expected} initializers, found {len(starts)}")
    offset = 0
    for original_start in starts:
        start = original_start + offset
        end = text.find("\n    };", start)
        if end < 0:
            raise PatchError(f"{runtime}: unterminated initializer")
        output = _output_name(text[:start])
        separator = "" if text[:end].rstrip().endswith(",") else ","
        insertion = f"{separator}\n        .marker_ptr = get_sm90_suite_marker_ptr({output}),"
        text = text[:end] + insertion + text[end:]
        offset += len(insertion)
    return text


def _patch_host_header(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    runtime = RUNTIME_NAMES[path.name]
    expected_initializers = {
        "sm90_bf16_gemm.hpp": 6,
        "sm90_fp8_gemm_1d2d.hpp": 4,
        "sm90_fp8_gemm_1d1d.hpp": 2,
        "sm90_bmk_bnk_mn.hpp": 1,
        "sm90_tf32_hc_prenorm_gemm.hpp": 1,
    }[path.name]
    struct_start = text.find(f"class {runtime}")
    struct_end = text.find("\n    };", struct_start)
    if struct_start < 0 or struct_end < 0:
        raise PatchError(f"{path.name}: runtime Args struct not found")
    text = text[:struct_end] + "\n        unsigned long long* marker_ptr;" + text[struct_end:]
    launch_start = text.find("DG_CUDA_UNIFIED_CHECK(launch_kernel", struct_end)
    launch_end = text.find("));", launch_start)
    if launch_start < 0 or launch_end < 0:
        raise PatchError(f"{path.name}: launch call not found")
    text = text[:launch_end] + ", args.marker_ptr" + text[launch_end:]
    text = _patch_initializers(text, runtime, expected_initializers)
    path.write_text(text, encoding="utf-8")


def _kernel_signature_end(text: str, kernel: str) -> int:
    name = text.find(kernel)
    if name < 0:
        raise PatchError(f"kernel not found: {kernel}")
    brace = text.find(" {", name)
    if brace < 0:
        brace = text.find("){", name)
    if brace < 0:
        brace = text.find(") {", name)
    if brace < 0:
        raise PatchError(f"kernel signature end not found: {kernel}")
    close = text.rfind(")", name, brace + 1)
    if close < 0:
        raise PatchError(f"kernel signature closing parenthesis not found: {kernel}")
    return close


def _patch_cuda_header(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if path.name == "sm90_bf16_gemm.cuh" and "Accuracy-only timing markers" in text:
        start = text.index("    // Accuracy-only timing markers")
        finish = text.index("    // Prefetch TMA descriptors", start)
        text = text[:start] + text[finish:]
        start = text.index("    if constexpr (kWithGroupedTiming) {", text.index("    // Use TMA store"))
        finish = text.index("#else", start)
        text = text[:start] + text[finish:]
    include_anchor = "#include <deep_gemm/common/types.cuh>"
    text = _replace_once(
        text,
        include_anchor,
        include_anchor + "\n#include <deep_gemm/common/sm90_suite_marker.cuh>",
        f"{path.name} marker include",
    )
    kernel = KERNEL_NAMES[path.name]
    close = _kernel_signature_end(text, kernel)
    text = text[:close] + ",\n                               unsigned long long* sm90_suite_markers" + text[close:]
    arch_anchor = "#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)"
    begin = arch_anchor + "\n    sm90_suite_marker::begin(sm90_suite_markers);"
    text = _replace_once(text, arch_anchor, begin, f"{path.name} marker begin")
    else_anchor = "#else\n    if (blockIdx.x == 0 and threadIdx.x == 0)"
    end = "    sm90_suite_marker::end(sm90_suite_markers);\n" + else_anchor
    text = _replace_once(text, else_anchor, end, f"{path.name} marker end")
    path.write_text(text, encoding="utf-8")


def _patch_layout_transpose(host_root: Path, impl_root: Path) -> list[Path]:
    """Mark Hopper-reachable transpose_fp32 only; leave UE8M0 pack unmarked."""

    cuda = impl_root / "smxx_layout.cuh"
    cuda_text = cuda.read_text(encoding="utf-8")
    cuda_text = _replace_once(
        cuda_text,
        "#pragma once\n\n#include <deep_gemm/common/math.cuh>",
        "#pragma once\n\n#include <deep_gemm/common/sm90_suite_marker.cuh>\n#include <deep_gemm/common/math.cuh>",
        "smxx_layout.cuh marker include",
    )
    cuda_text = _replace_once(
        cuda_text,
        "CUTLASS_GLOBAL void transpose_fp32(const float* sf, float* out, const uint32_t mn) {",
        "CUTLASS_GLOBAL void transpose_fp32(const float* sf, float* out, const uint32_t mn,\n"
        "                         unsigned long long* sm90_suite_markers) {\n"
        "    sm90_suite_marker::begin(sm90_suite_markers);",
        "smxx_layout.cuh transpose_fp32 signature",
    )
    cuda_text = _replace_once(
        cuda_text,
        "        out[sf_k_idx * tma_aligned_mn + global_mn_idx] = ptx::ld_shared(smem_buffer + mn_idx * PADDED_SF_K + sf_k_idx);\n    }\n}",
        "        out[sf_k_idx * tma_aligned_mn + global_mn_idx] = ptx::ld_shared(smem_buffer + mn_idx * PADDED_SF_K + sf_k_idx);\n"
        "    }\n"
        "    sm90_suite_marker::end(sm90_suite_markers);\n"
        "}",
        "smxx_layout.cuh transpose_fp32 marker end",
    )
    cuda.write_text(cuda_text, encoding="utf-8")

    host = host_root / "smxx_layout.hpp"
    host_text = host.read_text(encoding="utf-8")
    host_text = _replace_once(
        host_text,
        "        void *sf, *out;\n\n        LaunchArgs launch_args;",
        "        void *sf, *out;\n        unsigned long long* marker_ptr;\n\n        LaunchArgs launch_args;",
        "TransposeFP32Runtime marker_ptr",
    )
    host_text = _replace_once(
        host_text,
        "DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.sf, args.out, static_cast<uint32_t>(args.mn)));",
        "DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.sf, args.out, static_cast<uint32_t>(args.mn), args.marker_ptr));",
        "TransposeFP32Runtime launch marker",
    )
    marker_bytes = "16384" + "ull * 16" + "ull * 7" + "ull * sizeof(unsigned long long)"
    host_text = _replace_once(
        host_text,
        "    const auto out = torch::empty_strided({num_sf_batches, mn, sf_k},\n"
        "                                          {tma_aligned_mn * sf_k, 1, tma_aligned_mn},\n"
        "                                          batched_sf.options());\n",
        "    constexpr std::size_t kMarkerBytes = " + marker_bytes + ";\n"
        "    DG_HOST_ASSERT(kMarkerBytes % sizeof(float) == 0);\n"
        "    const auto padded_elems = static_cast<int64_t>(num_sf_batches) * tma_aligned_mn * sf_k;\n"
        "    const auto marker_elems = static_cast<int64_t>(kMarkerBytes / sizeof(float));\n"
        "    auto storage = torch::empty({padded_elems + marker_elems}, batched_sf.options());\n"
        "    storage.narrow(0, padded_elems, marker_elems).zero_();\n"
        "    const auto out = storage.as_strided(\n"
        "        {num_sf_batches, mn, sf_k}, {tma_aligned_mn * sf_k, 1, tma_aligned_mn});\n",
        "transpose_fp32 marked allocation",
    )
    host_text = _replace_once(
        host_text,
        "        const TransposeFP32Runtime::Args& args = {\n"
        "            .mn = mn,\n"
        "            .sf_k = sf_k,\n"
        "            .block_mn = block_mn,\n"
        "            .sf = batched_sf.data_ptr(),\n"
        "            .out = out.data_ptr(),\n"
        "            .launch_args = LaunchArgs({ceil_div(mn, block_mn), num_sf_batches}, num_threads, smem_size)\n"
        "        };\n",
        "        const TransposeFP32Runtime::Args& args = {\n"
        "            .mn = mn,\n"
        "            .sf_k = sf_k,\n"
        "            .block_mn = block_mn,\n"
        "            .sf = batched_sf.data_ptr(),\n"
        "            .out = out.data_ptr(),\n"
        "            .marker_ptr = reinterpret_cast<unsigned long long*>(\n"
        "                static_cast<char*>(storage.data_ptr()) + padded_elems * sizeof(float)),\n"
        "            .launch_args = LaunchArgs({ceil_div(mn, block_mn), num_sf_batches}, num_threads, smem_size)\n"
        "        };\n",
        "transpose_fp32 marked initializer",
    )
    host.write_text(host_text, encoding="utf-8")
    return [cuda, host]



def _patch_paged_mqa_metadata(host_root: Path, source: Path) -> list[Path]:
    """Mark the Hopper public get_paged_mqa_logits_metadata CUDA kernel."""

    cuda = source / "deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh"
    cuda_text = cuda.read_text(encoding="utf-8")
    cuda_text = _replace_once(
        cuda_text,
        "#pragma once\n\n#include <deep_gemm/common/math.cuh>",
        "#pragma once\n\n#include <deep_gemm/common/sm90_suite_marker.cuh>\n#include <deep_gemm/common/math.cuh>",
        "sm90_paged_mqa_logits.cuh marker include",
    )
    cuda_text = _replace_once(
        cuda_text,
        "void sm90_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t next_n, const bool is_context_lens_2d,\n"
        "                                    const uint32_t* context_lens, const uint32_t* indices, uint32_t* schedule_metadata) {\n"
        "    DG_STATIC_ASSERT(kAlignedBatchSize % 32 == 0, \"Invalid aligned batch size\");",
        "void sm90_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t next_n, const bool is_context_lens_2d,\n"
        "                                    const uint32_t* context_lens, const uint32_t* indices, uint32_t* schedule_metadata,\n"
        "                                    unsigned long long* sm90_suite_markers) {\n"
        "    DG_STATIC_ASSERT(kAlignedBatchSize % 32 == 0, \"Invalid aligned batch size\");\n"
        "    sm90_suite_marker::begin(sm90_suite_markers);",
        "sm90_paged_mqa_logits_metadata signature",
    )
    cuda_text = _replace_once(
        cuda_text,
        "        if (lane_idx == 0) {\n"
        "            schedule_metadata[kNumSMs * 2] = batch_size * num_next_n_atoms;\n"
        "            schedule_metadata[kNumSMs * 2 + 1] = 0;\n"
        "        }\n"
        "    }\n"
        "}\n",
        "        if (lane_idx == 0) {\n"
        "            schedule_metadata[kNumSMs * 2] = batch_size * num_next_n_atoms;\n"
        "            schedule_metadata[kNumSMs * 2 + 1] = 0;\n"
        "        }\n"
        "    }\n"
        "    sm90_suite_marker::end(sm90_suite_markers);\n"
        "}\n",
        "sm90_paged_mqa_logits_metadata marker end",
    )
    cuda.write_text(cuda_text, encoding="utf-8")

    host = host_root / "sm90_fp8_mqa_logits.hpp"
    host_text = host.read_text(encoding="utf-8")
    host_text = _replace_once(
        host_text,
        "        int* indices;\n"
        "        int* schedule_metadata;\n\n"
        "        LaunchArgs launch_args;",
        "        int* indices;\n"
        "        int* schedule_metadata;\n"
        "        unsigned long long* marker_ptr;\n\n"
        "        LaunchArgs launch_args;",
        "SM90PagedMQALogitsMetadataRuntime marker_ptr",
    )
    host_text = _replace_once(
        host_text,
        "        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,\n"
        "            args.batch_size,\n"
        "            args.next_n,\n"
        "            args.is_context_lens_2d,\n"
        "            args.context_lens,\n"
        "            args.indices,\n"
        "            args.schedule_metadata\n"
        "        ));",
        "        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,\n"
        "            args.batch_size,\n"
        "            args.next_n,\n"
        "            args.is_context_lens_2d,\n"
        "            args.context_lens,\n"
        "            args.indices,\n"
        "            args.schedule_metadata,\n"
        "            args.marker_ptr\n"
        "        ));",
        "SM90PagedMQALogitsMetadataRuntime launch marker",
    )
    host_text = _replace_once(
        host_text,
        "        .indices = const_cast<int*>(indices_ptr),\n"
        "        .schedule_metadata = schedule_metadata.data_ptr<int>(),\n"
        "        .launch_args = LaunchArgs(1, num_threads, smem_size)",
        "        .indices = const_cast<int*>(indices_ptr),\n"
        "        .schedule_metadata = schedule_metadata.data_ptr<int>(),\n"
        "        .marker_ptr = get_sm90_suite_marker_ptr(schedule_metadata),\n"
        "        .launch_args = LaunchArgs(1, num_threads, smem_size)",
        "SM90PagedMQALogitsMetadataRuntime initializer",
    )
    host.write_text(host_text, encoding="utf-8")

    api = source / "csrc/apis/attention.hpp"
    api_text = api.read_text(encoding="utf-8")
    marker_bytes = "16384ull * 16ull * 7ull * sizeof(unsigned long long)"
    api_text = _replace_once(
        api_text,
        "    auto schedule_metadata = torch::empty({num_sms + 1, 2}, context_lens.options());\n",
        "    constexpr std::size_t kMarkerBytes = " + marker_bytes + ";\n"
        "    DG_HOST_ASSERT(kMarkerBytes % sizeof(int) == 0);\n"
        "    const auto payload_elems = static_cast<int64_t>(num_sms + 1) * 2;\n"
        "    const auto marker_elems = static_cast<int64_t>(kMarkerBytes / sizeof(int));\n"
        "    auto storage = torch::empty({payload_elems + marker_elems}, context_lens.options());\n"
        "    storage.narrow(0, payload_elems, marker_elems).zero_();\n"
        "    auto schedule_metadata = storage.as_strided({num_sms + 1, 2}, {2, 1});\n",
        "paged MQA metadata marked allocation",
    )
    api.write_text(api_text, encoding="utf-8")
    return [cuda, host, api]


def patch_source(source: Path) -> dict[str, str]:
    source = source.resolve()
    version_file = source / "deep_gemm/__init__.py"
    if not version_file.is_file() or "__version__ = '2.6.1'" not in version_file.read_text(encoding="utf-8"):
        raise PatchError("source is not DeepGEMM 2.6.1")
    impl_root = source / "deep_gemm/include/deep_gemm/impls"
    host_root = source / "csrc/jit_kernels/impls"
    marker_header = source / "deep_gemm/include/deep_gemm/common/sm90_suite_marker.cuh"
    if marker_header.exists():
        raise PatchError(f"source is already instrumented: {marker_header}")
    marker_header.write_text(MARKER_HEADER, encoding="utf-8")
    runtime_utils = host_root / "runtime_utils.hpp"
    text = runtime_utils.read_text(encoding="utf-8")
    text = _replace_once(text, "namespace deep_gemm {", "namespace deep_gemm {\n" + HOST_HELPER, "host marker helper")
    runtime_utils.write_text(text, encoding="utf-8")
    for name in HOST_HEADERS:
        _patch_host_header(host_root / name)
    for name in CUDA_HEADERS:
        _patch_cuda_header(impl_root / name)
    layout_paths = _patch_layout_transpose(host_root, impl_root)
    metadata_paths = _patch_paged_mqa_metadata(host_root, source)
    paths: Iterable[Path] = [marker_header, runtime_utils]
    paths = list(paths) + [host_root / name for name in HOST_HEADERS] + [impl_root / name for name in CUDA_HEADERS] + layout_paths + metadata_paths
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _copy_source(source: Path, work_source: Path) -> None:
    ignored = shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", ".git")
    shutil.copytree(source, work_source, ignore=ignored)
    # A formal capture may read its source from a frozen tree whose regular files
    # and directories intentionally have no write bits.  copytree preserves those
    # modes, but the instrumented package is patched only in this disposable work
    # copy.  Make the copied tree owner-writable without touching the input tree.
    for directory, names, files in os.walk(work_source):
        directory_path = Path(directory)
        directory_path.chmod(stat.S_IMODE(directory_path.stat().st_mode) | stat.S_IWUSR)
        for name in names:
            path = directory_path / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
        for name in files:
            path = directory_path / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


def _matching_deepgemm_wheels(wheel_root: Path) -> list[Path]:
    """Accept both the plain 2.6.1 wheel and PEP 440 local builds."""
    return sorted(
        set(wheel_root.glob("deep_gemm-2.6.1-*.whl"))
        | set(wheel_root.glob("deep_gemm-2.6.1+*.whl"))
    )


ATTENTION_BANK_HOOK = r'''    {
        const auto base = static_cast<std::uint8_t*>(output.data_ptr());
        using ObserveAttentionBank = void (*)(int, const void*, const void*,
                                              unsigned long long, unsigned long long);
        static const auto observe_bank = reinterpret_cast<ObserveAttentionBank>(
            dlsym(RTLD_DEFAULT, "fi_deepgemm_attention_bank"));
        DG_HOST_ASSERT(getenv("DEEPGEMM_ACTUAL_BINDING_REQUIRED") == nullptr or observe_bank != nullptr);
        if (observe_bank != nullptr)
            observe_bank(bank, base + payload_bytes + bank * kMarkerBytes, base,
                         static_cast<unsigned long long>(payload_bytes),
                         static_cast<unsigned long long>(kMarkerBytes));
    }
'''


def _patch_attention_bank_hook(source: Path) -> dict[str, str]:
    """Report the physical marker-bank pointer of the launch about to happen.

    Only the full7 attention helper carries banks; a source without it is a
    core-only package and is left byte-identical (recorded by the empty
    return). The hook reuses the established host contact pattern: dlsym on
    RTLD_DEFAULT plus a fail-closed assert while actual binding is required.
    Device code, marker intervals and timing semantics are untouched.
    """
    path = source / "csrc/jit_kernels/impls/runtime_utils.hpp"
    text = path.read_text(encoding="utf-8")
    helper = "static unsigned long long* get_sm90_suite_attention_marker_ptr"
    count = text.count(helper)
    if count == 0:
        return {}
    if count != 1:
        raise PatchError("attention marker helper appears more than once")
    anchor = ("    const auto payload_bytes = storage_bytes - kMarkerBanks * kMarkerBytes;\n"
              "    return reinterpret_cast<unsigned long long*>(\n"
              "        static_cast<std::uint8_t*>(output.data_ptr()) + payload_bytes + bank * kMarkerBytes);")
    if text.count(anchor) != 1:
        raise PatchError("attention marker helper body did not match the full7 patch")
    replacement = ("    const auto payload_bytes = storage_bytes - kMarkerBanks * kMarkerBytes;\n"
                   + ATTENTION_BANK_HOOK
                   + "    return reinterpret_cast<unsigned long long*>(\n"
                     "        static_cast<std::uint8_t*>(output.data_ptr()) + payload_bytes + bank * kMarkerBytes);")
    text = text.replace(anchor, replacement, 1)
    include_anchor = "#include <torch/python.h>"
    if text.count(include_anchor) != 1:
        raise PatchError("runtime_utils include anchor not unique")
    includes = []
    if "#include <dlfcn.h>" not in text:
        includes.append("#include <dlfcn.h>")
    if "#include <cstdlib>" not in text:
        includes.append("#include <cstdlib>")
    if includes:
        text = text.replace(include_anchor, include_anchor + "\n" + "\n".join(includes), 1)
    path.write_text(text)
    return {"csrc/jit_kernels/impls/runtime_utils.hpp": hashlib.sha256(path.read_bytes()).hexdigest()}

ATTENTION_RESET_HOOK = r'''    {
        using ObserveAttentionReset = void (*)(const void*, unsigned long long);
        static const auto observe_reset = reinterpret_cast<ObserveAttentionReset>(
            dlsym(RTLD_DEFAULT, "fi_deepgemm_attention_reset"));
        DG_HOST_ASSERT(getenv("DEEPGEMM_ACTUAL_BINDING_REQUIRED") == nullptr or observe_reset != nullptr);
        if (observe_reset != nullptr)
            observe_reset(static_cast<const void*>(
                             static_cast<const std::uint8_t*>(storage.data_ptr())
                                 + payload_elements * element_bytes),
                         static_cast<unsigned long long>(marker_elements * element_bytes));
    }
'''


def _patch_attention_reset_hook(source: Path) -> dict[str, str]:
    """Report the actual marker tails before their ATen zeroing launches.

    Both the full7 output and paged metadata allocate a marker tail.
    The callback identifies its exact byte range; no launch-name whitelist
    or device-code change is needed to attribute the runtime reset.
    """
    path = source / "csrc/apis/attention.hpp"
    text = path.read_text(encoding="utf-8")
    helper = "static torch::Tensor allocate_sm90_suite_attention_output"
    count = text.count(helper)
    metadata_anchor = "    storage.narrow(0, payload_elems, marker_elems).zero_();"
    if count == 0 and metadata_anchor not in text:
        return {}
    if count > 1:
        raise PatchError("attention allocation helper appears more than once")
    if count == 1:
        anchor = "    storage.slice(0, payload_elements, payload_elements + marker_elements).zero_();"
        text = _replace_once(text, anchor, ATTENTION_RESET_HOOK + anchor,
                             "attention marker zeroing")
    if metadata_anchor in text:
        hook = LAYOUT_RESET_HOOK.replace("padded_elems", "payload_elems").replace("sizeof(float)", "sizeof(int)")
        text = _replace_once(text, metadata_anchor, hook + metadata_anchor,
                             "paged metadata marker zeroing")
    include_anchor = "#include \"../utils/compatibility.hpp\""
    if text.count(include_anchor) != 1:
        raise PatchError("attention api include anchor not unique")
    includes = []
    if "#include <dlfcn.h>" not in text:
        includes.append("#include <dlfcn.h>")
    if "#include <cstdlib>" not in text:
        includes.append("#include <cstdlib>")
    if includes:
        text = text.replace(include_anchor, include_anchor + "\n" + "\n".join(includes), 1)
    path.write_text(text)
    return {"csrc/apis/attention.hpp": hashlib.sha256(path.read_bytes()).hexdigest()}

LAYOUT_RESET_HOOK = r'''    {
        using ObserveAttentionReset = void (*)(const void*, unsigned long long);
        static const auto observe_reset = reinterpret_cast<ObserveAttentionReset>(
            dlsym(RTLD_DEFAULT, "fi_deepgemm_attention_reset"));
        DG_HOST_ASSERT(getenv("DEEPGEMM_ACTUAL_BINDING_REQUIRED") == nullptr or observe_reset != nullptr);
        if (observe_reset != nullptr)
            observe_reset(static_cast<const void*>(
                             static_cast<const std::uint8_t*>(storage.data_ptr())
                                 + padded_elems * static_cast<int64_t>(sizeof(float))),
                         static_cast<unsigned long long>(marker_elems * static_cast<int64_t>(sizeof(float))));
    }
'''


def _patch_layout_reset_hook(source: Path) -> dict[str, str]:
    """Report the marker region the layout-transpose helper is about to zero.

    get_mn_major_tma_aligned_tensor zeroes its output storage's marker tail
    (narrow(...).zero_()) inside the timed call, so the ATen fill lands in
    the trial window exactly like the attention allocation helper's. The
    same dlsym contact records the byte region; the contiguous-copy
    fallback below it stays unattested on purpose (a real op difference,
    surfaced loudly rather than silently admitted).
    """
    path = source / "csrc/jit_kernels/impls/smxx_layout.hpp"
    text = path.read_text(encoding="utf-8")
    helper = "get_mn_major_tma_aligned_tensor"
    count = text.count(helper)
    if count == 0:
        return {}
    if count != 1:
        raise PatchError("layout helper appears more than once")
    anchor = "    storage.narrow(0, padded_elems, marker_elems).zero_();"
    if text.count(anchor) != 1:
        raise PatchError("layout marker zeroing anchor did not match the full7 patch")
    text = text.replace(anchor, LAYOUT_RESET_HOOK + anchor, 1)
    include_anchor = "#include <torch/python.h>"
    if text.count(include_anchor) != 1:
        raise PatchError("layout include anchor not unique")
    includes = []
    if "#include <dlfcn.h>" not in text:
        includes.append("#include <dlfcn.h>")
    if "#include <cstdlib>" not in text:
        includes.append("#include <cstdlib>")
    if includes:
        text = text.replace(include_anchor, include_anchor + "\n" + "\n".join(includes), 1)
    path.write_text(text)
    return {"csrc/jit_kernels/impls/smxx_layout.hpp": hashlib.sha256(path.read_bytes()).hexdigest()}

def patch_host_only(source: Path) -> dict[str, str]:
    """Install only the canonical load/launch observation contacts.

    The caller supplies an already-instrumented original build source; none
    of patch_source's device/header instrumentation is executed on this path.
    """
    canonical = Path(__file__).resolve().parents[3] / "suites/deepgemm/upstream/csrc/jit/handle.hpp"
    reference = canonical.read_text()
    path = source / "csrc/jit/handle.hpp"
    text = path.read_text()
    if "fi_deepgemm_bind" in text:
        raise PatchError("source already has actual binding contacts")
    start = text.index("#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name)")
    end = text.index("DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);")
    ref_start = reference.index("#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name)")
    ref_end = reference.index("DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);")
    text = text[:start] + reference[ref_start:ref_end] + text[end:]
    text = _replace_once(text, "#include <filesystem>", "#include <filesystem>\n#include <cstdlib>", "host cstdlib")
    text = _replace_once(text, "    DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryEnumerateKernels);", "    DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryEnumerateKernels);\n    DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryGetModule);", "host library module")
    bind_start = reference.index("    using ObserveBind =")
    bind_end = reference.index("\n    if (library_opt", bind_start)
    anchor = "    DG_CUDA_DRIVER_CHECK(lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()));\n#endif"
    text = _replace_once(text, anchor, anchor + "\n" + reference[bind_start:bind_end], "host actual loaded code")
    unload_start = reference.index("    using ObserveUnload =")
    unload_end = reference.index("\n}", unload_start)
    anchor = "    DG_HOST_ASSERT(error == CUDA_SUCCESS or error == CUDA_ERROR_DEINITIALIZED);"
    text = _replace_once(text, anchor, anchor + "\n" + reference[unload_start:unload_end], "host unload generation")
    runtime_anchor = "    KernelHandle kernel{};"
    text = _replace_once(text, runtime_anchor, runtime_anchor + '\n    DG_HOST_ASSERT(getenv("DEEPGEMM_ACTUAL_BINDING_REQUIRED") == nullptr and\n                   "runtime cudaKernel_t cannot be attested as a driver CUfunction");', "runtime opaque handle boundary")
    path.write_text(text)
    hashes = {"csrc/jit/handle.hpp": hashlib.sha256(path.read_bytes()).hexdigest()}
    hashes.update(_patch_attention_bank_hook(source))
    hashes.update(_patch_attention_reset_hook(source))
    hashes.update(_patch_layout_reset_hook(source))
    return hashes


def build(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.resolve()
    output = args.output.resolve()
    work_root = args.work_root.resolve()
    if output.exists() or work_root.exists():
        raise RuntimeError("output and work-root must not already exist")
    work_source = work_root / "source"
    work_root.mkdir(parents=True)
    _copy_source(source, work_source)
    device_before = {str(path.relative_to(work_source)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (work_source / "deep_gemm/include").rglob("*") if path.is_file()}
    hashes = patch_host_only(work_source) if args.host_only else patch_source(work_source)
    metadata: dict[str, object] = {
        "schema_version": MARKER_SCHEMA,
        "deepgemm_version": "2.6.1",
        "deepgemm_revision": EXPECTED_REVISION,
        "max_ctas": MAX_MARKER_CTAS,
        "max_warps_per_cta": MAX_WARPS_PER_CTA,
        "marker_fields": ["start_clock64", "end_clock64", "start_globaltimer", "end_globaltimer", "sm_id", "cluster_cta_rank", "cluster_cta_count"],
        "source_hashes": hashes,
        "layout_transpose_fp32": True,
        "paged_mqa_logits_metadata": True,
        "kgroup_native_heuristic": True,
        "kgroup_cluster_required": False,
        "kgroup_cluster_even_tiles": False,
    }
    if args.host_only:
        original_metadata = json.loads(args.marker_metadata.read_text())
        metadata = {**original_metadata, "actual_binding_host_hashes": hashes,
                    "host_only_observer": True, "device_include_sha256": device_before,
                    "host_paged_metadata_reset_hook": (
                        "    storage.narrow(0, payload_elems, marker_elems).zero_();"
                        in (work_source / "csrc/apis/attention.hpp").read_text())}
        device_after = {str(path.relative_to(work_source)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (work_source / "deep_gemm/include").rglob("*") if path.is_file()}
        if device_after != device_before:
            raise PatchError("host-only patch changed device include bytes")
    if args.patch_only:
        metadata["patched_source"] = str(work_source)
        print(json.dumps(metadata, sort_keys=True))
        return metadata
    wheel_root = work_root / "wheel"
    wheel_root.mkdir()
    build_log = work_root / "build.log"
    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(wheel_root), str(work_source)]
    build_environment = os.environ.copy()
    if args.host_only:
        # Same CUDA-wheel header/link closure as scripts/build_cluster_sm.sh
        # and stage6_near_dg_qualification._build_env; no device flags added.
        import site
        include_dirs = [str(path) for root in site.getsitepackages()
                        for path in sorted(Path(root).glob("nvidia/*/include")) if path.is_dir()]
        links = work_root / "nvlinks"
        links.mkdir()
        for root in site.getsitepackages():
            for lib in sorted(Path(root).glob("nvidia/*/lib/*.so.*")):
                target = links / (lib.name.split(".so")[0] + ".so")
                if not target.exists():
                    target.symlink_to(lib)
        build_environment["CPATH"] = ":".join(include_dirs + ([build_environment["CPATH"]] if build_environment.get("CPATH") else []))
        build_environment["LIBRARY_PATH"] = ":".join(filter(None, [str(links), build_environment.get("LIBRARY_PATH")]))
        build_environment["DG_FORCE_BUILD"] = "1"
        metadata["host_build_environment"] = {key: build_environment.get(key) for key in ("CPATH", "LIBRARY_PATH", "DG_FORCE_BUILD", "CUDA_HOME")}
    with build_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, env=build_environment)
    if completed.returncode != 0:
        raise RuntimeError(f"DeepGEMM wheel build failed; see {build_log}")
    wheels = _matching_deepgemm_wheels(wheel_root)
    if len(wheels) != 1:
        raise RuntimeError(f"expected one DeepGEMM wheel, found {len(wheels)}")
    install_log = work_root / "install.log"
    install = [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(output), str(wheels[0])]
    with install_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(install, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"DeepGEMM wheel install failed; see {install_log}")
    tests_target = output / "source_tests"
    shutil.copytree(work_source / "tests", tests_target)
    metadata["wheel"] = str(wheels[0])
    metadata["build_log"] = str(build_log)
    metadata["install_log"] = str(install_log)
    metadata_path = output / "deep_gemm/sm90_suite_marker.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "package": str(output), "metadata": str(metadata_path)}, sort_keys=True))
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the formal DeepGEMM SM90 marker package")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--patch-only", action="store_true")
    parser.add_argument("--host-only", action="store_true", help="patch only actual-load/launch host contacts in original full7 source")
    parser.add_argument("--marker-metadata", type=Path, help="original package sm90_suite_marker.json for host-only builds")
    args = parser.parse_args(argv)
    if args.host_only and args.marker_metadata is None:
        parser.error("--host-only requires original --marker-metadata")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
