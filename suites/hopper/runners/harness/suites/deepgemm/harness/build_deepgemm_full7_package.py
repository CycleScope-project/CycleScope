#!/usr/bin/env python3
"""Build the pinned DeepGEMM package with markers for all seven SM90 modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.microbench.deepgemm import prepare_deepgemm_sm90_package as base
from tools.microbench.deepgemm_sm90_suite import EXPECTED_REVISION


MARKER_BYTES = base.MAX_MARKER_CTAS * base.MAX_WARPS_PER_CTA * base.MARKER_FIELDS * 8
ATTENTION_BANKS = 2
PACKAGE_ENV = "DEEPGEMM_DELIVERY_PACKAGE"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise base.PatchError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_runtime_class(text: str, runtime: str, bank: int) -> str:
    class_start = text.find(f"class {runtime}")
    if class_start < 0:
        raise base.PatchError(f"runtime class not found: {runtime}")
    args_end = text.find("\n    };", class_start)
    if args_end < 0:
        raise base.PatchError(f"Args struct end not found: {runtime}")
    text = text[:args_end] + "\n        unsigned long long* marker_ptr;" + text[args_end:]

    launch_start = text.find("DG_CUDA_UNIFIED_CHECK(launch_kernel", args_end)
    launch_end = text.find("));", launch_start)
    if launch_start < 0 or launch_end < 0:
        raise base.PatchError(f"launch call not found: {runtime}")
    text = text[:launch_end] + ", args.marker_ptr" + text[launch_end:]

    initializer = f"const {runtime}::Args"
    init_start = text.find(initializer, launch_end)
    init_end = text.find("\n    };", init_start)
    if init_start < 0 or init_end < 0:
        raise base.PatchError(f"initializer not found: {runtime}")
    insertion = (
        ",\n        .marker_ptr = "
        f"get_sm90_suite_attention_marker_ptr(logits, {bank})"
    )
    text = text[:init_end] + insertion + text[init_end:]
    return text


def patch_cuda_kernel(path: Path, kernel: str) -> None:
    text = path.read_text(encoding="utf-8")
    include_anchor = "#pragma once"
    text = replace_once(
        text,
        include_anchor,
        include_anchor + "\n\n#include <deep_gemm/common/sm90_suite_marker.cuh>",
        f"{path.name} marker include",
    )
    close = base._kernel_signature_end(text, kernel)
    text = text[:close] + ",\n                         unsigned long long* sm90_suite_markers" + text[close:]
    brace = text.find("{", close)
    if brace < 0:
        raise base.PatchError(f"kernel body not found: {kernel}")
    text = text[: brace + 1] + "\n    sm90_suite_marker::begin(sm90_suite_markers);" + text[brace + 1 :]
    function_end = text.rfind("\n}\n\n}")
    if function_end < 0:
        function_end = text.rfind("\n}\n\n} // namespace")
    if function_end < 0:
        raise base.PatchError(f"kernel end not found: {kernel}")
    text = text[:function_end] + "\n    sm90_suite_marker::end(sm90_suite_markers);" + text[function_end:]
    path.write_text(text, encoding="utf-8")


def patch_attention(source: Path) -> dict[str, str]:
    host_root = source / "csrc/jit_kernels/impls"
    cuda_root = source / "deep_gemm/include/deep_gemm/impls"

    runtime_utils = host_root / "runtime_utils.hpp"
    text = runtime_utils.read_text(encoding="utf-8")
    helper = f'''
static unsigned long long* get_sm90_suite_attention_marker_ptr(const torch::Tensor& output, const int bank) {{
    constexpr std::size_t kMarkerBytes = {MARKER_BYTES}ull;
    constexpr int kMarkerBanks = {ATTENTION_BANKS};
    DG_HOST_ASSERT(output.storage_offset() == 0);
    DG_HOST_ASSERT(bank >= 0 and bank < kMarkerBanks);
    const auto storage_bytes = output.storage().nbytes();
    DG_HOST_ASSERT(storage_bytes >= kMarkerBanks * kMarkerBytes);
    const auto payload_bytes = storage_bytes - kMarkerBanks * kMarkerBytes;
    return reinterpret_cast<unsigned long long*>(
        static_cast<std::uint8_t*>(output.data_ptr()) + payload_bytes + bank * kMarkerBytes);
}}
'''
    namespace_anchor = "namespace deep_gemm {"
    text = replace_once(text, namespace_anchor, namespace_anchor + helper, "attention marker helper")
    runtime_utils.write_text(text, encoding="utf-8")

    mqa_host = host_root / "sm90_fp8_mqa_logits.hpp"
    text = mqa_host.read_text(encoding="utf-8")
    text = patch_runtime_class(text, "SM90FP8MQALogitsRuntime", 0)
    text = patch_runtime_class(text, "SM90FP8PagedMQALogitsRuntime", 0)
    mqa_host.write_text(text, encoding="utf-8")

    clean_host = host_root / "smxx_clean_logits.hpp"
    text = clean_host.read_text(encoding="utf-8")
    text = patch_runtime_class(text, "SMXXCleanLogitsRuntime", 1)
    clean_host.write_text(text, encoding="utf-8")

    attention_api = source / "csrc/apis/attention.hpp"
    text = attention_api.read_text(encoding="utf-8")
    allocation_helper = f'''
static torch::Tensor allocate_sm90_suite_attention_output(const int64_t rows,
                                                           const int64_t columns,
                                                           const at::ScalarType dtype,
                                                           const torch::TensorOptions& options) {{
    constexpr int64_t kMarkerBytes = {MARKER_BYTES}ll;
    constexpr int64_t kMarkerBanks = {ATTENTION_BANKS};
    const int64_t element_bytes = static_cast<int64_t>(c10::elementSize(dtype));
    DG_HOST_ASSERT((kMarkerBanks * kMarkerBytes) % element_bytes == 0);
    const int64_t payload_elements = rows * columns;
    const int64_t marker_elements = kMarkerBanks * kMarkerBytes / element_bytes;
    auto storage = torch::empty({{payload_elements + marker_elements}}, options.dtype(dtype));
    storage.slice(0, payload_elements, payload_elements + marker_elements).zero_();
    return storage.slice(0, 0, payload_elements).view({{rows, columns}});
}}
'''
    namespace_anchor = "namespace deep_gemm::attention {"
    text = replace_once(text, namespace_anchor, namespace_anchor + allocation_helper, "attention allocation helper")
    old = "logits = torch::empty({aligned_seq_len, stride_logits}, q_fp.options().dtype(logits_dtype));"
    if text.count(old) != 2:
        raise base.PatchError(f"prefill output allocations: expected two, found {text.count(old)}")
    text = text.replace(
        old,
        "logits = allocate_sm90_suite_attention_output(aligned_seq_len, stride_logits, logits_dtype, q_fp.options());",
    )
    old = "auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q_fp.options().dtype(logits_dtype));"
    text = replace_once(
        text,
        old,
        "auto logits = allocate_sm90_suite_attention_output(batch_size * next_n, aligned_max_context_len, logits_dtype, q_fp.options());",
        "paged output allocation",
    )
    attention_api.write_text(text, encoding="utf-8")

    cuda_files = {
        "sm90_fp8_mqa_logits.cuh": "sm90_fp8_mqa_logits",
        "sm90_fp8_paged_mqa_logits.cuh": "sm90_fp8_paged_mqa_logits",
        "smxx_clean_logits.cuh": "smxx_clean_logits",
    }
    for name, kernel in cuda_files.items():
        patch_cuda_kernel(cuda_root / name, kernel)

    paths = [
        runtime_utils,
        mqa_host,
        clean_host,
        attention_api,
        *(cuda_root / name for name in cuda_files),
    ]
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def is_full7_package(path: Path) -> bool:
    """Public build.sh / run_100.sh cache contract (metadata only, no hash gate).

    A reusable marker package must have:
    - native vendor SM90 k-group scheduling (not the leftover cluster=2 force)
    - marked layout_transpose_fp32 capability
    - marked paged_mqa_logits_metadata capability
    Missing any flag, or kgroup_cluster_required=true, is not reusable.
    """

    meta = path / "deep_gemm/sm90_suite_marker.json"
    if not meta.is_file():
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    if (payload.get("single_sm_extension") or {}).get("enabled") is True:
        return False
    if payload.get("layout_transpose_fp32") is not True:
        return False
    if payload.get("kgroup_native_heuristic") is not True:
        return False
    if payload.get("kgroup_cluster_required") is True:
        return False
    if payload.get("paged_mqa_logits_metadata") is not True:
        return False
    if payload.get("cluster_sm_allow_num_sms_1") is True:
        return False
    return True


def reuse_env_package(output: Path) -> bool:
    """Reuse DEEPGEMM_DELIVERY_PACKAGE instead of rebuilding the wheel."""

    configured = os.environ.get(PACKAGE_ENV)
    if not configured:
        return False
    source = Path(configured)
    if not is_full7_package(source):
        raise RuntimeError(f"{PACKAGE_ENV} is not a full7 native-heuristic+layout+metadata marker package: {source}")
    source = source.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        if output.resolve() == source:
            print(json.dumps({"status": "reused", "package": str(output)}, sort_keys=True))
            return True
        raise RuntimeError(f"output already exists: {output}")
    output.symlink_to(source, target_is_directory=True)
    print(json.dumps(
        {"status": "reused", "package": str(output), "source": str(source)},
        sort_keys=True,
    ))
    return True


def build(source: Path, output: Path, work_root: Path) -> None:
    if output.exists() or work_root.exists():
        raise RuntimeError("output and work-root must not already exist")
    work_source = work_root / "source"
    work_root.mkdir(parents=True)
    base._copy_source(source.resolve(), work_source)
    hashes = base.patch_source(work_source)
    hashes.update(patch_attention(work_source))

    wheel_root = work_root / "wheel"
    wheel_root.mkdir()
    build_log = work_root / "build.log"
    command = [
        sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
        "-w", str(wheel_root), str(work_source),
    ]
    with build_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"DeepGEMM wheel build failed; see {build_log}")
    wheels = base._matching_deepgemm_wheels(wheel_root)
    if len(wheels) != 1:
        raise RuntimeError(f"expected one DeepGEMM wheel, found {len(wheels)}")

    install_log = work_root / "install.log"
    command = [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(output), str(wheels[0])]
    with install_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"DeepGEMM wheel install failed; see {install_log}")
    shutil.copytree(work_source / "tests", output / "source_tests")

    metadata = {
        "schema_version": base.MARKER_SCHEMA,
        "deepgemm_version": "2.6.1",
        "deepgemm_revision": EXPECTED_REVISION,
        "max_ctas": base.MAX_MARKER_CTAS,
        "max_warps_per_cta": base.MAX_WARPS_PER_CTA,
        "marker_fields": [
            "start_clock64", "end_clock64", "start_globaltimer", "end_globaltimer",
            "sm_id", "cluster_cta_rank", "cluster_cta_count",
        ],
        "attention_marker_banks": ["mqa", "clean_logits"],
        "covered_sm90_modules": [
            "sm90_bf16_gemm", "sm90_fp8_gemm_1d2d", "sm90_fp8_gemm_1d1d",
            "sm90_bmk_bnk_mn", "sm90_tf32_hc_prenorm_gemm",
            "sm90_fp8_mqa_logits", "smxx_clean_logits", "smxx_layout",
        ],
        "layout_transpose_fp32": True,
        "paged_mqa_logits_metadata": True,
        "kgroup_native_heuristic": True,
        "kgroup_cluster_even_tiles": False,
        "kgroup_cluster_required": False,
        "cluster_sm_allow_num_sms_1": False,
        "source_hashes": hashes,
        "wheel": str(wheels[0]),
        "build_log": str(build_log),
        "install_log": str(install_log),
    }
    metadata_path = output / "deep_gemm/sm90_suite_marker.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "package": str(output), "metadata": str(metadata_path)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    if reuse_env_package(args.output):
        return
    build(args.source, args.output, args.work_root)


if __name__ == "__main__":
    main()
