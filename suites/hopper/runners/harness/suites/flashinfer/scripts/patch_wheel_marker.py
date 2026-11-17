#!/usr/bin/env python3
"""Apply the delivery clock-marker instrumentation to a FlashInfer tree (T0005 修订 3-②/修订 5).

Mechanism (measurement-only, upstream-identical without this script):

  Every instrumented kernel gains an `#ifdef FI_DELIVERY_MARKER_TIMING` entry/exit
  CS2R SR_CLOCKLO+SR_GLOBALTIMERLO pair around its outermost body (one 7-field row per
  (cta, warp); harness/override/fi_delivery_marker.h) plus explicit pair closures at early
  `return`s, so every warp records exactly one (entry, exit) pair per launch.  All edits are
  inside the macro: a no-marker compile reproduces the upstream kernel bit-for-bit (FlashMLA
  lane guarantee).

Trees (either layout works, resolved by --tree):

  wheel site dir    <site>/flashinfer/data/include/... and <site>/flashinfer/data/csrc/...
  git source root   <root>/include/... and <root>/csrc/...   (suites/flashinfer/upstream)

The marker header (fi_delivery_marker.h) is copied next to the patched files so the inserted
`#include "fi_delivery_marker.h"` resolves.  --apply is idempotent and writes a JSON log
(upstream/patched sha256 per file) for the delivery audit trail; --check only verifies.

Usage:
  python3 patch_wheel_marker.py --tree <site-or-src-root> --apply  [--log PATH]
  python3 patch_wheel_marker.py --tree <site-or-src-root> --check
  python3 patch_wheel_marker.py --list            # print the per-class instrumentation table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MARKER_MACRO = "FI_DELIVERY_MARKER_TIMING"
MARKER_HEADER_NAME = "fi_delivery_marker.h"
SM_CAP_MACRO = "FI_DELIVERY_SM_CAP"
SM_CAP_ENV = "FI_DELIVERY_SM_CAP"

# -------------------------------------------------------------------------------------------
# Code snippets.  Region symbols map 1:1 to harness/marker_channel_config.py; primary =
# formal denominator for the op class that launches the kernel, secondary* = evidence only.
# -------------------------------------------------------------------------------------------


def entry_snippet(region: str) -> str:
    return (
        f"#ifdef {MARKER_MACRO}\n"
        f"    // Delivery measurement-only outermost clock-marker entry (T0005 rev5; before\n"
        f"    // any prologue; one 7-field row per (cta, warp)).\n"
        f"    const unsigned long long fi_marker_row = fi_marker_row_index();\n"
        f"    if (fi_marker_warp_leader()) {{\n"
        f"        fi_marker_entry({region}, fi_marker_row);\n"
        f"    }}\n"
        f"#endif\n"
    )


def exit_snippet(region: str) -> str:
    return (
        f"#ifdef {MARKER_MACRO}\n"
        f"    // Outermost clock-marker exit: on the kernel's single return path.\n"
        f"    if (fi_marker_warp_leader()) {{\n"
        f"        fi_marker_exit({region}, fi_marker_row);\n"
        f"    }}\n"
        f"#endif\n"
    )


def return_closure_snippet(region: str, indent: str = "      ") -> str:
    """Closes the marker pair before an early `return;` (FlashMLA 空调度 CTA 收口 pattern)."""
    return (
        f"#ifdef {MARKER_MACRO}\n"
        f"{indent}// Early exit: close the marker pair so every warp records one\n"
        f"{indent}// outermost (entry, exit) pair.\n"
        f"{indent}if (fi_marker_warp_leader()) {{\n"
        f"{indent}    fi_marker_exit({region}, fi_marker_row);\n"
        f"{indent}}}\n"
        f"{indent}return;\n"
        f"#else\n"
        f"{indent}return;\n"
        f"#endif\n"
    )


MARKER_INCLUDE = (
    f"#ifdef {MARKER_MACRO}\n"
    f'#include "{MARKER_HEADER_NAME}"\n'
    f"#endif\n"
)


def registrar_snippet(module_key: str) -> str:
    """TU registrar enrollment (FlashMLA RegistrarToken pattern, see the marker header)."""
    return (
        f"#ifdef {MARKER_MACRO}\n"
        f'FI_DELIVERY_MARKER_REGISTER_MODULE("{module_key}")\n'
        f"#endif\n"
    )

# -------------------------------------------------------------------------------------------
# Per-class instrumentation table.  File kinds: "include" (JIT include tree, angle-include
# layout flashinfer/...) and "csrc" (JIT csrc tree).  kernels: name -> region symbol.
# closures: (anchor regex replacing the early return, region) — applied inside the named
# kernel's body only, counted with re.subn and asserted non-zero.
# -------------------------------------------------------------------------------------------

PATCHES = [
    {
        "op_classes": ["append_paged_kv"],
        "kind": "include",
        "path": "flashinfer/page.cuh",
        "module": "page",
        "kernels": {"AppendPagedKVCacheKernel": "fi_marker_primary_ptr"},
        "closures": [],
    },
    {
        "op_classes": ["packbits", "segment_packbits"],
        "kind": "include",
        "path": "flashinfer/quantization.cuh",
        "module": "quantization",
        "kernels": {
            "PackBitsKernel": "fi_marker_primary_ptr",
            "SegmentPackBitsKernel": "fi_marker_secondary_ptr",
        },
        "closures": [],
    },
    {
        "op_classes": ["rmsnorm", "fused_add_rmsnorm"],
        "kind": "include",
        "path": "flashinfer/norm.cuh",
        "module": "norm",
        "kernels": {
            "RMSNormKernel": "fi_marker_primary_ptr",
            "FusedAddRMSNormKernel": "fi_marker_secondary_ptr",
        },
        "closures": [],
    },
    {
        "op_classes": ["rope"],
        "kind": "include",
        "path": "flashinfer/pos_enc.cuh",
        "module": "pos_enc",
        "kernels": {"BatchQKApplyRotaryKernel": "fi_marker_primary_ptr"},
        "closures": [],
    },
    {
        "op_classes": ["merge_state", "merge_states"],
        "kind": "include",
        "path": "flashinfer/attention/cascade.cuh",
        "module": "cascade",
        "kernels": {
            "MergeStateKernel": "fi_marker_primary_ptr",
            "MergeStatesKernel": "fi_marker_secondary_ptr",
            "PersistentVariableLengthMergeStatesKernel": "fi_marker_tertiary_ptr",
        },
        "closures": [
            # MergeStatesKernel fast paths (num_index_sets <= 2): whole-CTA uniform returns.
            {
                "kernel": "MergeStatesKernel",
                "region": "fi_marker_secondary_ptr",
                "count": 2,
                "pattern": re.compile(r"(?P<b>\}\n)(?P<i>[ ]+)return;\n"),
                "replacement": (
                    "{b}"
                    "#ifdef FI_DELIVERY_MARKER_TIMING\n"
                    "{i}if (fi_marker_warp_leader()) {{\n"
                    "{i}    fi_marker_exit({region}, fi_marker_row);\n"
                    "{i}}}\n"
                    "{i}return;\n"
                    "#else\n"
                    "{i}return;\n"
                    "#endif\n"
                ),
            },
        ],
    },
    {
        "op_classes": ["sample", "top_k_sample", "top_p_sample"],
        "kind": "include",
        "path": "flashinfer/sampling.cuh",
        "module": "sampling",
        "kernels": {
            "SamplingFromProbKernel": "fi_marker_primary_ptr",
            "TopKSamplingFromProbKernel": "fi_marker_secondary_ptr",
            "TopPSamplingFromProbKernel": "fi_marker_tertiary_ptr",
            "TopKTopPSamplingFromProbKernel": "fi_marker_quaternary_ptr",
        },
        "closures": [
            # last_valid_id == -1 fallback: every thread returns after tx==0 stores.
            {
                "kernel": name,
                "region": region,
                "count": 1,
                "pattern": re.compile(
                    r"(?P<head>if \(temp_storage\.last_valid_id == -1\) \{\n)"
                    r"(?P<i>[ ]+)if \(tx == 0\) \{\n"
                    r"(?P<i2>[ ]+)output\[bx\] = 0;\n"
                    r"(?P<i3>[ ]+)valid\[bx\] = false;\n"
                    r"(?P<i4>[ ]+)\}\n"
                    r"(?P<i5>[ ]+)return;\n"
                    r"(?P<i6>[ ]+)\}"
                ),
                "replacement": (
                    "{head}"
                    "{i}if (tx == 0) {{\n"
                    "{i2}output[bx] = 0;\n"
                    "{i3}valid[bx] = false;\n"
                    "{i4}}}\n"
                    "#ifdef FI_DELIVERY_MARKER_TIMING\n"
                    "{i5}if (fi_marker_warp_leader()) {{\n"
                    "{i5}    fi_marker_exit({region}, fi_marker_row);\n"
                    "{i5}}}\n"
                    "{i5}return;\n"
                    "#else\n"
                    "{i5}return;\n"
                    "#endif\n"
                    "{i6}}}"
                ),
            }
            for name, region in [
                ("SamplingFromProbKernel", "fi_marker_primary_ptr"),
                ("TopKSamplingFromProbKernel", "fi_marker_secondary_ptr"),
                ("TopPSamplingFromProbKernel", "fi_marker_tertiary_ptr"),
                ("TopKTopPSamplingFromProbKernel", "fi_marker_quaternary_ptr"),
            ]
        ],
    },
    {
        "op_classes": ["single_decode", "batch_decode_paged"],
        "kind": "include",
        "path": "flashinfer/attention/decode.cuh",
        "module": "decode",
        "kernels": {
            "SingleDecodeWithKVCacheKernel": "fi_marker_primary_ptr",
            "BatchDecodeWithPagedKVCacheKernel": "fi_marker_secondary_ptr",
            "BatchDecodeWithPagedKVCacheKernelMLA": "fi_marker_tertiary_ptr",
        },
        "closures": [
            {
                "kernel": "BatchDecodeWithPagedKVCacheKernelMLA",
                "region": "fi_marker_tertiary_ptr",
                "count": 1,
                "pattern": re.compile(
                    r"(?P<c>if \(block_valid_mask && !block_valid_mask\[batch_idx\]\) )return;"
                ),
                "replacement": (
                    "#ifdef FI_DELIVERY_MARKER_TIMING\n"
                    "  {c}{{\n"
                    "    // Early exit (over-launched CUDagraph block): close the marker pair\n"
                    "    // so every warp records one outermost (entry, exit) pair.\n"
                    "    if (fi_marker_warp_leader()) {{\n"
                    "        fi_marker_exit({region}, fi_marker_row);\n"
                    "    }}\n"
                    "    return;\n"
                    "  }}\n"
                    "#else\n"
                    "  {c}return;\n"
                    "#endif"
                ),
            }
        ],
    },
    {
        "op_classes": ["single_prefill", "batch_prefill_paged", "batch_prefill_ragged"],
        "kind": "include",
        "path": "flashinfer/attention/prefill.cuh",
        "module": "prefill",
        "kernels": {
            "SinglePrefillWithKVCacheKernel": "fi_marker_primary_ptr",
            "BatchPrefillWithRaggedKVCacheKernel": "fi_marker_secondary_ptr",
            "BatchPrefillWithPagedKVCacheKernel": "fi_marker_tertiary_ptr",
        },
        "closures": [
            {
                "kernel": "BatchPrefillWithRaggedKVCacheKernel",
                "region": "fi_marker_secondary_ptr",
                "count": 1,
                "pattern": re.compile(
                    r"(?P<head>if \(block_valid_mask && !block_valid_mask\[bx\]\) \{\n)"
                    r"(?P<i>[ ]+)return;\n"
                    r"(?P<i2>[ ]+)\}"
                ),
                "replacement": (
                    "{head}"
                    "#ifdef FI_DELIVERY_MARKER_TIMING\n"
                    "{i}if (fi_marker_warp_leader()) {{\n"
                    "{i}    fi_marker_exit(fi_marker_secondary_ptr, fi_marker_row);\n"
                    "{i}}}\n"
                    "{i}return;\n"
                    "#else\n"
                    "{i}return;\n"
                    "#endif\n"
                    "{i2}}}"
                ),
            }
        ],
    },
    {
        "op_classes": ["batch_prefill_paged", "batch_prefill_ragged", "single_prefill",
                       "batch_decode_paged"],
        "kind": "include",
        "path": "flashinfer/attention/hopper/prefill_sm90.cuh",
        "module": "prefill_sm90",
        "kernels": {"PrefillWithKVCacheKernel": "fi_marker_primary_ptr"},
        "closures": [],
    },
    {
        "op_classes": ["single_prefill", "batch_prefill_paged"],
        "kind": "include",
        "path": "flashinfer/attention/hopper/quantization/prefill_sm90.cuh",
        "module": "prefill_sm90_quant",
        "kernels": {"FP8PrefillWithKVCacheKernel": "fi_marker_primary_ptr"},
        "closures": [],
    },
    {
        "op_classes": ["batch_mla_paged"],
        "kind": "include",
        "path": "flashinfer/attention/mla.cuh",
        "module": "mla",
        "kernels": {"BatchMLAPagedAttentionKernel": "fi_marker_primary_ptr"},
        "closures": [],
    },
    {
        "op_classes": ["batch_mla_paged"],
        "kind": "include",
        "path": "flashinfer/attention/mla_hopper.cuh",
        "module": "mla_hopper",
        "kernels": {"BatchMLAPageAttentionHopperKernel": "fi_marker_primary_ptr"},
        "closures": [],
    },
    {
        "op_classes": ["mamba_state_update"],
        "kind": "include",
        "path": "flashinfer/mamba/kernel_selective_state_update_stp.cuh",
        "module": "mamba_stp",
        "kernels": {
            "selective_state_update_kernel_simple": "fi_marker_primary_ptr",
            "selective_state_update_kernel_producer_consumer_horizontal":
                "fi_marker_secondary_ptr",
            "selective_state_update_kernel_producer_consumer_vertical":
                "fi_marker_tertiary_ptr",
        },
        "closures": [],
    },
    {
        "op_classes": ["xqa_batch_decode"],
        "kind": "csrc",
        "path": "xqa/mha.cu",
        "module": "xqa",
        "kernels": {"kernel_mha": "fi_marker_primary_ptr"},
        "closures": [],
        # T0011.g freshness fix: the LAUNCHED kernel for the bf16 xqa path is the
        # CUBIN_EXPORT __global__ kernel_mha dispatcher in mha.cu (kernel_mha_impl
        # inlines into it).  The existing xqa/mha_sm90.cu entry instruments the
        # other TU's kernel; the audit's SASS lookup resolves mha.cu's section
        # (0 marker reads -> gate stale, sh4 2026-10-01 evidence).  Both TUs of
        # module "xqa" now carry the primary region; the registrar macro is
        # per-TU by design (see fi_delivery_marker.h REGISTER_MODULE note).
    },
    {
        "op_classes": ["xqa_batch_decode"],
        "kind": "csrc",
        "path": "xqa/mha_sm90.cu",
        "module": "xqa",
        "kernels": {"kernel_mha": "fi_marker_primary_ptr"},
        "closures": [
            # All four early returns are CTA-uniform (inputSeqLen / subSeq / multiBlockMode /
            # lastCta guards): close the marker pair before returning (FlashMLA 收口 pattern).
            {
                "kernel": "kernel_mha",
                "region": "fi_marker_primary_ptr",
                "count": 4,
                "pattern": re.compile(r"(?P<b>\{\n)(?P<i>[ ]+)return;\n"),
                "replacement": (
                    "{b}"
                    "#ifdef FI_DELIVERY_MARKER_TIMING\n"
                    "{i}if (fi_marker_warp_leader()) {{\n"
                    "{i}    fi_marker_exit({region}, fi_marker_row);\n"
                    "{i}}}\n"
                    "{i}return;\n"
                    "#else\n"
                    "{i}return;\n"
                    "#endif\n"
                ),
            },
        ],
    },
    {
        "op_classes": ["fp8_blockscale_gemm"],
        "kind": "csrc",
        "path": "nv_internal/tensorrt_llm/deep_gemm/fp8_gemm_impl.cuh",
        "module": "fp8_blockscale_gemm_sm90",
        "kernels": {
            "fp8_gemm_kernel": "fi_marker_primary_ptr",
            "fp8_gemm_kernel_swapAB": "fi_marker_secondary_ptr",
        },
        "closures": [],
    },
    {
        # T0005 rev9 (far-holdout G0 on-site item): the wheel-vendored CUTLASS
        # (flashinfer/data/cutlass, 4.5.0) device_kernel wrapper — the generic
        # 3.x launch template every cutlass GEMM instantiates.  Same override
        # technique as suites/cutlass/harness/override/cutlass/
        # device_kernel.h (296 suite), expressed as a wheel patch so the JIT
        # modules pick it up from the instrumented tree.  One region: per-case
        # primary selection (segment_gemm: single GemmUniversal launch, H100
        # probe 2026-09-30; cutlass_fused_moe: static primary
        # ("cutlass_device_kernel", 0, "device_kernel") per T0011 修订 5 ② — see
        # harness/marker_channel_config.py OP_CLASSES note for the criterion and
        # the two-launch row-overwrite semantics; validated by the t11a-moe-probe
        # H100 pod).
        "op_classes": ["segment_gemm", "cutlass_fused_moe"],
        "kind": "cutlass_include",
        "path": "cutlass/device_kernel.h",
        "module": "cutlass_device_kernel",
        "kernels": {"device_kernel": "fi_marker_primary_ptr"},
        "closures": [],
    },
]


# -------------------------------------------------------------------------------------------
# Kernel body location (brace matching after the __global__ signature).
# -------------------------------------------------------------------------------------------

# Mangled (Itanium) names of the header's TU-local __device__ pointer symbols, as emitted by
# nvcc into the cubin symbol table.  The bridge tries each region in order and skips regions
# the cubin does not carry.
DEEP_GEMM_BRIDGE = r"""
// T0011.d deep_gemm fp8_blockscale alternative marker channel (measurement-only).  The
// fp8_gemm_kernel cubin is nvcc -cubin compiled and loaded via cuLibraryLoadData
// (deep_gemm runtime.cuh, kJitUseNvcc=true), so its TU-local marker pointer symbols are not
// reachable through the module registrar and the registry's cudaMemcpyToSymbol setter never
// touches them.  Bridge: the module .so registers normally (registrar appended at EOF), the
// harness sets THESE TU-local pointers, and fi_delivery_bridge_library_markers copies the
// current pointer VALUES into the loaded library's globals, so the cubin's instrumented
// fp8_gemm_kernel writes the same rows the channel reads.  Upstream-identical without
// FI_DELIVERY_MARKER_TIMING (whole block is macro-guarded).
#ifdef FI_DELIVERY_MARKER_TIMING
#include <cuda.h>
#include <cuda_runtime.h>
#include "fi_delivery_marker.h"

namespace fi_delivery_marker {

// Host-side read of a TU-local __device__ pointer's current VALUE (cudaMemcpyFromSymbol via
// the compiler symbol stub; the same mechanism fi_marker_set_tu_buffers writes with).
inline unsigned long long* fi_marker_primary_ptr_value() {
  unsigned long long* v = nullptr;
  cudaMemcpyFromSymbol(&v, fi_marker_primary_ptr, sizeof(v));
  return v;
}
inline unsigned long long* fi_marker_secondary_ptr_value() {
  unsigned long long* v = nullptr;
  cudaMemcpyFromSymbol(&v, fi_marker_secondary_ptr, sizeof(v));
  return v;
}
inline unsigned long long* fi_marker_tertiary_ptr_value() {
  unsigned long long* v = nullptr;
  cudaMemcpyFromSymbol(&v, fi_marker_tertiary_ptr, sizeof(v));
  return v;
}
inline unsigned long long* fi_marker_quaternary_ptr_value() {
  unsigned long long* v = nullptr;
  cudaMemcpyFromSymbol(&v, fi_marker_quaternary_ptr, sizeof(v));
  return v;
}

inline void fi_delivery_bridge_library_markers(CUlibrary lib) {
  if (lib == nullptr) {
    return;
  }
  const char* names[4] = {
      "_ZN18fi_delivery_marker20fi_marker_primary_ptrE",
      "_ZN18fi_delivery_marker21fi_marker_secondary_ptrE",
      "_ZN18fi_delivery_marker21fi_marker_tertiary_ptrE",
      "_ZN18fi_delivery_marker22fi_marker_quaternary_ptrE",
  };
  unsigned long long* const ours[4] = {
      fi_marker_primary_ptr_value(), fi_marker_secondary_ptr_value(),
      fi_marker_tertiary_ptr_value(), fi_marker_quaternary_ptr_value(),
  };
  for (int i = 0; i < 4; ++i) {
    if (ours[i] == nullptr) {
      continue;
    }
    CUdeviceptr dst = 0;
    size_t size = 0;
    if (cuLibraryGetGlobal(&dst, &size, lib, names[i]) != CUDA_SUCCESS || dst == 0) {
      continue;
    }
    cudaMemcpy(reinterpret_cast<void*>(dst), &ours[i], sizeof(ours[i]),
               cudaMemcpyHostToDevice);
  }
}

}  // namespace fi_delivery_marker
#endif  // FI_DELIVERY_MARKER_TIMING
"""


# -------------------------------------------------------------------------------------------
# T0011.d host-side patches (measurement-only, all under opt-in guards; device kernels and
# their SASS are untouched — the FlashMLA build-patch contract, rev1 item 2).
#
# kinds:
#   "sm_cap"       — clamp cudaDevAttrMultiProcessorCount reads on the FA3/MLA-hopper host
#                    plan/launch sites to $FI_DELIVERY_SM_CAP, compiled in only with
#                    -DFI_DELIVERY_SM_CAP (enable via harness/marker_timing.enable_sm_cap_mode).
#                    Both sides of each op read the SAME clamped value: FA3 prefill plan
#                    (PrefillSM90Plan work bins) and launch grid (Scheduler::get_grid_dim);
#                    MLA FA3 plan (MLAPlan num_blks_y) and its cooperative launch grid.
#                    FA2 decode/prefill paths keep natural grids — those cases scale by GQA
#                    knobs (h_kv=1) per rev1, not by SM cap.
#   "marker_raw"   — deep_gemm fp8_blockscale alternative channel (T0011.d item 5): the
#                    fp8_gemm_kernel cubin is built by the vendored deep_gemm JIT compiler
#                    (nvcc -cubin, always; kJitUseNvcc) and loaded via cuLibraryLoadData, so
#                    the FI_DELIVERY_MARKER_TIMING macro never reaches it and its TU symbols
#                    are unreachable through the registrar.  Two raw edits bridge it:
#                    compiler.cuh propagates the marker macro into the cubin compile flags
#                    (env-gated), runtime.cuh registers the module .so and copies the TU-local
#                    buffer pointers into the loaded library globals on every get_kernel call.
# -------------------------------------------------------------------------------------------

SM_CAP_HELPER = (
    "#ifdef FI_DELIVERY_SM_CAP\n"
    "#include <cstdlib>\n"
    "namespace fi_delivery_sm_cap {\n"
    "// Delivery measurement-only host clamp (T0011.d rev1 item 2): min(device SM count,\n"
    "// $FI_DELIVERY_SM_CAP). 0/unset disables. Host launch/plan code only.\n"
    "static int fi_delivery_sm_cap_min(int n) {\n"
    "  static int cap = -1;\n"
    "  if (cap < 0) {\n"
    "    const char* s = std::getenv(\"FI_DELIVERY_SM_CAP\");\n"
    "    cap = (s != nullptr) ? std::atoi(s) : 0;\n"
    "    if (cap < 0) cap = 0;\n"
    "  }\n"
    "  return (cap > 0 && n > cap) ? cap : n;\n"
    "}\n"
    "}  // namespace fi_delivery_sm_cap\n"
    "#endif\n"
)



def _sm_cap_helper_broken(text: str) -> bool:
    """Helper namespace declared AFTER a clamp site (T0011.den-r12): use-before-
    declaration breaks the TU ("name followed by '::'..." at scheduler.cuh
    :1054/:1594).  The first "namespace fi_delivery_sm_cap" occurrence IS the
    helper (clamp sites only spell fi_delivery_sm_cap::)."""
    ns = text.find("namespace fi_delivery_sm_cap")
    use = text.find("fi_delivery_sm_cap::")
    return ns != -1 and use != -1 and use < ns


def _place_sm_cap_helper(text: str) -> str:
    """(Re)place SM_CAP_HELPER before the first top-level `namespace` line so it
    precedes every clamp site; EOF fallback only when the file has none."""
    if "namespace fi_delivery_sm_cap" in text:
        text = text.replace(SM_CAP_HELPER, "", 1)
        if SM_CAP_HELPER.rstrip("\n") in text:  # tolerate trailing-newline drift
            text = text.replace(SM_CAP_HELPER.rstrip("\n"), "", 1)
    insert_at = None
    lines = text.splitlines(keepends=True)
    for off, line in enumerate(lines):
        if line.startswith("namespace "):
            insert_at = sum(len(l) for l in lines[:off])
            break
    helper = SM_CAP_HELPER.rstrip("\n") + "\n\n"
    if insert_at is None:
        return text.rstrip("\n") + "\n\n" + SM_CAP_HELPER + "\n"
    return text[:insert_at] + helper + text[insert_at:]


def sm_cap_clamp_snippet(var: str, indent: str = "  ") -> str:
    return (
        f"#ifdef {SM_CAP_MACRO}\n"
        f"{indent}// T0011.d host-only SM cap (device SASS unchanged; FlashMLA build-patch contract).\n"
        f"{indent}{var} = fi_delivery_sm_cap::fi_delivery_sm_cap_min({var});\n"
        f"#endif\n"
    )


# Site scope: "file" = every MultiProcessorCount attribute read in the file; "function:NAME" =
# only inside the named function body (brace-matched).  scheduler.cuh carries FA2 plan sites
# that must NOT be clamped (GQA-knob cases) — hence the function scoping there.
HOST_PATCHES = [
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/hopper/prefill_sm90.cuh",
        "scope": "file",
        "var": "multiprocessor_count",
    },
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/hopper/quantization/prefill_sm90.cuh",
        "scope": "file",
        "var": "multiprocessor_count",
    },
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/scheduler.cuh",
        "scope": "function:PrefillSM90Plan",
        "var": "num_sm90_ctas",
    },
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/scheduler.cuh",
        "scope": "function:MLAPlan",
        "var": "num_sm",
    },
    # T0011.d fa2 plan bucketing sites (approved under rev1 item 2, same contract): the fa2
    # plans split/bucket total KV work across num_sm bins, so total-work scaling (KV/batch
    # xN) is absorbed by parallelism and the duration-match guard cannot amortize its fixed
    # offset (den-r12 C finding).  Clamping the plan's SM read caps the split count —
    # plan-consistent (launch grids derive from the same plan run), per-CTA work grows, and
    # CTA counts move toward the 1-2 target.  Host-only: device kernels untouched, SASS
    # audit unchanged.
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/scheduler.cuh",
        "scope": "function:BatchDecodeWithPagedKVCacheWorkEstimationDispatched",
        "var": "num_sm",
    },
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/scheduler.cuh",
        "scope": "function:BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA",
        "var": "num_sm",
    },
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/scheduler.cuh",
        "scope": "function:BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMlaCuteSM80",
        "var": "num_sm",
    },
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/scheduler.cuh",
        "scope": "function:PrefillPlanImpl",
        "var": "num_sm",
    },
    {
        "kind": "sm_cap",
        "path": "flashinfer/attention/scheduler.cuh",
        "scope": "function:TwoStageHolisticPlan",
        "var": "num_sm",
    },
    {
        "kind": "marker_raw",
        "path": "nv_internal/tensorrt_llm/deep_gemm/compiler.cuh",
        "module": "fp8_blockscale_gemm_sm90",
        "sentinel": "T0011.D-MARKER-CFLAG",
        "edits": [
            {
                # Anchor: the -cubin flag push block in compile(); the marker flag propagates
                # the harness's FLASHINFER_EXTRA_CUDAFLAGS decision into the cubin compile.
                "anchor": (
                    r"(?P<head>      flags\.push_back\(\"-cubin\"\);\n)"
                    r"(?P<i>      )flags\.push_back\(\"--expt-relaxed-constexpr\"\);"
                ),
                "insert": (
                    "{head}"
                    "{i}flags.push_back(\"--expt-relaxed-constexpr\");\n"
                    "{i}// T0011.D-MARKER-CFLAG: alternative marker channel — propagate the\n"
                    "{i}// delivery marker macro into the deep_gemm cubin compile (env-gated;\n"
                    "{i}// upstream-identical unless marker mode set FLASHINFER_EXTRA_CUDAFLAGS).\n"
                    "{i}{{\n"
                    "{i}  const char* fi_extra_flags = std::getenv(\"FLASHINFER_EXTRA_CUDAFLAGS\");\n"
                    "{i}  if (fi_extra_flags != nullptr &&\n"
                    "{i}      std::string(fi_extra_flags).find(\"FI_DELIVERY_MARKER_TIMING\") !=\n"
                    "{i}          std::string::npos) {{\n"
                    "{i}    flags.push_back(\"-DFI_DELIVERY_MARKER_TIMING\");\n"
                    "{i}  }}\n"
                    "{i}}}\n"
                ),
                "count": 1,
            }
        ],
    },
    {
        "kind": "marker_raw",
        "path": "nv_internal/tensorrt_llm/deep_gemm/runtime.cuh",
        "module": "fp8_blockscale_gemm_sm90",
        "sentinel": "fi_delivery_bridge_library_markers",
        "edits": [
            {
                # Anchor: after the deep_gemm jit includes, before the namespace opens — the
                # bridge must be declared before Runtime::get_kernel references it.
                "anchor": r"(?P<head>#include \"scheduler\.cuh\"\n)",
                "insert": "{head}\n" + DEEP_GEMM_BRIDGE.strip("\n") + "\n",
                "count": 1,
            },
            {
                # Anchor: after cuLibraryLoadData in Runtime::get_kernel; the bridge copies the
                # TU-local buffer pointers (set by the harness through the module registrar)
                # into the loaded library's globals, re-synced on every kernel fetch.
                "anchor": (
                    r"(?P<head>      CHECK_CUDA\(cuLibraryLoadData\(&lib_, cubin_\.data\(\), "
                    r"nullptr, nullptr, 0, nullptr, nullptr, 0\)\);\n)"
                ),
                "insert": (
                    "{head}"
                    "#ifdef FI_DELIVERY_MARKER_TIMING\n"
                    "      // T0011.d alternative marker channel: the cubin TU's static\n"
                    "      // __device__ fi_marker_*_ptr symbols are only addressable through\n"
                    "      // the library handle, so bridge the harness-set TU pointer values\n"
                    "      // into the library globals (cheap; re-synced per fetch).\n"
                    "      fi_delivery_marker::fi_delivery_bridge_library_markers(lib_);\n"
                    "#endif\n"
                ),
                "count": 1,
            },
        ],
    },
]




def _find_function_span(text: str, name: str) -> tuple[int, int] | None:
    """(open_brace, close_brace) of a named host function's body (first definition)."""
    pattern = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    for match in pattern.finditer(text):
        open_at = text.find("{", match.end() - 1)
        if open_at == -1:
            continue
        depth = 0
        i = open_at
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return open_at, i
            i += 1
    return None


MULTIPROCESSOR_ATTR = re.compile(
    r"(?P<head>[ \t]*FLASHINFER_CUDA_CALL\(\s*)?"
    r"cudaDeviceGetAttribute\(&(?P<var>\w+),\s*cudaDevAttrMultiProcessorCount,"
    r"\s*(?P<dev>\w+)\)\);"
)


def patch_raw_entry(tree: Path, spec: dict, apply: bool) -> dict:
    """Apply one HOST_PATCHES entry (sm_cap / marker_raw) to the tree; idempotent."""
    search_dirs = (
        [tree / "flashinfer/data/include", tree / "include"]
        if spec["kind"] == "sm_cap"
        else [tree / "flashinfer/data/csrc", tree / "csrc"]
    )
    for directory in search_dirs:
        candidate = directory / spec["path"]
        if candidate.is_file():
            path = candidate
            break
    else:
        raise FileNotFoundError(f"{spec['path']} not found under {tree}")

    original = path.read_text(encoding="utf-8")
    report: dict = {"path": str(path), "kind": spec["kind"], "op_classes": spec.get("op_classes", [])}

    if spec["kind"] == "sm_cap":
        scope = spec["scope"]
        clamp_call = f"{spec['var']} = fi_delivery_sm_cap::fi_delivery_sm_cap_min("
        if scope == "file":
            already = clamp_call in original
        else:
            span = _find_function_span(original, scope.split(":", 1)[1])
            body = original[span[0]:span[1] + 1] if span else ""
            already = clamp_call in body
        if already:
            if _sm_cap_helper_broken(original):
                # T0011.den-r12 repair: relocate the EOF helper ahead of the clamp
                # sites (persistent-tree fixup for the 0dfc244e-preceding builds).
                text = _place_sm_cap_helper(original)
                if apply:
                    path.write_text(text, encoding="utf-8")
                report.update({"status": "sm-cap-helper-relocated"})
                report["upstream_sha256"] = hashlib.sha256(original.encode()).hexdigest()
                report["patched_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                return report
            report.update({"status": "already-patched"})
            return report
        text = original
        if scope == "file":
            spans = [(0, len(text) - 1)]
        else:
            assert scope.startswith("function:"), scope
            span = _find_function_span(text, scope.split(":", 1)[1])
            if span is None:
                raise RuntimeError(f"{path}: function {scope} not found")
            spans = [span]
        inserts = 0
        # Apply from the tail so earlier offsets stay valid.
        for open_at, close_at in sorted(spans, reverse=True):
            body = text[open_at:close_at + 1]
            matches = list(MULTIPROCESSOR_ATTR.finditer(body))
            for match in reversed(matches):
                if match.group("var") != spec["var"]:
                    continue
                end = match.end()
                line_break = body.rfind("\n", 0, end)
                indent = re.match(r"[ \t]*", body[line_break + 1:]).group(0) if line_break != -1 else "  "
                body = (
                    body[:end]
                    + "\n"
                    + sm_cap_clamp_snippet(spec["var"], indent).rstrip("\n")
                    + body[end:]
                )
                inserts += 1
            text = text[:open_at] + body + text[close_at + 1:]
        if inserts == 0:
            raise RuntimeError(f"{path}: no cudaDevAttrMultiProcessorCount site for {spec['var']}")
        if "namespace fi_delivery_sm_cap" not in text:
            # T0011.den-r12: the helper must precede every clamp site.  Appending at
            # EOF broke scheduler.cuh (clamp sites at :1054/:1594 -> "name followed by
            # '::' must be a class or namespace name": use-before-declaration).
            # Insert before the first top-level `namespace` line (still namespace
            # scope); fall back to EOF only if the file has none.
            insert_at = None
            for off, line in enumerate(text.splitlines(keepends=True)):
                if line.startswith("namespace "):
                    insert_at = sum(len(l) for l in text.splitlines(keepends=True)[:off])
                    break
            helper = SM_CAP_HELPER if text.endswith("\n") else SM_CAP_HELPER + "\n"
            helper = "\n" + helper.rstrip("\n") + "\n\n"
            if insert_at is None:
                text = text.rstrip("\n") + "\n\n" + SM_CAP_HELPER + "\n"
            else:
                text = text[:insert_at] + helper.lstrip("\n") + text[insert_at:]
        report.update({"clamp_sites": inserts, "var": spec["var"]})
    else:  # marker_raw
        if spec["sentinel"] in original:
            report.update({"status": "already-patched"})
            return report
        text = original
        total = 0
        for edit in spec["edits"]:
            pattern = re.compile(edit["anchor"], re.S)
            text, count = pattern.subn(lambda m, _e=edit: render(_e["insert"], m.groupdict()), text)
            if count != edit["count"]:
                raise RuntimeError(f"{path}: raw anchor matched {count} != {edit['count']}")
            total += count
        report.update({"sites": total})
        if spec["path"].endswith("runtime.cuh"):
            text = text.rstrip("\n") + "\n\n" + registrar_snippet(spec["module"]) + "\n"
            report["registrar"] = spec["module"]

    report["upstream_sha256"] = hashlib.sha256(original.encode()).hexdigest()
    report["patched_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    if apply:
        path.write_text(text, encoding="utf-8")
    return report


def find_kernel_body(text: str, name: str) -> tuple[int, int] | None:
    """(open_brace_offset, close_brace_offset) of `__global__ ... NAME(...)` in text.

    Matches both the literal `__global__` qualifier and CUTLASS's `CUTLASS_GLOBAL`
    macro (the wheel-vendored cutlass headers use the macro form; T0005 rev9).
    """
    pattern = re.compile(r"\b(?:__global__|CUTLASS_GLOBAL)[^;{]*?\b" + re.escape(name) + r"\s*\(", re.S)
    for match in pattern.finditer(text):
        open_at = text.find("{", match.end() - 1)
        if open_at == -1:
            continue
        depth = 0
        i = open_at
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return open_at, i
            i += 1
    return None


def insert_include(text: str) -> str:
    if MARKER_HEADER_NAME in text:
        return text
    # T0011.g: prepend at the very top.  The previous "after the last top-of-file
    # include" heuristic landed inside mha.cu's nested `#ifndef NDEBUG` block while
    # the instrumented kernel_mha body lives under `#ifdef NDEBUG` (-DNDEBUG=1) —
    # mutually exclusive, so the header never reached the snippet (nvcc: identifier
    # fi_marker_* not found).  The header is self-contained (macro-guarded, pulls
    # its own <cuda_runtime.h> under __CUDACC__), so top-of-file is always valid.
    # Every previously patched tree short-circuits above via the sentinel check.
    return MARKER_INCLUDE + "\n" + text


def render(template: str, mapping: dict) -> str:
    """Replace {key} tokens, then unescape {{ / }}; leaves any other braces alone."""
    text = re.sub(r"\{(\w+)\}", lambda m: mapping.get(m.group(1), m.group(0)), template)
    return text.replace("{{", "{").replace("}}", "}")


def patch_tree_entry(tree: Path, spec: dict, apply: bool) -> dict:
    """Resolve, patch and report one table entry against the tree."""
    if spec["kind"] in ("include", "cutlass_include"):
        candidates = [tree / "flashinfer/data/include" / spec["path"],
                      tree / "include" / spec["path"]]
        if spec["kind"] == "cutlass_include":
            # wheel-vendored CUTLASS tree (T0005 rev9): flashinfer/data/cutlass/include
            candidates.insert(0, tree / "flashinfer/data/cutlass/include" / spec["path"])
            candidates.append(tree / "cutlass/include" / spec["path"])
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"{spec['path']} not found under {tree}")
    else:
        for candidate in (
            tree / "flashinfer/data/csrc" / spec["path"],
            tree / "csrc" / spec["path"],
        ):
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"{spec['path']} not found under {tree}")

    original = path.read_text(encoding="utf-8")
    if MARKER_HEADER_NAME in original:
        # Already instrumented (idempotent apply); verify all expected kernels are present.
        missing = [
            kernel for kernel in spec["kernels"]
            if find_kernel_body(original, kernel) is None
        ]
        if missing:
            raise RuntimeError(f"{path}: patched tree missing kernels {missing}")
        return {"path": str(path), "status": "already-patched"}

    text = original
    report: dict = {"path": str(path), "kernels": {}, "op_classes": spec["op_classes"]}
    body_spans: dict[str, tuple[int, int]] = {}
    text = insert_include(text)

    for kernel, region in spec["kernels"].items():
        span = find_kernel_body(text, kernel)
        if span is None:
            raise RuntimeError(f"{path}: kernel {kernel} not found")
        open_at, close_at = span
        body = text[open_at:close_at + 1]
        # Lambda returns inside the body are fine; a bare top-level `return;` needs a closure.
        returns = len(re.findall(r"\breturn\b", body))
        report["kernels"][kernel] = {"region": region, "returns_in_body": returns}
        body_spans[kernel] = span

    # Closures first (they change offsets); each is anchored inside one kernel's body.
    for closure in spec["closures"]:
        open_at, close_at = body_spans[closure["kernel"]]
        body = text[open_at:close_at + 1]

        def _sub(match: re.Match, _closure=closure) -> str:
            fields = {key: value for key, value in match.groupdict().items()
                      if value is not None}
            fields.setdefault("region", _closure["region"])
            return render(_closure["replacement"], fields)

        new_body, count = closure["pattern"].subn(_sub, body)
        expected = closure["count"]
        if count != expected:
            raise RuntimeError(
                f"{path}: closure for {closure['kernel']} matched {count} != {expected}"
            )
        text = text[:open_at] + new_body + text[close_at + 1:]
        report["kernels"][closure["kernel"]]["closed_returns"] = count
        # Recompute spans after edits.
        for other_kernel, _ in spec["kernels"].items():
            span = find_kernel_body(text, other_kernel)
            if span is None:
                raise RuntimeError(f"{path}: kernel {other_kernel} lost after closure edit")
            body_spans[other_kernel] = span

    # Entry/exit per kernel, from the tail so earlier offsets stay valid.
    for kernel, region in sorted(
        spec["kernels"].items(), key=lambda item: body_spans[item[0]][0], reverse=True
    ):
        open_at, close_at = body_spans[kernel]
        text = (
            text[:open_at + 1]
            + "\n"
            + entry_snippet(region).rstrip("\n")
            + "\n"
            + text[open_at + 1:close_at].rstrip("\n")
            + "\n"
            + exit_snippet(region).rstrip("\n")
            + "\n"
            + text[close_at:]
        )

    # Enroll this file's TU setter into the delivery registry (appended at file end; the macro
    # re-opens namespace fi_delivery_marker, so namespace context does not matter).
    text = text.rstrip("\n") + "\n\n" + registrar_snippet(spec["module"])

    report["upstream_sha256"] = hashlib.sha256(original.encode()).hexdigest()
    report["patched_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    if apply:
        path.write_text(text, encoding="utf-8")
    return report


def copy_marker_header(tree: Path, destination_dirs: list[Path]) -> list[str]:
    source = Path(__file__).resolve().parents[1] / "harness/override" / MARKER_HEADER_NAME
    copied = []
    for directory in destination_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / MARKER_HEADER_NAME
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(str(target))
    return copied


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, help="wheel site dir or git source root")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sm-cap", action="store_true", dest="sm_cap",
                        help="also apply/check the T0011.d host-side SM-cap patches "
                             "(compile-time -DFI_DELIVERY_SM_CAP; opt-in like the FlashMLA "
                             "build patch; device SASS unchanged)")
    parser.add_argument("--log", type=Path, help="JSON log path (apply mode)")
    parser.add_argument("--list", action="store_true", help="print the table and exit")
    args = parser.parse_args(argv)

    if args.list:
        for spec in PATCHES:
            print(f"{','.join(spec['op_classes'])}: {spec['path']}")
            for kernel, region in spec["kernels"].items():
                print(f"    {kernel} -> {region}")
        print("host patches (T0011.d):")
        for spec in HOST_PATCHES:
            scope = spec.get("scope", "edits")
            print(f"    {spec['kind']}: {spec['path']} [{scope}]")
        return 0
    if not args.tree:
        parser.error("--tree is required (or use --list)")
    if args.apply == args.check:
        parser.error("exactly one of --apply / --check")

    tree = args.tree.resolve()
    host_patches = [spec for spec in HOST_PATCHES
                    if args.sm_cap or spec["kind"] != "sm_cap"]
    if args.apply:
        # Pre-resolve file paths so the marker header can be copied next to every patched
        # file (quote-include resolution is relative to the including file's directory).
        resolved: list[tuple[dict, Path]] = []
        for spec in PATCHES:
            prefix = "flashinfer/data" if (tree / "flashinfer").is_dir() else "."
            sub = {"include": "include", "cutlass_include": "cutlass/include",
                   "csrc": "csrc"}[spec["kind"]]
            base = tree / prefix / sub
            path = base / spec["path"]
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
            resolved.append((spec, path))
        for spec in host_patches:
            prefix = "flashinfer/data" if (tree / "flashinfer").is_dir() else "."
            sub = "include" if spec["kind"] == "sm_cap" else "csrc"
            path = tree / prefix / sub / spec["path"]
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
            resolved.append((spec, path))
        copied = copy_marker_header(
            tree, sorted({path.parent for _, path in resolved})
        )
    else:
        resolved = []
        copied = []

    reports = []
    try:
        for spec, _path in (resolved or [(spec, None) for spec in PATCHES + host_patches]):
            if any(spec is host for host in HOST_PATCHES):
                reports.append(patch_raw_entry(tree, spec, apply=args.apply))
            else:
                reports.append(patch_tree_entry(tree, spec, apply=args.apply))
    except (FileNotFoundError, RuntimeError) as error:
        print(f"patch_wheel_marker: {error}", file=sys.stderr)
        return 1

    payload = {"tree": str(tree), "mode": "apply" if args.apply else "check",
               "marker_header_copies": copied, "files": reports}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
