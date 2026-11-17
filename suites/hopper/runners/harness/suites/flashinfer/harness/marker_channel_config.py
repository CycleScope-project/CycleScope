"""Per-op-class marker channel configuration for the FlashInfer delivery suite.

Source of truth for T0005 修订 5 (23 operation classes, dev 23 + sealed 23; dev-only
``packbits``, sealed-only ``segment_packbits``):

  - ``jit_modules``: the FlashInfer JIT module names the op loads (cached_ops/<name>/<name>.so);
    the marker channel dlopens exactly these and dlsym ``flashinfer_delivery_marker_set``.
  - ``primary``: (module, region index, kernel symbol stem) of the formal-denominator kernel —
    the kernel whose outermost clock-marker interval is ``timing.main_kernel`` (296 contract,
    docs/simulator.md §2.5).  Multi-kernel ops follow the marker_readiness rule: the formal
    denominator is the selected primary kernel, secondary kernels stay evidence-only.
  - ``sass_gate``: static SASS minimums (SR_CLOCKLO / SR_GLOBALTIMERLO >= 2 per instrumented
    kernel; scripts/marker_sass_audit.py).
  - ``aot``: static-verifiability class.  ``nvcc_header`` = JIT modules are plain nvcc units —
    CPU-pod AOT + SASS check fully verifies the patch; ``nvrt``/``dsl`` = generated/NVRTC/DSL
    paths that cannot be AOT-compiled standalone — patch present, runtime proof is a G0 item.
"""

from __future__ import annotations

PRIMARY_REGION = 0

# kernel symbol stem -> (jit module, region index) for every instrumented kernel.
KERNEL_REGIONS: dict[str, tuple[str, int]] = {
    # page / quantization
    "AppendPagedKVCacheKernel": ("page", 0),
    "PackBitsKernel": ("quantization", 0),
    "SegmentPackBitsKernel": ("quantization", 1),
    # norm
    "RMSNormKernel": ("norm", 0),
    "FusedAddRMSNormKernel": ("norm", 1),
    # rope
    "BatchQKApplyRotaryKernel": ("pos_enc", 0),
    # cascade
    "MergeStateKernel": ("cascade", 0),
    "MergeStatesKernel": ("cascade", 1),
    "PersistentVariableLengthMergeStatesKernel": ("cascade", 2),
    # sampling (sample / top_k_sample / top_p_sample)
    "SamplingFromProbKernel": ("sampling", 0),
    "TopKSamplingFromProbKernel": ("sampling", 1),
    "TopPSamplingFromProbKernel": ("sampling", 2),
    "TopKTopPSamplingFromProbKernel": ("sampling", 3),
    # fa2 attention
    "SingleDecodeWithKVCacheKernel": ("decode", 0),
    "BatchDecodeWithPagedKVCacheKernel": ("decode", 1),
    "BatchDecodeWithPagedKVCacheKernelMLA": ("decode", 2),
    "SinglePrefillWithKVCacheKernel": ("prefill", 0),
    "BatchPrefillWithRaggedKVCacheKernel": ("prefill", 1),
    "BatchPrefillWithPagedKVCacheKernel": ("prefill", 2),
    # fa3 / fp8 attention (hopper)
    "PrefillWithKVCacheKernel": ("prefill_sm90", 0),
    "FP8PrefillWithKVCacheKernel": ("prefill_sm90_quant", 0),
    # mla
    "BatchMLAPagedAttentionKernel": ("mla", 0),
    "BatchMLAPageAttentionHopperKernel": ("mla_hopper", 0),
    # mamba
    "selective_state_update_kernel_simple": ("mamba_stp", 0),
    "selective_state_update_kernel_producer_consumer_horizontal": ("mamba_stp", 1),
    "selective_state_update_kernel_producer_consumer_vertical": ("mamba_stp", 2),
    # xqa
    "kernel_mha": ("xqa", 0),
    # deepgemm sm90 fp8 (NVRTC-compiled kernel inside the fp8_blockscale_gemm_sm90 module)
    "fp8_gemm_kernel": ("fp8_blockscale_gemm_sm90", 0),
    "fp8_gemm_kernel_swapAB": ("fp8_blockscale_gemm_sm90", 1),
    # wheel-vendored cutlass generic launch wrapper (T0005 rev9, far-holdout G0
    # on-site item): one instrumented region around Operator::operator(); the
    # JIT modules carrying cutlass GEMMs are the segment-gemm / fused-moe ones.
    "device_kernel": ("cutlass_device_kernel", 0),
}

# JIT module name -> module-name prefixes that carry instrumented kernels.  The attention
# modules are generated per config with suffixed names, and fa3 variants share the fa2 prefix
# with a trailing `_sm90`, so the channel matches by (prefix, must/end) pairs (never exact).
MODULE_PREFIXES: dict[str, tuple[tuple[str, bool], ...]] = {
    "page": (("page", False),),
    "quantization": (("quantization", False),),
    "norm": (("norm", False),),
    # T0011.g freshness fix: 0.6.17's rope JIT module is named "rope"
    # (flashinfer/jit/rope.py gen_jit_spec("rope", ...)), not "pos_enc" — the
    # configured prefix never matched cached_ops, so every rope case audited as
    # "no loaded JIT binary for required primary module" (sh2/sh3 2026-10-01).
    "pos_enc": (("pos_enc", False), ("rope", False)),
    "cascade": (("cascade", False),),
    "sampling": (("sampling", False),),
    "decode": (("single_decode_with_kv_cache_", False), ("batch_decode_with_kv_cache_", False)),
    "prefill": (
        ("single_prefill_with_kv_cache_", False),
        ("batch_prefill_with_kv_cache_", False),
    ),
    "prefill_sm90": (
        ("single_prefill_with_kv_cache_", True),
        ("batch_prefill_with_kv_cache_", True),
    ),
    "prefill_sm90_quant": (
        ("single_prefill_with_kv_cache_", True),
        ("batch_prefill_with_kv_cache_", True),
    ),
    "mla": (("batch_mla_attention_", False),),
    "mla_hopper": (("batch_mla_attention_", True),),
    "mamba_stp": (("selective_state_update", False),),
    "xqa": (("xqa_", False),),
    "fp8_blockscale_gemm_sm90": (("fp8_blockscale_gemm_sm90", True),),
    # The registrar enrolls the patched cutlass device_kernel.h TU under this key; the
    # carrier JIT modules are the segment-gemm (gemm_sm90*) and fused-moe (fused_moe*)
    # ones (T0005 rev9).
    "cutlass_device_kernel": (("gemm_sm90", True), ("fused_moe", False)),
}

OP_CLASSES: dict[str, dict] = {
    "append_paged_kv": {
        "jit_modules": ["page"],
        "primary": ("page", 0, "AppendPagedKVCacheKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "packbits": {
        "jit_modules": ["quantization"],
        "primary": ("quantization", 0, "PackBitsKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "segment_packbits": {
        "jit_modules": ["quantization", "gemm"],
        "primary": ("quantization", 0, "SegmentPackBitsKernel"),
        "secondary": [],
        "aot": "nvcc_header_multi",
        "backend_key": None,
        "note": "multi-phase op (packbits + segment gemm); formal denominator = packbits phase "
                "per holdout_far_sealed marker_fields primary_kernel_selection=multi",
    },
    "rmsnorm": {
        "jit_modules": ["norm"],
        "primary": ("norm", 0, "RMSNormKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "fused_add_rmsnorm": {
        "jit_modules": ["norm"],
        "primary": ("norm", 1, "FusedAddRMSNormKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "rope": {
        "jit_modules": ["pos_enc"],
        "primary": ("pos_enc", 0, "BatchQKApplyRotaryKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "merge_state": {
        "jit_modules": ["cascade"],
        "primary": ("cascade", 0, "MergeStateKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "merge_states": {
        "jit_modules": ["cascade"],
        "primary": ("cascade", 1, "MergeStatesKernel"),
        "secondary": [("cascade", 2, "PersistentVariableLengthMergeStatesKernel")],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "sample": {
        "jit_modules": ["sampling"],
        "primary": ("sampling", 0, "SamplingFromProbKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "top_k_sample": {
        "jit_modules": ["sampling"],
        "primary": ("sampling", 1, "TopKSamplingFromProbKernel"),
        "secondary": [("sampling", 0, "SamplingFromProbKernel")],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "top_p_sample": {
        "jit_modules": ["sampling"],
        "primary": ("sampling", 2, "TopPSamplingFromProbKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
    },
    "single_decode": {
        "jit_modules": ["decode", "prefill_sm90"],
        "primary": ("decode", 0, "SingleDecodeWithKVCacheKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": "backend",
        "backend_overrides": {"fa3": ("prefill_sm90", 0, "PrefillWithKVCacheKernel")},
    },
    "batch_decode_paged": {
        "jit_modules": ["decode", "prefill", "prefill_sm90"],
        "primary": ("decode", 1, "BatchDecodeWithPagedKVCacheKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": "backend",
        "backend_overrides": {
            "fa3": ("prefill_sm90", 0, "PrefillWithKVCacheKernel"),
            "fa2_mla": ("decode", 2, "BatchDecodeWithPagedKVCacheKernelMLA"),
        },
    },
    "single_prefill": {
        "jit_modules": ["prefill", "prefill_sm90", "prefill_sm90_quant"],
        "primary": ("prefill", 0, "SinglePrefillWithKVCacheKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": "backend",
        "backend_overrides": {
            "fa3": ("prefill_sm90", 0, "PrefillWithKVCacheKernel"),
            "fa3_fp8": ("prefill_sm90_quant", 0, "FP8PrefillWithKVCacheKernel"),
        },
    },
    "batch_prefill_paged": {
        "jit_modules": ["prefill", "prefill_sm90", "prefill_sm90_quant"],
        "primary": ("prefill", 2, "BatchPrefillWithPagedKVCacheKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": "backend",
        "backend_overrides": {
            "fa3": ("prefill_sm90", 0, "PrefillWithKVCacheKernel"),
            "fa3_fp8": ("prefill_sm90_quant", 0, "FP8PrefillWithKVCacheKernel"),
        },
    },
    "batch_prefill_ragged": {
        "jit_modules": ["prefill", "prefill_sm90"],
        "primary": ("prefill", 1, "BatchPrefillWithRaggedKVCacheKernel"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": "backend",
        "backend_overrides": {"fa3": ("prefill_sm90", 0, "PrefillWithKVCacheKernel")},
    },
    "batch_mla_paged": {
        "jit_modules": ["mla", "mla_hopper"],
        "primary": ("mla_hopper", 0, "BatchMLAPageAttentionHopperKernel"),
        "secondary": [("mla", 0, "BatchMLAPagedAttentionKernel")],
        "aot": "nvcc_header",
        "backend_key": "backend",
        "backend_overrides": {"mla_fa2": ("mla", 0, "BatchMLAPagedAttentionKernel")},
    },
    "mamba_state_update": {
        "jit_modules": ["mamba_stp"],
        "primary": ("mamba_stp", 0, "selective_state_update_kernel_simple"),
        "secondary": [],
        "aot": "nvcc_header",
        "backend_key": None,
        "note": "the op may select the producer_consumer_horizontal/vertical variants at "
                "runtime (secondary regions 1/2); the timing channel records whichever region "
                "carries valid rows and the primary config is resolved per launch",
    },
    "xqa_batch_decode": {
        "jit_modules": ["xqa"],
        "primary": ("xqa", 0, "kernel_mha"),
        "secondary": [],
        "aot": "nvcc_csrc",
        "backend_key": None,
    },
    "fp8_blockscale_gemm": {
        "jit_modules": ["fp8_blockscale_gemm_sm90"],
        "primary": ("fp8_blockscale_gemm_sm90", 0, "fp8_gemm_kernel"),
        "secondary": [],
        "aot": "nvrtc",
        "backend_key": None,
        "note": "fp8_gemm_kernel is compiled by NVRTC from csrc/nv_internal deep_gemm sources "
                "at runtime; patch is applied to the tree the NVRTC include dirs point at, "
                "runtime SASS/trace proof is a G0 item",
    },
    "segment_gemm": {
        "jit_modules": ["gemm_sm90"],
        "primary": ("cutlass_device_kernel", 0, "device_kernel"),
        "secondary": [],
        "aot": "cutlass_device_kernel",
        "backend_key": None,
        "note": "kernel is cutlass::device_kernel<GemmUniversal<GroupProblemShape,...>> from "
                "the wheel's vendored cutlass; rev9 wheel patch instruments the generic "
                "device_kernel wrapper (module cutlass_device_kernel, region 0). H100 probe "
                "2026-09-30: exactly one GemmUniversal device_kernel launch per op run "
                "(+ one ~1.3us group_gemm args kernel), so region rows are single-kernel.",
    },
    "cutlass_fused_moe": {
        "jit_modules": ["fused_moe"],
        "primary": ("cutlass_device_kernel", 0, "device_kernel"),
        "secondary": [],
        "aot": "cutlass_device_kernel",
        "backend_key": None,
        # T0011 修订 5 ② static primary selection (closes the rev9 G0 item; replaces the
        # "decide single-instantiation vs per-launch resolution from the runtime list"
        # open question — see the timing-semantics paragraph below).
        #
        # Selection criterion (structural, not a guessed kernel name): the moe GEMM phases
        # (tensorrt_llm CutlassMoeFCRunner -> MoeGemmRunner sm90 TMA-WS launchers,
        # moe_gemm_tma_ws_launcher.inl / moe_gemm_tma_ws_mixed_input_launcher.inl) build
        # `cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>` and launch it via
        # gemm.run(), which is `cutlass::device_kernel<GemmKernel>` — the exact generic
        # 3.x launch template the rev9 wheel patch instruments (PATCHES entry
        # op_classes=["segment_gemm","cutlass_fused_moe"], cutlass/device_kernel.h, one
        # region under module key cutlass_device_kernel).  `device_kernel` is the only
        # kernel-symbol family shared verbatim with the ALREADY-AUDITED segment_gemm
        # carrier (MODULE_PREFIXES["cutlass_device_kernel"] lists both carriers:
        # gemm_sm90*_sm90 and fused_moe*), so the same SASS audit path (CS2R SR_CLOCKLO /
        # SR_GLOBALTIMERLO >= 2 per kernel, jit_freshness_audit on the loaded fused_moe*.so)
        # applies unchanged.  The moe routing/expert-map/activation kernels in the same
        # module are moe-private, NOT instrumented by any patch entry, and therefore cannot
        # carry a marker region — they are not primary candidates.
        #
        # Timing semantics: one op call launches TWO instrumented device_kernel instances
        # (gemm1 w13, gemm2 w2; persistent sm_count-sized grouped grids).  Marker rows use
        # per-launch linear (cta,warp) indices into the same region-0 buffer, so gemm2's
        # rows overwrite gemm1's at overlapping indices and the region retains the LAST
        # GEMM launch of the window; single region, single candidate, no per-launch
        # resolution.  The kineto duration guard compares the region span against the
        # median of the op's device_kernel events (H100 moe probe t11a-moe-probe proves
        # the pair passes the +/-10% band per case).
        "note": "multi-kernel op (moe topk/gemm phases); primary is the instrumented "
                "cutlass device_kernel wrapper shared with segment_gemm — selection "
                "criterion and two-launch overwrite semantics documented above",
    },
    "gdn_prefill": {
        "jit_modules": [],
        "primary": None,
        "secondary": [],
        "aot": "cute_dsl",
        "backend_key": None,
        "note": "gdn_kernels are CuTe-DSL (MLIR->PTX) compiled from Python, no nvcc TU in the "
                "compile path; C++ header instrumentation does not reach them",
    },
    "gdn_decode": {
        "jit_modules": [],
        "primary": None,
        "secondary": [],
        "aot": "cute_dsl",
        "backend_key": None,
        "note": "same CuTe-DSL path as gdn_prefill",
    },
}


def resolve_primary(op_class: str, case: dict | None = None) -> tuple[str, int, str] | None:
    """Primary (module, region, kernel stem) for one case, honoring backend overrides."""
    entry = OP_CLASSES[op_class]
    primary = entry["primary"]
    backend_key = entry.get("backend_key")
    if backend_key and case is not None:
        backend = str(case.get(backend_key) or "").lower()
        dtype = str(case.get("dtype") or "").lower()
        overrides = entry.get("backend_overrides") or {}
        if backend == "fa3" and "fp8" in dtype:
            fa3_fp8 = overrides.get("fa3_fp8")
            if fa3_fp8 is not None:
                return fa3_fp8
        if backend in overrides:
            return overrides[backend]
    return primary


def match_module(loaded_name: str, module_key: str, *, require_sm90: bool | None = None) -> bool:
    """Match a loaded JIT module name against one config module key.

    ``require_sm90``: None = any variant; True = only the fa3 (`..._sm90`) variant; False = only
    non-sm90 (fa2) variants.  mamba's plain prefix would also swallow `mamba_stp`-style names,
    which do not exist today; keep the guard anyway.
    """
    matched = False
    for prefix, is_sm90 in MODULE_PREFIXES.get(module_key, ((module_key, False),)):
        if loaded_name == prefix or loaded_name.startswith(prefix):
            if require_sm90 is None or require_sm90 == is_sm90:
                matched = True
    return matched
