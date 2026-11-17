"""Frozen far input contracts and real vendored-source import setup."""
from __future__ import annotations

import os
import sys
from pathlib import Path


class AssetContractError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def configure_flashinfer() -> Path:
    """Use the intact vendored package, never rewrite package initializers."""
    prefix = Path(os.environ.get(
        "BW_FI_SRC") or os.environ.get("BW_DEPS_DIR") or
        str(Path(os.environ["CYCLESCOPE_ARTIFACT_ROOT"])
            / "tools" / "blackwell-deps" / "flashinfer"))
    source = prefix / "src"
    if not (source / "flashinfer" / "__init__.py").is_file():
        raise AssetContractError("source_unavailable", f"missing FlashInfer source: {source}")
    sys.path.insert(0, str(source))
    os.environ.setdefault("FLASHINFER_NO_DOWNLOAD", "1")
    return source


def check_case_contract(case: dict) -> None:
    """Reject unavailable frozen entries/types instead of substituting kernels."""
    if case["family"] == "fi_grouped_contiguous":
        raise AssetContractError(
            "asset_api_unavailable", "vendored contiguous module supports FP8 only; "
            "frozen NVFP4 GroupedGemmContiguousBlackwell entry is unavailable")
    if case["family"] in ("fi_bs_gemm_sm100", "fi_bs_gemm_sm103"):
        ab_dtype = case["params"]["ab_dtype"]
        if ab_dtype not in ("float4_e2m1fn", "float8_e4m3fn", "float8_e5m2"):
            raise AssetContractError(
                "spec_dtype_unavailable", f"frozen ab_dtype={ab_dtype!r} is not a supported "
                "data operand; E8M0 is a scale format, no replacement dtype authorized")
    if case["family"] == "fi_fmha_prefill" and \
            case["params"].get("ex2_emulation"):
        raise AssetContractError(
            "spec_arch_unavailable", "frozen ex2_emulation=True is rejected by "
            "vendored FMHA for arch>=sm_103 (fmha.py assert); disabling it "
            "would drop the frozen measurement axis")
