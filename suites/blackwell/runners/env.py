"""Default environment for blackwell suite runners.

The capture driver imports this module and applies ``defaults()`` to the
subprocess environment before ``resolve`` (values mirror the old
blackwell ``scripts/cluster/gpu_capture.sh`` block).  All external
dependencies resolve under ``$CYCLESCOPE_ARTIFACT_ROOT/tools/``; see
``configs/capture_tools.json`` for what each name provisions.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def defaults() -> dict:
    groot = os.environ["CYCLESCOPE_ARTIFACT_ROOT"]
    tools = f"{groot}/tools"
    return {
        "CYCLESCOPE_RUNNER_ROOT": str(ROOT),
        "BW_CASES_DIR": str(ROOT.parent / "full_gpu"),
        "TRITON_SOURCE_ROOT": f"{tools}/triton-source",
        "BW_FI_SRC": f"{tools}/blackwell-flashinfer-deps",
        "BW_FLASHMLA_PREFIX": f"{tools}/blackwell-flashmla-build",
        "BW_FLASHMLA_SRC": f"{tools}/blackwell-flashmla-source",
        "FLASHINFER_WORKSPACE_BASE": "<CAPTURE>/scratch/flashinfer-workspace",
        "FLASHINFER_CUBIN_DIR": f"{tools}/flashinfer-cubins",
        "FLASHINFER_NVCC_THREADS": "4",
        "MAX_JOBS": "4",
        "BW_CUTLASS_DIR": f"{tools}/blackwell-cutlass-source",
        "BW_DEPS_DIR": f"{tools}/blackwell-deps",
        "BW_CORPUS_ROOT": f"{tools}/blackwell-corpus",
        "BW_FA4_PREFIX": f"{ROOT}/.deps/fa4",
        "BW_CUTLASS_BIN_ROOT": f"{tools}/blackwell-corpus/cutlass",
    }
