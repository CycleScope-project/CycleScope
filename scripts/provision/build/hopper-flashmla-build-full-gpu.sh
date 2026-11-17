#!/usr/bin/env bash
# hopper-flashmla-build-full-gpu: FlashMLA site-packages (sm90a) via the harness
# delivery pipeline — evidence: suites/flashmla/scripts/build_cluster_sm.sh
# (marker timing ON, CUTLASS pinned 147f5673 to tool_cache, per PINNED.md
# L46; FlashMLA rev ba89a34 is separate — do not confuse the two).
# inputs: hopper-flashmla-source, hopper-flashmla-cutlass-source,
#         cuda-toolkit-13.2.
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-flashmla-build-full-gpu
H=$CYCLESCOPE_ROOT/suites/hopper/runners/harness/suites/flashmla
S=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-flashmla-source
C=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-flashmla-cutlass-source
rm -rf "$T"
mkdir -p "$T" "$T/work"
cp -a "$H" "$T/work/suite"
rm -rf "$T/work/suite/upstream"
cp -a "$S" "$T/work/suite/upstream"        # deref read-only source link
# full-gpu variant predates the effective-SM patch: revert
# sparse_decode.cpp to hopper-prototype@86411893 (2026-10-20), i.e. drop
# flashmla_scaling::effective_sm_count added by 0f170d5e (10-21) —
# evidence: old c450_full/flashmla_build mtime 2026-10-20 06:59
patch -s -p1 -d "$T/work/suite/upstream" \
  -i "$CYCLESCOPE_ROOT/suites/hopper/runners/patches/flashmla/sparse_decode_fullgpu.patch"
export HOPPER_ARTIFACT_ROOT="$CYCLESCOPE_ARTIFACT_ROOT"   # tool_cache/cutlass/<sha>
export FLASHMLA_DELIVERY_ARTIFACTS_ROOT="$T"
export FLASHMLA_BUILD_ROOT="$T"                        # outputs at root (evidence)
export FLASHMLA_DELIVERY_MARKER_TIMING=1
export FLASHMLA_FORCE_REBUILD=1
ln -sfn "$C" "$CYCLESCOPE_ARTIFACT_ROOT/tool_cache/cutlass/147f5673d0c1c3dcf66f78d677fd647e4a020219" 2>/dev/null || { mkdir -p "$CYCLESCOPE_ARTIFACT_ROOT/tool_cache/cutlass"; ln -sfn "$C" "$CYCLESCOPE_ARTIFACT_ROOT/tool_cache/cutlass/147f5673d0c1c3dcf66f78d677fd647e4a020219"; }
# harness expects NVDISASM/CUOBJDUMP when not on PATH (L163-169)
# cuobjdump calls nvdisasm internally → it must be on PATH, not only
# in NVDISASM_PATH (fmles2 rc=1 evidence)
# append-only bin dir: nvcc must stay the image 12.9 (torch 12.9 match),
# cuobjdump needs nvdisasm on PATH internally (fmles3 evidence)
mkdir -p "$T/work/bin"
ln -sfn "$CYCLESCOPE_ARTIFACT_ROOT/tools/cuda-toolkit-13.2/bin/nvdisasm" "$T/work/bin/nvdisasm"
ln -sfn "$CYCLESCOPE_ARTIFACT_ROOT/tools/cuda-toolkit-13.2/bin/cuobjdump" "$T/work/bin/cuobjdump"
export PATH="$PATH:$T/work/bin"
export NVDISASM_PATH="$CYCLESCOPE_ARTIFACT_ROOT/tools/cuda-toolkit-13.2/bin/nvdisasm"
export CUOBJDUMP_PATH="$CYCLESCOPE_ARTIFACT_ROOT/tools/cuda-toolkit-13.2/bin/cuobjdump"
bash "$T/work/suite/scripts/build_cluster_sm.sh" h100
test -f "$T/site-packages-flashmla/flash_mla/__init__.py"
rm -rf "$T/work"
