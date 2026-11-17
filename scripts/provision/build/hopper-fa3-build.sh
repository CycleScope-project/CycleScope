#!/usr/bin/env bash
# hopper-fa3-build: FlashAttention-3 sm_90a site-packages (_C.so + flash_attn_3).
# evidence: harness suites/flashattention3/scripts/build_full_gpu.sh
#   (prepare -> 4 shards -> link; FLASH_ATTENTION_DISABLE_SM80=TRUE,
#    FORCE_CXX11_ABI=TRUE, TORCH_CUDA_ARCH_LIST=9.0a; L13-20).
# inputs: hopper-flashattention-source.
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-fa3-build
H=$CYCLESCOPE_ROOT/suites/hopper/runners/harness
S=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-flashattention-source
rm -rf "$T"
mkdir -p "$T"
ln -sfn "$S" "$H/suites/flashattention3/upstream"   # mirror .gitignore'd link
export FLASHATTENTION3_FULL_GPU_BUILD_ROOT="$T"     # evidence: build_full_gpu.sh:17
export FLASH_ATTENTION_DISABLE_SM80=TRUE            # evidence: build_full_gpu.sh:18
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE         # evidence: build_full_gpu.sh:19
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"  # evidence: build_full_gpu.sh:20
# shards in parallel — evidence: build_full_gpu.sh L9-12 "prefer 4
# parallel shard jobs"; launch the pod with --cpu 32 (~1h vs ~4h serial)
bash "$H/suites/flashattention3/scripts/build_full_gpu.sh" prepare
for s in 0 1 2 3; do
  bash "$H/suites/flashattention3/scripts/build_full_gpu.sh" shard "$s" &
done
wait
bash "$H/suites/flashattention3/scripts/build_full_gpu.sh" link
test -f "$T/site-packages/flash_attn_3/_C.so" || \
  find "$T" -name '_C*.so' | grep -q .
