#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:?}
HARDWARE=${HARDWARE:?}
CASES=${CASES:?}
TRACER=${TRACER:?}
A=${CYCLESCOPE_ARTIFACT_ROOT:?}
mkdir -p "$ROOT/environment"
trap 'rc=$?; printf "TMEM_CAPTURE_RC=%s\n" "$rc"; sync' EXIT
/usr/local/cuda/bin/nvcc --version > "$ROOT/environment/nvcc.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$ROOT/environment/gpu.csv"
while IFS= read -r cell; do
  out="$ROOT/cap/$cell"
  test ! -e "$out/launches.jsonl"
  mkdir -p "$out/scratch" "$out/home" "$out/tmp" "$out/placement"
  for mode in trace placement; do
    target="$out"
    if [ "$mode" = placement ]; then target="$out/placement"; fi
    (
      cd "$out/scratch"
      timeout 600 env CUDA_INJECTION64_PATH="$TRACER" CYCLESCOPE_TRACE_DIR="$target" \
        CYCLESCOPE_CAPTURE_LAUNCHES=".*k_pipeline.*@2" CYCLESCOPE_CAPTURE_CLUSTERS=all \
        CYCLESCOPE_CAPTURE_MODE="$mode" CYCLESCOPE_L2_FOOTPRINT=0 CYCLESCOPE_CUDA_STACK_SIZE=65536 \
        CYCLESCOPE_TRACER_VERBOSE=1 HOME="$out/home" TMPDIR="$out/tmp" \
        LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64 \
        PATH="$A/tools/bin:/usr/local/cuda/bin:$PATH" \
        NVDISASM="$A/tools/cuda-toolkit-13.2/bin/nvdisasm" \
        "$HARDWARE/build/tmem_pipeline" --only "=$cell" --out "$target" --launches 1 --reps 17
    ) > "$out/$mode-run.log" 2>&1
  done
  printf "CELL_CAPTURED=%s\n" "$cell"
done < "$CASES"
sync
