#!/usr/bin/env bash
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/tma_forms/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin"
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  # tma.cu cell sets first (default / store / blkseq / rank); each
  # writes tma.jsonl -> rename so later sets do not clobber earlier ones.
  for cs in default store blkseq rank; do
    timeout 1800 "$B/tma" --cells "$cs" --out "$OUT/$run" \
        --launches 16 --reps 33 >> "$OUT/$run.stdout" 2>&1 \
        || echo "PROBE_FAIL tma/$cs $run rc=$?"
    [ -f "$OUT/$run/tma.jsonl" ] && \
        mv "$OUT/$run/tma.jsonl" "$OUT/$run/tma_$cs.jsonl"
  done
  # addendum (a): one process per family so a failure cannot leak
  for fam in ldgsts stas gather4 scatter4 ublkcp ublkcp_mc im2col; do
    timeout 1800 "$B/tma_forms" --cells "$fam" --out "$OUT/$run" \
        --launches 16 --reps 33 >> "$OUT/$run.stdout" 2>&1 \
        || echo "PROBE_FAIL tma_forms/$fam $run rc=$?"
    [ -f "$OUT/$run/tma_forms.jsonl" ] && \
        mv "$OUT/$run/tma_forms.jsonl" "$OUT/$run/tma_forms_$fam.jsonl"
  done
done
