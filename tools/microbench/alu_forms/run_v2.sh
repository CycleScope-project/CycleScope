#!/usr/bin/env bash
#   TREE=<clone> OUT=<run dir> SM_ARCH=... [CUDA_HOME=..] [SKIP_BUILD=1] bash run_v2.sh
# Run 2: only the addendum-1 kinds imm_ax / rr_ax / imm_as (one --only
# invocation per kind; --only is a substring filter that cannot select
# all three at once), outputs concatenated into <run>/alu_forms.jsonl.
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
KINDS="imm_ax rr_ax imm_as"
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/alu_forms/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/alu_forms"
[ -x "$B" ] || { echo "build failed"; exit 1; }
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  : > "$OUT/$run/alu_forms.jsonl"
  for k in $KINDS; do
    mkdir -p "$OUT/$run/$k"
    timeout 3600 "$B" --out "$OUT/$run/$k" --launches 16 --reps 65 --only "$k" \
        >> "$OUT/$run.stdout" 2>&1 || echo "PROBE_FAIL $run $k"
    cat "$OUT/$run/$k/alu_forms.jsonl" >> "$OUT/$run/alu_forms.jsonl" 2>/dev/null \
        || echo "NO_JSONL $run $k"
  done
done
echo "run_v2.sh: alu_forms done ($SM_ARCH)"
