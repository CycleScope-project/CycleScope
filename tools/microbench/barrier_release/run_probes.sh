#!/usr/bin/env bash
# barrier_release probes are clock64 probes: plain execution, no ncu.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run_probes.sh
# Protocol v2 (protocol-v2.json): every probe --reps 256 (sync_bar kMaxReps)
# --launches 8, two runs (run-a, run-b) per device; the SASS checks of
# sync_mb / sync_xw are written to $OUT/checks.json (sass_check.py verbatim).
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
REPS=${REPS:-256}; LAUNCHES=${LAUNCHES:-8}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/barrier_release/build.sh" > "$OUT/build.log" 2>&1
fi
cat "$OUT/build/build.tsv" || true
python3 "$TREE/tools/microbench/barrier_release/sass_check.py" --sass "$OUT/build/sass" \
    --out "$OUT/checks.json" > "$OUT/checks.stdout" 2>&1 || echo "SASS_CHECK_FAIL rc=$?"
python3 "$TREE/tools/microbench/crossings/cta.release/sass_check.py" --sass "$OUT/build/sass" \
    --out "$OUT/checks_bar.json" > "$OUT/checks_bar.stdout" 2>&1 || echo "SASS_CHECK_BAR_FAIL rc=$?"
B="$OUT/build/bin"
# sm_90a has no tcgen05: the c1 / c2 (tcgen05.commit) and pair_skew /
# flag_rtt cells of sync_mb / sync_xw never complete there (unrealised);
# only the mbarrier arrive / tx / tw_only cells are run on sm_90a.
declare -A ONLY=()
if [ "$SM_ARCH" = sm_90a ]; then
  ONLY[sync_mb]="--only arr_,tx_,tw_only,arr_only"
  ONLY[sync_xw]="--only xw_arr,xw_tx"
fi
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  for p in sync_mb sync_bar sync_xw async; do
    [ -x "$B/$p" ] || { echo "SKIP $p (build failed)"; continue; }
    # shellcheck disable=SC2086
    timeout 1500 "$B/$p" --out "$OUT/$run" --reps "$REPS" --launches "$LAUNCHES" ${ONLY[$p]:-} > "$OUT/$run/$p.stdout" 2>&1 \
        || echo "PROBE_FAIL $run $p rc=$?"
  done
done
