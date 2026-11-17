#!/usr/bin/env bash
# constant_launch protocol v2 probe under ncu (same runner shape as v1).
#   TREE=<clone> OUT=<run dir> SM_ARCH=... NCU=<ncu> bash run_ncu_v2.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; NCU=${NCU:-$CUDA_HOME/bin/ncu}
LAUNCHES=${LAUNCHES:-34}
METRICS="launch__shared_mem_config_size,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__grid_dim_x,launch__grid_dim_y,launch__grid_dim_z,launch__block_dim_x,launch__block_dim_y,launch__block_dim_z,launch__cluster_dim_x,launch__cluster_dim_y,launch__cluster_dim_z,sm__cycles_active.max,sm__cycles_active.sum,sm__ctas_launched.sum,sm__cycles_elapsed.avg.per_second"
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
"$NCU" --version > "$OUT/ncu-version.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" PROBE=constlaunch_v2 \
      bash "$TREE/tools/microbench/constant_launch/build.sh" > "$OUT/build.log" 2>&1
fi
PYTHONPATH="${PYTHONPATH:-$TREE/python}" python3 "$TREE/tools/microbench/constant_launch/sass_check_v2.py" \
    --sass "$OUT/build/sass/constlaunch_v2.sass" --out "$OUT/checks.json" || echo "SASS_CHECK_FAIL"
for run in run-a run-b; do
  d="$OUT/$run"; mkdir -p "$d"
  command=("$NCU" --csv --page raw --log-file "$d/counters.csv"
    --print-units base
    --metrics "$METRICS"
    --kernel-name-base mangled --kernel-name 'regex:k_cl'
    --replay-mode kernel --cache-control all --clock-control none --check-exit-code yes
    "$OUT/build/bin/constlaunch_v2" --out "$d" --launches "$LAUNCHES" --reps 1)
  printf '%q ' "${command[@]}" > "$d/command.txt"; echo >> "$d/command.txt"
  timeout 3600 "${command[@]}" > "$d/launches.log" 2>&1 || echo "RUN_FAIL $run rc=$?"
  test -s "$d/counters.csv" || echo "NO_COUNTERS $run"
  timeout 120 "$OUT/build/bin/constlaunch_v2" --out "$d" --launches "$LAUNCHES" --reps 1 \
      > "$d/plain.log" 2>&1 || echo "PLAIN_FAIL $run rc=$?"
done
