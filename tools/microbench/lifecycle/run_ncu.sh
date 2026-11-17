#!/usr/bin/env bash
# lifecycle probe under ncu, cache-control all only (protocol installs `all`).
#   TREE=<clone> OUT=<run dir> SM_ARCH=... NCU=<ncu> bash run_ncu.sh
# Protocol v2 (protocol-v2.json): 64 launches per variant (model drops the
# first 2), two runs per device: $OUT/{run-a,run-b}/{counters.csv,launches.log}.
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; NCU=${NCU:-$CUDA_HOME/bin/ncu}
LAUNCHES=${LAUNCHES:-64}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
"$NCU" --version > "$OUT/ncu-version.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/lifecycle/build.sh" > "$OUT/build.log" 2>&1
fi
for run in run-a run-b; do
  d="$OUT/$run"; mkdir -p "$d"
  command=("$NCU" --csv --page raw --log-file "$d/counters.csv"
    --metrics sm__cycles_active.max,sm__cycles_elapsed.max,smsp__inst_executed.sum
    --kernel-name-base mangled --kernel-name 'regex:k_life'
    --replay-mode kernel --cache-control all --clock-control none --check-exit-code yes
    "$OUT/build/bin/lifecyc" --out "$d" --launches "$LAUNCHES" --reps 1)
  printf '%q ' "${command[@]}" > "$d/command.txt"; echo >> "$d/command.txt"
  timeout 3000 "${command[@]}" > "$d/launches.log" 2>&1 || echo "RUN_FAIL $run rc=$?"
  test -s "$d/counters.csv" || echo "NO_COUNTERS $run"
done
