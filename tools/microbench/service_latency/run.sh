#!/usr/bin/env bash
# service_latency probes are clock64 probes: plain execution, no ncu.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/service_latency/build.sh" > "$OUT/build.log" 2>&1
fi
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  for src in alu alu2 misc svc_p0 svc_p1 svc_p2 svc_p3 alu3 alu4 alu5 alu6 alu7; do
    B="$OUT/build/bin/$src"
    [ -x "$B" ] || { echo "SKIP $src (build failed)"; continue; }
    if [ "$src" = alu7 ] && [ "$SM_ARCH" = sm_90a ]; then
      # v6: h100 runs only the MATCH d32 cell set
      timeout 3600 "$B" --out "$OUT/$run" --launches 8 --reps 33 --only thr.MATCH_d32 > "$OUT/$run.$src.stdout" 2>&1           || echo "PROBE_FAIL $run $src rc=$?"
    else
      timeout 3600 "$B" --out "$OUT/$run" --launches 8 --reps 33 > "$OUT/$run.$src.stdout" 2>&1           || echo "PROBE_FAIL $run $src rc=$?"
    fi
  done
done
