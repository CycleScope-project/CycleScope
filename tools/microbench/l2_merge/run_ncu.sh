#!/usr/bin/env bash
# l2_merge protocol: two runs per device.  SET=m6r2 (default) runs the
# tma_scan set; SET=m1r4 (protocol v2) runs the ldg_scan lockstep set.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... [SET=m6r2|m1r4] [NCU=<ncu>] bash run_ncu.sh
# h100: launch.sh passes NCU=<external ncu>; b300: the image's ncu.
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}; SET=${SET:-m6r2}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; NCU=${NCU:-$CUDA_HOME/bin/ncu}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
"$NCU" --version > "$OUT/ncu-version.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/l2_merge/build.sh" > "$OUT/build.log" 2>&1
fi
if [ "$SET" = m1r4 ]; then
  B="$OUT/build/bin/ldg_scan"
else
  B="$OUT/build/bin/tma_scan"
fi
[ -x "$B" ] || { echo "SKIP $B (build failed)"; exit 1; }
if [ "$SET" = m1r4 ]; then
  # sass gate (protocol v2): a FAIL means the run is unrealised -- do not
  # spend GPU time
  first=$(head -1 "$OUT/build/sass_check.txt" 2>/dev/null || echo MISSING)
  echo "sass_check: $first"
  case "$first" in PASS*) ;; *) echo "ABORT: sass gate failed ($first)"; exit 1;; esac
fi
for run in run-a run-b; do
  d="$OUT/$run"; mkdir -p "$d"
  timeout 4800 python3 "$TREE/tools/microbench/l2_merge/run_scan.py" \
      --bin "$B" --out "$d" --trials 5 --ncu "$NCU" --set "$SET" \
      > "$d/run_scan.log" 2>&1 || echo "RUN_FAIL $run rc=$?"
done
