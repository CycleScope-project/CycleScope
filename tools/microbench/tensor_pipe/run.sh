#!/usr/bin/env bash
# tensor_pipe probes are clock64 probes: plain execution, no ncu.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/tensor_pipe/build.sh" > "$OUT/build.log" 2>&1
fi
if [ "$SM_ARCH" = sm_103a ]; then PROBES="${UMMA_PROBES:-umma umma_cadence}";
elif [ "${WGMMA_SP:-0}" = 1 ]; then PROBES="${WGMMA_PROBES:-wgmma_spf16 wgmma_spbf16 wgmma_spe4m3e4m3}";
else PROBES="${WGMMA_PROBES:-wgmma_f16 wgmma_f16a wgmma_bf16 wgmma_tf32 wgmma_e4m3e4m3 wgmma_e4m3e5m2 wgmma_e5m2e5m2 wgmma_s8s8}"; fi
for p in $PROBES; do
  B="$OUT/build/bin/$p"
  [ -x "$B" ] || { echo "SKIP $p (build failed)"; exit 1; }
  for run in run-a run-b; do
    mkdir -p "$OUT/$run"
    timeout 3600 "$B" --out "$OUT/$run" --launches 16 --reps 257 > "$OUT/$run.$p.stdout" 2>&1 \
        || echo "PROBE_FAIL $p $run rc=$?"
  done
done
