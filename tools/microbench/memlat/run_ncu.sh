#!/usr/bin/env bash
# memlat probe: memory binary modes {l1,l2,dram} + ops binary constant-chain
# ops, all under ncu --cache-control all (per protocol; per-point the probe
# writes its own jsonl samples, ncu supplies the cold cache flush).
#   TREE=<clone> OUT=<run dir> SM_ARCH=... NCU=<ncu> bash run_ncu.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; NCU=${NCU:-$CUDA_HOME/bin/ncu}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
"$NCU" --version > "$OUT/ncu-version.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/memlat/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin"
run_cell() {  # name binary args...
  local name=$1; shift; local bin=$1; shift
  local run="$OUT/$name"; mkdir -p "$run"
  local command=("$NCU" --csv --page raw --log-file "$run/counters.csv"
    --metrics sm__cycles_active.max --replay-mode kernel
    --cache-control all --clock-control none --check-exit-code yes
    "$B/$bin" --out "$run/$bin.jsonl" "$@")
  printf '%q ' "${command[@]}" > "$run/command.txt"; echo >> "$run/command.txt"
  timeout 7200 "${command[@]}" > "$run/launches.log" 2>&1 || echo "CELL_FAIL $name rc=$?"
}
# protocol cells: l1 / l2 (near+far sampled via the probe's own warm pass)
# dram (cache-control all) / constant cache ops
run_cell l1  stage27_memlat_memory --mode l1  --samples 32 --warmup 4
run_cell l2  stage27_memlat_memory --mode l2  --samples 32 --warmup 4
run_cell dram stage27_memlat_memory --mode dram --samples 32 --warmup 4
for op in ldc sel isetp ldg lds; do
  run_cell "ops_$op" stage27_memlat_ops --op "$op" --depth 64 --rounds 4 --launches 8
done
