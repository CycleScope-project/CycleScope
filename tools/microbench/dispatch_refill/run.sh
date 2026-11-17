#!/usr/bin/env bash
# dispatch_refill cells (protocol.json): the same list as CELLS in model.py, then the
# ncu method check on two cells.  TREE=<clone> OUT=<run dir> SM_ARCH=... [NCU=..] [SKIP_BUILD=1] bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/dispatch_refill/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/refill"

mkdir -p "$OUT/cells"
# <mode> <threads> <occ> <waves> <hold> <jitter> <cell name> -- same order as CELLS in model.py
CELLS=(
  "spin 128 0 4 0 0 spin.t128.o0.w4.h0.j0"
  "spin 128 0 12 0 0 spin.t128.o0.w12.h0.j0"
  "spin 128 0 4 800 0 spin.t128.o0.w4.h100.j0"
  "spin 128 0 12 800 0 spin.t128.o0.w12.h100.j0"
  "spin 128 0 4 1500 0 spin.t128.o0.w4.h1500.j0"
  "spin 128 0 12 1500 0 spin.t128.o0.w12.h1500.j0"
  "spin 128 0 4 3000 0 spin.t128.o0.w4.h3000.j0"
  "spin 128 0 12 3000 0 spin.t128.o0.w12.h3000.j0"
  "spin 128 0 4 6000 0 spin.t128.o0.w4.h6000.j0"
  "spin 128 0 12 6000 0 spin.t128.o0.w12.h6000.j0"
  "spin 128 0 4 12000 0 spin.t128.o0.w4.h12000.j0"
  "spin 128 0 12 12000 0 spin.t128.o0.w12.h12000.j0"
  "spin 128 0 4 3000 300 spin.t128.o0.w4.h3000.j300"
  "spin 128 0 12 3000 300 spin.t128.o0.w12.h3000.j300"
  "spin 128 0 12 6000 300 spin.t128.o0.w12.h6000.j300"
  "spin 128 4 12 3000 0 spin.t128.o4.w12.h3000.j0"
  "spin 128 8 12 3000 0 spin.t128.o8.w12.h3000.j0"
  "spin 256 0 12 3000 0 spin.t256.o0.w12.h3000.j0"
  "spin 512 0 12 3000 0 spin.t512.o0.w12.h3000.j0"
  "spin 32 0 12 3000 0 spin.t32.o0.w12.h3000.j0"
  "sleep 128 0 4 3000 0 sleep.t128.o0.w4.h3000.j0"
  "sleep 128 0 12 3000 0 sleep.t128.o0.w12.h3000.j0"
  "ldg 128 0 4 3000 0 ldg.t128.o0.w4.h3000.j0"
  "ldg 128 0 12 3000 0 ldg.t128.o0.w12.h3000.j0"
  "cpasync 128 0 4 3000 0 cpasync.t128.o0.w4.h3000.j0"
  "cpasync 128 0 12 3000 0 cpasync.t128.o0.w12.h3000.j0"
  "ldg 128 0 4 0 0 ldg.t128.o0.w4.h0.j0"
  "cpasync 128 0 4 0 0 cpasync.t128.o0.w4.h0.j0"
)
for c in "${CELLS[@]}"; do
  read -r mode threads occ waves hold jitter name <<< "$c"
  "$B" --mode "$mode" --threads "$threads" --occ "$occ" --waves "$waves" \
       --hold "$hold" --jitter "$jitter" --launches 5 \
       > "$OUT/cells/$name.txt" 2> "$OUT/cells/$name.err" || echo "CELL_FAIL $name rc=$?"
done

NCU=${NCU:-$CUDA_HOME/bin/ncu}
if [ ! -x "$NCU" ]; then
  echo "NCU_MISSING $NCU"
else
  mkdir -p "$OUT/ncu"
  "$NCU" --version > "$OUT/ncu/version.txt" 2>&1 || true
  STALL="barrier branch_resolving dispatch_stall drain imc_miss lg_throttle long_scoreboard math_pipe_throttle membar mio_throttle misc no_instruction not_selected selected short_scoreboard sleeping tex_throttle wait"
  METRICS="sm__cycles_active.max,sm__cycles_active.avg,sm__cycles_active.min,smsp__inst_executed.sum,sm__ctas_launched.sum"
  for r in $STALL; do METRICS+=",smsp__average_warp_latency_issue_stalled_${r}.ratio"; done
  ncu_cell() { # <mode> <threads> <occ> <waves> <hold> <jitter> <name>
    local mode=$1 threads=$2 occ=$3 waves=$4 hold=$5 jitter=$6 name=$7
    local m="$METRICS" i
    "$NCU" --csv --page raw --log-file "$OUT/ncu/$name.csv" --metrics "$m" \
        --replay-mode kernel --cache-control all --clock-control none \
        "$B" --mode "$mode" --threads "$threads" --occ "$occ" --waves "$waves" \
        --hold "$hold" --jitter "$jitter" --launches 3 \
        > "$OUT/ncu/$name.txt" 2> "$OUT/ncu/$name.err" && return 0
    # one retry: drop metric names ncu rejected on this device
    local dropped=""
    for i in $(grep -oE '[a-z_]+__[a-z_]+(\.[a-z_]+)*' "$OUT/ncu/$name.err" | sort -u); do
      if [[ ",$m," == *",$i,"* ]]; then dropped+="$i"$'\n'; m=$(echo "$m" | sed "s/\(^\|,\)$i\(,\|$\)/\1\2/g;s/,,*/,/g;s/^,//;s/,\$//"); fi
    done
    [ -n "$dropped" ] || { echo "NCU_FAIL $name rc=1"; return 0; }
    printf "%s: dropped %s" "$name" "$dropped" >> "$OUT/ncu/dropped.txt"
    "$NCU" --csv --page raw --log-file "$OUT/ncu/$name.csv" --metrics "$m" \
        --replay-mode kernel --cache-control all --clock-control none \
        "$B" --mode "$mode" --threads "$threads" --occ "$occ" --waves "$waves" \
        --hold "$hold" --jitter "$jitter" --launches 3 \
        > "$OUT/ncu/$name.txt" 2> "$OUT/ncu/$name.err" || echo "NCU_FAIL $name rc=$?"
  }
  ncu_cell spin 128 0 4 3000 0 spin.t128.o0.w4.h3000.j0
  ncu_cell cpasync 128 0 4 3000 0 cpasync.t128.o0.w4.h3000.j0
fi
echo "run.sh: dispatch_refill done"
