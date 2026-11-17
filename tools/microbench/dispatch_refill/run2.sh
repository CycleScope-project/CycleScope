#!/usr/bin/env bash
# dispatch_refill run 2 (protocol.json addendum 1): stage-2 spin cells + rms cells,
# then the ncu method check on rms_parts.  TREE=<clone> OUT=<run dir> SM_ARCH=... [NCU=..] [SKIP_BUILD=1] bash run2.sh
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
# <cell name> <probe args>
CELLS=(
  "spin.t128.o0.w6.h2200.j150|--mode spin --threads 128 --occ 0 --waves 6 --hold 2200 --jitter 150"
  "spin.t128.o0.w6.h4500.j150|--mode spin --threads 128 --occ 0 --waves 6 --hold 4500 --jitter 150"
  "spin.t128.o0.w6.h9000.j150|--mode spin --threads 128 --occ 0 --waves 6 --hold 9000 --jitter 150"
  "spin.t128.o0.w6.h4500.j0|--mode spin --threads 128 --occ 0 --waves 6 --hold 4500 --jitter 0"
  "rms.t128.g5612.h1400|--mode rms --threads 128 --occ 0 --grid 5612 --hold 1400"
  "rms.t128.g7552.h1400|--mode rms --threads 128 --occ 0 --grid 7552 --hold 1400"
  "rms.t128.g8192.h1400|--mode rms --threads 128 --occ 0 --grid 8192 --hold 1400"
  "rms.t128.g8192.h700|--mode rms --threads 128 --occ 0 --grid 8192 --hold 700"
  "rms.t128.g8192.h2800|--mode rms --threads 128 --occ 0 --grid 8192 --hold 2800"
)
for c in "${CELLS[@]}"; do
  name=${c%%|*}; args=${c#*|}
  # shellcheck disable=SC2086
  $B $args --launches 5 > "$OUT/cells/$name.txt" 2> "$OUT/cells/$name.err" \
      || echo "CELL_FAIL $name rc=$?"
done

NCU=${NCU:-$CUDA_HOME/bin/ncu}
if [ ! -x "$NCU" ]; then
  echo "NCU_MISSING $NCU"
else
  mkdir -p "$OUT/ncu"
  "$NCU" --version > "$OUT/ncu/version.txt" 2>&1 || true
  STALL="barrier branch_resolving dispatch_stall drain imc_miss lg_throttle long_scoreboard math_pipe_throttle membar mio_throttle misc no_instruction not_selected selected short_scoreboard sleeping tex_throttle wait"
  METRICS="sm__cycles_active.max,sm__cycles_active.avg,sm__cycles_active.min,smsp__inst_executed.sum,sm__ctas_launched.sum,dram__sectors_read.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_srcunit_ltcfabric.sum,l1tex__m_xbar2l1tex_read_sectors.sum,l1tex__m_l1tex2xbar_write_sectors.sum"
  for r in $STALL; do METRICS+=",smsp__average_warp_latency_issue_stalled_${r}.ratio"; done
  ncu_cell() { # <rms_parts> <name>
    local parts=$1 name=$2
    local m="$METRICS" i
    "$NCU" --csv --page raw --log-file "$OUT/ncu/$name.csv" --metrics "$m" \
        --replay-mode kernel --cache-control all --clock-control none \
        "$B" --mode rms --threads 128 --occ 0 --grid 8192 --hold 1400 \
        --rms-parts "$parts" --launches 3 \
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
        "$B" --mode rms --threads 128 --occ 0 --grid 8192 --hold 1400 \
        --rms-parts "$parts" --launches 3 \
        > "$OUT/ncu/$name.txt" 2> "$OUT/ncu/$name.err" || echo "NCU_FAIL $name rc=$?"
  }
  ncu_cell 3 rms.t128.g8192.h1400.p3
  ncu_cell 2 rms.t128.g8192.h1400.p2
  ncu_cell 1 rms.t128.g8192.h1400.p1
  ncu_cell 0 rms.t128.g8192.h1400.p0
fi
echo "run2.sh: dispatch_refill done"
