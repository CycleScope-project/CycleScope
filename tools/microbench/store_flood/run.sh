#!/usr/bin/env bash
set -euo pipefail
L=$1; OUT=$2; SM_ARCH=$3
A=${CYCLESCOPE_ARTIFACT_ROOT:?}
mkdir -p "$OUT/build" "$OUT/tmp" "$OUT/run-a" "$OUT/run-b"
export TMPDIR="$OUT/tmp"
if [ "${V:-1}" = "2" ]; then
  trap 'rc=$?; printf "STORE_FLOOD2_RC=%s\n" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
elif [ "${V:-1}" = "3" ]; then
  trap 'rc=$?; printf "STORE_FLOOD3_RC=%s\n" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
elif [ "$V" = "bridge" ]; then
  trap 'rc=$?; printf "STORE_FLOOD_BRIDGE_RC=%s\n" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
elif [ "${V:-1}" = "4" ]; then
  trap 'rc=$?; printf "STORE_FLOOD4_RC=%s\n" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
elif [ "${V:-1}" = "5" ]; then
  trap 'rc=$?; printf "STORE_FLOOD5_RC=%s\n" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
else
  trap 'rc=$?; printf "STORE_FLOOD_RC=%s\n" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
fi
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
export PATH="$A/tools/cuda-toolkit-13.2/bin:/usr/local/cuda/bin:$PATH"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr -I"$L/tools/microbench" "$L/tools/microbench/store_flood/probe.cu" -o "$OUT/build/store_flood" > "$OUT/build.log" 2>&1
cuobjdump -sass "$OUT/build/store_flood" > "$OUT/build/store_flood.sass"
FOCUS=""
[ "${V:-1}" = "4" ] && FOCUS='bisect|geo2|^4$'
[ "${V:-1}" = "5" ] && FOCUS='v5k'
python3 "$L/tools/microbench/store_flood/check.py" "$OUT/build/store_flood.sass" "$OUT/build/check.json" $FOCUS
if [ "${V:-1}" = "2" ]; then
  DEV=h100; [ "$SM_ARCH" = "sm_103a" ] && DEV=b300
  SETS="$OUT/sets"; mkdir -p "$SETS"
  python3 "$L/tools/microbench/tma_store_loaded/scope_sets.py" "$L/configs/hardware/$DEV/sm_topology.json" "$SETS" > "$SETS/sets.log" 2>&1
  echo "0" > "$SETS/sm1.txt"
  for run in a b; do
    R="$OUT/run-$run"; mkdir -p "$R/cells"
    for set in sm1 g1k4 g1k8 g1kall ek1 ek4 ekall; do
      mkdir -p "$R/cells/set-$set"
      "$OUT/build/store_flood" --out "$R/cells/set-$set" --active-file "$SETS/$set.txt" --set-name "$set"         --warps 4 --geos full16,line32,row4 --launches 8 --reps 17 > "$R/cells/$set.log" 2>&1
    done
    for set in sm1 ekall; do
      mkdir -p "$R/cells/none-$set"
      "$OUT/build/store_flood" --out "$R/cells/none-$set" --active-file "$SETS/$set.txt" --set-name "$set"         --warps 4 --geos none --launches 8 --reps 17 > "$R/cells/none-$set.log" 2>&1
    done
    for W in 1 4 8; do
      mkdir -p "$R/cells/single-$W"
      "$OUT/build/store_flood" --out "$R/cells/single-$W" --set-name sm1 --warps "$W"         --geos row4,row4_refill --launches 8 --reps 17 > "$R/cells/single-$W.log" 2>&1
    done
    cat "$R/cells"/*/store_flood.jsonl > "$R/store_flood.jsonl"
  done
  python3 "$L/tools/microbench/store_flood/table_v2.py" "$OUT" > "$OUT/table.txt"
elif [ "${V:-1}" = "3" ]; then
  for run in a b; do
    R="$OUT/run-$run"; mkdir -p "$R/cells"
    for W in 4 8; do
      for d in warm virgin; do
        mkdir -p "$R/cells/$d-$W"
        "$OUT/build/store_flood" --out "$R/cells/$d-$W" --set-name sm1 --dest "$d"           --warps "$W" --geos row4,epi,lace,epi_seq --launches 8 --reps 17           > "$R/cells/$d-$W.log" 2>&1
      done
    done
    mkdir -p "$R/cells/none"
    "$OUT/build/store_flood" --out "$R/cells/none" --set-name sm1 --dest warm       --warps 4 --geos none --launches 8 --reps 17 > "$R/cells/none.log" 2>&1
    cat "$R/cells"/*/store_flood.jsonl > "$R/store_flood.jsonl"
  done
  mkdir -p "$OUT/addr"
  "$OUT/build/store_flood" --out "$OUT/addr" --dump-addr row4,lace,epi,epi_seq     --geos none --only zzz --launches 1 --reps 1 > "$OUT/addr/dump.log" 2>&1
  python3 "$L/tools/microbench/store_flood/table_v3.py" "$OUT" > "$OUT/table.txt"
elif [ "$V" = "bridge" ]; then
  for run in a b; do
    R="$OUT/run-$run"; mkdir -p "$R/cells"
    for b in none filler lanedata once short all; do
      mkdir -p "$R/cells/$b"
      "$OUT/build/store_flood" --out "$R/cells/$b" --bridge "$b" > "$R/cells/$b.log" 2>&1
    done
    cat "$R/cells"/*/store_flood.jsonl > "$R/store_flood.jsonl"
  done
  if command -v ncu >/dev/null 2>&1; then
    for b in none all; do
      mkdir -p "$OUT/ncu-$b"
      ncu --cache-control all --clock-control none --metrics sm__cycles_active.max -- \
          "$OUT/build/store_flood" --out "$OUT/ncu-$b" --bridge "$b" \
          > "$OUT/ncu-$b/ncu.txt" 2>&1 || echo "NCU_FAIL $b" >> "$OUT/ncu-fail.txt"
    done
  else
    echo "NCU_FAIL ncu missing" > "$OUT/ncu-fail.txt"
  fi
  python3 "$L/tools/microbench/store_flood/table_bridge.py" "$OUT" > "$OUT/table.txt"
elif [ "${V:-1}" = "4" ]; then
  # whole-kernel SASS of the two anchors and MASK 0 / 31 for the lead's
  # out-of-window diff
  for k in 'k_floodILi4E' 'k_floodxILi2E' 'k_bisect_nILi0E' 'k_bisect_bILi31E'; do
    python3 - "$OUT/build/store_flood.sass" "$OUT/build/sass-$k.txt" "$k" <<'EOF'
import re, sys
sass = open(sys.argv[1]).read()
i = sass.find(sys.argv[3])
j = sass.find("Function :", i + 10)
open(sys.argv[2], "w").write(sass[i:j if j > i else len(sass)])
EOF
  done
  echo "0" > "$OUT/sm1.txt"
  for run in a b; do
    R="$OUT/run-$run"; mkdir -p "$R/cells"
    mkdir -p "$R/cells/bisect" "$R/cells/v1" "$R/cells/v2"
    "$OUT/build/store_flood" --out "$R/cells/bisect" --bisect         --launches 8 --reps 17 > "$R/cells/bisect.log" 2>&1
    "$OUT/build/store_flood" --out "$R/cells/v1"         --only w4.s128.l32.x4,none.x --launches 8 --reps 17         > "$R/cells/v1.log" 2>&1
    "$OUT/build/store_flood" --out "$R/cells/v2"         --active-file "$OUT/sm1.txt" --set-name sm1 --warps 4         --geos line32,none --launches 8 --reps 17         > "$R/cells/v2.log" 2>&1
    cat "$R/cells"/*/store_flood.jsonl > "$R/store_flood.jsonl"
  done
  python3 "$L/tools/microbench/store_flood/table_v4.py" "$OUT" > "$OUT/table.txt"
elif [ "${V:-1}" = "5" ]; then
  for run in a b; do
    R="$OUT/run-$run"; mkdir -p "$R/cells"
    for s in 0 16384 49152 98304 163840 max; do
      for g in 0 all; do
        mkdir -p "$R/cells/s$s-g$g"
        "$OUT/build/store_flood" --v5 --smem "$s" --siblings "$g" \
            --carveout default --out "$R/cells/s$s-g$g" \
            --launches 8 --reps 17 > "$R/cells/s$s-g$g.log" 2>&1
      done
    done
    for s in 0 49152; do
      for c in 0 100; do
        mkdir -p "$R/cells/s$s-c$c"
        "$OUT/build/store_flood" --v5 --smem "$s" --siblings 0 \
            --carveout "$c" --out "$R/cells/s$s-c$c" \
            --launches 8 --reps 17 > "$R/cells/s$s-c$c.log" 2>&1
      done
    done
    cat "$R/cells"/*/store_flood.jsonl > "$R/store_flood.jsonl"
  done
  python3 "$L/tools/microbench/store_flood/table_v5.py" "$OUT" > "$OUT/table.txt"
else
  for run in a b; do
    "$OUT/build/store_flood" --out "$OUT/run-$run" --launches 8 --reps 17 > "$OUT/run-$run.log" 2>&1
  done
  python3 "$L/tools/microbench/store_flood/model.py" "$OUT" "$OUT/measurement.json" > "$OUT/measurement.txt"
fi
