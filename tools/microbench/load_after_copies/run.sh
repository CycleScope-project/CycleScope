#!/usr/bin/env bash
# load_after_copies cells (protocol.json), runs a and b.
# Usage: run.sh <repo> <out> <sm_arch>
set -euo pipefail
L=$1; OUT=$2; SM_ARCH=${3:-${SM_ARCH:-sm_90a}}
A=${CYCLESCOPE_ARTIFACT_ROOT:?}
mkdir -p "$OUT/build" "$OUT/tmp"
export TMPDIR="$OUT/tmp"
V=${V:-1}
trap 'rc=$?; printf "LOAD_AFTER_COPIES%s_RC=%s\n" "$([ "$V" = 2 ] && echo 2; [ "$V" = 3 ] && echo 3)" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
export PATH="$A/tools/cuda-toolkit-13.2/bin:/usr/local/cuda/bin:$PATH"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr "$L/tools/microbench/load_after_copies/probe/load_after_copies.cu" -o "$OUT/build/load_after_copies" > "$OUT/build.log" 2>&1
cuobjdump -sass "$OUT/build/load_after_copies" > "$OUT/build/load_after_copies.sass"
python3 "$L/tools/microbench/load_after_copies/check.py" \
  "$OUT/build/load_after_copies.sass" "$OUT/build/check.json"
EXTRA=""
[ "$V" = 2 ] && EXTRA="--v2"
[ "$V" = 3 ] && EXTRA="--v3"
for run in a b; do
  "$OUT/build/load_after_copies" --out "$OUT/run-$run.jsonl" $EXTRA \
      > "$OUT/run-$run.log" 2>&1
done
if [ "$V" = 3 ]; then
python3 "$L/tools/microbench/load_after_copies/table_v3.py" "$OUT" \
    > "$OUT/table.txt"
elif [ "$V" = 2 ]; then
python3 "$L/tools/microbench/load_after_copies/table_v2.py" "$OUT" \
    > "$OUT/table.txt"
else
python3 "$L/tools/microbench/load_after_copies/table.py" "$OUT" \
    > "$OUT/table.txt"
fi
