#!/usr/bin/env bash
# run.sh <repo> <out dir> <sm_arch>
L=$1; OUT=$2; SM_ARCH=${3:-${SM_ARCH:-sm_90a}}
A=${CYCLESCOPE_ARTIFACT_ROOT:?}
mkdir -p "$OUT/build" "$OUT/tmp"
export TMPDIR="$OUT/tmp"
trap 'rc=$?; printf "ARRIVE_STATUS_RC=%s\n" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
export PATH="$A/tools/cuda-toolkit-13.2/bin:/usr/local/cuda/bin:$PATH"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr "$L/tools/microbench/arrive_status/probe/arrive_status.cu" -o "$OUT/build/arrive_status" > "$OUT/build.log" 2>&1
cuobjdump -sass "$OUT/build/arrive_status" > "$OUT/build/arrive_status.sass"
if [ "${V:-1}" = 2 ]; then
  trap 'rc=$?; printf "ARRIVE_STATUS2_RC=%s\n" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
  python3 "$L/tools/microbench/arrive_status/check.py" \
    "$OUT/build/arrive_status.sass" "$OUT/build/check.json" v2
  for run in a b; do
    "$OUT/build/arrive_status" --v2 --out "$OUT/run-$run.jsonl" \
        > "$OUT/run-$run.log" 2>&1
    python3 - "$OUT" "$OUT/run-$run.jsonl" <<'PY'
import json, sys
out, jf = sys.argv[1], sys.argv[2]
smap = json.loads(open(out + "/build/check.json.sass-map.json").read())
lines = []
for line in open(jf):
    r = json.loads(line)
    p = r["variant"].split(".")
    r["sass"] = smap.get(f"{p[2]}.{p[5]}")
    lines.append(json.dumps(r))
open(jf, "w").write("\n".join(lines) + "\n")
PY
  done
  python3 "$L/tools/microbench/arrive_status/table_v2.py" "$OUT" \
      > "$OUT/table.txt"
  exit 0
fi
python3 "$L/tools/microbench/arrive_status/check.py" \
  "$OUT/build/arrive_status.sass" "$OUT/build/check.json"
for run in a b; do
  "$OUT/build/arrive_status" --out "$OUT/run-$run.jsonl" \
      > "$OUT/run-$run.log" 2>&1
done
python3 "$L/tools/microbench/arrive_status/table.py" "$OUT" \
    > "$OUT/table.txt"
