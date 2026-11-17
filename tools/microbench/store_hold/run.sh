#!/usr/bin/env bash
set -euo pipefail
L=$1; OUT=$2; SM_ARCH=${3:-${SM_ARCH:-sm_90a}}
V=${V:-1}
A=${CYCLESCOPE_ARTIFACT_ROOT:?}
mkdir -p "$OUT/build" "$OUT/tmp" "$OUT/run-a" "$OUT/run-b"
export TMPDIR="$OUT/tmp"
trap 'rc=$?; printf "STORE_HOLD%s_RC=%s\n" "$([ "$V" = 2 ] && echo 2)" "$rc"; printf "%s\n" "$rc" > "$OUT/exit.txt"; sync' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
export PATH="$A/tools/cuda-toolkit-13.2/bin:/usr/local/cuda/bin:$PATH"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr -I"$L/tools/microbench" "$L/tools/microbench/store_hold/probe/store_hold.cu" -o "$OUT/build/store_hold" > "$OUT/build.log" 2>&1
cuobjdump -sass "$OUT/build/store_hold" > "$OUT/build/store_hold.sass"
if [ "$V" = 2 ]; then
python3 "$L/tools/microbench/store_hold/check.py" \
  "$OUT/build/store_hold.sass" "$OUT/build/check.json" \
  "$OUT/build/sass-loops.txt" --v2 || true
# keep only the kernels whose loop body satisfies the register-identity
# gate; the kept kernel name doubles as a --only substring (it matches all
# four N cells of that kernel).  Every loop.none.* control must be kept.
ONLY=$(python3 - "$OUT/build/check.json" "$OUT/dropped.txt" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
keep, drop = [], []
for r in j["kernels"]:
    (keep if r["ok"] else drop).append(r)
with open(sys.argv[2], "w") as f:
    for r in drop:
        f.write(f'{r["variant"]}: {"; ".join(r["fails"])}\n')
ctrl_bad = [r["variant"] for r in drop
            if r["variant"].startswith("loop.none.")]
if ctrl_bad:
    sys.exit("control cells failed SASS: " + ",".join(ctrl_bad))
if not keep:
    sys.exit("no cell kept")
print(",".join(r["variant"] for r in keep))
PY
)
echo "kept kernels: $(echo "$ONLY" | tr ',' '\n' | wc -l)"; wc -l < "$OUT/dropped.txt"
for run in a b; do
  SH_V2=1 "$OUT/build/store_hold" --out "$OUT/run-$run" --launches 8 --reps 33 --only "$ONLY" > "$OUT/run-$run.log" 2>&1
done
python3 "$L/tools/microbench/store_hold/table_v2.py" "$OUT" > "$OUT/table.txt"
exit 0
fi
python3 "$L/tools/microbench/store_hold/check.py" "$OUT/build/store_hold.sass" "$OUT/build/check.json" "$OUT/build/sass-cells.txt" || true
# keep only the cells whose SASS satisfies the register-identity gate;
# every none.* control must be present or the run is aborted
ONLY=$(python3 - "$OUT/build/check.json" "$OUT/dropped.txt" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
keep, drop = [], []
for r in j["kernels"]:
    (keep if r["ok"] else drop).append(r)
with open(sys.argv[2], "w") as f:
    for r in drop:
        f.write(f'{r["variant"]}: {"; ".join(r["fails"])}\n')
ctrl_bad = [r["variant"] for r in drop
            if r["variant"].startswith("none.")]
if ctrl_bad:
    sys.exit("control cells failed SASS: " + ",".join(ctrl_bad))
if not keep:
    sys.exit("no cell kept")
print(",".join("=" + r["variant"] for r in keep))
PY
)
echo "kept cells: $(echo "$ONLY" | tr ',' '\n' | wc -l)"; wc -l < "$OUT/dropped.txt"
for run in a b; do
  "$OUT/build/store_hold" --out "$OUT/run-$run" --launches 8 --reps 33 --only "$ONLY" > "$OUT/run-$run.log" 2>&1
done
python3 "$L/tools/microbench/store_hold/table.py" "$OUT" > "$OUT/table.txt"
