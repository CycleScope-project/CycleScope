#!/usr/bin/env bash
# ldgsts_l1 cells (protocol.json), runs a and b.  TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT/build"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
M="$TREE/tools/microbench/ldgsts_l1"
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" --expt-relaxed-constexpr \
    "$M/probe/ldgsts_l1.cu" -o "$OUT/build/ldgsts_l1" > "$OUT/build.log" 2>&1
"$CUDA_HOME/bin/cuobjdump" -sass "$OUT/build/ldgsts_l1" > "$OUT/build/ldgsts_l1.sass" 2>/dev/null || true
# SASS gate: the ca kernel holds LDGSTS without BYPASS, the cg kernel LDGSTS with BYPASS
python3 - "$OUT/build/ldgsts_l1.sass" > "$OUT/build/sass_check.txt" <<'P' || { echo "SASS GATE FAILED"; exit 3; }
import re, sys
ok = True
for b in re.split(r"\n\s*Function : ", open(sys.argv[1], errors="replace").read())[1:]:
    m = re.search(r"k_l1ILi(\d)E", b.split("\n", 1)[0])
    if not m: continue
    mode = int(m.group(1)); ops = re.findall(r"\b(LDGSTS\S*|LDG\.\S*)", b)
    byp = sum("BYPASS" in o for o in ops if o.startswith("LDGSTS")); nob = sum(o.startswith("LDGSTS") and "BYPASS" not in o for o in ops)
    good = {0: byp + nob == 0, 1: nob >= 1 and byp == 0, 2: byp >= 1 and nob == 0, 3: byp + nob == 0}[mode]
    print("OK " if good else "BAD", "mode", mode, sorted(set(ops)))
    ok = ok and good
sys.exit(0 if ok else 1)
P
for run in a b; do "$OUT/build/ldgsts_l1" --out "$OUT/run-$run.jsonl"; done
echo "run.sh: ldgsts_l1 done"
