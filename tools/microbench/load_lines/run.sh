#!/usr/bin/env bash
# load_lines cells (protocol.json), runs a and b.  TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT/build"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" --expt-relaxed-constexpr \
    "$TREE/tools/microbench/load_lines/probe/load_lines.cu" -o "$OUT/build/load_lines" > "$OUT/build.log" 2>&1
"$CUDA_HOME/bin/cuobjdump" -sass "$OUT/build/load_lines" > "$OUT/build/load_lines.sass" 2>/dev/null || true
# SASS gate: the local kernel holds LDL (stack loads), the global kernel LDG and no LDL
python3 - "$OUT/build/load_lines.sass" > "$OUT/build/sass_check.txt" <<'P' || { echo "SASS GATE FAILED"; exit 3; }
import re, sys
ok = True; n = 0
for b in re.split(r"\n\s*Function : ", open(sys.argv[1], errors="replace").read())[1:]:
    m = re.search(r"k_linesILi(\d)E", b.split("\n", 1)[0])
    if not m: continue
    n += 1; fam = int(m.group(1))
    ldl = len(re.findall(r"\bLDL\b", b)); ldg = len(re.findall(r"\bLDG\.", b))
    good = (ldl >= 8 and fam == 1) or (ldl == 0 and ldg >= 8 and fam == 0)
    print("OK " if good else "BAD", "fam", fam, "LDL", ldl, "LDG", ldg, sorted(set(re.findall(r"\b(LDL\S*|STL\S*)", b)))[:6])
    ok = ok and good
sys.exit(0 if ok and n == 2 else 1)
P
for run in a b; do "$OUT/build/load_lines" --out "$OUT/run-$run.jsonl"; done
echo "run.sh: load_lines done"
