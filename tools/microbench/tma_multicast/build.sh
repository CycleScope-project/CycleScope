#!/usr/bin/env bash
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=$CUDA_HOME/bin/cuobjdump
else NVCC=nvcc; CUOBJDUMP=cuobjdump; fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
    "$MECH_DIR/probe/tma_multicast.cu" -o "$OUT/bin/tma_multicast" -lcuda -ldl > "$OUT/sass/tma_multicast.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/tma_multicast" > "$OUT/sass/tma_multicast.sass" 2>/dev/null || true
# sass-counts: per kernel, UTMALDG.2D and UTMALDG.2D.MULTICAST tallies
python3 - "$OUT/sass/tma_multicast.sass" > "$OUT/sass-counts.txt" <<'PY'
import re, sys, collections
cur = None; c = collections.defaultdict(collections.Counter)
for l in open(sys.argv[1]):
    m = re.search(r'Function : (\S+)', l)
    if m: cur = m.group(1)
    for g in re.finditer(r'UTMALDG\.2D\.MULTICAST|UTMALDG\.2D', l):
        c[cur][g.group(0)] += 1
for k, v in c.items(): print(k, dict(v))
PY
echo "build.sh: tma_multicast done ($SM_ARCH)"
