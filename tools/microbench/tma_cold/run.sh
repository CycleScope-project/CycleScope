#!/usr/bin/env bash
# tma_cold probes are clock64 probes: plain execution, no ncu (the cold
# temperature is produced by a 512 MiB tensor read once per rep).
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/tma_cold/build.sh" > "$OUT/build.log" 2>&1
fi
cat "$OUT/build/build.tsv" 2>/dev/null || true
# SASS check: dc1 kernels must contain UTMALDG.2D, and no .MULTICAST/.2CTA.
SASS="$OUT/build/sass/tma.sass"; CHK="$OUT/sass-check.txt"
if [ "$SM_ARCH" = sm_90a ] && [ -s "$SASS" ]; then
  python3 - "$SASS" > "$CHK" <<'PY'
import re, sys
sass = open(sys.argv[1]).read()
funcs = re.findall(r"Function : (\S+)", sass)
dc1 = [f for f in funcs if re.match(r"_Z5k_tmaILi1ELi\d+EE", f)]
bad = []
for f in dc1:
    body = sass.split(f"Function : {f}")[1].split("Function :")[0]
    if "UTMALDG" not in body or "UTMALDG.2D" not in body.replace(" ", ""):
        bad.append(f"{f}: no UTMALDG.2D")
    for tok in ("MULTICAST", ".2CTA"):
        if tok in body:
            bad.append(f"{f}: has {tok}")
print(f"sass-check: dc1 kernels={len(dc1)} " + ("OK" if not bad else "FAIL"))
for b in bad: print(" ", b)
PY
  cat "$CHK"
else
  echo "sass-check: skipped (arch=$SM_ARCH or no sass)" > "$CHK"
fi
B="$OUT/build/bin/tma"
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 3600 "$B" --out "$OUT/$run" --launches 16 --reps 33 > "$OUT/$run.stdout" 2>&1 \
      || echo "PROBE_FAIL $run rc=$?"
done
