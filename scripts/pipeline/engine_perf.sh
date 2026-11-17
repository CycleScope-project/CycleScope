#!/usr/bin/env bash
# Engine perf harness: build the engine, generate synthetic jobs,
# time `cyclescope-engine simulate`, optionally gprof the L run.
#
#   OUT=... [SIZES="S L"] [PROFILE=1] bash scripts/pipeline/engine_perf.sh
#
# S = 16 SM x 16 warps x 1000 insts (256k instances, seconds on a
# workstation); L = 132 SM x 32 warps x 2000 insts (8.4M, cluster only).
# Outputs: $OUT/perf.tsv, $OUT/gprof-<size>-<kind>.txt, prints PERF_RC.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PP="$ROOT/python:$ROOT/tests/data/job"
OUT="${OUT:?set OUT}"
SIZES="${SIZES:-S L}"
PROFILE="${PROFILE:-0}"
mkdir -p "$OUT/jobs"

cmake -S "$ROOT" -B "$OUT/build-release" -DCMAKE_BUILD_TYPE=Release \
      >/dev/null && cmake --build "$OUT/build-release" -j8 >/dev/null || {
    echo "build failed"; exit 1; }
if [ "$PROFILE" = "1" ]; then
    cmake -S "$ROOT" -B "$OUT/build-pg" -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg >/dev/null \
        && cmake --build "$OUT/build-pg" -j8 >/dev/null
fi
ENGINE="$OUT/build-release/cyclescope-engine"
PGENGINE="$OUT/build-pg/cyclescope-engine"

declare -A PARAMS=(
    [S]="--sms 16 --warps-per-sm 16 --insts 1000"
    [L]="--sms 132 --warps-per-sm 32 --insts 2000")
# the sleepers kind overrides --insts (serial ALU chain)
declare -A SLEEP_INSTS=([S]=2000 [L]=20000)

rc=0
: > "$OUT/perf.tsv"
for size in $SIZES; do
  for kind in alu memory sleepers; do
    job="$OUT/jobs/$size-$kind.bin"
    if [ ! -f "$job" ]; then
      mflag=""
      params=${PARAMS[$size]}
      if [ "$kind" = memory ]; then mflag="--memory"
      elif [ "$kind" = sleepers ]; then
        mflag="--sleepers"
        params="--sms $(echo $params | grep -oP '(?<=--sms )\d+')"
        params="$params --warps-per-sm $(echo ${PARAMS[$size]} | grep -oP '(?<=--warps-per-sm )\d+') --insts ${SLEEP_INSTS[$size]}"
      fi
      PYTHONPATH="$PP" python3 "$ROOT/tests/data/job/make_perf_job.py" \
          $params $mflag --out "$job" || {
          echo "gen $size-$kind failed"; rc=1; continue; }
    fi
    python3 - "$ENGINE" "$job" "$size" "$kind" <<'EOF' >> "$OUT/perf.tsv"
import resource, subprocess, sys, time
eng, job, size, kind = sys.argv[1:5]
t0 = time.time()
out = subprocess.run([eng, "simulate", "--job", job, "--out", "/dev/null"],
                     capture_output=True, text=True)
wall = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
stats = out.stdout.strip().splitlines()[-1] if out.stdout else ""
run_s = wall
for line in out.stdout.splitlines():
    if "RUN=" in line:
        run_s = float(line.split("RUN=")[1].split()[0])
cycles = 0.0
for tok in stats.split():
    if tok.startswith("cycles="):
        cycles = float(tok.split("=", 1)[1])
cps = cycles / run_s if run_s else 0.0
print(f"{size}\t{kind}\t{wall:.2f}\t{rss}\t{stats}\t{cps:.0f}\t"
      f"{out.returncode}")
sys.exit(0 if out.returncode == 0 else 1)
EOF
    [ $? -ne 0 ] && rc=1
    if [ "$PROFILE" = "1" ] && [ -x "$PGENGINE" ]; then
      (cd "$OUT" && "$PGENGINE" simulate --job "$job" \
          --out /dev/null >/dev/null 2>&1 \
          && gprof -b -p "$PGENGINE" gmon.out \
             > "gprof-$size-$kind.txt" && rm -f gmon.out)
    fi
  done
done
echo "PERF_RC=$rc"
