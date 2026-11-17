#!/usr/bin/env bash
# global_atomic protocol r1: identity + sass gate + two runs (run-a/run-b)
# in the frozen cell order chase_l2, pingpong_addr, atom_chain_addr,
# red_same (nsm list), atom_same (1, full).  clocks.sm before/after a run.
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm --format=csv > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/global_atomic/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/global_atomic"
[ -x "$B" ] || { echo "SKIP global_atomic (build failed)"; exit 1; }
( cd "$TREE/tools/microbench/global_atomic" && \
  python3 sass_check.py --sass "$OUT/build/sass/global_atomic.sass" \
      --out "$OUT/sass/checks.json" ) || echo "SASS_CHECK_FAIL rc=$?"

# device SM count -> protocol cell nsm lists
case "$SM_ARCH" in
  sm_90a)   NSM=132; RED_NSMS="1 2 4 8 16 32 66 132" ;;
  sm_103a)  NSM=148; RED_NSMS="1 2 4 8 16 32 74 148" ;;
esac

RUNS=${RUNS:-a,b}
CELLS=${CELLS:-chase_l2,pingpong_addr,atom_chain_addr,red_same,atom_same}
for run in ${RUNS//,/ }; do
  R="$OUT/run-$run"
  mkdir -p "$R"
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader > "$R/clocks_before.txt"
  case ",$CELLS," in
    *,chase_l2,*) "$B" chase_l2 1 32 8192 5 > "$R/chase_l2.txt" ;;
  esac
  case ",$CELLS," in
    *,pingpong_addr,*) "$B" pingpong_addr "$NSM" 32 1000 3 peer=8 > "$R/pingpong_addr.txt" ;;
  esac
  case ",$CELLS," in
    *,atom_chain_addr,*) "$B" atom_chain_addr 1 32 1024 3 > "$R/atom_chain_addr.txt" ;;
  esac
  case ",$CELLS," in
    *,red_same,*) for n in $RED_NSMS; do
      "$B" red_same "$n" 1024 2048 5 > "$R/red_same.n${n}.txt"
    done ;;
  esac
  case ",$CELLS," in
    *,atom_same,*) "$B" atom_same 1 1024 2048 5 > "$R/atom_same.n1.txt"
      "$B" atom_same "$NSM" 1024 2048 5 > "$R/atom_same.n${NSM}.txt" ;;
  esac
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader > "$R/clocks_after.txt"
done
echo "run.sh: global_atomic done" > "$OUT/done.txt"
