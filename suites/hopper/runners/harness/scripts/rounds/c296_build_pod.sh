#!/usr/bin/env bash
# c296 build pod: rebuild the captured programs for one suite.
# Env: C296_SUITE=cutlass|deepgemm|fa3
#      C296_OUT=$ART/rebuild/c296/build/<suite>
#      C296_WORKERS (default 16)
# Marker: C296BUILD_RC=
set -uo pipefail
REPO="${CLUSTER_SM_REPO:?}"
ART="${HOPPER_ARTIFACT_ROOT:-/path/to/hopper-prototype/artifacts}"
SUITE="${C296_SUITE:?}"
OUT="${C296_OUT:-$ART/rebuild/c296/build/$SUITE}"
W="${C296_WORKERS:-16}"
mkdir -p "$OUT"
RC=0

case "$SUITE" in
cutlass)
  export CUTLASS_DELIVERY_ARTIFACTS_ROOT="$OUT"
  export CUTLASS_DELIVERY_COMPILE_ONLY=1
  export CUTLASS_DELIVERY_BUILD_MANIFEST="$REPO/suites/cutlass/manifests/cutlass_cluster_sm100.json"
  pids=()
  for g in 0 1 2 3; do
    bash "$REPO/suites/cutlass/scripts/build.sh" h100 "$g" \
      > "$OUT/group$g.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || RC=$((RC|1)); done
  n=$(find "$OUT/build" -maxdepth 1 -type f -executable | wc -l)
  echo "cutlass binaries: $n"
  ;;
deepgemm)
  # a failed build leaves root-owned output/work roots; clean them here
  rm -rf "$OUT/marker_package" "$OUT/package_work"
  # full7 marker package (pip wheel + install --target); the same build
  # covers force-multicast cases — DG_DELIVERY_FORCE_MULTICAST is env-gated.
  cuda_inc="$(python3 - <<'PY'
import site
from pathlib import Path
print(":".join(str(p) for r in site.getsitepackages()
               for p in sorted(Path(r).glob("nvidia/*/include")) if p.is_dir()))
PY
)"
  export CPATH="${cuda_inc}${CPATH:+:${CPATH}}"
  cuda_lib="$(python3 - <<'PY'
import site
from pathlib import Path
print(":".join(str(p) for r in site.getsitepackages()
               for p in sorted(Path(r).glob("nvidia/*/lib")) if p.is_dir()))
PY
)"
  export LD_LIBRARY_PATH="${cuda_lib}:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  # ld needs an unversioned libnvrtc.so; the pip package ships only .so.NN —
  # same shim as suites/deepgemm/scripts/cluster_sm_shard.sh
  link_dir="$OUT/cuda-links"; mkdir -p "$link_dir"
  python3 - <<'PY' "$link_dir"
import site, sys
from pathlib import Path
libs = [p for base in site.getsitepackages()
        for p in Path(base).glob("nvidia/*/lib/libnvrtc.so.*") if p.is_file()]
if not libs:
    raise SystemExit("no libnvrtc.so.* under site-packages nvidia/*/lib")
link = Path(sys.argv[1]) / "libnvrtc.so"
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(libs[0])
print(link, "->", libs[0])
PY
  export LIBRARY_PATH="${link_dir}:${cuda_lib}:/usr/local/cuda/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  (cd "$REPO/suites/deepgemm/harness" && \
    GEMM_DELIVERY_GPU=h100 python3 build_deepgemm_full7_package.py \
      --source "$REPO/suites/deepgemm/upstream" \
      --output "$OUT/marker_package" \
      --work-root "$OUT/package_work") > "$OUT/build.log" 2>&1 || RC=1
  ls "$OUT/marker_package/deep_gemm/sm90_suite_marker.json" || RC=1
  ;;
fa3)
  export FLASHATTENTION3_FULL_GPU_BUILD_ROOT="$OUT"
  export MAX_JOBS="$W"
  export NVCC_THREADS=2
  bash "$REPO/suites/flashattention3/scripts/build_full_gpu.sh" \
    all --workers "$W" > "$OUT/build.log" 2>&1 || RC=1
  ls "$OUT/site-packages"/*/flash_attn_3/_C.*.so \
      "$OUT/libdelivery_globaltimer_timer.so" >/dev/null 2>&1 || RC=$((RC|2))
  # own intermediates only: drop objects+shards after a successful link
  if [ "$RC" -eq 0 ]; then rm -rf "$OUT/objects" "$OUT/shards"; fi
  ;;
*) echo "unknown C296_SUITE=$SUITE"; RC=2 ;;
esac
echo "C296BUILD_${SUITE^^}_RC=$RC"
