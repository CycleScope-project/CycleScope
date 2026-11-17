#!/usr/bin/env bash
# c450v2 build pod: DeepGEMM marker package + cublaslt_case + FI jit cache.
# env: V2_ROOT (default $ART/rebuild/c450v2), V2_WORKERS (default 16)
# Marker: V2BUILD_RC=
set -uo pipefail
REPO="${CLUSTER_SM_REPO:?}"
ART="${HOPPER_ARTIFACT_ROOT:-/path/to/hopper-prototype/artifacts}"
V="${V2_ROOT:-$ART/rebuild/c450v2}"
C="$ART/rebuild/c450_full"
W="${V2_WORKERS:-16}"
mkdir -p "$V/bin" "$V/build/deepgemm" "$V/manifests"
RC=0

# 1) DeepGEMM marker package from this clone's vendored upstream
#    (carries DG_DELIVERY_FORCE_LAYOUT).
if [ "${V2_FORCE_DG:-0}" != 1 ] \
    && [ -f "$V/build/deepgemm/marker_package/deep_gemm/sm90_suite_marker.json" ]; then
  echo "V2BUILD deepgemm package present, skip"
else
rm -rf "$V/build/deepgemm/marker_package" "$V/build/deepgemm/package_work"
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
link_dir="$V/build/deepgemm/cuda-links"; mkdir -p "$link_dir"
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
    --output "$V/build/deepgemm/marker_package" \
    --work-root "$V/build/deepgemm/package_work") \
  > "$V/build/deepgemm/build.log" 2>&1 || RC=1
ls "$V/build/deepgemm/marker_package/deep_gemm/sm90_suite_marker.json" \
  >/dev/null 2>&1 || RC=$((RC|2))
fi

# 2) cublaslt_case -> $V/bin (nvcc from the image's CUDA toolkit)
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
cublas_lib="$(python3 - <<'PY'
import glob
print(":".join(sorted(glob.glob(
    "/usr/local/lib/python3.12/dist-packages/nvidia/*/lib"))))
PY
)"
cublas_inc="$(python3 - <<'PY'
import site
from pathlib import Path
for r in site.getsitepackages():
    for p in sorted(Path(r).glob("nvidia/cublas/include")):
        if p.is_dir():
            print(p)
PY
)"
lib_flags=""
IFS=':' read -ra _libs <<< "$cublas_lib"
# pip ships only the versioned soname; add an unversioned shim for -l
link_cublas="$V/build/deepgemm/cuda-links"
mkdir -p "$link_cublas"
for d in "${_libs[@]}"; do
  lib_flags+=" -L$d -Xlinker -rpath -Xlinker $d"
  for v in "$d"/libcublasLt.so.*; do
    [ -f "$v" ] && [ ! -e "$link_cublas/libcublasLt.so" ] \
      && ln -s "$v" "$link_cublas/libcublasLt.so"
  done
done
[ -e "$link_cublas/libcublasLt.so" ] \
  && lib_flags+=" -L$link_cublas -Xlinker -rpath -Xlinker $link_cublas"
"$NVCC" -O2 -arch=sm_90a \
  "$REPO/suites/c450_full/cublas/cublaslt_case.cpp" \
  -o "$V/bin/cublaslt_case" -lcublasLt -lcudart \
  -I/usr/local/cuda/include ${cublas_inc:+-I"$cublas_inc"} \
  -L/usr/local/cuda/lib64 $lib_flags \
  -Xlinker -rpath -Xlinker /usr/local/cuda/lib64 \
  > "$V/bin/cublaslt_case.build.log" 2>&1 || RC=$((RC|4))
ls "$V/bin/cublaslt_case" >/dev/null 2>&1 || RC=$((RC|8))

# 3) FlashInfer JIT cache copy (root-owned source; cp -a then fix ownership
#    is handled by the launcher chown; fall back to plain copy).
if [ ! -d "$V/flashinfer_jit_cache" ]; then
  cp -a "$C/flashinfer_jit_cache_pinned" "$V/flashinfer_jit_cache" 2>/dev/null \
    || { mkdir -p "$V/flashinfer_jit_cache" && \
         cp -r "$C/flashinfer_jit_cache_pinned/." "$V/flashinfer_jit_cache/"; } \
    || RC=$((RC|16))
fi

# 4) FlashMLA build -> $V/build/flashmla (carries the sparse-decode
#    FLASHMLA_EFFECTIVE_SM_COUNT patch in sparse_decode.cpp)
if [ ! -d "$V/build/flashmla/site-packages-flashmla/flash_mla" ]; then
  FLASHMLA_BUILD_ROOT="$V/build/flashmla" \
    bash "$REPO/suites/flashmla/scripts/build_cluster_sm.sh" h100 \
    > "$V/build/flashmla.build.log" 2>&1 || RC=$((RC|32))
  ls "$V/build/flashmla/site-packages-flashmla/flash_mla" \
    >/dev/null 2>&1 || RC=$((RC|64))
else
  echo "V2BUILD flashmla build present, skip"
fi

echo "V2BUILD_RC=$RC"
