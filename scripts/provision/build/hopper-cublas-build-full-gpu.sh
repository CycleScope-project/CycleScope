#!/usr/bin/env bash
# hopper-cublas-build-full-gpu: cublaslt_case (@23012ee1 src) + cublaslt_case_r13
# (@d5064803 src = c450_full/cublas/cublaslt_case.cpp).
# evidence: c450v2_build_pod.sh L63-81, c450_full/cublas/build.sh:13 link by path.
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-cublas-build-full-gpu
H=$CYCLESCOPE_ROOT/suites/hopper/runners/harness
rm -rf "$T"
mkdir -p "$T"
cublas_inc="$(python3 - <<'PY'
import site
from pathlib import Path
for r in site.getsitepackages():
    for p in sorted(Path(r).glob("nvidia/cublas/include")):
        if p.is_dir(): print(p)
PY
)"
LIBSO=$(find /usr /usr/local /opt -name 'libcublasLt.so*' 2>/dev/null | head -1)
CURT=$(find /usr/local/cuda /usr/lib/x86_64-linux-gnu /usr/local/lib/python3*/dist-packages/nvidia/cuda_runtime -name 'libcudart.so*' 2>/dev/null | head -1)
for pair in "cublaslt_case_full_gpu.cpp cublaslt_case"; do
  set -- $pair
  g++ -O2 -std=c++17 -I"$cublas_inc" -I/usr/local/cuda/include     "$H/suites/c450_full/cublas/$1" -o "$T/$2"     "$LIBSO" "$CURT" -Wl,-rpath,"$(dirname $LIBSO)":"$(dirname $CURT)"
done
sha256sum "$T"/cublaslt_case*
