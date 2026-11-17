#!/bin/bash
set -e
# locate cublasLt.h: CUDA toolkit or pip nvidia-cublas-cu12
INC=""
for d in /usr/local/cuda/include /usr/lib/python3*/dist-packages/nvidia/cublas/include /usr/local/lib/python3*/dist-packages/nvidia/cublas/include /usr/lib/python3*/site-packages/nvidia/cublas/include; do
  [ -f "$d/cublasLt.h" ] && INC="$d" && break
done
[ -z "$INC" ] && INC=$(find /usr /opt -name cublasLt.h -printf '%h\n' 2>/dev/null | head -1)
[ -z "$INC" ] && { echo "cublasLt.h not found"; exit 1; }
LIBSO=$(find /usr /usr/local /opt -name 'libcublasLt.so*' 2>/dev/null | head -1)
CURT=$(find /usr/local/cuda /usr/lib/x86_64-linux-gnu /usr/local/lib/python3*/dist-packages/nvidia/cuda_runtime -name 'libcudart.so*' 2>/dev/null | head -1)
echo "INC=$INC LIBSO=$LIBSO CURT=$CURT"
g++ -O2 -std=c++17 -I"$INC" -I/usr/local/cuda/include cublaslt_case.cpp -o cublaslt_case \
  "$LIBSO" "$CURT" -Wl,-rpath,"$(dirname $LIBSO)":"$(dirname $CURT)"
