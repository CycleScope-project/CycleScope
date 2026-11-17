#!/bin/bash
# build cublaslt_case for the B300 (CUDA 13.2) image -> artifact root binary
set -e
OUT=${OUT:-${CYCLESCOPE_ARTIFACT_ROOT:?CYCLESCOPE_ARTIFACT_ROOT is not set (cyclescope capture exports it)}/tools/blackwell-cublas-build}
mkdir -p "$OUT"
INC=/usr/local/cuda/include
LIB=/usr/local/cuda-13.2/targets/x86_64-linux/lib
nvcc -O2 -std=c++17 -I"$INC" "$(dirname "$0")/cublaslt_case.cpp" -o "$OUT/cublaslt_case" \
  -L"$LIB" -lcublasLt -lcudart -Xlinker -rpath -Xlinker "$LIB"
sha256sum "$OUT/cublaslt_case"
