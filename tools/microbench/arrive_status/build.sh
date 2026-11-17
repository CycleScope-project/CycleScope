#!/usr/bin/env bash
# build.sh <repo> <out dir> <sm_arch>
set -e
L=$1; OUT=$2; SM_ARCH=${3:-sm_90a}
mkdir -p "$OUT/build"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 \
  -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" \
  -lineinfo --expt-relaxed-constexpr \
  "$L/tools/microbench/arrive_status/probe/arrive_status.cu" \
  -o "$OUT/build/arrive_status"
