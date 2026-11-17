#!/usr/bin/env bash
# build only; run.sh does the same inline.  Usage: build.sh <repo> <out> <sm>
set -euo pipefail
L=$1; OUT=$2; SM_ARCH=$3
mkdir -p "$OUT"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr "$L/tools/microbench/load_after_copies/probe/load_after_copies.cu" -o "$OUT/load_after_copies"
