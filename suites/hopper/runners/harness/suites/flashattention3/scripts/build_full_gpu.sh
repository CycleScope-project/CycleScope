#!/usr/bin/env bash
# Build vendored FA3 SM90 flash_attn_3._C (flash_api.cpp, no feature stubs).
#
# usage:
#   build_full_gpu.sh --help
#   build_full_gpu.sh prepare
#   build_full_gpu.sh shard N          N ∈ 0..3   (4 ninja workers)
#   build_full_gpu.sh link
#   build_full_gpu.sh all              sequential shards; prefer 4 parallel shard jobs
#
# Objects land in $BUILD_ROOT/objects and are linked once.  Main agent should
# launch the four shard jobs concurrently, then link.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${root}/harness/build_full_gpu.py"
build_root="${FLASHATTENTION3_FULL_GPU_BUILD_ROOT:-/path/to/hopper-prototype/artifacts/fa3_full_gpu_100_20260920}"
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-4}"

usage() { sed -n '2,14p' "$0"; }

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  echo "BUILD_ROOT=${build_root}"
  echo "prepare: python3 ${python} --prepare --build-root ${build_root}"
  echo "shard0:  python3 ${python} --shard 0 --workers 4 --build-root ${build_root}"
  echo "shard1:  python3 ${python} --shard 1 --workers 4 --build-root ${build_root}"
  echo "shard2:  python3 ${python} --shard 2 --workers 4 --build-root ${build_root}"
  echo "shard3:  python3 ${python} --shard 3 --workers 4 --build-root ${build_root}"
  echo "link:    python3 ${python} --link --build-root ${build_root}"
  exit 0
fi

cmd="$1"
shift || true
case "${cmd}" in
  prepare)
    exec python3 "${python}" --prepare --build-root "${build_root}" "$@"
    ;;
  shard)
    if [[ $# -lt 1 || ! "$1" =~ ^[0-3]$ ]]; then
      echo "usage: $0 shard N   N ∈ 0..3" >&2
      exit 2
    fi
    shard="$1"
    shift
    exec python3 "${python}" --shard "${shard}" --workers 4 --build-root "${build_root}" "$@"
    ;;
  link)
    exec python3 "${python}" --link --build-root "${build_root}" "$@"
    ;;
  all)
    exec python3 "${python}" --all --workers 4 --build-root "${build_root}" "$@"
    ;;
  --list-sources|list-sources)
    exec python3 "${python}" --list-sources
    ;;
  *)
    echo "unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
