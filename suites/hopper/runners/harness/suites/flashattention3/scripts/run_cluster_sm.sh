#!/usr/bin/env bash
# FA3 1CTA/2CTA cluster-SM entry. Does not replace scripts/run_100.sh.
#
# usage:
#   run_cluster_sm.sh --help
#   run_cluster_sm.sh generate
#   run_cluster_sm.sh h100 probe CASE_ID
#   run_cluster_sm.sh h100 formal CASE_ID
#   run_cluster_sm.sh h100 shard SHARD [SHARDS]
# SHARD ∈ 0..SHARDS-1; default SHARDS=8
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${FLASHATTENTION3_FULL_GPU_BUILD_ROOT:-/path/to/hopper-prototype/artifacts/fa3_full_gpu_100_20260920}"
art_root="${FLASHATTENTION3_CLUSTER_SM_ROOT:-/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920/fa3}"
manifest="${FLASHATTENTION3_CLUSTER_SM_MANIFEST:-${root}/manifests/fa3_cluster_sm100.json}"
source_manifest="${FLASHATTENTION3_FULL_GPU_MANIFEST:-${root}/manifests/fa3_100.json}"
runner="${root}/harness/run_cluster_sm.py"

usage() { sed -n '2,12p' "$0"; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export PYTHONPATH="${root}/harness:${build_root}/site-packages:${PYTHONPATH:-}"
mkdir -p "${art_root}"

if [[ "${1:-}" == "generate" ]]; then
  exec python3 "${runner}" --phase generate \
    --source-manifest "${source_manifest}" \
    --derived-manifest "${manifest}" \
    --art-root "${art_root}" \
    --build-root "${build_root}"
fi

profile="${1:-}"
phase="${2:-}"
[[ "${profile}" == "h100" ]] || { echo "profile must be h100" >&2; usage >&2; exit 2; }

case "${phase}" in
  probe|formal)
    case_id="${3:?case id}"
    exec python3 "${runner}" --phase "${phase}" --case-id "${case_id}" \
      --source-manifest "${source_manifest}" \
      --derived-manifest "${manifest}" \
      --art-root "${art_root}" \
      --build-root "${build_root}"
    ;;
  shard)
    shard="${3:?shard}"
    shards="${4:-8}"
    exec python3 "${runner}" --phase shard --shard "${shard}" --shards "${shards}" \
      --source-manifest "${source_manifest}" \
      --derived-manifest "${manifest}" \
      --art-root "${art_root}" \
      --build-root "${build_root}"
    ;;
  merge)
    shards="${3:-8}"
    exec python3 "${runner}" --phase merge --shards "${shards}" \
      --source-manifest "${source_manifest}" \
      --derived-manifest "${manifest}" \
      --art-root "${art_root}" \
      --build-root "${build_root}"
    ;;
  repair)
    shift 2 || true
    export PYTHONUNBUFFERED=1
    exec python3 "${root}/harness/repair_cluster_sm.py" "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
