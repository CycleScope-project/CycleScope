#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 h100 SHARD(0..7) [SAMPLES]" >&2
  exit 2
fi

export CUTLASS_DELIVERY_MODE=cluster_sm
export CUTLASS_DELIVERY_ARTIFACTS_ROOT="${CUTLASS_DELIVERY_ARTIFACTS_ROOT:-/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920/cutlass}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
root="${CUTLASS_DELIVERY_SOURCE_ROOT}"
artifact_root="${CUTLASS_DELIVERY_ARTIFACTS_ROOT}"
profile="$1"
shard="$2"
repeats="${3:-9}"
case "${profile}" in h100) ;; *) echo "cluster_sm runner is H100-only" >&2; exit 2 ;; esac
case "${shard}" in 0|1|2|3|4|5|6|7) ;; *) echo "shard must be 0..7" >&2; exit 2 ;; esac
if [[ "${repeats}" -lt 1 || "${repeats}" -gt 9 ]]; then
  echo "samples must be 1..9" >&2
  exit 2
fi

export DELIVERY_CASE_MANIFEST="${DELIVERY_CASE_MANIFEST:-${root}/manifests/cutlass_cluster_sm100.json}"
source "${root}/scripts/device_profile.sh"
delivery_check_gpu_profile "${profile}"
plan="${DELIVERY_CASE_MANIFEST}"
result_set="${DELIVERY_RESULT_SET:-cluster_sm}"
[[ -f "${plan}" ]] || { echo "missing case manifest: ${plan}" >&2; exit 4; }

output="${artifact_root}/${result_set}/${profile}/shard${shard}"
delivery_write_gpu_metadata "${output}"
trap 'chown -R 10250:10250 "${output}" 2>/dev/null || true' EXIT

only_case_args=()
if [[ -n "${DELIVERY_ONLY_CASE:-}" ]]; then
  only_case_args+=(--only-case "${DELIVERY_ONLY_CASE}")
fi

if ! GEMM_DELIVERY_GPU="${profile}" CUTLASS_DELIVERY_MODE=cluster_sm python3 "${root}/harness/run_cutlass_case_plan.py" \
  --plan "${plan}" \
  --output-dir "${output}" \
  --shard-id "${shard}" \
  --shard-count 8 \
  --repeats "${repeats}" \
  "${only_case_args[@]}"; then
  echo "CUTLASS cluster_sm case plan failed" >&2
  exit 1
fi
