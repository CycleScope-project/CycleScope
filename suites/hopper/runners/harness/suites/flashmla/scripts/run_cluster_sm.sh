#!/usr/bin/env bash
# FlashMLA SM90 cluster-SM entry (G0 suite; no SM100 path).
#
# usage:
#   run_cluster_sm.sh --help
#   run_cluster_sm.sh h100 probe CASE_ID
#   run_cluster_sm.sh h100 formal CASE_ID [REPEATS]      # REPEATS defaults to 9
#   run_cluster_sm.sh h100 shard SHARD [SHARDS]          # SHARD ∈ 0..SHARDS-1, default SHARDS=8
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { sed -n '2,8p' "$0"; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

source "${root}/scripts/paths.sh"
profile="$1"
phase="$2"
case "${profile}" in
  h100) ;;
  *) echo "profile must be h100 (FlashMLA SM90 kernels are Hopper-only)" >&2; exit 2 ;;
esac

build_setting="${FLASHMLA_CLUSTER_SM_BUILD_ROOT:-${FLASHMLA_DELIVERY_ARTIFACTS_ROOT}}"
if [[ -d "${build_setting}/site-packages-flashmla" ]]; then
  build_root="${build_setting}"
else
  build_root="${build_setting}/build"
fi
art_root="${FLASHMLA_CLUSTER_SM_ROOT:-${HOPPER_PROJECT_ROOT}/artifacts/cluster_sm_400_20260920/flashmla}"
manifest="${FLASHMLA_CLUSTER_SM_MANIFEST:-${root}/manifests/flashmla_cluster_sm.json}"
site_root="${build_root}/site-packages-flashmla"

# Cluster-SM mode: the shared timer enforces 9 recorded samples unless this is set,
# so probe/shard sweeps may pass smaller --repeats.
export FLASHMLA_CLUSTER_SM_MODE=1
export FLASHMLA_DELIVERY_ARTIFACTS_ROOT="${build_setting}"
export FLASHMLA_BUILD_ROOT="${build_root}"
export PYTHONPATH="${site_root}:${root}/harness:${PYTHONPATH:-}"
mkdir -p "${art_root}/cases" "${art_root}/tmp"
source "${root}/scripts/device_profile.sh"
delivery_check_gpu_profile "${profile}" || exit $?

[[ -f "${build_root}/libdelivery_globaltimer_timer.so" && -d "${site_root}/flash_mla" ]] || {
  echo "missing flashmla build under ${build_root}; run scripts/build_cluster_sm.sh ${profile} first" >&2
  exit 4
}

run_one() {
  local phase="$1" case_id="$2" repeats="$3"
  local out_dir="${art_root}/cases/${case_id}"
  mkdir -p "${out_dir}"
  python3 "${root}/harness/flashmla_cluster_sm.py" \
    --manifest "${manifest}" \
    --case-id "${case_id}" \
    --mode timing \
    --repeats "${repeats}" \
    --output "${out_dir}/${phase}.json"
}

case_ids_from_shard() {
  local shard="$1" shards="$2"
  python3 - "${manifest}" "${shard}" "${shards}" <<'PY'
import json, sys

manifest, shard, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for case in json.load(open(manifest))["cases"]:
    if case["delivery_index"] % shards == shard:
        print(case["case_id"])
PY
}

case "${phase}" in
  probe)
    case_id="${3:?case id}"
    run_one probe "${case_id}" 3
    ;;
  formal)
    case_id="${3:?case id}"
    repeats="${4:-9}"
    run_one formal "${case_id}" "${repeats}"
    ;;
  shard)
    shard="${3:?shard}"
    shards="${4:-8}"
    status=0
    while IFS= read -r case_id; do
      echo "=== flashmla shard ${shard}/${shards}: ${case_id}"
      run_one formal "${case_id}" 9 || status=1
    done < <(case_ids_from_shard "${shard}" "${shards}")
    exit "${status}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
