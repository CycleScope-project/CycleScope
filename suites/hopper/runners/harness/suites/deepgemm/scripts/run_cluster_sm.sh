#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 h100 [CASE_ID]" >&2
  echo "env: CLUSTER_SM_SHARD CLUSTER_SM_SHARDS DEEPGEMM_DELIVERY_PACKAGE CLUSTER_SM_WORK" >&2
  exit 2
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
root="${DEEPGEMM_DELIVERY_SOURCE_ROOT}"
profile="$1"
case "${profile}" in
  h100) ;;
  *) echo "cluster_sm runner is H100-only" >&2; exit 2 ;;
esac
source "${root}/scripts/device_profile.sh"
delivery_check_gpu_profile "${profile}" || exit $?

manifest="${DELIVERY_CASE_MANIFEST:-${root}/manifests/deepgemm_cluster_sm100.json}"
package="${DEEPGEMM_DELIVERY_PACKAGE:-}"
artifact_root="${CLUSTER_SM_WORK:-${DEEPGEMM_DELIVERY_ARTIFACTS_ROOT}}"
if [[ -z "${package}" && -f "${artifact_root}/build/marker_package.path" ]]; then
  package="$(cat "${artifact_root}/build/marker_package.path")"
fi
package="${package:-${artifact_root}/build/marker_package}"
if [[ ! -f "${package}/deep_gemm/sm90_suite_marker.json" ]]; then
  echo "missing cluster_sm marker package at ${package}" >&2
  exit 4
fi
python3 - "${package}" "${root}/harness" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from build_deepgemm_full7_package import is_full7_package
pkg = Path(sys.argv[1])
if not is_full7_package(pkg):
    raise SystemExit("cluster_sm refuses a non-full7 marker package")
meta = json.loads((pkg / "deep_gemm/sm90_suite_marker.json").read_text())
if meta.get("cluster_sm_allow_num_sms_1") is True:
    raise SystemExit("cluster_sm refuses illegally relaxed num_sms=1 marker package")
if meta.get("kgroup_cluster_required") is True:
    raise SystemExit("cluster_sm refuses a cluster-forced k-group package")
PY

only="${2:-${DELIVERY_ONLY_CASE:-}}"
shard="${CLUSTER_SM_SHARD:-}"
shards="${CLUSTER_SM_SHARDS:-1}"
case_list="${CLUSTER_SM_CASE_LIST:-}"
export GEMM_DELIVERY_GPU="${profile}"
export DEEPGEMM_DELIVERY_RAW_PER_SAMPLE="${DEEPGEMM_DELIVERY_RAW_PER_SAMPLE:-4}"
mkdir -p "${artifact_root}"
delivery_write_gpu_metadata "${artifact_root}" 2>/dev/null || true

workers="${CLUSTER_SM_WORKERS:-4}"
if [[ "${workers}" -lt 4 || "${workers}" -gt 16 ]]; then
  echo "CLUSTER_SM_WORKERS=${workers} outside 4-16; using 4" >&2
  workers=4
fi
export CLUSTER_SM_WORKERS="${workers}"
export CLUSTER_SM_GPU_LOCK="${CLUSTER_SM_GPU_LOCK:-${artifact_root}/tmp/gpu.lock}"
mkdir -p "${artifact_root}/tmp"
echo "concurrency outer_shards=${shards} inner_workers=${workers} gpu_lock=serial shard=${shard:-all}"

run_ids=()
while IFS=$'\t' read -r action cid; do
  if [[ "${action}" == "REUSE" ]]; then
    echo "reuse existing cluster_sm summary case=${cid}"
    continue
  fi
  run_ids+=("${cid}")
done < <(python3 - "${manifest}" "${only}" "${shard}" "${shards}" "${artifact_root}" "${case_list}" <<'PY'
import json, sys
from pathlib import Path
data = json.load(open(sys.argv[1]))
only, shard, shards = sys.argv[2], sys.argv[3], int(sys.argv[4])
work = Path(sys.argv[5])
case_list = [item for item in sys.argv[6].split(",") if item] if len(sys.argv) > 6 else []
rows = data["cases"]
if only:
    rows = [row for row in rows if row["case_id"] == only]
elif case_list:
    order = {cid: i for i, cid in enumerate(case_list)}
    rows = [row for row in rows if row["case_id"] in order]
    rows.sort(key=lambda row: order[row["case_id"]])
    if shard != "":
        shard_i = int(shard)
        rows = [row for i, row in enumerate(rows) if i % shards == shard_i]
elif shard != "":
    shard_i = int(shard)
    rows = [row for row in rows if int(row["delivery_index"]) % shards == shard_i]
for row in rows:
    summary = work / "summaries" / f"{row['case_id']}.json"
    if summary.is_file():
        payload = json.loads(summary.read_text())
        formal = payload.get("formal") or {}
        samples = ((formal.get("payload") or {}).get("timing") or {}).get("samples_cycles") or []
        if payload.get("status") == "skipped" or (
            formal.get("status") == "pass" and len(samples) >= 9
        ):
            print("REUSE\t" + row["case_id"])
            continue
    print("RUN\t" + row["case_id"])
PY
)

if ((${#run_ids[@]})); then
  printf '%s\0' "${run_ids[@]}" | xargs -0 -P "${workers}" -n 1 bash -c '
    cid="$1"
    echo "start cluster_sm case=${cid}"
    python3 "'"${root}"'/scripts/cluster_sm_case.py" \
      --manifest "'"${manifest}"'" \
      --case-id "${cid}" \
      --package "'"${package}"'" \
      --work "'"${artifact_root}"'" || echo "cluster_sm case rc=$? case=${cid}"
  ' _
fi

if [[ -z "${shard}" && -z "${only}" ]] || [[ "${CLUSTER_SM_FINALIZE:-}" == "1" ]]; then
  python3 "${root}/scripts/cluster_sm_qualify.py"     --manifest "${manifest}"     --work "${artifact_root}"     --package "${package}"     --source-manifest "${root}/manifests/deepgemm_100.json"
fi
