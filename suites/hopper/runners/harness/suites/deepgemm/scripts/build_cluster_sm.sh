#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 h100" >&2
  exit 2
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
root="${DEEPGEMM_DELIVERY_SOURCE_ROOT}"
profile="$1"
source "${root}/scripts/device_profile.sh"
delivery_check_gpu_profile "${profile}" || exit $?

artifact_root="${CLUSTER_SM_WORK:-${DEEPGEMM_DELIVERY_ARTIFACTS_ROOT}}"
output="${artifact_root}/build/marker_package"
work_root="${artifact_root}/build/package"
log_root="${artifact_root}/logs/build"
pointer="${artifact_root}/build/marker_package.path"
mkdir -p "${log_root}" "${artifact_root}/build"
delivery_write_gpu_metadata "${log_root}" 2>/dev/null || true

keep="$(python3 - "${output}" "${pointer}" "${root}/harness" <<'PY' || true
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from build_deepgemm_full7_package import is_full7_package
root = Path(sys.argv[1])
pointer = Path(sys.argv[2])
candidates = []
if pointer.is_file():
    candidates.append(Path(pointer.read_text().strip()))
candidates.append(root)
candidates.extend(sorted(root.parent.glob("marker_package_*"), reverse=True))
candidates.append(Path("/path/to/hopper-prototype/artifacts/mechanism_coverage_20260920/deepgemm/rebuilt/build/marker_package_20260920T205531_140"))
for candidate in candidates:
    meta = candidate / "deep_gemm/sm90_suite_marker.json"
    if candidate.is_dir() and is_full7_package(candidate) and meta.is_file():
        payload = json.loads(meta.read_text())
        if payload.get("cluster_sm_allow_num_sms_1") is True:
            continue
        if payload.get("kgroup_cluster_required") is True:
            continue
        if payload.get("force_multicast") is not None:
            # DG_DELIVERY_FORCE_MULTICAST rescue package: measurement-only,
            # never reusable as the formal cluster_sm marker package.
            continue
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
)"
if [[ -n "${keep}" ]]; then
  printf '%s\n' "${keep}" > "${pointer}"
  echo "keeping cluster_sm marker package: ${keep}"
  exit 0
fi

if [[ -e "${output}" || -e "${work_root}" ]]; then
  stamp="$(date +%Y%m%dT%H%M%S)_$$"
  output="$(dirname "${output}")/marker_package_${stamp}"
  work_root="$(dirname "${work_root}")/package_${stamp}"
fi
printf '%s\n' "${output}" > "${pointer}"

cuda_python_includes="$(python3 - <<'PY'
import site
from pathlib import Path
print(":".join(str(path) for root in site.getsitepackages()
               for path in sorted(Path(root).glob("nvidia/*/include")) if path.is_dir()))
PY
)"
export CPATH="${cuda_python_includes}${CPATH:+:${CPATH}}"
cd "${root}/harness"
unset DEEPGEMM_DELIVERY_PACKAGE || true
GEMM_DELIVERY_GPU="${profile}" python3 build_deepgemm_full7_package.py \
  --source "${root}/upstream" \
  --output "${output}" \
  --work-root "${work_root}"

