#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 h20|h100|h200" >&2
  exit 2
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
root="${DEEPGEMM_DELIVERY_SOURCE_ROOT}"
artifact_root="${DEEPGEMM_DELIVERY_ARTIFACTS_ROOT}"
profile="$1"
python3 "${root}/../common/work_tree.py" check --source "${root}" --output "${artifact_root}"
source "${root}/scripts/device_profile.sh"
delivery_check_gpu_profile "${profile}"
output="${artifact_root}/build/marker_package"
work_root="${artifact_root}/build/package"
log_root="${artifact_root}/logs/build"
pointer="${artifact_root}/build/marker_package.path"
if ! mkdir -p "${log_root}" "${artifact_root}/build"; then
  alt="${TMPDIR:-/dev/shm}/deepgemm_fullgpu_$$"
  output="${alt}/marker_package"
  work_root="${alt}/package"
  log_root="${alt}/logs"
  pointer="${alt}/marker_package.path"
  mkdir -p "${log_root}" "$(dirname "${output}")"
  echo "DeepGEMM artifact build root is not writable; using ${alt}"
fi
delivery_write_gpu_metadata "${log_root}"
nvcc --version > "${log_root}/nvcc_version.txt"
trap 'chown -R 10250:10250 "${output}" "${work_root}" "${log_root}" 2>/dev/null || true' EXIT

keep_existing="$(python3 - "${output}" "${pointer}" "${root}/harness" <<'PY'
import sys
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
for candidate in candidates:
    if candidate.is_dir() and is_full7_package(candidate):
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
)" && {
  echo "keeping existing DeepGEMM marker package: ${keep_existing}"
  printf '%s\n' "${keep_existing}" > "${pointer}"
  exit 0
}
echo "existing DeepGEMM marker package is not a full7 native-heuristic+layout+metadata package; rebuilding"
if [[ -e "${output}" || -e "${work_root}" ]]; then
  stamp="$(date +%Y%m%dT%H%M%S)_$$"
  output="$(dirname "${output}")/marker_package_${stamp}"
  work_root="$(dirname "${work_root}")/package_${stamp}"
  echo "partial DeepGEMM build leftover present; using writable ${output}"
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
GEMM_DELIVERY_GPU="${profile}" python3 build_deepgemm_full7_package.py \
  --source "${root}/upstream" \
  --output "${output}" \
  --work-root "${work_root}"
