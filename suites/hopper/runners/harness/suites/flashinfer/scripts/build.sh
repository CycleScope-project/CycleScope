#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 h20|h100|h200" >&2
  exit 2
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
root="${FLASHINFER_DELIVERY_SOURCE_ROOT}"
artifact_root="${FLASHINFER_DELIVERY_ARTIFACTS_ROOT}"
profile="$1"
python3 "${root}/../common/work_tree.py" check --source "${root}" --output "${artifact_root}"
source "${root}/scripts/device_profile.sh"
delivery_check_gpu_profile "${profile}"
build_root="${artifact_root}/build"
site_root="${build_root}/site-packages-fi0617-cutlass470-tvmffi0113"
cutlass_python_root="${site_root}/nvidia_cutlass_dsl/dsl_packages"
export PATH="${site_root}/nvidia/cu13/bin:${PATH}"
export FLASHINFER_CUBIN_DIR="${artifact_root}/build/cubins"
flashinfer_export_cuda_wheel_paths "${site_root}"
# Full default fused-MoE kernel set; do not shrink instantiations.
log_root="${artifact_root}/logs/build/${profile}"
sass_root="${build_root}/sass"
mkdir -p "${build_root}" "${log_root}" "${sass_root}"
export TMPDIR="${TMPDIR:-/dev/shm}"
delivery_write_gpu_metadata "${log_root}"
python3 --version > "${log_root}/python_version.txt"
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)' > "${log_root}/image_torch.txt"
trap 'chown -R 10250:10250 "${build_root}" "${log_root}" 2>/dev/null || true' EXIT

reuse_ok=0
if [[ "${FLASHINFER_FORCE_REBUILD:-0}" != 1 && -d "${site_root}/flashinfer" && -f "${build_root}/source_revision.txt" ]]; then
  if [[ "$(tr -d '[:space:]' < "${build_root}/source_revision.txt")" == "a0a6b019b9b27d49d209f85d028a1ae5a9b347d7" ]]; then
    if PYTHONPATH="${cutlass_python_root}:${site_root}:${PYTHONPATH:-}" python3 - <<'PY'
import flashinfer
expected = "a0a6b019b9b27d49d209f85d028a1ae5a9b347d7"
actual = flashinfer.__git_commit__
assert flashinfer.__version__ == "0.6.17", flashinfer.__version__
assert actual != "unknown" and (expected.startswith(actual) or actual.startswith(expected)), actual
print("reusing FlashInfer", flashinfer.__version__, actual)
PY
    then
      reuse_ok=1
    fi
  fi
fi
if [[ "${reuse_ok}" -eq 0 ]]; then
  python3 -m pip install --upgrade --no-deps --target "${site_root}" 'flashinfer-python==0.6.17'
  python3 -m pip install --upgrade --no-deps --target "${site_root}" \
    'nvidia-cutlass-dsl==4.7.0' \
    'nvidia-cutlass-dsl-libs-base==4.7.0' \
    'nvidia-cutlass-dsl-libs-core==4.7.0' \
    'nvidia-cutlass-dsl-libs-cu12==4.7.0' \
    'apache-tvm-ffi==0.1.13.post3' \
    'backports.strenum==1.2.8' \
    'protobuf==6.33.6' \
    'nvidia-cuda-nvdisasm==13.3.73'
fi
PYTHONPATH="${cutlass_python_root}:${site_root}:${PYTHONPATH:-}" python3 - <<'PY'
import torch
import flashinfer
assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9, (
    torch.cuda.get_device_capability() if torch.cuda.is_available() else None
)
assert flashinfer.__version__ == "0.6.17", flashinfer.__version__
expected = "a0a6b019b9b27d49d209f85d028a1ae5a9b347d7"
actual = flashinfer.__git_commit__
assert actual != "unknown" and (expected.startswith(actual) or actual.startswith(expected)), actual
print(torch.__version__, torch.version.cuda, flashinfer.__version__, flashinfer.__git_commit__)
PY
package_root="$(PYTHONPATH="${cutlass_python_root}:${site_root}:${PYTHONPATH:-}" python3 -c 'import pathlib,flashinfer; print(pathlib.Path(flashinfer.__file__).parent)')"
while IFS= read -r -d '' binary; do
  relative="${binary#${package_root}/}"
  output="${sass_root}/${relative//\//_}.sass"
  cuobjdump --dump-sass "${binary}" > "${output}" || true
done < <(find "${package_root}" -type f \( -name '*.so' -o -name '*.cubin' \) -print0)
PYTHONPATH="${cutlass_python_root}:${site_root}:${PYTHONPATH:-}" python3 -m pip freeze > "${log_root}/pip_freeze.txt"
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -shared -Xcompiler=-fPIC \
  "${root}/harness/globaltimer_timer.cu" \
  -o "${build_root}/libdelivery_globaltimer_timer.so" \
  > "${log_root}/globaltimer_timer_build.stdout.txt" \
  2> "${log_root}/globaltimer_timer_build.stderr.txt"
cuobjdump --dump-sass "${build_root}/libdelivery_globaltimer_timer.so" \
  > "${log_root}/globaltimer_timer.sass"
grep -q 'SR_CLOCKLO' "${log_root}/globaltimer_timer.sass"
grep -q 'SR_GLOBALTIMERLO' "${log_root}/globaltimer_timer.sass"
# T0005 rev5 marker registry: linked into every instrumented JIT module
# (FLASHINFER_EXTRA_LDFLAGS, harness/marker_timing.py::enable_jit_marker_mode).
g++ -std=c++17 -O3 -shared -fPIC \
  "${root}/harness/override/fi_delivery_marker_registry.cpp" \
  -I "${root}/harness/override" \
  -o "${build_root}/libdelivery_fi_marker.so" \
  > "${log_root}/fi_marker_registry_build.stdout.txt" \
  2> "${log_root}/fi_marker_registry_build.stderr.txt"
nm -D "${build_root}/libdelivery_fi_marker.so" | grep -q flashinfer_delivery_marker_set_buffers
printf '%s\n' 'a0a6b019b9b27d49d209f85d028a1ae5a9b347d7' > "${build_root}/source_revision.txt"
