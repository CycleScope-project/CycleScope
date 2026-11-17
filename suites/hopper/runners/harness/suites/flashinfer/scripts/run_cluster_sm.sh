#!/usr/bin/env bash
# Independent FlashInfer 1CTA/2CTA cluster-SM runner. Default full-GPU path is unchanged.
set -euo pipefail

usage() {
  echo "usage: $0 h100 <case_id> [repeats]   OR   $0 h100 --shard N" >&2
  exit 2
}

if [[ $# -lt 2 ]]; then
  usage
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
root="${FLASHINFER_DELIVERY_SOURCE_ROOT}"
profile="$1"
shift

CLUSTER_ROOT="${FLASHINFER_CLUSTER_SM_ROOT:-/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920/flashinfer}"
# Previous verified 0.6.17 install (mechanism_coverage_20260920). Do not use
# FLASHINFER_DISABLE_VERSION_CHECK: uninstall the image's incompatible optional
# jit-cache and let the default package check run.
BUILD_ROOT="${FLASHINFER_CLUSTER_SM_BUILD_ROOT:-/path/to/hopper-prototype/artifacts/mechanism_coverage_20260920/flashinfer}"
export FLASHINFER_DELIVERY_ARTIFACTS_ROOT="${BUILD_ROOT}"
export FLASHINFER_CLUSTER_SM_MODE=1
unset FLASHINFER_DISABLE_VERSION_CHECK
export HOPPER_FLASHINFER_API_COMPAT="${HOPPER_FLASHINFER_API_COMPAT:-1}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export NINJAFLAGS="${NINJAFLAGS:--j2}"
# Image ships flashinfer-jit-cache 0.6.6; 0.6.17 refuses a mismatched optional cache.
python3 -m pip uninstall -y flashinfer-jit-cache >/dev/null 2>&1 || true
if [[ -f "${root}/../common/flashinfer_runtime_compat/sitecustomize.py" ]]; then
  export PYTHONPATH="${root}/../common/flashinfer_runtime_compat:${PYTHONPATH:-}"
fi

site_root="${BUILD_ROOT}/build/site-packages-fi0617-cutlass470-tvmffi0113"
cutlass_python_root="${site_root}/nvidia_cutlass_dsl/dsl_packages"
export PATH="${site_root}/nvidia/cu13/bin:${PATH}"
export FLASHINFER_CUBIN_DIR="${BUILD_ROOT}/build/cubins"
mkdir -p "${FLASHINFER_CUBIN_DIR}" "${CLUSTER_ROOT}"
source "${root}/scripts/device_profile.sh"
flashinfer_export_cuda_wheel_paths "${site_root}" || exit 4
delivery_check_gpu_profile "${profile}" || exit $?

[[ -f "${BUILD_ROOT}/build/source_revision.txt" && -d "${site_root}/flashinfer" ]] || {
  echo "missing FlashInfer build at ${BUILD_ROOT}" >&2
  exit 4
}
[[ -f "${BUILD_ROOT}/build/libdelivery_globaltimer_timer.so" ]] || {
  echo "missing globaltimer library" >&2
  exit 4
}

export PYTHONPATH="${cutlass_python_root}:${site_root}:${PYTHONPATH:-}"
python3 - <<'PY'
import importlib.util
import os
import pathlib
import flashinfer

assert not os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK"), "version check must stay enabled"
assert flashinfer.__version__ == "0.6.17", flashinfer.__version__
expected = "a0a6b019b9b27d49d209f85d028a1ae5a9b347d7"
actual = flashinfer.__git_commit__
assert actual != "unknown" and (expected.startswith(actual) or actual.startswith(expected)), actual
path = pathlib.Path(flashinfer.__file__).resolve()
print("loaded flashinfer", flashinfer.__version__, actual, path)
spec = importlib.util.find_spec("flashinfer_jit_cache")
if spec is None:
    print("flashinfer-jit-cache: not installed")
else:
    import flashinfer_jit_cache
    ver = getattr(flashinfer_jit_cache, "__version__", "?")
    print("flashinfer-jit-cache still importable:", ver)
    if not str(ver).startswith("0.6.17"):
        raise SystemExit(
            f"incompatible flashinfer-jit-cache {ver}; uninstall it, do not bypass the check"
        )
PY

trace_so="${CLUSTER_ROOT}/cluster_launch_trace.so"
# Match the image libcudart layout. Wheel cu13 headers change cudaDeviceProp and
# make Triton's get_device_properties return CUDA unknown error.
cuda_inc=""
for d in /usr/local/cuda/include "${site_root}/nvidia/cu13/include"; do
  if [[ -f "${d}/cuda_runtime_api.h" && -f "${d}/cuda.h" ]]; then
    cuda_inc="${d}"
    break
  fi
done
[[ -n "${cuda_inc}" ]] || { echo "cuda headers not found" >&2; exit 4; }
{
  flock 9
  tmp="${trace_so}.$$.tmp"
  gcc -shared -fPIC -O2 -I"${cuda_inc}" \
    "${root}/harness/cluster_launch_trace.c" \
    "${root}/harness/cluster_launch_getproc.c" \
    -o "${tmp}" -ldl
  mv -f "${tmp}" "${trace_so}"
  echo "built ${trace_so} -I${cuda_inc}"
} 9>"${CLUSTER_ROOT}/.cluster_launch_trace.lock"
# NFS LD_PRELOAD often fails with "cannot stat shared object". Use a local copy.
local_so="/tmp/flashinfer_cluster_launch_trace.$$.so"
cp -f "${trace_so}" "${local_so}"
test -s "${local_so}" || { echo "local interceptor copy failed" >&2; exit 4; }
trap 'rm -f "${local_so}"' EXIT
# Drop stale ninja locks left by killed JIT compiles.
find "${CLUSTER_ROOT}/jit-cache-shared" -name '.ninja_lock' -mmin +5 -delete 2>/dev/null || true

manifest="${DELIVERY_CASE_MANIFEST:-${root}/manifests/flashinfer_cluster_sm100.json}"
[[ -f "${manifest}" ]] || python3 "${root}/scripts/generate_cluster_sm_manifest.py"
[[ -f "${manifest}" ]] || { echo "missing ${manifest}" >&2; exit 4; }

run_one() {
  local case_id="$1"
  local repeats="${2:-9}"
  local attempt_dir="${3:-${FLASHINFER_CLUSTER_SM_ATTEMPT_DIR:-${CLUSTER_ROOT}/cases/${case_id}/manual}}"
  mkdir -p "${attempt_dir}"
  local output="${attempt_dir}/result.json"
  local launch_log="${attempt_dir}/launches.jsonl"
  : > "${launch_log}"
  local cache_root="${DELIVERY_JIT_CACHE_ROOT:-${CLUSTER_ROOT}/jit-cache-shared}"
  mkdir -p "${cache_root}"
  delivery_write_gpu_metadata "${attempt_dir}"
  echo "start ${profile} FlashInfer cluster-sm case=${case_id} repeats=${repeats} dir=${attempt_dir}"
  rm -f "${output}"
  env \
    FLASHINFER_CLUSTER_SM_MODE=1 \
    FLASHINFER_CLUSTER_LAUNCH_LOG="${launch_log}" \
    FLASHINFER_CLUSTER_SMID_LOG="${attempt_dir}/smid.json" \
    FLASHINFER_WORKSPACE_BASE="${cache_root}" \
    FLASHINFER_CLUSTER_SM_TARGET_SMS="${FLASHINFER_CLUSTER_SM_TARGET_SMS:-}" \
    LD_PRELOAD="${local_so}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    PYTHONPATH="${root}/harness:${PYTHONPATH:-}" \
    python3 "${root}/harness/run_case.py" \
      --manifest "${manifest}" \
      --case-id "${case_id}" \
      --output "${output}" \
      --repeats "${repeats}"
  local status=$?
  echo "exit ${status} case=${case_id}" >> "${attempt_dir}/command.txt"
  echo "repeats=${repeats}" >> "${attempt_dir}/command.txt"
  echo "${status}" > "${attempt_dir}/exit_code.txt"
  return ${status}
}

if [[ "${1:-}" == "--shard" ]]; then
  shard="${2:?shard index}"
  extra=()
  if [[ "${FLASHINFER_CLUSTER_SM_ONLY_UNRESOLVED:-}" == "1" ]]; then
    extra+=(--only-unresolved)
  fi
  if [[ -n "${FLASHINFER_CLUSTER_SM_CASE_IDS_FILE:-}" ]]; then
    extra+=(--case-ids-file "${FLASHINFER_CLUSTER_SM_CASE_IDS_FILE}")
  fi
  exec python3 "${root}/scripts/run_cluster_sm_shard.py" \
    --profile "${profile}" --shard "${shard}" --root "${CLUSTER_ROOT}" "${extra[@]}"
fi

case_id="$1"
repeats="${2:-9}"
attempt_dir="${FLASHINFER_CLUSTER_SM_ATTEMPT_DIR:-${CLUSTER_ROOT}/cases/${case_id}/manual}"
run_one "${case_id}" "${repeats}" "${attempt_dir}"
