#!/usr/bin/env bash
# Build the pinned FlashMLA suite from the (trimmed) upstream tree.
#
# CUTLASS is NOT vendored (T0005 rev3): the pinned commit is fetched at build
# time into $HOPPER_ARTIFACT_ROOT/tool_cache/cutlass/<commit> (reused when
# present) and symlinked to upstream/csrc/cutlass before pip runs. See
# upstream/PINNED.md for the pin and the verification that ruled out reusing
# suites/cutlass/upstream.
#
# Produces, under ${FLASHMLA_BUILD_ROOT:-${FLASHMLA_DELIVERY_ARTIFACTS_ROOT}/build}:
#   site-packages-flashmla/   flash_mla wheel built from upstream/ (sm90a gencode only)
#   libdelivery_globaltimer_timer.so  shared delivery globaltimer timer
#   source_revision.txt       expected upstream pin, reused on re-runs
#
# usage: build_cluster_sm.sh h100
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 h100" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${script_dir}/.." && pwd)"
export GIT_CEILING_DIRECTORIES="${source_root}${GIT_CEILING_DIRECTORIES:+:${GIT_CEILING_DIRECTORIES}}"
source "${script_dir}/paths.sh"
root="${FLASHMLA_DELIVERY_SOURCE_ROOT}"
artifact_root="${FLASHMLA_DELIVERY_ARTIFACTS_ROOT}"
profile="$1"
case "${profile}" in
  h100) ;;
  *) echo "unsupported profile for flashmla (SM90 paths are Hopper-only): ${profile}" >&2; exit 2 ;;
esac
source "${root}/scripts/device_profile.sh"
cpu_only="${FLASHMLA_CPU_ONLY_BUILD:-0}"
export FLASHMLA_CPU_ONLY_BUILD="${cpu_only}"
if [[ "${cpu_only}" != 1 ]]; then
  delivery_check_gpu_profile "${profile}"
fi

expected_revision="$(sed -n 's/^- Pinned revision: `\(.*\)`$/\1/p' "${root}/upstream/PINNED.md" | head -n 1)"
[[ -n "${expected_revision}" ]] || { echo "cannot read FlashMLA pin from upstream/PINNED.md" >&2; exit 4; }

build_root="${FLASHMLA_BUILD_ROOT:-${artifact_root}/build}"
site_root="${build_root}/site-packages-flashmla"
log_root="${FLASHMLA_BUILD_LOG_ROOT:-${artifact_root}/logs/build/${profile}}"
mkdir -p "${build_root}" "${log_root}"
export TMPDIR="${TMPDIR:-/dev/shm}"
if [[ "${cpu_only}" == 1 ]]; then
  printf '%s\n' "CPU-only build; GPU metadata collection skipped" \
    > "${log_root}/build_environment.txt"
else
  delivery_write_gpu_metadata "${log_root}"
fi
trap 'chown -R 10250:10250 "${build_root}" "${log_root}" 2>/dev/null || true' EXIT

reuse_ok=0
if [[ "${FLASHMLA_FORCE_REBUILD:-0}" != 1 && -d "${site_root}/flash_mla" && -f "${build_root}/source_revision.txt" ]] \
  && [[ "$(tr -d '[:space:]' < "${build_root}/source_revision.txt")" == "${expected_revision}" ]] \
  && [[ -f "${build_root}/marker_timing_enabled.txt" ]] \
  && [[ "$(tr -d '[:space:]' < "${build_root}/marker_timing_enabled.txt")" == "FLASHMLA_DELIVERY_MARKER_TIMING=1" ]]; then
  if PYTHONPATH="${site_root}:${PYTHONPATH:-}" python3 - <<'PY'
import flash_mla
assert flash_mla.__version__ == "1.0.0", flash_mla.__version__
assert hasattr(flash_mla, "flash_mla_sparse_fwd")
print("reusing FlashMLA", flash_mla.__version__)
PY
  then
    reuse_ok=1
  fi
fi
if [[ "${reuse_ok}" -eq 0 ]]; then
  # The delivery tree builds only the selected SM90 dense and sparse kernels.
  export MAX_JOBS="${MAX_JOBS:-32}"
  export NVCC_THREADS="${NVCC_THREADS:-8}"

  # T0005 rev3-②/rev5: the delivery binary must carry the in-kernel clock-marker channel
  # (296 marker scope; hardware timing and capture share this one instrumented binary).
  # setup.py turns the env flag into -DFLASHMLA_DELIVERY_MARKER_TIMING and adds
  # harness/override to the include path; without the flag the marker header is empty and
  # upstream SASS is reproduced bit-for-bit.
  export FLASHMLA_DELIVERY_MARKER_TIMING="${FLASHMLA_DELIVERY_MARKER_TIMING:-1}"

  # --- CUTLASS dependency: fetch the pinned commit at build time (see upstream/PINNED.md) ---
  cutlass_pin="$(sed -n 's/^- CUTLASS pinned revision: `\(.*\)`$/\1/p' "${root}/upstream/PINNED.md" | head -n 1)"
  [[ -n "${cutlass_pin}" ]] || { echo "cannot read CUTLASS pin from upstream/PINNED.md" >&2; exit 4; }
  artifact_data_root="${HOPPER_ARTIFACT_ROOT:-/path/to/hopper-prototype/artifacts}"
  cutlass_root="${artifact_data_root}/tool_cache/cutlass/${cutlass_pin}"
  if [[ ! -f "${cutlass_root}/include/cutlass/version.h" ]]; then
    echo "fetching pinned CUTLASS ${cutlass_pin} -> ${cutlass_root}"
    mkdir -p "${log_root}"
    fetch_tmp="$(mktemp -d "${TMPDIR:-/tmp}/cutlass_fetch.XXXXXX")"
    git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git "${fetch_tmp}/cutlass" \
      > "${log_root}/cutlass_fetch.stdout.txt" 2> "${log_root}/cutlass_fetch.stderr.txt"
    git -C "${fetch_tmp}/cutlass" checkout --detach "${cutlass_pin}" \
      >> "${log_root}/cutlass_fetch.stdout.txt" 2>> "${log_root}/cutlass_fetch.stderr.txt"
    rm -rf "${fetch_tmp}/cutlass/.git"
    mkdir -p "$(dirname "${cutlass_root}")"
    [[ -e "${cutlass_root}" ]] || mv "${fetch_tmp}/cutlass" "${cutlass_root}"
    rm -rf "${fetch_tmp}"
  fi
  [[ -f "${cutlass_root}/include/cutlass/version.h" ]] || {
    echo "pinned CUTLASS ${cutlass_pin} unavailable at ${cutlass_root} (clone/checkout failed; see ${log_root}/cutlass_fetch.stderr.txt)" >&2
    exit 4
  }
  echo "using CUTLASS ${cutlass_pin} from ${cutlass_root}" | tee "${log_root}/cutlass_source.txt"
  rm -f "${root}/upstream/csrc/cutlass"
  ln -s "${cutlass_root}" "${root}/upstream/csrc/cutlass"

  # The build image ships the CUDA compiler but not vendor-lib headers: ATen's
  # CUDAContextLight.h (included by the host TU csrc/api/dense_decode.cpp) needs cusparse.h,
  # cublas_v2.h, curand.h, ... which live in the pip nvidia-*-cu12 packages installed
  # alongside torch. Prepend every package include dir that exists to CPATH (gcc honors
  # CPATH; nvcc host compilation inherits it).
  if ! python3 -c 'from torch.utils.cpp_extension import CUDA_HOME; import os, sys; sys.exit(0 if os.path.isfile(os.path.join(CUDA_HOME or "", "include", "cusparse.h")) else 1)'; then
    vendor_incs="$(python3 - <<'PY'
import glob

incs = sorted(glob.glob('/usr/local/lib/python3.12/dist-packages/nvidia/*/include'))
print(':'.join(incs))
PY
)"
    [[ -n "${vendor_incs}" ]] || { echo "vendor CUDA headers missing from both CUDA_HOME/include and pip nvidia packages" >&2; exit 4; }
    echo "CUDA_HOME lacks vendor headers; CPATH += ${vendor_incs}" | tee "${log_root}/cpath_inject.txt"
    export CPATH="${vendor_incs}${CPATH:+:${CPATH}}"
  fi

  python3 -m pip install --no-build-isolation --no-deps --target "${site_root}" -v \
    "${root}/upstream" \
    > "${log_root}/flashmla_build.stdout.txt" 2> "${log_root}/flashmla_build.stderr.txt"
  printf '%s\n' "${expected_revision}" > "${build_root}/source_revision.txt"
  printf '%s\n' "FLASHMLA_DELIVERY_MARKER_TIMING=${FLASHMLA_DELIVERY_MARKER_TIMING}" > "${build_root}/marker_timing_enabled.txt"
fi

PYTHONPATH="${site_root}:${PYTHONPATH:-}" python3 - <<'PY'
import torch
import flash_mla
import flash_mla.cuda as flash_mla_cuda

import os

if os.environ.get("FLASHMLA_CPU_ONLY_BUILD") != "1":
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9, (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    )
assert flash_mla.__version__ == "1.0.0", flash_mla.__version__
assert hasattr(flash_mla, "get_mla_metadata") and hasattr(flash_mla, "flash_mla_with_kvcache")
assert hasattr(flash_mla, "flash_mla_sparse_fwd")
assert hasattr(flash_mla_cuda, "dense_decode_fwd")
assert hasattr(flash_mla_cuda, "sparse_decode_fwd")
assert hasattr(flash_mla_cuda, "sparse_prefill_fwd")
print(torch.__version__, torch.version.cuda, flash_mla.__version__)
PY

nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -shared -Xcompiler=-fPIC \
  "${root}/harness/globaltimer_timer.cu" \
  -o "${build_root}/libdelivery_globaltimer_timer.so" \
  > "${log_root}/globaltimer_timer_build.stdout.txt" \
  2> "${log_root}/globaltimer_timer_build.stderr.txt"

dump_tools="${HOPPER_ARTIFACT_ROOT:-/path/to/hopper-prototype/artifacts}/tool_cache/cuda_12.6.77_dump_tools"
nvdisasm_bin="${dump_tools}/cuda_nvdisasm-linux-x86_64-12.6.77-archive/bin"
cuobjdump_bin="${dump_tools}/cuda_cuobjdump-linux-x86_64-12.6.77-archive/bin"
if ! command -v nvdisasm >/dev/null 2>&1 \
  && { [[ -z "${NVDISASM_PATH:-}" ]] || [[ ! -x "${NVDISASM_PATH}" ]]; }; then
  [[ -x "${nvdisasm_bin}/nvdisasm" ]] || {
    echo "nvdisasm missing from PATH and dump-tools cache: ${nvdisasm_bin}" >&2
    exit 4
  }
  export PATH="${PATH:+${PATH}:}${nvdisasm_bin}"
  export NVDISASM_PATH="${nvdisasm_bin}/nvdisasm"
  echo "appended cached nvdisasm to PATH: ${nvdisasm_bin}" \
    | tee "${log_root}/nvdisasm_path.txt"
fi
if ! command -v cuobjdump >/dev/null 2>&1; then
  [[ -x "${cuobjdump_bin}/cuobjdump" ]] || {
    echo "cuobjdump missing from PATH and dump-tools cache: ${cuobjdump_bin}" >&2
    exit 4
  }
  export PATH="${PATH:+${PATH}:}${cuobjdump_bin}"
  echo "appended cached cuobjdump to PATH: ${cuobjdump_bin}" \
    | tee "${log_root}/cuobjdump_path.txt"
fi

cuobjdump --dump-sass "${build_root}/libdelivery_globaltimer_timer.so" \
  > "${log_root}/globaltimer_timer.sass"
grep -q 'SR_CLOCKLO' "${log_root}/globaltimer_timer.sass"
grep -q 'SR_GLOBALTIMERLO' "${log_root}/globaltimer_timer.sass"

# --- T0005 rev3-②/rev5: in-kernel marker SASS audit (296 marker scope) ---
# Every instrumented dense-decode kernel must contain at least one marker pair per compiled
# body: >=2 SR_CLOCKLO (entry+exit clock64) and >=2 SR_GLOBALTIMERLO (entry+exit globaltimer)
# CS2R reads, mirroring the DG static gate
# (tools/microbench/deepgemm_sm90_runner.py) and the CUTLASS
# capture-side marker-PC selection (opcode CS2R with SR_CLOCKLO operand, >=2 PCs).
flash_mla_so="$(ls -1 "${site_root}"/flash_mla/cuda*.so 2>/dev/null | head -n 1)"
[[ -n "${flash_mla_so}" ]] || { echo "flash_mla/cuda*.so not found under ${site_root}" >&2; exit 4; }
cuobjdump --dump-sass "${flash_mla_so}" > "${log_root}/flash_mla_cuda.sass"
python3 "${root}/scripts/marker_sass_audit.py" \
  "${log_root}/flash_mla_cuda.sass" "${log_root}/flashmla_marker_sass.json"
echo "flashmla marker SASS audit ok: ${log_root}/flashmla_marker_sass.json"
PYTHONPATH="${site_root}:${PYTHONPATH:-}" python3 -m pip freeze > "${log_root}/pip_freeze.txt"
echo "flashmla build ok: ${site_root}"
