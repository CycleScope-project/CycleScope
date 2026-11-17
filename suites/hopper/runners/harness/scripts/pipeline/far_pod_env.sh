# Shared pod-side environment for far-dev runs (all suites).
# Mirrors the pinned-runtime setup the old run_cluster_sm.sh drivers did
# around each harness call. Source before running cases.
REPO="${CLUSTER_SM_REPO:?}"
FI_SITE="/path/to/hopper-prototype/artifacts/cluster_sm_296_flow_20260921/flashinfer/build/site-packages-fi0617-cutlass470-tvmffi0113"
# image ships flashinfer-jit-cache 0.6.6; pinned 0.6.17 refuses it --
# uninstall like the old driver did (version check stays enabled).
python3 -m pip uninstall -y flashinfer-jit-cache >/dev/null 2>&1 || true
# pip may leave a namespace-package remnant (module dir without
# dist-info); flashinfer 0.6.17 then imports it and crashes on
# __version__.  Drop the directory outright.
python3 - <<'PY'
import importlib.util, os, shutil
spec = importlib.util.find_spec("flashinfer_jit_cache")
loc = None
if spec is not None:
    loc = spec.submodule_search_locations
    loc = list(loc) if loc else ([spec.origin] if spec.origin else [])
for d in loc or []:
    shutil.rmtree(d, ignore_errors=True) if os.path.isdir(d) else os.unlink(d)
PY
export PYTHONPATH="$REPO/suites/common/flashinfer_runtime_compat:$FI_SITE/nvidia_cutlass_dsl/dsl_packages:$FI_SITE${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$FI_SITE/nvidia/cu13/bin:$PATH"

# flashinfer JIT compiles against the CUDA wheel headers/libs, like the
# old driver: export CPATH/LIBRARY_PATH/LD_LIBRARY_PATH from
# flashinfer_export_cuda_wheel_paths (needs the case env's ARTIFACTS_ROOT).
export FLASHINFER_DELIVERY_ARTIFACTS_ROOT="${FLASHINFER_DELIVERY_ARTIFACTS_ROOT:-/path/to/hopper-prototype/artifacts/cluster_sm_296_flow_20260921/flashinfer}"
source "$REPO/suites/flashinfer/scripts/paths.sh"
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${FLASHINFER_DELIVERY_ARTIFACTS_ROOT}/build/cubins}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export NINJAFLAGS="${NINJAFLAGS:--j2}"
flashinfer_export_cuda_wheel_paths "$FI_SITE" || true
