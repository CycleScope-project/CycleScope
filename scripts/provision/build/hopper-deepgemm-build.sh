#!/usr/bin/env bash
# hopper-deepgemm-build: DeepGEMM sm90 marker delivery package.
# evidence: c450v2_build_pod.sh L44-50
#   (GEMM_DELIVERY_GPU=h100 build_deepgemm_full7_package.py
#    --source <upstream> --output marker_package --work-root package_work),
#   L52-58 cuda-links libnvrtc shim; deepgemm/scripts/build.sh L30 nvcc log.
# inputs: hopper-deepgemm-source (upstream snapshot), harness mirror.
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-deepgemm-build
H=$CYCLESCOPE_ROOT/suites/hopper/runners/harness
S=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-deepgemm-source
rm -rf "$T"
rm -rf "$T"
mkdir -p "$T/cuda-links" "$T/logs"
rm -rf "$T/marker_package" "$T/package_work"
cuda_inc="$(python3 - <<'PY'
import site
from pathlib import Path
print(":".join(str(p) for r in site.getsitepackages()
               for p in sorted(Path(r).glob("nvidia/*/include")) if p.is_dir()))
PY
)"
export CPATH="${cuda_inc}${CPATH:+:${CPATH}}"
cuda_lib="$(python3 - <<'PY'
import site
from pathlib import Path
print(":".join(str(p) for r in site.getsitepackages()
               for p in sorted(Path(r).glob("nvidia/*/lib")) if p.is_dir()))
PY
)"
export LD_LIBRARY_PATH="${cuda_lib}:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python3 - "$T/cuda-links" <<'PY'
import site, sys
from pathlib import Path
libs = [p for base in site.getsitepackages()
        for p in Path(base).glob("nvidia/*/lib/libnvrtc.so.*") if p.is_file()]
link = Path(sys.argv[1]) / "libnvrtc.so"
link.unlink(missing_ok=True); link.symlink_to(libs[0])
PY
export LIBRARY_PATH="$T/cuda-links:${cuda_lib}:/usr/local/cuda/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
nvcc --version > "$T/logs/nvcc_version.txt"                    # evidence: build.sh:30
export PYTHONPATH="$H:${PYTHONPATH:-}"
export S T
(cd "$H/suites/deepgemm/harness" && \
  GEMM_DELIVERY_GPU=h100 H="$H" python3 - <<'PYEOF'
import os, runpy, sys
sys.path.insert(0, os.environ["H"])
sys.argv = ["build_deepgemm_full7_package.py",
            "--source", os.environ["S"],
            "--output", os.environ["T"] + "/marker_package",
            "--work-root", os.environ["T"] + "/package_work"]
runpy.run_path("build_deepgemm_full7_package.py", run_name="__main__")
PYEOF
) > "$T/logs/build.log" 2>&1   # evidence: c450v2_build_pod.sh:44-50
test -f "$T/marker_package/deep_gemm/sm90_suite_marker.json"   # evidence: c450v2_build_pod.sh:53-54
