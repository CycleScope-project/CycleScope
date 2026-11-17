#!/usr/bin/env bash

flashinfer_source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if project_root="$(git -C "${flashinfer_source_root}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  project_root="$(cd "${flashinfer_source_root}/../.." && pwd)"
fi
artifact_setting="${FLASHINFER_DELIVERY_ARTIFACTS_ROOT:-${project_root}/artifacts/suites/flashinfer}"
if [[ "${artifact_setting}" = /* ]]; then
  flashinfer_artifact_root="${artifact_setting}"
else
  flashinfer_artifact_root="${project_root}/${artifact_setting}"
fi
export FLASHINFER_DELIVERY_SOURCE_ROOT="${flashinfer_source_root}"
export FLASHINFER_DELIVERY_ARTIFACTS_ROOT="${flashinfer_artifact_root}"
export HOPPER_PROJECT_ROOT="${project_root}"

flashinfer_export_cuda_wheel_paths() {
  local resolved includes libs
  resolved="$(python3 - "${1:-}" <<'PY'
import os
from pathlib import Path
import site
import sys
import tempfile

roots = [Path(p) for p in site.getsitepackages()]
if sys.argv[1]:
    roots.append(Path(sys.argv[1]))
includes = list(dict.fromkeys(str(p.resolve()) for root in roots
    for p in sorted(root.glob("nvidia/*/include")) if p.is_dir()))
libs = [p.resolve() for root in roots for pattern in ("nvidia/*/lib", "nvidia/*/lib64")
    for p in sorted(root.glob(pattern)) if p.is_dir()]
for raw in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
    if raw:
        home = Path(raw)
        libs.extend(p for suffix in ("lib64", "lib", "targets/x86_64-linux/lib")
                    if (p := home / suffix).is_dir())
libs.extend(Path(p) for p in os.environ.get("LIBRARY_PATH", "").split(":") if p)
libs.extend(Path(p) for p in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64"))
libs = list(dict.fromkeys(p.resolve() for p in libs if p.is_dir() and "stubs" not in p.parts))
nvrtc = next((p.resolve() for directory in libs
              for p in [directory / "libnvrtc.so", *sorted(directory.glob("libnvrtc.so.*"))]
              if p.is_file()), None)
if nvrtc is None:
    raise SystemExit("CUDA NVRTC library not found in CUDA wheels, Toolkit, or LIBRARY_PATH")
link_root = (Path(os.environ["FLASHINFER_DELIVERY_ARTIFACTS_ROOT"]) / "build/cuda-link").resolve()
link_root.mkdir(parents=True, exist_ok=True)
link = link_root / "libnvrtc.so"
if not link.is_symlink() or link.resolve() != nvrtc:
    # Parallel shards share this directory: replacement must not leave a link gap.
    with tempfile.TemporaryDirectory(dir=link_root) as tmp:
        staged = Path(tmp) / "libnvrtc.so"
        staged.symlink_to(nvrtc)
        staged.replace(link)
print(":".join(includes))
print(":".join(dict.fromkeys([str(link_root), *(str(p) for p in libs)])))
PY
)" || return 1
  includes="${resolved%%$'\n'*}"
  libs="${resolved#*$'\n'}"
  if [[ -n "${includes}" ]]; then
    export CPATH="${includes}${CPATH:+:${CPATH}}"
  fi
  export LIBRARY_PATH="${libs}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}
