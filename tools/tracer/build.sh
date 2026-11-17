#!/usr/bin/env bash
# Build the merged tracer for ARCH=sm_90a|sm_103a.
#
#   ARCH=sm_103a CYCLESCOPE_NVBIT_DIR=<nvbit release root> bash build.sh
#
# Output: $CYCLESCOPE_ARTIFACT_ROOT/build/tracer/<ARCH>/tracer.so plus
# build-manifest.json (input source sha256, nvcc/NVBit versions, libcuda
# provenance, getSassBinary path).  The NVBit release is detected from
# core/nvbit.h's NVBIT_VERSION; 1.8 keeps Instr::getSassBinary, older
# releases fall back to the device code-segment reader
# (CYCLESCOPE_NVBIT_HAS_SASS_BINARY=0, recorded in the manifest).
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
ARCH="${ARCH:-sm_103a}"
root="${CYCLESCOPE_ARTIFACT_ROOT:-$(PYTHONPATH="$src/../../python" python3 -m cyclescope.site --get artifact_root)}"
out="$root/build/tracer/$ARCH"
mkdir -p "$out"

command -v nvcc >/dev/null || { echo "nvcc not in PATH" >&2; exit 2; }
mem_avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
[ "$mem_avail" -ge 8192 ] || { echo "MemAvailable ${mem_avail}MiB < 8GiB, abort" >&2; exit 3; }

case "$ARCH" in
  sm_90a|sm_103a) ;;
  *) echo "unsupported ARCH=$ARCH (sm_90a|sm_103a)" >&2; exit 2 ;;
esac

nvbit_dir="${CYCLESCOPE_NVBIT_DIR:?CYCLESCOPE_NVBIT_DIR must point at the NVBit release root (core/nvbit.h, core/libnvbit.a)}"
[ -s "$nvbit_dir/core/libnvbit.a" ] || { echo "libnvbit.a missing: $nvbit_dir/core" >&2; exit 2; }
[ -s "$nvbit_dir/core/nvbit.h" ] || { echo "nvbit.h missing: $nvbit_dir/core" >&2; exit 2; }
nvbit_version="$(grep -o 'NVBIT_VERSION "[^"]*"' "$nvbit_dir/core/nvbit.h" | cut -d'"' -f2)"

sass_binary=1
case "$nvbit_version" in
  1.8*|1.9*|2.*) sass_binary=1 ;;
  *) sass_binary=0 ;;
esac

nvcc_bin="$(command -v nvcc)"
cuda_root="$(cd "$(dirname "$nvcc_bin")/.." && pwd)"

# Driver library or the CUDA toolkit's official stub is only a link
# input; no rpath/LD_LIBRARY_PATH is set.
libcuda=""
libcuda_mode=""
IFS=: read -r -a driver_env_dirs <<< "${LD_LIBRARY_PATH:-}"
for d in /usr/local/nvidia/lib64 /usr/lib/x86_64-linux-gnu "$cuda_root/lib64" "$cuda_root/lib" "${driver_env_dirs[@]}"; do
  [ -n "$d" ] || continue
  c="$d/libcuda.so.1"
  if [ -s "$c" ] && readelf -d "$c" 2>/dev/null | grep -q 'SONAME.*\[libcuda\.so\.1\]'; then
    libcuda="$c"; libcuda_mode="driver"; break
  fi
done
if [ -z "$libcuda" ]; then
  for d in "$cuda_root/lib64/stubs" "$cuda_root/targets/x86_64-linux/lib/stubs"; do
    c="$d/libcuda.so"
    if [ -s "$c" ] && readelf -d "$c" 2>/dev/null | grep -q 'SONAME.*\[libcuda\.so\.1\]'; then
      libcuda="$c"; libcuda_mode="cuda-toolkit-stub"; break
    fi
  done
fi
[ -n "$libcuda" ] || { echo "CUDA driver library / official toolkit stub missing" >&2; exit 2; }
NVCC_PATH="-L $cuda_root/lib64 -L $cuda_root/lib"

log="$out/build.log"
{
  echo "== tracer build.sh $(date -u +%FT%TZ)"
  nvcc --version | tail -1
  g++ --version | head -1
  echo "ARCH=$ARCH OBJDIR=$out NVBIT=$nvbit_dir ($nvbit_version) sass_binary=$sass_binary"
  echo "src=$src"
  echo "libcuda=$libcuda mode=$libcuda_mode"
  sha256sum "$nvbit_dir/core/libnvbit.a"
} >"$log"

rc=0
nice -n 15 make -B -j1 -C "$src" ARCH="$ARCH" CXX=g++ \
  NVBIT_ROOT="$nvbit_dir" NVBIT_PATH="$nvbit_dir/core" \
  NVBIT_LIB_DIR="$nvbit_dir/core" \
  CYCLESCOPE_NVBIT_HAS_SASS_BINARY="$sass_binary" \
  NVCC_PATH="$NVCC_PATH" CUDA_DRIVER_LIB="$libcuda" \
  EXTRA_DEFINES="${CYCLESCOPE_TRACER_EXTRA_DEFINES:-}" \
  OBJDIR="$out" tracer.so >>"$log" 2>&1 || rc=$?

tool="$out/tracer.so"

# DT_NEEDED libcuda.so.1 must be present or injection is silently skipped.
if [ $rc -eq 0 ]; then
  if ! readelf -d "$tool" | grep -q 'Shared library: \[libcuda\.so\.1\]'; then
    echo "FATAL: $tool has no DT_NEEDED libcuda.so.1" | tee -a "$log" >&2
    rc=4
  fi
fi

if [ $rc -eq 0 ]; then
  for sym in nvbit_at_cuda_event nvbit_at_ctx_init nvbit_at_ctx_term; do
    if ! nm -D "$tool" | grep " $sym\$" >/dev/null; then
      echo "FATAL: export missing: $sym" | tee -a "$log" >&2
      rc=5
    fi
  done
fi

if [ $rc -eq 0 ]; then
  python3 - "$src" "$out" "$libcuda_mode" "$ARCH" "$nvbit_dir" "$nvbit_version" "$sass_binary" "$libcuda" "${CYCLESCOPE_TRACER_EXTRA_DEFINES:-}" >>"$log" 2>&1 <<'PY' || rc=6
import hashlib, json, re, subprocess, sys
from pathlib import Path

src, out = map(Path, sys.argv[1:3])
libcuda_mode, arch, nvbit_dir, nvbit_version, sass_binary, libcuda, extra_defines = sys.argv[3:10]
inputs = {}
for p in sorted(src.iterdir()):
    if p.is_file() and (p.name == "Makefile" or p.suffix in (".cu", ".h", ".cpp", ".sh")):
        inputs[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
inputs["libnvbit.a"] = hashlib.sha256((Path(nvbit_dir) / "core/libnvbit.a").read_bytes()).hexdigest()
nvcc_v = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
gxx_v = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
tool = out / "tracer.so"
readelf = subprocess.run(["readelf", "-d", str(tool)], capture_output=True, text=True).stdout
needed = [ln.split("[")[1].rstrip("]") for ln in readelf.splitlines() if "NEEDED" in ln]
nm_out = subprocess.run(["nm", "-D", "--undefined-only", str(tool)],
                        capture_output=True, text=True).stdout
undef_cu = sorted(m.group(1) for ln in nm_out.splitlines() if (m := re.match(r"\s*U (cu\w+)$", ln)))
manifest = {
    "utc": subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True, text=True).stdout.strip(),
    "src_dir": str(src), "arch": arch, "nvcc": nvcc_v, "gxx": gxx_v,
    "nvbit_dir": nvbit_dir, "nvbit_version": nvbit_version,
    "sass_binary_path": "getSassBinary" if sass_binary == "1" else "device_code_segment_read",
    "libcuda": libcuda, "libcuda_mode": libcuda_mode, "input_sha256": inputs,
    "extra_defines": extra_defines,
    "tracer_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
    "dt_needed": needed, "undefined_cu_symbols": undef_cu,
    "needed_libcuda_so_1": "libcuda.so.1" in needed,
}
(out / "build-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print("manifest:", json.dumps({k: manifest[k] for k in
      ("tracer_sha256", "nvbit_version", "sass_binary_path", "libcuda", "needed_libcuda_so_1")}))
PY
fi

echo "$rc" > "$out/build_exit.txt"
echo "$(date -u +%FT%TZ) exit=$rc" >>"$log"
exit "$rc"
