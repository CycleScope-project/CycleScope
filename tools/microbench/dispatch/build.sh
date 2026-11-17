# dispatch: OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME] bash build.sh
set -e
SM_ARCH=${SM_ARCH:?}
OUT=${OUT:?}
NVCC=${NVCC:-nvcc}
TREE=${TREE:-$(cd "$(dirname "$0")/../../.." && pwd)}
mkdir -p "$OUT/bin" "$OUT/sass"
case "$SM_ARCH" in
  sm_90a|sm_103a) ;;
  *) echo "build.sh: unsupported SM_ARCH=$SM_ARCH" >&2; exit 2 ;;
esac
$NVCC -O2 -arch=$SM_ARCH -o "$OUT/bin/dispatch" "$TREE/tools/microbench/dispatch/probe/dispatch.cu"
cuobjdump -sass "$OUT/bin/dispatch" > "$OUT/sass/dispatch.sass"
echo "build.sh: dispatch done (SM_ARCH=$SM_ARCH)"
