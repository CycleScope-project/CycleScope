set -e
SM_ARCH=${SM_ARCH:?}; OUT=${OUT:?}; NVCC=${NVCC:-nvcc}
TREE=${TREE:-$(cd "$(dirname "$0")/../../.." && pwd)}
mkdir -p "$OUT/bin" "$OUT/sass"
$NVCC -O2 -arch=$SM_ARCH -o "$OUT/bin/descvis" \
  "$TREE/tools/microbench/descriptor_visibility/probe/descvis.cu" -lcuda
cuobjdump -sass "$OUT/bin/descvis" > "$OUT/sass/descvis.sass"
echo "build.sh: descvis done (SM_ARCH=$SM_ARCH)"
