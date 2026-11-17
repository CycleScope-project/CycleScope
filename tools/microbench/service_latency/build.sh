#!/usr/bin/env bash
# Build the service_latency probes (alu, alu2) for one SM architecture and
# dump SASS. Ops with no sm_90a realisation (f32x2 pipe, 3-input FMNMX3) are
# blanked out of a build-time source copy and listed in unrealised.txt.
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?OUT is required}
SM_ARCH=${SM_ARCH:?SM_ARCH is required (sm_90a or sm_103a)}
case "$SM_ARCH" in
  sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_} ;;
  *) echo "build.sh: unsupported SM_ARCH=$SM_ARCH" >&2; exit 2 ;;
esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}
  CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
  export PATH="$CUDA_HOME/bin:$PATH"
else
  NVCC=${NVCC:-nvcc}
  CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
fi
mkdir -p "$OUT/bin" "$OUT/sass"
SRC_DIR="$MECH_DIR/probe"
if [ "$SM_ARCH" = sm_90a ] || [ "$SM_ARCH" = sm_103a ]; then
  UNREALISED="FFMA2 FADD2 FMUL2 FMNMX3 HADD2.F32"
  if [ "$SM_ARCH" = sm_90a ]; then printf "%s\n" $UNREALISED > "$OUT/unrealised.txt"; else : > "$OUT/unrealised.txt"; fi
  SRC_DIR="$OUT/src"
  mkdir -p "$SRC_DIR"
  cp "$MECH_DIR/mb.cuh" "$OUT/mb.cuh"
  # sm_90a cannot realise the f32x2 pipe ops, 3-input FMNMX3, the whole
  # f32<-f16/bf16 mixed-precision family (FH*) and the k_mix/k_mixr cells
  # (h100-r1 emits only lat/thr variants). Generate a build-time source
  # copy with those classes removed; line count is preserved.
  python3 - "$MECH_DIR/probe" "$SRC_DIR" "$SM_ARCH" <<'PY'
import re, sys, pathlib
src, dst = map(pathlib.Path, sys.argv[1:3])
sm_arch = sys.argv[3]
ops = {"FFMA2", "FADD2", "FMUL2", "FMNMX3"}
ops |= set(re.findall(r"\bFH[A-Z0-9_]+", (src / "alu.cu").read_text()))
alt = "|".join(sorted(ops, key=len, reverse=True))
for name in ("alu.cu", "alu2.cu", "misc.cu", "svc.cu", "alu3.cu", "alu4.cu", "alu5.cu", "alu6.cu", "alu7.cu", "tmem_ops.cuh", "umma_impl.cuh"):
    t = (src / name).read_text()
    if name in ("alu.cu", "alu2.cu"):
        t = re.sub(r"^struct (?:FHADD|FHFMA|FHADD_BF16|FHFMA_BF16) \{.*?\n\};\n",
                   "", t, flags=re.S | re.M)
        t = re.sub(r"^template <[^\n]*\nvoid add_mixr?\(.*?\n\}\n", "", t,
                   flags=re.S | re.M)
    # 8-byte ops (DADD/DFMA/DMUL/B64) need 64KB of pinb in k_thr, over the
    # 48KB static-smem limit: skip their thr registration (h100-r1 has
    # lat.D* but no thr.D*).
    t = t.replace(
        "void add_thr(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {\n"
        "  constexpr int chains = single_chain_op<O>::value ? 1 : 8;\n"
        "  (v.push_back({std::string(UNI == 1 ? \"U.\" : UNI == 2 ? \"S0.\" : \"\") +\n"
        "                O::name, \"thr\", Ks, chains, &k_thr<O, Ks, UNI>}), ...);\n"
        "}\n",
        "void add_thr(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {\n"
        "  if constexpr (sizeof(typename O::T) <= 4) {\n"
        "  constexpr int chains = single_chain_op<O>::value ? 1 : 8;\n"
        "  (v.push_back({std::string(UNI == 1 ? \"U.\" : UNI == 2 ? \"S0.\" : \"\") +\n"
        "                O::name, \"thr\", Ks, chains, &k_thr<O, Ks, UNI>}), ...);\n"
        "  }\n"
        "}\n")
    if name not in ("alu.cu", "alu2.cu"):
        (dst / name).write_text(t)
        continue
    phys = t.splitlines(keepends=True)
    logical, cur = [], []
    for line in phys:
        cur.append(line)
        if not line.rstrip("\n").rstrip().endswith("\\"):
            logical.append(cur)
            cur = []
    if cur:
        logical.append(cur)
    out = []
    for group in logical:
        text = "".join(group)
        for op in ops:
            text = re.sub(
                rf"\b(?:X|MB_LONG_THR|MB_LONG_S0|MB_S0|MB_MIX3)\({op}\)", "", text)
        if re.search(r"\badd_mixr?\b|\bFIB_MIXED_TYPE\b|\bMB_MIX3\b", text):
            out.extend("\n" for _ in group)
        elif sm_arch == "sm_90a" and re.search(rf"\b(?:{alt})\b", text):
            out.extend("\n" for _ in group)
        else:
            out.append(text)
    (dst / name).write_text("".join(out))
PY
fi
# SVC_SM100 is the host-visible architecture switch for the sm_100+ only
# cells (f32x2 ops, tcgen05 / TMEM / UMMA kernels and their launch sites):
# __CUDA_ARCH__ is undefined in the host pass, so guarding kernel
# definitions with it leaves the host stub referencing undeclared kernels.
ARCH_DEFS=()
[ "$SM_ARCH" = sm_103a ] && ARCH_DEFS=(-DSVC_SM100=1)
for src in alu alu2 misc alu3 alu4 alu5 alu6 alu7; do
  "$NVCC" -O3 -std=c++17 -I"$OUT" "${ARCH_DEFS[@]}" \
      -gencode "arch=$COMPUTE,code=$SM_ARCH" \
      -lineinfo --expt-relaxed-constexpr \
      "$SRC_DIR/$src.cu" -o "$OUT/bin/$src" > "$OUT/sass/$src.ptxas.txt" 2>&1
  "$CUOBJDUMP" -sass "$OUT/bin/$src" > "$OUT/sass/$src.sass" 2>/dev/null || true
done
for part in 0 1 2 3; do
  # svc.cu launches its guard-patched cells through the driver API
  # (cuModuleLoadData / cuLaunchKernel): link libcuda as blackwell-prototype did
  "$NVCC" -O3 -std=c++17 -I"$OUT" -DSVC_PART=$part "${ARCH_DEFS[@]}" \
      -gencode "arch=$COMPUTE,code=$SM_ARCH" \
      -lineinfo --expt-relaxed-constexpr \
      "$SRC_DIR/svc.cu" -o "$OUT/bin/svc_p$part" -lcuda \
      > "$OUT/sass/svc_p$part.ptxas.txt" 2>&1 &
done
wait
for part in 0 1 2 3; do
  "$CUOBJDUMP" -sass "$OUT/bin/svc_p$part" \
      > "$OUT/sass/svc_p$part.sass" 2>/dev/null || true
done
echo "build.sh: service_latency done (SM_ARCH=$SM_ARCH)"
