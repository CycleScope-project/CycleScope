#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 DELIVERY_GROUP(0..3) [h20|h100|h200]" >&2
  exit 2
fi

delivery_group="$1"
gpu_profile="${2:-h100}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)/paths.sh"
cutlass_root="${CUTLASS_DELIVERY_SOURCE_ROOT}"
build_root="${CUTLASS_DELIVERY_ARTIFACTS_ROOT}/build"
log_root="${CUTLASS_DELIVERY_ARTIFACTS_ROOT}/logs/build"
build_target="${CUTLASS_DELIVERY_BUILD_TARGET:-}"
python3 "${cutlass_root}/../common/work_tree.py" check \
  --source "${cutlass_root}" --output "${CUTLASS_DELIVERY_ARTIFACTS_ROOT}"

case "${delivery_group}" in
  0|1|2|3) ;;
  *) echo "invalid delivery group: ${delivery_group}" >&2; exit 2 ;;
esac
manifest="${CUTLASS_DELIVERY_BUILD_MANIFEST:-${cutlass_root}/manifests/cutlass_100.json}"
[[ -f "${manifest}" ]] || { echo "missing CUTLASS build manifest: ${manifest}" >&2; exit 4; }

mkdir -p "${build_root}" "${log_root}"
trap 'chown -R 10250:10250 "${build_root}" "${log_root}" 2>/dev/null || true' EXIT

if [[ "${CUTLASS_DELIVERY_COMPILE_ONLY:-0}" == "1" ]]; then
  echo "compile-only CPU cluster build; GPU identity will be checked by the run script" \
    > "${log_root}/compile_environment_group${delivery_group}.txt"
else
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  case "${gpu_profile}" in
    h20) [[ "${gpu_name}" == *"NVIDIA H20"* && "${gpu_name}" != *"NVIDIA H200"* ]] ;;
    h100) [[ "${gpu_name}" == *"NVIDIA H100"* ]] ;;
    h200) [[ "${gpu_name}" == *"NVIDIA H200"* ]] ;;
    *) echo "unsupported GPU profile: ${gpu_profile}" >&2; exit 2 ;;
  esac || echo "WARNING: expected ${gpu_profile}, got: ${gpu_name}" >&2

  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory \
    --format=csv,noheader > "${log_root}/gpu_group${delivery_group}.csv"
  nvidia-smi -q > "${log_root}/nvidia_smi_q_group${delivery_group}.txt"
fi
nvcc --version > "${log_root}/nvcc_version_group${delivery_group}.txt"

# CPU compile images may ship nvcc without nvdisasm. Prefer the delivery dump tools.
if ! command -v nvdisasm >/dev/null 2>&1; then
  for _dump_bin in \
    /path/to/hopper-prototype/artifacts/cluster_sm_296_flow_20260921/flashinfer/build/site-packages-fi0617-cutlass470-tvmffi0113/nvidia/cu13/bin \
    /path/to/hopper-prototype/artifacts/tool_cache/cuda_12.6.77_dump_tools/cuda_cuobjdump-linux-x86_64-12.6.77-archive/bin
  do
    if [[ -x "${_dump_bin}/nvdisasm" || -x "${_dump_bin}/cuobjdump" ]]; then
      PATH="${_dump_bin}:${PATH}"
    fi
  done
  export PATH
fi

common=(
  -std=c++17
  -O3
  -lineinfo
  -arch=sm_90a
  --expt-relaxed-constexpr
  --expt-extended-lambda
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
  -I"${cutlass_root}"
  -I"${cutlass_root}/harness/override"
  -I"${cutlass_root}/upstream"
  -I"${cutlass_root}/upstream/include"
  -I"${cutlass_root}/upstream/tools/util/include"
  -I"${cutlass_root}/upstream/examples/common"
  -I"${cutlass_root}/upstream/examples/53_hopper_gemm_permute"
)

build_example() {
  local name="$1"
  local source="$2"
  local special_macro=""
  local extra_defines=()
  if [[ $# -ge 3 ]]; then
    special_macro="$3"
    extra_defines=("${@:4}")
  fi
  local output="${build_root}/${name}"
  local log="${log_root}/${name}.log"
  local extra=()
  if [[ ! -s "${cutlass_root}/${source}" ]]; then
    echo "manifest-selected CUTLASS source is missing: ${cutlass_root}/${source}" >&2
    return 4
  fi
  if [[ -n "${build_target}" && "${build_target}" != "${name}" ]]; then
    echo "skipping ${name}; requested build target is ${build_target}"
    return 0
  fi
  if [[ "${SKIP_EXISTING:-0}" == "1" && -x "${output}" ]]; then
    echo "keeping existing ${name}"
    return 0
  fi
  if [[ -n "${special_macro}" ]]; then
    extra+=("-D${special_macro}=1")
  fi
  extra+=("${extra_defines[@]}")
  echo "building ${name} from ${source}"
  nvcc "${common[@]}" "${extra[@]}" \
    -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 \
    "-DCUTLASS_DELIVERY_EXAMPLE=\"${source}\"" \
    "${cutlass_root}/harness/cutlass_example_cycle_wrapper.cu" \
    -lcuda -o "${output}" > "${log}" 2>&1
  cuobjdump -sass "${output}" > "${output}.sass"
}

build_dense() {
  local output="${build_root}/dense_algorithms"
  local log="${log_root}/dense_algorithms.log"
  if [[ -n "${build_target}" && "${build_target}" != "dense_algorithms" ]]; then
    echo "skipping dense_algorithms; requested build target is ${build_target}"
    return 0
  fi
  if [[ "${SKIP_EXISTING:-0}" == "1" && -x "${output}" ]]; then
    echo "keeping existing dense_algorithms"
    return 0
  fi
  echo "building dense_algorithms"
  nvcc "${common[@]}" \
    -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 \
    "${cutlass_root}/harness/cutlass_full_gpu_runner.cu" \
    -lcuda -o "${output}" > "${log}" 2>&1
  cuobjdump -sass "${output}" > "${output}.sass"
}

requirements_file="${log_root}/manifest_group${delivery_group}_requirements.tsv"
python3 "${cutlass_root}/harness/group_inputs.py" \
  --manifest "${manifest}" --group "${delivery_group}" > "${requirements_file}"
test -s "${requirements_file}"

while IFS=$'\t' read -r name source kind; do
  [[ -n "${name}" && -n "${source}" && -n "${kind}" ]] || {
    echo "malformed CUTLASS group requirement: ${name} ${source} ${kind}" >&2
    exit 4
  }
  case "${name}" in
    dense_algorithms)
      [[ "${source}" == "harness/cutlass_full_gpu_runner.cu" && "${kind}" == "dense_harness" ]]
      build_dense
      ;;
    example49_collective_builder)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE49
      ;;
    example50_epilogue_swizzle)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE50
      ;;
    delivery53_permute_full_gpu)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE53
      ;;
    example54_fp8_fused)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE54
      ;;
    example67_blockwise)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE67
      ;;
    example67_groupwise_streamk)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE67_GROUPWISE
      ;;
    example68_grouped_blockwise_pingpong)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE68_SPARSE_GROUPS
      ;;
    example69_grouped_mixed)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE69_MIXED
      ;;
    example69_grouped_int4_bf16)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE69_INT4_BF16
      ;;
    example69_grouped_int4_fp8)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE69_INT4_FP8
      ;;
    example68_grouped_blockwise_coop)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE68
      ;;
    example52_gather_scatter)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE52
      ;;
    example55_mixed_f16_fp8)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE55_F16_FP8
      ;;
    example55_int4_bf16)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE55_INT4_BF16
      ;;
    example55_int4_fp8)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE55_INT4_FP8
      ;;
    example56_ptr_array)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE56
      ;;
    example57_grouped_fastacc)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE57
      ;;
    example61_topk_softmax)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE61
      ;;
    example62_sparse)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE62
      ;;
    example63_weight_prefetch)
      build_example "${name}" "${source}" CUTLASS_DELIVERY_EXAMPLE63
      ;;
    *)
      echo "unsupported manifest-selected CUTLASS target: ${name}" >&2
      exit 4
      ;;
  esac
done < "${requirements_file}"

echo "build delivery group ${delivery_group} complete"
