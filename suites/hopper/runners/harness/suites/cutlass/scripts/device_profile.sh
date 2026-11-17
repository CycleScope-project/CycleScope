#!/usr/bin/env bash

delivery_check_gpu_profile() {
  local profile="$1"
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  case "${profile}" in
    h20) [[ "${gpu_name}" == *"NVIDIA H20"* && "${gpu_name}" != *"NVIDIA H200"* ]] ;;
    h100) [[ "${gpu_name}" == *"NVIDIA H100"* ]] ;;
    h200) [[ "${gpu_name}" == *"NVIDIA H200"* ]] ;;
    *) echo "GPU profile must be h20, h100, or h200" >&2; return 2 ;;
  esac || { echo "expected ${profile}, got ${gpu_name}" >&2; return 3; }
  printf '%s\n' "${gpu_name}"
}

delivery_write_gpu_metadata() {
  local output_dir="$1"
  mkdir -p "${output_dir}"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory \
    --format=csv,noheader > "${output_dir}/gpu_metadata.csv"
  nvidia-smi -q > "${output_dir}/nvidia_smi_q.txt"
}
