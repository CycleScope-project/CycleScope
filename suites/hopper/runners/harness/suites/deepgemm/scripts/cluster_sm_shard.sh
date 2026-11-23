#!/usr/bin/env bash
set -euo pipefail
# Inner H100 worker: 4-16 workers/shard prepare/queue; GPU section flock-serial.
WORK="${CLUSTER_SM_WORK:-/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920/deepgemm}"
SHARD="${CLUSTER_SM_SHARD:?}"
SHARDS="${CLUSTER_SM_SHARDS:-8}"
ROLE="${CLUSTER_SM_ROLE:-run}"
mkdir -p "$WORK/logs" "$WORK/tmp"
export TMPDIR="$WORK/tmp/shard${SHARD}"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export GEMM_DELIVERY_GPU=h100
export CLUSTER_SM_WORK="$WORK"
export DEEPGEMM_DELIVERY_ARTIFACTS_ROOT="$WORK"
export DEEPGEMM_DELIVERY_RAW_PER_SAMPLE="${DEEPGEMM_DELIVERY_RAW_PER_SAMPLE:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"  # compile slots; GPU kernels remain flock-serial
export PATH="/path/to/hopper-prototype/artifacts/cluster_sm_296_flow_20260921/flashinfer/build/site-packages-fi0617-cutlass470-tvmffi0113/nvidia/cu13/bin:/path/to/hopper-prototype/artifacts/tool_cache/cuda_12.6.77_dump_tools/cuda_cuobjdump-linux-x86_64-12.6.77-archive/bin:${PATH}"
exec > >(tee "$WORK/logs/${ROLE}.shard${SHARD}.log") 2>&1
trap 'rc=$?; echo "$rc" > "$WORK/logs/${ROLE}.shard${SHARD}.exit"; chown -R 10250:10250 "$WORK" 2>/dev/null || true; exit "$rc"' EXIT

python3 - <<'PY'
from pathlib import Path
import site
work = Path("/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920/deepgemm")
link_dir = work / "cuda-links"
link_dir.mkdir(parents=True, exist_ok=True)
libs = [p for base in site.getsitepackages() for p in Path(base).glob("nvidia/*/lib/libnvrtc.so.*") if p.is_file()]
if libs:
    link = link_dir / "libnvrtc.so"
    if not link.exists():
        link.symlink_to(sorted(libs)[0])
    print("NVRTC", link.resolve(), flush=True)
PY
export LIBRARY_PATH="${WORK}/cuda-links${LIBRARY_PATH:+:${LIBRARY_PATH}}"

cd /path/to/hopper-prototype
if [[ "${ROLE}" == "build" ]]; then
  bash suites/deepgemm/scripts/build_cluster_sm.sh h100
  exit 0
fi
if [[ "${ROLE}" == "finalize" ]]; then
  export CLUSTER_SM_FINALIZE=1
  python3 suites/deepgemm/scripts/cluster_sm_qualify.py \
    --manifest suites/deepgemm/manifests/deepgemm_cluster_sm100.json \
    --source-manifest suites/deepgemm/manifests/deepgemm_100.json \
    --work "$WORK" \
    --package "$(cat "$WORK/build/marker_package.path")" \
    --output "$WORK/qualification.json"
  cp -f "$WORK/qualification.json" /path/to/hopper-prototype/artifacts/cluster_sm_400_20260920/deepgemm/qualification.json
  exit 0
fi

export CLUSTER_SM_SHARDS="$SHARDS"
# >8 cases: 4-16 workers/shard. 4 workers prepare/queue; GPU section is flock-serial.
# CLUSTER_SM_SHARD_LIST=2,3,4 allows one GPU job to drain leftover shards when quota is tight.
WORKERS="${CLUSTER_SM_WORKERS:-4}"
if [[ "${WORKERS}" -lt 4 || "${WORKERS}" -gt 16 ]]; then
  echo "CLUSTER_SM_WORKERS=${WORKERS} outside 4-16; using 4" >&2
  WORKERS=4
fi
export CLUSTER_SM_WORKERS="${WORKERS}"
export CLUSTER_SM_GPU_LOCK="${WORK}/tmp/gpu.lock"
mkdir -p "${WORK}/tmp" "${WORK}/logs"
python3 - "${WORK}" "${SHARDS}" "${WORKERS}" "${SHARD}" "${CLUSTER_SM_SHARD_LIST:-}" <<'PY'
import json, sys
from pathlib import Path
work, shards, workers, shard, shard_list = sys.argv[1:]
payload = {
    "outer_shards": int(shards),
    "inner_workers": int(workers),
    "inner_workers_allowed": [4, 16],
    "gpu_section": "flock_serial",
    "prepare_queue_workers": int(workers),
    "shard": shard,
    "shard_list": shard_list,
}
path = Path(work) / "logs" / "concurrency.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    f"concurrency outer_shards={shards} inner_workers={workers} "
    f"gpu_lock=serial shard={shard} shard_list={shard_list}",
    flush=True,
)
PY
if [[ -n "${CLUSTER_SM_SHARD_LIST:-}" ]]; then
  IFS=',' read -r -a shard_ids <<< "${CLUSTER_SM_SHARD_LIST}"
  for shard_id in "${shard_ids[@]}"; do
    export CLUSTER_SM_SHARD="${shard_id}"
    echo "drain leftover shard=${shard_id}"
    bash suites/deepgemm/scripts/run_cluster_sm.sh h100
  done
else
  export CLUSTER_SM_SHARD="$SHARD"
  bash suites/deepgemm/scripts/run_cluster_sm.sh h100
fi
