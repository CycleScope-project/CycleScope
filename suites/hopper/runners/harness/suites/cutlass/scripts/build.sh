#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 h20|h100|h200 SHARD(0..3)" >&2
  exit 2
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
profile="$1"
shard="$2"
exec bash "${CUTLASS_DELIVERY_SOURCE_ROOT}/harness/build_algorithm_shard.sh" "${shard}" "${profile}"
