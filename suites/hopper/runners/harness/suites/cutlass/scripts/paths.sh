#!/usr/bin/env bash

cutlass_source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if project_root="$(git -C "${cutlass_source_root}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  project_root="$(cd "${cutlass_source_root}/../.." && pwd)"
fi
artifact_setting="${CUTLASS_DELIVERY_ARTIFACTS_ROOT:-${project_root}/artifacts/suites/cutlass}"
if [[ "${artifact_setting}" = /* ]]; then
  cutlass_artifact_root="${artifact_setting}"
else
  cutlass_artifact_root="${project_root}/${artifact_setting}"
fi

export CUTLASS_DELIVERY_SOURCE_ROOT="${cutlass_source_root}"
export CUTLASS_DELIVERY_ARTIFACTS_ROOT="${cutlass_artifact_root}"
export HOPPER_PROJECT_ROOT="${project_root}"
