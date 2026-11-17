#!/usr/bin/env bash

deepgemm_source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if project_root="$(git -C "${deepgemm_source_root}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  project_root="$(cd "${deepgemm_source_root}/../.." && pwd)"
fi
artifact_setting="${DEEPGEMM_DELIVERY_ARTIFACTS_ROOT:-${project_root}/artifacts/suites/deepgemm}"
if [[ "${artifact_setting}" = /* ]]; then
  deepgemm_artifact_root="${artifact_setting}"
else
  deepgemm_artifact_root="${project_root}/${artifact_setting}"
fi

export DEEPGEMM_DELIVERY_SOURCE_ROOT="${deepgemm_source_root}"
export DEEPGEMM_DELIVERY_ARTIFACTS_ROOT="${deepgemm_artifact_root}"
export HOPPER_PROJECT_ROOT="${project_root}"
