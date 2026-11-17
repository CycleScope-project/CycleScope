#!/usr/bin/env bash

flashmla_source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if project_root="$(git -C "${flashmla_source_root}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  project_root="$(cd "${flashmla_source_root}/../.." && pwd)"
fi
artifact_setting="${FLASHMLA_DELIVERY_ARTIFACTS_ROOT:-${project_root}/artifacts/suites/flashmla}"
if [[ "${artifact_setting}" = /* ]]; then
  flashmla_artifact_root="${artifact_setting}"
else
  flashmla_artifact_root="${project_root}/${artifact_setting}"
fi

export FLASHMLA_DELIVERY_SOURCE_ROOT="${flashmla_source_root}"
export FLASHMLA_DELIVERY_ARTIFACTS_ROOT="${flashmla_artifact_root}"
export HOPPER_PROJECT_ROOT="${project_root}"
