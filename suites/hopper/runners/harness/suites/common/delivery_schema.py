"""Shared 4x100 cases/DELIVERY public fields. Library extras stay on the payload."""

from __future__ import annotations


CASES_SCHEMA = "hopper.delivery.cases.v1"
PACKAGE_SCHEMA = "hopper.delivery.v1"
ARCHITECTURE = "sm_90a"
DELIVERY_GROUPS = 4
GROUP_ASSIGNMENT = "delivery_index modulo 4"
RUNNER = "scripts/run_100.sh"
CASE_FIELDS = ("case_id", "delivery_index", "delivery_group", "mechanisms")
DELIVERY_CORE = ("groups", "group_assignment", "runner")
HOPPER_PROFILES = ["h20", "h100", "h200"]
NOT_QUALIFIED = "not_two_round_qualified"
FA3_QUALIFIED = "qualified_h100_two_rounds"

SUITES = {
    "deepgemm": {
        "cases_name": "DeepGEMM Hopper 100-case mechanism-first set",
        "package_name": "DeepGEMM Hopper full-GPU delivery tests",
        "manifest": "manifests/deepgemm_100.json",
        "case_set": "deepgemm_100",
        "env": "DEEPGEMM_DELIVERY_ARTIFACTS_ROOT",
        "supported": list(HOPPER_PROFILES),
        "validated": [],
        "qualification_status": NOT_QUALIFIED,
        "historical": ("deepgemm_single_sm", "deepgemm_dual_sm"),
    },
    "cutlass": {
        "cases_name": "CUTLASS Hopper 100-case mechanism-first set",
        "package_name": "CUTLASS Hopper full-GPU delivery tests",
        "manifest": "manifests/cutlass_100.json",
        "case_set": "cutlass_100",
        "env": "CUTLASS_DELIVERY_ARTIFACTS_ROOT",
        "supported": list(HOPPER_PROFILES),
        "validated": [],
        "qualification_status": NOT_QUALIFIED,
        "historical": ("cutlass_single_sm", "cutlass_dual_sm"),
    },
    "flashinfer": {
        "cases_name": "FlashInfer Hopper 100-case mechanism-first set",
        "package_name": "FlashInfer Hopper full-GPU delivery tests",
        "manifest": "manifests/flashinfer_100.json",
        "case_set": "flashinfer_100",
        "env": "FLASHINFER_DELIVERY_ARTIFACTS_ROOT",
        "supported": list(HOPPER_PROFILES),
        "validated": [],
        "qualification_status": NOT_QUALIFIED,
        "historical": ("flashinfer_single_sm",),
    },
    "flashattention3": {
        "cases_name": "FlashAttention-3 Hopper full-GPU 100-case qualified delivery set",
        "package_name": "FlashAttention-3 Hopper delivery tests",
        "manifest": "manifests/fa3_100.json",
        "case_set": "fa3_100",
        "env": "FLASHATTENTION3_DELIVERY_ARTIFACTS_ROOT",
        "supported": ["h100"],
        "validated": ["h100"],
        "qualification_status": FA3_QUALIFIED,
        "historical": ("flashattention3_dual_sm",),
    },
}


def cases_header(
    suite: str,
    *,
    name: str,
    selected_count: int,
    extra_delivery: dict | None = None,
) -> dict:
    spec = SUITES[suite]
    delivery = {
        "groups": DELIVERY_GROUPS,
        "group_assignment": GROUP_ASSIGNMENT,
        "runner": RUNNER,
    }
    for key, value in dict(extra_delivery or {}).items():
        if key not in delivery:
            delivery[key] = value
    return {
        "schema_version": CASES_SCHEMA,
        "suite": suite,
        "name": name,
        "architecture": ARCHITECTURE,
        "selected_count": selected_count,
        "supported_gpu_profiles": list(spec["supported"]),
        "validated_gpu_profiles": list(spec["validated"]),
        "delivery": delivery,
    }


def merge_cases_manifest(payload: dict, suite: str) -> dict:
    spec = SUITES[suite]
    extra_delivery = {
        key: value
        for key, value in dict(payload.get("delivery") or {}).items()
        if key not in DELIVERY_CORE
    }
    header = cases_header(
        suite,
        name=str(payload["name"]) if payload.get("name") else spec["cases_name"],
        selected_count=int(payload.get("selected_count") or len(payload["cases"])),
        extra_delivery=extra_delivery,
    )
    out = dict(header)
    for key, value in payload.items():
        if key in header or key == "cases":
            continue
        out[key] = value
    out["cases"] = payload["cases"]
    return out


def assert_cases_manifest(payload: dict, suite: str) -> None:
    spec = SUITES[suite]
    if payload.get("schema_version") != CASES_SCHEMA:
        raise AssertionError(f"{suite} schema_version {payload.get('schema_version')!r}")
    if payload.get("suite") != suite:
        raise AssertionError(f"{suite} suite field {payload.get('suite')!r}")
    if payload.get("architecture") != ARCHITECTURE:
        raise AssertionError(f"{suite} architecture {payload.get('architecture')!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise AssertionError(f"{suite} cases is not a list")
    if int(payload.get("selected_count") or -1) != 100 or len(cases) != 100:
        raise AssertionError(
            f"{suite} want 100 cases, got selected_count={payload.get('selected_count')} n={len(cases)}"
        )
    if payload.get("supported_gpu_profiles") != list(spec["supported"]):
        raise AssertionError(f"{suite} supported_gpu_profiles {payload.get('supported_gpu_profiles')}")
    if payload.get("validated_gpu_profiles") != list(spec["validated"]):
        raise AssertionError(f"{suite} validated_gpu_profiles {payload.get('validated_gpu_profiles')}")
    if spec["supported"] != spec["validated"]:
        if payload["validated_gpu_profiles"] == payload["supported_gpu_profiles"]:
            raise AssertionError(f"{suite} supported_gpu_profiles must not equal validated")
    delivery = payload.get("delivery") or {}
    if delivery.get("groups") != DELIVERY_GROUPS:
        raise AssertionError(f"{suite} delivery.groups {delivery.get('groups')}")
    if delivery.get("group_assignment") != GROUP_ASSIGNMENT:
        raise AssertionError(f"{suite} delivery.group_assignment {delivery.get('group_assignment')}")
    if delivery.get("runner") != RUNNER:
        raise AssertionError(f"{suite} delivery.runner {delivery.get('runner')}")
    if "parameters" in payload:
        raise AssertionError(f"{suite} must not wrap cases in a parameters shell")
    ids = []
    for index, case in enumerate(cases):
        for key in CASE_FIELDS:
            if key not in case:
                raise AssertionError(f"{suite} case {index} missing {key}")
        if case["delivery_index"] != index:
            raise AssertionError(f"{suite} delivery_index {case['delivery_index']} != {index}")
        if case["delivery_group"] != index % DELIVERY_GROUPS:
            raise AssertionError(f"{suite} {case['case_id']} delivery_group != index modulo 4")
        mechs = case["mechanisms"]
        if not isinstance(mechs, list) or not mechs:
            raise AssertionError(f"{suite} {case['case_id']} mechanisms empty")
        if "parameters" in case:
            raise AssertionError(f"{suite} {case['case_id']} has redundant parameters shell")
        ids.append(case["case_id"])
    if len(set(ids)) != 100:
        raise AssertionError(f"{suite} case_id not unique")


def assert_delivery_package(payload: dict, suite: str) -> None:
    spec = SUITES[suite]
    if payload.get("schema_version") != PACKAGE_SCHEMA:
        raise AssertionError(f"{suite} DELIVERY schema_version {payload.get('schema_version')!r}")
    if payload.get("suite") != suite:
        raise AssertionError(f"{suite} DELIVERY suite {payload.get('suite')!r}")
    if payload.get("name") != spec["package_name"]:
        raise AssertionError(f"{suite} DELIVERY name {payload.get('name')!r}")
    if payload.get("architecture") != ARCHITECTURE:
        raise AssertionError(f"{suite} DELIVERY architecture {payload.get('architecture')!r}")
    if payload.get("formal_source_root") != f"suites/{suite}":
        raise AssertionError(f"{suite} formal_source_root {payload.get('formal_source_root')!r}")
    if payload.get("generated_artifacts_root") != f"artifacts/suites/{suite}":
        raise AssertionError(payload.get("generated_artifacts_root"))
    if payload.get("generated_artifacts_env") != spec["env"]:
        raise AssertionError(payload.get("generated_artifacts_env"))
    if payload.get("supported_gpu_profiles") != list(spec["supported"]):
        raise AssertionError(payload.get("supported_gpu_profiles"))
    if payload.get("validated_gpu_profiles") != list(spec["validated"]):
        raise AssertionError(payload.get("validated_gpu_profiles"))
    if payload.get("qualification_status") != spec["qualification_status"]:
        raise AssertionError(payload.get("qualification_status"))
    if "gpu_profiles" in payload:
        raise AssertionError(f"{suite} gpu_profiles alias must not remain")
    if "archived_case_sets" in payload:
        raise AssertionError(f"{suite} archived_case_sets alias must not remain")
    case_sets = payload.get("case_sets") or {}
    if set(case_sets) != {spec["case_set"]}:
        raise AssertionError(f"{suite} case_sets {sorted(case_sets)} (dual_sm must be historical)")
    entry = case_sets[spec["case_set"]]
    if entry.get("case_count") != 100:
        raise AssertionError(f"{suite} case_count {entry.get('case_count')}")
    if entry.get("manifest") != spec["manifest"]:
        raise AssertionError(entry.get("manifest"))
    if entry.get("qualification_status") != spec["qualification_status"]:
        raise AssertionError(entry.get("qualification_status"))
    if entry.get("validated_gpu_profiles") != list(spec["validated"]):
        raise AssertionError(entry.get("validated_gpu_profiles"))
    historical = payload.get("historical_case_sets")
    if not isinstance(historical, dict):
        raise AssertionError(f"{suite} historical_case_sets missing")
    if set(historical) != set(spec["historical"]):
        raise AssertionError(f"{suite} historical_case_sets {sorted(historical)}")
    for key, row in historical.items():
        if row.get("status") != "historical":
            raise AssertionError(f"{suite} {key} status {row.get('status')!r}")
        if not row.get("manifest"):
            raise AssertionError(f"{suite} {key} missing manifest")
        if "full100" in key or row.get("role") == "full100":
            raise AssertionError(f"{suite} historical {key} marked full100")
    if spec["supported"] != spec["validated"]:
        if payload["validated_gpu_profiles"] == payload["supported_gpu_profiles"]:
            raise AssertionError(f"{suite} supported_gpu_profiles must not equal validated")
