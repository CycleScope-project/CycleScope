"""Fail-closed consumer for DG's host-only actual-load/launch evidence.

This channel is independent of the historical whole-operation measurement.
Only actually loaded paths can supply cubin bytes and selected Function SASS.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any

FIELDS = ("start_clock64", "end_clock64", "start_globaltimer", "end_globaltimer", "sm_id", "cluster_cta_rank", "cluster_cta_count")
METHOD = "cuobjdump_dump_sass_function_pc_words_v1"


class BindingError(RuntimeError):
    pass

def participating_slots(launch: dict, capacity: int | None) -> frozenset[tuple[int, int]] | None:
    """Slots a launch writes if it targets the observed marker bank.

    sm90_suite_marker writes one row per participating warp and does not clear
    other slots. The row index is blockIdx.x * kMaxWarpsPerCTA + warp; grid.y
    and grid.z select other tiles of the same kernel but do not allocate
    further marker rows. Capacity, when recorded, is the producer
    max_warps_per_cta. A multidimensional block has no single linear warp
    index in that formula, so it cannot be compared.
    """
    grid, block = launch.get("grid"), launch.get("block")
    values = (*(grid or ()), *(block or ()))
    if (
        not isinstance(grid, list) or not isinstance(block, list)
        or len(grid) != 3 or len(block) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or min(values) <= 0 or block[1:] != [1, 1] or block[0] % 32
    ):
        return None
    warps = block[0] // 32
    if capacity is not None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            return None
        warps = min(warps, capacity)
    return frozenset((cta, warp) for cta in range(grid[0]) for warp in range(warps))



def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
        if digest is None:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

def preserve_device_library_root(deep_gemm: Any, package: Path) -> None:
    """Keep original include-path compiler signature with a host-only extension.

    The existing DG init API changes only host compiler/include-parser roots.
    Relocating the host package otherwise changes -I and its cache key, causing
    an unintended new device compilation. No cache name or Function is selected.
    """
    if not os.environ.get("DEEPGEMM_ACTUAL_BINDING_REQUIRED"):
        return
    original = Path(os.environ["DEEPGEMM_ORIGINAL_DEVICE_LIBRARY_ROOT"])
    metadata = json.loads((package / "deep_gemm/sm90_suite_marker.json").read_text())
    expected = metadata["device_include_sha256"]
    actual = {"deep_gemm/include/" + str(path.relative_to(original / "include")): sha256(path)
              for path in (original / "include").rglob("*") if path.is_file()}
    installed = {"deep_gemm/include/" + str(path.relative_to(package / "deep_gemm/include")): sha256(path)
                 for path in (package / "deep_gemm/include").rglob("*") if path.is_file()}
    if not expected or any(actual.get(key) != value for key, value in expected.items()) or actual != installed:
        raise BindingError("original device include bytes differ from host-only package")
    deep_gemm._C.init(str(original), deep_gemm._find_cuda_home())



def cursor() -> int | None:
    if not os.environ.get("DEEPGEMM_ACTUAL_BINDING_REQUIRED"):
        return None
    try:
        function = ctypes.CDLL(None).fi_launch_cursor
    except AttributeError as exc:
        raise BindingError("required actual launch observer is missing") from exc
    function.argtypes = []
    function.restype = ctypes.c_uint64
    return int(function())


def strict_jsonl(path: Path, sequence: str) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise BindingError(f"truncated observer log: {path}")
    rows = [json.loads(line) for line in data.splitlines()]
    seqs = [row[sequence] for row in rows]
    if seqs != list(range(1, len(rows) + 1)):
        raise BindingError(f"non-contiguous observer sequence: {path}")
    for row in rows:
        if "name" in row and (not isinstance(row["name"], str) or row["name_bytes"] != len(row["name"].encode())):
            raise BindingError("missing or truncated complete Function name")
    return rows


def actual_trial_binding(launches: list[dict], events: list[dict], window: dict) -> dict:
    lo, hi = window["lo_exclusive"], window["hi_inclusive"]
    if not isinstance(lo, int) or not isinstance(hi, int) or hi <= lo:
        raise BindingError("empty or invalid actual launch window")
    selected = [row for row in launches if lo < row["seq"] <= hi]
    if len(selected) != hi - lo:
        raise BindingError("actual window missing launch records")
    # Physical marker-bank bindings recorded by the host launcher at
    # Args-construction time: each event's launch_cursor is the seq of the
    # one launch that host thread enqueues next. Joining is by recorded
    # fact (bank + marker pointer arithmetic), never runtime-class name or
    # launch order; pid/thread must match the joined launch and a cursor may
    # carry at most one event, so a helper launch or cross-thread interleave
    # cannot silently steal a binding.
    def collect(event_name: str) -> dict[int, dict]:
        joined: dict[int, dict] = {}
        for event in events:
            if event.get("event") != event_name:
                continue
            seq = int(event["launch_cursor"]) + 1
            row = by_seq.get(seq)
            if row is None:
                continue
            if seq in joined:
                raise BindingError(f"ambiguous {event_name} binding for one launch")
            if event["pid"] != row["pid"]:
                raise BindingError(f"{event_name} event crossed processes")
            if "thread" in row and "thread" in event and event["thread"] != row["thread"]:
                raise BindingError(f"{event_name} event crossed host threads")
            joined[seq] = event
        return joined
    by_seq = {row["seq"]: row for row in selected}
    bank_events = collect("marker_bank")
    reset_events = collect("marker_reset")
    bound = []
    contexts = set()
    for row in selected:
        if row["enqueue_status"] not in (0, -1):
            raise BindingError("actual enqueue failed")
        if not row.get("driver_identity"):
            # Runtime-API launch. marker_reset joined by cursor+thread is the
            # attested bank zero. Anything else stays in the window as a
            # foreign call: not a selected target, not dropped, and not a
            # reason by itself to reject the trial.
            continue
        active: dict[str, dict] = {}
        for event in events:
            if event["pid"] != row["pid"] or event["launch_cursor"] >= row["seq"]:
                continue
            if event["event"] in ("marker_bank", "marker_reset"):
                continue
            if event["event"] == "unload":
                if event["status"] == 0:
                    active.pop(event["library"], None)
            elif event["event"] == "bind":
                if event["library"] in active:
                    raise BindingError("ambiguous library generation")
                active[event["library"]] = event
            else:
                raise BindingError("unknown loaded-code event")
        matches = [event for event in active.values() if all(event[key] == row[key] for key in ("func", "module", "context", "pid"))]
        if not matches:
            raise BindingError("driver launch has no matching loaded Function/module/context binding")
        if len(matches) != 1 or not matches[0]["binding_valid"] or not row["binding_valid"] or not row["geometry_valid"] or row["enqueue_status"] != 0:
            raise BindingError("invalid actual function/module/context/config binding")
        if row.get("config_complete") is not True or not isinstance(row.get("stream"), str) or row.get("num_attrs") != len(row.get("attributes", [])):
            raise BindingError("actual launch configuration is incomplete")
        event = matches[0]
        if event["name"] != row["name"]:
            raise BindingError("actual launch name differs from loaded Function")
        contexts.add(row["context"])
        bound.append({"launch": row, "load": event})
    if len(contexts) > 1:
        raise BindingError("actual trial window spans multiple contexts")
    if not bound:
        raise BindingError("no actual DeepGEMM loaded-function launch in trial")
    foreign = [row for row in selected if not row.get("driver_identity") and row["seq"] not in reset_events]
    return {"all_launches": selected, "bound_launches": bound, "window": window,
            "marker_banks": bank_events, "marker_resets": reset_events,
            "foreign_runtime_launches": foreign}


def selected_fingerprint(text: str, entry: str) -> str:
    # Use the campaign's exact method; reject ambiguous/malformed sections first.
    headers = re.findall(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.M)
    if headers.count(entry) != 1:
        raise BindingError("selected Function section is absent or nonunique")
    active = False
    for line in text.splitlines():
        header = re.match(r"^\s*Function\s*:\s*(\S+)\s*$", line)
        if header:
            active = header[1] == entry
        elif active and re.match(r"^\s*/\*[0-9a-fA-F]+\*/", line) and not re.match(r"^\s*/\*[0-9a-fA-F]+\*/\s*.*?\s*/\*\s*0x[0-9a-fA-F]{16}\s*\*/\s*$", line):
            raise BindingError("malformed selected Function PC/encoding")
    root = Path(__file__).resolve().parents[3]
    import sys
    python_root = str(root / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    from hoppersim.cluster_sm_campaign import _selected_function_sass_fingerprint
    digest, reason = _selected_function_sass_fingerprint(text, entry)
    if digest is None:
        raise BindingError(reason)
    return digest


def build_unit_denominator(spec: dict, trials: list[dict], bindings: list[dict], codes: dict, group_size: int,
                           marker_capacity: dict | None = None) -> dict:
    if spec.get("clock_marker_policy") != "outermost_pair":
        raise BindingError("unit must explicitly use original outermost_pair policy")
    grid, block = spec["grid"], spec["block"]
    grid_values = [int(grid[key]) for key in ("x", "y", "z")]
    block_values = [int(block[key]) for key in ("x", "y", "z")]
    if min(grid_values + block_values) <= 0 or grid_values[1:] != [1, 1] or block_values[1:] != [1, 1] or block_values[0] % 32:
        raise BindingError("original marker layout cannot distinguish this geometry")
    ctas = spec["ctas"]
    coordinates = [tuple(int(v) for v in cta) for cta in ctas]
    if len(set(coordinates)) != len(coordinates) or any(len(cta) != 3 or cta[1:] != (0, 0) or not 0 <= cta[0] < grid_values[0] for cta in coordinates):
        raise BindingError("invalid or duplicate unit CTA coordinates")
    # The census is over the instrumentation's marker-PARTICIPATING warp set,
    # never a claim that every block warp carries a marker. sm90_suite_marker
    # records at most kMaxWarpsPerCTA rows per CTA (begin/end return for
    # higher warps), and hardware and simulation share the same instrumented
    # binary, so the outermost marked interval on both sides is defined by
    # exactly this participating set. The capacity must come from the run's
    # recorded producer fact (package sm90_suite_marker.json), not a local
    # constant: without it the census stays strict (all block warps) and a
    # wider-than-capacity block is honestly unmeasured.
    block_warps = block_values[0] // 32
    if marker_capacity is None:
        marker_warp_count = block_warps
        capacity_fact = None
    else:
        capacity = int(marker_capacity.get("max_warps_per_cta", 0))
        if capacity < 1:
            raise BindingError("marker warp capacity fact is invalid")
        marker_warp_count = min(block_warps, capacity)
        capacity_fact = dict(marker_capacity)
    expected = {(cta[0], warp) for cta in coordinates for warp in range(marker_warp_count)}
    if not expected or spec.get("type") not in ("cta", "cluster", "launch"):
        raise BindingError("invalid unit type or empty unit")
    raw, ratio_facts, launch_facts, bank_facts = [], [], [], []
    for trial, binding in zip(trials, bindings, strict=True):
        rows = trial["example_rows"]
        launches = binding["all_launches"]
        bank_events = binding["marker_banks"]
        driver_launches = [row for row in launches if row.get("driver_identity")]
        runtime_launches = [row for row in launches if not row.get("driver_identity")]
        observed_banks = {int(row.get("bank", 0)) for row in rows}
        if any(seq not in {row["seq"] for row in driver_launches} for seq in bank_events):
            # A bank fact bound to a non-driver launch is a mis-wiring, not a
            # count coincidence to surface later as a KeyError.
            raise BindingError("marker-bank event bound a non-driver launch")
        for foreign in binding.get("foreign_runtime_launches", []):
            if foreign.get("name") == spec["kernel_name"]:
                raise BindingError("unattested runtime launch aliases the selected target")
        capacity_limit = None if capacity_fact is None else int(capacity_fact["max_warps_per_cta"])
        slot_owner = None
        if not bank_events:
            if observed_banks != {0} or not driver_launches:
                raise BindingError("trial without bank facts cannot attest its driver launches")
            observed_slots = frozenset((row["cta_id"], row["warp"]) for row in rows)
            owners = []
            for candidate in binding["bound_launches"]:
                slots = participating_slots(candidate["launch"], capacity_limit)
                if slots is None:
                    raise BindingError("driver launch geometry cannot be compared with the observed marker bank")
                if slots == observed_slots:
                    owners.append(candidate)
            if len(owners) != 1:
                raise BindingError("observed marker slots do not uniquely identify one physical enqueue")
            # Same loaded function may enqueue twice with different grids.
            # That is not a trial failure before the captured kernel is
            # selected; only one physical enqueue may own these rows.
            attribution = None
            slot_owner = owners[0]
        else:
            # Recorded bank facts exist (full7 attention): every DRIVER
            # launch must carry one; runtime launches are the host-attested
            # marker resets and must sit inside the claimed storage region.
            if len(bank_events) != len(driver_launches):
                raise BindingError("marker trial lacks complete per-driver-launch bank binding")
            claims: dict[int, dict] = {}
            for actual in binding["bound_launches"]:
                event = bank_events[actual["launch"]["seq"]]
                bank = int(event["bank"])
                if bank in claims:
                    raise BindingError("two launches claim the same marker bank")
                claims[bank] = actual
            bases = {(int(event["logits_ptr"]), int(event["payload_bytes"]), int(event["marker_bytes"]))
                     for event in bank_events.values()}
            if len(bases) != 1:
                raise BindingError("marker-bank events disagree on the output storage")
            logits_ptr, payload_bytes, marker_bytes = bases.pop()
            for event in bank_events.values():
                if int(event["marker_ptr"]) != logits_ptr + payload_bytes + int(event["bank"]) * marker_bytes:
                    raise BindingError("marker-bank pointer arithmetic contradicts the recorded bank")
            if not observed_banks <= set(claims):
                raise BindingError("marker rows fall in a bank no launch claimed")
            for event in binding["marker_resets"].values():
                base, size = int(event["base"]), int(event["bytes"])
                if size <= 0 or size % marker_bytes:
                    raise BindingError("marker-reset region is not a whole number of banks")
                for bank in claims:
                    pointer = logits_ptr + payload_bytes + bank * marker_bytes
                    if not (base <= pointer and pointer + marker_bytes <= base + size):
                        raise BindingError("marker-reset region does not cover a claimed bank")
            attribution = claims
        matches = [actual for actual in binding["bound_launches"] if actual["launch"]["name"] == spec["kernel_name"]]
        if len(matches) != 1:
            raise BindingError("selected Function launch is absent or nonunique in the trial")
        actual = matches[0]
        launch = actual["launch"]
        if launch["grid"] != grid_values or launch["block"] != block_values or launch["cluster"] != spec["cluster"]:
            raise BindingError("selected Function or original geometry differs")
        if slot_owner is not None and slot_owner["launch"]["seq"] != launch["seq"]:
            raise BindingError("selected Function is not the physical enqueue that owns the observed marker slots")
        code = codes[actual["load"]["loaded_path"]]
        if code["fingerprint"]["sha256"] != spec["kernel_sass_fingerprint"]["sha256"] or spec["kernel_sass_fingerprint"]["method"] != METHOD or spec["kernel_sass_fingerprint"]["entry"] != launch["name"]:
            raise BindingError("loaded-device selected Function fingerprint differs")
        if attribution is None:
            unit_bank = 0
        else:
            unit_bank = next(bank for bank, item in attribution.items() if item is actual)
        launch_rows = [row for row in rows if int(row.get("bank", 0)) == unit_bank]
        if attribution is not None and not launch_rows:
            raise BindingError("selected launch produced no marker rows in its bound bank")
        keys = [(row["cta_id"], row["warp"]) for row in launch_rows]
        if len(set(keys)) != len(keys):
            raise BindingError("duplicate CTA-warp marker row")
        full = {(cta, warp) for cta in range(grid_values[0]) for warp in range(marker_warp_count)}
        if set(keys) != full:
            raise BindingError("missing/extra warp in complete original launch")
        per_cta = []
        for cta in range(grid_values[0]):
            warps = sorted({warp for cta_id, warp in keys if cta_id == cta})
            per_cta.append(warps)
        if any(warps != per_cta[0] for warps in per_cta):
            raise BindingError("marker participating warps differ across CTAs")
        observed_warps = per_cta[0]
        if observed_warps != list(range(marker_warp_count)):
            raise BindingError("observed marker warps differ from the producer capacity set")
        selected_slots = frozenset(keys)
        for foreign in binding.get("foreign_runtime_launches", []):
            slots = participating_slots(foreign, capacity_limit)
            if slots is None:
                raise BindingError("unattested runtime launch geometry cannot be excluded from the observed marker bank")
            if slots == selected_slots:
                raise BindingError("unattested runtime launch aliases the selected marker slot set")
        unit_rows = [row for row in launch_rows if (row["cta_id"], row["warp"]) in expected]
        for row in rows:
            if any(key not in row or not isinstance(row[key], int) or row[key] < 0 for key in FIELDS):
                raise BindingError("incomplete seven-field raw marker row")
            if row["end_clock64"] <= row["start_clock64"] or row["end_globaltimer"] <= row["start_globaltimer"]:
                raise BindingError("nonpositive warp clock/timer interval")
        cluster = launch["cluster"]
        if cluster[1:] != [1, 1] or cluster[0] < 1:
            raise BindingError("cluster xyz not representable by original x-count marker")
        for cta in range(grid_values[0]):
            members = [row for row in launch_rows if row["cta_id"] == cta]
            if not members or len({(row["sm_id"], row["cluster_cta_rank"], row["cluster_cta_count"]) for row in members}) != 1:
                raise BindingError("within-CTA SM/rank/count contradiction")
            if members[0]["cluster_cta_count"] != cluster[0] or members[0]["cluster_cta_rank"] != cta % cluster[0]:
                raise BindingError("actual cluster configuration disagrees with device markers")
        if spec["type"] == "cluster" and (len(coordinates) != cluster[0] or cluster[0] == 1 or len({cta[0] // cluster[0] for cta in coordinates}) != 1 or {row["cluster_cta_rank"] for row in unit_rows} != set(range(cluster[0]))):
            raise BindingError("unit is not one complete actual cluster")
        if spec["type"] == "cta" and len(coordinates) != 1:
            raise BindingError("CTA unit has multiple members")
        if spec["type"] == "launch" and len(coordinates) != grid_values[0]:
            raise BindingError("launch unit omits original CTAs")
        ratios = [(row["end_clock64"] - row["start_clock64"]) / (row["end_globaltimer"] - row["start_globaltimer"]) for row in unit_rows]
        ratio = float(statistics.median(ratios))
        span = max(row["end_globaltimer"] for row in unit_rows) - min(row["start_globaltimer"] for row in unit_rows)
        raw.append(float(span * ratio))
        ratio_facts.append({"clock_domain": "per_sm_clock64/common_globaltimer", "reduction": "globaltimer_span_times_median_unit_warp_clock_timer_ratio", "ratio": ratio, "ratio_rows": len(unit_rows), "sm_ids": sorted({row["sm_id"] for row in unit_rows}), "clock_marker_warps": observed_warps})
        if attribution is not None:
            owner_rule = "per_launch_recorded_bank_pointer_arithmetic"
        elif len(driver_launches) == 1:
            owner_rule = "single_launch_owns_all_rows"
        else:
            owner_rule = "unique_observed_slot_enqueue"
        launch_facts.append(actual)
        bank_facts.append({"unit_bank": unit_bank,
                           "owner_rule": owner_rule,
                           "attribution": None if attribution is None else
                           {str(bank): {"seq": item["launch"]["seq"], "name": item["launch"]["name"],
                                        "marker_ptr": bank_events[item["launch"]["seq"]]["marker_ptr"]}
                            for bank, item in sorted(attribution.items())},
                           "slot_owner": None if slot_owner is None else {
                               "seq": slot_owner["launch"]["seq"],
                               "name": slot_owner["launch"]["name"],
                               "func": slot_owner["launch"]["func"],
                               "module": slot_owner["launch"]["module"],
                               "grid": slot_owner["launch"]["grid"],
                               "block": slot_owner["launch"]["block"]},
                           "nonowner_driver_launches": [
                               {"seq": row["seq"], "name": row["name"], "func": row["func"],
                                "module": row["module"], "grid": row["grid"], "block": row["block"]}
                               for row in driver_launches if row["seq"] != launch["seq"]],
                           "foreign_runtime_launches": [
                               {"seq": row["seq"], "name": row["name"], "grid": row["grid"], "api": row.get("api")}
                               for row in binding.get("foreign_runtime_launches", [])],
                           "marker_resets": [{"seq": seq, "base": event["base"], "bytes": event["bytes"]}
                                             for seq, event in sorted(binding["marker_resets"].items())]})
    if not raw or group_size not in (1, 3, 4) or len(raw) % group_size:
        raise BindingError("incomplete consecutive raw aggregation groups")
    if any(fact["clock_marker_warps"] != ratio_facts[0]["clock_marker_warps"] for fact in ratio_facts):
        raise BindingError("marker participating warps differ across trials")
    identities = []
    for fact in bank_facts:
        attribution = fact["attribution"]
        named = () if attribution is None else tuple(sorted(
            (int(bank), str(item["name"])) for bank, item in attribution.items()
        ))
        owner = fact["slot_owner"]
        owner_id = None if owner is None else (owner["name"], tuple(owner["grid"]), tuple(owner["block"]))
        identities.append((fact["unit_bank"], named, fact["owner_rule"], owner_id))
    if len(set(identities)) != 1:
        raise BindingError("marker unit bank or selected launch identity differs across trials")
    clock_marker_warps = ratio_facts[0]["clock_marker_warps"]
    samples = [float(statistics.median(raw[i:i + group_size])) for i in range(0, len(raw), group_size)]
    census_scope = ("marker_participating_warps_from_recorded_producer_capacity" if capacity_fact
                    else "strict_all_block_warps_no_capacity_fact")
    return {"schema": "delivery.unit-marker-denominator.v1", "status": "same_unit_matched", "eligible_for_simulator_accuracy": len(samples) == 9,
            "scope": "outermost_in_kernel_clock_marker_interval", "clock_marker_policy": "outermost_pair",
            "unit_scope": {**{key: spec[key] for key in ("type", "kernel_file", "ctas", "grid", "block")},
                           "slot_count": len(expected), "covers_complete_launch": len(coordinates) == grid_values[0],
                           "block_warps": block_warps, "marker_warp_capacity": None if capacity_fact is None else int(capacity_fact["max_warps_per_cta"]),
                           "clock_marker_warps": clock_marker_warps,
                           "nonparticipating_block_warp_count": block_warps - marker_warp_count,
                           "census_scope": census_scope,
                           "marker_capacity_fact": capacity_fact},
            "member_warp_binding": {"unit_ctas": ctas, "required_rows_per_launch": len(expected), "launches_reduced": len(raw),
                                    "clock_marker_warps": clock_marker_warps,
                                    "census_scope": census_scope},
            "code_identity": {"kernel_name": spec["kernel_name"], "kernel_sass_fingerprint": codes[launch_facts[0]["load"]["loaded_path"]]["fingerprint"]},
            "samples_cycles_raw": raw, "samples_cycles": samples, "median_cycles": float(statistics.median(samples)), "sample_count": len(samples), "raw_sample_count": len(raw), "raw_per_recorded_sample": group_size, "clock_trials": ratio_facts, "actual_launch_binding": launch_facts,
            "marker_bank_binding": {"rule": bank_facts[0]["owner_rule"],
                                    "unit_bank": bank_facts[0]["unit_bank"],
                                    "trials": bank_facts}}

def publish_simulator_accuracy(unit_denominator: dict | None, comparison_contract: dict | None = None) -> dict[str, Any]:
    """Accuracy fields from one already proven unit denominator.

    A comparison contract never grants eligibility and never supplies the median.
    """
    contract = comparison_contract if isinstance(comparison_contract, dict) else None
    digest = None if contract is None else contract.get("contract_digest")
    hardware_metric = (
        contract.get("hardware_metric")
        if contract is not None and contract.get("hardware_metric")
        else "timing.unit_denominator.median_cycles"
    )
    reason = unit_denominator.get("reason") if isinstance(unit_denominator, dict) else "unit denominator absent"
    rejected = {
        "status": "unmeasured",
        "eligible_for_simulator_accuracy": False,
        "median_cycles": None,
        "samples_cycles": None,
        "samples_cycles_raw": None,
        "selected_operation_kernel": None,
        "selected_unit_bank": None,
        "hardware_metric": hardware_metric,
        "hardware_value_field": "timing.unit_denominator.median_cycles",
        "contract_digest": digest,
        "reason": reason,
    }
    if not isinstance(unit_denominator, dict):
        return rejected
    binding = unit_denominator.get("marker_bank_binding")
    identity = unit_denominator.get("code_identity")
    bank = binding.get("unit_bank") if isinstance(binding, dict) else None
    kernel = identity.get("kernel_name") if isinstance(identity, dict) else None
    if (
        unit_denominator.get("status") != "same_unit_matched"
        or unit_denominator.get("eligible_for_simulator_accuracy") is not True
        or isinstance(bank, bool)
        or not isinstance(bank, int)
        or not isinstance(kernel, str)
        or not kernel
        or unit_denominator.get("median_cycles") is None
    ):
        if rejected["reason"] is None:
            rejected["reason"] = "unit denominator is not an eligible single-bank measurement"
        return rejected
    return {
        "status": "matched",
        "eligible_for_simulator_accuracy": True,
        "median_cycles": unit_denominator["median_cycles"],
        "samples_cycles": unit_denominator.get("samples_cycles"),
        "samples_cycles_raw": unit_denominator.get("samples_cycles_raw"),
        "selected_operation_kernel": kernel,
        "selected_unit_bank": bank,
        "hardware_metric": hardware_metric,
        "hardware_value_field": "timing.unit_denominator.median_cycles",
        "contract_digest": digest,
        "reason": None,
    }


def record_trial(trial: dict) -> None:
    if os.environ.get("DEEPGEMM_ACTUAL_BINDING_REQUIRED"):
        path = Path(os.environ["FLASHINFER_CLUSTER_LAUNCH_LOG"]).with_name("raw_marker_trials.jsonl")
        with path.open("a") as stream:
            stream.write(json.dumps(trial, sort_keys=True) + "\n")


def export_evidence(trials: list[dict], group_size: int, marker_capacity: dict | None = None) -> dict | None:
    if not os.environ.get("DEEPGEMM_ACTUAL_BINDING_REQUIRED"):
        return None
    launch_path = Path(os.environ["FLASHINFER_CLUSTER_LAUNCH_LOG"])
    code_path = Path(os.environ["FLASHINFER_LOADED_CODE_LOG"])
    manifest_path = Path(os.environ["DEEPGEMM_ACTUAL_SOURCE_MANIFEST"])
    inventory = json.loads(manifest_path.read_text())["files"]
    launches = strict_jsonl(launch_path, "seq")
    events = strict_jsonl(code_path, "code_seq")
    bindings = [actual_trial_binding(launches, events, trial["actual_launch_window"]) for trial in trials]
    codes = {}
    for binding in bindings:
        for actual in binding["bound_launches"]:
            load = actual["load"]
            path = Path(load["loaded_path"])
            if str(path) in codes:
                continue
            source = inventory.get(str(path))
            if source is None or sha256(path) != source["sha256"] or path.stat().st_size != source["size"]:
                raise BindingError("actual loaded path absent from immutable byte inventory or changed")
            command = [os.environ.get("CUOBJDUMP", "cuobjdump"), "--dump-sass", str(path)]
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            dump = launch_path.parent / f"loaded-{load['code_seq']}.sass"
            dump.write_text(completed.stdout)
            dump.with_suffix(".stderr").write_text(completed.stderr)
            if completed.returncode != 0:
                raise BindingError(f"actual loaded-cubin cuobjdump rc={completed.returncode}")
            if sha256(path) != source["sha256"]:
                raise BindingError("loaded source changed during SASS extraction")
            codes[str(path)] = {"source": source, "loaded_path": str(path), "loaded_cubin_sha256": source["sha256"], "cuobjdump_returncode": completed.returncode, "cuobjdump_command": command, "sass": str(dump), "sass_sha256": sha256(dump), "fingerprint": {"method": METHOD, "entry": load["name"], "sha256": selected_fingerprint(completed.stdout, load["name"])}}
    result = {"status": "actual_launch_bound", "launch_log": str(launch_path), "launch_log_sha256": sha256(launch_path), "loaded_code_log": str(code_path), "loaded_code_log_sha256": sha256(code_path), "source_manifest": str(manifest_path), "source_manifest_sha256": sha256(manifest_path), "trials": bindings, "loaded_code": codes, "marker_capacity": marker_capacity, "unit_denominator": None}
    unit_path = os.environ.get("DEEPGEMM_ACTUAL_UNIT_JSON")
    if unit_path:
        try:
            result["unit_denominator"] = build_unit_denominator(json.loads(Path(unit_path).read_text()), trials, bindings, codes, group_size, marker_capacity=marker_capacity)
        except (BindingError, KeyError, TypeError, ValueError) as exc:
            result["unit_denominator"] = {"schema": "delivery.unit-marker-denominator.v1", "status": "unmeasured", "eligible_for_simulator_accuracy": False, "reason": str(exc)}
    return result
