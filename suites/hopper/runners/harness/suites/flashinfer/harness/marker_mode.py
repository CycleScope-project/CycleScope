"""Marker-mode glue for the FlashInfer delivery harness (T0005 修订 3-② / 修订 5).

One instrumented binary for the marker measurement (修订 5): `--marker-timing`
(1) applies scripts/patch_wheel_marker.py to the installed wheel tree,
(2) enables `-DFI_DELIVERY_MARKER_TIMING` + registry linkage for every JIT build
(FLASHINFER_EXTRA_CUDAFLAGS / FLASHINFER_EXTRA_LDFLAGS — ninja's flag change deterministically
rebuilds the affected cached_ops), and (3) drives the marker channel
(harness/marker_timing.py).  Without the flag run_case.py keeps its exact previous
behaviour (patcher not invoked, env untouched).  This module is measurement-only:
the stage-6 capture path (`--capture-window`) is a pure run-once workload under
suites/common/capture_case.py and never touches the marker machinery.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable

HARNESS = Path(__file__).resolve().parent
SUITE = HARNESS.parent
SCRIPTS = SUITE / "scripts"

if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

import marker_channel_config as mcc  # noqa: E402
import marker_timing as mt  # noqa: E402


def _load_patcher():
    spec = spec_from_file_location("fi_patch_wheel_marker", SCRIPTS / "patch_wheel_marker.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheel_tree_root(flashinfer_module) -> Path:
    """Directory containing the `flashinfer/` package (wheel site dir)."""
    package_root = Path(flashinfer_module.__file__).resolve().parent
    tree = package_root.parent
    if not (tree / "flashinfer/data").is_dir():
        raise RuntimeError(
            f"{tree} does not look like a FlashInfer wheel site dir (flashinfer/data missing); "
            "the delivery marker patch targets the pip-installed wheel tree only"
        )
    return tree


def registry_lib_path(artifact_root: Path) -> Path:
    lib = Path(os.environ.get(mt.MARKER_LIB_ENV, "")) if os.environ.get(mt.MARKER_LIB_ENV) \
        else Path(artifact_root) / "build" / "libdelivery_fi_marker.so"
    return lib


def apply_marker_mode(flashinfer_module, artifact_root: Path) -> dict:
    """Patch the wheel tree (idempotent) and enable the marker JIT env; returns the record."""
    artifact_root = Path(artifact_root)
    tree = wheel_tree_root(flashinfer_module)
    log_dir = artifact_root / "build/marker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fi_marker_patch_log.json"
    patcher = _load_patcher()
    # --sm-cap is opt-in (rev1 item 2): the host-side SM-cap raw patches compile in only with
    # -DFI_DELIVERY_SM_CAP (enable_sm_cap_mode below), so applying them unconditionally keeps
    # the no-cap build upstream-identical.  The deep_gemm raw edits ride along with marker
    # mode (they are #ifdef FI_DELIVERY_MARKER_TIMING-guarded at compile time).
    argv = ["--tree", str(tree), "--apply", "--log", str(log_path)]
    if mt.sm_cap_requested() > 0:
        argv.append("--sm-cap")
    rc = patcher.main(argv)
    if rc != 0:
        raise RuntimeError(f"patch_wheel_marker --apply failed (rc={rc}); see {log_path}")
    lib_path = registry_lib_path(artifact_root)
    if not lib_path.is_file():
        raise RuntimeError(
            f"{lib_path} is missing: run scripts/build.sh (it compiles "
            "harness/override/fi_delivery_marker_registry.cpp) before marker timing/capture"
        )
    env_record = mt.enable_jit_marker_mode(lib_path)
    sm_cap_record = mt.enable_sm_cap_mode()
    if sm_cap_record:
        env_record["sm_cap"] = sm_cap_record
    return {"patch_log": str(log_path), **env_record}


def install_channel(torch, op_class: str, case: dict) -> tuple:
    """Build the MarkerChannel (JIT modules must be built/loaded already).

    Registry-health semantics (T0011.d, Q1④): ``expected_module_keys`` is the STATIC
    superset from marker_channel_config (probe evidence); modules the op path legitimately
    never builds are exempted from ``missing_keys`` — prefill_sm90_quant only exists for
    fp8 q/kv dtypes (gen_*_prefill_module fp8_enabled).  Health criterion = registered ⊇
    the resolved serving module (resolve_primary / resolve_primary_dynamic); ``missing``
    is informational, the pass/fail happens in the freshness gate + dynamic resolution.
    """
    lib_path = registry_lib_path(Path(os.environ["FLASHINFER_DELIVERY_ARTIFACTS_ROOT"]))
    channel = mt.MarkerChannel(torch, lib_path)
    entry = mcc.OP_CLASSES[op_class]
    expected = [key for key in entry["jit_modules"] if key in mcc.MODULE_PREFIXES]
    registered = set(channel.module_keys)
    dtype = str(case.get("dtype") or "").lower()
    exempt = {"prefill_sm90_quant"} if "fp8" not in dtype else set()
    missing = [key for key in expected if key not in registered and key not in exempt]
    waived = [key for key in expected if key not in registered and key in exempt]
    return channel, {"expected_module_keys": expected, "missing_keys": missing,
                     "missing_exempt_non_fp8": waived,
                     "registered": channel.module_keys}


def resolve_primary(op_class: str, case: dict):
    primary = mcc.resolve_primary(op_class, case)
    if primary is None:
        raise RuntimeError(
            f"{op_class}: marker primary kernel selection is a G0 item "
            f"(see harness/marker_channel_config.py note)"
        )
    return primary


def primary_candidates(op_class: str, case: dict) -> list[tuple[str, int, str]] | None:
    """Ordered (module, region, kernel) candidates for one case.

    rev9 dynamic lists (H100 probe2 evidence) win; otherwise [static primary] + secondaries
    (mamba-style).  None = the op has no resolvable primary (G0 selection item).
    """
    candidates = REV9_DYNAMIC_CANDIDATES.get(op_class)
    if candidates is not None:
        return list(candidates)
    primary = resolve_primary(op_class, case)
    if primary is None:
        return None
    entry = mcc.OP_CLASSES[op_class]
    return [primary] + [tuple(c) for c in (entry.get("secondary") or [])]


def resolve_primary_dynamic(torch, run, channel, candidates):
    """One probe launch; pick the first candidate whose (module, region) carries valid rows.

    Same semantics as collect_marker_main_kernel's resolve_dynamic prologue, exposed so the
    capture wrapper resolves the primary BEFORE the traced bracket with the same candidate
    order as the timing pass (rev9).  ensure_entries first: late-registered modules must be
    enumerated and given buffer pointers before the probe (den-r12 fix D).
    """
    channel.ensure_entries()
    channel.zero_all()
    run()
    torch.cuda.synchronize()
    for module, region, kernel in candidates:
        for entry_index in channel.resolve_entry(module):
            sample = channel.probe_region(entry_index, region)
            if sample is not None and sample["valid_rows"] > 0:
                return module, region, kernel
    raise RuntimeError(
        f"no candidate marker region carried rows (candidates={candidates}, "
        f"registered={channel.entries}); the op selected an uninstrumented kernel variant"
    )



# -------------------------------------------------------------------------------------------
# T0011.d JIT freshness gate (far45 runtime-variant class).
#
# T0005 rounds evidence: the loaded fa2/fa3 prefill & decode JIT binaries carried NO marker
# SASS ops (runtime_sass_audit.json: prefill/prefill_sm90/decode pass=False) while the mla
# modules passed — a stale cached_ops .so built without -DFI_DELIVERY_MARKER_TIMING was
# loaded instead of rebuilding from the patched tree, which is exactly the "registered module
# with empty rows" / "registry is empty" failure shape of the 15 runtime-variant cases.
# Gate: cuobjdump every candidate binary of the op's expected modules; a binary that defines
# any candidate kernel WITHOUT the >=2 CS2R SR_CLOCKLO / SR_GLOBALTIMERLO operand gate is
# stale -> purge its cached_ops dir and rebuild (run_case.py), then re-audit loudly.
# -------------------------------------------------------------------------------------------


def _load_sass_audit():
    spec = spec_from_file_location("fi_marker_sass_audit", SCRIPTS / "marker_sass_audit.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def marker_gate_kernels(op_class: str, case: dict | None = None) -> dict[str, list[str]]:
    """module_key -> candidate kernel stems whose loaded .so must carry the marker gate.

    NVRTC/cubin (fp8_blockscale_gemm) and CuTe-DSL (gdn) paths have no .so-side kernel to
    gate (their channels are bridged/unreachable respectively) and are skipped here.

    Modules that only appear as dynamic-candidate FALLBACKS (never as the backend-resolved
    primary) are marked by their absence from REQUIRED_MODULES: the op legitimately never
    builds them for this backend/dtype (e.g. prefill_sm90 for fa2 decode, prefill_sm90_quant
    for non-fp8 prefill), so "binary not found" for them is not a staleness signal — see
    jit_freshness_audit.
    """
    entry = mcc.OP_CLASSES[op_class]
    candidates = REV9_DYNAMIC_CANDIDATES.get(op_class)
    if candidates is not None:
        pairs = list(candidates)
    else:
        primary = entry.get("primary")
        pairs = ([tuple(primary)] if primary else []) + [
            tuple(c) for c in (entry.get("secondary") or [])
        ]
    per_module: dict[str, list[str]] = {}
    if entry.get("aot") in ("nvrtc", "cute_dsl"):
        return per_module
    primary = mcc.resolve_primary(op_class, case)
    for module_key, _region, kernel in pairs:
        if module_key not in mcc.MODULE_PREFIXES:
            continue
        stems = per_module.setdefault(module_key, [])
        if kernel not in stems:
            stems.append(kernel)
    return per_module


def required_gate_modules(op_class: str, case: dict | None = None) -> set[str]:
    """Module keys whose absence (not built by this backend/dtype) IS a staleness signal:
    the backend-resolved primary module(s).  Fallback-only candidate modules are exempt —
    the op legitimately never builds them (den_r12 finding: fa2 decode reported a spurious
    prefill_sm90 staleness)."""
    primary = mcc.resolve_primary(op_class, case)
    return {primary[0]} if primary else set()


def jit_freshness_audit(op_class: str, case: dict | None = None,
                        cuobjdump: str = "cuobjdump") -> list[dict]:
    """SASS-gate the candidate JIT binaries of one op's expected modules (T0011.d).

    A binary is stale when it defines a candidate kernel without the marker gate.  A module
    with no binary at all is stale only when it is the backend-resolved primary module
    (required_gate_modules) — fallback-only modules are simply never built for some
    backends/dtypes.  Returns one record per module:
    {module_key, kernels, binaries[], stale}.
    """
    import subprocess  # noqa: PLC0415

    audit = _load_sass_audit()
    required = required_gate_modules(op_class, case)
    reports = []
    for module_key, stems in marker_gate_kernels(op_class, case).items():
        rec: dict = {"module_key": module_key, "kernels": stems, "binaries": [], "stale": False}
        reports.append(rec)
        binaries = mt.candidate_jit_binaries(module_key)
        if not binaries:
            if module_key in required:
                rec["error"] = "no loaded JIT binary found for required primary module"
                rec["stale"] = True
            else:
                rec["error"] = "module not built for this backend/dtype (fallback-only)"
            continue
        for so_path in binaries:
            entry: dict = {"binary": str(so_path)}
            rec["binaries"].append(entry)
            try:
                proc = subprocess.run(
                    [cuobjdump, "--dump-sass", str(so_path)],
                    capture_output=True, text=True, timeout=600,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                rec["stale"] = True
                continue
            if proc.returncode != 0:
                entry.update({"dump_rc": proc.returncode,
                              "dump_error": (proc.stderr or "")[-300:]})
                rec["stale"] = True
                continue
            sections = audit.kernel_sections(proc.stdout or "")
            per_kernel = {}
            defined_fail = []
            for stem in stems:
                _matched, body = audit.find_kernel_section(sections, stem)
                clocklo = audit.clocklo_read_count(body) if body else 0
                timer = body.count("SR_GLOBALTIMERLO") if body else 0
                per_kernel[stem] = {
                    "present": body is not None,
                    "sr_clocklo_cs2r_reads": clocklo,
                    "sr_globaltimerlo_reads": timer,
                }
                if body is not None and (clocklo < 2 or timer < 2):
                    defined_fail.append(stem)
            entry["kernels"] = per_kernel
            if defined_fail:
                entry["uninstrumented_kernels"] = defined_fail
                rec["stale"] = True

    # T0011.g (dispatch-aware gate, main-dispatch ruling 2026-10-01): for ops with
    # REV9 dynamic candidates the launch may legally dispatch to a fallback module
    # (0.6.17 serves fa2 single decode as q_len-1 prefill), so "required module has
    # no binary" is NOT staleness when another candidate module is built AND
    # instrumented — the launch will emit marker rows from that module.  Conversely,
    # if NO candidate module is built+instrumented the op cannot produce rows: keep
    # the first candidate record stale with a loud error.
    dynamic = REV9_DYNAMIC_CANDIDATES.get(op_class)
    if dynamic:
        cand_modules = {module for module, _, _ in dynamic}
        instrumented = [
            r for r in reports
            if r["module_key"] in cand_modules and r.get("binaries")
            and not r.get("stale")
            and any(
                (v or {}).get("sr_globaltimerlo_reads", 0) >= 2
                for b in r["binaries"]
                for v in (b.get("kernels") or {}).values()
            )
        ]
        if instrumented:
            for r in reports:
                if r["module_key"] in cand_modules:
                    # The launch dispatches to the instrumented candidate; other
                    # candidates (absent or even uninstrumented) never launch, so
                    # they cannot affect marker rows.  The timing layer's dynamic
                    # resolution + duration guard still cover the actual launch.
                    note = (
                        "dispatch resolved to instrumented module "
                        f"{instrumented[0]['module_key']} (T0011.g)"
                    )
                    r["stale"] = False
                    r["error"] = note
                    for b in r.get("binaries") or []:
                        b["gate_note"] = note
        elif all(not r.get("binaries") for r in reports if r["module_key"] in cand_modules):
            first = next(r for r in reports if r["module_key"] in cand_modules)
            first["stale"] = True
            first["error"] = (
                "no dynamic candidate module built/instrumented for this op "
                "(T0011.g dispatch-aware gate)"
            )
    return reports


def purge_stale_modules(reports: list[dict]) -> list[str]:
    """Delete the cached_ops dirs of binaries that failed the freshness gate."""
    import shutil  # noqa: PLC0415

    purged = []
    for rec in reports or []:
        if not rec.get("stale"):
            continue
        for binary in rec.get("binaries", []):
            so_path = binary.get("binary")
            if not so_path:
                continue
            so = Path(so_path)
            if so.is_file():
                shutil.rmtree(so.parent, ignore_errors=True)
                purged.append(str(so.parent))
    return purged


def _clear_fi_module_caches(flashinfer_module) -> int:
    """Clear every @functools.cache'd FlashInfer module getter (T0011.d den-r12 fix A).

    get_page_module / get_batch_decode_module / get_batch_prefill_module / … are
    functools.cache'd (upstream page.py:41, decode.py:289, prefill.py:451): after the
    freshness gate purges a stale cached_ops dir, a plain rebuild() returns the CACHED
    module object (stale .so handle, stale registry entry) and never recompiles — the
    second audit then finds no binary on disk and the gate deadlocks.  Clearing the
    getters forces build_and_load to regenerate + relink the purged module and register
    a fresh entry.
    """
    import types  # noqa: PLC0415

    cleared = 0
    seen: set[int] = set()
    stack = [flashinfer_module]
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        for name, obj in list(vars(module).items()):
            if callable(obj) and hasattr(obj, "cache_clear"):
                try:
                    obj.cache_clear()
                    cleared += 1
                except Exception:
                    pass
            elif isinstance(obj, types.ModuleType) and str(getattr(obj, "__name__", "")).startswith(
                "flashinfer"
            ):
                stack.append(obj)
    return cleared


def ensure_fresh_instrumentation(op_class: str, case: dict | None, rebuild: Callable[[], object],
                                 flashinfer_module=None, cuobjdump: str = "cuobjdump") -> dict:
    """Audit -> purge stale cached_ops -> rebuild -> re-audit (T0011.d).

    ``case`` resolves the backend-aware primary (required_gate_modules): absence of a
    fallback-only module is not staleness (0dfc244e den finding, superseded here by the
    stricter backend-primary rule per T0011 integration decision).
    ``rebuild`` re-runs the op build path; when ``flashinfer_module`` is given, the
    functools module-getter caches are cleared first so the rebuild actually reloads the
    purged module (den-r12 fix A: cached getters made the rebuild a no-op and deadlocked
    the gate).  Raises when the re-audit still finds uninstrumented binaries — a loud
    dynamic-audit failure instead of empty marker rows.
    """
    audit = jit_freshness_audit(op_class, case, cuobjdump)
    result: dict = {"first_audit": audit}
    if not any(rec.get("stale") for rec in audit):
        result["status"] = "fresh"
        return result
    result["purged"] = purge_stale_modules(audit)
    if flashinfer_module is not None:
        result["caches_cleared"] = _clear_fi_module_caches(flashinfer_module)
    rebuild()
    reaudit = jit_freshness_audit(op_class, case, cuobjdump)
    result["second_audit"] = reaudit
    still_stale = [rec["module_key"] for rec in reaudit if rec.get("stale")]
    if still_stale:
        # T0011.den-r12: persist the full audit chain at raise time (run_case aborts
        # lose the result payload).  Re-applied on the .d version after the
        # integration merge dropped the earlier insertion.
        import json as _json
        import os as _os
        _root = _os.environ.get("FLASHINFER_DELIVERY_ARTIFACTS_ROOT")
        if _root:
            try:
                _dir = Path(_root) / "logs" / "freshness_gate"
                _dir.mkdir(parents=True, exist_ok=True)
                (_dir / f"{op_class}.json").write_text(
                    _json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
            except OSError:
                pass
        raise RuntimeError(
            f"marker freshness gate failed after purge+rebuild for {op_class}: modules "
            f"{still_stale} still define candidate kernels without marker SASS (see "
            "first_audit/second_audit in the result payload); the delivery binary would "
            "produce empty marker rows"
        )
    result["status"] = "rebuilt"
    return result


# T0005 rev9 (far-holdout G0 on-site item): runtime dispatch for some ops does not
# match the static config primary — H100 probe evidence (2026-09-30, probe2/):
#   - batch/single decode with backend=fa2 is served by the PREFILL module
#     (batch_prefill_with_kv_cache_* .so; BatchPrefillWithPagedKVCacheKernel;
#     0.6.17 implements paged decode as q_len-1 prefill), backend=auto builds the
#     decode module; batch_mla_paged serves fa2/fa3-fp8 from the mla module and
#     fa3-bf16 from mla_hopper.  Candidate lists below resolve dynamically at the
#     first launch (collect_marker_main_kernel resolve_dynamic), the container
#     records whichever candidate carried rows; duration guard uses the resolved
#     kernel.  Order: designated hopper/static path first, runtime fallbacks after.
#
# T0011.d extension (far45 runtime-variant class, rounds_table rev9 leftovers):
#   - single_prefill backend=fa3: T0005 rounds show the static fa3 -> prefill_sm90
#     override resolving a REGISTERED module with EMPTY rows (probe2: prefill
#     registered, prefill_sm90_quant missing; runtime_sass_audit: the loaded
#     prefill/prefill_sm90 .so carried no marker ops) — the op may serve the fa2
#     SinglePrefillWithKVCacheKernel (prefill module) instead.  Resolve dynamically.
#   - merge_states: unequal-len inputs serve PersistentVariableLengthMergeStatesKernel
#     (cascade region 2); rounds_table merge_states_02 failed the duration guard on the
#     static MergeStatesKernel expectation ("no CUDA activity ... in the profiler window").
#   - batch_prefill_paged / batch_prefill_ragged: same prefill_sm90/prefill duality as
#     decode (fa3 designation does not guarantee the sm90 module served the launch).
#   - batch_decode_paged: add the fa2_mla region so backend=auto cases that dispatch to
#     BatchDecodeWithPagedKVCacheKernelMLA resolve instead of hard-failing.
#   - single_decode: add the fa2 single-prefill module (q_len-1 prefill implementation).
REV9_DYNAMIC_CANDIDATES: dict[str, list[tuple[str, int, str]]] = {
    "batch_decode_paged": [
        ("decode", 1, "BatchDecodeWithPagedKVCacheKernel"),
        ("decode", 2, "BatchDecodeWithPagedKVCacheKernelMLA"),
        ("prefill", 2, "BatchPrefillWithPagedKVCacheKernel"),
        ("prefill_sm90", 0, "PrefillWithKVCacheKernel"),
    ],
    "single_decode": [
        ("decode", 0, "SingleDecodeWithKVCacheKernel"),
        ("prefill", 0, "SinglePrefillWithKVCacheKernel"),
        ("prefill", 2, "BatchPrefillWithPagedKVCacheKernel"),
        ("prefill_sm90", 0, "PrefillWithKVCacheKernel"),
    ],
    "batch_mla_paged": [
        ("mla_hopper", 0, "BatchMLAPageAttentionHopperKernel"),
        ("mla", 0, "BatchMLAPagedAttentionKernel"),
    ],
    "single_prefill": [
        ("prefill_sm90", 0, "PrefillWithKVCacheKernel"),
        ("prefill_sm90_quant", 0, "FP8PrefillWithKVCacheKernel"),
        ("prefill", 0, "SinglePrefillWithKVCacheKernel"),
    ],
    "batch_prefill_paged": [
        ("prefill_sm90", 0, "PrefillWithKVCacheKernel"),
        ("prefill_sm90_quant", 0, "FP8PrefillWithKVCacheKernel"),
        ("prefill", 2, "BatchPrefillWithPagedKVCacheKernel"),
    ],
    "batch_prefill_ragged": [
        ("prefill_sm90", 0, "PrefillWithKVCacheKernel"),
        ("prefill", 1, "BatchPrefillWithRaggedKVCacheKernel"),
    ],
    "merge_states": [
        ("cascade", 1, "MergeStatesKernel"),
        ("cascade", 2, "PersistentVariableLengthMergeStatesKernel"),
    ],
    # T0011.f (rev5 item 2 sibling fix, H100 t11a-diag-fp8 evidence 2026-10-01): the
    # deep_gemm NVRTC runner dispatches either fp8_gemm_kernel (region 0) or the swapAB
    # variant fp8_gemm_kernel_swapAB (region 1, fi_marker_secondary_ptr).  Kineto shows the
    # event as `void deep_gemm::fp8_gemm_kernel_swapAB<...>` — the static region-0 stem is
    # boundary-rejected by kernel_symbol_contains ("fp8_gemm_kernel" is followed by "_" in
    # the swapAB name), so the duration guard found no activity.  Resolve the serving
    # region dynamically like the mamba/decode multi-variant ops.
    "fp8_blockscale_gemm": [
        ("fp8_blockscale_gemm_sm90", 0, "fp8_gemm_kernel"),
        ("fp8_blockscale_gemm_sm90", 1, "fp8_gemm_kernel_swapAB"),
    ],
}


def run_marker_pass(torch, prepared, op_class: str, case: dict, channel, per_key: dict,
                    unit_slot_filter: set[int] | None = None,
                    unit_warps_per_cta: int | None = None) -> dict:
    """27 raw launches -> 9 formal samples on the resolved primary region.

    ``unit_slot_filter`` reduces every launch to the research unit's own
    slots (``marker_timing.unit_slot_set``).  With ``unit_warps_per_cta`` the
    pass returns the warp ids actually present on every selected CTA in
    every launch.  The whole-kernel CUDA activity rides along as a
    diagnostic only.
    """
    entry = mcc.OP_CLASSES[op_class]
    candidates = REV9_DYNAMIC_CANDIDATES.get(op_class)
    if candidates is not None:
        primary = candidates[0]
        dynamic = True
    else:
        primary = resolve_primary(op_class, case)
        dynamic = op_class == "mamba_state_update"
        if dynamic:
            candidates = [
                (module, region, kernel)
                for module, region, kernel in [primary] + list(entry.get("secondary") or [])
            ]
    kwargs: dict = {}
    if unit_slot_filter is not None:
        kwargs["unit_slot_filter"] = unit_slot_filter
        if unit_warps_per_cta is not None:
            kwargs["unit_warps_per_cta"] = unit_warps_per_cta
    result = mt.collect_marker_main_kernel(
        torch, prepared.run, channel, primary,
        resolve_dynamic=dynamic, candidates=candidates, **kwargs,
    )
    result["op_class"] = op_class
    result["channel"] = per_key
    return result


