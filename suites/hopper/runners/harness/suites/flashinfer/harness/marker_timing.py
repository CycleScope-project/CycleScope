"""FlashInfer in-kernel clock64/globaltimer marker timing (T0005 修订 3-② / 修订 5).

Kernel side: scripts/patch_wheel_marker.py instruments the JIT sources listed in
harness/marker_channel_config.py (all edits inside `#ifdef FI_DELIVERY_MARKER_TIMING`; every
patched TU enrolls its setter into the shared registry via
FI_DELIVERY_MARKER_REGISTER_MODULE).  Every instrumented kernel writes one
[start_clock64, end_clock64, start_globaltimer, end_globaltimer, smid, cluster_dim_packed,
cluster_ctarank] row per (cta, warp) around its outermost body — field layout, validity rule
and reduction mirror the FA3/CUTLASS device_kernel override and the FlashMLA patch exactly.

Host side (this module): dlopen build/libdelivery_fi_marker.so (linked into every instrumented
JIT module via FLASHINFER_EXTRA_LDFLAGS — the one shared registry instance), discover the
registered TUs, hand each 4 region buffers ("the runner sets the device pointer before
launching the kernel"; no ABI change).  Per raw launch: zero buffers -> one op -> synchronize
-> read -> reduce with the FA3 formula (timer_span = max(end_timer)-min(start_timer); ratio =
median per-row (end_clock-start_clock)/(end_timer-start_timer); cycles = span*ratio).  The
formal protocol records 27 raw main-kernel launches (3 x 9) and reduces them to 9 samples by
the median of each consecutive triple.  The whole-kernel CUDA activity has
two contracts: the historical whole-launch pass (no unit slot filter; the
far45/296 denominator producer) keeps the FA3-style +/-10% duration-match
guard and records status "pass", while the strict same-unit pass
(unit_slot_filter given, decision 2026-10-09) records the activity as a
diagnostic only -- it never gates the unit-scoped marker interval.  The
marker scope stays
`outermost_in_kernel_clock_marker_interval`; the whole-operation globaltimer span is never
used as the marker denominator (docs/simulator.md §2.5).
"""

from __future__ import annotations

import ctypes
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import marker_channel_config as mcc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
from hoppersim.kernel_symbols import kernel_symbol_contains  # noqa: E402

MAX_ROWS = 1 << 18
FIELDS = 7
MARKER_LEN = MAX_ROWS * FIELDS
U64_MOD = 1 << 64
SLOTS_PER_MODULE = 4

FIELD_START_CLOCK = 0
FIELD_END_CLOCK = 1
FIELD_START_TIMER = 2
FIELD_END_TIMER = 3
FIELD_SMID = 4
FIELD_CLUSTER_PACKED = 5
FIELD_CLUSTER_RANK = 6

MARKER_SCOPE = "outermost_in_kernel_clock_marker_interval"
MARKER_METHOD = "device_kernel_clock64_over_globaltimer"
MARKER_HARDWARE_METRIC = "max_hardware_warp_end_minus_min_hardware_warp_start"
MARKER_INTERVAL_DEFINITION = "max(all_warp_end)-min(all_warp_start)"
CLOCK_MARKER_POLICY = "outermost_pair"
MARKER_ENV_FLAG = "FLASHINFER_MARKER_TIMING"
MARKER_MACRO_FLAG = "-DFI_DELIVERY_MARKER_TIMING"
MARKER_LIB_ENV = "FI_DELIVERY_MARKER_LIB"

RAW_SPANS_PER_SAMPLE = 3
FORMAL_SAMPLES = 9
RAW_LAUNCHES_REQUIRED = RAW_SPANS_PER_SAMPLE * FORMAL_SAMPLES
DURATION_MATCH_RELATIVE = 0.10
REGISTRY_SET_SYMBOL = "flashinfer_delivery_marker_set_buffers"
REGISTRY_COUNT_SYMBOL = "flashinfer_delivery_marker_module_count"
REGISTRY_NAMES_SYMBOL = "flashinfer_delivery_marker_module_names"


def jit_module_dir() -> Path:
    """FLASHINFER_JIT_DIR of the installed flashinfer (cached_ops parent)."""
    from flashinfer.jit import env as jit_env  # noqa: PLC0415

    return Path(jit_env.FLASHINFER_JIT_DIR)


def marker_timing_requested(case: dict | None = None, env: dict | None = None) -> bool:
    """Marker mode switch: env flag or per-case key. Default off."""
    environ = os.environ if env is None else env
    return environ.get(MARKER_ENV_FLAG) == "1" or bool((case or {}).get("marker_timing"))


def enable_jit_marker_mode(lib_path: Path, env: dict | None = None) -> dict:
    """Enable the marker macro + registry linkage for every JIT build.

    Call before the first op build.  The patched sources keep their #ifdef guards, so without
    this the delivery tree compiles upstream-identical JIT modules; ninja's flag change forces
    a deterministic rebuild into FLASHINFER_JIT_DIR, which is what makes timing and capture
    share one binary (修订 5).
    """
    environ = os.environ if env is None else env
    cuda_flags = environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if MARKER_MACRO_FLAG not in cuda_flags:
        environ["FLASHINFER_EXTRA_CUDAFLAGS"] = (cuda_flags + " " + MARKER_MACRO_FLAG).strip()
    ld_flags = environ.get("FLASHINFER_EXTRA_LDFLAGS", "")
    link = f"-L{lib_path.parent} -ldelivery_fi_marker -Wl,-rpath,{lib_path.parent}"
    if "-ldelivery_fi_marker" not in ld_flags:
        environ["FLASHINFER_EXTRA_LDFLAGS"] = (ld_flags + " " + link).strip()
    environ[MARKER_LIB_ENV] = str(lib_path)
    return {
        "cuda_flags": environ["FLASHINFER_EXTRA_CUDAFLAGS"],
        "ld_flags": environ["FLASHINFER_EXTRA_LDFLAGS"],
        "registry_lib": str(lib_path),
    }


SM_CAP_ENV = "FI_DELIVERY_SM_CAP"
SM_CAP_MACRO_FLAG = "-DFI_DELIVERY_SM_CAP"


def sm_cap_requested(env: dict | None = None) -> int:
    """Host-side SM cap value from $FI_DELIVERY_SM_CAP; 0 = disabled (T0011.d rev1 item 2)."""
    environ = os.environ if env is None else env
    try:
        cap = int(environ.get(SM_CAP_ENV, "0") or "0")
    except ValueError:
        cap = 0
    return max(cap, 0)


def enable_sm_cap_mode(env: dict | None = None) -> dict | None:
    """Opt-in host-side SM cap for the FA3/MLA-hopper persistent launches (T0011.d rev1 ②).

    Activated by $FI_DELIVERY_SM_CAP=N (N>0): the wheel's patched host plan/launch sites
    (PrefillSM90Plan work bins, MLAPlan num_blks_y, prefill_sm90 launcher grids) clamp the
    device SM count to N under -DFI_DELIVERY_SM_CAP, and harness/sm_cap.py clamps the
    python-side get_device_properties().multi_processor_count to the same value.  Plan and
    launch stay consistent inside each JIT module; device kernels are untouched, so the
    no-cap compile (and the SASS audit) reproduces upstream bit-for-bit — the FlashMLA
    build-patch contract.  FA2 decode/prefill paths are not clamped (GQA-knob cases scale
    by h_kv=1 per rev1).
    """
    environ = os.environ if env is None else env
    cap = sm_cap_requested(environ)
    if cap <= 0:
        return None
    cuda_flags = environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if SM_CAP_MACRO_FLAG not in cuda_flags:
        environ["FLASHINFER_EXTRA_CUDAFLAGS"] = (cuda_flags + " " + SM_CAP_MACRO_FLAG).strip()
    return {"sm_cap": cap, "cuda_flags": environ["FLASHINFER_EXTRA_CUDAFLAGS"]}


def regions_for_module(module_key: str) -> int:
    """Highest region index any config kernel of this module key writes, + 1."""
    region_ids = {
        region
        for _kernel, (_module_key, region) in mcc.KERNEL_REGIONS.items()
        if _module_key == module_key
    }
    return (max(region_ids) + 1) if region_ids else 1


def candidate_jit_binaries(module_key: str) -> list[Path]:
    """Loaded JIT .so candidates for one config module key (prefix + sm90-suffix rule)."""
    base = jit_module_dir()
    if not base.is_dir():
        return []
    candidates = []
    for entry in sorted(base.iterdir()):
        so_path = entry / f"{entry.name}.so"
        if not so_path.is_file():
            continue
        if mcc.match_module(entry.name, module_key, require_sm90=entry.name.endswith("_sm90")):
            candidates.append(so_path)
    return candidates


def resolve_primary_binary(module_key: str, kernel_stem: str, cuobjdump: str = "cuobjdump") -> Path:
    """The JIT .so that actually defines kernel_stem (disambiguated by SASS dump)."""
    import subprocess  # noqa: PLC0415

    candidates = candidate_jit_binaries(module_key)
    if not candidates:
        raise RuntimeError(f"no loaded JIT binary found for module key {module_key!r}")
    if len(candidates) == 1:
        return candidates[0]
    for so_path in candidates:
        try:
            dump = subprocess.run([cuobjdump, "--dump-sass", str(so_path)],
                                  capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if kernel_stem in (dump.stdout or ""):
            return so_path
    raise RuntimeError(
        f"none of {candidates} defines {kernel_stem}; the op picked an unexpected variant"
    )


def warps_per_cta(block: list) -> int:
    """CUDA warps in one CTA under the device marker row linearization.

    Same domain as :func:`unit_slot_set`: ``bx`` multiple of 32, or a 1-D block.
    This is the capture slot stride, not the set of warps that executed the marker.
    """
    bx, by, bz = (int(v) for v in block)
    if not (bx % 32 == 0 or (by <= 1 and bz <= 1)):
        raise ValueError(
            f"block {(bx, by, bz)} is not supported by the device marker row "
            "format: CUDA warps follow the linear thread id, so with bx not a "
            "multiple of 32 and y/z > 1 one real warp spans several y/z rows "
            "while fi_marker_row_index would give them separate slots"
        )
    warp_x = (bx + 31) // 32
    return warp_x * (by if by > 1 else 1) * (bz if bz > 1 else 1)


def observed_clock_marker_warps(
    launch_slots: list[list[int]],
    unit_slot_filter: set[int],
    warps_per_cta_count: int,
) -> list[int]:
    """Shared marker-participant warp ids from valid unit slot identities.

    Each launch must carry the same warp set on every selected CTA. A missing
    CTA, a duplicate slot, or a set that differs across CTAs or launches
    fails closed. Warps that never appear are not participants. The result is
    not ``range(block_warps)`` and not a producer capacity constant.
    """
    if not isinstance(warps_per_cta_count, int) or isinstance(warps_per_cta_count, bool) or warps_per_cta_count < 1:
        raise ValueError(f"warps_per_cta {warps_per_cta_count!r} is not a positive int")
    if not unit_slot_filter:
        raise ValueError("unit slot filter is empty")
    selected = set()
    for slot in unit_slot_filter:
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
            raise ValueError(f"unit slot {slot!r} is not a buffer index")
        selected.add(slot // warps_per_cta_count)
    if not launch_slots:
        raise RuntimeError("no marker launches to derive clock_marker_warps")
    shared: list[int] | None = None
    for index, slots in enumerate(launch_slots):
        unit_slots = [slot for slot in slots if slot in unit_slot_filter]
        if len(unit_slots) != len(set(unit_slots)):
            raise RuntimeError(
                f"marker launch {index}: duplicate unit marker slot in the valid rows"
            )
        by_cta: dict[int, list[int]] = {}
        for slot in unit_slots:
            by_cta.setdefault(slot // warps_per_cta_count, []).append(slot % warps_per_cta_count)
        if set(by_cta) != selected:
            raise RuntimeError(
                f"marker launch {index}: missing unit CTA marker rows; "
                f"saw CTAs {sorted(by_cta)} expected {sorted(selected)}"
            )
        sets = [sorted(warps) for warps in by_cta.values()]
        if any(warps != sets[0] for warps in sets):
            raise RuntimeError(
                f"marker launch {index}: marker participating warps differ across CTAs: {sets}"
            )
        if not sets[0]:
            raise RuntimeError(f"marker launch {index}: unit CTAs carried no marker warps")
        if shared is None:
            shared = sets[0]
        elif sets[0] != shared:
            raise RuntimeError(
                f"marker launch {index}: marker participating warps {sets[0]} "
                f"differ from earlier launches {shared}"
            )
    return list(shared)


def unit_slot_set(ctas: list, grid: list, block: list) -> set[int]:
    """Buffer slot indexes of one research unit's (cta, warp) pairs.

    Mirrors the device header's fi_marker_row_index linearization exactly
    (y/z-aware warp numbering, cta = x + gridDim.x*(y + gridDim.y*z)), so a
    bare measurement can reduce ONLY the unit's own rows -- the same-unit
    marker denominator contract (strict same-unit marker verification).  A
    row outside this set belongs to a different CTA/warp of the launch and
    never enters the unit's span.
    """
    gx, gy, gz = (int(v) for v in grid)
    bx, by, bz = (int(v) for v in block)
    stride = warps_per_cta(block)
    warp_x = (bx + 31) // 32
    slots: set[int] = set()
    for cta in ctas or []:
        cx, cy, cz = (int(v) for v in cta)
        if not (0 <= cx < gx and 0 <= cy < gy and 0 <= cz < gz):
            raise ValueError(f"unit cta {(cx, cy, cz)} outside grid {(gx, gy, gz)}")
        cta_linear = cx + gx * (cy + gy * cz)
        for wy in range(by if by > 1 else 1):
            for wz in range(bz if bz > 1 else 1):
                for wx in range(warp_x):
                    warp = wx
                    if by > 1:
                        warp += wy * warp_x
                    if bz > 1:
                        warp += wz * warp_x * (by if by > 1 else 1)
                    slots.add(cta_linear * stride + warp)
    if not slots:
        raise ValueError("unit ctas produced an empty slot set")
    return slots


def decode_cluster_packed(packed: int) -> tuple[int, int, int, int]:
    """Same field packing as the CUTLASS/FA3/FlashMLA device_kernel marker rows."""
    nx = packed & 0xFF
    ny = (packed >> 8) & 0xFF
    nz = (packed >> 16) & 0xFF
    ncta = (packed >> 32) & 0xFF
    if ny == 0 and nz == 0:
        ncta = packed
        nx = ncta
        ny = 1
        nz = 1
    elif ncta == 0:
        ncta = nx * ny * nz
    return nx, ny, nz, ncta


def reduce_region_rows(rows: list[list[int]]) -> dict | None:
    """One launch sample for one (registry entry, region) — FA3 read_launches formula.

    A row counts only when row[0] > 0 and row[1] > row[0] and row[2] > 0 and row[3] > row[2];
    the launch is a sample only when every recorded warp carries the complete pair.
    """
    rows = [
        row for row in rows
        if row[FIELD_START_CLOCK] > 0
        and row[FIELD_END_CLOCK] > row[FIELD_START_CLOCK]
        and row[FIELD_START_TIMER] > 0
        and row[FIELD_END_TIMER] > row[FIELD_START_TIMER]
    ]
    if not rows:
        return None
    first_timer = min(row[FIELD_START_TIMER] for row in rows)
    last_timer = max(row[FIELD_END_TIMER] for row in rows)
    if last_timer <= first_timer:
        return None
    ratios = [
        float(row[FIELD_END_CLOCK] - row[FIELD_START_CLOCK])
        / float(row[FIELD_END_TIMER] - row[FIELD_START_TIMER])
        for row in rows
    ]
    sm_ids = sorted({row[FIELD_SMID] for row in rows})
    cluster_ranks = sorted({row[FIELD_CLUSTER_RANK] for row in rows})
    cluster_nx = cluster_ny = cluster_nz = 1
    cluster_ncta = 0
    for row in rows:
        nx, ny, nz, ncta = decode_cluster_packed(row[FIELD_CLUSTER_PACKED])
        cluster_nx, cluster_ny, cluster_nz = nx, ny, nz
        cluster_ncta = max(cluster_ncta, ncta)
    ratio = statistics.median(ratios)
    timer_span = float(last_timer - first_timer)
    return {
        "timer_span_ns": timer_span,
        "ratio": ratio,
        "cycles": timer_span * ratio,
        "sm_ids": sm_ids,
        "cluster_dim": [cluster_nx, cluster_ny, cluster_nz],
        "cluster_ncta": cluster_ncta,
        "cluster_ranks": cluster_ranks,
        "valid_rows": len(rows),
    }


def summarize_marker_samples(launches: list[dict]) -> dict:
    """Last 27 raw launches by launch index; median of each consecutive triple -> 9."""
    ordered = list(launches)
    if len(ordered) < RAW_LAUNCHES_REQUIRED:
        raise RuntimeError(
            f"marker timing needs {RAW_LAUNCHES_REQUIRED} main-kernel launches, "
            f"got {len(ordered)}"
        )
    raw = ordered[-RAW_LAUNCHES_REQUIRED:]
    raw_cycles = [float(row["cycles"]) for row in raw]
    samples = [
        statistics.median(raw_cycles[index:index + RAW_SPANS_PER_SAMPLE])
        for index in range(0, len(raw_cycles), RAW_SPANS_PER_SAMPLE)
    ]
    mean = statistics.fmean(samples)
    cv = statistics.stdev(samples) / mean * 100.0 if len(samples) > 1 and mean else 0.0
    return {
        "samples_cycles": samples,
        "samples_cycles_raw": raw_cycles,
        "raw_spans_per_sample": RAW_SPANS_PER_SAMPLE,
        "sample_count": len(samples),
        "raw_sample_count": len(raw_cycles),
        "median_cycles": statistics.median(samples),
        "cv_percent": cv,
        "aggregation": "median_of_each_consecutive_triple",
    }


def build_marker_main_kernel_timing(summary: dict, launch: dict, primary: tuple) -> dict:
    """timing.main_kernel container — same schema as the FA3/CUTLASS/FlashMLA suites."""
    module_key, region, kernel = primary
    return {
        "schema_version": "delivery.matched-kernel-timing.v1",
        "status": "matched",
        "eligible_for_simulator_accuracy": True,
        "scope": MARKER_SCOPE,
        "method": MARKER_METHOD,
        "hardware_metric": MARKER_HARDWARE_METRIC,
        "marker_interval_definition": MARKER_INTERVAL_DEFINITION,
        "clock_marker_policy": CLOCK_MARKER_POLICY,
        "samples_cycles": list(summary["samples_cycles"]),
        "samples_cycles_raw": list(summary["samples_cycles_raw"]),
        "raw_spans_per_sample": RAW_SPANS_PER_SAMPLE,
        "sample_count": summary["sample_count"],
        "raw_sample_count": summary["raw_sample_count"],
        "aggregation": summary["aggregation"],
        "median_cycles": summary["median_cycles"],
        "cv_percent": summary["cv_percent"],
        "kernel_name_expected": kernel,
        "kernel_name_observed": kernel,
        "marker_module": module_key,
        "marker_region": region,
        "grid_id": launch["launch_index"],
        "cluster_dim": list(launch["cluster_dim"]),
        "cluster_ncta": launch["cluster_ncta"],
        "cluster_ranks": list(launch["cluster_ranks"]),
        "sm_ids": list(launch["sm_ids"]),
        "valid_rows": launch["valid_rows"],
        "whole_operation_timing_must_not_be_used_for_simulator_accuracy": True,
    }


def measure_primary_kernel_activity_ns(torch: Any, run: Callable[[], object],
                                       kernel_name: str,
                                       alt_names: list[str] | None = None,
                                       profiled_marker_read: Callable[[], "dict | None"] | None = None,
                                       ) -> tuple[float, str, list[dict], dict | None]:
    """Median CUDA activity duration of the primary kernel (kineto) -- a
    whole-kernel diagnostic for the marker pass (it no longer gates; the two
    readings measure different ranges).

    The kernel name is the RESOLVED serving kernel (collect_marker_main_kernel resolves the
    dynamic primary before measuring); ``alt_names`` are the other candidate kernels of
    the same op — when the resolved name has no kineto event (demangling variants), the scan
    falls back to them and reports which name actually matched, so the guard never
    verifies against a name the launch did not produce.

    ``profiled_marker_read`` (optional) runs after the profiler context exits
    and the run's synchronize (the GPU->host copy stays out of the profiled
    activity): it reads the SAME physical launch's marker buffer, giving the
    profiled-marker span next to the matched activity event — the discriminator between
    profile perturbation and a real body/activity range difference (listed separately
    from the 27 unprofiled bare rows; no fixed-offset conclusion is drawn from it).
    """
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    # The slots read is a GPU->host copy: it runs after the profiler context
    # exits and the synchronize, before the next run, so the diagnostic copy
    # is never counted as profiled activity.  It still reads the buffer the
    # same physical launch wrote.
    profiled_marker = profiled_marker_read() if profiled_marker_read else None
    durations: list[float] = []
    matched_events: list[dict] = []
    matched_name = ""
    for stem in [kernel_name] + [n for n in (alt_names or []) if n != kernel_name]:
        durations = []
        matched_events = []
        for event in prof.events():
            name = str(getattr(event, "name", ""))
            if kernel_symbol_contains(name, stem):
                device_time_us = float(getattr(event, "device_time", 0.0) or 0.0)
                if device_time_us > 0:
                    time_range = getattr(event, "time_range", None)
                    matched_events.append({
                        "name": name,
                        "device_type": str(getattr(event, "device_type", "")),
                        "time_range_start_us": getattr(time_range, "start", None),
                        "time_range_end_us": getattr(time_range, "end", None),
                        "duration_ns": device_time_us * 1000.0,
                    })
                    durations.append(device_time_us * 1000.0)
        if durations:
            matched_name = stem
            break
    if not durations:
        raise RuntimeError(
            f"no CUDA activity for {kernel_name} (or alt names {alt_names}) in the profiler "
            "window; the whole-kernel activity diagnostic is unavailable"
        )
    if matched_name != kernel_name:
        # Guard on the kernel the op actually dispatched; the substitution is recorded by
        # the caller in duration_match for the container.
        kernel_name = matched_name
    return statistics.median(durations), kernel_name, matched_events, profiled_marker


class MarkerChannel:
    """ctypes bridge to the shared registry inside libdelivery_fi_marker.so.

    One buffer set per registered TU (registry order); every entry knows the patch-file module
    key it registered under (harness/marker_channel_config.py keys), so the primary region is
    addressed as (module_key, region) and resolved to the registry entry carrying rows.
    """

    def __init__(self, torch: Any, lib_path: Path):
        if not Path(lib_path).is_file():
            raise RuntimeError(
                f"delivery marker registry library not found: {lib_path}; build it with "
                "scripts/build.sh (delivery builds compile harness/override/"
                "fi_delivery_marker_registry.cpp)"
            )
        self.torch = torch
        self.lib = ctypes.CDLL(str(lib_path))
        try:
            set_buffers = getattr(self.lib, REGISTRY_SET_SYMBOL)
            module_count = getattr(self.lib, REGISTRY_COUNT_SYMBOL)
            module_names = getattr(self.lib, REGISTRY_NAMES_SYMBOL)
        except AttributeError as error:
            raise RuntimeError(
                f"{lib_path} does not export the delivery marker registry symbols: {error}"
            ) from error
        set_buffers.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        set_buffers.restype = None
        module_count.argtypes = []
        module_count.restype = ctypes.c_int
        module_names.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        module_names.restype = ctypes.c_int

        count = int(module_count())
        if count <= 0:
            raise RuntimeError(
                "the delivery marker registry is empty: no instrumented JIT module has been "
                "loaded yet; build/load the op first (the op build must run with "
                f"{MARKER_MACRO_FLAG}, see enable_jit_marker_mode)"
            )
        self._torch = torch
        self._module_count_fn = module_count
        self._module_names_fn = module_names
        self._refresh_entries(count)
        self._set_buffers = set_buffers
        self.set_buffers()

    def _refresh_entries(self, count: int) -> None:
        torch = self._torch
        stride = 64
        storage = ctypes.create_string_buffer(stride * count)
        returned = int(self._module_names_fn(storage, stride, count))
        if returned != count:
            raise RuntimeError(f"registry size changed mid-query: {returned} != {count}")
        self.entries = [storage[i * stride:(i * stride) + stride].decode().rstrip("\x00")
                        for i in range(count)]
        self.regions = [regions_for_module(name) for name in self.entries]
        self.buffers = [
            [
                torch.zeros(MARKER_LEN, dtype=torch.int64, device="cuda")
                for _ in range(regions)
            ]
            for regions in self.regions
        ]

    def ensure_entries(self) -> bool:
        """Pick up modules registered after this channel was constructed (T0011.d den-r12
        fix D).

        Entries registered late (module .so dlopen'd after install_channel) had no buffer
        pointers, so their marker rows were silently null-guarded away and resolve_entry
        could never see them; same-name re-registrations (freshness-gate rebuild) also
        shift entry indexes.  Re-enumerates the registry, allocates buffers for any new
        entries and re-points ALL entries (order can change).  Returns True when the
        entry set grew.
        """
        count = int(self._module_count_fn())
        if count <= len(self.entries):
            return False
        self._refresh_entries(count)
        self.set_buffers()
        return True

    @property
    def module_keys(self) -> list[str]:
        return list(self.entries)

    def set_buffers(self) -> None:
        pointers = (ctypes.c_void_p * (SLOTS_PER_MODULE * len(self.buffers)))()
        for entry_index, buffer_list in enumerate(self.buffers):
            for region_index, buffer in enumerate(buffer_list):
                pointers[entry_index * SLOTS_PER_MODULE + region_index] = ctypes.c_void_p(
                    buffer.data_ptr()
                )
        self._set_buffers(pointers, SLOTS_PER_MODULE * len(self.buffers))

    def zero_all(self) -> None:
        for buffer_list in self.buffers:
            for buffer in buffer_list:
                buffer.zero_()

    def zero_entry(self, entry_index: int) -> None:
        for buffer in self.buffers[entry_index]:
            buffer.zero_()

    def read_region(self, entry_index: int, region: int) -> list[list[int]]:
        table = self.buffers[entry_index][region].view(MAX_ROWS, FIELDS)
        start_clock = table[:, FIELD_START_CLOCK]
        end_clock = table[:, FIELD_END_CLOCK]
        start_timer = table[:, FIELD_START_TIMER]
        end_timer = table[:, FIELD_END_TIMER]
        valid = (
            (start_clock > 0) & (end_clock > start_clock)
            & (start_timer > 0) & (end_timer > start_timer)
        )
        rows = table[valid].cpu().tolist()
        return [[int(value) % U64_MOD for value in row] for row in rows]

    def read_region_slots(self, entry_index: int, region: int) -> list[list[int]]:
        """Valid rows with their buffer slot index prepended (coverage facts).

        The slot index is the row position the device header writes
        (fi_marker_row_index: linearized (cta, warp), y/z-aware) — the
        per-launch CTA+warp coverage the reduced sample discards.
        """
        table = self.buffers[entry_index][region].view(MAX_ROWS, FIELDS)
        start_clock = table[:, FIELD_START_CLOCK]
        end_clock = table[:, FIELD_END_CLOCK]
        start_timer = table[:, FIELD_START_TIMER]
        end_timer = table[:, FIELD_END_TIMER]
        valid = (
            (start_clock > 0) & (end_clock > start_clock)
            & (start_timer > 0) & (end_timer > start_timer)
        )
        rows = table[valid].cpu().tolist()
        slots = valid.nonzero(as_tuple=False).flatten().cpu().tolist()
        return [
            [int(slot)] + [int(value) % U64_MOD for value in row]
            for slot, row in zip(slots, rows)
        ]

    def resolve_entry(self, module_key: str) -> list[int]:
        """Registry entry indexes registered under one module key."""
        return [i for i, name in enumerate(self.entries) if name == module_key]

    def probe_region(self, entry_index: int, region: int) -> dict | None:
        """One-shot read (used for dynamic primary resolution)."""
        return reduce_region_rows(self.read_region(entry_index, region))


def collect_marker_main_kernel(
    torch: Any,
    run: Callable[[], object],
    channel: MarkerChannel,
    primary: tuple[str, int, str],
    *,
    raw_launches: int = RAW_LAUNCHES_REQUIRED,
    resolve_dynamic: bool = False,
    candidates: list[tuple[str, int, str]] | None = None,
    unit_slot_filter: set[int] | None = None,
    unit_warps_per_cta: int | None = None,
) -> dict:
    """Full marker pass: buffers are installed at channel construction; per raw launch
    zero -> run -> sync -> read -> reduce; 27 raw primary-region samples -> 9 formal samples.

    ``unit_slot_filter`` (strict same-unit marker verification): buffer slot
    indexes of the research unit (see :func:`unit_slot_set`).  When given,
    every launch reduces ONLY the unit's own rows -- the same CTA/warp
    members, same code, same entry/exit interval the capture and converter
    will use -- and the reduction is that unit's marker denominator.  Rows
    outside the unit stay facts in the diagnostics (total vs unit-scoped
    counts per launch) and never enter the span.

    The CUDA activity cross-check splits by mode.  Without a unit filter
    (the historical whole-launch pass, the far45/296 denominator producer)
    the FA3-style +/-10% duration-match guard GATES: a missing or drifting
    activity reading refuses the sample and ``duration_match.status`` is
    the original "pass".  With a unit filter (strict same-unit decision,
    2026-10-09) the activity is DIAGNOSTIC ONLY: its
    fullnames/durations/counts are recorded but never gate -- the kineto
    kernel duration and the unit's across-warp globaltimer span measure
    different ranges (pilot evidence far-marker-diagnostic-r1), so a
    mismatch is recorded as ``diagnostic_only``, never fabricated into a
    pass or a refusal.

    ``resolve_dynamic`` (multi-variant ops like mamba_state_update): pick, once, the candidate
    (module_key, region) that carries valid rows and keep it for the rest of the pass; the
    container records the resolved primary.
    """
    module_key, region, kernel = primary
    if resolve_dynamic and candidates:
        channel.ensure_entries()
        channel.zero_all()
        run()
        torch.cuda.synchronize()
        # T0011.g mechanism-4 fix: ANY candidate carrying valid rows wins — including
        # the static primary itself.  The previous form only broke out when a
        # candidate != primary won, so a primary that carried rows fell through the
        # whole loop into the raise (single_decode_01: decode/0 valid_rows=8 yet
        # "no candidate marker region carried rows", r2 wave 2026-10-01).
        resolved = False
        for candidate_key, candidate_region, candidate_kernel in candidates:
            for entry_index in channel.resolve_entry(candidate_key):
                sample = channel.probe_region(entry_index, candidate_region)
                if sample is not None and sample["valid_rows"] > 0:
                    module_key = candidate_key
                    region = candidate_region
                    kernel = candidate_kernel
                    resolved = True
                    break
            if resolved:
                break
        if not resolved:
            # T0011.den-r12 (d-author prescription): persist the channel state at the
            # no-rows failure - per-candidate x per-entry probe results and the loaded
            # .so paths - so "buffer pointers not set" vs "uninstrumented/offset template
            # instance" is decidable offline.
            import json as _json
            import os as _os
            _root = _os.environ.get("FLASHINFER_DELIVERY_ARTIFACTS_ROOT")
            if _root:
                try:
                    diag = {"primary": list(primary),
                            "candidates": [list(c) for c in candidates],
                            "registry_module_count": len(channel.entries),
                            "registry_module_names": list(channel.entries),
                            "probes": []}
                    for candidate_key, candidate_region, candidate_kernel in candidates:
                        for entry_index in channel.resolve_entry(candidate_key):
                            sample = channel.probe_region(entry_index, candidate_region)
                            diag["probes"].append({
                                "candidate_module": candidate_key,
                                "candidate_region": candidate_region,
                                "candidate_kernel": candidate_kernel,
                                "entry_index": entry_index,
                                "sample": None if sample is None else {
                                    "valid_rows": sample.get("valid_rows"),
                                    "timer_span_ns": sample.get("timer_span_ns"),
                                },
                            })
                    for mk in {c[0] for c in candidates}:
                        diag.setdefault("binaries", {})[mk] = [
                            str(x) for x in candidate_jit_binaries(mk)]
                    _dir = Path(_root) / "logs" / "dynamic_primary"
                    _dir.mkdir(parents=True, exist_ok=True)
                    (_dir / f"{kernel or primary[2]}.json").write_text(
                        _json.dumps(diag, indent=2, default=str) + "\n", encoding="utf-8")
                except Exception:
                    pass
            raise RuntimeError(
                "no candidate marker region carried rows for a dynamic-primary op; the "
                "delivery binary is not instrumented or the op selected an uninstrumented "
                "kernel variant"
            )

    entry_indexes = channel.resolve_entry(module_key)
    if not entry_indexes:
        raise RuntimeError(
            f"primary module {module_key!r} was not registered by any loaded instrumented "
            f"module; registered: {channel.entries}"
        )

    def _unit_rows(slots: list[list[int]]) -> list[list[int]]:
        if unit_slot_filter is None:
            return [row[1:] for row in slots]
        return [row[1:] for row in slots if row[0] in unit_slot_filter]

    def _profiled_marker_read() -> dict | None:
        """The profiled physical launch's own marker rows (slots + span)."""
        for entry_index in entry_indexes:
            slots = channel.read_region_slots(entry_index, region)
            rows = _unit_rows(slots)
            if rows:
                sample = reduce_region_rows(rows)
                if sample is not None:
                    return {
                        "registry_entry": entry_index,
                        "valid_rows": sample["valid_rows"],
                        "total_valid_rows": len(slots),
                        "timer_span_ns": sample["timer_span_ns"],
                        "cycles": sample["cycles"],
                        "slots": slots,
                    }
        return None

    alt_kernel_names = [candidate_kernel for _key, _region, candidate_kernel in
                        (candidates or []) if candidate_kernel != kernel]
    unit_scoped = unit_slot_filter is not None
    activity_median_ns = None
    activity_kernel_name = None
    activity_events: list[dict] = []
    profiled_marker = None
    activity_diagnostic_error = None
    if unit_scoped:
        # Strict same-unit mode: the whole-kernel activity reading is a
        # diagnostic only and can be genuinely unavailable in a traced
        # producer run (CUPTI and the NVBit tracer coexist poorly -- the
        # profiler window may see no kernel events at all).  Its absence
        # is recorded and never blocks the unit-scoped marker pass.
        try:
            activity_median_ns, activity_kernel_name, activity_events, profiled_marker = (
                measure_primary_kernel_activity_ns(
                    torch, run, kernel, alt_names=alt_kernel_names,
                    profiled_marker_read=_profiled_marker_read)
            )
        except RuntimeError as exc:
            activity_diagnostic_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - diagnostics must not crash the pass
            activity_diagnostic_error = f"{type(exc).__name__}: {exc}"
    else:
        # Historical whole-launch mode (the far45/296 denominator producer):
        # the FA3-style +/-10% CUDA-activity cross-check GATES the pass, and
        # a missing/unmatchable activity reading refuses the sample exactly
        # as it always did (2026-10-10 adjudication: the strict same-unit
        # decision relaxed nothing on this branch).
        activity_median_ns, activity_kernel_name, activity_events, profiled_marker = (
            measure_primary_kernel_activity_ns(
                torch, run, kernel, alt_names=alt_kernel_names,
                profiled_marker_read=_profiled_marker_read)
        )
    raw: list[dict] = []
    diag_launches: list[dict] = []
    unit_slot_ids: list[list[int]] = []
    for index in range(raw_launches):
        channel.ensure_entries()
        channel.zero_all()
        run()
        torch.cuda.synchronize()
        launch = None
        slots: list[list[int]] = []
        for entry_index in entry_indexes:
            slots = channel.read_region_slots(entry_index, region)
            rows = _unit_rows(slots)
            sample = reduce_region_rows(rows) if rows else None
            if sample is not None and sample["valid_rows"] > 0:
                launch = dict(sample)
                launch["registry_entry"] = entry_index
                break
        if launch is None:
            scope = "unit" if unit_slot_filter is not None else "region"
            raise RuntimeError(
                f"marker launch {index}: no valid {kernel} marker rows in {module_key}"
                f"[{region}] for the {scope} slot set; the JIT module is not instrumented, "
                "the pointer was not set, the unit does not intersect the launched grid, or "
                "some warps skipped the outermost pair (dynamic audit failure, not a sample)"
            )
        if unit_slot_filter is not None and unit_warps_per_cta is None and launch["valid_rows"] != len(unit_slot_filter):
            # No CTA stride was supplied, so slot identity cannot name warps.
            # Keep the historical full-filter check: a short write must not
            # shrink the denominator.
            raise RuntimeError(
                f"marker launch {index}: unit coverage incomplete -- "
                f"{launch['valid_rows']} of {len(unit_slot_filter)} unit slots "
                f"carried a complete pair in {module_key}[{region}]; the "
                "same-unit denominator requires all filtered slots when the "
                "CTA warp stride is not supplied"
            )
        if unit_slot_filter is not None:
            unit_slot_ids.append([row[0] for row in slots if row[0] in unit_slot_filter])
        launch["launch_index"] = index
        raw.append(launch)
        # Coverage facts the reduced sample discards: total valid slots of the
        # launch vs the unit-scoped count actually reduced.
        diag_launches.append({
            "launch_index": index,
            "valid_rows": launch["valid_rows"],
            "total_valid_rows": len(slots),
            "timer_span_ns": launch["timer_span_ns"],
            "cycles": launch["cycles"],
            "sm_ids": launch["sm_ids"],
        })

    median_span_ns = statistics.median(row["timer_span_ns"] for row in raw)
    relative = (
        abs(median_span_ns - activity_median_ns) / activity_median_ns
        if activity_median_ns
        else None
    )
    if not unit_scoped:
        # Historical whole-launch guard (unchanged contract): the median
        # marker globaltimer span must agree with the primary kernel's CUDA
        # activity within +/-10%, or the sample refuses.  A missing reading
        # cannot reach here (measure raises first); the defensive branch
        # keeps that fail-closed even if a future refactor changes it.
        if relative is None:
            raise RuntimeError(
                f"marker duration-match guard failed: no usable {kernel} CUDA "
                "activity reading for the whole-launch pass"
            )
        if relative > DURATION_MATCH_RELATIVE:
            raise RuntimeError(
                "marker duration-match guard failed: median marker globaltimer span "
                f"{median_span_ns:.1f} ns vs {kernel} CUDA activity "
                f"{activity_median_ns:.1f} ns exceeds "
                f"+/-{int(DURATION_MATCH_RELATIVE * 100)}%"
            )
    if unit_slot_filter is not None:
        # den-r12 pattern: the unit-scoped observation chain lands beside the
        # run (slot coverage, 27 unit spans, activity kept diagnostic-only).
        import json as _json
        import os as _os
        _root = _os.environ.get("FLASHINFER_DELIVERY_ARTIFACTS_ROOT")
        if _root:
            try:
                _dir = Path(_root) / "logs" / "duration_guard"
                _dir.mkdir(parents=True, exist_ok=True)
                (_dir / f"duration_guard_{kernel}.json").write_text(
                    _json.dumps({
                        "scope": MARKER_SCOPE,
                        "unit_scoped": True,
                        "unit_slot_count": len(unit_slot_filter),
                        "relative_difference_diagnostic": relative,
                        "median_span_ns": median_span_ns,
                        "primary": {
                            "module_key": module_key,
                            "region": region,
                            "kernel": kernel,
                        },
                        "registry_entries": channel.entries,
                        "activity_diagnostic_only": {
                            "resolved_name": kernel,
                            "matched_name": activity_kernel_name,
                            "median_ns": activity_median_ns,
                            "sample_count": len(activity_events),
                            "events": activity_events,
                            "error": activity_diagnostic_error,
                        },
                        "profiled_marker": profiled_marker,
                        "launches": diag_launches,
                    }, indent=2, default=str) + "\n",
                    encoding="utf-8")
            except OSError:
                pass

    clock_marker_warps = None
    if unit_slot_filter is not None and unit_warps_per_cta is not None:
        clock_marker_warps = observed_clock_marker_warps(
            unit_slot_ids, unit_slot_filter, unit_warps_per_cta)
    summary = summarize_marker_samples(raw)
    container = build_marker_main_kernel_timing(summary, raw[-1], (module_key, region, kernel))
    return {
        "main_kernel": container,
        "registry_entries": channel.entries,
        "raw_launch_count": len(raw),
        "unit_reduction": {
            "unit_scoped": unit_scoped,
            "unit_slot_count": len(unit_slot_filter) if unit_slot_filter else None,
            "marker_median_span_ns": median_span_ns,
            "launches_reduced": len(raw),
        },
        "duration_match": {
            # Whole-kernel CUDA activity.  Strict same-unit mode
            # (unit_slot_filter given): diagnostic only -- the kineto
            # duration and the unit's across-warp globaltimer span measure
            # different ranges, so a mismatch is recorded, never fabricated
            # into a pass or a refusal (decision 2026-10-09).  Historical
            # whole-launch mode: the +/-10% cross-check above GATED this
            # pass and the status is the original "pass" with
            # allowed_relative (2026-10-10 adjudication: nothing relaxed).
            "status": "diagnostic_only" if unit_scoped else "pass",
            "marker_median_span_ns": median_span_ns,
            "primary_kernel_activity_median_ns": activity_median_ns,
            "activity_kernel_name": activity_kernel_name,
            # Full kineto event name of the matched activity (demangled with
            # template args when CUPTI provides it): the measured-entry
            # identity of THIS bare run, recorded verbatim from the matched
            # event -- never a truncated stem.
            "activity_event_name_full": (
                str(activity_events[0].get("name"))
                if activity_events else None
            ),
            "resolved_kernel_name": kernel,
            "relative_difference": relative,
            **({} if unit_scoped else {"allowed_relative": DURATION_MATCH_RELATIVE}),
        },
        "raw_launches": raw,
        "clock_marker_warps": clock_marker_warps,
    }
