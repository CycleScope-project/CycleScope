#!/usr/bin/env python3
"""Run one FA3 full-GPU/cluster-SM case through full_gpu_ops + the shared timer.

Formal path: 2 caller rounds, 9 recorded samples, 4 raw launches each (36 raw).
`--probe` still calls the real operation once under the same timer; it does not
repeat the public API to inflate cycles.  Timed callable is not synchronized
here.  torch.profiler is a separate activity capture, not the cycle metric.

`--mode capture` is the pure run-once workload for the stage-6 unified
capture entry (suites/common/capture_case.py -> capture_binary.py):
build the case inputs, warm like the capture path always did, exactly one
real op call, no tracer env and no post-capture chain in this process.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


HARNESS_ROOT = Path(__file__).resolve().parent
SUITE_ROOT = HARNESS_ROOT.parent
DELIVERY_ROOT = HARNESS_ROOT.parents[1]
# Shared delivery timer (stage-6 consolidation): the timer import at the
# measurement call sites resolves to suites/common/globaltimer_timer.py
# (the former resolution path, sys.path.insert of the FA4 suite harness dir
# suites/flashattention/harness, no longer exists in this tree).
COMMON_ROOT = DELIVERY_ROOT / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
DEFAULT_BUILD_ROOT = Path(
    os.environ.get(
        "FLASHATTENTION3_FULL_GPU_BUILD_ROOT",
        str(Path(os.environ.get("CYCLESCOPE_ARTIFACT_ROOT", "/tmp"))
            / "tools" / "hopper-fa3-build"),
    )
)
DEFAULT_MANIFEST = SUITE_ROOT / "manifests" / "fa3_full_gpu_candidates.json"
SOURCE_REVISION = "98eb7998a0eba4047c7a30375522569d4b8efb20"
INPUT_SEED = 0
# Run-once capture mode: build_operation's own first launch plus this many
# warm launches before the single captured call (the pre-stage-6 capture
# chain warmed WARMUP_LAUNCHES=2 after its topology launch).
CAPTURE_WARMUP_LAUNCHES = 2


def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def normalize_gpu_uuid(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("UUID(") and text.endswith(")"):
        text = text[5:-1].strip().strip("'\"")
    text = text.strip("{}")
    if text.lower().startswith("gpu-"):
        text = text[4:]
    hex_only = text.replace("-", "").strip().lower()
    if len(hex_only) != 32 or any(char not in "0123456789abcdef" for char in hex_only):
        die(f"invalid GPU UUID {value!r}; refusing unknown fallback")
    pretty = (
        f"{hex_only[:8]}-{hex_only[8:12]}-{hex_only[12:16]}-"
        f"{hex_only[16:20]}-{hex_only[20:]}"
    )
    return f"GPU-{pretty}"


def parse_nvidia_smi_gpu_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            die(f"unparseable nvidia-smi GPU row: {line!r}")
        index_text, raw_uuid = parts[0], parts[1]
        name = ",".join(parts[2:]).strip()
        if not index_text.isdigit():
            die(f"unparseable nvidia-smi GPU index: {line!r}")
        rows.append({
            "index": int(index_text),
            "uuid": normalize_gpu_uuid(raw_uuid),
            "name": name,
        })
    return rows


def select_job_gpu_uuid(rows: list[dict], cuda_visible_devices: str = "") -> str:
    if not rows:
        die("nvidia-smi reported no GPUs")
    visible = (cuda_visible_devices or "").strip()
    if not visible:
        if len(rows) != 1:
            die(
                "this suite is 1 GPU per job; cannot map target UUID from "
                f"{len(rows)} GPUs without CUDA_VISIBLE_DEVICES"
            )
        return rows[0]["uuid"]
    parts = [part.strip() for part in visible.split(",") if part.strip()]
    if len(parts) != 1:
        die(
            "this suite is 1 GPU per job; cannot uniquely map "
            f"CUDA_VISIBLE_DEVICES={visible!r}"
        )
    token = parts[0]
    if token.isdigit():
        index = int(token)
        matches = [row for row in rows if row["index"] == index]
    else:
        want = normalize_gpu_uuid(token)
        matches = [row for row in rows if row["uuid"] == want]
    if len(matches) != 1:
        die(f"this suite is 1 GPU per job; cannot uniquely map {token!r} to one physical UUID")
    return matches[0]["uuid"]


def resolve_job_gpu_uuid() -> str:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            text=True,
        )
    except Exception as exc:
        die(f"nvidia-smi GPU inventory failed before CUDA init: {exc}")
    return select_job_gpu_uuid(
        parse_nvidia_smi_gpu_rows(text),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )


def realized_leftpad(state: dict):
    values = state.get("leftpad")
    if values is None:
        return None
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [int(item) for item in values]


def load_case(manifest: Path, case_id: str) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    matches = [dict(case) for case in data.get("cases") or [] if case.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected one case named {case_id}, found {len(matches)}")
    return matches[0]


def load_interface(build_root: Path):
    site = build_root / "site-packages"
    so_dir = site / "flash_attn_3"
    if not any(so_dir.glob("_C*.so")):
        die(f"missing flash_attn_3._C under {so_dir}; run build_full_gpu.py --link")
    config = site / "flash_attn_config.py"
    if not config.is_file():
        die(f"missing {config}")
    sys.path.insert(0, str(site))
    import flash_attn_3._C  # noqa: F401  # registers torch.ops.flash_attn_3
    import flash_attn_config

    if not getattr(flash_attn_config, "CONFIG", None):
        die("flash_attn_config.CONFIG is missing")
    import flash_attn_interface

    return flash_attn_interface


def gpu_identity(torch, uuid: str) -> dict:
    expected = normalize_gpu_uuid(uuid)
    if not torch.cuda.is_available():
        die("CUDA GPU is required")
    if int(torch.cuda.device_count()) != 1:
        die(f"this suite is 1 GPU per job; torch sees {torch.cuda.device_count()} devices")
    index = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(index)
    raw = str(getattr(props, "uuid", "") or "").strip()
    if not raw:
        die("torch device uuid is missing; refusing unknown fallback")
    observed = normalize_gpu_uuid(raw)
    if observed != expected:
        die(f"CUDA device uuid {observed} != locked uuid {expected}")
    name = torch.cuda.get_device_name(index)
    if "H100" not in name:
        die(f"CUDA device must be NVIDIA H100, got {name!r}")
    nvcc = ""
    try:
        nvcc = subprocess.check_output(["nvcc", "-V"], text=True).strip().splitlines()[-1]
    except Exception:
        nvcc = "unavailable"
    return {
        "name": name,
        "uuid": expected,
        "capability": list(torch.cuda.get_device_capability(index)),
        "index": index,
        "toolchain": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvcc": nvcc,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }


@contextlib.contextmanager
def exclusive_gpu_lock(uuid: str):
    uuid = normalize_gpu_uuid(uuid)
    lock_dir = Path(os.environ.get("FA3_GPU_LOCK_DIR", "/tmp/fa3_full_gpu_locks"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{uuid}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield str(path)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_sm_clocks(uuid: str) -> dict:
    uuid = normalize_gpu_uuid(uuid)
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                uuid,
                "--query-gpu=clocks.current.sm,clocks.max.sm,clocks.current.memory,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return {"error": str(exc)}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        die(f"clock query for {uuid} must return exactly one row, got {len(lines)}")
    parts = [part.strip() for part in lines[0].split(",")]
    try:
        return {
            "sm_mhz": float(parts[0]),
            "sm_max_mhz": float(parts[1]),
            "mem_mhz": float(parts[2]),
            "util_percent": float(parts[3]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def warmup_device(torch, uuid: str) -> dict:
    """Recorded SM-clock preload before any FA3 timing sample. Not the measured op.

    Idle H100 SM can sit at ~345 MHz. Seven short FA3 launches in measure_operation
    do not boost DVFS; this block holds a large GEMM until the clock rises or a
    hard duration cap is hit. Samples later are never filtered to cook CV.
    """

    before = read_sm_clocks(uuid)
    dim = 8192
    left = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)
    launches = 0
    target = None
    if isinstance(before.get("sm_max_mhz"), (int, float)):
        target = 0.80 * float(before["sm_max_mhz"])
    min_seconds = 3.0
    max_seconds = 8.0
    started = time.perf_counter()
    trace = [before]
    reached = False
    while True:
        torch.mm(left, right)
        launches += 1
        elapsed = time.perf_counter() - started
        poll = launches % 8 == 0 or elapsed >= min_seconds
        if not poll:
            continue
        torch.cuda.synchronize()
        clocks = read_sm_clocks(uuid)
        clocks["elapsed_seconds"] = elapsed
        clocks["gemm_launches"] = launches
        trace.append(clocks)
        current = clocks.get("sm_mhz")
        if target and isinstance(current, (int, float)) and current >= target and elapsed >= min_seconds:
            reached = True
            break
        if elapsed >= max_seconds:
            break
    torch.cuda.synchronize()
    after = read_sm_clocks(uuid)
    after["elapsed_seconds"] = time.perf_counter() - started
    after["gemm_launches"] = launches
    trace.append(after)
    del left, right
    torch.cuda.empty_cache()
    return {
        "method": "recorded_device_preload_outside_timing",
        "workload": "bf16_gemm_8192",
        "gemm_launches": launches,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "target_sm_mhz": target,
        "reached_target": reached,
        "elapsed_seconds": after["elapsed_seconds"],
        "clocks_before": before,
        "clocks_after": after,
        "clock_trace": trace,
        "measured_operation_used": False,
        "samples_not_filtered": True,
        "note": "runs before FA3 correctness launch and globaltimer; not used as a cycle sample",
    }


def capture_kernel_activities(torch, run) -> list[dict]:
    """Separate from timing; Kineto trace preserves actual grid/block dimensions."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(prefix="fa3-kernel-activity-") as folder:
        path = Path(folder) / "trace.json"
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    grouped: dict[str, dict] = {}
    for event in events:
        if event.get("cat") != "kernel" or float(event.get("dur", 0)) <= 0:
            continue
        name = event["name"]
        row = grouped.setdefault(name, {
            "name": name, "device_time_us": 0.0, "count": 0, "grids": [], "blocks": [],
        })
        row["device_time_us"] += float(event["dur"])
        row["count"] += 1
        for source, target in (("grid", "grids"), ("block", "blocks")):
            dimensions = event.get("args", {}).get(source)
            if dimensions is not None and dimensions not in row[target]:
                row[target].append(dimensions)
    if not grouped:
        raise RuntimeError("profiler recorded no CUDA kernel activity")
    return sorted(grouped.values(), key=lambda item: item["device_time_us"], reverse=True)


def probe_cycles(torch, run, library: Path) -> dict:
    from globaltimer_timer import GlobaltimerOperationTimer

    timer = GlobaltimerOperationTimer(torch, library)
    run()
    torch.cuda.synchronize()
    _, ticks, markers = timer.sample(run)
    ratio = float(markers["clock_per_globaltimer"])
    cycles = ticks * ratio
    return {
        "method": "queued device globaltimer whole-operation span",
        "observation_scope": "whole_operation",
        "probe": True,
        "sample_count": 1,
        "raw_sample_count": 1,
        "median_cycles": cycles,
        "cv_percent": 0.0,
        "samples_cycles": [cycles],
        "samples_globaltimer": [ticks],
        "samples_globaltimer_raw": [ticks],
        "samples_cycles_compatibility_raw": [cycles],
        "sample_clock_per_globaltimer": [ratio],
        "fixed_clock_per_globaltimer_for_compatibility": ratio,
        "marker_samples": [markers],
        "samples_not_filtered": True,
    }


def formal_cycles(torch, run, library: Path) -> dict:
    from globaltimer_timer import measure_operation

    _, timing = measure_operation(torch, run, 9, library, None)
    timing["observation_scope"] = "whole_operation"
    timing["samples_cycles"] = list(timing["samples_cycles_compatibility"])
    timing["sample_clock_per_globaltimer"] = [
        float(marker["clock_per_globaltimer"]) for marker in timing["marker_samples"]
    ]
    timing["samples_not_filtered"] = True
    timing["probe"] = False
    return timing


def probe_main_kernel_cycles(torch, run, case: dict, timing: dict) -> dict | None:
    """One recorded sample (4 raw launches) of the main kernel for cycle scouting."""
    ratio = float(
        timing.get("fixed_clock_per_globaltimer_for_compatibility")
        or timing.get("sample_clock_per_globaltimer", [0.0])[0]
        or 0.0
    )
    if ratio <= 0:
        return None
    try:
        from main_kernel_timing import measure_main_kernel_with_ratio
    except Exception as exc:  # pragma: no cover - harness import failure
        print(json.dumps({"main_kernel_timing_error": str(exc)[-400:]}), file=sys.stderr)
        return None
    try:
        return measure_main_kernel_with_ratio(
            torch,
            run,
            str(case.get("operation")),
            fixed_clock_per_globaltimer=ratio,
            repeats=1,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "main_kernel_timing_error": type(exc).__name__,
                    "main_kernel_timing_detail": str(exc)[-1200:],
                }
            ),
            file=sys.stderr,
        )
        return None


def main_kernel_cycles(torch, run, case: dict, timing: dict) -> dict | None:
    """Second, parallel timing channel: one main kernel, 9 recorded samples.

    Uses the same clock ratio the whole-operation channel calibrated, so the two
    channels are comparable. Rejected (returns None) when the operation launches
    the main kernel more than once per call, because a single-launch identity
    cannot be proven then.
    """
    try:
        from main_kernel_timing import measure_main_kernel_with_ratio
    except Exception as exc:  # pragma: no cover - harness import failure
        print(json.dumps({"main_kernel_timing_error": str(exc)[-400:]}), file=sys.stderr)
        return None
    ratio = float(
        timing.get("fixed_clock_per_globaltimer_for_compatibility")
        or timing.get("fixed_clock_per_globaltimer")
        or 0.0
    )
    if ratio <= 0:
        return None
    try:
        return measure_main_kernel_with_ratio(
            torch,
            run,
            str(case.get("operation")),
            fixed_clock_per_globaltimer=ratio,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "main_kernel_timing_error": type(exc).__name__,
                    "main_kernel_timing_detail": str(exc)[-1200:],
                }
            ),
            file=sys.stderr,
        )
        return None


def marker_main_kernel_timing(torch, run, case: dict, kernel_activities: list[dict], device_topo: dict | None) -> dict:
    """Marker-scope timing.main_kernel container (device_kernel clock64/globaltimer).

    Runs 27 clean launches of the op after one buffer memset, then reduces,
    selects and summarizes via marker_timing. Raises on any ambiguity or on a
    build without the device_kernel override (marker_tu_count() == 0).
    """
    from marker_timing import collect_marker_main_kernel

    from main_kernel_timing import select_main_kernel

    operation = str(case.get("operation"))
    primary = select_main_kernel(kernel_activities, operation)
    return collect_marker_main_kernel(
        torch,
        run,
        case,
        torch.ops.flash_attn_3,
        main_kernel_activity=primary,
        device_topo=device_topo,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure one FA3 full-GPU case. Timed callable is not synchronized."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--mode",
        choices=("timing", "capture"),
        default="timing",
        help="capture = pure run-once workload for the unified capture entry "
             "(common/capture_case.py): inputs + one real op call, no timer, "
             "no tracer env, no post-capture chain",
    )
    parser.add_argument("--round", type=int, default=1, choices=(1, 2))
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="one real timed operation for cycle scouting; no 36-launch inflation",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--marker-timing",
        action="store_true",
        help=(
            "add the device_kernel clock64/globaltimer marker main-kernel timing "
            "(timing.main_kernel); requires an _C.so built with the override"
        ),
    )
    return parser.parse_args()


def _output_tensors(value: object) -> list:
    if hasattr(value, "is_cuda"):
        return [value]
    if isinstance(value, (list, tuple)):
        found: list = []
        for item in value:
            found.extend(_output_tensors(item))
        return found
    return []


def run_once_capture(args: argparse.Namespace, case: dict) -> None:
    """Pure run-once capture mode (stage-6 unified capture entry).

    Build the case through full_gpu_ops.build_operation on the pinned build
    root (build_operation's own first launch materializes the op), warm it
    exactly like the pre-stage-6 capture chain did, then make exactly one
    real op call.  Requested device markers use the existing marker channel
    without a timing collector.  No host timer, SM-clock preload, exclusive
    GPU lock, tracer env or post-capture chain runs in this process:
    suites/common/capture_case.py injects the tracer from outside.
    """
    import torch
    if str(HARNESS_ROOT) not in sys.path:
        sys.path.insert(0, str(HARNESS_ROOT))
    from full_gpu_ops import build_operation

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        die("FA3 run-once capture requires a compute capability 9.x CUDA GPU")
    build_root = args.build_root.expanduser().resolve()
    interface = load_interface(build_root)
    from marker_timing import (
        allocate_marker_buffers, enable_markers, marker_timing_requested,
    )

    marker_buffers = None
    if args.marker_timing or marker_timing_requested(case):
        marker_buffers = allocate_marker_buffers(torch)
        enable_markers(torch.ops.flash_attn_3, marker_buffers)
    torch.manual_seed(int(case.get("input_seed") or 0))
    started = time.time()
    run, state = build_operation(torch, interface, case)
    for _ in range(CAPTURE_WARMUP_LAUNCHES):
        run()
        torch.cuda.synchronize()
    value = run()
    torch.cuda.synchronize()
    outputs = _output_tensors(value)
    if not outputs:
        die(f"{case['case_id']}: operation returned no tensor output")
    finite = all(bool(tensor.isfinite().all().item()) for tensor in outputs)
    nonzero = any(float(tensor.abs().max().item()) > 0 for tensor in outputs)
    if not finite or not nonzero:
        die(
            f"{case['case_id']}: operation output failed finite/nonzero check "
            f"(finite={finite}, nonzero={nonzero})"
        )
    result = {
        "schema_version": "flashattention3-run-once-v1",
        "status": "captured",
        "case_id": case["case_id"],
        "case": case,
        "mode": "run_once",
        "build_root": str(build_root),
        "source_revision": SOURCE_REVISION,
        "warmup_launches": CAPTURE_WARMUP_LAUNCHES + 1,
        "captured_launches": 1,
        "output_shapes": [list(tensor.shape) for tensor in outputs],
        "output_check": {"finite": finite, "nonzero": nonzero},
        "elapsed_seconds": time.time() - started,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(
        {
            "case_id": result["case_id"],
            "status": result["status"],
            "mode": result["mode"],
            "output_shapes": result["output_shapes"],
        },
        sort_keys=True,
    ))





def _device_topo_module():
    try:
        from flash_attn_3 import _C
        return _C
    except Exception:
        return None


def read_device_delivery_topo():
    """Read in-kernel %%smid / %%cluster_nctaid stores if this _C.so was built with them."""
    ext = _device_topo_module()
    if ext is None or not hasattr(ext, "read_topo"):
        return None
    rows = list(ext.read_topo())
    ctas = []
    fields = 4
    for cta in range(len(rows) // fields):
        base = cta * fields
        if int(rows[base + 3]) == 0:
            continue
        ctas.append({
            "cta": cta,
            "sm_id": int(rows[base + 0]),
            "cluster_nctaid_product": int(rows[base + 1]),
            "cluster_ctarank": int(rows[base + 2]),
        })
    if not ctas:
        return None
    sm_ids = sorted({row["sm_id"] for row in ctas})
    ncta = sorted({row["cluster_nctaid_product"] for row in ctas})
    return {
        "sm_ids": sm_ids,
        "observed_sms": sm_ids,
        "cluster_nctaid": ncta[0] if len(ncta) == 1 else ncta,
        "cluster_dim": [int(ncta[0]), 1, 1] if len(ncta) == 1 else None,
        "cluster_cta_ranks": sorted({row["cluster_ctarank"] for row in ctas}),
        "rows": ctas,
        "method": "device %%smid and %%cluster_nctaid stores in FA3 kernel bodies",
    }


def main() -> None:
    args = parse_args()
    case = load_case(args.manifest, args.case_id)
    if args.mode == "capture":
        if args.dry_run:
            print(json.dumps(
                {"status": "dry-run", "case_id": case["case_id"], "case": case,
                 "mode": "capture"},
                indent=2,
            ))
            return
        run_once_capture(args, case)
        return
    case["input_seed"] = INPUT_SEED
    if args.dry_run:
        print(json.dumps(
            {"status": "dry-run", "case_id": case["case_id"], "case": case, "input_seed": INPUT_SEED},
            indent=2,
        ))
        return

    build_root = args.build_root.expanduser().resolve()
    timer_so = build_root / "libdelivery_globaltimer_timer.so"
    if not timer_so.is_file():
        die(f"missing globaltimer library: {timer_so}")

    uuid = resolve_job_gpu_uuid()
    started = time.time()
    with exclusive_gpu_lock(uuid) as lock_path:
        if str(HARNESS_ROOT) not in sys.path:
            sys.path.insert(0, str(HARNESS_ROOT))
        from full_gpu_ops import build_operation, validate_outputs

        import torch

        gpu = gpu_identity(torch, uuid)
        interface = load_interface(build_root)
        gpu["clocks_before_warmup"] = read_sm_clocks(uuid)
        device_warmup = warmup_device(torch, uuid)
        gpu["clocks_after_warmup"] = device_warmup.get("clocks_after")
        torch.manual_seed(INPUT_SEED)
        run, state = build_operation(torch, interface, case)
        realized = realized_leftpad(state)
        ext = _device_topo_module()
        if ext is not None and hasattr(ext, "clear_topo"):
            ext.clear_topo()
        if args.probe:
            timing = probe_cycles(torch, run, timer_so)
            main_kernel = probe_main_kernel_cycles(torch, run, case, timing)
        else:
            timing = formal_cycles(torch, run, timer_so)
            main_kernel = main_kernel_cycles(torch, run, case, timing)
        kernel_activities = capture_kernel_activities(torch, run)
        device_topo = read_device_delivery_topo()
        marker_timing_result = None
        if not args.probe and (args.marker_timing or bool(case.get("marker_timing"))):
            marker_timing_result = marker_main_kernel_timing(
                torch, run, case, kernel_activities, device_topo
            )
            if device_topo is None:
                # hd256 fwd TUs carry no fa3_delivery_topo stores; the marker
                # rows recorded the same hardware %smid/%cluster_nctaid per CTA,
                # so keep the op's cluster sibling launch as the topo evidence.
                from marker_timing import fwd_cluster_topo_from_launches

                launches = marker_timing_result.get("launches") or []
                device_topo = fwd_cluster_topo_from_launches(launches)
                if device_topo is None and marker_timing_result.get("selected_grid_ids"):
                    # Non-cluster split launches (e.g. hdim64 bf16 split) live in
                    # TUs whose private fa3_delivery_topo copy the host reader
                    # cannot see; the marker rows still measured hardware %smid
                    # per CTA, so surface those as the observed-smid evidence.
                    selected = set(marker_timing_result["selected_grid_ids"])
                    smids = sorted({int(s) for row in launches
                                    if row.get("grid_id") in selected
                                    for s in (row.get("sm_ids") or [])})
                    if smids:
                        device_topo = {"observed_sms": smids,
                                       "sm_source": "marker %smid rows (non-cluster split fallback)"}
        correctness = validate_outputs(torch, case, state)
    status = "pass" if correctness and correctness.get("status") == "pass" else "fail"
    result = {
        "schema_version": "flashattention3-full-gpu-result-v1",
        "case_id": case["case_id"],
        "case": case,
        "status": status,
        "correctness": correctness,
        "device_warmup": device_warmup,
        "timing": timing,
        "main_kernel_timing": main_kernel,
        "kernel_activities": kernel_activities,
        "device_topo": device_topo,
        "gpu": gpu,
        "source_revision": SOURCE_REVISION,
        "round": int(args.round),
        "build_root": str(build_root),
        "gpu_lock": lock_path,
        "input_seed": INPUT_SEED,
        "realized_leftpad": realized,
        "elapsed_seconds": time.time() - started,
    }
    if marker_timing_result is not None:
        timing["main_kernel"] = marker_timing_result["main_kernel"]
        result["marker_timing"] = {
            "tu_count": marker_timing_result["tu_count"],
            "launch_count": marker_timing_result["launch_count"],
            "selected_grid_ids": marker_timing_result["selected_grid_ids"],
        }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        if device_topo:
            (args.out.parent / "ptx_topology.json").write_text(
                json.dumps(device_topo, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    summary = {
        "case_id": result["case_id"],
        "status": result["status"],
        "round": result["round"],
        "probe": bool(args.probe),
        "median_cycles": timing.get("median_cycles"),
        "main_kernel_median_cycles": (
            None if main_kernel is None else (main_kernel.get("timing") or {}).get("median_cycles")
        ),
        "cv_percent": timing.get("cv_percent"),
        "kernel_count": len(kernel_activities),
        "top_kernel": kernel_activities[0]["name"] if kernel_activities else None,
        "gpu_uuid": gpu["uuid"],
        "warmup_sm_mhz": (device_warmup.get("clocks_after") or {}).get("sm_mhz"),
        "warmup_reached_target": device_warmup.get("reached_target"),
        "input_seed": INPUT_SEED,
        "realized_leftpad": realized,
    }
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if status == "pass" else 1)


if __name__ == "__main__":
    main()
