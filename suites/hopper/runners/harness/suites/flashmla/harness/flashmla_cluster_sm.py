#!/usr/bin/env python3
"""Run one pinned FlashMLA SM90 delivery case through its public API.

The timed operation is one public dense decode, sparse decode, or sparse prefill
call for a manifest case.  Dense/sparse decode may launch metadata, main, and
combine kernels inside one operation; sparse prefill launches its phase-1 kernel:

    get_mla_metadata_kernel     (first decode call only: builds tile-scheduler metadata)
    FlashMLA SM90 decode kernel (dense or sparse)
    flash_fwd_mla_combine_kernel (cross-split reduction, when enabled)

Whole-operation device timing uses the shared delivery globaltimer protocol
(suites/common/globaltimer_timer.py, shared across suites).  When marker timing
is requested (env FLASHMLA_MARKER_TIMING=1 or per-case `marker_timing`), the formal
denominator comes from the in-kernel outermost clock-marker channel
(`marker_timing.py`, FLASHMLA_DELIVERY_MARKER_TIMING build); the whole-operation
globaltimer span is kept as additional information only and must not be used for
simulator accuracy (docs/simulator.md §2.5).
Capture mode (`--mode capture`) honors the same switch by registering that marker
channel around its single captured call (registration evidence only; the formal
denominator stays the bare-run marker protocol).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
COMMON_ROOT = HERE.parents[1] / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
from globaltimer_timer import measure_operation  # noqa: E402
from marker_timing import (  # noqa: E402
    MAIN_REGION,
    MarkerChannel,
    REGIONS,
    collect_marker_main_kernel,
    locate_extension,
    marker_timing_requested,
)

RESULT_SCHEMA = "flashmla-hopper-result-v1"
INVALID_BLOCK = 2147480000  # upstream test convention for out-of-range page entries
BLOCK_TABLE_GRANULARITY = 256  # upstream pads per-seq page tables to 256-token blocks

OUT_ABS_TOL = 8e-4
OUT_REL_TOL = 2.01 / 128
LSE_ABS_TOL = 1e-6
LSE_REL_TOL = 8.01 / 65536


def load_case(path: Path, case_id: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    found = [case for case in data["cases"] if case["case_id"] == case_id]
    if len(found) != 1:
        raise SystemExit(f"case {case_id} not found (exactly once) in {path}")
    return found[0]


def dtype(torch, name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def build_inputs(torch, case: dict, device: str = "cuda") -> dict:
    """Deterministic paged-MLA decode inputs, mirroring upstream
    tests/test_flash_mla_dense_decoding.py::generate_test_data but seeded from
    the manifest `input_seed` via explicit generators (device-independent)."""
    batch = int(case["batch"])
    s_q = int(case["seqlen_q"])
    s_k = int(case["seqlen_kv"])
    varlen = bool(case["is_varlen"])
    page_size = int(case["page_size"])
    h_q = int(case["heads_q"])
    h_kv = int(case["heads_kv"])
    head_dim_k = int(case["head_dim_k"])
    value_dtype = dtype(torch, case["dtype"])

    if h_q % h_kv:
        raise SystemExit(f"heads_q {h_q} not divisible by heads_kv {h_kv}")
    if page_size != 64:
        raise SystemExit(f"FlashMLA sm90 dense decode requires page_size 64, got {page_size}")
    if head_dim_k not in (576, 512):
        raise SystemExit(f"head_dim_k must be 576 or 512, got {head_dim_k}")

    generator = torch.Generator().manual_seed(int(case["input_seed"]))

    if varlen:
        # Deterministic spread around s_k (upstream draws normalvariate(s_k, s_k/2);
        # the delivery set pins a reproducible +/-40% sawtooth instead).
        step = max(1, s_k // 5)
        seqlens = [max(s_q, s_k + (index % 5 - 2) * step) for index in range(batch)]
    else:
        seqlens = [s_k] * batch
    cache_seqlens_cpu = torch.tensor(seqlens, dtype=torch.int32)
    max_seqlen = int(cache_seqlens_cpu.max().item())
    max_seqlen_pad = cdiv(max_seqlen, BLOCK_TABLE_GRANULARITY) * BLOCK_TABLE_GRANULARITY

    q_cpu = (torch.randn(batch, s_q, h_q, head_dim_k, generator=generator) / 10).clamp_(-1.0, 1.0)
    pages_per_seq = max_seqlen_pad // page_size
    total_pages = batch * pages_per_seq
    permutation = torch.randperm(total_pages, generator=generator)
    block_table_cpu = (
        torch.arange(total_pages, dtype=torch.int32).view(-1)[permutation].view(batch, pages_per_seq)
    )
    blocked_k_cpu = (
        torch.randn(total_pages, page_size, h_kv, head_dim_k, generator=generator) / 10
    ).clamp_(-1.0, 1.0)

    # Poison every page beyond each sequence's real length (upstream convention):
    # NaN payload + sentinel page ids, so stale reads would corrupt the output.
    for index in range(batch):
        length = int(cache_seqlens_cpu[index].item())
        used_blocks = cdiv(length, page_size)
        blocked_k_cpu[block_table_cpu[index][used_blocks:]] = float("nan")
        if length % page_size:
            blocked_k_cpu[block_table_cpu[index][used_blocks - 1]][length % page_size :] = float("nan")
        block_table_cpu[index][used_blocks:] = INVALID_BLOCK

    effective_causal = bool(case["causal"]) and s_q > 1
    # The C++ API forces is_causal=false for seqlen_q==1; encode the effective
    # config here so sched-meta reuse and the torch reference stay consistent.
    effective_softmax_scale = case.get("softmax_scale")
    if effective_softmax_scale is None:
        effective_softmax_scale = float(head_dim_k) ** -0.5
    return {
        "effective_causal": effective_causal,
        "effective_softmax_scale": effective_softmax_scale,
        "cache_seqlens": cache_seqlens_cpu.to(device=device, dtype=torch.int32),
        "q": q_cpu.to(device=device, dtype=value_dtype),
        "block_table": block_table_cpu.to(device=device, dtype=torch.int32),
        "blocked_k": blocked_k_cpu.to(device=device, dtype=value_dtype),
        "cache_seqlens_cpu": cache_seqlens_cpu,
    }


def _unique_indices(torch, generator, rows: int, seqlen: int, topk: int):
    if topk > seqlen:
        raise SystemExit(f"topk {topk} exceeds seqlen_kv {seqlen}")
    return torch.stack([
        torch.randperm(seqlen, generator=generator)[:topk]
        for _ in range(rows)
    ])


def build_sparse_decode_inputs(torch, case: dict, device: str = "cuda") -> dict:
    batch = int(case["batch"])
    s_q = int(case["seqlen_q"])
    h_q = int(case["heads_q"])
    s_kv = int(case["seqlen_kv"])
    topk = int(case["topk"])
    page_size = int(case["page_block_size"])
    kv_format = case["kv_format"]
    d_qk = {"v32": 576, "v4": 512}.get(kv_format)
    if d_qk is None:
        raise SystemExit(f"unsupported sparse decode kv_format {kv_format!r}")
    if page_size != 64:
        raise SystemExit(f"sparse decode page_block_size must be 64, got {page_size}")

    generator = torch.Generator().manual_seed(int(case.get("input_seed", 0)))
    q_cpu = (torch.randn(batch, s_q, h_q, d_qk, generator=generator) / 10).clamp_(-1, 1)
    indices_cpu = _unique_indices(
        torch, generator, batch * s_q, s_kv, topk
    ).view(batch, s_q, topk).to(torch.int32)
    num_blocks = cdiv(s_kv, page_size)
    bytes_per_token = 656 if kv_format == "v32" else 584
    random_k = (
        torch.randn(num_blocks, page_size, 1, d_qk, generator=generator) / 10
    ).clamp_(-1, 1).squeeze(2)

    if kv_format == "v32":
        kv_base_cpu = torch.empty(
            (num_blocks, page_size + 1, bytes_per_token),
            dtype=torch.float8_e4m3fn,
        )
        packed = kv_base_cpu[:, :page_size, :]
        packed[..., :512] = random_k[..., :512].to(torch.float8_e4m3fn)
        packed[..., 512:528].view(torch.float32).fill_(1.0)
        packed[..., 528:].view(torch.bfloat16).copy_(
            random_k[..., 512:].to(torch.bfloat16)
        )
        kv_base = kv_base_cpu.to(device=device)
        kv = kv_base[:, :page_size, :].view(
            num_blocks, page_size, 1, bytes_per_token
        )
    else:
        block_bytes = page_size * bytes_per_token
        padded_block_bytes = cdiv(block_bytes, 576) * 576
        raw_cpu = torch.empty(
            (num_blocks, padded_block_bytes), dtype=torch.float8_e4m3fn
        )
        packed = raw_cpu[:, :block_bytes]
        raw_bytes = packed.view(torch.uint8)
        data_bytes = raw_bytes[:, :page_size * 576].view(
            num_blocks, page_size, 576
        )
        data_bytes[..., :448].view(torch.float8_e4m3fn).copy_(
            random_k[..., :448].to(torch.float8_e4m3fn)
        )
        data_bytes[..., 448:].view(torch.bfloat16).copy_(
            random_k[..., 448:].to(torch.bfloat16)
        )
        scale_bytes = raw_bytes[:, page_size * 576:].view(
            num_blocks, page_size, 8
        )
        scale_bytes[..., :7].fill_(127)
        scale_bytes[..., 7].zero_()
        raw_device = raw_cpu.to(device=device)
        kv = raw_device[:, :block_bytes].view(
            num_blocks, page_size, 1, bytes_per_token
        )

    topk_length = None
    if bool(case.get("use_topk_length")):
        topk_length = torch.full((batch,), topk, dtype=torch.int32)
    attn_sink = None
    if bool(case.get("use_attn_sink")):
        attn_sink = torch.randn(h_q, generator=generator, dtype=torch.float32)
    return {
        "effective_causal": False,
        "effective_softmax_scale": float(d_qk) ** -0.5,
        "q": q_cpu.to(device=device, dtype=torch.bfloat16),
        "kv": kv,
        "indices": indices_cpu.to(device=device),
        "topk_length": None if topk_length is None else topk_length.to(device=device),
        "attn_sink": None if attn_sink is None else attn_sink.to(device=device),
        "d_qk": d_qk,
        "bytes_per_token": bytes_per_token,
    }


def build_sparse_prefill_inputs(torch, case: dict, device: str = "cuda") -> dict:
    s_q = int(case["s_q"])
    h_q = int(case["heads_q"])
    s_kv = int(case["seqlen_kv"])
    topk = int(case["topk"])
    d_qk = int(case["d_qk"])
    if d_qk not in (512, 576):
        raise SystemExit(f"sparse prefill d_qk must be 512 or 576, got {d_qk}")

    generator = torch.Generator().manual_seed(int(case.get("input_seed", 0)))
    q_cpu = (torch.randn(s_q, h_q, d_qk, generator=generator) / 10).clamp_(-1, 1)
    kv_cpu = (torch.randn(s_kv, 1, d_qk, generator=generator) / 10).clamp_(-1, 1)
    indices_cpu = _unique_indices(torch, generator, s_q, s_kv, topk)
    topk_length = None
    if bool(case.get("use_topk_length")):
        topk_length = torch.full((s_q,), topk, dtype=torch.int32)
    attn_sink = None
    if bool(case.get("use_attn_sink")):
        attn_sink = torch.randn(h_q, generator=generator, dtype=torch.float32)
    return {
        "effective_causal": False,
        "effective_softmax_scale": float(d_qk) ** -0.5,
        "q": q_cpu.to(device=device, dtype=torch.bfloat16),
        "kv": kv_cpu.to(device=device, dtype=torch.bfloat16),
        "indices": indices_cpu.to(device=device, dtype=torch.int32).unsqueeze(1),
        "topk_length": None if topk_length is None else topk_length.to(device=device),
        "attn_sink": None if attn_sink is None else attn_sink.to(device=device),
        "d_qk": d_qk,
    }


def build_operation(torch, flash_mla, case: dict):
    operation = case.get("operation", "dense_decode")
    if operation == "sparse_decode":
        inputs = build_sparse_decode_inputs(torch, case)
        sched_meta, num_splits = flash_mla.get_mla_metadata()

        def run():
            return flash_mla.flash_mla_with_kvcache(
                inputs["q"],
                inputs["kv"],
                None,
                None,
                512,
                sched_meta,
                num_splits,
                softmax_scale=inputs["effective_softmax_scale"],
                causal=False,
                is_fp8_kvcache=True,
                indices=inputs["indices"],
                attn_sink=inputs["attn_sink"],
                topk_length=inputs["topk_length"],
            )

        return run, None, inputs
    if operation == "sparse_prefill":
        inputs = build_sparse_prefill_inputs(torch, case)

        def run():
            return flash_mla.flash_mla_sparse_fwd(
                inputs["q"],
                inputs["kv"],
                inputs["indices"],
                inputs["effective_softmax_scale"],
                d_v=512,
                attn_sink=inputs["attn_sink"],
                topk_length=inputs["topk_length"],
            )

        return run, None, inputs
    if operation not in {"dense_decode", "mla_decode_dense_paged"}:
        raise SystemExit(f"unsupported FlashMLA operation {operation!r}")

    inputs = build_inputs(torch, case)
    causal = inputs["effective_causal"]
    sched_meta, num_splits = flash_mla.get_mla_metadata()
    softmax_scale = inputs["effective_softmax_scale"]

    def run():
        return flash_mla.flash_mla_with_kvcache(
            inputs["q"],
            inputs["blocked_k"],
            inputs["block_table"],
            inputs["cache_seqlens"],
            int(case["head_dim_v"]),
            sched_meta,
            num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    return run, verify_closure(torch, case, inputs), inputs


def reference_outputs(torch, case: dict, inputs: dict):
    """Float32 torch reference (adapted from upstream reference_torch)."""
    cache_seqlens = inputs["cache_seqlens_cpu"]
    block_table = inputs["block_table"].cpu()
    q = inputs["q"].float().cpu()
    blocked_k = inputs["blocked_k"].float().cpu()
    page_size = int(case["page_size"])
    head_dim_v = int(case["head_dim_v"])
    causal = inputs["effective_causal"]
    scale = inputs["effective_softmax_scale"]
    batch, s_q, h_q, _ = q.shape
    h_kv = blocked_k.shape[2]

    out = torch.empty(batch, s_q, h_q, head_dim_v, dtype=torch.float32)
    lse = torch.empty(batch, h_q, s_q, dtype=torch.float32)
    for index in range(batch):
        length = int(cache_seqlens[index].item())
        pages = block_table[index][: cdiv(length, page_size)]
        kv = blocked_k[pages].view(-1, h_kv, blocked_k.shape[-1])[:length]
        keys = kv.transpose(0, 1).float()
        queries = q[index].transpose(0, 1)  # (h_q, s_q, d)
        if h_kv != 1:
            keys = keys.repeat_interleave(h_q // h_kv, dim=0)
        logits = queries @ keys.transpose(-2, -1) * scale  # (h_q, s_q, s_k)
        if causal and s_q > 1:
            mask = torch.ones(s_q, length, dtype=torch.bool).tril(diagonal=length - s_q)
            logits = logits.masked_fill(~mask, float("-inf"))
        row_lse = logits.logsumexp(dim=-1)
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        out[index] = (weights @ keys[..., :head_dim_v]).transpose(0, 1)
        lse[index] = row_lse
    return out, lse


def verify_closure(torch, case: dict, inputs: dict):
    if not case.get("correctness_check"):
        return None

    def verify(value) -> dict:
        expected_out, expected_lse = reference_outputs(torch, case, inputs)
        actual_out, actual_lse = value
        actual_out = actual_out.float().cpu()
        actual_lse = actual_lse.float().cpu()
        out_violations = int(
            ((actual_out - expected_out).abs()
             > OUT_ABS_TOL + OUT_REL_TOL * expected_out.abs()).sum().item()
        )
        lse_violations = int(
            ((actual_lse - expected_lse).abs()
             > LSE_ABS_TOL + LSE_REL_TOL * expected_lse.abs()).sum().item()
        )
        return {
            "reference": "torch fp32 per-batch paged attention (upstream reference_torch adaptation)",
            "out_abs_tol": OUT_ABS_TOL,
            "out_rel_tol": OUT_REL_TOL,
            "lse_abs_tol": LSE_ABS_TOL,
            "lse_rel_tol": LSE_REL_TOL,
            "out_max_abs_error": (actual_out - expected_out).abs().max().item(),
            "lse_max_abs_error": (actual_lse - expected_lse).abs().max().item(),
            "out_violation_count": out_violations,
            "lse_violation_count": lse_violations,
            "status": "pass" if out_violations == 0 and lse_violations == 0 else "fail",
        }

    return verify


def tensors(value):
    if hasattr(value, "is_cuda"):
        return [value]
    if isinstance(value, (tuple, list)):
        flat = []
        for item in value:
            flat.extend(tensors(item))
        return flat
    return []


def check_output(torch, value, verify: Callable[[object], dict] | None = None) -> dict:
    outputs = tensors(value)
    if any(bool(torch.isnan(t.float()).any().item()) or bool(torch.isinf(t.float()).any().item()) for t in outputs):
        return {"status": "fail", "error": "NaN/Inf in operation output"}
    if verify is None:
        return {"status": "pass", "reference": "not_requested"}
    return verify(value)


def flashmla_build_root() -> Path:
    override = os.environ.get("FLASHMLA_BUILD_ROOT")
    if override:
        return Path(override)
    return Path(os.environ["FLASHMLA_DELIVERY_ARTIFACTS_ROOT"]) / "build"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=HERE.parent / "manifests/flashmla_cluster_sm.json")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, help="result JSON path (timing mode)")
    parser.add_argument("--mode", choices=("timing", "capture"), default="timing",
                        help="capture = pure run-once workload for the unified capture "
                             "entry (common/capture_case.py): inputs + one public API "
                             "call, no tracer env, no post-capture chain")
    parser.add_argument("--repeats", type=int, default=9,
                        help="recorded samples; formal protocol requires 9")
    parser.add_argument("--warmups", type=int, default=3,
                        help="capture-mode warmup launches before the single captured call")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    case = load_case(args.manifest, args.case_id)
    if args.dry_run:
        print(json.dumps({"status": "pass", "case": case}, indent=2))
        return

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError(
            "FlashMLA SM90 operations require a compute capability 9.x (Hopper) GPU"
        )
    import flash_mla

    started = time.time()

    if args.mode == "capture":
        result = capture_once(torch, flash_mla, case, args)
    else:
        run, verify, inputs = build_operation(torch, flash_mla, case)
        value, timing = measure_operation(
            torch,
            run,
            args.repeats,
            flashmla_build_root() / "libdelivery_globaltimer_timer.so",
        )
        operation = case.get("operation", "dense_decode")
        operation_notes = {
            "dense_decode": (
                "get_mla_metadata_kernel precedes the first recorded sample; dense decode "
                "runs flash_fwd_splitkv_mla_kernel and may run flash_fwd_mla_combine_kernel"
            ),
            "sparse_decode": (
                "get_mla_metadata_kernel precedes the first recorded sample; sparse decode "
                "runs its SM90 split-KV kernel and flash_fwd_mla_combine_kernel"
            ),
            "sparse_prefill": (
                "sparse prefill runs the SM90 sparse phase-1 kernel"
            ),
        }
        timing["operation_note"] = {
            "operation": operation,
            "kernel_sequence": operation_notes[operation],
            "effective_causal": inputs["effective_causal"],
            "effective_softmax_scale": inputs["effective_softmax_scale"],
        }
        if marker_timing_requested(case):
            # Formal marker denominator (296 scope): in-kernel outermost clock-marker
            # interval per (cta, warp); 27 raw launches -> 9 median-of-triple samples.
            # The whole-operation globaltimer fields above stay as additional information.
            import flash_mla.cuda as flash_mla_cuda

            marker = collect_marker_main_kernel(torch, run, flash_mla_cuda)
            timing["main_kernel"] = marker["main_kernel"]
            timing["marker_timing"] = {
                "enabled": True,
                "tu_count": marker["tu_count"],
                "raw_launch_count": marker["raw_launch_count"],
                "duration_match": marker["duration_match"],
                "main_kernel_activity_ns": marker["main_kernel_activity_ns"],
                "regions_secondary": marker["regions_secondary"],
            }
            timing["median_cycles_source"] = (
                "main_kernel outermost_in_kernel_clock_marker_interval; the median_cycles "
                "field above is whole-operation globaltimer compatibility only and must not "
                "be used for simulator accuracy"
            )
        correctness = check_output(torch, value, verify)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": correctness["status"],
            "case_id": case["case_id"],
            "case": case,
            "flash_mla_version": flash_mla.__version__,
            "source_revision": case["source_revision"],
            "hardware": {
                "name": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            },
            "correctness": correctness,
            "timing": timing,
            "elapsed_seconds": time.time() - started,
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "case_id": result["case_id"],
        "mode": args.mode,
        "status": result["status"],
    }
    if args.mode == "timing":
        summary.update(
            median_globaltimer=result["timing"]["median_globaltimer"],
            median_cycles_compatibility=result["timing"]["median_cycles"],
            cv_percent=result["timing"]["cv_percent"],
        )
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


def _capture_marker_facts(rows_by_region: dict, tu_count: int) -> dict:
    """Marker evidence for the single captured call; fail-closed on no main fact.

    ``rows_by_region`` comes from ``MarkerChannel.read_region_rows`` per region
    after the captured launch.  Only the main region is load-bearing:
    get_mla_metadata_kernel runs first-call only (inside the warmups, which the
    zero-after-warmup clears) and flash_fwd_mla_combine_kernel runs only when
    num_splits > 1, so an empty main region means the delivery binary was not
    instrumented or the pointers were never set -- a hard rejection, never a
    silent pass.  The counts are registration evidence only; the formal
    denominator stays the bare-run 27->9 marker protocol.
    """
    if not rows_by_region[MAIN_REGION]:
        raise RuntimeError(
            "marker capture: no valid " + MAIN_REGION + " rows after the captured "
            "call; the delivery binary is not instrumented or the marker pointers "
            "were not set before launch"
        )
    return {
        "enabled": True,
        "tu_count": tu_count,
        "region_row_counts": {
            region: len(rows_by_region[region]) for region in REGIONS
        },
        "role": (
            "capture_registration_evidence; the formal denominator stays the "
            "bare-run 27->9 marker protocol, never this single call"
        ),
    }


def capture_once(torch, flash_mla, case: dict, args) -> dict:
    """Pure run-once capture mode (stage-6 unified capture entry).

    This process is a plain workload: deterministic inputs, the public
    `flash_mla_with_kvcache` call, one real launch after warmup.  The tracer
    environment (NVBit injection, TMA recorder) is supplied externally by
    the unified capture entry (suites/common/capture_case.py ->
    hoppersim capture);
    this harness never injects it, never post-processes traces and never
    runs capture_all in-process.  Warmups land before the single captured
    call so JIT/first-launch overhead (including the tile-scheduler
    metadata build) stays outside the captured call itself.

    Marker mode uses the same switch and channel as the timing path (env
    FLASHMLA_MARKER_TIMING=1 or the per-case `marker_timing` key): the real
    marker buffers are registered with the instrumented extension before any
    launch, zeroed between the warmups and the captured call -- so the
    read-back can never select a warmup launch -- and read after the final
    sync, then archived through --output with the rest of the result.  No
    extra op launch, no timing collector.  The default unmarked flow is
    unchanged.
    """
    started = time.time()
    run, _verify, inputs = build_operation(torch, flash_mla, case)
    marker_channel = None
    if marker_timing_requested(case):
        import flash_mla.cuda as flash_mla_cuda

        marker_channel = MarkerChannel(torch, locate_extension(flash_mla_cuda))
    for _ in range(max(0, args.warmups)):
        run()
    torch.cuda.synchronize()
    if marker_channel is not None:
        marker_channel.zero_buffers()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    value = run()
    end.record()
    torch.cuda.synchronize()
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "case_id": case["case_id"],
        "case": case,
        "mode": "capture",
        "flash_mla_version": flash_mla.__version__,
        "source_revision": case["source_revision"],
        "hardware": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "traced_call": {
            "warmup_launches": max(0, args.warmups),
            "event_span_ms": begin.elapsed_time(end),
            "output_tensors": [list(t.shape) for t in tensors(value)],
            "effective_causal": inputs["effective_causal"],
        },
        "elapsed_seconds": time.time() - started,
    }
    if marker_channel is not None:
        result["marker_timing"] = _capture_marker_facts(
            {region: marker_channel.read_region_rows(region) for region in REGIONS},
            marker_channel.tu_count,
        )
    return result


if __name__ == "__main__":
    main()
