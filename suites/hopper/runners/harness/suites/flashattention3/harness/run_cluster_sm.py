#!/usr/bin/env python3
"""FA3 cluster-SM 1CTA/2CTA runner. Default full-GPU path is untouched.

Phases:
  generate  write manifests/fa3_cluster_sm100.json from fa3_100.json
  probe     one globaltimer sample + topology
  tune      binary-search native seqlen_k into [1e5, 5e5] cycles
  formal    9 recorded samples (run_full_gpu.py formal path)
  shard     tune then formal for a disjoint case slice
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
SUITE = HARNESS.parent
ROOT = SUITE.parents[1]
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from cluster_sm_lib import (  # noqa: E402
    CYCLES_MAX,
    CYCLES_MIN,
    DERIVED_MANIFEST,
    H100_SMS,
    SOURCE_MANIFEST,
    derive_all,
    derive_traits,
    mechanism_still_ok,
    scale_seqlen_k,
    sm_margin_for,
    topology_from_activities,
)

DEFAULT_BUILD = Path(
    os.environ.get(
        "FLASHATTENTION3_FULL_GPU_BUILD_ROOT",
        "/path/to/hopper-prototype/artifacts/fa3_full_gpu_100_20260920",
    )
)
DEFAULT_ART = Path(
    os.environ.get(
        "FLASHATTENTION3_CLUSTER_SM_ROOT",
        "/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920/fa3",
    )
)
RUN_FULL = HARNESS / "run_full_gpu.py"


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_manifest(source: Path, dest: Path) -> dict:
    src = load_json(source)
    cases = derive_all(src["cases"])
    doc = {
        "schema_version": "hopper.delivery.cases.v1",
        "suite": "flashattention3",
        "name": "FlashAttention-3 Hopper 1CTA/2CTA cluster-SM derived set (from fa3_100)",
        "architecture": "sm_90a",
        "selected_count": 100,
        "source_manifest": str(source),
        "source_case_set": "fa3_100",
        "not_historical_dual_sm": True,
        "supported_gpu_profiles": ["h100"],
        "implementation": "FA3",
        "source_revision": src.get("source_revision"),
        "cluster_sm": {
            "target_cycle_range": [CYCLES_MIN, CYCLES_MAX],
            "cluster_policy": "prefer 1CTA/1SM; else 2CTA/2SM with two distinct measured smids; cluster_dim=1 allowed for ordinary 2CTA; original cluster/multicast kept",
            "sm_margin": "official FA3 sm_margin; params.num_sm = H100_SMs - target_ctas",
            "h100_sms": H100_SMS,
            "timing": "queued device globaltimer whole-operation span",
        },
        "delivery": {
            "groups": 8,
            "group_assignment": "index modulo 8",
            "runner": "scripts/run_cluster_sm.sh",
        },
        "cases": cases,
    }
    write_json(dest, doc)
    art = DEFAULT_ART / "manifests" / dest.name
    write_json(art, doc)
    print(json.dumps({
        "wrote": str(dest),
        "artifact_copy": str(art),
        "n": len(cases),
        "cluster2": sum(1 for c in cases if c["uses_cluster"]),
        "cta1": sum(1 for c in cases if c["target_ctas"] == 1),
        "warnings": sum(1 for c in cases if c.get("derive_warning")),
    }, sort_keys=True))
    return doc


def cases_for_shard(cases: list[dict], shard: int, shards: int) -> list[dict]:
    return [c for i, c in enumerate(cases) if i % shards == shard]


def cases_for_worker(cases: list[dict], worker: int, workers: int) -> list[dict]:
    if workers <= 1:
        return list(cases)
    return [c for i, c in enumerate(cases) if i % workers == worker]


def invoke_full_gpu(case: dict, out: Path, *, probe: bool, build_root: Path, round_id: int = 1, marker_timing: bool = False) -> dict:
    tmp_manifest = out.parent / f"{case['case_id']}.manifest.json"
    write_json(tmp_manifest, {"schema_version": "hopper.delivery.cases.v1", "suite": "flashattention3", "cases": [case]})
    cmd = [
        sys.executable,
        str(RUN_FULL),
        "--manifest", str(tmp_manifest),
        "--case-id", case["case_id"],
        "--out", str(out),
        "--round", str(round_id),
        "--build-root", str(build_root),
    ]
    if probe:
        cmd.append("--probe")
    if marker_timing:
        cmd.append("--marker-timing")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{build_root / 'site-packages'}:{HARNESS}:{env.get('PYTHONPATH', '')}"
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(SUITE), env=env, capture_output=True, text=True)
    log = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-12000:],
        "elapsed_s": time.time() - started,
    }
    write_json(out.with_suffix(".cmd.json"), log)
    if not out.is_file():
        raise RuntimeError(f"run_full_gpu produced no result ({proc.returncode}): {proc.stderr[-2000:]}")
    return load_json(out)


def attach_cluster_fields(case: dict, result: dict, *, evidence: str | None = None, ptx: dict | None = None) -> dict:
    topo = topology_from_activities(
        result.get("kernel_activities") or [],
        target_ctas=int(case["target_ctas"]),
        uses_cluster=bool(case["uses_cluster"]),
        evidence=evidence,
        ptx=ptx,
    )
    ok, traits = mechanism_still_ok(case)
    timing = result.get("timing") or {}
    cycles = timing.get("median_cycles")
    marker_main = timing.get("main_kernel") or {}
    marker_median = (
        marker_main.get("median_cycles")
        if marker_main.get("scope") == "outermost_in_kernel_clock_marker_interval"
        else None
    )
    marker_active = marker_main.get("scope") == "outermost_in_kernel_clock_marker_interval"
    correctness = result.get("correctness") or {}
    result["cluster_sm"] = {
        "source_case_id": case["source_case_id"],
        "target_ctas": case["target_ctas"],
        "uses_cluster": case["uses_cluster"],
        "sm_margin": case.get("sm_margin"),
        "parameters": {
            "operation": case["operation"],
            "dtype": case["dtype"],
            "batch": case["batch"],
            "seqlen_q": case.get("seqlen_q"),
            "seqlen_k": case.get("seqlen_k"),
            "seqlens_q": case.get("seqlens_q"),
            "seqlens_k": case.get("seqlens_k"),
            "heads_q": case["heads_q"],
            "heads_kv": case["heads_kv"],
            "head_dim": case["head_dim"],
            "head_dim_v": case.get("head_dim_v"),
            "causal": case.get("causal"),
            "window_size": case.get("window_size"),
            "attention_chunk": case.get("attention_chunk"),
            "softcap": case.get("softcap"),
            "num_splits": case.get("num_splits"),
            "pack_gqa": case.get("pack_gqa"),
            "page_size": case.get("page_size"),
            "sm_margin": case.get("sm_margin"),
        },
        "mechanisms_preserved": ok,
        "derived_traits": traits,
        "topology": topo,
        "cycles_median": cycles,
        "main_kernel_cycles_median": ((result.get("main_kernel_timing") or {}).get("timing") or {}).get(
            "median_cycles"
        ),
        "whole_operation_cycles_median": timing.get("median_cycles"),
        "cycle_band": [0, CYCLES_MAX] if marker_active else [CYCLES_MIN, CYCLES_MAX],
        "in_cycle_band": (
            isinstance(cycles, (int, float))
            and (0 < float(cycles) <= CYCLES_MAX if marker_active else CYCLES_MIN <= float(cycles) <= CYCLES_MAX)
        ),
        "correctness_status": correctness.get("status"),
    }
    if marker_median is not None:
        result["cluster_sm"]["marker_main_kernel_cycles_median"] = marker_median
    return result


def inject_sm_margin(case: dict, sm_count: int) -> dict:
    out = dict(case)
    # CTA count = SM count for both cluster and ordinary 2CTA.
    out["sm_margin"] = sm_margin_for(int(case["target_ctas"]), sm_count)
    return out


def query_sm_count(build_root: Path) -> int:
    code = (
        "import torch; p=torch.cuda.get_device_properties(0); "
        "sms=getattr(p,'multi_processor_count',None) or getattr(p,'multiProcessorCount',None); "
        "print(int(sms)); print(torch.cuda.get_device_name(0))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{build_root / 'site-packages'}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"SM query failed: {proc.stderr}")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    sms = int(lines[0])
    name = lines[1] if len(lines) > 1 else ""
    if "H100" not in name:
        die(f"expected H100, got {name!r}")
    return sms


def _cycles(result: dict) -> float | None:
    """Main-kernel GPU cycles only.

    Whole-operation cycles are not a substitute: the tuner must never select a
    shape using a different metric than the qualification band.
    """
    value = ((result.get("main_kernel_timing") or {}).get("timing") or {}).get("median_cycles")
    return None if value is None else float(value)


def tune_case(case: dict, work: Path, build_root: Path, sm_count: int) -> dict:
    attempts = []
    current = inject_sm_margin(case, sm_count)
    lo, hi = 64, 32768
    page = current.get("page_size")
    if page:
        lo = max(int(page), lo)
        hi = (hi // int(page)) * int(page)

    # The main kernel's own grid product is the topology bound; for backward the
    # grid is ceil(seqlen_k/kBlockN)*h*b (flash_bwd_launch_template.h:176-184),
    # so K can only grow while the main kernel stays at 1-2 CTAs. Query length
    # does not enter the grid, so it is the free knob for cycle work.
    def main_kernel_cta_limit(trial: dict) -> int | None:
        traits = derive_traits(trial)
        op = str(trial.get("operation") or "")
        if op in ("fixed_bwd", "varlen_bwd"):
            kbn = int(traits["kBlockN"])
            k = int(trial.get("seqlen_k") or 0)
            if k <= 0:
                return None
            import math as _math

            return int(_math.ceil(k / kbn)) * int(trial.get("heads_q") or 1) * int(
                trial.get("batch") or 1
            )
        return None

    def within_cta_limit(trial: dict) -> bool:
        limit = main_kernel_cta_limit(trial)
        return limit is None or limit <= 2

    def probe_at(seqlen_k: int, tag: str, *, seqlen_q: int | None = None) -> dict:
        trial = scale_seqlen_k(current, seqlen_k)
        if seqlen_q is not None:
            from cluster_sm_lib import scale_seqlen_q

            trial = scale_seqlen_q(trial, int(seqlen_q))
        ok, traits = mechanism_still_ok(trial)
        out = work / "attempts" / f"{tag}.json"
        result = None
        error = None
        if not ok:
            error = f"mechanism broken at seqlen_k={seqlen_k} traits={traits}"
        elif not within_cta_limit(trial):
            error = (
                "main-kernel CTA limit exceeded at "
                f"seqlen_k={seqlen_k}: grid product {main_kernel_cta_limit(trial)} > 2"
            )
        else:
            try:
                result = invoke_full_gpu(trial, out, probe=True, build_root=build_root)
                result = attach_cluster_fields(trial, result)
                write_json(out, result)
            except Exception as exc:
                error = str(exc)
        row = {
            "tag": tag,
            "seqlen_k": trial["seqlen_k"],
            "parameters": {
                "seqlen_k": trial["seqlen_k"],
                "seqlens_k": trial.get("seqlens_k"),
                "batch": trial["batch"],
                "seqlen_q": trial.get("seqlen_q"),
                "num_splits": trial.get("num_splits"),
            },
            "main_kernel_cta_limit": main_kernel_cta_limit(trial),
            "cycles": None if result is None else _cycles(result),
            "status": None if result is None else result.get("status"),
            "correctness": None if result is None else (result.get("correctness") or {}).get("status"),
            "error": error,
            "artifact": str(out) if out.is_file() else None,
        }
        attempts.append(row)
        write_json(work / "attempts.json", attempts)
        print(json.dumps({"probe": row}, sort_keys=True))
        return {"trial": trial, "result": result, "row": row}

    first = probe_at(int(current["seqlen_k"]), "seed")
    best = first
    if first["result"] is None:
        # The seed itself violates the main-kernel CTA limit (backward grid is
        # ceil(K/kBlockN)*h*b and Q is free). Reduce K until the main kernel is
        # 1-2 CTAs, then grow Q for cycle work.
        if main_kernel_cta_limit(current) is not None:
            legal = probe_legal_backward_shape(current, work, probe_at)
            if legal is not None:
                return legal
        return {"case": current, "best": best, "attempts": attempts}

    cycles = first["row"]["cycles"] or 0.0
    # Keep probe away from the lower edge so formal (9-sample median) stays in band.
    if CYCLES_MIN * 1.3 <= cycles <= CYCLES_MAX and first["row"]["status"] == "pass":
        return {"case": first["trial"], "best": first, "attempts": attempts}

    # Geometric walk on native K. No kernel wrapping / sleep / dummy arithmetic.
    seq = int(current["seqlen_k"])
    if cycles < CYCLES_MIN:
        for i in range(8):
            seq = min(hi, max(seq * 2, seq + (int(page) if page else 128)))
            hit = probe_at(seq, f"up{i}")
            if hit["result"] is None:
                continue
            best = hit
            cycles = hit["row"]["cycles"] or 0.0
            if cycles >= CYCLES_MIN:
                break
    elif cycles > CYCLES_MAX:
        for i in range(8):
            seq = max(lo, seq // 2)
            hit = probe_at(seq, f"down{i}")
            if hit["result"] is None:
                continue
            best = hit
            cycles = hit["row"]["cycles"] or 0.0
            if cycles <= CYCLES_MAX:
                break

    # Bounded binary search if we crossed the band.
    low_k, high_k = lo, hi
    measured = [(a["seqlen_k"], a["cycles"]) for a in attempts if a["cycles"]]
    unders = [k for k, c in measured if c < CYCLES_MIN]
    overs = [k for k, c in measured if c > CYCLES_MAX]
    insides = [k for k, c in measured if CYCLES_MIN <= c <= CYCLES_MAX]
    if insides:
        k_star = insides[-1]
        hit = next(x for x in attempts if x["seqlen_k"] == k_star and x["cycles"])
        trial = scale_seqlen_k(current, k_star)
        return {"case": trial, "best": {"trial": trial, "row": hit}, "attempts": attempts}
    if unders:
        low_k = max(unders)
    if overs:
        high_k = min(overs)
    if low_k < high_k:
        for i in range(5):
            mid = (low_k + high_k) // 2
            if page:
                mid = _align_mid(mid, int(page), low_k, high_k)
            if mid <= low_k or mid >= high_k:
                break
            hit = probe_at(mid, f"bin{i}")
            if hit["result"] is None or not hit["row"]["cycles"]:
                break
            best = hit
            cyc = hit["row"]["cycles"]
            if CYCLES_MIN <= cyc <= CYCLES_MAX:
                return {"case": hit["trial"], "best": hit, "attempts": attempts}
            if cyc < CYCLES_MIN:
                low_k = mid
            else:
                high_k = mid

    trial = best.get("trial") or current
    return {"case": trial, "best": best, "attempts": attempts}


def _align_mid(mid: int, page: int, lo: int, hi: int) -> int:
    mid = max(page, (mid // page) * page)
    return min(max(mid, lo + page), hi - page) if hi - lo > 2 * page else mid


def probe_legal_backward_shape(current: dict, work: Path, probe_at) -> dict | None:
    """Find a legal 1-2 CTA backward shape, then walk Q for cycle work.

    The backward main kernel's grid is ceil(seqlen_k/kBlockN)*heads*batch
    (flash_bwd_launch_template.h:176-184) and ClusterShape is hardcoded 1x1x1
    (line 78), so sm_margin cannot cap it. Q does not enter that grid, so it is
    the free work axis. GQA ratio (heads_q / heads_kv) is preserved: only the
    absolute head count is reduced.
    """
    import math as _math

    traits = derive_traits(current)
    kbn = int(traits["kBlockN"])
    heads = int(current.get("heads_q") or 1)
    batch = int(current.get("batch") or 1)
    seqlen_k = int(current.get("seqlen_k") or 0)
    if kbn <= 0 or seqlen_k <= 0:
        return None
    needed = _math.ceil(seqlen_k / kbn) * heads * batch
    if needed <= 2:
        k_legal = seqlen_k
    else:
        # Keep the largest K whose ceil(K/kBlockN)*heads*batch <= 2.
        k_legal = max(1, (2 // (heads * batch)) * kbn)
        if _math.ceil(k_legal / kbn) * heads * batch > 2:
            return None
    hit = probe_at(k_legal, "klegal")
    if hit["result"] is None:
        return None
    best = hit
    cycles = hit["row"]["cycles"] or 0.0
    if CYCLES_MIN * 1.3 <= cycles <= CYCLES_MAX and hit["row"]["status"] == "pass":
        return {"case": hit["trial"], "best": hit, "attempts": []}
    q = int(current.get("seqlen_q") or 128)
    for i in range(8):
        q = min(32768, max(q * 2, q + 128))
        walk = probe_at(k_legal, f"q{i}", seqlen_q=q)
        if walk["result"] is None:
            continue
        best = walk
        cycles = walk["row"]["cycles"] or 0.0
        if cycles >= CYCLES_MIN:
            break
    return {"case": best.get("trial") or current, "best": best, "attempts": []}


def _load_ptx(work: Path | None, formal: dict | None = None) -> dict | None:
    if formal and isinstance(formal.get("device_topo"), dict) and formal["device_topo"].get("sm_ids"):
        return formal["device_topo"]
    if work is None:
        return None
    for name in ("ptx_topology.json", "device_topo.json", "smid.json"):
        path = work / name
        if path.is_file():
            data = load_json(path)
            if data:
                return data
    jsonl = work / "ptx_topology.jsonl"
    if jsonl.is_file():
        # One launch row that already lists both smids counts. Do not union
        # helper 1CTA smids across rows to complete a 2CTA claim.
        best_ids = None
        ncta = None
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            vals = []
            for key in ("sm_ids", "observed_sms"):
                raw = row.get(key)
                if isinstance(raw, list) and raw:
                    vals = [int(x) for x in raw if x is not None]
                    break
            if not vals:
                continue
            if best_ids is None or len(set(vals)) > len(set(best_ids)):
                best_ids = vals
                ncta = row.get("cluster_nctaid") or row.get("cluster_dim") or row.get("cluster")
        if best_ids:
            return {"sm_ids": sorted(set(best_ids)), "cluster_nctaid": ncta, "cluster_dim": ncta}
    return None


def _persist_topo_evidence(work: Path | None, topo: dict, ptx: dict | None) -> dict:
    """Write JSONL launch rows with measured smids. 2CTA evidence must contain smids."""
    if work is None:
        return topo
    sm_ids = []
    if isinstance(topo.get("observed_sms"), list):
        sm_ids = [int(x) for x in topo["observed_sms"] if x is not None]
    elif ptx:
        sm_ids = [int(x) for x in (ptx.get("sm_ids") or ptx.get("observed_sms") or []) if x is not None]
    grid = topo.get("grid")
    cluster = topo.get("cluster_dim")
    rows = [{
        "grid": grid,
        "launched_grid": grid,
        "cluster": cluster,
        "cluster_dim": cluster,
        "sm_ids": sm_ids,
        "observed_sms": sm_ids if sm_ids else topo.get("observed_sms"),
        "cta_count": topo.get("cta_count"),
    }]
    for sm in sm_ids:
        rows.append({
            "grid": grid,
            "cluster": cluster,
            "cluster_dim": cluster,
            "sm_id": sm,
            "smid": sm,
        })
    for kernel in topo.get("kernels") or []:
        kind = str(kernel.get("kind") or "")
        if not kind.startswith("fa3_"):
            continue
        rows.append({
            "kind": kind,
            "name": kernel.get("name"),
            "grid": kernel.get("grid"),
            "grids": kernel.get("grids"),
            "cta_count": kernel.get("cta_count"),
            "cluster": cluster,
            "cluster_dim": cluster,
        })
    jsonl = work / "ptx_topology.jsonl"
    jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    if ptx:
        write_json(work / "ptx_topology.json", ptx)
    topo = dict(topo)
    if sm_ids:
        topo["evidence"] = str(jsonl)
    return topo


def _ensure_attempts(tune: dict, case: dict, formal: dict | None) -> list:
    attempts = list(tune.get("attempts") or [])
    if attempts:
        return attempts
    timing = (formal or {}).get("timing") or {}
    return [{
        "tag": "formal_or_seed",
        "seqlen_k": case.get("seqlen_k"),
        "seqlen_q": case.get("seqlen_q"),
        "num_splits": case.get("num_splits"),
        "cycles": timing.get("median_cycles"),
        "status": (formal or {}).get("status"),
        "correctness": ((formal or {}).get("correctness") or {}).get("status"),
        "parameters": {
            "seqlen_q": case.get("seqlen_q"),
            "seqlen_k": case.get("seqlen_k"),
            "batch": case.get("batch"),
            "heads_q": case.get("heads_q"),
            "heads_kv": case.get("heads_kv"),
            "num_splits": case.get("num_splits"),
            "sm_margin": case.get("sm_margin"),
        },
    }]


def qualify_row(case: dict, formal: dict | None, tune: dict, *, reason: str | None = None) -> dict:
    evidence = str(tune.get("formal_path") or "")
    work = Path(evidence).parent if evidence else None
    ptx = _load_ptx(work, formal)
    activities = (formal or {}).get("kernel_activities") or []
    topo = topology_from_activities(
        activities,
        target_ctas=int(case["target_ctas"]),
        uses_cluster=bool(case["uses_cluster"]),
        evidence=evidence or None,
        ptx=ptx,
    )
    topo = _persist_topo_evidence(work, topo, ptx)
    if formal is not None:
        formal = attach_cluster_fields(case, formal, evidence=topo.get("evidence") or evidence or None, ptx=ptx)
        attached = (formal.get("cluster_sm") or {}).get("topology") or {}
        if topo.get("evidence"):
            attached["evidence"] = topo["evidence"]
        if topo.get("observed_sms") is not None:
            attached["observed_sms"] = topo["observed_sms"]
        if topo.get("cluster_dim") is not None:
            attached["cluster_dim"] = topo["cluster_dim"]
        topo = attached or topo
    cluster = (formal or {}).get("cluster_sm") or {}
    timing = (formal or {}).get("timing") or {}
    main = (formal or {}).get("main_kernel_timing") or {}
    main_timing = main.get("timing") or {}
    main_kernel = main.get("main_kernel") or {}
    # Marker-mode main-kernel timing (device_kernel clock64/globaltimer). When
    # present it becomes the qualification source; the CUDA activity channel
    # stays in main_kernel_timing untouched. Inert without the marker switch.
    marker_main_timing = timing.get("main_kernel") or {}
    marker_active = (
        marker_main_timing.get("scope") == "outermost_in_kernel_clock_marker_interval"
    )
    activity_main_timing = main_timing
    if marker_active:
        main_timing = marker_main_timing
    correctness = (formal or {}).get("correctness") or {}
    extra = list(topo.get("extra_cta_op_kernels") or [])
    fwd_ctas = topo.get("observed_fwd_cta_count")
    grid = topo.get("grid")
    grid_ctas = int(__import__("math").prod(grid)) if grid else None
    cluster_dim = topo.get("cluster_dim")
    cluster_prod = int(__import__("math").prod(cluster_dim)) if cluster_dim else None
    cycles = main_timing.get("median_cycles")
    whole_cycles = timing.get("median_cycles")
    # Marker scope: the qualification window is (0, CYCLES_MAX]; the 100k floor
    # only dilutes CUDA-activity timestamp error and has nothing to absorb here
    # (validator cycle_in_window applies the same rule).
    if marker_active:
        in_band = isinstance(cycles, (int, float)) and 0 < float(cycles) <= CYCLES_MAX
    else:
        in_band = isinstance(cycles, (int, float)) and CYCLES_MIN <= float(cycles) <= CYCLES_MAX
    mech = bool(cluster.get("mechanisms_preserved")) if cluster else False
    if cluster.get("mechanisms_preserved") is None:
        mech, traits = mechanism_still_ok(case)
        cluster["derived_traits"] = traits
        cluster["mechanisms_preserved"] = mech
    corr = correctness.get("status") == "pass" and (formal or {}).get("status") == "pass"
    main_samples = list(main_timing.get("samples_cycles") or [])
    n_ok = bool(main_timing) and len(main_samples) >= 9
    sms = topo.get("observed_sms")
    extra_note = ", ".join(f"{e.get('kind')}={e.get('cta_count')}" for e in extra[:8])
    primary = case.get("primary_mechanism")
    main_grid = main_kernel.get("grid")
    main_grid_ctas = None
    if main_grid:
        main_grid_ctas = int(__import__("math").prod(main_grid[0] if isinstance(main_grid[0], (list, tuple)) else main_grid))
    main_note = (
        f"main_kernel={main_kernel.get('kind')} grid={main_grid} cta_count={main_grid_ctas} "
        f"median_cycles={cycles} samples={len(main_samples)}"
    )

    if reason is None:
        skip_terminal = (
            case.get("skip_terminal")
            if isinstance(case.get("skip_terminal"), dict)
            else None
        )
        skip_reason = str(skip_terminal.get("reason") or "").strip() if skip_terminal else ""
        if skip_terminal:
            # Declared terminal skip (DeepGEMM-precedent semantics): the row
            # keeps its measured topology/timing/attempts, but the case can
            # never satisfy the 1-2 CTA main-kernel launch contract for a
            # structural, mechanism-preserving reason recorded in the reason.
            status = "skipped" if skip_reason else "unresolved"
            reason = skip_reason or "skip_terminal declaration missing reason"
        else:
            reason = ""
            if not n_ok:
                status = "unresolved"
                reason = (
                    "main-kernel timing unavailable: "
                    f"{main_note}; whole_operation_cycles={whole_cycles}. "
                    "qualification requires the main kernel's own 9 recorded samples."
                )
            elif not corr or not mech or not in_band:
                status = "unresolved"
                missing = []
                if not corr:
                    missing.append(f"correctness={correctness.get('status')}")
                if not mech:
                    missing.append("mechanism")
                if not in_band:
                    missing.append(f"main_kernel_cycles={cycles}")
                reason = "; ".join(missing) + f"; {main_note}"
            elif main_grid_ctas not in (1, 2):
                status = "unresolved"
                reason = (
                    "main kernel launch grid product is not 1 or 2: "
                    f"grid={main_grid} cta_count={main_grid_ctas}; {main_note}"
                )
            elif case["target_ctas"] == 1 and not case["uses_cluster"]:
                if main_grid_ctas != 1 or cluster_prod != 1:
                    status = "unresolved"
                    reason = (
                        "1CTA requires main-kernel launch grid product 1 and cluster_dim 1, "
                        f"got grid={main_grid} cluster_dim={cluster_dim}; {main_note}"
                    )
                else:
                    status = "qualified"
                    reason = ""
            elif case["target_ctas"] == 2 and case["uses_cluster"]:
                distinct = [int(x) for x in (sms or []) if x is not None] if isinstance(sms, list) else []
                if main_grid_ctas != 2 or cluster_prod != 2 or len(set(distinct)) != 2:
                    status = "unresolved"
                    reason = (
                        "2CTA cluster requires measured launch grid product 2, cluster_nctaid "
                        f"product 2 and two distinct smids; got grid={main_grid} "
                        f"cluster_dim={cluster_dim} observed_sms={sms}; {main_note}"
                    )
                else:
                    status = "qualified"
                    reason = ""
            elif case["target_ctas"] == 2 and not case["uses_cluster"]:
                distinct = [int(x) for x in (sms or []) if x is not None] if isinstance(sms, list) else []
                if main_grid_ctas != 2 or cluster_prod != 1 or len(set(distinct)) != 2:
                    status = "unresolved"
                    reason = (
                        "non-cluster 2 ordinary CTAs require launch grid product 2, cluster_dim "
                        f"1x1x1 and two distinct measured smids; got grid={main_grid} "
                        f"cluster_dim={cluster_dim} observed_sms={sms}; {main_note}"
                    )
                else:
                    status = "qualified"
                    reason = ""
            else:
                status = "unresolved"
                reason = f"target_ctas/uses_cluster mismatch {case['target_ctas']}/{case['uses_cluster']}"
    else:
        status = "unresolved"

    attempts = _ensure_attempts(tune, case, formal)
    return {
        "source_case_id": case["source_case_id"],
        "case_id": case["case_id"],
        "status": status,
        "target_ctas": case["target_ctas"],
        "uses_cluster": case["uses_cluster"],
        "parameters": cluster.get("parameters") or {
            "operation": case.get("operation"),
            "seqlen_q": case.get("seqlen_q"),
            "seqlen_k": case.get("seqlen_k"),
            "batch": case.get("batch"),
            "heads_q": case.get("heads_q"),
            "heads_kv": case.get("heads_kv"),
            "dtype": case.get("dtype"),
            "sm_margin": case.get("sm_margin"),
            "num_splits": case.get("num_splits"),
        },
        "mechanisms_preserved": mech,
        "mechanism_evidence": case.get("mechanism_evidence"),
        "topology": {
            **topo,
            "main_kernel": {
                "kind": main_kernel.get("kind"),
                "name": main_kernel.get("name"),
                "grid": main_kernel.get("grid"),
                "cta_count": main_grid_ctas,
                "blocks": main_kernel.get("blocks"),
                "same_name_ordinal": main_kernel.get("same_name_ordinal"),
                "same_name_launch_count": main_kernel.get("same_name_launch_count"),
                "device_time_us": main_kernel.get("device_time_us"),
            } if main_kernel else None,
        },
        "correctness": {
            "status": correctness.get("status"),
            "method": "full_gpu_ops.validate_outputs bounded full-shape FP32 references",
            "artifact": evidence or None,
        },
        "timing": {
            "cycles_median": timing.get("median_cycles"),
            "samples_cycles": list(timing.get("samples_cycles") or []),
            "scope": timing.get("observation_scope") or "whole_operation",
            "method": timing.get("method") or "queued device globaltimer whole-operation span",
            "artifact": evidence or None,
            "cv_percent": timing.get("cv_percent"),
            "probe": timing.get("probe"),
            "main_kernel": {
                "scope": main_timing.get("scope"),
                "method": main_timing.get("method"),
                "hardware_metric": main_timing.get("hardware_metric"),
                "samples_cycles": list(main_timing.get("samples_cycles") or []),
                "samples_cycles_raw": list(main_timing.get("samples_cycles_raw") or []),
                "median_cycles": main_timing.get("median_cycles"),
                "cv_percent": main_timing.get("cv_percent"),
                "sample_count": main_timing.get("sample_count"),
                "raw_sample_count": main_timing.get("raw_sample_count"),
                "contract_digest": main_timing.get("contract_digest"),
                "status": main_timing.get("status"),
                "kernel_name_expected": main_timing.get("kernel_name_expected"),
                "kernel_name_observed": main_timing.get("kernel_name_observed"),
                **(
                    {
                        "activity_channel": {
                            "scope": activity_main_timing.get("scope"),
                            "method": activity_main_timing.get("method"),
                            "hardware_metric": activity_main_timing.get("hardware_metric"),
                            "samples_cycles": list(activity_main_timing.get("samples_cycles") or []),
                            "median_cycles": activity_main_timing.get("median_cycles"),
                            "cv_percent": activity_main_timing.get("cv_percent"),
                        }
                    }
                    if marker_active
                    else {}
                ),
            } if main_timing else None,
        },
        "reason": reason,
        "attempts": attempts,
        "primary_mechanism": case.get("primary_mechanism"),
        "derived_traits": cluster.get("derived_traits"),
    }



def run_one(
    case: dict,
    art_root: Path,
    build_root: Path,
    sm_count: int,
    *,
    formal: bool,
    reuse_tuned: bool = False,
    marker_timing: bool = False,
) -> dict:
    work = art_root / "cases" / case["source_case_id"]
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "derived_seed.json", case)
    if reuse_tuned and (work / "tuned_case.json").is_file():
        # Measurement lane: reuse the already-tuned parameters exactly. No
        # tuner, no rebuild, no whole-operation substitution.
        final_case = load_json(work / "tuned_case.json")
        attempts = load_json(work / "attempts.json") if (work / "attempts.json").is_file() else []
        tuned = {"case": final_case, "best": None, "attempts": attempts}
    else:
        tuned = tune_case(case, work, build_root, sm_count)
        final_case = inject_sm_margin(tuned["case"], sm_count)
        write_json(work / "tuned_case.json", final_case)
    formal_result = None
    formal_path = work / "formal.json"
    if formal:
        marker_on = bool(marker_timing or final_case.get("marker_timing"))
        formal_result = invoke_full_gpu(final_case, formal_path, probe=False, build_root=build_root, marker_timing=marker_on)
        formal_result = attach_cluster_fields(final_case, formal_result, evidence=str(formal_path), ptx=_load_ptx(work, formal_result))
        write_json(formal_path, formal_result)
    tuned["formal_path"] = str(formal_path if formal_path.is_file() else "")
    row = qualify_row(final_case, formal_result, tuned)
    write_json(work / "qualify_row.json", row)
    print(json.dumps({
        "source_case_id": row["source_case_id"],
        "status": row["status"],
        "cycles": ((row.get("timing") or {}).get("main_kernel") or {}).get("median_cycles"),
        "whole_operation_cycles": (row["timing"] or {}).get("cycles_median"),
        "main_kernel_grid": ((row.get("topology") or {}).get("main_kernel") or {}).get("grid"),
        "main_kernel_cta_count": ((row.get("topology") or {}).get("main_kernel") or {}).get("cta_count"),
        "target_ctas": row["target_ctas"],
        "fwd_ctas": (row["topology"] or {}).get("observed_fwd_cta_count"),
        "cluster": (row["topology"] or {}).get("observed_uses_cluster"),
        "reason": row["reason"],
    }, sort_keys=True))
    return row


def merge_qualification(art_root: Path, derived_manifest: Path, shards: int) -> None:
    derived = load_json(derived_manifest)
    expected = [c["source_case_id"] for c in derived["cases"]]
    rows = []
    by_id = {}
    for sid in expected:
        qj = art_root / "cases" / sid / "qualify_row.json"
        if qj.is_file():
            by_id[sid] = load_json(qj)
    for shard in range(shards):
        path = art_root / "shards" / f"shard{shard}.json"
        if not path.is_file():
            continue
        for row in load_json(path):
            sid = row.get("source_case_id")
            if sid and sid not in by_id:
                by_id[sid] = row
    missing = []
    for sid in expected:
        if sid in by_id:
            rows.append(by_id[sid])
        else:
            missing.append(sid)
            rows.append({
                "source_case_id": sid,
                "case_id": f"{sid}_cluster_sm",
                "status": "unresolved",
                "target_ctas": None,
                "uses_cluster": None,
                "parameters": {},
                "mechanisms_preserved": None,
                "mechanism_evidence": None,
                "topology": None,
                "correctness": {"status": "missing", "method": None, "artifact": None},
                "timing": {"cycles_median": None, "samples_cycles": [], "scope": "whole_operation", "method": None, "artifact": None},
                "reason": f"no shard result for {sid}",
                "attempts": [],
            })
    write_qualification(art_root, rows, {"shards": shards, "missing": missing})
    print(json.dumps({
        "qualification": str(art_root / "qualification.json"),
        "n": len(rows),
        "missing": missing,
        "qualified": sum(1 for r in rows if r["status"] == "qualified"),
        "skipped": sum(1 for r in rows if r["status"] == "skipped"),
        "unresolved": sum(1 for r in rows if r["status"] == "unresolved"),
    }, sort_keys=True))


def write_qualification(art_root: Path, rows: list[dict], extra: dict | None = None) -> None:
    by_id = {r["source_case_id"]: r for r in rows}
    payload = {
        "suite": "flashattention3",
        "source_manifest": str(SOURCE_MANIFEST),
        "derived_manifest": str(DERIVED_MANIFEST),
        "runner": "suites/flashattention3/scripts/run_cluster_sm.sh",
        "cases": rows,
        "counts": {
            "n": len(rows),
            "qualified": sum(1 for r in rows if r["status"] == "qualified"),
            "skipped": sum(1 for r in rows if r["status"] == "skipped"),
            "unresolved": sum(1 for r in rows if r["status"] == "unresolved"),
        },
        "actual_concurrency": {
            "outer_shards": extra.get("shards") if extra else None,
            "timing_workers_per_gpu": 1,
            "note": "one GPU per cluster shard; workers serialize behind exclusive GPU lock",
        },
    }
    write_json(art_root / "qualification.json", payload)
    write_json(art_root / "qualification.partial.json", payload)
    alias = art_root.parent / "flashattention3" / "qualification.json"
    if alias.resolve() != (art_root / "qualification.json").resolve():
        write_json(alias, payload)


def recompute_all(art_root: Path, derived_manifest: Path, shards: int = 8) -> None:
    """Rewrite every qualify_row from on-disk formal/tuned/attempts. CPU only."""
    derived = load_json(derived_manifest)
    by_src = {c["source_case_id"]: c for c in derived["cases"]}
    rows = []
    for src_id in [c["source_case_id"] for c in derived["cases"]]:
        work = art_root / "cases" / src_id
        case_path = work / "tuned_case.json"
        if not case_path.is_file():
            case_path = work / "derived_seed.json"
        if not case_path.is_file():
            case = dict(by_src[src_id])
        else:
            case = load_json(case_path)
        formal_path = work / "formal.json"
        formal = load_json(formal_path) if formal_path.is_file() else None
        attempts = []
        if (work / "attempts.json").is_file():
            attempts = load_json(work / "attempts.json")
        tune = {"attempts": attempts, "formal_path": str(formal_path) if formal_path.is_file() else ""}
        row = qualify_row(case, formal, tune)
        write_json(work / "qualify_row.json", row)
        rows.append(row)
        print(json.dumps({
            "source_case_id": src_id,
            "status": row["status"],
            "cycles": ((row.get("timing") or {}).get("main_kernel") or {}).get("median_cycles"),
            "grid": (row.get("topology") or {}).get("grid"),
            "extra": (row.get("topology") or {}).get("extra_cta_op_kernels"),
            "reason": (row.get("reason") or "")[:160],
        }, sort_keys=True))
    write_qualification(art_root, rows, {"shards": shards, "recompute": True})
    print(json.dumps({
        "qualification": str(art_root / "qualification.json"),
        "n": len(rows),
        "qualified": sum(1 for r in rows if r["status"] == "qualified"),
        "skipped": sum(1 for r in rows if r["status"] == "skipped"),
        "unresolved": sum(1 for r in rows if r["status"] == "unresolved"),
    }, sort_keys=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("generate", "probe", "tune", "formal", "shard", "merge", "recompute"), required=True)
    p.add_argument("--case-id")
    p.add_argument("--shard", type=int)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--shard-workers", type=int, default=1)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--case-list", type=Path, default=None)
    p.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    p.add_argument("--derived-manifest", type=Path, default=DERIVED_MANIFEST)
    p.add_argument("--build-root", type=Path, default=DEFAULT_BUILD)
    p.add_argument("--art-root", type=Path, default=DEFAULT_ART)
    p.add_argument("--skip-formal", action="store_true")
    p.add_argument(
        "--marker-timing",
        action="store_true",
        help=(
            "enable the FA3 device_kernel clock64/globaltimer marker main-kernel "
            "timing in formal (also case marker_timing=true); default off"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.art_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate":
        generate_manifest(args.source_manifest, args.derived_manifest)
        return
    if args.phase == "merge":
        merge_qualification(args.art_root, args.derived_manifest, args.shards)
        return
    if args.phase == "recompute":
        recompute_all(args.art_root, args.derived_manifest, args.shards)
        return
    if not args.derived_manifest.is_file():
        generate_manifest(args.source_manifest, args.derived_manifest)
    doc = load_json(args.derived_manifest)
    cases = doc["cases"]
    sm_count = query_sm_count(args.build_root)
    if args.phase == "formal" and args.case_list is not None:
        # Formal-only measurement lane: reuse the on-disk tuned case, no tuner.
        requested = load_json(args.case_list)
        by_src = {c["source_case_id"]: c for c in cases}
        work_all = [by_src[sid] for sid in requested if sid in by_src]
        missing = [sid for sid in requested if sid not in by_src]
        if missing:
            die(f"case-list ids absent from derived manifest: {missing}")
        work_all = cases_for_worker(work_all, args.worker, args.shard_workers)
        for case in work_all:
            try:
                run_one(
                    case,
                    args.art_root,
                    args.build_root,
                    sm_count,
                    formal=True,
                    reuse_tuned=True,
                    marker_timing=args.marker_timing,
                )
            except Exception as exc:
                print(json.dumps({
                    "source_case_id": case["source_case_id"],
                    "status": "unresolved",
                    "error": str(exc),
                }, sort_keys=True))
        return
    if args.phase in ("probe", "tune", "formal"):
        if not args.case_id:
            die("--case-id required")
        matches = [c for c in cases if c["case_id"] == args.case_id or c["source_case_id"] == args.case_id]
        if len(matches) != 1:
            die(f"case {args.case_id} matched {len(matches)}")
        run_one(
            matches[0],
            args.art_root,
            args.build_root,
            sm_count,
            formal=args.phase == "formal",
            marker_timing=args.marker_timing,
        )
        return
    if args.phase == "shard":
        if args.shard is None:
            die("--shard required")
        slice_cases = cases_for_shard(cases, args.shard, args.shards)
        slice_cases = cases_for_worker(slice_cases, args.worker, args.shard_workers)
        out_rows = args.art_root / "shards" / f"shard{args.shard}.json"
        partial = out_rows.with_suffix(f".w{args.worker}.json")
        rows = []
        for case in slice_cases:
            try:
                rows.append(run_one(
                    case,
                    args.art_root,
                    args.build_root,
                    sm_count,
                    formal=not args.skip_formal,
                    marker_timing=args.marker_timing,
                ))
            except Exception as exc:
                rows.append({
                    "source_case_id": case["source_case_id"],
                    "case_id": case["case_id"],
                    "status": "unresolved",
                    "target_ctas": case["target_ctas"],
                    "uses_cluster": case["uses_cluster"],
                    "parameters": case,
                    "mechanisms_preserved": None,
                    "mechanism_evidence": case.get("mechanism_evidence"),
                    "topology": None,
                    "correctness": {"status": "error", "method": None, "artifact": None},
                    "timing": {"cycles_median": None, "samples_cycles": [], "scope": "whole_operation", "method": None, "artifact": None},
                    "reason": f"runner exception: {exc}",
                    "attempts": [],
                })
                print(json.dumps({"source_case_id": case["source_case_id"], "status": "unresolved", "error": str(exc)}))
        write_json(partial, rows)
        write_json(
            partial.with_suffix(".concurrency.json"),
            {
                "shard": int(args.shard),
                "shards": int(args.shards),
                "workers": int(args.shard_workers),
                "worker": int(args.worker),
                "timing_serialized_per_gpu": True,
                "note": (
                    "workers run concurrently on one GPU; the harness GPU lock "
                    "serializes timing inside each worker"
                ),
            },
        )
        return
    die(f"unknown phase {args.phase}")


if __name__ == "__main__":
    main()
