#!/usr/bin/env python3
"""Re-tune FA3 cluster-SM cases whose K-only sweep could not reach 100k-500k cycles."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from cluster_sm_lib import (  # noqa: E402
    CYCLES_MAX,
    CYCLES_MIN,
    clip_seq_lists,
    mechanism_still_ok,
    scale_num_splits,
    scale_seqlen_k,
    scale_seqlen_q,
)
from run_cluster_sm import (  # noqa: E402
    DEFAULT_ART,
    DEFAULT_BUILD,
    attach_cluster_fields,
    inject_sm_margin,
    invoke_full_gpu,
    qualify_row,
    query_sm_count,
    write_json,
)

ART = Path(os.environ.get("FLASHATTENTION3_CLUSTER_SM_ROOT", str(DEFAULT_ART)))
BUILD = Path(os.environ.get("FLASHATTENTION3_FULL_GPU_BUILD_ROOT", str(DEFAULT_BUILD)))
PROBE_FLOOR = CYCLES_MIN * 1.15


def load(path: Path):
    return json.loads(path.read_text())


def sanitize(case: dict, sm_count: int) -> dict:
    return inject_sm_margin(clip_seq_lists(case), sm_count)


def topology_ok(case: dict, result: dict) -> bool:
    topo = (result.get("cluster_sm") or {}).get("topology") or {}
    if bool(topo.get("observed_uses_cluster")) != bool(case["uses_cluster"]):
        return False
    if case["operation"] == "combine":
        comb = [k for k in (topo.get("kernels") or []) if k.get("kind") == "fa3_combine"]
        return bool(comb) and comb[0].get("cta_count") == 1
    fwd = topo.get("observed_fwd_cta_count")
    return fwd == case["target_ctas"]


def probe_variant(case: dict, work: Path, build: Path, tag: str) -> dict | None:
    ok, traits = mechanism_still_ok(case)
    if not ok:
        print(json.dumps({"tag": tag, "skip": "mechanism", "traits_n_m": traits.get("n_m_tiles")}))
        return None
    out = work / "attempts" / f"repair_{tag}.json"
    try:
        result = invoke_full_gpu(case, out, probe=True, build_root=build)
    except Exception as exc:
        print(json.dumps({"tag": tag, "error": str(exc)[-400:]}))
        return None
    result = attach_cluster_fields(case, result)
    write_json(out, result)
    cycles = (result.get("timing") or {}).get("median_cycles")
    topo = (result.get("cluster_sm") or {}).get("topology") or {}
    print(json.dumps({
        "tag": tag,
        "seqlen_q": case.get("seqlen_q"),
        "seqlen_k": case.get("seqlen_k"),
        "num_splits": case.get("num_splits"),
        "seqlens_k": case.get("seqlens_k"),
        "cycles": cycles,
        "status": result.get("status"),
        "fwd_ctas": topo.get("observed_fwd_cta_count"),
        "cluster": topo.get("observed_uses_cluster"),
        "topo_ok": topology_ok(case, result),
    }, sort_keys=True))
    if result.get("status") != "pass" or cycles is None:
        return None
    if not topology_ok(case, result):
        return None
    return {"case": case, "result": result, "cycles": float(cycles)}


def unique_candidates(seed: dict) -> list[tuple[str, dict]]:
    primary = seed["primary_mechanism"]
    q0 = int(seed.get("seqlen_q") or 1)
    k0 = int(seed.get("seqlen_k") or 1)
    sp0 = int(seed.get("num_splits") or 1)
    out: list[tuple[str, dict]] = []

    def add(tag: str, trial: dict) -> None:
        ok, _ = mechanism_still_ok(trial)
        if ok:
            out.append((tag, trial))

    if primary == "combine_standalone" or seed.get("operation") == "combine":
        for sp in (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024):
            if sp > sp0:
                add(f"splits{sp}", scale_num_splits(seed, sp))
        return out

    if primary in ("attention_chunk", "local_window"):
        for q in (q0 * 2, q0 * 3, q0 * 4, q0 * 6, q0 * 8, q0 * 12, 2048, 3072, 4096, 6144, 8192):
            if q > q0:
                add(f"q{q}", scale_seqlen_q(seed, q))
        return out

    if primary == "kvcache_append_rotary":
        for q, k in ((32, 4096), (64, 8192), (96, 8192), (128, 12288), (192, 16384), (256, 16384)):
            trial = scale_seqlen_q(seed, q)
            trial = scale_seqlen_k(trial, k)
            add(f"q{q}_k{trial['seqlen_k']}", trial)
        return out

    if primary == "paged_tma":
        add("clip", clip_seq_lists(seed))
        for k in (4096, 6144, 8192, 12288, 16384):
            if k > k0:
                add(f"k{k}", scale_seqlen_k(clip_seq_lists(seed), k))
        for q in (q0 * 2, q0 * 3, q0 * 4):
            add(f"q{q}", scale_seqlen_q(clip_seq_lists(seed), q))
        return out

    # Cluster parity is encoded in PRIMARY_OK / n_m; reject broken Q via add().
    if seed.get("uses_cluster") or primary == "cluster_odd":
        for mult in (1.15, 1.25, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0):
            k = max(k0 + 128, int(round(k0 * mult)))
            add(f"k{k}", scale_seqlen_k(seed, k))
        return out

    for mult in (1.15, 1.25, 1.4, 1.6, 2.0, 3.0, 4.0):
        k = max(k0 + 64, int(round(k0 * mult)))
        add(f"k{k}", scale_seqlen_k(seed, k))
    for q in (int(q0 * 1.25), int(q0 * 1.5), q0 * 2, q0 * 3, q0 * 4):
        if q > q0:
            add(f"q{q}", scale_seqlen_q(seed, q))
    if sp0 >= 2:
        for sp in (sp0 * 2, sp0 * 4, sp0 * 8):
            add(f"sp{sp}", scale_num_splits(seed, sp))
    return out


def repair_one(source_id: str, sm_count: int) -> dict | None:
    work = ART / "cases" / source_id
    seed_path = work / "tuned_case.json"
    if not seed_path.is_file():
        seed_path = work / "derived_seed.json"
    if not seed_path.is_file():
        man = load(Path(os.environ.get(
            "FLASHATTENTION3_CLUSTER_SM_MANIFEST",
            str(HARNESS.parent / "manifests" / "fa3_cluster_sm100.json"),
        )))
        hit = [c for c in man["cases"] if c["source_case_id"] == source_id]
        if not hit:
            print(json.dumps({"repaired": source_id, "error": "no seed"}))
            return None
        seed = hit[0]
    else:
        seed = load(seed_path)
    seed = sanitize(seed, sm_count)
    write_json(work / "repair_seed.json", seed)

    best = None
    for tag, trial in unique_candidates(seed)[:14]:
        trial = sanitize(trial, sm_count)
        hit = probe_variant(trial, work, BUILD, tag)
        if not hit:
            continue
        cyc = hit["cycles"]
        if PROBE_FLOOR <= cyc <= CYCLES_MAX:
            best = hit
            break
        if best is None or abs(cyc - 250000) < abs(best["cycles"] - 250000):
            best = hit
        if cyc > CYCLES_MAX and best["cycles"] > CYCLES_MAX:
            break

    if best is None:
        print(json.dumps({"repaired": source_id, "status": "unresolved", "reason": "no passing probe"}))
        return None
    final = sanitize(best["case"], sm_count)
    formal_path = work / "formal.json"
    formal = invoke_full_gpu(final, formal_path, probe=False, build_root=BUILD)
    formal = attach_cluster_fields(final, formal)
    write_json(formal_path, formal)
    write_json(work / "tuned_case.json", final)
    row = qualify_row(final, formal, {"attempts": [], "formal_path": str(formal_path)})
    write_json(work / "qualify_row.json", row)
    print(json.dumps({
        "repaired": source_id,
        "status": row["status"],
        "cycles": row["timing"]["cycles_median"],
        "reason": row["reason"],
        "fwd_ctas": (row.get("topology") or {}).get("observed_fwd_cta_count"),
        "cluster": (row.get("topology") or {}).get("observed_uses_cluster"),
    }, sort_keys=True))
    return row


def repair_ids() -> list[str]:
    if len(sys.argv) > 1:
        return sys.argv[1:]
    ids = []
    for d in sorted((ART / "cases").iterdir()):
        if not d.is_dir():
            continue
        q = d / "qualify_row.json"
        if not q.is_file():
            ids.append(d.name)
            continue
        row = load(q)
        if row.get("status") == "unresolved":
            ids.append(d.name)
    return ids


def main() -> None:
    sm_count = query_sm_count(BUILD)
    ids = repair_ids()
    print(json.dumps({"repair_ids": ids, "n": len(ids), "sm_count": sm_count}))
    for sid in ids:
        try:
            repair_one(sid, sm_count)
        except Exception as exc:
            print(json.dumps({"repaired": sid, "error": str(exc)[-500:]}))


if __name__ == "__main__":
    main()
