"""capture case_dir -> GSIMCAP01 facts (spec §4.1 last row).

    convert_capture(case_dir, arch, out_dir, launch=None,
                    clusters="all", shard=None, reference=None,
                    placement=None, cache=True)
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import model, static_code, poll_folding, control_flow
from . import operand_facts as _operand_facts
from . import tensor_map as _tensor_map
from . import global_sync_observation as _gsync
from . import trace_reader
from .facts_format import StreamingFactsWriter, write_statics_file
from .sass_decode import MissingFact, opcode_width_bytes

_ACCESS_ROOTS = {"LDG": "load", "STG": "store", "LD": "load", "ST": "store",
                 "LDS": "load", "STS": "store", "LDL": "load", "STL": "store",
                 "LDC": "load", "ULDC": "load", "LDCU": "load",
                 "LDSM": "load", "STSM": "store",
                 "LDGSTS": "async_copy", "LDGSTS": "async_copy",
                 "UTMALDG": "async_copy", "UTMASTG": "store",
                 "UTMAREDG": "async_reduce", "UTMAPF": "prefetch",
                 "UBLKCP": "async_copy", "UBLKRED": "async_reduce",
                 "ATOM": "atomic", "ATOMG": "atomic", "ATOMS": "atomic",
                 "RED": "atomic", "REDG": "atomic", "REDS": "atomic",
                 "STAS": "store", "STTM": "store", "LDTM": "load",
                 "SYNCS": "other", "ARRIVES": "other", "UTCBAR": "other",
                 "UCGABAR": "other", "ATOMICS": "atomic"}

_TMA_ROOTS = {"UTMALDG", "UTMASTG", "UTMAREDG", "UTMAPF"}
_BULK_ROOTS = {"UBLKCP", "UBLKRED"}


def _artifact_root() -> Path:
    from .. import site
    return site.artifact_root()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_digest() -> str:
    h = hashlib.sha256()
    base = Path(__file__).resolve().parent.parent
    for rel in ["convert/trace_reader.py", "convert/static_code.py",
                "convert/poll_folding.py", "convert/control_flow.py",
                "convert/operand_facts.py", "convert/tensor_map.py",
                "convert/global_sync_observation.py",
                "convert/capture_pipeline.py", "convert/sass_decode.py",
                "convert/facts_format.py",
                "arch/hopper/capture_rules.py",
                "arch/blackwell/capture_rules.py"]:
        f = base / rel
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()


def _cta_linear(cta, grid) -> int:
    gx, gy = max(1, int(grid[0])), max(1, int(grid[1]))
    return int(cta[0]) + int(cta[1]) * gx + int(cta[2]) * gx * gy


def _cluster_linear(cluster, dims, grid) -> int:
    cx, cy, cz = (max(1, int(d)) for d in dims)
    ncx = -(-int(grid[0]) // cx)
    ncy = -(-int(grid[1]) // cy)
    return int(cluster[0]) + int(cluster[1]) * ncx \
        + int(cluster[2]) * ncx * ncy


def _cluster_dims(row) -> tuple:
    eff = row.get("effective_cluster") or row.get("cluster") or [1, 1, 1]
    return tuple(max(1, int(v)) for v in eff)


def _cluster_filter_spec(clusters, dims, grid, sampled):
    """-> set of linear cluster ids to keep, or None (all)."""
    if clusters in (None, "all"):
        return None
    if clusters.startswith("sampled:"):
        n = int(clusters.split(":", 1)[1])
        ids = {_cluster_linear(list(c), dims, grid)
               for c in sampled[:n]}
        return ids
    return {int(t) for t in clusters.split(",") if t.strip()}


def split_capture(case_dir, nshards: int, out_dir, *, launch=None) -> Path:
    """One decompression pass -> per-shard trace files
    (cyclescope-trace-split).  Shard converts then read only their own
    records via --split-dir instead of rescanning the whole trace."""
    import shutil
    import subprocess
    case_dir = Path(case_dir)
    out_dir = Path(out_dir)
    launches = [json.loads(l) for l in
                (case_dir / "launches.jsonl").read_text().splitlines()]
    if launch is None:
        traced = [l for l in launches if l.get("traced")]
        if len(traced) != 1:
            raise SystemExit(f"unsupported:ambiguous_launch ({case_dir})")
        launch = int(traced[0]["launch"])
    row = next((l for l in launches if int(l["launch"]) == launch), None)
    if row is None:
        raise SystemExit(f"missing_fact:launch_row ({case_dir})")
    trace = case_dir / f"trace-{launch}.bin.zst"
    if not trace.is_file():
        trace = case_dir / f"trace-{launch}.bin"
    if not trace.is_file():
        raise SystemExit(f"missing_fact:trace_file ({case_dir})")
    grid = row.get("grid") or [1, 1, 1]
    dims = _cluster_dims(row)
    bin_ = shutil.which("cyclescope-trace-split")
    if bin_ is None:
        raise SystemExit("unsupported:no_trace_split_binary "
                         "(cyclescope-trace-split not in PATH)")
    out_dir.mkdir(parents=True, exist_ok=True)
    # trace sha256 sidecar: every shard convert needs the digest of the
    # whole compressed trace (META digest, launch.json trace_sha256);
    # hash it once here, overlapped with the split, instead of once per
    # shard.  `sha256sum` when available, hashlib otherwise.
    sha_proc = None
    if shutil.which("sha256sum"):
        sha_proc = subprocess.Popen(["sha256sum", str(trace)],
                                    stdout=subprocess.PIPE, text=True)
    r = subprocess.run(
        [bin_, "--trace", str(trace),
         "--functions", str(case_dir / "functions"),
         "--grid", ",".join(str(int(g)) for g in grid),
         "--cluster", ",".join(str(int(d)) for d in dims),
         "--shards", str(nshards), "--out", str(out_dir)])
    if sha_proc is not None:
        out, _ = sha_proc.communicate()
        sha = out.split()[0] if sha_proc.returncode == 0 and out else ""
    else:
        sha = ""
    if r.returncode != 0:
        raise SystemExit(f"unsupported:trace_split_failed ({case_dir})")
    if len(sha) != 64:
        sha = _file_sha256(trace)
    (out_dir / f"trace-{launch}.sha256").write_text(sha + "\n")
    return out_dir


def convert_capture(case_dir, arch: str, out_dir, *, launch=None,
                    clusters="all", shard=None, reference=None,
                    placement=None, cache=True, split_dir=None) -> Path:
    case_dir = Path(case_dir)
    out_dir = Path(out_dir)
    rules = importlib.import_module(f"cyclescope.arch.{arch}.rules")
    cap_rules = importlib.import_module(
        f"cyclescope.arch.{arch}.capture_rules")

    launches = [json.loads(l) for l in
                (case_dir / "launches.jsonl").read_text().splitlines()]
    if launch is None:
        traced = [l for l in launches if l.get("traced")]
        if len(traced) != 1:
            raise SystemExit(f"unsupported:ambiguous_launch ({case_dir})")
        launch = int(traced[0]["launch"])
    row = next((l for l in launches if int(l["launch"]) == launch), None)
    if row is None:
        raise SystemExit(f"missing_fact:launch_row ({case_dir})")

    trace = case_dir / f"trace-{launch}.bin.zst"
    if not trace.is_file():
        trace = case_dir / f"trace-{launch}.bin"
    if not trace.is_file():
        raise SystemExit(f"missing_fact:trace_file ({case_dir})")
    trace_sha = _file_sha256(trace)

    # facts-level cache
    key = hashlib.sha256(
        f"{trace_sha}|{launch}|{clusters}|{shard}|{_source_digest()}"
        .encode()).hexdigest()[:32]
    cache_dir = _artifact_root() / "cache" / "facts" / key
    if cache and (cache_dir / "launch.json").is_file() and \
            list(cache_dir.glob("*.gsc")):
        out_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in cache_dir.iterdir():
            shutil.copy2(f, out_dir / f.name)
        return out_dir

    dims = _cluster_dims(row)
    grid = row.get("grid") or [1, 1, 1]
    block = row.get("block") or [1, 1, 1]
    sampled = row.get("sampled_clusters") or []
    if sampled and isinstance(sampled[0], dict):
        sampled = [(c["x"], c["y"], c["z"]) for c in sampled]
    keep = _cluster_filter_spec(clusters, dims, grid, sampled)
    if shard:
        k, n = shard
        ids = keep if keep is not None else \
            {_cluster_linear(c, dims, grid) for c in sampled}
        keep = {c for c in ids if c % n == k}

    device = {}
    dpath = case_dir / "device.json"
    if dpath.is_file():
        device = json.loads(dpath.read_text())
    dev_name = str(device.get("name", "")).lower()
    device_key = "b300" if arch == "blackwell" else "h100"

    tables = static_code.load_case(case_dir, rules)
    # static_index: global launch-unique ordinal
    statics: list = []
    statics_rows: list = []
    sindex: dict = {}
    for fid in sorted(tables):
        for idx in sorted(tables[fid].rows):
            r = tables[fid].rows[idx]
            r.static_index = len(statics)
            statics.append(r.inst)
            statics_rows.append(r)
            sindex[(fid, idx)] = r

    def static_of(rec):
        return sindex.get((rec.function_id, rec.idx))

    by_pc_cache: dict = {}

    def static_by_pc_of(fid):
        got = by_pc_cache.get(fid)
        if got is None:
            got = {r.inst.pc: r for r in tables[fid].rows.values()}
            by_pc_cache[fid] = got
        return got

    tmaps = _tensor_map.load_tensor_maps(case_dir)
    tm_models: dict = {}
    tm_facts: dict = {}

    missing: dict = defaultdict(int)
    unsupported: dict = defaultdict(int)
    dropped_polls = 0

    ncx = -(-int(grid[0]) // dims[0])
    ncy = -(-int(grid[1]) // dims[1])

    def rec_keep(cta):
        if split_dir is not None and shard:
            return True           # split file already routed
        cl = (cta[0] // dims[0]) + (cta[1] // dims[1]) * ncx + \
            (cta[2] // dims[2]) * ncx * ncy
        return keep is None or cl in keep

    cta_group = 2 if any(".2CTA" in r.inst.opcode.upper()
                         for r in statics_rows) else 1
    hw_device = _hardware_device(device_key)
    tmem_columns = int(hw_device.get("tmem_columns", 0)) \
        if arch == "blackwell" else 0
    placement_trace: dict = {}

    # split trace stream (cyclescope-trace-split output): shard reads only
    # its own records; trace_sha/cache key stay on the original file
    trace_path = None
    gsync_sequences = None
    if split_dir is not None and shard:
        k, n = shard
        base = (Path(split_dir) / f"trace-{launch}"
                f".shard{k}of{n}")
        trace_path = Path(f"{base}.bin.zst")
        if not trace_path.is_file():
            raise SystemExit(
                f"missing_fact:split_trace ({trace_path})")
        seq_file = Path(f"{base}.gsync_seq")
        if not seq_file.is_file():
            raise SystemExit(
                f"missing_fact:split_gsync_sequence ({seq_file})")
        import struct as _st
        _data = seq_file.read_bytes()
        gsync_sequences = iter(
            _st.unpack(f"<{len(_data) // 8}Q", _data))

    out_dir.mkdir(parents=True, exist_ok=True)
    # buffer records per (cluster -> (cta,warp) -> [recs])
    t0 = time.monotonic()
    by_warp: dict = defaultdict(lambda: defaultdict(list))
    headers: dict = {}
    cta_info: dict = {}
    for rec in trace_reader.iter_records(case_dir, launch,
                                         cluster_filter=rec_keep,
                                         trace_path=trace_path,
                                         gsync_sequences=gsync_sequences,
                                         cta_info=cta_info):
        info = cta_info.get(rec.cta)
        if info is not None:
            cid = info["clusterid"]
            ncl = info["nclusterid"]
            shp = info["cluster_nctaid"]
            # mixed-shape launches: bare linear ids collide across
            # shapes; encode (lin, shape) into one int so each real
            # cluster gets a unique id deterministically (shards too).
            lin = cid[0] + cid[1] * ncl[0] + cid[2] * ncl[0] * ncl[1]
            cl = lin * 4096 + shp[0] * 256 + shp[1] * 16 + shp[2]
        else:
            cl = (rec.cta[0] // dims[0]
                  + (rec.cta[1] // dims[1]) * ncx
                  + (rec.cta[2] // dims[2]) * ncx * ncy)
        by_warp[cl][(rec.cta, rec.warp)].append(rec)
        headers[(rec.cta, rec.warp)] = (rec.sm, rec.warp_slot)

    for _key in sorted(by_warp):
        cl = _key
        writer = StreamingFactsWriter(split_statics=True)
        ctas_f: dict = {}
        warps_done: list = []
        used_maps: set = set()
        # two passes per CTA: control_participating_warps needs all warps
        for (cta, warp), recs in sorted(by_warp[_key].items()):
            folded, dropped, origins = poll_folding.fold_warp(
                recs, static_of)
            dropped_polls += dropped
            patches = control_flow.flow_patches(
                recs, static_of, static_by_pc_of, cta)
            warps_done.append((cta, warp, recs, folded, patches, origins))
        # participating warps per loop key
        patches_map = {f"{cta}-{warp}": p
                       for cta, warp, _r, _f, p, _o in warps_done}
        control_flow.fill_participating(patches_map)

        if cta_info:
            # hardware cluster facts: rank = %cluster_ctarank, cluster
            # size = product(%cluster_nctaid)
            def _hw_rank(c):
                inf = cta_info.get(c)
                return (inf["cluster_ctarank"] if inf is not None
                        else 0)
            cluster_rank_list = sorted({c for c, _w in by_warp[_key]},
                                       key=_hw_rank)
            rank_of = {c: _hw_rank(c) for c in cluster_rank_list}
            cluster_size = max(
                1, max((int(i["cluster_nctaid"][0])
                        * int(i["cluster_nctaid"][1])
                        * int(i["cluster_nctaid"][2])
                        for i in cta_info.values()), default=1))
        else:
            cluster_rank_list = sorted({c for c, _w in by_warp[_key]},
                                       key=lambda c: (c[2], c[1], c[0]))
            rank_of = {c: i for i, c in enumerate(cluster_rank_list)}
            cluster_size = len(cluster_rank_list)

        for cta, warp, recs, folded, patches, origins in warps_done:
            sm, wslot = headers[(cta, warp)]
            cl_linear = _cta_linear(cta, grid)
            if cta not in ctas_f:
                ctas_f[cta] = model.CtaFacts(
                    cta_linear=cl_linear, cta=cta,
                    cluster_rank=rank_of[cta],
                    sm_id=sm if sm is not None else 0xFFFFFFFF,
                    cta_group=cta_group,
                    is_pair_leader=int(cta_group == 2
                                       and rank_of[cta] % 2 == 0),
                    multicast_mask=(1 << cluster_size) - 1,
                    tmem_columns=tmem_columns)
            if warp == 0 or cta not in placement_trace:
                placement_trace.setdefault(
                    cta, {"sm_id": sm if sm is not None else 0xFFFFFFFF,
                          "warp_slot_base": wslot or 0, "source": 1})
                if warp == 0:
                    placement_trace[cta] = {
                        "sm_id": sm if sm is not None else 0xFFFFFFFF,
                        "warp_slot_base": wslot or 0, "source": 1}
            writer.begin_warp(cl_linear, cta, sm or 0, wslot or 0, warp)
            for pos, rec in enumerate(recs):
                if pos in folded:
                    continue
                srow = static_of(rec)
                if srow is None:
                    missing["static_row"] += 1
                    continue
                ranks = _operand_facts.remote_ranks(rec.ops or {},
                                                    rank_of[cta])
                ops = _operand_facts.operand_facts(
                    rec, srow, cap_rules, rank_of[cta], ranks)
                memory = _memory_refs(rec, srow, cap_rules, rules,
                                      tmaps, tm_models, tm_facts,
                                      used_maps, missing, unsupported,
                                      ranks, ops)
                patch = {k: v for k, v in patches.get(pos, {}).items()
                         if not k.startswith("_")}
                if patch:
                    if ops is None:
                        ops = model.SemanticOperands()
                    ops.uniform.update(patch)
                if pos in origins:
                    if ops is None:
                        ops = model.SemanticOperands()
                    ops.uniform["wait_origin"] = origins[pos]
                gsyncs, glane = _gsync.observations(rec, srow)
                if ops is None and glane:
                    ops = model.SemanticOperands(lane=glane)
                elif glane:
                    ops.lane.update(glane)
                writer.add_warp_instruction(model.DynamicInstruction(
                    static_index=srow.static_index,
                    active_mask=rec.active_mask,
                    predicate_mask=rec.exec_mask,
                    memory=tuple(memory), operands=ops,
                    global_sync=gsyncs[0] if gsyncs else None))
            writer.end_warp()

        by_rank = {c.cluster_rank: c for c in ctas_f.values()}
        for c in ctas_f.values():
            peer = by_rank.get(c.cluster_rank ^ 1) if cta_group == 2 else None
            c.peer_sm_id = peer.sm_id if peer is not None else c.sm_id
        mbpi = _mbarrier_preinits(arch, device_key, cta_group,
                                  [c.cta_linear for c in ctas_f.values()],
                                  device, unsupported)
        digest = hashlib.sha256(
            f"{trace_sha}|{launch}|{cl}".encode()).digest()
        writer.finish(cl, list(ctas_f.values()), statics,
                      {i: tm_models[i] for i in sorted(used_maps)
                       if i in tm_models},
                      mbpi, out_dir / f"{cl:06d}.gsc", digest=digest)

    write_statics_file(statics, out_dir / "statics.gsc")

    launch_doc = _launch_doc(case_dir, launch, row, device, device_key,
                             dims, keep, reference, placement,
                             trace_sha, dropped_polls, missing,
                             unsupported, placement_trace, grid,
                             cta_info_present=bool(cta_info))
    (out_dir / "launch.json").write_text(
        json.dumps(launch_doc, indent=1, sort_keys=True))
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in out_dir.iterdir():
            if f.name != "launch.json" or True:
                shutil.copy2(f, cache_dir / f.name)
    n_records = sum(len(r) for w in by_warp.values() for r in w.values())
    print(f"convert_capture {launch}: {n_records} records, "
          f"{sum(len(w) for w in by_warp.values())} warps -> "
          f"{len(by_warp)} clusters in "
          f"{time.monotonic() - t0:.1f}s", file=sys.stderr)
    return out_dir


def _access_kind(root: str) -> str:
    return _ACCESS_ROOTS.get(root, "other")


def _space_of(rec, srow, static_space) -> str:
    if static_space == "generic" and rec.ops.get("SPACE"):
        g, s, c = rec.ops["SPACE"]
        if c:
            return "shared_cluster"
        if s:
            return "shared"
        if g:
            return "global"
    return static_space


def _memory_refs(rec, srow, cap_rules, rules, tmaps, tm_models,
                 tm_facts, used_maps, missing, unsupported, ranks, ops
                 ) -> list:
    """MemoryReference list from a record's mref lanes + static layout."""
    root = srow.inst.opcode.split(".", 1)[0].upper()
    where = f"{srow.inst.pc:#x}"
    if root in NO_MEMORY_REFERENCE_ROOTS:
        return []
    if root in _CONSTANT_ROOTS:
        return _constant_refs(rec, srow, ops, missing)
    if not srow.mref_count and root not in _TMA_ROOTS \
            and root not in _BULK_ROOTS:
        return []
    if rec.exec_mask == 0 and (root in _TMA_ROOTS or root in _BULK_ROOTS):
        return []   # predicated-off issue: no transfer, no facts
    static_space = srow.inst.memory_space
    uniform = ops.uniform if ops is not None else {}
    multicast = uniform.get("multicast_mask")
    destination_count = None
    if multicast:
        destination_count = int(multicast).bit_count()
    out = []
    if root in _TMA_ROOTS:
        tm_id = ops.tensor_map if ops is not None else None
        if tm_id is not None:
            used_maps.add(tm_id)
        coords = list(ops.tensor_coordinates or ()) if ops is not None \
            else []
        entry = tmaps.get(tm_id) if tm_id is not None else None
        if tm_id is not None and tm_id not in tm_facts and entry is None:
            missing["tensor_map_row"] += 1
            tm_facts[tm_id] = {"geometry": None}
        if tm_id is not None and tm_id not in tm_facts:
            try:
                tm_models[tm_id], tm_facts[tm_id] = \
                    _tensor_map.tensor_map_entry(entry, srow.opcode_text,
                                                 where)
            except MissingFact:
                missing["tensor_map_decode"] += 1
                tm_facts[tm_id] = {"geometry": None}
        facts = tm_facts.get(tm_id, {}) if tm_id is not None else {}
        if tm_id is None:
            missing["tensor_map"] += 1
        sectors = []
        cut_rows = 0
        transfer = facts.get("global_bytes")
        geo = facts.get("geometry")
        if geo and coords:
            if geo.get("mode") == "im2col":
                if geo["element_strides"][0] != 1:
                    unsupported["tma_im2col_element_stride"] += 1
                else:
                    # loads carry one packed-offsets UR; store/reduce
                    # (im2col_no_offs forms) carry none
                    bare = _operand_facts._bare_ur_tokens(
                        list(srow.inst.operands))
                    want = 1 if root == "UTMALDG" else 0
                    offs = None
                    if len(bare) != want:
                        unsupported["tma_im2col_operands"] += 1
                    elif bare:
                        packed = uniform.get(f"UR{bare[0]}")
                        if packed is None:
                            missing["im2col_offsets"] += 1
                        else:
                            offs = _tensor_map._im2col_offsets(
                                geo["rank"], packed)
                    else:
                        offs = [0] * (geo["rank"] - 2)
                    if offs is not None:
                        try:
                            sectors = _tensor_map.im2col_box_sectors(
                                geo, coords[:geo["rank"]], offs)
                        except MissingFact:
                            unsupported["tma_im2col_coordinate"] += 1
            else:
                try:
                    if root == "UTMALDG":
                        # tma_oob v2: a tiled load whose box is partly
                        # in bounds along dimension 0 pays per cut row;
                        # whole rows outside (other dims) cost nothing
                        cut_rows = _tensor_map.tiled_cut_rows(
                            geo, coords)
                    sectors = _tensor_map.tma_box_sectors(
                        geo, coords)
                except MissingFact:
                    unsupported["tma_box"] += 1
        elif geo and not coords:
            missing["tensor_coordinates"] += 1
        elif facts.get("api") == "im2col":
            unsupported["tma_im2col_box"] += 1
        out.append(model.MemoryReference(
            space="global", access_kind=_access_kind(root),
            width_bytes=_SECTOR_, lane_addresses=tuple(sectors),
            lane_mask=0, destination_count=destination_count,
            transfer_bytes=transfer, cut_rows=cut_rows))
        return out
    if root in _BULK_ROOTS:
        size = uniform.get("bytes")
        if not _operand_facts.bulk_has_global_side(srow.opcode_text):
            # shared -> shared cluster copy: no global sectors, but the
            # transfer_bytes fact (service start / completion leaves)
            if size is None:
                missing["bulk_copy_bytes"] += 1
            out.append(model.MemoryReference(
                space="shared", access_kind=_access_kind(root),
                width_bytes=_SECTOR_, lane_addresses=(), lane_mask=0,
                destination_count=destination_count, transfer_bytes=size))
            return out
        if size is None:
            missing["bulk_copy_bytes"] += 1
        # the global side's address is the .G bracket's UR pair (the
        # tracer records both brackets since 2026-11-08; NVBit's mref is
        # the shared side for .G.S / .S.G forms); older traces fall back
        # to the mref and are reported as missing
        base = _operand_facts.bulk_global_address(
            srow.opcode_text, list(srow.inst.operands), uniform)
        if base is None:
            missing["bulk_global_address"] += 1
            base = int(rec.mrefs[0][0]) if rec.mrefs and rec.mrefs[0] \
                else None
        sectors = _tensor_map.bulk_sectors(base, size or 0) \
            if base is not None else []
        out.append(model.MemoryReference(
            space="global", access_kind=_access_kind(root),
            width_bytes=_SECTOR_, lane_addresses=tuple(sectors),
            lane_mask=0, destination_count=destination_count,
            transfer_bytes=size))
        return out
    try:
        width = _width(srow.opcode_text, where)
    except MissingFact:
        missing["opcode_width"] += 1
        width = 0
    kind = _access_kind(root)
    # LDGSTS: mref0 is the shared destination, mref1 the global source;
    # the global side comes first (the memory hierarchy reads the first
    # reference), the shared side carries the bank facts
    order = [1, 0] if root == "LDGSTS" and srow.mref_count == 2 \
        else range(srow.mref_count)
    for m in order:
        addrs = rec.mrefs[m] if m < len(rec.mrefs) else ()
        if root == "LDGSTS":
            space = "global" if m == 1 else "shared"
        elif m == 0:
            space = _space_of(rec, srow, static_space)
        else:
            space = static_space
        remote = None
        lane_ranks = ranks.get(m)
        if lane_ranks:
            space = "shared_cluster"
            own = None
            for v in lane_ranks:
                if v >= 0:
                    own = v
                    break
            remote = own
        dcount = destination_count
        if root == "STAS" and lane_ranks:
            dcount = len({v for v in lane_ranks if v >= 0})
        out.append(model.MemoryReference(
            space=space or "global", access_kind=kind,
            width_bytes=width or 0,
            lane_addresses=tuple(int(a) for a in addrs),
            lane_mask=rec.exec_mask, remote_rank=remote,
            destination_count=dcount))
    return out


_CONSTANT_ROOTS = frozenset({"LDC", "ULDC", "LDCU"})
_CONST_BANK_RE = re.compile(
    r"^c\[(0x[0-9a-fA-F]+)\]\[(?:(UR\d+|R\d+|RZ|URZ)\+?)?(0x[0-9a-fA-F]+)?\]$")


def _constant_refs(rec, srow, ops, missing) -> list:
    """LDC/ULDC/LDCU: constant address = bank offset immediate + captured
    register addend (per lane for R, uniform for UR); the SASS token is
    the static part, the register value the dynamic one."""
    if rec.exec_mask == 0:
        return []
    token = next((t for t in srow.inst.operands
                  if t.strip().startswith("c[")), None)
    m = _CONST_BANK_RE.match(token.strip()) if token else None
    if m is None:
        missing["constant_operand"] += 1
        return []
    bank = int(m.group(1), 16)
    imm = int(m.group(3), 16) if m.group(3) else 0
    reg = m.group(2)
    # bank in the high bits: c[0x0][a] and c[0x3][a] are different lines
    # in the engine's constant cache (bank 0 behaviour unchanged)
    def _addr(offset):
        return (bank << 32) | (offset & 0xFFFFFFFF)

    lanes = rec.exec_mask.bit_count()
    if reg is None or reg in ("RZ", "URZ"):      # zero register addend
        addrs = (_addr(imm),) * lanes
    elif reg.startswith("UR"):
        value = (ops.uniform.get(reg) if ops is not None else None)
        if value is None:
            missing["constant_address"] += 1
            return []
        addrs = (_addr(value + imm),) * lanes
    else:
        vals = ops.lane.get(reg) if ops is not None else None
        if not vals:
            missing["constant_address"] += 1
            return []
        addrs = tuple(_addr(v + imm) for v in vals)
    try:
        width = _width(srow.opcode_text, f"{srow.inst.pc:#x}") or 4
    except MissingFact:
        missing["opcode_width"] += 1
        width = 4
    return [model.MemoryReference(
        space="constant", access_kind="load", width_bytes=width,
        lane_addresses=addrs, lane_mask=rec.exec_mask)]


NO_MEMORY_REFERENCE_ROOTS = frozenset({
    "LDTM", "STTM", "UTCCP", "UTCHMMA", "UTCQMMA", "UTCOMMA", "UTCBMMA",
    "UTCIMMA", "UTCMMA", "UTCFMMA", "UTCDMMA", "UTCBFMMA", "UTCATOMSWS",
    "UGETNEXTWORKID", "UTMACCTL", "UTMACMDFLUSH", "UVIRTCOUNT",
})


_SECTOR_ = 32


def _width(opcode: str, where: str) -> int | None:
    return opcode_width_bytes(opcode, where)


def _hardware_device(device_key: str) -> dict:
    path = Path(__file__).resolve().parents[3] / "configs" / "hardware" \
        / device_key / "device.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _mbarrier_preinits(arch: str, device_key: str, cta_group: int,
                       cta_linears, device: dict, unsupported) -> list:
    """Hardware-preinitialized shared-memory mbarriers (blackwell
    reserved window), expanded for every CTA of the unit.  The predecessor
    emitted them only for barriers that were named but never initialized;
    emitting the whole table is a superset (objects never referenced stay
    idle)."""
    if arch != "blackwell":
        return []
    table_path = Path(__file__).resolve().parents[3] / "configs" \
        / "hardware" / device_key / "preinitialized_mbarriers.json"
    if not table_path.is_file():
        return []
    table = json.loads(table_path.read_text())
    want = str((table.get("identity") or {}).get("gpu") or "")
    name = str(device.get("name") or "")
    if want and name and want != name:
        unsupported["preinitialized_mbarriers:device"] += 1
        return []
    rows = []
    for cta_linear in cta_linears:
        for b in table.get("barriers", []):
            arrivals = b.get("expected_arrivals")
            if isinstance(arrivals, dict):
                arrivals = arrivals.get(str(cta_group),
                                        arrivals.get("default"))
            if arrivals is None:
                continue
            rows.append((int(cta_linear), int(b["offset"]), int(arrivals)))
    return rows


def _launch_doc(case_dir, launch, row, device, device_key, dims, keep,
                reference, placement, trace_sha, dropped_polls, missing,
                unsupported, placement_trace=None, grid=None,
                cta_info_present=False) -> dict:
    doc = {
        "device": device_key,
        "sm_arch": device.get("sm_arch") or
                   ("sm_103a" if device_key == "b300" else "sm_90a"),
        "launch": launch,
        "function": row.get("mangled") or row.get("name"),
        "grid": row.get("grid"), "block": row.get("block"),
        "cluster": list(dims),
        "cluster_source": "hardware" if cta_info_present else "attribute",
        "preferred_cluster": row.get("preferred_cluster"),
        "cluster_ids": sorted(keep) if keep is not None else None,
        "sampled_clusters": row.get("sampled_clusters"),
        "shared_static_bytes": row.get("smem_static", 0),
        "shared_dynamic_bytes": row.get("smem_dynamic", 0),
        "registers_per_thread": int(row["regs"])
        if row.get("regs") is not None else 0,
        "sm_count": device.get("sm_count"),
        "l2_bytes": device.get("l2_cache_bytes"),
        "occupancy_blocks_per_sm": row.get("occupancy_max_blocks_per_sm"),
        "occupancy_blocks_per_sm_tracer":
            row.get("occupancy_max_blocks_per_sm"),
        "occupancy_source": "tracer",
        "occupancy_active_clusters": row.get("occupancy_max_active_clusters"),
        "function_prior_launches": row.get("function_prior_launches", 0),
        "capture": str(case_dir),
        "trace_sha256": trace_sha,
        "dropped_polls": dropped_polls,
        "missing_facts": dict(missing),
        "unsupported": dict(unsupported),
        "placements": {str(_cta_linear(c, grid or row.get("grid")
                                       or [1, 1, 1])): p
                       for c, p in (placement_trace or {}).items()},
    }
    topo = Path(__file__).resolve().parents[3] / "configs" / "hardware" \
        / device_key / "sm_topology.json"
    if topo.is_file():
        doc["sm_topology"] = str(topo)
    occ = _ncu_occupancy(case_dir, row)
    if occ is not None:
        doc["occupancy_blocks_per_sm"] = occ
        doc["occupancy_source"] = "ncu"
    if reference:
        try:
            lines = Path(reference).read_text().splitlines()
            hdr = lines[0].split("\t")
            for line in lines[1:]:
                p = line.split("\t")
                if p and p[0].strip() == Path(case_dir).name:
                    row_map = dict(zip(hdr, p))
                    if row_map.get("sm_clock_khz"):
                        doc["reference_sm_clock_khz"] = int(
                            float(row_map["sm_clock_khz"]))
                    if row_map.get("cache_control"):
                        doc["reference_cache_control"] = {
                            "none": 1, "all": 2}.get(
                            row_map["cache_control"].strip(), 0)
                    doc["reference_source"] = str(reference)
                    break
        except (IndexError, ValueError, OSError):
            pass
    if placement:
        pdir = Path(placement)
        pj = pdir / "placement.jsonl" if (pdir / "placement.jsonl") \
            .is_file() else pdir
        if Path(pj).is_file():
            probe = _placements(Path(pj), row, dims, doc)
            if probe:
                doc["placements"] = probe
    return doc


def _ncu_occupancy(case_dir, row) -> int | None:
    """min(launch__occupancy_limit_*) for the traced launch's ncu row.
    Matched by kernel name + occurrence order + block_size; tracer-reported
    regs are post-instrumentation so they are not a match key.
    Sources, in order: the capture's own census (``<case>/ncu.csv``, old
    flow) or the reference run's ncu csv next to the capture
    (``<captures>/reference/<case>/<case>/ncu_r*.csv``, `cyclescope run`
    layout); the limits are static kernel properties, identical in both.
    A capture root may be a farm of per-case symlinks: the reference/
    sibling lives next to the resolved real case dir, so the resolved
    path is tried as a second candidate."""
    case_dir = Path(case_dir)
    candidates = [case_dir]
    resolved = case_dir.resolve()
    if resolved != case_dir:
        candidates.append(resolved)
    ncu = next((p for d in candidates
                for p in [d / "ncu.csv"] + sorted(
                    (d.parent / "reference" / d.name / d.name)
                    .glob("ncu_r*.csv"))
                if p.is_file()), None)
    if ncu is None:
        return None
    try:
        text = ncu.read_text().splitlines()
        hi = next((i for i, l in enumerate(text)
                   if l.startswith('"ID"')), None)
        if hi is None:
            return None
        import csv, io
        rows = [r for r in csv.DictReader(
            io.StringIO("\n".join(text[hi:]))) if r.get("ID")]
        fname = row.get("mangled") or row.get("name") or ""
        same = [r for r in rows
                if (r.get("Kernel Name") or "") == fname]
        nth = int(row.get("function_prior_launches") or 0)
        if nth >= len(same):
            return None
        cand = same[nth]
        block = row.get("block") or [1, 1, 1]
        want = int(block[0]) * int(block[1]) * int(block[2])
        got = cand.get("launch__block_size") or ""
        if got and int(float(got.replace(",", ""))) != want:
            return None
        lims = []
        for k in ("launch__occupancy_limit_registers",
                  "launch__occupancy_limit_shared_mem",
                  "launch__occupancy_limit_blocks",
                  "launch__occupancy_limit_warps"):
            v = (cand.get(k) or "").replace(",", "")
            if v:
                lims.append(int(float(v)))
        return min(lims) if lims else None
    except (ValueError, KeyError, IndexError, OSError):
        return None


def _placements(path, row, dims, doc) -> dict:
    """placement.jsonl entry records -> {cta_linear: Placement source 2}
    for the first wave only: non-cluster launches
    linear < sm_count * occupancy_blocks_per_sm, cluster launches
    cluster_id < occupancy_active_clusters."""
    grid = row.get("grid") or [1, 1, 1]
    sm_count = doc.get("sm_count") or 0
    occ_blocks = doc.get("occupancy_blocks_per_sm") or 0
    occ_clusters = doc.get("occupancy_active_clusters")
    cluster_launch = any(int(d) > 1 for d in dims)
    seen: dict = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("kind") != "entry":
            continue
        cta = tuple(d["cta"])
        key = _cta_linear(cta, grid)
        if cluster_launch:
            if occ_clusters is None:
                continue
            cl = (cta[0] // dims[0]) + (cta[1] // dims[1]) * \
                (-(-int(grid[0]) // dims[0])) + (cta[2] // dims[2]) * \
                (-(-int(grid[0]) // dims[0])) * (-(-int(grid[1]) // dims[1]))
            if cl >= occ_clusters:
                continue
        elif key >= sm_count * occ_blocks:
            continue
        if key not in seen or d["warp"] < seen[key]["_warp"]:
            seen[key] = {"sm_id": d["smid"],
                         "warp_slot_base": d["warpid"], "source": 2,
                         "_warp": d["warp"]}
    return {str(k): {kk: vv for kk, vv in v.items() if kk != "_warp"}
            for k, v in seen.items()}
