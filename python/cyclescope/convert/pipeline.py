"""Facts -> segments + launch.json (docs/en/design/convert.md).

    cyclescope convert --facts <dir> --arch hopper|blackwell \
        --tree configs/trees/<device>.json --out <dir>

Produces <out>/segments/<cluster_id>.seg plus <out>/launch.json consumed
by `cyclescope build`.
"""

from __future__ import annotations

import importlib
import json
import resource
import sys
import time
from pathlib import Path

from ..convert.mechanism_uses import TreeRuntime
from ..convert.semantics_common import ConversionError
from ..job import program_tables, segment_builder
from ..job import segment_format as sf


def _rules(arch: str):
    return importlib.import_module(f"cyclescope.arch.{arch}.rules")


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def _convert_units(units, launch, arch: str, tree_path: Path,
                   out_dir: Path, cache_dir: Path | None = None,
                   capture: str | None = None,
                   split_program: bool = True) -> dict:
    rules = _rules(arch)
    tree = json.loads(Path(tree_path).read_text())
    programs = program_tables.compile_program_tables(tree)
    runtime = TreeRuntime(tree)
    from ..tree.registry import load_registry
    registry = load_registry(
        Path(__file__).resolve().parents[3] / "configs" /
        "schema_registry.json")
    schema_targets = {name: s.target
                      for name, s in registry.schemas.items()}
    from ..job.cache import (SegmentCache, converter_digest, entry_name,
                             entry_prefix)
    src_digest = converter_digest()
    t0 = time.monotonic()
    t_read = time.monotonic()
    print(f"READ={t_read - t0:.2f} READ_RSS_MB={_rss_mb()}",
          file=sys.stderr)
    cache = None
    if cache_dir:
        cache = SegmentCache(Path(cache_dir))
    tree_digest = tree.get("content_digest", "00" * 32)
    counts = {"hit": 0, "relink": 0, "build": 0}

    out_dir = Path(out_dir)
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "tree_digest": bytes.fromhex(tree_digest),
        "program_digest": bytes.fromhex(
            tree.get("program_digest",
                     tree.get("content_digest", "00" * 32))),
        "converter_digest": bytes.fromhex(src_digest),
        "cluster_shape": launch.cluster,
    }
    segments = []
    missing: dict = {}
    unsupported: dict = {}
    for unit in units:
        rep = {}
        meta["schema_targets"] = schema_targets
        seg_name = f"{unit.cluster_id:06d}.seg"
        if cache is not None:
            name = entry_name(capture, unit.cluster_id, src_digest,
                              tree_digest)
            hit = cache.get(name)
            if hit is not None:
                (seg_dir / seg_name).write_bytes(hit)
                segments.append(seg_name)
                counts["hit"] += 1
                continue
            prior = cache.find_for_relink(
                capture, unit.cluster_id, src_digest)
            if prior is not None:
                from ..job.relink import relink_segment
                try:
                    image = relink_segment(prior[1], tree,
                                           runtime=runtime)
                    (seg_dir / seg_name).write_bytes(image)
                    cache.put(name, image, {
                        "capture": capture,
                        "cluster_id": unit.cluster_id,
                        "src": src_digest, "tree": tree_digest,
                        "relinked_from": prior[0].name})
                    segments.append(seg_name)
                    counts["relink"] += 1
                    continue
                except ConversionError as e:
                    if not str(e).startswith("unsupported:relink_"):
                        raise
                    # relink impossible -> rebuild
        image = segment_builder.build_segment(
            unit, rules, programs, launch_meta=meta, report=rep,
            runtime=runtime, split_program=split_program)
        for k, n in rep.get("missing_facts_builder", {}).items():
            missing[k] = missing.get(k, 0) + n
        for k, n in rep.get("unsupported_builder", {}).items():
            unsupported[k] = unsupported.get(k, 0) + n
        (seg_dir / seg_name).write_bytes(image)
        if cache is not None:
            cache.put(entry_name(capture, unit.cluster_id, src_digest,
                                 tree_digest),
                      image, {"capture": capture,
                              "cluster_id": unit.cluster_id,
                              "src": src_digest, "tree": tree_digest})
        segments.append(seg_name)
        counts["build"] += 1
    t_build = time.monotonic()
    print(f"BUILD={t_build - t_read:.2f} RSS_MB={_rss_mb()} "
          f"CACHE hit={counts['hit']} relink={counts['relink']} "
          f"build={counts['build']}", file=sys.stderr)

    launch_doc = _launch_doc(launch, segments, missing, unsupported)
    (seg_dir / "program.tables").write_bytes(
        sf.program_tables_payload(programs))

    (out_dir / "launch.json").write_text(
        json.dumps(launch_doc, indent=1, sort_keys=True))
    return {"segments": segments, "launch": launch_doc}


def _launch_doc(launch, segments, missing: dict,
                unsupported: dict) -> dict:
    """launch.json payload shared by the python and cpp build paths."""
    doc = {
        "device": launch.device or "",
        "grid": list(launch.grid),
        "block": list(launch.block),
        "cluster": list(launch.cluster),
        "shared_static_bytes": launch.shared_static_bytes,
        "shared_dynamic_bytes": launch.shared_dynamic_bytes,
        "registers_per_thread": launch.registers_per_thread,
        "sm_count": launch.sm_count,
        "l2_bytes": launch.l2_bytes,
        "occupancy_blocks_per_sm": launch.occupancy_blocks_per_sm,
        "missing_facts": {
            k: (getattr(launch, "extra", {}) or {})
            .get("missing_facts", {}).get(k, 0) + v
            for k, v in missing.items()
        } | {
            k: v
            for k, v in ((getattr(launch, "extra", {}) or {})
                         .get("missing_facts") or {}).items()
            if k not in missing},
        "unsupported": {
            k: (getattr(launch, "extra", {}) or {})
            .get("unsupported", {}).get(k, 0) + v
            for k, v in unsupported.items()
        } | {
            k: v
            for k, v in ((getattr(launch, "extra", {}) or {})
                         .get("unsupported") or {}).items()
            if k not in unsupported},
        "occupancy_active_clusters": launch.occupancy_active_clusters,
        "function_prior_launches": launch.function_prior_launches,
        "reference_sm_clock_khz": launch.reference_sm_clock_khz,
        "reference_cache_control": launch.reference_cache_control,
        "placements": {
            str(k): {"sm_id": p.sm_id, "warp_slot_base": p.warp_slot_base,
                     "source": p.source}
            for k, p in launch.placements.items()},
        "sm_topology": str(launch.sm_topology)
            if launch.sm_topology else None,
        "segments": segments,
    }
    for k, v in (getattr(launch, "extra", None) or {}).items():
        doc.setdefault(k, v)
    return doc


def _facts_launch(facts_dir: Path):
    """LaunchFacts from the case's launch.json sidecar."""
    import json
    from . import model
    launch_doc = json.loads((Path(facts_dir) / "launch.json").read_text())
    launch = model.LaunchFacts(
        device=launch_doc.get("device", ""),
        sm_arch=launch_doc.get("sm_arch", ""),
        grid=tuple(launch_doc.get("grid", (1, 1, 1))),
        block=tuple(launch_doc.get("block", (1, 1, 1))),
        cluster=tuple(launch_doc.get("cluster", (1, 1, 1))),
        shared_static_bytes=launch_doc.get("shared_static_bytes", 0),
        shared_dynamic_bytes=launch_doc.get("shared_dynamic_bytes", 0),
        registers_per_thread=max(0, int(launch_doc.get("registers_per_thread") or 0)),
        sm_count=launch_doc.get("sm_count") or 0,
        l2_bytes=launch_doc.get("l2_bytes") or 0,
        occupancy_blocks_per_sm=launch_doc.get("occupancy_blocks_per_sm")
        or 0,
        occupancy_active_clusters=launch_doc.get(
            "occupancy_active_clusters") or 0,
        function_prior_launches=launch_doc.get("function_prior_launches", 0),
        reference_sm_clock_khz=launch_doc.get("reference_sm_clock_khz")
        or 0,
        reference_cache_control=launch_doc.get("reference_cache_control")
        or 0,
        sm_topology=Path(launch_doc["sm_topology"])
        if launch_doc.get("sm_topology") else None,
    )
    for k, v in (launch_doc.get("placements") or {}).items():
        if isinstance(v, dict):
            launch.placements[int(k)] = model.Placement(
                int(v.get("sm_id", 0xFFFFFFFF)),
                int(v.get("warp_slot_base", 0)), int(v.get("source", 1)))
        else:
            launch.placements[int(k)] = model.Placement(*v)
    launch.extra = {k: launch_doc[k] for k in (
        "launch", "function", "cluster_ids", "sampled_clusters",
        "reference_source", "capture", "trace_sha256",
        "dropped_polls", "missing_facts", "unsupported")
        if k in launch_doc}
    return launch


def _facts_units(facts_dir: Path, arch: str):
    """GSIMCAP01 path: reconstruct ClusterUnits lazily + LaunchFacts."""
    from .facts_format import FactsReader
    from . import model
    launch = _facts_launch(facts_dir)
    units = []
    for g in sorted(Path(facts_dir).glob("[0-9]*.gsc")):
        rd = FactsReader(g)
        unit = model.ClusterUnit(cluster_id=rd.cluster_id)
        unit.ctas = rd.ctas()
        unit.statics = rd.statics()
        unit.warps = list(rd.warp_views())
        unit.tensor_maps = rd.tensor_maps()
        unit.preinitialized_mbarriers = rd.preinitialized_mbarriers()
        units.append(unit)
    return units, launch


def convert_facts(facts_dir: Path, arch: str, tree_path: Path,
                  out_dir: Path, cache_dir: Path | None = None,
                  split_program: bool = True,
                  builder: str = "python") -> dict:
    """_convert_units() with units sourced from GSIMCAP01."""
    if builder == "cpp":
        return convert_facts_cpp(facts_dir, arch, tree_path, out_dir,
                                 cache_dir)
    units, launch = _facts_units(Path(facts_dir), arch)
    return _convert_units(units, launch, arch, tree_path, out_dir,
                          cache_dir, split_program=split_program,
                          capture=_capture_identity(facts_dir))


def _capture_identity(facts_dir: Path) -> str:
    """Segment-cache identity of a facts dir: the trace content digest
    when the sidecar has one (a case dir can be re-captured in place),
    else the capture path, else the facts dir itself."""
    doc = json.loads((Path(facts_dir) / "launch.json").read_text())
    return (doc.get("trace_sha256") or doc.get("capture")
            or str(Path(facts_dir).resolve()))


def _program_table_file(tree_path: Path, cache_dir: Path | None) -> Path:
    """Compile tree -> GSIMPRG01 cache file (shared with cyclescope-convert)."""
    import hashlib
    from ..job import program_tables
    from ..job.cache import converter_digest
    tree = json.loads(Path(tree_path).read_text())
    tree_digest = tree.get("content_digest") or hashlib.sha256(
        Path(tree_path).read_bytes()).hexdigest()
    # the compiled image depends on the compiler too (PARM/PTS encoding):
    # key the cache entry by tree digest x converter source digest
    digest = hashlib.sha256(
        f"{tree_digest}|{converter_digest()}".encode()).hexdigest()
    from .. import site
    root = Path(cache_dir) if cache_dir else site.artifact_root()
    gsp = root / "cache" / "programs" / f"{digest}.gsp"
    gsp.parent.mkdir(parents=True, exist_ok=True)
    if not gsp.exists():
        from ..tree.registry import load_registry
        reg = load_registry(
            Path(__file__).resolve().parents[3] / "configs" /
            "schema_registry.json")
        import os, secrets
        data = program_tables.program_file_image(tree, reg)
        tmp = gsp.with_name(
            f"{digest}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        tmp.write_bytes(data)
        try:
            os.link(tmp, gsp)          # atomic; fails if it exists
            tmp.unlink()
        except FileExistsError:
            tmp.unlink()               # another writer won; use theirs
        except OSError:
            os.replace(tmp, gsp)
    return gsp


def convert_facts_cpp(facts_dir: Path, arch: str, tree_path: Path,
                      out_dir: Path, cache_dir: Path | None = None) -> dict:
    """C++ builder path: cyclescope-convert per facts dir (no relink)."""
    import os
    import shutil
    import subprocess
    import tempfile
    from .facts_format import FactsReader
    from ..job.cache import (SegmentCache, converter_digest, entry_name)
    facts_dir = Path(facts_dir)
    gsp = _program_table_file(tree_path, cache_dir)
    out_dir = Path(out_dir)
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    launch = _facts_launch(facts_dir)
    tree = json.loads(Path(tree_path).read_text())
    tree_digest = tree.get("content_digest", "00" * 32)
    src_digest = converter_digest()
    cap = _capture_identity(facts_dir)
    t0 = time.monotonic()
    cache = SegmentCache(Path(cache_dir)) if cache_dir else None
    plan = []   # (seg_name, entry_name, cached_bytes|None)
    for g in sorted(facts_dir.glob("[0-9]*.gsc")):
        cid = FactsReader(g).cluster_id
        seg_name = f"{cid:06d}.seg"
        name = entry_name(cap, cid, src_digest, tree_digest)
        hit = cache.get(name) if cache is not None else None
        plan.append((seg_name, name, hit))
    t_read = time.monotonic()
    print(f"READ={t_read - t0:.2f} READ_RSS_MB={_rss_mb()}",
          file=sys.stderr)
    counts = {"hit": 0, "relink": 0, "build": 0}
    missing: dict = {}
    unsupported: dict = {}
    if cache is not None and all(h is not None for _n, _e, h in plan):
        for seg_name, _name, hit in plan:
            (seg_dir / seg_name).write_bytes(hit)
            counts["hit"] += 1
    else:
        bin_ = shutil.which("cyclescope-convert") or str(
            Path(__file__).resolve().parents[3] / "build" /
            "cyclescope-convert")
        fd, mtmp = tempfile.mkstemp(
            prefix="missing_", suffix=".json", dir=out_dir)
        os.close(fd)
        cmd = [bin_, "--facts", str(facts_dir), "--programs", str(gsp),
               "--arch", arch, "--out", str(seg_dir),
               "--converter-digest", src_digest,
               "--missing-json", mtmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            os.unlink(mtmp)
            raise RuntimeError(
                f"cyclescope-convert failed: {r.stderr[:800]}")
        rep = json.loads(Path(mtmp).read_text())
        os.unlink(mtmp)
        missing = rep.get("missing_facts", {})
        unsupported = rep.get("unsupported", {})
        for seg_name, name, hit in plan:
            data = (seg_dir / seg_name).read_bytes()
            if cache is not None and hit is None:
                cache.put(name, data, {
                    "capture": cap, "cluster_id":
                    int(seg_name.split(".")[0]),
                    "src": src_digest, "tree": tree_digest})
            counts["build"] += 1
    t_build = time.monotonic()
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    print(f"BUILD={t_build - t_read:.2f} RSS_MB={rss} "
          f"CACHE hit={counts['hit']} relink={counts['relink']} "
          f"build={counts['build']}", file=sys.stderr)
    segments = [seg_name for seg_name, _n, _h in plan]
    launch_doc = _launch_doc(launch, segments, missing, unsupported)
    # cyclescope-convert normally writes segments/program.tables (segment
    # v2); an all-cache-hit run skips the binary, so emit it here too.
    pt = seg_dir / "program.tables"
    if not pt.is_file():
        import shutil
        shutil.copyfile(gsp, pt)

    (out_dir / "launch.json").write_text(
        json.dumps(launch_doc, indent=1, sort_keys=True))
    return {"segments": segments, "launch": launch_doc}
