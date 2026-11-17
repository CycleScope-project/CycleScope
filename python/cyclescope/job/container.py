"""Container ``GSIMJOB`` format, writer, and merged reader (spec §3)."""

from __future__ import annotations

import hashlib
import struct
from typing import Any
from pathlib import Path

from ..convert.semantics_common import ConversionError
from ..io.file_header import (
    FileFormatError, FileReader, HEADER_BYTES, align_up, write_file)
from ..io.fixed_tables import Record, TableReader, TableWriter
from . import segment_format as sf

LNCH = Record("<9IIIIIQIIIIBBHII16s", (
    "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
    "cluster_x", "cluster_y", "cluster_z", "shared_static_bytes",
    "shared_dynamic_bytes", "registers_per_thread", "sm_count",
    "l2_bytes", "occupancy_blocks_per_sm", "occupancy_active_clusters",
    "function_prior_launches",
    "reference_sm_clock_khz", "reference_cache_control",
    "device_facts_present", "pad", "flags", "native_ctas_launched",
    "device"))
SEGD = Record("<IIQQII32s", (
    "segment", "cluster_id", "byte_offset", "byte_length", "cta_base",
    "cta_count", "segment_sha256"))
PLCE = Record("<IIIB3sIIB3s", (
    "cta", "segment", "cta_in_segment", "placement", "pad", "sm_id",
    "warp_slot_base", "source", "pad2"))
GSYN = Record("<IIBBBBQQQQ", (
    "segment", "instance", "lane", "kind", "class_", "flags",
    "address", "after", "before", "sequence"))
TOPO = Record("<IIIIII", (
    "sm", "placement_domain", "l2_partition", "tpc_pair", "gpc", "pad"))

CONTAINER_RECORDS: dict[str, Record] = {
    "LNCH": LNCH, "SEGD": SEGD, "PLCE": PLCE, "GSYN": GSYN,
    "TOPO": TOPO,
}
CONTAINER_TABLES = list(CONTAINER_RECORDS)

LNCH_HAS_UNPLACED = 0x1
LNCH_HAS_GLOBAL_SYNC = 0x2
LNCH_HAS_CLC = 0x4
LNCH_PARTIAL_LAUNCH = 0x8

_SEGMENT_BLOB = "SEGM"


def _container_head(launch: dict, seg_meta: list[tuple[dict, int, bytes]],
                    placements: bytes, global_sync: bytes, topo: bytes,
                    program: dict | None):
    """Header + tables of a ``GSIMJOB`` for segments described by
    (META row, image length, image sha256).  Returns (head bytes with
    file_bytes already patched to the full length, image offsets,
    total length).  Shared by the in-memory and the streaming writer so
    both produce the same bytes."""
    lnch_w = TableWriter(LNCH)
    device = launch.get("device", b"")
    if isinstance(device, str):
        device = device.encode()
    lnch_w.append(tuple(launch.get(f, 0) for f in LNCH.fields[:-1])
                  + (device[:15].ljust(16, b"\0"),))
    for meta, _n, _d in seg_meta[1:]:
        for key in ("tree_digest", "program_digest", "converter_digest"):
            if meta[key] != seg_meta[0][0][key]:
                raise FileFormatError(
                    f"segment META {key} differs across segments")

    # image offsets sit after all table payloads; SEGD size is fixed so the
    # table end is known before SEGD is written.
    segd_size = SEGD.size * len(seg_meta)
    n_tables = len(CONTAINER_TABLES) + (
        len(sf.PROGRAM_TAGS) if program is not None else 0)
    table_end = align_up(HEADER_BYTES + n_tables * 32)
    order = [lnch_w.payload(), None, placements, global_sync, topo]
    if program is not None:
        order += [program[t] for t in sf.PROGRAM_TAGS]
    for i, payload in enumerate(order):
        # write_file pads between tables but not after the last one:
        # the images start at the last payload's unpadded end
        sz = segd_size if payload is None else len(payload)
        table_end += sz if i == len(order) - 1 else align_up(sz)

    segd = TableWriter(SEGD)
    cta_base = 0
    offset = table_end
    offsets = []
    for index, (meta, length, digest) in enumerate(seg_meta):
        segd.append_dict(
            segment=index, cluster_id=meta["cluster_id"],
            byte_offset=offset, byte_length=length,
            cta_base=cta_base, cta_count=meta["cta_count"],
            segment_sha256=digest)
        offsets.append(offset)
        cta_base += meta["cta_count"]
        offset = align_up(offset + length)
    total = align_up(offset)
    tables = [("LNCH", LNCH.size, lnch_w.payload()),
              ("SEGD", SEGD.size, segd.payload()),
              ("PLCE", PLCE.size, placements),
              ("GSYN", GSYN.size, global_sync),
              ("TOPO", TOPO.size, topo)]
    if program is not None:
        for tag in sf.PROGRAM_TAGS:
            tables.append((tag, sf.RECORDS[tag].size, program[tag]))
    head = bytearray(write_file(
        "container", tables, version=2 if program is not None else 1))
    if len(head) != table_end:
        raise FileFormatError("container table end mismatch")
    struct.pack_into("<Q", head, 16, total)
    return bytes(head), offsets, total


def _segment_meta(seg: bytes | memoryview, prog_reader) -> dict:
    r = sf.read_segment(seg, program=prog_reader)
    return TableReader(sf.META, r.require_table("META")).unpack(0)


def write_container(launch: dict, segments: list[bytes],
                    placements: bytes = b"", global_sync: bytes = b"",
                    topo: bytes = b"", program: dict | None = None) -> bytes:
    """Assemble a ``GSIMJOB``: launch tables + verbatim segment images.

    Segment images are appended after the last table, 8-byte aligned, and
    located through ``SEGD.byte_offset/byte_length``.
    """
    prog_reader = (FileReader(memoryview(program["__file__"]))
                   if program and "__file__" in program else None)
    seg_meta = [(_segment_meta(s, prog_reader), len(s),
                 hashlib.sha256(s).digest()) for s in segments]
    head, offsets, total = _container_head(
        launch, seg_meta, placements, global_sync, topo, program)
    out = bytearray(total)
    out[:len(head)] = head
    for off, seg in zip(offsets, segments):
        out[off:off + len(seg)] = seg
    return bytes(out)


def write_container_streamed(out_path, launch: dict, segment_paths: list,
                             placements: bytes = b"", global_sync: bytes = b"",
                             topo: bytes = b"",
                             program: dict | None = None,
                             seg_meta: list | None = None) -> None:
    """Same bytes as write_container, written straight to ``out_path``
    with one segment image in memory at a time (a 2.5 GB job no longer
    costs ~10 GB of concatenation copies).  ``seg_meta`` = precomputed
    (META row, length, sha256) per segment, else read here."""
    prog_reader = (FileReader(memoryview(program["__file__"]))
                   if program and "__file__" in program else None)
    if seg_meta is None:
        seg_meta = []
        for path in segment_paths:
            data = Path(path).read_bytes()
            seg_meta.append((_segment_meta(data, prog_reader), len(data),
                             hashlib.sha256(data).digest()))
            del data
    head, offsets, total = _container_head(
        launch, seg_meta, placements, global_sync, topo, program)
    with open(out_path, "wb") as fh:
        fh.write(head)
        pos = len(head)
        for off, path, (_m, length, _d) in zip(offsets, segment_paths,
                                               seg_meta):
            if off > pos:
                fh.write(b"\0" * (off - pos))
            with open(path, "rb") as sh:
                while True:
                    chunk = sh.read(64 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            pos = off + length
        if total > pos:
            fh.write(b"\0" * (total - pos))


def read_container(data: bytes | memoryview) -> FileReader:
    reader = FileReader(data, expect="container")
    for tag in CONTAINER_TABLES:
        reader.require_table(tag)
    return reader


def segment_images(reader: FileReader) -> list[memoryview]:
    """Slice segment images per SEGD and verify their digests."""
    segd = TableReader(SEGD, reader.require_table("SEGD"))
    view = memoryview(reader._view)
    out = []
    for i in range(len(segd)):
        row = segd.unpack(i)
        image = view[row["byte_offset"]:
                     row["byte_offset"] + row["byte_length"]]
        if hashlib.sha256(image).digest() != row["segment_sha256"]:
            raise FileFormatError(
                f"segment {i} sha256 mismatch (cluster {row['cluster_id']})")
        out.append(image)
    return out


# --- merged load (spec §3) --------------------------------------------------

# tag -> {field_name: referenced table tag}; values are added when merging
_REBASE_U32 = {
    "INST": {"template": "TMPL", "semantics": "SEMA",
             "memory_access": "MACC"},
    "SPLT": {"semantics": "SEMA"},
    "CODE": {"instance": "INST"},
    "CLKM": {"warp": "WARP"},
    "CTAS": {"first_warp": "WARP"},
}
_REBASE_U64 = {
    "WARP": {"first_instance": "INST"},
    "TMPL": {"source_begin": "REGS", "destination_begin": "REGS",
             "timing_begin": "TIME"},
    "MACC": {"sector_begin": "MSEC", "timing_begin": "TIME"},
    "CLKM": {"instance": "INST"},
}
# USES targets rebased against per-instance tables (spec §3)
_USES_TARGET_TABLE = {0: "TMPL", 1: "TIME", 2: "INST", 3: "MACC", 8: "INST"}


def _field_offsets(record: Record) -> dict[str, int]:
    import re
    offsets = {}
    pos = 0
    items = []
    for token in re.findall(r"\d*[a-zA-Z]", record.fmt.lstrip("<>=!")):
        count = int(token[:-1]) if len(token) > 1 else 1
        ch = token[-1]
        items.extend([ch] * (count if ch != "s" else 1))
    for token, name in zip(items, record.fields):
        size = struct.calcsize("<" + token)
        if name is not None:
            offsets[name] = pos
        pos += size
    return offsets


def _rebase(buf: bytearray, record: Record, counts: list[int],
            bases: dict[str, list[int]],
            u32: dict[str, str], u64: dict[str, str]) -> None:
    offsets = _field_offsets(record)
    row_start = sum(counts[:1])
    for seg_index in range(1, len(counts)):
        for row in range(row_start, row_start + counts[seg_index]):
            base = row * record.size
            for field, table in u32.items():
                off = offsets[field]
                v = struct.unpack_from("<I", buf, base + off)[0]
                if v != 0xFFFFFFFF:
                    struct.pack_into(
                        "<I", buf, base + off,
                        v + sum(bases[table][:seg_index]))
            for field, table in u64.items():
                off = offsets[field]
                v = struct.unpack_from("<Q", buf, base + off)[0]
                if v != 0xFFFFFFFFFFFFFFFF:
                    struct.pack_into(
                        "<Q", buf, base + off,
                        v + sum(bases[table][:seg_index]))
        row_start += counts[seg_index]


def build_container_from_dir(src_dir, out_path) -> Path:
    """Assemble segments/<n>.seg + launch.json into a GSIMJOB container."""
    import json as _json
    src_dir = Path(src_dir)
    out_path = Path(out_path)
    launch_doc = _json.loads((src_dir / "launch.json").read_text())
    seg_paths = [src_dir / "segments" / name
                 for name in launch_doc.get("segments") or []]
    if not seg_paths:
        seg_paths = sorted((src_dir / "segments").glob("*.seg"))
    if not seg_paths:
        raise FileFormatError(f"{src_dir}: no segment images")

    program = None
    pt = src_dir / "segments" / "program.tables"
    if pt.is_file():
        pr = FileReader(memoryview(pt.read_bytes()))
        program = {tag: bytes(pr.require_table(tag))
                   for tag in sf.PROGRAM_TAGS}
        program["__file__"] = pt.read_bytes()

    plce = TableWriter(PLCE)
    for cta_text, p in sorted(
            (launch_doc.get("placements") or {}).items(),
            key=lambda item: int(item[0])):
        cta = int(cta_text)
        plce.append((
            cta, 0, cta, int(p.get("placement", 1)), b"\0" * 3,
            int(p.get("sm_id", NONE_SENTINEL)),
            int(p.get("warp_slot_base", 0)),
            int(p.get("source", 1)), b"\0" * 3))

    topo = TableWriter(TOPO)
    topo_path = launch_doc.get("sm_topology")
    if topo_path:
        doc = _json.loads(Path(topo_path).read_text())
        for row in doc.get("rows", []):
            topo.append((
                row["sm"], row.get("placement_domain", NONE_SENTINEL),
                row.get("l2_partition", NONE_SENTINEL),
                row.get("tpc_pair", NONE_SENTINEL),
                row.get("gpc", NONE_SENTINEL), 0))

    v = int(launch_doc.get("reference_cache_control") or 0)
    if v == 0:
        raise ConversionError(
            "missing_fact:reference_cache_control (launch.json)")
    if v == 1:
        raise ConversionError(
            "unsupported:reference_cache_control:none (launch.json)")
    if v != 2:
        raise ConversionError(
            f"unsupported:reference_cache_control:{v} (launch.json)")

    launch = dict(launch_doc)
    for prefix in ("grid", "block", "cluster"):
        dims = launch_doc.get(prefix)
        if isinstance(dims, (list, tuple)) and len(dims) == 3:
            launch[f"{prefix}_x"], launch[f"{prefix}_y"], \
                launch[f"{prefix}_z"] = (int(d) for d in dims)
    for field in LNCH.fields[:-1]:
        if field in launch and not isinstance(launch[field], int):
            try:
                launch[field] = int(launch[field])
            except (TypeError, ValueError):
                launch[field] = 0
        if isinstance(launch.get(field), str):
            launch[field] = 0
    grid = launch_doc.get("grid") or []
    total_ctas = 1
    for d in grid:
        total_ctas *= int(d)
    placements_doc = launch_doc.get("placements") or {}
    launch["flags"] = (
        (LNCH_PARTIAL_LAUNCH if launch_doc.get("partial_launch") else 0)
        | (LNCH_HAS_UNPLACED
           if len(placements_doc) < total_ctas else 0))
    if "gsyn_rows" in launch_doc:
        raise FileFormatError(
            "launch.json carries gsyn_rows from a pre-segment-GSYN "
            "converter; rebuild the segments")
    # one pass over the images: META row, length, sha256 and the GSYN
    # rows; the images are then streamed into the file
    prog_reader = (FileReader(memoryview(program["__file__"]))
                   if program and "__file__" in program else None)
    gsync = bytearray()
    seg_gsyn = sf.OPTIONAL_RECORDS["GSYN"]
    seg_meta = []
    for s, path in enumerate(seg_paths):
        data = path.read_bytes()
        seg_meta.append((_segment_meta(data, prog_reader), len(data),
                         hashlib.sha256(data).digest()))
        payload = sf.read_segment(data).table("GSYN")
        del data
        if payload is None:
            continue
        for i in range(len(payload) // seg_gsyn.size):
            r = seg_gsyn.unpack(payload, i)
            gsync += GSYN.pack_row((
                s, r["instance"], r["lane"], r["kind"], r["class_"],
                r["flags"], r["address"], r["after"], r["before"],
                r["sequence"]))
    write_container_streamed(out_path, launch, seg_paths,
                             placements=plce.payload(),
                             global_sync=bytes(gsync),
                             topo=topo.payload(), program=program,
                             seg_meta=seg_meta)
    return out_path


NONE_SENTINEL = 0xFFFFFFFF


def merge_segments(images: list[bytes | memoryview],
                   program=None) -> dict[str, Any]:
    """Merge segments into container-wide table buffers (spec §3).

    Per-instance tables are concatenated with index fields rebased by the
    target table's running base; program tables come from segment 0 after
    checking every segment's META digests agree; USES rows keep segment 0
    for global targets and rebase per-instance targets.  STRS and FACT
    share the program-table prefix across segments and carry a
    per-segment tail (interned strings, per-instance FACT rows): the
    common prefix is kept once, each later segment's tail is appended and
    the referring offsets (TMPL.operation_class, FACT.axis,
    USES.fact_begin) are rebased.
    """
    if not images:
        raise FileFormatError("container has no segments")
    readers = [sf.read_segment(img, program=program)
               for img in images]
    metas = [TableReader(sf.META, r.require_table("META")).unpack(0)
             for r in readers]
    for meta in metas[1:]:
        for key in ("tree_digest", "program_digest", "converter_digest"):
            if meta[key] != metas[0][key]:
                raise FileFormatError(
                    f"segment META {key} differs across segments")

    merged: dict[str, bytearray] = {}
    counts: dict[str, list[int]] = {}
    for tag, record in sf.RECORDS.items():
        if record is None or tag in ("META", "STRS", "PROG", "NODE",
                                     "GRD ", "LEAF", "PARM", "PTS ",
                                     "FACT", "USES"):
            continue
        out = bytearray()
        rows = []
        for reader in readers:
            payload = reader.require_table(tag)
            rows.append(len(payload) // record.size)
            out.extend(payload)
        merged[tag] = out
        counts[tag] = rows
    for tag in ("PROG", "NODE", "GRD ", "LEAF", "PARM", "PTS ", "AXIS"):
        first = readers[0].require_table(tag)
        for reader in readers[1:]:
            if reader.require_table(tag) != first:
                raise FileFormatError(
                    f"container {tag} differs across segments")
        merged[tag] = bytearray(first)

    # STRS / FACT: shared prefix once, per-segment tails appended; USES:
    # segment-0 rows plus rebased per-instance-target rows
    strs = bytearray(readers[0].require_table("STRS"))
    strs0 = bytes(strs)
    facts = bytearray(readers[0].require_table("FACT"))
    facts0_rows = len(facts) // sf.FACT.size
    uses_out = bytearray(readers[0].require_table("USES"))
    tmpl_off = _field_offsets(sf.TMPL)["operation_class"]
    tmpl_row = 0
    for seg_index, reader in enumerate(readers[1:], start=1):
        tmpl_row += counts["TMPL"][seg_index - 1]
        seg_strs = bytes(reader.require_table("STRS"))
        k = 0
        lim = min(len(strs0), len(seg_strs))
        while k < lim and strs0[k] == seg_strs[k]:
            k += 1
        while k > 0 and seg_strs[k - 1] != 0:
            k -= 1
        strs_base = len(strs)
        strs.extend(seg_strs[k:])

        def remap(off, k=k, strs_base=strs_base):
            return off if off < k else off - k + strs_base

        tr = TableReader(sf.USES, reader.require_table("USES"))
        rows = [tr.unpack(i) for i in range(len(tr))]
        seg_facts = reader.require_table("FACT")
        seg_fact_rows = len(seg_facts) // sf.FACT.size
        k_fact = min([r["fact_begin"] for r in rows
                      if r["fact_count"] > 0
                      and r["target"] in _USES_TARGET_TABLE],
                     default=seg_fact_rows)
        if k_fact > facts0_rows or \
                bytes(seg_facts[:k_fact * sf.FACT.size]) != \
                bytes(facts[:k_fact * sf.FACT.size]):
            raise FileFormatError(
                f"segment {seg_index} FACT prefix differs from segment 0")
        fact_base = len(facts) // sf.FACT.size
        ft = TableReader(sf.FACT, seg_facts)
        for i in range(k_fact, seg_fact_rows):
            fr = ft.unpack(i)
            fr["axis"] = remap(fr["axis"])
            facts.extend(sf.FACT.pack(**fr))
        tbuf = merged["TMPL"]
        for r in range(counts["TMPL"][seg_index]):
            base = (tmpl_row + r) * sf.TMPL.size + tmpl_off
            v = struct.unpack_from("<I", tbuf, base)[0]
            struct.pack_into("<I", tbuf, base, remap(v))
        for row in rows:
            table = _USES_TARGET_TABLE.get(row["target"])
            if table is None:
                continue  # global target: segment-0 copy already kept
            row["target_index"] += sum(counts[table][:seg_index])
            if row["fact_count"] > 0:
                if row["fact_begin"] >= k_fact:
                    row["fact_begin"] += fact_base - k_fact
                elif row["fact_begin"] + row["fact_count"] > k_fact:
                    raise FileFormatError(
                        "USES fact range straddles the FACT prefix")
            uses_out.extend(sf.USES.pack(**row))
    merged["STRS"] = strs
    merged["FACT"] = facts
    merged["USES"] = uses_out

    # rebase index fields in per-instance tables
    bases = {t: counts[t] for t in counts}
    for tag, fields in _REBASE_U32.items():
        _rebase(merged[tag], sf.RECORDS[tag], counts[tag], counts,
                fields, _REBASE_U64.get(tag, {}))
    for tag, fields in _REBASE_U64.items():
        if tag not in _REBASE_U32:
            _rebase(merged[tag], sf.RECORDS[tag], counts[tag], counts,
                    {}, fields)
    # CTA identity: segments carry the launch-wide linear CTA id in
    # CTAS/WARP/MBPI/MBOB.cta_index; the merged job indexes CTAs densely
    # in table order (cta_index == row in CTAS)
    _renumber_ctas(merged, readers, counts)
    return {"tables": merged, "meta": metas[0], "counts": counts}


def _renumber_ctas(merged, readers, counts) -> None:
    ctas = merged["CTAS"]
    cta_off = _field_offsets(sf.CTAS)["cta_index"]
    row = 0
    seg_maps = []
    for seg_index, reader in enumerate(readers):
        local = {}
        for j in range(counts["CTAS"][seg_index]):
            base = (row + j) * sf.CTAS.size + cta_off
            linear = struct.unpack_from("<I", ctas, base)[0]
            if linear in local:
                raise FileFormatError("duplicate CTA id in a segment")
            local[linear] = row + j
            struct.pack_into("<I", ctas, base, row + j)
        seg_maps.append(local)
        row += counts["CTAS"][seg_index]
    for tag, rec in (("WARP", sf.WARP), ("MBPI", sf.MBPI)):
        buf = merged[tag]
        off = _field_offsets(rec)["cta_index"]
        r0 = 0
        for seg_index in range(len(readers)):
            for j in range(counts[tag][seg_index]):
                base = (r0 + j) * rec.size + off
                v = struct.unpack_from("<I", buf, base)[0]
                if v not in seg_maps[seg_index]:
                    raise FileFormatError(
                        f"{tag} names a CTA outside its segment")
                struct.pack_into("<I", buf, base, seg_maps[seg_index][v])
            r0 += counts[tag][seg_index]
    mbob = sf.OPTIONAL_RECORDS["MBOB"]
    out = bytearray()
    for seg_index, reader in enumerate(readers):
        payload = reader.table("MBOB")
        if not payload:
            continue
        for j in range(len(payload) // mbob.size):
            r = mbob.unpack(payload, j)
            if r["cta_index"] not in seg_maps[seg_index]:
                raise FileFormatError("MBOB names a CTA outside its segment")
            r["cta_index"] = seg_maps[seg_index][r["cta_index"]]
            out.extend(mbob.pack(**r))
    if out:
        merged["MBOB"] = out
