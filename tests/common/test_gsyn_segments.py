"""Segment-level GSYN table: container build, shard merge, old
launch.json rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyclescope.pipeline import shards as merge_facts_shards  # noqa: E402

from cyclescope.io.file_header import FileFormatError  # noqa: E402
from cyclescope.io.fixed_tables import TableReader  # noqa: E402
from cyclescope.job import container as cf, segment_format as sf  # noqa: E402

_SGSYN = sf.OPTIONAL_RECORDS["GSYN"]


def _segment(cluster_id: int, gsyn_rows: list[tuple] = ()) -> bytes:
    sw = sf.SegmentWriter()
    tables = {t: w.payload() for t, w in sw.writers.items()}
    tables["META"] = sf.META.pack(
        tree_digest=b"\1" * 32, program_digest=b"\2" * 32,
        converter_digest=b"\3" * 32, cluster_id=cluster_id, cta_count=1,
        warp_count=1, instruction_count=1, flags=0)
    tables["STRS"] = b""
    tables["CTAS"] = sf.CTAS.pack_row(
        (cluster_id, 0, 0, 0, 0, 1, 1, 1, 0, 0xFFFFFFFF, 0, 1, 0, 0, 0,
         1))
    tables["WARP"] = sf.WARP.pack_row((cluster_id, 0, 0, 0, 0, 1))
    tables["INST"] = sf.INST.pack_row((0, 0, 0xFFFFFFFF, 0))
    tables["TMPL"] = sf.TMPL.pack_row(
        (0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0,
         0))
    if gsyn_rows:
        tables["GSYN"] = b"".join(
            _SGSYN.pack_row(r) for r in gsyn_rows)
    return sf.write_segment(tables)


def _launch(dir_: Path, seg_names: list[str], extra: dict = None) -> None:
    doc = {"device": "h100", "reference_cache_control": 2,
           "segments": seg_names, "dropped_polls": 0,
           "cluster_ids": [int(n.split(".")[0]) for n in seg_names]}
    doc.update(extra or {})
    (dir_ / "launch.json").write_text(json.dumps(doc))


def _shard(dir_: Path, rows_by_cluster: dict[int, list[tuple]]) -> None:
    seg_dir = dir_ / "segments"
    seg_dir.mkdir(parents=True)
    names = []
    for cid, rows in sorted(rows_by_cluster.items()):
        name = f"{cid:06d}.seg"
        (seg_dir / name).write_bytes(_segment(cid, rows))
        names.append(name)
    _launch(dir_, names)


def _container_gsyn(path: Path) -> list[dict]:
    reader = cf.read_container(path.read_bytes())
    table = TableReader(cf.GSYN, reader.require_table("GSYN"))
    return [table.unpack(i) for i in range(len(table))]


def test_build_collects_gsyn_per_segment(tmp_path) -> None:
    rows0 = [(3, 0, 1, 4, 3, 0xA, 7, 5, 11)]
    rows1 = [(0, 1, 3, 1, 1, 0xB, 9, 2, 13),
             (4, 2, 3, 2, 3, 0xB, 8, 9, 17)]
    _shard(tmp_path, {0: rows0, 1: rows1})
    out = cf.build_container_from_dir(tmp_path, tmp_path / "job.bin")
    rows = _container_gsyn(out)
    assert [r["segment"] for r in rows] == [0, 1, 1]
    assert [r["instance"] for r in rows] == [3, 0, 4]
    for got, (inst, lane, kind, cls, flags, addr, after, before,
              seq) in zip(rows, rows0 + rows1):
        assert (got["lane"], got["kind"], got["class_"],
                got["flags"], got["address"], got["after"],
                got["before"], got["sequence"]) == \
            (lane, kind, cls, flags, addr, after, before, seq)


def test_build_rejects_launch_gsyn_rows(tmp_path) -> None:
    _shard(tmp_path, {0: [(3, 0, 1, 4, 3, 0xA, 7, 5, 11)]})
    doc = json.loads((tmp_path / "launch.json").read_text())
    doc["gsyn_rows"] = [{"instance": 3, "lane": 0, "kind": 1,
                         "class": 4, "flags": 3, "address": 0xA,
                         "after": 7, "before": 5, "sequence": 11}]
    (tmp_path / "launch.json").write_text(json.dumps(doc))
    with pytest.raises(FileFormatError, match="pre-segment-GSYN"):
        cf.build_container_from_dir(tmp_path, tmp_path / "job.bin")


def test_merged_shards_keep_per_segment_gsyn(tmp_path) -> None:
    s0 = tmp_path / "shard0"
    s1 = tmp_path / "shard1"
    _shard(s0, {0: [(1, 0, 1, 4, 3, 0xA, 7, 5, 1)],
                2: [(2, 1, 2, 3, 1, 0xC, 3, 0, 3)]})
    _shard(s1, {1: [(5, 2, 3, 1, 1, 0xB, 9, 2, 5)],
                3: [(0, 3, 3, 2, 3, 0xD, 8, 9, 7)]})
    merged = tmp_path / "merged"
    merge_facts_shards.merge(merged, [s0, s1], segments=True)
    out = cf.build_container_from_dir(merged, merged / "job.bin")
    rows = _container_gsyn(out)
    # merged segments list is the sorted union: 000000..000003
    assert [(r["segment"], r["instance"], r["sequence"])
            for r in rows] == [(0, 1, 1), (1, 5, 5),
                               (2, 2, 3), (3, 0, 7)]


def test_build_expands_launch_geometry(tmp_path) -> None:
    _shard(tmp_path, {0: []})
    doc = json.loads((tmp_path / "launch.json").read_text())
    doc.update({"grid": [4, 2, 1], "block": [384, 1, 1],
                "cluster": [1, 2, 1]})
    (tmp_path / "launch.json").write_text(json.dumps(doc))
    out = cf.build_container_from_dir(tmp_path, tmp_path / "job.bin")
    lnch = TableReader(
        cf.LNCH, cf.read_container(out.read_bytes())
        .require_table("LNCH")).unpack(0)
    for prefix, dims in (("grid", (4, 2, 1)), ("block", (384, 1, 1)),
                         ("cluster", (1, 2, 1))):
        assert (lnch[f"{prefix}_x"], lnch[f"{prefix}_y"],
                lnch[f"{prefix}_z"]) == dims
