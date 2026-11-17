"""cyclescope-trace-split end-to-end: per-shard streams must yield the same
DynRecord sequence (after_ready + launch-wide gsync seq included) as
cluster_filter-ing the original stream."""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cyclescope.convert.trace_reader import iter_records

HEAD = struct.Struct("<IHHHBBIIIII")


def _split_bin():
    """cyclescope-trace-split shells out to the zstd binary: both must be
    on PATH, otherwise the test is skipped."""
    bin_ = shutil.which("cyclescope-trace-split")
    if bin_ is None or shutil.which("zstd") is None:
        pytest.skip("cyclescope-trace-split / zstd not in PATH")
    return bin_


def _mk_case(tmp: Path):
    (tmp / "functions").mkdir()
    (tmp / "functions.jsonl").write_text(
        '{"function_id":1,"name":"f","mangled":"f"}\n')
    (tmp / "functions" / "1.tsv").write_text(
        "0\t0x0\t" + "00" * 16 + "\tLDG.E R2, [R2.64] ;\tR2\t1\n"
        "1\t0x10\t" + "00" * 16 + "\tSYNCS.PHASECHK R1, [R2] ;\tR1\t1\n")
    (tmp / "launches.jsonl").write_text(
        '{"launch":1,"traced":true,"grid":[8,1,1],"cluster":[4,1,1]}\n')
    recs = []

    def head(cta, warp, kind, fid, idx, am, em):
        return HEAD.pack(cta[0], cta[1], cta[2], warp, 0, kind, fid, idx,
                         am, em, warp)
    # cluster 0 (cta 0,0,0): instr + AFTER + GSYNC
    recs.append(head((0, 0, 0), 0, 0, 1, 0, 0xffffffff, 0xffffffff))
    recs.append(struct.pack("<32Q", *range(0x1000, 0x1020)))
    recs.append(struct.pack("<32I", *([7] * 32)))
    recs.append(head((0, 0, 0), 0, 1, 1, 0, 0xffffffff, 0xffffffff))
    recs.append(struct.pack("<I", 123))
    recs.append(head((0, 0, 0), 0, 2, 1, 0, 0xffffffff, 0x1))
    recs.append(struct.pack("<4B", 1, 1, 0, 0) + struct.pack("<1I", 5))
    # cluster 1 (cta 4,0,0): instr + GSYNC (launch-wide seq = 1)
    recs.append(head((4, 0, 0), 0, 0, 1, 0, 0xffffffff, 0xffffffff))
    recs.append(struct.pack("<32Q", *range(0x2000, 0x2020)))
    recs.append(struct.pack("<32I", *([9] * 32)))
    recs.append(head((4, 0, 0), 0, 2, 1, 0, 0xffffffff, 0x1))
    recs.append(struct.pack("<4B", 2, 1, 0, 0) + struct.pack("<1I", 6))
    body = b"".join(recs)
    fhdr = b"GSIMTRC\x00" + struct.pack("<2I", 1, 16)
    (tmp / "trace-1.bin").write_bytes(fhdr + body)


def _gsync_iters(path: Path):
    data = path.read_bytes()
    return iter(int.from_bytes(data[i:i + 8], "little")
                for i in range(0, len(data), 8))


def _rec_key(r):
    return (r.cta, r.warp, r.function_id, r.idx, r.active_mask,
            r.exec_mask, r.after_ready,
            [tuple(g[4] for g in r.gsync)],
            [tuple(m) for m in r.mrefs], dict(r.ops))


def test_split_shards_match_filtered(tmp_path):
    bin_ = _split_bin()
    _mk_case(tmp_path)
    out = tmp_path / "split"
    r = subprocess.run(
        [bin_, "--trace", str(tmp_path / "trace-1.bin"),
         "--functions", str(tmp_path / "functions"),
         "--grid", "8,1,1", "--cluster", "4,1,1",
         "--shards", "2", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    for k in range(2):
        base = out / f"trace-1.shard{k}of2"
        zst = Path(f"{base}.bin.zst")
        seq = Path(f"{base}.gsync_seq")
        assert zst.is_file() and seq.is_file()
        got = [_rec_key(x) for x in iter_records(
            tmp_path, 1, trace_path=zst,
            gsync_sequences=_gsync_iters(seq))]
        want = [_rec_key(x) for x in iter_records(
            tmp_path, 1,
            cluster_filter=lambda c, k=k: (c[0] // 4) % 2 == k)]
        assert got == want, f"shard {k} records differ"
    # shard1's GSYNC carries launch-wide seq 1 (not a local 0)
    seq_data = (out / "trace-1.shard1of2.gsync_seq").read_bytes()
    assert int.from_bytes(seq_data[:8], "little") == 1


def _split_one(tmp_path):
    bin_ = _split_bin()
    _mk_case(tmp_path)
    out = tmp_path / "split"
    r = subprocess.run(
        [bin_, "--trace", str(tmp_path / "trace-1.bin"),
         "--functions", str(tmp_path / "functions"),
         "--grid", "8,1,1", "--cluster", "4,1,1",
         "--shards", "2", "--out", str(out)])
    assert r.returncode == 0
    return out


def test_gsync_seq_short_raises(tmp_path):
    out = _split_one(tmp_path)
    zst = out / "trace-1.shard1of2.bin.zst"   # has 1 GSYNC
    with pytest.raises(ValueError, match="gsync_seq"):
        list(iter_records(tmp_path, 1, trace_path=zst,
                          gsync_sequences=iter([])))


def test_gsync_seq_long_raises(tmp_path):
    out = _split_one(tmp_path)
    zst = out / "trace-1.shard0of2.bin.zst"   # has 1 GSYNC
    with pytest.raises(ValueError, match="gsync_seq"):
        list(iter_records(tmp_path, 1, trace_path=zst,
                          gsync_sequences=iter([0, 5, 9])))


def test_missing_seq_file_fails(tmp_path):
    from cyclescope.convert.capture_pipeline import convert_capture
    _mk_case(tmp_path)
    out = tmp_path / "split"
    (out).mkdir()
    (out / "trace-1.shard0of2.bin.zst").write_bytes(b"GSIMTRC\x00")
    with pytest.raises(SystemExit, match="split_gsync_sequence"):
        convert_capture(tmp_path, "hopper", tmp_path / "o",
                        shard=(0, 2), split_dir=out, cache=False)


def test_split_kind3_zst_probe_stops_early(tmp_path):
    """A compressed trace whose first record is a kind-3 CTA record: the
    probe pass stops at it (leaving zstd mid-stream) and the split must
    still succeed; records route by the kind-3 cluster id."""
    bin_ = _split_bin()
    _mk_case(tmp_path)
    body = (tmp_path / "trace-1.bin").read_bytes()
    fhdr, recs = body[:16], body[16:]
    # kind-3 for cta (4,0,0): rank 0, nctaid 4,1,1, clusterid 1,0,0,
    # nclusterid 2,1,1 -> cluster 1 (same as the attribute path)
    cta3 = HEAD.pack(4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0) + struct.pack(
        "<10I", 0, 4, 1, 1, 1, 0, 0, 2, 1, 1)
    # pad the stream so zstd still has output pending when the probe
    # stops at the first record
    filler = b"".join(
        HEAD.pack(0, 0, 0, 0, 0, 1, 1, 0, 0xffffffff, 0xffffffff, 0) +
        struct.pack("<I", i) for i in range(20000))
    raw = tmp_path / "trace-1.bin"
    raw.write_bytes(fhdr + cta3 + recs + filler)
    subprocess.run(["zstd", "-q", "-f", str(raw), "-o",
                    str(tmp_path / "trace-1.bin.zst")], check=True)
    raw.unlink()
    out = tmp_path / "split"
    r = subprocess.run(
        [bin_, "--trace", str(tmp_path / "trace-1.bin.zst"),
         "--functions", str(tmp_path / "functions"),
         "--grid", "8,1,1", "--cluster", "4,1,1",
         "--shards", "2", "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "total: 20006 records, 2 gsync" in r.stdout
    got = [_rec_key(x) for x in iter_records(
        tmp_path, 1, trace_path=out / "trace-1.shard1of2.bin.zst",
        gsync_sequences=_gsync_iters(out / "trace-1.shard1of2.gsync_seq"))]
    assert len(got) == 1 and got[0][0] == (4, 0, 0)
    # cta (0,0,0) has no kind-3 record: its records are buffered to EOF
    # and must keep their launch-wide gsync ordinal (0); (4,0,0)'s is 1
    assert list(_gsync_iters(out / "trace-1.shard0of2.gsync_seq")) == [0]
    assert list(_gsync_iters(out / "trace-1.shard1of2.gsync_seq")) == [1]


def test_split_bad_layout_fails(tmp_path):
    bin_ = _split_bin()
    _mk_case(tmp_path)
    body = bytearray((tmp_path / "trace-1.bin").read_bytes())
    body += HEAD.pack(0, 0, 0, 0, 0, 0, 1, 99, 0xffffffff, 0xffffffff, 0)
    body += struct.pack("<32Q", *range(32)) + struct.pack("<32I", *range(32))
    (tmp_path / "trace-1.bin").write_bytes(bytes(body))
    r = subprocess.run(
        [bin_, "--trace", str(tmp_path / "trace-1.bin"),
         "--functions", str(tmp_path / "functions"),
         "--grid", "8,1,1", "--cluster", "4,1,1",
         "--shards", "2", "--out", str(tmp_path / "sp")])
    assert r.returncode != 0
