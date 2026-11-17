import struct, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cyclescope.convert.trace_reader import iter_records, load_static

HEAD = struct.Struct("<IHHHBBIIIII")

def _mk_case(tmp: Path):
    (tmp / "functions").mkdir()
    (tmp / "functions.jsonl").write_text(
        '{"function_id":1,"name":"f","mangled":"f"}\n')
    (tmp / "functions" / "1.tsv").write_text(
        "0\t0x0\t" + "00"*16 + "\tLDG.E R2, [R2.64] ;\tR2\t1\n"
        "1\t0x10\t" + "00"*16 + "\tSYNCS.PHASECHK R1, [R2] ;\tR1\t1\n")
    (tmp / "launches.jsonl").write_text(
        '{"launch":1,"traced":true}\n')
    recs = []
    def head(cta, warp, kind, fid, idx, am, em):
        return HEAD.pack(cta[0], cta[1], cta[2], warp, 0, kind, fid, idx,
                         am, em, warp)
    for cta in ((0,0,0), (4,0,0)):
        recs.append(head(cta, 0, 0, 1, 0, 0xffffffff, 0xffffffff))
        recs.append(struct.pack("<32Q", *range(0x1000, 0x1020)))
        recs.append(struct.pack("<32I", *([7]*32)))
    body = b"".join(recs)
    fhdr = b"GSIMTRC\x00" + struct.pack("<2I", 1, 16)
    (tmp / "trace-1.bin").write_bytes(fhdr + body)

def test_iter_records_basic(tmp_path):
    _mk_case(tmp_path)
    recs = list(iter_records(tmp_path, 1))
    assert len(recs) == 2
    assert recs[0].mrefs[0][0] == 0x1000
    assert recs[0].ops["R2"][0] == 7

def test_cluster_filter(tmp_path):
    _mk_case(tmp_path)
    recs = list(iter_records(tmp_path, 1,
                cluster_filter=lambda c: c == (0,0,0)))
    assert len(recs) == 1 and recs[0].cta == (0,0,0)


def test_gsync_records_attach_with_sequence(tmp_path):
    _mk_case(tmp_path)
    # append a third instruction with two GSYNC attachments (one lane each)
    HEADP = HEAD.pack
    body = bytearray((tmp_path / "trace-1.bin").read_bytes())
    body += HEADP(0, 0, 0, 1, 0, 0, 1, 0, 0xffffffff, 0x1, 1)
    body += struct.pack("<1Q", 0x2000) + struct.pack("<1I", 9)
    for phase in (1, 2):
        body += HEADP(0, 0, 0, 1, 0, 2, 1, 0, 0xffffffff, 0x1, 1)
        body += struct.pack("<4B", phase, 1, 0, 0) + struct.pack("<1I", 5)
    (tmp_path / "trace-1.bin").write_bytes(bytes(body))
    recs = list(iter_records(tmp_path, 1))
    assert len(recs) == 3
    g = recs[-1].gsync
    assert [x[0] for x in g] == [1, 2]
    assert [x[4] for x in g] == [0, 1]       # launch-wide sequence


def test_cluster_filter_skips_attachments_of_filtered_records(tmp_path):
    _mk_case(tmp_path)
    body = bytearray((tmp_path / "trace-1.bin").read_bytes())
    # AFTER + GSYNC attached to the filtered-out cluster (4,0,0) record
    body += HEAD.pack(4, 0, 0, 0, 0, 1, 1, 0, 0xffffffff, 0xffffffff, 0)
    body += struct.pack("<I", 1)
    body += HEAD.pack(4, 0, 0, 0, 0, 2, 1, 0, 0xffffffff, 0x1, 0)
    body += struct.pack("<4B", 1, 1, 0, 0) + struct.pack("<1I", 5)
    # and a GSYNC on the kept cluster afterwards: sequence counts both
    body += HEAD.pack(0, 0, 0, 0, 0, 2, 1, 0, 0xffffffff, 0x1, 0)
    body += struct.pack("<4B", 2, 1, 0, 0) + struct.pack("<1I", 6)
    (tmp_path / "trace-1.bin").write_bytes(bytes(body))
    recs = list(iter_records(tmp_path, 1,
                cluster_filter=lambda c: c == (0, 0, 0)))
    assert len(recs) == 1 and recs[0].cta == (0, 0, 0)
    assert recs[0].gsync and recs[0].gsync[0][4] == 1

def test_cta_info_records(tmp_path):
    _mk_case(tmp_path)
    # prepend a kind-3 CTA record (per warp, duplicates allowed)
    body = bytearray((tmp_path / "trace-1.bin").read_bytes())
    hdr, rest = bytes(body[:16]), bytes(body[16:])
    cta_body = struct.pack("<10I", 3, 8, 1, 1, 7, 1, 1, 18, 1, 1)
    ins = HEAD.pack(4, 0, 0, 0, 0, 3, 1, 0, 0xffffffff, 0xffffffff, 0) + \
        cta_body + \
        HEAD.pack(4, 0, 0, 1, 0, 3, 1, 0, 0xffffffff, 0xffffffff, 1) + \
        cta_body
    (tmp_path / "trace-1.bin").write_bytes(hdr + ins + rest)
    info = {}
    recs = list(iter_records(tmp_path, 1, cta_info=info))
    assert len(recs) == 2                       # CTA records never yielded
    assert info[(4, 0, 0)]["cluster_ctarank"] == 3
    assert info[(4, 0, 0)]["cluster_nctaid"] == [8, 1, 1]
    assert info[(4, 0, 0)]["clusterid"] == [7, 1, 1]
    assert info[(4, 0, 0)]["nclusterid"] == [18, 1, 1]

def test_cta_info_mixed_shape_keys(tmp_path):
    _mk_case(tmp_path)
    body = bytearray((tmp_path / "trace-1.bin").read_bytes())
    hdr, rest = bytes(body[:16]), bytes(body[16:])
    ins = b""
    # CTA (0,0,0): 2-CTA cluster shape, clusterid (17,0,0) of (36,2,1)
    # CTA (4,0,0): 8-CTA cluster shape, clusterid (17,0,0) of (18,1,1)
    # bare linear ids collide (17 vs 17) -> shape must be in the key
    for cta, ctar, nct, cid, ncl in (
            ((0,0,0), 0, (2,1,1), (17,0,0), (36,2,1)),
            ((4,0,0), 3, (4,2,1), (17,0,0), (18,1,1))):
        ins += HEAD.pack(*cta, 0, 0, 3, 1, 0, 0xffffffff, 0xffffffff, 0)
        ins += struct.pack("<10I", ctar, *nct, *cid, *ncl)
    (tmp_path / "trace-1.bin").write_bytes(hdr + ins + rest)
    info = {}
    list(iter_records(tmp_path, 1, cta_info=info))
    k0 = (tuple(info[(0,0,0)]["cluster_nctaid"]),
          info[(0,0,0)]["clusterid"][0] + info[(0,0,0)]["clusterid"][1]*36)
    k4 = (tuple(info[(4,0,0)]["cluster_nctaid"]),
          info[(4,0,0)]["clusterid"][0] + info[(4,0,0)]["clusterid"][1]*18)
    assert k0 != k4 and k0[1] == k4[1] == 17   # same lin, different shape

def test_ncu_occupancy_override(tmp_path):
    from cyclescope.convert.capture_pipeline import _ncu_occupancy
    (tmp_path / "ncu.csv").write_text(
        "==WARNING== junk\n"
        '"ID","Process ID","Process Name","Host Name","Kernel Name",'
        '"launch__block_size","launch__occupancy_limit_registers",'
        '"launch__occupancy_limit_shared_mem","launch__occupancy_limit_blocks",'
        '"launch__occupancy_limit_warps"\n'
        '"1","1","p","h","other_kernel","128","16","32","32","16"\n'
        '"2","1","p","h","kern_a","128","3","8","32","16"\n'
        '"3","1","p","h","kern_a","256","2","9","32","16"\n'
        '"4","1","p","h","kern_a","128","5","7","32","16"\n')
    row = {"mangled": "kern_a", "block": [128, 1, 1],
           "function_prior_launches": 0}
    assert _ncu_occupancy(tmp_path, row) == 3
    row2 = dict(row, function_prior_launches=2)
    assert _ncu_occupancy(tmp_path, row2) == 5
    row3 = dict(row, block=[256, 1, 1], function_prior_launches=0)
    assert _ncu_occupancy(tmp_path, row3) is None
    # `cyclescope run` layout: no census in the case dir, limits come from the
    # reference run's ncu csv next to the capture
    case = tmp_path / "caps" / "kern_case"
    ref = tmp_path / "caps" / "reference" / "kern_case" / "kern_case"
    case.mkdir(parents=True)
    ref.mkdir(parents=True)
    assert _ncu_occupancy(case, row) is None
    (ref / "ncu_r0.csv").write_text((tmp_path / "ncu.csv").read_text())
    assert _ncu_occupancy(case, row) == 3
