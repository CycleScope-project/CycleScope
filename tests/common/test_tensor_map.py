"""Tensor map decode: packed sub-byte (NVF4) descriptors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
def test_packed_nvf4_map_rescales_innermost_dimension_to_bytes():
    from cyclescope.convert import tensor_map as tm
    enc = {"api": "tiled", "data_type": 13, "rank": 3,
           "global_dim": [832, 128, 1], "global_strides": [416, 0],
           "box_dim": [256, 32, 1], "element_strides": [1, 1, 1],
           "global_address": 0x10000}
    layout = tm._from_encode(enc)
    assert layout["element_bytes"] == 1
    assert layout["box_dim"] == [128, 32, 1]        # 256 x 4 bit = 128 B
    assert layout["global_dim"] == [416, 128, 1]    # matches stride 416
    model, facts = tm.tensor_map_entry({"encode": enc}, "UTMALDG.3D.2CTA",
                                       "t")
    assert model.element_bytes == 1 and list(model.box_dim) == [128, 32, 1]


def test_packed_u4_align16b_counts_four_bits_per_value():
    from cyclescope.convert import tensor_map as tm
    enc = {"api": "tiled", "data_type": 14, "rank": 2,
           "global_dim": [1024, 128], "global_strides": [512],
           "box_dim": [128, 128], "element_strides": [1, 1]}
    layout = tm._from_encode(enc)
    # 128 values x 4 bit = 64 B per row; 128 rows -> 8192 B total
    assert layout["element_bytes"] == 1
    assert layout["box_dim"] == [64, 128]
    assert layout["global_dim"] == [512, 128]


def test_decode_tiled_code11_packed_4bit():
    from cyclescope.convert import tensor_map as tm
    # code 11 (packed 4-bit): 256x128 box -> 16384 B total (4 bit/value)
    d = 0
    d |= 0x400  # rank bits? control at >>64: build full word
    ctrl = 0
    ctrl |= (5 - 1) << 4   # rank 5? use rank 2 instead
    ctrl = ((2 - 1) << 4) | (11 << 7)
    d = ctrl << 64
    d |= 0x1000  # global addr
    d |= (256 - 1) << 256      # global_dim[0] = 256
    d |= (128 - 1) << (256 + 32)  # gdim1 = 128
    d |= (255) << 440          # box_dim[0] = 256
    d |= (127) << 448          # box_dim[1] = 128
    d |= (128 >> 4) << 96      # stride0 = 128 <<4? stride = raw<<4 -> raw=8? use raw=8 ->128
    r = tm.decode_tiled(d.to_bytes(0x40, "little").hex())
    assert r is not None
    assert r["element_bytes"] == 1
    assert r["box_dim"][0] == 128   # 256 values x 4 bit = 128 B
    assert r["global_dim"][0] == 128


def test_encode_dtype11_tf32_stays_unpacked():
    from cyclescope.convert import tensor_map as tm
    enc = {"api": "tiled", "data_type": 11, "rank": 2,
           "global_dim": [256, 128], "global_strides": [1024],
           "box_dim": [64, 32], "element_strides": [1, 1]}
    layout = tm._from_encode(enc)
    assert layout["element_bytes"] == 4
    assert layout["box_dim"] == [64, 32]
    assert layout["global_dim"] == [256, 128]


_IM2COL_GEO = {"mode": "im2col", "rank": 4, "global_address": 0x10000,
               "element_bytes": 2, "global_dim": [8, 4, 2, 1],
               "global_strides": [16, 64, 128],
               "element_strides": [1, 1, 1, 1], "cpp": 8, "ppc": 8,
               "lower": [-1, -1], "upper": [1, 1]}
_NEG1 = 0xFFFFFFFF


def test_im2col_box_first_tile_single_pixel():
    from cyclescope.convert import tensor_map as tm
    # C=8x2B=16B/px, W=4,H=2,N=1; lo=up=[-1,-1]; start (0,-1,-1,0).
    # Row h=-1 all OOB, (w=-1,h=0) OOB, (w=0,h=0) -> one 16 B pixel.
    sec = tm.im2col_box_sectors(_IM2COL_GEO, [0, _NEG1, _NEG1, 0], [0, 0])
    assert sec == [0x10000]


def test_im2col_box_with_offsets():
    from cyclescope.convert import tensor_map as tm
    # offsets (1,1): accessed = base + 1 -> row h=-1 accesses h=0
    # (4 in-bounds w) plus row h=0 accesses h=1 (4 more) = 3 sectors
    sec = tm.im2col_box_sectors(_IM2COL_GEO, [0, _NEG1, _NEG1, 0], [1, 1])
    assert sec == [0x10000, 0x10020, 0x10040]


def test_im2col_box_n_overflow_no_memory():
    from cyclescope.convert import tensor_map as tm
    # N=1: pixels whose linear index crosses into n>=1 touch nothing
    sec = tm.im2col_box_sectors(_IM2COL_GEO, [0, 0, 0, 0], [0, 0])
    # base w=0,h=0 row: pixels at (0..4,h=0) then overflow into n=1
    assert all(s < 0x10000 + 5 * 16 + 32 for s in sec)
    assert len(sec) > 0


def test_im2col_offsets_unpack_5d():
    from cyclescope.convert import tensor_map as tm
    word = 1 | (2 << 5) | (3 << 10)
    assert tm._im2col_offsets(5, word) == [1, 2, 3]
    assert tm._im2col_offsets(4, 0x0302) == [2, 3]
    assert tm._im2col_offsets(3, 0x0003) == [3]


def test_im2col_store_form_zero_offsets():
    from cyclescope.convert import tensor_map as tm
    sec = tm.im2col_box_sectors(_IM2COL_GEO, [0, _NEG1, _NEG1, 0],
                                [0] * (_IM2COL_GEO["rank"] - 2))
    assert sec == [0x10000]


def test_im2col_entry_requires_corners():
    import pytest
    from cyclescope.convert import tensor_map as tm
    from cyclescope.convert.sass_decode import MissingFact
    enc = {"api": "im2col", "data_type": 1, "rank": 4,
           "global_dim": [8, 4, 2, 1], "global_strides": [16, 64, 128],
           "box_dim": [8, 8], "element_strides": [1, 1, 1, 1],
           "global_address": 0x10000}
    with pytest.raises(MissingFact):
        tm.tensor_map_entry({"encode": enc}, "UTMALDG.4D.IM2COL", "t")
    enc["pixel_box_lower_corner"] = [-1, -1]
    enc["pixel_box_upper_corner"] = [1, 1]
    tmobj, facts = tm.tensor_map_entry(
        {"encode": enc}, "UTMALDG.4D.IM2COL", "t")
    assert facts["api"] == "im2col"
    assert facts["shared_bytes"] == 8 * 8 * 2
    assert facts["geometry"]["mode"] == "im2col"
    assert tmobj.box_dim == (8, 8)
    assert tmobj.element_bytes == 2


def test_im2col_entry_unknown_dtype_rejected():
    import pytest
    from cyclescope.convert import tensor_map as tm
    from cyclescope.convert.sass_decode import MissingFact
    enc = {"api": "im2col", "data_type": 99, "rank": 4,
           "global_dim": [8, 4, 2, 1], "global_strides": [16, 64, 128],
           "box_dim": [8, 8], "element_strides": [1, 1, 1, 1],
           "global_address": 0x10000,
           "pixel_box_lower_corner": [-1, -1],
           "pixel_box_upper_corner": [1, 1]}
    with pytest.raises(MissingFact):
        tm.tensor_map_entry({"encode": enc}, "UTMALDG.4D.IM2COL", "t")


# fa3_fwd_hdim_small_09 K map (tensor_maps.jsonl id 2): the encode record
# written by the tracer's cuTensorMapEncodeTiled hook carries no base
# pointer; the image's first 8 bytes do (0x7ef746000000)
_FA3_K_IMAGE = ("00000046f77e0000306504001000000010000000008000000000000000000000"
                "7f000000ff07000000000000ff000000000000000000003f7f00000000000000"
                "0040000000000000000400000000000000000000000000000000000000000000"
                "0000000000000000000000000000000000000000000000000000000000000000")
_FA3_K_ENCODE = {"api": "tiled", "data_type": 9, "rank": 4,
                 "global_dim": [128, 2048, 1, 256],
                 "global_strides": [256, 256, 524288],
                 "box_dim": [64, 128, 1, 1], "element_strides": [1, 1, 1, 1],
                 "interleave": 0, "swizzle": 3, "l2_promotion": 2,
                 "oob_fill": 0}


def test_tiled_entry_takes_base_address_from_image_when_encode_lacks_it():
    from cyclescope.convert import tensor_map as tm
    assert tm.decode_tiled(_FA3_K_IMAGE)["global_address"] == 0x7ef746000000
    tmobj, facts = tm.tensor_map_entry(
        {"image": _FA3_K_IMAGE, "encode": dict(_FA3_K_ENCODE)},
        "UTMALDG.4D", "t")
    assert tmobj.global_address == 0x7ef746000000
    geo = facts["geometry"]
    assert geo["global_address"] == 0x7ef746000000
    # box at (64, 128, 0, 3): second column half of K block 1 of batch 3
    sectors = tm.tma_box_sectors(geo, [64, 128, 0, 3])
    assert len(sectors) == 512
    assert sectors[0] == 0x7ef746000000 + 3 * 524288 + 128 * 256 + 128
    # two batches / the two column halves never share a sector
    other = tm.tma_box_sectors(geo, [0, 128, 0, 3])
    assert not set(sectors) & set(other)
    assert not set(sectors) & set(tm.tma_box_sectors(geo, [64, 128, 0, 4]))


def test_tiled_entry_without_any_base_address_is_rejected():
    import pytest
    from cyclescope.convert import tensor_map as tm
    from cyclescope.convert.sass_decode import MissingFact
    with pytest.raises(MissingFact, match="tensor_map_global_address"):
        tm.tensor_map_entry({"encode": dict(_FA3_K_ENCODE)}, "UTMALDG.4D",
                            "t")
    # an explicit encode address still wins over the image
    enc = dict(_FA3_K_ENCODE, global_address=0x20000)
    tmobj, facts = tm.tensor_map_entry(
        {"image": _FA3_K_IMAGE, "encode": enc}, "UTMALDG.4D", "t")
    assert tmobj.global_address == 0x20000
    assert facts["geometry"]["global_address"] == 0x20000


# CUF-C72C-MXF8-R05 B map (tensor_maps.jsonl id 2): dtype 14
# (16U4_ALIGN16B) -- the image's element code 12 is not in the layout
# decoder's table, so decode_tiled is None, yet bytes 0..7 hold the base
_MXF8_U4_IMAGE = ("00c06479fc7e0000206624000400000000000000000000000000000000000000"
                  "7f0000007f0900000000000000000000000000000000007f7f00000000000000"
                  "0000000000000000000400000000000000000000000000000000000000000000"
                  "0000000000000000000000000000000000000000000000000000000000000000")
_MXF8_U4_ENCODE = {"api": "tiled", "data_type": 14, "rank": 3, "global_dim": [128, 2432, 1], "global_strides": [64, 0], "box_dim": [128, 128, 1], "element_strides": [1, 1, 1], "interleave": 0, "swizzle": 3, "l2_promotion": 2, "oob_fill": 0}


def test_tiled_entry_takes_image_address_when_layout_is_undecodable():
    from cyclescope.convert import tensor_map as tm
    assert tm.decode_tiled(_MXF8_U4_IMAGE) is None
    assert tm.image_global_address(_MXF8_U4_IMAGE) == 0x7efc7964c000
    tmobj, facts = tm.tensor_map_entry(
        {"image": _MXF8_U4_IMAGE, "encode": dict(_MXF8_U4_ENCODE)},
        "UTMALDG.3D", "t")
    assert tmobj.global_address == 0x7efc7964c000
    assert facts["geometry"]["global_address"] == 0x7efc7964c000
    assert tmobj.element_bytes == 1 and tmobj.box_dim[0] == 64  # 128 x 4 bit
    assert tm.image_global_address("zz") is None
    assert tm.image_global_address("00ff") is None


def test_bulk_global_address_from_g_bracket():
    from cyclescope.convert.operand_facts import bulk_global_address
    uni = {"UR4": 0x45000000, "UR5": 0x7efc, "UR6": 0x1000, "UR7": 0x0,
           "UR8": 1024, "UR14": 0x46000000, "UR15": 0x7efc}
    # UBLKRED.G.S: first bracket is the global destination
    assert bulk_global_address("UBLKRED.G.S.ADD.F32.RN",
                               ["[UR4]", "[UR6]", "UR8"], uni) == 0x7efc45000000
    # UBLKCP.S.G: second bracket is the global source
    assert bulk_global_address("UBLKCP.S.G", ["[UR4]", "[UR14]", "UR6"],
                               uni) == 0x7efc46000000
    # shared -> shared / missing pair -> None
    assert bulk_global_address("UBLKCP.S.S", ["[UR4]", "[UR6]", "UR8"], uni) is None
    assert bulk_global_address("UBLKRED.G.S.ADD.F32.RN", ["[UR20]", "[UR6]", "UR8"],
                               uni) is None


def test_bulk_shared_to_shared_has_no_global_side():
    from cyclescope.convert.operand_facts import bulk_has_global_side
    assert bulk_has_global_side("UBLKRED.G.S.ADD.F32.RN")
    assert bulk_has_global_side("UBLKCP.S.G")
    assert not bulk_has_global_side("UBLKCP.S.S")


def test_tiled_cut_rows():
    from cyclescope.convert import tensor_map as tm
    geo = {"element_bytes": 2, "rank": 2, "box": [64, 128],
           "global_dim": [64, 4096], "global_strides": [128],
           "element_strides": [1, 1], "global_address": 0}
    # fully in bounds along dim 0 -> 0
    assert tm.tiled_cut_rows(geo, [0, 0]) == 0
    # box cut in half along dim 0 -> the 128 in-bounds rows count
    assert tm.tiled_cut_rows(geo, [32, 0]) == 128
    assert tm.tiled_cut_rows(geo, [48, 0]) == 128
    # fully outside along dim 0 -> 0 (not modelled)
    assert tm.tiled_cut_rows(geo, [64, 0]) == 0
    # whole rows outside along dim 1 -> 0 (nothing cut along dim 0)
    assert tm.tiled_cut_rows(geo, [0, 4096]) == 0


def test_tiled_cut_rows_rank3_rows_outside():
    from cyclescope.convert import tensor_map as tm
    # rank-3 map {64, R rows, P planes}, box {64, box_rows, 1}: rows
    # beyond R are whole-row out of bounds -> cut_rows stays 0 even
    # though fewer than box rows land in bounds
    geo = {"element_bytes": 2, "rank": 3, "box": [64, 128, 1],
           "global_dim": [64, 96, 64],
           "global_strides": [128, 12288],
           "element_strides": [1, 1, 1], "global_address": 0}
    assert tm.tiled_cut_rows(geo, [0, 0, 0]) == 0
    assert tm.tiled_cut_rows(geo, [0, 0, 40]) == 0
    # cut along dim 0 in the same rank-3 map -> the in-bounds rows
    # of the plane count (96 of the 128 box rows exist)
    assert tm.tiled_cut_rows(geo, [32, 0, 0]) == 96
