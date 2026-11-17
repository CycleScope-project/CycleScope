"""tensor_maps.jsonl -> model.TensorMap + TMA box sectors/bytes.

The 128 B CUtensorMap image is decoded with the bit layout fitted and
verified on the predecessor's randomized encode corpus (hopper
``tma_descriptor.decode_tiled``):

  byte 0..7   global address (u64)
  byte 8..11  control: bit0 im2col mode, bits4-6 rank-1, bits7-10 element
              code, bits11-12 interleave, bits13-14 swizzle, bit15 OOB
              fill NaN, bit16 tfloat32, bits17-18 L2 promotion
  byte 12+4i  low 32 bits of global_strides[i] >> 4 (i < rank-1)
  bit 224+4i  high 4 bits of global_strides[i] >> 4
  byte 32+4i  global_dim[i] - 1 (u32)
  bit 416+3i  element_strides[i] - 1 (3 bits)
  byte 55+i   box_dim[i] - 1 (u8)

Host encode arguments (``encode``) take precedence when present; a
decodable image that disagrees with them is ``missing_fact:
tensor_map_decode``.  Box -> 32 B sectors follows the in-bounds
rectangle rule (out-of-bounds elements touch no memory).
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

from cyclescope.convert import model
from cyclescope.convert.sass_decode import MissingFact, TENSOR_DTYPE_BYTES

_SECTOR = 32

# hardware element code -> element bytes (fitted; tfloat32 shares the
# float32 codes and is distinguished by control bit 16)
ELEMENT_BYTES = {0: 1, 1: 2, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 4,
                 9: 8, 10: 2}


def load_tensor_maps(case_dir) -> dict[int, dict]:
    path = Path(case_dir) / "tensor_maps.jsonl"
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[int(row["id"])] = {"image": row.get("image") or "",
                               "encode": row.get("encode") or {}}
    return out


def image_global_address(image_hex: str) -> int | None:
    """Base pointer from the image's first 8 bytes; None when the image
    is missing / short / not hex.  Independent of the layout decode: an
    image whose element code the decoder does not know (e.g. hardware
    code 12 = 16U4_ALIGN16B) still carries its address here."""
    try:
        raw = bytes.fromhex(image_hex or "")
    except ValueError:
        return None
    if len(raw) < 8:
        return None
    return int.from_bytes(raw[:8], "little")


def decode_tiled(image_hex: str) -> dict | None:
    """Decode a tiled map image; None when not decodable (bad hex, short,
    unknown element code, rank > 5); im2col maps return {"mode": 1}."""
    try:
        raw = bytes.fromhex(image_hex)
    except ValueError:
        return None
    if len(raw) < 0x40:
        return None
    d = int.from_bytes(raw, "little")
    control = (d >> 64) & 0xFFFFFFFF
    if control & 1:
        return {"mode": 1, "global_address": d & 0xFFFFFFFFFFFFFFFF}
    rank = ((control >> 4) & 7) + 1
    code = (control >> 7) & 0xF
    packed = ELEMENT_PACKED_BITS.get(code)
    element = 1 if packed is not None else ELEMENT_BYTES.get(code)
    if element is None or rank > 5:
        return None
    strides = []
    for i in range(rank - 1):
        low = (d >> (96 + 32 * i)) & 0xFFFFFFFF
        high = (d >> (224 + 4 * i)) & 0xF
        strides.append(((high << 32) | low) << 4)
    box = [((d >> (440 + 8 * i)) & 0xFF) + 1 for i in range(rank)]
    gdim = [((d >> (256 + 32 * i)) & 0xFFFFFFFF) + 1 for i in range(rank)]
    if packed is not None:
        logical, physical = packed
        if (box[0] * physical) % 8 or (gdim[0] * logical) % 8:
            return None
        box[0] = box[0] * physical // 8
        gdim[0] = gdim[0] * logical // 8
    return {
        "mode": 0,
        "global_address": d & 0xFFFFFFFFFFFFFFFF,
        "rank": rank,
        "element_bytes": element,
        "interleave": (control >> 11) & 3,
        "swizzle": (control >> 13) & 3,
        "l2_promotion": (control >> 17) & 3,
        "global_strides": strides,
        "global_dim": gdim,
        "element_strides": [((d >> (416 + 3 * i)) & 7) + 1
                            for i in range(rank)],
        "box_dim": box,
    }


# packed sub-byte CUtensorMapDataType codes -> (logical bits per value,
# physical shared bits per value): 13 16U4_ALIGN8B and 14 16U4_ALIGN16B
# are both 4 bits per value (16B only widens the address alignment);
# 15 16U6_ALIGN16B packs 6 bits into 8.  They are carried as 1-byte
# elements with the innermost dimension rescaled to bytes (box:
# physical, global: logical), so every downstream byte count stays
# exact.  Evidence DG3-FP8FP4-*: expect_tx matches box 128x128 dtype-14
# tile at 8192 B = 4 bit/value, not 16384 B.
PACKED_DTYPE_BITS = {13: (4, 4), 14: (4, 4), 15: (6, 8)}

# hardware element codes (image control word, decode_tiled)
# that are packed sub-byte: code 11 = 4 bit/value.  Evidence
# CU3-C75-GRP-BS: expect_tx 3x16384+3x2048 for three 256x128
# boxes + three smaller ones -> 256x128 at 16384 B = 4
# bit/value, matching the encode path PACKED_DTYPE_BITS.
ELEMENT_PACKED_BITS = {11: (4, 4)}


def _from_encode(encode: dict) -> dict | None:
    if not encode or not encode.get("box_dim"):
        return None
    dtype = encode.get("data_type")
    dtype_bytes = TENSOR_DTYPE_BYTES.get(dtype)
    packed = PACKED_DTYPE_BITS.get(dtype)
    if packed is None and (dtype_bytes is None
                           or dtype_bytes != int(dtype_bytes)):
        return None
    rank = int(encode.get("rank") or len(encode["box_dim"]))
    box = [int(v) for v in encode["box_dim"][:rank]]
    gdim = [int(v) for v in (encode.get("global_dim") or [])[:rank]]
    if packed is not None:
        logical, physical = packed
        if (box[0] * physical) % 8 or (gdim and (gdim[0] * logical) % 8):
            return None
        box[0] = box[0] * physical // 8
        if gdim:
            gdim[0] = gdim[0] * logical // 8
        dtype_bytes = 1
    # the tracer's encode hook does not record the base pointer: the
    # address lives only in the 128 B image (tensor_map_entry falls
    # back to the decoded image; neither -> missing_fact)
    address = encode.get("global_address")
    return {
        "mode": 1 if encode.get("api") == "im2col" else 0,
        "global_address": int(address) if address else None,
        "rank": rank,
        "element_bytes": int(dtype_bytes),
        "global_strides": [int(v) for v in
                           (encode.get("global_strides") or [])[:rank - 1]],
        "global_dim": gdim,
        "element_strides": [int(v) or 1 for v in
                            (encode.get("element_strides") or [1] * rank)[:rank]],
        "box_dim": box,
    }


def tensor_map_entry(entry: dict, opcode: str, where: str) -> tuple:
    """(model.TensorMap, facts) for one descriptor use.  The opcode's
    ``.ND``/``.IM2COL`` modifiers must agree with the map."""
    upper = opcode.upper()
    m = re.search(r"\.(\d)D\b", upper)
    if m is None:
        raise MissingFact(where, f"tma rank from opcode {opcode}")
    rank = int(m.group(1))
    api = "im2col" if ".IM2COL" in upper else "tiled"
    decoded = decode_tiled(entry.get("image") or "")
    encoded = _from_encode(entry.get("encode") or {})
    layout = encoded or decoded
    if layout is None:
        raise MissingFact(where, "tensor map layout")
    if decoded is not None and encoded is not None and \
            decoded.get("mode") == 0 and (
                decoded["rank"] != encoded["rank"]
                or decoded["box_dim"] != encoded["box_dim"]
                or decoded["global_dim"] != encoded["global_dim"]):
        raise MissingFact(where, "tensor_map_decode")
    if layout.get("mode") == 1 or api == "im2col":
        enc = entry.get("encode") or {}
        if enc.get("api") != "im2col" or not enc.get("box_dim") \
                or enc.get("pixel_box_lower_corner") is None \
                or enc.get("pixel_box_upper_corner") is None:
            raise MissingFact(where, "tensor_map_im2col_corners")
        lo = [int(v) for v in enc["pixel_box_lower_corner"]]
        up = [int(v) for v in enc["pixel_box_upper_corner"]]
        if len(lo) != rank - 2 or len(up) != rank - 2:
            raise MissingFact(where, "tensor_map_im2col_corners")
        if not layout.get("element_bytes"):
            raise MissingFact(where, "tensor_map_element_bytes")
        box = [int(v) for v in enc["box_dim"][:2]]
        address = int(enc.get("global_address")
                      or image_global_address(entry.get("image")) or 0)
        if not address:
            raise MissingFact(where, "tensor_map_global_address")
        tm = model.TensorMap(rank=rank, element_bytes=layout.get(
            "element_bytes", 0), global_address=address,
            box_dim=(box[0], box[1]))
        geometry = {
            "mode": "im2col", "rank": rank,
            "global_address": address,
            "element_bytes": int(layout["element_bytes"]),
            "global_dim": [int(v) for v in
                           (enc.get("global_dim") or [])[:rank]],
            "global_strides": [int(v) for v in
                               (enc.get("global_strides") or [])[:rank - 1]],
            "element_strides": [int(v) or 1 for v in
                                (enc.get("element_strides") or [1] * rank)[:rank]],
            "cpp": box[0], "ppc": box[1],
            "lower": lo, "upper": up}
        transfer = geometry["cpp"] * geometry["ppc"] \
            * geometry["element_bytes"]
        return tm, {"rank": rank, "api": "im2col", "geometry": geometry,
                    "shared_bytes": transfer, "global_bytes": transfer}
    if layout["rank"] != rank:
        raise MissingFact(where, "tma encode rank contradicts opcode")
    if layout.get("global_address") is None:
        layout["global_address"] = image_global_address(entry.get("image"))
    if not layout.get("global_address"):
        raise MissingFact(where, "tensor_map_global_address")
    tm = model.TensorMap(
        rank=rank, element_bytes=layout["element_bytes"],
        global_dim=tuple(layout["global_dim"]),
        stride_bytes=tuple(layout["global_strides"]),
        box_dim=tuple(layout["box_dim"]),
        global_address=int(layout["global_address"]))
    box_bytes = math.prod(tm.box_dim) * tm.element_bytes
    geometry = {"global_address": tm.global_address,
                "element_bytes": tm.element_bytes,
                "rank": rank, "box": list(tm.box_dim),
                "global_dim": list(tm.global_dim),
                "global_strides": list(tm.stride_bytes),
                "element_strides": list(layout.get("element_strides")
                                        or [1] * rank)}
    return tm, {"rank": rank, "api": "tiled", "geometry": geometry,
                "shared_bytes": box_bytes, "global_bytes": box_bytes}


def tiled_box_rectangle(geometry: dict, coords) -> dict:
    """In-bounds rectangle of one tiled box at ``coords`` (hopper
    ``tiled_box_rectangle``): per dimension k the traversal visits j in
    [0, ceil(box_k / es_k)) at tensor index c_k + j * es_k; only
    0 <= index < dim_k touches memory."""
    rank = geometry["rank"]
    if len(coords) < rank:
        raise MissingFact("tma_box", "coordinate count below tensor rank")
    elem = geometry["element_bytes"]
    strides = [elem] + list(geometry["global_strides"])
    base = geometry["global_address"]
    counts, steps = [], []
    offset = 0
    for k in range(rank):
        es = geometry["element_strides"][k]
        n = -(-geometry["box"][k] // es)
        c = int(coords[k])
        dim = geometry["global_dim"][k]
        j_lo = 0 if c >= 0 else (-c + es - 1) // es
        j_hi = min(n - 1, (dim - 1 - c) // es) if c < dim else -1
        count = max(0, j_hi - j_lo + 1)
        counts.append(count)
        steps.append(es * strides[k])
        offset += (c + j_lo * es) * strides[k]
    touched = 1
    for count in counts:
        touched *= count
    return {"base": base + offset if touched else base, "elem_bytes": elem,
            "counts": counts, "steps": steps, "bytes": elem * touched}


def tiled_cut_rows(geometry: dict, coords) -> int:
    """tma_oob v2: rows of the box left in bounds when a tiled load is
    cut along dimension 0 (0 < in-bounds traversal < full box dim).
    Whole rows outside along other dimensions and fully inside /
    fully outside boxes return 0."""
    rect = tiled_box_rectangle(geometry, coords)
    es0 = geometry["element_strides"][0]
    n0 = -(-geometry["box"][0] // es0)
    c0 = rect["counts"][0]
    if 0 < c0 < n0:
        return math.prod(rect["counts"][1:])
    return 0


def tma_box_sectors(geometry: dict, coords) -> list[int]:
    """Sorted 32 B sector base addresses touched by the in-bounds box."""
    rect = tiled_box_rectangle(geometry, coords)
    if rect["bytes"] == 0:
        return []
    counts, steps = rect["counts"], rect["steps"]
    sectors: set[int] = set()
    # dimension 0 is contiguous when its step equals the element size
    if steps[0] == rect["elem_bytes"]:
        row_bytes = counts[0] * rect["elem_bytes"]
        inner = range(1)
        row_span = row_bytes
    else:
        inner = range(counts[0])
        row_span = rect["elem_bytes"]
    for offsets in itertools.product(*[range(c) for c in counts[1:]]):
        row = rect["base"] + sum(o * steps[i + 1]
                                 for i, o in enumerate(offsets))
        for j in inner:
            start = row + j * steps[0]
            sectors.update(range(start // _SECTOR * _SECTOR,
                                 (start + row_span - 1) // _SECTOR * _SECTOR
                                 + _SECTOR, _SECTOR))
    return sorted(sectors)


def bulk_sectors(address: int, size: int) -> list[int]:
    if size <= 0:
        return []
    lo = address // _SECTOR * _SECTOR
    hi = (address + size - 1) // _SECTOR * _SECTOR
    return list(range(lo, hi + _SECTOR, _SECTOR))


def _im2col_offsets(rank: int, value: int) -> list[int]:
    """Unpack the im2col packed-offsets word: field width is 16/8/5
    bits for rank 3/4/5, spatial dim i at bits [i*w, (i+1)*w) with W
    lowest (inferred from the ULEA shifts 5/10 building the word in
    the 5D SASS)."""
    width = {3: 16, 4: 8, 5: 5}[rank]
    return [(value >> (i * width)) & ((1 << width) - 1)
            for i in range(rank - 2)]


def im2col_box_sectors(geometry: dict, coords, offsets) -> list[int]:
    """Ordered 32 B sectors of one im2col load.  ``coords`` are the
    first ``rank`` coordinate words (c, spatial innermost-first, n)
    as u32; ``offsets`` the unpacked per-spatial-dim offsets."""
    rank = geometry["rank"]
    if len(coords) < rank:
        raise MissingFact("im2col_box", "coordinate count below rank")
    c = _s32(coords[0])
    spatial = [_s32(coords[1 + i]) for i in range(rank - 2)]
    n = _s32(coords[rank - 1])
    ns = rank - 2
    dim = geometry["global_dim"]
    stride = geometry["global_strides"]
    es = geometry["element_strides"]
    lo, up = geometry["lower"], geometry["upper"]
    if es[0] != 1:
        raise MissingFact("im2col_box", "tma_im2col_element_stride")
    cpp, ppc = geometry["cpp"], geometry["ppc"]
    size = dim[1:rank - 1]
    P, q0 = [], []
    for i in range(ns):
        p = (size[i] + up[i] - lo[i] - 1) // es[i] + 1
        num = spatial[i] - lo[i]
        if num % es[i] or num // es[i] < 0 or num // es[i] >= p:
            raise MissingFact("im2col_box", "tma_im2col_coordinate")
        P.append(p)
        q0.append(num // es[i])
    L = n
    for i in range(ns - 1, -1, -1):
        L = q0[i] + P[i] * L
    elem = geometry["element_bytes"]
    big_n = dim[rank - 1] if rank > 2 else 1
    sectors: list[int] = []
    seen: set[int] = set()
    for k in range(ppc):
        rest = L + k
        q = []
        for i in range(ns):
            q.append(rest % P[i])
            rest //= P[i]
        n_k = rest
        if n_k >= big_n:
            continue  # out of bounds: no memory
        a = [lo[i] + q[i] * es[i] + offsets[i] for i in range(ns)]
        if any(a[i] < 0 or a[i] >= size[i] for i in range(ns)):
            continue  # zero fill, no memory
        c_lo = max(c, 0)
        c_hi = min(c + cpp, dim[0])
        if c_lo >= c_hi:
            continue
        addr = (geometry["global_address"] + c_lo * elem
                + sum(a[i] * stride[i] for i in range(ns))
                + n_k * stride[ns])
        end = addr + (c_hi - c_lo) * elem - 1
        for s in range(addr // _SECTOR * _SECTOR,
                       end // _SECTOR * _SECTOR + _SECTOR, _SECTOR):
            if s not in seen:
                seen.add(s)
                sectors.append(s)
    return sectors


def _s32(v: int) -> int:
    v = int(v) & 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v
