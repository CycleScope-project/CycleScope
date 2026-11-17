"""PTX 9.7.18.4.1/.2 descriptor field encoders for UMMA factor forms."""

from __future__ import annotations

FORMAT = {
    "F16": 0, "BF16": 1, "TF32": 2,
    "E4M3": 0, "E5M2": 1, "E2M3": 3, "E3M2": 4, "E2M1": 5,
}
MXF4_FORMAT = {"E2M1": 1}
SCALE_FORMAT = {"UE4M3": 0, "UE8M0": 1}
D_FORMAT = {"F16": 0, "F32": 1}
SWIZZLE = {
    "NONE": 0, "SW128_32B_ATOM": 1, "SW128": 2, "SW64": 4, "SW32": 6,
}
SWIZZLE_REPEAT = {
    "NONE": 128, "SW128_32B_ATOM": 1024, "SW128": 1024,
    "SW64": 512, "SW32": 256,
}
SWIZZLE_ATOM_MN = {
    "NONE": 1, "SW128_32B_ATOM": 8, "SW128": 8, "SW64": 4, "SW32": 2,
}
SWIZZLE_ATOM_K = {
    "NONE": 8, "SW128_32B_ATOM": 4, "SW128": 8, "SW64": 8, "SW32": 8,
}


def encode_idesc(kind, m, n, a_type, b_type, d_type="F32",
                 transpose=("K", "K"), negate=("none", "none"),
                 sf_id=(0, 0), scale_type="UE8M0", max_shift=0, k=None,
                 sparse=False, sparsity_version=0):
    if m % 16 or n % 8 or max_shift not in range(4):
        raise ValueError("invalid UMMA shape or max_shift")
    if (len(sf_id) != 2 or any(value not in range(4) for value in sf_id) or
            sparsity_version not in range(2)):
        raise ValueError("invalid UMMA scale-factor or sparsity ID")
    if kind in {"f16", "f8f6f4", "tf32"}:
        if any(sf_id) or sparsity_version:
            raise ValueError("non-block-scaled UMMA has no SF-ID/version fields")
        if k is None:
            k = {"f16": 16, "tf32": 8, "f8f6f4": 32}[kind]
        allowed_k = {
            "f16": {16, 32},
            "tf32": {8, 16},
            "f8f6f4": {32, 64, 128},
        }[kind]
        if k not in allowed_k:
            raise ValueError(f"invalid K={k} for {kind}")
        if sparse and k not in {
                "f16": {32}, "tf32": {16},
                "f8f6f4": {64, 128},
        }[kind]:
            raise ValueError(f"invalid sparse K={k} for {kind}")
        if not sparse and k in {
                "f16": {32}, "tf32": {16},
                "f8f6f4": {128},
        }[kind]:
            raise ValueError(f"invalid dense K={k} for {kind}")
        if kind == "tf32":
            a_format = b_format = FORMAT["TF32"]
        else:
            a_format, b_format = FORMAT[a_type], FORMAT[b_type]
        desc = (D_FORMAT[d_type] << 4) | (a_format << 7) | (b_format << 10)
        if negate[0] != "none":
            desc |= 1 << 13
        if negate[1] != "none":
            desc |= 1 << 14
        if transpose[0] == "MN":
            desc |= 1 << 15
        if transpose[1] == "MN":
            desc |= 1 << 16
        desc |= (n >> 3) << 17
        desc |= (m >> 4) << 24
        if (kind == "f8f6f4" and
                ((k == 64 and not sparse) or (k == 128 and sparse))):
            desc |= 1 << 29
        desc |= max_shift << 30
        if sparse:
            desc |= 1 << 2
        return desc
    if kind not in {"mxf8f6f4", "mxf4", "mxf4nvf4"}:
        raise ValueError(f"unsupported kind: {kind}")
    if max_shift:
        raise ValueError("block-scaled UMMA has no max_shift field")
    if k is None:
        k = {"mxf8f6f4": 32, "mxf4": 64, "mxf4nvf4": 64}[kind]
    dense_k = {
        "mxf8f6f4": {32, 64},
        "mxf4": {64, 96},
        "mxf4nvf4": {64, 96},
    }[kind]
    sparse_k = {
        "mxf8f6f4": {64, 128},
        "mxf4": {128},
        "mxf4nvf4": {128},
    }[kind]
    if k not in (sparse_k if sparse else dense_k):
        qualifier = "sparse" if sparse else "dense"
        raise ValueError(f"invalid {qualifier} K={k} for {kind}")
    if sparsity_version and not sparse:
        raise ValueError("dense UMMA has no sparsity-version field")
    if kind == "mxf8f6f4" and sparsity_version:
        raise ValueError("mxf8f6f4 has no sparsity-version field")
    allowed_scale_types = {
        "mxf8f6f4": {"UE8M0"},
        "mxf4": {"UE8M0"},
        "mxf4nvf4": {"UE4M3", "UE8M0"},
    }[kind]
    if scale_type not in allowed_scale_types:
        raise ValueError(f"invalid scale type {scale_type} for {kind}")
    if kind in {"mxf4", "mxf4nvf4"}:
        a_format, b_format = MXF4_FORMAT[a_type], MXF4_FORMAT[b_type]
    else:
        a_format, b_format = FORMAT[a_type], FORMAT[b_type]
    desc = (sf_id[1] << 4) | (a_format << 7) | (b_format << 10)
    if negate[0] != "none":
        desc |= 1 << 13
    if negate[1] != "none":
        desc |= 1 << 14
    if transpose[0] == "MN":
        desc |= 1 << 15
    if transpose[1] == "MN":
        desc |= 1 << 16
    desc |= (n >> 3) << 17
    desc |= SCALE_FORMAT[scale_type] << 23
    desc |= (m >> 7) << 27
    desc |= sf_id[0] << 29
    if sparse:
        desc |= 1 << 2
        if kind in {"mxf4", "mxf4nvf4"}:
            desc |= sparsity_version << 12
    if ((kind == "mxf8f6f4" and
         ((k == 64 and not sparse) or (k == 128 and sparse))) or
            (kind in {"mxf4", "mxf4nvf4"} and k == 96 and not sparse)):
        desc |= 1 << 31
    return desc


def encode_smem(start, lbo, sbo, swizzle, *, lbo_mode="relative",
                base_offset=0):
    if start % 16 or lbo % 16 or sbo % 16:
        raise ValueError("SMEM descriptor addresses and strides must be 16B aligned")
    if swizzle not in SWIZZLE or lbo_mode not in {"relative", "absolute"}:
        raise ValueError("unknown swizzle or LBO mode")
    if not 0 <= base_offset < 8:
        raise ValueError("base_offset must fit three bits")
    fields = (start >> 4, lbo >> 4, sbo >> 4)
    if any(value >= 1 << 14 for value in fields):
        raise ValueError("SMEM descriptor field exceeds the 14-bit encoding")
    return (
        fields[0] | (fields[1] << 16) | (fields[2] << 32) |
        (1 << 46) | (base_offset << 49) |
        (int(lbo_mode == "absolute") << 52) |
        (SWIZZLE[swizzle] << 61)
    )


def base_offset_for(pattern_start, swizzle):
    if pattern_start < 0 or pattern_start % 16:
        raise ValueError("pattern start must be a nonnegative 16B address")
    if swizzle not in SWIZZLE_REPEAT:
        raise ValueError(f"unknown swizzle: {swizzle}")
    repeat = SWIZZLE_REPEAT[swizzle]
    if pattern_start % repeat == 0:
        return 0
    return (pattern_start >> 7) & 0x7


def canonical_lbo_sbo(rows, swizzle, major):
    """Canonical byte strides from CUTLASS's make_umma_desc layout atoms.

    `SwizzleAtomMNSize` and `SwizzleAtomKSize` follow the cases in
    cute/arch/mma_sm100_desc.hpp.  The descriptor fields are byte offsets;
    encode_smem performs the PTX 16-byte-unit conversion.
    """
    if major not in {"K", "MN"} or swizzle not in SWIZZLE:
        raise ValueError("unknown UMMA major or swizzle")
    if major == "K":
        if swizzle == "SW128_32B_ATOM":
            raise ValueError("SW128_32B_ATOM is invalid for Major::K")
        if swizzle == "NONE":
            return rows * 16, 128
        return 16, SWIZZLE_ATOM_MN[swizzle] * 128
    repeat = SWIZZLE_REPEAT[swizzle]
    if swizzle == "SW128_32B_ATOM":
        # MN-major 32-bit operands (tf32): the only accepted layout is
        # SWIZZLE_128B_BASE32B — 512 B atom = 32 elements x 4 K-rows,
        # LBO = MN-atom stride, SBO = 4-row K-group stride (r6 addendum)
        return 512, 512 * rows // 32
    return rows * (repeat // 16), SWIZZLE_ATOM_MN[swizzle] * 128


def descriptor_view_span(rows, k, atype_bits, swizzle, major, lbo, sbo):
    """Conservative 16-byte-granular span for the UMMA swizzle atom view."""
    atom_m = SWIZZLE_ATOM_MN[swizzle]
    atom_k = SWIZZLE_ATOM_K[swizzle]
    t = max(1, 128 // atype_bits)
    if major == "K":
        row_groups = (rows + atom_m - 1) // atom_m
        k_groups = max(1, (k + 2 * t - 1) // (2 * t))
        row_stride, k_stride = sbo, lbo
    else:
        row_groups = (rows + t * atom_m - 1) // (t * atom_m)
        k_groups = max(1, (k + t * atom_k - 1) // (t * atom_k))
        row_stride, k_stride = lbo, sbo
    return (max(0, row_groups - 1) * row_stride +
            max(0, k_groups - 1) * k_stride + 128)


def one_encoding(type_name):
    """Return (storage bits, scalar one code) for the PTX input/scale type."""
    if type_name == "F16":
        return 16, 0x3C00
    if type_name == "BF16":
        return 16, 0x3F80
    if type_name == "F32":
        return 32, 0x3F800000
    if type_name == "TF32":
        return 32, 0x3F800000
    if type_name == "E4M3":
        return 8, 0x38
    if type_name == "E5M2":
        return 8, 0x3C
    if type_name == "E2M3":
        return 6, 0x08
    if type_name == "E3M2":
        return 6, 0x0C
    if type_name == "E2M1":
        return 4, 0x02
    if type_name == "UE4M3":
        return 8, 0x38
    if type_name == "UE8M0":
        return 8, 0x7F
    raise ValueError(f"unknown numeric type: {type_name}")
