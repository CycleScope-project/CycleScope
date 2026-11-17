"""Block-scaled UMMA layouts (verbatim subset of blackwell-prototype
python/blackwellsim/umma_layout.py -- the generator only needs these two
objects; the rest of that module depends on the blackwellsim package)."""

BLOCK_SCALED_LAYOUTS = {
    "mxf8f6f4": {
        32: ("1X", [0, 1, 2, 3]),
        64: ("1X", [0, 1, 2, 3]),
    },
    "mxf4": {
        64: ("2X", [0, 2]),
        96: ("3X", [0, 1, 2, 3]),
        128: ("2X", [0, 2]),
    },
    "mxf4nvf4": {
        64: ("4X", [0]),
        96: ("6X", [0, 2]),
        128: ("8X", [0]),
    },
}


def block_scaled_layout(kind, tensor_k, sparse=False):
    dense_k = tensor_k // 2 if sparse else tensor_k
    return BLOCK_SCALED_LAYOUTS.get(kind, {}).get(dense_k)
