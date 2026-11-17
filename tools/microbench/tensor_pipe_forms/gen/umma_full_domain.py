#!/usr/bin/env python3
"""Expand the sm_103a tcgen05.mma shape domain from PTX Table 48."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from umma_layout import (  # noqa: E402
    BLOCK_SCALED_LAYOUTS, block_scaled_layout,
)

KINDS = ("f16", "tf32", "f8f6f4", "mxf8f6f4", "mxf4", "mxf4nvf4")
A_SRCS = ("smem", "tmem")
WS_M32_SHIFT3_GAP = (
    "probe triggers illegal instruction, cause unknown, PTX has no "
    "corresponding restriction"
)
def _scale_factor_facts(kind, k):
    layout = block_scaled_layout(kind, k)
    if layout is None:
        return {}
    scale_vec, sf_id_values = layout
    return {
        "scale_vec": f"scale_vec::{scale_vec}",
        "sf_id_values": sf_id_values,
    }


def _ns(first, step):
    return list(range(first, 257, step))


def _rows():
    rows = []

    def add(kind, operation, cg, m_values, n_values, k_values,
            *, sparse=False, ws=False):
        for m in m_values:
            for n in n_values:
                for k in k_values:
                    rows.append({
                        "kind": kind, "operation": operation,
                        "cta_group": cg, "M": m, "N": n, "K": k,
                        "sparse": sparse, "ws": ws,
                        # protocol addendum 2026-11-05 (r4): f16 / tf32
                        # take the f8f6f4 major rule (expand() narrows
                        # per N and a_src); the mx kinds stay K-major
                        "major_values": (
                            [0, 1, 2, 3]
                            if kind in {"f8f6f4", "mxf8f6f4", "f16", "tf32"}
                            else [0]
                        ),
                        "max_shift_values": [0, 1, 2, 3]
                        if operation == "ws" else [0],
                        "unmeasurable_max_shift_values": (
                            [3] if operation == "ws" and m == 32 else []
                        ),
                        "a_src_values": list(A_SRCS),
                        **_scale_factor_facts(kind, k),
                    })

    for kind, dense_k, sparse_k in (("f16", [16], [32]),
                                    ("tf32", [8], [16])):
        for cg, ms, first, step in (
                (1, [64, 128], 8, 8), (2, [128, 256], 16, 16)):
            ns = _ns(first, step)
            add(kind, "normal", cg, ms, ns, dense_k)
            add(kind, "sp", cg, ms, ns, sparse_k, sparse=True)
        for m in (32, 64, 128):
            add(kind, "ws", 1, [m], [64, 128, 256], dense_k, ws=True)
            add(kind, "ws_sp", 1, [m], [64, 128], sparse_k,
                sparse=True, ws=True)

    for cg, ms, first, step in (
            (1, [64, 128], 8, 8), (2, [128, 256], 16, 16)):
        ns = _ns(first, step)
        add("f8f6f4", "normal", cg, ms, ns, [32])
        add("f8f6f4", "sp", cg, ms, ns, [64], sparse=True)
    for m in (32, 64, 128):
        add("f8f6f4", "ws", 1, [m], [64, 128, 256], [32], ws=True)
        add("f8f6f4", "ws_sp", 1, [m], [64, 128], [64],
            sparse=True, ws=True)

    add("mxf8f6f4", "normal", 1, [128], _ns(8, 8), [32])
    add("mxf8f6f4", "normal", 2, [128, 256], _ns(16, 16), [32])
    add("mxf8f6f4", "sp", 1, [128], _ns(8, 8), [64], sparse=True)
    add("mxf8f6f4", "sp", 2, [256], _ns(16, 16), [64], sparse=True)

    add("mxf4", "normal", 1, [128], _ns(8, 8), [64, 96])
    add("mxf4", "normal", 2, [128], _ns(16, 16), [64])
    add("mxf4", "normal", 2, [256], _ns(16, 16), [64, 96])
    add("mxf4", "sp", 1, [128], _ns(8, 8), [128], sparse=True)
    add("mxf4", "sp", 2, [256], _ns(16, 16), [128], sparse=True)

    add("mxf4nvf4", "normal", 1, [128], _ns(8, 8), [64, 96])
    add("mxf4nvf4", "normal", 2, [128], _ns(16, 16), [64])
    add("mxf4nvf4", "normal", 2, [256], _ns(16, 16), [64, 96])
    add("mxf4nvf4", "sp", 1, [128], _ns(8, 8), [128], sparse=True)
    add("mxf4nvf4", "sp", 2, [256], _ns(16, 16), [128], sparse=True)

    return rows


def expand():
    result = []
    for row in _rows():
        for a_src in row.pop("a_src_values"):
            major_values = row["major_values"]
            if a_src == "tmem" and row["kind"] in {"f16", "tf32"}:
                major_values = [0, 1]
            elif a_src == "tmem" and row["kind"] in {"f8f6f4", "mxf8f6f4"}:
                major_values = [0, 1] if (
                    row["N"] >= (16 if row["cta_group"] == 1 else 32) and
                    row["N"] % (16 if row["cta_group"] == 1 else 32) == 0
                ) else [0]
            elif row["kind"] in {"f16", "tf32"}:
                # addendum 2026-11-05 (r5): the N >= 16 / 32 B-MN rule is
                # the f8 family's (PTX Table 63); 16-bit / tf32 B may be
                # MN-major at every N (hardware runs f16 N8 major 3)
                major_values = [0, 1, 2, 3]
            elif row["kind"] in {"f8f6f4", "mxf8f6f4"}:
                b_mn_valid = (
                    row["N"] >= (16 if row["cta_group"] == 1 else 32) and
                    row["N"] % (16 if row["cta_group"] == 1 else 32) == 0
                )
                major_values = [0, 1, 2, 3] if b_mn_valid else [0, 2]
            result.append({**row, "a_src": a_src,
                           "major_values": major_values})
    result.sort(key=lambda r: (
        r["kind"], r["cta_group"], r["a_src"], r["M"], r["N"], r["K"],
        r["operation"],
    ))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    doc = {
        "schema": "mechtree.umma_ptx_shape_domain.v1",
        "source": {
            "document": "PTX ISA 9.4",
            "shape_table": "9.7.18.2.1 Table 48",
            "transpose_b_table": "9.7.18.10.1 Table 63",
            "target_notes": "9.7.18.2.1.1, target sm_103a",
            "scale_vector_size_table": "PTX ISA 9.3 section 9.7.18.10.7.1 Table 69",
            "scale_factor_ids": (
                "PTX ISA 9.3 sections 9.7.18.10.7.2 and 9.7.18.10.7.3"
            ),
            "artifact": "mechtree/analysis/ptx-isa-9.3.txt",
        },
        "target": "sm_103a",
        "target_restrictions": [
            "K64 dense and K128 sparse f8f6f4/mxf8f6f4 are excluded (sm_107f+ only); K64 sparse is included",
            "K192 sparse mxf4/mxf4nvf4 is excluded (sm_107f+ only); K128 sparse is included",
            "K96 dense mxf4/mxf4nvf4 is included for sm_103a",
            "f8-family B-MN transpose requires N>=16 step16 for CG1 and N>=32 step32 for CG2",
            "c450 sparse cases use f16; the full sparse PTX domain includes f8 and mx kinds",
            "mxf8f6f4 and mx4 WS forms are invalid in Table 48",
        ],
        "known_measurement_gaps": [{
            "operation": "ws", "M": 32, "max_shift": 3,
            "status": "unmeasurable", "reason": WS_M32_SHIFT3_GAP,
        }],
        "shapes": expand(),
    }
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print("shape_rows", len(doc["shapes"]))


if __name__ == "__main__":
    main()
