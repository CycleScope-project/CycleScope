#!/usr/bin/env python3
"""Generate UMMA factor forms for the PTX full-domain probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen_umma_factor_forms import BASE_TYPES, add_form, emit_header
from umma_full_domain import (
    WS_M32_SHIFT3_GAP, block_scaled_layout, expand,
)

MX_SPARSE_UNSUPPORTED = (
    "Smoke umma-mxsp-smoke-r1 demonstrated the layout is usable; see "
    "tools/microbench/umma_mx_sparse_protocol.json."
)
_SECOND_BATCH_MX_SPARSE_SKIP_REASON = (
    "Block-scaled sparse metadata starts at TMEM column 384, while the "
    "scale_vec layout starting at column 320 requires 128-column strides. "
    "The current 512-column probe cannot represent both without overlap; "
    "no timing values are inferred or borrowed."
)
SF_ID_DOMAIN_SOURCE = (
    "PTX ISA 9.3 section 9.7.18.10.7.1 Table 69 and sections "
    "9.7.18.10.7.2 and 9.7.18.10.7.3"
)


def _cell_id(row):
    if (row["kind"], row["cta_group"], row["a_src"], row["M"],
            row["N"], row["K"], row["operation"]) == (
                "f16", 1, "smem", 128, 64, 16, "normal"):
        return "C1"
    if (row["kind"], row["cta_group"], row["a_src"], row["M"],
            row["N"], row["K"], row["operation"]) == (
                "f16", 1, "smem", 128, 128, 16, "normal"):
        return "C2"
    if (row["kind"], row["cta_group"], row["a_src"], row["M"],
            row["N"], row["K"], row["operation"]) == (
                "f16", 1, "smem", 128, 256, 16, "normal"):
        return "C3"
    if (row["kind"], row["cta_group"], row["a_src"], row["M"],
            row["N"], row["K"], row["operation"]) == (
                "f16", 1, "tmem", 128, 128, 16, "normal"):
        return "C4"
    if (row["kind"], row["cta_group"], row["a_src"], row["M"],
            row["N"], row["K"], row["operation"]) == (
                "f16", 2, "smem", 256, 128, 16, "normal"):
        return "C5"
    return (
        f'{row["kind"]}.cg{row["cta_group"]}.{row["a_src"]}.'
        f'm{row["M"]}.n{row["N"]}.k{row["K"]}.{row["operation"]}'
    )


def _add(forms, row, cell_id, factor, level, *, transpose=("K", "K"),
         max_shift=0, sf_id=(0, 0), scale_type=None, b_zero=False,
         overlap="none", status=None, acc=1, arot=0, brot=0, grp=0,
         issuers=1, kalu=0, elected=0):
    cell = {
        "kind": row["kind"], "cta_group": row["cta_group"],
        "a_src": row["a_src"], "M": row["M"], "N": row["N"],
        "K": row["K"], "swizzle": "SW128",
    }
    add_form(
        forms, cell_id, cell, factor, level,
        operation=row["operation"], sparse=row["sparse"],
        sparsity_version=0, transpose=transpose, max_shift=max_shift,
        sf_id=sf_id, scale_type=scale_type, b_zero=b_zero,
        overlap=overlap, acc=acc, arot=arot, brot=brot, grp=grp,
        issuers=issuers, kalu=kalu, elected=elected,
    )
    if elected:
        forms[-1]["form_id"] += ".elect"
    if status is not None:
        forms[-1]["run_status"] = status


def _controls():
    row128 = {
        "kind": "f16", "cta_group": 1, "a_src": "smem",
        "M": 128, "N": 128, "K": 16, "operation": "normal",
        "sparse": False,
    }
    row256 = {**row128, "N": 256}
    forms = []
    _add(forms, row128, "C2", "base", "base")
    _add(forms, row256, "C2", "probe_control", "N256",
         status="control")
    _add(forms, row256, "C3", "base", "base")
    row128_from_256 = {**row128, "N": 128}
    _add(forms, row128_from_256, "C3", "probe_control", "N128",
         status="control")
    _add(forms, row128, "C2", "probe_control", "B_zero",
         b_zero=True, status="control")
    return forms


def _is_in_batch(row, batch):
    if batch in {"accrot", "elect", "forms"}:
        return True
    sparse_mixed = row["sparse"] and row["kind"] not in {"f16", "tf32"}
    if batch == "first":
        return not sparse_mixed
    return sparse_mixed


def generate(batch, kind=None, cg=None, smoke=False, smoke_ws_m=None,
             smoke_ws_a_src=None):
    all_shapes = expand()
    selected = [
        row for row in all_shapes
        if _is_in_batch(row, batch)
        and (kind is None or row["kind"] == kind)
        and (cg is None or row["cta_group"] == cg)
    ]
    if smoke:
        chosen = []

        def choose(rows):
            if not rows:
                return []
            base = min(rows, key=lambda r: (r["M"], r["N"], r["K"]))
            result = [base]
            if kind in {"f8f6f4", "mxf8f6f4"}:
                b_mn = [row for row in rows if 1 in row["major_values"]]
                if b_mn:
                    second = min(b_mn, key=lambda r: (r["M"], r["N"], r["K"]))
                    if second not in result:
                        result.append(second)
            return result

        for source in ("smem", "tmem"):
            rows = [
                row for row in selected
                if row["a_src"] == source
                and row["operation"] == "normal" and not row["sparse"]
            ]
            chosen.extend(choose(rows))
        for operation in ("ws", "sp", "ws_sp"):
            rows = [row for row in selected if row["operation"] == operation]
            if operation == "ws" and kind in {"f16", "tf32", "f8f6f4"}:
                if smoke_ws_m is not None:
                    rows = [row for row in rows if row["M"] == smoke_ws_m]
                if smoke_ws_a_src is not None:
                    rows = [row for row in rows
                            if row["a_src"] == smoke_ws_a_src]
                if rows:
                    chosen.append(max(
                        rows, key=lambda r: (
                            r["N"], -r["M"], r["a_src"] == "smem")))
            else:
                chosen.extend(choose(rows))
        selected = chosen
    forms = []
    ids = set()

    def append(row, factor, level, **kwargs):
        cell_id = _cell_id(row)
        form_id = f"{cell_id}.{factor}.{level}"
        if kwargs.get("elected"):
            form_id += ".elect"
        if form_id in ids:
            return
        _add(forms, row, cell_id, factor, level, **kwargs)
        ids.add(form_id)

    if batch == "mxsp-smoke":
        if not smoke:
            raise ValueError("mxsp-smoke requires --smoke")
        targets = (
            ("mxf4nvf4", 128, (0,)),
            ("mxf8f6f4", 64, (0, 1, 2, 3)),
        )
        for target_kind, k, sf_ids in targets:
            row = next(
                item for item in all_shapes
                if item["kind"] == target_kind
                and item["cta_group"] == 2
                and item["a_src"] == "smem"
                and item["M"] == 256
                and item["N"] == 128
                and item["K"] == k
                and item["operation"] == "sp"
                and item["sparse"]
            )
            append(row, "base", "base")
            for sf_id in sf_ids:
                if sf_id:
                    append(row, "sf_id", f"{sf_id}/{sf_id}",
                           sf_id=(sf_id, sf_id))
        forms.extend(_controls())
        return forms

    if batch == "mxsp":
        for row in selected:
            append(row, "base", "base")
            for major in row["major_values"]:
                if major == 0:
                    continue
                transpose = ("MN" if major & 2 else "K",
                             "MN" if major & 1 else "K")
                append(row, "umma_operand_major", major,
                       transpose=transpose)
            for shift in row["max_shift_values"]:
                if shift:
                    append(row, "max_shift", shift, max_shift=shift)
            dense_k = row["K"] // 2
            layout = block_scaled_layout(row["kind"], dense_k)
            if layout is None:
                raise ValueError(
                    f"unmapped mx sparse sf_id kind={row['kind']} "
                    f"dense K={dense_k}")
            sf_id_values = layout[1]
            for a_id in sf_id_values:
                for b_id in sf_id_values:
                    if (a_id, b_id) != (0, 0):
                        append(row, "sf_id", f"{a_id}/{b_id}",
                               sf_id=(a_id, b_id))
            if row["kind"] == "mxf4nvf4":
                base_scale_type = BASE_TYPES[row["kind"]][3]
                for scale_type in ("UE4M3", "UE8M0"):
                    if scale_type != base_scale_type:
                        append(row, "scale_type", scale_type,
                               scale_type=scale_type)
        forms.extend(_controls())
        return forms

    if batch == "genaudit":
        # Frozen grid: tools/microbench/umma_factor_coverage_protocol.json
        proto = json.loads(
            (Path(__file__).resolve().parent /
             "umma_factor_coverage_protocol.json").read_text())["factors"]

        def find_row(c):
            row = next(
                r for r in all_shapes
                if r["kind"] == c["kind"]
                and r["cta_group"] == c["cta_group"]
                and r["a_src"] == c["a_src"]
                and r["M"] == c["M"] and r["N"] == c["N"]
                and r["K"] == c["K"] and r["operation"] == c["operation"]
                and not r["sparse"])
            return row

        for fname, fspec in proto.items():
            for c in fspec["cells"]:
                if kind is not None and c["kind"] != kind:
                    continue
                if cg is not None and c["cta_group"] != cg:
                    continue
                row = find_row(c)
                append(row, "base", "base")
                if fname == "umma_overlap":
                    for lvl, ov in (("1", "partial"), ("2", "same")):
                        append(row, "umma_overlap", lvl, overlap=ov)
                elif fname == "sf_id":
                    for a_id in row["sf_id_values"]:
                        for b_id in row["sf_id_values"]:
                            if (a_id, b_id) == (0, 0):
                                continue
                            append(row, "sf_id", f"{a_id}/{b_id}",
                                   sf_id=(a_id, b_id))
                elif fname == "scale_type":
                    append(row, "scale_type", "UE8M0",
                           scale_type="UE8M0")
                elif fname == "transpose_f16":
                    for lvl in fspec["levels"][c["a_src"]]:
                        a_t, b_t = lvl.split(",")
                        append(row, "transpose", lvl,
                               transpose=(a_t, b_t))
        for form in _controls():
            if form["form_id"] not in ids:
                forms.append(form)
                ids.add(form["form_id"])
        return forms

    if batch == "accrot":
        # Frozen set: tools/microbench/umma_accumulator_cadence_protocol.json.
        # tf32/f16 cg1 smem|tmem M64|M128 N{8..256} ACC{1,2,4,8} (ACC*N<=512);
        # f8f6f4 cg1 smem M128 N{16,64,256} ACC{1,8|4|2}.
        for row in selected:
            if row["operation"] != "normal" or row["sparse"] \
                    or row["cta_group"] != 1:
                continue
            if row["kind"] in {"tf32", "f16"} \
                    and row["M"] in {64, 128} \
                    and row["N"] in {8, 16, 32, 64, 128, 256}:
                accs = [a for a in (1, 2, 4, 8) if a * row["N"] <= 512]
            elif row["kind"] == "f8f6f4" and row["a_src"] == "smem" \
                    and row["M"] == 128 and row["N"] in {16, 64, 256}:
                accs = {16: [1, 8], 64: [1, 4], 256: [1, 2]}[row["N"]]
            else:
                continue
            for acc in accs:
                append(row, "accrot", f"acc{acc}", acc=acc)
            # Amendment 1 cells: operand rotation + kernel replica, all at ACC=8.
            is_a1 = (row["kind"] == "tf32" and row["a_src"] == "tmem"
                     and row["M"] == 64 and row["N"] == 16) or                 (row["kind"] == "f16" and row["a_src"] == "smem"
                 and row["M"] == 128 and row["N"] in {16, 64})
            if is_a1:
                append(row, "accrot", "acc8.arot", acc=8, arot=1)
                append(row, "accrot", "acc8.arotbrot", acc=8, arot=1, brot=1)
            if (row["kind"] == "tf32" and row["a_src"] == "tmem"
                    and row["M"] == 64 and row["N"] == 16):
                append(row, "accrot", "acc8.grp8", acc=8, arot=1, brot=1, grp=8)
                # Amendment 3: B1 issuer count, B2 ALU interleave.
                for iw in (2, 4):
                    append(row, "accrot", f"acc8.arotbrot.w{iw}",
                           acc=8, arot=1, brot=1, issuers=iw)
                for ka in (2, 4, 8, 16):
                    append(row, "accrot", f"acc8.arotbrot.k{ka}",
                           acc=8, arot=1, brot=1, kalu=ka)
        return forms

    if batch == "elect":
        # Frozen cells: tools/microbench/umma_accumulator_cadence_protocol.json
        # round_8_cells (E0-E3). All cg1 normal non-sparse; elected variants
        # get form_id ".elect" and the ELECTED=1 kernel form.
        for row in selected:
            if row["operation"] != "normal" or row["sparse"] \
                    or row["cta_group"] != 1:
                continue
            if row["kind"] == "tf32":
                if row["a_src"] == "tmem" and row["M"] == 64 \
                        and row["N"] in {8, 16, 32, 64, 128, 256}:
                    if row["N"] == 16:
                        # E0 controls on today's (ELECTED=0) form.
                        append(row, "accrot", "acc1", acc=1)
                        append(row, "accrot", "acc8.arotbrot",
                               acc=8, arot=1, brot=1)
                        # E2 issuer count, E3 kernel replica.
                        for iw in (2, 4):
                            append(row, "accrot", f"acc8.arotbrot.w{iw}",
                                   acc=8, arot=1, brot=1, issuers=iw,
                                   elected=1)
                        append(row, "accrot", "acc8.grp8",
                               acc=8, arot=1, brot=1, grp=8, elected=1)
                    # E1: acc1 on all N; acc8.arotbrot where ACC x N <= 512.
                    append(row, "accrot", "acc1", acc=1, elected=1)
                    if row["N"] <= 64:
                        append(row, "accrot", "acc8.arotbrot",
                               acc=8, arot=1, brot=1, elected=1)
                elif row["a_src"] == "smem" and row["M"] == 128 \
                        and row["N"] in {8, 16, 32, 64, 128, 256}:
                    append(row, "accrot", "acc1", acc=1, elected=1)
            elif row["kind"] == "f16":
                if row["a_src"] == "smem" and row["M"] == 128 \
                        and row["N"] in {8, 16, 32, 64, 128, 256}:
                    append(row, "accrot", "acc1", acc=1, elected=1)
                    if row["N"] <= 64:
                        append(row, "accrot", "acc8.arotbrot",
                               acc=8, arot=1, brot=1, elected=1)
                elif row["a_src"] == "tmem" and row["M"] == 128 \
                        and row["N"] in {16, 64, 256}:
                    append(row, "accrot", "acc1", acc=1, elected=1)
            elif row["kind"] == "f8f6f4":
                if row["a_src"] == "smem" and row["M"] == 128 \
                        and row["N"] in {16, 64, 256}:
                    append(row, "accrot", "acc1", acc=1, elected=1)
        return forms

    if batch == "forms":
        # tensor_pipe_forms batch: every expand() shape row, all elected=1,
        # base form plus one form per non-zero major and per non-zero
        # max_shift.  ws M32 shift 3 and mx-kind sparse stay unmeasurable.
        for row in selected:
            append(row, "base", "base", elected=1)
            for major in row["major_values"]:
                if major == 0:
                    continue
                transpose = ("MN" if major & 2 else "K",
                             "MN" if major & 1 else "K")
                append(row, "umma_operand_major", major,
                       transpose=transpose, elected=1)
            for shift in row["max_shift_values"]:
                if shift == 0:
                    continue
                append(row, "max_shift", shift, max_shift=shift, elected=1)
        for form in forms:
            if form["sparse"] and form["kind"].startswith("mx"):
                form["run_status"] = "unmeasurable"
                form["skip_reason"].append(
                    _SECOND_BATCH_MX_SPARSE_SKIP_REASON)
            if (form["operation"] == "ws" and form["M"] == 32 and
                    form["factor"] == "max_shift" and form["level"] == "3"):
                form["run_status"] = "unmeasurable"
                form["skip_reason"].append(WS_M32_SHIFT3_GAP)
        return forms

    for row in selected:
        cell_id = _cell_id(row)
        if f"{cell_id}.base.base" not in ids:
            append(row, "base", "base")
        for major in row["major_values"]:
            if major == 0:
                continue
            transpose = ("MN" if major & 2 else "K",
                         "MN" if major & 1 else "K")
            append(row, "umma_operand_major", major,
                   transpose=transpose)
        for shift in row["max_shift_values"]:
            if shift == 0:
                continue
            append(row, "max_shift", shift, max_shift=shift)

    for form in _controls():
        if form["form_id"] not in ids:
            forms.append(form)
            ids.add(form["form_id"])

    if (not smoke and batch == "first" and
            kind in (None, "f16") and cg in (None, 1, 2)):
        overlap_rows = [
            row for row in selected
            if row["kind"] == "f16" and row["operation"] == "normal"
            and row["a_src"] == "smem" and not row["sparse"]
            and _cell_id(row) in {"C1", "C2", "C3", "C5"}
        ]
        for row in overlap_rows:
            for level, overlap in (("1", "partial"), ("2", "same")):
                append(row, "umma_overlap", level, overlap=overlap)

    if batch == "first" and not smoke:
        sf_rows = []
        if kind in (None, "mxf8f6f4") and cg in (None, 2):
            sf_rows.append(next(
                item for item in all_shapes
                if item["kind"] == "mxf8f6f4"
                and item["operation"] == "normal"
                and item["cta_group"] == 2 and item["a_src"] == "smem"
                and item["M"] == 256 and item["N"] == 128
                and item["K"] == 32
            ))
        for sf_kind in ("mxf4", "mxf4nvf4"):
            if kind not in (None, sf_kind):
                continue
            for sf_cg in (1, 2):
                if cg not in (None, sf_cg):
                    continue
                for k in (64, 96):
                    sf_rows.append(next(
                        item for item in all_shapes
                        if item["kind"] == sf_kind
                        and item["operation"] == "normal"
                        and item["cta_group"] == sf_cg
                        and item["a_src"] == "smem"
                        and item["M"] == 128 * sf_cg
                        and item["N"] == 128 and item["K"] == k
                    ))
        for row in sf_rows:
            cell_id = _cell_id(row)
            for a_id in row["sf_id_values"]:
                for b_id in row["sf_id_values"]:
                    if (a_id, b_id) == (0, 0):
                        continue
                    append(row, "sf_id", f"{a_id}/{b_id}",
                           sf_id=(a_id, b_id))

        scale_type_cells = {
            "mxf8f6f4": (2, 256, 128, 32),
            "mxf4": (2, 256, 128, 64),
            "mxf4nvf4": (2, 256, 128, 64),
        }
        for sf_kind, (sf_cg, m, n, k) in scale_type_cells.items():
            if kind not in (None, sf_kind) or cg not in (None, sf_cg):
                continue
            row = next(
                item for item in all_shapes
                if item["kind"] == sf_kind and item["operation"] == "normal"
                and item["cta_group"] == sf_cg and item["a_src"] == "smem"
                and item["M"] == m and item["N"] == n and item["K"] == k
            )
            scale_types = (("UE8M0",) if sf_kind != "mxf4nvf4"
                           else ("UE4M3", "UE8M0"))
            for scale_type in scale_types:
                if scale_type != BASE_TYPES[sf_kind][3]:
                    append(row, "scale_type", scale_type,
                           scale_type=scale_type)

    if batch == "second":
        for form in forms:
            if form["sparse"] and form["kind"].startswith("mx"):
                form["run_status"] = "unmeasurable"
                form["skip_reason"].append(
                    _SECOND_BATCH_MX_SPARSE_SKIP_REASON)
    for form in forms:
        if (form["operation"] == "ws" and form["M"] == 32 and
                form["factor"] == "max_shift" and form["level"] == "3"):
            form["run_status"] = "unmeasurable"
            form["skip_reason"].append(WS_M32_SHIFT3_GAP)
    return forms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=(
        "first", "second", "genaudit", "mxsp", "mxsp-smoke", "accrot",
        "elect", "forms"), required=True)
    parser.add_argument("--kind", choices=(
        "f16", "tf32", "f8f6f4", "mxf8f6f4", "mxf4", "mxf4nvf4"))
    parser.add_argument("--cta-group", type=int, choices=(1, 2))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-ws-m", type=int, choices=(32, 64, 128))
    parser.add_argument("--smoke-ws-a-src", choices=("smem", "tmem"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    args = parser.parse_args()
    forms = generate(args.batch, args.kind, args.cta_group, args.smoke,
                     args.smoke_ws_m, args.smoke_ws_a_src)
    emit_header(forms, args.header)
    deferred = [
        row for row in expand()
        if args.batch == "first" and row["sparse"]
        and row["kind"] not in {"f16", "tf32"}
    ]
    args.out.write_text(json.dumps({
        "schema": "mechtree.umma_full_factor_forms.v1",
        "batch": args.batch, "kind": args.kind, "cta_group": args.cta_group,
        "shape_manifest": "tools/microbench/umma_full_domain.json",
        "smoke": args.smoke, "smoke_ws_m": args.smoke_ws_m,
        "smoke_ws_a_src": args.smoke_ws_a_src,
        "forms": forms, "deferred_sparse_shapes": deferred,
        "sf_id_domain_source": SF_ID_DOMAIN_SOURCE,
        "unsupported_sparse_reason": MX_SPARSE_UNSUPPORTED,
        "known_measurement_gaps": [{
            "operation": "ws", "M": 32, "max_shift": 3,
            "status": "unmeasurable", "reason": WS_M32_SHIFT3_GAP,
        }],
    }, indent=1) + "\n")
    print("forms", len(forms), "deferred_sparse_shapes", len(deferred))


if __name__ == "__main__":
    main()
