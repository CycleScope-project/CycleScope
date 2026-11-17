"""Instance semantics kinds and named field slots (spec §4).

Each ``SEMA`` record carries ``kind`` plus six ``u64`` slots; the dataclasses
here name the slots per kind.  ``generation`` keeps the legacy encoding:
bit 63 = unresolved flag, bit 0 = parity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any

GENERATION_UNRESOLVED = 1 << 63

# SEMA.flags bits (also mirrored in convert.semantics_common)
SEMA_FLAG_AFTER_READY = 0x1
SEMA_FLAG_ALSO_ARRIVES = 0x2
SEMA_FLAG_IMPLICIT_COMMIT = 0x4
SEMA_FLAG_GROUP_OPENER = 0x8     # wgmma opened the current group itself
SEMA_FLAG_GROUP_COMMIT = 0x10    # wgmma operand carries gsb0 (closes unit)
SEMA_FLAG_STAT = 0x20            # LDTM.STAT variant
SEMA_FLAG_CTA_PAIR = 0x80        # TMA .2CTA: completions post on the
                                 # destination rank's pair leader (rank & ~1)
SEMA_FLAG_COMPLETE_TX = 0x100    # mbarrier_arrive: transaction_bytes are
                                 # complete_tx (0 arrivals), not expect_tx
SEMA_FLAG_SYNCS = 0x40           # NANOSLEEP.SYNCS: wakes on the last
                                 # polled mbarrier's phase completion

# "no object" sentinel for u32 object-reference fields (job-format §2)
SEMA_NO_OBJECT = 0xFFFFFFFF
SEMA_UNKNOWN = 0xFFFFFFFF   # unresolved value -> engine resolves (like
                            # GENERATION_UNRESOLVED; distinct meaning from
                            # SEMA_NO_OBJECT despite the same bits)


class SemanticsKind(IntEnum):
    none = 0
    mbarrier_init = 1
    mbarrier_arrive = 2
    mbarrier_expect_tx = 3
    mbarrier_wait = 4
    mbarrier_inval = 5
    cta_barrier_arrive = 6
    cta_barrier_wait = 7
    cluster_barrier_arrive = 8
    cluster_barrier_wait = 9
    tma_load = 10
    bulk_group_issue = 11
    bulk_group_commit = 12
    bulk_group_wait = 13
    async_copy_issue = 14
    async_copy_commit = 15
    async_copy_wait = 16
    bulk_copy = 17
    bulk_reduce = 18
    warpgroup_arrive = 19
    warpgroup_wait = 20
    wgmma = 22                 # Hopper
    umma_issue = 23            # Blackwell
    umma_commit = 24           # Blackwell
    tmem_alloc = 25            # Blackwell
    tmem_dealloc = 26
    tmem_relinquish = 27
    tmem_load = 28
    tmem_store = 29
    tmem_copy = 30
    tmem_wait_store = 31
    clc_query = 32             # Blackwell
    shared_store_async = 33
    global_sync = 34
    reconvergence_start = 35
    reconvergence_complete = 36
    nanosleep = 37
    descriptor_visibility = 38
    clock_marker = 39
    dependents_launch = 40


# kind -> field names for slots 0..5 ("-" = unused).  Slots named
# "<a>|<b>" pack a in the high 32 bits and b in the low 32 bits.
SEMANTICS_FIELDS: dict[SemanticsKind, tuple[str, ...]] = {
    SemanticsKind.none: (),
    SemanticsKind.mbarrier_init: (
        "object", "expected_arrivals", "scope", "initial_phase", "-",
        "object_list_begin|object_list_count"),
    SemanticsKind.mbarrier_arrive: (
        "object", "generation", "transaction_bytes", "arrival_count",
        "destination_mask", "object_list_begin|object_list_count"),
    SemanticsKind.mbarrier_expect_tx: (
        "object", "generation", "transaction_bytes", "scope"),
    SemanticsKind.mbarrier_wait: (
        "object", "generation|parity", "block_offset", "scope",
        "timeout_hint", "after_ready_mask"),
    SemanticsKind.mbarrier_inval: ("object",),
    SemanticsKind.cta_barrier_arrive: (
        "barrier", "expected_threads", "generation"),
    SemanticsKind.cta_barrier_wait: (
        "barrier", "expected_threads", "generation"),
    SemanticsKind.cluster_barrier_arrive: ("generation", "destination_mask"),
    SemanticsKind.cluster_barrier_wait: ("generation", "destination_mask"),
    SemanticsKind.tma_load: (
        "object", "generation", "multicast_mask", "shared_bytes",
        "global_bytes", "tensor_map"),
    SemanticsKind.bulk_group_issue: ("group", "pending_allowed", "bytes"),
    SemanticsKind.bulk_group_commit: ("group", "pending_allowed"),
    SemanticsKind.bulk_group_wait: ("group", "pending_allowed"),
    SemanticsKind.async_copy_issue: ("group", "pending_allowed", "object"),
    SemanticsKind.async_copy_commit: ("group", "pending_allowed", "object",
                                      "arrival_count", "transaction_count"),
    SemanticsKind.async_copy_wait: ("group", "pending_allowed", "object"),
    SemanticsKind.bulk_copy: (
        "object", "generation", "bytes", "multicast_mask"),
    SemanticsKind.bulk_reduce: (
        "object", "generation", "bytes", "multicast_mask"),
    SemanticsKind.warpgroup_arrive: ("group", "pending_allowed"),
    SemanticsKind.warpgroup_wait: ("group", "pending_allowed"),
    SemanticsKind.wgmma: (
        "group",
        "accumulator_registers_begin|accumulator_registers_count",
        "shape_packed", "input_dtype", "accumulator_dtype",
        "source_mode"),
    SemanticsKind.umma_issue: (
        "kind", "cta_group", "a_source", "tensor_m|tensor_n",
        "descriptor"),
    SemanticsKind.umma_commit: (
        "object", "cta_group", "commit_mask", "scope"),
    SemanticsKind.tmem_alloc: ("columns",),
    SemanticsKind.tmem_dealloc: ("columns",),
    SemanticsKind.tmem_relinquish: ("columns",),
    SemanticsKind.tmem_load: ("shape", "repeat"),
    SemanticsKind.tmem_store: ("shape", "repeat"),
    SemanticsKind.tmem_copy: ("shape", "repeat"),
    SemanticsKind.tmem_wait_store: (),
    SemanticsKind.clc_query: (
        "object", "generation", "destination_mask", "response_identity",
        "outcome"),
    SemanticsKind.shared_store_async: (
        "object", "bytes", "destination_rank",
        "object_list_begin|object_list_count"),
    SemanticsKind.global_sync: ("observation_kind", "class"),
    SemanticsKind.reconvergence_start: ("region", "envelope_cycles"),
    SemanticsKind.reconvergence_complete: ("region", "envelope_cycles"),
    SemanticsKind.nanosleep: ("requested_ns",),
    SemanticsKind.descriptor_visibility: ("operation", "descriptor_count",
                                          "descriptor_address_0",
                                          "descriptor_address_1",
                                          "descriptor_address_2",
                                          "descriptor_address_3"),
    SemanticsKind.clock_marker: ("kind",),
    SemanticsKind.dependents_launch: (),
}


def pack_fields(sema_kind: SemanticsKind, **values: Any) -> tuple[int, ...]:
    """Map named semantic fields onto the six u64 slots."""
    slots = [0] * 6
    names = SEMANTICS_FIELDS[sema_kind]
    for index, name in enumerate(names):
        if "|" in name:
            hi_name, lo_name = name.split("|")
            hi = values.get(hi_name, 0) & 0xFFFFFFFF
            lo = values.get(lo_name, 0) & 0xFFFFFFFF
            slots[index] = (hi << 32) | lo
        elif name == "object":
            value = values.get(name)
            slots[index] = SEMA_NO_OBJECT if value is None else value
        else:
            slots[index] = values.get(name, 0)
    return tuple(slots)


def unpack_fields(sema_kind: SemanticsKind, slots: tuple[int, ...]) -> dict:
    out: dict[str, int] = {}
    for index, name in enumerate(SEMANTICS_FIELDS[sema_kind]):
        if "|" in name:
            hi_name, lo_name = name.split("|")
            out[hi_name] = (slots[index] >> 32) & 0xFFFFFFFF
            out[lo_name] = slots[index] & 0xFFFFFFFF
        elif name == "object" and slots[index] == SEMA_NO_OBJECT:
            out[name] = None
        else:
            out[name] = slots[index]
    return out
