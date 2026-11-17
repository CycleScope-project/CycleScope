"""Unified ``TIME.kind`` enumeration.

One mechanism program corresponds to one kind.  The table of kinds and the
mechanisms that use them is in ``docs/en/formats.md`` (section 6).
"""

from __future__ import annotations

from enum import IntEnum


class TimingKind(IntEnum):
    global_load_dependency = 1
    global_store_acceptance = 2
    shared_store_visibility = 3
    global_store_visibility = 4
    shared_issue_interval_smsp = 5
    shared_store_acceptance_sm = 6
    dependency_result_smsp = 7  # retired, never applied (id not reused)
    tma_verified_inflight_domain = 9
    tma_overlap_operation = 10
    hopper_wgmma_handoff_consumer = 13
    atomic_issue_interval_sm = 15
    nanosleep_warp_delay = 16
    cta_barrier_release = 17
    async_copy_mbarrier_arrival = 18
    tensor_map_descriptor_visibility = 19
    lifecycle_contract = 20
    service_start_interval = 21
    mbarrier_phase_release = 22            # merged: was 22/23
    mbarrier_trywait_timeout = 61
    hopper_qgmma_fmnmx_handoff_consumer = 29
    tma_overlap_operation_signature = 31
    tma_store_background_drain_extension = 32
    tma_operation_completion = 38
    storage_ldgsts_group_completion = 39  # was hopper_*; common since ldgsts-common
    cluster_barrier_release = 42
    hopper_tma_descriptor_refetch = 43
    hopper_wgmma_register_file_issue_steal = 44
    blackwell_umma_commit = 51
    blackwell_tmem_load = 52
    blackwell_tmem_store = 53
    blackwell_tmem_alloc = 54
    blackwell_tmem_copy = 55
    blackwell_clc_query = 56
    blackwell_tmem_load_stat = 57
    blackwell_umma_pair_arrive_skew = 59
    # shared tensor pipe (mechanism-merge-plan §2.1): per-SM pipe beat and
    # result latency of wgmma / tcgen05.mma ops, both architectures
    tensor_pipe_beat = 63
    tensor_result_latency = 64
    # per-SMSP issue interval of a global / local store by sector count
    # (storage.global.store_issue, lsu_accept protocol v5)
    global_store_issue_interval_smsp = 65
    # proxy fence after synchronous shared stores drains their
    # acceptance backlog (storage.shared.fence, tma_store_loaded
    # protocol-v2 addendum)
    shared_fence_drain = 66
    # per-SMSP acceptance of a synchronous shared store by wavefront
    # count (storage.shared.acceptance_smsp, shared protocol-v3
    # store_step addendum)
    shared_store_acceptance_smsp = 80
    # per-sector cost multiplier of a TMA reduce store on the SM store
    # port and the shared L2-write service; the bulk group also waits
    # for the DRAM fetch of destination sectors not resident in L2
    # (storage.tma.reduce, tma_store_loaded protocol-v5 store_dest)
    tma_reduce_store = 81
