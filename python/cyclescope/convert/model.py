"""Typed converter model — the only interface between converters and the
segment builder (docs/en/design/convert.md).

Everything here is a captured fact: no inferred generations, pairings, or
group relations. Inference happens inside the builder (cluster-local) and
the container assembler (cross-cluster).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Placement:
    sm_id: int
    warp_slot_base: int
    source: int = 1  # PLCE.source: 1 trace / 2 probe


@dataclass(slots=True)
class ControlWord:
    stall: int = 0
    yield_hint: int = 0
    read_barrier: int = 7
    write_barrier: int = 7
    wait_mask: int = 0
    reuse_mask: int = 0


@dataclass(slots=True)
class StaticInstruction:
    pc: int
    code_address: int | None
    opcode: str
    operation_class: str      # mechanism-tree axis spelling
    operands: tuple[str, ...] = ()
    predicate: str | None = None
    predicate_inverted: bool = False
    source_registers: tuple[str, ...] = ()
    destination_registers: tuple[str, ...] = ()
    register_roles: dict = field(default_factory=dict)   # name -> "data"|"address"
    control: ControlWord = field(default_factory=ControlWord)
    memory_space: str | None = None
    branch_target: int | None = None


@dataclass(slots=True)
class MemoryReference:
    space: str              # global/shared/local/constant/generic
    access_kind: str        # load/store/atomic/async_copy/async_reduce/prefetch
    width_bytes: int
    lane_addresses: tuple = ()   # per-lane byte address, lane order
    lane_mask: int = 0
    remote_rank: int | None = None   # cluster DSMEM target rank
    cache_hint: int = 0            # bit field (Blackwell L1/L2 hints; Hopper 0)
    destination_count: int | None = None  # multicast/st-async destinations
    transfer_bytes: int | None = None     # TMA/bulk tile/copy bytes
    cut_rows: int | None = None           # tiled TMA: in-bounds rows when
                                          # the box is cut along dim 0


@dataclass(slots=True)
class SemanticOperands:
    uniform: dict = field(default_factory=dict)        # UR name -> value
    lane: dict = field(default_factory=dict)           # R name -> per-lane values
    ranks: dict = field(default_factory=dict)          # mref index -> getctarank per lane
    after_ready_mask: int | None = None                # PHASECHK/TRYWAIT dest ballot
    tensor_map: int | None = None
    tensor_coordinates: tuple | None = None
    descriptor_words: tuple | None = None              # UMMA idesc/gdesc/tmem


@dataclass(slots=True)
class GlobalSyncObservation:
    kind: int               # 1 release / 2 acquire / 3 atomic
    address: int
    value: int
    lane: int = 0
    acquire_mode: int = 0
    post_known: int = 0
    before: int = 0         # representative lane's pre-op word
    sequence: int = 0       # launch-wide GSYNC record ordinal


@dataclass(slots=True)
class DynamicInstruction:
    static_index: int
    active_mask: int
    predicate_mask: int = 0
    memory: tuple = ()
    operands: SemanticOperands | None = None
    global_sync: GlobalSyncObservation | None = None
    raw_key: bytes = b""


@dataclass(slots=True)
class TensorMap:
    rank: int = 0
    element_bytes: int = 0
    global_dim: tuple = ()          # per-dim element count
    stride_bytes: tuple = ()        # per-dim byte stride (dims 1..rank-1)
    box_dim: tuple = ()             # per-dim box size in elements
    global_address: int = 0


@dataclass(slots=True)
class WarpStream:
    cta_linear: int
    cta: tuple
    warp_in_cta: int
    sm_id: int
    warp_slot: int
    instructions: list = field(default_factory=list)   # program order


@dataclass(slots=True)
class CtaFacts:
    cta_linear: int
    cta: tuple                    # (x, y, z)
    cluster_rank: int = 0
    sm_id: int = 0xFFFFFFFF
    peer_sm_id: int = 0xFFFFFFFF
    multicast_mask: int = 0
    cta_group: int = 0
    is_pair_leader: int = 0
    tmem_columns: int = 0


@dataclass(slots=True)
class ClusterUnit:
    cluster_id: int
    ctas: list = field(default_factory=list)           # CtaFacts
    warps: list = field(default_factory=list)          # WarpStream
    statics: list = field(default_factory=list)        # StaticInstruction
    tensor_maps: dict = field(default_factory=dict)    # id -> TensorMap
    preinitialized_mbarriers: list = field(default_factory=list)


@dataclass(slots=True)
class LaunchFacts:
    device: str = ""
    sm_arch: str = ""
    grid: tuple = (1, 1, 1)
    block: tuple = (1, 1, 1)
    cluster: tuple = (1, 1, 1)
    shared_static_bytes: int = 0
    shared_dynamic_bytes: int = 0
    registers_per_thread: int = 0
    sm_count: int | None = None
    l2_bytes: int | None = None
    occupancy_blocks_per_sm: int | None = None
    occupancy_active_clusters: int | None = None
    function_prior_launches: int = 0
    reference_sm_clock_khz: int | None = None
    reference_cache_control: str | None = None     # "none" | "all"
    placements: dict = field(default_factory=dict)  # cta_linear -> Placement
    sm_topology: Path | None = None
    extra: dict = field(default_factory=dict)       # pass-through launch.json keys
