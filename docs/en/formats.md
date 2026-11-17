# Binary job formats

[中文](../zh/formats.md)

The binary layouts of the files the pipeline reads and writes:
`GSIMSEG` (segment), `GSIMJOB` (container), `GSIMRES` (result),
`GSIMJNL` (issue journal), `GSIMWLG` (wait ledger). You need this page
when you parse or diff the files — everything else links here rather
than repeating a layout. All integers are little-endian; record widths
are declared in each file's table directory, so never hard-code
offsets. [design/job-format.md](design/job-format.md) explains what
the tables are for.

## 1. File shell

```
FileHeader (72 B)
  magic[8]          "GSIMSEG\0" | "GSIMJOB\0" | "GSIMRES\0" | "GSIMJNL\0" | "GSIMWLG\0"
  u32  format_version = 1
  u32  header_bytes = 72
  u64  file_bytes
  u64  directory_offset
  u32  table_count
  u32  flags            bit0 = payload sha256 present
  u8[32] payload_sha256  (hash of the file with this field zeroed)
TableDirectory: table_count x TableEntry (32 B)
  u32 tag (4-char ASCII) | u32 record_bytes | u64 record_count |
  u64 offset (8 B aligned) | u64 reserved = 0
```

- `record_bytes = 0` marks a byte blob (STRS); then `record_count` is the
  byte length. Unknown tags are ignored. `u32 0xFFFFFFFF` is the only
  "none" sentinel (`u64` fields use `0xFFFFFFFFFFFFFFFF`).

## 2. Segment `GSIMSEG` (one cluster)

**Version 2 (split program tables)**: `format_version = 2`. The seven
shared program tables (`PROG`, `NODE`, `GRD `, `LEAF`, `PARM`, `PTS `,
`AXIS`) are absent from the segment; they live once per segments dir in
`segments/program.tables` (a `GSIMPRG0`-magic file holding exactly those
tables). Readers splice them back; META `program_digest` still refers to
the shared tables. The C++ (`cyclescope-convert`) and Python
(`program_tables_payload`) writers emit byte-identical `program.tables`.
Version 1 remains readable.


| tag | record | fields (bytes) |
|---|---|---|
| META | 120 | tree_digest[32], program_digest[32], converter_digest[32], cluster_id, cta_count, warp_count u32, instruction_count u64, flags u32 (bit0 model_complete) |
| CTAS | 56 | cta_index, cta_x/y/z, cluster_rank, cluster_shape_x/y/z, sm_id, peer_sm_id, multicast_mask u32; cta_group, is_pair_leader u8; tmem_columns u16; first_warp, warp_count u32 |
| WARP | 32 | cta_index, warp_in_cta, warp_slot, smsp u32; first_instance, instance_count u64 |
| TMPL | 68 | operation_class u32 (STRS offset), flags u32, resource u32, issue_group_resource u32; source_begin u64, source_count u32, destination_begin u64, destination_count u32; scoreboard_read/write/wait_mask u8, control u8 (bit0-3 stall, bit4 yield); timing_begin u64, timing_count u32, semantic_kind u32, semantic_u64 u64 |
| INST | 16 | template, semantics, memory_access, code_line u32 |
| REGS | 4 | register u32: bits 0-23 register id (namespaces: R<n>→n, UR<n>→256+n, P<n>→512+n, UP<n>→768+n, B<n>→1024+n; hardwired or special names such as RZ/URZ/PT/UPT/SR_* are never issued and never enter template keys), bits 24-31 flags (bit0 reuse_served: the static instruction at pc-16 carries a reuse flag on the same operand slot of the same register; bit2 reuse_carrier: this instruction's own control-word reuse_mask bit on that source slot; bit1 pair: a 64-bit source operand (a `.64`/`.F32x2` suffix, or a DADD/DMUL/DFMA/DSETP/DMNMX class, or an FFMA2/FADD2/FMUL2-class token without the `.F32` suffix — `.F32` broadcasts a scalar, not a pair); destination-register rows have flags=0. Source-slot order = operand text minus the destination register and any following plain predicate destinations (PT/UPT/P<n>/UP<n>); non-register tokens occupy slots too and correspond to the reuse bits position-wise) |
| SEMA | 60 | kind u16, flags u16, earliest_issue u64, field[6] u64 (§5) |
| MACC | 48 | sector_begin u64, sector_count u32, flags u32 (low 3 bits = space: 1 shared, 3 constant; bit5 remote; bits 6-8 = access_kind: 0x0 load / 0x40 store / 0x80 atomic / 0xC0 async_copy / 0x100 async_reduce / 0x140 prefetch; bit9 = LDGSTS without .BYPASS fills the L1); facts space `shared_cluster` resolves to 1 — bit5 set when the reference carries a remote rank — `global_system` to 0, `param` to 4), sequence u64, shape u64, timing_begin u64, timing_count u32, cache_hint u32 (bit field, Blackwell L1/L2 hints) |
| MSEC | 16 | address u64, byte_mask u32, lane_mask u32 (sector rows merge lanes only within one access; different accesses keep separate rows even on the same sector) |
| TIME | 40 | kind, selector_a, selector_b, ready_warps u32; selector_c u64; cycles f64; enabled u8, pad[7] |
| PROG | 32 | mechanism u32 (STRS), event u8, schema_id u16, timing_kind u8, root_node u32, field_mask u64, operation_class_begin u64 (STRS), operation_class_count u32 |
| NODE | 24 | parent, child_begin, child_count, guard_begin, guard_count, leaf u32 |
| GRD  | 24 | axis u32, operator u8 (0 eq / 1 range), pad[3], low u64, high u64 |
| LEAF | 8 | parameter u32 (first PARM row), status u8 (0 measured/1 experimental/2 placeholder/3 unsupported/4 structural), pad[3] |
| PARM | 48 | schema_id u16, field_id u16, model_kind u8 (0 constant/1 discrete/2 linear/3 affine), pad[3], axis u32, value f64, slope f64, intercept f64, points_begin u64, points_count u32. discrete: value = placeholder_default, slope = the axis count width, intercept != 0 = fail_closed (a table miss raises `unsupported:engine:discrete_miss`; leaves without a placeholder_default get 1 from the builder) |
| PTS  | 16 | axis_value f64, value f64. Constant uses none; linear-with-points stores (x,y) pairs; discrete stores, per point, `width` axis rows then one value row |
| FACT | 12 | axis u32, value u64 (enum axes store the value index) |
| USES | 38 | node u32, target u8, pad[3], target_index u64, schema_id u16, field_mask u64, fact_begin u64, fact_count u32 |
| AXIS | 4 | name u32 (STRS offset); row index = numeric axis id used by GRD/PARM/init-FACT |
| SPLT | 16 | semantics, rank, credit, pad u32 |
| MBPI | 24 | cta_index, object u32, address u64, arrival_count u32, pad u32 |
| MBOL | 4 | object u32 |
| MBOB (optional) | 12 | cta_index u32, address u64 — one row per segment-local object naming its owner CTA and CTA-local shared address (engine 3-3 multicast mapping) |
| GSYN (optional) | 40 | instance u32 (segment-local), lane, kind, class u8, flags u8, address, after, before, sequence u64 — same field semantics as the container GSYN table minus the segment column; written only when the segment has observations |
| CODE | 16 | instance u32, pad u32, code_address u64 |
| CLKM | 16 | warp, kind u32, instance u64 |
| STRS | blob | NUL-terminated strings |

USE targets: 0 InstructionTemplate, 1 TimingRow, 2 InstanceTiming,
3 MemoryAccess, 4 Resource, 5 MemoryLevel, 6 GpuMemory, 7 Scheduler,
8 Instance, 15 none.

## 3. Program tables `GSIMPRG01` (one tree)

`cyclescope tree compile` output: the compiled mechanism program tables a
converter needs without re-walking the JSON tree. Same file shell as
§1 (magic `GSIMPRG0`). Tables:

| tag | record / blob | contents |
|---|---|---|
| META | 64 B blob | tree_digest[32] + program_digest[32] |
| PROG | §2 PROG rows (pad byte = timing_kind) | mechanism programs in sorted-name order |
| NODE / GRD  / LEAF / PARM / PTS  / FACT / USES / AXIS / STRS | same record layouts as §2 | traversal inputs, incl. init-path FACT/USES |
| SCHM | 4 B: schema_id u16, target u8, pad u8 | parameter-schema id -> USES target (schema_registry.json) |
| AXV  | 16 B: kind u32 (1 enum), owner u8 (1 engine), pad u8, pad[2], values_begin u32, values_count u32 | axis kind/owner + enum values window |
| AXVN | blob | NUL-separated enum value names, indexed by AXV.values_begin |
| RAX  | 8 B: begin, count u32 | per NODE row -> RAXV span |
| RAXV | 4 B: axis u32 | required capture axes per node (post-order union of leaf-model axes) |

Consumers: `cyclescope-convert` (C++ segment builder) reads it via
`--programs`; the python builder holds the same payload set in memory.

## 4. Container `GSIMJOB` (one launch)

| tag | record | fields |
|---|---|---|
| LNCH | 104 | grid_x/y/z, block_x/y/z, cluster_x/y/z u32 x9; shared_static_bytes, shared_dynamic_bytes, registers_per_thread, sm_count u32; l2_bytes u64; occupancy_blocks_per_sm, occupancy_active_clusters, function_prior_launches, reference_sm_clock_khz u32; reference_cache_control u8 (0 unknown/1 none/2 all; `cyclescope build` accepts only 2: 0 -> `missing_fact:reference_cache_control`, 1 -> `unsupported:reference_cache_control:none`), device_facts_present u8, pad u16; flags u32 (bit0 has_unplaced, bit1 has_global_sync, bit2 has_clc, bit3 partial_launch), native_ctas_launched u32; device char[16] |
| SEGD | 64 | segment, cluster_id u32, byte_offset, byte_length u64, cta_base, cta_count u32, segment_sha256[32] |
| PLCE | 28 | cta, segment, cta_in_segment u32, placement u8 (0 unplaced/1 placed), pad[3], sm_id, warp_slot_base u32, source u8 (1 trace/2 probe), pad[3] |
| GSYN | 44 | segment (the segment's index in SEGD; rows are collected by `cyclescope build` from each segment's optional GSYN table), instance u32 (segment-local), lane, kind, class u8 (1 atom/2 red/3 ld_strong/4 st_strong), flags u8 (bit0 after/bit1 before valid), address, after, before, sequence u64. before = pre_value (RED, tracer phase-0 tail words) or the returned old value (ATOM, phase-1 words); after = post(op, before, operands) for RED/ATOM (ADD/EXCH/CAS/MIN/MAX/INC/DEC/AND/OR/XOR, float ADD), the stored operand (ST) or the destination word (LD); a 0 pre_value is a known value |
| TOPO | 24 | sm, placement_domain, l2_partition, tpc_pair, gpc, pad u32 (`gpc` is the L2 read-merge domain for the request kinds selected by `memory.l2_merge.domain_gpc`) |

The container has no L2 footprint tables (the former `L2FP`/`L2FM` were
removed with the cold-start reference condition, 2026-11-03): the engine
starts every launch with empty caches.

Segment images: each complete `GSIMSEG` file appended 8-byte-aligned after
the tables, located via `SEGD.byte_offset/byte_length`.

Merge rules (§4): CTA identity — a segment's `CTAS/WARP/MBPI/MBOB.cta_index`
carry the launch-wide linear CTA id (`x + y*gx + z*gx*gy`); the merged job
renumbers CTAs densely in CTAS table order (`cta_index` == row in the merged
CTAS), rewrites the referring fields and translates container `PLCE.cta`
(launch-wide) to the dense id, dropping rows for CTAs absent from the job
(partial launches). `WARP.first_instance`,
`CTAS.first_warp`, `INST.{template,semantics,memory_access}`,
`TMPL.{source,destination,timing}_begin`, `MACC.{sector,timing}_begin`,
`SPLT.semantics`, `MBPI.cta_index`, `CODE.instance`,
`CLKM.{warp,instance}` are rebased by the summed row counts of earlier
segments. USES rows with per-instance targets (0/1/2/3/8) are kept from
every segment with `target_index` rebased; global-target USES and the
PROG/NODE/GRD/LEAF/PARM/PTS/AXIS group are taken from segment 0 after
checking all segments share identical META digests. STRS and FACT start
with the shared program-table prefix in every segment and carry a
per-segment tail (interned opcode / class strings, per-instance FACT rows):
segment 0 is taken whole; for each later segment the longest common STRS
prefix (at a NUL boundary) is reused and the tail appended, the FACT prefix
= rows before the smallest `fact_begin` of its per-instance USES rows
(must equal segment 0's rows) and the tail appended; `TMPL.operation_class`,
tail `FACT.axis` and `USES.fact_begin` of that segment are rebased
accordingly (a FACT range straddling the prefix is a format error).

## 5. Instance semantics kinds (SEMA.kind → field[0..5])

Field names per `python/cyclescope/job/semantics.py`. `generation` keeps the
mbarrier encoding: bit 63 = unresolved flag, bit 0 = parity.
`mbarrier_wait.field1` packs generation|parity; `object_list_begin|count`
packs high|low u32 in one slot; `wgmma.field1` packs
accumulator_registers_begin|count likewise.

| kind | fields |
|---|---|
| none | - |
| mbarrier_init | object, expected_arrivals, scope (0 cta/1 cluster), initial_phase (0/1, default 0; Hopper: SYNCS.EXCH.64 init value high word bit 31, Blackwell: decoded from the init value, converter-filled), -, object_list_begin\|count (a lane-parallel store promoted to init initialises every listed object, e.g. CUTLASS one-STS.64-per-stage-barrier init) |
| mbarrier_arrive | object, generation, transaction_bytes, arrival_count, destination_mask, object_list_begin\|count. Per-lane barrier words that differ across lanes (`mbarrier_words` operand fact; CLC response arrive: lane i -> CTA i) become destination_mask + SPLT credit (lanes per rank) on the issuer's object when the CTA-local offset is shared, an MBOL list otherwise. `SYNCS.*.RED` words are shared::cluster (untagged = rank 0); non-RED SYNCS words are the issuer's own. flags bit8 (0x100) = `.A<n>TX` complete_tx: `arrival_count` 0 and `transaction_bytes` complete expected tx instead of arriving/expecting (nvjet posts the 16-byte CLC response this way when CLC is unused; `.A<n>TR` remains expect_tx) |
| mbarrier_expect_tx | object, generation, transaction_bytes, scope |
| mbarrier_wait | object, generation\|parity, block_offset, scope, timeout_hint, after_ready_mask (flags bit0 = known) |
| mbarrier_inval | object |
| cta_barrier_arrive / cta_barrier_wait | barrier, expected_threads (0 = whole CTA), generation; BAR.SYNC / BAR.RED are cta_barrier_wait with flags bit1 = also arrives. Operands: immediates as-is; a register id is the captured value's low 16 bits and a register count next to a register id its high 16 bits (`BAR.SYNC R0, R0` packs `(count << 16) \| id`), a register count next to an immediate id is the raw value; no count = 0: every non-exited CTA warp (engine deducts exited warps); a register without a captured value is `missing_fact:cta_barrier_operand` |
| cluster_barrier_arrive / cluster_barrier_wait | generation, destination_mask |
| tma_load | object, generation, multicast_mask, shared_bytes, global_bytes, tensor_map. `shared_bytes` = box bytes (`tile::gather4` / `scatter4`: 4 box rows); flags bit7 (0x80) = `.2CTA` (cta_group::2): the engine posts each destination rank's bytes on its pair leader (rank & ~1). The mbarrier word of TMA / bulk / UTCBAR / CLC operands is a shared::cluster address whose owner is the rank tag (bits 24+), rank 0 when untagged |
| bulk_group_issue / commit / wait | group, pending_allowed, bytes (issue only); the PTX `cp.async.bulk` async-group: UTMASTG / UTMAREDG / S2G UBLKCP / UBLKRED issue, UTMACMDFLUSH and UTMACCTL.COMMIT (both architectures; UTMACCTL.IV is descriptor_visibility_acquire, MEMBAR.ALL.GPU descriptor_visibility_publish) commit, DEPBAR.LE SB0,N waits; DEPBAR implies a commit when open issues exist (SEMA.flags bit2 = implicit commit) |
| async_copy_issue / commit / wait | group, pending_allowed, object, arrival_count, transaction_count (commit only; both 0 on LDGDEPBAR; ARRIVES.LDGSTSBAR: .TRANSCNT → arrival_count 0 + transaction_count = lanes, else arrival_count = lanes) |
| bulk_copy / bulk_reduce | object, generation, bytes, multicast_mask |
| warpgroup_arrive / wait | group (per-warp ordinal / committed-unit count), pending_allowed (DEPBAR keep; missing immediate -> missing_fact:warpgroup_wait_keep, non-gsb0 -> unsupported:warpgroup_scoreboard, keep>7 -> unsupported) |
| wgmma (hopper) | group, accumulator_registers_begin\|count, shape_packed (m<<32\|n<<16\|k), input_dtype, accumulator_dtype, source_mode (1 ss/2 rs); folded opener -> flags GROUP_OPENER, gsb0 operand -> GROUP_COMMIT |
| umma_issue / umma_commit (blackwell) | issue: kind (1 f16/2 tf32/3 f8f6f4/4 mxf8f6f4/5 mxf4/6 mxf4nvf4), cta_group, a_source (1 smem/2 tmem), tensor_m\|tensor_n, descriptor (idesc captured word); commit: object, cta_group, commit_mask, scope |
| tmem_alloc / dealloc / relinquish | columns (captured UR mapped via 32-col granularity; relinquish 0; dealloc with no captured value -> SEMA_UNKNOWN, engine resolves per tcgen05.dealloc = free the CTA\u2019s held allocation; alloc still missing_fact) |
| tmem_load / store | shape enum (1 16x64b/2 16x128b/3 16x256b/4 32x32b, 5 16x32bx2 half (16dp32bit_t0_t15 / _t16_t31); dp-bit spellings map to same codes; bare xN and xN.MAX.* = shape 4 + repeat N), repeat; LDTM.STAT -> flags STAT |
| tmem_copy | UTCCP only: shape = (N<<16)\|M packed from Nx32dpMbit or single-block <D>dp<M>bit, repeat 1 |
| tmem_wait_store | (none; FENCE.VIEW.ASYNC.T) |
| clc_query (blackwell) | object, generation, destination_mask, response_identity, outcome |
| shared_store_async | object, bytes, destination_rank, object_list_begin\|object_list_count |
| global_sync | observation_kind (1 release/2 acquire/3 atomic), class (1 atom/2 red/3 ld_strong) — emitted for every GSYN-observed ld_strong / atom / red (st_strong is a producer only, no row); the engine keys the acquire wait and the gate before an atomic on this row, pairing itself stays engine-side (GSYN table) |
| reconvergence_start / complete | region, envelope_cycles |
| nanosleep | requested_ns; flags bit6 (0x40) = `.SYNCS` (wait set = every known-failed poll object in its poll region — the span between two non-failed mbarrier waits — plus the region-closing wait's (object, parity); any already released -> no sleep; sleep duration capped at 1 ms per PTX ISA) |
| descriptor_visibility | operation (1 publish/3 error_check/4 acquire/5 proxy_fence; store registers candidates only, emits no row), descriptor_count, descriptor_address_0..3 |
| clock_marker | kind |
| dependents_launch | (none; PREEXIT executed = griddepcontrol.launch_dependents; the engine charges the dispatch serializer at CTA completion) |

### 5.1 Field sources

| field | source |
|---|---|
| object | mbarrier shared address from the instruction's memory-reference lane addresses, else explicit `mbarrier_address` operand fact (Blackwell UTMALDG.* recovered mechanically from the encoded object id of the captured UR value); `(cta_index, address)` → object index; absent object packs as `SEMA_NO_OBJECT` = 0xFFFFFFFF (job-format §2 "none" sentinel), unpacks back to None |
| expected_arrivals / arrival_count / expected_threads | `barrier_operands` immediate token (register token → 0 pending operand-value sidecars) |
| transaction_bytes | arrive.expect_tx operand (`opN` immediate fact) |
| parity | `mbarrier_phase_operand.value` (bundle payload fact) |
| generation | always `GENERATION_UNRESOLVED` (bit63); the engine resolves phases |
| after_ready_mask | `after_ready_mask` operand fact; SEMA.flags bit0 set when present |
| destination_mask / multicast_mask | per-lane `getctarank` rank facts OR'ed |
| barrier | named-barrier id operand (`op0` immediate) |
| group | per-warp sequential async-group counter; commit closes the group |
| pending_allowed | wait immediate (`DEPBAR`/wait_group operand) |
| bytes | instruction size operand (`opN` immediate), else TMA box bytes |
| tensor_map | tensor-map fact id; shared/global_bytes = box_dim product × element_bytes |
| descriptor operation/count/address_N | publish: complete 128-byte windows (8 x 16B STG chunks) sorted by base, count=N (0 -> no row, >4 unsupported); acquire: single base operand fact; error_check/proxy_fence: count=0 |
| pending_allowed / depbar_keep | `DEPBAR.LE SB0,N` trailing immediate token (barrier_operands / operands); only SB0 FIFO counts, N<=7 always, N<=63 requires prior LDGDEPBAR/bulk-flush evidence in the same warp |
| requested_ns | NANOSLEEP immediate |
| scope | 0 = CTA, 1 = cluster when remote-shared addressing/rank facts exist |
| object_list (MBOL) | per-active-lane distinct mbarrier addresses for divergent arrives |
| missing facts | `missing_fact:<operand> (<opcode> @pc)` at decode; the builder emits `INST.semantics = 0xFFFFFFFF` and counts it when the bundle lacks the value |

## 6. Timing-row kinds (TIME.kind)

One mechanism program corresponds to one kind.

| id | name | mechanism | used by |
|---|---|---|---|
| 1 | `global_load_dependency` | — | — |
| 2 | `global_store_acceptance` | — | — |
| 3 | `shared_store_visibility` | `storage.shared.visibility` | both |
| 4 | `global_store_visibility` | — | — |
| 5 | `shared_issue_interval_smsp` | `storage.shared.issue` | both |
| 6 | `shared_store_acceptance_sm` | `storage.shared.acceptance_sm` | both |
| 7 | `dependency_result_smsp` | — | — |
| 9 | `tma_verified_inflight_domain` | — | — |
| 10 | `tma_overlap_operation` | — | — |
| 13 | `hopper_wgmma_handoff_consumer` | `hopper.wgmma.accumulator_handoff.consumer` | Hopper |
| 15 | `atomic_issue_interval_sm` | `storage.atomic.issue` | both |
| 16 | `nanosleep_warp_delay` | `control.nanosleep` | both |
| 17 | `cta_barrier_release` | `synchronization.cta.release` | both |
| 18 | `async_copy_mbarrier_arrival` | `synchronization.cp_async.arrival` | both |
| 19 | `tensor_map_descriptor_visibility` | `synchronization.descriptor.visibility` | both |
| 20 | `lifecycle_contract` | — | — |
| 21 | `service_start_interval` | `storage.async.service_start` | both |
| 22 | `mbarrier_phase_release` | `synchronization.mbarrier.release` | both |
| 29 | `hopper_qgmma_fmnmx_handoff_consumer` | `hopper.wgmma.fp8_fmnmx_handoff.consumer` | Hopper |
| 31 | `tma_overlap_operation_signature` | — | — |
| 32 | `tma_store_background_drain_extension` | — | — |
| 38 | `tma_operation_completion` | `storage.tma.completion` | both |
| 39 | `storage_ldgsts_group_completion` | `storage.ldgsts.completion` | both |
| 42 | `cluster_barrier_release` | `synchronization.cluster.release` | both |
| 43 | `hopper_tma_descriptor_refetch` | `storage.tma.descriptor_refetch` | both |
| 44 | `hopper_wgmma_register_file_issue_steal` | `hopper.wgmma.register_file_issue_steal` | Hopper |
| 51 | `blackwell_umma_commit` | `blackwell.umma.commit` | Blackwell |
| 52 | `blackwell_tmem_load` | `blackwell.tmem.load` | Blackwell |
| 53 | `blackwell_tmem_store` | `blackwell.tmem.store` | Blackwell |
| 54 | `blackwell_tmem_alloc` | `blackwell.tmem.alloc` | Blackwell |
| 55 | `blackwell_tmem_copy` | `blackwell.tmem.copy` | Blackwell |
| 56 | `blackwell_clc_query` | `blackwell.clc.query` | Blackwell |
| 57 | `blackwell_tmem_load_stat` | `blackwell.tmem.load_stat` | Blackwell |
| 59 | `blackwell_umma_pair_arrive_skew` | `blackwell.umma.commit.pair_skew` | Blackwell |
| 61 | `mbarrier_trywait_timeout` | `synchronization.mbarrier.trywait_timeout` | both |
| 63 | `tensor_pipe_beat` | `tensor.pipe` | both |
| 64 | `tensor_result_latency` | `tensor.completion` | both |
| 65 | `global_store_issue_interval_smsp` | `storage.global.store_issue` | both |
| 66 | `shared_fence_drain` | `storage.shared.fence` | both |
| 80 | `shared_store_acceptance_smsp` | `storage.shared.acceptance_smsp` | both |
| 81 | `tma_reduce_store` | `storage.tma.reduce` | both |

Source of the list: `TimingKind` in `python/cyclescope/job/timing_rows.py`.
A kind with "—" is defined there but no program in the shipped trees
uses it.

## 7. Result `GSIMRES` / `GSIMJNL` / `GSIMWLG`

RSLT (one row): cycles, issued, completed, hard_constraint_failures u64,
flags u32. SMSP: sm, smsp u32, first_active, last_active u64. WAIT:
scope, cause_kind u32, cycles u64. RWAIT: resource, cause_kind u32,
cycles u64. TELE: group, counter u32, value u64.
JRNL (GSIMJNL): warp, smsp, instance u32, issue_cycle u64,
operation_class, template u32. WLDG (GSIMWLG): warp, cause_kind u32,
cycles u64, count u32.

### Facts files (`GSIMCAP`)

The `convert --capture` step writes per-cluster facts files in the
same shell. One table worth knowing: `MREX`, one extension row per
memory reference `transfer_bytes, addr_begin, addr_count, cut_rows`
(20 B; `cut_rows` is the in-bounds row count when a tiled TMA load's
dimension-0 box is partially out of bounds, else 0/-1). An MREX table
with a different record width is rejected as
`missing_fact:mrex_record_bytes`.
