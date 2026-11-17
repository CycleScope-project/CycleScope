# Microbenchmarks: measuring mechanism facts

[中文](../zh/microbench.md)

A mechanism leaf is measured on its own: a small kernel where the
mechanism is the only cost that varies — sweep the axis or the
repetitions and read the slope, so fixed overheads cancel. Cluster-local
mechanisms are measured with one SM active, gpu-shared ones with the
card loaded. What a mechanism and a leaf are:
[design/mechanism-tree.md](design/mechanism-tree.md).

## 1. A protocol directory

Each probe lives in `tools/microbench/<name>/` (not every directory
has every file):

| file | role |
|---|---|
| `protocol*.json` | the protocol, written before measuring: what is measured, the cells, the criterion |
| `probe/` | the CUDA kernels under test |
| `build.sh` / `run.sh` | compile and run the probe (`run_probes.sh`, `run_ncu.sh` variants exist) |
| `model.py` | turns raw timings into the protocol's statistic |
| `sass_check.py` | checks the compiled SASS is what the probe intends |
| `install.py` / `table*.py` | apply the result to the tree fragments |

## 2. The life of a measurement

1. Write the protocol first — cells, controls and the criterion are
   fixed before the probe runs.
2. Compile and check the SASS is the measurement the protocol
   intends.
3. Run on each device, twice.
4. Apply the protocol's criterion to the results.
5. Install the value with its evidence level, or leave the leaf
   unchanged and record why.

## 3. Running one locally

```bash
env CYCLESCOPE_ARTIFACT_ROOT=<artifact root> \
    PROTOCOL=<name> DEVICE=h100|b300 RUN=<tag> \
    [CELL=<subdir under tools/microbench>] [OUT_ROOT=<dir>] \
    [NCU=<ncu>] [TIMEOUT=3600] \
    bash tools/microbench/run_local.sh
```

The script runs `run.sh` (else `run_probes.sh`, else `run_ncu.sh` with
`NCU`) inside `tools/microbench/<dir>` — `<dir>` is `CELL` when set,
else `PROTOCOL`. Output goes to
`$CYCLESCOPE_ARTIFACT_ROOT/microbench/<dir>/<device>-<run>` (or
`OUT_ROOT`). Everything runs under `timeout`; `TREE` and `SM_ARCH`
are set for the probe, and `WGMMA_*`, `UMMA_*`, `V2`, `V3`,
`SASS_CHECK`, `RUNS`, `CELLS`, `SKIP_BUILD`, `KINDS`, `PROBE`, `SET`
pass through. Success prints `CELLS_DONE`; a failing cell prints
`CELL_FAIL`.

## 4. Catalogue

One row per directory under `tools/microbench/`, alphabetical. The
last column lists the mechanisms that hold a value citing a source
from that directory, in either device's tree. "—" means no installed
value cites it: the measurement confirmed what was already installed,
informed a rule in the engine, or was not installed.

| directory | what it measures | mechanisms with values from it |
|---|---|---|
| `alu_forms` | per-instruction cost of integer ALU forms | `instruction.service` |
| `arrive_status` | when the state an `SYNCS.ARRIVE` returns becomes usable | `instruction.service` |
| `atomic_issue` | per-SM issue serialization of atomics | `storage.atomic.issue` |
| `barrier_release` | release latency of mbarrier / CTA / cluster / cp.async barriers | `instruction.service`, `synchronization.cluster.release`, `synchronization.cp_async.arrival`, `synchronization.cta.release`, `synchronization.mbarrier.release` |
| `constant_launch` | constant-cache miss cost of first traversal | `memory.constant_cache` |
| `control_branch` | cost of control-flow branches | `control.branch` |
| `control_word` | latencies of fixed-latency instruction classes, derived from the control words the compiler emits | — |
| `control_yield` | issue floor a `YIELD` gives the warp | `instruction.service` |
| `crossings` | a collection of small probes: constant cache, CTA barrier release, descriptor refetch, L1 bandwidth and residency, reconvergence, store port, try-wait timeout | `control.reconvergence`, `memory.constant_cache`, `memory.dram.service`, `memory.l1.residency`, `memory.l1.service`, `memory.l2.service`, `memory.store_dram.service`, `memory.store_l2.service`, `storage.tma.descriptor_refetch`, `synchronization.mbarrier.trywait_timeout` |
| `descriptor_visibility` | publish → acquire visibility delay of async descriptors | `synchronization.descriptor.visibility` |
| `dispatch` | per-CTA launch dispatch serialization | `launch.dispatch` |
| `dispatch_contention` | issue contention under dispatch (register-file operand reads, fma issue) | `instruction.service`, `register_file.operand_read` |
| `dispatch_refill` | how long a freed CTA slot stays empty | `launch.dispatch` |
| `dsmem` | cost of remote shared-memory access inside a cluster | `storage.shared.acceptance_sm`, `storage.shared.issue`, `storage.shared.visibility` |
| `ffma2_operands` | FFMA2/FADD2/FMUL2 throughput on Blackwell | — |
| `fp32_mix` | FP32 streams interleaved with MUFU on one SMSP | — |
| `fp_stream_mix` | FP32/MUFU mixes at several warps per SMSP | — |
| `gather_hit` | wait for back-to-back gather loads on one row | `memory.load_port` |
| `gather_latency` | load latency when many warps and SMs gather from a row | — |
| `global_atomic` | release-to-acquire delay of global atomics | `memory.global_atomic` |
| `ifetch` | L0/L1 instruction-fetch misses and prefetch behaviour | `instruction.fetch` |
| `l2_merge` | merging of reads of the same L2 miss | `memory.l2_merge` |
| `ldgsts_l1` | which cp.async fills allocate in the SM's L1 | — |
| `ldgsts_loaded` | L1/L2 service under cp.async load | `memory.l2_merge` |
| `lifecycle` | the launch envelope: entry, exit, drain | `kernel.lifecycle` |
| `load_after_copies` | how non-bypass async copies displace local lines from L1 | `memory.l1.service` |
| `load_after_stores` | cost of a stack reload waiting on stores | `memory.store_l2.service` |
| `load_lines` | load cost when lanes touch several L1 lines | — |
| `load_port_lines` | whether the SM load port charges per sector or per line | — |
| `lsu_accept` | acceptance queue and store issue in front of the memory hierarchy | `storage.global.store_issue` |
| `math_occupancy` | per-instruction cost when warps interleave two classes | `instruction.service` |
| `memlat` | load-to-use latency at each cache level | `memory.l1.service`, `memory.l2.service` |
| `memlat_loaded` | L2/DRAM latency under device-wide load | `memory.dram.service`, `memory.l2.service` |
| `memory_request` | saturated per-SM request rates per level, and stores | `memory.dram.capacity`, `memory.l1.capacity`, `memory.l2.capacity` |
| `mma_sync` | warp-level mma.sync class timing | `instruction.service` |
| `mufu_rcp` | throughput of the `MUFU` reciprocal | — |
| `nanosleep` | NANOSLEEP duration and `.SYNCS` wake behaviour | `control.nanosleep` |
| `register_operands` | how register-source forms change the issue interval | — |
| `row_gather` | load latency as a function of DRAM rows per warp | `memory.dram.service`, `memory.dram_partition` |
| `service_latency` | instruction.service latency fields per class | `instruction.service` |
| `shared` | shared-memory load/store cost, wavefronts, fences | `instruction.service`, `storage.shared.acceptance_sm`, `storage.shared.acceptance_smsp`, `storage.shared.fence`, `storage.shared.issue`, `storage.shared.visibility` |
| `shfl_scope` | SHFL issue scope (SMSP vs SM) | `instruction.service` |
| `store_flood` | saturated store cost per SM and device-wide | `memory.store_l2.capacity`, `memory.store_l2.service` |
| `store_hold` | when a following instruction may overwrite store data | — |
| `store_scatter` | store issue interval by width vs generated lines | `storage.global.store_issue` |
| `tensor_pipe` | per-SM tensor pipe beat and completion | `tensor.completion`, `tensor.pipe` |
| `tensor_pipe_forms` | the same beats across MMA forms | `tensor.completion`, `tensor.pipe` |
| `tma_cold` | TMA completion hot vs cold | `storage.tma.completion` |
| `tma_forms` | TMA service/completion across instruction forms | `storage.async.service_start`, `storage.ldgsts.completion`, `storage.tma.completion` |
| `tma_loaded` | L2→SM delivery under device-wide TMA load | — |
| `tma_multicast` | multicast tile-load delivery to destination CTAs | — |
| `tma_oob` | extra latency of a TMA load partly out of bounds | `storage.async.service_start`, `storage.tma.completion` |
| `tma_store_loaded` | TMA store timing and shared store/L2 port contention | `memory.shared_service`, `memory.store_l2.service`, `storage.tma.reduce` |
| `tmem` | issue interval of tensor-memory loads (`LDTM`) | `blackwell.tmem.load`, `instruction.service` |
| `tmem_pipeline` | TMEM copy/load pipeline behaviour | `blackwell.tmem.copy`, `tensor.pipe` |
| `trywait_wake` | wake latency after a failed try_wait | `synchronization.mbarrier.release` |
| `wgmma_handoff` | accumulator handoff from wgmma to consumer | `hopper.wgmma.accumulator_handoff.consumer` |
| `wgmma_overlap` | issue cost while a wgmma is in flight | `hopper.wgmma.register_file_issue_steal` |

## 5. Writing a new probe

- Write the protocol first; measure only afterwards.
- Always check the compiled SASS — the compiler will rewrite an
  ill-placed timed window into something the protocol did not intend.
- Isolate one mechanism: make it the only cost that varies across the
  cells, and keep controls (hot path, no-op variant) in the same
  kernel.
- Sweep repetitions or the axis and read the slope; a single point
  mixes fixed overhead into the measured value.
- Pick the measurement shape by ownership: cluster-local with one SM
  active, gpu-shared with the card loaded.
