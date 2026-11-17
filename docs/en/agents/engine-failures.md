# Engine failures

[中文](../../zh/agents/engine-failures.md)

The simulator itself reports a failure. The terminal class lives in
`sim/<case>/status.json` (`sim` stage); the thrown string and the
parked-warp report are in `sim.log` and under `out-<builder>/`.
Classes below are the `EngineError` strings thrown in `cpp/core/`,
`cpp/arch/<arch>/` modules, and `cpp/main.cpp`. Conversion-time
classes (`ConversionError`, Python `convert/`) are in
[convert-failures.md](convert-failures.md).

## 1. `unsupported:engine:*` thrown by the engine

| class (file) | meaning / first check |
|---|---|
| `unsupported:engine:deadlock` (`simulator.cpp`) | no warp can make progress; stderr prints a `deadlock:` report first — see §3 |
| `unsupported:engine:discrete_miss` (`mechanism_runtime.cpp`) | a discrete leaf was evaluated at an axis value with no row — see [missing-leaf.md](missing-leaf.md) |
| `unsupported:engine:discrete_axis_id` (`mechanism_runtime.cpp`) | a discrete-axis selector row resolves to an id outside the axis domain — tree fragment vs `axes.json` mismatch |
| `unsupported:engine:unknown_axis` (`mechanism_runtime.cpp`) | a mechanism names an axis that does not exist in the tree |
| `unsupported:engine:parm_model` (`mechanism_runtime.cpp`) | a leaf's parameter model kind is unknown to the runtime — schema/tree inconsistency |
| `unsupported:engine:no_reference_clock` (`mechanism_runtime.cpp`, `simulator.cpp`) | a field needs ns→cycle conversion but the job carries no `reference_sm_clock_khz` |
| `unsupported:engine:use_index` (`mechanism_runtime.cpp`) | a USES index references a capture-facts block the job does not have |
| `unsupported:engine:resource_state` (`simulator.cpp`) | a resource's in-flight completions cannot free a slot — the resource model dead-ended |
| `unsupported:engine:fetch_line_bytes` (`simulator.cpp`) | fetch mechanism enabled with `line_bytes == 0` — leaf value missing/zero |
| `unsupported:engine:operand_read_collector` (`simulator.cpp`) | an operand-read collector id the runtime does not know |
| `unsupported:engine:wait_block_offset` (`simulator.cpp`) | a deferred mbarrier wait would block past `end_instance` — segment/fact inconsistency |
| `unsupported:engine:mbarrier_uninitialized` (`sync_objects.cpp`) | arrive/wait on an mbarrier object never initialised |
| `unsupported:engine:mbarrier_reinit` (`sync_objects.cpp`) | re-init of an mbarrier that still has pending arrivals/tx/waiters |
| `unsupported:engine:mbarrier_overrun` (`sync_objects.cpp`) | more arrivals than the phase's expected count |
| `unsupported:engine:cta_barrier_overrun` (`sync_objects.cpp`) | CTA barrier arrives exceeded its generation's count |
| `unsupported:engine:inter_cta_global_sync:<detail>` (`global_sync.cpp`) | a global-sync polling pattern whose semantics conflict or cannot be resolved |
| `unsupported:l2_merge_domain` / `unsupported:l2_merge_domain_gpc` (`simulator.cpp`, `memory_hierarchy.hpp`) | computed L2 merge-domain id out of representable range — topology/config inconsistency |
| `unsupported:umma_kind` (`arch/blackwell/umma.cpp`) | an UMMA issue kind bit the engine does not implement (the same string is also thrown at conversion time by `convert_rules.cpp`) |

## 2. `missing_fact:*` thrown by the engine

Topology/launch shape: `sm_topology`, `sm_topology.gpc`, `sm_count`,
`launch_block_dims`, `occupancy_blocks_per_sm` — all in
`simulator.cpp`; the job lacks the placement/topology facts a
full-GPU run needs.
Program/service resolution: `instruction.service` (`simulator.cpp`)
— an opcode has no service class in the tree;
`missing_fact:mechanism_leaf`, `missing_fact:axis:<name>`,
`missing_fact:axis:tensor_k` (`mechanism_runtime.cpp`,
`arch/blackwell/umma.cpp`) — a leaf or axis a mechanism needs is not
in the job.
Synchronisation objects: `mbarrier_object_table` (`simulator.cpp`,
an object id cannot be resolved to a rank/address through the MBOB
table), `mbarrier_init_count`, `mbarrier_arrive_count`
(`sync_objects.cpp`, `simulator.cpp` — an arrive/wait lacks the
init/expected-count fact), `global_counter_initial_value`
(`global_sync.cpp` — a global-sync counter with no recorded initial
value).
Data movement: `tma_transfer_bytes` (`simulator.cpp` — a TMA
completion carries no byte count), `tmem_columns`
(`arch/blackwell/tmem.cpp` — a tmem alloc without its column count).

## 3. Wait buckets

The journal / wait ledger attributes stalled cycles per warp to
`WarpState::last_wait` / `block_bucket`, an enum in
`cpp/core/simulator.hpp`:

| bucket | charged when (where set) |
|---|---|
| `kDispatch` | warp is dispatch-throttled (`simulator.cpp` dispatch path) |
| `kFetch` | waiting on an instruction-fetch line |
| `kCollective` | collective synchronisation: mbarrier phase wait, cp.async/TMA group wait, `nanosleep` wake, barrier waits |
| `kSemantic` | semantic waits: global-sync polling windows, scoreboard/semantic dependencies |
| `kResource` | an execution unit or shared resource is busy (unit waiters, register file ports, resource backpressure) |
| `kRegister` | register dependency (scoreboard/register-ready retry) |
| `kScheduler` | default/scheduler — no specific wait recorded |

The ledger is per-warp per-stall; the journal (`--journal`) records
instruction-level events. Telemetry counters (`tele_[]` in
`simulator.cpp`, e.g. `kTeleGsyncWaitCycles`, `kTeleNanosleeps`)
aggregate named wait kinds — read `simulator.cpp` for the full list.

## 4. Debug switches (`getenv` in `cpp/`)

`CYCLESCOPE_DEBUG_TMEM` (Blackwell tmem alloc), `CYCLESCOPE_DEBUG_TMA`,
`CYCLESCOPE_DEBUG_CTABAR`, `CYCLESCOPE_DEBUG_MBARRIER`,
`CYCLESCOPE_DEBUG_CPASYNC`, `CYCLESCOPE_DEBUG_INST` (arg = an
instance index; enables diagnostics for that instance). Set them on
`cyclescope-engine simulate`.

`cyclescope-engine` subcommands: `simulate --job <j> --out <r>
[--journal f] [--wait-ledger f]`, `info --job`, `dump --job`,
`validate --job`. `cyclescope run --ledgers --keep` keeps the journal
and wait ledger under `out-<builder>/`. `scripts/pipeline/
compare_results.py` diffs two `result.bin`s.

## 5. Deadlock walk-through

1. `unsupported:engine:deadlock` is preceded by a `deadlock:` report
   on stderr (parked warps, instruction, wait bucket) — read it in
   `sim.log`; re-run with `--journal`/`--wait-ledger` if it is not
   enough.
2. Find the object the warps wait on (mbarrier / CTA barrier /
   cluster barrier / async group / global-sync poll / resource).
3. Check the object's facts: `mbarrier_init_count`,
   `mbarrier_arrive_count` in `facts/launch.json` and the
   corresponding segment rows; for a global-sync poll check the
   `global_counter_initial_value` fact and the
   `inter_cta_global_sync` records.
4. Decide: fact never recorded → capture gap (`tools/tracer/`); fact
   recorded but segment row wrong/missing → converter rule
   (`python/cyclescope/arch/<arch>/rules.py` +
   `cpp/arch/<arch>/convert_rules.cpp`); facts all present and
   plausible → engine rule (`cpp/core/` for shared objects,
   `cpp/arch/<arch>/` for arch modules like `tmem`).
