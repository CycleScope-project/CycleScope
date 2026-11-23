# The engine binary

[中文](../zh/engine.md)

`cyclescope-engine` is the cycle-level simulator of one launch. Its
input is a job container (one segment per cluster, plus the
launch-level tables) and its output is an execution record: the total
cycle count plus, on request, a row per instruction. This page is the
interface description; [design/engine.md](design/engine.md) explains
the model.

## 1. Command line

```bash
cyclescope-engine simulate --job job.bin --out result.bin \
    [--journal j.bin] [--wait-ledger w.bin] [--stall-ledger stall.tsv]
    [--progress]
cyclescope-engine validate --job job.bin   # structural check
cyclescope-engine info     --job job.bin   # segment/instance statistics
```

stderr ends with `LOAD=` / `RUN=` (seconds) and
`cycles=/issued=/completed=/hard=`.

## 2. What it models

A short tour; each item is one or two lines, the full picture is in
[design/engine.md](design/engine.md).

- **The compiler's schedule**: an instruction waits for the registers
  it reads and for the scoreboards its control word names, and
  respects the control word's stall count.
- **Barriers**: mbarriers, CTA named barriers and cluster barriers are
  state machines; arrivals and transferred bytes decide when a phase
  completes and its waiters resume.
- **Asynchronous copies and TMA**: copies are grouped per warp; a
  group is committed and waited on, and a transfer reports completion
  through its mbarrier.
- **Memory hierarchy**: per-SM L1, shared L2 and DRAM, tracked per
  cache sector, with ports and queues. Every launch starts with empty
  caches.
- **Execution units**: an instruction occupies the unit its leaf
  names, for the interval its leaf gives.
- **Tensor cores**: Hopper WGMMA and Blackwell UMMA queue on a per-SM
  tensor pipeline.
- **Thread-block dispatch**: thread blocks enter according to the
  captured placement; later ones enter as slots free up.
- **Instruction fetch**: code is served from per-SM instruction caches
  with prefetch, empty at the start of a launch.
- **Per-warp queue** (`instruction.warp_queue`): an issued STS, LDS,
  SHFL, LDG or STG is an entry of its warp's queue until a serial
  release server frees it; a warp at the queue's depth limit cannot
  issue.
- **Kernel boundary**: `cycles` is the largest count of active cycles
  of any SM — the union of its warps' residency intervals
  [entry, done); the former last-completion-plus-envelope value is
  telemetry `last_completion_cycle`.
- **Architecture modules**: warpgroups and WGMMA on Hopper, UMMA,
  tensor memory and CLC on Blackwell, behind one interface.

## 3. Output

`simulate --out` writes `GSIMRES`: the totals (`cycles`,
`issued`/`completed`, hard-constraint failures), per-SMSP active
spans, wait cycles per bucket, per-resource waits, and the telemetry
counters. `--journal` and `--wait-ledger` add the per-instruction
journal and the per-warp wait ledger
([results.md](results.md), layouts in [formats.md](formats.md) §7).

Telemetry rows are `(group, counter, value)`:

| group | what it counts |
|---|---|
| 0 | engine totals (issued instances) |
| 1 | synchronization: mbarrier phases/arrivals/tx bytes, barrier generations, polls, gsync, TMA completions |
| 2 | dispatch, lifecycle and fetch: entries, waves, wait cycles, fetch stalls, per-CTA issue spans |
| 3 | architecture-module counters (`wgmma_*`, `umma_*`, `tmem_*`, …) |
| 4 | memory hierarchy: L1/L2/DRAM sectors, merges, fills, evictions, shared-service waits, port shares, phantom sectors |

Counter names are the `SyncTele` / `EngineTele` enums in
`cpp/core/simulator.hpp` (group 4 order is fixed in
`cpp/core/simulator.cpp`).

## 4. Stall ledger

`--stall-ledger <path>` writes a tab-separated per-PC stall ledger
(schema `cyclescope.stall-ledger.v1`). It is statistics-only: a run
with the option produces a byte-identical `result.bin` and journal,
and no timing decision reads ledger state.

```
#schema	cyclescope.stall-ledger.v1
#warps	<warps that became resident>
#cycles	<kernel cycles, the same value the result reports>
residency	<code address hex>	<operation_class>	<reason>	<warp_cycles>	<events>
transition	<from code address hex>	<to code address hex>	<count>
```

Every cycle a resident warp is evaluated for issue is charged to
exactly one `(code address of its next instruction, reason)`; only
the issue cycle itself is charged `issued`, and cycles before the
warp is resident are `dispatch`. Cycles in which a warp is ready and
waits only because its SMSP's issue slot went to another warp are
`issue_slot`. Reasons are `<site>[:<detail>]` assigned at the verdict
site: `barrier:<op>[:<detail>][:read]` (a wait on a producer's
dependency barrier), `unit:<name>` / `unit_outbid:<name>` (unit names
of `configs/mechanism_tree/common/resources.json`), `semantic:<kind>`,
`suspended:<kind>`, `arch:<module rule>`, `warp_queue:mio` /
`warp_queue:lg` (the per-warp queue), the control floors
`control:branch` / `control:yield` / `control:bsync`, and one site
name per remaining verdict site (`fetch`, `cta_barrier`,
`cluster_barrier`, `control_floor`, `issue_slot`, `exit_branch`,
`exit_drain`, `other:<site>`, …). `transition` rows count, per warp,
each pair of consecutively issued instructions (sequential
fall-through included).

Per-warp conservation:

    resident_cycles == (last_issue_cycle - first_resident_cycle + 1)
                       + exit_drain_cycles

## 5. Failures

Two data failure classes, the same as the rest of the pipeline:
`missing_fact:<fact> (<where>)` when the job lacks a fact the model
needs, and `unsupported:<reason> (<where>)` when the model does not
cover it. A run that cannot progress stops with
`unsupported:engine:deadlock` and a dump of the stalled warps and
synchronization objects.

## 6. Benchmarking the engine

```bash
OUT=<dir> [SIZES="S L"] [PROFILE=1] bash scripts/pipeline/engine_perf.sh
```

Generates synthetic jobs (S ≈ 256 k instances, L ≈ 8.4 M), builds a
Release engine, times `simulate`, and writes `perf.tsv`
(wall/RSS/cycles/cycles-per-second) plus an optional gprof profile.

```bash
scripts/pipeline/engine_fixture_digest.sh <engine> <fixtures_dir> <out.tsv>
scripts/pipeline/engine_fixture_digest.sh --compare <a.wrk> <b.wrk>
```

Hashes the result and journal binaries of every `*.job` fixture;
`--compare` diffs two such work directories table by table (every
table must match except WAIT).

## 7. Debug switches

`CYCLESCOPE_DEBUG_*` environment variables print diagnostics to
stderr; each is off unless set:

| variable | what it prints |
|---|---|
| `CYCLESCOPE_DEBUG_INST=<instance>` | readiness checks and writer-table updates for one instance |
| `CYCLESCOPE_DEBUG_MBARRIER=<seg>:<obj>` | every state change of one mbarrier |
| `CYCLESCOPE_DEBUG_CTABAR=<cta>:<barrier>` | every state change of one CTA named barrier |
| `CYCLESCOPE_DEBUG_CPASYNC` | cp.async issue and wait decisions |
| `CYCLESCOPE_DEBUG_TMA` | TMA completion timing detail |
| `CYCLESCOPE_DEBUG_TMEM` | TMEM alloc/dealloc attempts (Blackwell) |
