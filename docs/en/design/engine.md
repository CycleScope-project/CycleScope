# The engine: replaying against the tree

[中文](../../zh/design/engine.md)

The capture says what ran; the mechanism tree says how the hardware
responds. The engine replays each warp's captured instruction stream
against the tree and places every instruction at the earliest cycle
the hardware would have allowed — recording, when it issued late, what
held it back.

## 1. The structural model

"What the hardware allows" needs a structure. The engine assumes a
small one, stated as assumptions and kept to what the reconstruction
requires:

| abstraction | implementing object |
|---|---|
| SM sub-partitions with issue slots and a warp candidate list | the SMSP state in `cpp/core/simulator.{hpp,cpp}` |
| the compiler's schedule: registers, scoreboards, control word | per-warp `reg_ready` / `barrier_free` tables and the template's control fields |
| barriers as state machines | `cpp/core/sync_objects.{hpp,cpp}` |
| asynchronous operation groups | `cpp/core/async_groups.{hpp,cpp}` |
| execution units and shared services | resources resolved through `cpp/core/mechanism_runtime.{hpp,cpp}` |
| memory hierarchy: per-SM L1, shared L2, DRAM, store queues | `cpp/core/memory_hierarchy.{hpp,cpp}`, `cache_policy.hpp` |
| code caches and prefetch | `cpp/core/instruction_fetch.{hpp,cpp}` |
| thread-block dispatch and kernel envelope | `cpp/core/dispatch.{hpp,cpp}`, `lifecycle.hpp` |
| architecture-specific machinery | a module behind `cpp/arch/architecture_module.hpp` |

The structure holds no numbers: every interval, bandwidth, capacity
and delay in it is a leaf. The engine computes no data values and
simulates nothing below a mechanism.

## 2. The main loop

The engine is event driven. `Simulator::run` keeps an event queue
(`Event::Kind`: `kResultReady`, `kReadBarrierFree`, `kResourceFree`,
`kWarpDone`, `kWake`, `kBarrierRelease`, `kDeferredArrive`,
`kTransferComplete`, `kDispatch`, `kArch`) and the current cycle
`now_`. Each round: schedule every event due at `now_`, then step the
SM sub-partitions that can change — one is skipped unless it has a
retry due or an event marked it — then advance `now_` to the nearer of
the earliest due retry and the next event. Time is resolved to the
cycle but jumps from one change of state to the next.

`step_smsp` walks the partition's warps in round-robin order starting
after the last issued warp (or at it again when its instruction did
not yield), and issues as many instructions per cycle as the
`issue_slots` field of the `scheduler.policy` leaf allows. A blocked
warp ends the sweep unless the `try_next_on_block` field says
otherwise.

## 3. Can this instruction issue?

The readiness checks run in a fixed order (`try_issue` and the sweep
in `step_smsp`):

1. **dispatch** — the warp's CTA is not placed yet, or its entry cycle
   has not come;
2. **fetch** — the instruction's code line is not in the code cache;
3. **suspensions** — the warp is parked until a known release cycle
   (`blocked_until`);
4. **collective waits** — deferred mbarrier wait, CTA barrier, cluster
   barrier;
5. **semantics** — what the instruction does to a barrier or group is
   not satisfied (polls, async-group waits, arch-owned semantics);
6. **resources** — its execution unit is busy, a port or issue
   interval has not elapsed, the operand collector is occupied, an
   arch memory-queue hold applies;
7. **registers** — a source register, a scoreboard, or the
   control-word floor has not cleared;
8. **arch module** — the architecture's own readiness.

The first check that fails is the wait's cause, charged to one of
seven buckets until the next verdict:

| bucket | meaning |
|---|---|
| `kDispatch` | waiting for the CTA to be dispatched and the warp to enter |
| `kFetch` | waiting on instruction fetch |
| `kCollective` | waiting on a barrier release or a suspended wake |
| `kSemantic` | waiting on a semantic gate (a poll, an async group, an arch rule) |
| `kResource` | waiting on a busy unit, port, queue or issue interval |
| `kRegister` | waiting on a register value or the compiler's schedule |
| `kScheduler` | not attributed to any check above: the initial cause before a warp's first failed check |

## 4. The compiler's schedule

Static facts settle the first kind of stall. An instruction waits for
the registers it reads and for the scoreboards its control word
names; it also respects the control word's stall count and its yield
bit only rotates the candidate order. A register becomes ready, and a
scoreboard clears, when the instruction that set it delivers its
result — after the latency its leaf gives. The engine infers nothing
about the schedule: it follows what the compiler wrote.

## 5. Other warps

Runtime facts settle the second kind.

- **mbarrier** (`MbarrierTable`): an object tracks expected arrivals,
  pending arrivals, outstanding transaction bytes and phase; the
  captured operands say which object, how many arrivals and bytes, and
  which parity — the generation itself is solved here. When arrivals
  and bytes reach their expected values the phase completes, and
  waiters resume after the measured release delay.
- **CTA named barriers and the cluster barrier**: generations counted
  the same way, per CTA and per cluster
  (`CtaBarrierTable`, `ClusterBarrierTable`).
- **asynchronous groups** (`AsyncGroups`): bulk, `cp.async` and
  warpgroup lanes; a commit closes the open group, a wait allows at
  most N newer groups still pending.
- **synchronization through global memory** (`global_sync.{hpp,cpp}`):
  a read is paired with the write that produced the value the trace
  observed, and it becomes ready once that write completes and is
  visible.

The logical state is exact — it is updated by PTX semantics; only the
durations come from leaves.

## 6. Hardware capacity

Mechanism facts settle the third kind.

- **Memory hierarchy**: each captured access arrives pre-split into
  sectors. The model tracks which sectors reside in a per-SM L1 and in
  the shared, partitioned L2 (with reader-side replicas and dirty
  write-back), which fills are in flight, per-SM ports, and DRAM
  queues; reads of the same miss merge (`memory.l2_merge`). Every
  launch starts with empty caches — the reference condition is
  cold-cache.
- **Execution units and shared services**: an instruction occupies the
  resource its leaf names, for the interval its leaf gives;
  whole-GPU services (L2 and DRAM bandwidth, dispatch) are shared by
  all SMs.
- **Shared memory and async copies**: shared-service acceptance
  per SMSP and per SM, TMA and bulk-copy service start and completion
  (`async_groups`, `memory_hierarchy`).
- **Dispatch and lifecycle** (`dispatch`, `lifecycle`): first-wave
  placement from the captured `PLCE` rows, queued CTAs entering as
  slots free, serial entry spacing, and a per-launch envelope.
- **Instruction fetch** (`instruction_fetch`): per-SM code caches with
  sequential and launch prefetch; an instruction whose code line is
  not resident waits in the fetch bucket.

## 7. Architecture modules

The core never branches on a device. Everything Hopper- or
Blackwell-specific sits behind one interface
(`cpp/arch/architecture_module.hpp`), instantiated per simulation
from the job's device by `cpp/arch/registry.cpp`:

```cpp
virtual bool     instance_ready(Simulator&, WarpState&,
                     const InstructionInstance&, const InstanceSemantics&);
virtual void     on_instance_issue(Simulator&, WarpState&, …, finish);
virtual void     on_event(Simulator&, const Event&);     // incl. Event::kArch
virtual uint64_t memory_queue_retry(sm, smsp, now);
virtual IssueSteal issue_steal(const Simulator&, const WarpState&);
virtual bool     coresident_dependent_penalty(…);
virtual void     on_load(Simulator&);  on_warp_exit(…);  collect_tele(…);
```

The Hopper module adds warpgroup collectives and the WGMMA group FIFO,
accumulator handoff, register-file issue steal and the proxy fence.
The Blackwell module adds UMMA issue and commit, the tensor-memory
budget and loads/stores, CLC queries and async stores. See
[arch-semantics.md](arch-semantics.md).

## 8. Output

`cyclescope-engine simulate --job job.bin --out result.bin` writes the
execution record:

- the **result** (`GSIMRES`): total cycles, instructions issued and
  completed, stall cycles per bucket and per resource, counters;
- the **issue journal** (`GSIMJNL`, `--journal`): one `JournalRow` per
  issued instruction — warp, SMSP, instance, issue cycle, wait cause
  and cycles, the object waited on;
- the **wait ledger** (`GSIMWLG`, `--wait-ledger`): wait cycles per
  warp and cause.

Layouts: [../formats.md](../formats.md); the counters and telemetry
meanings: [../engine.md](../engine.md).

## 9. Failures

If no warp can make progress — no due retry, no pending event — the
engine dumps the blocked state and stops with
`unsupported:engine:deadlock`. `Simulator::Options::max_cycles` can
bound a run; crossing it stops the run and counts a hard failure.
The command line does not expose it.

## 10. In the code

| file | role |
|---|---|
| `cpp/core/simulator.{hpp,cpp}` | main loop, event queue, SMSP sweep, readiness order, wait buckets, issue journal |
| `cpp/core/mechanism_runtime.{hpp,cpp}` | resolves USES rows into leaf fields; engine-owned axes; ns→cycles; selection cache |
| `cpp/core/generated/parameter_schema.hpp` | generated field ids per schema |
| `cpp/core/sync_objects.{hpp,cpp}` | mbarrier, CTA barrier, cluster barrier state machines |
| `cpp/core/async_groups.{hpp,cpp}` | bulk / cp.async / warpgroup group lanes |
| `cpp/core/global_sync.{hpp,cpp}` | global-sync acquire plan and pairing |
| `cpp/core/memory_hierarchy.{hpp,cpp}` | L1 / L2 / DRAM, ports, in-flight fills, read merge |
| `cpp/core/cache_policy.hpp` | weighted global LRU and cache-hint behaviour |
| `cpp/core/instruction_fetch.{hpp,cpp}` | code caches and prefetch |
| `cpp/core/dispatch.{hpp,cpp}` | CTA dispatch, occupancy, entry spacing |
| `cpp/core/lifecycle.hpp` | per-launch envelope |
| `cpp/core/job_format.hpp`, `job_reader.{hpp,cpp}` | binary job records and the merged container reader |
| `cpp/core/result_writer.{hpp,cpp}` | `GSIMRES` / `GSIMJNL` / `GSIMWLG` writers |
| `cpp/core/semantics_kinds.hpp` | the `SEMA.kind` enum the engine dispatches on |
| `cpp/arch/architecture_module.hpp`, `registry.cpp` | the arch interface and module selection |
| `cpp/arch/hopper/` | `module`, `wgmma`, `proxy_fence`, `capture_rules`, `convert_rules` |
| `cpp/arch/blackwell/` | `module`, `umma`, `tmem`, `clc`, `stas`, `capture_rules`, `convert_rules` |
| `cpp/main.cpp` | the `cyclescope-engine` subcommands `simulate`, `validate`, `info`, `dump` |
| `cpp/tests/` | CTest fixtures: simulator, memory hierarchy, sync, fetch, arch modules, job reader |
