# Design overview

[中文](../../zh/design/README.md)

CycleScope answers one question: what happened, cycle by cycle, when
this kernel ran on this GPU? This page explains the idea. The other
design notes explain each part, in the order a launch flows through
them.

## 1. The question

Two different questions can be asked about a GPU.

- A hardware architect asks **what may happen** if a GPU is built a
  certain way. The machine is hypothetical, so its behaviour has to be
  modelled and its parameters configured or calibrated.
- A kernel developer asks **what has happened** to a kernel on a GPU
  that exists. The execution already took place, so it can be
  captured. The machine exists, so its behaviour can be measured.

CycleScope is built for the second question. It reconstructs one real
execution; it does not predict how the kernel would run on a different
machine.

## 2. Three kinds of facts

The reconstruction uses three kinds of facts and nothing else.

| fact | what it says | where it comes from |
|---|---|---|
| runtime facts | what each warp did: the instructions it executed, the active threads, the addresses it accessed, the operands of its synchronization and asynchronous operations | a tracer that runs with the kernel on the real GPU |
| static facts | what the compiler decided: each instruction's operation, the registers it reads and writes, and the control word that carries the compiler's schedule | the kernel binary |
| mechanism facts | how the hardware responds: latencies, issue intervals, bandwidths, capacities, the execution unit an instruction occupies | microbenchmarks, run once per GPU model |

Runtime and static facts settle **what** ran. Mechanism facts settle
**how long** things take. The trace is not used as a timing source:
instrumentation slows the kernel down, so the timing of the traced run
is not the timing of the real one.

## 3. Two rules

- **Traceable.** Every simulated cycle goes back to a captured fact or
  to a measurement. A missing fact is never replaced by a default: the
  run stops with `missing_fact:<fact>`. A situation the model does not
  cover stops with `unsupported:<reason>`.
- **Unfitted.** No parameter is chosen to match the cycle count of any
  workload. There are no rules keyed on a kernel, a library or an input
  shape, and every value in the mechanism tree records its source and
  an evidence level.

## 4. Two lifetimes

How the hardware behaves is measured once. What a kernel did is
captured for every launch.

```
once per GPU model                 for every kernel launch

microbenchmarks                    kernel on the GPU + tracer
      │                                   │  capture
      ▼                                   ▼
mechanism tree ──────────────┐     capture directory
(one per device)             │            │  convert   (one cluster at a time)
                             │            ▼
                             ├──────► segments
                             │            │  build     (one job per launch)
                             │            ▼
                             │         job.bin
                             │            │  simulate  (cycle by cycle)
                             │            ▼
                             └──────► execution record
```

| stage | what it settles | command | design note |
|---|---|---|---|
| capture | which execution happened | `cyclescope capture` | [capture.md](capture.md) |
| convert | everything that does not depend on time: what each instruction does to a barrier or a copy, which cache sectors it touches, which leaves it will use | `cyclescope convert` | [convert.md](convert.md) |
| build | joins the clusters of one launch into one input file | `cyclescope build` | [job-format.md](job-format.md) |
| simulate | when each instruction issued and, if it issued late, why | `cyclescope-engine simulate` | [engine.md](engine.md) |

Convert works on one cluster of thread blocks at a time, and nothing it
settles for one cluster depends on another, so clusters are converted
in parallel. Simulation cannot be split the same way: warps wait on
each other and all thread blocks share the L2 cache and DRAM, so one
launch advances as a whole.

## 5. Mechanisms, axes and leaves

A **mechanism** is one hardware behaviour that a kernel can observe and
that a microbenchmark can trigger on its own: the latency of an
instruction class, the delay between a barrier completing and its
waiters resuming, the bandwidth of the L2 cache. It is described from
the outside, by what a kernel experiences, not by the circuit behind
it.

The same mechanism behaves differently under different conditions. Each
condition is an **axis**: a data type, a shape, an access width, the
number of warps waiting. For an execution that already happened, every
axis value is known when the mechanism is used. It is either a captured
or static fact, or a state the engine keeps.

So a mechanism is stored as a tree. An internal node splits on one
axis. A **leaf** holds the behaviour for one combination of conditions,
as a constant, a linear function of an axis, or a table. Each device
has its own tree, with about fifty mechanisms and a few hundred
leaves. See [mechanism-tree.md](mechanism-tree.md).

## 6. The engine

The engine replays each warp's captured instruction stream against the
tree and places every instruction at the earliest cycle the hardware
would have allowed.

"What the hardware allows" needs a structure. The engine assumes a
small one: the issue slots of an SM sub-partition, execution units,
memory ports and cache levels, the tensor-core pipeline, and barriers
as state machines defined by the instruction set. The structure holds
no numbers. Every interval, bandwidth, capacity and delay in it is a
leaf.

Three things can hold an instruction back, and each is resolved mainly
by one kind of fact.

| what holds the instruction back | resolved by | examples |
|---|---|---|
| the compiler's schedule | static facts | register dependences, the stall count and scoreboards named by the control word |
| other warps | runtime facts | barriers, synchronization through global memory |
| the capacity of the hardware | mechanism facts | execution units, memory ports, cache levels, L2 and DRAM bandwidth shared by all SMs |

The engine computes no data values and simulates nothing below a
mechanism. It is event driven: time is resolved to the cycle, but the
engine jumps from one change of state to the next. Hopper and Blackwell
share one engine; each architecture adds a module for its own tensor
core and synchronization instructions and brings its own tree. See
[engine.md](engine.md) and [arch-semantics.md](arch-semantics.md).

## 7. The execution record

The output of a simulation is a record of the execution, not only a
total:

- the result: total cycles, instructions issued and completed, stall
  cycles by cause, counters;
- the issue journal: one row per issued instruction, with its issue
  cycle, how long it waited, the cause of the wait and, for a
  synchronization operation, the object it addressed;
- the wait ledger: wait cycles per warp and cause.

A performance question then becomes a lookup in the record instead of
an inference from aggregate counters. The file layouts are in
[../formats.md](../formats.md).

## 8. Where things live

| part | code | design note |
|---|---|---|
| tracer and capture driver | `tools/tracer/`, `python/cyclescope/capture/` | [capture.md](capture.md) |
| conversion | `cpp/capture/` (trace → facts), `cpp/convert/` (facts → segments), `python/cyclescope/convert/`, `python/cyclescope/arch/` | [convert.md](convert.md) |
| job files | `python/cyclescope/job/`, `python/cyclescope/io/` | [job-format.md](job-format.md) |
| mechanism tree | `configs/mechanism_tree/`, `configs/trees/`, `python/cyclescope/tree/`, `tools/microbench/` | [mechanism-tree.md](mechanism-tree.md) |
| engine | `cpp/core/`, `cpp/arch/` | [engine.md](engine.md) |
| what differs between Hopper and Blackwell | — | [arch-semantics.md](arch-semantics.md) |
| terms and naming rules | — | [naming.md](naming.md) |
