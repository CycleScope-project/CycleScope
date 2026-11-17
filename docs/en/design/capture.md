# Capture: runtime and static facts

[中文](../../zh/design/capture.md)

The tracer has one job: pin down the execution that happened, so that
nothing about it has to be modelled. It records what decides which
instructions ran and what each of them waited on; every duration is
left to the other facts.

## 1. What is captured

One record per executed instruction, per warp. A record carries the
static-instruction identity, the active-thread mask, and the parts the
instruction needs later:

| fact | what is recorded |
|---|---|
| instruction stream | per warp, in execution order: which static instruction ran and with which lanes active |
| memory addresses | per lane, for every load, store, atomic, async copy and TMA operand; the addresses later decide which sectors and which memory level serve the access |
| synchronization operands | which barrier object an operation addresses (its shared address), arrival counts, expected transaction bytes, the phase parity waited on, per-lane cluster ranks (`%getctarank`) for multicast and remote targets |
| TMA copies | the tensor-map descriptor image and the coordinates, which give the block of memory being moved |
| barrier polls | for `PHASECHK`/`TRYWAIT`-class polls, an AFTER record with the ready mask the poll observed |
| global-memory synchronization | the value words a thread block read or wrote around each strong load, store, atomic or reduction used for cross-CTA signalling |
| cluster membership | per CTA: its rank inside the cluster and the cluster's shape, read from the `%cluster*` registers |
| launch facts | grid, block and cluster dims, device, and the CTA→SM placement (from the placement run) |

Two outcomes of the run are captured rather than derived: whether a
poll observed its barrier ready, and the values a synchronization
variable took. Everything else about *when* things happened is
reconstructed.

## 2. What is not captured, and why

- **Time.** Instrumentation slows the kernel down, so the timing of
  the traced run is not the timing of the real one. The trace answers
  *what*; static and mechanism facts answer *when*.
- **A poll's spin count, as timing.** A run of failed polls on the
  same barrier collapses into one wait whose length the engine
  decides. The folding happens in conversion
  (`cpp/capture/poll_folding.cpp`,
  `python/cyclescope/convert/poll_folding.py`): a run on one
  (object, parity) whose final sample reports ready folds to the
  surviving poll when the instructions between polls have no side
  effects.

## 3. How the tracer decides what to record

The tracer is an NVBit tool (`tools/tracer/`): the kernel runs on the
real GPU with its real inputs, and injected code writes records before
the instrumented instructions. Which operands to record is decided per
instruction form, from the binary alone, via the architecture table
(`tools/tracer/arch_table.h` with `arch_hopper.h` and
`arch_blackwell.h`): the operand-capture opcode families, which
opcodes carry an mbarrier word or a companion word, where a TMA
coordinate bank sits, and whether the SASS grammar has descriptor
tokens. Nothing is inferred from the kernel, its library or its input
shape.

## 4. Static facts

The binary supplies what the compiler decided. For every function the
tracer dumps the SASS it saw (`functions/<id>.tsv` plus
`functions.jsonl`): each instruction's operation, operand text and
control word — the stall count, yield hint, scoreboards set and waited
on, and reuse flags. Conversion decodes that into the static
instruction model; runtime facts give the values the static form
leaves open.

## 5. The capture directory

`cyclescope capture --suite <suite> --case <case> --out <dir>` leaves
one directory per case:

| file | contents |
|---|---|
| `run.json` | what ran: selection, return codes, tracer hash, verdict |
| `run_env.json` | the recorded environment (secrets redacted) |
| `app.log` | the application's own output |
| `launches.jsonl` | every kernel launch the process made |
| `functions.jsonl`, `functions/<id>.tsv` | static SASS per function |
| `tensor_maps.jsonl` | TMA tensor-map images seen by the tracer |
| `trace-<launch>.bin.zst` | the record stream of the selected launch |
| `placement/` | the placement run's own directory (same layout) |
| `trace_check.txt` | the completeness check's report |

## 6. Choosing the launch and the runs

A suite case names a kernel by regex and occurrence. The driver
resolves the case (`python/cyclescope/capture/driver.py`,
`launch_selection.py`), optionally takes an ncu census of the app's
launches, and selects one launch — it never traces an arbitrary one.
Two traced runs follow:

- the **trace run**, which records the instruction stream described
  above;
- the **placement run** (`CYCLESCOPE_CAPTURE_MODE=placement`), a
  lighter pass that records each warp's `%smid` and warp slot at
  function entry and exit — the CTA→SM placement the simulation needs.

The hardware reference is a separate, uninstrumented run:
`cyclescope reference` profiles the launch with ncu
(`--cache-control all`), so the reference clock and the cold-cache
condition are themselves facts recorded in the job.

## 7. In the code

| part | where |
|---|---|
| NVBit tool, record emission | `tools/tracer/tracer.cu`, `inject.cu`, `record_format.h` |
| per-architecture capture table | `tools/tracer/arch_table.h`, `arch_hopper.h`, `arch_blackwell.h` |
| TMA descriptor operands | `tools/tracer/tensor_map_operand.h` |
| capture driver, launch selection, completeness check | `python/cyclescope/capture/{driver,launch_selection,ncu_tools,trace_check}.py` |
| hardware reference | `python/cyclescope/reference/ncu_runner.py` |

Tracer environment variables the driver sets (others exist for
diagnostics; see `tracer.cu`):

| variable | effect |
|---|---|
| `CYCLESCOPE_TRACE_DIR` | output directory for the record stream |
| `CYCLESCOPE_CAPTURE_LAUNCHES` | which launches to trace (`<kernel>@<occurrence>`) |
| `CYCLESCOPE_CAPTURE_CLUSTERS` | which clusters to trace (`all` or an id list) |
| `CYCLESCOPE_CAPTURE_MODE` | `placement` for the placement run |
| `CUDA_INJECTION64_PATH` | loads the tracer into the CUDA process |
