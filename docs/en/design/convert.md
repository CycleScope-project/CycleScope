# Convert: settle everything that does not depend on time

[中文](../../zh/design/convert.md)

Conversion turns a capture into simulation input, without ever asking
*when*. It runs in two steps: the trace becomes per-cluster **facts**
(`cyclescope convert --capture`), and the facts become **segments**
(`cyclescope convert --facts`). The unit of work is one cluster of
thread blocks, and nothing settled for one cluster depends on
another.

## 1. The facts-only rule

A dynamic instruction's facts come only from its own record, or from
records keyed by its id or its operand values. Nothing is inferred
across warps: which barrier generation a wait observed, or which
write produced a value a read saw, is *not* decided here. Barrier
generations and cross-thread-block pairings are solved by the engine
at simulation time — a generation field leaves conversion marked
unresolved. A fact that is missing is not replaced by a default:
conversion stops with an error (§8).

## 2. Step 1: trace → facts

The record stream is replayed into a typed model
(`python/cyclescope/convert/model.py`; the C++ side lives in
`cpp/capture/`):

- **launch facts**: grid/block/cluster dims, device, shared-memory
  sizes, register count, the CTA→SM placement, the reference clock;
- **static instructions**: the decoded SASS — opcode, operation class,
  source and destination registers, control word;
- **dynamic instructions**: one per execution, with the active mask,
  the predicate result, and its operand facts;
- **memory references**: space, kind, width and per-lane addresses;
- **warp streams**: each warp's dynamic instructions in program order;
- **cluster units**: the CTAs of one cluster with their warp streams,
  tensor maps and pre-initialized barriers.

Poll folding and the global-sync observations land here too: failed
polls fold as described in [capture.md](capture.md), and the values
seen at each strong global access are carried through as observations
for the engine to pair. The output is one `GSIMCAP` facts file per
cluster plus a `launch.json` sidecar.

## 3. Step 2: facts → segment

Per cluster, the builder writes one segment. What each dynamic
instruction gets (layouts in [../formats.md](../formats.md)):

| rows | meaning |
|---|---|
| `TMPL`/`REGS` | the instruction template: operation class, register sources and destinations, scoreboards and control word |
| `INST` | the instance itself: which template, which semantics row, which memory access |
| `SEMA` | what the instruction *does to* a barrier, group or copy: object, counts, tensor map, descriptor — the unpacked operand facts |
| `MACC`/`MSEC` | which cache sectors its memory accesses touch (sector lists merged per access) |
| `TIME` | the timing rows: which leaf applies at which program event |
| `USES`/`FACT` | the binding: which mechanism program applies, walked down to the node where the remaining axes are engine-owned, with the captured axis values |

Auxiliary tables carry the rest: `CTAS`/`WARP` (cluster membership and
warp slots), `MBPI`/`MBOL`/`MBOB` (pre-initialized and multi-object
barriers), `GSYN` (global-sync observations), `CODE` (code addresses
for the fetch model), `STRS` (strings).

## 4. Choosing leaves ahead of time

Every mechanism program is guarded by axes. Conversion resolves each
axis whose owner is `capture`, `converter` or `launch` — the data
type, the access width, the operation class, the cluster shape — using
the axis functions in `python/cyclescope/convert/axes.py`
(`cpp/convert/axes.cpp`). Where a walk reaches an axis owned by the
`engine` — a state that exists only during simulation, such as how
many warps have arrived — conversion stops and writes a `USES` row
pinned at that node; the engine resumes the walk from there. A guard
with no matching child means the program does not apply; a missing
value for a capture-owned guard is a `missing_fact`. Axis owners are
declared per axis in `configs/mechanism_tree/*/axes.json`.

## 5. Where architecture differences enter

Conversion shares one model; the arch-specific parts are pluggable
rules: operand-fact interpretation, effective operation classes and
semantic decoding live in `python/cyclescope/arch/<arch>/` and
`cpp/arch/<arch>/{capture_rules,convert_rules}.cpp`. See
[arch-semantics.md](arch-semantics.md).

## 6. Two builders

The C++ builder is the default (`--builder cpp`; binaries
`cyclescope-capture-facts` for step 1 and `cyclescope-convert` for
step 2). The Python builder (`--builder python`) is the reference
implementation. `tests/common/test_convert_cpp_builder.py` converts
the same facts with both and asserts the required segment tables are
byte-identical between the two, so a semantic change cannot land in
one builder alone without a test noticing.

## 7. Parallelism and sharding

Clusters are independent, so conversion shards by cluster and the
shards run in parallel; `cyclescope-trace-split` pre-splits large
record streams. Commands and output layout:
[../pipeline.md](../pipeline.md).

## 8. Failures

Two failure classes cover every data problem:

- `missing_fact:<fact> (<where>)` — a fact the model needs is absent
  from the capture, e.g. `missing_fact:parity` when a barrier wait's
  phase operand was not captured, or `missing_fact:cta_barrier_operand`
  for a named-barrier count held in a register nobody recorded;
- `unsupported:<reason> (<where>)` — the fact exists but the model
  does not cover it, e.g. `unsupported:mixed_async_groups` when one
  warp interleaves async-group kinds the state machine does not track.

Application and asset errors (corrupt files, digest mismatches) are
format errors and never enter this vocabulary.

## 9. In the code

| part | where |
|---|---|
| trace → facts (C++) | `cpp/capture/` (`trace_reader`, `control_flow`, `memory_refs`, `operand_facts`, `poll_folding`, `global_sync`, `tensor_map`, `static_code`, `facts_writer`, `launch_doc`) |
| facts → segments (C++) | `cpp/convert/` (`facts_reader`, `segment_builder`, `semantics`, `traversal`, `axes`, `trace_split`) |
| the same two steps in Python | `python/cyclescope/convert/` (`model`, `trace_reader`, `static_code`, `control_flow`, `poll_folding`, `operand_facts`, `global_sync_observation`, `tensor_map`, `semantics_common`, `mechanism_uses`, `axes`, `facts_format`, `pipeline`, `capture_pipeline`) |
| arch rules | `python/cyclescope/arch/<arch>/`, `cpp/arch/<arch>/` |
