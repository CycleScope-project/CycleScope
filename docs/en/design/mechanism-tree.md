# The mechanism tree

[中文](../../zh/design/mechanism-tree.md)

A **mechanism** is one hardware behaviour a kernel can observe and a
microbenchmark can trigger on its own: the latency of an instruction
class, the delay between a barrier completing and its waiters
resuming, the bandwidth of the L2 cache. Each condition that changes
the behaviour is an **axis**; a **leaf** holds the behaviour for one
combination of conditions. One generated tree per device holds every
mechanism.

## 1. A mechanism and its leaves

An internal node splits on one axis; a guard selects the child for
one value or range. On B300, `tensor.pipe` — the interval a tensor
operation occupies the shared tensor pipe — splits on
`operation_class`, then `umma_kind`, then `cta_group`, then `a_src`:

```
tensor.pipe
└── operation_class = UTCHMMA
    └── umma_kind = f16
        └── cta_group = 1            (one CTA, not a pair)
            └── a_src = smem  ──► leaf
```

That leaf's `cycles` field is a discrete table over the operation's
shape axes (`tensor_m`, `tensor_n`, `tensor_k`, `umma_operation`…):

| M | N | K | operation | cycles |
|---|---|---|---|---|
| 32 | 64 | 16 | ws | 28 |
| 32 | 128 | 16 | ws | 44 |
| 64 | 8 | 16 | normal | 23 |

Every axis value is known when the mechanism is used — a captured or
static fact, or a state the engine keeps — so the walk always lands on
one leaf.

## 2. What a leaf holds

A leaf is a set of named **fields** from a parameter schema in
`configs/schema_registry.json`: `cycles`, `enabled`, latencies,
bandwidths, capacities, resource ids. Each field's value is a model:

| model | meaning |
|---|---|
| `constant` | one number |
| `linear` | a piecewise-linear curve through measured points on one axis, held constant beyond the end points |
| `affine` | `slope · axis + intercept` over a declared domain |
| `discrete` | a table keyed by one or more axes; a miss fails closed |

A leaf also records a **scope** (the entity the values describe: an
instruction, warp, CTA, SMSP, SM, cluster, or the whole device) and
units. Fields declared in nanoseconds are converted to cycles with the
launch's reference SM clock (`reference_sm_clock_khz`) when the engine
resolves them (`cpp/core/mechanism_runtime.cpp`).

## 3. Where values come from

Every field names its **sources** — keys into
`configs/mechanism_tree/<arch>/sources.json`, where each source
records an evidence level and a description of the measurement:

| evidence | meaning |
|---|---|
| `measured` | read off a microbenchmark by the rules of its frozen protocol |
| `experimental` | measured, but the protocol's criterion was only partly met (for example two runs disagree beyond the bound) — the reason is in the source |
| `structural` | fixed by construction: the model's own constants (an enable flag, a unit conversion, an assumed structure) |
| `placeholder` | a stand-in until a protocol measures it |

A leaf's status is derived from its sources: all-structural →
`structural`; measured and structural sources → `measured`; any
experimental source → `experimental`; otherwise `placeholder`. A
device entry can also be `unsupported`: the class does not exist on
that architecture (for example Hopper's `HGMMA` in the B300 tree);
such nodes are kept so the partition stays complete.

The usual measurement shape: a small kernel in which the mechanism
under study is the only cost that varies; sweep the repetitions or
the axis itself and read the value from the slope, so fixed overhead
cancels; the protocol is written down and frozen
(`tools/microbench/<name>/protocol*.json`) before the number goes in.
The protocol catalogue is [../microbench.md](../microbench.md).

## 4. Ownership: cluster-local and gpu-shared

Whether a leaf can even be *observed* depends on what else is on the
card. With one cluster alone on the GPU there is no L2 or DRAM
contention, no atomic contention, no dispatch beyond the first wave —
leaves whose value depends on the load of other SMs are
**gpu-shared** and are measured with the card loaded. Leaves inside
one SM or cluster are **cluster-local**.

| ownership | mechanisms |
|---|---|
| cluster-local | `instruction.*`, `scheduler.policy`, `register_file.*`, `control.*`, `synchronization.*`, `tensor.*`, `hopper.*`, `blackwell.umma.*`, `blackwell.tmem.*` within one thread block's budget, `storage.shared.*`, `storage.async.*`, `storage.tma.*`, `storage.atomic.issue`, `memory.l1.*`, `memory.constant_cache`, `memory.layout`, `kernel.lifecycle` for a launch on one or two SMs, and the per-SM port rate and uncontended latency in `memory.l2.service`, `memory.dram.service`, `memory.store_l2.service`, `memory.store_dram.service` |
| gpu-shared | `memory.l2.capacity`, `memory.l2.residency`, `memory.dram.capacity`, `memory.store_l2.capacity`, `memory.store_dram.capacity`, `memory.shared_service`, `memory.l2_merge`, `memory.l2_partition`, `memory.global_atomic`, `launch.dispatch`, the load-dependent latency in `memory.l2.service` and `memory.dram.service`, `kernel.lifecycle` across many SMs, `blackwell.clc.*` when several SMs compete, the `blackwell.tmem.*` budget shared across thread blocks |

The split can run through one mechanism: `memory.l2.service` holds a
per-SM port rate (cluster-local) next to a load-dependent latency
(gpu-shared). A mechanism not listed here is classified by the same
criterion.

## 5. When a leaf is picked

Each mechanism declares a program **event** — the moment the engine
asks for it — and a guard tree. The events are `initialization` (once
at load), `issue` and `memory_issue` (when an instruction issues),
`barrier_release` (when a barrier phase completes),
`control_transition` (a branch or reconvergence) and
`collective_launch` (a warpgroup-wide op going to the tensor pipe).
At conversion time the walk resolves every axis whose owner is
`capture`, `converter` or `launch` and stops where the remaining axes
are `engine`-owned; at run time the engine finishes the walk with the
state it keeps. See [convert.md](convert.md) §4.

## 6. Source layout

```
configs/mechanism_tree/
  common/        mechanisms, axes, resources, sources, class_resource:
                 shared definitions
  hopper/        per-arch axes + leaf values for Hopper
  blackwell/     the same for Blackwell, plus its own mechanisms
configs/schema_registry.json     parameter schemas (fields, units)
configs/trees/{h100,b300}.json   the generated trees
```

`cyclescope tree build` merges common + arch fragments into
`configs/trees/<device>.json`; `--emit-cpp` additionally regenerates
`cpp/core/generated/parameter_schema.hpp` and
`python/cyclescope/tree/generated_schema.py`, which the engine and the
builders use to address fields by id. `cyclescope tree compile`
emits the binary program tables (`GSIMPRG0`) a converter consumes
without walking JSON.

## 7. Validation and the strict gate

`cyclescope tree validate --strict` builds the trees and checks both
structural validity (guards partition their axis's domain, models'
declared axes exist and have the right kind) and five convention
classes (`python/cyclescope/tree/conventions.py`):

- `field_model` — every device leaf field of a shared mechanism uses
  the model kind and axes the mechanism declares in `field_models`;
- `discrete_axis` — a `discrete` table keys only on enum axes
  (quantities the instruction encoding fixes); continuous quantities
  use `affine` or `linear`;
- `unit` — nanosecond fields appear only in `memory.shared_service`,
  `memory.global_atomic` and `launch.dispatch`;
- `class_resource` — an `instruction.service` leaf's resource matches
  the class binding in `common/class_resource.json`;
- `evidence_crossing` — a shared mechanism is not measured on one
  device and placeholder on the other.

The gate passes only when all five lists are empty. Other tree
commands: `tree show <tree> <mechanism>` prints one guard tree;
`tree diff` compares two generated trees.

## 8. Changing or adding a leaf

1. Measure: run the microbenchmark under its frozen protocol
   ([../microbench.md](../microbench.md)).
2. Add a source entry with its evidence level in
   `configs/mechanism_tree/<arch>/sources.json`.
3. Set the leaf value and its source keys in the arch's `leaves/` or
   `mechanisms/` fragment.
4. `cyclescope tree build`, then `cyclescope tree validate --strict`.
5. If the schema registry changed (a new field or unit), rebuild with
   `tree build --emit-cpp` so the generated schema stays in sync.

## 9. In the code and tests

| part | where |
|---|---|
| build, merge, digest | `python/cyclescope/tree/{build,source}.py` |
| validation | `python/cyclescope/tree/{validate,conventions,registry}.py` |
| generated schema | `cpp/core/generated/parameter_schema.hpp`, `python/cyclescope/tree/generated_schema.py` |
| run-time leaf resolution | `cpp/core/mechanism_runtime.{hpp,cpp}` |
| converter-side traversal | `python/cyclescope/convert/mechanism_uses.py`, `cpp/convert/traversal.cpp` |
| tests | `tests/common/test_generated_schema.py`, `test_tree_validate*` (`tests/`), `cpp/tests/mechanism_runtime_test.cpp` |
