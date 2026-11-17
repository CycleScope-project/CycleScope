# Missing mechanism

[中文](../../zh/agents/missing-mechanism.md)

The hardware does something the model has no rule for. **Attribute
first**: a mechanism is added to explain observed behaviour, not to
shrink one case's error.

## 1. Localise

1. `cyclescope run --ledgers --keep …`: per-stage journal +
   wait ledger under `sim/<case>/out-<builder>/` show where sim time
   and hardware time diverge per warp.
2. Hardware counters: `cyclescope reference --metrics <m1,m2>` adds
   raw ncu csv (`ncu_metrics.csv` per case); `--source-stalls` runs a
   SourceCounters pass (`ncu_source.csv`, SASS-level stall sampling).
3. A targeted microbenchmark once the gap is hypothesis-shaped (see
   [missing-leaf.md](missing-leaf.md) §3).

Attribution means you can state: which stage diverges, on which
warp/instruction, waiting on what, and which candidate resource or
synchronisation object explains it. If you cannot, keep narrowing —
do not install a mechanism you cannot phrase as a question.

## 2. Where a mechanism enters (in order)

1. **Schema**: new field(s) in `configs/schema_registry.json`
   (unit/type per field).
2. **Tree fragments**: `configs/mechanism_tree/common/` (shared) and
   `configs/mechanism_tree/hopper/` + `configs/mechanism_tree/
   blackwell/` (per-arch leaves/axes/sources; may be one arch only).
3. **Generated schema**: `cyclescope tree build --emit-cpp` →
   `cpp/core/generated/parameter_schema.hpp`,
   `python/cyclescope/tree/generated_schema.py`; then
   `tree validate --strict`.
4. **Engine rule**: shared behaviour in `cpp/core/`; arch behaviour in
   `cpp/arch/<arch>/`. Ownership: cluster-local leaves are charged in
   the single-cluster loop; gpu-shared leaves (L2 capacity/hit, shared
   bandwidth, atomic contention, dispatch, global sync, clock) in the
   full-GPU loop — see [../design/mechanism-tree.md](../design/mechanism-tree.md).
5. **Converter facts / USES**: if the rule needs a new capture fact,
   extend the tracer records + fact extraction (`GSIMSEG` readers must
   stay backward-compatible) and declare the mechanism USES.
6. **Tests**: a CTest fixture under `cpp/tests/` plus pytest under
   `tests/`; Python/C++ builders stay byte-identical.
7. **Regression**: the shipped examples and `expected.tsv` results
   must stay identical when the new field is absent/off; only then
   the affected cases.

Design notes per layer: [../design/mechanism-tree.md](../design/mechanism-tree.md),
[../design/engine.md](../design/engine.md),
[../design/job-format.md](../design/job-format.md),
[../design/convert.md](../design/convert.md).

## 3. Checklist

- [ ] attributed (ledger/counters/probe), not fit to a case
- [ ] `schema_registry.json` field + `--emit-cpp` schema regenerated
- [ ] tree fragments + `sources.json` records with evidence level
- [ ] ownership honoured (cluster-local vs gpu-shared)
- [ ] converter fact/USES added if needed, format backward-compatible
- [ ] CTest + pytest added
- [ ] examples/expected.tsv byte-identical with the field off
- [ ] strict validate all-zero
