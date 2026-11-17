# Convert and segments failures

[中文](../../zh/agents/convert-failures.md)

Facts reached the capture but a segment or the job cannot be built.
The terminal class lives in `sim/<case>/status.json`; the thrown
string is in `convert.log` / `segments.log` / `build.log`, and the
per-launch `missing_facts` / `unsupported` maps in
`facts/launch.json` name the fact and the `<where>` that consumed it
— the `<where>` is the first pointer to the rule to look at.

## 1. Who throws

| layer | code | error type |
|---|---|---|
| facts extraction | `python/cyclescope/convert/` and `cyclescope-capture-facts` | `missing_fact:*` |
| segment build (Python) | `python/cyclescope/convert/`, `python/cyclescope/arch/<arch>/rules.py` | `missing_fact:*`, `unsupported:*` |
| segment build (C++) | `cpp/arch/<arch>/convert_rules.cpp` | `ConversionError` with the same `missing_fact:*` / `unsupported:*` strings |
| job build / split / relink | `cyclescope-convert`, `python/cyclescope/convert/capture_pipeline.py`, `cyclescope relink` | `unsupported:trace_split_failed`, `unsupported:no_trace_split_binary`, `unsupported:ambiguous_launch`, `unsupported:relink_*`, `missing_fact:split_trace`, `missing_fact:split_gsync_sequence`, `missing_fact:trace_file`, `missing_fact:launch_row`, `missing_fact:facts_statics`, `missing_fact:mrex_record_bytes` |
| engine load/run | `cpp/` | see [engine-failures.md](engine-failures.md) |

Conversion-time `unsupported:` strings actually thrown
(`convert_rules.cpp`, `capture_pipeline.py`, `convert/`): arch
semantics `unsupported:{umma_kind, umma_a_source, tmem_shape,
tmem_alloc_op, wgmma_source_operands, warpgroup_wait_keep,
warpgroup_scoreboard, semantic_modifier}`; conversion-side
`missing_fact:` includes `wgmma_source_operands`,
`wgmma_accumulator_register`, `warpgroup_wait_keep`,
`umma_descriptor`, `umma_commit_mask`, `tmem_columns`,
`mbarrier_address`, `clc_mbarrier_address`, `clc_multicast_class`.

## 2. Cause families and where the rule lives

- **Opcode/operand form the tracer does not record** — arch
  instrumentation tables `tools/tracer/arch_*.h` + operand-family
  coverage (`CYCLESCOPE_TRACER_OPERAND_FAMILIES`); a missing record
  here is a capture gap, fix the tracer and re-capture.
- **Opcode the converter has no semantics for** — Python rules
  `python/cyclescope/arch/<arch>/rules.py`, C++ mirror
  `cpp/arch/<arch>/convert_rules.cpp`, common semantics under
  `python/cyclescope/convert/` / `cpp/core/`. Throws
  `unsupported:semantic_modifier` when a modifier form is unknown.
- **A capture axis the mechanism needs is absent** — `axes.json` in
  the mechanism tree; the mechanism raises `missing_fact:axis:<name>`
  when the fact is not in the job.
- **Program/service class missing in the tree** for an opcode
  (`instruction.service`) — reported as `missing_fact` at sim time,
  but the coverage decision is made when the segment is built.
- **Tensor-map / MMA forms** — descriptor/commit/tmem facts
  (`umma_descriptor`, `umma_commit_mask`, `tmem_columns`,
  `wgmma_source_operands`, `wgmma_accumulator_register`,
  `warpgroup_wait_keep`) missing → conversion-side `missing_fact:*`,
  or `unsupported:*` when the form itself is outside the supported
  set.
- **`launch_table_mismatch:{diff}`** — a `launches.tsv` row pinned a
  `(kernel, grid, block, cluster)` tuple the traced launch does not
  have.

## 3. The Python/C++ byte-identical rule

Facts and segments must be identical between the Python and C++
builders (`--builder python|cpp`); `tests/` parity tests guard it.
`cyclescope convert --no-cache` recomputes instead of trusting
`$CYCLESCOPE_ARTIFACT_ROOT/cache/segments`. A converter fix needs a
unit test plus a parity check.

## 4. Reproduce one stage by hand

```
cyclescope convert --capture <capdir> --out <facts>                 # facts
cyclescope convert --facts <facts> --builder python|cpp --out <seg> # segments
cyclescope build <dir> --out <job.bin>                              # job
cyclescope-engine validate --job <job.bin>
```

Compare tools: `scripts/pipeline/compare_facts.py`,
`compare_segments.py`, `compare_captures.py`,
`compare_results.py` (each takes two dirs/files of the same stage and
prints the first difference).

## 5. Decision

- Record never written → capture gap → `tools/tracer/` fix +
  re-capture; keep `GSIMSEG`-reader backward compatibility.
- Record present, fact not extracted → fact-extraction mapping
  (`python/cyclescope/convert/`); add a fixture.
- Fact present, segment row missing → segment builder rule; keep both
  builders in step.
- Intentionally unavailable (e.g. an occupancy/axis fact a suite does
  not record) → it must stay `missing_fact`; do not default it.
