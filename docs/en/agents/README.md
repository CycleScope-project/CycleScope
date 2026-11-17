# Agent playbooks

[中文](../../zh/agents/README.md)

Playbooks for agents debugging captures, conversions, and simulations
in this repository. Read this file first; then the playbook matching
your symptom.

## Ground rules (from the project principles)

1. **Mechanisms, not lookup tables.** Never add a timing rule or a
   leaf value keyed on a case id, kernel name, shape, or library. A
   fix that only works for one case is not a fix.
2. **Capture records facts, conversion moves facts.** A dynamic
   instruction's facts come only from its own record or records keyed
   by its id / operand values. Generations of mbarriers and CTA
   barriers are solved by the engine at simulation time, never
   inferred at conversion time.
3. **Missing facts are rejected, never defaulted.** The only data
   failure classes are `missing_fact:<fact> (<where>)` and
   `unsupported:<reason> (<where>)`. Do not invent a fallback value
   to make a case pass.
4. **Every leaf value has an evidence level and a source** (`measured
   / experimental / structural / placeholder` in
   `configs/mechanism_tree/<arch>/sources.json`). A new measurement
   starts with a frozen protocol (`tools/microbench/<name>/
   protocol.json`); if the criterion fails, the leaf is not installed.
5. **Attribute before installing.** Do not pick a mechanism or a leaf
   value because it shrinks a case's error.
6. **Pure refactors keep results byte-identical.** Verify with
   `cyclescope run` on the shipped examples and the
   `suites/<arch>/full_gpu/expected.tsv` tables.

## Stage map and where to look

`cyclescope run --suite <s> --case <c> --out <out>` writes:

```
<out>/captures/<case>/         capture stage (run.json, trace-*.bin.zst,
                               launches.jsonl, functions/, app.log,
                               ncu_launches.jsonl, trace_check.txt)
<out>/captures/reference/<case>/  reference stage (reference.tsv, timing.json,
                               ncu repeats)
<out>/sim/<case>/              convert.log, segments.log, build.log, sim.log,
                               stages.tsv, status.json, facts/launch.json,
                               out-<builder>/ (job.bin, result.bin, journal,
                               wait ledger)
```

Terminal `stage` / `class` live in `sim/<case>/status.json`;
`stages.tsv` has one `stage wall_s max_rss_kb rc cpu_s` row per stage.
`cyclescope run` exits 0 only for `stage=done, class=ok`.

## Triage

| symptom (stage / class or where seen) | playbook |
|---|---|
| Need to capture a kernel not in the suites | [new-kernel.md](new-kernel.md) |
| Need to capture a library/framework never seen | [new-library.md](new-library.md) |
| stage `capture` or `reference`; `app_failed:`, `ncu_failed:rc=…`, `ncu_timeout`, `selection_failed:`, `zstd_missing`, `bad trace header`, `no traced launch`, `missing trace-N.bin.zst`, `missing functions/<id>.tsv` | [capture-failures.md](capture-failures.md) |
| stage `convert` or `segments`; `missing_fact:*`, `unsupported:*` in `launch.json` / logs | [convert-failures.md](convert-failures.md) |
| stage `sim`; `unsupported:engine:*`, `deadlock`, `crash`, `issued_ne_completed`, nonzero hard constraints | [engine-failures.md](engine-failures.md) |
| Mechanism exists, leaf missing / placeholder / `unsupported:engine:discrete_miss` | [missing-leaf.md](missing-leaf.md) |
| Hardware behaviour with no mechanism rule at all | [missing-mechanism.md](missing-mechanism.md) |

## Playbooks

- [new-kernel.md](new-kernel.md) — capture + simulate a kernel not in
  the suites
- [new-library.md](new-library.md) — bring up a library/framework
  never captured
- [capture-failures.md](capture-failures.md) — tracer / census /
  reference failures
- [convert-failures.md](convert-failures.md) — `missing_fact:` /
  `unsupported:` during conversion
- [engine-failures.md](engine-failures.md) — simulator errors and
  deadlock triage
- [missing-leaf.md](missing-leaf.md) — missing/placeholder leaf
  parameters
- [missing-mechanism.md](missing-mechanism.md) — adding a mechanism
  end to end
