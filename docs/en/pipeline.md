# The pipeline, stage by stage

[中文](../zh/pipeline.md)

One case flows through six stages. `cyclescope run` drives all of them
for a single case; every stage is also a command of its own. Two
stages need the GPU (capture and the hardware reference); everything
after them runs on the CPU, so you can replay an existing capture on
any machine.

```
capture ──► reference ──► convert ──► build ──► simulate ──► report
(tracer,    (ncu,         (facts +     (job.bin)  (result +    (row vs
 GPU)        GPU)          segments)              ledgers)     baseline)
```

- **capture** runs the case under the NVBit tracer and records what
  each warp executed (runtime facts) plus the SASS (static facts).
- **reference** times the same launch with Nsight Compute, without the
  tracer. It is the hardware number the report compares against.
- **convert** turns the capture into per-cluster facts and segments —
  everything that can be settled without timing.
- **build** packs the segments and the launch-level tables into one
  job container.
- **simulate** runs the C++ engine against the mechanism tree and
  writes the execution record.
- **report** prints one row per case and compares it with the shipped
  baseline.

## 1. `cyclescope run`

```bash
cyclescope run --suite hopper/full_gpu --case <case> --out <dir> \
    [--captures <root>] [--ledgers] [--keep] [--jobs N]
```

It resolves the case from the suite tables, runs the stages in order,
and stops at the first failure. With `--captures <root>` the GPU
stages are skipped: the case capture is `<root>/<case>` and its
reference is `<root>/reference/<case>` when present.

| flag | what it does |
|---|---|
| `--suite`, `--case`, `--out` | which suite case to run and where the output tree goes |
| `--captures` | reuse an existing capture root instead of capturing |
| `--reference-tsv` | use this reference table instead of the capture's own |
| `--skip-reference` | do not run the ncu stage (no `hw_cycles` column) |
| `--tree` | mechanism tree to bind (default: the suite's generated tree) |
| `--engine` | engine binary (default: `cyclescope-engine` on `PATH`) |
| `--builder` | `cpp` (default) or `python` segment builder |
| `--jobs` | parallel convert shards (default: shard count, capped at CPU count) |
| `--shard-records` | records per shard before conversion splits (default 1.5 M) |
| `--shard-max` | hard cap on the shard count (default 16) |
| `--no-placement` | convert without the captured CTA→SM placement |
| `--keep` | keep the intermediate `facts/`, `segments/`, `job.bin` |
| `--ledgers` | also write the issue journal and the wait ledger |
| `--repeats` | ncu repeats for the reference (default 3) |
| `--timeout` | seconds for the capture/reference subprocesses (default 5400) |
| `--tracer` | tracer library path (default: site configuration) |
| `--scratch` | do the conversion work in this directory, copy results back |

## 2. Running the stages by hand

```bash
cyclescope capture    --suite <suite> --case <case> --out <dir>
cyclescope reference  --suite <suite> --cases <case> --out <dir> [--repeats N]
cyclescope convert    --capture <dir> --arch <arch> --out <facts>
cyclescope convert    --facts <facts> --arch <arch> --out <dir>
cyclescope build      <dir> --out job.bin
cyclescope-engine simulate --job job.bin --out result.bin \
    [--journal j.bin] [--wait-ledger w.bin] [--progress]
cyclescope report     <out>
```

`capture` also accepts `--mode {trace,placement,both}`, `--ncu-repeats`,
`--timeout`, `--dry-run` and `--reference` (trace the launch a
reference run timed). `reference` accepts `--captures` (time the
launch each capture traced), `--metrics` (extra ncu metrics, saved as
`ncu_metrics.csv`) and `--source-stalls` (a SASS-level stall-sampling
pass, saved as `ncu_source.csv`). `convert` takes `--shard k/N`,
`--split-shards N`, `--builder`, `--tree`, `--cache`, `--no-cache`,
`--launch`, `--clusters`, `--reference`, `--placement`,
`--reference-sm-clock-khz` and `--reference-cache-control`.

Auxiliary commands, one line each:

- `cyclescope split <job.bin> --out <dir>` — unpack a container back
  into a segments directory.
- `cyclescope relink <seg|job.bin> --tree <t.json>` — recompile the
  mechanism-program tables onto a different generated tree without
  reconverting.
- `cyclescope tree build [--emit-cpp] | validate [--strict] | compile |
  diff | show` — the mechanism-tree tools
  ([design/mechanism-tree.md](design/mechanism-tree.md)).
- `cyclescope-engine validate|info|dump --job <file>` — inspect a job
  without simulating it.

## 3. What a run writes

```
<out>/
  captures/<case>/            capture directory (run.json, launches.jsonl,
                              functions/, tensor_maps.jsonl,
                              trace-<launch>.bin.zst, placement/ ...)
  captures/reference/<case>/  hardware reference (ncu repeats, reference.tsv)
  sim/<case>/
    status.json               the final state (table below)
    stages.tsv                one row per stage (section 6)
    convert.log segments.log build.log sim.log
    facts/ launch.json        per-cluster facts (kept with --keep)
    segments/                 GSIMSEG segments (kept with --keep)
    job.bin                   GSIMJOB container (kept with --keep)
    out-<builder>/
      result.bin              the execution record totals (GSIMRES)
      launch.json             the launch-level job document
      journal.bin             issue journal   (with --ledgers)
      wait.bin                wait ledger     (with --ledgers)
```

`status.json` is what `cyclescope report` reads. Its fields:

| field | meaning |
|---|---|
| `suite`, `case` | which case this is |
| `stage` | the stage the run stopped at (`capture` `reference` `convert` `segments` `build` `sim`) or `done` |
| `class` | `ok`, or a failure class (section 6) |
| `records` | dynamic-instruction records in the trace |
| `dropped_polls` | poll observations folded into waits |
| `missing`, `unsupported` | counts per missing-fact / unsupported key |
| `issued`, `completed` | instruction instances issued / retired by the engine |
| `cycles` | simulated cycle count |
| `hard` | hard-constraint failures counted by the engine |
| `walls` | wall seconds per stage (`convert`, `segments`, `build`, `sim`, `total`) |
| `rss_kb` | peak memory per stage, kilobytes |
| `hw_cycles` | the hardware reference cycles, when a reference exists |
| `reference_cache_control` | the cache condition the reference was taken under |
| `placement_digest` | first 16 hex of the SHA-256 over the captured CTA→SM placement — what the report compares with the baseline |
| `shards` | how many convert shards ran |

## 4. Large captures: sharding

A capture can hold more records than fits one convert run. The
converter then works on shards of the trace: `--split-shards N` splits
the traced launch by cluster, each shard runs `convert --capture …
--shard k/N` plus `convert --facts …`, shards run in parallel
(`--jobs`), and the facts and segment tables are merged with
consistency checks. `cyclescope run` plans the split itself: more than
`--shard-records` records (default 1.5 M) triggers it, up to
`--shard-max` shards (default 16). `stages.tsv` gets one line per
shard; `status.json` reports the count in `shards`.

## 5. When a run fails

Two failure classes describe data problems:

- `missing_fact:<fact> (<where>)` — a fact the pipeline needs is not
  in the capture. It is not replaced by a default.
- `unsupported:<reason> (<where>)` — the fact exists but the model
  does not cover it. An engine deadlock reports as
  `unsupported:engine:deadlock`.

`status.json.class` also uses `crash` (a stage exited nonzero) and
`issued_ne_completed` (the engine finished but not every instance
retired). Stage logs are the first place to look: `convert.log`,
`segments.log`, `build.log`, `sim.log` under `sim/<case>/`.

The [agent playbooks](agents/README.md) take it from there: a triage
table maps the stage and class to the playbook for capture failures,
conversion failures, engine failures, missing leaf values and missing
mechanisms.

## 6. Timing a run

Stage times live in `sim/<case>/stages.tsv`, one line per stage:

```
<stage>\t<wall_s>\t<max_rss_kb>\t<rc>\t<cpu_s>
```

`cpu_s` is the stage process tree's user+system time; sharded stages
add per-shard lines, so the sum can exceed wall time. `status.json`
mirrors the same numbers in `walls` / `rss_kb`.

When you compare simulator runtimes, count per case the CPU-side work
between the trace files landing on disk and the result being written:
trace parsing, fact extraction, job building, simulation and the I/O
between them. The instrumented GPU run, the ncu reference, queue wait
and report aggregation are not part of it. Pair successful cases
one-to-one and report ratios of wall and CPU time, stating which side
used intra-case parallelism.
