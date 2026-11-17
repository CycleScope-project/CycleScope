# Missing leaf / missing parameter

[中文](../../zh/agents/missing-leaf.md)

The mechanism exists but a leaf parameter is missing, a placeholder,
or a discrete axis value hits no leaf row
(`unsupported:engine:discrete_miss`).

## 1. Find the leaf

1. `cyclescope tree show <mechanism>` prints the mechanism's nodes and
   leaves from `configs/trees/{h100,b300}.json`.
2. Source fragments are per-mechanism files
   `configs/mechanism_tree/{common,hopper,blackwell}/mechanisms/
   <mechanism>.json` and `…/leaves/<mechanism>.json`, plus
   `axes.json`, `device.json`, `resources.json` and the provenance
   file `sources.json` in each of those directories.
3. Evidence levels: `measured` / `experimental` / `structural` /
   `placeholder`. Current `sources.json` counts — hopper:
   95 measured, 57 experimental, 10 structural, 205 placeholder;
   blackwell: 195, 78, 21, 4; common: 4, 4, 4, 3.

## 2. `discrete_miss` symptoms

A discrete leaf is a table over an axis value (e.g. an access width
or a warp count). `unsupported:engine:discrete_miss`
(`cpp/core/mechanism_runtime.cpp`) means the simulated value fell
between/outside recorded rows — check the `discrete_axis` convention
entry in the fragment for the recorded domain, then either measure
the missing cell (procedure below) or confirm the axis value itself
is a fact problem (capture/convert playbooks).

## 3. The measurement procedure (as the repo practises it)

1. Write **and freeze** `tools/microbench/<name>/protocol.json`
   before any data. Keys that real protocols use: `schema`
   (`cyclescope.microbench-protocol.v1`), `name`, `version`,
   `frozen_before_gpu`, `question`, `mechanisms`, `ownership`,
   `devices`, `probe`, `cells`, `sass_gate`, `predictions_before_data`
   / `prediction_of_the_installed_model…` / `alternative…`,
   `acceptance` (the install criterion), `decision`, `install`,
   `controls`, `runs`, `not_allowed`, `addenda` (dated corrections),
   `supersedes`.
2. Probe + SASS gate: the probe must compile to the intended SASS;
   the gate is recorded under `sass_gate` and checked by the
   protocol's own scripts.
3. Two runs per device, environment-variable driven:

   ```
   PROTOCOL=<name> DEVICE=h100|b300 RUN=<tag> \
       [CELL=<subdir>] [OUT_ROOT=<dir>] [NCU=<ncu>] [TIMEOUT=3600] \
       bash tools/microbench/run_local.sh
   ```

   It runs the driver's `run.sh` / `run_probes.sh` / `run_ncu.sh`
   under `tools/microbench/<name>` into
   `<artifact_root>/microbench/<dir>/<device>-<run>`. Keep both runs;
   do not cherry-pick.
4. Model script + the `acceptance` criterion decide **install vs
   unrealised** (`decision`/`install` fields); if the criterion fails
   the leaf is not installed — record the outcome in the protocol.
5. Corrections go in as dated `addenda`, never by editing the
   original predictions.
6. Install the value into the leaf fragment + a `sources.json` record
   with its evidence level.
7. `cyclescope tree build && cyclescope tree validate --strict` — all
   five conventions (`field_model`, `discrete_axis`, `unit`,
   `evidence_crossing`, `class_resource`) must report 0; if
   `configs/schema_registry.json` changed, `cyclescope tree build
   --emit-cpp` regenerates `cpp/core/generated/parameter_schema.hpp`
   and `python/cyclescope/tree/generated_schema.py`.
8. Re-run the shipped examples and the affected cases; report
   counterexamples.

Never pick a value because a case's error shrinks — attribute first,
then measure.
