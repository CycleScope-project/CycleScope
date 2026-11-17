# The evaluation suites

[中文](../zh/suites.md)

The project is evaluated against 500 production kernels per
architecture, taken from seven libraries. Each case is one launch of a
real kernel run — the table below shows where the kernels come from and
how many cases each library contributes.

| library | what it is | H100 | B300 |
|---|---|---|---|
| cuBLAS | NVIDIA's closed-source BLAS library | 50 | 50 |
| CUTLASS | NVIDIA's open-source templates and DSLs | 100 | 100 |
| DeepGEMM | DeepSeek's tensor-core kernel library | 100 | 100 |
| FlashAttention | fast, memory-efficient exact attention (version 3 on Hopper, 4 on Blackwell) | 100 | 100 |
| FlashInfer | kernels for LLM serving | 50 | 50 |
| FlashMLA | DeepSeek's latent attention | 50 | 50 |
| Triton | kernel language and compiler | 50 | 50 |

The suites are `suites/hopper/full_gpu` (H100, `sm_90a`) and
`suites/blackwell/full_gpu` (B300, `sm_103a`).

## 1. Running a case

With captures already on disk (as the shipped examples are):

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_einsum_05 \
    --captures examples/captures/hopper --out out/example
```

Without `--captures`, the case is captured on the local GPU first —
see [capture-tools.md](capture-tools.md) for the setup that requires.

`representative.txt` in each suite lists 20 cases — 4 per 100-case
library, 2 per 50-case library, chosen to cover every runner path.
They are the cases used for the checks in
[release-validation.md](release-validation.md).

## 2. The files of a suite

Everything under `suites/<arch>/full_gpu/`:

| file | what it is |
|---|---|
| `cases.tsv` (hopper) | the case table: `set`, `case_id`, `kernel_regex`, `occurrence`, `argv_json`, `cwd`, `env_json`, `env_profile` |
| `cases.txt` + `cases_<family>.json` (blackwell) | family → case id list, and the per-case spec (argv, shape, kernel) each runner resolves |
| `launches.tsv` | the launch each case simulates: `case`, `kernel`, `occurrence`, `grid`, `block`, `cluster` |
| `reference.tsv` | the hardware reference: `cycles_median/min/max`, `sm_clock_khz`, `cache_control`, `repeats`, `status` — Nsight Compute `sm__cycles_active.max`, `--cache-control all`, median of 3 |
| `expected.tsv` | the shipped baseline: `case`, `class`, `issued`, `sim_cycles`, `hw_cycles`, `placement_digest` |
| `results/` | one full run of the 500 cases on the baseline captures, kept as a reference: `report.tsv` (the `cyclescope report` output), `sim/<case>/status.json` and `stages.tsv` (final state, stage times, peak memory), `result_bin.tar.zst` (every case's `out-cpp/result.bin`) and `result_bin.sha256` (their checksums) |
| `representative.txt` | the 20 representative case ids |
| `inputs/` (hopper) | input files case argv reads that pre-existed the capture |

`results/` has the layout of a run output directory, so
`cyclescope report suites/<arch>/full_gpu/results` prints its table
again. To compare a run of your own byte for byte, check its
`sim/<case>/out-cpp/result.bin` against `result_bin.sha256`, or unpack
the archive with `zstd -dc result_bin.tar.zst | tar -xf - -C <dir>`.
The ten example captures under `examples/` reproduce their entries
exactly; a run on a new capture of the same case generally does not,
because the placement and the captured facts differ. The stage times
and memory figures are those of the machine that produced the run.
The issue journal and wait ledger of every case are not in this
directory; they are published in
[CycleScope-records](https://github.com/CycleScope-project/CycleScope-records).

## 3. How a case is resolved

Case argv, cwd and env use three path forms, expanded when the case
runs:

- `$CYCLESCOPE_ROOT/…` — this repository checkout.
- `$CYCLESCOPE_ARTIFACT_ROOT/tools/<name>/…` — a provisioned tool or
  third-party source registered in `configs/capture_tools.json`.
- `<CAPTURE>/…` — the capture output directory, substituted per run.

JIT caches and workspaces are *not* inputs: cache-root arguments
(`--cache-root`, `FLASHINFER_WORKSPACE_BASE`, …) resolve to an empty
writable directory under `<CAPTURE>/scratch/` and the runtime JIT
recompiles them. A cold JIT is the reason a first capture can take
longer — see [capture-tools.md](capture-tools.md).

## 4. Runners

`suites/<arch>/runners/` holds the per-library code that turns a case
row into an executable command. Third-party sources are provisioned
into `$CYCLESCOPE_ARTIFACT_ROOT/tools/<name>` and linked in at resolve
time — nothing third-party is vendored.

- **hopper**: `suites/hopper/runners/harness/` is the runner harness.
  It keeps its own internal layout because its files locate each other
  by relative path. `environment_far.sh` selects the "far" flashinfer
  profile for the cases that use it.
- **blackwell**: `suites/blackwell/runners/` holds one adapter per
  family plus shared `common/` helpers; adapters resolve the
  `cases_<family>.json` spec into argv, env and a kernel regex.

## 5. Using your own kernel

Capture resolves cases from the suite tables, so a new case is new
rows there. On Hopper the minimal steps are:

1. Add a row to `suites/hopper/full_gpu/cases.tsv`: a `case_id`, the
   `kernel_regex` + `occurrence` selecting the launch, and the
   resolved `argv_json` / `cwd` / `env_json` (the same three path
   forms apply).
2. Optionally add a row to `launches.tsv` pinning the exact launch.

On Blackwell a case is a `cases.txt` line plus an entry in the
family's `cases_<family>.json`, resolved by that family's adapter — so
a new case family means a new adapter, not only a table row.

The step-by-step version, with a worked example, is
[agents/new-kernel.md](agents/new-kernel.md); for a library the
suites do not cover, see
[agents/new-library.md](agents/new-library.md).
