# Accuracy and validation

[中文](../zh/release-validation.md)

What the shipped accuracy numbers mean, how they were measured, and
what else was checked on the reference environments
([environment.md](environment.md)).

## 1. How accuracy is measured

The reference is the launch's own hardware timing: Nsight Compute's
`sm__cycles_active.max` cycle count of that launch, collected with
`--cache-control all` (caches flushed before every launch), median of
3 repeats, in a separate run without the tracer.

`cyclescope report` compares simulated cycles against it with three
numbers:

- **MAPE** — mean absolute percent error, `mean |sim/hw - 1|`.
- **P90 absolute error** — the 90th percentile of the same errors.
- **median ratio** — the median of `sim/hw`: 1.000 would mean no bias.

`cyclescope report --baseline suites/<arch>/full_gpu/expected.tsv`
prints them for a shipped table.

## 2. Accuracy of the shipped baseline

`suites/<arch>/full_gpu/expected.tsv` holds 500 cases per
architecture, all `ok`:

| device | cases | MAPE | P90 absolute error | median sim/hw |
|---|---|---|---|---|
| H100 | 500 | 7.93 % | 17.2 % | 0.947 |
| B300 | 500 | 7.64 % | 14.8 % | 0.952 |

## 3. What else was checked

**Determinism.** Re-simulating the baseline captures of all 500 cases
per architecture with the release code reproduces `expected.tsv`
exactly — instruction count, simulated cycles, placement digest — and
the baseline `result.bin` byte for byte: 1000 of 1000, including
cases that run through the sharded convert path. The output of that
run is kept in `suites/<arch>/full_gpu/results/`
([suites.md](suites.md), section 2).

**Provisioning from scratch.** All 31 entries of
`configs/capture_tools.json` were provisioned into an empty artifact
root from repository content alone and verified
([capture-tools.md](capture-tools.md)).

**Re-capture fidelity.** The 20 + 20 representative cases were
re-captured on the reference environments with the provisioned tools.
Static capture facts — launch list, traced launch, functions, SASS —
match the baseline captures, with two exceptions:

- B300 case `TRF-TUT-ATTN-FWD-FP16-CAUSAL-C05`: 9 SASS lines of one
  JIT-compiled kernel differ only in uniform-register numbering; two
  independent new captures are identical to each other. The cause is
  not established.
- CuTe-DSL tensor-map tail bytes carry host-resident data that is not
  decoded; they differ between captures.

Simulated cycles of the re-captures versus baseline: B300 had the
baseline CTA→SM placement in 20 of 20 cases and stayed within 3 %
(largest 2.46 %). H100 had it in 4 of 20; the other 16 placed
differently and ranged up to 4.26 % (`fa3_fwd_hdim_small_08` — two
further captures of it reproduced the baseline placement at −1.84 %
and −1.25 %).

**Reference drift.** The hardware reference cycles themselves moved by
up to 3.6 % between capture batches.

**Case resolution.** The argv, environment and files that each of the
1000 suite cases resolves to are the same as when the baseline was
captured (compared case by case).

## 4. What was not checked

- Public container images.
- Driver or toolkit versions other than the reference environments.
- GPUs other than the two reference cards (NVIDIA H100, NVIDIA B300
  SXM6 AC).
- Re-capture of the 480 non-representative cases per architecture;
  their case resolution was compared (section 3) but they were not
  re-captured.
