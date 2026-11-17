# DeepGEMM full-GPU suite

Public case JSON follows `hopper.delivery.cases.v1`. This suite directory is
`deepgemm`. `manifests/deepgemm_100.json` is the registered 100-case set.

## Source

DeepGEMM 2.6.1 at revision `559d79fb6994a58b8a15b4b93bf13ccc16edf247`, with
the sibling pinned CUTLASS tree as its CUTLASS dependency.

- `upstream/`: pinned DeepGEMM/JIT source
- `DELIVERY.json`, `reference/`: source package and previous-collection metadata

## Inventory

100 cases: 86 core GEMM and 14 attention. The 25 original core families stay
covered. Eight size-only duplicates were replaced with FP8 public nn/tn
wrappers, k-grouped unequal-K / small-K tensor-map shapes, FP8 M-grouped psum,
an independent Hopper `transpose_fp32` layout kernel, and an independently
timed Hopper paged MQA metadata kernel.

`delivery` is four groups, `delivery_index modulo 4`, runner
`scripts/run_100.sh`. `supported_gpu_profiles` are `h20`, `h100`, `h200`.
`validated_gpu_profiles` is empty. `DELIVERY.json` `qualification_status` is
`not_two_round_qualified`.

`manifests/deepgemm_single_sm.json` and `manifests/deepgemm_dual_sm.json` remain
as archived identity data. They are not registered in `DELIVERY.json` and have
no build/run/capture entry.

## Mechanisms

Cases carry a suite-local `mechanisms` list. Selection is mechanism-first on a
single GPU with normal multi-CTA scheduling and no artificial SM cap.
Library-specific fields stay on the case (`class_id`, `shape`, `suite_part`,
implementation module names). There is no shared `parameters` wrapper.

MQA cases remain logits scoring (prefill clean/compressed and paged), not
full attention. Independent `smxx_layout` `transpose_fp32` is a marked timed
entry via public `get_mn_major_tma_aligned_tensor`. Independent
`sm90_paged_mqa_logits_metadata` is a marked timed entry via public
`get_paged_mqa_logits_metadata`; paged logits cases still call that API in
prepare without counting it as kernel coverage. UE8M0 pack is SM100-only and
is not counted. NN/TN wrappers and SM90 1d2d identity scales do not launch
`transpose_fp32`.

## Build

```bash
bash suites/deepgemm/scripts/build.sh h20|h100|h200
```

The packager must emit a full7 native-heuristic+layout+metadata marker
package (`layout_transpose_fp32`, `paged_mqa_logits_metadata`,
`kgroup_native_heuristic=true`, `kgroup_cluster_required` not true). Missing
layout, missing metadata, and old cluster-forced packages are not reusable.
`build.sh` / `run_100.sh` reuse only a package whose marker metadata matches
that contract. Old marker artifacts remain as evidence.

The build needs CUDA PyTorch, `nvcc`, CUDA development headers/libraries
(including the linker name `libnvrtc.so`), and `cuobjdump` with `nvdisasm` on
`PATH`. NVIDIA Python-wheel include directories are discovered by the build
script; a runtime-only image may still need its NVRTC development link through
`LIBRARY_PATH`.

## Run

```bash
bash suites/deepgemm/scripts/run_100.sh h20|h100|h200 ROUND GROUP
```

`ROUND` is `1` or `2`. `GROUP` is `0..3`. This script takes exactly three
arguments; there is no positional SAMPLES argument. Formal recorded samples
are 9. Raw intervals per sample default through
`DEEPGEMM_DELIVERY_RAW_PER_SAMPLE` (4). GPU execution uses the
cluster job flow.

The harness launches the public DeepGEMM APIs. Hardware timing is the
whole-kernel `globaltimer` span converted with the per-launch
`clock64`/`globaltimer` ratio. That kernel span is not FA3 whole-operation
timing and is not interchangeable simulator-accuracy evidence.

## Validation

```bash
python3 suites/deepgemm/scripts/validate_delivery.py
```

This checks source and the case contract, not a new GPU run. Collection status
goes through `suites/summarize_cycle_collection.py`. A missing result
file is missing, never a pass. DeepGEMM pass requires both
`correctness.status` and `topology.status` equal to `pass`; do not read a
CUTLASS `passed` flag or an FA3 `status` field on these results.

## Evidence

`DELIVERY.json` `qualification_status` is `not_two_round_qualified`.

H100 first-round snapshot
`/path/to/hopper-prototype/artifacts/mechanism_coverage_20260920/first-full-verification.json`:
`expected=100`, `executed=100`, `verified=100`, `failed=0`,
`cv_over_3_percent=0`. Scope is
`single_gpu_h100_execution_and_recorded_checks_not_two_round_timing_qualification`.
This is not two-round qualification with FA3, and it does not make the
400-case collection passed.


Retained `reference_cycles` / round cycle / SASS fields and `reference/`
numbers are previous-collection history, not a current qualified conclusion.

Collection policy for a future two-round qualification: numerical
correctness, in-kernel markers, and single-GPU topology from observed CTA/SM
marker identities. Topology does not require every legal launch to occupy
128 SMs. The 100k–5M cycle window is collection-level statistics, not a
per-case pass/fail or exemption.


## Force-multicast rescue path (measurement facility)

The 36 cluster cases whose full-GPU config uses TMA multicast/cluster=2 were
skipped at num_sms=2 because the vendor SM90 cost model rejects single-wave
cluster launches (`heuristics/sm90.hpp` `get_layout_info`). A user-authorized
env-gated override rescues them for measurement only:

- Patch: `upstream/csrc/jit_kernels/heuristics/sm90.hpp` adds
  `delivery_force_multicast()`, reading `DG_DELIVERY_FORCE_MULTICAST`. With
  the env unset (the default) the heuristic is byte-identical to vendor.
  With `DG_DELIVERY_FORCE_MULTICAST=1`, the single-wave multicast rejection
  is skipped and `get_layout_candidates` keeps only legal 2CTA cluster
  candidates (asserting with the full GemmDesc if none exists). Every
  legality check (cluster divisibility, masked/psum even-tile rule, block
  sizes, swizzle, stages) stays active.
- Package: `harness/build_deepgemm_force_multicast_package.py` builds the
  rescue package (full7 marker pipeline + a `force_multicast` metadata
  stamp). `scripts/build_cluster_sm.sh` refuses to reuse it for the formal
  path; the 64-case qualified formal package is untouched.
- Run: `scripts/force_multicast_case.py` (probe -> formal, no shape tuning,
  original manifest shapes; masked/psum get at most the minimal even-tile n
  bump, recorded when applied) driven by `scripts/force_multicast_shard.sh`,
  launched via `scripts/launch_force_multicast.py` (predict-first cluster job).
- n-fit second pass (`force_multicast_case.py --n-fit`, user-authorized
  2026-09-21): for a case whose original-shape 2SM formal median exceeds the
  500k ceiling, `n` — the only adjustable axis; dtype/gemm_type/kernel_type/
  grouping/M/K/num_sms stay frozen — is shrunk along a legality-checked
  ladder (masked/psum keep the even-tile rule at every step) until a probe
  measures <= 450k, then the 9-sample formal runs. Each touched summary
  records `n_adjustment` with original n, fitted n, and the full probe
  curve; no lower bound applies. Cases still above the ceiling at the
  smallest legal n stay unresolved with the probe curve as evidence.
- Reporting: every rescue attempt embeds
  `delivery_force_multicast` in its result artifact and summary
  (`forced single-wave cluster at 2SM (DG_DELIVERY_FORCE_MULTICAST=1)`).
  Rescue timing is a num_sms=2 measurement condition; it makes no full-GPU
  steady-state multicast claim. Qualification is recomputed by the standard
  `scripts/cluster_sm_qualify.py` chain.

## Limits

- No artificial SM cap.
- Marker-package rebuild is required after packager source changes that
  affect `transpose_fp32`, `sm90_paged_mqa_logits_metadata`, or k-group
  scheduling.
- Historical single-SM / dual-SM JSON files are not executable modes.
