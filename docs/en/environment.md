# Reference environments

[中文](../zh/environment.md)

A *reference environment* is the machine CycleScope's shipped numbers
were produced on: the GPU that ran the captures and the hardware
reference, and the host that built the tools. Recreating the listed
versions is the surest way to reproduce the baseline; other
configurations are not validated. `configs/capture_environment.json`
holds the full record, including the package lists (`pip_freeze`).
The non-pip fields:

## Host (build / convert / simulate)

| field | value |
|---|---|
| os | Ubuntu 22.04 |
| g++ | 11.3.0 |
| cmake | 3.22.1 |
| python | 3.10.6 |
| zstd | 1.4.8 |
| libraries | OpenSSL libcrypto development files |

## h100 (capture + hardware reference)

| field | value |
|---|---|
| gpu | NVIDIA H100 |
| driver | 570.124.06 |
| os | Ubuntu 22.04 |
| cuda | 12.8 runtime, nvcc release 12.9 (V12.9.86) |
| g++ | 11.4.0 |
| python | 3.12.13 |
| ncu | Nsight Compute 2025.1.0.0 (build 35237751), installed separately (`$CYCLESCOPE_ARTIFACT_ROOT/tools/nsight-compute`); the environment ships none |
| nvbit | 1.8 |
| key packages | torch 2.10.0+cu129, triton 3.6.0 (triton cases use the provisioned 3.7.0 wheels), numpy 2.2.6, nvidia-cutlass-dsl 4.4.2, cuda-python 12.9.4, flashinfer-python 0.6.6 (the far profile uses the provisioned 0.6.17 site) |
| capture_env | `CYCLESCOPE_CUDA_STACK_SIZE=65536` |

## b300 (capture + hardware reference)

| field | value |
|---|---|
| gpu | NVIDIA B300 SXM6 AC (SM103) |
| driver | 580.159.03 |
| os | Ubuntu 24.04 |
| cuda | 13.2, nvcc release 13.2 (V13.2.78) |
| g++ | 13.3.0 |
| python | 3.12.3 |
| ncu | Nsight Compute 2026.1.1.0 (build 37634170), from the environment |
| nvbit | 1.8 |
| key packages | torch 2.12.0+cu132, triton 3.7.0, numpy 1.26.4, nvidia-cutlass-dsl 4.5.2, cuda-python 13.4.1 |
| capture_env | `CYCLESCOPE_CUDA_STACK_SIZE=65536`, `FLASHINFER_NVCC_THREADS=4`, `MAX_JOBS=4` |

## Reference cards

`configs/hardware/<device>/` holds the per-card facts the pipeline
reads:

- `device.json` — device name, `sm_arch` (`sm_90a` / `sm_103a`), B300
  `tmem_columns`.
- `sm_topology.json` — one row per SM (`sm`, `placement_domain`,
  `l2_partition`, `tpc_pair`, `gpc`): 132 SMs for h100, 148 for b300
  (six b300 SMs have no GPC assignment — the `4294967295` sentinel).
- `preinitialized_mbarriers.json` (b300 only) — hardware-reserved
  mbarriers expanded per cta_group.

The `slow_gpc_mask` leaf of `launch.dispatch` declares which GPC is
slower on these particular cards — a declared fact about the cards,
not a fitted value.

What was checked on these environments, and what was not:
[release-validation.md](release-validation.md).
