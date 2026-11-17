# Third-party notices

cyclescope is licensed under the MIT license (see LICENSE). This file lists
the third-party projects whose material is included in, derived by, or
downloaded for the repository's benchmark harnesses and provisioning.

## Material included in or derived by this repository

### CUTLASS (BSD-3-Clause, NVIDIA CORPORATION & AFFILIATES)

- `suites/blackwell/runners/cutlass/src/sc/*.cu` — benchmark sources
  derived from the CUTLASS examples.
- `suites/blackwell/runners/patches/cutlass/*.patch` and
  `suites/blackwell/runners/cutlass/patches/*` — patches applied to the
  downloaded CUTLASS tree.
- `suites/hopper/runners/harness/suites/cutlass/harness/**` — delivery
  harness sources derived from CUTLASS example code, including the
  override header.
- License verified at `LICENSE.txt` of the provisioned CUTLASS trees
  (e.g. `hopper-cutlass-source`, `blackwell-cutlass-source`).

### Triton (MIT, Philippe Tillet / OpenAI)

- `suites/blackwell/runners/triton/official/**` — extracted upstream
  tutorials plus `upstream-extraction.diff`.
- License verified at `triton-source/LICENSE`.

### DeepGEMM (MIT, DeepSeek)

- Snapshot / marker patches under `suites/hopper/runners/patches/deepgemm/**`.
- License verified at `hopper-deepgemm-source/LICENSE`.

### FlashMLA (MIT, DeepSeek)

- Snapshot / marker patches under `suites/hopper/runners/patches/flashmla/**`
  (including `PINNED.md`, a project snapshot record, not an upstream file)
  and `suites/blackwell/runners/flashmla/patches/**`.
- License verified at `hopper-flashmla-source/LICENSE` and
  `blackwell-flashmla-source/LICENSE`.

### flash-attention (BSD-3-Clause, Tri Dao et al.)

- `suites/hopper/runners/patches/flashattention3/**` — patches applied to
  the downloaded FA3 tree.
- The provisioned snapshot carries no license file; source headers read
  "Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay
  Thakkar, Pradeep Ramani, Tri Dao." The upstream repository
  (Dao-AILab/flash-attention) is published under BSD-3-Clause.

### FlashInfer (Apache-2.0)

- Patches and site files under `suites/hopper/runners/patches/flashinfer/**`
  and `scripts/provision/flashinfer/**`.
- License verified at `hopper-flashinfer-source/LICENSE` and
  `blackwell-flashinfer-source/LICENSE` (Apache License 2.0).

## Downloaded at provision time, not redistributed

`configs/capture_tools.json` entries are fetched and built by
`scripts/provision/`; the repository records urls, revisions, hashes and
patches only:

- NVBit (NVIDIA; distributed under the license in its release archive,
  `nvbit/LICENSE` and `nvbit/EULA.txt`): the instrumentation library the
  capture tracer links against.
- Nsight Compute (NVIDIA): manual download, placed under
  `tools/nsight-compute` of the artifact root.
- The CUDA toolkit (NVIDIA): the `cuda-toolkit-13.2` provisioning entry
  copies an installed toolkit into the artifact root.
- zstd (BSD / GPLv2 dual license): built from the upstream 1.4.8 source
  archive by the `bin` provisioning entry.
- Python wheels listed in `configs/capture_locks/*.lock` and the prebuilt
  FlashInfer kernel files listed in
  `configs/capture_locks/flashinfer-cubins.files.json`, each under its own
  license.
