# FlashMLA upstream pin

- Repository: https://github.com/deepseek-ai/FlashMLA (official DeepSeek repo)
- Pinned revision: `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`
- Revision date: 2026-09-15 09:50:44 +0800
- Revision subject: `fix: use longlong2 for the 128-bit st.async PTX transport (MSVC LLP64) (#224)`
- Package version at pin: `flash_mla.__version__ == "1.0.0"` (wheel version `1.0.0+ba89a34`)
- Clone URL: `git clone https://github.com/deepseek-ai/FlashMLA` (public GitHub, direct clone worked
  from the dev machine on 2026-09-29; no proxy or pip mirror was needed)

## Vendored tree (T0005 trim + C-batch SM90 sparse support; CUTLASS is not vendored)

`upstream/` is a plain working-tree export at the pinned commit (`.git/` removed), trimmed to the
SM90 dense-decode, sparse-decode, sparse-prefill paths and their harness:

- Kept: `setup.py` with the dense and selected sparse SM90 TUs and sm_90a-only gencode;
  `flash_mla/` with `get_mla_metadata`, dense/sparse `flash_mla_with_kvcache`, and
  `flash_mla_sparse_fwd`; `csrc/api/{api.cpp,common.h,dense_decode.cpp,sparse_decode.cpp,
  sparse_prefill.cpp}`; `csrc/kernels/{defines.h,params.h,utils.h,kv_cache_format.h}`;
  `csrc/kernels/sm90/decode/dense/**`; `csrc/kernels/sm90/decode/sparse/**`
  (including `components/{config.h,dequant.h,helpers.h}`, `splitkv_mla.{h,cuh}`, and the
  four `v32/v4_persistent_h64/h128` instantiations); `csrc/kernels/sm90/prefill/sparse/**`
  (including `phase1.{h,cuh}` and the four `phase1_k512/k576[_topklen]` instantiations);
  `csrc/kernels/sm90/helpers.h`; `csrc/kernels/smxx/decode/**`;
  `csrc/kerutils/include/**` (kept whole: `device/device.cuh` pulls per-arch headers),
  `LICENSE`, `.gitignore`.
- The sparse decode/prefill support above is restored from this same pinned commit because the
  C-batch harness now measures these SM90 operations. The sparse API TUs are trimmed to SM90;
  no `csrc/kernels/sm100/**` file or SM100 kernel instantiation is added or built. The existing
  `csrc/kerutils/include/kerutils/device/sm100/**` utility headers are retained unchanged.
- Not committed (dropped at export): `csrc/cutlass` (see next section), all
  `csrc/kernels/sm100/**` and SM100 sparse/prefill implementations, upstream dense prefill/bwd
  kernels and API TUs (`csrc/api/{dense_fwd,dense_bwd,
  fused_norm_rope_attn_rope_cast_fwd}.cpp`), `tests/`, `docs/`, `benchmark/`, upstream
  `README.md`, `.gitmodules`, `flash_mla/fused_norm_rope_attn_rope_cast.py`.
- Deviations from upstream files: `setup.py` only builds the selected SM90 dense/sparse sources
  with sm_90a gencode; wheel version remains fixed to `1.0.0+ba89a34` (upstream derives it from
  `git rev-parse`, meaningless inside this delivery tree), it fails fast when `csrc/cutlass` is
  absent, and does not call `git submodule`; `api.cpp` registers dense decode and the two sparse
  APIs; `sparse_decode.cpp` and `sparse_prefill.cpp` retain only the SM90 implementation;
  the Python interface retains dense decode and restores the sparse decode/prefill branches.
  The existing measurement-only marker patch remains in place.

## CUTLASS dependency (fetched at build time, NOT vendored)

- CUTLASS pinned revision: `147f5673d0c1c3dcf66f78d677fd647e4a020219`
- Revision date: 2026-01-12 17:05:31 -0800 (`New RMS Norm example with unit tests (#2917)`)
- Upstream: https://github.com/NVIDIA/cutlass — FlashMLA pins it as the `csrc/cutlass` submodule at
  ba89a34. Version at pin: CUTLASS 4.3.5.
- Acquisition (performed by `scripts/build_cluster_sm.sh` before `pip install`):
  1. Target `$HOPPER_ARTIFACT_ROOT/tool_cache/cutlass/147f5673d0c1c3dcf66f78d677fd647e4a020219`
     (env default `/path/to/hopper-prototype/artifacts`). Reused as-is when
     `include/cutlass/version.h` already exists.
  2. Otherwise `git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git` and
     `git checkout --detach 147f5673…`, then `.git` is dropped and the tree is `mv`-ed into the
     tool-cache path (machines without GitHub access can populate that path from any clone; the
     directory is content-addressed by the commit).
  3. Symlink `upstream/csrc/cutlass -> <tool-cache path>` so `setup.py`'s include_dirs resolve
     unchanged (`csrc/cutlass` is gitignored).
- Why the repo's `suites/cutlass/upstream` (CUTLASS 4.2.1 @ `f3fde583`) is NOT reused
  (static verification method and result, 2026-09-29, no GPU on that machine):
  - Method: blob-hash comparison of the full `include/` trees — 805 files at 147f5673 vs 802 files
    at f3fde583 — followed by a per-file diff that ignores whitespace and the copyright-year line;
  - Result: 713/805 files differ only in the copyright-year line, **89 files differ substantively**
    — including 32 `cute/arch/**` and 28 `cute/atom/**` files, the SM90 kernel codegen core — and 3
    files exist only at 147f5673; the dense-decode include closure itself hits substantive diffs
    (`cutlass/bfloat16.h`: `bfloat16_t::epsilon()` is `bitcast(0x3c00)` at 4.3.5 vs
    `bitcast(0x1000)` at 4.2.1; `cutlass/half.h` differs in the same way);
  - Conclusion: 4.3.5 → 4.2.1 semantic drift inside directly-included headers means identical-SASS
    invariance cannot be claimed statically, so the build fetches the pinned commit (this section)
    instead of reusing the repo copy;
  - **G0 pending**: the actual H100 compile (`pip install` of this tree with the fetched CUTLASS,
    `dense_decode_fwd` present, SASS check) is still to be verified on the GPU pod; record the
    result in the T0005 G0 ledger when it runs.

## Delivery measurement-only patch (T0005 修订 3-②/修订 5)

The delivery tree carries a measurement-only clock-marker patch on top of the pinned upstream
export. Everything is gated behind `FLASHMLA_DELIVERY_MARKER_TIMING` (delivery builds set it;
without the flag the compiled upstream semantics are unchanged — verified by the no-macro
SASS regression check in `$HOPPER_ARTIFACT_ROOT/tasks/T0005/flashmla_marker_check/`).
Kernel signatures, `params.h` and the public ABI are untouched; the row layout matches the
FA3/CUTLASS `device_kernel` marker contract (7 fields per (cta, warp); plan:
`$HOPPER_ARTIFACT_ROOT/tasks/T0005/marker_timing_plan_flashmla.md`).

- Patched files (all edits inside `#ifdef FLASHMLA_DELIVERY_MARKER_TIMING` unless noted):
  - `csrc/kernels/sm90/decode/dense/splitkv_mla.cuh` — outermost clock-marker pair around
    `flash_fwd_splitkv_mla_kernel` (entry read before the TMA prologue, exit on the single
    return path, and pair close-out on the empty-schedule early return);
  - `csrc/kernels/smxx/decode/combine/combine.cu` — pair around
    `flash_fwd_mla_combine_kernel` (both early returns close the pair);
  - `csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu` — pair
    around `get_mla_metadata_kernel`;
  - `csrc/api/api.cpp` — `flash_mla_delivery_marker_set` / `flash_mla_delivery_marker_tu_count`
    (ctypes setter bridge; marker builds only);
  - `setup.py` — `FLASHMLA_DELIVERY_MARKER_TIMING` env flag → `-D` plus the
    `../harness/override` include dir (flag-off builds keep upstream compile inputs
    bit-identical); CPATH fallback injecting the pip `nvidia-*-cu12` include dir when the
    image CUDA lacks vendor-lib headers (`cusparse.h`, required by ATen's
    `CUDAContextLight.h` in the host TU; discovered on the build image 2026-09-29).
- New delivery-only files (not upstream files):
  `../harness/override/flashmla_marker.h` (device row buffer + cross-TU registry),
  `../harness/marker_timing.py` (host-side reduction, FA3-style),
  `../scripts/marker_sass_audit.py` (static `SR_CLOCKLO`/`SR_GLOBALTIMERLO` >= 2 gate,
  called by `../scripts/build_cluster_sm.sh`).
- Vendored-source sha256 over 56 files under `upstream/` (`PINNED.md` and the `csrc/cutlass`
  symlink target excluded): `e476945dabf0a99ce55af257cdf8738ed10114e1252c21f0467a227a291ce47a`
  (recompute: `cd suites/flashmla && find upstream -type f ! -path 'upstream/PINNED.md' ! -path 'upstream/csrc/cutlass/*' | sort | xargs sha256sum | sha256sum`).
- No-marker regression evidence (T0005 marker check, CPU pod AOT): formal-denominator kernel
  (`flash_fwd_splitkv_mla_kernel`, both dtype instantiations) SASS **byte-identical** to the
  pristine upstream; `combine.cu` / `get_decoding_sched_meta.cu` SASS differ **only** in
  `FLASH_DEVICE_ASSERT`-embedded `__LINE__` immediates (the patch inserts source lines above
  the asserts; no instruction-semantic change, zero marker reads without the flag).
- The upstream pin itself is unchanged; `upstream_sha` semantics are unaffected — patch
  state is recorded here, per file, under this section.

## Reproduce

```bash
git clone https://github.com/deepseek-ai/FlashMLA && cd FlashMLA
git checkout ba89a3466e9470ad08ab39738d4e7bb66989e1e7
git submodule update --init --recursive   # csrc/cutlass -> 147f5673d0c1c3dcf66f78d677fd647e4a020219
pip install -v .
```

The delivery tree reproduces the same build without the submodule: run
`suites/flashmla/scripts/build_cluster_sm.sh h100`.
