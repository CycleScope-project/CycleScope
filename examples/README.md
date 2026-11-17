# Example captures

[中文](README.zh-CN.md)

Ten captures from the shipped baseline, five per architecture, so the
CPU side of the pipeline (convert → build → simulate → report) can be
tried without a GPU. They were picked for a small error against the
hardware reference and a small trace; they are not a sample of the
suite's accuracy (see `suites/<arch>/full_gpu/expected.tsv` for all
500 cases per architecture).

| suite | case | instructions | simulated cycles | hardware cycles | error |
|---|---|---|---|---|---|
| `hopper/full_gpu` | `deepgemm_einsum_05` | 357980 | 15787 | 15625 | 1.04 % |
| `hopper/full_gpu` | `flashinfer_merge_state_03` | 800768 | 16646 | 16819 | 1.03 % |
| `hopper/full_gpu` | `deepgemm_fp8_mgroup_contiguous_07` | 1321471 | 11426 | 11389 | 0.32 % |
| `hopper/full_gpu` | `triton_p9_matmul_tma_02` | 2256896 | 30775 | 30808 | 0.11 % |
| `hopper/full_gpu` | `cutlass_stream_k_splitk_04` | 4265140 | 62336 | 62814 | 0.76 % |
| `blackwell/full_gpu` | `CUF-DSL-DENSE-F16-2CTA-C02` | 455267 | 19938 | 19872 | 0.33 % |
| `blackwell/full_gpu` | `DGF-MGC-FP8-NN-C02` | 1300285 | 21714 | 21733 | 0.09 % |
| `blackwell/full_gpu` | `MLF-MLDEC-H64-NOSPLIT-R05` | 1904665 | 36469 | 36862 | 1.07 % |
| `blackwell/full_gpu` | `FAF-FWD-SINK-1C-C01` | 2303126 | 20833 | 21102 | 1.27 % |
| `blackwell/full_gpu` | `FIF-CLASSIC-APPLY-ROPE-P01` | 3697792 | 54073 | 54318 | 0.45 % |

Run them with one command from the repository root (needs `cmake`, a
C++20 compiler, OpenSSL development files, Python >= 3.10 and `zstd`;
the engine is built on first use, nothing has to be installed):

```bash
bash examples/run_examples.sh                 # all ten
bash examples/run_examples.sh blackwell       # one architecture
bash examples/run_examples.sh hopper deepgemm_einsum_05   # one case
```

The script prints one line per case and then the report; every case
should show the simulated cycles of the table and verdict `ok` with a
0.00 % delta against the baseline. Outputs go to `examples/out/`
(`OUT=<dir>` to change). The same by hand, after building the engine and
`pip install -e .` (see
[docs/en/getting-started.md](../docs/en/getting-started.md)):

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_einsum_05 \
    --captures examples/captures/hopper --out /tmp/cyclescope-example
cyclescope report /tmp/cyclescope-example
```

Layout: `captures/<arch>/<case>/` holds what the converter reads from
a capture (`launches.jsonl`, `functions.jsonl`, `functions/`,
`tensor_maps.jsonl`, `device.json`, `trace-<launch>.bin.zst`,
`trace_check.txt`, `placement/placement.jsonl`);
`captures/<arch>/reference/<case>/` holds the hardware reference
(`reference.tsv` and the three Nsight Compute CSVs). Run logs, the
recorded process environment and JIT scratch directories of the
original captures are not included.
