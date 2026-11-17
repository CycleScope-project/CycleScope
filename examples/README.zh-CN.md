# 示例采集

[English](README.md)

随基线发布的十个采集，每个架构五个，用来在没有 GPU 的机器上试跑管线的 CPU 部分（转换 → 构建 → 仿真 → 报告）。挑选标准是对硬件参照的误差小、trace 体积小；它们不代表套件的整体精度（每架构 500 例的结果见 `suites/<arch>/full_gpu/expected.tsv`）。

| 套件 | 用例 | 指令数 | 仿真周期 | 硬件周期 | 误差 |
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

在仓库根目录一条命令运行（需要 `cmake`、支持 C++20 的编译器、OpenSSL 开发文件、Python >= 3.10 与 `zstd`；首次运行会自动构建引擎，不需要预先安装任何东西）：

```bash
bash examples/run_examples.sh                 # 全部十个
bash examples/run_examples.sh blackwell       # 单个架构
bash examples/run_examples.sh hopper deepgemm_einsum_05   # 单个用例
```

脚本每例打印一行，最后打印报告；每例应给出表中的仿真周期，判定为 `ok`，相对基线的差为 0.00 %。输出在 `examples/out/`（用 `OUT=<dir>` 更改）。手动方式（先构建引擎并 `pip install -e .`，见 [docs/zh/getting-started.md](../docs/zh/getting-started.md)）：

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_einsum_05 \
    --captures examples/captures/hopper --out /tmp/cyclescope-example
cyclescope report /tmp/cyclescope-example
```

目录结构：`captures/<arch>/<case>/` 是转换器从一次采集中读取的文件（`launches.jsonl`、`functions.jsonl`、`functions/`、`tensor_maps.jsonl`、`device.json`、`trace-<launch>.bin.zst`、`trace_check.txt`、`placement/placement.jsonl`）；`captures/<arch>/reference/<case>/` 是硬件参照（`reference.tsv` 与三份 Nsight Compute CSV）。原始采集里的运行日志、进程环境记录和 JIT 临时目录不包含在内。
