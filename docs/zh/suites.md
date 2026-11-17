# 评估套件

[English](../en/suites.md)

项目用每架构 500 个生产 kernel 做评估，来自七个库。每个用例是一次真实 kernel 运行中的一个 launch——下表给出 kernel 的来源与各库的用例数。

| 库 | 是什么 | H100 | B300 |
|---|---|---|---|
| cuBLAS | NVIDIA 的闭源 BLAS 库 | 50 | 50 |
| CUTLASS | NVIDIA 的开源模板与 DSL | 100 | 100 |
| DeepGEMM | DeepSeek 的张量核 kernel 库 | 100 | 100 |
| FlashAttention | 快速、省显存的精确注意力（Hopper 上第 3 版，Blackwell 上第 4 版） | 100 | 100 |
| FlashInfer | LLM 推理 kernel | 50 | 50 |
| FlashMLA | DeepSeek 的潜在注意力 | 50 | 50 |
| Triton | kernel 语言与编译器 | 50 | 50 |

两个套件是 `suites/hopper/full_gpu`（H100，`sm_90a`）与 `suites/blackwell/full_gpu`（B300，`sm_103a`）。

## 1. 跑一个用例

capture 已在盘上时（随发示例就是这样）：

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_einsum_05 \
    --captures examples/captures/hopper --out out/example
```

不给 `--captures`，用例先在本地 GPU 上采集——所需的环境见 [capture-tools.md](capture-tools.md)。

每个套件里的 `representative.txt` 列出 20 个用例——每个 100 例的库取 4 个、每个 50 例的库取 2 个，选取覆盖全部 runner 路径。[release-validation.md](release-validation.md) 里的各项检查用的就是它们。

## 2. 套件的文件

`suites/<arch>/full_gpu/` 下的全部内容：

| 文件 | 是什么 |
|---|---|
| `cases.tsv`（hopper） | 用例表：`set`、`case_id`、`kernel_regex`、`occurrence`、`argv_json`、`cwd`、`env_json`、`env_profile` |
| `cases.txt` + `cases_<family>.json`（blackwell） | family → 用例 id 清单，以及每个 runner 解析的逐用例规格（argv、形状、kernel） |
| `launches.tsv` | 每个用例仿真的 launch：`case`、`kernel`、`occurrence`、`grid`、`block`、`cluster` |
| `reference.tsv` | 硬件参照：`cycles_median/min/max`、`sm_clock_khz`、`cache_control`、`repeats`、`status`——Nsight Compute `sm__cycles_active.max`、`--cache-control all`、3 次取中位 |
| `expected.tsv` | 随发基线：`case`、`class`、`issued`、`sim_cycles`、`hw_cycles`、`placement_digest` |
| `results/` | 500 例在基线 capture 上完整跑一遍的输出，留作参考：`report.tsv`（`cyclescope report` 的输出）、`sim/<case>/status.json` 与 `stages.tsv`（最终状态、各阶段用时、峰值内存）、`result_bin.tar.zst`（每例的 `out-cpp/result.bin`）、`result_bin.sha256`（它们的校验和） |
| `representative.txt` | 20 个代表用例 id |
| `inputs/`（hopper） | 用例 argv 读取的、早于 capture 就存在的输入文件 |

`results/` 就是一次运行的输出目录布局，所以 `cyclescope report suites/<arch>/full_gpu/results` 能把这张表再打一遍。要逐字节核对自己的运行，把它的 `sim/<case>/out-cpp/result.bin` 与 `result_bin.sha256` 比对，或用 `zstd -dc result_bin.tar.zst | tar -xf - -C <dir>` 解开归档。`examples/` 下的十个示例 capture 能逐字节复现对应条目；对同一用例重新采集后再跑一般不能，因为放置与采到的事实不同。各阶段用时与内存是产生这次运行的那台机器上的数字。每例的发射日志与等待账本不在这个目录里，它们发布在 [CycleScope-records](https://github.com/CycleScope-project/CycleScope-records)。

## 3. 用例怎么解析

用例的 argv、cwd 与 env 只用三种路径形式，在运行时展开：

- `$CYCLESCOPE_ROOT/…`——本仓库检出。
- `$CYCLESCOPE_ARTIFACT_ROOT/tools/<name>/…`——登记在 `configs/capture_tools.json` 里的一份 provisioned 工具或第三方源码。
- `<CAPTURE>/…`——capture 输出目录，按次代入。

JIT 缓存与工作区*不是*输入：缓存根参数（`--cache-root`、`FLASHINFER_WORKSPACE_BASE` 等）解析为 `<CAPTURE>/scratch/` 下的空可写目录，运行时 JIT 重新编译。冷 JIT 是首次采集变慢的原因——见 [capture-tools.md](capture-tools.md)。

## 4. Runner

`suites/<arch>/runners/` 放的是把一行用例变成可执行命令的逐库代码。第三方源码被 provision 到 `$CYCLESCOPE_ARTIFACT_ROOT/tools/<name>`，在解析时链入——仓库不 vendoring 任何第三方代码。

- **hopper**：`suites/hopper/runners/harness/` 是 runner harness。它保持自己的内部布局，因为它的文件靠相对路径互相定位。`environment_far.sh` 为使用它的用例选择 "far" flashinfer profile。
- **blackwell**：`suites/blackwell/runners/` 每 family 一个适配器加共享的 `common/` 助手；适配器把 `cases_<family>.json` 的规格解析成 argv、env 与 kernel 正则。

## 5. 用你自己的 kernel

采集从套件表解析用例，所以新用例就是表里的新行。Hopper 上的最小步骤：

1. 在 `suites/hopper/full_gpu/cases.tsv` 加一行：`case_id`、选中目标 launch 的 `kernel_regex` + `occurrence`，以及解析好的 `argv_json` / `cwd` / `env_json`（同样三种路径形式）。
2. 可选地在 `launches.tsv` 加一行，钉死要采的那个 launch。

Blackwell 上一个用例是 `cases.txt` 的一行加该 family 的 `cases_<family>.json` 里的一项，由 family 的适配器解析——所以新 case family 意味着新适配器，而不只是加表行。

带实例的逐步做法见 [agents/new-kernel.md](agents/new-kernel.md)；套件没有覆盖的库见 [agents/new-library.md](agents/new-library.md)。
