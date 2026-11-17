# 采集与参照失败

[English](../../en/agents/capture-failures.md)

应用能跑但采集或 ncu 参照失败。状态串来自
`python/cyclescope/capture/driver.py` 与
`python/cyclescope/reference/ncu_runner.py`；检查器是
`python/cyclescope/capture/trace_check.py`（结果写在采集目录的
`trace_check.txt`）。

## 1. 检查顺序

1. `captures/<case>/run.json`：`status`、`rc`、`utc_start`、
   `prepass`、`toolchain`、`tracer_sha256`。
2. `captures/<case>/app.log`：应用自身 stdout/stderr。
3. `captures/<case>/trace_check.txt`：`trace_check` 判定。
4. `captures/<case>/ncu_launches.jsonl`：census（每 launch 一
   行）—— launch 选择与 occupancy 事实都要它。

## 2. 没有跟踪到 launch / 跟踪错 launch

- `no traced launch (no trace-*.bin.zst, no launches.jsonl row
  with traced=true)` —— 应用没到选中的 launch：`--timeout` 超
  时、选择不匹配或 launch 被跳过。查 `launches.jsonl` 看实际跑
  了什么。
- `selection_failed:<e>` —— 正则/occurrence 对不上 census
  （occurrence 超出 census 数会报 `occurrence N of <regex>
  beyond ncu census`）。把预热算进去。
- `launch_table_mismatch:{diff}` —— `launches.tsv` 钉的对
  census 里没有。
- `ambiguous_launch` —— 选择有歧义，收紧 `kernel_regex`。

## 3. trace_check 失败串

`quick_check`/全量检查输出：`zstd_missing`（PATH 上没有 `zstd`
—— 跟踪文件是 `trace-<launch>.bin.zst`）、`zstd frame check
failed`、`bad trace header`（文件不以 `GSIMTRC\0` 开头）、
`missing functions/<id>.tsv`（被跟踪函数的静态表缺失或为空）、
`missing trace-N.bin.zst`、`trace-N: <exception>`（记录流不可解
码 —— 通常是截断写入）、`FAIL load_static: <exc>`。

## 4. 只在 tracer 下挂死或崩溃

1. 先确认不带 tracer 能跑（同 argv/env/cwd，无
   `CUDA_INJECTION64_PATH`）。
2. 用 tracer 诊断开关收窄插桩面（全部按名读于
   `tools/tracer/tracer.cu` / `inject.cu`）：

   | 变量 | 作用 |
   |---|---|
   | `CYCLESCOPE_TRACE_DIR` | 输出目录 |
   | `CYCLESCOPE_CAPTURE_LAUNCHES` | `all` | 逗号序号 | 正则 |
   | `CYCLESCOPE_CAPTURE_CLUSTERS` | `all`（默认）| `sampled:<n>` | cluster id 列表 |
   | `CYCLESCOPE_CAPTURE_MODE` | `trace` | `placement` |
   | `CYCLESCOPE_CAPTURE_LAUNCH_LIMIT` | 只跟踪序号 ≤ n 的 launch |
   | `CYCLESCOPE_CUDA_STACK_SIZE` | 设备栈大小 |
   | `CYCLESCOPE_TRACER_{SKIP,BARE,NO_MREF,NO_RANK,NO_XRANK,NO_UR,NO_R}_OPCODES` | 从插桩中去掉 opcode 类（二分：去掉直到不再挂死再逐步加回） |
   | `CYCLESCOPE_TRACER_OPERAND_{OPCODES,FAMILIES}` | 限制操作数捕获 |
   | `CYCLESCOPE_TRACER_NO_AFTER` | 关闭 after-injection 记录 |
   | `CYCLESCOPE_TRACER_NO_GLOBAL_SYNC` | 关闭全局同步记录 |
   | `CYCLESCOPE_TRACER_IDX_RANGE` | 限制记录的指令索引 |
   | `CYCLESCOPE_TRACER_VERBOSE` | tracer 侧日志 |

   二分：挂死仍在 → 减半 opcode 类；挂死消失 → 触发在被去掉的
   一半。
3. `CYCLESCOPE_L2_FOOTPRINT` 被 driver 强制为 `0`（冷缓存）；保持原样。

## 5. 超时与进程组

`--timeout`（默认 1800 s）约束每次运行；census 上限
`min(timeout_s, 900)`，prepass 上限 `min(timeout_s, 1200)`。超
时杀整个进程组；超时的采集通常以 `no traced launch` 或截断记
录流（`trace-N: …`）收场。

## 6. 参照失败

- `ncu_missing` —— PATH 无 `ncu`；H100 用 `CYCLESCOPE_NCU`
  （环境不自带 —— 按 `configs/capture_environment.json` 装），
  B300 用环境自带 ncu。
- `ncu_timeout`（TIMEOUT_RC）/ `ncu_failed:rc=<n>` —— 被分析运
  行失败；查 `app.log` 与参照目录的 ncu 日志。
- `reference launch <k> not in the ncu census` —— `--captures`
  回放指向了新 census 里没有的 launch。
- `app_failed:triton_prepass` —— triton autotune prepass 失败
  （日志：workdir 里 `triton_prepass.log`）。
- `placement` 运行失败：mode `placement` 写
  `placement.jsonl`；缺/空且 case 依赖 placement → 下游 build
  时报 `missing_fact`。
