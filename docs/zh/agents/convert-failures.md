# 转换与段失败

[English](../../en/agents/convert-failures.md)

事实已采到，但段或作业构建失败。终态类在
`sim/<case>/status.json`；抛出的字符串在 `convert.log` /
`segments.log` / `build.log`，`facts/launch.json` 中 per-launch
的 `missing_facts` / `unsupported` 映射按事实名键入并带消费它
的 `<where>` —— `<where>` 是找规则的第一个指针。

## 1. 谁抛出

| 层 | 代码 | 错误类型 |
|---|---|---|
| 事实提取 | `python/cyclescope/convert/` 与 `cyclescope-capture-facts` | `missing_fact:*` |
| 段构建（Python） | `python/cyclescope/convert/`、`python/cyclescope/arch/<arch>/rules.py` | `missing_fact:*`、`unsupported:*` |
| 段构建（C++） | `cpp/arch/<arch>/convert_rules.cpp` | `ConversionError`，同样的 `missing_fact:*` / `unsupported:*` 串 |
| 作业构建 / split / relink | `cyclescope-convert`、`python/cyclescope/convert/capture_pipeline.py`、`cyclescope relink` | `unsupported:trace_split_failed`、`unsupported:no_trace_split_binary`、`unsupported:ambiguous_launch`、`unsupported:relink_*`、`missing_fact:split_trace`、`missing_fact:split_gsync_sequence`、`missing_fact:trace_file`、`missing_fact:launch_row`、`missing_fact:facts_statics`、`missing_fact:mrex_record_bytes` |
| 引擎加载/运行 | `cpp/` | 见 [engine-failures.md](engine-failures.md) |

转换期实际抛出的 `unsupported:` 串（`convert_rules.cpp`、
`capture_pipeline.py`、`convert/`）：arch 语义
`unsupported:{umma_kind, umma_a_source, tmem_shape,
tmem_alloc_op, wgmma_source_operands, warpgroup_wait_keep,
warpgroup_scoreboard, semantic_modifier}`；转换侧
`missing_fact:` 含 `wgmma_source_operands`、
`wgmma_accumulator_register`、`warpgroup_wait_keep`、
`umma_descriptor`、`umma_commit_mask`、`tmem_columns`、
`mbarrier_address`、`clc_mbarrier_address`、`clc_multicast_class`。

## 2. 原因族与规则所在

- **tracer 不记录的 opcode/操作数形式** —— arch 插桩表
  `tools/tracer/arch_*.h` + 操作数族覆盖
  （`CYCLESCOPE_TRACER_OPERAND_FAMILIES`）；此处缺记录是采集缺
  口，修 tracer 并重采。
- **转换器没有语义的 opcode** —— Python 规则
  `python/cyclescope/arch/<arch>/rules.py`，C++ 对应
  `cpp/arch/<arch>/convert_rules.cpp`，公共语义在
  `python/cyclescope/convert/` / `cpp/core/`。修饰符形式未知时
  抛 `unsupported:semantic_modifier`。
- **机制需要的采集轴缺失** —— 机制树 `axes.json`；事实不在作
  业中时机制抛 `missing_fact:axis:<name>`。
- **opcode 在树里缺程序/服务类**（`instruction.service`）——
  仿真时报 `missing_fact`，但覆盖决定在段构建期做出。
- **tensor-map / MMA 形式** —— 描述符/commit/tmem 事实
  （`umma_descriptor`、`umma_commit_mask`、`tmem_columns`、
  `wgmma_source_operands`、`wgmma_accumulator_register`、
  `warpgroup_wait_keep`）缺失 → 转换侧 `missing_fact:*`；形式
  本身超出支持集时抛 `unsupported:*`。
- **`launch_table_mismatch:{diff}`** —— `launches.tsv` 钉的
  `(kernel, grid, block, cluster)` 与被跟踪的 launch 不符。

## 3. Python/C++ 逐字节一致规则

Python 与 C++ 两个构建器的 facts 和 segments 必须完全一致
（`--builder python|cpp`）；`tests/` 的 parity 测试守住。
`cyclescope convert --no-cache` 绕过
`$CYCLESCOPE_ARTIFACT_ROOT/cache/segments` 重算。转换器修复需要
单测 + parity 检查。

## 4. 手工复现单阶段

```
cyclescope convert --capture <capdir> --out <facts>                 # facts
cyclescope convert --facts <facts> --builder python|cpp --out <seg> # 段
cyclescope build <dir> --out <job.bin>                              # 作业
cyclescope-engine validate --job <job.bin>
```

对比工具：`scripts/pipeline/compare_facts.py`、
`compare_segments.py`、`compare_captures.py`、
`compare_results.py`（均取同阶段两个目录/文件，打印首个差
异）。

## 5. 判定

- 记录从未写入 → 采集缺口 → 修 `tools/tracer/` 并重采；保持
  `GSIMSEG` 读侧向后兼容。
- 记录在、事实未提取 → 事实提取映射
  （`python/cyclescope/convert/`）；加 fixture。
- 事实在、段行缺/错 → 段构建规则；两个构建器同步改。
- 有意不可得（套件不记录的 occupancy/轴事实）→ 必须保持
  `missing_fact`，不补默认。
