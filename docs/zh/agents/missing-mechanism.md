# 缺机制

[English](../../en/agents/missing-mechanism.md)

硬件做了模型没有规则的事。**先归因**：机制为解释观测到的行为
而加，不为缩小某 case 误差而加。

## 1. 定位

1. `cyclescope run --ledgers --keep …`：`sim/<case>/out-<builder>/`
   下分阶段 journal + wait ledger，显示 per-warp 仿真时间与硬件
   时间在哪分叉。
2. 硬件计数器：`cyclescope reference --metrics <m1,m2>` 附加原始
   ncu csv（每 case `ncu_metrics.csv`）；`--source-stalls` 加跑
   SourceCounters（`ncu_source.csv`，SASS 级停顿采样）。
3. 缺口形成假设后做定向微基准（见
   [missing-leaf.md](missing-leaf.md) §3）。

归因意味着你能说清：哪个阶段分叉、哪个 warp/指令、在什么上等
待、哪个候选资源或同步对象能解释。说不清就继续收窄 —— 不要
装一个无法表述成问题的机制。

## 2. 机制从哪进（按序）

1. **Schema**：`configs/schema_registry.json` 加字段（每字段标
   单位/类型）。
2. **树片段**：`configs/mechanism_tree/common/`（共享）与
   `configs/mechanism_tree/hopper/` + `configs/mechanism_tree/
   blackwell/`（分架构叶/轴/出处；可只一架构）。
3. **生成 schema**：`cyclescope tree build --emit-cpp` →
   `cpp/core/generated/parameter_schema.hpp`、
   `python/cyclescope/tree/generated_schema.py`；然后
   `tree validate --strict`。
4. **引擎规则**：共享行为在 `cpp/core/`；arch 行为在
   `cpp/arch/<arch>/`。归属：cluster-local 叶在单 cluster 循环
   计费；gpu-shared 叶（L2 容量/命中、共享带宽、原子争用、分
   派、全局同步、时钟）在全卡循环 —— 见
   [../design/mechanism-tree.md](../design/mechanism-tree.md)。
5. **转换事实 / USES**：规则需要新采集事实时，扩 tracer 记录 +
   事实提取（`GSIMSEG` 读侧保持向后兼容），并声明机制 USES。
6. **测试**：`cpp/tests/` 加 CTest fixture + `tests/` 加
   pytest；Python/C++ 构建器保持逐字节一致。
7. **回归**：新字段缺省/关闭时出货示例与 `expected.tsv` 结果
   必须逐字节不变；然后跑受影响 case。

各层设计注记：[../design/mechanism-tree.md](../design/mechanism-tree.md)、
[../design/engine.md](../design/engine.md)、
[../design/job-format.md](../design/job-format.md)、
[../design/convert.md](../design/convert.md)。

## 3. 检查单

- [ ] 已归因（ledger/计数器/probe），非按 case 拟合
- [ ] `schema_registry.json` 字段 + `--emit-cpp` schema 已重生成
- [ ] 树片段 + 带证据等级的 `sources.json` 记录
- [ ] 归属正确（cluster-local vs gpu-shared）
- [ ] 需要时已加转换事实/USES，格式向后兼容
- [ ] CTest + pytest 已加
- [ ] 字段关闭时示例/expected.tsv 逐字节不变
- [ ] strict validate 全零
