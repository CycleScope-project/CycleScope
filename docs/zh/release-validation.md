# 精度与验证

[English](../en/release-validation.md)

随发精度数字的含义、测量方式，以及在参考环境（[environment.md](environment.md)）上还检查过什么。

## 1. 精度怎么量

参照就是该 launch 自己的硬件计时：Nsight Compute 对该 launch 测出的 `sm__cycles_active.max` 周期数，以 `--cache-control all`（每次 launch 前冲刷全部缓存）采集，3 次取中位，在不带 tracer 的独立运行里完成。

`cyclescope report` 用三个数把仿真周期与之比较：

- **MAPE**——平均绝对百分比误差，`mean |sim/hw - 1|`。
- **P90 绝对误差**——同一批误差的第 90 百分位。
- **中位比值**——`sim/hw` 的中位数：1.000 表示无偏。

`cyclescope report --baseline suites/<arch>/full_gpu/expected.tsv` 打印一张随发表的这三项。

## 2. 随发基线的精度

`suites/<arch>/full_gpu/expected.tsv` 每架构 500 例，全部 `ok`：

| 设备 | 用例数 | MAPE | P90 绝对误差 | 中位 sim/hw |
|---|---|---|---|---|
| H100 | 500 | 7.93 % | 17.2 % | 0.947 |
| B300 | 500 | 7.64 % | 14.8 % | 0.952 |

## 3. 还检查了什么

**确定性。** 用发布版代码重新仿真每架构全部 500 例的基线 capture，`expected.tsv` 完全复现——指令数、仿真周期、放置摘要——基线 `result.bin` 逐字节复现：1000/1000，含走分片转换路径的用例。这次运行的输出留在 `suites/<arch>/full_gpu/results/`（[suites.md](suites.md) 第 2 节）。

**从零 provisioning。** `configs/capture_tools.json` 全部 31 项曾仅靠仓库内容 provision 进空 artifact root 并通过 verify（[capture-tools.md](capture-tools.md)）。

**重新采集的保真。** 20 + 20 代表用例在参考环境上用 provisioned 工具重新采集。静态采集事实——launch 清单、被采 launch、functions、SASS——与基线 capture 一致，例外两条：

- B300 用例 `TRF-TUT-ATTN-FWD-FP16-CAUSAL-C05`：一个 JIT 编译 kernel 的 9 行 SASS 只在 uniform 寄存器编号上不同；两次独立新采集彼此一致。原因未查明。
- CuTe-DSL tensor map 尾字节携带未解码的宿主数据，各次采集间不同。

重新采集的仿真周期对基线：B300 20/20 复现了基线 CTA→SM 放置且都在 3 % 内（最大 2.46 %）。H100 4/20 复现了该放置；其余 16 个放置不同，最大差到 4.26 %（`fa3_fwd_hdim_small_08`——对该用例再采两次复现了基线放置，分别为 −1.84 % 与 −1.25 %）。

**参照漂移。** 硬件参照周期本身在不同采集批次间最多移动了 3.6 %。

**用例解析。** 全部 1000 个套件用例解析出的 argv、环境与文件，与基线采集时一致（逐用例比对）。

## 4. 没检查什么

- 公共容器镜像。
- 参考环境以外的驱动或工具包版本。
- 两张参考卡以外的 GPU（NVIDIA H100、NVIDIA B300 SXM6 AC）。
- 每架构其余 480 个非代表用例的重新采集；它们的用例解析已比对（第 3 节）但未重新采集。
